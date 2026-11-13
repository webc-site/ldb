//! 空闲空间管理回归（Tier 分配 / 尾部拆分 / 跨级回退 / 公布 / 并发 / WAL 注入）。
//!
//! 语义对照上游 `ZLeanStore/src/storage/free_page_manager.cc` 与
//! `leanstore.latency/src/storage/free_storage.cc`（行号见各用例注释）。

use std::{
  sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
  },
  thread,
};

use futures_lite::future::block_on;
use ldb_free::{ExtentTier, FreeStorageManager, NO_TIERS, tier_index, tier_size};
use ldb_page::hook::TxnLogHook;
use ldb_type::PageId;

/// 测试用 WAL 钩子（计数 + 记录最后一次 pid）
#[derive(Default)]
struct CountingHook {
  freed: AtomicU64,
  last_pid: AtomicU64,
}

impl TxnLogHook for CountingHook {
  async fn log_pid2_offset(&self, _pid: PageId) {}
  async fn log_valid_page(&self, _pid: PageId) {}
  async fn log_free_page(&self, pid: PageId) {
    self.freed.fetch_add(1, Ordering::Relaxed);
    self.last_pid.store(pid, Ordering::Relaxed);
  }
}

/// 分级/尺寸定界：**穷尽** `page_cnt = 0..=4096` 的上界 + 最小性（唯一击杀 `tier_index` 的
/// off-by-one 变异）。
#[test]
fn tier_index_tier_size_boundaries() {
  // n=0 与 n=1 都归 tier0。上游 `TierIndex(n, false)` = `find_if_not(TIER_SIZE, other < n)`，
  // 首个 `TIER_SIZE[i] >= n` 即 i；`TIER_SIZE[0] = 1` ⇒ n∈{0,1} 均为 0（无 panic；
  // `extent_list.h:190-199`）。
  assert_eq!(tier_index(0), 0, "n=0 归 tier0（非 panic）");
  assert_eq!(tier_index(1), 0, "TIER_SIZE[0]=1 >= 1");
  // 穷尽枚举：上界 `tier_size(tier_index(n)) >= n`（上游 free_page_manager.cc:37 的 Ensure）
  // 与最小性 `tier_size(tier_index(n)-1) < n`（n>=2；`extent_list.h:190` 的 find_if_not 取最小）。
  for n in 1u64..=4096 {
    let idx = tier_index(n);
    assert!(
      tier_size(idx) >= n,
      "上界失败: n={n} idx={idx} size={}",
      tier_size(idx)
    );
    if n >= 2 {
      assert!(
        tier_size(idx - 1) < n,
        "最小性失败: n={n} idx={idx} prev_size={}",
        tier_size(idx - 1)
      );
    }
  }
  // 2 的幂处逐个钉死：tier_index(2^t) == t（1/2/4/8/16/.../4096）。
  for t in 0u8..=12 {
    assert_eq!(tier_index(1u64 << t), t, "tier_index(2^{t})");
    assert_eq!(tier_size(t), 1u64 << t, "tier_size({t})");
  }
  // 非 2 的幂处：向上取整到下一个 2 的幂。
  assert_eq!(
    [
      tier_index(3),
      tier_index(5),
      tier_index(6),
      tier_index(7),
      tier_index(9)
    ],
    [2, 3, 3, 3, 4]
  );
  assert_eq!(tier_index(4096), 12);
  assert_eq!(tier_size(63), 1u64 << 63);
  assert_eq!(NO_TIERS, 64);
}

#[compio::test]
async fn tier_roundtrip() {
  let fsm = FreeStorageManager::new(true, 4);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(100, 6, &hook).await;
  assert_eq!(fsm.number_of_free_entries(), 0, "提交前不可见");
  fsm.public_free_extents().await;
  assert_eq!(fsm.number_of_free_entries(), 1);
  let mut split = Vec::new();
  let got = fsm
    .request_free_extent(64, |_| true, &hook, &mut split)
    .await;
  assert_eq!(got, Some(100));
  assert!(split.is_empty(), "恰好等于该级尺寸 ⇒ 不拆分");
  assert_eq!(fsm.number_of_free_entries(), 0);
  // 分配路径与释放路径各记一次 WAL（上游 LogFreePage(false)/LogFreePage(true)）
  assert_eq!(hook.freed.load(Ordering::Relaxed), 2);
}

#[compio::test]
async fn request_validates() {
  let fsm = FreeStorageManager::new(true, 2);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(7, 3, &hook).await;
  fsm.public_free_extents().await;
  let mut split = Vec::new();
  // 校验失败 → 不消耗，回队列（上游 free_page_manager.cc:58 push 回原级）
  let none = fsm
    .request_free_extent(8, |pid| pid != 7, &hook, &mut split)
    .await;
  assert!(none.is_none());
  assert_eq!(fsm.number_of_free_entries(), 1, "失败请求须回队");
  let ok = fsm
    .request_free_extent(8, |pid| pid == 7, &hook, &mut split)
    .await;
  assert_eq!(ok, Some(7));
  assert!(split.is_empty());
}

/// `validate_extent` 恒 false：必须**跳过该 tier、继续扫描、正常终止**（不得死循环、
/// 不得吞掉条目）；且失败后回队顺序为 FIFO 尾插（上游 `tiers_[tier_index].push(...)`，
/// `free_page_manager.cc:58` 追加而非前插）。
///
/// 唯一击杀：M12（`push_back` → `push_front`）—— 回队顺序变化后下一次成功请求的 pid 改变。
#[compio::test]
async fn validate_false_skips_tier_and_terminates() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  for pid in 1..=5u64 {
    fsm.prepare_free_tier(pid, 3, &hook).await; // tier3 放 5 个 extent
  }
  fsm.public_free_extents().await;

  let mut split = Vec::new();
  // 恒 false：同级 3 次全失败（各回队），回退自 62 向下到 tier4 全空、tier3 映射顶级也为空 ⇒ None。
  let got = fsm
    .request_free_extent(8, |_| false, &hook, &mut split)
    .await;
  assert_eq!(got, None, "全校验失败返回 None（上游 return false，:83）");
  assert!(split.is_empty());
  assert_eq!(
    fsm.number_of_free_entries(),
    5,
    "坏 extent 全部回队，无丢失"
  );

  // 队列顺序：[1,2,3,4,5] 前 3 次 pop 后尾插 ⇒ [4,5,1,2,3]；下一次成功必取 4（FIFO）。
  // push_front 变异会取回 1，从而失败。
  let ok = fsm
    .request_free_extent(8, |_| true, &hook, &mut split)
    .await;
  assert_eq!(ok, Some(4), "尾插回队 ⇒ FIFO 顺序为 4");
  assert!(split.is_empty(), "tier3 的 extent 恰为 8 页，不拆分");
  assert_eq!(fsm.number_of_free_entries(), 4);

  // focus 4（drain vs take 等价）：**逐一取回**剩余全部，顺序与内容必须与归还顺序一致
  // ⇒ `public_free_extents` 的 `drain` 不丢条目、不改序（`mem::take` 变体同序、仅缓冲容量差异）。
  let mut rest = Vec::new();
  for _ in 0..4 {
    let mut s = Vec::new();
    rest.push(fsm.request_free_extent(8, |_| true, &hook, &mut s).await);
    assert!(s.is_empty());
  }
  assert_eq!(
    rest,
    [Some(5), Some(1), Some(2), Some(3)],
    "逐一取回顺序 == 尾插后 FIFO 顺序"
  );
  assert_eq!(fsm.number_of_free_entries(), 0);
}

/// 同级命中且请求 < extent 尺寸 ⇒ 尾部拆成整级 extent（上游 free_page_manager.cc:49-54
/// → PrepareFreePages，:113-120）。
#[compio::test]
async fn split_tail_less_than_extent() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(1000, 3, &hook).await; // tier 3 = 8 页
  fsm.public_free_extents().await;

  let mut split = Vec::new();
  let got = fsm
    .request_free_extent(5, |_| true, &hook, &mut split)
    .await;
  assert_eq!(got, Some(1000), "请求 5 页由 8 页 extent 满足");
  // 尾部 [1005, 1008) 3 页 = 2 + 1 ⇒ tier1@1005 + tier0@1007（贪心二进制分解）
  assert_eq!(
    split,
    [
      ExtentTier {
        start_pid: 1005,
        tier: 1,
      },
      ExtentTier {
        start_pid: 1007,
        tier: 0,
      },
    ]
  );
  // 拆分片在 `to_free_extents` 待公布：公布前不计入 free_tiers
  assert_eq!(fsm.number_of_free_entries(), 0);
  fsm.public_free_extents().await;
  assert_eq!(
    fsm.number_of_free_entries(),
    2,
    "分隔片公布后入 tier1/tier0"
  );
  // 再次公布为空（已 take 清空）
  fsm.public_free_extents().await;
  assert_eq!(fsm.number_of_free_entries(), 2);
}

/// 同级空 → 跨级回退：扫描更高 tier（上游 free_page_manager.cc:63-81）。更高 tier 优先
/// （自高向低扫描）：tier6 先于 tier4 命中。
#[compio::test]
async fn cross_tier_fallback_prefers_higher() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(400, 4, &hook).await; // 16 页
  fsm.prepare_free_tier(600, 6, &hook).await; // 64 页
  fsm.public_free_extents().await;

  let mut split = Vec::new();
  let got = fsm
    .request_free_extent(8, |_| true, &hook, &mut split)
    .await;
  assert_eq!(got, Some(600), "自高向低：tier6 优先于 tier4");
  // 尾部 [608, 664) 56 页 = 32+16+8 ⇒ tier5@608, tier4@640, tier3@656
  assert_eq!(
    split,
    [
      ExtentTier {
        start_pid: 608,
        tier: 5
      },
      ExtentTier {
        start_pid: 640,
        tier: 4
      },
      ExtentTier {
        start_pid: 656,
        tier: 3
      },
    ]
  );
  fsm.public_free_extents().await;
  // 剩余：tier4(400) 1 + tier0(600 拆分后的 0？) —— tier6 被取走后余 3 片
  assert_eq!(fsm.number_of_free_entries(), 4);
}

/// `i == tier_index` 时取顶级 tier（上游 free_page_manager.cc:68）。仅顶级非空 ⇒ 命中顶级。
#[compio::test]
async fn cross_tier_fallback_reaches_top_tier() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(5, 63, &hook).await; // tier 63 = 2^63 页（顶级）
  fsm.public_free_extents().await;

  let mut split = Vec::new();
  let got = fsm
    .request_free_extent(1, |_| true, &hook, &mut split)
    .await;
  assert_eq!(got, Some(5), "顶级在 i==tier_index 时被检查");
  // 2^63-1 贪心分解 = 63 片（tier62..tier0）
  assert_eq!(split.len(), 63);
  assert_eq!(split[0].tier, 62);
  assert_eq!(split[62].tier, 0);
}

/// 所有 tier 空 ⇒ None（上游 free_page_manager.cc:83）。
#[compio::test]
async fn all_tiers_empty_returns_none() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  let mut split = Vec::new();
  assert_eq!(
    fsm
      .request_free_extent(1, |_| true, &hook, &mut split)
      .await,
    None
  );
  assert_eq!(
    fsm
      .request_free_extent(4096, |_| true, &hook, &mut split)
      .await,
    None
  );
}

/// 同级校验失败后跨级找到可用的更大 extent（上游 :42-81 两段串联）。
#[compio::test]
async fn fallback_after_validation_failure() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(50, 3, &hook).await; // 8 页，校验会失败
  fsm.prepare_free_tier(70, 4, &hook).await; // 16 页，校验通过
  fsm.public_free_extents().await;

  let mut split = Vec::new();
  let got = fsm
    .request_free_extent(8, |pid| pid == 70, &hook, &mut split)
    .await;
  assert_eq!(got, Some(70), "跳过 tier3 的坏 extent，回退到 tier4");
  assert_eq!(
    split,
    [ExtentTier {
      start_pid: 78,
      tier: 3
    }]
  );
  assert_eq!(fsm.number_of_free_entries(), 1, "坏 extent 留在 tier3");
}

/// `public_free_ranges` 先快照后公布：首调返回非空，次调返回空（回归 R1 的 take-then-
/// iterate-empty 缺陷；上游 GroupCommitExecutor PhaseFive，group_commit.cc:441/467）。
#[compio::test]
async fn public_free_ranges_snapshots_before_publish() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(100, 2, &hook).await;
  fsm.prepare_free_tier(200, 0, &hook).await;
  let first = fsm.public_free_ranges().await;
  assert_eq!(first, [(100, 4), (200, 1)]);
  assert_eq!(fsm.number_of_free_entries(), 2, "公布后入队");
  let second = fsm.public_free_ranges().await;
  assert!(second.is_empty(), "挂起表已清空");
}

/// RangeLock 形态（`enable_extent_tier == false`，上游 free_storage.cc:8-10/34-37）。
#[compio::test]
async fn range_lock_form_locks_and_publishes() {
  let fsm = FreeStorageManager::new(false, 4);
  assert!(fsm.try_lock_range(0, 10, 5).await); // [10,14]
  assert!(!fsm.try_lock_range(0, 12, 3).await, "与已锁区间相交");
  let hook = CountingHook::default();
  // 公布该区间 ⇒ unlock_range 解锁后复用
  fsm.prepare_free_tier(10, 0, &hook).await;
  fsm.public_free_extents().await;
  assert!(fsm.try_lock_range(0, 10, 5).await, "解锁后可再次锁住");
}

/// 单页释放路径（`todo.md` §3.4 第 15 条「单页空闲队列」）。
#[compio::test]
async fn log_free_page_records_single_page() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  fsm.log_free_page(9, &hook).await;
  assert_eq!(hook.freed.load(Ordering::Relaxed), 1);
  assert_eq!(hook.last_pid.load(Ordering::Relaxed), 9);
}

/// `n=0` 语义（focus 2）：上游 ZLeanStore `RequestFreeExtent(0, …)`：
/// `TierIndex(0, false)` 取首个 `TIER_SIZE[i] >= 0` ⇒ i=0（`extent_list.h:190-199`），
/// `Ensure(ExtentSize(0) >= 0)` 平凡成立（`free_page_manager.cc:36-37`）⇒ 归 **tier0**（非断言/非空）；
/// 命中后 `TIER_SIZE[0](=1) > 0` ⇒ `PrepareFreePages(pid+0, 1)` 把整块拆回 tier0（`:49-54`）。
/// 本仓 `tier_index(0)=0` 与之一致。
///
/// 唯一击杀：M16（`split_tail` 加 `page_cnt > 0` 守卫 ⇒ n=0 不再拆回，`out_split_exts` 变空）。
#[compio::test]
async fn request_zero_pages_returns_tier0_and_splits_whole() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = CountingHook::default();
  fsm.prepare_free_tier(77, 0, &hook).await;
  fsm.public_free_extents().await;
  let mut split = Vec::new();
  let got = fsm
    .request_free_extent(0, |_| true, &hook, &mut split)
    .await;
  assert_eq!(got, Some(77), "n=0 归 tier0（非断言/非 None）并命中");
  assert_eq!(
    split,
    [ExtentTier {
      start_pid: 77,
      tier: 0
    }],
    "n=0 ⇒ 整块拆回 tier0（上游 TIER_SIZE[0]>0 分支）"
  );
}

/// drain vs take 语义等价（M15 复核，focus 4）：见 `validate_false_skips_tier_and_terminates`
/// 末尾的「归还多个 extent 后逐一取回、顺序与内容一致」断言 —— `public_free_extents` 的
/// `drain` 与 `mem::take` 发布同一集合、同一顺序，差异仅为缓冲容量（1.00 vs 0.00 alloc/op）。
///
/// 并发：4 线程交错 alloc/free 1e5 次，占用位图 + 计数守恒。
/// 唯一击杀：同一 extent 双分配 / 泄漏。
fn concurrent_alloc_free(threads: usize) {
  const K: u64 = 512; // tier0（1 页）extent 池：pid 1..=K
  let fsm = Arc::new(FreeStorageManager::new(true, threads));
  let hook = Arc::new(CountingHook::default());
  let in_use: Arc<Vec<AtomicBool>> = Arc::new((0..=K).map(|_| AtomicBool::new(false)).collect());

  let init = block_on(async {
    for pid in 1..=K {
      fsm.prepare_free_tier(pid, 0, &*hook).await;
    }
    fsm.public_free_extents().await;
    fsm.number_of_free_entries()
  });
  assert_eq!(init, K);

  // Miri 下 1e5 规模不可行（解释器 ~100x 慢），缩到 64 次仍覆盖双分配/泄漏判定。
  #[cfg(miri)]
  const ITERS: usize = 64;
  #[cfg(not(miri))]
  const ITERS: usize = 100_000 / 8;
  thread::scope(|scope| {
    for _ in 0..threads {
      let fsm = Arc::clone(&fsm);
      let hook = Arc::clone(&hook);
      let in_use = Arc::clone(&in_use);
      scope.spawn(move || {
        block_on(async {
          let mut split = Vec::new();
          for _ in 0..ITERS {
            split.clear();
            if let Some(pid) = fsm
              .request_free_extent(1, |_| true, &*hook, &mut split)
              .await
            {
              // 双分配检测：该 pid 不应当已在用
              assert!(
                !in_use[pid as usize].swap(true, Ordering::AcqRel),
                "同一 extent {pid} 被双分配"
              );
              fsm.public_free_extents().await; // 释放分裂片
              // 归还：先清占用位，再经 to_free_extents 入池（避免池内带占用位）
              in_use[pid as usize].store(false, Ordering::Release);
              fsm.prepare_free_tier(pid, 0, &*hook).await;
            }
            fsm.public_free_extents().await;
          }
        });
      });
    }
  });

  block_on(async {
    fsm.public_free_extents().await;
    assert_eq!(
      fsm.number_of_free_entries(),
      K,
      "最终可用 == 初始（无泄漏）"
    );
  });
  assert!(
    in_use.iter().all(|b| !b.load(Ordering::Acquire)),
    "不应存在仍占用的 extent"
  );
}

#[test]
fn concurrent_4_threads() {
  concurrent_alloc_free(4);
}

#[test]
fn concurrent_8_threads() {
  concurrent_alloc_free(8);
}

/// 重复 3 次 4/8 线程交错 alloc/free（本轮复核要求「重复 3 次」，断言每轮无双分配/无泄漏）。
///
/// 重负载（3 × (4+8) 线程 × 1e5/8 次，累计 > E2 的 100 ms 阈值）⇒ `#[ignore]`，
/// 用 `cargo nextest run -p ldb_free --run-ignored all` 显式运行。
#[test]
#[ignore = "重负载：3 次 × 4/8 线程"]
fn concurrent_repeat_3x() {
  for _ in 0..3 {
    concurrent_alloc_free(4);
    concurrent_alloc_free(8);
  }
}
