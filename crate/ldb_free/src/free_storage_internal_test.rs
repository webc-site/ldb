//! 内部统计测试：直接读 `free_tiers` 各级队列长度（公开语义测试在 `tests/free_test.rs`）。
//!
//! 1e5 随机 alloc/free（混合尺寸）后的分级分布、最大可用 extent、以及大请求失败率曲线。

use std::{
  array::from_fn,
  sync::atomic::{AtomicU64, Ordering},
};

use futures_lite::future::block_on;

use super::*;

/// 计数钩子
#[derive(Default)]
struct Hook {
  freed: AtomicU64,
}

impl TxnLogHook for Hook {
  async fn log_pid2_offset(&self, _pid: PageId) {}
  async fn log_valid_page(&self, _pid: PageId) {}
  async fn log_free_page(&self, _pid: PageId) {
    self.freed.fetch_add(1, Ordering::Relaxed);
  }
}

/// 确定性 xorshift（避免引入 rand 依赖）
struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    let mut x = self.0;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.0 = x;
    x
  }
}

impl FreeStorageManager {
  /// 空闲页总数（`free_tiers` 各级长度 × 级尺寸）
  fn free_page_count(&self) -> u64 {
    (0..NO_TIERS)
      .map(|t| self.free_tiers[t].lock().len() as u64 * tier_size(t as u8))
      .sum()
  }

  /// 各级队列长度
  fn tier_counts(&self) -> [u64; NO_TIERS] {
    from_fn(|t| self.free_tiers[t].lock().len() as u64)
  }

  /// 各级队列内容快照（A2 对拍）
  fn tier_snapshot(&self) -> [VecDeque<PageId>; NO_TIERS] {
    from_fn(|t| self.free_tiers[t].lock().clone())
  }

  /// 待公布 extent 快照（A2 对拍）
  fn pending_snapshot(&self) -> Vec<ExtentTier> {
    self.to_free_extents.lock().clone()
  }
}

/// 初始单块所在 tier（比最大请求大一级 ⇒ 首块必被拆分、永不重组，见断言）。
const INIT_TIER: u8 = 21;
/// 初始总页数
const INIT_PAGES: u64 = 1u64 << INIT_TIER;
/// 最大请求级
const MAX_REQ_TIER: u8 = 20;

#[compio::test]
async fn fragmentation_distribution_and_large_request_failure() {
  let fsm = FreeStorageManager::new(true, 4);
  let hook = Hook::default();
  fsm.prepare_free_tier(1, INIT_TIER, &hook).await;
  fsm.public_free_extents().await;

  let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
  let mut live: Vec<(PageId, u8)> = Vec::new();
  let mut split = Vec::new();

  for step in 0..100_000u64 {
    let r = rng.next();
    if r % 3 != 0 && !live.is_empty() {
      let idx = (rng.next() as usize) % live.len();
      let (pid, tier) = live.swap_remove(idx);
      fsm.prepare_free_tier(pid, tier, &hook).await;
    } else {
      let k = (rng.next() % (u64::from(MAX_REQ_TIER) + 1)) as u8;
      split.clear();
      if let Some(pid) = fsm
        .request_free_extent(tier_size(k), |_| true, &hook, &mut split)
        .await
      {
        live.push((pid, k));
      }
    }
    fsm.public_free_extents().await;
    // 页数守恒：空闲页 + 在用页 == 初始（既不泄漏也不重复分配）
    if step % 10_000 == 0 {
      let live_pages: u64 = live.iter().map(|(_, k)| tier_size(*k)).sum();
      assert_eq!(
        fsm.free_page_count() + live_pages,
        INIT_PAGES,
        "step {step}: 页数不守恒"
      );
    }
  }

  // 分布 + 最大可用 extent
  let counts = fsm.tier_counts();
  let nonempty: Vec<(u8, u64)> = (0..NO_TIERS as u8)
    .filter(|&t| counts[t as usize] > 0)
    .map(|t| (t, counts[t as usize]))
    .collect();
  let counts_lit: Vec<(u8, u64)> = nonempty;
  assert!(
    !counts_lit.is_empty(),
    "碎片化后仍应有空闲 extent：{counts_lit:?}"
  );
  let largest = (0..NO_TIERS as u8)
    .rev()
    .find(|&t| counts[t as usize] > 0)
    .map_or(0, tier_size);
  // 初始 2^21 单块一旦被首次分配即拆分，且低 tier 片永不重组 ⇒ tier21 恒空。
  assert_eq!(counts[INIT_TIER as usize], 0, "初始块不应重组");
  assert!(largest <= tier_size(MAX_REQ_TIER));
  assert!(largest < INIT_PAGES);

  // 大请求失败率曲线：每个请求级探测 100 次（成功即原样归还，不改变总页数）
  let mut curve: Vec<(u8, u32)> = Vec::new();
  for k in 0..=MAX_REQ_TIER {
    let want = tier_size(k);
    let mut miss = 0u32;
    for _ in 0..100 {
      split.clear();
      match fsm
        .request_free_extent(want, |_| true, &hook, &mut split)
        .await
      {
        Some(pid) => {
          fsm.public_free_extents().await;
          fsm.prepare_free_tier(pid, k, &hook).await;
          fsm.public_free_extents().await;
        }
        None => miss += 1,
      }
    }
    curve.push((k, miss));
  }
  // 曲线为确定性（固定种子）：逐点字面量断言
  assert_eq!(
    curve,
    [
      (0, 0u32),
      (1, 0),
      (2, 0),
      (3, 0),
      (4, 0),
      (5, 0),
      (6, 0),
      (7, 0),
      (8, 0),
      (9, 0),
      (10, 0),
      (11, 0),
      (12, 0),
      (13, 0),
      (14, 0),
      (15, 0),
      (16, 0),
      (17, 0),
      (18, 0),
      (19, 100),
      (20, 100),
    ],
    "大请求失败率曲线（k, miss/100）"
  );
  // 曲线含义：碎片化后最大可用 extent 恰为 tier18（2^18 = 262144 页）；
  // tier19/2^19 以上请求 100% 失败。
  assert_eq!(largest, tier_size(18));
  // 各级非空分布（tier, 条目数）；1e5 随机 alloc/free 后的确定性结果
  assert_eq!(
    counts_lit,
    [
      (0, 3u64),
      (1, 3),
      (2, 3),
      (3, 5),
      (4, 4),
      (5, 4),
      (6, 4),
      (7, 4),
      (8, 4),
      (9, 6),
      (10, 5),
      (11, 5),
      (12, 5),
      (13, 4),
      (14, 5),
      (15, 5),
      (16, 5),
      (17, 4),
      (18, 3),
    ],
    "碎片化后各 tier 分布"
  );

  // 模型校验：full（同级 + 跨级回退）能成功 ⇔ 存在 tier >= k 非空（上游 :67-81）；
  // sametier（仅同级）能成功 ⇔ tier k 非空。逐点与实测失败曲线一致。
  let largest_tier = (0..NO_TIERS as u8)
    .rev()
    .find(|&t| counts[t as usize] > 0)
    .unwrap_or(0);
  for (k, miss) in &curve {
    let sametier = counts[*k as usize] > 0;
    let full = largest_tier >= *k;
    assert_eq!(full, *miss == 0, "模型-实测不一致: k={k} miss={miss}");
    // 本分布（tier 0..=18 全非空）下两条策略一致
    let _ = sametier;
  }

  // 对照实验（请求尺寸 × 碎片程度）：只放 3 个 tier12(4096 页) extent、无任何更小 tier
  // ⇒ 「仅同级」对小/中请求 100% 失败，而「同级+跨级回退」全部成功。
  let big = FreeStorageManager::new(true, 1);
  block_on(async {
    for i in 0..3u64 {
      big.prepare_free_tier(1000 + i * 4096, 12, &hook).await;
    }
    big.public_free_extents().await;
  });
  let bc = big.tier_counts();
  let bl = (0..NO_TIERS as u8)
    .rev()
    .find(|&t| bc[t as usize] > 0)
    .unwrap_or(0);
  assert_eq!((bl, bc[12], bc[11]), (12, 3, 0), "池只有 tier12");
  let benefit: Vec<(u8, bool, bool)> = (0u8..=13)
    .map(|k| (k, bc[k as usize] > 0, bl >= k))
    .collect();
  // 字面量：k=0..=11 仅同级=false / 全策略=true；k=12 池恰为该级（仅同级也 true）；k=13 两者皆 false
  assert_eq!(
    benefit,
    [
      (0, false, true),
      (1, false, true),
      (2, false, true),
      (3, false, true),
      (4, false, true),
      (5, false, true),
      (6, false, true),
      (7, false, true),
      (8, false, true),
      (9, false, true),
      (10, false, true),
      (11, false, true),
      (12, true, true),
      (13, false, false),
    ],
    "(请求级, 仅同级可成功, 全策略可成功)"
  );
  // 真实取值：请求 5 页（tier3）由 tier12 拆分成功；尾部 4091 = 4096-5 的贪心分解（11 片）
  let mut bsplit = Vec::new();
  let bgot = block_on(big.request_free_extent(5, |_| true, &hook, &mut bsplit));
  assert_eq!(bgot, Some(1000));
  assert_eq!(bsplit.len(), 11, "4091 = 2048+...+1 共 11 片");
  assert_eq!(
    bsplit[0],
    ExtentTier {
      start_pid: 1005,
      tier: 11
    }
  );
  assert_eq!(
    bsplit[10],
    ExtentTier {
      start_pid: 5095,
      tier: 0
    }
  );

  // focus 5：收益表 3 档**实测复测**。构造同一「仅 tier12」池，逐档真实调用
  // `request_free_extent(2^k)` 计成功/失败，验证 k<12 时「仅同级」池空（无回退必失败）
  // 而「同级+回退」必成功（回退自 tier12，分裂尾部非空）。
  for k in [3u8, 8, 12] {
    let f = FreeStorageManager::new(true, 1);
    for i in 0..3u64 {
      f.prepare_free_tier(1000 + i * 4096, 12, &hook).await;
    }
    f.public_free_extents().await;
    if k < 12 {
      assert_eq!(
        f.tier_counts()[k as usize],
        0,
        "k={k}: 仅同级池空（无回退必失败）"
      );
    }
    let mut s = Vec::new();
    let got = f
      .request_free_extent(tier_size(k), |_| true, &hook, &mut s)
      .await;
    assert_eq!(got, Some(1000), "k={k}: 全策略成功（回退自 tier12）");
    // k<12：回退自 tier12 并有尾部拆分；k=12：恰为该级，不拆分
    assert_eq!(
      s.is_empty(),
      k == 12,
      "k={k}: 分裂尾部是否为空（k<12 非空，k=12 恰级为空）"
    );
  }
}

/// FNV-1a 64
#[inline]
fn fnv_mix(h: &mut u64, x: u64) {
  *h ^= x;
  *h = h.wrapping_mul(0x0000_0100_0000_01b3);
}

/// A2 对拍：上游 `TailExtent::SplitToExtents`（`extent_list.cc:5-23`）的函数体原样
/// 编译为 C++ oracle（仅把 `TIER_SIZE` 换成本仓 `2^tier` 表），对全部 `N=1..=4096`
/// 的分解逐片 `(n, tier, pid)` 取 FNV-1a，得 checksum `14869159801385279179`。
/// 此处调用被测 `prepare_free_pages` 产出同序列并比对：MISMATCH 即不等。
#[test]
fn split_decomposition_matches_upstream_oracle() {
  let fsm = FreeStorageManager::new(true, 1);
  let hook = Hook::default();
  let mut out = Vec::new();
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  for n in 1u64..=4096 {
    out.clear();
    block_on(fsm.prepare_free_pages(1_000_000, n, &mut out, &hook));
    for e in &out {
      fnv_mix(&mut h, n);
      fnv_mix(&mut h, u64::from(e.tier));
      fnv_mix(&mut h, e.start_pid);
    }
    block_on(fsm.public_free_extents());
  }
  assert_eq!(
    h, 14_869_159_801_385_279_179,
    "split 分解与上游 oracle MISMATCH"
  );
}

/// 上游 `FreePageManager`（`ZLeanStore/src/storage/free_page_manager.cc`）的**独立重实现**，
/// 仅覆盖 `PrepareFreePages` + `TailExtent::SplitToExtents` + 跨级回退三段核心语义。
///
/// 与生产实现刻意**不同的写法**：tier 表用 `[VecDeque; NO_TIERS]`，拆分照抄上游
/// `extent_list.cc:5-23` 的「index 自顶级递减」字面循环（不用 `ilog2`），请求照抄
/// `free_page_manager.cc:42-81` 的两段结构。若两者语义等价，对拍 MISMATCH=0。
///
/// （合并说明：R2 的单 seed oracle 与本 oracle 击杀集相同 ⇒ 按 E2 反模式 7 合并为本
/// 多 seed 版，覆盖更广。）
struct Oracle {
  tiers: [VecDeque<PageId>; NO_TIERS],
  pending: Vec<ExtentTier>,
}

impl Oracle {
  fn new() -> Self {
    Self {
      tiers: from_fn(|_| VecDeque::new()),
      pending: Vec::new(),
    }
  }

  /// 上游 `PrepareFreeTier`（`free_page_manager.cc:91-98`）
  fn prepare_free_tier(&mut self, start_pid: PageId, tier: u8) {
    self.pending.push(ExtentTier { start_pid, tier });
  }

  /// 上游 `PublicFreeRanges`（`free_page_manager.cc:100-107`）：drain 顺序入队
  fn public_free_extents(&mut self) {
    for e in self.pending.drain(..) {
      self.tiers[e.tier as usize].push_back(e.start_pid);
    }
  }

  /// 上游 `TailExtent::SplitToExtents`（`extent_list.cc:5-23`）字面循环
  fn split_to_extents(&mut self, start_pid: PageId, mut page_cnt: u64, out: &mut Vec<ExtentTier>) {
    let mut index = (NO_TIERS - 1) as i64;
    let mut pid = start_pid;
    while page_cnt > 0 {
      while page_cnt < tier_size(index as u8) {
        index -= 1;
      }
      self.prepare_free_tier(pid, index as u8);
      out.push(ExtentTier {
        start_pid: pid,
        tier: index as u8,
      });
      pid += tier_size(index as u8);
      page_cnt -= tier_size(index as u8);
    }
  }

  /// 上游 `RequestFreeExtent`（`free_page_manager.cc:29-84`）两段结构
  fn request(
    &mut self,
    page_cnt: u64,
    mut validate: impl FnMut(PageId) -> bool,
  ) -> (Option<PageId>, Vec<ExtentTier>) {
    let tier = tier_index(page_cnt) as usize;
    let mut out = Vec::new();
    // 同级最多 3 次（:42-61）
    for _ in 0..3 {
      let Some(pid) = self.tiers[tier].pop_front() else {
        break;
      };
      if validate(pid) {
        self.split_to_extents(pid + page_cnt, tier_size(tier as u8) - page_cnt, &mut out);
        return (Some(pid), out);
      }
      self.tiers[tier].push_back(pid);
    }
    // 自最高级向下；i==tier 时看顶级（:67-81）
    for i in (tier..NO_TIERS - 1).rev() {
      let x = if i == tier { NO_TIERS - 1 } else { i };
      let Some(pid) = self.tiers[x].pop_front() else {
        continue;
      };
      if validate(pid) {
        self.split_to_extents(pid + page_cnt, tier_size(x as u8) - page_cnt, &mut out);
        return (Some(pid), out);
      }
      self.tiers[x].push_back(pid);
    }
    (None, out)
  }
}

/// A2/focus 1 差分对拍：**5 seeds × 2e4 = 1e5** 随机操作（prepare / public / request 混合），
/// 逐点比对被测 `request_free_extent` 与上游 `Oracle`：返回 pid、`out_split_exts`、
/// 全 tier 队列内容、pending 表。`MISMATCH=0` 才算通过。
///
/// 跨 5 个种子（不同 `validate` 拒绝律 + 不同 tier 序列）⇒ 覆盖单种子遗漏的路径组合。
///
/// 唯一击杀：M14（删除回退循环 `push_back`——校验失败的更高 tier extent 被吞掉；
/// 该路径不被其它用例触达，故只有本对拍能击杀）。
///
/// 重负载（1e5 次 `block_on` + 全状态快照，超 E2 的 100 ms 阈值）⇒ `#[ignore]`，
/// 用 `cargo nextest run -p ldb_free --run-ignored all` 显式运行。
#[test]
#[ignore = "A2/focus1 差分对拍 5 seeds × 2e4 = 1e5（超 E2 100ms）"]
fn request_free_extent_matches_upstream_oracle() {
  let seeds = [
    0xD1B5_4A32_D192_ED03u64,
    0x1234_5678_9ABC_DEF0,
    0xDEAD_BEEF_CAFE_F00D,
    0x0F1E_2D3C_4B5A_6978,
    0xA5A5_5A5A_C3C3_3C3C,
  ];
  for &seed in &seeds {
    let fsm = FreeStorageManager::new(true, 1);
    let hook = Hook::default();
    let mut oracle = Oracle::new();
    let mut rng = Rng(seed);
    let mut split = Vec::new();
    let mut mismatch = 0u64;
    for _ in 0..20_000u64 {
      match rng.next() % 3 {
        // prepare_free_tier：pid 1..=4096、tier 0..=20
        0 => {
          let pid = 1 + rng.next() % 4096;
          let tier = (rng.next() % 21) as u8;
          block_on(fsm.prepare_free_tier(pid, tier, &hook));
          oracle.prepare_free_tier(pid, tier);
        }
        // public_free_extents
        1 => {
          block_on(fsm.public_free_extents());
          oracle.public_free_extents();
        }
        // request：尺寸 2^k（k=0..=12）；校验按 pid+seed 随机拒绝（覆盖回退失败路径）
        _ => {
          let k = (rng.next() % 13) as u8;
          let want = tier_size(k);
          let s = rng.next();
          let vf = move |pid: PageId| pid.wrapping_add(s) % 7 != 0;
          split.clear();
          let got = block_on(fsm.request_free_extent(want, vf, &hook, &mut split));
          let (o_got, o_split) = oracle.request(want, vf);
          if got != o_got || split != o_split {
            mismatch += 1;
          }
        }
      }
      if fsm.tier_snapshot() != oracle.tiers || fsm.pending_snapshot() != oracle.pending {
        mismatch += 1;
      }
    }
    assert_eq!(mismatch, 0, "seed {seed:#x} 对拍 MISMATCH={mismatch}");
  }
}
