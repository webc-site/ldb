//! 空闲空间管理（合并上游 `leanstore.latency` FreeStorageManager 与 ZLeanStore
//! FreePageManager，消除重命名重复）。
//!
//! 两种形态：
//! - Extent 分级（Blob 关闭 / `FLAGS_blob_buffer_pool_gb == 0`）：`free_tiers_[tier]`
//!   并发队列，按 2 的幂分级（见下「分级方案」）。
//! - RangeLock（Blob 开启 / `FLAGS_blob_buffer_pool_gb != 0`）：一维区间锁分配
//!   （上游 `free_range_`，`free_storage.h:42` / `free_storage.cc:9`）。`RangeLock`
//!   自身 `Send + Sync`（内部 `Mutex<Inner>`，`ldb_range::range_lock`），无需外层再套锁。
//!
//! # 分级方案（与上游的**有意偏差**，与 `ldb_blob` 逐字对齐）
//!
//! 上游 `ExtentList::NO_TIERS == 120`，`TIER_SIZE[i]` 是 `(level+1)^8` 的几何插值
//! （`extent_list.h:74-100`）。本仓统一改「64 级 × `2^tier`」方案，与同层
//! `ldb_blob::extent_list`（同一 `ExtentList` 的另一半）逐字一致 —— 两侧共享 extent
//! 页数语义，必须同构，否则 `ldb_free` 归还的 extent 尺寸与 `ldb_blob` 认领的不符。
//! 上游 `TierIndex` 语义为「取最小的 `TIER_SIZE >= page_cnt`」（`extent_list.h:190-199`），
//! `Ensure(ExtentSize(tier_index) >= required_page_cnt)`（`free_page_manager.cc:37`）在
//! 2 的幂方案下等价于「向上取整到 2 的幂」，即 `next_power_of_two().trailing_zeros()`。
//!
//! # WAL 注入
//!
//! `LogFreePage`/`PrepareFreeTier` 等需写 WAL 的部分经 `TxnLogHook`(L2) 注入，
//! **不得**直接 `use ldb_txn`（消除 L3 → L4 上向边）。
//! 空闲队列低竞争冷路径用 `parking_lot::Mutex<VecDeque>`。

use std::{collections::VecDeque, fmt};

use ldb_page::hook::TxnLogHook;
use ldb_range::RangeLock;
use ldb_type::PageId;
use parking_lot::Mutex;

/// Extent 分级数。
///
/// 上游 `ExtentList::NO_TIERS == NUMBER_OF_LEVELS * EXTENT_PER_LEVEL == 120`
/// （`extent_list.h:74`）；本仓与 `ldb_blob::extent_list::NO_TIERS` 统一为 64（见模块头
/// 「分级方案」）。`extidx_t` 为 `u8`（`common/typedefs.h:19`），64 与 120 均在其值域内。
pub const NO_TIERS: usize = 64;

/// 各级 extent 页数（`2^tier`；见模块头「分级方案」）
#[inline]
pub const fn tier_size(tier: u8) -> u64 {
  1u64 << tier
}

/// 页数 → 分级下标（向上取整到 2 的幂）。
///
/// 语义等价上游 `ExtentList::TierIndex(page_cnt, /*exact_lookup=*/false)`
/// （`extent_list.h:190-199`：最小的 `TIER_SIZE >= page_cnt`）。上界保证
/// `tier_size(tier_index(n)) >= n`（上游 `free_page_manager.cc:37` 的 `Ensure`）。
#[inline]
pub fn tier_index(page_cnt: u64) -> u8 {
  if page_cnt <= 1 {
    return 0;
  }
  page_cnt.next_power_of_two().trailing_zeros() as u8
}

/// 待公布 extent（事务提交后入队；上游 `storage::LargePageTier`，`large_page.h:20-25`）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtentTier {
  /// 起始页
  pub start_pid: PageId,
  /// 分级
  pub tier: u8,
}

/// 空闲空间管理器
pub struct FreeStorageManager {
  free_page: Mutex<VecDeque<PageId>>,
  enable_extent_tier: bool,
  /// Blob 开启时的一维区间分配（`RangeLock` 自带内部锁，`&self` 即可）
  free_range: Option<RangeLock>,
  free_tiers: Vec<Mutex<VecDeque<PageId>>>,
  /// 事务待公布 extent（上游 `Txn::ToFreeExtents()`；Mutex 内部可变性 —— 本管理器
  /// 经 `Arc` 共享，公布路径 `&self`）
  to_free_extents: Mutex<Vec<ExtentTier>>,
}

impl fmt::Debug for FreeStorageManager {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("FreeStorageManager")
      .field("enable_extent_tier", &self.enable_extent_tier)
      .field("pending", &self.to_free_extents.lock().len())
      .finish_non_exhaustive()
  }
}

impl FreeStorageManager {
  /// 构造：`enable_extent_tier == false` 时启用 RangeLock 分配（上游
  /// `FLAGS_blob_buffer_pool_gb != 0` 分支，`free_storage.cc:8-10`）
  pub fn new(enable_extent_tier: bool, workers: usize) -> Self {
    Self {
      free_page: Mutex::default(),
      enable_extent_tier,
      free_range: (!enable_extent_tier).then(|| RangeLock::new(workers)),
      free_tiers: (0..NO_TIERS).map(|_| Mutex::default()).collect(),
      to_free_extents: Mutex::default(),
    }
  }

  /// 单页空闲计数（测试用；非精确 —— 上游 `NumberOfFreeEntries`，`free_storage.cc:27`）
  pub fn number_of_free_entries(&self) -> u64 {
    self.free_tiers.iter().map(|q| q.lock().len() as u64).sum()
  }

  /// 尝试锁住 `[start_pid, start_pid + page_count)`（仅 RangeLock 形态）。
  /// 上游 `TryLockRange`（`free_storage.cc:34-37`）反向 `Ensure(!enable_extent_tier_)`。
  pub async fn try_lock_range(&self, wid: usize, start_pid: PageId, page_count: u64) -> bool {
    assert!(
      !self.enable_extent_tier,
      "仅 RangeLock 形态可用（上游 Ensure 反向条件，free_storage.cc:35）"
    );
    match &self.free_range {
      Some(rl) => rl.try_lock_range(wid, start_pid, page_count),
      None => false,
    }
  }

  /// 从分级队列取一个 `>= page_cnt` 的空闲 extent（上游
  /// `FreePageManager::RequestFreeExtent`，`free_page_manager.cc:29-84`）。
  ///
  /// 逐条对照上游：
  /// 1. 先在同级（`tier_index(page_cnt)`）尝试 3 次（`free_page_manager.cc:42-61`）；
  /// 2. 命中且该级 extent 大于请求时，把尾部 `tier_size - page_cnt` 拆成整级 extent
  ///    经 `PrepareFreePages` 入队并写入 `out_split_exts`（`free_page_manager.cc:49-54`）；
  /// 3. 同级失败后退化为「从最高级向下扫描到 `tier_index`，`i == tier_index` 时取顶级」
  ///    （`free_page_manager.cc:63-81`：`x_tier = (i == tier_index) ? NO_TIERS-1 : i`）；
  /// 4. `validate_extent` 失败时把 extent 压回原级并继续（`free_page_manager.cc:58/79`）；
  /// 5. 全空/全校验失败返回 `None`。
  ///
  /// `out_split_exts` 为分裂出的较小 extent（上游同名 out 参数）。
  pub async fn request_free_extent(
    &self,
    page_cnt: u64,
    mut validate_extent: impl FnMut(PageId) -> bool,
    hook: &impl TxnLogHook,
    out_split_exts: &mut Vec<ExtentTier>,
  ) -> Option<PageId> {
    let tier = tier_index(page_cnt);
    // 上游 `Ensure(ExtentList::ExtentSize(tier_index) >= required_page_cnt)`
    // （`free_page_manager.cc:37`）：tier 尺寸不足即不得再分裂，直接断言。
    assert!(
      tier_size(tier) >= page_cnt,
      "tier {tier} 尺寸 {} 小于请求页数 {page_cnt}（上游 Ensure，free_page_manager.cc:37）",
      tier_size(tier)
    );
    let tier = tier as usize;

    // ① 同级尝试 3 次。取出的守卫是临时值，本语句结束即释放 —— 不在 await 期间持锁。
    for _ in 0..3 {
      let popped = self.free_tiers[tier].lock().pop_front();
      let Some(pid) = popped else { break };
      if validate_extent(pid) {
        hook.log_free_page(pid).await;
        self
          .split_tail(pid, page_cnt, tier_size(tier as u8), out_split_exts, hook)
          .await;
        return Some(pid);
      }
      self.free_tiers[tier].lock().push_back(pid);
    }

    // ② 从最高级向下扫描到 `tier`，`i == tier` 时看顶级（上游 :63-81）。
    for i in (tier..NO_TIERS - 1).rev() {
      let x_tier = if i == tier { NO_TIERS - 1 } else { i };
      let popped = self.free_tiers[x_tier].lock().pop_front();
      let Some(pid) = popped else { continue };
      if validate_extent(pid) {
        hook.log_free_page(pid).await;
        self
          .split_tail(pid, page_cnt, tier_size(x_tier as u8), out_split_exts, hook)
          .await;
        return Some(pid);
      }
      self.free_tiers[x_tier].lock().push_back(pid);
    }
    None
  }

  /// 命中 extent 尺寸大于请求时，拆分尾部并入库（上游 `free_page_manager.cc:49-54/73-75`）。
  #[inline]
  async fn split_tail(
    &self,
    start_pid: PageId,
    page_cnt: u64,
    extent_size: u64,
    out_split_exts: &mut Vec<ExtentTier>,
    hook: &impl TxnLogHook,
  ) {
    if extent_size > page_cnt {
      self
        .prepare_free_pages(
          start_pid + page_cnt,
          extent_size - page_cnt,
          out_split_exts,
          hook,
        )
        .await;
    }
  }

  /// 把 `[start_pid, start_pid + page_cnt)` 拆成整级 extent 逐级准备入库
  /// （上游 `FreePageManager::PrepareFreePages`，`free_page_manager.cc:113-120`；
  /// 分解算法等价 `TailExtent::SplitToExtents`，`extent_list.cc:5-23`：从最大级贪心下调）。
  pub(crate) async fn prepare_free_pages(
    &self,
    start_pid: PageId,
    mut page_cnt: u64,
    out_split_exts: &mut Vec<ExtentTier>,
    hook: &impl TxnLogHook,
  ) {
    let mut pid = start_pid;
    while page_cnt > 0 {
      let tier = page_cnt.ilog2() as u8;
      let sz = tier_size(tier);
      self.prepare_free_tier(pid, tier, hook).await;
      out_split_exts.push(ExtentTier {
        start_pid: pid,
        tier,
      });
      pid += sz;
      page_cnt -= sz;
    }
  }

  /// 预挂起空闲分级（事务提交后 `public_free_extents` 生效；上游 `PrepareFreeTier`，
  /// `free_page_manager.cc:91-98`）
  pub async fn prepare_free_tier(&self, start_pid: PageId, tier: u8, hook: &impl TxnLogHook) {
    self
      .to_free_extents
      .lock()
      .push(ExtentTier { start_pid, tier });
    hook.log_free_page(start_pid).await;
  }

  /// 事务提交后公布全部挂起 extent（上游 `PublicFreeExtents`，`free_storage.cc:83-94` /
  /// `ZLeanStore PublicFreeRanges`，`free_page_manager.cc:100-107`）。
  ///
  /// 就地 `drain` 而非 `take`：`take` 会把挂起表换成空 `Vec`（容量归零）并在循环末尾
  /// 释放取走的缓冲，导致每次公布一次 malloc+free（实测 1.00 alloc/op）；`drain` 复用
  /// 同一缓冲（0 alloc/op）。循环体内只锁 `free_tiers`/`free_range`（叶锁，无反向获取）
  /// 且**无 await**，故不构成跨 await 持锁。
  pub async fn public_free_extents(&self) {
    let mut pending = self.to_free_extents.lock();
    for ext in pending.drain(..) {
      if self.enable_extent_tier {
        self.free_tiers[ext.tier as usize]
          .lock()
          .push_back(ext.start_pid);
      } else if let Some(rl) = &self.free_range {
        rl.unlock_range(0, ext.start_pid);
      }
    }
  }

  /// 对外公布空闲区间（`EvictHook::public_free_ranges` 的 free 侧实现）。
  ///
  /// 先快照后公布 —— 上游 `GroupCommitExecutor` PhaseFive 裁剪的是**已提交**的
  /// `ToFreeExtents`（`group_commit.cc:441/467`）；原实现先 `public_free_extents()`
  /// （take 清空挂起表）再遍历已空的表，恒返回空表。
  pub async fn public_free_ranges(&self) -> Vec<(u64, u64)> {
    let out: Vec<(u64, u64)> = self
      .to_free_extents
      .lock()
      .iter()
      .map(|e| (e.start_pid, tier_size(e.tier)))
      .collect();
    self.public_free_extents().await;
    out
  }

  /// 记录页释放（单页队列 + WAL）。上游 `free_storage.h:38` 的 `free_page_`
  /// 单页空闲队列（`todo.md` §3.4 第 15 条「单页空闲队列」）。
  pub async fn log_free_page(&self, start_pid: PageId, hook: &impl TxnLogHook) {
    self.free_page.lock().push_back(start_pid);
    hook.log_free_page(start_pid).await;
  }
}

/// 内部统计/不变量测试（直接读 `free_tiers`；公开语义测试在 `tests/free_test.rs`）。
#[cfg(test)]
#[path = "free_storage_internal_test.rs"]
mod internal_test;
