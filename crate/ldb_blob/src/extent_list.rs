//! ExtentList（指数分级 extent 链，上游 `storage/extent/extent_list.h`）。
//!
//! Tier 尺寸 2^tier 页；尾 extent（TailExtent）容纳不满整级的尾部数据，
//! 攒满后 split_to_extents 一次拆出整级 extent。
//!
//! # 布局决策（R3）：保留链式实现，对齐**上限语义**而非**内存布局**
//!
//! 上游 `ExtentList` 是**扁平柔性数组** `pageid_t extent_pid[]`（`extent_list.h:109-111`）
//! —— 头 12 B（`static_assert(sizeof(ExtentList) == 12)`）+ 每 extent 8 B（一个
//! `pageid_t`），`MallocSize(127) = 12 + 8*127 = 1028`（`extent_list.h:186-188,235`）。
//! `BlobState`/`BlobHandler` 把这一数组内联并定长到 1100 B（`blob_state.h:27,114`）。
//!
//! 本仓保留 `Arc<Node>` 单链，理由：
//! 1. **上限语义已对齐**（本轮核心）：超过 127 个 extent 时 [`ExtentList::append`]
//!    返回 [`Error::TooManyExtents`]，与上游「定长 1100 B 结构放不下第 128 个
//!    extent」的拒绝语义一致；
//! 2. 链式给出 O(1) 头插 + 快照式迭代（`iter()` 物化 `Vec<Extent>`），并为
//!    [`Node`] 的迭代式 `Drop`（R2 P0 修复）提供载体；
//! 3. 改动面最小、回归风险最低：扁平数组需要「按需扩容 / 溢出层」，且会牵动
//!    `begin/end/iter/append` 全部调用点。
//!
//! **内存代价（实测，见报告 §3.2）**：链式每 extent 为 `Node`（`Extent` 16 B +
//! `Option<Arc<Node>>` 8 B = 24 B）**加上每次 `Arc` 分配头 16 B**（强/弱计数），
//! 合计 **~40 B/extent**；上游为 **8 B/extent**（一个 `pageid_t`）。逐 extent 多
//! **~32 B**。因为上界已封顶到 127，单 blob 的额外开销有界（≤ ~4 KiB），
//! 不再随 `NO_TIERS` 无界增长。

use std::{
  fmt,
  mem::size_of,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
};

use ldb_type::{PAGE_SIZE, PageId};
use parking_lot::RwLock;

use crate::error::{Error, Result};

/// 分级数（上游 NO_TIERS）
pub const NO_TIERS: usize = 64;

/// 单个 blob 的 extent 数上限。
///
/// 对齐上游 `ExtentList::EXTENT_CNT_MASK = 0x7F`（即 `2^7 - 1 = 127`）：上游把
/// genesis `extent_cnt` 存成 **7 位位域**（`extent_list.h:109`），并把
/// `BlobState`/`BlobHandler` 的实际存储定长到 `MallocSize(EXTENT_CNT_MASK) = 1100`
/// 字节（`blob_state.h:27,114`、`blob_handler.h:46,134`）；`BlobManager` 的
/// `blob_state_storage` 也是这一固定尺寸的 `std::array<u8, 1100>`（`blob_manager.h:36-38`）
/// —— 超过 127 个 extent 在结构上放不下，`extent_cnt` 会位域回绕并越界写栈上
/// 缓冲，故上游**结构上拒绝**超限（`blob_handler.h:35-36` 原文：
/// "we want set a limit of 127 extents (7 bits, 1 reserved bit for later use)")。
/// 本仓以链式实现承载同一上界语义：超过即 [`Error::TooManyExtents`]，绝不 panic、
/// 绝不静默截断。
pub const MAX_EXTENTS: usize = 127;

/// 各级 extent 页数（2^tier）
#[inline]
pub const fn extent_size(tier: u8) -> u64 {
  1u64 << tier
}

/// 页数 → tier（向上取整）
#[inline]
pub fn tier_index(page_cnt: u64) -> u8 {
  if page_cnt <= 1 {
    0
  } else {
    page_cnt.next_power_of_two().trailing_zeros() as u8
  }
}

/// extent 描述（起始页 + tier）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
  /// 起始页号
  pub start_pid: PageId,
  /// 分级
  pub tier: u8,
}

impl Extent {
  /// 页数
  #[inline]
  pub fn pages(&self) -> u64 {
    extent_size(self.tier)
  }
}

/// 链表节点（Arc 链式；快照迭代安全）
#[derive(Debug)]
struct Node {
  ext: Extent,
  next: Option<Arc<Node>>,
}

impl Drop for Node {
  /// 迭代拆链：默认递归 Drop 在长链（实测 1e5 节点即 SIGABRT）会栈溢出。
  /// 每步只对「唯一持有」的节点取走 `next`；仍被外部快照持有的节点，
  /// 由其自身 Drop 从此处继续迭代 —— 递归深度恒为 1。
  fn drop(&mut self) {
    let mut cur = self.next.take();
    while let Some(arc) = cur {
      match Arc::try_unwrap(arc) {
        Ok(mut node) => cur = node.next.take(),
        Err(_) => break,
      }
    }
  }
}

/// 尾 extent（不足整级的动态缓冲，上游 TailExtent）
#[derive(Debug)]
pub struct TailExtent {
  /// 起始页号
  pub start_pid: PageId,
  /// 已用页数
  pub used_pages: u64,
  /// 缓冲容量（页）
  pub capacity_pages: u64,
  /// 数据缓冲（连续）
  pub data: Vec<u8>,
}

impl TailExtent {
  /// 构造
  pub fn new(start_pid: PageId, capacity_pages: u64) -> Self {
    Self {
      start_pid,
      used_pages: 0,
      capacity_pages,
      data: Vec::with_capacity(capacity_pages as usize * PAGE_SIZE),
    }
  }

  /// 追加数据（返回是否新增整页；上游语义：先收后拆 —— 超容量由
  /// `split_to_extents` 弹出整级，不拒收）
  pub fn append(&mut self, bytes: &[u8]) -> bool {
    self.data.extend_from_slice(bytes);
    let new_pages = self.data.len().div_ceil(PAGE_SIZE) as u64;
    let grew = new_pages > self.used_pages;
    self.used_pages = new_pages;
    grew
  }

  /// 剩余字节（最后未满页）
  #[inline]
  pub fn remain_bytes(&self) -> u64 {
    (self.data.len() % PAGE_SIZE) as u64
  }

  /// 攒满拆级：从尾缓冲前端弹出整级数据块（上游 SplitToExtents 的贪心拆分）。
  ///
  /// 返回每个整级 extent 的**页数据**（各 `full_pages * PAGE_SIZE` 字节）；
  /// 起始 pid 由调用方（`BlobManager`）在**独立分配**区段后写入 —— 本函数不
  /// 触碰 `start_pid`：弹出的整级块成为独立 extent，剩余尾数据留在本区段前端
  /// （`used_pages` 减到 ≤ `full_pages`，尾区始终容纳得下）。
  ///
  /// 上游 `TailExtent::SplitToExtents(start_pid, page_cnt, cb)` 在**已分配**的
  /// 连续区间内按 tier 贪心切分；本仓把「分配」交还调用方，避免多个整级 extent
  /// 共享同一尾区段造成 pid 重叠（上游每个 extent 由 `AllocExtent` 单独分配）。
  pub fn split_to_extents(&mut self) -> Vec<Vec<u8>> {
    let full_pages = extent_size(tier_index(self.capacity_pages.max(1)));
    let full_bytes = full_pages as usize * PAGE_SIZE;
    let mut out = Vec::new();
    while self.used_pages > full_pages && full_pages > 0 {
      let bytes: Vec<u8> = self.data.drain(..full_bytes).collect();
      self.used_pages -= full_pages;
      out.push(bytes);
    }
    out
  }
}

/// 预测 [`TailExtent::split_to_extents`] 将弹出的整级 extent 数（`used_pages` 为
/// **追加之后**的页数）。与拆级循环同式：`while used > full { used -= full; n += 1 }`
/// 等价于 `used > full ? (used - 1) / full : 0`。用于在真正拆级（会 `drain` 掉尾数据、
/// 不可回滚）**之前**判定是否会越过 [`MAX_EXTENTS`]。
#[inline]
fn split_count(used_pages: u64, full_pages: u64) -> u64 {
  if full_pages == 0 || used_pages <= full_pages {
    0
  } else {
    (used_pages - 1) / full_pages
  }
}

/// ExtentList（头节点链 + 尾缓冲；迭代器快照安全）
pub struct ExtentList {
  head: RwLock<Option<Arc<Node>>>,
  /// 尾 extent
  pub tail: RwLock<TailExtent>,
  /// extent 计数
  count: AtomicU64,
  /// 跨 extent 字节累计（整级部分）
  bytes: AtomicU64,
}

impl fmt::Debug for ExtentList {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ExtentList")
      .field("count", &self.count.load(Ordering::Relaxed))
      .field("tail_used", &self.tail.read().used_pages)
      .finish()
  }
}

impl ExtentList {
  /// 构造（尾缓冲起始于 `tail_start_pid`，容量 = tier 容量）
  pub fn new(tail_start_pid: PageId, tail_capacity_pages: u64) -> Self {
    Self {
      head: RwLock::new(None),
      tail: RwLock::new(TailExtent::new(tail_start_pid, tail_capacity_pages)),
      count: AtomicU64::new(0),
      bytes: AtomicU64::new(0),
    }
  }

  /// extent 计数
  #[inline]
  pub fn number_of_extents(&self) -> u64 {
    self.count.load(Ordering::Relaxed)
  }

  /// 整级字节总量（不含尾）
  #[inline]
  pub fn total_size_extents(&self) -> u64 {
    self.bytes.load(Ordering::Relaxed)
  }

  /// 内存占用（上游 MallocSize）
  pub fn malloc_size(&self) -> u64 {
    self.number_of_extents() * size_of::<Node>() as u64 + self.tail.read().data.len() as u64
  }

  /// 指定级 extent 尺寸
  #[inline]
  pub fn extent_size(&self, tier: u8) -> u64 {
    extent_size(tier)
  }

  /// 页数定级
  #[inline]
  pub fn tier_index(&self, page_cnt: u64) -> u8 {
    tier_index(page_cnt)
  }

  /// 追加 extent（头插；上游 push）。
  ///
  /// 这是**无界的结构化原语**：只负责把 `Extent` 头插进链，不校验
  /// [`MAX_EXTENTS`]。生产增长路径一律经 [`Self::append`]（在那里按上游定长
  /// 语义拒绝超限）；本原语保留无界只是为了长链结构压强测试（Drop 迭代性）。
  pub fn push_extent(&self, ext: Extent) {
    let mut h = self.head.write();
    *h = Some(Arc::new(Node {
      ext,
      next: h.clone(),
    }));
    drop(h);
    self.count.fetch_add(1, Ordering::Relaxed);
    self
      .bytes
      .fetch_add(ext.pages() * PAGE_SIZE as u64, Ordering::Relaxed);
  }

  /// 首_extent
  pub fn begin(&self) -> Option<Extent> {
    self.head.read().as_ref().map(|n| n.ext)
  }

  /// 末 extent（链尾；O(n) 冷路径）
  pub fn end(&self) -> Option<Extent> {
    let h = self.head.read();
    let mut cur = h.clone();
    let mut last = None;
    while let Some(n) = cur {
      last = Some(n.ext);
      cur = n.next.clone();
    }
    last
  }

  /// 数据追加（尾缓冲；攒满自动拆级）。**可失败**：若本次拆级会使 extent 总数
  /// 越过 [`MAX_EXTENTS`]，在**改动尾缓冲之前**返回 [`Error::TooManyExtents`]
  /// （上游 `BlobState` 定长 1100B 结构上放不下第 128 个 extent，见 `MAX_EXTENTS` 文档）。
  ///
  /// 返回本次弹出的整级数据块（每块 `full_pages * PAGE_SIZE` 字节）。调用方
  /// （`BlobManager`）负责为每块**独立分配**区段、构造 `Extent` 并
  /// [`Self::push_extent`]。
  ///
  /// 上界判定在 `drain` 之前完成（拆级会消费尾数据、不可回滚），判据 =
  /// 已登记 extent 数 + 本次将弹出的块数（[`split_count`]），与
  /// [`TailExtent::split_to_extents`] 的贪心循环严格同式。
  pub fn append(&self, data: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut t = self.tail.write();
    let full_pages = extent_size(tier_index(t.capacity_pages.max(1)));
    let used_pages = (t.data.len() + data.len()).div_ceil(PAGE_SIZE) as u64;
    if self.count.load(Ordering::Relaxed) + split_count(used_pages, full_pages) > MAX_EXTENTS as u64
    {
      return Err(Error::TooManyExtents);
    }
    t.append(data);
    Ok(t.split_to_extents())
  }

  /// 尾部剩余字节
  #[inline]
  pub fn remain_bytes_in_last_extent(&self) -> u64 {
    self.tail.read().remain_bytes()
  }

  /// 无跨界 extent 判定：链上**每个** extent 是否都不跨 `block_pages` 页的物理块边界。
  ///
  /// **与上游同名函数语义不同（R5 核实，保留并注明）**：上游
  /// `ExtentList::NoSpanExtents(pageid_t no_pages) -> extidx_t`
  /// （`extent_list.h:205-218`）计算「`no_pages` 页的 payload 跨多少个 extent（从 tier 0 起）」，
  /// 是个**计数**函数，被 `BlobState::RemainBytesInLastExtent`（`blob_state.h:85`）与
  /// `FreshBlobAllocation`/`ExtendExistingBlob` 生产引用；本函数是个**谓词**
  /// （`bool`），与本 crate 的链式布局配套（本实现按提领顺序读 extent，无跨物理块需求）。
  ///
  /// 本函数为账本立项目标（`tool/map.js` ldb_blob `ExtentList::NoSpanExtents` 行 88-90，
  /// `LDB_BLOB_SYNC_FNS`），当前**无生产消费者**（仅 `tests/extent_list_test.rs`）；
  /// 因改名/改语义会破坏账本映射且 `tool/map.js` 不在本 crate 改动范围，故保留原名，
  /// 在此写明差异（详见 `CROSS.md`）。
  pub fn no_span_extents(&self, block_pages: u64) -> bool {
    self.iter().into_iter().all(|e| {
      let pages = e.pages();
      let start = e.start_pid % block_pages;
      start + pages <= block_pages || pages >= block_pages
    })
  }

  /// 遍历所有 extent（快照）
  pub fn iter(&self) -> Vec<Extent> {
    let mut out = Vec::with_capacity(self.number_of_extents() as usize);
    let h = self.head.read();
    let mut cur = h.clone();
    while let Some(n) = cur {
      out.push(n.ext);
      cur = n.next.clone();
    }
    out.reverse(); // 头插序反转回追加序
    out
  }
}

/// extent 页读取闭包（数据后端句柄）
pub type ExtentFetcher = Box<dyn Fn(PageId, u64, &mut Vec<u8>) + Send + Sync>;

/// 迭代器（账本 Iterator 形态：检索 extent 内数据）
pub struct Iterator {
  /// 剩余 extent 序列
  pub rest: Vec<Extent>,
  /// 消费游标（`retrieve_data` 推进；避免 `rest.remove(0)` 的 O(n²)）
  cursor: usize,
  /// 数据后端（页读取闭包）
  pub fetch: ExtentFetcher,
}

impl Iterator {
  /// 构造（rest + 数据后端；游标归零）
  pub fn new(rest: Vec<Extent>, fetch: ExtentFetcher) -> Self {
    Self {
      rest,
      cursor: 0,
      fetch,
    }
  }
}

impl fmt::Debug for Iterator {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Iterator")
      .field("rest", &self.rest)
      .finish_non_exhaustive()
  }
}

impl Iterator {
  /// 当前 tier
  pub fn tier_index(&self) -> Option<u8> {
    self.rest.first().map(|e| e.tier)
  }

  /// 检索当前 extent 数据（整级页聚合；游标下标推进 —— O(1) 消费）
  pub fn retrieve_data(&mut self, out: &mut Vec<u8>) -> bool {
    let Some(ext) = self.rest.get(self.cursor).copied() else {
      return false;
    };
    self.cursor += 1;
    let pages = ext.pages();
    (self.fetch)(ext.start_pid, pages, out);
    true
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 独立数学式（按 `while used > full { used -= full; n += 1 }` 闭式重写，
  /// 故意不复用实现里的 `split_count`）。
  fn oracle_split_count(used: u64, full: u64) -> u64 {
    if full == 0 || used <= full {
      0
    } else {
      (used - 1) / full
    }
  }

  /// 黑箱观测 `split_to_extents` 贪心循环实际弹出的块数与剩余页数。
  fn loop_pops(full: u64, used: u64) -> (u64, u64) {
    let mut t = TailExtent::new(0, full);
    t.data = vec![0u8; (used * PAGE_SIZE as u64) as usize];
    t.used_pages = used;
    (t.split_to_extents().len() as u64, t.used_pages)
  }

  /// 上限判据 `split_count` 必须与拆级循环**严格同式**（差一个即出现
  /// 「判据放过但拆级越界」或「判据过严」）。逐例对拍判据 == 独立 oracle == 循环弹出数。
  #[test]
  fn split_count_equals_oracle_and_split_loop() {
    // full 为 2 的幂（容量 == full 时 `full_pages == capacity`）
    let mut cases: Vec<(u64, u64)> = Vec::new();
    for full in [1u64, 2, 4, 8] {
      cases.extend((0..=(4 * full)).map(|used| (full, used)));
    }
    // 较大 full 取稀疏（含精确倍数 ±1），守住 100ms 预算
    for full in [16u64, 32] {
      cases.extend(
        [
          0,
          full - 1,
          full,
          full + 1,
          2 * full,
          2 * full + 1,
          4 * full,
        ]
        .into_iter()
        .map(|used| (full, used)),
      );
    }
    for (full, used) in cases {
      let judge = split_count(used, full);
      assert_eq!(
        judge,
        oracle_split_count(used, full),
        "判据≠oracle used={used} full={full}"
      );
      let (pops, remain) = loop_pops(full, used);
      assert_eq!(judge, pops, "判据≠循环弹出数 used={used} full={full}");
      assert_eq!(
        remain,
        used - pops * full,
        "剩余页数 used={used} full={full}"
      );
      assert!(remain <= full, "剩余必须 <= full");
    }
  }
}
