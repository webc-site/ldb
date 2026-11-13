//! VMCache 缓冲池管理器（上游 `buffer_manager.cc` 的 compio 收敛版）。
//!
//! 协议要点（todo.md §3.4）：
//! - 缺页加载持页排他闩跨 await（上游 HandlePageFault 语义）；
//! - 空间回收禁止忙等 —— 单线程每核下直接同步驱逐或 async 让出；
//! - 驱逐持共享闩跨 I/O；WAL 预留由 ldb_wal 自行 await；
//! - page-provider 线程池 → `run_page_provider_task`（compio spawn 的驱逐循环）。

#[cfg(test)]
use std::cell::Cell;
use std::{
  cell::RefCell,
  fmt,
  future::Future,
  io,
  mem::{offset_of, take},
  ptr,
  rc::Rc,
  sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering, fence},
  },
  time::Duration,
};

use compio::time::sleep;
use futures_lite::future::yield_now;
use ldb_clock::ResidentPageSet;
use ldb_io::MemBackend;
use ldb_page::hook::{EvictHook, TxnLogHook};
use ldb_space::{NoTxnLog, SpaceManager};
use ldb_stat::StatCollector;
use ldb_sync::PageStateExt;
use ldb_type::{
  BlockId, GsnWatermark, LogId, PAGE_SIZE, PageId, PageState, PageStateMode, VERSION_MASK,
  next_version_new_state, same_version_new_state,
};
use ldb_vmem::VirtualMemoryArena;

use crate::error::{Error, Result};

/// 页 I/O 后端（compio 文件 / 内存后端统一；RPITIT 捕获全部生命期）。
///
/// 全部 `&self`：compio 单线程每核下多协程并发缺页时，后端不得独占借用
/// 跨 await（`RefCell::borrow_mut` 挂起期间重入即 panic）；`CompioFileBackend`
/// 内部共享 fd、`MemBackend` 内部 `RefCell` 短借，均天然满足。
pub trait PageBackend {
  /// 读一个 4KiB 页
  fn read_page(
    &self,
    pid: PageId,
    dst: &mut [u8; PAGE_SIZE],
  ) -> impl Future<Output = io::Result<()>>;
  /// 写一个 4KiB 页
  fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> impl Future<Output = io::Result<()>>;
  /// fdatasync
  fn sync(&self) -> impl Future<Output = io::Result<()>>;
}

impl PageBackend for MemBackend {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
    // 直接读入调用方页帧：原版「栈缓冲 → dst」多一次 4KiB 拷贝
    self.read_at_page(pid, dst).await.map(|_| ())
  }

  async fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> io::Result<()> {
    self.write_at_page(pid, src).await
  }

  async fn sync(&self) -> io::Result<()> {
    Ok(())
  }
}

/// 共享后端：`Arc<T>` 透传（后端方法全部 `&self`，测试 / Facade 共享同一实例）
impl<T: PageBackend> PageBackend for Arc<T> {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
    (**self).read_page(pid, dst).await
  }

  async fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> io::Result<()> {
    (**self).write_page(pid, src).await
  }

  async fn sync(&self) -> io::Result<()> {
    (**self).sync().await
  }
}

/// `Rc<T>` 透传（compio 单线程每核下的 !Send 后端 —— 故障注入 / 内存后端）
impl<T: PageBackend> PageBackend for Rc<T> {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
    (**self).read_page(pid, dst).await
  }

  async fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> io::Result<()> {
    (**self).write_page(pid, src).await
  }

  async fn sync(&self) -> io::Result<()> {
    (**self).sync().await
  }
}

/// 缓冲页帧标志（上游 BufferFrame 语义的最小集）
#[derive(Debug, Clone, Copy, Default)]
pub struct BufferFrame {
  /// 帧对应的页号
  pub pid: PageId,
  /// 是否 extent 大页槽
  pub extent: bool,
  /// extent 页数
  pub extent_pages: u32,
  /// 待写标记
  pub to_write: bool,
  /// 脏页
  pub dirty: bool,
}

impl BufferFrame {
  /// 构造
  pub fn init(pid: PageId) -> Self {
    Self {
      pid,
      ..Self::default()
    }
  }

  /// 重置
  pub fn reset(&mut self, pid: PageId) {
    *self = Self::init(pid);
  }

  /// 置标志
  pub fn set_flag(&mut self, dirty: bool, to_write: bool) {
    self.dirty = dirty;
    self.to_write = to_write;
  }

  /// 是否 extent
  #[inline]
  pub fn is_extent(&self) -> bool {
    self.extent
  }

  /// 转待写 extent 描述
  pub fn to_write_extent(&mut self, pages: u32) -> (PageId, u32) {
    self.extent = true;
    self.extent_pages = pages;
    self.to_write = true;
    (self.pid, pages)
  }
}

/// 脏计数分片数（按脏位图字分桶 —— 位翻转才动分片计数，消除单缓存行争用）
const DIRTY_SHARDS: usize = 64;

/// Clock 单批扫描槽位数（上游 `FLAGS_bm_evict_batch_size` 默认 128，
/// `ZLeanStore/src/config.cc:29`；采样窗口与上游一致）
const EVICT_BATCH: u64 = 128;

/// 共享固定守卫（零拷贝页视图；Drop 减 Shared 计数）
pub struct ShareGuard<'a, B: PageBackend, T: TxnLogHook = NoTxnLog> {
  /// 所属缓冲池
  mgr: &'a BufferManager<B, T>,
  /// 固定的页号
  pid: PageId,
}

impl<B: PageBackend, T: TxnLogHook> fmt::Debug for ShareGuard<'_, B, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ShareGuard")
      .field("pid", &self.pid)
      .finish()
  }
}

impl<B: PageBackend, T: TxnLogHook> ShareGuard<'_, B, T> {
  /// 常驻页帧零拷贝只读视图（4KiB）。
  ///
  /// 持有期间写者无法进入 EXCLUSIVE、驱逐者无法过 MARKED→EVICTED 的 CAS，
  /// 视图严格无竞争；`&self` 借用保证视图生命周期不超过 guard。
  #[inline]
  pub fn bytes(&self) -> &[u8; PAGE_SIZE] {
    // SAFETY: Shared 计数 >= 1（构造即持有，Drop 才减），排除 EXCLUSIVE 写者
    // 与驱逐（advise_dontneed 前必经 EVICTED 转换，而 Shared→EVICTED 无路径）；
    // 返回引用绑定 `&self`，同一 guard 不会借出 `&mut`，其它 guard 只读。
    unsafe { &*(self.mgr.page_ptr(self.pid) as *const [u8; PAGE_SIZE]) }
  }

  /// 固定的页号
  #[inline]
  pub fn pid(&self) -> PageId {
    self.pid
  }
}

impl<B: PageBackend, T: TxnLogHook> Drop for ShareGuard<'_, B, T> {
  fn drop(&mut self) {
    self.mgr.unfix_share(self.pid);
  }
}

/// 页状态只读视图（外部观察用；状态字写入只允许 crate 内闩协议）。
///
/// 上游 `GetPageState` 返回可写引用是 C++ 语义；在 Rust 里从安全 API 暴露
/// `&PageState` 会让外部代码用 `store/cas` 任意改写闩状态（可构造
/// `ShareGuard` 视图存活期间页被重新缺页写回的别名 UB）。故公开入口只给
/// 读视图，写入口全部收在 `pub(crate)` 的 [`BufferManager::page_state_raw`]。
#[derive(Debug, Clone, Copy)]
pub struct PageStateView<'a> {
  /// 只读状态字
  state: &'a PageState,
}

impl PageStateView<'_> {
  /// 当前锁状态
  #[inline]
  pub fn lock_state(&self) -> PageStateMode {
    self.state.lock_state()
  }

  /// 当前版本号（低 56 位）
  #[inline]
  pub fn version(&self) -> u64 {
    self.state.version()
  }
}

/// 读取页帧头部的 `p_gsn`（WAL-before-page 门用）。
///
/// 偏移取自 [`ldb_page::PageHeader`] 的编译期 `offset_of!`（页头布局单一来源，
/// 不硬编码常量）；页头按内存原生序读写，故用 `from_ne_bytes`。
#[inline]
fn read_p_gsn(frame: &[u8; PAGE_SIZE]) -> LogId {
  const OFF: usize = offset_of!(ldb_page::PageHeader, p_gsn);
  let mut b = [0u8; 8];
  b.copy_from_slice(&frame[OFF..OFF + 8]);
  u64::from_ne_bytes(b)
}

/// EXCLUSIVE 闩下的页帧独占视图（crate 内部，零成本）。
///
/// 与 [`ShareGuard`] 对称：不可 `Copy`/`Clone`，`bytes_mut` 借用 `&mut self`，
/// 同一 guard 不能产生第二个可变借用；guard 存活期间页状态为 EXCLUSIVE，
/// 其它协程的 `fix_*` 无法进入。等价于 `UnsafeCell<[u8; PAGE_SIZE]>` + 锁守卫
/// 的类型级建模（`UnsafeCell::get` 同样只给 `&mut self` 借出的裸指针）。
///
/// 构造只允许在成功取得 EXCLUSIVE 之后（[`Self::assume_locked`] 私有且
/// `unsafe`）；Drop 默认按「正常解锁」处理（版本 +1 → UNLOCKED），
/// 转 EVICTED / 失败回滚路径必须先调用对应 `done` 方法消费。
struct PageWriteGuard<'a, B: PageBackend, T: TxnLogHook = NoTxnLog> {
  /// 所属缓冲池
  mgr: &'a BufferManager<B, T>,
  /// 固定的页号
  pid: PageId,
  /// 已显式结束（Drop 不再重复解锁）
  done: bool,
}

impl<'a, B: PageBackend, T: TxnLogHook> PageWriteGuard<'a, B, T> {
  /// 调用方必须已持该页 EXCLUSIVE 闩
  ///
  /// # Safety
  /// 仅可在「本协程成功 CAS/抢占 EXCLUSIVE 且尚未释放」时调用；guard 存续
  /// 期间不得用其它路径（`page_state_raw` 的 store/cas、`unfix_exclusive`）
  /// 改写该页状态字。
  #[inline]
  unsafe fn assume_locked(mgr: &'a BufferManager<B, T>, pid: PageId) -> Self {
    debug_assert_eq!(
      mgr.page_state_raw(pid).lock_state(),
      PageStateMode::Exclusive,
      "PageWriteGuard 必须以 EXCLUSIVE 状态构造"
    );
    Self {
      mgr,
      pid,
      done: false,
    }
  }

  /// 独占只读视图
  #[inline]
  fn bytes(&self) -> &[u8; PAGE_SIZE] {
    // SAFETY: guard 以 EXCLUSIVE 构造；EXCLUSIVE 期间没有任何其它读者/写者
    // （`fix_shared`/`fix_share_impl`/`fix_exclusive` 都不会进入）。
    unsafe { &*(self.mgr.page_ptr(self.pid) as *const [u8; PAGE_SIZE]) }
  }

  /// 独占可变视图（借用 `&mut self`，同一时刻只能有一个）
  #[inline]
  fn bytes_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
    // SAFETY: 同 `bytes`；`&mut self` 保证本 guard 不会同时借出第二个 `&mut`。
    unsafe { &mut *(self.mgr.page_ptr(self.pid) as *mut [u8; PAGE_SIZE]) }
  }

  /// 正常解锁（版本 +1 → UNLOCKED）
  #[inline]
  fn unlock(mut self) {
    self.done = true;
    self.mgr.unfix_exclusive(self.pid);
  }

  /// 写失败回滚 UNLOCKED（保留版本；页未被修改，读者无需 Restart）
  #[inline]
  fn restore_unlocked(mut self) {
    self.done = true;
    let ps = self.mgr.page_state_raw(self.pid);
    let cur = ps.load(Ordering::Relaxed);
    ps.store(
      same_version_new_state(cur, PageStateMode::UNLOCKED),
      Ordering::Release,
    );
  }

  /// 缺页失败回滚 EVICTED（保留版本，等待下次重试）
  #[inline]
  fn rollback_evicted(mut self) {
    self.done = true;
    let ps = self.mgr.page_state_raw(self.pid);
    let cur = ps.load(Ordering::Relaxed);
    ps.store(
      same_version_new_state(cur, PageStateMode::EVICTED),
      Ordering::Release,
    );
  }

  /// 持锁驱逐（版本 +1 → EVICTED；常驻表/帧计数由调用方处理）
  #[inline]
  fn into_evicted(mut self) {
    self.done = true;
    self
      .mgr
      .page_state_raw(self.pid)
      .unlock_exclusive_and_evict();
  }
}

impl<B: PageBackend, T: TxnLogHook> Drop for PageWriteGuard<'_, B, T> {
  fn drop(&mut self) {
    if !self.done {
      self.mgr.unfix_exclusive(self.pid);
    }
  }
}

/// VMCache 缓冲池。
///
/// `T: TxnLogHook` 透传给 `SpaceManager<T>`（热路径映射变更写 WAL，
/// 泛型静态分发）；默认 [`NoTxnLog`]。
pub struct BufferManager<B: PageBackend, T: TxnLogHook = NoTxnLog> {
  /// 虚拟地址 arena
  ///
  /// 不公开：`VirtualMemoryArena::{decommit_pages, advise_dontneed}` 是安全
  /// API，外部在 `ShareGuard` 视图存活期间调用会让引用指向已释放/未提交页。
  /// 只读地址观测走 [`Self::page_addr`]。
  pub(crate) arena: ldb_vmem::VirtualMemoryArena,
  /// 页状态表（pid → PageState）
  ///
  /// 不公开：`PageState::store/cas` 是公开方法，暴露 `&PageState` 会让安全
  /// 外部代码绕过闩协议改写状态字（可构造页帧别名 UB）。外部只读走
  /// [`Self::get_page_state`] 的 [`PageStateView`]。
  pub(crate) page_states: Arc<[PageState]>,
  /// 常驻集合（Clock）
  pub resident: ResidentPageSet,
  /// 最大页号
  max_pid: AtomicU64,
  /// 物理帧上限
  pub physical_cnt: u64,
  /// 已用物理帧
  pub physical_used: AtomicU64,
  /// 空间管理器（`Rc` —— compio 单线程每核，TxnLogHook 载线程局部 LogManager）
  pub space: Rc<SpaceManager<T>>,
  /// 页 I/O 后端（&self 并发安全，见 `PageBackend`）
  pub backend: B,
  /// 空闲空间管理
  pub free_mgr: Arc<ldb_free::FreeStorageManager>,
  /// 运行标记
  pub is_running: Arc<AtomicBool>,
  /// 脏位图（每 pid 1 bit；替代"标脏即 +1"的粗计数 —— 驱逐无条件 -1 曾致下溢）
  dirty_bits: Box<[AtomicU64]>,
  /// 已释放位图（每 pid 1 bit）：dealloc 幂等 + 拒绝已释放页的缺页读回
  /// （上游 `HandlePageFault` 的 `PIDIsDeallocated` 门；无此门时 stale pid
  /// 会把丢弃前的旧盘面内容读回，或与复用后的新逻辑页混用）
  deallocated_bits: Box<[AtomicU64]>,
  /// 脏页计数分片（求和得总数；替代原全局单 AtomicU64 的缓存行热点）
  dirty_shards: Box<[AtomicU64]>,
  /// 全局最小已刷 GSN 水位（与 `LogManager` 共享同一实例 —— D5 消除双轨：
  /// 原版私有 `wal_checkpoint_offset` 与 WAL 落盘水位脱钩）
  pub min_gsn_flushed: Arc<GsnWatermark>,
  /// 检查点已推进偏移（fuzzy checkpoint 高水位）
  checkpoint_offset: AtomicU64,
  /// 统计收集器（L5 注入；默认空实例 —— 未被注入时计数丢弃，无热路径分支）
  stats: Arc<StatCollector>,
  /// 待处理释放队列的页数（alloc 路径据此按需调用
  /// `SpaceManager::process_dealloc_list` —— 原版只入队不消费，复用池永远为空）
  dealloc_pending: AtomicU64,
  /// 测试专用：在快照拷贝中途确定性注入「写者改数据/改状态」（见
  /// `copy_page_out`；`cfg(test)` 下才存在，release 零字段/零分支）
  #[cfg(test)]
  mid_copy_inject: Cell<MidCopyInject>,
}

/// 快照拷贝中途的确定性写者注入模式（仅测试构建）
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum MidCopyInject {
  /// 无注入（正常路径）
  #[default]
  None,
  /// 写者：改前 64B 数据 + 版本 +1 回 UNLOCKED
  WriteDataAndVersion,
  /// 写者：只把状态改成同版本 EXCLUSIVE（不改数据）
  StateExclusiveOnly,
  /// 驱逐者：版本 +1 → EVICTED
  StateEvicted,
}

impl<B: PageBackend, T: TxnLogHook> fmt::Debug for BufferManager<B, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BufferManager")
      .field("max_pid", &self.max_pid.load(Ordering::Relaxed))
      .field("physical_used", &self.physical_used.load(Ordering::Relaxed))
      .field("physical_cnt", &self.physical_cnt)
      .field("dirty_cnt", &self.dirty_cnt())
      .finish_non_exhaustive()
  }
}

impl<B: PageBackend, T: TxnLogHook> BufferManager<B, T> {
  /// 构造（上游 Construction 修复版：单一 arena、无 2× 预留，C-9 修复）。
  ///
  /// `virtual_cnt` 为虚拟页数（arena 容量），`physical_frames` 为常驻帧上限
  /// （VMCache 核心：虚拟 >> 物理，驱逐回收帧）；`min_gsn_flushed` 须与
  /// `LogManager` 共享同一 `GsnWatermark` 实例（D5）
  pub fn new(
    virtual_cnt: u64,
    physical_frames: u64,
    space: Rc<SpaceManager<T>>,
    free_mgr: Arc<ldb_free::FreeStorageManager>,
    backend: B,
    is_running: Arc<AtomicBool>,
    min_gsn_flushed: Arc<GsnWatermark>,
  ) -> Result<Self> {
    // 构造期一次性校验（冷路径）：0 帧或虚拟容量溢出都是配置错误，
    // 不能留到首次缺页才以 arena 长度不足的形式暴露。
    if physical_frames == 0 {
      return Err(Error::InvalidConfig("physical_frames 必须 > 0".into()));
    }
    if physical_frames > virtual_cnt {
      return Err(Error::InvalidConfig(format!(
        "physical_frames({physical_frames}) 不得超过 virtual_cnt({virtual_cnt})"
      )));
    }
    let bytes = usize::try_from(virtual_cnt)
      .ok()
      .and_then(|n| n.checked_mul(PAGE_SIZE))
      .ok_or_else(|| Error::InvalidConfig("virtual_cnt * PAGE_SIZE 溢出".into()))?;
    let arena = VirtualMemoryArena::new_arena(bytes)?;
    // 初始态必须 EVICTED（VMCache 语义：虚拟槽由磁盘 backing，首访问缺页加载）。
    // 修复（崩溃重开测试发现）：原版 PageState::new() 为 UNLOCKED —— fix_share
    // 对未加载页跳过 handle_page_fault 直接读 arena 零页 = 静默数据丢失
    let states: Vec<PageState> = (0..virtual_cnt)
      .map(|_| {
        let s = PageState::new();
        s.init();
        s
      })
      .collect();
    let arc_states: Arc<[PageState]> = states.into();
    let resident = ResidentPageSet::new(virtual_cnt, arc_states.clone(), 1);
    let dirty_words = (virtual_cnt as usize).div_ceil(u64::BITS as usize);
    Ok(Self {
      arena,
      page_states: arc_states,
      resident,
      max_pid: AtomicU64::new(0),
      physical_cnt: physical_frames,
      physical_used: AtomicU64::new(0),
      space,
      backend,
      free_mgr,
      is_running,
      dirty_bits: (0..dirty_words).map(|_| AtomicU64::new(0)).collect(),
      deallocated_bits: (0..dirty_words).map(|_| AtomicU64::new(0)).collect(),
      dirty_shards: (0..DIRTY_SHARDS).map(|_| AtomicU64::new(0)).collect(),
      min_gsn_flushed,
      checkpoint_offset: AtomicU64::new(0),
      stats: Arc::new(StatCollector::default()),
      dealloc_pending: AtomicU64::new(0),
      #[cfg(test)]
      mid_copy_inject: Cell::new(MidCopyInject::None),
    })
  }

  /// 注入统计收集器（L5 装配；缺省为空实例，计数被丢弃）。
  ///
  /// 与 [`Self::new`] 分离以保持既有构造签名不变（消费者零改动）；须在
  /// `Rc` 化之前调用（`&mut self`）。
  pub fn set_stats(&mut self, stats: Arc<StatCollector>) {
    self.stats = stats;
  }

  /// 页号边界校验（映射表可能来自不可信磁盘数据；越界必须返回错误而非索引 panic）
  #[inline]
  fn check_pid(&self, pid: PageId) -> Result<()> {
    if pid as usize >= self.page_states.len() {
      return Err(Error::InvalidPage(pid));
    }
    Ok(())
  }

  /// 页状态只读视图
  ///
  /// # Panics
  /// `pid` 必须小于虚拟页数（本 crate 分配路径与 `check_pid` 保证）。
  #[inline]
  pub fn get_page_state(&self, pid: PageId) -> PageStateView<'_> {
    PageStateView {
      state: self.page_state_raw(pid),
    }
  }

  /// 页状态原始引用（crate 内闩协议专用 —— 外部可见的 `store/cas` 会破坏
  /// 状态机不变量，禁止外泄）
  #[inline]
  pub(crate) fn page_state_raw(&self, pid: PageId) -> &PageState {
    &self.page_states[pid as usize]
  }

  /// 当前最大页号
  #[inline]
  pub fn get_max_pid(&self) -> u64 {
    self.max_pid.load(Ordering::Acquire)
  }

  /// WAL 信息（检查点偏移视图）
  #[inline]
  pub fn get_wal_info(&self) -> u64 {
    self.checkpoint_offset.load(Ordering::Acquire)
  }

  /// 空闲管理器引用
  #[inline]
  pub fn get_free_page_manager(&self) -> &Arc<ldb_free::FreeStorageManager> {
    &self.free_mgr
  }

  /// 页在 arena 内的地址（诊断/测试观测用）
  ///
  /// 只返回地址数值，不授予任何内存访问能力（解引用仍需 `unsafe`）；
  /// 避免为了测试暴露整个 [`ldb_vmem::VirtualMemoryArena`]（其
  /// `decommit_pages` 会破坏存活视图）。
  #[inline]
  pub fn page_addr(&self, pid: PageId) -> usize {
    self.arena.to_ptr(pid) as usize
  }

  /// 页指针（arena 内）
  ///
  /// # Safety
  /// 别名由页闩协议约束（EXCLUSIVE 独占 / SHARED 共享）；裸指针不携带
  /// 生命周期，只能交给 [`PageWriteGuard`]/[`ShareGuard`] 这类有状态的视图
  /// 使用，不得在安全代码里直接解引用。
  #[inline]
  pub(crate) unsafe fn page_ptr(&self, pid: PageId) -> *mut u8 {
    self.arena.to_ptr(pid)
  }

  /// 页快照拷出（arena → 调用方缓冲）
  ///
  /// 调用方已通过页状态校验（快照时刻非 EXCLUSIVE/EVICTED）；单线程每核的
  /// compio 模型下拷贝段内无其它协程可运行，跨线程别名由 `BufferManager: !Sync`
  /// 排除。拷贝后的版本漂移由 [`Self::validate_shared_snapshot`] 兜底。
  fn copy_page_out(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) {
    // SAFETY: 见函数文档；快照读取不写入页帧
    let src = unsafe { self.page_ptr(pid) };
    #[cfg(not(test))]
    // SAFETY: 源为 arena 页帧、目标为调用方缓冲，两段不重叠；无并发写者
    unsafe {
      ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), PAGE_SIZE);
    }
    #[cfg(test)]
    {
      // 测试注入：把 4KiB 拷贝拆成两半，在中间以同线程程序序确定性执行
      // 「写者改数据/状态」「驱逐者改状态」，验证「读状态 → 拷贝 → fence →
      // 校验完整状态字」必须 Restart。默认 None 时与生产路径逐字节一致。
      let half = PAGE_SIZE / 2;
      // SAFETY: 同生产路径；两段互不重叠
      unsafe {
        ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), half);
      }
      self.test_mid_copy_inject(pid);
      // SAFETY: 同上
      unsafe {
        ptr::copy_nonoverlapping(src.add(half), dst.as_mut_ptr().add(half), PAGE_SIZE - half);
      }
    }
  }

  /// 快照拷贝中途的确定性注入（仅测试构建）
  #[cfg(test)]
  fn test_mid_copy_inject(&self, pid: PageId) {
    match self.mid_copy_inject.replace(MidCopyInject::None) {
      MidCopyInject::None => {}
      MidCopyInject::WriteDataAndVersion => {
        let ps = self.page_state_raw(pid);
        let cur = ps.load(Ordering::Relaxed);
        ps.store(
          same_version_new_state(cur, PageStateMode::EXCLUSIVE),
          Ordering::Release,
        );
        // 写后半段：读者已经把前半段拷走、尚未读后半段 ⇒ 注入后快照首尾
        // 不一致（撕裂），校验必须 Restart 丢弃。单线程程序序，无并发访问。
        // SAFETY: arena 页帧在 BufferManager 存续期内有效，注入点无并发
        unsafe {
          ptr::write_bytes(self.page_ptr(pid).add(PAGE_SIZE / 2), 0xAB, 64);
        }
        let cur = ps.load(Ordering::Relaxed);
        ps.store(
          next_version_new_state(cur, PageStateMode::UNLOCKED),
          Ordering::Release,
        );
      }
      MidCopyInject::StateExclusiveOnly => {
        let ps = self.page_state_raw(pid);
        let cur = ps.load(Ordering::Relaxed);
        ps.store(
          same_version_new_state(cur, PageStateMode::EXCLUSIVE),
          Ordering::Release,
        );
      }
      MidCopyInject::StateEvicted => {
        let ps = self.page_state_raw(pid);
        let cur = ps.load(Ordering::Relaxed);
        ps.store(
          next_version_new_state(cur, PageStateMode::EVICTED),
          Ordering::Release,
        );
      }
    }
  }

  /// pid 的位图槽（脏位/已释放位共用：字下标 + 位掩码）
  #[inline]
  fn pid_bit_slot(&self, pid: PageId) -> (usize, u64) {
    let p = pid as usize;
    (p / u64::BITS as usize, 1u64 << (p % u64::BITS as usize))
  }

  /// pid 的脏计数分片下标（按位图字散列 —— 相邻 pid 落不同分片）
  #[inline]
  fn dirty_shard_idx(&self, pid: PageId) -> usize {
    ((pid as usize / u64::BITS as usize) % DIRTY_SHARDS).min(self.dirty_shards.len() - 1)
  }

  /// 置脏位（0→1 翻转才递增分片计数；重复标脏幂等）
  fn set_dirty_bit(&self, pid: PageId) {
    let (w, b) = self.pid_bit_slot(pid);
    if self.dirty_bits[w].fetch_or(b, Ordering::Release) & b == 0 {
      self.dirty_shards[self.dirty_shard_idx(pid)].fetch_add(1, Ordering::Relaxed);
    }
  }

  /// 清脏位（1→0 翻转才递减分片计数；返回是否真清了）
  fn clear_dirty_bit(&self, pid: PageId) -> bool {
    let (w, b) = self.pid_bit_slot(pid);
    if self.dirty_bits[w].fetch_and(!b, Ordering::Release) & b != 0 {
      self.dirty_shards[self.dirty_shard_idx(pid)].fetch_sub(1, Ordering::Relaxed);
      true
    } else {
      false
    }
  }

  /// 脏页总数（分片求和）
  #[inline]
  pub fn dirty_cnt(&self) -> u64 {
    self
      .dirty_shards
      .iter()
      .map(|s| s.load(Ordering::Relaxed))
      .sum()
  }

  /// 页是否脏（精确位图；驱逐 / flush 判定用）
  ///
  /// # Panics
  /// `pid` 必须小于虚拟页数。
  #[inline]
  pub fn page_is_dirty(&self, pid: PageId) -> bool {
    let (w, b) = self.pid_bit_slot(pid);
    self.dirty_bits[w].load(Ordering::Acquire) & b != 0
  }

  /// 标脏（幂等；位翻转才动计数）
  ///
  /// # Panics
  /// `pid` 必须小于虚拟页数。
  pub fn make_page_dirty(&self, pid: PageId) {
    self.set_dirty_bit(pid);
  }

  /// 标净（幂等）
  ///
  /// # Panics
  /// `pid` 必须小于虚拟页数。
  pub fn make_page_clean(&self, pid: PageId) {
    let _ = self.clear_dirty_bit(pid);
  }

  /// 页是否已释放（dealloc 后、复用前）
  #[inline]
  fn is_deallocated(&self, pid: PageId) -> bool {
    let (w, b) = self.pid_bit_slot(pid);
    self.deallocated_bits[w].load(Ordering::Acquire) & b != 0
  }

  /// 标记已释放（0→1 返回 true；重复释放返回 false，用于幂等）
  fn mark_deallocated(&self, pid: PageId) -> bool {
    let (w, b) = self.pid_bit_slot(pid);
    self.deallocated_bits[w].fetch_or(b, Ordering::AcqRel) & b == 0
  }

  /// 清除已释放标记（复用时调用）
  fn clear_deallocated(&self, pid: PageId) {
    let (w, b) = self.pid_bit_slot(pid);
    let _ = self.deallocated_bits[w].fetch_and(!b, Ordering::AcqRel);
  }

  /// 是否 extent 槽
  #[inline]
  pub fn is_extent(&self, _pid: PageId) -> bool {
    false
  }

  /// 帧视图（供 ledger buffer_frame 查询）
  #[inline]
  pub fn buffer_frame(&self, pid: PageId) -> BufferFrame {
    BufferFrame::init(pid)
  }

  // ---- fix 族（OLC 乐观锁耦合）----

  /// 乐观读页（拷出快照；版本漂移返回 Restart，上层重试）
  pub async fn fix_share(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<u64> {
    self.fix_share_impl(pid, dst).await
  }

  /// fix_share 内核（上游 `FixShareImpl`；缺页后版本前移 → 重取快照重试）
  ///
  /// OLC 顺序固定为「读状态 → 拷贝非原子页帧 → `fence(Acquire)` → 再校验」：
  /// 单个 Acquire load 只约束其后的访问，不能阻止其前的页数据读被重排到校验
  /// 之后（ldb_sync round12 在 Apple M1 实测 stale-accept，见 CROSS.md）。
  /// `EXCLUSIVE` 期间不拷贝 —— 写者就地修改同一帧，拷贝只会得到撕裂数据；
  /// 与上游一样让出协程等待解锁，而不是先拷贝再靠版本校验「事后补救」。
  /// 单线程每核 + `BufferManager: !Sync` 保证拷贝段内没有其它协程/线程访问，
  /// 因此这里是无数据竞争的确定性读；版本校验负责逻辑一致性（Restart）。
  pub async fn fix_share_impl(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<u64> {
    self.check_pid(pid)?;
    // 本调用是否触发过缺页（miss 已在 handle_page_fault 记账；命中只计未缺页者，
    // 与上游 `FixShare`（`buffer_manager.cc:2060-2068`：`FixShareImpl` 无需读盘即
    // `buffer_hit_cnt++`）一致，避免同一次访问既记 miss 又记 hit）
    let mut faulted = false;
    loop {
      let ps = self.page_state_raw(pid);
      let v0 = ps.load(Ordering::Acquire);
      match PageStateMode::from_state_word(v0) {
        PageStateMode::Evicted => {
          self.handle_page_fault(pid).await?;
          faulted = true;
          continue; // 版本已前移，重取快照
        }
        PageStateMode::Exclusive => {
          if !self.is_running.load(Ordering::Relaxed) {
            return Err(Error::ShuttingDown);
          }
          yield_now().await;
          continue;
        }
        // UNLOCKED / Shared(n) / MARKED：MARKED 语义上无写者，读后可被驱逐者
        // 抢占为 EXCLUSIVE，由拷出后的版本校验兜底。
        PageStateMode::Unlocked | PageStateMode::Shared(_) | PageStateMode::Marked => {}
      }
      self.copy_page_out(pid, dst);
      // 拷出后校验：写者/驱逐者插队则 Restart（上层重进乐观锁）
      let version = self.validate_shared_snapshot(pid, v0)?;
      if !faulted {
        self.stats.record_page_hit();
      }
      return Ok(version);
    }
  }

  /// 校验一份已拷出的乐观快照（`observed` 为拷贝前的完整状态字）。
  ///
  /// 接受条件与 `ldb_sync::OptimisticGuard::validate_or_restart` 逐状态对齐：
  /// - 状态字完全不变；
  /// - 版本不变且状态为 `UNLOCKED` / `Shared(n)`；
  /// - 版本不变且状态为 `MARKED` → `Unmark` 成功（页未写，可继续读）。
  ///
  /// 其余（`EXCLUSIVE` / `EVICTED` / 版本漂移）→ `Restart`。
  fn validate_shared_snapshot(&self, pid: PageId, observed: u64) -> Result<u64> {
    // LoadLoad 屏障必须在本次状态读取之前：Acquire load 不能阻止其前的
    // 页数据读被重排到校验之后（CROSS.md → ldb_sync round12 实测）。
    fence(Ordering::Acquire);
    let ps = self.page_state_raw(pid);
    let latest = ps.load(Ordering::Acquire);
    let version = observed & VERSION_MASK;
    if latest == observed {
      return Ok(version);
    }
    if (latest & VERSION_MASK) == version {
      match PageStateMode::from_state_word(latest) {
        PageStateMode::Unlocked | PageStateMode::Shared(_) => return Ok(version),
        PageStateMode::Marked => {
          let mut new_state = 0u64;
          if ps.unmark(latest, &mut new_state) {
            return Ok(version);
          }
        }
        PageStateMode::Exclusive | PageStateMode::Evicted => {}
      }
    }
    Err(Error::Restart)
  }

  /// 缓存命中路径（上游 `FixShareCachedPage` 的乐观快照版）
  ///
  /// 调用方持有此前 `fix_share` 返回的 `version`。协议同样是「读状态 → 拷贝
  /// → fence → 再校验」；只做拷贝前的一次校验会把「校验后写者插队」的撕裂页
  /// 当作有效数据返回。
  pub async fn fix_share_cached_page(
    &self,
    pid: PageId,
    dst: &mut [u8; PAGE_SIZE],
    version: u64,
  ) -> Result<u64> {
    self.check_pid(pid)?;
    let ps = self.page_state_raw(pid);
    let v0 = ps.load(Ordering::Acquire);
    if (v0 & VERSION_MASK) != version {
      return Err(Error::Restart);
    }
    match PageStateMode::from_state_word(v0) {
      PageStateMode::Unlocked | PageStateMode::Shared(_) | PageStateMode::Marked => {}
      PageStateMode::Exclusive | PageStateMode::Evicted => return Err(Error::Restart),
    }
    self.copy_page_out(pid, dst);
    match self.validate_shared_snapshot(pid, v0) {
      Ok(v) if v == version => Ok(v),
      _ => Err(Error::Restart),
    }
  }

  /// 零拷贝共享固定（上游 `FixShareImpl` 的 `TryLockShared` 语义）：
  /// 返回持有 Shared(n) 计数的页帧视图守卫，Drop 自动减计数。
  ///
  /// 与 `fix_share`（快照拷出 + 事后版本校验）的取舍：
  /// - 快照版在单线程每核模型下拷贝段无协程切换（`!Sync` 排除跨线程），
  ///   不存在数据竞争；版本校验只负责「拷贝前后状态一致」，否则 Restart；
  /// - 本版持真 Shared 计数，写者（fix_exclusive 只认 UNLOCKED/MARKED）
  ///   无法进入，`bytes()` 视图严格无竞争，且省 4KiB 拷贝；
  /// - MARKED 被读者顶替为 Shared(1) 后，驱逐者的 CAS（steal_marked）
  ///   必然失败 → 放弃该候选，闭环不丢数据。
  pub async fn fix_shared(&self, pid: PageId) -> Result<ShareGuard<'_, B, T>> {
    self.check_pid(pid)?;
    // 本调用是否触发过缺页（miss 已在 handle_page_fault 内记账；命中只计未缺页者，
    // 与上游 buffer_hit_cnt「页已在缓冲池、无缺页即取到 Guard」一致，避免同一次
    // 访问既记 miss 又记 hit）
    let mut faulted = false;
    loop {
      let ps = self.page_state_raw(pid);
      let v0 = ps.load(Ordering::Acquire);
      match PageStateMode::from_state_word(v0) {
        PageStateMode::Evicted => {
          self.handle_page_fault(pid).await?;
          faulted = true;
        }
        PageStateMode::Exclusive => {
          yield_now().await;
        }
        // UNLOCKED / Shared(n) / MARKED（顶替为单读者）→ 计数 +1
        PageStateMode::Unlocked | PageStateMode::Shared(_) | PageStateMode::Marked => {
          if ps.try_lock_shared(v0) {
            if !faulted {
              self.stats.record_page_hit();
            }
            return Ok(ShareGuard { mgr: self, pid });
          }
        }
      }
      if !self.is_running.load(Ordering::Relaxed) {
        return Err(Error::ShuttingDown);
      }
    }
  }

  /// 排他固定（写路径；页必须已常驻 —— 缺页由调用方先 handle_page_fault）
  pub async fn fix_exclusive(&self, pid: PageId) -> Result<()> {
    self.check_pid(pid)?;
    let ps = self.page_state_raw(pid);
    let mut faulted = false;
    loop {
      let cur = ps.load(Ordering::Acquire);
      let state = PageStateMode::from_state_word(cur);
      match state {
        PageStateMode::Unlocked | PageStateMode::Marked => {
          let next = same_version_new_state(cur, PageStateMode::EXCLUSIVE);
          if ps.cas(cur, next, Ordering::AcqRel).is_ok() {
            if !faulted {
              self.stats.record_page_hit();
            }
            return Ok(());
          }
        }
        PageStateMode::Evicted => {
          self.handle_page_fault(pid).await?;
          faulted = true;
        }
        PageStateMode::Shared(_) | PageStateMode::Exclusive => {
          yield_now().await;
        }
      }
      if !self.is_running.load(Ordering::Relaxed) {
        return Err(Error::ShuttingDown);
      }
    }
  }

  /// 解除排他（版本 +1 回 UNLOCKED）
  #[inline]
  pub fn unfix_exclusive(&self, pid: PageId) {
    let ps = self.page_state_raw(pid);
    let cur = ps.load(Ordering::Relaxed);
    debug_assert_eq!(
      PageStateMode::from_state_word(cur),
      PageStateMode::Exclusive,
      "unfix_exclusive 必须由持锁者调用"
    );
    ps.store(
      next_version_new_state(cur, PageStateMode::UNLOCKED),
      Ordering::Release,
    );
  }

  /// 解除共享（读者计数 -1；Shared(n)→Shared(n-1)，n=1 时回 UNLOCKED）
  pub fn unfix_share(&self, pid: PageId) {
    let ps = self.page_state_raw(pid);
    loop {
      let cur = ps.load(Ordering::Acquire);
      match PageStateMode::from_state_word(cur) {
        PageStateMode::Shared(n) => {
          // n 来自解码器，合法域 1..=MAX_SHARED；n=1 时 0 == UNLOCKED
          let next = n - 1;
          if ps
            .cas(cur, same_version_new_state(cur, next), Ordering::AcqRel)
            .is_ok()
          {
            return;
          }
        }
        // 非共享态：误调不得把状态字改成非法编码（上游此处 assert 共享）
        PageStateMode::Unlocked
        | PageStateMode::Exclusive
        | PageStateMode::Marked
        | PageStateMode::Evicted => {
          debug_assert!(false, "unfix_share 未持共享闩: {cur:#x}");
          return;
        }
      }
    }
  }

  /// 缺页加载（协议 1/10：容量预留先行 —— 自身页不参与驱逐；随后 EXCLUSIVE
  /// 持闩跨 await 读盘，登记常驻后回 UNLOCKED）。
  ///
  /// 磁盘读直接落入 arena 页帧（EXCLUSIVE 期间无他者访问），不再经栈上中转
  /// —— 原版「栈缓冲 → 4KiB 拷贝」使每次缺页多一次整页拷贝。
  ///
  /// Windows 注意：arena 初始只 `MEM_RESERVE`；写 reserved 槽位会触发
  /// 访问违例。首次触页前必须先 `commit_pages`（Unix 上是 no-op）。
  pub async fn handle_page_fault(&self, pid: PageId) -> Result<()> {
    self.check_pid(pid)?;
    // 已释放页不得缺页读回：否则 stale pid 会把丢弃前的旧盘面内容读回，
    // 或与复用后的新逻辑页混用（上游 HandlePageFault 的 PIDIsDeallocated 门）
    if self.is_deallocated(pid) {
      return Err(Error::InvalidPage(pid));
    }
    // 容量预留：不持有任何页闩，驱逐不会与本页交互
    self.ensure_free_pages().await?;
    let ps = self.page_state_raw(pid);
    // 抢 EXCLUSIVE（仅 EVICTED 态可进；他人加载中则让出等待，
    // 加载完成（UNLOCKED / 共享 / MARKED）即返回 —— 页已常驻）
    loop {
      let cur = ps.load(Ordering::Acquire);
      match PageStateMode::from_state_word(cur) {
        PageStateMode::Evicted => {
          let next = same_version_new_state(cur, PageStateMode::EXCLUSIVE);
          if ps.cas(cur, next, Ordering::AcqRel).is_ok() {
            // SAFETY: CAS 成功，本协程自此独占该页；所有退出路径消费 guard
            // （rollback_evicted / unlock），不会把页留在 EXCLUSIVE。
            let mut g = unsafe { PageWriteGuard::assume_locked(self, pid) };
            // 硬上限预留：ensure_free_pages 与本次 CAS 之间存在协程交错窗口，
            // 多路并发缺页都通过 98% 水位检查后会一起自增 physical_used。
            // 用 CAS 在读取前先占住一帧，保证任何时刻 physical_used <= physical_cnt。
            if let Err(e) = self.reserve_frame().await {
              g.rollback_evicted();
              return Err(e);
            }
            // Windows：reserved → committed；否则首次写入即 AV。
            if let Err(e) = self.arena.commit_pages(pid as usize * PAGE_SIZE, PAGE_SIZE) {
              self.physical_used.fetch_sub(1, Ordering::Release);
              g.rollback_evicted();
              return Err(e.into());
            }
            // 磁盘读（持 EXCLUSIVE 跨 await —— 协议 1）；读失败必须传播，
            // 否则零页被登记常驻 = 数据丢失。读缓冲即 arena 页帧本体。
            if let Err(e) = self.backend.read_page(pid, g.bytes_mut()).await {
              self.physical_used.fetch_sub(1, Ordering::Release);
              // 回滚到 EVICTED（同版本）：否则本页被钉死在 EXCLUSIVE，
              // 后续 fix_* 恒 Restart / 自旋到关停
              g.rollback_evicted();
              return Err(e.into());
            }
            // 登记常驻（EXCLUSIVE 态满足 Clock 断言）+ 回 UNLOCKED（版本 +1）
            if !self.resident.insert(pid) {
              self.physical_used.fetch_sub(1, Ordering::Release);
              g.rollback_evicted();
              return Err(Error::MemoryPressure);
            }
            g.unlock();
            self.stats.record_page_miss();
            return Ok(());
          }
        }
        // 他协程持闩加载中：让出待其完成（他协程负责登记常驻）
        PageStateMode::Exclusive => {}
        // 已常驻（UNLOCKED / 共享 / MARKED）：无需加载
        PageStateMode::Unlocked | PageStateMode::Marked | PageStateMode::Shared(_) => return Ok(()),
      }
      if !self.is_running.load(Ordering::Relaxed) {
        return Err(Error::ShuttingDown);
      }
      yield_now().await;
    }
  }

  /// 在物理帧上限内预留一帧（CAS 自增；满则驱逐后重试）。
  ///
  /// 预留发生在缺页读盘之前，且 `physical_used` 的检查与自增是同一个 CAS，
  /// 因此并发缺页不会把常驻帧数推过 `physical_cnt`（原版在读取完成后才
  /// `fetch_add`，多路缺页可同时通过水位检查后一起越界）。
  async fn reserve_frame(&self) -> Result<()> {
    loop {
      let used = self.physical_used.load(Ordering::Acquire);
      if used < self.physical_cnt {
        if self
          .physical_used
          .compare_exchange_weak(used, used + 1, Ordering::AcqRel, Ordering::Acquire)
          .is_ok()
        {
          return Ok(());
        }
        continue;
      }
      let evicted = self.evict(1).await?;
      if evicted == 0 {
        return Err(Error::MemoryPressure);
      }
      yield_now().await;
    }
  }

  /// GC 读路径缺页（上游 `HandlePageFaultGC`：读入后标脏 —— GC 搬移期间页
  /// 必须被重写到新块；映射失效由 `ldb_space` 的 GC 写路径负责）
  pub async fn handle_page_fault_gc(&self, pid: PageId) -> Result<()> {
    self.handle_page_fault(pid).await?;
    self.make_page_dirty(pid);
    Ok(())
  }

  /// 确保物理余量（协议 3/10：无 page-provider —— 单线程下直接同步驱逐）
  pub async fn ensure_free_pages(&self) -> Result<()> {
    // 上游 0.98 水位；超出即驱逐一批
    while self.physical_used.load(Ordering::Relaxed) * 100 >= self.physical_cnt * 98 {
      let evicted = self.evict(1).await?;
      if evicted == 0 {
        // 无可驱逐（全脏且写失败）—— 报内存压力
        return Err(Error::MemoryPressure);
      }
      yield_now().await;
    }
    Ok(())
  }

  /// Clock 批量驱逐（协议 4：两段式 —— 先全批 TryMark（二次机会窗口），
  /// 再对仍 MARKED 的候选写盘驱逐；对齐上游 Evict 的 mark/evict 两遍）。
  ///
  /// 单次调用扫过一轮表（`count / EVICT_BATCH` 批，每批 128 槽 —— 上游默认
  /// 批大小）；表小于一批时绕圈，clock 指针跨调用持久推进，长期公平性由
  /// 多轮调用合成
  pub async fn evict(&self, count: u64) -> Result<u64> {
    let mut evicted = 0u64;
    let rounds = (self.resident.count() / EVICT_BATCH).max(1);
    for _ in 0..rounds {
      if evicted >= count {
        break;
      }
      // 线程局部缓冲在同步回调内填充，借用在 await 前结束
      CANDIDATES.with(|c| {
        let mut buf = c.borrow_mut();
        buf.clear();
        self
          .resident
          .iterate_clock_batch(EVICT_BATCH, |pid| buf.push(pid));
      });
      let mut candidates = CANDIDATES.with(|c| take(&mut *c.borrow_mut()));
      // 阶段 1：只标记（读者/写者此刻仍可抢占 MARKED —— 二次机会）。
      // 原地 retain，避免 `filter().collect()` 每次 evict 分配一个新 Vec。
      candidates.retain(|&pid| self.try_mark_page(pid));
      // 阶段 2：写盘 + 校验性驱逐（被抢占者自然落选）
      let mut failure = None;
      for &pid in &candidates {
        if evicted >= count {
          break;
        }
        match self.evict_marked(pid).await {
          Ok(true) => evicted += 1,
          Ok(false) => {}
          Err(e) => {
            failure = Some(e);
            break;
          }
        }
      }
      // 归还缓冲容量（下一轮 clear 后复用），消除每轮驱逐的堆分配
      CANDIDATES.with(|c| *c.borrow_mut() = candidates);
      if let Some(e) = failure {
        return Err(e);
      }
    }
    Ok(evicted)
  }

  /// UNLOCKED → MARKED（Clock 第一阶段：只标记，不驱逐 —— 给读者/写者
  /// 一个批间隔的二次机会，对齐上游 Evict 的两段式）
  fn try_mark_page(&self, pid: PageId) -> bool {
    let ps = self.page_state_raw(pid);
    let cur = ps.load(Ordering::Acquire);
    if PageStateMode::from_state_word(cur) != PageStateMode::Unlocked {
      return false;
    }
    ps.cas(
      cur,
      same_version_new_state(cur, PageStateMode::MARKED),
      Ordering::AcqRel,
    )
    .is_ok()
  }

  /// MARKED → EXCLUSIVE 校验性抢占（上游 UnmarkAndLockExclusive）：
  /// 写者已把 MARKED 抢成 EXCLUSIVE、或读者顶替成 Shared 时 CAS 失败 → 放弃
  fn steal_marked(&self, pid: PageId) -> bool {
    let ps = self.page_state_raw(pid);
    let cur = ps.load(Ordering::Acquire);
    if PageStateMode::from_state_word(cur) != PageStateMode::Marked {
      return false;
    }
    ps.cas(
      cur,
      same_version_new_state(cur, PageStateMode::EXCLUSIVE),
      Ordering::AcqRel,
    )
    .is_ok()
  }

  /// 对已 MARKED 的页完成驱逐（上游 EvictCleanPages 主体）：
  /// MARKED→EXCLUSIVE 校验性抢占 → **持 EXCLUSIVE 写盘** → 常驻摘除 →
  /// EVICTED（版本 +1）→ 帧回收。
  ///
  /// 修复 A2：原版「先写盘后 steal」—— 写盘 `.await` 期间页仅 MARKED，
  /// 写者可合法 CAS 进 EXCLUSIVE 改写同一帧（`&[u8]` 与 `&mut [u8]` 并发，
  /// aliasing UB + 撕裂页）。现先抢占 EXCLUSIVE 再写盘（§3.4 协议 4：
  /// 持闩跨 I/O，写者/读者被挡在闩外）。
  async fn evict_marked(&self, pid: PageId) -> Result<bool> {
    if !self.steal_marked(pid) {
      return Ok(false); // MARKED 被写者/读者抢占：放弃，页保持常驻
    }
    // SAFETY: steal_marked 成功 = 本协程持 EXCLUSIVE；guard 的所有退出路径
    // 都会消费（restore_unlocked / into_evicted），不会重复解锁。
    let g = unsafe { PageWriteGuard::assume_locked(self, pid) };
    // 写回仅限脏页：上游 Evict 对干净候选直接 madvise 驱逐（不产生页写）。
    // 原版无条件写回，使每次驱逐干净页都多一次 4KiB 设备写。
    if self.page_is_dirty(pid) {
      if let Err(e) = self.backend.write_page(pid, g.bytes()).await {
        // 修复 A9：写失败回滚 UNLOCKED（同版本）—— 页仍可被再次标记驱逐 /
        // 被写者取走；原版滞留 MARKED 后 try_mark_page 不再接受，候选永久钉死
        g.restore_unlocked();
        return Err(e.into());
      }
      // 脏页落盘计入页驱逐 + 物理写字节（上游 evict_writes，字节；见 ldb_stat）
      self.stats.record_page_evict(PAGE_SIZE as u64);
    }
    if !self.resident.remove(pid) {
      // 常驻表无此页（并发已摘）：不重复扣帧，仅落 EVICTED
      g.into_evicted();
      return Ok(false);
    }
    g.into_evicted();
    let _ = self.clear_dirty_bit(pid);
    // 先归还帧再 madvise：madvise 失败不得让物理帧计数永久泄漏
    // （页已 EVICTED 且摘出常驻表，帧必须归还）
    self.physical_used.fetch_sub(1, Ordering::Relaxed);
    self
      .arena
      .advise_dontneed(pid as usize * PAGE_SIZE, PAGE_SIZE)?;
    Ok(true)
  }

  /// 清洁页驱逐（UNLOCKED→MARKED→写盘→EVICTED 一步式；批量两段式走
  /// `try_mark_page` + `evict_marked`）
  pub async fn evict_clean_pages(&self, pid: PageId) -> Result<bool> {
    self.check_pid(pid)?;
    // 上游 EvictCleanPages 跳过脏页（脏页交由 Evict 的写回路径处理）
    if self.page_is_dirty(pid) {
      return Ok(false);
    }
    if !self.try_mark_page(pid) {
      return Ok(false);
    }
    self.evict_marked(pid).await
  }

  /// 持 EXCLUSIVE 驱逐（调用方刚 `fix_exclusive` 成功；上游
  /// `UnlockExclusiveAndEvict` 语义 —— EvictDeallocPage / EvictExtent 路径）。
  ///
  /// 修复：原版 dealloc 后接 evict_clean_pages 只认 UNLOCKED，页被永久
  /// 钉死在 EXCLUSIVE（此后 fix_share 恒 Restart、fix_exclusive 自旋到关停）
  /// `write == false`：已释放页 / 丢弃路径不写回（上游 `EvictDeallocPage`
  /// 直接 madvise，页内容无需持久化）。
  async fn evict_locked(&self, pid: PageId, write: bool) -> Result<bool> {
    let ps = self.page_state_raw(pid);
    let cur = ps.load(Ordering::Acquire);
    if PageStateMode::from_state_word(cur) != PageStateMode::Exclusive {
      return Ok(false);
    }
    // SAFETY: 上一行确认 EXCLUSIVE；调用方（fix_exclusive / 抢占 CAS）把锁
    // 所有权移交本函数，guard 的全部退出路径都会消费。
    let g = unsafe { PageWriteGuard::assume_locked(self, pid) };
    if write && self.page_is_dirty(pid) {
      if let Err(e) = self.backend.write_page(pid, g.bytes()).await {
        g.unlock(); // 写失败不得把页钉死在 EXCLUSIVE（版本 +1 回 UNLOCKED）
        return Err(e.into());
      }
      self.stats.record_page_evict(PAGE_SIZE as u64);
    }
    let removed = self.resident.remove(pid);
    let _ = self.clear_dirty_bit(pid);
    g.into_evicted();
    // 仅常驻表真正摘除时才归还帧（状态与常驻表不一致时不得下溢）；
    // 在 madvise 之前归还，避免 madvise 失败造成帧计数泄漏
    if removed {
      self.physical_used.fetch_sub(1, Ordering::Relaxed);
    }
    self
      .arena
      .advise_dontneed(pid as usize * PAGE_SIZE, PAGE_SIZE)?;
    Ok(true)
  }

  /// extent 驱逐（上游 `EvictExtent`：只处理已常驻页；`TryLockExclusive`
  /// 失败即跳过，不为了驱逐而把 EVICTED 页从磁盘读进来再做无谓写回）
  pub async fn evict_extent(&self, start_pid: PageId, page_cnt: u64) -> Result<()> {
    for p in 0..page_cnt {
      let pid = start_pid + p;
      if pid >= self.get_max_pid() {
        continue;
      }
      let ps = self.page_state_raw(pid);
      let cur = ps.load(Ordering::Acquire);
      match PageStateMode::from_state_word(cur) {
        PageStateMode::Evicted => continue,
        PageStateMode::Unlocked | PageStateMode::Marked => {
          if ps
            .cas(
              cur,
              same_version_new_state(cur, PageStateMode::EXCLUSIVE),
              Ordering::AcqRel,
            )
            .is_ok()
          {
            let _ = self.evict_locked(pid, true).await?;
          }
        }
        // 被读者/写者占用：上游 TryLockExclusive 同样放弃本页
        PageStateMode::Exclusive | PageStateMode::Shared(_) => {}
      }
    }
    Ok(())
  }

  /// 已释放页驱逐（上游 `EvictDeallocPage`）：已 EVICTED 无需处理；
  /// 常驻页持 EXCLUSIVE 后直接丢弃，不写回。
  pub async fn evict_dealloc_page(&self, pid: PageId) -> Result<()> {
    self.check_pid(pid)?;
    if PageStateMode::from_state_word(self.page_state_raw(pid).load(Ordering::Acquire))
      == PageStateMode::Evicted
    {
      return Ok(());
    }
    self.fix_exclusive(pid).await?;
    let _ = self.evict_locked(pid, false).await?;
    Ok(())
  }

  // ---- 分配 / 释放 ----

  /// 认领复用池中的 pid。
  ///
  /// 复用池正常只含 `dealloc_page` 已驱逐的 EVICTED 槽位；但 `SpaceManager`
  /// 的 `add_reusable_pid` 是公开 API（`space` 字段亦公开），外部误把常驻 pid
  /// 塞入时，直接 `init()` 会破坏 resident/`physical_used` 不变量并与存活
  /// 视图构成别名 UB。统一先走 `evict_dealloc_page`（EVICTED 时空操作）。
  async fn claim_reusable_pid(&self, pid: PageId) -> Result<PageId> {
    self.evict_dealloc_page(pid).await?;
    self.clear_deallocated(pid); // 新化身：允许再次缺页
    self.page_state_raw(pid).init(); // EVICTED 起始
    Ok(pid)
  }

  /// 分配一个页（VMCache 语义：虚拟地址常驻，pid 即虚拟槽）。
  /// D2 接线：优先复用已释放页池（`dealloc_list` → `reusable_pids` 的消费方）
  pub async fn alloc_page(&self) -> Result<PageId> {
    if let Some(pid) = self.space.take_reusable_pid() {
      return self.claim_reusable_pid(pid).await;
    }
    // 上游 `ReuseDeallocatedSpace`：先把 dealloc 列表处理进复用池。
    // 原版只 take 不 process ⇒ 复用池永远为空、dealloc 列表只进不出，
    // 虚拟页号只能单调增长到 arena 容量。
    if self.dealloc_pending.swap(0, Ordering::AcqRel) > 0 {
      let _ = self.space.process_dealloc_list();
      if let Some(pid) = self.space.take_reusable_pid() {
        return self.claim_reusable_pid(pid).await;
      }
    }
    let pid = self.max_pid.fetch_add(1, Ordering::AcqRel);
    // 虚拟上限 = arena 容量（物理帧超-commit 由缺页驱逐回收）
    if pid as usize >= self.arena.len() / PAGE_SIZE {
      self.max_pid.fetch_sub(1, Ordering::AcqRel);
      return Err(Error::MemoryPressure);
    }
    self.page_state_raw(pid).init(); // EVICTED 起始
    Ok(pid)
  }

  /// 分配 extent（连续页组；Blob 大页）
  pub async fn alloc_extent(&self, page_cnt: u32) -> Result<PageId> {
    let start = self
      .max_pid
      .fetch_add(u64::from(page_cnt), Ordering::AcqRel);
    if (start + u64::from(page_cnt)) as usize > self.arena.len() / PAGE_SIZE {
      self
        .max_pid
        .fetch_sub(u64::from(page_cnt), Ordering::AcqRel);
      return Err(Error::MemoryPressure);
    }
    for p in 0..page_cnt {
      self.page_state_raw(start + u64::from(p)).init();
    }
    Ok(start)
  }

  /// 释放页
  ///
  /// 幂等：重复释放不会二次入复用池（否则两次 alloc 会拿到同一 pid）；
  /// 每次调用都会确保该 pid 的常驻副本被丢弃。
  pub async fn dealloc_page(&self, pid: PageId) -> Result<()> {
    if pid >= self.get_max_pid() {
      return Err(Error::InvalidPage(pid));
    }
    let first = self.mark_deallocated(pid);
    if let Err(e) = self.evict_dealloc_page(pid).await {
      if first {
        self.clear_deallocated(pid);
      }
      return Err(e);
    }
    if first {
      self.space.add_pid_to_dealloc_list(pid);
      self.dealloc_pending.fetch_add(1, Ordering::Release);
    }
    Ok(())
  }

  /// 复用已释放空间（上游 ReuseDeallocatedSpace：更新映射 + 状态复位）
  ///
  /// 复用前必须先把该 pid 的常驻页安全丢弃：直接 `init()` 会把状态改成
  /// EVICTED，但 resident 表仍持有 pid、`physical_used` 不扣减，后续缺页
  /// `resident.insert` 恒失败 ⇒ 该 pid 永久 MemoryPressure；若此时仍有
  /// `ShareGuard` 视图存活，还会与重新缺页的写入构成别名 UB。
  /// `evict_dealloc_page` 对 EVICTED 是空操作，对常驻页持 X（等读者退出）
  /// 后丢弃不写回 —— 已释放页内容无需持久化。
  pub async fn reuse_deallocated_space(&self, pid: PageId, off: u64, comp_sz: u16) -> Result<()> {
    self.evict_dealloc_page(pid).await?;
    self.clear_deallocated(pid); // 新化身：允许再次缺页
    self.space.update_pid2_offset(pid, off, comp_sz).await;
    self.page_state_raw(pid).init();
    Ok(())
  }

  /// 尝试复用 extent（上游 TryReuseExtent：区间校验后批量复用）
  pub async fn try_reuse_extent(&self, start_pid: PageId, page_cnt: u32, off: u64) -> Result<bool> {
    // 区间校验：偏移按页对齐、pid 区间在虚拟容量内、偏移加法不溢出。
    // 上游经 `FreeStorageManager::RequestFreeExtent` 校验空闲区间；本层无该
    // 管理器句柄，只能校验地址域（原版 `get_block_offset(0) > off` 恒 false，
    // 等于完全没有校验）。
    let need = u64::from(page_cnt) * PAGE_SIZE as u64;
    let Some(end_pid) = start_pid.checked_add(u64::from(page_cnt)) else {
      return Ok(false);
    };
    if !off.is_multiple_of(PAGE_SIZE as u64)
      || end_pid > self.page_states.len() as u64
      || off.checked_add(need).is_none()
    {
      return Ok(false);
    }
    for p in 0..u64::from(page_cnt) {
      self
        .reuse_deallocated_space(
          start_pid + p,
          off + p * PAGE_SIZE as u64,
          ldb_pack::UNCOMPRESSED,
        )
        .await?;
    }
    Ok(true)
  }

  // ---- 读 ----

  /// 读页（写缓冲命中 → 常驻读 → 磁盘）
  pub async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
    // 磁盘直读（经由映射表偏移的块布局由 space 管理；此处页号即磁盘页号）
    self.fix_share_impl(pid, dst).await.map(|_| ())
  }

  /// 读 PID（bool 形态，上游 ReadPID）
  pub async fn read_pid(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<bool> {
    if !self.space.page_is_valid(pid) {
      return Ok(false);
    }
    self.read_page(pid, dst).await.map(|_| true)
  }

  /// 批量 extent 读（Blob 大页）
  ///
  /// 逐页直接读入调用方缓冲（`split_first_chunk_mut` 给出 `&mut [u8; PAGE_SIZE]`），
  /// 原版「栈页 → dst」多一次整页拷贝。
  pub async fn read_extents(&self, start_pid: PageId, page_cnt: u32, dst: &mut [u8]) -> Result<()> {
    let need = (page_cnt as usize)
      .checked_mul(PAGE_SIZE)
      .ok_or_else(|| Error::InvalidConfig("extent 读取长度溢出".into()))?;
    if dst.len() < need {
      return Err(Error::InvalidConfig(format!(
        "extent 读取目标缓冲不足: {} < {need}",
        dst.len()
      )));
    }
    let mut rest = &mut dst[..need];
    for p in 0..u64::from(page_cnt) {
      let (page, tail) = rest
        .split_first_chunk_mut::<PAGE_SIZE>()
        .ok_or_else(|| Error::InvalidConfig("extent 读取缓冲分片失败".into()))?;
      self.read_page(start_pid + p, page).await?;
      rest = tail;
    }
    Ok(())
  }

  /// 就地更新页内容（上游 UpdatePID：持 EXCLUSIVE 写后解锁）
  pub async fn update_pid(&self, pid: PageId, payload: &[u8]) -> Result<bool> {
    if pid >= self.get_max_pid() {
      return Ok(false);
    }
    self.fix_exclusive(pid).await?;
    // SAFETY: fix_exclusive 返回 Ok 即本协程持 pid 的 EXCLUSIVE 闩
    let mut g = unsafe { PageWriteGuard::assume_locked(self, pid) };
    {
      // bytes_mut 借用 `&mut guard`（等价 UnsafeCell::get 的借用模型：
      // &mut self 期间独占页帧；不实现 DerefMut 是因为 ./sh/unused.js 把
      // trait impl 方法计为未引用，而 clippy::explicit_deref_methods 又禁止
      // 显式调用 —— 显式访问器是同一零成本模型）
      let buf: &mut [u8; PAGE_SIZE] = g.bytes_mut();
      let n = payload.len().min(PAGE_SIZE);
      buf[..n].copy_from_slice(&payload[..n]);
    }
    self.make_page_dirty(pid);
    g.unlock();
    Ok(true)
  }

  /// 修改页内容（上游 ModifyPageContent：写 + 标脏）
  pub async fn modify_page_content(&self, pid: PageId, payload: &[u8]) -> Result<()> {
    self.update_pid(pid, payload).await.map(|_| ())
  }

  /// EXCLUSIVE 闩下的就地页修改（闭包式可变页视图）。
  ///
  /// 语义：固定 `pid` 为 EXCLUSIVE → 把可变页字节交给闭包 → 闭包返回后**标脏并
  /// 解锁**。与 [`Self::update_pid`]（整页覆盖）互补：调用方在页内**就地编码**
  /// （如 B+Tree 节点槽位/空间账更新），无需先重建整页再拷贝 —— 消除「整页读改写」
  /// 的额外分配与拷贝，并让写路径成为 `ldb_btree` 阶段 2 的落点。
  ///
  /// 闩语义与 [`Self::update_pid`] 一致：闭包执行期间页状态为 `EXCLUSIVE`，其它
  /// 协程的 `fix_share`/`fix_shared`/`fix_exclusive` 都无法进入（[`PageWriteGuard`]
  /// 的生命周期即临界区）。
  ///
  /// # 失败/panic 安全
  /// `fix_exclusive`（含缺页加载）失败即返回 `Err`，不进入闭包；闭包 panic 时
  /// [`PageWriteGuard`] 的 Drop 兜底解锁（版本 +1 → `UNLOCKED`），页状态不会停在
  /// `EXCLUSIVE`。注意：闭包若在修改页字节**之后** panic，本次标脏尚未执行 ⇒ 该页
  /// 保持原脏态（可能不写回），调用方须把 panic 视作修改未定义（与 C++ 无 panic 的
  /// 上游 `UpdatePID` 一致；Rust 侧 panic 即中止该次操作）。
  pub async fn with_page_mut<R>(
    &self,
    pid: PageId,
    f: impl FnOnce(&mut [u8; PAGE_SIZE]) -> R,
  ) -> Result<R> {
    self.fix_exclusive(pid).await?;
    // SAFETY: fix_exclusive 返回 Ok 即本协程持 pid 的 EXCLUSIVE 闩。
    let mut g = unsafe { PageWriteGuard::assume_locked(self, pid) };
    let r = f(g.bytes_mut());
    self.make_page_dirty(pid);
    g.unlock();
    Ok(r)
  }

  // ---- 写 / 刷盘 ----

  /// 批量写页（上游 WritePages 分发：就地 / 异地由 comp_sz 哨兵决定 —— 本层
  /// 统一 in-place；异地装箱编排见 ldb_space::page_io）。
  ///
  /// 修复 A3：逐页持 Shared 闩写盘（`fix_exclusive` 只认 UNLOCKED/MARKED，
  /// 持 Shared 期间并发写者无法进入）—— 原版无闩直写帧，与并发写者构成
  /// 数据竞争；guard 借用页帧，不再伪造 `'static`（多余 unsafe）
  pub async fn write_pages(&self, pids: &[PageId]) -> Result<()> {
    for &pid in pids {
      let g = self.fix_shared(pid).await?;
      self.backend.write_page(pid, g.bytes()).await?;
      // 上游 WritePagesInPlace：写盘后 MakePageClean（闩内清位；写者必须
      // 等闩释放后再标脏，不会丢后续写入）
      let _ = self.clear_dirty_bit(pid);
      drop(g);
    }
    Ok(())
  }

  /// 就地写（上游 WritePagesInPlace）
  pub async fn write_pages_in_place(&self, pids: &[PageId]) -> Result<()> {
    self.write_pages(pids).await
  }

  /// 异地写（上游 WritePagesOutOfPlace：新块写入 + 映射表更新）
  pub async fn write_pages_out_of_place(&self, pids: &[PageId], new_block: BlockId) -> Result<()> {
    let base = self.space.get_block_offset(new_block);
    for (i, &pid) in pids.iter().enumerate() {
      let off = base + (i as u64) * PAGE_SIZE as u64;
      // 异地写的落盘位置是目标块偏移，不是页号的 in-place 偏移。
      // `PageBackend` 的页号参数是数据文件的平坦页索引：off / PAGE_SIZE。
      // 原版写 `write_page(pid, ...)` 把新版本写回旧位置，目标块从未收到数据。
      let dst_page = off / PAGE_SIZE as u64;
      let g = self.fix_shared(pid).await?;
      self.backend.write_page(dst_page, g.bytes()).await?;
      // 不在此清脏：本层读路径仍按 `pid*PAGE_SIZE` 缺页（CROSS-BUF-5 未接线
      // 映射感知读），保留脏位使驱逐时按 in-place 写回 pid，避免搬移页在
      // 驱逐后重载读到旧/已擦除块。映射感知读路径落地后再对齐上游 MakePageClean。
      drop(g); // 映射表更新不触页帧，先释放持闩窗口
      self
        .space
        .update_pid2_offset(pid, off, ldb_pack::UNCOMPRESSED)
        .await;
    }
    Ok(())
  }

  /// 脏页批量刷盘（上游 WriteDirtyPages：扫脏位图写盘 + 清位）。
  ///
  /// 修复：原版扫 `MARKED` 态判脏 —— 但 MARKED 仅在驱逐瞬态出现，
  /// 脏页从不 MARKED → flush_all 恒空转（模糊检查点丢数据窗口）。
  /// 修复 A3：持 Shared 闩写盘，清位在闩内完成 —— 写者必须等闩释放，
  /// 其后再标脏会重新置位，不会丢写。
  ///
  /// 本路径**无 WAL-before-page 门**（上游 `WriteDirtyPages` 同样无门）：
  /// `flush_all`/`end_data_load`/`shutdown` 要求落盘全部脏页。带门的检查点
  /// 路径见 [`Self::fuzzy_checkpoint`]。
  pub async fn write_dirty_pages(&self) -> Result<u64> {
    self.write_dirty_pages_upto(u64::MAX).await
  }

  /// 带 WAL-before-page 门的脏页刷盘（上游 `FlushDirtyPages` 的
  /// `page->p_gsn <= flushed_log_gsn` 门，`buffer_manager.cc:1550-1552`）：
  /// 仅落盘 `p_gsn <= max_p_gsn` 的脏页，其余保持脏位留待水位推进后重试。
  /// 否则崩溃后页比日志新，重放无法恢复（页写必须先于其 WAL 记录落盘的反向约束）。
  async fn write_dirty_pages_upto(&self, max_p_gsn: LogId) -> Result<u64> {
    let mut written = 0u64;
    let mut wi = 0usize;
    while wi < self.dirty_bits.len() {
      // 本字待处理位快照：处理过的位从掩码移除，避免「跳过不写 → 位仍脏 →
      // 重扫同一 tz」的死循环；写入期间新置的位留给下一遍。
      let mut todo = self.dirty_bits[wi].load(Ordering::Acquire);
      while todo != 0 {
        let tz = todo.trailing_zeros();
        todo &= todo - 1;
        let pid = wi as u64 * u64::from(u64::BITS) + u64::from(tz);
        let g = self.fix_shared(pid).await?;
        if read_p_gsn(g.bytes()) > max_p_gsn {
          // WAL 尚未落盘：本遍跳过（保持脏位，水位推进后重试）
          drop(g);
          continue;
        }
        self.backend.write_page(pid, g.bytes()).await?;
        self.clear_dirty_bit(pid); // 闩内清位：此后写者的标脏必然重新置位
        drop(g);
        written += 1;
      }
      wi += 1;
    }
    Ok(written)
  }

  /// 全量刷盘
  pub async fn flush_all(&self) -> Result<()> {
    self.write_dirty_pages().await?;
    self.backend.sync().await?;
    Ok(())
  }

  /// 模糊检查点（上游 FuzzyCheckpoint 的 C-17 修复版：min_flushed 读自与
  /// LogManager 共享的 GsnWatermark（D5），checkpoint 高水位单调推进）。
  ///
  /// 刷脏经 WAL-before-page 门（水位 = 已刷 WAL 的 `min_gsn_flushed`）：
  /// 只把 `p_gsn` 已落 WAL 的脏页写盘，其余留待下一轮。
  pub async fn fuzzy_checkpoint(&self, target_gsn: LogId) -> Result<LogId> {
    let min_flushed = self.min_gsn_flushed.load().max(target_gsn);
    self
      .checkpoint_offset
      .fetch_max(min_flushed, Ordering::AcqRel);
    self
      .write_dirty_pages_upto(self.min_gsn_flushed.load())
      .await?;
    self.backend.sync().await?;
    Ok(min_flushed)
  }

  /// WAL 信息更新（组提交回调；推进共享水位，LogManager 同视图）
  #[inline]
  pub fn update_wal_info(&self, flushed: LogId) {
    self.min_gsn_flushed.advance_min(flushed);
  }

  // ---- GC 协同（ldb_gc 注入调用）----

  /// GC 读有效页（上游 GCReadValidPIDs：逐块收集有效页到缓冲）
  pub async fn gc_read_valid_pi_ds(&self, bids: &[BlockId]) -> Result<Vec<PageId>> {
    let mut out = Vec::new();
    // 槽位数先落地：若把 `sspace.read()` 写进 `for` 头，临时读守卫会活到循环
    // 结束，而循环体的 `convert_offset2_pids` 要拿 `sspace.write()` ⇒ 自锁死。
    let slots = self.space.sspace.read().page_cnt_per_block;
    for &bid in bids {
      for slot in 0..slots {
        out.extend(self.space.convert_offset2_pids(bid, slot));
      }
    }
    out.retain(|&pid| self.space.page_is_valid(pid));
    Ok(out)
  }

  /// GC 读块内有效页（磁盘版）
  pub async fn gc_read_valid_pi_ds_in_block_on_disk(&self, bid: BlockId) -> Result<Vec<PageId>> {
    self.gc_read_valid_pi_ds(&[bid]).await
  }

  /// GC 写有效页（上游 GCWriteValidPIDs：搬移到 GC 写缓冲）
  pub async fn gc_write_valid_pi_ds(&self, pids: &[PageId]) -> Result<u64> {
    let mut moved = 0;
    for &pid in pids {
      if !self.space.page_is_valid(pid) {
        continue;
      }
      // 登记镜像（写 WAL 由注入的 TxnLogOf 完成 —— 上层编排）
      self.space.log_valid_page(pid).await;
      moved += 1;
    }
    Ok(moved)
  }

  /// 块内 GC 有效页计数（磁盘版）
  pub async fn gc_valid_pid_cnt_in_block_on_disk(&self, bid: BlockId) -> Result<u32> {
    Ok(self.space.valid_pi_ds_cnt_in_block(bid))
  }

  /// 把有效页刷入目标块（上游 FlushValidPIDsToBlock：装箱写 + 映射更新；
  /// A3：逐页持 Shared 闩写盘）
  pub async fn flush_valid_pi_ds_to_block(
    &self,
    pids: &[PageId],
    dst_block: BlockId,
  ) -> Result<u64> {
    let base = self.space.get_block_offset(dst_block);
    for (i, &pid) in pids.iter().enumerate() {
      let off = base + (i as u64) * PAGE_SIZE as u64;
      // 与 write_pages_out_of_place 同一落盘语义：写入目标块偏移
      let dst_page = off / PAGE_SIZE as u64;
      let g = self.fix_shared(pid).await?;
      self.backend.write_page(dst_page, g.bytes()).await?;
      // GC 搬移落盘：gc_move 页计数 + 物理写字节（上游 gced_page_cnt/gc_writes）
      self.stats.record_gc_move(PAGE_SIZE as u64);
      // 同 write_pages_out_of_place：保留脏位作为 in-place 回退（CROSS-BUF-5）
      drop(g);
      self
        .space
        .update_pid2_offset(pid, off, ldb_pack::UNCOMPRESSED)
        .await;
    }
    Ok(pids.len() as u64)
  }

  // ---- extent 辅助（Blob 大页预备；完整版在 ldb_blob）----

  /// extent 环境预备（上游 PrepareExtentEnv）
  pub async fn prepare_extent_env(&self, page_cnt: u32) -> Result<PageId> {
    self.alloc_extent(page_cnt).await
  }

  /// extent 驱逐预备（上游 PrepareExtentEviction）
  pub async fn prepare_extent_eviction(&self, start_pid: PageId, page_cnt: u32) -> Result<()> {
    self.evict_extent(start_pid, u64::from(page_cnt)).await
  }

  /// 尾 extent 预备（上游 PrepareTailExtent）
  pub async fn prepare_tail_extent(&self, start_pid: PageId) -> Result<PageId> {
    Ok(start_pid)
  }

  /// 分配元数据页（0 号页；上游 `AllocMetadataPage`）
  ///
  /// 原版直接对未固定的 arena 槽位写 8 字节：页处于 EVICTED（未加载）、
  /// Windows 上还是 reserved，写后也不标脏 ⇒ 数据静默丢失。改为走
  /// `update_pid`（EXCLUSIVE 缺页加载 + 写 + 标脏 + 解锁）。
  pub async fn alloc_metadata_page(&self) -> Result<PageId> {
    let pid = self.alloc_page().await?;
    if !self.update_pid(pid, &0u64.to_le_bytes()).await? {
      return Err(Error::InvalidPage(pid));
    }
    Ok(pid)
  }

  /// extent 页数（frame 登记）
  #[inline]
  pub fn extent_page_count(&self, _pid: PageId) -> u32 {
    0
  }

  /// GC 磁盘路径排他固定（上游 FixExclusiveOnDiskGC：与内存路径同构）
  pub async fn fix_exclusive_on_disk_gc(&self, pid: PageId) -> Result<()> {
    self.fix_exclusive(pid).await
  }

  /// GC 读路径共享固定（上游 FixShareImplGC）
  pub async fn fix_share_impl_gc(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<u64> {
    self.fix_share_impl(pid, dst).await
  }

  /// 空闲管理器（账本命名）
  #[inline]
  pub fn free_storage_manager(&self) -> &Arc<ldb_free::FreeStorageManager> {
    &self.free_mgr
  }

  /// GC 有效页批量登记（账本命名 gc_valid_pi_ds；= gc_write_valid_pi_ds）
  pub async fn gc_valid_pi_ds(&self, pids: &[PageId]) -> Result<u64> {
    self.gc_write_valid_pi_ds(pids).await
  }

  /// 驱逐循环线程（账本保留上游名；compio 化为 task —— 见 run_page_provider_task）
  pub async fn run_page_provider_threads(&self) -> Result<()> {
    self.run_page_provider_task().await
  }

  /// 分块操作（上游 ChunkOperation：大批量按 chunk 切分执行）
  pub async fn chunk_operation(
    &self,
    mut op: impl AsyncFnMut(PageId, u32) -> Result<()>,
    start_pid: PageId,
    total: u32,
    chunk: u32,
  ) -> Result<()> {
    let mut done = 0u32;
    while done < total {
      let n = chunk.min(total - done);
      op(start_pid + u64::from(done), n).await?;
      done += n;
    }
    Ok(())
  }

  /// 驱逐循环 task（上游 RunPageProviderThreads 的 compio 化：
  /// 周期性驱逐保水位；由运行时 spawn）
  pub async fn run_page_provider_task(&self) -> Result<()> {
    while self.is_running.load(Ordering::Relaxed) {
      let _ = self.ensure_free_pages().await;
      sleep(Duration::from_millis(10)).await;
    }
    Ok(())
  }
}

// 驱逐候选暂存（iterate_clock_batch 回调不可 async → 线程局部中转）
std::thread_local! {
    static CANDIDATES: RefCell<Vec<PageId>> = const { RefCell::new(Vec::new()) };
}

/// 组提交 PhaseFour 驱逐适配器（`EvictHook` 生产实现 —— D4 落地：
/// 原版 trait 仅测试实现，PhaseFour 的 `G: EvictHook` 在 L5 无法装配）。
///
/// 由 L5 组装：`GroupCommitExecutor::new(..., BufEvictHook { mgr: buf.clone() }, ...)`
pub struct BufEvictHook<B: PageBackend, T: TxnLogHook = NoTxnLog> {
  /// 缓冲池（`Rc` —— hook 与缓冲池同线程）
  pub mgr: Rc<BufferManager<B, T>>,
}

impl<B: PageBackend, T: TxnLogHook> fmt::Debug for BufEvictHook<B, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BufEvictHook").finish_non_exhaustive()
  }
}

impl<B: PageBackend, T: TxnLogHook> EvictHook for BufEvictHook<B, T> {
  /// 驱逐单页 extent（上游 EvictExtent 的 PhaseFour 语义：lp_req 逐 extent 回调）
  async fn evict_extent(&mut self, extent: PageId) {
    if let Err(e) = self.mgr.evict_extent(extent, 1).await {
      // PhaseFour 上游亦为 best-effort 驱逐：失败记日志，组提交流程不中断
      log::warn!("PhaseFour evict_extent(pid={extent}) 失败: {e}");
    }
  }

  /// 对外公布空闲区间（PhaseFive 列表裁剪；空闲管理器区间快照）
  async fn public_free_ranges(&mut self) -> Vec<(u64, u64)> {
    self.mgr.free_mgr.clone().public_free_ranges().await
  }
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::AtomicBool;

  /// 测试内 `Result` 解包（D1 禁 unwrap，统一走 panic 明确报错）
  macro_rules! ok {
    ($e:expr) => {
      match $e {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
      }
    };
  }

  use ldb_free::FreeStorageManager;
  use ldb_space::SpaceManager;

  use super::*;

  fn mk(pages: u64) -> BufferManager<MemBackend> {
    let space = Rc::new(SpaceManager::new(pages, 64, 2048, 4, pages * 64, 2));
    let free = Arc::new(FreeStorageManager::new(true, 2));
    let Ok(bm) = BufferManager::new(
      pages * 4,
      pages,
      space,
      free,
      MemBackend::new(),
      Arc::new(AtomicBool::new(true)),
      Arc::new(GsnWatermark::new(0)),
    ) else {
      panic!("BufferManager 构造失败");
    };
    bm
  }

  /// MARKED（同版本、无写者）必须被接受，并 Unmark 回 UNLOCKED；
  /// 变异「任何状态变化都 Restart」会被本条击杀。
  #[compio::test]
  async fn olc_validation_accepts_marked_and_unmarks() {
    let bm = mk(4);
    let pid = ok!(bm.alloc_page().await);
    let mut page = [0u8; PAGE_SIZE];
    let v = ok!(bm.fix_share_impl(pid, &mut page).await);
    assert_eq!(v, 1, "缺页加载后版本前移");
    let observed = bm.page_state_raw(pid).load(Ordering::Acquire);
    assert!(bm.try_mark_page(pid), "UNLOCKED → MARKED");
    assert_eq!(
      bm.validate_shared_snapshot(pid, observed).ok(),
      Some(1),
      "同版本 MARKED 快照可接受"
    );
    assert_eq!(
      bm.get_page_state(pid).lock_state(),
      PageStateMode::Unlocked,
      "校验需把 MARKED Unmark 回 UNLOCKED"
    );
  }

  /// 共享计数闭环：两次 fix_shared → Shared(2) → 逐个 Drop 回 UNLOCKED。
  /// 变异「unfix_share 直接回 UNLOCKED」会被第二条断言击杀。
  #[compio::test]
  async fn shared_count_decrements_one_by_one() {
    let bm = mk(4);
    let pid = ok!(bm.alloc_page().await);
    let g1 = ok!(bm.fix_shared(pid).await);
    let g2 = ok!(bm.fix_shared(pid).await);
    assert_eq!(
      bm.get_page_state(pid).lock_state(),
      PageStateMode::Shared(2)
    );
    // 零拷贝：视图指针必须就是 arena 页帧地址
    assert_eq!(g1.bytes().as_ptr() as usize, bm.page_addr(pid));
    drop(g1);
    assert_eq!(
      bm.get_page_state(pid).lock_state(),
      PageStateMode::Shared(1),
      "Drop 一个读者只减一"
    );
    drop(g2);
    assert_eq!(bm.get_page_state(pid).lock_state(), PageStateMode::Unlocked);
  }

  /// 元数据页必须走 EXCLUSIVE 加载 + 标脏（原版直接写 EVICTED 槽位：
  /// Windows 上写 reserved 页 = AV，且不标脏导致内容不落盘）。
  #[compio::test]
  async fn metadata_page_is_locked_dirty_and_persisted() {
    let bm = mk(4);
    let pid = ok!(bm.alloc_metadata_page().await);
    assert!(bm.page_is_dirty(pid), "分配元数据页后必须标脏");
    assert_eq!(
      bm.get_page_state(pid).lock_state(),
      PageStateMode::Unlocked,
      "写后必须解锁"
    );
    ok!(bm.flush_all().await);
    assert!(!bm.page_is_dirty(pid), "刷盘后清脏位");
    ok!(bm.evict(1).await);
    let mut back = [1u8; PAGE_SIZE];
    ok!(bm.read_page(pid, &mut back).await);
    assert_eq!(&back[..8], &0u64.to_le_bytes(), "元数据页内容持久");
  }

  /// OLC 确定性交错：在 4KiB 快照拷贝的中途注入写者/驱逐者，验证
  /// 「读状态 → 拷贝 → fence → 校验完整状态字」协议：
  /// - 写者改数据 + 版本 → Restart（且快照确实被撕裂，证明校验不是形式主义）；
  /// - 写者只改状态（同版本 EXCLUSIVE）→ Restart；
  /// - 驱逐者版本 +1 → EVICTED → Restart；
  /// - 无写者 → 通过。
  ///
  /// 变异「去掉校验 / 只校验状态不校验版本 / 拷贝前校验」会被本条唯一击杀。
  #[compio::test]
  async fn olc_mid_copy_interleavings_force_restart() {
    let bm = mk(4);
    let pid = ok!(bm.alloc_page().await);
    ok!(bm.update_pid(pid, &[0x11u8; 64]).await);
    let mut dst = [0u8; PAGE_SIZE];

    // 基线：无写者通过，快照与页一致（缺页 v1 + update_pid 解锁 v2）
    let v = ok!(bm.fix_share_impl(pid, &mut dst).await);
    assert_eq!(v, 2);
    assert_eq!(&dst[..64], &[0x11u8; 64]);

    // 1) 写者中途改后半段数据 + 版本 +1 → 快照撕裂，必须 Restart
    bm.mid_copy_inject.set(MidCopyInject::WriteDataAndVersion);
    assert!(
      matches!(bm.fix_share_impl(pid, &mut dst).await, Err(Error::Restart)),
      "拷贝途中版本漂移必须 Restart"
    );
    assert_eq!(&dst[..64], &[0x11u8; 64], "前半段是旧值");
    assert_eq!(
      &dst[PAGE_SIZE / 2..PAGE_SIZE / 2 + 64],
      &[0xABu8; 64],
      "后半段是新值 —— 快照确实跨越了写者（必须由 Restart 丢弃）"
    );
    let mut latest = [0u8; PAGE_SIZE];
    let v2 = ok!(bm.fix_share_impl(pid, &mut latest).await);
    assert_eq!(v2, v + 1, "写者解锁版本 +1");
    assert_eq!(
      &latest[PAGE_SIZE / 2..PAGE_SIZE / 2 + 64],
      &[0xABu8; 64],
      "页内容为写者新值"
    );

    // 2) 只改状态（同版本 EXCLUSIVE）→ 同版本也必须 Restart
    bm.mid_copy_inject.set(MidCopyInject::StateExclusiveOnly);
    assert!(
      matches!(bm.fix_share_impl(pid, &mut dst).await, Err(Error::Restart)),
      "拷贝途中出现 EXCLUSIVE 必须 Restart"
    );
    // 注入模拟写者已退出 → 手动恢复同版本 UNLOCKED（否则后续 fix 自旋）
    let ps = bm.page_state_raw(pid);
    let cur = ps.load(Ordering::Relaxed);
    ps.store(
      same_version_new_state(cur, PageStateMode::UNLOCKED),
      Ordering::Release,
    );

    // 3) 驱逐者版本 +1 → EVICTED → Restart
    bm.mid_copy_inject.set(MidCopyInject::StateEvicted);
    assert!(
      matches!(bm.fix_share_impl(pid, &mut dst).await, Err(Error::Restart)),
      "拷贝途中被驱逐必须 Restart"
    );
    assert_eq!(
      bm.get_page_state(pid).lock_state(),
      PageStateMode::Evicted,
      "注入的驱逐已生效"
    );
  }

  /// 缓存命中路径同样必须做拷贝后校验：版本先匹配、拷贝中途写者插队 →
  /// Restart。变异「fix_share_cached_page 只做拷贝前校验」被本条击杀。
  #[compio::test]
  async fn cached_mid_copy_interleaving_forces_restart() {
    let bm = mk(4);
    let pid = ok!(bm.alloc_page().await);
    ok!(bm.update_pid(pid, &[0x22u8; 64]).await);
    let mut page = [0u8; PAGE_SIZE];
    let v = ok!(bm.fix_share_impl(pid, &mut page).await);
    bm.mid_copy_inject.set(MidCopyInject::WriteDataAndVersion);
    let mut stale = [0u8; PAGE_SIZE];
    assert!(
      matches!(
        bm.fix_share_cached_page(pid, &mut stale, v).await,
        Err(Error::Restart)
      ),
      "缓存路径拷贝后版本漂移必须 Restart"
    );
  }

  /// `evict` 的候选缓冲必须跨轮复用（线程局部 Vec 原地 retain + 归还），
  /// 不得每轮 `take` 后丢弃导致重新分配。
  /// 变异「删掉归还候选缓冲的语句」使容量/指针在第二轮变为 0/新分配。
  #[compio::test]
  async fn evict_reuses_candidate_buffer() {
    let bm = mk(4);
    let mut pids = Vec::new();
    for _ in 0..4 {
      pids.push(ok!(bm.alloc_page().await));
    }
    for &pid in &pids {
      let mut out = [0u8; PAGE_SIZE];
      ok!(bm.fix_share_impl(pid, &mut out).await);
    }
    assert_eq!(ok!(bm.evict(1).await), 1);
    let (cap1, ptr1) = CANDIDATES.with(|c| {
      let b = c.borrow();
      (b.capacity(), b.as_ptr() as usize)
    });
    assert!(
      cap1 > 0,
      "驱逐候选缓冲必须持有容量（原版每轮 take 后丢弃 = 每轮堆分配）"
    );
    let _ = ok!(bm.evict(1).await);
    let (cap2, ptr2) = CANDIDATES.with(|c| {
      let b = c.borrow();
      (b.capacity(), b.as_ptr() as usize)
    });
    assert_eq!(cap1, cap2, "容量必须跨轮复用");
    assert_eq!(ptr1, ptr2, "底层分配必须跨轮复用（无重新分配）");
  }

  /// `gc_read_valid_pi_ds` 不得在 for 头持有 `sspace.read()` 临时守卫：
  /// 循环体 `convert_offset2_pids` 取 `sspace.write()`，同线程自锁。
  /// 变异「把槽位数读取写回 for 头」会使本用例无限挂起（harness 超时击杀）。
  #[compio::test]
  async fn gc_read_valid_pi_ds_completes_without_self_lock() {
    let bm = mk(4);
    bm.space.update_offset2_pids(0, 0, 0);
    let pids = ok!(bm.gc_read_valid_pi_ds(&[0]).await);
    // 映射表默认 comp_sz=0 ⇒ page_is_valid=false，结果为空但调用必须返回
    assert!(pids.is_empty(), "未映射页不得被判有效");
    // 磁盘别名路径同样不得自锁（内部转调 gc_read_valid_pi_ds）
    let pids = ok!(bm.gc_read_valid_pi_ds_in_block_on_disk(0).await);
    assert!(pids.is_empty());
  }
}
