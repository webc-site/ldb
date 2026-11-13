//! WAL 引擎：环形缓冲 + 每 Worker 日志器 + LogManager 协调者。
//!
//! 上游：`leanstore.latency/src/recovery/{log_buffer,log_worker,log_manager}.cc`。
//! 与上游差异（todo.md §3.4 协议）：
//! - `ensure_enough_space` 空间不足时 `.await` 异步落盘（compio 让出），禁止 AsmYield 忙等；
//! - 裸 io_uring 写 → `WalSink` trait（compio / 内存后端实现）；
//! - `global_min_gsn_flushed` → L0 `GsnWatermark`（消除 ldb_buf ↔ ldb_wal 环）；
//! - `tsctime` → `coarsetime`；预提交事务队列持最小票据结构（L4 转换）。

use std::{
  cell::RefCell,
  cmp, fmt,
  future::{Future, ready},
  io,
  pin::Pin,
  sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
  },
};

use futures_lite::future::yield_now;
use ldb_io::MemBackend;
use ldb_sync::HybridLatch;
use ldb_type::{GsnWatermark, PageId};
use parking_lot::Mutex;

use crate::{
  entry::{self, CR_ENTRY_SIZE, DATA_ENTRY_HDR, EntryType, FREE_EXTENT_HDR, LOG_ENTRY_HDR},
  error::{Error, Result},
};

/// 环回标记位（write_cursor 最高位，上游 wal_cursor_circ）。
///
/// 与上游对照结论：leanstore.latency 分支已按 log_buffer.h 头注释的 "Improvement"
/// 彻底消除置位 —— 环回方向由 wal_cursor 与 write_cursor 的大小关系推导
/// （wal_cursor >= write_cursor ⇒ 未环回；< ⇒ 已环回）。本实现与该分支同源：
/// store 侧永不置位，本常量与 CURSOR_MASK 仅作读取侧防御保留（掩码恒为恒等）。
/// ZLeanStore 分支保留置位语义（gct_cursor 最高位），若后续对齐该分支须补：
/// 环回 padding 处置位、组提交刷完尾部后清位。
pub const WAL_CIRC: usize = 1 << 63;
/// write_cursor 有效值掩码
pub const CURSOR_MASK: usize = !WAL_CIRC;

/// WAL 落盘对齐粒度（上游 `BLK_BLOCK_SIZE = 4096`，
/// `leanstore.latency/src/include/common/constants.h:22`；
/// `LogBuffer::PersistLog` 非 force 分支与 `EnsureEnoughSpace` 慢路径均传它，
/// `leanstore.latency/src/recovery/log_buffer.cc:44,46` +
/// `leanstore.latency/src/recovery/log_worker.cc:288,300`）。
///
/// DECISION（P2-ALIGN）：全链 4096 对齐分配，不用对齐垫、不用缓冲写降级 ——
/// `wal_offset` 从 4096 倍数的 `file.end` 起按 4096 倍数段长 fetch_sub 递减，
/// 偏移与长度天然保持 4096 倍数，Linux `O_DIRECT`（4Kn 盘要求 4096）不降级。
pub const WAL_WRITE_ALIGN: usize = 4096;
/// WAL 环回余量（上游 `WAL_BUFFER_SIZE = 2 MiB`，
/// `ZLeanStore/src/include/recovery/log_worker.h:25`：`w_offset_ + WAL_BUFFER_SIZE
/// >= w_end_offset_` 即回卷 —— 组提交执行器前向游标保留一个整缓冲的余量）。
pub(crate) const WAL_BUFFER_HEADROOM: u64 = 2 * 1024 * 1024;

/// 提交协议变体（上游 `CommitProtocol`）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitVariant {
  /// 集中式组提交
  Baseline,
  /// Worker 主动写自己的日志（WILO 写）
  WorkersWriteLog,
  /// WILO + 同伴日志窃取
  WiloSteal,
}

/// 预提交事务票据（L3 内最小表示；L4 `ldb_txn` 转换，避免 L3 → L4 依赖）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreCommitTicket {
  /// 提交时间戳
  pub commit_ts: u64,
  /// 观测到的最大 GSN
  pub max_observed_gsn: u64,
  /// 事务起始时间戳
  pub start_ts: u64,
  /// 关联 Blob 起始页（0 = 无）
  pub blob_page: u64,
}

impl PreCommitTicket {
  /// 是否等待组提交确认（上游 NeedsRemoteFlush）
  #[inline]
  pub fn is_remote_flush(&self) -> bool {
    self.blob_page != 0
  }
}

/// WAL 落盘目标（compio 文件 / 内存后端共用）。
///
/// 方法显式装箱 Future 以保持 dyn 兼容（LogManager 以 `Box<dyn WalSink>` 持有）；
/// I/O 开销远大于装箱开销，符合 SKILL「热路径避免 dyn」的例外权衡。
pub trait WalSink {
  /// 在 `off` 处写 `data`（拥有数据）
  fn write_wal(
    &mut self,
    data: Vec<u8>,
    off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>>;
  /// fdatasync
  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>>;
}

impl WalSink for ldb_io::CompioFileBackend {
  fn write_wal(
    &mut self,
    data: Vec<u8>,
    off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(self.write_owned(data, off))
  }

  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(self.fdatasync())
  }
}

impl WalSink for ldb_io::MemBackend {
  fn write_wal(
    &mut self,
    data: Vec<u8>,
    off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    // 字节偏移直写（原实现把 512B 扇区号传给按 pid*PAGE_SIZE 寻址的
    // write_at_page，偏移 8× 错位 —— ldb_io-round1 CROSS-IO-2）。
    // 上游无扇区覆写语义（io_uring 整段 pwrite，4096 对齐）；
    // MemBackend 无对齐约束，字节偏移直写即等价语义。
    Box::pin(async move { MemBackend::write_bytes(self, off, &data).await })
  }

  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(ready(Ok(())))
  }
}

impl WalSink for Box<dyn WalSink> {
  fn write_wal(
    &mut self,
    data: Vec<u8>,
    off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    (**self).write_wal(data, off)
  }

  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    (**self).sync()
  }
}

/// Worker 一致状态（原子发布，上游 `WorkerConsistentState`）
#[derive(Debug, Default)]
pub struct WorkerConsistentState {
  inner: Mutex<ToCommitState>,
  /// 乐观读闩（版本校验）
  pub latch: HybridLatch,
}

/// 非原子一致状态快照（上游 `ToCommitState`）
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ToCommitState {
  /// 已写 cursor
  pub last_wal_cursor: u64,
  /// 本地 GSN
  pub last_gsn: u64,
  /// 最近未固化提交时间戳
  pub precommitted_tx_commit_ts: u64,
}

impl ToCommitState {
  /// 全序比较（上游按 commit_ts）
  pub fn less_than(&self, r: &Self) -> bool {
    self.precommitted_tx_commit_ts < r.precommitted_tx_commit_ts
  }
}

impl WorkerConsistentState {
  /// 阻塞式读取（上游 `WorkerConsistentState::Clone` 的悲观等价）。
  /// 原实现为 128 轮自旋重试伪乐观读 —— 每轮重试含一次互斥锁加锁 + 两次原子读，
  /// 恒劣于直接单次加锁，重试纯为空转
  pub fn clone_blocking(&self) -> ToCommitState {
    self.snapshot()
  }

  /// X 闩写入（上游 SyncClone）
  pub async fn sync_clone(&self, other: ToCommitState) {
    self.latch.lock_exclusive().await;
    *self.inner.lock() = other;
    self.latch.unlock_exclusive();
  }

  /// 状态快照（上游 Clone 重载；命名避开标准 Clone trait 语义）
  pub fn snapshot(&self) -> ToCommitState {
    self.inner.lock().clone()
  }
}

/// 环形 WAL 缓冲（上游 `LogBuffer`；wal_cursor 仅 owner 写，write_cursor 双方共享）
#[derive(Debug)]
pub struct LogBuffer {
  buf: Box<[u8]>,
  /// 写游标（owner 独占）
  pub wal_cursor: usize,
  /// 已落盘游标（组提交 / owner 共享，最高位为环回标记）
  pub write_cursor: AtomicUsize,
}

impl LogBuffer {
  /// 构造。
  ///
  /// `size` 须为 [`WAL_WRITE_ALIGN`] 的倍数：`write_log_buffer` 的上界
  /// `UpAlign(wal_cursor)` 在 `cap % align == 0` 时才保证 `<= cap`
  /// （上游 buffer_size = `FLAGS_wal_buffer_size_mb * MB`，整数 MB ⇒ 天然满足）。
  pub fn new(size: usize) -> Self {
    assert!(
      size.is_multiple_of(WAL_WRITE_ALIGN) && size >= CR_ENTRY_SIZE * 2,
      "WAL 缓冲须 4KiB 对齐且容得下两条目"
    );
    Self {
      buf: vec![0u8; size].into_boxed_slice(),
      wal_cursor: 0,
      write_cursor: AtomicUsize::new(0),
    }
  }

  /// 容量
  #[inline]
  pub fn capacity(&self) -> usize {
    self.buf.len()
  }

  /// 总空闲（环回感知，上游 TotalFreeSpace）
  pub fn total_free_space(&self) -> u64 {
    let w = self.write_cursor.load(Ordering::Acquire) & CURSOR_MASK;
    if w == self.wal_cursor {
      return self.buf.len() as u64;
    }
    if self.wal_cursor > w {
      (w + self.buf.len() - self.wal_cursor) as u64
    } else {
      (w - self.wal_cursor) as u64
    }
  }

  /// 连续空闲（环回后可整段插入的空间，上游 ContiguousFreeSpaceForNewEntry）
  pub fn contiguous_free_space_for_new_entry(&self) -> u64 {
    let w = self.write_cursor.load(Ordering::Acquire) & CURSOR_MASK;
    if self.wal_cursor < w {
      (w - self.wal_cursor) as u64
    } else {
      (self.buf.len() - self.wal_cursor) as u64
    }
  }

  /// 当前写入点视图
  pub fn current(&mut self) -> &mut [u8] {
    &mut self.buf[self.wal_cursor..]
  }

  /// 读 `[off, off+len)`（ steal / 检查点用）
  pub fn slice(&self, off: usize, len: usize) -> &[u8] {
    &self.buf[off..off + len]
  }

  /// 把 `[write_cursor, wal_cursor)` 切成对齐段回调（上游 `WriteLogBuffer`，
  /// `leanstore.latency/src/recovery/log_buffer.cc:93-127`）。
  ///
  /// 段形状（重放契约，与上游的差别）：上游 `DownAlign(wal_written_cursor)` 把
  /// 段起点回拉到对齐边界 —— 段首重复携带上一段尾部的半条目，文件内字节重复
  /// 且段首不可解析（latency 分支无 replayer，重放语义无人消费）。本实现：
  /// - 段起点 = 写前沿（条目链对齐）—— 文件内每条目恰出现一次；
  /// - 内容尾部零填充到 `align` 倍数 —— 段长与落盘偏移保持 O_DIRECT 对齐；
  /// - 段首即条目首、零填充终止 → 逐段可独立重放。
  pub fn write_log_buffer(
    &self,
    wal_cursor: usize,
    align: usize,
    mut write_fn: impl FnMut(Vec<u8>, u64),
  ) {
    let written = self.write_cursor.load(Ordering::Acquire) & CURSOR_MASK;
    let cap = self.buf.len();
    let pad = |mut seg: Vec<u8>| {
      let alloc = seg.len().div_ceil(align) * align;
      seg.resize(alloc, 0);
      seg
    };
    if wal_cursor > written {
      write_fn(pad(self.buf[written..wal_cursor].to_vec()), written as u64);
    } else if wal_cursor < written {
      // 环回：先尾部（至环边界）再头部（至写前沿）；两段各自对齐补零
      write_fn(pad(self.buf[written..cap].to_vec()), written as u64);
      write_fn(pad(self.buf[..wal_cursor].to_vec()), 0);
    }
  }

  /// 已脏未落盘字节（上游 DIRTY_LOG_SIZE 宏；相等即全落盘 = 0，
  /// 修正上游宏在相等分支误回环增量的边界缺陷）
  pub fn dirty_size(&self) -> u64 {
    let w = self.write_cursor.load(Ordering::Acquire) & CURSOR_MASK;
    match self.wal_cursor.cmp(&w) {
      cmp::Ordering::Greater => (self.wal_cursor - w) as u64,
      cmp::Ordering::Less => (self.wal_cursor + self.buf.len() - w) as u64,
      cmp::Ordering::Equal => 0,
    }
  }
}

/// 窃取决策（上游 `StealDecision`）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StealDecision {
  /// 窃取同伴日志
  ToSteal,
  /// 本地直接写
  ToWriteLocally,
  /// 不动
  Nothing,
}

impl fmt::Debug for LogWorker {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("LogWorker")
      .field("wid", &self.wid)
      .field("lsn", &self.w_lsn_counter.load(Ordering::Relaxed))
      .field("gsn", &self.w_gsn_clock.load(Ordering::Relaxed))
      .finish_non_exhaustive()
  }
}

/// 每 Worker 日志器（上游 `LogWorker`；全部经 `Arc` 共享）
pub struct LogWorker {
  /// Worker 编号
  pub wid: usize,
  /// 提交协议
  pub variant: CommitVariant,
  /// 环形缓冲
  pub buffer: Mutex<LogBuffer>,
  /// LSN 计数
  pub w_lsn_counter: AtomicU64,
  /// 本地 GSN 时钟
  pub w_gsn_clock: AtomicU64,
  /// RFA 已刷 GSN 本地副本
  pub rfa_gsn_flushed: AtomicU64,
  /// 最近未固化提交时间戳
  pub last_unharden_commit_ts: AtomicU64,
  /// 对外发布的一致状态
  pub w_state: WorkerConsistentState,
  /// 组提交消费游标
  pub gct_cursor: AtomicUsize,
  /// 活动条目（reserve→submit 期间有效；页 X 锁内串行）
  pub active: Mutex<Option<ActiveEntry>>,
  /// RFA 事务队列（本地管理提交态；上游 `ms_queue_rfa`，WorkersWriteLog 非
  /// remote-flush 票据 —— `leanstore.latency/src/transaction/transaction_manager.cc:231`）
  pub ms_queue_rfa: Mutex<Vec<PreCommitTicket>>,
  /// 预提交队列（组提交消费；remote-flush 票据，上游 `precommitted_queue`）
  pub precommitted: Mutex<Vec<PreCommitTicket>>,
  /// RFA 组队列（组提交消费；上游 `precommitted_queue_rfa`，WiloSteal/blob 票据
  /// —— `transaction_manager.cc:229,244`）
  pub precommitted_rfa: Mutex<Vec<PreCommitTicket>>,
  /// 待发布状态停车场（WiloSteal 修复乱序发布，上游 parking_lot_log_flush）
  pub parking_lot_log_flush: Mutex<Vec<ToCommitState>>,
  pub parking_latch: HybridLatch,
  /// 触发本地写的脏字节阈值
  pub worker_write_batch: u64,
  /// 窃取组大小
  pub stealing_group: usize,
  /// 运行标记（共享自 LogManager）
  pub is_running: Arc<AtomicBool>,
}

/// 活动条目句柄（reserve 与 submit 之间持有；禁止并发 reserve）
#[derive(Debug, Clone, Copy)]
pub struct ActiveEntry {
  /// 条目在缓冲内的偏移
  pub offset: usize,
  /// 条目总长
  pub size: usize,
  /// 类型
  pub ty: EntryType,
}

impl LogWorker {
  /// 构造（上游 LogWorker ctor；io_uring 初始化由 compio 运行时取代）。
  ///
  /// 默认值取上游 config.cc：`wal_batch_write_kb = 16`（:33）、
  /// `wal_stealing_group_size = 8`（:39）
  pub fn init(
    size: usize,
    wid: usize,
    variant: CommitVariant,
    is_running: Arc<AtomicBool>,
  ) -> Arc<Self> {
    Arc::new(Self {
      wid,
      variant,
      buffer: Mutex::new(LogBuffer::new(size)),
      w_lsn_counter: AtomicU64::new(0),
      w_gsn_clock: AtomicU64::new(0),
      rfa_gsn_flushed: AtomicU64::new(0),
      last_unharden_commit_ts: AtomicU64::new(0),
      w_state: WorkerConsistentState::default(),
      gct_cursor: AtomicUsize::new(0),
      active: Mutex::new(None),
      ms_queue_rfa: Mutex::default(),
      precommitted: Mutex::default(),
      precommitted_rfa: Mutex::default(),
      parking_lot_log_flush: Mutex::default(),
      parking_latch: HybridLatch::new(),
      worker_write_batch: 16 * 1024,
      stealing_group: 8,
      is_running,
    })
  }

  // ---- GSN / 状态发布 ----

  /// 读本地 GSN
  #[inline]
  pub fn get_current_gsn(&self) -> u64 {
    self.w_gsn_clock.load(Ordering::Relaxed)
  }

  /// 设置本地 GSN（LogManager 周期同步用）
  #[inline]
  pub fn set_current_gsn(&self, gsn: u64) {
    self.w_gsn_clock.store(gsn, Ordering::Relaxed);
  }

  /// 发布 WAL cursor
  pub async fn public_w_cursor(&self) {
    let cur = self.buffer.lock().wal_cursor as u64;
    self.public_w_cursor_at(cur).await;
  }

  /// 发布本地 GSN（RFA 用）
  pub async fn public_local_gsn(&self) {
    let cur = self.buffer.lock().wal_cursor as u64;
    self.public_local_gsn_at(cur).await;
  }

  /// 发布提交时间戳（组提交 ack 依据）
  pub async fn public_commit_ts(&self) {
    let cur = self.buffer.lock().wal_cursor as u64;
    self.public_commit_ts_at(cur).await;
  }

  /// 已知 cursor 的发布半区（submit 路径前移 cursor 时顺带取值，省一次 buffer 锁）
  async fn public_w_cursor_at(&self, cur: u64) {
    let st = ToCommitState {
      last_wal_cursor: cur,
      ..self.w_state.snapshot()
    };
    self.w_state.sync_clone(st).await;
  }

  /// 已知 cursor 的 GSN 发布半区
  async fn public_local_gsn_at(&self, cur: u64) {
    let st = ToCommitState {
      last_wal_cursor: cur,
      last_gsn: self.w_gsn_clock.load(Ordering::Relaxed),
      precommitted_tx_commit_ts: self.w_state.snapshot().precommitted_tx_commit_ts,
    };
    self.w_state.sync_clone(st).await;
  }

  /// 已知 cursor 的提交时间戳发布半区
  async fn public_commit_ts_at(&self, cur: u64) {
    let st = ToCommitState {
      last_wal_cursor: cur,
      last_gsn: self.w_gsn_clock.load(Ordering::Relaxed),
      precommitted_tx_commit_ts: self.last_unharden_commit_ts.load(Ordering::Relaxed),
    };
    self.w_state.sync_clone(st).await;
  }

  // ---- 条目预留 / 提交 ----

  /// 确保缓冲有 `requested` 连续空间（不足则按协议落盘并让出；§3.4 协议 5）。
  /// 快路径单次持锁完成预算计算（原实现同一判定量拆成 5 次独立 lock/unlock，
  /// 无竞争下每次仍是一条 CAS + 两次原子读，条目级热路径纯属放大）
  async fn ensure_enough_space(self: &Arc<Self>, mgr: &LogManager, requested: usize) -> Result<()> {
    // 预留含环回 padding 的空间预算（上游 EnsureEnoughSpace 首段）
    let prepared = {
      let buf = self.buffer.lock();
      let tail = buf.capacity() - buf.wal_cursor;
      if tail < requested + CR_ENTRY_SIZE {
        requested + CR_ENTRY_SIZE + tail
      } else {
        requested + CR_ENTRY_SIZE
      }
    };
    self.ensure_space_slow(mgr, prepared).await?;
    // 环回 padding：单次持锁写 CR 条目并回卷 cursor
    let mut buf = self.buffer.lock();
    if buf.contiguous_free_space_for_new_entry() < (requested + CR_ENTRY_SIZE) as u64 {
      let cap = buf.capacity();
      let cur = buf.wal_cursor;
      let e = &mut buf.current()[..CR_ENTRY_SIZE];
      entry::write_hdr(e, EntryType::CarriageReturn, 0, (cap - cur) as u64);
      entry::seal_chksum(e);
      buf.wal_cursor = 0;
    }
    Ok(())
  }

  /// 空间不足慢路径：本地刷盘 + 让出，直到空闲达标（守卫严格限于非 await 域）。
  /// 上游 EnsureEnoughSpace 慢路径对 WORKERS_WRITE_LOG/WILO_STEAL 均调
  /// `PersistLog(owner, BLK_BLOCK_SIZE)`（log_buffer.cc:44,46）—— 对齐取 4096
  async fn ensure_space_slow(self: &Arc<Self>, mgr: &LogManager, prepared: usize) -> Result<()> {
    loop {
      {
        let buf = self.buffer.lock();
        if buf.total_free_space() >= prepared as u64 {
          return Ok(());
        }
        // 停机路径：复核空间（上游 N-15 修复 —— 不再静默溢出）
        if !self.is_running.load(Ordering::Relaxed) {
          if buf.total_free_space() < prepared as u64 {
            return Err(Error::WalOverflow);
          }
          return Ok(());
        }
      }
      // Baseline 中心执行器唤醒经 M5 通道接线；此处回退为本地刷以释放空间
      self.persist_log(mgr, WAL_WRITE_ALIGN).await?;
      if self.variant == CommitVariant::WiloSteal {
        self.public_commit_ts().await;
        let st = self.w_state.snapshot();
        mgr.commit_state(self.wid, st).await;
      }
      yield_now().await;
    }
  }

  /// 预留一条日志条目（写入头 + payload，cursor 不前移；页 X 锁内调用）
  pub async fn reserve_wal_entry(
    self: &Arc<Self>,
    mgr: &LogManager,
    ty: EntryType,
    pid: PageId,
    payload: &[u8],
  ) -> Result<ActiveEntry> {
    // FreeExtent/ReuseExtent：payload 即 8B LE extent size（编码进 lp_size 头字段，
    // 不进负载 —— 上游 FreeExtentEntry 恒 48B：start_pid@24 lp_size@32 gsn@40，
    // `leanstore.latency/src/include/recovery/log_entry.h:50-54`；
    // 原实现把 payload 长度误写进 lp_size 且总长 56B 偏离上游布局）
    let (total, lp_size) = match ty {
      EntryType::FreeExtent | EntryType::ReuseExtent => {
        let Ok(b) = <[u8; 8]>::try_from(payload) else {
          return Err(Error::InvalidConfig(
            "extent 条目 payload 须为 8B LE size".into(),
          ));
        };
        (FREE_EXTENT_HDR, u64::from_le_bytes(b))
      }
      _ => (DATA_ENTRY_HDR + payload.len(), 0),
    };
    self.ensure_enough_space(mgr, total).await?;
    let lsn = self.w_lsn_counter.fetch_add(1, Ordering::Relaxed);
    let gsn = self.get_current_gsn();

    let mut buf = self.buffer.lock();
    let offset = buf.wal_cursor;
    let e = &mut buf.current()[..total];
    match ty {
      EntryType::FreeExtent | EntryType::ReuseExtent => {
        entry::write_hdr(e, ty, lsn, total as u64);
        e[24..32].copy_from_slice(&pid.to_le_bytes()); // start_pid
        e[entry::off::LP_SIZE..entry::off::LP_SIZE + 8].copy_from_slice(&lp_size.to_le_bytes());
        e[40..48].copy_from_slice(&gsn.to_le_bytes());
      }
      _ => {
        entry::write_data_hdr(e, lsn, total as u64, pid, gsn);
        // 元类型修正（A4 配套）：DATA 布局承载 TX_START/TX_COMMIT/TX_ABORT/
        // FreePage/ReusePage/PageImg —— write_data_hdr 固定写 DataEntry 类型，
        // 恢复侧靠类型区分事务边界，必须覆盖为真实类型
        e[entry::off::TYPE] = ty.to_u8();
        e[DATA_ENTRY_HDR..total].copy_from_slice(payload);
      }
    }
    entry::seal_chksum(e);
    let active = ActiveEntry {
      offset,
      size: total,
      ty,
    };
    *self.active.lock() = Some(active);
    Ok(active)
  }

  /// 提交活动条目（前移 cursor + 按类型发布状态；返回是否刚触发本地落盘）
  pub async fn submit_active_log_entry(
    self: &Arc<Self>,
    mgr: &LogManager,
    commit_ts: u64,
  ) -> Result<bool> {
    let active = self.active.lock().take().ok_or(Error::NoActiveEntry)?;
    let mut just_write = false;
    // 前移时顺带取回新 cursor：后续发布不再重复加 buffer 锁读 wal_cursor
    let cur = {
      let mut buf = self.buffer.lock();
      buf.wal_cursor += active.size;
      buf.wal_cursor as u64
    };
    // P1-5：本地 GSN 随条目提交推进（原版只有 set/read —— 全链冻结，
    // min_gsn_flushed 恒 0 → durable 等待与 phase_five 固化门双双空转）
    self.w_gsn_clock.fetch_add(1, Ordering::Relaxed);

    if active.ty == EntryType::TxCommit {
      self
        .last_unharden_commit_ts
        .store(commit_ts, Ordering::Relaxed);
      match self.variant {
        CommitVariant::Baseline => self.public_commit_ts_at(cur).await,
        CommitVariant::WorkersWriteLog => {
          if self.should_steal_log() == StealDecision::ToWriteLocally {
            self.worker_writes_log(mgr, false).await?;
            just_write = true;
          } else {
            self.public_local_gsn_at(cur).await;
          }
        }
        CommitVariant::WiloSteal => {
          self.public_commit_ts_at(cur).await;
          match self.should_steal_log() {
            StealDecision::Nothing => {}
            to_steal => {
              just_write = true;
              self
                .worker_steals_log(mgr, to_steal == StealDecision::ToSteal, false)
                .await?;
            }
          }
        }
      }
    } else if active.ty == EntryType::DataEntry {
      self.public_local_gsn_at(cur).await;
    }
    Ok(just_write)
  }

  /// 在调用方缓冲中构造条目后入队（上游 PrepareDataLogEntry + SubmitPreparedLogEntry）
  pub async fn prepare_wal_entry(
    self: &Arc<Self>,
    mgr: &LogManager,
    buffer: &mut [u8],
    pid: PageId,
  ) -> Result<ActiveEntry> {
    let hdr = entry::read_hdr(buffer)?;
    let total = hdr.size as usize;
    // 预备路径不做空间等待（上游注释：prepared buffer 不等待）
    let lsn = self.w_lsn_counter.fetch_add(1, Ordering::Relaxed);
    let gsn = self.get_current_gsn();
    entry::write_data_hdr(
      &mut buffer[..DATA_ENTRY_HDR.min(total)],
      lsn,
      total as u64,
      pid,
      gsn,
    );
    entry::seal_chksum(&mut buffer[..total]);
    let active = {
      let mut buf = self.buffer.lock();
      // 上游“预备路径不等待”≠不检查：连续空间不足须显式报错，而非 slice 越界 panic
      if (buf.contiguous_free_space_for_new_entry() as usize) < total {
        return Err(Error::WalOverflow);
      }
      buf.current()[..total].copy_from_slice(&buffer[..total]);
      let a = ActiveEntry {
        offset: buf.wal_cursor,
        size: total,
        ty: EntryType::DataEntry,
      };
      *self.active.lock() = Some(a);
      a
    };
    self.submit_active_log_entry(mgr, 0).await?;
    Ok(active)
  }

  /// guard_wal 快捷提交（读取 payload 的完整预留 + 提交）
  pub async fn reserve_and_submit(
    self: &Arc<Self>,
    mgr: &LogManager,
    ty: EntryType,
    pid: PageId,
    payload: &[u8],
    commit_ts: u64,
  ) -> Result<u64> {
    let act = self.reserve_wal_entry(mgr, ty, pid, payload).await?;
    self.submit_active_log_entry(mgr, commit_ts).await?;
    Ok(act.size as u64)
  }

  // ---- 落盘 ----

  /// 把 `[write_cursor, wal_cursor)` 对齐写出到 WAL 文件（上游 PersistLog）
  pub async fn persist_log(self: &Arc<Self>, mgr: &LogManager, align: usize) -> Result<()> {
    // 收集切片（持锁拷贝，随后异步写）。
    // 零拷贝不可行：write_owned 需要数据所有权，且 write_cursor 推进后环形区即可被
    // owner 复写 —— 拷出是语义必需；但外层容器收敛为至多两段的栈数组（原 Vec 嵌套
    // 双重分配），回调的 off 死值也不再入容器（WAL 偏移由 wal_offset 自减分配）
    let mut segs: [Option<Vec<u8>>; 2] = [None, None];
    {
      let buf = self.buffer.lock();
      buf.write_log_buffer(buf.wal_cursor, align, |data, _| {
        if segs[0].is_none() {
          segs[0] = Some(data);
        } else {
          segs[1] = Some(data);
        }
      });
      buf.write_cursor.store(buf.wal_cursor, Ordering::Release);
    }
    for seg in segs.into_iter().flatten() {
      let size = seg.len() as u64;
      // 上游语义：w_offset_ 自减式分配（自文件尾向头部写）。
      // A6 守卫：下探越过 WAL 区起点（含 u64 下溢）= 环形容量耗尽（检查点未推进），
      // 必须报错而非回卷覆写未固化日志 / 下溢超大偏移写越界
      // （原实现 `off < file.start` 在 start=0 时恒假，下溢偏移直接落盘）
      let old = mgr.wal_offset.fetch_sub(size, Ordering::AcqRel);
      let off = old.wrapping_sub(size);
      if size > old || off < mgr.file.start {
        mgr.wal_offset.fetch_add(size, Ordering::AcqRel);
        return Err(Error::WalOverflow);
      }
      // swap-out：await 期间不持有 RefCell 借用（同线程重入安全）。
      // Option take/put-back：无堆分配、无双虚调用（原实现每次构造 Box<dyn> 胖指针）
      let sink = mgr.sink.borrow_mut().take();
      let r = match sink {
        Some(mut s) => {
          let r = s.write_wal(seg, off).await;
          *mgr.sink.borrow_mut() = Some(s);
          r
        }
        None => Ok(()),
      };
      r.map_err(Error::from)?;
    }
    self.public_commit_ts().await;
    // RFA 本地固化（上游 PersistLog → TryCommitRFATxns，log_buffer.cc:90）：
    // 票据离开队列即完成状态迁移；等待方的 ack 由 L4 durable_commit 的
    // 水位等待承担（ldb_txn txn_mgr.rs durable_commit），故此处回调为空钩子
    self.try_commit_rfa_txns(|_| ());
    Ok(())
  }

  /// WILO 本地写（上游 WorkerWritesLog：非 force 传 `BLK_BLOCK_SIZE`，force 传
  /// `FLAGS_wal_force_commit_alignment = 4KiB` —— 两者同为 4096）。
  ///
  /// `force_commit=true` 时在写出后对 sink 下达一次**设备屏障**（`WalSink::sync`）：
  /// 上游由**常驻**组提交线程的 PhaseTwo `Fsync` 承担该屏障（`leanstore.cc:64`
  /// `StartGroupCommitThread`），但本仓组提交为 task 化、且 `durable_commit` 的自驱
  /// 回退路径会绕过它 ⇒ 自驱提交必须在 force 分支自带屏障，否则 `wal_sync=true`
  /// 「提交返回即持久」不成立（掉电丢已提交事务）。`wal_sync` 门控由门面的
  /// `SyncPolicySink` 在 sink 层承担（false ⇒ 0 屏障）。非 force（后台补水 / 空闲
  /// 补偿）只写 OS/设备缓存，不刷盘，故不引入逐提交双屏障。
  pub async fn worker_writes_log(
    self: &Arc<Self>,
    mgr: &LogManager,
    force_commit: bool,
  ) -> Result<()> {
    assert_eq!(self.variant, CommitVariant::WorkersWriteLog);
    let dirty = {
      let b = self.buffer.lock();
      b.wal_cursor != b.write_cursor.load(Ordering::Acquire) & CURSOR_MASK
    };
    if dirty {
      self.persist_log(mgr, WAL_WRITE_ALIGN).await?;
    }
    if force_commit {
      Self::sync_sink(mgr).await?;
    }
    Ok(())
  }

  /// 对 WAL sink 下达一次落盘屏障（`WalSink::sync`）。
  ///
  /// swap-out 语义与 `persist_log`/`phase_two` 一致：await 期间不持 `RefCell` 借用
  /// （take/put-back），`None` 窗口按 NullSink 处理（不 panic）。
  async fn sync_sink(mgr: &LogManager) -> Result<()> {
    let sink = mgr.sink.borrow_mut().take();
    let r = match sink {
      Some(s) => {
        let r = s.sync().await;
        *mgr.sink.borrow_mut() = Some(s);
        r
      }
      None => Ok(()),
    };
    r.map_err(Error::from)
  }

  /// 是否应写 / 窃取（上游 ShouldStealLog）
  pub fn should_steal_log(&self) -> StealDecision {
    let dirty = self.buffer.lock().dirty_size();
    match self.variant {
      CommitVariant::WorkersWriteLog => {
        if dirty >= self.worker_write_batch {
          StealDecision::ToWriteLocally
        } else {
          StealDecision::Nothing
        }
      }
      CommitVariant::WiloSteal => {
        if dirty >= self.worker_write_batch {
          StealDecision::ToWriteLocally
        } else if dirty
          < self.worker_write_batch / u64::from(self.stealing_group.max(1).ilog2() + 1)
        {
          StealDecision::Nothing
        } else {
          StealDecision::ToSteal
        }
      }
      CommitVariant::Baseline => StealDecision::Nothing,
    }
  }

  /// WILO 窃取（上游 WorkerStealsLog；group_size==0 的 VLA 缺陷已消除，C-27）
  pub async fn worker_steals_log(
    self: &Arc<Self>,
    mgr: &LogManager,
    write_only: bool,
    force_commit: bool,
  ) -> Result<()> {
    assert_eq!(self.variant, CommitVariant::WiloSteal);
    if write_only {
      {
        let st = self.w_state.snapshot();
        self.parking_lot_log_flush.lock().push(st);
      }
      let dirty = {
        let b = self.buffer.lock();
        b.wal_cursor != b.write_cursor.load(Ordering::Acquire) & CURSOR_MASK
      };
      if dirty {
        // 对齐档位与 worker_writes_log 同源（force = 4KiB = BLK_BLOCK_SIZE）
        let _ = force_commit;
        self.persist_log(mgr, WAL_WRITE_ALIGN).await?;
      }
      let st = self.w_state.snapshot();
      self.try_publish_commit_state(mgr, self.wid, st).await;
      return Ok(());
    }

    let group = self.stealing_group;
    let start_wid = (self.wid / group) * group;
    let group_size = group.min(mgr.worker_cnt() - start_wid);
    let mut success = vec![false; group_size];
    let mut peers = vec![ToCommitState::default(); group_size];
    for idx in 0..group_size {
      let pid_ = start_wid + idx;
      if pid_ == self.wid {
        let st = self.w_state.snapshot();
        self.parking_lot_log_flush.lock().push(st);
        continue;
      }
      if let Some(st) = self.try_steal_logs(mgr, pid_) {
        success[idx] = true;
        peers[idx] = st;
      }
    }
    // 上游 WorkerStealsLog 非 write_only 分支 PersistLog(this, BLK_BLOCK_SIZE)
    // （leanstore.latency/src/recovery/log_worker.cc:321，BLK_BLOCK_SIZE=4096，
    // constants.h:22）—— P2-ALIGN 决策：全链 4096 对齐段长/偏移
    self.persist_log(mgr, WAL_WRITE_ALIGN).await?;
    let st = self.w_state.snapshot();
    self.try_publish_commit_state(mgr, self.wid, st).await;
    for idx in 0..group_size {
      if success[idx] {
        self
          .try_publish_commit_state(mgr, start_wid + idx, peers[idx].clone())
          .await;
      }
    }
    Ok(())
  }

  /// 窃取单个同伴的脏日志（上游 TryStealLogs；成功返回其一致状态）。
  ///
  /// 并发契约（与上游 parking_latch X 闩 {emplace, CAS} 临界区同义）：
  /// CAS 接管 `peer.write_cursor` 与停车场登记在 `peer.parking_lot_log_flush`
  /// 互斥锁内原子完成 —— `try_publish_commit_state` 亦持该锁，故
  /// 「接管但未登记」的窗口为零；同伴自身 `persist_log` 的 write_cursor 推进
  /// 不经此锁，由 `compare_exchange` 拒绝（在途刷盘时窃取失败返回 None）。
  /// 环形缓冲锁保持全局 wid 升序（互窃防 ABBA 死锁）。
  pub fn try_steal_logs(&self, mgr: &LogManager, peer_id: usize) -> Option<ToCommitState> {
    let peer = mgr.worker(peer_id)?;
    let out_state = peer.w_state.clone_blocking();
    let last_written = peer.buffer.lock().write_cursor.load(Ordering::Acquire) & CURSOR_MASK;
    if out_state.last_wal_cursor == last_written as u64 {
      return None;
    }
    // 本缓冲须容纳同伴整段脏日志 + CR
    let need = {
      let p = peer.buffer.lock();
      let dirty = if out_state.last_wal_cursor as usize > last_written {
        out_state.last_wal_cursor as usize - last_written
      } else {
        out_state.last_wal_cursor as usize + p.capacity() - last_written
      };
      dirty + CR_ENTRY_SIZE
    };
    {
      let me = self.buffer.lock();
      if (me.contiguous_free_space_for_new_entry() as usize) < need {
        return None;
      }
    }
    // 拷贝（持 peer 停车场锁，覆盖 CAS 直至登记的整个临界区）+ 按 wid 升序锁环
    let mut lot = peer.parking_lot_log_flush.lock();
    let (mut mine, pbuf) = if self.wid < peer_id {
      (self.buffer.lock(), peer.buffer.lock())
    } else {
      let pbuf = peer.buffer.lock();
      (self.buffer.lock(), pbuf)
    };
    let total_copy;
    if out_state.last_wal_cursor as usize > last_written {
      total_copy = out_state.last_wal_cursor as usize - last_written;
      let seg = pbuf.slice(last_written, total_copy).to_vec();
      mine.current()[..total_copy].copy_from_slice(&seg);
    } else {
      let first = pbuf.capacity() - last_written;
      let seg1 = pbuf.slice(last_written, first).to_vec();
      let seg2 = pbuf.slice(0, out_state.last_wal_cursor as usize).to_vec();
      mine.current()[..first].copy_from_slice(&seg1);
      mine.current()[first..first + seg2.len()].copy_from_slice(&seg2);
      total_copy = first + seg2.len();
    }
    drop(pbuf);
    drop(mine);
    // CAS 接管（真原子 CAS：同伴在途 persist 的 write_cursor 推进即失败）
    let taken = peer
      .buffer
      .lock()
      .write_cursor
      .compare_exchange(
        last_written,
        out_state.last_wal_cursor as usize,
        Ordering::AcqRel,
        Ordering::Acquire,
      )
      .is_ok();
    if !taken {
      return None;
    }
    lot.push(out_state.clone());
    drop(lot);
    let mut me = self.buffer.lock();
    me.wal_cursor += total_copy;
    Some(out_state)
  }

  /// 修正乱序发布的提交状态发布（上游 TryPublishCommitState，修复 CloneIfHigherTS bug）。
  ///
  /// 判定 + 合并收敛为单临界区（原实现 merge 分支二次加锁，两段之间停车场
  /// 可能被并发改写，merge 落到错误条目上）；判定结果在锁内解出，
  /// `mgr.commit_state` 的 await 在锁外。
  pub async fn try_publish_commit_state(&self, mgr: &LogManager, w_id: usize, st: ToCommitState) {
    let Some(peer) = mgr.worker(w_id) else { return };
    enum Decision {
      /// 最旧待发布：直接对外发布
      Publish(ToCommitState),
      /// 存在更旧在途写：已合并到其上（上游 `*(ite - 1) = *ite` 语义）
      Merged,
    }
    let decision = {
      let mut lot = peer.parking_lot_log_flush.lock();
      lot.sort_by_key(|a| a.precommitted_tx_commit_ts);
      let pos = lot
        .iter()
        .position(|s| !s.less_than(&st))
        .unwrap_or(lot.len());
      if pos == lot.len() {
        // 上游此处 Ensure(ite != end)（log_worker.cc:424）：写侧须先登记后刷盘，
        // 找不到 = 协议违例。debug 断言暴露，release 静默跳过（不发布错误状态）
        debug_assert!(false, "发布未登记的提交状态 wid={w_id}");
        None
      } else if pos == 0 {
        Some(Decision::Publish(lot.remove(pos)))
      } else {
        lot.remove(pos);
        if let Some(last) = lot.last_mut() {
          *last = st;
        }
        Some(Decision::Merged)
      }
    };
    if let Some(Decision::Publish(found)) = decision {
      mgr.commit_state(w_id, found).await;
    }
  }

  // ---- 杂项 ----

  /// 遍历缓冲内全部未落盘条目（上游函数体 throw 未实现异常 —— 本实现补全，C-36）。
  ///
  /// 边界语义：条目链止于写前沿 `wal_cursor`（submit 才前移）—— 半条目
  /// （头部已写、payload 未写完、cursor 未前移）不进回调；环容量只作兜底
  /// （原实现仅查容量，半条目会被当完整条目误报）
  pub fn iterate_active_txn_entries(&self, mut log_cb: impl FnMut(&entry::EntryHdr, &[u8])) {
    let buf = self.buffer.lock();
    let mut cur = buf.write_cursor.load(Ordering::Acquire) & CURSOR_MASK;
    loop {
      if cur == buf.wal_cursor {
        break;
      }
      let remain = buf.capacity() - cur;
      if remain < LOG_ENTRY_HDR {
        break; // CR 尾部
      }
      let Ok(hdr) = entry::read_hdr(buf.slice(cur, remain)) else {
        break;
      };
      let size = hdr.size as usize;
      if size == 0 || size > remain || cur + size > buf.wal_cursor {
        break;
      }
      log_cb(&hdr, buf.slice(cur, size));
      if hdr.ty == EntryType::CarriageReturn {
        cur = 0;
        continue;
      }
      cur += size;
    }
  }

  /// RFA 事务固化确认（上游 TryCommitRFATxns；tsctime → 直接时间戳比较）
  pub fn try_commit_rfa_txns(&self, mut durable_cb: impl FnMut(PreCommitTicket)) {
    if self.variant != CommitVariant::WorkersWriteLog {
      return;
    }
    let unhardened = self.last_unharden_commit_ts.load(Ordering::Relaxed);
    let mut q = self.ms_queue_rfa.lock();
    let keep = q.partition_point(|t| t.commit_ts <= unhardened);
    for t in q.drain(..keep) {
      durable_cb(t);
    }
  }

  /// 预留元日志条目（TX_START / TX_COMMIT / TX_ABORT 等定长条目）
  pub async fn reserve_log_meta_entry(
    self: &Arc<Self>,
    mgr: &LogManager,
    ty: EntryType,
  ) -> Result<ActiveEntry> {
    self.reserve_wal_entry(mgr, ty, 0, &[]).await
  }

  /// 预留 extent 释放 / 复用日志
  pub async fn reserve_free_extent_log_entry(
    self: &Arc<Self>,
    mgr: &LogManager,
    to_free: bool,
    start_pid: PageId,
    size: u64,
  ) -> Result<ActiveEntry> {
    let ty = if to_free {
      EntryType::FreeExtent
    } else {
      EntryType::ReuseExtent
    };
    self
      .reserve_wal_entry(mgr, ty, start_pid, &size.to_le_bytes())
      .await
  }
}

/// WAL 文件区段描述（独立文件布局：wal.log 单文件环形）
#[derive(Debug, Clone, Copy)]
pub struct WalFileState {
  /// WAL 起始文件偏移
  pub start: u64,
  /// WAL 结束偏移（开区间）
  pub end: u64,
  /// 分片大小（检查点目标粒度）
  pub shard: u64,
}

impl fmt::Debug for LogManager {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("LogManager")
      .field("workers", &self.workers.len())
      .field("min_gsn_flushed", &self.min_gsn_flushed.load())
      .finish_non_exhaustive()
  }
}

/// 日志管理协调者（上游 `LogManager`）
pub struct LogManager {
  /// 全局最小已刷 GSN 水位（L0 下沉件；`Arc` 与 `BufferManager` 共享同一实例
  /// —— D5 消除 buf 侧私有水位双轨）
  pub min_gsn_flushed: Arc<GsnWatermark>,
  /// 全局同步 GSN（防 RFA 本地 GSN 漂移）
  pub global_sync_gsn: AtomicU64,
  /// WAL 文件写偏移（自 end 向 start 递减，fetch_sub 分配）
  pub wal_offset: AtomicU64,
  /// 落盘目标（单 runtime 线程持有；多核接入用 per-thread manager，M5 接线）。
  /// None 窗口 = swap-out 中（await 期间被 take，落点为 NullSink 语义）
  pub sink: RefCell<Option<Box<dyn WalSink>>>,
  /// 全部 worker
  workers: Vec<Arc<LogWorker>>,
  /// 各 worker 已发布的提交状态
  commit_states: Vec<Arc<WorkerConsistentState>>,
  /// 运行标记
  pub is_running: Arc<AtomicBool>,
  /// 文件区段
  pub file: WalFileState,
}

impl LogManager {
  /// 构造：`worker_cnt` 个 worker + 全局状态。
  ///
  /// 文件区段边界须为 [`WAL_WRITE_ALIGN`] 倍数（偏移/段长全链 4KiB 对齐，
  /// P2-ALIGN 决策）；`wal_offset` 自 `file.end` 反向下探。
  ///
  /// 返回非 Arc：compio File 线程局部（!Send），跨线程共享由 M5 用
  /// per-thread manager 或 send-wrapper 接线；单 runtime 线程内直接 `Rc` 亦可。
  pub fn new(
    worker_cnt: usize,
    buffer_size: usize,
    variant: CommitVariant,
    sink: Box<dyn WalSink>,
    file: WalFileState,
  ) -> Self {
    debug_assert!(
      file.start.is_multiple_of(WAL_WRITE_ALIGN as u64)
        && file.end.is_multiple_of(WAL_WRITE_ALIGN as u64)
        && file.shard.is_multiple_of(WAL_WRITE_ALIGN as u64),
      "WAL 文件区段边界须 4KiB 对齐"
    );
    let is_running = Arc::new(AtomicBool::new(true));
    let workers = (0..worker_cnt)
      .map(|wid| LogWorker::init(buffer_size, wid, variant, is_running.clone()))
      .collect();
    let commit_states = (0..worker_cnt)
      .map(|_| Arc::new(WorkerConsistentState::default()))
      .collect();
    Self {
      min_gsn_flushed: Arc::new(GsnWatermark::new(0)),
      global_sync_gsn: AtomicU64::new(0),
      wal_offset: AtomicU64::new(file.end),
      sink: RefCell::new(Some(sink)),
      workers,
      commit_states,
      is_running,
      file,
    }
  }

  /// worker 数
  #[inline]
  pub fn worker_cnt(&self) -> usize {
    self.workers.len()
  }

  /// 取 worker
  #[inline]
  pub fn worker(&self, wid: usize) -> Option<&Arc<LogWorker>> {
    self.workers.get(wid)
  }

  /// 本地 worker（单线程每核：wid == 当前线程编号）
  #[inline]
  pub fn local_log_worker(&self, wid: usize) -> &Arc<LogWorker> {
    &self.workers[wid % self.workers.len()]
  }

  /// 读取 worker 提交状态
  pub async fn commit_state(&self, wid: usize, st: ToCommitState) {
    if let Some(cs) = self.commit_states.get(wid) {
      cs.sync_clone(st).await;
    }
  }

  /// 读取已发布提交状态快照
  pub fn commit_state_snapshot(&self, wid: usize) -> ToCommitState {
    match self.commit_states.get(wid) {
      Some(s) => s.snapshot(),
      None => ToCommitState::default(),
    }
  }

  /// 触发一轮组提交（由 L5 持执行器调用；此处仅暴露文件偏移视图）
  #[inline]
  pub fn should_keep_running(&self) -> bool {
    self.is_running.load(Ordering::Relaxed)
  }

  /// WAL 文件当前偏移
  #[inline]
  pub fn wal_file_offset(&self) -> u64 {
    self.wal_offset.load(Ordering::Acquire)
  }

  /// WAL 偏移（账本命名）
  #[inline]
  pub fn wal_offset(&self) -> u64 {
    self.wal_file_offset()
  }

  /// 停机（缓冲内日志须先 persist）
  pub fn shutdown(&self) {
    self.is_running.store(false, Ordering::Release);
  }
}

/// 预提交入队（上游 `TransactionManager::DurableCommit` 的队列路由半区，
/// `leanstore.latency/src/transaction/transaction_manager.cc:222-247`）：
/// - remote-flush（blob）→ `precommitted`（组提交 phase_three/four 写 Blob 后固化）
/// - WorkersWriteLog 其余 → `ms_queue_rfa`（`persist_log` → `try_commit_rfa_txns` 本地即时固化）
/// - WiloSteal 其余 → `precommitted_rfa`（组提交 phase_five 按 `commit_ts <=
///   worker_states[w].precommitted_tx_commit_ts` 固化，latency COMPLETE_RFA_QUEUE）
pub fn enqueue_precommitted(worker: &LogWorker, ticket: PreCommitTicket) {
  if ticket.is_remote_flush() {
    worker.precommitted.lock().push(ticket);
  } else if worker.variant == CommitVariant::WorkersWriteLog {
    worker.ms_queue_rfa.lock().push(ticket);
  } else {
    worker.precommitted_rfa.lock().push(ticket);
  }
}
