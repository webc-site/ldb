//! 事务对象（上游 `leanstore.latency transaction/transaction.h` + ZLeanStore）。
//!
//! 显式 `&mut Transaction` 传递（§3.4 协议 8：替代线程局部 `TM::active_txn`）。

use std::{fmt, mem::size_of};

/// 事务隔离级别（上游 IsolationLevel）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
  /// 读未提交
  ReadUncommitted,
  /// 读已提交
  ReadCommitted,
  /// 快照隔离
  SnapshotIsolation,
  /// 串行化（占位：SSI 未实现，按 SI 处理）
  Serializable,
}

/// 事务状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
  /// 运行中
  Running,
  /// 预提交（等待 WAL 固化）
  Precommitted,
  /// 已提交
  Committed,
  /// 已回滚
  Aborted,
}

/// 事务类型标记（上游 OLAP_BIT / RC_BIT）
pub mod tx_flag {
  /// RC 模式位
  pub const RC_BIT: u64 = 1 << 62;
  /// OLAP 模式位
  pub const OLAP_BIT: u64 = 1 << 61;
  /// 闩位（快照更新中）
  pub const LATCH_BIT: u64 = 1 << 63;
  /// 干净位掩码（纯 start_ts）
  pub const CLEAN_BITS_MASK: u64 = !(RC_BIT | OLAP_BIT | LATCH_BIT);
}

/// 事务（每 worker 单活动事务；compio 下显式传递）
pub struct Transaction {
  /// 状态
  pub state: TxnState,
  /// 起始时间戳
  pub start_ts_val: u64,
  /// 提交时间戳（Precommitted 起有效）
  pub commit_ts_val: u64,
  /// 隔离级别
  pub isolation: IsolationLevel,
  /// OLAP 标记
  pub olap: bool,
  /// 是否有写
  pub wrote: bool,
  /// 是否含 Blob 操作
  pub has_blob_op: bool,
  /// WAL 起始 cursor（恢复用）
  pub start_wal_cursor_val: u64,
  /// 待释放 extent 列表（事务提交后生效，上游 ToFreeExtents）
  pub to_free_extents: Vec<(u64, u8)>,
  /// 待驱逐 extent 列表（上游 ToEvictedExtents）
  pub to_evicted_extents: Vec<u64>,
  /// 待刷大页列表（上游 ToFlushedLargePages）
  pub to_flushed_large_pages: Vec<u64>,
  /// WAL 日志器（弱引用式：单线程每核下直接裸指针经由 manager 注入）
  pub wal_pid: Option<u64>,
}

impl fmt::Debug for Transaction {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Transaction")
      .field("state", &self.state)
      .field("start_ts", &self.start_ts_val)
      .field("commit_ts", &self.commit_ts_val)
      .field("isolation", &self.isolation)
      .field("olap", &self.olap)
      .finish_non_exhaustive()
  }
}

impl Transaction {
  /// 初始化（构造 + 状态复位一体）
  pub fn initialize(&mut self, start_ts: u64, isolation: IsolationLevel) {
    self.reset_state();
    self.state = TxnState::Running;
    self.start_ts_val = start_ts;
    self.isolation = isolation;
    self.wrote = false;
    self.has_blob_op = false;
    self.start_wal_cursor_val = 0;
    self.to_free_extents.clear();
    self.to_evicted_extents.clear();
    self.to_flushed_large_pages.clear();
  }

  /// 全量复位（事务结束复用对象）
  pub fn reset_state(&mut self) {
    self.state = TxnState::Aborted;
    self.commit_ts_val = 0;
    self.olap = false;
    self.wal_pid = None;
  }

  /// 是否运行中
  #[inline]
  pub fn is_running(&self) -> bool {
    self.state == TxnState::Running
  }

  /// 起始时间戳
  #[inline]
  pub fn start_ts(&self) -> u64 {
    self.start_ts_val
  }

  /// 提交时间戳
  #[inline]
  pub fn commit_ts(&self) -> u64 {
    self.commit_ts_val
  }

  /// 是否只读
  #[inline]
  pub fn read_only(&self) -> bool {
    !self.wrote
  }

  /// 是否含 Blob
  #[inline]
  pub fn has_blob(&self) -> bool {
    self.has_blob_op
  }

  /// 是否 OLTP
  #[inline]
  pub fn is_oltp(&self) -> bool {
    !self.olap
  }

  /// 是否 OLAP
  #[inline]
  pub fn is_olap(&self) -> bool {
    self.olap
  }

  /// 是否读未提交
  #[inline]
  pub fn is_read_uncommitted(&self) -> bool {
    self.isolation == IsolationLevel::ReadUncommitted
  }

  /// 是否读已提交
  #[inline]
  pub fn is_read_committed(&self) -> bool {
    self.isolation == IsolationLevel::ReadCommitted
  }

  /// 是否快照隔离
  #[inline]
  pub fn is_si(&self) -> bool {
    matches!(
      self.isolation,
      IsolationLevel::SnapshotIsolation | IsolationLevel::Serializable
    )
  }

  /// 至少 SI（SI 或更高）
  #[inline]
  pub fn at_least_si(&self) -> bool {
    self.is_si()
  }

  /// 可用单版本优化（RC 且无写意图时）
  #[inline]
  pub fn can_use_single_version(&self) -> bool {
    self.is_read_committed() && self.read_only()
  }

  /// 是否已写入
  #[inline]
  pub fn has_wrote(&self) -> bool {
    self.wrote
  }

  /// 标记写意图
  #[inline]
  pub fn mark_as_write(&mut self) {
    self.wrote = true;
  }

  /// 是否处于持久化提交路径（上游 gen1 `startTX`：`active_tx.is_durable = FLAGS_wal`，
  /// `Worker.cpp:93` —— 本引擎 WAL 恒启用 ⇒ 事务一旦启动即持久路径，终态 Aborted 除外）
  #[inline]
  pub fn is_durable(&self) -> bool {
    !matches!(self.state, TxnState::Aborted)
  }

  /// WAL 起始 cursor
  #[inline]
  pub fn start_wal_cursor(&self) -> u64 {
    self.start_wal_cursor_val
  }

  /// GSN 依赖检测（页 GSN 落后于全局水位时须等待 —— 上游 DetectGSNDependency；
  /// 单线程每核下直接返回是否需等待，由调用方 yield 重试）
  #[inline]
  pub fn detect_gsn_dependency(&self, page_gsn: u64, flushed_watermark: u64) -> bool {
    page_gsn > flushed_watermark
  }

  /// 推进页 GSN（本事务观测到的最大 GSN）
  #[inline]
  pub fn advance_page_gsn(&self, cur: u64, page_gsn: u64) -> u64 {
    cur.max(page_gsn)
  }

  /// Blob 日志登记（记录关联页）
  pub fn log_blob(&mut self, pid: u64) {
    self.has_blob_op = true;
    self.wal_pid = Some(pid);
  }

  /// 释放 extent 登记（提交后公布）
  pub fn to_free_extent(&mut self, start_pid: u64, tier: u8) {
    self.to_free_extents.push((start_pid, tier));
  }

  /// 序列化尺寸估算（上游 SerializedSize：状态 + extent 列表）
  pub fn serialized_size(&self) -> u64 {
    64 + 16 * self.to_free_extents.len() as u64
      + 8 * self.to_evicted_extents.len() as u64
      + 8 * self.to_flushed_large_pages.len() as u64
  }

  /// 内存占用（上游 MemorySize）
  pub fn memory_size(&self) -> u64 {
    size_of::<Self>() as u64 + self.serialized_size()
  }

  /// 序列化到缓冲（上游 Serialize：定长头 + 变长列表）
  pub fn serialize_into(&self, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&self.start_ts_val.to_le_bytes());
    buf.extend_from_slice(&self.commit_ts_val.to_le_bytes());
    buf.extend_from_slice(&(self.isolation as u8).to_le_bytes());
    buf.push(u8::from(self.olap));
    buf.extend_from_slice(&(self.to_free_extents.len() as u32).to_le_bytes());
    for &(pid, tier) in &self.to_free_extents {
      buf.extend_from_slice(&pid.to_le_bytes());
      buf.push(tier);
    }
  }

  /// 非法缓冲判定（上游 InvalidByteBuffer：头不足）
  pub fn invalid_byte_buffer(buf: &[u8]) -> bool {
    buf.len() < 18
  }

  /// WAL 日志器视图（账本名：本事务绑定的日志 worker id）
  pub async fn log_worker(&self) -> Option<u64> {
    self.wal_pid
  }

  /// 待驱逐 extent 列表（账本名；async 形态对齐）
  pub async fn to_evicted_extents(&self) -> &[u64] {
    &self.to_evicted_extents
  }

  /// 待刷大页列表（账本名；async 形态对齐）
  pub async fn to_flushed_large_pages(&self) -> &[u64] {
    &self.to_flushed_large_pages
  }

  /// 待释放 extent 列表（账本名；async 形态对齐）
  pub async fn to_free_extents(&self) -> &[(u64, u8)] {
    &self.to_free_extents
  }

  /// 驱逐 extent 偏移登记
  pub fn offset_evict_extents(&mut self, pid: u64) {
    self.to_evicted_extents.push(pid);
  }

  /// 释放 extent 偏移登记
  pub fn offset_free_extents(&mut self, pid: u64, tier: u8) {
    self.to_free_extents.push((pid, tier));
  }
}

impl Default for Transaction {
  fn default() -> Self {
    Self {
      state: TxnState::Aborted,
      start_ts_val: 0,
      commit_ts_val: 0,
      isolation: IsolationLevel::SnapshotIsolation,
      olap: false,
      wrote: false,
      has_blob_op: false,
      start_wal_cursor_val: 0,
      to_free_extents: Vec::new(),
      to_evicted_extents: Vec::new(),
      to_flushed_large_pages: Vec::new(),
      wal_pid: None,
    }
  }
}

/// 事务工厂（跨协程移交用）
pub fn new_transaction(start_ts: u64, isolation: IsolationLevel) -> Transaction {
  let mut t = Transaction::default();
  t.initialize(start_ts, isolation);
  t
}
