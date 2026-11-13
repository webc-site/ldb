//! 页守卫的 WAL 协作层（上游 `sync/page_guard/page_guard.{h,cc}` 的 WAL 成员）。
//!
//! 上游经线程局部 `TM::active_txn` 取 worker（compio 同线程多协程会串味，
//! §3.4 协议 8）→ Rust 侧全部显式传 `&Arc<LogWorker>`；
//! 调用前提：调用方持有页排他闩（ReserveDataLog 断言 EXCLUSIVE 的等价约束）。

use std::sync::Arc;

use ldb_type::PageId;

use crate::{
  entry::EntryType,
  error::Result,
  log_worker::{ActiveEntry, LogManager, LogWorker},
};

/// 预留 WAL 数据条目（锁内调用；payload 立即拷入缓冲，cursor 由 submit 前移）
pub async fn reserve_wal_entry(
  worker: &Arc<LogWorker>,
  mgr: &LogManager,
  pid: PageId,
  payload: &[u8],
) -> Result<ActiveEntry> {
  worker
    .reserve_wal_entry(mgr, EntryType::DataEntry, pid, payload)
    .await
}

/// 预备式提交（调用方已自备缓冲的零拷贝路径，上游 PrepareDataLogEntry）
pub async fn prepare_wal_entry(
  worker: &Arc<LogWorker>,
  mgr: &LogManager,
  buffer: &mut [u8],
  pid: PageId,
) -> Result<ActiveEntry> {
  worker.prepare_wal_entry(mgr, buffer, pid).await
}

/// 提交活动条目（页 X 锁释放前调用；`commit_ts` 仅 TX_COMMIT 使用）
pub async fn submit_active_wal_entry(
  worker: &Arc<LogWorker>,
  mgr: &LogManager,
  commit_ts: u64,
) -> Result<bool> {
  worker.submit_active_log_entry(mgr, commit_ts).await
}

/// PID→偏移映射变更记录（上游 `SpaceManager::LogPID2Offset` 的 WAL 侧收口）
pub async fn log_pid2_offset(worker: &Arc<LogWorker>, mgr: &LogManager, pid: PageId) -> Result<()> {
  worker
    .reserve_and_submit(mgr, EntryType::DataEntry, pid, &pid.to_le_bytes(), 0)
    .await
    .map(|_| ())
}
