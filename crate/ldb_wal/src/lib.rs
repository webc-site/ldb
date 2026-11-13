//! ldb_wal —— L3 自主提交 WAL 层（WILO + 组提交 + 分片检查点）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod entry;
pub mod error;
pub mod group_commit;
pub mod guard_wal;
pub mod log_worker;

pub use entry::{
  CR_ENTRY_SIZE, DATA_ENTRY_HDR, EntryHdr, EntryType, FREE_EXTENT_HDR, LOG_ENTRY_HDR, PAGE_IMG_HDR,
  compute_chksum, read_data_fields, read_hdr, seal_chksum, validate_chksum, write_data_hdr,
  write_hdr,
};
pub use error::{Error, Result};
pub use group_commit::{GcCfg, GroupCommitExecutor};
pub use guard_wal::{
  log_pid2_offset, prepare_wal_entry, reserve_wal_entry, submit_active_wal_entry,
};
pub use log_worker::{
  CURSOR_MASK, CommitVariant, LogBuffer, LogManager, LogWorker, PreCommitTicket, StealDecision,
  ToCommitState, WAL_CIRC, WAL_WRITE_ALIGN, WalFileState, WalSink, WorkerConsistentState,
  enqueue_precommitted,
};
