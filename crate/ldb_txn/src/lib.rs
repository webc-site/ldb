//! ldb_txn —— L4 事务引擎层（OSIC MVCC + WAL 钩子）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod gsn_sync;
pub mod osic_mvcc;
pub mod txn;
pub mod txn_mgr;

pub use error::{Error, Result};
pub use gsn_sync::{GsnCtx, PageGuard, advance_gsn, detect_gsn_dependency};
pub use osic_mvcc::{
  COMMIT_TREE_CAPACITY, CommitTree, ConcurrencyControl, VersionEntry, Visibility, Worker, WorkerCc,
};
pub use txn::{IsolationLevel, Transaction, TxnState, new_transaction};
pub use txn_mgr::{TransactionManager, WalTxnLogHook, barrier_ready};
