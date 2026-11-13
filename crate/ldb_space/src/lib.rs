//! ldb_space —— L3 异地写空间管理层（ZLeanStore SpaceManager 收敛版）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod edt;
pub mod error;
pub mod page_io;
pub mod space_mgr;
pub mod state;

pub use edt::EdtTracker;
pub use error::{Error, Result};
pub use page_io::{Bins, PageIo, WriteBufferGroup, ZipInfo, ZipPage};
pub use space_mgr::{MAX_REUSABLE_PIDS, NoTxnLog, SpaceManager};
pub use state::{BlockMetadata, BlockState, MutexVec, Offset2Pids, STATE_CNT, StorageSpace};
