//! ldb_recovery —— L4 崩溃恢复层（HistoryTree + Replayer）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod history_tree;
pub mod replayer;

pub use error::{Error, Result};
pub use history_tree::{HistoryTree, HistoryVersion};
pub use replayer::Replayer;
