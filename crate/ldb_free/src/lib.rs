//! ldb_free —— L3 空闲空间管理层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod free_storage;

pub use error::{Error, Result};
pub use free_storage::{ExtentTier, FreeStorageManager, NO_TIERS, tier_index, tier_size};
