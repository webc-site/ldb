//! ldb_sync —— L1 乐观锁耦合（OLC）同步原语层。
//!
//! 语义对齐上游 `ZLeanStore/src/{include/,}sync/{hybrid_latch,hybrid_guard,page_state}`：
//! 64 位状态字（高 8 位状态 / 低 56 位版本），乐观读靠版本与状态双条件校验，
//! 排他解锁推进版本。阻塞式获取在本层为 `async`（compio 单线程每核，见 todo.md §3.4）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod guard;
pub mod latch;
pub mod page_state;

pub use error::{Error, Result};
pub use guard::{ExclusiveGuard, GuardMode, HybridGuard, OptimisticGuard, PageGuard, SharedGuard};
pub use latch::{HybridLatch, latch_mode};
pub use page_state::{PageStateExt, page_mode};
