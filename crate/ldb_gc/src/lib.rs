//! ldb_gc —— L3 垃圾回收执行器层（6 策略 + NoWA 均衡）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod collector;
pub mod error;

pub use collector::{GarbageCollector, GcStrategy};
pub use error::{Error, Result};
