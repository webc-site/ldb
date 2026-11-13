//! ldb_clock —— L0 常驻页集合与 Clock 置换层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod resident_set;

pub use resident_set::{EMPTY, ResidentPageSet, TOMBSTONE};
