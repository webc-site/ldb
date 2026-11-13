//! ldb_edt —— L0 死亡时间纯估算层（零分配、零状态）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod edt;

pub use edt::{
  WriteHistory, count_valid_write_histories, estimate_death_time, estimate_next_gc_write,
  get_interval_gap,
};
