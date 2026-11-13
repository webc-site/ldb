//! ldb_range —— L0 一维区间互斥锁层（跳表 + epoch 回收）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

mod range_lock;

pub use range_lock::RangeLock;
