//! ldb_zipf —— L0 Zipf 分布采样层（fastrand 驱动，输出 1-based）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod zipf;

pub use zipf::{RejectionInversionZipfSampler, ScrambledZipfGenerator, ZipfGenerator};
