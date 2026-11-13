//! ldb_bench —— L5 基准套件层（redb 维度三方对比）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod compare;
pub mod db_bench;
pub mod error;

pub use compare::{BenchConfig, BenchDb, BenchRunner, Compression, KeyDist, PhaseResult};
pub use db_bench::{DbBench, PeerPhase, map_phase, parse_db_bench_report};
pub use error::{Error, Result};
