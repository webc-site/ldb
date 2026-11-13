//! ldb_stat —— L3 纯软件统计层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod metric;

pub use error::{Error, Result};
pub use metric::{Metric, StatCollector};
