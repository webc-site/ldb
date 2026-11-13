//! ldb_testkit —— L5 测试工具层（SSD 模拟器 + 故障注入后端）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod fault_inject;
pub mod ssd_sim;

pub use error::{Error, Result};
pub use fault_inject::{FaultInjectBackend, FaultKind, SharedFaultBackend};
pub use ssd_sim::{Block, Ssd, UNUSED};
