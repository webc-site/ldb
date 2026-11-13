//! ldb —— L5 顶层嵌入式数据库 Facade。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod engine;
pub mod error;

pub use engine::{Config, LeanStore, MEM_PATH};
pub use error::{Error, Result};
