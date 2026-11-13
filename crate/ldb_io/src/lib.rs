//! ldb_io —— L1 compio 跨平台异步存储后端层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod compio_backend;
pub mod doublewrite;
pub mod mem_backend;

pub use compio_backend::CompioFileBackend;
pub use doublewrite::{DWB_AREA, DWB_RECORD, Doublewrite, DwbWriter};
pub use mem_backend::MemBackend;
