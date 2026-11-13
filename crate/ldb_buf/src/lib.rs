//! ldb_buf —— L3 VMCache 缓冲池层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod buf_mgr;
pub mod error;

pub use buf_mgr::{
  BufEvictHook, BufferFrame, BufferManager, PageBackend, PageStateView, ShareGuard,
};
pub use error::{Error, Result};
