//! ldb_codec —— L2 mzstd 页/Blob 编解码层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod compressor;
pub mod error;
pub mod page_codec;

pub use error::{Error, Result};
pub use page_codec::PageCodec;
