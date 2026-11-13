//! ldb_blob —— L4 巨型对象管理层（ExtentList + 内容寻址去重）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod blob_mgr;
pub mod error;
pub mod extent_list;

pub use blob_mgr::{BlobManager, BlobPageStore, BlobState, NoBlobStore, PREFIX_LENGTH};
pub use error::{Error, Result};
pub use extent_list::{
  Extent, ExtentList, Iterator, MAX_EXTENTS, NO_TIERS, TailExtent, extent_size, tier_index,
};
