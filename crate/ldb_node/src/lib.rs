//! ldb_node —— L2 B+Tree 节点布局与页内操作层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod contention_split;
pub mod error;
pub mod node;

pub use error::{Error, Result};
pub use node::{
  BTreeNode, BTreeNodeHeader, Cmp, ContentionTracker, FenceKeySlot, MEMCMP, PageSlot,
  SIZE_UNDER_FULL, SplitCtx, cmp_memcmp, sep::SeparatorInfo,
};
