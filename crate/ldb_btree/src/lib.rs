//! ldb_btree —— L4 并发 B+Tree 层（OLC + Contention Split + XMerge）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod contention_split;
pub mod error;
pub mod store;
pub mod tree;

pub use error::{Error, Result};
pub use store::BufStore;
pub use tree::BTree;
