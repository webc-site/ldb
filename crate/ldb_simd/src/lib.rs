//! ldb_simd —— L0 键搜索 SIMD 辅助层。

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

mod btree_simd;

pub use btree_simd::{common_prefix, get_head, make_hint, search_hint, update_hint};
