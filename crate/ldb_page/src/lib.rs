//! ldb_page —— L2 页面格式与依赖倒置 trait 层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod delta;
pub mod error;
pub mod hook;
pub mod large_page;
pub mod page;

pub use delta::{DeltaSlot, META_FIX, build_delta_header, metadata_size, update_delta_payload};
pub use error::{Error, Result};
pub use hook::{EvictHook, GcHook, TxnLogHook};
pub use large_page::{LargePage, LargePageTier};
pub use page::{MetadataPage, PAGE_BODY, Page, PageHeader, ROOT_SLOTS};
