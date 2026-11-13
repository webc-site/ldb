//! ldb_tuple —— L4 MVCC 元组层（ChainedTuple / FatTuple / BTreeVI）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod btree_vi;
pub mod error;

pub use btree_vi::{
  BTreeVi, ChainedTuple, DELTA_ARRAY_MAX_SIZE, Delta, DeltaArray, FatTuple, Meta, Tuple,
  UndoVersion, UpdateVersion, convert_chained_to_fat_tuple_different_attributes,
};
pub use error::{Error, Result};
