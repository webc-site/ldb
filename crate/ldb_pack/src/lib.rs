//! ldb_pack —— L0 PID↔偏移位域与 4KiB 装箱纯函数层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod bin_pack;
pub mod pid2offset;

pub use bin_pack::{
  Bin, BinSlot, MAX_PAGES_PER_BIN, ZipEntry, bin_packing_per_block, copy_compressed_pages,
};
pub use pid2offset::{
  COMP_SZ_BITS, MAX_COMP_SZ, MAX_OFFSET, OFFSET_BITS, Pid2Offset, UNCOMPRESSED,
};
