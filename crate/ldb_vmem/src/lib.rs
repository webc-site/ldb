//! ldb_vmem —— L1 跨平台虚拟内存 arena 层。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod arena;

pub use arena::VirtualMemoryArena;
