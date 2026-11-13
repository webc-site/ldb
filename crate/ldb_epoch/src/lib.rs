//! ldb_epoch —— L0 基于 Epoch 的并发延迟回收层。
//!
//! # 目标要求
//!
//! 本 crate 的 epoch 计数值是 `u64`（上游 `epoch_handler.h:14` 的 `MAX_VALUE = ~0ULL`），
//! 因此要求目标提供 64 位原子操作（`target_has_atomic = "64"`）。已验证：
//! x86_64 / i686（32 位）/ aarch64 / armv7 / powerpc64（大端）/ s390x（大端）。
//! 无 64 位原子指令的 32 位目标（armv5te、mips32、riscv32 等）不支持；若需支持，
//! 应引入 `portable-atomic` 的锁降级实现，而不是把 epoch 改成 32 位（会缩短回绕周期）。

#![warn(missing_debug_implementations)]

#[cfg(not(target_has_atomic = "64"))]
compile_error!(
  "ldb_epoch requires 64-bit atomics (`target_has_atomic = \"64\"`); \
   32-bit targets without them (armv5te / mips32 / riscv32) are unsupported"
);

mod epoch;

pub use epoch::EpochHandler;
