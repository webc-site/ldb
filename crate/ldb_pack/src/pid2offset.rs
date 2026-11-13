//! PID→物理偏移紧凑映射位域（上游 `space_manager.h:47-70` 的 `struct PID2Offset`）。
//!
//! 64 位 = 49-bit 物理字节偏移 << 15 | 15-bit 压缩后尺寸。
//!
//! 上游 `space_manager.h:55-59` 写的是 50-bit 偏移，但 50+15=65 位溢出 u64：
//! `offset << 15` 静默截断最高位（todo.md §9 C-35），offset ≥ 2^49 时读回错误
//! 偏移。本实现改为 49 位（512 TiB 上限，覆盖任何 VPS 单盘）。

use ldb_type::PAGE_SIZE;
use static_assertions::{const_assert, const_assert_eq};

/// 偏移位宽
pub const OFFSET_BITS: u32 = 49;
/// 压缩尺寸位宽
pub const COMP_SZ_BITS: u32 = 15;
/// 偏移上限（2^49 - 1，512 TiB）
pub const MAX_OFFSET: u64 = (1u64 << OFFSET_BITS) - 1;
/// 压缩尺寸上限（2^15 - 1 = 32767，> PAGE_SIZE 以容纳「未压缩」哨兵）
pub const MAX_COMP_SZ: u16 = 0x7FFF;
/// 未压缩整页哨兵（压缩膨胀时回退存原文，见 todo.md §9 C-19）
pub const UNCOMPRESSED: u16 = PAGE_SIZE as u16;

const_assert_eq!(OFFSET_BITS + COMP_SZ_BITS, 64);
// 编码上界恰为 u64::MAX：`(2^49-1) << 15 | (2^15-1) = 2^64-1`，无溢出位；
// 等价于 `MAX_OFFSET == u64::MAX >> 15`（偏移位宽由尺寸位宽反推）。
const_assert_eq!(MAX_OFFSET, u64::MAX >> COMP_SZ_BITS);
/// 页内偏移位宽（log2 PAGE_SIZE）
const PAGE_BITS: u32 = 12;
const_assert_eq!(1usize << PAGE_BITS, PAGE_SIZE);
// 最大可编码字节数 = `MAX_OFFSET + 1` = 2^49 = 512 TiB；按 4 KiB 页折算
// 最大 PID 数 = 2^49 / 4096 = 2^37 = 137_438_953_472，任何单盘 VPS 都远超。
const_assert_eq!(
  (MAX_OFFSET + 1) / PAGE_SIZE as u64,
  1u64 << (OFFSET_BITS - PAGE_BITS)
);
// 尺寸域满宽：防止常量被改小后「未压缩」哨兵不可编码
const_assert_eq!(MAX_COMP_SZ, (1u16 << COMP_SZ_BITS) - 1);
// 未压缩哨兵必须能被 15 位尺寸域表示（否则 C-19 回退路径无法编码）
const_assert!(UNCOMPRESSED <= MAX_COMP_SZ);

/// PID→偏移位域（上游 packed struct 的 Rust 等价，零 unsafe）
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Pid2Offset {
  packed: u64,
}

impl Pid2Offset {
  /// 写入偏移与压缩尺寸。
  ///
  /// # Panics
  /// 仅在 offset 超 49 位 / comp_sz 超 15 位时 panic（上游 Ensure 等价，
  /// 调用方构造的值恒在域内）。上游对 offset 用 `& ((1<<50)-1)` 静默截断，
  /// 本实现改为硬断言（todo.md §9 C-35）。
  #[inline]
  pub fn set(&mut self, offset: u64, comp_size: u16) {
    assert!(comp_size <= MAX_COMP_SZ, "comp_size 超 15 位");
    assert!(offset <= MAX_OFFSET, "offset 超 49 位");
    self.packed = (offset << COMP_SZ_BITS) | u64::from(comp_size);
  }

  /// 物理字节偏移（49 位）
  #[inline]
  pub fn get_offset(&self) -> u64 {
    (self.packed >> COMP_SZ_BITS) & MAX_OFFSET
  }

  /// 压缩后尺寸（15 位）
  #[inline]
  pub fn get_comp_sz(&self) -> u16 {
    (self.packed & u64::from(MAX_COMP_SZ)) as u16
  }
}
