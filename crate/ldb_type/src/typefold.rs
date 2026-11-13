//! 字典序保序类型折叠（移植自 `ZLeanStore/share_libs/typefold/typefold.cc`）。
//!
//! 折叠规则（输出为大端 + 符号位翻转，字节序比较等价于数值比较）：
//! - `Integer`(i32)：`bswap32(x ^ (1<<31))`
//! - `UInteger`(u32)：`bswap32(x)`
//! - `Timestamp`(i64)：`bswap64(x ^ (1<<63))`
//! - `UniqueID`(u64)：`bswap64(x)`
//! - `Varchar`：原样拷贝 + NUL 终止（上游 `x.length == Leng` 时写越界 1 字节的
//!   缺陷已修复：本实现运行时校验缓冲区容量；`v.len() + 1` 超出 u16 域同样返回
//!   `BufferTooSmall`，不做上游 `uint16_t` 静默截断）。
//!
//! 上游不存在 u8/u16/i8/i16/f32/f64/bytes 折叠（早期规划捏造的 22 个函数已删除）。

use memchr::memchr;

use crate::error::{Error, Result};

#[inline]
fn put32(writer: &mut [u8], bytes: [u8; 4]) -> Result<u16> {
  // get_mut(..4) 成功时长度恰为 4，无需再 [..4] 重切一次边界检查
  writer
    .get_mut(..4)
    .ok_or(Error::BufferTooSmall(4))?
    .copy_from_slice(&bytes);
  Ok(4)
}

#[inline]
fn put64(writer: &mut [u8], bytes: [u8; 8]) -> Result<u16> {
  writer
    .get_mut(..8)
    .ok_or(Error::BufferTooSmall(8))?
    .copy_from_slice(&bytes);
  Ok(8)
}

#[inline]
fn take32(input: &[u8]) -> Result<[u8; 4]> {
  let b = input.get(..4).ok_or(Error::BufferTooSmall(4))?;
  let mut out = [0; 4];
  out.copy_from_slice(b);
  Ok(out)
}

#[inline]
fn take64(input: &[u8]) -> Result<[u8; 8]> {
  let b = input.get(..8).ok_or(Error::BufferTooSmall(8))?;
  let mut out = [0; 8];
  out.copy_from_slice(b);
  Ok(out)
}

/// i32 → 大端 + 符号位翻转
#[inline]
pub fn fold_integer(writer: &mut [u8], x: i32) -> Result<u16> {
  put32(writer, (x as u32 ^ (1 << 31)).to_be_bytes())
}

/// u32 → 大端
#[inline]
pub fn fold_uinteger(writer: &mut [u8], x: u32) -> Result<u16> {
  put32(writer, x.to_be_bytes())
}

/// i64 → 大端 + 符号位翻转
#[inline]
pub fn fold_timestamp(writer: &mut [u8], x: i64) -> Result<u16> {
  put64(writer, (x as u64 ^ (1 << 63)).to_be_bytes())
}

/// u64 → 大端
#[inline]
pub fn fold_unique_id(writer: &mut [u8], x: u64) -> Result<u16> {
  put64(writer, x.to_be_bytes())
}

/// 变长字节串 → 原样 + NUL 终止，返回写入长度（`v.len() + 1`）
///
/// 两个边界都返回 [`Error::BufferTooSmall`] 而非静默出错：目标缓冲装不下
/// `v.len() + 1` 字节（上游 `x.length == Leng` 的越界写缺陷），以及 `v.len() + 1`
/// 超出 u16 域（上游返回 `uint16_t`，截断会让调用方游标与实际写入字节数不一致）。
#[inline]
pub fn fold_varchar(writer: &mut [u8], v: &[u8]) -> Result<u16> {
  // usize +1 不可能溢出；n 须同时落在 u16 域（上游返回 uint16_t）与目标缓冲内
  let n = v.len() + 1;
  if n > u16::MAX as usize {
    return Err(Error::BufferTooSmall(n));
  }
  let dst = writer.get_mut(..n).ok_or(Error::BufferTooSmall(n))?;
  dst[..v.len()].copy_from_slice(v);
  dst[v.len()] = 0;
  Ok(n as u16)
}

/// 大端 + 符号位翻转 → i32
#[inline]
pub fn unfold_integer(input: &[u8]) -> Result<i32> {
  Ok(u32::from_be_bytes(take32(input)?) as i32 ^ (1i32 << 31))
}

/// 大端 → u32
#[inline]
pub fn unfold_uinteger(input: &[u8]) -> Result<u32> {
  Ok(u32::from_be_bytes(take32(input)?))
}

/// 大端 + 符号位翻转 → i64
#[inline]
pub fn unfold_timestamp(input: &[u8]) -> Result<i64> {
  Ok(u64::from_be_bytes(take64(input)?) as i64 ^ (1i64 << 63))
}

/// 大端 → u64
#[inline]
pub fn unfold_unique_id(input: &[u8]) -> Result<u64> {
  Ok(u64::from_be_bytes(take64(input)?))
}

/// NUL 终止字节串 → 切片（不含 NUL）
#[inline]
pub fn unfold_varchar(input: &[u8]) -> Result<&[u8]> {
  // memchr_zero 已返回「NUL 位置或 len」，两种情况都可直接作切片终点
  Ok(&input[..memchr_zero(input)])
}

/// 首个 NUL 的偏移（未找到返回 `input.len()`）
///
/// 用 SIMD 字节扫描库 `memchr`（`.agents/skills/rust_review/SKILL.md` §9 指定库），语义与
/// `input.iter().position(|&b| b == 0).unwrap_or(input.len())` 逐值等价。
/// 上游 `Fold`/`Unfold` 只显式实例化 `Varchar<16>` 与 `Varchar<128>` 两档
/// （`typefold.cc:70-73`），实测这两档手写标量循环比 memchr 慢 3.6x / 16x
/// （round8 基准：16 B 3.10→0.81 ns，128 B 47.98→3.77 ns），故不再自造扫描。
#[inline]
fn memchr_zero(input: &[u8]) -> usize {
  memchr(0, input).unwrap_or(input.len())
}
