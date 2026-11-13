//! ldb_codec 错误类型。

use std::result;

use thiserror::Error;

/// ldb_codec 的统一错误枚举。外部库错误一律用 #[error(transparent)] 透明转发。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置/尺寸非法（替代上游 Ensure/断言式失败）。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// 调用方给的切片短于声明长度（截断帧 / 坏元信息）。
  #[error("输入截断: 需要 {expected} 字节，实际 {got} 字节")]
  Truncated {
    /// 声明/需要的字节数。
    expected: usize,
    /// 实际可读的字节数。
    got: usize,
  },

  /// 压缩尺寸字段越界：合法值 `1..PAGE_SIZE` 或原文哨兵 `PAGE_SIZE`。
  #[error("非法压缩尺寸 {0}（合法范围 1..=4096）")]
  InvalidCompressedSize(u16),

  /// mzstd 压缩错误。
  #[error(transparent)]
  MzstdEncode(#[from] mzstd::EncodeError),

  /// mzstd 解压错误。
  #[error(transparent)]
  MzstdDecode(#[from] mzstd::DecodeError),

  /// 页解压输出长度非一页（畸形输入）。
  #[error("解压输出 {0} 字节，非 4KiB 页")]
  CorruptPage(usize),

  /// Blob 解压输出长度与元数据声明的原长不符。
  #[error("Blob 解压输出 {got} 字节，声明原长 {expected} 字节")]
  BlobSizeMismatch {
    /// 元数据声明的原长。
    expected: usize,
    /// 实际解压出的字节数。
    got: usize,
  },
}

/// ldb_codec 结果别名。
pub type Result<T> = result::Result<T, Error>;
