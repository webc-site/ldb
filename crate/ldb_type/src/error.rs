//! ldb_type 错误类型。

use std::result;

use thiserror::Error;

/// ldb_type 的统一错误枚举。
///
/// 本 crate 为 L0 零 I/O 层：不含 I/O 变体，也不转发外部库错误。
/// 唯一的失败来源是「长度不足」，出现在两个方向：
/// 折叠时调用方给的目标缓冲装不下 `v.len() + 1` 字节；
/// 解折叠时输入切片不足定长的 4 / 8 字节。
#[derive(Debug, Error)]
pub enum Error {
  /// 折叠目标缓冲容量不足、或解折叠输入切片不足（携带所需字节数）。
  #[error("长度不足: 需要 {0} 字节")]
  BufferTooSmall(usize),
}

/// ldb_type 结果别名。
pub type Result<T> = result::Result<T, Error>;
