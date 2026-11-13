//! ldb_tuple 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_tuple 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// 重复键。
  #[error("重复键: {0:?}")]
  DuplicateKey(Vec<u8>),

  /// 状态损坏（反序列化截断等）。
  #[error("状态损坏")]
  CorruptState,

  /// 页锁重试。
  #[error("需要重试")]
  Restart,

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),
}

/// ldb_tuple 结果别名。
pub type Result<T> = result::Result<T, Error>;
