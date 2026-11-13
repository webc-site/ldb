//! ldb_gc 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_gc 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置非法。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),

  /// 页锁重试。
  #[error("需要重试")]
  Restart,

  /// 空间耗尽。
  #[error("空间耗尽")]
  SpaceExhausted,
}

/// ldb_gc 结果别名。
pub type Result<T> = result::Result<T, Error>;
