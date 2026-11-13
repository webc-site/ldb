//! ldb_stat 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_stat 的统一错误枚举。外部库错误一律用 #[error(transparent)] 透明转发。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置非法（替代上游 Ensure/断言式启动失败）。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),

  /// 页锁重试：调用方须重进乐观锁（替代 C++ throw Restart）。
  #[error("需要重试")]
  Restart,
}

/// ldb_stat 结果别名。
pub type Result<T> = result::Result<T, Error>;
