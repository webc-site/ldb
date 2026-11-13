//! ldb_testkit 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_testkit 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置非法。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),

  /// 注入的 I/O 失败（故障注入）。
  #[error("故障注入: {0}")]
  Injected(String),
}

/// ldb_testkit 结果别名。
pub type Result<T> = result::Result<T, Error>;
