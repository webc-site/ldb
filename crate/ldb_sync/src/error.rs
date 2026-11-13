//! ldb_sync 错误类型。

use std::result;

use thiserror::Error;

/// ldb_sync 的统一错误枚举。
///
/// L1 同步层不做 I/O、不解析配置，唯一失败模式是乐观锁重试，故只保留该变体。
#[derive(Debug, Error)]
pub enum Error {
  /// 页锁重试：调用方须重进乐观锁（替代 C++ `throw RestartException`）。
  #[error("需要重试")]
  Restart,
}

/// ldb_sync 结果别名。
pub type Result<T> = result::Result<T, Error>;
