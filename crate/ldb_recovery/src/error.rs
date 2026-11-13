//! ldb_recovery 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_recovery 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// 检查点损坏（截断 / 长度不符）。
  #[error("检查点损坏")]
  CheckpointCorrupt,

  /// WAL 帧损坏（CRC）。
  #[error("WAL 帧损坏于 {0}")]
  WalCorrupt(u64),

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),
}

/// ldb_recovery 结果别名。
pub type Result<T> = result::Result<T, Error>;
