//! ldb_txn 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_txn 的统一错误枚举。
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

  /// WAL 预留失败（缓冲停机溢出）。
  #[error("WAL 预留失败")]
  WalReserveFailed,

  /// 事务状态非法（未运行中提交/回滚）。
  #[error("事务状态非法: {0}")]
  InvalidTxnState(String),
}

/// ldb_txn 结果别名。
pub type Result<T> = result::Result<T, Error>;
