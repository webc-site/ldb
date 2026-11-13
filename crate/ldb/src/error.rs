//! ldb 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置非法。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),

  /// Blob 错误。
  #[error(transparent)]
  Blob(#[from] ldb_blob::Error),

  /// WAL 错误。
  #[error(transparent)]
  Wal(#[from] ldb_wal::Error),

  /// 事务层错误。
  #[error(transparent)]
  WalTxn(#[from] ldb_txn::Error),

  /// 缓冲池错误。
  #[error(transparent)]
  Buf(#[from] ldb_buf::Error),

  /// B+Tree 错误。
  #[error(transparent)]
  Btree(#[from] ldb_btree::Error),
}

/// ldb 结果别名。
pub type Result<T> = result::Result<T, Error>;
