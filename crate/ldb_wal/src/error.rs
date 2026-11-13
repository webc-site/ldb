//! ldb_wal 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_wal 的统一错误枚举。外部库错误一律用 #[error(transparent)] 透明转发。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置非法。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),

  /// 页锁重试：调用方须重进乐观锁。
  #[error("需要重试")]
  Restart,

  /// WAL 缓冲溢出（停机路径复核失败，修复上游 N-15 静默溢出）。
  #[error("WAL 缓冲空间不足且运行时已停止")]
  WalOverflow,

  /// 无活动条目（submit 前未 reserve）。
  #[error("无活动日志条目")]
  NoActiveEntry,

  /// 日志损坏（CRC / 类型非法 / 截断）。
  #[error("日志损坏: {0}")]
  CorruptLog(String),
}

/// ldb_wal 结果别名。
pub type Result<T> = result::Result<T, Error>;
