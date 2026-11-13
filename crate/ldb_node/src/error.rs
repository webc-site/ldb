//! ldb_node 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_node 的统一错误枚举。外部库错误一律用 #[error(transparent)] 透明转发。
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

  /// 页内容损坏（槽位越界/重叠、计数或 fence 非法、bool 字节非 0/1）。
  /// 由 `BTreeNode::validate_bytes` 在把原始页 cast 成节点前返回。
  #[error("页损坏: {0}")]
  Corrupt(String),
}

/// ldb_node 结果别名。
pub type Result<T> = result::Result<T, Error>;
