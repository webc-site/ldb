//! ldb_buf 错误类型。

use std::{io, result};

use ldb_type::PageId;
use thiserror::Error;

/// ldb_buf 的统一错误枚举。
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

  /// 页号超出 arena 容量（损坏映射表 / 外部输入未校验）。
  #[error("页号越界: {0}")]
  InvalidPage(PageId),

  /// 物理帧耗尽且无可驱逐页。
  #[error("缓冲池内存压力")]
  MemoryPressure,

  /// 运行时停机中。
  #[error("运行时停机")]
  ShuttingDown,
}

/// ldb_buf 结果别名。
pub type Result<T> = result::Result<T, Error>;
