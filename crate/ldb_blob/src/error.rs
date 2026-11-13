//! ldb_blob 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_blob 的统一错误枚举。
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

  /// 页存取端口失败（extent 页落盘 / 加载）。
  #[error("Blob 页存取失败: {0}")]
  StorePage(String),

  /// Blob 不存在。
  #[error("Blob {0} 不存在")]
  BlobNotFound(u64),

  /// 区间越界。
  #[error("Blob {0} 读取越界")]
  BlobOutOfRange(u64),

  /// extent 数超过上限（上游 `ExtentList::EXTENT_CNT_MASK = 127`；见
  /// [`crate::extent_list::MAX_EXTENTS`]）。
  #[error("extent 数超过上限 127")]
  TooManyExtents,
}

/// ldb_blob 结果别名。
pub type Result<T> = result::Result<T, Error>;
