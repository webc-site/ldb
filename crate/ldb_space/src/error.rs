//! ldb_space 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_space 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),

  /// 页压缩/解压失败（坏压缩页：映射与盘上数据不一致，上层可重试或重建）。
  #[error(transparent)]
  Codec(#[from] ldb_codec::Error),

  /// 缓冲内偏移越界（`start_ptr + 需求长度` 超过缓冲容量）：
  /// 坏映射/坏元信息立即报错，禁止越界 panic 或静默截断。
  #[error("缓冲偏移越界: start_ptr={start_ptr} 需要 {need} 字节，容量 {cap} 字节")]
  OffsetOutOfBounds {
    /// 起始偏移。
    start_ptr: u64,
    /// 需要的字节数。
    need: usize,
    /// 缓冲容量（字节）。
    cap: usize,
  },

  /// 空间耗尽（无空块且 GC 无法回收；调用方推进 GC 后重试）。
  #[error("空间耗尽")]
  SpaceExhausted,
}

/// ldb_space 结果别名。
pub type Result<T> = result::Result<T, Error>;
