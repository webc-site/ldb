//! ldb_page 错误类型（页面格式层唯一可失败路径：Delta 解析/构造）。

use std::result;

use thiserror::Error;

/// ldb_page 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// delta 缓冲不足（元数据、槽位表或载荷放不下）。
  #[error("delta 缓冲不足: 需要 {need} 字节, 实际 {actual} 字节")]
  DeltaBufferTooSmall {
    /// 需要的最小字节数
    need: usize,
    /// 实际可用字节数
    actual: usize,
  },

  /// delta 槽位越过 tuple payload。
  #[error("delta 槽位越界: offset={offset}, length={length}, payload_len={payload_len}")]
  DeltaSlotOutOfBounds {
    /// 记录内偏移
    offset: u16,
    /// 拷贝长度
    length: u16,
    /// 被更新的定长记录长度
    payload_len: usize,
  },

  /// 槽位数超过 u8 编码上限（上游 `FixedSizeDelta::count` 为 u8）。
  #[error("delta 槽位数 {0} 超过 u8 上限 255")]
  DeltaTooManySlots(usize),

  /// delta 总长超过 u16 编码上限（上游 `FixedSizeDelta::size` 为 u16）。
  #[error("delta 总长 {0} 超过 u16 上限 65535")]
  DeltaSizeOverflow(usize),
}

/// ldb_page 结果别名。
pub type Result<T> = result::Result<T, Error>;
