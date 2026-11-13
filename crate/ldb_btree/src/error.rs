//! ldb_btree 错误类型。

use std::{io, result};

use thiserror::Error;

/// ldb_btree 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置非法。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// I/O 错误。
  #[error(transparent)]
  Io(#[from] io::Error),

  /// 缓冲池错误（`ldb_buf`；仅落盘接线路径产生）。
  #[error(transparent)]
  Buf(#[from] ldb_buf::Error),

  /// 页锁重试（jumpmu Result 化）。
  #[error("需要重试")]
  Restart,

  /// 树结构损坏（页缺失 / 父指针断裂）。
  #[error("树损坏: {0}")]
  CorruptTree(String),

  /// 单条记录超过 `MAX_RECORD_SIZE`（上游 `BTree::Insert/Update` 前置断言）。
  ///
  /// 超过页内单条访问单元上限的记录永远放不下，若不在入口拒绝，分裂路径会因
  /// `has_space_for_kv` 恒 false 而无限重试（旧版 `insert` 死循环根因）。
  #[error("记录过大: key+payload={size} 字节 > MAX_RECORD_SIZE={max}")]
  RecordTooLarge {
    /// 实际 key+payload 字节数
    size: u32,
    /// `ldb_node::MAX_RECORD_SIZE`
    max: u32,
  },
}

/// ldb_btree 结果别名。
pub type Result<T> = result::Result<T, Error>;
