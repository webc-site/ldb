//! ldb_type —— L0 基础类型与纯算法层。
//!
//! 分层约束：零 I/O、零阻塞，仅可被同层与上层依赖（todo.md §3.0 层内 DAG）。

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod page_state;
pub mod typefold;
pub mod watermark;

pub use error::{Error, Result};
pub use page_state::{
  PageState, PageStateMode, VERSION_MASK, next_version_new_state, same_version_new_state,
};
pub use watermark::GsnWatermark;

/// 页大小（字节，上游 `constants.h:9`）
pub const PAGE_SIZE: usize = 4096;
/// 参与装箱的最小压缩页尺寸（上游 `constants.h:24`）
pub const MIN_COMP_SIZE: u16 = 2048;
/// 页写入历史定长（上游 `constants.h:25`）
pub const WRITE_HISTORY_LENGTH: usize = 8;
/// B+Tree 16-Hint 数组长度（上游 `node.h:82`）
pub const HINT_COUNT: usize = 16;

/// 页 ID（虚拟地址空间的槽位序号）
pub type PageId = u64;
/// 日志序列号（WAL LSN / 全局 GSN 共用编码）
pub type LogId = u64;
/// 逻辑擦除块 ID
pub type BlockId = u64;
/// Worker 线程 / 协程编号
pub type WorkerId = u16;
/// 事务 ID
pub type TxnId = u64;
/// 事务内命令序号
pub type CommandId = u32;
/// 索引（表）ID
pub type IndexId = u16;
/// NULL 页哨兵（上游 `buffer_manager.cc:73` 无括号宏的修复，见 todo.md §9 C-26）
pub const NULL_PID: PageId = u64::MAX - 1;

/// 比较算子（替代上游外观层纯虚 `KVInterface::SetComparisonOperator` 的动态分发，
/// 见 todo.md §3.0 反向边 7：不下沉 dyn，拆为枚举 + 泛型闭包）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
  /// memcmp 字典序（唯一支持前缀压缩的路径）
  Memcmp,
  /// 调用方提供的全序比较
  Custom,
}

/// 键比较语义：算子标记 + 全序比较闭包
#[derive(Debug, Clone, Copy)]
pub struct ComparisonLambda<F> {
  /// 算子（决定前缀压缩可用性）
  pub op: ComparisonOperator,
  /// 比较函数
  pub func: F,
}
