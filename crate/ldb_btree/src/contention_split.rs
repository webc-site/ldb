//! CIDR'21 Contention Split 采样门（常量统一取自 `ldb_node`，消除旧版重复定义）。
//!
//! 上游 `BTreeGenericIterator::contentionSplit`
//! （`leanstore/backend/leanstore/storage/btree/core/BTreeGenericIterator.hpp:526-557`）
//! 有两层几何采样门 `cm_update_on=7` / `cm_period=14` 与阈值
//! `cm_slowpath_threshold=1`（`leanstore/backend/leanstore/Config.cpp:39-41`）。
//! 完整状态机已在 `ldb_node::contention_split` 实现，由
//! [`ldb_node::ContentionTracker::note_access`] 消费（同一 `rand` 同时用于两层门）。
//!
//! 本模块只保留写路径在进入 `note_access` 之前可用的**第一层快速门**：
//! 约 127/128 的写访问在此短路，无需为争用采样做槽位定位。旧版此处的
//! `CM_PERIOD=512` / `CM_SLOWPATH_THRESHOLD=16` 与 ldb_node 的 `14` / `1` 数值
//! 不一致（重复定义），已删除。

use ldb_node::contention_split::UPDATE_ON_EXP;

/// 第一层更新采样门：`rand & (2^cm_update_on - 1) == 0`（上游 `:532`）。
///
/// 返回 `true` 才需要继续 `ContentionTracker::note_access`（其余写访问直接跳过）。
#[inline]
pub fn update_gate(rand: u64) -> bool {
  rand & ((1u64 << UPDATE_ON_EXP) - 1) == 0
}
