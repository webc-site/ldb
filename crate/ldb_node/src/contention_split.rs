//! CIDR'21 争用分裂的节点侧状态机。
//!
//! 上游分两处：
//! - `BufferFrame::Header::ContentionTracker`（`leanstore/backend/leanstore/
//!   storage/buffer-manager/BufferFrame.hpp:32-42`）：`last_modified_pos: s32
//!   = -1`、`restarts_counter: u32`、`access_counter: u32` + `reset()`；
//! - `BTreeGenericIterator::contentionSplit`（`.../btree/core/
//!   BTreeGenericIterator.hpp:526-557`）：两层几何采样门（`cm_update_on = 7`、
//!   `cm_period = 14`，`Config.cpp:39-40`）后按
//!   `100 * restarts / accesses >= cm_slowpath_threshold`（默认 1，
//!   `Config.cpp:41`）判定，分裂点取 `min(last_modified_pos, cur)`；
//!   `last_modified_pos = -1` 时 `trySplit` 走 `findSep` 回退。
//!
//! VMCache 世代（`ZLeanStore`/`blob`/`latency`）没有该结构，本模块是按上述
//! 上游语义重写；`longjmp` 已由 `Option<i32>` 的返回值承载（`None` 不分裂，
//! `Some(-1)` 正常选分隔键，`Some(pos)` 强制切在 `pos`）。

use crate::node::{BTreeNode, ContentionTracker};

/// `FLAGS_cm_update_on` 指数（`Config.cpp:39`，概率 1/2^7）
pub const UPDATE_ON_EXP: u32 = 7;
/// `FLAGS_cm_period` 指数（`Config.cpp:40`，概率 1/2^14）
pub const PERIOD_EXP: u32 = 14;
/// `FLAGS_cm_slowpath_threshold` 默认值（百分比，`Config.cpp:41`）
pub const SLOWPATH_THRESHOLD: u32 = 1;

impl ContentionTracker {
  /// 清空计数（上游 `ContentionTracker::reset`；`last_modified_pos` 一并置 -1）
  #[inline]
  pub fn reset(&mut self) {
    *self = Self::default();
  }

  /// 记录一次被采样到的写访问，返回争用分裂动作。
  ///
  /// 与上游 `contentionSplit` 逐行对应：
  /// - `rand & ((1<<cm_update_on)-1) == 0` 才记录；
  /// - `last_modified_pos := slot`，`restarts_counter += faced_contention`，
  ///   `access_counter += 1`；
  /// - `rand & ((1<<cm_period)-1) == 0` 才结算：归一化重启率
  ///   `100 * restarts / accesses`（整数除法，与上游 double 比较在
  ///   非负整数阈值下等价），随后清零两个计数；
  /// - 仅当 `last != slot && normalized >= threshold && count > 2` 时返回
  ///   `min(last, slot)`；`last == -1` 时返回 `Some(-1)`（上游
  ///   `favored_split_pos < 0` → `findSep` 回退）。
  ///
  /// `rand` 由调用方提供（上游 `utils::RandomGenerator::getRandU64()`），
  /// 同一 64 位值同时用于两层门 —— 这一点必须保持，否则采样相关性改变。
  #[inline]
  pub fn note_access(
    &mut self,
    slot: u32,
    faced_contention: bool,
    rand: u64,
    count: u32,
  ) -> Option<i32> {
    if rand & ((1u64 << UPDATE_ON_EXP) - 1) != 0 {
      return None;
    }
    let last = self.last_modified_pos;
    self.last_modified_pos = slot as i32;
    self.restarts_counter += u32::from(faced_contention);
    self.access_counter += 1;
    if rand & ((1u64 << PERIOD_EXP) - 1) != 0 {
      return None;
    }
    let normalized = 100 * u64::from(self.restarts_counter) / u64::from(self.access_counter);
    self.restarts_counter = 0;
    self.access_counter = 0;
    if last != slot as i32 && normalized >= u64::from(SLOWPATH_THRESHOLD) && count > 2 {
      Some(last.min(slot as i32))
    } else {
      None
    }
  }

  /// 当前窗口的归一化重启率是否达到慢路径阈值（派生谓词，供上层快速判断）。
  ///
  /// 真正的分裂决策必须走 [`ContentionTracker::note_access`] —— 上游只在
  /// `cm_period` 采样点结算一次。
  #[inline]
  fn is_hot(&self) -> bool {
    self.access_counter > 0
      && 100 * u64::from(self.restarts_counter)
        >= u64::from(SLOWPATH_THRESHOLD) * u64::from(self.access_counter)
  }
}

impl BTreeNode {
  /// 节点级热点判定（保留给 `ldb_btree` 现有接线；语义为当前窗口归一化
  /// 重启率超阈值，等价于上游 `normalized_restarts >= cm_slowpath_threshold`）。
  ///
  /// 注意：上游同名 `HybridPageGuard::hasFacedContention`
  /// （`sync-primitives/PageGuard.hpp:162`）是**单次访问**的 guard 失败标志，
  /// 不是节点谓词；本方法是为 VMCache 重设计后的派生谓词。
  #[inline]
  pub fn has_faced_contention(&self) -> bool {
    self.hdr.contention.is_hot()
  }
}
