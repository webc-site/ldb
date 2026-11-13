//! HistoryTree（每 worker 的 Undo/删除 历史存储；上游
//! `leanstore/backend/leanstore/concurrency-recovery/HistoryTree.{hpp,cpp}`）。
//!
//! 上游把版本按 `is_remove` 分存两棵 BTreeLL：`update_btrees[worker]`（Undo 版本）
//! 与 `remove_btrees[worker]`（删除墓碑）。键 = `fold(tx_id) ++ fold(command_id)`
//! （`HistoryTree.cpp:34-38`），LE 定宽整数折叠 ⇒ 键序等价于 `(tx_id, command_id)`
//! 字典序。版本值 = `VersionMeta { called_before, dt_id, payload[] }`
//! （`HistoryTree.hpp:25-29`）。
//!
//! 本实现以单一 `HistoryTree` 承载**一个 worker** 的两棵树（worker 维度由调用方按
//! worker 实例化，与上游每 worker 一份的数组同构）。**只做版本存储，不做快照可见性**
//! —— 可见性（`is_visible_for_me`/LCB）由 `ldb_txn` 的 `CommitTree` 负责（职责边界，
//! 见报告 §F1），避免与 ldb_txn 冻结语义重复。

use std::collections::BTreeMap;

/// 版本键（上游 `fold(tx_id) ++ fold(command_id)`；LE 定宽 ⇒ 字典序等价）
type VersionKey = (u64, u16);

/// 版本记录（上游 `VersionMeta`：dt_id + payload + called_before 持久标志）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryVersion {
  /// 事务 ID
  pub tx_id: u64,
  /// 命令 ID（同一事务内多次写的区分；上游 key 的第二分量）
  pub command_id: u16,
  /// 表 ID
  pub dt_id: u16,
  /// 载荷
  pub payload: Vec<u8>,
  /// 「此前已被访问」标志（上游 `VersionMeta::called_before`）；purge/visit 时
  /// 读出旧值传给回调，并置 true —— 供 graveyard 迁移幂等判定
  pub called_before: bool,
}

/// 历史树（单 worker：update_btrees + remove_btrees 两棵有序树）
#[derive(Debug, Default)]
pub struct HistoryTree {
  /// Undo 版本树（`is_remove == false`）
  update: BTreeMap<VersionKey, HistoryVersion>,
  /// 删除版本树（`is_remove == true`，graveyard 迁移源）
  remove: BTreeMap<VersionKey, HistoryVersion>,
}

impl HistoryTree {
  /// 构造
  pub fn new() -> Self {
    Self::default()
  }

  /// 选择目标树
  #[inline]
  fn tree_mut(&mut self, is_remove: bool) -> &mut BTreeMap<VersionKey, HistoryVersion> {
    if is_remove {
      &mut self.remove
    } else {
      &mut self.update
    }
  }

  /// 选择目标树（只读）
  #[inline]
  fn tree(&self, is_remove: bool) -> &BTreeMap<VersionKey, HistoryVersion> {
    if is_remove {
      &self.remove
    } else {
      &self.update
    }
  }

  /// 插入版本（上游 `insertVersion`：按 `is_remove` 选树，键重复即 upsert 覆写）
  pub async fn insert_version(&mut self, v: HistoryVersion, is_remove: bool) {
    self.tree_mut(is_remove).insert((v.tx_id, v.command_id), v);
  }

  /// 检索版本（上游 `retrieveVersion`：按 (tx_id, command_id) **精确** seek 对应树；
  /// 不按快照过滤 —— 快照可见性属 `ldb_txn`）
  pub async fn retrieve_version(
    &self,
    tx_id: u64,
    command_id: u16,
    is_remove: bool,
  ) -> Option<&HistoryVersion> {
    self.tree(is_remove).get(&(tx_id, command_id))
  }

  /// 净化版本（上游 `purgeVersions`：`[from, to]` 闭区间）。
  ///
  /// 顺序与上游一致：先删 `remove` 树并对每条版本回调 `(version, called_before)`
  /// （`called_before` 为**读取到的旧值**，删前读出），再删 `update` 树（无回调）。
  /// 返回被删除的版本总数。
  pub async fn purge_versions(
    &mut self,
    from: u64,
    to: u64,
    mut cb: impl FnMut(&HistoryVersion, bool),
  ) -> usize {
    let mut n = 0usize;
    // 先收集 remove 树的键（避免边遍历边删借用冲突）
    let victims: Vec<VersionKey> = self
      .remove
      .range((from, u16::MIN)..=(to, u16::MAX))
      .map(|(k, _)| *k)
      .collect();
    for k in victims {
      if let Some(v) = self.remove.remove(&k) {
        cb(&v, v.called_before);
        n += 1;
      }
    }
    // update 树：上游整页 reset，语义等价于删掉区间内全部版本（无回调）
    let victims: Vec<VersionKey> = self
      .update
      .range((from, u16::MIN)..=(to, u16::MAX))
      .map(|(k, _)| *k)
      .collect();
    n += victims.len();
    for k in victims {
      self.update.remove(&k);
    }
    n
  }

  /// 访问删除版本（上游 `visitRemoveVersions`：`[from, to]` 闭区间，**不删除**）。
  ///
  /// 逐条把 `called_before` 置 true 并回调 `(version, 旧 called_before)` —— 供
  /// graveyard 迁移读取删除载荷。返回被访问的版本数。
  /// （上游对 `called_before == true` 有 `ensure(...== false)`；本实现不 panic，
  /// 把旧值交给回调由调用方判定，见报告 NOTES。）
  pub async fn visit_remove_versions(
    &mut self,
    from: u64,
    to: u64,
    mut cb: impl FnMut(&HistoryVersion, bool),
  ) -> usize {
    let keys: Vec<VersionKey> = self
      .remove
      .range((from, u16::MIN)..=(to, u16::MAX))
      .map(|(k, _)| *k)
      .collect();
    let mut n = 0usize;
    for k in keys {
      if let Some(v) = self.remove.get_mut(&k) {
        let before = v.called_before;
        v.called_before = true;
        cb(v, before);
        n += 1;
      }
    }
    n
  }

  /// 条目数（两棵树合计）
  pub fn len(&self) -> usize {
    self.update.len() + self.remove.len()
  }

  /// 是否为空
  pub fn is_empty(&self) -> bool {
    self.update.is_empty() && self.remove.is_empty()
  }
}
