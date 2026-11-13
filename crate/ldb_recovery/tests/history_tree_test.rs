//! HistoryTree 回归（对标上游 `HistoryTree.cpp` 的 insert/retrieve/purge/visit 语义）。
//!
//! 逐条钉死：键 = (tx_id, command_id) 精确检索（**非**快照可见性）、
//! update/remove 两树分离、purge 区间含**两端**且回调带 called_before 旧值、
//! visitRemoveVersions 只标记不删除。每个用例各配一个唯一致命变异（见报告 §E1）。

use ldb_recovery::{HistoryTree, HistoryVersion};

fn v(tx: u64, cmd: u16, dt: u16, payload: &[u8]) -> HistoryVersion {
  HistoryVersion {
    tx_id: tx,
    command_id: cmd,
    dt_id: dt,
    payload: payload.to_vec(),
    called_before: false,
  }
}

/// 键重复 = upsert 覆写（上游 `HistoryTree.cpp:81-85` DUPLICATE → removeCurrent）。
/// 唯一击杀：M13（insert → entry().or_insert()）。
#[compio::test]
async fn insert_upsert_replaces_duplicate_key() {
  let mut ht = HistoryTree::new();
  ht.insert_version(v(5, 0, 1, b"a"), false).await;
  ht.insert_version(v(5, 0, 1, b"b"), false).await;
  assert_eq!(ht.len(), 1);
  assert_eq!(
    ht.retrieve_version(5, 0, false).await.unwrap().payload,
    b"b"
  );
}

/// 检索 = (tx_id, command_id) 精确 seek 对应树；update/remove 两树独立；
/// 不得「回退另一树」，不得「<= snapshot 最近可见」。
/// 唯一击杀：M11（tree() 恒返回 update）、M14（retrieve 回退另一树）。
#[compio::test]
async fn retrieve_is_exact_key_and_trees_are_separate() {
  let mut ht = HistoryTree::new();
  ht.insert_version(v(5, 0, 1, b"upd"), false).await;
  ht.insert_version(v(5, 0, 2, b"del"), true).await;
  ht.insert_version(v(7, 0, 1, b"upd-only"), false).await;
  assert_eq!(ht.len(), 3);
  assert_eq!(
    ht.retrieve_version(5, 0, false).await.unwrap().payload,
    b"upd"
  );
  assert_eq!(
    ht.retrieve_version(5, 0, true).await.unwrap().payload,
    b"del"
  );
  // (7,0) 只在 update 树 → 以 remove 树检索须 None（不得回退；无「最近<=7」语义）
  assert!(ht.retrieve_version(7, 0, true).await.is_none());
  assert!(ht.retrieve_version(6, 0, false).await.is_none());
  // command_id 区分同事务两次写
  assert!(ht.retrieve_version(5, 1, false).await.is_none());
}

/// purge：先删 remove 树并逐条回调（带旧 called_before），再删 update 树（无回调）；
/// 返回删除总数。区间外保留。
/// 唯一击杀：M16（update 树不再删除）。
#[compio::test]
async fn purge_removes_remove_and_update_trees() {
  let mut ht = HistoryTree::new();
  ht.insert_version(v(10, 0, 1, b"r10"), true).await;
  ht.insert_version(v(11, 0, 1, b"r11"), true).await;
  ht.insert_version(v(10, 0, 1, b"u10"), false).await;
  ht.insert_version(v(11, 0, 1, b"u11"), false).await;
  ht.insert_version(v(2000, 0, 1, b"keep"), true).await; // 区间外

  let mut calls: Vec<(u64, bool)> = Vec::new();
  let n = ht
    .purge_versions(0, 1000, |ver, before| calls.push((ver.tx_id, before)))
    .await;
  assert_eq!(n, 4, "2 remove + 2 update");
  assert_eq!(calls, vec![(10, false), (11, false)], "仅 remove 树回调");
  assert_eq!(ht.len(), 1, "仅区间外 tx2000 残留");
}

/// purge 区间含两端：purge(10,10) 恰删 tx==10。
/// 唯一击杀：M27（remove 树下界 from → from+1）。
#[compio::test]
async fn purge_range_inclusive_bounds() {
  let mut ht = HistoryTree::new();
  for tx in [9u64, 10, 11] {
    ht.insert_version(v(tx, 0, 1, b"x"), true).await;
  }
  let n = ht.purge_versions(10, 10, |_, _| {}).await;
  assert_eq!(n, 1);
  assert_eq!(ht.len(), 2, "9 与 11 保留");
}

/// 同一 tx_id 的多个 command_id 都落在键区间内。
/// 唯一击杀：M19（remove 树上界收窄到 command 0）。
#[compio::test]
async fn purge_covers_all_command_ids_of_tx() {
  let mut ht = HistoryTree::new();
  ht.insert_version(v(5, 0, 1, b"c0"), true).await;
  ht.insert_version(v(5, 7, 1, b"c7"), true).await;
  let n = ht.purge_versions(0, 5, |_, _| {}).await;
  assert_eq!(n, 2);
  assert!(ht.is_empty());
}

/// purge 回调读到 visit 置过的 called_before=true（旧值语义）。
/// 唯一击杀：M15（purge 回调恒传 false）。
#[compio::test]
async fn purge_callback_reports_prior_visit() {
  let mut ht = HistoryTree::new();
  ht.insert_version(v(10, 0, 1, b"r"), true).await;
  ht.visit_remove_versions(10, 10, |_, _| {}).await;
  let mut seen = Vec::new();
  ht.purge_versions(0, 10, |_, before| seen.push(before))
    .await;
  assert_eq!(seen, vec![true]);
}

/// visit 只标记 called_before 并回调旧值，**不删除**；返回访问数。
/// 唯一击杀：M17（visit 计数 n += 0）。
#[compio::test]
async fn visit_remove_versions_flags_without_deleting() {
  let mut ht = HistoryTree::new();
  ht.insert_version(v(5, 0, 1, b"a"), true).await;
  ht.insert_version(v(5, 1, 1, b"b"), true).await;
  // 同区间另有 update 版本（不影响 remove 树访问计数）
  ht.insert_version(v(5, 0, 1, b"ua"), false).await;
  ht.insert_version(v(5, 1, 1, b"ub"), false).await;

  let mut first: Vec<(u16, bool)> = Vec::new();
  let n = ht
    .visit_remove_versions(5, 5, |ver, before| first.push((ver.command_id, before)))
    .await;
  assert_eq!(n, 2);
  assert_eq!(first, vec![(0, false), (1, false)]);
  assert_eq!(ht.len(), 4, "visit 不删除");

  let mut second: Vec<(u16, bool)> = Vec::new();
  ht.visit_remove_versions(5, 5, |ver, before| second.push((ver.command_id, before)))
    .await;
  assert_eq!(second, vec![(0, true), (1, true)], "已访问过，旧值 true");
}

/// visit 只看 remove 树，不动 update 树（非对称内容使错树可见）。
/// 唯一击杀：M18b（visit 区间 from/to 交换）。
#[compio::test]
async fn visit_only_touches_remove_tree() {
  let mut ht = HistoryTree::new();
  // update 树 3 条（tx 5/6/7），remove 树 1 条（tx 100）
  for tx in [5u64, 6, 7] {
    ht.insert_version(v(tx, 0, 1, b"u"), false).await;
  }
  ht.insert_version(v(100, 0, 1, b"r"), true).await;
  let mut seen = Vec::new();
  ht.visit_remove_versions(5, 100, |ver, _| seen.push(ver.tx_id))
    .await;
  assert_eq!(seen, vec![100], "仅 remove 树的 tx100");
  assert_eq!(ht.len(), 4, "两树均未删");
  assert_eq!(
    ht.retrieve_version(5, 0, false).await.unwrap().payload,
    b"u"
  );
}
