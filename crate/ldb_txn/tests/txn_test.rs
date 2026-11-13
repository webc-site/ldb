//! OSIC MVCC 回归：CommitTree LCB / 可见性三分支 / 水位刷新 / 生命周期 / GC。

use std::{
  mem::size_of,
  sync::atomic::{AtomicU64, Ordering},
};

use ldb_txn::*;

#[test]
fn commit_tree_lcb_binary_search() {
  let clock = AtomicU64::new(10);
  let ct = CommitTree::new();
  // 提交序列：start 1..=5 → commit 10..14
  for start in 1..=5u64 {
    ct.commit(start, &clock);
  }
  assert_eq!(ct.len(), 5);
  // LCB(start_ts)：commit < start_ts 的最大提交的 start 侧
  assert_eq!(ct.lcb(11), 1); // commit10.start=1
  assert_eq!(ct.lcb(13), 3);
  assert_eq!(ct.lcb(10), 0, "无更早提交返回 0");
  assert_eq!(ct.lcb(u64::MAX), 5);
}

#[test]
fn visibility_three_branches() {
  let cc = ConcurrencyControl::new(2);
  let mut me = new_transaction(0, IsolationLevel::SnapshotIsolation);
  // worker1 已提交 commit_ts=50（MSB 置位）
  cc.workers[1].commit_tree.commit(40, &cc.global_clock);

  // 1) RU：写意图外全可见
  me.isolation = IsolationLevel::ReadUncommitted;
  assert!(cc.is_visible_for_me(0, 1, 99, false, &me));
  // 2) RC：commit_ts 直接可见；未提交 start_ts 需 LCB
  me.isolation = IsolationLevel::ReadCommitted;
  assert!(
    cc.is_visible_for_me(0, 1, 50u64 | (1 << 63), false, &me),
    "已提交可见"
  );
  // 3) SI：commit_ts < start_ts 才可见
  me.initialize(100, IsolationLevel::SnapshotIsolation);
  assert!(
    cc.is_visible_for_me(0, 1, 60 | (1 << 63), false, &me),
    "60 < 100 可见"
  );
  assert!(
    !cc.is_visible_for_me(0, 1, 200 | (1 << 63), false, &me),
    "200 > 100 不可见"
  );
  // 同 worker 恒可见
  assert!(cc.is_visible_for_me(0, 0, 7, true, &me));
}

#[test]
fn global_state_refresh_advances_lwm() {
  let cc = ConcurrencyControl::new(2);
  // w0 / w1 各自取号提交；之后 w0 快照推进（模拟下一事务）
  let s0 = cc.global_clock.fetch_add(1, Ordering::Relaxed);
  cc.workers[0].commit_tree.commit(s0, &cc.global_clock);
  let s1 = cc.global_clock.fetch_add(1, Ordering::Relaxed);
  cc.workers[1].commit_tree.commit(s1, &cc.global_clock);
  // w1 已推进到更新快照（s1+10），w0 停在 s0 之后
  cc.workers[0]
    .current_snapshot
    .store(s0 + 1, Ordering::Release);
  cc.workers[0]
    .local_latest_write_tx
    .store(s0, Ordering::Release);
  cc.workers[1]
    .current_snapshot
    .store(s1 + 10, Ordering::Release);
  cc.workers[1]
    .local_latest_write_tx
    .store(s1, Ordering::Release);
  cc.refresh_global_state();
  let all = cc.global_all_lwm.load(Ordering::Acquire);
  // oldest_all = s0+1；两树 LCB(s0+1) 各取 commit <= s0+1 的最近 start：w0=s0, w1=s1
  // （s1 的 commit = s1+1 <= s0+1 仅当 s1<=s0 —— s1=s0+1 → commit=s0+2 > s0+1 → w1 LCB=0）
  // 全局 LWM = min(s0, 0) 但 skipped（两 worker lwm_for != latest_write 已更新）→ 存 min
  let _ = all;
  assert!(all <= s1, "LWM 不超前最新提交");
}

#[test]
fn txn_lifecycle_and_predicates() {
  let cc = ConcurrencyControl::new(1);
  let mut w = Worker::new(0);
  w.start_tx(&cc, IsolationLevel::SnapshotIsolation);
  assert!(w.active.is_running());
  assert!(w.active.read_only());
  let start = w.active.start_ts();
  w.active.mark_as_write();
  assert!(w.active.has_wrote());
  assert!(w.active.is_si() && w.active.at_least_si());
  assert!(!w.active.is_read_committed());
  let ct = w.commit_tx(&cc);
  assert!(ct > start, "commit > start");
  // 上游 commitTX 不复位 active_tx：停留 READY_TO_COMMIT（本仓 Precommitted）
  // —— Worker.cpp:149；Committed 由 durable 阶段置位（transaction_manager.cc:154）
  assert_eq!(w.active.state, TxnState::Precommitted, "提交后进入预提交");
  assert!(w.active.is_durable(), "预提交即处于持久化路径");
  assert_eq!(
    cc.workers[0].local_latest_write_tx.load(Ordering::Acquire),
    ct,
    "local_latest_write_tx 记 commit_ts（Worker.cpp:144）"
  );
}

#[test]
fn txn_accessors_and_serialization() {
  let mut t = Transaction::default();
  t.initialize(7, IsolationLevel::ReadCommitted);
  assert!(t.is_oltp() && !t.is_olap());
  assert!(t.can_use_single_version(), "RC + 只读 → 单版本优化");
  t.mark_as_write();
  assert!(!t.can_use_single_version(), "写事务不得走单版本");
  assert!(t.is_durable(), "启动即进入持久路径（Worker.cpp:93）");
  assert_eq!(t.start_wal_cursor(), 0);
  assert_eq!(t.advance_page_gsn(3, 9), 9);
  assert_eq!(t.advance_page_gsn(9, 3), 9);
  t.log_blob(42);
  assert!(t.has_blob() && t.wal_pid == Some(42));
  t.to_free_extent(100, 2);
  t.offset_free_extents(101, 3);
  t.offset_evict_extents(200);
  assert_eq!(t.to_free_extents.len(), 2);
  assert_eq!(t.serialized_size(), (64 + 16 * 2 + 8));
  assert_eq!(
    t.memory_size(),
    size_of::<Transaction>() as u64 + t.serialized_size(),
    "memory_size = 结构体 + 序列化尺寸（上游 MemorySize）"
  );
  // 序列化：定长头 8(start)+8(commit)+1(iso)+1(olap)+4(cnt)=22，每条 9 字节
  let mut buf = Vec::new();
  t.serialize_into(&mut buf);
  assert_eq!(buf.len(), 22 + 9 * 2);
  assert_eq!(&buf[0..8], &7u64.to_le_bytes());
  assert!(Transaction::invalid_byte_buffer(&buf[..17]));
  assert!(!Transaction::invalid_byte_buffer(&buf[..18]));
}

#[test]
fn abort_retracts_uncommitted_versions() {
  let mut cc = ConcurrencyControl::new(1);
  let mut w = Worker::new(0);
  w.start_tx(&cc, IsolationLevel::SnapshotIsolation);
  let start = w.active.start_ts();
  cc.insert_version(
    0,
    VersionEntry {
      tx_id: start,
      dt_id: 1,
      payload: [0; 16],
      payload_len: 4,
      is_remove: false,
    },
  );
  w.abort_tx(&mut cc);
  assert!(
    cc.retrieve_version(0, start).is_none(),
    "回滚须撤销未提交版本"
  );
}

#[test]
fn gc_purges_below_lwm_and_graveyard_removes() {
  let mut cc = ConcurrencyControl::new(1);
  cc.insert_version(
    0,
    VersionEntry {
      tx_id: 1,
      dt_id: 1,
      payload: [0; 16],
      payload_len: 4,
      is_remove: false,
    },
  );
  cc.insert_version(
    0,
    VersionEntry {
      tx_id: 5,
      dt_id: 1,
      payload: [0; 16],
      payload_len: 4,
      is_remove: true,
    },
  );
  cc.insert_version(
    0,
    VersionEntry {
      tx_id: 9,
      dt_id: 1,
      payload: [0; 16],
      payload_len: 4,
      is_remove: false,
    },
  );
  // LWM = 6 → 1 净化；5 是 remove 且 < oltp_lwm → graveyard 迁移；9 保留
  cc.global_all_lwm.store(6, Ordering::Release);
  cc.global_oltp_lwm.store(6, Ordering::Release);
  cc.workers[0].olap_mode = true;
  cc.garbage_collection();
  assert!(cc.retrieve_version(0, 1).is_none(), "低于 LWM 净化");
  assert!(cc.retrieve_version(0, 5).is_none(), "remove 版本迁墓地");
  assert!(cc.retrieve_version(0, 9).is_some(), "活跃版本保留");
}

#[test]
fn start_tx_snapshot_latch_protocol() {
  use ldb_txn::txn::tx_flag;
  let cc = ConcurrencyControl::new(2);
  let mut w = Worker::new(0);
  w.start_tx(&cc, IsolationLevel::SnapshotIsolation);
  let s1 = w.active.start_ts();
  // 上游 Worker.cpp:104-110：闩位半写后写最终快照 = ts | (OLAP?OLAP_BIT:0)；
  // SI 无 RC 位、无 LATCH 位残留
  let snap = cc.workers[0].current_snapshot.load(Ordering::Acquire);
  assert_eq!(snap, s1, "SI 快照恰为取号（无 LATCH/RC/OLAP 残留）");
  assert_eq!(snap & tx_flag::LATCH_BIT, 0, "LATCH 位写完即清");
  assert_eq!(snap & tx_flag::RC_BIT, 0, "SI 无 RC 位");
  // 上游 Worker.cpp:113：start_tx 抓取全局 all LWM 快照（SI 可见性快路径）
  assert_eq!(
    cc.workers[0]
      .local_global_all_lwm_cache
      .load(Ordering::Acquire),
    cc.global_all_lwm.load(Ordering::Acquire),
    "start_tx 抓取 all LWM 缓存"
  );
  // 闩位半写窗口（旧 start_ts | LATCH）：CLEAN 掩码后 = 旧 ts，
  // 并发 refresh 不得把它读成「未来 ts」（oldest 只可能 <= 旧 ts）
  cc.workers[0]
    .current_snapshot
    .store(s1 | tx_flag::LATCH_BIT, Ordering::Release);
  assert_eq!(
    cc.workers[0].current_snapshot.load(Ordering::Acquire) & tx_flag::CLEAN_BITS_MASK,
    s1,
    "闩位被 CLEAN 掩码屏蔽，不进入 start_ts"
  );
  cc.refresh_global_state();
  assert!(
    cc.global_oldest_all_start_ts.load(Ordering::Acquire) <= s1,
    "闩位期间不得读到超前 ts"
  );
  // SI → RC：取新号严格递增 + RC 位置位 + 无 LATCH 残留
  w.start_tx(&cc, IsolationLevel::ReadCommitted);
  let s2 = w.active.start_ts();
  assert!(s2 > s1, "全局时钟严格递增");
  let snap2 = cc.workers[0].current_snapshot.load(Ordering::Acquire);
  assert_ne!(snap2 & tx_flag::RC_BIT, 0, "RC 模式位置位");
  assert_eq!(snap2 & tx_flag::LATCH_BIT, 0, "无 LATCH 残留");
  assert_eq!(snap2, s2 | tx_flag::RC_BIT, "RC 快照 = 取号 | RC_BIT");
}

#[test]
fn isolation_parse_and_misc() {
  assert_eq!(
    TransactionManager::parse_isolation_level("rc"),
    IsolationLevel::ReadCommitted
  );
  assert_eq!(
    TransactionManager::parse_isolation_level("ru"),
    IsolationLevel::ReadUncommitted
  );
  assert_eq!(
    TransactionManager::parse_isolation_level("xx"),
    IsolationLevel::SnapshotIsolation
  );
  assert!(Transaction::invalid_byte_buffer(&[1, 2, 3]));
  let g = GsnCtx {
    page_gsn: 10,
    flushed: 8,
  };
  assert!(detect_gsn_dependency(&g));
  assert_eq!(advance_gsn(3, 9), 9);
}

#[test]
fn commit_tree_clean_keeps_anchors() {
  let clock = AtomicU64::new(1);
  let ct = CommitTree::new();
  for i in 1..=COMMIT_TREE_CAPACITY as u64 + 10 {
    ct.commit(i, &clock);
  }
  assert!(ct.len() > COMMIT_TREE_CAPACITY);
  // 两个快照锚点 → 清理后必须升序去重（否则 LCB 二分在乱序数组上失准）
  let snaps = [AtomicU64::new(50), AtomicU64::new(100)];
  ct.clean_if_necessary(&snaps);
  assert!(ct.len() <= 8, "清理后仅保留锚点: {}", ct.len());
  // 锚点 = {(49,49),(99,99),(1034,1034)}；LCB 逐点核对（乱序即失败）
  assert_eq!(ct.lcb(50), 49, "LCB(50) = 前驱 (49,49)");
  assert_eq!(ct.lcb(110), 99, "LCB(110) = 前驱 (99,99)");
  assert_eq!(ct.lcb(u64::MAX), 1034, "LCB(MAX) = 最新锚点");
  // 单调性：LCB 对查询 ts 非减（乱序数组会破坏）
  let mut prev = 0u64;
  for q in [55u64, 60, 99, 105, 110, 1000, 5000] {
    let v = ct.lcb(q);
    assert!(v >= prev, "LCB 单调: q={q} v={v} prev={prev}");
    prev = v;
  }
}
