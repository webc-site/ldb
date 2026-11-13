//! OSIC MVCC 上游对拍（清单 A1/A2）与状态机闭环（B1/B2）。
//!
//! oracle 逐行转写自上游 `leanstore/backend/leanstore/concurrency-recovery/
//! ConcurrencyControl.cpp`：
//! - `CommitTree::LCBUnsafe`（`:266-279`）、`LCB`（`:281-291`）
//! - `isVisibleForMe`（`:191-242`）、`isVisibleForAll`（`:244-253`）、
//!   `getCommitTimestamp`（`:176-188`）、`isVisibleForIt`（`:163-174`）
//!
//! 三条差分均 ≥1e5 例，`MISMATCH=0` 才通过。避免引入 rng dev-dep：用固定种子的
//! xorshift64* 生成可复现序列。

use std::{
  future::Future,
  io,
  pin::Pin,
  sync::atomic::{AtomicU64, Ordering},
};

use futures_lite::future::block_on;
use ldb_txn::{
  CommitTree, ConcurrencyControl, IsolationLevel, Transaction, TransactionManager, TxnState,
  Visibility, Worker,
};
use ldb_wal::{CommitVariant, WalFileState, WalSink};

const MSB: u64 = 1 << 63;
const RC_BIT: u64 = 1 << 62;

/// 确定性 xorshift64*（种子固定 ⇒ 可复现）
struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    let mut x = self.0;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    self.0 = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
  }

  fn bounded(&mut self, n: u64) -> u64 {
    self.next() % n
  }
}

/// 上游 `CommitTree::LCBUnsafe`（ConcurrencyControl.cpp:266-279）
fn oracle_lcb(tree: &[(u64, u64)], start_ts: u64) -> u64 {
  // std::lower_bound(..., pair.first < ts) → 首个 commit_ts >= start_ts，取其前驱
  let idx = tree.partition_point(|&(c, _)| c < start_ts);
  if idx == 0 { 0 } else { tree[idx - 1].1 }
}

/// 上游 `isVisibleForMe` 的判定输入（拆结构以满足 clippy 参数上限）
struct VisCase {
  tx_ts: u64,
  to_write: bool,
  is_ru: bool,
  is_rc: bool,
  same_worker: bool,
}

/// 上游 `isVisibleForMe`（ConcurrencyControl.cpp:191-242）
fn oracle_visible_for_me(
  tree: &[(u64, u64)],
  global_clock: u64,
  my_start_ts: u64,
  case: &VisCase,
  local_all_lwm_cache: u64,
  cache: &mut u64,
  cache_ts: &mut u64,
) -> bool {
  let is_commit_ts = case.tx_ts & MSB != 0;
  let committed_ts = if is_commit_ts { case.tx_ts & !MSB } else { 0 };
  let start_ts = case.tx_ts & !MSB;
  if !case.to_write && case.is_ru {
    return true;
  }
  if case.same_worker {
    return true;
  }
  if case.is_rc || case.is_ru {
    if is_commit_ts {
      return true;
    }
    if *cache >= start_ts {
      return true;
    }
    let cur = global_clock + 1;
    let lcb = oracle_lcb(tree, cur);
    *cache = lcb;
    *cache_ts = cur;
    return lcb >= start_ts;
  }
  if is_commit_ts {
    return my_start_ts > committed_ts;
  }
  if start_ts < local_all_lwm_cache {
    return true;
  }
  if *cache_ts == my_start_ts {
    return *cache >= start_ts;
  } else if *cache >= start_ts {
    return true;
  }
  let lcb = oracle_lcb(tree, my_start_ts);
  if lcb != 0 {
    *cache = lcb;
    *cache_ts = my_start_ts;
    return lcb >= start_ts;
  }
  false
}

#[test]
fn commit_tree_lcb_matches_upstream_oracle() {
  let tclock = AtomicU64::new(1000);
  let ct = CommitTree::new();
  let mut rng = Rng(0x1234_5678_9abc_def0);
  let mut tree: Vec<(u64, u64)> = Vec::with_capacity(5000);
  for _ in 0..5000 {
    let s = rng.bounded(1_000_000);
    let c = ct.commit(s, &tclock);
    tree.push((c, s));
  }
  let mut mismatch = 0u64;
  for _ in 0..200_000 {
    let q = rng.bounded(1_000_000 + 6000);
    let want = oracle_lcb(&tree, q);
    if ct.lcb(q) != want {
      mismatch += 1;
    }
    let uns = ct.lcb_unsafe(q).map(|(_, s)| s).unwrap_or(0);
    if uns != want {
      mismatch += 1;
    }
  }
  assert_eq!(mismatch, 0, "CommitTree LCB 与上游 oracle 不一致");
}

#[test]
fn visible_for_me_matches_upstream_oracle() {
  let cc = ConcurrencyControl::new(2);
  let tclock = AtomicU64::new(2000);
  let mut rng = Rng(0xdead_beef_cafe_1234);
  let mut tree: Vec<(u64, u64)> = Vec::with_capacity(4000);
  for _ in 0..4000 {
    let s = rng.bounded(5000);
    let c = cc.workers[1].commit_tree.commit(s, &tclock);
    tree.push((c, s));
  }
  let mut mismatch = 0u64;
  let mut checks = 0u64;
  for _ in 0..150_000 {
    let iso = match rng.bounded(4) {
      0 => IsolationLevel::ReadUncommitted,
      1 => IsolationLevel::ReadCommitted,
      2 => IsolationLevel::SnapshotIsolation,
      _ => IsolationLevel::Serializable,
    };
    let my_start = rng.bounded(5000);
    let tx_ts = if rng.bounded(2) == 0 {
      rng.bounded(6000) | MSB
    } else {
      rng.bounded(6000)
    };
    let to_write = rng.bounded(2) == 0;
    let all_lwm = rng.bounded(5000);
    let clock = rng.bounded(6000);
    let c0 = rng.bounded(5000);
    let t0 = rng.bounded(5000);

    let mut me = Transaction::default();
    me.initialize(my_start, iso);
    cc.global_clock.store(clock, Ordering::Release);
    cc.workers[0]
      .local_global_all_lwm_cache
      .store(all_lwm, Ordering::Release);
    cc.workers[0].local_snapshot_cache[1].store(c0, Ordering::Release);
    cc.workers[0].local_snapshot_cache_ts[1].store(t0, Ordering::Release);

    let got = cc.is_visible_for_me(0, 1, tx_ts, to_write, &me);
    let case = VisCase {
      tx_ts,
      to_write,
      is_ru: iso == IsolationLevel::ReadUncommitted,
      is_rc: iso == IsolationLevel::ReadCommitted,
      same_worker: false,
    };
    let (mut oc, mut oct) = (c0, t0);
    let want = oracle_visible_for_me(&tree, clock, my_start, &case, all_lwm, &mut oc, &mut oct);

    let got_c = cc.workers[0].local_snapshot_cache[1].load(Ordering::Acquire);
    let got_t = cc.workers[0].local_snapshot_cache_ts[1].load(Ordering::Acquire);
    if got != want || got_c != oc || got_t != oct {
      mismatch += 1;
    }
    checks += 1;
  }
  assert_eq!(checks, 150_000);
  assert_eq!(mismatch, 0, "is_visible_for_me 与上游 oracle 不一致");
}

#[test]
fn visible_for_all_matches_upstream_oracle() {
  let cc = ConcurrencyControl::new(1);
  let mut rng = Rng(0xabcd_ef01_2345_6789);
  let mut mismatch = 0u64;
  for _ in 0..150_000 {
    let oldest = rng.bounded(8000);
    let all_lwm = rng.bounded(8000);
    cc.global_oldest_all_start_ts
      .store(oldest, Ordering::Release);
    cc.global_all_lwm.store(all_lwm, Ordering::Release);
    let is_commit = rng.bounded(2) == 0;
    let base = rng.bounded(8000);
    let ts = if is_commit { base | MSB } else { base };
    let got = cc.is_visible_for_all(ts);
    // 上游 isVisibleForAll（ConcurrencyControl.cpp:244-253）
    let want = if is_commit {
      (ts & !MSB) < oldest
    } else {
      ts < all_lwm
    };
    if got != want {
      mismatch += 1;
    }
  }
  assert_eq!(mismatch, 0, "is_visible_for_all 与上游 oracle 不一致");
}

/// D5：`min_gsn_flushed` 单调不回退，且单写者自驱不越界（水位恰为票据 GSN）
#[test]
fn durable_commit_watermark_monotone_and_bounded() {
  let mgr = mem_manager(CommitVariant::WorkersWriteLog);
  block_on(async {
    let mut last = mgr.log_mgr.min_gsn_flushed.load();
    for _ in 0..3 {
      mgr.start_transaction(0, IsolationLevel::SnapshotIsolation);
      let cr = mgr.commit_transaction(0).await;
      assert!(cr.is_ok(), "commit 应成功: {cr:?}");
      let Ok(t) = cr else {
        unreachable!("commit 成功已断言")
      };
      let gsn = t.max_observed_gsn;
      assert!(gsn > 0, "提交推进本地 GSN");
      let dr = mgr.durable_commit(t).await;
      assert!(dr.is_ok(), "durable 应成功: {dr:?}");
      let now = mgr.log_mgr.min_gsn_flushed.load();
      assert!(now >= last, "水位单调不回退: {now} < {last}");
      assert!(now >= gsn, "水位须 >= 票据观测 GSN");
      assert_eq!(
        now, gsn,
        "单写者（worker 0）自驱：水位恰为票据 GSN，无越界抬高"
      );
      last = now;
    }
    // 直接钉死共享水位的单调契约（advance_min 只增不减，D5）
    let w = mgr.log_mgr.min_gsn_flushed.clone();
    assert_eq!(w.advance_min(last + 5), last + 5);
    assert_eq!(w.advance_min(0), last + 5, "小值不得回退水位");
  });
}

#[test]
fn visible_for_it_and_commit_timestamp() {
  let cc = ConcurrencyControl::new(2);
  let tclock = AtomicU64::new(500);
  let c0 = cc.workers[1].commit_tree.commit(40, &tclock);
  let c1 = cc.workers[1].commit_tree.commit(45, &tclock);
  assert_eq!((c0, c1), (500, 501));
  // 快照 600 > 501 → 已可见
  cc.workers[1].current_snapshot.store(600, Ordering::Release);
  assert_eq!(cc.is_visible_for_it(1, 501), Visibility::VisibleAlready);
  cc.workers[1].current_snapshot.store(400, Ordering::Release);
  assert_eq!(cc.is_visible_for_it(1, 501), Visibility::VisibleNextRound);
  // getCommitTimestamp：commit_ts 直接剥位；start_ts 走 LCB，无命中 → MAX
  assert_eq!(cc.get_commit_timestamp(1, 45 | MSB), 45);
  assert_eq!(
    cc.get_commit_timestamp(1, 502),
    45,
    "LCB(502) 前任 start=45"
  );
  assert_eq!(cc.get_commit_timestamp(1, 0), u64::MAX, "无更早提交 → MAX");
}

#[test]
fn start_commit_abort_state_machine() {
  let mut cc = ConcurrencyControl::new(2);
  let mut w = Worker::new(0);

  // SI 启动：Running，快照无 RC/OLAP 位，SI LWM 快照被登记（Worker.cpp:113）
  w.start_tx(&cc, IsolationLevel::SnapshotIsolation);
  assert_eq!(w.active.state, TxnState::Running);
  let snap = cc.workers[0].current_snapshot.load(Ordering::Acquire);
  assert_eq!(snap & RC_BIT, 0, "SI 无 RC 位");
  assert_eq!(
    cc.workers[0]
      .local_global_all_lwm_cache
      .load(Ordering::Acquire),
    cc.global_all_lwm.load(Ordering::Acquire)
  );

  // 提交：Precommitted + commit_tree 记账 + local_latest_write_tx=commit_ts
  w.active.mark_as_write();
  let ct = w.commit_tx(&cc);
  assert_eq!(w.active.state, TxnState::Precommitted);
  assert_eq!(
    cc.workers[0].local_latest_write_tx.load(Ordering::Acquire),
    ct
  );
  assert_eq!(cc.workers[0].commit_tree.len(), 1);
  assert!(w.active.is_durable(), "预提交处于持久路径");

  // RC 启动：跨模式切换置 RC 位（switchToReadCommittedMode）+ Running
  w.start_tx(&cc, IsolationLevel::ReadCommitted);
  assert_eq!(w.active.state, TxnState::Running);
  assert_ne!(
    cc.workers[0].current_snapshot.load(Ordering::Acquire) & RC_BIT,
    0,
    "RC 位已置"
  );

  // 回滚：Aborted + 快照清零
  w.abort_tx(&mut cc);
  assert_eq!(w.active.state, TxnState::Aborted);
  assert_eq!(cc.workers[0].current_snapshot.load(Ordering::Acquire), 0);
}

#[test]
fn switch_modes_pin_upstream_snapshot_effects() {
  let cc = ConcurrencyControl::new(2);
  cc.global_clock.store(777, Ordering::Release);
  cc.switch_to_snapshot_isolation_mode(0);
  assert_eq!(
    cc.workers[0].current_snapshot.load(Ordering::Acquire),
    777,
    "SI 切换登记当前全局时钟（ConcurrencyControl.cpp:93）"
  );
  cc.switch_to_read_committed_mode(1);
  assert_ne!(
    cc.workers[1].current_snapshot.load(Ordering::Acquire) & RC_BIT,
    0,
    "RC 切换置位（ConcurrencyControl.cpp:103）"
  );
}

/// 丢弃式 sink：只为让 WAL 水位推进可测，不做真实落盘
struct NullSink;

impl WalSink for NullSink {
  fn write_wal(
    &mut self,
    _data: Vec<u8>,
    _off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(async { Ok(()) })
  }

  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(async { Ok(()) })
  }
}

fn mem_manager(variant: CommitVariant) -> TransactionManager {
  TransactionManager::new(
    2,
    64,
    variant,
    Box::new(NullSink),
    WalFileState {
      start: 0,
      end: 1 << 20,
      shard: 1 << 18,
    },
  )
}

#[test]
fn manager_commit_durable_finalizes_and_abort_closes_state() {
  let mgr = mem_manager(CommitVariant::WorkersWriteLog);
  block_on(async {
    // 开事务 → 提交 → durable 定稿（上游 DurableCommit：state=COMMITTED）
    mgr.start_transaction(0, IsolationLevel::SnapshotIsolation);
    let commit_res = mgr.commit_transaction(0).await;
    assert!(commit_res.is_ok(), "commit 应成功: {commit_res:?}");
    let Ok(ticket) = commit_res else {
      unreachable!("commit 成功已断言")
    };
    assert!(ticket.commit_ts > 0);
    let durable_res = mgr.durable_commit(ticket).await;
    assert!(durable_res.is_ok(), "durable 应成功: {durable_res:?}");
    assert_eq!(
      mgr.workers[0].lock().active.state,
      TxnState::Committed,
      "durable 定稿到 Committed"
    );

    // 屏障事务：预置 LWM 已追平 → 立即返回当前时钟号并推进 +2
    mgr
      .cc
      .lock()
      .global_all_lwm
      .store(u64::MAX, Ordering::Release);
    let snapshot = mgr.cc.lock().global_clock.load(Ordering::Acquire);
    let barrier = mgr.add_barrier_transaction().await;
    assert_eq!(barrier, snapshot, "屏障返回当前时钟号");
    assert_eq!(
      mgr.cc.lock().global_clock.load(Ordering::Acquire),
      snapshot + 2,
      "屏障推进 +2"
    );

    // 回滚闭环
    mgr.start_transaction(0, IsolationLevel::ReadCommitted);
    assert!(mgr.abort_transaction(0).await.is_ok(), "abort 应成功");
    // 未开事务提交 → 显式错误（P1-6 状态机闭环）
    assert!(mgr.commit_transaction(0).await.is_err());
  });
}
