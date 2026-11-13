//! 第 3 轮：独立 oracle 重跑 + 协议/交错验证（**不引用** `tests/osic_oracle.rs`）。
//!
//! oracle 逐行转写自上游（磁盘实读）：
//! - `leanstore/backend/leanstore/concurrency-recovery/ConcurrencyControl.cpp`
//!   `CommitTree::LCBUnsafe`（:266-279）/ `isVisibleForMe`（:191-242）/
//!   `isVisibleForAll`（:244-253）；
//! - `Worker.cpp:104-110`（startTX 的 LATCH 半写快照协议）、`:142-149`（commitTX）；
//! - `Worker.hpp:226-231`（VISIBILITY）、`Transaction.hpp:57-60`（隔离级别谓词）。
//!
//! 差分 ≥ 7e5 例；负对照（改一个 oracle 分支）必须产生大量不一致。

use std::{
  collections::VecDeque,
  io,
  pin::Pin,
  sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
  },
  thread,
};

use futures_lite::future::block_on;
use ldb_txn::{
  CommitTree, ConcurrencyControl, IsolationLevel, Result as TxnResult, Transaction,
  TransactionManager, TxnState, Worker, txn::tx_flag,
};
use ldb_wal::{CommitVariant, PreCommitTicket, WalFileState, WalSink};

const MSB: u64 = 1 << 63;

/// 断言 `Ok` 并取值（禁 `expect`/`unwrap`，满足 D1 红线）
fn ok<T>(r: TxnResult<T>) -> T {
  if let Ok(v) = r {
    v
  } else {
    unreachable!("Result 应为 Ok")
  }
}

// ---------------------------------------------------------------------------
// 确定性 xorshift64*（固定种子 ⇒ 可复现）
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 独立 LCB oracle：上游 LCBUnsafe（ConcurrencyControl.cpp:266-279）
// ---------------------------------------------------------------------------
fn oracle_lcb(tree: &[(u64, u64)], q: u64) -> u64 {
  // std::lower_bound(first < q) 的手写二分；取前驱 start_ts；无前驱 = 0
  let (mut lo, mut hi) = (0usize, tree.len());
  while lo < hi {
    let mid = lo + (hi - lo) / 2;
    if tree[mid].0 < q {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if lo == 0 { 0 } else { tree[lo - 1].1 }
}

/// 负对照变体：`<` → `<=`（lower_bound 谓词注入）
fn oracle_lcb_mutant(tree: &[(u64, u64)], q: u64) -> u64 {
  let (mut lo, mut hi) = (0usize, tree.len());
  while lo < hi {
    let mid = lo + (hi - lo) / 2;
    if tree[mid].0 <= q {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if lo == 0 { 0 } else { tree[lo - 1].1 }
}

// ---------------------------------------------------------------------------
// 独立 isVisibleForMe oracle（ConcurrencyControl.cpp:191-242）
// ---------------------------------------------------------------------------
/// 隔离级别编码（与上游 TX_ISOLATION_LEVEL 数值顺序一致）
#[derive(Clone, Copy, PartialEq, Eq)]
enum Iso {
  Ru,
  Rc,
  Si,
  Ser,
}

impl Iso {
  fn from_u8(v: u64) -> Self {
    match v {
      0 => Iso::Ru,
      1 => Iso::Rc,
      2 => Iso::Si,
      _ => Iso::Ser,
    }
  }

  fn is_ru(self) -> bool {
    self == Iso::Ru
  }

  fn is_rc(self) -> bool {
    self == Iso::Rc
  }

  fn at_least_si(self) -> bool {
    matches!(self, Iso::Si | Iso::Ser)
  }
}

fn isolation_of(iso: Iso) -> IsolationLevel {
  match iso {
    Iso::Ru => IsolationLevel::ReadUncommitted,
    Iso::Rc => IsolationLevel::ReadCommitted,
    Iso::Si => IsolationLevel::SnapshotIsolation,
    Iso::Ser => IsolationLevel::Serializable,
  }
}

/// 一次判定的输入（含 is_durable 生命周期旋钮）
struct Case {
  tx_ts: u64,
  to_write: bool,
  iso: Iso,
  same_worker: bool,
  my_start: u64,
  clock: u64,
  local_lwm: u64,
}

/// oracle 状态（其它 worker 提交树 + 本 worker 快照缓存）
struct OracleState<'a> {
  tree: &'a [(u64, u64)],
  cache: u64,
  cache_ts: u64,
  mutant: u8,
  real_global_lwm: u64,
}

impl OracleState<'_> {
  fn visible_for_me(&mut self, case: &Case) -> bool {
    let is_commit_ts = case.tx_ts & MSB != 0;
    let committed_ts = if is_commit_ts { case.tx_ts & !MSB } else { 0 };
    let start_ts = case.tx_ts & !MSB;
    // 变异3：删 RU 写意图外全可见早退
    if !case.to_write && case.iso.is_ru() && self.mutant != 3 {
      return true;
    }
    if case.same_worker {
      return true;
    }
    if case.iso.is_rc() || case.iso.is_ru() {
      if is_commit_ts {
        return true;
      }
      if self.cache >= start_ts {
        return true;
      }
      let cur = case.clock + 1;
      let lcb = oracle_lcb(self.tree, cur);
      self.cache = lcb;
      self.cache_ts = cur;
      return lcb >= start_ts;
    }
    if case.iso.at_least_si() {
      if is_commit_ts {
        return case.my_start > committed_ts;
      }
      // 变异2：SI 快路径读实时全局 lwm（而非 start_tx 抓取的本地缓存）
      let lwm = if self.mutant == 2 {
        self.real_global_lwm
      } else {
        case.local_lwm
      };
      if start_ts < lwm {
        return true;
      }
      if self.cache_ts == case.my_start {
        return self.cache >= start_ts;
      }
      // 变异1：删「缓存值（不同时点）仍 >= start_ts → 直接可见」
      if self.mutant != 1 && self.cache >= start_ts {
        return true;
      }
      let lcb = oracle_lcb(self.tree, case.my_start);
      if lcb != 0 {
        self.cache = lcb;
        self.cache_ts = case.my_start;
        return lcb >= start_ts;
      }
      return false;
    }
    unreachable!("隔离级别全覆盖")
  }
}

/// 用随机序列填充 worker[1] 的提交树，返回同步维护的 oracle 视图
fn build_tree(cc: &ConcurrencyControl, rng: &mut Rng, n: usize, span: u64) -> Vec<(u64, u64)> {
  let mut tree = Vec::with_capacity(n);
  for _ in 0..n {
    let s = rng.bounded(span);
    let c = cc.workers[1].commit_tree.commit(s, &cc.global_clock);
    tree.push((c, s));
  }
  tree
}

/// 单批随机例：实现 vs oracle；`mutant` = oracle 侧注入（0=忠实）；比较判定+缓存
fn run_vis_for_me(cases: u64, seed: u64, span: u64, mutant: u8) -> u64 {
  let cc = ConcurrencyControl::new(2);
  let mut rng = Rng(seed);
  let tree = build_tree(&cc, &mut rng, 4096, span);
  let mut mismatch = 0u64;
  for _ in 0..cases {
    let iso = Iso::from_u8(rng.bounded(4));
    let my_start = rng.bounded(span);
    let tx_ts = if rng.bounded(2) == 0 {
      rng.bounded(span + 1000) | MSB
    } else {
      rng.bounded(span + 1000)
    };
    let to_write = rng.bounded(2) == 0;
    let same = rng.bounded(8) == 0;
    let local_lwm = rng.bounded(span);
    let real_global_lwm = rng.bounded(span);
    let clock = rng.bounded(span + 1000);
    let c0 = rng.bounded(span);
    let t0 = rng.bounded(span);
    let durable = rng.bounded(2) == 0;

    let mut me = Transaction::default();
    me.initialize(my_start, isolation_of(iso));
    // is_durable 旋钮：不改变可见性判定（上游 isVisibleForMe 不读 state）
    me.state = if durable {
      TxnState::Running
    } else {
      TxnState::Precommitted
    };

    let other = if same { 0 } else { 1 };
    cc.global_clock.store(clock, Ordering::Release);
    cc.workers[0]
      .local_global_all_lwm_cache
      .store(local_lwm, Ordering::Release);
    cc.workers[0].local_snapshot_cache[other].store(c0, Ordering::Release);
    cc.workers[0].local_snapshot_cache_ts[other].store(t0, Ordering::Release);
    cc.global_all_lwm.store(real_global_lwm, Ordering::Release);

    let got = cc.is_visible_for_me(0, other, tx_ts, to_write, &me);
    let case = Case {
      tx_ts,
      to_write,
      iso,
      same_worker: same,
      my_start,
      clock,
      local_lwm,
    };
    let mut os = OracleState {
      tree: &tree,
      cache: c0,
      cache_ts: t0,
      mutant,
      real_global_lwm,
    };
    let want = os.visible_for_me(&case);

    let got_c = cc.workers[0].local_snapshot_cache[other].load(Ordering::Acquire);
    let got_t = cc.workers[0].local_snapshot_cache_ts[other].load(Ordering::Acquire);
    let cache_ok = mutant != 0 || (got_c == os.cache && got_t == os.cache_ts);
    if got != want || !cache_ok {
      mismatch += 1;
    }
  }
  mismatch
}

// ---------------------------------------------------------------------------
// 独立 LCB 差分
// ---------------------------------------------------------------------------
fn run_lcb(cases: u64, seed: u64, mutant: bool) -> u64 {
  let clock = AtomicU64::new(1000);
  let ct = CommitTree::new();
  let mut rng = Rng(seed);
  let mut tree: Vec<(u64, u64)> = Vec::with_capacity(5000);
  for _ in 0..5000 {
    let s = rng.bounded(1_000_000);
    tree.push((ct.commit(s, &clock), s));
  }
  let mut mismatch = 0u64;
  for _ in 0..cases {
    let q = rng.bounded(1_000_000 + 6000);
    let want = if mutant {
      oracle_lcb_mutant(&tree, q)
    } else {
      oracle_lcb(&tree, q)
    };
    if ct.lcb(q) != want {
      mismatch += 1;
    }
    if ct.lcb_unsafe(q).map(|(_, s)| s).unwrap_or(0) != want {
      mismatch += 1;
    }
  }
  mismatch
}

// ---------------------------------------------------------------------------
// isVisibleForAll 差分（ConcurrencyControl.cpp:244-253）
// ---------------------------------------------------------------------------
fn run_vis_all(cases: u64, seed: u64) -> u64 {
  let cc = ConcurrencyControl::new(1);
  let mut rng = Rng(seed);
  let mut mismatch = 0u64;
  for _ in 0..cases {
    let oldest = rng.bounded(8000);
    let all_lwm = rng.bounded(8000);
    cc.global_oldest_all_start_ts
      .store(oldest, Ordering::Release);
    cc.global_all_lwm.store(all_lwm, Ordering::Release);
    let base = rng.bounded(8000);
    let ts = if rng.bounded(2) == 0 {
      base | MSB
    } else {
      base
    };
    let got = cc.is_visible_for_all(ts);
    let want = if ts & MSB != 0 {
      (ts & !MSB) < oldest
    } else {
      ts < all_lwm
    };
    if got != want {
      mismatch += 1;
    }
  }
  mismatch
}

// ---------------------------------------------------------------------------
// 用例：独立 oracle（累加 ≥ 7e5 例，单例 < 100ms）
// ---------------------------------------------------------------------------
#[test]
fn r3_oracle_visible_for_me_a() {
  assert_eq!(run_vis_for_me(100_000, 0x51c3_0f2a_9e11_0001, 5000, 0), 0);
}

#[test]
fn r3_oracle_visible_for_me_b() {
  assert_eq!(run_vis_for_me(100_000, 0x51c3_0f2a_9e11_0002, 5000, 0), 0);
}

#[test]
fn r3_oracle_visible_for_me_c() {
  assert_eq!(run_vis_for_me(100_000, 0x51c3_0f2a_9e11_0003, 5000, 0), 0);
}

#[test]
fn r3_oracle_visible_for_me_d() {
  assert_eq!(run_vis_for_me(100_000, 0x51c3_0f2a_9e11_0004, 5000, 0), 0);
}

#[test]
fn r3_oracle_visible_for_all() {
  assert_eq!(run_vis_all(100_000, 0xabcd_ef01_2345_6789), 0);
}

#[test]
fn r3_oracle_lcb() {
  assert_eq!(run_lcb(100_000, 0x1234_5678_9abc_def0, false), 0);
  assert_eq!(run_lcb(100_000, 0x1234_5678_9abc_def1, false), 0);
}

/// 负对照：任一单分支变异必须产生大量不一致（差分灵敏度）
#[test]
fn r3_oracle_negative_control() {
  let cases = 20_000;
  let seed = 0x51c3_0f2a_9e11_0001;
  let faithful = run_vis_for_me(cases, seed, 5000, 0);
  assert_eq!(faithful, 0, "忠实 oracle 必须 0 不一致");
  let m1 = run_vis_for_me(cases, seed, 5000, 1);
  let m2 = run_vis_for_me(cases, seed, 5000, 2);
  let m3 = run_vis_for_me(cases, seed, 5000, 3);
  let mlcb = run_lcb(20_000, 0x1234_5678_9abc_def0, true);
  assert!(m1 > 100, "变异1（删 SI else-if）不一致应大量: {m1}");
  assert!(m2 > 100, "变异2（SI 快路径读实时 lwm）不一致应大量: {m2}");
  assert!(m3 > 100, "变异3（删 RU 早退）不一致应大量: {m3}");
  assert!(mlcb > 100, "LCB 谓词变异不一致应大量: {mlcb}");
}

// ---------------------------------------------------------------------------
// Focus 2：start_tx LATCH 快照协议（Worker.cpp:104-110）
// ---------------------------------------------------------------------------

/// 屏障编排：W 反复 start_tx + commit_tx，R 并发 refreshGlobalState；
/// R 断言不读到未来 ts（哨兵 MAX 除外）；W 断言 start_tx 后快照不丢。
#[test]
fn r3_start_tx_latch_barrier_interleave() {
  use std::sync::Barrier;
  let cc = Arc::new(ConcurrencyControl::new(2));
  let barrier = Arc::new(Barrier::new(2));
  let rounds = 20_000u64;
  let cc_r = Arc::clone(&cc);
  let b_r = Arc::clone(&barrier);
  let r = thread::spawn(move || {
    let mut viol = 0u64;
    let mut first = (0u64, 0u64, 0u64);
    for _ in 0..rounds {
      b_r.wait();
      cc_r.refresh_global_state();
      let oldest = cc_r.global_oldest_all_start_ts.load(Ordering::Acquire);
      let clock = cc_r.global_clock.load(Ordering::Acquire);
      // 哨兵 u64::MAX = 「无在途 SI 事务」（无约束，非未来 ts）
      if oldest != u64::MAX && oldest > clock {
        if viol == 0 {
          first = (
            oldest,
            clock,
            cc_r.workers[0].current_snapshot.load(Ordering::Acquire),
          );
        }
        viol += 1;
      }
    }
    (viol, first)
  });
  let cc_w = Arc::clone(&cc);
  let b_w = Arc::clone(&barrier);
  let mut w = Worker::new(0);
  for _ in 0..rounds {
    b_w.wait();
    w.start_tx(&cc_w, IsolationLevel::SnapshotIsolation);
    // 取号写快照后，快照必为本事务 start_ts（LATCH 半写已收尾，不丢快照）
    assert_eq!(
      cc_w.workers[0].current_snapshot.load(Ordering::Acquire) & tx_flag::CLEAN_BITS_MASK,
      w.active.start_ts(),
      "start_tx 返回后快照 = 本事务 start_ts"
    );
    w.commit_tx(&cc_w);
  }
  let (viol, first) = match r.join() {
    Ok(v) => v,
    Err(_) => unreachable!("join"),
  };
  assert_eq!(
    viol, 0,
    "并发 refresh 读到未来 ts 次数: {viol} first(oldest,clock,snap)={first:?}"
  );
}

/// 并发压测：writer 反复 start_tx/commit_tx；reader 断言从不读到未来 ts
#[test]
fn r3_start_tx_no_future_ts_concurrent() {
  let cc = Arc::new(ConcurrencyControl::new(2));
  let done = Arc::new(AtomicBool::new(false));
  let iters = 50_000u64;
  let reader_cc = Arc::clone(&cc);
  let reader_done = Arc::clone(&done);
  let reader = thread::spawn(move || {
    let mut checks = 0u64;
    while !reader_done.load(Ordering::Acquire) {
      let snap = reader_cc.workers[0]
        .current_snapshot
        .load(Ordering::Acquire);
      let clean = snap & tx_flag::CLEAN_BITS_MASK;
      let clock = reader_cc.global_clock.load(Ordering::Acquire);
      assert!(
        clean <= clock,
        "读到未来 ts: clean={clean} clock={clock} raw={snap}"
      );
      checks += 1;
    }
    checks
  });
  let cc_w = Arc::clone(&cc);
  let mut w = Worker::new(0);
  for _ in 0..iters {
    w.start_tx(&cc_w, IsolationLevel::SnapshotIsolation);
    let ts = w.active.start_ts();
    assert_eq!(
      cc_w.workers[0].current_snapshot.load(Ordering::Acquire) & tx_flag::CLEAN_BITS_MASK,
      ts,
      "start_tx 返回后快照 = 本事务 start_ts（不丢快照）"
    );
    w.commit_tx(&cc_w);
  }
  done.store(true, Ordering::Release);
  let checks = match reader.join() {
    Ok(v) => v,
    Err(_) => unreachable!("reader"),
  };
  assert!(checks > 0, "reader 至少观察到一次快照");
}

// ---------------------------------------------------------------------------
// Focus 3：D5 水位（monotone + bounded）
// ---------------------------------------------------------------------------
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
      end: 1 << 31,
      shard: 1 << 18,
    },
  )
}

/// 参与写者的最小本地 GSN（本地 GSN = 0 的 worker 未参与写）
fn min_participating_gsn(mgr: &TransactionManager) -> u64 {
  (0..mgr.log_mgr.worker_cnt())
    .filter_map(|i| mgr.log_mgr.worker(i))
    .map(|w| w.get_current_gsn())
    .filter(|g| *g > 0)
    .min()
    .unwrap_or(0)
}

/// 双 worker 交替提交 2000 次：水位单调、>= 票据 GSN、<= 参与写者 min 本地 GSN
#[test]
fn r3_durable_watermark_monotone_serial() {
  let mgr = mem_manager(CommitVariant::WorkersWriteLog);
  block_on(async {
    let mut last = mgr.log_mgr.min_gsn_flushed.load();
    for i in 0..1000u64 {
      let wid = (i % 2) as usize;
      mgr.start_transaction(wid, IsolationLevel::SnapshotIsolation);
      let t = ok(mgr.commit_transaction(wid).await);
      let gsn = t.max_observed_gsn;
      ok(mgr.durable_commit(t).await);
      let now = mgr.log_mgr.min_gsn_flushed.load();
      assert!(now >= last, "水位不回退: {now} < {last}");
      assert!(now >= gsn, "水位 >= 票据观测 GSN: {now} < {gsn}");
      assert!(
        now <= min_participating_gsn(&mgr),
        "水位 <= 参与写者 min 本地 GSN: {now} > {}",
        min_participating_gsn(&mgr)
      );
      last = now;
    }
    assert_watermark_contract(&mgr);
  });
}

/// 多 worker 交错 + 乱序 durable 1e5 次（>100ms，显式 --ignored 运行）
#[test]
#[ignore = "1e5 压测超 100ms；显式 --run-ignored 运行"]
fn r3_durable_watermark_monotone_interleaved_1e5() {
  let mgr = mem_manager(CommitVariant::WorkersWriteLog);
  block_on(async {
    let mut last = mgr.log_mgr.min_gsn_flushed.load();
    let mut pending: VecDeque<PreCommitTicket> = VecDeque::new();
    let mut rng = Rng(0xfeed_face_dead_beef);
    for i in 0..100_000u64 {
      let wid = (i % 2) as usize;
      mgr.start_transaction(wid, IsolationLevel::SnapshotIsolation);
      let t = ok(mgr.commit_transaction(wid).await);
      pending.push_back(t);
      // 乱序固化：随机挑一个待固化票据
      let idx = if pending.len() > 1 && rng.bounded(3) == 0 {
        rng.bounded(pending.len() as u64) as usize
      } else {
        0
      };
      let t = match pending.remove(idx) {
        Some(t) => t,
        None => unreachable!("非空"),
      };
      let gsn = t.max_observed_gsn;
      ok(mgr.durable_commit(t).await);
      let now = mgr.log_mgr.min_gsn_flushed.load();
      assert!(now >= last, "水位不回退 @{i}: {now} < {last}");
      assert!(now >= gsn, "水位 >= 票据 GSN @{i}: {now} < {gsn}");
      assert!(
        now <= min_participating_gsn(&mgr),
        "水位 <= 参与写者 min 本地 GSN @{i}: {now} > {}",
        min_participating_gsn(&mgr)
      );
      last = now;
    }
    while let Some(t) = pending.pop_front() {
      ok(mgr.durable_commit(t).await);
    }
    // 收尾后：水位仍不越界
    assert!(mgr.log_mgr.min_gsn_flushed.load() <= min_participating_gsn(&mgr));
  });
}

/// advance_min 单调契约（并入 serial 用例，避免零击杀独立用例）
fn assert_watermark_contract(mgr: &TransactionManager) {
  let w = mgr.log_mgr.min_gsn_flushed.clone();
  let cur = w.load();
  assert_eq!(w.advance_min(cur), cur, "等值不推进");
  assert_eq!(w.advance_min(0), cur, "小值不得回退");
  assert_eq!(w.advance_min(cur + 7), cur + 7, "大值单调推进");
  assert_eq!(w.load(), cur + 7);
}

// ---------------------------------------------------------------------------
// Focus 4：precommitted / RFA 队列路由语义（ldb_txn → ldb_wal）
// ---------------------------------------------------------------------------
#[test]
fn r3_precommitted_routing_wwl_and_wilo() {
  // WorkersWriteLog 非 remote-flush → ms_queue_rfa（本地即时固化）
  let mgr = mem_manager(CommitVariant::WorkersWriteLog);
  block_on(async {
    mgr.start_transaction(0, IsolationLevel::SnapshotIsolation);
    let t = ok(mgr.commit_transaction(0).await);
    assert_eq!(mgr.log.ms_queue_rfa.lock().len(), 1, "WWL 入 ms_queue_rfa");
    assert_eq!(mgr.log.precommitted.lock().len(), 0);
    assert_eq!(mgr.log.precommitted_rfa.lock().len(), 0);
    {
      let q = mgr.log.ms_queue_rfa.lock();
      assert_eq!(q[0].commit_ts, t.commit_ts, "票据 commit_ts 保真");
      assert_eq!(q[0].max_observed_gsn, t.max_observed_gsn);
      assert_eq!(q[0].start_ts, t.start_ts);
      assert_eq!(q[0].blob_page, 0, "非 remote-flush");
      assert!(q[0].commit_ts >= q[0].start_ts, "commit_ts >= start_ts");
    }
    // exactly-once：再提交一次 → 队列恰 +1；durable 后经 persist_log 排空
    mgr.start_transaction(0, IsolationLevel::SnapshotIsolation);
    let t2 = ok(mgr.commit_transaction(0).await);
    assert_eq!(mgr.log.ms_queue_rfa.lock().len(), 2, "每次提交恰入队一次");
    ok(mgr.durable_commit(t2).await);
    assert_eq!(
      mgr.log.ms_queue_rfa.lock().len(),
      0,
      "durable 经 persist_log → try_commit_rfa_txns 排空（exactly-once）"
    );
  });

  // WiloSteal 非 remote-flush → precommitted_rfa（组提交 phase_five 按 commit_ts 固化）
  let mgr2 = mem_manager(CommitVariant::WiloSteal);
  block_on(async {
    mgr2.start_transaction(0, IsolationLevel::SnapshotIsolation);
    let _t = ok(mgr2.commit_transaction(0).await);
    assert_eq!(
      mgr2.log.precommitted_rfa.lock().len(),
      1,
      "WiloSteal 入 precommitted_rfa"
    );
    assert_eq!(mgr2.log.precommitted.lock().len(), 0);
    assert_eq!(mgr2.log.ms_queue_rfa.lock().len(), 0);
  });

  // remote-flush（blob_page != 0）→ precommitted（组提交 phase_five 按 gsn+start_ts 固化）
  let mgr3 = mem_manager(CommitVariant::WorkersWriteLog);
  let ticket = PreCommitTicket {
    commit_ts: 7,
    max_observed_gsn: 3,
    start_ts: 2,
    blob_page: 4096,
  };
  ldb_wal::enqueue_precommitted(&mgr3.log, ticket);
  assert_eq!(
    mgr3.log.precommitted.lock().len(),
    1,
    "remote-flush 入 precommitted"
  );
  assert!(mgr3.log.precommitted.lock()[0].is_remote_flush());
  assert_eq!(mgr3.log.ms_queue_rfa.lock().len(), 0);
}
