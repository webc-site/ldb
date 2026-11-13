//! OSIC 多版本并发控制（VLDB'23，上游 `cr/ConcurrencyControl.cpp`）。
//!
//! CommitTree：定长 (commit_ts, start_ts) 数组 + LCB 二分；
//! 可见性：RC 走 LCB 快照缓存，SI 走 start_ts < global_all_lwm 或 LCB；
//! LCB 快照缓存 / 全局水位（all/oltp）/ 墓地迁移（graveyard）钩子。

use std::{
  fmt,
  mem::take,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
};

use parking_lot::{Mutex, RwLock};

use crate::txn::{IsolationLevel, Transaction, TxnState, tx_flag};

/// CommitTree 容量（上游固定 1024）
pub const COMMIT_TREE_CAPACITY: usize = 1024;

/// worker 快照数组别名（clean_if_necessary 参数收敛）
pub type WorkerSnapshots = [AtomicU64];

/// 可见性判定结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
  /// 本轮可见
  VisibleAlready,
  /// 下一轮可见（仍持有闩）
  VisibleNextRound,
}

/// 提交树（上游 `Worker::ConcurrencyControl::CommitTree`）
#[derive(Debug)]
pub struct CommitTree {
  /// (commit_ts, start_ts) 对（游标 = array.len()，锁内维护）
  array: Mutex<Vec<(u64, u64)>>,
}

impl CommitTree {
  /// 构造
  pub fn new() -> Self {
    Self {
      array: Mutex::new(Vec::with_capacity(COMMIT_TREE_CAPACITY)),
    }
  }

  /// 提交登记（全局时钟取号），返回 commit_ts
  pub fn commit(&self, start_ts: u64, global_clock: &AtomicU64) -> u64 {
    let commit_ts = global_clock.fetch_add(1, Ordering::Relaxed);
    self.array.lock().push((commit_ts, start_ts));
    commit_ts
  }

  /// LCB 不安全版（无锁读，调用方持读侧）
  pub fn lcb_unsafe(&self, start_ts: u64) -> Option<(u64, u64)> {
    let arr = self.array.lock();
    // lower_bound by commit_ts；取其前驱（已提交 && start 语义见上游 assert）
    let idx = arr.partition_point(|&(c, _)| c < start_ts);
    if idx == 0 { None } else { Some(arr[idx - 1]) }
  }

  /// LCB 安全版（返回 start_ts 侧，无命中返回 0）
  pub fn lcb(&self, start_ts: u64) -> u64 {
    self.lcb_unsafe(start_ts).map(|(_, s)| s).unwrap_or(0)
  }

  /// 满则清理：保留各 worker 快照锚点（上游 `CommitTree::cleanIfNecessary`，
  /// `ConcurrencyControl.cpp:293-328`）
  pub fn clean_if_necessary(&self, worker_snapshots: &WorkerSnapshots) {
    let mut arr = self.array.lock();
    if arr.len() < COMMIT_TREE_CAPACITY {
      return;
    }
    // 锚点 = 最新提交 + 各 worker 快照各自的 LCB（保留跨快照查询能力）。
    // 上游用 `std::set` 去重；锚点数 ≤ worker_cnt+1，用 Vec + 排序去重
    // （小 N 缓存友好，todo.md §9 S4；语义等价：set 字典序 = commit_ts 升序）。
    let mut keep: Vec<(u64, u64)> = Vec::with_capacity(worker_snapshots.len() + 1);
    if let Some(last) = arr.last() {
      keep.push(*last);
    }
    for snap in worker_snapshots {
      let ts = snap.load(Ordering::Acquire) & tx_flag::CLEAN_BITS_MASK;
      if ts == 0 {
        if let Some(first) = arr.first() {
          keep.push(*first);
        }
      } else {
        let idx = arr.partition_point(|&(c, _)| c < ts);
        if idx > 0 {
          keep.push(arr[idx - 1]);
        }
      }
    }
    keep.sort_unstable();
    keep.dedup();
    // 原地换出：保留外部容量（上游复用定长数组，不做 shrink）
    arr.clear();
    arr.extend(keep);
  }

  /// 当前条目数
  #[inline]
  pub fn len(&self) -> usize {
    self.array.lock().len()
  }

  /// 是否为空
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

impl Default for CommitTree {
  fn default() -> Self {
    Self::new()
  }
}

/// 历史（Undo/待办）版本登记项（ ldb_recovery::HistoryTree 的条目形态）
#[derive(Debug, Clone, Copy)]
pub struct VersionEntry {
  /// 事务 ID
  pub tx_id: u64,
  /// 表 ID
  pub dt_id: u16,
  /// 载荷偏移（由 HistoryTree 管理）
  pub payload: [u8; 16],
  /// 载荷长度
  pub payload_len: u16,
  /// 是否删除版本（graveyard 迁移标记）
  pub is_remove: bool,
}

/// graveyard 迁移回调（提交路径中频；dyn 可接受的权衡点 —— 见 SKILL 热路径例外的冷/中频讨论）
pub type GraveyardHook = RwLock<Option<Arc<dyn Fn(u64, u16, &[u8], bool) + Send + Sync>>>;

/// Worker 并发控制状态（每 worker 一份；上游 Worker::cc）
#[derive(Debug)]
pub struct WorkerCc {
  /// worker 编号
  pub worker_id: usize,
  /// 本 worker 提交树
  pub commit_tree: CommitTree,
  /// 本 worker 当前快照（start_ts | 标志位）
  pub current_snapshot: AtomicU64,
  /// 本 worker 最新写事务 start_ts（LCB 跳过优化）
  pub local_latest_write_tx: AtomicU64,
  /// 本 worker 最新 LWM 缓存
  pub local_latest_lwm_for_tx: AtomicU64,
  /// all LWM 接收器
  pub all_lwm_receiver: AtomicU64,
  /// oltp LWM 接收器
  pub oltp_lwm_receiver: AtomicU64,
  /// LWM 闩（偶数 = 稳定）
  pub local_lwm_latch: AtomicU64,
  /// 本地快照缓存（对每个其它 worker 的 LCB 缓存）
  pub local_snapshot_cache: Vec<AtomicU64>,
  /// 缓存对应的观测时点
  pub local_snapshot_cache_ts: Vec<AtomicU64>,
  /// 本 worker 在上次 start_tx 时抓取的全局 all LWM（上游 cc.local_global_all_lwm_cache，
  /// `Worker.cpp:113`；SI 可见性用它做快路径）
  pub local_global_all_lwm_cache: AtomicU64,
  /// 历史版本存储（插入 / 检索；HistoryTree 由 ldb_recovery 提供实现）
  pub history: Vec<VersionEntry>,
  /// OLAP 模式（graveyard 启用）
  pub olap_mode: bool,
}

/// 并发控制器（全局 + 每 worker 状态集）
pub struct ConcurrencyControl {
  /// 全局时钟（WORKERS_INCREMENT = worker 数 起步）
  pub global_clock: AtomicU64,
  /// 全部 worker 状态
  pub workers: Vec<WorkerCc>,
  /// 全局最老事务 start_ts（all）
  pub global_oldest_all_start_ts: AtomicU64,
  /// 全局最老 OLTP 事务 start_ts
  pub global_oldest_oltp_start_ts: AtomicU64,
  /// 全局最新 OLAP 事务 start_ts
  pub global_newest_olap_start_ts: AtomicU64,
  /// 全局 all LWM（低水位）
  pub global_all_lwm: AtomicU64,
  /// 全局 oltp LWM
  pub global_oltp_lwm: AtomicU64,
  /// 全局刷新锁
  pub global_mutex: Mutex<()>,
  /// 历史 GC（graveyard 迁移回调注册）
  pub graveyard_hook: GraveyardHook,
}

impl fmt::Debug for ConcurrencyControl {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ConcurrencyControl")
      .field("workers", &self.workers.len())
      .field("global_clock", &self.global_clock.load(Ordering::Relaxed))
      .field(
        "global_all_lwm",
        &self.global_all_lwm.load(Ordering::Relaxed),
      )
      .finish_non_exhaustive()
  }
}

impl ConcurrencyControl {
  /// 构造（worker 数决定时钟起点与缓存表）
  pub fn new(worker_cnt: usize) -> Self {
    Self {
      global_clock: AtomicU64::new(worker_cnt as u64),
      workers: (0..worker_cnt)
        .map(|wid| WorkerCc {
          worker_id: wid,
          commit_tree: CommitTree::new(),
          current_snapshot: AtomicU64::new(0),
          local_latest_write_tx: AtomicU64::new(0),
          local_latest_lwm_for_tx: AtomicU64::new(0),
          all_lwm_receiver: AtomicU64::new(0),
          oltp_lwm_receiver: AtomicU64::new(0),
          local_lwm_latch: AtomicU64::new(0),
          local_snapshot_cache: (0..worker_cnt).map(|_| AtomicU64::new(0)).collect(),
          local_snapshot_cache_ts: (0..worker_cnt).map(|_| AtomicU64::new(0)).collect(),
          local_global_all_lwm_cache: AtomicU64::new(0),
          history: Vec::new(),
          olap_mode: false,
        })
        .collect(),
      global_oldest_all_start_ts: AtomicU64::new(u64::MAX),
      global_oldest_oltp_start_ts: AtomicU64::new(u64::MAX),
      global_newest_olap_start_ts: AtomicU64::new(0),
      global_all_lwm: AtomicU64::new(0),
      global_oltp_lwm: AtomicU64::new(0),
      global_mutex: Mutex::new(()),
      graveyard_hook: RwLock::new(None),
    }
  }

  /// worker 数
  #[inline]
  pub fn worker_cnt(&self) -> usize {
    self.workers.len()
  }

  /// 切换 SI 模式（登记快照 + 刷新全局态）
  pub fn switch_to_snapshot_isolation_mode(&self, wid: usize) {
    {
      let _g = self.global_mutex.lock();
      let ts = self.global_clock.load(Ordering::Relaxed);
      self.workers[wid]
        .current_snapshot
        .store(ts, Ordering::Release);
    }
    self.refresh_global_state();
  }

  /// 切换 RC 模式（置 RC 位）
  pub fn switch_to_read_committed_mode(&self, wid: usize) {
    {
      let _g = self.global_mutex.lock();
      let flagged = self.workers[wid].current_snapshot.load(Ordering::Relaxed) | tx_flag::RC_BIT;
      self.workers[wid]
        .current_snapshot
        .store(flagged, Ordering::Release);
    }
    self.refresh_global_state();
  }

  /// 刷新全局态（上游 refreshGlobalState：算最老/最新水位并分发 LWM）
  pub fn refresh_global_state(&self) {
    let Some(_g) = self.global_mutex.try_lock() else {
      return;
    };
    let _n = self.worker_cnt();
    let mut oldest_all = u64::MAX;
    let mut oldest_oltp = u64::MAX;
    let mut newest_olap = 0;
    for w in &self.workers {
      let snap = w.current_snapshot.load(Ordering::Acquire) & tx_flag::CLEAN_BITS_MASK;
      let is_rc = w.current_snapshot.load(Ordering::Relaxed) & tx_flag::RC_BIT != 0;
      let is_olap = w.current_snapshot.load(Ordering::Relaxed) & tx_flag::OLAP_BIT != 0;
      if !is_rc && snap != 0 {
        oldest_all = oldest_all.min(snap);
        if is_olap {
          newest_olap = newest_olap.max(snap);
        } else {
          oldest_oltp = oldest_oltp.min(snap);
        }
      }
    }
    self
      .global_oldest_all_start_ts
      .store(oldest_all, Ordering::Release);
    self
      .global_oldest_oltp_start_ts
      .store(oldest_oltp, Ordering::Release);
    self
      .global_newest_olap_start_ts
      .store(newest_olap, Ordering::Release);

    let mut all_lwm_buf = u64::MAX;
    let mut oltp_lwm_buf = u64::MAX;
    let mut skipped = false;
    for w in &self.workers {
      if w.local_latest_lwm_for_tx.load(Ordering::Relaxed)
        == w.local_latest_write_tx.load(Ordering::Relaxed)
      {
        skipped = true;
        continue;
      }
      w.local_latest_lwm_for_tx.store(
        w.local_latest_write_tx.load(Ordering::Relaxed),
        Ordering::Release,
      );
      let all = w.commit_tree.lcb(oldest_all);
      let oltp = if self.workers.iter().any(|x| x.olap_mode) && oldest_all != oldest_oltp {
        w.commit_tree.lcb(oldest_oltp)
      } else {
        all
      };
      all_lwm_buf = all_lwm_buf.min(all);
      oltp_lwm_buf = oltp_lwm_buf.min(oltp);
      // 分发（闩对：+1 lock / +1 release）
      w.local_lwm_latch.fetch_add(1, Ordering::Release);
      w.all_lwm_receiver.store(all, Ordering::Release);
      w.oltp_lwm_receiver.store(oltp, Ordering::Release);
      w.local_lwm_latch.fetch_add(1, Ordering::Release);
    }
    if !skipped {
      self.global_all_lwm.store(all_lwm_buf, Ordering::Release);
      self.global_oltp_lwm.store(oltp_lwm_buf, Ordering::Release);
    }
  }

  /// 对其它 worker 的可见性（上游 isVisibleForIt(whom, commit_ts)）
  pub fn is_visible_for_it(&self, whom_worker_id: usize, commit_ts: u64) -> Visibility {
    let snap = self.workers[whom_worker_id]
      .current_snapshot
      .load(Ordering::Acquire)
      & tx_flag::CLEAN_BITS_MASK;
    if snap > commit_ts {
      Visibility::VisibleAlready
    } else {
      Visibility::VisibleNextRound
    }
  }

  /// 本事务对指定 worker 上事务的可见性（上游 isVisibleForMe 三分支）
  ///
  /// `tx_ts`：MSB=1 → commit_ts（已提交）；否则 start_ts（在途）
  pub fn is_visible_for_me(
    &self,
    my_worker_id: usize,
    other_worker_id: usize,
    tx_ts: u64,
    to_write: bool,
    active: &Transaction,
  ) -> bool {
    const MSB: u64 = 1 << 63;
    const MSB_MASK: u64 = !MSB;
    let is_commit_ts = tx_ts & MSB != 0;
    let committed_ts = if is_commit_ts { tx_ts & MSB_MASK } else { 0 };
    let start_ts = tx_ts & MSB_MASK;

    if !to_write && active.is_read_uncommitted() {
      return true;
    }
    if my_worker_id == other_worker_id {
      return true;
    }
    if active.is_read_committed() || active.is_read_uncommitted() {
      if is_commit_ts {
        return true;
      }
      let cached =
        self.workers[my_worker_id].local_snapshot_cache[other_worker_id].load(Ordering::Acquire);
      if cached >= start_ts {
        return true;
      }
      let cur = self.global_clock.load(Ordering::Relaxed) + 1;
      let lcb = self.workers[other_worker_id].commit_tree.lcb(cur);
      self.workers[my_worker_id].local_snapshot_cache[other_worker_id]
        .store(lcb, Ordering::Release);
      self.workers[my_worker_id].local_snapshot_cache_ts[other_worker_id]
        .store(cur, Ordering::Release);
      lcb >= start_ts
    } else if active.at_least_si() {
      if is_commit_ts {
        return active.start_ts() > committed_ts;
      }
      // 上游 ConcurrencyControl.cpp:222-238：快路径 = 本 worker start_tx 时抓取的
      // 全局 all LWM 缓存（非实时 global_all_lwm）
      let all_lwm = self.workers[my_worker_id]
        .local_global_all_lwm_cache
        .load(Ordering::Acquire);
      if start_ts < all_lwm {
        return true;
      }
      // 上游三分支：①缓存时点命中 → 用缓存；②缓存值（不同时点）仍 ≥ start_ts → 直接可见
      let cached =
        self.workers[my_worker_id].local_snapshot_cache[other_worker_id].load(Ordering::Acquire);
      let cache_ts =
        self.workers[my_worker_id].local_snapshot_cache_ts[other_worker_id].load(Ordering::Acquire);
      if cache_ts == active.start_ts() {
        return cached >= start_ts;
      } else if cached >= start_ts {
        return true;
      }
      let lcb = self.workers[other_worker_id]
        .commit_tree
        .lcb(active.start_ts());
      if lcb != 0 {
        self.workers[my_worker_id].local_snapshot_cache[other_worker_id]
          .store(lcb, Ordering::Release);
        self.workers[my_worker_id].local_snapshot_cache_ts[other_worker_id]
          .store(active.start_ts(), Ordering::Release);
        return lcb >= start_ts;
      }
      false
    } else {
      unreachable!("隔离级别全覆盖")
    }
  }

  /// 对全部事务可见（GC 安全判定，上游 isVisibleForAll）
  pub fn is_visible_for_all(&self, ts: u64) -> bool {
    const MSB: u64 = 1 << 63;
    const MSB_MASK: u64 = !MSB;
    if ts & MSB != 0 {
      (ts & MSB_MASK) < self.global_oldest_all_start_ts.load(Ordering::Acquire)
    } else {
      ts < self.global_all_lwm.load(Ordering::Acquire)
    }
  }

  /// 查询其它 worker 事务的提交时间戳（上游 getCommitTimestamp）
  pub fn get_commit_timestamp(&self, worker_id: usize, tx_ts: u64) -> u64 {
    const MSB: u64 = 1 << 63;
    const MSB_MASK: u64 = !MSB;
    if tx_ts & MSB != 0 {
      return tx_ts & MSB_MASK;
    }
    let lcb = self.workers[worker_id].commit_tree.lcb(tx_ts);
    if lcb != 0 { lcb } else { u64::MAX }
  }

  /// 版本插入（HistoryTree 追加；ldb_recovery 提供树实现，此处登记）
  pub fn insert_version(&mut self, wid: usize, entry: VersionEntry) {
    self.workers[wid].history.push(entry);
  }

  /// 版本检索（按 tx_id）
  pub fn retrieve_version(&self, wid: usize, tx_id: u64) -> Option<&VersionEntry> {
    self.workers[wid]
      .history
      .iter()
      .rev()
      .find(|v| v.tx_id == tx_id)
  }

  /// 历史回收（上游 garbageCollection：按 LWM 清版本 + graveyard 迁移）
  pub fn garbage_collection(&mut self) {
    let snapshots: Vec<AtomicU64> = self
      .workers
      .iter()
      .map(|w| AtomicU64::new(w.current_snapshot.load(Ordering::Relaxed)))
      .collect();
    for w in &self.workers {
      w.commit_tree.clean_if_necessary(&snapshots);
    }
    let all_lwm = self.global_all_lwm.load(Ordering::Acquire);
    let oltp_lwm = self.global_oltp_lwm.load(Ordering::Acquire);
    let any_olap = self.workers.iter().any(|x| x.olap_mode);
    let hook = self.graveyard_hook.read().clone();
    for w in &mut self.workers {
      // 1) 清 all_lwm 之前的版本
      let mut kept = Vec::with_capacity(w.history.len());
      for v in take(&mut w.history) {
        if v.tx_id < all_lwm {
          if let Some(cb) = hook.as_ref() {
            cb(v.tx_id, v.dt_id, &v.payload[..v.payload_len as usize], true);
          }
        } else {
          kept.push(v);
        }
      }
      // 2) graveyard：oltp_lwm 前的 remove 版本迁移
      if any_olap && all_lwm != oltp_lwm && oltp_lwm > 0 {
        let mut still = Vec::with_capacity(kept.len());
        for v in kept {
          if v.is_remove && v.tx_id < oltp_lwm {
            if let Some(cb) = hook.as_ref() {
              cb(
                v.tx_id,
                v.dt_id,
                &v.payload[..v.payload_len as usize],
                false,
              );
            }
          } else {
            still.push(v);
          }
        }
        w.history = still;
      } else {
        w.history = kept;
      }
    }
  }
}

/// Worker 事务生命周期（上游 `cr::Worker` 的 tx 外壳，`Worker.cpp:59-202`）
#[derive(Debug)]
pub struct Worker {
  /// worker 编号
  pub wid: usize,
  /// 活动事务（提交后停留在 `Precommitted`，由 durable 阶段推进到 `Committed`；
  /// 上游 `commitTX` 不搬移 `active_tx`，下一次 `startTX` 覆盖 —— `Worker.cpp:123-175`）
  pub active: Transaction,
}

impl Worker {
  /// 构造
  pub fn new(wid: usize) -> Self {
    Self {
      wid,
      active: Transaction::default(),
    }
  }

  /// 开启事务（登记快照 + 隔离模式切换；上游 `Worker::startTX`，`Worker.cpp:59-121`）
  pub fn start_tx(&mut self, cc: &ConcurrencyControl, isolation: IsolationLevel) {
    let prev_isolation = self.active.isolation;
    // 上游：仅当上一事务处于另一隔离类时切换模式（Worker.cpp:97-119）
    let is_si_next = matches!(
      isolation,
      IsolationLevel::SnapshotIsolation | IsolationLevel::Serializable
    );
    if is_si_next {
      if matches!(
        prev_isolation,
        IsolationLevel::ReadCommitted | IsolationLevel::ReadUncommitted
      ) {
        cc.switch_to_snapshot_isolation_mode(self.wid);
      }
    } else if matches!(
      prev_isolation,
      IsolationLevel::SnapshotIsolation | IsolationLevel::Serializable
    ) {
      cc.switch_to_read_committed_mode(self.wid);
    }
    // LATCH 协议：先以旧 start_ts 置闩位，取号后写最终快照（Worker.cpp:104-110）
    let cc_w = &cc.workers[self.wid];
    cc_w.current_snapshot.store(
      self.active.start_ts_val | tx_flag::LATCH_BIT,
      Ordering::Release,
    );
    let ts = cc.global_clock.fetch_add(1, Ordering::Relaxed);
    self.active.initialize(ts, isolation);
    let flag = if is_si_next { 0 } else { tx_flag::RC_BIT }
      | if self.active.olap {
        tx_flag::OLAP_BIT
      } else {
        0
      };
    cc_w.current_snapshot.store(ts | flag, Ordering::Release);
    cc_w.local_latest_write_tx.store(ts, Ordering::Release);
    // 上游 Worker.cpp:113：start_tx 时抓取全局 all LWM 快照供 SI 可见性快路径
    cc_w
      .local_global_all_lwm_cache
      .store(cc.global_all_lwm.load(Ordering::Acquire), Ordering::Release);
  }

  /// 提交事务（CommitTree 取号；上游 `Worker::commitTX`，`Worker.cpp:123-175`）
  pub fn commit_tx(&mut self, cc: &ConcurrencyControl) -> u64 {
    let start_ts = self.active.start_ts();
    let commit_ts = cc.workers[self.wid]
      .commit_tree
      .commit(start_ts, &cc.global_clock);
    // 上游 Worker.cpp:144：local_latest_write_tx 记的是最新**写**事务的 commit_ts
    cc.workers[self.wid]
      .local_latest_write_tx
      .store(commit_ts, Ordering::Release);
    self.active.state = TxnState::Precommitted;
    self.active.commit_ts_val = commit_ts;
    cc.refresh_global_state();
    commit_ts
  }

  /// durable 阶段定稿（上游 `TransactionManager::DurableCommit`：`txn.state = COMMITTED`，
  /// `transaction_manager.cc:153-154`）。仅推进仍未定稿的本 worker 活动事务
  pub(crate) fn finalize_durable(&mut self, commit_ts: u64) {
    if self.active.commit_ts_val == commit_ts && self.active.is_durable() {
      self.active.state = TxnState::Committed;
    }
  }

  /// 回滚事务（上游 Undo 未实现 —— 本实现状态回滚 + 版本登记交由 HistoryTree）
  pub fn abort_tx(&mut self, cc: &mut ConcurrencyControl) {
    self.active.state = TxnState::Aborted;
    cc.workers[self.wid]
      .current_snapshot
      .store(0, Ordering::Release);
    // 已插入的未提交版本须撤销：从历史里摘除本事务条目
    let start = self.active.start_ts();
    let wid = self.wid;
    cc.workers[wid].history.retain(|v| v.tx_id != start);
    self.active.reset_state();
  }
}
