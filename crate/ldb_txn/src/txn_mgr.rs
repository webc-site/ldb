//! 事务管理器（上游 `leanstore.latency transaction_manager.cc` + ZLeanStore）。
//!
//! 实现 L2 `TxnLogHook`（log_pid2_offset / log_valid_page / log_free_page），
//! 供 L3 `ldb_space` / `ldb_gc` 注入 —— 消除上游 L3→L4 上向边。
//!
//! A4 修复（2026-11-11）：事务→WAL 链从空壳接通 ——
//! - `commit_transaction` 写 `TX_COMMIT` 条目 + 预提交入队（原版不写日志）；
//! - `abort_transaction` 写 `TX_ABORT`（C-21：上游两仓库均抛未实现异常）；
//! - `durable_commit` await 共享 GsnWatermark 到票据 GSN（原版原样返回票据）；
//! - `WalTxnLogHook` 三方法写真实 WAL 条目（原版全 `Box::pin(ready(()))`）。

use std::{
  fmt,
  rc::Rc,
  sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
  },
};

use futures_lite::future::yield_now;
use ldb_page::hook::TxnLogHook;
use ldb_stat::StatCollector;
use ldb_type::PageId;
use ldb_wal::{
  CommitVariant, LogManager, LogWorker, PreCommitTicket, WalFileState, WalSink, entry::EntryType,
};
use parking_lot::Mutex;

use crate::{
  error::{Error, Result},
  osic_mvcc::{ConcurrencyControl, Worker},
  txn::IsolationLevel,
};

/// 事务管理器（`Arc` 共享形态：worker 状态经内部 `Mutex` 承载，
/// 全部方法 `&self` —— 锁作用域严格不跨 `.await`，见各方法实现）
pub struct TransactionManager {
  /// 每 worker 事务外壳
  pub workers: Vec<Mutex<Worker>>,
  /// 并发控制（OSIC）
  pub cc: Mutex<ConcurrencyControl>,
  /// 日志协调者（worker 组 + 共享 GSN 水位 + sink；`Rc` —— compio 单线程每核，
  /// LogManager 内含线程局部 sink（RefCell），本就非 Send/Sync）
  pub log_mgr: Rc<LogManager>,
  /// 本地日志器（0 号 worker；提交条目统一走它）
  pub log: Arc<LogWorker>,
  /// 全局运行标记（与 log_mgr 共享）
  pub is_running: Arc<AtomicBool>,
  /// 统计收集器（L5 注入；默认空实例，未注入时计数丢弃）
  stats: Arc<StatCollector>,
}

impl fmt::Debug for TransactionManager {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("TransactionManager")
      .field("workers", &self.workers.len())
      .finish_non_exhaustive()
  }
}

impl TransactionManager {
  /// 构造（`sink` / `file` 装配进 `LogManager` —— 原版这两个参数被丢弃）
  pub fn new(
    worker_cnt: usize,
    buffer_kb: usize,
    variant: CommitVariant,
    sink: Box<dyn WalSink>,
    file: WalFileState,
  ) -> Self {
    let log_mgr = Rc::new(LogManager::new(
      worker_cnt,
      buffer_kb * 1024,
      variant,
      sink,
      file,
    ));
    let is_running = log_mgr.is_running.clone();
    let log = log_mgr.local_log_worker(0).clone();
    Self {
      workers: (0..worker_cnt)
        .map(|wid| Mutex::new(Worker::new(wid)))
        .collect(),
      cc: Mutex::new(ConcurrencyControl::new(worker_cnt)),
      log_mgr,
      log,
      is_running,
      stats: Arc::new(StatCollector::default()),
    }
  }

  /// 注入统计收集器（L5 装配；须在 `Rc` 化前调用）。
  pub fn set_stats(&mut self, stats: Arc<StatCollector>) {
    self.stats = stats;
  }

  /// 日志协调者引用
  #[inline]
  pub fn log_manager(&self) -> &Rc<LogManager> {
    &self.log_mgr
  }

  /// 隔离级别解析（上游 ParseIsolationLevel：字符串枚举）
  pub fn parse_isolation_level(s: &str) -> IsolationLevel {
    match s.to_ascii_lowercase().as_str() {
      "ru" | "read-uncommitted" | "read_uncommitted" => IsolationLevel::ReadUncommitted,
      "rc" | "read-committed" | "read_committed" => IsolationLevel::ReadCommitted,
      "ser" | "serializable" => IsolationLevel::Serializable,
      _ => IsolationLevel::SnapshotIsolation,
    }
  }

  /// 开启事务
  pub fn start_transaction(&self, wid: usize, isolation: IsolationLevel) {
    let cc = self.cc.lock();
    self.workers[wid].lock().start_tx(&cc, isolation);
  }

  /// 提交事务：WAL `TX_COMMIT` → CommitTree 取号 → 预提交队列入队，
  /// 返回票据（调用方以 [`Self::durable_commit`] 等待固化）。
  /// 锁作用域（worker / cc）均在 `.await` 前释放
  pub async fn commit_transaction(&self, wid: usize) -> Result<PreCommitTicket> {
    // P1-6 状态机闭环：未运行事务提交 → 显式错误（原版静默产生幻影提交）
    let (commit_ts, start_ts) = {
      let cc = self.cc.lock();
      let mut w = self.workers[wid].lock();
      if !w.active.is_running() {
        return Err(Error::InvalidTxnState(format!("worker {wid} 无活动事务")));
      }
      // P1-4：commit_tx 会 take(active)（start_ts 归零）→ 先捕
      let start = w.active.start_ts();
      w.active.mark_as_write();
      let ts = w.commit_tx(&cc);
      (ts, start)
    };
    // WAL TX_COMMIT 条目；P1-5：submit 内部推进本地 GSN，
    // 票据观测提交条目自身的 GSN（submit 后的时钟值）
    self
      .log
      .reserve_and_submit(
        &self.log_mgr,
        EntryType::TxCommit,
        0,
        &commit_ts.to_le_bytes(),
        commit_ts,
      )
      .await
      .map_err(|_| Error::WalReserveFailed)?;
    let ticket = PreCommitTicket {
      commit_ts,
      max_observed_gsn: self.log.get_current_gsn(),
      start_ts,
      blob_page: 0,
    };
    self.queue_transaction(ticket);
    self.stats.record_txn_commit();
    Ok(ticket)
  }

  /// 回滚事务（C-21 修复：上游抛未实现异常 —— 撤销未提交版本 + 写 `TX_ABORT`
  /// 记录，恢复侧据此跳过该事务全部条目）
  pub async fn abort_transaction(&self, wid: usize) -> Result<()> {
    {
      let mut cc = self.cc.lock();
      let w = self.workers[wid].lock();
      // P1-6：对已终态事务写 TX_ABORT = 恢复侧幻影回滚 → 显式拒绝
      if !w.active.is_running() {
        drop(w);
        return Err(Error::InvalidTxnState(format!("worker {wid} 无活动事务")));
      }
      drop(w);
      self.workers[wid].lock().abort_tx(&mut cc);
    }
    self
      .log
      .reserve_and_submit(&self.log_mgr, EntryType::TxAbort, 0, &[], 0)
      .await
      .map_err(|_| Error::WalReserveFailed)?;
    self.stats.record_txn_abort();
    Ok(())
  }

  /// 屏障事务（上游 AddBarrierTransaction）：先记当前全局时钟快照，等
  /// `global_all_lwm` 追平快照（全部在途事务固化、无更早活跃者）后取
  /// 未来号 —— 自身提交后，任何 start_ts < 屏障号的事务都已完成，
  /// 屏障号即「全可见点」
  pub async fn add_barrier_transaction(&self) -> u64 {
    let snapshot = {
      let cc = self.cc.lock();
      cc.global_clock.load(Ordering::Acquire)
    };
    loop {
      let lwm = self.cc.lock().global_all_lwm.load(Ordering::Acquire);
      if lwm >= snapshot {
        break;
      }
      yield_now().await;
    }
    self.cc.lock().global_clock.fetch_add(2, Ordering::Relaxed)
  }

  /// 持久化提交（上游 DurableCommit：等待组提交把共享 GSN 水位推过票据的
  /// 观测 GSN —— 原版原样返回票据，等于不等待落盘）。
  ///
  /// 组提交 task（L5 spawn）负责推进 `min_gsn_flushed`；本协程让出等待，
  /// 不自旋、不阻塞事件循环。
  pub async fn durable_commit(&self, ticket: PreCommitTicket) -> Result<PreCommitTicket> {
    loop {
      if self.log_mgr.min_gsn_flushed.load() >= ticket.max_observed_gsn {
        // durable 定稿（上游 `DurableCommit`：`txn.state = COMMITTED`）：把持有该
        // commit_ts 的 worker 活动事务推进到 Committed。持锁不外跨 `.await`
        for w in &self.workers {
          w.lock().finalize_durable(ticket.commit_ts);
        }
        return Ok(ticket);
      }
      if !self.is_running.load(Ordering::Relaxed) {
        return Err(Error::WalReserveFailed);
      }
      // WILO 自驱落盘（组提交 task 未运行时的唯一水位推进点）：
      // 强制刷本 worker 缓冲，再把全局水位抬到本 worker 当前 GSN ——
      // 票据条目已 durable，watermark ≥ 票据 GSN 为真值
      self
        .log
        .worker_writes_log(&self.log_mgr, true)
        .await
        .map_err(|_| Error::WalReserveFailed)?;
      self
        .log_mgr
        .min_gsn_flushed
        .advance_min(self.log.get_current_gsn());
      yield_now().await;
    }
  }

  /// 预提交队列入队（组提交消费）
  pub fn queue_transaction(&self, ticket: PreCommitTicket) {
    ldb_wal::enqueue_precommitted(&self.log, ticket);
  }
}

/// TxnLogHook 实现（注入 `SpaceManager<T>` / `GarbageCollector<T>`：
/// 映射变更 / 有效页镜像 / 页释放全走真实 WAL 条目）
#[derive(Debug, Default)]
pub struct WalTxnLogHook {
  /// 日志 worker（提交条目）
  worker: Option<Arc<LogWorker>>,
  /// 日志协调者（空间预算 / 停机标记；`Rc` 见 TransactionManager 注）
  log_mgr: Option<Rc<LogManager>>,
}

impl WalTxnLogHook {
  /// 构造（绑定提交链路）
  pub fn new(worker: Arc<LogWorker>, log_mgr: Rc<LogManager>) -> Self {
    Self {
      worker: Some(worker),
      log_mgr: Some(log_mgr),
    }
  }

  /// 预留 + 提交一条 WAL 条目；失败记日志（hook 契约 `Output = ()`：
  /// WAL 满属上游致命态，组提交/空间预算推进即释放，不 panic）
  async fn write_entry(&self, ty: EntryType, pid: PageId, payload: &[u8]) {
    let (Some(w), Some(m)) = (&self.worker, &self.log_mgr) else {
      log::warn!("WalTxnLogHook 未绑定 LogManager，丢弃 {ty:?} 条目");
      return;
    };
    if let Err(e) = w.reserve_and_submit(m, ty, pid, payload, 0).await {
      log::error!("WAL 条目 {ty:?}(pid={pid}) 写入失败: {e}");
    }
  }
}

impl TxnLogHook for WalTxnLogHook {
  /// 记录 PID→偏移映射变更（上游 `SpaceManager::LogPID2Offset` → `ReserveDataLog`）
  async fn log_pid2_offset(&self, pid: PageId) {
    self
      .write_entry(EntryType::DataEntry, pid, &pid.to_le_bytes())
      .await;
  }

  /// 记录有效页镜像（GC 搬移前；DataEntry 载页号）
  async fn log_valid_page(&self, pid: PageId) {
    self
      .write_entry(EntryType::DataEntry, pid, &pid.to_le_bytes())
      .await;
  }

  /// 记录页释放（上游 `LogFreePage` → FreePage 条目）
  async fn log_free_page(&self, pid: PageId) {
    self.write_entry(EntryType::FreePage, pid, &[]).await;
  }
}

/// 预提交可见性门（PhaseOne 裁剪条件复刻）
pub fn barrier_ready(ticket: &PreCommitTicket, min_all_gsn: u64, min_hardened_ts: u64) -> bool {
  ticket.max_observed_gsn <= min_all_gsn && ticket.start_ts <= min_hardened_ts
}
