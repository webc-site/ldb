//! 组提交执行器（上游 `ZLeanStore/src/recovery/group_commit.cc` 5 阶段 +
//! `leanstore.latency` WILO 协作）。
//!
//! - PhaseOne：收集各 worker 一致状态（RFA）+ 集中变体（Baseline）聚合写脏 WAL；
//!   WILO 变体 worker 自写自盘，组提交不重复写（latency group_commit.cc:205-208）
//! - PhaseTwo：fdatasync（同一 fd 覆盖 worker 与组两条写出路径）
//! - PhaseThree：Blob 大页预写（remote-flush + RFA 两队列切点展开）
//! - PhaseFour：Blob 落盘后 `EvictHook::evict_extent`（L2 注入，消除 ldb_wal→ldb_buf 环）
//! - PhaseFive：集中变体推进 gct_cursor + 按 GSN/TS 条件固化两队列事务
//!
//! 上游 C-21 的 `AbortTransaction` 缺陷与本文件无关；ZLeanStore PhaseOne 的
//! `AddCheckpointTarget` 分片检查点编排保留于集中写路径（latency 版没有）。

use std::{collections::BTreeMap, fmt, mem::take, sync::atomic::Ordering, time::Duration};

use compio::time::sleep;
use crossfire::mpmc::List;
use futures_lite::future::yield_now;
use gxhash::HashSet as GxHashSet;
use ldb_page::hook::EvictHook;
use roaring::RoaringTreemap;

use crate::{
  error::Result,
  log_worker::{
    CommitVariant, LogManager, LogWorker, PreCommitTicket, ToCommitState, WAL_BUFFER_HEADROOM,
    WAL_CIRC, WAL_WRITE_ALIGN,
  },
};

/// 组提交配置（构造 9 参收敛）
#[derive(Debug, Clone, Copy)]
pub struct GcCfg {
  /// 组内起始 worker（含）
  pub start_wid: usize,
  /// 组内结束 worker（不含）
  pub end_wid: usize,
  /// 聚合写缓冲
  pub write_buffer_size: usize,
  /// WAL 环形区起始
  pub wal_start: u64,
  /// WAL 环形区结束
  pub wal_end: u64,
  /// 分片大小
  pub shard_wal_size: u64,
}

/// 组提交执行器（一个 worker 组一个实例）
pub struct GroupCommitExecutor<G: EvictHook> {
  /// 组内起始 worker（含）
  pub start_wid: usize,
  /// 组内结束 worker（不含）
  pub end_wid: usize,
  /// 组提交聚合写缓冲
  write_buffer: Box<[u8]>,
  buffer_offset: usize,
  /// WAL 前向环形区（组提交模式）
  w_start_offset: u64,
  w_end_offset: u64,
  w_cur_offset: u64,
  /// 分片大小（检查点目标粒度）
  pub shard_wal_size: u64,
  /// 每分片检查点目标：shard -> (offset, gsn)
  /// 分片检查点目标（BTreeMap：按分片序推进；冷路径）
  cp_target: BTreeMap<u64, (u64, u64)>,
  last_target_shard: u64,
  checkpoint_added: bool,
  /// 检查点已推进偏移
  checkpoint_offset: u64,
  round: u64,
  completed_txn: u64,
  // RFA 极值
  min_all_workers_gsn: u64,
  max_all_workers_gsn: u64,
  min_hardened_commit_ts: u64,
  /// 各 worker 预提交切点（remote-flush 队列；上游 ready_to_commit_cut_）
  ready_cut: Vec<usize>,
  /// RFA 组队列切点（上游 ready_to_commit_rfa_cut_，latency CollectConsistentState:283-289），
  /// phase_three/blob 与 phase_five/固化消费
  ready_rfa_cut: Vec<usize>,
  /// 各 worker 一致状态快照
  worker_states: Vec<ToCommitState>,
  /// 待写 Blob 大页请求 (start_pid, page_cnt)
  lp_req: Vec<(u64, u64)>,
  already_prep: GxHashSet<u64>,
  completed_lp: roaring::RoaringTreemap,
  has_blob: bool,
  /// Blob 驱逐钩子（PhaseFour）
  evict: G,
  /// 固化事务通知管道（crossfire；L4 消费）
  durable_tx: Option<crossfire::MTx<List<PreCommitTicket>>>,
  /// 检查点就绪通知（shard, offset, gsn）
  checkpoint_tx: Option<crossfire::MTx<List<(u64, u64, u64)>>>,
}

impl<G: EvictHook> fmt::Debug for GroupCommitExecutor<G> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("GroupCommitExecutor")
      .field("workers", &(self.start_wid..self.end_wid))
      .field("round", &self.round)
      .field("completed_txn", &self.completed_txn)
      .field("w_cur_offset", &self.w_cur_offset)
      .finish_non_exhaustive()
  }
}

impl<G: EvictHook> GroupCommitExecutor<G> {
  /// 构造（9 参收敛：组拓扑走 [`GcCfg`]）
  pub fn new(
    cfg: GcCfg,
    evict: G,
    durable_tx: Option<crossfire::MTx<List<PreCommitTicket>>>,
    checkpoint_tx: Option<crossfire::MTx<List<(u64, u64, u64)>>>,
  ) -> Self {
    let cnt = cfg.end_wid.saturating_sub(cfg.start_wid);
    debug_assert!(
      cfg.wal_start.is_multiple_of(WAL_WRITE_ALIGN as u64)
        && cfg.wal_end.is_multiple_of(WAL_WRITE_ALIGN as u64)
        && cfg.shard_wal_size.is_multiple_of(WAL_WRITE_ALIGN as u64)
        && cfg.write_buffer_size.is_multiple_of(WAL_WRITE_ALIGN),
      "组提交 WAL 区段 / 聚合缓冲须 4KiB 对齐"
    );
    Self {
      start_wid: cfg.start_wid,
      end_wid: cfg.end_wid,
      write_buffer: vec![0u8; cfg.write_buffer_size].into_boxed_slice(),
      buffer_offset: 0,
      w_start_offset: cfg.wal_start,
      w_end_offset: cfg.wal_end,
      w_cur_offset: cfg.wal_start,
      shard_wal_size: cfg.shard_wal_size,
      cp_target: BTreeMap::new(),
      last_target_shard: 0,
      checkpoint_added: false,
      checkpoint_offset: cfg.wal_start,
      round: 0,
      completed_txn: 0,
      min_all_workers_gsn: u64::MAX,
      max_all_workers_gsn: 0,
      min_hardened_commit_ts: u64::MAX,
      ready_cut: vec![0; cnt],
      ready_rfa_cut: vec![0; cnt],
      worker_states: vec![ToCommitState::default(); cnt],
      lp_req: Vec::new(),
      already_prep: GxHashSet::default(),
      completed_lp: RoaringTreemap::new(),
      has_blob: false,
      evict,
      durable_tx,
      checkpoint_tx,
    }
  }

  /// 已固化事务数
  #[inline]
  pub fn completed_txn_cnt(&self) -> u64 {
    self.completed_txn
  }

  /// 聚合单个 worker 的脏日志段 `[gct_cursor, last_wal_cursor)`（含环形回绕，
  /// 缓冲满即刷出 —— 上游 WriteLogBuffer + PrepareWrite 完整语义）
  async fn prepare_worker_dirty_range(
    &mut self,
    mgr: &LogManager,
    logger: &LogWorker,
    gct: usize,
    written: usize,
  ) -> Result<()> {
    if written > gct {
      let seg = logger.buffer.lock().slice(gct, written - gct).to_vec();
      self.prepare_write_all(&seg, mgr).await
    } else if written < gct {
      let (seg1, seg2) = {
        let buf = logger.buffer.lock();
        (
          buf.slice(gct, buf.capacity() - gct).to_vec(),
          buf.slice(0, written).to_vec(),
        )
      };
      self.prepare_write_all(&seg1, mgr).await?;
      self.prepare_write_all(&seg2, mgr).await
    } else {
      Ok(())
    }
  }

  /// 轮次初始化（上游 InitializeRound）
  pub fn initialize_round(&mut self) {
    self.round += 1;
    self.lp_req.clear();
    self.already_prep.clear();
    self.has_blob = false;
    self.completed_txn = 0;
    self.min_all_workers_gsn = u64::MAX;
    self.max_all_workers_gsn = 0;
    self.min_hardened_commit_ts = u64::MAX;
  }

  /// 轮次收尾：发布 RFA 水位（上游 latency CompleteRound `UpdateMax`：
  /// group_commit.cc:112-113 —— 单调推进，与 ZLeanStore 的 store 语义不同；
  /// 水位与 ldb_buf/ldb_txn 共享同一 GsnWatermark（D5），后者可独立推进，
  /// 故无 ZLeanStore 版 `assert(global_min <= min_all_workers_gsn)` 的前置不变量）
  pub fn complete_round(&mut self, mgr: &LogManager) {
    if self.min_all_workers_gsn != u64::MAX {
      mgr.min_gsn_flushed.advance_min(self.min_all_workers_gsn);
    }
    mgr
      .global_sync_gsn
      .store(self.max_all_workers_gsn, Ordering::Release);
    // 周期性防漂移同步：把 worker 本地 GSN 拉齐到全局（上游在
    // TransactionManager 事务起点做同向同步：transaction_manager.cc:52-55）
    for wid in self.start_wid..self.end_wid {
      if let Some(w) = mgr.worker(wid)
        && w.get_current_gsn() < self.max_all_workers_gsn
      {
        w.set_current_gsn(self.max_all_workers_gsn);
      }
    }
  }

  /// 聚合写：拷入能容纳的前缀，返回已消费字节数（上游 PrepareWrite 的
  /// 「缓冲满 → 立即 FlushBuffer」语义 —— 刷盘需 `.await` sink，故由 async
  /// 调用方在返回值未消费完时先 `flush_buffer` 再续写）。
  ///
  /// 修复 A1：原版 `while done < src.len()` 在 `room == 0` 时 `chunk == 0`
  /// 空转且无 break/await —— 缓冲满即整线程挂死。
  pub fn prepare_write(&mut self, src: &[u8]) -> usize {
    let room = self.write_buffer.len() - self.buffer_offset;
    let chunk = src.len().min(room);
    self.write_buffer[self.buffer_offset..self.buffer_offset + chunk]
      .copy_from_slice(&src[..chunk]);
    self.buffer_offset += chunk;
    chunk
  }

  /// 聚合写全部字节：缓冲满即刷出后续写（上游 PrepareWrite 完整语义）
  pub async fn prepare_write_all(&mut self, mut src: &[u8], mgr: &LogManager) -> Result<()> {
    while !src.is_empty() {
      let n = self.prepare_write(src);
      src = &src[n..];
      if !src.is_empty() {
        self.flush_buffer(mgr).await?;
      }
    }
    Ok(())
  }

  /// 刷出聚合缓冲到 WAL 文件（环形回绕；上游 FlushBuffer 的
  /// `UpAlign(buffer_offset_)` 按 `BLOCK_ALIGNMENT_MASK = 4095` 对齐 —— 常量
  /// `ZLeanStore/src/include/common/constants.h:22`）。
  ///
  /// 修复 A5：sink 写失败也必须先 put-back 再传播错误 —— 原版 `?` 在
  /// put-back 之前，首个写失败后 `LogManager.sink` 恒为 `None`，其后全部
  /// WAL 写走 `None => Ok(())` 静默成功。
  pub async fn flush_buffer(&mut self, mgr: &LogManager) -> Result<()> {
    let aligned = self.buffer_offset.next_multiple_of(WAL_WRITE_ALIGN);
    if aligned == 0 {
      return Ok(());
    }
    // 尾部填充清零（段重放契约：段尾零填充终止；缓冲复用残留必须清掉）
    self.write_buffer[self.buffer_offset..aligned].fill(0);
    let data = self.write_buffer[..aligned].to_vec();
    // swap-out：await 期间不持有 RefCell 借用（Option take/put-back，
    // 无堆分配与胖指针替换 —— 原实现每轮构造 Box<NullSinkMarker> 胖指针）
    let sink = mgr.sink.borrow_mut().take();
    let Some(mut s) = sink else {
      // 无 sink（未接线/停机排水）：丢弃缓冲保持推进，语义与写侧一致
      self.buffer_offset = 0;
      return Ok(());
    };
    let r: Result<()> = async {
      if self.w_cur_offset + aligned as u64 > self.w_end_offset {
        let first = (self.w_end_offset - self.w_cur_offset) as usize;
        let second = aligned - first;
        s.write_wal(data[..first].to_vec(), self.w_cur_offset)
          .await?;
        s.write_wal(data[first..].to_vec(), self.w_start_offset)
          .await?;
        self.w_cur_offset = self.w_start_offset + second as u64;
      } else {
        s.write_wal(data, self.w_cur_offset).await?;
        self.w_cur_offset += aligned as u64;
      }
      Ok(())
    }
    .await;
    *mgr.sink.borrow_mut() = Some(s); // 无条件 put-back（A5）
    r?;
    self.buffer_offset = 0;
    Ok(())
  }

  /// 一轮组提交（上游 ExecuteOneRound）
  pub async fn execute_one_round(&mut self, mgr: &LogManager) -> Result<()> {
    assert!(
      mgr.is_running.load(Ordering::Relaxed),
      "组提交在稳态崩溃（上游 StartExecution FATAL 语义）"
    );
    self.initialize_round();
    self.phase_one(mgr).await?;
    self.phase_two(mgr).await?;
    // Blob 阶段：无 lp_req 时跳过（上游 FLAGS_enable_blob 门控）
    self.phase_three(mgr);
    self.phase_four(mgr).await?;
    self.phase_five(mgr);
    self.complete_round(mgr);
    Ok(())
  }

  /// 执行主循环（上游 StartExecution；由运行时 spawn，不可重入）
  pub async fn start_execution(&mut self, mgr: &LogManager) -> Result<()> {
    while mgr.is_running.load(Ordering::Relaxed) {
      self.execute_one_round(mgr).await?;
      yield_now().await;
    }
    Ok(())
  }

  /// PhaseOne：收集状态 + 写 WAL（含分片检查点触发，ZLeanStore 特有）。
  ///
  /// 变体门控（上游 `leanstore.latency/src/recovery/group_commit.cc:205-208`）：
  /// WILO 变体（WorkersWriteLog / WiloSteal）worker 自写自盘，组提交只收集
  /// 一致状态；集中变体（Baseline）由组聚合写出 —— 原实现不分变体一律写，
  /// 与 worker persist_log 构成同一字节双重落盘 + 前/后向分配器同区竞争。
  pub async fn phase_one(&mut self, mgr: &LogManager) -> Result<()> {
    let centralized = mgr
      .worker(self.start_wid)
      .is_some_and(|w| w.variant == CommitVariant::Baseline);
    for w_i in self.start_wid..self.end_wid {
      let Some(logger) = mgr.worker(w_i) else {
        continue;
      };
      let Some(st) = self.collect_consistent_state(mgr, w_i) else {
        continue;
      };

      if centralized {
        // 分片检查点触发（上游 ZLeanStore group_commit.cc:272-280：以前向游标
        // `w_offset_ - w_start_offset_` 判分片）—— 原实现误用 mgr.wal_offset
        //（worker persist_log 的反向下探分配器，初始即 = w_end_offset，语义相反）
        if mgr.min_gsn_flushed.load() > 0
          && (self.w_cur_offset - self.w_start_offset) / self.shard_wal_size
            > self.last_target_shard
          && self.w_cur_offset != self.w_end_offset
        {
          self.add_checkpoint_target(mgr.min_gsn_flushed.load(), mgr);
          self.last_target_shard += 1;
        }
        // WAL 写到尾 → 回卷（上游 ZLeanStore group_commit.cc:295-303：
        // `w_offset_ + WAL_BUFFER_SIZE >= w_end_offset_`，余量 = 2 MiB
        // WAL_BUFFER_SIZE（log_worker.h:25）—— 复位的是执行器自己的前向游标）
        if self.w_cur_offset + WAL_BUFFER_HEADROOM >= self.w_end_offset {
          self.w_cur_offset = self.w_start_offset;
          self.checkpoint_added = false;
          self.last_target_shard = 0;
          // 上游 TrimWAL(sudo nvme dsm) 已剔除 → punch_hole 由 L5 定期执行
        }
        // 把 [gct_cursor, written) 写入组缓冲
        let gct = logger.gct_cursor.load(Ordering::Acquire) & !WAL_CIRC;
        let written = st.last_wal_cursor as usize;
        self
          .prepare_worker_dirty_range(mgr, logger, gct, written)
          .await?;
      }
    }
    self.flush_buffer(mgr).await
  }

  /// 登记分片检查点目标（上游 ZLeanStore AddCheckpointTarget :360-371：
  /// 目标偏移 = 执行器前向游标 `w_offset_`，分片按同一游标计算）
  pub fn add_checkpoint_target(&mut self, min_flushed_gsn: u64, _mgr: &LogManager) {
    let shard = (self.w_cur_offset - self.w_start_offset) / self.shard_wal_size;
    self
      .cp_target
      .insert(shard, (self.w_cur_offset, min_flushed_gsn));
    self.checkpoint_added = true;
  }

  /// PhaseTwo：fdatasync（fsync 覆盖同一 fd 上 worker persist_log 与组聚合
  /// 两条写入路径的已写出字节）。集中变体刷盘后推进各 worker 的 write_cursor
  /// （上游 latency group_commit.cc:224-231：write_cursor.store(last_wal_cursor)）
  pub async fn phase_two(&mut self, mgr: &LogManager) -> Result<()> {
    let sink = mgr.sink.borrow_mut().take();
    let r = match sink {
      Some(s) => {
        let r = s.sync().await;
        *mgr.sink.borrow_mut() = Some(s);
        r
      }
      None => Ok(()),
    };
    r?;
    let centralized = mgr
      .worker(self.start_wid)
      .is_some_and(|w| w.variant == CommitVariant::Baseline);
    if centralized {
      for w_i in self.start_wid..self.end_wid {
        if let Some(logger) = mgr.worker(w_i) {
          let idx = w_i - self.start_wid;
          logger.buffer.lock().write_cursor.store(
            self.worker_states[idx].last_wal_cursor as usize,
            Ordering::Release,
          );
        }
      }
    }
    Ok(())
  }

  /// PhaseThree：收集待写大页（上游 latency PREPARE_LARGE_PAGE_WRITE 宏
  /// `group_commit.cc:22-31`：remote-flush 与 RFA 两队列都按切点展开）
  pub fn phase_three(&mut self, mgr: &LogManager) {
    for w_i in self.start_wid..self.end_wid {
      let Some(logger) = mgr.worker(w_i) else {
        continue;
      };
      let idx = w_i - self.start_wid;
      let cut = self.ready_cut[idx];
      let rfa_cut = self.ready_rfa_cut[idx];
      let (q, qr): (Vec<PreCommitTicket>, Vec<PreCommitTicket>) = {
        let a = logger.precommitted.lock();
        let b = logger.precommitted_rfa.lock();
        (
          a.iter().take(cut).copied().collect(),
          b.iter().take(rfa_cut).copied().collect(),
        )
      };
      for t in q.into_iter().chain(qr) {
        self.prepare_large_page_write(t);
      }
    }
  }

  /// 大页写预备（上游 PrepareLargePageWrite）
  pub fn prepare_large_page_write(&mut self, t: PreCommitTicket) {
    if t.blob_page != 0 && self.already_prep.insert(t.blob_page) {
      self.lp_req.push((t.blob_page, 1));
      self.has_blob = true;
    }
  }

  /// PhaseFour：Blob 落盘后驱逐 extent（EvictHook 注入点）
  pub async fn phase_four(&mut self, mgr: &LogManager) -> Result<()> {
    if self.has_blob {
      let sink = mgr.sink.borrow_mut().take();
      let r = match sink {
        Some(s) => {
          let r = s.sync().await;
          *mgr.sink.borrow_mut() = Some(s);
          r
        }
        None => Ok(()),
      };
      r?;
      let req = take(&mut self.lp_req);
      for (start_pid, pages) in req {
        if self.completed_lp.insert(start_pid) {
          self.evict.evict_extent(start_pid).await;
          let _ = pages;
        }
      }
    }
    Ok(())
  }

  /// PhaseFive：推进游标 + 条件固化事务（上游 ZLeanStore PhaseFive + latency
  /// COMPLETE_NORMAL_QUEUE / COMPLETE_RFA_QUEUE）。
  ///
  /// 固化条件（双上游引用）：
  /// - remote-flush 队列：`max_observed_gsn <= min_all_workers_gsn &&
  ///   start_ts <= min_hardened_commit_ts`（ZLeanStore group_commit.cc:431-433
  ///   用 StartTS；latency COMPLETE_NORMAL_QUEUE :39-41 用 commit_ts —— 本实现
  ///   取 ZLeanStore 侧，与 ldb_txn barrier_ready 同源）
  /// - RFA 队列：`commit_ts <= worker_states[w].precommitted_tx_commit_ts`
  ///   （latency COMPLETE_RFA_QUEUE :59-63）
  pub fn phase_five(&mut self, mgr: &LogManager) {
    let centralized = mgr
      .worker(self.start_wid)
      .is_some_and(|w| w.variant == CommitVariant::Baseline);
    for w_i in self.start_wid..self.end_wid {
      let Some(logger) = mgr.worker(w_i) else {
        continue;
      };
      let idx = w_i - self.start_wid;
      if centralized {
        // 消费游标仅组写出变体有意义（WILO 变体的写出前沿由 worker
        // persist_log 的 write_cursor 承担 —— latency 组提交不碰 gct）
        logger.gct_cursor.store(
          self.worker_states[idx].last_wal_cursor as usize,
          Ordering::Release,
        );
      }

      // remote-flush 队列：取切点前票据，按 GSN/TS 条件固化
      let cut = self.ready_cut[idx];
      let tickets: Vec<PreCommitTicket> = {
        let mut q = logger.precommitted.lock();
        let n = cut.min(q.len());
        let ready = q[..n]
          .iter()
          .take_while(|t| {
            t.max_observed_gsn <= self.min_all_workers_gsn
              && t.start_ts <= self.min_hardened_commit_ts
          })
          .count();
        q.drain(..ready).collect()
      };
      for t in tickets {
        self.complete_transaction(t);
      }
      // RFA 队列（上游 latency :459-470）
      let rfa_cut = self.ready_rfa_cut[idx];
      let hardened = self.worker_states[idx].precommitted_tx_commit_ts;
      let rfa: Vec<PreCommitTicket> = {
        let mut q = logger.precommitted_rfa.lock();
        let n = rfa_cut.min(q.len());
        let ready = q[..n]
          .iter()
          .take_while(|t| t.commit_ts <= hardened)
          .count();
        q.drain(..ready).collect()
      };
      for t in rfa {
        self.complete_transaction(t);
      }
    }
  }

  /// 固化事务（上游 CompleteTransaction + CompleteTxnBlobs 合并：置 COMMITTED、
  /// 解除 Blob 大页占用标记 —— `completed_lp_.erase`，使同一 blob 页的下轮
  /// 预写/驱逐可再次触发 —— 并通知）。
  pub fn complete_transaction(&mut self, t: PreCommitTicket) {
    self.completed_txn += 1;
    if t.blob_page != 0 {
      self.completed_lp.remove(t.blob_page);
    }
    if let Some(tx) = &self.durable_tx {
      let _ = tx.try_send(t);
    }
  }

  /// fdatasync（上游 Fsync）
  pub async fn fsync(&mut self, mgr: &LogManager) -> Result<()> {
    self.phase_two(mgr).await
  }

  /// 收集单个 worker 一致状态（上游 CollectConsistentState:278-300：队列切点 +
  /// 状态源 + RFA 极值；phase_one 每 worker 调用）。
  /// WiloSteal 读 commit_state（durable 发布态），其余读 worker 发布态
  pub fn collect_consistent_state(
    &mut self,
    mgr: &LogManager,
    w_i: usize,
  ) -> Option<ToCommitState> {
    let logger = mgr.worker(w_i)?;
    let idx = w_i - self.start_wid;
    self.ready_cut[idx] = logger.precommitted.lock().len();
    self.ready_rfa_cut[idx] = logger.precommitted_rfa.lock().len();
    let st = if logger.variant == CommitVariant::WiloSteal {
      mgr.commit_state_snapshot(w_i)
    } else {
      logger.w_state.clone_blocking()
    };
    self.min_all_workers_gsn = self.min_all_workers_gsn.min(st.last_gsn);
    self.max_all_workers_gsn = self.max_all_workers_gsn.max(st.last_gsn);
    self.min_hardened_commit_ts = self
      .min_hardened_commit_ts
      .min(st.precommitted_tx_commit_ts);
    self.worker_states[idx] = st.clone();
    Some(st)
  }

  // ---- 检查点偏移管理（ZLeanStore Checkpointer 编排）----

  /// 检查点偏移
  #[inline]
  pub fn get_checkpoint_offset(&self) -> u64 {
    self.checkpoint_offset
  }

  /// 推进检查点偏移（账本命名 plain 版：以全局水位门控全部就绪分片）
  pub fn update_checkpoint_offset(&mut self, mgr: &LogManager) {
    let min_flushed = mgr.min_gsn_flushed.load();
    self.update_checkpoint_offset_until_cur_gsn(min_flushed);
  }

  /// 推进检查点到所有 `gsn <= min_flushed` 的目标（上游 UpdateCheckpointOffsetUntilCurGSN）
  pub fn update_checkpoint_offset_until_cur_gsn(&mut self, min_flushed: u64) {
    let ready: Vec<u64> = self
      .cp_target
      .iter()
      .filter(|(_, (_, gsn))| *gsn <= min_flushed)
      .map(|(shard, _)| *shard)
      .collect();
    for shard in ready {
      if let Some((off, gsn)) = self.cp_target.remove(&shard) {
        self.checkpoint_offset = self.checkpoint_offset.max(off);
        if let Some(tx) = &self.checkpoint_tx {
          let _ = tx.try_send((shard, off, gsn));
        }
      }
    }
  }

  /// 推进检查点到指定分片（上游 UpdateCheckpointOffsetUntilShard；复用 GSN 门控实现）
  pub fn update_checkpoint_offset_until_shard(&mut self, upto_shard: u64) {
    let min_flushed = self
      .cp_target
      .range(..=upto_shard)
      .map(|(_, (_, gsn))| *gsn)
      .max()
      .unwrap_or(0);
    self.update_checkpoint_offset_until_cur_gsn(min_flushed);
  }

  /// 更新分片检查点 GSN（上游 UpdateShardCheckpointGSN）
  pub fn update_shard_checkpoint_gsn(&mut self, mgr: &LogManager, shard: u64, gsn: u64) {
    let off = self.w_start_offset + (shard + 1) * self.shard_wal_size;
    self
      .cp_target
      .insert(shard, (off.min(mgr.wal_file_offset()), gsn));
  }

  /// 检查点主循环（上游 RunCheckpointThread；就绪分片经管道通知 L5 消费）
  pub async fn run_checkpoint_thread(&mut self, mgr: &LogManager) -> Result<()> {
    while mgr.is_running.load(Ordering::Relaxed) {
      if self.checkpoint_added && !self.cp_target.is_empty() {
        let min_flushed = mgr.min_gsn_flushed.load();
        self.update_checkpoint_offset_until_cur_gsn(min_flushed);
      }
      yield_now().await;
      sleep(Duration::from_millis(1)).await;
    }
    Ok(())
  }
}
