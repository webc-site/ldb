//! GC 执行器（上游 `ZLeanStore/src/storage/space/garbage_collector.cc`）。
//!
//! 已修复的上游缺陷：
//! - C-14：`RandomGC` 空列表 `% 0` 除零 SIGFPE + `rand()` 非线程安全 → 空表早退 + fastrand；
//! - C-15：`TwoRGreedyGC` 哨兵 bid 越界读 + 重复选择竞态 → 空表守卫 + 单次选择缓存；
//! - C-24：`GreedyAmongImbalancedGroups` 均值/总量混比 + 32 位常量 → 口径统一 u64::MAX；
//! - C-29：`ValidPIDsSizeInBlock` 把条目计数加进字节数 → 纯字节公式；
//! - C-31：`FIFOGC` 空表越界 + 策略不可达 → 空表早退（本实现补全可达 case）；
//! - N-17：`block_lists` 无锁遍历 → 经 ldb_space 快照 API。

use std::{cmp::Reverse, fmt};

use gxhash::HashSet as GxHashSet;
use ldb_page::hook::TxnLogHook;
use ldb_space::{BlockState, NoTxnLog, SpaceManager};
use ldb_type::{BlockId, LogId, MIN_COMP_SIZE, PAGE_SIZE, PageId};

/// GC 状态（victim 集合）。
///
/// `T: TxnLogHook` 为事务日志钩子（L2 AFIT trait；GC 搬移每页一次
/// `log_valid_page`，泛型静态分发）；默认 [`NoTxnLog`]。
pub struct GarbageCollector<T: TxnLogHook = NoTxnLog> {
  /// 已标记为 GC 牺牲的块（防重复选择）
  gc_victims: GxHashSet<BlockId>,
  /// 写分组均衡阈值（组间有效页差 > 该值视为失衡）
  imbalance_threshold: u64,
  /// 事务日志钩子（有效页镜像写 WAL）
  txn_log: T,
}

impl<T: TxnLogHook> fmt::Debug for GarbageCollector<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("GarbageCollector")
      .field("victims", &self.gc_victims.len())
      .field("imbalance_threshold", &self.imbalance_threshold)
      .finish_non_exhaustive()
  }
}

impl<T: TxnLogHook + Default> GarbageCollector<T> {
  /// 构造（默认 [`NoTxnLog`] 钩子）
  pub fn new(imbalance_threshold: u64) -> Self {
    Self::with_txn_log(imbalance_threshold, T::default())
  }
}

impl<T: TxnLogHook> GarbageCollector<T> {
  /// 构造并注入事务日志钩子
  pub fn with_txn_log(imbalance_threshold: u64, hook: T) -> Self {
    Self {
      gc_victims: GxHashSet::default(),
      imbalance_threshold,
      txn_log: hook,
    }
  }

  // ---- 有效页统计 ----

  /// 块内有效 PID 计数（上游 ValidPIDsCntInBlock；委托空间层元数据）
  #[inline]
  pub fn valid_pi_ds_cnt_in_block(&self, mgr: &SpaceManager<T>, bid: BlockId) -> u32 {
    mgr.block_meta(bid).valid_cnt()
  }

  /// 块内有效数据字节（C-29 修复：纯字节公式，不再把条目计数混入）
  #[inline]
  pub fn valid_pi_ds_size_in_block(&self, mgr: &SpaceManager<T>, bid: BlockId) -> u64 {
    u64::from(self.valid_pi_ds_cnt_in_block(mgr, bid)) * PAGE_SIZE as u64
  }

  /// 页有效性（space 侧判定：已映射且未释放）
  #[inline]
  pub fn page_is_valid(&self, mgr: &SpaceManager<T>, pid: PageId) -> bool {
    mgr.page_is_valid(pid)
  }

  /// 页物理偏移
  #[inline]
  pub fn get_page_offset(&self, mgr: &SpaceManager<T>, pid: PageId) -> u64 {
    mgr.get_pid2_offset(pid).get_offset()
  }

  /// 偏移反查 PID
  #[inline]
  pub fn convert_offset2_pid(&self, mgr: &SpaceManager<T>, off: u64) -> Option<PageId> {
    mgr.get_pid_from_offset(off)
  }

  /// 收集块内全部有效 PID（上游 GetValidPIDsInBlock；C-7 static 竞争修复 ——
  /// 返回值替代 static vector）。
  ///
  /// R1 修（P0 死锁）：原 `for slot in 0..mgr.sspace.read().page_cnt_per_block`
  /// 的 range 表达式临时 `sspace` 读守卫被延长至整个循环体，体内
  /// `convert_offset2_pids`（内部取 `sspace.write()`）与之冲突 —— 同线程读未释
  /// 即求写（parking_lot 非可重入）→ 死锁；该函数此前无调用方故长期潜伏。
  /// 先取值、守卫在语句末即释。
  pub fn get_valid_pi_ds_in_block(&self, mgr: &SpaceManager<T>, bid: BlockId) -> Vec<PageId> {
    let page_cnt = mgr.sspace.read().page_cnt_per_block;
    let mut out = Vec::new();
    for slot in 0..page_cnt {
      for pid in mgr.convert_offset2_pids(bid, slot) {
        if self.page_is_valid(mgr, pid) {
          out.push(pid);
        }
      }
    }
    out
  }

  // ---- victim 集合 ----

  /// 标记 GC 牺牲块
  #[inline]
  pub fn add_block_as_gc_victim(&mut self, bid: BlockId) {
    self.gc_victims.insert(bid);
  }

  /// 是否已是牺牲块
  #[inline]
  pub fn block_is_gc_victim(&self, bid: BlockId) -> bool {
    self.gc_victims.contains(&bid)
  }

  /// 失效历史登记（GC 侧委托；上游 AddBlockInvalidationHistory 的
  /// `block_invalidation_cnt++` 落在 space 层元数据）。
  ///
  /// R1 修：原实现额外维护 `invalidation_history: Vec<(bid,gsn)>`，但全仓无任何
  /// 读取方（仅 Debug 打印长度）且随 GC 无限增长 —— 属写死代码，已删（F2）。
  /// 上游 `AddBlockInvalidationHistory`（garbage_collector.cc:735-754）亦只
  /// 递增计数、无历史向量。
  pub fn add_block_invalidation_history(
    &mut self,
    mgr: &SpaceManager<T>,
    bid: BlockId,
    gsn: LogId,
  ) {
    mgr.add_block_invalidation_history(bid, gsn);
  }

  // ---- 策略 1：Greedy（含 SSDWA1 2 参版）----

  /// 贪心扫描内核（对候选集单遍扫描；greedy_gc 与 greedy_gc_n 共用，
  /// 后者复用同一份列表快照，消除每轮 victim 选择的全列表克隆 —— 万级块时
  /// 每轮 8B×N 的 memcpy 分配放大）。
  ///
  /// R1 修（C2 实测 1024 块 greedy=11.6µs）：原实现对每块经 `mgr.block_meta`
  /// 取**两次** `sspace` 读锁（`meta` 与 `valid` 各一次），1024 块即 2048 次
  /// 锁获取；且 `meta.state`/`valid` 来自两次独立读取，并发写者下可撕裂。
  /// 改为整轮持单读锁、就地索引元数据 —— 语义不变，锁获取从 O(N) 降为 O(1)。
  fn greedy_pick(
    &self,
    mgr: &SpaceManager<T>,
    list: &[BlockId],
    want: BlockState,
    exclude: &GxHashSet<BlockId>,
  ) -> BlockId {
    let s = mgr.sspace.read();
    let mut min_valid =
      u64::from(s.page_cnt_per_block) * u64::from(PAGE_SIZE as u32 / MIN_COMP_SIZE as u32);
    let mut min_bid = BlockId::MAX;
    for &bid in list {
      if exclude.contains(&bid) {
        continue;
      }
      let meta = &s.block_metadata[bid as usize];
      if meta.state != want {
        continue;
      }
      let valid = u64::from(meta.valid_cnt());
      if valid == u64::from(meta.total_pid_cnt) {
        continue; // 无失效，不值得回收
      }
      if valid == 0 {
        return bid; // 全死块直接选
      }
      if valid < min_valid {
        min_valid = valid;
        min_bid = bid;
      }
    }
    min_bid
  }

  /// 贪心牺牲块：有效页最少的 FULL/GC_READ 块（上游 GreedyGC(bool)）
  pub fn greedy_gc(&self, mgr: &SpaceManager<T>, read: bool) -> BlockId {
    let want = if read {
      BlockState::Full
    } else {
      BlockState::GcRead
    };
    let list = mgr.sspace.read().block_lists[want.idx()].snapshot();
    self.greedy_pick(mgr, &list, want, &GxHashSet::default())
  }

  /// SSDWA1 NoWA 版：贪心选 victim_cnt 个 + 失衡感知精化（上游 GreedyGC(bool,int)）
  pub fn greedy_gc_n(
    &mut self,
    mgr: &SpaceManager<T>,
    read: bool,
    victim_cnt: usize,
  ) -> Vec<BlockId> {
    let want = if read {
      BlockState::Full
    } else {
      BlockState::GcRead
    };
    // 单次快照全轮复用（原实现每轮 greedy_gc 重新克隆整个列表）
    let list = mgr.sspace.read().block_lists[want.idx()].snapshot();
    let mut victims = Vec::with_capacity(victim_cnt);
    while victims.len() < victim_cnt {
      let min_bid = self.greedy_pick(mgr, &list, want, &self.gc_victims);
      if min_bid == BlockId::MAX {
        break;
      }
      // 注：内核已按 gc_victims 过滤候选（原实现每轮不过滤重扫，最小块被标记后
      // 即触发 break —— victim_cnt > 1 时第二轮必空手而归，只能选出 1 个）
      // 失衡感知：会引发写组失衡则找候选替换
      if self.will_trigger_imbalance(mgr, min_bid) {
        let alt = self.search_for_other_candidate(mgr, read, &victims);
        if alt != BlockId::MAX {
          victims.push(alt);
          self.add_block_as_gc_victim(alt);
          continue;
        }
      }
      victims.push(min_bid);
      self.add_block_as_gc_victim(min_bid);
    }
    victims
  }

  /// 换一个不与已选重叠的候选（上游 SearchForOtherCandidate）
  pub fn search_for_other_candidate(
    &self,
    mgr: &SpaceManager<T>,
    read: bool,
    exclude: &[BlockId],
  ) -> BlockId {
    let st = if read {
      BlockState::Full
    } else {
      BlockState::GcRead
    };
    let list = mgr.sspace.read().block_lists[st.idx()].snapshot();
    list
      .into_iter()
      .find(|&bid| !exclude.contains(&bid) && !self.block_is_gc_victim(bid))
      .unwrap_or(BlockId::MAX)
  }

  // ---- 策略 2：TwoRGreedy ----

  /// 两轮贪心（上游 TwoRGreedyGC；C-15 修复：空表守卫 + 单次选择缓存）
  pub fn two_r_greedy_gc(&mut self, mgr: &SpaceManager<T>, read: bool) -> Vec<BlockId> {
    let mut selected = Vec::new();
    let greedy_pick = self.greedy_gc(mgr, false); // C-15：单次选择并缓存
    if greedy_pick == BlockId::MAX {
      return selected; // C-15：GC_READ 空表早退（上游哨兵越界读路径消除）
    }
    let pick_valid = u64::from(self.valid_pi_ds_cnt_in_block(mgr, greedy_pick));
    let page_cnt = u64::from(mgr.sspace.read().page_cnt_per_block);
    let mincnt = page_cnt / page_cnt.saturating_sub(pick_valid).max(1) + 1;
    let src = {
      let st = if read {
        BlockState::Full
      } else {
        BlockState::GcRead
      };
      mgr.sspace.read().block_lists[st.idx()].snapshot()
    };
    if (src.len() as u64) < mincnt && mgr.get_gc_target_size() != 0 {
      return selected;
    }
    if src.len() == 1 {
      selected.push(src[0]);
      return selected;
    }
    if pick_valid < page_cnt / 8 {
      selected.push(greedy_pick);
      return selected;
    }
    // 按无效字节降序收集
    let mut candidates: Vec<(BlockId, u64)> = src
      .iter()
      .filter(|&&bid| mgr.block_meta(bid).state == BlockState::GcRead)
      .map(|&bid| {
        let valid = self.valid_pi_ds_size_in_block(mgr, bid);
        let capacity = mgr.sspace.read().max_w_ptr;
        (bid, capacity.saturating_sub(valid))
      })
      .collect();
    candidates.sort_by_key(|&(_, c)| Reverse(c));
    selected.extend(candidates.iter().map(|(bid, _)| *bid));
    selected
  }

  // ---- 策略 3：KGreedy ----

  /// K 采样贪心（上游 KGreedyGC：随机采 k 个再取有效最少）。
  ///
  /// R1 修：候选扫描整轮单读锁（原每候选 2 次 `block_meta` 取锁）。
  pub fn k_greedy_gc(&mut self, mgr: &SpaceManager<T>, k: u32) -> BlockId {
    let full = mgr.sspace.read().block_lists[BlockState::Full.idx()].snapshot();
    let candidates: Vec<BlockId> = if full.len() <= k as usize {
      full
    } else {
      // 随机采样 k 个（fastrand；上游 mt19937 替代）
      let mut pool = full;
      for i in (1..pool.len()).rev() {
        pool.swap(i, fastrand::usize(..=i));
      }
      pool.into_iter().take(k as usize).collect()
    };
    let s = mgr.sspace.read();
    let mut min_valid =
      u64::from(s.page_cnt_per_block) * u64::from(PAGE_SIZE as u32 / MIN_COMP_SIZE as u32);
    let mut min_bid = BlockId::MAX;
    for bid in candidates {
      let meta = &s.block_metadata[bid as usize];
      if meta.state != BlockState::Full || self.gc_victims.contains(&bid) {
        continue;
      }
      let valid = u64::from(meta.valid_cnt());
      if valid < min_valid {
        min_valid = valid;
        min_bid = bid;
      }
    }
    min_bid
  }

  // ---- 策略 4：FIFO（C-31 修复：可达 + 空表安全）----

  /// FIFO 牺牲块（上游 FIFOGC 死代码激活：窥视最早入列的 FULL 块）。
  ///
  /// 上游 `FIFOGC`（garbage_collector.cc:695-698）返回 `block_lists[FULL][0]`
  /// 且**不移出列表**；R1 修：原实现 `pop_front()` 只从 Vec 弹出、不更新
  /// `block_metadata[bid].state`，块既不在任何列表而 state 仍为 `Full` —— 列表/
  /// 元数据失联、块丢失（C-30 同类）。改为 `front()` 窥视，出列表由调用方
  /// 经 `move_block_to_list` 完成（上游 AddBlockAsGCVictim 语义）。
  pub fn fifo_gc(&self, mgr: &SpaceManager<T>) -> BlockId {
    let s = mgr.sspace.read();
    let list = &s.block_lists[BlockState::Full.idx()];
    if list.is_empty() {
      return BlockId::MAX; // C-31：空表早退（上游 [0] 越界修复）
    }
    list.front().unwrap_or(BlockId::MAX)
  }

  // ---- 策略 5：Random（C-14 修复）----

  /// 随机牺牲块（C-14 三项修复：空表早退防 % 0 SIGFPE + `fastrand` 线程安全 +
  /// **状态过滤**）。
  ///
  /// 上游 `RandomGC`（garbage_collector.cc:700-704）是 6 个选择器里唯一不校验
  /// `block_metadata[bid].state` 的（对比 `GreedyGC:375` / `EDTBasedGC:716` 都查）；
  /// C-30「列表/元数据失联」时会把非 FULL 块当牺牲块下发，下游
  /// `MoveBlockToList(GC_READ, GC_WRITE)` 必失败（C-39）。此处采样后在同一读锁内
  /// O(1) 校验 state；仅当命中失联块（异常态）才回退线性取首个 Full 块 —— 常见
  /// 路径零额外遍历（`random_gc` 仍是 1 alloc/1 读锁）。
  pub fn random_gc(&self, mgr: &SpaceManager<T>) -> BlockId {
    let s = mgr.sspace.read();
    let list = s.block_lists[BlockState::Full.idx()].snapshot();
    if list.is_empty() {
      return BlockId::MAX; // C-14：上游 `rand() % 0` SIGFPE 修复
    }
    let bid = list[fastrand::usize(..list.len())];
    if s.block_metadata[bid as usize].state == BlockState::Full {
      return bid;
    }
    // 失联兜底：随机命中非 FULL 块时，替代为任一 state=Full 的块
    list
      .iter()
      .copied()
      .find(|&b| s.block_metadata[b as usize].state == BlockState::Full)
      .unwrap_or(BlockId::MAX)
  }

  // ---- 策略 6：EDTBased ----

  /// EDT 信任判定（上游 ShouldTrustEDT：flush 水位前的 EDT 可信）
  pub fn should_trust_edt(&self, mgr: &SpaceManager<T>, bid: BlockId, flushed_gsn: LogId) -> bool {
    let m = mgr.block_meta(bid);
    m.edt != 0 && m.edt <= flushed_gsn
  }

  /// 基于 EDT 的牺牲块（上游 EDTBasedGC：EDT 最小优先，同 EDT 取有效最少）。
  ///
  /// R1 修：整轮单读锁（原每块 3 次锁：`block_meta` + `valid_cnt` + `should_trust_edt`）。
  /// flush 信任过滤内联自 [`Self::should_trust_edt`]（同一不变量，避免二次取锁）。
  ///
  /// R3 修（P1）：状态谓词收敛为 `== st`（上游 `EDTBasedGC` 的 `expected_state =
  /// read ? FULL : GC_READ` 严格过滤，garbage_collector.cc:714-717）。原实现接受
  /// `Full | GcRead` 两种，在 C-30/C-39「列表/元数据失联」（FULL 列表含 state=GcRead
  /// 块）时会把**已进入 GC 的块重复选为 victim**，与其余选择器（greedy/k_greedy/random
  /// 均严格 `== want`）及上游口径不一致。列表源与 `read` 一致 ⇒ 正常不变式下无行为变化
  /// （1e5 差分 MISMATCH=0 不变）。
  pub fn edt_based_gc(&self, mgr: &SpaceManager<T>, read: bool, flushed_gsn: LogId) -> BlockId {
    let st = if read {
      BlockState::Full
    } else {
      BlockState::GcRead
    };
    let list = mgr.sspace.read().block_lists[st.idx()].snapshot();
    let s = mgr.sspace.read();
    let mut victim = BlockId::MAX;
    let mut min_edt = LogId::MAX;
    let mut min_valid = u32::MAX;
    for bid in list {
      let meta = &s.block_metadata[bid as usize];
      if meta.state != st {
        continue;
      }
      if self.gc_victims.contains(&bid) {
        continue;
      }
      let valid = meta.valid_cnt();
      let trusted = meta.edt != 0 && meta.edt <= flushed_gsn;
      let edt = if trusted { meta.edt } else { LogId::MAX };
      if edt < min_edt || (edt == min_edt && valid < min_valid) {
        min_edt = edt;
        min_valid = valid;
        victim = bid;
      }
    }
    victim
  }

  // ---- NoWA 写组均衡 ----

  /// 组平均有效页（上游 AvgGroupValidPages）
  pub fn avg_group_valid_pages(&self, mgr: &SpaceManager<T>, group: &[BlockId]) -> u64 {
    if group.is_empty() {
      return 0;
    }
    let total: u64 = group
      .iter()
      .map(|&bid| u64::from(self.valid_pi_ds_cnt_in_block(mgr, bid)))
      .sum();
    total / group.len() as u64
  }

  /// 是否会触发写组失衡（上游 WillTriggerImbalance：以活跃块按 bid 分组近似
  /// 写组成员；回收 bid 后其组平均有效页偏离全体均值超阈值则判失衡）
  pub fn will_trigger_imbalance(&self, mgr: &SpaceManager<T>, bid: BlockId) -> bool {
    let group_cnt = mgr.write_group_cnt().max(1) as usize;
    let s = mgr.sspace.read();
    let mut group_valid = vec![0u64; group_cnt];
    let mut group_blocks = vec![0u64; group_cnt];
    for (i, meta) in s.block_metadata.iter().enumerate() {
      if meta.state == BlockState::Empty {
        continue;
      }
      let g = i % group_cnt;
      group_valid[g] += u64::from(meta.valid_cnt());
      group_blocks[g] += 1;
    }
    drop(s);
    let g = bid as usize % group_cnt;
    if group_blocks[g] == 0 {
      return false;
    }
    let this_after = (group_valid[g] - u64::from(self.valid_pi_ds_cnt_in_block(mgr, bid))).max(1)
      / group_blocks[g];
    let total_avg = group_valid.iter().sum::<u64>() / group_blocks.iter().sum::<u64>().max(1);
    this_after.abs_diff(total_avg) > self.imbalance_threshold
  }

  /// 是否应选失衡块（上游 ShouldSelectImbalancedBlock）
  pub fn should_select_imbalanced_block(&self, mgr: &SpaceManager<T>, bid: BlockId) -> bool {
    !self.will_trigger_imbalance(mgr, bid)
  }

  /// 失衡组间贪心（C-24 修复：总量口径一致 + u64::MAX）
  pub fn greedy_among_imbalanced_groups(
    &self,
    mgr: &SpaceManager<T>,
    groups: &[Vec<BlockId>],
  ) -> BlockId {
    let mut lowest_group_idx = usize::MAX;
    let mut lowest_valid = u64::MAX; // C-24：上游 UINT_MAX 32 位常量修复
    for (idx, group) in groups.iter().enumerate() {
      let total: u64 = group
        .iter()
        .map(|&bid| u64::from(self.valid_pi_ds_cnt_in_block(mgr, bid)))
        .sum();
      // C-24：比较与存储统一为总量口径
      if total < lowest_valid {
        lowest_valid = total;
        lowest_group_idx = idx;
      }
    }
    groups
      .get(lowest_group_idx)
      .and_then(|g| g.first().copied())
      .unwrap_or(BlockId::MAX)
  }

  // ---- 调度与列表迁移 ----

  /// 读源选择（策略调度入口；上游 SelectVictimBlockToGCRead）
  pub fn select_victim_block_to_gc_read(
    &mut self,
    mgr: &SpaceManager<T>,
    strategy: GcStrategy,
    flushed_gsn: LogId,
  ) -> BlockId {
    let bid = match strategy {
      GcStrategy::Greedy => self.greedy_gc(mgr, true),
      GcStrategy::TwoRGreedy => self
        .two_r_greedy_gc(mgr, true)
        .first()
        .copied()
        .unwrap_or(BlockId::MAX),
      GcStrategy::EdtBased => self.edt_based_gc(mgr, true, flushed_gsn),
      GcStrategy::KGreedy(k) => self.k_greedy_gc(mgr, k),
      GcStrategy::Fifo => self.fifo_gc(mgr),
      GcStrategy::Random => self.random_gc(mgr),
    };
    if bid != BlockId::MAX {
      self.add_block_as_gc_victim(bid);
    }
    bid
  }

  /// 写目标组选择（上游 SelectVictimBlocksToGCWrite：SSDWA1 路径）
  pub fn select_victim_blocks_to_gc_write(
    &mut self,
    mgr: &SpaceManager<T>,
    victim_cnt: usize,
  ) -> Vec<BlockId> {
    self.greedy_gc_n(mgr, false, victim_cnt)
  }

  /// FULL → GC_READ 迁移（上游 MoveBlockFromFullToGCBlockList）
  pub fn move_block_from_full_to_gc_block_list(&self, mgr: &SpaceManager<T>, bid: BlockId) -> bool {
    mgr.move_block_to_list(bid, BlockState::Full, BlockState::GcRead)
  }

  /// GC 完成块 → 空闲（上游 MoveBlocksFromGCToEmptyBlockList）。
  /// 迁移由 reset_block_space 原子完成（内部“先迁移后清零”）；原实现外层的
  /// 二次 move 以重置后的 state 再迁移必然失败，计数恒为 0
  pub fn move_blocks_from_gc_to_empty_block_list(
    &mut self,
    mgr: &SpaceManager<T>,
    bids: &[BlockId],
  ) -> usize {
    let mut n = 0;
    for &bid in bids {
      if mgr.reset_block_space(bid) {
        n += 1;
      }
      self.gc_victims.remove(&bid);
    }
    n
  }

  /// GC 完成块 → 打开（上游 MoveBlocksFromGCToOpenBlockList）
  pub fn move_blocks_from_gc_to_open_block_list(
    &mut self,
    mgr: &SpaceManager<T>,
    bids: &[BlockId],
  ) -> usize {
    let mut n = 0;
    for &bid in bids {
      let st = mgr.block_meta(bid).state;
      if mgr.move_block_to_list(bid, st, BlockState::Open) {
        n += 1;
      }
      self.gc_victims.remove(&bid);
    }
    n
  }

  /// 反向表清空（委托）
  #[inline]
  pub fn reset_offset2_pids(&self, mgr: &SpaceManager<T>, bid: BlockId) {
    mgr.reset_offset2_pids(bid);
  }

  /// GC 写块擦除（GC_WRITE 态；上游 EraseBlock 去 TrimBlock —— 打孔回收由 L5 执行。
  /// async：真实落盘路径将 await compio 打孔，保持签名一致）
  pub async fn erase_block(&self, mgr: &SpaceManager<T>, bid: BlockId) -> bool {
    mgr.erase_block(bid)
  }

  /// 有效页镜像登记（GC 搬移前写 WAL；经注入的 TxnLogHook）
  pub async fn log_valid_page(&self, pid: PageId) {
    self.txn_log.log_valid_page(pid).await;
  }

  /// 注入事务日志钩子（替换默认 [`NoTxnLog`]）
  pub fn set_txn_log(&mut self, hook: T) {
    self.txn_log = hook;
  }
}

/// GC 策略选择（上游 victim-selection switch 的枚举化）
#[derive(Debug, Clone, Copy)]
pub enum GcStrategy {
  /// 贪心
  Greedy,
  /// 两轮贪心
  TwoRGreedy,
  /// EDT 优先
  EdtBased,
  /// K 采样贪心
  KGreedy(u32),
  /// FIFO（上游不可达策略的补全实现）
  Fifo,
  /// 随机
  Random,
}
