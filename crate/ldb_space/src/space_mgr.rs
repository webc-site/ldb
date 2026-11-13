//! SpaceManager 主体（上游 `space_manager.cc` 的跨平台收敛版）。
//!
//! 已修复的上游缺陷：
//! - C-1：`should_trigger_gc` 硬编码 `ShouldTriggerGCZNS` 短路 → 重写为通用容量
//!   + WAF 最优判定；同步移除 `select_block_id_to_write` 的 `!trigger_gc` 空块禁用门控（C-32）；
//! - C-12：`reached_alloc_size` 重复条件 → 单次判定；
//! - C-35：PID2Offset 位域溢出 → `ldb_pack::Pid2Offset`（49 位）；
//! - C-30：block_lists 无锁遍历 → 快照；
//! - `update_pid2_offset` 的 3 行 ZNS 守卫删除（audit 1.1 恢复立项）；
//! - 异地写编排去 IOBackend 直调（compio 由 L1 后端承接，经 `ldb_buf` 注入）。

use std::{
  fmt,
  future::Future,
  io,
  mem::{size_of, take},
  pin::Pin,
  rc::Rc,
  sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
  },
};

use ldb_pack::Pid2Offset;
use ldb_page::hook::{GcHook, TxnLogHook};
use ldb_type::{BlockId, LogId, PAGE_SIZE, PageId};
use parking_lot::RwLock;

use crate::{
  edt::calc_block_avg_edt_of,
  error::{Error, Result},
  page_io::PageIo,
  state::{BlockMetadata, BlockState, Offset2Pids, StorageSpace},
};

/// GC 触发的水位阈值（80% 逻辑用量）
const GC_WATERMARK_PCT: u64 = 80;
/// GC 触发的 WAF 上界
const GC_WAF_LIMIT: f64 = 1.5;
/// 可复用 PID 容量（上游 TRIM_BUFFER_SIZE/MAX_REUSABLE_PIDS）
pub const MAX_REUSABLE_PIDS: usize = 64 * 1024;

/// 空间管理器。
///
/// `T: TxnLogHook` 为事务日志钩子（L2 AFIT trait，L4 实现注入）：
/// 热路径 `update_pid2_offset` 每页一次 hook 调用，泛型静态分发禁 `dyn`；
/// 默认 [`NoTxnLog`] 无操作（测试 / 无 WAL 场景）。
pub struct SpaceManager<T: TxnLogHook = NoTxnLog> {
  /// 存储空间主体
  pub sspace: RwLock<StorageSpace>,
  /// PID→偏移映射表
  pid2offset: RwLock<Vec<Pid2Offset>>,
  /// 事务日志钩子（log_pid2_offset / log_valid_page / log_free_page）
  pub txn_log: T,
  /// GC 钩子（冷路径：块耗尽 / GC 触发点回调，`dyn` 可接受；
  /// `OnceLock` 允许 Arc 化后补注入 —— L5 装配顺序 space→buf→gc）
  pub gc_hook: OnceLock<Rc<dyn GcHook>>,
  /// 待释放页列表
  dealloc_list: RwLock<Vec<PageId>>,
  /// 可复用 PID 池
  reusable_pids: RwLock<Vec<PageId>>,
  /// 块大小配置（字节）
  block_size: u64,
  /// 上一次 GC 触发时的已分配量
  last_gc_alloc: AtomicU64,
  /// 前次 GC 块（AvoidPad 语义）
  prev_gced_block: AtomicU64,
  /// 写分组数
  write_group_cnt: u32,
}

impl<T: TxnLogHook> fmt::Debug for SpaceManager<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SpaceManager")
      .field("block_size", &self.block_size)
      .field("write_group_cnt", &self.write_group_cnt)
      .finish_non_exhaustive()
  }
}

/// 无操作事务日志钩子（测试 / 无 WAL 场景的默认注入）
#[derive(Debug, Default, Clone, Copy)]
pub struct NoTxnLog;

impl TxnLogHook for NoTxnLog {
  #[inline]
  async fn log_pid2_offset(&self, _pid: PageId) {}
  #[inline]
  async fn log_valid_page(&self, _pid: PageId) {}
  #[inline]
  async fn log_free_page(&self, _pid: PageId) {}
}

impl<T: TxnLogHook> SpaceManager<T> {
  /// 构造（默认 [`NoTxnLog`] 钩子，要求 `T: Default`）
  pub fn new(
    block_cnt: u64,
    page_cnt_per_block: u32,
    min_comp_size: u32,
    max_open_block_cnt: u32,
    _max_mapped_pages: u64,
    write_group_cnt: u32,
  ) -> Self
  where
    T: Default,
  {
    Self::with_txn_log(
      block_cnt,
      page_cnt_per_block,
      min_comp_size,
      max_open_block_cnt,
      _max_mapped_pages,
      write_group_cnt,
      T::default(),
    )
  }

  /// 构造并注入事务日志钩子（L4 `WalTxnLogHook` 等非 `Default` 类型走此入口；
  /// 上游 Construction 语义：独立文件布局 mapping_offset = 0，C-6 修复）
  pub fn with_txn_log(
    block_cnt: u64,
    page_cnt_per_block: u32,
    min_comp_size: u32,
    max_open_block_cnt: u32,
    _max_mapped_pages: u64,
    write_group_cnt: u32,
    hook: T,
  ) -> Self {
    let block_size = u64::from(page_cnt_per_block) * PAGE_SIZE as u64;
    let sspace = StorageSpace::new(
      block_cnt,
      page_cnt_per_block,
      min_comp_size,
      block_size,
      block_size,
      max_open_block_cnt,
    );
    Self {
      sspace: RwLock::new(sspace),
      pid2offset: RwLock::new(Vec::new()),
      txn_log: hook,
      gc_hook: OnceLock::new(),
      dealloc_list: RwLock::new(Vec::new()),
      reusable_pids: RwLock::new(Vec::with_capacity(1024)),
      block_size,
      last_gc_alloc: AtomicU64::new(0),
      prev_gced_block: AtomicU64::new(BlockId::MAX),
      write_group_cnt,
    }
  }

  /// 注入事务日志钩子（替换默认 [`NoTxnLog`]）
  pub fn set_txn_log(&mut self, hook: T) {
    self.txn_log = hook;
  }

  /// 注入 GC 钩子（冷路径；`&self` —— 共享化后仍可注入，重复注入忽略）
  pub fn set_gc_hook(&self, hook: Rc<dyn GcHook>) {
    let _ = self.gc_hook.set(hook);
  }

  /// 块容量（字节）
  #[inline]
  pub fn get_block_size(&self) -> u64 {
    self.block_size
  }

  // ---- 查询 ----

  /// 块元数据读
  #[inline]
  pub fn block_meta(&self, bid: BlockId) -> BlockMetadata {
    self.sspace.read().block_metadata[bid as usize]
  }

  /// 状态列表长度
  #[inline]
  pub fn get_block_list_size(&self, st: BlockState) -> usize {
    self.sspace.read().block_list_size(st)
  }

  /// 块是否在列表
  #[inline]
  pub fn is_block_in_list(&self, bid: BlockId, st: BlockState) -> bool {
    self.sspace.read().block_in_list(bid, st)
  }

  /// 块内已写字节
  #[inline]
  pub fn get_block_write_offset(&self, bid: BlockId) -> u64 {
    self.block_meta(bid).cur_size
  }

  /// 块起始文件偏移（`bid * block_size`；独立文件布局，无上游 64GiB 硬编码）
  #[inline]
  pub fn get_block_offset(&self, bid: BlockId) -> u64 {
    bid * self.block_size
  }

  /// 活跃块表大小
  #[inline]
  pub fn get_active_block_size(&self) -> u64 {
    self.sspace.read().active_block_list.len() as u64
  }

  /// DB 逻辑尺寸（全部非空块已写字节和；上游 GetDBSize 修复版 —— zns 注释死代码不移植）
  pub fn get_db_size(&self) -> u64 {
    let s = self.sspace.read();
    s.block_metadata
      .iter()
      .filter(|m| m.state != BlockState::Empty)
      .map(|m| m.cur_size)
      .sum()
  }

  /// GC 目标规模（上游死代码覆盖 bug（N-10）修复：直接统计）
  pub fn get_gc_target_size(&self) -> u64 {
    let s = self.sspace.read();
    s.block_lists[BlockState::Full.idx()].len() as u64 * self.block_size
  }

  /// PID→偏移表长度
  #[inline]
  pub fn get_max_pid(&self) -> u64 {
    self.pid2offset.read().len() as u64
  }

  /// 读映射项
  #[inline]
  pub fn get_pid2_offset(&self, pid: PageId) -> Pid2Offset {
    self
      .pid2offset
      .read()
      .get(pid as usize)
      .copied()
      .unwrap_or_default()
  }

  /// 读压缩尺寸
  #[inline]
  pub fn get_pid_comp_size(&self, pid: PageId) -> u16 {
    self.get_pid2_offset(pid).get_comp_sz()
  }

  /// PID 所属块（按物理偏移除）
  #[inline]
  pub fn pid_block(&self, pid: PageId) -> BlockId {
    self.get_pid2_offset(pid).get_offset() / self.block_size
  }

  /// 偏移 → PID 槽位（块内偏移 / 页大小）
  #[inline]
  pub fn pid_offset_block(&self, off: u64) -> (BlockId, u64) {
    (
      off / self.block_size,
      off % self.block_size / PAGE_SIZE as u64,
    )
  }

  /// 偏移是否落在块内写指针内
  #[inline]
  pub fn pid_offset_is_in_block(&self, bid: BlockId, off: u64) -> bool {
    let (b, slot) = self.pid_offset_block(off);
    b == bid && slot < u64::from(self.sspace.read().page_cnt_per_block)
  }

  /// 偏移是否有效（已映射且未失效）
  pub fn pid_offset_is_valid(&self, pid: PageId) -> bool {
    let e = self.get_pid2_offset(pid);
    if e.get_comp_sz() == 0 {
      return false;
    }
    let (bid, _slot) = self.pid_offset_block(e.get_offset());
    let s = self.sspace.read();
    (bid as usize) < s.block_metadata.len()
      && s.block_metadata[bid as usize].state != BlockState::Empty
  }

  /// PID 是否在待释放列表
  #[inline]
  pub fn pid_is_deallocated(&self, pid: PageId) -> bool {
    self.dealloc_list.read().contains(&pid)
  }

  /// 有效页判定（GC 视角；上游 PageIsValid 的 space 侧 —— 修 C-25 同名混比的 space 侧语义）
  pub fn page_is_valid(&self, pid: PageId) -> bool {
    let e = self.get_pid2_offset(pid);
    if e.get_comp_sz() == 0 {
      return false;
    }
    !self.pid_is_deallocated(pid)
  }

  // ---- 映射表编排 ----

  /// 更新 PID→偏移（唯一实现；删除上游 3 行 ZNS 守卫，audit 1.1 恢复）。
  ///
  /// 覆写失效（上游 UpdatePID2Offset space_manager.cc:1890-1913）：异地写覆盖旧
  /// 副本后，旧块的有效页计数必须递减（`prev_sz > 0 && prev_offset > 0` 守卫；
  /// 上游 `comp_sz == PAGE_SIZE - 1` 分支属旧哨兵时代死逻辑，C-19 哨兵已改
  /// `PAGE_SIZE`，不移植）。
  /// **分歧登记**：上游 GC 写路径（`binpacked_gc_` → `sync_gc=true`）跳过失效计数
  /// （:1905 `!sync_gc` 门控）—— 源块随后被擦除，计数是临时值；本实现恒计数，
  /// 源块 `reset_block_metadata` 时清零，净效果一致。
  /// WAL 钩子在映射落表后调用 —— L4 钩子按 pid 自取映射序列化后镜像
  /// （upstream 先 log 后 set 捕获前镜像，语义归 ldb_wal 轮裁定）。
  /// 写守卫作用域块收敛（clippy await_holding_lock 要求 await 点零守卫存活）
  pub async fn update_pid2_offset(&self, pid: PageId, off: u64, comp_sz: u16) {
    let prev = {
      let mut table = self.pid2offset.write();
      if pid as usize >= table.len() {
        table.resize(pid as usize + 1, Pid2Offset::default());
      }
      let prev = table[pid as usize];
      table[pid as usize].set(off, comp_sz);
      prev
    };
    if prev.get_comp_sz() > 0 && prev.get_offset() > 0 {
      self.update_block_invalid_count(prev.get_offset(), pid);
    }
    self.txn_log.log_pid2_offset(pid).await;
  }

  /// 反向表更新（上游 UpdateOffset2PIDs：sspace_->blocks 直写归 L3）
  pub fn update_offset2_pids(&self, bid: BlockId, slot: u32, pid: PageId) {
    let mut s = self.sspace.write();
    let slot_ent = &mut s.blocks[bid as usize][slot as usize];
    slot_ent.push(pid);
  }

  /// 反向表读取（上游 ConvertOffset2PIDs）
  pub fn convert_offset2_pids(&self, bid: BlockId, slot: u32) -> Vec<PageId> {
    let s = self.sspace.write();
    s.blocks
      .get(bid as usize)
      .and_then(|b| b.get(slot as usize))
      .map(|e| e.valid().to_vec())
      .unwrap_or_default()
  }

  /// 反向表清空（上游 ResetOffset2PIDs）
  pub fn reset_offset2_pids(&self, bid: BlockId) {
    let mut s = self.sspace.write();
    if let Some(b) = s.blocks.get_mut(bid as usize) {
      for ent in b {
        *ent = Offset2Pids::default();
      }
    }
  }

  /// 映射表扩张（分配 PID 时）
  pub fn assign_pid_upon_alloc(&self, want: u64) -> u64 {
    let mut table = self.pid2offset.write();
    let pid = table.len() as u64;
    if want > pid {
      table.resize(want as usize, Pid2Offset::default());
    }
    pid
  }

  // ---- 块选择 ----

  /// 是否应打开新块（上游 ShouldOpenNewBlockID space_manager.cc:657-690）：
  /// 活跃表空 → 必开；活跃表满额 → 必不开；否则 `avg_edt` 落入任一未满活跃块
  /// 的 `[block_min_edt, block_max_edt]` → 不开（EDT 已被覆盖，聚合优先）。
  pub fn should_open_new_block_id(&self, avg_edt: LogId) -> bool {
    let s = self.sspace.read();
    let cnt = s.active_block_cnt.load(Ordering::Relaxed);
    if cnt == 0 {
      return true;
    }
    if cnt >= s.max_open_block {
      return false;
    }
    let scan = (s.max_open_block as usize).min(s.active_block_list.len());
    !s.active_block_list[..scan].iter().any(|&b| {
      b != BlockId::MAX && {
        let m = &s.block_metadata[b as usize];
        m.cur_size < s.max_w_ptr && avg_edt >= m.block_min_edt && avg_edt <= m.block_max_edt
      }
    })
  }

  /// 选择最相近 EDT 的未满打开块（上游 SelectUnfilledOpenBlockID）
  pub fn select_unfilled_open_block_id(&self, avg_edt: LogId) -> BlockId {
    let s = self.sspace.read();
    let mut closest = BlockId::MAX;
    let mut min_diff = LogId::MAX;
    for &active in s.active_block_list.iter().take(s.max_open_block as usize) {
      if active == BlockId::MAX {
        continue;
      }
      let meta = &s.block_metadata[active as usize];
      if meta.state != BlockState::Open || meta.cur_size >= s.max_w_ptr {
        continue;
      }
      if meta.edt == 0 {
        return active;
      }
      let diff = meta.edt.abs_diff(avg_edt);
      if diff < min_diff {
        min_diff = diff;
        closest = active;
      }
    }
    drop(s);
    if closest != BlockId::MAX
      && self
        .sspace
        .write()
        .move_block_to_list(closest, BlockState::Open, BlockState::InUse)
    {
      return closest;
    }
    BlockId::MAX
  }

  /// 活跃表登记（上游 AddBlockIDToActiveBlockList）
  pub fn add_block_id_to_active_block_list(&self, bid: BlockId) -> bool {
    self.sspace.write().add_to_active_block_list(bid)
  }

  /// 从活跃表移除
  pub fn remove_block_id_from_active_block_list(&self, bid: BlockId) -> bool {
    let mut s = self.sspace.write();
    for slot in s.active_block_list.iter_mut() {
      if *slot == bid {
        *slot = BlockId::MAX;
        s.active_block_cnt.fetch_sub(1, Ordering::Relaxed);
        return true;
      }
    }
    false
  }

  /// 活跃表中首个 GC_IN_USE 块
  pub fn find_first_gc_in_use_block_in_active_list(&self) -> BlockId {
    let s = self.sspace.read();
    s.active_block_list
      .iter()
      .copied()
      .find(|&b| b != BlockId::MAX && s.block_metadata[b as usize].state == BlockState::GcInUse)
      .unwrap_or(BlockId::MAX)
  }

  /// 从 EMPTY 列表取一个空块（上游 SelectEmptyBlockID space_manager.cc:977-1017）。
  ///
  /// ① EDT 早退（:984-997）：某活跃块 `[block_min_edt, block_max_edt)` 覆盖
  ///   `avg_edt` 且装得下本写 → 不新开块（返回哨兵）；
  /// ② 选择策略（:999-1001）：失效历史优先（`SelectEmptyBlockWithMinInvalidationCnt`
  ///   —— 上游 `FLAGS_invalidation_history_based_empty_block_selection` 分支），退化为
  ///   队首块（`SelectEmptyBlockRandom` :1049-1060 的实际生效体，随机体被上游注释）；
  /// ③ 提升：登记活跃表（要求 EMPTY 态，:1010）→ EMPTY 列表迁出 → OPEN + 块 EDT。
  ///
  /// 分歧登记：上游 ③ 失败（活跃表满）返回哨兵并在调用方进入 sleep+goto 重试
  /// （:1010-1014）；A7 修复下 compio 单线程无人推进重试，改为照常分配、仅跳过
  /// 活跃表登记（自由块存在时报 SpaceExhausted 属假耗尽）。
  pub fn select_empty_block_id(&self, write_sz: u64, avg_edt: LogId) -> BlockId {
    {
      let s = self.sspace.read();
      let scan = (s.max_open_block as usize).min(s.active_block_list.len());
      let covered = s.active_block_list[..scan].iter().any(|&b| {
        b != BlockId::MAX && {
          let m = &s.block_metadata[b as usize];
          m.block_min_edt <= avg_edt
            && m.block_max_edt > avg_edt
            && m.cur_size + write_sz <= s.max_w_ptr
        }
      });
      if covered {
        // 上游 :996-998：avg_edt 已被活跃块覆盖 → 不新开块（返回哨兵）
        return BlockId::MAX;
      }
    }
    let bid = match self.select_empty_block_with_min_invalidation_cnt(write_sz) {
      BlockId::MAX => self.select_empty_block_random(write_sz),
      b => b,
    };
    if bid == BlockId::MAX {
      return BlockId::MAX;
    }
    self.promote_empty_block(bid, avg_edt)
  }

  /// 提升 EMPTY 块：活跃表登记 → 出 EMPTY 列表 → OPEN + 块 EDT
  fn promote_empty_block(&self, bid: BlockId, avg_edt: LogId) -> BlockId {
    let mut s = self.sspace.write();
    // 上游顺序：先登记活跃表（要求 EMPTY|GC_IN_USE 态）；活跃表满（max_open 限制
    // 并发打开数）时块仍可分配，只是不进打开块扫描集（分歧见 select_empty_block_id）
    let _ = s.add_to_active_block_list(bid);
    if !s.move_block_to_list(bid, BlockState::Empty, BlockState::Open) {
      return BlockId::MAX;
    }
    s.block_metadata[bid as usize].edt = avg_edt;
    bid
  }

  /// 队首空块（上游 SelectEmptyBlockRandom space_manager.cc:1046-1060：
  /// 实际生效体取 EMPTY 队首，`uniform_int_distribution` 随机体被注释）。
  /// 只窥视不移出 —— 出列表由 `promote_empty_block` 的 EMPTY→OPEN 迁移完成。
  pub fn select_empty_block_random(&self, _write_sz: u64) -> BlockId {
    self.sspace.read().block_lists[BlockState::Empty.idx()]
      .front()
      .unwrap_or(BlockId::MAX)
  }

  /// 选失效数最少的空块（上游 SelectEmptyBlockWithMinInvalidationCnt
  /// space_manager.cc:1019-1043：命中 0 失效即早退）。只窥视不移出。
  pub fn select_empty_block_with_min_invalidation_cnt(&self, _write_sz: u64) -> BlockId {
    let s = self.sspace.read();
    let empty = s.block_lists[BlockState::Empty.idx()].snapshot();
    if empty.is_empty() {
      return BlockId::MAX;
    }
    let mut min_bid = empty[0];
    let mut min_cnt = s.block_metadata[min_bid as usize].block_invalidation_cnt;
    if min_cnt == 0 {
      return min_bid;
    }
    for &bid in empty.iter().skip(1) {
      let cnt = s.block_metadata[bid as usize].block_invalidation_cnt;
      if cnt == 0 {
        return bid;
      }
      if cnt < min_cnt {
        min_bid = bid;
        min_cnt = cnt;
      }
    }
    min_bid
  }

  /// 打开块选择（写尺寸 + EDT 聚合，上游 SelectOpenBlockID :701-745 经
  /// SelectOpenBlockIDWithClosestEDT 的活跃表等价实现）。
  ///
  /// 写热路径：只遍历固定 `max_open_block` 槽的活跃表（无列表快照、无克隆），
  /// 纯只读走读锁。未命中 EDT 最近块时回退随机选择（上游 Case 3/4：
  /// `FLAGS_use_edt` 关闭或单活跃块时走 `SelectOpenBlockIDRandom`）。
  pub fn select_open_block_id(
    &self,
    write_sz: u64,
    avg_edt: LogId,
    _min_edt: LogId,
    _max_edt: LogId,
  ) -> BlockId {
    let s = self.sspace.read();
    let mut closest = BlockId::MAX;
    let mut min_diff = LogId::MAX;
    for &active in s.active_block_list.iter().take(s.max_open_block as usize) {
      if active == BlockId::MAX {
        continue;
      }
      let meta = &s.block_metadata[active as usize];
      if meta.state != BlockState::Open && meta.state != BlockState::InUse {
        continue;
      }
      if meta.cur_size + write_sz > s.max_w_ptr {
        continue;
      }
      if meta.edt == 0 {
        return active;
      }
      let diff = meta.edt.abs_diff(avg_edt);
      if diff < min_diff {
        min_diff = diff;
        closest = active;
      }
    }
    if closest != BlockId::MAX {
      return closest;
    }
    drop(s);
    self.select_open_block_id_random(write_sz)
  }

  /// 随机打开块（上游 SelectOpenBlockIDRandom space_manager.cc:876-910：
  /// OPEN 列表快照内取容量足够的随机候选；上游扫描中对超容块就地
  /// OPEN→FULL_PAD+padding，本实现保持只读快照 —— padding 由
  /// `close_one_open_block`/`issue_padding_writes_to_full_block` 统一执行）。
  pub fn select_open_block_id_random(&self, write_sz: u64) -> BlockId {
    let s = self.sspace.read();
    let candidates: Vec<BlockId> = s.block_lists[BlockState::Open.idx()]
      .snapshot()
      .into_iter()
      .filter(|&bid| {
        let m = &s.block_metadata[bid as usize];
        m.state == BlockState::Open && m.cur_size + write_sz <= s.max_w_ptr
      })
      .collect();
    if candidates.is_empty() {
      return BlockId::MAX;
    }
    candidates[fastrand::usize(..candidates.len())]
  }

  /// 未满打开块直选（write group 路径）
  pub fn select_read_g_ced_blocks_to_gc_write(&self) -> Vec<BlockId> {
    self.sspace.read().block_lists[BlockState::GcRead.idx()].snapshot()
  }

  /// 选满块做 GC 读源（上游 SelectFullBlockToGCRead space_manager.cc:1192-1252 的
  /// 单牺牲块版）；选中即记录为「前次 GC 块」（上游 prev_gced_bids_ 语义：
  /// padding/重选时避让同块，io_interface.cc:1392-1398）
  pub fn select_full_block_to_gc_read(&self) -> BlockId {
    let mut s = self.sspace.write();
    let Some(bid) = s.block_lists[BlockState::Full.idx()].pop() else {
      return BlockId::MAX;
    };
    // pop 已出 FULL 列表：直推 GC_READ（与 select_empty 同一直推语义）
    s.block_lists[BlockState::GcRead.idx()].push(bid);
    s.block_metadata[bid as usize].state = BlockState::GcRead;
    drop(s);
    self.set_prev_g_ced_block(bid);
    bid
  }

  /// 写分组选择（NoWA 均衡组；上游 SelectOpenBlockIDWithClosestEDT 的分组版）
  pub fn select_open_block_id_with_closest_edt(&self, avg_edt: LogId, group: usize) -> BlockId {
    let s = self.sspace.write();
    let mut closest = BlockId::MAX;
    let mut min_diff = LogId::MAX;
    for &active in s.active_block_list.iter().take(s.max_open_block as usize) {
      if active == BlockId::MAX {
        continue;
      }
      // 分组亲和：块 % 组数 == group
      if self.write_group_cnt > 1 && active as usize % self.write_group_cnt as usize != group {
        continue;
      }
      let meta = &s.block_metadata[active as usize];
      if meta.state != BlockState::Open || meta.cur_size >= s.max_w_ptr {
        continue;
      }
      let diff = meta.edt.abs_diff(avg_edt);
      if diff < min_diff {
        min_diff = diff;
        closest = active;
      }
    }
    closest
  }

  /// 写满关闭（上游 CloseOneOpenBlock space_manager.cc:1273-1305：OPEN→FULL_PAD
  /// 后经 IssuePaddingWritesToFullBlock 收 FULL）。
  ///
  /// Rust 认领模型下关闭发生在 IN_USE 态（上游从 OPEN 列表扫描，分歧登记）。
  /// 状态映射与上游一致：未满（`final_size < max_w_ptr`）→ `FullPad`（待补写，
  /// 保留活跃表直至 padding 完成 —— 上游 :1391-1430 的中间态）；已满 → `Full`
  /// 并撤出活跃表（上游 fast-path :1381-1385）。上游不变量：FULL 列表的块
  /// `cur_size == max_w_ptr`（GC 目标规模按整块计）。
  pub fn close_one_open_block(&self, bid: BlockId, final_size: u64) -> bool {
    let to = {
      let mut s = self.sspace.write();
      if s.block_metadata[bid as usize].state != BlockState::InUse {
        return false;
      }
      s.block_metadata[bid as usize].cur_size = final_size;
      let to = if final_size < s.max_w_ptr {
        BlockState::FullPad
      } else {
        BlockState::Full
      };
      s.move_block_to_list(bid, BlockState::InUse, to);
      to
    };
    if to == BlockState::Full {
      self.remove_block_id_from_active_block_list(bid);
    }
    true
  }

  /// 全部打开块关闭（上游 CloseAllOpenBlocks）
  pub fn close_all_open_blocks(&self) -> usize {
    let mut closed = 0;
    let ids: Vec<BlockId> = {
      let s = self.sspace.write();
      s.active_block_list
        .iter()
        .copied()
        .filter(|&b| b != BlockId::MAX)
        .collect()
    };
    for bid in ids {
      let sz = self.get_block_write_offset(bid);
      if self.close_one_open_block(bid, sz) {
        closed += 1;
      }
    }
    closed
  }

  // ---- GC 触发（C-1 重写：不再短路 ZNS）----

  /// GC 触发判定（C-1 重写：容量水位 + WAF 最优双条件，忽略入参 write_size 的
  /// 上游混乱语义（C-12）——以逻辑用量占比为准）
  pub fn should_trigger_gc(&self, write_size: u64) -> bool {
    // 锁外预取派生量（get_db_size 内部再取读锁 —— parking_lot 不可重入）
    let db_size = self.get_db_size();
    let mapped = self.pid2offset.read().len() as u64;
    // 只读聚合 + 独立原子标记：读锁足够（原 write() 使 GC 判定与全部写者互斥）
    let s = self.sspace.read();
    let total = s.block_cnt;
    if total == 0 {
      return false;
    }
    // 空块余量充足则不触发（C-32 同步修复：不再禁用空块分配）
    let empty = s.block_list_size(BlockState::Empty);
    let used_ratio = (total - empty as u64) * 100 / total;
    if used_ratio < GC_WATERMARK_PCT {
      s.gc_triggered.store(false, Ordering::Relaxed);
      return false;
    }
    // 有效页占比（mapped==0 视为无有效数据 => 碎片严重）
    let valid: u64 = s
      .block_metadata
      .iter()
      .filter(|m| m.state != BlockState::Empty)
      .map(|m| u64::from(m.valid_cnt()))
      .sum();
    let fragmented = mapped == 0 || {
      let waf = (valid.max(1) as f64) / (mapped as f64) * GC_WAF_LIMIT;
      waf < 1.0
    };
    // 容量水位 达标 && （碎片严重 || 写入增量触达）
    let trigger = fragmented || self.reached_alloc_size_with(db_size, write_size);
    s.gc_triggered.store(trigger, Ordering::Relaxed);
    trigger
  }

  /// 容量触达判定（上游 ReachedAllocSize；C-12 重复条件合并）
  pub fn reached_alloc_size(&self, write_size: u64) -> bool {
    self.reached_alloc_size_with(self.get_db_size(), write_size)
  }

  /// 纯计算版（db_size 由调用方预取，消除嵌套锁）
  fn reached_alloc_size_with(&self, db_size: u64, write_size: u64) -> bool {
    let last = self.last_gc_alloc.load(Ordering::Relaxed);
    // 自上次 GC 后新写入超过 1/4 块容量即再触发（C-12 重复条件合并）
    db_size + write_size > last + self.block_size / 4
  }

  /// 登记GC 后水位（GC 完成时调用）
  pub fn mark_gc_done(&self) {
    self
      .last_gc_alloc
      .store(self.get_db_size(), Ordering::Relaxed);
  }

  // ---- 释放 / 复用 ----

  /// 加入待释放列表
  pub fn add_pid_to_dealloc_list(&self, pid: PageId) {
    self.dealloc_list.write().push(pid);
  }

  /// 处理待释放列表（上游 ProcessDeallocList：映射清零 + 入复用池）
  pub fn process_dealloc_list(&self) -> usize {
    let pids = take(&mut *self.dealloc_list.write());
    let n = pids.len();
    {
      let mut table = self.pid2offset.write();
      for pid in &pids {
        if let Some(e) = table.get_mut(*pid as usize) {
          *e = Pid2Offset::default();
        }
      }
    }
    let mut pool = self.reusable_pids.write();
    for pid in pids {
      if pool.len() < MAX_REUSABLE_PIDS {
        pool.push(pid);
      }
    }
    n
  }

  /// 取可复用 PID（分配路径优先复用已释放页 —— 修复 D2 孤儿：
  /// 原版 `add_pid_to_dealloc_list` 只入列无消费方，池只进不出）
  pub fn take_reusable_pid(&self) -> Option<PageId> {
    self.reusable_pids.write().pop()
  }

  /// 登记可复用 PID（上游 AddReusablePID space_manager.cc:2135-2142 的
  /// FIFO 尾入；容量上限同 `MAX_REUSABLE_PIDS`）
  pub fn add_reusable_pid(&self, pid: PageId) {
    let mut pool = self.reusable_pids.write();
    if pool.len() < MAX_REUSABLE_PIDS {
      pool.push(pid);
    }
  }

  /// 池非空判定（上游 HasReusablePID space_manager.cc:2077-2088：
  /// head != tail 即有可复用 PID —— 谓词，不弹出）
  pub fn has_reusable_pid(&self) -> bool {
    !self.reusable_pids.read().is_empty()
  }

  // ---- 块失效历史 / 元数据更新族 ----

  /// 失效计数更新（上游 UpdateBlockInvalidCount space_manager.cc:1945-1955）：
  /// 仅当 `pid` 仍登记在 `page_offset` 所在物理槽的反向表时递增
  /// `invalidated_pid_cnt`（GC 已搬移的页不重复计数；`block_invalidation_cnt`
  /// 由 `add_block_invalidation_history` 单独负责 —— 上游同语义）。
  /// 返回是否命中。
  pub fn update_block_invalid_count(&self, page_offset: u64, pid: PageId) -> bool {
    let (bid, slot) = self.pid_offset_block(page_offset);
    let mut s = self.sspace.write();
    let hit = s
      .blocks
      .get_mut(bid as usize)
      .and_then(|b| b.get_mut(slot as usize))
      .is_some_and(|e| e.valid().contains(&pid));
    if hit {
      s.block_metadata[bid as usize].invalidated_pid_cnt += 1;
    }
    hit
  }

  /// 写后元数据更新（上游 UpdateBlockMetadataAfterWrite space_manager.cc:1491-1505；
  /// EDT 采用 CalcBlockAvgEDT 加权滑动平均 —— 单页写入等价 `pid_cnt = 1`）。
  pub fn update_block_metadata_after_write(&self, bid: BlockId, size: u64, edt: LogId) {
    let mut s = self.sspace.write();
    let avg = calc_block_avg_edt_of(&s.block_metadata[bid as usize], edt, 1);
    let m = &mut s.block_metadata[bid as usize];
    m.cur_size = size;
    m.edt = avg;
    m.total_pid_cnt += 1;
    m.block_min_edt = m.block_min_edt.min(edt);
    m.block_max_edt = m.block_max_edt.max(edt);
  }

  /// GC 写后元数据（上游 UpdateBlockMetadataAfterGCWrite）
  pub fn update_block_metadata_after_gc_write(&self, bid: BlockId, size: u64, edt: LogId) {
    self.update_block_metadata_after_write(bid, size, edt);
  }

  /// 异步 GC 写后（上游 UpdateBlockMetadataAfterAsyncGCWrite）
  pub fn update_block_metadata_after_async_gc_write(&self, bid: BlockId, size: u64, edt: LogId) {
    self.update_block_metadata_after_gc_write(bid, size, edt);
  }

  /// 同步 GC 写后（上游 UpdateBlockMetadataAfterSyncGCWrite）
  pub fn update_block_metadata_after_sync_gc_write(&self, bid: BlockId, size: u64, edt: LogId) {
    self.update_block_metadata_after_gc_write(bid, size, edt);
  }

  /// GC 读后（上游 UpdateBlockMetadataAfterReadGC space_manager.cc:692-699：
  /// `Ensure(state == IN_USE)` + `MoveBlockToList(IN_USE → GC_READ)` —— 列表与
  /// 元数据必须同步迁移，旧实现只改 state 造成列表/状态失联（C-30 同类））
  pub fn update_block_metadata_after_read_gc(&self, bid: BlockId) -> bool {
    let mut s = self.sspace.write();
    s.move_block_to_list(bid, BlockState::InUse, BlockState::GcRead)
  }

  /// 失效历史登记（上游 AddBlockInvalidationHistory；space 侧纯登记）
  pub fn add_block_invalidation_history(&self, bid: BlockId, gsn: LogId) {
    let mut s = self.sspace.write();
    let m = &mut s.block_metadata[bid as usize];
    m.block_invalidation_cnt += 1;
    m.block_max_edt = m.block_max_edt.max(gsn);
  }

  /// 写组历史（上游 AddWriteGroupHistory）
  pub fn add_write_group_history(&self, group: usize, bid: BlockId) {
    let mut s = self.sspace.write();
    if s.write_group.len() <= group {
      s.write_group.resize(group + 1, Vec::new());
    }
    s.write_group[group].push(bid);
  }

  /// 写组计数
  #[inline]
  pub fn write_group_cnt(&self) -> u32 {
    self.write_group_cnt
  }

  /// 前次 GC 块
  #[inline]
  pub fn get_prev_g_ced_block(&self) -> BlockId {
    self.prev_gced_block.load(Ordering::Relaxed)
  }

  /// 设置前次 GC 块
  #[inline]
  pub fn set_prev_g_ced_block(&self, bid: BlockId) {
    self.prev_gced_block.store(bid, Ordering::Relaxed);
  }

  /// 块计数
  #[inline]
  pub fn total_page_cnt_in_block(&self, bid: BlockId) -> u32 {
    self.block_meta(bid).total_pid_cnt
  }

  /// 活跃表空槽计数
  pub fn empty_slot_count_in_active_block_list(&self) -> u32 {
    let s = self.sspace.read();
    s.active_block_list
      .iter()
      .filter(|&&b| b == BlockId::MAX)
      .count() as u32
  }

  /// GC 进行中的活跃块计数
  pub fn gc_in_progess_block_count_in_active_list(&self) -> u32 {
    let s = self.sspace.read();
    s.active_block_list
      .iter()
      .filter(|&&b| b != BlockId::MAX && s.block_metadata[b as usize].state == BlockState::GcInUse)
      .count() as u32
  }

  /// 写满关闭前的 padding 检查（单块谓词：IN_USE 且未写满 —— 「必须先关闭再补写」
  /// 的前置条件）。上游同名 `CheckOpenBlockForPadding(write_sz)` 是批量扫描
  /// （space_manager.cc:1062-1084：OPEN 块装不下新写 → 关 FULL_PAD + 补写收 FULL），
  /// 编排半区随 ldb_buf 写路径接线；本函数为账本 port_rewrite 内的单块谓词。
  pub fn check_open_block_for_padding(&self, bid: BlockId) -> bool {
    let s = self.sspace.read();
    let m = &s.block_metadata[bid as usize];
    m.state == BlockState::InUse && m.cur_size < s.max_w_ptr
  }

  /// 已写 PID 计数（上游 CountWrittenPIDsInBlock）
  pub fn count_written_pi_ds_in_block(&self, bid: BlockId) -> u32 {
    self.block_meta(bid).total_pid_cnt
  }

  /// 唯一偏移判定（上游 IsUniqueOffset：复用池去重）
  pub fn is_unique_offset(&self, off: u64) -> bool {
    let table = self.pid2offset.read();
    table.iter().all(|e| e.get_offset() != off)
  }

  /// 有效页是否已安置（上游 ValidPIDsInSecurePlace：GC 迁移后核对）
  pub fn valid_pi_ds_in_secure_place(&self, pid: PageId, old_bid: BlockId) -> bool {
    let e = self.get_pid2_offset(pid);
    let (bid, _slot) = self.pid_offset_block(e.get_offset());
    bid != old_bid
  }

  /// PID 读放大视图（上游 ReadPID 的空间侧：读出所在块）
  pub fn pid_in_gc_block(&self, pid: PageId) -> Option<(BlockState, BlockId)> {
    let e = self.get_pid2_offset(pid);
    if e.get_comp_sz() == 0 {
      return None;
    }
    let (bid, _) = self.pid_offset_block(e.get_offset());
    let s = self.sspace.read();
    let st = s.block_metadata.get(bid as usize)?.state;
    match st {
      BlockState::GcRead | BlockState::GcWrite | BlockState::GcInUse => Some((st, bid)),
      _ => None,
    }
  }

  /// PID 在 GC 读块
  #[inline]
  pub fn pid_in_gcr_block(&self, pid: PageId) -> bool {
    matches!(self.pid_in_gc_block(pid), Some((BlockState::GcRead, _)))
  }

  /// PID 在 GC 写块
  #[inline]
  pub fn pid_in_gcw_block(&self, pid: PageId) -> bool {
    matches!(
      self.pid_in_gc_block(pid),
      Some((BlockState::GcWrite | BlockState::GcInUse, _))
    )
  }

  /// 块 GC 读计数
  #[inline]
  pub fn block_gc_read_cnt(&self, bid: BlockId) -> u32 {
    u32::from(self.block_meta(bid).state == BlockState::GcRead)
  }

  /// 块需 GC 判定（space 侧语义：写满且失效多）
  #[inline]
  pub fn block_needs_gc(&self, bid: BlockId) -> bool {
    let m = self.block_meta(bid);
    m.state == BlockState::Full && m.invalidated_pid_cnt > 0
  }

  /// 块处于 GC 占用
  #[inline]
  pub fn block_in_gc_use(&self, bid: BlockId) -> bool {
    self.block_meta(bid).state == BlockState::GcInUse
  }

  /// 元数据重置（上游 ResetBlockMetadata：块回收清零）
  pub fn reset_block_metadata(&self, bid: BlockId) {
    let mut s = self.sspace.write();
    s.block_metadata[bid as usize] = BlockMetadata::new(0, 0, BlockState::Empty);
  }

  /// 块空间重置（上游 ResetBlockSpace：反向表清空 + 状态回 EMPTY）。
  ///
  /// 顺序必须“先迁移后清零”：原实现先 reset_block_metadata（直接把 state 置
  /// Empty）再迁移，move_block_to_list(Empty→Empty) 因块仍挂在旧列表而必然
  /// 失败 —— 块永久卡死在旧列表且 state=Empty，列表与元数据失联（不可回收）
  pub fn reset_block_space(&self, bid: BlockId) -> bool {
    self.reset_offset2_pids(bid);
    let moved = {
      let mut s = self.sspace.write();
      let st = s.block_metadata[bid as usize].state;
      s.move_block_to_list(bid, st, BlockState::Empty)
    };
    self.reset_block_metadata(bid);
    moved
  }

  /// 块擦除（GC_WRITE 态路径；上游 EraseBlock 去 TrimBlock —— 打孔回收由 L5 执行）
  pub fn erase_block(&self, bid: BlockId) -> bool {
    let s = self.sspace.write();
    let st = s.block_metadata[bid as usize].state;
    drop(s);
    match st {
      BlockState::GcWrite => {
        self.reset_block_space(bid);
        true
      }
      _ => false,
    }
  }

  /// 销毁（上游 Destroy 的可移植半区：清空 dealloc 列表；trim 线程停止与
  /// 裸指针释放不移植 —— compio 单线程模型无独立线程，Rust 所有权自管理）
  pub fn destroy(&self) {
    self.process_dealloc_list();
  }

  /// GC 读取块迁移回写列表（`GC_READ → OPEN`）。
  ///
  /// **分歧登记**：状态迁移对应上游 `GarbageCollector::MoveBlocksFromGCToOpenBlockList`
  /// （garbage_collector.cc:836-849）；`cur_size != 0` 的 EDT 区间补洞对应
  /// `SpaceManager::MoveGCedBlockToOpenList`（space_manager.cc:1614-1664，其状态迁移
  /// 是 `GC_WRITE → OPEN`，`cur_size == 0` 分支走 `SetMinMaxEDTToMinimum`）。本实现
  /// 把「GC 读后重开」归一轮 `GC_READ → OPEN`，未移植 `SetMinMaxEDTToMinimum` 半区
  /// （其 `lsn_gap` 阈值状态属缓冲池，归 `ldb_buf`）；上游 `GC_WRITE → OPEN` 路径在本
  /// crate 由 `erase_block`（GcWrite→Empty）承担。
  pub fn move_g_ced_block_to_open_list(
    &self,
    bid: BlockId,
    global_min_edt: LogId,
    global_max_edt: LogId,
  ) -> bool {
    if self.block_meta(bid).cur_size != 0 {
      let _ = self.adjust_edt_range(global_min_edt, global_max_edt);
    }
    let mut s = self.sspace.write();
    s.move_block_to_list(bid, BlockState::GcRead, BlockState::Open)
  }

  /// 上游 AdjustEDTRange（space_manager.cc:523-655）对拍版：把
  /// `[global_min_edt, global_max_edt)` 中未被活跃块 `[block_min_edt,
  /// block_max_edt]` 覆盖的空洞并入相邻块。
  ///
  /// 补洞搜索与上游 :584-608 逐行一致：**单遍按活跃表序**扫描，块内先测
  /// `block_max_edt == gap_start`（扩上界至 `gap_end`）再测
  /// `block_min_edt == gap_end`（扩下界至 `gap_start`），命中即 break —— 取
  /// 活跃表序中首个命中「任一条件」的块（不是「全表先找上界匹配再找下界匹配」，
  /// 后者的归属在区间交叉时会偏离上游）。扩展**立即写回**（上游同：后续洞看到
  /// 已更新的 min/max）。
  /// 无相邻匹配则不补（上游同语义，留待尾部归并）。
  /// 尾部余量并入 `block_max_edt` 最大的块（:619-647：`max = gap_end`、
  /// `min = min(min, gap_start)` —— 下界必须随洞下探，否则 `[gap_start, 块min)`
  /// 仍未被任何块覆盖）。返回最后被扩展的块；无洞返回 None。
  /// C-36：哨兵槽一律跳过，不越界（上游尾洞循环未过滤哨兵，会读越界元数据）。
  ///
  /// **分歧登记**：上游 :534-556 的区间收集只取 `OPEN|IN_USE` 且
  /// `cur_size != max_w_ptr` 的块，并排除入参 `bid`（被 GC 重开的块）；本实现
  /// 无 `bid` 入参（`move_g_ced_block_to_open_list` 调用点未传），且不按状态过滤
  /// —— 活跃表内的 FullPad/GcInUse 块也会参与区间覆盖。当前唯一调用点
  /// `move_g_ced_block_to_open_list` 无生产消费方（`ldb_buf` 未接线），登记 NOTES。
  pub fn adjust_edt_range(&self, global_min_edt: LogId, global_max_edt: LogId) -> Option<BlockId> {
    let mut s = self.sspace.write();
    // 活跃表序（上游单遍扫描的匹配顺序），跳过哨兵槽与 (MAX,0) 空区间
    let scan = (s.max_open_block as usize).min(s.active_block_list.len());
    let bids: Vec<BlockId> = s.active_block_list[..scan]
      .iter()
      .copied()
      .filter(|&b| b != BlockId::MAX)
      .filter(|&b| {
        let m = &s.block_metadata[b as usize];
        !(m.block_min_edt == LogId::MAX && m.block_max_edt == 0)
      })
      .collect();
    if bids.is_empty() {
      return None;
    }
    // 按区间起点排序（纯计算，不动锁内状态）
    let mut sorted: Vec<(LogId, LogId)> = bids
      .iter()
      .map(|&b| {
        let m = &s.block_metadata[b as usize];
        (m.block_min_edt, m.block_max_edt)
      })
      .collect();
    sorted.sort_unstable();
    let mut cursor = global_min_edt;
    let mut last = None;
    for &(start, end) in &sorted {
      if cursor < start {
        let gap_start = cursor;
        let gap_end = start;
        // 单遍按活跃表序；块内先测上界匹配（上游 :584-608）
        let mut extend: Option<(BlockId, bool)> = None;
        for &b in &bids {
          let m = &s.block_metadata[b as usize];
          if m.block_max_edt == gap_start {
            extend = Some((b, true));
            break;
          }
          if m.block_min_edt == gap_end {
            extend = Some((b, false));
            break;
          }
        }
        if let Some((b, upper)) = extend {
          let m = &mut s.block_metadata[b as usize];
          if upper {
            m.block_max_edt = gap_end;
          } else {
            m.block_min_edt = gap_start;
          }
          last = Some(b);
        }
      }
      cursor = cursor.max(end);
    }
    if cursor < global_max_edt {
      // 尾洞 → block_max_edt 最大的块（上游 :619-647：max 置 gap_end、min 随洞下探）
      let mut best = BlockId::MAX;
      let mut hi_max = 0u64;
      for &b in &bids {
        let hi = s.block_metadata[b as usize].block_max_edt;
        if hi > hi_max {
          hi_max = hi;
          best = b;
        }
      }
      if best != BlockId::MAX {
        let m = &mut s.block_metadata[best as usize];
        m.block_max_edt = global_max_edt;
        m.block_min_edt = m.block_min_edt.min(cursor);
        last = Some(best);
      }
    }
    last
  }

  /// padding 后块迁移（上游 MovePaddedBlocksToFullList）
  pub fn move_padded_blocks_to_full_list(&self, bid: BlockId) -> bool {
    let mut s = self.sspace.write();
    s.move_block_to_list(bid, BlockState::FullPad, BlockState::Full)
  }

  /// GC 认领满块为写目标（上游 garbage_collector.cc:817 `FULL → GC_IN_USE`）。
  /// 补齐状态机缺口：GcInUse 原先无任何置位入口 ——
  /// find_first_gc_in_use_block_in_active_list / gc_in_progess_block_count_in_active_list
  /// / block_in_gc_use / add_to_active_block_list 的 GcInUse 分支全部不可达
  pub fn mark_block_gc_in_use(&self, bid: BlockId) -> bool {
    let mut s = self.sspace.write();
    if !s.move_block_to_list(bid, BlockState::Full, BlockState::GcInUse) {
      return false;
    }
    // GcInUse 允许进活跃表被续写（add_to_active_block_list 的守卫分支）；
    // 活跃表满时不入表，块仍可被 GC 写路径直接使用（上游同语义）
    let _ = s.add_to_active_block_list(bid);
    true
  }

  /// GC 写目标选定（上游 garbage_collector.cc:358 `GC_READ → GC_WRITE`：
  /// "Move block to GC_WRITE state only when it is selected"）。
  /// 补齐后 erase_block 的 `GcWrite → Empty` 收尾才可达（原实现 GcWrite 无入口，
  /// erase_block 恒 false，GC 写块永远无法擦除回收）
  pub fn mark_block_gc_write(&self, bid: BlockId) -> bool {
    self.move_block_to_list(bid, BlockState::GcRead, BlockState::GcWrite)
  }

  /// 列表迁移通用入口
  #[inline]
  pub fn move_block_to_list(&self, bid: BlockId, from: BlockState, to: BlockState) -> bool {
    if bid >= self.sspace.read().block_cnt {
      return false;
    }
    self.sspace.write().move_block_to_list(bid, from, to)
  }

  /// 记录有效页镜像（上游 SpaceManager::LogValidPage；GC 搬移前经 TxnLogHook 写 WAL）
  pub async fn log_valid_page(&self, pid: PageId) {
    self.txn_log.log_valid_page(pid).await;
  }

  /// PID 是否在待释放列表（账本命名）
  #[inline]
  pub fn is_pid_in_deallocated_list(&self, pid: PageId) -> bool {
    self.pid_is_deallocated(pid)
  }

  /// 块是否在活跃表
  #[inline]
  pub fn block_id_in_active_block_list(&self, bid: BlockId) -> bool {
    let s = self.sspace.read();
    s.active_block_list.contains(&bid)
  }

  /// 偏移反查 PID（线性扫映射表；冷路径 GC/恢复用）
  pub fn get_pid_from_offset(&self, off: u64) -> Option<PageId> {
    self
      .pid2offset
      .read()
      .iter()
      .position(|e| e.get_offset() == off && e.get_comp_sz() != 0)
      .map(|i| i as PageId)
  }

  /// 块内有效 PID 计数（账本命名；GC 牺牲块选择依据）
  #[inline]
  pub fn valid_pi_ds_cnt_in_block(&self, bid: BlockId) -> u32 {
    self.block_meta(bid).valid_cnt()
  }

  /// 初始化（上游 SpaceManager::Init；与 new 同构，独立文件布局 mapping_offset=0）
  pub fn init(
    block_cnt: u64,
    page_cnt_per_block: u32,
    min_comp_size: u32,
    max_open_block_cnt: u32,
    max_mapped_pages: u64,
    write_group_cnt: u32,
  ) -> Self
  where
    T: Default,
  {
    Self::new(
      block_cnt,
      page_cnt_per_block,
      min_comp_size,
      max_open_block_cnt,
      max_mapped_pages,
      write_group_cnt,
    )
  }

  /// 核心写路径：选择目标块（上游 SelectBlockIDToWrite；C-32 修复 —— 不再在
  /// GC 触发期禁用空块分配）。
  ///
  /// 修复 A7：上游 `try_again: sleep(1ms) + goto` 无界重试 —— compio 单线程
  /// 每核下若 GC 执行体与等待者同线程，`yield_now` 让出后无人推进 GC，即活锁。
  /// 改为触发 GC 判定后显式返回 `SpaceExhausted`，由调用方（GC task）推进回收后重进。
  pub async fn select_block_id_to_write(
    &self,
    write_sz: u64,
    avg_edt: LogId,
    min_edt: LogId,
    max_edt: LogId,
  ) -> Result<BlockId> {
    // 1. 打开块优先
    let bid = self.select_open_block_id(write_sz, avg_edt, min_edt, max_edt);
    if bid != BlockId::MAX {
      return Ok(bid);
    }
    // 2. GC 占用块续写
    let gc_bid = self.find_first_gc_in_use_block_in_active_list();
    if gc_bid != BlockId::MAX {
      return Ok(gc_bid);
    }
    // 3. 上游 :275-281 门控：不应开新块且 GC 已触发 → 等待 GC 推进（A7 替代
    //    sleep+goto：compio 单线程显式返回，GC task 回收后调用方重进）
    if !self.should_open_new_block_id(avg_edt)
      && self.sspace.read().gc_triggered.load(Ordering::Relaxed)
    {
      let _ = self.should_trigger_gc(write_sz);
      if let Some(h) = self.gc_hook.get() {
        let _ = h.on_gc_needed().await;
      }
      return Err(Error::SpaceExhausted);
    }
    // 4. 空块（C-32：GC 触发期不再禁用）
    let bid = self.select_empty_block_id(write_sz, avg_edt);
    if bid != BlockId::MAX {
      return Ok(bid);
    }
    // 5. 触发 GC 选读源 + 回调 GC 钩子（冷路径）后报耗尽（GC task 推进回收后调用方重试）
    let _ = self.should_trigger_gc(write_sz);
    if let Some(h) = self.gc_hook.get() {
      let _ = h.on_gc_needed().await;
    }
    Err(Error::SpaceExhausted)
  }

  /// 写分组版选择（page_io 侧；带分组亲和；A7 同步修复 —— 耗尽显式报错）
  pub async fn select_block_id_for_write_group(
    &self,
    write_sz: u64,
    avg_edt: LogId,
    group: usize,
  ) -> Result<BlockId> {
    let closest = self.select_open_block_id_with_closest_edt(avg_edt, group);
    if closest != BlockId::MAX {
      return Ok(closest);
    }
    self
      .select_block_id_to_write(write_sz, avg_edt, avg_edt, avg_edt)
      .await
  }

  /// NoWA 块关闭补写（上游 IssuePaddingWritesToFullBlock space_manager.cc:1379-1432
  /// 对拍版；ZNS 分支剔除，padding 零页经 PageIo GC 缓冲聚合，落盘由 ldb_buf 执行）。
  ///
  /// 对拍要点：
  /// - 入口态 `FullPad`（上游 :1387 `IsBlockInList(FULL_PAD)` 门控）；
  /// - 快路径：已 `Full` → 撤活跃表即返（上游 :1381-1385）；
  /// - 零页磁盘写由 `use_ssdwa1` 门控（上游 :1400 `FLAGS_use_SSDWA1_pattern ||
  ///   FLAGS_use_ZNS`，ZNS 已剔除），但**簿记无条件**（上游 :1420 的
  ///   `cur_size += padding_size`、:1426 `IN_USE→FULL`、:1430 撤活跃表不挂 flag）
  ///   —— 非 NoWA 时只跳过 staging（返回 0），块仍收账到 `Full`；
  /// - staging 不足（GC 缓冲满）不落半账，调用方 flush 后重试（上游直写无此失败面）。
  ///
  /// 返回 staged 零页数（0 = 无 staging 或已满）。padding 哨兵 pid
  /// （`PageId::MAX - n`）由调用方在 `update_space_metadata` 前过滤（CROSS 登记）；
  /// 上游额外把哨兵 pid 登进反表（:1414-1418 `UpdateOffset2PIDs(max_mapped_pages_cnt_,..)`），
  /// 本实现尾部槽反表恒空、GC 按「无登记」跳过，等价不加哨兵。
  pub async fn issue_padding_writes_to_full_block(
    &self,
    io: &mut PageIo,
    bid: BlockId,
    use_ssdwa1: bool,
  ) -> u32 {
    let (st, cur, max_w) = {
      let s = self.sspace.read();
      let m = &s.block_metadata[bid as usize];
      (m.state, m.cur_size, s.max_w_ptr)
    };
    // 快路径：已满（上游 :1381-1385 Ensure(cur==max) + 撤活跃表）
    if st == BlockState::Full {
      debug_assert_eq!(cur, max_w, "Full 块必须已补齐（上游不变量）");
      self.remove_block_id_from_active_block_list(bid);
      return 0;
    }
    if st != BlockState::FullPad || cur >= max_w {
      return 0;
    }
    let tail_pages = ((max_w - cur) / PAGE_SIZE as u64) as u32;
    let mut staged = 0u32;
    if use_ssdwa1 {
      let zero = [0u8; ldb_type::PAGE_SIZE];
      while staged < tail_pages
        && io.add_page_to_gc_write_buffer(PageId::MAX - staged as PageId, &zero, 0)
      {
        staged += 1;
      }
      if staged < tail_pages {
        // GC 缓冲不足：不落半账（cur_size/状态未动），调用方 flush 后重试
        return 0;
      }
    }
    // 簿记无条件（上游 :1420/:1426-1430）：cur_size 补齐、FullPad→Full、撤活跃表
    {
      let mut s = self.sspace.write();
      s.block_metadata[bid as usize].cur_size = cur + u64::from(tail_pages) * PAGE_SIZE as u64;
      let _ = s.move_block_to_list(bid, BlockState::FullPad, BlockState::Full);
    }
    self.remove_block_id_from_active_block_list(bid);
    staged
  }

  /// 映射表检查点（上游 CheckpointMappingTable：裸 liburing SQE → compio write；
  /// mapping.ckpt 独立文件、mapping_offset = 0，C-6 修复）
  pub async fn checkpoint_mapping_table(&self, sink: &mut dyn Sink) -> crate::Result<u64> {
    // 读守卫作用域块收敛：序列化在块内完成，await 点零守卫存活
    let (len, buf) = {
      let table = self.pid2offset.read();
      // 序列化：[len u64][每项 packed u64]
      let mut buf = Vec::with_capacity(8 + table.len() * 8);
      buf.extend_from_slice(&(table.len() as u64).to_le_bytes());
      for e in table.iter() {
        let packed = (e.get_offset() << ldb_pack::COMP_SZ_BITS) | u64::from(e.get_comp_sz());
        buf.extend_from_slice(&packed.to_le_bytes());
      }
      (table.len() as u64, buf)
    };
    sink.write_all_at(0, buf).await?;
    Ok(len)
  }

  /// 映射表检查点落盘尺寸（上游 AllocMappingTableSpace 的独立文件替代：
  /// mapping.ckpt 独立文件、mapping_offset = 0，C-6 修复）
  pub fn alloc_mapping_table_space(&self) -> u64 {
    let table = self.pid2offset.read();
    (table.len() as u64) * size_of::<Pid2Offset>() as u64
  }
}

impl Default for BlockMetadata {
  fn default() -> Self {
    Self::new(0, 0, BlockState::Empty)
  }
}

/// 映射表检查点落盘目标（ldb_io 后端适配；独立 mapping.ckpt 文件）
pub trait Sink {
  /// 整段写（覆盖式检查点）
  fn write_all_at(
    &mut self,
    off: u64,
    data: Vec<u8>,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>>;
}

impl Sink for ldb_io::CompioFileBackend {
  fn write_all_at(
    &mut self,
    off: u64,
    data: Vec<u8>,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(self.write_owned(data, off))
  }
}

impl Sink for ldb_io::MemBackend {
  fn write_all_at(
    &mut self,
    off: u64,
    data: Vec<u8>,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(async move {
      let start = off as usize;
      let _end = start + data.len();
      for (i, chunk) in data.chunks(ldb_type::PAGE_SIZE).enumerate() {
        let mut page = [0u8; ldb_type::PAGE_SIZE];
        page[..chunk.len()].copy_from_slice(chunk);
        let _ = self
          .write_at_page(((start / ldb_type::PAGE_SIZE) + i) as u64, &page)
          .await;
      }
      Ok(())
    })
  }
}
