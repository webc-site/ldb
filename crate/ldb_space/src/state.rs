//! 空间管理状态结构（上游 `space_manager.h:36-110`）。
//!
//! 上游缺陷修复：C-30（`block_lists` 三套锁并存、GC 线程无锁遍历 UB）→
//! 统一为每状态 `parking_lot::Mutex<Vec<BlockId>>` 快照迭代。

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use ldb_type::{BlockId, LogId, PageId};
use parking_lot::RwLock;

/// 逻辑块状态机（上游 `State`）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum BlockState {
  /// 空闲（可分配）
  Empty = 0,
  /// 活跃写入中（被 worker 认领）
  InUse = 1,
  /// 打开待写（未认领）
  Open = 2,
  /// 已写满关闭
  Full = 3,
  /// GC 认领（写入中）
  GcInUse = 4,
  /// GC 读源
  GcRead = 5,
  /// GC 写目标
  GcWrite = 6,
  /// NoWA padding 后满块
  FullPad = 7,
}

pub const STATE_CNT: usize = 8;

impl BlockState {
  /// 数组下标
  #[inline]
  pub fn idx(self) -> usize {
    self as usize
  }

  /// 从 u16 解码
  #[inline]
  pub fn from_u16(v: u16) -> Self {
    match v {
      1 => Self::InUse,
      2 => Self::Open,
      3 => Self::Full,
      4 => Self::GcInUse,
      5 => Self::GcRead,
      6 => Self::GcWrite,
      7 => Self::FullPad,
      _ => Self::Empty,
    }
  }
}

/// 物理页槽 → PID 反向表条目（上游 `Offset2PIDs`）
#[derive(Debug, Clone, Default)]
pub struct Offset2Pids {
  /// 该物理槽位承载的 PID 列表（压缩装箱时多页同槽）
  pub pids: Vec<PageId>,
  /// 有效计数
  pub cnt: u16,
}

impl Offset2Pids {
  /// 追加 PID
  pub fn push(&mut self, pid: PageId) {
    if (self.cnt as usize) < self.pids.len() {
      self.pids[self.cnt as usize] = pid;
    } else {
      self.pids.push(pid);
    }
    self.cnt += 1;
  }

  /// 有效 PID 视图
  pub fn valid(&self) -> &[PageId] {
    &self.pids[..self.cnt as usize]
  }
}

/// 逻辑块元数据（上游 `BlockMetadata`）
#[derive(Debug, Clone, Copy)]
pub struct BlockMetadata {
  /// 已写字节
  pub cur_size: u64,
  /// 累计页数
  pub total_pid_cnt: u32,
  /// 已失效页数
  pub invalidated_pid_cnt: u32,
  /// 失效事件数
  pub block_invalidation_cnt: u16,
  /// 状态
  pub state: BlockState,
  /// 块 EDT（最新）
  pub edt: LogId,
  /// 块内最小 EDT
  pub block_min_edt: LogId,
  /// 块内最大 EDT
  pub block_max_edt: LogId,
}

impl BlockMetadata {
  /// 空块构造
  pub fn new(edt: LogId, size: u64, st: BlockState) -> Self {
    Self {
      cur_size: size,
      total_pid_cnt: 0,
      invalidated_pid_cnt: 0,
      block_invalidation_cnt: 0,
      state: st,
      edt,
      block_min_edt: LogId::MAX,
      block_max_edt: 0,
    }
  }

  /// 有效页数
  #[inline]
  pub fn valid_cnt(&self) -> u32 {
    self.total_pid_cnt.saturating_sub(self.invalidated_pid_cnt)
  }
}

/// 存储空间主体（上游 `StorageSpace`；锁模型修复见模块注释）
#[derive(Debug)]
pub struct StorageSpace {
  /// 每块的物理槽反向表
  pub blocks: Vec<Vec<Offset2Pids>>,
  /// 每块元数据
  pub block_metadata: Vec<BlockMetadata>,
  /// 每状态块 ID 列表（互斥保护，C-30 修复）
  pub block_lists: Vec<MutexVec>,
  /// 块元数据总锁（粗粒度写保护）
  pub blk_state_lock: parking_lot::RwLock<()>,
  /// padding 补写锁
  pub padding_lock: parking_lot::RwLock<()>,
  /// 最小压缩尺寸
  pub min_comp_size: u32,
  /// 每块页数
  pub page_cnt_per_block: u32,
  /// 块总数
  pub block_cnt: u64,
  /// 块对齐尺寸
  pub block_size: u64,
  /// 块内写指针上限
  pub max_w_ptr: u64,
  /// 可映射最大页数
  pub max_mapped_pages_cnt: u64,
  /// 最大打开块数
  pub max_open_block: u32,
  /// 活跃块环形表（EMPTY 哨兵 = BlockId::MAX）
  pub active_block_list: Vec<BlockId>,
  /// 活跃块计数
  pub active_block_cnt: AtomicU32,
  /// 写分组（NoWA 均衡组）
  pub write_group: Vec<Vec<BlockId>>,
  /// GC 已触发
  pub gc_triggered: AtomicBool,
  /// 未补偿写计数
  pub not_compensation_write: AtomicU32,
  /// 补偿写中
  pub compensation_write: AtomicBool,
}

/// 带锁块列表（快照迭代语义）
#[derive(Debug, Default)]
pub struct MutexVec {
  inner: parking_lot::Mutex<Vec<BlockId>>,
}

impl MutexVec {
  /// 长度
  #[inline]
  pub fn len(&self) -> usize {
    self.inner.lock().len()
  }

  /// 是否为空
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.inner.lock().is_empty()
  }

  /// 追加
  pub fn push(&self, bid: BlockId) {
    self.inner.lock().push(bid);
  }

  /// 头插（EMPTY 列表复用语义，上游 insert(begin)）
  pub fn push_front(&self, bid: BlockId) {
    self.inner.lock().insert(0, bid);
  }

  /// 弹出尾部
  pub fn pop(&self) -> Option<BlockId> {
    self.inner.lock().pop()
  }

  /// 弹出头部
  pub fn pop_front(&self) -> Option<BlockId> {
    let mut v = self.inner.lock();
    if v.is_empty() {
      None
    } else {
      Some(v.remove(0))
    }
  }

  /// 窥视头部（不移出）
  pub fn front(&self) -> Option<BlockId> {
    self.inner.lock().first().copied()
  }

  /// 移除指定元素
  pub fn remove(&self, bid: BlockId) -> bool {
    let mut v = self.inner.lock();
    let Some(pos) = v.iter().position(|&b| b == bid) else {
      return false;
    };
    v.remove(pos);
    true
  }

  /// 包含判定
  pub fn contains(&self, bid: BlockId) -> bool {
    self.inner.lock().contains(&bid)
  }

  /// 快照（GC 遍历用，C-30 修复：不再无锁迭代）
  pub fn snapshot(&self) -> Vec<BlockId> {
    self.inner.lock().clone()
  }
}

impl StorageSpace {
  /// 构造（上游 StorageSpace ctor）
  pub fn new(
    block_cnt: u64,
    page_cnt_per_block: u32,
    min_comp_size: u32,
    block_size: u64,
    max_w_ptr: u64,
    max_open_block_cnt: u32,
  ) -> Self {
    let blocks: Vec<Vec<Offset2Pids>> = (0..block_cnt)
      .map(|_| vec![Offset2Pids::default(); page_cnt_per_block as usize])
      .collect();
    Self {
      blocks,
      block_metadata: (0..block_cnt)
        .map(|_| BlockMetadata::new(0, 0, BlockState::Empty))
        .collect(),
      block_lists: {
        // 全部块初始登记 EMPTY（上游 ctor 语义）
        let lists: Vec<MutexVec> = (0..STATE_CNT).map(|_| MutexVec::default()).collect();
        for bid in (0..block_cnt).rev() {
          lists[BlockState::Empty.idx()].push(bid);
        }
        lists
      },
      blk_state_lock: RwLock::new(()),
      padding_lock: RwLock::new(()),
      min_comp_size,
      page_cnt_per_block,
      block_cnt,
      block_size,
      max_w_ptr,
      max_mapped_pages_cnt: 0,
      max_open_block: max_open_block_cnt,
      active_block_list: vec![BlockId::MAX; max_open_block_cnt as usize],
      active_block_cnt: AtomicU32::new(0),
      write_group: Vec::new(),
      gc_triggered: AtomicBool::new(false),
      not_compensation_write: AtomicU32::new(0),
      compensation_write: AtomicBool::new(false),
    }
  }

  /// 块状态迁移（上游 MoveBlockToList：元数据与列表双重校验）。
  /// 原实现 contains + remove 两次 O(n) 线性扫万级块列表 —— remove 自带
  /// 定位语义，单次即可；死代码 `_first/_second`（算了锁序却从未使用）一并移除
  /// （实际串行化由外层 sspace 闩提供，列表内层锁仅护 Vec 自身）
  pub fn move_block_to_list(&mut self, bid: BlockId, from: BlockState, to: BlockState) -> bool {
    if self.block_metadata[bid as usize].state != from {
      return false;
    }
    if !self.block_lists[from.idx()].remove(bid) {
      return false;
    }
    self.block_metadata[bid as usize].state = to;
    if to == BlockState::Empty {
      self.block_lists[to.idx()].push_front(bid);
    } else {
      self.block_lists[to.idx()].push(bid);
    }
    true
  }

  /// 更新块元数据（上游 UpdateBlockMetadata）
  pub fn update_block_metadata(&mut self, bid: BlockId, size: u64, edt: LogId, state: BlockState) {
    debug_assert!(bid < self.block_cnt);
    let meta = &mut self.block_metadata[bid as usize];
    meta.cur_size = size;
    meta.edt = edt;
    meta.state = state;
  }

  /// 状态列表长度
  #[inline]
  pub fn block_list_size(&self, st: BlockState) -> usize {
    self.block_lists[st.idx()].len()
  }

  /// 登记活跃块表（上游 AddBlockIDToActiveBlockList；要求 EMPTY|GC_IN_USE 态）
  pub fn add_to_active_block_list(&mut self, bid: BlockId) -> bool {
    let meta = self.block_metadata[bid as usize].state;
    if meta != BlockState::Empty && meta != BlockState::GcInUse {
      return false;
    }
    for slot in self.active_block_list.iter_mut() {
      if *slot == BlockId::MAX {
        *slot = bid;
        self.active_block_cnt.fetch_add(1, Ordering::Relaxed);
        return true;
      }
    }
    false
  }

  /// 块是否在列表
  #[inline]
  pub fn block_in_list(&self, bid: BlockId, st: BlockState) -> bool {
    self.block_lists[st.idx()].contains(bid)
  }
}
