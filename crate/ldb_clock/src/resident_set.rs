//! 常驻页集合：固定容量开放寻址散列表 + Clock 二次机会置换
//! （移植自 `ZLeanStore/src/buffer/resident_set.cc`）。
//!
//! 已修复的上游缺陷（逐条见 `todo.md` §9 C-22 / C-26）：
//! - C-22：`IterateClockBatch` CAS 预留后重读 `clock_pos_`（批次漂移）→ 用 CAS 时的
//!   `pos` 遍历；
//! - C-22：`valid_counts_` 增减全被注释，唯一读者 `GetShardWithMostValidPages` 在全仓
//!   无调用方 ⇒ 删除分片统计与 `GetShardForSlot/ForPID/WithMostValidPages` 三个死接口，
//!   插入/删除不再做无消费方的原子 RMW；
//! - C-26：`pid == TOMBSTONE` 会被插进 EMPTY 槽（`Insert` 先比较 `curr == page_id`，
//!   `curr` 为 EMPTY 时不等，随后 CAS 成功），该槽此后被遍历/查找永久跳过 →
//!   三个入口统一拒绝 `EMPTY`/`TOMBSTONE` 保留值；
//! - 满表 `Insert` 与「表内无 EMPTY 时 `Remove` 不存在的页」在上游是无界自旋 →
//!   线性探测走完一圈即返回 `false`（容量契约被违反时降级为失败而非挂死）；
//! - `SetGCPosToCP`/`gc_pos_` 上游只写不读（全仓 0 调用方）⇒ 删除。
//!
//! 散列用 gxhash（替代上游 murmur 变体 HashFn）；种子固定，槽位寻址跨调用确定。

use std::{
  fmt,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
};

use gxhash::gxhash64;
use ldb_type::{PageId, PageState, PageStateMode};

/// 固定散列种子：槽位寻址必须跨调用确定性
/// （`GxBuildHasher::default` 是 RandomState 派生 seed，每次 build 不同，不可用）
const HASH_SEED: i64 = 0x1db;

/// 空槽哨兵（上游 `resident_set.h:15`）
pub const EMPTY: u64 = u64::MAX;
/// 删除墓碑（上游 `resident_set.h:16`）
pub const TOMBSTONE: u64 = u64::MAX - 1;

/// 槽位负载因子（上游 `resident_set.cc:19` 的 `1.8`）
const LOAD_NUM: f64 = 1.8;

/// 页号散列：`to_le_bytes` 显式固定字节序，槽位布局只由页号决定，
/// 不随目标端序变化（`to_ne_bytes` 在大端上喂入不同字节序列 ⇒ 不同槽位）。
/// 本表不落盘、启动即重建，故该确定性只为跨平台可复现（测试/诊断），无兼容负担。
#[inline]
fn hash_pid(pid: PageId) -> u64 {
  gxhash64(&pid.to_le_bytes(), HASH_SEED)
}

/// 常驻页集合
pub struct ResidentPageSet {
  capacity: u64,
  count: u64,
  /// `count` 为 2 的幂，取模退化为掩码
  mask: u64,
  clock_pos: AtomicU64,
  cp_pos: AtomicU64,
  slots: Box<[AtomicU64]>,
  page_state: Arc<[PageState]>,
}

impl fmt::Debug for ResidentPageSet {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ResidentPageSet")
      .field("capacity", &self.capacity)
      .field("count", &self.count)
      .field("clock_pos", &self.clock_pos.load(Ordering::Relaxed))
      .finish()
  }
}

impl ResidentPageSet {
  /// 构造：`max_count` 为页号空间容量（表最多容纳的不同页号数）；
  /// `page_state` 为 L0 页状态表切片。
  ///
  /// `count = bit_ceil(trunc(max_count * 1.8))`（上游 `resident_set.cc:19` 先
  /// `static_cast<u64>` 再 `bit_ceil`；此处不能写成 `ceil(max_count * 1.8)`，
  /// 否则 capacity=9/18/36/569... 会多分配一倍槽位）。
  ///
  /// 容量契约：`max_count` 必须 ≥ 页号空间（`ldb_buf` 传 `virtual_cnt`）。
  /// 此时常驻页集合见过的不同页号数 ≤ `max_count`；每个页号首次插入最多消费
  /// 一个 EMPTY 槽（删除只留墓碑、同页号再插入复用墓碑），故占用槽
  /// ≤ `max_count`，EMPTY ≥ `count - max_count`：连续值约 44.4%，`trunc` 后
  /// 最坏 7/16 = 43.75%（capacity=1 时 count=1，无 EMPTY）。探测链长度因此
  /// 有界。
  ///
  /// 若传入远小于页号空间的值（上游传的是
  /// `physical_cnt_`，`buffer_manager.cc:123`），墓碑会在进程生命周期内耗尽
  /// 全部 EMPTY 槽，`contain` miss 与两遍 `insert` 的探测链退化到 O(count)
  /// —— 上游同样无墓碑压缩/rehash（设计限制，非移植缺陷）。
  ///
  /// `_shard_cnt`：上游 `iohandler` 只用于分片计数（已删，见模块注释），
  /// 保留参数以兼容 `ldb_buf` 现有调用点。
  pub fn new(max_count: u64, page_state: Arc<[PageState]>, _shard_cnt: u32) -> Self {
    assert!(max_count > 0, "页号空间容量必须为正");
    let count = ((max_count as f64 * LOAD_NUM) as u64).next_power_of_two();
    let slots: Box<[AtomicU64]> = (0..count).map(|_| AtomicU64::new(EMPTY)).collect();
    Self {
      capacity: max_count,
      count,
      mask: count - 1,
      clock_pos: AtomicU64::new(0),
      cp_pos: AtomicU64::new(0),
      slots,
      page_state,
    }
  }

  /// 槽位引用（探测热路径统一入口：pos 恒由掩码产生，免逐次边界检查）
  #[inline]
  fn slot(&self, pos: u64) -> &AtomicU64 {
    // SAFETY: pos 一律来自 `hash & mask` 或 `(pos + 1) & mask`；
    // mask = count - 1（count 为 2 的幂）且 slots.len() == count，
    // 故 pos < slots.len() 恒成立（构造时由 new 保证两者同步分配）
    unsafe { self.slots.get_unchecked(pos as usize) }
  }

  /// 构造容量（页号空间容量，见 [`Self::new`]）
  #[inline]
  pub fn capacity(&self) -> u64 {
    self.capacity
  }

  /// 槽位总数
  #[inline]
  pub fn count(&self) -> u64 {
    self.count
  }

  /// 页状态表引用（L0 PageState）
  #[inline]
  fn get_page_state(&self, pid: PageId) -> &PageState {
    &self.page_state[pid as usize]
  }

  /// 是否常驻（`pid` 必须是 `0..TOMBSTONE` 的合法页号）
  pub fn contain(&self, page_id: PageId) -> bool {
    if page_id >= TOMBSTONE {
      return false;
    }
    let mut pos = hash_pid(page_id) & self.mask;
    let start = pos;
    loop {
      let curr = self.slot(pos).load(Ordering::Acquire);
      if curr == EMPTY {
        return false;
      }
      if curr == page_id {
        return true;
      }
      pos = (pos + 1) & self.mask;
      if pos == start {
        return false;
      }
    }
  }

  /// 插入（须已持有该页 EXCLUSIVE 锁，调用方保证；debug 下校验）
  ///
  /// 两遍探测：先沿探测链证明键不存在（走到 EMPTY 或走完一圈），再回填到
  /// 遇到的第一个墓碑/该 EMPTY。证明与回填共享同一次链上遍历（边扫边记
  /// `first_tomb`，不是两次内存扫描）；相对上游单遍实现只多走「首个墓碑之后
  /// 到 EMPTY」这一段。上游单遍实现在「本键因更早的碰撞被放到槽 k+1，
  /// 随后槽 k 的占用者被删除留下墓碑」时会把同一 pid 再插进槽 k，形成重复项
  /// （对拍实测：capacity=8/seed=1 第 131 个操作重复；后果：remove 只摘一个、
  /// clock 批次重复回调、常驻页计数虚高）。该修复不可退化为「单遍 + 标记」：
  /// 页号空间覆盖除 `EMPTY`/`TOMBSTONE` 外的全部 u64 值，没有第三个值可充当
  /// 「已认领」标记；而任何幂等实现都必须扫到 EMPTY/绕圈才能证明键不存在。
  pub fn insert(&self, page_id: PageId) -> bool {
    if page_id >= TOMBSTONE {
      return false;
    }
    debug_assert_eq!(
      self.get_page_state(page_id).lock_state(),
      PageStateMode::Exclusive
    );
    let home = hash_pid(page_id) & self.mask;
    'retry: loop {
      let mut pos = home;
      let mut first_tomb = None;
      loop {
        let curr = self.slot(pos).load(Ordering::Acquire);
        if curr == page_id {
          return false;
        }
        if curr == EMPTY {
          break;
        }
        if curr == TOMBSTONE && first_tomb.is_none() {
          first_tomb = Some(pos);
        }
        pos = (pos + 1) & self.mask;
        if pos == home {
          // 全表无 EMPTY：只能复用墓碑；连墓碑也没有 ⇒ 容量契约被违反
          // （上游此处无界自旋）。
          let Some(tomb) = first_tomb else {
            return false;
          };
          if self
            .slot(tomb)
            .compare_exchange(TOMBSTONE, page_id, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
          {
            return true;
          }
          continue 'retry;
        }
      }
      let target = first_tomb.unwrap_or(pos);
      let expected = if target == pos { EMPTY } else { TOMBSTONE };
      if self
        .slot(target)
        .compare_exchange(expected, page_id, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
      {
        return true;
      }
      // 目标槽被并发改写：重扫（键也可能已被并发插入 → 下一遍返回 false）
      continue 'retry;
    }
  }

  /// 移除（调用方须持该页 EXCLUSIVE；上游 `resident_set.cc:99` 同断言）
  pub fn remove(&self, page_id: PageId) -> bool {
    if page_id >= TOMBSTONE {
      return false;
    }
    debug_assert_eq!(
      self.get_page_state(page_id).lock_state(),
      PageStateMode::Exclusive
    );
    let mut pos = hash_pid(page_id) & self.mask;
    let start = pos;
    loop {
      let curr = self.slot(pos).load(Ordering::Acquire);
      if curr == EMPTY {
        return false;
      }
      if curr == page_id
        && self
          .slot(pos)
          .compare_exchange(curr, TOMBSTONE, Ordering::AcqRel, Ordering::Acquire)
          .is_ok()
      {
        return true;
      }
      pos = (pos + 1) & self.mask;
      // 墓碑永不回落为 EMPTY：满表（无 EMPTY 哨兵）时删除不存在的页会永久自旋
      if pos == start {
        return false;
      }
    }
  }

  /// CAS 预留 clock 批次并遍历（修复 C-22：用 CAS 时的 pos，不再重读）
  ///
  /// `batch > count` 时会绕圈重复遍历同一批槽位，`evict_fn` 对同一页号可能被
  /// 回调多次（上游 `resident_set.cc:135-143` 的 `for idx < batch` 同语义）。
  /// 调用方须按幂等处理重复候选：`ldb_buf::evict` 靠 `try_mark_page` 的
  /// UNLOCKED→MARKED CAS 吸收第二次回调（第二次必失败）。
  pub fn iterate_clock_batch(&self, batch: u64, mut evict_fn: impl FnMut(PageId)) {
    let mut pos;
    loop {
      pos = self.clock_pos.load(Ordering::Acquire);
      let new_pos = pos.wrapping_add(batch) & self.mask;
      if self
        .clock_pos
        .compare_exchange_weak(pos, new_pos, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
      {
        break;
      }
    }
    for _ in 0..batch {
      let curr = self.slot(pos).load(Ordering::Acquire);
      if curr != TOMBSTONE && curr != EMPTY {
        evict_fn(curr);
      }
      pos = (pos + 1) & self.mask;
    }
  }

  /// 从 checkpoint 位置起取一个常驻页（无则 TOMBSTONE）
  pub fn get_pid(&self, cnt: u64) -> PageId {
    let pos = self.cp_pos.load(Ordering::Acquire);
    let slot = cnt.wrapping_add(pos) & self.mask;
    let curr = self.slot(slot).load(Ordering::Acquire);
    let _ = self
      .cp_pos
      .compare_exchange(pos, slot, Ordering::AcqRel, Ordering::Acquire);
    if curr != TOMBSTONE && curr != EMPTY {
      curr
    } else {
      TOMBSTONE
    }
  }

  /// 从 clock 位置偏移 `batch` 取一个驱逐候选（无则 TOMBSTONE）
  pub fn get_pid_to_evict(&self, batch: u64) -> PageId {
    let pos = self.clock_pos.load(Ordering::Acquire);
    let slot = batch.wrapping_add(pos) & self.mask;
    let curr = self.slot(slot).load(Ordering::Acquire);
    let _ = self
      .clock_pos
      .compare_exchange(pos, slot, Ordering::AcqRel, Ordering::Acquire);
    if curr != TOMBSTONE && curr != EMPTY {
      curr
    } else {
      TOMBSTONE
    }
  }
}
