//! 页面状态字（定义下沉 L0 的依据见 todo.md §3.0 反向边 5）。
//!
//! 64 位原子值：高 8 位状态（`PageStateMode`），低 56 位版本号
//! （上游 `ZLeanStore/src/include/sync/hybrid_latch.h:32` 的 `VERSION_MASK` 与
//! `:72-82` 的 `SameVersionNewState` / `NextVersionNewState` 位布局；
//! `PageStateMode` 编码见 `ZLeanStore/src/include/sync/page_state.h:13-18`）。
//! 完整锁转换状态机在 L1 `ldb_sync`。

use std::sync::atomic::{AtomicU64, Ordering};

/// 低 56 位版本掩码（上游 `hybrid_latch.h:32` 的 `VERSION_MASK`）
pub const VERSION_MASK: u64 = (1u64 << 56) - 1;

// 状态字高 8 位的协议编码（上游 `page_state.h:13-18`）。字面量只在此处书写一次：
// 枚举判别式与 `from_state_word` 的解码臂都引用这些常量，任一处漂移即编译失败。
const CODE_UNLOCKED: u8 = 0;
const CODE_MAX_SHARED: u8 = 252;
const CODE_EXCLUSIVE: u8 = 253;
const CODE_MARKED: u8 = 254;
const CODE_EVICTED: u8 = 255;

/// 页面锁状态编码（状态字高 8 位）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageStateMode {
  /// 无持有者（0）
  Unlocked = CODE_UNLOCKED,
  /// 共享持有数（直接以计数值编码，合法域 `1..=MAX_SHARED`，仅 [`PageStateMode::from_state_word`] 产生）
  Shared(u8),
  /// 排他（253）
  Exclusive = CODE_EXCLUSIVE,
  /// Clock 置换标记（254）
  Marked = CODE_MARKED,
  /// 已驱逐（255）
  Evicted = CODE_EVICTED,
}

impl PageStateMode {
  /// 无持有者编码（上游 `sync/page_state.h:13`）
  pub const UNLOCKED: u8 = CODE_UNLOCKED;
  /// 共享持有数上限（上游 `sync/page_state.h:14-15`）：共享码域为 `1..=MAX_SHARED`
  pub const MAX_SHARED: u8 = CODE_MAX_SHARED;
  /// 排他编码（上游 `sync/page_state.h:16`）
  pub const EXCLUSIVE: u8 = CODE_EXCLUSIVE;
  /// Clock 置换标记编码（上游 `sync/page_state.h:17`）
  pub const MARKED: u8 = CODE_MARKED;
  /// 已驱逐编码（上游 `sync/page_state.h:18`）
  pub const EVICTED: u8 = CODE_EVICTED;

  /// 从状态字高 8 位解码（`const`：可在编译期构造锁定初值状态字）
  #[must_use]
  #[inline]
  pub const fn from_state_word(v: u64) -> Self {
    match (v >> 56) as u8 {
      Self::UNLOCKED => Self::Unlocked,
      Self::EXCLUSIVE => Self::Exclusive,
      Self::MARKED => Self::Marked,
      Self::EVICTED => Self::Evicted,
      n => Self::Shared(n),
    }
  }
}

// 上游 `hybrid_latch.h:65-69` 的共享态判定为 `UNLOCKED < state < EXCLUSIVE`，
// 附加态只允许排在 EXCLUSIVE 之后 —— 共享码域必须严格夹在两端之间，编译期钉死。
const _: () = {
  assert!(PageStateMode::UNLOCKED < PageStateMode::MAX_SHARED);
  assert!(PageStateMode::MAX_SHARED < PageStateMode::EXCLUSIVE);
};

/// 页面状态原子字
#[derive(Debug, Default)]
pub struct PageState {
  state_and_version: AtomicU64,
}

impl PageState {
  /// 构造初始状态（UNLOCKED / 版本 0）
  #[must_use]
  #[inline]
  pub const fn new() -> Self {
    Self {
      state_and_version: AtomicU64::new(0),
    }
  }

  /// 读原始状态字
  #[inline]
  pub fn load(&self, ord: Ordering) -> u64 {
    self.state_and_version.load(ord)
  }

  /// 写原始状态字（状态机转换用）
  #[inline]
  pub fn store(&self, v: u64, ord: Ordering) {
    self.state_and_version.store(v, ord);
  }

  /// CAS 状态字
  #[inline]
  pub fn cas(&self, cur: u64, new: u64, ord: Ordering) -> Result<u64, u64> {
    match self
      .state_and_version
      .compare_exchange(cur, new, ord, Ordering::Relaxed)
    {
      Ok(v) => Ok(v),
      Err(actual) => Err(actual),
    }
  }

  /// 当前锁状态
  ///
  /// 上游 `hybrid_latch.cc:26-28` 的成员 `LockState()` 用 `load()` 默认 seq_cst；
  /// 本函数保持同一内存序。aarch64 RCpc 目标上 `Acquire` 会编译为 `ldapr`（RCpc），
  /// 而 seq_cst 编译为 `ldar`（RCsc，与上游一致），两者并非同一指令
  /// （round12 实测 asm；本机两类 load 均 ~0.9 ns/op，无热路径回归）。
  #[inline]
  pub fn lock_state(&self) -> PageStateMode {
    PageStateMode::from_state_word(self.load(Ordering::SeqCst))
  }

  /// 当前版本号（低 56 位）
  ///
  /// 上游 `hybrid_latch.cc:30-32` 的成员 `Version()` 用 `load()` 默认 seq_cst，
  /// 本函数保持同一内存序（理由同 [`PageState::lock_state`]）。
  ///
  /// 注意：单次 load（即使 seq_cst）只约束其后的访问，不阻止其前的页数据读被
  /// 重排到校验之后；乐观读协议必须由调用方在页数据读取后、第二次校验前插入
  /// `fence(Ordering::Acquire)` 并比较完整状态字（见 CROSS.md → ldb_sync）。
  #[inline]
  pub fn version(&self) -> u64 {
    self.load(Ordering::SeqCst) & VERSION_MASK
  }
}

/// 保留旧版本、换新状态（`new_state` 取 [`PageStateMode`] 的编码常量，如
/// [`PageStateMode::EXCLUSIVE`]；`const`：可在编译期算出锁定初值状态字）
#[must_use]
#[inline]
pub const fn same_version_new_state(old: u64, new_state: u8) -> u64 {
  (old & VERSION_MASK) | (new_state as u64) << 56
}

/// 版本 +1、换新状态（上游 `hybrid_latch.h:78-82`，语义与上游逐位一致：
/// 版本位满值进位会污染状态位，与上游同）
#[must_use]
#[inline]
pub const fn next_version_new_state(old: u64, new_state: u8) -> u64 {
  ((old & VERSION_MASK) + 1) | (new_state as u64) << 56
}
