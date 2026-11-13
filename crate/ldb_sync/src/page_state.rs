//! PageState 的 Clock 置换状态机扩展（上游 `page_state.cc`）。
//!
//! L0 `ldb_type::PageState` 只定义原子字与解码；本模块以扩展 trait 补齐
//! MARKED/EVICTED 转换（孤儿规则下的合法形态）。
//!
//! 状态编码、版本掩码与位运算的**唯一来源是 `ldb_type`**：本模块不写任何
//! 状态字面量，也不手写掩码。`page_mode` 仅为 `ldb_buf` 保留的兼容别名
//! （常量转发，无独立取值），待其改为直接引 `ldb_type::PageStateMode` 后删除，见 CROSS.md。

use std::sync::atomic::Ordering;

use ldb_type::{
  PageState, PageStateMode, VERSION_MASK, next_version_new_state as next_ver,
  same_version_new_state as same_ver,
};

/// `ldb_buf` 兼容别名（上游 `page_state.h:13-18` 的五个编码）
///
/// 取值全部由 `ldb_type::PageStateMode` 派生，本模块不写字面量；待 `ldb_buf`
/// 改为直接引 `ldb_type::PageStateMode` 后整模块删除（CROSS.md）。
pub mod page_mode {
  use ldb_type::PageStateMode as LdbTypePageStateMode;

  /// 无持有者
  pub const UNLOCKED: u8 = LdbTypePageStateMode::UNLOCKED;
  /// 共享上限
  pub const MAX_SHARED: u8 = LdbTypePageStateMode::MAX_SHARED;
  /// 排他
  pub const EXCLUSIVE: u8 = LdbTypePageStateMode::EXCLUSIVE;
  /// Clock 标记
  pub const MARKED: u8 = LdbTypePageStateMode::MARKED;
  /// 已驱逐
  pub const EVICTED: u8 = LdbTypePageStateMode::EVICTED;
}

/// 初始状态字：版本 0 / EVICTED（上游 `page_state.cc:9-12`，编译期算出）
const EVICTED_INIT: u64 = same_ver(0, PageStateMode::EVICTED);

/// Clock 置换状态机扩展
pub trait PageStateExt {
  /// 初始化为 EVICTED
  fn init(&self);
  /// 调试字符串
  fn to_string(&self) -> String;
  /// 尝试共享上锁（MARKED 态可重入为单读者）
  fn try_lock_shared(&self, old: u64) -> bool;
  /// 共享解锁（读计数 -1；上游 `HybridLatchImpl<PageStateMode>::UnlockShared`）
  fn unlock_shared(&self);
  /// UNLOCKED → MARKED
  fn try_mark(&self, old: u64) -> bool;
  /// MARKED → UNLOCKED（输出新状态字）
  fn unmark(&self, old: u64, new_state: &mut u64) -> bool;
  /// MARKED → EXCLUSIVE
  fn unmark_and_lock_exclusive(&self, old: u64) -> bool;
  /// 排他解锁并驱逐
  fn unlock_exclusive_and_evict(&self);
  /// 排他解锁并标记
  fn unlock_exclusive_and_mark(&self);
}

impl PageStateExt for PageState {
  #[inline]
  fn init(&self) {
    self.store(EVICTED_INIT, Ordering::Release);
  }

  fn to_string(&self) -> String {
    let v = self.load(Ordering::Relaxed);
    format!("[State: {}, Version: {}]", v >> 56, v & VERSION_MASK)
  }

  #[inline]
  fn try_lock_shared(&self, old: u64) -> bool {
    let s = (old >> 56) as u8;
    // 上游 `page_state.cc:38-51`：共享码域 0..MAX_SHARED 直接 +1；
    // MARKED（驱逐者持有）以「1 读者」顶替，原标记者 CAS 必然失败后重试。
    if s < PageStateMode::MAX_SHARED {
      return self
        .cas(old, same_ver(old, s + 1), Ordering::AcqRel)
        .is_ok();
    }
    if s == PageStateMode::MARKED {
      return self.cas(old, same_ver(old, 1), Ordering::AcqRel).is_ok();
    }
    false
  }

  #[inline]
  fn unlock_shared(&self) {
    loop {
      let old = self.load(Ordering::Relaxed);
      debug_assert!(matches!(
        PageStateMode::from_state_word(old),
        PageStateMode::Shared(_)
      ));
      let s = (old >> 56) as u8;
      // 与上游 `old_state - 1` 的 u64 无符号回绕逐位一致（误用时高 8 位退化为 255）
      if self
        .cas(old, same_ver(old, s.wrapping_sub(1)), Ordering::AcqRel)
        .is_ok()
      {
        return;
      }
    }
  }

  #[inline]
  fn try_mark(&self, old: u64) -> bool {
    debug_assert_eq!((old >> 56) as u8, PageStateMode::UNLOCKED);
    self
      .cas(old, same_ver(old, PageStateMode::MARKED), Ordering::AcqRel)
      .is_ok()
  }

  #[inline]
  fn unmark(&self, old: u64, new_state: &mut u64) -> bool {
    debug_assert_eq!((old >> 56) as u8, PageStateMode::MARKED);
    *new_state = same_ver(old, PageStateMode::UNLOCKED);
    self.cas(old, *new_state, Ordering::AcqRel).is_ok()
  }

  #[inline]
  fn unmark_and_lock_exclusive(&self, old: u64) -> bool {
    debug_assert_eq!((old >> 56) as u8, PageStateMode::MARKED);
    self
      .cas(
        old,
        same_ver(old, PageStateMode::EXCLUSIVE),
        Ordering::AcqRel,
      )
      .is_ok()
  }

  #[inline]
  fn unlock_exclusive_and_evict(&self) {
    let cur = self.load(Ordering::Relaxed);
    debug_assert_eq!((cur >> 56) as u8, PageStateMode::EXCLUSIVE);
    self.store(next_ver(cur, PageStateMode::EVICTED), Ordering::Release);
  }

  #[inline]
  fn unlock_exclusive_and_mark(&self) {
    let cur = self.load(Ordering::Relaxed);
    debug_assert_eq!((cur >> 56) as u8, PageStateMode::EXCLUSIVE);
    self.store(next_ver(cur, PageStateMode::MARKED), Ordering::Release);
  }
}
