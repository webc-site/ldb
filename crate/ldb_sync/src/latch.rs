//! 混合闩：64 位原子字（高 8 位状态 / 低 56 位版本）。
//!
//! 上游 `hybrid_latch.cc` 的 `for(;;) AsmYield(counter)` 是**无界纯自旋**：`AsmYield`
//! 忽略 counter，只发一条 `pause`/`yield` 指令（`ZLeanStore/src/common/utils.cc:111-119`）。
//! compio 单线程每核模型下，若闩持有者是与自旋者同线程的协程，纯自旋会饿死
//! I/O 完成事件轮询（todo.md §3.4 协议），故阻塞式获取一律 `async`：先短自旋
//! 吸收快路径，再 `yield_now().await` 让出，等待者超限则 `Restart`。

use std::{
  hint::spin_loop,
  sync::atomic::{AtomicU64, Ordering},
};

use futures_lite::future::yield_now;
use ldb_type::{
  VERSION_MASK, next_version_new_state as next_ver, same_version_new_state as same_ver,
};

use crate::error::{Error, Result};

/// HybridLatch 专用状态编码（上游 `HybridLatchMode`：MAX_SHARED=254 / EXCLUSIVE=255，
/// 与 PageStateMode 的 252/253 不同）
pub mod latch_mode {
  /// 无持有者
  pub const UNLOCKED: u8 = 0;
  /// 共享上限
  pub const MAX_SHARED: u8 = 254;
  /// 排他
  pub const EXCLUSIVE: u8 = 255;
}

/// `spin_wait_page_state` 的让出上限：超出即返回 Restart，交上层重进乐观锁
const SPIN_YIELD_LIMIT: u32 = 64;

/// 纯自旋轮数上限：短临界区几条指令即释放，先自旋吸收快路径，省去每轮
/// 协程调度往返；之后每轮让出
const SPIN_BEFORE_YIELD: u32 = 16;

/// 阻塞式获取的退避：前 [`SPIN_BEFORE_YIELD`] 轮 `spin_loop`，之后每轮
/// `yield_now`（让 compio 调度同线程持锁协程与 I/O 完成事件推进）
#[inline]
async fn backoff(spins: &mut u32) {
  *spins += 1;
  if *spins <= SPIN_BEFORE_YIELD {
    spin_loop();
  } else {
    yield_now().await;
  }
}

/// 混合闩
#[derive(Debug, Default)]
pub struct HybridLatch {
  state_and_version: AtomicU64,
}

impl HybridLatch {
  /// 构造（UNLOCKED / 版本 0）
  #[must_use]
  #[inline]
  pub const fn new() -> Self {
    Self {
      state_and_version: AtomicU64::new(0),
    }
  }

  /// 保留旧版本、换新状态（上游 `SameVersionNewState`）
  #[must_use]
  #[inline]
  pub fn same_version_new_state(old: u64, new_state: u8) -> u64 {
    same_ver(old, new_state)
  }

  /// 版本 +1、换新状态（上游 `NextVersionNewState`）
  #[must_use]
  #[inline]
  pub fn next_version_new_state(old: u64, new_state: u8) -> u64 {
    next_ver(old, new_state)
  }

  /// 状态字高 8 位
  #[must_use]
  #[inline]
  pub fn lock_state(v: u64) -> u8 {
    (v >> 56) as u8
  }

  /// 状态字低 56 位
  #[must_use]
  #[inline]
  pub fn version(v: u64) -> u64 {
    v & VERSION_MASK
  }

  /// 原始状态字引用（供上层 CAS 组合操作）
  #[must_use]
  #[inline]
  pub fn state_and_version(&self) -> &AtomicU64 {
    &self.state_and_version
  }

  /// 当前是否排他
  #[must_use]
  #[inline]
  pub fn is_exclusively_latched(&self, v: u64) -> bool {
    Self::lock_state(v) == latch_mode::EXCLUSIVE
  }

  /// 当前是否共享
  #[must_use]
  #[inline]
  pub fn is_shared_latched(&self, v: u64) -> bool {
    let s = Self::lock_state(v);
    (latch_mode::UNLOCKED < s) && (s < latch_mode::EXCLUSIVE)
  }

  /// 尝试排他上锁（状态须为 UNLOCKED）
  ///
  /// 上游 `hybrid_latch.cc:43-45` 的条件为 `UNLOCKED < state <= EXCLUSIVE`；
  /// `EXCLUSIVE` 是 u8 上界，故与 `state != UNLOCKED` 等价。
  #[must_use]
  #[inline]
  pub fn try_lock_exclusive(&self, old: u64) -> bool {
    if Self::lock_state(old) != latch_mode::UNLOCKED {
      return false;
    }
    self
      .state_and_version
      .compare_exchange(
        old,
        same_ver(old, latch_mode::EXCLUSIVE),
        Ordering::AcqRel,
        Ordering::Relaxed,
      )
      .is_ok()
  }

  /// 尝试共享上锁（读计数 +1，达 MAX_SHARED 拒绝）
  #[must_use]
  #[inline]
  pub fn try_lock_shared(&self, old: u64) -> bool {
    let s = Self::lock_state(old);
    if s < latch_mode::MAX_SHARED {
      return self
        .state_and_version
        .compare_exchange(
          old,
          same_ver(old, s + 1),
          Ordering::AcqRel,
          Ordering::Relaxed,
        )
        .is_ok();
    }
    false
  }

  /// 排他解锁（版本 +1）
  #[inline]
  pub fn unlock_exclusive(&self) {
    let cur = self.state_and_version.load(Ordering::Relaxed);
    debug_assert!(self.is_exclusively_latched(cur));
    self
      .state_and_version
      .store(next_ver(cur, latch_mode::UNLOCKED), Ordering::Release);
  }

  /// 排他降级为单读者（版本 +1）
  #[inline]
  pub fn downgrade_lock(&self) {
    let cur = self.state_and_version.load(Ordering::Relaxed);
    debug_assert!(self.is_exclusively_latched(cur));
    // 1 = 单读者（上游 `hybrid_latch.cc:75` 注释「1 here means 1 reader」）
    self
      .state_and_version
      .store(next_ver(cur, 1), Ordering::Release);
  }

  /// 共享解锁（读计数 -1）
  #[inline]
  pub fn unlock_shared(&self) {
    loop {
      let old = self.state_and_version.load(Ordering::Relaxed);
      debug_assert!(self.is_shared_latched(old));
      let s = Self::lock_state(old);
      // 与上游 `old_state - 1` 的 u64 无符号回绕逐位一致（误用时高 8 位退化为 255）
      if self
        .state_and_version
        .compare_exchange_weak(
          old,
          same_ver(old, s.wrapping_sub(1)),
          Ordering::AcqRel,
          Ordering::Relaxed,
        )
        .is_ok()
      {
        return;
      }
    }
  }

  /// 单读者升级为排他（try 语义，上游 `UpgradeLock`）
  #[must_use]
  #[inline]
  pub fn upgrade_lock(&self, old: u64) -> bool {
    if Self::lock_state(old) != 1 {
      return false;
    }
    self
      .state_and_version
      .compare_exchange_weak(
        old,
        same_ver(old, latch_mode::EXCLUSIVE),
        Ordering::AcqRel,
        Ordering::Relaxed,
      )
      .is_ok()
  }

  /// 阻塞式排他获取（上游无界自旋 → 退避：先自旋后让出协程）
  pub async fn lock_exclusive(&self) {
    let mut spins = 0u32;
    loop {
      let old = self.state_and_version.load(Ordering::Relaxed);
      if self.try_lock_exclusive(old) {
        return;
      }
      backoff(&mut spins).await;
    }
  }

  /// 阻塞式共享获取
  pub async fn lock_shared(&self) {
    let mut spins = 0u32;
    loop {
      let old = self.state_and_version.load(Ordering::Relaxed);
      if self.try_lock_shared(old) {
        return;
      }
      backoff(&mut spins).await;
    }
  }

  /// 阻塞式单读者强制升级
  pub async fn force_upgrade_lock(&self) {
    let mut spins = 0u32;
    loop {
      let old = self.state_and_version.load(Ordering::Relaxed);
      if self.upgrade_lock(old) {
        return;
      }
      backoff(&mut spins).await;
    }
  }

  /// 等待任意持有者释放（状态归零）；让出式
  pub async fn wait_until_unlocked(&self) {
    let mut spins = 0u32;
    loop {
      let v = self.state_and_version.load(Ordering::Acquire);
      if Self::lock_state(v) == latch_mode::UNLOCKED {
        return;
      }
      backoff(&mut spins).await;
    }
  }

  /// 自旋等待状态字到达 `expect`（或超出让出上限返回 Restart）
  pub async fn spin_wait_page_state(&self, expect: u64) -> Result<()> {
    let mut spins = 0u32;
    loop {
      let v = self.state_and_version.load(Ordering::Acquire);
      if v == expect {
        return Ok(());
      }
      spins += 1;
      if spins > SPIN_YIELD_LIMIT {
        return Err(Error::Restart);
      }
      yield_now().await;
    }
  }
}
