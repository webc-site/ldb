//! 页守卫：乐观 / 共享 / 排他三形态（上游 `hybrid_guard.cc` + `page_guard/*.cc`）。
//!
//! 与上游差异：
//! - C++ 析构抛 `RestartException` 的路径全部改为显式 `validate_*` 返回 `Result`；
//! - 乐观升级的 `for(;;) AsmYield` 改 `yield_now().await`（todo.md §3.4 协议 2）；
//! - 页加载 `page_loader_func` 属 L2 buffer manager，L1 不可上依赖：`OptimisticGuard::init`
//!   在 EVICTED 时 fail-closed 返回 `Restart`，调用方须先完成缺页加载；
//! - GSN/WAL 相关成员（DetectGSNDependency 等）不在本 crate 固有 impl（孤儿规则），
//!   由 L3/L4 的事务层包装实现（见 `ldb_txn::gsn_sync`）。

use std::{
  fmt,
  hint::spin_loop,
  ptr::NonNull,
  sync::atomic::{Ordering, fence},
};

use futures_lite::future::yield_now;
use ldb_type::{PageId, PageState, PageStateMode};

use crate::{
  error::{Error, Result},
  latch::{HybridLatch, latch_mode},
  page_state::PageStateExt,
};

/// 同步上下文里 `optimistic_lock` 的自旋上限（上游无界 `AsmYield`；同步函数无法
/// 让出协程，超限返回 Restart 交调用方重试）
const OPTIMISTIC_SPIN_LIMIT: u32 = 64;

/// 阻塞式升级的让出上限（超出即 Restart：共享读者长期持有时不饿死，
/// 交上层重进乐观锁；上限取 [`OPTIMISTIC_SPIN_LIMIT`] 的 16 倍以宽限慢读者批量通过）
const UPGRADE_YIELD_LIMIT: u32 = OPTIMISTIC_SPIN_LIMIT * 16;

/// 守卫模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardMode {
  /// 乐观（版本快照，不阻塞他人）
  Optimistic,
  /// 共享
  Shared,
  /// 排他
  Exclusive,
  /// 已转移 / 已释放
  Moved,
}

/// 乐观 / 共享 / 排他通用守卫（上游 `HybridGuard`）
pub struct HybridGuard<'a> {
  latch: &'a HybridLatch,
  mode: GuardMode,
  state: u64,
}

impl fmt::Debug for HybridGuard<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("HybridGuard")
      .field("mode", &self.mode)
      .field("state", &self.state)
      .finish()
  }
}

impl<'a> HybridGuard<'a> {
  /// 乐观快照（非 EXCLUSIVE 即成功）
  ///
  /// 上游 `hybrid_guard.cc:42-51` 在 EXCLUSIVE 时 `for(;;) AsmYield()` 无界自旋；
  /// 本函数同步、不能让出协程，故自旋 [`OPTIMISTIC_SPIN_LIMIT`] 次后返回
  /// `Err(Restart)` 交上层重试（不修改闩状态）。
  pub fn optimistic_lock(latch: &'a HybridLatch) -> Result<Self> {
    let mut spins = 0u32;
    loop {
      let state = latch.state_and_version().load(Ordering::Acquire);
      if HybridLatch::lock_state(state) != latch_mode::EXCLUSIVE {
        return Ok(Self {
          latch,
          mode: GuardMode::Optimistic,
          state,
        });
      }
      spins += 1;
      if spins > OPTIMISTIC_SPIN_LIMIT {
        return Err(Error::Restart);
      }
      spin_loop();
    }
  }

  /// 共享获取（阻塞式 async）
  pub async fn lock_shared(latch: &'a HybridLatch) -> Self {
    latch.lock_shared().await;
    Self {
      latch,
      mode: GuardMode::Shared,
      state: 0,
    }
  }

  /// 排他获取（阻塞式 async）
  pub async fn lock_exclusive(latch: &'a HybridLatch) -> Self {
    latch.lock_exclusive().await;
    Self {
      latch,
      mode: GuardMode::Exclusive,
      state: 0,
    }
  }

  /// 校验乐观快照（版本不变且未被排他写）
  ///
  /// `fence(Acquire)` 必须在本次状态字读取之前：单个 Acquire/SeqCst load 只约束其
  /// 后的访问，不阻止其前的页数据读被重排到校验之后（CROSS.md → ldb_sync round12
  /// 在 Apple M1 上实测 stale-accept）。
  pub fn validate_optimistic_lock(&mut self) -> Result<()> {
    if self.mode != GuardMode::Optimistic {
      return Ok(());
    }
    self.mode = GuardMode::Moved;
    fence(Ordering::Acquire);
    let latest = self.latch.state_and_version().load(Ordering::Acquire);
    if HybridLatch::lock_state(latest) == latch_mode::EXCLUSIVE
      || HybridLatch::version(latest) != HybridLatch::version(self.state)
    {
      return Err(Error::Restart);
    }
    Ok(())
  }

  /// 乐观 → 排他（阻塞式：版本漂移即 Restart，上游 for(;;) AsmYield；
  /// 让出超限同样 Restart 防饿死）
  pub async fn upgrade_optimistic_to_exclusive(&mut self) -> Result<()> {
    let mut spins = 0u32;
    loop {
      let old = self.latch.state_and_version().load(Ordering::Acquire);
      if HybridLatch::version(old) != HybridLatch::version(self.state) {
        return Err(Error::Restart);
      }
      if self.latch.try_lock_exclusive(old) {
        self.mode = GuardMode::Exclusive;
        return Ok(());
      }
      spins += 1;
      if spins > UPGRADE_YIELD_LIMIT {
        return Err(Error::Restart);
      }
      yield_now().await;
    }
  }

  /// 解锁（共享/排他）；乐观模式无操作
  #[inline]
  pub fn unlock(&mut self) {
    match self.mode {
      GuardMode::Shared => self.latch.unlock_shared(),
      GuardMode::Exclusive => self.latch.unlock_exclusive(),
      GuardMode::Optimistic | GuardMode::Moved => {}
    }
    self.mode = GuardMode::Moved;
  }

  /// 当前模式
  #[must_use]
  #[inline]
  pub fn mode(&self) -> GuardMode {
    self.mode
  }
}

impl Drop for HybridGuard<'_> {
  fn drop(&mut self) {
    self.unlock();
  }
}

/// 页守卫基座（上游 `PageGuard<PageClass>`）
///
/// `'a` 绑定页状态的生命期：守卫不能比 [`PageState`] 活得久。
pub struct PageGuard<'a, P> {
  pid: PageId,
  ptr: NonNull<P>,
  ps: &'a PageState,
  mode: GuardMode,
}

// SAFETY: 守卫只保存 `NonNull<P>` 与 `&PageState`，不拥有页内存；`P: Send` 是
// 「把页的共享/独占使用权移到另一线程」的同一约束（与 `&mut P: Send` 同界）。闩是
// 原子字、无线程亲和：任意线程的 CAS/Release store 都合法，跨线程 drop 的
// happens-before 由调用方的线程传递（send/join）建立，不依赖释放线程身份。
// `&PageState` 为 Send（PageState: Sync）。页指针的有效期由上层 buffer manager 的
// 帧所有权保证（`'a` 只绑定 PageState，见 CROSS：裸指针 API 契约）。
// 未实现 Sync：同一守卫被多线程并发解锁（尤其排他 store 两次）会破坏协议。
unsafe impl<P: Send> Send for PageGuard<'_, P> {}

impl<'a, P> PageGuard<'a, P> {
  /// 页 ID
  #[must_use]
  #[inline]
  pub fn page_id(&self) -> PageId {
    self.pid
  }

  /// 页数据指针（`Moved` 后不可再访问）
  #[must_use]
  #[inline]
  pub fn ptr(&self) -> *mut P {
    debug_assert_ne!(self.mode, GuardMode::Moved);
    self.ptr.as_ptr()
  }

  /// 模式
  #[must_use]
  #[inline]
  pub fn mode(&self) -> GuardMode {
    self.mode
  }

  /// 页状态引用
  #[must_use]
  #[inline]
  pub fn page_state(&self) -> &'a PageState {
    self.ps
  }

  /// 从 parts 构造（仅 crate 内部与上层 buffer manager 使用）
  #[inline]
  pub(crate) fn from_parts(
    pid: PageId,
    ptr: NonNull<P>,
    ps: &'a PageState,
    mode: GuardMode,
  ) -> Self {
    Self { pid, ptr, ps, mode }
  }
}

impl<P> fmt::Debug for PageGuard<'_, P> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("PageGuard")
      .field("pid", &self.pid)
      .field("mode", &self.mode)
      .finish_non_exhaustive()
  }
}

/// 乐观页守卫（上游 `OptimisticGuard`）
pub struct OptimisticGuard<'a, P> {
  guard: PageGuard<'a, P>,
  /// 完整状态字快照（高 8 位状态 + 低 56 位版本）
  state: u64,
}

impl<P> fmt::Debug for OptimisticGuard<'_, P> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("OptimisticGuard")
      .field("guard", &self.guard)
      .field("state", &self.state)
      .finish()
  }
}

impl<'a, P> OptimisticGuard<'a, P> {
  /// 从常驻页取乐观快照（上游 `optimistic_guard.cc:68-96` 的 `Init`）
  ///
  /// - `UNLOCKED` / `Shared(n)`：记录完整状态字；
  /// - `MARKED`：`Unmark` 成功则记录 UNLOCKED 状态字（页常驻，无 I/O）；
  /// - `EXCLUSIVE`：上游为 `for(;;) AsmYield(cnt)` 无界自旋（`optimistic_guard.cc:70-95`）；
  ///   本函数是同步 API、不能让出协程（todo.md §3.4 协议 2），故自旋
  ///   [`OPTIMISTIC_SPIN_LIMIT`] 次后返回 `Err(Restart)`，且不修改状态字；
  ///   调用方必须按 `Restart` 重试（与 `validate_or_restart` 的错误路径同一处理）；
  /// - `EVICTED`：fail-closed 返回 `Restart` —— 缺页加载属 L2 buffer manager。
  pub fn init(pid: PageId, ptr: NonNull<P>, ps: &'a PageState) -> Result<Self> {
    let mut spins = 0u32;
    loop {
      let v = ps.load(Ordering::Acquire);
      match PageStateMode::from_state_word(v) {
        PageStateMode::Unlocked | PageStateMode::Shared(_) => {
          return Ok(Self {
            guard: PageGuard::from_parts(pid, ptr, ps, GuardMode::Optimistic),
            state: v,
          });
        }
        PageStateMode::Marked => {
          let mut new_state = 0;
          if ps.unmark(v, &mut new_state) {
            return Ok(Self {
              guard: PageGuard::from_parts(pid, ptr, ps, GuardMode::Optimistic),
              state: new_state,
            });
          }
        }
        PageStateMode::Exclusive => {}
        PageStateMode::Evicted => return Err(Error::Restart),
      }
      spins += 1;
      if spins > OPTIMISTIC_SPIN_LIMIT {
        return Err(Error::Restart);
      }
      spin_loop();
    }
  }

  /// 校验乐观快照（上游 `optimistic_guard.cc:99-137` 的 `ValidateOrRestart`）
  ///
  /// 与上游逐状态对齐：状态字完全不变 → 通过；版本相同且状态为
  /// `UNLOCKED`/`Shared(n)` → 通过；状态为 `MARKED` → `Unmark` 成功后通过；
  /// 其余（`EXCLUSIVE`/`EVICTED`/版本漂移）→ `Restart`。
  pub fn validate_or_restart(&mut self) -> Result<()> {
    if self.guard.mode == GuardMode::Moved {
      return Ok(());
    }
    debug_assert_eq!(self.guard.mode, GuardMode::Optimistic);
    self.guard.mode = GuardMode::Moved;
    // 页数据读与本次校验读之间必须有 LoadLoad 屏障：单个 Acquire/SeqCst load 不能
    // 阻止其前的数据读被重排到校验之后（CROSS.md → ldb_sync round12 实测）。
    fence(Ordering::Acquire);
    let ps = self.guard.page_state();
    let latest = ps.load(Ordering::Acquire);
    if latest == self.state {
      return Ok(());
    }
    if HybridLatch::version(latest) == HybridLatch::version(self.state) {
      match PageStateMode::from_state_word(latest) {
        PageStateMode::Unlocked | PageStateMode::Shared(_) => return Ok(()),
        PageStateMode::Marked => {
          let mut new_state = 0;
          if ps.unmark(latest, &mut new_state) {
            return Ok(());
          }
        }
        PageStateMode::Exclusive | PageStateMode::Evicted => {}
      }
    }
    Err(Error::Restart)
  }

  /// 内部守卫引用
  #[must_use]
  #[inline]
  pub fn inner(&self) -> &PageGuard<'a, P> {
    &self.guard
  }

  /// 消费并升级为排他（上游 `ExclusiveGuard(OptimisticGuard&&)`；阻塞式 async；
  /// 让出超限 Restart 防饿死）
  pub async fn into_exclusive(self) -> Result<ExclusiveGuard<'a, P>> {
    if self.guard.mode != GuardMode::Optimistic {
      return Err(Error::Restart);
    }
    let ps = self.guard.page_state();
    let mut spins = 0u32;
    loop {
      let old = ps.load(Ordering::Acquire);
      if HybridLatch::version(old) != HybridLatch::version(self.state) {
        return Err(Error::Restart);
      }
      let state = (old >> 56) as u8;
      if (state == PageStateMode::UNLOCKED || state == PageStateMode::MARKED)
        && ps
          .cas(
            old,
            HybridLatch::same_version_new_state(old, PageStateMode::EXCLUSIVE),
            Ordering::AcqRel,
          )
          .is_ok()
      {
        let mut guard = self.guard;
        guard.mode = GuardMode::Exclusive;
        return Ok(ExclusiveGuard { guard });
      }
      spins += 1;
      if spins > UPGRADE_YIELD_LIMIT {
        return Err(Error::Restart);
      }
      yield_now().await;
    }
  }

  /// 消费并降级为共享（上游 `SharedGuard(OptimisticGuard&&)`：
  /// `TryLockShared(old_state)`，失败即 Restart）
  pub fn into_shared(self) -> Result<SharedGuard<'a, P>> {
    if self.guard.mode != GuardMode::Optimistic {
      return Err(Error::Restart);
    }
    let ps = self.guard.page_state();
    if !ps.try_lock_shared(self.state) {
      return Err(Error::Restart);
    }
    let mut guard = self.guard;
    guard.mode = GuardMode::Shared;
    Ok(SharedGuard { guard })
  }
}

/// 排他页守卫（上游 `ExclusiveGuard`）
pub struct ExclusiveGuard<'a, P> {
  guard: PageGuard<'a, P>,
}

impl<P> fmt::Debug for ExclusiveGuard<'_, P> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct(stringify!(ExclusiveGuard))
      .field("guard", &self.guard)
      .finish()
  }
}

impl<P> ExclusiveGuard<'_, P> {
  /// 解锁（版本 +1 回 UNLOCKED）
  pub fn unlock(mut self) {
    self.unlock_inplace();
  }

  /// 就地解锁（供 Drop 复用）
  fn unlock_inplace(&mut self) {
    if self.guard.mode == GuardMode::Exclusive {
      let ps = self.guard.page_state();
      let cur = ps.load(Ordering::Relaxed);
      debug_assert_eq!((cur >> 56) as u8, PageStateMode::EXCLUSIVE);
      ps.store(
        HybridLatch::next_version_new_state(cur, PageStateMode::UNLOCKED),
        Ordering::Release,
      );
      self.guard.mode = GuardMode::Moved;
    }
  }

  /// 解锁并交回页指针（写回路径）
  pub fn unlock_and_get_ptr(mut self) -> *mut P {
    let p = self.guard.ptr();
    self.unlock_inplace();
    p
  }

  /// 内部守卫引用
  #[must_use]
  #[inline]
  pub fn inner(&self) -> &PageGuard<'_, P> {
    &self.guard
  }
}

impl<P> Drop for ExclusiveGuard<'_, P> {
  fn drop(&mut self) {
    self.unlock_inplace();
  }
}

/// 共享页守卫（上游 `SharedGuard`）
pub struct SharedGuard<'a, P> {
  guard: PageGuard<'a, P>,
}

impl<P> fmt::Debug for SharedGuard<'_, P> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct(stringify!(SharedGuard))
      .field("guard", &self.guard)
      .finish()
  }
}

impl<P> SharedGuard<'_, P> {
  /// 解锁
  pub fn unlock(mut self) {
    self.unlock_inplace();
  }

  fn unlock_inplace(&mut self) {
    if self.guard.mode == GuardMode::Shared {
      self.guard.page_state().unlock_shared();
      self.guard.mode = GuardMode::Moved;
    }
  }

  /// 内部守卫引用
  #[must_use]
  #[inline]
  pub fn inner(&self) -> &PageGuard<'_, P> {
    &self.guard
  }
}

impl<P> Drop for SharedGuard<'_, P> {
  fn drop(&mut self) {
    self.unlock_inplace();
  }
}
