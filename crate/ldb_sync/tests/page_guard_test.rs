//! 页守卫三形态（上游 `page_guard/{page_guard,optimistic_guard,shared_guard,exclusive_guard}.cc`）。

use std::{
  fmt,
  ptr::{self, NonNull},
  sync::{
    Barrier,
    atomic::{AtomicU64, Ordering},
  },
  thread,
};

use futures_lite::future::block_on;
use ldb_sync::{Error, GuardMode, OptimisticGuard, PageStateExt};
use ldb_type::PageState;

/// 低 56 位版本掩码（测试侧独立字面量）
const VERSION_MASK: u64 = 0x00FF_FFFF_FFFF_FFFF;
/// 页状态编码（与上游 page_state.h 对应，测试侧独立字面量）
const UNLOCKED: u8 = 0;
const EXCLUSIVE: u8 = 253;
const MARKED: u8 = 254;
const EVICTED: u8 = 255;

#[derive(Debug)]
struct Page {
  value: AtomicU64,
}

fn must<T, E: fmt::Debug>(r: Result<T, E>, what: &str) -> T {
  match r {
    Ok(v) => v,
    Err(e) => panic!("{what}: {e:?}"),
  }
}

/// 编译期钉死 `PageGuard: Send`（跨线程 drop 用例依赖它；`Sync` 有意未实现）
fn require_send<T: Send>(_: &T) {}

fn state_of(ps: &PageState) -> u8 {
  (ps.load(Ordering::Relaxed) >> 56) as u8
}

fn ver_of(ps: &PageState) -> u64 {
  ps.load(Ordering::Relaxed) & VERSION_MASK
}

fn exclusive_word(v: u64) -> u64 {
  (v & VERSION_MASK) | (u64::from(EXCLUSIVE) << 56)
}

fn next_word(v: u64, state: u8) -> u64 {
  ((v & VERSION_MASK) + 1) | (u64::from(state) << 56)
}

#[test]
fn guard_accessors_and_ptr() {
  let ps = PageState::new();
  let mut page = Page {
    value: AtomicU64::new(7),
  };
  let ptr = NonNull::from(&mut page);
  let g = must(OptimisticGuard::init(42, ptr, &ps), "常驻页乐观锁");
  assert_eq!(g.inner().page_id(), 42);
  assert_eq!(g.inner().mode(), GuardMode::Optimistic);
  assert_eq!(g.inner().ptr(), ptr.as_ptr());
  assert!(ptr::eq(g.inner().page_state(), &ps));
  assert_eq!(state_of(&ps), UNLOCKED);
  assert_eq!(ver_of(&ps), 0);
}

#[test]
fn optimistic_init_rejects_evicted() {
  let ps = PageState::new();
  ps.init();
  assert_eq!(state_of(&ps), EVICTED);
  let mut page = Page {
    value: AtomicU64::new(0),
  };
  assert!(matches!(
    OptimisticGuard::init(1, NonNull::from(&mut page), &ps),
    Err(Error::Restart)
  ));
}

#[test]
fn optimistic_init_rejects_exclusive() {
  let ps = PageState::new();
  let v = ps.load(Ordering::Relaxed);
  assert!(ps.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
  let mut page = Page {
    value: AtomicU64::new(0),
  };
  assert!(matches!(
    OptimisticGuard::init(1, NonNull::from(&mut page), &ps),
    Err(Error::Restart)
  ));
  // 有界自旋超限返回错误，且不得改动持有者的状态字（上游此分支为无界自旋：
  // `optimistic_guard.cc:70-95` 的 `for(;;) AsmYield(cnt)`）。
  assert_eq!(state_of(&ps), EXCLUSIVE, "超限失败不得改动闩");
  assert_eq!(ver_of(&ps), 0);
  // 写者释放后重试可成功并记录新版本（版本 0 -> 1），说明 Restart 是可恢复的。
  let held = ps.load(Ordering::Relaxed);
  ps.store(next_word(held, UNLOCKED), Ordering::Release);
  let g = must(
    OptimisticGuard::init(1, NonNull::from(&mut page), &ps),
    "写者释放后重试",
  );
  assert_eq!(ver_of(&ps), 1);
  assert_eq!(g.inner().mode(), GuardMode::Optimistic);
}

#[test]
fn optimistic_init_unmarks_marked() {
  let ps = PageState::new();
  assert!(ps.try_mark(0));
  assert_eq!(state_of(&ps), MARKED);
  let mut page = Page {
    value: AtomicU64::new(0),
  };
  let g = must(
    OptimisticGuard::init(1, NonNull::from(&mut page), &ps),
    "MARKED 可解标",
  );
  assert_eq!(state_of(&ps), UNLOCKED, "Init 必须 Unmark");
  assert_eq!(ver_of(&ps), 0);
  assert_eq!(g.inner().mode(), GuardMode::Optimistic);
}

/// 同版本下的状态位校验：EXCLUSIVE（活跃写者）→ Restart；MARKED → 解标后 Ok。
/// 只比较版本会接受并发写者正在修改的页（核心 OLC 不变量）。
#[test]
fn optimistic_validation_checks_state_bits() {
  // 活跃写者持 EXCLUSIVE，版本不变 -> 必须 Restart
  let ps = PageState::new();
  let mut page = Page {
    value: AtomicU64::new(7),
  };
  let mut g = must(
    OptimisticGuard::init(1, NonNull::from(&mut page), &ps),
    "乐观锁",
  );
  let v = ps.load(Ordering::Relaxed);
  assert!(ps.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
  assert_eq!(ver_of(&ps), 0, "排他上锁不推版本");
  assert!(
    matches!(g.validate_or_restart(), Err(Error::Restart)),
    "并发写者临界区内必须 Restart"
  );
  // 写者解锁（版本 +1）后可重新取快照
  let held = ps.load(Ordering::Relaxed);
  ps.store(next_word(held, UNLOCKED), Ordering::Release);

  // MARKED 未驱逐 -> 解标后 Ok，且状态回到 UNLOCKED
  let ps2 = PageState::new();
  let mut page2 = Page {
    value: AtomicU64::new(8),
  };
  let mut g2 = must(
    OptimisticGuard::init(2, NonNull::from(&mut page2), &ps2),
    "乐观锁",
  );
  assert!(ps2.try_mark(0));
  assert!(
    g2.validate_or_restart().is_ok(),
    "MARKED 未驱逐，校验须解标后通过"
  );
  assert_eq!(state_of(&ps2), UNLOCKED, "校验须清掉 MARKED");
}

#[test]
fn optimistic_validation_accepts_shared_cycle() {
  // 无写者：读页数据后校验必须通过
  let ps0 = PageState::new();
  let mut page0 = Page {
    value: AtomicU64::new(7),
  };
  let mut g0 = must(
    OptimisticGuard::init(1, NonNull::from(&mut page0), &ps0),
    "乐观锁",
  );
  assert_eq!(page0.value.load(Ordering::Acquire), 7);
  assert!(g0.validate_or_restart().is_ok(), "无写者必须通过");

  // 同版本共享持有不改变页内容 -> 通过
  let ps = PageState::new();
  let mut page = Page {
    value: AtomicU64::new(7),
  };
  let mut g = must(
    OptimisticGuard::init(2, NonNull::from(&mut page), &ps),
    "乐观锁",
  );
  assert!(ps.try_lock_shared(0));
  assert!(
    g.validate_or_restart().is_ok(),
    "同版本共享持有不改变页内容"
  );
  ps.unlock_shared();
  assert_eq!(state_of(&ps), UNLOCKED);
}

#[test]
fn optimistic_validation_rejects_version_bump() {
  // (a) 同线程确定性交错：读者先读页数据 -> 写者取 EXCLUSIVE 改数据 -> 解锁推版本 -> 校验
  let ps = PageState::new();
  let mut page = Page {
    value: AtomicU64::new(7),
  };
  let mut g = must(
    OptimisticGuard::init(1, NonNull::from(&mut page), &ps),
    "乐观锁",
  );
  assert_eq!(page.value.load(Ordering::Acquire), 7, "乐观读者先读页数据");
  let v = ps.load(Ordering::Relaxed);
  assert!(ps.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
  page.value.store(8, Ordering::Relaxed);
  let held = ps.load(Ordering::Relaxed);
  ps.store(next_word(held, UNLOCKED), Ordering::Release);
  assert_eq!(page.value.load(Ordering::Relaxed), 8);
  assert!(
    matches!(g.validate_or_restart(), Err(Error::Restart)),
    "写者改数据并推版本后必须 Restart"
  );

  // (b) 只改版本不改数据：版本漂移本身即必须 Restart
  let ps = PageState::new();
  let mut page = Page {
    value: AtomicU64::new(7),
  };
  let mut g = must(
    OptimisticGuard::init(2, NonNull::from(&mut page), &ps),
    "乐观锁",
  );
  assert_eq!(page.value.load(Ordering::Relaxed), 7);
  let v = ps.load(Ordering::Relaxed);
  assert!(ps.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
  let held = ps.load(Ordering::Relaxed);
  ps.store(next_word(held, UNLOCKED), Ordering::Release);
  assert_eq!(page.value.load(Ordering::Relaxed), 7, "写者未改页数据");
  assert!(
    matches!(g.validate_or_restart(), Err(Error::Restart)),
    "只推进版本也必须 Restart"
  );

  // (c) 两线程 + 屏障的确定性交错：读者完成数据读 -> 写者改数据并解锁 -> 读者校验
  let ps = PageState::new();
  let page = Page {
    value: AtomicU64::new(7),
  };
  let barrier = Barrier::new(2);
  let mut g = must(
    OptimisticGuard::init(3, NonNull::from(&page), &ps),
    "乐观锁",
  );
  assert_eq!(page.value.load(Ordering::Acquire), 7, "读者先读");
  thread::scope(|scope| {
    scope.spawn(|| {
      barrier.wait(); // 等读者完成数据读
      let v = ps.load(Ordering::Acquire);
      assert!(ps.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
      page.value.store(99, Ordering::Relaxed);
      let held = ps.load(Ordering::Relaxed);
      ps.store(next_word(held, UNLOCKED), Ordering::Release);
      barrier.wait(); // 通知读者写者已完成
    });
    barrier.wait(); // 读者已读完
    barrier.wait(); // 等写者完成
  });
  assert_eq!(page.value.load(Ordering::Relaxed), 99);
  assert_eq!(state_of(&ps), UNLOCKED);
  assert!(
    matches!(g.validate_or_restart(), Err(Error::Restart)),
    "跨线程写者完成后必须 Restart"
  );
}

#[test]
fn optimistic_validation_is_idempotent_after_moved() {
  let ps = PageState::new();
  let mut page = Page {
    value: AtomicU64::new(7),
  };
  let mut g = must(
    OptimisticGuard::init(1, NonNull::from(&mut page), &ps),
    "乐观锁",
  );
  assert!(g.validate_or_restart().is_ok());
  // 已 MOVED：上游直接返回，不重复校验
  let v = ps.load(Ordering::Relaxed);
  assert!(ps.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
  assert!(g.validate_or_restart().is_ok());
}

/// 用 `futures_lite::future::block_on` 而不是 compio Runtime：无争用路径不进入
/// `yield_now`，普通 executor 足够；Miri 可执行（compio 的 kqueue 驱动在 Miri 下不可用）。
#[test]
fn exclusive_guard_unlock_paths() {
  block_on(async {
    // 1. 显式 unlock
    let ps = PageState::new();
    let mut page = Page {
      value: AtomicU64::new(7),
    };
    let g = must(
      OptimisticGuard::init(9, NonNull::from(&mut page), &ps),
      "乐观锁",
    );
    let e = must(g.into_exclusive().await, "UNLOCKED 可升级");
    assert!(ptr::eq(e.inner().page_state(), &ps));
    assert_eq!(e.inner().mode(), GuardMode::Exclusive);
    assert_eq!(state_of(&ps), EXCLUSIVE);
    assert_eq!(ver_of(&ps), 0);
    e.unlock();
    assert_eq!(state_of(&ps), UNLOCKED);
    assert_eq!(ver_of(&ps), 1);

    // 2. Drop 自动解锁
    let ps2 = PageState::new();
    let mut page2 = Page {
      value: AtomicU64::new(8),
    };
    let g2 = must(
      OptimisticGuard::init(3, NonNull::from(&mut page2), &ps2),
      "乐观锁",
    );
    let e2 = must(g2.into_exclusive().await, "UNLOCKED 可升级");
    drop(e2);
    assert_eq!(state_of(&ps2), UNLOCKED);
    assert_eq!(ver_of(&ps2), 1);

    // 3. unlock_and_get_ptr
    let ps3 = PageState::new();
    let mut page3 = Page {
      value: AtomicU64::new(9),
    };
    let ptr3 = NonNull::from(&mut page3);
    let g3 = must(OptimisticGuard::init(8, ptr3, &ps3), "乐观锁");
    let e3 = must(g3.into_exclusive().await, "UNLOCKED 可升级");
    let p = e3.unlock_and_get_ptr();
    assert_eq!(p, ptr3.as_ptr());
    assert_eq!(state_of(&ps3), UNLOCKED);
    assert_eq!(ver_of(&ps3), 1);
    // SAFETY: 排他守卫已释放，页内存由本测试的 `page3` 独占，指针来自其 `&mut`。
    assert_eq!(unsafe { (*p).value.load(Ordering::Relaxed) }, 9);

    // 4. 跨线程 drop：PageGuard 仅 Send（非 Sync）；闩是原子字、无线程亲和，
    //    移动后由任意线程 Release store 均合法（写者数据经 send 的 happens-before 发布）。
    let ps4 = PageState::new();
    let mut page4 = Page {
      value: AtomicU64::new(10),
    };
    let g4 = must(
      OptimisticGuard::init(12, NonNull::from(&mut page4), &ps4),
      "乐观锁",
    );
    let e4 = must(g4.into_exclusive().await, "UNLOCKED 可升级");
    require_send(&e4);
    thread::scope(|scope| {
      scope.spawn(move || drop(e4));
    });
    assert_eq!(state_of(&ps4), UNLOCKED, "跨线程 drop 必须释放闩");
    assert_eq!(ver_of(&ps4), 1, "跨线程 drop 同样推进版本");

    // 5. 快照后页被 Clock 标记（同版本 MARKED）：上游允许直接 X 抢占
    //    （exclusive_guard.cc:31-33 允许 MARKED，PageState 的 TryLockExclusive 对 254 放行）
    let ps5 = PageState::new();
    let mut page5 = Page {
      value: AtomicU64::new(11),
    };
    let g5 = must(
      OptimisticGuard::init(13, NonNull::from(&mut page5), &ps5),
      "乐观锁",
    );
    assert!(ps5.try_mark(0));
    let e5 = must(g5.into_exclusive().await, "MARKED 同版本可升级");
    assert_eq!(state_of(&ps5), EXCLUSIVE);
    e5.unlock();
    assert_eq!(state_of(&ps5), UNLOCKED);
    assert_eq!(ver_of(&ps5), 1);
  });
}

#[test]
fn into_exclusive_rejects_shared() {
  block_on(async {
    // 共享态：Init 必须接受（上游 `optimistic_guard.cc:90-93` default 分支），
    // 但升级为排他必须失败（`exclusive_guard.cc:31-33` 只接受 UNLOCKED/MARKED）
    let ps = PageState::new();
    assert!(ps.try_lock_shared(0));
    let mut page = Page {
      value: AtomicU64::new(7),
    };
    let g = must(
      OptimisticGuard::init(4, NonNull::from(&mut page), &ps),
      "乐观锁",
    );
    assert_eq!(g.inner().mode(), GuardMode::Optimistic);
    let mut g2 = must(
      OptimisticGuard::init(4, NonNull::from(&mut page), &ps),
      "共享态可乐观读",
    );
    assert!(g2.validate_or_restart().is_ok(), "共享态同版本校验必须通过");
    assert!(matches!(g.into_exclusive().await, Err(Error::Restart)));
    ps.unlock_shared();
    assert_eq!(state_of(&ps), UNLOCKED);

    // 版本漂移（写者已完成）后不得升级：CAS 以快照完整字为期望值，第二轮不得读取
    // 当前 UNLOCKED 字绕过版本校验
    let ps2 = PageState::new();
    let mut page2 = Page {
      value: AtomicU64::new(8),
    };
    let g3 = must(
      OptimisticGuard::init(5, NonNull::from(&mut page2), &ps2),
      "乐观锁",
    );
    let v = ps2.load(Ordering::Relaxed);
    assert!(ps2.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
    let held = ps2.load(Ordering::Relaxed);
    ps2.store(next_word(held, UNLOCKED), Ordering::Release);
    assert!(
      matches!(g3.into_exclusive().await, Err(Error::Restart)),
      "版本漂移后不得升级为排他"
    );
    assert_eq!(state_of(&ps2), UNLOCKED);

    // into_shared 同样以快照完整字做 TryLockShared，不得读当前字绕过版本
    let ps3 = PageState::new();
    let mut page3 = Page {
      value: AtomicU64::new(9),
    };
    let g4 = must(
      OptimisticGuard::init(6, NonNull::from(&mut page3), &ps3),
      "乐观锁",
    );
    let v = ps3.load(Ordering::Relaxed);
    assert!(ps3.cas(v, exclusive_word(v), Ordering::AcqRel).is_ok());
    let held = ps3.load(Ordering::Relaxed);
    ps3.store(next_word(held, UNLOCKED), Ordering::Release);
    assert!(
      matches!(g4.into_shared(), Err(Error::Restart)),
      "版本漂移后不得转共享"
    );
    assert_eq!(state_of(&ps3), UNLOCKED);
  });
}

#[test]
fn shared_guard_unlock_and_drop() {
  // 显式 unlock
  let ps = PageState::new();
  let mut page = Page {
    value: AtomicU64::new(7),
  };
  let g = must(
    OptimisticGuard::init(5, NonNull::from(&mut page), &ps),
    "乐观锁",
  );
  let s = must(g.into_shared(), "UNLOCKED 可转共享");
  assert!(ptr::eq(s.inner().page_state(), &ps));
  assert_eq!(s.inner().mode(), GuardMode::Shared);
  assert_eq!(state_of(&ps), 1);
  s.unlock();
  assert_eq!(state_of(&ps), 0);
  assert_eq!(ver_of(&ps), 0, "共享解锁不推版本");

  // Drop 自动解锁
  let ps2 = PageState::new();
  let mut page2 = Page {
    value: AtomicU64::new(8),
  };
  let g2 = must(
    OptimisticGuard::init(6, NonNull::from(&mut page2), &ps2),
    "乐观锁",
  );
  let s2 = must(g2.into_shared(), "UNLOCKED 可转共享");
  assert_eq!(state_of(&ps2), 1);
  drop(s2);
  assert_eq!(state_of(&ps2), 0);
  assert_eq!(ver_of(&ps2), 0);

  // 快照后页被 Clock 标记：上游 `SharedGuard(OptimisticGuard&&)` 用**快照旧字**
  // 调 `TryLockShared`（shared_guard.cc:23），旧字是 UNLOCKED，CAS 对 MARKED 失败
  // ⇒ Restart（page_state.cc:45-47 的 MARKED→1 分支只对「当前字」调用生效）。
  let ps3 = PageState::new();
  let mut page3 = Page {
    value: AtomicU64::new(9),
  };
  let g3 = must(
    OptimisticGuard::init(7, NonNull::from(&mut page3), &ps3),
    "乐观锁",
  );
  assert!(ps3.try_mark(0));
  assert!(
    matches!(g3.into_shared(), Err(Error::Restart)),
    "快照后变 MARKED，旧字 CAS 必须失败"
  );
  assert_eq!(state_of(&ps3), MARKED, "失败不得改动状态字");
}

#[test]
fn into_after_validate_is_rejected() {
  block_on(async {
    // into_exclusive：已校验(MOVED)的守卫不得再升级
    let ps = PageState::new();
    let mut page = Page {
      value: AtomicU64::new(7),
    };
    let mut g = must(
      OptimisticGuard::init(2, NonNull::from(&mut page), &ps),
      "乐观锁",
    );
    assert!(g.validate_or_restart().is_ok());
    assert!(matches!(g.into_exclusive().await, Err(Error::Restart)));
    assert_eq!(state_of(&ps), UNLOCKED);

    // into_shared：同上
    let ps2 = PageState::new();
    let mut page2 = Page {
      value: AtomicU64::new(8),
    };
    let mut g2 = must(
      OptimisticGuard::init(2, NonNull::from(&mut page2), &ps2),
      "乐观锁",
    );
    assert!(g2.validate_or_restart().is_ok());
    assert!(matches!(g2.into_shared(), Err(Error::Restart)));
    assert_eq!(state_of(&ps2), UNLOCKED);
  });
}
