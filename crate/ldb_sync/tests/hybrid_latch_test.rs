//! HybridLatch / HybridGuard 状态机与 OLC 校验回归（上游 `hybrid_latch.cc` + `hybrid_guard.cc`）。

use std::{
  fmt::Debug,
  hint::{black_box, spin_loop},
  sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
  },
  thread,
  time::Duration,
};

use futures_lite::future::block_on;
use ldb_sync::{Error, GuardMode, HybridGuard, HybridLatch};

/// 状态字（高 8 位状态 / 低 56 位版本），测试侧独立计算，不复用生产代码
fn sv(l: &HybridLatch) -> u64 {
  l.state_and_version().load(Ordering::Relaxed)
}

fn must<T, E: Debug>(r: Result<T, E>, what: &str) -> T {
  match r {
    Ok(v) => v,
    Err(e) => panic!("{what}: {e:?}"),
  }
}

/// xorshift64*（测试用确定性伪随机）
fn next_rand(s: &mut u64) -> u64 {
  let mut x = *s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *s = x;
  x
}

#[test]
fn exclusive_lock_unlock_bumps_version() {
  let l = HybridLatch::new();
  let old = sv(&l);
  assert!(l.try_lock_exclusive(old));
  assert_eq!(HybridLatch::lock_state(sv(&l)), 255);
  assert!(l.is_exclusively_latched(sv(&l)));
  l.unlock_exclusive();
  let now = sv(&l);
  assert_eq!(HybridLatch::lock_state(now), 0);
  assert!(!l.is_exclusively_latched(now));
  assert_eq!(HybridLatch::version(now), 1, "解锁必须推版本");
}

#[test]
fn shared_counting_and_upgrade() {
  let l = HybridLatch::new();
  let v0 = sv(&l);
  assert_eq!(HybridLatch::version(v0), 0);
  assert!(l.try_lock_shared(v0));
  let v1 = sv(&l);
  assert_eq!(HybridLatch::lock_state(v1), 1);
  assert!(l.try_lock_shared(v1));
  let v2 = sv(&l);
  assert_eq!(HybridLatch::lock_state(v2), 2);
  assert_eq!(HybridLatch::version(v2), 0, "共享上锁不推版本");
  assert!(!l.upgrade_lock(v2), "两读者不可升级");
  l.unlock_shared();
  let v3 = sv(&l);
  assert_eq!(HybridLatch::lock_state(v3), 1);
  // 上游 `UpgradeLock` 用 `compare_exchange_weak`（`hybrid_latch.cc:114`），
  // 允许虚假失败；单线程下状态未变时重试，不改变被测语义（Miri 会真实触发假失败）。
  let mut upgraded = false;
  for _ in 0..1024 {
    let cur = sv(&l);
    if l.upgrade_lock(cur) {
      upgraded = true;
      break;
    }
    assert_eq!(HybridLatch::lock_state(sv(&l)), 1, "升级失败不得改变状态");
  }
  assert!(upgraded, "单读者可升级（weak CAS 假失败允许重试）");
  assert_eq!(HybridLatch::lock_state(sv(&l)), 255);
  l.unlock_exclusive();
  assert_eq!(HybridLatch::version(sv(&l)), 1);
}

#[test]
fn optimistic_guard_validation() {
  let l = HybridLatch::new();
  let mut g = must(HybridGuard::optimistic_lock(&l), "乐观锁必须成功");
  assert_eq!(g.mode(), GuardMode::Optimistic);
  assert!(g.validate_optimistic_lock().is_ok());

  // 他人写导致版本漂移 -> Restart
  let v = sv(&l);
  assert!(l.try_lock_exclusive(v));
  assert!(
    matches!(HybridGuard::optimistic_lock(&l), Err(Error::Restart)),
    "排他持有期间乐观获取必须 Restart"
  );
  l.unlock_exclusive();
  let mut g2 = must(HybridGuard::optimistic_lock(&l), "乐观锁必须成功");
  assert!(g2.validate_optimistic_lock().is_ok(), "快照在写之后应有效");
  drop(g);
}

/// 核心 OLC 不变量：写者持有 EXCLUSIVE 时版本**不变**，仅比较版本会接受脏读。
#[test]
fn optimistic_validation_rejects_active_writer() {
  let l = HybridLatch::new();
  let mut g = must(HybridGuard::optimistic_lock(&l), "乐观锁必须成功");
  let v = sv(&l);
  assert_eq!(HybridLatch::lock_state(v), 0);
  assert!(l.try_lock_exclusive(v));
  let held = sv(&l);
  assert_eq!(HybridLatch::lock_state(held), 255);
  assert_eq!(
    HybridLatch::version(held),
    HybridLatch::version(v),
    "排他上锁不推版本"
  );
  assert!(
    matches!(g.validate_optimistic_lock(), Err(Error::Restart)),
    "写者临界区内必须 Restart"
  );
  l.unlock_exclusive();
}

#[test]
fn optimistic_validation_accepts_shared_cycle() {
  let l = HybridLatch::new();
  let mut g = must(HybridGuard::optimistic_lock(&l), "乐观锁必须成功");
  let v = sv(&l);
  assert!(l.try_lock_shared(v));
  assert!(
    g.validate_optimistic_lock().is_ok(),
    "同版本的共享持有不改变页内容"
  );
  l.unlock_shared();
}

/// 用 `futures_lite::future::block_on` 驱动而不是 compio Runtime：被测函数在无争用时
/// 不进入 `yield_now` 分支，普通 executor 足够；这样 Miri 可执行（compio 的 kqueue 驱动
/// 在 Miri 下不可用），且测试不再依赖异步运行时。
#[test]
fn optimistic_upgrade_to_exclusive() {
  block_on(async {
    let l = HybridLatch::new();
    let mut g = must(HybridGuard::optimistic_lock(&l), "乐观锁必须成功");
    must(g.upgrade_optimistic_to_exclusive().await, "升级必须成功");
    assert_eq!(g.mode(), GuardMode::Exclusive);
    let v = sv(&l);
    assert_eq!(HybridLatch::lock_state(v), 255);
    assert_eq!(HybridLatch::version(v), 0, "升级不推版本");
    g.unlock();
    assert_eq!(HybridLatch::version(sv(&l)), 1);
  });
}

#[test]
fn force_upgrade_single_reader() {
  block_on(async {
    let l = HybridLatch::new();
    let v = sv(&l);
    assert!(l.try_lock_shared(v));
    l.force_upgrade_lock().await;
    assert_eq!(HybridLatch::lock_state(sv(&l)), 255);
    l.unlock_exclusive();
    assert_eq!(HybridLatch::version(sv(&l)), 1);
  });
}

#[test]
fn wait_until_unlocked_yields() {
  let l = Arc::new(HybridLatch::new());
  let v = sv(&l);
  assert!(l.try_lock_exclusive(v));
  let l2 = Arc::clone(&l);
  let writer = thread::spawn(move || {
    thread::sleep(Duration::from_millis(1));
    l2.unlock_exclusive();
  });
  block_on(l.wait_until_unlocked());
  assert_eq!(HybridLatch::lock_state(sv(&l)), 0);
  must(writer.join(), "解锁线程必须成功");
}

#[test]
fn spin_wait_page_state_immediate_and_restart() {
  block_on(async {
    let l = HybridLatch::new();
    assert!(l.spin_wait_page_state(0).await.is_ok(), "当前状态即期望值");

    let v = sv(&l);
    assert!(l.try_lock_exclusive(v));
    assert!(
      matches!(l.spin_wait_page_state(0).await, Err(Error::Restart)),
      "状态永不归零时必须超限 Restart"
    );
    l.unlock_exclusive();
  });
}

// ---- 影子模型对拍：模型逐字复刻上游 C++ 的位运算与 CAS 语义 ----

mod model {
  pub const UNLOCKED: u64 = 0;
  pub const MAX_SHARED: u64 = 254;
  pub const EXCLUSIVE: u64 = 255;

  #[must_use]
  pub fn m_lock_state(v: u64) -> u64 {
    v >> 56
  }

  #[must_use]
  pub fn m_same(v: u64, state: u64) -> u64 {
    ((v << 8) >> 8) | (state << 56)
  }

  #[must_use]
  pub fn m_next(v: u64, state: u64) -> u64 {
    (((v << 8) >> 8) + 1) | (state << 56)
  }

  #[derive(Debug)]
  pub struct Word {
    pub v: u64,
  }

  impl Word {
    #[must_use]
    pub fn m_new() -> Self {
      Self { v: 0 }
    }

    fn cas(&mut self, old: u64, new: u64) -> bool {
      if self.v == old {
        self.v = new;
        true
      } else {
        false
      }
    }

    pub fn m_try_exclusive(&mut self, old: u64) -> bool {
      let s = m_lock_state(old);
      if s > UNLOCKED && s <= EXCLUSIVE {
        return false;
      }
      self.cas(old, m_same(old, EXCLUSIVE))
    }

    pub fn m_try_shared(&mut self, old: u64) -> bool {
      let s = m_lock_state(old);
      if s < MAX_SHARED {
        return self.cas(old, m_same(old, s + 1));
      }
      false
    }

    pub fn m_unlock_exclusive(&mut self) {
      let s = m_lock_state(self.v);
      assert_eq!(s, EXCLUSIVE);
      self.v = m_next(self.v, UNLOCKED);
    }

    pub fn m_unlock_shared(&mut self) {
      let s = m_lock_state(self.v);
      assert!(UNLOCKED < s && s < EXCLUSIVE);
      self.v = m_same(self.v, s - 1);
    }

    pub fn m_upgrade(&mut self, old: u64) -> bool {
      if m_lock_state(old) != 1 {
        return false;
      }
      self.cas(old, m_same(old, EXCLUSIVE))
    }

    pub fn m_downgrade(&mut self) {
      let s = m_lock_state(self.v);
      assert_eq!(s, EXCLUSIVE);
      self.v = m_next(self.v, 1);
    }
  }
}

#[test]
fn state_machine_differential_100k() {
  let l = HybridLatch::new();
  let mut m = model::Word::m_new();
  let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
  // 覆盖计数：UNLOCKED / Shared / EXCLUSIVE 必须都被走到
  let mut seen = [false; 3];
  let mut seen_op = [false; 8];

  for step in 0..100_000u32 {
    let old = sv(&l);
    assert_eq!(old, m.v, "step {step}: 初始状态漂移");
    assert_eq!(
      HybridLatch::lock_state(old) as u64,
      model::m_lock_state(m.v)
    );
    // 版本为低 56 位，测试侧独立写字面量掩码
    assert_eq!(HybridLatch::version(old), m.v & 0x00FF_FFFF_FFFF_FFFF);
    seen[match model::m_lock_state(m.v) {
      0 => 0,
      255 => 2,
      _ => 1,
    }] = true;

    let s = model::m_lock_state(old);
    let op = match s {
      0 => [0u8, 1, 6][(next_rand(&mut rng) % 3) as usize],
      255 => [0u8, 5, 7][(next_rand(&mut rng) % 3) as usize],
      1 => [0u8, 1, 3, 4][(next_rand(&mut rng) % 4) as usize],
      _ => [0u8, 1, 3][(next_rand(&mut rng) % 3) as usize],
    };

    let (real_ok, model_ok) = match op {
      0 => {
        let r = l.try_lock_exclusive(old);
        let mm = m.m_try_exclusive(old);
        (r, mm)
      }
      1 => {
        let r = l.try_lock_shared(old);
        let mm = m.m_try_shared(old);
        (r, mm)
      }
      3 => {
        l.unlock_shared();
        m.m_unlock_shared();
        (true, true)
      }
      4 => {
        // 上游 `UpgradeLock` 用 `compare_exchange_weak`（`hybrid_latch.cc:114`）：
        // 单线程下 weak CAS 仍允许虚假失败；状态未变时重试，模型按强 CAS 记账
        // （不重试会把「假失败」误判为语义不一致，Miri 可复现）。
        let mut r = l.upgrade_lock(old);
        let mut tries = 0u32;
        while !r && sv(&l) == old {
          tries += 1;
          assert!(tries < 1024, "step {step}: weak CAS 持续虚假失败");
          r = l.upgrade_lock(old);
        }
        let mm = m.m_upgrade(old);
        (r, mm)
      }
      5 => {
        l.downgrade_lock();
        m.m_downgrade();
        (true, true)
      }
      6 => {
        let x = next_rand(&mut rng);
        let ns = next_rand(&mut rng) as u8;
        assert_eq!(
          HybridLatch::same_version_new_state(x, ns),
          model::m_same(x, u64::from(ns))
        );
        assert_eq!(
          HybridLatch::next_version_new_state(x, ns),
          model::m_next(x, u64::from(ns))
        );
        (true, true)
      }
      7 => {
        l.unlock_exclusive();
        m.m_unlock_exclusive();
        (true, true)
      }
      _ => unreachable!(),
    };

    assert_eq!(real_ok, model_ok, "step {step}: 操作 {op} 结果不一致");
    assert_eq!(sv(&l), m.v, "step {step}: 操作 {op} 后状态不一致");
    seen_op[usize::from(op)] = true;
  }
  assert_eq!(seen, [true; 3], "UNLOCKED/Shared/EXCLUSIVE 必须全部覆盖");
  assert!(
    seen_op[..2].iter().all(|x| *x) && seen_op[3..].iter().all(|x| *x),
    "7 种操作（除未使用的 op2）必须全部覆盖"
  );
}

/// 4 线程真实争用：共享/排他互斥 + 读计数不丢失。
#[test]
fn mutual_exclusion_under_contention() {
  const THREADS: usize = 4;
  const OPS: usize = 5_000;

  let latch = Arc::new(HybridLatch::new());
  let readers = Arc::new(AtomicUsize::new(0));
  let writer = Arc::new(AtomicBool::new(false));
  let mut handles = Vec::with_capacity(THREADS);

  for tid in 0..THREADS {
    let latch = Arc::clone(&latch);
    let readers = Arc::clone(&readers);
    let writer = Arc::clone(&writer);
    handles.push(thread::spawn(move || {
      let mut rng = 0x1234_5678_9ABC_DEF0_u64 ^ (tid as u64 + 1);
      for _ in 0..OPS {
        if next_rand(&mut rng) & 3 == 0 {
          let mut spins = 0u32;
          loop {
            let v = latch.state_and_version().load(Ordering::Acquire);
            if latch.try_lock_exclusive(v) {
              break;
            }
            spins += 1;
            assert!(spins < 1_000_000, "写者长期无法推进：共享计数可能丢更新");
            if spins < 64 {
              spin_loop();
            } else {
              thread::yield_now();
            }
          }
          assert!(!writer.swap(true, Ordering::AcqRel), "两个写者同时持闩");
          assert_eq!(readers.load(Ordering::Acquire), 0, "写者与读者并存");
          black_box(latch.state_and_version().load(Ordering::Relaxed));
          writer.store(false, Ordering::Release);
          latch.unlock_exclusive();
        } else {
          let mut spins = 0u32;
          loop {
            let v = latch.state_and_version().load(Ordering::Acquire);
            if latch.try_lock_shared(v) {
              break;
            }
            spins += 1;
            if spins < 64 {
              spin_loop();
            } else {
              thread::yield_now();
            }
          }
          assert!(!writer.load(Ordering::Acquire), "读者与写者并存(前)");
          readers.fetch_add(1, Ordering::AcqRel);
          assert!(!writer.load(Ordering::Acquire), "读者与写者并存(后)");
          black_box(latch.state_and_version().load(Ordering::Relaxed));
          readers.fetch_sub(1, Ordering::AcqRel);
          latch.unlock_shared();
        }
      }
    }));
  }

  for h in handles {
    must(h.join(), "并发测试线程不得 panic");
  }
  assert_eq!(HybridLatch::lock_state(sv(&latch)), 0, "全部释放后状态归零");
  assert_eq!(readers.load(Ordering::Acquire), 0);
  assert!(!writer.load(Ordering::Acquire));
}
