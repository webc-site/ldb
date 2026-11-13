//! PageState Clock 状态机（上游 `page_state.cc` + `HybridLatchImpl<PageStateMode>` 继承语义）。

use std::sync::atomic::Ordering;

use ldb_sync::{PageStateExt, page_mode};
use ldb_type::PageState;

/// 低 56 位版本掩码（测试侧独立字面量，不复用生产常量）
const VERSION_MASK: u64 = 0x00FF_FFFF_FFFF_FFFF;

fn word(ps: &PageState) -> u64 {
  ps.load(Ordering::Relaxed)
}

fn state_of(ps: &PageState) -> u8 {
  (word(ps) >> 56) as u8
}

fn ver_of(ps: &PageState) -> u64 {
  word(ps) & VERSION_MASK
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

/// `page_mode` 常量转发必须与上游 `page_state.h:13-18` 的冻结编码逐值一致。
/// 右值为测试侧独立字面量（不复用生产常量）；本 crate 源码不消费该别名，
/// 因此该断言是这组转发常量唯一的接线证据。
#[test]
fn page_mode_forwarding_matches_upstream_encoding() {
  assert_eq!(page_mode::UNLOCKED, 0);
  assert_eq!(page_mode::MAX_SHARED, 252);
  assert_eq!(page_mode::EXCLUSIVE, 253);
  assert_eq!(page_mode::MARKED, 254);
  assert_eq!(page_mode::EVICTED, 255);
}

#[test]
fn init_sets_evicted_version_zero() {
  let ps = PageState::new();
  assert_eq!(state_of(&ps), 0, "ldb_type::PageState::new 为 UNLOCKED");
  ps.init();
  assert_eq!(state_of(&ps), 255);
  assert_eq!(ver_of(&ps), 0);
  assert_eq!(ps.to_string(), "[State: 255, Version: 0]");
}

// ---- 影子模型对拍：模型逐字复刻上游 page_state.cc 的位运算与 CAS 语义 ----

mod model {
  pub const MAX_SHARED: u8 = 252;
  pub const EXCLUSIVE: u8 = 253;
  pub const MARKED: u8 = 254;
  pub const EVICTED: u8 = 255;

  #[must_use]
  pub fn m_same(v: u64, state: u8) -> u64 {
    ((v << 8) >> 8) | (u64::from(state) << 56)
  }

  #[must_use]
  pub fn m_next(v: u64, state: u8) -> u64 {
    (((v << 8) >> 8) + 1) | (u64::from(state) << 56)
  }

  #[derive(Debug)]
  pub struct Word {
    pub v: u64,
  }

  impl Word {
    #[must_use]
    pub const fn m_new() -> Self {
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

    fn m_state(&self) -> u8 {
      (self.v >> 56) as u8
    }

    pub fn m_try_shared(&mut self, old: u64) -> bool {
      let s = (old >> 56) as u8;
      if s < MAX_SHARED {
        return self.cas(old, m_same(old, s + 1));
      }
      if s == MARKED {
        return self.cas(old, m_same(old, 1));
      }
      false
    }

    pub fn m_unlock_shared(&mut self) {
      let s = self.m_state();
      assert!(0 < s && s <= MAX_SHARED);
      self.v = m_same(self.v, s - 1);
    }

    pub fn m_try_mark(&mut self, old: u64) -> bool {
      assert_eq!((old >> 56) as u8, 0);
      self.cas(old, m_same(old, MARKED))
    }

    pub fn m_unmark(&mut self, old: u64) -> bool {
      assert_eq!((old >> 56) as u8, MARKED);
      self.cas(old, m_same(old, 0))
    }

    pub fn m_unmark_and_lock_exclusive(&mut self, old: u64) -> bool {
      assert_eq!((old >> 56) as u8, MARKED);
      self.cas(old, m_same(old, EXCLUSIVE))
    }

    pub fn m_unlock_exclusive_and_evict(&mut self) {
      assert_eq!(self.m_state(), EXCLUSIVE);
      self.v = m_next(self.v, EVICTED);
    }

    pub fn m_unlock_exclusive_and_mark(&mut self) {
      assert_eq!(self.m_state(), EXCLUSIVE);
      self.v = m_next(self.v, MARKED);
    }
  }
}

#[test]
fn state_machine_differential_100k() {
  let mut rng = 0xC0FF_EE00_1234_5678_u64;
  // 覆盖计数：UNLOCKED / Shared / EXCLUSIVE / MARKED / EVICTED 必须都被走到
  let mut seen = [false; 5];
  let mut seen_op = [false; 8];
  let mut total = 0u32;
  let mut segments = 0u32;

  // EVICTED 是 L1 可见状态机的吸收态（缺页加载属 L2 buffer manager），
  // 到达后本段结束，用全新 PageState 继续下一段，累计对拍 ≥1e5 步。
  while total < 100_000 {
    let ps = PageState::new();
    let mut m = model::Word::m_new();
    segments += 1;
    seen[0] = true;

    loop {
      let old = word(&ps);
      assert_eq!(old, m.v, "step {total}: 初始状态漂移");
      assert_eq!(u64::from(state_of(&ps)), m.v >> 56);
      assert_eq!(ver_of(&ps), m.v & 0x00FF_FFFF_FFFF_FFFF);

      let s = (m.v >> 56) as u8;
      if s == 255 {
        break;
      }
      let ops: &[u8] = match s {
        0 => &[0, 1],
        253 => &[0, 4, 5],
        254 => &[0, 2, 3],
        _ => &[0, 7],
      };
      let op = ops[(next_rand(&mut rng) % ops.len() as u64) as usize];

      let (real_ok, model_ok) = match op {
        0 => (ps.try_lock_shared(old), m.m_try_shared(old)),
        1 => (ps.try_mark(old), m.m_try_mark(old)),
        2 => {
          let mut new_state = 0;
          let r = ps.unmark(old, &mut new_state);
          let mm = m.m_unmark(old);
          if r {
            assert_eq!(new_state, m.v, "step {total}: unmark 输出字不一致");
          }
          (r, mm)
        }
        3 => (
          ps.unmark_and_lock_exclusive(old),
          m.m_unmark_and_lock_exclusive(old),
        ),
        4 => {
          ps.unlock_exclusive_and_evict();
          m.m_unlock_exclusive_and_evict();
          (true, true)
        }
        5 => {
          ps.unlock_exclusive_and_mark();
          m.m_unlock_exclusive_and_mark();
          (true, true)
        }
        7 => {
          ps.unlock_shared();
          m.m_unlock_shared();
          (true, true)
        }
        _ => unreachable!(),
      };

      assert_eq!(real_ok, model_ok, "step {total}: 操作 {op} 结果不一致");
      assert_eq!(word(&ps), m.v, "step {total}: 操作 {op} 后状态不一致");
      seen_op[usize::from(op)] = true;
      total += 1;
      seen[match (m.v >> 56) as u8 {
        0 => 0,
        253 => 2,
        254 => 3,
        255 => 4,
        _ => 1,
      }] = true;
      if total >= 100_000 || (m.v >> 56) as u8 == 255 {
        break;
      }
    }
  }
  assert_eq!(seen, [true; 5], "五种状态必须全部覆盖（{segments} 段）");
  assert!(
    seen_op[..6].iter().all(|x| *x) && seen_op[7],
    "7 种状态转换必须全部覆盖"
  );
  assert_eq!(total, 100_000);
}
