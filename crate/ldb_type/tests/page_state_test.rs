//! PageState 位布局与原子状态字回归。
//!
//! 布局依据上游 `ZLeanStore/src/include/sync/hybrid_latch.h:32,72-87`
//! （高 8 位状态 / 低 56 位版本）
//! 与 `ZLeanStore/src/include/sync/page_state.h:13-18`（UNLOCKED=0 / MAX_SHARED=252 /
//! EXCLUSIVE=253 / MARKED=254 / EVICTED=255）。
//!
//! 关键协议值（版本掩码宽度、状态码、溢出进位）按字面量钉死，以独立校验与上游 C++
//! 一致；组合验证用例可引用公开常量：`shared_code_domain` 用 `MAX_SHARED + 1 ==
//! EXCLUSIVE` 的关系钉死共享域上界，256 个码值的字面量映射由
//! `state_word_decoding_is_exhaustive` 穷尽覆盖。

use std::{
  sync::{Arc, atomic::Ordering},
  thread::spawn,
};

use ldb_type::{
  PageState, PageStateMode, VERSION_MASK, next_version_new_state, same_version_new_state,
};

/// `UNLOCKED` 编码（上游 `page_state.h:13`）
const UNLOCKED_CODE: u8 = 0;
/// `EXCLUSIVE` 编码（上游 `page_state.h:16`）
const EXCLUSIVE_CODE: u8 = 253;
/// `MARKED` 编码（上游 `page_state.h:17`）
const MARKED_CODE: u8 = 254;

#[test]
fn state_word_bit_layout_matches_upstream() {
  // 保留版本位、替换状态位（hybrid_latch.h:72-76）
  let v = same_version_new_state(0x0034_5678_9ABC_DEF0, EXCLUSIVE_CODE);
  assert_eq!(v, 0xFD34_5678_9ABC_DEF0);
  assert_eq!(v & VERSION_MASK, 0x0034_5678_9ABC_DEF0);

  // 版本 +1、换状态为 UNLOCKED(0)（hybrid_latch.h:78-82）
  let n = next_version_new_state(v, 0);
  assert_eq!(n, 0x0034_5678_9ABC_DEF1);
  assert_eq!(n & VERSION_MASK, 0x0034_5678_9ABC_DEF0 + 1);
}

#[test]
fn state_word_decoding_is_exhaustive() {
  let decode = |state: u8| PageStateMode::from_state_word(u64::from(state) << 56);
  // 高 8 位 256 个码值全部穷尽：命名码 + 共享计数域，期望值为上游编码的字面映射
  for state in 0u8..=255 {
    let expect = match state {
      0 => PageStateMode::Unlocked,
      EXCLUSIVE_CODE => PageStateMode::Exclusive,
      MARKED_CODE => PageStateMode::Marked,
      255 => PageStateMode::Evicted,
      n => PageStateMode::Shared(n),
    };
    assert_eq!(decode(state), expect, "state={state}");
  }
}

#[test]
fn page_state_atomic_word_roundtrip() {
  let ps = PageState::new();
  assert_eq!(ps.version(), 0);
  assert_eq!(ps.lock_state(), PageStateMode::Unlocked);

  // 状态替换不动版本（5 = Shared 持有数）
  let cur = ps.load(Ordering::Relaxed);
  ps.store(same_version_new_state(cur, 5), Ordering::Release);
  assert_eq!(ps.lock_state(), PageStateMode::Shared(5));
  assert_eq!(ps.version(), 0);

  // 版本推进（+1）不动状态：回 UNLOCKED(0)；CAS 成功返回【旧值】
  let cur = ps.load(Ordering::Relaxed);
  let next = next_version_new_state(cur, 0);
  assert_eq!(ps.cas(cur, next, Ordering::AcqRel), Ok(cur));
  assert_eq!(ps.version(), 1);
  assert_eq!(ps.lock_state(), PageStateMode::Unlocked);

  // CAS 失败返回实际值，不覆盖
  assert_eq!(ps.cas(cur, 0, Ordering::AcqRel), Err(next));
  assert_eq!(ps.load(Ordering::Relaxed), next);

  // clock 标记转换：UNLOCKED → MARKED（同版本）
  let marked = same_version_new_state(next, MARKED_CODE);
  assert_eq!(ps.cas(next, marked, Ordering::AcqRel), Ok(next));
  assert_eq!(ps.lock_state(), PageStateMode::Marked);
  assert_eq!(ps.version(), 1);
}

/// `PageStateMode` 编码常量可在 `const` 上下文参与编译期状态字构造
/// （上游 `page_state.cc:9-12` `Init()` 写 EVICTED；`hybrid_latch.h:78-82` 版本推进）。
#[test]
fn state_codes_are_const_usable() {
  // 版本 0 / EVICTED —— 上游 Init() 的初值状态字
  const EVICTED_INIT: u64 = same_version_new_state(0, PageStateMode::EVICTED);
  assert_eq!(EVICTED_INIT, 0xFF00_0000_0000_0000);
  assert_eq!(
    PageStateMode::from_state_word(EVICTED_INIT),
    PageStateMode::Evicted
  );

  // 版本 +1 / MARKED —— EVICTED(全错位) 下的低位进位不得污染状态位
  const MARKED_V1: u64 = next_version_new_state(EVICTED_INIT, PageStateMode::MARKED);
  assert_eq!(MARKED_V1, 0xFE00_0000_0000_0001);
  assert_eq!(
    PageStateMode::from_state_word(MARKED_V1),
    PageStateMode::Marked
  );
}

/// 版本掩码宽度与「版本位满值进位污染状态位」按上游字面量钉死。
///
/// 依据上游 `hybrid_latch.h:32`（`VERSION_MASK = (1 << 56) - 1`，即低 56 位）
/// 与 `:74/80`（`((v << 8) >> 8)` 先左移再右移 = 取低 56 位；`+ 1` 不做钳位，
/// 版本位满值时的进位直接落到状态位）；`page_state.h:13-18` 的状态码占高 8 位。
///
/// 全部断言用字面量而非实现常量：掩码宽度（56 位）与进位行为一旦漂移，
/// 状态字与 C++ 侧的位布局就不再兼容，必须在编译期形状之外由用例独立钉死。
#[test]
fn version_mask_is_56_bits_and_overflow_carries_into_state() {
  // 掩码宽度：低 56 位全 1、高 8 位清零（`u64::MAX` 的高 8 位必须被掩掉）
  assert_eq!(
    same_version_new_state(u64::MAX, UNLOCKED_CODE),
    0x00FF_FFFF_FFFF_FFFF
  );
  // 掩码宽度 + 状态左移 56：只保留高 8 位状态
  assert_eq!(
    same_version_new_state(u64::MAX, EXCLUSIVE_CODE),
    0xFDFF_FFFF_FFFF_FFFF
  );

  // 版本位满值（2^56 - 1）+1：进位落到状态位、与状态位做按位或（上游同，不钳位）
  let full = 0x00FF_FFFF_FFFF_FFFF | (u64::from(MARKED_CODE) << 56);
  assert_eq!(
    next_version_new_state(full, MARKED_CODE),
    0xFF00_0000_0000_0000
  );

  // 原子字上的表现：版本恰为 56 位全域时，状态位不被版本位覆盖
  let ps = PageState::new();
  ps.store(
    0x00FF_FFFF_FFFF_FFFF | (u64::from(MARKED_CODE) << 56),
    Ordering::Release,
  );
  assert_eq!(ps.version(), 0x00FF_FFFF_FFFF_FFFF);
  assert_eq!(ps.lock_state(), PageStateMode::Marked);
}

/// 状态位与版本位必须同字原子：并发 CAS 下任何一次观察到的状态字，
/// 其高 8 位只能是被写入过的编码，且版本恰好等于已提交的步进数。
///
/// 依据上游 `hybrid_latch.h:72-82` 的「高 8 位状态 / 低 56 位版本」单字布局
/// （两侧分别由 `SameVersionNewState` / `NextVersionNewState` 构造）。
#[test]
fn cas_word_is_atomic_under_contention() {
  const THREADS: usize = 8;
  const STEPS: usize = 2_000;
  let ps = Arc::new(PageState::new());
  let handles: Vec<_> = (0..THREADS)
    .map(|t| {
      // 偶数线程换 UNLOCKED(0)、奇数线程换 Shared(7)：任意时刻状态只能落在这两个编码上
      let want = if t % 2 == 0 { 0 } else { 7 };
      let ps = Arc::clone(&ps);
      spawn(move || {
        for _ in 0..STEPS {
          loop {
            let cur = ps.load(Ordering::Acquire);
            assert!(
              matches!(
                PageStateMode::from_state_word(cur),
                PageStateMode::Unlocked | PageStateMode::Shared(7)
              ),
              "观察到撕裂状态字 {cur:#x}"
            );
            assert!((cur & VERSION_MASK) <= (THREADS * STEPS) as u64);
            let next = next_version_new_state(cur, want);
            if ps.cas(cur, next, Ordering::AcqRel).is_ok() {
              break;
            }
          }
        }
      })
    })
    .collect();
  for h in handles {
    h.join().unwrap();
  }
  // 每次成功的 CAS 恰好提交一个「版本 +1」步进，故终值严格等于步进总数
  assert_eq!(ps.version(), (THREADS * STEPS) as u64);
  assert!(matches!(
    ps.lock_state(),
    PageStateMode::Unlocked | PageStateMode::Shared(7)
  ));
}

/// 共享码域上界：按上游 `page_state.cc:40` 的 `state < MAX_SHARED` 守卫逐级上锁，
/// 计数连续经过 `Shared(1)..=Shared(252)` 且**永不**落到命名码域（253/254/255）。
///
/// 这正是该守卫必须写 `<` 而不能写 `<=` 的原因：`MAX_SHARED + 1` 就是 `EXCLUSIVE`，
/// 共享计数若再加一级就会被解码成排他态。编码域由本 crate 定义，故在此钉死。
#[test]
fn shared_code_domain_saturates_below_exclusive_code() {
  let ps = PageState::new();
  let mut cur = ps.load(Ordering::Relaxed);
  assert_eq!(PageStateMode::from_state_word(cur), PageStateMode::Unlocked);
  for k in 1..=PageStateMode::MAX_SHARED {
    let next = same_version_new_state(cur, k);
    assert_eq!(ps.cas(cur, next, Ordering::AcqRel), Ok(cur));
    cur = next;
    assert_eq!(
      PageStateMode::from_state_word(cur),
      PageStateMode::Shared(k)
    );
    assert_eq!(cur & VERSION_MASK, 0, "共享计数上锁不得推进版本");
  }
  // 满值 +1 即撞上 EXCLUSIVE 编码（守卫必须拦住这一步）
  assert_eq!(
    PageStateMode::from_state_word(same_version_new_state(0, PageStateMode::MAX_SHARED + 1)),
    PageStateMode::Exclusive
  );
}
