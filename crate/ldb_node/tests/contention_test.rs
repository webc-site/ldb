//! CIDR'21 Contention Split 节点侧状态机（上游 `BTreeGenericIterator.hpp:526-557`）。

use ldb_node::{BTreeNode, ContentionTracker};

/// 上游参考实现：`cm_update_on=7`、`cm_period=14`、阈值 1%（`Config.cpp:39-41`）。
fn reference_note(
  last: &mut i32,
  restarts: &mut u32,
  accesses: &mut u32,
  slot: u32,
  faced: bool,
  rand: u64,
  count: u32,
) -> Option<i32> {
  if rand & ((1u64 << 7) - 1) != 0 {
    return None;
  }
  let prev = *last;
  *last = slot as i32;
  *restarts += u32::from(faced);
  *accesses += 1;
  if rand & ((1u64 << 14) - 1) != 0 {
    return None;
  }
  let normalized = 100 * u64::from(*restarts) / u64::from(*accesses);
  *restarts = 0;
  *accesses = 0;
  if prev != slot as i32 && normalized >= 1 && count > 2 {
    Some(prev.min(slot as i32))
  } else {
    None
  }
}

struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    self.0 ^= self.0 >> 12;
    self.0 ^= self.0 << 25;
    self.0 ^= self.0 >> 27;
    self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
  }
}

#[test]
fn default_sentinel_and_field_types() {
  let t = ContentionTracker::default();
  let last: i32 = t.last_modified_pos;
  let restarts: u32 = t.restarts_counter;
  let accesses: u32 = t.access_counter;
  assert_eq!((last, restarts, accesses), (-1, 0, 0));
  let mut t2 = t;
  t2.last_modified_pos = 5;
  t2.restarts_counter = 3;
  t2.access_counter = 7;
  t2.reset();
  assert_eq!(
    (t2.last_modified_pos, t2.restarts_counter, t2.access_counter),
    (-1, 0, 0)
  );
}

#[test]
fn update_gate_is_one_in_128_and_period_gate_one_in_16384() {
  let mut t = ContentionTracker::default();
  // bit7 未清零：不采样，状态不变
  for rand in [1u64, 2, 63, 127, 128 | 1] {
    assert_eq!(t.note_access(5, true, rand, 10), None);
    assert_eq!(
      (t.last_modified_pos, t.restarts_counter, t.access_counter),
      (-1, 0, 0)
    );
  }
  // bit7 清零但 bit14 未清零：记录状态但不结算
  assert_eq!(t.note_access(5, true, 128, 10), None);
  assert_eq!(
    (t.last_modified_pos, t.restarts_counter, t.access_counter),
    (5, 1, 1)
  );
  // 2^14 同时满足两层门：结算并清零计数；上一采样点 last=5 → 分裂点 5
  assert_eq!(t.note_access(7, true, 1 << 14, 10), Some(5));
  assert_eq!(
    (t.last_modified_pos, t.restarts_counter, t.access_counter),
    (7, 0, 0)
  );
}

#[test]
fn split_position_is_min_of_last_and_current() {
  let mut t = ContentionTracker::default();
  assert_eq!(t.note_access(9, true, 0, 10), Some(-1));
  assert_eq!(t.note_access(3, true, 0, 10), Some(3));
  assert_eq!(t.note_access(8, true, 0, 10), Some(3));
  // 同槽位不触发
  assert_eq!(t.note_access(8, true, 0, 10), None);
  // count <= 2 不触发
  assert_eq!(t.note_access(4, true, 0, 2), None);
}

#[test]
fn hot_predicate_uses_normalized_ratio() {
  let mut n = BTreeNode::new(true, 0);
  assert!(!n.has_faced_contention());
  n.hdr.contention.access_counter = 100;
  n.hdr.contention.restarts_counter = 0;
  assert!(!n.has_faced_contention());
  n.hdr.contention.restarts_counter = 1;
  assert!(n.has_faced_contention(), "1/100 = 1% 达到阈值");
  n.hdr.contention.access_counter = 101;
  assert!(!n.has_faced_contention(), "1/101 < 1%");
}

/// 1e5 随机访问与上游逐行参考实现比对：每次的返回动作与三个字段全部一致。
#[test]
fn state_machine_matches_upstream_reference_100k() {
  let mut rng = Rng(0xDEAD_BEEF_1234_5678);
  let mut t = ContentionTracker::default();
  let (mut last, mut restarts, mut accesses) = (-1i32, 0u32, 0u32);
  for step in 0..100_000 {
    let rand = rng.next();
    let slot = (rand % 228) as u32;
    let faced = ((rand >> 8) & 1) != 0;
    let count = ((rand >> 16) % 240) as u32;
    let got = t.note_access(slot, faced, rand, count);
    let want = reference_note(
      &mut last,
      &mut restarts,
      &mut accesses,
      slot,
      faced,
      rand,
      count,
    );
    assert_eq!(got, want, "步骤 {step} 分裂动作不一致");
    assert_eq!(
      (t.last_modified_pos, t.restarts_counter, t.access_counter),
      (last, restarts, accesses),
      "步骤 {step} 追踪器状态不一致"
    );
  }
}
