//! ldb_range 区间锁回归（跳表语义 + 边界 + 槽位复用；期望值全部字面量）。
//!
//! 语义对照上游 `leanstore.latency/src/sync/range_lock.cc`
//! （TryLockRange `:67`、TestRange `:91`、UnlockRange `:106`）。
//!
//! 用例与变异的「唯一击杀」关系见 `target/crate-review/ldb_range-round1.md` §7.1：
//! 覆盖集被其它用例完全包含的用例已合并，合并后每个用例至少有一个只有它能击杀的变异。

use std::panic::catch_unwind;

use ldb_range::RangeLock;

/// 探针：`[start, start+len-1]` 是否与已锁区间冲突（只读，不改状态）
fn conflicts(rl: &RangeLock, start: u64, len: u64) -> bool {
  !rl.test_range(0, start, len)
}

/// 相交矩阵：部分重叠 / 包含 / 被包含 / 同起点 / 同终点 / 左右相邻 / 最小起点（唯一击杀 M02）
#[test]
fn overlap_adjacency_containment_matrix() {
  let rl = RangeLock::new(4);
  assert!(rl.try_lock_range(0, 10, 10)); // [10,19]
  assert!(!rl.try_lock_range(0, 15, 10), "与 [10,19] 相交");
  assert!(!rl.try_lock_range(0, 5, 10), "尾端落入 [10,19]");
  assert!(!rl.try_lock_range(0, 19, 5), "起点落在 [10,19] 内");
  assert!(!rl.try_lock_range(0, 10, 1), "同起点单页");
  assert!(!rl.try_lock_range(0, 19, 1), "同终点单页");
  assert!(rl.try_lock_range(0, 20, 5), "紧邻右端不算相交"); // [20,24]
  assert!(rl.try_lock_range(0, 1, 9), "紧邻左端不算相交"); // [1,9]
  assert!(!rl.try_lock_range(0, 19, 2), "[19,20] 与两侧都相交");

  // 包含：大区间锁住后，其内部任意子区间都冲突
  let rl = RangeLock::new(4);
  assert!(rl.try_lock_range(0, 10, 91)); // [10,100]
  assert!(!rl.try_lock_range(0, 20, 30), "完全包含于 [10,100]");
  assert!(!rl.try_lock_range(0, 100, 1), "同终点单页");
  assert!(rl.try_lock_range(0, 101, 1));

  // start=1（最小合法起点）的单页
  let rl = RangeLock::new(4);
  assert!(rl.try_lock_range(0, 1, 1));
  assert!(!rl.try_lock_range(0, 1, 1), "单页重复加锁应失败");
  assert!(conflicts(&rl, 1, 1));
}

/// 零长度区间（`end = start - 1`）：与上游 `start + len - 1` 同值（唯一击杀 M20）
#[test]
fn zero_length_range_is_degenerate() {
  let rl = RangeLock::new(4);
  assert!(rl.try_lock_range(0, 10, 0), "零长度区间 [10,9] 可加锁");
  assert!(rl.try_lock_range(0, 11, 1), "相邻真实区间 [11,11] 不受影响");
  assert!(rl.try_lock_range(0, 10, 1), "零长度区间不占用 10 号起点");
  assert!(!rl.test_range(0, 10, 1), "[10,10] 已锁");
  // 解锁按 start 摘除「首个」同 start 节点：先摘掉先插入的零长度区间
  rl.unlock_range(0, 10);
  assert!(!rl.test_range(0, 10, 1), "仍持有 [10,10]");
  rl.unlock_range(0, 10);
  assert!(rl.test_range(0, 10, 1), "两个同 start 节点都已释放");
  assert!(
    rl.test_range(0, 10, 0),
    "探针 [10,9] 的前驱是 head（10 > 9），判空闲"
  );
}

/// u64::MAX 邻域：饱和端点、MAX 起点的退化探针（唯一击杀 M03）
#[test]
fn max_value_neighborhood_is_consistent() {
  let rl = RangeLock::new(4);
  let top = u64::MAX;
  assert!(
    rl.try_lock_range(0, top - 4, 5),
    "越过上界应饱和封顶而非回绕"
  ); // [MAX-4, MAX-1]
  assert!(conflicts(&rl, top - 4, 5));
  assert!(
    rl.test_range(0, top, 1),
    "起点=MAX 的探针在封顶后退化为空区间（上游同路径为 UB）"
  ); // end = MAX-1
  assert!(rl.try_lock_range(0, top, 1), "起点=MAX 的零长度区间可加锁");
  // 起点=MAX 的节点只能靠严格前驱（tail.start == MAX 不满足 `< start`）摘除；
  // 若遍历写成 `<=` 会踏上 tail 的 NodeId::MAX 前向值并越界 panic
  rl.unlock_range(0, top);
  assert!(rl.test_range(0, top, 1), "起点=MAX 的区间可解锁");
  // 二次 unlock（该 start 已不存在）必须是 no-op：strict-`<` 仍不踏 tail，结构不得被破坏
  rl.unlock_range(0, top);
  assert!(rl.test_range(0, top, 1), "二次 unlock 不得破坏结构");
  assert!(
    rl.try_lock_range(0, top, 1),
    "二次 unlock 后起点=MAX 的区间可重新加锁"
  );
  rl.unlock_range(0, top);
  assert!(rl.test_range(0, top, 1));
  assert!(!conflicts(&rl, top - 5, 1), "封顶区间之前的单页仍可用");
}

/// 解锁未持有的 start 必须是 no-op（上游 `Ensure` 会终止进程，本仓静默返回）（唯一击杀 M04）
#[test]
fn unlock_unknown_start_is_noop() {
  let rl = RangeLock::new(4);
  assert!(rl.try_lock_range(0, 10, 10)); // [10,19]
  // 只有单节点时，未持有的 start 必须既不移除它、也不越权摘除后继节点
  rl.unlock_range(0, 11); // 落在 [10,19] 内部但起点不等
  assert!(conflicts(&rl, 10, 10), "未持有的 start 不得移除已锁区间");
  rl.unlock_range(0, 1000);
  assert!(conflicts(&rl, 15, 5));
  // 双节点：解锁未持有的起点不得摘掉“后继”的另一个区间
  assert!(rl.try_lock_range(0, 30, 10)); // [30,39]
  rl.unlock_range(0, 20); // 介于两者之间，未持有
  assert!(
    conflicts(&rl, 30, 10),
    "未持有的 start 不得摘除后继区间 [30,39]"
  );
  assert!(conflicts(&rl, 10, 10), "也不得摘除前驱区间 [10,19]");
  rl.unlock_range(0, 40); // 越过表尾
  assert!(conflicts(&rl, 30, 10));
  // 解锁已持有的 start 后再解锁一次：该 start 已不存在，必须是 no-op 且不破坏结构
  rl.unlock_range(0, 30);
  assert!(!conflicts(&rl, 30, 10), "已解锁区间应空闲");
  rl.unlock_range(0, 30);
  assert!(!conflicts(&rl, 30, 10), "二次 unlock 不得改变状态");
  assert!(
    rl.try_lock_range(0, 30, 10),
    "二次 unlock 后同一区间可重新加锁"
  );
}

/// 解锁→重锁、`test_range` 只读、槽位复用、3000 区间深层（唯一击杀 M27）
#[test]
fn unlock_relock_slot_reuse_and_deep_levels() {
  let rl = RangeLock::new(4);
  assert!(rl.try_lock_range(0, 100, 10));
  assert!(!rl.test_range(0, 100, 10));
  rl.unlock_range(0, 100);
  assert!(
    rl.try_lock_range(0, 100, 10),
    "解锁后可重新加锁（槽位复用）"
  );
  for _ in 0..3 {
    assert!(!rl.test_range(0, 100, 10));
  }
  assert!(
    rl.try_lock_range(0, 200, 1),
    "test_range 无副作用：200 未被占用"
  );
  rl.unlock_range(0, 200);
  rl.unlock_range(0, 100);

  // 200 个区间正序加锁、逆序解锁、再正序加锁（free 列表复用全部槽位）
  let ranges: Vec<(u64, u64)> = (0..200).map(|i| (i * 10 + 1, 9)).collect();
  for &(s, l) in &ranges {
    assert!(rl.try_lock_range(0, s, l), "首次加锁 {s} 应成功");
  }
  for &(s, _) in ranges.iter().rev() {
    rl.unlock_range(0, s);
  }
  for &(s, l) in &ranges {
    assert!(rl.try_lock_range(0, s, l), "槽位复用后加锁 {s} 应成功");
  }
  assert!(conflicts(&rl, 1001, 9), "复用后原区间仍应互相排斥");
  assert!(!conflicts(&rl, 2001, 9), "空隙仍可用");

  // 3000 个区间：层高越过 8，覆盖深层查找与部分摘除
  let rl = RangeLock::new(4);
  for i in 0..3000u64 {
    assert!(rl.try_lock_range(0, i * 4 + 1, 2), "第 {i} 个区间");
  }
  assert!(conflicts(&rl, 6001, 2), "深层跳表仍能找到相交区间"); // 第 1500 个区间 [6001,6002]
  assert!(conflicts(&rl, 6002, 4), "跨相邻两区间的探针冲突"); // [6002,6005]
  assert!(!conflicts(&rl, 6003, 1), "介于两区间之间可锁"); // 6003 空闲
  for i in (0..3000u64).step_by(3) {
    rl.unlock_range(0, i * 4 + 1);
  }
  for i in (0..3000u64).step_by(3) {
    assert!(rl.try_lock_range(0, i * 4 + 1, 2), "已释放的第 {i} 个区间");
  }
  for i in (1..3000u64).step_by(3) {
    assert!(conflicts(&rl, i * 4 + 1, 2), "未释放的第 {i} 个区间仍锁住");
  }
}

/// wid 越界必须钳住而不是 panic（唯一击杀 M11）
#[test]
fn out_of_range_wid_unlock_is_safe() {
  // ldb_epoch 回收队列按 wid 索引（`epoch.rs:154`）；每次 unlock 以 1/100（RECLAIM_RATE）
  // 概率进入索引分支，5 万次下未被钳住必 panic。
  let rl = RangeLock::new(1);
  for _ in 0..50_000 {
    assert!(rl.try_lock_range(0, 10, 10));
    rl.unlock_range(999, 10);
  }
  assert!(!conflicts(&rl, 10, 10));
}

#[test]
fn zero_workers_rejected() {
  let r = catch_unwind(|| RangeLock::new(0));
  assert!(
    r.is_err(),
    "workers=0 必须在构造期拒绝（epoch 队列按 wid 索引）"
  );
}

#[test]
#[should_panic(expected = "区间起点必须为正")]
fn zero_start_try_lock_panics() {
  let rl = RangeLock::new(4);
  rl.try_lock_range(0, 0, 1);
}

#[test]
#[should_panic(expected = "区间起点必须为正")]
fn zero_start_test_range_panics() {
  let rl = RangeLock::new(4);
  rl.test_range(0, 0, 1);
}

#[test]
#[should_panic(expected = "区间起点必须为正")]
fn zero_start_unlock_panics() {
  let rl = RangeLock::new(4);
  rl.unlock_range(0, 0);
}
