//! 跳表内部结构不变量测试（`#[cfg(test)]`，随 `cargo nextest run -p ldb_range` 执行）。
//!
//! 归属说明：三条用例都直接读写 `Inner` 的私有字段（`arena`/`free`/`level`/`append_id`），
//! 集成测试是独立 crate，访问私有字段会得到 E0616（本轮实测），因此保留在 `src/` 的
//! `#[cfg(test)]` 模块、不进入 release；可观测的上游语义测试仍在 `tests/`。
//!
//! `tests/range_lock_test.rs` 钉的是上游可观测语义；这里直接读 `Inner` 的 arena/free/level，
//! 钉死四条**内部**不变量（上游语义测试无法观察，层高冻结这类缺陷就属于此列）：
//!
//! 1. `level` 恒等于最高非空层（增长连新顶层 + 收缩真实生效）；
//! 2. level-0 链序 == 影子模型（含重复 start 的退化区间的新者在前）；
//! 3. 每层链都是 level-0 链的子序列（跳表分层不变），无环、无游离节点；
//! 4. free 列表与链严格不相交；槽位复用（ABA）后内容与影子一致。
//!
//! 1e6 规模的同口径验证与层高分布见 `target/crate-review/ldb_range-round2.md` §4.2。

use std::collections::VecDeque;
#[cfg(target_pointer_width = "64")]
use std::panic::catch_unwind;

use fastrand::Rng;

use super::{Inner, MAX_LEVEL, MAX_NODES, NodeId, RangeLock, range_end};

/// 最高非空层：head 在该层的前向不是 tail（空表返回 0）
fn top_nonempty(inner: &Inner) -> usize {
  let head = &inner.arena[inner.head as usize];
  (0..MAX_LEVEL)
    .rev()
    .find(|&k| head.forward[k] != inner.tail)
    .unwrap_or(0)
}

/// 收集第 `level` 层的链（节点 id 顺序）；长度超过 `live` 即判成环
fn chain(inner: &Inner, level: usize, live: usize) -> Vec<NodeId> {
  let mut out = Vec::with_capacity(live + 1);
  let mut p = inner.head;
  loop {
    let q = inner.arena[p as usize].forward[level];
    if q == inner.tail {
      return out;
    }
    assert!(out.len() <= live, "层 {level} 链成环或含游离节点");
    out.push(q);
    p = q;
  }
}

/// 全形态检查：level/arena/free/每层链 与影子模型逐项一致
fn check_shape(
  rl: &RangeLock,
  live: &[(u64, u64, NodeId)],
  free: &VecDeque<NodeId>,
  next_slot: usize,
) {
  let inner = rl.inner.lock();
  assert_eq!(
    inner.level,
    top_nonempty(&inner),
    "level 必须等于最高非空层"
  );
  assert!(inner.level < MAX_LEVEL, "level 越过 MAX_LEVEL");
  assert_eq!(inner.arena.len(), next_slot, "arena 槽位数与影子不一致");
  let actual_free: Vec<NodeId> = inner.free.iter().copied().collect();
  assert_eq!(
    actual_free,
    free.iter().copied().collect::<Vec<_>>(),
    "free 列表顺序/内容不一致"
  );

  let expect: Vec<NodeId> = live.iter().map(|x| x.2).collect();
  assert_eq!(
    chain(&inner, 0, live.len()),
    expect,
    "level-0 链序与影子不一致"
  );
  assert_eq!(
    inner.arena[inner.head as usize].forward[0],
    expect.first().copied().unwrap_or(inner.tail),
    "head 的 level-0 前向与影子首节点不一致"
  );

  let mut on_chain = vec![false; next_slot];
  for id in &expect {
    assert!(
      !on_chain[*id as usize],
      "level-0 链出现重复节点（成环/残留）"
    );
    on_chain[*id as usize] = true;
  }
  let mut is_free = vec![false; next_slot];
  for id in free {
    assert!(!on_chain[*id as usize], "free 槽位仍挂在链上（ABA 污染）");
    is_free[*id as usize] = true;
  }
  for slot in 2..next_slot {
    assert!(
      on_chain[slot] || is_free[slot],
      "槽位 {slot} 既不在链上也不在 free"
    );
  }

  // 每层链必须是 level-0 链的子序列（分层不变），且最高层必须非空
  for k in 0..=inner.level {
    let seq = chain(&inner, k, live.len());
    let mut it = expect.iter();
    for id in &seq {
      assert!(it.any(|e| e == id), "层 {k} 链不是 level-0 链的子序列");
    }
  }
}

/// 层高跟住 log n、增长/收缩两条路径都真实生效、全量释放后收缩到 0
#[test]
fn level_tracks_log_n_and_shrink_is_real() {
  const N: usize = 4096;
  let rl = RangeLock::new(4);
  // 固定层数采样序列：断言对任意种子都成立（见下方概率界），固定后 CI 可复现
  rl.inner.lock().rng = Rng::with_seed(0x5EED_2026);
  let mut live = VecDeque::new();
  let mut s = 1u64;
  let mut grows = 0usize;
  for i in 0..N {
    let before = rl.inner.lock().level;
    assert!(rl.try_lock_range(0, s, 2));
    {
      // 插入后层高必须等于最高非空层：增长分支若不连新顶层（上游写法），此处立即失败
      let inner = rl.inner.lock();
      assert_eq!(
        inner.level,
        top_nonempty(&inner),
        "插入后 level ≠ 最高非空层"
      );
      if inner.level > before {
        grows += 1;
      }
    }
    live.push_back(s);
    s += 4;
    if i > 64
      && i % 2 == 0
      && let Some(old) = live.pop_front()
    {
      let before = rl.inner.lock().level;
      rl.unlock_range(0, old);
      let inner = rl.inner.lock();
      assert_eq!(
        inner.level,
        top_nonempty(&inner),
        "删除后 level ≠ 最高非空层"
      );
      assert!(inner.level <= before);
    }
  }
  let n = live.len();
  // 位长 = floor(log2(n)) + 1；p=1/2 的跳表期望层高 ≈ log2(n)（本实现的"每次插入最多长一层"
  // 使期望值再低约 1 层）。两侧各留 5 层余量：下界失败需所有节点 lv <= 7，P = (1 - 2^-7)^n
  // ≈ e^-16（n≈2080），即 1e-7 量级；上界失败需某节点连续抽中 18 层以上（P = 2^-17/节点）。
  // 任何种子都不会误报；层高冻结（下界）与均匀层数变异（上界）都被击杀。
  let bits = usize::BITS as usize - n.leading_zeros() as usize;
  {
    let inner = rl.inner.lock();
    assert!(
      inner.level + 5 >= bits,
      "level={} 远低于 log2({n})≈{}（层高冻结回归）",
      inner.level,
      bits - 1
    );
    assert!(
      inner.level <= bits + 5,
      "level={} 远高于 log2({n})≈{}（层数分布异常，均匀层数变异的症状）",
      inner.level,
      bits - 1
    );
  }
  assert!(grows > 0, "4096 次插入必然发生层增长");

  // 全量 LIFO 释放：层高逐级收缩到 0，所有槽位回到 free 且互不重复，arena 不缩容。
  // 增长期按 FIFO 释放最老区间，最高层节点（最近增长者）不会被摘，故收缩断言放在这里。
  let mut shrinks = 0usize;
  while let Some(old) = live.pop_back() {
    let before = rl.inner.lock().level;
    rl.unlock_range(0, old);
    let inner = rl.inner.lock();
    assert_eq!(
      inner.level,
      top_nonempty(&inner),
      "全量释放过程中 level ≠ 最高非空层"
    );
    if inner.level < before {
      shrinks += 1;
    }
  }
  assert!(shrinks > 0, "层收缩路径必须真实生效");
  let inner = rl.inner.lock();
  assert_eq!(inner.level, 0, "空表必须收缩到 level 0");
  assert_eq!(inner.arena[inner.head as usize].forward[0], inner.tail);
  let arena_len = inner.arena.len();
  assert!(
    arena_len <= N + 2,
    "arena 槽位数不得超过插入次数 + 2 个哨兵"
  );
  assert_eq!(
    inner.free.len(),
    arena_len - 2,
    "全量释放后所有槽位都必须回到 free"
  );
  let mut seen = vec![false; arena_len];
  for &id in &inner.free {
    assert!(!seen[id as usize], "槽位 {id} 在 free 中重复（双重入栈）");
    seen[id as usize] = true;
  }
  assert!((2..arena_len).all(|i| seen[i]), "仍有槽位未回到 free");
}

/// 随机操作 vs `Vec<(start, end, slot)>` 影子：链序、free 复用（ABA）、零长度/端点相等/
/// 包含/u64::MAX 全覆盖
#[test]
fn arena_free_list_and_chain_stay_consistent() {
  const OPS: usize = 3000;
  let rl = RangeLock::new(4);
  let mut rng = Rng::with_seed(20261111);
  // 影子按 level-0 链序保存（start 升序；同 start 的退化区间新者在前）
  let mut live: Vec<(u64, u64, NodeId)> = Vec::new();
  let mut free: VecDeque<NodeId> = VecDeque::new();
  let mut next_slot = 2usize; // head=0 / tail=1

  for step in 0..OPS {
    if !live.is_empty() && rng.u8(..100) < 40 {
      // 解锁：60% 取已持有起点，40% 取任意起点（含未持有 ⇒ no-op 路径）
      let start = if rng.bool() {
        live[rng.usize(..live.len())].0
      } else {
        match rng.u8(..10) {
          0 => u64::MAX - rng.u64(..4),
          1 => 1 + rng.u64(..3),
          _ => 1 + rng.u64(..3000),
        }
      };
      rl.unlock_range(0, start);
      let idx = live.partition_point(|x| x.0 < start);
      if idx < live.len() && live[idx].0 == start {
        let slot = live.remove(idx).2;
        free.push_back(slot);
      }
      assert_eq!(
        rl.inner.lock().free.iter().copied().collect::<Vec<_>>(),
        free.iter().copied().collect::<Vec<_>>(),
        "step {step}: unlock 后 free 列表不一致"
      );
    } else {
      // 起点/长度覆盖：MAX 邻域、最小起点、密集小域（重叠/包含/端点相等）、零长度
      let start = match rng.u8(..10) {
        0 => u64::MAX - rng.u64(..4),
        1 => 1 + rng.u64(..3),
        _ => 1 + rng.u64(..3000),
      };
      let len = match rng.u8(..10) {
        0 => 0,
        1 => 1,
        2 => rng.u64(..2000),
        _ => rng.u64(..10),
      };
      let end = range_end(start, len);
      // 独立谓词：线性扫影子（不看跳表内部）
      let conflict = live.iter().any(|&(s, e, _)| s <= end && e >= start);
      let (front, len_before) = {
        let inner = rl.inner.lock();
        (inner.free.front().copied(), inner.arena.len())
      };
      if rng.u8(..100) < 70 {
        let ok = rl.try_lock_range(0, start, len);
        assert_eq!(
          ok, !conflict,
          "step {step}: lock [{start},{end}] 期望冲突={conflict} 实得 {ok}"
        );
        if ok {
          let (slot, reused) = match front {
            Some(id) => (id, true),
            None => (len_before as NodeId, false),
          };
          if reused {
            assert_eq!(
              free.pop_front(),
              Some(slot),
              "step {step}: 影子 free 与跳表分配不一致"
            );
          } else {
            next_slot += 1;
            assert_eq!(next_slot, len_before + 1);
          }
          let idx = live.partition_point(|x| x.0 <= end);
          live.insert(idx, (start, end, slot));
          let inner = rl.inner.lock();
          assert_eq!(
            (
              inner.arena[slot as usize].start,
              inner.arena[slot as usize].end
            ),
            (start, end)
          );
        } else {
          assert_eq!(front, rl.inner.lock().free.front().copied());
          assert_eq!(len_before, rl.inner.lock().arena.len());
        }
      } else {
        let free_now = rl.test_range(0, start, len);
        assert_eq!(
          free_now, !conflict,
          "step {step}: test [{start},{end}] 失配"
        );
        assert_eq!(front, rl.inner.lock().free.front().copied());
        assert_eq!(len_before, rl.inner.lock().arena.len());
      }
    }
    if step % 200 == 0 {
      check_shape(&rl, &live, &free, next_slot);
    }
  }
  check_shape(&rl, &live, &free, next_slot);
}

/// `u32` 节点 id 上限：越过 2^32 必须显式 panic，不得静默截断成 head 的 id 0
#[test]
fn node_id_capacity_overflow_panics() {
  assert_eq!(Inner::append_id(2), 1);
  // 字面量钉死 u32 索引上限（不用 NodeId::MAX 自我参照）
  assert_eq!(MAX_NODES, 4_294_967_295);
  assert_eq!(Inner::append_id(MAX_NODES), 4_294_967_294);
  #[cfg(target_pointer_width = "64")]
  {
    let r = catch_unwind(|| Inner::append_id(4_294_967_296));
    assert!(r.is_err(), "越过 u32 索引上限必须 panic，不能静默截断");
  }
}
