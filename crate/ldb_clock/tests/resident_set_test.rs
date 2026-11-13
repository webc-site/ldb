//! `ldb_clock::ResidentPageSet` 回归：上游对拍、边界、墓碑复用、幂等与并发。
//!
//! 对拍基准：`ZLeanStore/src/buffer/resident_set.cc`（md5 48e3d47b…）仅把
//! `Insert` 换成两遍探测后的 C++ 参照（该函数上游存在重复插入缺陷，见模块注释），
//! 其余函数逐字未改；`tests/data/upstream_fixed_*.txt` 为其输出的 8192 操作转录，
//! 本文件用同一 xorshift 序列逐字节比对。

use std::{
  sync::{Arc, Barrier, atomic::Ordering},
  thread,
};

use gxhash::gxhash64;
use ldb_clock::{EMPTY, ResidentPageSet, TOMBSTONE};
use ldb_type::{PageState, PageStateMode, same_version_new_state};

/// 与 `resident_set::HASH_SEED` 相同的固定种子（测试侧独立计算期望槽位）
const SEED: i64 = 0x1db;

/// 全部页状态置 EXCLUSIVE（插入契约：调用方持页排他闩）
fn locked_states(n: u64) -> Arc<[PageState]> {
  (0..n)
    .map(|_| {
      let st = PageState::new();
      st.store(
        same_version_new_state(0, PageStateMode::EXCLUSIVE),
        Ordering::Relaxed,
      );
      st
    })
    .collect::<Vec<_>>()
    .into()
}

fn table(cap: u64, pages: u64, shards: u32) -> ResidentPageSet {
  ResidentPageSet::new(cap, locked_states(pages), shards)
}

/// 独立复算槽位（gxhash + 固定种子 + 掩码 + 与实现相同的字节序）
fn slot_of(rs: &ResidentPageSet, pid: u64) -> u64 {
  gxhash64(&pid.to_le_bytes(), SEED) & (rs.count() - 1)
}

/// 与上游 C++ 参照相同的操作转录：ops 个 随机 Insert/Remove/Contain + 全量扫描
fn transcript(cap: u64, ops: u64, seed: u64) -> String {
  let rs = table(cap, cap, 1);
  let mut s = seed;
  let mut out = String::with_capacity((ops + cap + 2) as usize);
  for _ in 0..ops {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    let res = match s % 3 {
      0 => rs.insert((s >> 8) % cap),
      1 => rs.remove((s >> 8) % cap),
      _ => rs.contain((s >> 8) % cap),
    };
    out.push(if res { '1' } else { '0' });
  }
  out.push('\n');
  for pid in 0..cap {
    out.push(if rs.contain(pid) { '1' } else { '0' });
  }
  out.push('\n');
  out
}

#[test]
fn upstream_transcript_matches_cpp_reference() {
  // 三个容量覆盖 count=16/256/1024；转录由修正 Insert 的 C++ 参照生成。
  // 上游单遍 Insert 的重复项缺陷在此金标下必然暴露：cap=8/seed=1 第 115 个操作起
  // （同 home 的键被墓碑截断后二次插入）即与上游逐字背离。
  assert_eq!(
    transcript(8, 8192, 1),
    include_str!("data/upstream_fixed_cap8_ops8192_s1.txt")
  );
  assert_eq!(
    transcript(64, 8192, 7),
    include_str!("data/upstream_fixed_cap64_ops8192_s7.txt")
  );
  assert_eq!(
    transcript(569, 8192, 3),
    include_str!("data/upstream_fixed_cap569_ops8192_s3.txt")
  );
}

#[test]
fn insert_idempotent_over_whole_capacity() {
  let rs = table(64, 128, 1);
  assert_eq!(rs.count(), 128);
  assert!(!rs.contain(100), "空表查询");
  assert!(!rs.remove(100), "空表删除不存在页");
  for pid in 0..64 {
    assert!(rs.insert(pid), "首次插入 {pid}");
  }
  for pid in 0..64 {
    assert!(!rs.insert(pid), "幂等：二次插入 {pid} 必须 false");
  }
  for pid in 0..64 {
    assert!(rs.contain(pid));
  }
  for pid in 0..64 {
    assert!(rs.remove(pid), "删除 {pid}");
  }
  for pid in 0..64 {
    assert!(!rs.contain(pid), "删除后不可见 {pid}");
  }
  for pid in 0..64 {
    assert!(rs.insert(pid), "墓碑复用 {pid}");
  }
  for pid in 0..64 {
    assert!(rs.contain(pid));
  }
}

#[test]
fn capacity_one_full_table_and_100pct_tombstone() {
  // capacity=1 ⇒ count=1（上游 1.8 截断）；shards=4 大于槽数也不得 panic
  let rs = table(1, 8, 4);
  assert_eq!(rs.capacity(), 1);
  assert_eq!(rs.count(), 1);
  assert!(rs.insert(0));
  assert!(!rs.insert(1), "满表插入不存在页：上游自旋，Rust 返回 false");
  assert!(rs.contain(0));
  assert!(!rs.contain(1));
  assert!(!rs.remove(1), "满表删除不存在页：走完一圈返回 false");
  assert!(rs.remove(0));
  assert!(!rs.contain(0), "100% 墓碑后不可见");
  assert!(rs.insert(1), "墓碑复用");
  assert!(rs.contain(1));
}

#[test]
fn count_formula_matches_upstream_trunc_then_bit_ceil() {
  // 上游 resident_set.cc:19：bit_ceil(static_cast<u64>(capacity * 1.8))
  let cases = [
    (1u64, 1u64),
    (2, 4),
    (8, 16),
    (9, 16),  // 若写成 ceil(1.8*c) 会得到 32
    (18, 32), // ceil 得 64
    (36, 64), // ceil 得 128
    (100, 256),
    (128, 256),
    (569, 1024), // ceil 得 2048
  ];
  for (cap, count) in cases {
    let rs = table(cap, cap, 1);
    assert_eq!(rs.capacity(), cap);
    assert_eq!(rs.count(), count, "capacity={cap}");
  }
}

#[test]
#[should_panic(expected = "页号空间容量必须为正")]
fn capacity_zero_rejected() {
  let _ = table(0, 8, 1);
}

#[test]
fn reserved_pids_never_enter_table() {
  assert_eq!(EMPTY, u64::MAX);
  assert_eq!(TOMBSTONE, u64::MAX - 1);
  let rs = table(8, 2048, 1);
  assert!(!rs.insert(EMPTY));
  assert!(!rs.insert(TOMBSTONE), "无守卫时 TOMBSTONE 会被 CAS 进空槽");
  assert!(!rs.contain(EMPTY));
  assert!(!rs.contain(TOMBSTONE));
  assert!(!rs.remove(EMPTY));
  assert!(!rs.remove(TOMBSTONE));
  // 在 TOMBSTONE 的 home 槽制造一个墓碑：无守卫时 contain/remove(TOMBSTONE)
  // 会把这个墓碑当成命中
  let home = slot_of(&rs, TOMBSTONE);
  let Some(pid) = (0..2048u64).find(|&pid| slot_of(&rs, pid) == home) else {
    panic!("pids 0..2048 中必有 home 槽碰撞");
  };
  assert!(rs.insert(pid));
  assert!(rs.remove(pid));
  assert!(!rs.contain(TOMBSTONE));
  assert!(!rs.remove(TOMBSTONE));
  // 保留值不得破坏表：正常页仍可插入/查询
  assert!(rs.insert(0));
  assert!(rs.contain(0));
}

#[test]
fn insert_rejects_key_displaced_past_tombstone() {
  // 上游单遍 Insert 的重复项缺陷最小复现：a、b 同 home 槽，
  // a 占 home、b 被挤到 home+1；删 a 留墓碑后，再次 insert(b) 必须 false。
  // 单遍实现会 CAS 墓碑，把 b 插成两份（remove 只摘一份 ⇒ 常驻表残留幽灵页）。
  let rs = table(569, 4096, 1);
  let mut groups: Vec<Vec<u64>> = vec![Vec::new(); rs.count() as usize];
  for pid in 0..4096u64 {
    groups[slot_of(&rs, pid) as usize].push(pid);
  }
  let Some(pair) = groups.into_iter().find(|group| group.len() >= 2) else {
    panic!("4096 个 pid 必有 2 个同 home 槽");
  };
  let (a, b) = (pair[0], pair[1]);
  assert!(rs.insert(a));
  assert!(rs.insert(b), "b 被 a 挤到下一个槽");
  assert!(rs.remove(a), "home 槽留下墓碑");
  assert!(!rs.insert(b), "已被挤到墓碑之后的键不得重复插入");
  // 全表一圈统计 b 的槽位数必须恰为 1（重复项会让 clock 重复回调同一页）
  let mut copies = 0u32;
  rs.iterate_clock_batch(rs.count(), |pid| copies += u32::from(pid == b));
  assert_eq!(copies, 1, "b 只能占一个槽");
  assert!(rs.remove(b));
  assert!(!rs.contain(b), "单次删除必须摘除唯一副本");
}

#[test]
fn clock_batch_exceeding_count_repeats_pids() {
  // 上游同语义（resident_set.cc:135-143 的 for idx < batch）：
  // batch > count 时绕圈重复回调同一页；调用方须按幂等吸收。
  let rs = table(2, 8, 1);
  assert_eq!(rs.count(), 4);
  assert!(rs.insert(0));
  assert!(rs.insert(1));
  let mut seen = Vec::new();
  rs.iterate_clock_batch(8, |pid| seen.push(pid));
  seen.sort_unstable();
  assert_eq!(seen, vec![0, 0, 1, 1], "batch=2*count ⇒ 每页恰好两次");
}

#[test]
fn clock_batch_skips_tombstones_and_empties() {
  let rs = table(71, 71, 1);
  for pid in 0..64 {
    assert!(rs.insert(pid));
  }
  for pid in (0..64).step_by(2) {
    assert!(rs.remove(pid));
  }
  let mut seen = Vec::new();
  for _ in 0..16 {
    rs.iterate_clock_batch(8, |pid| seen.push(pid));
  }
  seen.sort_unstable();
  assert_eq!(seen, (1..64).step_by(2).collect::<Vec<_>>());
}

#[test]
fn clock_batch_iterates_reserved_window() {
  let rs = table(71, 71, 1);
  assert_eq!(rs.count(), 128);
  let pids: Vec<u64> = (0..64).collect();
  for &pid in &pids {
    assert!(rs.insert(pid));
  }
  // 构造后 clock_pos=0：4×16 批只应遍历 [0,64)，上游 C-22 漂移会从 [16,80) 开始
  let mut got = Vec::new();
  for _ in 0..4 {
    rs.iterate_clock_batch(16, |pid| got.push(pid));
  }
  let mut expected: Vec<(u64, u64)> = pids
    .iter()
    .map(|&pid| (slot_of(&rs, pid), pid))
    .filter(|&(slot, _)| slot < 64)
    .collect();
  expected.sort_unstable();
  let expected: Vec<u64> = expected.into_iter().map(|(_, pid)| pid).collect();
  assert_eq!(got, expected);
}

#[test]
fn get_pid_walks_cp_pos() {
  let rs = table(71, 71, 1);
  assert_eq!(rs.get_pid(0), TOMBSTONE);
  assert_eq!(rs.get_pid_to_evict(0), TOMBSTONE);
  let pid = 42u64;
  assert!(rs.insert(pid));
  let home = slot_of(&rs, pid);
  // 构造后 cp_pos/clock_pos = 0：偏移 home 命中该页
  assert_eq!(rs.get_pid(home), pid);
  assert_eq!(rs.get_pid(0), pid, "CAS 后 cp_pos 已指向命中槽");
  assert_eq!(rs.get_pid_to_evict(home), pid);
  assert_eq!(rs.get_pid_to_evict(0), pid, "CAS 后 clock_pos 已指向命中槽");
  // 删除后同位置返回 TOMBSTONE（不是 EMPTY）
  assert!(rs.remove(pid));
  assert_eq!(rs.get_pid(0), TOMBSTONE);
  assert_eq!(rs.get_pid_to_evict(0), TOMBSTONE);
  // 大偏移不得在 debug 下整型溢出 panic
  let v = rs.get_pid(u64::MAX);
  assert!(v == TOMBSTONE || v == pid);
  let v = rs.get_pid_to_evict(u64::MAX);
  assert!(v == TOMBSTONE || v == pid);
}

#[test]
fn concurrent_insert_remove_no_loss_or_duplicate() {
  const THREADS: u64 = 4;
  const PER: u64 = 512;
  const ROUNDS: u64 = 2000;
  let rs = table(4096, THREADS * PER, 1);
  let barrier = Barrier::new(THREADS as usize);
  thread::scope(|scope| {
    for t in 0..THREADS {
      let rs = &rs;
      let barrier = &barrier;
      scope.spawn(move || {
        let base = t * PER;
        barrier.wait();
        for round in 0..ROUNDS {
          let pid = base + round % PER;
          assert!(rs.insert(pid), "并发插入 {pid}");
          assert!(rs.contain(pid), "并发可见 {pid}");
          assert!(rs.remove(pid), "并发删除 {pid}");
          assert!(!rs.contain(pid), "并发删除后不可见 {pid}");
        }
        for pid in base..base + PER {
          assert!(rs.insert(pid), "终态插入 {pid}");
        }
      });
    }
  });
  for pid in 0..THREADS * PER {
    assert!(rs.contain(pid), "丢失常驻页 {pid}");
  }
  // 只删一次即消失 ⇒ 无重复项
  for pid in 0..THREADS * PER {
    assert!(rs.remove(pid), "终态删除 {pid}");
    assert!(!rs.contain(pid), "重复项仍可见 {pid}");
  }
}

#[test]
fn concurrent_insert_same_probe_chain() {
  // 4 个 pid 共享同一 home 槽：并发插入必须靠 CAS 失败重扫各占一槽，不得互相覆盖
  let rs = table(569, 4096, 1);
  assert_eq!(rs.count(), 1024);
  let mut groups: Vec<Vec<u64>> = vec![Vec::new(); 1024];
  for pid in 0..4096u64 {
    groups[slot_of(&rs, pid) as usize].push(pid);
  }
  let Some(group) = groups.into_iter().find(|group| group.len() >= 8) else {
    panic!("4096 个 pid 必有 8 个同 home 槽");
  };
  let pids: Vec<u64> = group.into_iter().take(4).collect();
  let barrier = Barrier::new(pids.len());
  for round in 0..100 {
    thread::scope(|scope| {
      for &pid in &pids {
        let rs = &rs;
        let barrier = &barrier;
        scope.spawn(move || {
          // 先对齐到同一时刻再插入：把「同 home 并发落位」的竞争窗口放到最大
          // （barrier 放在 insert 之后会让线程启动抖动主导，CAS→store 变异
          //  仅在极小概率下才丢项，无法稳定作为 E1 的击杀证据）
          barrier.wait();
          assert!(rs.insert(pid));
        });
      }
    });
    for &pid in &pids {
      assert!(rs.contain(pid), "第 {round} 轮并发插入丢失 {pid}");
    }
    for &pid in &pids {
      assert!(rs.remove(pid));
      assert!(!rs.contain(pid));
    }
  }
}
