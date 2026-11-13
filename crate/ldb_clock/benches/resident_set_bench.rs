//! `ResidentPageSet` 微基准：insert / contain / remove 的 ns 级证据（清单 C2）。
//!
//! 全部路径为纯内存无锁原子操作；基准只测稳定态（表已按构造容量预填）。

use std::{
  hint::black_box,
  sync::{Arc, atomic::Ordering},
};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ldb_clock::ResidentPageSet;
use ldb_type::{PageState, PageStateMode, same_version_new_state};

/// 预填：构造容量个页，页状态置 EXCLUSIVE（调用方持页闩后登记常驻的契约）
fn setup(cap: u64) -> ResidentPageSet {
  let ps: Arc<[PageState]> = (0..cap)
    .map(|_| {
      let st = PageState::new();
      st.store(
        same_version_new_state(0, PageStateMode::EXCLUSIVE),
        Ordering::Relaxed,
      );
      st
    })
    .collect::<Vec<_>>()
    .into();
  let rs = ResidentPageSet::new(cap, ps, 1);
  for pid in 0..cap {
    assert!(rs.insert(pid));
  }
  rs
}

fn bench(c: &mut Criterion) {
  const CAP: u64 = 1 << 20;
  let rs = setup(CAP);
  let pids: Vec<u64> = (0..CAP).collect();

  let mut group = c.benchmark_group("resident_set");
  group.throughput(Throughput::Elements(CAP));

  group.bench_function("contain_hit", |b| {
    b.iter(|| {
      let mut hits = 0u64;
      for &pid in &pids {
        hits += u64::from(rs.contain(black_box(pid)));
      }
      black_box(hits)
    });
  });

  group.bench_function("contain_miss", |b| {
    b.iter(|| {
      let mut hits = 0u64;
      for &pid in &pids {
        hits += u64::from(rs.contain(black_box(pid + CAP)));
      }
      black_box(hits)
    });
  });

  // 驱逐 + 缺页回填的稳定态循环：墓碑复用路径（remove→insert 同一 pid）
  const CYCLE: usize = 1024;
  group.throughput(Throughput::Elements(CYCLE as u64));
  group.bench_function("remove_insert_cycle", |b| {
    b.iter(|| {
      for &pid in &pids[..CYCLE] {
        assert!(rs.remove(black_box(pid)));
        assert!(rs.insert(black_box(pid)));
      }
    });
  });

  // 同上但跨全表随机取 pid（真实驱逐/回填：两次随机访问 = 两次缓存未命中，
  // 而非上面只打前 1024 个热槽）。两遍探测相对单遍多走「首个墓碑→EMPTY」段。
  group.bench_function("remove_insert_random", |b| {
    let mut s = 0x243f_6a88_85a3_08d3u64;
    b.iter(|| {
      for _ in 0..CYCLE {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let pid = (s >> 8) % CAP;
        assert!(rs.remove(black_box(pid)));
        assert!(rs.insert(black_box(pid)));
      }
    });
  });

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
