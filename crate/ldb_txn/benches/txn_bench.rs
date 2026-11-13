//! OSIC 可见性 / CommitTree 热路径基准。

use std::{hint::black_box, sync::atomic::Ordering};

use criterion::{Criterion, criterion_group, criterion_main};
use ldb_txn::{ConcurrencyControl, IsolationLevel, new_transaction};

fn bench_visibility(c: &mut Criterion) {
  let cc = ConcurrencyControl::new(4);
  let mut me = new_transaction(1000, IsolationLevel::SnapshotIsolation);
  me.initialize(1000, IsolationLevel::SnapshotIsolation);
  // w1 预置提交
  let s = cc.global_clock.fetch_add(1, Ordering::Relaxed);
  cc.workers[1].commit_tree.commit(s, &cc.global_clock);
  c.bench_function("osic/si_hit", |b| {
    b.iter(|| {
      black_box(cc.is_visible_for_me(
        black_box(0),
        black_box(1),
        black_box(s),
        false,
        black_box(&me),
      ))
    })
  });
}

fn bench_commit_tree(c: &mut Criterion) {
  let cc = ConcurrencyControl::new(1);
  c.bench_function("commit_tree/commit+lcb", |b| {
    b.iter(|| {
      let ct = &cc.workers[0].commit_tree;
      let start = cc.global_clock.fetch_add(1, Ordering::Relaxed);
      black_box(ct.commit(start, &cc.global_clock));
      black_box(ct.lcb(black_box(start)));
    })
  });
}

criterion_group!(benches, bench_visibility, bench_commit_tree);
criterion_main!(benches);
