//! EpochHandler 热路径基准：enter / leave / defer / epoch_operation。

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use ldb_epoch::EpochHandler;

fn bench_ops(c: &mut Criterion) {
  let h = EpochHandler::new(4);

  // 同一地址的重复原子写会被 LLVM 合并/提升出循环（单测 enter/leave 分别 0.7ns/0.4ns 即被优化掉），
  // 因此在 enter/leave 之间读一次本地 epoch 并 black_box，形成真实依赖链。
  c.bench_function("epoch/enter_leave", |b| {
    b.iter(|| {
      let h = black_box(&h);
      h.enter(black_box(0));
      black_box(h.local(black_box(0)));
      h.leave(black_box(0));
      black_box(h.local(black_box(0)));
    });
  });

  c.bench_function("epoch/defer", |b| {
    // 先进入保护区间，使入队 epoch=0；每 1024 次清理一次，避免队列无限增长
    h.enter(0);
    let mut n = 0usize;
    b.iter(|| {
      h.defer_free_pointer(0, Box::new(black_box(1u64)));
      n += 1;
      if n & 1023 == 0 {
        h.leave(0);
        h.epoch_operation(0, &mut |_| 0);
        h.enter(0);
      }
    });
  });

  // 触发回收扫描：4 个 worker 取 min + 空队列
  c.bench_function("epoch/operation_scan", |b| {
    b.iter(|| {
      let mut rng = |bound: u64| if bound == 200 { 1 } else { 0 };
      h.epoch_operation(black_box(0), &mut rng);
    });
  });

  // 触发推进：全局 epoch fetch_add（1/200 路径）
  c.bench_function("epoch/operation_advance", |b| {
    b.iter(|| {
      let mut rng = |bound: u64| if bound == 200 { 0 } else { 1 };
      h.epoch_operation(black_box(0), &mut rng);
    });
  });

  // 回收 1 个条目的完整路径（含 Box drop）
  c.bench_function("epoch/operation_reclaim_one", |b| {
    b.iter_batched(
      || {
        h.enter(0);
        h.defer_free_pointer(0, Box::new(black_box(7u64)));
        h.leave(0);
      },
      |()| {
        h.epoch_operation(0, &mut |_| 0);
      },
      BatchSize::SmallInput,
    );
  });
}

/// 回收侧 min 扫描随 worker 数的代价：`local_epoch` 每槽独占一行（128B），
/// 256 worker 时一次扫描触达 32KiB。上游同样线性扫描全部 worker
/// （`epoch_handler.cc:21-22`），此处量化本实现的绝对成本。
fn bench_scan_workers(c: &mut Criterion) {
  let mut group = c.benchmark_group("epoch/operation_scan_workers");
  for workers in [4usize, 16, 64, 256] {
    let h = EpochHandler::new(workers);
    group.bench_with_input(BenchmarkId::from_parameter(workers), &workers, |b, _| {
      // 推进采样非 0（不推进），回收采样为 0（进入 min 扫描 + 空队列锁）
      let mut rng = |bound: u64| if bound == 200 { 1 } else { 0 };
      b.iter(|| h.epoch_operation(black_box(0), &mut rng));
    });
  }
  group.finish();
}

criterion_group!(benches, bench_ops, bench_scan_workers);
criterion_main!(benches);
