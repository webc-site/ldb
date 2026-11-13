//! ldb_blob 热路径基准：分配（去重 miss/hit）/ 常驻加载 / ExtentList 追加。
//!
//! 用 `NoBlobStore`（常驻内存态）隔离纯内存开销；异步入口用就绪轮询
//! `block_on`（NoBlobStore 的 future 全部立即 Ready，无需真实执行器）。

use std::{
  future::Future,
  hint::black_box,
  pin::pin,
  task::{Context, Poll, Waker},
};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ldb_blob::{BlobManager, ExtentList, NoBlobStore};
use ldb_type::PAGE_SIZE;

/// 就绪轮询执行器（NoBlobStore 的 future 不真正挂起）。
fn block_on<F: Future>(fut: F) -> F::Output {
  let waker = Waker::noop();
  let mut cx = Context::from_waker(waker);
  let mut fut = pin!(fut);
  loop {
    if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
      return v;
    }
  }
}

fn payload(len: usize, seed: u8) -> Vec<u8> {
  (0..len).map(|i| (i as u8) ^ seed).collect()
}

fn bench_allocate(c: &mut Criterion) {
  let mut g = c.benchmark_group("allocate_blob");
  for kib in [4usize, 64, 1024] {
    let bytes = kib * 1024;
    g.throughput(Throughput::Bytes(bytes as u64));
    // 去重 miss：每次全新内容（构造不计时）
    g.bench_with_input(BenchmarkId::new("dedup_miss", kib), &bytes, |b, &bytes| {
      let mgr = BlobManager::<NoBlobStore>::new(64);
      let mut n = 0u8;
      b.iter_batched(
        || {
          n = n.wrapping_add(1);
          payload(bytes, n)
        },
        |p| {
          black_box(block_on(mgr.allocate_blob(&p)).unwrap());
        },
        BatchSize::SmallInput,
      );
    });
    // 去重 hit：同内容反复分配（命中 dedup）
    g.bench_with_input(BenchmarkId::new("dedup_hit", kib), &bytes, |b, &bytes| {
      let mgr = BlobManager::<NoBlobStore>::new(64);
      let p = payload(bytes, 7);
      block_on(mgr.allocate_blob(&p)).unwrap();
      b.iter(|| {
        black_box(block_on(mgr.allocate_blob(&p)).unwrap());
      });
    });
  }
  g.finish();
}

fn bench_load_resident(c: &mut Criterion) {
  let mut g = c.benchmark_group("load_blob_resident");
  for kib in [4usize, 64, 1024] {
    let bytes = kib * 1024;
    g.throughput(Throughput::Bytes(bytes as u64));
    let mgr = BlobManager::<NoBlobStore>::new(64);
    let p = payload(bytes, 3);
    let id = block_on(mgr.allocate_blob(&p)).unwrap();
    g.bench_with_input(BenchmarkId::from_parameter(kib), &bytes, |b, _| {
      b.iter(|| {
        black_box(block_on(mgr.load_blob(id)).unwrap());
      });
    });
  }
  g.finish();
}

fn bench_extent_append(c: &mut Criterion) {
  let mut g = c.benchmark_group("extent_list_append");
  g.throughput(Throughput::Bytes(PAGE_SIZE as u64));
  g.bench_function("4KiB", |b| {
    b.iter_batched(
      || ExtentList::new(0, 64),
      |el| {
        // 恰好一整页：不触发拆级，仅尾缓冲追加
        black_box(el.append(&[0u8; PAGE_SIZE]).expect("未越上限"));
      },
      BatchSize::SmallInput,
    );
  });
  g.finish();
}

criterion_group!(
  benches,
  bench_allocate,
  bench_load_resident,
  bench_extent_append
);
criterion_main!(benches);
