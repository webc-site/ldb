//! SSD 模拟器热路径基准：单次主机写 / 块压缩 / WL 写的每操作开销（ns 级）。
//!
//! 目的（清单 C2）：量化「模拟器自身开销」，确保基准测量不被模拟器主导。
//! 每轮用 `iter_batched_ref` 重建一个进入目标状态的模拟器（`&mut Ssd` 复用，析构不计入
//! 计时），计时段只含被测单次操作。
//! 8 MiB / 64 KiB 区块 / 4 KiB 页 = 128 块 × 16 页；填充 50% = 1024 逻辑页。

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ldb_testkit::Ssd;

/// 测试用模拟器（与 tests/ssd_test.rs 的 `sim()` 同参数）
fn sim() -> Ssd {
  Ssd::new(8 << 20, 64 << 10, 4096, 0.5)
}

/// 建一个「块 0 留 1 空槽」的状态，供单次写计时
fn armed_block0() -> Ssd {
  let s = sim();
  for p in 0..15u64 {
    s.write_page_without_caching(p, 0, -1);
  }
  s
}

fn bench_host_write(c: &mut Criterion) {
  c.bench_function("ssd/host_write_1page", |b| {
    b.iter_batched_ref(
      armed_block0,
      |s| s.write_page_without_caching(black_box(15), 0, -1),
      BatchSize::SmallInput,
    );
  });
}

fn bench_compact(c: &mut Criterion) {
  c.bench_function("ssd/compact_block_15valid", |b| {
    b.iter_batched_ref(
      armed_block0,
      |s| s.compact_block(black_box(0)),
      BatchSize::SmallInput,
    );
  });
}

fn bench_wl_write(c: &mut Criterion) {
  c.bench_function("ssd/wl_write_1page", |b| {
    b.iter_batched_ref(
      || {
        let s = sim();
        s.set_wear_leveling(true);
        s.set_wear_leveling_threshold(1024);
        s
      },
      |s| s.write_page_wl(black_box(0), -1),
      BatchSize::SmallInput,
    );
  });
}

criterion_group!(benches, bench_host_write, bench_compact, bench_wl_write);
criterion_main!(benches);
