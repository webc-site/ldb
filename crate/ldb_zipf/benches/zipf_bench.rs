//! ldb_zipf 微基准（无第三方依赖）：ns/op 与 alloc/op。
//!
//! `cargo bench -p ldb_zipf` 直接运行；输出走 stderr。
//! 三个采样器都在值域 n=1e6 下测量（YCSB 典型量级）。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  io::{Write as _, stderr},
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use fastrand::Rng;
use ldb_zipf::{RejectionInversionZipfSampler, ScrambledZipfGenerator, ZipfGenerator};

/// 计数分配器：只统计次数，转发给 System
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: 所有方法只做计数后原样转发给 System，语义与 System 完全一致。
unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: 转发给 System，layout 由调用方保证合法。
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: ptr/layout 来自本分配器的 alloc，原样转发。
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: 转发给 System，ptr/layout 来自本分配器的 alloc。
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocs() -> usize {
  ALLOCS.load(Ordering::Relaxed)
}

/// 返回 (ns/op, allocs/op)。alloc 统计区间不含任何输出。
fn bench<F: FnMut() -> u64>(name: &str, iters: u32, mut f: F) {
  for _ in 0..(iters / 10).max(1) {
    black_box(f());
  }
  let a0 = allocs();
  let t = Instant::now();
  let mut acc = 0u64;
  for _ in 0..iters {
    acc = acc.wrapping_add(black_box(f()));
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(iters);
  let ap = (allocs() - a0) as f64 / f64::from(iters);
  black_box(acc);
  let _ = writeln!(stderr(), "{name:<44}{ns:9.2} ns/op{ap:8.3} alloc/op");
}

fn main() {
  const N: u64 = 1_000_000;
  let mut rng = Rng::with_seed(0x5eed_1234);

  // 构造成本：zeta 为 O(n) 次 powf（上游同）
  for theta in [0.5f64, 0.99] {
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..10 {
      acc = acc.wrapping_add(black_box(ZipfGenerator::new(N, theta).rand(&mut rng)));
    }
    let ns = t.elapsed().as_nanos() as f64 / 10.0;
    black_box(acc);
    let _ = writeln!(
      stderr(),
      "{:<44}{ns:9.0} ns/构造 (n=1e6, θ={theta})",
      "ZipfGenerator::new + rand"
    );
  }

  for theta in [0.5f64, 0.99, 1.2] {
    let g = ZipfGenerator::new(N + 1, theta);
    bench(
      &format!("ZipfGenerator::rand θ={theta}"),
      10_000_000,
      || g.rand(&mut rng),
    );
    let s = RejectionInversionZipfSampler::new(N, theta);
    bench(
      &format!("RejectionInversionZipfSampler::sample θ={theta}"),
      10_000_000,
      || s.sample(&mut rng),
    );
  }

  let sc = ScrambledZipfGenerator::new(0, N, 0.99);
  bench("ScrambledZipfGenerator::rand θ=0.99", 10_000_000, || {
    sc.rand(&mut rng)
  });
}
