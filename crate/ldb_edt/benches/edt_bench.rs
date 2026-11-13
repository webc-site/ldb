//! ldb_edt 微基准（无第三方依赖）：ns/op 与 alloc/op。
//!
//! `cargo bench -p ldb_edt` 直接运行；输出走 stderr。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  io::{Write as _, stderr},
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use ldb_edt::{WriteHistory, estimate_death_time, estimate_next_gc_write, get_interval_gap};
use ldb_type::WRITE_HISTORY_LENGTH;

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
    // SAFETY: 转发给 System，ptr/layout 由调用方保证合法。
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocs() -> usize {
  ALLOCS.load(Ordering::Relaxed)
}

/// 返回（ns/op, allocs/op）。alloc 统计区间不含任何输出。
fn bench<F: FnMut() -> u64>(name: &str, iters: u32, mut f: F) -> (f64, f64) {
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
  let _ = writeln!(stderr(), "{name:<52}{ns:9.2} ns/op{ap:8.2} alloc/op");
  (ns, ap)
}

/// 旧实现（排序法）基线：栈上收集 + 插入排序，用于量化本轮「免排序」改写
fn old_sort_based(hist: &WriteHistory) -> u64 {
  let mut valid = [0u64; WRITE_HISTORY_LENGTH];
  let mut n = 0usize;
  for &g in hist {
    if g != 0 {
      let mut i = n;
      while i > 0 && valid[i - 1] > g {
        valid[i] = valid[i - 1];
        i -= 1;
      }
      valid[i] = g;
      n += 1;
    }
  }
  match n {
    0 => 0,
    1 => valid[0],
    _ => {
      let total: u64 = valid[1..n]
        .iter()
        .zip(valid[..n - 1].iter())
        .map(|(hi, lo)| hi - lo)
        .sum();
      valid[n - 1].wrapping_add(total / (n as u64 - 1))
    }
  }
}

fn main() {
  let full: WriteHistory = black_box([8, 1, 7, 2, 6, 3, 5, 4]);
  let sparse: WriteHistory = black_box([20, 0, 10, 0, 40, 0, 0, 0]);
  let empty: WriteHistory = [0; 8];

  let (new_full, a1) = bench("estimate_death_time full8", 50_000_000, || {
    estimate_death_time(black_box(&full))
  });
  let (new_sparse, _) = bench("estimate_death_time sparse3", 50_000_000, || {
    estimate_death_time(black_box(&sparse))
  });
  let (new_empty, _) = bench("estimate_death_time empty", 50_000_000, || {
    estimate_death_time(black_box(&empty))
  });
  let (old_full, _) = bench("baseline old sort-based full8", 50_000_000, || {
    old_sort_based(black_box(&full))
  });
  let (gap, _) = bench("get_interval_gap full8", 50_000_000, || {
    get_interval_gap(black_box(&full))
  });
  let (next_edt, _) = bench("estimate_next_gc_write full8 -> EDT", 50_000_000, || {
    estimate_next_gc_write(black_box(1), black_box(0), black_box(&full), true)
  });
  let (next_empty, _) = bench("estimate_next_gc_write empty", 50_000_000, || {
    estimate_next_gc_write(black_box(7), black_box(0), black_box(&empty), true)
  });

  let _ = writeln!(
    stderr(),
    "\nfull8: new={new_full:.2} ns old_sort={old_full:.2} ns delta={:+.1}% | alloc/op={a1:.2}",
    (new_full - old_full) / old_full * 100.0
  );
  let _ = writeln!(
    stderr(),
    "sparse3={new_sparse:.2} ns empty={new_empty:.2} ns gap={gap:.2} ns \
     next_edt={next_edt:.2} ns next_empty={next_empty:.2} ns"
  );
}
