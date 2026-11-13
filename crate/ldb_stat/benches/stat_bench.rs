//! ldb_stat 微基准（无第三方依赖）：record_* 的 ns/op 与 alloc/op。
//!
//! `cargo bench -p ldb_stat` 直接运行；输出走 stderr。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  io::{Write as _, stderr},
  sync::atomic::{AtomicUsize, Ordering},
  thread,
  time::Instant,
};

use ldb_stat::StatCollector;

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

/// 打印 `ns/op` 与 `alloc/op`；统计区间不含任何输出，避免污染计时。
fn bench<F: FnMut()>(name: &str, iters: u64, mut f: F) {
  for _ in 0..(iters / 10).max(1) {
    f();
  }
  let a0 = ALLOCS.load(Ordering::Relaxed);
  let t = Instant::now();
  for _ in 0..iters {
    f();
  }
  let ns = t.elapsed().as_nanos() as f64 / iters as f64;
  let allocs = (ALLOCS.load(Ordering::Relaxed) - a0) as f64 / iters as f64;
  let _ = writeln!(stderr(), "{name:<18} {ns:8.3} ns/op  {allocs:5.3} alloc/op");
}

fn main() {
  let s = StatCollector::default();
  // 预热
  s.record_logical_write(1);
  for _ in 0..10_000 {
    s.record_page_hit();
  }
  bench("record_page_hit", 100_000_000, || s.record_page_hit());
  bench("record_page_miss", 100_000_000, || s.record_page_miss());
  bench("record_gc_move", 100_000_000, || s.record_gc_move(4096));
  bench("calc_db_waf", 100_000_000, || {
    black_box(s.calc_db_waf());
  });
  bench("snapshot_metric", 10_000_000, || {
    black_box(s.snapshot_metric());
  });
  bench("export_json", 1_000_000, || {
    black_box(s.export_json());
  });
  check_concurrent_exact();
}

/// 4 / 8 线程 × 1e6 并发递增（三个计数器），逐次断言总数精确 —— Relaxed 原子
/// 无丢失更新。这是 R1 之外的独立重跑（每线程计 page_hit + txn_commit + gc_move）。
fn check_concurrent_exact() {
  for threads in [4u64, 8u64] {
    run_concurrent(threads, 1_000_000);
  }
}

/// 单次压测：`threads` 个线程各 `per_thread` 次，三计数器各自断言精确总数。
fn run_concurrent(threads: u64, per_thread: u64) {
  let s = StatCollector::default();
  let t = Instant::now();
  thread::scope(|scope| {
    for _ in 0..threads {
      scope.spawn(|| {
        for _ in 0..per_thread {
          s.record_page_hit();
          s.record_txn_commit();
          s.record_gc_move(1);
        }
      });
    }
  });
  let snap = s.snapshot_metric();
  let expect = threads * per_thread;
  assert_eq!(
    snap.page_hit, expect,
    "page_hit 并发丢失更新: {}",
    snap.page_hit
  );
  assert_eq!(
    snap.txn_commit, expect,
    "txn_commit 并发丢失更新: {}",
    snap.txn_commit
  );
  assert_eq!(
    snap.gc_move, expect,
    "gc_move 并发丢失更新: {}",
    snap.gc_move
  );
  let _ = writeln!(
    stderr(),
    "concurrent {threads}x{per_thread}  page_hit={} txn_commit={} gc_move={}  {:?}",
    snap.page_hit,
    snap.txn_commit,
    snap.gc_move,
    t.elapsed()
  );
}
