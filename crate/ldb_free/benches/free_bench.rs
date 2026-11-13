//! ldb_free 微基准（无第三方依赖）：ns/op 与 alloc/op。
//!
//! `cargo bench -p ldb_free` 直接运行；输出走 stderr。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  io::{Write as _, stderr},
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use futures_lite::future::block_on;
use ldb_free::FreeStorageManager;
use ldb_page::hook::TxnLogHook;
use ldb_type::PageId;

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
  let _ = writeln!(stderr(), "{name:<44}{ns:9.2} ns/op{ap:8.2} alloc/op");
  (ns, ap)
}

/// 空 WAL 钩子
struct Hook;

impl TxnLogHook for Hook {
  async fn log_pid2_offset(&self, _pid: PageId) {}
  async fn log_valid_page(&self, _pid: PageId) {}
  async fn log_free_page(&self, _pid: PageId) {}
}

fn main() {
  let hook = Hook;

  // 1) 稳态分配+归还（tier0，1 页）：pop_front + validate + hook，及 prepare+public 回路
  let fsm = FreeStorageManager::new(true, 8);
  block_on(fsm.prepare_free_tier(1, 0, &hook));
  block_on(fsm.public_free_extents());
  let mut buf = Vec::new();
  let (cycle, a_cycle) = bench("alloc+free cycle tier0 (1 page)", 1_000_000, || {
    buf.clear();
    let pid = block_on(fsm.request_free_extent(1, |_| true, &hook, &mut buf));
    let got = u64::from(pid.is_some());
    if let Some(pid) = pid {
      block_on(fsm.prepare_free_tier(pid, 0, &hook));
      block_on(fsm.public_free_extents());
    }
    got
  });

  // 2) 全空 miss：同级 3 次空判 + 自 62 级向下的回退扫描（最坏冷路径）
  let empty = FreeStorageManager::new(true, 8);
  let (miss, _) = bench("alloc miss (all tiers empty)", 1_000_000, || {
    buf.clear();
    u64::from(block_on(empty.request_free_extent(1, |_| true, &hook, &mut buf)).is_none())
  });

  // 3) 分裂路径：tier4(16 页) 命中、请求 5 页 ⇒ 尾部 11 页拆成 8+2+1
  let split_fsm = FreeStorageManager::new(true, 8);
  const N: u32 = 200_000;
  for i in 0..N {
    block_on(split_fsm.prepare_free_tier(1 + u64::from(i) * 16, 4, &hook));
  }
  block_on(split_fsm.public_free_extents());
  let (split_ns, _) = bench("alloc split tier4 -> 5 pages", N, || {
    buf.clear();
    let pid = block_on(split_fsm.request_free_extent(5, |_| true, &hook, &mut buf));
    black_box(&buf);
    u64::from(pid.is_some())
  });

  let _ = writeln!(
    stderr(),
    "\nsummary: cycle={cycle:.2} ns/op ({a_cycle:.2} alloc/op)  miss={miss:.2} ns/op  split={split_ns:.2} ns/op"
  );
}
