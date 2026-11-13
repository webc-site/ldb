//! ldb_gc 微基准（无第三方依赖）：victim 选取的 ns/op 与 alloc/op。
//!
//! `cargo bench -p ldb_gc` 直接运行；输出走 stderr。
//! 关注点：`greedy_gc`/`edt_based_gc`/`random_gc`/`k_greedy_gc` 每轮对状态列表
//! 做一次快照（C-30 锁模型修复的代价，1 次 Vec 克隆分配）；`fifo_gc` 只窥视队首，
//! 零分配（R1 由 `pop_front` 改为 `front` 后）。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  io::{Write as _, stderr},
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use ldb_gc::{GarbageCollector, GcStrategy};
use ldb_space::{BlockState, SpaceManager};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: 只计数后原样转发 System。
unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: 转发 System，layout 由调用方保证合法。
    unsafe { System.alloc(layout) }
  }
  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: ptr/layout 来自本分配器 alloc。
    unsafe { System.dealloc(ptr, layout) }
  }
  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: 转发 System。
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocs() -> usize {
  ALLOCS.load(Ordering::Relaxed)
}

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
  let _ = writeln!(stderr(), "{name:<46}{ns:9.2} ns/op{ap:8.2} alloc/op");
  (ns, ap)
}

fn main() {
  const BLOCKS: u64 = 1024;
  let mgr: SpaceManager = SpaceManager::new(BLOCKS, 64, 2048, 4, BLOCKS * 64, 2);
  {
    let mut s = mgr.sspace.write();
    for bid in (0..BLOCKS).rev() {
      let m = &mut s.block_metadata[bid as usize];
      m.state = BlockState::Full;
      m.total_pid_cnt = 8;
      m.invalidated_pid_cnt = (bid as u32 % 7) + 1; // 有效页 1..=7，避免 0/total 短路
      m.edt = bid;
      s.block_lists[BlockState::Full.idx()].push(bid);
    }
  }
  let mut gc = GarbageCollector::new(4);

  let (ns_g, ap_g) = bench("greedy_gc(read=Full) 1024blk", 200_000, || {
    gc.greedy_gc(&mgr, true)
  });
  let (ns_e, ap_e) = bench("edt_based_gc(read=Full,flush=MAX)", 200_000, || {
    gc.edt_based_gc(&mgr, true, u64::MAX)
  });
  let (ns_r, ap_r) = bench("random_gc 1024blk", 200_000, || gc.random_gc(&mgr));
  let (ns_k, ap_k) = bench("k_greedy_gc(k=16) 1024blk", 200_000, || {
    gc.k_greedy_gc(&mgr, 16)
  });
  let (ns_f, ap_f) = bench("fifo_gc peek 1024blk", 5_000_000, || gc.fifo_gc(&mgr));

  // 列表快照成本基线（空选：快照 + 立即返回）
  let (ns_snap, ap_snap) = bench("snapshot FULL list only", 200_000, || {
    black_box(mgr.sspace.read().block_lists[BlockState::Full.idx()].snapshot()).len() as u64
  });
  let (ns_size, ap_size) = bench("get_block_list_size (no snapshot)", 20_000_000, || {
    mgr.get_block_list_size(BlockState::Full) as u64
  });
  let (ns_sel, ap_sel) = bench("select_victim_block_to_gc_read(Greedy)", 200_000, || {
    let mut g = GarbageCollector::new(4);
    g.select_victim_block_to_gc_read(&mgr, GcStrategy::Greedy, 0)
  });

  let _ = writeln!(
    stderr(),
    "\nsummary: greedy={ns_g:.1}ns/{ap_g:.1}alloc edt={ns_e:.1}ns/{ap_e:.1}alloc \
     random={ns_r:.1}ns/{ap_r:.1}alloc kgreedy={ns_k:.1}ns/{ap_k:.1}alloc \
     fifo={ns_f:.1}ns/{ap_f:.1}alloc snapshot={ns_snap:.1}ns/{ap_snap:.1}alloc \
     size={ns_size:.1}ns/{ap_size:.1}alloc sel={ns_sel:.1}ns/{ap_sel:.1}alloc"
  );
}
