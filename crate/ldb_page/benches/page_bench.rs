//! ldb_page 微基准（无第三方依赖）：页头写历史与 delta 编解码的 ns/op、
//! alloc/op，附 `GcHook` 冷路径装箱 Future 的分配开销对照。
//!
//! `cargo bench -p ldb_page` 直接运行；输出走 stderr。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  future::{Future, ready},
  hint::black_box,
  io::{Write as _, stderr},
  pin::Pin,
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use ldb_page::{MetadataPage, PageHeader, build_delta_header, update_delta_payload};

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

/// 返回（ns/op, allocs/op）；alloc 统计区间不含任何输出
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

/// 非 ZST future 的装箱（实际 GcHook 实现捕获页/块状态；ZST 装箱不分配）
fn boxed_ready(v: u64) -> Pin<Box<dyn Future<Output = u64>>> {
  Box::pin(async move { v })
}

fn main() {
  let mut full = PageHeader::new();
  for g in 1..=8u64 {
    full.add_write_gsn(g * 100);
  }

  let (ns_add, ap_add) = bench("PageHeader::add_write_gsn full8", 20_000_000, || {
    full.add_write_gsn(black_box(900));
    black_box(full.w_p_gsn[0])
  });

  let mut churn = full;
  let (ns_remove, ap_remove) = bench("PageHeader::remove_last_write_gsn hit", 20_000_000, || {
    churn.w_p_gsn[7] = 700;
    churn.remove_last_write_gsn();
    u64::from(black_box(churn.wcnt))
  });

  let (ns_gc, ap_gc) = bench("PageHeader::update_last_gc_write", 20_000_000, || {
    full.update_last_gc_write(black_box(12345));
    black_box(full.prev_gcwrite)
  });

  // 根表查找：Option 边界检查在合法路径上的开销（与裸索引对比）
  let mut meta = MetadataPage::new();
  meta.roots[3] = 42;
  let (ns_root, ap_root) = bench(
    "MetadataPage::get_root valid slot (Option)",
    20_000_000,
    || meta.get_root(black_box(3)).unwrap_or_default(),
  );
  let (ns_idx, ap_idx) = bench(
    "MetadataPage::roots[3] baseline (index)",
    20_000_000,
    || black_box(meta.roots[black_box(3)]),
  );

  let slots1 = [(3u16, 4u16)];
  let slots4 = [(0u16, 4u16), (4, 4), (8, 4), (12, 4)];
  let tuple = [7u8; 16];
  let mut delta1 = [0u8; 64];
  let mut delta4 = [0u8; 64];

  let (ns_build1, ap_build1) =
    bench(
      "build_delta_header count=1",
      10_000_000,
      || match build_delta_header(&mut delta1, &slots1) {
        Ok(s) => u64::from(s),
        Err(_) => 0,
      },
    );
  let (ns_build4, ap_build4) =
    bench(
      "build_delta_header count=4",
      10_000_000,
      || match build_delta_header(&mut delta4, &slots4) {
        Ok(s) => u64::from(s),
        Err(_) => 0,
      },
    );
  let (ns_update1, ap_update1) =
    bench(
      "update_delta_payload count=1",
      10_000_000,
      || match update_delta_payload(&mut delta1, &tuple) {
        Ok(n) => n as u64,
        Err(_) => 0,
      },
    );
  let (ns_update4, ap_update4) =
    bench(
      "update_delta_payload count=4",
      10_000_000,
      || match update_delta_payload(&mut delta4, &tuple) {
        Ok(n) => n as u64,
        Err(_) => 0,
      },
    );

  let (ns_box, ap_box) = bench("GcHook boxed future (8B payload)", 10_000_000, || {
    let f = boxed_ready(black_box(7));
    black_box(&f);
    0
  });
  let (ns_direct, ap_direct) = bench("direct ready future (baseline)", 10_000_000, || {
    let f = ready(black_box(7));
    black_box(&f);
    0
  });

  let _ = writeln!(
    stderr(),
    "\nsummary: add={ns_add:.2}ns/{ap_add:.2}alloc remove={ns_remove:.2}ns/{ap_remove:.2}alloc \
     gcwrite={ns_gc:.2}ns/{ap_gc:.2}alloc \
     build1={ns_build1:.2}ns/{ap_build1:.2}alloc build4={ns_build4:.2}ns/{ap_build4:.2}alloc \
     update1={ns_update1:.2}ns/{ap_update1:.2}alloc update4={ns_update4:.2}ns/{ap_update4:.2}alloc \
     root={ns_root:.2}ns/{ap_root:.2}alloc idx={ns_idx:.2}ns/{ap_idx:.2}alloc \
     boxed={ns_box:.2}ns/{ap_box:.2}alloc direct={ns_direct:.2}ns/{ap_direct:.2}alloc"
  );
}
