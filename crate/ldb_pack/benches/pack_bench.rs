//! ldb_pack 微基准（无第三方依赖：工作区未声明 criterion，且禁止 `cargo add`）。
//!
//! 输出 ns/op 与每次调用的堆分配次数；`cargo bench -p ldb_pack` 直接运行。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  io::{Write as _, stderr},
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use ldb_pack::{Bin, Pid2Offset, ZipEntry, bin_packing_per_block, copy_compressed_pages};

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

/// 返回（ns/op, allocs/op）。alloc 统计区间不含任何 println。
fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) -> (f64, f64) {
  for _ in 0..(iters / 10).max(1) {
    f();
  }
  let a0 = allocs();
  let t = Instant::now();
  for _ in 0..iters {
    f();
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(iters);
  let ap = (allocs() - a0) as f64 / f64::from(iters);
  // 写 stderr：bench 输出走独立流，且只用 writeln!（规避 D1 的打印宏红线）
  let _ = writeln!(stderr(), "{name:<56}{ns:10.2} ns/op{ap:10.2} alloc/op");
  (ns, ap)
}

/// 确定性 xorshift，避免依赖 rand
struct Rng(u64);
impl Rng {
  fn next(&mut self) -> u64 {
    self.0 ^= self.0 << 13;
    self.0 ^= self.0 >> 7;
    self.0 ^= self.0 << 17;
    self.0
  }
  fn range(&mut self, lo: u32, hi: u32) -> u32 {
    lo + (self.next() % u64::from(hi - lo + 1)) as u32
  }
}

fn make_entries(n: usize, comp: impl Fn(usize) -> u16) -> Vec<ZipEntry> {
  (0..n)
    .map(|i| ZipEntry {
      comp_sz: comp(i),
      death_time: i as u64,
      src_off: (i * 4096) as u32,
    })
    .collect()
}

fn main() {
  // ---- Pid2Offset ----
  let mut p = Pid2Offset::default();
  bench("Pid2Offset set+get_offset+get_comp_sz", 10_000_000, || {
    p.set(black_box(0x0001_2345_6789_ABCD), black_box(2048));
    black_box(p.get_offset());
    black_box(p.get_comp_sz());
  });

  // ---- bin_packing_per_block ----
  let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
  // 复位基线：计时区间含 `bins.copy_from_slice`，单独测出来供扣减
  for n in [8usize, 64, 512] {
    let template = vec![Bin::default(); n];
    let mut bins = template.clone();
    bench(
      &format!("baseline bins reset n={n}"),
      200_000 / n as u32,
      || bins.copy_from_slice(&template),
    );
  }
  for (n, label) in [
    (8usize, "n=8  uniform"),
    (64, "n=64 uniform"),
    (512, "n=512 uniform"),
  ] {
    let entries = make_entries(n, |_| 2048);
    let template = vec![Bin::default(); n];
    let mut bins = template.clone();
    bench(
      &format!("bin_packing_per_block {label}"),
      200_000 / n as u32,
      || {
        bins.copy_from_slice(&template);
        black_box(bin_packing_per_block(
          black_box(&entries),
          black_box(&mut bins),
        ));
      },
    );
  }
  {
    let n = 64;
    let mut sizes = vec![0u16; n];
    for s in &mut sizes {
      *s = rng.range(512, 4096) as u16;
    }
    let entries = make_entries(n, |i| sizes[i]);
    let template = vec![Bin::default(); n];
    let mut bins = template.clone();
    bench("bin_packing_per_block n=64 mixed(512..4096)", 5_000, || {
      bins.copy_from_slice(&template);
      black_box(bin_packing_per_block(
        black_box(&entries),
        black_box(&mut bins),
      ));
    });
  }

  // ---- copy_compressed_pages ----
  {
    let n = 64usize;
    let src = make_entries(n, |_| 2048);
    let src_buf = vec![0xABu8; n * 4096];
    let mut dst_buf = vec![0u8; n * 4096];
    bench("copy_compressed_pages n=64 2048B", 100_000, || {
      black_box(copy_compressed_pages(
        black_box(&src),
        black_box(&src_buf),
        black_box(&mut dst_buf),
        0,
        n - 1,
      ));
    });
  }
}
