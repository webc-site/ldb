//! 手工基准：1e4 次/组，输出 MiB/s、压缩比、每操作分配次数。
//!
//! 运行：`cargo bench -p ldb_codec`（bench profile, opt-level=3, lto=fat）。
//! 分配计数用包装 `System` 的全局分配器；页路径要求 0 alloc/op。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  hint::black_box,
  process,
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use fastrand::Rng;
use ldb_codec::PageCodec;
use ldb_pack::UNCOMPRESSED;
use ldb_type::PAGE_SIZE;

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: 仅统计后转发给 System，不改变分配语义；计数器为原子操作。
unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(layout.size(), Ordering::Relaxed);
    // SAFETY: 调用方满足 GlobalAlloc::alloc 的布局契约
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: ptr 来自 System.alloc 且布局一致
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(layout.size(), Ordering::Relaxed);
    // SAFETY: 同 alloc
    unsafe { System.alloc_zeroed(layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
    // SAFETY: ptr/layout 来自 System，new_size 由调用方保证合法
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const ITER: usize = 10_000;
const WARMUP: usize = 1_000;
const PAGES: usize = 64;

fn repeat_page(i: usize) -> [u8; PAGE_SIZE] {
  let mut p = [0u8; PAGE_SIZE];
  for (j, b) in p.iter_mut().enumerate() {
    *b = ((j + i) % 7) as u8;
  }
  p
}

fn random_page(i: usize) -> [u8; PAGE_SIZE] {
  let mut p = [0u8; PAGE_SIZE];
  Rng::with_seed(i as u64).fill(&mut p);
  p
}

fn text_page(i: usize) -> [u8; PAGE_SIZE] {
  let phrase = b"the quick brown fox jumps over the lazy dog; 4KiB page compression. ";
  let mut p = [0u8; PAGE_SIZE];
  for (j, b) in p.iter_mut().enumerate() {
    *b = phrase[(j + i) % phrase.len()];
  }
  p
}

fn corpus(name: &str) -> Vec<[u8; PAGE_SIZE]> {
  (0..PAGES)
    .map(|i| match name {
      "repeat" => repeat_page(i),
      "random" => random_page(i),
      _ => text_page(i),
    })
    .collect()
}

fn allocs() -> usize {
  ALLOCS.load(Ordering::Relaxed)
}

/// 返回 (ns/op, MiB/s, ratio, allocs/op)
fn bench_compress(pages: &[[u8; PAGE_SIZE]]) -> (f64, f64, f64, f64) {
  let mut dst = [0u8; PAGE_SIZE];
  for i in 0..WARMUP {
    let p = &pages[i % pages.len()];
    black_box(PageCodec::compress_page(black_box(p), &mut dst).unwrap());
  }
  let a0 = allocs();
  let t = Instant::now();
  let mut stored = 0usize;
  for i in 0..ITER {
    let p = &pages[i % pages.len()];
    let sz = PageCodec::compress_page(black_box(p), &mut dst).unwrap();
    stored += if sz == UNCOMPRESSED {
      PAGE_SIZE
    } else {
      sz as usize
    };
    black_box(&dst[..64]);
  }
  let el = t.elapsed();
  let ns = el.as_nanos() as f64 / ITER as f64;
  let mib = (ITER * PAGE_SIZE) as f64 / el.as_secs_f64() / (1024.0 * 1024.0);
  let ratio = stored as f64 / (ITER * PAGE_SIZE) as f64;
  let allocs_per_op = (allocs() - a0) as f64 / ITER as f64;
  (ns, mib, ratio, allocs_per_op)
}

/// 返回 (ns/op, MiB/s, allocs/op)
fn bench_decompress(pages: &[[u8; PAGE_SIZE]]) -> (f64, f64, f64) {
  // 预压缩：压缩页用帧；不可压缩页用原文（comp_sz = UNCOMPRESSED）
  let mut frames: Vec<(Vec<u8>, u16)> = Vec::with_capacity(pages.len());
  let mut dst = [0u8; PAGE_SIZE];
  for p in pages {
    let sz = PageCodec::compress_page(p, &mut dst).unwrap();
    if sz == UNCOMPRESSED {
      frames.push((p.to_vec(), UNCOMPRESSED));
    } else {
      frames.push((dst[..sz as usize].to_vec(), sz));
    }
  }
  let mut out = [0u8; PAGE_SIZE];
  for i in 0..WARMUP {
    let (f, sz) = &frames[i % frames.len()];
    black_box(PageCodec::decompress_page(black_box(f), *sz, &mut out).is_ok());
  }
  let a0 = allocs();
  let t = Instant::now();
  for i in 0..ITER {
    let (f, sz) = &frames[i % frames.len()];
    PageCodec::decompress_page(black_box(f), *sz, &mut out).unwrap();
    black_box(&out[..64]);
  }
  let el = t.elapsed();
  let ns = el.as_nanos() as f64 / ITER as f64;
  let mib = (ITER * PAGE_SIZE) as f64 / el.as_secs_f64() / (1024.0 * 1024.0);
  let allocs_per_op = (allocs() - a0) as f64 / ITER as f64;
  (ns, mib, allocs_per_op)
}

/// Blob 路径：1 KiB 重复数据，1e4 次。
fn bench_blob() -> (f64, f64, f64, f64) {
  let blob: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
  let mut c = PageCodec::new();
  for _ in 0..WARMUP {
    c.compress_blob(&blob).unwrap();
    black_box(c.as_slice());
  }
  let a0 = allocs();
  let t = Instant::now();
  let mut stored = 0usize;
  for _ in 0..ITER {
    c.compress_blob(black_box(&blob)).unwrap();
    stored += c.compressed_len();
  }
  let el = t.elapsed();
  let ns = el.as_nanos() as f64 / ITER as f64;
  let mib = (ITER * blob.len()) as f64 / el.as_secs_f64() / (1024.0 * 1024.0);
  let ratio = stored as f64 / (ITER * blob.len()) as f64;
  let allocs_per_op = (allocs() - a0) as f64 / ITER as f64;
  (ns, mib, ratio, allocs_per_op)
}

fn main() {
  println!("=== ldb_codec bench: {ITER} ops/group, bench profile ===");
  println!(
    "{:<8} {:>10} {:>10} {:>8} {:>10}",
    "corpus", "ns/op", "MiB/s", "ratio", "allocs/op"
  );
  let mut page_allocs_ok = true;
  for name in ["repeat", "random", "text"] {
    let pages = corpus(name);
    let (c_ns, c_mib, c_ratio, c_alloc) = bench_compress(&pages);
    println!("{name:<8} {c_ns:>10.1} {c_mib:>10.1} {c_ratio:>8.3} {c_alloc:>10.2}  (compress)");
    let (d_ns, d_mib, d_alloc) = bench_decompress(&pages);
    println!(
      "{name:<8} {d_ns:>10.1} {d_mib:>10.1} {:>8} {d_alloc:>10.2}  (decompress)",
      "-"
    );
    page_allocs_ok &= c_alloc == 0.0 && d_alloc == 0.0;
  }
  let (b_ns, b_mib, b_ratio, b_alloc) = bench_blob();
  println!(
    "{:<8} {b_ns:>10.1} {b_mib:>10.1} {b_ratio:>8.3} {b_alloc:>10.2}  (blob compress 1KiB)",
    "blob"
  );
  if !page_allocs_ok {
    println!("FAIL: 页热路径出现分配（要求 0 alloc/op）");
    process::exit(1);
  }
}
