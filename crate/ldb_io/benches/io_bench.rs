//! ldb_io 微基准（无第三方依赖）：单页读/写 ns、批量写吞吐、fsync/打孔延迟。
//!
//! `cargo bench -p ldb_io` 直接运行；输出走 stderr。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  env::temp_dir,
  fs::{create_dir_all, remove_dir_all},
  hint::black_box,
  io::{Write as _, stderr},
  process::id,
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use ldb_io::{CompioFileBackend, DWB_RECORD, MemBackend, doublewrite::DWB_RECORDS};
use ldb_type::PAGE_SIZE;

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

static PAGE: [u8; PAGE_SIZE] = [0xAB; PAGE_SIZE];
static BATCH: [u8; 64 * PAGE_SIZE] = [0xCD; 64 * PAGE_SIZE];

/// DWB 记录原样（64B 头 + 4096B 页）：4160B 既非 4 KiB 倍数，槽偏移也不对齐。
static DWB_REC: [u8; DWB_RECORD] = [0xEE; DWB_RECORD];

/// 方案 a 的 8 KiB 记录（4 KiB 对齐 + 8 KiB 长度），可在 Linux O_DIRECT 下写。
#[repr(align(4096))]
struct AlignedRecord([u8; 2 * PAGE_SIZE]);
static DWB_REC_8K: AlignedRecord = AlignedRecord([0xDD; 2 * PAGE_SIZE]);

const PAGES: u64 = 4096;
const ITERS: u32 = 2048;

fn out(name: &str, ns: f64, allocs_per_op: f64) {
  let _ = writeln!(
    stderr(),
    "{name:<28} {ns:>9.1} ns/op  {allocs_per_op:>6.2} allocs/op"
  );
}

#[compio::main]
async fn main() {
  let dir = temp_dir().join(format!("ldb_io_bench_{}", id()));
  create_dir_all(&dir).unwrap();
  let path = dir.join("bench.db");
  let b = CompioFileBackend::open_direct_or_buffered(&path)
    .await
    .unwrap();

  // 预写 PAGES 页，保证读路径不是稀疏空洞
  for p in 0..PAGES {
    b.write_at_page(p, &PAGE).await.unwrap();
  }
  b.fdatasync().await.unwrap();

  // 单页读（栈数组 by value，无堆分配）
  let mut acc = 0u64;
  let a0 = allocs();
  let t = Instant::now();
  for i in 0..ITERS {
    let p = u64::from(i) % PAGES;
    let buf = b.read_owned(p, [0u8; PAGE_SIZE]).await.unwrap();
    acc = acc.wrapping_add(u64::from(buf[0]));
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(ITERS);
  out(
    "page_read_owned",
    ns,
    (allocs() - a0) as f64 / f64::from(ITERS),
  );
  black_box(acc);

  // 单页写（零拷贝：静态页 → 内核）
  let t = Instant::now();
  for i in 0..ITERS {
    let p = u64::from(i) % PAGES;
    b.write_at_page(p, &PAGE).await.unwrap();
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(ITERS);
  out("page_write_at_page(0copy)", ns, 0.0);

  // 单页写（含调用方 `to_vec` 一次 4KiB 拷贝 —— 引擎 StoreBackend 路径）
  let a0 = allocs();
  let t = Instant::now();
  for i in 0..ITERS {
    let p = u64::from(i) % PAGES;
    b.write_owned(PAGE.to_vec(), p * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(ITERS);
  out(
    "page_write_owned(1copy)",
    ns,
    (allocs() - a0) as f64 / f64::from(ITERS),
  );

  // 批量写：64 页/次，单次 write_all_at
  let batches = 64u32;
  let t = Instant::now();
  for i in 0..batches {
    let off = u64::from(i) * BATCH.len() as u64;
    b.write_bytes_at(off, &BATCH).await.unwrap();
  }
  let elapsed = t.elapsed().as_secs_f64();
  let mib = f64::from(batches) * BATCH.len() as f64 / (1024.0 * 1024.0);
  let _ = writeln!(
    stderr(),
    "{:<28} {:>9.1} MiB/s  ({:.1} ns/页)",
    "batch_write_extent(64p)",
    mib / elapsed,
    elapsed * 1e9 / (f64::from(batches) * 64.0)
  );

  // fsync / 打孔延迟
  let t = Instant::now();
  b.fdatasync().await.unwrap();
  out("fdatasync", t.elapsed().as_nanos() as f64, 0.0);

  let t = Instant::now();
  match b.punch_hole(0, PAGE_SIZE as u64).await {
    Ok(()) => out("punch_hole", t.elapsed().as_nanos() as f64, 0.0),
    Err(e) => {
      let _ = writeln!(stderr(), "{:<28} 不支持: {e}", "punch_hole");
    }
  }

  // ---- DWB 布局对比（R3 决策 c：4160B 记录 + 缓冲 I/O）----
  let pages = DWB_RECORDS as u64;
  let mib = pages as f64 * DWB_RECORD as f64 / (1024.0 * 1024.0);
  let report = |name: &str, elapsed: f64, direct: bool| {
    let _ = writeln!(
      stderr(),
      "{name:<28} {:>9.1} MiB/s  ({:.1} ns/页, direct={direct})",
      mib / elapsed,
      elapsed * 1e9 / pages as f64
    );
  };

  // 方案 c：4160B 记录，缓冲 I/O（DWB 生产路径）
  let dwb_buf = CompioFileBackend::open_buffered(dir.join("dwb_4160_buffered.dat"))
    .await
    .unwrap();
  dwb_buf.set_len(pages * DWB_RECORD as u64).await.unwrap();
  let t = Instant::now();
  for i in 0..pages {
    dwb_buf
      .write_bytes_at(i * DWB_RECORD as u64, &DWB_REC)
      .await
      .unwrap();
  }
  report(
    "dwb_4160_buffered",
    t.elapsed().as_secs_f64(),
    dwb_buf.is_direct(),
  );
  let t = Instant::now();
  dwb_buf.sync_durable().await.unwrap();
  out(
    "dwb_sync_durable(8.4MiB)",
    t.elapsed().as_nanos() as f64,
    0.0,
  );

  // 同文件同负载：OS fdatasync（macOS 仅进程崩溃安全）作为 F_FULLFSYNC 的对照
  for i in 0..pages {
    dwb_buf
      .write_bytes_at(i * DWB_RECORD as u64, &DWB_REC)
      .await
      .unwrap();
  }
  let t = Instant::now();
  dwb_buf.fdatasync().await.unwrap();
  out("dwb_fdatasync(8.4MiB)", t.elapsed().as_nanos() as f64, 0.0);

  // 方案 c 的对照：4160B 记录走 Direct-ish（macOS F_NOCACHE 可写；Linux O_DIRECT
  // 会在首次非对齐写时降级，direct_after 会变 false —— 正是方案 c 要避免的）
  let dwb_direct = CompioFileBackend::open_direct_or_buffered(dir.join("dwb_4160_direct.dat"))
    .await
    .unwrap();
  dwb_direct.set_len(pages * DWB_RECORD as u64).await.unwrap();
  let direct_before = dwb_direct.is_direct();
  let t = Instant::now();
  for i in 0..pages {
    dwb_direct
      .write_bytes_at(i * DWB_RECORD as u64, &DWB_REC)
      .await
      .unwrap();
  }
  let el = t.elapsed().as_secs_f64();
  let _ = writeln!(
    stderr(),
    "{:<28} {:>9.1} MiB/s  ({:.1} ns/页, direct_before={direct_before} direct_after={})",
    "dwb_4160_direct_probe",
    mib / el,
    el * 1e9 / pages as f64,
    dwb_direct.is_direct()
  );
  // 方案 a：8192B 对齐记录（容量减半，但 Linux O_DIRECT 可写）
  let dwb_8k = CompioFileBackend::open_direct_or_buffered(dir.join("dwb_8192.dat"))
    .await
    .unwrap();
  dwb_8k.set_len(pages * 2 * PAGE_SIZE as u64).await.unwrap();
  let t = Instant::now();
  for i in 0..pages {
    dwb_8k
      .write_bytes_at(i * 2 * PAGE_SIZE as u64, &DWB_REC_8K.0)
      .await
      .unwrap();
  }
  let el = t.elapsed().as_secs_f64();
  let mib8 = pages as f64 * (2 * PAGE_SIZE) as f64 / (1024.0 * 1024.0);
  let _ = writeln!(
    stderr(),
    "{:<28} {:>9.1} MiB/s  ({:.1} ns/页, 2×字节, direct={})",
    "dwb_8192_aligned",
    mib8 / el,
    el * 1e9 / pages as f64,
    dwb_8k.is_direct()
  );
  let t = Instant::now();
  dwb_8k.sync_durable().await.unwrap();
  out(
    "dwb_8192_sync_durable(16.5MiB)",
    t.elapsed().as_nanos() as f64,
    0.0,
  );
  drop(dwb_buf);
  drop(dwb_direct);
  drop(dwb_8k);

  // MemBackend 基线（无系统调用）
  let m = MemBackend::new();
  let t = Instant::now();
  for i in 0..ITERS {
    m.write_at_page(u64::from(i) % PAGES, &PAGE).await.unwrap();
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(ITERS);
  out("mem_write_page", ns, 0.0);

  let t = Instant::now();
  let mut page = [0u8; PAGE_SIZE];
  for i in 0..ITERS {
    m.read_at_page(u64::from(i) % PAGES, &mut page)
      .await
      .unwrap();
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(ITERS);
  out("mem_read_page", ns, 0.0);

  let _ = writeln!(stderr(), "direct={}", b.is_direct());
  drop(b);
  drop(m);
  remove_dir_all(&dir).unwrap();
}
