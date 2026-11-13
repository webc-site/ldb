//! ldb_recovery 重放扫描微基准（无第三方依赖，`cargo bench -p ldb_recovery`）。
//!
//! 度量 `Replayer::replay_wal_shard` 的吞吐（ns/条目）与堆分配次数（C1/C2 证据）。
//! 负载构造模仿 `ldb_wal` 落盘段：DataEntry 条目链 + 每段尾零填充到 4KiB 倍数，
//! 间隔插入 TX_START/TX_COMMIT 覆盖事务配对路径。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  future::{Future, ready},
  hint::black_box,
  io,
  pin::Pin,
  sync::atomic::{AtomicUsize, Ordering},
  time::Instant,
};

use compio::runtime::Runtime;
use ldb_recovery::Replayer;
use ldb_wal::{
  CommitVariant, EntryType, LogManager, WAL_WRITE_ALIGN, WalFileState, WalSink, entry,
};

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
    // SAFETY: 指针/layout 由配套 alloc 产生，转发给 System。
    unsafe { System.dealloc(ptr, layout) }
  }
  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: 转发给 System，参数由调用方保证合法。
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// 空转 sink
struct NullSink;

impl WalSink for NullSink {
  fn write_wal(
    &mut self,
    _data: Vec<u8>,
    _off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(ready(io::Result::Ok(())))
  }
  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(ready(io::Result::Ok(())))
  }
}

/// 构造分片：`n` 条 DataEntry（payload = `pl` 字节）+ 每 4KiB 段尾零填充，
/// 每 8 条插入一对 TX_START/TX_COMMIT（lsn 递增）。
fn build_shard(n: u64, pl: usize) -> Vec<u8> {
  let mut out: Vec<u8> = Vec::new();
  let mut seg: Vec<u8> = Vec::new();
  let mut lsn = 0u64;
  let flush = |seg: &mut Vec<u8>, out: &mut Vec<u8>| {
    let pad = seg.len().div_ceil(WAL_WRITE_ALIGN) * WAL_WRITE_ALIGN;
    seg.resize(pad, 0);
    out.extend_from_slice(seg);
    seg.clear();
  };
  for i in 0..n {
    if i % 8 == 2 {
      let mut e = vec![0u8; entry::LOG_ENTRY_HDR];
      entry::write_hdr(&mut e, EntryType::TxStart, lsn, entry::LOG_ENTRY_HDR as u64);
      entry::seal_chksum(&mut e);
      lsn += 1;
      seg.extend_from_slice(&e);
    }
    let total = entry::DATA_ENTRY_HDR + pl;
    let mut e = vec![0u8; total];
    entry::write_data_hdr(&mut e, lsn, total as u64, i, i);
    e[entry::DATA_ENTRY_HDR..].fill(0xAB);
    entry::seal_chksum(&mut e);
    lsn += 1;
    seg.extend_from_slice(&e);
    if i % 8 == 6 {
      let mut e = vec![0u8; entry::LOG_ENTRY_HDR];
      entry::write_hdr(
        &mut e,
        EntryType::TxCommit,
        lsn,
        entry::LOG_ENTRY_HDR as u64,
      );
      entry::seal_chksum(&mut e);
      lsn += 1;
      seg.extend_from_slice(&e);
    }
    if seg.len() >= WAL_WRITE_ALIGN {
      flush(&mut seg, &mut out);
    }
  }
  if !seg.is_empty() {
    flush(&mut seg, &mut out);
  }
  out
}

fn main() {
  let rt = Runtime::new().expect("runtime");
  let mgr = LogManager::new(
    1,
    64 * 1024,
    CommitVariant::WorkersWriteLog,
    Box::new(NullSink),
    WalFileState {
      start: 0,
      end: 1 << 30,
      shard: 1 << 30,
    },
  );

  let n: u64 = 200_000;
  let shard = build_shard(n, 16);
  let frames: u64 = rt.block_on(async {
    let mut r = Replayer::new();
    r.replay_wal_shard(&shard, &mgr).await.expect("replay")
  });

  let mut reps = 0u32;
  let allocs0 = ALLOCS.load(Ordering::Relaxed);
  let start = Instant::now();
  while start.elapsed().as_secs_f64() < 1.5 {
    rt.block_on(async {
      let mut r = Replayer::new();
      black_box(r.replay_wal_shard(black_box(&shard), &mgr).await.unwrap());
    });
    reps += 1;
  }
  let el = start.elapsed();
  let allocs = ALLOCS.load(Ordering::Relaxed) - allocs0;

  let per_rep = el.as_nanos() as f64 / f64::from(reps);
  let ns_per_entry = per_rep / frames as f64;
  let alloc_per_rep = allocs as f64 / f64::from(reps);
  eprintln!(
    "replay/shard: {} 帧/rep, {:.3} ms/rep, {:.2} ns/条目, {:.2} alloc/rep, reps={}",
    frames,
    per_rep / 1e6,
    ns_per_entry,
    alloc_per_rep,
    reps
  );
}
