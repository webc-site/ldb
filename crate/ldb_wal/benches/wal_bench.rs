//! WAL 编解码 / 预留提交热路径基准。

use std::{
  future::{Future, ready},
  hint::black_box,
  io,
  pin::Pin,
};

use compio::runtime::Runtime;
use criterion::{Criterion, criterion_group, criterion_main};
use ldb_wal::{CommitVariant, LogManager, WalFileState, WalSink, entry};

fn bench_entry_codec(c: &mut Criterion) {
  let mut buf = [0u8; 128];
  entry::write_data_hdr(&mut buf, 42, 128, 7, 99);
  c.bench_function("entry/hdr_roundtrip", |b| {
    b.iter(|| {
      let e = black_box(&mut buf);
      entry::write_data_hdr(e, 42, 128, 7, 99);
      entry::seal_chksum(e);
      black_box(entry::validate_chksum(e).is_ok());
    })
  });
  c.bench_function("entry/crc_only", |b| {
    b.iter(|| black_box(entry::compute_chksum(black_box(&buf))))
  });
}

fn bench_buffer(c: &mut Criterion) {
  use ldb_wal::LogBuffer;
  let lb = LogBuffer::new(64 * 1024);
  c.bench_function("walbuf/total_free", |b| {
    b.iter(|| {
      let lb = black_box(&lb);
      black_box(lb.total_free_space());
    })
  });
}

/// reserve → submit 每条目成本（零 I/O sink；运行时与 LogWorker 环外构造，
/// 迭代内只测 200 次 reserve+submit —— 每次 iter 环内新写入 ~8.6KB）
fn bench_reserve_submit(c: &mut Criterion) {
  let rt = Runtime::new().unwrap();
  let mgr = rt.block_on(async {
    LogManager::new(
      1,
      4 * 1024 * 1024,
      CommitVariant::WorkersWriteLog,
      Box::new(NullBenchSink),
      WalFileState {
        start: 0,
        end: 1 << 62,
        shard: 1 << 61,
      },
    )
  });
  let w = mgr.local_log_worker(0).clone();
  c.bench_function("wal/reserve_submit_64b", |b| {
    b.iter(|| {
      rt.block_on(async {
        for i in 0u64..200 {
          w.reserve_wal_entry(&mgr, entry::EntryType::DataEntry, i, &[0u8; 64])
            .await
            .unwrap();
          w.submit_active_log_entry(&mgr, 0).await.unwrap();
        }
        black_box(w.get_current_gsn());
      })
    })
  });
}

/// 内存空转 sink（零 I/O 成本；隔离 reserve/submit 自身开销）
struct NullBenchSink;

impl WalSink for NullBenchSink {
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
criterion_group!(
  benches,
  bench_entry_codec,
  bench_buffer,
  bench_reserve_submit
);
criterion_main!(benches);
