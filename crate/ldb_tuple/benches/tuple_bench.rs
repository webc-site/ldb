//! ldb_tuple 热路径基准：`value_at` / `reconstruct_tuple` / `get_delta`（零分配）/
//! `append`（更新）。给出 ns/op 以支撑"热路径零成本"主张。

use std::hint::black_box;

use compio::runtime::Runtime;
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ldb_tuple::{BTreeVi, ChainedTuple, FatTuple, UndoVersion};

/// 造 n 个版本（tx = 10,20,...）的链式元组，载荷均为 4 字节。
fn mk_chained(n: u64) -> ChainedTuple {
  let mut ct = ChainedTuple {
    head: vec![0xAB; 4],
    head_tx: n * 10,
    ..ChainedTuple::default()
  };
  for i in 1..n {
    ct.undo.push(UndoVersion {
      tx_id: i * 10,
      payload: vec![i as u8; 4],
      removed: false,
    });
  }
  ct
}

/// 造 n 个版本（旧→新）的胖元组，载荷均为 4 字节。
fn mk_fat(n: u64) -> FatTuple {
  let mut ft = FatTuple::default();
  for i in 1..=n {
    ft.append(i * 10, &[i as u8; 4], 4096);
  }
  ft
}

fn bench_reads(c: &mut Criterion) {
  let ct = mk_chained(16);
  c.bench_function("chained_value_at", |b| {
    b.iter(|| black_box(ct.value_at(black_box(95))))
  });

  let ft = mk_fat(16);
  c.bench_function("fat_reconstruct_tuple", |b| {
    b.iter(|| black_box(ft.reconstruct_tuple(black_box(95))))
  });

  c.bench_function("fat_get_delta_zero_alloc", |b| {
    b.iter(|| black_box(ft.versions.get_delta(black_box(7))))
  });
}

fn bench_update(c: &mut Criterion) {
  let ft = mk_fat(8);
  c.bench_function("fat_append_update", |b| {
    b.iter_batched(
      || ft.clone(),
      |mut ft| {
        black_box(ft.append(black_box(999), black_box(&[7u8; 4]), 4096));
        ft
      },
      BatchSize::SmallInput,
    )
  });
}

/// 构造代表性检查点状态：40 个多版本键（含晋升为胖）+ 40 个单版本链式键 +
/// 1 个已迁墓键；用于 serialize/deserialize ns 与体积测量。
fn build_checkpoint_state(rt: &Runtime) -> BTreeVi {
  rt.block_on(async {
    let mut t = BTreeVi::create(64, 3, 4096);
    for i in 0..40u8 {
      t.insert(&[i], &[i; 8], 10).await.unwrap();
    }
    for i in 100..140u8 {
      t.insert(&[i], &[i; 8], 10).await.unwrap(); // 无更新 → 保持链式
    }
    t.insert(&[250], &[250; 8], 10).await.unwrap();
    t.remove(&[250], 15).await.unwrap(); // 将被迁墓
    for round in 0..5u64 {
      for i in 0..40u8 {
        t.update_same_size_in_place(&[i], &[i.wrapping_add(round as u8 + 1); 8], 20 + round)
          .await
          .unwrap();
      }
    }
    t.trigger_page_wise_garbage_collection(16).await; // 迁墓 [250]，保留各版本
    t
  })
}

fn bench_checkpoint(c: &mut Criterion) {
  let rt = Runtime::new().unwrap();
  let t = build_checkpoint_state(&rt);
  let mut buf = Vec::new();
  rt.block_on(t.serialize(&mut buf));
  black_box(&buf);
  c.bench_function("serialize_checkpoint", |b| {
    b.iter(|| {
      let mut out = Vec::with_capacity(buf.len());
      rt.block_on(t.serialize(&mut out));
      black_box(out)
    })
  });
  c.bench_function("deserialize_checkpoint", |b| {
    b.iter_batched(
      || BTreeVi::create(64, 3, 4096),
      |mut r| {
        rt.block_on(r.deserialize(black_box(&buf))).unwrap();
        black_box(r)
      },
      BatchSize::SmallInput,
    )
  });
}

criterion_group!(tuple_bench, bench_reads, bench_update, bench_checkpoint);
criterion_main!(tuple_bench);
