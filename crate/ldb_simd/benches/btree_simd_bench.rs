//! ldb_simd 微基准：common_prefix 三档实现对比 + hint 函数 ns 级基线。

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ldb_simd::{common_prefix, get_head, make_hint, search_hint, update_hint};

/// 改造前的实现：逐字节 `zip + take_while`，作为性能基线。
fn common_prefix_scalar(lhs: &[u8], rhs: &[u8]) -> usize {
  lhs.iter().zip(rhs).take_while(|(a, b)| a == b).count()
}

/// 8 字节字宽 XOR 实现（与 `common_prefix` 的 8..16 字节档一致），
/// 用于量化「u64 vs fearless_simd」的分档依据。
fn common_prefix_words(lhs: &[u8], rhs: &[u8]) -> usize {
  let mut lhs = lhs;
  let mut rhs = rhs;
  let mut off = 0;
  while let (Some((a, next_lhs)), Some((b, next_rhs))) =
    (lhs.split_first_chunk::<8>(), rhs.split_first_chunk::<8>())
  {
    let diff = u64::from_le_bytes(*a) ^ u64::from_le_bytes(*b);
    if diff != 0 {
      return off + diff.trailing_zeros() as usize / 8;
    }
    off += 8;
    lhs = next_lhs;
    rhs = next_rhs;
  }
  off + lhs.iter().zip(rhs).take_while(|(a, b)| a == b).count()
}

/// 构造 length 字节、在 pos 处首次不同的两个 key（pos == length 表示全等）。
fn pair_of(len: usize, pos: usize) -> (Vec<u8>, Vec<u8>) {
  let lhs: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37)).collect();
  let mut rhs = lhs.clone();
  if pos < len {
    rhs[pos] = rhs[pos].wrapping_add(1);
  }
  // 避免常量折叠：输入指针来自 black_box(Vec)
  (lhs, rhs)
}

fn bench_common_prefix(c: &mut Criterion) {
  let mut group = c.benchmark_group("common_prefix");
  for len in [4usize, 8, 16, 32, 128, 1024] {
    let input = pair_of(len, len / 2);
    group.bench_with_input(
      BenchmarkId::new("scalar_baseline", len),
      &input,
      |b, (l, r)| b.iter(|| black_box(common_prefix_scalar(black_box(l), black_box(r)))),
    );
    group.bench_with_input(BenchmarkId::new("u64_words", len), &input, |b, (l, r)| {
      b.iter(|| black_box(common_prefix_words(black_box(l), black_box(r))))
    });
    group.bench_with_input(BenchmarkId::new("current", len), &input, |b, (l, r)| {
      b.iter(|| black_box(common_prefix(black_box(l), black_box(r))))
    });
  }
  group.finish();
}

/// 改造前的实现：`min(4)` + 单次 `copy_from_slice`（编译为 memcpy 调用）。
fn get_head_memcpy(key: &[u8]) -> [u8; 4] {
  let mut ret = [0; 4];
  let n = key.len().min(4);
  ret[..n].copy_from_slice(&key[..n]);
  ret
}

fn bench_get_head(c: &mut Criterion) {
  let mut group = c.benchmark_group("get_head");
  for (name, key) in [
    ("2B", &b"ab"[..]),
    ("4B", &b"abcd"[..]),
    ("9B", &b"k00000001"[..]),
  ] {
    group.bench_function(format!("memcpy/{name}"), |b| {
      b.iter(|| black_box(get_head_memcpy(black_box(key))))
    });
    group.bench_function(format!("current/{name}"), |b| {
      b.iter(|| black_box(get_head(black_box(key))))
    });
  }
  group.finish();
}

/// 改造前的实现（逐 hint 线性扫 + 等值 run 二次扫），作为 SIMD 的基线。
fn search_hint_scalar(
  hints: &[[u8; 4]; 16],
  key_head: &[u8; 4],
  upper_bound: usize,
) -> (usize, usize) {
  let hint_count = hints.len();
  if upper_bound <= hint_count * 2 {
    return (0, upper_bound);
  }
  let dist = upper_bound / (hint_count + 1);
  let key = u32::from_be_bytes(*key_head);
  let mut lower_hint = hint_count;
  let mut upper_hint = hint_count;
  for (idx, head) in hints.iter().enumerate() {
    if lower_hint == hint_count && u32::from_be_bytes(*head) >= key {
      lower_hint = idx;
      for (u, h) in hints.iter().enumerate().skip(idx) {
        if u32::from_be_bytes(*h) != key {
          upper_hint = u;
          break;
        }
      }
      break;
    }
  }
  let lower = lower_hint * dist;
  let upper = if upper_hint < hint_count {
    (upper_hint + 1) * dist
  } else {
    upper_bound
  };
  (lower, upper)
}

/// 构造 16 个递增 head，`key_idx` 处的 head 等于 key；`None` 表示 key 大于
/// 全部 hint（未命中，lower 落到 16*dist）。
fn sorted_hints(key_idx: Option<usize>) -> ([[u8; 4]; 16], [u8; 4]) {
  let mut hints = [[0u8; 4]; 16];
  for (i, h) in hints.iter_mut().enumerate() {
    *h = [((i + 1) * 8) as u8, 0, 0, 0];
  }
  let key = match key_idx {
    Some(i) => hints[i],
    None => [0xFF, 0, 0, 0],
  };
  (hints, key)
}

fn bench_hints(c: &mut Criterion) {
  let mut heads = [[0u8; 4]; 65];
  for (i, h) in heads.iter_mut().enumerate() {
    *h = [(i & 0xFF) as u8, (i >> 8) as u8, 0xAA, 0x55];
  }
  let mut hints = [[0u8; 4]; 16];
  c.bench_function("make_hint/64", |b| {
    b.iter(|| make_hint(black_box(&mut hints), black_box(64), |i| heads[i]))
  });

  let mut hints_u = [[0u8; 4]; 16];
  c.bench_function("update_hint/64", |b| {
    b.iter(|| update_hint(black_box(&mut hints_u), 64, 48, |i| heads[i]))
  });

  // dist == 0：上游把 16 个 hint 全部写成 slots[0].head（小节点插入路径）
  let mut hints_s = [[0u8; 4]; 16];
  c.bench_function("update_hint/small16", |b| {
    b.iter(|| update_hint(black_box(&mut hints_s), 16, 0, |i| heads[i]))
  });

  let mut group = c.benchmark_group("search_hint");
  for (name, count) in [("33", 33usize), ("64", 64), ("1000", 1000)] {
    for (tag, key_idx) in [
      ("miss", None),
      ("hit0", Some(0)),
      ("hit7", Some(7)),
      ("hit15", Some(15)),
    ] {
      let (hints, key_head) = sorted_hints(key_idx);
      let id = BenchmarkId::new(format!("scalar_{tag}"), name);
      group.bench_with_input(id, &(hints, key_head), |b, (h, k)| {
        b.iter(|| black_box(search_hint_scalar(black_box(h), black_box(k), count)))
      });
      let id = BenchmarkId::new(format!("simd_{tag}"), name);
      group.bench_with_input(id, &(hints, key_head), |b, (h, k)| {
        b.iter(|| black_box(search_hint(black_box(h), black_box(k), count)))
      });
    }
  }
  group.finish();
}

fn bench_search_mixed(c: &mut Criterion) {
  // 4096 个不同 key、随机的命中位置：单条循环里的 `head >= key` 分支无法被
  // 分支预测器记忆（对比固定输入的 hit0/hit15 用例，后者对 scalar 最有利）。
  let mut cases = Vec::with_capacity(4096);
  let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
  for _ in 0..4096 {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    let pos = (rng % 16) as usize;
    let (hints, key) = sorted_hints(Some(pos));
    cases.push((hints, key));
  }
  let mut group = c.benchmark_group("search_hint_batch4096");
  group.throughput(Throughput::Elements(cases.len() as u64));
  group.bench_function("scalar", |b| {
    b.iter(|| {
      let mut acc = 0usize;
      for (h, k) in &cases {
        let (l, u) = search_hint_scalar(black_box(h), black_box(k), 64);
        acc = acc.wrapping_add(l ^ u);
      }
      black_box(acc)
    })
  });
  group.bench_function("simd", |b| {
    b.iter(|| {
      let mut acc = 0usize;
      for (h, k) in &cases {
        let (l, u) = search_hint(black_box(h), black_box(k), 64);
        acc = acc.wrapping_add(l ^ u);
      }
      black_box(acc)
    })
  });
  group.finish();
}

criterion_group!(
  benches,
  bench_common_prefix,
  bench_get_head,
  bench_hints,
  bench_search_mixed
);
criterion_main!(benches);
