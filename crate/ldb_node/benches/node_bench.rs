//! B+Tree 节点热路径基准（lower_bound / insert / upsert / remove / compactify）。

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ldb_node::{BTreeNode, MEMCMP, node::NUM_SLOTS};

fn mk_node(keys: impl IntoIterator<Item = Vec<u8>>, payload_len: usize) -> Box<BTreeNode> {
  let mut node = BTreeNode::new(true, 0);
  node.set_fences(b"", b"", MEMCMP);
  for key in keys {
    assert!(node.insert_key_value(&key, &vec![0u8; payload_len], MEMCMP));
  }
  node
}

/// 9 B key + 8 B payload + 16 B 槽位 = 33 B/条；110 条留出余量
const N_KEYS: u32 = 110;

/// 键的 4B head 必须随 i 变化，否则 16-Hint 全相等、窄化失效（bench 名不副实）
fn key_for(i: u32) -> Vec<u8> {
  format!(
    "{:08x}",
    (u64::from(i).wrapping_mul(0x9E37_79B9) & 0xFFFF_FFFF) as u32
  )
  .into_bytes()
}

fn make_keys(n: u32) -> Vec<Vec<u8>> {
  (0..n).map(key_for).collect()
}

fn bench_lower_bound(c: &mut Criterion) {
  let keys = make_keys(N_KEYS);
  let node = mk_node(keys.clone(), 8);
  c.bench_function("lower_bound/hint_hit", |b| {
    b.iter(|| {
      let mut exact = false;
      for k in &keys {
        black_box(node.lower_bound(black_box(k), &mut exact, MEMCMP));
      }
    })
  });
  let miss = b"zzzzzzzz";
  c.bench_function("lower_bound/miss", |b| {
    b.iter(|| {
      let mut exact = false;
      black_box(node.lower_bound(black_box(miss), &mut exact, MEMCMP));
    })
  });
}

/// 1e6 个均匀随机查询的批量下界（criterion 时间 / 1e6 = ns/op）
fn bench_lookup_1e6(c: &mut Criterion) {
  let node = mk_node(make_keys(N_KEYS), 8);
  let mut queries = vec![[0u8; 8]; 1_000_000];
  let mut state = 0x1234_5678_9ABC_DEF0u64;
  for q in &mut queries {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let key = key_for((state % u64::from(N_KEYS)) as u32);
    q.copy_from_slice(&key);
  }
  c.bench_function("lookup/random_1e6", |b| {
    b.iter(|| {
      let mut exact = false;
      for q in &queries {
        black_box(node.lower_bound(black_box(&q[..]), &mut exact, MEMCMP));
      }
    })
  });
}

fn bench_insert(c: &mut Criterion) {
  c.bench_function("insert/sequential", |b| {
    b.iter_batched(
      || BTreeNode::new(true, 0),
      |mut node| {
        for k in make_keys(80) {
          black_box(node.insert_key_value(black_box(&k), &[0; 8], MEMCMP));
        }
        black_box(node)
      },
      BatchSize::PerIteration,
    )
  });
  // 同长 payload 原位覆写（update-in-place 路径）
  c.bench_function("insert/upsert_same_length", |b| {
    b.iter_batched(
      || mk_node(make_keys(80), 8),
      |mut node| {
        for k in make_keys(80) {
          black_box(node.insert_key_value(black_box(&k), &[1; 8], MEMCMP));
        }
        black_box(node)
      },
      BatchSize::PerIteration,
    )
  });
}

fn bench_remove(c: &mut Criterion) {
  c.bench_function("remove/half_random", |b| {
    b.iter_batched(
      || {
        let keys = make_keys(100);
        let node = mk_node(keys.clone(), 8);
        // 确定性打乱删除顺序
        let mut order: Vec<u32> = (0..100).collect();
        let mut state = 0xDEAD_BEEFu64;
        for i in (1..order.len()).rev() {
          state ^= state >> 12;
          state ^= state << 25;
          state ^= state >> 27;
          let j = (state % (i as u64 + 1)) as usize;
          order.swap(i, j);
        }
        order.truncate(50);
        (node, keys, order)
      },
      |(mut node, keys, order)| {
        for i in order {
          black_box(node.remove_key(&keys[i as usize], MEMCMP));
        }
        black_box(node)
      },
      BatchSize::PerIteration,
    )
  });
}

fn bench_compactify(c: &mut Criterion) {
  c.bench_function("compactify/fragments", |b| {
    b.iter_batched(
      || {
        let mut node = mk_node(make_keys(100), 8);
        for i in (0..100u32).step_by(3) {
          assert!(node.remove_key(&key_for(i), MEMCMP));
        }
        node
      },
      |mut node| {
        node.compactify(MEMCMP);
        black_box(&node);
      },
      BatchSize::PerIteration,
    )
  });
}

fn bench_find_separator(c: &mut Criterion) {
  let node = mk_node(make_keys(NUM_SLOTS as u32 / 2), 8);
  c.bench_function("find_separator/leaf", |b| {
    b.iter(|| black_box(node.find_separator(black_box(false), MEMCMP)))
  });
}

criterion_group!(
  benches,
  bench_lower_bound,
  bench_lookup_1e6,
  bench_insert,
  bench_remove,
  bench_compactify,
  bench_find_separator
);
criterion_main!(benches);
