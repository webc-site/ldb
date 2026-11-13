//! ldb_simd 行为测试：逐条钉死上游 BTreeNode hint/search 辅助语义。
//!
//! 期望值全部是字面量（由上游 `node.cc` 的循环手工推导），不复用被测实现。

use ldb_simd::{common_prefix, get_head, make_hint, search_hint, update_hint};

// ---- get_head ----

#[test]
fn get_head_copies_first_four_bytes() {
  assert_eq!(get_head(b"abcd"), [b'a', b'b', b'c', b'd']);
  assert_eq!(get_head(b"abcdef"), [b'a', b'b', b'c', b'd']);
}

#[test]
fn get_head_zero_pads_short_keys() {
  assert_eq!(get_head(b""), [0, 0, 0, 0]);
  assert_eq!(get_head(b"a"), [b'a', 0, 0, 0]);
  assert_eq!(get_head(&[0xFE, 0xDC]), [0xFE, 0xDC, 0, 0]);
}

// ---- common_prefix ----

#[test]
fn common_prefix_byte_path_matches_upstream_min_len() {
  assert_eq!(common_prefix(b"", b""), 0);
  assert_eq!(common_prefix(b"", b"a"), 0);
  assert_eq!(common_prefix(b"a", b""), 0);
  assert_eq!(common_prefix(b"a", b"b"), 0);
  assert_eq!(common_prefix(b"abc", b"abd"), 2);
  assert_eq!(common_prefix(b"abc", b"abc"), 3);
  assert_eq!(common_prefix(b"ab", b"abc"), 2);
  assert_eq!(common_prefix(b"abc", b"ab"), 2);
  assert_eq!(common_prefix(&[0, 0, 1], &[0, 0, 2]), 2);
}

#[test]
fn common_prefix_word_path_finds_first_diff_byte() {
  // 9 字节：u64 字宽路径 + 1 字节尾
  assert_eq!(common_prefix(b"1234567X9", b"1234567Y9"), 7);
  assert_eq!(common_prefix(b"12345678X", b"12345678Y"), 8);
  assert_eq!(common_prefix(b"123456789", b"123456789"), 9);
  assert_eq!(common_prefix(b"12345678", b"12345678"), 8);
}

#[test]
fn common_prefix_simd_path_finds_first_diff_lane() {
  let a = b"0123456789abcdefX";
  let b = b"0123456789abcdefY";
  assert_eq!(common_prefix(a, b), 16);

  // 前 16 字节全等，差异落在向量后 tail 的第 2 字节：结果 = 向量长度 + tail 下标
  assert_eq!(
    common_prefix(b"aaaaaaaaaaaaaaaaXY", b"aaaaaaaaaaaaaaaaXZ"),
    17
  );

  let mut x = vec![7u8; 128];
  let mut y = x.clone();
  y[100] = 8;
  assert_eq!(common_prefix(&x, &y), 100);

  x = vec![3u8; 1024];
  y = x.clone();
  y[1000] = 4;
  assert_eq!(common_prefix(&x, &y), 1000);
}

#[test]
fn common_prefix_simd_uses_min_len() {
  // 16B 与 32B：共同部分 16B 全等，结果必须停在 min(len)
  let a = b"0123456789abcdef";
  let b = b"0123456789abcdef0123456789abcdef";
  assert_eq!(common_prefix(a, b), 16);
  assert_eq!(common_prefix(b, a), 16);
}

// ---- make_hint ----

#[test]
fn make_hint_uses_dist_count_over_hint_count_plus_one() {
  let head = |i: usize| [i as u8, 0, 0, 0];
  let mut hints = [[0u8; 4]; 16];
  // count=48 → dist=48/17=2 → 取 slots[2], slots[4], ..., slots[32]
  make_hint(&mut hints, 48, head);
  let want: [[u8; 4]; 16] = [
    [2, 0, 0, 0],
    [4, 0, 0, 0],
    [6, 0, 0, 0],
    [8, 0, 0, 0],
    [10, 0, 0, 0],
    [12, 0, 0, 0],
    [14, 0, 0, 0],
    [16, 0, 0, 0],
    [18, 0, 0, 0],
    [20, 0, 0, 0],
    [22, 0, 0, 0],
    [24, 0, 0, 0],
    [26, 0, 0, 0],
    [28, 0, 0, 0],
    [30, 0, 0, 0],
    [32, 0, 0, 0],
  ];
  assert_eq!(hints, want);
}

#[test]
fn make_hint_dist_zero_writes_slot_zero() {
  let head = |i: usize| [i as u8, 0, 0, 0];
  let mut hints = [[0xEEu8; 4]; 16];
  // count=16 → dist=0 → 上游把全部 hint 写成 slots[0].head
  make_hint(&mut hints, 16, head);
  assert_eq!(hints, [[0u8, 0, 0, 0]; 16]);

  // count=0 与上游一致：读 slots[0]（未初始化槽位），本模型写作 head(0)
  let mut empty = [[0xEEu8; 4]; 16];
  make_hint(&mut empty, 0, head);
  assert_eq!(empty, [[0u8, 0, 0, 0]; 16]);
}

// ---- update_hint ----

#[test]
fn update_hint_dist_zero_rewrites_all_hints() {
  let head = |i: usize| [i as u8, 0, 0, 0];
  let mut hints = [[0xEEu8; 4]; 16];
  // count=16 → dist=0，三条件均不成立 → begin=0，全部写成 slots[0].head
  update_hint(&mut hints, 16, 0, head);
  assert_eq!(hints, [[0u8, 0, 0, 0]; 16]);
}

#[test]
fn update_hint_full_rebuild_when_dist_changes() {
  let head = |i: usize| [i as u8, 0, 0, 0];
  let mut hints = [[0xEEu8; 4]; 16];
  // count=34 → dist=2；(33)/17=1 != 2 → begin=0，全部重扫 slots[2..=32]
  update_hint(&mut hints, 34, 20, head);
  let want: [[u8; 4]; 16] = [
    [2, 0, 0, 0],
    [4, 0, 0, 0],
    [6, 0, 0, 0],
    [8, 0, 0, 0],
    [10, 0, 0, 0],
    [12, 0, 0, 0],
    [14, 0, 0, 0],
    [16, 0, 0, 0],
    [18, 0, 0, 0],
    [20, 0, 0, 0],
    [22, 0, 0, 0],
    [24, 0, 0, 0],
    [26, 0, 0, 0],
    [28, 0, 0, 0],
    [30, 0, 0, 0],
    [32, 0, 0, 0],
  ];
  assert_eq!(hints, want);
}

#[test]
fn update_hint_starts_at_slot_over_dist_minus_one() {
  let head = |i: usize| [i as u8, 0, 0, 0];
  let mut hints = [[0xEEu8; 4]; 16];
  // count=35 → dist=2；(34)/17=2 == dist；24/2=12>1 → begin=11
  update_hint(&mut hints, 35, 24, head);
  let want: [[u8; 4]; 16] = [
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [0xEE, 0xEE, 0xEE, 0xEE],
    [24, 0, 0, 0],
    [26, 0, 0, 0],
    [28, 0, 0, 0],
    [30, 0, 0, 0],
    [32, 0, 0, 0],
  ];
  assert_eq!(hints, want);
}

#[test]
fn update_hint_small_count_ignores_tail_heuristic() {
  let head = |i: usize| [i as u8, 0, 0, 0];
  let mut hints = [[0xEEu8; 4]; 16];
  // count=33 → dist=1；count > 33 不成立 → begin=0，重扫 slots[1..=16]
  update_hint(&mut hints, 33, 20, head);
  let want: [[u8; 4]; 16] = [
    [1, 0, 0, 0],
    [2, 0, 0, 0],
    [3, 0, 0, 0],
    [4, 0, 0, 0],
    [5, 0, 0, 0],
    [6, 0, 0, 0],
    [7, 0, 0, 0],
    [8, 0, 0, 0],
    [9, 0, 0, 0],
    [10, 0, 0, 0],
    [11, 0, 0, 0],
    [12, 0, 0, 0],
    [13, 0, 0, 0],
    [14, 0, 0, 0],
    [15, 0, 0, 0],
    [16, 0, 0, 0],
  ];
  assert_eq!(hints, want);
}

// ---- search_hint ----

#[test]
fn search_hint_leaves_bounds_when_count_at_most_32() {
  let hints = [[0xFFu8; 4]; 16];
  let key = [0x10u8, 0, 0, 0];
  assert_eq!(search_hint(&hints, &key, 32), (0, 32));
  assert_eq!(search_hint(&hints, &key, 0), (0, 0));
}

#[test]
fn search_hint_brackets_equal_head_run() {
  // count=34 → dist=2；hints[0..4] < key，hints[4..8] == key，hints[8..16] > key
  let below = [0x00u8, 0, 0, 0];
  let key = [0x10u8, 0, 0, 0];
  let above = [0x20u8, 0, 0, 0];
  let mut hints = [[0u8; 4]; 16];
  hints[..4].fill(below);
  hints[4..8].fill(key);
  hints[8..].fill(above);
  // lower_hint=4 → lower=4*2=8；upper_hint=8 → upper=(8+1)*2=18
  assert_eq!(search_hint(&hints, &key, 34), (8, 18));
}

#[test]
fn search_hint_keeps_upper_bound_when_no_unequal_hint() {
  // count=40 → dist=2；全部 hint < key → lower_hint=upper_hint=16
  let below = [0x00u8, 0, 0, 0];
  let key = [0x10u8, 0, 0, 0];
  let hints = [below; 16];
  assert_eq!(search_hint(&hints, &key, 40), (32, 40));
}

#[test]
fn search_hint_dist_and_simd_lane_big_endian() {
  // count=33 → dist=1；hints[8..10] == key → lower=8，upper=(10+1)*1=11
  let below = [0x00u8, 0, 0, 0];
  let key = [0x10u8, 0, 0, 0];
  let above = [0x20u8, 0, 0, 0];
  let mut hints = [[0u8; 4]; 16];
  hints[..8].fill(below);
  hints[8..10].fill(key);
  hints[10..].fill(above);
  assert_eq!(search_hint(&hints, &key, 33), (8, 11));

  // 手工按大端折叠：lane 4 = 0x00000002 < key=0x01000000，lane 5 == key。
  // 若按小端解读，lane 4 = 0x02000000 > key，pos 会错误地变成 4。
  let key = [0x01u8, 0x00, 0x00, 0x00];
  let mut hints = [[0x00u8; 4]; 16];
  hints[4] = [0x00, 0x00, 0x00, 0x02];
  hints[5] = key;
  hints[6] = [0x02, 0x00, 0x00, 0x00];
  hints[7..].fill([0x03, 0x00, 0x00, 0x00]);
  // count=34 → dist=2；pos=5 → lower=10；run 终点 6 → upper=(6+1)*2=14
  assert_eq!(search_hint(&hints, &key, 34), (10, 14));
}

#[test]
fn search_hint_head_lane0_big_endian_without_equal_run() {
  // 手工按大端折叠：hints[0]=0x02010000 > key=0x01020000。
  // 若按小端解读，0x00000102 < 0x00000201，pos 会错误地越过 lane 0。
  let key = [0x01u8, 0x02, 0x00, 0x00];
  let mut hints = [[0x03u8, 0x00, 0x00, 0x00]; 16];
  hints[0] = [0x02, 0x01, 0x00, 0x00];
  hints[1] = key;
  hints[2] = [0x02, 0x02, 0x00, 0x00];
  // count=34 → dist=2；pos=0，hints[0] != key → upper=(0+1)*2=2
  assert_eq!(search_hint(&hints, &key, 34), (0, 2));

  // 首个 >= key 的 hint 不等于 key（无等值 run）：hints[0..4] < key，hints[4] 起
  // 全部 > key → upper=(4+1)*2=10
  let key = [0x10u8, 0, 0, 0];
  let mut hints = [[0u8; 4]; 16];
  hints[..4].fill([0x00, 0, 0, 0]);
  hints[4..].fill([0x20, 0, 0, 0]);
  assert_eq!(search_hint(&hints, &key, 34), (8, 10));
}

// ---- search_hint / hint 重建的穷尽 lane 边界 ----

/// 上游 `SearchHint`（`node.cc:241-266`）的独立标量参考。
///
/// 与被测代码的差异是刻意的：hint 比较直接用 `[u8; 4]` 的字典序
/// （Rust 数组 `<`/`==` 即 C++ `std::memcmp` 的判定），不经过 `u32::from_be_bytes`，
/// 因此能同时验证「大端解码」与「lane 定位」两件事。
fn ref_search_hint<const N: usize>(
  hints: &[[u8; 4]; N],
  key_head: &[u8; 4],
  count: usize,
) -> (usize, usize) {
  if count <= N * 2 {
    return (0, count);
  }
  let dist = count / (N + 1);
  let mut pos = 0;
  while pos < N && hints[pos] < *key_head {
    pos += 1;
  }
  let mut pos2 = pos;
  while pos2 < N && hints[pos2] == *key_head {
    pos2 += 1;
  }
  let lower = pos * dist;
  (lower, if pos2 < N { (pos2 + 1) * dist } else { count })
}

/// 上游 `MakeHint`（`node.cc:220-225`）的独立参考。
fn ref_make_hint<const N: usize>(count: usize, head_at: impl Fn(usize) -> [u8; 4]) -> [[u8; 4]; N] {
  let dist = count / (N + 1);
  let mut out = [[0u8; 4]; N];
  for (idx, hint) in out.iter_mut().enumerate() {
    *hint = head_at(dist * (idx + 1));
  }
  out
}

/// 上游 `UpdateHint`（`node.cc:227-239`）的独立参考。
fn ref_update_hint<const N: usize>(
  count: usize,
  slot_id: usize,
  head_at: impl Fn(usize) -> [u8; 4],
) -> [[u8; 4]; N] {
  let dist = count / (N + 1);
  let begin = if (count > N * 2 + 1) && ((count - 1) / (N + 1) == dist) && (slot_id / dist > 1) {
    slot_id / dist - 1
  } else {
    0
  };
  let mut out = ref_make_hint::<N>(count, head_at);
  for hint in out.iter_mut().take(begin) {
    *hint = [0xEE; 4];
  }
  out
}

// 三个固定的 head 档位：严格小于 / 等于 / 严格大于 key（大端字节序）。
const BELOW: [u8; 4] = [0x0F, 0xFF, 0xFF, 0xFF];
const KEY: [u8; 4] = [0x10, 0x00, 0x00, 0x00];
const ABOVE: [u8; 4] = [0x10, 0x00, 0x00, 0x01];

fn count_set<const N: usize>() -> [usize; 11] {
  // guard 两侧 + dist 边界 + 非 (N+1) 整倍数 + 大 count
  [
    0,
    1,
    N,
    2 * N - 1,
    2 * N,
    2 * N + 1,
    2 * N + 2,
    3 * N + 3,
    7 * N + 5,
    64,
    1000,
  ]
}

/// N∈1..=40 × pos∈0..=N × {小于, 等于(+run 长度), 大于} × count 边界 全组合。
fn check_lane_boundaries<const N: usize>() -> usize {
  let mut cases = 0usize;
  for count in count_set::<N>() {
    for pos in 0..=N {
      let mut shapes: Vec<[[u8; 4]; N]> = Vec::with_capacity(N + 4);

      // 关系 = 小于：hints[..=pos] < key，其余 > key（pos == N 表示全部小于）
      let mut h = [ABOVE; N];
      h[..(pos + 1).min(N)].fill(BELOW);
      shapes.push(h);

      // 关系 = 大于：hints[..pos] < key，hints[pos..] > key
      let mut h = [BELOW; N];
      h[pos..].fill(ABOVE);
      shapes.push(h);

      // 关系 = 等于：等值 run 长度 1..=N-pos（含 run 直达末尾）
      for run in 1..=(N - pos) {
        let mut h = [ABOVE; N];
        h[..pos].fill(BELOW);
        h[pos..pos + run].fill(KEY);
        shapes.push(h);
      }

      // 非递增的尖刺：pos 处等于 key，pos+1 处又回到 below（run 长度 1）
      if pos + 1 < N {
        let mut h = [BELOW; N];
        h[pos] = KEY;
        h[N - 1] = ABOVE;
        shapes.push(h);
      }

      for hints in shapes {
        let want = ref_search_hint(&hints, &KEY, count);
        let got = search_hint(&hints, &KEY, count);
        assert_eq!(got, want, "N={N} count={count} pos={pos} hints={hints:?}");
        // 区间合法性与上游不变量（lower <= upper、两端都在 [0, count] 内）
        let (lower, upper) = got;
        assert!(lower <= upper && upper <= count, "N={N} count={count}");
        cases += 1;
      }
    }
  }
  cases
}

/// 随机 hint 形态（含全相等 / 全低于 / 全高于 / 无序），覆盖参考实现的分支组合。
fn check_random_shapes<const N: usize>(seed: u64) -> usize {
  let mut rng = seed | 1;
  let mut next = move || {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    rng
  };
  let mut cases = 0usize;
  for count in count_set::<N>() {
    for _ in 0..16 {
      let mut hints = [[0u8; 4]; N];
      for hint in hints.iter_mut() {
        *hint = match next() % 4 {
          0 => BELOW,
          1 => KEY,
          2 => ABOVE,
          _ => [0x10, 0x00, 0x00, (next() % 3) as u8],
        };
      }
      let want = ref_search_hint(&hints, &KEY, count);
      assert_eq!(
        search_hint(&hints, &KEY, count),
        want,
        "N={N} count={count}"
      );
      cases += 1;
    }
  }
  cases
}

/// 极端 N 的字面量锚点：保证参考实现本身没有和被测代码「同错」。
#[test]
fn search_hint_n1_n2_literal_bounds() {
  // N=1, count=3 → dist=1
  assert_eq!(search_hint(&[BELOW], &KEY, 3), (1, 3));
  assert_eq!(search_hint(&[KEY], &KEY, 3), (0, 3));
  assert_eq!(search_hint(&[ABOVE], &KEY, 3), (0, 1));
  // N=1, count=2 不满足 count > 2
  assert_eq!(search_hint(&[ABOVE], &KEY, 2), (0, 2));
  // N=2, count=5 → dist=1
  assert_eq!(search_hint(&[BELOW, KEY], &KEY, 5), (1, 5));
  assert_eq!(search_hint(&[KEY, ABOVE], &KEY, 5), (0, 2));
  assert_eq!(search_hint(&[BELOW, ABOVE], &KEY, 5), (1, 2));
  // N=2, count=4 不满足 count > 4
  assert_eq!(search_hint(&[ABOVE, ABOVE], &KEY, 4), (0, 4));
}

#[test]
fn search_hint_matches_scalar_reference_for_all_lane_boundaries() {
  let mut cases = 0usize;
  macro_rules! each_n {
    ($($n:literal)*) => {$({
      cases += check_lane_boundaries::<$n>();
      cases += check_random_shapes::<$n>(0xA5A5_0000 + $n as u64);
    })*};
  }
  each_n!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    33 34 35 36 37 38 39 40);
  // 精确用例数：N=1..40 × 11 档 count × (2 + 等值 run + 尖刺) 形态 153,780
  // + 随机形态 40 × 11 × 16 = 7,040
  assert_eq!(cases, 160_820, "cases={cases}");
}

#[test]
fn hint_rebuild_matches_reference_for_all_hint_counts() {
  let mut cases = 0usize;
  macro_rules! each_n {
    ($($n:literal)*) => {$({
      for count in 0..=(3 * $n + 4) {
        let head_at = |i: usize| [(i % 251) as u8, (i / 251) as u8, 0x5A, 0xC3];
        let mut got = [[0u8; 4]; $n];
        make_hint(&mut got, count, head_at);
        assert_eq!(got, ref_make_hint::<$n>(count, head_at), "make_hint N={} count={count}", $n);
        // slot_id 覆盖 begin 两侧：0（slot/dist <= 1）与 count-1（begin > 0）
        let raw = [0, count / 2, count.saturating_sub(1)];
        let mut uniq = [0usize; 3];
        let mut uniq_len = 0;
        for s in raw {
          if !uniq[..uniq_len].contains(&s) {
            uniq[uniq_len] = s;
            uniq_len += 1;
          }
        }
        for &slot in &uniq[..uniq_len] {
          // 预填 0xEE：begin 之前的 hint 必须原样保留（增量更新语义）
          let mut got = [[0xEEu8; 4]; $n];
          update_hint(&mut got, count, slot, head_at);
          assert_eq!(
            got,
            ref_update_hint::<$n>(count, slot, head_at),
            "update_hint N={} count={count} slot={slot}",
            $n
          );
          cases += 1;
        }
        cases += 1;
      }
    })*};
  }
  each_n!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    33 34 35 36 37 38 39 40);
  // 精确用例数：N=1..40 的 make (3N+5) + update (9N+10) = 12N+15 → 10,440
  assert_eq!(cases, 10_440, "cases={cases}");
}
