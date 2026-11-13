//! TypeFold 性质与结构变异测试（对拍上游 `typefold.cc:5-68`）。
//!
//! 与 `typefold_test.rs` 的分工：那里钉死具体字节与边界值（布局断言），本文件只做
//! 「性质 + 变异」——确定性伪随机样本 + 截断/尾部填充/位翻转/内嵌 NUL 变异，
//! 断言四类不变量（覆盖 `Integer`/`UInteger`/`Timestamp`/`UniqueID`/`Varchar` 全部 10 个函数）：
//!
//! 1. 保序同构：`fold(a).cmp(fold(b)) == a.cmp(b)`（随机对；B+Tree 的 memcmp 比较依赖它）
//! 2. 精确逆：`unfold(fold(x)) == x`，且截断必 Err、尾部多余字节不影响解码
//! 3. 位映射：翻转编码第 k 位 ⇒ 解码值必为 `x ^ (1 << k)`（同时校验大端序与符号位翻转）
//! 4. 采样扫描：边界 + 等步长跨量级 + LCG 采样三类样本并集，排序去重后编码字节
//!    严格递增，且每个样本 `unfold(fold(v)) == v`（采样域内 100% 保序；全域等价由
//!    差分对拍与编码定义保证，不由采样用例单独证明）
//!
//! 组合键协议依据上游 `FoldKey`（`pos += Fold(out + pos, ...)` 逐列推进游标；
//! 本用例如 4 列键，对应 `CustomerWDCType::FoldKey`
//! `ZLeanStore/benchmark/src/include/benchmark/tpcc/schema.h:146-153`，
//! 3 列例见 `:112-115`，单列例见 `:29-31`）。

use std::fmt::Debug;

use ldb_type::{Error, typefold::*};

/// 每类随机对样本数（本文件性质用例总量 > 100_000）
const PAIRS: usize = 25_000;
/// 结构变异样本数
const MUTATIONS: usize = 2_000;
/// 组合键样本数
const KEYS: usize = 5_000;

/// 确定性 LCG（Knuth MMIX）步进，避免为此引入 rand 依赖
fn lcg(seed: &mut u64) -> u64 {
  *seed = seed
    .wrapping_mul(6_364_136_223_846_793_005)
    .wrapping_add(1_442_695_040_888_963_407);
  *seed
}

/// i32 编码（目标缓冲恰为 4 字节，折叠不可能失败）
fn enc_i32(x: i32) -> [u8; 4] {
  let mut b = [0u8; 4];
  fold_integer(&mut b, x).unwrap();
  b
}

/// u32 编码
fn enc_u32(x: u32) -> [u8; 4] {
  let mut b = [0u8; 4];
  fold_uinteger(&mut b, x).unwrap();
  b
}

/// i64 编码
fn enc_i64(x: i64) -> [u8; 8] {
  let mut b = [0u8; 8];
  fold_timestamp(&mut b, x).unwrap();
  b
}

/// u64 编码
fn enc_u64(x: u64) -> [u8; 8] {
  let mut b = [0u8; 8];
  fold_unique_id(&mut b, x).unwrap();
  b
}

/// 保序同构：编码字节的字典序 == 原值序（随机对，含大量跨符号位与跨字节进位样本）
#[test]
fn fold_is_order_isomorphic_on_random_pairs() {
  let mut seed = 0x5DEE_CE66_D000_0001;
  let mut cases = 0usize;
  for _ in 0..PAIRS {
    let (a, b) = (lcg(&mut seed) as i32, lcg(&mut seed) as i32);
    assert_eq!(enc_i32(a).cmp(&enc_i32(b)), a.cmp(&b), "i32: {a} vs {b}");
    let (a, b) = (lcg(&mut seed) as u32, lcg(&mut seed) as u32);
    assert_eq!(enc_u32(a).cmp(&enc_u32(b)), a.cmp(&b), "u32: {a} vs {b}");
    let (a, b) = (lcg(&mut seed) as i64, lcg(&mut seed) as i64);
    assert_eq!(enc_i64(a).cmp(&enc_i64(b)), a.cmp(&b), "i64: {a} vs {b}");
    let (a, b) = (lcg(&mut seed), lcg(&mut seed));
    assert_eq!(enc_u64(a).cmp(&enc_u64(b)), a.cmp(&b), "u64: {a} vs {b}");
    cases += 4;
  }
  assert!(cases >= 100_000, "性质用例规模不足: {cases}");
}

/// 采样保序 + 逆映射 + 位映射（todo.md:351：编码后 memcmp 序与原始数值序等价）。
///
/// i32 值域 2^32、i64/i64 值域 2^64 无法穷举，故取「边界 + 等步长跨量级 + 确定性 LCG
/// 采样」三类样本的并集：排序去重后断言编码字节严格递增（采样域内任何反转都会被捕获），
/// 并对**每个**样本断言 `unfold(fold(v)) == v` 与定宽返回长度 —— 单次遍历同时钉死
/// 长度、可逆、采样域保序三重不变量。
///
/// 位映射：翻转编码第 k 位 ⇒ 解码值恰为 `x ^ (1 << k)`。该式对有无符号同时成立
/// （上游 `bswap(x ^ (1<<31))`：符号位翻转是与位翻转可交换的双射），因此它同时校验
/// 「大端字节序」和「符号位翻转」两个要素 —— 任何字节序错乱或漏翻符号位都会让某个 k
/// 的断言失败。与保序/逆映射共用本用例，避免两条用例被同一组变异等价击杀。
#[test]
fn fold_sampled_order_inverse_and_bit_mapping() {
  assert_fold_order_and_inverse(&sample_i32(), 4, fold_integer, unfold_integer);
  assert_fold_order_and_inverse(&sample_u32(), 4, fold_uinteger, unfold_uinteger);
  assert_fold_order_and_inverse(&sample_i64(), 8, fold_timestamp, unfold_timestamp);
  assert_fold_order_and_inverse(&sample_u64(), 8, fold_unique_id, unfold_unique_id);

  let mut seed = 0x1357_9BDF_2468_ACE0;
  for _ in 0..MUTATIONS {
    let x = lcg(&mut seed) as i32;
    let e = enc_i32(x);
    for k in 0..32usize {
      let mut m = e;
      m[3 - k / 8] ^= 1 << (k % 8);
      assert_eq!(
        unfold_integer(&m).unwrap(),
        x ^ (1i32 << k),
        "i32 位映射错误 k={k}"
      );
    }

    let x = lcg(&mut seed) as u32;
    let e = enc_u32(x);
    for k in 0..32usize {
      let mut m = e;
      m[3 - k / 8] ^= 1 << (k % 8);
      assert_eq!(unfold_uinteger(&m).unwrap(), x ^ (1u32 << k));
    }

    let x = lcg(&mut seed) as i64;
    let e = enc_i64(x);
    for k in 0..64usize {
      let mut m = e;
      m[7 - k / 8] ^= 1 << (k % 8);
      assert_eq!(unfold_timestamp(&m).unwrap(), x ^ (1i64 << k));
    }

    let x = lcg(&mut seed);
    let e = enc_u64(x);
    for k in 0..64usize {
      let mut m = e;
      m[7 - k / 8] ^= 1 << (k % 8);
      assert_eq!(unfold_unique_id(&m).unwrap(), x ^ (1u64 << k));
    }
  }
}

/// 精确逆 + 截断拒绝 + 尾部多余字节无影响（定长四类全跑）
#[test]
fn unfold_is_exact_inverse_under_structural_mutation() {
  let mut seed = 0xA5A5_1234_5678_9ABC;
  for _ in 0..MUTATIONS {
    let x = lcg(&mut seed) as i32;
    let e = enc_i32(x);
    assert_eq!(unfold_integer(&e).unwrap(), x);
    // 截断到 k < 4 字节：必须报错（上游是裸 reinterpret_cast，越界即 UB）
    for k in 0..e.len() {
      assert!(
        matches!(unfold_integer(&e[..k]), Err(Error::BufferTooSmall(4))),
        "i32 截断 {k} 字节未报错"
      );
    }
    // 尾部多余字节：只读前 4 字节，不得影响解码（前缀语义）
    for pad in [&[0u8][..], &[0xFFu8][..], &[0x12, 0x34][..], &[0xFF; 3][..]] {
      let mut ext = e.to_vec();
      ext.extend_from_slice(pad);
      assert_eq!(
        unfold_integer(&ext).unwrap(),
        x,
        "i32 尾部 {pad:?} 影响解码"
      );
    }

    let x = lcg(&mut seed) as u32;
    let e = enc_u32(x);
    assert_eq!(unfold_uinteger(&e).unwrap(), x);
    for k in 0..e.len() {
      assert!(matches!(
        unfold_uinteger(&e[..k]),
        Err(Error::BufferTooSmall(4))
      ));
    }
    let mut ext = e.to_vec();
    ext.extend_from_slice(&[0xFF; 2]);
    assert_eq!(unfold_uinteger(&ext).unwrap(), x);

    let x = lcg(&mut seed) as i64;
    let e = enc_i64(x);
    assert_eq!(unfold_timestamp(&e).unwrap(), x);
    for k in 0..e.len() {
      assert!(matches!(
        unfold_timestamp(&e[..k]),
        Err(Error::BufferTooSmall(8))
      ));
    }
    let mut ext = e.to_vec();
    ext.extend_from_slice(&[0x00; 3]);
    assert_eq!(unfold_timestamp(&ext).unwrap(), x);

    let x = lcg(&mut seed);
    let e = enc_u64(x);
    assert_eq!(unfold_unique_id(&e).unwrap(), x);
    for k in 0..e.len() {
      assert!(matches!(
        unfold_unique_id(&e[..k]),
        Err(Error::BufferTooSmall(8))
      ));
    }
    let mut ext = e.to_vec();
    ext.extend_from_slice(&[0x5A; 2]);
    assert_eq!(unfold_unique_id(&ext).unwrap(), x);
  }
}

/// Varchar：往返、容量边界不写坏、尾部多余字节按 `strlen` 截断、内嵌 NUL 的截断语义
#[test]
fn varchar_roundtrip_and_structural_mutation() {
  let mut seed = 0x0F1E_2D3C_4B5A_6978;
  for _ in 0..MUTATIONS {
    // 长度覆盖 Leng ∈ {15,16,128} 全域（含 0 与满长）
    let len = (lcg(&mut seed) as usize) % 129;
    let data: Vec<u8> = (0..len).map(|_| lcg(&mut seed) as u8 | 1).collect();
    let mut buf = vec![0xFFu8; len + 1];
    let n = fold_varchar(&mut buf, &data).unwrap();
    assert_eq!(n as usize, len + 1);
    assert_eq!(&buf[..len], &data[..]);
    assert_eq!(buf[len], 0, "NUL 终止位置错误");
    assert_eq!(unfold_varchar(&buf).unwrap(), &data[..]);

    // 容量少 1 字节：必须报错且不写坏缓冲（上游 `x.length == Leng` 时越界写 1 字节）
    let mut short = vec![0xFFu8; len];
    assert!(
      matches!(fold_varchar(&mut short, &data), Err(Error::BufferTooSmall(m)) if m == len + 1)
    );
    assert!(short.iter().all(|&b| b == 0xFF), "短缓冲被写坏");

    // 尾部多余字节：解折叠在首个 NUL 处终止（上游 strlen）
    let mut ext = buf.clone();
    ext.extend_from_slice(&[0xFF; 2]);
    assert_eq!(unfold_varchar(&ext).unwrap(), &data[..]);

    // 内嵌 NUL：折叠逐字节拷贝、解折叠按 strlen 截断（与上游 typefold.cc:63 同语义，
    // 本层不做额外扫描以避免热路径多一次遍历）
    if len > 0 {
      let pos = (lcg(&mut seed) as usize) % len;
      let mut with_nul = data.clone();
      with_nul[pos] = 0;
      let mut b2 = vec![0u8; len + 1];
      assert_eq!(fold_varchar(&mut b2, &with_nul).unwrap(), (len + 1) as u16);
      assert_eq!(unfold_varchar(&b2).unwrap(), &with_nul[..pos]);
    }
  }
}

/// 组合键游标协议：按上游 `FoldKey` 的 `pos += Fold(out + pos, ...)` 拼
/// (i32, u32, Varchar, i64) 四列，断言「拼接结果 == 各列独立编码的拼接」，
/// 且用返回长度反向游走能完全还原（`UnfoldKey` 侧依赖 `l + 1` 的游标步进）。
#[test]
fn composite_key_cursor_matches_upstream_foldkey() {
  let mut seed = 0xDEAD_BEEF_CAFE_1234;
  for _ in 0..KEYS {
    let a = lcg(&mut seed) as i32;
    let b = lcg(&mut seed) as u32;
    let len = (lcg(&mut seed) as usize) % 129;
    let s: Vec<u8> = (0..len).map(|_| lcg(&mut seed) as u8 | 1).collect();
    let t = lcg(&mut seed) as i64;

    let mut out = [0u8; 4 + 4 + 129 + 8];
    let mut pos = 0usize;
    pos += fold_integer(&mut out[pos..], a).unwrap() as usize;
    pos += fold_uinteger(&mut out[pos..], b).unwrap() as usize;
    pos += fold_varchar(&mut out[pos..], &s).unwrap() as usize;
    pos += fold_timestamp(&mut out[pos..], t).unwrap() as usize;

    let mut expect = Vec::with_capacity(pos);
    expect.extend_from_slice(&enc_i32(a));
    expect.extend_from_slice(&enc_u32(b));
    expect.extend_from_slice(&s);
    expect.push(0);
    expect.extend_from_slice(&enc_i64(t));
    assert_eq!(&out[..pos], &expect[..], "组合键布局不一致");

    let mut p = 0usize;
    assert_eq!(unfold_integer(&out[p..]).unwrap(), a);
    p += 4;
    assert_eq!(unfold_uinteger(&out[p..]).unwrap(), b);
    p += 4;
    let got = unfold_varchar(&out[p..]).unwrap();
    assert_eq!(got, &s[..]);
    p += got.len() + 1;
    assert_eq!(unfold_timestamp(&out[p..]).unwrap(), t);
    p += 8;
    assert_eq!(p, pos, "反向游走长度与折叠长度不一致");
  }
}

/// 每类 LCG 采样规模
const SAMPLES: usize = 1 << 14;
/// 等步长跨量级样本数
const STRIDES: u64 = 128;

/// 边界 + 等步长 + LCG 采样（i32 取 LCG 低 32 位，覆盖含负值的全域）
///
/// 边界项含 `typefold_test.rs` 原先手写用例断言过的全部取值（`MIN/MIN+1/-1000/-1/0/1/777/1000/MAX-1/MAX`），
/// 保证合并后无断言丢失。
fn sample_i32() -> Vec<i32> {
  let mut xs = vec![
    i32::MIN,
    i32::MIN + 1,
    -1000,
    -1,
    0,
    1,
    777,
    1000,
    i32::MAX - 1,
    i32::MAX,
  ];
  xs.extend((0..STRIDES).map(|k| i32::MIN.wrapping_add((k << 25) as i32)));
  let mut seed = 0x243F_6A88_85A3_08D3;
  xs.extend((0..SAMPLES).map(|_| lcg(&mut seed) as u32 as i32));
  xs
}

/// 边界 + 等步长 + LCG 采样
fn sample_u32() -> Vec<u32> {
  let mut xs = vec![0, 1, 777, 1000, u32::MAX / 2, u32::MAX - 1, u32::MAX];
  xs.extend((0..STRIDES).map(|k| (k << 25) as u32));
  let mut seed = 0x1319_8A2E_0370_7344;
  xs.extend((0..SAMPLES).map(|_| lcg(&mut seed) as u32));
  xs
}

/// 边界 + 等步长 + LCG 采样（i64 直接取 LCG 全 64 位）
fn sample_i64() -> Vec<i64> {
  let mut xs = vec![
    i64::MIN,
    i64::MIN + 1,
    -1000,
    -1,
    0,
    1,
    777,
    1000,
    i64::MAX - 2,
    i64::MAX - 1,
    i64::MAX,
  ];
  xs.extend((0..STRIDES).map(|k| i64::MIN.wrapping_add((k << 57) as i64)));
  let mut seed = 0xA409_3822_299F_31D0;
  xs.extend((0..SAMPLES).map(|_| lcg(&mut seed) as i64));
  xs
}

/// 边界 + 等步长 + LCG 采样
fn sample_u64() -> Vec<u64> {
  let mut xs = vec![0, 1, 1000, u64::MAX / 2, u64::MAX - 1, u64::MAX];
  xs.extend((0..STRIDES).map(|k| k << 57));
  let mut seed = 0x082E_FA98_EC4E_6C89;
  xs.extend((0..SAMPLES).map(|_| lcg(&mut seed)));
  xs
}

/// 每个样本断言「定宽返回长度 + 逆映射」，排序去重后断言编码字节严格递增
/// （即 `memcmp(编码)` 的序 == 数值序）。
fn assert_fold_order_and_inverse<T: Copy + Ord + Debug>(
  vals: &[T],
  size: usize,
  fold: impl Fn(&mut [u8], T) -> Result<u16, Error>,
  unfold: impl Fn(&[u8]) -> Result<T, Error>,
) {
  let mut rows: Vec<(T, [u8; 8])> = Vec::with_capacity(vals.len());
  for &v in vals {
    let mut buf = [0u8; 8];
    let n = fold(&mut buf, v).unwrap();
    assert_eq!(n as usize, size, "定长编码长度必须为 {size}");
    assert_eq!(unfold(&buf[..size]).unwrap(), v, "逆映射失败: {v:?}");
    rows.push((v, buf));
  }
  rows.sort_unstable_by_key(|r| r.0);
  rows.dedup_by_key(|r| r.0);
  for w in rows.windows(2) {
    let (pv, pb) = &w[0];
    let (v, b) = &w[1];
    assert!(
      pb[..size] < b[..size],
      "保序失败: {pv:?} -> {v:?} 编码序反转（{:?} >= {:?}）",
      &pb[..size],
      &b[..size]
    );
  }
}
