//! 打散 Zipf（ScrambledZipfGenerator）回归：FNV-1a 映射、内层 2n 域、值域偏移。
//!
//! 上游：`leanstore/backend/leanstore/utils/ScrambledZipfGenerator.cpp:10-14`
//! （`zipf_generator.rand()` 后 `min + FNV::hash(v) % n`，内层域 `(max-min)*2`）。
//! 期望值全部为字面量（FNV-1a 64 位对 8 字节小端做 8 轮，独立 Python 参考实现算出）。

use std::time::Instant;

use fastrand::Rng;
use ldb_zipf::ScrambledZipfGenerator;

/// 固定种子（可复现）
const SEED: u64 = 0x5eed_1234_5678_9abc;

#[test]
fn fnv_pushforward_and_domain() {
  // (a) θ=0 ⇒ 内层是 [1,127] 均匀（n=64，内层域 2n-1=127）。
  //     FNV-1a(z)%64 把 127 个 z 映射到 64 个残差：除残差 5 只命中 1 次外其余各 2 次，
  //     ⇒ 理论 P=2/127（残差 5 为 1/127）。任何 FNV/域/取模错误都会打乱这组计数。
  let n = 64u64;
  let g = ScrambledZipfGenerator::new(0, n, 0.0);
  let mut rng = Rng::with_seed(SEED);
  let mut c = vec![0u64; n as usize];
  let samples = 200_000u64;
  for _ in 0..samples {
    let v = g.rand(&mut rng);
    assert!(v < n, "越界: {v}");
    c[v as usize] += 1;
  }
  // 2/127*2e5 = 3149.6；1/127*2e5 = 1574.8（±250 ≈ 4.5σ）
  for (i, &x) in c.iter().enumerate() {
    let expect = if i == 5 { 1575.0 } else { 3150.0 };
    assert!(
      (x as f64 - expect).abs() < 250.0,
      "残差 {i} 计数 {x} 偏离期望 {expect}"
    );
  }
  assert_eq!(c.iter().filter(|&&x| x > 0).count(), 64, "打散塌缩");

  // (b) 值域 [min, min+n)：输出必须落在偏移后的区间，且 64 个成员都被命中
  let (min, n) = (1_000u64, 64u64);
  let g = ScrambledZipfGenerator::new(min, n, 0.99);
  let mut seen = vec![false; n as usize];
  for _ in 0..20_000 {
    let v = g.rand(&mut rng);
    assert!((min..min + n).contains(&v), "越界: {v}");
    seen[(v - min) as usize] = true;
  }
  assert!(seen.iter().all(|&s| s), "有值域成员从未被采到");

  // (c) n=1：单点域，内层域退化为 [1,1]
  let g = ScrambledZipfGenerator::new(7, 1, 0.5);
  for _ in 0..100 {
    assert_eq!(g.rand(&mut rng), 7);
  }
}

#[test]
fn hot_residues_at_n1000_pin_fnv_prime_and_rounds() {
  // n=1000：fnv(1)%1000=996（原像 1/567/1555）、fnv(2)%1000=223（原像 2/1134/1564）。
  // 期望占比 = 理想 zipf 在这些原像集上的和（字面量）：P(996)=0.11831915，P(223)=0.05961116。
  // 改动 FNV prime / 轮数 / 字节序都会把这些概率搬到别的残差上
  // （n=64 的分组计数对 prime 不敏感，必须用 n=1000 才钉得住）。
  let n = 1000u64;
  let g = ScrambledZipfGenerator::new(0, n, 0.99);
  let mut rng = Rng::with_seed(SEED);
  let mut c = vec![0u64; n as usize];
  let samples = 200_000u64;
  for _ in 0..samples {
    c[g.rand(&mut rng) as usize] += 1;
  }
  let p1 = c[996] as f64 / samples as f64;
  let p2 = c[223] as f64 / samples as f64;
  assert!((p1 - 0.118_319_15).abs() < 0.006, "残差 996 占比 {p1}");
  assert!((p2 - 0.059_611_16).abs() < 0.004, "残差 223 占比 {p2}");
}

#[test]
fn theta1_pushforward_matches_exact_zipf() {
  // θ=1 路由到精确路径后：内层 P(z)=1/(z·H_1999)，FNV 推前到 n=1000 后
  // P(996)=0.12257555912457463、P(223)=0.06132664577740246（独立 Python 计算）；
  // 1999 个原像只覆盖 827 个残差（其余 173 个无原像）。
  // 上游/旧实现在 θ=1 时内层恒返回 ex_n=2000 ⇒ 全部质量塌缩到 1 个残差。
  let n = 1000u64;
  let g = ScrambledZipfGenerator::new(0, n, 1.0);
  let mut rng = Rng::with_seed(SEED);
  let mut c = vec![0u64; n as usize];
  let samples = 200_000u64;
  for _ in 0..samples {
    let v = g.rand(&mut rng);
    assert!(v < n, "越界: {v}");
    c[v as usize] += 1;
  }
  let p1 = c[996] as f64 / samples as f64;
  let p2 = c[223] as f64 / samples as f64;
  assert!(
    (p1 - 0.122_575_559_124_574_63).abs() < 0.006,
    "残差 996 占比 {p1}"
  );
  assert!(
    (p2 - 0.061_326_645_777_402_46).abs() < 0.004,
    "残差 223 占比 {p2}"
  );
  assert_eq!(
    c.iter().filter(|&&x| x > 0).count(),
    827,
    "θ=1 的可达残差数应为 827（FNV 原像覆盖），塌缩或 FNV 变化都会改变它"
  );
}

#[test]
fn large_domain_above_2_pow_32() {
  // n=3·2^32：内层域 2n≈2.6e10 远超 u32（FNV 输入 u64）；θ=0 的 ζ=n 闭式让构造为 O(1)
  // （若退回 O(n) 求和需 ~1e10 次 powf，本用例会跑分钟级——1s 上限即回归护栏）。
  const N: u64 = 3u64 << 32;
  const MIN: u64 = 1_000_000;
  let t = Instant::now();
  let g = ScrambledZipfGenerator::new(MIN, N, 0.0);
  assert!(
    t.elapsed().as_secs() < 1,
    "θ=0 构造应为 O(1)，实际 {:?}",
    t.elapsed()
  );
  let mut rng = Rng::with_seed(SEED);
  let mut min_seen = u64::MAX;
  let mut max_seen = 0u64;
  for _ in 0..100_000 {
    let v = g.rand(&mut rng);
    assert!((MIN..MIN + N).contains(&v), "越界: {v}");
    min_seen = min_seen.min(v);
    max_seen = max_seen.max(v);
  }
  // 均匀分布 10 万样本必然跨过 2^32 与 2·2^32（P(全在某侧)=(2/3)^1e5）：
  // 任何把 n / FNV 结果截断成 u32 的实现都落在 [MIN, MIN+2^32) 内，被这两条击杀。
  assert!(min_seen < MIN + (1u64 << 32), "最小值 {min_seen} 偏大");
  assert!(max_seen > MIN + (2u64 << 32), "最大值 {max_seen} 偏小");
}

#[test]
#[should_panic(expected = "值域大小必须为正")]
fn rejects_zero_n() {
  let _ = ScrambledZipfGenerator::new(0, 0, 0.5);
}

#[test]
#[should_panic(expected = "min+n 溢出 u64")]
fn rejects_domain_overflow() {
  let _ = ScrambledZipfGenerator::new(1, u64::MAX, 0.5);
}

#[test]
#[should_panic(expected = "2n 溢出 u64")]
fn rejects_double_n_overflow() {
  // min+n 不溢出，仅 2n 溢出
  let _ = ScrambledZipfGenerator::new(0, u64::MAX / 2 + 1, 0.5);
}
