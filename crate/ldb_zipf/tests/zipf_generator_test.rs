//! 快速近似 Zipf 生成器（Gray 算法）回归：ζ 修正（todo.md §9 C-34）、头部分布、
//! 近似误差上界、θ≈1 路由到精确路径、值域夹紧。
//!
//! 上游：`leanstore/backend/leanstore/utils/ZipfGenerator.cpp:15-44`。
//! 语义要点：`new(ex_n, θ)` 的值域是 `[1, ex_n-1]`；θ=1 时上游 71%(n=100)/90%(n=1e6) 的采样
//! 返回 ex_n（越界），本实现在 `|θ-1|<1e-9` 时路由到精确路径。统计检验固定种子，N=2e5。

mod common;

use common::{chi2, rng, theory_zipf, tv};
use ldb_zipf::ZipfGenerator;

const N: u64 = 200_000;
/// df≈99 时 p≈1e-8 的 χ² 分位
const CHI2_MAX: f64 = 260.0;

/// 返回 (各值计数, 值域 n)；越界即失败
fn counts(ex_n: u64, theta: f64) -> (Vec<u64>, u64) {
  let n = ex_n - 1;
  let g = ZipfGenerator::new(ex_n, theta);
  let mut rng = rng();
  let mut c = vec![0u64; n as usize];
  for _ in 0..N {
    let v = g.rand(&mut rng);
    assert!((1..=n).contains(&v), "越界: {v} 不在 [1,{n}]");
    c[v as usize - 1] += 1;
  }
  (c, n)
}

#[test]
fn distribution_and_domain() {
  // P(1)=1/ζ(n,θ)、P(2)=2^-θ/ζ(n,θ) 是 Gray 算法的精确头部（字面量来自 Σi^-θ）；
  // C-34 的 bug 版 ζ=n·n^-θ 会让 θ=0.99 时 P(1)→0.95、P(2)→0.05。
  // 4 个 θ 覆盖 α>1（θ<1）/ α<0（θ>1）两支。
  for (theta, p1, p2) in [
    (0.0, 0.01, 0.01),
    (0.5, 0.053_793_507_888_897_21, 0.038_037_754_212_051_26),
    (0.99, 0.188_872_792_449_822_83, 0.095_093_253_306_029_93),
    (1.2, 0.277_543_935_968_710_74, 0.120_808_014_898_492_27),
  ] {
    let (c, n) = counts(101, theta);
    let e1 = c[0] as f64 / N as f64;
    let e2 = c[1] as f64 / N as f64;
    assert!((e1 - p1).abs() < 0.004, "θ={theta} P(1)={e1} 应≈{p1}");
    assert!((e2 - p2).abs() < 0.004, "θ={theta} P(2)={e2} 应≈{p2}");
    let t = theory_zipf(n, theta, 100);
    if theta == 0.0 {
      // θ=0 时 Gray 算法精确退化为均匀：χ² 可紧检；上界 n 概率 0.01
      let (x2, df) = chi2(&c, &t, N);
      assert!(x2 < CHI2_MAX, "θ=0 χ²={x2} df={df}");
      let pn = c[n as usize - 1] as f64 / N as f64;
      assert!((pn - 0.01).abs() < 0.002, "P(n)={pn} 应≈0.01");
    } else {
      // θ>0 时是连续近似：TV 上界见本轮报告（实测 θ=0.99 为 0.0172，θ=1.2 为 0.0192）
      let d = tv(&c, &t, N);
      assert!(d < 0.03, "θ={theta} TV={d} 超近似误差上界");
    }
  }

  // 极大 θ：ζ(100,100)=1（f64 精度内），全部质量在 1
  let g = ZipfGenerator::new(101, 100.0);
  let mut rng = rng();
  for _ in 0..10_000 {
    assert_eq!(g.rand(&mut rng), 1);
  }

  // θ>1（α<0）在大值域下仍不越界（n=1e6，内层 η 为大负数）
  let g = ZipfGenerator::new(1_000_001, 1.2);
  for _ in 0..100_000 {
    let v = g.rand(&mut rng);
    assert!((1..=1_000_000).contains(&v), "越界: {v}");
  }
}

#[test]
fn theta1_routes_to_exact() {
  // θ=1 时 P(i)=1/(i·H_100)，H_100=5.187377517639621（独立计算）。
  // 上游 Gray 在 θ=1 返回 ex_n（越界）；本实现路由到精确路径，分布必须与理想 Zipf 一致。
  let (c, n) = counts(101, 1.0);
  let t = theory_zipf(n, 1.0, 100);
  let (x2, df) = chi2(&c, &t, N);
  assert!(x2 < CHI2_MAX, "θ=1 χ²={x2} df={df}");
  let e1 = c[0] as f64 / N as f64;
  assert!(
    (e1 - 0.192_775_635_973_960_05).abs() < 0.004,
    "θ=1 P(1)={e1}"
  );
  // 值域 [1,100] 每个值都必须被采到（P(100)=1/(100·H_100)≈0.00193，期望 385 次）
  assert_eq!(
    c.iter().filter(|&&x| x > 0).count(),
    n as usize,
    "θ=1 未覆盖全域"
  );

  // n=1e6 的 θ=1：精确路径 + 调和数理论（O(n) 加法，无 powf），守 <100ms。
  // P(1)=1/H_1e6=0.06947953777315177。
  let n = 1_000_000u64;
  const M: u64 = 100_000;
  let g = ZipfGenerator::new(n + 1, 1.0);
  let mut rng = rng();
  let mut c = vec![0u64; n as usize];
  for _ in 0..M {
    let v = g.rand(&mut rng);
    assert!((1..=n).contains(&v), "越界: {v}");
    c[v as usize - 1] += 1;
  }
  let t = theory_zipf(n, 1.0, 100);
  let (x2, df) = chi2(&c, &t, M);
  assert!(x2 < CHI2_MAX, "n=1e6 θ=1 χ²={x2} df={df}");
  let p1 = c[0] as f64 / M as f64;
  assert!(
    (p1 - 0.069_479_537_773_151_77).abs() < 0.004,
    "θ=1 P(1)={p1}"
  );
}

#[test]
fn near_one_theta_routes_to_exact() {
  // |θ-1|<1e-9 一律走精确路径：Gray 的 α=1/(1-θ) 会把 η 的 1 ulp 舍入放大成
  // 2.2e-16/(1-θ) 的对数误差（|1-θ|=1e-12 时 ~2e-4，1e-15 时 ~0.2）。
  // 用 4 个代表值覆盖带内两侧与 1 的两个相邻 f64。
  let one = 1.0f64;
  for theta in [
    1.0 - 5e-10,
    1.0 + 5e-10,
    f64::from_bits(one.to_bits() - 1), // 0.9999999999999999
    f64::from_bits(one.to_bits() + 1), // 1.0000000000000002
  ] {
    let (c, n) = counts(101, theta);
    // |θ-1|<1e-9 ⇒ 与 θ=1 的 TV 差 <1e-9，用 θ=1 的理论分布即可
    let t = theory_zipf(n, 1.0, 100);
    let (x2, df) = chi2(&c, &t, N);
    assert!(x2 < CHI2_MAX, "θ={theta} χ²={x2} df={df}");
    let e1 = c[0] as f64 / N as f64;
    assert!(
      (e1 - 0.192_775_635_973_960_05).abs() < 0.004,
      "θ={theta} P(1)={e1}"
    );
  }
}

#[test]
fn outside_band_uses_the_gray_approximation() {
  // 带外（|θ-1| ≥ 1e-9）必须仍走 Gray：n=100 时它在 P(3) 上比精确高 14.6%
  // （解析值 0.073675 vs 0.064259，TV=1.71e-2），这是算法切换的固有差，不是阈值错。
  // 本用例把「路由带只有 1e-9 宽」钉在外部：若把带放宽到 ≥2e-9（或全部路由精确），
  // P(3) 会落到 0.0643±噪声。N=2e5 ⇒ σ=5.9e-4，容差 0.002 = 3.4σ、离精确值 4.7 倍。
  for theta in [1.0 - 2e-9, 1.0 + 2e-9] {
    let (c, _) = counts(101, theta);
    let p3 = c[2] as f64 / N as f64;
    assert!(
      (p3 - 0.073_675_28).abs() < 0.002,
      "θ={theta} P(3)={p3}（带外应走 Gray，解析值 0.073675）"
    );
  }
}

#[test]
#[should_panic(expected = "ex_n 必须 > 1")]
fn exn_must_exceed_one() {
  let _ = ZipfGenerator::new(1, 0.5);
}

#[test]
#[should_panic(expected = "θ 必须有限且非负")]
fn negative_theta_is_rejected() {
  let _ = ZipfGenerator::new(101, -0.5);
}

#[test]
#[should_panic(expected = "θ 必须有限且非负")]
fn infinite_theta_is_rejected() {
  // θ=inf ⇒ alpha=0、ζ=1，会静默退化成常量 1
  let _ = ZipfGenerator::new(101, f64::INFINITY);
}
