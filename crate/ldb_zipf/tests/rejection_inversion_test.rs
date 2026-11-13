//! 拒绝-反转 Zipf 采样器（精确路径）回归：分布对理论 P(i)∝i^-θ、值域、退化域、边界。
//!
//! 对拍：Commons Math `RejectionInversionZipfSampler`（上游副本
//! `ZLeanStore/src/include/storage/space/sim/shared/RejectionInversionZipf.hpp:35-53`）。
//! 统计检验固定种子（common::SEED）、N=2e5（n=1e6 时 N=1e5）；
//! χ² 阈值 260 ≈ df=99 的 p=1e-8 分位，真 bug 的 χ² 在 1e3 量级（实测值见本轮报告）。

mod common;

use common::{chi2, rng, theory_zipf, tv};
use ldb_zipf::RejectionInversionZipfSampler;

/// df≈99 时 p≈1e-8 的 χ² 分位
const CHI2_MAX: f64 = 260.0;

fn counts(n: u64, theta: f64, samples: u64) -> Vec<u64> {
  let s = RejectionInversionZipfSampler::new(n, theta);
  let mut rng = rng();
  let mut c = vec![0u64; n as usize];
  for _ in 0..samples {
    let v = s.sample(&mut rng);
    assert!((1..=n).contains(&v), "越界: {v} 不在 [1,{n}]");
    c[v as usize - 1] += 1;
  }
  c
}

#[test]
fn matches_zipf_theory() {
  const N: u64 = 200_000;

  // (a) θ=0 退化为均匀分布：P=1/100（字面量 0.01），且值域上界可达
  let n = 100u64;
  let c = counts(n, 0.0, N);
  let (x2, df) = chi2(&c, &theory_zipf(n, 0.0, 100), N);
  assert!(x2 < CHI2_MAX, "θ=0 均匀性 χ²={x2} df={df}");
  let p1 = c[0] as f64 / N as f64;
  assert!((p1 - 0.01).abs() < 0.002, "θ=0 P(1)={p1}");
  assert!(c[n as usize - 1] > 0, "上界 {n} 未被采到");

  // (b) θ=0.5 / 0.99：χ² 对理想离散 Zipf；P(1) 钉死 ζ(100,θ)（字面量）
  for (theta, p1) in [
    (0.5, 0.053_793_507_888_897_21),
    (0.99, 0.188_872_792_449_822_83),
  ] {
    let c = counts(n, theta, N);
    let (x2, df) = chi2(&c, &theory_zipf(n, theta, 100), N);
    assert!(x2 < CHI2_MAX, "θ={theta} χ²={x2} df={df}");
    let e1 = c[0] as f64 / N as f64;
    assert!((e1 - p1).abs() < 0.005, "θ={theta} P(1)={e1} 应≈{p1}");
  }
}

#[test]
fn small_domains_match_theory() {
  // (c) 退化/小值域：n=1 单点；n=2 两点分布 P(1)=1/(1+2^-0.99)=0.66512456；
  //     n=3, θ=1.2（α<0 分支）P=(0.587249, 0.255615, 0.157136)
  const N: u64 = 200_000;
  let c1 = counts(1, 0.5, 10_000);
  assert_eq!(c1, vec![10_000]);
  let c2 = counts(2, 0.99, N);
  let p1 = c2[0] as f64 / N as f64;
  assert!(
    (p1 - 0.665_124_564_266_430_5).abs() < 0.006,
    "n=2 P(1)={p1}"
  );
  let c3 = counts(3, 1.2, N);
  let e = [
    c3[0] as f64 / N as f64,
    c3[1] as f64 / N as f64,
    c3[2] as f64 / N as f64,
  ];
  for (got, want) in e.iter().zip([
    0.587_248_784_475_827_6,
    0.255_614_880_060_198,
    0.157_136_335_463_974_35,
  ]) {
    assert!((got - want).abs() < 0.006, "n=3 P={got} 应≈{want}");
  }

  // (d) 极大 θ：ζ(100,100)=1（f64 精度内），全部质量在 1
  let s = RejectionInversionZipfSampler::new(100, 100.0);
  let mut rng = rng();
  let mut ones = 0u32;
  for _ in 0..10_000 {
    ones += u32::from(s.sample(&mut rng) == 1);
  }
  assert!(ones >= 9_999, "θ=100 只有 {ones}/10000 落在 1");
}

#[test]
fn n1m_matches_zipf() {
  // n=1e6：大值域下 f64 精度与分组 χ²（θ=0.5 与 θ>1 各一）。
  // N 取 1e5（每箱期望 ≈1e3）以守住单测 <100ms（E2）；N=1e6 的实测证据见本轮报告。
  let n = 1_000_000u64;
  const N: u64 = 100_000;
  for theta in [0.5, 1.2] {
    let c = counts(n, theta, N);
    let t = theory_zipf(n, theta, 100);
    let (x2, df) = chi2(&c, &t, N);
    assert!(x2 < CHI2_MAX, "n=1e6 θ={theta} χ²={x2} df={df}");
    let d = tv(&c, &t, N);
    assert!(d < 0.015, "n=1e6 θ={theta} TV={d}");
  }
}

#[test]
#[should_panic(expected = "元素数必须为正")]
fn rejects_zero_elements() {
  let _ = RejectionInversionZipfSampler::new(0, 0.5);
}

#[test]
#[should_panic(expected = "指数必须非负且有限")]
fn rejects_negative_exponent() {
  let _ = RejectionInversionZipfSampler::new(10, -0.5);
}

#[test]
#[should_panic(expected = "指数必须非负且有限")]
fn rejects_infinite_exponent() {
  // +inf 会让 h_integral 产生 0*inf=NaN
  let _ = RejectionInversionZipfSampler::new(10, f64::INFINITY);
}
