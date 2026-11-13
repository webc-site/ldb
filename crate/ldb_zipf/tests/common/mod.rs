//! 统计检验辅助（三个测试二进制共用）：固定种子、χ²、TV 距离。

use fastrand::Rng;

/// 固定种子：所有统计测试可复现，禁止用 `Rng::new()`（不可复现 ⇒ flaky）
pub const SEED: u64 = 0x5eed_1234_5678_9abc;

/// 固定种子的 RNG
pub fn rng() -> Rng {
  Rng::with_seed(SEED)
}

/// 理想离散 Zipf 理论分布 P(i) ∝ i^-θ，按连续分组聚成 `nbins` 个箱。
///
/// 值 `i` 落入箱 `(i-1)*nbins/n`，与 [`chi2`] 的分组严格一致。
/// θ=1 用精确调和数 `1/i`（省掉 n 次 `powf`：n=1e6 的 θ=1 用例靠它守 <100ms）。
pub fn theory_zipf(n: u64, theta: f64, nbins: usize) -> Vec<f64> {
  let mut t = vec![0.0f64; nbins];
  let mut total = 0.0f64;
  for i in 1..=n {
    let w = if theta == 1.0 {
      1.0 / (i as f64)
    } else {
      (i as f64).powf(-theta)
    };
    total += w;
    t[((i - 1) as usize * nbins) / n as usize] += w;
  }
  for x in &mut t {
    *x /= total;
  }
  t
}

/// 分组 χ²：期望频数 < 5 的箱并入丢弃（标准做法），返回 (χ², 自由度)。
pub fn chi2(obs: &[u64], prob: &[f64], samples: u64) -> (f64, usize) {
  let n = obs.len();
  let nb = prob.len();
  assert_eq!(n % nb.max(1), 0, "箱数必须整除值域");
  let mut x = 0.0f64;
  let mut used = 0usize;
  for (b, p) in prob.iter().enumerate() {
    let lo = b * n / nb;
    let hi = (b + 1) * n / nb;
    let o: u64 = obs[lo..hi].iter().sum();
    let e = p * samples as f64;
    if e >= 5.0 {
      x += (o as f64 - e) * (o as f64 - e) / e;
      used += 1;
    }
  }
  (x, used.saturating_sub(1))
}

/// 全变差距离 TV = ½·Σ|p̂ - p|
pub fn tv(obs: &[u64], prob: &[f64], samples: u64) -> f64 {
  let n = obs.len();
  let nb = prob.len();
  0.5
    * (0..nb)
      .map(|b| {
        let lo = b * n / nb;
        let hi = (b + 1) * n / nb;
        let o: u64 = obs[lo..hi].iter().sum();
        (o as f64 / samples as f64 - prob[b]).abs()
      })
      .sum::<f64>()
}
