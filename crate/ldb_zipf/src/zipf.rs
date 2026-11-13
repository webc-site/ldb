//! Zipf 分布采样（合并三类上游实现，输出一律 **1-based**）：
//!
//! 1. [`RejectionInversionZipfSampler`]：精确路径（Commons Math 拒绝-反转算法）
//!    - 上游副本：`ZLeanStore/src/include/storage/space/sim/shared/RejectionInversionZipf.hpp:35-53`
//! 2. [`ZipfGenerator`]：快速近似路径（Gray 1994 算法，O(1) 内存）
//!    - 上游：`leanstore/backend/leanstore/utils/ZipfGenerator.cpp:15-44`
//!    - 上游 `zeta` 把 `pow(1.0/n, theta)` 累加 n 遍（应为 `Σ i^-θ`），本次已修正（todo.md §9 C-34）
//!    - `|θ-1| < 1e-9` 时 α=1/(1-θ) 病态（上游在 θ=1 直接返回越界的 `ex_n`），
//!      构造函数自动改走精确路径；见 [`ZipfGenerator::new`] 的数值误差推导
//! 3. [`ScrambledZipfGenerator`]：FNV-1a 打散热点
//!    - 上游：`leanstore/backend/leanstore/utils/ScrambledZipfGenerator.cpp:10-14`
//!    - FNV 与 `leanstore/backend/leanstore/utils/FNVHash.cpp:10-22` 逐字节一致
//!
//! 值域语义：三种采样器均返回 `[1, n]`（或 `[min, min+n)`），**不会返回 0**；
//! 上游两个家族（`ZipfGenerator::rand` 的 `return 1` / `RejectionInversionZipfSampler::sample`
//! 的 `k<1 ⇒ k=1`）同为 1-based，调用方须自行 `-1` 映射到 0-based key。
//!
//! 快速近似是 Gray 的连续近似，非精确离散 Zipf：实测 `n=100, θ=0.99` 时
//! `P(3)` 比理论高 14.6%，尾部低约 5%（TV=0.017）；需要精确分布时用
//! [`RejectionInversionZipfSampler`]（同参数 TV≤0.003，等于抽样噪声）。

use fastrand::Rng;

/// 拒绝-反转 Zipf 采样器（精确）。值域 `[1, n]`。
#[derive(Debug, Clone)]
pub struct RejectionInversionZipfSampler {
  n: u64,
  exponent: f64,
  h_integral_x1: f64,
  h_integral_n: f64,
  s: f64,
}

impl RejectionInversionZipfSampler {
  /// 构造：`number_of_elements ≥ 1`，`exponent ≥ 0` 且有限。
  ///
  /// `exponent == 0` 退化为均匀分布（`h(x)=1`，接受条件恒真），与 Gray 算法 θ=0 一致；
  /// 上游 Commons Math 对此抛异常，本实现允许并已用 χ² 验证。
  pub fn new(number_of_elements: u64, exponent: f64) -> Self {
    assert!(number_of_elements > 0, "元素数必须为正");
    assert!(
      exponent >= 0.0 && exponent.is_finite(),
      "指数必须非负且有限: {exponent}"
    );
    let n = number_of_elements as f64;
    let s = 2.0
      - Self::h_integral_inverse(
        Self::h_integral(2.5, exponent) - Self::h(2.0, exponent),
        exponent,
      );
    Self {
      n: number_of_elements,
      exponent,
      h_integral_x1: Self::h_integral(1.5, exponent) - 1.0,
      h_integral_n: Self::h_integral(n + 0.5, exponent),
      s,
    }
  }

  /// 采样一个 `[1, n]` 的 Zipf 变量（1-based）。
  pub fn sample(&self, rng: &mut Rng) -> u64 {
    loop {
      // u ~ U[h_integral_x1, h_integral_n)。端点开闭不影响分布：
      // ZLeanStore 副本用 [x1, n)，Commons Math 用 (x1, n]
      let u = self.h_integral_x1 + rng.f64() * (self.h_integral_n - self.h_integral_x1);
      if let Some(k) = self.step(u) {
        return k;
      }
    }
  }

  /// 单次拒绝判定：接受返回 `k`，拒绝返回 `None`（调用方重抽 `u`）。
  ///
  /// 拆出 `u` 以便对端点做逐位回归（[`exact_at_max_u_stays_in_domain`]）：
  /// `u = 1-2^-53`（fastrand `f64()` 的最大值）时 `x` 舍入到 `n+0.5`，
  /// `k=(x+0.5) as i64 = n+1`，末尾上夹紧收到 `n`；不夹紧就返回越界值。
  #[inline]
  fn step(&self, u: f64) -> Option<u64> {
    let exponent = self.exponent;
    let x = Self::h_integral_inverse(u, exponent);
    let k = (x + 0.5) as i64;
    // 浮点误差可能把 k 推出 [1, n]（上游 hpp:43-47 同）：
    // 下夹紧不可达（x 的最小值在 u=h_integral_x1 处 ≥ 0.5，故 k ≥ 1，θ=0 恰好取 0.5），
    // 上夹紧可达（u→1 时 x 舍入到 n+0.5，见上）。保留两支与上游逐行对齐。
    let k = if k < 1 {
      1
    } else if k as u64 > self.n {
      self.n
    } else {
      k as u64
    };
    let kf = k as f64;
    // 接受条件（上游 hpp:49）
    if kf - x <= self.s || u >= Self::h_integral(kf + 0.5, exponent) - Self::h(kf, exponent) {
      Some(k)
    } else {
      None
    }
  }

  /// h(x) = x^-exponent
  #[inline]
  fn h(x: f64, exponent: f64) -> f64 {
    (-exponent * x.ln()).exp()
  }

  /// ∫h 的解析式：(x^(1-θ)-1)/(1-θ)
  #[inline]
  fn h_integral(x: f64, exponent: f64) -> f64 {
    let log_x = x.ln();
    helper2((1.0 - exponent) * log_x) * log_x
  }

  /// ∫h 的反函数：(1+(1-θ)x)^(1/(1-θ))
  #[inline]
  fn h_integral_inverse(x: f64, exponent: f64) -> f64 {
    let mut t = x * (1.0 - exponent);
    if t < -1.0 {
      t = -1.0;
    }
    (helper1(t) * x).exp()
  }
}

/// helper1：log1p(x)/x 的数值稳定式（上游 hpp:72-78）
#[inline]
fn helper1(x: f64) -> f64 {
  if x.abs() > 1e-8 {
    x.ln_1p() / x
  } else {
    1.0 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x))
  }
}

/// helper2：expm1(x)/x 的数值稳定式（上游 hpp:80-86）
#[inline]
fn helper2(x: f64) -> f64 {
  if x.abs() > 1e-8 {
    x.exp_m1() / x
  } else {
    1.0 + x * 0.5 * (1.0 + x * 1.0 / 3.0 * (1.0 + 0.25 * x))
  }
}

/// Gray 快速公式的病态带：`|θ-1| < GRAY_THETA_EPS` 时改走精确路径。
///
/// 误差推导：η 的分子 `1-(2/n)^(1-θ)` 是用两个相近数相减算的，绝对误差 ≈ 1 ulp
/// (2.2e-16)，于是 `base = 1-η` 的绝对误差也 ≈ 1 ulp；`base^α`（α=1/(1-θ)）
/// 把该误差放大 `α·2.2e-16 = 2.2e-16/|1-θ|` 倍（对数尺度）。取 1e-9 ⇒ 相对误差 ≤ 2.2e-7
/// （即使 libm 的 powf 有数 ulp 误差也只有 ~1e-6），远小于抽样噪声。
/// 实测：θ=1 时上游返回 `ex_n`（n=100 71.13%、n=1e6 89.61% 越界）。
///
/// 阈值下界由 Gray 的**固有**近似误差决定：数值误差只需远小于
/// `δ(n) = TV(Gray, 精确 Zipf)`（n=100 时 1.71e-2、n=1e6 时 5.6e-3、n=1e7 时 4.1e-3），
/// 即 `|1-θ| ≥ 2.2e-16/δ(n)` ≈ 1.3e-14（n=100）。1e-9 比该下界保守 4 个数量级。
///
/// 带边界处必然存在一个 **TV=δ(n) 的分布跳变**：带内是精确分布、带外是 Gray 近似，
/// 两者的差就是 δ(n)，且**与阈值位置无关**（任意 θ 上 TV(exact, gray) 都等于 δ(n)）。
/// 实测（解析 pmf，无抽样噪声；N=1e6 采样复测一致）：n=100 时 1.71e-2、n=1e6 时 5.6e-3。
/// 需要 θ→分布连续或分布敏感的实验，请全程使用 [`RejectionInversionZipfSampler`]。
///
/// f64 边界不对称：`1.0+1e-9-1.0 = 1.0000000827e-9 > 1e-9`（`1.0-1e-9` 则是
/// 9.9999997e-10），所以名义上的 `θ=1±1e-9` 分别落在带外/带内；数值上无影响——
/// 该点的放大误差只有 2.2e-7，实测 χ²/df=88.3/99（vs 自身 Gray 解析 pmf）。
const GRAY_THETA_EPS: f64 = 1e-9;

/// Gray 公式的形态。
#[derive(Debug, Clone)]
enum Mode {
  /// `|1-θ| ≥ GRAY_THETA_EPS`：Gray 连续近似
  Gray {
    /// 值域上界（= ex_n-1）
    n: f64,
    alpha: f64,
    zetan: f64,
    eta: f64,
    /// 0.5^θ，构造期算一次（上游每次采样重算 `pow(0.5, theta)`）
    half_theta: f64,
    /// 值域上界 n（u64 形式，末尾夹紧用）
    n_max: u64,
  },
  /// `|1-θ| < GRAY_THETA_EPS`：α 病态，改用精确拒绝-反转
  Exact(RejectionInversionZipfSampler),
}

/// 快速近似 Zipf 生成器（Gray 1994；ζ 已修正，见 todo.md §9 C-34）。
///
/// 值域 `[1, ex_n-1]`；`θ` 必须有限且非负。`|θ-1| < 1e-9` 时自动改走
/// [`RejectionInversionZipfSampler`]（精确分布 P(i)∝i^-θ），因此 θ=1 可用且不越界。
#[derive(Debug, Clone)]
pub struct ZipfGenerator {
  mode: Mode,
}

impl ZipfGenerator {
  /// 构造（`ex_n = 值域上界 + 1`，与上游 `ZipfGenerator(ex_n, θ)` 对齐）。
  ///
  /// θ=1 时上游的 `α=1/(1-θ)` 为 +inf、`η=0`，`0^inf` 前的特例分支未命中就返回
  /// `ex_n`（实测 n=100 时 71.13%、n=1e6 时 89.61% 的采样越界）。上游 YCSB 的参数
  /// 校验允许 `0 ≤ θ ≤ 1`（`ZLeanStore/benchmark/src/include/benchmark/ycsb/workload.h:87`），
  /// 所以这里不 panic，而是**路由到精确路径**：语义正确、调用方零改动。
  /// 近邻带 `|θ-1| < 1e-9` 同样路由（Gray 的 α 会把 η 的 1 ulp 舍入放大到 ~1e-6 相对误差）。
  pub fn new(ex_n: u64, theta: f64) -> Self {
    assert!(ex_n > 1, "ex_n 必须 > 1（值域为 [1, ex_n-1]）");
    assert!(
      theta >= 0.0 && theta.is_finite(),
      "θ 必须有限且非负: {theta}"
    );
    if (theta - 1.0).abs() < GRAY_THETA_EPS {
      return Self {
        mode: Mode::Exact(RejectionInversionZipfSampler::new(ex_n - 1, theta)),
      };
    }
    let n = (ex_n - 1) as f64;
    let zetan = Self::zeta(ex_n - 1, theta);
    Self {
      mode: Mode::Gray {
        n,
        // Gray: α = 1/(1-θ)
        alpha: 1.0 / (1.0 - theta),
        zetan,
        // Gray: η = (1-(2/n)^(1-θ)) / (1-ζ(2,θ)/ζ(n,θ))，ζ(2,θ)=1+2^-θ
        eta: (1.0 - (2.0 / n).powf(1.0 - theta)) / (1.0 - (1.0 + 0.5f64.powf(theta)) / zetan),
        half_theta: 0.5f64.powf(theta),
        n_max: ex_n - 1,
      },
    }
  }

  /// ζ(n, θ) = Σ_{i=1..n} i^-θ（上游误写 n·n^-θ，见 todo.md §9 C-34）。
  ///
  /// θ=0 时闭式 ζ=n（上游 `url/email/wikititle` 的默认 θ），省去 O(n) 次 `powf`。
  fn zeta(n: u64, theta: f64) -> f64 {
    if theta == 0.0 {
      return n as f64;
    }
    (1..=n).map(|i| (i as f64).powf(-theta)).sum()
  }

  /// 采样 `[1, ex_n-1]`（1-based）。
  pub fn rand(&self, rng: &mut Rng) -> u64 {
    match &self.mode {
      Mode::Gray {
        n,
        alpha,
        zetan,
        eta,
        half_theta,
        n_max,
      } => gray_at(*n, *alpha, *eta, *zetan, *half_theta, *n_max, rng.f64()),
      Mode::Exact(s) => s.sample(rng),
    }
  }
}

/// Gray 映射：`u ∈ [0,1)` → `[1, n]`（n = ex_n-1）。
///
/// 末尾 `min` 兜住 u→1 的相消舍入：fastrand 的 `f64()` 最大值是 `1-2^-53`，
/// `|η|` 不太大时 `η(u-1)` 落在 1 的半 ulp 内，`ηu-η+1` 会舍入成 1.0 ⇒ g=n ⇒ k=n+1
/// （越界 1）。数学极限是 n，夹紧编译成 `cmp/csinc`，无分支、无分配。
///
/// 夹紧的语义代价（R3 枚举 fastrand `f64()` 的 top g 原像，逐位复算）：
/// 触发概率 `P(clamp) = 2^-54/|η|`（n=100 θ=0.99 实测 1.0e-15，n=1e6 实测 4.4e-16；
/// 带边缘 `|1-θ|=1e-9` 处 `|η| ∝ |1-θ|` 最小，上界 ≈5e-9）。
/// 被夹紧样本的**数学正确值就是 n**（条件 `n·|α|·2^-54 ≤ 1`，即 `n ≤ 2^54/|α|`；
/// n=1e6、|1-θ|=1e-9 时该量仅 0.056），所以 `P(n)` 无偏——夹紧是纠错而非加质量；
/// 不夹紧才会返回 `n+1`。
#[inline]
fn gray_at(n: f64, alpha: f64, eta: f64, zetan: f64, half_theta: f64, n_max: u64, u: f64) -> u64 {
  let uz = u * zetan;
  if uz < 1.0 {
    return 1;
  }
  if uz < 1.0 + half_theta {
    return 2;
  }
  (1 + (n * (eta * u - eta + 1.0).powf(alpha)) as u64).min(n_max)
}

/// 打散 Zipf（FNV 打散热点，值域 `[min, min+n)`）。
#[derive(Debug, Clone)]
pub struct ScrambledZipfGenerator {
  min: u64,
  n: u64,
  inner: ZipfGenerator,
}

impl ScrambledZipfGenerator {
  /// 构造：`n ≥ 1`，`min + n` 与 `2n` 不得溢出 u64。
  ///
  /// 内层 zipf 域取 `2n`（上游 `ScrambledZipfGenerator.hpp:20`：
  /// `zipf_generator((max-min)*2, theta)`，其返回值域为 `[1, 2n-1]`），再 `FNV % n` 打散。
  /// 构造复杂度随内层 ζ 求和：θ=0 为 O(1) 闭式，θ>0 为 O(2n) 次 `powf`（上游同阶），
  /// 因此 n 很大（>1e8）时 θ>0 的构造成本不可忽略；θ=0 的均匀负载可直接用超大值域。
  pub fn new(min: u64, n: u64, theta: f64) -> Self {
    assert!(n > 0, "值域大小必须为正");
    assert!(min.checked_add(n).is_some(), "min+n 溢出 u64");
    assert!(n.checked_mul(2).is_some(), "2n 溢出 u64");
    Self {
      min,
      n,
      inner: ZipfGenerator::new(n * 2, theta),
    }
  }

  /// 采样 `[min, min+n)`。
  pub fn rand(&self, rng: &mut Rng) -> u64 {
    self.min + fnv1a(self.inner.rand(rng)) % self.n
  }
}

/// FNV-1a 64：与上游 `FNVHash.cpp:10-22` 逐字节一致（8 轮、自最低字节起、
/// offset basis 0xcbf29ce484222325、prime 0x100000001b3）。
///
/// 8 轮共消费 64 位，输入大于 2^32 时高位同样参与（期望值见本文件 `tests` 模块）。
#[inline]
fn fnv1a(k: u64) -> u64 {
  const PRIME: u64 = 0x0000_0100_0000_01b3;
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  let mut val = k;
  for _ in 0..8 {
    h ^= val & 0xff;
    h = h.wrapping_mul(PRIME);
    val >>= 8;
  }
  h
}

#[cfg(test)]
mod tests {
  use super::{Mode, RejectionInversionZipfSampler, ZipfGenerator, fnv1a, gray_at};

  /// 期望值由独立 Python 参考实现（8 轮 FNV-1a 64，小端取字节）算出。
  /// 覆盖 >2^32 的输入：8 轮必须消费全部 64 位，否则高 32 位会被丢弃。
  #[test]
  fn fnv1a_consumes_full_u64_input() {
    assert_eq!(fnv1a(0), 0xa8c7_f832_281a_39c5);
    assert_eq!(fnv1a(1), 0x89cd_3129_1d2a_efa4);
    assert_eq!(fnv1a(1u64 << 32), 0x08cd_4c29_d1e4_7d34);
    assert_eq!(fnv1a((1u64 << 32) + 1), 0x29c7_dd31_7360_ac35);
    assert_eq!(fnv1a(u64::MAX), 0x8cf5_1a8b_fca3_883d);
  }

  /// `u = 1-2^-53` 是 fastrand `f64()` 能返回的最大值（约每次 1.1e-16 概率）。
  /// `|η|` 不太大时 `ηu-η+1` 相消舍入成 1.0 ⇒ g=n ⇒ k=n+1（越界 1）；数学极限是 n。
  /// 期望值 n 是字面量（ex_n-1）；去掉末尾 `min` 时这里会得到 101/4/4。
  #[test]
  fn gray_at_max_u_stays_in_domain() {
    let u = f64::from_bits(1.0f64.to_bits() - 1); // 0.9999999999999999
    for (ex_n, theta) in [(101u64, 0.99f64), (4, 0.99), (4, 1.2), (101, 0.5)] {
      let g = ZipfGenerator::new(ex_n, theta);
      match g.mode {
        Mode::Gray {
          n,
          alpha,
          zetan,
          eta,
          half_theta,
          n_max,
        } => assert_eq!(
          gray_at(n, alpha, eta, zetan, half_theta, n_max, u),
          ex_n - 1,
          "ex_n={ex_n} θ={theta} 在 u=1-2^-53 越界"
        ),
        Mode::Exact(_) => panic!("θ={theta} 不应路由到精确路径"),
      }
    }
  }

  /// 精确路径的**上夹紧可达**：`u_max = 1-2^-53` 时 `x` 舍入到 `n+0.5`，
  /// `k=(x+0.5) as i64 = n+1`（越界）；`step` 的 `k > n` 分支把它收回 `n`。
  /// 4 组 (n, θ) 均由 R3 报告逐位枚举筛出；去掉上夹紧时 `step` 返回 `Some(n+1)`。
  #[test]
  fn exact_at_max_u_stays_in_domain() {
    let f = f64::from_bits(1.0f64.to_bits() - 1);
    for (n, theta) in [(2u64, 0.99f64), (4, 0.0), (10, 0.5), (1_000_000, 0.5)] {
      let s = RejectionInversionZipfSampler::new(n, theta);
      let u = s.h_integral_x1 + f * (s.h_integral_n - s.h_integral_x1);
      let x = RejectionInversionZipfSampler::h_integral_inverse(u, theta);
      assert!(
        (x + 0.5) as i64 > n as i64,
        "n={n} θ={theta} 未触发上夹紧（用例失去对抗性）"
      );
      assert_eq!(s.step(u), Some(n), "n={n} θ={theta} 在 u_max 越界");
    }
  }
}
