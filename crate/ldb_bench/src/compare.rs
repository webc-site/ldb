//! 标准基准套件（redb `lmdb_benchmark` 维度；todo.md §5）。
//!
//! 后端统一 [`BenchDb`] trait：ldb（本库）/ RocksDB / redb 三实现由 M5 接线期
//! 注入 —— 本文件承载 10 个 phase 的驱动与指标采集。
//!
//! # 参数对齐（RocksDB `db_bench`）
//! [`BenchConfig`] 的每个字段都给出对应的 db_bench flag（字段文档注明；`key_dist`
//! 为 ldb 侧专属、db_bench 无对应 flag），因此同一份配置可在 ldb 与 RocksDB 上跑
//! 同一 phase 序列。反向映射（配置 → `db_bench` 命令行）见
//! [`crate::db_bench::BenchConfig::db_bench_args`]。
//!
//! # 可复现性
//! 随机相位（`random_reads` / `range_reads` / `removals`）的键序列由 `seed` 派生：
//! 每个 phase 以 `seed ^ FNV1a(phase 名)` 独立播种，故**操作序列只依赖配置、与
//! phase 执行顺序无关**。每次采样的键 id 折进 [`PhaseResult::seq_hash`]（FNV-1a）；
//! 同配置两次运行的 `seq_hash` 必须逐位一致（耗时（`ms`/`ops`）可抖动，操作序列不可变）。
//!
//! # 输出契约
//! [`BenchRunner::report_json`] 产出**合法** JSON（sonic-rs）；WAF 未接线时显式
//! 输出 `"status":"not-wired"` 且 `"value":null`，绝不把缺失值伪装成 `0`
//! （见 [`WafOut`]）。稳定的结构字段（无耗时）由 [`BenchRunner::report_static_json`]
//! 导出，供 golden 回归逐字节比对。

use std::{fmt, sync::Arc, time::Instant};

use fastrand::Rng;
use ldb_stat::StatCollector;
use ldb_zipf::ZipfGenerator;

use crate::error::{Error, Result};

/// JSON schema 版本串。
const SCHEMA: &str = "ldb_bench.v1";
/// `range_reads` 每个区间的连续条数（todo.md §5「随机起点连续读取 100 条 KV」）。
const RANGE_LEN: usize = 100;
/// `batch_writes` 默认批次大小（todo.md §5「每批 100 / 1000 条」）。
const DEFAULT_BATCH: usize = 100;
/// 键/值填充字节（与 db_bench 一致：内容不参与比较，仅定长）。
const VAL_FILL: u8 = 0x07;
/// 大对象填充字节。
const BLOB_FILL: u8 = 0xAB;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// 键 id → 十进制位数（编译期可折叠）。
const fn decimal_digits(mut v: u64) -> usize {
  let mut d = 1;
  while v >= 10 {
    v /= 10;
    d += 1;
  }
  d
}

/// ldb 侧键采样分布。
///
/// 注意：RocksDB `db_bench`（11.8.1）**没有** `--key_distribution`/`--zipf_const`
/// 这类键分布 flag（实机 `--help` 0 命中，见报告 §2）。因此本枚举描述的是 **ldb 侧**
/// 采样：`Uniform` 对应 db_bench 默认（`--read_random_exp_range=0`）；`Zipfian(θ)` 在
/// db_bench 侧无法精确表达，只能用 [`BenchConfig::read_random_exp_range`] 近似。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyDist {
  /// 均匀分布（db_bench 默认，无需分布 flag）。
  Uniform,
  /// Zipf 分布（仅 ldb 侧；db_bench 侧近似见 [`BenchConfig::read_random_exp_range`]）。
  ///
  /// `theta` 与 YCSB 一致，取值 `0.0..=1.0`；`theta=1.0` 由 [`ZipfGenerator`]
  /// 自动路由到精确路径（ldb_zipf 契约）。
  Zipfian { theta: f64 },
}

/// 压缩类型（对齐 db_bench `--compression_type`，仅作配置记录；实际编解压由后端决定）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
  /// 不压缩。
  None,
  /// LZ4。
  Lz4,
  /// Zstd（ldb 侧对应 `ldb_codec` 的 `mzstd`）。
  Zstd,
}

/// 基准配置（字段 ↔ db_bench flag 一一对应）。
#[derive(Debug, Clone, Copy)]
pub struct BenchConfig {
  /// 记录数。db_bench：`--num`。
  pub num_records: u64,
  /// 键长（字节）。db_bench：`--key_size`（默认 16）。
  pub key_len: usize,
  /// 值长（字节）。db_bench：`--value_size`（默认 100）。
  pub val_len: usize,
  /// ldb 侧键采样分布（db_bench 无对应 flag；倾斜读的近似旋钮见 `read_random_exp_range`）。
  pub key_dist: KeyDist,
  /// db_bench `--read_random_exp_range`（仅 `readrandom`/`multireadrandom` 生效）。
  ///
  /// db_bench 侧键 = `num·exp(-r)`（`r ~ U[0,R)`），再经置换；源码
  /// `GetRandomKey`（`db_bench_tool.cc:7314`）使质量律**恒为 ∝1/rank、截断于
  /// `num·e^{-R}`**。故 `R ≥ ln(num)` 时支撑覆盖全键域 ⇒ 与 **scrambled Zipf(θ=1)**
  /// 完全一致；YCSB 默认 θ≈0.99 时形状 TV≈0.01–0.02（近似）；θ≪1 无精确表达。
  /// `0.0` = 均匀（db_bench 默认）。仅当 `key_dist = Uniform` 时有意义（ldb 侧倾斜
  /// 采样用 [`KeyDist::Zipfian`]）。
  pub read_random_exp_range: f64,
  /// 随机种子。db_bench：`--seed`（db_bench 侧 `0` 表示以时间派生，非确定性；
  /// ldb_bench 的 `0` 仍是确定种子，仅映射值相同）。
  pub seed: u64,
  /// 每次提交是否 fdatasync。db_bench：`--sync`（默认 0）。
  pub sync: bool,
  /// 块缓存字节。db_bench：`--cache_size`（db_bench 默认 32 MiB，`db_bench_tool.cc:628`）。
  pub cache_size: u64,
  /// 写缓冲字节。db_bench：`--write_buffer_size`（默认 64 MiB）。
  pub write_buffer_size: u64,
  /// 复用既有库（不重建）。db_bench：`--use_existing_db`（默认 0）。
  pub use_existing_db: bool,
  /// 压缩类型。db_bench：`--compression_type`。
  pub compression: Compression,
}

impl Default for BenchConfig {
  /// 与 db_bench 默认值对齐：`--num=1000000 --key_size=16 --value_size=100
  /// --read_random_exp_range=0 --sync=0 --cache_size=32MiB --write_buffer_size=64MiB
  /// --use_existing_db=0`；`compression` 取 `None`（ldb 默认不压缩）。
  fn default() -> Self {
    Self {
      num_records: 1_000_000,
      key_len: 16,
      val_len: 100,
      key_dist: KeyDist::Uniform,
      read_random_exp_range: 0.0,
      seed: 0,
      sync: false,
      cache_size: 32 << 20,
      write_buffer_size: 64 << 20,
      use_existing_db: false,
      compression: Compression::None,
    }
  }
}

impl BenchConfig {
  /// 校验配置；非法即 [`Error::InvalidConfig`]，在启动前 fail-fast。
  ///
  /// 硬约束：`num_records ≥ 1`；`key_len ≥ num_records 的十进制位数`（否则高位键被
  /// 截断产生键碰撞，静默改变负载）；`val_len ≥ 1`；Zipf 指数有限且 `∈ [0,1]`。
  pub fn validate(&self) -> Result<()> {
    if self.num_records == 0 {
      return Err(Error::InvalidConfig("num_records 必须 > 0".into()));
    }
    let need = decimal_digits(self.num_records);
    if self.key_len < need {
      return Err(Error::InvalidConfig(format!(
        "key_len={} 不足以容纳 num_records={} 的 {need} 位十进制键",
        self.key_len, self.num_records
      )));
    }
    if self.val_len == 0 {
      return Err(Error::InvalidConfig("val_len 必须 > 0".into()));
    }
    if let KeyDist::Zipfian { theta } = self.key_dist
      && (!theta.is_finite() || !(0.0..=1.0).contains(&theta))
    {
      return Err(Error::InvalidConfig(format!(
        "zipf theta 必须在 [0,1]: {theta}"
      )));
    }
    Ok(())
  }
}

/// 基准后端抽象（ldb / rocksdb / redb 适配）。
pub trait BenchDb {
  /// 名称。
  fn name(&self) -> &'static str;
  /// 批量装载一批 KV。
  fn put(&mut self, key: &[u8], val: &[u8]) -> Result<()>;
  /// 点查；返回是否命中。
  fn get(&mut self, key: &[u8], out: &mut Vec<u8>) -> Result<bool>;
  /// 删除；返回是否删除了存在的键。
  fn del(&mut self, key: &[u8]) -> Result<bool>;
  /// 全表计数。
  fn len(&mut self) -> Result<u64>;
  /// 区间扫描：自 `lo` 起最多 `cnt` 条，返回条数。
  fn scan(&mut self, lo: &[u8], cnt: usize) -> Result<usize>;
  /// 是否每次提交 `sync`。
  fn set_sync(&mut self, on: bool);
  /// 提交当前批次/事务（db_bench 每次 benchmark 迭代即一次提交；立即落库的后端可空实现）。
  fn commit(&mut self) -> Result<()> {
    Ok(())
  }
  /// 是否为空（`len` 的伴随谓词 —— clippy `len_without_is_empty`）。
  fn is_empty(&mut self) -> Result<bool> {
    Ok(self.len()? == 0)
  }
  /// 大对象写。
  fn put_blob(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
    self.put(key, val)
  }
  /// 大对象读。
  fn get_blob(&mut self, key: &[u8], out: &mut Vec<u8>) -> Result<bool> {
    self.get(key, out)
  }
  /// 是否已接线 [`StatCollector`] 事件计数。
  ///
  /// 决定报告里 WAF 字段能否取真实值：未接线时 [`BenchRunner::report_json`]
  /// 输出 `"not-wired"` 而非伪造的 `0.0`。
  fn stat_wired(&self) -> bool {
    false
  }
  /// 注入统计收集器句柄。
  ///
  /// 接线 ldb_stat 事件的后端（实现 `record_page_evict`/`record_logical_write` 等）
  /// 在此保存 `stat` 并在操作处 `record_*`，随后 [`BenchDb::stat_wired`] 返回 `true`；
  /// 默认空实现 = 未接线。
  fn set_stat(&mut self, _stat: Arc<StatCollector>) {}
  /// 关闭（统计落盘）。
  fn close(&mut self) -> Result<()> {
    Ok(())
  }
}

/// 单 phase 结果。
#[derive(Debug, Clone, Copy)]
pub struct PhaseResult {
  /// phase 名。
  pub phase: &'static str,
  /// 耗时 ms。
  pub ms: u64,
  /// 吞吐 ops/s（`count / 秒`）。
  pub ops: f64,
  /// 逻辑操作数。
  pub count: u64,
  /// 命中数（读/删成功数）。
  pub hits: u64,
  /// 未命中数。
  pub misses: u64,
  /// 传输字节数（无字节语义的相位为 0）。
  pub bytes: u64,
  /// 操作序列哈希（FNV-1a over 访问的键字节，确定性证据）。
  pub seq_hash: u64,
}

/// bench 内部计数（phase 体返回，`timed` 换算成 [`PhaseResult`]）。
#[derive(Debug, Default, Clone, Copy)]
struct Counts {
  ops: u64,
  hits: u64,
  misses: u64,
  bytes: u64,
  seq_hash: u64,
}

/// FNV-1a 续算：把 `bytes` 折进已有哈希。
#[inline]
fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
  for &b in bytes {
    h ^= u64::from(b);
    h = h.wrapping_mul(FNV_PRIME);
  }
  h
}

/// 键 id → 定长十进制字节（左补 `'0'`）；`id` 位数超过 `key_len` 时高位被截断，
/// 故 [`BenchConfig::validate`] 强制 `key_len ≥ num_records` 的位数。
#[inline]
fn encode_key(id: u64, key_len: usize, out: &mut Vec<u8>) {
  out.clear();
  out.resize(key_len, b'0');
  let mut n = id;
  let mut i = key_len;
  while i > 0 {
    i -= 1;
    out[i] = b'0' + (n % 10) as u8;
    n /= 10;
    if n == 0 {
      break;
    }
  }
}

/// 十进制追加（大对象键 `blob-<i>`）。
#[inline]
fn push_decimal(mut v: u64, out: &mut Vec<u8>) {
  let mut buf = [0u8; 20];
  let mut i = buf.len();
  loop {
    i -= 1;
    buf[i] = b'0' + (v % 10) as u8;
    v /= 10;
    if v == 0 {
      break;
    }
  }
  out.extend_from_slice(&buf[i..]);
}

/// 由全局 seed + phase 名派生的独立 RNG（操作序列与 phase 顺序无关）。
#[inline]
fn phase_rng(seed: u64, phase: &str) -> Rng {
  Rng::with_seed(seed ^ fnv1a(phase.as_bytes(), FNV_OFFSET))
}

/// 采样一个 `0`-based 键 id，`∈ [0, n)`。
#[inline]
fn sample_id(rng: &mut Rng, dist: KeyDist, n: u64, zipf: &mut Option<ZipfGenerator>) -> u64 {
  match dist {
    KeyDist::Uniform => rng.u64(0..n),
    KeyDist::Zipfian { theta } => {
      // ZipfGenerator 值域 [1, ex_n-1]（1-based）；ex_n=n+1 覆盖 id [0,n)。
      let z = zipf.get_or_insert_with(|| ZipfGenerator::new(n + 1, theta));
      z.rand(rng) - 1
    }
  }
}

/// 计时包装：跑 `body` 得到 [`Counts`]，换算 [`PhaseResult`]。
fn timed<D: BenchDb + ?Sized>(
  cfg: &BenchConfig,
  db: &mut D,
  phase: &'static str,
  body: impl FnOnce(&mut D, &mut Rng) -> Result<Counts>,
) -> Result<PhaseResult> {
  let mut rng = phase_rng(cfg.seed, phase);
  let t = Instant::now();
  let c = body(db, &mut rng)?;
  let secs = t.elapsed().as_secs_f64();
  let ms = u64::try_from((secs * 1000.0) as u128).unwrap_or(u64::MAX);
  let ops = if secs > 0.0 { c.ops as f64 / secs } else { 0.0 };
  Ok(PhaseResult {
    phase,
    ms,
    ops,
    count: c.ops,
    hits: c.hits,
    misses: c.misses,
    bytes: c.bytes,
    seq_hash: c.seq_hash,
  })
}

/// 断言后端长度等于期望（相位后置不变量）。
fn check_len<D: BenchDb + ?Sized>(db: &mut D, expect: u64, phase: &str) -> Result<()> {
  let len = db.len()?;
  if len != expect {
    return Err(Error::Invariant(format!(
      "{phase} 后 len={len} 与预期 {expect} 不一致"
    )));
  }
  Ok(())
}

/// `bulk_load`：顺序装载 `num_records` 条 KV（须在空库上运行）。
fn bulk_load<D: BenchDb + ?Sized>(cfg: &BenchConfig, db: &mut D) -> Result<PhaseResult> {
  let n = cfg.num_records;
  let val = vec![VAL_FILL; cfg.val_len];
  let key_bytes = cfg.key_len as u64 + cfg.val_len as u64;
  let r = timed(cfg, db, "bulk_load", |db, _rng| {
    let mut key = Vec::with_capacity(cfg.key_len);
    let mut h = FNV_OFFSET;
    for i in 0..n {
      encode_key(i, cfg.key_len, &mut key);
      h = fnv1a(&key, h);
      db.put(&key, &val)?;
    }
    Ok(Counts {
      ops: n,
      bytes: n * key_bytes,
      seq_hash: h,
      ..Default::default()
    })
  })?;
  check_len(db, n, "bulk_load")?;
  Ok(r)
}

/// `individual_writes`：每记录一次独立提交。
fn individual_write<D: BenchDb + ?Sized>(cfg: &BenchConfig, db: &mut D) -> Result<PhaseResult> {
  let n = cfg.num_records;
  db.set_sync(cfg.sync);
  let r = timed(cfg, db, "individual_writes", |db, _rng| {
    let mut key = Vec::with_capacity(cfg.key_len);
    let mut h = FNV_OFFSET;
    for i in 0..n {
      encode_key(i, cfg.key_len, &mut key);
      h = fnv1a(&key, h);
      db.put(&key, b"v")?;
      db.commit()?;
    }
    Ok(Counts {
      ops: n,
      bytes: n * (cfg.key_len as u64 + 1),
      seq_hash: h,
      ..Default::default()
    })
  })?;
  check_len(db, n, "individual_writes")?;
  Ok(r)
}

/// `batch_writes`：每 `batch_size` 条一次提交。
fn batch_write<D: BenchDb + ?Sized>(
  cfg: &BenchConfig,
  db: &mut D,
  batch_size: usize,
) -> Result<PhaseResult> {
  let n = cfg.num_records;
  let bs = batch_size.max(1) as u64;
  let r = timed(cfg, db, "batch_writes", |db, _rng| {
    let mut key = Vec::with_capacity(cfg.key_len);
    let mut h = FNV_OFFSET;
    for i in 0..n {
      encode_key(i, cfg.key_len, &mut key);
      h = fnv1a(&key, h);
      db.put(&key, b"v")?;
      if (i + 1).is_multiple_of(bs) {
        db.commit()?;
      }
    }
    if !n.is_multiple_of(bs) {
      db.commit()?;
    }
    Ok(Counts {
      ops: n,
      seq_hash: h,
      ..Default::default()
    })
  })?;
  check_len(db, n, "batch_writes")?;
  Ok(r)
}

/// `nosync_writes`：关闭每提交 `fdatasync`（对齐 db_bench 默认 `WriteOptions.sync=false`）。
fn nosync_write<D: BenchDb + ?Sized>(cfg: &BenchConfig, db: &mut D) -> Result<PhaseResult> {
  let n = cfg.num_records;
  db.set_sync(false);
  let r = timed(cfg, db, "nosync_writes", |db, _rng| {
    let mut key = Vec::with_capacity(cfg.key_len);
    let mut h = FNV_OFFSET;
    for i in 0..n {
      encode_key(i, cfg.key_len, &mut key);
      h = fnv1a(&key, h);
      db.put(&key, b"v")?;
    }
    db.commit()?;
    Ok(Counts {
      ops: n,
      seq_hash: h,
      ..Default::default()
    })
  })?;
  check_len(db, n, "nosync_writes")?;
  Ok(r)
}

/// `len`：全表计数 + 顺序全扫，并要求扫描条数等于 `len`（前缀压缩/预取带宽）。
fn len_scan<D: BenchDb + ?Sized>(cfg: &BenchConfig, db: &mut D) -> Result<PhaseResult> {
  let len = db.len()?;
  let mut key = Vec::with_capacity(cfg.key_len);
  encode_key(0, cfg.key_len, &mut key);
  let r = timed(cfg, db, "len", |db, _rng| {
    let scanned = db.scan(&key, usize::try_from(len).unwrap_or(usize::MAX))?;
    Ok(Counts {
      ops: len,
      hits: scanned as u64,
      seq_hash: fnv1a(&key, FNV_OFFSET),
      ..Default::default()
    })
  })?;
  if r.hits != len {
    return Err(Error::Invariant(format!(
      "全表扫描条数 {} != len {len}",
      r.hits
    )));
  }
  Ok(r)
}

/// `random_reads_{1,n}`：`n` 次随机点读；`threads` 只决定命名档位（单驱动 = 单线程档）。
fn random_read<D: BenchDb + ?Sized>(
  cfg: &BenchConfig,
  db: &mut D,
  n: u64,
  threads: u32,
) -> Result<PhaseResult> {
  let name: &'static str = if threads <= 1 {
    "random_reads_1"
  } else {
    "random_reads_n"
  };
  timed(cfg, db, name, |db, rng| {
    let mut key = Vec::with_capacity(cfg.key_len);
    let mut out = Vec::new();
    let mut zipf = None;
    let (mut hits, mut misses) = (0u64, 0u64);
    let mut h = FNV_OFFSET;
    for _ in 0..n {
      let id = sample_id(rng, cfg.key_dist, cfg.num_records, &mut zipf);
      encode_key(id, cfg.key_len, &mut key);
      h = fnv1a(&key, h);
      if db.get(&key, &mut out)? {
        hits += 1;
      } else {
        misses += 1;
      }
    }
    Ok(Counts {
      ops: n,
      hits,
      misses,
      seq_hash: h,
      ..Default::default()
    })
  })
}

/// `range_reads`：`n` 个随机起点，各连续读 [`RANGE_LEN`] 条。
fn range_read<D: BenchDb + ?Sized>(cfg: &BenchConfig, db: &mut D, n: u64) -> Result<PhaseResult> {
  timed(cfg, db, "range_reads", |db, rng| {
    let mut key = Vec::with_capacity(cfg.key_len);
    let mut zipf = None;
    let (mut pairs, mut empty) = (0u64, 0u64);
    let mut h = FNV_OFFSET;
    for _ in 0..n {
      let start = sample_id(rng, cfg.key_dist, cfg.num_records, &mut zipf);
      encode_key(start, cfg.key_len, &mut key);
      h = fnv1a(&key, h);
      match db.scan(&key, RANGE_LEN)? {
        0 => empty += 1,
        got => pairs += got as u64,
      }
    }
    Ok(Counts {
      ops: n,
      hits: pairs,
      misses: empty,
      seq_hash: h,
      ..Default::default()
    })
  })
}

/// `removals`：均匀随机删除已写入键（`rng.bool()` 每次 50%）。
fn removal<D: BenchDb + ?Sized>(cfg: &BenchConfig, db: &mut D, n: u64) -> Result<PhaseResult> {
  let len_before = db.len()?;
  let r = timed(cfg, db, "removals", |db, rng| {
    let mut key = Vec::with_capacity(cfg.key_len);
    let mut h = FNV_OFFSET;
    let mut deleted = 0u64;
    for i in 0..n {
      encode_key(i, cfg.key_len, &mut key);
      h = fnv1a(&key, h);
      if rng.bool() && db.del(&key)? {
        deleted += 1;
      }
    }
    Ok(Counts {
      ops: n,
      hits: deleted,
      seq_hash: h,
      ..Default::default()
    })
  })?;
  let len_after = db.len()?;
  if len_before != len_after + r.hits {
    return Err(Error::Invariant(format!(
      "removals 后 len {len_before} -> {len_after} 与删除数 {} 不自洽",
      r.hits
    )));
  }
  Ok(r)
}

/// `large_blob_rw`：`n` 个 `blob_kib` KiB 大对象的写 + 读。
fn large_blob_rw<D: BenchDb + ?Sized>(
  cfg: &BenchConfig,
  db: &mut D,
  n: u64,
  blob_kib: usize,
) -> Result<PhaseResult> {
  let blob = vec![BLOB_FILL; blob_kib * 1024];
  let bytes = n * (blob_kib as u64) * 1024;
  timed(cfg, db, "large_blob_rw", |db, _rng| {
    let mut key = Vec::with_capacity(24);
    let mut out = Vec::new();
    let (mut hits, mut misses) = (0u64, 0u64);
    let mut h = FNV_OFFSET;
    for i in 0..n {
      key.clear();
      key.extend_from_slice(b"blob-");
      push_decimal(i, &mut key);
      h = fnv1a(&key, h);
      db.put_blob(&key, &blob)?;
      if db.get_blob(&key, &mut out)? {
        hits += 1;
      } else {
        misses += 1;
      }
    }
    Ok(Counts {
      ops: 2 * n,
      hits,
      misses,
      bytes,
      seq_hash: h,
    })
  })
}

/// 跑 8 个吞吐相位序列（用于对拍外部后端；`len` 为元数据相位，由 run_len_scan 单独采集）。
fn drive_phase_set<D: BenchDb + ?Sized>(cfg: &BenchConfig, db: &mut D) -> Result<Vec<PhaseResult>> {
  Ok(vec![
    bulk_load(cfg, db)?,
    individual_write(cfg, db)?,
    batch_write(cfg, db, DEFAULT_BATCH)?,
    nosync_write(cfg, db)?,
    random_read(cfg, db, cfg.num_records, 1)?,
    range_read(cfg, db, cfg.num_records)?,
    removal(cfg, db, cfg.num_records)?,
    large_blob_rw(cfg, db, 64, 64)?,
  ])
}

/// WAF 字段的显式输出契约：未接线时 `value=null, status="not-wired"`。
#[derive(Debug, serde::Serialize)]
struct WafOut {
  /// 实测 WAF；未接线为 `null`（不是 0）。
  value: Option<f64>,
  /// `"ok"` / `"not-wired"` / `"external"`。
  status: &'static str,
  /// 人读备注。
  note: &'static str,
}

/// 完整相位输出（含耗时，随时序抖动）。
#[derive(Debug, serde::Serialize)]
struct PhaseFull<'a> {
  phase: &'a str,
  count: u64,
  ms: u64,
  ops: f64,
  hits: u64,
  misses: u64,
  bytes: u64,
  seq_hash: u64,
}

/// 稳定相位输出（仅相位名 / 逻辑操作数 / 字节；**无耗时、无结果计数**）。
///
/// 结果计数（hits/misses）与操作序列哈希（seq_hash）随相位语义而变，作为确定性/语义
/// 证据由 [`BenchRunner::results`] 与 [`BenchRunner::report_json`] 暴露；本结构只保留
/// 与后端实现无关、可逐字节 golden 比对的结构事实。
#[derive(Debug, serde::Serialize)]
struct PhaseStatic<'a> {
  phase: &'a str,
  count: u64,
  bytes: u64,
}

#[derive(Debug, serde::Serialize)]
struct ReportFull<'a> {
  schema: &'static str,
  phases: Vec<PhaseFull<'a>>,
  waf: WafOut,
}

#[derive(Debug, serde::Serialize)]
struct ReportStatic<'a> {
  schema: &'static str,
  phases: Vec<PhaseStatic<'a>>,
}

#[derive(Debug, serde::Serialize)]
struct CompareOut<'a> {
  schema: &'static str,
  ldb: ReportFull<'a>,
  rocksdb: ReportFull<'a>,
  redb: ReportFull<'a>,
}

/// 基准驱动。
pub struct BenchRunner<D: BenchDb> {
  /// 配置。
  pub cfg: BenchConfig,
  /// 被测后端。
  pub db: D,
  /// 统计（后端经 [`BenchDb::set_stat`] 注入同一句柄后，WAF 可取真实值）。
  pub stat: Arc<StatCollector>,
  results: Vec<PhaseResult>,
}

impl<D: BenchDb> BenchRunner<D> {
  /// 构造（配置非法即 [`Error::InvalidConfig`]；同时把统计句柄注入后端）。
  pub fn new(cfg: BenchConfig, mut db: D) -> Result<Self> {
    cfg.validate()?;
    let stat = Arc::new(StatCollector::default());
    db.set_stat(Arc::clone(&stat));
    Ok(Self {
      cfg,
      db,
      stat,
      results: Vec::new(),
    })
  }

  /// 已完成的相位结果。
  pub fn results(&self) -> &[PhaseResult] {
    &self.results
  }

  fn push(&mut self, r: PhaseResult) -> PhaseResult {
    self.results.push(r);
    r
  }

  /// 批量装载（todo.md §5 `bulk_load`）。
  pub async fn run_bulk_load(&mut self) -> Result<PhaseResult> {
    let r = bulk_load(&self.cfg, &mut self.db)?;
    Ok(self.push(r))
  }

  /// 单条独立写（todo.md §5 `individual_writes`）。
  pub async fn run_individual_write(&mut self) -> Result<PhaseResult> {
    let r = individual_write(&self.cfg, &mut self.db)?;
    Ok(self.push(r))
  }

  /// 批量写（todo.md §5 `batch_writes`；`batch_size` 条/事务）。
  pub async fn run_batch_write(&mut self, batch_size: usize) -> Result<PhaseResult> {
    let r = batch_write(&self.cfg, &mut self.db, batch_size)?;
    Ok(self.push(r))
  }

  /// 无 sync 写（todo.md §5 `nosync_writes`）。
  pub async fn run_nosync_write(&mut self) -> Result<PhaseResult> {
    let r = nosync_write(&self.cfg, &mut self.db)?;
    Ok(self.push(r))
  }

  /// 全表计数 / 全扫（todo.md §5 `len`）。
  pub async fn run_len_scan(&mut self) -> Result<PhaseResult> {
    let r = len_scan(&self.cfg, &mut self.db)?;
    Ok(self.push(r))
  }

  /// 随机点读（todo.md §5 `random_reads_{1..32}`；`threads` 决定命名档位）。
  pub async fn run_random_read(&mut self, n: u64, threads: u32) -> Result<PhaseResult> {
    let r = random_read(&self.cfg, &mut self.db, n, threads)?;
    Ok(self.push(r))
  }

  /// 区间读（todo.md §5 `range_reads`）。
  pub async fn run_range_read(&mut self, n: u64) -> Result<PhaseResult> {
    let r = range_read(&self.cfg, &mut self.db, n)?;
    Ok(self.push(r))
  }

  /// 随机删 ~50%（todo.md §5 `removals`）。
  pub async fn run_removal(&mut self, n: u64) -> Result<PhaseResult> {
    let r = removal(&self.cfg, &mut self.db, n)?;
    Ok(self.push(r))
  }

  /// 大对象读写（todo.md §5 `large_blob_rw`）。
  pub async fn run_large_blob_rw(&mut self, n: u64, blob_kib: usize) -> Result<PhaseResult> {
    let r = large_blob_rw(&self.cfg, &mut self.db, n, blob_kib)?;
    Ok(self.push(r))
  }

  /// WAF 输出（未接线显式 `not-wired`，不伪造 0）。
  fn waf(&self) -> WafOut {
    if self.db.stat_wired() {
      WafOut {
        value: Some(self.stat.calc_db_waf()),
        status: "ok",
        note: "",
      }
    } else {
      WafOut {
        value: None,
        status: "not-wired",
        note: "ldb_stat record_* 无生产接线（CROSS STAT-R3-4）；0.0 不是真实 WAF",
      }
    }
  }

  /// 结果 JSON（合法；含耗时字段）。
  pub fn report_json(&self) -> Result<String> {
    let out = ReportFull {
      schema: SCHEMA,
      phases: self.results.iter().map(phase_full).collect(),
      waf: self.waf(),
    };
    sonic_rs::to_string(&out).map_err(|e| Error::Backend(e.into()))
  }

  /// 稳定结果 JSON（仅 schema + 相位结构字段，无耗时/结果/WAF；golden 逐字节比对）。
  pub fn report_static_json(&self) -> Result<String> {
    let out = ReportStatic {
      schema: SCHEMA,
      phases: self.results.iter().map(phase_static).collect(),
    };
    sonic_rs::to_string(&out).map_err(|e| Error::Backend(e.into()))
  }

  /// 人读表格。
  pub fn report_table(&self) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
      s,
      "{:<16} {:>10} {:>10} {:>16} {:>14} {:>18}",
      "phase", "count", "ms", "ops/s", "bytes", "seq_hash"
    );
    for r in &self.results {
      let _ = writeln!(
        s,
        "{:<16} {:>10} {:>10} {:>16.0} {:>14} {:>18x}",
        r.phase, r.count, r.ms, r.ops, r.bytes, r.seq_hash
      );
    }
    s
  }

  /// 三方对比：对 `rocksdb` / `redb` 各跑同一 9 相位序列后合并为单个合法 JSON。
  ///
  /// 外部后端的 WAF 由各自引擎采集，故此处 `waf.status="external"`。
  /// ldb 的相位由调用方先行 `run_*`（本方法不重复跑 ldb）。
  pub async fn compare_rocksdb_redb(
    &mut self,
    rocks: &mut dyn BenchDb,
    redb: &mut dyn BenchDb,
  ) -> Result<String> {
    let rocks_phases = drive_phase_set(&self.cfg, rocks)?;
    let redb_phases = drive_phase_set(&self.cfg, redb)?;
    let out = CompareOut {
      schema: SCHEMA,
      ldb: ReportFull {
        schema: SCHEMA,
        phases: self.results.iter().map(phase_full).collect(),
        waf: self.waf(),
      },
      rocksdb: peer_report(&rocks_phases),
      redb: peer_report(&redb_phases),
    };
    sonic_rs::to_string(&out).map_err(|e| Error::Backend(e.into()))
  }
}

fn phase_full(r: &PhaseResult) -> PhaseFull<'_> {
  PhaseFull {
    phase: r.phase,
    count: r.count,
    ms: r.ms,
    ops: r.ops,
    hits: r.hits,
    misses: r.misses,
    bytes: r.bytes,
    seq_hash: r.seq_hash,
  }
}

fn phase_static(r: &PhaseResult) -> PhaseStatic<'_> {
  PhaseStatic {
    phase: r.phase,
    count: r.count,
    bytes: r.bytes,
  }
}

fn peer_report(phases: &[PhaseResult]) -> ReportFull<'_> {
  ReportFull {
    schema: SCHEMA,
    phases: phases.iter().map(phase_full).collect(),
    waf: WafOut {
      value: None,
      status: "external",
      note: "外部引擎 WAF 由 db_bench/引擎自身采集",
    },
  }
}

impl<D: BenchDb> fmt::Debug for BenchRunner<D> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BenchRunner")
      .field("db", &self.db.name())
      .field("phases", &self.results.len())
      .finish()
  }
}
