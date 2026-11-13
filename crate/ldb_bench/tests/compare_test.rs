//! ldb_bench 回归：JSON golden / 参数映射 / 操作序列确定性 / 相位语义 / WAF 契约。

use std::{collections::BTreeMap, str, sync::Arc};

use ldb_bench::{
  BenchConfig, BenchDb, BenchRunner, Compression, Error, KeyDist, Result, map_phase,
  parse_db_bench_report,
};
use ldb_stat::StatCollector;
use serde::de::IgnoredAny;

/// 一页字节（WAF 分母口径演示用）。
const PAGE: u64 = 4096;
/// 「热点」阈值：id < 100 视为热点键（zipf 集中度观察）。
const HOT: u64 = 100;

/// 内存 KV 后端；记录提交次数、键形态与统计接线状态。
#[derive(Debug, Default)]
struct MemDb {
  map: BTreeMap<Vec<u8>, Vec<u8>>,
  commits: u64,
  first_key: Option<Vec<u8>>,
  last_put_key: Option<Vec<u8>>,
  get_lt_hot: u64,
  get_total: u64,
  /// 点读键 id 的直方图（zipf 形态与理想 Zipf 的 TV 距离量化用）。
  get_hist: BTreeMap<u64, u64>,
  /// 是否声明「支持统计接线」（true 时 `set_stat` 才保存句柄）。
  wire: bool,
  stat: Option<Arc<StatCollector>>,
}

impl MemDb {
  fn wired() -> Self {
    Self {
      wire: true,
      ..Default::default()
    }
  }
}

impl BenchDb for MemDb {
  fn name(&self) -> &'static str {
    "mem"
  }
  fn put(&mut self, k: &[u8], v: &[u8]) -> Result<()> {
    if self.first_key.is_none() {
      self.first_key = Some(k.to_vec());
    }
    self.last_put_key = Some(k.to_vec());
    if let Some(s) = &self.stat {
      // 逻辑写 = 值字节；物理写按一页演示（对齐 ldb_stat 的页/字节双记口径）。
      s.record_logical_write(v.len() as u64);
      s.record_page_evict(PAGE);
    }
    self.map.insert(k.to_vec(), v.to_vec());
    Ok(())
  }
  fn get(&mut self, k: &[u8], out: &mut Vec<u8>) -> Result<bool> {
    // 只提取数字位还原 id：使填充字符（非数字）的变化不影响热点统计。
    let digits: Vec<u8> = k.iter().copied().filter(u8::is_ascii_digit).collect();
    if let Ok(id) = str::from_utf8(&digits).unwrap_or("").parse::<u64>() {
      self.get_total += 1;
      if id < HOT {
        self.get_lt_hot += 1;
      }
      *self.get_hist.entry(id).or_insert(0) += 1;
    }
    match self.map.get(k) {
      Some(v) => {
        out.clear();
        out.extend_from_slice(v);
        Ok(true)
      }
      None => Ok(false),
    }
  }
  fn del(&mut self, k: &[u8]) -> Result<bool> {
    Ok(self.map.remove(k).is_some())
  }
  fn len(&mut self) -> Result<u64> {
    Ok(self.map.len() as u64)
  }
  fn scan(&mut self, lo: &[u8], cnt: usize) -> Result<usize> {
    Ok(self.map.range(lo.to_vec()..).take(cnt).count())
  }
  fn set_sync(&mut self, _on: bool) {}
  fn commit(&mut self) -> Result<()> {
    self.commits += 1;
    Ok(())
  }
  fn stat_wired(&self) -> bool {
    self.wire
  }
  fn set_stat(&mut self, stat: Arc<StatCollector>) {
    if self.wire {
      self.stat = Some(stat);
    }
  }
}

/// 基准调用失败即测试失败（不做强制解包）。
fn must<T>(r: Result<T>) -> T {
  r.unwrap_or_else(|e| panic!("基准返回错误: {e}"))
}

/// golden 用的固定配置（小数、确定性）。
fn golden_cfg() -> BenchConfig {
  BenchConfig {
    num_records: 4,
    key_len: 4,
    val_len: 3,
    key_dist: KeyDist::Uniform,
    read_random_exp_range: 0.0,
    seed: 42,
    sync: true,
    cache_size: 1 << 20,
    write_buffer_size: 1 << 20,
    use_existing_db: false,
    compression: Compression::None,
  }
}

async fn run_golden() -> BenchRunner<MemDb> {
  let mut r = must(BenchRunner::new(golden_cfg(), MemDb::default()));
  must(r.run_bulk_load().await);
  must(r.run_random_read(4, 1).await);
  must(r.run_range_read(2).await);
  must(r.run_removal(4).await);
  must(r.run_large_blob_rw(2, 1).await);
  r
}

/// 各相位操作序列哈希（确定性证据：两次运行必须逐位一致）。
fn seq_hashes(r: &BenchRunner<MemDb>) -> Vec<u64> {
  r.results().iter().map(|p| p.seq_hash).collect()
}

#[compio::test]
async fn static_json_matches_golden_literal() {
  let r = run_golden().await;
  let json = must(r.report_static_json());
  assert_eq!(json, GOLDEN_STATIC, "\nactual={json}\n");
}

/// 稳定 JSON 的 golden（无耗时/结果字段，逐字节固定）。
const GOLDEN_STATIC: &str = "{\"schema\":\"ldb_bench.v1\",\"phases\":[{\"phase\":\"bulk_load\",\"count\":4,\"bytes\":28},{\"phase\":\"random_reads_1\",\"count\":4,\"bytes\":0},{\"phase\":\"range_reads\",\"count\":2,\"bytes\":0},{\"phase\":\"removals\",\"count\":4,\"bytes\":0},{\"phase\":\"large_blob_rw\",\"count\":4,\"bytes\":2048}]}";

#[compio::test]
async fn report_json_is_valid_json_and_has_schema() {
  let r = run_golden().await;
  let json = must(r.report_json());
  // 语法有效（拒绝早期 `\"` 手工拼接的非法输出）。
  assert!(
    sonic_rs::from_str::<IgnoredAny>(&json).is_ok(),
    "report_json 非合法 JSON: {json}"
  );
  assert!(json.contains("\"schema\":\"ldb_bench.v1\""), "{json}");
  assert!(
    json.contains("\"phases\":["),
    "字段名必须是 phases（非 phrases）: {json}"
  );
  assert!(!json.contains("phrases"), "{json}");
  assert!(!json.contains("\\\""), "不得含手工转义的反斜杠引号: {json}");
  assert!(r.report_table().contains("phase"), "人读表格须有表头");
}
#[compio::test]
async fn phase_sequence_is_deterministic_across_runs() {
  let a = run_golden().await;
  let b = run_golden().await;
  assert_eq!(
    seq_hashes(&a),
    seq_hashes(&b),
    "同种子两次运行的操作序列哈希必须逐位一致"
  );
  assert_eq!(
    must(a.report_static_json()),
    must(b.report_static_json()),
    "稳定字段必须逐字节一致"
  );
}

#[compio::test]
async fn different_seed_changes_operation_sequence() {
  let mut ca = golden_cfg();
  ca.seed = 1;
  let mut cb = golden_cfg();
  cb.seed = 2;
  let mut ra = must(BenchRunner::new(ca, MemDb::default()));
  let mut rb = must(BenchRunner::new(cb, MemDb::default()));
  must(ra.run_bulk_load().await);
  must(rb.run_bulk_load().await);
  must(ra.run_random_read(8, 1).await);
  must(rb.run_random_read(8, 1).await);
  assert_ne!(
    seq_hashes(&ra),
    seq_hashes(&rb),
    "不同种子应产生不同操作序列（seq_hash 不同）"
  );
}

#[compio::test]
async fn bulk_load_key_encoding_and_len() {
  let mut r = must(BenchRunner::new(golden_cfg(), MemDb::default()));
  must(r.run_bulk_load().await);
  assert_eq!(must(r.db.len()), 4);
  assert_eq!(
    r.db.first_key.as_deref(),
    Some(&b"0000"[..]),
    "首位键须左补零"
  );
  assert_eq!(
    r.db.last_put_key.as_deref(),
    Some(&b"0003"[..]),
    "末位键编码"
  );
}

#[compio::test]
async fn random_read_hit_rate_is_exact_after_load() {
  let mut r = must(BenchRunner::new(golden_cfg(), MemDb::default()));
  must(r.run_bulk_load().await);
  let rr = must(r.run_random_read(4, 1).await);
  assert_eq!(rr.count, 4);
  assert_eq!(rr.hits, 4, "键域 [0,4) 已全量装载，随机读必须 100% 命中");
  assert_eq!(rr.misses, 0);
}

#[compio::test]
async fn commit_semantics_distinguish_individual_and_batch() {
  let mut cfg = golden_cfg();
  cfg.num_records = 10;
  let mut ri = must(BenchRunner::new(cfg, MemDb::default()));
  must(ri.run_bulk_load().await);
  ri.db.commits = 0;
  must(ri.run_individual_write().await);
  assert_eq!(ri.db.commits, 10, "单条写：每记录一次提交");

  let mut rb = must(BenchRunner::new(cfg, MemDb::default()));
  must(rb.run_bulk_load().await);
  rb.db.commits = 0;
  must(rb.run_batch_write(4).await);
  assert_eq!(rb.db.commits, 3, "批量写 10 条 / 批 4 => 3 次提交（4+4+2）");

  let mut rn = must(BenchRunner::new(cfg, MemDb::default()));
  must(rn.run_bulk_load().await);
  rn.db.commits = 0;
  must(rn.run_nosync_write().await);
  assert_eq!(rn.db.commits, 1, "nosync 写：循环结束后单次提交");
}

#[compio::test]
async fn removal_keeps_len_consistent() {
  let mut r = must(BenchRunner::new(golden_cfg(), MemDb::default()));
  must(r.run_bulk_load().await);
  let before = must(r.db.len());
  let rm = must(r.run_removal(4).await);
  let after = must(r.db.len());
  assert_eq!(before, after + rm.hits, "len 下降必须等于删除成功数");
  assert_eq!(rm.hits, 3, "种子 42 下 4 键删 3（约一半，确定性）");
}

#[compio::test]
async fn len_scan_counts_all_records() {
  let mut r = must(BenchRunner::new(golden_cfg(), MemDb::default()));
  must(r.run_bulk_load().await);
  let ls = must(r.run_len_scan().await);
  assert_eq!(ls.count, 4);
  assert_eq!(ls.hits, 4, "全表扫描条数必须等于 len");
}

#[compio::test]
async fn zipfian_concentrates_hot_keys_uniform_does_not() {
  let mut cz = golden_cfg();
  cz.num_records = 1000;
  cz.key_len = 4;
  cz.key_dist = KeyDist::Zipfian { theta: 1.0 };
  let mut rz = must(BenchRunner::new(cz, MemDb::default()));
  must(rz.run_bulk_load().await);
  must(rz.run_random_read(500, 1).await);
  let zhot = rz.db.get_lt_hot as f64 / rz.db.get_total as f64;

  let mut cu = golden_cfg();
  cu.num_records = 1000;
  cu.key_len = 4;
  let mut ru = must(BenchRunner::new(cu, MemDb::default()));
  must(ru.run_bulk_load().await);
  must(ru.run_random_read(500, 1).await);
  let uhot = ru.db.get_lt_hot as f64 / ru.db.get_total as f64;

  assert!(zhot > 0.5, "zipf θ=1 热点集中度应 > 0.5，实测 {zhot}");
  assert!(uhot < 0.3, "uniform 热点占比应 ≈0.1，实测 {uhot}");
}

/// 理想 Zipf(θ) 在 `[0,n)` 上的 PMF：`p_i ∝ (i+1)^-θ`（与 ldb_zipf 的 1-based 值域一致）。
fn ideal_zipf_pmf(n: u64, theta: f64) -> Vec<f64> {
  let w: Vec<f64> = (1..=n).map(|i| (i as f64).powf(-theta)).collect();
  let z: f64 = w.iter().sum();
  w.into_iter().map(|x| x / z).collect()
}

/// 经验直方图与给定 PMF 的全变差距离 `TV = ½·Σ|emp - p|`。
fn tv_distance(hist: &BTreeMap<u64, u64>, total: u64, pmf: &[f64]) -> f64 {
  let t = total as f64;
  let acc: f64 = pmf
    .iter()
    .enumerate()
    .map(|(i, p)| {
      let emp = hist.get(&(i as u64)).copied().unwrap_or(0) as f64 / t;
      (emp - p).abs()
    })
    .sum();
  0.5 * acc
}

/// 量化 zipf 采样与**理想 Zipf 分布**的形态差异（同 θ 下 TV 距离）。
///
/// 说明：RocksDB `db_bench`（11.8.1）**没有** `--key_distribution`/`--zipf_const`
/// （实机核对，见报告 §2）；倾斜读只有 `--read_random_exp_range`（质量律 ∝1/rank、
/// 截断于 `num·e^{-R}`，仅 θ=1 精确），非任意 zipf(θ)，故不存在「RocksDB zipfian」
/// 可逐位对拍；上游 `ZLeanStore/benchmark` 的 `ZipfGenerator` 即精确 Zipf（累积概率）
/// 分布，故以理想 Zipf(θ) 律为 oracle。θ=1 时 ldb_zipf 走精确拒绝-反转路径。
#[compio::test]
async fn zipfian_matches_ideal_distribution_within_tv() {
  let n = 1000u64;
  let theta = 1.0;
  let samples = 100_000u64;
  let mut cfg = golden_cfg();
  cfg.num_records = n;
  cfg.key_len = 4;
  cfg.key_dist = KeyDist::Zipfian { theta };
  let mut r = must(BenchRunner::new(cfg, MemDb::default()));
  must(r.run_bulk_load().await);
  must(r.run_random_read(samples, 1).await);
  assert_eq!(r.db.get_total, samples, "采样数应等于请求数");
  let pmf = ideal_zipf_pmf(n, theta);
  let tv = tv_distance(&r.db.get_hist, r.db.get_total, &pmf);
  // 采样噪声下 TV 应远小于 zipf-vs-uniform 的距离（θ=1,n=1000 时约 0.6）。
  assert!(tv < 0.1, "zipf(θ=1,n=1000) 与理想分布 TV={tv} 应 < 0.1");
}

#[compio::test]
async fn waf_is_explicitly_not_wired_by_default() {
  let r = run_golden().await;
  let json = must(r.report_json());
  assert!(
    json.contains("\"value\":null,\"status\":\"not-wired\""),
    "未接线 WAF 必须显式 null + not-wired（不得静默给 0）: {json}"
  );
  assert!(!json.contains("\"value\":0.0"), "{json}");
}

#[compio::test]
async fn waf_reports_value_when_wired() {
  let mut cfg = golden_cfg();
  cfg.num_records = 8;
  let mut r = must(BenchRunner::new(cfg, MemDb::wired()));
  must(r.run_bulk_load().await);
  let json = must(r.report_json());
  assert!(json.contains("\"status\":\"ok\""), "{json}");
  // 8 条 × 一页物理写 / 8 条 × 3 字节逻辑写 = 4096/3
  assert!(json.contains("\"value\":1365."), "WAF 应取真实值: {json}");
}

#[compio::test]
async fn compare_report_covers_three_backends_as_valid_json() {
  let mut ldb = run_golden().await;
  let mut rocks = MemDb::default();
  let mut redb = MemDb::default();
  let json = must(ldb.compare_rocksdb_redb(&mut rocks, &mut redb).await);
  assert!(
    sonic_rs::from_str::<IgnoredAny>(&json).is_ok(),
    "对比报告必须是合法 JSON: {json}"
  );
  assert!(json.contains("\"ldb\":"), "{json}");
  assert!(json.contains("\"rocksdb\":"), "{json}");
  assert!(json.contains("\"redb\":"), "{json}");
  assert!(json.contains("\"status\":\"external\""), "{json}");
}

/// 配置 → db_bench 命令行：只含 **真实存在** 的 flag（无 `--key_distribution`/
/// `--zipf_const`/`--random_seed`；种子用 `--seed`），逐字对齐实机 `db_bench --help`。
#[test]
fn db_bench_args_map_config_one_to_one() {
  let cfg = BenchConfig {
    num_records: 1_000_000,
    key_len: 16,
    val_len: 100,
    key_dist: KeyDist::Uniform,
    read_random_exp_range: 13.0,
    seed: 7,
    sync: true,
    cache_size: 8 << 20,
    write_buffer_size: 64 << 20,
    use_existing_db: true,
    compression: Compression::Zstd,
  };
  let args = must(cfg.db_bench_args(&["fillrandom", "readrandom"]));
  assert_eq!(
    args,
    vec![
      "--num=1000000",
      "--key_size=16",
      "--value_size=100",
      "--read_random_exp_range=13",
      "--seed=7",
      "--sync=1",
      "--cache_size=8388608",
      "--write_buffer_size=67108864",
      "--use_existing_db=1",
      "--compression_type=zstd",
      "--benchmarks=fillrandom,readrandom",
    ]
  );
  assert_eq!(map_phase("bulk_load"), Some("fillrandom"));
  assert_eq!(map_phase("random_reads_8"), Some("readrandom"));
  assert_eq!(map_phase("range_reads"), Some("seekrandom"));
  assert_eq!(map_phase("removals"), Some("deleterandom"));
  assert_eq!(map_phase("nonexistent"), None);
}

/// 不可表达即 Err：db_bench 无 key 分布 flag（zipfian 无法等价），空负载同理。
#[test]
fn db_bench_args_rejects_unrepresentable() {
  let cfg = BenchConfig {
    key_dist: KeyDist::Zipfian { theta: 0.8 },
    ..BenchConfig::default()
  };
  match cfg.db_bench_args(&["readrandom"]) {
    Err(Error::DbBench(m)) => {
      assert!(m.contains("zipfian"), "{m}");
      assert!(m.contains("无 key 分布 flag"), "{m}");
      assert!(
        m.contains("--read_random_exp_range"),
        "错误信息须指向实机倾斜旋钮: {m}"
      );
    }
    other => panic!("zipfian 必须被拒绝（db_bench 无该 flag），实得 {other:?}"),
  }
  let uni = BenchConfig {
    key_dist: KeyDist::Uniform,
    ..BenchConfig::default()
  };
  assert!(
    matches!(uni.db_bench_args(&[]), Err(Error::DbBench(_))),
    "空 benchmarks 必须 Err"
  );
}

/// 解析真实 db_bench stdout（本机 `rocksdb-11.8.1/db_bench` 实跑逐字捕获）：
/// 多 benchmark 行、含 `(N of M found)` 后缀。
#[test]
fn parse_db_bench_stdout_report() {
  let stdout = "\
DB path: [/tmp/ldbbench_r2/db]
fillrandom   :       4.201 micros/op 237783 ops/sec 0.008 seconds 2000 operations;   26.3 MB/s
readrandom   :       0.690 micros/op 1444043 ops/sec 0.001 seconds 2000 operations;  159.7 MB/s (2000 of 2000 found)
overwrite    :       4.013 micros/op 249097 ops/sec 0.008 seconds 2000 operations;   27.6 MB/s
";
  let phases = must(parse_db_bench_report(stdout));
  assert_eq!(phases.len(), 3, "{phases:?}");
  assert_eq!(phases[0].phase, "fillrandom");
  assert_eq!(phases[0].micros_per_op, 4.201);
  assert_eq!(phases[0].ops_per_sec, 237783.0);
  assert_eq!(phases[1].phase, "readrandom");
  assert_eq!(phases[1].ops_per_sec, 1444043.0);
  assert_eq!(phases[2].phase, "overwrite");
  assert_eq!(phases[2].micros_per_op, 4.013);
}

/// 噪声行（配置 dump / 速率行 / GC-compaction 统计 / 进度）被跳过，只取 benchmark 行。
#[test]
fn parse_db_bench_skips_noise_lines() {
  let stdout = "\
Initializing RocksDB Options from command-line flags
Keys:       16 bytes each (+ 0 bytes user-defined timestamp)
Read rate: 0 ops/second
rocksdb.compaction.times.micros COUNT : 0
... finished 2000 ops
fillrandom   :       4.201 micros/op 237783 ops/sec 0.008 seconds 2000 operations;   26.3 MB/s
";
  let phases = must(parse_db_bench_report(stdout));
  assert_eq!(phases.len(), 1, "{phases:?}");
  assert_eq!(phases[0].phase, "fillrandom");
}

/// 真实 `--histogram=1` 输出（本机 `rocksdb-11.8.1/db_bench` 逐字捕获）：histogram
/// 段（`Count:`/`Min:`/`Percentiles:`/桶行）与 `compact` 行都被正确区分——只取
/// benchmark 结果行（含 `micros/op`），histogram 噪声行自然跳过。
#[test]
fn parse_db_bench_skips_histogram_and_reads_compact() {
  let stdout = "\
DB path: [/tmp/x/db2]
fillrandom   :       3.670 micros/op 272458 ops/sec 0.184 seconds 50000 operations;   30.1 MB/s
Microseconds per write:
Count: 50000 Average: 3.6702  StdDev: 9.09
Min: 2  Median: 2.7311  Max: 2014
Percentiles: P50: 2.73 P75: 3.46 P99: 9.48 P99.9: 13.17 P99.99: 21.22
------------------------------------------------------
(       1,       2 ]     2925   5.850%   5.850% #
(       2,       3 ]    30195  60.390%  66.240% ############
(    1900,    2900 ]        1   0.002% 100.000% 

DB path: [/tmp/x/db2]
compact      :   59342.000 micros/op 16 ops/sec 0.059 seconds 1 operations;
";
  let phases = must(parse_db_bench_report(stdout));
  assert_eq!(phases.len(), 2, "{phases:?}");
  assert_eq!(phases[0].phase, "fillrandom");
  assert_eq!(phases[0].micros_per_op, 3.670);
  assert_eq!(phases[0].ops_per_sec, 272458.0);
  assert_eq!(phases[1].phase, "compact");
  assert_eq!(phases[1].micros_per_op, 59342.0);
}

/// 解析失败必须 Err（不得静默 0）：无 benchmark 行、或含 `micros/op` 却缺 `ops/sec`。
#[test]
fn parse_db_bench_errors_instead_of_silent_zero() {
  match parse_db_bench_report("DB path: [/tmp/x]\nnothing here\n") {
    Err(Error::DbBench(m)) => assert!(m.contains("未发现"), "{m}"),
    other => panic!("空结果必须 Err，实得 {other:?}"),
  }
  match parse_db_bench_report("fillrandom : 4.2 micros/op 0.008 seconds\n") {
    Err(Error::DbBench(m)) => assert!(m.contains("无法解析"), "{m}"),
    other => panic!("畸形行必须 Err，实得 {other:?}"),
  }
}
