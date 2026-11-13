//! 外部 RocksDB `db_bench` 驱动 —— 跨引擎「同参数跑同一 phase 序列」的落地路径。
//!
//! ldb 的持久化当前仅到阶段 1，无法在进程内与 RocksDB 端到端对拍；本模块提供
//! **可执行的替代路径**：把 [`BenchConfig`] 映射成 `db_bench` 命令行，注入同一份
//! 参数（键长/值长/种子/sync/缓存/写缓冲/复用库/压缩），再解析其 stdout。
//! 本机未编译 `db_bench` 时 [`DbBench::locate`] 返回 `Err`（含可读指引），
//! **不伪造任何数据**。
//!
//! # 参数映射（本仓字段 ↔ db_bench flag）
//!
//! flag 名与语义已对 `rocksdb-11.8.1/tools/db_bench_tool.cc` 与实机 `db_bench --help`
//! 逐条核对（报告 §2）。**RocksDB `db_bench` 没有 `--key_distribution`/`--zipf_const`**
//! （实测 `db_bench --key_distribution=x` 报 `ERROR: unknown command line flag`）；
//! 倾斜读的真实旋钮是 `--read_random_exp_range`（仅 `readrandom`/`multireadrandom`，
//! `db_bench_tool.cc:474`）：键 = `num·exp(-r)`（`r ~ U[0,R)`）再经置换，源码
//! `GetRandomKey`（`:7314`）使其质量律**恒为 ∝1/rank、截断于 `num·e^{-R}`**。故
//! `R ≥ ln(num)` 时与 **scrambled Zipf(θ=1)** 完全一致；YCSB 默认 θ≈0.99 时形状
//! TV≈0.01–0.02（近似）；θ≪1 无精确表达 ⇒ `KeyDist::Zipfian` 仍显式返回 `Err`
//! （不静默近似，见报告 §2「倾斜读对齐」）。另 `--mixgraph --key_dist_a/--key_dist_b`
//! 提供 `f(x)=a·x^b` 可调幂律，但属 Get/Put/Iterator 混合负载，非 YCSB 点读。
//!
//! | ldb_bench | db_bench | 语义 |
//! | --- | --- | --- |
//! | `num_records` | `--num` | 键/值条数 |
//! | `key_len` | `--key_size`（默认 16） | 键字节宽度（内容编码不同：db_bench 用二进制，本 crate 用定长十进制） |
//! | `val_len` | `--value_size`（默认 100） | 值字节 |
//! | `key_dist=Uniform` | `--read_random_exp_range=0` | 均匀键（db_bench 默认） |
//! | `read_random_exp_range=R` | `--read_random_exp_range=R` | 倾斜读；`R≥ln(num)` ⇒ scrambled Zipf(θ=1) |
//! | `key_dist=Zipfian` | **不可精确表达** → Err | db_bench 无 zipf(θ) flag；用上方 `R` 近似或自建驱动 |
//! | `seed` | `--seed`（默认 0 = 以时间派生） | RNG 种子基 |
//! | `sync` | `--sync=0/1` | 每写 fdatasync |
//! | `cache_size` | `--cache_size`（默认 32 MiB） | 块缓存字节 |
//! | `write_buffer_size` | `--write_buffer_size`（默认 64 MiB） | memtable 字节 |
//! | `use_existing_db` | `--use_existing_db=0/1` | 复用库 |
//! | `compression` | `--compression_type=none/lz4/zstd` | 压缩 |
//!
//! # phase 映射（ldb_bench phase ↔ db_bench `--benchmarks`）
//!
//! | ldb_bench phase | db_bench benchmark | 备注 |
//! | --- | --- | --- |
//! | `bulk_load` | `fillrandom` | |
//! | `individual_writes` / `batch_writes` / `nosync_writes` | `overwrite` | 联合 `--batch_size` / `--sync` 区分 |
//! | `len` | `readseq` | 近似全扫 |
//! | `random_reads_*` | `readrandom` | |
//! | `range_reads` | `seekrandom` | 需另配 `--seek_nexts=100`，本函数不自动补 |
//! | `removals` | `deleterandom` | |
//! | `large_blob_rw` | `fillrandom` | 需另配大 `--value_size`，本函数不自动补 |

use std::{
  env,
  io::{self, Read},
  path::{Path, PathBuf},
  process::{Command, Stdio},
  thread,
  time::{Duration, Instant},
};

use crate::{
  compare::{BenchConfig, Compression, KeyDist},
  error::{Error, Result},
};

/// 单次 db_bench 运行的默认超时；可用 `$DB_BENCH_TIMEOUT_SECS`（秒）覆盖。
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
/// stdout / stderr 单流最多保留的字节数；超出部分继续排空但不保留，防内存无界。
const MAX_CAPTURE: usize = 8 << 20;
/// 等待子进程退出的轮询间隔。
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// db_bench 单条 benchmark 的解析结果。
#[derive(Debug, Clone, PartialEq)]
pub struct PeerPhase {
  /// benchmark 名（如 `fillrandom`）。
  pub phase: String,
  /// 每次操作微秒数。
  pub micros_per_op: f64,
  /// 每秒操作数。
  pub ops_per_sec: f64,
}

/// ldb_bench phase → db_bench `--benchmarks` 名（无直接对应返回 `None`）。
pub fn map_phase(phase: &str) -> Option<&'static str> {
  Some(match phase {
    "bulk_load" => "fillrandom",
    "individual_writes" => "overwrite",
    "batch_writes" => "overwrite",
    "nosync_writes" => "overwrite",
    "len" => "readseq",
    "range_reads" => "seekrandom",
    "removals" => "deleterandom",
    "large_blob_rw" => "fillrandom",
    p if p.starts_with("random_reads") => "readrandom",
    _ => return None,
  })
}

impl BenchConfig {
  /// 生成 `db_bench` 命令行参数（不含可执行文件名）。
  ///
  /// 只产出 db_bench 真实存在的 flag（名字与语义见模块文档表）；`benchmarks`
  /// 以逗号连接传给 `--benchmarks`。**不可精确表达即 Err**：
  /// - `key_dist=Zipfian`：db_bench 无 `--key_distribution`/`--zipf_const`，zipf(θ)
  ///   无法精确表达（倾斜读旋钮是 `--read_random_exp_range`，见字段文档）；
  /// - `benchmarks` 为空：没有要跑的负载。
  pub fn db_bench_args(&self, benchmarks: &[&str]) -> Result<Vec<String>> {
    if benchmarks.is_empty() {
      return Err(Error::DbBench("benchmarks 为空，无可运行负载".into()));
    }
    if let KeyDist::Zipfian { theta } = self.key_dist {
      return Err(Error::DbBench(format!(
        "db_bench(11.8.1) 无 key 分布 flag（--key_distribution/--zipf_const 不存在），\
         无法精确表达 zipfian(theta={theta})；倾斜读的 db_bench 旋钮是 \
         --read_random_exp_range（质量律 ∝1/rank，R≥ln(num) 时 = scrambled Zipf(θ=1)）。\
         请用 key_dist=Uniform + read_random_exp_range 近似，或用自建 zipf 键序列驱动"
      )));
    }
    let mut a = Vec::with_capacity(16);
    a.push(format!("--num={}", self.num_records));
    a.push(format!("--key_size={}", self.key_len));
    a.push(format!("--value_size={}", self.val_len));
    // 倾斜读旋钮（Uniform 时 = 0，与 db_bench 默认一致；不再发不存在的分布 flag）。
    a.push(format!(
      "--read_random_exp_range={}",
      self.read_random_exp_range
    ));
    a.push(format!("--seed={}", self.seed));
    a.push(format!("--sync={}", u8::from(self.sync)));
    a.push(format!("--cache_size={}", self.cache_size));
    a.push(format!("--write_buffer_size={}", self.write_buffer_size));
    a.push(format!(
      "--use_existing_db={}",
      u8::from(self.use_existing_db)
    ));
    a.push(format!(
      "--compression_type={}",
      match self.compression {
        Compression::None => "none",
        Compression::Lz4 => "lz4",
        Compression::Zstd => "zstd",
      }
    ));
    a.push(format!("--benchmarks={}", benchmarks.join(",")));
    Ok(a)
  }
}

/// 外部 db_bench 可执行。
#[derive(Debug)]
pub struct DbBench {
  bin: PathBuf,
}

impl DbBench {
  /// 定位 db_bench：优先 `$DB_BENCH`，其次在 `PATH` 中查找 `db_bench`；找不到返回
  /// 含指引的 [`Error::DbBench`]（而非静默 `None`）。
  pub fn locate() -> Result<Self> {
    if let Some(p) = env::var_os("DB_BENCH") {
      let pb = PathBuf::from(&p);
      if pb.is_file() {
        return Ok(Self { bin: pb });
      }
      return Err(Error::DbBench(format!(
        "$DB_BENCH={} 不是可执行文件，请指向已编译的 db_bench",
        pb.display()
      )));
    }
    if let Some(paths) = env::var_os("PATH")
      && let Some(bin) = env::split_paths(&paths)
        .map(|d| d.join("db_bench"))
        .find(|c| c.is_file())
    {
      return Ok(Self { bin });
    }
    Err(Error::DbBench(
      "未找到 db_bench：请设置 $DB_BENCH=/path/to/db_bench，或将其所在目录加入 $PATH".into(),
    ))
  }

  /// 可执行路径。
  pub fn bin(&self) -> &Path {
    &self.bin
  }

  /// 用同一配置跑 `benchmarks`，返回解析后的相位指标。
  ///
  /// 超时（默认 600s，可经 `$DB_BENCH_TIMEOUT_SECS` 覆盖）会终止子进程并返回
  /// [`Error::DbBench`]；非零退出码返回含 stderr 摘要的错误；stdout 解析失败一律
  /// Err（不静默返回空结果）。
  pub fn run(&self, cfg: &BenchConfig, benchmarks: &[&str]) -> Result<Vec<PeerPhase>> {
    self.run_with_timeout(cfg, benchmarks, run_timeout())
  }

  /// 带显式超时的 [`DbBench::run`]（超时可测试）。
  fn run_with_timeout(
    &self,
    cfg: &BenchConfig,
    benchmarks: &[&str],
    timeout: Duration,
  ) -> Result<Vec<PeerPhase>> {
    let args = cfg.db_bench_args(benchmarks)?;
    let mut child = Command::new(&self.bin)
      .args(&args)
      .stdout(Stdio::piped())
      .stderr(Stdio::piped())
      .spawn()
      .map_err(|e| Error::DbBench(format!("启动 {} 失败: {e}", self.bin.display())))?;
    let stdout = child
      .stdout
      .take()
      .ok_or_else(|| Error::DbBench("子进程 stdout 管道缺失".into()))?;
    let stderr = child
      .stderr
      .take()
      .ok_or_else(|| Error::DbBench("子进程 stderr 管道缺失".into()))?;
    // 边读边等：读取线程持续排空管道，主线程轮询退出 + 超时，避免满管道死锁。
    let out_h = thread::spawn(move || read_capped(stdout));
    let err_h = thread::spawn(move || read_capped(stderr));
    let deadline = Instant::now() + timeout;
    let status = loop {
      match child
        .try_wait()
        .map_err(|e| Error::DbBench(format!("等待 db_bench 失败: {e}")))?
      {
        Some(s) => break s,
        None if Instant::now() >= deadline => {
          // 只 kill 直接子进程并立即返回，**不 join 读取线程**：若子进程有孙进程
          // 仍持有管道写端，join 会一直阻塞（30s 级），使超时形同虚设。线程随
          // 进程退出回收。
          let _ = child.kill();
          let _ = child.wait();
          return Err(Error::DbBench(format!(
            "db_bench 超时（>{}s）已终止",
            timeout.as_secs()
          )));
        }
        None => thread::sleep(POLL_INTERVAL),
      }
    };
    let stdout = out_h
      .join()
      .map_err(|_| Error::DbBench("stdout 读取线程 panic".into()))?
      .map_err(|e| Error::DbBench(format!("读取 stdout 失败: {e}")))?;
    let stderr = err_h
      .join()
      .map_err(|_| Error::DbBench("stderr 读取线程 panic".into()))?
      .unwrap_or_default();
    if !status.success() {
      return Err(Error::DbBench(format!(
        "db_bench 退出码 {:?}: {}",
        status.code(),
        String::from_utf8_lossy(&stderr).trim()
      )));
    }
    parse_db_bench_report(&String::from_utf8_lossy(&stdout))
  }
}

/// 从环境读取超时（秒，须 > 0），否则 [`DEFAULT_TIMEOUT`]。
fn run_timeout() -> Duration {
  env::var("DB_BENCH_TIMEOUT_SECS")
    .ok()
    .and_then(|v| v.parse::<u64>().ok())
    .map(Duration::from_secs)
    .filter(|d| !d.is_zero())
    .unwrap_or(DEFAULT_TIMEOUT)
}

/// 读取全部字节但最多保留 [`MAX_CAPTURE`]；其余继续读丢弃以排空管道（防写阻塞）。
fn read_capped(mut r: impl Read) -> io::Result<Vec<u8>> {
  let mut kept = Vec::new();
  let mut chunk = [0u8; 64 * 1024];
  loop {
    let n = r.read(&mut chunk)?;
    if n == 0 {
      break;
    }
    if kept.len() < MAX_CAPTURE {
      let room = MAX_CAPTURE - kept.len();
      kept.extend_from_slice(&chunk[..n.min(room)]);
    }
  }
  Ok(kept)
}

/// 解析 db_bench stdout 的 benchmark 行。
///
/// 命中判据是**含 `micros/op`**（每条 benchmark 结果行都有该标记，
/// `DB path:`/配置 dump/GC-compaction 统计行都没有），因此这些噪声行被自然跳过。
/// 一旦某行含 `micros/op` 却取不到名字 / `micros/op` / `ops/sec` 数值，或整段
/// 输出**一条 benchmark 行都没有**，一律返回 [`Error::DbBench`]（绝不静默返回空）。
pub fn parse_db_bench_report(stdout: &str) -> Result<Vec<PeerPhase>> {
  let mut phases = Vec::new();
  for line in stdout.lines() {
    if !line.contains("micros/op") {
      continue;
    }
    let p = parse_line(line).ok_or_else(|| {
      Error::DbBench(format!(
        "无法解析 db_bench 输出行（格式可能变更）: {line:?}"
      ))
    })?;
    phases.push(p);
  }
  if phases.is_empty() {
    return Err(Error::DbBench(
      "db_bench 输出中未发现任何 benchmark 结果行（运行失败或格式变更）".into(),
    ));
  }
  Ok(phases)
}

/// 解析单行，形如
/// `fillrandom   :       4.201 micros/op 237783 ops/sec 0.008 seconds 2000 operations;   26.3 MB/s`。
fn parse_line(line: &str) -> Option<PeerPhase> {
  let (name, rest) = line.split_once(':')?;
  let name = name.trim();
  if name.is_empty() || name.contains(char::is_whitespace) {
    return None;
  }
  let micros_per_op = token_before(rest, "micros/op")?.parse::<f64>().ok()?;
  let ops_per_sec = token_before(rest, "ops/sec")?.parse::<f64>().ok()?;
  Some(PeerPhase {
    phase: name.to_string(),
    micros_per_op,
    ops_per_sec,
  })
}

/// `marker` 之前紧邻的数值 token。
fn token_before<'a>(s: &'a str, marker: &str) -> Option<&'a str> {
  let idx = s.find(marker)?;
  s[..idx].split_whitespace().last()
}

#[cfg(test)]
mod tests {
  use std::fs;

  use super::*;

  /// 挂起脚本测试：用真实子进程验证超时终止路径（默认 600s 太久，这里注入 60ms）。
  #[cfg(unix)]
  #[test]
  fn run_kills_process_on_timeout() {
    let path = write_script("ldb_bench_hang", "#!/bin/sh\nexec sleep 30\n");
    let Some(path) = path else { return };
    let db = DbBench { bin: path.clone() };
    let cfg = BenchConfig::default();
    let r = db.run_with_timeout(&cfg, &["readrandom"], Duration::from_millis(60));
    let _ = fs::remove_file(&path);
    match r {
      Err(Error::DbBench(m)) => assert!(m.contains("超时"), "{m}"),
      other => panic!("期望超时错误，实得 {other:?}"),
    }
  }

  /// 非零退出码 → Err 且含退出码与 stderr。
  #[cfg(unix)]
  #[test]
  fn run_reports_nonzero_exit_code() {
    let path = write_script(
      "ldb_bench_fail",
      "#!/bin/sh\necho 'boom' 1>&2\nreadrandom : 1.0 micros/op 1000 ops/sec 1 seconds\nexit 3\n",
    );
    let Some(path) = path else { return };
    let db = DbBench { bin: path.clone() };
    let cfg = BenchConfig::default();
    let r = db.run_with_timeout(&cfg, &["readrandom"], Duration::from_secs(10));
    let _ = fs::remove_file(&path);
    match r {
      Err(Error::DbBench(m)) => {
        assert!(m.contains("退出码"), "{m}");
        assert!(m.contains('3'), "应含退出码数值: {m}");
        assert!(m.contains("boom"), "应含 stderr 摘要: {m}");
      }
      other => panic!("期望退出码错误，实得 {other:?}"),
    }
  }

  /// 成功子进程 + stdout 解析 → Ok（覆盖 spawn/读取/解析全链路）。
  #[cfg(unix)]
  #[test]
  fn run_parses_successful_child_stdout() {
    let path = write_script(
      "ldb_bench_ok",
      "#!/bin/sh\necho 'fillrandom   :       4.201 micros/op 237783 ops/sec 0.008 seconds 2000 operations;   26.3 MB/s'\n",
    );
    let Some(path) = path else { return };
    let db = DbBench { bin: path.clone() };
    let cfg = BenchConfig::default();
    let r = db.run_with_timeout(&cfg, &["fillrandom"], Duration::from_secs(10));
    let _ = fs::remove_file(&path);
    match r {
      Ok(p) => {
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].phase, "fillrandom");
        assert_eq!(p[0].ops_per_sec, 237783.0);
      }
      other => panic!("期望成功解析，实得 {other:?}"),
    }
  }

  /// 输出截断：喂入超过上限的流，只保留上限字节。
  ///
  /// 期望值与输入上限均写字面量 `8 << 20`（**不复用 `MAX_CAPTURE` 常量**），
  /// 故改动常量或截断逻辑都会被本用例击杀。
  #[test]
  fn read_capped_truncates_to_limit() {
    let cap: usize = 8 << 20;
    let src = io::repeat(0u8).take(cap as u64 + 4096);
    let got = match read_capped(src) {
      Ok(v) => v,
      Err(e) => panic!("读取失败: {e}"),
    };
    assert_eq!(got.len(), cap);
    let empty = match read_capped(io::empty()) {
      Ok(v) => v,
      Err(e) => panic!("读取失败: {e}"),
    };
    assert!(empty.is_empty());
  }

  /// 写可执行脚本到临时目录；失败（非 Unix）返回 None 跳过。
  #[cfg(unix)]
  fn write_script(tag: &str, body: &str) -> Option<PathBuf> {
    use std::{os::unix::fs::PermissionsExt as _, process};
    let path = env::temp_dir().join(format!("{tag}_{}.sh", process::id()));
    fs::write(&path, body).ok()?;
    let mut perm = fs::metadata(&path).ok()?.permissions();
    perm.set_mode(0o755);
    fs::set_permissions(&path, perm).ok()?;
    Some(path)
  }
}
