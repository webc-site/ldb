//! 纯软件性能计数器（上游 `ZLeanStore/src/include/leanstore/statistics.h` / `statistics.cc`）。
//!
//! 上游声明了 36 个 `std::atomic<u64>`（`statistics.h:10-62`）；本 crate 只保留缓冲池 /
//! 事务 / GC 三类核心指标与 DB 写放大。硬件相关的实现（CPU 性能计数器、硬盘自检日志、
//! 特权块设备命令）按 aim.md「不依赖特定硬件（NVMe 即可）」整体剔除，且不靠条件编译掩藏。
//!
//! 每个计数器都是单调递增的 `fetch_add`，与上游同名量的触发点逐条对应：
//!
//! | 本 crate | 上游 | 上游「何时 +1」 |
//! | --- | --- | --- |
//! | `page_hit` | `buffer::buffer_hit_cnt`（`statistics.h:30`） | 页已在缓冲池、**无缺页**即取到 Guard：`sync/page_guard/shared_guard.cc:36,57,68`、`exclusive_guard.cc:45,70`、`optimistic_guard.cc:35,56` |
//! | `page_miss` | `buffer::buffer_miss_cnt`（`statistics.h:31`） | 缺页加载后把 pid 插入缓冲池：`buffer/buffer_manager.cc:533,567`（批量 `:1423`） |
//! | `page_evict` | `storage::space::evict_writes`（`statistics.h:45`） | 脏页驱逐落盘：`storage/space/backend/io_backend.cc:254` `evict_writes += write_size`。**上游按字节**，本方法一次调用 = 一页脏页落盘：页计数 `page_evict` +1 **且** `physical_write += bytes`（与 `record_gc_move` 同构，见下「驱逐单位」） |
//! | `txn_commit` | `total_txn_completed` / `txn_processed`（`statistics.h:10-11`） | 事务提交：`transaction/transaction_manager.cc:145`、`recovery/group_commit.cc:94`（汇总 `leanstore.cc:376-378`） |
//! | `txn_abort` | 无（本仓新增） | MVCC 回滚计数，上游无对应量，见报告 A1 |
//! | `gc_move` | `storage::space::gced_page_cnt`（`statistics.h:51`） | GC 搬移有效页：`buffer/buffer_manager.cc:1350` |
//! | `physical_write` | `storage::space::total_writes`（`statistics.h:42-43`） | 物理落盘字节。上游 `total_writes` = `evict_writes + checkpoint_writes + doublewrite_writes + gc_writes`；本仓当前**已建模**驱逐 + GC 两类（checkpoint/DWB 未建模，见 `CROSS.md`） |
//! | `logical_write` | 无（本仓新增） | 用户逻辑写入字节；上游 `db_waf` 由 GC 有效率建模（`leanstore.cc:455`），本仓改用字节比，见报告 A1 |
//!
//! # 驱逐单位（evict）决策
//! 上游 `evict_writes` 是**字节**（`io_backend.cc:254` `evict_writes += write_size`），
//! 且 `total_writes`（字节）明确声明为 `evict_writes + checkpoint_writes + doublewrite_writes + gc_writes`
//! （`statistics.h:42-43`）。本仓 WAF 的分子 `physical_write` 对应上游 `total_writes`，
//! 因此**驱逐必须先折成字节并计入 `physical_write`**，否则驱逐写会被 WAF 漏计（相对 RocksDB
//! 的低估）。故 [`record_page_evict`](StatCollector::record_page_evict) 取 `bytes` 参数
//! （调用点 = `ldb_buf` 驱逐落盘，一页 `PAGE_SIZE` 字节），与
//! [`record_gc_move`](StatCollector::record_gc_move) 同构：**一次事件同时更新页计数与物理字节**，
//! 消除「调用方只记页数、漏记字节」的接线缺陷。仅**脏页落盘**才调用（净页驱逐不产生写，
//! 上游 `evict_writes` 同样只在 `WritePage` 路径自增）。
//!
//! # 物理写唯一入口（双计防护）
//! 上游把「总写」与「类别写」在**同一次调用**里一起累加：`BufferManager::WritePages`
//! （`buffer_manager.cc:904-941`）→ `WritePagesInPlace`（`io_backend.cc:62-82`，`:72`
//! `WriteStatsUpdate(PAGE_SIZE,1,false,false)`）→ `WriteStatsUpdate`
//! （`io_backend.cc:237-269`）里 **同时** `evict_writes += write_size`（`:254`）与
//! `total_writes += write_size`（`:267`）；GC 写走 `:246-251` 的 `gcwrite` 分支，同样
//! `gc_writes` 与 `total_writes` 一起加。上游**不存在**「只加 total_writes、不加类别」的
//! 独立入口。故本 crate 也保持「一次类别事件 = 页计数 + 物理字节」：
//! [`record_page_evict`](StatCollector::record_page_evict) /
//! [`record_gc_move`](StatCollector::record_gc_move) 是 `physical_write` 的**唯一公开触发点**，
//! 二者内部统一调用 `pub(crate)` 的 `record_physical_write`。该内部入口**不对外暴露**，
//! 因此外部接线方（`ldb_buf`/`ldb_gc`）**不可能**对同一次设备写既调类别方法、又调独立的
//! `record_physical_write` 造成双计——双计在类型层即不可表达。`tests` 的
//! `page_evict_counts_bytes_exactly_once` 用字面量钉死「恰好一次」。
//! （checkpoint/DWB 物理写尚未建模；若将来需要纳入 WAF 分子，应新增各自带账本条目的
//! 类别方法，而不是恢复一个公开的通用 `record_physical_write`，否则双计风险回归。）
//!
//! # 溢出
//! 计数器均为 `u64`，`fetch_add` 溢出按 **回绕（wrap）** 语义：到 `u64::MAX` 后 +1 归零，
//! 与上游 `std::atomic<u64>++` 一致（非饱和）。`tests/metric_test.rs` 用字面量钉死该行为。

use std::sync::atomic::{AtomicU64, Ordering};

/// 统计收集器（全原子，无锁读）。
///
/// # 并发
/// 各计数器互相独立，**无跨变量不变量**，故用 `Ordering::Relaxed` 足够：
/// - `fetch_add` 在 Relaxed 下仍是原子读改写，并发递增不丢更新（`tests` 中 8 线程 × 1e6 精确总数）；
/// - 计数器不用于发布/消费其它内存，无需 Acquire/Release 语义。
///
/// # 快照
/// [`snapshot_metric`](Self::snapshot_metric) / [`calc_db_waf`](Self::calc_db_waf) 逐字段 `load`，
/// **不保证同一时刻**——上游同样逐字段 `exchange(0)`（`leanstore.cc:333-352`）后按秒采样，
/// 无原子快照语义。若将来需要同一时刻快照，应在消费侧用 seqlock，而非给每个计数器加开销。
#[derive(Debug, Default)]
pub struct StatCollector {
  page_hit: AtomicU64,
  page_miss: AtomicU64,
  /// 脏页驱逐落盘次数（页数；对应字节已折入 `physical_write`）
  page_evict: AtomicU64,
  txn_commit: AtomicU64,
  txn_abort: AtomicU64,
  gc_move: AtomicU64,
  /// 逻辑写入字节数
  logical_write: AtomicU64,
  /// 物理写入字节数（= 驱逐 + GC 搬移；对应上游 `total_writes` 的已建模子集）
  physical_write: AtomicU64,
}

impl StatCollector {
  /// 记录页命中（对应上游 `buffer::buffer_hit_cnt`）。
  #[inline]
  pub fn record_page_hit(&self) {
    self.page_hit.fetch_add(1, Ordering::Relaxed);
  }

  /// 记录页缺失，即触发缺页加载（对应上游 `buffer::buffer_miss_cnt`）。
  #[inline]
  pub fn record_page_miss(&self) {
    self.page_miss.fetch_add(1, Ordering::Relaxed);
  }

  /// 记录一次**脏页驱逐落盘**：`page_evict` +1，并把 `bytes`（一页 = `PAGE_SIZE`）累加进
  /// `physical_write`。
  ///
  /// 对应上游 `storage::space::evict_writes`（**字节**，`statistics.h:45`、
  /// `io_backend.cc:254`），其字节并入 `total_writes`（`statistics.h:42-43`）。本方法把
  /// 「驱逐页数」与「驱逐字节」一次记全（上游 `WriteStatsUpdate` 亦在同一次调用里同时加
  /// `evict_writes` 与 `total_writes`，见模块文档「物理写唯一入口」）。净页驱逐不产生设备写，
  /// **不要**调用本方法。
  ///
  /// 本方法是 `physical_write` 的公开触发点之一；字节经 `pub(crate)` 的
  /// `record_physical_write` 统一累加 ⇒ **一次驱逐只计一次字节**（无独立公开入口可再补记）。
  #[inline]
  pub fn record_page_evict(&self, bytes: u64) {
    self.page_evict.fetch_add(1, Ordering::Relaxed);
    self.record_physical_write(bytes);
  }

  /// 记录事务提交（对应上游 `total_txn_completed`）。
  #[inline]
  pub fn record_txn_commit(&self) {
    self.txn_commit.fetch_add(1, Ordering::Relaxed);
  }

  /// 记录事务回滚（本仓新增，上游无）。
  #[inline]
  pub fn record_txn_abort(&self) {
    self.txn_abort.fetch_add(1, Ordering::Relaxed);
  }

  /// 记录一次 GC 页搬移：`gc_move` +1，并把这页的物理写字节累加进 `physical_write`。
  ///
  /// 对应上游 GC 写时 `gced_page_cnt`（页数）与 `gc_writes`（字节）在同一次统计更新里一起
  /// 累加（`io_backend.cc:246-251` 的 `gcwrite` 分支）；本方法一次调用同时更新页计数与物理
  /// 字节，避免调用方漏计物理写。字节经 `pub(crate)` 的 `record_physical_write` 统一累加
  /// ⇒ 一次搬移只计一次字节。
  #[inline]
  pub fn record_gc_move(&self, bytes: u64) {
    self.gc_move.fetch_add(1, Ordering::Relaxed);
    self.record_physical_write(bytes);
  }

  /// 记录逻辑写（字节）。
  #[inline]
  pub fn record_logical_write(&self, bytes: u64) {
    self.logical_write.fetch_add(bytes, Ordering::Relaxed);
  }

  /// `physical_write` 的**唯一内部累加入口**（`pub(crate)`，不对外暴露）。
  ///
  /// 上游没有「只加 `total_writes`、不加类别」的独立写入口：`evict_writes`/`gc_writes`/
  /// `checkpoint_writes`/`doublewrite_writes` 与 `total_writes` 都在 `WriteStatsUpdate`
  /// 同一次调用里累加（`io_backend.cc:237-269`）。因此本方法只由
  /// [`record_page_evict`](Self::record_page_evict) 与
  /// [`record_gc_move`](Self::record_gc_move) 内部调用；**保持 `pub(crate)` 是为了让外部
  /// 接线方无法对同一次设备写双计**（若恢复为 `pub`，`ldb_buf` 中「类别 + 显式物理写」两调
  /// 就会把一页字节记两次，WAF 翻倍）。
  #[inline]
  pub(crate) fn record_physical_write(&self, bytes: u64) {
    self.physical_write.fetch_add(bytes, Ordering::Relaxed);
  }

  /// DB 写放大系数 `physical_write / logical_write`；无逻辑写时返回 0。
  ///
  /// 分子 `physical_write` = 驱逐字节 + GC 搬移字节（对应上游
  /// `total_writes = evict_writes + checkpoint_writes + doublewrite_writes + gc_writes`
  /// 的**已建模子集**；checkpoint/DWB 未建模，见 `CROSS.md`）。与上游**建模式**不同：
  /// 上游 `leanstore.cc:455` 用 GC 有效率 `valid_page_cnt / gced_page_cnt` 反推
  /// `db_waf = 1/(1-valid_pct)*comp_ratio`（前向估计，与实测字节无关）；本仓改用可直接
  /// 观测的物理/逻辑字节比（后向实测）。二者量纲不同、用途互补：前者用于运行时决策，
  /// 后者用于与 RocksDB 的 WAF 实测对比。
  pub fn calc_db_waf(&self) -> f64 {
    let lw = self.logical_write.load(Ordering::Relaxed);
    if lw == 0 {
      return 0.0;
    }
    self.physical_write.load(Ordering::Relaxed) as f64 / lw as f64
  }

  /// 指标快照（逐字段 `load`，非同一时刻）。
  pub fn snapshot_metric(&self) -> Metric {
    Metric {
      page_hit: self.page_hit.load(Ordering::Relaxed),
      page_miss: self.page_miss.load(Ordering::Relaxed),
      page_evict: self.page_evict.load(Ordering::Relaxed),
      txn_commit: self.txn_commit.load(Ordering::Relaxed),
      txn_abort: self.txn_abort.load(Ordering::Relaxed),
      gc_move: self.gc_move.load(Ordering::Relaxed),
      db_waf: self.calc_db_waf(),
    }
  }

  /// 导出 JSON（sonic-rs）。
  ///
  /// `Metric` 只含 `u64` 与有限 `f64`（`db_waf` 在两计数为有限值时不可能为 NaN/∞），
  /// 序列化不会失败；`unwrap_or_default()` 只是「不可达错误路径返回空缓冲」的无 panic 兜底。
  pub fn export_json(&self) -> Vec<u8> {
    let m = self.snapshot_metric();
    sonic_rs::to_vec(&m).unwrap_or_default()
  }
}

/// 指标快照值
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Metric {
  /// 页命中数
  pub page_hit: u64,
  /// 页缺失数
  pub page_miss: u64,
  /// 页驱逐数
  pub page_evict: u64,
  /// 事务提交数
  pub txn_commit: u64,
  /// 事务回滚数
  pub txn_abort: u64,
  /// GC 搬移页数
  pub gc_move: u64,
  /// DB 写放大
  pub db_waf: f64,
}
