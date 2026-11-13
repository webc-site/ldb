//! 统计收集器回归：触发点语义、WAF 公式、溢出回绕、并发精确性、JSON 导出。
//!
//! 每条断言都对应一个可注入的变异（见报告 E1 表），期望值一律写字面量。

use std::thread;

use ldb_stat::StatCollector;

/// 逐计数器触发点 + WAF 公式 + 快照/JSON 导出（含 golden 字节串）。
///
/// 唯一击杀：删任一 `record_*` 的 `fetch_add`、`record_gc_move`/`record_page_evict`
/// 漏计物理写、`snapshot_metric` 返回默认值、`export_json` 丢字段或改字段名/顺序。
#[test]
fn counter_triggers_and_waf() {
  let s = StatCollector::default();
  for _ in 0..60 {
    s.record_page_hit();
  }
  for _ in 0..40 {
    s.record_page_miss();
  }
  for _ in 0..7 {
    s.record_page_evict(4096); // 7 页脏页落盘 = 28672 字节物理写
  }
  for _ in 0..3 {
    s.record_txn_commit();
  }
  for _ in 0..5 {
    s.record_txn_abort();
  }
  // gc_move(8192) 同时计物理写 8192；7 次驱逐各 4096 = 28672
  // -> physical = 8192 + 28672 = 36864；logical = 4096
  s.record_gc_move(8192);
  s.record_logical_write(4096);
  let m = s.snapshot_metric();
  assert_eq!(m.page_hit, 60);
  assert_eq!(m.page_miss, 40);
  assert_eq!(m.page_evict, 7);
  assert_eq!(m.txn_commit, 3);
  assert_eq!(m.txn_abort, 5);
  assert_eq!(m.gc_move, 1);
  assert!((m.db_waf - 9.0).abs() < 1e-9, "WAF = 36864/4096 = 9.0");

  let json = s.export_json();
  // golden：字段名/顺序/类型（u64 整数、f64 小数）钉死，与 sonic-rs 默认序列化一致。
  let golden = concat!(
    r#"{"page_hit":60,"page_miss":40,"page_evict":7,"txn_commit":3,"#,
    r#""txn_abort":5,"gc_move":1,"db_waf":9.0}"#,
  );
  assert_eq!(json.as_slice(), golden.as_bytes(), "golden JSON 不匹配");
  // 再走一遍 sonic-rs 解析，确认可被消费者读回。
  assert!(
    sonic_rs::from_slice::<sonic_rs::Value>(&json).is_ok(),
    "JSON 解析"
  );
}

/// WAF = physical / logical；logical=0 时返回 0（而非 inf/NaN）。
///
/// 唯一击杀：把公式写成 logical/physical（得 0.5 而非 2.0）；删 `lw == 0` 早退（得 inf）。
#[test]
fn waf_formula_and_zero_guard() {
  let z = StatCollector::default();
  z.record_gc_move(1000); // 只产生物理写，无逻辑写
  assert_eq!(z.calc_db_waf(), 0.0, "无逻辑写 -> 0.0，不是 inf");

  let f = StatCollector::default();
  f.record_logical_write(50);
  f.record_gc_move(100); // physical = 100
  assert!(
    (f.calc_db_waf() - 2.0).abs() < 1e-12,
    "physical/logical = 2.0"
  );
}

/// u64 计数器溢出 **回绕**（wrap），非饱和；用公开 API（类别事件字节）即可钉死。
///
/// 唯一击杀：把 `fetch_add` 换成饱和加（saturating_add）——两处断言都会失败。
#[test]
fn counters_wrap_not_saturate_at_u64_boundary() {
  // 物理写回绕：u64::MAX + 1 -> 0；logical = 1 => waf = 0/1 = 0.0（饱和则 waf 巨大）。
  let p = StatCollector::default();
  p.record_gc_move(u64::MAX);
  p.record_gc_move(1);
  p.record_logical_write(1);
  assert_eq!(p.calc_db_waf(), 0.0, "physical 回绕到 0");

  // 逻辑写回绕：u64::MAX + 1 -> 0 => 走 0 分支；饱和则 waf = 7/u64::MAX != 0。
  let l = StatCollector::default();
  l.record_logical_write(u64::MAX);
  l.record_logical_write(1);
  l.record_gc_move(7);
  assert_eq!(l.calc_db_waf(), 0.0, "logical 回绕到 0");

  // 驱逐字节确实并入 physical_write：先填满 u64::MAX，再闪驱 1 字节 -> 回绕到 0。
  // 若 `record_page_evict` 漏计字节，physical 停在 u64::MAX，则 waf = MAX/1 != 0（本断言失败）。
  let e = StatCollector::default();
  e.record_gc_move(u64::MAX);
  e.record_page_evict(1);
  e.record_logical_write(1);
  assert_eq!(e.snapshot_metric().page_evict, 1);
  assert_eq!(e.calc_db_waf(), 0.0, "驱逐字节并入 physical 后回绕到 0");
}

/// 双计防护（本轮重点①）：一次类别事件只把物理字节计**恰好一次**。
///
/// `record_page_evict`/`record_gc_move` 是 `physical_write` 的公开触发点，二者内部经
/// `pub(crate)` 的 `record_physical_write` 统一累加；独立入口不对外暴露 ⇒ 外部接线方
/// 无法对同一次设备写「类别 + 显式物理写」双记。
///
/// 唯一击杀：把 `record_page_evict`/`record_gc_move` 的字节累加写成两遍（双计）——
/// 下面断言会从 1.0 变成 2.0 而失败；漏计则变 0.0。
#[test]
fn page_evict_counts_bytes_exactly_once() {
  let e = StatCollector::default();
  e.record_logical_write(4096);
  e.record_page_evict(4096);
  assert_eq!(e.snapshot_metric().page_evict, 1);
  assert_eq!(e.calc_db_waf(), 1.0, "一页驱逐 = 恰好一次 4096 字节");

  let g = StatCollector::default();
  g.record_logical_write(8192);
  g.record_gc_move(8192);
  assert_eq!(g.snapshot_metric().gc_move, 1);
  assert_eq!(g.calc_db_waf(), 1.0, "一次 GC 搬移 = 恰好一次 8192 字节");
}

/// 8 线程并发递增，各计数器总数精确（Relaxed 原子无丢失更新）。
///
/// 迭代量刻意压到 <100ms（E2）；完整 8×1e6 压测放在 release 的
/// `benches/stat_bench.rs`（`cargo bench -p ldb_stat`）。
///
/// 唯一击杀：把任一计数器换非原子（`UnsafeCell<u64>` 裸读写）——必丢更新。
#[test]
fn concurrent_updates_are_exact() {
  const THREADS: u64 = 8;
  const PER_THREAD: u64 = 40_000;
  let s = StatCollector::default();
  thread::scope(|scope| {
    for _ in 0..THREADS {
      scope.spawn(|| {
        for _ in 0..PER_THREAD {
          s.record_page_hit();
          s.record_txn_commit();
          s.record_gc_move(1);
        }
      });
    }
  });
  let snap = s.snapshot_metric();
  assert_eq!(snap.page_hit, THREADS * PER_THREAD);
  assert_eq!(snap.txn_commit, THREADS * PER_THREAD);
  assert_eq!(snap.gc_move, THREADS * PER_THREAD);
  assert_eq!(snap.db_waf, 0.0, "无逻辑写");
}
