//! GsnWatermark 单调性回归（替代上游 `LogManager::global_min_gsn_flushed`
//! 的直读；`advance_min` 语义 = 只增不减的最小已刷盘 GSN，`store` 仅供恢复/测试
//! 直接设置，允许回退）。

use std::{sync::Arc, thread::spawn};

use ldb_type::GsnWatermark;

#[test]
fn watermark_is_monotonic() {
  let wm = GsnWatermark::new(10);
  assert_eq!(wm.load(), 10);
  assert_eq!(wm.advance_min(5), 10); // 不回退
  assert_eq!(wm.advance_min(10), 10); // 相等不推进
  assert_eq!(wm.advance_min(99), 99); // 推进并返回新值
  assert_eq!(wm.load(), 99);
  wm.store(7);
  assert_eq!(wm.load(), 7); // store 仅恢复/测试路径，允许回退
}

/// 并发推进：CAS 循环下最终水位必须等于全局最大值，且任何一次读取都不回退。
#[test]
fn watermark_advance_is_monotonic_under_contention() {
  const THREADS: u64 = 4;
  const STEPS: u64 = 20_000;
  let wm = Arc::new(GsnWatermark::new(0));
  let handles: Vec<_> = (0..THREADS)
    .map(|t| {
      let wm = Arc::clone(&wm);
      spawn(move || {
        let mut prev = wm.load();
        for i in 0..STEPS {
          let seen = wm.advance_min(i * THREADS + t);
          assert!(seen >= prev, "水位回退: {prev} -> {seen}");
          prev = seen;
          // 单点观察也不得低于已发布值
          assert!(wm.load() >= i * THREADS + t);
        }
      })
    })
    .collect();
  for h in handles {
    h.join().unwrap();
  }
  assert_eq!(wm.load(), (STEPS - 1) * THREADS + (THREADS - 1));
}
