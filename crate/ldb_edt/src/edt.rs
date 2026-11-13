//! 页面死亡时间（EDT）纯估算（上游 `page.h:26-140` 区间内 4 个 `PageHeader` 纯估算方法的重写）。
//!
//! 上游每次调用分配 `std::vector`、含死代码与空 vector 越界（todo.md §9 C-13）；
//! 本实现为零分配、单次遍历的定长数组纯函数，显式处理 0/1 元素边界。
//!
//! 语义对照（逐函数经 C++ 真实函数体差分核验，含 u64 回绕边界）：
//! - `count_valid_write_histories`（`page.h:39-48`）：上游 `return wcnt`（其后循环为死代码）→ 直接返回 `wcnt`
//! - `estimate_death_time`（`page.h:51-71`）：`max_gsn + avg_gap`，平均间隔向下取整；
//!   上游 `n == 1` 返回该值，`n == 0` 空 vector `back()`/除零为 UB → 返回 0
//! - `get_interval_gap`（`page.h:73-99`）：上游 `n < 1` 空 vector 下标 UB、`n < 2` 返回 0、
//!   `n >= 2` 在排序前返回数组顺序第一个非零 GSN（其后为死代码）→ 空历史返回 0
//! - `estimate_next_gc_write`（`page.h:116-140`）：`FLAGS_use_edt` 入参化；空历史 `cur_ts * 2`
//!
//! 排序不必要：升序相邻差之和恒等于 `max - min`（望远镜求和），故只需求非零数、最小、最大值。

use ldb_type::{LogId, WRITE_HISTORY_LENGTH};

/// 定长写入历史视图
pub type WriteHistory = [LogId; WRITE_HISTORY_LENGTH];

/// 一次遍历求（非零数, 最小值, 最大值）；空历史返回 `(0, LogId::MAX, 0)`
#[inline]
fn history_stats(hist: &WriteHistory) -> (u32, LogId, LogId) {
  let mut n = 0u32;
  let mut min = LogId::MAX;
  let mut max = 0;
  for &g in hist {
    if g != 0 {
      n += 1;
      min = min.min(g);
      max = max.max(g);
    }
  }
  (n, min, max)
}

/// 由统计量算 EDT：`max + (max - min) / (n - 1)`，向下取整
#[inline]
fn death_time(n: u32, min: LogId, max: LogId) -> LogId {
  match n {
    0 => 0,
    1 => max,
    // 上游 u64 无符号溢出回绕（C++ 有定义）；wrapping_add 使 debug/release 与上游一致且不 panic
    _ => max.wrapping_add((max - min) / (n as LogId - 1)),
  }
}

/// 写入计数：原样返回调用方的 `wcnt`（上游 `page.h:40` 的 `return wcnt`，
/// 其后的数组扫描是死代码）。
///
/// **信任边界**：不校验 `wcnt` 与数组非零数的关系。上游 `AddWriteGSN` 每次写都
/// `wcnt++`，环形数组被覆盖后 `wcnt` 可大于 8（它是累计写次数，不是数组有效条数）；
/// 上游 `CalcEDT` 只用 `Ensure(CountValidWriteHistories())` 保证非零
/// （`buffer_manager.cc:1269-1270`）。调用方若需要数组真实非零数，必须自行扫描。
#[inline]
pub fn count_valid_write_histories(_hist: &WriteHistory, wcnt: u32) -> u32 {
  wcnt
}

/// 估算死亡时间：`max_gsn + avg_gap`（零分配、无排序）
#[inline]
pub fn estimate_death_time(hist: &WriteHistory) -> LogId {
  let (n, min, max) = history_stats(hist);
  death_time(n, min, max)
}

/// 间隔取值：上游仅在 `n >= 2` 时返回数组顺序第一个非零 GSN；`n < 2` 返回 0
/// （`n == 0` 上游是空 vector 下标 UB，本实现返回 0）
#[inline]
pub fn get_interval_gap(hist: &WriteHistory) -> LogId {
  let mut n = 0u32;
  let mut first = 0;
  for &g in hist {
    if g != 0 {
      if n == 0 {
        first = g;
      }
      n += 1;
    }
  }
  if n < 2 { 0 } else { first }
}

/// 估算下一次 GC 写时间
///
/// - `use_edt == false` → 0
/// - 历史为空 → `cur_ts * 2`（u64 回绕，与上游一致）
/// - 非零历史数 > 2、上次 GC 写落后于最新 GSN、且 EDT 超过当前时间 → EDT
/// - 否则 → `cur_ts * 2`
#[inline]
pub fn estimate_next_gc_write(
  cur_ts: LogId,
  prev_gcwrite: LogId,
  hist: &WriteHistory,
  use_edt: bool,
) -> LogId {
  if !use_edt {
    return 0;
  }
  let (n, min, max) = history_stats(hist);
  if n > 2 && prev_gcwrite < max {
    let edt = death_time(n, min, max);
    if edt > cur_ts {
      return edt;
    }
  }
  cur_ts.wrapping_mul(2)
}
