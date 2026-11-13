//! EDT 纯估算边界回归（todo.md §9 C-13：空历史 UB、死代码、零分配重写）。
//!
//! 期望值全部来自上游 `page.h:26-140` 的真实函数体（C++ oracle 差分见本轮报告 A2），
//! 使用字面量钉死，不复用被测常量。

use ldb_edt::{
  WriteHistory, count_valid_write_histories, estimate_death_time, estimate_next_gc_write,
  get_interval_gap,
};

const fn hist(a: [u64; 8]) -> WriteHistory {
  a
}

#[test]
fn empty_history_returns_zero() {
  let h = hist([0; 8]);
  assert_eq!(estimate_death_time(&h), 0, "空历史上游 UB，目标语义固定 0");
  assert_eq!(get_interval_gap(&h), 0);
  assert_eq!(count_valid_write_histories(&h, 0), 0);
  assert_eq!(estimate_next_gc_write(7, 0, &h, true), 14);
}

#[test]
fn single_element_returns_itself() {
  let h = hist([0, 0, 0, 0, 0, 42, 0, 0]);
  assert_eq!(estimate_death_time(&h), 42);
  assert_eq!(estimate_next_gc_write(7, 0, &h, true), 14, "n=1 不触发 EDT");
}

#[test]
fn average_gap_floors_and_ignores_order() {
  // 排序后 [10,20,40]：max 40 + (10+20)/2 = 55
  assert_eq!(estimate_death_time(&hist([20, 10, 40, 0, 0, 0, 0, 0])), 55);
  // 满 8 个：max 8 + (8-1)/7 = 9
  assert_eq!(estimate_death_time(&hist([8, 1, 7, 2, 6, 3, 5, 4])), 9);
  // 平均间隔 5/2 向下取整为 2：15 + 2 = 17（向上取整会得 18）
  assert_eq!(estimate_death_time(&hist([10, 12, 15, 0, 0, 0, 0, 0])), 17);
  // 重复 GSN 保留在 n 中：[10,10,20,40] → 40 + 30/3 = 50
  assert_eq!(estimate_death_time(&hist([40, 10, 20, 10, 0, 0, 0, 0])), 50);
}

#[test]
fn interval_gap_needs_two_entries_and_keeps_array_order() {
  // 上游 n<2 返回 0（n==0 的下标是 UB，本实现也返回 0）
  assert_eq!(get_interval_gap(&hist([0, 0, 0, 0, 0, 42, 0, 0])), 0);
  assert_eq!(get_interval_gap(&hist([0; 8])), 0);
  // n>=2 返回数组顺序第一个非零，而非排序后的最小值
  assert_eq!(get_interval_gap(&hist([20, 10, 40, 0, 0, 0, 0, 0])), 20);
  assert_eq!(get_interval_gap(&hist([0, 0, 0, 0, 0, 7, 3, 0])), 7);
  assert_eq!(get_interval_gap(&hist([40, 10, 20, 10, 0, 0, 0, 0])), 40);
}

#[test]
fn next_gc_requires_more_than_two_entries() {
  // n=2 时即使 EDT(25) > cur_ts 也走 cur_ts*2
  let h = hist([20, 10, 0, 0, 0, 0, 0, 0]);
  assert_eq!(estimate_next_gc_write(1, 0, &h, true), 2);
}

#[test]
fn next_gc_requires_prev_write_behind_latest_gsn() {
  let h = hist([20, 10, 40, 0, 0, 0, 0, 0]);
  // prev == max 不满足严格小于
  assert_eq!(estimate_next_gc_write(1, 40, &h, true), 2);
  assert_eq!(estimate_next_gc_write(1, u64::MAX, &h, true), 2);
}

#[test]
fn next_gc_uses_edt_only_when_future() {
  let h = hist([20, 10, 40, 0, 0, 0, 0, 0]);
  // EDT=55：cur=54 时 55 > 54 → 55；cur=55 时严格大于不成立 → 110
  assert_eq!(estimate_next_gc_write(54, 0, &h, true), 55);
  assert_eq!(estimate_next_gc_write(55, 0, &h, true), 110);
}

#[test]
fn next_gc_flag_off_returns_zero() {
  let h = hist([8, 1, 7, 2, 6, 3, 5, 4]);
  assert_eq!(estimate_next_gc_write(7, 0, &h, false), 0);
}

#[test]
fn next_gc_wraps_like_upstream_unsigned() {
  let empty = hist([0; 8]);
  // 上游 cur_ts * 2 为 u64 无符号回绕
  assert_eq!(
    estimate_next_gc_write(u64::MAX, 0, &empty, true),
    18446744073709551614
  );
  let full = hist([8, 1, 7, 2, 6, 3, 5, 4]);
  assert_eq!(estimate_next_gc_write(1, 0, &full, true), 9);
}

#[test]
fn count_valid_write_histories_returns_wcnt_not_array_scan() {
  // 上游 return wcnt（其后循环是死代码），wcnt 可与数组内容不一致
  assert_eq!(
    count_valid_write_histories(&hist([1, 2, 3, 0, 0, 0, 0, 0]), 9),
    9
  );
  assert_eq!(count_valid_write_histories(&hist([0; 8]), 5), 5);
  assert_eq!(
    count_valid_write_histories(&hist([8, 1, 7, 2, 6, 3, 5, 4]), 3),
    3
  );
}

#[test]
fn death_time_wraps_like_upstream_unsigned() {
  // max + avg 溢出 u64：MAX + 10 回绕为 9
  assert_eq!(
    estimate_death_time(&hist([u64::MAX, u64::MAX - 10, 0, 0, 0, 0, 0, 0])),
    9
  );
  // n=3：max MAX + avg 5 回绕为 4
  assert_eq!(
    estimate_death_time(&hist([
      u64::MAX,
      u64::MAX - 10,
      u64::MAX - 5,
      0,
      0,
      0,
      0,
      0
    ])),
    4
  );
}
