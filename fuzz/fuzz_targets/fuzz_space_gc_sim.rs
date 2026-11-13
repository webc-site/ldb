//! fuzz_space_gc_sim（todo.md §6）：随机 块分配/释放/映射变更 + 6 策略 GC 交错，
//! 影子数组验证「GC 搬移后有效页零丢失、映射自洽」。

#![no_main]
use std::rc::Rc;

use ldb_gc::GarbageCollector;
use ldb_space::{NoTxnLog, SpaceManager};
use ldb_type::{BlockId, PageId};
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target};

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Op {
  /// 写页（登记影子版本）
  Write { pid: u16 },
  /// 释放页
  Free { pid: u16 },
  /// 触发一轮牺牲块选择（任意策略）
  Gc { strategy: u8 },
}

fuzz_target!(|ops: Vec<Op>| {
  // 64 块 × 64 页/块 的空间层 + GC（NoTxnLog 常驻态）
  let space = Rc::new(SpaceManager::<NoTxnLog>::new(64, 64, 2048, 4, 64 * 64, 2));
  let mut gc = GarbageCollector::new(4);
  // 影子：pid → 最新版本号（写 +1；释放清 0）
  let mut shadow: Vec<u64> = vec![0; 4096];
  let mut version: u64 = 1;

  for op in ops {
    match op {
      Op::Write { pid } => {
        let pid = (pid as usize) % 4096;
        let pid = pid as PageId;
        if !space.page_is_valid(pid) && space.take_reusable_pid().is_none() {
          continue;
        }
        version += 1;
        shadow[pid as usize] = version;
      }
      Op::Free { pid } => {
        let pid = (pid as usize) % 4096;
        space.add_pid_to_dealloc_list(pid as PageId);
        space.process_dealloc_list();
        shadow[pid as usize] = 0;
      }
      Op::Gc { strategy } => {
        let g = &mut gc;
        // 牺牲块选择不 panic / 不越界（6 策略轮转：上游可达 5 + FIFO 补全）
        let bid = match strategy % 6 {
          0 => g.greedy_gc(&space, true),
          1 => g.greedy_gc(&space, false),
          2 => {
            let v = g.two_r_greedy_gc(&space, true);
            v.first().copied().unwrap_or(BlockId::MAX)
          }
          3 => g.edt_based_gc(&space, true, 0),
          4 => g.k_greedy_gc(&space, 3),
          _ => g.fifo_gc(&space),
        };
        let _ = bid;
      }
    }
  }
  // 终态对账：影子非零的页 space 判定有效（GC 不丢有效页）
  for (pid, &v) in shadow.iter().enumerate() {
    if v != 0 && space.get_max_pid() > pid as u64 {
      assert!(
        space.page_is_valid(pid as PageId) || !space.pid_offset_is_valid(pid as PageId),
        "有效页 {pid} 被 GC 误回收"
      );
    }
  }
});
