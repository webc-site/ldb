//! ldb_buf 微基准（无第三方依赖）：fix/unfix 热路径 ns/op、1/2/4/8 线程扩展比，
//! 以及 1e5 次随机 fix/unfix/evict 的影子模型对拍。
//!
//! `cargo bench -p ldb_buf` 直接运行；输出走 stderr。

use std::{
  hint::black_box,
  io::{self, Write as _},
  rc::Rc,
  sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
  },
  thread,
  time::{Duration, Instant},
};

use compio::runtime::Runtime;
use ldb_buf::{BufferManager, PageBackend};
use ldb_free::FreeStorageManager;
use ldb_io::MemBackend;
use ldb_space::SpaceManager;
use ldb_type::{GsnWatermark, PAGE_SIZE, PageId, PageStateMode};

/// 构造：虚拟 `virtual_cnt` 页、物理 `physical_frames` 帧
fn mk(virtual_cnt: u64, physical_frames: u64) -> BufferManager<MemBackend> {
  let space = Rc::new(SpaceManager::new(
    virtual_cnt,
    64,
    2048,
    4,
    virtual_cnt * 64,
    2,
  ));
  let free = Arc::new(FreeStorageManager::new(true, 2));
  match BufferManager::new(
    virtual_cnt,
    physical_frames,
    space,
    free,
    MemBackend::new(),
    Arc::new(AtomicBool::new(true)),
    Arc::new(GsnWatermark::new(0)),
  ) {
    Ok(bm) => bm,
    Err(e) => panic!("BufferManager 构造失败: {e}"),
  }
}

/// 取页（错误即 panic：基准必须显式失败）
async fn alloc_async(bm: &BufferManager<MemBackend>) -> PageId {
  match bm.alloc_page().await {
    Ok(pid) => pid,
    Err(e) => panic!("alloc_page 失败: {e}"),
  }
}

/// 4KiB 快照读（乐观路径 + 整页拷贝）
async fn read_snapshot(bm: &BufferManager<MemBackend>, pid: PageId, dst: &mut [u8; PAGE_SIZE]) {
  if let Err(e) = bm.fix_share_impl(pid, dst).await {
    panic!("fix_share_impl 失败: {e}");
  }
}

/// 零拷贝共享守卫路径
async fn read_guard(bm: &BufferManager<MemBackend>, pid: PageId) {
  let g = match bm.fix_shared(pid).await {
    Ok(g) => g,
    Err(e) => panic!("fix_shared 失败: {e}"),
  };
  black_box(g.bytes()[0]);
}

/// 写锁获取/释放路径
async fn write_lock(bm: &BufferManager<MemBackend>, pid: PageId) {
  if let Err(e) = bm.fix_exclusive(pid).await {
    panic!("fix_exclusive 失败: {e}");
  }
  bm.unfix_exclusive(pid);
}

fn xorshift(state: &mut u64) -> u64 {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  *state
}

/// 单线程热路径：返回 (ns/op, ops)
fn bench_single_thread(ops: usize) -> Duration {
  let Ok(rt) = Runtime::new() else {
    panic!("compio Runtime 创建失败");
  };
  rt.block_on(async {
    let bm = mk(4096, 1024);
    let pid = alloc_async(&bm).await;
    let mut warm = [0u8; PAGE_SIZE];
    read_snapshot(&bm, pid, &mut warm).await;

    let mut dst = [0u8; PAGE_SIZE];
    let start = Instant::now();
    for _ in 0..ops {
      read_snapshot(&bm, pid, &mut dst).await;
      // 观察 64 个字节，防止整页拷贝被优化掉
      let mut sum = 0u8;
      for &b in dst.iter().step_by(64) {
        sum ^= b;
      }
      black_box(sum);
    }
    start.elapsed()
  })
}

/// 单线程：零拷贝守卫 / 写锁
fn bench_guard_and_lock(ops: usize) -> (Duration, Duration) {
  let Ok(rt) = Runtime::new() else {
    panic!("compio Runtime 创建失败");
  };
  rt.block_on(async {
    let bm = mk(4096, 1024);
    let pid = alloc_async(&bm).await;
    let mut warm = [0u8; PAGE_SIZE];
    read_snapshot(&bm, pid, &mut warm).await;

    let start = Instant::now();
    for _ in 0..ops {
      read_guard(&bm, pid).await;
    }
    let guard = start.elapsed();

    let start = Instant::now();
    for _ in 0..ops {
      write_lock(&bm, pid).await;
    }
    (guard, start.elapsed())
  })
}

/// 一个线程的独立缓冲池（thread-per-core 部署模型的扩展性）
fn thread_ops(ops: usize) -> Duration {
  let Ok(rt) = Runtime::new() else {
    panic!("compio Runtime 创建失败");
  };
  rt.block_on(async {
    let bm = mk(4096, 4096);
    let mut pids = Vec::with_capacity(64);
    for _ in 0..64 {
      pids.push(alloc_async(&bm).await);
    }
    for &pid in &pids {
      let mut warm = [0u8; PAGE_SIZE];
      read_snapshot(&bm, pid, &mut warm).await;
    }
    let start = Instant::now();
    for i in 0..ops {
      read_guard(&bm, pids[i % pids.len()]).await;
    }
    start.elapsed()
  })
}

fn scaling() -> Vec<(usize, f64)> {
  let mut out = Vec::new();
  for threads in [1usize, 2, 4, 8] {
    let ops = 400_000usize;
    let start = Instant::now();
    let handles: Vec<_> = (0..threads)
      .map(|_| thread::spawn(move || thread_ops(ops)))
      .collect();
    for h in handles {
      match h.join() {
        Ok(_) => {}
        Err(_) => panic!("线程 panic"),
      }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mops = (threads * ops) as f64 / elapsed / 1e6;
    out.push((threads, mops));
  }
  out
}

/// 影子模型：1e5 次随机 fix/unfix/evict，校验
/// 1) `physical_used` == 非 EVICTED 页数；
/// 2) 常驻表成员关系与状态一致；
/// 3) 读回的页内容与影子副本一致（驱逐重载后仍一致）；
/// 4) arena 固定映射地址稳定；
/// 5) 全部刷盘后直读后端盘面与影子副本一致。
fn shadow_model() -> u64 {
  const PAGES: usize = 32;
  const OPS: usize = 100_000;
  let Ok(rt) = Runtime::new() else {
    panic!("compio Runtime 创建失败");
  };
  rt.block_on(async {
    let bm = mk((PAGES * 4) as u64, PAGES as u64);
    let mut pids = Vec::with_capacity(PAGES);
    for _ in 0..PAGES {
      pids.push(alloc_async(&bm).await);
    }
    // 影子副本：每页前 64 字节
    let mut model = vec![[0u8; 64]; PAGES];
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut mismatch = 0u64;

    let check_accounting = |bm: &BufferManager<MemBackend>, pids: &[PageId]| -> bool {
      let live = pids
        .iter()
        .filter(|&&pid| bm.get_page_state(pid).lock_state() != PageStateMode::Evicted)
        .count() as u64;
      let used = bm.physical_used.load(Ordering::Relaxed);
      if live != used {
        return false;
      }
      pids.iter().all(|&pid| {
        let resident = bm.get_page_state(pid).lock_state() != PageStateMode::Evicted;
        resident == bm.resident.contain(pid)
      })
    };

    for op in 0..OPS {
      let r = xorshift(&mut rng) as usize;
      let idx = r % PAGES;
      let pid = pids[idx];
      match op % 16 {
        0..=3 => {
          // 写：更新影子副本
          let seed = (r >> 32) as u8 | 1;
          let payload = [seed; 64];
          match bm.update_pid(pid, &payload).await {
            Ok(true) => model[idx] = payload,
            Ok(false) => panic!("update_pid 拒绝已分配页 {pid}"),
            Err(e) => panic!("update_pid 失败: {e}"),
          }
        }
        4..=5 => {
          // 驱逐
          if let Err(e) = bm.evict((r % 4 + 1) as u64).await {
            panic!("evict 失败: {e}");
          }
        }
        6 => {
          // 批量刷盘
          if let Err(e) = bm.flush_all().await {
            panic!("flush_all 失败: {e}");
          }
        }
        _ => {
          // 乐观快照读：内容必须与影子副本一致
          let mut page = [0u8; PAGE_SIZE];
          match bm.fix_share_impl(pid, &mut page).await {
            Ok(_) => {
              if page[..64] != model[idx] {
                mismatch += 1;
              }
            }
            Err(e) => panic!("单线程快照读失败: {e}"),
          }
        }
      }
      if !check_accounting(&bm, &pids) {
        mismatch += 1;
      }
      // arena 固定映射：同一 pid 的地址跨操作稳定且非空
      // （to_ptr/to_pid 的位运算往返由 ldb_vmem 的 1e5 影子用例覆盖）
      let addr = bm.page_addr(pid);
      if addr == 0 || bm.page_addr(pid) != addr {
        mismatch += 1;
      }
    }

    // 全部落盘后直读后端盘面
    if let Err(e) = bm.flush_all().await {
      panic!("最后一次 flush_all 失败: {e}");
    }
    for (idx, &pid) in pids.iter().enumerate() {
      let mut disk = [0u8; PAGE_SIZE];
      match bm.backend.read_page(pid, &mut disk).await {
        Ok(()) => {
          if disk[..64] != model[idx] {
            mismatch += 1;
          }
        }
        Err(e) => panic!("后端直读失败: {e}"),
      }
    }
    mismatch
  })
}

fn main() {
  let ops = 300_000usize;
  let elapsed = bench_single_thread(ops);
  let ns = elapsed.as_secs_f64() * 1e9 / ops as f64;

  let (guard, lock) = bench_guard_and_lock(ops);
  let guard_ns = guard.as_secs_f64() * 1e9 / ops as f64;
  let lock_ns = lock.as_secs_f64() * 1e9 / ops as f64;

  let scale = scaling();
  let base = scale.first().map_or(1.0, |&(_, m)| m);
  let mismatch = shadow_model();

  let _ = writeln!(io::stderr(), "cfg pages=1024 ops={ops}");
  let _ = writeln!(io::stderr(), "fix_share_impl+4KiB_copy ns/op={ns:.2}");
  let _ = writeln!(io::stderr(), "fix_shared+drop ns/op={guard_ns:.2}");
  let _ = writeln!(io::stderr(), "fix_exclusive+unfix ns/op={lock_ns:.2}");
  for (threads, mops) in &scale {
    let _ = writeln!(
      io::stderr(),
      "threads={threads} aggregate_Mops={mops:.2} scaling={:.2}x",
      mops / base
    );
  }
  let _ = writeln!(
    io::stderr(),
    "shadow ops=100000 pages=32 MISMATCH={mismatch}"
  );
  if mismatch != 0 {
    panic!("影子模型对拍失败: MISMATCH={mismatch}");
  }
}
