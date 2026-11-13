//! ldb_range 微基准（无第三方依赖）：ns/op 与 alloc/op。
//!
//! `cargo bench -p ldb_range` 直接运行；输出走 stderr。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  collections::VecDeque,
  hint::black_box,
  io::{Write as _, stderr},
  sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  },
  thread,
  time::Instant,
};

use ldb_range::RangeLock;

/// 计数分配器：只统计次数，转发给 System
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: 所有方法只做计数后原样转发给 System，语义与 System 完全一致。
unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: 转发给 System，layout 由调用方保证合法。
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: ptr/layout 来自本分配器的 alloc，原样转发。
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: 转发给 System，ptr/layout 由调用方保证合法。
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocs() -> usize {
  ALLOCS.load(Ordering::Relaxed)
}

/// 返回（ns/op, allocs/op）。alloc 统计区间不含任何输出。
fn bench<F: FnMut() -> u64>(name: &str, iters: u32, mut f: F) -> (f64, f64) {
  for _ in 0..(iters / 10).max(1) {
    black_box(f());
  }
  let a0 = allocs();
  let t = Instant::now();
  let mut acc = 0u64;
  for _ in 0..iters {
    acc = acc.wrapping_add(black_box(f()));
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(iters);
  let ap = (allocs() - a0) as f64 / f64::from(iters);
  black_box(acc);
  let _ = writeln!(stderr(), "{name:<46}{ns:9.2} ns/op{ap:8.2} alloc/op");
  (ns, ap)
}

/// 旧实现基线（R1 之前）：节点前向指针为 `Vec<NodeId>`，每次插入一次堆分配
struct OldArena {
  arena: Vec<(u64, u64, Vec<u32>)>,
}

impl OldArena {
  fn push(&mut self, start: u64, end: u64, level: usize) {
    self.arena.push((start, end, vec![0u32; level]));
  }
}

/// 新实现：前向指针定长内联（`[NodeId; 32]`），插入零分配
struct NewArena {
  arena: Vec<(u64, u64, [u32; 32])>,
}

impl NewArena {
  fn push(&mut self, start: u64, end: u64, level: usize) {
    let _ = level;
    self.arena.push((start, end, [0u32; 32]));
  }
}

fn main() {
  // 1) 稳态 lock+unlock 循环（槽位复用，无增长）
  let rl = RangeLock::new(8);
  let (cycle, a_cycle) = bench("try_lock+unlock steady state", 1_000_000, || {
    let ok = rl.try_lock_range(0, black_box(4096), 8);
    rl.unlock_range(0, 4096);
    u64::from(ok)
  });

  // 2) 冲突路径（命中已锁区间，最坏路径）
  let rl2 = RangeLock::new(8);
  assert!(rl2.try_lock_range(0, 4096, 8));
  let (conflict, _) = bench("try_lock conflict (hit)", 1_000_000, || {
    u64::from(!rl2.try_lock_range(0, black_box(4093), 8))
  });

  // 3) test_range（只读探针）
  let (probe, _) = bench("test_range hit", 1_000_000, || {
    u64::from(rl2.test_range(0, black_box(4093), 8))
  });

  // 4) 增长路径：100k 个不重叠区间（含跳表层增长）
  let rl3 = RangeLock::new(8);
  let mut i = 0u64;
  let (grow, _) = bench("try_lock disjoint grow", 100_000, || {
    i += 4;
    u64::from(rl3.try_lock_range(0, black_box(i), 2))
  });

  // 5) 节点分配基线对比（旧 Vec 前向 vs 新定长内联）
  let mut old = OldArena {
    arena: Vec::with_capacity(1_000_000),
  };
  let (old_ns, old_a) = bench("node push: Vec forward (old)", 1_000_000, || {
    old.push(black_box(1), 2, black_box(16));
    old.arena.len() as u64
  });
  let mut new = NewArena {
    arena: Vec::with_capacity(1_000_000),
  };
  let (new_ns, new_a) = bench("node push: fixed [u32;32] (new)", 1_000_000, || {
    new.push(black_box(1), 2, black_box(16));
    new.arena.len() as u64
  });

  // 6) churn：活跃集随 n 增长、插入与解锁交错（层高必须跟住 log n，见文件头偏差 4）
  for n in [1_000usize, 100_000, 1_000_000] {
    let rl = RangeLock::new(8);
    let mut live = VecDeque::new();
    let mut s = 1u64;
    for i in 0..n {
      assert!(rl.try_lock_range(0, s, 2));
      live.push_back(s);
      s += 4;
      if i > 64
        && i % 2 == 0
        && let Some(old) = live.pop_front()
      {
        rl.unlock_range(0, old);
      }
    }
    let keys: Vec<u64> = live.into_iter().collect();
    let mut idx = 0usize;
    bench(
      &format!("churn probe (live={})", keys.len()),
      100_000,
      || {
        idx = (idx + 7919) % keys.len();
        u64::from(rl.test_range(0, black_box(keys[idx]), 2))
      },
    );
  }

  // 7) 多线程吞吐：1/2/4/8 线程，各线程操作不相交区间（只竞争 inner 锁）
  let total_ops = 2_000_000usize;
  let mut line = String::from("threads:");
  for &t in &[1usize, 2, 4, 8] {
    let rl = Arc::new(RangeLock::new(t));
    let per = total_ops / t;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..t)
      .map(|i| {
        let rl = Arc::clone(&rl);
        thread::spawn(move || {
          let base = (i as u64) * (per as u64) * 8 + 1;
          let mut acc = 0u64;
          for j in 0..per {
            let s = base + (j as u64) * 8;
            acc += u64::from(rl.try_lock_range(i, s, 2));
            rl.unlock_range(i, s);
          }
          acc
        })
      })
      .collect();
    let acc: u64 = handles.into_iter().map(|h| h.join().unwrap_or(0)).sum();
    let ns = t0.elapsed().as_nanos() as f64;
    black_box(acc);
    let mops = total_ops as f64 / ns * 1e3;
    line.push_str(&format!(" {t}T={mops:.2}Mops/s"));
  }
  let _ = writeln!(stderr(), "{line}");

  let _ = writeln!(
    stderr(),
    "\nlock+unlock={cycle:.2} ns conflict={conflict:.2} ns probe={probe:.2} ns grow={grow:.2} ns \
     | steady-state alloc/op={a_cycle:.2}"
  );
  let _ = writeln!(
    stderr(),
    "node push: old={old_ns:.2} ns/{old_a:.2} alloc new={new_ns:.2} ns/{new_a:.2} alloc \
     delta={:+.1}%",
    (new_ns - old_ns) / old_ns * 100.0
  );
}
