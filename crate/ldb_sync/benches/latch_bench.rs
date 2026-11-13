//! ldb_sync 微基准（无第三方依赖）：无争用原子成本 + 1/2/4/8 线程阻塞闩扩展比。
//!
//! `cargo bench -p ldb_sync` 直接运行；输出走 stderr。
//! 争用路径每线程自建 compio Runtime（thread-per-core，todo.md §3.4）。

use std::{
  hint::black_box,
  io::{self, Write as _},
  ptr::NonNull,
  sync::{
    Arc, Barrier,
    atomic::{AtomicU64, Ordering, fence},
  },
  thread,
  time::Instant,
};

use compio::runtime::Runtime;
use ldb_sync::{HybridGuard, HybridLatch, OptimisticGuard, PageStateExt};
use ldb_type::{PageState, PageStateMode};

/// 单线程 ns/op（返回耗时，顺带打印）
fn bench_ns<F: FnMut() -> u64>(name: &str, iters: u32, mut f: F) -> f64 {
  for _ in 0..(iters / 10).max(1) {
    black_box(f());
  }
  let t = Instant::now();
  let mut acc = 0u64;
  for _ in 0..iters {
    acc = acc.wrapping_add(black_box(f()));
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(iters);
  black_box(acc);
  let _ = writeln!(io::stderr(), "{name:<52}{ns:9.2} ns/op");
  ns
}

/// 单线程 async 排他闩 ns/op（无争用，锁在首轮 CAS 成功）
fn bench_async_lock(name: &str, iters: u32) -> io::Result<f64> {
  let latch = HybridLatch::new();
  let rt = Runtime::new()?;
  for _ in 0..(iters / 10).max(1) {
    rt.block_on(async {
      latch.lock_exclusive().await;
      latch.unlock_exclusive();
    });
  }
  let t = Instant::now();
  for _ in 0..iters {
    black_box(rt.block_on(async {
      latch.lock_exclusive().await;
      latch.unlock_exclusive();
      1u64
    }));
  }
  let ns = t.elapsed().as_nanos() as f64 / f64::from(iters);
  let _ = writeln!(io::stderr(), "{name:<52}{ns:9.2} ns/op");
  Ok(ns)
}

/// N 线程争用同一把排他闩：每线程 ns/op，返回（单线程均值 ns/op）
fn contended_ns(threads: usize, iters: u32) -> io::Result<f64> {
  let latch = Arc::new(HybridLatch::new());
  let barrier = Arc::new(Barrier::new(threads));
  let mut handles = Vec::with_capacity(threads);
  for _ in 0..threads {
    let latch = Arc::clone(&latch);
    let barrier = Arc::clone(&barrier);
    handles.push(thread::spawn(move || -> io::Result<f64> {
      let rt = Runtime::new()?;
      barrier.wait();
      let t = Instant::now();
      rt.block_on(async {
        for _ in 0..iters {
          latch.lock_exclusive().await;
          // 临界区：一次原子读，防止整个循环被折叠
          black_box(latch.state_and_version().load(Ordering::Relaxed));
          latch.unlock_exclusive();
        }
      });
      Ok(t.elapsed().as_nanos() as f64 / f64::from(iters))
    }));
  }
  let mut sum = 0.0;
  for h in handles {
    match h.join() {
      Ok(Ok(ns)) => sum += ns,
      _ => return Err(io::Error::other("bench thread failed")),
    }
  }
  Ok(sum / threads as f64)
}

fn main() -> io::Result<()> {
  let _ = writeln!(io::stderr(), "== ldb_sync 单线程原子/闩路径 ==");
  let latch = HybridLatch::new();
  bench_ns(
    "hybrid_latch/try_exclusive+unlock_exclusive",
    5_000_000,
    || {
      let old = latch.state_and_version().load(Ordering::Relaxed);
      let ok = latch.try_lock_exclusive(old);
      latch.unlock_exclusive();
      u64::from(ok)
    },
  );
  bench_ns("hybrid_latch/try_shared+unlock_shared", 5_000_000, || {
    let old = latch.state_and_version().load(Ordering::Relaxed);
    let ok = latch.try_lock_shared(old);
    latch.unlock_shared();
    u64::from(ok)
  });
  bench_ns(
    "hybrid_guard/optimistic_lock+validate",
    5_000_000,
    || match HybridGuard::optimistic_lock(&latch) {
      Ok(mut g) => u64::from(g.validate_optimistic_lock().is_ok()),
      Err(_) => 1,
    },
  );
  let ps = PageState::new();
  bench_ns("page_state/try_shared+unlock_shared", 5_000_000, || {
    let old = ps.load(Ordering::Relaxed);
    let ok = ps.try_lock_shared(old);
    ps.unlock_shared();
    u64::from(ok)
  });

  // 有界自旋失败路径：EXCLUSIVE 持有时 `OptimisticGuard::init` 自旋 64 次后返回
  // `Err(Restart)`；上游同分支是无界自旋（`optimistic_guard.cc:70-95`）。该数字
  // 是「一次可恢复 Restart」的固定成本上界（不随持锁时间增长）。
  let ps_locked = PageState::new();
  let v = ps_locked.load(Ordering::Relaxed);
  ps_locked.store(
    ldb_type::same_version_new_state(v, PageStateMode::EXCLUSIVE),
    Ordering::Release,
  );
  bench_ns(
    "page_guard/init_exclusive_spin_bound(64)",
    1_000_000,
    || {
      u64::from(
        OptimisticGuard::init(0, NonNull::<u64>::dangling(), &ps_locked)
          .err()
          .is_some(),
      )
    },
  );

  // fence A/B：隔离 `fence(Acquire)` 的机器指令成本（x86 为编译器屏障，
  // aarch64 为 `dmb ishld`）。两次 load 之间用 black_box 阻止编译器合并。
  let word = AtomicU64::new(0);
  bench_ns("ordering/fence_acquire+load_acquire", 20_000_000, || {
    fence(Ordering::Acquire);
    word.load(Ordering::Acquire)
  });
  bench_ns("ordering/load_acquire_only", 20_000_000, || {
    word.load(Ordering::Acquire)
  });
  bench_ns("ordering/fence_acquire+load_relaxed", 20_000_000, || {
    fence(Ordering::Acquire);
    word.load(Ordering::Relaxed)
  });

  // 校验路径 A/B 副本：与 `HybridGuard::optimistic_lock + validate_optimistic_lock`
  // 同形（快照 load -> 页数据读 -> 校验 load），只差 fence。
  let replica = HybridLatch::new();
  bench_ns("hybrid_guard/validate_replica_no_fence", 5_000_000, || {
    let snapshot = replica.state_and_version().load(Ordering::Acquire);
    black_box(&replica);
    let latest = replica.state_and_version().load(Ordering::Acquire);
    u64::from(latest == snapshot)
  });
  bench_ns(
    "hybrid_guard/validate_replica_with_fence",
    5_000_000,
    || {
      let snapshot = replica.state_and_version().load(Ordering::Acquire);
      black_box(&replica);
      fence(Ordering::Acquire);
      let latest = replica.state_and_version().load(Ordering::Acquire);
      u64::from(latest == snapshot)
    },
  );
  bench_ns(
    "hybrid_guard/validate_replica_fence_relaxed_load",
    5_000_000,
    || {
      let snapshot = replica.state_and_version().load(Ordering::Acquire);
      black_box(&replica);
      fence(Ordering::Acquire);
      let latest = replica.state_and_version().load(Ordering::Relaxed);
      u64::from(latest == snapshot)
    },
  );

  let _ = writeln!(io::stderr(), "== ldb_sync async 无争用路径 ==");
  bench_async_lock("hybrid_latch/lock_exclusive(async,uncontended)", 200_000)?;

  let _ = writeln!(io::stderr(), "== ldb_sync 排他闩争用扩展比 ==");
  let iters = 30_000;
  let base = contended_ns(1, iters)?;
  let _ = writeln!(
    io::stderr(),
    "{:<8}{:>14}{:>14}{:>16}",
    "threads",
    "ns/op/线程",
    "吞吐(Mops/s)",
    "扩展比(×1线程)"
  );
  for threads in [1usize, 2, 4, 8] {
    let ns = if threads == 1 {
      base
    } else {
      contended_ns(threads, iters)?
    };
    let mops = threads as f64 * 1e3 / ns;
    // 扩展比 = 总吞吐相对 1 线程的倍数（>1 正扩展，<1 负扩展）
    let scale = threads as f64 * base / ns;
    let _ = writeln!(
      io::stderr(),
      "{:<8}{:>14.2}{:>14.2}{:>16.2}",
      threads,
      ns,
      mops,
      scale
    );
  }
  Ok(())
}
