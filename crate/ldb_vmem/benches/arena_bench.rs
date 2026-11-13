//! ldb_vmem 微基准（无第三方依赖）：O(1) 地址换算 ns/op 与首次触页曲线。
//!
//! `cargo bench -p ldb_vmem` 直接运行；输出走 stderr。

#[cfg(unix)]
use std::process::{Command, id};
use std::{
  hint::black_box,
  io::{self, Write as _},
  iter, slice,
  time::Instant,
};

use ldb_type::PageId;
use ldb_vmem::VirtualMemoryArena;

const PAGE: usize = 4096;

/// 当前进程 RSS（KiB）；`ps` 不可用时返回 0
#[cfg(unix)]
fn rss_kb() -> usize {
  let pid = id().to_string();
  match Command::new("ps").args(["-o", "rss=", "-p", &pid]).output() {
    Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
      .trim()
      .parse()
      .unwrap_or(0),
    _ => 0,
  }
}

/// 当前进程 RSS（KiB）；不支持的平台返回 0
#[cfg(not(unix))]
fn rss_kb() -> usize {
  0
}

/// 页字节视图
fn page_slice(arena: &VirtualMemoryArena, pid: PageId) -> &[u8] {
  // SAFETY: pid < 页数；每个逻辑页独占槽位（stride >= 4KiB）；单线程
  unsafe { slice::from_raw_parts(arena.to_ptr(pid), PAGE) }
}

/// 按 64 字节周期写页并返回影子副本
fn write_pattern(arena: &VirtualMemoryArena, pid: usize, seed: u8) -> Box<[u8; PAGE]> {
  let mut expected = Box::new([0u8; PAGE]);
  for (i, chunk) in expected.as_chunks_mut::<64>().0.iter_mut().enumerate() {
    chunk.fill(seed.wrapping_add(i as u8));
  }
  // SAFETY: pid < 页数；单线程独占该页
  let page = unsafe { slice::from_raw_parts_mut(arena.to_ptr(pid as PageId), PAGE) };
  page.copy_from_slice(&expected[..]);
  expected
}

fn next_rand(state: &mut u64) -> u64 {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  *state
}

fn pick_live(live: &[bool], start: usize) -> Option<usize> {
  (0..live.len())
    .map(|k| (start + k) % live.len())
    .find(|&pid| live[pid])
}

/// 影子模型：1e5 次随机 alloc/free/write/read，逐字节比对。
///
/// 分配/复用策略模拟 `ldb_buf::alloc_page` + `ldb_space::take_reusable_pid`
/// 的 pid 复用（真实空闲表在 L3）；本基准钉死 arena 的地址映射与页隔离。
/// 作为基准而非 `#[test]`：变异集与 address_mapping_roundtrip 重合，且
/// 100k 次逐字节比对超过清单 E2 的 100ms 上限。
fn shadow_model() {
  const CNT: usize = 64;
  const OPS: usize = 100_000;
  let Ok(arena) = VirtualMemoryArena::new_arena(PAGE * CNT) else {
    let _ = writeln!(io::stderr(), "shadow: arena 预留失败");
    return;
  };
  // Windows new_arena 只 reserve；基准直接写页前需显式提交（Unix 为 no-op）
  if let Err(e) = arena.commit_pages(0, PAGE * CNT) {
    let _ = writeln!(io::stderr(), "shadow: commit_pages 失败: {e}");
    return;
  }
  let mut shadow: Vec<Option<Box<[u8; PAGE]>>> = iter::repeat_with(|| None).take(CNT).collect();
  let mut live = [false; CNT];
  let mut free: Vec<usize> = Vec::new();
  let mut next = 0usize;
  let mut rng = 0x9E37_79B9_7F4A_7C15u64;
  let mut mismatches = 0u64;

  for _ in 0..OPS {
    let op = next_rand(&mut rng) & 3;
    match op {
      0 | 1 => {
        // 池满时按 VMCache 语义驱逐一个随机 live 页再复用（物理帧有上限）
        if free.is_empty() && next == CNT {
          match pick_live(&live, rng as usize % CNT) {
            Some(victim) => {
              live[victim] = false;
              shadow[victim] = None;
              free.push(victim);
            }
            None => continue,
          }
        }
        let pid = match free.pop() {
          Some(pid) => pid,
          None => {
            assert!(next < CNT, "基准页池耗尽");
            let pid = next;
            next += 1;
            pid
          }
        };
        assert!(!live[pid], "重复分配 pid={pid}");
        live[pid] = true;
        let seed = (pid as u8) ^ (rng as u8);
        shadow[pid] = Some(write_pattern(&arena, pid, seed));
      }
      2 => {
        if let Some(pid) = pick_live(&live, rng as usize % CNT) {
          assert!(live[pid], "重复释放 pid={pid}");
          live[pid] = false;
          shadow[pid] = None;
          free.push(pid);
        }
      }
      _ => {
        if let Some(pid) = pick_live(&live, rng as usize % CNT) {
          let seed = (rng as u8) ^ (pid as u8);
          let expected = write_pattern(&arena, pid, seed);
          if page_slice(&arena, pid as PageId)[..] != expected[..] {
            mismatches += 1;
          }
          shadow[pid] = Some(expected);
        }
      }
    }
  }

  for (pid, slot) in shadow.iter().enumerate() {
    if let Some(expected) = slot.as_deref()
      && page_slice(&arena, pid as PageId)[..] != expected[..]
    {
      mismatches += 1;
    }
  }
  let _ = writeln!(
    io::stderr(),
    "shadow ops={OPS} live={} mismatches={mismatches}",
    live.iter().filter(|v| **v).count()
  );
  assert_eq!(mismatches, 0, "影子模型逐字节不一致");
}

fn bench_ns<F: FnMut(u64) -> u64>(name: &str, iters: u64, mut f: F) {
  for i in 0..(iters / 10).max(1) {
    black_box(f(i));
  }
  let t = Instant::now();
  let mut acc = 0u64;
  for i in 0..iters {
    acc = acc.wrapping_add(black_box(f(black_box(i))));
  }
  let ns = t.elapsed().as_nanos() as f64 / iters as f64;
  black_box(acc);
  let _ = writeln!(io::stderr(), "{name:<28}{ns:9.2} ns/op");
}

fn touch_curve() {
  let Ok(probe) = VirtualMemoryArena::new_arena(PAGE * 2) else {
    let _ = writeln!(io::stderr(), "touch: stride 探针 arena 预留失败");
    return;
  };
  let stride = probe.to_ptr(1).addr() - probe.to_ptr(0).addr();
  for &pages in &[1usize, 64, 1024, 8192, 16384] {
    let Ok(arena) = VirtualMemoryArena::new_arena(PAGE * pages) else {
      let _ = writeln!(io::stderr(), "touch pages={pages}: arena 预留失败");
      continue;
    };
    if let Err(e) = arena.commit_pages(0, PAGE * pages) {
      let _ = writeln!(io::stderr(), "touch pages={pages}: commit_pages 失败: {e}");
      continue;
    }
    let rss0 = rss_kb();

    let t = Instant::now();
    for pid in 0..pages {
      // SAFETY: pid < pages，to_ptr 返回槽位内合法地址
      unsafe { arena.to_ptr(pid as PageId).write(1) };
    }
    let first = t.elapsed();

    let t = Instant::now();
    let mut sum = 0u8;
    for pid in 0..pages {
      // SAFETY: 同上，页已驻留
      sum ^= unsafe { arena.to_ptr(pid as PageId).read() };
    }
    let second = t.elapsed();
    black_box(sum);

    let rss1 = rss_kb();
    let logical_mb = (pages * PAGE) as f64 / (1 << 20) as f64;
    let physical_mb = (pages * stride) as f64 / (1 << 20) as f64;
    let ns_page = first.as_nanos() as f64 / pages as f64;
    let _ = writeln!(
      io::stderr(),
      "touch pages={pages:<6} stride={stride:<6} logical={logical_mb:7.2}MiB \
       slot={physical_mb:7.2}MiB first={:?} second={:?} {ns_page:8.1} ns/page \
       rss_delta={}KiB",
      first,
      second,
      rss1.saturating_sub(rss0)
    );
  }
}

fn main() {
  let Ok(arena) = VirtualMemoryArena::new_arena(PAGE * (1 << 16)) else {
    let _ = writeln!(io::stderr(), "arena 预留失败");
    return;
  };
  let mask = (1u64 << 16) - 1;
  let iters = 10_000_000u64;

  bench_ns("to_ptr (shift)", iters, |i| {
    arena.to_ptr(black_box(i) & mask).addr() as u64
  });
  bench_ns("to_pid (shift)", iters, |i| {
    let p = arena.to_ptr(black_box(i) & mask);
    arena.to_pid(black_box(p))
  });
  bench_ns("is_valid_ptr", iters, |i| {
    let p = arena.to_ptr(black_box(i) & mask);
    u64::from(arena.is_valid_ptr(black_box(p)))
  });

  shadow_model();
  touch_curve();
}
