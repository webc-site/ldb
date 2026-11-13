//! 虚拟内存 arena 回归：地址换算、主机页槽位、区间校验与驱逐语义。
//!
//! 1e5 次地址往返影子模型（4KiB/16KiB/64KiB stride）在 `src/arena.rs` 的
//! 单测中；1e5 次 alloc/free/write/read 页内容影子模型在
//! `benches/arena_bench.rs` 的 `shadow_model` 中（超出清单 E2 的 100ms 上限）。

#[cfg(any(target_os = "linux", windows))]
use std::slice;
#[cfg(all(unix, target_pointer_width = "64"))]
use std::{env, process::Command};
use std::{error::Error as StdError, io, io::ErrorKind};

#[cfg(any(target_os = "linux", windows))]
use ldb_type::PageId;
use ldb_vmem::VirtualMemoryArena;

const PAGE: usize = 4096;

#[cfg(unix)]
fn host_page_size() -> usize {
  // SAFETY: sysconf 只读系统常量
  let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
  assert!(size > 0, "sysconf(_SC_PAGESIZE) 失败");
  size as usize
}

/// 逻辑页槽位步长：相邻两个合法页号指针之差
fn stride_of(arena: &VirtualMemoryArena) -> usize {
  arena.to_ptr(1).addr() - arena.to_ptr(0).addr()
}

/// 页字节视图
#[cfg(any(target_os = "linux", windows))]
fn page_slice(arena: &VirtualMemoryArena, pid: PageId) -> &[u8] {
  // SAFETY: pid 由调用方保证 < 页数；每个逻辑页独占槽位（stride >= 4KiB），
  // 单线程测试无并发写；arena 在借用期内存活。
  unsafe { slice::from_raw_parts(arena.to_ptr(pid), PAGE) }
}

/// 以 64 字节为周期写入 seed+i，便于检测错位
#[cfg(any(target_os = "linux", windows))]
fn fill_page(arena: &VirtualMemoryArena, pid: PageId, seed: u8) {
  // SAFETY: pid 由调用方保证 < 页数；每个逻辑页独占槽位（stride >= 4KiB），
  // 单线程测试无并发写；arena 在借用期内存活，&mut 不与其它视图重叠。
  let page = unsafe { slice::from_raw_parts_mut(arena.to_ptr(pid), PAGE) };
  for (i, chunk) in page.as_chunks_mut::<64>().0.iter_mut().enumerate() {
    chunk.fill(seed.wrapping_add(i as u8));
  }
}

/// 校验整页 4KiB 与 fill_page 的模式逐字节一致
#[cfg(any(target_os = "linux", windows))]
fn page_matches(arena: &VirtualMemoryArena, pid: PageId, seed: u8) -> bool {
  page_slice(arena, pid)
    .as_chunks::<64>()
    .0
    .iter()
    .enumerate()
    .all(|(i, chunk)| chunk.iter().all(|&b| b == seed.wrapping_add(i as u8)))
}

fn assert_invalid(result: io::Result<()>) {
  match result {
    Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
    Ok(()) => panic!("非法区间必须返回 InvalidInput"),
  }
}

/// 构造：向上取整、0 长度拒绝、溢出拒绝
#[test]
fn new_arena_rounding_and_errors() -> Result<(), Box<dyn StdError>> {
  let one = VirtualMemoryArena::new_arena(1)?;
  assert_eq!(one.len(), 4096);
  assert!(!one.is_empty());

  let two = VirtualMemoryArena::new_arena(4097)?;
  assert_eq!(two.len(), 8192);

  assert!(VirtualMemoryArena::new_arena(0).is_err());
  assert!(VirtualMemoryArena::new_arena(usize::MAX).is_err());
  Ok(())
}

/// pid → 指针 → pid 往返；槽位按主机页隔离；边界页号合法
#[test]
fn address_mapping_roundtrip() -> Result<(), Box<dyn StdError>> {
  const CNT: u64 = 1024;
  let arena = VirtualMemoryArena::new_arena(PAGE * CNT as usize)?;
  let stride = stride_of(&arena);
  assert!(stride >= PAGE);
  assert_eq!(stride & (stride - 1), 0, "stride 必须是 2 的幂");

  #[cfg(unix)]
  {
    let host = host_page_size();
    assert_eq!(stride, host.max(PAGE), "stride 必须等于主机页大小");
  }

  let pids = [0u64, 1, 2, 3, 7, 99, CNT - 1];
  for pid in pids {
    let p = arena.to_ptr(pid);
    assert!(arena.validate_pid(pid));
    assert!(arena.is_valid_ptr(p));
    assert_eq!(arena.to_pid(p), pid);
    assert_eq!(p.addr() % PAGE, 0);
    assert_eq!(p.addr() % stride, 0);
    // SAFETY: p 是合法槽位起点，PAGE-1 仍在逻辑页内
    let last_byte = unsafe { p.add(PAGE - 1) };
    assert!(arena.is_valid_ptr(last_byte));
    if stride > PAGE {
      // SAFETY: stride > PAGE 时 p+PAGE 仍在同一槽位的填充区
      let padding = unsafe { p.add(PAGE) };
      assert!(!arena.is_valid_ptr(padding), "槽位填充区不得视为合法页");
      assert_eq!(arena.to_pid(padding), pid, "填充区按槽位向下取整");
    }
  }

  assert!(arena.validate_pid(CNT - 1));
  assert!(!arena.validate_pid(CNT));
  assert!(!arena.validate_pid(u64::MAX));

  // SAFETY: 最后一个槽位起点 + stride 是整段预留的 one-past
  let one_past = unsafe { arena.to_ptr(CNT - 1).add(stride) };
  assert!(!arena.is_valid_ptr(one_past));
  // 低于基址的指针不得 panic，且必须被 validate_pid 拒绝
  let below = arena.to_ptr(0).wrapping_offset(-1);
  assert!(!arena.is_valid_ptr(below));
  assert!(!arena.validate_pid(arena.to_pid(below)));
  Ok(())
}

/// 物理帧换算必须按主机页槽位、向下取整
#[test]
fn frames_for_bytes_scales_by_host_stride() -> Result<(), Box<dyn StdError>> {
  let arena = VirtualMemoryArena::new_arena(PAGE * 8)?;
  let stride = stride_of(&arena) as u64;
  assert_eq!(VirtualMemoryArena::frames_for_bytes(stride * 8)?, 8);
  assert_eq!(VirtualMemoryArena::frames_for_bytes(stride * 8 - 1)?, 7);
  assert_eq!(VirtualMemoryArena::frames_for_bytes(0)?, 0);
  Ok(())
}

/// 驱逐/提交接口的 release 构建区间校验
#[test]
fn range_validation() -> Result<(), Box<dyn StdError>> {
  const CNT: usize = 8;
  let arena = VirtualMemoryArena::new_arena(PAGE * CNT)?;

  // 合法区间：提交 → 驱逐 → 解除提交 → 重新提交
  arena.commit_pages(0, PAGE)?;
  arena.advise_dontneed(0, PAGE)?;
  arena.decommit_pages(0, PAGE)?;
  arena.commit_pages(0, PAGE)?;
  arena.commit_pages(0, PAGE * CNT)?;
  arena.advise_dontneed(0, 0)?;
  arena.commit_pages(0, 0)?;
  arena.decommit_pages(0, 0)?;
  // 整段合法区间：end == logical_len 必须接受（上界是开区间）
  arena.advise_dontneed(0, PAGE * CNT)?;

  // 未对齐偏移 / 未对齐长度 / 越界 / 溢出
  assert_invalid(arena.advise_dontneed(1, PAGE));
  assert_invalid(arena.advise_dontneed(0, 1));
  assert_invalid(arena.advise_dontneed(PAGE * CNT, PAGE));
  assert_invalid(arena.advise_dontneed(usize::MAX & !(PAGE - 1), PAGE));

  assert_invalid(arena.commit_pages(1, PAGE));
  assert_invalid(arena.commit_pages(0, 1));
  assert_invalid(arena.commit_pages(PAGE * CNT, PAGE));
  assert_invalid(arena.commit_pages(usize::MAX & !(PAGE - 1), PAGE));

  assert_invalid(arena.decommit_pages(1, PAGE));
  assert_invalid(arena.decommit_pages(0, 1));
  assert_invalid(arena.decommit_pages(PAGE * CNT, PAGE));
  assert_invalid(arena.decommit_pages(usize::MAX & !(PAGE - 1), PAGE));
  Ok(())
}

/// 进程级 ENOMEM：子进程 `RLIMIT_AS` 限制下请求超大 arena，必须返回
/// `io::Error(OutOfMemory)` 而不是 abort/panic。
///
/// macOS 的 `setrlimit(RLIMIT_AS)` 只接受 >= 1 TiB 的值（实测 1 GiB/4 GiB/
/// 8 GiB 均 EINVAL），故子进程设 1 TiB、请求 2 TiB 逻辑长度；16KiB 主机页下
/// 实际预留 8 TiB，必然触发 ENOMEM。子进程自身返回成功即证明未 abort。
#[cfg(all(unix, target_pointer_width = "64"))]
#[test]
fn oversized_arena_returns_enomem_without_abort() -> Result<(), Box<dyn StdError>> {
  const CHILD_ENV: &str = "LDB_VMEM_ENOMEM_CHILD";

  if env::var_os(CHILD_ENV).is_some() {
    let limit = 1u64 << 40;
    let rlim = libc::rlimit {
      rlim_cur: limit as libc::rlim_t,
      rlim_max: limit as libc::rlim_t,
    };
    // SAFETY: setrlimit 只修改本进程资源上限，rlim 指向合法栈内存
    if unsafe { libc::setrlimit(libc::RLIMIT_AS, &rlim) } != 0 {
      return Err(
        format!(
          "setrlimit(RLIMIT_AS, 1TiB) 失败: {}",
          io::Error::last_os_error()
        )
        .into(),
      );
    }
    match VirtualMemoryArena::new_arena(2usize << 40) {
      Ok(arena) => panic!("1TiB RLIMIT_AS 下 2TiB arena 不应成功: len={}", arena.len()),
      Err(e) => assert_eq!(e.kind(), ErrorKind::OutOfMemory, "错误类型: {e}"),
    }
    return Ok(());
  }

  let exe = env::current_exe()?;
  let output = Command::new(exe)
    .args([
      "--exact",
      "oversized_arena_returns_enomem_without_abort",
      "--nocapture",
    ])
    .env(CHILD_ENV, "1")
    .output()?;
  let stdout = String::from_utf8_lossy(&output.stdout);
  let stderr = String::from_utf8_lossy(&output.stderr);
  assert!(
    stdout.contains("1 passed"),
    "子进程未实际执行用例: status={:?}\nstdout={stdout}\nstderr={stderr}",
    output.status
  );
  assert!(
    output.status.success(),
    "子进程异常退出: status={:?}\nstdout={stdout}\nstderr={stderr}",
    output.status
  );
  Ok(())
}

/// Linux：MADV_DONTNEED 立即零填充，且不波及相邻逻辑页。
#[cfg(target_os = "linux")]
#[test]
fn evict_zero_fills_linux() -> Result<(), Box<dyn StdError>> {
  let arena = VirtualMemoryArena::new_arena(PAGE * 4)?;
  fill_page(&arena, 0, 0x11);
  fill_page(&arena, 1, 0x21);
  fill_page(&arena, 2, 0x31);

  arena.advise_dontneed(PAGE, PAGE)?;

  assert!(page_matches(&arena, 0, 0x11), "左邻页被误驱逐");
  assert!(page_matches(&arena, 2, 0x31), "右邻页被误驱逐");
  assert!(
    page_slice(&arena, 1).iter().all(|&b| b == 0),
    "驱逐页未零填充"
  );

  fill_page(&arena, 1, 0x41);
  assert!(page_matches(&arena, 1, 0x41), "驱逐后重写读回失败");
  Ok(())
}

/// Windows：`new_arena` 只 reserve；`advise_dontneed` 必须把槽位退回 reserved。
#[cfg(windows)]
#[test]
fn new_arena_reserves_and_decommits_windows() -> Result<(), Box<dyn StdError>> {
  use std::mem::size_of;

  use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
  };

  fn page_state(addr: *const u8) -> Result<u32, Box<dyn StdError>> {
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: info 是可写结构；addr 来自 arena，按页查询
    let n = unsafe {
      VirtualQuery(
        addr.cast(),
        &mut info,
        size_of::<MEMORY_BASIC_INFORMATION>(),
      )
    };
    if n == 0 {
      return Err(io::Error::last_os_error().into());
    }
    Ok(info.State)
  }

  let arena = VirtualMemoryArena::new_arena(PAGE * 4)?;
  assert_eq!(
    page_state(arena.to_ptr(0))?,
    MEM_RESERVE,
    "new_arena 不得全量 commit"
  );
  arena.commit_pages(0, PAGE * 2)?;
  assert_eq!(
    page_state(arena.to_ptr(1))?,
    MEM_COMMIT,
    "commit_pages 未覆盖整段"
  );
  arena.advise_dontneed(0, PAGE)?;
  assert_eq!(
    page_state(arena.to_ptr(0))?,
    MEM_RESERVE,
    "advise_dontneed 未释放 commit charge"
  );
  assert_eq!(
    page_state(arena.to_ptr(1))?,
    MEM_COMMIT,
    "advise_dontneed 波及相邻槽位"
  );
  Ok(())
}

/// Windows：`advise_dontneed`/`decommit_pages` 释放提交，重新提交为零页。
#[cfg(windows)]
#[test]
fn commit_decommit_roundtrip_windows() -> Result<(), Box<dyn StdError>> {
  let arena = VirtualMemoryArena::new_arena(PAGE * 4)?;
  arena.commit_pages(0, PAGE * 2)?;
  fill_page(&arena, 0, 0x3C);
  fill_page(&arena, 1, 0x5A);
  assert!(page_matches(&arena, 0, 0x3C));
  assert!(page_matches(&arena, 1, 0x5A));

  arena.advise_dontneed(0, PAGE)?;
  arena.commit_pages(0, PAGE)?;
  assert!(
    page_slice(&arena, 0).iter().all(|&b| b == 0),
    "advise_dontneed 后重新提交未零填充"
  );

  arena.decommit_pages(PAGE, PAGE)?;
  arena.commit_pages(PAGE, PAGE)?;
  assert!(
    page_slice(&arena, 1).iter().all(|&b| b == 0),
    "decommit_pages 后重新提交未零填充"
  );
  Ok(())
}
