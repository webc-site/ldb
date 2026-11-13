//! 跨平台大虚拟地址空间 arena（上游 `ZLeanStore/src/buffer/buffer_manager.cc:173-277`
//! 的 vmem 预留重写）。
//!
//! 分平台矩阵（todo.md §3.2）：
//! | 平台 | 预留 | 大页建议 | 驱逐 |
//! | Linux | mmap(MAP_PRIVATE\|MAP_ANONYMOUS\|MAP_NORESERVE) | MADV_NOHUGEPAGE | madvise(MADV_DONTNEED) |
//! | macOS | mmap(MAP_ANON) | 无 | madvise(MADV_FREE_REUSABLE)（惰性回收） |
//! | Windows | VirtualAlloc(MEM_RESERVE) | 无 | MEM_DECOMMIT（释放物理页与 commit charge） |
//!
//! 主机页 > 4KiB 时，每个 4KiB 逻辑页独占一个主机页槽位
//! （`stride = max(主机页大小, 4KiB)`）。原因：madvise 以主机页为粒度，
//! Linux `madvise(2)` 明确 “addr must be page-aligned” 且 “size is rounded up
//! to a multiple of page size”；Darwin 实测对 16KiB 主机页内的 4KiB 子区间调用
//! MADV_ZERO 会清零整个 16KiB。若 4 个逻辑页共享一个主机页，单独驱逐其中一页
//! 会静默丢弃其余 3 页。代价是 VA 与每帧物理内存 ×(主机页/4KiB)
//! （Apple Silicon 16KiB → 4×），上层须按 OS 主机页大小 / 4KiB 缩放
//! `physical_frames`。x86_64 4KiB 主机页下 stride == 4KiB，与上游逐字节一致。
//!
//! 物理内存预算换算（上层 `ldb_buf` / `ldb` 配置 `physical_frames` 时使用）：
//! `rss_bytes = physical_frames * stride`，预算 `B` 字节时应配置
//! `physical_frames = B / stride`（向下取整）。在 16KiB 主机页的 Apple Silicon
//! 上，8192 帧的满池 RSS = 8192 * 16KiB = **128 MiB**（4KiB 主机页为 32 MiB）；
//! 实测见 `cargo bench -p ldb_vmem` 的 `touch pages=8192` 行（`rss_delta`）。
//!
//! Windows 提交契约：`new_arena` 只做 `MEM_RESERVE`，不产生 commit charge；
//! 调用方在首次访问任一合法槽位前必须对该区间调用 [`VirtualMemoryArena::commit_pages`]。
//! `advise_dontneed`/`decommit_pages` 用 `MEM_DECOMMIT` 释放物理页与 commit charge，
//! 槽位回到 reserved；再次访问前必须重新 `commit_pages`。这样 commit charge 只覆盖
//! 当前已提交的槽位，而不是 `virtual_size`。上游默认 `bm_virtual_gb=64`
//! （`ZLeanStore/src/config.cc:22`），若在构造期 `MEM_COMMIT` 全部 64 GiB，按
//! Microsoft `VirtualAlloc` 文档会一次性计入 64 GiB commit charge（RAM + paging file
//! 的提交额度），普通 VPS 的默认 CommitLimit 无法保证，且违背 todo.md §3.2 的
//! “Windows 按需 MEM_COMMIT”。因此不在构造期全量提交。
//!
//! 实测（Wine 8.0.1，宿主 64 GiB）：64 GiB reserve 构造约 3.8 ms；Wine 不执行
//! 真实 CommitLimit 检查，且其 `PROCESS_MEMORY_COUNTERS.PagefileUsage` 对
//! reserve 也记账，不能作为 commit charge 证据；真机额度以 Microsoft 文档为准。
//!
//! overcommit 说明（Linux）：
//! - `MAP_NORESERVE` 只在 `vm.overcommit_memory != 2` 时生效；`=2` 时内核在
//!   `do_mmap()` 中不设置 `VMA_NORESERVE`，映射仍按 `CommitLimit` 记账
//!   （linux `mm/mmap.c`），大 arena 仍会 `ENOMEM`。本层把 `mmap` 失败原样
//!   返回 `io::Error`，不 abort。
//! - `ulimit -v`（RLIMIT_AS）按虚拟地址空间计费，`mmap` 超限返回 `ENOMEM`。
//! - cgroup 内存在首次触页时计费，`MAP_NORESERVE` 不能绕过；触页超过限额会
//!   被 OOM kill（SIGKILL，非可捕获错误）。上层 `physical_cnt` 必须低于配额。
//!
//! 上游缺陷修复：C-8（MADV_HUGEPAGE Linux-only）、C-9（blob 开启时 2× 预留）。

#[cfg(windows)]
use std::ptr::null;
#[cfg(unix)]
use std::ptr::null_mut;
use std::{
  io,
  io::{Error, ErrorKind},
  ptr::NonNull,
};

use ldb_type::{PAGE_SIZE, PageId};
#[cfg(windows)]
use windows_sys::Win32::System::{
  Memory::{
    MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE,
    VirtualAlloc, VirtualFree,
  },
  SystemInformation::{GetSystemInfo, SYSTEM_INFO},
};

#[cfg(not(any(unix, windows)))]
compile_error!("ldb_vmem 仅支持 unix / windows 平台");

/// 虚拟内存 arena
///
/// 逻辑页大小固定为 4KiB（[`ldb_type::PAGE_SIZE`]）；每个逻辑页的虚拟槽位
/// 步长为 `stride = max(主机页大小, 4KiB)`，保证单页 madvise 不会波及相邻
/// 逻辑页。
#[derive(Debug)]
pub struct VirtualMemoryArena {
  /// 数据区起始地址（已跳过平台预留头；Windows 为 reserve 基址）
  ///
  /// 保存带 provenance 的指针而非 `usize`：地址换算用 `wrapping_add`
  /// （严格 provenance，Miri `-Zmiri-strict-provenance` 可通过），不经过
  /// 整数往返。
  base: NonNull<u8>,
  /// 逻辑页数
  page_cnt: usize,
  /// 逻辑字节数 = page_cnt * PAGE_SIZE
  logical_len: usize,
  /// 逻辑页槽位步长（2 的幂）
  stride: usize,
  /// log2(stride)，地址换算用位移避免运行时除法
  stride_shift: u32,
  /// 实际保留的虚拟地址字节数 = page_cnt * stride
  reserved_len: usize,
}

// SAFETY: arena 只保存地址与长度；所有可写内存访问都经 `to_ptr` 返回的裸指针，
// 由调用方按页闩协议保证别名与生命周期。`advise_dontneed(&self)` 只向 OS 发
// 页回收/解除提交调用，不通过安全引用读写页内容。
unsafe impl Send for VirtualMemoryArena {}
// SAFETY: 同 Send；`&Self` 上只有纯算术查询与 OS 建议调用。
unsafe impl Sync for VirtualMemoryArena {}

impl VirtualMemoryArena {
  /// 预留 `len` 字节逻辑地址空间（不提交物理页）
  ///
  /// `len` 向上取整到 4KiB；[`Self::len`] 返回取整后的逻辑长度。
  /// 实际保留的虚拟地址为 `page_cnt * stride`。
  ///
  /// Windows 的 `new_arena` 只 `MEM_RESERVE`：调用方必须先对目标区间调用
  /// [`Self::commit_pages`] 才能读写；`advise_dontneed`/`decommit_pages` 之后
  /// 同样需要重新提交。Linux/macOS 匿名页在首次触页时惰性提交。
  ///
  /// # Errors
  /// - `len == 0`、长度/VA 预留溢出、主机页大小非 2 的幂：`InvalidInput`
  /// - `mmap` / `VirtualAlloc(MEM_RESERVE)` 失败（地址空间耗尽、`RLIMIT_AS`、
  ///   `vm.overcommit_memory=2` 的 `CommitLimit` 不足）：对应 `io::Error`
  pub fn new_arena(len: usize) -> io::Result<Self> {
    let host_page = host_page_size()?;
    Self::new_arena_with_host_page(len, host_page)
  }

  /// 按物理预算字节数计算不超预算的常驻帧数（向下取整）
  ///
  /// 公式：`frames = bytes / stride`，`stride = max(host_page_size, 4KiB)`。
  /// 冷路径（每次重新探测主机页大小，不缓存）；供 `ldb_buf` / `ldb` 在
  /// 16KiB/64KiB 主机页上换算 `physical_frames`，避免按 4KiB 配置导致
  /// RSS 被放大 `host_page / 4KiB` 倍。
  ///
  /// # Errors
  /// 主机页大小探测失败，或返回值不是 2 的幂：`io::Error`
  pub fn frames_for_bytes(bytes: u64) -> io::Result<u64> {
    let stride = stride_for_host_page(host_page_size()?)?;
    Ok(bytes / stride as u64)
  }

  /// 按指定主机页大小构造（单测用于覆盖 4KiB/16KiB/64KiB stride；
  /// 生产路径经 [`Self::new_arena`] 读取 OS 页大小）
  fn new_arena_with_host_page(len: usize, host_page: usize) -> io::Result<Self> {
    let page_cnt = len
      .checked_add(PAGE_SIZE - 1)
      .ok_or_else(|| invalid("arena 长度向上取整溢出"))?
      / PAGE_SIZE;
    if page_cnt == 0 {
      return Err(invalid("arena 长度不得为 0"));
    }
    let logical_len = page_cnt
      .checked_mul(PAGE_SIZE)
      .ok_or_else(|| invalid("arena 逻辑长度溢出"))?;
    let stride = stride_for_host_page(host_page)?;
    let reserved_len = page_cnt
      .checked_mul(stride)
      .ok_or_else(|| invalid("arena 虚拟地址预留溢出"))?;
    // SAFETY: reserved_len > 0；reserve 只会返回有效映射或 Err
    let base = unsafe { reserve(reserved_len)? };
    Ok(Self {
      base,
      page_cnt,
      logical_len,
      stride,
      stride_shift: stride.trailing_zeros(),
      reserved_len,
    })
  }

  /// 页号 → 指针（O(1) 位移）
  ///
  /// 调用方须先以 [`Self::validate_pid`] 校验页号；debug 构建断言。
  /// 指针保留 arena 映射的 provenance（`wrapping_add`，越界也不产生 UB，
  /// 解引用前须由调用方校验）。Windows 上槽位初始为 reserved，解引用前
  /// 必须先 `commit_pages` 且未被 `advise_dontneed`/`decommit_pages` 解除提交。
  #[inline]
  pub const fn to_ptr(&self, pid: PageId) -> *mut u8 {
    debug_assert!(self.validate_pid(pid));
    self
      .base
      .as_ptr()
      .wrapping_add((pid as usize) << self.stride_shift)
  }

  /// 指针 → 页号（O(1) 位移）
  ///
  /// 指针低于 arena 基址或落在槽位填充区时，按槽位向下取整返回页号；
  /// 不 panic。越界结果由 [`Self::validate_pid`] / [`Self::is_valid_ptr`] 判定。
  #[inline]
  pub fn to_pid(&self, ptr: *const u8) -> PageId {
    let off = ptr.addr().wrapping_sub(self.base.as_ptr().addr());
    (off >> self.stride_shift) as PageId
  }

  /// 指针是否落在某个逻辑页内（不含槽位填充区）
  ///
  /// 只检查地址范围与槽位，不检查 Windows 的提交状态；reserved 槽位仍返回
  /// `true`，解引用前必须由调用方保证已 `commit_pages`。
  #[inline]
  pub fn is_valid_ptr(&self, ptr: *const u8) -> bool {
    let off = ptr.addr().wrapping_sub(self.base.as_ptr().addr());
    off < self.reserved_len && (off & (self.stride - 1)) < PAGE_SIZE
  }

  /// 页号是否合法
  #[inline]
  pub const fn validate_pid(&self, pid: PageId) -> bool {
    pid < self.page_cnt as PageId
  }

  /// 提交 `[off, off+len)` 物理页（Windows MEM_COMMIT；Unix 匿名页惰性提交）
  ///
  /// `off`/`len` 必须 4KiB 对齐且在逻辑长度内（release 构建也检查）。
  /// Unix 返回 `Ok(())` 且不产生系统调用；Windows 上 `new_arena` 后、首次访问前
  /// 必须调用本方法，`advise_dontneed`/`decommit_pages` 后也必须重新调用。
  /// 对已提交区间幂等；每个逻辑页槽位单独提交（`stride == 4KiB` 时合并为一次）。
  ///
  /// # Errors
  /// 区间未对齐/越界/溢出：`InvalidInput`；OS 提交失败：对应 `io::Error`
  pub fn commit_pages(&self, off: usize, len: usize) -> io::Result<()> {
    self.checked_range(off, len)?;
    #[cfg(unix)]
    {
      // Unix 匿名页无显式提交；直接返回，避免整段 arena 的 O(页数) 空循环
      Ok(())
    }
    #[cfg(windows)]
    {
      if len == 0 {
        return Ok(());
      }
      for_each_slot(self, off, len, |addr, slot_len| {
        // SAFETY: checked_range 保证区间在 arena 内；addr/slot_len 为对齐槽位
        unsafe { commit(addr, slot_len) }
      })
    }
  }

  /// 解除提交物理页（Windows MEM_DECOMMIT；Unix 无显式提交，返回 Ok）
  ///
  /// `off`/`len` 必须 4KiB 对齐且在逻辑长度内（release 构建也检查）。
  /// Unix 上释放物理页请用 [`Self::advise_dontneed`]；Windows 上解除提交会释放
  /// 物理页与 commit charge，槽位回到 reserved，再次访问前必须调用
  /// [`Self::commit_pages`]。
  ///
  /// # Errors
  /// 区间未对齐/越界/溢出：`InvalidInput`；OS 解除提交失败：对应 `io::Error`
  pub fn decommit_pages(&self, off: usize, len: usize) -> io::Result<()> {
    self.checked_range(off, len)?;
    #[cfg(unix)]
    {
      // Unix 不解除提交；释放物理页请用 advise_dontneed
      Ok(())
    }
    #[cfg(windows)]
    {
      if len == 0 {
        return Ok(());
      }
      for_each_slot(self, off, len, |addr, slot_len| {
        // SAFETY: checked_range 保证区间在 arena 内；addr/slot_len 为对齐槽位
        unsafe { decommit(addr, slot_len) }
      })
    }
  }

  /// 驱逐物理页帧
  ///
  /// Linux `MADV_DONTNEED` 立即丢弃并零填充；macOS `MADV_FREE_REUSABLE`
  /// 标记可回收（惰性，RSS 不保证立即下降）；Windows `MEM_DECOMMIT`
  /// 释放物理页与 commit charge，槽位回到 reserved，再次访问前必须
  /// 调用 [`Self::commit_pages`]。
  ///
  /// `off`/`len` 必须 4KiB 对齐且在逻辑长度内（release 构建也检查）。
  /// `stride == 4KiB` 时整段一次系统调用；主机页更大时每个逻辑页一次
  /// （每页独占槽位，调用方本身也是逐页驱逐）。
  ///
  /// # Errors
  /// 区间未对齐/越界/溢出：`InvalidInput`；OS 调用失败：对应 `io::Error`
  pub fn advise_dontneed(&self, off: usize, len: usize) -> io::Result<()> {
    self.checked_range(off, len)?;
    if len == 0 {
      return Ok(());
    }
    for_each_slot(self, off, len, |addr, slot_len| {
      // SAFETY: checked_range 保证区间在 arena 内；addr/slot_len 按主机页对齐
      unsafe { release_pages(addr, slot_len) }
    })
  }

  /// arena 逻辑字节长度（不含槽位填充）
  #[inline]
  pub fn len(&self) -> usize {
    self.logical_len
  }

  /// 是否为空 arena
  ///
  /// 构造器拒绝 0 长度，故恒为 `false`；保留以满足 `len`/`is_empty` 成对约定。
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.logical_len == 0
  }

  /// 校验 `[off, off+len)`：4KiB 对齐、无溢出、逻辑长度内
  fn checked_range(&self, off: usize, len: usize) -> io::Result<()> {
    if !off.is_multiple_of(PAGE_SIZE) || !len.is_multiple_of(PAGE_SIZE) {
      return Err(invalid("区间偏移/长度必须按 4KiB 页对齐"));
    }
    let end = off
      .checked_add(len)
      .ok_or_else(|| invalid("区间结束地址溢出"))?;
    if end > self.logical_len {
      return Err(invalid("区间超出 arena 逻辑长度"));
    }
    Ok(())
  }
}

impl Drop for VirtualMemoryArena {
  fn drop(&mut self) {
    // SAFETY: base/reserved_len 来自 reserve 的成功返回，Drop 只执行一次
    unsafe {
      let _ = release(self.base, self.reserved_len);
    }
  }
}

/// 对 `[off, off+len)` 覆盖的每个逻辑页槽位调用 `f(addr, slot_len)`
///
/// `stride == 4KiB` 时整段合并为一次调用；否则逐逻辑页调用（槽位不连续）。
#[inline]
fn for_each_slot(
  arena: &VirtualMemoryArena,
  off: usize,
  len: usize,
  mut f: impl FnMut(*mut u8, usize) -> io::Result<()>,
) -> io::Result<()> {
  if arena.stride == PAGE_SIZE {
    return f(arena.base.as_ptr().wrapping_add(off), len);
  }
  let first = off / PAGE_SIZE;
  let cnt = len / PAGE_SIZE;
  for pid in first..first + cnt {
    let addr = arena.base.as_ptr().wrapping_add(pid << arena.stride_shift);
    f(addr, arena.stride)?;
  }
  Ok(())
}

/// 校验主机页大小并返回槽位步长
fn stride_for_host_page(host_page: usize) -> io::Result<usize> {
  if !host_page.is_power_of_two() {
    return Err(invalid("主机页大小必须是 2 的幂"));
  }
  Ok(host_page.max(PAGE_SIZE))
}

/// 主机页大小（字节）
///
/// 使用 POSIX `sysconf(_SC_PAGESIZE)`（而非非标准 `getpagesize()`）：三者
/// 在 Linux/macOS 上返回同一值，`sysconf` 可移植且失败可带 errno。
/// 不加 `OnceLock` 缓存：仅在冷路径 `new_arena`/`frames_for_bytes` 调用；
/// 本机实测 sysconf 约 3.1 ns/次，libc 已缓存页大小常量。
#[cfg(unix)]
fn host_page_size() -> io::Result<usize> {
  // SAFETY: sysconf 只读系统常量，无指针参数
  let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
  if size <= 0 {
    return Err(Error::last_os_error());
  }
  let size = size as usize;
  if !size.is_power_of_two() {
    return Err(invalid("sysconf(_SC_PAGESIZE) 返回非 2 的幂"));
  }
  Ok(size)
}

/// 主机页大小（字节）
///
/// `GetSystemInfo` 是冷路径；Windows 页大小进程生命周期内不变，不额外缓存。
#[cfg(windows)]
fn host_page_size() -> io::Result<usize> {
  let mut info = SYSTEM_INFO::default();
  // SAFETY: GetSystemInfo 写满 SYSTEM_INFO，无别名
  unsafe { GetSystemInfo(&mut info) };
  let size = info.dwPageSize as usize;
  if !size.is_power_of_two() {
    return Err(invalid("GetSystemInfo 返回非 2 的幂页大小"));
  }
  Ok(size)
}

#[inline]
fn invalid(msg: &'static str) -> Error {
  Error::new(ErrorKind::InvalidInput, msg)
}

/// Linux：mmap + MAP_NORESERVE；macOS：无 NORESERVE；均不提交物理页
#[cfg(unix)]
unsafe fn reserve(len: usize) -> io::Result<NonNull<u8>> {
  use libc::{MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, mmap};

  #[cfg(target_os = "linux")]
  let flags = MAP_PRIVATE | MAP_ANONYMOUS | libc::MAP_NORESERVE;
  #[cfg(not(target_os = "linux"))]
  let flags = MAP_PRIVATE | MAP_ANONYMOUS;

  // SAFETY: len 非 0 且为页大小的整数倍；失败返回 MAP_FAILED
  let p = unsafe {
    mmap(
      null_mut(),
      len,
      libc::PROT_READ | libc::PROT_WRITE,
      flags,
      -1,
      0,
    )
  };
  if p == MAP_FAILED {
    return Err(Error::last_os_error());
  }
  #[cfg(target_os = "linux")]
  {
    // 上游 Construction:225 对虚拟池使用 MADV_NOHUGEPAGE；老内核/THP 关闭时
    // 可能 EINVAL，属可选建议，失败不影响映射语义。
    // SAFETY: p/len 是刚建立的映射，可按页对齐使用
    let _ = unsafe { libc::madvise(p, len, libc::MADV_NOHUGEPAGE) };
  }
  // SAFETY: p 不是 MAP_FAILED；mmap_min_addr=0 的极端配置下可能为 0，
  // 此时释放映射并按失败处理（裸指针 0 不可解引用）。
  match NonNull::new(p.cast::<u8>()) {
    Some(base) => Ok(base),
    None => {
      // SAFETY: p/len 是刚建立的映射（p 为 0 时 munmap 返回 EINVAL，无害）
      let _ = unsafe { libc::munmap(p, len) };
      Err(Error::new(
        ErrorKind::OutOfMemory,
        "虚拟地址空间预留返回空指针",
      ))
    }
  }
}

/// Unix：munmap 整个预留区
#[cfg(unix)]
unsafe fn release(base: NonNull<u8>, len: usize) -> io::Result<()> {
  // SAFETY: base/len 来自 reserve，调用方保证只释放一次
  if unsafe { libc::munmap(base.as_ptr().cast(), len) } != 0 {
    return Err(Error::last_os_error());
  }
  Ok(())
}

/// Linux：立即丢弃并零填充
#[cfg(all(unix, target_os = "linux"))]
unsafe fn release_pages(addr: *mut u8, len: usize) -> io::Result<()> {
  // SAFETY: addr/len 由 checked_range 保证在 arena 内且按主机页对齐
  if unsafe { libc::madvise(addr.cast(), len, libc::MADV_DONTNEED) } != 0 {
    return Err(Error::last_os_error());
  }
  Ok(())
}

/// Darwin：标记页可被任意方复用；内核延迟回收，下次访问零填充。
///
/// 实测（macOS 27 / arm64 / 16KiB 主机页）：`MADV_DONTNEED` 对匿名页是空操作，
/// 内容与 RSS 均不变；`MADV_ZERO` 会清零整个 16KiB 主机页（即使只传 4KiB 子区间），
/// 故不能用于逐 4KiB 页驱逐。`MADV_FREE_REUSABLE` 不立即丢内容，但把页交还内核
/// 回收池，是 Darwin 上可用的惰性回收原语。
#[cfg(all(unix, target_os = "macos"))]
unsafe fn release_pages(addr: *mut u8, len: usize) -> io::Result<()> {
  // SAFETY: addr/len 由 checked_range 保证在 arena 内且按主机页对齐
  if unsafe { libc::madvise(addr.cast(), len, libc::MADV_FREE_REUSABLE) } != 0 {
    return Err(Error::last_os_error());
  }
  Ok(())
}

/// 其他 Unix：按 POSIX MADV_DONTNEED 尽力回收
#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
unsafe fn release_pages(addr: *mut u8, len: usize) -> io::Result<()> {
  // SAFETY: addr/len 由 checked_range 保证在 arena 内且按主机页对齐
  if unsafe { libc::madvise(addr.cast(), len, libc::MADV_DONTNEED) } != 0 {
    return Err(Error::last_os_error());
  }
  Ok(())
}

/// Windows：仅预留虚拟地址，不占 commit charge
#[cfg(windows)]
unsafe fn reserve(len: usize) -> io::Result<NonNull<u8>> {
  // SAFETY: VirtualAlloc 仅预留；失败返回 null
  let p = unsafe { VirtualAlloc(null(), len, MEM_RESERVE, PAGE_NOACCESS) };
  // SAFETY: 同 reserve 的契约；p 非空才构造 NonNull
  NonNull::new(p.cast::<u8>()).ok_or_else(Error::last_os_error)
}

/// Windows：MEM_RELEASE 要求 dwSize == 0
#[cfg(windows)]
unsafe fn release(base: NonNull<u8>, _len: usize) -> io::Result<()> {
  // SAFETY: base 来自 reserve；MEM_RELEASE 释放整个预留区
  if unsafe { VirtualFree(base.as_ptr().cast(), 0, MEM_RELEASE) } == 0 {
    return Err(Error::last_os_error());
  }
  Ok(())
}

/// Windows：提交区间（commit charge；物理页仍在首次触页时分配）
#[cfg(windows)]
unsafe fn commit(addr: *mut u8, len: usize) -> io::Result<()> {
  // SAFETY: addr 在已 reserve 区间内；失败返回 null
  let p = unsafe { VirtualAlloc(addr.cast(), len, MEM_COMMIT, PAGE_READWRITE) };
  if p.is_null() {
    return Err(Error::last_os_error());
  }
  Ok(())
}

/// Windows：解除提交，页面回到 reserved（后续访问需先 commit）
#[cfg(windows)]
unsafe fn decommit(addr: *mut u8, len: usize) -> io::Result<()> {
  // SAFETY: addr 在预留区内；MEM_DECOMMIT 可作用于未提交页
  if unsafe { VirtualFree(addr.cast(), len, MEM_DECOMMIT) } == 0 {
    return Err(Error::last_os_error());
  }
  Ok(())
}

/// Windows：释放物理页与 commit charge（槽位回到 reserved）
///
/// 使用 `VirtualFree(MEM_DECOMMIT)` 而非 `MEM_RESET`：后者保持提交，
/// commit charge 会随触页累积到整个 virtual_size，失去 VMCache 超-commit。
/// `MEM_DECOMMIT` 对未提交页不失败（Microsoft `VirtualFree` 文档）。
#[cfg(windows)]
unsafe fn release_pages(addr: *mut u8, len: usize) -> io::Result<()> {
  // SAFETY: addr 在预留区内；MEM_DECOMMIT 可作用于未提交页
  if unsafe { VirtualFree(addr.cast(), len, MEM_DECOMMIT) } == 0 {
    return Err(Error::last_os_error());
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  /// xorshift64：仅测试用的确定性随机源
  fn next_rand(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
  }

  /// 影子模型：1e5 个随机 pid 在 4KiB / 16KiB / 64KiB stride 下往返
  /// （to_ptr → to_pid、槽位填充区、相邻槽位边界、页内末字节）
  #[test]
  fn shadow_roundtrip_all_strides() {
    const CNT: usize = 4096;
    const OPS: usize = 100_000;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for host_page in [PAGE_SIZE, 16 * 1024, 64 * 1024] {
      let arena = match VirtualMemoryArena::new_arena_with_host_page(PAGE_SIZE * CNT, host_page) {
        Ok(a) => a,
        Err(e) => panic!("host_page={host_page} arena 预留失败: {e}"),
      };
      let base_addr = arena.to_ptr(0).addr();
      let stride = arena.to_ptr(1).addr() - base_addr;
      assert_eq!(stride, host_page.max(PAGE_SIZE));
      assert_eq!(arena.stride, stride);

      for pid in [0, 1, 2, 3, 7, 99, CNT - 1] {
        let p = arena.to_ptr(pid as PageId);
        // 槽位地址相对基址按 stride 对齐；绝对地址的 stride 对齐由 OS 保证
        // （真实主机页下 mmap 返回主机页对齐地址；本测试的 64KiB 强制 stride
        // 在 16KiB 主机页上只验证算术，不调用 madvise）。
        assert_eq!(
          (p.addr() - base_addr) % stride,
          0,
          "stride={stride} pid={pid}"
        );
        assert!(arena.is_valid_ptr(p));
        assert_eq!(arena.to_pid(p), pid as PageId);
      }

      for _ in 0..OPS {
        let pid = (next_rand(&mut rng) as usize) % CNT;
        let p = arena.to_ptr(pid as PageId);
        assert_eq!(arena.to_pid(p), pid as PageId, "stride={stride} pid={pid}");
        assert!(arena.is_valid_ptr(p), "stride={stride} pid={pid}");
        // SAFETY: pid < CNT，p+PAGE-1 仍在逻辑页内
        let last = unsafe { p.add(PAGE_SIZE - 1) };
        assert_eq!(arena.to_pid(last), pid as PageId);
        assert!(arena.is_valid_ptr(last));
        if stride > PAGE_SIZE {
          // SAFETY: p+PAGE 仍在同一槽位的填充区
          let pad = unsafe { p.add(PAGE_SIZE) };
          assert!(!arena.is_valid_ptr(pad), "槽位填充区不得合法");
          assert_eq!(arena.to_pid(pad), pid as PageId);
        }
        if pid + 1 < CNT {
          // SAFETY: 下一槽位起点仍在预留区内
          let next = unsafe { p.add(stride) };
          assert_eq!(arena.to_pid(next), (pid + 1) as PageId);
          assert!(arena.is_valid_ptr(next));
        }
        if pid > 0 {
          // SAFETY: 上一字节在预留区内
          let prev = unsafe { p.sub(1) };
          assert_eq!(arena.to_pid(prev), (pid - 1) as PageId);
        }
      }
    }
  }

  /// 主机页非 2 的幂必须拒绝（stride 位移要求），且错误信息可读
  #[test]
  fn host_page_must_be_power_of_two() {
    for host_page in [0, 4097, 12 * 1024, 16 * 1024 + 1, 64 * 1024 - 1] {
      match VirtualMemoryArena::new_arena_with_host_page(PAGE_SIZE, host_page) {
        Ok(_) => panic!("非 2 的幂主机页 {host_page} 必须拒绝"),
        Err(e) => {
          assert_eq!(e.kind(), ErrorKind::InvalidInput, "host_page={host_page}");
          assert!(
            e.to_string().contains("2 的幂"),
            "host_page={host_page} 错误不可读: {e}"
          );
        }
      }
    }
  }

  /// VA 预留长度溢出（page_cnt * stride）必须拒绝，不得回绕后分配过小 arena
  #[test]
  fn reserved_len_overflow_rejected() {
    // page_cnt = 2^52-1；stride=16KiB 时 page_cnt*stride 溢出 usize
    let len = usize::MAX - (PAGE_SIZE - 1);
    match VirtualMemoryArena::new_arena_with_host_page(len, 16 * 1024) {
      Ok(_) => panic!("VA 预留溢出必须拒绝"),
      Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidInput),
    }
  }
}
