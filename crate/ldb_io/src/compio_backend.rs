//! compio 跨平台异步文件后端（上游 `io_backend.cc`/`device.cc` 与
//! `leanstore.latency/src/recovery/log_manager.cc:26` 的重写）。
//!
//! - Direct I/O 自动回退：Linux 先试 `O_DIRECT`（tmpfs/overlayfs 上 `open` 返回
//!   EINVAL 即回退；打开成功但 I/O 仍返回 EINVAL/EFAULT/EOPNOTSUPP 时同样回退），
//!   macOS `fcntl(F_NOCACHE, 1)`，Windows `FILE_FLAG_NO_BUFFERING`。
//!   **对齐约束**：Linux/Windows 的 Direct I/O 要求缓冲区地址、文件偏移与长度
//!   均按逻辑块对齐（本实现取 `PAGE_SIZE`，覆盖 512e/4Kn）。不满足时整文件降级
//!   为标准缓存 I/O（一次性、永久），而不是把非对齐请求交给 `write(2)` 撞 EINVAL。
//!   macOS `F_NOCACHE` 只旁路页缓存、无对齐要求，故不做降级。
//! - 文件打孔回收：Linux `fallocate(PUNCH_HOLE|KEEP_SIZE)`；macOS `F_PUNCHHOLE`
//!   （`libc::fpunchhole_t`，含 8 字节头，不能用两字段的自定义结构体）；
//!   Windows `FSCTL_SET_ZERO_DATA` 待补。
//! - 独立文件布局：调用方按 `data.db`/`wal.log`/`dwb.dat` 分文件打开（替代上游
//!   单 fd 硬编码 64GiB+10GiB+64MiB 偏移，todo.md §9 C-6）。
//! - `open_buffered` 显式跳过 Direct I/O 探测：DWB 记录 4160 B、WAL 任意字节偏移
//!   都不满足 4 KiB 对齐，直接缓冲打开可省掉「O_DIRECT open + 首次 I/O 降级」。
//! - 新建文件后 fsync 父目录（Unix），保证目录项持久化；见 [`sync_parent_dir`]。
//! - [`CompioFileBackend::sync_durable`] 是崩溃一致性要求的屏障：Linux `fdatasync`，
//!   macOS `F_FULLFSYNC`（Darwin 的 fsync 不保证设备缓存落盘）。

use std::{
  io,
  mem::replace,
  path::{Path, PathBuf},
  sync::{
    RwLock,
    atomic::{AtomicBool, Ordering},
  },
};

#[cfg(unix)]
use compio::runtime::{JoinError, spawn_blocking};
use compio::{
  buf::{BufResult, IoBuf, IoBufExt, IoBufMut, IoBufMutExt},
  fs::File,
  io::{AsyncReadAtExt, AsyncWriteAtExt},
};
use ldb_type::PAGE_SIZE;

/// Direct I/O 对齐粒度。
///
/// Linux `O_DIRECT` 要求地址/偏移/长度按逻辑块对齐（512e 盘为 512、4Kn 盘为
/// 4096）；统一取 4 KiB 可同时覆盖两者。Windows `FILE_FLAG_NO_BUFFERING`
/// 要求扇区对齐。macOS `F_NOCACHE` 无对齐要求。
const DIRECT_ALIGN: usize = if cfg!(any(target_os = "linux", windows)) {
  PAGE_SIZE
} else {
  1
};

/// compio 异步文件后端。
///
/// 内部 fd 用 `RwLock<File>` 保护：正常路径只取读锁克隆 `File`（compio 的
/// `SharedFd` 共享引用，克隆不复制 fd），Direct I/O 无法满足对齐时在写锁下一次性
/// 替换为缓冲句柄。锁不跨 await 持有。
///
/// # 并发契约
///
/// compio 默认 `unsync` 配置下 `File` 的共享引用是 `Rc`，因此本类型
/// **`!Send + !Sync`**，只能在创建它的 runtime/线程内使用（thread-per-core）。
/// 同一 runtime 内不同偏移的请求可并发；同一文件偏移的并发读写必须由调用方
/// 串行化（L3 页锁）。`degrade` 只影响**之后**的请求：在途请求各自持有旧 `File`
/// 克隆（`SharedFd` 共享引用），旧 fd 在请求完成前不会被关闭；单次请求若在部分
/// 传输后收到 EINVAL，会用已降级的 fd 从原偏移重写**整个** buffer，成功返回时页
/// 内容不会混合。因此「同一页并发写 + degrade」的混合语义不在本后端的保证范围。
///
/// 「多 worker 共享一个 backend」必须在编译期失败（而不是运行期 UB）：
///
/// ```compile_fail
/// fn require_send_sync<T: Send + Sync>() {}
/// require_send_sync::<ldb_io::CompioFileBackend>();
/// ```
#[derive(Debug)]
pub struct CompioFileBackend {
  path: PathBuf,
  file: RwLock<File>,
  /// 当前句柄是否处于 Direct I/O（O_DIRECT / F_NOCACHE / NO_BUFFERING）模式
  direct: AtomicBool,
  /// 本后端声明的 Direct I/O 对齐粒度（测试可注入 4096 以覆盖 Linux 路径）
  direct_align: usize,
}

impl CompioFileBackend {
  /// 打开（必要时创建）存储文件，优先 Direct I/O、失败自动回退缓冲 I/O。
  ///
  /// 新建文件时会 fsync 父目录（Unix），见 [`sync_parent_dir`]。
  pub async fn open_direct_or_buffered(path: impl AsRef<Path>) -> io::Result<Self> {
    Self::open_inner(path.as_ref(), true).await
  }

  /// 打开（必要时创建）存储文件，**不做** Direct I/O 探测，始终缓冲 I/O。
  ///
  /// 适用于偏移/长度不满足 4 KiB 对齐的文件：
  /// - `dwb.dat`：记录 = 64 B 头 + 4096 B 页 = 4160 B，槽偏移 `base + n*4160`
  ///   既非 4 KiB 对齐、长度也不是 4 KiB 倍数（见 `doublewrite` 模块文档的
  ///   布局决策）；
  /// - `wal.log`：WAL 记录按任意字节偏移追加。
  ///
  /// 这些文件用 `open_direct_or_buffered` 也能工作，但会先成功 `O_DIRECT` open、
  /// 在首次非对齐 I/O 时再降级，多一次 open 且语义不够直白。`data.db` 等整页
  /// 访问仍应使用 [`open_direct_or_buffered`](Self::open_direct_or_buffered)
  /// 以保留 Direct I/O。
  pub async fn open_buffered(path: impl AsRef<Path>) -> io::Result<Self> {
    Self::open_inner(path.as_ref(), false).await
  }

  async fn open_inner(path: &Path, direct_first: bool) -> io::Result<Self> {
    // 记录是否新建：目录项 fsync 只在新文件上付一次成本。
    let created = !path.try_exists()?;
    let (file, direct) = if direct_first {
      match try_open_direct(path).await {
        Ok(file) => (file, true),
        Err(e) => {
          log::debug!("Direct I/O 回退缓冲 I/O: {e}");
          (open_buffered_file(path).await?, false)
        }
      }
    } else {
      (open_buffered_file(path).await?, false)
    };
    if created {
      // 掉电后目录项可能未持久化：DWB 文件若消失，而最终位置已落盘，恢复将
      // 无记录可回放。目录 fsync 是一次性启动成本，不在 I/O 热路径。
      sync_parent_dir(path).await?;
    }
    Ok(Self::with_file(path.to_path_buf(), file, direct))
  }

  fn with_file(path: PathBuf, file: File, direct: bool) -> Self {
    Self {
      path,
      file: RwLock::new(file),
      direct: AtomicBool::new(direct),
      direct_align: DIRECT_ALIGN,
    }
  }

  /// 是否 Direct I/O 模式
  pub fn is_direct(&self) -> bool {
    self.direct.load(Ordering::Acquire)
  }

  /// 后端绑定的文件路径（crate 内用于校验 DWB/data 必须不同文件）
  pub(crate) fn path(&self) -> &Path {
    &self.path
  }

  /// 当前 fd 句柄（`File` 内部为 `SharedFd` 共享引用，克隆后不持锁跨 await）
  fn handle(&self) -> File {
    match self.file.read() {
      Ok(g) => g.clone(),
      Err(poisoned) => poisoned.into_inner().clone(),
    }
  }

  /// 本次 I/O 是否会因 Direct I/O 对齐要求被拒
  fn needs_degrade(&self, ptr: usize, len: usize, off: u64) -> bool {
    self.is_direct() && misaligned(ptr, len, off, self.direct_align)
  }

  /// 一次性把整文件降级为标准缓存 I/O。
  ///
  /// O_DIRECT 写不经过页缓存、直接进入设备（缓存），切到缓冲 fd 后仍可读到；
  /// 旧 fd 的在途 I/O 持有自己的共享 fd，替换不影响其完成。
  async fn degrade(&self) -> io::Result<()> {
    if !self.is_direct() {
      return Ok(());
    }
    let buffered = open_buffered_file(&self.path).await?;
    let old = {
      let mut guard = match self.file.write() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
      };
      if self.is_direct() {
        self.direct.store(false, Ordering::Release);
        Some(replace(&mut *guard, buffered))
      } else {
        None
      }
    };
    drop(old);
    Ok(())
  }

  /// 读精确长度；Direct I/O 被拒时降级并重试一次。
  async fn read_exact_at_retry<T: IoBufMut>(&self, mut buf: T, off: u64) -> io::Result<T> {
    let len = buf.buf_capacity();
    if len == 0 {
      return Ok(buf);
    }
    if self.needs_degrade(buf.buf_mut_ptr() as usize, len, off) {
      self.degrade().await?;
    }
    loop {
      let file = self.handle();
      let BufResult(res, ret) = file.read_exact_at(buf, off).await;
      match res {
        Ok(()) => return Ok(ret),
        Err(e) if self.is_direct() && is_direct_reject(&e) => {
          self.degrade().await?;
          buf = ret;
        }
        Err(e) => return Err(e),
      }
    }
  }

  /// 写完整长度；Direct I/O 被拒时降级并重试一次。
  async fn write_all_at_retry<T: IoBuf>(&self, mut buf: T, off: u64) -> io::Result<()> {
    let len = buf.buf_len();
    if len == 0 {
      return Ok(());
    }
    if self.needs_degrade(buf.buf_ptr() as usize, len, off) {
      self.degrade().await?;
    }
    loop {
      let file = self.handle();
      let mut file_ref = &file;
      let BufResult(res, ret) = file_ref.write_all_at(buf, off).await;
      match res {
        Ok(()) => return Ok(()),
        Err(e) if self.is_direct() && is_direct_reject(&e) => {
          self.degrade().await?;
          buf = ret;
        }
        Err(e) => return Err(e),
      }
    }
  }

  /// 读一个 4 KiB 页到 `buf`（**`pid` 是页号**，偏移 = `pid * PAGE_SIZE`）。
  ///
  /// compio 的 AFIT 要求 buffer `'static`：VMCache 页帧来自 mmap arena，
  /// 天然满足（`&'static mut [u8]` 实现 `IoBufMut`），且 arena 页 4 KiB 对齐，
  /// 在 Linux/Windows 上可走 Direct I/O 快路径。
  ///
  /// 堆 buffer 请改用 [`read_page_owned`](Self::read_page_owned)（by value 传入、
  /// 读毕归还，无泄漏）—— `Box::leak` 凑 `'static` 的写法是反模式。栈/堆缓冲
  /// 未按 `PAGE_SIZE` 对齐时，本方法自动降级到缓冲 I/O（仅多一次 open）。
  pub async fn read_at_page(
    &self,
    pid: u64,
    buf: &'static mut [u8],
  ) -> io::Result<&'static mut [u8]> {
    self.read_extent(page_offset(pid)?, buf).await
  }

  /// 按**字节偏移**读连续区间（历史名；与 `read_bytes_owned` 同为字节语义）。
  ///
  /// `off` 不是页号。页接口内部也复用它。
  pub async fn read_extent(
    &self,
    off: u64,
    buf: &'static mut [u8],
  ) -> io::Result<&'static mut [u8]> {
    self.read_exact_at_retry(buf, off).await
  }

  /// 拥有 buffer 的整页读（**`pid` 是页号**）。`buf` by value 传入，compio 读毕
  /// 经 `BufResult` 原样归还，调用方继续使用或 drop 回收 —— 零泄漏。
  ///
  /// `[u8; N]` / `Vec<u8>` / `Box<[u8]>` 均实现 `IoBufMut`，可直接传值。
  /// **长度语义**：compio 以 `buf_capacity()` 为读取长度；`Vec<u8>` 的容量若大于
  /// 长度，会读入容量大小而非 `len`。整页读请用 `vec![0u8; PAGE_SIZE]`
  /// （容量恰为 4096）或定长数组。按**字节偏移**读取用
  /// [`read_bytes_owned`](Self::read_bytes_owned)。
  pub async fn read_page_owned<T: IoBufMut>(&self, pid: u64, buf: T) -> io::Result<T> {
    self.read_bytes_owned(page_offset(pid)?, buf).await
  }

  /// [`read_page_owned`](Self::read_page_owned) 的兼容名（L2/L3 现有消费者）。
  ///
  /// `pid` 是页号，不是字节偏移；WAL/extent 等任意字节偏移必须用
  /// [`read_bytes_owned`](Self::read_bytes_owned)。
  pub async fn read_owned<T: IoBufMut>(&self, pid: u64, buf: T) -> io::Result<T> {
    self.read_page_owned(pid, buf).await
  }

  /// 按**字节偏移**拥有式读取（extent/WAL/DWB 恢复路径）。
  ///
  /// 与 [`read_page_owned`](Self::read_page_owned) 相同的长度语义
  /// （按 `buf_capacity()`）。
  pub async fn read_bytes_owned<T: IoBufMut>(&self, off: u64, buf: T) -> io::Result<T> {
    self.read_exact_at_retry(buf, off).await
  }

  /// 写一个 4 KiB 页（**`pid` 是页号**，偏移 = `pid * PAGE_SIZE`）
  pub async fn write_at_page(&self, pid: u64, buf: &'static [u8]) -> io::Result<()> {
    self.write_extent(page_offset(pid)?, buf).await
  }

  /// 按**字节偏移**写连续区间（历史名；与 `write_bytes_at` 同义）。
  ///
  /// `off` 不是页号。页接口内部也复用它。
  pub async fn write_extent(&self, off: u64, buf: &'static [u8]) -> io::Result<()> {
    self.write_all_at_retry(buf, off).await
  }

  /// 按**字节偏移**写连续区间（extent / DWB 恢复）
  pub async fn write_bytes_at(&self, off: u64, buf: &'static [u8]) -> io::Result<()> {
    self.write_extent(off, buf).await
  }

  /// 拥有数据的整段写（**`off` 是字节偏移**；WAL 追加路径：Vec 直接进入 IoBuf，
  /// 无额外拷贝）。
  ///
  /// 泛型 `T: IoBuf`：`Vec<u8>`、`[u8; N]` 或 4 KiB 对齐的拥有式页缓冲均可直接
  /// 传入（DWB 恢复回放用后者保持 data 后端的 Direct I/O 对齐）。
  pub async fn write_bytes_owned<T: IoBuf>(&self, data: T, off: u64) -> io::Result<()> {
    self.write_all_at_retry(data, off).await
  }

  /// [`write_bytes_owned`](Self::write_bytes_owned) 的兼容名（L2/L3 现有消费者）。
  ///
  /// `off` 是字节偏移，不是页号；整页写用 [`write_at_page`](Self::write_at_page)。
  pub async fn write_owned<T: IoBuf>(&self, data: T, off: u64) -> io::Result<()> {
    self.write_bytes_owned(data, off).await
  }

  /// 文件打孔回收 `[off, off+len)`（替代裸盘 trim）。
  ///
  /// `fallocate`/`fcntl` 是可能阻塞的元数据系统调用，放进 compio 阻塞线程池
  /// 执行，避免卡住单线程事件循环。
  pub async fn punch_hole(&self, off: u64, len: u64) -> io::Result<()> {
    punch_hole(self.handle(), off, len).await
  }

  /// fdatasync（数据落盘，OS 默认语义）。
  ///
  /// **macOS 注意**：Darwin 的 `fdatasync/fsync` 只把脏页交给驱动，不保证设备
  /// 写缓存落盘；掉电持久性必须用 [`sync_durable`](Self::sync_durable)。Linux
  /// 上两者等价（`fdatasync` 已包含必要的 cache flush）。
  pub async fn fdatasync(&self) -> io::Result<()> {
    self.handle().sync_data().await
  }

  /// fsync（数据 + 元数据，OS 默认语义）。
  pub async fn sync_all(&self) -> io::Result<()> {
    self.handle().sync_all().await
  }

  /// 崩溃一致性协议要求的**持久化屏障**：返回时数据已越过设备写缓存。
  ///
  /// - Linux：`fdatasync`（数据 + 恢复读取所需元数据；不刷无关元数据）
  /// - Windows：`FlushFileBuffers`（compio `sync_data`）
  /// - macOS：`fcntl(F_FULLFSYNC)`；文件系统不支持时回退 `fsync` 并记日志
  ///
  /// DWB（`write_page → sync → 最终写 → checkpoint`）与 WAL 组提交必须用本方
  /// 法建立「DWB/WAL 先持久化、最终位置后写」的顺序；进程崩溃语义下
  /// [`fdatasync`](Self::fdatasync) 足够，但掉电/设备缓存重排场景不够。
  pub async fn sync_durable(&self) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
      full_fsync(self.handle()).await
    }
    #[cfg(not(target_os = "macos"))]
    {
      self.handle().sync_data().await
    }
  }

  /// 文件长度（`File::metadata().len()`，替代 BLKGETSIZE64）
  pub async fn file_len(&self) -> io::Result<u64> {
    self.handle().metadata().await.map(|m| m.len())
  }

  /// 预分配/截断长度
  pub async fn set_len(&self, len: u64) -> io::Result<()> {
    self.handle().set_len(len).await
  }
}

/// Direct I/O 对齐判定：地址/长度/偏移任一不满足 `align` 即需降级。
///
/// `align == 1`（macOS `F_NOCACHE`）恒为 false。
fn misaligned(ptr: usize, len: usize, off: u64, align: usize) -> bool {
  align != 1
    && (!ptr.is_multiple_of(align)
      || !len.is_multiple_of(align)
      || !off.is_multiple_of(align as u64))
}

/// `pid` → 字节偏移（带溢出检查；合法 pid < 2^49，正常不可达）
fn page_offset(pid: u64) -> io::Result<u64> {
  pid
    .checked_mul(PAGE_SIZE as u64)
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pid 页偏移溢出"))
}

/// Direct I/O 拒绝判定：EINVAL/EFAULT（对齐）或 EOPNOTSUPP（文件系统不支持）。
#[cfg(unix)]
fn is_direct_reject(e: &io::Error) -> bool {
  matches!(
    e.kind(),
    io::ErrorKind::InvalidInput | io::ErrorKind::Unsupported
  ) || matches!(
    e.raw_os_error(),
    Some(libc::EINVAL) | Some(libc::EFAULT) | Some(libc::EOPNOTSUPP)
  )
}

/// Windows：`ERROR_INVALID_PARAMETER`(87) / `ERROR_INVALID_FUNCTION`(1) /
/// `ERROR_NOT_SUPPORTED`(50) 均表示 NO_BUFFERING 对齐/支持问题。
#[cfg(windows)]
fn is_direct_reject(e: &io::Error) -> bool {
  matches!(
    e.kind(),
    io::ErrorKind::InvalidInput | io::ErrorKind::Unsupported
  ) || matches!(e.raw_os_error(), Some(1) | Some(50) | Some(87))
}

#[cfg(not(any(unix, windows)))]
fn is_direct_reject(_e: &io::Error) -> bool {
  false
}

#[cfg(all(unix, target_os = "linux"))]
async fn try_open_direct(path: &Path) -> io::Result<File> {
  use compio::fs::OpenOptions;
  OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(false)
    .custom_flags(libc::O_DIRECT)
    .open(path)
    .await
}

#[cfg(all(unix, target_os = "macos"))]
async fn try_open_direct(path: &Path) -> io::Result<File> {
  use std::os::fd::AsRawFd;

  use compio::fs::OpenOptions;
  // macOS：libc crate 不暴露 O_DIRECT；统一 F_NOCACHE 等价实现页缓存旁路
  let file = OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(false)
    .open(path)
    .await?;
  // SAFETY: fd 来自已打开的 File，生命周期由 File 持有
  let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
  if rc != 0 {
    return Err(io::Error::last_os_error());
  }
  Ok(file)
}

#[cfg(not(unix))]
async fn try_open_direct(path: &Path) -> io::Result<File> {
  use compio::fs::OpenOptions;
  /// `FILE_FLAG_NO_BUFFERING`（Win32 `CreateFileW` 标志，非 libc 常量）
  const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
  OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(false)
    .custom_flags(FILE_FLAG_NO_BUFFERING)
    .open(path)
    .await
}

async fn open_buffered_file(path: &Path) -> io::Result<File> {
  use compio::fs::OpenOptions;
  OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(false)
    .open(path)
    .await
}

/// 新建文件后 fsync 父目录，使目录项（文件名 → inode）持久化。
///
/// 只在文件原本不存在时调用一次（启动/建库路径），不在 I/O 热路径。
/// 目录以只读 fd 打开；`fsync(dir)` 在 Linux ext4/xfs/tmpfs 与 macOS APFS
/// 均成功（实测 APFS `fsync(dir)` 返回 0）。
#[cfg(unix)]
async fn sync_parent_dir(path: &Path) -> io::Result<()> {
  let parent = path
    .parent()
    .filter(|p| !p.as_os_str().is_empty())
    .unwrap_or_else(|| Path::new("."));
  let dir = File::open(parent).await?;
  dir.sync_all().await
}

/// Windows：用户态没有等价的「目录 fsync」API（`CreateFileW` 打开目录需
/// `FILE_FLAG_BACKUP_SEMANTICS`，`FlushFileBuffers` 也不保证目录项）；
/// NTFS 目录项由元数据日志保证，故此处不额外操作。
#[cfg(not(unix))]
async fn sync_parent_dir(_path: &Path) -> io::Result<()> {
  Ok(())
}

/// macOS 掉电持久化：`F_FULLFSYNC` 才等价于 Linux `fdatasync` 的设备 cache flush。
///
/// `fsync/fdatasync` 只把数据交给驱动；`F_FULLFSYNC` 会清空设备写缓存。
/// 文件系统不支持时（ENOTSUP/EINVAL）回退 `fsync`，此时仅进程崩溃安全。
#[cfg(target_os = "macos")]
async fn full_fsync(file: File) -> io::Result<()> {
  use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

  // SAFETY: dup 只复制 fd，不转移 file 所有权；失败返回负值
  let fd = unsafe { libc::dup(file.as_raw_fd()) };
  if fd < 0 {
    return Err(io::Error::last_os_error());
  }
  // SAFETY: fd 由 dup 新建且尚未被接管，OwnedFd 取得唯一所有权
  let owned = unsafe { OwnedFd::from_raw_fd(fd) };
  let res = spawn_blocking(move || {
    // SAFETY: fd 有效；F_FULLFSYNC 清空设备写缓存并把文件数据写到介质
    let rc = unsafe { libc::fcntl(owned.as_raw_fd(), libc::F_FULLFSYNC) };
    if rc == 0 {
      Ok(())
    } else {
      Err(io::Error::last_os_error())
    }
  })
  .await;
  match res {
    Ok(Ok(())) => Ok(()),
    Ok(Err(e)) if matches!(e.raw_os_error(), Some(libc::ENOTSUP) | Some(libc::EINVAL)) => {
      log::debug!("F_FULLFSYNC 不支持，回退 fsync（仅进程崩溃安全）: {e}");
      file.sync_all().await
    }
    Ok(Err(e)) => Err(e),
    Err(_) => Err(io::Error::other("F_FULLFSYNC 线程异常终止")),
  }
}

/// 打孔：Linux `fallocate`；macOS `F_PUNCHHOLE`；Windows 暂不支持。
#[cfg(all(unix, target_os = "linux"))]
async fn punch_hole(file: File, off: u64, len: u64) -> io::Result<()> {
  use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

  // SAFETY: dup 仅复制 fd，不转移 file 所有权；失败返回负值
  let fd = unsafe { libc::dup(file.as_raw_fd()) };
  if fd < 0 {
    return Err(io::Error::last_os_error());
  }
  // SAFETY: fd 由 dup 新建且尚未被接管，OwnedFd 取得唯一所有权
  let owned = unsafe { OwnedFd::from_raw_fd(fd) };
  let res = spawn_blocking(move || {
    // SAFETY: fd 有效；PUNCH_HOLE|KEEP_SIZE 对普通文件打孔
    let rc = unsafe {
      libc::fallocate(
        owned.as_raw_fd(),
        libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
        off as libc::off_t,
        len as libc::off_t,
      )
    };
    if rc != 0 {
      return Err(io::Error::last_os_error());
    }
    Ok(())
  })
  .await;
  join_blocking(res)
}

#[cfg(all(unix, target_os = "macos"))]
async fn punch_hole(file: File, off: u64, len: u64) -> io::Result<()> {
  use std::{
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    ptr::from_ref,
  };

  // SAFETY: dup 仅复制 fd，不转移 file 所有权；失败返回负值
  let fd = unsafe { libc::dup(file.as_raw_fd()) };
  if fd < 0 {
    return Err(io::Error::last_os_error());
  }
  // SAFETY: fd 由 dup 新建且尚未被接管，OwnedFd 取得唯一所有权
  let owned = unsafe { OwnedFd::from_raw_fd(fd) };
  let res = spawn_blocking(move || {
    // Darwin `<sys/fcntl.h>`：fpunchhole_t { u32 fp_flags; u32 reserved;
    // off_t fp_offset; off_t fp_length; } —— 必须用 libc 定义，手写两字段
    // 结构体会把 flags/reserved 当成 offset，打错区间甚至静默丢数据。
    let st = libc::fpunchhole_t {
      fp_flags: 0,
      reserved: 0,
      fp_offset: off as libc::off_t,
      fp_length: len as libc::off_t,
    };
    // SAFETY: fd 有效；fpunchhole 与 Linux PUNCH_HOLE 语义一致（长度保持）
    let rc = unsafe {
      libc::fcntl(
        owned.as_raw_fd(),
        libc::F_PUNCHHOLE,
        from_ref(&st).cast_mut(),
      )
    };
    if rc != 0 {
      return Err(io::Error::last_os_error());
    }
    Ok(())
  })
  .await;
  join_blocking(res)
}

/// 把阻塞线程池的 panic/取消映射为 I/O 错误（不 panic 到调用方）。
#[cfg(unix)]
fn join_blocking(res: Result<io::Result<()>, JoinError>) -> io::Result<()> {
  match res {
    Ok(r) => r,
    Err(_) => Err(io::Error::other("打孔线程异常终止")),
  }
}

#[cfg(not(unix))]
async fn punch_hole(_file: File, _off: u64, _len: u64) -> io::Result<()> {
  // Windows FSCTL_SET_ZERO_DATA 需 Overlapped I/O，compio File 暂不暴露；
  // 先返回不支持（调用方按配置降级为顺序覆写）
  Err(io::Error::new(
    io::ErrorKind::Unsupported,
    "Windows 打孔待 M5 补",
  ))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn direct_alignment_gate() {
    // 4 KiB 对齐：地址/长度/偏移三者都对才放行
    assert!(!misaligned(4096, 4096, 8192, 4096));
    assert!(misaligned(1, 4096, 8192, 4096));
    assert!(misaligned(4096, 512, 8192, 4096));
    assert!(misaligned(4096, 4096, 512, 4096));
    // macOS F_NOCACHE 无对齐要求
    assert!(!misaligned(1, 512, 123, 1));
  }

  #[cfg(unix)]
  #[test]
  fn direct_reject_kinds() {
    for err in [libc::EINVAL, libc::EFAULT, libc::EOPNOTSUPP] {
      assert!(is_direct_reject(&io::Error::from_raw_os_error(err)));
    }
    assert!(!is_direct_reject(&io::Error::from(io::ErrorKind::NotFound)));
  }

  /// `open_buffered` 不得尝试 Direct I/O（DWB 4160B 记录的生产入口）；
  /// `sync_durable` 写入后必须能读回。
  #[compio::test]
  async fn open_buffered_stays_buffered_and_sync_durable_roundtrips() -> io::Result<()> {
    use std::{
      env::temp_dir,
      fs,
      process::id,
      sync::atomic::{AtomicU64, Ordering},
    };

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = temp_dir().join(format!(
      "ldb_io_open_buffered_{}_{}",
      id(),
      SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir)?;
    let b = CompioFileBackend::open_buffered(dir.join("dwb.dat")).await?;
    assert!(!b.is_direct(), "open_buffered 必须是缓冲 I/O");
    b.write_bytes_owned(vec![0x5A; PAGE_SIZE], 0).await?;
    b.sync_durable().await?;
    let got = b.read_page_owned(0, vec![0u8; PAGE_SIZE]).await?;
    assert!(got.iter().all(|&x| x == 0x5A));
    drop(b);
    fs::remove_dir_all(&dir)?;
    Ok(())
  }

  /// Darwin ABI：`fpunchhole_t` 必须是 24 B（4+4+8+8），字段偏移 8/16。
  /// 手写两字段结构体会把 flags/reserved 当 offset（R1 P0 的回归护栏）。
  #[cfg(target_os = "macos")]
  #[test]
  fn fpunchhole_layout_is_24_bytes() {
    use std::mem::{offset_of, size_of};

    assert_eq!(size_of::<libc::fpunchhole_t>(), 24);
    assert_eq!(offset_of!(libc::fpunchhole_t, fp_offset), 8);
    assert_eq!(offset_of!(libc::fpunchhole_t, fp_length), 16);
  }

  /// 并发：对齐整页写（可能仍在 Direct 路径）与未对齐字节写（触发 degrade）
  /// 同时 in-flight。旧 fd 由 `File` 克隆的 SharedFd 持有到完成；降级后每个
  /// 请求要么完整走旧 direct fd，要么完整走新 buffered fd，读回内容不得混合。
  #[compio::test]
  async fn concurrent_unaligned_degrade_keeps_pages_whole() -> io::Result<()> {
    use std::{
      env::temp_dir,
      fs,
      process::id,
      rc::Rc,
      sync::atomic::{AtomicU64, Ordering},
    };

    use compio::runtime::spawn;

    #[repr(align(4096))]
    struct Aligned([u8; PAGE_SIZE]);
    static ALIGNED_PAGE: Aligned = Aligned([0xA5; PAGE_SIZE]);
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let dir = temp_dir().join(format!(
      "ldb_io_degrade_{}_{}",
      id(),
      SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir)?;
    let mut b = CompioFileBackend::open_direct_or_buffered(dir.join("data.db")).await?;
    let was_direct = b.is_direct();
    // APFS 的 F_NOCACHE 本来不要求对齐；注入 4096 以在本机覆盖 Linux 降级路径
    b.direct_align = PAGE_SIZE;
    let b = Rc::new(b);
    b.set_len(64 * PAGE_SIZE as u64).await?;

    let mut handles = Vec::new();
    for p in 0..16u64 {
      let b = Rc::clone(&b);
      handles.push(spawn(
        async move { b.write_at_page(p, &ALIGNED_PAGE.0).await },
      ));
    }
    for t in 0..16u64 {
      let b = Rc::clone(&b);
      handles.push(spawn(async move {
        let val = 0x40 + t as u8;
        let off = (16 + t) * PAGE_SIZE as u64 + 123;
        b.write_bytes_owned(vec![val; 777], off).await
      }));
    }
    for h in handles {
      h.await??;
    }
    b.fdatasync().await?;
    if was_direct {
      assert!(!b.is_direct(), "未对齐请求后必须已降级为缓冲 I/O");
    }

    for p in 0..16u64 {
      let got = b.read_page_owned(p, vec![0u8; PAGE_SIZE]).await?;
      assert!(got.iter().all(|&x| x == 0xA5), "对齐页 {p} 内容混合");
    }
    for t in 0..16u64 {
      let val = 0x40 + t as u8;
      let off = (16 + t) * PAGE_SIZE as u64;
      let got = b.read_bytes_owned(off, vec![0u8; PAGE_SIZE]).await?;
      assert!(got[..123].iter().all(|&x| x == 0));
      assert!(got[123..900].iter().all(|&x| x == val));
      assert!(got[900..].iter().all(|&x| x == 0));
    }
    drop(b);
    fs::remove_dir_all(&dir)?;
    Ok(())
  }
}
