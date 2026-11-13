//! 纯内存异步后端（单测 / fuzz / SSD 模拟用）。
//!
//! 页级接口（`read_at_page`/`write_at_page`/`punch_hole`/`file_len`）与
//! `CompioFileBackend` 对齐；存储单位为 `PAGE_SIZE`，`read_at_page` 对未写过
//! 的区域返回零（稀疏文件/打孔语义），而非 EOF。
//!
//! # 页号 vs 字节偏移（防 8× 错位）
//!
//! | 参数语义 | 方法 |
//! |---|---|
//! | 页号 `pid`（偏移 = `pid * PAGE_SIZE`） | [`read_at_page`](Self::read_at_page)、[`write_at_page`](Self::write_at_page) |
//! | 字节偏移 `off` | [`read_bytes`](Self::read_bytes)、[`write_bytes`](Self::write_bytes)、[`punch_hole`](Self::punch_hole) |
//!
//! WAL 的 512 B 扇区号/任意字节偏移**不得**传给页级方法：`pid=512` 会被当成
//! 第 512 页（偏移 2 MiB），而不是 512 字节处。`write_at_page` 允许非整页长度
//! 仅为兼容上游扇区写语义，偏移始终按页号计算。
//!
//! 并发形状与 `CompioFileBackend` 对齐：全部方法 `&self`（compio 单线程每核下
//! 多协程可并发发起 I/O 而不互相阻塞）；内部可变性走 `RefCell`，借用在方法
//! 返回前释放。纯内存操作立即完成，**不**插入 `yield_now`：实测 compio 单次
//! `yield_now` ≈ 15 µs（`cargo bench -p ldb_io`），比 4 KiB 拷贝贵两个数量级，
//! 只会拖慢 fuzz/SSD 模拟；async 签名仅为与文件后端接口一致。

use std::{cell::RefCell, io};

use ldb_type::PAGE_SIZE;

/// 内存后端：按页扩展的 Vec 存储，接口与 `CompioFileBackend` 的页级接口对齐
#[derive(Debug, Default)]
pub struct MemBackend {
  data: RefCell<Vec<u8>>,
}

impl MemBackend {
  /// 构造
  pub fn new() -> Self {
    Self::default()
  }

  /// `pid` → 字节偏移（带溢出检查）
  fn page_offset(pid: u64) -> io::Result<usize> {
    usize::try_from(pid)
      .ok()
      .and_then(|p| p.checked_mul(PAGE_SIZE))
      .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "pid 页偏移溢出"))
  }

  /// 读一页（签名与 CompioFileBackend 对齐：归还 buffer）。
  ///
  /// `pid` 是页号（偏移 = `pid * PAGE_SIZE`）；任意字节偏移用
  /// [`read_bytes`](Self::read_bytes)。稀疏洞语义：未写过的区域读零
  /// （与稀疏文件 / 打孔回收区一致），而非 EOF —— 缺页加载新页依赖零页行为。
  pub async fn read_at_page<'a>(&self, pid: u64, buf: &'a mut [u8]) -> io::Result<&'a mut [u8]> {
    self.read_bytes(Self::page_offset(pid)? as u64, buf).await
  }

  /// 在 `pid * PAGE_SIZE` 处写 `buf`（任意长度；自动扩容）。
  ///
  /// `pid` 是页号；任意字节偏移用 [`write_bytes`](Self::write_bytes)。
  pub async fn write_at_page(&self, pid: u64, buf: &[u8]) -> io::Result<()> {
    self.write_bytes(Self::page_offset(pid)? as u64, buf).await
  }

  /// 按**字节偏移**读，未写过/越过 EOF 的区域读零。
  pub async fn read_bytes<'a>(&self, off: u64, buf: &'a mut [u8]) -> io::Result<&'a mut [u8]> {
    let start = usize::try_from(off)
      .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "读偏移溢出"))?;
    // 短借拷贝，方法返回前释放
    let data = self.data.borrow();
    let valid = data.len().saturating_sub(start).min(buf.len());
    if valid > 0 {
      buf[..valid].copy_from_slice(&data[start..start + valid]);
    }
    buf[valid..].fill(0);
    Ok(buf)
  }

  /// 按**字节偏移**写 `data`（任意长度；自动扩容，中间空洞读零）。
  pub async fn write_bytes(&self, off: u64, data: &[u8]) -> io::Result<()> {
    let start = usize::try_from(off)
      .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "写偏移溢出"))?;
    let end = start
      .checked_add(data.len())
      .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "写区间溢出"))?;
    let mut buf = self.data.borrow_mut();
    if end > buf.len() {
      buf.resize(end, 0);
    }
    buf[start..end].copy_from_slice(data);
    Ok(())
  }

  /// 打孔（置零 `[off, off+len)` 与现有数据重叠的部分）。
  ///
  /// 区间越过当前长度时只清现有部分；越过 EOF 的部分按稀疏语义本来就读零。
  pub async fn punch_hole(&self, off: u64, len: u64) -> io::Result<()> {
    let start = usize::try_from(off)
      .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "打孔偏移溢出"))?;
    let len = usize::try_from(len)
      .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "打孔长度溢出"))?;
    let end = start
      .checked_add(len)
      .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "打孔区间溢出"))?;
    let mut data = self.data.borrow_mut();
    let hi = end.min(data.len());
    if start < hi {
      data[start..hi].fill(0);
    }
    Ok(())
  }

  /// 长度
  pub async fn file_len(&self) -> io::Result<u64> {
    Ok(self.data.borrow().len() as u64)
  }
}
