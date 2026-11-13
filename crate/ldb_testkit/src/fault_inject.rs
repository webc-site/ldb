//! 故障注入 I/O 后端（todo.md §4.1 第 4 防线 —— 崩溃一致性与掉电回归）。
//!
//! 三种页粒度故障（async，compio 单线程每核）：
//! - [`FaultKind::DropWrite`]：写直接失败（模拟掉电丢页）；
//! - [`FaultKind::BitFlip`]：写前翻转指定字节（模拟介质位翻转）；
//! - [`FaultKind::Truncate`]：写前截断尾部（模拟半页撕裂写）。
//!
//! 故障按「写入次数」触发：第 `at_write` 次 `write_page` 起生效；
//! `None` 即直通。计数与模式均为内部可变性（`&self` 注入 BufferManager）。

use std::{
  cell::RefCell,
  fmt, io,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
};

use ldb_buf::PageBackend;
use ldb_type::{PAGE_SIZE, PageId};

/// 故障模式
#[derive(Debug, Clone, Copy, Default)]
pub enum FaultKind {
  /// 无故障（直通）
  #[default]
  None,
  /// 自第 `at_write` 次写起全部失败（掉电丢页）
  DropWrite {
    /// 触发的写入序号（0 基）
    at_write: u64,
  },
  /// 自第 `at_write` 次写起，每次写翻转页内 `offset` 字节的最低位（介质位翻转）
  BitFlip {
    /// 触发的写入序号（0 基）
    at_write: u64,
    /// 翻转的字节偏移（< PAGE_SIZE）
    offset: usize,
  },
  /// 自第 `at_write` 次写起，每次写只保留前 `keep` 字节、尾部清零（半页撕裂）
  Truncate {
    /// 触发的写入序号（0 基）
    at_write: u64,
    /// 保留前缀长度（< PAGE_SIZE）
    keep: usize,
  },
}

/// 故障注入后端（包装任意 [`PageBackend`]）
pub struct FaultInjectBackend<B: PageBackend> {
  /// 被包装后端
  inner: B,
  mode: RefCell<FaultKind>,
  /// 已执行写次数
  write_cnt: AtomicU64,
  /// 已注入故障次数（断言用）
  injected: AtomicU64,
}

impl<B: PageBackend> fmt::Debug for FaultInjectBackend<B> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("FaultInjectBackend")
      .field("mode", &*self.mode.borrow())
      .field("write_cnt", &self.write_cnt.load(Ordering::Relaxed))
      .field("injected", &self.injected.load(Ordering::Relaxed))
      .finish_non_exhaustive()
  }
}

impl<B: PageBackend> FaultInjectBackend<B> {
  /// 包装构造（初始无故障）
  pub fn new(inner: B) -> Self {
    Self {
      inner,
      mode: RefCell::new(FaultKind::None),
      write_cnt: AtomicU64::new(0),
      injected: AtomicU64::new(0),
    }
  }

  /// 设置故障模式
  pub fn set_mode(&self, m: FaultKind) {
    *self.mode.borrow_mut() = m;
  }

  /// 清除故障（恢复直通）
  pub fn clear(&self) {
    *self.mode.borrow_mut() = FaultKind::None;
  }

  /// 已注入故障次数
  pub fn injected_cnt(&self) -> u64 {
    self.injected.load(Ordering::Relaxed)
  }

  /// 已执行写次数
  pub fn write_cnt(&self) -> u64 {
    self.write_cnt.load(Ordering::Relaxed)
  }

  /// 被包装后端引用（恢复 / 检查落盘真值）
  pub fn inner(&self) -> &B {
    &self.inner
  }

  /// 应用故障变换：Ok(None)=直通，Ok(Some(页))=变换后写，Err=写失败
  fn apply(&self, src: &[u8; PAGE_SIZE]) -> Result<Option<[u8; PAGE_SIZE]>, io::Error> {
    let n = self.write_cnt.fetch_add(1, Ordering::Relaxed);
    match *self.mode.borrow() {
      FaultKind::None => Ok(None),
      FaultKind::DropWrite { at_write } if n >= at_write => {
        self.injected.fetch_add(1, Ordering::Relaxed);
        Err(io::Error::other("注入掉电：写丢失"))
      }
      FaultKind::BitFlip { at_write, offset } if n >= at_write => {
        debug_assert!(offset < PAGE_SIZE, "翻转偏移须 < PAGE_SIZE");
        self.injected.fetch_add(1, Ordering::Relaxed);
        let mut page = *src;
        page[offset % PAGE_SIZE] ^= 1;
        Ok(Some(page))
      }
      FaultKind::Truncate { at_write, keep } if n >= at_write => {
        debug_assert!(keep < PAGE_SIZE, "保留前缀须 < PAGE_SIZE");
        self.injected.fetch_add(1, Ordering::Relaxed);
        let mut page = *src;
        page[keep % PAGE_SIZE..].fill(0);
        Ok(Some(page))
      }
      _ => Ok(None),
    }
  }
}

impl<B: PageBackend> PageBackend for FaultInjectBackend<B> {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
    self.inner.read_page(pid, dst).await
  }

  async fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> io::Result<()> {
    match self.apply(src) {
      Ok(None) => self.inner.write_page(pid, src).await,
      Ok(Some(page)) => self.inner.write_page(pid, &page).await,
      Err(e) => Err(e),
    }
  }

  async fn sync(&self) -> io::Result<()> {
    self.inner.sync().await
  }
}

/// `Arc` 化便捷别名（Facade / 测试共享）
pub type SharedFaultBackend<B> = Arc<FaultInjectBackend<B>>;
