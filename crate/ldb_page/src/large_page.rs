//! 大页（连续页区间）与分级描述符（上游 `src/include/storage/extent/large_page.h`）。
//!
//! 两种类型均为 16 B（clang++ 对拍；`LargePageTier.tier_index` 后有 7 B 隐式
//! 对齐填充）。跨 worktree：`leanstore.blob` 的 `LargePageTier` 字段名为
//! `start_pid`/`tier_index`，`leanstore.latency` 的同类物名为 `ExtentTier`
//! 且字段为 `start_pid`/`tier`；本 crate 采用 `ZLeanStore` 命名（与
//! `todo.md:231` 的 `LargePage`/`LargePageTier` 一致）。

use std::mem::{offset_of, size_of};

use ldb_type::PageId;
use static_assertions::const_assert_eq;

/// 任意页数的连续大页区间（上游 `LargePage`）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct LargePage {
  /// 起始页 ID
  pub start_pid: PageId,
  /// 页数
  pub page_cnt: u64,
}

/// 大页分级描述符（上游 `LargePageTier` / `ExtentTier`）
///
/// 上游 `tier_index` 后有 7 B 隐式对齐填充；本结构显式建模为保留位，
/// 保证按字节序列化（extent 列表落盘/WAL）时不读取未初始化内存。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct LargePageTier {
  /// 起始页 ID
  pub start_pid: PageId,
  /// 指数分级下标（0 → 1 页，1 → 2 页，…）
  pub tier_index: u8,
  /// `tier_index` 后的对齐填充（上游隐式 7 B；9..16）。
  ///
  /// **保留位**：恒为 0。上游未来若新增字段，必须用真实字段替换本字段，
  /// 并同步更新全部 offset/size 断言。
  _pad: [u8; 7],
}

impl LargePageTier {
  /// 构造（上游 `LargePageTier(pageid_t, u8)`；保留位清零）
  pub const fn new(start_pid: PageId, tier_index: u8) -> Self {
    Self {
      start_pid,
      tier_index,
      _pad: [0; 7],
    }
  }
}

const_assert_eq!(size_of::<PageId>(), 8);
const_assert_eq!(size_of::<LargePage>(), 16);
const_assert_eq!(offset_of!(LargePage, start_pid), 0);
const_assert_eq!(offset_of!(LargePage, page_cnt), 8);
const_assert_eq!(size_of::<LargePageTier>(), 16);
const_assert_eq!(offset_of!(LargePageTier, start_pid), 0);
const_assert_eq!(offset_of!(LargePageTier, tier_index), 8);
const_assert_eq!(offset_of!(LargePageTier, _pad), 9);
