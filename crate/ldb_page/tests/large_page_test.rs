//! LargePage / LargePageTier 布局（上游
//! `src/include/storage/extent/large_page.h`）。
//!
//! clang++ 对拍：两者 `sizeof == 16`、`alignof == 8`、`start_pid` 偏移 0、
//! 第二字段偏移 8；`LargePageTier` 的 9..16 为上游隐式填充，Rust 侧显式建模。

use std::{
  mem::{MaybeUninit, align_of, offset_of, size_of},
  ptr,
};

use ldb_page::{LargePage, LargePageTier};

/// 把值写入预填充 `prefill` 的同尺寸槽后取回全部字节。
///
/// 若类型的某个字节未被构造覆盖（遗留隐式 padding），该字节会保留预填充值，
/// 两个不同 `prefill` 的字节映像因此不等。
///
/// # Safety
/// 槽先被 `write_bytes` 完整初始化为 `prefill`；随后 `write` 构造值。
/// 类型若含未初始化 padding，`write` 之后读取该字节在抽象机上未定义 ——
/// 这正是本审计要暴露的缺陷（修复后所有字节均为显式字段）。
unsafe fn written_bytes<T, const N: usize>(value: T, prefill: u8) -> [u8; N] {
  assert_eq!(size_of::<T>(), N);
  let mut slot = MaybeUninit::<T>::uninit();
  // SAFETY: T 恰为 N 字节，指针有效且对齐；write 后整块已初始化。
  unsafe {
    ptr::write_bytes(slot.as_mut_ptr().cast::<u8>(), prefill, N);
    slot.as_mut_ptr().write(value);
    slot.as_ptr().cast::<[u8; N]>().read()
  }
}

#[test]
fn large_page_layout_matches_upstream_cxx() {
  assert_eq!(size_of::<LargePage>(), 16);
  assert_eq!(align_of::<LargePage>(), 8);
  assert_eq!(offset_of!(LargePage, start_pid), 0);
  assert_eq!(offset_of!(LargePage, page_cnt), 8);

  assert_eq!(size_of::<LargePageTier>(), 16);
  assert_eq!(align_of::<LargePageTier>(), 8);
  assert_eq!(offset_of!(LargePageTier, start_pid), 0);
  assert_eq!(offset_of!(LargePageTier, tier_index), 8);
}

/// 两种栈垃圾（0xAA/0x55）下构造的 16 B 必须逐字节相同：
/// `start_pid` 8 B + `tier_index` 1 B + 保留位 7 B 全部被构造覆盖。
#[test]
fn large_page_structs_have_no_implicit_padding() {
  let page = LargePage {
    start_pid: 7,
    page_cnt: 1u64 << 40,
  };
  let a: [u8; 16] = unsafe { written_bytes(page, 0xAA) };
  let b: [u8; 16] = unsafe { written_bytes(page, 0x55) };
  assert_eq!(a, b, "LargePage 两次构造的字节映像必须相等");
  let mut expected = [0u8; 16];
  expected[..8].copy_from_slice(&7u64.to_le_bytes());
  expected[8..].copy_from_slice(&(1u64 << 40).to_le_bytes());
  assert_eq!(a, expected);

  let tier = LargePageTier::new(7, 3);
  let a: [u8; 16] = unsafe { written_bytes(tier, 0xAA) };
  let b: [u8; 16] = unsafe { written_bytes(tier, 0x55) };
  assert_eq!(a, b, "LargePageTier 两次构造的字节映像必须相等");
  let mut expected = [0u8; 16];
  expected[..8].copy_from_slice(&7u64.to_le_bytes());
  expected[8] = 3;
  assert_eq!(a, expected, "9..16 的保留位必须为 0");
}

#[test]
fn large_page_fields_roundtrip() {
  let page = LargePage {
    start_pid: 7,
    page_cnt: 1u64 << 40,
  };
  assert_eq!(page.start_pid, 7);
  assert_eq!(page.page_cnt, 1_099_511_627_776);
  let same = page;
  assert_eq!(page, same);

  let tier = LargePageTier::new(7, 3);
  assert_eq!(tier.start_pid, 7);
  assert_eq!(tier.tier_index, 3);
  assert_ne!(tier, LargePageTier::new(7, 4));

  // 字段宽度钉死：u64/u64/u8
  let _: u64 = page.start_pid;
  let _: u64 = page.page_cnt;
  let _: u64 = tier.start_pid;
  let _: u8 = tier.tier_index;
}
