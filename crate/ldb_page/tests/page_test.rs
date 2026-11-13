//! PageHeader 字节布局、写历史环形语义、元数据页根表（上游
//! `ZLeanStore/src/include/storage/page.h`）。
//!
//! 布局期望值全部来自 clang++ 对真实上游头文件的 `sizeof`/`offsetof` 输出：
//! `sizeof(PageHeader)=104`，字段偏移 `0/8/16/24/32/96/97`，
//! `sizeof(Page)=sizeof(MetadataPage)=4096`、`roots` 偏移 104、499 槽。

use std::{
  env, fs,
  mem::{MaybeUninit, align_of, offset_of, size_of},
  path::Path,
  process, ptr,
};

use ldb_page::{MetadataPage, PAGE_BODY, Page, PageHeader, ROOT_SLOTS};
use ldb_type::{PAGE_SIZE, WRITE_HISTORY_LENGTH};

/// 把值写入预填充 `prefill` 的同尺寸槽后取回全部字节。
///
/// 用途：若类型的某个字节未被构造覆盖（遗留隐式 padding），该字节会保留
/// 预填充值，两个不同 `prefill` 的字节映像因此不等。
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
fn page_layout_matches_upstream_cxx() {
  assert_eq!(size_of::<PageHeader>(), 104);
  assert_eq!(align_of::<PageHeader>(), 8);
  assert_eq!(offset_of!(PageHeader, idx_id), 0);
  assert_eq!(offset_of!(PageHeader, p_gsn), 8);
  assert_eq!(offset_of!(PageHeader, prev_gcwrite), 16);
  assert_eq!(offset_of!(PageHeader, wcnt), 24);
  assert_eq!(offset_of!(PageHeader, w_p_gsn), 32);
  assert_eq!(offset_of!(PageHeader, dirty), 96);
  assert_eq!(offset_of!(PageHeader, gc_read), 97);

  assert_eq!(PAGE_BODY, 4096 - 104);
  assert_eq!(ROOT_SLOTS, 499);
  assert_eq!(size_of::<Page>(), PAGE_SIZE);
  assert_eq!(size_of::<MetadataPage>(), PAGE_SIZE);
  assert_eq!(align_of::<Page>(), PAGE_SIZE);
  assert_eq!(align_of::<MetadataPage>(), PAGE_SIZE);
  assert_eq!(offset_of!(MetadataPage, roots), 104);

  // 字段宽度钉死（宽度变化即使不移动偏移（如 tier_index）也会在编译期失败）
  let h = PageHeader::new();
  let _: u16 = h.idx_id;
  let _: u64 = h.p_gsn;
  let _: u64 = h.prev_gcwrite;
  let _: u32 = h.wcnt;
  let _: [u64; WRITE_HISTORY_LENGTH] = h.w_p_gsn;
  let _: bool = h.dirty;
  let _: bool = h.gc_read;
}

#[test]
fn page_header_new_is_zeroed() {
  let h = PageHeader::new();
  assert_eq!(h.idx_id, 0);
  assert_eq!(h.p_gsn, 0);
  assert_eq!(h.prev_gcwrite, 0);
  assert_eq!(h.wcnt, 0);
  assert_eq!(h.w_p_gsn, [0u64; 8]);
  assert!(!h.dirty);
  assert!(!h.gc_read);
}

/// 104 字节全部由显式字段覆盖：两次构造逐字节相等且全零。
///
/// 上游 `PageHeader` 在 `idx_id` 后（2..8）与 `wcnt` 后（28..32）、
/// `gc_read` 后（98..104）有隐式对齐填充；Rust 侧必须显式建模，
/// 否则 typed copy 可能把未初始化字节写入落盘镜像/WAL/哈希。
#[test]
fn page_header_has_no_implicit_padding() {
  let a: [u8; 104] = unsafe { written_bytes(PageHeader::new(), 0xAA) };
  let b: [u8; 104] = unsafe { written_bytes(PageHeader::new(), 0x55) };
  assert_eq!(a, b, "两次构造的字节映像必须确定且相等");
  assert_eq!(a, [0u8; 104], "new() 必须把全部 104 字节（含填充）清零");
}

/// 元数据页 4096 B 全部清零：页头 104 B + 根表 499×8 B 恰好铺满，
/// 根表之后没有未覆盖的尾部。
#[test]
fn metadata_page_new_zeroes_all_bytes() {
  let a: [u8; PAGE_SIZE] = unsafe { written_bytes(MetadataPage::new(), 0xAA) };
  let b: [u8; PAGE_SIZE] = unsafe { written_bytes(MetadataPage::new(), 0x55) };
  assert_eq!(a, b, "两次构造的字节映像必须确定且相等");
  assert_eq!(a, [0u8; PAGE_SIZE]);
}

/// 普通页 4096 B 全部清零（页头 + 页体），且落盘字节确定：
/// 同一逻辑页在 0xAA/0x55 两种栈垃圾下构造的字节相等；写入文件两次，
/// 读回哈希一致且等于内存字节哈希。
#[test]
fn page_new_zeroes_all_bytes() {
  let a: [u8; PAGE_SIZE] = unsafe { written_bytes(Page::new(), 0xAA) };
  let b: [u8; PAGE_SIZE] = unsafe { written_bytes(Page::new(), 0x55) };
  assert_eq!(a, b, "两次构造的字节映像必须确定且相等");
  assert_eq!(a, [0u8; PAGE_SIZE]);

  let path = env::temp_dir().join(format!("ldb_page_r3_{}.bin", process::id()));
  let Ok(()) = fs::write(&path, a) else {
    panic!("首次写页失败: {}", path.display());
  };
  let first = fnv1a(&read_file(&path));
  let Ok(()) = fs::write(&path, b) else {
    panic!("二次写页失败: {}", path.display());
  };
  let second = fnv1a(&read_file(&path));
  let _ = fs::remove_file(&path);

  assert_eq!(first, second, "两次落盘读回的哈希必须一致");
  assert_eq!(first, fnv1a(&a), "落盘字节必须与内存构造字节一致");
}

/// FNV-1a（确定性、无依赖；仅用于测试内文件往返比对）
fn fnv1a(bytes: &[u8]) -> u64 {
  let mut hash = 0xCBF2_9CE4_8422_2325_u64;
  for &byte in bytes {
    hash ^= u64::from(byte);
    hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
  }
  hash
}

fn read_file(path: &Path) -> Vec<u8> {
  match fs::read(path) {
    Ok(bytes) => bytes,
    Err(err) => panic!("读回落盘页失败: {err}"),
  }
}

#[test]
fn add_write_gsn_inserts_after_max_index() {
  let mut h = PageHeader::new();
  // 全零历史的最大下标 = 0 → 首写落在 1（上游 AddWriteGSN 的真实行为）
  h.add_write_gsn(100);
  assert_eq!(h.w_p_gsn, [0, 100, 0, 0, 0, 0, 0, 0]);

  for g in [200, 300, 400, 500, 600, 700] {
    h.add_write_gsn(g);
  }
  assert_eq!(h.w_p_gsn, [0, 100, 200, 300, 400, 500, 600, 700]);

  h.add_write_gsn(800); // 最大在 7 → 覆盖 0
  assert_eq!(h.w_p_gsn, [800, 100, 200, 300, 400, 500, 600, 700]);
  h.add_write_gsn(900); // 最大在 0 → 覆盖 1
  assert_eq!(h.w_p_gsn, [800, 900, 200, 300, 400, 500, 600, 700]);
  assert_eq!(h.wcnt, 9);
}

#[test]
fn add_write_gsn_ties_pick_lowest_index() {
  let mut h = PageHeader::new();
  // 下标 0 与 7 并列最大：上游用 `>`，取先遇到的 0 → 插入 1
  h.w_p_gsn = [5, 0, 0, 0, 0, 0, 0, 5];
  h.add_write_gsn(9);
  assert_eq!(h.w_p_gsn, [5, 9, 0, 0, 0, 0, 0, 5]);
}

#[test]
fn remove_last_write_gsn_clears_rightmost_index_not_max_value() {
  let mut h = PageHeader::new();
  // 最右非零项是 3（下标 7），不是最大值 5（下标 0）
  h.w_p_gsn = [5, 0, 0, 0, 0, 0, 0, 3];
  h.remove_last_write_gsn();
  assert_eq!(h.w_p_gsn, [5, 0, 0, 0, 0, 0, 0, 0]);

  // 空历史：no-op
  let mut empty = PageHeader::new();
  empty.remove_last_write_gsn();
  assert_eq!(empty.w_p_gsn, [0; WRITE_HISTORY_LENGTH]);

  // 上游不递减 wcnt
  let mut h = PageHeader::new();
  h.wcnt = 7;
  h.w_p_gsn = [1, 0, 0, 0, 0, 0, 0, 0];
  h.remove_last_write_gsn();
  assert_eq!(h.wcnt, 7);
}

#[test]
fn w_history_init_only_clears_array() {
  let mut h = PageHeader::new();
  h.wcnt = 3;
  h.w_p_gsn = [1, 2, 3, 4, 5, 6, 7, 8];
  h.w_history_init();
  assert_eq!(h.wcnt, 3, "上游 WHistoryInit 不清 wcnt");
  assert_eq!(h.w_p_gsn, [0u64; 8]);
}

#[test]
fn update_last_gc_write_only_sets_prev_gcwrite() {
  let mut h = PageHeader::new();
  h.p_gsn = 11;
  h.wcnt = 2;
  h.update_last_gc_write(999);
  assert_eq!(h.prev_gcwrite, 999);
  assert_eq!(h.p_gsn, 11);
  assert_eq!(h.wcnt, 2);
}

#[test]
fn metadata_roots_are_bounds_checked() {
  let mut m = MetadataPage::new();
  assert_eq!(m.get_root(0), Some(0));
  m.roots[3] = 42;
  assert_eq!(m.get_root(3), Some(42));
  assert_eq!(m.get_root(498), Some(0));
  assert_eq!(m.get_root(499), None);
  assert_eq!(m.get_root(usize::MAX), None);
  assert_eq!(m.roots.len(), 499);
}

/// 与真实上游 `PageHeader` 的 100000 次 Add/Remove 序列对拍。
///
/// 期望值由 clang++ 编译 `ZLeanStore/src/include/storage/page.h` 后输出：
/// `wcnt=66516`、数组如上、FNV-1a 状态哈希 `7150830501392083404`。
#[test]
fn write_history_matches_cpp_differential_100k() {
  let mut h = PageHeader::new();
  let mut state = 0x9E37_79B9_7F4A_7C15_u64;
  for _ in 0..100_000 {
    state = state
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    let op = (state >> 33) % 3;
    if op < 2 {
      h.add_write_gsn(state >> 1);
    } else {
      h.remove_last_write_gsn();
    }
  }

  assert_eq!(h.wcnt, 66_516);
  assert_eq!(
    h.w_p_gsn,
    [
      0,
      1_828_087_025_998_302_269,
      7_542_068_099_514_690_039,
      8_153_912_288_220_147_393,
      8_149_322_752_774_536_922,
      0,
      0,
      0,
    ]
  );

  let mut hash = 1_469_598_103_934_665_603_u64;
  let mut mix = |v: u64| hash = (hash ^ v).wrapping_mul(1_099_511_628_211);
  mix(u64::from(h.wcnt));
  for &g in &h.w_p_gsn {
    mix(g);
  }
  assert_eq!(hash, 7_150_830_501_392_083_404);
}
