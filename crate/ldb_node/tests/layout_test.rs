//! 布局与落盘确定性（`ldb_page` 轮次登记的跨 crate P1）。
//!
//! 期望值来源：
//! - 上游 clang++ 探针（真实 `ZLeanStore/src/include/storage/btree/node.h`）：
//!   `sizeof(PageHeader)=104 sizeof(FenceKeySlot)=4 sizeof(PageSlot)=10
//!    sizeof(BTreeNodeHeader)=208 sizeof(BTreeNode)=4096`；
//!   `idx_id=0 p_gsn=8 prev_gcwrite=16 wcnt=24 w_p_gsn=32 dirty=96 gc_read=97
//!    right_most_child=104 prev_leaf_node=112 lower_fence=120 upper_fence=124
//!    is_leaf=128 count=130 space_used=132 data_offset=134 prefix_len=136 hints=138`。
//! - ldb 自有格式把槽位/长度字段加宽到 u32 并追加 `contention`，绝对偏移由
//!   `src/node/mod.rs` 的 `offset_of!` 断言钉死；本文件复核公开字段与全字节。

use std::{
  env, fs,
  mem::{MaybeUninit, align_of, offset_of, size_of},
  path::PathBuf,
  process, ptr,
};

use ldb_node::{
  BTreeNode, BTreeNodeHeader, ContentionTracker, FenceKeySlot, MEMCMP, PageSlot, SplitCtx,
  node::{EMPTY_NEIGHBOR, NUM_SLOTS},
};
use ldb_type::PAGE_SIZE;

/// 把值写入预填充 `prefill` 的同尺寸槽后取回全部字节；隐式 padding 会保留
/// 预填充值，从而令 0xAA/0x55 两次构造的字节映像不相等。
///
/// # Safety
/// 槽先被 `write_bytes` 完整初始化为 `prefill`；随后 `write` 构造值。类型若
/// 含未初始化 padding，`write` 后读取该字节在抽象机上未定义 —— 这正是本审计
/// 要暴露的缺陷（修复后所有字节均为显式字段）。
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

fn fnv1a(bytes: &[u8]) -> u64 {
  let mut hash = 0xCBF2_9CE4_8422_2325_u64;
  for &byte in bytes {
    hash ^= u64::from(byte);
    hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
  }
  hash
}

#[test]
fn header_layout_matches_wire_format() {
  assert_eq!(size_of::<PageSlot>(), 16);
  assert_eq!(size_of::<FenceKeySlot>(), 8);
  assert_eq!(size_of::<ContentionTracker>(), 12);
  assert_eq!(size_of::<BTreeNodeHeader>(), 232);
  assert_eq!(align_of::<BTreeNodeHeader>(), 8);
  assert_eq!(size_of::<BTreeNode>(), PAGE_SIZE);
  assert_eq!(align_of::<BTreeNode>(), PAGE_SIZE);

  assert_eq!(offset_of!(BTreeNodeHeader, page), 0);
  assert_eq!(offset_of!(BTreeNodeHeader, right_most_child), 104);
  assert_eq!(offset_of!(BTreeNodeHeader, prev_leaf_node), 112);
  assert_eq!(offset_of!(BTreeNodeHeader, lower_fence), 120);
  assert_eq!(offset_of!(BTreeNodeHeader, upper_fence), 128);
  assert_eq!(offset_of!(BTreeNodeHeader, is_leaf), 136);
  assert_eq!(offset_of!(BTreeNodeHeader, count), 140);
  assert_eq!(offset_of!(BTreeNodeHeader, space_used), 144);
  assert_eq!(offset_of!(BTreeNodeHeader, data_offset), 148);
  assert_eq!(offset_of!(BTreeNodeHeader, prefix_len), 152);
  assert_eq!(offset_of!(BTreeNodeHeader, hints), 156);
  assert_eq!(offset_of!(BTreeNodeHeader, contention), 220);
  assert_eq!(offset_of!(BTreeNode, slots), 232);
  assert_eq!(NUM_SLOTS, 227);
}

/// 头部的 232 B 在 0xAA/0x55 两种栈垃圾下逐字节相等，且等于字面量期望：
/// 无任何隐式 padding（`is_leaf` 后 3 B 由 `_pad_leaf` 显式覆盖，尾部无填充）。
#[test]
fn header_has_no_implicit_padding_and_exact_bytes() {
  let a: [u8; 232] = unsafe { written_bytes(BTreeNodeHeader::new(true, 5), 0xAA) };
  let b: [u8; 232] = unsafe { written_bytes(BTreeNodeHeader::new(true, 5), 0x55) };
  assert_eq!(a, b, "两次构造的字节映像必须确定且相等");

  let mut expect = [0u8; 232];
  expect[0..2].copy_from_slice(&5u16.to_le_bytes()); // idx_id
  expect[104..112].copy_from_slice(&EMPTY_NEIGHBOR.to_le_bytes());
  expect[112..120].copy_from_slice(&EMPTY_NEIGHBOR.to_le_bytes());
  expect[136] = 1; // is_leaf
  expect[148..152].copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
  expect[220..224].copy_from_slice(&(-1i32).to_le_bytes());
  assert_eq!(a, expect, "头部字段布局/初值与线上格式不符");
}

/// 4096 B 节点全字节确定：构造后 `dirty=1`、`idx_id=index`、尾 232 B 全零；
/// 0xAA/0x55 两种栈垃圾下相等，写文件两次读回哈希一致。
#[test]
fn node_new_is_byte_deterministic_and_persists_stably() {
  let a: [u8; PAGE_SIZE] = unsafe { written_bytes(*BTreeNode::new(true, 7), 0xAA) };
  let b: [u8; PAGE_SIZE] = unsafe { written_bytes(*BTreeNode::new(true, 7), 0x55) };
  assert_eq!(a, b, "4096 B 构造必须逐字节确定");

  assert_eq!(a[0..2], 7u16.to_le_bytes());
  assert_eq!(a[96], 1, "构造即置 dirty");
  assert_eq!(a[136], 1, "is_leaf");
  assert_eq!(a[148..152], (PAGE_SIZE as u32).to_le_bytes());
  assert_eq!(a[220..224], (-1i32).to_le_bytes());
  assert_eq!(
    &a[3864..4096],
    &[0u8; 232],
    "槽位数组之后的 232 B 必须由构造清零"
  );

  let path: PathBuf = env::temp_dir().join(format!("ldb_node_r1_{}.bin", process::id()));
  let Ok(()) = fs::write(&path, a) else {
    panic!("首次写页失败: {}", path.display());
  };
  let Ok(first_bytes) = fs::read(&path) else {
    panic!("首次读页失败: {}", path.display());
  };
  let Ok(()) = fs::write(&path, b) else {
    panic!("二次写页失败: {}", path.display());
  };
  let Ok(second_bytes) = fs::read(&path) else {
    panic!("二次读页失败: {}", path.display());
  };
  let _ = fs::remove_file(&path);
  assert_eq!(fnv1a(&first_bytes), fnv1a(&second_bytes));
  assert_eq!(fnv1a(&first_bytes), fnv1a(&a));
}

fn mk_key(i: u64) -> Vec<u8> {
  format!("key-{i:08}").into_bytes()
}

/// 修改类操作必须保留 `PageHeader`（`p_gsn`/写历史/`idx_id`）。
///
/// 上游 ZLeanStore 的 `BTreeNode tmp(is_leaf, idx_id)` + `CopyNodeContent`
/// 会把 `p_gsn`/`wcnt`/`w_p_gsn` 清零；对 WAL/可见性语义是数据丢失，本版
/// 在 compactify / split / merge 中显式继承目标页的页头。
#[test]
fn mutations_preserve_page_header() {
  let mut n = BTreeNode::new(true, 9);
  n.set_fences(b"aa", b"az", MEMCMP);
  n.hdr.page.p_gsn = 0x1234_5678;
  n.hdr.page.prev_gcwrite = 77;
  n.hdr.page.wcnt = 5;
  n.hdr.page.w_p_gsn[3] = 4242;
  n.hdr.page.dirty = false;
  for i in 0..30u64 {
    n.insert_key_value(format!("aa{i:03}").as_bytes(), &[0u8; 8], MEMCMP);
  }
  n.compactify(MEMCMP);
  assert_eq!(n.hdr.page.p_gsn, 0x1234_5678);
  assert_eq!(n.hdr.page.prev_gcwrite, 77);
  assert_eq!(n.hdr.page.wcnt, 5);
  assert_eq!(n.hdr.page.w_p_gsn[3], 4242);
  assert_eq!(n.hdr.page.idx_id, 9);
  assert!(n.hdr.page.dirty, "修改后必须置脏");

  // split：左页保留自身页头，右页保留调用方初始化的页头
  let mut left = BTreeNode::new(true, 9);
  left.set_fences(b"aa", b"zz", MEMCMP);
  left.hdr.page.p_gsn = 111;
  for i in 0..24u64 {
    left.insert_key_value(&mk_key(i), &[0u8; 8], MEMCMP);
  }
  let sep = left.find_separator(false, MEMCMP);
  let mut sep_key = vec![0u8; sep.len as usize];
  left.get_separator_key(&mut sep_key, &sep);
  let mut right = BTreeNode::new(true, 9);
  right.hdr.page.p_gsn = 222;
  let mut parent = BTreeNode::new(false, 9);
  parent.set_fences(b"aa", b"zz", MEMCMP);
  parent.hdr.right_most_child = 1;
  let ctx = SplitCtx {
    parent: &mut parent,
    node_right: &mut right,
    left_pid: 1,
    right_pid: 2,
  };
  left.split_node(ctx, sep.slot, &sep_key, MEMCMP);
  assert_eq!(left.hdr.page.p_gsn, 111, "左页 GSN 被分裂清零");
  assert_eq!(right.hdr.page.p_gsn, 222, "右页 GSN 被分裂覆盖");

  // merge：right 是存活页，页头保持 right 自身
  let mut survivor = BTreeNode::new(true, 9);
  survivor.hdr.page.p_gsn = 333;
  assert!(left.merge_nodes(0, &mut parent, &mut survivor, MEMCMP));
  assert_eq!(survivor.hdr.page.p_gsn, 333);
}

/// 页号/索引 id 在构造时写入（上游 `PageHeader::idx_id = index`）。
#[test]
fn new_sets_index_id_and_dirty() {
  let leaf = BTreeNode::new(true, 42);
  assert_eq!(leaf.hdr.page.idx_id, 42);
  assert!(leaf.hdr.page.dirty);
  let inner = BTreeNode::new(false, 17);
  assert_eq!(inner.hdr.page.idx_id, 17);
  assert!(inner.is_inner());
}

/// `ContentionTracker` 的线上字节：`last_modified_pos = -1`、计数器 0。
#[test]
fn contention_tracker_wire_bytes() {
  let a: [u8; 12] = unsafe { written_bytes(ContentionTracker::default(), 0xAA) };
  let b: [u8; 12] = unsafe { written_bytes(ContentionTracker::default(), 0x55) };
  assert_eq!(a, b);
  let mut expect = [0u8; 12];
  expect[0..4].copy_from_slice(&(-1i32).to_le_bytes());
  assert_eq!(a, expect);
}
