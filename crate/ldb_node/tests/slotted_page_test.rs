//! slotted page 边界与对抗性输入（`ldb_page` 轮次登记的跨 crate P1）。

use std::{
  ptr::{copy_nonoverlapping, from_mut},
  slice::from_raw_parts,
};

use ldb_node::{BTreeNode, MEMCMP, node::NUM_SLOTS};
use ldb_page::Page;
use ldb_type::PAGE_SIZE;

fn bytes_of(n: &BTreeNode) -> [u8; PAGE_SIZE] {
  let mut out = [0u8; PAGE_SIZE];
  // SAFETY: BTreeNode 恰为 PAGE_SIZE 字节且已初始化。
  out.copy_from_slice(unsafe { from_raw_parts(n.ptr(), PAGE_SIZE) });
  out
}

fn put_u32(b: &mut [u8; PAGE_SIZE], off: usize, v: u32) {
  b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn get_u32(b: &[u8; PAGE_SIZE], off: usize) -> u32 {
  u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn head_of(key: &[u8]) -> [u8; 4] {
  let mut h = [0u8; 4];
  let n = key.len().min(4);
  h[..n].copy_from_slice(&key[..n]);
  h
}

/// 写一个槽位（16 B packed：offset/key_len/payload_len/head）
fn put_slot(b: &mut [u8; PAGE_SIZE], slot: usize, offset: u32, key: &[u8], payload_len: u32) {
  let base = 232 + slot * 16;
  put_u32(b, base, offset);
  put_u32(b, base + 4, key.len() as u32);
  put_u32(b, base + 8, payload_len);
  b[base + 12..base + 16].copy_from_slice(&head_of(key));
}

fn assert_corrupt(bytes: &[u8; PAGE_SIZE]) {
  assert!(
    BTreeNode::validate_bytes(bytes, MEMCMP).is_err(),
    "畸形页必须返回 Err"
  );
}

#[test]
fn empty_node_boundaries() {
  let mut n = BTreeNode::new(true, 0);
  assert_eq!(n.hdr.count, 0);
  assert_eq!(n.free_space(), (PAGE_SIZE - 232) as u32);
  assert_eq!(n.free_space_after_compaction(), (PAGE_SIZE - 232) as u32);
  assert!(n.get_prefix().is_empty());

  let mut exact = true;
  assert_eq!(n.lower_bound(b"", &mut exact, MEMCMP), 0);
  assert!(!exact);
  assert_eq!(n.lower_bound(b"anything", &mut exact, MEMCMP), 0);
  assert!(!exact);
  assert!(!n.remove_key(b"", MEMCMP));
  assert!(n.validate(MEMCMP).is_ok());

  // 空 fence：offset 落到页尾，len=0 表示无穷界
  n.set_fences(b"", b"", MEMCMP);
  assert_eq!(n.hdr.lower_fence.len, 0);
  assert_eq!(n.hdr.lower_fence.offset, PAGE_SIZE as u32);
  assert!(n.hdr.is_lower_fence_infinity());
  assert!(n.hdr.is_upper_fence_infinity());
  assert!(n.validate(MEMCMP).is_ok());
}

#[test]
fn zero_length_key_and_payload() {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  assert!(n.insert_key_value(b"", b"", MEMCMP));
  assert_eq!(n.hdr.count, 1);
  let mut exact = false;
  assert_eq!(n.lower_bound(b"", &mut exact, MEMCMP), 0);
  assert!(exact);
  assert!(n.get_key(0).is_empty());
  assert!(n.get_payload(0).is_empty());
  assert!(n.validate(MEMCMP).is_ok());

  assert!(n.insert_key_value(b"", b"v", MEMCMP), "同键变长 upsert");
  assert_eq!(n.hdr.count, 1);
  assert_eq!(n.get_payload(0), b"v");
  assert!(n.validate(MEMCMP).is_ok());

  assert!(n.insert_key_value(b"a", b"", MEMCMP));
  assert_eq!(n.hdr.count, 2);
  assert_eq!(n.get_payload(1), b"");
  assert!(n.validate(MEMCMP).is_ok());

  assert!(n.remove_key(b"", MEMCMP));
  assert!(!n.remove_key(b"", MEMCMP));
  assert_eq!(n.hdr.count, 1);
  assert!(n.validate(MEMCMP).is_ok());
}

/// 1 B 唯一键 × 227 = 槽位数组容量恰好用满；第 228 个必须失败而不是越界。
#[test]
fn full_page_stops_at_slot_capacity() {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  for i in 0..NUM_SLOTS {
    let key = [i as u8];
    assert!(n.insert_key_value(&key, b"", MEMCMP), "第 {i} 个键应可插入");
  }
  assert_eq!(n.hdr.count as usize, NUM_SLOTS);
  assert_eq!(n.free_space_after_compaction(), 5);
  assert!(!n.has_space_for_kv(1, 0), "满页必须预判为无空间");
  // count=227 > 2*HINT_COUNT：逐个点查必须走 hint 窄化路径且全部命中。
  // 这也钉死 `search_hint` 第 3 参传的是槽位总数 count（而非收窄后的 upper）。
  for i in 0..NUM_SLOTS {
    let key = [i as u8];
    let mut exact = false;
    let pos = n.lower_bound(&key, &mut exact, MEMCMP);
    assert!(exact, "满页点查 {i} 未命中");
    assert_eq!(pos, i as u32);
  }
  assert!(
    !n.insert_key_value(&[0xE3], b"", MEMCMP),
    "第 228 个键必须失败"
  );
  assert_eq!(n.hdr.count as usize, NUM_SLOTS);
  assert!(n.validate(MEMCMP).is_ok());
}

/// 单条记录上限：16 B 槽位 + 3848 B key 恰好铺满数据区；再多 1 B 必须失败。
/// 入参长度超过整页：必须返回 false 而不是 u32 截断后越界写。
#[test]
fn oversized_input_is_rejected() {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  assert!(!n.insert_key_value(&vec![0u8; PAGE_SIZE + 1], b"", MEMCMP));
  assert!(!n.insert_key_value(b"", &vec![0u8; PAGE_SIZE + 1], MEMCMP));
  assert_eq!(n.hdr.count, 0);
  assert!(!n.has_space_for_kv(u32::MAX, u32::MAX));
  assert_eq!(n.space_required_for_kv(u32::MAX, u32::MAX), u32::MAX);
  assert!(n.validate(MEMCMP).is_ok());
}

#[test]
fn maximum_record_boundary() {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  let big = vec![0x5A; PAGE_SIZE - 232 - 16];
  assert!(n.insert_key_value(&big, b"", MEMCMP));
  assert_eq!(n.hdr.data_offset, 232 + 16);
  assert_eq!(n.get_key(0).len(), big.len());
  assert_eq!(n.get_key(0), &big[..]);
  assert!(n.validate(MEMCMP).is_ok());
  assert!(!n.insert_key_value(b"x", b"", MEMCMP));
  assert!(n.validate(MEMCMP).is_ok());
}

/// 合法但极端：单槽恰好占满整页（16 B 槽 + 3848 B 数据）。
///
/// 覆盖「payload 最大」「键 0 长 + payload 最大」「4 B 键 + payload 余量最大」
/// 三条边界；与 `maximum_record_boundary`（最大键 + 空 payload）互补。
#[test]
fn extreme_record_sizes_are_valid() {
  let max_data = PAGE_SIZE - 232 - 16;

  // 空键 + 最大 payload
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  let payload = vec![0xA7; max_data];
  assert!(n.insert_key_value(b"", &payload, MEMCMP));
  assert_eq!(n.hdr.data_offset, 232 + 16);
  assert_eq!(n.get_key(0), b"");
  assert_eq!(n.get_payload(0), &payload[..]);
  assert!(n.validate(MEMCMP).is_ok());
  assert!(!n.insert_key_value(b"x", b"", MEMCMP));

  // 4 B 键 + 3844 B payload：key+payload 仍恰好填满 3848 B
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  let payload = vec![0x5C; max_data - 4];
  assert!(n.insert_key_value(b"kkkk", &payload, MEMCMP));
  assert_eq!(n.hdr.data_offset, 232 + 16);
  assert_eq!(n.get_key(0), b"kkkk");
  assert_eq!(n.get_payload(0), &payload[..]);
  assert!(n.validate(MEMCMP).is_ok());
}

/// 合法但极端：两个 fence 恰好占满数据区（`data_offset == slots_end == 232`），
/// `count == 0`。`validate_bytes` 必须接受，且任何插入因 `free_space == 0` 失败。
#[test]
fn fences_fill_data_area_exactly() {
  let mut n = BTreeNode::new(true, 0);
  let lower = vec![0x10; 1932];
  let upper = vec![0x20; 1932];
  n.set_fences(&lower, &upper, MEMCMP);
  assert_eq!(n.hdr.prefix_len, 0);
  assert_eq!(n.hdr.data_offset, 232);
  assert_eq!(n.hdr.space_used, 3864);
  assert_eq!(n.free_space(), 0);
  assert_eq!(n.free_space_after_compaction(), 0);
  assert!(n.validate(MEMCMP).is_ok());
  assert!(!n.insert_key_value(b"x", b"", MEMCMP));
  assert!(n.validate(MEMCMP).is_ok());
}

/// 变长 upsert 删除尾槽后重插（`copy_within(s..c, s+1)` 的 s==c 边界）。
#[test]
fn upsert_variable_length_and_tail_slot() {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  assert!(n.insert_key_value(b"a", &[1, 2, 3, 4], MEMCMP));
  assert!(n.insert_key_value(b"z", &[5, 6, 7, 8], MEMCMP));
  // 同长原位覆写路径（4B -> 4B）
  assert!(n.insert_key_value(b"z", &[1, 2, 3, 4], MEMCMP));
  assert_eq!(n.hdr.count, 2);
  assert_eq!(n.get_payload(1), &[1u8, 2, 3, 4]);
  // 变长路径（4B -> 8B -> 0B）
  assert!(n.insert_key_value(b"z", &[9; 8], MEMCMP));
  assert_eq!(n.get_payload(1), &[9u8; 8]);
  assert!(n.validate(MEMCMP).is_ok());

  assert!(n.insert_key_value(b"z", b"", MEMCMP));
  assert_eq!(n.hdr.count, 2);
  assert_eq!(n.get_payload(1), b"");
  assert!(n.validate(MEMCMP).is_ok());

  assert!(n.insert_key_value(b"a", &[7; 32], MEMCMP));
  assert_eq!(n.hdr.count, 2);
  assert_eq!(n.get_key(0), b"a");
  assert_eq!(n.get_payload(0), &[7u8; 32]);
  assert!(n.validate(MEMCMP).is_ok());
}

/// 变长 upsert 放不下时：返回 false 且**保留旧值**（不得先删旧槽再失败）。
#[test]
fn failed_upsert_preserves_old_value() {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  let big = vec![0x11; 1000];
  assert!(n.insert_key_value(b"a", &big, MEMCMP));
  assert!(n.insert_key_value(b"b", &big, MEMCMP));
  let count = n.hdr.count;
  assert!(!n.insert_key_value(b"b", &vec![0x22; 3000], MEMCMP));
  assert_eq!(n.hdr.count, count, "失败 upsert 不得删除旧槽");
  assert_eq!(n.get_payload(1), &big[..], "旧 payload 必须保留");
  assert_eq!(n.get_key(0), b"a");
  assert!(n.validate(MEMCMP).is_ok());
}

#[test]
fn compaction_preserves_records_and_space() {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"", b"", MEMCMP);
  for i in 0..100u64 {
    let key = format!("k{i:03}").into_bytes();
    assert!(n.insert_key_value(&key, &[i as u8; 16], MEMCMP));
  }
  for i in (0..100u64).step_by(3) {
    assert!(n.remove_key(format!("k{i:03}").as_bytes(), MEMCMP));
  }
  let before = n.free_space_after_compaction();
  let after_compaction_before = before;
  n.compactify(MEMCMP);
  assert_eq!(n.free_space(), after_compaction_before);
  assert_eq!(n.free_space_after_compaction(), before);
  assert!(n.validate(MEMCMP).is_ok());
  for i in 0..100u64 {
    let key = format!("k{i:03}").into_bytes();
    let mut exact = false;
    let pos = n.lower_bound(&key, &mut exact, MEMCMP);
    assert_eq!(exact, i % 3 != 0, "key {i} 存在性错误");
    if exact {
      assert_eq!(n.get_payload(pos), &[i as u8; 16]);
    }
  }
}

/// 恶意槽头：越界/重叠/计数超限/非法 bool/坏 head/未排序 → Err，绝不 panic。
#[test]
fn malicious_headers_are_err_not_panic() {
  let mut valid = BTreeNode::new(true, 0);
  valid.set_fences(b"", b"", MEMCMP);
  for k in [b"abc".as_slice(), b"bcd", b"cde"] {
    assert!(valid.insert_key_value(k, b"", MEMCMP));
  }
  let base = bytes_of(&valid);
  assert!(BTreeNode::validate_bytes(&base, MEMCMP).is_ok());

  let mut b = base;
  put_u32(&mut b, 140, NUM_SLOTS as u32 + 1);
  assert_corrupt(&b);

  let mut b = base;
  put_u32(&mut b, 148, 100); // data_offset < slots_end
  assert_corrupt(&b);

  let mut b = base;
  put_u32(&mut b, 144, 4000); // space_used 超过数据区
  assert_corrupt(&b);

  // space_used 少于实际存活字节：仍在 `PAGE_SIZE - data_offset` 上界内，
  // 只有精确账能抓住；漏掉会让 compactify 估算出过多空间并在 release 越界写。
  let mut b = base;
  assert_eq!(get_u32(&b, 144), 9, "3 个 3 B key 的实际 space_used");
  put_u32(&mut b, 144, 8);
  assert_corrupt(&b);

  let mut b = base;
  put_u32(&mut b, 232, 4090); // slot0 offset 越界
  put_u32(&mut b, 236, 100);
  assert_corrupt(&b);

  let mut b = base;
  b[136] = 2; // invalid bool
  assert_corrupt(&b);
  let mut b = base;
  b[96] = 2; // dirty
  assert_corrupt(&b);
  let mut b = base;
  b[97] = 2; // gc_read
  assert_corrupt(&b);
  let mut b = base;
  b[137] = 1; // 保留位
  assert_corrupt(&b);

  let mut b = base;
  b[232 + 12] ^= 0xFF; // head 与 key 不符
  assert_corrupt(&b);

  let mut b = base;
  let s0 = b[232..248].to_vec();
  let s1 = b[248..264].to_vec();
  b[232..248].copy_from_slice(&s1);
  b[248..264].copy_from_slice(&s0);
  assert_corrupt(&b); // 槽位非严格递增

  let mut b = base;
  put_u32(&mut b, 120, 4090); // fence offset+len 越界
  put_u32(&mut b, 124, 100);
  assert_corrupt(&b);

  let mut b = base;
  put_u32(&mut b, 152, 100); // prefix_len 超 fence 长度
  assert_corrupt(&b);

  // 两个非空 fence 必须严格递增：空节点没有键可交叉验证，lower >= upper 的
  // 路由区间为空却会通过其余检查（上游 SetFenses 的区间语义）。
  let mut inverted = BTreeNode::new(true, 0);
  inverted.set_fences(b"zz", b"aa", MEMCMP);
  assert_corrupt(&bytes_of(&inverted));
  let mut equal = BTreeNode::new(true, 0);
  equal.set_fences(b"aa", b"aa", MEMCMP);
  assert_corrupt(&bytes_of(&equal));

  // 内节点 payload 短于 8 B：validate 必须拒绝，否则 get_child 的 p[..8] panic
  let mut inner = BTreeNode::new(false, 0);
  inner.set_fences(b"", b"", MEMCMP);
  assert!(inner.insert_key_value(b"a", b"", MEMCMP));
  inner.hdr.right_most_child = 1;
  assert_corrupt(&bytes_of(&inner));

  // 重叠：两个合法有序的 key 被放到相交的字节区间。
  // 三键页的数据布局（插入序 abc,bcd,cde）：cde@4087、bcd@4090、abc@4093。
  // 保留 abc@4093 与 bcd 的前一字节起读出的 "dab"@4092，区间 [4093,4096)
  // 与 [4092,4095) 相交，且 "abc" < "dab" 通过有序性检查。
  let mut b = base;
  put_u32(&mut b, 140, 2);
  put_u32(&mut b, 144, 6);
  put_u32(&mut b, 148, 4090);
  assert_eq!(&b[4092..4095], b"dab");
  put_slot(&mut b, 0, 4093, b"abc", 0);
  put_slot(&mut b, 1, 4092, b"dab", 0);
  assert_corrupt(&b);

  // fence 越界：full key 必须 > lower 且 <= upper
  let mut fenced = BTreeNode::new(true, 0);
  fenced.set_fences(b"aa", b"az", MEMCMP);
  assert!(fenced.insert_key_value(b"aa0", b"", MEMCMP));
  let base = bytes_of(&fenced);
  assert!(BTreeNode::validate_bytes(&base, MEMCMP).is_ok());
  let mut b = base;
  let off = get_u32(&b, 232) as usize;
  b[off..off + 2].copy_from_slice(b"zz"); // full = "azz" > upper "az"
  b[232 + 12..232 + 16].copy_from_slice(&head_of(b"zz"));
  assert_corrupt(&b);

  let mut b = base;
  put_u32(&mut b, 232 + 4, 0); // 去掉前缀后 key 长度 0 -> full = "a" < lower "aa"
  put_u32(&mut b, 144, 0);
  put_u32(&mut b, 148, off as u32);
  b[232 + 12..232 + 16].copy_from_slice(&[0; 4]);
  assert_corrupt(&b);

  // 5000 个随机页：只要求不 panic（绝大多数返回 Err）
  let mut state = 0x1234_5678_9ABC_DEF0u64;
  for _ in 0..5_000 {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    let mut raw = [0u8; PAGE_SIZE];
    for chunk in raw.as_chunks_mut::<8>().0 {
      *chunk = state.to_le_bytes();
      state = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    }
    let _ = BTreeNode::validate_bytes(&raw, MEMCMP);
  }
}

/// 内节点槽位 payload 是 8 B 子页号：0..7 B 与 9 B 的畸形页必须 Err
/// （`get_child` 只解释前 8 B），恰好 8 B 必须 Ok。覆盖 count=1/2/8。
/// 叶节点 payload 长度任意（0 B 合法，与上游一致）。
#[test]
fn inner_slot_payload_must_equal_page_id() {
  fn page(count: usize, plen: u32) -> [u8; PAGE_SIZE] {
    let mut b = [0u8; PAGE_SIZE];
    let total = count as u32 * (1 + plen);
    let data_offset = PAGE_SIZE as u32 - total;
    put_u32(&mut b, 140, count as u32);
    put_u32(&mut b, 144, total);
    put_u32(&mut b, 148, data_offset);
    let mut off = data_offset;
    for i in 0..count {
      let key = [i as u8 + 1];
      put_slot(&mut b, i, off, &key, plen);
      b[off as usize] = key[0];
      for z in 0..plen as usize {
        b[off as usize + 1 + z] = 0xAB;
      }
      off += 1 + plen;
    }
    b
  }

  for count in [1usize, 2, 8] {
    for plen in 0..8u32 {
      let err =
        BTreeNode::validate_bytes(&page(count, plen), MEMCMP).expect_err("短 payload 必须 Err");
      assert!(
        format!("{err}").contains("payload"),
        "count={count} plen={plen} 的错误信息应指出 payload: {err}"
      );
    }
    assert!(
      BTreeNode::validate_bytes(&page(count, 8), MEMCMP).is_ok(),
      "count={count} 的 8 B 子页号页必须合法"
    );
    assert!(
      BTreeNode::validate_bytes(&page(count, 9), MEMCMP).is_err(),
      "count={count} 的多余字节 payload 必须 Err"
    );
  }

  // 叶节点 payload 0 B 合法
  let mut leaf = BTreeNode::new(true, 0);
  leaf.set_fences(b"", b"", MEMCMP);
  assert!(leaf.insert_key_value(b"k", b"", MEMCMP));
  assert!(leaf.validate(MEMCMP).is_ok());
}

/// `from_page_mut` 先校验后 cast：合法页 Ok，畸形页 Err。
#[test]
fn checked_page_cast() {
  let mut n = BTreeNode::new(true, 3);
  n.set_fences(b"", b"", MEMCMP);
  assert!(n.insert_key_value(b"k", b"v", MEMCMP));
  let raw = bytes_of(&n);

  let mut page = Page::new();
  // SAFETY: Page 恰为 4096 B，raw 也是 4096 B；测试独占。
  unsafe {
    copy_nonoverlapping(raw.as_ptr(), from_mut(&mut page).cast::<u8>(), PAGE_SIZE);
  }
  let node = BTreeNode::from_page_mut(&mut page, MEMCMP);
  assert!(node.is_ok());
  let node = BTreeNode::from_page(&page, MEMCMP);
  assert!(node.is_ok());

  let mut bad = raw;
  put_u32(&mut bad, 140, NUM_SLOTS as u32 + 1);
  let mut page = Page::new();
  // SAFETY: 同上。
  unsafe {
    copy_nonoverlapping(bad.as_ptr(), from_mut(&mut page).cast::<u8>(), PAGE_SIZE);
  }
  assert!(BTreeNode::from_page_mut(&mut page, MEMCMP).is_err());
  assert!(BTreeNode::from_page(&page, MEMCMP).is_err());
}
