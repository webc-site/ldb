//! 分裂/合并不变量（上游 `SplitNode`/`MergeNodes`；C++ 对拍见审查报告）。

use std::{cmp::Ordering, slice::from_raw_parts};

use ldb_node::{BTreeNode, MEMCMP, SplitCtx, node::EMPTY_NEIGHBOR};
use ldb_type::PAGE_SIZE;

const LEFT_PID: u64 = 10;
const RIGHT_PID: u64 = 20;

fn full_key(n: &BTreeNode, i: u32) -> Vec<u8> {
  let mut k = n.get_prefix().to_vec();
  k.extend_from_slice(n.get_key(i));
  k
}

fn keys_of(n: &BTreeNode) -> Vec<Vec<u8>> {
  (0..n.hdr.count).map(|i| full_key(n, i)).collect()
}

fn payloads_of(n: &BTreeNode) -> Vec<Vec<u8>> {
  (0..n.hdr.count)
    .map(|i| n.get_payload(i).to_vec())
    .collect()
}

fn build(inner: bool, count: u32) -> Box<BTreeNode> {
  let mut n = BTreeNode::new(!inner, 7);
  n.set_fences(b"aa", b"az", MEMCMP);
  for i in 0..count {
    let key = format!("aa{i:03}").into_bytes();
    let payload = (u64::from(i) + 100).to_le_bytes();
    assert!(n.insert_key_value(&key, &payload, MEMCMP));
  }
  if inner {
    n.hdr.right_most_child = 9000;
  }
  n
}

fn split(
  mut left: Box<BTreeNode>,
  bias: bool,
) -> (Box<BTreeNode>, Box<BTreeNode>, Box<BTreeNode>, Vec<u8>) {
  let info = left.find_separator(bias, MEMCMP);
  let mut sep = vec![0u8; info.len as usize];
  left.get_separator_key(&mut sep, &info);
  let mut right = BTreeNode::new(left.hdr.is_leaf, 7);
  let mut parent = BTreeNode::new(false, 7);
  parent.set_fences(b"aa", b"az", MEMCMP);
  parent.hdr.right_most_child = LEFT_PID;
  let ctx = SplitCtx {
    parent: &mut parent,
    node_right: &mut right,
    left_pid: LEFT_PID,
    right_pid: RIGHT_PID,
  };
  left.split_node(ctx, info.slot, &sep, MEMCMP);
  (left, right, parent, sep)
}

#[test]
fn leaf_split_and_merge_preserve_records() {
  for count in 3..=40u32 {
    for bias in [false, true] {
      let left = build(false, count);
      assert_eq!(left.hdr.prefix_len, 1, "fences aa/az 公共前缀 1");
      let orig_keys = keys_of(&left);
      let orig_payloads = payloads_of(&left);
      let (mut left, mut right, mut parent, sep) = split(left, bias);

      assert_eq!(
        left.hdr.count + right.hdr.count,
        count,
        "count={count} bias={bias}"
      );
      let mut joined = keys_of(&left);
      joined.extend(keys_of(&right));
      assert_eq!(joined, orig_keys, "叶分裂必须保留全部 key 与顺序");
      let mut joined_p = payloads_of(&left);
      joined_p.extend(payloads_of(&right));
      assert_eq!(joined_p, orig_payloads);
      assert_eq!(left.get_lower_fence(), b"aa");
      assert_eq!(left.get_upper_fence(), sep.as_slice());
      assert_eq!(right.get_lower_fence(), sep.as_slice());
      assert_eq!(right.get_upper_fence(), b"az");
      assert_eq!(left.hdr.right_most_child, RIGHT_PID);
      assert_eq!(right.hdr.prev_leaf_node, LEFT_PID);
      assert_eq!(right.hdr.right_most_child, EMPTY_NEIGHBOR);
      assert_eq!(parent.hdr.count, 1);
      assert_eq!(parent.get_child(0), LEFT_PID);
      assert_eq!(parent.hdr.right_most_child, RIGHT_PID);
      assert_eq!(full_key(&parent, 0), sep);
      assert!(left.validate(MEMCMP).is_ok());
      assert!(right.validate(MEMCMP).is_ok());
      assert!(parent.validate(MEMCMP).is_ok());

      for k in keys_of(&left) {
        assert!(
          (MEMCMP.f)(&k, &sep) != Ordering::Greater,
          "左节点 key 必须 <= 分隔键"
        );
      }
      for k in keys_of(&right) {
        assert!(
          (MEMCMP.f)(&k, &sep) == Ordering::Greater,
          "右节点 key 必须 > 分隔键"
        );
      }

      assert!(left.merge_nodes(0, &mut parent, &mut right, MEMCMP));
      assert_eq!(right.hdr.count, count);
      assert_eq!(keys_of(&right), orig_keys, "合并必须还原全部 key");
      assert_eq!(payloads_of(&right), orig_payloads);
      assert_eq!(right.get_lower_fence(), b"aa");
      assert_eq!(right.get_upper_fence(), b"az");
      assert_eq!(parent.hdr.count, 0);
      assert_eq!(parent.hdr.right_most_child, RIGHT_PID);
      assert!(right.validate(MEMCMP).is_ok());
      assert!(parent.validate(MEMCMP).is_ok());
    }
  }
}

#[test]
fn inner_split_moves_separator_up_and_merge_puts_it_back() {
  for count in 4..=30u32 {
    let mut left = build(true, count);
    let orig_keys = keys_of(&left);
    let orig_payloads = payloads_of(&left);
    let orig_right_most = left.hdr.right_most_child;
    let info = left.find_separator(false, MEMCMP);
    assert_eq!(info.slot, count / 2);
    let mut sep = vec![0u8; info.len as usize];
    left.get_separator_key(&mut sep, &info);
    let (mut left, mut right, mut parent) = {
      let mut right = BTreeNode::new(false, 7);
      let mut parent = BTreeNode::new(false, 7);
      parent.set_fences(b"aa", b"az", MEMCMP);
      parent.hdr.right_most_child = LEFT_PID;
      let ctx = SplitCtx {
        parent: &mut parent,
        node_right: &mut right,
        left_pid: LEFT_PID,
        right_pid: RIGHT_PID,
      };
      left.split_node(ctx, info.slot, &sep, MEMCMP);
      (left, right, parent)
    };
    let slot = info.slot as usize;
    assert_eq!(left.hdr.count as usize, slot);
    assert_eq!(right.hdr.count as usize, count as usize - slot - 1);
    assert_eq!(keys_of(&left), orig_keys[..slot]);
    assert_eq!(keys_of(&right), orig_keys[slot + 1..]);
    let mut child = [0u8; 8];
    child.copy_from_slice(&orig_payloads[slot]);
    assert_eq!(left.hdr.right_most_child, u64::from_le_bytes(child));
    assert_eq!(right.hdr.right_most_child, orig_right_most);
    assert_eq!(parent.hdr.count, 1);
    assert_eq!(full_key(&parent, 0), orig_keys[slot]);
    assert!(left.validate(MEMCMP).is_ok());
    assert!(right.validate(MEMCMP).is_ok());
    assert!(parent.validate(MEMCMP).is_ok());

    assert!(left.merge_nodes(0, &mut parent, &mut right, MEMCMP));
    assert_eq!(right.hdr.count, count);
    assert_eq!(keys_of(&right), orig_keys, "内节点合并必须复原分隔键");
    assert_eq!(payloads_of(&right), orig_payloads);
    assert_eq!(right.hdr.right_most_child, orig_right_most);
    assert_eq!(parent.hdr.count, 0);
    assert!(right.validate(MEMCMP).is_ok());
  }
}

#[test]
fn merge_rejects_when_over_page_and_leaves_target_untouched() {
  let mut left = BTreeNode::new(true, 7);
  left.set_fences(b"", b"", MEMCMP);
  let mut right = BTreeNode::new(true, 7);
  right.set_fences(b"", b"", MEMCMP);
  right.hdr.page.p_gsn = 777;
  for i in 0..90u32 {
    assert!(left.insert_key_value(format!("l{i:03}").as_bytes(), &[1; 4], MEMCMP));
    assert!(right.insert_key_value(format!("r{i:03}").as_bytes(), &[2; 4], MEMCMP));
  }
  let mut parent = BTreeNode::new(false, 7);
  parent.set_fences(b"", b"", MEMCMP);
  assert!(parent.insert_key_value(b"m", &LEFT_PID.to_le_bytes(), MEMCMP));
  parent.hdr.right_most_child = RIGHT_PID;

  let before = {
    let mut b = [0u8; PAGE_SIZE];
    // SAFETY: BTreeNode 恰为一页且已初始化
    b.copy_from_slice(unsafe { from_raw_parts(right.ptr(), PAGE_SIZE) });
    b
  };
  assert!(!left.merge_nodes(0, &mut parent, &mut right, MEMCMP));
  let after = {
    let mut b = [0u8; PAGE_SIZE];
    b.copy_from_slice(unsafe { from_raw_parts(right.ptr(), PAGE_SIZE) });
    b
  };
  assert_eq!(before, after, "合并失败不得改动目标节点");
  assert_eq!(right.hdr.page.p_gsn, 777);
  assert_eq!(parent.hdr.count, 1);
}
