//! B+Tree 节点核心回归：插入/查找/删除/分裂/合并（todo.md §4.1 防线 1）。

use ldb_node::{BTreeNode, MEMCMP};

fn mk_key(i: u32) -> Vec<u8> {
  format!("key-{i:08}").into_bytes()
}

#[test]
fn insert_lookup_remove_roundtrip() {
  let mut node = BTreeNode::new(true, 0);
  node.set_fences(b"", b"", MEMCMP);

  for i in 0..50u32 {
    assert!(node.insert_key_value(&mk_key(i), &i.to_le_bytes(), MEMCMP));
  }
  // 有序性
  for i in 1..50u32 {
    let a = node.get_key(i - 1);
    let b = node.get_key(i);
    assert!(a < b, "槽位有序: {a:?} !< {b:?}");
  }
  // 点查全部命中
  for i in 0..50u32 {
    let mut exact = false;
    let pos = node.lower_bound(&mk_key(i), &mut exact, MEMCMP);
    assert!(exact, "key {i} 应命中 @ {pos}");
    assert_eq!(node.get_payload(pos), i.to_le_bytes());
  }
  // 删除一半
  for i in (0..50u32).step_by(2) {
    assert!(node.remove_key(&mk_key(i), MEMCMP));
  }
  for i in (0..50u32).step_by(2) {
    let mut exact = false;
    node.lower_bound(&mk_key(i), &mut exact, MEMCMP);
    assert!(!exact, "key {i} 应已删除");
  }
  for i in (1..50u32).step_by(2) {
    let mut exact = false;
    node.lower_bound(&mk_key(i), &mut exact, MEMCMP);
    assert!(exact);
  }
  assert!(node.validate(MEMCMP).is_ok());
}

#[test]
fn compactify_frees_fragments() {
  let mut node = BTreeNode::new(true, 0);
  node.set_fences(b"", b"", MEMCMP);
  for i in 0..40u32 {
    assert!(node.insert_key_value(&mk_key(i), &[0; 40], MEMCMP));
  }
  for i in 0..40u32 {
    if i % 3 == 0 {
      assert!(node.remove_key(&mk_key(i), MEMCMP));
    }
  }
  let before = node.free_space();
  node.compactify(MEMCMP);
  let after = node.free_space();
  assert!(after >= before, "压缩后空闲不减: {after} < {before}");
  // 数据完好
  for i in 0..40u32 {
    let mut exact = false;
    node.lower_bound(&mk_key(i), &mut exact, MEMCMP);
    assert_eq!(exact, i % 3 != 0, "压缩后 key {i} 存在性错误");
  }
  assert!(node.validate(MEMCMP).is_ok());
}

#[test]
fn debug_strings_are_stable() {
  let mut node = BTreeNode::new(true, 0);
  node.set_fences(b"", b"", MEMCMP);
  assert!(node.insert_key_value(b"key-00000000", b"", MEMCMP));
  assert_eq!(
    node.key_to_string(0),
    "6b65792d3030303030303030",
    "key 十六进制字面量"
  );
  assert_eq!(
    node.to_debug_string(),
    "BTreeNode{kind:leaf, count:1, prefix_len:0, lower:∞, upper:∞}"
  );
}

#[test]
fn blob_key_child_lookup() {
  let mut node = BTreeNode::new(false, 0);
  node.set_fences(b"", b"", MEMCMP);
  for (k, child) in [(b"b", 1u64), (b"d", 2), (b"f", 3)] {
    assert!(node.insert_key_value(k, &child.to_le_bytes(), MEMCMP));
  }
  node.hdr.right_most_child = 4;

  let cmp = |a: &[u8], b: &[u8]| a.cmp(b);
  let mut pos = 0u32;
  assert_eq!(
    node.find_child_with_blob_key(ldb_simd::get_head(b"c"), b"c", &mut pos, cmp),
    2
  );
  assert_eq!(pos, 1);
  assert_eq!(
    node.find_child_with_blob_key(ldb_simd::get_head(b"f"), b"f", &mut pos, cmp),
    3
  );
  assert_eq!(pos, 2);
  assert_eq!(
    node.find_child_with_blob_key(ldb_simd::get_head(b"z"), b"z", &mut pos, cmp),
    4
  );
  assert_eq!(pos, 3);

  let mut exact = false;
  assert_eq!(
    node.lower_bound_with_blob_key(ldb_simd::get_head(b"d"), b"d", &mut exact, cmp),
    1
  );
  assert!(exact);
  assert_eq!(
    node.lower_bound_with_blob_key(ldb_simd::get_head(b"e"), b"e", &mut exact, cmp),
    2
  );
  assert!(!exact);
}
