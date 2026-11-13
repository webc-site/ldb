//! SMO 协议回归：split/merge/XMerge 的父指针、叶链、fence 不变量与交错序列。
//!
//! 覆盖审查要求：父指针丢失 / 重复分裂 / 丢记录 / 兄弟不一致。

mod common;

use std::collections::BTreeMap;

use common::{check_tree, full_key, mk};
use ldb_btree::BTree;

/// 确定性 xorshift（测试用）
fn next_rand(state: &mut u64) -> u64 {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  *state
}

fn key(i: u32) -> Vec<u8> {
  format!("key-{i:06}").into_bytes()
}

/// 深树：数千键强制多级分裂（内节点分裂路径），每个键可查且结构不变量成立。
#[compio::test]
async fn deep_tree_multi_level_split() {
  let mut t = mk();
  let n = 4000u32;
  for i in 0..n {
    t.insert(&key(i), &i.to_le_bytes()).await.unwrap();
  }
  assert!(t.count_entries().await.unwrap() == n as u64);
  let mut out = Vec::new();
  for i in 0..n {
    assert!(t.look_up(&key(i), &mut out).await.unwrap(), "i={i} 丢失");
    assert_eq!(out, i.to_le_bytes(), "i={i} 值不符");
  }
  // 结构不变量：父指针唯一 / 叶链完整 / fence 升序 / 路由一致
  check_tree(&t);
}

/// 多级结构 + 全键可查：内节点分裂形态正确性由 `long_separator_split`（长分隔键
/// 压低内节扇出，强制非根内节点分裂）覆盖；此处验证多内节点的路由/不变量。
#[compio::test]
async fn multi_level_lookup_invariants() {
  let mut t = mk();
  // 长键压低扇出：单叶约 19 条、单内节约 19 分隔键 → 需多级内节点
  let big = |i: u32| format!("{i:06}-{}", "z".repeat(180)).into_bytes();
  for i in 0..3000u32 {
    t.insert(&big(i), &i.to_le_bytes()).await.unwrap();
  }
  // 全键可查（内节点分裂若产出叶页 → 路由进入伪叶 → 键丢失/类型错乱）
  let mut out = Vec::new();
  for i in 0..3000u32 {
    assert!(t.look_up(&big(i), &mut out).await.unwrap(), "i={i} 丢失");
    assert_eq!(out, i.to_le_bytes(), "i={i} 值不符");
  }
  let nodes = t.nodes.read();
  let inner_cnt = nodes.values().filter(|n| !n.hdr.is_leaf).count();
  assert!(
    inner_cnt >= 2,
    "长键 3000 条应产生 ≥2 内节点，实得 {inner_cnt}"
  );
  drop(nodes);
  check_tree(&t);
}

/// 递减插入（append_bias 反向）：叶链/fence 仍升序，全键可查。
#[compio::test]
async fn descending_insert_invariants() {
  let mut t = mk();
  let n = 1500u32;
  for i in (0..n).rev() {
    t.insert(&key(i), &i.to_le_bytes()).await.unwrap();
  }
  let mut out = Vec::new();
  for i in 0..n {
    assert!(t.look_up(&key(i), &mut out).await.unwrap(), "i={i}");
  }
  check_tree(&t);
}

/// 插入/删除交错 + 影子模型：每 512 步结构不变量 + 每步对账。
#[compio::test]
async fn interleaved_insert_remove_shadow() {
  let mut t = mk();
  let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
  let mut rng = 0xC0FFEE_u64;
  for step in 0..20_000u32 {
    let r = next_rand(&mut rng);
    let i = (r % 2000) as u32;
    let k = key(i);
    if r & 1 == 0 {
      let v = (r as u32).to_le_bytes();
      t.insert(&k, &v).await.unwrap();
      model.insert(k, v.to_vec());
    } else {
      let got = t.remove(&k).await.unwrap();
      let expect = model.remove(&k).is_some();
      assert_eq!(got, expect, "step {step} 删除返回不符 key={i}");
    }
    if step % 512 == 0 {
      check_tree(&t);
      assert_eq!(
        t.count_entries().await.unwrap(),
        model.len() as u64,
        "step {step} 条目数不符"
      );
    }
  }
  check_tree(&t);
  // 全量对账
  let mut out = Vec::new();
  for (k, v) in &model {
    assert!(t.look_up(k, &mut out).await.unwrap(), "缺失 {k:?}");
    assert_eq!(&out, v, "值不符 {k:?}");
  }
  // 扫描结果 == 模型
  let mut scanned = Vec::new();
  t.scan_ascending(b"", b"", |k, v| {
    scanned.push((k.to_vec(), v.to_vec()));
    true
  })
  .await
  .unwrap();
  let expect: Vec<(Vec<u8>, Vec<u8>)> = model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
  assert_eq!(scanned, expect, "扫描与模型不一致");
}

/// 直接调用 `merge_left_into_right`：右节点原有记录不得丢失（旧实现清空右 count）。
#[compio::test]
async fn merge_left_into_right_preserves_both_sides() {
  use ldb_node::{BTreeNode, MEMCMP, node::EMPTY_NEIGHBOR};

  let mut t = mk();
  // 手工构造 root(内) + 两叶 A(1) / B(2)：root 分隔键 = A.upper_fence "m"，右子 = B
  {
    let mut nodes = t.nodes.write();
    nodes.clear();
    let mut root = BTreeNode::new(false, 0);
    root.set_fences(b"", b"", MEMCMP);
    let mut a = BTreeNode::new(true, 0);
    a.set_fences(b"", b"m", MEMCMP);
    assert!(a.insert_key_value(b"a", b"1", MEMCMP));
    assert!(a.insert_key_value(b"b", b"2", MEMCMP));
    a.hdr.prev_leaf_node = EMPTY_NEIGHBOR;
    a.hdr.right_most_child = 2;
    let mut b = BTreeNode::new(true, 0);
    b.set_fences(b"m", b"", MEMCMP);
    assert!(b.insert_key_value(b"n", b"3", MEMCMP));
    assert!(b.insert_key_value(b"o", b"4", MEMCMP));
    b.hdr.prev_leaf_node = 1;
    b.hdr.right_most_child = EMPTY_NEIGHBOR;
    assert!(root.insert_key_value(b"m", &1u64.to_le_bytes(), MEMCMP));
    root.hdr.right_most_child = 2;
    nodes.insert(0, root);
    nodes.insert(1, a);
    nodes.insert(2, b);
  }
  t.root_pid = 0;
  check_tree(&t);
  let mut out = Vec::new();
  for (k, v) in [(b"a", b"1"), (b"b", b"2"), (b"n", b"3"), (b"o", b"4")] {
    assert!(t.look_up(k, &mut out).await.unwrap());
    assert_eq!(&out, v);
  }
  // left(1) 并入 right(2)：right 必须同时持有合并前 A+B 的全部键
  let merged = t.merge_left_into_right(1, 2).await.unwrap();
  assert!(merged, "两叶合并应成功");
  {
    let nodes = t.nodes.read();
    assert!(!nodes.contains_key(&1), "left 页应被回收");
    let nd = nodes.get(&2).expect("right 页应存活");
    let mut got: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    for s in 0..nd.hdr.count {
      got.insert(full_key(nd, s), nd.get_payload(s).to_vec());
    }
    let expect: BTreeMap<Vec<u8>, Vec<u8>> = [
      (b"a".to_vec(), b"1".to_vec()),
      (b"b".to_vec(), b"2".to_vec()),
      (b"n".to_vec(), b"3".to_vec()),
      (b"o".to_vec(), b"4".to_vec()),
    ]
    .into_iter()
    .collect();
    assert_eq!(got, expect, "合并后键集合 = 两侧并集（不得丢右节点记录）");
  }
  let mut out = Vec::new();
  for (k, v) in [(b"a", b"1"), (b"b", b"2"), (b"n", b"3"), (b"o", b"4")] {
    assert!(t.look_up(k, &mut out).await.unwrap(), "合并后 {k:?} 丢失");
    assert_eq!(&out, v);
  }
  check_tree(&t);
}

/// 大量删除触发合并/XMerge 后：条目数、全键、扫描、不变量全部一致。
#[compio::test]
async fn mass_delete_merge_scan_consistent() {
  let mut t = mk();
  let n = 1200u32;
  for i in 0..n {
    t.insert(&key(i), &[3u8; 8]).await.unwrap();
  }
  for i in (0..n).step_by(3) {
    assert!(t.remove(&key(i)).await.unwrap(), "删除 {i}");
  }
  let survivors: Vec<u32> = (0..n).filter(|i| i % 3 != 0).collect();
  assert_eq!(t.count_entries().await.unwrap(), survivors.len() as u64);
  let mut out = Vec::new();
  for &i in &survivors {
    assert!(t.look_up(&key(i), &mut out).await.unwrap(), "存活 {i} 丢失");
  }
  let mut scanned = Vec::new();
  t.scan_ascending(b"", b"", |k, _| {
    scanned.push(k.to_vec());
    true
  })
  .await
  .unwrap();
  let expect: Vec<Vec<u8>> = survivors.iter().map(|&i| key(i)).collect();
  assert_eq!(scanned, expect, "扫描顺序/集合与存活键不一致");
  check_tree(&t);
}

/// XMerge 冷端级联后不丢键且结构不变量成立（多叶场景）。
#[compio::test]
async fn x_merge_full_run_no_loss() {
  let mut t = mk();
  let n = 900u32;
  for i in 0..n {
    t.insert(&key(i), &[1u8; 4]).await.unwrap();
  }
  let before_pages = t.count_pages().await.unwrap();
  let _ = t.x_merge(0).await.unwrap();
  assert_eq!(t.count_entries().await.unwrap(), n as u64);
  let mut out = Vec::new();
  for i in 0..n {
    assert!(
      t.look_up(&key(i), &mut out).await.unwrap(),
      "i={i} XMerge 后丢失"
    );
  }
  assert!(t.count_pages().await.unwrap() <= before_pages);
  check_tree(&t);
}

/// 空树 / 单键 / 全删空：边界不 panic。
#[compio::test]
async fn empty_single_reinsert() {
  let mut t = mk();
  assert!(!t.is_not_empty());
  assert!(!t.remove(b"nope").await.unwrap());
  t.insert(b"only", b"v").await.unwrap();
  assert!(t.is_not_empty());
  check_tree(&t);
  assert!(t.remove(b"only").await.unwrap());
  assert_eq!(t.count_entries().await.unwrap(), 0);
  check_tree(&t);
}

/// 变长 upsert（同键不同长 payload）不丢原记录、不 panic。
#[compio::test]
async fn variable_length_upsert() {
  let mut t = BTree::create(ldb_node::MEMCMP);
  t.insert(b"k", &[1u8; 4]).await.unwrap();
  assert!(t.update(b"k", &[2u8; 40]).await.unwrap());
  let mut out = Vec::new();
  assert!(t.look_up(b"k", &mut out).await.unwrap());
  assert_eq!(out, vec![2u8; 40]);
  assert!(t.update_in_place(b"k", &[3u8; 40]).await.unwrap());
  assert!(t.look_up(b"k", &mut out).await.unwrap());
  assert_eq!(out, vec![3u8; 40]);
  check_tree(&t);
}
