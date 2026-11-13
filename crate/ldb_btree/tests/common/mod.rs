//! 测试共享工具：B+Tree 结构不变量检查 + 影子模型助手。
//!
//! 不变量覆盖本 crate 的 SMO / key range / fence / separator 契约：
//! - 根存在且从根可达的页集合 == 节点表全集（无悬挂/丢失父指针）；
//! - 每个非根页恰被一个父槽位引用；
//! - 内节点分隔键（子节点 upper fence）严格升序；
//! - 叶链（`prev_leaf_node`/`right_most_child`）唯一、完整、升序，且与页集合一致；
//! - 每个叶的首键经根下行 `find_child` 必须路由回该叶（fence/路由一致）。

use std::collections::HashSet;

use ldb_btree::BTree;
use ldb_node::{BTreeNode, Cmp, node::EMPTY_NEIGHBOR};

/// 取槽位全键（前缀 + 去前缀 key）
pub fn full_key(n: &BTreeNode, slot: u32) -> Vec<u8> {
  let mut k = n.get_prefix().to_vec();
  k.extend_from_slice(n.get_key(slot));
  k
}

/// 从根下行到叶（与 `BTree::find_leaf_*` 同路由语义）
pub fn route(
  nodes: &gxhash::HashMap<ldb_type::PageId, Box<BTreeNode>>,
  root: ldb_type::PageId,
  key: &[u8],
  cmp: Cmp,
) -> ldb_type::PageId {
  let mut pid = root;
  loop {
    let n = nodes.get(&pid).expect("路由命中缺失页");
    if n.hdr.is_leaf {
      return pid;
    }
    let mut pos = 0u32;
    pid = n.find_child(key, &mut pos, cmp);
  }
}

/// 全量结构不变量检查；违反即 panic（附页号/原因）
pub fn check_tree(t: &BTree) {
  let nodes = t.nodes.read();
  let root = t.root_pid;
  assert!(nodes.get(&root).is_some(), "根页 {root} 缺失");

  // 可达性 + 父指针唯一性 + 内节点分隔键升序
  let mut reachable: HashSet<ldb_type::PageId> = HashSet::new();
  let mut stack = vec![root];
  while let Some(pid) = stack.pop() {
    assert!(
      reachable.insert(pid),
      "页 {pid} 被多个父槽位引用（重复父指针）"
    );
    let n = nodes
      .get(&pid)
      .unwrap_or_else(|| panic!("被引用的子页 {pid} 缺失"));
    if n.hdr.is_leaf {
      continue;
    }
    for i in 1..n.hdr.count {
      let a = full_key(n, i - 1);
      let b = full_key(n, i);
      assert!(a < b, "内节点 {pid} 分隔键未升序: {a:?} >= {b:?}");
    }
    for i in 0..n.hdr.count {
      stack.push(n.get_child(i));
    }
    stack.push(n.hdr.right_most_child);
  }
  assert_eq!(
    reachable.len(),
    nodes.len(),
    "存在不可达页（父指针丢失/悬挂）: 可达 {} vs 全表 {}",
    reachable.len(),
    nodes.len()
  );

  // 叶链：唯一起点、完整、升序、前后指针互指
  let leaf_cnt = nodes.values().filter(|n| n.hdr.is_leaf).count();
  let mut starts = Vec::new();
  for (&pid, n) in nodes.iter() {
    if n.hdr.is_leaf && n.hdr.prev_leaf_node == EMPTY_NEIGHBOR {
      starts.push(pid);
    }
  }
  assert_eq!(starts.len(), 1, "叶链起点数 != 1: {starts:?}");
  let mut chain = Vec::with_capacity(leaf_cnt);
  let mut cur = starts[0];
  loop {
    chain.push(cur);
    let n = nodes.get(&cur).unwrap();
    let next = n.hdr.right_most_child;
    if next == EMPTY_NEIGHBOR {
      break;
    }
    let nn = nodes
      .get(&next)
      .unwrap_or_else(|| panic!("叶 {cur} next 指向缺失页 {next}"));
    assert!(nn.hdr.is_leaf, "叶 {cur} next 指向内节点 {next}");
    assert_eq!(
      nn.hdr.prev_leaf_node, cur,
      "叶 {next}.prev != {cur}（链不一致）"
    );
    cur = next;
  }
  assert_eq!(
    chain.len(),
    leaf_cnt,
    "叶链长度 {} != 叶页数 {leaf_cnt}（链断裂）",
    chain.len()
  );
  // 链上 fence 升序（upper fence 严格递增）
  for w in chain.windows(2) {
    let a = nodes.get(&w[0]).unwrap();
    let b = nodes.get(&w[1]).unwrap();
    let au = a.get_upper_fence().to_vec();
    let bu = b.get_upper_fence().to_vec();
    // 右端叶 upper fence 为空（无穷）
    if !bu.is_empty() {
      assert!(au < bu, "叶链 fence 未升序: {au:?} >= {bu:?}");
    }
  }

  // 每个叶的首键路由回自身（fence/route 一致）
  for &pid in &chain {
    let n = nodes.get(&pid).unwrap();
    if n.hdr.count == 0 {
      continue;
    }
    let first = full_key(n, 0);
    assert_eq!(
      route(&nodes, root, &first, t.cmp),
      pid,
      "首键 {first:?} 未路由回叶 {pid}"
    );
    // 末键（若 upper fence 非无穷）必须 <= upper fence（inclusive）
    let last = full_key(n, n.hdr.count - 1);
    let uf = n.get_upper_fence();
    if !uf.is_empty() {
      assert!(
        last.as_slice() <= uf,
        "叶 {pid} 末键 {last:?} > upper fence {uf:?}"
      );
    }
    let lf = n.get_lower_fence();
    if !lf.is_empty() {
      assert!(
        first.as_slice() > lf,
        "叶 {pid} 首键 {first:?} <= lower fence {lf:?}（排除语义）"
      );
    }
  }
}

/// 新建空树
pub fn mk() -> BTree {
  BTree::create(ldb_node::MEMCMP)
}
