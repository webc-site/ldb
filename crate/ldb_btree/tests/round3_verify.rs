//! Round-3 独立复核（主审自写，不引用 round2 harness）。
//!
//! 目标（冻结轮）：
//! 1. **0 号槽分裂 P0 回归**：1196 B（共享 1192 B 前缀）长键纯插入第 4 条触发
//!    `count==2` 叶分裂 —— 修复前 panic（`ldb_node` `assert!(separator_slot > 0)`），
//!    修复后不 panic 且键/结构完好。见 `r3_long_key_slot0_split_repro`。
//! 2. **`split_leaf_two` 等价性**：对 2 记录叶的 0 号槽分裂，逐字段（键分布 / fence /
//!    separator / 叶链）与上游 `BTreeNode::SplitNode`（ZLeanStore `node.cc:540-580`；
//!    Rust 端口 `ldb_node::split_node`）语义比对，并把**右页整页字节**与
//!    `split_node(slot=1)` 在等价场景下的右页逐字节比对。见
//!    `r3_slot0_split_fields_semantics` / `r3_slot0_right_page_matches_split_node`。
//! 3. **分裂重试非终止兜底**：`ensure_space_for_split` 对 `count<=1` 且无空间的页
//!    返回 `Err`（否则 `insert` 无限重试）。见 `r3_ensure_space_stuck_returns_err`。

use std::slice::from_raw_parts;

use ldb_btree::Error;
mod common;

use common::{check_tree, full_key, mk};
use ldb_btree::BTree;
use ldb_node::{
  BTreeNode, MEMCMP,
  node::{EMPTY_NEIGHBOR, MAX_RECORD_SIZE},
};
use ldb_type::{PAGE_SIZE, PageId};

/// 长键：1192 B 公共前缀 + 4 B 大端计数（总长 1196 B < `MAX_RECORD_SIZE`=1208）。
fn long_key(i: u32) -> Vec<u8> {
  let mut k = vec![b'a'; 1192];
  k.extend_from_slice(&i.to_be_bytes());
  k
}

/// 手工构造 `root`（空内节点，`right_most_child = leaf_pid`）+ 一个叶：
/// 叶持有 `keys`，双 fence 由参数给定。返回 (`root_pid`, `leaf_pid`)。
///
/// 空内根（`count == 0`）使任意路由键都落 `right_most_child`，从而把插入/分裂
/// 精确导向该叶，且**非根**（走 `split_child` 而非 `split_root` 分支）。
const ROOT_PID: PageId = 0;
const LEAF_PID: PageId = 5;

fn build_root_empty_leaf(keys: &[Vec<u8>], lower: &[u8], upper: &[u8]) -> BTree {
  let mut t = BTree::create(MEMCMP);
  {
    let mut nodes = t.nodes.write();
    nodes.clear();
    let mut root = BTreeNode::new(false, 0);
    root.set_fences(b"", b"", MEMCMP);
    root.hdr.right_most_child = LEAF_PID;
    let mut leaf = BTreeNode::new(true, 0);
    leaf.set_fences(lower, upper, MEMCMP);
    for k in keys {
      assert!(
        leaf.insert_key_value(k, b"", MEMCMP),
        "叶内插入失败（键过大？len={}）",
        k.len()
      );
    }
    leaf.hdr.prev_leaf_node = EMPTY_NEIGHBOR;
    leaf.hdr.right_most_child = EMPTY_NEIGHBOR;
    nodes.insert(ROOT_PID, root);
    nodes.insert(LEAF_PID, leaf);
  }
  t.root_pid = ROOT_PID;
  t
}

/// 整页字节快照（4096 B）
fn page_bytes(n: &BTreeNode) -> Vec<u8> {
  // SAFETY: BTreeNode 恰为 PAGE_SIZE 字节；只读切片借用。
  unsafe { from_raw_parts(n.ptr(), PAGE_SIZE).to_vec() }
}

// ---------------------------------------------------------------------------
// 1：0 号槽分裂 P0 —— 纯插入长键回归
// ---------------------------------------------------------------------------

/// 纯插入 1196 B（共享 1192 B 前缀）长键序列：第 4 条即在 `count==2` 叶触发
/// `find_separator` 返回 slot 0。修复前 `split_child` 直接走 `ldb_node::split_node`
/// → `assert!(separator_slot > 0)` panic（变异 M3-1 复现）；修复后走 `split_leaf_two`
/// → 不 panic、键全可查、结构不变量 + 逐页 `validate` 通过。
#[compio::test]
async fn r3_long_key_slot0_split_repro() {
  assert!(
    long_key(0).len() as u32 <= MAX_RECORD_SIZE,
    "用例前提：长键须 <= MAX_RECORD_SIZE"
  );
  let mut t = mk();
  for i in 0..400u32 {
    t.insert(&long_key(i), &[]).await.unwrap();
  }
  let mut out = Vec::new();
  for i in 0..400u32 {
    assert!(
      t.look_up(&long_key(i), &mut out).await.unwrap(),
      "i={i} 丢失"
    );
  }
  assert_eq!(t.count_entries().await.unwrap(), 400);
  // 逐页字节自洽（split_leaf_two 手工构造页必须满足）
  {
    let nodes = t.nodes.read();
    for (&pid, n) in nodes.iter() {
      n.validate(MEMCMP)
        .unwrap_or_else(|e| panic!("页 {pid} validate 失败: {e}"));
    }
  }
  check_tree(&t);
}

/// 直接钉死「2 记录叶的 `find_separator` 返回 slot 0」这一前提（不是猜测）：
/// 断言 slot==0 且 sep 为全键 `key[0]`（`kb.len() > common+1` 不成立 ⇒ 不截断）。
#[compio::test]
async fn r3_two_record_find_separator_is_slot0() {
  let k0 = long_key(0);
  let k1 = long_key(1);
  let t = build_root_empty_leaf(&[k0.clone(), k1.clone()], b"", b"");
  let nodes = t.nodes.read();
  let leaf = nodes.get(&LEAF_PID).unwrap();
  let sep = leaf.find_separator(true, MEMCMP);
  assert_eq!(
    sep.slot, 0,
    "append_bias 下 count==2 叶 find_separator 必须 slot 0"
  );
  assert!(
    !sep.is_truncated,
    "1196 B 长键不应截断（kb.len() > common+1 不成立）"
  );
  let mut buf = vec![0u8; sep.len as usize];
  leaf.get_separator_key(&mut buf, &sep);
  assert_eq!(buf, k0, "0 号槽分隔键 == 全键 key[0]");
}

// ---------------------------------------------------------------------------
// 2：split_leaf_two 等价性
// ---------------------------------------------------------------------------

/// 字段级等价：`try_split_with_parent` 走 0 号槽 → 产出页逐字段满足上游
/// `BTreeNode::SplitNode`（`node.cc:568-580`）在 `separator_slot == 0` 时的语义：
/// 左页保留 `key[0]`、右页得 `key[1]`、父分隔键 = `key[0]`、fence 与叶链按
/// `SetFences` + `prev/next` 契约，且三页 `validate` + 树不变量通过。
#[compio::test]
async fn r3_slot0_split_fields_semantics() {
  let k0 = long_key(0);
  let k1 = long_key(1);
  let mut t = build_root_empty_leaf(&[k0.clone(), k1.clone()], b"", b"");
  // 非根 2 记录叶分裂（走 split_child 的 0 号槽分支）
  t.try_split_with_parent(LEAF_PID, Some(ROOT_PID), None)
    .await
    .unwrap();

  let nodes = t.nodes.read();
  let root = nodes.get(&ROOT_PID).unwrap();
  assert!(root.hdr.count == 1, "根应新增 1 个分隔键");
  assert_eq!(full_key(root, 0), k0, "父分隔键 == key[0]");
  assert_eq!(root.get_child(0), LEAF_PID, "分隔键 payload 指左页");
  let right_pid = root.hdr.right_most_child;
  assert_ne!(right_pid, LEAF_PID);

  let left = nodes.get(&LEAF_PID).unwrap();
  assert!(left.hdr.is_leaf);
  assert_eq!(left.hdr.count, 1, "左页保留 1 条");
  assert_eq!(full_key(left, 0), k0, "左页 = key[0]");
  assert_eq!(left.get_lower_fence(), b"", "左页下界 = 原下界");
  assert_eq!(left.get_upper_fence(), k0.as_slice(), "左页上界 = 分隔键");
  assert_eq!(left.hdr.prev_leaf_node, EMPTY_NEIGHBOR, "左页 prev 不变");
  assert_eq!(left.hdr.right_most_child, right_pid, "左页 next = 新右页");

  let right = nodes.get(&right_pid).unwrap();
  assert!(right.hdr.is_leaf);
  assert_eq!(right.hdr.count, 1, "右页得 1 条");
  assert_eq!(full_key(right, 0), k1, "右页 = key[1]");
  assert_eq!(right.get_lower_fence(), k0.as_slice(), "右页下界 = 分隔键");
  assert_eq!(right.get_upper_fence(), b"", "右页上界 = 原上界");
  assert_eq!(right.hdr.prev_leaf_node, LEAF_PID, "右页 prev = 左页");
  assert_eq!(
    right.hdr.right_most_child, EMPTY_NEIGHBOR,
    "右页 next = 原 next"
  );

  for (&pid, n) in nodes.iter() {
    n.validate(MEMCMP)
      .unwrap_or_else(|e| panic!("页 {pid} validate 失败: {e}"));
  }
  drop(nodes);
  check_tree(&t);
}

/// 跨实现字节等价：右页整页字节 == `ldb_node::split_node`（上游 `SplitNode` 端口）
/// 在**等价场景**（同 `sep_key`、同 `upper`、同 `old_next`、同 pid）下产出的右页。
///
/// 场景：
/// - S2（0 号槽，本 crate `split_leaf_two`）：2 记录叶 `[k0,k1]`，`find_separator`
///   → slot 0，`sep = k0`；右页 = `[k1]`，fence `(k0, "")`。
/// - S1（`split_node(slot=1)`）：3 记录叶 `[x,k0,k1]`（`x < k0`），`find_separator`
///   → slot 1，`sep = k0`；`split_node` 右页 = `[k1]`，fence `(k0, "")`。
///
/// 两场景的右页输入（sep / upper / old_next / left_pid / 新 pid）相同，故上游
/// `SplitNode` 输出必须逐字节一致 —— 直接证明 `split_leaf_two` 的右半页构造
/// 与 `split_node` 同构（左半页差异仅是 slot 定义：0 号槽只留 1 条）。
#[compio::test]
async fn r3_slot0_right_page_matches_split_node() {
  let k0 = long_key(0);
  let k1 = long_key(1);
  // x < k0（'0' = 0x30 < 'a' = 0x61），长度仍 <= MAX_RECORD_SIZE
  let mut x = vec![b'0'];
  x.extend_from_slice(&long_key(99));

  // ---- S1：3 记录叶 → split_node(slot 1) ----
  let mut t1 = build_root_empty_leaf(&[x.clone(), k0.clone(), k1.clone()], b"", b"");
  {
    let nodes = t1.nodes.read();
    let sep = nodes.get(&LEAF_PID).unwrap().find_separator(true, MEMCMP);
    assert_eq!(sep.slot, 1, "S1 前提：3 记录叶 find_separator slot 1");
  }
  t1.try_split_with_parent(LEAF_PID, Some(ROOT_PID), None)
    .await
    .unwrap();
  let r1 = {
    let nodes = t1.nodes.read();
    let rp = nodes.get(&ROOT_PID).unwrap().hdr.right_most_child;
    page_bytes(nodes.get(&rp).unwrap())
  };

  // ---- S2：2 记录叶 → split_leaf_two(slot 0) ----
  let mut t2 = build_root_empty_leaf(&[k0.clone(), k1.clone()], b"", b"");
  {
    let nodes = t2.nodes.read();
    let sep = nodes.get(&LEAF_PID).unwrap().find_separator(true, MEMCMP);
    assert_eq!(sep.slot, 0, "S2 前提：2 记录叶 find_separator slot 0");
  }
  t2.try_split_with_parent(LEAF_PID, Some(ROOT_PID), None)
    .await
    .unwrap();
  let r2 = {
    let nodes = t2.nodes.read();
    let rp = nodes.get(&ROOT_PID).unwrap().hdr.right_most_child;
    // 逐字段确认右页内容后再比字节
    let r = nodes.get(&rp).unwrap();
    assert_eq!(r.hdr.count, 1);
    assert_eq!(full_key(r, 0), k1);
    assert_eq!(r.get_lower_fence(), k0.as_slice());
    assert_eq!(r.get_upper_fence(), b"");
    assert_eq!(r.hdr.prev_leaf_node, LEAF_PID);
    assert_eq!(r.hdr.right_most_child, EMPTY_NEIGHBOR);
    page_bytes(r)
  };

  assert_eq!(
    r1, r2,
    "split_leaf_two 右页字节必须与 split_node(slot=1) 等价场景右页逐字节一致"
  );
}

// ---------------------------------------------------------------------------
// 3：分裂重试非终止兜底（P2）
// ---------------------------------------------------------------------------

/// `count<=1` 且无空间的叶：`ensure_space_for_split` 必须在冷路径返回 `Err`
/// （否则 `insert` 的 `continue` 无限重试）。构造单记录 + 双长 fence 的叶，
/// 使下一条同长记录放不下；断言 `insert` 返回 `Err(CorruptTree)` 而非挂死。
#[compio::test]
async fn r3_ensure_space_stuck_returns_err() {
  let lo = vec![b'a'; 1000];
  let hi = vec![b'b'; 1000];
  let mut t = build_root_empty_leaf(&[long_key(0)], &lo, &hi);
  // 前提：该叶确实放不下下一条 1196 B 记录
  {
    let nodes = t.nodes.read();
    let leaf = nodes.get(&LEAF_PID).unwrap();
    assert_eq!(leaf.hdr.count, 1);
    assert!(
      !leaf.has_space_for_kv(long_key(1).len() as u32, 0),
      "用例前提：单记录 + 双长 fence 的叶放不下第二条长键"
    );
  }
  let err = t.insert(&long_key(1), &[]).await.unwrap_err();
  assert!(
    matches!(err, Error::CorruptTree(_)),
    "count<=1 且无空间必须 Err(CorruptTree)，实得 {err:?}"
  );
}
