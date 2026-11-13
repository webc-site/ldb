//! Round-2 独立复核（主审自写，不引用 round1 harness）。
//!
//! 目标：对抗性复核 R1 的四处 SMO P0 修复、Contention Split 接线、merge/x_merge
//! 不丢记录，以及本 crate 新发现的 0 号槽分裂 panic 回归。
//!
//! 结构不变量（父指针唯一 / 叶链互指 / fence 升序 / 路由一致）复用
//! `common::check_tree`；本文件额外加入 **B+Tree 叶深度一致性**（R1 未覆盖：
//! 内节点分裂若产出叶页会破坏「所有叶同深度」）。

mod common;

use std::collections::{BTreeMap, HashMap, VecDeque};

use common::{check_tree, full_key, mk};
use ldb_btree::BTree;
use ldb_node::{BTreeNode, MEMCMP, node::EMPTY_NEIGHBOR};

/// 所有叶必须处于同一深度（B+Tree 高度一致不变量）。返回叶深度。
fn assert_uniform_leaf_depth(t: &BTree) -> u32 {
  let nodes = t.nodes.read();
  let mut depth: HashMap<ldb_type::PageId, u32> = HashMap::new();
  let mut q: VecDeque<(ldb_type::PageId, u32)> = VecDeque::new();
  q.push_back((t.root_pid, 0));
  depth.insert(t.root_pid, 0);
  let mut leaf_depth: Option<u32> = None;
  while let Some((pid, d)) = q.pop_front() {
    let n = nodes
      .get(&pid)
      .unwrap_or_else(|| panic!("可达子页 {pid} 缺失"));
    if n.hdr.is_leaf {
      match leaf_depth {
        None => leaf_depth = Some(d),
        Some(x) => assert_eq!(x, d, "叶 {pid} 深度 {d} != 兄弟叶深度 {x}"),
      }
      continue;
    }
    for i in 0..n.hdr.count {
      q.push_back((n.get_child(i), d + 1));
    }
    q.push_back((n.hdr.right_most_child, d + 1));
  }
  leaf_depth.expect("树无叶节点")
}

/// 每个页字节级自洽（`validate_bytes`）：space_used 精确记账、槽位/fence 在界内、
/// head 与 key 一致。`split_leaf_two` 手工构造页必须满足此不变量。
fn assert_all_valid(t: &BTree) {
  let nodes = t.nodes.read();
  for (&pid, n) in nodes.iter() {
    n.validate(MEMCMP)
      .unwrap_or_else(|e| panic!("页 {pid} 字节校验失败: {e}"));
  }
}

/// 长键：1192 B 公共前缀 + 4 B 计数（总长 1196 == MAX_RECORD_SIZE）。
fn long_key(i: u32) -> Vec<u8> {
  let mut k = vec![b'a'; 1192];
  k.extend_from_slice(&i.to_be_bytes());
  k
}

// ---------------------------------------------------------------------------
// 1a/1b/1c/1d：R1 四处新 P0 的独立复核
// ---------------------------------------------------------------------------

/// 1a 内节点分裂必须建内页：深树 + 长键，断言所有叶同深度、根为内、≥2 内层，
/// 且任何内节点的子都不可能是「伪叶」（深度一致性即击杀）。
#[compio::test]
async fn r2_inner_split_builds_inner_page() {
  let mut t = mk();
  // 长键压低扇出：单叶 ~3 条、单内节 ~3 分隔键 → 多级内节点
  let key = |i: u32| format!("{i:06}-{}", "z".repeat(300)).into_bytes();
  for i in 0..3000u32 {
    t.insert(&key(i), &i.to_le_bytes()).await.unwrap();
  }
  let mut out = Vec::new();
  for i in 0..3000u32 {
    assert!(t.look_up(&key(i), &mut out).await.unwrap(), "i={i} 丢失");
    assert_eq!(out, i.to_le_bytes());
  }
  check_tree(&t);
  let d = assert_uniform_leaf_depth(&t);
  assert!(d >= 2, "长键 3000 条应产生 ≥2 级内节点，叶深度实得 {d}");
  let nodes = t.nodes.read();
  assert!(
    !nodes.get(&t.root_pid).unwrap().hdr.is_leaf,
    "深树根必须是内节点"
  );
}

/// 1b 分裂前为父层分隔键预留空间：递减 + 长键强制「子分裂时父满」路径，
/// 断言不 panic、键不丢、结构完好。
#[compio::test]
async fn r2_parent_sep_space_reserved_long_keys() {
  let mut t = mk();
  let key = |i: u32| format!("{i:06}-{}", "y".repeat(400)).into_bytes();
  for i in (0..1200u32).rev() {
    t.insert(&key(i), &i.to_le_bytes()).await.unwrap();
  }
  let mut out = Vec::new();
  for i in 0..1200u32 {
    assert!(t.look_up(&key(i), &mut out).await.unwrap(), "i={i} 丢失");
  }
  check_tree(&t);
}

/// 1c 分裂后叶链 prev/next 双向一致：正走 `right_most_child`、反走
/// `prev_leaf_node`，断言两条链覆盖同一叶集合且互指。
#[compio::test]
async fn r2_leaf_chain_bidirectional() {
  let mut t = mk();
  // 乱序插入：使分裂发生在「非最右叶」上，逼出 `fix_right_neighbor_prev`
  // 才修的原 next.prev（升序插入只在最右叶分裂，old_next 恒为 EMPTY，测不到）
  let n = 900u32;
  let mut idx: Vec<u32> = (0..n).collect();
  let mut s = 0x1357_9BDF_2468_ACE1u64;
  for i in (1..idx.len()).rev() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    let j = (s % (i as u64 + 1)) as usize;
    idx.swap(i, j);
  }
  for &i in &idx {
    t.insert(&format!("k{i:06}").into_bytes(), &[0u8; 8])
      .await
      .unwrap();
  }
  // 再删一批，触发叶合并，链更易断
  for i in (0..900u32).step_by(4) {
    t.remove(&format!("k{i:06}").into_bytes()).await.unwrap();
  }
  check_tree(&t);
  let nodes = t.nodes.read();
  // 找链头（prev == EMPTY）
  let mut heads = nodes
    .iter()
    .filter(|(_, n)| n.hdr.is_leaf && n.hdr.prev_leaf_node == EMPTY_NEIGHBOR)
    .map(|(&p, _)| p);
  let head = heads.next().expect("应有唯一链头");
  assert!(heads.next().is_none(), "链头唯一");
  // 正向
  let mut forward = Vec::new();
  let mut cur = head;
  loop {
    forward.push(cur);
    let next = nodes.get(&cur).unwrap().hdr.right_most_child;
    if next == EMPTY_NEIGHBOR {
      break;
    }
    assert_eq!(
      nodes.get(&next).unwrap().hdr.prev_leaf_node,
      cur,
      "叶 {next}.prev != {cur}"
    );
    cur = next;
  }
  // 反向
  let mut backward = vec![*forward.last().unwrap()];
  let mut cur = *forward.last().unwrap();
  while nodes.get(&cur).unwrap().hdr.prev_leaf_node != EMPTY_NEIGHBOR {
    let p = nodes.get(&cur).unwrap().hdr.prev_leaf_node;
    assert_eq!(
      nodes.get(&p).unwrap().hdr.right_most_child,
      cur,
      "叶 {p}.next != {cur}"
    );
    backward.push(p);
    cur = p;
  }
  backward.reverse();
  assert_eq!(forward, backward, "正/反向叶链不一致");
}

/// 1d 内层根分裂必须能抬高根：分层插入并记录每次根号变化；断言存在一次变化
/// 其「旧根为内节点」（即内层根分裂），且最终根为内、叶深度 ≥2。
#[compio::test]
async fn r2_inner_root_split_raises_root() {
  let mut t = mk();
  let key = |i: u32| format!("r{i:06}-{}", "q".repeat(120)).into_bytes();
  let mut prev_root = t.root_pid;
  let mut saw_inner_root_split = false;
  for i in 0..4000u32 {
    t.insert(&key(i), &i.to_le_bytes()).await.unwrap();
    if t.root_pid != prev_root {
      let old_is_leaf = {
        let nodes = t.nodes.read();
        nodes.get(&prev_root).map(|n| n.hdr.is_leaf)
      };
      if old_is_leaf == Some(false) {
        saw_inner_root_split = true;
      }
      prev_root = t.root_pid;
    }
  }
  assert!(saw_inner_root_split, "应发生至少一次内层根分裂");
  let d = assert_uniform_leaf_depth(&t);
  assert!(d >= 2, "内层根分裂后叶深度应 ≥2，实得 {d}");
  let mut out = Vec::new();
  for i in 0..4000u32 {
    assert!(t.look_up(&key(i), &mut out).await.unwrap(), "i={i} 丢失");
  }
  check_tree(&t);
}

// ---------------------------------------------------------------------------
// 1e：0 号槽分裂（count==2 叶）P0 回归
// ---------------------------------------------------------------------------

/// 纯插入 1196 B 长键（共享 1192 B 前缀）——R1 声称 `MAX_RECORD_SIZE` 守卫使
/// `count==2` 分裂不可达；实际第 4 条即触发 `ldb_node` 的
/// `assert!(separator_slot > 0)` panic。本用例钉死修复：不 panic、键不丢、
/// 结构完好、叶深度一致。
#[compio::test]
async fn r2_two_record_leaf_split_no_panic() {
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
  check_tree(&t);
  assert_all_valid(&t);
  assert_uniform_leaf_depth(&t);
}

/// favored=0：`min(last, cur) == 0` 的上游合法分裂点（master `BTreeNode.cpp:370`
/// 允许 0 号槽）。断言不 panic 且分裂点被安全回退。
#[compio::test]
async fn r2_favored_zero_no_panic() {
  let mut t = mk();
  for i in 0..4u32 {
    t.insert(&format!("c{i:03}").into_bytes(), &[0u8; 8])
      .await
      .unwrap();
  }
  assert!(!t.contention_split(0, 0, true, 128).await.unwrap());
  let _ = t.contention_split(0, 3, true, 0).await.unwrap();
  check_tree(&t);
}

// ---------------------------------------------------------------------------
// 2：Contention Split 接线（固定 rand 序列，触发 / 不触发）
// ---------------------------------------------------------------------------

/// 双采样门确定性序列：rand=1 未过第一层门 → 不分裂；rand=128 仅记录；
/// rand=0 同时过两层且 last!=slot → 结算分裂（favored=min）。
#[compio::test]
async fn r2_contention_split_gates_deterministic() {
  let mut t = mk();
  for i in 0..4u32 {
    t.insert(&format!("g{i:03}").into_bytes(), &[0u8; 8])
      .await
      .unwrap();
  }
  let pages = t.count_pages().await.unwrap();
  // rand=1：1 & 127 != 0 → 未过第一层门
  assert!(!t.contention_split(0, 0, true, 1).await.unwrap());
  assert_eq!(t.count_pages().await.unwrap(), pages, "未过门不得分裂");
  // rand=128：过第一层（128 & 127 == 0）但未过第二层（128 & 16383 != 0）→ 只记录
  assert!(!t.contention_split(0, 1, true, 128).await.unwrap());
  assert_eq!(t.count_pages().await.unwrap(), pages, "仅记录不得分裂");
  // rand=0：两层全过；last=1 != slot=3；restarts=2, accesses=2, normalized=100≥1; count=4>2
  // → favored=min(1,3)=1
  assert!(t.contention_split(0, 3, true, 0).await.unwrap());
  check_tree(&t);
}

/// faced_contention=false 时 normalized=0 < 阈值 → 不分裂。
#[compio::test]
async fn r2_contention_split_cold_no_split() {
  let mut t = mk();
  for i in 0..8u32 {
    t.insert(&format!("h{i:03}").into_bytes(), &[0u8; 8])
      .await
      .unwrap();
  }
  let pages = t.count_pages().await.unwrap();
  assert!(!t.contention_split(0, 0, false, 0).await.unwrap());
  assert_eq!(t.count_pages().await.unwrap(), pages);
}

// ---------------------------------------------------------------------------
// 3：merge / x_merge 不丢记录
// ---------------------------------------------------------------------------

/// 手工构造 root(内) + 三叶 A/B/C，每叶多记录（含边界键），逐对合并：
/// 合并后右页键集合 == 两侧并集；父分隔键不重复；结构不变量成立。
#[compio::test]
async fn r2_merge_multi_record_union() {
  let mut t = BTree::create(MEMCMP);
  {
    let mut nodes = t.nodes.write();
    nodes.clear();
    let mut root = BTreeNode::new(false, 0);
    root.set_fences(b"", b"", MEMCMP);
    // 三叶：A=[a1,a2,a3] B=[b1,b2,b3] C=[c1,c2,c3]，分隔键 a3/b3
    let mut a = BTreeNode::new(true, 0);
    a.set_fences(b"", b"a3", MEMCMP);
    for k in [b"a1", b"a2", b"a3"] {
      assert!(a.insert_key_value(k, b"v", MEMCMP));
    }
    a.hdr.prev_leaf_node = EMPTY_NEIGHBOR;
    a.hdr.right_most_child = 2;
    let mut b = BTreeNode::new(true, 0);
    b.set_fences(b"a3", b"b3", MEMCMP);
    for k in [b"b1", b"b2", b"b3"] {
      assert!(b.insert_key_value(k, b"v", MEMCMP));
    }
    b.hdr.prev_leaf_node = 1;
    b.hdr.right_most_child = 3;
    let mut c = BTreeNode::new(true, 0);
    c.set_fences(b"b3", b"", MEMCMP);
    for k in [b"c1", b"c2", b"c3"] {
      assert!(c.insert_key_value(k, b"v", MEMCMP));
    }
    c.hdr.prev_leaf_node = 2;
    c.hdr.right_most_child = EMPTY_NEIGHBOR;
    assert!(root.insert_key_value(b"a3", &1u64.to_le_bytes(), MEMCMP));
    assert!(root.insert_key_value(b"b3", &2u64.to_le_bytes(), MEMCMP));
    root.hdr.right_most_child = 3;
    nodes.insert(0, root);
    nodes.insert(1, a);
    nodes.insert(2, b);
    nodes.insert(3, c);
  }
  t.root_pid = 0;
  check_tree(&t);
  assert_all_valid(&t);
  // 合并 A(1) → B(2)
  assert!(t.merge_left_into_right(1, 2).await.unwrap());
  {
    let nodes = t.nodes.read();
    assert!(!nodes.contains_key(&1), "left 页应回收");
    let nd = nodes.get(&2).unwrap();
    let set: BTreeMap<Vec<u8>, ()> = (0..nd.hdr.count).map(|s| (full_key(nd, s), ())).collect();
    let expect: BTreeMap<Vec<u8>, ()> = [b"a1", b"a2", b"a3", b"b1", b"b2", b"b3"]
      .iter()
      .map(|k| (k.to_vec(), ()))
      .collect();
    assert_eq!(
      set, expect,
      "合并后键集合 == A∪B（含边界键，不得丢右页记录）"
    );
    // 父分隔键不得重复
    let root = nodes.get(&0).unwrap();
    for s in 1..root.hdr.count {
      assert!(
        full_key(root, s - 1) < full_key(root, s),
        "父分隔键重复/未升序"
      );
    }
  }
  check_tree(&t);
  let mut out = Vec::new();
  for k in [
    b"a1", b"a2", b"a3", b"b1", b"b2", b"b3", b"c1", b"c2", b"c3",
  ] {
    assert!(t.look_up(k, &mut out).await.unwrap(), "{k:?} 丢失");
  }
  // 再合并 B(2) → C(3)
  assert!(t.merge_left_into_right(2, 3).await.unwrap());
  check_tree(&t);
  for k in [b"a1", b"b1", b"b3", b"c1", b"c3"] {
    assert!(t.look_up(k, &mut out).await.unwrap(), "二次合并 {k:?} 丢失");
  }
}

/// XMerge 多兄弟级联：键集合不丢。
#[compio::test]
async fn r2_xmerge_multi_no_loss() {
  let mut t = mk();
  let n = 900u32;
  for i in 0..n {
    t.insert(&format!("x{i:06}").into_bytes(), &[1u8; 4])
      .await
      .unwrap();
  }
  let before = t.count_pages().await.unwrap();
  let _ = t.x_merge(0).await.unwrap();
  assert_eq!(t.count_entries().await.unwrap(), n as u64);
  let mut out = Vec::new();
  for i in 0..n {
    assert!(
      t.look_up(&format!("x{i:06}").into_bytes(), &mut out)
        .await
        .unwrap()
    );
  }
  assert!(t.count_pages().await.unwrap() <= before);
  check_tree(&t);
}

/// `ensure_underfull_inners_for_merge` 上溯合并：深树大量删除后，
/// 父分隔键无重复、结构不变量成立、存活键全可查。
#[compio::test]
async fn r2_underfull_inners_merge_no_dup_sep() {
  let mut t = mk();
  let key = |i: u32| format!("u{i:06}-{}", "w".repeat(60)).into_bytes();
  let n = 1500u32;
  for i in 0..n {
    t.insert(&key(i), &[7u8; 8]).await.unwrap();
  }
  for i in (0..n).step_by(2) {
    t.remove(&key(i)).await.unwrap();
  }
  // check_tree 内部断言内节点分隔键严格升序（重复即失败）
  check_tree(&t);
  let survivors: Vec<u32> = (0..n).filter(|i| i % 2 == 1).collect();
  assert_eq!(t.count_entries().await.unwrap(), survivors.len() as u64);
  let mut out = Vec::new();
  for &i in &survivors {
    assert!(t.look_up(&key(i), &mut out).await.unwrap(), "存活 {i} 丢失");
  }
}

/// 随机长键 + 强制 favored 分裂 + 删除交织：结构不变量与键集合保持。
#[compio::test]
async fn r2_favored_split_fuzz_invariants() {
  let mut t = mk();
  let mut model: BTreeMap<Vec<u8>, ()> = BTreeMap::new();
  let mut rng = 0xDEAD_BEEF_1234_5678u64;
  let mut next = || {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    rng
  };
  for step in 0..6000u32 {
    let r = next();
    let i = (r % 1500) as u32;
    let k = long_key(i.min(1000)); // 长键压页
    if r & 3 != 0 {
      t.insert(&k, &[]).await.unwrap();
      model.insert(k.clone(), ());
    } else {
      let got = t.remove(&k).await.unwrap();
      assert_eq!(got, model.remove(&k).is_some(), "step {step}");
    }
    if r % 8 == 0 {
      // 强制结算门（rand=0），slot=0 触发 favored=0 回退路径
      let pid = {
        let nodes = t.nodes.read();
        let mut pid = t.root_pid;
        loop {
          let nd = nodes.get(&pid).unwrap();
          if nd.hdr.is_leaf {
            break pid;
          }
          let mut pos = 0u32;
          pid = nd.find_child(&k, &mut pos, MEMCMP);
        }
      };
      let _ = t.contention_split(pid, 0, true, 0).await.unwrap();
    }
    if step % 1500 == 0 {
      assert_eq!(
        t.count_entries().await.unwrap(),
        model.len() as u64,
        "step {step}"
      );
    }
  }
  check_tree(&t);
  let mut out = Vec::new();
  for k in model.keys() {
    assert!(t.look_up(k, &mut out).await.unwrap(), "丢失 {k:?}");
  }
}

// ---------------------------------------------------------------------------
// 4：持久化阶段 1 复核
// ---------------------------------------------------------------------------

mod store_round {
  use std::{
    rc::Rc,
    slice::from_raw_parts,
    sync::{Arc, atomic::AtomicBool},
  };

  use ldb_btree::{BufStore, Error};
  use ldb_buf::BufferManager;
  use ldb_free::FreeStorageManager;
  use ldb_io::MemBackend;
  use ldb_node::MEMCMP;
  use ldb_space::SpaceManager;
  use ldb_type::{GsnWatermark, PAGE_SIZE};

  pub fn mk_buf(pages: u64) -> BufferManager<MemBackend> {
    let space = Rc::new(SpaceManager::new(pages * 4, 64, 2048, 4, pages * 64, 2));
    let free = Arc::new(FreeStorageManager::new(true, 2));
    BufferManager::new(
      pages * 4,
      pages,
      space,
      free,
      MemBackend::new(),
      Arc::new(AtomicBool::new(true)),
      Arc::new(GsnWatermark::new(0)),
    )
    .unwrap()
  }

  /// 独立复跑「写 → flush → 驱逐 → 缺页读回」闭环（多页）。
  #[compio::test]
  async fn r2_store_multi_page_roundtrip() {
    let bm = mk_buf(32);
    let store = BufStore::new(&bm, MEMCMP);
    let mut pids = Vec::new();
    for p in 0..8u32 {
      let (pid, mut node) = store.create(true, 0).await.unwrap();
      for i in 0..(p + 1) {
        assert!(node.insert_key_value(format!("p{p}k{i}").as_bytes(), b"v", MEMCMP));
      }
      store.put(pid, &node).await.unwrap();
      pids.push((pid, p + 1));
    }
    store.flush().await.unwrap();
    for &(pid, _) in &pids {
      assert!(
        bm.evict_clean_pages(pid).await.unwrap(),
        "flush 后应可驱逐 {pid}"
      );
    }
    for &(pid, cnt) in &pids {
      let back = store.get(pid).await.unwrap();
      assert_eq!(back.hdr.count, cnt, "页 {pid} 记录数不符");
      assert!(back.hdr.is_leaf);
    }
  }

  /// 畸形页必须被 `validate_bytes` 拒收（三种破坏：is_leaf 非法字节、
  /// 记录 key 字节被篡改导致 head 不符、整页 0xFF）。
  #[compio::test]
  async fn r2_store_malformed_variants() {
    let bm = mk_buf(16);
    let store = BufStore::new(&bm, MEMCMP);
    let (pid, mut node) = store.create(true, 0).await.unwrap();
    for i in 0..5u32 {
      assert!(node.insert_key_value(format!("k{i}").as_bytes(), b"val", MEMCMP));
    }
    store.put(pid, &node).await.unwrap();
    // 合法页可读回
    assert!(store.get(pid).await.is_ok());

    let snapshot = || unsafe { from_raw_parts(node.ptr(), PAGE_SIZE).to_vec() };

    // (a) is_leaf 字节越界
    let mut a = snapshot();
    a[136] = 7;
    bm.update_pid(pid, &a).await.unwrap();
    assert!(
      matches!(store.get(pid).await, Err(Error::CorruptTree(_))),
      "非法 is_leaf 字节应拒收"
    );

    // (b) 篡改记录首字节：head 与 key 不符 / 有序性破坏
    let mut b = snapshot();
    let off = node.slots[0].offset as usize;
    b[off] ^= 0xFF;
    bm.update_pid(pid, &b).await.unwrap();
    assert!(
      matches!(store.get(pid).await, Err(Error::CorruptTree(_))),
      "篡改记录字节应拒收"
    );

    // (c) 恢复合法内容
    bm.update_pid(pid, &snapshot()).await.unwrap();
    assert!(store.get(pid).await.is_ok());
  }

  /// 撕裂页（半个 write）：把合法页后半替换为另一版本的字节。展示
  /// `validate_bytes` 能否检出 —— 结论见报告（无页校验和，结构自洽的撕裂
  /// 无法检出）。
  #[compio::test]
  async fn r2_store_torn_page_behavior() {
    let bm = mk_buf(16);
    let store = BufStore::new(&bm, MEMCMP);
    let (pida, mut a) = store.create(true, 0).await.unwrap();
    for i in 0..5u32 {
      assert!(a.insert_key_value(format!("k{i}").as_bytes(), b"aaaa", MEMCMP));
    }
    store.put(pida, &a).await.unwrap();
    let (pidb, mut b) = store.create(true, 0).await.unwrap();
    for i in 0..5u32 {
      assert!(b.insert_key_value(format!("z{i}").as_bytes(), b"bbbb", MEMCMP));
    }
    store.put(pidb, &b).await.unwrap();
    // 撕裂：前半 A、后半 B（含 header 从 A）—— struct 自洽性被破坏
    let mut torn = vec![0u8; PAGE_SIZE];
    let abytes = unsafe { from_raw_parts(a.ptr(), PAGE_SIZE) };
    let bbytes = unsafe { from_raw_parts(b.ptr(), PAGE_SIZE) };
    torn[..PAGE_SIZE / 2].copy_from_slice(&abytes[..PAGE_SIZE / 2]);
    torn[PAGE_SIZE / 2..].copy_from_slice(&bbytes[PAGE_SIZE / 2..]);
    bm.update_pid(pida, &torn).await.unwrap();
    // 观察行为（不强制断言，报告中描述）：结构自洽性破损 → 应拒收
    let res = store
      .get(pida)
      .await
      .map(|_| ())
      .map_err(|e| format!("{e:?}"));
    // 记录并断言「要么拒收，要么读回后仍满足 validate」——此处撕裂破槽位/账，应拒收
    assert!(res.is_err(), "跨 header/data 的撕裂页应被拒收: {res:?}");
  }
}
