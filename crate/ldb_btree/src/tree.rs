//! 并发 B+Tree（上游 `ZLeanStore/src/storage/btree/tree.cc` + master
//! `leanstore/backend/leanstore/storage/btree/core/BTreeGeneric*.{hpp,cpp}`）。
//!
//! - OLC 乐观路径：`Restart` 即重试（jumpmu 语义的 Result 化）
//! - 分裂/合并的新页分配在锁外完成（§3.5 协议：禁止持 X 闩跨 alloc await）
//! - Contention Split：热点键空间二分（CIDR'21；采样状态机在 `ldb_node`）
//! - XMerge：跨多兄弟合并（master BTreeGeneric）
//!
//! # 页存储现状（本轮）
//!
//! 主路径仍是进程内 `nodes: RwLock<HashMap<PageId, BTreeNode>>` 节点表（**未接**
//! `ldb_buf`/VMCache）。落盘接线（单页 fix/unfix + 编解码 + 脏页回写）见
//! [`crate::store`]；`BTree` 接入 `ldb_buf` 的阶段划分见该模块文档与审查报告。

use std::{cmp::Ordering, fmt};

use gxhash::HashMap;
use ldb_node::{
  BTreeNode, Cmp, SIZE_UNDER_FULL, SeparatorInfo, SplitCtx,
  node::{EMPTY_NEIGHBOR, MAX_RECORD_SIZE},
};
use ldb_type::{PAGE_SIZE, PageId};
use parking_lot::RwLock;

use crate::{
  contention_split::update_gate,
  error::{Error, Result},
};

/// 持久节点槽（页号 → 拥有节点；O(1) 定位，Box 内容地址稳定）
pub type NodeSlot = gxhash::HashMap<PageId, Box<BTreeNode>>;

/// 争用采样随机状态初值（黄金比例常数；xorshift64 非零种子）
const RAND_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// B+Tree（根持有 + 页分配回调注入；生产路径待接 ldb_buf，当前用内存 map）
pub struct BTree {
  /// 根页号
  pub root_pid: PageId,
  /// 比较器
  pub cmp: Cmp,
  /// 追加偏置（右端插入密集时分裂点右移）
  pub append_bias: bool,
  /// 页分配计数（测试内存模型）
  next_pid: PageId,
  /// 节点存储（pid → node；单库单树无共享需求 —— Arc 移除）
  pub nodes: RwLock<NodeSlot>,
  /// 下行路径暂存（insert 用：try_split 的父栈，跨调用复用免分配）
  path_buf: Vec<PageId>,
  /// 争用采样随机数状态（上游 `utils::RandomGenerator::getRandU64()` 的确定性替身）
  rand_state: u64,
}

impl fmt::Debug for BTree {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BTree")
      .field("root", &self.root_pid)
      .field("next_pid", &self.next_pid)
      .field("nodes", &self.nodes.read().len())
      .finish()
  }
}

impl BTree {
  /// 构造（含空根）
  pub fn create(cmp: Cmp) -> Self {
    let mut nodes: NodeSlot = HashMap::default();
    nodes.insert(0, BTreeNode::new(true, 0));
    Self {
      root_pid: 0,
      cmp,
      append_bias: true,
      next_pid: 1,
      nodes: RwLock::new(nodes),
      path_buf: Vec::with_capacity(8),
      rand_state: RAND_SEED,
    }
  }

  /// 分配新页（锁外 —— 协议：分裂前预留；返回右页叶形态的占位节点）
  pub async fn alloc_pid(&mut self) -> PageId {
    self.alloc_node(true).await
  }

  /// 分配指定形态的新页（分裂时新页必须与源页同形态：上游 `BTreeNode(!node->IsInner())`）
  async fn alloc_node(&mut self, is_leaf: bool) -> PageId {
    let pid = self.next_pid;
    self.next_pid += 1;
    self.nodes.write().insert(pid, BTreeNode::new(is_leaf, 0));
    pid
  }

  /// 设置比较算子（账本：set_comparison_operator）
  pub fn set_comparison_operator(&mut self, cmp: Cmp) {
    self.cmp = cmp;
  }

  /// 追加偏置开关
  pub fn toggle_append_bias_mode(&mut self) {
    self.append_bias = !self.append_bias;
  }

  /// 树非空判定
  pub fn is_not_empty(&self) -> bool {
    self
      .nodes
      .read()
      .get(&self.root_pid)
      .is_some_and(|n| n.hdr.count > 0)
  }

  /// 尺寸（MiB）
  pub fn size_in_mb(&self) -> f64 {
    self.nodes.read().len() as f64 * PAGE_SIZE as f64 / (1024.0 * 1024.0)
  }

  /// 下一采样随机数（xorshift64；等价上游 `RandomGenerator::getRandU64`）
  #[inline]
  fn next_rand(&mut self) -> u64 {
    let mut x = self.rand_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.rand_state = x;
    x
  }

  // ---- 查找 ----

  /// 乐观下行找叶（上游 FindLeafOptimistic：逐层乐观快照；简化持有模型下
  /// 单次下行，层间版本校验语义由节点闩协议承载）
  async fn find_leaf_optimistic(&self, key: &[u8]) -> Result<PageId> {
    let nodes = self.nodes.read();
    let mut pid = self.root_pid;
    loop {
      let node = nodes
        .get(&pid)
        .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
      if node.hdr.is_leaf {
        return Ok(pid);
      }
      let mut pos = 0u32;
      pid = node.find_child(key, &mut pos, self.cmp);
    }
  }

  /// 共享下行找叶（上游 FindLeafShared）
  async fn find_leaf_shared(&self, key: &[u8]) -> Result<PageId> {
    self.find_leaf_optimistic(key).await
  }

  /// 点查（共享下行找叶 —— 上游 FindLeafShared 读路径）
  pub async fn look_up(&self, key: &[u8], out: &mut Vec<u8>) -> Result<bool> {
    let pid = self.find_leaf_shared(key).await?;
    let nodes = self.nodes.read();
    let leaf = nodes
      .get(&pid)
      .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
    let mut exact = false;
    let pos = leaf.lower_bound(key, &mut exact, self.cmp);
    if exact {
      let payload = leaf.get_payload(pos);
      out.clear();
      out.extend_from_slice(payload);
      Ok(true)
    } else {
      Ok(false)
    }
  }

  /// Blob 键点查（泛型比较器闭包）
  pub async fn look_up_blob(
    &self,
    key_head: [u8; 4],
    key_rest: &[u8],
    blob_cmp: impl Fn(&[u8], &[u8]) -> Ordering,
  ) -> Result<Option<Vec<u8>>> {
    let pid = self.find_leaf_shared(key_rest).await?;
    let nodes = self.nodes.read();
    let leaf = nodes
      .get(&pid)
      .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
    let mut exact = false;
    let pos = leaf.lower_bound_with_blob_key(key_head, key_rest, &mut exact, blob_cmp);
    if exact {
      Ok(Some(leaf.get_payload(pos).to_vec()))
    } else {
      Ok(None)
    }
  }

  /// 读 PID（直接页读语义）
  pub async fn read_pid(&self, pid: PageId, out: &mut Vec<u8>) -> Result<bool> {
    let nodes = self.nodes.read();
    match nodes.get(&pid) {
      Some(n) => {
        out.clear();
        out.extend_from_slice(n.get_payload(0));
        Ok(true)
      }
      None => Ok(false),
    }
  }

  // ---- 变更 ----

  /// 插入（单锁下行记父栈；容量检查与写入合并进同一写锁 —— TOCTOU 消除；
  /// 容量不足先分裂，新页预分配在锁外 —— 协议合规）
  ///
  /// 入口按 `MAX_RECORD_SIZE` 拒绝超限记录（上游 `BTree::Insert` 前置断言，
  /// `ZLeanStore/src/storage/btree/tree.cc:303`）：否则 `has_space_for_kv` 恒
  /// false，分裂路径会无限重试。
  pub async fn insert(&mut self, key: &[u8], payload: &[u8]) -> Result<()> {
    if (key.len() + payload.len()) as u32 > MAX_RECORD_SIZE {
      return Err(Error::RecordTooLarge {
        size: (key.len() + payload.len()) as u32,
        max: MAX_RECORD_SIZE,
      });
    }
    // `faced`：插入循环是否经历过一次分裂（等价上游 guard 的 facedContention）。
    let mut faced = false;
    let leaf_pid = loop {
      // 下行：单次读锁贯穿全层，同时记录内层路径（父栈）
      let (leaf_pid, parent) = {
        let nodes = self.nodes.read();
        self.path_buf.clear();
        let mut pid = self.root_pid;
        loop {
          let node = nodes
            .get(&pid)
            .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
          if node.hdr.is_leaf {
            break;
          }
          self.path_buf.push(pid);
          let mut pos = 0u32;
          pid = node.find_child(key, &mut pos, self.cmp);
        }
        (pid, self.path_buf.last().copied())
      };
      // 空间预检（上游 EnsureSpaceForSplit）：不足即先分裂并重走下行
      // （分裂可能改写父分隔键，键宿主叶会变化）
      if self
        .ensure_space_for_split(leaf_pid, key.len() as u32, payload.len() as u32)
        .await?
      {
        faced = true;
        continue;
      }
      {
        // 空间检查与写入同一写锁（原 读锁检查→drop→写锁 有间隙）
        let mut nodes = self.nodes.write();
        let leaf = nodes
          .get_mut(&leaf_pid)
          .ok_or_else(|| Error::CorruptTree(format!("页 {leaf_pid} 不存在")))?;
        if leaf.has_space_for_kv(key.len() as u32, payload.len() as u32)
          && leaf.insert_key_value(key, payload, self.cmp)
        {
          break leaf_pid;
        }
      }
      // 预检后仍不足（并发写者插队 / 变长 upsert 放不下）：带父提示分裂，重启循环
      self.try_split_with_parent(leaf_pid, parent, None).await?;
      faced = true;
    };
    // 争用分裂采样（上游 BTreeExclusiveIterator::contentionSplit 的写入侧接线）。
    // 第一层门约 127/128 短路，故仅在极少数写访问上做槽位定位。
    let rand = self.next_rand();
    if update_gate(rand) {
      let slot = {
        let nodes = self.nodes.read();
        let leaf = nodes
          .get(&leaf_pid)
          .ok_or_else(|| Error::CorruptTree(format!("页 {leaf_pid} 不存在")))?;
        let mut exact = false;
        leaf.lower_bound(key, &mut exact, self.cmp)
      };
      let _ = self.contention_split(leaf_pid, slot, faced, rand).await?;
    }
    Ok(())
  }

  /// 删除（触发合并检查）
  pub async fn remove(&mut self, key: &[u8]) -> Result<bool> {
    let leaf_pid = self.find_leaf_optimistic(key).await?;
    let removed = {
      let mut nodes = self.nodes.write();
      let leaf = nodes
        .get_mut(&leaf_pid)
        .ok_or_else(|| Error::CorruptTree(format!("页 {leaf_pid} 不存在")))?;
      leaf.remove_key(key, self.cmp)
    };
    if removed && self.merge_if_needed(leaf_pid).await? {
      // 叶合并成功后其父层可能下溢：递归检查内层下溢（上游 EnsureUnderfullInnersForMerge）
      self.ensure_underfull_inners_for_merge(leaf_pid).await?;
    }
    Ok(removed)
  }

  /// 更新（删除 + 重插）
  pub async fn update(&mut self, key: &[u8], payload: &[u8]) -> Result<bool> {
    self.remove(key).await?;
    self.insert(key, payload).await.map(|_| true)
  }

  /// 原地更新（定长 payload；页内偏移写）
  pub async fn update_in_place(&mut self, key: &[u8], payload: &[u8]) -> Result<bool> {
    let leaf_pid = self.find_leaf_optimistic(key).await?;
    let mut nodes = self.nodes.write();
    let leaf = nodes
      .get_mut(&leaf_pid)
      .ok_or_else(|| Error::CorruptTree(format!("页 {leaf_pid} 不存在")))?;
    let mut exact = false;
    let pos = leaf.lower_bound(key, &mut exact, self.cmp);
    if !exact {
      return Ok(false);
    }
    let slot = leaf.slots[pos as usize];
    // 定长原位覆写：直接比较长度，免去旧版 `get_payload(pos).to_vec()` 的整段拷贝
    if slot.payload_length as usize == payload.len() {
      let off = slot.offset + slot.key_length;
      leaf.write_bytes(off, payload);
      Ok(true)
    } else {
      Ok(false)
    }
  }

  /// 页内更新（read_pid 对偶）
  pub async fn update_pid(&mut self, pid: PageId, payload: &[u8]) -> Result<bool> {
    let mut nodes = self.nodes.write();
    match nodes.get_mut(&pid) {
      Some(n) => {
        n.write_bytes(n.slots[0].offset + n.slots[0].key_length, payload);
        Ok(true)
      }
      None => Ok(false),
    }
  }

  // ---- SMO：分裂 ----

  /// 确保父节点有分裂空间（递归上溢检查；上游 EnsureSpaceForSplit）
  async fn ensure_space_for_split(
    &mut self,
    pid: PageId,
    key_len: u32,
    payload_len: u32,
  ) -> Result<bool> {
    let need = {
      let nodes = self.nodes.read();
      let node = nodes
        .get(&pid)
        .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
      !node.has_space_for_kv(key_len, payload_len)
    };
    if !need {
      return Ok(false);
    }
    // 分裂会改动父层分隔键：调用方须重走下行（键宿主可能已换叶）
    self.try_split(pid).await?;
    // 非终止兜底：`try_split` 对 `count <= 1` 的页直接返回（`tree.rs:409`，与
    // 上游 `trySplit` 的 `count <= 1 → return` 同构），既不切分本页也不抬升祖先；
    // 若本页 count<=1 且仍放不下记录，则调用方重走下行后必然回到同一状态 ——
    // `insert` 的 `continue` 会无限重试。此时返回错误而非空转。
    let stuck = {
      let nodes = self.nodes.read();
      match nodes.get(&pid) {
        Some(node) => node.hdr.count <= 1 && !node.has_space_for_kv(key_len, payload_len),
        None => false,
      }
    };
    if stuck {
      return Err(Error::CorruptTree(format!(
        "页 {pid} count<=1 且无空间（key+payload={}），无法分裂插入",
        key_len + payload_len
      )));
    }
    Ok(true)
  }

  /// 找父页与子页在父中的槽位（上游 `findParent` 的 slot 解析）。
  ///
  /// 路由键取子节点首全键；空子节点回退 `lower_fence`。返回 `pos == 父.count`
  /// 表示子是最右子（`right_most_child`）。用于分裂/合并的父指针更新。
  fn find_parent_and_slot(&self, pid: PageId) -> Option<(PageId, u32)> {
    let nodes = self.nodes.read();
    let child = nodes.get(&pid)?;
    let key: Vec<u8> = if child.hdr.count > 0 {
      let mut k = child.get_prefix().to_vec();
      k.extend_from_slice(child.get_key(0));
      k
    } else {
      child.get_lower_fence().to_vec()
    };
    nodes.iter().find_map(|(cand_pid, cand)| {
      if cand.hdr.is_leaf {
        return None;
      }
      let mut pos = 0u32;
      (cand.find_child(&key, &mut pos, self.cmp) == pid).then_some((*cand_pid, pos))
    })
  }

  /// 尝试分裂（新右页预分配在锁外 → 协议合规；普通分裂）
  pub async fn try_split(&mut self, pid: PageId) -> Result<()> {
    self.try_split_with_parent(pid, None, None).await
  }

  /// 尝试分裂（`parent_hint` = 下行父栈末位 O(1) 找父；`favored` = 争用分裂指定槽位）
  ///
  /// `favored` 与上游 `trySplit(to_split, favored_split_pos)` 一致：
  /// `Some(pos)` 且 `0 <= pos < count-1` 时切在 `pos`；`None` 或超界回退
  /// `find_separator`（`BTreeGeneric.cpp:51`；`-1` 即 `Some(-1)` 走回退）。
  pub async fn try_split_with_parent(
    &mut self,
    pid: PageId,
    parent_hint: Option<PageId>,
    favored: Option<i32>,
  ) -> Result<()> {
    let (sep, is_leaf, old_next) = {
      let nodes = self.nodes.read();
      let node = nodes
        .get(&pid)
        .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
      if node.hdr.count <= 1 {
        return Ok(()); // 单键不可分
      }
      let sep = match favored {
        // `pos == 0` 会让 `ldb_node::split_node` 的 `assert!(separator_slot > 0)`
        // 触发 panic（该断言与 master `BTreeNode.cpp:370` 中已注释掉的同名断言
        // 相悖 —— master 允许 0 号槽，见 CROSS）。`Some(-1)`（`last == -1` 回退）
        // 与 `Some(0)` 一并回退 `find_separator`：在本 crate 内只使用
        // `split_node` 能接受的槽位。
        Some(pos) if pos > 0 && (pos as u32) < node.hdr.count - 1 => SeparatorInfo {
          len: node.hdr.prefix_len + node.slots[pos as usize].key_length,
          slot: pos as u32,
          is_truncated: false,
        },
        _ => node.find_separator(self.append_bias, self.cmp),
      };
      // 叶分裂后新右叶接管原 next；原 next 叶的 prev 需由本函数改指新右叶
      let old_next = if node.hdr.is_leaf {
        node.hdr.right_most_child
      } else {
        EMPTY_NEIGHBOR
      };
      (sep, node.hdr.is_leaf, old_next)
    };
    let is_root = pid == self.root_pid;
    // 非根分裂必须保证父节点有空间放分隔键（上游 `TrySplit` 的
    // `p_guard->hasEnoughSpaceFor(space_needed_for_separator)`）：不足则先分裂父层，
    // 本次返回由调用方重走下行的语义（`insert` 的循环会重试）。旧实现直接
    // `split_node` → 父满时 `ldb_node` 断言 panic。
    if !is_root {
      let Some(ppid) = (match parent_hint {
        Some(p) => Some(p),
        None => self.find_parent_and_slot(pid).map(|(p, _)| p),
      }) else {
        return Err(Error::CorruptTree(format!("页 {pid} 的父页未找到")));
      };
      let parent_lacks_space = {
        let nodes = self.nodes.read();
        match nodes.get(&ppid) {
          Some(p) => !p.has_space_for_kv(sep.len, 8),
          None => return Err(Error::CorruptTree(format!("父页 {ppid} 不存在"))),
        }
      };
      if parent_lacks_space {
        // 递归上溯（树高量级；仅父满的冷路径触发，故 Box::pin 分配可接受）
        Box::pin(self.try_split_with_parent(ppid, None, None)).await?;
        return Ok(());
      }
      self.split_child(pid, ppid, &sep, is_leaf, old_next).await
    } else {
      self.split_root(pid, &sep, is_leaf, old_next).await
    }
  }

  /// 根分裂：新建内根（叶/内根统一走上游 `isMetaNode` 分支）
  async fn split_root(
    &mut self,
    pid: PageId,
    sep: &SeparatorInfo,
    is_leaf: bool,
    old_next: PageId,
  ) -> Result<()> {
    // 新页与源页形态一致（上游 `BTreeNode(!node->IsInner())`）
    let right_pid = self.alloc_node(is_leaf).await;
    let mut sep_key = [0u8; MAX_RECORD_SIZE as usize];
    let sep_len = sep.len as usize;
    let new_root_pid = {
      let mut nodes = self.nodes.write();
      let new_root_pid = self.next_pid;
      self.next_pid += 1;
      let mut new_root = BTreeNode::new(false, 0);
      new_root.set_fences(b"", b"", self.cmp);
      new_root.hdr.right_most_child = pid;
      nodes.insert(new_root_pid, new_root);
      // SAFETY: 三页两两不同（新根/右页/源页），且都已在上面写入 nodes，
      // Box 内容地址稳定；unwrap_unchecked 由「已插入」保证不空
      unsafe {
        let np: *mut BTreeNode = nodes.get_mut(&new_root_pid).unwrap_unchecked().as_mut();
        let nr: *mut BTreeNode = nodes.get_mut(&right_pid).unwrap_unchecked().as_mut();
        let node: *mut BTreeNode = nodes.get_mut(&pid).unwrap_unchecked().as_mut();
        (*node).get_separator_key(&mut sep_key[..sep_len], sep);
        // 0 号槽分隔只可能来自 count==2 叶，而根叶恒无 fence、count==2 必有空间容纳
        // 下一条 `≤MAX_RECORD_SIZE` 记录，故根叶永不需在 count==2 分裂；根内节点
        // `find_separator` 返回 `count/2 ≥ 1`。故根分裂恒 `sep.slot ≥ 1`。
        debug_assert!(sep.slot > 0, "根分裂分隔槽位必须 > 0");
        let ctx = SplitCtx {
          parent: &mut *np,
          node_right: &mut *nr,
          left_pid: pid,
          right_pid,
        };
        (*node).split_node(ctx, sep.slot, &sep_key[..sep_len], self.cmp);
      }
      new_root_pid
    };
    self.root_pid = new_root_pid;
    self.fix_right_neighbor_prev(right_pid, old_next);
    Ok(())
  }

  /// 非根分裂：父页由 `split_node` 就地插入分隔键
  async fn split_child(
    &mut self,
    pid: PageId,
    ppid: PageId,
    sep: &SeparatorInfo,
    is_leaf: bool,
    old_next: PageId,
  ) -> Result<()> {
    let right_pid = self.alloc_node(is_leaf).await;
    let mut sep_key = [0u8; MAX_RECORD_SIZE as usize];
    let sep_len = sep.len as usize;
    {
      let mut nodes = self.nodes.write();
      // SAFETY: 节点经 Box 存储地址稳定；pid/ppid/right_pid 两两不同（父 != 子，
      // 右为新分配页），且都存在于 nodes；unwrap_unchecked 由存在性保证不空
      unsafe {
        let np: *mut BTreeNode = nodes.get_mut(&ppid).unwrap_unchecked().as_mut();
        let nr: *mut BTreeNode = nodes.get_mut(&right_pid).unwrap_unchecked().as_mut();
        let node: *mut BTreeNode = nodes.get_mut(&pid).unwrap_unchecked().as_mut();
        (*node).get_separator_key(&mut sep_key[..sep_len], sep);
        if sep.slot == 0 {
          // 0 号槽分隔只来自 count==2 叶（`find_separator` 对 inner 返回 ≥1）：
          // 长 fence 使 2 记录叶也可能放不下下一条记录，而 `split_node` 的
          // `assert!(separator_slot > 0)` 会 panic（master `BTreeNode.cpp:370`
          // 明确允许 0 号槽），改走本 crate 的等价两记录分裂
          split_leaf_two(
            &mut *node,
            &mut *np,
            &mut *nr,
            pid,
            right_pid,
            &sep_key[..sep_len],
            self.cmp,
          );
        } else {
          let ctx = SplitCtx {
            parent: &mut *np,
            node_right: &mut *nr,
            left_pid: pid,
            right_pid,
          };
          (*node).split_node(ctx, sep.slot, &sep_key[..sep_len], self.cmp);
        }
      }
    }
    self.fix_right_neighbor_prev(right_pid, old_next);
    Ok(())
  }

  /// 叶分裂后：原 `old_next` 叶的 `prev` 必须改指新右叶（上游 `trySplit` 对
  /// `new_child->next_leaf_node` 的 `prev_leaf_node` 修正）。
  fn fix_right_neighbor_prev(&mut self, right_pid: PageId, old_next: PageId) {
    if old_next == EMPTY_NEIGHBOR {
      return;
    }
    let mut nodes = self.nodes.write();
    if let Some(nn) = nodes.get_mut(&old_next)
      && nn.hdr.is_leaf
      && nn.hdr.prev_leaf_node != right_pid
    {
      nn.hdr.prev_leaf_node = right_pid;
    }
  }

  // ---- SMO：合并 ----

  /// 合并检查（低于 `SIZE_UNDER_FULL` 触发；上游 MergeIfNeeded）
  ///
  /// 上游 `mergeIfNeeded`（`BTreeGenericIterator.hpp:583`）与 `tryMerge`
  /// （`BTreeGeneric.cpp:223`）均为 `freeSpaceAfterCompaction() >= underFullSize`
  /// 才尝试合并（空闲越大 = 越 underfull）。旧版用 `>` 会漏掉恰好等于阈值的节点。
  pub async fn merge_if_needed(&mut self, pid: PageId) -> Result<bool> {
    let under = {
      let nodes = self.nodes.read();
      let Some(n) = nodes.get(&pid) else {
        return Ok(false);
      };
      pid != self.root_pid && n.free_space_after_compaction() >= SIZE_UNDER_FULL
    };
    if !under {
      return Ok(false);
    }
    self.try_merge(pid).await
  }

  /// 尝试合并（找右兄弟：沿用父槽位邻居 —— 上游 `tryMerge` 的 `merge_right`）
  ///
  /// 叶/内节点统一处理：右兄弟 = 父中 `slot+1` 子，或（slot 为末槽时）父的
  /// `right_most_child`；`slot` 即父的 `right_most_child` 时无右兄弟。
  pub async fn try_merge(&mut self, pid: PageId) -> Result<bool> {
    let Some((ppid, slot)) = self.find_parent_and_slot(pid) else {
      return Ok(false);
    };
    let right = {
      let nodes = self.nodes.read();
      let parent = nodes
        .get(&ppid)
        .ok_or_else(|| Error::CorruptTree(format!("页 {ppid} 不存在")))?;
      if parent.hdr.is_leaf || slot >= parent.hdr.count {
        None
      } else if slot + 1 < parent.hdr.count {
        Some(parent.get_child(slot + 1))
      } else {
        Some(parent.hdr.right_most_child)
      }
    };
    let Some(right) = right else {
      return Ok(false);
    };
    self.merge_left_into_right(pid, right).await
  }

  /// 左并入右（上游 `BTreeNode::merge` 的 Result 化；`ldb_node::merge_nodes` 端口）。
  ///
  /// `left` 的全部记录并入 `right`（**不丢右节点原有记录** —— 旧实现 `r.hdr.count = 0`
  /// 会静默丢弃），父分隔键、叶链 `prev/next`、槽位由 `merge_nodes` 一并更新；
  /// 成功后从节点表移除 `left` 并修正其左邻叶的 `next` 指针。
  pub async fn merge_left_into_right(&mut self, left: PageId, right: PageId) -> Result<bool> {
    if left == right {
      return Ok(false);
    }
    let Some((ppid, slot)) = self.find_parent_and_slot(left) else {
      return Ok(false);
    };
    if ppid == left || ppid == right {
      return Ok(false);
    }
    let (merged, left_prev) = {
      let mut nodes = self.nodes.write();
      if !nodes.contains_key(&ppid) || !nodes.contains_key(&left) || !nodes.contains_key(&right) {
        return Ok(false);
      }
      let (p, l, r) = nodes.get_disjoint_mut3(ppid, left, right);
      // left 必须是父中 slot 对应的子；slot 越界（left 为父最右子）无右兄弟可并
      if slot >= p.hdr.count || p.get_child(slot) != left {
        return Ok(false);
      }
      let prev = l.hdr.prev_leaf_node;
      let ok = l.merge_nodes(slot, p, r, self.cmp);
      (ok, prev)
    };
    if merged {
      // 叶链修正：left 的左邻叶 next 指向 left，left 被移除后须改指 right
      if left_prev != EMPTY_NEIGHBOR {
        let mut nodes = self.nodes.write();
        if let Some(prev) = nodes.get_mut(&left_prev)
          && prev.hdr.is_leaf
          && prev.hdr.right_most_child == left
        {
          prev.hdr.right_most_child = right;
        }
      }
      self.nodes.write().remove(&left);
    }
    Ok(merged)
  }

  /// 确保内层父节点低水位（递归下溢检查；上游 EnsureUnderfullInnersForMerge）
  ///
  /// 从 `pid` 向上逐层：内层节点 underfull 即尝试与右兄弟合并。上游只处理
  /// `to_merge` 及其父链；本实现遍历整条祖先链（合并更多 but 均合法）。
  pub async fn ensure_underfull_inners_for_merge(&mut self, pid: PageId) -> Result<()> {
    let mut cur = pid;
    while let Some((ppid, _pos)) = self.find_parent_and_slot(cur) {
      let under = {
        let nodes = self.nodes.read();
        match nodes.get(&cur) {
          Some(n) => !n.hdr.is_leaf && n.free_space_after_compaction() >= SIZE_UNDER_FULL,
          None => false,
        }
      };
      if under && cur != self.root_pid {
        let _ = self.try_merge(cur).await?;
      }
      cur = ppid;
    }
    Ok(())
  }

  /// XMerge：跨多兄弟级联合并（master `BTreeGeneric::XMerge`，`BTreeGeneric.cpp:417`）。
  ///
  /// 门：当前页填满率（compaction 后）`>= 0.9` 直接不合并（`:422`）。随后从
  /// `pid` 起把 underfull 叶逐页并入右兄弟（`merge_nodes`，记录全保留），
  /// 直至右邻不存在或合并空间不足。
  pub async fn x_merge(&mut self, pid: PageId) -> Result<bool> {
    let mut any = false;
    let mut cur = pid;
    loop {
      if self.fill_factor_after_compaction(cur) >= 0.9 {
        break;
      }
      let Some(right) = self.right_sibling(cur) else {
        break;
      };
      if !self.merge_left_into_right(cur, right).await? {
        break;
      }
      any = true;
      cur = right;
    }
    Ok(any)
  }

  // ---- Contention Split（CIDR'21；采样状态机在 ldb_node）----

  /// 热点争用分裂：一次写访问的争用采样 + 命中后的 favored 分裂。
  ///
  /// 逐行对应上游 `BTreeExclusiveIterator::contentionSplit`
  /// （`BTreeGenericIterator.hpp:526-557`）：
  /// - `rand` 由调用方提供（上游 `RandomGenerator::getRandU64()`），同一值用于
  ///   两层采样门（`ldb_node::ContentionTracker::note_access`）；
  /// - `slot` = 本次被修改的槽位（上游 `cur`）；`faced_contention` = 本次访问是否
  ///   经历过争用/重启；
  /// - `note_access` 返回 `Some(pos)`（`min(last_modified_pos, slot)`；`-1` 走
  ///   回退）时按 favored pos 分裂，正是上游传给 `trySplit` 的
  ///   `favored_split_pos`。
  ///
  /// 返回 `true` 表示触发了分裂。
  pub async fn contention_split(
    &mut self,
    pid: PageId,
    slot: u32,
    faced_contention: bool,
    rand: u64,
  ) -> Result<bool> {
    let action = {
      let mut nodes = self.nodes.write();
      let node = nodes
        .get_mut(&pid)
        .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
      let count = node.hdr.count;
      node
        .hdr
        .contention
        .note_access(slot, faced_contention, rand, count)
    };
    let Some(favored) = action else {
      return Ok(false);
    };
    self.try_split_with_parent(pid, None, Some(favored)).await?;
    Ok(true)
  }

  // ---- 扫描 ----

  /// 升序扫描（`until` 键或全部；迭代回调）
  pub async fn scan_ascending(
    &self,
    start: &[u8],
    until: &[u8],
    mut cb: impl FnMut(&[u8], &[u8]) -> bool,
  ) -> Result<usize> {
    let mut n = 0usize;
    let mut pid = self.find_leaf_optimistic(start).await?;
    let mut first = true;
    loop {
      // 拷出本叶键值（持读锁最短化：回调在锁外执行）
      let entries: Vec<(Vec<u8>, Vec<u8>)> = {
        let nodes = self.nodes.read();
        let leaf = nodes
          .get(&pid)
          .ok_or_else(|| Error::CorruptTree(format!("页 {pid} 不存在")))?;
        let mut exact = false;
        let pos = if first {
          leaf.lower_bound(start, &mut exact, self.cmp)
        } else {
          0
        };
        first = false;
        let prefix = leaf.get_prefix().to_vec();
        (pos..leaf.hdr.count)
          .map(|i| {
            let mut full = prefix.clone();
            full.extend_from_slice(leaf.get_key(i));
            (full, leaf.get_payload(i).to_vec())
          })
          .collect()
      };
      let mut stop = false;
      for (k, v) in entries {
        if !until.is_empty() && k.as_slice() >= until {
          stop = true;
          break;
        }
        n += 1;
        if !cb(&k, &v) {
          return Ok(n);
        }
      }
      if stop {
        return Ok(n);
      }
      // 右兄弟链推进
      let next = {
        let nodes = self.nodes.read();
        let Some(leaf) = nodes.get(&pid) else {
          return Ok(n);
        };
        if leaf.hdr.right_most_child != EMPTY_NEIGHBOR {
          Some(leaf.hdr.right_most_child)
        } else {
          None
        }
      };
      match next {
        Some(np) => pid = np,
        None => return Ok(n),
      }
    }
  }

  /// 降序扫描（升序收集后倒序回调；生产可双链反向遍历）
  pub async fn scan_descending(
    &self,
    start: &[u8],
    until: &[u8],
    mut cb: impl FnMut(&[u8], &[u8]) -> bool,
  ) -> Result<usize> {
    // 语义：回调 (until, start] 区间的降序 —— 升序扫 [until, start) 再倒序
    let mut collected = Vec::new();
    self
      .scan_ascending(until, start, |k, v| {
        collected.push((k.to_vec(), v.to_vec()));
        true
      })
      .await?;
    collected.reverse();
    let mut n = 0;
    for (k, v) in collected {
      n += 1;
      if !cb(&k, &v) {
        return Ok(n);
      }
    }
    Ok(n)
  }

  /// 全节点遍历（迭代回调每页）
  pub async fn iterate_all_nodes(&self, mut cb: impl FnMut(&BTreeNode) -> bool) -> Result<usize> {
    let nodes = self.nodes.read();
    let mut n = 0;
    for node in nodes.values() {
      n += 1;
      if !cb(node) {
        break;
      }
    }
    Ok(n)
  }

  /// untils 多段扫描（每段独立区间）
  pub async fn iterate_untils(
    &self,
    ranges: &[(Vec<u8>, Vec<u8>)],
    mut cb: impl FnMut(&[u8], &[u8]) -> bool,
  ) -> Result<usize> {
    let mut total = 0;
    for (lo, hi) in ranges {
      total += self.scan_ascending(lo, hi, |k, v| cb(k, v)).await?;
    }
    Ok(total)
  }

  /// 条目计数（全树扫描）
  pub async fn count_entries(&self) -> Result<u64> {
    let nodes = self.nodes.read();
    Ok(
      nodes
        .values()
        .filter(|n| n.hdr.is_leaf)
        .map(|n| n.hdr.count as u64)
        .sum(),
    )
  }

  /// 页计数
  pub async fn count_pages(&self) -> Result<u64> {
    Ok(self.nodes.read().len() as u64)
  }

  // ---- 内部工具 ----

  /// 节点 compaction 后填满率（上游 `fillFactorAfterCompaction`）
  fn fill_factor_after_compaction(&self, pid: PageId) -> f64 {
    let nodes = self.nodes.read();
    match nodes.get(&pid) {
      Some(n) => 1.0 - f64::from(n.free_space_after_compaction()) / PAGE_SIZE as f64,
      None => 1.0,
    }
  }

  /// 叶的右邻（叶 next 链）；非叶或最右返回 `None`
  fn right_sibling(&self, pid: PageId) -> Option<PageId> {
    let nodes = self.nodes.read();
    let n = nodes.get(&pid)?;
    if n.hdr.is_leaf && n.hdr.right_most_child != EMPTY_NEIGHBOR {
      Some(n.hdr.right_most_child)
    } else {
      None
    }
  }
}

/// 两记录叶分裂（分隔键落在 0 号槽）。
///
/// `ldb_node::split_node` 断言 `separator_slot > 0`，对「左页只剩 1 条记录」
/// 的最小分裂（0 号槽）不可用；master `BTreeNode.cpp:370` 已把同名断言注释掉，
/// 明确允许 0 号槽；ZLeanStore 的 `FindSeparator`（`node.cc:458`）在
/// `append_bias` 下对 `count == 2` 也返回 slot 0。故本 crate 就地实现等价分裂：
/// 左页保留 `key[0]`，右页得 `key[1]`，父层分隔键 = 全键 `key[0]`。记录一律经
/// `copy_key_value_range` 跨前缀拷贝，`space_used` 由 `set_fences` + 拷贝精确
/// 记账（与 `split_node` 同构）。
///
/// 复用 `split_node` 的叶分支语义 + 父槽位改写顺序（先把指向左页的子指针
/// 改为右页，再插入分隔键 → 左页）。
fn split_leaf_two(
  left: &mut BTreeNode,
  parent: &mut BTreeNode,
  right: &mut BTreeNode,
  left_pid: PageId,
  right_pid: PageId,
  sep_key: &[u8],
  cmp: Cmp,
) {
  debug_assert!(
    left.hdr.is_leaf && left.hdr.count == 2,
    "仅用于 count==2 叶"
  );
  // 父层：把指向 left 的子指针改为 right
  let mut found = false;
  let old_slot = parent.lower_bound(sep_key, &mut found, cmp);
  if old_slot == parent.hdr.count {
    debug_assert_eq!(parent.hdr.right_most_child, left_pid);
    parent.hdr.right_most_child = right_pid;
  } else {
    debug_assert_eq!(parent.get_child(old_slot), left_pid);
    let off = parent.slots[old_slot as usize].offset + parent.slots[old_slot as usize].key_length;
    parent.write_bytes(off, &right_pid.to_le_bytes());
  }
  assert!(
    parent.insert_key_value(sep_key, &left_pid.to_le_bytes(), cmp),
    "父节点无空间插入分隔键"
  );

  // 左页：仅保留 key[0]（原地改写 —— 先构造临时页再整页覆盖）
  let mut page = left.hdr.page;
  page.dirty = true;
  let idx = left.hdr.page.idx_id;
  let mut tmp = BTreeNode::new(true, idx);
  tmp.hdr.page = page;
  tmp.set_fences(left.get_lower_fence(), sep_key, cmp);
  left.copy_key_value_range(&mut tmp, 0, 0, 1);
  tmp.hdr.prev_leaf_node = left.hdr.prev_leaf_node;
  tmp.hdr.right_most_child = right_pid;
  tmp.make_hint_all();
  // 右页：取得 key[1]，接管原 next
  right.set_fences(sep_key, left.get_upper_fence(), cmp);
  left.copy_key_value_range(right, 0, 1, 1);
  right.hdr.prev_leaf_node = left_pid;
  right.hdr.right_most_child = left.hdr.right_most_child;
  right.make_hint_all();
  BTreeNode::copy_node_content(left, &mut tmp);
}

/// 三 disjoint 可变借用（合并需同时改父 / 左 / 右）
trait GetDisjoint3 {
  fn get_disjoint_mut3(
    &mut self,
    a: PageId,
    b: PageId,
    c: PageId,
  ) -> (&mut BTreeNode, &mut BTreeNode, &mut BTreeNode);
}

impl GetDisjoint3 for NodeSlot {
  fn get_disjoint_mut3(
    &mut self,
    a: PageId,
    b: PageId,
    c: PageId,
  ) -> (&mut BTreeNode, &mut BTreeNode, &mut BTreeNode) {
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_ne!(a, c);
    // SAFETY: a/b/c 两两不同且 Box 内容地址稳定，三个可变引用不别名
    unsafe {
      let pa: *mut BTreeNode = self.get_mut(&a).unwrap_unchecked().as_mut();
      let pb: *mut BTreeNode = self.get_mut(&b).unwrap_unchecked().as_mut();
      let pc: *mut BTreeNode = self.get_mut(&c).unwrap_unchecked().as_mut();
      (&mut *pa, &mut *pb, &mut *pc)
    }
  }
}
