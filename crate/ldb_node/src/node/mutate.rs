//! 节点变更：fence、插入/删除、碎片压缩、分裂与合并。

use std::{
  mem::size_of,
  ptr::{copy_nonoverlapping, from_mut, from_ref},
};

use ldb_simd::{common_prefix, get_head, make_hint};
use ldb_type::PAGE_SIZE;

use super::{BTreeNode, Cmp, FenceKeySlot, NUM_SLOTS, PageSlot};

impl BTreeNode {
  /// 写入一个 fence 键（数据自页尾向前分配），返回槽位描述
  pub fn insert_fence(&mut self, key: &[u8]) -> FenceKeySlot {
    // 上游 `InsertFence`：assert(FreeSpace() >= key.size())；fence 只在新节点
    // 或分裂临时节点上写入，此时空闲区完整
    debug_assert!(self.free_space() >= key.len() as u32);
    self.hdr.data_offset -= key.len() as u32;
    self.hdr.space_used += key.len() as u32;
    let fk = FenceKeySlot {
      offset: self.hdr.data_offset,
      len: key.len() as u32,
    };
    self.write_bytes_unchecked(fk.offset, key);
    fk
  }

  /// 设置双 fence 并计算公共前缀（仅 MEMCMP 路径）
  pub fn set_fences(&mut self, lower: &[u8], upper: &[u8], cmp: Cmp) {
    let lo = self.insert_fence(lower);
    self.hdr.lower_fence = lo;
    let up = self.insert_fence(upper);
    self.hdr.upper_fence = up;
    // 复用 ldb_simd::common_prefix（与 `FindSeparator` 同一实现）；上游
    // `SetFences` 的逐字节循环是同一语义，重复实现会漂移
    self.hdr.prefix_len = if cmp.memcmp {
      common_prefix(lower, upper) as u32
    } else {
      0
    };
  }

  /// 存储记录数据（含前缀剥离）
  pub fn store_record_data(&mut self, slot_id: u32, full_key: &[u8], payload: &[u8]) {
    debug_assert!(
      full_key.len() >= self.hdr.prefix_len as usize,
      "key 短于节点前缀（路由契约被破坏）"
    );
    let k = &full_key[self.hdr.prefix_len as usize..];
    self.store_record_data_without_prefix(slot_id, k, payload);
  }

  /// 存储记录数据（key 已去前缀）
  pub fn store_record_data_without_prefix(&mut self, slot_id: u32, key: &[u8], payload: &[u8]) {
    self.store_record_data_joined(slot_id, key, &[], payload);
  }

  /// 存储「两段拼接」的去前缀 key（避免 `copy_key_value` 为拼全键分配 Vec）。
  ///
  /// 线上仍是连续布局：`key_a || key_b || payload`。
  fn store_record_data_joined(&mut self, slot_id: u32, key_a: &[u8], key_b: &[u8], payload: &[u8]) {
    let key_len = key_a.len() + key_b.len();
    let required = (key_len + payload.len()) as u32;
    debug_assert!(
      self.free_space() >= required,
      "写入前必须由调用方 reserve/compactify"
    );
    debug_assert!((slot_id as usize) < NUM_SLOTS);
    self.hdr.data_offset -= required;
    self.hdr.space_used += required;
    let off = self.hdr.data_offset;
    let head = if key_b.is_empty() {
      get_head(key_a)
    } else {
      head_joined(key_a, key_b)
    };
    self.slots[slot_id as usize] = PageSlot {
      offset: off,
      key_length: key_len as u32,
      payload_length: payload.len() as u32,
      head,
    };
    self.write_bytes_unchecked(off, key_a);
    self.write_bytes_unchecked(off + key_a.len() as u32, key_b);
    self.write_bytes_unchecked(off + key_len as u32, payload);
  }

  /// 插入 KV（需要则先碎片压缩；重复键 upsert —— 同长原位覆写，变长删旧槽重插）。
  ///
  /// 返回 `false` 表示节点容量不足（记录本身或槽位容量），调用方须分裂；
  /// 这是对上游 `InsertKeyValue` 静默忽略失败/重复键的修正（上游重复键
  /// 直接 `return`，`ldb_btree` 的 update = remove+insert 需要 upsert）。
  pub fn insert_key_value(&mut self, key: &[u8], payload: &[u8], cmp: Cmp) -> bool {
    // 单条记录不可能超过整页；拦截 >PAGE_SIZE 的入参，避免后续 u32 截断/越界写
    if key.len() > PAGE_SIZE || payload.len() > PAGE_SIZE {
      return false;
    }
    let prefix_len = self.hdr.prefix_len as usize;
    // key 不覆盖节点前缀 = key 不在本节点 fence 范围（上游会无符号下溢后越界写）
    if key.len() < prefix_len {
      return false;
    }
    debug_assert!(!cmp.memcmp || key.starts_with(self.get_prefix()));

    let mut found = false;
    let slot_id = self.lower_bound(key, &mut found, cmp);
    if found {
      let old_len = self.slots[slot_id as usize].payload_length as usize;
      if old_len == payload.len() {
        let off = self.slots[slot_id as usize].offset + self.slots[slot_id as usize].key_length;
        self.write_bytes_unchecked(off, payload);
        return true;
      }
      // 变长：先确认替换后仍放得下（旧记录字节 + 复用的 16B 槽位），再删旧槽；
      // 否则保留旧值并返回 false —— 不得让失败的 upsert 丢掉原记录
      let old_kv = self.slots[slot_id as usize]
        .key_length
        .saturating_add(self.slots[slot_id as usize].payload_length);
      let new_kv = (key.len() - prefix_len + payload.len()) as u32;
      if new_kv > self.free_space_after_compaction().saturating_add(old_kv) {
        return false;
      }
      self.remove_slot(slot_id);
    }
    let need = self.space_required_for_kv(key.len() as u32, payload.len() as u32);
    if need > self.free_space() {
      if need > self.free_space_after_compaction() {
        return false;
      }
      self.compactify(cmp);
    }
    // 槽位后移一格（copy_within 单次搬移）。`s == c` 是尾部插入 / 删除尾槽后
    // 重插的常见路径：跳过空搬移（`s == NUM_SLOTS` 的边界由空间检查排除）
    let (s, c) = (slot_id as usize, self.hdr.count as usize);
    debug_assert!(s <= c);
    if s < c {
      self.slots.copy_within(s..c, s + 1);
    }
    self.store_record_data(slot_id, key, payload);
    self.hdr.count += 1;
    self.rebuild_hint(slot_id);
    true
  }

  /// 删除槽位
  pub fn remove_slot(&mut self, slot_id: u32) -> bool {
    debug_assert!(slot_id < self.hdr.count);
    let slot_ref = self.slots[slot_id as usize];
    self.hdr.space_used -= slot_ref.key_length + slot_ref.payload_length;
    let (s, c) = (slot_id as usize, self.hdr.count as usize);
    self.slots.copy_within(s + 1..c, s);
    self.hdr.count -= 1;
    self.make_hint_all();
    true
  }

  /// 按 key 删除
  pub fn remove_key(&mut self, key: &[u8], cmp: Cmp) -> bool {
    let mut found = false;
    let slot_id = self.lower_bound(key, &mut found, cmp);
    match found {
      true => self.remove_slot(slot_id),
      false => false,
    }
  }

  /// 单条 KV 跨前缀拷贝（前缀增长场景用整键；前缀收缩场景直接拼接两段）
  pub fn copy_key_value(&self, dst: &mut BTreeNode, src_slot: u32, dst_slot: u32) {
    let src_prefix = self.hdr.prefix_len as usize;
    let dst_prefix = dst.hdr.prefix_len as usize;
    let key = self.get_key(src_slot);
    let payload = self.get_payload(src_slot);
    if src_prefix >= dst_prefix {
      // 前缀收缩或相等：dst 保留 src_prefix[dst_prefix..] + key
      debug_assert_eq!(
        self.get_prefix()[..dst_prefix],
        dst.get_prefix()[..dst_prefix]
      );
      dst.store_record_data_joined(dst_slot, &self.get_prefix()[dst_prefix..], key, payload);
    } else {
      // 前缀增长：dst 前缀比记录更长，剥去 key 的差量（上游 `key_len - diff`）
      let diff = dst_prefix - src_prefix;
      debug_assert!(key.len() >= diff);
      dst.store_record_data_joined(dst_slot, &[], &key[diff..], payload);
    }
    dst.hdr.count += 1;
  }

  /// 范围拷贝 KV（处理前缀差量）
  pub fn copy_key_value_range(
    &mut self,
    dst: &mut BTreeNode,
    dst_slot: u32,
    src_slot: u32,
    src_count: u32,
  ) {
    assert!(
      dst.hdr.count as usize + src_count as usize <= NUM_SLOTS,
      "槽位容量溢出（调用方必须预留槽位）"
    );
    let n = (self.hdr.prefix_len.min(dst.hdr.prefix_len)) as usize;
    debug_assert!(self.get_prefix()[..n] == dst.get_prefix()[..n]);
    if self.hdr.prefix_len <= dst.hdr.prefix_len {
      // 前缀变长：剥去差量
      let diff = (dst.hdr.prefix_len - self.hdr.prefix_len) as usize;
      for idx in 0..src_count as usize {
        let src_key = self.get_key(src_slot + idx as u32);
        debug_assert!(src_key.len() >= diff, "记录全键短于目标节点前缀");
        let key = &src_key[diff..];
        let payload = self.get_payload(src_slot + idx as u32);
        dst.store_record_data_without_prefix(dst_slot + idx as u32, key, payload);
      }
      dst.hdr.count += src_count;
    } else {
      for idx in 0..src_count {
        self.copy_key_value(dst, src_slot + idx, dst_slot + idx);
      }
    }
    if !self.is_inner() {
      let _ = self.zero_free_space();
    }
  }

  /// 整页内容拷贝（dst <- src）
  pub fn copy_node_content(dst: &mut BTreeNode, src: &mut BTreeNode) {
    // SAFETY: 同类型同大小 4096B 节点，按字节搬移
    unsafe {
      copy_nonoverlapping(
        from_ref(src).cast::<u8>(),
        from_mut(dst).cast::<u8>(),
        PAGE_SIZE,
      );
    }
    // 上游 CopyNodeContent 对叶节点双侧清零空闲区（利于页压缩）
    if !dst.is_inner() && !src.is_inner() {
      let _ = dst.zero_free_space();
      let _ = src.zero_free_space();
    }
  }

  /// 碎片压缩（重排数据区连续存放）。
  ///
  /// `tmp` 继承 `self.hdr.page`（GSN/写历史/idx_id），只把 `dirty` 置 1；
  /// 上游 `BTreeNode tmp(is_leaf, idx_id)` 会把这些字段清零 —— 对 WAL 的
  /// `p_gsn` 与页归属 `idx_id` 是数据丢失，本版显式保留。
  pub fn compactify(&mut self, cmp: Cmp) {
    let before = self.free_space_after_compaction();
    let mut page = self.hdr.page;
    page.dirty = true;
    let mut tmp = BTreeNode::new(self.hdr.is_leaf, self.hdr.page.idx_id);
    tmp.hdr.page = page;
    tmp.set_fences(self.get_lower_fence(), self.get_upper_fence(), cmp);
    let count = self.hdr.count;
    self.copy_key_value_range(&mut tmp, 0, 0, count);
    tmp.hdr.right_most_child = self.hdr.right_most_child;
    tmp.hdr.prev_leaf_node = self.hdr.prev_leaf_node;
    Self::copy_node_content(self, &mut tmp);
    self.make_hint_all();
    debug_assert_eq!(self.free_space_after_compaction(), before);
  }

  /// 分裂：`self`（左）+ `node_right`，分隔键上提到 `parent`
  ///
  /// # Panics
  /// separator_slot 越界或父槽位不匹配（上游 assert 等价）
  pub fn split_node(
    &mut self,
    ctx: super::SplitCtx<'_>,
    separator_slot: u32,
    sep_key: &[u8],
    cmp: Cmp,
  ) {
    let super::SplitCtx {
      parent,
      node_right,
      left_pid,
      right_pid,
    } = ctx;
    assert!(separator_slot > 0);
    assert!(separator_slot < self.hdr.count);
    // 上游 ZLeanStore SplitNode 的叶链自检（Ensure → Rust debug_assert）
    if self.hdr.is_leaf {
      debug_assert!(!self.hdr.is_upper_fence_infinity() || !self.hdr.has_right_neighbor());
      debug_assert!(!self.hdr.is_lower_fence_infinity() || !self.hdr.has_left_neighbor());
    }
    let mut page = self.hdr.page;
    page.dirty = true;
    let mut tmp = BTreeNode::new(self.hdr.is_leaf, self.hdr.page.idx_id);
    tmp.hdr.page = page;

    tmp.set_fences(self.get_lower_fence(), sep_key, cmp);
    node_right.set_fences(sep_key, self.get_upper_fence(), cmp);

    // 父节点更新分隔（右移位 or 原位替换）
    let mut found = false;
    let old_slot = parent.lower_bound(sep_key, &mut found, cmp);
    if old_slot == parent.hdr.count {
      assert_eq!(parent.hdr.right_most_child, left_pid);
      parent.hdr.right_most_child = right_pid;
    } else {
      assert_eq!(parent.get_child(old_slot), left_pid);
      parent.write_bytes_unchecked(
        parent.slots[old_slot as usize].offset + parent.slots[old_slot as usize].key_length,
        &right_pid.to_le_bytes(),
      );
    }
    // 上游 `assert(parent->canInsert(sepLength, sizeof(SwipType)))`：父节点
    // 必须在分裂前预留一个分隔键槽位；失败即 SMO 契约被破坏，不得静默丢子指针
    assert!(
      parent.insert_key_value(sep_key, &left_pid.to_le_bytes(), cmp),
      "父节点无空间插入分隔键"
    );

    if self.hdr.is_leaf {
      let count = self.hdr.count;
      self.copy_key_value_range(&mut tmp, 0, 0, separator_slot + 1);
      self.copy_key_value_range(node_right, 0, tmp.hdr.count, count - tmp.hdr.count);
      tmp.hdr.prev_leaf_node = self.hdr.prev_leaf_node;
      tmp.hdr.right_most_child = right_pid; // 叶：next 指针
      node_right.hdr.prev_leaf_node = left_pid;
      node_right.hdr.right_most_child = self.hdr.right_most_child;
    } else {
      // 内节点：分隔键上提，不留在子节点
      self.copy_key_value_range(&mut tmp, 0, 0, separator_slot);
      self.copy_key_value_range(
        node_right,
        0,
        tmp.hdr.count + 1,
        self.hdr.count - tmp.hdr.count - 1,
      );
      tmp.hdr.right_most_child = self.get_child(tmp.hdr.count);
      node_right.hdr.right_most_child = self.hdr.right_most_child;
    }
    tmp.make_hint_all();
    node_right.make_hint_all();
    Self::copy_node_content(self, &mut tmp);
  }

  /// 合并 self（左）入 `right`；成功则左节点可释放
  ///
  /// 空间/槽位上界超页则放弃合并（返回 false）。
  pub fn merge_nodes(
    &mut self,
    left_slot_id: u32,
    parent: &mut BTreeNode,
    right: &mut BTreeNode,
    cmp: Cmp,
  ) -> bool {
    debug_assert_eq!(self.hdr.is_leaf, right.hdr.is_leaf);
    debug_assert!(left_slot_id < parent.hdr.count);
    // 槽位容量上界（零长记录只占槽位；字节上界放得下也可能超槽位数组）
    let extra_slots = u32::from(self.is_inner());
    if self.hdr.count + right.hdr.count + extra_slots > NUM_SLOTS as u32 {
      return false;
    }

    let mut page = right.hdr.page;
    page.dirty = true;
    let mut tmp = BTreeNode::new(self.hdr.is_leaf, self.hdr.page.idx_id);
    tmp.hdr.page = page;
    tmp.set_fences(self.get_lower_fence(), right.get_upper_fence(), cmp);

    // 空间上界（保守估算：前缀膨胀 + 槽位表）
    let hdr_sz = size_of::<super::BTreeNodeHeader>() as u64;
    let slot_sz = size_of::<PageSlot>() as u64;
    let mut bound = hdr_sz
      + u64::from(self.hdr.space_used)
      + u64::from(right.hdr.space_used)
      + u64::from(self.hdr.prefix_len.saturating_sub(tmp.hdr.prefix_len))
        * u64::from(self.hdr.count)
      + u64::from(right.hdr.prefix_len.saturating_sub(tmp.hdr.prefix_len))
        * u64::from(right.hdr.count)
      + slot_sz * u64::from(self.hdr.count + right.hdr.count);
    if self.is_inner() {
      // 内节点：父分隔键下放
      let extra = u64::from(parent.hdr.prefix_len)
        + u64::from(parent.slots[left_slot_id as usize].key_length);
      bound += extra + slot_sz + 8;
    }
    if bound > PAGE_SIZE as u64 {
      return false;
    }

    let count = self.hdr.count;
    self.copy_key_value_range(&mut tmp, 0, 0, count);
    if self.hdr.is_leaf {
      right.copy_key_value_range(&mut tmp, count, 0, right.hdr.count);
      tmp.hdr.right_most_child = right.hdr.right_most_child;
      tmp.hdr.prev_leaf_node = self.hdr.prev_leaf_node;
    } else {
      // 父分隔键 + 左子 right_most 下放
      parent.write_bytes_unchecked(
        parent.slots[left_slot_id as usize].offset + parent.slots[left_slot_id as usize].key_length,
        &self.hdr.right_most_child.to_le_bytes(),
      );
      parent.copy_key_value(&mut tmp, left_slot_id, count);
      let tmp_cnt = tmp.hdr.count;
      right.copy_key_value_range(&mut tmp, tmp_cnt, 0, right.hdr.count);
      tmp.hdr.right_most_child = right.hdr.right_most_child;
    }
    tmp.make_hint_all();
    Self::copy_node_content(right, &mut tmp);
    parent.remove_slot(left_slot_id);
    true
  }

  // ---- 内部工具 ----

  /// 重建全部 hint（就地写入，无堆分配 —— remove_slot 每次调用走此路径）
  pub fn make_hint_all(&mut self) {
    let count = self.hdr.count as usize;
    make_hint(&mut self.hdr.hints, count, |i| self.slots[i].head);
  }

  /// 插入后增量更新受影响 hint 段
  fn rebuild_hint(&mut self, slot_id: u32) {
    ldb_simd::update_hint(
      &mut self.hdr.hints,
      self.hdr.count as usize,
      slot_id as usize,
      |i| self.slots[i].head,
    );
  }

  /// 页内写字节（公有 API：越界时 panic 而不是 UB）
  pub fn write_bytes(&mut self, offset: u32, data: &[u8]) {
    assert!(
      (offset as usize) <= PAGE_SIZE && data.len() <= PAGE_SIZE - offset as usize,
      "页内写越界: offset={offset} len={}",
      data.len()
    );
    self.write_bytes_unchecked(offset, data);
  }

  /// 页内写字节（内部契约：offset+len 已在页内；调用方由 free-space/槽位校验保证）
  #[inline]
  fn write_bytes_unchecked(&mut self, offset: u32, data: &[u8]) {
    let base = from_mut(self).cast::<u8>();
    unsafe {
      copy_nonoverlapping(data.as_ptr(), base.add(offset as usize), data.len());
    }
  }
}

/// 两段拼接 key 的 4 字节 head（不足补 0）
#[inline]
fn head_joined(a: &[u8], b: &[u8]) -> [u8; 4] {
  let mut head = [0u8; 4];
  for (n, &byte) in a.iter().chain(b).take(4).enumerate() {
    head[n] = byte;
  }
  head
}
