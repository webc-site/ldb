//! 原始页校验：把不可信页字节 cast 成 [`BTreeNode`] 之前的结构性检查。
//!
//! 上游 `BTreeNode::FromPage` 一类的路径直接把页指针 `reinterpret_cast` 成
//! `BTreeNode*`；磁盘上任意一个越界槽位（`offset/key_len/payload_len`）或
//! 非 0/1 的 `bool` 字节都会让后续读取产生 UB。本模块把校验集中到
//! [`BTreeNode::validate_bytes`]：**只读原始字节、绝不构造节点引用**，任何
//! 畸形输入返回 [`Error::Corrupt`] 而不是 panic/越界读取。
//!
//! 校验项（与上游结构不变量逐条对应）：
//! - `is_leaf`/`dirty`/`gc_read` 字节必须 ∈ {0, 1}（Rust `bool` 的合法位模式）；
//! - `count <= NUM_SLOTS`，`slots_end <= data_offset <= PAGE_SIZE`；
//! - `space_used` 必须**恰好等于**所有槽位 `key_length + payload_length` 与两个
//!   fence 长度之和（上游 `BTreeNode.cpp:222-226` 的 debug 断言）。只查上界
//!   不够：压缩按 `space_used` 估算剩余空间，低估会让 `compactify` /
//!   `store_record_data` 在 release 下回绕 `data_offset` 并越界写；
//! - fence 的 `offset/len` 在页内，非空 fence / 记录均位于 `data_offset` 之后；
//!   两个非空 fence 必须严格递增（`lower < upper`，上游 `SetFences` 的区间语义）；
//! - `prefix_len` 不超过两个 fence 长度，且是两 fence 的公共前缀；非 MEMCMP
//!   比较器前缀必须为 0（上游 `SetFences` 只对 MEMCMP 计算前缀）；
//! - 每个槽位的 `head` 等于其去前缀 key 的前 4 字节；
//! - 槽位按去前缀 key 严格递增（同一前缀下等价于全键递增），无重复键；
//! - 记录与 fence 的字节区间两两不重叠。
//!
//! 本校验假设页字节是**本机字节序**的内存镜像（与 `BTreeNode::write_bytes`
//! 的写入方式一致）；跨大端落盘不在本 crate 支持范围（见审查报告残留风险）。

use std::{
  cmp::Ordering,
  mem::{offset_of, size_of},
  ptr::{from_mut, from_ref},
};

use ldb_page::Page;
use ldb_simd::get_head;
use ldb_type::{PAGE_SIZE, PageId};

use super::{BTreeNode, BTreeNodeHeader, Cmp, NUM_SLOTS, PageSlot};
use crate::Error;

const HDR: usize = size_of::<BTreeNodeHeader>();
const SLOT: usize = size_of::<PageSlot>();
/// fence 槽位在头部内的偏移（与 `offset_of!` 断言一致）
const LOWER_FENCE: usize = offset_of!(BTreeNodeHeader, lower_fence);
const UPPER_FENCE: usize = offset_of!(BTreeNodeHeader, upper_fence);
const COUNT_OFF: usize = offset_of!(BTreeNodeHeader, count);
const SPACE_USED_OFF: usize = offset_of!(BTreeNodeHeader, space_used);
const DATA_OFFSET_OFF: usize = offset_of!(BTreeNodeHeader, data_offset);
const PREFIX_OFF: usize = offset_of!(BTreeNodeHeader, prefix_len);
const PAD_LEAF_OFF: usize = offset_of!(BTreeNodeHeader, _pad_leaf);
/// `PageHeader` 布尔字段偏移（由 `ldb_page` 的布局断言钉死）
const DIRTY_OFF: usize = offset_of!(ldb_page::PageHeader, dirty);
const GC_READ_OFF: usize = offset_of!(ldb_page::PageHeader, gc_read);

/// 读本机字节序 `u32`（调用方保证 `off + 4 <= PAGE_SIZE`）。
#[inline]
fn read_u32(bytes: &[u8; PAGE_SIZE], off: usize) -> u32 {
  u32::from_ne_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

/// 页内区间；`start < end` 才有字节实体。
#[inline]
fn get(bytes: &[u8; PAGE_SIZE], offset: usize, len: usize) -> Option<&[u8]> {
  let end = offset.checked_add(len)?;
  (end <= PAGE_SIZE).then(|| &bytes[offset..end])
}

#[inline]
fn corrupt(msg: String) -> Error {
  Error::Corrupt(msg)
}

impl BTreeNode {
  /// 从只读页构造节点引用（先校验后 cast）。
  ///
  /// 失败返回 [`Error::Corrupt`]，绝不产生越界/非法 `bool` 引用。
  pub fn from_page(page: &Page, cmp: Cmp) -> Result<&Self, Error> {
    // SAFETY: `Page` 是 4096 B、4096 对齐的整页；只读借用期间无别名可变引用。
    let bytes: &[u8; PAGE_SIZE] = unsafe { &*from_ref(page).cast() };
    Self::validate_bytes(bytes, cmp)?;
    // SAFETY: 字节已按 BTreeNode 布局校验（bool 规范化、槽位全部在界内）。
    Ok(unsafe { &*from_ref(page).cast::<Self>() })
  }

  /// 从可写页构造节点引用（先校验后 cast）。
  pub fn from_page_mut(page: &mut Page, cmp: Cmp) -> Result<&mut Self, Error> {
    // SAFETY: 只读借用仅用于校验，校验结束即释放。
    let bytes: &[u8; PAGE_SIZE] = unsafe { &*from_ref(page).cast() };
    Self::validate_bytes(bytes, cmp)?;
    // SAFETY: 同上；`&mut Page` 保证独占。
    Ok(unsafe { &mut *from_mut(page).cast::<Self>() })
  }

  /// 校验节点自身（内存中由本 crate 构造的节点，或已通过
  /// [`BTreeNode::from_page_mut`] 校验的节点）。
  pub fn validate(&self, cmp: Cmp) -> Result<(), Error> {
    // SAFETY: `BTreeNode` 恰为 4096 B；作为有效 Rust 值，全部字节已初始化。
    let bytes: &[u8; PAGE_SIZE] = unsafe { &*self.ptr().cast() };
    Self::validate_bytes(bytes, cmp)
  }

  /// 校验任意原始页字节（安全入口：不构造节点引用）。
  ///
  /// `cmp` 只用于槽位有序性比较；`memcmp == false` 时 `prefix_len` 必须为 0。
  pub fn validate_bytes(bytes: &[u8; PAGE_SIZE], cmp: Cmp) -> Result<(), Error> {
    // ---- bool 字节规范化（在 cast 前拦截 invalid bool） ----
    let is_leaf_off = offset_of!(BTreeNodeHeader, is_leaf);
    if bytes[is_leaf_off] > 1 {
      return Err(corrupt(format!(
        "is_leaf 字节 {} 非 0/1",
        bytes[is_leaf_off]
      )));
    }
    let is_leaf = bytes[is_leaf_off] != 0;
    // PageHeader 的 dirty/gc_read 偏移由 `offset_of!` 绑定（ldb_page 布局断言）
    for (name, off) in [("dirty", DIRTY_OFF), ("gc_read", GC_READ_OFF)] {
      if bytes[off] > 1 {
        return Err(corrupt(format!("{name} 字节 {} 非 0/1", bytes[off])));
      }
    }
    // 显式对齐填充必须为 0（保留位）
    for (i, &b) in bytes[PAD_LEAF_OFF..PAD_LEAF_OFF + 3].iter().enumerate() {
      if b != 0 {
        return Err(corrupt(format!("保留字节 {PAD_LEAF_OFF}+{i} 为 {b}")));
      }
    }

    let count = read_u32(bytes, COUNT_OFF) as usize;
    if count > NUM_SLOTS {
      return Err(corrupt(format!("count {count} 超过槽位容量 {NUM_SLOTS}")));
    }
    let slots_end = HDR + count * SLOT;
    let data_offset = read_u32(bytes, DATA_OFFSET_OFF) as usize;
    if data_offset > PAGE_SIZE || data_offset < slots_end {
      return Err(corrupt(format!(
        "data_offset {data_offset} 越界 [{slots_end}, {PAGE_SIZE}]"
      )));
    }
    let space_used = read_u32(bytes, SPACE_USED_OFF) as usize;
    if space_used > PAGE_SIZE - data_offset {
      return Err(corrupt(format!(
        "space_used {space_used} 超过数据区 {}",
        PAGE_SIZE - data_offset
      )));
    }

    // ---- fence 边界 ----
    let mut fence_li = [(0usize, 0usize); 2];
    for (i, base) in [LOWER_FENCE, UPPER_FENCE].into_iter().enumerate() {
      let off = read_u32(bytes, base) as usize;
      let len = read_u32(bytes, base + 4) as usize;
      // len == 0：无穷界；允许构造器默认值 offset == 0 或已分配游标
      if len == 0 {
        if off != 0 && (off < slots_end || off > PAGE_SIZE) {
          return Err(corrupt(format!("空 fence{i} offset {off} 越界")));
        }
      } else {
        if off < data_offset || get(bytes, off, len).is_none() {
          return Err(corrupt(format!("fence{i} [{off}, +{len}) 越界")));
        }
        fence_li[i] = (off, off + len);
      }
    }

    // ---- prefix ----
    let prefix_len = read_u32(bytes, PREFIX_OFF) as usize;
    let lower_len = fence_li[0].1 - fence_li[0].0;
    let upper_len = fence_li[1].1 - fence_li[1].0;
    if prefix_len > lower_len || prefix_len > upper_len {
      return Err(corrupt(format!(
        "prefix_len {prefix_len} 超过 fence 长度 ({lower_len}, {upper_len})"
      )));
    }
    if !cmp.memcmp && prefix_len != 0 {
      return Err(corrupt(format!("非 MEMCMP 比较器 prefix_len={prefix_len}")));
    }
    if prefix_len > 0
      && bytes[fence_li[0].0..fence_li[0].0 + prefix_len]
        != bytes[fence_li[1].0..fence_li[1].0 + prefix_len]
    {
      return Err(corrupt("prefix 不是两个 fence 的公共前缀".to_string()));
    }

    // ---- fence 区间非空：两个非空 fence 必须严格递增 ----
    // 单看每个键的 lower < key <= upper 已隐含该条件，但空节点（count == 0）
    // 没有键可查，lower >= upper 的页会让路由区间为空却通过校验。
    if fence_li[0].1 > fence_li[0].0 && fence_li[1].1 > fence_li[1].0 {
      let lower = &bytes[fence_li[0].0..fence_li[0].1];
      let upper = &bytes[fence_li[1].0..fence_li[1].1];
      if (cmp.f)(lower, upper) != Ordering::Less {
        return Err(corrupt(format!(
          "lower fence [{}, {}) 不严格小于 upper fence [{}, {})",
          fence_li[0].0, fence_li[0].1, fence_li[1].0, fence_li[1].1
        )));
      }
    }

    // ---- 槽位 ----
    // `space_used` 的精确账：fence 字节 + 全部存活记录字节。
    // 上游 `BTreeNode.cpp:222-226` 在 debug block 中逐条累加并断言相等；
    // 这里作为安全不变量（release 也查），因为 `free_space_after_compaction`
    // 用它决定能否压缩后放下新记录。
    let mut live_bytes = (lower_len + upper_len) as u64;
    let mut intervals = [(0u32, 0u32); NUM_SLOTS + 2];
    let mut nint = 0usize;
    for (off, end) in fence_li {
      if end > off {
        intervals[nint] = (off as u32, end as u32);
        nint += 1;
      }
    }

    let mut prev_key: Option<&[u8]> = None;
    for i in 0..count {
      let base = HDR + i * SLOT;
      let off = read_u32(bytes, base) as usize;
      let klen = read_u32(bytes, base + 4) as usize;
      let plen = read_u32(bytes, base + 8) as usize;
      let end = off
        .checked_add(klen)
        .and_then(|v| v.checked_add(plen))
        .ok_or_else(|| corrupt(format!("槽位 {i} 长度溢出")))?;
      if off < data_offset || end > PAGE_SIZE {
        return Err(corrupt(format!(
          "槽位 {i} [{off}, +{klen}+{plen}) 越界 (data_offset {data_offset})"
        )));
      }
      let key = &bytes[off..off + klen];
      if bytes[base + 12..base + 16] != get_head(key) {
        return Err(corrupt(format!("槽位 {i} head 与 key 不符")));
      }
      // 内节点 payload 是 8 B 子页号：validate 通过后 `get_child` 会读 p[..8]，
      // 短 payload 会把「Err」变成后续 panic；多出的字节也不是上游形态
      //（`GetChild` 只解释 `sizeof(pageid_t)`），一并拒绝。叶节点 payload 任意。
      if !is_leaf && plen != size_of::<PageId>() {
        return Err(corrupt(format!(
          "内节点槽位 {i} payload {plen} != {size_of_pageid}",
          size_of_pageid = size_of::<PageId>()
        )));
      }
      // fence 边界：lower 排他、upper 包含（上游 compareKeyWithBoundaries）
      if fence_li[0].1 > fence_li[0].0 || fence_li[1].1 > fence_li[1].0 {
        let mut full = Vec::with_capacity(prefix_len + klen);
        full.extend_from_slice(&bytes[fence_li[0].0..fence_li[0].0 + prefix_len]);
        full.extend_from_slice(key);
        if fence_li[0].1 > fence_li[0].0
          && (cmp.f)(&full, &bytes[fence_li[0].0..fence_li[0].1]) != Ordering::Greater
        {
          return Err(corrupt(format!("槽位 {i} 未严格大于 lower fence")));
        }
        if fence_li[1].1 > fence_li[1].0
          && (cmp.f)(&full, &bytes[fence_li[1].0..fence_li[1].1]) == Ordering::Greater
        {
          return Err(corrupt(format!("槽位 {i} 超过 upper fence")));
        }
      }
      if prev_key.is_some_and(|prev| (cmp.f)(prev, key) != Ordering::Less) {
        return Err(corrupt(format!("槽位 {} 与 {i} 非严格递增", i - 1)));
      }
      prev_key = Some(key);
      live_bytes += (klen + plen) as u64;
      intervals[nint] = (off as u32, end as u32);
      nint += 1;
    }

    // ---- space_used 精确账（见上） ----
    if space_used as u64 != live_bytes {
      return Err(corrupt(format!(
        "space_used {space_used} != 槽位+fence 字节和 {live_bytes}"
      )));
    }

    // ---- 区间不重叠（含 fence） ----
    let live = &mut intervals[..nint];
    live.sort_unstable_by_key(|&(start, _)| start);
    for w in live.windows(2) {
      if w[0].1 > w[1].0 {
        return Err(corrupt(format!(
          "字节区间重叠 [{}, {}) 与 [{}, {})",
          w[0].0, w[0].1, w[1].0, w[1].1
        )));
      }
    }

    Ok(())
  }
}
