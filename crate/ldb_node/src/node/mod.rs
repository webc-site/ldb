//! B+Tree 节点（上游 `ZLeanStore/src/storage/btree/node.{h,cc}`）。
//!
//! 布局：`[BTreeNodeHeader][PageSlot 槽位数组 ... | 数据自页尾向前]`，
//! 整节点 4096 字节对齐。前缀压缩（fence 公共前缀不重复存储）+
//! 16-Hint 数组加速二分 + 4B head 粗比较。
//!
//! # 线上布局
//!
//! clang++ 对真实上游头文件（`ZLeanStore/src/include/storage/btree/node.h` +
//! `storage/page.h`）的 `sizeof`/`offsetof` 输出（见 crate 审查报告）：
//!
//! ```text
//! sizeof(PageHeader)=104 sizeof(FenceKeySlot)=4 sizeof(PageSlot)=10
//! sizeof(BTreeNodeHeader)=208 sizeof(BTreeNode)=4096
//! idx_id=0 p_gsn=8 prev_gcwrite=16 wcnt=24 w_p_gsn=32 dirty=96 gc_read=97
//! right_most_child=104 prev_leaf_node=112 lower_fence=120 upper_fence=124
//! is_leaf=128 count=130 space_used=132 data_offset=134 prefix_len=136 hints=138
//! ```
//!
//! # 落盘兼容性（208 B 不是约束）
//!
//! 本格式是 **ldb 自有落盘格式，不与 C++ 上游互通**：页编解码、WAL、恢复
//! 全部由 Rust 侧实现；`grep -rn BTreeNodeHeader crate/` 在其它 crate 0 命中，
//! 没有任何读取 C++ 页镜像的代码。因此上游
//! `sizeof(BTreeNodeHeader)=208` 只是对照基准，不是兼容约束；本格式头部
//! **232 B** 是「上游 208 B + 24 B 显式设计差」，逐项对账如下。
//!
//! | 差项 | 字节 | 依据 |
//! | :--- | ---: | :--- |
//! | `lower_fence`/`upper_fence` 的 `offset/len` `u16`→`u32` | +8 | 自有格式统一用 u32 长度（`PageSlot` 同） |
//! | `count`/`space_used`/`data_offset`/`prefix_len` `u16`→`u32` | +8 | 同上；避免 64 KiB 以上页/长 fence 的二轮迁移 |
//! | `is_leaf` 后显式 `_pad_leaf[3]`（上游为隐式 1 B） | +2 | 使后续 u32 字段 4 B 对齐，且**无隐式 padding** |
//! | 去掉上游 `hints` 后的隐式尾部 padding 6 B | −6 | 上游靠 8 B 对齐留白；本格式尾部由 `contention` 占满 |
//! | 尾部 `contention: ContentionTracker` | +12 | `todo.md` §3.3 明确要求；CIDR'21 争用分裂状态 |
//! | **合计** | **+24** | 208 + 24 = **232** |
//!
//! 结论：24 B 中 12 B 是项目要求新增的字段（不是「多加了不该存的」），
//! 另 12 B 来自 u32 加宽与显式对齐（+18 B）并抵扣上游尾部隐式 padding
//! 6 B；没有上游有而本格式缺失的字段。上游两个隐式 padding 位置
//! （`is_leaf` 后 1 B、`hints` 后 6 B）在本格式中由 `_pad_leaf` 与
//! `contention` 覆盖，`BTreeNodeHeader` 232 B 内 **不存在编译器 padding**
//! （`RUSTFLAGS=-Zprint-type-sizes` 实测无 padding 行）。
//!
//! # `contention` 字段的持久化/恢复语义
//!
//! `contention` 位于页镜像 220..232，因此会随页一起写入磁盘；但它只是
//! CIDR'21 争用分裂的**易失启发式状态**，不参与任何结构不变量：
//! - `validate_bytes` 不解释该字段；`last_modified_pos` 允许指向已不存在的槽位。
//! - `note_access` 的返回值必须像上游 `BTreeGeneric::trySplit`
//!   （`BTreeGeneric.cpp:41-60`：`favored_split_pos < 0 || >= count - 1` 回退
//!   `findSep`）那样由消费方做范围检查，不得直接当槽位使用。
//! - 页从磁盘载入可写帧（未来 `ldb_buf`）时必须调用
//!   `hdr.contention.reset()`：上游把该字段放在 `BufferFrame::Header`（不落盘，
//!   帧分配即默认值），本格式把它放进页镜像，恢复时若不重置，上一次运行的
//!   计数会污染本轮采样（最坏是一次多余的争用分裂尝试，消费方范围检查兜底）。
//!
//! # 字段级布局
//!
//! | 上游 offset/宽度 | 字段 | 本格式 offset/宽度 | 语义 |
//! | :--- | :--- | :--- | :--- |
//! | 0 / 104 | `PageHeader`（idx_id/p_gsn/wcnt/w_p_gsn/dirty/gc_read） | 0 / 104 | 逐字节相同（`ldb_page` 断言） |
//! | 104 / 8 | `right_most_child`（叶：next） | 104 / 8 | 相同 |
//! | 112 / 8 | `prev_leaf_node` | 112 / 8 | 相同 |
//! | 120 / 4 | `lower_fence`（u16+u16） | 120 / 8（u32+u32） | 加宽 |
//! | 124 / 4 | `upper_fence`（u16+u16） | 128 / 8（u32+u32） | 加宽 |
//! | 128 / 1 | `is_leaf` | 136 / 1 | 相同 |
//! | 129 / 1（隐式 pad） | — | 137 / 3 `_pad_leaf` | 显式零填充 |
//! | 130 / 2 | `count` | 140 / 4 | 加宽 |
//! | 132 / 2 | `space_used` | 144 / 4 | 加宽 |
//! | 134 / 2 | `data_offset` | 148 / 4 | 加宽 |
//! | 136 / 2 | `prefix_len` | 152 / 4 | 加宽 |
//! | 138 / 64 | `hints[16]` | 156 / 64 | 相同 |
//! | 202 / 6（隐式 pad） | — | 220 / 12 `contention` | 新增字段（s32/u32/u32） |
//! | **208** | | **232** | |
//!
//! 槽位数组不属于头部：上游 `PageSlot` 10 B（u16×3+head），本格式 16 B
//! （u32×3+head，`repr(C, packed)`），`slots` 起点 232。

use std::{
  cmp::Ordering,
  fmt,
  mem::{offset_of, size_of},
  ptr::{from_mut, from_ref, write_bytes},
  slice::from_raw_parts,
};

use ldb_page::PageHeader;
pub mod mutate;
pub mod search;
pub mod sep;
mod validate;

use ldb_type::{IndexId, PAGE_SIZE, PageId};

/// 槽位（packed 16B：offset + key_len + payload_len + 4B head）
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C, packed)]
pub struct PageSlot {
  /// 数据区偏移（自页头起）
  pub offset: u32,
  /// 去前缀后 key 长度
  pub key_length: u32,
  /// payload 长度
  pub payload_length: u32,
  /// key 前 4 字节（不足补 0）
  pub head: [u8; 4],
}

/// fence 键槽位
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FenceKeySlot {
  /// 数据区偏移
  pub offset: u32,
  /// 长度（0 = 无穷界）
  pub len: u32,
}

/// 争用追踪器（CIDR'21 Contention Split 所需，VMCache 世代上游完全缺失）。
///
/// 字段类型/初值与上游 `BufferFrame::Header::ContentionTracker`
/// （`leanstore/backend/leanstore/storage/buffer-manager/BufferFrame.hpp:32-42`）
/// 逐项一致：`last_modified_pos` 是 `s32` 且初值 **-1**（无修改哨兵），
/// `restarts_counter`/`access_counter` 是 `u32`。
///
/// 本字段随页镜像落盘，但**不具持久语义**：载入可写页时必须 [`Self::reset`]
/// （见文件头「`contention` 字段的持久化/恢复语义」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ContentionTracker {
  /// 最近修改槽位（-1 = 尚未记录）
  pub last_modified_pos: i32,
  /// 重启计数
  pub restarts_counter: u32,
  /// 访问计数
  pub access_counter: u32,
}

impl Default for ContentionTracker {
  #[inline]
  fn default() -> Self {
    Self {
      last_modified_pos: -1,
      restarts_counter: 0,
      access_counter: 0,
    }
  }
}

/// 头部（含 PageHeader 的 GSN/写历史，经 `ldb_page::PageHeader` 组合）。
///
/// 字段顺序与上游 `class BTreeNodeHeader : public PageHeader` 一致；所有显式
/// 偏移由本文件末尾的 `offset_of!` 断言钉死。
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BTreeNodeHeader {
  /// 页头（GSN / 写历史 / 脏标记）
  pub page: ldb_page::PageHeader, // 0..104
  /// 内节点：最右子；叶节点：右兄弟（union 语义）
  pub right_most_child: PageId, // 104..112
  /// 叶节点：左兄弟
  pub prev_leaf_node: PageId, // 112..120
  /// 下界 fence（排除）
  pub lower_fence: FenceKeySlot, // 120..128
  /// 上界 fence（包含）
  pub upper_fence: FenceKeySlot, // 128..136
  /// 是否叶节点
  pub is_leaf: bool, // 136
  /// `is_leaf` 后的显式对齐填充（上游隐式 1 B；保留位恒 0）
  _pad_leaf: [u8; 3], // 137..140
  /// 槽位数
  pub count: u32, // 140..144
  /// key+payload+fence 已用字节
  pub space_used: u32, // 144..148
  /// 数据区游标（自页头起，向低地址生长）
  pub data_offset: u32, // 148..152
  /// 前缀长度
  pub prefix_len: u32, // 152..156
  /// 16-Hint 数组
  pub hints: [[u8; 4]; ldb_type::HINT_COUNT], // 156..220
  /// 争用追踪（本工程新增字段）
  pub contention: ContentionTracker, // 220..232
}

/// 无右邻居哨兵
pub const EMPTY_NEIGHBOR: PageId = u64::MAX;
/// 低于此用量可合并
pub const SIZE_UNDER_FULL: u32 = (PAGE_SIZE as u32) * 3 / 4;

/// `MAX_RECORD_SIZE`/`NUM_SLOTS` 共用的头部上界：232 B 头 + 228 B 预留。
const HEADER_BOUND: usize = 460;

/// 单条记录（key 去前缀后）最大尺寸。
///
/// 与上游 `MAX_RECORD_SIZE = (PAGE_SIZE - sizeof(BTreeNodeHeader) - 3*sizeof(PageSlot)) / 3`
/// 同形；本格式头部 + 预留上界取 460 B（232 B 头 + 228 B 预留，保证槽位数组
/// 尾部留出固定余量）。上游 `blob::BlobState::MAX_MALLOC_SIZE = 1100` 小于本值。
pub const MAX_RECORD_SIZE: u32 =
  (PAGE_SIZE as u32 - HEADER_BOUND as u32 - 3 * size_of::<PageSlot>() as u32) / 3;

/// 槽位数组容量
pub const NUM_SLOTS: usize = (PAGE_SIZE - HEADER_BOUND) / size_of::<PageSlot>();

/// B+Tree 节点（严格 4096B）
#[repr(C, align(4096))]
pub struct BTreeNode {
  /// 头部
  pub hdr: BTreeNodeHeader,
  /// 槽位数组
  pub slots: [PageSlot; NUM_SLOTS],
}

/// 分裂上下文（收敛 split_node 8 参签名）
#[derive(Debug)]
pub struct SplitCtx<'a> {
  /// 父节点
  pub parent: &'a mut BTreeNode,
  /// 新右节点
  pub node_right: &'a mut BTreeNode,
  /// 左子页号
  pub left_pid: PageId,
  /// 右子页号
  pub right_pid: PageId,
}

/// 键比较器（MEMCMP 主路径零开销；Custom 走函数指针）
#[derive(Debug, Clone, Copy)]
pub struct Cmp {
  /// 是否 memcmp 字典序（决定前缀压缩可用性）
  pub memcmp: bool,
  /// 比较函数
  pub f: fn(&[u8], &[u8]) -> Ordering,
}

/// 默认 memcmp 比较器
pub const MEMCMP: Cmp = Cmp {
  memcmp: true,
  f: cmp_memcmp,
};

/// 字典序比较（等价 C memcmp 三态）
pub fn cmp_memcmp(a: &[u8], b: &[u8]) -> Ordering {
  a.cmp(b)
}

impl BTreeNodeHeader {
  /// 头部构造（上游 `BTreeNodeHeader::BTreeNodeHeader(bool, indexid_t)`：
  /// 只设置 `idx_id`/`is_leaf`，其余字段走默认值）
  pub fn new(is_leaf: bool, index: IndexId) -> Self {
    let mut page = PageHeader::new();
    page.idx_id = index;
    Self {
      page,
      right_most_child: EMPTY_NEIGHBOR,
      prev_leaf_node: EMPTY_NEIGHBOR,
      lower_fence: FenceKeySlot::default(),
      upper_fence: FenceKeySlot::default(),
      is_leaf,
      _pad_leaf: [0; 3],
      count: 0,
      space_used: 0,
      data_offset: PAGE_SIZE as u32,
      prefix_len: 0,
      hints: [[0; 4]; ldb_type::HINT_COUNT],
      contention: ContentionTracker::default(),
    }
  }

  /// 是否有左兄弟
  #[inline]
  pub fn has_left_neighbor(&self) -> bool {
    self.prev_leaf_node != EMPTY_NEIGHBOR
  }

  /// 是否有右兄弟
  #[inline]
  pub fn has_right_neighbor(&self) -> bool {
    self.right_most_child != EMPTY_NEIGHBOR
  }

  /// 下界是否无穷（len == 0）
  #[inline]
  pub fn is_lower_fence_infinity(&self) -> bool {
    self.lower_fence.len == 0
  }

  /// 上界是否无穷
  #[inline]
  pub fn is_upper_fence_infinity(&self) -> bool {
    self.upper_fence.len == 0
  }
}

impl BTreeNode {
  /// 节点转储（上游 `BTreeNode::ToString` 调试语义；命名避让 `ToString` 惯例
  /// —— 固有 `to_string` 会遮蔽 `Display`，clippy inherent_to_string 禁用）
  pub fn to_debug_string(&self) -> String {
    let fence = |f: &FenceKeySlot| {
      if f.len == 0 {
        "∞".to_string()
      } else {
        format!("{}B", f.len)
      }
    };
    format!(
      "BTreeNode{{kind:{}, count:{}, prefix_len:{}, lower:{}, upper:{}}}",
      if self.is_inner() { "inner" } else { "leaf" },
      self.hdr.count,
      self.get_prefix().len(),
      fence(&self.hdr.lower_fence),
      fence(&self.hdr.upper_fence),
    )
  }

  /// 节点构造（脏页语义：上游 `BTreeNode::BTreeNode` 构造即置 `dirty = true`）
  pub fn new(is_leaf: bool, index: IndexId) -> Box<Self> {
    // 整页零初始化：free_space/tail padding 全字节确定（落盘哈希可复现）
    let mut node = unsafe { Box::<Self>::new_zeroed().assume_init() };
    node.hdr = BTreeNodeHeader::new(is_leaf, index);
    node.hdr.page.dirty = true;
    node
  }

  /// 节点起始字节
  #[inline]
  pub fn ptr(&self) -> *const u8 {
    from_ref(self).cast::<u8>()
  }

  /// 是否内节点
  #[inline]
  pub fn is_inner(&self) -> bool {
    !self.hdr.is_leaf
  }

  /// 前缀视图
  #[inline]
  pub fn get_prefix(&self) -> &[u8] {
    self.slice(self.hdr.lower_fence.offset, self.hdr.prefix_len)
  }

  /// 槽位 key（去前缀）
  #[inline]
  pub fn get_key(&self, slot: u32) -> &[u8] {
    self.slice(
      self.slots[slot as usize].offset,
      self.slots[slot as usize].key_length,
    )
  }

  /// 槽位 payload
  #[inline]
  pub fn get_payload(&self, slot: u32) -> &[u8] {
    let s = &self.slots[slot as usize];
    self.slice(s.offset + s.key_length, s.payload_length)
  }

  /// 内节点子指针
  #[inline]
  pub fn get_child(&self, slot: u32) -> PageId {
    let p = self.get_payload(slot);
    let mut b = [0u8; 8];
    b.copy_from_slice(&p[..8]);
    u64::from_le_bytes(b)
  }

  /// 下界 fence
  #[inline]
  pub fn get_lower_fence(&self) -> &[u8] {
    self.slice(self.hdr.lower_fence.offset, self.hdr.lower_fence.len)
  }

  /// 上界 fence
  #[inline]
  pub fn get_upper_fence(&self) -> &[u8] {
    self.slice(self.hdr.upper_fence.offset, self.hdr.upper_fence.len)
  }

  /// 剩余空间（自槽位数组尾至数据游标；纯读 —— 插入热路径判定用，
  /// 不做 memset，见 `zero_free_space`）
  #[inline]
  pub fn free_space(&self) -> u32 {
    (self.hdr.data_offset as usize - self.slots_end()) as u32
  }

  /// 清零空闲区（上游 `FreeSpace()` 的 memset 语义：利于页压缩）。
  ///
  /// 仅在冷路径调用：碎片压缩 / 整页内容拷贝后；上游把 memset 混在
  /// `FreeSpace()` 里导致 `InsertKeyValue` 每次插入都白清一次空闲区
  /// （最多 ~3.5KiB），本版拆分后插入路径零 memset
  pub fn zero_free_space(&mut self) -> u32 {
    let free = self.free_space() as usize;
    let base = from_mut(self).cast::<u8>();
    // SAFETY: 空闲区 [slots_end, data_offset) 恒在 4096B 页界内
    unsafe {
      write_bytes(base.add(self.slots_end()), 0, free);
    }
    free as u32
  }

  /// 压缩后剩余空间
  #[inline]
  pub fn free_space_after_compaction(&self) -> u32 {
    (PAGE_SIZE - self.slots_end() - self.hdr.space_used as usize) as u32
  }

  /// 一条 KV 所需空间（槽位 + 去前缀 key + payload）。
  ///
  /// `saturating_sub`：key 短于前缀的调用属契约外输入（路由保证 key 含前缀），
  /// 上游此处是无符号下溢（回绕成巨值 → 恒触发压缩断言崩溃），本版饱和不 panic
  #[inline]
  pub fn space_required_for_kv(&self, key_len: u32, payload_len: u32) -> u32 {
    // 饱和加法：公有谓词收到 u32::MAX 时返回 MAX（判定放不下）而不是 debug panic
    (size_of::<PageSlot>() as u32)
      .saturating_add(key_len.saturating_sub(self.hdr.prefix_len))
      .saturating_add(payload_len)
  }

  /// 是否放得下
  ///
  /// 字节闸门已隐含槽位闸门：228 条互异 key 的最小占用（228×16B 槽位 +
  /// ≥228B 去前缀 key）超过 4096 − 232，故 227 槽后 `free_space` 必然
  /// 拒绝下一条记录（见 `slotted_page_test::full_page_stops_at_slot_capacity`）。
  #[inline]
  pub fn has_space_for_kv(&self, key_len: u32, payload_len: u32) -> bool {
    self.space_required_for_kv(key_len, payload_len) <= self.free_space_after_compaction()
  }

  /// 键调试串
  pub fn key_to_string(&self, slot: u32) -> String {
    let k = self.get_key(slot);
    let mut out = String::with_capacity(k.len() * 2);
    for b in k {
      use std::fmt::Write as _;
      let _ = write!(out, "{b:02x}");
    }
    out
  }

  /// 槽位数组尾偏移
  #[inline]
  fn slots_end(&self) -> usize {
    size_of::<BTreeNodeHeader>() + self.hdr.count as usize * size_of::<PageSlot>()
  }

  /// 页内切片
  #[inline]
  fn slice(&self, offset: u32, len: u32) -> &[u8] {
    // SAFETY: 节点内偏移切片，调用方保证 offset+len <= PAGE_SIZE
    unsafe { from_raw_parts(self.ptr().add(offset as usize), len as usize) }
  }
}

impl fmt::Debug for BTreeNode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BTreeNode")
      .field("is_leaf", &self.hdr.is_leaf)
      .field("count", &self.hdr.count)
      .field("space_used", &self.hdr.space_used)
      .field("prefix_len", &self.hdr.prefix_len)
      .finish_non_exhaustive()
  }
}

/// 节点转字符串（账本名 to_string：Debug 序列化；走 Display 获得）
impl fmt::Display for BTreeNode {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt::Debug::fmt(self, f)
  }
}

// ---- 线上布局编译期断言 ----
//
// 期望值分两类：
// - 与上游 C++ 相同的字段（page / right_most_child / prev_leaf_node /
//   is_leaf / count / space_used / data_offset / prefix_len / hints 的相对顺序）；
// - 本格式加宽/新增字段后的绝对偏移（见文件头「线上布局」）。
// 任何字段增删/换序/宽度漂移都会在此拦截。

static_assertions::const_assert_eq!(size_of::<BTreeNodeHeader>(), 232);
static_assertions::const_assert_eq!(size_of::<BTreeNode>(), PAGE_SIZE);
static_assertions::const_assert_eq!(size_of::<PageSlot>(), 16);
static_assertions::const_assert_eq!(size_of::<FenceKeySlot>(), 8);
static_assertions::const_assert_eq!(size_of::<ContentionTracker>(), 12);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, page), 0);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, right_most_child), 104);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, prev_leaf_node), 112);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, lower_fence), 120);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, upper_fence), 128);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, is_leaf), 136);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, _pad_leaf), 137);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, count), 140);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, space_used), 144);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, data_offset), 148);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, prefix_len), 152);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, hints), 156);
static_assertions::const_assert_eq!(offset_of!(BTreeNodeHeader, contention), 220);
// 头部各段无缝衔接：hints 64 B 后即 contention，contention 12 B 后恰好 232 B。
// 与上游 208 B 的 24 B 差已在文件头逐项对账；任何字段漂移在此拦截。
static_assertions::const_assert_eq!(
  offset_of!(BTreeNodeHeader, hints) + 4 * ldb_type::HINT_COUNT,
  offset_of!(BTreeNodeHeader, contention)
);
static_assertions::const_assert_eq!(
  offset_of!(BTreeNodeHeader, contention) + size_of::<ContentionTracker>(),
  232
);
static_assertions::const_assert_eq!(offset_of!(BTreeNode, slots), 232);
static_assertions::const_assert_eq!(NUM_SLOTS, 227);
static_assertions::const_assert!(
  size_of::<BTreeNodeHeader>() + NUM_SLOTS * size_of::<PageSlot>() <= PAGE_SIZE
);
static_assertions::const_assert!(MAX_RECORD_SIZE >= 1100);
