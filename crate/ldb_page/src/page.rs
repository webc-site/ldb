//! 4 KiB 页头与元数据页（上游 `ZLeanStore/src/include/storage/page.h`）。
//!
//! `PageHeader` 的字段顺序/宽度与上游 `struct PageHeader` 逐字节一致（104 B，
//! 经 clang++ 对真实头文件 `sizeof`/`offsetof` 对拍；断言见本文件与
//! `tests/page_test.rs`）。
//!
//! 跨 worktree 对照：`leanstore.blob/src/include/storage/page.h` 只有
//! `dirty`/`idx_id`/`p_gsn`（无写历史、无 GC 标记），`leanstore.latency/.../page.h`
//! 只有 `p_gsn`；二者都早于 EDT/写历史语义。本 crate 以 `ZLeanStore` 版本为准
//! （与 `todo.md:230-231` 的 `p_gsn`/`index_id`/`dirty`/`gc_read`/EDT 承诺一致）。

use std::{
  fmt,
  mem::{align_of, offset_of, size_of},
};

use ldb_type::{IndexId, LogId, PAGE_SIZE, PageId, WRITE_HISTORY_LENGTH};
use static_assertions::const_assert_eq;

/// 页头（GSN、索引 ID、写历史、脏/GC 标记）。
///
/// 字节布局（与上游 clang++ 输出一致）：
/// `0 idx_id(u16) | 2 对齐填充 6 B | 8 p_gsn(u64) | 16 prev_gcwrite(u64) |
///  24 wcnt(u32) | 28 对齐填充 4 B | 32 w_p_gsn([u64; 8]) | 96 dirty(bool) |
///  97 gc_read(bool) | 98..104 尾部填充 6 B`。
///
/// 三段填充全部显式建模：页头属于落盘格式，全字节覆盖可保证按字节序列化
/// （落盘 / WAL 镜像 / 哈希）时不读取未初始化内存（上游为隐式 padding）。
/// 三段填充均为**保留位**（构造后恒为 0，不承载业务状态）；上游若在这些
/// 区间新增字段，必须删除对应保留字段并按新布局迁移，编译期 `offset_of!`/
/// `size_of` 断言会拦截未同步的改动。
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PageHeader {
  /// 所属索引
  pub idx_id: IndexId,
  /// `idx_id` 后的对齐填充（上游隐式 6 B；2..8）。
  ///
  /// **保留位**：恒为 0。上游未来若在 `idx_id` 与 `p_gsn` 之间新增字段，
  /// 必须用真实字段替换本字段，并同步更新全部 offset 断言。
  _pad_head: [u8; 6],
  /// 页 GSN
  pub p_gsn: LogId,
  /// 上次 GC 写 GSN
  pub prev_gcwrite: LogId,
  /// 写历史有效计数
  pub wcnt: u32,
  /// `wcnt` 后的对齐填充（上游隐式 4 B；28..32，保留位恒 0）
  _pad_align: u32,
  /// 环形写历史（无显式下标，插入位置 = 最大元素后一格）
  pub w_p_gsn: [LogId; WRITE_HISTORY_LENGTH],
  /// 脏标记
  pub dirty: bool,
  /// GC 读路径标记
  pub gc_read: bool,
  /// 尾部对齐填充（上游隐式 6 B；98..104，保留位恒 0）
  _pad_tail: [u8; 6],
}

impl PageHeader {
  /// 全零构造
  pub const fn new() -> Self {
    Self {
      idx_id: 0,
      _pad_head: [0; 6],
      p_gsn: 0,
      prev_gcwrite: 0,
      wcnt: 0,
      _pad_align: 0,
      w_p_gsn: [0; WRITE_HISTORY_LENGTH],
      dirty: false,
      gc_read: false,
      _pad_tail: [0; 6],
    }
  }

  /// 写历史清零（上游 `WHistoryInit`：不清 `wcnt`）
  #[inline]
  pub fn w_history_init(&mut self) {
    self.w_p_gsn = [0; WRITE_HISTORY_LENGTH];
  }

  /// 追加一个写 GSN（环形：覆盖最大元素后一格，上游 `AddWriteGSN`）
  #[inline]
  pub fn add_write_gsn(&mut self, wp_gsn: LogId) {
    self.wcnt = self.wcnt.wrapping_add(1);
    let mut max_idx = 0usize;
    for i in 1..WRITE_HISTORY_LENGTH {
      if self.w_p_gsn[i] > self.w_p_gsn[max_idx] {
        max_idx = i;
      }
    }
    let insert_pos = (max_idx + 1) % WRITE_HISTORY_LENGTH;
    self.w_p_gsn[insert_pos] = wp_gsn;
  }

  /// 移除最近一次写 GSN（最右非零项；上游 `RemoveLastWriteGSN` 不递减 `wcnt`）
  #[inline]
  pub fn remove_last_write_gsn(&mut self) {
    for i in (0..WRITE_HISTORY_LENGTH).rev() {
      if self.w_p_gsn[i] != 0 {
        self.w_p_gsn[i] = 0;
        return;
      }
    }
  }

  /// 记录 GC 写 GSN
  #[inline]
  pub fn update_last_gc_write(&mut self, gsn: LogId) {
    self.prev_gcwrite = gsn;
  }
}

impl Default for PageHeader {
  fn default() -> Self {
    Self::new()
  }
}

/// 4096 对齐的原始页
#[repr(C, align(4096))]
pub struct Page {
  /// 页头（B+Tree 节点头部以此为前缀）
  pub hdr: PageHeader,
  _rest: [u8; PAGE_BODY],
}

/// 0 号元数据页：Header + 根指针表
#[repr(C, align(4096))]
pub struct MetadataPage {
  /// 页头
  pub hdr: PageHeader,
  /// 各索引的根页指针
  pub roots: [PageId; ROOT_SLOTS],
}

/// 页体（扣除页头）
pub const PAGE_BODY: usize = PAGE_SIZE - size_of::<PageHeader>();
/// 根指针槽位数（上游 `(PAGE_SIZE - sizeof(PageHeader)) / sizeof(pageid_t)`）
pub const ROOT_SLOTS: usize = PAGE_BODY / size_of::<PageId>();

// 落盘格式固化：以下期望值全部来自 clang++ 对真实上游头文件
// `ZLeanStore/src/include/storage/page.h` 的 `sizeof`/`offsetof` 输出。
// 字段增删/换序若导致漂移，在编译期拦截（BTreeNodeHeader 布局以此为基）。
const_assert_eq!(PAGE_SIZE, 4096);
const_assert_eq!(size_of::<PageHeader>(), 104);
const_assert_eq!(offset_of!(PageHeader, idx_id), 0);
const_assert_eq!(offset_of!(PageHeader, _pad_head), 2);
const_assert_eq!(offset_of!(PageHeader, p_gsn), 8);
const_assert_eq!(offset_of!(PageHeader, prev_gcwrite), 16);
const_assert_eq!(offset_of!(PageHeader, wcnt), 24);
const_assert_eq!(offset_of!(PageHeader, _pad_align), 28);
const_assert_eq!(offset_of!(PageHeader, w_p_gsn), 32);
const_assert_eq!(offset_of!(PageHeader, dirty), 96);
const_assert_eq!(offset_of!(PageHeader, gc_read), 97);
const_assert_eq!(offset_of!(PageHeader, _pad_tail), 98);
const_assert_eq!(PAGE_BODY, 3992);
const_assert_eq!(ROOT_SLOTS, 499);
const_assert_eq!(size_of::<Page>(), PAGE_SIZE);
const_assert_eq!(size_of::<MetadataPage>(), PAGE_SIZE);
const_assert_eq!(align_of::<Page>(), PAGE_SIZE);
const_assert_eq!(align_of::<MetadataPage>(), PAGE_SIZE);
const_assert_eq!(offset_of!(MetadataPage, roots), 104);
// 无尾部空洞：页体/根表恰好铺满 4096 B
const_assert_eq!(offset_of!(Page, _rest) + PAGE_BODY, PAGE_SIZE);
const_assert_eq!(
  offset_of!(MetadataPage, roots) + ROOT_SLOTS * size_of::<PageId>(),
  PAGE_SIZE
);

impl Page {
  /// 全零构造（页头 + 页体），用于新建页/测试落盘字节确定性
  pub const fn new() -> Self {
    Self {
      hdr: PageHeader::new(),
      _rest: [0; PAGE_BODY],
    }
  }
}

impl Default for Page {
  fn default() -> Self {
    Self::new()
  }
}

impl MetadataPage {
  /// 全零构造
  pub fn new() -> Self {
    Self {
      hdr: PageHeader::new(),
      roots: [0; ROOT_SLOTS],
    }
  }

  /// 读第 `slot` 个索引的根页；越界返回 `None`（上游 `GetRoot` 直接索引是 UB）。
  ///
  /// 树对象持有的 slot 由建树时分配（上游 `btree_slot_counter++`，无上界检查），
  /// 正常路径恒 `< ROOT_SLOTS`；`Option` 仅为损坏页头/越界 slot 兜底，合法路径
  /// 代价是一次边界比较（可预测分支，见 bench `get_root valid slot`）。
  #[inline]
  pub fn get_root(&self, slot: usize) -> Option<PageId> {
    self.roots.get(slot).copied()
  }
}

impl Default for MetadataPage {
  fn default() -> Self {
    Self::new()
  }
}

impl fmt::Debug for Page {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Page")
      .field("hdr", &self.hdr)
      .finish_non_exhaustive()
  }
}

impl fmt::Debug for MetadataPage {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("MetadataPage")
      .field("hdr", &self.hdr)
      .finish_non_exhaustive()
  }
}
