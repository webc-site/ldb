//! 4 KiB 对齐装箱纯函数（入参完全参数化版，上游 `io_interface.cc:1168/1223`）。
//!
//! 带状态编排（`BinPacking`/`BinPackGroups`/`UniformBinPacking`）已归位 L3
//! `ldb_space::page_io`（依赖 `wbuffer_` 成员状态）；本模块只保留
//! `BinPackingPerBlock` / `CopyCompressedPages` 两条纯函数路径。
//!
//! 不变量：任意压缩页不跨 4 KiB 物理扇区边界（todo.md:183）；每 4 KiB 最多
//! `PAGE_SIZE / MIN_COMP_SIZE` 页（避免单页碎片过多）。

use std::{cmp::Reverse, collections::BTreeMap};

use ldb_type::{LogId, MIN_COMP_SIZE, PAGE_SIZE};

/// 每 4 KiB 槽位可容纳的最大页数
pub const MAX_PAGES_PER_BIN: u32 = (PAGE_SIZE as u32) / (MIN_COMP_SIZE as u32);

/// 至少能装一页，否则装箱无解
const _: () = assert!(MIN_COMP_SIZE as usize <= PAGE_SIZE);

/// 装箱页描述（调用方持有的元数据视图）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZipEntry {
  /// 压缩后尺寸（`1..=PAGE_SIZE`；`UNCOMPRESSED` 表示原文直存）
  pub comp_sz: u16,
  /// 页死亡时间（GDT 分组用）
  pub death_time: LogId,
  /// 页内数据在源缓冲的偏移（u32 上限 4 GiB，单块远小于此）
  pub src_off: u32,
}

/// 装箱方案：页 → (bin 下标, bin 内字节偏移)
///
/// 全局物理偏移 = `bin * PAGE_SIZE + off`；`off` 是 4 KiB 槽位内偏移，
/// 不是整块写缓冲偏移。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinSlot {
  pub bin: u32,
  pub off: u32,
}

/// 每个物理 4 KiB 槽位的占用状态
///
/// `sum` 是本槽位内已用字节（**不是**整块写缓冲偏移），`count` 是槽内页数。
/// 前置契约（调用方保证）：`sum <= PAGE_SIZE` 且 `count <= MAX_PAGES_PER_BIN`；
/// 新块的 `bins` 必须从零基状态开始（不得混入其它块的全局偏移/累计值）。
/// 违反 `sum` 契约在 debug/release 下都会 panic（见 `bin_remaining`），
/// 不做静默回绕：回绕会让 `BinSlot.off` 越过 4 KiB 槽位、映射表指向缓冲外。
#[derive(Debug, Clone, Copy, Default)]
pub struct Bin {
  pub sum: u32,
  pub count: u32,
}

/// 单块贪心装箱（上游 `BinPackingPerBlock` 的纯函数化）：按 `comp_sz` 降序、
/// best-fit（最小可容纳剩余空间）放置。
///
/// 返回 `plan` 与**入参 `entries` 原顺序**一一对齐（`plan[i]` 属于 `entries[i]`）；
/// 入参不被排序或修改。上游按 `std::multimap` 的 `lower_bound` 选取同剩余空间的
/// 首个 bin（FIFO），本实现用 `(剩余空间, 插入序号)` 复合键复现同一顺序。
///
/// 放不下的页在结果中留 `None`：上游此时会新建 bin（`bincount++`），本函数不扩容，
/// 调用方须预置足够多的空 bin。
///
/// # Panics
/// `Bin.sum > PAGE_SIZE`（`bin_remaining`）或 `comp_sz > PAGE_SIZE`（逐页断言）时
/// panic，debug/release 一致。二者都是调用方违约：前者会让剩余空间回绕并把
/// `off` 写到槽位外，后者无法满足 4 KiB 不变量；静默回绕比 panic 更糟。
///
/// 上游同时写 `dest->cur_size = physical_pages_cnt * PAGE_SIZE` 并返回/写回
/// `avg_edt`（`io_interface.cc:1315-1318`）；本函数只返回布局，二者由调用方按
/// 实际使用的 bin 数与 `entries` 的 `death_time` 计算。
pub fn bin_packing_per_block(entries: &[ZipEntry], bins: &mut [Bin]) -> Vec<Option<BinSlot>> {
  // 降序（大件先装）。稳定排序：同尺寸页保持原顺序（上游 std::sort 不保证）。
  let mut order: Vec<usize> = (0..entries.len()).collect();
  order.sort_by_key(|&i| Reverse(entries[i].comp_sz));

  // 键 = 剩余空间 << 32 | 插入序号：按剩余空间取 lower_bound，同剩余空间按插入序 FIFO。
  // 一个 bin 只占一个键（不额外分配候选列表）；序号单调递增保证等价于 multimap 插入序。
  let mut seq: u32 = 0;
  let mut free: BTreeMap<u64, u32> = BTreeMap::new();
  for (i, b) in bins.iter().enumerate() {
    if b.count < MAX_PAGES_PER_BIN {
      free.insert(free_key(bin_remaining(*b), seq), i as u32);
      seq += 1;
    }
  }

  let mut out = vec![None; entries.len()];
  for i in order {
    let sz = u32::from(entries[i].comp_sz);
    assert!(sz <= PAGE_SIZE as u32, "comp_sz 超 PAGE_SIZE（调用方违约）");
    // best-fit：最小可容纳剩余空间（复合键高位即剩余空间）
    let Some((&key, &b)) = free.range(free_key(sz, 0)..).next() else {
      continue;
    };
    free.remove(&key);
    let off = bins[b as usize].sum;
    bins[b as usize].sum += sz;
    bins[b as usize].count += 1;
    // 未满则按新的剩余空间 + 新序号重新入池（对应上游 erase 后 insert）
    if bins[b as usize].count < MAX_PAGES_PER_BIN {
      free.insert(free_key(bin_remaining(bins[b as usize]), seq), b);
      seq += 1;
    }
    out[i] = Some(BinSlot { bin: b, off });
  }
  out
}

/// 槽位剩余空间。`sum > PAGE_SIZE` 是调用方违约（例如把整块写缓冲的全局偏移
/// 写进 `sum`）；release 下 `PAGE_SIZE - sum` 会静默回绕成近 4 GiB 的“剩余空间”，
/// 让任意页都入选并把 `off` 写到槽位外（映射表静默损坏），故 debug/release 都 panic。
#[inline]
const fn bin_remaining(b: Bin) -> u32 {
  assert!(
    b.sum <= PAGE_SIZE as u32,
    "Bin.sum 超 PAGE_SIZE（调用方违约）"
  );
  PAGE_SIZE as u32 - b.sum
}

/// 复合键：高 32 位剩余空间，低 32 位插入序号
#[inline]
const fn free_key(remaining: u32, seq: u32) -> u64 {
  ((remaining as u64) << 32) | seq as u64
}

/// 装箱拷贝（上游 `CopyCompressedPages` 的纯函数化）：把 `src[start..=end]` 的
/// 压缩数据按 4 KiB 扇区不跨界写入 `dst_buf`，返回（写入后的页描述, 平均死亡时间）。
///
/// 写入后的 `ZipEntry.src_off` 是 `dst_buf` 内偏移，顺序与 `src[start..=end]` 一致。
///
/// 上游同时写 `dest->cur_size = align(末页末端, PAGE_SIZE) + PAGE_SIZE`
/// （`io_interface.cc:1215-1218`）；该状态写回由调用方负责。
///
/// 对齐规则：当前 4 KiB 段页数达 `MAX_PAGES_PER_BIN`，或本页会跨 4 KiB 边界时，
/// 先对齐到下一段起点。上游只按页数对齐（`io_interface.cc:1184-1193`），混合尺寸下
/// 会产生跨扇区页；本实现按 todo.md:183 的不变量收紧。
///
/// # Panics
/// `start > end` 或 `end >= src.len()`（调用方契约，上游 `Ensure(src->zpage_cnt > 0)`
/// 的等价物）；`dst_buf`/`src_buf` 不足以容纳数据（切片越界，调用方须按
/// `end - start + 1` 页 × `PAGE_SIZE` 预留）。
///
/// 前置契约：`comp_sz` 在 `1..=PAGE_SIZE`（压缩输出上限，C-19 回退路径恰为
/// `PAGE_SIZE`）；超过时无法满足 4 KiB 不变量，逐页 `assert!` 在 debug/release
/// 下都 panic（上游 `Ensure` 等价物，不是可静默忽略的 `None`）。
pub fn copy_compressed_pages(
  src: &[ZipEntry],
  src_buf: &[u8],
  dst_buf: &mut [u8],
  start: usize,
  end: usize,
) -> (Vec<ZipEntry>, LogId) {
  assert!(start <= end && end < src.len(), "页区间越界");

  let mut w_ptr = 0usize;
  let mut pages_in_seg = 0u32;
  let mut sum_edt: LogId = 0;
  let mut placed = Vec::with_capacity(end - start + 1);

  for e in &src[start..=end] {
    let sz = e.comp_sz as usize;
    assert!(sz <= PAGE_SIZE, "comp_sz 超 PAGE_SIZE（调用方违约）");
    // 段满或跨 4 KiB：对齐到下一段起点
    if pages_in_seg >= MAX_PAGES_PER_BIN || (w_ptr % PAGE_SIZE) + sz > PAGE_SIZE {
      w_ptr = w_ptr.div_ceil(PAGE_SIZE) * PAGE_SIZE;
      pages_in_seg = 0;
    }
    let s = e.src_off as usize;
    dst_buf[w_ptr..w_ptr + sz].copy_from_slice(&src_buf[s..s + sz]);
    placed.push(ZipEntry {
      comp_sz: e.comp_sz,
      death_time: e.death_time,
      src_off: w_ptr as u32,
    });
    w_ptr += sz;
    pages_in_seg += 1;
    sum_edt += e.death_time;
  }

  // 整数除法向零截断，与上游 `avg_edt / copied_pages_cnt` 一致
  let avg_edt = sum_edt / placed.len() as LogId;
  (placed, avg_edt)
}
