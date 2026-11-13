//! B+Tree 键搜索辅助（上游 `BTreeNode` 的 `MakeHint`/`SearchHint`/`UpdateHint`/
//! `GetHead`/`CommonPrefix` 重写为切片自由函数）。
//!
//! # 上游对照
//!
//! 新一代布局（本 crate 注释引用的行号基址）：
//! `ZLeanStore/src/storage/btree/node.cc` ——
//! `GetHead:47-51`、`CommonPrefix:68-81`、`MakeHint:220-225`、
//! `UpdateHint:227-239`、`SearchHint:241-266`，调用点
//! `LowerBound:300-302`、Blob 路径 `:683-686`。
//!
//! 旧一代布局 `leanstore/backend/leanstore/storage/btree/core/`：
//! `BTreeNode.cpp:16-21`（`makeHint`）、`:23-33`（`updateHint`）、
//! `BTreeNode.hpp:204-218`（`head`）、`:223-279`（`searchHint`）、
//! `BTreeNode.cpp:280-296`（`commonPrefix`）。两代算法等价；差异仅在 hint
//! 的存储形态（旧代 `u32` 大端值，新代 `[u8; 4]`）、`searchHint` 的 `dist`
//! 取自 `count` 还是 `upper_bound`（调用点均令 `upper_bound == count`），以及
//! 旧代的可选 AVX-512 变体（`FLAGS_btree_hints == 2`）。本 crate 以标量参考路径
//! （新代 `SearchHint`，等价于旧代 `FLAGS_btree_hints == 1`）为语义基准。
//!
//! # 16-Hint 数组
//!
//! `hints[i]` 存 `slots[dist * (i + 1)].head`，`dist = count / (HINT_COUNT + 1)`，
//! `HINT_COUNT = 16`。`SearchHint` 先用 hint 数组收缩 `[lower, upper)`，
//! 再在该区间内二分。hint 比较是 4 字节大端无符号比较（等价于上游
//! `std::memcmp(head, key_head, 4)` 的符号判定）。
//!
//! # 实现分档
//!
//! `common_prefix`：`< 8` 字节逐字节、`8..16` 每轮 8 字节 XOR、`>= 16`
//! `fearless_simd` 原生宽度向量。`search_hint`：lane 0 先标量判定（均匀 key
//! 分布下 `pos == 0` 约占 `1/(N+1)`，避免为最常见的命中付向量掩码固定开销），
//! 其余 hint 数 `>= 4` 走原生宽度向量；128 位后端从 lane 1 标量扫到 lane 4，
//! 再按宽度对齐做向量扫描。
//!
//! 实测（Apple M 系 aarch64/NEON，`benches/btree_simd_bench.rs`）：
//! - `common_prefix` 1024B：标量 176.9ns -> SIMD 17.32ns（10.2x）；
//!   128B 22.54ns -> 3.53ns（6.4x）；16B 3.99ns -> 1.48ns（2.7x）。
//! - `search_hint`（16 hint，count=64）：未命中 3.13ns -> 2.40ns（1.31x）；
//!   hint[15] 命中 3.46ns -> 2.72ns（1.27x）；hint[7] 命中 2.62ns -> 2.39ns
//!   （1.10x）；hint[0] 命中 1.50ns -> 1.49ns（持平，lane 0 走标量快路径）；
//!   4096 个随机命中位置批量 12.28µs -> 10.97µs（1.12x）。
//!
//! 全部实现无堆分配、无锁、无 `unsafe`。

use fearless_simd::{Level, Simd, SimdBase, SimdFrom, SimdMask, dispatch};

/// `fearless_simd` 原生向量最小宽度（128 位）；低于该长度走标量路径。
const SIMD_MIN: usize = 16;

/// 原生 `u32` 向量最大 lane 数（AVX-512）。
const MAX_LANES: usize = 16;

/// 键头：key 前 4 字节（不足补 0）。
///
/// 与 `ZLeanStore` `GetHead(node.cc:47-51)` 的 `memcpy(ret, key, min(len, 4))`
/// 逐字节一致；也等价于旧代 `head(BTreeNode.hpp:204-218)` 返回的大端 `u32`
/// （长度 1/2/3 分别左移 24/16/8 位）。
#[inline]
pub fn get_head(key: &[u8]) -> [u8; 4] {
  let mut ret = [0; 4];
  if key.len() >= 4 {
    ret.copy_from_slice(&key[..4]);
  } else {
    ret[..key.len()].copy_from_slice(key);
  }
  ret
}

/// 两个键的公共前缀长度（上游 `CommonPrefix(node.cc:68-81)` 的切片等价）。
///
/// 返回 `[0, min(lhs.len(), rhs.len())]`；首个不同字节的下标即返回值。
/// 旧代 `leanstore/.../BTreeNode.cpp:280-296` 在 `count == 0` 时直接返回 0
/// （避免访问未初始化槽位）；切片接口无 `count` 概念，调用方不应在
/// `count == 0` 时取键，等价性只在 `count > 0` 的调用域成立。
#[inline]
pub fn common_prefix(lhs: &[u8], rhs: &[u8]) -> usize {
  let n = lhs.len().min(rhs.len());
  if n < 8 {
    return prefix_bytes(lhs, rhs);
  }
  if n < SIMD_MIN {
    return prefix_words(lhs, rhs);
  }
  let level = Level::new();
  dispatch!(level, simd => prefix_simd(simd, lhs, rhs))
}

/// 逐字节前缀。
#[inline]
fn prefix_bytes(lhs: &[u8], rhs: &[u8]) -> usize {
  lhs.iter().zip(rhs).take_while(|(a, b)| a == b).count()
}

/// 8 字节字宽前缀：XOR 后 `trailing_zeros` 定位首个不同字节。
#[inline]
fn prefix_words(lhs: &[u8], rhs: &[u8]) -> usize {
  let mut lhs = lhs;
  let mut rhs = rhs;
  let mut off = 0;
  while let (Some((a, next_lhs)), Some((b, next_rhs))) =
    (lhs.split_first_chunk::<8>(), rhs.split_first_chunk::<8>())
  {
    // from_le_bytes 令字节 0 落在最低位，首个不同字节 = 最低置位 / 8。
    let diff = u64::from_le_bytes(*a) ^ u64::from_le_bytes(*b);
    if diff != 0 {
      return off + diff.trailing_zeros() as usize / 8;
    }
    off += 8;
    lhs = next_lhs;
    rhs = next_rhs;
  }
  off + prefix_bytes(lhs, rhs)
}

/// SIMD 前缀：每轮比较一个原生宽度向量，掩码定位首个不等 lane。
fn prefix_simd<S: Simd>(simd: S, lhs: &[u8], rhs: &[u8]) -> usize {
  let n = lhs.len().min(rhs.len());
  let width = S::u8s::LEN;
  let full = if width == 64 {
    u64::MAX
  } else {
    (1u64 << width) - 1
  };
  let mut off = 0;
  while off + width <= n {
    let a = S::u8s::from_slice(simd, &lhs[off..off + width]);
    let b = S::u8s::from_slice(simd, &rhs[off..off + width]);
    let equal = a.simd_eq(b).to_bitmask();
    if equal != full {
      return off + ((!equal) & full).trailing_zeros() as usize;
    }
    off += width;
  }
  off + prefix_words(&lhs[off..n], &rhs[off..n])
}

/// 由槽位 key head 就地重建 16-Hint 数组（上游 `MakeHint`）。
///
/// 契约：`N` 是 hint 数，调用方取 `HINT_COUNT`（= 16，`node.h:82`）。
/// 写入调用方提供的定长切片 —— 无堆分配（删除路径每次全量重建，
/// Vec 版会在 remove_slot 热路径上产生 64B 分配）。
///
/// 索引上界：`dist = count / (N + 1)`，
/// `dist * (idx + 1) <= N * count / (N + 1) < count`
/// （`count > 0`），故 `head_at` 只需覆盖 `[0, count)`。
/// `count == 0` 时上游同样读 `slots[0]`（未初始化槽位，C++ 侧为未定义行为），
/// 本函数保持该行为，由调用方保证 `head_at(0)` 可读。
#[inline]
pub fn make_hint<const N: usize>(
  hints: &mut [[u8; 4]; N],
  count: usize,
  head_at: impl Fn(usize) -> [u8; 4],
) {
  let dist = count / (N + 1);
  for (idx, hint) in hints.iter_mut().enumerate() {
    *hint = head_at(dist * (idx + 1));
  }
}

/// 槽位插入后增量更新受影响的 hint 段（上游 `UpdateHint`）。
///
/// 契约：`N` 是 hint 数，调用方取 `HINT_COUNT`（= 16）。
/// 三条件与上游逐字一致：
/// `count > 2 * N + 1`、`(count - 1) / (N + 1) == dist`、
/// `slot_id / dist > 1`；任一不满足则从 0 全量重扫。
///
/// `dist == 0`（`count <= N`）时不进入条件分支（短路求值，
/// 不会执行 `slot_id / dist`），循环写入 `head_at(0)` ——
/// 与上游把全部 hint 写成 `slots[0].head` 一致。
#[inline]
pub fn update_hint<const N: usize>(
  hints: &mut [[u8; 4]; N],
  count: usize,
  slot_id: usize,
  head_at: impl Fn(usize) -> [u8; 4],
) {
  let dist = count / (N + 1);
  let begin = if count > N * 2 + 1 && (count - 1) / (N + 1) == dist && slot_id / dist > 1 {
    slot_id / dist - 1
  } else {
    0
  };
  for (idx, hint) in hints.iter_mut().enumerate().skip(begin) {
    *hint = head_at(dist * (idx + 1));
  }
}

/// 用 16-Hint 数组收缩二分边界（上游 `SearchHint`）。
///
/// 契约：`N` 是 hint 数，调用方取 `HINT_COUNT`（= 16）；`count` 必须是
/// 节点槽位总数（上游 `SearchHint` 内的 `count`，也是 `lowerBound` 的
/// `upper` 初值，`node.cc:300-302`、`:683-686`）。此时守卫
/// `count <= N * 2` 与上游 `count > HINT_COUNT * 2` 取反等价。
///
/// 返回 `(lower, upper)`：`lower` 是首个 head >= key 的 hint 下标 * dist；
/// `upper` 是 `(首个 head != key 的 hint 下标 + 1) * dist`（未命中 `N`
/// 时保持入参上界 `count`）。两个边界都是合法的二分区间：
/// 所有 hint < key 时 `lower = N * dist` 仍不超过真实插入点。
///
/// 旧代 AVX-512 变体（`BTreeNode.hpp:233-234`）在 `compareMask == 0` 时直接
/// 返回、不更新 `lower`；本函数采用标量参考路径（`lower` 收紧到
/// `N * dist`），两者都只影响二分起点、不影响正确性。
#[inline]
pub fn search_hint<const N: usize>(
  hints: &[[u8; 4]; N],
  key_head: &[u8; 4],
  count: usize,
) -> (usize, usize) {
  if count <= N * 2 {
    return (0, count);
  }
  let dist = count / (N + 1);
  let key = u32::from_be_bytes(*key_head);

  // 首个 head >= key 的 hint（上游第一段循环）与等值 run 终点（上游第二段
  // 循环）一次返回。SIMD 只向量化第一段，run 终点在命中点确定后标量扫，
  // 与上游 AVX-512 变体同形。
  let (pos, pos2) = if N >= 4 {
    let level = Level::new();
    dispatch!(level, simd => first_ge_run_end(simd, hints, key))
  } else {
    let pos = first_ge_scalar(hints, key);
    (pos, equal_run_end(hints, key, pos))
  };

  let lower = pos * dist;
  let upper = if pos2 < N { (pos2 + 1) * dist } else { count };
  (lower, upper)
}

/// 从 `from` 起越过与 key 相等的 hint run，返回首个 `head != key` 的下标
/// （run 直达末尾则返回 `hints.len()`）。上游 `SearchHint` 第二段循环。
#[inline(always)]
fn equal_run_end(hints: &[[u8; 4]], key: u32, from: usize) -> usize {
  let mut idx = from;
  while idx < hints.len() && u32::from_be_bytes(hints[idx]) == key {
    idx += 1;
  }
  idx
}

/// 标量 `>=` 扫描：首个满足 `head >= key` 的下标，未命中返回 `hints.len()`。
#[inline]
fn first_ge_scalar(hints: &[[u8; 4]], key: u32) -> usize {
  hints
    .iter()
    .position(|h| u32::from_be_bytes(*h) >= key)
    .unwrap_or(hints.len())
}

/// SIMD `>=` 扫描 + 等值 run 终点：每轮吃一个原生宽度向量，掩码最低置位
/// 即首个命中 lane。
///
/// lane 0 单独用标量判定并直接返回 `(0, run_end)`：均匀 key 分布下
/// `pos == 0` 约占 `1/(N+1)`（hint 把节点均匀切分），上游标量实现把首段
/// 循环完全展开、`pos == 0` 走已知下标分支；若不单独判定，hit0 比标量基线
/// 慢约 20%（bench 实测）。128 位后端随后从 lane 1 起标量扫到 lane 4，
/// 与预扫循环共用同一条 lane 0 分支，不增加不可预测分支。
///
/// `#[inline(always)]` 是 `fearless_simd` 的要求：`dispatch!` 通过
/// `Simd::vectorize` 的闭包携带目标特性，只有内联进闭包的函数才会用该
/// 目标特性编译。
#[inline(always)]
fn first_ge_run_end<S: Simd>(simd: S, hints: &[[u8; 4]], key: u32) -> (usize, usize) {
  let width = S::u32s::LEN;
  let n = hints.len();
  // 常量分支：当前各 backend 宽度 <= 16，编译期消除；未来更宽的后端退回标量，
  // 保证下面的定长 lane 缓冲不越界。
  if width > MAX_LANES {
    let pos = first_ge_scalar(hints, key);
    return (pos, equal_run_end(hints, key, pos));
  }
  if u32::from_be_bytes(hints[0]) >= key {
    return (0, equal_run_end(hints, key, 0));
  }
  // 128 位后端（NEON / SSE2 / wasm simd128）：单个 4-lane 向量比较的掩码固定
  // 开销（NEON 为 `and+addv+fmov+rbit+clz`，约 8 条指令）高于数次标量比较，
  // 故 lane 1..4 标量扫；从 lane 4 起仍是宽度对齐的向量扫描。更宽后端
  // （AVX2/AVX-512）从 lane 0 向量化（lane 0 已判定 < key，重复比较无害）。
  let pre = if width == 4 { 4.min(n) } else { 0 };
  let mut idx = if width == 4 { 1 } else { 0 };
  while idx < pre {
    if u32::from_be_bytes(hints[idx]) >= key {
      return (idx, equal_run_end(hints, key, idx));
    }
    idx += 1;
  }
  let key_vec = S::u32s::simd_from(simd, key);
  while idx + width <= n {
    let mut lanes = [0u32; MAX_LANES];
    for (lane, head) in lanes.iter_mut().zip(&hints[idx..idx + width]) {
      *lane = u32::from_be_bytes(*head);
    }
    let heads = S::u32s::from_slice(simd, &lanes[..width]);
    let mask = heads.simd_ge(key_vec).to_bitmask();
    if mask != 0 {
      let pos = idx + mask.trailing_zeros() as usize;
      return (pos, equal_run_end(hints, key, pos));
    }
    idx += width;
  }
  let pos = idx + first_ge_scalar(&hints[idx..], key);
  (pos, equal_run_end(hints, key, pos))
}
