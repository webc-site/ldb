//! 节点内键搜索（16-Hint 窄化 + 二分）。

use std::cmp::Ordering;

use ldb_simd::{get_head, search_hint};
use ldb_type::PageId;

use super::{BTreeNode, Cmp};

impl BTreeNode {
  /// 定位 `search_key` 的下界槽位（首个 >= key），命中置 `exact_found`。
  ///
  /// 返回 `Ok(slot)`；槽位 == count 表示落最右。
  pub fn lower_bound(&self, search_key: &[u8], exact_found: &mut bool, cmp: Cmp) -> u32 {
    *exact_found = false;

    // 前缀预筛（仅 MEMCMP 路径）
    if cmp.memcmp {
      let n = search_key.len().min(self.hdr.prefix_len as usize);
      let ret = (cmp.f)(&search_key[..n], self.get_prefix());
      if ret == Ordering::Less {
        return 0;
      }
      if ret == Ordering::Greater {
        return self.hdr.count;
      }
    }

    // key 是前缀的真子串：落首槽
    if search_key.len() < self.hdr.prefix_len as usize {
      return 0;
    }
    let key = &search_key[self.hdr.prefix_len as usize..];
    let key_head = get_head(key);

    // hint 窄化（与 blob 路径统一走 ldb_simd::search_hint，消除重复实现）。
    // 契约（ldb_simd R3 NOTES P2-2）：第 3 参必须是节点槽位总数 count，
    // 上游 `SearchHint` 的 `dist = upper_bound / (N+1)` 也以 count 为基；
    // 传已收窄的 upper 会改变 dist 与 hint 下标解释（P1）。
    let count = self.hdr.count as usize;
    let (mut lower, mut upper) = (0usize, count);
    if count > ldb_type::HINT_COUNT * 2 {
      let (l, u) = search_hint(&self.hdr.hints, &key_head, count);
      lower = l;
      if u < upper {
        upper = u;
      }
    }

    // 二分（head 4B 粗比较 + 全键细比较）
    while lower < upper {
      let mid = lower + (upper - lower) / 2;
      // 安全索引：公有字段可被外部改写，get_unchecked 会把畸形 count 变成 UB；
      // 合法路径由 `validate_bytes` 保证，越界时 panic 而非越界读
      let slot = &self.slots[mid];
      match key_head.cmp(&slot.head) {
        Ordering::Less => upper = mid,
        Ordering::Greater => lower = mid + 1,
        Ordering::Equal => {
          let slot_key = self.slice(slot.offset, slot.key_length);
          let n = key.len().min(slot_key.len());
          let ret = (cmp.f)(&key[..n], &slot_key[..n]);
          let ord = if ret == Ordering::Equal {
            key.len().cmp(&slot_key.len())
          } else {
            ret
          };
          match ord {
            Ordering::Less => upper = mid,
            Ordering::Greater => lower = mid + 1,
            Ordering::Equal => {
              *exact_found = true;
              return mid as u32;
            }
          }
        }
      }
    }
    lower as u32
  }

  /// 内节点找子指针
  pub fn find_child(&self, search_key: &[u8], pos: &mut u32, cmp: Cmp) -> PageId {
    let mut found = false;
    let p = self.lower_bound(search_key, &mut found, cmp);
    *pos = p;
    if p == self.hdr.count {
      self.hdr.right_most_child
    } else {
      self.get_child(p)
    }
  }

  /// Blob 键下界（泛型化：L2 不依赖 L4 的 BlobLookupKey，比较器由调用方提供）。
  ///
  /// 上游前置条件：prefix_len == 0 且 cmp.op == BLOB_LOOKUP；
  /// Rust 侧以 `blob_cmp` 闭包承载 BLOB_LOOKUP 语义（调用方负责把
  /// `cmp.func(GetKey(mid), key, ...)` 的实参顺序包装进闭包）。
  pub fn lower_bound_with_blob_key(
    &self,
    key_head: [u8; 4],
    key_rest: &[u8],
    exact_found: &mut bool,
    blob_cmp: impl Fn(&[u8], &[u8]) -> Ordering,
  ) -> u32 {
    *exact_found = false;
    debug_assert_eq!(self.hdr.prefix_len, 0, "Blob 索引前缀必须为空");

    let count = self.hdr.count as usize;
    let (mut lower, mut upper) = (0usize, count);
    if count > ldb_type::HINT_COUNT * 2 {
      let (l, u) = search_hint(&self.hdr.hints, &key_head, count);
      lower = l;
      if u < upper {
        upper = u;
      }
    }

    while lower < upper {
      let mid = lower + (upper - lower) / 2;
      let slot = &self.slots[mid];
      let slot_key = self.slice(slot.offset, slot.key_length);
      match blob_cmp(key_rest, slot_key) {
        Ordering::Less => upper = mid,
        Ordering::Greater => lower = mid + 1,
        Ordering::Equal => {
          *exact_found = true;
          return mid as u32;
        }
      }
    }
    lower as u32
  }

  /// Blob 键找子指针
  pub fn find_child_with_blob_key(
    &self,
    key_head: [u8; 4],
    key_rest: &[u8],
    pos: &mut u32,
    blob_cmp: impl Fn(&[u8], &[u8]) -> Ordering,
  ) -> PageId {
    let mut found = false;
    let p = self.lower_bound_with_blob_key(key_head, key_rest, &mut found, blob_cmp);
    *pos = p;
    if p == self.hdr.count {
      self.hdr.right_most_child
    } else {
      self.get_child(p)
    }
  }
}
