//! 分隔键选择（最短截断分隔符，上游 `FindSeparator`）。

use ldb_simd::common_prefix;

use super::{BTreeNode, Cmp};

/// 分隔信息
#[derive(Debug, Clone, Copy)]
pub struct SeparatorInfo {
  /// 新分隔键长度
  pub len: u32,
  /// 分裂槽位
  pub slot: u32,
  /// 是否截断（true: 从 slot+1 取 len 字节）
  pub is_truncated: bool,
}

impl BTreeNode {
  /// 选分裂分隔键：内节点取中位；叶节点在 `[count/2 - count/16, count/2)` 内
  /// 找公共前缀最长处，并尝试截断（"aaaz"/"abaa" → "ab"）。
  pub fn find_separator(&self, append_bias: bool, cmp: Cmp) -> SeparatorInfo {
    assert!(self.hdr.count > 1);
    if self.is_inner() {
      let slot_id = self.hdr.count / 2;
      return SeparatorInfo {
        len: self.hdr.prefix_len + self.slots[slot_id as usize].key_length,
        slot: slot_id,
        is_truncated: false,
      };
    }
    debug_assert!(self.hdr.is_leaf);
    let count = self.hdr.count;
    let best_slot: u32 = if append_bias {
      count - 2
    } else if count > 16 && cmp.memcmp {
      let lower = count / 2 - count / 16;
      let upper = count / 2;
      let cp_with_first = |slot: u32| -> usize {
        let a = self.get_key(slot);
        let b = self.get_key(0);
        common_prefix(a, b)
      };
      let best_prefix_len = cp_with_first(lower);
      if cp_with_first(upper - 1) == best_prefix_len {
        lower
      } else {
        // 在窗口内推进到公共前缀变化处
        let mut s = lower + 1;
        while s < upper && cp_with_first(s) == best_prefix_len {
          s += 1;
        }
        s
      }
    } else {
      (count - 1) / 2
    };

    // 截断尝试
    if cmp.memcmp && best_slot + 1 < count {
      let ka = self.get_key(best_slot);
      let kb = self.get_key(best_slot + 1);
      let common = common_prefix(ka, kb);
      let truncate = ka.len() > common && kb.len() > common + 1;
      if truncate {
        return SeparatorInfo {
          len: self.hdr.prefix_len + common as u32 + 1,
          slot: best_slot,
          is_truncated: true,
        };
      }
    }

    SeparatorInfo {
      len: self.hdr.prefix_len + self.slots[best_slot as usize].key_length,
      slot: best_slot,
      is_truncated: false,
    }
  }

  /// 取分隔键字节（前缀 + 截断/全键）
  pub fn get_separator_key(&self, out: &mut [u8], info: &SeparatorInfo) {
    let pl = self.hdr.prefix_len as usize;
    debug_assert!(info.len as usize >= pl);
    debug_assert!(out.len() >= info.len as usize);
    let skip = u32::from(info.is_truncated);
    debug_assert!(info.slot + skip < self.hdr.count);
    let src = self.get_key(info.slot + skip);
    out[..pl].copy_from_slice(self.get_prefix());
    let body = info.len as usize - pl;
    debug_assert!(body <= src.len());
    out[pl..pl + body].copy_from_slice(&src[..body]);
  }
}
