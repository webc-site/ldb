//! 分隔键选择（上游 `BTreeNode::FindSeparator`）字面量与参考模型差分。

use std::collections::BTreeSet;

use ldb_node::{BTreeNode, Cmp, MEMCMP};

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
  a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// 上游 `FindSeparator` 逐行转写（叶节点部分）；`keys` 为去前缀后的 key。
fn find_sep_ref(
  keys: &[Vec<u8>],
  prefix_len: u32,
  append_bias: bool,
  memcmp: bool,
) -> (u32, u32, bool) {
  let count = keys.len() as u32;
  assert!(count > 1);
  let best_slot = if append_bias {
    count - 2
  } else if count > 16 && memcmp {
    let lower = count / 2 - count / 16;
    let upper = count / 2;
    let cp = |s: u32| common_prefix(&keys[s as usize], &keys[0]);
    let best_prefix_len = cp(lower);
    if cp(upper - 1) == best_prefix_len {
      lower
    } else {
      let mut s = lower + 1;
      while s < upper && cp(s) == best_prefix_len {
        s += 1;
      }
      s
    }
  } else {
    (count - 1) / 2
  };
  if memcmp && best_slot + 1 < count {
    let common = common_prefix(&keys[best_slot as usize], &keys[best_slot as usize + 1]);
    if keys[best_slot as usize].len() > common && keys[best_slot as usize + 1].len() > common + 1 {
      return (prefix_len + common as u32 + 1, best_slot, true);
    }
  }
  (
    prefix_len + keys[best_slot as usize].len() as u32,
    best_slot,
    false,
  )
}

fn build_leaf(fences: (&[u8], &[u8]), keys: &[Vec<u8>], cmp: Cmp) -> Box<BTreeNode> {
  let mut node = BTreeNode::new(true, 0);
  node.set_fences(fences.0, fences.1, cmp);
  for key in keys {
    assert!(
      node.insert_key_value(key, b"v", cmp),
      "构造用 key 必须可插入"
    );
  }
  node
}

fn check_case(fences: (&[u8], &[u8]), keys: &[Vec<u8>], append_bias: bool, cmp: Cmp) {
  let node = build_leaf(fences, keys, cmp);
  let prefix_len = node.hdr.prefix_len;
  let stored: Vec<Vec<u8>> = (0..node.hdr.count)
    .map(|i| node.get_key(i).to_vec())
    .collect();
  let (len, slot, trunc) = find_sep_ref(&stored, prefix_len, append_bias, cmp.memcmp);
  let info = node.find_separator(append_bias, cmp);
  assert_eq!(
    (info.len, info.slot, info.is_truncated),
    (len, slot, trunc),
    "分隔信息与上游参考模型不符: fences={fences:?} bias={append_bias}"
  );
  let mut got = vec![0u8; info.len as usize];
  node.get_separator_key(&mut got, &info);
  let body = (len - prefix_len) as usize;
  let mut expect = node.get_prefix()[..prefix_len as usize].to_vec();
  let src = if trunc { slot + 1 } else { slot } as usize;
  expect.extend_from_slice(&stored[src][..body]);
  assert_eq!(got, expect, "分隔键字节不符");
  assert!(node.validate(cmp).is_ok());
}

#[test]
fn upstream_truncation_example_aaaz_abaa() {
  let keys: Vec<Vec<u8>> = [b"aaaa", b"aaaz", b"abaa", b"accc"]
    .iter()
    .map(|k| k.to_vec())
    .collect();
  let node = build_leaf((b"", b""), &keys, MEMCMP);
  let info = node.find_separator(false, MEMCMP);
  assert_eq!((info.len, info.slot, info.is_truncated), (2, 1, true));
  let mut sep = [0u8; 2];
  node.get_separator_key(&mut sep, &info);
  assert_eq!(&sep, b"ab", "上游注释示例: aaaz/abaa -> ab");
}

#[test]
fn inner_node_splits_in_the_middle_without_truncation() {
  let keys: Vec<Vec<u8>> = [b"b", b"d", b"f", b"h", b"j"]
    .iter()
    .map(|k| k.to_vec())
    .collect();
  let mut node = build_leaf((b"", b""), &keys, MEMCMP);
  node.hdr.is_leaf = false;
  let info = node.find_separator(false, MEMCMP);
  assert_eq!((info.len, info.slot, info.is_truncated), (1, 2, false));
  let mut sep = [0u8; 1];
  node.get_separator_key(&mut sep, &info);
  assert_eq!(&sep, b"f");
}

#[test]
fn prefix_is_prepended_to_separator() {
  let full: Vec<Vec<u8>> = [b"ab", b"ac", b"ad", b"ae", b"af"]
    .iter()
    .map(|k| k.to_vec())
    .collect();
  let node = build_leaf((b"aa", b"az"), &full, MEMCMP);
  assert_eq!(node.hdr.prefix_len, 1);
  let info = node.find_separator(false, MEMCMP);
  assert_eq!((info.len, info.slot, info.is_truncated), (2, 2, false));
  let mut sep = [0u8; 2];
  node.get_separator_key(&mut sep, &info);
  assert_eq!(&sep, b"ad");
}

#[test]
fn empty_keys_and_boundaries() {
  // count=2：best_slot = 0，空键给出 len 0
  let keys: Vec<Vec<u8>> = [b"".as_slice(), b"a"].iter().map(|k| k.to_vec()).collect();
  let node = build_leaf((b"", b""), &keys, MEMCMP);
  let info = node.find_separator(false, MEMCMP);
  assert_eq!((info.len, info.slot, info.is_truncated), (0, 0, false));
  let mut sep = Vec::new();
  node.get_separator_key(&mut sep, &info);
  assert!(sep.is_empty());

  // append_bias：count=3 取 slot=1
  let keys: Vec<Vec<u8>> = [b"a".as_slice(), b"b", b"c"]
    .iter()
    .map(|k| k.to_vec())
    .collect();
  let node = build_leaf((b"", b""), &keys, MEMCMP);
  let info = node.find_separator(true, MEMCMP);
  assert_eq!((info.len, info.slot, info.is_truncated), (1, 1, false));
}

#[test]
fn separator_matches_upstream_reference_over_random_cases() {
  let mut state = 0x9E37_79B9_7F4A_7C15u64;
  let mut next = || {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
  };
  let memcmp_cmp = MEMCMP;
  let custom_cmp = Cmp {
    memcmp: false,
    f: ldb_node::cmp_memcmp,
  };
  for case in 0..500 {
    let count = 2 + (next() % 59) as usize;
    let prefix_mode = case % 4 == 0;
    let cmp = if case % 7 == 0 {
      custom_cmp
    } else {
      memcmp_cmp
    };
    let mut set = BTreeSet::new();
    while set.len() < count {
      let len = (next() % 9) as usize;
      let mut key = Vec::with_capacity(len + 2);
      if prefix_mode {
        // 必须 > lower fence "aa" 且 < upper fence "az"
        key.push(b'a');
        key.push(b'a' + (next() % 24) as u8);
        for _ in 0..=len {
          key.push(b'0' + (next() % 6) as u8);
        }
      } else {
        for _ in 0..len {
          key.push(b'0' + (next() % 6) as u8);
        }
      }
      set.insert(key);
    }
    let keys: Vec<Vec<u8>> = set.into_iter().collect();
    let fences: (&[u8], &[u8]) = if prefix_mode {
      (b"aa", b"az")
    } else {
      (b"", b"")
    };
    check_case(fences, &keys, (next() & 1) == 1, cmp);
  }
}
