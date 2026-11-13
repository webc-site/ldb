//! 1e5 随机操作与影子模型逐字节比对（`ldb_page` 轮次登记的跨 crate P1）。
//!
//! 影子模型：`BTreeMap<Vec<u8>, Vec<u8>>` 提供有序 Oracle；每次操作后把节点
//! 的每个槽位（前缀 + 去前缀 key）与 payload 逐字节对比，并调用
//! `BTreeNode::validate` 复核结构不变量。
//!
//! 默认用例 1,000 步（<100 ms）；1e5 步用例标 `#[ignore]`，用
//! `cargo nextest run -p ldb_node --run-ignored only model_differential_100k`
//! 运行。

use std::collections::BTreeMap;

use ldb_node::{BTreeNode, MEMCMP, node::NUM_SLOTS};

struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    self.0 ^= self.0 >> 12;
    self.0 ^= self.0 << 25;
    self.0 ^= self.0 >> 27;
    self.0.wrapping_mul(2_685_821_657_736_338_717)
  }

  fn below(&mut self, n: usize) -> usize {
    (self.next() % n as u64) as usize
  }
}

const UNIVERSE: usize = 160;

/// 键空间：`a` + 第二字符 a..x + 十六进制 id；全部落在 fence `(aa, az]` 内，
/// 节点公共前缀为 1 字节（`a`），覆盖前缀剥离路径。
fn uni_key(id: usize) -> Vec<u8> {
  let ch = b'a' + (id % 24) as u8;
  let mut k = Vec::from([b'a', ch]);
  k.extend_from_slice(format!("{id:x}").as_bytes());
  k
}

fn uni_payload(id: usize, salt: u64) -> Vec<u8> {
  let len = ((id as u64 * 7 + salt * 13) % 48) as usize;
  (0..len)
    .map(|j| ((id as u64 * 31 + j as u64 * 17 + salt * 5) & 0xff) as u8)
    .collect()
}

fn assert_matches(n: &BTreeNode, model: &BTreeMap<Vec<u8>, Vec<u8>>) {
  assert_eq!(n.hdr.count as usize, model.len(), "槽位数与影子模型不一致");
  let prefix = n.get_prefix();
  for (i, (key, payload)) in model.iter().enumerate() {
    let slot = i as u32;
    let stored = n.get_key(slot);
    assert_eq!(
      prefix.len() + stored.len(),
      key.len(),
      "槽位 {i} 全键长度不符"
    );
    assert!(key.starts_with(prefix), "槽位 {i} 不含节点前缀");
    assert_eq!(&key[prefix.len()..], stored, "槽位 {i} key 字节不符");
    assert_eq!(
      n.get_payload(slot),
      payload.as_slice(),
      "槽位 {i} payload 不符"
    );
  }
  assert!(
    n.validate(MEMCMP).is_ok(),
    "节点结构校验失败：{:?}",
    n.validate(MEMCMP)
  );
}

fn run(steps: usize) {
  let mut n = BTreeNode::new(true, 0);
  n.set_fences(b"aa", b"az", MEMCMP);
  let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
  let mut rng = Rng(0xC0FF_EE12_3456_789A);

  for step in 0..steps {
    let r = rng.next();
    match r % 10 {
      // insert / upsert（含变长 payload）
      0..=3 => {
        let id = rng.below(UNIVERSE);
        let key = uni_key(id);
        let payload = uni_payload(id, step as u64);
        if n.insert_key_value(&key, &payload, MEMCMP) {
          model.insert(key, payload);
        } else {
          assert!(n.hdr.count as usize <= NUM_SLOTS, "失败路径不得破坏槽位数");
        }
      }
      // 删除
      4 | 5 => {
        if !model.is_empty() {
          let idx = rng.below(model.len());
          let key = model.keys().nth(idx).cloned();
          if let Some(key) = key {
            assert!(n.remove_key(&key, MEMCMP), "删除影子模型中的键必须成功");
            model.remove(&key);
          }
        }
      }
      // lower_bound 点查（命中/未命中/边界）
      6 | 7 => {
        let query = match r % 5 {
          0 => b"aa".to_vec(),
          1 => b"az".to_vec(),
          2 => b"zz".to_vec(), // 高于 upper fence：命中 prefix 预筛 Greater 分支
          3 => b"b".to_vec(),  // 首字节大于前缀 'a'
          _ => uni_key(rng.below(UNIVERSE)),
        };
        let mut exact = false;
        let pos = n.lower_bound(&query, &mut exact, MEMCMP);
        let expect_pos = model.range(..query.clone()).count();
        let found = model.contains_key(&query);
        assert_eq!(pos as usize, expect_pos, "步骤 {step} lower_bound 位置");
        assert_eq!(exact, found, "步骤 {step} exact 标志");
      }
      // 碎片压缩
      8 => n.compactify(MEMCMP),
      // 全键 payload 覆写（同长/变长）
      _ => {
        if !model.is_empty() {
          let idx = rng.below(model.len());
          let key = model.keys().nth(idx).cloned();
          if let Some(key) = key {
            let payload = uni_payload(idx % UNIVERSE, step as u64);
            if n.insert_key_value(&key, &payload, MEMCMP) {
              model.insert(key, payload);
            }
          }
        }
      }
    }
    assert_matches(&n, &model);
  }
}

#[test]
fn model_differential_1k() {
  run(1_000);
}

#[test]
#[ignore = "1e5 随机操作差分，>100ms；用 --run-ignored 手动运行"]
fn model_differential_100k() {
  run(100_000);
}
