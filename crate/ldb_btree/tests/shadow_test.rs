//! 影子模型差分对拍（≥1e5 操作）：`BTree` vs `std::collections::BTreeMap`。
//!
//! 覆盖审查要求：find_child 路由正确性 + key range/fence/separator 不变量在
//! 大量操作下保持（`common::check_tree` 周期性全量校验）。

mod common;

use std::collections::BTreeMap;

use common::{check_tree, mk};

/// 确定性 xorshift64
struct Rng(u64);
impl Rng {
  fn next(&mut self) -> u64 {
    let mut x = self.0;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.0 = x;
    x
  }
}

fn key_of(i: u32, tag: u8) -> Vec<u8> {
  // 键空间足够大且带可变长度（2/4/8 字节尾），压 key range 路由
  let mut k = format!("u{tag:02}-{i:07}").into_bytes();
  k.extend_from_slice(&i.to_be_bytes()[..(i % 3 + 1) as usize]);
  k
}

#[compio::test]
async fn shadow_model_1e5_ops() {
  let mut t = mk();
  let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
  let mut rng = Rng(0x1234_5678_9ABC_DEF0);
  const OPS: u32 = 100_000;
  for step in 0..OPS {
    let r = rng.next();
    let i = (r % 5000) as u32;
    let tag = ((r >> 20) % 4) as u8;
    let k = key_of(i, tag);
    match r % 8 {
      // 70% 插入/upsert
      0..=4 => {
        let vlen = ((r >> 32) % 48) as usize + 1;
        let v = vec![(r & 0xFF) as u8; vlen];
        t.insert(&k, &v).await.unwrap();
        model.insert(k, v);
      }
      // 25% 删除
      5..=6 => {
        let expect = model.remove(&k).is_some();
        let got = t.remove(&k).await.unwrap();
        assert_eq!(got, expect, "step {step} 删除返回不符 key={i}");
      }
      // 5% 点查
      _ => {
        let mut out = Vec::new();
        let expect = model.get(&k).cloned();
        let got = t.look_up(&k, &mut out).await.unwrap();
        assert_eq!(got, expect.is_some(), "step {step} 命中不符 key={i}");
        if got {
          assert_eq!(out, expect.unwrap(), "step {step} 值不符 key={i}");
        }
      }
    }
    if step % 4096 == 0 {
      check_tree(&t);
      assert_eq!(
        t.count_entries().await.unwrap(),
        model.len() as u64,
        "step {step} 条目数不符"
      );
    }
  }
  // 终局全量对账
  check_tree(&t);
  assert_eq!(t.count_entries().await.unwrap(), model.len() as u64);
  let mut out = Vec::new();
  for (k, v) in &model {
    assert!(t.look_up(k, &mut out).await.unwrap(), "缺失 {k:?}");
    assert_eq!(&out, v, "值不符 {k:?}");
  }
  // 升序扫描 == 模型有序键值
  let mut scanned: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
  t.scan_ascending(b"", b"", |k, v| {
    scanned.push((k.to_vec(), v.to_vec()));
    true
  })
  .await
  .unwrap();
  let expect: Vec<(Vec<u8>, Vec<u8>)> = model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
  assert_eq!(scanned, expect, "扫描与模型不一致");
}
