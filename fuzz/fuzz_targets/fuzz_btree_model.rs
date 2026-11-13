//! fuzz_btree_model（todo.md §6）：随机 KV 操作序列 vs `BTreeMap` 串行模型
//! 差分 + 键序不变量。（`BTreeMap` 为 fuzz 差分 Oracle —— §6 豁免口径）

#![no_main]
use std::collections::BTreeMap;

use compio::runtime::Runtime;
use ldb_btree::BTree;
use ldb_node::MEMCMP;
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target};

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Op {
  Insert { k: u16, v: u8 },
  Update { k: u16, v: u8 },
  Remove { k: u16 },
  LookUp { k: u16 },
}

fn key(k: u16) -> Vec<u8> {
  format!("k{k:05}").into_bytes()
}

fuzz_target!(|ops: Vec<Op>| {
  Runtime::new().unwrap().block_on(async {
    let mut tree = BTree::create(MEMCMP);
    let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    for (i, op) in ops.iter().enumerate() {
      match *op {
        Op::Insert { k, v } => {
          let key = key(k);
          tree.insert(&key, &[v; 8]).await.unwrap();
          model.insert(key, vec![v; 8]);
        }
        Op::Update { k, v } => {
          if model.contains_key(&key(k)) {
            tree.update(&key(k), &[v; 8]).await.unwrap();
            model.insert(key(k), vec![v; 8]);
          }
        }
        Op::Remove { k } => {
          let expect = model.remove(&key(k)).is_some();
          let got = tree.remove(&key(k)).await.unwrap();
          assert_eq!(got, expect, "remove 语义漂移 @ op{i}");
        }
        Op::LookUp { k } => {
          let mut out = Vec::new();
          let got = tree.look_up(&key(k), &mut out).await.unwrap();
          assert_eq!(got, model.contains_key(&key(k)), "look_up 语义漂移 @ op{i}");
          if got {
            assert_eq!(out, model[&key(k)], "值不一致 @ op{i}");
          }
        }
      }
    }
    // 全量对账：树内键数 = 模型键数
    let n = model.len() as u64;
    assert_eq!(tree.count_entries().await.unwrap(), n, "键数漂移");
  });
});
