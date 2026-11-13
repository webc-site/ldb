//! upsert 回归（fuzz_btree_model 差分发现）：同键重插必须覆写旧值

use std::collections::BTreeMap;

use ldb_btree::BTree;
use ldb_node::MEMCMP;

fn k(i: u16) -> Vec<u8> {
  format!("k{i:05}").into_bytes()
}

#[compio::test]
async fn upsert_overwrites_duplicate_insert() {
  // 枚举 3 操作 × 2 键 × 2 值，找到最小失败序列
  for a in 0..4u8 {
    for b in 0..4u8 {
      for c in 0..4u8 {
        let mut t = BTree::create(MEMCMP);
        let mut m: BTreeMap<Vec<u8>, Vec<u8>> = Default::default();
        for (n, (op, key_id, val)) in [(a, 0u16, 255u8), (b, 1u16, 7u8), (c, 0u16, 9u8)]
          .into_iter()
          .enumerate()
        {
          let key = k(key_id);
          match op {
            0 => {
              t.insert(&key, &[val; 8]).await.unwrap();
              m.insert(key, vec![val; 8]);
            }
            1 => {
              if m.contains_key(&key) {
                t.update(&key, &[val; 8]).await.unwrap();
                *m.get_mut(&key).unwrap() = vec![val; 8];
              }
            }
            _ => {
              let expect = m.remove(&key).is_some();
              let got = t.remove(&key).await.unwrap();
              assert_eq!(got, expect);
            }
          }
          // 每步后全量对账
          for (mk, mv) in &m {
            let mut out = Vec::new();
            assert!(
              t.look_up(mk, &mut out).await.unwrap() && out == *mv,
              "序列 a={a} b={b} c={c} @ step{n}: op{op} key={mk:?} 期望 {mv:?} 得 {out:?}"
            );
          }
        }
      }
    }
  }
  eprintln!("枚举完成：2 键 3 操作内全部 upsert 语义正确");
}
