//! BlobManager / BlobState 回归（上游 `storage/blob/blob_manager.cc`）：
//! 分配/去重 / 扩展 / 区间读 / 删除 / 卸载与搬家 / 小对象前缀（C-20）/
//! 状态机影子模型。

use std::{cmp::Ordering, collections::BTreeMap};

use ldb_blob::*;
use ldb_codec::PageCodec;
use ldb_type::PAGE_SIZE;

#[compio::test]
async fn allocate_dedup_and_lookup() {
  let bm = BlobManager::new(4);
  let payload: Vec<u8> = (0..10_000u32).map(|i| (i % 253) as u8).collect();
  let a = bm.allocate_blob(&payload).await.unwrap();
  let b = bm.allocate_blob(&payload).await.unwrap();
  assert_eq!(a, b, "同内容必须去重为同一 blob");

  let other: Vec<u8> = (0..999u32).map(|i| (i % 7) as u8).collect();
  let c = bm.allocate_blob(&other).await.unwrap();
  assert_ne!(a, c);

  let back = bm.load_blob(a).await.unwrap();
  assert_eq!(back.len(), payload.len());
  assert_eq!(&back[..64], &payload[..64]);

  // 区间读
  let mid = bm.load_blob_content(a, 100, 50).await.unwrap();
  assert_eq!(mid.len(), 50);
  assert_eq!(&mid[..], &payload[100..150]);
  assert!(
    bm.load_blob_content(a, 9_999, 10).await.is_err(),
    "越界必须报错"
  );
}

#[compio::test]
async fn extend_and_remain() {
  let bm = BlobManager::new(4);
  let a = bm.allocate_blob(&[1u8; 100]).await.unwrap();
  let len = bm.extend_existing_blob(a, &[2u8; 200]).await.unwrap();
  assert_eq!(len, 300);
  // 尾页补齐
  let pad = bm.allocate_remain_content(a).await.unwrap();
  assert!(pad > 0 && pad < 4096, "补齐至整页: {pad}");
  let data = bm.load_blob(a).await.unwrap();
  assert_eq!(data.len(), 300 + pad as usize);
}

#[compio::test]
async fn tiny_blob_prefix_c20_fix() {
  // C-20 修复验证：< 32 字节对象的前缀拷贝不越界
  let bm = BlobManager::new(4);
  let a = bm.allocate_blob(b"tiny").await.unwrap();
  let guard = bm.blobs.read();
  let st = guard.get(&a).unwrap().read();
  // 前缀 = 内容前 4 字节 + 补零（上游 memcpy 语义；C-20 修复验证）
  assert_eq!(&st.blob_prefix[..4], b"tiny");
  assert!(st.blob_prefix[4..].iter().all(|&b| b == 0));
}

#[compio::test]
async fn remove_and_eviction_marks() {
  let bm = BlobManager::new(4);
  let a = bm.allocate_blob(&[9u8; 64]).await.unwrap();
  assert!(bm.mark_tail_extent_for_eviction(a));
  assert!(bm.remove_blob(a).await.unwrap());
  assert!(bm.load_blob(a).await.is_err());
  assert!(!bm.remove_blob(a).await.unwrap(), "重复删除返回 false");
}

#[compio::test]
async fn unload_all_and_move_tail() {
  let bm = BlobManager::new(4);
  for i in 0..5u64 {
    bm.allocate_blob(&vec![i as u8; 256]).await.unwrap();
  }
  let n = bm.unload_all_blobs().await;
  assert_eq!(n, 5, "全部卸载");
  // 尾搬家后仍可再写入
  let a = bm.allocate_blob(&[7u8; 128]).await.unwrap();
  bm.move_tail_extent(a, 9_999).await.unwrap();
  bm.write_new_data_to_last_extent(a, &[8u8; 64])
    .await
    .unwrap();
  let d = bm.load_blob(a).await.unwrap();
  assert_eq!(d.len(), 192);
}

#[compio::test]
async fn blob_state_misc() {
  let bm = BlobManager::new(4);
  let a = bm.allocate_blob(&vec![5u8; 900]).await.unwrap();
  {
    let guard = bm.blobs.read();
    let st = guard.get(&a).unwrap().read();
    assert_eq!(st.blob_id(), a);
    assert_eq!(st.page_count(), 1);
    assert!(st.malloc_size() > 900);
    assert!(!st.data().is_empty());
  }
  // 内容前缀比对（blob_prefix = 内容前 32 字节）
  let mut expect = [5u8; 32];
  let digest = PageCodec::sha256_digest(&vec![5u8; 900]);
  assert_eq!(
    bm.blob_state_comparison(a, &expect).await.unwrap(),
    Ordering::Equal
  );
  expect[0] = 0;
  assert_eq!(
    bm.blob_state_compare_with_string(a, &expect).await.unwrap(),
    Ordering::Greater
  );
  // 去重查找（SHA-256 键）
  assert_eq!(bm.look_up_by_digest(&digest).await, Some(a));
  let _ = &mut expect;
}

/// 删除后去重索引必须清空 ⇒ 再分配同内容得到**新** id，且旧 id 不可加载。
/// 唯一击杀：`remove_blob` 改回按 `blob_prefix`（≠ SHA-256）摘除 ⇒ 去重残留
/// 悬空 id，`allocate_blob` 返回旧 id，`load_blob` 报 BlobNotFound（断言 `b != a` 失败）。
#[compio::test]
async fn remove_clears_dedup_index_and_reallocation_is_fresh() {
  let bm = BlobManager::new(4);
  let payload = vec![3u8; 700];
  let digest = PageCodec::sha256_digest(&payload);
  let a = bm.allocate_blob(&payload).await.unwrap();
  assert_eq!(bm.look_up_by_digest(&digest).await, Some(a));

  assert!(bm.remove_blob(a).await.unwrap());
  assert_eq!(
    bm.look_up_by_digest(&digest).await,
    None,
    "删除后去重索引必须摘除（键 = SHA-256，不是内容前缀）"
  );

  // 同内容再分配：必须是与 a 不同的新 id，且可正常加载
  let b = bm.allocate_blob(&payload).await.unwrap();
  assert_ne!(a, b, "删除后重分配不得复用悬空 id");
  assert_eq!(bm.load_blob(b).await.unwrap(), payload);
  assert_eq!(bm.look_up_by_digest(&digest).await, Some(b));
}

/// `calculate_sha256` 必须写入并返回内容 SHA-256（上游 `CalculateSHA256` 成员语义），
/// 且不改动内容前缀 `blob_prefix`。
#[compio::test]
async fn calculate_sha256_writes_digest() {
  let bm = BlobManager::new(4);
  let payload = vec![7u8; 100];
  let id = bm.allocate_blob(&payload).await.unwrap();
  let expect = PageCodec::sha256_digest(&payload);
  let blobs = bm.blobs.read();
  let mut st = blobs.get(&id).unwrap().write();
  // fresh 路径已把全摘要写入 sha2_val（去重索引键）
  assert_eq!(st.sha2_val, expect);
  // 重新计算：返回值与写入值均为全摘要
  let got = st.calculate_sha256(&payload);
  assert_eq!(got, expect);
  assert_eq!(st.sha2_val, expect);
  // 前缀仍是内容前 32 字节
  assert_eq!(&st.blob_prefix[..4], &payload[..4]);
}

/// `fresh_blob_allocation`（`FreshBlobAllocation` 公开形态）跳过去重：同内容也建新 blob。
#[compio::test]
async fn fresh_blob_allocation_public_path() {
  let bm = BlobManager::new(4);
  let payload = vec![42u8; 500];
  let digest = PageCodec::sha256_digest(&payload);
  let id = bm.fresh_blob_allocation(digest, &payload).await.unwrap();
  assert_eq!(bm.load_blob(id).await.unwrap(), payload);
  assert_eq!(bm.look_up_by_digest(&digest).await, Some(id));
  // 直接建链，不再命中已有 blob
  let id2 = bm.fresh_blob_allocation(digest, &payload).await.unwrap();
  assert_ne!(id, id2, "fresh 路径必须跳过去重");
  assert_eq!(bm.load_blob(id2).await.unwrap(), payload);
}

/// LCG 伪随机（避免引入 dev-dep；测试内自足）。
struct Lcg(u64);

impl Lcg {
  fn below(&mut self, n: u64) -> u64 {
    self.0 = self
      .0
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    (self.0 >> 33) % n
  }
}

/// BlobManager 状态机影子模型：随机 create/load/remove 序列，断言
/// 「去重命中返回同 id」「删除后不可加载」「去重索引无悬挂」「内容逐字节可回读」。
async fn blob_shadow_model(rounds: usize) {
  let bm = BlobManager::new(4);
  let mut rng = Lcg(0xfeed_face_cafe_beef);
  let mut live: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
  let mut by_content: BTreeMap<Vec<u8>, u64> = BTreeMap::new();

  for _ in 0..rounds {
    match rng.below(3) {
      0 => {
        // allocate：内容随机，长度 0..3 页 + 尾零头
        let len = rng.below(3 * PAGE_SIZE as u64 + 17) as usize;
        let content: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
        let id = bm.allocate_blob(&content).await.unwrap();
        match by_content.get(&content) {
          Some(&e) => assert_eq!(id, e, "同内容必须去重为同一 id"),
          None => {
            by_content.insert(content.clone(), id);
            live.insert(id, content.clone());
          }
        }
        assert_eq!(bm.load_blob(id).await.unwrap(), content);
      }
      1 => {
        // load：随机选一个存活 id；无存活则分配一个
        let id = live.keys().next().copied();
        match id {
          Some(id) => assert_eq!(bm.load_blob(id).await.unwrap(), live[&id]),
          None => {
            let c = vec![9u8; 33];
            let id = bm.allocate_blob(&c).await.unwrap();
            by_content.insert(c.clone(), id);
            live.insert(id, c);
          }
        }
      }
      _ => {
        // remove：随机存活 id；删除后必须不可加载、去重索引清空
        let Some(id) = live.keys().next().copied() else {
          continue;
        };
        let content = live.remove(&id).unwrap();
        by_content.remove(&content);
        assert!(bm.remove_blob(id).await.unwrap());
        assert!(bm.load_blob(id).await.is_err(), "删除后不可加载");
        let digest = PageCodec::sha256_digest(&content);
        assert_eq!(bm.look_up_by_digest(&digest).await, None, "去重索引悬挂");
      }
    }
  }
}

#[compio::test]
async fn blob_shadow_model_short() {
  blob_shadow_model(300).await;
}

#[compio::test]
#[ignore = "1e5 随机 create/load/remove 状态机序列（重活，手动 --ignored 运行）"]
async fn blob_shadow_model_1e5() {
  blob_shadow_model(100_000).await;
}
