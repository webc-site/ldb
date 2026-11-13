//! 第 2 轮**独立复核**测试（不引用 R1 的 `blob_store_test`/`blob_mgr_test` 用例）：
//! - P0-1：>2× 尾容量大对象盘面往返（3×/5×/恰好 2×；多整级 extent + 尾区段混合）
//! - `alloc_region` 每次取**独立**区段（记录全部分配区间并断言互不重叠）
//! - P0-2：删除后的去重索引/重复删除/悬空键
//! - extent 长度（`lp_size`/`length`）大于实际数据 → `BlobOutOfRange`
//! - SHA-256：`sha2_val` = 全内容摘要、去重命中复用、卸载后仍命中
//! - 扩展后 `sha2_val` 与去重索引必须随内容更新（上游 `AllocateBlob` 每次都重算）

use std::{
  cell::{Cell, RefCell},
  collections::BTreeMap,
  rc::Rc,
};

use ldb_blob::{BlobManager, BlobPageStore, Error, NoBlobStore};
use ldb_codec::PageCodec;
use ldb_type::{PAGE_SIZE, PageId};

/// 内存页仓 + 记录每次 `alloc_pages` 的 `(start, page_cnt)`（判定区段独立）。
#[derive(Debug, Default)]
struct RecStore {
  pages: RefCell<BTreeMap<PageId, [u8; PAGE_SIZE]>>,
  next: Cell<u64>,
  ranges: RefCell<Vec<(PageId, u64)>>,
}

impl RecStore {
  fn ranges(&self) -> Vec<(PageId, u64)> {
    self.ranges.borrow().clone()
  }
}

impl BlobPageStore for RecStore {
  const MANAGED: bool = true;

  async fn alloc_pages(&self, page_cnt: u32) -> ldb_blob::Result<PageId> {
    let start = self.next.get();
    self.next.set(start + u64::from(page_cnt));
    self.ranges.borrow_mut().push((start, u64::from(page_cnt)));
    Ok(start)
  }

  async fn write_page(&self, pid: PageId, page: &[u8; PAGE_SIZE]) -> ldb_blob::Result<()> {
    self.pages.borrow_mut().insert(pid, *page);
    Ok(())
  }

  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> ldb_blob::Result<()> {
    let pages = self.pages.borrow();
    let page = pages
      .get(&pid)
      .ok_or_else(|| Error::StorePage(format!("页 {pid} 未写")))?;
    *dst = *page;
    Ok(())
  }
}

/// 断言所有分配区段两两不重叠，且每段大小恒为 `pgs`。
fn assert_regions_disjoint(ranges: &[(PageId, u64)], pgs: u64) {
  for (_, c) in ranges {
    assert_eq!(*c, pgs, "每个 alloc_region 必须恰好取 {pgs} 页");
  }
  for i in 0..ranges.len() {
    for j in (i + 1)..ranges.len() {
      let (s1, c1) = ranges[i];
      let (s2, c2) = ranges[j];
      assert!(
        s1 + c1 <= s2 || s2 + c2 <= s1,
        "区段重叠: [{s1},{}) 与 [{s2},{})",
        s1 + c1,
        s2 + c2
      );
    }
  }
}

/// 尾部容量 `cap` 页、数据 `pages × PAGE_SIZE` 字节：落盘 → 卸载 → 逐字节读回。
async fn roundtrip_pages(cap: u64, pages: u64) {
  let store = Rc::new(RecStore::default());
  let mgr = BlobManager::with_store(cap, store.clone());
  let payload: Vec<u8> = (0..pages * PAGE_SIZE as u64)
    .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) % 251) as u8)
    .collect();
  let id = mgr.allocate_blob(&payload).await.unwrap();

  assert_regions_disjoint(&store.ranges(), cap);

  mgr.unload_all_blobs().await;
  let back = mgr.load_blob(id).await.unwrap();
  assert_eq!(
    back.len(),
    payload.len(),
    "cap={cap} pages={pages} 长度不符"
  );
  assert_eq!(
    back, payload,
    "cap={cap} pages={pages} 盘面重建不逐字节一致"
  );
}

/// 3×/5×/恰好 2× 尾容量（两种尾容量）。
#[compio::test]
async fn disk_roundtrip_scaled_tail_capacity() {
  roundtrip_pages(2, 4).await; // 恰好 2×
  roundtrip_pages(2, 6).await; // 3×
  roundtrip_pages(2, 10).await; // 5×
  roundtrip_pages(4, 8).await; // 恰好 2×
  roundtrip_pages(4, 12).await; // 3×
  roundtrip_pages(4, 20).await; // 5×
}

/// 多整级 extent + 非整页尾区段：落盘 → 卸载 → 逐字节读回 + 跨边界区间读。
#[compio::test]
async fn disk_roundtrip_multi_extent_plus_partial_tail() {
  let store = Rc::new(RecStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  // 7 页 + 123 字节：拆出 3 个 tier1 extent，尾 1 页 + 123 字节
  let len = 7 * PAGE_SIZE + 123;
  let payload: Vec<u8> = (0..len).map(|i| ((i * 7 + 3) % 241) as u8).collect();
  let id = mgr.allocate_blob(&payload).await.unwrap();
  assert_regions_disjoint(&store.ranges(), 2);

  mgr.unload_all_blobs().await;
  let back = mgr.load_blob(id).await.unwrap();
  assert_eq!(back, payload, "多 extent + 半尾页重建不一致");

  // 跨「最后整级 extent → 尾区段」边界的区间读（边界 = 3 extents × 2 页）
  let boundary = 3 * 2 * PAGE_SIZE as u64;
  let mid = mgr.load_blob_content(id, boundary - 8, 16).await.unwrap();
  assert_eq!(
    mid,
    &payload[(boundary - 8) as usize..(boundary + 8) as usize]
  );

  // 读尾部零头
  let tail = mgr
    .load_blob_content(id, len as u64 - 16, 16)
    .await
    .unwrap();
  assert_eq!(tail, &payload[len - 16..]);
}

/// 删除后同内容重分配：得到**新** id，旧 id 不可加载，去重索引指向新 id。
#[compio::test]
async fn remove_then_realloc_same_content_is_fresh() {
  let bm = BlobManager::new(4);
  let payload = vec![0xA5u8; 1234];
  let digest = PageCodec::sha256_digest(&payload);
  let a = bm.allocate_blob(&payload).await.unwrap();
  assert!(bm.remove_blob(a).await.unwrap());
  assert_eq!(
    bm.look_up_by_digest(&digest).await,
    None,
    "删除须摘除去重索引"
  );

  let b = bm.allocate_blob(&payload).await.unwrap();
  assert_ne!(a, b, "同内容重分配不得复用已删除 id");
  assert!(bm.load_blob(a).await.is_err(), "旧 id 必须不可加载");
  assert_eq!(bm.load_blob(b).await.unwrap(), payload);
  assert_eq!(bm.look_up_by_digest(&digest).await, Some(b));
}

/// 去重命中：同内容两次分配同 id；删除共享 id 后索引摘除；**异内容** blob 不受影响。
#[compio::test]
async fn dedup_shared_id_removal_does_not_affect_other_blobs() {
  let bm = BlobManager::new(4);
  let content = vec![7u8; 40];
  let digest = PageCodec::sha256_digest(&content);

  let a1 = bm.allocate_blob(&content).await.unwrap();
  let a2 = bm.allocate_blob(&content).await.unwrap();
  assert_eq!(a1, a2, "去重命中必须返回同一 id");

  // 另一异内容 blob
  let other = vec![9u8; 900];
  let oid = bm.allocate_blob(&other).await.unwrap();
  assert_ne!(oid, a1);

  assert!(bm.remove_blob(a1).await.unwrap());
  assert_eq!(bm.look_up_by_digest(&digest).await, None, "共享键须摘除");
  // 异内容 blob 完好
  assert_eq!(bm.load_blob(oid).await.unwrap(), other);
  assert_eq!(
    bm.look_up_by_digest(&PageCodec::sha256_digest(&other))
      .await,
    Some(oid)
  );
}

/// `fresh_blob_allocation` 同内容建两个独立 id：删除其一不得破坏另一个，且索引不悬挂。
#[compio::test]
async fn fresh_duplicate_content_remove_one_keeps_other() {
  let bm = BlobManager::new(4);
  let content = vec![3u8; 555];
  let digest = PageCodec::sha256_digest(&content);
  let id1 = bm.fresh_blob_allocation(digest, &content).await.unwrap();
  let id2 = bm.fresh_blob_allocation(digest, &content).await.unwrap();
  assert_ne!(id1, id2);

  // 删除 id2：id1 必须仍可加载
  assert!(bm.remove_blob(id2).await.unwrap());
  assert_eq!(bm.load_blob(id1).await.unwrap(), content, "另一副本被误删");
  // 索引要么指向 id1、要么空，但绝不能悬挂指向已删 id2
  assert_ne!(bm.look_up_by_digest(&digest).await, Some(id2), "索引悬挂");
}

/// 删除不存在的 blob / 重复删除。
#[compio::test]
async fn remove_nonexistent_and_double_remove() {
  let bm = BlobManager::new(4);
  assert!(!bm.remove_blob(9999).await.unwrap(), "删除不存在须 false");
  let a = bm.allocate_blob(&[1u8; 64]).await.unwrap();
  assert!(bm.remove_blob(a).await.unwrap());
  assert!(!bm.remove_blob(a).await.unwrap(), "重复删除须 false");
}

/// extent 长度（`lp_size`/`length`）大于实际数据覆盖 → `BlobOutOfRange`（彻底分析：本
/// crate 的校验只依赖内存 `BlobState.length` 与链上 tier 尺寸，**不**依赖 ldb_free 的
/// free-page WAL `lp_size`）。
#[compio::test]
async fn declared_length_beyond_chain_is_out_of_range() {
  let store = Rc::new(RecStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  let id = mgr.allocate_blob(&[1u8; 100]).await.unwrap();
  mgr.unload_all_blobs().await;
  {
    let blobs = mgr.blobs.read();
    let st = blobs.get(&id).unwrap();
    // 声明长度 = 尾缓冲覆盖(1 页) + 10 字节 → 超出链覆盖
    st.write().length = PAGE_SIZE as u64 + 10;
  }
  assert!(
    matches!(mgr.load_blob(id).await, Err(Error::BlobOutOfRange(_))),
    "声明长度超出链覆盖必须 BlobOutOfRange"
  );
  // 区间读同样受校验
  assert!(matches!(
    mgr.load_blob_content(id, PAGE_SIZE as u64, 10).await,
    Err(Error::BlobOutOfRange(_))
  ));
}

/// `sha2_val` = 全内容 SHA-256；去重键 = 该系统值；卸载（模拟重开）后去重仍命中。
#[compio::test]
async fn sha2_val_is_full_digest_and_dedup_survives_unload() {
  let bm = BlobManager::new(4);
  let content = vec![0x5Cu8; 2048];
  let digest = PageCodec::sha256_digest(&content);
  let id = bm.allocate_blob(&content).await.unwrap();
  {
    let blobs = bm.blobs.read();
    let st = blobs.get(&id).unwrap().read();
    assert_eq!(st.sha2_val, digest, "sha2_val 必须是全内容摘要");
    assert_eq!(&st.blob_prefix[..4], &content[..4], "前缀 = 内容前 32 字节");
  }
  bm.unload_all_blobs().await;
  let again = bm.allocate_blob(&content).await.unwrap();
  assert_eq!(again, id, "卸载后去重索引仍应命中同一 id");
}

/// `fresh_blob_allocation` 直接使用调用方传入的 `digest`（证明 callee 不再重算 SHA）：
/// 传入一个「错误」摘要，去重索引必须按该错误摘要登记。
#[compio::test]
async fn fresh_uses_passed_digest_without_recompute() {
  let bm = BlobManager::new(4);
  let content = vec![0x11u8; 700];
  let correct = PageCodec::sha256_digest(&content);
  let wrong = [0xABu8; 32];
  assert_ne!(correct, wrong);
  let id = bm.fresh_blob_allocation(wrong, &content).await.unwrap();
  assert_eq!(
    bm.look_up_by_digest(&wrong).await,
    Some(id),
    "callee 未使用传入摘要"
  );
  assert_eq!(
    bm.look_up_by_digest(&correct).await,
    None,
    "callee 若重算摘要则正确摘要会被登记（证伪）"
  );
  let _ = NoBlobStore;
}

/// 扩展后 `sha2_val` 与去重索引必须随内容更新（上游 `AllocateBlob` 每次都重算 SHA）。
/// 否则 `allocate_blob(旧内容)` 会命中已变更的 blob → 返回内容不符的数据。
#[compio::test]
async fn extend_refreshes_digest_and_dedup_index() {
  let bm = BlobManager::new(4);
  let head = vec![0x11u8; 300];
  let id = bm.allocate_blob(&head).await.unwrap();
  let mut full = head.clone();
  full.extend_from_slice(&[0x22u8; 500]);
  bm.extend_existing_blob(id, &[0x22u8; 500]).await.unwrap();
  assert_eq!(bm.load_blob(id).await.unwrap(), full, "扩展后内容");

  let expect = PageCodec::sha256_digest(&full);
  {
    let blobs = bm.blobs.read();
    assert_eq!(
      blobs.get(&id).unwrap().read().sha2_val,
      expect,
      "扩展后 sha2_val 必须 = 全内容摘要"
    );
  }
  // 新内容命中 id
  assert_eq!(
    bm.allocate_blob(&full).await.unwrap(),
    id,
    "扩展后新内容应去重命中"
  );
  // 旧内容不得命中 id（否则返回内容不符的 blob）
  let old = bm.allocate_blob(&head).await.unwrap();
  assert_ne!(old, id, "旧内容不得命中已扩展的 blob");
  assert_eq!(
    bm.load_blob(old).await.unwrap(),
    head,
    "旧内容必须得到独立 blob"
  );
}
