//! Blob 磁盘路径回归：extent 页经 `BlobPageStore` 落盘 / 加载（卸载常驻副本后
//! 读盘面；拆级页数据随 `split_to_extents` 持久化）。

use std::{
  cell::{Cell, RefCell},
  collections::BTreeMap,
  rc::Rc,
};

use ldb_blob::{BlobManager, BlobPageStore, Error, NoBlobStore};
use ldb_type::{PAGE_SIZE, PageId};

/// 内存页仓（`BTreeMap`：fuzz/tests 豁免口径 —— 有序断言需要）
#[derive(Debug, Default, Clone)]
struct MemPageStore {
  pages: RefCell<BTreeMap<PageId, [u8; PAGE_SIZE]>>,
  next: Cell<u64>,
  allocs: Cell<u64>,
}

impl MemPageStore {
  fn written_pages(&self) -> usize {
    self.pages.borrow().len()
  }
}

impl BlobPageStore for MemPageStore {
  const MANAGED: bool = true;

  async fn alloc_pages(&self, page_cnt: u32) -> ldb_blob::Result<PageId> {
    let start = self.next.get();
    self.next.set(start + u64::from(page_cnt));
    self.allocs.set(self.allocs.get() + 1);
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

#[compio::test]
async fn fresh_persists_extents_and_tail() {
  let store = Rc::new(MemPageStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  // 3 页（12 KiB）→ 触发一次拆级（尾容量 2 页）
  let payload: Vec<u8> = (0..3 * PAGE_SIZE).map(|i| (i % 251) as u8).collect();
  let id = mgr.allocate_blob(&payload).await.unwrap();

  assert!(store.written_pages() > 0, "extent 页必须落盘");
  // 盘面直读整级 extent 首页（其 pid 由分配器给出，不再假定为 0）
  let ext_pid = {
    let b = mgr.blobs.read();
    b.get(&id)
      .unwrap()
      .read()
      .extents
      .begin()
      .unwrap()
      .start_pid
  };
  let mut page = [0u8; PAGE_SIZE];
  store.read_page(ext_pid, &mut page).await.unwrap();
  assert_eq!(&page[..16], &payload[..16]);
  let _ = id;
}

#[compio::test]
async fn unload_then_load_reads_from_store() {
  let store = Rc::new(MemPageStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  // 跨拆级边界的大对象（4 页）
  let payload: Vec<u8> = (0..4 * PAGE_SIZE).map(|i| (i % 13) as u8).collect();
  let id = mgr.allocate_blob(&payload).await.unwrap();

  // 卸载常驻副本 → load 只能走盘面 extent 链
  assert_eq!(mgr.unload_all_blobs().await, 1);
  let back = mgr.load_blob(id).await.unwrap();
  assert_eq!(back.len(), payload.len());
  assert_eq!(back, payload, "盘面重建必须与原内容逐字节一致");
}

#[compio::test]
async fn ranged_read_after_unload() {
  let store = Rc::new(MemPageStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  let payload: Vec<u8> = (0..3 * PAGE_SIZE).map(|i| (i % 97) as u8).collect();
  let id = mgr.allocate_blob(&payload).await.unwrap();
  mgr.unload_all_blobs().await;

  // 跨尾缓冲 + 拆级页的区间读
  let mid = mgr
    .load_blob_content(id, PAGE_SIZE as u64 - 8, 16)
    .await
    .unwrap();
  assert_eq!(mid, &payload[PAGE_SIZE - 8..PAGE_SIZE + 8]);

  let tail = mgr
    .load_blob_content(id, (3 * PAGE_SIZE - 4) as u64, 4)
    .await
    .unwrap();
  assert_eq!(tail, &payload[3 * PAGE_SIZE - 4..]);
}

#[compio::test]
async fn extend_appends_and_persists() {
  let store = Rc::new(MemPageStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  let id = mgr.allocate_blob(&[1u8; PAGE_SIZE]).await.unwrap();
  let total = mgr
    .extend_existing_blob(id, &[2u8; PAGE_SIZE + 16])
    .await
    .unwrap();
  assert_eq!(total, (2 * PAGE_SIZE + 16) as u64);

  mgr.unload_all_blobs().await;
  let back = mgr.load_blob(id).await.unwrap();
  assert_eq!(&back[..PAGE_SIZE], &[1u8; PAGE_SIZE]);
  assert_eq!(&back[PAGE_SIZE..], &[2u8; PAGE_SIZE + 16]);
}

#[compio::test]
async fn resident_mode_still_works() {
  // NoBlobStore 常驻内存态：行为与磁盘路径无关（读走 resident）
  let mgr = BlobManager::<NoBlobStore>::new(4);
  let id = mgr.allocate_blob(&[7u8; 100]).await.unwrap();
  assert_eq!(mgr.load_blob(id).await.unwrap(), vec![7u8; 100]);
}

/// 盘面读必须按 extent 链的实际覆盖长度校验（extent 页数 = tier 尺寸属
/// 不可信元数据）：链覆盖不足 `length` 时必须报越界，而不是静默补零。
/// 唯一击杀：删掉 `read_range` 的覆盖校验 ⇒ 返回「前缀正确 + 尾部全零」的 Ok。
#[compio::test]
async fn disk_read_rejects_chain_shorter_than_length() {
  let store = Rc::new(MemPageStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  let id = mgr.allocate_blob(&[1u8; 100]).await.unwrap();
  mgr.unload_all_blobs().await; // 清 resident ⇒ 强制走盘面路径
  {
    let blobs = mgr.blobs.read();
    let st = blobs.get(&id).unwrap();
    st.write().length = PAGE_SIZE as u64 + 10; // 超出尾缓冲覆盖
  }
  assert!(
    matches!(mgr.load_blob(id).await, Err(Error::BlobOutOfRange(_))),
    "extent 链短于 length 时必须报越界而非静默补零"
  );
}

/// 多次拆级（数据 > 2×尾容量）后，尾区重锚不得覆盖已落盘的整级 extent。
/// 尾容量 2 页、6 页数据 → 拆出 2 个 tier1 extent + 1 页尾。
#[compio::test]
async fn multi_extent_large_blob_disk_roundtrip() {
  let store = Rc::new(MemPageStore::default());
  let mgr = BlobManager::with_store(2, store.clone());
  let payload: Vec<u8> = (0..6 * PAGE_SIZE).map(|i| (i % 199) as u8).collect();
  let id = mgr.allocate_blob(&payload).await.unwrap();
  mgr.unload_all_blobs().await;
  let back = mgr.load_blob(id).await.unwrap();
  assert_eq!(back.len(), payload.len());
  assert_eq!(back, payload, "多次拆级后盘面重建必须逐字节一致");
}
