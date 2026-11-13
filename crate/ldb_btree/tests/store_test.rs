//! 落盘接线阶段 1 回归：`ldb_buf` 单页 fix/unfix + `ldb_node` 编解码 + 脏页回写。
//!
//! 证明节点页可经 BufferManager 往返（写 → flush → 驱逐 → 缺页读回 → 校验），
//! 并拒绝畸形页（先校验后 cast）。

use std::{
  cell::RefCell,
  io,
  rc::Rc,
  sync::{Arc, atomic::AtomicBool},
};

use ldb_btree::{BufStore, Error};
use ldb_buf::{BufferManager, PageBackend};
use ldb_free::FreeStorageManager;
use ldb_io::MemBackend;
use ldb_node::MEMCMP;
use ldb_space::SpaceManager;
use ldb_type::{GsnWatermark, PAGE_SIZE, PageId};

/// 虚拟 4x 物理帧的内存缓冲池
fn mk_buf(pages: u64) -> BufferManager<MemBackend> {
  let space = Rc::new(SpaceManager::new(pages * 4, 64, 2048, 4, pages * 64, 2));
  let free = Arc::new(FreeStorageManager::new(true, 2));
  BufferManager::new(
    pages * 4,
    pages,
    space,
    free,
    MemBackend::new(),
    Arc::new(AtomicBool::new(true)),
    Arc::new(GsnWatermark::new(0)),
  )
  .unwrap()
}

/// 跨引擎实例共享同一持久化字节的后端（模拟「关停 → 重开」，落在同一盘面）。
#[derive(Clone, Debug)]
struct SharedStore(Rc<RefCell<Vec<u8>>>);

impl PageBackend for SharedStore {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
    let data = self.0.borrow();
    let off = (pid as usize) * PAGE_SIZE;
    let n = data.len().saturating_sub(off).min(PAGE_SIZE);
    dst.fill(0);
    if n > 0 {
      dst[..n].copy_from_slice(&data[off..off + n]);
    }
    Ok(())
  }

  async fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> io::Result<()> {
    let off = (pid as usize) * PAGE_SIZE;
    let mut data = self.0.borrow_mut();
    if data.len() < off + PAGE_SIZE {
      data.resize(off + PAGE_SIZE, 0);
    }
    data[off..off + PAGE_SIZE].copy_from_slice(src);
    Ok(())
  }

  async fn sync(&self) -> io::Result<()> {
    Ok(())
  }
}

/// 基于共享持久化字节构造缓冲池（重开引擎用；cache/space 全新，盘面不变）。
fn mk_shared_buf(pages: u64, store: Rc<RefCell<Vec<u8>>>) -> BufferManager<SharedStore> {
  let space = Rc::new(SpaceManager::new(pages * 4, 64, 2048, 4, pages * 64, 2));
  let free = Arc::new(FreeStorageManager::new(true, 2));
  BufferManager::new(
    pages * 4,
    pages,
    space,
    free,
    SharedStore(store),
    Arc::new(AtomicBool::new(true)),
    Arc::new(GsnWatermark::new(0)),
  )
  .unwrap()
}

#[compio::test]
async fn node_page_roundtrip_flush_evict_reload() {
  let bm = mk_buf(16);
  let store = BufStore::new(&bm, MEMCMP);

  let (pid, mut node) = store.create(true, 0).await.unwrap();
  for i in 0..5u32 {
    assert!(node.insert_key_value(format!("k{i}").as_bytes(), b"val", MEMCMP));
  }
  store.put(pid, &node).await.unwrap();
  // flush：脏页写盘 + 后端 sync
  store.flush().await.unwrap();

  // 驱逐常驻副本，强制 get 走缺页读回（证明磁盘镜像正确）
  let evicted = bm.evict_clean_pages(pid).await.unwrap();
  assert!(evicted, "flush 后应可驱逐该页");

  let back = store.get(pid).await.unwrap();
  assert_eq!(back.hdr.count, 5);
  assert!(back.hdr.is_leaf);
  for i in 0..5u32 {
    let mut exact = false;
    let pos = back.lower_bound(format!("k{i}").as_bytes(), &mut exact, MEMCMP);
    assert!(exact, "k{i} 缺失");
    assert_eq!(back.get_payload(pos), b"val");
  }
}

#[compio::test]
async fn corrupt_page_rejected_by_validation() {
  let bm = mk_buf(8);
  let store = BufStore::new(&bm, MEMCMP);
  let pid = store.alloc().await.unwrap();
  // 写入非法 is_leaf 字节（偏移 136，非 0/1）的整页镜像
  let mut bad = vec![0u8; PAGE_SIZE];
  bad[136] = 5;
  bm.update_pid(pid, &bad).await.unwrap();
  let err = store.get(pid).await.unwrap_err();
  assert!(
    matches!(err, Error::CorruptTree(_)),
    "畸形页必须 Err(CorruptTree) 而非 cast 使用: {err:?}"
  );
}

#[compio::test]
async fn alloc_dealloc_page_cycle() {
  let bm = mk_buf(16);
  let store = BufStore::new(&bm, MEMCMP);
  let p1 = store.alloc().await.unwrap();
  let p2 = store.alloc().await.unwrap();
  assert_ne!(p1, p2);
  store.dealloc(p2).await.unwrap();
  // 复用池可回收
  let p3 = store.alloc().await.unwrap();
  assert!(p3 >= p2, "分配页号单调或复用 p2");
}

/// 阶段 2 端到端持久化：**就地写**（`BufStore::modify` 经 `with_page_mut` 的 EXCLUSIVE
/// 视图页内编码）→ `flush_all` → 驱逐 → 缺页读回，断言键值逐字节一致。
///
/// 唯一击杀：把 `modify` 的写路径改回「不标脏」（删 `with_page_mut` 的
/// `make_page_dirty`）→ flush 无页可写 → 驱逐后读回 count 归零而 FAIL；或把就地编码
/// 换成「丢弃后重建」→ 键值不落页 → FAIL。
#[compio::test]
async fn stage2_inplace_write_flush_evict_reload_bytewise() {
  let bm = mk_buf(16);
  let store = BufStore::new(&bm, MEMCMP);
  let (pid, _) = store.create(true, 0).await.unwrap();
  // 页内就地插入 12 条键值（不重建整页）
  store
    .modify(pid, |node| {
      for i in 0..12u32 {
        assert!(node.insert_key_value(format!("k{i:03}").as_bytes(), &[i as u8; 32], MEMCMP));
      }
      Ok(())
    })
    .await
    .unwrap();
  // flush：脏页批量写盘 + 后端 sync
  store.flush().await.unwrap();

  // 驱逐常驻副本：强制缺页从盘读回
  assert!(bm.evict_clean_pages(pid).await.unwrap(), "flush 后应可驱逐");
  let back = store.get(pid).await.unwrap();
  assert_eq!(back.hdr.count, 12);
  for i in 0..12u32 {
    let mut exact = false;
    let pos = back.lower_bound(format!("k{i:03}").as_bytes(), &mut exact, MEMCMP);
    assert!(exact, "k{i:03} 缺页读回后缺失");
    assert_eq!(
      back.get_payload(pos),
      &[i as u8; 32],
      "k{i:03} 载荷字节不符"
    );
  }
}

/// 阶段 2 端到端持久化（**重开引擎**版）：写 → flush → 丢弃整个引擎（cache/space）
/// → 用同一盘面字节新建引擎 → 缺页读回，断言键值逐字节一致。
#[compio::test]
async fn stage2_reopen_engine_reloads_from_backend() {
  let backing: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::new()));
  let pid: PageId;
  {
    let bm = mk_shared_buf(16, backing.clone());
    let store = BufStore::new(&bm, MEMCMP);
    let (p, _) = store.create(true, 0).await.unwrap();
    pid = p;
    store
      .modify(pid, |node| {
        for i in 0..8u32 {
          assert!(node.insert_key_value(format!("r{i:02}").as_bytes(), &[i as u8; 16], MEMCMP));
        }
        Ok(())
      })
      .await
      .unwrap();
    store.flush().await.unwrap();
    // bm/store 在此 drop —— 模拟关停整个引擎
  }
  // 重开：全新 BufferManager（空 cache、新 space），同一持久化字节
  let bm2 = mk_shared_buf(16, backing.clone());
  let store2 = BufStore::new(&bm2, MEMCMP);
  let back = store2.get(pid).await.unwrap();
  assert_eq!(back.hdr.count, 8, "重开后节点计数应一致");
  for i in 0..8u32 {
    let mut exact = false;
    let pos = back.lower_bound(format!("r{i:02}").as_bytes(), &mut exact, MEMCMP);
    assert!(exact, "重开后 r{i:02} 缺失");
    assert_eq!(back.get_payload(pos), &[i as u8; 16]);
  }
}

/// 阶段 2 `p_gsn` 生产者：写路径经 `modify_with_gsn` 在页头登记该次修改的 WAL GSN
/// （上游 `transaction.cc:119`），使 WAL-before-page 门在**真实写路径**可达。
///
/// 唯一击杀：删 `modify_with_gsn` 的 `p_gsn` 赋值 → `node.hdr.page.p_gsn == 100`
/// 断言失败（读回 0）；或让写路径不标脏 → 首段 `page_is_dirty` 断言失败。
#[compio::test]
async fn stage2_p_gsn_producer_makes_gate_reachable() {
  let bm = mk_buf(16);
  let store = BufStore::new(&bm, MEMCMP);
  let (pid, _) = store.create(true, 0).await.unwrap();
  store
    .modify_with_gsn(pid, 100, |node| {
      assert!(node.insert_key_value(b"gk", b"gv", MEMCMP));
      Ok(())
    })
    .await
    .unwrap();
  // 生产者写入 p_gsn=100
  let node = store.get(pid).await.unwrap();
  assert_eq!(node.hdr.page.p_gsn, 100, "写路径必须登记 WAL GSN");

  // 已刷 WAL 水位 0 < p_gsn=100 → 检查点门挡下：页保持脏、不写出
  let _ = bm.fuzzy_checkpoint(0).await.unwrap();
  assert!(bm.page_is_dirty(pid), "p_gsn 未落 WAL 的脏页不得刷盘");
  // 水位推进到 100 → 允许写出
  bm.min_gsn_flushed.advance_min(100);
  let _ = bm.fuzzy_checkpoint(0).await.unwrap();
  assert!(!bm.page_is_dirty(pid), "水位追平后应写出并清脏");
}
