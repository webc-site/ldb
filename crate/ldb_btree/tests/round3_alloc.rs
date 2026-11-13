//! Round-3 复核：`ldb_node::validate_bytes` 每记录 1 个 `Vec` 的分配/时延代价
//! （跨 crate：`crate/ldb_node/src/node/validate.rs:242` 的 `Vec::with_capacity`）。
//!
//! 带 fence 的页在 `BufStore::get`（页加载路径）逐记录分配一个 `full` 向量做
//! fence 比较；无 fence 的页走短路不分配。本文件用计数分配器量化 allocs/op 与
//! ns/op，给出对最终吞吐的影响量级（登记 CROSS）。
//!
//! 说明：全局分配器仅影响本测试二进制（不影响其它用例）。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  rc::Rc,
  sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
  },
  time::Instant,
};

use ldb_btree::BufStore;
use ldb_buf::BufferManager;
use ldb_free::FreeStorageManager;
use ldb_io::MemBackend;
use ldb_node::{BTreeNode, MEMCMP};
use ldb_space::SpaceManager;
use ldb_type::GsnWatermark;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, l: Layout) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(l.size(), Ordering::Relaxed);
    unsafe { System.alloc(l) }
  }
  unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
    unsafe { System.dealloc(p, l) }
  }
  unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(n, Ordering::Relaxed);
    unsafe { System.realloc(p, l, n) }
  }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

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

async fn measure(
  buf: &BufferManager<MemBackend>,
  pid: ldb_type::PageId,
  iters: u32,
  label: &str,
) -> (f64, f64) {
  let store = BufStore::new(buf, MEMCMP);
  // 预热：首帧缺页加载的内部分配不计入
  let _ = store.get(pid).await.unwrap();
  let a0 = ALLOCS.load(Ordering::Relaxed);
  let b0 = BYTES.load(Ordering::Relaxed);
  let t0 = Instant::now();
  let mut sink = 0u64;
  for _ in 0..iters {
    let n = store.get(pid).await.unwrap();
    sink += n.hdr.count as u64;
  }
  let dt = t0.elapsed();
  let allocs = ALLOCS.load(Ordering::Relaxed) - a0;
  let bytes = BYTES.load(Ordering::Relaxed) - b0;
  let per_op = dt.as_nanos() as f64 / f64::from(iters);
  let alloc_per_op = allocs as f64 / f64::from(iters);
  eprintln!(
    "{label}: {per_op:.1} ns/op, allocs/op={alloc_per_op:.2}, bytes/op={:.1} (sink={sink})",
    bytes as f64 / f64::from(iters),
  );
  (alloc_per_op, per_op)
}

#[compio::test]
async fn r3_validate_allocs_measure() {
  let bm = mk_buf(16);
  let store = BufStore::new(&bm, MEMCMP);

  // 有条目、无 fence 的页（40 记录）
  let (pid_plain, mut plain) = store.create(true, 0).await.unwrap();
  for i in 0..40u32 {
    assert!(plain.insert_key_value(format!("k{i:03}").as_bytes(), b"v", MEMCMP));
  }
  store.put(pid_plain, &plain).await.unwrap();

  // 有条目、双 fence 的页（40 记录，键区间与 fence 相容）
  let (pid_fenced, mut fenced) = store.create(true, 0).await.unwrap();
  fenced.set_fences(b"f", b"z", MEMCMP);
  for i in 0..40u32 {
    assert!(fenced.insert_key_value(format!("g{i:03}").as_bytes(), b"v", MEMCMP));
  }
  store.put(pid_fenced, &fenced).await.unwrap();

  // 自洽性：两页均 validate 通过
  plain.validate(MEMCMP).unwrap();
  fenced.validate(MEMCMP).unwrap();

  let (plain_alloc, _plain_ns) =
    measure(&bm, pid_plain, 20_000, "STORE get(40rec, 无 fence)").await;
  let (fenced_alloc, _fenced_ns) =
    measure(&bm, pid_fenced, 20_000, "STORE get(40rec, 有 fence)").await;

  // 无 fence 页：`BufStore::get` 自身仅 1 次分配（返回的 owning `Box<BTreeNode>`），
  // 证明页加载路径零多余缓冲（`fix_share` 直接以 Box 内存为目标页）。
  assert_eq!(plain_alloc, 1.0, "无 fence 页 get 应仅 1 次分配（Box）");
  // 有 fence 页：1(Box) + 40(每记录一个 fence 比较 `Vec`) —— 钉死跨 crate 缺陷
  // （`ldb_node/src/node/validate.rs:242`）作为 CROSS 金丝雀；ldb_node 修复后
  // 本断言与 CROSS 条目一并更新。
  assert!(
    fenced_alloc >= 1.0 + 40.0,
    "有 fence 页 get 应每条记录 1 个 Vec（1+40），实得 {fenced_alloc}"
  );
  let _ = BTreeNode::new(true, 0);
}
