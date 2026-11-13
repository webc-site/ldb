//! 故障注入回归（todo.md §4.1 第 4 防线）：掉电丢页 / 位翻转 / 撕裂写 +
//! 端到端「崩溃 → 重开」持久化语义 + WAL 损坏帧前缀一致性。

use std::{
  rc::Rc,
  sync::{Arc, atomic::AtomicBool},
};

use ldb_buf::{BufferManager, PageBackend};
use ldb_free::FreeStorageManager;
use ldb_io::MemBackend;
use ldb_page::hook::EvictHook;
use ldb_recovery::Replayer;
use ldb_space::SpaceManager;
use ldb_testkit::{FaultInjectBackend, FaultKind};
use ldb_type::{GsnWatermark, PAGE_SIZE, PageId};
use ldb_wal::{
  CommitVariant, GcCfg, GroupCommitExecutor, LogManager, WalFileState, entry::EntryType,
};

/// 内存缓冲池 + 故障注入后端
fn mk_buf(
  pages: u64,
  fb: &Rc<FaultInjectBackend<MemBackend>>,
) -> BufferManager<Rc<FaultInjectBackend<MemBackend>>> {
  let space = Rc::new(SpaceManager::new(pages, 64, 2048, 4, pages * 64, 2));
  let free = Arc::new(FreeStorageManager::new(true, 2));
  BufferManager::new(
    pages * 4,
    pages,
    space,
    free,
    fb.clone(),
    Arc::new(AtomicBool::new(true)),
    Arc::new(GsnWatermark::new(0)),
  )
  .unwrap()
}

#[compio::test]
async fn drop_write_fails_and_keeps_old_data() {
  let fb = Rc::new(FaultInjectBackend::new(MemBackend::new()));
  let bm = mk_buf(8, &fb);

  let pid: PageId = bm.alloc_page().await.unwrap();
  let payload = vec![0xA5u8; 128];
  bm.update_pid(pid, &payload).await.unwrap();
  // 固化第一版（写盘成功：脏页 → 1 次写）
  bm.flush_all().await.unwrap();
  assert_eq!(fb.write_cnt(), 1, "flush 后恰好一次页写");

  // 注入掉电：此后全部写失败
  fb.set_mode(FaultKind::DropWrite { at_write: 1 });
  let _ = bm.update_pid(pid, &[0x5A; 128]).await;
  let r = bm.flush_all().await;
  assert!(r.is_err(), "掉电注入后刷盘必须失败");
  assert_eq!(fb.injected_cnt(), 1);

  // 重开（同后端）：读到的是上次固化版本，未固化改写必须丢失
  let bm2 = mk_buf(8, &fb);
  let mut back = [0u8; PAGE_SIZE];
  bm2.fix_share(pid, &mut back).await.unwrap();
  assert!(
    back[..128].iter().all(|&b| b == 0xA5),
    "崩溃后必须回到上次固化版本"
  );
}

#[compio::test]
async fn bit_flip_is_visible_on_reread() {
  let fb = Rc::new(FaultInjectBackend::new(MemBackend::new()));
  let bm = mk_buf(8, &fb);
  let pid: PageId = bm.alloc_page().await.unwrap();
  let payload = vec![0x00u8; PAGE_SIZE];
  bm.update_pid(pid, &payload).await.unwrap();

  fb.set_mode(FaultKind::BitFlip {
    at_write: 0,
    offset: 7,
  });
  bm.flush_all().await.unwrap();
  fb.clear();

  // 直读盘面（常驻帧是未翻转的内存视图）
  let mut back = [0u8; PAGE_SIZE];
  fb.inner().read_page(pid, &mut back).await.unwrap();
  assert_eq!(back[7], 1, "位翻转必须落在第 7 字节盘面");
  assert_eq!(fb.injected_cnt(), 1);
}

#[compio::test]
async fn truncate_zeroes_tail() {
  let fb = Rc::new(FaultInjectBackend::new(MemBackend::new()));
  let bm = mk_buf(8, &fb);
  let pid: PageId = bm.alloc_page().await.unwrap();
  let payload = vec![0xFFu8; PAGE_SIZE];
  bm.update_pid(pid, &payload).await.unwrap();

  fb.set_mode(FaultKind::Truncate {
    at_write: 0,
    keep: 512,
  });
  bm.flush_all().await.unwrap();
  fb.clear();

  // 直读盘面（撕裂写语义在盘上，不在常驻帧）
  let mut back = [0u8; PAGE_SIZE];
  fb.inner().read_page(pid, &mut back).await.unwrap();
  assert!(back[..512].iter().all(|&b| b == 0xFF), "前缀完好");
  assert!(back[512..].iter().all(|&b| b == 0), "尾部被撕裂清零");
}

#[compio::test]
async fn mem_backend_page_granularity_roundtrip() {
  // 直通后端读写往返（故障注入底座健全性）
  let m = MemBackend::new();
  let page = [0x42u8; PAGE_SIZE];
  m.write_page(3, &page).await.unwrap();
  let mut out = [0u8; PAGE_SIZE];
  m.read_page(3, &mut out).await.unwrap();
  assert_eq!(out, page);
  // 稀疏洞语义：未写页读零
  let mut hole = [1u8; PAGE_SIZE];
  m.read_page(9, &mut hole).await.unwrap();
  assert!(hole.iter().all(|&b| b == 0));
}

#[compio::test]
async fn crash_reopen_wal_group_commit_prefix() {
  // 端到端：LogManager(MemBackend sink) 写 TX_COMMIT → 组提交 5 阶段落盘
  // → 「崩溃」（丢执行器）→ 重开 → WAL 前缀一致性（重放计数固定）
  let sink = MemBackend::new();
  let file = WalFileState {
    start: 0,
    end: 1 << 20,
    shard: 1 << 18,
  };
  let mgr = LogManager::new(2, 64 * 1024, CommitVariant::Baseline, Box::new(sink), file);
  let w = mgr.local_log_worker(0).clone();

  // 两个已提交事务 + 一个只有 TX_START 的事务（崩溃时刻在途）
  for (ty, ts) in [
    (EntryType::TxStart, 0),
    (EntryType::TxCommit, 1),
    (EntryType::TxStart, 0),
    (EntryType::TxCommit, 2),
    (EntryType::TxStart, 0),
  ] {
    w.reserve_and_submit(&mgr, ty, 0, &[], ts).await.unwrap();
  }
  // 组提交一轮：聚合 + fdatasync（MemBackend sync 为 no-op）
  let mut exec = GroupCommitExecutor::new(
    GcCfg {
      start_wid: 0,
      end_wid: 2,
      write_buffer_size: 64 * 1024,
      wal_start: file.start,
      wal_end: file.end,
      shard_wal_size: file.shard,
    },
    NoEvict,
    None,
    None,
  );
  exec.execute_one_round(&mgr).await.unwrap();

  // 「崩溃」：从 sink 读出 WAL 区内容做重放（重开形态）
  let wal_bytes = mgr
    .local_log_worker(0)
    .buffer
    .lock()
    .slice(0, 4096)
    .to_vec();
  drop(exec);

  let mut rp = Replayer::new();
  let frames = rp.replay_wal_shard(&wal_bytes, &mgr).await.unwrap();
  assert!(frames >= 5, "5 条帧全部有效（前缀完整），实际 {frames}");
  assert_eq!(rp.active_txns.len(), 1, "在途 TX_START 事务被识别");
  let undone = rp.undo_active_txns().await.unwrap();
  assert_eq!(undone, 1, "未提交事务回滚计数");
}

/// 组提交 PhaseFour 无驱逐钩子（单测形态）
struct NoEvict;

impl EvictHook for NoEvict {
  async fn evict_extent(&mut self, _extent: PageId) {}
  async fn public_free_ranges(&mut self) -> Vec<(u64, u64)> {
    Vec::new()
  }
}
