//! Facade 回归：开库 / CRUD / 事务 / Blob / 统计 / 任务钩子 / 版本。

use std::{
  env::temp_dir,
  fs::{create_dir_all, remove_dir_all},
  process::id,
  sync::atomic::Ordering,
};

#[ctor::ctor(unsafe)]
fn init_logging() {
  log_init::init();
}

use ldb::{Config, LeanStore};
use ldb_node::{BTreeNode, MEMCMP};
use ldb_space::BlockState;
use ldb_txn::IsolationLevel;
use ldb_type::{BlockId, PAGE_SIZE};

#[compio::test]
async fn open_crud_txn_roundtrip() {
  let mut db = LeanStore::open(Config::default()).await.unwrap();
  assert!(db.get_version() != 0);

  // KV CRUD（主树）
  for i in 0..40u64 {
    db.tree
      .insert(&format!("f{i:03}").into_bytes(), &i.to_le_bytes())
      .await
      .unwrap();
  }
  let mut out = Vec::new();
  assert!(db.tree.look_up(b"f007", &mut out).await.unwrap());
  assert_eq!(out, 7u64.to_le_bytes());

  // 事务生命周期（真实 WAL 链：TX_COMMIT → 组提交固化 → 水位）
  db.start_transaction(IsolationLevel::SnapshotIsolation)
    .await;
  assert!(db.txns.workers[0].lock().active.is_running());
  let ts = db.commit_transaction().await.unwrap();
  assert!(ts > 0);

  db.start_transaction(IsolationLevel::ReadCommitted).await;
  db.abort_transaction().await.unwrap();
  assert!(!db.txns.workers[0].lock().active.is_running());
}

#[compio::test]
async fn blob_lifecycle_via_facade() {
  let mut db = LeanStore::open(Config::default()).await.unwrap();
  let id = db.create_new_blob(&vec![9u8; 300]).await.unwrap();
  let data = db.load_blob(id).await.unwrap();
  assert_eq!(data.len(), 300);
  assert!(db.remove_blob(id).await.unwrap());
  assert!(!db.remove_blob(id).await.unwrap());
}

#[compio::test]
async fn stats_and_tasks() {
  let mut db = LeanStore::open(Config {
    frames: 1024,
    ..Default::default()
  })
  .await
  .unwrap();
  let idx = db.register_table("t").await;
  assert_eq!(idx, 0);
  assert!(db.retrieve_index(0).await.is_some());
  assert!(db.retrieve_index(9).await.is_none());
  let _ = db.retrieve_comparison_func().await;
  let _ = db.db_size().await;
  let _ = db.wal_size().await;
  let _ = db.allocated_pi_ds_cnt().await;
  let _ = db.allocated_size().await;
  db.start_gc_task().await.unwrap();
  db.start_checkpoint_task().await.unwrap();
  db.start_group_commit_thread().await.unwrap();
  let _ = db.check_during_idle().await;
  db.end_data_load().await;
  db.drop_cache().await;
  db.shutdown().await;
  assert!(!db.txns.is_running.load(Ordering::Relaxed));
}

/// P0 CROSS-GC-R2：GC 编排闭合 —— `on_gc_needed` 必须走完
/// `FULL → GC_READ → GC_WRITE → EMPTY` 并返回真实回收结果。
///
/// 唯一击杀：还原 engine 的「`pids.is_empty()` 直接 `erase_block; return true`」
/// （不迁移 state）→ victim 恒为 Full、`erase_block` 返 false、块仍 Full ⇒
/// 末行 `state == Empty` 断言失败；或「只把返回值改成 erase 结果而不迁移」→
/// `reclaimed` 断言失败。
#[compio::test]
async fn gc_reclaims_full_victim_end_to_end() {
  let db = LeanStore::open(Config {
    frames: 1024,
    ..Default::default()
  })
  .await
  .unwrap();
  // 造一个 FULL 全死块（valid=0 立即被贪心选中）：选空块 → 关闭至 FULL →
  // 注入 total/invalid（与 ldb_gc 测试 occupy 同构）。
  let bid = db.space.select_empty_block_id(0, 42);
  assert!(bid != BlockId::MAX, "应取到空块");
  assert!(
    db.space
      .move_block_to_list(bid, BlockState::Open, BlockState::InUse)
  );
  assert!(
    db.space.close_one_open_block(bid, 64 * PAGE_SIZE as u64),
    "关闭至 FULL 失败"
  );
  {
    let mut s = db.space.sspace.write();
    let m = &mut s.block_metadata[bid as usize];
    m.total_pid_cnt = 8;
    m.invalidated_pid_cnt = 8; // valid == 0 → 立即选为 victim
  }
  assert_eq!(db.space.block_meta(bid).state, BlockState::Full);
  // FacadeGc 经 space.set_gc_hook 注入 → 直接驱动一轮 GC 编排
  let hook = db.space.gc_hook.get().expect("GC hook 应已注入");
  let reclaimed = hook.on_gc_needed().await;
  assert!(reclaimed, "GC 应推进回收（返回 erase_block 的真实结果）");
  assert_eq!(
    db.space.block_meta(bid).state,
    BlockState::Empty,
    "victim 必须被真正擦除（走完 FULL→GC_WRITE→EMPTY）"
  );
}

/// B7 接线：`ldb_txn` 提交/回滚计数经 facade 注入的 `StatCollector` 可见。
///
/// 唯一击杀：删 `open()` 的 `txns.set_stats`（收集器恒空）或删 `txn_mgr` 的
/// `record_txn_commit`/`record_txn_abort`，对应断言归零而 FAIL。
#[compio::test]
async fn stat_counters_wired_through_facade() {
  let db = LeanStore::open(Config::default()).await.unwrap();
  db.start_transaction(IsolationLevel::SnapshotIsolation)
    .await;
  db.commit_transaction().await.unwrap();
  db.start_transaction(IsolationLevel::ReadCommitted).await;
  db.abort_transaction().await.unwrap();
  let m = db.stat().snapshot_metric();
  assert_eq!(m.txn_commit, 1, "一次提交记一次 commit");
  assert_eq!(m.txn_abort, 1, "一次回滚记一次 abort");
}

/// B6：`wal_sync` 进入配置指纹（改语义配置必改版本号）。
#[compio::test]
async fn wal_sync_affects_version_fingerprint() {
  let on = LeanStore::open(Config {
    wal_sync: true,
    ..Default::default()
  })
  .await
  .unwrap();
  let off = LeanStore::open(Config {
    wal_sync: false,
    ..Default::default()
  })
  .await
  .unwrap();
  assert_ne!(
    on.get_version(),
    off.get_version(),
    "wal_sync 必须进入配置指纹"
  );
}

/// 文件后端装配（`path != ":memory:"` → `data.db`/`wal.log` 分文件；WAL 落盘屏障
/// 走 [`ldb::engine`] 的 `DurableWalFile` → `sync_durable`）：开库 → KV → 提交 →
/// 关停，全程不 panic 且两文件落地。
///
/// 唯一击杀：把 `open()` 的 `config.path == MEM_PATH` 分派改成恒走 Mem 后端
/// （文件分支消失）→ 末两行 `data.db`/`wal.log` 存在断言 FAIL；并顺带覆盖
/// `DurableWalFile`（文件 WAL sink）的构造与提交全链。
#[compio::test]
async fn file_backend_roundtrip() {
  let dir = temp_dir().join(format!("ldb_r2_file_{}", id()));
  let _ = remove_dir_all(&dir);
  create_dir_all(&dir).unwrap();
  {
    let mut db = LeanStore::open(Config {
      path: dir.to_string_lossy().into_owned(),
      frames: 64,
      ..Default::default()
    })
    .await
    .unwrap();
    db.tree.insert(b"k", b"v").await.unwrap();
    db.start_transaction(IsolationLevel::SnapshotIsolation)
      .await;
    let ts = db.commit_transaction().await.unwrap();
    assert!(ts > 0, "文件后端提交应返回有效时间戳");
    db.shutdown().await;
  }
  assert!(dir.join("data.db").exists(), "应落地 data.db");
  assert!(dir.join("wal.log").exists(), "应落地 wal.log");
  let _ = remove_dir_all(&dir);
}

/// B7 接线（缓冲池侧）：`open()` 的 `buf.set_stats` 必须把门面 `StatCollector`
/// 注入缓冲池 —— 一次「缺页 → 命中 → 脏页驱逐」经 `db.stat()` 可见，且各恰好 1。
///
/// 唯一击杀：删 `open()` 的 `buf.set_stats(stat.clone())` → 全部断言归零 FAIL；
/// 在 `fix_shared` 命中路径也记 miss（双记）→ `page_miss` 断言失败。
#[compio::test]
async fn stat_buf_wiring_through_facade() {
  let db = LeanStore::open(Config {
    frames: 64,
    ..Default::default()
  })
  .await
  .unwrap();
  let pid = db.buf.alloc_page().await.unwrap();
  // 首次触达：EVICTED → 缺页加载一次（miss=1，不追加 hit）
  db.buf.update_pid(pid, &[0xAAu8; 16]).await.unwrap();
  assert_eq!(db.stat().snapshot_metric().page_miss, 1, "一次缺页记一次");
  // 页已常驻：共享快照读（btree/blob 生产路径 `fix_share`）命中一次（hit=1）
  let mut out = [0u8; PAGE_SIZE];
  db.buf.fix_share(pid, &mut out).await.unwrap();
  assert_eq!(db.stat().snapshot_metric().page_hit, 1, "一次命中记一次");
  // 脏页驱逐：设备写一次（page_evict=1）
  assert_eq!(db.buf.evict(1).await.unwrap(), 1, "脏页应被驱逐");
  let m = db.stat().snapshot_metric();
  assert_eq!(m.page_evict, 1, "一次脏页驱逐记一次");
}

/// GC 多轮回收不泄漏空间：同一批空块反复 FULL→GC→EMPTY，分配水位不增长。
///
/// 唯一击杀：`reclaim_victim` 的迁移链只走一半（victim 留 `GcRead` 不再被
/// `greedy_gc` 选中）→ 第 2 轮 `on_gc_needed` 返回 false / 块非 Empty → FAIL。
#[compio::test]
async fn gc_multi_round_no_space_leak() {
  let db = LeanStore::open(Config {
    frames: 1024,
    ..Default::default()
  })
  .await
  .unwrap();
  let hook = db.space.gc_hook.get().expect("GC hook 应已注入");
  let before = db.allocated_pi_ds_cnt().await;
  for round in 0..8u64 {
    let bid = db.space.select_empty_block_id(0, 42 + round);
    assert!(bid != BlockId::MAX, "round {round}: 应取到空块");
    assert!(
      db.space
        .move_block_to_list(bid, BlockState::Open, BlockState::InUse)
    );
    assert!(db.space.close_one_open_block(bid, 64 * PAGE_SIZE as u64));
    {
      let mut s = db.space.sspace.write();
      let m = &mut s.block_metadata[bid as usize];
      m.total_pid_cnt = 8;
      m.invalidated_pid_cnt = 8;
    }
    assert_eq!(db.space.block_meta(bid).state, BlockState::Full);
    assert!(hook.on_gc_needed().await, "round {round}: GC 应推进回收");
    assert_eq!(
      db.space.block_meta(bid).state,
      BlockState::Empty,
      "round {round}: victim 必须被擦除回收"
    );
  }
  // 回收复用同一批块 ⇒ 页分配水位不增长（无泄漏新块）
  assert_eq!(
    db.allocated_pi_ds_cnt().await,
    before,
    "多轮 GC 不得泄漏新块"
  );
}

/// `page_hit` 记账三类序列精确 +1（对齐上游 `BufferManager::FixShare`
/// `buffer_manager.cc:2060-2072`：`FixShareImpl` 返回 true（EVICTED 抢到排他）即
/// `HandlePageFault`（记 miss），否则 `buffer_hit_cnt++`；一次访问不得既 miss 又 hit）。
///
/// 序列：① `fix_share` 首次触达 EVICTED 页 → miss+1、hit 不变；② `fix_share` 常驻读
/// → hit+1；③ 另一 EVICTED 页 `fix_share` → miss+1、hit 不变。三类操作各恰好 +1。
///
/// 唯一击杀：删 `fix_share_impl` 的 `faulted` 门（缺页后仍记 hit）→ ① 后 hit==1
/// 而断言为 0 → FAIL；或删 `record_page_hit`（常驻读不记 hit）→ ② FAIL。
#[compio::test]
async fn page_hit_miss_exact_three_class_sequence() {
  let db = LeanStore::open(Config {
    frames: 64,
    ..Default::default()
  })
  .await
  .unwrap();
  let p0 = db.buf.alloc_page().await.unwrap();
  let p1 = db.buf.alloc_page().await.unwrap();
  let mut out = [0u8; PAGE_SIZE];

  // ① 缺页载入（EVICTED → 读盘）：miss+1，且不得同时记 hit
  db.buf.fix_share(p0, &mut out).await.unwrap();
  let m = db.stat().snapshot_metric();
  assert_eq!((m.page_miss, m.page_hit), (1, 0), "缺页只记 miss，不记 hit");

  // ② 常驻命中：hit+1，miss 不变
  db.buf.fix_share(p0, &mut out).await.unwrap();
  let m = db.stat().snapshot_metric();
  assert_eq!((m.page_miss, m.page_hit), (1, 1), "常驻读只记 hit");

  // ③ 另一 EVICTED 页缺页：再 miss+1，hit 不变
  db.buf.fix_share(p1, &mut out).await.unwrap();
  let m = db.stat().snapshot_metric();
  assert_eq!((m.page_miss, m.page_hit), (2, 1), "第二次缺页只记 miss");
}

/// GC 多轮不泄漏 —— 列表守恒版（独立于 `gc_multi_round_no_space_leak` 的
/// `allocated_pi_ds_cnt` 断言）：8 轮 FULL→GC→EMPTY 后 Empty/Open/InUse/Full/
/// GcRead/GcWrite 各列表长度守恒，victim 回到 Empty 列表且不在任何 GC 列表。
///
/// 唯一击杀：`erase_block` 的 `reset_block_space` 漏掉 `move_block_to_list(st,Empty)`
/// （只 `reset_block_metadata` 把 state 置 Empty）→ 块 state==Empty 但仍在 Full 列表
/// ⇒ Empty 列表少 1、Full 列表多 1 → 守恒断言 FAIL（`gc_multi_round_no_space_leak`
/// 只看 state 与 pid 水位，杀不掉这个变异）。
#[compio::test]
async fn gc_multi_round_lists_conserved() {
  let db = LeanStore::open(Config {
    frames: 1024,
    ..Default::default()
  })
  .await
  .unwrap();
  let hook = db.space.gc_hook.get().expect("GC hook 应已注入");
  let lists = |st: BlockState| db.space.get_block_list_size(st);
  let base = [
    (BlockState::Empty, lists(BlockState::Empty)),
    (BlockState::Open, lists(BlockState::Open)),
    (BlockState::InUse, lists(BlockState::InUse)),
    (BlockState::Full, lists(BlockState::Full)),
    (BlockState::GcRead, lists(BlockState::GcRead)),
    (BlockState::GcWrite, lists(BlockState::GcWrite)),
  ];
  let before_pids = db.allocated_pi_ds_cnt().await;
  for round in 0..8u64 {
    let bid = db.space.select_empty_block_id(0, 42 + round);
    assert!(bid != BlockId::MAX, "round {round}: 应取到空块");
    assert!(
      db.space
        .move_block_to_list(bid, BlockState::Open, BlockState::InUse)
    );
    assert!(db.space.close_one_open_block(bid, 64 * PAGE_SIZE as u64));
    {
      let mut s = db.space.sspace.write();
      let m = &mut s.block_metadata[bid as usize];
      m.total_pid_cnt = 8;
      m.invalidated_pid_cnt = 8;
    }
    assert_eq!(db.space.block_meta(bid).state, BlockState::Full);
    assert!(hook.on_gc_needed().await, "round {round}: GC 应推进回收");
    // victim 回到 Empty 列表（OPEN 或 EMPTY 均视为已释放；本 erase 路径为 EMPTY）
    assert!(
      db.space.is_block_in_list(bid, BlockState::Empty),
      "round {round}: victim 必须回到 Empty 列表"
    );
  }
  for (st, n) in base {
    assert_eq!(lists(st), n, "多轮 GC 后 {st:?} 列表长度必须守恒");
  }
  assert_eq!(
    db.allocated_pi_ds_cnt().await,
    before_pids,
    "PID 水位不得增长"
  );
}

/// 阶段 2 端到端持久化（门面真实装配）：经门面 `BufferManager` 的 EXCLUSIVE 就地
/// 视图（`with_page_mut`）写入一个 B+Tree 节点页 → `flush_all` → `drop_cache`
/// （全量驱逐）→ 缺页读回，断言键值逐字节一致；随后**重开引擎**（同路径新实例）
/// 再次读回校验同一键值。
///
/// 唯一击杀：删 `with_page_mut` 的 `make_page_dirty` → flush 无页可写 → 读回 count
/// 归零 FAIL；或门面文件后端写/读偏移错位 → 重开后键值不符 FAIL。
#[compio::test]
async fn stage2_facade_e2e_persist_and_reopen() {
  let dir = temp_dir().join(format!("ldb_stage2_e2e_{}", id()));
  let _ = remove_dir_all(&dir);
  create_dir_all(&dir).unwrap();
  let path = dir.to_string_lossy().into_owned();

  let pid;
  {
    let db = LeanStore::open(Config {
      path: path.clone(),
      frames: 64,
      ..Default::default()
    })
    .await
    .unwrap();
    pid = db.buf.alloc_page().await.unwrap();
    let mut node = BTreeNode::new(true, 0);
    for i in 0..6u32 {
      assert!(node.insert_key_value(format!("f{i:02}").as_bytes(), &[i as u8; 24], MEMCMP));
    }
    let bytes: &[u8; PAGE_SIZE] = unsafe {
      // SAFETY: BTreeNode 恰为 PAGE_SIZE 字节；只读借用 node，不越界。
      &*(node.ptr() as *const [u8; PAGE_SIZE])
    };
    // 门面真实缓冲池的 EXCLUSIVE 就地写
    db.buf
      .with_page_mut(pid, |page| page.copy_from_slice(bytes))
      .await
      .unwrap();
    // flush 落盘 → 全量驱逐 → 缺页读回
    db.buf.flush_all().await.unwrap();
    db.drop_cache().await;

    let mut back = BTreeNode::new(true, 0);
    let dst: &mut [u8; PAGE_SIZE] = unsafe {
      // SAFETY: 同 store 的 4096 对齐拥有式读回缓冲；&mut Box 独占。
      &mut *(back.as_mut() as *mut BTreeNode as *mut [u8; PAGE_SIZE])
    };
    db.buf.fix_share(pid, dst).await.unwrap();
    back.validate(MEMCMP).unwrap();
    assert_eq!(back.hdr.count, 6, "驱逐后缺页读回计数应一致");
    for i in 0..6u32 {
      let mut exact = false;
      let pos = back.lower_bound(format!("f{i:02}").as_bytes(), &mut exact, MEMCMP);
      assert!(exact, "驱逐读回后 f{i:02} 缺失");
      assert_eq!(back.get_payload(pos), &[i as u8; 24]);
    }
    db.shutdown().await;
  }

  // 重开引擎：同路径新实例（空 cache、新 space），盘面不变
  {
    let db = LeanStore::open(Config {
      path: path.clone(),
      frames: 64,
      ..Default::default()
    })
    .await
    .unwrap();
    let mut back = BTreeNode::new(true, 0);
    let dst: &mut [u8; PAGE_SIZE] = unsafe {
      // SAFETY: 同上。
      &mut *(back.as_mut() as *mut BTreeNode as *mut [u8; PAGE_SIZE])
    };
    db.buf.fix_share(pid, dst).await.unwrap();
    back.validate(MEMCMP).unwrap();
    assert_eq!(back.hdr.count, 6, "重开后节点计数应一致");
    for i in 0..6u32 {
      let mut exact = false;
      let pos = back.lower_bound(format!("f{i:02}").as_bytes(), &mut exact, MEMCMP);
      assert!(exact, "重开后 f{i:02} 缺失");
      assert_eq!(back.get_payload(pos), &[i as u8; 24], "重开后字节须一致");
    }
    db.shutdown().await;
  }
  let _ = remove_dir_all(&dir);
}

/// WAF 分母生产者：门面 `put` 经真实 WAL 数据日志追加「键‖值」用户负载并记
/// `record_logical_write`；配合一次脏页驱逐（物理写）⇒ `db_waf > 0`（status=ok，
/// 此前分母恒 0 ⇒ WAF 不可用）。
///
/// 唯一击杀：删 `LeanStore::put` 的 `record_logical_write` → 分母 0 → `db_waf>0`
/// 断言失败。
#[compio::test]
async fn waf_logical_write_producer_gives_positive_waf() {
  let mut db = LeanStore::open(Config {
    frames: 64,
    ..Default::default()
  })
  .await
  .unwrap();
  for i in 0..16u32 {
    db.put(format!("w{i:02}").as_bytes(), &[i as u8; 64])
      .await
      .unwrap();
  }
  // 生产者未破坏 KV 读路径
  let mut out = Vec::new();
  assert!(db.get(b"w03", &mut out).await.unwrap());
  assert_eq!(out, [3u8; 64]);

  // 物理写：制造一个脏页并驱逐（page_evict → physical_write）
  let pid = db.buf.alloc_page().await.unwrap();
  db.buf.with_page_mut(pid, |p| p[0] = 1).await.unwrap();
  assert!(db.buf.evict(4).await.unwrap() >= 1, "脏页应被驱逐");

  let m = db.stat().snapshot_metric();
  assert_eq!(m.txn_commit, 0, "本用例不走事务提交");
  assert!(m.page_evict >= 1, "应有脏页驱逐: {m:?}");
  assert!(m.db_waf > 0.0, "WAF 分母非 0 且分子>0 ⇒ 比值>0: {m:?}");
  db.shutdown().await;
}
