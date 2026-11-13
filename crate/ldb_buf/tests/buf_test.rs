//! VMCache 回归：缺页加载 / 驱逐 / OLC 乐观重试 / 分配释放（todo.md §4.1 防线 2 的
//! 单线程基线 —— 多协程压力版在 ldb_testkit）。

use std::{
  cell::Cell,
  io,
  rc::Rc,
  sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
  },
};

use compio::runtime::spawn;
use futures_lite::future::yield_now;
use ldb_buf::{BufferFrame, BufferManager, Error, PageBackend};
use ldb_free::FreeStorageManager;
use ldb_io::MemBackend;
use ldb_space::SpaceManager;
use ldb_stat::StatCollector;
use ldb_type::{GsnWatermark, PAGE_SIZE, PageId, PageStateMode};

/// 指定虚拟/物理帧数的通用构造
fn mk_sized<B: PageBackend>(
  virtual_cnt: u64,
  physical_frames: u64,
  backend: B,
) -> BufferManager<B> {
  let space = Rc::new(SpaceManager::new(
    virtual_cnt,
    64,
    2048,
    4,
    virtual_cnt * 64,
    2,
  ));
  let free = Arc::new(FreeStorageManager::new(true, 2));
  BufferManager::new(
    virtual_cnt,
    physical_frames,
    space,
    free,
    backend,
    Arc::new(AtomicBool::new(true)),
    Arc::new(GsnWatermark::new(0)),
  )
  .unwrap()
}

/// 虚拟 4x 物理帧（VMCache 超-commit 语义）
fn mk_backend<B: PageBackend>(pages: u64, backend: B) -> BufferManager<B> {
  mk_sized(pages * 4, pages, backend)
}

fn mk(pages: u64) -> BufferManager<MemBackend> {
  mk_backend(pages, MemBackend::new())
}

/// 测试用确定性随机源
fn next_rand(state: &mut u64) -> u64 {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  *state
}

/// 计数后端：记录页 I/O 次数与缺页读的落点指针
#[derive(Debug, Default)]
struct CountBackend {
  inner: MemBackend,
  reads: Cell<u64>,
  writes: Cell<u64>,
  last_read_dst: Cell<usize>,
}

impl PageBackend for CountBackend {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
    self.reads.set(self.reads.get() + 1);
    self.last_read_dst.set(dst.as_ptr() as usize);
    self.inner.read_page(pid, dst).await
  }

  async fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> io::Result<()> {
    self.writes.set(self.writes.get() + 1);
    self.inner.write_page(pid, src).await
  }

  async fn sync(&self) -> io::Result<()> {
    Ok(())
  }
}

/// 读路径在完成前让出：放大并发缺页的协程交错窗口
#[derive(Debug)]
struct YieldReadBackend {
  inner: MemBackend,
}

impl PageBackend for YieldReadBackend {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> io::Result<()> {
    yield_now().await;
    self.inner.read_page(pid, dst).await
  }

  async fn write_page(&self, pid: PageId, src: &[u8; PAGE_SIZE]) -> io::Result<()> {
    self.inner.write_page(pid, src).await
  }

  async fn sync(&self) -> io::Result<()> {
    Ok(())
  }
}

#[compio::test]
async fn alloc_fault_read_write_roundtrip() {
  let bm = mk(16);
  let pid = bm.alloc_page().await.unwrap();
  assert_eq!(bm.get_max_pid(), 1);

  // 缺页加载：EVICTED → EXCLUSIVE → 读盘（零页）→ UNLOCKED
  let mut page = [1u8; PAGE_SIZE];
  bm.fix_share(pid, &mut page).await.unwrap();
  assert!(page.iter().all(|&b| b == 0), "新页应清零");

  // 就地写
  let payload = vec![0xABu8; 128];
  assert!(bm.update_pid(pid, &payload).await.unwrap());
  let mut back = [0u8; PAGE_SIZE];
  bm.read_page(pid, &mut back).await.unwrap();
  assert_eq!(&back[..128], &payload[..], "写后读一致");
}

#[compio::test]
async fn evict_and_reload_persists() {
  let bm = mk(16);
  let pid = bm.alloc_page().await.unwrap();
  let payload = vec![0x5A; 64];
  bm.update_pid(pid, &payload).await.unwrap();

  // 驱逐（写盘）→ 再读（缺页加载）→ 数据持久
  let evicted = bm.evict(1).await.unwrap();
  let _ = evicted;
  let mut back = [0u8; PAGE_SIZE];
  bm.read_page(pid, &mut back).await.unwrap();
  assert_eq!(&back[..64], &payload[..], "驱逐后重读持久");
}

#[compio::test]
async fn olc_restart_on_concurrent_write() {
  let bm = mk(8);
  let pid = bm.alloc_page().await.unwrap();

  // 乐观快照 v0
  let mut page = [0u8; PAGE_SIZE];
  let v0 = bm.fix_share_impl(pid, &mut page).await.unwrap();

  // 写者插队：EXCLUSIVE 修改 + 解锁（版本 +1）
  bm.fix_exclusive(pid).await.unwrap();
  bm.unfix_exclusive(pid);

  // 旧快照校验失败 → Restart
  let r = bm.fix_share_cached_page(pid, &mut page, v0).await;
  assert!(
    matches!(r, Err(Error::Restart)),
    "版本漂移必须 Restart: {r:?}"
  );
}

#[compio::test]
async fn memory_pressure_triggers_eviction() {
  // 98% 水位的「提前」语义必须可区分：50 帧池载入 49 页（49*100 >= 50*98），
  // ensure_free_pages 必须驱逐；水位改成 100%（9800→10000 口径）则不触发。
  let bm = mk(50);
  for _ in 0..49 {
    let pid = bm.alloc_page().await.unwrap();
    let mut sink = [0u8; PAGE_SIZE];
    bm.fix_share(pid, &mut sink).await.unwrap();
  }
  assert_eq!(bm.physical_used.load(Ordering::Relaxed), 49);
  bm.ensure_free_pages().await.unwrap();
  assert_eq!(
    bm.physical_used.load(Ordering::Relaxed),
    48,
    "98% 水位必须在未满时提前驱逐（49/50 即触发）"
  );

  // 池 4 帧：占满 4 页后继续缺页 → 水位触发驱逐
  let bm = mk(4);
  let mut pids = Vec::new();
  for _ in 0..4 {
    let pid = bm.alloc_page().await.unwrap();
    let mut sink = [0u8; PAGE_SIZE];
    bm.fix_share(pid, &mut sink).await.unwrap();
    pids.push(pid);
  }
  assert_eq!(bm.physical_used.load(Ordering::Relaxed), 4);
  // 98% 水位（4 * 98% = 3.92）：ensure_free_pages 必须先驱逐一页再继续
  bm.ensure_free_pages().await.unwrap();
  assert_eq!(
    bm.physical_used.load(Ordering::Relaxed),
    3,
    "水位线必须提前驱逐"
  );
  // 第 5 页缺页 → 驱逐一旧页后再载入（超-commit 守恒：总量 = 物理帧数）
  let pid5 = bm.alloc_page().await.unwrap();
  let mut sink = [0u8; PAGE_SIZE];
  bm.fix_share(pid5, &mut sink).await.unwrap();
  let used = bm.physical_used.load(Ordering::Relaxed);
  assert!(used <= 4, "物理帧守恒: used={used}");
  // pid5 必然常驻（Unlocked），且至少一页旧页被逐（Evicted）
  assert_eq!(
    bm.get_page_state(pid5).lock_state(),
    PageStateMode::Unlocked
  );
  let evicted_cnt = (0..5u64)
    .filter(|&p| p != pid5 && bm.get_page_state(p).lock_state() == PageStateMode::Evicted)
    .count();
  assert!(evicted_cnt >= 1, "应有旧页被驱逐");
}

#[test]
fn frame_flags() {
  let mut f = BufferFrame::init(7);
  assert_eq!(f.pid, 7);
  assert!(!f.is_extent());
  let (pid, pages) = f.to_write_extent(4);
  assert_eq!((pid, pages), (7, 4));
  assert!(f.is_extent() && f.to_write);
  f.set_flag(true, false);
  assert!(f.dirty && !f.to_write);
  f.reset(9);
  assert_eq!(f.pid, 9);
  assert!(!f.is_extent());
}

#[compio::test]
async fn out_of_place_write_updates_mapping() {
  let bm = mk(16);
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[1u8; 32]).await.unwrap();
  // 异地写到块 1（非零偏移，默认值 0 无法冒充）→ 映射表更新
  bm.write_pages_out_of_place(&[pid], 1).await.unwrap();
  let off = bm.space.get_pid2_offset(pid).get_offset();
  assert_eq!(off, 64 * PAGE_SIZE as u64, "块 1 起始偏移");
  assert_eq!(bm.space.get_pid_comp_size(pid), ldb_pack::UNCOMPRESSED);
}

/// 异地写必须落到目标块偏移：`PageBackend` 的页号参数是文件平坦页索引。
/// 变异「仍写 write_page(pid)」会让目标块保持零、旧位置被新数据覆盖。
#[compio::test]
async fn out_of_place_write_lands_at_target_block() {
  let bm = mk_backend(16, CountBackend::default());
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0xC3u8; 64]).await.unwrap();
  bm.write_pages_out_of_place(&[pid], 1).await.unwrap();

  // 目标块 1 的平坦页索引 = block_size / PAGE_SIZE = 64（字面量，不读映射表：
  // 本条只钉「数据落到哪」，映射更新由 out_of_place_write_updates_mapping 钉）
  let target_page = 64u64;
  let mut target = [0u8; PAGE_SIZE];
  bm.backend
    .inner
    .read_at_page(target_page, &mut target)
    .await
    .unwrap();
  assert_eq!(&target[..64], &[0xC3u8; 64], "目标块必须收到页内容");

  let mut old = [0u8; PAGE_SIZE];
  bm.backend.inner.read_at_page(pid, &mut old).await.unwrap();
  assert!(old.iter().all(|&b| b == 0), "旧 in-place 位置不得被写入");
}

/// GC 搬移写目标块：`flush_valid_pi_ds_to_block` 同样按目标偏移落盘。
/// 变异「写 write_page(pid)」使目标块为空白，本条唯一击杀。
#[compio::test]
async fn gc_flush_writes_to_destination_block() {
  let bm = mk_backend(16, CountBackend::default());
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x5Cu8; 64]).await.unwrap();
  let written = bm.flush_valid_pi_ds_to_block(&[pid], 2).await.unwrap();
  assert_eq!(written, 1);

  let off = bm.space.get_pid2_offset(pid).get_offset();
  assert_eq!(off, 2 * 64 * PAGE_SIZE as u64, "块 2 起始偏移");
  let mut target = [0u8; PAGE_SIZE];
  bm.backend
    .inner
    .read_at_page(off / PAGE_SIZE as u64, &mut target)
    .await
    .unwrap();
  assert_eq!(&target[..64], &[0x5Cu8; 64], "目标块必须收到 GC 搬移页");
}

#[compio::test]
async fn fuzzy_checkpoint_advances_watermark() {
  // C-17 修复验证：min_flushed 单调，不被覆盖回退
  let bm = mk(8);
  bm.update_wal_info(100);
  let cp = bm.fuzzy_checkpoint(50).await.unwrap();
  assert_eq!(cp, 100, "回退目标不得拉低水位");
  let cp2 = bm.fuzzy_checkpoint(150).await.unwrap();
  assert_eq!(cp2, 150, "新目标推进水位");
}

/// 缺页读必须直接落入 arena 页帧：后端观察到的 dst 指针 == arena 页地址。
/// 变异「读入栈缓冲再拷贝」会把 dst 指到栈上，本条唯一击杀。
#[compio::test]
async fn fault_reads_directly_into_arena_frame() {
  let bm = mk_backend(4, CountBackend::default());
  let pid = bm.alloc_page().await.unwrap();
  let mut out = [0u8; PAGE_SIZE];
  bm.read_page(pid, &mut out).await.unwrap();
  assert_eq!(bm.backend.reads.get(), 1, "缺页恰好一次磁盘读");
  assert_eq!(
    bm.backend.last_read_dst.get(),
    bm.page_addr(pid),
    "读缓冲必须是 arena 页帧（不得经栈中转）"
  );
}

/// 干净页驱逐不得写盘（上游 Evict：clean 候选直接 madvise）。
/// 变异「驱逐无条件写回」使 writes 从 1 变 2，本条唯一击杀。
#[compio::test]
async fn evict_clean_page_skips_writeback() {
  let bm = mk_backend(4, CountBackend::default());
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x7Au8; 64]).await.unwrap();
  bm.flush_all().await.unwrap();
  assert_eq!(bm.backend.writes.get(), 1, "首次刷脏一次写");
  assert!(!bm.page_is_dirty(pid), "刷盘后已干净");
  assert_eq!(bm.evict(1).await.unwrap(), 1);
  assert_eq!(bm.backend.writes.get(), 1, "干净页驱逐不得产生页写");
  let mut back = [0u8; PAGE_SIZE];
  bm.read_page(pid, &mut back).await.unwrap();
  assert_eq!(&back[..64], &[0x7Au8; 64], "驱逐后内容仍可从盘读回");
}

/// 释放页直接丢弃：脏页 dealloc 不得写回。
/// 变异「evict_locked 对 dealloc 仍写盘」使 writes=1，本条唯一击杀。
#[compio::test]
async fn dealloc_page_discards_without_writeback() {
  let bm = mk_backend(4, CountBackend::default());
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x33u8; 64]).await.unwrap();
  assert_eq!(bm.backend.writes.get(), 0, "尚未刷盘");
  bm.dealloc_page(pid).await.unwrap();
  assert_eq!(bm.backend.writes.get(), 0, "释放页不得写回");
  assert_eq!(bm.get_page_state(pid).lock_state(), PageStateMode::Evicted);
}

/// dealloc 列表必须被 alloc 路径消费：复用 PID 且 max_pid 不增长。
/// 变异「alloc 只 take 不 process_dealloc_list」→ 返回新 pid，本条唯一击杀。
#[compio::test]
async fn alloc_reuses_deallocated_pid() {
  let bm = mk(4);
  let p0 = bm.alloc_page().await.unwrap();
  bm.update_pid(p0, &[1u8; 8]).await.unwrap();
  let p1 = bm.alloc_page().await.unwrap();
  assert_eq!((p0, p1), (0, 1));
  bm.dealloc_page(p0).await.unwrap();
  assert_eq!(bm.get_max_pid(), 2);
  let p2 = bm.alloc_page().await.unwrap();
  assert_eq!(p2, p0, "必须复用已释放 PID");
  assert_eq!(bm.get_max_pid(), 2, "复用不得推进 max_pid");
}

/// dealloc 后 stale pid 不得被缺页读回旧盘面内容。
/// 变异「handle_page_fault 删掉 is_deallocated 检查」→ 返回 Ok（旧内容）。
#[compio::test]
async fn dealloc_then_fault_is_rejected() {
  let bm = mk(4);
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x66u8; 64]).await.unwrap();
  bm.dealloc_page(pid).await.unwrap();
  let mut out = [0u8; PAGE_SIZE];
  assert!(
    matches!(bm.fix_share(pid, &mut out).await, Err(Error::InvalidPage(p)) if p == pid),
    "已释放页必须拒绝缺页读回"
  );
}

/// 重复释放必须幂等：不得把同一 pid 两次放进复用池（否则两次 alloc 返回
/// 同一 pid，两个逻辑页别名）。
/// 变异「dealloc_page 无条件 add_pid_to_dealloc_list」→ 第二次 alloc 仍是 p0。
#[compio::test]
async fn double_dealloc_does_not_duplicate_reuse_pool() {
  let bm = mk(4);
  let p0 = bm.alloc_page().await.unwrap();
  bm.update_pid(p0, &[0x77u8; 64]).await.unwrap();
  bm.dealloc_page(p0).await.unwrap();
  bm.dealloc_page(p0).await.unwrap(); // 幂等：不得二次入池
  assert_eq!(bm.physical_used.load(Ordering::Relaxed), 0);
  let a = bm.alloc_page().await.unwrap();
  let b = bm.alloc_page().await.unwrap();
  let c = bm.alloc_page().await.unwrap();
  assert_eq!((a, b, c), (p0, 1, 2), "复用池只能含 p0 一次");
  assert_eq!(bm.get_max_pid(), 3);
}

/// 释放后经复用池重新分配：已释放标记必须清除，新化身可正常缺页加载。
/// 变异「claim_reusable_pid 删掉 clear_deallocated」→ 新页缺页 InvalidPage。
#[compio::test]
async fn reused_pid_can_be_loaded_again() {
  let bm = mk(4);
  let p0 = bm.alloc_page().await.unwrap();
  bm.update_pid(p0, &[0x88u8; 64]).await.unwrap();
  bm.dealloc_page(p0).await.unwrap();
  let again = bm.alloc_page().await.unwrap();
  assert_eq!(again, p0);
  let mut out = [1u8; PAGE_SIZE];
  bm.fix_share(again, &mut out).await.unwrap();
  assert_eq!(
    bm.get_page_state(again).lock_state(),
    PageStateMode::Unlocked,
    "复用后的新化身必须能缺页加载"
  );
}

/// extent 复用区间校验：未对齐/越界的请求必须返回 false，不得写出映射。
/// 变异「try_reuse_extent 去掉区间校验」→ 负例断言失败。
#[compio::test]
async fn try_reuse_extent_validates_range() {
  let bm = mk(4); // 虚拟 16 页
  assert!(
    !bm
      .try_reuse_extent(0, 1, PAGE_SIZE as u64 / 2)
      .await
      .unwrap(),
    "未按页对齐的偏移必须拒绝"
  );
  assert!(
    !bm.try_reuse_extent(15, 4, 0).await.unwrap(),
    "pid 区间越界必须拒绝"
  );
  assert!(bm.try_reuse_extent(0, 1, 0).await.unwrap());
}

/// 驱逐未加载的 extent 不得为了「驱逐」把它读进来再做写回。
/// 变异「evict_extent 走 fix_exclusive」→ reads=4，本条唯一击杀。
#[compio::test]
async fn evict_extent_skips_nonresident_pages() {
  let bm = mk_backend(16, CountBackend::default());
  let start = bm.alloc_extent(4).await.unwrap();
  assert_eq!(start, 0);
  bm.evict_extent(start, 4).await.unwrap();
  assert_eq!(bm.backend.reads.get(), 0, "未加载 extent 驱逐不得缺页读");
  assert_eq!(bm.backend.writes.get(), 0, "未加载 extent 驱逐不得写盘");
}

/// 复用已释放空间时若目标 pid 仍常驻（外部误用 / 尚未驱逐），必须先安全
/// 丢弃再复位：直接 `init()` 会把状态改成 EVICTED 而 resident 表与
/// `physical_used` 不变，后续缺页因 `resident.insert` 恒失败而永久
/// MemoryPressure；若此时 ShareGuard 存活，还会与重新缺页的写入构成别名 UB。
/// 变异「reuse_deallocated_space 删掉 evict_dealloc_page」→ resident/used 断言失败。
#[compio::test]
async fn reuse_evicts_resident_page_and_keeps_accounting() {
  let bm = mk(8);
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x44u8; 64]).await.unwrap();
  assert_eq!(bm.physical_used.load(Ordering::Relaxed), 1);
  assert!(bm.resident.contain(pid));

  let off = 64 * PAGE_SIZE as u64;
  assert!(bm.try_reuse_extent(pid, 1, off).await.unwrap());
  assert_eq!(bm.get_page_state(pid).lock_state(), PageStateMode::Evicted);
  assert!(!bm.resident.contain(pid), "复用必须从常驻表摘除");
  assert_eq!(
    bm.physical_used.load(Ordering::Relaxed),
    0,
    "复用必须归还物理帧计数"
  );
  assert_eq!(bm.space.get_pid2_offset(pid).get_offset(), off);
  assert_eq!(bm.space.get_pid_comp_size(pid), ldb_pack::UNCOMPRESSED);

  // 复用后的 pid 必须能重新缺页加载（旧实现 resident.insert=false →
  // MemoryPressure 永久钉死）
  let mut out = [0u8; PAGE_SIZE];
  bm.fix_share(pid, &mut out).await.unwrap();
  assert_eq!(bm.get_page_state(pid).lock_state(), PageStateMode::Unlocked);
  assert_eq!(bm.physical_used.load(Ordering::Relaxed), 1);
}

/// 复用池被外部塞入常驻 pid 时，`alloc_page` 必须先驱逐再发号，不得直接
/// `init()` 破坏 resident/`physical_used` 不变量。
/// 变异「alloc_page 复用分支删掉 claim_reusable_pid 的驱逐」→ 断言失败。
#[compio::test]
async fn alloc_reclaims_resident_pid_from_reusable_pool() {
  let bm = mk(8);
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x55u8; 64]).await.unwrap();
  assert_eq!(bm.physical_used.load(Ordering::Relaxed), 1);
  // 模拟外部误用（`space` 字段与 `add_reusable_pid` 均为 pub）
  bm.space.add_reusable_pid(pid);
  let got = bm.alloc_page().await.unwrap();
  assert_eq!(got, pid, "复用池优先");
  assert_eq!(bm.get_page_state(pid).lock_state(), PageStateMode::Evicted);
  assert!(!bm.resident.contain(pid));
  assert_eq!(bm.physical_used.load(Ordering::Relaxed), 0);
}

/// 并发缺页的物理帧硬上限：预留发生在读盘前，`physical_used` 的检查与
/// 自增是同一个 CAS，任何交错下都不得超过 `physical_cnt`。
///
/// 变异「读完再 fetch_add」让 8 个协程同时通过水位检查后各自 +1 ⇒ used=8 > 2。
#[compio::test]
async fn concurrent_faults_respect_physical_cap() {
  let bm = Rc::new(mk_sized(
    8,
    2,
    Rc::new(YieldReadBackend {
      inner: MemBackend::new(),
    }),
  ));
  let mut pids = Vec::new();
  for _ in 0..8 {
    pids.push(bm.alloc_page().await.unwrap());
  }
  let mut handles = Vec::new();
  for &pid in &pids {
    let bm = bm.clone();
    handles.push(spawn(async move {
      let mut out = [0u8; PAGE_SIZE];
      bm.fix_share(pid, &mut out).await
    }));
  }
  let mut loaded = 0u32;
  let mut pressured = 0u32;
  for h in handles {
    match h.await {
      Ok(Ok(_)) => loaded += 1,
      Ok(Err(Error::MemoryPressure)) => pressured += 1,
      Ok(Err(e)) => panic!("非预期错误: {e:?}"),
      Err(e) => panic!("任务 join 失败: {e:?}"),
    }
  }
  // 物理帧只有 2：前两个协程各预留一帧后读盘让出，其余全部内存压力
  assert_eq!(loaded, 2, "物理帧上限内只能加载 2 页");
  assert_eq!(loaded + pressured, 8, "只允许成功或内存压力");
  assert!(
    bm.physical_used.load(Ordering::Relaxed) <= 2,
    "physical_used 不得超过物理帧上限: {}",
    bm.physical_used.load(Ordering::Relaxed)
  );
  // 状态机与常驻表对拍：非 EVICTED 页数 == physical_used
  let resident = pids
    .iter()
    .filter(|&&pid| bm.get_page_state(pid).lock_state() != PageStateMode::Evicted)
    .count();
  assert_eq!(
    resident as u64,
    bm.physical_used.load(Ordering::Relaxed),
    "常驻状态数与物理帧计数必须一致"
  );

  // 第二阶段：并发 fix / write / evict 压测（liveness + 内容一致性）。
  // 2 读者 + 2 写者 + 2 驱逐者交错 500 轮；页内容只能是某个写者种子的
  // 整段副本（前 64B 全等），撕裂/丢失都会被抓到。
  let mut stress = Vec::new();
  for t in 0..2u64 {
    let bm = bm.clone();
    let pids = pids.clone();
    stress.push(spawn(async move {
      let mut rng = 0x1234_5678_9ABC_DEF0u64 ^ (t + 1);
      for _ in 0..500 {
        let pid = pids[(next_rand(&mut rng) as usize) % pids.len()];
        let mut out = [0u8; PAGE_SIZE];
        if bm.fix_share(pid, &mut out).await.is_ok() {
          assert!(out[..64].iter().all(|&b| b == out[0]), "并发读到了撕裂页");
        }
      }
    }));
  }
  for t in 0..2u64 {
    let bm = bm.clone();
    let pids = pids.clone();
    stress.push(spawn(async move {
      let mut rng = 0xDEAD_BEEF_CAFE_0000u64 ^ (t + 1);
      for _ in 0..500 {
        let pid = pids[(next_rand(&mut rng) as usize) % pids.len()];
        let seed = (next_rand(&mut rng) as u8) | 1;
        let _ = bm.update_pid(pid, &[seed; 64]).await;
      }
    }));
  }
  for t in 0..2u64 {
    let bm = bm.clone();
    stress.push(spawn(async move {
      let mut rng = 0xA5A5_5A5A_1234_0000u64 ^ (t + 1);
      for _ in 0..500 {
        let n = next_rand(&mut rng) % 3 + 1;
        let _ = bm.evict(n).await;
      }
    }));
  }
  for h in stress {
    match h.await {
      Ok(()) => {}
      Err(e) => panic!("压测任务 join 失败: {e:?}"),
    }
  }
  bm.flush_all().await.unwrap();
  for &pid in &pids {
    let mut out = [0u8; PAGE_SIZE];
    bm.fix_share(pid, &mut out).await.unwrap();
    assert!(out[..64].iter().all(|&b| b == out[0]), "压测后页内容不一致");
  }
  let resident = pids
    .iter()
    .filter(|&&pid| bm.get_page_state(pid).lock_state() != PageStateMode::Evicted)
    .count();
  assert_eq!(
    resident as u64,
    bm.physical_used.load(Ordering::Relaxed),
    "压测后常驻状态数与物理帧计数必须一致"
  );
}

/// Windows：arena 初始只 `MEM_RESERVE`，缺页读入前必须 `MEM_COMMIT`。
/// 原版直接写 reserved 槽位 → 访问违例（无此断言时测试进程直接崩溃）。
#[cfg(windows)]
#[compio::test]
async fn fault_commits_reserved_slot_before_write() {
  let bm = mk(4);
  let pid = bm.alloc_page().await.unwrap();
  let mut page = [1u8; PAGE_SIZE];
  bm.fix_share(pid, &mut page).await.unwrap();
  assert!(page.iter().all(|&b| b == 0), "新页读零");
  bm.update_pid(pid, &[0x5Au8; 32]).await.unwrap();
  let mut back = [0u8; PAGE_SIZE];
  bm.read_page(pid, &mut back).await.unwrap();
  assert_eq!(&back[..32], &[0x5Au8; 32], "写后读一致");
}

/// 账本命名的 API 面必须真实接线（不是空壳）：逐个调用并校验可观察行为。
/// 变异「prepare_tail_extent 返回 start_pid+1 / write_pages_in_place 空转 /
/// read_extents 只读首页」都会被本条唯一击杀。
#[compio::test]
async fn ledger_named_api_surface_is_wired() {
  let bm = mk_backend(16, CountBackend::default());

  // 水位/空闲管理器视图
  bm.update_wal_info(42);
  let cp = bm.fuzzy_checkpoint(42).await.unwrap();
  assert_eq!(cp, 42);
  assert_eq!(bm.get_wal_info(), 42, "检查点偏移视图");
  assert!(Arc::ptr_eq(bm.get_free_page_manager(), &bm.free_mgr));
  assert!(Arc::ptr_eq(bm.free_storage_manager(), &bm.free_mgr));

  // 标脏 / 标净 / GC 缺页标脏
  let p = bm.alloc_page().await.unwrap();
  bm.update_pid(p, &[0x11u8; 8]).await.unwrap();
  assert!(bm.page_is_dirty(p));
  bm.make_page_clean(p);
  assert!(!bm.page_is_dirty(p));
  let pgc = bm.alloc_page().await.unwrap();
  bm.handle_page_fault_gc(pgc).await.unwrap();
  assert!(bm.page_is_dirty(pgc), "GC 读入必须标脏");

  // 复用已释放空间（区间校验通过）
  assert!(bm.try_reuse_extent(0, 1, 0).await.unwrap());
  assert_eq!(bm.get_page_state(0).lock_state(), PageStateMode::Evicted);

  // extent 批量读：两页内容分别落到目标缓冲的正确位置
  let start = bm.alloc_extent(2).await.unwrap();
  bm.update_pid(start, &[0xA1u8; 64]).await.unwrap();
  bm.update_pid(start + 1, &[0xB2u8; 64]).await.unwrap();
  let mut ext = vec![0u8; 2 * PAGE_SIZE];
  bm.read_extents(start, 2, &mut ext).await.unwrap();
  assert_eq!(&ext[..64], &[0xA1u8; 64]);
  assert_eq!(&ext[PAGE_SIZE..PAGE_SIZE + 64], &[0xB2u8; 64]);

  // modify_page_content / write_pages_in_place
  let p3 = bm.alloc_page().await.unwrap();
  bm.modify_page_content(p3, &[0xCCu8; 32]).await.unwrap();
  let mut out = [0u8; PAGE_SIZE];
  bm.read_page(p3, &mut out).await.unwrap();
  assert_eq!(&out[..32], &[0xCCu8; 32]);
  bm.update_pid(p3, &[0xDDu8; 32]).await.unwrap();
  assert!(bm.page_is_dirty(p3));
  let before = bm.backend.writes.get();
  bm.write_pages_in_place(&[p3]).await.unwrap();
  assert_eq!(bm.backend.writes.get(), before + 1, "就地写必须落盘");
  assert!(!bm.page_is_dirty(p3), "就地写盘后必须清脏");

  // GC 反向表查询（先登记 offset2pids）
  bm.space
    .update_pid2_offset(p3, 3 * 64 * PAGE_SIZE as u64, ldb_pack::UNCOMPRESSED)
    .await;
  bm.space.update_offset2_pids(3, 0, p3);
  let _ = bm.gc_valid_pid_cnt_in_block_on_disk(3).await.unwrap();

  // extent 辅助：分配 / 驱逐预备 / 尾 extent / 页数
  let env_pid = bm.prepare_extent_env(2).await.unwrap();
  assert!(env_pid < bm.get_max_pid());
  bm.prepare_extent_eviction(env_pid, 2).await.unwrap();
  assert_eq!(bm.prepare_tail_extent(env_pid).await.unwrap(), env_pid);
  assert_eq!(bm.extent_page_count(env_pid), 0, "本层不维护 extent 元数据");

  // GC 固定族
  bm.fix_exclusive_on_disk_gc(p3).await.unwrap();
  assert_eq!(bm.get_page_state(p3).lock_state(), PageStateMode::Exclusive);
  bm.unfix_exclusive(p3);
  let mut gc_out = [0u8; PAGE_SIZE];
  assert!(bm.fix_share_impl_gc(p3, &mut gc_out).await.unwrap() >= 1);
  assert_eq!(&gc_out[..32], &[0xDDu8; 32]);

  // 驱逐 task 在停机标记下立即返回（不阻塞事件循环）
  bm.is_running.store(false, Ordering::Relaxed);
  bm.run_page_provider_threads().await.unwrap();
  bm.is_running.store(true, Ordering::Relaxed);

  // 分块操作：5 页、块大小 2 → 2+2+1
  let counter = Rc::new(Cell::new(0u32));
  let c = counter.clone();
  bm.chunk_operation(
    async move |_pid, n| {
      c.set(c.get() + n);
      Ok(())
    },
    0,
    5,
    2,
  )
  .await
  .unwrap();
  assert_eq!(counter.get(), 5, "分块必须覆盖全部页");
}

/// 缓存命中路径的乐观快照：驱逐导致版本漂移必须 Restart（不得返回旧副本）。
#[compio::test]
async fn cached_page_restarts_after_eviction() {
  let bm = mk(4);
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x11u8; 32]).await.unwrap();
  let mut page = [0u8; PAGE_SIZE];
  let _ = bm.fix_share_impl(pid, &mut page).await.unwrap();
  assert_eq!(bm.evict(1).await.unwrap(), 1);
  // 用驱逐后的**当前**版本调用：版本层已无法区分，必须靠 EVICTED 状态拒绝
  // （变异「去掉 Evicted 分支/把 EVICTED 当可读」会让本条返回 Ok）。
  let current = bm.get_page_state(pid).version();
  let mut stale = [0u8; PAGE_SIZE];
  let r = bm.fix_share_cached_page(pid, &mut stale, current).await;
  assert!(
    matches!(r, Err(Error::Restart)),
    "页已驱逐，当前版本也不能返回快照: {r:?}"
  );
}

/// `ldb_stat` 接线：命中/缺页按「本次取页是否触发缺页」唯一划分。
///
/// 唯一击杀：删 `fix_shared` 常驻臂的 `record_page_hit`（二次固定不再 +1）或
/// 删 `handle_page_fault` 的 `record_page_miss`（首访不记 miss）都会让本用例 FAIL。
#[compio::test]
async fn stats_record_hit_and_miss() {
  let mut bm = mk(4);
  let stats = Arc::new(StatCollector::default());
  bm.set_stats(stats.clone());
  let pid = bm.alloc_page().await.unwrap();
  // 首次固定：EVICTED → 缺页加载（记 miss）；同一次访问不得再记 hit
  {
    let g = bm.fix_shared(pid).await.unwrap();
    drop(g);
  }
  assert_eq!(
    stats.snapshot_metric().page_miss,
    1,
    "首次缺页应恰记一次 miss"
  );
  assert_eq!(
    stats.snapshot_metric().page_hit,
    0,
    "缺页那次不得同时记 hit"
  );
  // 二次固定：页已常驻 → 记 hit
  {
    let g = bm.fix_shared(pid).await.unwrap();
    drop(g);
  }
  assert_eq!(stats.snapshot_metric().page_hit, 1, "常驻页固定应记 hit");
  assert_eq!(stats.snapshot_metric().page_miss, 1, "命中不得追加 miss");
}

/// `ldb_stat` 接线：脏页驱逐落盘记一次 `page_evict`（干净页驱逐不记）。
///
/// 唯一击杀：删 `evict_marked` 的 `record_page_evict` 会让第一段断言失败；
/// 把它挪到「干净页也记」会让第二段断言失败。
#[compio::test]
async fn stats_record_page_evict_only_on_dirty_writeback() {
  let mut bm = mk(4);
  let stats = Arc::new(StatCollector::default());
  bm.set_stats(stats.clone());
  let pid = bm.alloc_page().await.unwrap();
  bm.update_pid(pid, &[0x33u8; 16]).await.unwrap();
  assert_eq!(stats.snapshot_metric().page_evict, 0, "驱逐前无页写");
  assert_eq!(bm.evict(1).await.unwrap(), 1, "脏页应被驱逐");
  assert_eq!(stats.snapshot_metric().page_evict, 1, "脏页驱逐记一次页写");
  // 已干净页：再次加载后驱逐不产生页写，计数不变
  let g = bm.fix_shared(pid).await.unwrap();
  drop(g);
  assert_eq!(bm.evict(1).await.unwrap(), 1, "干净页可被驱逐");
  assert_eq!(
    stats.snapshot_metric().page_evict,
    1,
    "干净页驱逐不得记页写"
  );
}

/// WAL-before-page 门（上游 `buffer_manager.cc:1550-1552`，检查点 `FlushDirtyPages`）：
/// 页 `p_gsn` 未落 WAL 时检查点不得刷盘；水位追平后才允许写。
///
/// 唯一击杀：删除 `write_dirty_pages_upto` 的 `read_p_gsn(g.bytes()) > max_p_gsn`
/// 分支（未刷 WAL 也写盘）→ 首段 `writes == 0` 断言失败（实际会写成 1）。
#[compio::test]
async fn wal_before_page_gate_defers_unflushed_write() {
  let bm = mk_backend(4, CountBackend::default());
  let pid = bm.alloc_page().await.unwrap();
  // 构造 payload：p_gsn（页头偏移 8）置 100，其余保持 0
  let mut payload = [0u8; 16];
  payload[8..16].copy_from_slice(&100u64.to_ne_bytes());
  bm.update_pid(pid, &payload).await.unwrap();
  let mut out = [0u8; PAGE_SIZE];
  bm.fix_share_impl(pid, &mut out).await.unwrap();
  let mut gsn = [0u8; 8];
  gsn.copy_from_slice(&out[8..16]);
  assert_eq!(u64::from_ne_bytes(gsn), 100, "p_gsn 写入生效");
  // 已刷 WAL 水位 = 0 < p_gsn=100 → 检查点门挡下，不写盘且保持脏
  let _ = bm.fuzzy_checkpoint(0).await.unwrap();
  assert_eq!(bm.backend.writes.get(), 0, "WAL 未刷的页不得刷盘");
  assert!(bm.page_is_dirty(pid), "被门挡下的页应保持脏");
  // 水位推进到 100 → 允许刷盘
  bm.min_gsn_flushed.advance_min(100);
  let _ = bm.fuzzy_checkpoint(0).await.unwrap();
  assert_eq!(bm.backend.writes.get(), 1, "水位追平后应刷盘一次");
  assert!(!bm.page_is_dirty(pid), "刷盘后页应干净");
}

/// `with_page_mut`（阶段 2 唯一新增接口）：闭包在 EXCLUSIVE 闩内拿到可变页字节，
/// 退出后**标脏并解锁**；页内容就地落页可读回。
///
/// 唯一击杀：删 `with_page_mut` 的 `make_page_dirty` → 末段 `page_is_dirty` 断言
/// 失败；或闭包不经 EXCLUSIVE（直接改帧）→ `lock_state == Exclusive` 断言失败。
#[compio::test]
async fn with_page_mut_exclusive_view_dirties_and_unlocks() {
  let bm = mk_backend(4, CountBackend::default());
  let pid = bm.alloc_page().await.unwrap();
  assert!(!bm.page_is_dirty(pid), "新分配页初始不脏");

  let observed = Cell::new(None);
  bm.with_page_mut(pid, |page| {
    // 闭包执行期间本页为 EXCLUSIVE（其它 fix 无法进入该页）
    observed.set(Some(bm.get_page_state(pid).lock_state()));
    page[0] = 0xAB;
    page[PAGE_SIZE - 1] = 0xCD;
  })
  .await
  .unwrap();

  assert_eq!(
    observed.get(),
    Some(PageStateMode::Exclusive),
    "闭包内页必须为 EXCLUSIVE"
  );
  assert!(bm.page_is_dirty(pid), "退出必须标脏");
  assert_eq!(
    bm.get_page_state(pid).lock_state(),
    PageStateMode::Unlocked,
    "退出必须解锁（版本 +1 → UNLOCKED）"
  );

  let mut out = [0u8; PAGE_SIZE];
  bm.fix_share_impl(pid, &mut out).await.unwrap();
  assert_eq!((out[0], out[PAGE_SIZE - 1]), (0xAB, 0xCD), "就地修改应落页");
}
