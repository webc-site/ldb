//! 顶层 Facade（上游 `leanstore.cc` LeanStore + config）。
//!
//! 端到端装配（M6.3，2026-11-11）：`TransactionManager`(LogManager) →
//! `SpaceManager<WalTxnLogHook>` → `BufferManager`（共享 GSN 水位 D5）→
//! `GarbageCollector<WalTxnLogHook>` + `BufEvictHook`（D4 生产实现注入）；
//! 后台任务（组提交 5 阶段 / GC 循环 / 模糊检查点 / page-provider）以
//! compio spawn task 承载（上游 `StartXxxThread` 的 Thread-per-Core 化）。

use std::{
  cmp, fmt,
  future::Future,
  io,
  pin::Pin,
  rc::Rc,
  sync::{Arc, OnceLock, atomic},
  time::Duration,
};

use coarsetime::Instant;
use compio::{runtime::Runtime, time::sleep};
use ldb_blob::{BlobManager, BlobPageStore, Error as LdbBlobError};
use ldb_btree::BTree;
use ldb_buf::{BufEvictHook, BufferManager, PageBackend};
use ldb_free::FreeStorageManager;
use ldb_gc::GarbageCollector;
use ldb_io::{CompioFileBackend, MemBackend};
use ldb_node::MEMCMP;
use ldb_page::hook::GcHook;
use ldb_space::SpaceManager;
use ldb_stat::StatCollector;
use ldb_txn::{ConcurrencyControl, IsolationLevel, TransactionManager, WalTxnLogHook};
use ldb_type::{BlockId, PAGE_SIZE, PageId};
use ldb_vmem::VirtualMemoryArena;
use ldb_wal::{
  CommitVariant, EntryType, GcCfg, GroupCommitExecutor, LogManager, WalFileState, WalSink,
};
use parking_lot::Mutex as PLMutex;

use crate::error::{Error, Result};

/// 内存模式路径哨兵（`Config::path == ":memory:"` → 纯内存后端，测试 / Fuzz）
pub const MEM_PATH: &str = ":memory:";

/// WAL 文件容量（独立 `wal.log`；环形写满经检查点推进 + 打孔回收）
const WAL_FILE_CAPACITY: u64 = 16 << 20;
/// WAL 分片大小（分片检查点粒度）
const WAL_SHARD_SIZE: u64 = 1 << 20;
/// 组提交聚合缓冲
const GROUP_COMMIT_BUFFER: usize = 1 << 20;
/// GC task 轮询周期
const GC_TICK: Duration = Duration::from_millis(10);
/// 检查点 task 周期
const CHECKPOINT_TICK: Duration = Duration::from_millis(100);
/// 单轮 GC 搬移的最大页数
const GC_BATCH_PAGES: usize = 256;
/// GC 组间失衡阈值（上游 GreedyAmongImbalancedGroups 的 mincnt）
const GC_IMBALANCE_THRESHOLD: u64 = 4;

/// 数据库配置
#[derive(Debug, Clone)]
pub struct Config {
  /// 库目录（`":memory:"` = 纯内存后端）
  pub path: String,
  /// worker 数
  pub workers: usize,
  /// 缓冲物理帧预算（单位：4 KiB 逻辑帧；虚拟页 = 4×物理帧，VMCache 超-commit）。
  ///
  /// **语义（主机页感知）**：该值是**4 KiB 帧数预算**，启动时经
  /// [`ldb_vmem::VirtualMemoryArena::frames_for_bytes`] 换算为常驻帧上限 ——
  /// `physical_cnt = frames * 4096 / max(host_page, 4096)`。4 KiB 主机页上
  /// 换算结果等于 `frames`（无变化）；16 KiB/64 KiB 主机页上按比例缩小，使
  /// 满池 RSS 上界恒为 `frames * 4 KiB`，避免单帧独占一个主机页导致 RSS ×4/×16。
  pub frames: u64,
  /// WAL worker 缓冲 KB
  pub wal_kb: usize,
  /// 追加偏置
  pub append_bias: bool,
  /// WAL 落盘同步开关（对齐 `db_bench --sync=1/0`）。
  ///
  /// `true`（默认，`--sync=1`）：组提交 PhaseTwo 对 `wal.log` 底座做真实刷盘屏障
  /// —— Linux `fdatasync`、macOS `F_FULLFSYNC`（Darwin fsync 不保证设备缓存落盘，
  /// 实测约 4.27 ms/次）。提交返回即为持久化点。
  /// `false`（`--sync=0`）：跳过刷盘屏障（仅写 OS 页缓存），吞吐优先；
  /// **丢数据窗口** = 崩溃/掉电时最近一段未刷盘 WAL 与之外对应的已提交事务丢失
  /// （页写路径本身仍受 WAL-before-page 门保护，不会先落页后落日志）。
  pub wal_sync: bool,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      path: MEM_PATH.into(),
      workers: 1,
      frames: 4096,
      wal_kb: 1024,
      append_bias: true,
      wal_sync: true,
    }
  }
}

/// 存储后端（文件 / 内存统一分派 —— 后端选型是启动期决策，enum 静态分发）
pub enum StoreBackend {
  /// compio 文件后端（`data.db`；Direct I/O 自动回退）
  File(CompioFileBackend),
  /// 纯内存后端（测试 / Fuzz）
  Mem(MemBackend),
}

impl fmt::Debug for StoreBackend {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("StoreBackend")
      .field(
        "kind",
        &match self {
          Self::File(_) => "file",
          Self::Mem(_) => "mem",
        },
      )
      .finish_non_exhaustive()
  }
}

impl PageBackend for StoreBackend {
  async fn read_page(&self, pid: PageId, dst: &mut [u8; 4096]) -> io::Result<()> {
    match self {
      Self::File(f) => {
        let tmp = [0u8; 4096];
        // 稀疏文件语义：从未写过的页 / 文件短于 `(pid+1)*PAGE_SIZE` 时按零页返回
        // （与 `MemBackend::read_at_page` 的「未写区域读零」契约一致）。文件后端
        // 的 `read_exact` 在越过 EOF 时报 `UnexpectedEof`，若直接传播则**新库首个
        // 页写入**（缺页读零 → 就地写）无法进行；此处把「越 EOF」归一为零页，
        // 真实 I/O 错误仍原样传播。
        match f.read_owned(pid, tmp).await {
          Ok(tmp) => dst.copy_from_slice(&tmp),
          Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => dst.fill(0),
          Err(e) => return Err(e),
        }
        Ok(())
      }
      Self::Mem(m) => m.read_page(pid, dst).await,
    }
  }

  async fn write_page(&self, pid: PageId, src: &[u8; 4096]) -> io::Result<()> {
    match self {
      // write_at_page 要求 'static → 拷为 owned（I/O 开销远大于 4KiB 拷贝）
      Self::File(f) => f.write_owned(src.to_vec(), pid * 4096).await.map(|_| ()),
      Self::Mem(m) => m.write_page(pid, src).await,
    }
  }

  async fn sync(&self) -> io::Result<()> {
    match self {
      Self::File(f) => f.fdatasync().await,
      Self::Mem(_) => Ok(()),
    }
  }
}

/// WAL 刷盘策略包装（`Config::wal_sync`）。
///
/// 组提交 PhaseTwo 统一经 [`WalSink::sync`] 落盘屏障；本包装把「是否真刷盘」
/// 决策收敛到一处，底座与上层组提交逻辑零改动。`sync_enabled=false` 时
/// 直接返回 `Ok`（不产生设备屏障），对应 `db_bench --sync=0`。
struct SyncPolicySink {
  inner: Box<dyn WalSink>,
  sync_enabled: bool,
}

impl fmt::Debug for SyncPolicySink {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SyncPolicySink")
      .field("sync_enabled", &self.sync_enabled)
      .finish_non_exhaustive()
  }
}

impl WalSink for SyncPolicySink {
  fn write_wal(
    &mut self,
    data: Vec<u8>,
    off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    self.inner.write_wal(data, off)
  }

  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    if self.sync_enabled {
      self.inner.sync()
    } else {
      Box::pin(async { Ok(()) })
    }
  }
}

/// 文件 WAL 的掉电落盘屏障适配。
///
/// `ldb_wal` 的 `impl WalSink for CompioFileBackend` 把 `sync` 映射成
/// `fdatasync`；在 macOS 上 `fdatasync`→`fsync` 只把脏页交给驱动，**不清空设备
/// 写缓存**（`ldb_io` 文档亦如此声明）。`Config::wal_sync=true` 的语义是「提交
/// 返回即掉电持久」，故本适配把 WAL 屏障收敛到
/// [`CompioFileBackend::sync_durable`]（macOS `F_FULLFSYNC`；Linux/Windows 为
/// `fdatasync`/`FlushFileBuffers`），在 `ldb` 门面内落地、不改 `ldb_wal`。
///
/// 实测（`cargo bench -p ldb_io`，macOS）：同 8.4 MiB 负载
/// `dwb_sync_durable` 11.85 ms vs `dwb_fdatasync` 1.40 ms（8.5×）——设备缓存
/// flush 是真实存在的额外成本，正是 `wal_sync=true` 要买到的持久性。
struct DurableWalFile(CompioFileBackend);

impl fmt::Debug for DurableWalFile {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("DurableWalFile").finish_non_exhaustive()
  }
}

impl WalSink for DurableWalFile {
  fn write_wal(
    &mut self,
    data: Vec<u8>,
    off: u64,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(self.0.write_owned(data, off))
  }

  fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    Box::pin(self.0.sync_durable())
  }
}

/// Blob 页存取端口（`BlobPageStore` 生产实现 —— extent 页经 VMCache：
/// 写 = `update_pid`（EXCLUSIVE + 标脏，驱逐/刷盘统一承载），读 = `fix_share`
///（缺页自动从盘加载），分配 = `alloc_extent`（与树页共享分配器）
pub struct BufBlobStore {
  buf: Rc<BufferManager<StoreBackend, WalTxnLogHook>>,
}

impl fmt::Debug for BufBlobStore {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BufBlobStore").finish_non_exhaustive()
  }
}

impl BlobPageStore for BufBlobStore {
  const MANAGED: bool = true;

  async fn alloc_pages(&self, page_cnt: u32) -> ldb_blob::Result<PageId> {
    self
      .buf
      .alloc_extent(page_cnt)
      .await
      .map_err(|e| LdbBlobError::StorePage(format!("extent 分配失败: {e}")))
  }

  async fn write_page(&self, pid: PageId, page: &[u8; 4096]) -> ldb_blob::Result<()> {
    self
      .buf
      .update_pid(pid, page)
      .await
      .map(|_| ())
      .map_err(|e| LdbBlobError::StorePage(format!("extent 页写失败: {e}")))
  }

  async fn read_page(&self, pid: PageId, dst: &mut [u8; 4096]) -> ldb_blob::Result<()> {
    self
      .buf
      .fix_share_impl(pid, dst)
      .await
      .map(|_| ())
      .map_err(|e| LdbBlobError::StorePage(format!("extent 页读失败: {e}")))
  }
}

/// GC 钩子（L5 装配：空间耗尽 / 块满回调 GC 执行器 —— GcHook 生产实现）
struct FacadeGc {
  gc: Rc<PLMutex<GarbageCollector<WalTxnLogHook>>>,
  space: Rc<SpaceManager<WalTxnLogHook>>,
  buf: Rc<BufferManager<StoreBackend, WalTxnLogHook>>,
}

impl fmt::Debug for FacadeGc {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("FacadeGc").finish_non_exhaustive()
  }
}

impl FacadeGc {
  /// 走完上游块状态迁移链 `FULL → GC_READ → GC_WRITE` 后再擦除回收 victim，
  /// 返回 [`SpaceManager::erase_block`] 的**真实**结果。
  ///
  /// 上游编排：`MoveBlockFromFullToGCBlockList`（`garbage_collector.cc:816-823`，
  /// `FULL → GC_IN_USE`）→ `UpdateBlockMetadataAfterReadGC`（`space_manager.cc:690-698`，
  /// `GC_IN_USE → GC_READ`）→ `SelectVictimBlocksToGCWrite`（`:358`，`GC_READ → GC_WRITE`）
  /// → `EraseBlock`（`space_manager.cc:1334-1364` → `ResetBlockMetadata` `:1718-1730`，
  /// `GC_WRITE → EMPTY`；`GC_WRITE → OPEN` 的 `MoveGCedBlockToOpenList` `:1614-1663`
  /// 是另一条重开路径，本 crate 由 [`SpaceManager::move_g_ced_block_to_open_list`] 承担）。
  /// ldb_gc 的选择器（`greedy_gc` 等）只读不改 state，故这两步迁移必须由引擎显式执行
  /// —— 否则 victim 恒为 `Full`，`space_mgr.rs` 的 `erase_block` 走 `_ => false`
  /// 分支（`erase_block` 只接受 `GC_WRITE`），victim 永不回收且原实现谎报 `true`。
  fn reclaim_victim(&self, victim: BlockId) -> bool {
    if !self
      .gc
      .lock()
      .move_block_from_full_to_gc_block_list(&self.space, victim)
    {
      log::warn!("GC victim {victim} FULL→GC_READ 迁移失败（非 Full 态），本轮放弃回收");
      return false;
    }
    if !self.space.mark_block_gc_write(victim) {
      log::warn!("GC victim {victim} GC_READ→GC_WRITE 迁移失败，本轮放弃回收");
      return false;
    }
    self.space.erase_block(victim)
  }

  /// 单块 GC 搬移（上游 GCWriteValidPIDs 编排：选牺牲块 → 读有效页 →
  /// 写新块 + 映射更新 → 旧块迁移到 GC_WRITE 后擦除）。返回是否推进了回收
  async fn gc_one_block(&self) -> bool {
    let victim = { self.gc.lock().greedy_gc(&self.space, true) };
    if victim == BlockId::MAX {
      return false;
    }
    let pids = match self.buf.gc_read_valid_pi_ds(&[victim]).await {
      Ok(p) => p,
      Err(e) => {
        log::warn!("GC 读牺牲块 {victim} 失败: {e}");
        return false;
      }
    };
    if pids.is_empty() {
      // 全死块：无需搬移，直接走迁移链回收（返回 erase_block 真实结果）
      return self.reclaim_victim(victim);
    }
    if pids.len() > GC_BATCH_PAGES {
      // 单轮上限：超长牺牲块留下轮继续（victim 不迁移、不擦除，仍在 FULL 列表）
      return false;
    }
    let dst = self
      .space
      .select_empty_block_id(pids.len() as u64 * PAGE_SIZE as u64, 0);
    if dst == BlockId::MAX {
      return false;
    }
    match self.buf.flush_valid_pi_ds_to_block(&pids, dst).await {
      Ok(_) => self.reclaim_victim(victim),
      Err(e) => {
        log::warn!("GC 搬移 {victim}→{dst} 失败: {e}");
        false
      }
    }
  }
}

impl GcHook for FacadeGc {
  fn on_block_full(&self, _bid: BlockId) -> Pin<Box<dyn Future<Output = ()> + '_>> {
    // 牺牲块选择由 GC task 周期驱动（上游 AddBlockAsGCVictim 的 task 化等价）
    Box::pin(async {})
  }

  fn on_gc_needed(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
    Box::pin(self.gc_one_block())
  }
}

/// 顶层数据库（LeanStore 外观）
pub struct LeanStore {
  /// 配置
  pub config: Config,
  /// 主 B+Tree
  pub tree: BTree,
  /// Blob 管理器（extent 页经 VMCache 落盘/加载）
  pub blobs: BlobManager<BufBlobStore>,
  /// 空间管理器（挂 WalTxnLogHook —— 映射变更写真实 WAL）
  pub space: Rc<SpaceManager<WalTxnLogHook>>,
  /// 空闲空间管理
  pub free: Arc<FreeStorageManager>,
  /// 日志协调者（与 txns 共享；GSN 水位发布点）
  pub log_mgr: Rc<LogManager>,
  /// VMCache 缓冲池
  pub buf: Rc<BufferManager<StoreBackend, WalTxnLogHook>>,
  /// GC 执行器（FacadeGc 注入 space.gc_hook）
  pub gc: Rc<PLMutex<GarbageCollector<WalTxnLogHook>>>,
  /// 事务管理器
  pub txns: Rc<TransactionManager>,
  /// OSIC 并发控制（互斥保护；可见性查询经锁）
  pub cc: PLMutex<ldb_txn::ConcurrencyControl>,
  /// 统计收集器（注入 `ldb_buf`/`ldb_txn` 生产点；WAF/命中率读取点）
  pub stat: Arc<StatCollector>,
  /// 版本指纹（上游 GetVersion 的配置摘要）
  version: u64,
  /// 最近活动时刻（毫秒；coarsetime。事务开/提交/回滚与 KV 写刷新 —— Poisson 空闲补偿判定基准）
  last_activity_ms: atomic::AtomicU64,
}

/// 空闲判定阈值（毫秒）
const IDLE_THRESHOLD_MS: u64 = 100;

/// 进程内单调毫秒刻度（coarsetime；epoch = 首次调用时刻）
fn now_ms() -> u64 {
  static EPOCH: OnceLock<coarsetime::Instant> = OnceLock::new();
  let epoch = EPOCH.get_or_init(Instant::now);
  // 必须用 now()（内部刷新 recent 缓存）；recent() 只读缓存，无刷新点则恒 0
  Instant::now().duration_since(*epoch).as_u64()
}

impl fmt::Debug for LeanStore {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("LeanStore")
      .field("path", &self.config.path)
      .field("workers", &self.config.workers)
      .finish_non_exhaustive()
  }
}

impl LeanStore {
  /// 打开（装配全部组件）
  pub async fn open(config: Config) -> Result<Self> {
    let workers = config.workers.max(1);
    // 独立文件布局（C-6）：`data.db` / `wal.log` 分文件，mapping_offset = 0
    let (backend, wal_sink, file) = if config.path == MEM_PATH {
      (
        StoreBackend::Mem(MemBackend::new()),
        Box::new(MemBackend::new()) as Box<dyn WalSink>,
        WalFileState {
          start: 0,
          end: WAL_FILE_CAPACITY,
          shard: WAL_SHARD_SIZE,
        },
      )
    } else {
      (
        StoreBackend::File(
          CompioFileBackend::open_direct_or_buffered(format!("{}/data.db", config.path)).await?,
        ),
        Box::new(DurableWalFile(
          CompioFileBackend::open_direct_or_buffered(format!("{}/wal.log", config.path)).await?,
        )) as Box<dyn WalSink>,
        WalFileState {
          start: 0,
          end: WAL_FILE_CAPACITY,
          shard: WAL_SHARD_SIZE,
        },
      )
    };
    let tm = TransactionManager::new(
      workers,
      config.wal_kb,
      CommitVariant::WorkersWriteLog,
      Box::new(SyncPolicySink {
        inner: wal_sink,
        sync_enabled: config.wal_sync,
      }),
      file,
    );
    let is_running = tm.is_running.clone();
    let log_mgr = tm.log_mgr.clone();
    let mut txns = tm;
    // 统计收集器（L5 生成、注入各 L3/L4 生产点；facade 经 `stat()` 读取）
    let stat = Arc::new(StatCollector::default());
    txns.set_stats(stat.clone());
    let txns = Rc::new(txns);
    // 事务日志钩子（L4 实现注入 L3 —— 映射变更 / 有效页镜像 / 页释放走真 WAL）
    let hook = WalTxnLogHook::new(log_mgr.local_log_worker(0).clone(), log_mgr.clone());
    let space = Rc::new(SpaceManager::with_txn_log(
      (config.frames / 64).max(16),
      64,
      2048,
      4,
      config.frames,
      2,
      hook,
    ));
    let free = Arc::new(FreeStorageManager::new(true, workers));
    // 物理帧预算按主机页换算（`config.frames` 语义 = 4 KiB 帧数）：
    // 主机页 16 KiB 时 physical_cnt = frames/4，使满池 RSS 上界恒为 frames*4 KiB。
    let physical_frames = VirtualMemoryArena::frames_for_bytes(config.frames * PAGE_SIZE as u64)?;
    // 缓冲池：与 LogManager 共享同一 GsnWatermark（D5）
    let mut buf = BufferManager::new(
      config.frames * 4,
      physical_frames.max(1),
      space.clone(),
      free.clone(),
      backend,
      is_running.clone(),
      log_mgr.min_gsn_flushed.clone(),
    )
    .map_err(Error::Buf)?;
    buf.set_stats(stat.clone());
    let buf = Rc::new(buf);
    let gc = Rc::new(PLMutex::new(GarbageCollector::new(GC_IMBALANCE_THRESHOLD)));
    // GcHook 注入（Arc 化后经 OnceLock 补挂）
    space.set_gc_hook(Rc::new(FacadeGc {
      gc: gc.clone(),
      space: space.clone(),
      buf: buf.clone(),
    }));
    // page-provider 驱逐 task（上游 RunPageProviderThreads 的 task 化）
    Runtime::current()
      .spawn({
        let buf = buf.clone();
        async move {
          let _ = buf.run_page_provider_task().await;
        }
      })
      .detach();
    let cc = PLMutex::new(ConcurrencyControl::new(workers));
    let version = Self::fingerprint(&config);
    let blobs = BlobManager::with_store(4, BufBlobStore { buf: buf.clone() });
    Ok(Self {
      tree: BTree::create(MEMCMP),
      blobs,
      config,
      space,
      free,
      log_mgr,
      buf,
      gc,
      txns,
      cc,
      stat,
      version,
      last_activity_ms: atomic::AtomicU64::new(now_ms()),
    })
  }

  /// 刷新活动时间戳（事务/KV 路径调用）
  #[inline]
  fn touch_activity(&self) {
    self
      .last_activity_ms
      .store(now_ms(), atomic::Ordering::Relaxed);
  }

  /// 配置指纹（上游 GetVersion：各 flags 算术合成）
  pub fn get_version(&self) -> u64 {
    self.version
  }

  /// 统计收集器（只读共享视图；`snapshot_metric`/`calc_db_waf` 经此读取）
  pub fn stat(&self) -> &StatCollector {
    &self.stat
  }

  fn fingerprint(c: &Config) -> u64 {
    let mut v: u64 = 0x1db;
    v = v.wrapping_mul(0x100000001b3) ^ c.workers as u64;
    v = v.wrapping_mul(0x100000001b3) ^ c.frames;
    v = v.wrapping_mul(0x100000001b3) ^ u64::from(c.append_bias);
    v = v.wrapping_mul(0x100000001b3) ^ u64::from(c.wal_sync);
    v
  }

  // ---- 表 / 索引 ----

  /// 注册表（上游 RegisterTable：索引槽分配；此处主树即 0 号索引）
  pub async fn register_table(&mut self, _name: &str) -> u16 {
    0
  }

  /// 取索引（上游 RetrieveIndex；单树形态返回主树 pid 视图）
  pub async fn retrieve_index(&self, idx: u16) -> Option<PageId> {
    (idx == 0).then_some(self.tree.root_pid)
  }

  /// 取比较函数（上游 RetrieveComparisonFunc）
  pub async fn retrieve_comparison_func(&self) -> fn(&[u8], &[u8]) -> cmp::Ordering {
    self.tree.cmp.f
  }

  // ---- KV 读写（WAL 追加路径）----

  /// KV 写入（门面 WAL 追加路径）：把「键‖值」用户负载追加进 WAL 数据日志，
  /// 并按 STAT-R3-2 口径记 [`StatCollector::record_logical_write`]（WAF 分母），
  /// 随后写入主树。
  ///
  /// 取数点必须是**用户负载 append** 处（不是 `write_log_buffer` 的段切分，后者含
  /// 条目头/对齐零填充）。此前门面 KV 路径未接 WAL ⇒ `logical_write` 无生产者、
  /// `calc_db_waf()` 分母恒 0；本方法落地该生产者。
  pub async fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
    self.touch_activity();
    let mut payload = Vec::with_capacity(key.len() + value.len());
    payload.extend_from_slice(key);
    payload.extend_from_slice(value);
    self
      .txns
      .log
      .reserve_and_submit(
        &self.log_mgr,
        EntryType::DataEntry,
        self.tree.root_pid,
        &payload,
        0,
      )
      .await?;
    self.stat.record_logical_write(payload.len() as u64);
    self.tree.insert(key, value).await?;
    Ok(())
  }

  /// KV 读取（主树）
  pub async fn get(&self, key: &[u8], out: &mut Vec<u8>) -> Result<bool> {
    Ok(self.tree.look_up(key, out).await?)
  }

  // ---- 事务（真实 WAL 链：TX_COMMIT → 组提交 → 水位固化）----

  /// 开启事务（0 号 worker）
  pub async fn start_transaction(&self, isolation: IsolationLevel) {
    self.touch_activity();
    self.txns.start_transaction(0, isolation);
  }

  /// 提交事务（写 TX_COMMIT + 入组提交队列 + 等待固化落盘）
  pub async fn commit_transaction(&self) -> Result<u64> {
    let ticket = self
      .txns
      .commit_transaction(0)
      .await
      .map_err(Error::WalTxn)?;
    let durable = self
      .txns
      .durable_commit(ticket)
      .await
      .map_err(Error::WalTxn)?;
    Ok(durable.commit_ts)
  }

  /// 回滚事务（TX_ABORT 记录 + 版本撤销）
  pub async fn abort_transaction(&self) -> Result<()> {
    self.touch_activity();
    self.txns.abort_transaction(0).await.map_err(Error::WalTxn)
  }

  // ---- Blob ----

  /// 新建 blob
  pub async fn create_new_blob(&mut self, payload: &[u8]) -> Result<u64> {
    Ok(self.blobs.allocate_blob(payload).await?)
  }

  /// 加载 blob
  pub async fn load_blob(&self, id: u64) -> Result<Vec<u8>> {
    Ok(self.blobs.load_blob(id).await?)
  }

  /// 删除 blob
  pub async fn remove_blob(&mut self, id: u64) -> Result<bool> {
    Ok(self.blobs.remove_blob(id).await?)
  }

  // ---- 统计 ----

  /// DB 逻辑尺寸
  pub async fn db_size(&self) -> u64 {
    self.space.get_db_size()
  }

  /// WAL 已用量（文件偏移 - 起始）
  pub async fn wal_size(&self) -> u64 {
    self.log_mgr.file.end - self.log_mgr.wal_file_offset()
  }

  /// 已分配 PID 数
  pub async fn allocated_pi_ds_cnt(&self) -> u64 {
    self.space.get_max_pid()
  }

  /// 已分配字节数
  pub async fn allocated_size(&self) -> u64 {
    self.space.get_db_size()
  }

  // ---- 后台任务（compio spawn；is_running 关停自退）----

  /// 启动 GC 任务（上游 StartGarbageCollectorThread：触发判定 → 搬移循环 +
  /// 背压 sleep；task 化 spawn，`shutdown` 后自退）
  pub async fn start_gc_task(&self) -> Result<()> {
    let space = self.space.clone();
    let buf = self.buf.clone();
    let gc = self.gc.clone();
    let log_mgr = self.log_mgr.clone();
    let hook = FacadeGc {
      gc,
      space: space.clone(),
      buf,
    };
    Runtime::current()
      .spawn(async move {
        while log_mgr.should_keep_running() {
          if space.should_trigger_gc(0) {
            hook.gc_one_block().await;
          }
          sleep(GC_TICK).await;
        }
      })
      .detach();
    Ok(())
  }

  /// 启动检查点任务（模糊检查点循环：推进共享水位 + 全量刷脏）
  pub async fn start_checkpoint_task(&self) -> Result<()> {
    let buf = self.buf.clone();
    let log_mgr = self.log_mgr.clone();
    Runtime::current()
      .spawn(async move {
        while log_mgr.should_keep_running() {
          let _ = buf.fuzzy_checkpoint(log_mgr.min_gsn_flushed.load()).await;
          sleep(CHECKPOINT_TICK).await;
        }
      })
      .detach();
    Ok(())
  }

  /// 启动组提交任务（上游 StartGroupCommitThread：5 阶段循环；
  /// EvictHook 以 `BufEvictHook` 生产实现注入 —— D4）
  pub async fn start_group_commit_thread(&self) -> Result<()> {
    let mgr = self.log_mgr.clone();
    let evict = BufEvictHook {
      mgr: self.buf.clone(),
    };
    let f = self.log_mgr.file;
    let mut exec = GroupCommitExecutor::new(
      GcCfg {
        start_wid: 0,
        end_wid: self.config.workers.max(1),
        write_buffer_size: GROUP_COMMIT_BUFFER,
        wal_start: f.start,
        wal_end: f.end,
        shard_wal_size: f.shard,
      },
      evict,
      None,
      None,
    );
    Runtime::current()
      .spawn(async move {
        if let Err(e) = exec.start_execution(&mgr).await {
          log::error!("组提交循环退出: {e}");
        }
      })
      .detach();
    Ok(())
  }

  /// 空闲补偿写（SIGMOD'25 Poisson 模型）：距最近活动超过阈值且 1/4 概率
  /// 触发一轮本地 WAL 落盘（coarsetime 低开销时钟 —— SKILL 时间戳选型）
  pub async fn check_during_idle(&self) -> bool {
    let now = now_ms();
    let last = self.last_activity_ms.load(atomic::Ordering::Relaxed);
    if now.saturating_sub(last) < IDLE_THRESHOLD_MS || fastrand::u64(..4) != 0 {
      return false;
    }
    self.touch_activity();
    let _ = self.txns.log.worker_writes_log(&self.log_mgr, false).await;
    true
  }

  /// 数据装载完成（上游 EndDataLoad 的 sleep(10) 排水 → 真实刷盘 + 偏置切换）
  pub async fn end_data_load(&mut self) {
    let _ = self.buf.flush_all().await;
    self.tree.toggle_append_bias_mode();
  }

  /// 清缓存（上游 DropCache：全量驱逐语义）
  pub async fn drop_cache(&self) {
    while self.buf.physical_used.load(atomic::Ordering::Relaxed) > 0 {
      match self.buf.evict(64).await {
        Ok(0) => break, // 无可驱逐（全被闩住/失败）
        Ok(_) => {}
        Err(_) => break,
      }
    }
  }

  /// 关停（WAL 缓冲先 persist 由组提交 task 排水；is_running 置假令全部 task 自退）
  pub async fn shutdown(&self) {
    let _ = self.buf.flush_all().await;
    self.log_mgr.shutdown();
    self.txns.is_running.store(false, atomic::Ordering::Release);
  }
}

#[cfg(test)]
mod tests {
  use std::sync::atomic::{AtomicU64, Ordering as O};

  use super::*;

  /// 计数型 `WalSink`（唯一用途：验证 `SyncPolicySink` 是否真调底座 sync）。
  struct CountingSink {
    syncs: Arc<AtomicU64>,
  }

  impl fmt::Debug for CountingSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.debug_struct("CountingSink").finish_non_exhaustive()
    }
  }

  impl WalSink for CountingSink {
    fn write_wal(
      &mut self,
      _data: Vec<u8>,
      _off: u64,
    ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
      Box::pin(async { Ok(()) })
    }

    fn sync(&self) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
      self.syncs.fetch_add(1, O::Relaxed);
      Box::pin(async { Ok(()) })
    }
  }

  /// `wal_sync=false` 不得调用底座刷盘屏障；`true` 必须透传。
  ///
  /// 唯一击杀：把 `SyncPolicySink::sync` 改成无条件 `self.inner.sync()`
  /// → `sync=0` 分支断言（期望 0）失败。
  #[compio::test]
  async fn sync_policy_sink_gates_backend_sync() {
    let syncs = Arc::new(AtomicU64::new(0));
    let make = |enabled: bool| SyncPolicySink {
      inner: Box::new(CountingSink {
        syncs: syncs.clone(),
      }),
      sync_enabled: enabled,
    };
    assert!(make(false).sync().await.is_ok());
    assert_eq!(syncs.load(O::Relaxed), 0, "wal_sync=false 不得刷盘");
    assert!(make(true).sync().await.is_ok());
    assert_eq!(syncs.load(O::Relaxed), 1, "wal_sync=true 必须刷盘");
  }

  /// 指纹必须区分 `wal_sync`（配置语义变更 ⇒ 版本号变更）。
  ///
  /// 唯一击杀：从 `fingerprint` 删掉 `wal_sync` 项 → 两配置同指纹 → 断言失败。
  #[test]
  fn fingerprint_distinguishes_wal_sync() {
    let base = Config {
      wal_sync: true,
      ..Default::default()
    };
    let flipped = Config {
      wal_sync: false,
      ..base.clone()
    };
    assert_ne!(
      LeanStore::fingerprint(&base),
      LeanStore::fingerprint(&flipped),
      "wal_sync 必须进入配置指纹"
    );
  }

  /// P1 修复：`durable_commit` 的**自驱回退路径**在 `wal_sync=true` 时必须真正
  /// 经过设备屏障（此前 `worker_writes_log` 丢弃 `force_commit`、`persist_log` 只
  /// `write_wal` ⇒ 提交返回未越过设备写缓存）。
  ///
  /// 唯一击杀：① 还原 `worker_writes_log` 丢弃 `force_commit` → `>= 1` 断言在
  /// `enabled=true` 侧失败（0 屏障）；② 去掉 `SyncPolicySink` 的门 → `enabled=false`
  /// 侧 `== 0` 断言失败（误刷盘）。
  #[compio::test]
  async fn durable_commit_self_driven_path_establishes_barrier() {
    let make = |enabled: bool, syncs: Arc<AtomicU64>| {
      TransactionManager::new(
        1,
        64,
        CommitVariant::WorkersWriteLog,
        Box::new(SyncPolicySink {
          inner: Box::new(CountingSink { syncs }),
          sync_enabled: enabled,
        }),
        WalFileState {
          start: 0,
          end: WAL_FILE_CAPACITY,
          shard: WAL_SHARD_SIZE,
        },
      )
    };

    // wal_sync=true：一次「开事务 → 提交 → durable」必须至少过一道设备屏障
    let syncs_on = Arc::new(AtomicU64::new(0));
    let tm_on = make(true, syncs_on.clone());
    tm_on.start_transaction(0, IsolationLevel::SnapshotIsolation);
    let t = tm_on.commit_transaction(0).await.unwrap();
    tm_on.durable_commit(t).await.unwrap();
    assert!(
      syncs_on.load(O::Relaxed) >= 1,
      "wal_sync=true 自驱 durable_commit 必须建立设备屏障（实测 {}）",
      syncs_on.load(O::Relaxed)
    );

    // wal_sync=false：同一路径必须 0 屏障（门控在 sink 层）
    let syncs_off = Arc::new(AtomicU64::new(0));
    let tm_off = make(false, syncs_off.clone());
    tm_off.start_transaction(0, IsolationLevel::SnapshotIsolation);
    let t2 = tm_off.commit_transaction(0).await.unwrap();
    tm_off.durable_commit(t2).await.unwrap();
    assert_eq!(
      syncs_off.load(O::Relaxed),
      0,
      "wal_sync=false 不得产生设备屏障"
    );
  }
}
