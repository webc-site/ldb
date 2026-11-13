//! 依赖倒置 trait 枢纽（todo.md §3.0 反向边 2/3/4 的消除方案）。
//!
//! trait 定义在 L2，由 L3 `ldb_buf` / L4 `ldb_txn` 实现并注入。
//!
//! 签名约定（2026-11-11 D4 修订）：
//! - **无 `Send` 约束**：compio 单线程每核，`Runtime::spawn` 只要求
//!   `Future + 'static`（`compio-runtime/src/lib.rs:221`），task 体内
//!   `!Send` 类型（arena 指针 / 线程局部后端）合法；
//! - `TxnLogHook` 用 `&self`（经 `SpaceManager<T>` 泛型热路径分发，
//!   共享态由实现方内部可变性承载）；
//! - `EvictHook` 用 `&mut self`（`GroupCommitExecutor` 按值持有，执行器独占）；
//! - `GcHook` 用 `&self`（`Rc<dyn GcHook>` 共享注入，实现方内部可变性承载；
//!   块满/GC 为冷路径，每事件一次装箱 Future 分配）。

use std::{future::Future, pin::Pin};

use ldb_type::{BlockId, PageId};

/// 驱逐钩子：L3 WAL 组提交 PhaseFour 回调缓冲池（消除 ldb_wal → ldb_buf 环）
pub trait EvictHook {
  /// 驱逐一个 extent 的缓冲页（上游 `EvictExtent`：逐页 TryLockExclusive + 写盘）
  fn evict_extent(&mut self, extent: PageId) -> impl Future<Output = ()>;

  /// 对外公布空闲区间（组提交 PhaseFive 列表裁剪）
  fn public_free_ranges(&mut self) -> impl Future<Output = Vec<(u64, u64)>>;
}

/// GC 钩子：L3 空间管理回调 GC 执行器（消除 ldb_space → ldb_gc 环）。
///
/// 冷路径（块满 / 空间耗尽才触发）→ 装箱 Future 保持 dyn 兼容，由
/// `SpaceManager::gc_hook: OnceLock<Arc<dyn GcHook>>` 单点注入；
/// `&self` 接收器 —— 共享态由实现方内部可变性（Mutex）承载。
pub trait GcHook {
  /// 块满回调（触发牺牲块选择）
  fn on_block_full(&self, bid: BlockId) -> Pin<Box<dyn Future<Output = ()> + '_>>;

  /// GC 需要回调（有效页搬移读取，含 I/O）；返回是否推进了回收
  fn on_gc_needed(&self) -> Pin<Box<dyn Future<Output = bool> + '_>>;
}

/// 事务日志钩子：L3 空间/释放管理回调 WAL（消除 L3 → L4 上向边）。
///
/// 由 `SpaceManager<T>` / `GarbageCollector<T>` 泛型承载（热路径每页一次
/// `update_pid2_offset` 调用，禁 `dyn` —— SKILL 热路径静态分发条款）。
pub trait TxnLogHook {
  /// 记录 PID→偏移映射变更（上游 `LogPID2Offset`，经 ReserveDataLog 可能等待 WAL flush）
  fn log_pid2_offset(&self, pid: PageId) -> impl Future<Output = ()>;

  /// 记录有效页镜像（GC 搬移前）
  fn log_valid_page(&self, pid: PageId) -> impl Future<Output = ()>;

  /// 记录页释放
  fn log_free_page(&self, pid: PageId) -> impl Future<Output = ()>;
}
