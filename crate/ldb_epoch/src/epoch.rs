//! 基于 Epoch 的并发延迟回收（移植自 `leanstore.latency/src/sync/epoch_handler.{h,cc}`）。
//!
//! 上游延迟释放裸 `void*`；Rust 侧改为类型擦除的 `Box<dyn Any + Send>`，回收时 drop。
//! 采样率与上游一致：推进 1/200（`epoch_handler.cc:17`）、回收 1/100（`epoch_handler.cc:20`）。
//!
//! # 并发契约
//!
//! 每个 `wid` 对应一个独立的待释放队列（对应上游 `wid_t` = worker id）：`defer_free_pointer`
//! 与 `epoch_operation` 只访问 `to_free[wid]`，不同 `wid` 的队列互不相交，仅共享原子量，因此
//! 全部方法取 `&self`，共享 `&EpochHandler` 即可并发调用（`todo.md:178` 的并发要求）。队列用
//! `parking_lot::Mutex` 保护：不同 `wid` 仍完全并行；同一 `wid` 被多个线程并发调用时由队列锁
//! 串行化，是内存安全的（不再是 UB）。每 `wid` 的队列锁与本地 epoch 都独占缓存行（`Padded`）：
//! 2 线程写相邻 `wid` 实测（Apple arm64，2026-11-10，3 次中位数），队列不对齐时 `defer`
//! 6.7→41ns、`epoch_operation` 2.4→24ns，本地 epoch 不对齐时 `enter`+`leave` 2.3→21.5ns。
//! `enter`/`leave` 只碰原子量，任何 `wid` 都可由任意线程调用。
//!
//! 队列锁内不执行用户代码：`epoch_operation` 先在锁内摘下可回收前缀，再在锁外 `drop`，因此
//! 延迟对象的析构器回调本 handler（含同一 `wid`）不会死锁。但消费方若在调用 `epoch_operation`
//! 时另持外层锁（`ldb_range`/`ldb_txn` 均持有 `Mutex<EpochHandler>`），析构器回调该外层锁仍会
//! 死锁，属调用方契约。
//!
//! 注意：单 `wid` 的 `enter`/`leave`/`defer` 交错属于上游未定义的调用协议，本实现保证其内存
//! 安全，但不保证 EBR 释放时机的语义正确性（与上游一致，属调用方契约）。
//!
//! # 内存序
//!
//! 上游全部使用 `std::atomic` 默认的 seq_cst。本实现保持 seq_cst，唯一例外是 `enter` 的本地
//! epoch 发布：先 `Relaxed` 存储再 `fence(SeqCst)`。原因是 aarch64 的 seq_cst 存储只编译为
//! `stlr`（release），后续普通 load 仍可被重排到该存储之前，回收线程可能读到旧哨兵并把正在
//! 读取的对象释放。`fence(SeqCst)`（`dmb ish`）建立 store→load 屏障；这与 crossbeam-epoch 的
//! pin 实现相同（`crossbeam-epoch-0.9.21/src/internal.rs:409-443` 原文注释：*The fence makes sure
//! that any future loads from `Atomic`s will not happen before this store*）。
//!
//! # 与上游的已知差异（均有意为之）
//!
//! - `new` 把各 worker 的本地 epoch 初始化为哨兵 `MAX_VALUE`；上游构造只做值初始化，元素实际为
//!   0（`std::vector<std::atomic<u64>>(n)` 的值初始化），会让“从未进入”的 worker 以 epoch 0 阻塞
//!   回收。本实现按 `EpochGuard` 析构写入的哨兵语义初始化，更符合哨兵契约；worker 进入过保护
//!   区间后，两者行为一致。
//! - 上游 `to_free_ptr` 是定长 `[_; MAX_NUMBER_OF_WORKER]`（256，`common/constants.h:20`）；
//!   Rust 用与 `workers` 等长的切片，无 256 上限，也不复刻上游的定长 worker 数组。
//! - 超出 `workers` 的 `wid` 会 panic（上游越界是 UB）；属调用方契约。

use std::{
  any::Any,
  fmt,
  mem::take,
  sync::atomic::{AtomicU64, Ordering, fence},
};

use parking_lot::Mutex;

/// 本地 epoch 哨兵：worker 不在保护区间（上游 `EpochHandler::MAX_VALUE = ~0ULL`，`epoch_handler.h:14`）
const MAX_VALUE: u64 = u64::MAX;
/// 推进采样率：`Rand(200) % 200 == 0`（`epoch_handler.cc:17`）
const ADVANCE_RATE: u64 = 200;
/// 回收采样率：`Rand(100) % 100 == 0`（`epoch_handler.cc:20`）
const RECLAIM_RATE: u64 = 100;

/// 延迟释放条目：类型擦除指针 + 入队时的本地 epoch
struct Deferred {
  ptr: Box<dyn Any + Send>,
  epoch: u64,
}

impl Deferred {
  /// 就地释放（消费条目）
  fn release(self) {
    drop(self.ptr);
  }
}

/// 缓存行独占包装：相邻 `wid` 的可写状态不得共享缓存行。
///
/// `Mutex<Vec<Deferred>>` 只有 32B、`AtomicU64` 只有 8B；若连续排列，相邻 `wid` 的锁状态 /
/// 本地 epoch 落在同一缓存行，两个 worker 并发写各自的槽位会互相失效缓存行。x86_64/aarch64
/// 用 128B（同时规避相邻行预取），其它平台用 64B。
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
  not(any(target_arch = "x86_64", target_arch = "aarch64")),
  repr(align(64))
)]
struct Padded<T>(T);

/// 多 worker epoch 回收器
pub struct EpochHandler {
  /// 全局 epoch（概率性单调推进，`epoch_handler.h:24`）
  global_epoch: AtomicU64,
  /// 各 worker 本地 epoch 快照（`epoch_handler.h:25`），独占缓存行
  local_epoch: Box<[Padded<AtomicU64>]>,
  /// 各 worker 待释放队列（`epoch_handler.h:26`）：每 `wid` 一把锁，跨 `wid` 无竞争
  to_free: Box<[Padded<Mutex<Vec<Deferred>>>]>,
}

impl fmt::Debug for EpochHandler {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("EpochHandler")
      .field("global_epoch", &self.global_epoch.load(Ordering::Relaxed))
      .field("workers", &self.local_epoch.len())
      .finish()
  }
}

impl EpochHandler {
  /// 构造（worker 数固定）
  pub fn new(workers: usize) -> Self {
    Self {
      global_epoch: AtomicU64::new(0),
      local_epoch: (0..workers)
        .map(|_| Padded(AtomicU64::new(MAX_VALUE)))
        .collect(),
      to_free: (0..workers)
        .map(|_| Padded(Mutex::new(Vec::new())))
        .collect(),
    }
  }

  /// 进入本 worker 的保护区间：本地 epoch 对齐全局（上游 `EpochGuard` 构造，`epoch_handler.h:31-33`）
  #[inline]
  pub fn enter(&self, wid: usize) {
    let g = self.global_epoch.load(Ordering::SeqCst);
    self.local_epoch[wid].0.store(g, Ordering::Relaxed);
    // 见模块文档「内存序」：发布本地 epoch 后才能读取受保护对象
    fence(Ordering::SeqCst);
  }

  /// 离开保护区间：本地 epoch 置哨兵，不再阻碍回收（上游 `EpochGuard` 析构，`epoch_handler.h:35`）
  #[inline]
  pub fn leave(&self, wid: usize) {
    self.local_epoch[wid].0.store(MAX_VALUE, Ordering::SeqCst);
  }

  /// 周期操作：概率性推进全局 epoch 并回收已安全释放的指针（上游 `epoch_handler.cc:15-31`）
  ///
  /// `rng(bound)` 必须返回 `[0, bound)` 内的均匀采样；`bound` 依次为 200、100。
  pub fn epoch_operation(&self, wid: usize, rng: &mut impl FnMut(u64) -> u64) {
    // 推进（`epoch_handler.cc:17`）
    if rng(ADVANCE_RATE) == 0 {
      self.global_epoch.fetch_add(1, Ordering::SeqCst);
    }
    // 回收：min(local_epoch) 之前的 epoch 已无人可见（`epoch_handler.cc:20-30`）
    if rng(RECLAIM_RATE) == 0 {
      let mut min_epoch = MAX_VALUE;
      for e in &*self.local_epoch {
        min_epoch = min_epoch.min(e.0.load(Ordering::SeqCst));
      }
      // 与上游一致：从头扫描，遇到第一个不可回收条目即停（`epoch_handler.cc:24-28`）。
      // 队列只保证「前缀」按入队顺序可回收：允许在保护区间外入队（epoch=哨兵），
      // 该条目会阻断其后的条目，因此整体不一定有序，不能用二分（partition_point）。
      //
      // 先在锁内摘下可回收前缀，再在锁外 drop：延迟对象的析构器在锁内执行时若回调本
      // handler 的同一 wid 会死锁。整段前缀可回收（常见稳态）用 mem::take 零分配；只有
      // 被哨兵条目阻断的部分回收才 collect 一次，为析构器换队列锁外的执行环境。
      let mut q = self.to_free[wid].0.lock();
      let mut idx = 0;
      while idx < q.len() && q[idx].epoch < min_epoch {
        idx += 1;
      }
      if idx == 0 {
        return;
      }
      let expired = if idx == q.len() {
        take(&mut *q)
      } else {
        q.drain(..idx).collect()
      };
      drop(q);
      for d in expired {
        d.release();
      }
    }
  }

  /// 延迟释放：记录入队时的本地 epoch（上游 `epoch_handler.cc:33-35`）
  ///
  /// 必须在 `enter` 之后调用：在保护区间外入队会记录哨兵 `MAX_VALUE`，该条目会阻断其所在
  /// 队列的前缀回收，直到 handler 析构（与上游 `break` 语义一致）。
  pub fn defer_free_pointer(&self, wid: usize, ptr: Box<dyn Any + Send>) {
    let epoch = self.local_epoch[wid].0.load(Ordering::SeqCst);
    self.to_free[wid].0.lock().push(Deferred { ptr, epoch });
  }

  /// 全局 epoch 读数（对应上游 public 字段 `global_epoch`，`epoch_handler.h:24`）
  #[inline]
  pub fn global(&self) -> u64 {
    self.global_epoch.load(Ordering::SeqCst)
  }

  /// 本地 epoch 读数（对应上游 public 字段 `local_epoch`，`epoch_handler.h:25`）
  #[inline]
  pub fn local(&self, wid: usize) -> u64 {
    self.local_epoch[wid].0.load(Ordering::SeqCst)
  }
}
