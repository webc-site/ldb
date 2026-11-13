//! epoch 延迟回收回归：采样边界、保护区间、多 worker 取 min、前缀回收顺序、析构释放、并发。

use std::{
  any::Any,
  sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
    mpsc,
  },
  thread,
  time::Duration,
};

use ldb_epoch::EpochHandler;

/// drop 探针：记录「是否已释放 / 释放顺序」
#[derive(Clone)]
struct DropLog {
  dropped: Arc<AtomicU64>,
  order: Arc<AtomicU64>,
  seq: Arc<AtomicU64>,
}

impl DropLog {
  fn new(seq: &Arc<AtomicU64>) -> Self {
    Self {
      dropped: Arc::new(AtomicU64::new(0)),
      order: Arc::new(AtomicU64::new(0)),
      seq: Arc::clone(seq),
    }
  }

  fn probe(&self) -> Box<dyn Any + Send> {
    Box::new(Probe(self.clone()))
  }

  fn dropped(&self) -> bool {
    self.dropped.load(Ordering::SeqCst) != 0
  }

  fn order(&self) -> u64 {
    self.order.load(Ordering::SeqCst)
  }
}

struct Probe(DropLog);

impl Drop for Probe {
  fn drop(&mut self) {
    let seq = self.0.seq.fetch_add(1, Ordering::SeqCst);
    self.0.order.store(seq, Ordering::SeqCst);
    self.0.dropped.store(1, Ordering::SeqCst);
  }
}

/// 可控采样器：每次 epoch_operation 的第 1 次调用返回 advance、第 2 次返回 reclaim；
/// 0 触发、非 0 跳过。与 bound 数值解耦，便于单独钉死采样率常量。
struct Mock {
  advance: u64,
  reclaim: u64,
  calls: usize,
  bounds: Vec<u64>,
}

impl Mock {
  fn new() -> Self {
    Self {
      advance: 1,
      reclaim: 1,
      calls: 0,
      bounds: Vec::new(),
    }
  }

  fn next(&mut self, bound: u64) -> u64 {
    self.bounds.push(bound);
    let value = if self.calls % 2 == 0 {
      self.advance
    } else {
      self.reclaim
    };
    self.calls += 1;
    value
  }

  fn op(&mut self, h: &EpochHandler, wid: usize) {
    h.epoch_operation(wid, &mut |bound| self.next(bound));
  }
}

#[test]
fn sampling_and_protection() {
  let h = EpochHandler::new(2);
  let seq = Arc::new(AtomicU64::new(0));
  let log = DropLog::new(&seq);
  let mut mock = Mock::new();

  // 一次 epoch_operation 依次采样 200、100；推进采样非 0 不得推进
  mock.advance = 1;
  mock.reclaim = 1;
  mock.op(&h, 0);
  assert_eq!(mock.bounds.as_slice(), &[200, 100], "采样上界与顺序");
  assert_eq!(h.global(), 0, "推进采样非 0 不得推进");

  // 推进采样为 0：全局 epoch 单调 +1
  mock.advance = 0;
  mock.op(&h, 0);
  assert_eq!(h.global(), 1);

  // enter 前本地 epoch 为哨兵（构造即哨兵，不走上游的值初始化 0）
  assert_eq!(h.local(1), u64::MAX, "构造后本地 epoch 为哨兵");

  // enter 后本地 epoch 对齐全局
  h.enter(1);
  assert_eq!(h.local(1), 1, "worker1 进入后对齐全局 epoch");
  h.enter(0);
  assert_eq!(h.local(0), 1, "worker0 进入后对齐全局 epoch");

  // 入队 epoch = 入队时的本地 epoch（1）；leave 后置哨兵
  h.defer_free_pointer(0, log.probe());
  h.leave(0);
  assert_eq!(h.local(0), u64::MAX, "leave 后本地 epoch 置哨兵");

  // 回收采样返回非 0：即使满足条件也不得回收
  mock.advance = 1;
  mock.reclaim = 1;
  mock.op(&h, 0);
  assert!(!log.dropped(), "回收采样非 0 不得回收");

  // min(local_epoch)=1 与入队 epoch=1 相等，不满足 epoch < min
  mock.reclaim = 0;
  mock.op(&h, 0);
  assert!(!log.dropped(), "保护区间内不得回收");

  // 离开后本地 epoch 置哨兵，不再阻碍回收
  h.leave(1);
  assert_eq!(h.local(1), u64::MAX);
  mock.op(&h, 0);
  assert!(log.dropped(), "全部离开后应回收");
}

#[test]
fn multi_worker_min() {
  let h = EpochHandler::new(3);
  let seq = Arc::new(AtomicU64::new(0));
  let (a, b, c) = (DropLog::new(&seq), DropLog::new(&seq), DropLog::new(&seq));
  let mut mock = Mock::new();
  mock.advance = 1;
  mock.reclaim = 0;

  // 阶段 1：worker1/2 同时保护，逐个离开；min 必须覆盖全部 worker 槽位
  h.enter(1);
  h.enter(2);
  h.enter(0);
  h.defer_free_pointer(0, a.probe());
  h.leave(0);

  mock.op(&h, 0);
  assert!(!a.dropped(), "worker1/2 均在保护 epoch 0");
  h.leave(1);
  mock.op(&h, 0);
  assert!(!a.dropped(), "worker2 仍保护 epoch 0");
  h.leave(2);
  mock.op(&h, 0);
  assert!(a.dropped(), "所有 worker 离开后才可回收");

  // 阶段 2：反向离开顺序，排除“只读某个固定槽位”的实现
  h.enter(1);
  h.enter(2);
  h.enter(0);
  h.defer_free_pointer(0, b.probe());
  h.leave(0);
  h.leave(2);
  mock.op(&h, 0);
  assert!(!b.dropped(), "worker1 仍保护 epoch 0");
  h.leave(1);
  mock.op(&h, 0);
  assert!(b.dropped(), "最后一个保护者离开后才可回收");

  // 阶段 3：调用者自身在保护区间内时，min 必须包含调用者自己的槽位
  h.enter(1);
  h.defer_free_pointer(1, c.probe());
  mock.op(&h, 1);
  assert!(!c.dropped(), "调用者自身在保护区间内不得回收");
  h.leave(1);
  mock.op(&h, 1);
  assert!(c.dropped(), "调用者离开后才可回收");
}

#[test]
fn prefix_fifo_and_epoch_tags() {
  let h = EpochHandler::new(4);
  let seq = Arc::new(AtomicU64::new(0));
  let (a, b, c) = (DropLog::new(&seq), DropLog::new(&seq), DropLog::new(&seq));
  let mut mock = Mock::new();

  // 三次进入/离开之间推进全局 epoch：入队 epoch 依次 0、1、2
  h.enter(0);
  h.defer_free_pointer(0, a.probe());
  h.leave(0);
  mock.advance = 0;
  mock.reclaim = 1;
  mock.op(&h, 3);

  h.enter(0);
  h.defer_free_pointer(0, b.probe());
  h.leave(0);
  mock.op(&h, 3);

  h.enter(0);
  h.defer_free_pointer(0, c.probe());
  h.leave(0);

  // worker1 进入于 epoch 2，min=2：只能回收前缀 tag 0、1，且按 FIFO
  h.enter(1);
  mock.advance = 1;
  mock.reclaim = 0;
  mock.op(&h, 0);
  assert!(a.dropped(), "tag 0 应回收");
  assert!(b.dropped(), "tag 1 应回收");
  assert!(!c.dropped(), "tag 2 不满足 epoch < min");
  assert!(a.order() < b.order(), "前缀按入队顺序释放");
}

#[test]
fn queue_scan_stops_at_first_unreclaimable() {
  let h = EpochHandler::new(1);
  let seq = Arc::new(AtomicU64::new(0));
  let first = DropLog::new(&seq);
  let second = DropLog::new(&seq);
  let mut mock = Mock::new();
  mock.reclaim = 0;

  // 未进入保护区间时入队：epoch 记录为哨兵，阻断其后所有条目（上游 break 语义）
  h.defer_free_pointer(0, first.probe());
  h.enter(0);
  h.defer_free_pointer(0, second.probe());
  h.leave(0);
  mock.op(&h, 0);
  assert!(!first.dropped(), "哨兵 epoch 不满足回收条件");
  assert!(!second.dropped(), "扫描在首个不可回收条目处停止");

  // 析构释放残留（上游 EpochHandler 析构 free 全部 to_free_ptr）
  drop(h);
  assert!(first.dropped(), "析构应释放残留条目");
  assert!(second.dropped(), "析构应释放残留条目");
}

#[test]
fn concurrent_workers_reclaim_independently() {
  let h = Arc::new(EpochHandler::new(4));
  let count = Arc::new(AtomicUsize::new(0));
  thread::scope(|scope| {
    for wid in 0..4 {
      let h = Arc::clone(&h);
      let count = Arc::clone(&count);
      scope.spawn(move || {
        for _ in 0..25 {
          h.enter(wid);
          h.defer_free_pointer(wid, Box::new(Counted(Arc::clone(&count))));
          h.leave(wid);
        }
      });
    }
  });

  // 全部 worker 已离开：min=MAX，所有 tag=0 的条目均可回收。
  // 逐个 wid 回收，每次只能释放该 wid 的 25 个条目（队列按 wid 独立）。
  assert_eq!(count.load(Ordering::SeqCst), 0);
  for wid in 0..4 {
    h.epoch_operation(wid, &mut |_| 0);
    assert_eq!(
      count.load(Ordering::SeqCst),
      (wid + 1) * 25,
      "每个 worker 的队列各自回收"
    );
  }
}

struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
  fn drop(&mut self) {
    self.0.fetch_add(1, Ordering::SeqCst);
  }
}

/// 延迟对象的析构器回调同一 handler 的同一 wid 时不得死锁：
/// `epoch_operation` 必须在队列锁之外执行 drop。
#[test]
fn drop_may_reenter_same_wid() {
  struct Reenter {
    h: Arc<EpochHandler>,
    wid: usize,
    done: mpsc::Sender<()>,
  }

  impl Drop for Reenter {
    fn drop(&mut self) {
      self.h.defer_free_pointer(self.wid, Box::new(0u64));
      let _ = self.done.send(());
    }
  }

  let h = Arc::new(EpochHandler::new(1));
  let (tx, rx) = mpsc::channel();
  h.enter(0);
  h.defer_free_pointer(
    0,
    Box::new(Reenter {
      h: Arc::clone(&h),
      wid: 0,
      done: tx,
    }),
  );
  h.leave(0);

  let worker = Arc::clone(&h);
  let _ = thread::spawn(move || worker.epoch_operation(0, &mut |_| 0));
  // 若 drop 在队列锁内执行，析构器重入同 wid 会死锁，本断言在 2s 后失败
  assert!(
    rx.recv_timeout(Duration::from_secs(2)).is_ok(),
    "析构器回调同 wid 不得死锁"
  );
}
