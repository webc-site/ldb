//! Hook 层形状与分发（`todo.md:232`：热路径泛型、冷路径可 `dyn`）。
//!
//! 本用例钉死三件事：`TxnLogHook`/`EvictHook` 可用具体类型静态分发；
//! `GcHook` 保持对象安全（`Box<dyn GcHook>`，`SpaceManager` 以
//! `Rc<dyn GcHook>` 注入）；五个方法的异步签名可被调用。

use std::{
  cell::Cell,
  future::Future,
  hint::spin_loop,
  pin::{Pin, pin},
  task::{Context, Poll, Waker},
};

use ldb_page::{EvictHook, GcHook, TxnLogHook};
use ldb_type::{BlockId, PageId};

#[derive(Debug, Default)]
struct Mock {
  evicted: Vec<PageId>,
  logged: Cell<u32>,
}

impl EvictHook for Mock {
  fn evict_extent(&mut self, extent: PageId) -> impl Future<Output = ()> {
    self.evicted.push(extent);
    async {}
  }

  fn public_free_ranges(&mut self) -> impl Future<Output = Vec<(u64, u64)>> {
    let ranges = vec![(0, 1), (4, 2)];
    async move { ranges }
  }
}

impl GcHook for Mock {
  fn on_block_full(&self, _bid: BlockId) -> Pin<Box<dyn Future<Output = ()> + '_>> {
    Box::pin(async {})
  }

  fn on_gc_needed(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
    Box::pin(async { true })
  }
}

impl TxnLogHook for Mock {
  fn log_pid2_offset(&self, _pid: PageId) -> impl Future<Output = ()> {
    self.logged.set(self.logged.get() + 1);
    async {}
  }

  fn log_valid_page(&self, _pid: PageId) -> impl Future<Output = ()> {
    self.logged.set(self.logged.get() + 1);
    async {}
  }

  fn log_free_page(&self, _pid: PageId) -> impl Future<Output = ()> {
    self.logged.set(self.logged.get() + 1);
    async {}
  }
}

fn block_on<F: Future>(fut: F) -> F::Output {
  let mut fut = pin!(fut);
  let waker = Waker::noop();
  let mut cx = Context::from_waker(waker);
  loop {
    match fut.as_mut().poll(&mut cx) {
      Poll::Ready(v) => return v,
      Poll::Pending => spin_loop(),
    }
  }
}

#[test]
fn hooks_dispatch_generically_and_gc_hook_is_object_safe() {
  let mut mock = Mock::default();
  block_on(mock.evict_extent(7));
  assert_eq!(mock.evicted, [7]);
  assert_eq!(block_on(mock.public_free_ranges()), [(0, 1), (4, 2)]);

  block_on(mock.log_pid2_offset(11));
  block_on(mock.log_valid_page(12));
  block_on(mock.log_free_page(13));
  assert_eq!(mock.logged.get(), 3);

  // 对象安全：SpaceManager 以 `Rc<dyn GcHook>` 单点注入
  let dyn_gc: Box<dyn GcHook> = Box::new(Mock::default());
  block_on(dyn_gc.on_block_full(3));
  assert!(block_on(dyn_gc.on_gc_needed()));
}
