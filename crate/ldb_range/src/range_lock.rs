//! 一维区间互斥锁（移植自 `leanstore.latency/src/sync/range_lock.cc` 的跳表实现）。
//!
//! 上游以 `HybridLatch`(L1) + 裸指针 flexible-array 节点 + `EpochHandler` 延迟回收实现；
//! L0 不得依赖 L1，本实现做语义等价的改写：
//!
//! - 整棵跳表由 `parking_lot::Mutex` 保护，全部操作取 `&self`（类型 `Send + Sync`，
//!   调用方无需再套外层锁；区间锁操作是短临界区）。
//! - 节点放入 arena，前向指针用**定长** `[NodeId; MAX_LEVEL]`，消除每次插入的一次堆分配，
//!   也不引入自引用 `unsafe` 与 `free` 生命周期问题。
//! - 上游 `DeferFreePointer`（EBR 延迟释放，`range_lock.cc:127`）在整树互斥下不必要：
//!   没有并发读者能看到中间态，槽位在 `unlock_range` 内立即回到 free 列表。
//! - 上游两个函数末尾的 `epoch_.EpochOperation(...)` 写在 `while (true)` 之后，是**不可达
//!   死代码**（`range_lock.cc:88`、`range_lock.cc:135`）；本实现在 `unlock_range` 内真实调用
//!   一次 `epoch_operation` 推进全局 epoch（`wid` 供 epoch 队列索引，见 `RangeLock::unlock_range`）。
//!
//! 语义：`start > 0` 且 `[start, start+len-1]` 与任何已锁区间不相交才可加锁。
//!
//! 与上游的有意偏差（逐条依据见报告 §3）：
//! 1. 右端点由上游的 `start + len - 1`（u64 回绕；且当 `end == MAX_VALUE` 时上游 `FindNodes`
//!    会把 `p` 推到 tail 再读其 `nullptr` 前向指针，即空指针解引用 UB）改为**饱和相加再减一**
//!    （结果恒 `<= MAX_VALUE - 1`），`try_lock_range` 与 `test_range` 共用同一个 `range_end`
//!    （此前二者不一致：前者饱和、后者裸加）。
//! 2. `unlock_range` 对未持有的 `start` 静默返回；上游 `Ensure(q->start == start)`
//!    （`range_lock.cc:125`）会直接终止进程。Rust 侧 panic 会打进调用方事务路径
//!    （`ldb_free::public_free_extents`），故按契约返回。特别的，`start == MAX_VALUE` 时
//!    上游严格 `<` 前驱遍历不会踏上 tail（`range_lock.cc:118`），但 `q` 仍会是 tail 且
//!    `Ensure(tail->start == MAX_VALUE)` 通过；二次解锁会把 tail 摘链并 `free`，结构损坏
//!    （复现 `L MAX 0; U MAX; U MAX; T MAX 0` → 上游 SIGSEGV）。本仓先判 `q == tail`
//!    返回，二次解锁是 no-op。
//! 3. 层收缩条件写“`head.forward[level] == tail`”才生效；上游写 `== nullptr`
//!    （`range_lock.cc:130`），而构造期已把 `root_->forward[i]` 全填成 tail（`:14`），
//!    故上游该循环恒假（层只增不减）。
//! 4. 层增长时把新节点同时连到新顶层；上游 `lv = ++level_; nodes[lv] = root_;`
//!    （`range_lock.cc:51-54`）只加层高、循环却是 `k < lv`，新顶层留空，靠下一次增长补齐。
//!    上游因收缩恒假而安全；本仓收缩真实生效（偏差 3），照搬会让层高在每次 unlock 后冻结
//!    （活跃 5e5 区间、插入/解锁交错时层高 9，纯插入为 19），查找退化到 O(n/2^level)。
//!    层数分布与层高都不改变任何可观测语义（对拍 15 万步 0 失配）。

use std::{collections::VecDeque, fmt};

use fastrand::Rng;
use ldb_epoch::EpochHandler;
use parking_lot::Mutex;

/// 跳表最高层（上游 `MAX_LEVEL`，`range_lock.h:25`）
const MAX_LEVEL: usize = 32;
/// 值域上界哨兵（上游 `MAX_VALUE`，`range_lock.h:26`）
const MAX_VALUE: u64 = u64::MAX;

/// arena 内节点 ID
type NodeId = u32;
/// arena 节点数上限：`NodeId` 为 `u32`，且末位 `NodeId::MAX` 保留为 tail 哨兵的空前向值，
/// 故真实节点 id 只能取 `0..NodeId::MAX`，最多 `NodeId::MAX` 个节点（≈576 GiB arena）。
const MAX_NODES: usize = NodeId::MAX as usize;

/// 跳表节点（前向指针定长内联，插入不再分配）
#[derive(Debug)]
struct Node {
  start: u64,
  end: u64,
  forward: [NodeId; MAX_LEVEL],
}

#[derive(Debug)]
struct Inner {
  arena: Vec<Node>,
  /// 空闲槽（删除后复用）
  free: VecDeque<NodeId>,
  head: NodeId,
  tail: NodeId,
  level: usize,
  rng: Rng,
}

impl Inner {
  fn alloc_node(&mut self, start: u64, end: u64) -> NodeId {
    let tail = self.tail;
    let node = Node {
      start,
      end,
      forward: [tail; MAX_LEVEL],
    };
    if let Some(id) = self.free.pop_front() {
      self.arena[id as usize] = node;
      id
    } else {
      // 容量检查必须在 `push` **之前**：panic 被 `catch_unwind` 捕获时，arena/链/free
      // 必须保持调用前的状态（parking_lot 无 poisoning，其他线程随后会重新拿到锁）。
      let id = Self::append_id(self.arena.len() + 1);
      self.arena.push(node);
      id
    }
  }

  /// 追加节点的 id（仅 append 路径：稳态复用 free 槽不经过这里）。
  ///
  /// 入参是 `push` 之后的 arena 长度；返回值是新节点的 id。调用点必须在 `push` 前调用，
  /// 使容量 panic 不留下部分更新（旧实现先 `push` 再检查，捕获 panic 后会留下一个
  /// 未链接、未入 free 的槽位，且 `level` 可能已被增长分支自增）。
  ///
  /// `NodeId` 是 `u32`：节点数超过 `MAX_NODES` 后，朴素 `(len - 1) as NodeId` 会先占用 tail
  /// 哨兵值 `NodeId::MAX`，再回绕成 0（head）⇒ 跳表被静默破坏。此处显式 panic（可预测分支；
  /// `Vec::push` 本身已含容量检查）。
  fn append_id(len: usize) -> NodeId {
    assert!(
      len <= MAX_NODES,
      "ldb_range arena 节点数超过 u32 索引上限 {MAX_NODES}（≈576 GiB）；请改 NodeId 为 u64 或分段"
    );
    (len - 1) as NodeId
  }

  /// 每层找 start <= key 的最大前驱（上游 FindNodes，`range_lock.cc:39-46`）
  ///
  /// 前置条件：`key <= MAX_VALUE - 1`（`range_end` 的封顶保证）。据此 tail 哨兵
  /// （`start == MAX_VALUE`）永不满足 `start <= key`，循环不会踏上 tail —— 因此不需要
  /// 额外的 `q != tail` 比较（上游在 `key == MAX_VALUE` 时会把 `p` 推到 tail 再读其
  /// `nullptr` 前向指针，即空指针解引用 UB，属上游缺陷；本实现从键域上排除该情形）。
  ///
  /// `out` 预填成 head：高于当前层高的槽位不会被遍历，head 正好是这些层的空前驱。
  fn find_nodes(&self, key: u64, out: &mut [NodeId; MAX_LEVEL]) {
    let mut p = self.head;
    for k in (0..=self.level).rev() {
      loop {
        let q = self.arena[p as usize].forward[k];
        if self.arena[q as usize].start <= key {
          p = q;
        } else {
          break;
        }
      }
      out[k] = p;
    }
  }

  /// 按 nodes 前驱插入新区间（上游 InsertRange，`range_lock.cc:48-63`）
  fn insert_range(&mut self, nodes: &[NodeId; MAX_LEVEL], start: u64, end: u64) {
    // 层数取几何分布（p=1/2）：`P(lv >= k) = 2^(1-k)`，第 k 层链表期望 n/2^k 个节点。
    // 上游写 `GetRand<u8>(1, MAX_LEVEL)` 即 [1,31] **均匀**（`common/rand.h` 注释），
    // 于是 `P(lv >= k) = (32-k)/31`，每层链表都接近 n 个节点，查找退化为 O(n)：
    // 实测纯查找（test_range，无分配）n=1e3/1e4/1e5 = 96.9/453.5/9827 ns（报告 C2/P1）。
    // 属有意偏离上游的内部细节（层数不改变任何可观测语义）。
    let mut lv = 1usize;
    while lv < MAX_LEVEL && self.rng.bool() {
      lv += 1;
    }
    // 增长判定与层高自增必须放在 alloc_node 之后：容量耗尽时 alloc_node panic，
    // 而层高、arena、链都不得被改动（parking_lot 无 poisoning，其他线程随后会看到该状态）。
    // 新顶层（旧 level + 1）的 nodes 槽位是预填的 head。
    let grow = lv > self.level + 1;
    if grow {
      lv = self.level + 2;
    }
    let q = self.alloc_node(start, end);
    if grow {
      self.level += 1;
    }
    for k in 0..lv {
      let p = nodes[k];
      let fwd = self.arena[p as usize].forward[k];
      self.arena[q as usize].forward[k] = fwd;
      self.arena[p as usize].forward[k] = q;
    }
  }
}

/// 区间右端（含）。见文件头偏差 1。
///
/// `saturating_add` 后 `saturating_sub(1)` 恒 `<= MAX_VALUE - 1`（加法已封在 `MAX_VALUE`），
/// 无需再 `min(MAX_VALUE - 1)`。该上界正是 tail 哨兵（`start == MAX_VALUE`）不会被 find_nodes
/// 的键命中的依据。
#[inline]
fn range_end(start: u64, len: u64) -> u64 {
  start.saturating_add(len).saturating_sub(1)
}

/// 跳表区间锁（`Send + Sync`，可跨线程共享）
pub struct RangeLock {
  inner: Mutex<Inner>,
  epoch: EpochHandler,
  workers: usize,
}

impl fmt::Debug for RangeLock {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RangeLock")
      .field("nodes", &self.inner.lock().arena.len())
      .field("workers", &self.workers)
      .field("epoch", &self.epoch)
      .finish_non_exhaustive()
  }
}

impl RangeLock {
  /// 构造（`workers` 为 epoch 回收队列宽度，必须为正）
  pub fn new(workers: usize) -> Self {
    assert!(
      workers > 0,
      "worker 数必须为正（epoch 回收队列按 wid 索引）"
    );
    let mut inner = Inner {
      arena: Vec::new(),
      free: VecDeque::new(),
      head: 0,
      tail: 1,
      level: 0,
      rng: Rng::new(),
    };
    // 上游构造：root_/tail_ 两个哨兵，root_ 的每层前向都指向 tail_（`range_lock.cc:12-15`）
    inner.arena.push(Node {
      start: 0,
      end: 0,
      forward: [1; MAX_LEVEL],
    });
    inner.arena.push(Node {
      start: MAX_VALUE,
      end: MAX_VALUE,
      forward: [NodeId::MAX; MAX_LEVEL],
    });
    Self {
      inner: Mutex::new(inner),
      epoch: EpochHandler::new(workers),
      workers,
    }
  }

  /// 尝试锁 `[start, start+len-1]`：与已锁区间相交则失败（上游 TryLockRange，`range_lock.cc:67`）
  pub fn try_lock_range(&self, _wid: usize, start: u64, len: u64) -> bool {
    assert!(
      start > 0,
      "区间起点必须为正（上游 Ensure，range_lock.cc:68）"
    );
    let end = range_end(start, len);
    let mut inner = self.inner.lock();
    let mut nodes = [inner.head; MAX_LEVEL];

    inner.find_nodes(end, &mut nodes);
    // 只有 level-0 前驱可能是相交者：区间互不相交，后继的 start 必 > end
    if inner.arena[nodes[0] as usize].end >= start {
      return false;
    }
    inner.insert_range(&nodes, start, end);
    true
  }

  /// 测试 `[start, start+len-1]` 是否可锁（无副作用，上游 TestRange，`range_lock.cc:91`）
  pub fn test_range(&self, _wid: usize, start: u64, len: u64) -> bool {
    assert!(
      start > 0,
      "区间起点必须为正（上游 Ensure，range_lock.cc:92）"
    );
    let end = range_end(start, len);
    let inner = self.inner.lock();
    let mut nodes = [inner.head; MAX_LEVEL];
    inner.find_nodes(end, &mut nodes);
    inner.arena[nodes[0] as usize].end < start
  }

  /// 解锁起点为 `start` 的区间（槽位立即回收到 free 列表；未持有时静默返回，见文件头偏差 2）
  pub fn unlock_range(&self, wid: usize, start: u64) {
    assert!(
      start > 0,
      "区间起点必须为正（上游 Ensure，range_lock.cc:107）"
    );
    let mut inner = self.inner.lock();
    let mut preds = [inner.head; MAX_LEVEL];

    // 每层找 start 的严格前驱。tail 哨兵 start == MAX_VALUE 且调用方保证 start <= MAX_VALUE，
    // `MAX_VALUE < start` 恒假，故前驱遍历不会踏上 tail。但 start == MAX_VALUE 时 `q` 仍可能
    // 等于 tail（不存在该区间），下方 `q == tail` 必须先行返回：上游缺这一步，二次解锁会摘掉
    // tail 并 `free`（复现见文件头偏差 2）。
    let mut p = inner.head;
    for k in (0..=inner.level).rev() {
      loop {
        let q = inner.arena[p as usize].forward[k];
        if inner.arena[q as usize].start < start {
          p = q;
        } else {
          break;
        }
      }
      preds[k] = p;
    }
    let q = inner.arena[preds[0] as usize].forward[0];
    if q == inner.tail || inner.arena[q as usize].start != start {
      return;
    }

    // 逐层摘除：自 level 0 向上，前驱在该层的 forward 不再指向 q 即止
    // （上游 `range_lock.cc:126` 同构；只摘 level 0 会让高层前向残留指向已回收槽位，
    // 槽位复用后跳表被复用节点 id 污染，unlock_relock_slot_reuse_and_deep_levels 即此症状）
    let mut k = 0usize;
    while k <= inner.level {
      let pk = preds[k];
      if inner.arena[pk as usize].forward[k] == q {
        inner.arena[pk as usize].forward[k] = inner.arena[q as usize].forward[k];
        k += 1;
      } else {
        break;
      }
    }
    inner.free.push_back(q);

    // 收缩空层（上游该循环恒假，见文件头偏差 3）
    while inner.level > 0 && inner.arena[inner.head as usize].forward[inner.level] == inner.tail {
      inner.level -= 1;
    }

    // epoch 周期操作。ldb_epoch 的回收队列按 wid 索引（`epoch.rs:154`），越界会 panic；
    // 本 crate 从不 `defer_free_pointer`，具体 wid 无语义影响，故钳到有效范围。
    // ldb_epoch 冻结契约要求 `epoch_operation` 不持外层锁（`.agents/reviews/DONE.md`：
    // 「defer 在摘链后、epoch_operation 在释放外层锁后」）：采样在锁内取（RNG 属于 inner），
    // 推进/回收在 `drop(inner)` 之后执行，避免与 epoch 队列锁嵌套。
    // `global_epoch` 为 u64 且按 1/200 概率自增：回绕需 2^64 × 200 ≈ 3.7e21 次 unlock，
    // 即使 1e9 次/s 也要 ≈1.2e5 年。本 crate 不 defer、回收队列恒空 ⇒ 回绕无悬垂引用。
    // 若将来接线 defer，回绕有两个理论后果（登记 NOTES，不阻塞）：① 全局 epoch 经过
    // u64::MAX 的那一个世代（~200 次 unlock），local_epoch 的哨兵值 MAX_VALUE 与「在保护
    // 区间内」重合，该世代失去 EBR 保护（上游同为哨兵设计）；② 回绕后 `epoch < min_epoch`
    // 对回绕前条目恒假，该队列前缀永久滞留（内存滞留，非 UB）。
    let samples = [inner.rng.u64(..), inner.rng.u64(..)];
    let wid = wid.min(self.workers - 1);
    drop(inner);
    let mut i = 0usize;
    self.epoch.epoch_operation(wid, &mut |bound| {
      let x = samples[i];
      i += 1;
      x % bound.max(1)
    });
  }
}

/// 内部不变量测试：直接读 `Inner` 的 arena/free/level（上游语义测试在 `tests/`）。
#[cfg(test)]
#[path = "range_lock_internal_test.rs"]
mod internal_test;
