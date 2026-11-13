//! GC 策略回归：6 策略可达性 + C-14/C-15/C-24/C-29/C-31 修复验证 +
//! 列表状态机边界 + 影子模型差分（1e5 随机状态对拍）。
//!
//! 期望值一律用字面量（禁自我参照）。

use ldb_gc::{GarbageCollector, GcStrategy};
use ldb_space::{BlockMetadata, BlockState, MutexVec, SpaceManager};
use ldb_type::{BlockId, LogId, MIN_COMP_SIZE, PAGE_SIZE};

fn mk(blocks: u64) -> SpaceManager {
  SpaceManager::new(blocks, 64, 2048, 4, blocks * 64, 2)
}

/// 占用 n 个块关闭至 FULL，并注入递增死页分布（第 i 块有效页 = 8 - (i%4)*2）。
///
/// 修复（R1 红）：`close_one_open_block(bid, max_w_ptr)` 已按上游
/// `CloseOneOpenBlock`（`space_manager.cc:1273-1301`，末行 `Ensure(state==FULL)`）
/// 与 `IssuePaddingWritesToFullBlock`（`:1426` `MoveBlockToList(IN_USE, FULL)`；
/// `:1381-1385` 为「已 FULL」快路径）把满尺寸块直接置 `Full`（无 FullPad 中间态）；
/// 旧断言 `move_padded_blocks_to_full_list` 依赖的是已废弃的倒置语义（满块→FullPad），
/// 现仅对 `FullPad` 源态生效，故对满块必返回 false。删该陈旧断言。
///
/// 注：`select_empty_block_id` 取 EMPTY 队首，`StorageSpace::new` 逆序压栈 ⇒
/// 首次占用拿到最大 bid（n 块时为 n-1），依次递减。
fn occupy(mgr: &SpaceManager, n: u64) {
  for i in 0..n {
    let bid = mgr.select_empty_block_id(0, 42);
    assert!(bid < 64);
    assert!(mgr.move_block_to_list(bid, BlockState::Open, BlockState::InUse));
    assert!(
      mgr.close_one_open_block(bid, 64 * PAGE_SIZE as u64),
      "关闭失败 bid={bid}"
    );
    assert_eq!(
      mgr.block_meta(bid).state,
      BlockState::Full,
      "满尺寸关闭应直接入 FULL（上游 :1381-1385）"
    );
    // 模拟真实写入：8 页/块，循环失效 0/2/4/6 页（下标回绕避免 > total）
    let mut s = mgr.sspace.write();
    let m = &mut s.block_metadata[bid as usize];
    m.total_pid_cnt = 8;
    m.invalidated_pid_cnt = ((i as u32) % 4) * 2;
  }
}

#[test]
fn greedy_picks_min_valid() {
  let mgr = mk(8);
  occupy(&mgr, 3);
  let mut gc = GarbageCollector::new(4);
  // 块 7/6/5 有效页 8/6/4；块 7 满（有效==总）被跳过 ⇒ 选有效最少的块 5
  let v = gc.select_victim_block_to_gc_read(&mgr, GcStrategy::Greedy, 0);
  assert_eq!(v, 5, "贪心应选出有效页最少的 FULL 块");
  assert!(gc.block_is_gc_victim(v));
  // 重复选择：块仍在 FULL 列表（select 只标记，不搬移）
  assert!(mgr.is_block_in_list(5, BlockState::Full));
}

#[test]
fn random_gc_no_div_zero_on_empty() {
  // C-14 修复验证：FULL 列表为空时返回 MAX 而非 SIGFPE
  let mgr = mk(8);
  let gc = GarbageCollector::new(4);
  assert_eq!(gc.random_gc(&mgr), BlockId::MAX, "空表必须早退");
  occupy(&mgr, 2);
  assert!(gc.random_gc(&mgr) < 8, "非空表应可随机选");
}

#[test]
fn random_gc_filters_non_full_state() {
  // C-14「状态过滤」验证：FULL 列表中混入一个 state 已被并发迁移为 GcRead 的块
  // （C-30 列表/元数据失联场景），random_gc 不得选中它。
  let mgr = mk(8);
  occupy(&mgr, 3); // FULL = [7, 6, 5]，三者 state 均 Full
  // 把块 5 正常移出到 GcRead（state→GcRead），再把 5 人为塞回 FULL 列表尾，
  // 制造「列表含 5、state=GcRead」的失联态（等价 C-30 / C-39）。
  assert!(mgr.move_block_to_list(5, BlockState::Full, BlockState::GcRead));
  mgr.sspace.write().block_lists[BlockState::Full.idx()].push(5);
  assert_eq!(mgr.block_meta(5).state, BlockState::GcRead);
  let gc = GarbageCollector::new(4);
  for _ in 0..2000 {
    let v = gc.random_gc(&mgr);
    assert_ne!(v, 5, "state≠Full 的失联块不得被 random_gc 选中");
    assert!(v == 7 || v == 6, "只能选 FULL 列表中 state=Full 的块: {v}");
  }
}

#[test]
fn edt_based_gc_excludes_gc_read_state_block() {
  // R3 修（P1）：上游 EDTBasedGC 按 expected_state（read?FULL:GC_READ）严格过滤
  // （garbage_collector.cc:714-717）；旧实现接受 Full|GcRead 两种，在 C-30 失联态
  // （FULL 列表含 state=GcRead 块）下会把已在 GC 中的块重复选为 victim。修后 read=true
  // 只接受 state==Full。
  let mgr = mk(8);
  occupy(&mgr, 3); // FULL=[7,6,5]；有效 8/6/4（块 5 有效最少）
  // 正常移出块 5 到 GcRead（state→GcRead），再人为塞回 FULL 列表尾 → 失联态。
  assert!(mgr.move_block_to_list(5, BlockState::Full, BlockState::GcRead));
  mgr.sspace.write().block_lists[BlockState::Full.idx()].push(5);
  assert_eq!(mgr.block_meta(5).state, BlockState::GcRead);
  let gc = GarbageCollector::new(4);
  let v = gc.edt_based_gc(&mgr, true, u64::MAX);
  assert_ne!(v, 5, "state=GcRead 的块不得被 read 路径重复选为 victim");
  assert_eq!(v, 6, "应选 state=Full 且有效最少的块（上游严格过滤）");
}

#[test]
fn fifo_gc_reachable_and_safe() {
  // C-31 修复验证：FIFO 可达且空表安全
  let mgr = mk(8);
  let gc = GarbageCollector::new(4);
  assert_eq!(gc.fifo_gc(&mgr), BlockId::MAX, "空表早退");
  occupy(&mgr, 2);
  let v = gc.fifo_gc(&mgr);
  assert!(v < 8, "FIFO 应弹出最早块");
}

#[test]
fn fifo_peek_does_not_drop_block() {
  // R1 修复验证：上游 FIFOGC 只窥视 [0] 不移出；旧 pop_front 会把块从 FULL
  // 列表弹走而不更新 state → 块丢失。修后：同块重复可选中、列表长度不变。
  let mgr = mk(8);
  let gc = GarbageCollector::new(4);
  occupy(&mgr, 3); // FULL = [7, 6, 5]
  let v = gc.fifo_gc(&mgr);
  assert_eq!(v, 7, "FIFO 取队首（最早入列）");
  assert!(
    mgr.is_block_in_list(7, BlockState::Full),
    "窥视不得移出列表"
  );
  assert_eq!(mgr.block_meta(7).state, BlockState::Full, "窥视不得改状态");
  assert_eq!(mgr.get_block_list_size(BlockState::Full), 3, "列表长度不变");
  // 再调一次仍取同块（上游 [0] 语义）
  assert_eq!(gc.fifo_gc(&mgr), 7, "重复调用返回同一队首块");
}

#[test]
fn two_r_greedy_guards_empty_gc_read() {
  // C-15 修复验证：GC_READ 空表不越界
  let mgr = mk(8);
  let mut gc = GarbageCollector::new(4);
  assert!(gc.two_r_greedy_gc(&mgr, true).is_empty());
  // 占用 6 块 + 迁移 1 块（带死页）到 GC_READ：FULL 余 5 >= mincnt(2) 进入候选
  occupy(&mgr, 6);
  let b = mgr.select_full_block_to_gc_read();
  assert!(b < 8, "FULL 应有块可选: {b}");
  let picked = gc.two_r_greedy_gc(&mgr, true);
  assert!(!picked.is_empty(), "FULL 达 mincnt 应有候选");
}

#[test]
fn k_greedy_and_edt_based() {
  let mgr = mk(16);
  occupy(&mgr, 6);
  let mut gc = GarbageCollector::new(4);
  let v = gc.select_victim_block_to_gc_read(&mgr, GcStrategy::KGreedy(3), 0);
  assert!(v < 16);
  let v2 = gc.select_victim_block_to_gc_read(&mgr, GcStrategy::EdtBased, 999);
  assert!(v2 < 16, "EDTBased 应选出 victim: {v2}");
  assert!(gc.should_trust_edt(&mgr, v2, u64::MAX));
}

#[test]
fn imbalance_group_total_consistency() {
  // C-24 修复验证：组内总量口径 + 组间择优（取总量最小组的首块）
  let mgr = mk(8);
  let gc = GarbageCollector::new(4);
  occupy(&mgr, 4); // 块 7/6/5/4 有效 8/6/4/2
  let groups = vec![vec![7, 6], vec![5, 4]]; // 总量 14 vs 6
  let pick = gc.greedy_among_imbalanced_groups(&mgr, &groups);
  assert_eq!(pick, 5, "应选总量最小组 [5,4] 的首块");
}

#[test]
fn valid_size_pure_byte_formula() {
  // C-29 修复验证：valid size = valid_cnt * PAGE_SIZE（不混入条目计数）
  let mgr = mk(4);
  let gc = GarbageCollector::new(4);
  {
    let mut s = mgr.sspace.write();
    let m = &mut s.block_metadata[0];
    m.total_pid_cnt = 8;
    m.invalidated_pid_cnt = 3;
  }
  assert_eq!(gc.valid_pi_ds_cnt_in_block(&mgr, 0), 5);
  assert_eq!(gc.valid_pi_ds_size_in_block(&mgr, 0), 5 * PAGE_SIZE as u64);
}

#[test]
fn invalidation_history_delegates_to_space() {
  // GC 侧只委托 space 元数据递增（原写死 Vec 已删）
  let mgr = mk(4);
  let mut gc = GarbageCollector::new(4);
  gc.add_block_invalidation_history(&mgr, 0, 7);
  gc.add_block_invalidation_history(&mgr, 0, 9);
  assert_eq!(mgr.block_meta(0).block_invalidation_cnt, 2);
}

#[test]
fn move_transitions_roundtrip() {
  let mgr = mk(8);
  let mut gc = GarbageCollector::new(4);
  occupy(&mgr, 2);
  let v = gc.select_victim_block_to_gc_read(&mgr, GcStrategy::Greedy, 0);
  assert!(
    gc.move_block_from_full_to_gc_block_list(&mgr, v)
      || mgr.block_meta(v).state == BlockState::GcRead
  );
  let n = gc.move_blocks_from_gc_to_empty_block_list(&mgr, &[v]);
  assert!(n <= 1);
  assert_eq!(
    mgr.block_meta(v).state,
    BlockState::Empty,
    "回收后应归 EMPTY"
  );
}

#[test]
fn move_boundaries_no_dual_list_no_loss() {
  let mgr = mk(8);
  occupy(&mgr, 2); // FULL = [7, 6]
  let gc = GarbageCollector::new(4);
  // FULL → GC_READ：移出源列表、入目标列表，绝不同时在两侧
  assert!(gc.move_block_from_full_to_gc_block_list(&mgr, 7));
  assert_eq!(mgr.block_meta(7).state, BlockState::GcRead);
  assert!(!mgr.is_block_in_list(7, BlockState::Full));
  assert!(mgr.is_block_in_list(7, BlockState::GcRead));
  // 重复搬移（源态不符）→ false，且块不丢
  assert!(!gc.move_block_from_full_to_gc_block_list(&mgr, 7));
  assert!(mgr.is_block_in_list(7, BlockState::GcRead));
  // 搬移不存在的块（越界）→ false
  assert!(!gc.move_block_from_full_to_gc_block_list(&mgr, 999));
  assert!(!gc.move_block_from_full_to_gc_block_list(&mgr, BlockId::MAX));
  // 空批次 → 0
  let mut gc2 = GarbageCollector::new(4);
  assert_eq!(gc2.move_blocks_from_gc_to_empty_block_list(&mgr, &[]), 0);
  assert_eq!(gc2.move_blocks_from_gc_to_open_block_list(&mgr, &[]), 0);
  // GC_READ → EMPTY，块归 EMPTY 列表
  assert_eq!(gc2.move_blocks_from_gc_to_empty_block_list(&mgr, &[7]), 1);
  assert_eq!(mgr.block_meta(7).state, BlockState::Empty);
  assert!(mgr.is_block_in_list(7, BlockState::Empty));
}

#[test]
fn move_gc_read_to_open() {
  let mgr = mk(8);
  let mut gc = GarbageCollector::new(4);
  occupy(&mgr, 2);
  assert!(gc.move_block_from_full_to_gc_block_list(&mgr, 6));
  assert_eq!(gc.move_blocks_from_gc_to_open_block_list(&mgr, &[6]), 1);
  assert_eq!(mgr.block_meta(6).state, BlockState::Open);
  assert!(mgr.is_block_in_list(6, BlockState::Open));
  assert!(!mgr.is_block_in_list(6, BlockState::GcRead));
}

#[test]
fn empty_selectors_return_max() {
  let mgr = mk(8);
  let mut gc = GarbageCollector::new(4);
  assert_eq!(gc.greedy_gc(&mgr, true), BlockId::MAX);
  assert_eq!(gc.greedy_gc(&mgr, false), BlockId::MAX);
  assert_eq!(gc.edt_based_gc(&mgr, true, u64::MAX), BlockId::MAX);
  assert_eq!(gc.edt_based_gc(&mgr, false, u64::MAX), BlockId::MAX);
  assert_eq!(gc.k_greedy_gc(&mgr, 3), BlockId::MAX);
  assert_eq!(gc.fifo_gc(&mgr), BlockId::MAX);
  assert_eq!(gc.random_gc(&mgr), BlockId::MAX);
  assert_eq!(gc.two_r_greedy_gc(&mgr, false), Vec::<BlockId>::new());
  assert_eq!(
    gc.select_victim_block_to_gc_read(&mgr, GcStrategy::Greedy, 0),
    BlockId::MAX
  );
}

#[test]
fn greedy_n_selects_distinct_victims() {
  // 修复验证：victim_cnt>1 时内核按 gc_victims 过滤，逐轮选出不同块
  // （原实现每轮重扫不过滤 → 第二轮即空手）
  let mgr = mk(8);
  occupy(&mgr, 5); // 块 7/6/5/4/3 有效 8/6/4/2/8
  let mut gc = GarbageCollector::new(u64::MAX); // 关失衡路径，纯贪心
  let victims = gc.greedy_gc_n(&mgr, true, 3);
  assert_eq!(victims, vec![4, 5, 6], "应按有效升序选 3 个互异块");
}

#[test]
fn port_api_accessors_and_write_selection() {
  // 消费 5 个 map.js 声明的 port 函数（get_page_offset / get_valid_pi_ds_in_block /
  // avg_group_valid_pages / select_victim_blocks_to_gc_write），断言真实语义。
  use futures_lite::future::block_on;
  let mgr = mk(8);
  let gc = GarbageCollector::new(4);
  let block_bytes = mgr.get_block_size();
  // pid=3 映射到 block2 起始偏移，压缩尺寸非 0（有效）
  block_on(mgr.update_pid2_offset(3, 2 * block_bytes, 4096));
  assert_eq!(gc.get_page_offset(&mgr, 3), 2 * block_bytes);
  assert!(gc.page_is_valid(&mgr, 3));
  assert_eq!(gc.convert_offset2_pid(&mgr, 2 * block_bytes), Some(3));
  // 反向表登记 pid3 到 block2 slot0 ⇒ 收集应命中 [3]
  mgr.update_offset2_pids(2, 0, 3);
  assert_eq!(gc.get_valid_pi_ds_in_block(&mgr, 2), vec![3]);
  // 未登记块 → 空
  assert!(gc.get_valid_pi_ds_in_block(&mgr, 5).is_empty());

  // 有效页 8/6/4/2 的块 7/6/5/4
  occupy(&mgr, 4);
  assert_eq!(gc.avg_group_valid_pages(&mgr, &[7, 6]), 7);
  assert_eq!(gc.avg_group_valid_pages(&mgr, &[]), 0);

  // 写目标选择：GC_READ 列表选有效页最少的块（此处仅块 5）
  let mut s = mgr.sspace.write();
  assert!(s.move_block_to_list(5, BlockState::Full, BlockState::GcRead));
  drop(s);
  let mut gcw = GarbageCollector::new(u64::MAX);
  assert_eq!(gcw.select_victim_blocks_to_gc_write(&mgr, 1), vec![5]);
}

#[test]
fn get_valid_pi_ds_in_block_no_self_deadlock() {
  // P0 死锁回归（R1 修）：`get_valid_pi_ds_in_block` 须先在循环外取
  // `page_cnt_per_block`（sspace 读锁），循环体每 slot 再调 `convert_offset2_pids`
  // ——后者在 ldb_space 内取 `sspace.write()`。若把 `sspace.read()` 写回 `for` 头，
  // range 表达式临时读守卫被延长到整个循环体，同线程读未释即求写
  // （parking_lot 非可重入）→ 自锁死，本用例挂起（TIMEOUT）。
  use futures_lite::future::block_on;
  let mgr = mk(8);
  let gc = GarbageCollector::new(4);
  let block_bytes = mgr.get_block_size();
  // 在块 1 的首槽/末槽各登记一个有效 pid（page_cnt_per_block=64 ⇒ 64 次写锁路径）
  block_on(mgr.update_pid2_offset(7, block_bytes, 4096));
  block_on(mgr.update_pid2_offset(8, 2 * block_bytes, 4096));
  mgr.update_offset2_pids(1, 0, 7);
  mgr.update_offset2_pids(1, 63, 8);
  let got = gc.get_valid_pi_ds_in_block(&mgr, 1);
  assert_eq!(got, vec![7, 8], "应按 slot 升序收集两个有效页");
  // 无有效页的块仍返回空（不得因过滤逻辑误判）
  assert!(gc.get_valid_pi_ds_in_block(&mgr, 6).is_empty());
}

#[test]
fn should_select_imbalanced_block_matches_will_trigger() {
  // 构造确定的失衡：组 0（块 0/2）平均有效 0，组 1（块 1/3）平均 100 ⇒ 触发
  let mgr = mk(4);
  {
    let mut s = mgr.sspace.write();
    for bid in 0..4u64 {
      s.block_metadata[bid as usize].state = BlockState::Full;
    }
    for bid in [1u64, 3] {
      s.block_metadata[bid as usize].total_pid_cnt = 100;
    }
  }
  let gc = GarbageCollector::new(4);
  assert!(gc.will_trigger_imbalance(&mgr, 0), "组 0 应判失衡");
  assert!(!gc.should_select_imbalanced_block(&mgr, 0));
}

// ---- 影子模型差分（1e5 随机状态）----

/// 确定性 PRNG（splitmix64），避免引入额外依赖
struct Rng(u64);
impl Rng {
  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }
  fn below(&mut self, n: u64) -> u64 {
    if n == 0 { 0 } else { self.next_u64() % n }
  }
}

/// (bid, state, total_pid_cnt, invalidated_pid_cnt, edt)
type Entry = (BlockId, BlockState, u32, u32, LogId);

fn setup_full(mgr: &SpaceManager, entries: &[Entry]) {
  let mut s = mgr.sspace.write();
  s.block_lists[BlockState::Full.idx()] = MutexVec::default();
  for m in s.block_metadata.iter_mut() {
    *m = BlockMetadata::new(0, 0, BlockState::Empty);
  }
  for &(bid, st, total, invalid, edt) in entries {
    let mi = bid as usize;
    s.block_metadata[mi].state = st;
    s.block_metadata[mi].total_pid_cnt = total;
    s.block_metadata[mi].invalidated_pid_cnt = invalid;
    s.block_metadata[mi].edt = edt;
    if st == BlockState::Full {
      s.block_lists[BlockState::Full.idx()].push(bid);
    }
  }
}

/// 独立重实现：上游 `GreedyGC(bool)`（garbage_collector.cc:365-409）语义
fn ref_greedy(entries: &[Entry]) -> BlockId {
  let mut min_valid = 64 * (PAGE_SIZE / MIN_COMP_SIZE as usize) as u64;
  let mut min_bid = BlockId::MAX;
  for &(bid, st, total, invalid, _) in entries {
    if st != BlockState::Full {
      continue;
    }
    let valid = u64::from(total.saturating_sub(invalid));
    if valid == u64::from(total) {
      continue;
    }
    if valid == 0 {
      return bid;
    }
    if valid < min_valid {
      min_valid = valid;
      min_bid = bid;
    }
  }
  min_bid
}

/// 独立重实现：上游 `EDTBasedGC(bool)` + 本 crate 文档化的 flush 信任过滤
fn ref_edt(entries: &[Entry], flushed: LogId) -> BlockId {
  let mut victim = BlockId::MAX;
  let mut min_edt = LogId::MAX;
  let mut min_valid = u32::MAX;
  for &(bid, st, total, invalid, edt) in entries {
    if st != BlockState::Full {
      continue;
    }
    let valid = total.saturating_sub(invalid);
    let eff = if edt != 0 && edt <= flushed {
      edt
    } else {
      LogId::MAX
    };
    if eff < min_edt || (eff == min_edt && valid < min_valid) {
      min_edt = eff;
      min_valid = valid;
      victim = bid;
    }
  }
  victim
}

fn random_entries(rng: &mut Rng, blocks: u64) -> Vec<Entry> {
  let mut bids: Vec<BlockId> = (0..blocks).collect();
  for i in (1..bids.len()).rev() {
    bids.swap(i, rng.below(i as u64 + 1) as usize);
  }
  let n = rng.below(blocks + 1) as usize;
  bids[..n]
    .iter()
    .map(|&bid| {
      let st = match rng.below(3) {
        0 => BlockState::Full,
        1 => BlockState::GcRead,
        _ => BlockState::Empty,
      };
      let total = rng.below(9) as u32;
      let invalid = rng.below(u64::from(total) + 1) as u32;
      let edt = rng.below(1000);
      (bid, st, total, invalid, edt)
    })
    .collect()
}

#[test]
#[ignore = "1e5 差分对拍 >100ms（E2）：显式 nextest run --run-ignored all 运行"]
fn greedy_differential_1e5() {
  let mgr = mk(16);
  let mut rng = Rng(0x5EED_0001);
  let mut mism = 0u64;
  for _ in 0..100_000 {
    let entries = random_entries(&mut rng, 16);
    setup_full(&mgr, &entries);
    let gc = GarbageCollector::new(4);
    let got = gc.greedy_gc(&mgr, true);
    let want = ref_greedy(&entries);
    if got != want {
      mism += 1;
    }
  }
  assert_eq!(mism, 0, "greedy MISMATCH={mism}");
}

#[test]
#[ignore = "1e5 差分对拍 >100ms（E2）：显式 nextest run --run-ignored all 运行"]
fn edt_differential_1e5() {
  let mgr = mk(16);
  let mut rng = Rng(0x5EED_0002);
  let mut mism = 0u64;
  for i in 0..100_000u64 {
    let entries = random_entries(&mut rng, 16);
    setup_full(&mgr, &entries);
    let flushed = (i % 2) * 500 + (rng.below(3) * 100);
    let gc = GarbageCollector::new(4);
    let got = gc.edt_based_gc(&mgr, true, flushed);
    let want = ref_edt(&entries, flushed);
    if got != want {
      mism += 1;
    }
  }
  assert_eq!(mism, 0, "edt MISMATCH={mism}");
}

#[test]
fn greedy_n_differential_1e4() {
  let mgr = mk(16);
  let mut rng = Rng(0x5EED_0003);
  let mut mism = 0u64;
  for _ in 0..10_000 {
    let entries = random_entries(&mut rng, 16);
    setup_full(&mgr, &entries);
    let cnt = rng.below(8) as usize + 1;
    let mut gc = GarbageCollector::new(u64::MAX); // 关失衡
    let got = gc.greedy_gc_n(&mgr, true, cnt);
    // 影子：逐轮取当前可选最小有效块（排除已选）
    let mut want = Vec::new();
    while want.len() < cnt {
      let mut best = BlockId::MAX;
      let mut mv = 64 * (PAGE_SIZE / MIN_COMP_SIZE as usize) as u64;
      for &(bid, st, total, invalid, _) in &entries {
        if st != BlockState::Full || want.contains(&bid) {
          continue;
        }
        let valid = u64::from(total.saturating_sub(invalid));
        if valid == u64::from(total) {
          continue;
        }
        if valid == 0 {
          best = bid;
          break;
        }
        if valid < mv {
          mv = valid;
          best = bid;
        }
      }
      if best == BlockId::MAX {
        break;
      }
      want.push(best);
    }
    if got != want {
      mism += 1;
    }
  }
  assert_eq!(mism, 0, "greedy_n MISMATCH={mism}");
}
