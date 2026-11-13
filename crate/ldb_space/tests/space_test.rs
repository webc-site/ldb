//! ldb_space 集成回归（E1 用例表见报告；每条断言钉死上游行为 + 唯一击杀变异编号）。
//!
//! 覆盖面：状态机闭环 / 块选择上游边 / GC 触发重写 / PID2Offset 生命周期 /
//! 写缓冲 C-18 / 装箱契约（ldb_pack CROSS 收口）/ 读路径 Err 传播（ldb_codec
//! CROSS 收口）/ EDT 编排 / NoWA padding / 检查点 / 影子模型对拍。

use std::{future, future::Future, io, pin::Pin, sync::atomic::Ordering};

use fastrand::Rng;
use futures_lite::future::block_on;
use ldb_space::{space_mgr::Sink, *};
use ldb_type::{BlockId, GsnWatermark, LogId, PAGE_SIZE, WRITE_HISTORY_LENGTH};

fn mk_mgr(blocks: u64) -> SpaceManager {
  SpaceManager::new(blocks, 8, 2048, 4, blocks * 8, 2)
}

fn tracker() -> EdtTracker {
  EdtTracker
}

/// T1 状态机闭环：EMPTY→OPEN→IN_USE→FULL(PAD)→GC_IN_USE→GC_WRITE→erase 回 EMPTY；
/// 回收后无泄漏（M-A/M-B/M-C/M-D）
#[test]
fn state_machine_moves() {
  let mgr = mk_mgr(8);
  let bid = mgr.select_empty_block_id(0, 100);
  assert!(bid < 8);
  assert_eq!(mgr.block_meta(bid).state, BlockState::Open);
  assert_eq!(mgr.get_block_list_size(BlockState::Empty), 7);

  assert!(mgr.move_block_to_list(bid, BlockState::Open, BlockState::InUse));
  // 状态不匹配的迁移必须拒绝（上游 MoveBlockToList 双重校验，M-A）
  assert!(!mgr.move_block_to_list(bid, BlockState::Open, BlockState::Full));
  mgr.update_block_metadata_after_write(bid, 8 * PAGE_SIZE as u64, 55);
  // 已满关闭直 Full 并撤活跃表（上游 fast-path space_manager.cc:1381-1385；
  // 旧实现 Full/FullPad 判定倒置 —— 未满块直标 Full 破坏「FULL⇒cur_size==max_w」）
  assert!(mgr.close_one_open_block(bid, 8 * PAGE_SIZE as u64));
  assert_eq!(mgr.block_meta(bid).state, BlockState::Full);
  assert!(!mgr.block_id_in_active_block_list(bid), "Full 关闭撤活跃表");

  // FULL → GC_IN_USE（garbage_collector.cc:817）→ 活跃表登记
  assert!(mgr.mark_block_gc_in_use(bid));
  assert_eq!(mgr.block_meta(bid).state, BlockState::GcInUse);
  assert!(mgr.block_in_gc_use(bid));
  assert!(mgr.block_id_in_active_block_list(bid));
  // 活跃表登记公开入口（GC_IN_USE 态可重复登记；EMPTY 态见 select_empty 路径）
  assert!(mgr.remove_block_id_from_active_block_list(bid));
  assert!(
    mgr.add_block_id_to_active_block_list(bid),
    "GC_IN_USE 可入活跃表"
  );
  assert!(mgr.block_id_in_active_block_list(bid));

  // FULL → GC_READ → GC_WRITE → erase 回 EMPTY
  assert!(mgr.move_block_to_list(bid, BlockState::GcInUse, BlockState::Full));
  assert_eq!(mgr.select_full_block_to_gc_read(), bid);
  // GC 读块迁移回写列表（上游 MoveGCedBlockToOpenList；cur_size=0 跳过 EDT 补洞）
  assert!(mgr.move_g_ced_block_to_open_list(bid, 0, 100));
  assert_eq!(mgr.block_meta(bid).state, BlockState::Open);
  assert!(mgr.move_block_to_list(bid, BlockState::Open, BlockState::InUse));
  assert!(
    mgr.update_block_metadata_after_read_gc(bid),
    "IN_USE→GC_READ"
  );
  assert_eq!(mgr.block_meta(bid).state, BlockState::GcRead);
  assert!(mgr.mark_block_gc_write(bid), "M-D: GC_READ→GC_WRITE");
  assert_eq!(mgr.block_meta(bid).state, BlockState::GcWrite);
  assert!(mgr.erase_block(bid), "M-C: GC_WRITE 块必须可擦除回收");
  assert_eq!(mgr.block_meta(bid).state, BlockState::Empty);
  assert_eq!(mgr.get_block_list_size(BlockState::Empty), 8, "回收无泄漏");

  // 非 GC_WRITE 态擦除必须拒绝
  let other = mgr.select_empty_block_id(0, 5);
  assert!(!mgr.erase_block(other));

  // IN_USE 读 GC 迁移（上游 UpdateBlockMetadataAfterReadGC：IN_USE→GC_READ，
  // 列表与元数据同步迁移 —— 旧实现只改 state 的失联缺陷）
  let _ = mgr.move_block_to_list(other, BlockState::Open, BlockState::InUse);
  assert!(mgr.update_block_metadata_after_read_gc(other));
  assert_eq!(mgr.block_meta(other).state, BlockState::GcRead);
  // reset_block_space 顺序：先迁移后清零（M-B）
  assert!(mgr.reset_block_space(other));

  // 未满关闭 → FullPad 待补写（上游 CloseOneOpenBlock :1284 OPEN→FULL_PAD，
  // 保留活跃表直至 padding 完成）→ move_padded 收 Full（:1085-1093）
  let pad_block = mgr.select_empty_block_id(0, 3);
  let _ = mgr.move_block_to_list(pad_block, BlockState::Open, BlockState::InUse);
  assert!(mgr.close_one_open_block(pad_block, 0));
  assert_eq!(mgr.block_meta(pad_block).state, BlockState::FullPad);
  assert!(
    mgr.block_id_in_active_block_list(pad_block),
    "FullPad 待补写保留活跃表"
  );
  assert!(mgr.move_padded_blocks_to_full_list(pad_block));
  assert_eq!(mgr.block_meta(pad_block).state, BlockState::Full);
  let _ = mgr.remove_block_id_from_active_block_list(pad_block);
  assert!(mgr.reset_block_space(pad_block));
  assert_eq!(mgr.get_block_list_size(BlockState::Empty), 8);
}

/// T2 块选择上游边：EDT 早退 / should_open 三态 / 未满块认领 / 最近 EDT /
/// 随机回退 / 失效历史优先（M-E/M-F/M-G/M-H）
#[test]
fn block_selection_upstream_edges() {
  let mgr = mk_mgr(16);
  let a = mgr.select_empty_block_id(0, 500);
  // 覆盖 avg_edt：把块 a 的 EDT 区间登记为 [400, 600)
  mgr.update_block_metadata_after_write(a, 0, 400);
  tracker().set_min_max_edt_per_block(&mgr, a, 400, 600);
  // avg_edt 落在活跃块区间且装得下 → 不新开（上游 space_manager.cc:986-997 哨兵）
  assert_eq!(mgr.select_empty_block_id(0, 500), BlockId::MAX, "M-E");
  // 区间外 → 新开
  let b = mgr.select_empty_block_id(0, 900);
  assert_ne!(a, b);

  // should_open_new_block_id 三态（上游 :657-694）
  let two = mk_mgr(8);
  assert!(two.should_open_new_block_id(1), "活跃表空必开");
  let x = two.select_empty_block_id(0, 100);
  tracker().set_min_max_edt_per_block(&two, x, 50, 150);
  assert!(
    !two.should_open_new_block_id(100),
    "M-F: avg_edt 已覆盖不开"
  );
  assert!(two.should_open_new_block_id(500), "区间外可开");
  let full: SpaceManager = SpaceManager::new(8, 8, 2048, 1, 64, 2);
  let _ = full.select_empty_block_id(0, 1);
  assert!(!full.should_open_new_block_id(1), "活跃表满额必不开");

  // 未满打开块直选（认领：OPEN→IN_USE，上游 SelectUnfilledOpenBlockID :317-367）
  let claim = mgr.select_unfilled_open_block_id(510);
  assert_eq!(claim, a, "最近 EDT 认领");
  assert_eq!(mgr.block_meta(a).state, BlockState::InUse, "M-G: 认领迁移");

  // 最近 EDT 打开块 + 超容拒绝
  let hit = mgr.select_open_block_id(1024, 510, 0, 1024);
  assert!(hit == a || hit == b, "最近 EDT 应命中");
  let big = mgr.select_open_block_id(8 * PAGE_SIZE as u64 + 1, 510, 0, 1024);
  assert_eq!(big, BlockId::MAX, "写尺寸超出块容量拒绝");

  // 随机回退：容量足够的 OPEN 候选（先关回 OPEN 态）
  assert!(mgr.move_block_to_list(a, BlockState::InUse, BlockState::Open));
  let r = mgr.select_open_block_id_random(64);
  assert_ne!(r, BlockId::MAX, "存在容量足够候选");
  let none = mgr.select_open_block_id_random(9 * PAGE_SIZE as u64);
  assert_eq!(none, BlockId::MAX, "无候选返回哨兵");

  // 失效历史优先：给块 c 记失效，min-invalidation 必须绕开它（M-H）
  let c = mgr.select_empty_block_id(0, 5);
  mgr.add_block_invalidation_history(c, 7);
  assert_eq!(mgr.block_meta(c).block_invalidation_cnt, 1);
  let picked = mgr.select_empty_block_with_min_invalidation_cnt(0);
  assert_ne!(picked, c, "0 失效块优先");
  let front = mgr.select_empty_block_random(0);
  assert!(front < 16, "队首窥视合法");
}

/// T3 GC 触发重写（C-1）：水位 + mark_gc_done 增量复位（M-WATER/M-DONE）
#[test]
fn gc_trigger_rewritten_not_zns_shortcircuit() {
  let mgr = mk_mgr(10);
  assert!(!mgr.should_trigger_gc(0));
  for _ in 0..9 {
    assert!(mgr.select_empty_block_id(0, 42) < 10);
  }
  assert!(mgr.should_trigger_gc(0), "高水位必须触发（上游死代码激活）");
  assert!(mgr.sspace.read().gc_triggered.load(Ordering::Relaxed));
  // GC 完成登记水位：增量归零后 reached 判定复位
  mgr.mark_gc_done();
  assert!(!mgr.reached_alloc_size(0), "M-DONE: 水位登记后增量触达复位");
}

/// T4 PID2Offset 生命周期：覆写失效接线 / 释放复用 / 唯一偏移 / 反查
/// （M-I/M-J/M-K/M-L）
#[test]
fn pid2offset_lifecycle() {
  let mgr = mk_mgr(8);
  let block_bytes = mgr.get_block_size();
  // 页 1 落在被选中的块（min-invalidation 策略取 EMPTY 队首）
  let b0 = mgr.select_empty_block_id(0, 1);
  block_on(mgr.update_pid2_offset(1, b0 * block_bytes, 777));
  mgr.update_offset2_pids(b0, 0, 1);
  assert_eq!(mgr.get_pid2_offset(1).get_offset(), b0 * block_bytes);
  assert_eq!(mgr.get_pid_comp_size(1), 777);
  assert_eq!(mgr.pid_block(1), b0);
  assert!(mgr.pid_offset_is_valid(1));
  assert!(mgr.page_is_valid(1));
  assert_eq!(mgr.get_max_pid(), 2, "映射表扩张到 pid+1");

  // 异地覆写：旧副本所在块失效计数 +1（上游 UpdatePID2Offset :1903-1912 接线）
  let b1 = mgr.select_empty_block_id(0, 2);
  block_on(mgr.update_pid2_offset(1, b1 * block_bytes, 700));
  assert_eq!(mgr.pid_block(1), b1);
  assert_eq!(
    mgr.block_meta(b0).invalidated_pid_cnt,
    1,
    "M-I: 覆写后旧副本失效计数"
  );
  assert_eq!(mgr.valid_pi_ds_cnt_in_block(b0), 0);
  assert!(mgr.valid_pi_ds_in_secure_place(1, b0), "已迁出旧块");

  // 无关块误报失效必须不命中（M-J）
  assert!(!mgr.update_block_invalid_count(3 * block_bytes, 1));
  assert_eq!(mgr.block_meta(3).invalidated_pid_cnt, 0);

  // 释放 → 复用池
  mgr.add_pid_to_dealloc_list(1);
  assert!(mgr.is_pid_in_deallocated_list(1));
  assert!(mgr.pid_is_deallocated(1));
  assert!(!mgr.page_is_valid(1));
  assert_eq!(mgr.process_dealloc_list(), 1);
  assert_eq!(mgr.get_pid_comp_size(1), 0, "释放后映射清零");
  assert!(mgr.has_reusable_pid());
  assert_eq!(mgr.take_reusable_pid(), Some(1), "M-K");
  assert!(!mgr.has_reusable_pid(), "取尽后谓词复位");

  // 可复用 PID 登记 + 池化
  mgr.add_reusable_pid(42);
  assert!(mgr.has_reusable_pid());
  assert_eq!(mgr.take_reusable_pid(), Some(42));

  // 反查 / 唯一偏移
  block_on(mgr.update_pid2_offset(2, 2 * PAGE_SIZE as u64, 4096));
  assert_eq!(mgr.get_pid_from_offset(2 * PAGE_SIZE as u64), Some(2));
  assert!(!mgr.is_unique_offset(2 * PAGE_SIZE as u64), "M-L");
  assert!(mgr.is_unique_offset(6 * PAGE_SIZE as u64));
  // 分配扩张
  assert_eq!(mgr.assign_pid_upon_alloc(16), 3, "返回旧表长并扩张");
  assert_eq!(mgr.get_max_pid(), 16);
}

/// T5 影子模型对拍（A2）：随机 alloc/claim/write/close/gc/recycle 序列，
/// 断言无双重分配、状态/列表/元数据一致、块 EDT 加权链与上游公式一致、
/// 终态无泄漏。默认 2e4 步（<100ms）；1e5 步 `--ignored` 档（A2 证据）。
fn shadow_run(steps: u64, seed: u64) {
  const BLOCKS: u64 = 16;
  let mgr = mk_mgr(BLOCKS);
  let mut rng = Rng::with_seed(seed);
  // 影子状态：bid → (state, min_edt, max_edt, total_pid)
  let mut shadow: Vec<(BlockState, LogId, LogId, u32)> =
    vec![(BlockState::Empty, LogId::MAX, 0, 0); BLOCKS as usize];
  let mut shadow_edt = vec![0u8 as LogId; BLOCKS as usize];
  let mut live_hist: Vec<ldb_edt::WriteHistory> = vec![[0; WRITE_HISTORY_LENGTH]; BLOCKS as usize];
  let mut live_wcnt = vec![0u32; BLOCKS as usize];

  let check = |shadow: &[(BlockState, LogId, LogId, u32)], mgr: &SpaceManager| {
    for (bid, (st, min_edt, max_edt, pages)) in shadow.iter().enumerate() {
      let m = mgr.block_meta(bid as BlockId);
      assert_eq!(m.state, *st, "状态与影子不一致 bid={bid}");
      assert!(
        mgr.is_block_in_list(bid as BlockId, *st),
        "块须在其状态列表 bid={bid}"
      );
      assert_eq!(m.block_min_edt, *min_edt, "min EDT 漂移 bid={bid}");
      assert_eq!(m.block_max_edt, *max_edt, "max EDT 漂移 bid={bid}");
      assert_eq!(m.total_pid_cnt, *pages, "页计数漂移 bid={bid}");
    }
  };

  for step in 0..steps {
    match rng.u64(..7) {
      // alloc：select_empty 只能发影子 Empty 的块（无双重分配）
      0 => {
        let avg = rng.u64(..1024);
        let bid = mgr.select_empty_block_id(0, avg);
        if bid != BlockId::MAX {
          assert_eq!(
            shadow[bid as usize].0,
            BlockState::Empty,
            "双重分配 bid={bid}"
          );
          shadow[bid as usize].0 = BlockState::Open;
        }
      }
      // claim：OPEN → IN_USE（认领后才能 close）
      1 => {
        let bid = rng.u64(..BLOCKS) as usize;
        if shadow[bid].0 == BlockState::Open {
          assert!(
            mgr.move_block_to_list(bid as BlockId, BlockState::Open, BlockState::InUse),
            "认领必须成功"
          );
          shadow[bid].0 = BlockState::InUse;
        }
      }
      // write：EDT 链与上游 CalcBlockAvgEDT 公式对拍
      2 => {
        let bid = rng.u64(..BLOCKS) as usize;
        if shadow[bid].0 != BlockState::Empty {
          let gsn = step * 3 + 1;
          let page_edt = {
            let hist = &mut live_hist[bid];
            let w = &mut live_wcnt[bid];
            *w = w.wrapping_add(1);
            let mut max_idx = 0;
            for i in 1..WRITE_HISTORY_LENGTH {
              if hist[i] > hist[max_idx] {
                max_idx = i;
              }
            }
            hist[(max_idx + 1) % WRITE_HISTORY_LENGTH] = gsn;
            ldb_edt::estimate_death_time(hist)
          };
          let prev_total = shadow[bid].3;
          let new_total = prev_total + 1;
          let new_block_edt = if prev_total == 0 {
            page_edt
          } else {
            let w =
              (shadow_edt[bid] as f64 * prev_total as f64 + page_edt as f64) / new_total as f64;
            w.round() as LogId
          };
          mgr.update_block_metadata_after_write(bid as BlockId, 4096, page_edt);
          let m = mgr.block_meta(bid as BlockId);
          assert_eq!(
            m.edt, new_block_edt,
            "块 EDT 加权链与上游公式不一致 bid={bid}"
          );
          assert_eq!(m.total_pid_cnt, new_total);
          shadow_edt[bid] = new_block_edt;
          let s = &mut shadow[bid];
          s.3 = new_total;
          s.1 = s.1.min(page_edt);
          s.2 = s.2.max(page_edt);
        }
      }
      // close：IN_USE → FULL_PAD（未满待补写）/FULL（已满），FullPad 经 move_padded 收 FULL
      3 => {
        let bid = rng.u64(..BLOCKS) as usize;
        if shadow[bid].0 == BlockState::InUse {
          let sz = mgr.get_block_write_offset(bid as BlockId);
          assert!(mgr.close_one_open_block(bid as BlockId, sz));
          let st = if sz < mgr.get_block_size() {
            BlockState::FullPad
          } else {
            BlockState::Full
          };
          shadow[bid].0 = st;
          if st == BlockState::FullPad {
            assert!(
              mgr.move_padded_blocks_to_full_list(bid as BlockId),
              "FullPad 待补写块可收 Full"
            );
            shadow[bid].0 = BlockState::Full;
          }
        } else {
          assert!(
            !mgr.close_one_open_block(bid as BlockId, 0),
            "非 IN_USE 关闭拒绝"
          );
        }
      }
      // gc claim + gc write
      4 => {
        let bid = rng.u64(..BLOCKS) as usize;
        if shadow[bid].0 == BlockState::Full {
          assert!(mgr.mark_block_gc_in_use(bid as BlockId), "FULL→GC_IN_USE");
          shadow[bid].0 = BlockState::GcInUse;
        } else if shadow[bid].0 == BlockState::GcInUse {
          let prev_total = shadow[bid].3;
          let new_total = prev_total + 1;
          let new_block_edt = if prev_total == 0 {
            9
          } else {
            let w = (shadow_edt[bid] as f64 * prev_total as f64 + 9.0) / new_total as f64;
            w.round() as LogId
          };
          mgr.update_block_metadata_after_gc_write(bid as BlockId, 4096, 9);
          let m = mgr.block_meta(bid as BlockId);
          assert_eq!(m.edt, new_block_edt, "GC 写 EDT 链漂移 bid={bid}");
          shadow_edt[bid] = new_block_edt;
          let s = &mut shadow[bid];
          s.3 = new_total;
          s.1 = s.1.min(9);
          s.2 = s.2.max(9);
        }
      }
      // recycle：reset_block_space 回 EMPTY（无泄漏断言核心）
      5 => {
        let bid = rng.u64(..BLOCKS) as usize;
        let st = shadow[bid].0;
        if st != BlockState::Empty {
          assert!(
            mgr.reset_block_space(bid as BlockId),
            "M-B: {st:?} 块必须可回收"
          );
          if mgr.block_id_in_active_block_list(bid as BlockId) {
            assert!(mgr.remove_block_id_from_active_block_list(bid as BlockId));
          }
          shadow[bid] = (BlockState::Empty, LogId::MAX, 0, 0);
          shadow_edt[bid] = 0;
          live_hist[bid] = [0; WRITE_HISTORY_LENGTH];
          live_wcnt[bid] = 0;
        } else {
          // EMPTY 块回收为幂等 no-op（Empty→Empty 迁移合法），状态必须不变
          let _ = mgr.reset_block_space(bid as BlockId);
          assert_eq!(mgr.block_meta(bid as BlockId).state, BlockState::Empty);
        }
      }
      // gc 触发判定对拍（C-1 重写规则的水位镜像）
      _ => {
        let empty = shadow.iter().filter(|s| s.0 == BlockState::Empty).count() as u64;
        let used_ratio = (BLOCKS - empty) * 100 / BLOCKS;
        assert_eq!(
          mgr.should_trigger_gc(0),
          used_ratio >= 80,
          "触发判定与水位模型不一致"
        );
      }
    }
    if step % 4096 == 0 {
      check(&shadow, &mgr);
    }
  }
  // 全量回收 → 无泄漏
  for bid in 0..BLOCKS as usize {
    if shadow[bid].0 != BlockState::Empty {
      assert!(
        mgr.reset_block_space(bid as BlockId),
        "终态回收失败 bid={bid}"
      );
      if mgr.block_id_in_active_block_list(bid as BlockId) {
        mgr.remove_block_id_from_active_block_list(bid as BlockId);
      }
      shadow[bid] = (BlockState::Empty, LogId::MAX, 0, 0);
      shadow_edt[bid] = 0;
    }
  }
  assert_eq!(
    mgr.get_block_list_size(BlockState::Empty),
    BLOCKS as usize,
    "终态 free == 初始"
  );
  for st in [
    BlockState::Open,
    BlockState::InUse,
    BlockState::Full,
    BlockState::GcRead,
    BlockState::GcWrite,
    BlockState::GcInUse,
    BlockState::FullPad,
  ] {
    assert_eq!(mgr.get_block_list_size(st), 0, "{st:?} 列表残留");
  }
  assert_eq!(
    mgr.sspace.read().active_block_cnt.load(Ordering::Relaxed),
    0
  );
  check(&shadow, &mgr);
}

#[test]
fn shadow_model_20k() {
  shadow_run(20_000, 0xC0FFEE);
}

/// A2 证据档：1e5 步（`cargo nextest run -p ldb_space shadow_model_100k -- --ignored`）
#[test]
#[ignore]
fn shadow_model_100k() {
  shadow_run(100_000, 0x5EED);
}

/// T6 写缓冲读路径（C-18 + 读缓存跨缓冲错配修复）：
/// M-3（组缓冲页错按 GC 缓冲切片）
#[test]
fn write_buffer_read_paths() {
  let mut io = PageIo::new(2, 8);
  let mut rng = Rng::with_seed(11);
  let mut page = [0u8; PAGE_SIZE];
  rng.fill(&mut page);
  // pid 0 → 组 0：可压缩页（重复字节）
  let mut comp_page = [0u8; PAGE_SIZE];
  comp_page[..100].fill(7);
  assert!(io.add_page_to_write_buffer(0, &comp_page, 10));
  assert!(io.add_page_to_gc_write_buffer(9, &page, 20));

  // 组缓冲命中（M-3：修复前 read_cache 命中按 GC 缓冲切片 → 错页）
  let mut out = [0u8; PAGE_SIZE];
  assert!(
    matches!(io.read_page(0, &mut out), Ok(true)),
    "组缓冲页命中"
  );
  assert_eq!(out, comp_page, "M-3: 组缓冲页必须从组缓冲读出");
  assert!(matches!(io.read_page(1, &mut out), Ok(false)), "未命中");
  assert!(matches!(io.read_page(9, &mut out), Ok(true)));
  assert_eq!(out, page, "GC 缓冲原文直存往返");
  assert!(matches!(io.read_page(u64::MAX, &mut out), Ok(false)));

  // page_in_* 家族
  assert!(io.page_in_user_w_buffer(0));
  assert!(io.page_in_user_w_buffer_per_group(0, 0));
  assert!(!io.page_in_user_w_buffer_per_group(0, 1));
  assert!(io.page_in_gcw_buffer(9));
  assert!(io.page_in_w_buffer(9));
  assert!(io.page_in_read_buffer(0));
  assert!(io.page_in_read_buffer(9));
  assert!(!io.page_in_read_buffer(u64::MAX));

  // 批量读
  let pids = [0u64, 9, 7];
  let mut bufs = [[0u8; PAGE_SIZE]; 3];
  let hits = io.read_pages_out_of_place(&pids, &mut bufs);
  assert!(matches!(hits[0], Ok(true)));
  assert!(matches!(hits[1], Ok(true)));
  assert!(matches!(hits[2], Ok(false)));

  // GC 写缓冲批量刷出（bins 独立于组缓冲；刷出后 GC 缓冲清零）
  let (gc_data, gc_infos) = io.flush_gc_write_buffer().expect("GC 缓冲非空");
  assert_eq!(gc_data.len(), PAGE_SIZE, "单页原文对齐");
  assert_eq!(gc_infos[0].pid, 9);
  assert!(io.flush_gc_write_buffer().is_none(), "刷出后为空");
  assert!(!io.page_in_gcw_buffer(9));

  // 访问器族
  assert_eq!(io.get_wbuffer_pid_cnt_per_group(0), 1);
  assert_eq!(io.get_wbuffer_pid_per_group(0), vec![0]);
  assert!(io.get_wbuffer_size_per_group(0) > 0);
  assert_eq!(io.get_wbuffer_write_cnt(0), 1);
  assert_eq!(
    io.get_wbuffer_write_size(0),
    io.get_wbuffer_size_per_group(0).div_ceil(PAGE_SIZE as u64) * PAGE_SIZE as u64
  );
  io.update_cached_page_info(0, io.groups[0].zpage.zpage_info[0]);
  io.update_read_cached_page_info(123, ZipInfo::default());
  assert!(io.page_in_read_buffer(123));

  // 重置族（bins 同步回零基）；批量刷出只收非空组
  io.add_page_to_write_buffer(2, &comp_page, 30);
  let flushed = io.flush_write_buffer_per_group();
  assert_eq!(flushed.len(), 1, "只有组 0 非空");
  assert_eq!(flushed[0].0, 0, "组号");
  assert!(io.flush_write_buffer_per_group().is_empty(), "全空后无产出");
  io.reset_w_buffer_per_group(0);
  assert_eq!(io.get_wbuffer_pid_cnt_per_group(0), 0);
  assert!(
    io.groups[0]
      .bins
      .bin
      .iter()
      .all(|b| b.sum == 0 && b.count == 0)
  );
  io.reset_buffer(0);
  io.reset_buffer(1);
  io.reset_buffer(2);
  assert!(!io.page_in_read_buffer(123), "reset_buffer(2) 清读缓存");
}

/// T7 坏页 → Err → 上层可重试（ldb_codec CROSS 收口；M-1/M-2）
#[test]
fn bad_page_propagates_err() {
  let mut io = PageIo::new(1, 4);
  let mut good = [0u8; PAGE_SIZE];
  good[..64].fill(0xAB);
  assert!(io.add_page_to_gc_write_buffer(5, &good, 1));
  let info = io.gc_write.zpage_info[0];

  // 正常读回
  let mut out = [0u8; PAGE_SIZE];
  io.read_page_out_of_place(&info, &mut out)
    .unwrap_or_else(|e| panic!("好页必须可读: {e}"));
  assert_eq!(out, good);

  // 坏压缩帧：破坏缓冲后按原元信息读 → 必须 Err，不得静默返回旧数据（M-1）
  io.gc_write.buffer[..16].fill(0xFF);
  assert!(
    io.read_page_out_of_place(&info, &mut out).is_err(),
    "M-1: 坏压缩页必须 Err"
  );
  // 上层可重试语义：同参数重读仍 Err（无内部状态污染）
  assert!(io.read_page_out_of_place(&info, &mut out).is_err());

  // start_ptr 越界 → Err(OffsetOutOfBounds)，禁止 panic（M-2）
  let mut bad = info;
  bad.start_ptr = u32::MAX;
  let e = io.read_page_out_of_place(&bad, &mut out).unwrap_err();
  assert!(matches!(e, Error::OffsetOutOfBounds { .. }), "实际 {e:?}");
  let e2 = io.handle_uncompressed_read(&bad, &mut out).unwrap_err();
  assert!(matches!(e2, Error::OffsetOutOfBounds { .. }));
  let e3 = io.handle_compressed_read(&bad, &mut out).unwrap_err();
  assert!(matches!(e3, Error::OffsetOutOfBounds { .. }));
  let e4 = io
    .handle_compressed_binpacked_read(&bad, &mut out)
    .unwrap_err();
  assert!(matches!(e4, Error::OffsetOutOfBounds { .. }));
  let e5 = io.read_page_in_place(&mut out[..], u64::MAX);
  assert!(matches!(e5, Err(Error::OffsetOutOfBounds { .. })));
  // 分组缓冲路径同样 Err（read_page 走组扫描）
  let mut io2 = PageIo::new(1, 4);
  assert!(io2.add_page_to_write_buffer(1, &good, 1));
  assert!(matches!(io2.read_page(1, &mut out), Ok(true)));
}

/// 手工布局压缩页（绕过压缩器，comp_sz 精确可控）：顺序写 buffer + zpage_info，
/// 模拟 `add_page` 产物（start_ptr 顺序累加、cur_size 为顺序末端）
fn stage_pages(zp: &mut ZipPage, specs: &[(u64, u16, LogId, u8)]) {
  let mut off = 0u32;
  for (i, &(pid, sz, edt, fill)) in specs.iter().enumerate() {
    let s = off as usize;
    zp.buffer[s..s + sz as usize].fill(fill);
    zp.zpage_info[i] = ZipInfo {
      pid,
      comp_sz: sz,
      start_ptr: off,
      death_time: edt,
    };
    off += u32::from(sz);
  }
  zp.zpage_cnt = specs.len() as u32;
  zp.cur_size = u64::from(off);
}

/// T8 装箱契约（ldb_pack CROSS 收口）：
/// M-4（只按页数换段 → 跨 4KiB）/ M-5（全局偏移进 Bin.sum → 再装箱 panic）/
/// M-6（不搬数据 → flush 读到旧布局）/ M-7（flush 后 bins 残留）
#[test]
fn bin_packing_contract() {
  // MAX_PAGES_PER_BIN = PAGE_SIZE / MIN_COMP_SIZE = 2
  let mut io = PageIo::new(1, 8);
  stage_pages(
    &mut io.groups[0].zpage,
    &[(0, 3000, 1, 0xAA), (1, 3000, 2, 0xBB)],
  );
  io.uniform_bin_packing(0);
  let g = &io.groups[0];
  // 3000 + 3000 > 4096：页 2 必须换段（M-4：跨界检测）
  assert_eq!(
    g.zpage.zpage_info[1].start_ptr, PAGE_SIZE as u32,
    "M-4: 跨界必须换段"
  );
  assert!(g.bins.bin[0].sum <= PAGE_SIZE as u32, "Bin.sum 槽内契约");
  assert_eq!(g.bins.bin[0].sum, 3000);
  assert_eq!(g.bins.bin[1].sum, 3000);

  // 5 个 512B 页：段上限 2 → 3 个 bin；第 5 页 start_ptr = 2*PAGE_SIZE
  let mut io2 = PageIo::new(1, 8);
  stage_pages(
    &mut io2.groups[0].zpage,
    &[
      (0, 512, 1, 0x11),
      (1, 512, 2, 0x22),
      (2, 512, 3, 0x33),
      (3, 512, 4, 0x44),
      (4, 512, 5, 0x55),
    ],
  );
  let bins_used = io2.uniform_bin_packing(0);
  assert_eq!(bins_used, 3, "MAX_PAGES_PER_BIN=2 段上限");
  let g2 = &io2.groups[0];
  assert_eq!(g2.bins.bin[0].count, 2);
  assert_eq!(g2.bins.bin[0].sum, 2 * 512);
  assert_eq!(g2.bins.bin[2].sum, 512);
  assert_eq!(g2.zpage.zpage_info[4].start_ptr, 2 * PAGE_SIZE as u32);
  assert_eq!(
    g2.zpage.cur_size,
    3 * PAGE_SIZE as u64,
    "cur_size 对齐 bin 数"
  );

  // 布局搬移：flush 数据在新 start_ptr 处仍是原页内容（M-6）
  let (data, infos) = io2.flush_write_buffer(0).expect("非空组必须可刷");
  assert_eq!(data.len(), 3 * PAGE_SIZE, "对齐整 bin");
  for z in &infos {
    let fill = (z.pid as u8) * 0x11 + 0x11;
    assert!(
      data[z.start_ptr as usize..z.start_ptr as usize + z.comp_sz as usize]
        .iter()
        .all(|&b| b == fill),
      "M-6: pid={} 新布局内容漂移",
      z.pid
    );
  }
  // flush 后 bins 回零基（M-7）
  assert!(
    io2.groups[0]
      .bins
      .bin
      .iter()
      .all(|b| b.sum == 0 && b.count == 0)
  );

  // uniform 之后多 EDT bin_packing：不得因 stale sum panic（M-5），且按 plan 搬移
  let mut io3 = PageIo::new(1, 8);
  stage_pages(
    &mut io3.groups[0].zpage,
    &[
      (0, 2048, 1, 0xA1),
      (1, 3000, 2, 0xB2),
      (2, 1500, 1, 0xC3),
      (3, 4000, 2, 0xD4),
    ],
  );
  io3.uniform_bin_packing(0);
  let n = io3.bin_packing(0, true, 1, 2);
  assert_eq!(n, 4);
  let g3 = &io3.groups[0];
  assert!(
    g3.bins.bin.iter().all(|b| b.sum <= PAGE_SIZE as u32),
    "M-5: sum 槽内契约"
  );
  assert_eq!(
    g3.zpage.cur_size % PAGE_SIZE as u64,
    0,
    "bin_packing 后对齐"
  );
  let (data3, infos3) = io3.flush_write_buffer(0).unwrap();
  // best-fit：1500 与 2048 同 bin（剩余 548 容不下 3000/4000）→ 3 个 bin
  assert_eq!(data3.len(), 3 * PAGE_SIZE);
  for z in &infos3 {
    let fill = [0xA1u8, 0xB2, 0xC3, 0xD4][z.pid as usize];
    assert!(
      data3[z.start_ptr as usize..z.start_ptr as usize + z.comp_sz as usize]
        .iter()
        .all(|&b| b == fill),
      "M-6: bin_packing 后 pid={} 数据未按 plan 搬移",
      z.pid
    );
  }

  // 非 EDT 路径与单一 EDT 路径走统一装箱；空组返回 0
  let cnt = io3.bin_packing(0, false, 0, 0);
  assert_eq!(cnt, 0, "flush 后空组返回 0");

  // 外部契约：bin 满时 bin_pack_groups 返回 None（调用方显式处置）
  let mut io4 = PageIo::new(1, 8);
  let mut big = [0u8; PAGE_SIZE];
  big[..4000].fill(9);
  assert!(io4.add_page_to_write_buffer(0, &big, 1));
  let mut entries = vec![ldb_pack::ZipEntry {
    comp_sz: 4000,
    death_time: 1,
    src_off: 0,
  }];
  let mut full_bins = vec![ldb_pack::Bin {
    sum: PAGE_SIZE as u32,
    count: 2,
  }];
  let plan = io4.bin_pack_groups(0, &mut entries, &mut full_bins);
  assert_eq!(plan[0], None, "bin 满时返回 None（外部调用方自行处置）");
}

/// T9 ZipPage 生命周期：不可压缩回退（C-19）/ avg_edt / reset / 容量满
#[test]
fn zip_page_lifecycle() {
  let mut zp = ZipPage::new(2);
  let mut rng = Rng::with_seed(7);
  let mut page = [0u8; PAGE_SIZE];
  rng.fill(&mut page);
  assert!(zp.add_page(9, &page, 33));
  assert_eq!(
    zp.zpage_info[0].comp_sz,
    ldb_pack::UNCOMPRESSED,
    "C-19 回退哨兵"
  );
  assert_eq!(zp.zpage_info[0].start_ptr, 0);
  assert_eq!(zp.cur_size, PAGE_SIZE as u64);
  let mut comp = [0u8; PAGE_SIZE];
  comp[..128].fill(3);
  assert!(zp.add_page(10, &comp, 44));
  assert!(zp.zpage_info[1].comp_sz < 128, "重复字节页必可压缩");
  assert_eq!(zp.avg_edt(), (33 + 44) / 2);
  assert!(!zp.add_page(11, &comp, 55), "容量满返回 false");
  zp.reset(0);
  assert_eq!(zp.zpage_cnt, 0);
  assert_eq!(zp.cur_size, 0);
  assert_eq!(zp.avg_edt(), 0, "N-20：空缓冲返回 0");
}

/// T10 EDT 编排（M-P/M-Q/M-R/M-S）
#[test]
fn edt_tracker_orchestration() {
  let mgr = mk_mgr(8);
  let t = EdtTracker;
  let bid = mgr.select_empty_block_id(0, 100);
  // pid 3 映射进被选块的槽 0（min-invalidation 策略取 EMPTY 队首，非固定块 0）
  block_on(mgr.update_pid2_offset(3, bid * mgr.get_block_size(), 4096));

  // calc_edt：水位哨兵（0 → 1）+ 环形插入（M-P/M-S）
  let wm = GsnWatermark::new(0);
  let mut hist = [0u8 as LogId; WRITE_HISTORY_LENGTH];
  let mut wcnt = 0u32;
  let e1 = block_on(t.calc_edt(&mgr, &wm, 3, &mut hist, &mut wcnt));
  assert_eq!(wcnt, 1);
  assert_eq!(hist[1], 1, "M-P: 全零历史 max 槽为 0 → 插入槽 1");
  assert_eq!(e1, 1, "M-S: 水位哨兵 + 单点历史 EDT = 该 GSN");
  let e2 = block_on(t.calc_edt(&mgr, &wm, 3, &mut hist, &mut wcnt));
  assert_eq!(hist[2], 1, "第二次插入仍在 max 槽下一环位");
  assert!(e2 >= e1);
  assert_eq!(mgr.block_meta(bid).block_max_edt, e2, "calc_edt 块级登记");

  // min/max 编排（用新块：bid 的 min/max 已被 calc_edt 登记为 1）
  let b2 = mgr.select_empty_block_id(0, 200);
  t.update_min_max_edt(&mgr, b2, 300);
  t.update_min_max_edt(&mgr, b2, 200);
  let m = mgr.block_meta(b2);
  assert_eq!(m.block_min_edt, 200);
  assert_eq!(m.block_max_edt, 300);
  assert!(t.block_edt_is_outdated(&mgr, b2, 301));
  assert!(!t.block_edt_is_outdated(&mgr, b2, 299));

  // CalcBlockAvgEDT 上游公式（space_manager.cc:1476-1487）：
  // total=0 → 直接取新值；round((prev*n + new*k) / (n+k))
  mgr.update_block_metadata_after_write(bid, 4096, 100);
  let m = mgr.block_meta(bid);
  assert_eq!(m.total_pid_cnt, 1);
  assert_eq!(m.edt, 100, "首写直接取新值");
  assert_eq!(t.calc_block_avg_edt(&mgr, bid, 100, 3), 100);
  // total=1,edt=100 再写 new=200,k=1 → round((100+200)/2) = 150
  mgr.update_block_metadata_after_write(bid, 8192, 200);
  assert_eq!(mgr.block_meta(bid).edt, 150);
  // total=2,edt=150 再来一次 new=100,k=1 → round((150*2+100)/3) = 133（M-Q）
  assert_eq!(
    t.calc_block_avg_edt(&mgr, bid, 100, 1),
    133,
    "M-Q: 加权平均"
  );

  t.set_min_max_edt_to_minimum(&mgr, bid);
  assert_eq!(mgr.block_meta(bid).block_min_edt, 0);
  t.remove_min_max_edt(&mgr, bid);
  assert_eq!(mgr.block_meta(bid).block_min_edt, LogId::MAX);

  // GAP / 最高 EDT / 最小 EDT 块
  t.set_min_max_edt_per_block(&mgr, bid, 100, 900);
  assert_eq!(t.get_block_edt_gap(&mgr, bid), 800);
  assert!(t.get_highest_edt(&mgr) >= 900);
  assert_eq!(t.get_min_edt_block_id(&mgr), bid, "OPEN 列表内最小 EDT 块");
  // 最近 EDT + 分组亲和（bid=7 落写组 1；bid.edt=150 距 avg=100 最近）
  assert_eq!(
    mgr.select_open_block_id_with_closest_edt(100, (bid % 2) as usize),
    bid,
    "最近 EDT（写组亲和）"
  );

  // adjust_edt_range：[900, 2000) 洞必须被补（M-R）
  let hole_block = mgr.select_empty_block_id(0, 2_000);
  t.set_min_max_edt_per_block(&mgr, hole_block, 2_000, 2_100);
  let touched = mgr.adjust_edt_range(100, 2_000);
  assert!(touched.is_some(), "存在洞必须返回被扩展块");
  assert!(
    mgr.block_meta(bid).block_max_edt >= 2_000 || mgr.block_meta(hole_block).block_min_edt <= 900,
    "M-R: 洞 [900,2000) 必须被补"
  );

  // 分组 / 排序 / 唯一集合
  let infos = vec![
    ZipInfo {
      death_time: 5,
      ..Default::default()
    },
    ZipInfo {
      death_time: 1,
      ..Default::default()
    },
    ZipInfo {
      death_time: 5,
      ..Default::default()
    },
  ];
  let mut sorted = infos.clone();
  t.sort_pages_by_edt(&mut sorted);
  assert_eq!(
    sorted.iter().map(|z| z.death_time).collect::<Vec<_>>(),
    vec![1, 5, 5]
  );
  assert_eq!(t.extract_unique_edts(&infos), vec![1, 5]);
  let ranges = t.group_pages_by_edt(&infos, 1, 5);
  assert_eq!(ranges.first().copied(), Some((0, infos.len() as u32)));
  assert_eq!(t.group_pages_by_edt(&infos, 5, 5), vec![(0, 3)]);

  // GCW 最小 EDT（N-20）
  let mut io = PageIo::new(1, 4);
  assert_eq!(t.get_min_edt_from_gcw_buffer(&io), LogId::MAX, "空缓冲哨兵");
  let mut p = [0u8; PAGE_SIZE];
  p[..32].fill(1);
  assert!(io.add_page_to_gc_write_buffer(1, &p, 77));
  assert!(io.add_page_to_gc_write_buffer(2, &p, 12));
  assert_eq!(t.get_min_edt_from_gcw_buffer(&io), 12);

  // LSN 间隔纯计算半区
  assert_eq!(t.update_lsn_gap(&wm, 100), 100, "水位 0 → 全量差");
}

/// T10b adjust_edt_range 补洞归属：单遍按活跃表序取首个命中「上界==gap_start 或
/// 下界==gap_end」的块（上游 space_manager.cc:584-608）。区间交叉时若先全局找上界
/// 匹配再找下界匹配，会把洞归给活跃表序更靠后的块 —— 本用例唯一钉死活跃表序优先。
#[test]
fn adjust_edt_range_active_order_priority() {
  let mgr = mk_mgr(8);
  // 活跃表序：b1(先开), b2, b3
  let b1 = mgr.select_empty_block_id(0, 10);
  let b2 = mgr.select_empty_block_id(0, 20);
  let b3 = mgr.select_empty_block_id(0, 30);
  {
    let mut s = mgr.sspace.write();
    // b2 命中「下界==gap_end」，b3 命中「上界==gap_start」，但 b2 在活跃表序更前
    for (b, lo, hi) in [(b1, 2774, 4317), (b2, 1670, 2470), (b3, 734, 1314)] {
      let m = &mut s.block_metadata[b as usize];
      m.block_min_edt = lo;
      m.block_max_edt = hi;
    }
  }
  let _ = mgr.adjust_edt_range(30, 398);
  // 洞 [1314,1670)：上游按活跃表序取 b2（下界下探到 1314），而非 b3（上界抬到 1670）
  assert_eq!(
    mgr.block_meta(b2).block_min_edt,
    1314,
    "活跃表序在前的匹配块 b2 应被扩展（下界下探）"
  );
  assert_eq!(
    mgr.block_meta(b3).block_max_edt,
    1314,
    "活跃表序在后的 b3 上界不得被抬到 1670（旧全局扫描 bug）"
  );
  assert_eq!(
    mgr.block_meta(b3).block_min_edt,
    30,
    "洞 [30,734) 归 b3 下探"
  );
}

/// T11 NoWA padding + 关闭路径 + 计数器族（M-T/M-U/M-V/M-PAD2）
#[test]
fn padding_and_counters() {
  let mgr = mk_mgr(8);
  let mut io = PageIo::new(1, 8);

  // M-T: InUse 未满谓词（补写入口态是关闭后的 FULL_PAD，上游 :1388）
  let bid = mgr.select_empty_block_id(0, 1);
  let _ = mgr.move_block_to_list(bid, BlockState::Open, BlockState::InUse);
  assert!(
    mgr.check_open_block_for_padding(bid),
    "M-T: InUse 未满可判补写"
  );

  // 未满关闭 → FullPad 待补写（上游 CloseOneOpenBlock :1284 OPEN→FULL_PAD）
  assert!(mgr.close_one_open_block(bid, 0));
  assert_eq!(mgr.block_meta(bid).state, BlockState::FullPad);

  // padding：staged 8 零页 → Full + cur_size 补齐 + 撤活跃表（上游 :1421/:1429-1435）
  let padded = block_on(mgr.issue_padding_writes_to_full_block(&mut io, bid, true));
  assert_eq!(padded, 8, "M-U: 从 0 补到 max_w 需要 8 个零页");
  assert_eq!(mgr.block_meta(bid).state, BlockState::Full, "补写收 FULL");
  assert_eq!(
    mgr.block_meta(bid).cur_size,
    mgr.get_block_size(),
    "cur_size 簿记补齐（上游 :1421）"
  );
  assert!(
    !mgr.block_id_in_active_block_list(bid),
    "补写完成撤活跃表（上游 :1434）"
  );
  // 快路径：已 Full 再补 → 0（上游 :1381-1385）
  assert_eq!(
    block_on(mgr.issue_padding_writes_to_full_block(&mut io, bid, true)),
    0
  );

  // 非 NoWA：不 staging（返回 0）但簿记无条件（上游 :1403 flag 只门控磁盘写）
  let b2 = mgr.select_empty_block_id(0, 1);
  let _ = mgr.move_block_to_list(b2, BlockState::Open, BlockState::InUse);
  assert!(mgr.close_one_open_block(b2, 0));
  assert_eq!(
    block_on(mgr.issue_padding_writes_to_full_block(&mut io, b2, false)),
    0,
    "非 NoWA 不 staging"
  );
  assert_eq!(
    mgr.block_meta(b2).state,
    BlockState::Full,
    "簿记无条件收 FULL"
  );
  assert_eq!(mgr.block_meta(b2).cur_size, mgr.get_block_size());

  // 已满关闭直 Full（上游 fast-path :1381-1385）
  let b5 = mgr.select_empty_block_id(0, 1);
  let _ = mgr.move_block_to_list(b5, BlockState::Open, BlockState::InUse);
  assert!(mgr.close_one_open_block(b5, mgr.get_block_size()));
  assert_eq!(mgr.block_meta(b5).state, BlockState::Full);
  assert!(!mgr.block_id_in_active_block_list(b5));

  // move_padded：FullPad 未补写直接收 Full（上游 MovePaddedBlocksToFullList :1085-1093）
  let b4 = mgr.select_empty_block_id(0, 1);
  let _ = mgr.move_block_to_list(b4, BlockState::Open, BlockState::InUse);
  assert!(mgr.close_one_open_block(b4, 0));
  assert!(mgr.move_padded_blocks_to_full_list(b4));
  assert_eq!(mgr.block_meta(b4).state, BlockState::Full);
  assert!(
    mgr.remove_block_id_from_active_block_list(b4),
    "手动收尾撤表"
  );

  // 计数器族（独立 InUse 块）
  let b6 = mgr.select_empty_block_id(0, 1);
  let _ = mgr.move_block_to_list(b6, BlockState::Open, BlockState::InUse);
  mgr.update_block_metadata_after_write(b6, 4096, 7);
  assert_eq!(mgr.total_page_cnt_in_block(b6), 1);
  assert_eq!(mgr.count_written_pi_ds_in_block(b6), 1);
  assert_eq!(mgr.valid_pi_ds_cnt_in_block(b6), 1);
  mgr.update_block_metadata_after_async_gc_write(b6, 4096, 8);
  assert_eq!(mgr.total_page_cnt_in_block(b6), 2);
  mgr.update_block_metadata_after_sync_gc_write(b6, 4096, 9);
  assert_eq!(mgr.total_page_cnt_in_block(b6), 3);
  assert!(!mgr.block_needs_gc(b6), "无失效不计 GC");
  assert_eq!(mgr.block_gc_read_cnt(b6), 0);
  // b6 未满关闭（4096 < max_w）→ FullPad 待补写（补写收尾由 ldb_buf 驱动）
  assert!(mgr.close_one_open_block(b6, 4096));
  assert_eq!(mgr.block_meta(b6).state, BlockState::FullPad);

  // close_all 只收 InUse 态（b6 已 FullPad，仅 b3 InUse）
  let b3 = mgr.select_empty_block_id(0, 1);
  let _ = mgr.move_block_to_list(b3, BlockState::Open, BlockState::InUse);
  assert_eq!(mgr.close_all_open_blocks(), 1, "close_all 只收 InUse 态");
  assert_eq!(mgr.block_meta(b3).state, BlockState::FullPad);
  assert!(mgr.move_padded_blocks_to_full_list(b3));
  assert_eq!(mgr.block_meta(b3).state, BlockState::Full);

  // GC 读源选择 + 前次 GC 块
  assert_eq!(mgr.select_full_block_to_gc_read(), b3);
  assert_eq!(mgr.block_gc_read_cnt(b3), 1, "GC_READ 态计数");
  assert!(!mgr.pid_in_gcr_block(0), "未映射 pid 不在 GC 块");
  assert!(!mgr.pid_in_gcw_block(0));
  assert_eq!(mgr.get_prev_g_ced_block(), b3, "M-V: 选中即记录前次 GC 块");
  assert_eq!(mgr.write_group_cnt(), 2);
  mgr.add_write_group_history(0, b3);
  assert_eq!(mgr.sspace.read().write_group[0], vec![b3]);
  assert!(mgr.select_read_g_ced_blocks_to_gc_write().contains(&b3));
  // 活跃表收尾：b6/b3 关闭后仍登记（FullPad 保留 / move_padded 不撤），手动撤出
  assert!(mgr.remove_block_id_from_active_block_list(b6));
  assert!(mgr.remove_block_id_from_active_block_list(b3));
  assert_eq!(
    mgr.empty_slot_count_in_active_block_list(),
    4,
    "全部撤出活跃表"
  );
  assert_eq!(mgr.get_active_block_size(), 4);
  assert_eq!(mgr.gc_in_progess_block_count_in_active_list(), 0);
  assert!(mgr.pid_offset_is_in_block(0, 0));
  assert!(!mgr.pid_offset_is_in_block(1, 0));
  assert_eq!(mgr.get_block_size(), 8 * PAGE_SIZE as u64);
  assert_eq!(mgr.get_block_offset(2), 2 * 8 * PAGE_SIZE as u64);
  assert_eq!(mgr.pid_offset_block(8 * PAGE_SIZE as u64 + 4096), (1, 1));
  assert_eq!(mgr.get_block_write_offset(b6), 4096);
}

/// T12 检查点 + 写组选择 + 耗尽显式报错 + 双表登记（M-W/M-X）
#[test]
fn checkpoint_and_write_group() {
  let mgr = mk_mgr(2);
  block_on(mgr.update_pid2_offset(0, 0, 4096));
  block_on(mgr.update_pid2_offset(1, 4096, 2048));

  // 检查点序列化：[len u64][packed u64]...（M-X）
  let mut sink = MemSink::default();
  let n = block_on(mgr.checkpoint_mapping_table(&mut sink)).unwrap_or_else(|e| panic!("{e}"));
  assert_eq!(n, 2);
  assert_eq!(sink.data.len(), 8 + 2 * 8);
  assert_eq!(u64::from_le_bytes(sink.data[0..8].try_into().unwrap()), 2);
  let packed1 = u64::from_le_bytes(sink.data[8..16].try_into().unwrap());
  assert_eq!(packed1 >> 15, 0, "pid0 偏移 0");
  assert_eq!(packed1 & 0x7FFF, 4096);
  let packed2 = u64::from_le_bytes(sink.data[16..24].try_into().unwrap());
  assert_eq!(packed2 >> 15, 4096, "pid1 偏移 4096");
  assert_eq!(packed2 & 0x7FFF, 2048);
  assert_eq!(mgr.alloc_mapping_table_space(), 2 * 8);

  // 反向表：update_space_metadata 双表登记（M-W）；GC 写基地址 = 目标块起点
  let target = mgr.select_empty_block_id(0, 1);
  let mut io = PageIo::with_base(1, 8, mgr.get_block_offset(target));
  let mut p = [0u8; PAGE_SIZE];
  p[..32].fill(5);
  assert!(io.add_page_to_gc_write_buffer(7, &p, 3));
  let infos = io.gc_write.zpage_info[..1].to_vec();
  block_on(io.update_space_metadata(target, &infos, &mgr));
  let e = mgr.get_pid2_offset(7);
  assert_eq!(
    e.get_offset(),
    mgr.get_block_offset(target) + u64::from(infos[0].start_ptr)
  );
  assert_eq!(e.get_comp_sz(), infos[0].comp_sz);
  assert_eq!(
    mgr.convert_offset2_pids(target, 0),
    vec![7],
    "M-W: 反向表同步登记"
  );
  mgr.reset_offset2_pids(target);
  assert!(mgr.convert_offset2_pids(target, 0).is_empty());

  // 写组选择：分组亲和（无活跃块 → 空块路径）
  assert!(block_on(mgr.select_block_id_for_write_group(0, 10, 0)).is_ok());
  // 2 块占满后超容写入 → 无打开块可容、无空块 → SpaceExhausted（A7：显式 Err 替代忙等）
  let _ = mgr.select_empty_block_id(0, 1);
  let r = block_on(mgr.select_block_id_to_write(9 * PAGE_SIZE as u64, 99, 0, 1024));
  assert!(matches!(r, Err(Error::SpaceExhausted)), "实际 {r:?}");
  mgr.destroy();

  // 空块管理器：直接耗尽
  let empty = mk_mgr(0);
  let r2 = block_on(empty.select_block_id_to_write(1, 0, 0, 0));
  assert!(matches!(r2, Err(Error::SpaceExhausted)));
  assert_eq!(empty.get_db_size(), 0);
  assert_eq!(empty.get_gc_target_size(), 0);
}

/// 检查点内存落盘目标（独立 mapping.ckpt 的测试替身）
#[derive(Default)]
struct MemSink {
  data: Vec<u8>,
}
impl Sink for MemSink {
  fn write_all_at(
    &mut self,
    _off: u64,
    data: Vec<u8>,
  ) -> Pin<Box<dyn Future<Output = io::Result<()>> + '_>> {
    self.data = data;
    Box::pin(future::ready(Ok(())))
  }
}
