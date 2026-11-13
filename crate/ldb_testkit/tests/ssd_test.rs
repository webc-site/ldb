//! SSD 模拟器回归：写映射 / 失效 / GC 压缩 / WL / WAF 度量。

use ldb_testkit::*;

fn sim() -> Ssd {
  // 8 MiB / 64 KiB 区块 / 4 KiB 页 = 128 块 × 16 页；填充 50% = 1024 逻辑页
  Ssd::new(8 << 20, 64 << 10, 4096, 0.5)
}

#[test]
fn write_maps_and_invalidates() {
  let s = sim();
  assert_eq!(s.blocks().len(), 128);
  s.write_page(0, -1);
  s.write_page(0, -1); // 覆写 → 旧物理位失效
  assert_eq!(s.phys_writes(), 2, "两次物理写");
  let b0 = &s.blocks()[0];
  assert_eq!(b0.valid_cnt(), 1, "旧位已失效");
  let addr = s.ltp_mapping()[0];
  assert_eq!(s.get_zone(addr), 0);
}

#[test]
fn compact_reclaims_invalid() {
  let s = sim();
  for i in 0..16u64 {
    s.write_page(i, -1);
  }
  // 失效一半
  for i in (0..16u64).step_by(2) {
    let addr = s.ltp_mapping()[i as usize];
    s.blocks(); // 快照
    let z = s.get_zone(addr);
    let p = s.get_page(addr);
    // 经 set_unused 语义：通过写覆盖自动失效 —— 直接再写同逻辑页到新位
    let _ = (z, p);
    s.write_page_without_caching(i, 1, -1);
  }
  let before = s.blocks()[0].valid_cnt();
  assert_eq!(before, 8, "块 0 半失效");
  s.compact_block(0);
  let after = s.blocks()[0].valid_cnt();
  assert_eq!(after, 8, "压缩保留有效页");
  assert!(s.blocks()[0].written_by_gc);
  // 映射仍指向有效数据
  assert_ne!(s.ltp_mapping()[1], UNUSED);
}

#[test]
fn move_pages_to_destination() {
  let s = sim();
  for i in 0..6u64 {
    s.write_page_without_caching(i, 0, -1);
  }
  // 失效 2 页（块 0 有效 4）
  s.write_page_without_caching(0, 1, -1);
  s.write_page_without_caching(1, 1, -1);
  let residue = s.move_valid_pages_to(0, 2);
  assert!(!residue, "块 0 全无效后搬迁完成");
  assert_eq!(s.blocks()[2].valid_cnt(), 4);
  assert!(s.blocks()[2].written_by_gc);
}

#[test]
fn compact_until_free_produces_erased() {
  let s = sim();
  // 填两块
  for i in 0..32u64 {
    s.write_page_without_caching(i, i / 16, -1);
  }
  // 失效块 0 的一半 → GC 可回收
  for i in 0..8u64 {
    s.write_page_without_caching(i, 2, -1);
  }
  let order = [0u64, 1, 2, 3];
  let mut idx = 0;
  let (freed, _) = s.compact_until_free_block(-1, || {
    let b = order[idx % order.len()];
    idx += 1;
    b
  });
  assert!(s.blocks()[freed as usize].is_erased(), "回收块应处擦除态");
}

#[test]
fn wear_leveling_round_robin() {
  let s = sim();
  s.set_wear_leveling(true);
  s.set_wear_leveling_threshold(8);
  assert!(s.wear_leveling_lun_cnt() > 1);
  // 高频写同 LUN → 阈值轮转
  for i in 0..64u64 {
    s.write_page_wl(i % 4, -1);
  }
  // WL 生效：写分布在多个块（物理写 ≥ 逻辑写）
  assert!(s.phys_writes() >= 64);
}

#[test]
fn waf_metrics_and_hack() {
  let s = sim();
  // 块 0 共 16 槽：8 新写 + 8 覆写 = 16 物理写 / 16 逻辑更新 → WAF = 1
  for i in 0..8u64 {
    s.write_page(i, -1);
  }
  for i in 0..8u64 {
    s.write_page(i, -1); // 覆写：旧位失效 + 新位
  }
  let wa = s.get_ssdwa();
  assert!(wa >= 1.0, "WAF 至少 1: {wa}");
  s.hack_for_optimal_wa_set_phys_writes(16);
  assert!((s.get_ssdwa() - 1.0).abs() < 1e-9, "NoWA 基线注入");
  s.reset_physical_counters();
  assert_eq!(s.phys_writes(), 0);
}

#[test]
fn block_lifecycle_predicates() {
  let s = sim();
  let b = &s.blocks()[5];
  assert!(b.is_erased());
  assert!(b.can_write());
  assert!(b.all_invalid());
  assert!(!b.fully_written());
  assert_eq!(b.write_pos(), 0);
  assert_eq!(s.block_to_lun(5), 5 % s.wear_leveling_lun_cnt());
  assert_eq!(s.log_page_to_lun(9), 9 % s.wear_leveling_lun_cnt());
  assert_eq!(s.get_addr(3, 7), 3 * 16 + 7);
  let _ = s.print_info();
  let _ = s.print_blocks_stats();
  let json = s.write_stats_file();
  assert!(json.starts_with('{') && json.ends_with('}'));
}
