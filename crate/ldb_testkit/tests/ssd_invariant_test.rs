//! 不变量与口径回归（清单 A1）：
//! - 新建块保持「未经历 GC」态（gcAge==-1、eraseCount==0、擦除态）；
//! - GC 仅重写有效页，且搬运后映射仍指向有效数据（论文 §GC「有效比 / WAF」）；
//! - SSD WAF 分母是主机逻辑写（上游 `sim.cpp:81`），WL 搬移产生写放大时应 >1；
//! - 磨损均衡的写块必须留在本 LUN 内（LUN 亲和）。

use std::panic::{AssertUnwindSafe, catch_unwind};

use ldb_testkit::{Ssd, UNUSED};

fn sim() -> Ssd {
  // 8 MiB / 64 KiB 区块 / 4 KiB 页 = 128 块 × 16 页；填充 50% = 1024 逻辑页
  Ssd::new(8 << 20, 64 << 10, 4096, 0.5)
}

#[test]
fn fresh_blocks_have_negative_gc_age() {
  let s = sim();
  for b in s.blocks() {
    assert!(b.is_erased(), "块应初始即擦除态");
    assert_eq!(
      b.erase_count, 0,
      "新建块擦除计数须为 0（不得在构造时 erase）"
    );
    assert_eq!(b.gc_age, -1, "新建块 gcAge 须为 -1（未经历 GC）");
  }
}

#[test]
fn gc_rewrites_only_valid_pages() {
  let s = sim();
  // 块 0 写满 16 页（全有效）
  for i in 0..16u64 {
    s.write_page_without_caching(i, 0, -1);
  }
  // 覆写 log 0..4 到块 1 → 块 0 剩 12 有效（VR=0.75），块 1 得 4
  for i in 0..4u64 {
    s.write_page_without_caching(i, 1, -1);
  }
  assert_eq!(s.blocks()[0].valid_cnt(), 12);
  let phys_before = s.phys_writes();

  s.compact_block(0);
  // 论文 §GC：GC 只重写有效页 —— 12 次物理写回收 4 个失效槽
  assert_eq!(
    s.phys_writes() - phys_before,
    12,
    "GC 物理写数须等于有效页数（VR=0.75 → 12/16）"
  );
  assert_eq!(s.blocks()[0].write_pos(), 12);
  assert_eq!(s.blocks()[0].valid_cnt(), 12);
  // 映射仍指向有效数据：0..4 在块 1，4..16 在块 0
  for i in 0..4u64 {
    let a = s.ltp_mapping()[i as usize];
    assert_ne!(a, UNUSED);
    assert_eq!(s.get_zone(a), 1, "被覆写页应指向块 1");
  }
  for i in 4..16u64 {
    let a = s.ltp_mapping()[i as usize];
    assert_ne!(a, UNUSED);
    assert_eq!(s.get_zone(a), 0, "存活页应仍在块 0");
  }
}

#[test]
fn ssd_waf_denominator_is_host_writes() {
  let s = sim();
  s.set_wear_leveling(true);
  s.set_wear_leveling_threshold(4);
  // 小工作集（log 0..4，分属 4 个 LUN）：覆写使每块有效页恒 ≤1，块不会写满，
  // 而 LUN 轮转仍不断把有效页搬入新块 → 物理写 > 主机写 → WAF > 1。
  let mut host = 0u64;
  for i in 0..2000u64 {
    s.write_page_wl(i % 4, -1);
    host += 1;
  }
  assert_eq!(s.logical_writes(), host, "逻辑写数=主机写次数");
  let phys = s.phys_writes();
  assert!(
    phys > host,
    "WL 轮转搬移应产生写放大：phys={phys} host={host}"
  );
  let wa = s.get_ssdwa();
  assert!(wa > 1.0, "写放大须 >1：{wa}");
  assert!(
    (wa - phys as f64 / host as f64).abs() < 1e-12,
    "WAF 须等于 phys/host：{wa}"
  );
}

#[test]
fn ssd_waf_counts_gc_in_numerator() {
  // 三类负载手算 WAF（分子含 GC 搬移/compact，分母仅主机写）：
  // W1 纯追加 16 页（块 0 写满）→ 16/16 = 1.0
  let s = sim();
  for i in 0..16u64 {
    s.write_page(i, -1);
  }
  assert_eq!(s.logical_writes(), 16);
  assert_eq!(s.phys_writes(), 16);
  assert!((s.get_ssdwa() - 1.0).abs() < 1e-12, "W1 纯追加 WAF=1");

  // W2 覆盖同一页 16 次（无 GC 触发）→ 16/16 = 1.0
  let s = sim();
  for _ in 0..16u64 {
    s.write_page(0, -1);
  }
  assert_eq!(s.logical_writes(), 16);
  assert!((s.get_ssdwa() - 1.0).abs() < 1e-12, "W2 覆盖无 GC WAF=1");

  // W3 覆盖 + GC：16 主机写填块 0；8 页 GC 直写搬入块 1（+8）；compact 块 0 重写 8 有效页（+8）
  //        → 32 物理写 / 16 主机写 = 2.0。若 GC 搬移不计入分子则为 1.0。
  let s = sim();
  for i in 0..16u64 {
    s.write_page(i, -1);
  }
  for i in 0..8u64 {
    s.write_page_without_caching(i, 1, -1);
  }
  s.compact_block(0);
  assert_eq!(s.phys_writes(), 32, "GC 搬移与 compact 重写均计入分子");
  assert_eq!(s.logical_writes(), 16, "分母只计主机写，不含 GC");
  assert!((s.get_ssdwa() - 2.0).abs() < 1e-12, "W3 覆盖+GC WAF=2.0");
}

#[test]
fn compact_until_free_block_accumulates_gc_stats() {
  let s = sim();
  // 填满块 0、块 1（各 16 有效）
  for i in 0..32u64 {
    s.write_page_without_caching(i, i / 16, -1);
  }
  // 块 0 的 log 0..8 覆写到块 2 → 块 0 剩 8 有效
  for i in 0..8u64 {
    s.write_page_without_caching(i, 2, -1);
  }
  let (_, n0, c0) = s.stats();
  assert_eq!((n0, c0), (0, 0), "前置 gced 统计应为 0");

  let order = [0u64, 1, 2, 3, 4, 5, 6, 7];
  let mut idx = 0usize;
  let (freed, _) = s.compact_until_free_block(-1, || {
    let b = order[idx % order.len()];
    idx += 1;
    b
  });
  assert!(s.blocks()[freed as usize].is_erased(), "回收块须处擦除态");

  // move 路径（gcedNormal/Cold）与 compact 路径都必须累加统计（旧内联实现丢失）
  let (p1, n1, c1) = s.stats();
  assert!(p1 > 0, "GC 须产生物理写");
  assert!(n1 > 0, "move 有效页须累加 gcedNormal，实测 {n1}");
  assert!(c1 > 0, "compact 须累加 gcedCold，实测 {c1}");
}

#[test]
fn wear_leveling_keeps_blocks_in_own_lun() {
  let s = sim();
  s.set_wear_leveling(true);
  s.set_wear_leveling_threshold(2);
  let lun_cnt = s.wear_leveling_lun_cnt();
  assert_eq!(lun_cnt, 64, "LUN 数 = min(64, zones)");
  // 只写映射到 LUN 0 的逻辑页（log % 64 == 0），交替覆写两个页使块永不写满
  for i in 0..60u64 {
    s.write_page_wl((i % 2) * 64, -1);
  }
  // 凡被使用的块（写过或被 WL 擦过）必须都属于 LUN 0（块号 % 64 == 0）
  for b in s.blocks() {
    if b.write_pos() > 0 || b.erase_count > 0 {
      assert_eq!(
        b.block_id % lun_cnt,
        0,
        "块 {} 不属于 LUN 0（LUN 亲和被破坏）",
        b.block_id
      );
    }
  }
}

fn panics(f: impl FnOnce()) -> bool {
  catch_unwind(AssertUnwindSafe(f)).is_err()
}

/// 越界索引必须「确定 panic（不得 UB）」，且逻辑页越界须先于块变更失败。
#[test]
fn out_of_range_indices_panic_deterministically() {
  // sim() 的 logical_pages = (8MiB/4KiB)*0.5 = 1024；边界内最后一页正常。
  let s = sim();
  s.write_page_without_caching(1023, 0, -1);
  assert_eq!(s.blocks()[0].valid_cnt(), 1, "log_page=容量-1 正常写入");

  // log_page = 容量 / 容量+1 / u64::MAX：须 panic，且 panic 前不得改动块（对齐上游
  // `SSD.hpp:364 _ltpMapping.at(logPage)` 的先抛语义），不得计入物理写。
  for lp in [1024u64, 1025, u64::MAX] {
    let s2 = sim();
    assert!(
      panics(|| s2.write_page_without_caching(lp, 0, -1)),
      "log_page={lp} 越界须 panic"
    );
    assert_eq!(
      s2.blocks()[0].valid_cnt(),
      0,
      "log_page={lp} 越界 panic 前不得改动块 0"
    );
    assert_eq!(s2.phys_writes(), 0, "log_page={lp} 越界不得计入物理写");
  }

  // 块号越界（zones=128）须 panic
  let s3 = sim();
  assert!(
    panics(|| s3.write_page_without_caching(0, 9999, -1)),
    "block_id 越界须 panic"
  );

  // compact_until_free_block 的 next() 返回越界块须 panic（不得 UB / 越界索引）
  let s4 = sim();
  for i in 0..32u64 {
    s4.write_page_without_caching(i, i / 16, -1);
  }
  for i in 0..8u64 {
    s4.write_page_without_caching(i, 2, -1);
  }
  assert!(
    panics(|| {
      s4.compact_until_free_block(-1, || 9999u64);
    }),
    "next() 越界块须 panic"
  );
}

/// `compact_until_free_block` 的 `next()` 契约：返回 gc 块自身会造成自我搬移，
/// 必须「确定 panic」而非死循环。
#[test]
fn compact_until_free_block_rejects_self_victim() {
  let s = sim();
  for i in 0..32u64 {
    s.write_page_without_caching(i, i / 16, -1);
  }
  for i in 0..8u64 {
    s.write_page_without_caching(i, 2, -1);
  }
  // next() 恒返回块 0：第 1 次当选 gc 块，第 2 次即 victim == gc 块（契约违例）。
  let mut calls = 0u64;
  let panicked = panics(|| {
    s.compact_until_free_block(-1, || {
      calls += 1;
      if calls > 64 {
        panic!("R3: next() 被调用 >64 次 —— 疑似死循环");
      }
      0u64
    });
  });
  assert!(panicked, "victim == gc 块须 panic");
  assert!(
    calls <= 4,
    "契约违例须尽早 panic，实际 next() 调用 {calls} 次"
  );
}
