//! 影子模型对拍（清单 A2）：独立复刻 SSD 页状态机（页写入 / 失效 / compact 搬移 /
//! LTP 映射 / 物理写与映射统计），与 `Ssd` 内部状态逐操作比对 ≥1e5 次；并断言不变量：
//! 有效页数守恒、写区无空洞（块内页不越界）、ltp_mapping 与影子一一对应、统计口径一致。

use ldb_testkit::{Ssd, UNUSED};

const ZONES: usize = 4;
const PPZ: usize = 16;
const LOGICAL: usize = 32;
const OPS: u64 = 100_000;
/// 全量核对间隔（含 ltp_mapping 逐项与统计向量）
const FULL_EVERY: u64 = 256;

/// 固定种子 LCG（确定性；禁止系统熵）
struct Rng(u64);
impl Rng {
  fn next(&mut self) -> u64 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (self.0 >> 11) & 0xFFFF_FFFF
  }
}

/// 影子模型：与 `Ssd` 完全独立的状态复刻
struct Shadow {
  /// block -> slot -> log page（UNUSED 为空）
  lba: Vec<Vec<u64>>,
  /// block -> 写指针
  wpos: Vec<u64>,
  /// log page -> (block, slot)
  owner: Vec<Option<(u64, u64)>>,
  phys: u64,
  host: u64,
  /// 每逻辑页的映射更新次数（= mapping_updated_cnt）
  map_upd: Vec<u64>,
  /// 每逻辑页的 GC 映射更新次数（= mapping_updated_gc）
  map_upd_gc: Vec<u64>,
}

impl Shadow {
  fn new() -> Self {
    Self {
      lba: vec![vec![UNUSED; PPZ]; ZONES],
      wpos: vec![0; ZONES],
      owner: vec![None; LOGICAL],
      phys: 0,
      host: 0,
      map_upd: vec![0; LOGICAL],
      map_upd_gc: vec![0; LOGICAL],
    }
  }

  fn valid_in(&self, b: usize) -> u64 {
    self.lba[b].iter().filter(|&&x| x != UNUSED).count() as u64
  }

  /// 主机写一页（等价 `write_page_without_caching`）：旧位失效 + 追加 + 统计
  fn host_write(&mut self, log: u64, blk: u64) {
    let (blk, log) = (blk as usize, log as usize);
    self.invalidate_owner(log);
    let slot = self.wpos[blk];
    assert!(slot < PPZ as u64, "影子：目标块已满");
    assert_eq!(self.lba[blk][slot as usize], UNUSED, "影子：槽位应空闲");
    self.lba[blk][slot as usize] = log as u64;
    self.wpos[blk] = slot + 1;
    self.owner[log] = Some((blk as u64, slot));
    self.phys += 1;
    self.host += 1;
    self.map_upd[log] += 1;
  }

  fn invalidate_owner(&mut self, log: usize) {
    if let Some((ob, os)) = self.owner[log] {
      assert_eq!(
        self.lba[ob as usize][os as usize], log as u64,
        "影子：owner 不一致"
      );
      self.lba[ob as usize][os as usize] = UNUSED;
    }
  }

  /// compact（等价 `compactBlock`）：页序紧凑 + LTP 回写 + phys/gc 统计
  fn compact(&mut self, blk: u64) {
    let b = blk as usize;
    let mut wp = 0usize;
    for s in 0..PPZ {
      let lp = self.lba[b][s];
      if lp != UNUSED {
        self.lba[b][wp] = lp;
        wp += 1;
      }
    }
    self.lba[b][wp..PPZ].fill(UNUSED);
    self.wpos[b] = wp as u64;
    for s in 0..wp {
      let lp = self.lba[b][s] as usize;
      self.owner[lp] = Some((blk, s as u64));
      self.map_upd_gc[lp] += 1;
      self.phys += 1;
    }
  }

  /// move（等价 `moveValidPagesTo`）：把 src 有效页搬入 dst（dst 满则停）
  fn mov(&mut self, src: u64, dst: u64) {
    let s = src as usize;
    let d = dst as usize;
    let mut p = 0usize;
    while p < PPZ && self.wpos[d] < PPZ as u64 {
      let lp = self.lba[s][p];
      if lp != UNUSED {
        let log = lp as usize;
        self.invalidate_owner(log);
        let slot = self.wpos[d];
        self.lba[d][slot as usize] = lp;
        self.wpos[d] = slot + 1;
        self.owner[log] = Some((dst, slot));
        self.phys += 1;
        self.map_upd[log] += 1;
      }
      p += 1;
    }
  }
}

fn sim() -> Ssd {
  // 256 KiB / 64 KiB 区块 / 4 KiB 页 = 4 块 × 16 页；填充 50% = 32 逻辑页
  Ssd::new(256 << 10, 64 << 10, 4096, 0.5)
}

/// 轻量核对：受影响块 + 有效页守恒
fn check_blocks(s: &Ssd, sh: &Shadow, touched: &[usize], ctx: &str) {
  let blocks = s.blocks();
  assert_eq!(blocks.len(), ZONES);
  for &b in touched {
    assert_eq!(blocks[b].write_pos(), sh.wpos[b], "{ctx}: 块 {b} write_pos");
    assert!(
      blocks[b].write_pos() <= PPZ as u64,
      "{ctx}: 块 {b} 写指针越界"
    );
    assert_eq!(
      blocks[b].valid_cnt(),
      sh.valid_in(b),
      "{ctx}: 块 {b} valid_cnt"
    );
    // 写指针之后的尾部槽位必须为空（[0, write_pos) 内允许因 set_unused 产生空洞）
    let ptl = blocks[b].ptl();
    for p in blocks[b].write_pos()..PPZ as u64 {
      assert_eq!(ptl[p as usize], UNUSED, "{ctx}: 块 {b} 写指针后槽位应空闲");
    }
  }
  let total_valid: u64 = blocks.iter().map(|b| b.valid_cnt()).sum();
  let written_logs = sh.owner.iter().filter(|o| o.is_some()).count() as u64;
  assert_eq!(total_valid, written_logs, "{ctx}: 有效页数守恒");
}

/// 逐操作轻量核对：LTP 映射逐项 + 物理写计数（单次 Vec 克隆，开销可控）
fn check_ltp(s: &Ssd, sh: &Shadow, ctx: &str) {
  let ltp = s.ltp_mapping();
  for log in 0..LOGICAL {
    match sh.owner[log] {
      Some((b, slot)) => assert_eq!(ltp[log], b * PPZ as u64 + slot, "{ctx}: ltp[{log}]"),
      None => assert_eq!(ltp[log], UNUSED, "{ctx}: ltp[{log}] 应为 UNUSED"),
    }
  }
  assert_eq!(s.phys_writes(), sh.phys, "{ctx}: phys_writes");
}

/// 全量核对：块状态 + ltp_mapping + 物理/映射统计逐项
fn check_full(s: &Ssd, sh: &Shadow, ctx: &str) {
  let all: Vec<usize> = (0..ZONES).collect();
  check_blocks(s, sh, &all, ctx);
  check_ltp(s, sh, ctx);
  assert_eq!(
    s.mapping_updated_cnt(),
    sh.map_upd,
    "{ctx}: mapping_updated_cnt"
  );
  assert_eq!(
    s.mapping_updated_gc(),
    sh.map_upd_gc,
    "{ctx}: mapping_updated_gc"
  );
  // 本驱动走 GC 直写入口（write_page_without_caching），不计入 logical_writes
  assert_eq!(s.logical_writes(), 0, "{ctx}: 直写入口不计主机写");
}

fn pick_room(sh: &Shadow, start: usize) -> Option<usize> {
  (0..ZONES)
    .map(|k| (start + k) % ZONES)
    .find(|&b| sh.wpos[b] < PPZ as u64)
}

/// 无空位时压缩一个含失效页的块腾出空间（总有效页 ≤ LOGICAL < 物理容量，必有可回收块）
fn make_room(s: &Ssd, sh: &mut Shadow) {
  for b in 0..ZONES {
    if sh.valid_in(b) < sh.wpos[b] {
      s.compact_block(b as u64);
      sh.compact(b as u64);
      return;
    }
  }
  panic!("无失效页可回收：逻辑页数应 < 物理容量");
}

#[test]
fn shadow_differential_100k() {
  let s = sim();
  let mut sh = Shadow::new();
  let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
  let mut head = 0usize;
  let mut ops = 0u64;
  let mut guard = 0u64;

  while ops < OPS {
    guard += 1;
    assert!(guard < OPS * 10, "驱动未收敛（可能死循环）");
    let r = rng.next() % 100;

    if r < 78 {
      // 主机写
      let log = rng.next() % LOGICAL as u64;
      if let Some(b) = pick_room(&sh, head) {
        head = b;
        s.write_page_without_caching(log, b as u64, -1);
        sh.host_write(log, b as u64);
        ops += 1;
      } else {
        make_room(&s, &mut sh);
      }
    } else if r < 90 {
      // compact 一个有内容的块
      let b = (rng.next() % ZONES as u64) as usize;
      if sh.wpos[b] > 0 {
        s.compact_block(b as u64);
        sh.compact(b as u64);
        ops += 1;
      }
    } else {
      // move：源含失效页、目标有空位
      let src = (rng.next() % ZONES as u64) as usize;
      if sh.wpos[src] > 0
        && sh.valid_in(src) < sh.wpos[src]
        && let Some(dst) = pick_room(&sh, (src + 1) % ZONES)
        && dst != src
      {
        let residue = s.move_valid_pages_to(src as u64, dst as u64);
        sh.mov(src as u64, dst as u64);
        let sh_residue = (0..PPZ).any(|x| sh.lba[src][x] != UNUSED);
        assert_eq!(residue, sh_residue, "move 残留判定不一致");
        ops += 1;
      }
    }

    // 逐操作：LTP 映射 + 物理写计数；周期性：块状态 + 映射统计全量
    check_ltp(&s, &sh, "逐操作");
    if ops.is_multiple_of(FULL_EVERY) {
      check_full(&s, &sh, "周期性全量");
    }
  }
  check_full(&s, &sh, "终态");
  assert!(sh.host > 0, "必须产生主机写");
}
