//! 纯软件 SSD 模拟器（上游 `ZLeanStore/src/include/storage/space/sim/SSD.hpp`
//! 自 ssd-iq 移植；已从 device.{h,cc} 物理拆出 —— 宿主文件的 libnvme/liburing
//! 传染被切断）。
//!
//! 语义：区块 PTL 表（phys→log）、LTP 映射（log→phys）、物理写计数（WAF 度量）、
//! naive 磨损均衡（LUN 空闲池 + 轮转）、GC 压缩（compact / move / until-free）。

use std::{
  cell::{Cell, RefCell},
  collections::VecDeque,
  fmt,
};

/// 未使用哨兵
pub const UNUSED: u64 = u64::MAX;

/// 物理块
#[derive(Debug, Clone)]
pub struct Block {
  /// phys→log 映射表
  ptl_vec: Vec<u64>,
  /// 有效页数
  valid_cnt: u64,
  /// 写指针
  write_pos: u64,
  /// 擦除计数
  pub erase_count: u64,
  /// 每块页数
  pub pages_per_zone: u64,
  /// 块号
  pub block_id: u64,
  /// GC 代龄（-1 = 未经历 GC）
  pub gc_age: i64,
  pub gc_generation: i64,
  /// 写组
  pub group: i64,
  /// 是否被 GC 写过
  pub written_by_gc: bool,
}

impl Block {
  fn new(pages_per_block: u64, block_id: u64) -> Self {
    Self {
      ptl_vec: vec![UNUSED; pages_per_block as usize],
      valid_cnt: 0,
      write_pos: 0,
      erase_count: 0,
      pages_per_zone: pages_per_block,
      block_id,
      gc_age: -1,
      gc_generation: 0,
      group: -1,
      written_by_gc: false,
    }
  }

  /// PTL 视图
  pub fn ptl(&self) -> &[u64] {
    &self.ptl_vec
  }

  /// 有效页数
  pub fn valid_cnt(&self) -> u64 {
    self.valid_cnt
  }

  /// 写指针
  pub fn write_pos(&self) -> u64 {
    self.write_pos
  }

  /// 写一页（返回物理槽位）
  pub fn write(&mut self, log_page_id: u64) -> u64 {
    assert!(self.can_write(), "块已满不可写");
    assert_eq!(self.ptl_vec[self.write_pos as usize], UNUSED, "槽位须空闲");
    self.ptl_vec[self.write_pos as usize] = log_page_id;
    self.valid_cnt += 1;
    let pos = self.write_pos;
    self.write_pos += 1;
    pos
  }

  /// 置槽无效
  pub fn set_unused(&mut self, pos: u64) {
    assert_ne!(self.ptl_vec[pos as usize], UNUSED);
    self.ptl_vec[pos as usize] = UNUSED;
    self.valid_cnt -= 1;
  }

  /// 写满
  pub fn fully_written(&self) -> bool {
    self.write_pos == self.pages_per_zone
  }

  /// 可写
  pub fn can_write(&self) -> bool {
    self.write_pos < self.pages_per_zone
  }

  /// 全有效
  pub fn all_valid(&self) -> bool {
    self.valid_cnt == self.pages_per_zone
  }

  /// 全无效
  pub fn all_invalid(&self) -> bool {
    self.valid_cnt == 0
  }

  /// 已擦除
  pub fn is_erased(&self) -> bool {
    self.write_pos == 0
  }

  /// 紧凑化（不更新 SSD 级 LTP —— 上游 CompactNoMappingUpdate）
  pub fn compact_no_mapping_update(&mut self, erase_age: &mut u64) {
    self.write_pos = 0;
    for p in 0..self.pages_per_zone as usize {
      let lp = self.ptl_vec[p];
      if lp != UNUSED {
        self.ptl_vec[self.write_pos as usize] = lp;
        self.write_pos += 1;
      }
    }
    self.valid_cnt = self.write_pos;
    for p in self.ptl_vec.iter_mut().skip(self.write_pos as usize) {
      *p = UNUSED;
    }
    self.erase_count += 1;
    self.gc_age = *erase_age as i64;
    *erase_age += 1;
    self.written_by_gc = true;
  }

  /// 擦除
  pub fn erase(&mut self, erase_age: &mut u64) {
    for p in self.ptl_vec.iter_mut() {
      *p = UNUSED;
    }
    self.write_pos = 0;
    self.erase_count += 1;
    self.gc_age = *erase_age as i64;
    *erase_age += 1;
    self.written_by_gc = false;
    self.valid_cnt = 0;
    self.group = -1;
  }

  /// 打印（统计行）
  pub fn print(&self) -> String {
    format!(
      "age:{} gcGen:{} wbgc:{} vc:{}",
      self.gc_age, self.gc_generation, self.written_by_gc, self.valid_cnt
    )
  }
}

/// SSD 模拟器（recursive_mutex → ReentrantLock 语义：内部互调）
pub struct Ssd {
  /// 全局锁（上游 recursive_mutex；Rust 侧 RefCell 化 —— 测试单线程，断言可重入）
  inner: RefCell<SsdInner>,
  /// 擦除代龄计数器（上游 Block::_eraseAgeCounter 静态位下沉实例）
  erase_age: Cell<u64>,
}

/// 内部状态（锁保护体）
struct SsdInner {
  blocks: Vec<Block>,
  /// log→phys 映射
  ltp_mapping: Vec<u64>,
  mapping_updated_cnt: Vec<u64>,
  mapping_updated_gc: Vec<u64>,
  phys_writes: u64,
  /// 主机逻辑写数（仅统计主机入口，不含 GC/WL 内部搬移）—— WAF 分母。
  /// 上游 `sim.cpp:81` WAF = `physWrites()/writesPerRep`，`writesPerRep` 由驱动侧的
  /// 主机 writePage 调用计数；`mapping_updated_cnt` 因含 GC 搬移不能作分母。
  logical_writes: u64,
  /// 填充率 [0,1]
  pub ssd_fill: f64,
  pub capacity: u64,
  pub zone_size: u64,
  pub page_size: u64,
  pub zones: u64,
  pub pages_per_zone: u64,
  pub logical_pages: u64,
  pub physical_pages: u64,
  gced_normal_block: u64,
  gced_cold_block: u64,
  // 磨损均衡
  wear_leveling_enabled: bool,
  wl_lun_cnt: u64,
  wl_threshold_t: u64,
  wl_free_pools: Vec<VecDeque<u64>>,
  wl_in_free_pool: Vec<bool>,
  wl_current_block: Vec<u64>,
  wl_update_counter: Vec<u64>,
}

impl fmt::Debug for Ssd {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let i = self.inner.borrow();
    f.debug_struct("Ssd")
      .field("zones", &i.zones)
      .field("logical_pages", &i.logical_pages)
      .field("phys_writes", &i.phys_writes)
      .finish()
  }
}

impl Ssd {
  /// 账本命名构造（上游 InitSSDSimulator → new 委托）
  pub fn init_ssd_simulator(capacity: u64, zone_size: u64, page_size: u64, ssd_fill: f64) -> Self {
    Self::new(capacity, zone_size, page_size, ssd_fill)
  }
}

// 安全两段式构造
impl Ssd {
  /// 构造（安全两段式：先建 inner 再入壳）
  pub fn new(capacity: u64, zone_size: u64, page_size: u64, ssd_fill: f64) -> Self {
    let s = Self {
      inner: RefCell::new(SsdInner {
        blocks: Vec::new(),
        ltp_mapping: Vec::new(),
        mapping_updated_cnt: Vec::new(),
        mapping_updated_gc: Vec::new(),
        phys_writes: 0,
        logical_writes: 0,
        ssd_fill,
        capacity,
        zone_size,
        page_size,
        zones: capacity / zone_size,
        pages_per_zone: zone_size / page_size,
        logical_pages: ((capacity / page_size) as f64 * ssd_fill) as u64,
        physical_pages: 0,
        gced_normal_block: 0,
        gced_cold_block: 0,
        wear_leveling_enabled: false,
        wl_lun_cnt: 1,
        wl_threshold_t: 1024,
        wl_free_pools: Vec::new(),
        wl_in_free_pool: Vec::new(),
        wl_current_block: Vec::new(),
        wl_update_counter: Vec::new(),
      }),
      erase_age: Cell::new(0),
    };
    let mut inner = s.inner.borrow_mut();
    let zones = inner.zones;
    let ppz = inner.pages_per_zone;
    let logical = inner.logical_pages as usize;
    let wl_cnt = 64u64.min(zones).max(1);
    inner.blocks = (0..zones).map(|z| Block::new(ppz, z)).collect();
    inner.ltp_mapping = vec![UNUSED; logical];
    inner.mapping_updated_cnt = vec![0; logical];
    inner.mapping_updated_gc = vec![0; logical];
    inner.physical_pages = zones * ppz;
    inner.wl_lun_cnt = wl_cnt;
    inner.wl_free_pools = (0..wl_cnt).map(|_| Default::default()).collect();
    inner.wl_in_free_pool = vec![false; zones as usize];
    inner.wl_current_block = vec![UNUSED; wl_cnt as usize];
    inner.wl_update_counter = vec![0; wl_cnt as usize];
    // 初始全部块即擦除态（Block::new 已是 write_pos==0，无需 erase()——否则
    // erase_count/gc_age 被虚增，背离上游“新建块 gcAge==-1”），并入空闲池（上游构造尾
    // `for b in 0..zones: wlPushFreeBlock(b)`；此前漏做导致 WL 兜底扫描跨 LUN 选块）。
    let mut age = 0u64;
    for b in 0..zones {
      inner.wl_push_free_block(b, &mut age);
    }
    drop(inner);
    s
  }
}

impl SsdInner {
  fn block_to_lun(&self, block_id: u64) -> u64 {
    if self.wl_lun_cnt == 0 {
      0
    } else {
      block_id % self.wl_lun_cnt
    }
  }

  fn log_page_to_lun(&self, log_page: u64) -> u64 {
    if self.wl_lun_cnt == 0 {
      0
    } else {
      log_page % self.wl_lun_cnt
    }
  }

  fn wl_push_free_block(&mut self, block_id: u64, _age: &mut u64) {
    let lun = self.block_to_lun(block_id);
    if !self.wl_in_free_pool[block_id as usize] {
      self.wl_free_pools[lun as usize].push_back(block_id);
      self.wl_in_free_pool[block_id as usize] = true;
    }
  }

  fn wl_pop_free_block(&mut self, lun: u64) -> u64 {
    let q = &mut self.wl_free_pools[lun as usize];
    while let Some(b) = q.pop_front() {
      self.wl_in_free_pool[b as usize] = false;
      if self.blocks[b as usize].is_erased() {
        return b;
      }
    }
    // 兜底扫描：仅扫本 LUN 的块（上游 `for (b=lun; b<size; b+=wlLunCnt)`；
    // 曾误写为步长 1，会跨 LUN 抢块，破坏 WL 的 LUN 亲和）
    let stride = self.wl_lun_cnt.max(1);
    let mut b = lun;
    while b < self.blocks.len() as u64 {
      if self.blocks[b as usize].is_erased() && !self.wl_in_free_pool[b as usize] {
        return b;
      }
      b += stride;
    }
    panic!("WL 空闲块耗尽 (lun={lun})");
  }

  fn wl_rotate(&mut self, lun: u64, age: &mut u64) {
    if self.wl_current_block[lun as usize] == UNUSED {
      let b = self.wl_pop_free_block(lun);
      self.wl_current_block[lun as usize] = b;
      self.wl_update_counter[lun as usize] = 0;
      return;
    }
    let old_id = self.wl_current_block[lun as usize];
    let new_id = self.wl_pop_free_block(lun);
    // 搬有效页到新块
    let lbas: Vec<u64> = self.blocks[old_id as usize]
      .ptl()
      .iter()
      .copied()
      .filter(|&l| l != UNUSED)
      .collect();
    for lba in lbas {
      self.write_page_without_caching(lba, new_id, -1, age);
    }
    // 擦旧回池
    let old = &mut self.blocks[old_id as usize];
    old.erase(age);
    old.gc_generation = 0;
    self.wl_push_free_block(old_id, age);
    self.wl_current_block[lun as usize] = new_id;
    self.wl_update_counter[lun as usize] = 0;
  }

  fn wl_current_write_block_for(&mut self, log_page: u64, age: &mut u64) -> u64 {
    let lun = self.log_page_to_lun(log_page);
    if self.wl_current_block[lun as usize] == UNUSED {
      self.wl_rotate(lun, age);
    }
    if !self.blocks[self.wl_current_block[lun as usize] as usize].can_write() {
      self.wl_rotate(lun, age);
    }
    self.wl_current_block[lun as usize]
  }

  fn write_page_without_caching(
    &mut self,
    log_page: u64,
    block_id: u64,
    group: i64,
    age: &mut u64,
  ) {
    let _ = age;
    if self.blocks[block_id as usize].group == -1 {
      self.blocks[block_id as usize].group = group;
    }
    // 越界逻辑页须「先于块变更」失败 —— 读映射即直接索引（对齐上游
    // `SSD.hpp:364 _ltpMapping.at(logPage)` 先抛 out_of_range；旧实现用 `.get()` 静默跳过后
    // 仍先改块再在写入索引处 panic，留下半污染状态）。
    let addr = self.ltp_mapping[log_page as usize];
    if addr != UNUSED {
      let (z, p) = (addr / self.pages_per_zone, addr % self.pages_per_zone);
      self.blocks[z as usize].set_unused(p);
    }
    let pos = self.blocks[block_id as usize].write(log_page);
    self.ltp_mapping[log_page as usize] = block_id * self.pages_per_zone + pos;
    self.mapping_updated_cnt[log_page as usize] += 1;
    self.phys_writes += 1;
  }

  fn compact_block_by_id(&mut self, block_id: u64, age: &mut u64) {
    self.blocks[block_id as usize].compact_no_mapping_update(age);
    self.blocks[block_id as usize].gc_generation += 1;
    if self.blocks[block_id as usize].written_by_gc {
      self.gced_cold_block += 1;
    } else {
      self.gced_normal_block += 1;
    }
    self.blocks[block_id as usize].written_by_gc = true;
    let ppz = self.pages_per_zone;
    for p in 0..self.blocks[block_id as usize].write_pos() {
      let lp = self.blocks[block_id as usize].ptl()[p as usize];
      self.ltp_mapping[lp as usize] = block_id * ppz + p;
      self.mapping_updated_gc[lp as usize] += 1;
      self.phys_writes += 1;
    }
  }

  /// 搬有效页（源→目标）：累加 GC 冷/热块计数、置目标 written_by_gc，返回源是否仍有残留。
  /// 即上游 `moveValidPagesTo(sourceId, destinationId)`；`compact_until_free_block` 亦复用此助手
  /// （旧实现内联搬迁时漏掉这两处统计，gcedNormal/Cold 与上游发散）。
  fn move_valid_pages_inner(&mut self, source_id: u64, destination_id: u64, age: &mut u64) -> bool {
    if self.blocks[source_id as usize].written_by_gc {
      self.gced_cold_block += 1;
    } else {
      self.gced_normal_block += 1;
    }
    self.blocks[destination_id as usize].written_by_gc = true;
    let ppz = self.pages_per_zone;
    for p in 0..ppz {
      if !self.blocks[destination_id as usize].can_write() {
        break;
      }
      let lba = self.blocks[source_id as usize].ptl()[p as usize];
      if lba != UNUSED {
        self.write_page_without_caching(lba, destination_id, -1, age);
      }
    }
    !self.blocks[source_id as usize].all_invalid()
  }
}

// 公开 API（锁内委托）
impl Ssd {
  /// 块表视图
  pub fn blocks(&self) -> Vec<Block> {
    self.inner.borrow().blocks.clone()
  }

  /// LTP 映射视图
  pub fn ltp_mapping(&self) -> Vec<u64> {
    self.inner.borrow().ltp_mapping.clone()
  }

  /// 映射更新计数
  pub fn mapping_updated_cnt(&self) -> Vec<u64> {
    self.inner.borrow().mapping_updated_cnt.clone()
  }

  /// GC 映射更新计数
  pub fn mapping_updated_gc(&self) -> Vec<u64> {
    self.inner.borrow().mapping_updated_gc.clone()
  }

  /// 物理写数
  pub fn phys_writes(&self) -> u64 {
    self.inner.borrow().phys_writes
  }

  /// 主机逻辑写数（WAF 分母；不含 GC/WL 内部搬移）
  pub fn logical_writes(&self) -> u64 {
    self.inner.borrow().logical_writes
  }

  /// SSD WAF = 物理写数 / 主机逻辑写数（上游 `sim.cpp:81` 口径）。
  /// 注意：分母是主机写，不是 `mapping_updated_cnt` 之和——后者含 GC/WL 搬移写，
  /// 会让 WAF 在 GC 场景被严重低估（旧实现如此）。
  pub fn get_ssdwa(&self) -> f64 {
    let i = self.inner.borrow();
    if i.logical_writes == 0 {
      0.0
    } else {
      i.phys_writes as f64 / i.logical_writes as f64
    }
  }

  /// WAF 直设（NoWA 最优基线注入；上游 HackForOptimalWASetPhysWrites）
  pub fn hack_for_optimal_wa_set_phys_writes(&self, phy_writes: u64) {
    self.inner.borrow_mut().phys_writes = phy_writes;
  }

  /// 物理地址 → 块
  pub fn get_zone(&self, phys_addr: u64) -> u64 {
    phys_addr / self.inner.borrow().pages_per_zone
  }

  /// 物理地址 → 页
  pub fn get_page(&self, phys_addr: u64) -> u64 {
    phys_addr % self.inner.borrow().pages_per_zone
  }

  /// (块,页) → 物理地址
  pub fn get_addr(&self, zone: u64, pos: u64) -> u64 {
    zone * self.inner.borrow().pages_per_zone + pos
  }

  /// 块 → LUN
  pub fn block_to_lun(&self, block_id: u64) -> u64 {
    self.inner.borrow().block_to_lun(block_id)
  }

  /// 逻辑页 → LUN
  pub fn log_page_to_lun(&self, log_page: u64) -> u64 {
    self.inner.borrow().log_page_to_lun(log_page)
  }

  /// 磨损均衡开关
  pub fn set_wear_leveling(&self, enabled: bool) {
    self.inner.borrow_mut().wear_leveling_enabled = enabled;
  }

  /// 磨损均衡阈值
  pub fn set_wear_leveling_threshold(&self, t: u64) {
    self.inner.borrow_mut().wl_threshold_t = t.max(1);
  }

  /// LUN 数
  pub fn wear_leveling_lun_cnt(&self) -> u64 {
    self.inner.borrow().wl_lun_cnt
  }

  /// 主写入口（无 WL 直写块 0）
  pub fn write_page(&self, log_page: u64, group: i64) {
    let mut age = self.erase_age.get();
    let mut i = self.inner.borrow_mut();
    i.logical_writes += 1;
    i.write_page_without_caching(log_page, 0, group, &mut age);
    drop(i);
    self.erase_age.set(age);
  }

  /// WL 主写入口（LUN 亲和 + 阈值轮转）
  pub fn write_page_wl(&self, log_page: u64, group: i64) {
    let mut age = self.erase_age.get();
    let mut i = self.inner.borrow_mut();
    i.logical_writes += 1;
    if !i.wear_leveling_enabled {
      i.write_page_without_caching(log_page, 0, group, &mut age);
      drop(i);
      self.erase_age.set(age);
      return;
    }
    let lun = i.log_page_to_lun(log_page);
    i.wl_update_counter[lun as usize] += 1;
    if i.wl_update_counter[lun as usize] >= i.wl_threshold_t {
      i.wl_rotate(lun, &mut age);
    }
    let b = i.wl_current_write_block_for(log_page, &mut age);
    i.write_page_without_caching(log_page, b, group, &mut age);
    drop(i);
    self.erase_age.set(age);
  }

  /// GC 直写（不进写缓冲）
  pub fn write_page_without_caching(&self, log_page: u64, block_id: u64, group: i64) {
    let mut age = self.erase_age.get();
    self
      .inner
      .borrow_mut()
      .write_page_without_caching(log_page, block_id, group, &mut age);
    self.erase_age.set(age);
  }

  /// 上游 WriteToSSD（块 0 直写别名）
  pub fn write_to_ssd(&self, log_page: u64) {
    self.write_page(log_page, -1);
  }

  /// 擦块
  pub fn erase_block(&self, block_id: u64) {
    let mut age = self.erase_age.get();
    let mut i = self.inner.borrow_mut();
    i.blocks[block_id as usize].erase(&mut age);
    if i.wear_leveling_enabled {
      i.wl_push_free_block(block_id, &mut age);
    }
    drop(i);
    self.erase_age.set(age);
  }

  /// 压缩块（页序紧凑 + LTP 回写）
  pub fn compact_block(&self, block_id: u64) {
    let mut age = self.erase_age.get();
    self
      .inner
      .borrow_mut()
      .compact_block_by_id(block_id, &mut age);
    self.erase_age.set(age);
  }

  /// 搬迁有效页（源 → 目标；返回源是否仍有残留）
  pub fn move_valid_pages_to(&self, source_id: u64, destination_id: u64) -> bool {
    let mut age = self.erase_age.get();
    let residue =
      self
        .inner
        .borrow_mut()
        .move_valid_pages_inner(source_id, destination_id, &mut age);
    self.erase_age.set(age);
    residue
  }

  /// 压缩直至空出一个块（上游 CompactUntilFreeBlock 简式：victim 逐个搬入 gc 块）
  pub fn compact_until_free_block(
    &self,
    mut gc_block_id: i64,
    mut next: impl FnMut() -> u64,
  ) -> (u64, i64) {
    let mut age = self.erase_age.get();
    let mut i = self.inner.borrow_mut();
    if gc_block_id == -1 || i.blocks[gc_block_id as usize].all_valid() {
      let b = next();
      i.compact_block_by_id(b, &mut age);
      gc_block_id = b as i64;
    }
    let mut victim = next();
    loop {
      // 契约校验：`next()` 必须给出「区别于 gc 块、可被搬空」的 victim。若返回 gc 块自身，
      // `move_valid_pages_inner(v, v)` 会自我搬移而永不置空 → 非终止死循环（上游同结构；
      // 其另一重载有 `ensure(fullDest != victimId)`）。此处以确定 panic 取代静默死循环。
      assert_ne!(
        victim, gc_block_id as u64,
        "compact_until_free_block: next() 返回 victim == gc 块（违反契约，将自我搬移死循环）"
      );
      // 复用上游 moveValidPagesTo（含 gced*/written_by_gc 统计）
      if !i.move_valid_pages_inner(victim, gc_block_id as u64, &mut age) {
        break;
      }
      i.compact_block_by_id(victim, &mut age);
      gc_block_id = victim as i64;
      victim = next();
    }
    i.blocks[victim as usize].erase(&mut age);
    i.blocks[victim as usize].gc_generation = 0;
    if i.wear_leveling_enabled {
      i.wl_push_free_block(victim, &mut age);
    }
    drop(i);
    self.erase_age.set(age);
    (victim, gc_block_id)
  }

  /// 复位物理写计数（上游 `SSD::resetPhysicalCounters` 仅清零 `_physWrites`；
  /// 映射表与映射更新统计保留。旧实现连 `mapping_updated_*` 也清零，与上游发散）
  pub fn reset_physical_counters(&self) {
    self.inner.borrow_mut().phys_writes = 0;
  }

  /// 统计快照
  pub fn stats(&self) -> (u64, u64, u64) {
    let i = self.inner.borrow();
    (i.phys_writes, i.gced_normal_block, i.gced_cold_block)
  }

  /// 信息打印
  pub fn print_info(&self) -> String {
    let i = self.inner.borrow();
    format!(
      "capacity={} zoneSize={} pageSize={} ssdFill={:.2} zones={} logical={} physical={} physWrites={} wa={:.3}",
      i.capacity,
      i.zone_size,
      i.page_size,
      i.ssd_fill,
      i.zones,
      i.logical_pages,
      i.physical_pages,
      i.phys_writes,
      if i.phys_writes == 0 {
        1.0
      } else {
        i.phys_writes as f64 / i.logical_pages as f64
      }
    )
  }

  /// 块统计打印
  pub fn print_blocks_stats(&self) -> String {
    let i = self.inner.borrow();
    i.blocks
      .iter()
      .map(|b| format!("b{}:{}", b.block_id, b.print()))
      .collect::<Vec<_>>()
      .join(" ")
  }

  /// 统计写 JSON 行文件（上游 WriteStatsFile；格式化到调用方落盘）
  pub fn write_stats_file(&self) -> String {
    let i = self.inner.borrow();
    let mut out = String::from("{\"phys_writes\":");
    out.push_str(&i.phys_writes.to_string());
    out.push_str(",\"gced_normal\":");
    out.push_str(&i.gced_normal_block.to_string());
    out.push_str(",\"gced_cold\":");
    out.push_str(&i.gced_cold_block.to_string());
    out.push('}');
    out
  }
}
