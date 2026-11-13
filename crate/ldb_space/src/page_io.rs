//! 异地写缓冲编排（上游 `io_interface.cc` 的 wbuffer 族）。
//!
//! 修复：C-18（`PageInUserWBufferPerGroup` 的 `[idx]`→`[gidx]` 下标错写）、
//! C-19（不可压缩页回退存原文哨兵）、N-20（`GetMinEDTFromGCWBuffer` 空指针）。
//! 压缩统一 mzstd；I/O 落盘由 `ldb_buf` 驱动（本模块只管缓冲编排）。

use gxhash::HashMap;
use ldb_codec::PageCodec;
use ldb_pack::{Bin, BinSlot, MAX_PAGES_PER_BIN, ZipEntry};
use ldb_page::hook::TxnLogHook;
use ldb_type::{BlockId, LogId, PAGE_SIZE, PageId};

use crate::{
  error::{Error, Result},
  space_mgr::SpaceManager,
};

/// 压缩页元信息（上游 ZipInfo）
#[derive(Debug, Clone, Copy, Default)]
pub struct ZipInfo {
  /// 页号
  pub pid: PageId,
  /// 压缩后尺寸（UNCOMPRESSED 哨兵 = 原文直存）
  pub comp_sz: u16,
  /// 页内偏移
  pub start_ptr: u32,
  /// 死亡时间
  pub death_time: LogId,
}

/// 压缩聚合页（上游 ZipPage）
#[derive(Debug)]
pub struct ZipPage {
  /// 4KiB×n 连续缓冲
  pub buffer: Box<[u8]>,
  /// 页元信息表
  pub zpage_info: Vec<ZipInfo>,
  /// 有效页数（N-20 修复：判空后再访问）
  pub zpage_cnt: u32,
  /// 已写字节
  pub cur_size: u64,
}

impl ZipPage {
  /// 构造（容量 = max_pages 个压缩槽 + 缓冲）
  pub fn new(pages: u32) -> Self {
    Self {
      buffer: vec![0u8; pages as usize * PAGE_SIZE].into_boxed_slice(),
      zpage_info: vec![ZipInfo::default(); pages as usize],
      zpage_cnt: 0,
      cur_size: 0,
    }
  }

  /// 追加压缩页（不可压缩回退原文，C-19）
  ///
  /// 返回 false = 缓冲满
  pub fn add_page(&mut self, pid: PageId, raw: &[u8; PAGE_SIZE], death_time: LogId) -> bool {
    if self.zpage_cnt as usize >= self.zpage_info.len() {
      return false;
    }
    let mut dst = [0u8; PAGE_SIZE];
    let comp_sz = match PageCodec::compress_page(raw, &mut dst) {
      Ok(sz) => sz,
      Err(_) => ldb_pack::UNCOMPRESSED,
    };
    let (data, sz) = if comp_sz == ldb_pack::UNCOMPRESSED {
      (&raw[..], PAGE_SIZE as u16)
    } else {
      (&dst[..], comp_sz)
    };
    let start = self.cur_size as usize;
    if start + data.len() > self.buffer.len() {
      return false;
    }
    self.buffer[start..start + data.len()].copy_from_slice(data);
    self.zpage_info[self.zpage_cnt as usize] = ZipInfo {
      pid,
      comp_sz: sz,
      start_ptr: start as u32,
      death_time,
    };
    self.zpage_cnt += 1;
    self.cur_size = (start + data.len()) as u64;
    true
  }

  /// 重置（上游 ResetBuffer；zpage_cnt=0 时元信息表保留 —— N-20 修复不置 nullptr）
  pub fn reset(&mut self, size: u64) {
    self.zpage_cnt = 0;
    self.cur_size = 0;
    let _ = size;
  }

  /// 平均 EDT（空缓冲返回 0，N-20 修复）
  pub fn avg_edt(&self) -> LogId {
    if self.zpage_cnt == 0 {
      return 0;
    }
    self.zpage_info[..self.zpage_cnt as usize]
      .iter()
      .map(|z| z.death_time)
      .sum::<LogId>()
      / self.zpage_cnt as LogId
  }
}

/// 装箱槽位组（上游 Bins）
#[derive(Debug, Clone, Default)]
pub struct Bins {
  /// 各 bin 占用
  pub bin: Vec<Bin>,
}

impl Bins {
  /// 构造 n 个空 bin
  pub fn new(n: u32) -> Self {
    Self {
      bin: vec![Bin::default(); n as usize],
    }
  }
}

/// 写缓冲分组（上游 wbuffer_group_ 条目）
#[derive(Debug)]
pub struct WriteBufferGroup {
  /// 聚合页
  pub zpage: ZipPage,
  /// 装箱槽
  pub bins: Bins,
  /// 已触发写计数
  pub write_cnt: u64,
}

/// 写缓冲编排器（上游 IOInterface 的缓冲半区）
#[derive(Debug)]
pub struct PageIo {
  /// 写分组（partition_cnt 个）
  pub groups: Vec<WriteBufferGroup>,
  /// GC 写聚合页
  pub gc_write: ZipPage,
  /// 读缓存（pid -> ZipInfo 视图）
  pub read_cache: gxhash::HashMap<PageId, ZipInfo>,
  /// 每分组页容量
  pub pages_per_group: u32,
  /// 上一 GC 块
  pub prev_gced_block: BlockId,
  /// GC 写缓冲在文件内的基地址（块起始偏移）
  pub gc_write_base: u64,
}

impl PageIo {
  /// 构造（上游 IOInterface::Construction 的缓冲半区）
  pub fn new(partition_cnt: u32, pages_per_group: u32) -> Self {
    Self {
      groups: (0..partition_cnt)
        .map(|_| WriteBufferGroup {
          zpage: ZipPage::new(pages_per_group),
          bins: Bins::new(pages_per_group),
          write_cnt: 0,
        })
        .collect(),
      gc_write: ZipPage::new(pages_per_group),
      read_cache: HashMap::default(),
      pages_per_group,
      prev_gced_block: BlockId::MAX,
      gc_write_base: 0,
    }
  }

  /// 分组选择（pid 哈希取模，上游 worker_id % partition）
  #[inline]
  pub fn group_of(&self, pid: PageId, partition_cnt: u32) -> u32 {
    (pid % u64::from(partition_cnt.max(1))) as u32
  }

  /// 加页入写缓冲（返回 false = 该组满，调用方须先 flush）
  pub fn add_page_to_write_buffer(
    &mut self,
    pid: PageId,
    raw: &[u8; PAGE_SIZE],
    death_time: LogId,
  ) -> bool {
    let g = self.group_of(pid, self.groups.len() as u32) as usize;
    let grp = &mut self.groups[g];
    let ok = grp.zpage.add_page(pid, raw, death_time);
    if ok {
      grp.write_cnt += 1;
      let idx = grp.zpage.zpage_cnt as usize - 1;
      let info = grp.zpage.zpage_info[idx];
      self.read_cache.insert(pid, info);
    }
    ok
  }

  /// 加页入 GC 写缓冲
  pub fn add_page_to_gc_write_buffer(
    &mut self,
    pid: PageId,
    raw: &[u8; PAGE_SIZE],
    death_time: LogId,
  ) -> bool {
    let ok = self.gc_write.add_page(pid, raw, death_time);
    if ok {
      let idx = self.gc_write.zpage_cnt as usize - 1;
      let info = self.gc_write.zpage_info[idx];
      self.read_cache.insert(pid, info);
    }
    ok
  }

  /// 页是否在本 worker 写缓冲（跨分组查；上游 PageInUserWBuffer）
  pub fn page_in_user_w_buffer(&self, pid: PageId) -> bool {
    self.groups.iter().any(|g| {
      g.zpage.zpage_info[..g.zpage.zpage_cnt as usize]
        .iter()
        .any(|z| z.pid == pid)
    })
  }

  /// 页是否在指定分组写缓冲（C-18 修复：上界取目标分组 gidx 的 zpage_cnt）
  pub fn page_in_user_w_buffer_per_group(&self, pid: PageId, gidx: u32) -> bool {
    let Some(g) = self.groups.get(gidx as usize) else {
      return false;
    };
    g.zpage.zpage_info[..g.zpage.zpage_cnt as usize]
      .iter()
      .any(|z| z.pid == pid)
  }

  /// 页是否在 GC 写缓冲
  pub fn page_in_gcw_buffer(&self, pid: PageId) -> bool {
    self.gc_write.zpage_info[..self.gc_write.zpage_cnt as usize]
      .iter()
      .any(|z| z.pid == pid)
  }

  /// 页是否在读缓存
  pub fn page_in_read_buffer(&self, pid: PageId) -> bool {
    self.read_cache.contains_key(&pid)
  }

  /// 页是否在任一缓冲（w/gcw/read）
  pub fn page_in_w_buffer(&self, pid: PageId) -> bool {
    self.page_in_user_w_buffer(pid) || self.page_in_gcw_buffer(pid)
  }

  /// 单组统一装箱（上游 `UniformBinPacking` 的 `FLAGS_use_binpacking=false` 分支
  /// `CopyCompressedPages`（io_interface.cc:1168-1221）的组内版：顺序填 4KiB 段；
  /// 段满（`MAX_PAGES_PER_BIN`）**或本页将跨 4KiB 边界**时换段 —— 只按页数换段
  /// 会让混合尺寸页跨扇区（todo.md:183 不变量，ldb_pack-round1 CROSS 登记）。
  ///
  /// `Bin.sum` 按 `ldb_pack::Bin` 契约写**槽内已用字节**（`<= PAGE_SIZE`，不是整块
  /// 偏移 —— 旧实现把全局偏移写进 `sum`，同组再走 `bin_packing` 会触发
  /// `bin_remaining` 下溢，ldb_pack-round2 CROSS 实测）；`ZipInfo::start_ptr` 按
  /// `BinSlot` 契约写全局物理偏移 `bin * PAGE_SIZE + off`。布局有变更时经 staging
  /// 一次性搬移压缩字节（上游逐页 memcpy 的等价物）。
  ///
  /// bins 全量重置、不收缩（容量保持 `pages_per_group`）。
  /// 返回占用的 4KiB bin 数；空缓冲返回 0。
  pub fn uniform_bin_packing(&mut self, gidx: u32) -> u32 {
    let Some(g) = self.groups.get_mut(gidx as usize) else {
      return 0;
    };
    let cnt = g.zpage.zpage_cnt as usize;
    for b in g.bins.bin.iter_mut() {
      *b = Bin::default();
    }
    if cnt == 0 {
      g.zpage.cur_size = 0;
      return 0;
    }
    let mut plan: Vec<(usize, u32)> = Vec::with_capacity(cnt);
    let mut bin_idx = 0u32;
    let mut bin_sum = 0u32;
    let mut pages_in_bin = 0u32;
    for (i, z) in g.zpage.zpage_info[..cnt].iter().enumerate() {
      let sz = u32::from(z.comp_sz);
      if pages_in_bin == MAX_PAGES_PER_BIN || bin_sum + sz > PAGE_SIZE as u32 {
        bin_idx += 1;
        bin_sum = 0;
        pages_in_bin = 0;
      }
      g.bins.bin[bin_idx as usize] = Bin {
        sum: bin_sum + sz,
        count: pages_in_bin + 1,
      };
      plan.push((i, bin_idx * PAGE_SIZE as u32 + bin_sum));
      bin_sum += sz;
      pages_in_bin += 1;
    }
    relayout(&mut g.zpage, &plan);
    g.zpage.cur_size = u64::from(bin_idx + 1) * PAGE_SIZE as u64;
    bin_idx + 1
  }

  /// 按 EDT 分组装箱（上游 BinPackGroups 的组内版：ldb_pack::bin_packing_per_block 复用）。
  ///
  /// 调用方契约（`ldb_pack::bin_packing_per_block` 文档）：`bins` 从零基状态开始、
  /// 预置足够空 bin、`Bin.sum <= PAGE_SIZE`、`count <= MAX_PAGES_PER_BIN`。
  pub fn bin_pack_groups(
    &mut self,
    gidx: u32,
    entries: &mut [ZipEntry],
    bins: &mut [Bin],
  ) -> Vec<Option<BinSlot>> {
    let _ = gidx;
    ldb_pack::bin_packing_per_block(entries, bins)
  }

  /// 块级装箱编排（上游 BinPacking io_interface.cc:1145 的组内版）：
  /// 非 EDT 路径 / 单一 EDT → 统一装箱；多 EDT → `bin_packing_per_block` best-fit，
  /// 且**按 plan 搬移压缩字节**（上游 BinPackingPerBlock 逐页 memcpy，
  /// io_interface.cc:1276-1277/1294-1295 —— 旧实现只改 start_ptr 不搬数据，
  /// flush 后读到旧布局内容，ldb_pack-round1 CROSS 登记的搬移缺口）。
  ///
  /// `plan[i] == None`（bin 不足）在上游是新开 bin（`bincount++`）；本实现预置
  /// `pages_per_group` 个零基 bin 且 `zpage_cnt <= pages_per_group`，None 不可达，
  /// 可达即调用方违约 —— 按 `Ensure` 等价 panic，禁止静默跳过（会保留旧
  /// start_ptr 使映射指向旧布局，ldb_pack-round2 CROSS）。
  ///
  /// 返回占用页数（上游返回 total_cnt 的组内份额）。
  pub fn bin_packing(&mut self, gidx: u32, use_edt: bool, _min_edt: LogId, _max_edt: LogId) -> u32 {
    let cnt = self
      .groups
      .get(gidx as usize)
      .map_or(0, |g| g.zpage.zpage_cnt as usize);
    if cnt == 0 {
      return 0;
    }
    if !use_edt {
      return self.uniform_bin_packing(gidx);
    }
    let single_edt = {
      let g = &self.groups[gidx as usize];
      let unique: GxSet = g.zpage.zpage_info[..cnt]
        .iter()
        .map(|z| z.death_time)
        .collect();
      unique.len() <= 1
    };
    if single_edt {
      return self.uniform_bin_packing(gidx);
    }
    // 多 EDT：零基 bins（禁止 clone 组内旧状态 —— 旧状态可能含其它块的累计值）
    let (entries, bin_cap) = {
      let g = &self.groups[gidx as usize];
      let entries: Vec<ZipEntry> = g.zpage.zpage_info[..cnt]
        .iter()
        .map(|z| ZipEntry {
          comp_sz: z.comp_sz,
          death_time: z.death_time,
          src_off: z.start_ptr,
        })
        .collect();
      (entries, g.bins.bin.len())
    };
    let mut bins = vec![Bin::default(); bin_cap];
    let plan = ldb_pack::bin_packing_per_block(&entries, &mut bins);
    let mut moves: Vec<(usize, u32)> = Vec::with_capacity(plan.len());
    let mut used_bins = 0u32;
    for (i, slot) in plan.iter().enumerate() {
      match slot {
        Some(s) => {
          used_bins = used_bins.max(s.bin + 1);
          moves.push((i, s.bin * PAGE_SIZE as u32 + s.off));
        }
        None => panic!(
          "bin_packing: entries[{i}] 无 bin 可放（预置 bins 不足 = 调用方违约，禁止静默跳过）"
        ),
      }
    }
    let g = &mut self.groups[gidx as usize];
    g.bins.bin.copy_from_slice(&bins);
    relayout(&mut g.zpage, &moves);
    g.zpage.cur_size = u64::from(used_bins) * PAGE_SIZE as u64;
    cnt as u32
  }

  /// 刷出写缓冲（返回待写数据与元信息；落盘由 ldb_buf 执行）。
  /// bins 同步回零基（ldb_pack Bin 前置契约：新块不得混入旧布局累计值）。
  pub fn flush_write_buffer(&mut self, gidx: u32) -> Option<(Vec<u8>, Vec<ZipInfo>)> {
    let g = self.groups.get_mut(gidx as usize)?;
    if g.zpage.zpage_cnt == 0 {
      return None;
    }
    let aligned = g.zpage.cur_size.div_ceil(PAGE_SIZE as u64) * PAGE_SIZE as u64;
    let data = g.zpage.buffer[..aligned as usize].to_vec();
    let infos = g.zpage.zpage_info[..g.zpage.zpage_cnt as usize].to_vec();
    g.zpage.reset(0);
    for b in g.bins.bin.iter_mut() {
      *b = Bin::default();
    }
    Some((data, infos))
  }

  /// 分组刷出（多组批量）
  pub fn flush_write_buffer_per_group(&mut self) -> Vec<(u32, Vec<u8>, Vec<ZipInfo>)> {
    let mut out = Vec::new();
    for gidx in 0..self.groups.len() as u32 {
      if let Some((data, infos)) = self.flush_write_buffer(gidx) {
        out.push((gidx, data, infos));
      }
    }
    out
  }

  /// 刷出 GC 写缓冲
  pub fn flush_gc_write_buffer(&mut self) -> Option<(Vec<u8>, Vec<ZipInfo>)> {
    if self.gc_write.zpage_cnt == 0 {
      return None;
    }
    let aligned = self.gc_write.cur_size.div_ceil(PAGE_SIZE as u64) * PAGE_SIZE as u64;
    let data = self.gc_write.buffer[..aligned as usize].to_vec();
    let infos = self.gc_write.zpage_info[..self.gc_write.zpage_cnt as usize].to_vec();
    self.gc_write.reset(0);
    Some((data, infos))
  }

  /// 重置用户写缓冲（上游 ResetUserWriteBuffer）；bins 同步回零基
  pub fn reset_user_write_buffer(&mut self) {
    for g in &mut self.groups {
      g.zpage.reset(0);
      for b in g.bins.bin.iter_mut() {
        *b = Bin::default();
      }
    }
    self.read_cache.clear();
  }

  /// 重置 GC 写缓冲
  pub fn reset_gc_write_buffer(&mut self) {
    self.gc_write.reset(0);
  }

  /// 重置读缓存
  pub fn reset_read_buffer(&mut self) {
    self.read_cache.clear();
  }

  /// 重置指定分组；bins 同步回零基
  pub fn reset_w_buffer_per_group(&mut self, gidx: u32) {
    if let Some(g) = self.groups.get_mut(gidx as usize) {
      g.zpage.reset(0);
      for b in g.bins.bin.iter_mut() {
        *b = Bin::default();
      }
    }
  }

  // ---- 访问器（账本 get_wbuffer_* 族）----

  /// 分组页容量
  #[inline]
  pub fn get_wbuffer_pid_cnt_per_group(&self, gidx: u32) -> u32 {
    self
      .groups
      .get(gidx as usize)
      .map_or(0, |g| g.zpage.zpage_cnt)
  }

  /// 分组 PID 列表
  pub fn get_wbuffer_pid_per_group(&self, gidx: u32) -> Vec<PageId> {
    self
      .groups
      .get(gidx as usize)
      .map(|g| {
        g.zpage.zpage_info[..g.zpage.zpage_cnt as usize]
          .iter()
          .map(|z| z.pid)
          .collect()
      })
      .unwrap_or_default()
  }

  /// 分组字节
  #[inline]
  pub fn get_wbuffer_size_per_group(&self, gidx: u32) -> u64 {
    self
      .groups
      .get(gidx as usize)
      .map_or(0, |g| g.zpage.cur_size)
  }

  /// 分组写计数
  #[inline]
  pub fn get_wbuffer_write_cnt(&self, gidx: u32) -> u64 {
    self.groups.get(gidx as usize).map_or(0, |g| g.write_cnt)
  }

  /// 对齐后写尺寸（上游 GetWbufferWriteSize）
  pub fn get_wbuffer_write_size(&self, gidx: u32) -> u64 {
    self
      .get_wbuffer_size_per_group(gidx)
      .div_ceil(PAGE_SIZE as u64)
      * PAGE_SIZE as u64
  }

  // ---- 读路径 ----
  //
  // 读接口一律 `Result`：坏映射（start_ptr 越界）与坏压缩页（解压失败）返回
  // Err 交上层处置（重试/重建），禁止静默读旧数据或越界 panic
  // （ldb_codec R1/R2 登记的 `let _ = decompress` 债务收口）。

  /// 原位读（GC 写缓冲内偏移直拷）
  pub fn read_page_in_place(&self, buf: &mut [u8], off: u64) -> Result<()> {
    let start = off as usize;
    let Some(end) = start.checked_add(buf.len()) else {
      return Err(Error::OffsetOutOfBounds {
        start_ptr: off,
        need: buf.len(),
        cap: self.gc_write.buffer.len(),
      });
    };
    if end > self.gc_write.buffer.len() {
      return Err(Error::OffsetOutOfBounds {
        start_ptr: off,
        need: buf.len(),
        cap: self.gc_write.buffer.len(),
      });
    }
    buf.copy_from_slice(&self.gc_write.buffer[start..end]);
    Ok(())
  }

  /// 异地读（按 ZipInfo 解压；C-19 哨兵支持原文直存）
  pub fn read_page_out_of_place(&self, info: &ZipInfo, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
    read_from_zpage(&self.gc_write, info, dst)
  }

  /// 压缩读处理（上游 HandleCompressedRead）
  pub fn handle_compressed_read(&self, info: &ZipInfo, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
    self.read_page_out_of_place(info, dst)
  }

  /// 装箱压缩读（上游 HandleCompressedBinpackedRead：bin 内偏移解压）
  pub fn handle_compressed_binpacked_read(
    &self,
    info: &ZipInfo,
    dst: &mut [u8; PAGE_SIZE],
  ) -> Result<()> {
    self.handle_compressed_read(info, dst)
  }

  /// 未压缩读
  pub fn handle_uncompressed_read(&self, info: &ZipInfo, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
    copy_raw(&self.gc_write.buffer, info.start_ptr, dst)
  }

  /// 从写缓冲读页（命中返回 true；上游 ReadPageFromWBuffer io_interface.cc:351-393：
  /// 先扫用户分组缓冲 `wbuffer_`，再扫 GC 写缓冲 `wbuffer_gc_`，各自用自家 buffer。
  /// 旧实现先查 read_cache 并按 GC 缓冲切片 —— 用户分组页的 start_ptr 是组内偏移，
  /// 会从错误的缓冲读出静默错页，本轮修复为上游的线性扫描序）。
  pub fn read_page_from_w_buffer(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<bool> {
    for g in &self.groups {
      let hit = g.zpage.zpage_info[..g.zpage.zpage_cnt as usize]
        .iter()
        .find(|z| z.pid == pid);
      if let Some(z) = hit {
        read_from_zpage(&g.zpage, z, dst)?;
        return Ok(true);
      }
    }
    let hit = self.gc_write.zpage_info[..self.gc_write.zpage_cnt as usize]
      .iter()
      .find(|z| z.pid == pid);
    if let Some(z) = hit {
      read_from_zpage(&self.gc_write, z, dst)?;
      return Ok(true);
    }
    Ok(false)
  }

  /// 异地批量读（返回逐页命中；单页 Err 不吞——坏页必须浮出到调用方）
  pub fn read_pages_out_of_place(
    &self,
    pids: &[PageId],
    bufs: &mut [[u8; PAGE_SIZE]],
  ) -> Vec<Result<bool>> {
    pids
      .iter()
      .zip(bufs.iter_mut())
      .map(|(pid, b)| self.read_page_from_w_buffer(*pid, b))
      .collect()
  }

  /// 更新写缓存信息（上游 UpdateCachedPageInfo）
  pub fn update_cached_page_info(&mut self, pid: PageId, info: ZipInfo) {
    self.read_cache.insert(pid, info);
  }

  /// 更新读缓存信息（上游 UpdateReadCachedPageInfo）
  pub fn update_read_cached_page_info(&mut self, pid: PageId, info: ZipInfo) {
    self.read_cache.entry(pid).or_insert(info);
  }

  /// 统一读页入口（上游 IOInterface::ReadPage：写缓冲命中优先，未命中由
  /// ldb_buf 走磁盘路径；Err = 命中页损坏，上层可重试或走盘重建）
  pub fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<bool> {
    self.read_page_from_w_buffer(pid, dst)
  }

  /// 缓冲重置（上游 ResetBuffer 分发：kind = 0 用户写 / 1 GC 写 / 2 读缓存）
  pub fn reset_buffer(&mut self, kind: u8) {
    match kind {
      0 => self.reset_user_write_buffer(),
      1 => self.reset_gc_write_buffer(),
      _ => self.reset_read_buffer(),
    }
  }

  /// 空间元数据同步（上游 UpdateSpaceMetadata io_interface.cc:1340-1372 的组内版）：
  /// 逐页更新 PID→偏移 **并**登记反向表 `blocks[bid].pages[slot]`（旧实现漏掉
  /// 反向表半区，GC 有效页统计将恒为空 —— 上游 UpdateSpaceMetadata 同时调
  /// UpdatePID2Offset 与 UpdateOffset2PIDs）。
  ///
  /// 越块断言对应上游 `Ensure(GetWbufferWriteSize + (disk_base -
  /// GetBlockOffset(bid)) <= max_w_ptr_)`（io_interface.cc:1341-1342）。
  pub async fn update_space_metadata<T: TxnLogHook>(
    &self,
    bid: BlockId,
    infos: &[ZipInfo],
    mgr: &SpaceManager<T>,
  ) {
    let block_base = mgr.get_block_offset(bid);
    let block_cap = mgr.get_block_size();
    for z in infos {
      let off = self.gc_write_base + u64::from(z.start_ptr);
      assert!(
        off >= block_base && off - block_base < block_cap,
        "GC 写映射越块: off={off} block_base={block_base} cap={block_cap}"
      );
      let slot = ((off - block_base) / PAGE_SIZE as u64) as u32;
      mgr.update_pid2_offset(z.pid, off, z.comp_sz).await;
      mgr.update_offset2_pids(bid, slot, z.pid);
    }
  }

  /// GC 写缓冲基地址（块内偏移）
  #[inline]
  pub fn gc_write_base(&self) -> u64 {
    self.gc_write_base
  }
}

type GxSet = gxhash::HashSet<LogId>;

/// 从聚合页按 `ZipInfo` 读出一页：原文哨兵直拷，压缩页解压。
/// 越界 → Err（OffsetOutOfBounds）；坏压缩帧 → Err（Codec，上层可重试）。
fn read_from_zpage(zp: &ZipPage, z: &ZipInfo, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
  if z.comp_sz == ldb_pack::UNCOMPRESSED {
    return copy_raw(&zp.buffer, z.start_ptr, dst);
  }
  let s = z.start_ptr as usize;
  let e = s + usize::from(z.comp_sz);
  if e > zp.buffer.len() {
    return Err(Error::OffsetOutOfBounds {
      start_ptr: u64::from(z.start_ptr),
      need: usize::from(z.comp_sz),
      cap: zp.buffer.len(),
    });
  }
  PageCodec::decompress_page(&zp.buffer[s..e], z.comp_sz, dst)?;
  Ok(())
}

/// 原文整页拷贝（越界 → Err）
fn copy_raw(buf: &[u8], start_ptr: u32, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
  let s = start_ptr as usize;
  let e = s + PAGE_SIZE;
  if e > buf.len() {
    return Err(Error::OffsetOutOfBounds {
      start_ptr: u64::from(start_ptr),
      need: PAGE_SIZE,
      cap: buf.len(),
    });
  }
  dst.copy_from_slice(&buf[s..e]);
  Ok(())
}

/// 按 `(页下标, 新 start_ptr)` 计划把压缩数据搬移到新布局（重叠安全：先 staging
/// 旧缓冲再回写），并回写 `start_ptr`（上游 BinPackingPerBlock/CopyCompressedPages
/// 逐页 memcpy 的等价物；无布局变更时零拷贝跳过）。
fn relayout(zpage: &mut ZipPage, plan: &[(usize, u32)]) {
  let differs = plan
    .iter()
    .any(|&(i, d)| d as usize != zpage.zpage_info[i].start_ptr as usize);
  if differs {
    let staged = zpage.buffer[..zpage.cur_size as usize].to_vec();
    for &(i, dst) in plan {
      let z = &zpage.zpage_info[i];
      let s = z.start_ptr as usize;
      let n = usize::from(z.comp_sz);
      zpage.buffer[dst as usize..dst as usize + n].copy_from_slice(&staged[s..s + n]);
    }
  }
  for &(i, dst) in plan {
    zpage.zpage_info[i].start_ptr = dst;
  }
}

impl PageIo {
  /// 构造扩展（带 GC 写基地址）
  pub fn with_base(partition_cnt: u32, pages_per_group: u32, gc_base: u64) -> Self {
    let mut io = Self::new(partition_cnt, pages_per_group);
    io.gc_write_base = gc_base;
    io
  }
}
