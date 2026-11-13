//! 块级 EDT 编排（上游 `space_manager.cc:1448-1666` 与 `buffer_manager.cc:1251-1340`
//! 的 EDT 编排函数族，统一挂 `EdtTracker`）。
//!
//! L0 `ldb_edt` 只做纯估算；本模块读写块元数据（block_min/max_edt）与打开块上限。
//! 全局 min/max EDT 的 CAS 半区（`buffer_manager.cc:1276-1306` 操作的
//! `min_edt`/`max_edt` 原子量）属缓冲池状态，归 `ldb_buf`；本模块负责块级登记。

use ldb_edt::WriteHistory;
use ldb_type::{BlockId, LogId, WRITE_HISTORY_LENGTH};

use crate::{
  page_io::{PageIo, ZipInfo},
  space_mgr::SpaceManager,
  state::{BlockMetadata, BlockState},
};

/// EDT 追踪器（薄封装；块状态由 SpaceManager 承载）
#[derive(Debug, Default)]
pub struct EdtTracker;

impl EdtTracker {
  /// 计算页 EDT 并更新块级 min/max（上游 `CalcEDT` buffer_manager.cc:1251-1273）：
  /// ① 水位哨兵 —— 组提交未推进时 `cur_ts` 取 1（:1256-1261），保证
  /// `AddWriteGSN`/min-max 登记不因 0 值退化；
  /// ② `AddWriteGSN`（page.h:26-36）：GSN 写入「最大 GSN 槽位的下一环位」；
  /// ③ 返回 `estimate_death_time`（`max_gsn + avg_gap`）。
  ///
  /// 上游 `:1263-1268` 的 `edt_gap/edt_cnt` 统计累加归 `ldb_stat`（本侧无统计
  /// 汇聚点，不做无消费者的空算）；全局 min/max CAS（:1276-1306）归 `ldb_buf`，
  /// 此处保留块级登记半区。
  pub async fn calc_edt(
    &self,
    mgr: &SpaceManager,
    watermark: &ldb_type::GsnWatermark,
    pid: ldb_type::PageId,
    hist: &mut WriteHistory,
    wcnt: &mut u32,
  ) -> LogId {
    let mut cur = watermark.load();
    if cur == 0 {
      cur = 1;
    }
    add_write_gsn(hist, wcnt, cur);
    let edt = ldb_edt::estimate_death_time(hist);
    // 块级 min/max 登记（页未映射时跳过 —— 偏移 0 会误指块 0）
    if mgr.get_pid_comp_size(pid) > 0 {
      let bid = mgr.pid_block(pid);
      self.update_min_max_edt(mgr, bid, edt);
    }
    edt
  }

  /// 更新块 min/max EDT（上游 UpdateMinMaxEDT 的块级半区，space_manager.cc:1507-1518
  /// SetMinMaxEDTPerBlock 的累积式：只在更小/更大时收敛）。
  ///
  /// 不写 `m.edt` —— 上游块 EDT 由 `CalcBlockAvgEDT` 加权链独占写
  /// （space_manager.cc:1501 `metadata.edt = calc_edt`）；若此处覆写为最新页 EDT，
  /// 会击穿加权平均链（不变量 I6）。
  pub fn update_min_max_edt(&self, mgr: &SpaceManager, bid: BlockId, edt: LogId) {
    let mut s = mgr.sspace.write();
    let m = &mut s.block_metadata[bid as usize];
    m.block_min_edt = m.block_min_edt.min(edt);
    m.block_max_edt = m.block_max_edt.max(edt);
  }

  /// 移除块的 min/max 登记（上游 RemoveMinMaxEDT：块回收时清零）
  pub fn remove_min_max_edt(&self, mgr: &SpaceManager, bid: BlockId) {
    let mut s = mgr.sspace.write();
    let m = &mut s.block_metadata[bid as usize];
    m.block_min_edt = LogId::MAX;
    m.block_max_edt = 0;
  }

  /// LSN 间隔纯计算半区：`lsn - flushed 水位`。
  ///
  /// **分歧登记**：上游 `UpdateLSNGap`（buffer_manager.cc:946-970）是
  /// 「累计写数过阈值 → 推进 `next_write_threshold` 并把分组 LSN 间隔发布到
  /// `cur_lsn_gap_per_wgroup`」的阈值状态机，其阈值/发布状态在缓冲池 ——
  /// 归 `ldb_buf`。本函数只保留「LSN 对 flushed 水位的差值」纯计算半区，
  /// 供状态机与测试复用；完整状态机随 `ldb_buf` 接线时补齐。
  pub fn update_lsn_gap(&self, watermark: &ldb_type::GsnWatermark, lsn: LogId) -> LogId {
    lsn.saturating_sub(watermark.load())
  }

  /// 块 EDT 是否过期。
  ///
  /// **分歧登记**：上游 `BlockEDTIsOutdated`（space_manager.cc:1464-1474）用块平均
  /// EDT `metadata.edt <= cur_gsn`（`bid >= block_cnt_` 恒 true）；本实现用块内最大
  /// `block_max_edt < flushed` —— 仅当块内**全部**页 EDT 都低于 flush 水位才算过期，
  /// 是更保守的块级语义，差异登记 NOTES。
  pub fn block_edt_is_outdated(&self, mgr: &SpaceManager, bid: BlockId, flushed: LogId) -> bool {
    mgr.block_meta(bid).block_max_edt < flushed
  }

  /// 块平均 EDT（上游 CalcBlockAvgEDT space_manager.cc:1476-1487 的等价入口）：
  /// `round((prev_edt * total_pid_cnt + new_edt * pid_cnt) / (total_pid_cnt + pid_cnt))`，
  /// 块无历史 → 直接取新值。`SpaceManager::update_block_metadata_after_write`
  /// 每页调用一次（`pid_cnt = 1`）。
  pub fn calc_block_avg_edt(
    &self,
    mgr: &SpaceManager,
    bid: BlockId,
    new_edt: LogId,
    pid_cnt: u32,
  ) -> LogId {
    calc_block_avg_edt_of(&mgr.block_meta(bid), new_edt, pid_cnt)
  }

  /// 每块设置 min/max（上游 SetMinMaxEDTPerBlock space_manager.cc:1507-1518：
  /// 累积式 —— `min` 只在更小时收敛、`max` 只在更大时收敛，区间只增不减）。
  ///
  /// 旧实现是显式赋值（`m.block_min_edt = min`），会在 `min` 大于当前值时把
  /// 下界抬高（收缩覆盖区间），与上游「累积」语义相反。
  pub fn set_min_max_edt_per_block(
    &self,
    mgr: &SpaceManager,
    bid: BlockId,
    min: LogId,
    max: LogId,
  ) {
    let mut s = mgr.sspace.write();
    let m = &mut s.block_metadata[bid as usize];
    if min < m.block_min_edt {
      m.block_min_edt = min;
    }
    if max > m.block_max_edt {
      m.block_max_edt = max;
    }
  }

  /// min/max 置最小（上游 SetMinMaxEDTToMinimum：GC 目标优先级提升）
  pub fn set_min_max_edt_to_minimum(&self, mgr: &SpaceManager, bid: BlockId) {
    self.set_min_max_edt_per_block(mgr, bid, 0, 0);
  }

  /// 全体块最高 EDT（上游 GetHighestEDT space_manager.cc:1666-1698）。
  ///
  /// **分歧登记**：上游函数体的比较条件 `if (bm.edt > lowest_edt)` 初值
  /// `logid_t::MAX` 恒假 —— 循环是死代码，实际恒返回 `global_min_gsn_flushed`
  /// 水位（注释声明「find the minimum」也未实现）。本实现按注释意图取
  /// 非 EMPTY 块 `block_max_edt` 的最大值；上游「无块回落水位」半区以
  /// 返回 0 表达（无数据）。
  pub fn get_highest_edt(&self, mgr: &SpaceManager) -> LogId {
    let s = mgr.sspace.read();
    s.block_metadata
      .iter()
      .filter(|m| m.state != BlockState::Empty)
      .map(|m| m.block_max_edt)
      .max()
      .unwrap_or(0)
  }

  /// 最小 EDT 块（上游 GetMinEDTBlockID space_manager.cc:1448-1462：OPEN 列表内
  /// 块 EDT 最小者；空列表返回哨兵）
  pub fn get_min_edt_block_id(&self, mgr: &SpaceManager) -> BlockId {
    let s = mgr.sspace.read();
    s.block_lists[BlockState::Open.idx()]
      .snapshot()
      .into_iter()
      .min_by_key(|&b| s.block_metadata[b as usize].edt)
      .unwrap_or(BlockId::MAX)
  }

  /// 块 EDT 极差（`block_max_edt - block_min_edt`）。
  ///
  /// **分歧登记**：上游同名 `GetBlockEDTGap`（space_manager.cc:747-792）统计的是
  /// 「EDT 超过 `WRITE_HISTORY_LENGTH` 阈值的块」的分布清单（阈值治理用途），
  /// 非单块极差；单块极差是 Rust 侧 GC 牺牲块排序的实用量，账本
  /// `port_rewrite` 口径内重写。
  pub fn get_block_edt_gap(&self, mgr: &SpaceManager, bid: BlockId) -> LogId {
    let m = mgr.block_meta(bid);
    m.block_max_edt.saturating_sub(m.block_min_edt)
  }

  /// 选择 EDT 最相近的打开块（上游 SelectOpenBlockIDWithClosestEDT 的 EDT 半区；
  /// 完整版带分组亲和，见 SpaceManager 同名方法）
  pub fn select_open_block_id_with_closest_edt(
    &self,
    mgr: &SpaceManager,
    avg_edt: LogId,
  ) -> BlockId {
    mgr.select_open_block_id_with_closest_edt(avg_edt, 0)
  }

  /// 页按 EDT 升序排序（寿命近的排前）。**分歧登记**：上游 `SortPagesByEDT`
  /// （io_interface.cc:1138-1142）按 `death_time` **降序**；本实现升序，与本文件
  /// `group_pages_by_edt` 的升序桶序自洽，装箱分组差异见残留风险段。
  pub fn sort_pages_by_edt(&self, infos: &mut [ZipInfo]) {
    infos.sort_by_key(|z| z.death_time);
  }

  /// 抽取唯一 EDT 集合（上游 ExtractUniqueEDTs）
  pub fn extract_unique_edts(&self, infos: &[ZipInfo]) -> Vec<LogId> {
    let mut v: Vec<LogId> = infos.iter().map(|z| z.death_time).collect();
    v.sort_unstable();
    v.dedup();
    v
  }

  /// 按 EDT 分组（上游 GroupPagesByEDT io_interface.cc:1069-1136 的等分版：
  /// 返回 [start,end) 切点对。分歧登记 —— 上游桶宽按 `(i+1)` 加权递增，
  /// 本实现等宽切分，分组用途（装箱亲和）不变，加权差异见残留风险段）
  pub fn group_pages_by_edt(
    &self,
    infos: &[ZipInfo],
    min_edt: LogId,
    max_edt: LogId,
  ) -> Vec<(u32, u32)> {
    if infos.is_empty() || min_edt >= max_edt {
      return vec![(0, infos.len() as u32)];
    }
    let groups = (infos.len() as u32).clamp(1, 8);
    let step = (max_edt - min_edt) / LogId::from(groups);
    let mut out = Vec::new();
    let mut start = 0u32;
    let mut lo = min_edt;
    while lo < max_edt {
      let hi = (lo + step).min(max_edt);
      let mut end = start;
      while end < infos.len() as u32 && infos[end as usize].death_time <= hi {
        end += 1;
      }
      if end > start {
        out.push((start, end));
        start = end;
      }
      lo = hi;
    }
    if start < infos.len() as u32 {
      out.push((start, infos.len() as u32));
    }
    out
  }

  /// GC 写缓冲最小 EDT（上游 GetMinEDTFromGCWBuffer；N-20 修复：判空）
  pub fn get_min_edt_from_gcw_buffer(&self, io: &PageIo) -> LogId {
    if io.gc_write.zpage_cnt == 0 {
      return LogId::MAX;
    }
    io.gc_write.zpage_info[..io.gc_write.zpage_cnt as usize]
      .iter()
      .map(|z| z.death_time)
      .min()
      .unwrap_or(LogId::MAX)
  }
}

/// 上游 `AddWriteGSN`（page.h:26-36）：GSN 写入最大 GSN 槽位的下一环位，
/// `wcnt` 累加。上游 `wcnt++` 无溢出防护，`wrapping_add` 对齐 u64 语义。
fn add_write_gsn(hist: &mut WriteHistory, wcnt: &mut u32, gsn: LogId) {
  *wcnt = wcnt.wrapping_add(1);
  let mut max_idx = 0;
  for i in 1..WRITE_HISTORY_LENGTH {
    if hist[i] > hist[max_idx] {
      max_idx = i;
    }
  }
  hist[(max_idx + 1) % WRITE_HISTORY_LENGTH] = gsn;
}

/// 上游 CalcBlockAvgEDT（space_manager.cc:1476-1487）：块 EDT 加权滑动平均。
/// 块无历史（`total_pid_cnt == 0`）或本次无页（`pid_cnt == 0`）→ 直接取新值。
pub(crate) fn calc_block_avg_edt_of(m: &BlockMetadata, new_edt: LogId, pid_cnt: u32) -> LogId {
  if m.total_pid_cnt == 0 || pid_cnt == 0 {
    return new_edt;
  }
  // 上游 double + round 语义（u64→f64 丢精度 >2^53 与上游一致）
  let w = (m.edt as f64 * m.total_pid_cnt as f64 + new_edt as f64 * pid_cnt as f64)
    / (m.total_pid_cnt as f64 + pid_cnt as f64);
  w.round() as LogId
}
