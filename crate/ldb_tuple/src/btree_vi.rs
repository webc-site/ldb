//! BTreeVI：MVCC 元组层（上游 master `BTreeVI.hpp/.cpp` + ConcurrencyControl 协作）。
//!
//! ChainedTuple：就地主版本 + Undo 链；
//! FatTuple：高频更新键的页内紧凑多版本 Delta 数组（自动升降级）；
//! graveyard：删除版本经 BTreeVI 分支迁墓（复用 `ldb_btree` 实例语义）。

use std::{collections::BTreeSet, fmt, mem::take, sync::atomic::AtomicU64};

use gxhash::HashMap;

use crate::error::{Error, Result};

/// `DeltaArray::buf` 的字节上限：偏移表项为 `u16`（上游 `getDeltaOffsets()`
/// 返回 `u16*`，`BTreeVI.hpp:161`），故 delta 缓冲总长不得超过 `u16::MAX`，
/// 否则 [`Delta::offset`]（`u16`）在追加时会被截断。`has_space_for`/`append`
/// 据此把有效容量收敛到 `min(capacity, DELTA_ARRAY_MAX_SIZE)`，超限时按上游语义
/// **拒绝追加**（返回 `false`）——调用方回退链式（`maybe_promote`）或转失败
/// （`convert_chained_to_fat_tuple_different_attributes` 返回 `None`）。
pub const DELTA_ARRAY_MAX_SIZE: usize = u16::MAX as usize;

/// 版本链节点（ChainedTuple 的 undo 记录）
#[derive(Debug, Clone)]
pub struct UndoVersion {
  /// 提交时间戳（MSB 置位语义由 TS 编码承载）
  pub tx_id: u64,
  /// 该版本载荷
  pub payload: Vec<u8>,
  /// 是否删除
  pub removed: bool,
}

/// 链式元组（主版本就地 + undo 链；上游 ChainedTuple）
#[derive(Debug, Clone, Default)]
pub struct ChainedTuple {
  /// 创建事务
  pub creator: u64,
  /// 最新已提交载荷
  pub head: Vec<u8>,
  /// head 的版本事务号
  pub head_tx: u64,
  /// head 是否删除墓碑
  pub removed_head: bool,
  /// undo 链（追加序 **旧→新**，末尾为次新版本；`tx_id` = 该版本的创建事务，
  /// `payload` = 该版本的值）。上游把次级版本放在每 Worker 的 HistoryTree
  /// （`BTreeVI.cpp:351` `insertVersion` / `:900` `retrieveVersion`），本模型内联为
  /// `undo`，语义等价：head 之后按事务序回放，取首个 `tx <= snapshot` 的版本。
  pub undo: Vec<UndoVersion>,
  /// 写锁持有者（0 = 无锁）
  pub write_lock_holder: u64,
}

impl ChainedTuple {
  /// 快照可见值（版本窗口语义：取 tx <= snapshot 的最新版本；
  /// 该版本为删除墓碑 → None，不回退更旧版本）
  pub fn value_at(&self, snapshot: u64) -> Option<&[u8]> {
    if self.head_tx <= snapshot {
      return if self.removed_head {
        None
      } else {
        Some(&self.head)
      };
    }
    // head 窗口未开启（更新过新）→ undo 追加序为旧→新，rev() 得新→旧
    for v in self.undo.iter().rev() {
      if v.tx_id <= snapshot {
        return if v.removed { None } else { Some(&v.payload) };
      }
    }
    None
  }
}

impl ChainedTuple {
  /// 是否终态（无 undo 即最终版本）
  pub fn is_final(&self) -> bool {
    self.undo.is_empty()
  }

  /// 状态复位
  pub fn reset(&mut self) {
    self.undo.clear();
    self.write_lock_holder = 0;
  }
}

/// 元组通用谓词（上游 Tuple::IsFinal/IsWriteLocked/Unlock/WriteLock 谓词族）
pub trait Tuple {
  /// 是否终态版本
  fn is_final(&self) -> bool;
  /// 是否被写锁
  fn is_write_locked(&self) -> bool;
  /// 解写锁（返回持有者）
  fn unlock(&mut self) -> u64;
  /// 加写锁（持有者事务号；已锁返回 None）
  fn write_lock(&mut self, holder: u64) -> Option<u64>;
}

impl Tuple for ChainedTuple {
  fn is_final(&self) -> bool {
    ChainedTuple::is_final(self)
  }
  fn is_write_locked(&self) -> bool {
    self.write_lock_holder != 0
  }
  fn unlock(&mut self) -> u64 {
    let h = self.write_lock_holder;
    self.write_lock_holder = 0;
    h
  }
  fn write_lock(&mut self, holder: u64) -> Option<u64> {
    if self.write_lock_holder == 0 {
      self.write_lock_holder = holder;
      Some(holder)
    } else {
      None
    }
  }
}

/// 更新版本视图（上游 UpdateVersion：事务侧观察）
#[derive(Debug, Clone, Copy)]
pub struct UpdateVersion {
  /// 版本事务
  pub tx: u64,
  /// 是否最终
  pub final_version: bool,
}

impl UpdateVersion {
  /// 是否终态
  pub fn is_final(&self) -> bool {
    self.final_version
  }
}

/// Delta 描述符（FatTuple 页内增量；上游 Delta 系）
///
/// 上游 `BTreeVI.hpp:123-134` 的 `Delta` 为 `{ worker_id:u16, tx_ts:u64,
/// command_id:u32, payload[] }`（C++ 探针：`Delta.payload` 偏移 14），payload 内是
/// `UpdateDescriptor + Diff`。本模型把每个版本存为**整记录**（D1 起即完整记录，
/// 无 diff），故只保留 `offset`/`length`/`tx` 三项：`offset` 是该版本在
/// `DeltaArray::buf` 内的起始字节（`append`/`allocate_delta` 增量维护；
/// 对应上游 `getDeltaOffsets()[d_i]`，`BTreeVI.hpp:161`），`length` 是记录字节数
/// （对应 `value_length`/Diff 长度），`tx` 是该版本创建事务（对应 `tx_ts`）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Delta {
  /// 记录在 `DeltaArray::buf` 内的起始偏移（维护，非恒 0）
  pub offset: u16,
  /// 增量长度
  pub length: u16,
  /// 版本事务
  pub tx: u64,
}

impl Delta {
  /// 常量描述符（定长记录的单属性形态）
  pub fn get_constant_descriptor(offset: u16, length: u16) -> Self {
    Self {
      offset,
      length,
      tx: 0,
    }
  }

  /// 描述符（带事务号）
  pub fn get_descriptor(offset: u16, length: u16, tx: u64) -> Self {
    Self { offset, length, tx }
  }

  /// 增量总长
  pub fn total_length(&self) -> u32 {
    u32::from(self.length)
  }
}

/// 增量数组视图（FatTuple 内的紧凑 Delta 表）
///
/// 顺序与上游一致：**旧→新**（index 0 最旧、末尾最新），对应
/// `BTreeVI.hpp:141` 的 `payload[]; // value, Delta+Descriptor+Diff[] O2N` 及
/// `FatTupleDifferentAttributes.cpp:432-434`（链序新→旧，转胖后 `std::reverse`
/// 重排为 O2N）。上游把最新值放 `getValue()`、历史放 delta 表；本模型把**所有**
/// 版本按旧→新放进 `deltas`，末位即"当前值"，语义等价。
#[derive(Debug, Clone, Default)]
pub struct DeltaArray {
  /// delta 表（**旧→新**，index 0 最旧、末尾最新；与上游 FatTuple O2N 同序）
  pub deltas: Vec<Delta>,
  /// 底层字节（各版本载荷按旧→新拼接）
  pub buf: Vec<u8>,
}

impl DeltaArray {
  /// 偏移表（各 delta 在 buf 内的起点；取自维护好的 `Delta.offset`）
  pub fn get_delta_offsets(&self) -> Vec<u16> {
    self.deltas.iter().map(|d| d.offset).collect()
  }

  /// 常量版偏移表（编译期友好的纯函数形态）
  pub fn get_delta_offsets_constant(deltas: &[Delta]) -> Vec<u16> {
    let mut out = Vec::with_capacity(deltas.len());
    let mut cur = 0u16;
    for d in deltas {
      out.push(cur);
      cur += d.length;
    }
    out
  }

  /// 取第 i 版增量数据（零分配：按维护的 `offset` 直接切片，O(1)）
  pub fn get_delta(&self, i: usize) -> &[u8] {
    let d = &self.deltas[i];
    let s = d.offset as usize;
    &self.buf[s..s + d.length as usize]
  }

  /// 常量版取增量（零分配）
  pub fn get_delta_constant<'a>(buf: &'a [u8], deltas: &[Delta], i: usize) -> &'a [u8] {
    let s: usize = deltas[..i].iter().map(|d| d.length as usize).sum();
    let l = deltas[i].length as usize;
    &buf[s..s + l]
  }

  /// 取第 i 版全值（增量即全值形态：FatTuple D1 = 完整记录）
  pub fn get_value(&self, i: usize) -> &[u8] {
    self.get_delta(i)
  }

  /// 常量版取值
  pub fn get_value_constant<'a>(buf: &'a [u8], deltas: &[Delta], i: usize) -> &'a [u8] {
    Self::get_delta_constant(buf, deltas, i)
  }
}

/// FatTuple（页内多版本紧凑数组；上游 FatTuple + DifferentAttributes 变体）
#[derive(Debug, Clone, Default)]
pub struct FatTuple {
  /// 基准键创建者
  pub creator: u64,
  /// 多版本表
  pub versions: DeltaArray,
  /// 各版本是否删除
  pub removed: Vec<bool>,
  /// 写锁持有者
  pub write_lock_holder: u64,
}

impl FatTuple {
  /// 是否放得下（页容量约束）。
  ///
  /// 上游 `FatTupleDifferentAttributes.cpp:198-202` 以 `total_space`/`data_offset`/
  /// `used_space` 记账，且每条 delta 额外占 `sizeof(u16)` 偏移表项与 `sizeof(Delta)`
  /// 元数据。本模型 `deltas` 元数据在独立 `Vec`、`buf` 只存载荷，故契约简化为
  /// **载荷字节容量**：`buf_len + new_len <= min(capacity, DELTA_ARRAY_MAX_SIZE)`。
  /// 收敛到 [`DELTA_ARRAY_MAX_SIZE`] 是因为偏移表为 `u16`（上游 `BTreeVI.hpp:161`），
  /// 超限会使 `Delta.offset` 截断——与上游"页放不下即拒绝/回退"一致。
  pub fn has_space_for(buf_len: usize, new_len: usize, capacity: usize) -> bool {
    buf_len + new_len <= capacity.min(DELTA_ARRAY_MAX_SIZE)
  }

  /// 追加版本（页内 append；返回是否成功）。
  ///
  /// 维护 `Delta.offset`（= 追加前 `buf.len()`），并保持旧→新顺序。
  pub fn append(&mut self, tx: u64, payload: &[u8], capacity: usize) -> bool {
    if !Self::has_space_for(self.versions.buf.len(), payload.len(), capacity) {
      return false;
    }
    self.versions.deltas.push(Delta {
      offset: self.versions.buf.len() as u16,
      length: payload.len() as u16,
      tx,
    });
    self.versions.buf.extend_from_slice(payload);
    self.removed.push(false);
    true
  }

  /// 分配增量（预留式；上游 AllocateDelta，`FatTupleDifferentAttributes.cpp:204`）
  pub fn allocate_delta(&mut self, tx: u64, len: u16) -> &mut Delta {
    self.versions.deltas.push(Delta {
      offset: self.versions.buf.len() as u16,
      length: len,
      tx,
    });
    self
      .versions
      .buf
      .resize(self.versions.buf.len() + len as usize, 0);
    self.removed.push(false);
    self.versions.deltas.last_mut().unwrap()
  }

  /// 更新（同 append 语义 + 事务号）
  pub fn update(&mut self, tx: u64, payload: &[u8], capacity: usize) -> bool {
    self.append(tx, payload, capacity)
  }

  /// 撤销最近更新（页内回滚；上游 `undoLastUpdate`，
  /// `FatTupleDifferentAttributes.cpp:33-45`）
  pub fn undo_last_update(&mut self) -> bool {
    match self.versions.deltas.pop() {
      Some(d) => {
        let cut = self.versions.buf.len() - d.length as usize;
        self.versions.buf.truncate(cut);
        self.removed.pop();
        true
      }
      None => false,
    }
  }

  /// 重构元组（版本窗口语义：deltas **旧→新**，自末位（最新）向旧取首个
  /// `tx <= snapshot` 的版本定窗口；该窗口为删除墓碑 → None，不回退更旧版本）。
  ///
  /// 与上游 `FatTupleDifferentAttributes.cpp:304-333` 同构：上游在
  /// `isVisibleForMe(head)` 时返回 `getValue()`，否则自 `d_i = deltas_count-1`
  /// 逆序回放直至首个可见版本；上游无墓碑，本模型新增 `removed` 墓碑分支
  /// （对应链式 `BTreeVI.cpp:488/882` 与 `BTreeVI.hpp:456` 的 `NOT_FOUND`）。
  pub fn reconstruct_tuple(&self, snapshot: u64) -> Option<Vec<u8>> {
    for (i, d) in self.versions.deltas.iter().enumerate().rev() {
      if d.tx <= snapshot {
        return if self.removed[i] {
          None
        } else {
          Some(self.versions.get_delta(i).to_vec())
        };
      }
    }
    None
  }

  /// 页内 GC（清除 watermark 内锚点之前的旧版本；锚点 = `tx <= watermark` 的最新
  /// 版本，须保留——它是所有 `snapshot >= watermark` 读者的重构基值）。
  ///
  /// 与上游 `FatTupleDifferentAttributes.cpp:49-196` 的
  /// `deltas_visible_by_all_counter` 边界同构：保留"可见于全部活跃事务"的最新
  /// delta 及其后更新版本（上游 `:74-80` 从最新向旧找首个 `isVisibleForAll`）。
  pub fn garbage_collection(&mut self, watermark: u64) -> usize {
    // 数组旧→新：倒序首个 tx <= watermark 即锚点；锚点及其后（更新版本）保留
    let Some(a) = self.versions.deltas.iter().rposition(|d| d.tx <= watermark) else {
      return 0;
    };
    let mut deltas = Vec::with_capacity(self.versions.deltas.len() - a);
    let mut buf = Vec::with_capacity(self.versions.buf.len());
    for i in a..self.versions.deltas.len() {
      let d = self.versions.deltas[i];
      let offset = buf.len() as u16;
      let s = d.offset as usize;
      buf.extend_from_slice(&self.versions.buf[s..s + d.length as usize]);
      deltas.push(Delta {
        offset,
        length: d.length,
        tx: d.tx,
      });
    }
    self.removed.drain(..a);
    self.versions = DeltaArray { deltas, buf };
    a
  }

  /// 容量查询（页内重排）：当前载荷是否已落在 `new_capacity` 内。
  ///
  /// 上游 `resize`（`FatTupleDifferentAttributes.cpp:335-358`）需把 delta 搬进
  /// 新缓冲；本模型载荷存于 `Vec`（自动扩容），无需搬移，故为纯谓词。
  pub fn resize(&mut self, new_capacity: usize) -> bool {
    self.versions.buf.len() <= new_capacity
  }

  /// 降级为链式（版本过多 / 页满；上游 `convertToChained`，
  /// `FatTupleDifferentAttributes.cpp:468-496`：把**历史** delta 逐个写回
  /// HistoryTree，最新值留作 `ChainedTuple.payload`）。因此 `undo` 只取
  /// `deltas[..n-1]`（不含末位 = head），避免把 head 重复塞进 undo。
  pub fn convert_to_chained(&self) -> ChainedTuple {
    let n = self.versions.deltas.len();
    let (head, head_tx, removed_head) = if n == 0 {
      (Vec::new(), self.creator, false)
    } else {
      (
        self.versions.get_value(n - 1).to_vec(),
        self.versions.deltas[n - 1].tx,
        self.removed[n - 1],
      )
    };
    ChainedTuple {
      creator: self.creator,
      head,
      head_tx,
      removed_head,
      undo: self.versions.deltas[..n.saturating_sub(1)]
        .iter()
        .enumerate()
        .map(|(i, d)| UndoVersion {
          tx_id: d.tx,
          payload: self.versions.get_value(i).to_vec(),
          removed: self.removed[i],
        })
        .collect(),
      write_lock_holder: self.write_lock_holder,
    }
  }
}

/// 元数据（上游 GetMeta：列描述）
#[derive(Debug, Clone, Copy)]
pub struct Meta {
  /// 记录定长
  pub record_len: u16,
  /// 转胖阈值（连续同键更新次数）
  pub fat_threshold: u32,
}

/// BTreeVI（MVCC B+Tree 的元组编排外壳）
pub struct BTreeVi {
  /// 元数据
  pub meta: Meta,
  /// 键 → 链式元组（普通路径）
  pub chained: gxhash::HashMap<Vec<u8>, ChainedTuple>,
  /// 键 → 胖元组（热点路径）
  pub fat: gxhash::HashMap<Vec<u8>, FatTuple>,
  /// 墓地（删除键集合；复用 BTree 实例的语义由 L5 接线 —— 此处集合承载）
  pub graveyard: BTreeSet<Vec<u8>>,
  /// 页容量
  pub capacity: usize,
  /// 全局时钟（版本事务号）
  pub clock: AtomicU64,
}

impl fmt::Debug for BTreeVi {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BTreeVi")
      .field("chained", &self.chained.len())
      .field("fat", &self.fat.len())
      .field("graveyard", &self.graveyard.len())
      .finish()
  }
}

impl BTreeVi {
  /// 建（元数据 + 容量）
  pub fn create(record_len: u16, fat_threshold: u32, capacity: usize) -> Self {
    Self {
      meta: Meta {
        record_len,
        fat_threshold,
      },
      chained: HashMap::default(),
      fat: HashMap::default(),
      graveyard: BTreeSet::new(),
      capacity,
      clock: AtomicU64::new(1),
    }
  }

  /// 元数据
  pub fn get_meta(&self) -> Meta {
    self.meta
  }

  /// 插入
  pub async fn insert(&mut self, key: &[u8], payload: &[u8], tx: u64) -> Result<()> {
    if self.chained.contains_key(key) || self.fat.contains_key(key) {
      return Err(Error::DuplicateKey(key.to_vec()));
    }
    let mut ct = ChainedTuple {
      creator: tx,
      head: payload.to_vec(),
      head_tx: tx,
      removed_head: false,
      undo: Vec::new(),
      write_lock_holder: 0,
    };
    ct.reset();
    self.chained.insert(key.to_vec(), ct);
    Ok(())
  }

  /// 点查（乐观：无闩读 + 可见性闭包）
  pub async fn lookup_optimistic(&self, key: &[u8], snapshot: u64) -> Result<Option<Vec<u8>>> {
    // 胖路径
    if let Some(ft) = self.fat.get(key) {
      return Ok(ft.reconstruct_tuple(snapshot));
    }
    // 链路径：head_tx 与 undo 全序竞争取最新可见
    let Some(ct) = self.chained.get(key) else {
      return Ok(None);
    };
    Ok(ct.value_at(snapshot).map(|v| v.to_vec()))
  }

  /// 悲观点查（持锁；与乐观同构 —— 闩协议由 L5 接线）
  pub async fn lookup_pessimistic(&self, key: &[u8], snapshot: u64) -> Result<Option<Vec<u8>>> {
    self.lookup_optimistic(key, snapshot).await
  }

  /// 点查（账本命名）
  pub async fn lookup(&self, key: &[u8], snapshot: u64) -> Result<Option<Vec<u8>>> {
    self.lookup_optimistic(key, snapshot).await
  }

  /// 可见性判定（对某版本）。简化模型：`tx <= snapshot` 即可见；
  /// 上游 `BTreeVI.hpp:441` 委托 `ConcurrencyControl::isVisibleForMe`
  /// （OSIC/CommitTree），本 crate 以谓词解耦，真实 OSIC 判定由 `ldb_txn` 接线。
  pub fn is_visible_for_me(&self, version_tx: u64, snapshot: u64) -> bool {
    version_tx <= snapshot
  }

  /// 同尺寸原地更新（RC 优化路径；上游 `updateSameSizeInPlace`，`BTreeVI.cpp:232`）。
  /// 链式分支在写入后调用 `maybe_promote`，与上游 `:289-328` 的晋升触发一致。
  pub async fn update_same_size_in_place(
    &mut self,
    key: &[u8],
    payload: &[u8],
    tx: u64,
  ) -> Result<bool> {
    if let Some(ct) = self.chained.get_mut(key) {
      if ct.head.len() != payload.len() {
        return Ok(false);
      }
      ct.undo.push(UndoVersion {
        tx_id: ct.head_tx,
        payload: ct.head.clone(),
        removed: ct.removed_head,
      });
      ct.head.copy_from_slice(payload);
      ct.head_tx = tx;
      ct.removed_head = false;
    } else if let Some(ft) = self.fat.get_mut(key) {
      // 当前值 = 最新版本（末位），非最旧（首位）
      let cur_len = ft.versions.deltas.last().map(|d| d.length).unwrap_or(0);
      if cur_len == payload.len() as u16 {
        return Ok(ft.update(tx, payload, self.capacity));
      }
      return Ok(false);
    } else {
      return Ok(false);
    }
    self.maybe_promote(key);
    Ok(true)
  }

  /// 确定性更新预备（锁外构造 Delta；上游 `prepareDeterministicUpdate`，
  /// `BTreeVI.cpp:155`）。`Delta.offset` 仅在入 `buf` 时才有意义，预备态取 0。
  pub fn prepare_deterministic_update(
    &self,
    _key: &[u8],
    payload: &[u8],
    tx: u64,
  ) -> Result<Delta> {
    Ok(Delta::get_descriptor(0, payload.len() as u16, tx))
  }

  /// 执行确定性更新（预备后落盘）
  pub async fn execute_deterministric_update(
    &mut self,
    key: &[u8],
    delta: Delta,
    payload: &[u8],
  ) -> Result<bool> {
    let _ = delta;
    if let Some(ft) = self.fat.get_mut(key) {
      return Ok(ft.append(delta.tx, payload, self.capacity));
    }
    if let Some(ct) = self.chained.get_mut(key) {
      ct.undo.push(UndoVersion {
        tx_id: ct.head_tx,
        payload: take(&mut ct.head),
        removed: ct.removed_head,
      });
      ct.head = payload.to_vec();
      ct.head_tx = delta.tx;
      ct.removed_head = false;
      // 热点晋升检查
      self.maybe_promote(key);
      return Ok(true);
    }
    Ok(false)
  }

  /// 热点晋升阈值（连续更新 >= fat_threshold 转胖）
  pub async fn convert_to_fat_tuple_threshold(&self) -> u32 {
    self.meta.fat_threshold
  }

  /// 胖元组最大长度（页容量约束）
  pub async fn max_fat_tuple_length(&self) -> usize {
    self.capacity
  }

  /// 晋升检查（链 → 胖；版本数含 head）。
  ///
  /// 阈值口径与上游**严格 `>` 一致**：上游 `updateSameSizeInPlace` 用
  /// `tuple_head.updates_counter > convertToFatTupleThreshold()`（`BTreeVI.cpp:308`），
  /// `convertChainedToFatTupleDifferentAttributes` 用
  /// `number_of_deltas_to_replace > convertToFatTupleThreshold()`（`FatTupleDifferentAttributes.cpp:443`）。
  /// 本模型以"版本数（含 head）= `undo.len() + 1`"为计数，故晋升条件为
  /// `undo.len() + 1 > fat_threshold`（即**严格大于**，不是 `>=`）。
  /// 上游在 `convertChainedToFatTupleDifferentAttributes`（`:360`）失败可回退链式；
  /// 本模型同构：整键载荷放不下（含超出 `u16` 偏移上限）则**保持链式**，不静默丢版本。
  fn maybe_promote(&mut self, key: &[u8]) {
    let Some(ct) = self.chained.get(key) else {
      return;
    };
    if ((ct.undo.len() + 1) as u32) <= self.meta.fat_threshold {
      return;
    }
    let needed = ct.undo.iter().map(|v| v.payload.len()).sum::<usize>() + ct.head.len();
    if !FatTuple::has_space_for(0, needed, self.capacity) {
      return; // 页放不下或超 u16 偏移上限 → 回退链式
    }
    let Some(ct) = self.chained.remove(key) else {
      return;
    };
    let mut ft = FatTuple {
      creator: ct.creator,
      ..Default::default()
    };
    // 旧→新序（deltas 数组语义）：undo 追加序旧→新直入，head 最新最后
    for v in &ct.undo {
      ft.append(v.tx_id, &v.payload, self.capacity);
      if v.removed
        && let Some(last) = ft.removed.last_mut()
      {
        *last = true;
      }
    }
    ft.append(ct.head_tx, &ct.head, self.capacity);
    if ct.removed_head
      && let Some(last) = ft.removed.last_mut()
    {
      *last = true;
    }
    self.fat.insert(key.to_vec(), ft);
  }

  /// 链式元组重构（undo 回放至快照）
  pub async fn reconstruct_chained_tuple(&self, key: &[u8], snapshot: u64) -> Option<Vec<u8>> {
    let ct = self.chained.get(key)?;
    ct.value_at(snapshot).map(|v| v.to_vec())
  }

  /// 元组重构（统一入口）
  pub async fn reconstruct_tuple(&self, key: &[u8], snapshot: u64) -> Option<Vec<u8>> {
    if let Some(ft) = self.fat.get(key) {
      return ft.reconstruct_tuple(snapshot);
    }
    self.reconstruct_chained_tuple(key, snapshot).await
  }

  /// 胖元组直访（测试 / 审计）
  pub fn fat_tuple(&self, key: &[u8]) -> Option<&FatTuple> {
    self.fat.get(key)
  }

  /// 胖元组可变直访
  pub fn fat_tuple_mut(&mut self, key: &[u8]) -> Option<&mut FatTuple> {
    self.fat.get_mut(key)
  }

  /// 删除（版本链上标记 removed；OLAP GC 时迁墓；上游 `remove`，`BTreeVI.cpp:450`）
  pub async fn remove(&mut self, key: &[u8], tx: u64) -> Result<bool> {
    if let Some(ct) = self.chained.get_mut(key) {
      ct.undo.push(UndoVersion {
        tx_id: ct.head_tx,
        payload: ct.head.clone(),
        removed: ct.removed_head,
      });
      ct.head_tx = tx;
      ct.removed_head = true; // head 变删除墓碑（removed_head 为准，字节保留）
      return Ok(true);
    }
    if let Some(ft) = self.fat.get_mut(key) {
      // 删除版本 = 空载荷 + removed（上游 `BTreeVI.cpp:452` 注释 "remove fat tuple" 标注未支持，本模型扩展）
      if ft.append(tx, &[], self.capacity) {
        if let Some(last) = ft.removed.last_mut() {
          *last = true;
        }
        return Ok(true);
      }
    }
    Ok(false)
  }

  /// 撤销（事务回滚）。链：按 `head_tx` 摘头还原，回滚到创建前则摘键
  /// （上游 `undo` 的 WALInsert 分支 `BTreeVI.cpp:549-564` 即 `removeCurrent`）；
  /// 胖：摘除该 `tx` 的增量并重建 `buf`/`removed`（修正错位）。
  pub async fn undo(&mut self, key: &[u8], tx: u64) -> bool {
    if self.chained.contains_key(key) {
      let mut remove_key = false;
      if let Some(ct) = self.chained.get_mut(key) {
        if ct.head_tx == tx {
          match ct.undo.pop() {
            Some(v) => {
              ct.head = v.payload;
              ct.head_tx = v.tx_id;
              ct.removed_head = v.removed;
            }
            None => remove_key = true, // 回滚到创建前 → 摘键
          }
        } else {
          let before = ct.undo.len();
          ct.undo.retain(|v| v.tx_id != tx);
          if ct.undo.len() == before {
            return false;
          }
        }
      }
      if remove_key {
        self.chained.remove(key);
      }
      return true;
    }
    if let Some(ft) = self.fat.get_mut(key) {
      let Some(pos) = ft.versions.deltas.iter().position(|d| d.tx == tx) else {
        return false;
      };
      let d = ft.versions.deltas[pos];
      let s = d.offset as usize;
      ft.versions.buf.drain(s..s + d.length as usize);
      ft.versions.deltas.remove(pos);
      ft.removed.remove(pos);
      // 重算摘除点之后各 delta 的 offset
      let mut off = s as u16;
      for k in pos..ft.versions.deltas.len() {
        ft.versions.deltas[k].offset = off;
        off += ft.versions.deltas[k].length;
      }
      return true;
    }
    false
  }

  /// 解写锁（账本 BTreeVI::unlock）
  pub async fn unlock(&mut self, key: &[u8]) -> u64 {
    if let Some(ct) = self.chained.get_mut(key) {
      return ct.unlock();
    }
    if let Some(ft) = self.fat.get_mut(key) {
      let h = ft.write_lock_holder;
      ft.write_lock_holder = 0;
      return h;
    }
    0
  }

  /// 加写锁
  pub async fn write_lock_tuple(&mut self, key: &[u8], holder: u64) -> bool {
    if let Some(ct) = self.chained.get_mut(key) {
      return ct.write_lock(holder).is_some();
    }
    if let Some(ft) = self.fat.get_mut(key)
      && ft.write_lock_holder == 0
    {
      ft.write_lock_holder = holder;
      return true;
    }
    false
  }

  /// 升序扫描
  pub async fn scan_asc(
    &self,
    lo: &[u8],
    hi: &[u8],
    snapshot: u64,
    mut cb: impl FnMut(&[u8], &[u8]) -> bool,
  ) -> usize {
    self.scan(lo, hi, snapshot, &mut cb, true).await
  }

  /// 降序扫描
  pub async fn scan_desc(
    &self,
    lo: &[u8],
    hi: &[u8],
    snapshot: u64,
    mut cb: impl FnMut(&[u8], &[u8]) -> bool,
  ) -> usize {
    self.scan(lo, hi, snapshot, &mut cb, false).await
  }

  /// 通用扫描（升/降序；graveyard 键跳过）
  pub async fn scan(
    &self,
    lo: &[u8],
    hi: &[u8],
    snapshot: u64,
    cb: &mut impl FnMut(&[u8], &[u8]) -> bool,
    ascending: bool,
  ) -> usize {
    let mut items: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    // 链
    for (k, ct) in &self.chained {
      if k.as_slice() >= lo
        && (hi.is_empty() || k.as_slice() < hi)
        && let Some(v) = ct.value_at(snapshot)
      {
        items.push((k.clone(), v.to_vec()));
      }
    }
    // 胖
    for (k, ft) in &self.fat {
      if k.as_slice() >= lo
        && (hi.is_empty() || k.as_slice() < hi)
        && let Some(v) = ft.reconstruct_tuple(snapshot)
      {
        items.push((k.clone(), v));
      }
    }
    items.sort_by(|a, b| a.0.cmp(&b.0));
    if !ascending {
      items.reverse();
    }
    let mut n = 0;
    for (k, v) in items {
      if self.graveyard.contains(&k) {
        continue; // 墓地键跳过（OLAP 加速）
      }
      n += 1;
      if !cb(&k, &v) {
        break;
      }
    }
    n
  }

  /// OLAP 扫描（graveyard 参与：墓中删除键也回调，用于反连接）
  pub async fn scan_olap(
    &self,
    lo: &[u8],
    hi: &[u8],
    snapshot: u64,
    mut cb: impl FnMut(&[u8], &[u8]) -> bool,
  ) -> usize {
    let mut n = self.scan(lo, hi, snapshot, &mut cb, true).await;
    // 墓地区间段：BTreeSet 有序，自 `lo` 起遍历，越 `hi`（非空）即停；
    // `hi` 为空表示无上界（与 `scan` 语义一致），不得退化为空区间。
    for k in self.graveyard.range(lo.to_vec()..) {
      if !hi.is_empty() && k.as_slice() >= hi {
        break;
      }
      n += 1;
      if !cb(k, &[]) {
        break;
      }
    }
    n
  }

  /// 空间利用率检查（胖元组页满触发降级）
  pub async fn check_space_utilization(&mut self, key: &[u8]) -> bool {
    if let Some(ft) = self.fat.get(key) {
      ft.versions.buf.len() >= self.capacity * 9 / 10
    } else {
      false
    }
  }

  /// 触发页级 GC（清除 watermark 内锚点之前的旧版本 + 墓键迁移）。
  ///
  /// 链式 GC 与胖式一致：保留 `tx <= watermark` 的**最新**版本（锚点）及其后
  /// 更新版本。锚点是所有 `snapshot >= watermark` 读者的重构基值，绝不能删——
  /// 若锚点是删除墓碑而把它删掉，`snapshot ∈ (锚点.tx, 下一版本)` 会错误回溯到
  /// 更旧的存活版本（原实现只按 `removed` 过滤，存在此缺陷）。
  pub async fn trigger_page_wise_garbage_collection(&mut self, watermark: u64) -> usize {
    let mut purged = 0;
    // 胖：原位 retain，避免全键克隆
    self.fat.retain(|_, ft| {
      purged += ft.garbage_collection(watermark);
      true
    });
    // 链：保留锚点及其后；最新为墓碑且过水位 → 整键迁墓
    let mut migrated: Vec<Vec<u8>> = Vec::new();
    self.chained.retain(|k, ct| {
      if ct.removed_head && ct.head_tx <= watermark {
        migrated.push(k.clone());
        purged += 1;
        return false;
      }
      let keep_from = if ct.head_tx <= watermark {
        ct.undo.len() // 锚点即 head，undo 全清
      } else {
        ct.undo
          .iter()
          .rposition(|v| v.tx_id <= watermark)
          .unwrap_or(0) // 无 <= watermark 版本 → 全保留
      };
      purged += keep_from;
      ct.undo.drain(..keep_from);
      true
    });
    for k in migrated {
      self.graveyard.insert(k);
    }
    purged
  }

  /// 序列化（检查点）。
  ///
  /// **无损**编码本 crate 的全部多版本状态：链式元组（`creator`/`head_tx`/
  /// `removed_head`/`head` + 完整 `undo` 链）、胖元组（各版本 `tx`/墓碑/载荷）、以及
  /// 墓地的已删除键。上游检查点由 `BTreeGeneric::serialize` 承载整页镜像
  /// （`BTreeVI.hpp:249-252`），本模型把版本链内联在元组内，故检查点须一并持久化，
  /// 否则恢复后可见性/版本数与预期不符（`write_lock_holder` 为瞬态闩状态，不入盘）。
  /// 键长用 `u16`（上游键长上限），载荷长用 `u32`。
  pub async fn serialize(&self, out: &mut Vec<u8>) {
    out.extend_from_slice(&(self.chained.len() as u32).to_le_bytes());
    for (k, ct) in &self.chained {
      put_short(out, k);
      out.extend_from_slice(&ct.creator.to_le_bytes());
      out.extend_from_slice(&ct.head_tx.to_le_bytes());
      out.push(u8::from(ct.removed_head));
      put_blob(out, &ct.head);
      out.extend_from_slice(&(ct.undo.len() as u32).to_le_bytes());
      for v in &ct.undo {
        out.extend_from_slice(&v.tx_id.to_le_bytes());
        out.push(u8::from(v.removed));
        put_blob(out, &v.payload);
      }
    }
    out.extend_from_slice(&(self.fat.len() as u32).to_le_bytes());
    for (k, ft) in &self.fat {
      put_short(out, k);
      out.extend_from_slice(&ft.creator.to_le_bytes());
      out.extend_from_slice(&(ft.versions.deltas.len() as u32).to_le_bytes());
      for (i, d) in ft.versions.deltas.iter().enumerate() {
        out.extend_from_slice(&d.tx.to_le_bytes());
        out.push(u8::from(ft.removed[i]));
        put_blob(out, ft.versions.get_delta(i));
      }
    }
    out.extend_from_slice(&(self.graveyard.len() as u32).to_le_bytes());
    for k in &self.graveyard {
      put_short(out, k);
    }
  }

  /// 反序列化（检查点恢复），返回消耗的字节数。
  ///
  /// 输入是**不可信字节**（磁盘/网络）：所有长度前缀读取均带边界校验，越界或申明
  /// 长度超过 `u16` 上限返回 [`Error::CorruptState`]，绝不 panic，也**不按不可信长度
  /// 预分配**（修正 todo §9.3 点名的 `btree_vi.rs:817-827`）。
  pub async fn deserialize(&mut self, buf: &[u8]) -> Result<usize> {
    let mut pos = 0usize;
    let n_chained = read_u32(buf, pos)? as usize;
    pos += 4;
    for _ in 0..n_chained {
      let (key, p) = read_short(buf, pos)?;
      pos = p;
      let creator = read_u64(buf, pos)?;
      pos += 8;
      let head_tx = read_u64(buf, pos)?;
      pos += 8;
      let removed_head = read_u8(buf, pos)? != 0;
      pos += 1;
      let (head, p) = read_blob(buf, pos)?;
      pos = p;
      let n_undo = read_u32(buf, pos)? as usize;
      pos += 4;
      let mut undo = Vec::new();
      for _ in 0..n_undo {
        let tx_id = read_u64(buf, pos)?;
        pos += 8;
        let removed = read_u8(buf, pos)? != 0;
        pos += 1;
        let (payload, p) = read_blob(buf, pos)?;
        pos = p;
        undo.push(UndoVersion {
          tx_id,
          payload: payload.to_vec(),
          removed,
        });
      }
      self.chained.insert(
        key.to_vec(),
        ChainedTuple {
          creator,
          head: head.to_vec(),
          head_tx,
          removed_head,
          undo,
          write_lock_holder: 0,
        },
      );
    }
    let n_fat = read_u32(buf, pos)? as usize;
    pos += 4;
    for _ in 0..n_fat {
      let (key, p) = read_short(buf, pos)?;
      pos = p;
      let creator = read_u64(buf, pos)?;
      pos += 8;
      let n_delta = read_u32(buf, pos)? as usize;
      pos += 4;
      let mut ft = FatTuple {
        creator,
        ..Default::default()
      };
      for _ in 0..n_delta {
        let tx = read_u64(buf, pos)?;
        pos += 8;
        let removed = read_u8(buf, pos)? != 0;
        pos += 1;
        let (payload, p) = read_blob(buf, pos)?;
        pos = p;
        if payload.len() > u16::MAX as usize || !ft.append(tx, payload, DELTA_ARRAY_MAX_SIZE) {
          // 单条载荷超 u16 或累计缓冲超 u16 偏移上限 → 拒绝（不可信输入）
          return Err(Error::CorruptState);
        }
        if removed && let Some(last) = ft.removed.last_mut() {
          *last = true;
        }
      }
      self.fat.insert(key.to_vec(), ft);
    }
    let n_grave = read_u32(buf, pos)? as usize;
    pos += 4;
    for _ in 0..n_grave {
      let (key, p) = read_short(buf, pos)?;
      pos = p;
      self.graveyard.insert(key.to_vec());
    }
    Ok(pos)
  }

  /// 异步回调执行器（上游 `dt_registry.todo` 注册形态，`BTreeVI.cpp:685`）。
  ///
  /// 本单线程模型无在途调度队列，点级迁移/清除由
  /// [`BTreeVi::trigger_page_wise_garbage_collection`] 统一完成，故此处为空实现。
  pub async fn todo(&mut self, dt_id: u16, payload: &[u8], called_before: bool) {
    if !called_before {
      // 首次登记（审计点）
      let _ = (dt_id, payload);
    }
  }
}

/// 读取小端 `u32`（越界 → [`Error::CorruptState`]）
fn read_u32(buf: &[u8], pos: usize) -> Result<u32> {
  buf
    .get(pos..pos + 4)
    .and_then(|s| s.try_into().ok())
    .map(u32::from_le_bytes)
    .ok_or(Error::CorruptState)
}

/// 读取小端 `u16`（越界 → [`Error::CorruptState`]）
fn read_u16(buf: &[u8], pos: usize) -> Result<u16> {
  buf
    .get(pos..pos + 2)
    .and_then(|s| s.try_into().ok())
    .map(u16::from_le_bytes)
    .ok_or(Error::CorruptState)
}

/// 读取定长切片（越界 → [`Error::CorruptState`]）
fn read_bytes(buf: &[u8], pos: usize, len: usize) -> Result<&[u8]> {
  buf.get(pos..pos + len).ok_or(Error::CorruptState)
}

/// 读取小端 `u64`（越界 → [`Error::CorruptState`]）
fn read_u64(buf: &[u8], pos: usize) -> Result<u64> {
  buf
    .get(pos..pos + 8)
    .and_then(|s| s.try_into().ok())
    .map(u64::from_le_bytes)
    .ok_or(Error::CorruptState)
}

/// 读取单字节（越界 → [`Error::CorruptState`]）
fn read_u8(buf: &[u8], pos: usize) -> Result<u8> {
  buf.get(pos).copied().ok_or(Error::CorruptState)
}

/// 读取 `u16` 长度前缀 + 字节，返回 `(切片, 新位置)`
fn read_short(buf: &[u8], pos: usize) -> Result<(&[u8], usize)> {
  let len = read_u16(buf, pos)? as usize;
  let p = pos + 2;
  Ok((read_bytes(buf, p, len)?, p + len))
}

/// 读取 `u32` 长度前缀 + 字节，返回 `(切片, 新位置)`
fn read_blob(buf: &[u8], pos: usize) -> Result<(&[u8], usize)> {
  let len = read_u32(buf, pos)? as usize;
  let p = pos + 4;
  Ok((read_bytes(buf, p, len)?, p + len))
}

/// 写 `u16` 长度前缀 + 字节
fn put_short(out: &mut Vec<u8>, bytes: &[u8]) {
  out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
  out.extend_from_slice(bytes);
}

/// 写 `u32` 长度前缀 + 字节
fn put_blob(out: &mut Vec<u8>, bytes: &[u8]) {
  out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
  out.extend_from_slice(bytes);
}

/// ChainedTuple → FatTuple 转换（不同属性变体；上游
/// `BTreeVI::convertChainedToFatTupleDifferentAttributes`，
/// `FatTupleDifferentAttributes.cpp:360`）。
///
/// 保留链式 `undo` 的墓碑标记（`removed`）与 head 的 `removed_head`——否则转换后
/// 删除窗口会被错误复活（上游链式删除语义 `BTreeVI.cpp:488/882`）。
pub async fn convert_chained_to_fat_tuple_different_attributes(
  ct: &ChainedTuple,
  capacity: usize,
) -> Option<FatTuple> {
  let mut ft = FatTuple {
    creator: ct.creator,
    ..Default::default()
  };
  for v in &ct.undo {
    if !ft.append(v.tx_id, &v.payload, capacity) {
      return None;
    }
    if v.removed
      && let Some(last) = ft.removed.last_mut()
    {
      *last = true;
    }
  }
  if !ft.append(ct.head_tx, &ct.head, capacity) {
    return None;
  }
  if ct.removed_head
    && let Some(last) = ft.removed.last_mut()
  {
    *last = true;
  }
  Some(ft)
}
