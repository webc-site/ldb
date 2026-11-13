//! Doublewrite 双写缓冲（上游 `doublewrite.cc` 的偏移语义 + 本项目崩溃安全实现）。
//!
//! # 上游语义（逐条对照）
//!
//! `ZLeanStore/src/storage/space/doublewrite.h:10-18` 自述
//! 「current implementation only mocks the DWB behavior」：上游只有每 worker
//! 8 MiB 区的偏移游标，没有任何落盘代码。写路径在
//! `io_interface.cc:765-772`（`AddPageToWriteBuffer`）：
//!
//! 1. `SetWriteOffset()`（只重算 start/end，**不**重置 `cnt`/`cur`）；
//! 2. `AdvanceWriteOffset()`（`cnt` 环形 +1，`cur = start + cnt*PAGE_SIZE`）；
//! 3. `WritePageSynchronously(..., 999)` —— 同步提交 DWB 页（io_uring 完成即返回，
//!    **不 fsync**）；
//! 4. 页进入写缓冲，稍后 `FlushWriteBuffer` 写最终位置（同样无 fsync）。
//!
//! 因此上游 DWB **不存在**「DWB 先落盘、再写最终位置」的持久化顺序，也没有
//! 校验/恢复判定；它是 mock。本模块 `Doublewrite` 严格保留该偏移语义（A1），
//! 不在 L1 编造上游没有的顺序。
//!
//! # 本项目新实现：`DwbWriter`
//!
//! 真正防半页写（torn write）的协议必须自己定义持久化边界，本模块给出可验证
//! 的记录日志：
//!
//! - 区布局（每 worker `DWB_AREA` 字节，`area_base` 起）：连续记录
//!   `[64B 头 | 4096B 页]`，记录头含 magic/version/len/crc32/generation/final_off。
//! - 写批：逐页 `write_page` 追加记录 → `sync`（DWB 持久化屏障）→ 写最终位置 →
//!   `checkpoint`（先 `data.sync_durable()`，再清零区首记录头）。
//! - 恢复 `recover`：从区首线性扫描，校验 magic/len/crc，只接受与首条记录相同
//!   `generation` 的连续记录；按记录顺序回放到各自 `final_off`，data 持久化后
//!   清零区首记录头（失效），使重复恢复幂等。
//! - 崩溃语义：DWB 未 fsync 时记录可能丢失/撕裂 → CRC 拒绝 → 最终位置保持旧值；
//!   DWB 已 fsync → 恢复必回放整批；最终位置已 fsync 后崩溃 → 回放同一内容（幂等）。
//! - `generation` 每次 checkpoint +1、进程启动随机初始化：新一轮批从区首覆写旧
//!   记录时，旧批记录（generation 不同）不会被误回放。
//! - 槽位失效：checkpoint 把区首记录头清零（不额外 fsync），`cur` 回到区首；旧批
//!   在后续槽位留下的记录由 generation 区分。失效头丢失只是让下次启动多回放一遍
//!   已提交批次（幂等），不构成正确性依赖。generation 回绕（u64 用尽）时
//!   checkpoint 清零**全部**记录头，杜绝旧代号与旧槽位碰撞。
//! - 唯一安全入口是 [`DwbWriter::open`]（先 recover 再返回可写句柄），保证不会在
//!   未失效的有效记录上从区首覆写；区满时 `write_page` 返回 `StorageFull`。
//!
//! # DWB 与 Direct I/O（R3 布局决策：方案 c）
//!
//! 记录 `[64B 头 | 4096B 页] = 4160B`，槽偏移 `area_base + n*4160`。Linux/Windows
//! 的 Direct I/O 要求地址/长度/偏移均为 4 KiB 倍数，而 4160 与 `n*4160` 在 `n>0`
//! 时都不满足 ⇒ 若用 `O_DIRECT` 打开 DWB，首次 `write_page`/`recover` 读即触发
//! 整文件降级。三种候选布局的对比：
//!
//! | 方案 | 空间 | 每页 DWB 写字节 | 崩溃一致性 | 复杂度 |
//! |---|---|---|---|---|
//! | a. 记录扩到 8192B | 1024 槽（减半） | 8192（2×） | 头+页 CRC 仍成立；尾部 4032B 填充不参与校验 | 低 |
//! | b. 头区/页区分段 | 头区 32 页 + 2016 页槽 = 8MiB 刚好 | 4096 + 头摊销（64 头/4KiB 块 ≈ 64B/页） | 头与页两次写，任一先持久化都由 CRC 拒绝（头 CRC 覆盖页） | 高（两套偏移/批量刷头） |
//! | **c. 保持 4160B，DWB 走缓冲 I/O** | **2016 槽（满 8MiB 区）** | **4160（1×，页缓存合并写）** | **`sync_durable` 返回即全部记录持久化；未 sync 的任意前缀/乱序由 CRC 停扫** | **低** |
//!
//! 选 (c)：
//! 1. DWB 在写最终位置之前必须 `fdatasync` 一次；缓冲写让 2016 条 4160B 记录在
//!    页缓存合并成少量大块回写，设备写放大 = 1 页/页（方案 a 是 2 页/页，再加最终
//!    写共 3 页/页）。RocksDB 对比关注的是写放大，方案 a 的 2× DWB 流量不可接受。
//! 2. 方案 b 省空间且带宽最优，但需要「头块批量刷 + 页区刷」两套写序，且恢复扫描
//!    要跨两段；当前 DWB 尚未接 L3，先不引入双重协议。
//! 3. 正确性不依赖 DWB 是否 Direct：`DwbWriter::sync` → `dwb.sync_durable()`
//!    （Linux fdatasync / macOS F_FULLFSYNC）是持久化屏障；调用方在其返回后才写
//!    最终位置；`checkpoint` 先 `data.sync_durable()` 再清零区首头。**fsync 调用点**：
//!    `sync`（DWB 记录）、`checkpoint` 的 data 屏障、`recover` 的 data 回放屏障。
//!    失效头（清零区首 64B）**不 fsync**：丢失后旧批记录仍在，恢复幂等重放；新批
//!    覆写槽 0 时 generation/CRC 决定停扫位置。缓冲 I/O 的 fsync 语义与 Direct I/O
//!    相同（都等待设备确认）；未 sync 时允许任意前缀/乱序持久化，恢复按 CRC 停扫，
//!    绝不回放半条记录。
//! 4. Linux/Windows 上用 `CompioFileBackend::open_buffered` 打开 DWB；若误传
//!    Direct 句柄，[`DwbWriter::recover`] 直接返回 `InvalidInput`（macOS
//!    `F_NOCACHE` 无对齐要求，不在此列）。
//! 5. DWB 恢复回放使用 4 KiB 对齐的 [`AlignedPage`]，避免 `to_vec()` 的非对齐
//!    页缓冲把 `data.db` 也拖入缓冲 I/O。
//!
//! 恢复必须在服务新写入之前调用（`DwbWriter::open` 已内置）。
//!
//! 注意：`Doublewrite`（上游页槽游标）与 `DwbWriter`（记录日志）是同一
//! `DWB_AREA` 的两种互斥布局，实际使用二选一；新实现只用 `DwbWriter`。

use std::{
  collections::hash_map::RandomState,
  hash::BuildHasher,
  io,
  time::{SystemTime, UNIX_EPOCH},
};

use compio::buf::IoBuf;
use crc32fast::Hasher;
use ldb_type::PAGE_SIZE;

use crate::compio_backend::CompioFileBackend;

/// 每 worker 的 DWB 区大小（8 MiB，与上游 `MB*8` 一致）
pub const DWB_AREA: usize = 8 * 1024 * 1024;

/// DWB 记录头大小（一个缓存行；含 32B 保留区）
pub const DWB_HEADER: usize = 64;

/// DWB 记录大小（头 + 一页）
pub const DWB_RECORD: usize = DWB_HEADER + PAGE_SIZE;

/// 每区最多容纳的记录数
pub const DWB_RECORDS: usize = DWB_AREA / DWB_RECORD;

const DWB_MAGIC: u32 = 0x4C44_5742; // "LDWB"
const DWB_VERSION: u16 = 1;

static ZERO_HEADER: [u8; DWB_HEADER] = [0; DWB_HEADER];

/// Doublewrite 偏移游标（每 worker 一个实例）。
///
/// 与上游 `Doublewrite` 逐字段一致；`advance_write_offset` 的环形推进与
/// `set_write_offset` 的「只重算边界、保留 cnt/cur」语义均按
/// `ZLeanStore/src/storage/space/doublewrite.cc:20-39` 保留。
#[derive(Debug, Clone)]
pub struct Doublewrite {
  /// 区起始偏移（文件内）
  pub dwb_start_offset: usize,
  /// 区结束偏移（开区间）
  pub dwb_end_offset: usize,
  /// 当前写偏移
  pub dwb_cur_w_offset: usize,
  /// 已写页计数（环形）
  pub cnt: usize,
}

impl Doublewrite {
  /// 构造：`area_base` 为本 worker 区在 dwb 文件内的起始偏移
  pub fn new(area_base: usize) -> Self {
    debug_assert!(area_base.is_multiple_of(PAGE_SIZE));
    let end = area_base + DWB_AREA;
    Self {
      dwb_start_offset: area_base,
      dwb_end_offset: end,
      dwb_cur_w_offset: area_base,
      cnt: 0,
    }
  }

  /// 重置区边界（worker 迁移 / 恢复后重绑定）。
  ///
  /// 上游 `SetWriteOffset`（`doublewrite.cc:20-25`）只改 start/end，**不**改
  /// `dwb_cur_w_offset_` 与 `cnt`；下一次 `AdvanceWriteOffset` 会在新区的同一
  /// `cnt` 槽继续。此处保持同样语义。
  pub fn set_write_offset(&mut self, area_base: usize) {
    debug_assert!(area_base.is_multiple_of(PAGE_SIZE));
    self.dwb_start_offset = area_base;
    self.dwb_end_offset = area_base + DWB_AREA;
  }

  /// 推进一个页槽（回卷）
  pub fn advance_write_offset(&mut self) {
    self.cnt = if self.cnt + 1 == Self::capacity_pages() {
      0
    } else {
      self.cnt + 1
    };
    self.dwb_cur_w_offset = self.dwb_start_offset + self.cnt * PAGE_SIZE;
    debug_assert!(self.dwb_cur_w_offset >= self.dwb_start_offset);
    debug_assert!(self.dwb_cur_w_offset < self.dwb_end_offset);
  }

  /// 区容量（页数，与上游 `MB*8/PAGE_SIZE` 一致）
  pub const fn capacity_pages() -> usize {
    DWB_AREA / PAGE_SIZE
  }
}

/// 崩溃安全的 DWB 记录日志写端（每次 flush 批次一个；每 worker 独立区）。
#[derive(Debug)]
pub struct DwbWriter {
  area_base: u64,
  area_end: u64,
  cur: u64,
  generation: u64,
}

/// 记录头解析结果
#[derive(Debug, Clone, Copy)]
struct Record {
  final_off: u64,
  generation: u64,
}

/// 4 KiB 对齐的拥有式页缓冲：恢复回放时保证 `data.db` 仍可走 Direct I/O。
///
/// `Vec<u8>` 只保证字节对齐；把 `rec[64..]` 拷进本类型再写，地址轴满足
/// `DIRECT_ALIGN`，避免一次恢复回放就把 data 后端降级为缓冲 I/O。
#[repr(align(4096))]
struct AlignedPage([u8; PAGE_SIZE]);

impl IoBuf for AlignedPage {
  fn as_init(&self) -> &[u8] {
    &self.0
  }
}

impl DwbWriter {
  /// 打开一个 worker 的 DWB 区：**先回放并失效旧批**，再返回可写句柄与回放页数。
  ///
  /// 这是唯一的安全构造入口。若区首仍有有效记录（上次崩溃在 checkpoint 之前），
  /// 必须先回放；否则新批从区首覆写会静默破坏唯一副本。新建/空文件回放 0 页。
  ///
  /// `dwb` 在 Linux/Windows 上必须用
  /// [`CompioFileBackend::open_buffered`](crate::CompioFileBackend::open_buffered)
  /// 打开：记录 4160 B 不满足 Direct I/O 对齐，Direct 句柄会被
  /// [`recover`](Self::recover) 直接拒绝（`InvalidInput`）。`data` 用
  /// `open_direct_or_buffered` 保持整页 Direct I/O。
  pub async fn open(
    dwb: &CompioFileBackend,
    data: &CompioFileBackend,
    area_base: u64,
  ) -> io::Result<(Self, u64)> {
    let applied = Self::recover(dwb, data, area_base).await?;
    Ok((Self::new_checked(area_base)?, applied))
  }

  /// 私有构造：调用方必须已确保区首无效（新建文件或刚 recover）。
  fn new_checked(area_base: u64) -> io::Result<Self> {
    let area_end = area_bounds(area_base)?;
    Ok(Self {
      area_base,
      area_end,
      cur: area_base,
      generation: fresh_generation(),
    })
  }

  /// 追加一页到 DWB 区（不 fsync；整批写完调用 [`sync`](Self::sync)）。
  ///
  /// `final_off` 是该页最终落盘的字节偏移，恢复时原样回放。
  pub async fn write_page(
    &mut self,
    dwb: &CompioFileBackend,
    page: &[u8],
    final_off: u64,
  ) -> io::Result<()> {
    if page.len() != PAGE_SIZE {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "DWB 只接收整页（PAGE_SIZE）",
      ));
    }
    // `final_off` 是字节偏移，不是页号；整页写的落点必须页对齐
    debug_assert!(
      final_off.is_multiple_of(PAGE_SIZE as u64),
      "final_off 必须是按 PAGE_SIZE 对齐的字节偏移"
    );
    let off = self.next_slot().ok_or_else(|| {
      io::Error::new(io::ErrorKind::StorageFull, "DWB 区满：先 checkpoint 再继续")
    })?;

    let mut rec = vec![0u8; DWB_RECORD];
    rec[0..4].copy_from_slice(&DWB_MAGIC.to_le_bytes());
    rec[4..6].copy_from_slice(&DWB_VERSION.to_le_bytes());
    rec[6..8].copy_from_slice(&(PAGE_SIZE as u16).to_le_bytes());
    rec[16..24].copy_from_slice(&final_off.to_le_bytes());
    rec[24..32].copy_from_slice(&self.generation.to_le_bytes());
    rec[DWB_HEADER..].copy_from_slice(page);
    let crc = record_crc(&rec);
    rec[8..12].copy_from_slice(&crc.to_le_bytes());

    // 写成功后才推进 `cur`：失败/取消时下次仍覆写同一槽位，扫描不会遇到空洞
    dwb.write_bytes_owned(rec, off).await?;
    self.advance_slot(off);
    Ok(())
  }

  /// 下一个记录槽位的起始偏移（不推进 `cur`）；区满返回 `None`。
  ///
  /// 槽位只按 `area_base + n * DWB_RECORD` 线性推进，不自动回卷；回卷只发生在
  /// `checkpoint`（清零区首头 + `cur = area_base` + generation+1）。因此同一批内
  /// 绝不会覆盖任何已写记录，满 2016 条后必须 checkpoint。
  fn next_slot(&self) -> Option<u64> {
    let next = self.cur.checked_add(DWB_RECORD as u64)?;
    (next <= self.area_end).then_some(self.cur)
  }

  /// 记录写入成功后推进 `cur` 到下一个槽位。
  fn advance_slot(&mut self, off: u64) {
    self.cur = off + DWB_RECORD as u64;
    debug_assert!(self.cur <= self.area_end);
  }

  /// DWB 区持久化（**持久化屏障**）。**必须在写最终位置之前调用**，否则崩溃后
  /// DWB 记录与最终位置可能都没落盘。
  ///
  /// Linux = fdatasync；macOS = F_FULLFSYNC（设备缓存落盘）。缓冲 I/O 不改变
  /// 本屏障语义：返回时页缓存中本文件的脏页已全部提交给设备并确认。
  pub async fn sync(&self, dwb: &CompioFileBackend) -> io::Result<()> {
    dwb.sync_durable().await
  }

  /// 批次完成：最终位置先持久化，再使本区记录失效（清零区首头）。
  ///
  /// 失效头写**不** fsync：它只让下次启动跳过回放；丢失后旧批记录仍在，
  /// 恢复会幂等重放（最终位置已由本函数的 data 屏障持久化）。省掉一次
  /// 每批 64B 的 fdatasync，批提交热路径少一个设备屏障。
  pub async fn checkpoint(
    &mut self,
    dwb: &CompioFileBackend,
    data: &CompioFileBackend,
  ) -> io::Result<()> {
    data.sync_durable().await?;
    dwb.write_bytes_at(self.area_base, &ZERO_HEADER).await?;
    self.generation = match self.generation.checked_add(1) {
      Some(next) => next,
      None => {
        // u64 用尽：旧代号可能与新代号碰撞，必须让整个区不再有可回放记录
        invalidate_area(dwb, self.area_base, self.area_end).await?;
        fresh_generation()
      }
    };
    self.cur = self.area_base;
    Ok(())
  }

  /// 恢复：回放 `area_base` 起的有效记录（返回回放页数），随后清零区首头。
  ///
  /// 必须在接受新写入之前调用；同一进程内重复调用幂等（区首头已清零）。
  /// 清零区首头不额外 fsync：掉电丢失该写只会让下次恢复重复回放（幂等）。
  pub async fn recover(
    dwb: &CompioFileBackend,
    data: &CompioFileBackend,
    area_base: u64,
  ) -> io::Result<u64> {
    if dwb.path() == data.path() {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "DWB 与 data 必须是不同文件",
      ));
    }
    // DWB 记录 4160B 不满足 4 KiB 对齐：Linux/Windows 的 Direct 句柄会在首次
    // 非对齐 I/O 时整文件降级。这里直接拒绝，强制调用方用 open_buffered 打开
    // dwb.dat，避免「先 O_DIRECT open 再降级」的隐式路径（macOS F_NOCACHE 无
    // 对齐要求，不拒绝）。
    #[cfg(any(target_os = "linux", windows))]
    if dwb.is_direct() {
      return Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "DWB 记录 4160 B 非 4 KiB 对齐；请用 CompioFileBackend::open_buffered 打开 dwb.dat",
      ));
    }
    let area_end = area_bounds(area_base)?;
    let mut off = area_base;
    let mut generation = None;
    let mut applied = 0u64;
    while off <= area_end.saturating_sub(DWB_RECORD as u64) {
      let rec = match dwb.read_bytes_owned(off, vec![0u8; DWB_RECORD]).await {
        Ok(rec) => rec,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
        Err(e) => return Err(e),
      };
      let Some(record) = decode(&rec) else { break };
      match generation {
        Some(g) if record.generation != g => break,
        None => generation = Some(record.generation),
        Some(_) => {}
      }
      let mut page = AlignedPage([0u8; PAGE_SIZE]);
      page.0.copy_from_slice(&rec[DWB_HEADER..]);
      // IoBuf 视图必须恰好覆盖整页：写入长度取 `buf_len()`，若 impl 退化为
      // 空切片，data 会收到 0 字节写（静默丢页）。
      debug_assert_eq!(page.as_init().len(), PAGE_SIZE);
      data.write_bytes_owned(page, record.final_off).await?;
      applied += 1;
      off = off.saturating_add(DWB_RECORD as u64);
    }

    if applied > 0 {
      // 回放的页先持久化，之后才允许调用方覆写 DWB 区
      data.sync_durable().await?;
      // 失效头不 fsync：丢失只会让下次恢复重复回放（幂等）
      dwb.write_bytes_at(area_base, &ZERO_HEADER).await?;
    }
    Ok(applied)
  }
}

/// 校验并返回区结束偏移（开区间）。
fn area_bounds(area_base: u64) -> io::Result<u64> {
  if !area_base.is_multiple_of(PAGE_SIZE as u64) {
    return Err(io::Error::new(
      io::ErrorKind::InvalidInput,
      "DWB 区起点必须按 PAGE_SIZE 对齐",
    ));
  }
  area_base
    .checked_add(DWB_AREA as u64)
    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "DWB 区结束偏移溢出"))
}

/// 清零区内所有记录头并 fsync（generation 回绕时的全量失效）。
async fn invalidate_area(dwb: &CompioFileBackend, area_base: u64, area_end: u64) -> io::Result<()> {
  let mut off = area_base;
  while let Some(next) = off.checked_add(DWB_RECORD as u64) {
    if next > area_end {
      break;
    }
    dwb.write_bytes_at(off, &ZERO_HEADER).await?;
    off = next;
  }
  dwb.fdatasync().await
}

/// 随机 + 时间混合的批代号：进程重启后与旧批记录不同，避免误回放。
fn fresh_generation() -> u64 {
  let nanos = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map_or(0, |d| d.as_nanos() as u64);
  RandomState::new().hash_one(nanos) | 1
}

/// 计算记录 CRC（CRC 字段按 0 参与，编码/解码一致）
fn record_crc(rec: &[u8]) -> u32 {
  let mut hasher = Hasher::new();
  hasher.update(&rec[..8]);
  hasher.update(&[0u8; 4]);
  hasher.update(&rec[12..]);
  hasher.finalize()
}

/// 解析并校验一条记录；任何字段不合法都返回 `None`（调用方停止扫描）。
fn decode(rec: &[u8]) -> Option<Record> {
  if rec.len() != DWB_RECORD {
    return None;
  }
  let magic = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
  let version = u16::from_le_bytes([rec[4], rec[5]]);
  let len = u16::from_le_bytes([rec[6], rec[7]]);
  let crc = u32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
  if magic != DWB_MAGIC || version != DWB_VERSION || usize::from(len) != PAGE_SIZE {
    return None;
  }
  if record_crc(rec) != crc {
    return None;
  }
  let final_off = u64::from_le_bytes([
    rec[16], rec[17], rec[18], rec[19], rec[20], rec[21], rec[22], rec[23],
  ]);
  // 恢复只允许整页回放：未对齐的 final_off 会跨页写穿相邻页（若 data 后端因此
  // 降级为缓冲 I/O 更会静默破坏两页）。CRC 虽能挡随机损坏，仍按损坏记录拒绝。
  if !final_off.is_multiple_of(PAGE_SIZE as u64) {
    return None;
  }
  let generation = u64::from_le_bytes([
    rec[24], rec[25], rec[26], rec[27], rec[28], rec[29], rec[30], rec[31],
  ]);
  Some(Record {
    final_off,
    generation,
  })
}

#[cfg(test)]
mod tests {
  use std::{
    env::temp_dir,
    fs,
    path::PathBuf,
    process::id,
    sync::atomic::{AtomicU64, Ordering},
  };

  use super::*;

  static SEQ: AtomicU64 = AtomicU64::new(0);

  fn tmp_dir(tag: &str) -> io::Result<PathBuf> {
    let dir = temp_dir().join(format!(
      "ldb_io_dwb_unit_{tag}_{}_{}",
      id(),
      SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir)?;
    Ok(dir)
  }

  fn page(val: u8) -> Vec<u8> {
    vec![val; PAGE_SIZE]
  }

  /// 1e6 次槽位分配/回收：偏移恒为 `base + slot*DWB_RECORD`，不随轮次漂移。
  #[test]
  fn slot_alloc_1e6_no_drift() -> io::Result<()> {
    const SLOTS: usize = 2016; // 8 MiB / 4160，字面量期望
    assert_eq!(DWB_AREA, 8 * 1024 * 1024);
    assert_eq!(DWB_HEADER, 64);
    assert_eq!(DWB_RECORD, 4160);
    assert_eq!(DWB_RECORDS, SLOTS);

    let mut w = DwbWriter::new_checked(0)?;
    let mut counts = vec![0u32; SLOTS];
    let mut resets = 0u32;
    for _ in 0..1_000_000 {
      match w.next_slot() {
        Some(off) => {
          let rel = off - w.area_base;
          assert_eq!(rel % DWB_RECORD as u64, 0, "槽位偏移漂移");
          let idx = (rel / DWB_RECORD as u64) as usize;
          assert!(idx < SLOTS);
          counts[idx] += 1;
          w.advance_slot(off);
        }
        None => {
          // 区满：不覆盖任何有效记录，等 checkpoint 回收（此处只做纯状态模拟）
          assert_eq!(
            w.area_end - w.cur,
            DWB_AREA as u64 - SLOTS as u64 * DWB_RECORD as u64
          );
          w.cur = w.area_base;
          w.generation = w.generation.wrapping_add(1);
          resets += 1;
        }
      }
    }
    let (mut min, mut max) = (u32::MAX, 0);
    for &count in &counts {
      min = min.min(count);
      max = max.max(count);
    }
    assert!(max - min <= 1, "槽位分配不均: {min}..{max}");
    assert_eq!(resets, (1_000_000 / (SLOTS + 1)) as u32);
    Ok(())
  }

  /// generation 用尽：checkpoint 必须清零全区记录头，旧槽位不得与新代号碰撞。
  #[compio::test]
  #[ignore = "回绕路径需清零 2016 个槽位头，约 0.13s; --run-ignored 验证"]
  async fn generation_wrap_invalidates_whole_area() -> io::Result<()> {
    let dir = tmp_dir("gen_wrap")?;
    let dwb = CompioFileBackend::open_buffered(dir.join("dwb.dat")).await?;
    let data = CompioFileBackend::open_direct_or_buffered(dir.join("data.db")).await?;

    let mut w = DwbWriter::new_checked(0)?;
    w.generation = u64::MAX;
    w.write_page(&dwb, &page(0xA5), 0).await?;
    w.sync(&dwb).await?;
    w.checkpoint(&dwb, &data).await?;

    let img = fs::read(dir.join("dwb.dat"))?;
    for i in 0..DWB_RECORDS {
      let base = i * DWB_RECORD;
      assert!(
        img[base..base + 4].iter().all(|&b| b == 0),
        "回绕后槽位 {i} 的记录头未清零"
      );
    }

    // 新批（新代号）只写一页；恢复只能回放这一页
    w.write_page(&dwb, &page(0x5A), 0).await?;
    w.sync(&dwb).await?;
    assert_eq!(DwbWriter::recover(&dwb, &data, 0).await?, 1);
    let got = data.read_page_owned(0, page(0)).await?;
    assert!(got.iter().all(|&b| b == 0x5A), "新记录未回放");
    assert_eq!(DwbWriter::recover(&dwb, &data, 0).await?, 0);

    drop(data);
    drop(dwb);
    fs::remove_dir_all(&dir)?;
    Ok(())
  }
}
