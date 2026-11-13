//! WAL 记录编解码（上游 `leanstore.latency/src/include/recovery/log_entry.h`）。
//!
//! 头部二进制布局（与上游 C 布局等宽）：`chksum@0(u32) type@4(u8) lsn@8(u64)
//! size@16(u64)`，共 24 字节。数据类条目头部追加 `pid(u64) gsn(u64)`。
//! 以字节游标读写（不用结构体强转，杜绝对齐/端序隐患）。

use crc32fast::Hasher;

use crate::error::{Error, Result};

/// LogEntry 头部大小
pub const LOG_ENTRY_HDR: usize = 24;
/// DataEntry / FreePageEntry 头部（+ pid + gsn）
pub const DATA_ENTRY_HDR: usize = LOG_ENTRY_HDR + 16;
/// FreeExtentEntry 头部（+ start_pid + lp_size + gsn）
pub const FREE_EXTENT_HDR: usize = LOG_ENTRY_HDR + 24;
/// CR 填充条目大小（上游 CR_ENTRY_SIZE = sizeof(LogMetaEntry)）
pub const CR_ENTRY_SIZE: usize = LOG_ENTRY_HDR;
/// PageImgEntry 头部（+ pid + gsn）
pub const PAGE_IMG_HDR: usize = LOG_ENTRY_HDR + 16;

/// 条目类型编码（上游 `LogEntry::Type`）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryType {
  /// 环回填充
  CarriageReturn = 0,
  TxStart = 1,
  TxCommit = 2,
  TxAbort = 3,
  /// 页数据日志
  DataEntry = 4,
  FreePage = 6,
  ReusePage = 7,
  FreeExtent = 8,
  ReuseExtent = 9,
  /// 整页镜像（checkpoint 用）
  PageImg = 10,
}

impl EntryType {
  /// 编码
  #[inline]
  pub fn to_u8(self) -> u8 {
    self as u8
  }

  /// 解码（非法值返回 Err）
  #[inline]
  pub fn from_u8(v: u8) -> Result<Self> {
    Ok(match v {
      0 => Self::CarriageReturn,
      1 => Self::TxStart,
      2 => Self::TxCommit,
      3 => Self::TxAbort,
      4 => Self::DataEntry,
      6 => Self::FreePage,
      7 => Self::ReusePage,
      8 => Self::FreeExtent,
      9 => Self::ReuseExtent,
      10 => Self::PageImg,
      _ => return Err(Error::CorruptLog(format!("非法日志类型: {v}"))),
    })
  }
}

/// 头部字段偏移（二进制格式契约；恢复/对拍侧共用）
pub mod off {
  pub const CHKSUM: usize = 0;
  pub const TYPE: usize = 4;
  pub const LSN: usize = 8;
  pub const SIZE: usize = 16;
  pub const PID: usize = 24;
  pub const GSN: usize = 32;
  pub const LP_SIZE: usize = 32;
}

/// 写头部公共字段（chksum 置 0，由 compute_chksum 回填）。
/// type 与 lsn 之间的 3 字节 padding 必须显式清零：环形缓冲槽位复用时残留旧字节，
/// 而 CRC 覆盖 [4, size) —— 不清零则同一逻辑条目因槽位历史产生不同校验和，
/// 破坏重放/对比的幂等性。
/// （24B 定长头不做“一次 memcpy”强转：逐字段写已被内联为 4 条 store，
/// 整块 memcpy 无收益且引入对齐/端序 unsafe）
pub fn write_hdr(entry: &mut [u8], ty: EntryType, lsn: u64, size: u64) {
  entry[off::CHKSUM..off::CHKSUM + 4].copy_from_slice(&0u32.to_le_bytes());
  entry[off::TYPE] = ty.to_u8();
  entry[off::TYPE + 1..off::LSN].fill(0);
  entry[off::LSN..off::LSN + 8].copy_from_slice(&lsn.to_le_bytes());
  entry[off::SIZE..off::SIZE + 8].copy_from_slice(&size.to_le_bytes());
}

/// 写 DataEntry 头
pub fn write_data_hdr(entry: &mut [u8], lsn: u64, size: u64, pid: u64, gsn: u64) {
  write_hdr(entry, EntryType::DataEntry, lsn, size);
  entry[off::PID..off::PID + 8].copy_from_slice(&pid.to_le_bytes());
  entry[off::GSN..off::GSN + 8].copy_from_slice(&gsn.to_le_bytes());
}

/// 读头部
#[derive(Debug, Clone, Copy)]
pub struct EntryHdr {
  /// 类型
  pub ty: EntryType,
  /// 日志序号
  pub lsn: u64,
  /// 条目总长（含头）
  pub size: u64,
}

/// 定长小端读（切片长度由调用点静态保证，`copy_from_slice` 编译为同一 load 序列）
#[inline]
fn le_u64(entry: &[u8], at: usize) -> u64 {
  let mut b = [0u8; 8];
  b.copy_from_slice(&entry[at..at + 8]);
  u64::from_le_bytes(b)
}

/// 定长小端读（4B）
#[inline]
fn le_u32(entry: &[u8], at: usize) -> u32 {
  let mut b = [0u8; 4];
  b.copy_from_slice(&entry[at..at + 4]);
  u32::from_le_bytes(b)
}

/// 读头部（size 至少 LOG_ENTRY_HDR）
pub fn read_hdr(entry: &[u8]) -> Result<EntryHdr> {
  if entry.len() < LOG_ENTRY_HDR {
    return Err(Error::CorruptLog("条目头不完整".into()));
  }
  Ok(EntryHdr {
    ty: EntryType::from_u8(entry[off::TYPE])?,
    lsn: le_u64(entry, off::LSN),
    size: le_u64(entry, off::SIZE),
  })
}

/// 读 DataEntry 的 pid / gsn
pub fn read_data_fields(entry: &[u8]) -> (u64, u64) {
  (le_u64(entry, off::PID), le_u64(entry, off::GSN))
}

/// 计算 CRC 校验码（覆盖 chksum 字段之后的全部字节，上游 ComputeChksum：
/// `leanstore.latency/src/include/recovery/log_entry.h:31`
/// `ComputeCRC(this + 4, size - 4)`，算法为 IEEE CRC-32）。
///
/// CR 填充条目例外：仅覆盖头部 `[4, LOG_ENTRY_HDR)` —— 其 size 指向环尾，
/// 尾部的环残留字节会被后续环回周期的条目覆写，覆盖 `[4, size)` 将使校验码
/// 不稳定（上游仅在 `FLAGS_wal_debug` 下给 CR 计 CRC，`log_buffer.cc:66`，
/// 稳态无 CRC；本 crate 恒校验，故取头部覆盖这一稳定子集）。
///
/// `size` 来自不可信字节流（恢复路径逐帧校验），越界一律夹取到 `[4, len]`
/// —— 完好条目不受影响，损坏条目得到稳定的 Err 而非 panic。
///
/// 不做 thread_local hasher 复用：crc32fast 的 `finalize(self)` 按值消费所有权，
/// 复用模式下每次仍须 `mem::replace` 重建实例，反而叠加 thread_local 访问与
/// RefCell 借用开销；而 `Hasher::new()` 首次做 CPU 特性检测后仅剩零堆分配的
/// 栈构造（状态体 ~几十字节），每条目一次的成本已是下界
pub fn compute_chksum(entry: &[u8]) -> u32 {
  let ty = entry[off::TYPE];
  let end = if ty == EntryType::CarriageReturn.to_u8() {
    LOG_ENTRY_HDR
  } else {
    let size = le_u64(entry, off::SIZE) as usize;
    size.clamp(LOG_ENTRY_HDR, entry.len())
  };
  let mut h = Hasher::new();
  h.update(&entry[4..end]);
  h.finalize()
}

/// 回填校验码
pub fn seal_chksum(entry: &mut [u8]) -> u32 {
  let c = compute_chksum(entry);
  entry[off::CHKSUM..off::CHKSUM + 4].copy_from_slice(&c.to_le_bytes());
  c
}

/// 校验完整性（短于头部或 CRC 不符均返回 Err）
pub fn validate_chksum(entry: &[u8]) -> Result<()> {
  if entry.len() < LOG_ENTRY_HDR {
    return Err(Error::CorruptLog("条目头不完整".into()));
  }
  let stored = le_u32(entry, off::CHKSUM);
  if stored == compute_chksum(entry) {
    Ok(())
  } else {
    Err(Error::CorruptLog("CRC 校验失败".into()))
  }
}
