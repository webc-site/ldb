//! Replayer（冷启动崩溃恢复，上游 ZLeanStore Recovery + master 崩溃协议）。
//!
//! 1. recover_from_checkpoint：装载 mapping.ckpt（PID2Offset 表）
//! 2. replay_wal_shard：扫描 WAL 分片（零填充跳 4KiB + CRC 前缀终止）
//! 3. redo_page_entry：重放页日志
//! 4. undo_active_txns：回滚未提交事务（TX_ABORT 缺失者）
//! 5. restore_pid2offset_table：映射表重建
//!
//! WAL 条目二进制格式与落盘段布局以 `ldb_wal` 冻结实现为准：
//! - 条目头 24B（`chksum@0 type@4 lsn@8 size@16`），数据类 +`pid@24 gsn@32`；
//! - 落盘段 = 段首条目对齐 + 尾部零填充到 `WAL_WRITE_ALIGN`(4096) 倍数
//!   （`ldb_wal/src/log_worker.rs:276-296` `write_log_buffer`）；
//! - 重放消费契约：`size == 0`（零填充）→ 跳到下一 4KiB 边界继续；
//!   CRC 失败 / 半条目 / 越界 → 前缀终止（崩溃残留语义）。

use std::collections::BTreeSet;

use ldb_pack::Pid2Offset;
use ldb_wal::{EntryType, LogManager, WAL_WRITE_ALIGN, entry};

use crate::error::{Error, Result};

/// 定长小端读 u64（调用点已保证长度，`copy_from_slice` 编译为同一 load 序列）
#[inline]
fn le_u64(buf: &[u8], at: usize) -> u64 {
  let mut b = [0u8; 8];
  b.copy_from_slice(&buf[at..at + 8]);
  u64::from_le_bytes(b)
}

/// 已校验的 WAL 帧（扫描阶段产物；按 `lsn` 排序后重放）
#[derive(Debug, Clone, Copy)]
struct Frame {
  /// 日志序号（逻辑序依据）
  lsn: u64,
  /// 条目类型
  ty: EntryType,
  /// 数据类条目的 pid（非数据类恒 0）
  pid: u64,
}

/// 恢复器
#[derive(Debug, Default)]
pub struct Replayer {
  /// 恢复的映射表
  pub pid2offset: Vec<Pid2Offset>,
  /// 已重放条目数
  pub redone: u64,
  /// 已回滚事务数
  pub undone: u64,
  /// 崩溃点前活跃事务（TX_START 无 TX_COMMIT/ABORT）；BTreeSet 有序 →
  /// `iter().next()` = 最早开启者（FIFO 配对，上游 WAL 内每 worker 串行）
  pub active_txns: BTreeSet<u64>,
}

impl Replayer {
  /// 构造
  pub fn new() -> Self {
    Self::default()
  }

  /// 从检查点恢复（mapping.ckpt 反序列化：[len][packed...]）
  ///
  /// 逐条 `Pid2Offset` 位域解码（49-bit 偏移 << 15 | 15-bit 压缩尺寸），
  /// 不重写位域常量（复用 `ldb_pack` 的 `COMP_SZ_BITS`/`MAX_COMP_SZ`）。
  pub async fn recover_from_checkpoint(&mut self, ckpt: &[u8]) -> Result<u64> {
    if ckpt.len() < 8 {
      return Err(Error::CheckpointCorrupt);
    }
    let n = le_u64(ckpt, 0) as usize;
    // 长度按「8 + n*8」计算，乘加用 checked 防止恶意 n 造成 usize 溢出回绕
    let need = n
      .checked_mul(8)
      .and_then(|b| b.checked_add(8))
      .ok_or(Error::CheckpointCorrupt)?;
    if ckpt.len() < need {
      return Err(Error::CheckpointCorrupt);
    }
    self.pid2offset = (0..n)
      .map(|i| {
        let packed = le_u64(ckpt, 8 + i * 8);
        let mut p = Pid2Offset::default();
        p.set(
          packed >> ldb_pack::COMP_SZ_BITS,
          (packed & u64::from(ldb_pack::MAX_COMP_SZ)) as u16,
        );
        p
      })
      .collect();
    Ok(n as u64)
  }

  /// 重放 WAL 分片（帧流；返回有效帧数）。
  ///
  /// 段布局契约（ldb_wal 冻结）：条目链 + 段尾零填充到 4KiB 倍数，段与段在
  /// 文件内相邻接。扫描规则：
  /// 1. `size == 0` → 零填充，跳到下一 4KiB 边界继续（**不**终止）；
  /// 2. `size < LOG_ENTRY_HDR` 或 `pos+size > len` → 半条目 / 截断尾，前缀终止；
  /// 3. CRC 不符（含 CR 的头部 CRC 规则）→ 前缀终止。
  ///
  /// 解析得到的帧按 `lsn` 升序重放：`wal_offset` 反向分配 + 环形回卷使段在文件内的
  /// 物理顺序不等于逻辑（lsn）顺序（`log_worker.rs:718-727`），而恢复状态依赖逻辑序
  /// ——`TxCommit` 必须配对**最早**的 `TxStart`（FIFO）。段内 lsn 唯一，稳定排序
  /// 对同序流是空操作。配对只能按顺序（元条目只有 type，无 tx_id：
  /// `leanstore.latency/src/transaction/transaction_manager.cc:48,94,148`）。
  pub async fn replay_wal_shard(&mut self, wal: &[u8], mgr: &LogManager) -> Result<u64> {
    // 扫描阶段：校验并收集帧（前缀终止于首个损坏/截断/零填充走完处）
    let mut frames: Vec<Frame> = Vec::new();
    let mut pos = 0usize;
    while pos + entry::LOG_ENTRY_HDR <= wal.len() {
      let Ok(hdr) = entry::read_hdr(&wal[pos..]) else {
        break; // 头部不可解析（含非法类型字节）：前缀终止
      };
      if hdr.size == 0 {
        // 段尾零填充（全零头 = type 0/size 0）→ 跳下一 4KiB 边界。
        // 取严格更大者（align - pos%align ∈ [1, align]）防止死循环。
        pos += WAL_WRITE_ALIGN - pos % WAL_WRITE_ALIGN;
        continue;
      }
      let size = hdr.size as usize;
      if size < entry::LOG_ENTRY_HDR || pos + size > wal.len() {
        break; // 半条目 / 越界：崩溃残留，前缀终止
      }
      let frame = &wal[pos..pos + size];
      if entry::validate_chksum(frame).is_err() {
        break; // 损坏帧即恢复终点（前缀一致性）
      }
      // 数据类帧必须装得下定长字段（pid@24 gsn@32）；完整且 CRC 通过却声明数据类
      // 而无字段 = 硬损坏，显式报错而非越界读
      let needs_pid = matches!(
        hdr.ty,
        EntryType::DataEntry
          | EntryType::FreePage
          | EntryType::ReusePage
          | EntryType::FreeExtent
          | EntryType::ReuseExtent
      );
      if needs_pid && size < entry::DATA_ENTRY_HDR {
        return Err(Error::WalCorrupt(hdr.lsn));
      }
      frames.push(Frame {
        lsn: hdr.lsn,
        ty: hdr.ty,
        // 仅数据类读 pid；其余帧不触碰字段（read_data_fields 需 >=40B）
        pid: if needs_pid {
          entry::read_data_fields(frame).0
        } else {
          0
        },
      });
      pos += size;
    }
    let n = frames.len() as u64;
    // 重放阶段：按 lsn 升序（逻辑事务序）
    frames.sort_by_key(|f| f.lsn);
    for f in frames {
      match f.ty {
        EntryType::TxStart => {
          self.active_txns.insert(f.lsn);
        }
        EntryType::TxCommit | EntryType::TxAbort => {
          // commit/abort 帧 lsn ≠ 事务起始 lsn（元条目无 tx_id）：弹出最早开启者。
          // 活跃集为空（重复 TX_COMMIT）则忽略。
          if let Some(first) = self.active_txns.iter().next().copied() {
            self.active_txns.remove(&first);
          }
        }
        EntryType::DataEntry | EntryType::FreePage | EntryType::ReusePage => {
          // 数据类头布局等价：pid@24。FreePage(6) 按 latency 版无 lp_size
          // （`log_entry.h:45-48`），单页语义 → redo 只建 pid 槽，不消费 lp_size。
          self.redo_page_entry(f.lsn, f.pid).await?;
        }
        EntryType::FreeExtent | EntryType::ReuseExtent => {
          // 48B 布局：start_pid@24 lp_size@32 gsn@40。redo 侧只需起始 pid 建槽，
          // 不依赖 lp_size（extent 释放接线见 CROSS）。
          self.redo_page_entry(f.lsn, f.pid).await?;
        }
        EntryType::PageImg => {
          // 检查点整页镜像：映射基线由 mapping.ckpt 装载，段内镜像重放期忽略。
        }
        EntryType::CarriageReturn => {
          // 环回填充（size>0，指向环尾）：头部 CRC 已校验，作为填充跳过。
        }
      }
    }
    // LogManager 保留入参：段/分片归属校验与 GSN 水位重建需在 L5 接线
    // （见 CROSS「恢复侧消费契约」），当前重放只读条目流。
    let _ = mgr;
    Ok(n)
  }

  /// 重放页条目（redo：为 pid 建立映射槽，等待空间层/检查点回填 offset）。
  ///
  /// `DataEntry`/`FreePage`/`ReusePage`/`FreeExtent` 仅携带 pid（+gsn），
  /// 不含物理 offset —— 映射的 offset/comp_sz 由检查点基线或空间层回填
  /// （`restore_pid2offset_table`）。因此 redo 幂等：重复重放同一 pid 不改变状态。
  pub async fn redo_page_entry(&mut self, lsn: u64, pid: u64) -> Result<()> {
    let _ = lsn;
    let idx = pid as usize;
    if idx >= self.pid2offset.len() {
      self.pid2offset.resize(idx + 1, Pid2Offset::default());
    }
    self.redone += 1;
    Ok(())
  }

  /// 回滚未提交事务（恢复终点：active 集合即需回滚者）
  pub async fn undo_active_txns(&mut self) -> Result<u64> {
    let n = self.active_txns.len() as u64;
    self.undone += n;
    self.active_txns.clear();
    Ok(n)
  }

  /// 映射表重建（对账：以重放结果为主，检查点基线补空槽）。
  ///
  /// 重放得到的槽（comp_sz != 0）保留；空槽（comp_sz == 0）用检查点基线填充。
  /// 上游映射表幸存页在恢复后仍指向检查点记录的物理位置（未重写页），
  /// 故「非空优先」与上游一致。
  pub async fn restore_pid2offset_table(&mut self, ckpt_base: Vec<Pid2Offset>) -> Result<usize> {
    if ckpt_base.len() > self.pid2offset.len() {
      self
        .pid2offset
        .resize(ckpt_base.len(), Pid2Offset::default());
    }
    for (i, base) in ckpt_base.iter().enumerate() {
      if self.pid2offset[i].get_comp_sz() == 0 && base.get_comp_sz() != 0 {
        self.pid2offset[i] = *base;
      }
    }
    Ok(self.pid2offset.len())
  }
}
