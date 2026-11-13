//! 定长记录原地更新 Delta（上游 `leanstore.latency/src/include/common/delta.h`）。
//!
//! 线上布局与 C++ `FixedSizeDelta` 逐字节一致：
//! `[count: u8][pad: u8][size: u16 LE][Slot{offset,len} × count][payload...]`。
//! clang++ 对真实头文件输出：`sizeof(FixedSizeDelta) == 4`、
//! `offsetof(count) == 0`、`offsetof(size) == 2`、`offsetof(slots) == 4`、
//! `MetadataSize(n) == 4 + 4n`（`FixedSizeDelta` 含柔性数组成员）。
//!
//! `size` 语义（上游构造函数 + `DELTA_FILL_SLOT`）：`4 + 4*count + Σlength`。
//! 上游 `UpdateDeltaPayload`（`delta.h:26-32`）的 `off` 在循环内不递增，仅在
//! `count == 1` 时正确；本实现按「槽位顺序连续写入」的既定语义修复该缺陷，
//! 并在 `count == 1` 时与上游真实输出逐字节对拍（见 `tests/delta_test.rs`）。
//!
//! 不变量：
//! 1. `count <= 255`（u8 编码上限），元数据长 `4 + 4*count`；
//! 2. `size == 4 + 4*count + Σlength` 且 `size <= 65535`（u16 编码上限）；
//! 3. `size <= delta.len()`，载荷区自 `4 + 4*count` 起按槽位顺序紧排；
//! 4. 每个槽位 `offset + length <= tuple_payload.len()`（0 长槽位允许
//!    `offset == tuple_payload.len()`）；
//! 5. 槽位可重叠（上游不禁止），应用顺序为槽位表顺序。

use std::mem::{offset_of, size_of};

use static_assertions::const_assert_eq;

use crate::error::{Error, Result};

/// Delta 槽位描述（上游 `FixedSizeDelta::Slot`）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct DeltaSlot {
  /// 记录内偏移
  pub offset: u16,
  /// 拷贝长度
  pub length: u16,
}

/// 头部字节数（上游 `sizeof(FixedSizeDelta)`）
pub const META_FIX: usize = 4;

/// `size` 字段在 delta 内的偏移（上游 `offsetof(FixedSizeDelta, size)`）
const SIZE_OFF: usize = 2;

const_assert_eq!(size_of::<DeltaSlot>(), 4);
const_assert_eq!(offset_of!(DeltaSlot, offset), 0);
const_assert_eq!(offset_of!(DeltaSlot, length), 2);
const_assert_eq!(META_FIX, 4);
const_assert_eq!(SIZE_OFF, 2);

/// 元数据总长（含槽位表；上游 `FixedSizeDelta::MetadataSize`）。
///
/// `count` 为 u8，最大 `4 + 255*4 = 1024`，故 `as u16` 不会截断。
#[inline]
pub fn metadata_size(count: u8) -> u16 {
  (META_FIX + usize::from(count) * size_of::<DeltaSlot>()) as u16
}

/// 把 `tuple_payload` 的各槽位数据按序写入 delta 缓冲（上游 `UpdateDeltaPayload`）。
///
/// `delta` 布局由 `build_delta_header` 构造，必须是 `size` 字节的镜像。
/// 与上游一致，`size` 字段不参与写入决策（以 count/槽位表为准）。
/// 返回写入的载荷字节数（`Σlength`）。缓冲区不足、槽位越过 tuple payload
/// 时返回 `Err`，绝不 panic。
pub fn update_delta_payload(delta: &mut [u8], tuple_payload: &[u8]) -> Result<usize> {
  let buf_len = delta.len();
  let Some(&count) = delta.first() else {
    return Err(Error::DeltaBufferTooSmall { need: 1, actual: 0 });
  };
  let meta = usize::from(metadata_size(count));
  if buf_len < meta {
    return Err(Error::DeltaBufferTooSmall {
      need: meta,
      actual: buf_len,
    });
  }

  let mut written = meta;
  for idx in 0..usize::from(count) {
    // 边界：slot 表区间已由 `buf_len >= meta` 保证在界内。
    let s = META_FIX + idx * size_of::<DeltaSlot>();
    let offset = u16::from_le_bytes([delta[s], delta[s + 1]]);
    let length = u16::from_le_bytes([delta[s + 2], delta[s + 3]]);
    let start = usize::from(offset);
    let end = start + usize::from(length);
    let Some(seg) = tuple_payload.get(start..end) else {
      return Err(Error::DeltaSlotOutOfBounds {
        offset,
        length,
        payload_len: tuple_payload.len(),
      });
    };
    let write_end = written + seg.len();
    let Some(dst) = delta.get_mut(written..write_end) else {
      return Err(Error::DeltaBufferTooSmall {
        need: write_end,
        actual: buf_len,
      });
    };
    dst.copy_from_slice(seg);
    written = write_end;
  }
  Ok(written - meta)
}

/// 构造 delta 头 + 槽位表（对应上游 `FixedSizeDelta` 构造函数与
/// `DELTA_FILL_SLOT` 宏族）。
///
/// `slots`: `(offset, length)` 列表；返回总长 `size`。
/// 槽位数超过 255、`size` 超过 u16、或 `delta` 容纳不下总长时返回 `Err`。
pub fn build_delta_header(delta: &mut [u8], slots: &[(u16, u16)]) -> Result<u16> {
  let count = u8::try_from(slots.len()).map_err(|_| Error::DeltaTooManySlots(slots.len()))?;
  let meta = usize::from(metadata_size(count));
  let payload: usize = slots.iter().map(|&(_, len)| usize::from(len)).sum();
  let total = meta + payload;
  let size = u16::try_from(total).map_err(|_| Error::DeltaSizeOverflow(total))?;
  let buf_len = delta.len();
  let Some(buf) = delta.get_mut(..total) else {
    return Err(Error::DeltaBufferTooSmall {
      need: total,
      actual: buf_len,
    });
  };

  buf[0] = count;
  // 上游该字节是隐式对齐填充（placement-new 不初始化）；显式清零保证
  // 落盘/WAL 字节映像确定，读方（按 count/size/slots 解析）忽略之。
  buf[1] = 0;
  buf[SIZE_OFF..SIZE_OFF + 2].copy_from_slice(&size.to_le_bytes());
  for (idx, &(offset, length)) in slots.iter().enumerate() {
    let s = META_FIX + idx * size_of::<DeltaSlot>();
    buf[s..s + 2].copy_from_slice(&offset.to_le_bytes());
    buf[s + 2..s + 4].copy_from_slice(&length.to_le_bytes());
  }
  Ok(size)
}
