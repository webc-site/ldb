//! ldb_pack 位域回归（账本 target `PID2Offset::{set,get_offset,get_comp_sz}`）。
//!
//! 上游 `space_manager.h:47-70`；位宽修正见 todo.md §9 C-35。

use std::{
  mem::{size_of, transmute},
  panic::catch_unwind,
};

use ldb_pack::{COMP_SZ_BITS, MAX_COMP_SZ, MAX_OFFSET, OFFSET_BITS, Pid2Offset, UNCOMPRESSED};

/// 上游 `PID2Offset::set/get_offset/get_comp_sz` 的逐字转写（`space_manager.h:50-69`）：
/// 50-bit 偏移掩码 + 15-bit 尺寸；`offset << 15` 在 offset ≥ 2^49 时静默丢最高位。
fn upstream_pack(offset: u64, comp_size: u16) -> u64 {
  let offset = offset & ((1u64 << 50) - 1);
  let comp_size = u64::from(comp_size) & 0x7FFF;
  (offset << 15) | comp_size
}

/// 读取私有位域（`#[repr(transparent)]`，与 u64 同布局）
fn raw_of(p: Pid2Offset) -> u64 {
  // SAFETY: Pid2Offset 是 #[repr(transparent)] 的 u64 包装，尺寸/对齐与 u64 相同。
  unsafe { transmute(p) }
}

/// 确定性 xorshift64（差分对拍的输入源，不引第三方依赖）
struct Rng(u64);

impl Rng {
  fn next(&mut self) -> u64 {
    self.0 ^= self.0 << 13;
    self.0 ^= self.0 >> 7;
    self.0 ^= self.0 << 17;
    self.0
  }
}

/// 字面量位布局：offset 在 bit15..63，comp_sz 在 bit0..14
#[test]
fn literal_bit_layout_49_15() {
  const _: () = assert!(size_of::<Pid2Offset>() == 8);

  // 域常量字面量（E3：期望值独立于被测常量）
  assert_eq!(OFFSET_BITS, 49);
  assert_eq!(COMP_SZ_BITS, 15);
  assert_eq!(MAX_OFFSET, 0x0001_FFFF_FFFF_FFFF);
  assert_eq!(MAX_COMP_SZ, 0x7FFF);
  // 域上限编码后恰好填满 64 位（49+15=64 的数学证明在编译期 const_assert）
  assert_eq!(
    (MAX_OFFSET << COMP_SZ_BITS) | u64::from(MAX_COMP_SZ),
    u64::MAX
  );

  let mut p = Pid2Offset::default();
  // 2^48 << 15 | 0x6BEF = 0x8000_0000_0000_6BEF（字面量，独立于 OFFSET_BITS/COMP_SZ_BITS）
  p.set(0x0001_0000_0000_0000, 0x6BEF);
  // SAFETY: Pid2Offset 是 #[repr(transparent)] 的 u64 包装，尺寸/对齐与 u64 相同。
  let raw: u64 = unsafe { transmute(p) };
  assert_eq!(raw, 0x8000_0000_0000_6BEF);
  assert_eq!(p.get_offset(), 0x0001_0000_0000_0000);
  assert_eq!(p.get_comp_sz(), 0x6BEF);

  // 单字段定位：offset=1 → bit15；comp_sz=1 → bit0
  p.set(1, 0);
  // SAFETY: 同上：Pid2Offset 是 #[repr(transparent)] 的 u64 包装，尺寸/对齐与 u64 相同。
  assert_eq!(
    unsafe { transmute::<Pid2Offset, u64>(p) },
    0x0000_0000_0000_8000
  );
  p.set(0, 1);
  // SAFETY: 同上：Pid2Offset 是 #[repr(transparent)] 的 u64 包装，尺寸/对齐与 u64 相同。
  assert_eq!(unsafe { transmute::<Pid2Offset, u64>(p) }, 1);

  // 两字段同时取域上限 → 64 位全 1（49+15 恰好填满，无溢出位）
  p.set(0x0001_FFFF_FFFF_FFFF, 0x7FFF);
  // SAFETY: 同上：Pid2Offset 是 #[repr(transparent)] 的 u64 包装，尺寸/对齐与 u64 相同。
  assert_eq!(unsafe { transmute::<Pid2Offset, u64>(p) }, u64::MAX);
  assert_eq!(p.get_offset(), 0x0001_FFFF_FFFF_FFFF);
  assert_eq!(p.get_comp_sz(), 0x7FFF);

  // 单字段满宽不污染另一字段：偏移满宽 → 低 15 位必须为 0
  p.set(0x0001_FFFF_FFFF_FFFF, 0);
  assert_eq!(raw_of(p), 0xFFFF_FFFF_FFFF_8000);
  // 尺寸满宽 → 高 49 位必须为 0
  p.set(0, 0x7FFF);
  assert_eq!(raw_of(p), 0x0000_0000_0000_7FFF);

  // A2 差分对拍（与上面的字面量布局同属一个行为面，合并为一条用例）：
  // 20 万例编码/解码逐位对照上游 50-bit 参考实现。偏移取 `0..=2^49-1`
  // （两实现语义相同的域；2^49 处上游丢位由 `offset_boundary_2_pow_49` 单列证据）；
  // 解码侧另用任意 u64 位型，覆盖低位垃圾/高位全 1，校验 getter 的 15-bit 掩码。
  let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
  let mut mismatches = 0u32;
  for i in 0..200_000u32 {
    let offset = match i % 4 {
      0 => 0,
      1 => MAX_OFFSET,
      2 => rng.next() & MAX_OFFSET,
      _ => rng.next() % 0x0010_0000, // 小偏移密集
    };
    let comp = match i % 4 {
      0 => 0,
      1 => MAX_COMP_SZ,
      2 => UNCOMPRESSED,
      _ => (rng.next() & 0x7FFF) as u16,
    };

    p.set(offset, comp);
    if raw_of(p) != upstream_pack(offset, comp)
      || p.get_offset() != offset
      || p.get_comp_sz() != comp
    {
      mismatches += 1;
    }

    // 任意位型解码：与上游 `(packed >> 15) & ((1<<50)-1)` / `packed & 0x7FFF` 一致
    let raw = rng.next();
    // SAFETY: Pid2Offset 是 #[repr(transparent)] 的 u64 包装，任意 u64 位型都是合法值。
    let q: Pid2Offset = unsafe { transmute(raw) };
    if q.get_offset() != (raw >> 15) & ((1u64 << 50) - 1)
      || q.get_comp_sz() != (raw & 0x7FFF) as u16
    {
      mismatches += 1;
    }
  }
  assert_eq!(mismatches, 0, "MISMATCH>0：与上游参考实现不一致");
}

/// 49 位边界：2^49-1 可编码，2^49 必须 panic（上游 50 位静默截断，C-35）
#[test]
fn offset_boundary_2_pow_49() {
  let mut p = Pid2Offset::default();
  p.set(0x0001_FFFF_FFFF_FFFF, 0);
  assert_eq!(p.get_offset(), 0x0001_FFFF_FFFF_FFFF);

  // C-35 证据：同一输入走上游 50-bit 掩码 + <<15 后，偏移被截成 0
  assert_eq!(upstream_pack(0x0002_0000_0000_0000, 0) >> 15, 0);

  assert!(
    catch_unwind(|| {
      let mut q = Pid2Offset::default();
      q.set(0x0002_0000_0000_0000, 0);
    })
    .is_err(),
    "offset = 2^49 必须 panic"
  );
}

/// 15 位边界：32767 可编码，32768 必须 panic
#[test]
fn comp_sz_boundary_15_bits() {
  assert_eq!(MAX_COMP_SZ, 0x7FFF);
  let mut p = Pid2Offset::default();
  p.set(0, 0x7FFF);
  assert_eq!(p.get_comp_sz(), 0x7FFF);

  assert!(
    catch_unwind(|| {
      let mut q = Pid2Offset::default();
      q.set(0, 0x8000);
    })
    .is_err(),
    "comp_size = 32768 必须 panic"
  );
}

/// C-19 哨兵：未压缩整页尺寸必须可编码，且等于页大小
#[test]
fn uncompressed_sentinel() {
  assert_eq!(UNCOMPRESSED, 4096);
  assert_eq!(UNCOMPRESSED, ldb_type::PAGE_SIZE as u16);

  let mut p = Pid2Offset::default();
  p.set(4096 * 7, UNCOMPRESSED);
  assert_eq!(p.get_offset(), 4096 * 7);
  assert_eq!(p.get_comp_sz(), UNCOMPRESSED);
}

/// 默认值 = 未初始化槽位（offset 0 / comp_sz 0），读回不 panic
#[test]
fn default_is_all_zero() {
  let p = Pid2Offset::default();
  assert_eq!(p.get_offset(), 0);
  assert_eq!(p.get_comp_sz(), 0);
}
