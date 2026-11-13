//! FixedSizeDelta 字节布局、越界拒绝与影子模型差分（上游
//! `leanstore.latency/src/include/common/delta.h`）。
//!
//! 期望字节来自 clang++ 对真实上游头文件的运行结果：
//! count=1、slot=(3,4)、payload=[0..15] 时
//! `size=12`、`bytes=[1,0,12,0,3,0,4,0,3,4,5,6]`。

use std::{
  array::from_fn,
  mem::{align_of, size_of},
};

use ldb_page::{
  DeltaSlot, Error, META_FIX, build_delta_header, metadata_size, update_delta_payload,
};

fn unwrap_ok<T>(r: ldb_page::Result<T>) -> T {
  match r {
    Ok(v) => v,
    Err(e) => panic!("unexpected delta error: {e}"),
  }
}

#[test]
fn delta_types_match_upstream_layout() {
  assert_eq!(META_FIX, 4);
  assert_eq!(size_of::<DeltaSlot>(), 4);
  assert_eq!(align_of::<DeltaSlot>(), 2);
  assert_eq!(metadata_size(0), 4);
  assert_eq!(metadata_size(1), 8);
  assert_eq!(metadata_size(2), 12);
  assert_eq!(metadata_size(255), 1024);
}

#[test]
fn delta_wire_layout_count1_matches_cpp_multi_slot_is_sequential() {
  // count=1：与 C++ `FixedSizeDelta::UpdateDeltaPayload` 逐字节一致
  let mut buf = [0u8; 64];
  let size = unwrap_ok(build_delta_header(&mut buf, &[(3, 4)]));
  assert_eq!(size, 12);
  // [count][pad=0][size=12 LE][offset=3][length=4]
  assert_eq!(&buf[..12], &[1, 0, 12, 0, 3, 0, 4, 0, 0, 0, 0, 0]);

  let payload: [u8; 16] = from_fn(|i| i as u8);
  let written = unwrap_ok(update_delta_payload(&mut buf, &payload));
  assert_eq!(written, 4);
  assert_eq!(&buf[..12], &[1, 0, 12, 0, 3, 0, 4, 0, 3, 4, 5, 6]);

  // count=2：槽位表顺序 + 载荷紧排
  let mut two = [0xAAu8; 64];
  let size = unwrap_ok(build_delta_header(&mut two, &[(0, 4), (8, 2)]));
  assert_eq!(size, 18);
  assert_eq!(&two[..12], &[2, 0, 18, 0, 0, 0, 4, 0, 8, 0, 2, 0]);
  let written = unwrap_ok(update_delta_payload(&mut two, b"AAAABBBBCCDD"));
  assert_eq!(written, 6);
  assert_eq!(&two[12..18], b"AAAACC");

  // count=3：与 C++ 探针同槽位（独立编译上游 delta.h 输出）
  // 上游 `UpdateDeltaPayload` 的 `off` 不递增（count>1 全部写同一偏移），
  // 本实现按既定语义顺序紧排；size/槽位表与上游构造函数一致。
  let mut three = [0u8; 64];
  let size = unwrap_ok(build_delta_header(&mut three, &[(0, 4), (8, 2), (10, 3)]));
  assert_eq!(size, 25);
  assert_eq!(
    &three[..16],
    &[3, 0, 25, 0, 0, 0, 4, 0, 8, 0, 2, 0, 10, 0, 3, 0]
  );
  let payload: [u8; 16] = from_fn(|i| i as u8);
  let written = unwrap_ok(update_delta_payload(&mut three, &payload));
  assert_eq!(written, 9);
  assert_eq!(&three[16..25], &[0, 1, 2, 3, 8, 9, 10, 11, 12]);

  // 0 长槽位：offset == payload.len() 合法
  let mut zero = [0u8; 16];
  let size = unwrap_ok(build_delta_header(&mut zero, &[(3, 0)]));
  assert_eq!(size, 8);
  assert_eq!(unwrap_ok(update_delta_payload(&mut zero, b"abc")), 0);

  // 预填充无关性：0xAA/0x55 两种栈垃圾下 build+update 的 `size` 字节
  // （含 count 后的对齐填充与全部载荷区）必须逐字节相同。
  let slots = [(0u16, 4u16), (8, 2), (10, 3)];
  let mut a = [0xAAu8; 64];
  let mut b = [0x55u8; 64];
  let size_a = unwrap_ok(build_delta_header(&mut a, &slots));
  let size_b = unwrap_ok(build_delta_header(&mut b, &slots));
  assert_eq!(size_a, size_b);
  unwrap_ok(update_delta_payload(&mut a, &payload));
  unwrap_ok(update_delta_payload(&mut b, &payload));
  let end = usize::from(size_a);
  assert_eq!(&a[..end], &b[..end], "delta 线上字节必须确定");
  assert_eq!(a[1], 0, "count 后的对齐填充字节必须为 0");
}

#[test]
fn delta_slot_count_boundaries() {
  let mut empty = [0xFFu8; 1024];
  assert_eq!(unwrap_ok(build_delta_header(&mut empty, &[])), 4);
  assert_eq!(&empty[..4], &[0, 0, 4, 0]);
  assert_eq!(unwrap_ok(update_delta_payload(&mut empty, b"ignored")), 0);

  // count=255（u8 上限）+ 全 0 长度
  let slots = [(0u16, 0u16); 255];
  let mut max = [0u8; 1024];
  let size = unwrap_ok(build_delta_header(&mut max, &slots));
  assert_eq!(size, 1024);
  assert_eq!(max[0], 255);
  assert_eq!(u16::from_le_bytes([max[2], max[3]]), 1024);
  assert_eq!(unwrap_ok(update_delta_payload(&mut max, b"")), 0);

  // 256 槽超过 u8 编码上限
  let too_many = [(0u16, 0u16); 256];
  let mut big = [0u8; 2048];
  assert!(matches!(
    build_delta_header(&mut big, &too_many),
    Err(Error::DeltaTooManySlots(256))
  ));

  // Σlength 使 size 超过 u16（4 + 4*255 + 255*300 = 77524）
  let overflow = [(0u16, 300u16); 255];
  assert!(matches!(
    build_delta_header(&mut big, &overflow),
    Err(Error::DeltaSizeOverflow(77524))
  ));

  // 缓冲容纳不下 size（4 + 4 + 4 = 12 > 11）
  let mut small = [0u8; 11];
  assert!(matches!(
    build_delta_header(&mut small, &[(0, 4)]),
    Err(Error::DeltaBufferTooSmall {
      need: 12,
      actual: 11
    })
  ));
}

#[test]
fn delta_rejects_malformed_input_without_panic() {
  let mut empty: [u8; 0] = [];
  assert!(matches!(
    update_delta_payload(&mut empty, b""),
    Err(Error::DeltaBufferTooSmall { need: 1, actual: 0 })
  ));

  // count=1 但缓冲不足 4+4 元数据
  let mut short = [1u8, 0, 8, 0];
  assert!(matches!(
    update_delta_payload(&mut short, b""),
    Err(Error::DeltaBufferTooSmall { need: 8, actual: 4 })
  ));

  // count=255 但缓冲只有 8 字节
  let mut huge_count = [255u8, 0, 0, 0, 1, 0, 1, 0];
  assert!(matches!(
    update_delta_payload(&mut huge_count, b""),
    Err(Error::DeltaBufferTooSmall {
      need: 1024,
      actual: 8
    })
  ));

  // 槽位越过 tuple payload
  let mut oob = [0u8; 108];
  let _ = unwrap_ok(build_delta_header(&mut oob, &[(0, 100)]));
  assert!(matches!(
    update_delta_payload(&mut oob, &[0u8; 4]),
    Err(Error::DeltaSlotOutOfBounds {
      offset: 0,
      length: 100,
      payload_len: 4
    })
  ));

  // 头部声明 count=1/slot(0,4) 但载荷区只有 8 字节
  let mut trunc = [0u8; 8];
  trunc[0] = 1;
  trunc[2..4].copy_from_slice(&12u16.to_le_bytes());
  trunc[4..6].copy_from_slice(&0u16.to_le_bytes());
  trunc[6..8].copy_from_slice(&4u16.to_le_bytes());
  assert!(matches!(
    update_delta_payload(&mut trunc, b"abcd"),
    Err(Error::DeltaBufferTooSmall {
      need: 12,
      actual: 8
    })
  ));

  // 任意字节流（含伪造 count/size/slot 表）只能返回 Ok/Err，不得 panic
  let mut state = 0x1234_5678_9ABC_DEF0_u64;
  let mut buf = [0u8; 64];
  let payload = [0u8; 32];
  for _ in 0..10_000 {
    for b in &mut buf {
      state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
      *b = (state >> 24) as u8;
    }
    let payload_len = ((state >> 8) & 31) as usize;
    let _ = update_delta_payload(&mut buf, &payload[..payload_len]);
  }
}

/// 影子模型差分：随机 tuple / 槽位表 → build+update 后逐字节比对，
/// 并把 delta 段按槽位回填到 tuple 副本验证可逆。
#[test]
fn delta_model_differential_100k() {
  let mut state = 0xDEAD_BEEF_CAFE_F00D_u64;
  let mut rng = move || {
    state = state
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    state
  };

  let mut tuple = [0u8; 64];
  let mut delta = [0u8; 256];
  let mut expected = [0u8; 128];

  for _ in 0..100_000 {
    let tlen = (rng() % 65) as usize;
    let fill = rng();
    for (i, b) in tuple[..tlen].iter_mut().enumerate() {
      *b = (fill >> ((i & 7) * 8)) as u8;
    }

    let n = (rng() % 5) as usize;
    let mut slots = [(0u16, 0u16); 8];
    for slot in &mut slots[..n] {
      let offset = (rng() % (tlen as u64 + 1)) as usize;
      let max_len = tlen - offset;
      let length = (rng() % 9) as usize;
      *slot = (offset as u16, length.min(max_len) as u16);
    }

    let size = unwrap_ok(build_delta_header(&mut delta, &slots[..n]));
    let meta = META_FIX + n * size_of::<DeltaSlot>();
    let total_len: usize = slots[..n].iter().map(|&(_, l)| usize::from(l)).sum();
    assert_eq!(usize::from(size), meta + total_len);
    assert_eq!(delta[1], 0, "对齐填充必须为 0");
    assert_eq!(u16::from_le_bytes([delta[2], delta[3]]), size);

    let written = unwrap_ok(update_delta_payload(&mut delta, &tuple[..tlen]));
    assert_eq!(written, total_len);

    let mut w = 0usize;
    for &(offset, length) in &slots[..n] {
      let (o, l) = (usize::from(offset), usize::from(length));
      expected[w..w + l].copy_from_slice(&tuple[o..o + l]);
      w += l;
    }
    assert_eq!(&delta[meta..meta + w], &expected[..w]);

    // 按槽位把 delta 段回填到 tuple 副本，必须与源 tuple 完全一致
    let mut applied = tuple;
    let mut r = meta;
    for &(offset, length) in &slots[..n] {
      let (o, l) = (usize::from(offset), usize::from(length));
      applied[o..o + l].copy_from_slice(&delta[r..r + l]);
      r += l;
    }
    assert_eq!(applied[..tlen], tuple[..tlen]);
  }
}
