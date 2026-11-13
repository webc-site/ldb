//! 页编解码：mzstd 帧格式、对抗输入、上游页布局字节模式。
//!
//! 页在本 crate 中是**不透明 4 KiB 字节块**（页头/槽位解析属 `ldb_page`）；
//! 因此对抗性要求是：任意 4 KiB 字节模式（合法页头、满槽、全 0、全 FF）
//! 经压缩/解压必须逐字节还原，畸形输入必须 `Err` 且不 panic。

use std::{
  alloc::{GlobalAlloc, Layout, System},
  cell::Cell,
};

use fastrand::Rng;
use ldb_codec::{Error, PageCodec};
use ldb_pack::UNCOMPRESSED;
use ldb_type::PAGE_SIZE;
use mzstd::DecodeError;

// 线程局部分配计数：只统计当前测试线程，用于证明「恶意帧在分配前被拒」。
// 用 `const` 初始化避免计数器的 TLS 惰性初始化本身触发分配/递归。
thread_local! {
  static ALLOC_CNT: Cell<usize> = const { Cell::new(0) };
  static ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: 仅统计后原样转发给 System，不改变任何分配语义；计数器为线程局部 Cell。
unsafe impl GlobalAlloc for Counting {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    let _ = ALLOC_CNT.try_with(|c| c.set(c.get() + 1));
    let _ = ALLOC_BYTES.try_with(|c| c.set(c.get() + layout.size()));
    // SAFETY: 调用方满足 GlobalAlloc::alloc 的布局契约
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: ptr 来自 System.alloc 且布局一致
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    let _ = ALLOC_CNT.try_with(|c| c.set(c.get() + 1));
    let _ = ALLOC_BYTES.try_with(|c| c.set(c.get() + layout.size()));
    // SAFETY: 同 alloc
    unsafe { System.alloc_zeroed(layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    let _ = ALLOC_CNT.try_with(|c| c.set(c.get() + 1));
    let _ = ALLOC_BYTES.try_with(|c| c.set(c.get() + new_size.saturating_sub(layout.size())));
    // SAFETY: ptr/layout 来自 System，new_size 由调用方保证合法
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn alloc_cnt() -> usize {
  ALLOC_CNT.with(Cell::get)
}

fn alloc_bytes() -> usize {
  ALLOC_BYTES.with(Cell::get)
}

/// 上游 `PageHeader` 大小（`page.h` / ldb_page `const_assert_eq!(size_of::<PageHeader>(), 104)`）
const HEADER_LEN: usize = 104;
/// 页内最大 8 字节槽位数：`(4096 - 104) / 8 = 499`
const MAX_SLOTS: usize = (PAGE_SIZE - HEADER_LEN) / 8;

fn zeros() -> [u8; PAGE_SIZE] {
  [0u8; PAGE_SIZE]
}

fn all(value: u8) -> [u8; PAGE_SIZE] {
  [value; PAGE_SIZE]
}

fn ascending() -> [u8; PAGE_SIZE] {
  let mut p = [0u8; PAGE_SIZE];
  for (i, b) in p.iter_mut().enumerate() {
    *b = i as u8;
  }
  p
}

fn text() -> [u8; PAGE_SIZE] {
  let phrase = b"the quick brown fox jumps over the lazy dog; 4KiB page compression. ";
  let mut p = [0u8; PAGE_SIZE];
  for (i, b) in p.iter_mut().enumerate() {
    *b = phrase[i % phrase.len()];
  }
  p
}

fn random(seed: u64) -> [u8; PAGE_SIZE] {
  let mut p = [0u8; PAGE_SIZE];
  Rng::with_seed(seed).fill(&mut p);
  p
}

/// 按上游 `page.h` 布局构造页：104B 头 + `slots` 个 8B 根/槽指针。
fn page_layout(slots: usize) -> [u8; PAGE_SIZE] {
  assert!(slots <= MAX_SLOTS);
  let mut p = [0u8; PAGE_SIZE];
  p[0..8].copy_from_slice(&7u64.to_le_bytes()); // idx_id
  p[8..16].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes()); // p_gsn
  p[16..24].copy_from_slice(&0x8877_6655_4433_2211u64.to_le_bytes()); // prev_gcwrite
  p[24..28].copy_from_slice(&(slots as u32).to_le_bytes()); // wcnt
  for i in 0..8 {
    p[32 + i * 8..40 + i * 8].copy_from_slice(&(1000 + i as u64).to_le_bytes()); // w_p_gsn
  }
  p[96] = 1; // dirty
  p[97] = 1; // gc_read
  for i in 0..slots {
    p[HEADER_LEN + i * 8..HEADER_LEN + (i + 1) * 8]
      .copy_from_slice(&(0xDEAD_0000_0000_0000u64 | i as u64).to_le_bytes());
  }
  p
}

fn roundtrip(page: &[u8; PAGE_SIZE]) -> u16 {
  let mut dst = [0u8; PAGE_SIZE];
  let sz = PageCodec::compress_page(page, &mut dst).expect("压缩不应失败");
  assert!(
    sz == UNCOMPRESSED || sz < PAGE_SIZE as u16,
    "压缩尺寸必须 < 4KiB 或原文哨兵，实际 {sz}"
  );
  let mut back = [0u8; PAGE_SIZE];
  if sz == UNCOMPRESSED {
    PageCodec::decompress_page(page, sz, &mut back).expect("原文回退解压失败");
  } else {
    PageCodec::decompress_page(&dst[..sz as usize], sz, &mut back).expect("压缩页解压失败");
  }
  assert_eq!(back, *page, "4KiB 页往返不一致");
  sz
}

#[test]
fn page_roundtrip_all_byte_patterns() {
  let mut cases: Vec<(&str, [u8; PAGE_SIZE])> = vec![
    ("zeros", zeros()),
    ("ones", all(0xFF)),
    ("ascending", ascending()),
    ("text", text()),
    ("random", random(42)),
    ("layout_0_slots", page_layout(0)),
    ("layout_1_slot", page_layout(1)),
    ("layout_max_slots", page_layout(MAX_SLOTS)),
  ];
  for v in [0x01u8, 0x7F, 0x80, 0xFE] {
    cases.push(("uniform", all(v)));
  }
  for (name, page) in cases {
    roundtrip(&page);
    println!("{name}: ok");
  }
}

#[test]
fn compressible_pages_shrink_and_random_falls_back() {
  assert!(roundtrip(&zeros()) < UNCOMPRESSED, "全 0 页应压缩");
  assert!(roundtrip(&all(0xFF)) < UNCOMPRESSED, "全 FF 页应压缩");
  assert!(roundtrip(&text()) < UNCOMPRESSED, "文本页应压缩");
  assert_eq!(roundtrip(&random(7)), UNCOMPRESSED, "随机页必须回退原文");
}

/// 与上游 `ZSTD_compress(..., level=1)` 的帧逐字节一致（C zstd 1.5.7
/// `zstd -1 --no-check` 对 4096 个 0 的输出，字面量钉死格式/级别）。
#[test]
fn zero_page_frame_matches_c_zstd_level1() {
  const WANT: [u8; 19] = [
    0x28, 0xb5, 0x2f, 0xfd, 0x60, 0x00, 0x0f, 0x4d, 0x00, 0x00, 0x10, 0x00, 0x00, 0x01, 0x00, 0xfb,
    0xf7, 0x01, 0x16,
  ];
  let mut dst = [0u8; PAGE_SIZE];
  let sz = PageCodec::compress_page(&zeros(), &mut dst).unwrap();
  assert_eq!(sz as usize, WANT.len());
  assert_eq!(dst[..WANT.len()], WANT);
}

/// level 1 与 level 3 在该页上帧长/字节不同（C zstd 1.5.7 实测：
/// level1 = 2523B / crc 0x60694589，level3 = 2338B / crc 0xC5B57603），
/// 用字面量钉死「上游固定 level 1」而非 mzstd 默认 level 3。
#[test]
fn page_level_is_upstream_one() {
  let mut p = [0u8; PAGE_SIZE];
  for i in 0..PAGE_SIZE / 16 {
    let k = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    p[i * 16..i * 16 + 8].copy_from_slice(&k.to_le_bytes());
    p[i * 16 + 8..i * 16 + 16].copy_from_slice(&(i as u64 % 97).to_le_bytes());
  }
  let mut dst = [0u8; PAGE_SIZE];
  let sz = PageCodec::compress_page(&p, &mut dst).unwrap() as usize;
  assert_eq!(sz, 2523);
  assert_eq!(PageCodec::crc32_checksum(&dst[..sz]), 0x6069_4589);
}

#[test]
fn every_truncated_compressed_frame_is_rejected() {
  let src = text();
  let mut dst = [0u8; PAGE_SIZE];
  let sz = PageCodec::compress_page(&src, &mut dst).unwrap() as usize;
  assert!(sz > 1, "文本页应产生真实压缩帧");
  let mut back = [0u8; PAGE_SIZE];
  for k in 0..sz {
    let r = PageCodec::decompress_page(&dst[..k], sz as u16, &mut back);
    assert!(r.is_err(), "截断到 {k}/{sz} 字节仍被接受");
  }
}

#[test]
fn every_truncated_raw_page_is_rejected() {
  let src = random(9);
  // 1..=4095 全部拒绝，且 `dst` 一个字节都不能被写：否则忽略 Err 的调用方
  // （ldb_space/src/page_io.rs:458/487/500）会读到「半页新数据 + 半页旧数据」。
  for k in 1..PAGE_SIZE {
    let mut back = [0x5Au8; PAGE_SIZE];
    let r = PageCodec::decompress_page(&src[..k], UNCOMPRESSED, &mut back);
    assert!(
      matches!(r, Err(Error::Truncated { .. })),
      "原文截断到 {k} 字节未被拒绝"
    );
    assert_eq!(back, [0x5Au8; PAGE_SIZE], "原文截断到 {k} 字节仍写入了 dst");
  }
  // 长度 0 同样拒绝（无任何可用页数据）。
  let mut back = [0x5Au8; PAGE_SIZE];
  assert!(matches!(
    PageCodec::decompress_page(&[], UNCOMPRESSED, &mut back),
    Err(Error::Truncated { .. })
  ));
  assert_eq!(back, [0x5Au8; PAGE_SIZE]);
  PageCodec::decompress_page(&src, UNCOMPRESSED, &mut back).unwrap();
  assert_eq!(back, src);
}

#[test]
fn raw_page_may_carry_following_binpacked_bytes() {
  let src = random(10);
  // 后续字节是一整页「下一页」（随机内容，与 src 必然不同），不是哨兵垃圾。
  let next = random(11);
  let mut padded = src.to_vec();
  padded.extend_from_slice(&next);
  let mut back = [0u8; PAGE_SIZE];
  PageCodec::decompress_page(&padded, UNCOMPRESSED, &mut back).unwrap();
  assert_eq!(back, src, "原文路径只取前 4096 字节，不得读入下一页");
}

#[test]
fn comp_sz_is_authoritative_and_validated() {
  let src = text();
  let mut dst = [0u8; PAGE_SIZE];
  let sz = PageCodec::compress_page(&src, &mut dst).unwrap();
  // 装箱输入：本页帧后紧邻下一页的**合法压缩帧**（真实 bin-packing 场景，
  // 不是 0xAA 垃圾——垃圾帧只能证明「读多了会报错」，不能证明「没读多」）。
  let next = zeros();
  let mut next_dst = [0u8; PAGE_SIZE];
  let next_sz = PageCodec::compress_page(&next, &mut next_dst).unwrap();
  assert_ne!(next_sz, UNCOMPRESSED);
  let mut binpacked = dst[..sz as usize].to_vec();
  binpacked.extend_from_slice(&next_dst[..next_sz as usize]);

  let mut back = [0u8; PAGE_SIZE];
  // 切片比 comp_sz 长：只解压前 comp_sz 字节（装箱页场景）
  PageCodec::decompress_page(&binpacked, sz, &mut back).unwrap();
  assert_eq!(back, src, "comp_sz 必须只解本页，不得触碰下一页帧");

  // 反向证明上面不是「解两帧也恰好通过」：把 comp_sz 扩到两帧时输出 8192B，
  // 必须因超过一页而失败（解码器是 Concatenated 模式，会尝试第二帧）。
  assert!(
    PageCodec::decompress_page(&binpacked, sz + next_sz, &mut back).is_err(),
    "两帧串联不应解出单页"
  );

  // 切片比 comp_sz 短：截断
  assert!(matches!(
    PageCodec::decompress_page(&dst[..sz as usize - 1], sz, &mut back),
    Err(Error::Truncated { .. })
  ));

  // 恶意/损坏尺寸字段
  for bad in [0u16, UNCOMPRESSED + 1, 0x7FFF] {
    assert!(matches!(
      PageCodec::decompress_page(&binpacked, bad, &mut back),
      Err(Error::InvalidCompressedSize(v)) if v == bad
    ));
  }

  // comp_sz 声明 4096（原文哨兵）但只有压缩帧长度：按截断拒绝
  assert!(matches!(
    PageCodec::decompress_page(&dst[..sz as usize], UNCOMPRESSED, &mut back),
    Err(Error::Truncated { .. })
  ));
}

#[test]
fn malformed_frames_are_errors_not_panics() {
  let mut back = [0u8; PAGE_SIZE];
  // 空输入
  assert!(PageCodec::decompress_page(&[], 1, &mut back).is_err());
  // 随机垃圾
  for len in [1usize, 4, 8, 13, 64, 4095] {
    let mut junk = vec![0u8; len];
    Rng::with_seed(len as u64).fill(&mut junk);
    assert!(PageCodec::decompress_page(&junk, len as u16, &mut back).is_err());
    let ff = vec![0xFFu8; len];
    assert!(PageCodec::decompress_page(&ff, len as u16, &mut back).is_err());
  }
  // 篡改 magic 的合法帧
  let src = text();
  let mut dst = [0u8; PAGE_SIZE];
  let sz = PageCodec::compress_page(&src, &mut dst).unwrap() as usize;
  dst[0] ^= 0xFF;
  assert!(PageCodec::decompress_page(&dst[..sz], sz as u16, &mut back).is_err());
}

/// 手工构造单段帧：声明 1 GiB 内容（帧仅 13 字节）。
fn bomb_1gib_single_segment() -> Vec<u8> {
  let mut f = Vec::new();
  f.extend_from_slice(&0xFD2F_B528u32.to_le_bytes()); // zstd magic
  f.push(0xA0); // 单段 + 4B 内容长度
  f.extend_from_slice(&((1u32 << 30) - 256).to_le_bytes());
  let block: u32 = (131_071 << 3) | (1 << 1) | 1; // last + RLE + 128KiB-1
  f.extend_from_slice(&block.to_le_bytes()[..3]);
  f.push(0x41);
  f
}

#[test]
fn zip_bomb_declared_1gib_rejected_before_allocation() {
  let frame = bomb_1gib_single_segment();
  assert_eq!(frame.len(), 13);
  let mut dst = [0u8; PAGE_SIZE];
  // 页路径：窗口硬限 4 KiB，读任何块之前拒绝
  assert!(PageCodec::decompress_page(&frame, frame.len() as u16, &mut dst).is_err());
  // Vec 路径：单段帧窗口 = 声明内容 1 GiB > 128 MiB 上限
  assert!(PageCodec::mzstd_decompress(&frame).is_err());
  // 定长 Blob 路径：输出上界 4 KiB
  let mut c = PageCodec::new();
  assert!(
    c.zstd_decompress_blob(&frame, PAGE_SIZE, PAGE_SIZE)
      .is_err()
  );
}

#[test]
fn window_limit_rejects_oversized_window_frame() {
  // 预热线程局部页解码器（首次构造有分配），使下面的计数只反映本次解码。
  let src = zeros();
  let mut enc = [0u8; PAGE_SIZE];
  let sz = PageCodec::compress_page(&src, &mut enc).unwrap();
  let mut dst = [0u8; PAGE_SIZE];
  PageCodec::decompress_page(&enc[..sz as usize], sz, &mut dst).unwrap();

  // 非单段帧，窗口描述符声明 128 MiB（log = 10 + (0x88 >> 3) = 27），
  // 远大于页解码器上限 12（4 KiB）；帧体还声明 1 GiB 内容。
  let mut f = Vec::new();
  f.extend_from_slice(&0xFD2F_B528u32.to_le_bytes());
  f.push(0x80); // 非单段 + 4B 内容长度
  f.push(0x88); // window descriptor: log 27（128 MiB）
  f.extend_from_slice(&((1u32 << 30) - 256).to_le_bytes());
  let block: u32 = (131_071 << 3) | (1 << 1) | 1;
  f.extend_from_slice(&block.to_le_bytes()[..3]);
  f.push(0x41);

  let before = alloc_cnt();
  let before_bytes = alloc_bytes();
  let r = PageCodec::decompress_page(&f, f.len() as u16, &mut dst);
  let after = alloc_cnt();
  let after_bytes = alloc_bytes();
  assert!(
    matches!(
      r,
      Err(Error::MzstdDecode(DecodeError::WindowLimitExceeded { .. }))
    ),
    "声明 128 MiB 窗口的帧必须在读取块之前被页解码器拒绝"
  );
  // 逐帧解码会话本身有一次小额分配（mzstd 内部状态，实测 < 1 KiB）；
  // 但绝不允许按帧声明的 128 MiB 窗口分配内存。
  let bytes = after_bytes - before_bytes;
  println!(
    "window-reject alloc delta: {} allocs / {bytes} bytes",
    after - before
  );
  assert!(
    bytes < 64 * 1024,
    "窗口超限不得分配窗口级内存：本次 {after} 次分配 / {bytes} 字节（128 MiB 窗口至少 128 MiB）",
    after = after - before,
  );
}

#[test]
fn valid_frame_with_non_page_content_is_rejected() {
  let mut back = [0u8; PAGE_SIZE];
  // 合法帧但原长 100：解压输出必须恰为一页
  let small = PageCodec::mzstd_compress(&[0x42u8; 100]).unwrap();
  assert!(matches!(
    PageCodec::decompress_page(&small, small.len() as u16, &mut back),
    Err(Error::CorruptPage(100))
  ));
  // 合法帧但原长 8192：输出超过一页，解码器在切片满时拒绝
  let big = PageCodec::mzstd_compress(&[0u8; 2 * PAGE_SIZE]).unwrap();
  assert!(PageCodec::decompress_page(&big, big.len() as u16, &mut back).is_err());
}

#[test]
fn digest_vectors_are_literal() {
  // CRC-32/ISO-HDLC "123456789" 标准校验值（crc32fast）；与上游
  // `CRC::CRC_32()` 的文档 check value 一致（crc.h:1532）
  assert_eq!(PageCodec::crc32_checksum(b"123456789"), 0xCBF4_3926);
  let want: [u8; 32] = [
    0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
    0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
  ];
  assert_eq!(PageCodec::sha256_digest(b"abc"), want);
}
