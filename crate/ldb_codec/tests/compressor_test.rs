//! Blob 压缩缓冲：上游 `BlobCompressor` 语义（Reset/CompressBlob/ToSpan）+ 对抗输入。

use fastrand::Rng;
use ldb_codec::{Error, PageCodec};
use ldb_type::PAGE_SIZE;

fn pattern(len: usize) -> Vec<u8> {
  (0..len).map(|i| (i % 251) as u8).collect()
}

fn random_vec(len: usize, seed: u64) -> Vec<u8> {
  let mut v = vec![0u8; len];
  Rng::with_seed(seed).fill(&mut v);
  v
}

#[test]
fn blob_roundtrip_across_sizes() {
  let mut c = PageCodec::new();
  for len in [
    0usize, 1, 2, 7, 63, 64, 251, 4095, 4096, 4097, 10_000, 65_536,
  ] {
    let blob = pattern(len);
    c.compress_blob(&blob).unwrap();
    assert_eq!(c.compressed_len(), c.as_slice().len());
    assert_eq!(c.to_span().len() % PAGE_SIZE, 0, "ToSpan 必须页对齐");
    assert!(c.to_span().len() >= c.compressed_len());
    if len > 1024 {
      assert!(c.compressed_len() < len, "重复模式 len={len} 应压缩变小");
    }
    // 精确视图（不含对齐填充）可解压
    let comp = c.as_slice().to_vec();
    assert_eq!(
      PageCodec::mzstd_decompress(&comp).unwrap(),
      blob,
      "len={len}"
    );
    // 定长 Blob 路径（不可信输入用）：max_size 取与原长相同的硬上限
    let back = c.zstd_decompress_blob(&comp, len, len).unwrap();
    assert_eq!(back, blob, "len={len} 定长解压");
  }
}

#[test]
fn empty_blob_does_not_panic() {
  let mut c = PageCodec::new();
  c.compress_blob(&[]).unwrap();
  assert!(c.compressed_len() > 0, "空 blob 仍应生成 zstd 帧");
  assert_eq!(c.to_span().len(), PAGE_SIZE, "空 blob 帧不足一页，补齐一页");
  assert_eq!(
    PageCodec::mzstd_decompress(c.as_slice()).unwrap(),
    Vec::new()
  );
  let comp = c.as_slice().to_vec();
  assert_eq!(c.zstd_decompress_blob(&comp, 0, 0).unwrap(), &[] as &[u8]);
}

#[test]
fn to_span_padding_is_zeroed() {
  let mut c = PageCodec::new();
  // 先写一轮不可压缩数据弄脏缓冲，再压小 blob：填充位必须被清零而非残留旧数据
  c.compress_blob(&random_vec(8192, 11)).unwrap();
  let blob = vec![0x5Au8; 1000];
  c.compress_blob(&blob).unwrap();
  let span = c.to_span();
  assert!(
    span[c.compressed_len()..].iter().all(|&b| b == 0),
    "对齐填充须为 0"
  );
}

#[test]
fn reset_requires_page_aligned_nonzero_size() {
  let mut c = PageCodec::new();
  assert!(matches!(c.reset(0), Err(Error::InvalidConfig(_))));
  assert!(matches!(c.reset(4095), Err(Error::InvalidConfig(_))));
  assert!(matches!(c.reset(4097), Err(Error::InvalidConfig(_))));
  c.reset(8192).unwrap();
  assert_eq!(c.as_slice().len(), 0);
  assert_eq!(c.compressed_len(), 0);
  c.compress_blob(&pattern(100)).unwrap();
  // 已有 8KiB 缓冲，小 blob 不复用 Reset（不缩容）
  assert!(c.compressed_len() > 0);
  // Reset 必须清空有效长度（不能只换 buffer 留下旧 size）
  c.reset(PAGE_SIZE as u64).unwrap();
  assert_eq!(c.compressed_len(), 0);
  assert_eq!(c.as_slice(), &[] as &[u8]);

  c.destroy_buffer();
  assert_eq!(c.as_slice(), &[] as &[u8]);
  assert_eq!(c.to_span(), &[] as &[u8]);
  // 释放后仍可压缩（内部按需重建）
  c.compress_blob(&pattern(3000)).unwrap();
  assert!(c.compressed_len() > 0);
}

#[test]
fn blob_decompress_rejects_wrong_expected_size() {
  let mut c = PageCodec::new();
  let blob = pattern(5000);
  c.compress_blob(&blob).unwrap();
  let comp = c.as_slice().to_vec();

  // 帧声明 5000B，期望 5001B：解出 5000 → BlobSizeMismatch（声明 vs 期望不符）
  assert!(matches!(
    c.zstd_decompress_blob(&comp, blob.len() + 1, blob.len() + 1),
    Err(Error::BlobSizeMismatch { expected, got })
      if expected == blob.len() + 1 && got == blob.len()
  ));
  // 目标比实际短：解码器在切片满时即拒绝
  assert!(
    c.zstd_decompress_blob(&comp, blob.len() - 1, blob.len() - 1)
      .is_err()
  );

  // 另一个真实帧（3000B）配期望 5000B：解出 3000 → BlobSizeMismatch，
  // 不能因为「成功解完」就忽略元数据声明的原长。
  let small = pattern(3000);
  c.compress_blob(&small).unwrap();
  let small_comp = c.as_slice().to_vec();
  assert!(matches!(
    c.zstd_decompress_blob(&small_comp, 5000, 5000),
    Err(Error::BlobSizeMismatch {
      expected: 5000,
      got: 3000
    })
  ));
  // 该帧按真实原长解压成功（排除「帧本身坏了」的误判）。
  assert_eq!(
    c.zstd_decompress_blob(&small_comp, 3000, 3000).unwrap(),
    small.as_slice()
  );

  // 帧 6000B 配期望 5000B：输出上界 5000，解码器在切片满时拒绝。
  let big = pattern(6000);
  c.compress_blob(&big).unwrap();
  let big_comp = c.as_slice().to_vec();
  assert!(c.zstd_decompress_blob(&big_comp, 5000, 5000).is_err());
}

/// `max_size` 是独立于 Blob 元数据的可信硬上限：声明原长超过它时必须在分配前拒绝。
#[test]
fn blob_expect_size_exceeding_max_is_rejected() {
  let mut c = PageCodec::new();
  let blob = pattern(1000);
  c.compress_blob(&blob).unwrap();
  let comp = c.as_slice().to_vec();

  // 元数据声明 5000B，但可信上限 4096B：不得按 5000B resize。
  assert!(matches!(
    c.zstd_decompress_blob(&comp, 5000, 4096),
    Err(Error::InvalidConfig(_))
  ));
  // 边界：expect_size == max_size 合法。
  assert_eq!(
    c.zstd_decompress_blob(&comp, 1000, 1000).unwrap(),
    blob.as_slice()
  );
}

#[test]
fn blob_corrupt_inputs_are_errors() {
  let mut c = PageCodec::new();
  let blob = pattern(3000);
  c.compress_blob(&blob).unwrap();
  let comp = c.as_slice().to_vec();

  // 截断
  for cut in [0usize, 1, comp.len() / 2, comp.len() - 1] {
    assert!(
      c.zstd_decompress_blob(&comp[..cut], blob.len(), blob.len())
        .is_err()
    );
  }
  // 随机垃圾
  let junk = random_vec(128, 3);
  assert!(
    c.zstd_decompress_blob(&junk, blob.len(), blob.len())
      .is_err()
  );
  assert!(PageCodec::mzstd_decompress(&junk).is_err());
}

#[test]
fn blob_zip_bomb_bounded_by_expect_size() {
  // 单段帧声明 1 GiB，实际帧 13 字节
  let mut frame = Vec::new();
  frame.extend_from_slice(&0xFD2F_B528u32.to_le_bytes());
  frame.push(0xA0);
  frame.extend_from_slice(&((1u32 << 30) - 256).to_le_bytes());
  let block: u32 = (131_071 << 3) | (1 << 1) | 1;
  frame.extend_from_slice(&block.to_le_bytes()[..3]);
  frame.push(0x41);

  let mut c = PageCodec::new();
  // 期望原长 4 KiB：窗口按 expect_size 收紧到 2^12，帧声明 1 GiB 窗口 → 拒绝
  assert!(matches!(
    c.zstd_decompress_blob(&frame, PAGE_SIZE, PAGE_SIZE),
    Err(Error::MzstdDecode(_))
  ));
  // 可信 Vec 路径同样拒绝（默认 128 MiB 窗口上限）
  assert!(PageCodec::mzstd_decompress(&frame).is_err());

  // 非单段帧、窗口 128 KiB、声明 1 GiB：定长路径在 4 KiB 处停下
  let mut f2 = Vec::new();
  f2.extend_from_slice(&0xFD2F_B528u32.to_le_bytes());
  f2.push(0x80);
  f2.push(0x38);
  f2.extend_from_slice(&((1u32 << 30) - 256).to_le_bytes());
  for i in 0..1024u32 {
    let last = u32::from(i == 1023);
    let b: u32 = (131_071 << 3) | (1 << 1) | last;
    f2.extend_from_slice(&b.to_le_bytes()[..3]);
    f2.push(0x41);
  }
  assert!(c.zstd_decompress_blob(&f2, PAGE_SIZE, PAGE_SIZE).is_err());
}
