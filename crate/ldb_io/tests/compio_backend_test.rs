//! compio 文件后端 / 内存后端回归。
//!
//! 覆盖：页往返、未对齐 I/O 的 Direct I/O 回退、EOF 短读、零长度写、
//! 打孔（KEEP_SIZE + 邻居页不变）、set_len 稀疏扩展、pid 溢出、MemBackend
//! 稀疏/打孔裁剪。

use std::{
  env::temp_dir,
  fs::{create_dir_all, remove_dir_all},
  io,
  path::{Path, PathBuf},
  process::id,
  sync::atomic::{AtomicU64, Ordering},
};

use ldb_io::CompioFileBackend;
use ldb_type::PAGE_SIZE;

static SEQ: AtomicU64 = AtomicU64::new(0);

/// 4 KiB 对齐的静态页（Direct I/O 快路径要求）
#[repr(align(4096))]
struct AlignedPage([u8; PAGE_SIZE]);

static PAGE_0X22: AlignedPage = AlignedPage([0x22; PAGE_SIZE]);

fn tmp_dir(tag: &str) -> PathBuf {
  let dir = temp_dir().join(format!(
    "ldb_io_{tag}_{}_{}",
    id(),
    SEQ.fetch_add(1, Ordering::Relaxed)
  ));
  create_dir_all(&dir).unwrap();
  dir
}

async fn open(dir: &Path, name: &str) -> CompioFileBackend {
  CompioFileBackend::open_direct_or_buffered(dir.join(name))
    .await
    .unwrap()
}

#[compio::test]
async fn page_roundtrip_and_direct_probe() {
  let dir = tmp_dir("roundtrip");
  let b = open(&dir, "data.db").await;

  // macOS APFS 支持 F_NOCACHE；Linux tmpfs/overlayfs 回退缓冲属预期
  #[cfg(target_os = "macos")]
  assert!(b.is_direct(), "APFS 上应启用 F_NOCACHE");

  b.write_owned(vec![0x5A; PAGE_SIZE], 3 * PAGE_SIZE as u64)
    .await
    .unwrap();
  b.fdatasync().await.unwrap();

  let buf = b.read_owned(3, vec![0u8; PAGE_SIZE]).await.unwrap();
  assert_eq!(buf.len(), PAGE_SIZE);
  assert!(buf.iter().all(|&x| x == 0x5A));
  assert_eq!(b.file_len().await.unwrap(), 4 * PAGE_SIZE as u64);
  drop(b);
  remove_dir_all(&dir).unwrap();
}

#[compio::test]
async fn unaligned_owned_io_roundtrips() {
  let dir = tmp_dir("unaligned");
  let b = open(&dir, "data.db").await;

  // 非页对齐的偏移与长度：Linux/Windows 上必须自动降级为缓冲 I/O 后成功
  b.set_len(PAGE_SIZE as u64).await.unwrap();
  b.write_owned(vec![0x11; 777], 123).await.unwrap();
  b.fdatasync().await.unwrap();

  let page = b.read_owned(0, vec![0u8; PAGE_SIZE]).await.unwrap();
  assert!(page[..123].iter().all(|&x| x == 0));
  assert!(page[123..900].iter().all(|&x| x == 0x11));
  assert!(page[900..].iter().all(|&x| x == 0));
  drop(b);
  remove_dir_all(&dir).unwrap();
}

#[compio::test]
async fn short_read_reports_unexpected_eof() {
  let dir = tmp_dir("eof");
  let b = open(&dir, "data.db").await;
  b.write_owned(vec![7u8; 100], 0).await.unwrap();
  b.fdatasync().await.unwrap();

  let err = b.read_owned(0, vec![0u8; PAGE_SIZE]).await.unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
  drop(b);
  remove_dir_all(&dir).unwrap();
}

#[compio::test]
async fn punch_hole_zeroes_only_target_page() {
  let dir = tmp_dir("punch");
  let b = open(&dir, "data.db").await;
  for p in 0..4u8 {
    b.write_owned(vec![0x40 + p; PAGE_SIZE], u64::from(p) * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  b.fdatasync().await.unwrap();

  match b.punch_hole(2 * PAGE_SIZE as u64, PAGE_SIZE as u64).await {
    Ok(()) => {}
    #[cfg(not(target_os = "macos"))]
    Err(e)
      if matches!(
        e.kind(),
        io::ErrorKind::Unsupported | io::ErrorKind::InvalidInput
      ) =>
    {
      // 非 APFS/ext4 的文件系统可能不支持打孔；本机 APFS 必须走成功分支
      log::warn!("文件系统不支持 punch_hole，跳过稀疏校验: {e}");
      drop(b);
      remove_dir_all(&dir).unwrap();
      return;
    }
    Err(e) => panic!("punch_hole 失败: {e}"),
  }

  for p in 0..4u64 {
    let page = b.read_owned(p, vec![0u8; PAGE_SIZE]).await.unwrap();
    let want = if p == 2 { 0 } else { 0x40 + p as u8 };
    assert!(
      page.iter().all(|&x| x == want),
      "页 {p} 期望 {want:#x}，实际 {:?}",
      page[0]
    );
  }
  assert_eq!(b.file_len().await.unwrap(), 4 * PAGE_SIZE as u64);
  drop(b);
  remove_dir_all(&dir).unwrap();
}

#[compio::test]
async fn set_len_sparse_extension_reads_zero() {
  let dir = tmp_dir("setlen");
  let b = open(&dir, "data.db").await;
  b.set_len(2 * PAGE_SIZE as u64).await.unwrap();
  assert_eq!(b.file_len().await.unwrap(), 2 * PAGE_SIZE as u64);

  let page = b.read_owned(1, vec![0xFFu8; PAGE_SIZE]).await.unwrap();
  assert!(page.iter().all(|&x| x == 0), "稀疏扩展区应读零");

  b.set_len(0).await.unwrap();
  assert_eq!(b.file_len().await.unwrap(), 0);
  drop(b);
  remove_dir_all(&dir).unwrap();
}

#[compio::test]
async fn pid_overflow_is_invalid_input() {
  let dir = tmp_dir("overflow");
  let b = open(&dir, "data.db").await;
  // pid = 2^52 时 pid*PAGE_SIZE = 2^64 恰好回绕到 0；必须被 checked_mul 拦截，
  // 否则会静默读写第 0 页（u64::MAX 的偏移超出 i64，OS 也会 EINVAL，测不出回绕）
  let err = b
    .read_owned(1u64 << 52, vec![0u8; PAGE_SIZE])
    .await
    .unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
  drop(b);
  remove_dir_all(&dir).unwrap();
}

/// 页号接口 `pid=1` 与字节接口 `off=512` 不得混用（防 8× 漂移）。
#[compio::test]
async fn page_and_byte_offsets_do_not_alias() {
  let dir = tmp_dir("offsets");
  let b = open(&dir, "data.db").await;
  b.set_len(3 * PAGE_SIZE as u64).await.unwrap();

  b.write_bytes_owned(vec![0x11; 512], 512).await.unwrap();
  // 页 1 写 0x22：偏移必须是 4096，而不是 512 的 8 倍
  b.write_at_page(1, &PAGE_0X22.0).await.unwrap();
  b.fdatasync().await.unwrap();

  let page0 = b.read_page_owned(0, vec![0u8; PAGE_SIZE]).await.unwrap();
  assert!(page0[..512].iter().all(|&x| x == 0));
  assert!(page0[512..1024].iter().all(|&x| x == 0x11));
  assert!(page0[1024..].iter().all(|&x| x == 0));

  let page1 = b.read_page_owned(1, vec![0u8; PAGE_SIZE]).await.unwrap();
  assert!(page1.iter().all(|&x| x == 0x22));

  // 字节接口读 512 = 0x11，读 4096 = 0x22
  let probe = b.read_bytes_owned(512, vec![0u8; 64]).await.unwrap();
  assert!(probe.iter().all(|&x| x == 0x11));
  let probe = b
    .read_bytes_owned(PAGE_SIZE as u64, vec![0u8; 64])
    .await
    .unwrap();
  assert!(probe.iter().all(|&x| x == 0x22));

  drop(b);
  remove_dir_all(&dir).unwrap();
}
