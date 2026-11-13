//! R3 独立最小崩溃复现（不复用 R2 的 `dwb_crash_matrix_test` harness）。
//!
//! 只验证 3 个关键状态 + 1 个新增校验：
//! 1. DWB 写后未 sync 崩溃：任意持久化前缀/乱序都不得产生混合页；
//! 2. DWB sync 后最终位置半页新：恢复必须给出整页新值；
//! 3. 失效头写一半：允许「回放」或「不回放」，但页必须整页新且二次恢复幂等；
//! 4. 记录头 CRC 有效但 `final_off` 非页对齐：按损坏记录拒绝，不得跨页写穿。
//!
//! 记录格式（协议契约）：`[4B magic | 2B version | 2B len | 4B crc | 4B reserved |
//! 8B final_off | 8B generation | 32B reserved | 4096B page]`，CRC 覆盖
//! `[..8] + 4 个零字节 + [12..]`。

use std::{
  env::temp_dir,
  fs::{create_dir_all, read, remove_dir_all, write},
  io,
  path::{Path, PathBuf},
  process::id,
  sync::atomic::{AtomicU64, Ordering},
};

use crc32fast::Hasher;
use ldb_io::{CompioFileBackend, DWB_RECORD, DwbWriter};
use ldb_type::PAGE_SIZE;

const OLD: u8 = 0xAA;
const DWB_HEADER: usize = 64;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
  let dir = temp_dir().join(format!(
    "ldb_io_r3_probe_{tag}_{}_{}",
    id(),
    SEQ.fetch_add(1, Ordering::Relaxed)
  ));
  create_dir_all(&dir).unwrap();
  dir
}

async fn open_data(path: &Path) -> CompioFileBackend {
  CompioFileBackend::open_direct_or_buffered(path)
    .await
    .unwrap()
}

async fn open_dwb(path: &Path) -> CompioFileBackend {
  CompioFileBackend::open_buffered(path).await.unwrap()
}

fn page(val: u8) -> Vec<u8> {
  vec![val; PAGE_SIZE]
}

fn page_image(vals: &[u8]) -> Vec<u8> {
  let mut img = Vec::with_capacity(vals.len() * PAGE_SIZE);
  for &v in vals {
    img.extend(page(v));
  }
  img
}

fn record_crc(rec: &[u8]) -> u32 {
  let mut h = Hasher::new();
  h.update(&rec[..8]);
  h.update(&[0u8; 4]);
  h.update(&rec[12..]);
  h.finalize()
}

/// 改写 final_off 并重算 CRC（模拟「CRC 自洽但落点非法」的最坏记录）。
fn patch_final_off(rec: &mut [u8], off: u64) {
  rec[16..24].copy_from_slice(&off.to_le_bytes());
  let crc = record_crc(rec);
  rec[8..12].copy_from_slice(&crc.to_le_bytes());
}

/// 用给定镜像建一对新文件并打开（每个状态独立子目录）。
async fn boot(
  root: &Path,
  tag: &str,
  data_img: &[u8],
  dwb_img: &[u8],
) -> (CompioFileBackend, CompioFileBackend) {
  let dir = root.join(tag);
  create_dir_all(&dir).unwrap();
  let dp = dir.join("data.db");
  let wp = dir.join("dwb.dat");
  write(&dp, data_img).unwrap();
  write(&wp, dwb_img).unwrap();
  (open_data(&dp).await, open_dwb(&wp).await)
}

async fn assert_pages(data: &CompioFileBackend, want_new: usize, vals: &[u8]) {
  for (p, &v) in vals.iter().enumerate() {
    let got = data
      .read_page_owned(p as u64, vec![0u8; PAGE_SIZE])
      .await
      .unwrap();
    let want = if p < want_new { v } else { OLD };
    assert!(
      got.iter().all(|&x| x == want),
      "页 {p} 期望整页 {want:#04x}，实际 first={:#04x} last={:#04x}",
      got[0],
      got[PAGE_SIZE - 1]
    );
  }
}

/// 状态 1：DWB `write_page` 后、`sync` 前崩溃。
///
/// 未 sync 的记录允许丢失、也允许完整持久化，但恢复只能回放完整前缀；
/// 任何半条记录（撕裂）必须被 CRC 拒绝，页不得混合。
#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn unsynced_dwb_prefix_never_tears() -> io::Result<()> {
  assert_eq!(DWB_RECORD, 4160);
  let root = tmp_dir("unsynced");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  let vals = [0x11u8, 0x22, 0x33];
  write(&dp, page_image(&[OLD; 3]))?;
  let data = open_data(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.sync_durable().await?;

  let (mut w, applied) = DwbWriter::open(&dwb, &data, 0).await?;
  assert_eq!(applied, 0);
  for (p, &v) in vals.iter().enumerate() {
    w.write_page(&dwb, &page(v), p as u64 * PAGE_SIZE as u64)
      .await?;
  }
  // 故意不调用 w.sync：模拟掉电时只有任意前缀持久化
  let staged = read(&wp)?;
  assert_eq!(staged.len(), 3 * DWB_RECORD, "DWB 文件长度应是 3 条记录");

  for cut in [0usize, 64, 100, 4159, 4160, 4161, 8320, 12479, 12480] {
    let (data2, dwb2) = boot(
      &root,
      &format!("cut{cut}"),
      &page_image(&[OLD; 3]),
      &staged[..cut],
    )
    .await;
    let want = cut / DWB_RECORD;
    assert_eq!(
      DwbWriter::recover(&dwb2, &data2, 0).await?,
      want as u64,
      "cut={cut}"
    );
    assert_pages(&data2, want, &vals).await;
    drop(data2);
    drop(dwb2);
  }

  // 乱序持久化：记录 0 丢失、记录 1/2 完整。恢复必须停在 0，不能跳读后续记录。
  let mut hole = staged.clone();
  hole[..DWB_HEADER].fill(0);
  let (data2, dwb2) = boot(&root, "hole0", &page_image(&[OLD; 3]), &hole).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await?, 0);
  assert_pages(&data2, 0, &vals).await;
  drop(data2);
  drop(dwb2);

  drop(data);
  drop(dwb);
  remove_dir_all(&root)?;
  Ok(())
}

/// 状态 2：DWB 已 sync，最终位置只写了半页。
#[compio::test]
async fn synced_dwb_repairs_torn_final_page() -> io::Result<()> {
  let root = tmp_dir("torn_final");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  let vals = [0x44u8, 0x55];
  write(&dp, page_image(&[OLD; 2]))?;
  let data = open_data(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.sync_durable().await?;

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await?;
  for (p, &v) in vals.iter().enumerate() {
    w.write_page(&dwb, &page(v), p as u64 * PAGE_SIZE as u64)
      .await?;
  }
  w.sync(&dwb).await?;
  let dwb_img = read(&wp)?;

  // 页 0 只有前半页新值，页 1 仍旧
  let mut torn = page_image(&[OLD; 2]);
  torn[..PAGE_SIZE / 2].copy_from_slice(&page(vals[0])[..PAGE_SIZE / 2]);
  let (data2, dwb2) = boot(&root, "half", &torn, &dwb_img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await?, 2);
  assert_pages(&data2, 2, &vals).await;
  // 失效后重复恢复幂等
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await?, 0);
  assert_pages(&data2, 2, &vals).await;

  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root)?;
  Ok(())
}

/// 状态 3：最终位置已全部持久化，清零失效头的 64B 只写了一部分。
#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn half_written_invalidation_is_idempotent() -> io::Result<()> {
  let root = tmp_dir("invalidate");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  let vals = [0x66u8, 0x77];
  write(&dp, page_image(&[OLD; 2]))?;
  let data = open_data(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.sync_durable().await?;

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await?;
  for (p, &v) in vals.iter().enumerate() {
    w.write_page(&dwb, &page(v), p as u64 * PAGE_SIZE as u64)
      .await?;
  }
  w.sync(&dwb).await?;
  // 最终位置全部落盘（checkpoint 的 data sync 已完成，尚未失效）
  for (p, &v) in vals.iter().enumerate() {
    data
      .write_bytes_owned(page(v), p as u64 * PAGE_SIZE as u64)
      .await?;
  }
  data.sync_durable().await?;
  let dwb_img = read(&wp)?;
  let all_new = page_image(&vals);

  // 失效写是 64 个零字节；崩溃可能持久化任意子集。已为零的保留区子集
  // 是幂等 no-op（记录仍可解码），破坏 magic/CRC 的子集则 0 回放。
  let zero_prefix = |n: usize| -> Vec<usize> { (0..n).collect() };
  let prefix1 = zero_prefix(1);
  let prefix4 = zero_prefix(4);
  let prefix8 = zero_prefix(8);
  let prefix16 = zero_prefix(16);
  let prefix64 = zero_prefix(64);
  let reserved_hi = (32..64).collect::<Vec<_>>();
  let cases: [(&str, &[usize], u64); 8] = [
    ("prefix1", &prefix1, 0),
    ("prefix4", &prefix4, 0),
    ("prefix8", &prefix8, 0),
    ("prefix16", &prefix16, 0),
    ("prefix64", &prefix64, 0),
    ("crc_zero", &[8, 9, 10, 11], 0),
    ("reserved_lo", &[12, 13, 14, 15], 2),
    ("reserved_hi", &reserved_hi, 2),
  ];
  for (tag, bytes, want) in cases {
    let mut img = dwb_img.clone();
    for &b in bytes {
      img[b] = 0;
    }
    let (data2, dwb2) = boot(&root, tag, &all_new, &img).await;
    assert_eq!(
      DwbWriter::recover(&dwb2, &data2, 0).await?,
      want,
      "{tag}: 失效头部分持久化的回放决策"
    );
    assert_pages(&data2, 2, &vals).await;
    assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await?, 0);
    drop(data2);
    drop(dwb2);
  }

  drop(data);
  drop(dwb);
  remove_dir_all(&root)?;
  Ok(())
}

/// CRC 自洽但 `final_off` 非页对齐：必须拒绝（否则 4096B 写会跨两页）。
#[compio::test]
async fn misaligned_final_off_is_rejected() -> io::Result<()> {
  let root = tmp_dir("misaligned");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  let vals = [0x88u8];
  write(&dp, page_image(&[OLD; 2]))?;
  let data = open_data(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.sync_durable().await?;

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await?;
  w.write_page(&dwb, &page(vals[0]), 0).await?;
  w.sync(&dwb).await?;
  let mut img = read(&wp)?;
  patch_final_off(&mut img[..DWB_RECORD], 4095);

  let (data2, dwb2) = boot(&root, "crash", &page_image(&[OLD; 2]), &img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await?, 0);
  assert_pages(&data2, 0, &vals).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await?, 0);

  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root)?;
  Ok(())
}
