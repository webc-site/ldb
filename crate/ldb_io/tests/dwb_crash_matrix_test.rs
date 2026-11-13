//! DWB 崩溃矩阵（R2 独立 harness，不复用 R1 helper）。
//!
//! 每个可见状态都用「文件快照」落成一对 `data.db`/`dwb.dat` 镜像：
//! - DWB 写后未 fsync：任意前缀/撕裂/乱序持久化；
//! - DWB fsync 后：完整记录 + 最终位置旧/半页新/整页新；
//! - 最终位置 fsync 后、失效前：DWB 完整 + 数据全新；
//! - 失效（清零区首头）中途：头的任意字节子集被清零；
//! - 恢复中途崩溃：手工构造「部分页已回放」的磁盘状态再重跑恢复。
//!
//! 断言只有一条硬性质：恢复后每个页要么整页旧值，要么整页新值，绝不混合。

use std::{
  env::temp_dir,
  fs::{create_dir_all, read, remove_dir_all, write},
  io,
  path::{Path, PathBuf},
  process::id,
  sync::atomic::{AtomicU64, Ordering},
};

use crc32fast::Hasher;
use ldb_io::{
  CompioFileBackend, DWB_RECORD, DwbWriter,
  doublewrite::{DWB_HEADER, DWB_RECORDS},
};
use ldb_type::PAGE_SIZE;

const NPAGES: usize = 4;
const OLD: u8 = 0xAA;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
  let dir = temp_dir().join(format!(
    "ldb_io_matrix_{tag}_{}_{}",
    id(),
    SEQ.fetch_add(1, Ordering::Relaxed)
  ));
  create_dir_all(&dir).unwrap();
  dir
}

async fn open(path: &Path) -> CompioFileBackend {
  CompioFileBackend::open_direct_or_buffered(path)
    .await
    .unwrap()
}

/// DWB 记录 4160B 非 4KiB 对齐：生产路径用缓冲打开（R3 布局决策 c）
async fn open_dwb(path: &Path) -> CompioFileBackend {
  CompioFileBackend::open_buffered(path).await.unwrap()
}

fn page(val: u8) -> Vec<u8> {
  vec![val; PAGE_SIZE]
}

fn old_image() -> Vec<u8> {
  vec![OLD; NPAGES * PAGE_SIZE]
}

fn write_img(path: &Path, bytes: &[u8]) {
  write(path, bytes).unwrap();
}

fn read_img(path: &Path) -> Vec<u8> {
  read(path).unwrap()
}

/// 用一对镜像模拟掉电后的磁盘内容并打开（每个状态一个独立子目录）。
async fn boot_from(
  root: &Path,
  tag: &str,
  data_img: &[u8],
  dwb_img: &[u8],
) -> (CompioFileBackend, CompioFileBackend, PathBuf, PathBuf) {
  let dir = root.join(tag);
  create_dir_all(&dir).unwrap();
  let dp = dir.join("data.db");
  let wp = dir.join("dwb.dat");
  write_img(&dp, data_img);
  write_img(&wp, dwb_img);
  (open(&dp).await, open_dwb(&wp).await, dp, wp)
}

async fn page_bytes(data: &CompioFileBackend, p: usize) -> Vec<u8> {
  data
    .read_page_owned(p as u64, vec![0u8; PAGE_SIZE])
    .await
    .unwrap()
}

/// 精确断言前 `applied` 页为新值，其余为旧值（扫描是连续前缀）。
async fn assert_prefix(data: &CompioFileBackend, new: &[u8], applied: usize) {
  for (p, &new_val) in new.iter().enumerate().take(NPAGES) {
    let got = page_bytes(data, p).await;
    let want = if p < applied { new_val } else { OLD };
    assert!(
      got.iter().all(|&x| x == want),
      "页 {p} 期望 {want:#04x}，实际 first={:#04x} last={:#04x}",
      got[0],
      got[PAGE_SIZE - 1]
    );
  }
}

/// 独立实现记录格式：magic/version/len/crc/generation 与生产代码一致。
fn record_crc(rec: &[u8]) -> u32 {
  let mut h = Hasher::new();
  h.update(&rec[..8]);
  h.update(&[0u8; 4]);
  h.update(&rec[12..]);
  h.finalize()
}

fn patch_generation(rec: &mut [u8], generation: u64) {
  rec[24..32].copy_from_slice(&generation.to_le_bytes());
  let crc = record_crc(rec);
  rec[8..12].copy_from_slice(&crc.to_le_bytes());
}

async fn stage_batch(dwb: &CompioFileBackend, vals: &[u8], writer: &mut DwbWriter) {
  for (p, &val) in vals.iter().enumerate() {
    writer
      .write_page(dwb, &page(val), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
}

fn new_vals() -> Vec<u8> {
  (0..NPAGES).map(|p| 0xB0 + p as u8).collect()
}

/// C0/C1：DWB fsync 之前的任意持久化前缀/撕裂/乱序都不得产生混合页。
#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn before_sync_prefixes_never_tear() {
  // 记录格式字面量锚点：64B 头 + 4096B 页
  assert_eq!(DWB_RECORD, 4160);
  let root = tmp_dir("before_sync");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_img(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let (mut w, applied) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  assert_eq!(applied, 0);
  let new = new_vals();
  stage_batch(&dwb, &new, &mut w).await;
  let staged = read_img(&wp);

  // 任意前缀（含每个记录的边界与内部撕裂点）
  for cut in [
    0usize, 1, 8, 63, 64, 100, 511, 512, 4095, 4159, 4160, 4161, 8300, 8319, 8320, 12479, 12480,
    16639, 16640,
  ] {
    let img = &staged[..cut.min(staged.len())];
    let (data2, dwb2, ..) = boot_from(&root, &format!("cut{cut}"), &old_image(), img).await;
    let got = DwbWriter::recover(&dwb2, &data2, 0).await.unwrap();
    let want = (cut / 4160).min(NPAGES) as u64;
    assert_eq!(got, want, "cut={cut}");
    assert_prefix(&data2, &new, got as usize).await;
    drop(data2);
    drop(dwb2);
  }

  // 单条记录翻位：CRC 必须拒绝该记录并停止扫描
  for rec in 0..NPAGES {
    let mut img = staged.clone();
    img[rec * DWB_RECORD + DWB_HEADER + 10] ^= 0xFF;
    let (data2, dwb2, ..) = boot_from(&root, &format!("flip{rec}"), &old_image(), &img).await;
    assert_eq!(
      DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(),
      rec as u64
    );
    assert_prefix(&data2, &new, rec).await;
    drop(data2);
    drop(dwb2);
  }

  // 乱序持久化：记录 0 丢失、后续记录完整 → 扫描必须停在 0，不能回放后面的页
  let mut img = staged.clone();
  img[..DWB_HEADER].fill(0);
  let (data2, dwb2, ..) = boot_from(&root, "hole0", &old_image(), &img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
  assert_prefix(&data2, &new, 0).await;
  drop(data2);
  drop(dwb2);

  // 记录 1 丢失、记录 0 完整 → 只回放记录 0
  let mut img = staged.clone();
  img[DWB_RECORD..DWB_RECORD + DWB_HEADER].fill(0);
  let (data2, dwb2, ..) = boot_from(&root, "hole1", &old_image(), &img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 1);
  assert_prefix(&data2, &new, 1).await;
  drop(data2);
  drop(dwb2);

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}

/// C2/C3/C4/C5：DWB 已 fsync，最终位置旧/半页新/整页新 → 恢复必回放整批。
#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn after_sync_replays_whole_batch_for_every_final_image() {
  let root = tmp_dir("after_sync");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_img(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  let new = new_vals();
  stage_batch(&dwb, &new, &mut w).await;
  w.sync(&dwb).await.unwrap();
  let dwb_img = read_img(&wp);

  // 最终位置：全旧、每页半新、每页全新
  let mut images = vec![old_image()];
  for torn in 0..NPAGES {
    let mut img = old_image();
    let half = page(new[torn]);
    img[torn * PAGE_SIZE..torn * PAGE_SIZE + PAGE_SIZE / 2].copy_from_slice(&half[..PAGE_SIZE / 2]);
    images.push(img);
  }
  let mut all_new = Vec::new();
  for &v in &new {
    all_new.extend(page(v));
  }
  images.push(all_new);

  for (i, data_img) in images.iter().enumerate() {
    let (data2, dwb2, ..) = boot_from(&root, &format!("final{i}"), data_img, &dwb_img).await;
    assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 4);
    assert_prefix(&data2, &new, NPAGES).await;
    // 失效后重复恢复幂等
    assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
    assert_prefix(&data2, &new, NPAGES).await;
    drop(data2);
    drop(dwb2);
  }

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}

/// C6/C8：恢复中途崩溃 + 失效头部分清零，重跑必须收敛到同一全量新值。
#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn recovery_crash_is_idempotent() {
  let root = tmp_dir("recovery");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_img(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  let new = new_vals();
  stage_batch(&dwb, &new, &mut w).await;
  w.sync(&dwb).await.unwrap();
  let dwb_img = read_img(&wp);
  let all_new = {
    let mut v = Vec::new();
    for &x in &new {
      v.extend(page(x));
    }
    v
  };

  // (a) 回放了一半：页 0/1 已新、2/3 仍旧
  let mut half = old_image();
  half[..2 * PAGE_SIZE].copy_from_slice(&all_new[..2 * PAGE_SIZE]);
  let (data2, dwb2, ..) = boot_from(&root, "half", &half, &dwb_img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 4);
  assert_prefix(&data2, &new, NPAGES).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
  drop(data2);
  drop(dwb2);

  // (b) 回放页 1 时撕裂（半页新半页旧）
  let mut torn = old_image();
  torn[PAGE_SIZE..PAGE_SIZE + PAGE_SIZE / 2].copy_from_slice(&page(new[1])[..PAGE_SIZE / 2]);
  let (data2, dwb2, ..) = boot_from(&root, "torn_replay", &torn, &dwb_img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 4);
  assert_prefix(&data2, &new, NPAGES).await;
  drop(data2);
  drop(dwb2);

  // (c) 回放完成、data fsync 完成、失效前崩溃：DWB 完整 + 数据全新
  let (data2, dwb2, ..) = boot_from(&root, "before_invalidate", &all_new, &dwb_img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 4);
  assert_prefix(&data2, &new, NPAGES).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
  drop(data2);
  drop(dwb2);

  // (d) 清零区首头中途的任意字节子集：能解码则幂等回放，不能解码则 0 回放
  let cases: Vec<(&str, Vec<usize>)> = vec![
    ("zero1", vec![0]),
    ("zero4", vec![0, 1, 2, 3]),
    ("zero8", vec![0, 1, 2, 3, 4, 5, 6, 7]),
    ("zero16", (0..16).collect()),
    ("zero63", (0..63).collect()),
    ("crc_zero", vec![8, 9, 10, 11]),
    ("reserved_lo", vec![12, 13, 14, 15]),
    ("reserved_hi", (32..64).collect()),
  ];
  for (tag, bytes) in &cases {
    let mut img = dwb_img.clone();
    for &b in bytes {
      img[b] = 0;
    }
    let (data2, dwb2, ..) = boot_from(&root, tag, &all_new, &img).await;
    let got = DwbWriter::recover(&dwb2, &data2, 0).await.unwrap();
    assert!(got == 0 || got == 4, "{tag}: applied={got}");
    assert_prefix(&data2, &new, NPAGES).await;
    assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
    drop(data2);
    drop(dwb2);
  }

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}

/// 槽位复用：上一轮槽位里的旧 generation 记录不得回放（含旧值已被独立更新）。
#[compio::test]
async fn slot_reuse_ignores_previous_generation() {
  let root = tmp_dir("reuse");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_img(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  let first = new_vals();
  stage_batch(&dwb, &first, &mut w).await;
  w.sync(&dwb).await.unwrap();
  for (p, &val) in first.iter().enumerate() {
    data
      .write_bytes_owned(page(val), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  w.checkpoint(&dwb, &data).await.unwrap();

  // checkpoint 后页 1 被独立更新（不经过 DWB）
  data
    .write_bytes_owned(page(0xDD), PAGE_SIZE as u64)
    .await
    .unwrap();
  data.fdatasync().await.unwrap();

  // 新批只写页 0，覆盖槽 0；槽 1..3 仍是上一轮记录
  w.write_page(&dwb, &page(0xC0), 0).await.unwrap();
  w.sync(&dwb).await.unwrap();
  let img = read_img(&wp);

  // 同一槽位偏移 0 被复用，但记录里的 generation 与上一轮不同
  let gen0 = u64::from_le_bytes(img[24..32].try_into().unwrap());
  let gen1 = u64::from_le_bytes(img[DWB_RECORD + 24..DWB_RECORD + 32].try_into().unwrap());
  assert_ne!(gen0, gen1, "复用槽位未换 generation");

  let (data2, dwb2, ..) = boot_from(&root, "crash", &read_img(&dp), &img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 1);
  let p0 = page_bytes(&data2, 0).await;
  let p1 = page_bytes(&data2, 1).await;
  let p2 = page_bytes(&data2, 2).await;
  assert!(p0.iter().all(|&x| x == 0xC0), "新批页 0 未回放");
  assert!(
    p1.iter().all(|&x| x == 0xDD),
    "旧 generation 覆盖了独立更新"
  );
  assert!(p2.iter().all(|&x| x == first[2]), "上一轮已提交页被回滚");
  drop(data2);
  drop(dwb2);

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}

/// 构造一条与当前批同格式的旧记录，验证 generation 是唯一批次判别式；
/// 同时验证 `open` 会先 recover，不会在有效记录上从区首覆写。
#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn open_recovers_before_new_batch() {
  let root = tmp_dir("open_recover");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_img(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  let first = new_vals();
  stage_batch(&dwb, &first, &mut w).await;
  w.sync(&dwb).await.unwrap();
  let img = read_img(&wp);

  // 模拟重启：DWB 里是一批未 checkpoint 的有效记录
  let (data2, dwb2, d2p, w2p) = boot_from(&root, "restart", &old_image(), &img).await;
  let (mut w2, applied) = DwbWriter::open(&dwb2, &data2, 0).await.unwrap();
  assert_eq!(applied, 4);
  assert_prefix(&data2, &first, NPAGES).await;

  // 回放后区首已失效；新批从槽 0 开始
  w2.write_page(&dwb2, &page(0xC0), 0).await.unwrap();
  w2.sync(&dwb2).await.unwrap();
  let img2 = read_img(&w2p);
  let data_img = read_img(&d2p);
  drop(data2);
  drop(dwb2);

  let (data3, dwb3, ..) = boot_from(&root, "restart2", &data_img, &img2).await;
  assert_eq!(DwbWriter::recover(&dwb3, &data3, 0).await.unwrap(), 1);
  let got = page_bytes(&data3, 0).await;
  assert!(got.iter().all(|&x| x == 0xC0));
  drop(data3);
  drop(dwb3);

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}

/// 区满：所有槽位有效时必须报 StorageFull，且一个字节都不能被覆盖。
/// 填满 8 MiB 约 0.1 s（`cargo nextest --run-ignored`）。
#[compio::test]
#[ignore = "填满 8 MiB DWB 区约 0.1s；按需 --run-ignored 验证"]
async fn area_full_blocks_without_overwriting() {
  let root = tmp_dir("full");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  let vals = [0x11u8, 0x22, 0x33, 0x44];
  for i in 0..DWB_RECORDS {
    w.write_page(&dwb, &page(vals[i % vals.len()]), 0)
      .await
      .unwrap();
  }
  let before = read_img(&wp);

  let err = w.write_page(&dwb, &page(0xEE), 0).await.unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::StorageFull);
  let after = read_img(&wp);
  assert_eq!(before, after, "区满时写入了数据，覆盖了有效记录");

  // 重新打开：open 必须先回放 2016 条并失效，然后才允许新批
  let (mut w2, applied) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  assert_eq!(applied, DWB_RECORDS as u64);
  w2.write_page(&dwb, &page(0x77), 0).await.unwrap();
  w2.sync(&dwb).await.unwrap();
  let img = read_img(&wp);
  let data_img = read_img(&dp);
  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();

  let root2 = tmp_dir("full_verify");
  let (data3, dwb3, ..) = boot_from(&root2, "crash", &data_img, &img).await;
  assert_eq!(DwbWriter::recover(&dwb3, &data3, 0).await.unwrap(), 1);
  let got = page_bytes(&data3, 0).await;
  assert!(got.iter().all(|&x| x == 0x77));
  drop(data3);
  drop(dwb3);
  remove_dir_all(&root2).unwrap();
}

/// 独立构造「旧批记录 + 新批记录」镜像，验证非首槽的 generation 差异足以停扫。
#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn crafted_generation_is_the_batch_discriminator() {
  let root = tmp_dir("crafted");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_img(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  stage_batch(&dwb, &[0xB0, 0xB1], &mut w).await;
  w.sync(&dwb).await.unwrap();
  let mut img = read_img(&wp);

  // 槽 0：页 0xC0、新批代号 7；槽 1：保留旧批记录（代号 patch 为 9）
  img[DWB_HEADER..DWB_RECORD].fill(0xC0);
  patch_generation(&mut img[..DWB_RECORD], 7);
  patch_generation(&mut img[DWB_RECORD..2 * DWB_RECORD], 9);

  let (data2, dwb2, ..) = boot_from(&root, "crash", &old_image(), &img).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 1);
  let p0 = page_bytes(&data2, 0).await;
  let p1 = page_bytes(&data2, 1).await;
  assert!(p0.iter().all(|&x| x == 0xC0));
  assert!(p1.iter().all(|&x| x == OLD), "跨 generation 的旧槽位被回放");
  drop(data2);
  drop(dwb2);

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}

/// 随机状态组合（1000 例）：DWB 有效前缀 k + 撕裂/损坏尾记录，最终位置
/// 只允许在已 fsync 的前 k 页上出现旧/半新/全新。恢复必须恰好回放 k 页，
/// 且每页收敛到整页新值；再次恢复幂等。
#[compio::test]
#[ignore = "随机 1000 状态组合；--run-ignored 验证"]
async fn randomized_crash_states_never_tear() {
  let root = tmp_dir("random");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_img(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  let new = new_vals();
  stage_batch(&dwb, &new, &mut w).await;
  w.sync(&dwb).await.unwrap();
  let staged = read_img(&wp);
  let rec = 4160usize;
  let mut x = 0x9E37_79B9_7F4A_7C15u64;

  for case in 0..1000usize {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let k = (x % 5) as usize; // 0..=4 条有效记录
    let tail = ((x >> 8) % 3) as u8;
    let mut img = staged[..k * rec].to_vec();
    if k < NPAGES {
      match tail {
        0 => {}
        1 => {
          let cut = ((x >> 16) as usize % (rec - 1)) + 1;
          img.extend_from_slice(&staged[k * rec..k * rec + cut]);
        }
        _ => {
          let mut next = staged[k * rec..(k + 1) * rec].to_vec();
          next[DWB_HEADER + 7] ^= 0x5A;
          img.extend_from_slice(&next);
        }
      }
    }

    // 最终位置：f <= k 页可能已写（半页/整页），其余仍是旧值
    let f = ((x >> 24) as usize) % (k + 1);
    let mut data_img = old_image();
    for p in 0..f {
      let val = page(new[p]);
      if ((x >> (32 + p)) & 1) == 0 {
        data_img[p * PAGE_SIZE..(p + 1) * PAGE_SIZE].copy_from_slice(&val);
      } else {
        data_img[p * PAGE_SIZE..p * PAGE_SIZE + PAGE_SIZE / 2]
          .copy_from_slice(&val[..PAGE_SIZE / 2]);
      }
    }

    let (data2, dwb2, ..) = boot_from(&root, &format!("case{case}"), &data_img, &img).await;
    let applied = DwbWriter::recover(&dwb2, &data2, 0).await.unwrap();
    assert_eq!(applied, k as u64, "case {case}: k={k} tail={tail}");
    assert_prefix(&data2, &new, k).await;
    assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
    drop(data2);
    drop(dwb2);
  }

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}
