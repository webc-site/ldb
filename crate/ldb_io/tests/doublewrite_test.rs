//! Doublewrite：上游偏移语义 + 本项目崩溃安全记录日志。
//!
//! 崩溃点用「文件快照」模拟：DWB 未 fsync 时把快照取为旧内容/撕裂前缀；
//! DWB 已 fsync 时取完整记录；最终位置可整页旧、半页新或整页新。恢复后
//! 每页必须要么全旧、要么全新，不许混合。

use std::{
  env::temp_dir,
  fs::{create_dir_all, read, remove_dir_all, write},
  io,
  path::{Path, PathBuf},
  process::id,
  sync::atomic::{AtomicU64, Ordering},
};

use ldb_io::{
  CompioFileBackend, DWB_AREA, DWB_RECORD, Doublewrite, DwbWriter, doublewrite::DWB_RECORDS,
};
use ldb_type::PAGE_SIZE;

const NPAGES: usize = 4;
const OLD: u8 = 0xAA;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_dir(tag: &str) -> PathBuf {
  let dir = temp_dir().join(format!(
    "ldb_io_dwb_{tag}_{}_{}",
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

/// 唯一安全入口：先 recover 再返回可写句柄（空区回放 0 页）
async fn new_writer(dwb: &CompioFileBackend, data: &CompioFileBackend) -> DwbWriter {
  let (w, applied) = DwbWriter::open(dwb, data, 0).await.unwrap();
  assert_eq!(applied, 0);
  w
}

fn page(val: u8) -> Vec<u8> {
  vec![val; PAGE_SIZE]
}

fn old_image() -> Vec<u8> {
  vec![OLD; NPAGES * PAGE_SIZE]
}

fn write_bytes(path: &Path, bytes: &[u8]) {
  write(path, bytes).unwrap();
}

fn read_bytes(path: &Path) -> Vec<u8> {
  read(path).unwrap()
}

async fn assert_pages(data: &CompioFileBackend, want: &[u8]) {
  assert_eq!(want.len(), NPAGES);
  for (p, &val) in want.iter().enumerate() {
    let got = data
      .read_owned(p as u64, vec![0u8; PAGE_SIZE])
      .await
      .unwrap();
    assert!(
      got.iter().all(|&x| x == val),
      "页 {p} 期望全 {val:#04x}，实际首字节 {:#04x}",
      got[0]
    );
  }
}

/// 用给定镜像建一对新文件并打开（模拟掉电后的磁盘内容）
async fn crash_open(
  root: &Path,
  tag: &str,
  data_img: &[u8],
  dwb_img: &[u8],
) -> (CompioFileBackend, CompioFileBackend, PathBuf, PathBuf) {
  let dir = root.join(tag);
  create_dir_all(&dir).unwrap();
  let dp = dir.join("data.db");
  let wp = dir.join("dwb.dat");
  write_bytes(&dp, data_img);
  write_bytes(&wp, dwb_img);
  (open(&dp).await, open_dwb(&wp).await, dp, wp)
}

#[test]
fn doublewrite_cursor_matches_upstream() {
  let pages = Doublewrite::capacity_pages();
  // 8 MiB / 4 KiB = 2048（字面量期望，禁止用被测常量自证）
  assert_eq!(pages, 2048);

  let mut dwb = Doublewrite::new(0);
  assert_eq!(dwb.cnt, 0);
  assert_eq!(dwb.dwb_cur_w_offset, 0);

  // 推进 5 次：cnt=5, cur=5*PAGE_SIZE（字面量期望，禁止自我参照）
  for _ in 0..5 {
    dwb.advance_write_offset();
  }
  assert_eq!(dwb.cnt, 5);
  assert_eq!(dwb.dwb_cur_w_offset, 5 * PAGE_SIZE);

  // 上游 SetWriteOffset 只改 start/end，保留 cnt/cur（doublewrite.cc:20-25）
  let base = 100 * PAGE_SIZE;
  dwb.set_write_offset(base);
  assert_eq!(dwb.dwb_start_offset, base);
  assert_eq!(dwb.dwb_end_offset, base + DWB_AREA);
  assert_eq!(dwb.cnt, 5);
  assert_eq!(dwb.dwb_cur_w_offset, 5 * PAGE_SIZE);

  // 下一次推进落到新区的同一 cnt 槽
  dwb.advance_write_offset();
  assert_eq!(dwb.cnt, 6);
  assert_eq!(dwb.dwb_cur_w_offset, base + 6 * PAGE_SIZE);

  // 环形回卷：从 0 推进 2048（8 MiB/4 KiB）次回到 0
  let mut ring = Doublewrite::new(0);
  for _ in 0..2048 {
    ring.advance_write_offset();
  }
  assert_eq!(ring.cnt, 0);
  assert_eq!(ring.dwb_cur_w_offset, 0);
}

/// 上游 `doublewrite.cc:20-39` 的逐字模型对拍：10 万步随机 set/advance。
#[test]
fn doublewrite_matches_upstream_model_1e5() {
  let mut dwb = Doublewrite::new(0);
  // 模型字段顺序：(cnt, cur, start, end)
  let mut model = (0usize, 0usize, 0usize, DWB_AREA);
  let mut x = 0x1234_5678_9abc_def0u64;
  for step in 0..100_000 {
    x = x
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    if x % 7 == 0 {
      // 区起点按 8 MiB 对齐（与 worker_thread_id * MB * 8 一致）
      let base = ((x >> 8) as usize % 16) * DWB_AREA;
      dwb.set_write_offset(base);
      model.2 = base;
      model.3 = base + DWB_AREA;
      // 上游 SetWriteOffset 不动 cnt/cur
    } else {
      dwb.advance_write_offset();
      // 上游 MB*8/PAGE_SIZE = 2048，写字面量避免与实现常量自我参照
      model.0 = if model.0 == 2047 { 0 } else { model.0 + 1 };
      model.1 = model.2 + model.0 * PAGE_SIZE;
    }
    assert_eq!(
      (
        dwb.cnt,
        dwb.dwb_cur_w_offset,
        dwb.dwb_start_offset,
        dwb.dwb_end_offset
      ),
      model,
      "step {step}"
    );
  }
}

#[compio::test]
async fn dwb_before_sync_leaves_old_data() {
  let root = tmp_dir("before_sync");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_bytes(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let mut w = new_writer(&dwb, &data).await;
  for p in 0..NPAGES {
    w.write_page(&dwb, &page(0xB0 + p as u8), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }

  // 掉电发生在 DWB fsync 之前：记录可能全丢，也可能只剩撕裂前缀
  let staged = read_bytes(&wp);
  for (tag, img) in [("empty", Vec::new()), ("torn", staged[..1000].to_vec())] {
    let (data2, dwb2, ..) = crash_open(&root, tag, &old_image(), &img).await;
    assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
    assert_pages(&data2, &[OLD; NPAGES]).await;
  }

  // 撕成整长但内容损坏（CRC 必须拒绝）
  let mut corrupt = staged;
  corrupt[1000] ^= 0xFF;
  let (data2, dwb2, ..) = crash_open(&root, "corrupt_pre", &old_image(), &corrupt).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
  assert_pages(&data2, &[OLD; NPAGES]).await;
  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root).unwrap();
}

#[compio::test]
async fn dwb_after_sync_replays_whole_batch() {
  let root = tmp_dir("after_sync");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_bytes(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let mut w = new_writer(&dwb, &data).await;
  for p in 0..NPAGES {
    w.write_page(&dwb, &page(0xB0 + p as u8), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  w.sync(&dwb).await.unwrap();

  // 崩溃：DWB 已持久化，最终位置仍是旧页
  let (data2, dwb2, ..) = crash_open(&root, "crash", &read_bytes(&dp), &read_bytes(&wp)).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 4);
  assert_pages(&data2, &[0xB0, 0xB1, 0xB2, 0xB3]).await;

  // 恢复后区首头已失效：重复恢复幂等
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 0);
  assert_pages(&data2, &[0xB0, 0xB1, 0xB2, 0xB3]).await;
  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root).unwrap();
}

#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn dwb_torn_final_page_is_repaired() {
  let root = tmp_dir("torn_final");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_bytes(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let mut w = new_writer(&dwb, &data).await;
  for p in 0..NPAGES {
    w.write_page(&dwb, &page(0xB0 + p as u8), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  w.sync(&dwb).await.unwrap();

  // 页 1 只写了前 2048 字节（半页写），其余页还是旧内容
  let mut mixed = old_image();
  let new1 = page(0xB1);
  mixed[PAGE_SIZE..PAGE_SIZE + PAGE_SIZE / 2].copy_from_slice(&new1[..PAGE_SIZE / 2]);
  let (data2, dwb2, ..) = crash_open(&root, "crash", &mixed, &read_bytes(&wp)).await;

  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 4);
  assert_pages(&data2, &[0xB0, 0xB1, 0xB2, 0xB3]).await;
  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root).unwrap();
}

#[compio::test]
async fn dwb_rejects_corrupt_record_and_stops_scan() {
  let root = tmp_dir("corrupt_rec");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_bytes(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let mut w = new_writer(&dwb, &data).await;
  for p in 0..3 {
    w.write_page(&dwb, &page(0xB0 + p as u8), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  w.sync(&dwb).await.unwrap();

  // 第二条记录的数据区翻转一位：CRC 必须拒绝，只回放第一条
  let mut img = read_bytes(&wp);
  img[DWB_RECORD + 64 + 10] ^= 0xFF;
  let (data2, dwb2, ..) = crash_open(&root, "crash", &old_image(), &img).await;

  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 1);
  assert_pages(&data2, &[0xB0, OLD, OLD, OLD]).await;
  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root).unwrap();
}

#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn dwb_generation_stops_stale_records() {
  let root = tmp_dir("generation");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_bytes(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let mut w = new_writer(&dwb, &data).await;
  // 批 1：写页 0/1 并 checkpoint
  for p in 0..2 {
    w.write_page(&dwb, &page(0xB0 + p as u8), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  w.sync(&dwb).await.unwrap();
  data.write_owned(page(0xB0), 0).await.unwrap();
  data
    .write_owned(page(0xB1), PAGE_SIZE as u64)
    .await
    .unwrap();
  data.fdatasync().await.unwrap();
  w.checkpoint(&dwb, &data).await.unwrap();

  // 批 1 之后页 1 又被更新为新内容（不经 DWB 的独立写，模拟更晚的落盘）
  data
    .write_owned(page(0xDD), PAGE_SIZE as u64)
    .await
    .unwrap();
  data.fdatasync().await.unwrap();

  // 批 2：只写页 0 到区首（覆盖旧批第一条记录）；旧批第二条记录仍在槽 1
  w.write_page(&dwb, &page(0xC0), 0).await.unwrap();

  let (data2, dwb2, ..) = crash_open(&root, "crash", &read_bytes(&dp), &read_bytes(&wp)).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 1);
  // 旧批页 1 的记录（generation 不同）不得回放覆盖 0xDD
  assert_pages(&data2, &[0xC0, 0xDD, OLD, OLD]).await;
  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root).unwrap();
}

#[compio::test]
#[cfg_attr(
  target_os = "macos",
  ignore = "macOS F_FULLFSYNC 4–11 ms/次；崩溃矩阵每个状态真实刷盘，--run-ignored 验证"
)]
async fn dwb_checkpoint_resets_and_invalidates() {
  let root = tmp_dir("checkpoint");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_bytes(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let mut w = new_writer(&dwb, &data).await;
  for p in 0..2 {
    w.write_page(&dwb, &page(0xB0 + p as u8), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  w.sync(&dwb).await.unwrap();
  data.write_owned(page(0xB0), 0).await.unwrap();
  data
    .write_owned(page(0xB1), PAGE_SIZE as u64)
    .await
    .unwrap();
  w.checkpoint(&dwb, &data).await.unwrap();

  // checkpoint 后区首头为 0 → 恢复无记录
  assert_eq!(DwbWriter::recover(&dwb, &data, 0).await.unwrap(), 0);

  // 新批从区首开始；若 checkpoint 未重置 cur，这里会写进槽 2，恢复只得 0
  w.write_page(&dwb, &page(0xC0), 0).await.unwrap();
  w.sync(&dwb).await.unwrap();
  let (data2, dwb2, ..) = crash_open(&root, "crash", &read_bytes(&dp), &read_bytes(&wp)).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 1);
  assert_pages(&data2, &[0xC0, 0xB1, OLD, OLD]).await;
  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root).unwrap();
}

#[compio::test]
async fn dwb_torn_tail_is_ignored() {
  let root = tmp_dir("torn_tail");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  write_bytes(&dp, &old_image());
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;
  data.fdatasync().await.unwrap();

  let mut w = new_writer(&dwb, &data).await;
  for p in 0..2 {
    w.write_page(&dwb, &page(0xB0 + p as u8), p as u64 * PAGE_SIZE as u64)
      .await
      .unwrap();
  }
  w.sync(&dwb).await.unwrap();

  let staged = read_bytes(&wp);
  let (data2, dwb2, ..) =
    crash_open(&root, "crash", &old_image(), &staged[..DWB_RECORD + 100]).await;
  assert_eq!(DwbWriter::recover(&dwb2, &data2, 0).await.unwrap(), 1);
  assert_pages(&data2, &[0xB0, OLD, OLD, OLD]).await;
  drop(data);
  drop(dwb);
  drop(data2);
  drop(dwb2);
  remove_dir_all(&root).unwrap();
}

#[compio::test]
#[ignore = "填满 8 MiB DWB 区约 116ms；需要时 --run-ignored 验证"]
async fn dwb_area_full_requires_checkpoint() {
  let root = tmp_dir("full");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;

  let mut w = new_writer(&dwb, &data).await;
  let page = page(1);
  for _ in 0..DWB_RECORDS {
    w.write_page(&dwb, &page, 0).await.unwrap();
  }
  let err = w.write_page(&dwb, &page, 0).await.unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::StorageFull);

  w.checkpoint(&dwb, &data).await.unwrap();
  w.write_page(&dwb, &page, 0).await.unwrap();
  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}

/// 非整页长度 / 非页对齐区起点必须报错，不能被静默截断或按字节误解释。
#[compio::test]
async fn dwb_rejects_partial_page_and_misaligned_area() {
  let root = tmp_dir("validate");
  let dp = root.join("data.db");
  let wp = root.join("dwb.dat");
  let data = open(&dp).await;
  let dwb = open_dwb(&wp).await;

  let err = DwbWriter::open(&dwb, &data, 123).await.unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
  let err = DwbWriter::recover(&dwb, &data, PAGE_SIZE as u64 + 1)
    .await
    .unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

  // DWB 与 data 指向同一文件会互相覆盖，必须拒绝
  let same = open(&dp).await;
  let err = DwbWriter::open(&same, &data, 0).await.unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
  drop(same);

  let (mut w, _) = DwbWriter::open(&dwb, &data, 0).await.unwrap();
  let err = w.write_page(&dwb, &[0u8; 100], 0).await.unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

  drop(data);
  drop(dwb);
  remove_dir_all(&root).unwrap();
}
