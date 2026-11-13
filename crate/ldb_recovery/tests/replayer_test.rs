//! Replayer 崩溃恢复回归：检查点装载 / WAL 段布局重放 / 事务 FIFO 配对 /
//! 崩溃残留前缀一致性 / 幂等性 / 真实文件后端端到端。
//!
//! 载荷一律用字面量（反模式 6：禁自我参照断言）。帧构造直接调 `ldb_wal::entry`
//! 冻结写头/封 CRC —— 与生产落盘同源。

use std::{env, fs, process};

use ldb_io::{CompioFileBackend, MemBackend};
use ldb_pack::Pid2Offset;
use ldb_recovery::{Error, Replayer};
use ldb_wal::{CommitVariant, EntryType, LogManager, WalFileState, entry};

/// 4KiB 段对齐字面量（上游 BLK_BLOCK_SIZE；`ldb_wal::WAL_WRITE_ALIGN`）
const SEG: usize = 4096;

fn mgr() -> LogManager {
  LogManager::new(
    1,
    64 * 1024,
    CommitVariant::WorkersWriteLog,
    Box::new(MemBackend::new()),
    WalFileState {
      start: 0,
      end: 1 << 20,
      shard: 1 << 20,
    },
  )
}

/// 24B 元条目帧（TX_START/TX_COMMIT/TX_ABORT）
fn meta(ty: EntryType, lsn: u64) -> Vec<u8> {
  let mut e = vec![0u8; entry::LOG_ENTRY_HDR];
  entry::write_hdr(&mut e, ty, lsn, entry::LOG_ENTRY_HDR as u64);
  entry::seal_chksum(&mut e);
  e
}

/// 数据条目帧（DATA/FREE_PAGE/REUSE_PAGE；pid@24）
fn data(ty: EntryType, lsn: u64, pid: u64, payload: &[u8]) -> Vec<u8> {
  let total = entry::DATA_ENTRY_HDR + payload.len();
  let mut e = vec![0u8; total];
  entry::write_data_hdr(&mut e, lsn, total as u64, pid, 0);
  e[entry::off::TYPE] = ty.to_u8();
  e[entry::DATA_ENTRY_HDR..].copy_from_slice(payload);
  entry::seal_chksum(&mut e);
  e
}

/// 续接帧到 4KiB 边界（段尾零填充，ldb_wal 落盘段布局）
fn segment(buf: &mut Vec<u8>) {
  let p = buf.len().div_ceil(SEG) * SEG;
  buf.resize(p, 0);
}

#[compio::test]
async fn checkpoint_roundtrip() {
  let mut r = Replayer::new();
  // [len=3][packed...]；packed = (offset << 15) | comp_sz
  let mut buf = Vec::new();
  buf.extend_from_slice(&3u64.to_le_bytes());
  for i in 0..3u64 {
    let packed = ((i * 4096) << 15) | 100;
    buf.extend_from_slice(&packed.to_le_bytes());
  }
  let n = r.recover_from_checkpoint(&buf).await.unwrap();
  assert_eq!(n, 3);
  assert_eq!(r.pid2offset[1].get_offset(), 4096);
  assert_eq!(r.pid2offset[2].get_comp_sz(), 100);

  // A2 差分：1e5 条随机 (offset, comp_sz) 位域往返，期望值由独立位运算给出
  // （不借 Pid2Offset::get_* 自我参照 —— 反模式 6）
  const N: usize = 100_000;
  let mut state = 0x9E37_79B9_7F4A_7C15u64; // SplitMix64 常量，确定性
  let mut next = || {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  };
  let mut big = Vec::with_capacity(8 + N * 8);
  big.extend_from_slice(&(N as u64).to_le_bytes());
  let mut expect: Vec<(u64, u16)> = Vec::with_capacity(N);
  for _ in 0..N {
    let rv = next();
    let off = rv & ((1u64 << 49) - 1); // 49-bit
    let mut sz = ((rv >> 49) & ((1 << 15) - 1)) as u16; // 15-bit
    if sz == 0 {
      sz = 1;
    }
    let packed = (off << 15) | u64::from(sz);
    big.extend_from_slice(&packed.to_le_bytes());
    expect.push((off, sz));
  }
  let mut br = Replayer::new();
  let bn = br.recover_from_checkpoint(&big).await.unwrap();
  assert_eq!(bn, N as u64);
  let mut mismatch = 0usize;
  for (i, (off, sz)) in expect.iter().enumerate() {
    if br.pid2offset[i].get_offset() != *off || br.pid2offset[i].get_comp_sz() != *sz {
      mismatch += 1;
    }
  }
  assert_eq!(mismatch, 0, "1e5 位域往返 MISMATCH=0");
}

#[compio::test]
async fn checkpoint_rejects_truncated_and_overflow() {
  let mut r = Replayer::new();
  // 头都不够
  assert!(matches!(
    r.recover_from_checkpoint(&[0u8; 4]).await,
    Err(Error::CheckpointCorrupt)
  ));
  // 声明 len 超实际
  let mut bad = Vec::new();
  bad.extend_from_slice(&100u64.to_le_bytes());
  bad.extend_from_slice(&[0u8; 8]);
  assert!(matches!(
    r.recover_from_checkpoint(&bad).await,
    Err(Error::CheckpointCorrupt)
  ));
  // len = u64::MAX → n*8 溢出须报错而非回绕（checked 乘加）
  let mut huge = Vec::new();
  huge.extend_from_slice(&u64::MAX.to_le_bytes());
  assert!(matches!(
    r.recover_from_checkpoint(&huge).await,
    Err(Error::CheckpointCorrupt)
  ));
}

#[compio::test]
async fn replay_data_entry_and_undo_uncommitted() {
  let m = mgr();
  let mut wal = Vec::new();
  wal.extend_from_slice(&meta(EntryType::TxStart, 1));
  wal.extend_from_slice(&data(EntryType::DataEntry, 2, 7, b"redo"));
  wal.extend_from_slice(&meta(EntryType::TxCommit, 3));
  wal.extend_from_slice(&meta(EntryType::TxStart, 4)); // 崩溃：无 commit

  let mut r = Replayer::new();
  let frames = r.replay_wal_shard(&wal, &m).await.unwrap();
  assert_eq!(frames, 4);
  assert_eq!(r.redone, 1);
  assert_eq!(r.pid2offset.len(), 8, "pid=7 → 槽数 8");
  assert_eq!(r.active_txns.iter().copied().collect::<Vec<_>>(), vec![4]);
  assert_eq!(r.undo_active_txns().await.unwrap(), 1);
  assert!(r.active_txns.is_empty());
}

/// FIFO：TX_COMMIT 消灭**最早**开启的 TX_START（lsn 最小者），非最近者。
/// 若把配对写成 LIFO/等号匹配，残留活跃集会是 {10} 而非 {20}。
#[compio::test]
async fn replay_txn_pairs_earliest_start_fifo() {
  let m = mgr();
  let mut wal = Vec::new();
  wal.extend_from_slice(&meta(EntryType::TxStart, 10));
  wal.extend_from_slice(&meta(EntryType::TxStart, 20));
  wal.extend_from_slice(&meta(EntryType::TxCommit, 30));
  let mut r = Replayer::new();
  r.replay_wal_shard(&wal, &m).await.unwrap();
  assert_eq!(
    r.active_txns.iter().copied().collect::<Vec<_>>(),
    vec![20],
    "提交应消灭最早开启者 10，残留 20"
  );
}

/// TX_ABORT 亦配对最老者（S1）；重复 TX_COMMIT（活跃集空）静默忽略（S2）。
/// 唯一击杀：M22（TxAbort 不参与配对）。
#[compio::test]
async fn replay_abort_pairs_and_duplicate_commit_ignored() {
  let m = mgr();
  // S1：仅一个 START + 一个 ABORT → 配对后活跃集必空
  let s1 = {
    let mut w = Vec::new();
    w.extend_from_slice(&meta(EntryType::TxStart, 5));
    w.extend_from_slice(&meta(EntryType::TxAbort, 6));
    w
  };
  let mut r1 = Replayer::new();
  r1.replay_wal_shard(&s1, &m).await.unwrap();
  assert!(r1.active_txns.is_empty(), "TX_ABORT 应配对并清空");
  // S2：多余 TX_COMMIT（活跃集空）忽略，不 panic
  let s2 = {
    let mut w = Vec::new();
    w.extend_from_slice(&meta(EntryType::TxStart, 20));
    w.extend_from_slice(&meta(EntryType::TxCommit, 21));
    w.extend_from_slice(&meta(EntryType::TxCommit, 22));
    w
  };
  let mut r2 = Replayer::new();
  r2.replay_wal_shard(&s2, &m).await.unwrap();
  assert!(r2.active_txns.is_empty());
}

/// 零填充（段尾对齐补零）→ 跳下一 4KiB 边界继续，第二个段的条目必须被重放。
/// 若实现把零填充当终止（或当条目），第二段 2 条 DATA 会漏掉。
#[compio::test]
async fn replay_zero_fill_skips_to_next_segment() {
  let m = mgr();
  let mut wal = Vec::new();
  wal.extend_from_slice(&data(EntryType::DataEntry, 1, 3, b"a"));
  segment(&mut wal); // 段 1 尾零填充到 4KiB
  wal.extend_from_slice(&data(EntryType::DataEntry, 2, 4, b"b"));
  wal.extend_from_slice(&data(EntryType::DataEntry, 3, 5, b"c"));
  segment(&mut wal);

  let mut r = Replayer::new();
  let frames = r.replay_wal_shard(&wal, &m).await.unwrap();
  assert_eq!(frames, 3, "跨零填充两段共 3 条");
  assert_eq!(r.redone, 3);
  assert_eq!(r.pid2offset.len(), 6, "最大 pid=5 → 槽数 6");
}

/// CRC 损坏帧 = 前缀终止：损坏点之后的条目一律不重放。
#[compio::test]
async fn replay_crc_corruption_prefix_terminates() {
  let m = mgr();
  let mut wal = Vec::new();
  wal.extend_from_slice(&data(EntryType::DataEntry, 1, 3, b"a"));
  let second = data(EntryType::DataEntry, 2, 4, b"bbbb");
  let at = wal.len();
  wal.extend_from_slice(&second);
  wal[at + entry::DATA_ENTRY_HDR] ^= 0xFF; // 破坏第二帧 payload
  wal.extend_from_slice(&data(EntryType::DataEntry, 3, 5, b"c"));

  let mut r = Replayer::new();
  let frames = r.replay_wal_shard(&wal, &m).await.unwrap();
  assert_eq!(frames, 1, "首帧有效，第二帧 CRC 失败即停");
  assert_eq!(r.redone, 1);
}

/// 半条目（尾截断）：>0 但 pos+size 越界 → 前缀终止，已完整者保留。
#[compio::test]
async fn replay_torn_tail_prefix_terminates() {
  let m = mgr();
  let mut wal = Vec::new();
  wal.extend_from_slice(&data(EntryType::DataEntry, 1, 3, b"a"));
  wal.extend_from_slice(&data(EntryType::DataEntry, 2, 4, b"bbbb"));
  wal.truncate(wal.len() - 3); // 尾 3 字节丢失 → 半条目
  let mut r = Replayer::new();
  let frames = r.replay_wal_shard(&wal, &m).await.unwrap();
  assert_eq!(frames, 1);
  assert_eq!(r.redone, 1);
}

/// 结构不可满足的数据帧（type=DATA 但 size < DATA_ENTRY_HDR，CRC 仍通过）
/// → 显式 WalCorrupt，而非越界读 panic。
#[compio::test]
async fn replay_short_data_frame_errors() {
  let m = mgr();
  // 伪造：type=DataEntry、size=24、CRC 覆盖 [4,24)
  let mut e = vec![0u8; entry::LOG_ENTRY_HDR];
  entry::write_hdr(&mut e, EntryType::DataEntry, 1, entry::LOG_ENTRY_HDR as u64);
  entry::seal_chksum(&mut e);
  let mut r = Replayer::new();
  assert!(matches!(
    r.replay_wal_shard(&e, &m).await,
    Err(Error::WalCorrupt(1))
  ));
}

/// 乱序：物理序为 [高 lsn 段][低 lsn 段]（模拟 wal_offset 反向分配 + 环回），
/// 重放须按 lsn 升序 —— COMMIT(200) 应在 START(10) 之后配对。
/// 若不排序，COMMIT 先到且活跃集空 → 被忽略，残留 {10}。
#[compio::test]
async fn replay_out_of_order_segments_sorted_by_lsn() {
  let m = mgr();
  // 物理序：COMMIT(200) 先出现，其后才是 START(10)/DATA(30)
  let mut wal = Vec::new();
  wal.extend_from_slice(&meta(EntryType::TxCommit, 200));
  wal.extend_from_slice(&meta(EntryType::TxStart, 10));
  wal.extend_from_slice(&data(EntryType::DataEntry, 30, 9, b"x"));

  let mut r = Replayer::new();
  assert_eq!(r.replay_wal_shard(&wal, &m).await.unwrap(), 3);
  assert!(
    r.active_txns.is_empty(),
    "按 lsn 排序后 COMMIT(200) 消灭 START(10)"
  );
  assert_eq!(r.redone, 1);
}

/// 映射表合并：redo 建槽后，检查点基线填充空槽（非空优先）。
/// 唯一击杀：M24（restore 条件取反）。
#[compio::test]
async fn redo_and_restore_mapping_merge() {
  let mut r = Replayer::new();
  r.redo_page_entry(1, 5).await.unwrap();
  assert_eq!(r.pid2offset.len(), 6);
  let mut base = vec![Pid2Offset::default(); 3];
  base[2].set(8192, 55); // 检查点基线 pid2
  let n = r.restore_pid2offset_table(base).await.unwrap();
  assert!(n >= 6);
  assert_eq!(r.pid2offset[2].get_offset(), 8192, "空槽用检查点基线填充");
}

// ---------------------------------------------------------------------------
// 真实文件后端端到端（ldb_wal 冻结落盘实现）
// ---------------------------------------------------------------------------

/// 一条待写入条目（期望回读的 (type, pid, payload)）
struct Expect {
  ty: EntryType,
  pid: u64,
  payload: Vec<u8>,
}

/// 用真实 CompioFileBackend 生成 WAL → 关闭 → 重开 → 读回 [frontier, end)
/// → 逐条解析断言与写入序列一致 → Replayer 重放。
async fn run_e2e(name: &str, ops: &[Expect]) -> (Vec<(EntryType, u64, Vec<u8>)>, Replayer) {
  let end: u64 = 4 * 1024 * 1024;
  let dir = env::temp_dir().join(format!("ldb_recovery_e2e_{}_{}", process::id(), name));
  fs::create_dir_all(&dir).unwrap();
  let wal_path = dir.join("wal.log");
  let backend = CompioFileBackend::open_direct_or_buffered(wal_path.to_str().unwrap())
    .await
    .unwrap();
  let m = LogManager::new(
    1,
    256 * 1024,
    CommitVariant::WorkersWriteLog,
    Box::new(backend),
    WalFileState {
      start: 0,
      end,
      shard: end,
    },
  );
  let w = m.local_log_worker(0).clone();
  for op in ops {
    w.reserve_wal_entry(&m, op.ty, op.pid, &op.payload)
      .await
      .unwrap();
    w.submit_active_log_entry(&m, 0).await.unwrap();
  }
  w.persist_log(&m, ldb_wal::WAL_WRITE_ALIGN).await.unwrap();
  let frontier = m.wal_file_offset();
  // 落盘段长须 4KiB 对齐（ldb_wal 段尾零填充契约；跨 crate 变异 M_e2e 即破坏此项）
  assert_eq!((end - frontier) % 4096, 0, "落盘段长须 4KiB 对齐");
  m.shutdown();
  let _sink = m.sink.borrow_mut().take(); // 关句柄

  // 重开同一文件回读
  let reader = CompioFileBackend::open_direct_or_buffered(wal_path.to_str().unwrap())
    .await
    .unwrap();
  let len = (end - frontier) as usize;
  let raw = reader
    .read_bytes_owned(frontier, vec![0u8; len])
    .await
    .unwrap()
    .to_vec();

  // 逐条解析：零填充跳 4KiB + CRC 前缀终止
  let mut parsed: Vec<(EntryType, u64, Vec<u8>)> = Vec::new();
  let mut pos = 0usize;
  while pos + entry::LOG_ENTRY_HDR <= raw.len() {
    let Ok(hdr) = entry::read_hdr(&raw[pos..]) else {
      break;
    };
    if hdr.size == 0 {
      pos += SEG - pos % SEG;
      continue;
    }
    let sz = hdr.size as usize;
    if sz < entry::LOG_ENTRY_HDR || pos + sz > raw.len() {
      break;
    }
    let frame = &raw[pos..pos + sz];
    assert!(entry::validate_chksum(frame).is_ok(), "第 {pos} 帧 CRC");
    let (ty, pid, pl) = match hdr.ty {
      EntryType::DataEntry | EntryType::FreePage | EntryType::ReusePage => {
        let (pid, _) = entry::read_data_fields(frame);
        (hdr.ty, pid, frame[entry::DATA_ENTRY_HDR..].to_vec())
      }
      EntryType::FreeExtent | EntryType::ReuseExtent => {
        // 48B 布局：start_pid@24 lp_size@32；回读 lp_size 验证往返
        let (start_pid, _) = entry::read_data_fields(frame);
        (
          hdr.ty,
          start_pid,
          frame[entry::off::LP_SIZE..entry::off::LP_SIZE + 8].to_vec(),
        )
      }
      other => (other, 0, Vec::new()),
    };
    parsed.push((ty, pid, pl));
    pos += sz;
  }
  // 段尾零填充
  assert!(
    raw[pos.min(raw.len())..].iter().all(|&b| b == 0),
    "条目链终点后应全零"
  );
  // 逐条与写入序列一致（type + pid + payload）
  let want: Vec<(EntryType, u64, Vec<u8>)> = ops
    .iter()
    .map(|o| (o.ty, o.pid, o.payload.clone()))
    .collect();
  assert_eq!(parsed, want, "回读序列 == 关闭前写入序列");

  let mut r = Replayer::new();
  r.replay_wal_shard(&raw, &m).await.unwrap();
  // 幂等：另一个 Replayer 重放同一段字节 → 映射一致；同一实例重复重放映射不变
  let mut r2 = Replayer::new();
  r2.replay_wal_shard(&raw, &m).await.unwrap();
  assert_eq!(r2.pid2offset, r.pid2offset, "重复重放映射一致（幂等）");
  let snap = r2.pid2offset.clone();
  r2.replay_wal_shard(&raw, &m).await.unwrap();
  assert_eq!(r2.pid2offset, snap, "同一实例重复重放映射不变");
  fs::remove_dir_all(&dir).ok();
  (parsed, r)
}

/// 真实文件后端端到端（4 种负载）：写 → 关闭 → 重开回读逐字节一致 → 重放。
/// 该集成用例的唯一击杀物是 **ldb_wal 写入侧段布局**（`write_log_buffer` 的
/// 段尾零填充）——见报告 §E1 的跨 crate 变异 M_e2e。
#[compio::test]
async fn e2e_file_backend_recovery() {
  // 负载 A：纯插入（12 条 DATA）
  let a: Vec<Expect> = (0..12u64)
    .map(|i| Expect {
      ty: EntryType::DataEntry,
      pid: i,
      payload: format!("ins-{i}").into_bytes(),
    })
    .collect();
  let (parsed, r) = run_e2e("pure_insert", &a).await;
  assert_eq!(parsed.len(), 12);
  assert_eq!(r.redone, 12);
  assert_eq!(r.pid2offset.len(), 12);
  assert!(r.active_txns.is_empty());

  // 负载 B：更新 + 删除（8 DATA + 4 FREE_PAGE）
  let mut b = Vec::new();
  for i in 0..8u64 {
    b.push(Expect {
      ty: EntryType::DataEntry,
      pid: i,
      payload: format!("upd-{i}").into_bytes(),
    });
  }
  for i in 3..7u64 {
    b.push(Expect {
      ty: EntryType::FreePage,
      pid: i,
      payload: Vec::new(),
    });
  }
  let (parsed, r) = run_e2e("update_delete", &b).await;
  assert_eq!(parsed.len(), 12);
  assert_eq!(r.redone, 12);
  assert_eq!(r.pid2offset.len(), 8, "最大 pid=7");

  // 负载 C：混合 + 映射变更（TX_START/DATA/TX_COMMIT/FREE_EXTENT），一个未提交事务
  let c = vec![
    Expect {
      ty: EntryType::TxStart,
      pid: 0,
      payload: Vec::new(),
    },
    Expect {
      ty: EntryType::DataEntry,
      pid: 2,
      payload: 4096u64.to_le_bytes().to_vec(),
    },
    Expect {
      ty: EntryType::DataEntry,
      pid: 5,
      payload: 8192u64.to_le_bytes().to_vec(),
    },
    Expect {
      ty: EntryType::TxCommit,
      pid: 0,
      payload: Vec::new(),
    },
    Expect {
      ty: EntryType::TxStart,
      pid: 0,
      payload: Vec::new(),
    },
    Expect {
      ty: EntryType::FreeExtent,
      pid: 9,
      payload: 4u64.to_le_bytes().to_vec(),
    },
  ];
  let (parsed, mut r) = run_e2e("mixed_map", &c).await;
  assert_eq!(parsed.len(), 6);
  assert_eq!(r.redone, 3, "2 DATA + 1 FREE_EXTENT 重放");
  assert_eq!(r.pid2offset.len(), 10, "最大 start_pid=9");
  assert_eq!(
    r.undo_active_txns().await.unwrap(),
    1,
    "第二个 TX_START 未提交"
  );

  // 负载 D：FREE/REUSE 页 + 页/区段释放（pid-only 语义，不依赖 lp_size）
  let d = vec![
    Expect {
      ty: EntryType::FreePage,
      pid: 3,
      payload: Vec::new(),
    },
    Expect {
      ty: EntryType::ReusePage,
      pid: 4,
      payload: Vec::new(),
    },
    Expect {
      ty: EntryType::FreeExtent,
      pid: 9,
      payload: 4u64.to_le_bytes().to_vec(),
    },
    Expect {
      ty: EntryType::ReuseExtent,
      pid: 20,
      payload: 8u64.to_le_bytes().to_vec(),
    },
  ];
  let (parsed, r) = run_e2e("free_reuse", &d).await;
  assert_eq!(parsed.len(), 4);
  assert_eq!(r.redone, 4);
  assert_eq!(r.pid2offset.len(), 21, "最大 start_pid=20");
}
