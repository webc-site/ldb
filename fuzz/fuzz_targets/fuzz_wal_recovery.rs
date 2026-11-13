//! fuzz_wal_recovery（todo.md §6）：WAL 帧流 + 随机位翻转/尾部截断 →
//! Replayer 重放，验证 CRC 拦截 + 前缀一致性（帧数 ≤ 总帧数、无 panic）。

#![no_main]
use compio::runtime::Runtime;
use ldb_io::MemBackend;
use ldb_recovery::Replayer;
use ldb_wal::{CommitVariant, LogManager, WalFileState, entry};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
  Runtime::new().unwrap().block_on(async {
    // 1) 造合法帧流（若干 TX_START/COMMIT + DataEntry）
    let mut wal: Vec<u8> = Vec::new();
    let mut push = |ty: entry::EntryType, lsn: u64, payload: &[u8]| {
      let total = if matches!(ty, entry::EntryType::TxStart | entry::EntryType::TxCommit) {
        entry::LOG_ENTRY_HDR
      } else {
        entry::DATA_ENTRY_HDR + payload.len()
      };
      let mut e = vec![0u8; total];
      if total == entry::LOG_ENTRY_HDR {
        ldb_wal::write_hdr(&mut e, ty, lsn, total as u64);
      } else {
        entry::write_data_hdr(&mut e, lsn, total as u64, 7, 0);
        e[entry::DATA_ENTRY_HDR..].copy_from_slice(payload);
      }
      entry::seal_chksum(&mut e);
      wal.extend_from_slice(&e);
    };
    push(entry::EntryType::TxStart, 1, b"");
    push(entry::EntryType::DataEntry, 2, b"redo-0");
    push(entry::EntryType::TxCommit, 3, b"");
    push(entry::EntryType::DataEntry, 4, b"redo-1");
    let total_frames = 4u64;

    // 2) 变异：按 fuzz 输入做位翻转 + 截断
    let mut bytes = wal;
    if data.len() >= 3 {
      let flips = (data[0] as usize % 8) + 1;
      for i in 0..flips {
        let pos = (data[1] as usize * 7919 + i * 104729) % bytes.len();
        bytes[pos] ^= data[(i + 2) % data.len()];
      }
      let keep = (data[2] as usize * 331) % (bytes.len() + 1);
      bytes.truncate(keep);
    }

    // 3) 重放：CRC 拦截 + 前缀一致性（帧数 ≤ 总帧数）
    let mgr = LogManager::new(
      2,
      64 * 1024,
      CommitVariant::Baseline,
      Box::new(MemBackend::new()),
      WalFileState {
        start: 0,
        end: 1 << 20,
        shard: 1 << 18,
      },
    );
    let mut rp = Replayer::new();
    let frames = rp.replay_wal_shard(&bytes, &mgr).await.unwrap();
    assert!(frames <= total_frames, "重放帧数越界: {frames}");
    let undone = rp.undo_active_txns().await.unwrap();
    assert!(undone <= total_frames, "回滚计数越界: {undone}");
  });
});
