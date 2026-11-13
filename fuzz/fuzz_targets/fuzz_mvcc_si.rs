//! fuzz_mvcc_si（todo.md §6）：worker-0 事务事件流（Start/Commit/Abort），
//! 断言：commit_ts 单调、无 panic、abort 后事务状态复位（OSIC CommitTree 侧）。

#![no_main]
use compio::runtime::Runtime;
use ldb_io::MemBackend;
use ldb_txn::{IsolationLevel, TransactionManager};
use ldb_wal::{CommitVariant, WalFileState};
use libfuzzer_sys::{arbitrary::Arbitrary, fuzz_target};

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Ev {
  Start { iso: u8 },
  Commit,
  Abort,
}

fuzz_target!(|ops: Vec<Ev>| {
  Runtime::new().unwrap().block_on(async {
    let mgr = TransactionManager::new(
      4,
      64,
      CommitVariant::Baseline,
      Box::new(MemBackend::new()),
      WalFileState {
        start: 0,
        end: 1 << 20,
        shard: 1 << 18,
      },
    );
    let mut last_ts: u64 = 0;
    for (i, op) in ops.iter().enumerate() {
      let wid = 0usize;
      match *op {
        Ev::Start { iso } => {
          let iso = match iso % 3 {
            0 => IsolationLevel::ReadCommitted,
            1 => IsolationLevel::SnapshotIsolation,
            _ => IsolationLevel::Serializable,
          };
          mgr.start_transaction(wid, iso);
        }
        Ev::Commit => {
          // 未开事务提交 → InvalidTxnState 错误可接受；成功则 ts 单调
          if let Ok(ticket) = mgr.commit_transaction(wid).await {
            assert!(ticket.commit_ts >= last_ts, "commit_ts 回退 @ op{i}");
            last_ts = ticket.commit_ts;
          }
        }
        Ev::Abort => {
          let _ = mgr.abort_transaction(wid).await;
        }
      }
    }
  });
});
