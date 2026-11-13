//! 1e6 次操作与 `gxhash::HashMap` 影子模型逐操作对拍（清单 A2 差分 / 本轮重点 1b）。
//!
//! 为什么放在 `benches/`：debug 构建下 1e6 次双向比对约 225 ms，超出单测 100 ms
//! 预算（E2）；release 下约 36 ms。`cargo bench -p ldb_clock` 与 §3 固定执行。
//!
//! 比对口径：`insert` 返回「是否已存在」（= `!model.insert().is_none()`）、
//! `remove` 返回「是否存在」、`contains` 一致；页号空间 = capacity，故常驻数
//! ≤ capacity ≤ count/1.8，`insert` 永不因满表失败，模型无需模拟容量上限。

use std::{
  hint::black_box,
  io::stdout,
  sync::{Arc, atomic::Ordering},
  time::Instant,
};

use gxhash::HashMap;
use ldb_clock::ResidentPageSet;
use ldb_type::{PageState, PageStateMode, same_version_new_state};

fn locked_states(n: u64) -> Arc<[PageState]> {
  (0..n)
    .map(|_| {
      let st = PageState::new();
      st.store(
        same_version_new_state(0, PageStateMode::EXCLUSIVE),
        Ordering::Relaxed,
      );
      st
    })
    .collect::<Vec<_>>()
    .into()
}

fn main() {
  const CAP: u64 = 128;
  const OPS: u64 = 1_000_000;
  let rs = ResidentPageSet::new(CAP, locked_states(CAP), 1);
  let mut model = HashMap::default();
  let mut s = 0x9e37_79b9_7f4a_7c15u64;
  let t = Instant::now();
  let mut checksum = 0u64;
  for op in 0..OPS {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    let pid = (s >> 8) % CAP;
    match s % 10 {
      0..=3 => assert_eq!(
        black_box(rs.insert(pid)),
        model.insert(pid, ()).is_none(),
        "op {op}: insert({pid})"
      ),
      4..=6 => assert_eq!(
        black_box(rs.remove(pid)),
        model.remove(&pid).is_some(),
        "op {op}: remove({pid})"
      ),
      _ => assert_eq!(
        black_box(rs.contain(pid)),
        model.contains_key(&pid),
        "op {op}: contain({pid})"
      ),
    }
    checksum = checksum.wrapping_add(pid);
  }
  for pid in 0..CAP {
    assert_eq!(rs.contain(pid), model.contains_key(&pid), "终态 {pid}");
  }
  use std::io::Write;
  writeln!(
    stdout(),
    "map_differential: MISMATCH=0 ops={OPS} capacity={CAP} count={} live={} \
     checksum={checksum} elapsed={:?}",
    rs.count(),
    model.len(),
    t.elapsed()
  )
  .ok();
}
