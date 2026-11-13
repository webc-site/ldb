//! 并发 B+Tree 回归：CRUD / 分裂 / 扫描 / ContentionSplit / XMerge / 合并不丢键。
//!
//! 结构不变量（父指针 / 叶链 / fence / 路由）由 `common::check_tree` 统一校验。

use ldb_btree::Error;
use ldb_node::node::MAX_RECORD_SIZE;
mod common;

use common::{check_tree, mk};

#[compio::test]
async fn crud_and_lookup() {
  let mut t = mk();
  for i in 0..50u64 {
    t.insert(&format!("k{i:04}").into_bytes(), &i.to_le_bytes())
      .await
      .unwrap();
  }
  assert!(t.is_not_empty());
  let mut out = Vec::new();
  for i in 0..50u64 {
    assert!(
      t.look_up(&format!("k{i:04}").into_bytes(), &mut out)
        .await
        .unwrap()
    );
    assert_eq!(out, i.to_le_bytes());
  }
  assert!(!t.look_up(b"zzz", &mut out).await.unwrap());
  // 更新 / 原地更新 / 删除
  assert!(
    t.update(&b"k0007"[..], &999u64.to_le_bytes())
      .await
      .unwrap()
  );
  assert!(t.look_up(b"k0007", &mut out).await.unwrap());
  assert_eq!(out, 999u64.to_le_bytes());
  assert!(
    t.update_in_place(b"k0008", &12345u64.to_le_bytes())
      .await
      .unwrap()
  );
  assert!(t.remove(b"k0009").await.unwrap());
  assert!(!t.look_up(b"k0009", &mut out).await.unwrap());
  check_tree(&t);
}

#[compio::test]
async fn split_grows_tree() {
  let mut t = mk();
  let n = 200u64;
  for i in 0..n {
    t.insert(&format!("key-{i:05}").into_bytes(), &[0; 24])
      .await
      .unwrap();
  }
  assert!(t.count_pages().await.unwrap() > 1, "200 键必须分裂");
  assert_eq!(t.count_entries().await.unwrap(), n);
  let mut out = Vec::new();
  for i in 0..n {
    assert!(
      t.look_up(&format!("key-{i:05}").into_bytes(), &mut out)
        .await
        .unwrap(),
      "i={i}"
    );
  }
  check_tree(&t);
}

#[compio::test]
async fn scan_ranges() {
  let mut t = mk();
  for i in 0..60u64 {
    t.insert(&format!("s{i:03}").into_bytes(), &i.to_le_bytes())
      .await
      .unwrap();
  }
  let mut seen = Vec::new();
  let n = t
    .scan_ascending(b"s010", b"s020", |k, _| {
      seen.push(String::from_utf8_lossy(k).into_owned());
      true
    })
    .await
    .unwrap();
  assert_eq!(n, 10);
  assert_eq!(seen.first().unwrap(), "s010");
  assert_eq!(seen.last().unwrap(), "s019");
  let mut desc = Vec::new();
  let nd = t
    .scan_descending(b"s020", b"s010", |k, _| {
      desc.push(String::from_utf8_lossy(k).into_owned());
      true
    })
    .await
    .unwrap();
  assert_eq!(nd, 10, "[s010, s020) 半开区间 10 项");
  assert_eq!(desc.first().unwrap(), "s019", "降序首项 s019");
  let total = t
    .iterate_untils(
      &[
        (b"s000".to_vec(), b"s010".to_vec()),
        (b"s050".to_vec(), b"s060".to_vec()),
      ],
      |_, _| true,
    )
    .await
    .unwrap();
  assert_eq!(total, 20);
}

#[compio::test]
async fn merge_reclaims_underfull() {
  let mut t = mk();
  for i in 0..80u64 {
    t.insert(&format!("m{i:03}").into_bytes(), &[0; 8])
      .await
      .unwrap();
  }
  let before = t.count_pages().await.unwrap();
  for i in (0..80u64).step_by(2) {
    t.remove(&format!("m{i:03}").into_bytes()).await.unwrap();
  }
  let after = t.count_pages().await.unwrap();
  assert!(after <= before, "删除后页数不增: {after} vs {before}");
  // 剩余可查
  let mut out = Vec::new();
  for i in (1..80u64).step_by(2) {
    assert!(
      t.look_up(&format!("m{i:03}").into_bytes(), &mut out)
        .await
        .unwrap(),
      "m{i:03} 应存活"
    );
  }
  check_tree(&t);
}

#[compio::test]
async fn x_merge_cascades() {
  let mut t = mk();
  for i in 0..120u64 {
    t.insert(&format!("x{i:03}").into_bytes(), &[0; 4])
      .await
      .unwrap();
  }
  let before = t.count_pages().await.unwrap();
  let merged = t.x_merge(0).await.unwrap();
  let _ = merged;
  assert_eq!(t.count_entries().await.unwrap(), 120, "XMerge 不丢键");
  // 逐键复核（旧实现丢右节点记录，仅 count_entries 断言无法发现）
  let mut out = Vec::new();
  for i in 0..120u64 {
    assert!(
      t.look_up(&format!("x{i:03}").into_bytes(), &mut out)
        .await
        .unwrap(),
      "x{i:03} 应存活于 XMerge 后"
    );
  }
  assert!(t.count_pages().await.unwrap() <= before);
  check_tree(&t);
}

#[compio::test]
async fn contention_split_favored_position() {
  let mut t = mk();
  for i in 0..4u64 {
    t.insert(&format!("c{i:03}").into_bytes(), &[0; 8])
      .await
      .unwrap();
  }
  assert_eq!(t.count_pages().await.unwrap(), 1, "4 键单叶");
  // 第一层更新门：rand=128（`128 & 127 == 0` 但 `128 & 16383 != 0`）→ 只记录
  // last_modified_pos=1，不结算、不分裂。
  assert!(!t.contention_split(0, 1, true, 128).await.unwrap());
  // 结算门：rand=0 同时过两层；last=1 != slot=3 → favored = min(1,3) = 1。
  assert!(t.contention_split(0, 3, true, 0).await.unwrap());
  // favored_split_pos=1 → 叶左半保留 slots[0..=1] = 2 条；
  // 若忽略 favored（默认 append_bias 取 count-2=2）左半会是 3 条。
  let left_cnt = t.nodes.read().get(&0).unwrap().hdr.count;
  assert_eq!(
    left_cnt, 2,
    "favored_split_pos=1 必须被尊重（左半=2 而非 3）"
  );
  check_tree(&t);
}

#[compio::test]
async fn contention_split_cold_path_no_split() {
  let mut t = mk();
  for i in 0..8u64 {
    t.insert(&format!("d{i:03}").into_bytes(), &[0; 8])
      .await
      .unwrap();
  }
  let pages = t.count_pages().await.unwrap();
  // 未过第一层门（rand=1）：直接 None，不分裂
  assert!(!t.contention_split(0, 0, true, 1).await.unwrap());
  // 过了门但 faced_contention=false → normalized=0 < 1 → 不分裂
  assert!(!t.contention_split(0, 0, false, 0).await.unwrap());
  assert_eq!(t.count_pages().await.unwrap(), pages, "冷路径不得分裂");
}

#[compio::test]
async fn oversized_record_rejected_not_retried() {
  let mut t = mk();
  // key+payload > MAX_RECORD_SIZE → 立即 Err（旧实现无限重试/分裂 panic）
  let big = vec![0u8; MAX_RECORD_SIZE as usize + 1];
  let err = t.insert(&big, b"v").await.unwrap_err();
  assert!(
    matches!(err, Error::RecordTooLarge { .. }),
    "超限记录必须 Err(RecordTooLarge): {err:?}"
  );
  // 恰好等于上界（key+payload == MAX_RECORD_SIZE）必须接受（边界，非 >）
  let max = MAX_RECORD_SIZE as usize;
  let exact = vec![b'q'; max - 4];
  t.insert(&exact, &[0u8; 4]).await.unwrap();
  assert_eq!(t.count_entries().await.unwrap(), 1);
}

/// 长分隔键分裂：`find_separator.len > 512` 时旧 512 B 栈缓冲 panic。
#[compio::test]
async fn long_separator_split() {
  let mut t = mk();
  // 601 B 键（前 599 字节相同、尾 2 字节区分）→ 分隔键 ≈ 600 B > 512
  for i in 0..200u32 {
    let mut k = vec![b'a'; 599];
    k.push((i & 0xFF) as u8);
    k.push((i >> 8) as u8);
    t.insert(&k, &[0u8; 4]).await.unwrap();
  }
  assert!(t.count_pages().await.unwrap() > 1, "长键必须发生分裂");
  let mut out = Vec::new();
  for i in 0..200u32 {
    let mut k = vec![b'a'; 599];
    k.push((i & 0xFF) as u8);
    k.push((i >> 8) as u8);
    assert!(t.look_up(&k, &mut out).await.unwrap(), "i={i} 长键丢失");
  }
  check_tree(&t);
}

/// 账本公开 API 冒烟：`iterate_all_nodes` / `look_up_blob` / `alloc_pid`。
#[compio::test]
async fn ledger_api_smoke() {
  let mut t = mk();
  for i in 0..20u32 {
    t.insert(&format!("b{i:03}").into_bytes(), &[i as u8; 4])
      .await
      .unwrap();
  }
  // iterate_all_nodes：访问页数 == 页计数
  let mut visited = 0u64;
  let n = t
    .iterate_all_nodes(|_n| {
      visited += 1;
      true
    })
    .await
    .unwrap();
  assert_eq!(n as u64, t.count_pages().await.unwrap());
  assert_eq!(visited, n as u64);
  // look_up_blob：单叶根 prefix_len == 0，key_rest 传全键即命中
  let k = b"b005";
  let head: [u8; 4] = k[..4].try_into().unwrap();
  let got = t.look_up_blob(head, k, |a, b| a.cmp(b)).await.unwrap();
  assert_eq!(got, Some(vec![5u8; 4]));
  // alloc_pid：分配页号
  let p = t.alloc_pid().await;
  assert!(p >= 1);
}

#[test]
fn misc_config() {
  let mut t = mk();
  t.toggle_append_bias_mode();
  assert!(!t.append_bias);
  t.set_comparison_operator(ldb_node::Cmp {
    memcmp: false,
    f: |a: &[u8], b: &[u8]| a.cmp(b),
  });
  assert!(!t.cmp.memcmp);
  assert!(t.size_in_mb() > 0.0);
}
