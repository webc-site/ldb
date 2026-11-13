//! 第 4 轮**独立复核**：R3 引入的 `MAX_EXTENTS`（127）与可失败 `append` 的验证。
//!
//! - 重点 1：用数学式独立重写 `split_count`，与拆级贪心循环 / `append` 返回块数逐例对拍
//!   （`used = 0..4*full` 穷尽 + 大容量边界样本）。
//! - 重点 2：blob_mgr 层「先判定后改状态」端到端 —— 越上限时
//!   `length` / `resident.len()` / `number_of_extents()` / `total_size_extents()` / 尾缓冲
//!   全不变；fresh 路径无 id/去重索引/登记副作用。

use ldb_blob::{BlobManager, Error, ExtentList, MAX_EXTENTS, TailExtent, extent_size, tier_index};
use ldb_codec::PageCodec;
use ldb_type::PAGE_SIZE;

const PG: usize = PAGE_SIZE;

/// 独立 oracle：`while used > full { used -= full; n += 1 }` 的闭式解。
/// 此处按数学式独立重写，故意不引用实现里的 `split_count`。
fn oracle_split_count(used_pages: u64, full_pages: u64) -> u64 {
  if full_pages == 0 || used_pages <= full_pages {
    0
  } else {
    (used_pages - 1) / full_pages
  }
}

/// 直接对拍 `TailExtent::split_to_extents` 贪心循环：弹出块数 / 每块整级尺寸 / 剩余页数与字节。
fn check_tail_split(capacity: u64, used: u64) {
  let full = extent_size(tier_index(capacity.max(1)));
  assert_eq!(full, capacity, "本用例只取 2 的幂容量 ⇒ full == capacity");
  let mut t = TailExtent::new(0, capacity);
  t.data = vec![0u8; (used * PG as u64) as usize];
  t.used_pages = used;
  let blocks = t.split_to_extents();
  let expect = oracle_split_count(used, full);
  assert_eq!(blocks.len() as u64, expect, "cap={capacity} used={used}");
  assert_eq!(
    t.used_pages,
    used - expect * full,
    "剩余页数 cap={capacity} used={used}"
  );
  assert!(t.used_pages <= full, "剩余必须 <= full");
  assert_eq!(
    t.data.len() as u64,
    t.used_pages * PG as u64,
    "剩余字节 cap={capacity} used={used}"
  );
  for b in &blocks {
    assert_eq!(b.len() as u64, full * PG as u64, "每块必须整级尺寸");
  }
}

#[test]
fn split_count_oracle_vs_split_loop() {
  // 小容量穷尽 used = 0..=4*full（廉价：cap<=8 ⇒ 单例 ≤128 KiB）
  for capacity in [1u64, 2, 4, 8] {
    for used in 0..=(4 * capacity) {
      check_tail_split(capacity, used);
    }
  }
  // 中等容量（含 4*full 精确边界）取稀疏样本；≤ 1 MiB/例，守住 100ms 预算
  for capacity in [64u64] {
    let pts = [
      0,
      capacity - 1,
      capacity,
      capacity + 1,
      2 * capacity,
      2 * capacity + 1,
      4 * capacity - 1,
      4 * capacity,
    ];
    for used in pts {
      check_tail_split(capacity, used);
    }
  }
}

/// 大容量（1024/4096 页整级）样本：单例 16–64 MiB 分配，超 100ms ⇒ 按 E2 归入 `#[ignore]`，
/// 手动 `--run-ignored all` 运行（本轮已跑）。
#[test]
#[ignore = "大容量 split 对拍（16–64 MiB/例，手动 --ignored 运行）"]
fn split_count_oracle_vs_split_loop_large_caps() {
  for capacity in [1024u64, 4096] {
    let pts = [
      0,
      capacity - 1,
      capacity,
      capacity + 1,
      2 * capacity,
      2 * capacity + 1,
      4 * capacity - 1,
      4 * capacity,
    ];
    for used in pts {
      check_tail_split(capacity, used);
    }
  }
}

#[test]
fn split_count_oracle_vs_append_chunks() {
  // 单次 append：返回块数 == oracle（count 恒 0，远离上限）
  for capacity in [1u64, 2, 4, 8, 64] {
    let full = extent_size(tier_index(capacity));
    // cap=64 取稀疏 used 以守 100ms 预算；小 cap 穷尽
    let used_range: Vec<u64> = if capacity <= 8 {
      (0..=(4 * capacity)).collect()
    } else {
      vec![0, 63, 64, 65, 127, 128, 129, 256]
    };
    for used in used_range {
      let el = ExtentList::new(0, capacity);
      let data = vec![0u8; (used * PG as u64) as usize];
      let blocks = el.append(&data).expect("count=0，远未越上限");
      let expect = oracle_split_count(used, full);
      assert_eq!(blocks.len() as u64, expect, "cap={capacity} used={used}");
      assert_eq!(el.tail.read().used_pages, used - expect * full);
      for b in &blocks {
        assert_eq!(b.len() as u64, full * PG as u64);
      }
    }
  }
  // 逐页累加：k 页后累计弹出块数 == oracle(k, full)
  for capacity in [1u64, 2, 4, 8] {
    let full = extent_size(tier_index(capacity));
    let el = ExtentList::new(0, capacity);
    let mut pops = 0u64;
    for used in 1..=(4 * capacity) {
      let blocks = el.append(&vec![0u8; PG]).expect("ok");
      pops += blocks.len() as u64;
      assert_eq!(pops, oracle_split_count(used, full), "增量 used={used}");
    }
  }
}

/// blob 层不变量快照：`(length, resident.len(), #extent, total_size_extents, 尾缓冲字节)`。
fn blob_invariants(bm: &BlobManager, id: u64) -> (u64, usize, u64, u64, usize) {
  let blobs = bm.blobs.read();
  let st = blobs.get(&id).expect("blob 必须存在").read();
  (
    st.length,
    st.resident.len(),
    st.extents.number_of_extents(),
    st.extents.total_size_extents(),
    st.extents.tail.read().data.len(),
  )
}

/// 重点 2（end-to-end）：用尾容量 1 页把小 blob 撑到 127 extents，再 extend/write_new_data
/// 触发 `TooManyExtents`；断言该 blob 的全部可见状态与调用前**完全一致**，且数据仍可完整读回。
#[compio::test]
async fn blob_extend_over_cap_leaves_state_unchanged() {
  let bm = BlobManager::new(1); // tail_capacity == 1 页
  // 128 页 → 尾 1 页 + 拆出 127 块 ⇒ 恰好 127 个 extent（上限）
  let payload = vec![0xABu8; 128 * PG];
  let id = bm.allocate_blob(&payload).await.unwrap();
  let before = blob_invariants(&bm, id);
  assert_eq!(
    before.2, MAX_EXTENTS as u64,
    "128 页 / 尾容量 1 页 ⇒ 恰好 127 个 extent"
  );
  assert_eq!(before.1, payload.len());

  // extend：1 字节起即越上限（尾已满 1 页，再拆 1 块 ⇒ count 128）
  for more in [1usize, PG, 5 * PG] {
    let e = bm.extend_existing_blob(id, &vec![0xCDu8; more]).await;
    assert!(
      matches!(e, Err(Error::TooManyExtents)),
      "extend {more}B 必须 TooManyExtents"
    );
    assert_eq!(
      blob_invariants(&bm, id),
      before,
      "extend {more}B 拒绝后状态必须完全不变"
    );
  }

  // write_new_data_to_last_extent：同样必须先判定后改状态
  let e = bm.write_new_data_to_last_extent(id, &[0xEFu8; PG]).await;
  assert!(matches!(e, Err(Error::TooManyExtents)));
  assert_eq!(blob_invariants(&bm, id), before);

  // 拒绝后数据仍逐字节完整
  assert_eq!(bm.load_blob(id).await.unwrap(), payload);
}

/// 重点 2（fresh 路径）：超上限的新分配必须无任何副作用 —— 不 bump id、不登记、不污染去重索引。
#[compio::test]
async fn blob_fresh_over_cap_is_side_effect_free() {
  let bm = BlobManager::new(1);
  // 129 页 → 需 128 个 extent > 127
  let payload = vec![0u8; 129 * PG];
  let digest = PageCodec::sha256_digest(&payload);
  assert!(
    matches!(bm.allocate_blob(&payload).await, Err(Error::TooManyExtents)),
    "129 页 / 尾容量 1 页 ⇒ TooManyExtents"
  );
  assert_eq!(bm.blobs.read().len(), 0, "失败分配不得登记 blob");
  assert_eq!(
    bm.look_up_by_digest(&digest).await,
    None,
    "失败分配不得污染去重索引"
  );
  // next_id 未被消耗：首个成功分配拿到 id=1
  let id = bm.allocate_blob(b"hello").await.unwrap();
  assert_eq!(id, 1, "失败分配不得 bump next_id");

  // 边界：恰好 128 页必须成功（127 extents = 上限）
  let ok = vec![0u8; 128 * PG];
  let id2 = bm.allocate_blob(&ok).await.unwrap();
  assert_eq!(blob_invariants(&bm, id2).2, MAX_EXTENTS as u64);
}

/// R5 端到端抽测：越上限（127 extents）行为在**多个 tier/容量组合**下一致。
///
/// 尾容量分别取 1 / 4 / 8 页（tier 0 / 2 / 3），把 blob 撑到恰好 127 个整级
/// extent 后，`extend_existing_blob(+1 整级块)` 必返回 `TooManyExtents`，且
/// `(length, resident.len(), #extent, total_size_extents, 尾缓冲字节)` 逐字段不变、
/// 数据逐字节可读回。`BlobManager::new(cap)` 的 `cap` 会归一到所属 tier（此处皆 2 的幂）。
#[compio::test]
async fn blob_cap_holds_across_tail_capacities() {
  for cap in [1u64, 4, 8] {
    let bm = BlobManager::new(cap);
    // (MAX_EXTENTS + 1) 个整级块 ⇒ 拆出恰好 127 个 extent
    let pages = (MAX_EXTENTS + 1) * cap as usize;
    let payload = vec![0xA5u8; pages * PG];
    let id = bm.allocate_blob(&payload).await.unwrap();
    let before = blob_invariants(&bm, id);
    assert_eq!(
      before.2, 127u64,
      "cap={cap}: {pages} 页应恰好 127 个 extent（上游 EXTENT_CNT_MASK=0x7F）"
    );
    assert_eq!(before.1, payload.len(), "cap={cap}: 常驻副本完整");

    // 尾已占满 1 个整级块，再加 1 整级块 ⇒ 需拆 1 块、count 128 ⇒ 越上限
    let e = bm
      .extend_existing_blob(id, &vec![0x01u8; cap as usize * PG])
      .await;
    assert!(
      matches!(e, Err(Error::TooManyExtents)),
      "cap={cap}: 越上限必须 TooManyExtents"
    );
    assert_eq!(
      blob_invariants(&bm, id),
      before,
      "cap={cap}: 拒绝后状态不变"
    );
    assert_eq!(
      bm.load_blob(id).await.unwrap(),
      payload,
      "cap={cap}: 数据完整"
    );
  }
}
