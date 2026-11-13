//! MVCC 元组回归：Chained/Fat 语义 / 可见性 / 升降级 / GC / 墓地 / 序列化。
//!
//! 期望值一律写字面量（禁自我参照）。上游行号以 master
//! `leanstore/backend/leanstore/storage/btree/{BTreeVI.hpp,BTreeVI.cpp,FatTupleDifferentAttributes.cpp}` 为准。

use std::collections::HashMap;

use ldb_tuple::*;

/// 影子模型：键 → 版本列表（旧→新；`None` = 墓碑）。
type ShadowMap = HashMap<Vec<u8>, Vec<(u64, Option<Vec<u8>>)>>;

fn vi() -> BTreeVi {
  BTreeVi::create(64, 3, 4096)
}

#[compio::test]
async fn insert_lookup_visibility() {
  let mut t = vi();
  t.insert(b"a", b"v1", 10).await.unwrap();
  assert!(t.insert(b"a", b"x", 11).await.is_err(), "重复键报错");
  // 快照 5（创建前）不可见
  assert_eq!(t.lookup(b"a", 5).await.unwrap(), None);
  // 快照 10+ 可见
  assert_eq!(t.lookup(b"a", 10).await.unwrap().unwrap(), b"v1".to_vec());
  assert!(t.is_visible_for_me(10, 15));
  assert!(!t.is_visible_for_me(16, 15));
}

#[compio::test]
async fn update_chain_and_undo() {
  let mut t = vi();
  t.insert(b"k", b"v1", 10).await.unwrap();
  assert!(t.update_same_size_in_place(b"k", b"v2", 20).await.unwrap());
  assert_eq!(
    t.lookup(b"k", 15).await.unwrap().unwrap(),
    b"v1".to_vec(),
    "旧快照见旧版"
  );
  assert_eq!(t.lookup(b"k", 25).await.unwrap().unwrap(), b"v2".to_vec());
  // 撤销 tx20
  assert!(t.undo(b"k", 20).await);
  assert_eq!(
    t.lookup(b"k", 25).await.unwrap().unwrap(),
    b"v1".to_vec(),
    "撤销后回到 v1"
  );
}

/// 回滚到创建前必须把键摘掉（上游 undo 的 WALInsert 分支 `BTreeVI.cpp:549-564`）。
#[compio::test]
async fn undo_insert_rollback_removes_key() {
  let mut t = vi();
  t.insert(b"k", b"v1", 10).await.unwrap();
  assert!(t.undo(b"k", 10).await);
  assert!(
    t.lookup(b"k", 99).await.unwrap().is_none(),
    "回滚插入应摘键，head 不留存"
  );
  assert!(!t.chained.contains_key(b"k".as_slice()));
  // 摘键后可重新插入（无重复键）
  t.insert(b"k", b"v2", 11).await.unwrap();
  assert_eq!(t.lookup(b"k", 99).await.unwrap().unwrap(), b"v2".to_vec());
}

#[compio::test]
async fn fat_promotion_and_demotion() {
  let mut t = vi(); // threshold=3
  t.insert(b"hot", b"v1", 10).await.unwrap();
  t.update_same_size_in_place(b"hot", b"v2", 20)
    .await
    .unwrap();
  t.update_same_size_in_place(b"hot", b"v3", 30)
    .await
    .unwrap();
  // 2 次更新后 undo=2 → 含 head 共 3 版本 == 阈值，严格 `>` 不晋升
  assert!(t.fat_tuple(b"hot").is_none(), "3 版本 == 阈值不晋升");
  t.update_same_size_in_place(b"hot", b"v4", 40)
    .await
    .unwrap();
  // 第 3 次更新后 undo=3 → 含 head 共 4 版本 > 阈值，自动晋升
  // （上游 `updates_counter > convertToFatTupleThreshold()`，BTreeVI.cpp:308）
  assert!(t.fat_tuple(b"hot").is_some(), "热点键应转胖");
  assert_eq!(t.reconstruct_tuple(b"hot", 35).await, Some(b"v3".to_vec()));
  // 胖直接更新 + 撤销最近
  let ft = t.fat_tuple_mut(b"hot").unwrap();
  assert!(ft.update(50, b"v5", 4096));
  assert!(ft.undo_last_update());
  // 降级（head = 最新可见版本 v4；undo 链承载 v1/v2/v3）
  let ct = ft.convert_to_chained();
  assert_eq!(ct.head, b"v4".to_vec());
  assert_eq!(ct.head_tx, 40);
  assert_eq!(ct.value_at(15), Some(b"v1".to_vec().as_slice()));
}

/// 晋升阈值边界：严格 `>` 口径——版本数（含 head）**等于**阈值不晋升，
/// **大于**阈值晋升（对齐上游 `updates_counter > threshold`，`BTreeVI.cpp:308`）。
#[compio::test]
async fn promotion_threshold_boundary() {
  let mut t = BTreeVi::create(64, 3, 4096);
  t.insert(b"k", b"v0", 10).await.unwrap();
  // 2 次更新 → undo=2 → 含 head 共 3 版本 == 3：严格 `>` 不晋升
  t.update_same_size_in_place(b"k", b"v1", 20).await.unwrap();
  t.update_same_size_in_place(b"k", b"v2", 30).await.unwrap();
  assert!(t.fat_tuple(b"k").is_none(), "3 版本 == 阈值不晋升");
  // 3 次更新 → undo=3 → 含 head 共 4 版本 > 3：晋升
  t.update_same_size_in_place(b"k", b"v3", 40).await.unwrap();
  assert!(t.fat_tuple(b"k").is_some(), "4 版本 > 阈值晋升");
  // 降级后版本序旧→新
  let ft = t.fat_tuple(b"k").unwrap();
  let ct = ft.convert_to_chained();
  assert_eq!(ct.head, b"v3".to_vec());
  assert_eq!(ct.head_tx, 40);
  assert_eq!(ct.undo.len(), 3);
  assert_eq!(ct.undo[0].tx_id, 10);
  assert_eq!(ct.undo[1].tx_id, 20);
  assert_eq!(ct.undo[2].tx_id, 30);
}

/// 阈值口径差异的可观测对照：同一序列下，`>=`（旧）会在第 2 次更新就转胖，
/// `>`（上游/本仓现行）在第 3 次更新才转胖——差一次更新，经 `fat_tuple` 可观测。
#[compio::test]
async fn promotion_timing_aligned_with_upstream() {
  let mut t = BTreeVi::create(64, 3, 4096);
  t.insert(b"k", b"v0", 10).await.unwrap();
  t.update_same_size_in_place(b"k", b"v1", 20).await.unwrap(); // 2 版本
  t.update_same_size_in_place(b"k", b"v2", 30).await.unwrap(); // 3 版本
  assert!(
    t.fat_tuple(b"k").is_none(),
    "第 2 次更新后（3 版本）按上游严格 > 仍未晋升"
  );
  assert!(t.chained.contains_key(b"k".as_slice()), "仍在链式表");
  t.update_same_size_in_place(b"k", b"v3", 40).await.unwrap(); // 4 版本
  assert!(t.fat_tuple(b"k").is_some(), "第 3 次更新后晋升");
  assert!(!t.chained.contains_key(b"k".as_slice()), "已移出链式表");
}

/// 容量边界：恰好填满接受，超 1 字节拒绝（上游 `hasSpaceFor`，
/// `FatTupleDifferentAttributes.cpp:198-202`）。
#[test]
fn capacity_boundary_literal() {
  assert!(FatTuple::has_space_for(4, 4, 8));
  assert!(!FatTuple::has_space_for(4, 5, 8));
  assert!(FatTuple::has_space_for(0, 8, 8));

  let mut ft = FatTuple::default();
  assert!(ft.append(10, &[1, 2, 3, 4], 8));
  assert!(ft.append(20, &[5, 6, 7, 8], 8));
  assert!(!ft.append(30, &[9], 8), "超容量拒绝追加");
  assert_eq!(ft.versions.buf.len(), 8);
  assert_eq!(ft.versions.deltas.len(), 2);
  // resize 谓词：当前载荷是否落在新容量内
  assert!(ft.resize(8));
  assert!(!ft.resize(7));
}

/// allocate_delta 维护 offset 与 buf（上游 `allocateDelta`，
/// `FatTupleDifferentAttributes.cpp:204-212`）。
#[test]
fn allocate_delta_maintains_offset_and_buffer() {
  let mut ft = FatTuple::default();
  ft.allocate_delta(10, 3);
  ft.allocate_delta(20, 2);
  assert_eq!(ft.versions.buf.len(), 5);
  assert_eq!(ft.versions.get_delta_offsets(), vec![0, 3]);
  assert_eq!(ft.versions.deltas[0].offset, 0);
  assert_eq!(ft.versions.deltas[1].offset, 3);
  assert_eq!(ft.versions.deltas[1].length, 2);
  assert_eq!(ft.versions.deltas[1].tx, 20);
  assert_eq!(ft.versions.buf, vec![0, 0, 0, 0, 0]);
  assert_eq!(ft.versions.deltas[1].total_length(), 2);
}

/// Delta 数组线上布局：旧→新、整记录、offset 表累计（上游 `BTreeVI.hpp:141`
/// O2N 与 `:161` 偏移表）。
#[test]
fn delta_array_layout_matches_upstream() {
  let mut ft = FatTuple::default();
  assert!(ft.append(100, b"AAAA", 64)); // index0 最旧
  assert!(ft.append(200, b"BB", 64)); // index1
  assert!(ft.append(300, b"C", 64)); // index2 最新
  // buf 按旧→新拼接
  assert_eq!(ft.versions.buf, b"AAAABBC".to_vec());
  // offset 表累计且被维护
  assert_eq!(ft.versions.get_delta_offsets(), vec![0, 4, 6]);
  assert_eq!(ft.versions.get_delta(0), b"AAAA");
  assert_eq!(ft.versions.get_delta(1), b"BB");
  assert_eq!(ft.versions.get_delta(2), b"C");
  // 方向：index0 最旧
  assert_eq!(ft.versions.deltas[0].tx, 100);
  assert_eq!(ft.versions.deltas[2].tx, 300);
  // 编译期纯函数形态（按 length 累计）
  let deltas = [
    Delta::get_constant_descriptor(0, 4),
    Delta::get_constant_descriptor(0, 2),
    Delta::get_constant_descriptor(0, 1),
  ];
  assert_eq!(
    DeltaArray::get_delta_offsets_constant(&deltas),
    vec![0, 4, 6]
  );
  assert_eq!(
    DeltaArray::get_value_constant(ft.versions.buf.as_slice(), &deltas, 1),
    b"BB"
  );
}

/// 链式窗口语义：取 `tx <= snapshot` 的最新版本；该版本为墓碑 → None，不回退
/// （上游 `BTreeVI.cpp:875-944` `reconstructChainedTuple`）。
#[test]
fn chained_window_semantics_literal() {
  let mut ct = ChainedTuple {
    head: b"v2".to_vec(),
    head_tx: 20,
    ..ChainedTuple::default()
  };
  ct.undo.push(UndoVersion {
    tx_id: 10,
    payload: b"v1".to_vec(),
    removed: false,
  });
  assert_eq!(ct.value_at(5), None, "早于首个版本不可见");
  assert_eq!(ct.value_at(10), Some(b"v1".as_slice()));
  assert_eq!(ct.value_at(15), Some(b"v1".as_slice()));
  assert_eq!(ct.value_at(20), Some(b"v2".as_slice()));

  // 墓碑 head：可见即 None，且不回退到 undo 中的存活版本
  let mut ct = ChainedTuple {
    head_tx: 30,
    removed_head: true,
    ..ChainedTuple::default()
  };
  ct.undo.push(UndoVersion {
    tx_id: 20,
    payload: b"v2".to_vec(),
    removed: false,
  });
  assert_eq!(ct.value_at(25), Some(b"v2".as_slice()));
  assert_eq!(ct.value_at(30), None, "墓碑窗口不回退");
  assert_eq!(ct.value_at(99), None);
}

/// 胖窗口语义：删除窗口 → None，不回退（本模型对上游 FatTuple 的墓碑扩展）。
#[test]
fn fat_window_semantics_tombstone_literal() {
  let mut ft = FatTuple::default();
  assert!(ft.append(10, b"v1", 64));
  assert!(ft.append(20, b"v2", 64));
  assert!(ft.append(30, b"", 64)); // 墓碑
  *ft.removed.last_mut().unwrap() = true;
  assert!(ft.append(40, b"v4", 64));

  assert_eq!(ft.reconstruct_tuple(5), None);
  assert_eq!(ft.reconstruct_tuple(10), Some(b"v1".to_vec()));
  assert_eq!(ft.reconstruct_tuple(15), Some(b"v1".to_vec()));
  assert_eq!(ft.reconstruct_tuple(25), Some(b"v2".to_vec()));
  assert_eq!(ft.reconstruct_tuple(35), None, "墓碑窗口不回退 v2");
  assert_eq!(ft.reconstruct_tuple(45), Some(b"v4".to_vec()));
}

/// 窗口按存储位置（数组末位 = 最新）而非 max(tx) 判定；模型要求 append 序 = 提交序，
/// 乱序用例只用于钉死"按位置回放"这一事实（上游 `BTreeVI.cpp:315-325` 亦按位置回放）。
#[test]
fn window_follows_positional_order_not_tx_max() {
  let mut ft = FatTuple::default();
  assert!(ft.append(50, b"b50", 64)); // index0，位置较早
  assert!(ft.append(30, b"b30", 64)); // index1，位置最新（tx 反而更小）
  assert_eq!(
    ft.reconstruct_tuple(60),
    Some(b"b30".to_vec()),
    "取位置最新而非 max tx"
  );
  assert_eq!(ft.reconstruct_tuple(40), Some(b"b30".to_vec()));
  assert_eq!(ft.reconstruct_tuple(20), None);
}

/// 链式 GC 必须保留"水位内最新版本"锚点，墓碑锚点不得删除（否则快照区间回退错值）。
#[compio::test]
async fn chained_gc_keeps_tombstone_anchor() {
  let mut t = BTreeVi::create(64, 100, 4096); // 阈值高 → 不晋升，专测链式 GC
  t.insert(b"k", b"ab", 10).await.unwrap();
  t.remove(b"k", 20).await.unwrap(); // 删除 @20
  t.update_same_size_in_place(b"k", b"cd", 30).await.unwrap(); // 复活 @30
  assert_eq!(t.lookup(b"k", 26).await.unwrap(), None, "GC 前删除窗口");
  // 锚点取 watermark 内**最新**版本（index1），净掉更旧的 1 条（若取最旧则净 0 条）
  assert_eq!(
    t.trigger_page_wise_garbage_collection(25).await,
    1,
    "锚点=最新<=watermark，仅净掉更旧版本"
  );
  assert_eq!(
    t.lookup(b"k", 26).await.unwrap(),
    None,
    "GC 后墓碑锚点仍界定删除窗口"
  );
  assert_eq!(
    t.lookup(b"k", 35).await.unwrap().unwrap(),
    b"cd".to_vec(),
    "复活版本可见"
  );
  // 恢复新版本后删除应回退到 v1（纯更新链）
  let mut t = BTreeVi::create(64, 100, 4096);
  t.insert(b"x", b"v1", 10).await.unwrap();
  t.update_same_size_in_place(b"x", b"v2", 20).await.unwrap();
  t.update_same_size_in_place(b"x", b"v3", 30).await.unwrap();
  assert_eq!(t.trigger_page_wise_garbage_collection(25).await, 1);
  assert_eq!(
    t.lookup(b"x", 26).await.unwrap().unwrap(),
    b"v2".to_vec(),
    "锚点=tx20 保 v2"
  );
}

/// 台账点名函数的真实行为（get_meta / lookup_pessimistic / execute_deterministric_update /
/// convert_to_fat_tuple_threshold / max_fat_tuple_length / write_lock_tuple / check_space_utilization）。
#[compio::test]
async fn ledger_api_surface() {
  let mut t = BTreeVi::create(64, 3, 4096);
  let meta = t.get_meta();
  assert_eq!(meta.record_len, 64);
  assert_eq!(meta.fat_threshold, 3);
  assert_eq!(t.convert_to_fat_tuple_threshold().await, 3);
  assert_eq!(t.max_fat_tuple_length().await, 4096);

  t.insert(b"k", b"abcd", 10).await.unwrap();
  // 悲观与乐观同构
  assert_eq!(
    t.lookup_pessimistic(b"k", 15).await.unwrap().unwrap(),
    b"abcd".to_vec()
  );
  // 写锁：一次成功、二次失败、解锁返回持有者
  assert!(t.write_lock_tuple(b"k", 77).await);
  assert!(!t.write_lock_tuple(b"k", 88).await);
  assert_eq!(t.unlock(b"k").await, 77);

  // 预备 + 执行（链式路径）
  let d = t.prepare_deterministic_update(b"k", b"efgh", 20).unwrap();
  assert_eq!((d.offset, d.length, d.tx), (0, 4, 20));
  assert!(
    t.execute_deterministric_update(b"k", d, b"efgh")
      .await
      .unwrap()
  );
  assert_eq!(t.lookup(b"k", 25).await.unwrap().unwrap(), b"efgh".to_vec());
  assert_eq!(t.lookup(b"k", 15).await.unwrap().unwrap(), b"abcd".to_vec());

  // 空间利用率：胖元组载荷 >= capacity*9/10 判真（字段 pub，直接构造）
  let mut small = BTreeVi::create(64, 1, 100);
  let mut ft = FatTuple::default();
  assert!(ft.append(1, &[0u8; 95], 100)); // 95 >= 100*9/10
  small.fat.insert(b"z".to_vec(), ft);
  assert!(small.check_space_utilization(b"z").await);
  assert!(
    !small.check_space_utilization(b"none").await,
    "无胖元组判假"
  );
}

/// 可见性差分对拍：随机多版本链（含墓碑），链/胖两条路径对独立 oracle ≥1e4 例。
#[test]
fn visibility_differential_oracle() {
  let mut s = 0x1234_5678_9ABC_DEF0u64;
  let mut rng = move || {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
  };
  let mut mism = 0u32;
  let mut cases = 0u32;
  for _ in 0..10_000u32 {
    let n = 1 + (rng() % 6) as usize;
    let mut versions: Vec<(u64, u8, bool)> = Vec::with_capacity(n);
    let mut tx = 1 + rng() % 5;
    for _ in 0..n {
      let removed = rng() % 6 == 0;
      versions.push((tx, (rng() % 256) as u8, removed));
      tx += 1 + rng() % 8;
    }
    let last = *versions.last().unwrap();
    let mut ct = ChainedTuple {
      head: if last.2 { Vec::new() } else { vec![last.1] },
      head_tx: last.0,
      removed_head: last.2,
      ..ChainedTuple::default()
    };
    for v in &versions[..n - 1] {
      ct.undo.push(UndoVersion {
        tx_id: v.0,
        payload: vec![v.1],
        removed: v.2,
      });
    }
    let mut ft = FatTuple::default();
    for v in &versions {
      ft.append(v.0, &[v.1], 4096);
      if v.2 {
        *ft.removed.last_mut().unwrap() = true;
      }
    }
    let max_tx = last.0 + 5;
    let snaps = [0u64, max_tx, 1 + rng() % max_tx, 1 + rng() % max_tx];
    for snap in snaps {
      // oracle：位置序（新→旧）首个 tx<=snap；墓碑 → None
      let expect = versions
        .iter()
        .rev()
        .find(|v| v.0 <= snap)
        .and_then(|v| if v.2 { None } else { Some(vec![v.1]) });
      let got_ct = ct.value_at(snap).map(|v| v.to_vec());
      let got_ft = ft.reconstruct_tuple(snap);
      if got_ct != expect {
        mism += 1;
      }
      if got_ft != expect {
        mism += 1;
      }
      cases += 1;
    }
  }
  assert!(cases >= 10_000, "对拍例数 {cases}");
  assert_eq!(mism, 0, "可见性差分 MISMATCH");
}

/// BTreeVI 影子模型差分（返回 MISMATCH 计数）。键 → 版本列表（旧→新），与
/// `BTreeVi` 逐点对拍（插入/更新/删除/晋升/GC/查询）；查询快照 ∈ [watermark, clock]，
/// GC 不改变该区间结果。
async fn shadow_differential(ops: u64) -> u64 {
  let mut t = BTreeVi::create(64, 4, 4096);
  // shadow: key → Vec<(tx, Option<value>)>（旧→新；None=墓碑）
  let mut shadow: ShadowMap = ShadowMap::new();
  let mut keys: Vec<Vec<u8>> = Vec::new();
  let mut next_key = 0u64;
  let mut clock = 1000u64;
  let mut watermark = 0u64;
  let val: Vec<u8> = b"vvvv".to_vec(); // 定长 4 字节

  let mut s = 0xDEAD_BEEF_CAFE_1234u64;
  let mut rng = move || {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
  };

  let mut mism = 0u64;
  for _ in 0..ops {
    let op = rng() % 100;
    clock += 1 + rng() % 3;
    let tx = clock;
    if op < 24 {
      // 新键插入
      let k = format!("k{next_key}").into_bytes();
      next_key += 1;
      t.insert(&k, &val, tx).await.unwrap();
      shadow.insert(k.clone(), vec![(tx, Some(val.clone()))]);
      keys.push(k);
    } else if keys.is_empty() {
      continue;
    } else {
      let k = keys[(rng() as usize) % keys.len()].clone();
      if op < 52 {
        // 同尺寸更新
        if t.update_same_size_in_place(&k, &val, tx).await.unwrap() {
          shadow.get_mut(&k).unwrap().push((tx, Some(val.clone())));
        }
      } else if op < 66 {
        // 删除
        if t.remove(&k, tx).await.unwrap() {
          shadow.get_mut(&k).unwrap().push((tx, None));
        }
      } else if op < 78 {
        // 页级 GC（晋升已由 update_same_size_in_place 自动触发）
        let w = clock.saturating_sub(40);
        t.trigger_page_wise_garbage_collection(w).await;
        watermark = watermark.max(w);
      } else {
        // 查询（快照 ∈ [watermark, clock]）
        let span = clock - watermark;
        let snap = watermark + rng() % (span + 1);
        let expect = shadow.get(&k).and_then(|vs| {
          vs.iter()
            .rev()
            .find(|v| v.0 <= snap)
            .and_then(|v| v.1.clone())
        });
        let got = t.lookup(&k, snap).await.unwrap();
        if got != expect {
          mism += 1;
        }
        let got2 = t.reconstruct_tuple(&k, snap).await;
        if got2 != expect {
          mism += 1;
        }
      }
    }
  }
  mism
}

/// 影子模型差分（内联 5e3 操作，<100ms 符合 E2）。
#[compio::test]
async fn btree_vi_shadow_model_oracle() {
  assert_eq!(
    shadow_differential(5_000).await,
    0,
    "BTreeVI 影子模型 MISMATCH"
  );
}

/// 影子模型差分（1e5 操作，>100ms 故 `#[ignore]`；显式 `--run-ignored` 慢跑取证）。
#[ignore = "慢：1e5 操作差分，用 nextest --run-ignored 取证"]
#[compio::test]
async fn shadow_model_differential_100k() {
  assert_eq!(
    shadow_differential(100_000).await,
    0,
    "1e5 操作 BTreeVI 影子模型 MISMATCH"
  );
}

#[compio::test]
async fn remove_gc_and_graveyard() {
  let mut t = vi();
  t.insert(b"dead", b"payload", 10).await.unwrap();
  assert!(t.remove(b"dead", 20).await.unwrap());
  assert_eq!(t.lookup(b"dead", 25).await.unwrap(), None, "删除后不可见");
  // GC：watermark=30 → 删除版本净化 → 键迁墓
  let purged = t.trigger_page_wise_garbage_collection(30).await;
  assert!(purged >= 1);
  assert!(
    t.graveyard.contains(b"dead".to_vec().as_slice()),
    "删除键应迁墓"
  );
  // OLAP 扫描能看到墓键
  let mut seen = 0;
  t.scan_olap(b"d", b"e", 40, |_, _| {
    seen += 1;
    true
  })
  .await;
  assert_eq!(seen, 1, "墓键参与 OLAP 扫描");
}

#[compio::test]
async fn scan_ranges_order() {
  let mut t = BTreeVi::create(64, 100, 4096);
  for i in 0..10u8 {
    t.insert(&[i], &[i; 4], 10).await.unwrap();
  }
  let mut asc = Vec::new();
  let n = t
    .scan_asc(&[2], &[6], 99, |k, _| {
      asc.push(k[0]);
      true
    })
    .await;
  assert_eq!((n, asc), (4, vec![2, 3, 4, 5]));
  let mut desc = Vec::new();
  let nd = t
    .scan_desc(&[2], &[6], 99, |k, _| {
      desc.push(k[0]);
      true
    })
    .await;
  assert_eq!((nd, desc), (4, vec![5, 4, 3, 2]));
}

#[compio::test]
async fn serialize_roundtrip() {
  let mut t = vi();
  t.insert(b"key1", b"value1", 10).await.unwrap();
  t.insert(b"key2", b"value2", 10).await.unwrap();
  let mut buf = Vec::new();
  t.serialize(&mut buf).await;
  let mut restored = vi();
  let used = restored.deserialize(&buf).await.unwrap();
  assert_eq!(used, buf.len());
  assert_eq!(
    restored.lookup(b"key1", 99).await.unwrap().unwrap(),
    b"value1".to_vec()
  );
  // 截断报错
  assert!(restored.deserialize(&buf[..3]).await.is_err());
}

#[test]
fn delta_views_and_chained_predicates() {
  let mut ct = ChainedTuple {
    head: vec![1, 2],
    ..ChainedTuple::default()
  };
  assert!(ct.is_final());
  assert!(!ct.is_write_locked());
  assert_eq!(ct.write_lock(7), Some(7));
  assert!(ct.is_write_locked());
  assert_eq!(ct.write_lock(9), None);
  assert_eq!(ct.unlock(), 7);

  let deltas = vec![
    Delta::get_constant_descriptor(0, 2),
    Delta::get_constant_descriptor(4, 3),
  ];
  let buf = [9u8, 9, 8, 8, 8];
  assert_eq!(DeltaArray::get_delta_offsets_constant(&deltas), vec![0, 2]);
  assert_eq!(DeltaArray::get_value_constant(&buf, &deltas, 0), &[9, 9]);
  assert_eq!(DeltaArray::get_value_constant(&buf, &deltas, 1), &[8, 8, 8]);
  assert_eq!(deltas[1].total_length(), 3);
}

// ============================ 第 2 轮独立对抗性用例 ============================
// 以下用例为第 2 轮主审自写，**不引用**第 1 轮 harness，逐条独立复现 6 个 P1 修复
// 与 serialize 完整性，并给出独立的可见性 oracle。

/// 独立 xorshift64 RNG（用例内联，避免共享状态）。
fn rng64(seed: u64) -> impl FnMut() -> u64 {
  let mut s = seed | 1;
  move || {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
  }
}

/// P1-3 独立复核：`convert_to_chained` 的 `undo` 只含 head 之前的版本，head 不重复。
#[test]
fn convert_to_chained_undo_excludes_head() {
  let mut ft = FatTuple::default();
  assert!(ft.append(10, b"v1", 64));
  assert!(ft.append(20, b"v2", 64));
  assert!(ft.append(30, b"v3", 64));
  let ct = ft.convert_to_chained();
  assert_eq!(ct.head, b"v3".to_vec());
  assert_eq!(ct.head_tx, 30);
  assert_eq!(ct.undo.len(), 2, "undo 不含 head");
  assert_eq!(ct.undo[0].tx_id, 10);
  assert_eq!(ct.undo[0].payload, b"v1".to_vec());
  assert_eq!(ct.undo[1].tx_id, 20);
  assert_eq!(ct.undo[1].payload, b"v2".to_vec());
}

/// P1-3 等价性：胖→链降级后逐快照可见性与胖一致（随机多版本 + 墓碑）。
#[test]
fn convert_to_chained_matches_fat_visibility() {
  let mut rng = rng64(0x51ED_2701_ABCD_1234);
  let mut cases = 0u32;
  let mut mism = 0u32;
  for _ in 0..3_000u32 {
    let n = 1 + (rng() % 5) as usize;
    let mut ft = FatTuple::default();
    let mut tx = 1 + rng() % 8;
    for _ in 0..n {
      let v = (rng() % 256) as u8;
      ft.append(tx, &[v], 4096);
      if rng() % 5 == 0 {
        *ft.removed.last_mut().unwrap() = true;
      }
      tx += 1 + rng() % 6;
    }
    let ct = ft.convert_to_chained();
    for snap in 0..48u64 {
      let a = ft.reconstruct_tuple(snap);
      let b = ct.value_at(snap).map(|v| v.to_vec());
      if a != b {
        mism += 1;
      }
      cases += 1;
    }
  }
  assert!(cases >= 10_000);
  assert_eq!(mism, 0, "胖/链降级可见性不一致 MISMATCH={mism}");
}

/// P1-4 独立复核：链式 undo 连续回滚至创建前必须真摘键，且可重新插入。
#[compio::test]
async fn undo_chain_to_creation_removes_key() {
  let mut t = BTreeVi::create(64, 100, 4096); // 高阈值：保持链式
  t.insert(b"z", b"1", 10).await.unwrap();
  t.update_same_size_in_place(b"z", b"2", 20).await.unwrap();
  t.update_same_size_in_place(b"z", b"3", 30).await.unwrap();
  assert!(t.undo(b"z", 30).await, "回滚 tx30");
  assert_eq!(t.lookup(b"z", 99).await.unwrap().unwrap(), b"2".to_vec());
  assert!(t.undo(b"z", 20).await, "回滚 tx20");
  assert_eq!(t.lookup(b"z", 99).await.unwrap().unwrap(), b"1".to_vec());
  assert!(t.undo(b"z", 10).await, "回滚创建 tx10");
  assert!(
    !t.chained.contains_key(b"z".as_slice()),
    "回滚到创建前必须摘键（BTreeVI.cpp:549-564 removeCurrent）"
  );
  assert_eq!(t.lookup(b"z", 99).await.unwrap(), None);
  // 摘键后可重新插入
  t.insert(b"z", b"9", 40).await.unwrap();
  assert_eq!(t.lookup(b"z", 99).await.unwrap().unwrap(), b"9".to_vec());
}

/// P1-5 独立复核：胖式 undo 后 `buf`/`removed`/`offset` 三者一致（逐字节重建验证）。
#[compio::test]
async fn btree_undo_fat_rebuilds_offsets() {
  let mut t = BTreeVi::create(64, 1, 4096);
  let mut ft = FatTuple::default();
  assert!(ft.append(10, b"aa", 4096));
  assert!(ft.append(20, b"bbb", 4096));
  assert!(ft.append(30, b"cccc", 4096));
  t.fat.insert(b"k".to_vec(), ft);
  assert!(t.undo(b"k", 20).await, "摘除中间版本");
  let ft = t.fat_tuple(b"k").unwrap();
  assert_eq!(ft.versions.deltas.len(), 2);
  assert_eq!(ft.removed.len(), 2, "removed 与 deltas 同长");
  assert_eq!(ft.versions.deltas[0].tx, 10);
  assert_eq!(ft.versions.deltas[1].tx, 30);
  assert_eq!(ft.versions.deltas[0].offset, 0);
  assert_eq!(
    ft.versions.deltas[1].offset, 2,
    "offset 重算：摘除点之后前移"
  );
  assert_eq!(ft.versions.get_delta(0), b"aa");
  assert_eq!(ft.versions.get_delta(1), b"cccc");
  assert_eq!(ft.versions.buf, b"aacccc".to_vec());
}

/// P1-5 姊妹路径：`FatTuple::undo_last_update` 后逐字节一致性（随机）。
#[test]
fn fat_undo_last_update_bytewise_consistent() {
  let mut rng = rng64(0x0BAD_F00D_1234_5678);
  for _ in 0..2_000u32 {
    let n = 2 + (rng() % 6) as usize;
    let mut ft = FatTuple::default();
    let mut tx = 10u64;
    for _ in 0..n {
      let len = 1 + (rng() % 6) as usize;
      let bytes: Vec<u8> = (0..len).map(|_| (rng() % 256) as u8).collect();
      assert!(ft.append(tx, &bytes, 4096));
      if rng() % 4 == 0 {
        *ft.removed.last_mut().unwrap() = true;
      }
      tx += 1 + rng() % 5;
    }
    if ft.undo_last_update() {
      let mut cat: Vec<u8> = Vec::new();
      let mut off = 0u16;
      for i in 0..ft.versions.deltas.len() {
        assert_eq!(ft.versions.deltas[i].offset, off, "offset 累计");
        cat.extend_from_slice(ft.versions.get_delta(i));
        off += ft.versions.deltas[i].length;
      }
      assert_eq!(ft.versions.buf, cat, "buf 与 offset 视图一致");
      assert_eq!(ft.removed.len(), ft.versions.deltas.len());
    }
  }
}

/// P1-6 独立复核：`maybe_promote` 容量不足时必须保持链式且**不丢版本**。
#[compio::test]
async fn promotion_capacity_fallback_keeps_versions() {
  // threshold=2 触发晋升，但容量仅 6 字节；2×4 字节放不下 → 留链式
  let mut t = BTreeVi::create(64, 2, 6);
  t.insert(b"k", b"aaaa", 10).await.unwrap();
  assert!(
    t.update_same_size_in_place(b"k", b"bbbb", 20)
      .await
      .unwrap()
  );
  assert!(t.fat_tuple(b"k").is_none(), "容量不足应保持链式");
  assert_eq!(
    t.lookup(b"k", 15).await.unwrap().unwrap(),
    b"aaaa".to_vec(),
    "旧版本不得丢失"
  );
  assert_eq!(t.lookup(b"k", 25).await.unwrap().unwrap(), b"bbbb".to_vec());
  // 显式转换在容量不足时返回 None（不产生截断的胖元组）
  let mut ct = ChainedTuple {
    head: b"bbbb".to_vec(),
    head_tx: 20,
    ..ChainedTuple::default()
  };
  ct.undo.push(UndoVersion {
    tx_id: 10,
    payload: b"aaaa".to_vec(),
    removed: false,
  });
  assert!(
    convert_chained_to_fat_tuple_different_attributes(&ct, 6)
      .await
      .is_none(),
    "容量不足返回 None"
  );
}

/// 独立复核（清单外新发现）：`convert_chained_to_fat_tuple_different_attributes`
/// 必须保留 undo 中的墓碑标记，否则删除窗口被错误复活。
#[compio::test]
async fn convert_chained_to_fat_preserves_tombstone() {
  let mut ct = ChainedTuple {
    creator: 10,
    head: b"v3".to_vec(),
    head_tx: 30,
    removed_head: false,
    ..ChainedTuple::default()
  };
  ct.undo.push(UndoVersion {
    tx_id: 10,
    payload: b"v1".to_vec(),
    removed: false,
  });
  ct.undo.push(UndoVersion {
    tx_id: 20,
    payload: b"v2".to_vec(),
    removed: true,
  });
  let made = convert_chained_to_fat_tuple_different_attributes(&ct, 4096).await;
  assert!(made.is_some(), "容量充足应成功");
  if let Some(ft) = made {
    assert_eq!(ft.reconstruct_tuple(15), Some(b"v1".to_vec()));
    assert_eq!(ft.reconstruct_tuple(25), None, "墓碑窗口必须保持删除");
    assert_eq!(ft.reconstruct_tuple(35), Some(b"v3".to_vec()));
  }
}

/// P1-2 独立复核：GC 前后所有 snapshot >= watermark 的可见性完全不变（链 + 胖，随机）。
#[test]
fn gc_preserves_visibility_above_watermark() {
  let mut rng = rng64(0x1CEB_00DA_1234_9876);
  let mut cases = 0u32;
  for _ in 0..3_000u32 {
    let n = 1 + (rng() % 5) as usize;
    let mut versions: Vec<(u64, u8, bool)> = Vec::with_capacity(n);
    let mut tx = 1 + rng() % 4;
    for _ in 0..n {
      let removed = rng() % 5 == 0;
      versions.push((tx, (rng() % 251) as u8 + 1, removed));
      tx += 1 + rng() % 6;
    }
    let last = versions[n - 1];
    let w = 1 + rng() % (last.0 + 2);
    // 胖
    let mut ft = FatTuple::default();
    for (t, v, r) in &versions {
      ft.append(*t, &[*v], 4096);
      if *r {
        *ft.removed.last_mut().unwrap() = true;
      }
    }
    // 链
    let mut ct = ChainedTuple {
      head: if last.2 { Vec::new() } else { vec![last.1] },
      head_tx: last.0,
      removed_head: last.2,
      ..ChainedTuple::default()
    };
    for v in &versions[..n - 1] {
      ct.undo.push(UndoVersion {
        tx_id: v.0,
        payload: vec![v.1],
        removed: v.2,
      });
    }
    // 采样 snapshots >= w
    let hi = last.0 + 3;
    let mut snaps: Vec<u64> = vec![w, w.saturating_add(1), last.0, hi];
    if w + 2 <= hi {
      snaps.push(w + 2);
    }
    let pre_f: Vec<Option<Vec<u8>>> = snaps.iter().map(|s| ft.reconstruct_tuple(*s)).collect();
    let pre_c: Vec<Option<Vec<u8>>> = snaps
      .iter()
      .map(|s| ct.value_at(*s).map(|v| v.to_vec()))
      .collect();
    ft.garbage_collection(w);
    let mut ct2 = ct.clone();
    let keep_from = if ct2.head_tx <= w {
      ct2.undo.len()
    } else {
      ct2.undo.iter().rposition(|v| v.tx_id <= w).unwrap_or(0)
    };
    ct2.undo.drain(..keep_from);
    let post_f: Vec<Option<Vec<u8>>> = snaps.iter().map(|s| ft.reconstruct_tuple(*s)).collect();
    let post_c: Vec<Option<Vec<u8>>> = snaps
      .iter()
      .map(|s| ct2.value_at(*s).map(|v| v.to_vec()))
      .collect();
    assert_eq!(pre_f, post_f, "胖 GC 改变可见性");
    assert_eq!(pre_c, post_c, "链 GC 改变可见性");
    cases += snaps.len() as u32;
  }
  assert!(cases >= 10_000);
}

/// 独立可见性 oracle ≥1e4：多版本 + 墓碑 + **乱序 tx**，快照落在墓碑窗口边界。
#[test]
fn visibility_oracle_tombstone_windows() {
  let mut rng = rng64(0xDEAD_BEEF_1234_5678);
  let mut cases = 0u64;
  let mut mism = 0u64;
  for _ in 0..4_000u32 {
    let n = 1 + (rng() % 5) as usize;
    let mut vs: Vec<(u64, u8, bool)> = Vec::with_capacity(n);
    let mut tx = 1 + rng() % 10;
    for _ in 0..n {
      let removed = rng() % 4 == 0;
      vs.push((tx, (rng() % 256) as u8, removed));
      tx = 1 + rng() % 40; // 乱序：不保证递增
    }
    let last = vs[n - 1];
    let mut ct = ChainedTuple {
      head: if last.2 { Vec::new() } else { vec![last.1] },
      head_tx: last.0,
      removed_head: last.2,
      ..ChainedTuple::default()
    };
    for v in &vs[..n - 1] {
      ct.undo.push(UndoVersion {
        tx_id: v.0,
        payload: vec![v.1],
        removed: v.2,
      });
    }
    let mut ft = FatTuple::default();
    for v in &vs {
      ft.append(v.0, &[v.1], 4096);
      if v.2 {
        *ft.removed.last_mut().unwrap() = true;
      }
    }
    let mut snaps: Vec<u64> = vec![0, 200];
    for v in &vs {
      if v.0 > 0 {
        snaps.push(v.0 - 1);
      }
      snaps.push(v.0);
      snaps.push(v.0 + 1);
    }
    for snap in snaps {
      // oracle：位置序（新→旧）首个 tx<=snap；墓碑 → None（上游 BTreeVI.cpp:875-944）
      let expect = vs
        .iter()
        .rev()
        .find(|v| v.0 <= snap)
        .and_then(|v| if v.2 { None } else { Some(vec![v.1]) });
      let got_c = ct.value_at(snap).map(|v| v.to_vec());
      let got_f = ft.reconstruct_tuple(snap);
      if got_c != expect {
        mism += 1;
      }
      if got_f != expect {
        mism += 1;
      }
      cases += 1;
    }
  }
  assert!(cases >= 10_000, "cases={cases}");
  assert_eq!(mism, 0, "可见性 oracle MISMATCH={mism}");
}

/// 独立复核（清单外新发现）：serialize/deserialize 检查点必须无损——
/// 版本链、墓碑标记、head_tx、胖元组与 graveyard 全部往返一致。
#[compio::test]
async fn serialize_preserves_versions_and_tombstones() {
  let mut t = BTreeVi::create(64, 3, 4096);
  t.insert(b"a", b"v1", 10).await.unwrap();
  t.update_same_size_in_place(b"a", b"v2", 20).await.unwrap();
  t.update_same_size_in_place(b"a", b"v3", 30).await.unwrap();
  t.update_same_size_in_place(b"a", b"v4", 40).await.unwrap(); // 4 版本 > 3 → 晋升为胖
  t.insert(b"b", b"bv", 11).await.unwrap();
  t.remove(b"b", 60).await.unwrap(); // 水位(25)之上的链式墓碑 → 不迁墓
  t.insert(b"c", b"cv", 12).await.unwrap();
  t.remove(b"c", 22).await.unwrap(); // 水位内删除 → 迁墓
  t.insert(b"d", b"dv", 13).await.unwrap();
  t.remove(b"d", 45).await.unwrap(); // 水位(25)之上的链式墓碑 → 不迁墓
  t.trigger_page_wise_garbage_collection(25).await; // c（水位内删除）迁墓；a 留下 3 版本
  assert!(t.fat_tuple(b"a").is_some(), "a 应已晋升");
  assert!(!t.chained.contains_key(b"c".as_slice()), "c 应迁墓");
  assert!(t.chained.contains_key(b"b".as_slice()), "b 链式墓碑保留");
  assert!(t.chained.contains_key(b"d".as_slice()), "d 链式墓碑保留");

  let mut buf = Vec::new();
  t.serialize(&mut buf).await;
  let mut r = BTreeVi::create(64, 3, 4096);
  assert_eq!(r.deserialize(&buf).await.unwrap(), buf.len());

  for s in [0u64, 9, 10, 15, 20, 21, 25, 30, 35, 44, 45, 59, 60, 61, 99] {
    assert_eq!(
      r.lookup(b"a", s).await.unwrap(),
      t.lookup(b"a", s).await.unwrap(),
      "a@{s}"
    );
    assert_eq!(
      r.lookup(b"b", s).await.unwrap(),
      t.lookup(b"b", s).await.unwrap(),
      "b@{s}"
    );
    assert_eq!(
      r.lookup(b"c", s).await.unwrap(),
      t.lookup(b"c", s).await.unwrap(),
      "c@{s}"
    );
    assert_eq!(
      r.lookup(b"d", s).await.unwrap(),
      t.lookup(b"d", s).await.unwrap(),
      "d@{s}"
    );
  }
  assert_eq!(
    r.fat_tuple(b"a").map(|f| f.versions.deltas.len()),
    t.fat_tuple(b"a").map(|f| f.versions.deltas.len()),
    "胖元组版本数往返一致"
  );
  assert_eq!(
    r.fat_tuple(b"a").map(|f| f.versions.deltas.len()),
    Some(3),
    "a 经 GC 后应留 3 版本（v2/v3/v4）"
  );
  assert_eq!(r.graveyard.len(), t.graveyard.len(), "graveyard 往返一致");
  assert!(r.graveyard.contains(b"c".to_vec().as_slice()));
}

/// 独立复核 P1-1 姊妹路径：反序列化对截断/长度篡改/越界/随机垃圾必须 Err 不 panic。
#[compio::test]
async fn deserialize_tamper_and_truncate_no_panic() {
  let mut t = vi();
  t.insert(b"abc", b"hello", 10).await.unwrap();
  t.insert(b"x", b"y", 11).await.unwrap();
  let mut buf = Vec::new();
  t.serialize(&mut buf).await;
  // 合法全量
  let mut r = vi();
  assert_eq!(r.deserialize(&buf).await.unwrap(), buf.len());
  // 逐字节截断：一律 Err，绝不 panic
  for cut in 0..buf.len() {
    let mut r2 = vi();
    assert!(r2.deserialize(&buf[..cut]).await.is_err(), "cut={cut}");
  }
  // 篡改首个键长字段（u16 @ offset 4）为 0xFFFF → 越界
  let mut bad = buf.clone();
  bad[4] = 0xFF;
  bad[5] = 0xFF;
  let mut r3 = vi();
  assert!(r3.deserialize(&bad).await.is_err());
  // 逻辑损坏：条目计数 = u32::MAX
  let mut bad2 = buf.clone();
  bad2[..4].copy_from_slice(&u32::MAX.to_le_bytes());
  let mut r4 = vi();
  assert!(r4.deserialize(&bad2).await.is_err());
  // 随机垃圾：绝不 panic（结果任意）
  let mut rng = rng64(0xC0FF_EE12_3456_7890);
  for _ in 0..256 {
    let len = (rng() % 96) as usize;
    let g: Vec<u8> = (0..len).map(|_| (rng() % 256) as u8).collect();
    let mut rg = vi();
    let _ = rg.deserialize(&g).await;
  }
}

// ============================ 第 3 轮独立对抗性用例 ============================
// 本轮聚焦：无损 serialize 全量/空/逐字节确定、u16 offset 上限、P1-14 墓碑传播、
// 阈值口径可观测性、scan_olap 空上界。均不引用上轮 harness。

/// 无损检查点全量面：链式多版本（含乱序 tx 与 undo 墓碑）、removed_head 墓碑、
/// 胖多版本（含 delta 墓碑）、graveyard；同实例两次序列化**逐字节一致**；空
/// `BTreeVi` 往返。载荷用 0xAA/0x55 以暴露字节序/字段错位。
#[compio::test]
async fn serialize_lossless_full_surface() {
  let mut t = BTreeVi::create(64, 1, 4096);
  // 链式：多版本 + 乱序 tx + 一条 undo 墓碑；head_tx=50
  let ct = ChainedTuple {
    creator: 0xAA,
    head: vec![0xAA; 3],
    head_tx: 50,
    removed_head: false,
    undo: vec![
      UndoVersion {
        tx_id: 30,
        payload: vec![0x55; 3],
        removed: false,
      },
      UndoVersion {
        tx_id: 10,
        payload: vec![0xAA; 3],
        removed: true,
      },
      UndoVersion {
        tx_id: 20,
        payload: vec![0x55; 3],
        removed: false,
      },
    ],
    write_lock_holder: 0,
  };
  t.chained.insert(b"ch".to_vec(), ct);
  // 链式 head 为墓碑（removed_head）；瞬态闩 holder=5 不应入盘
  let ct2 = ChainedTuple {
    creator: 0x55,
    head: b"gone".to_vec(),
    head_tx: 70,
    removed_head: true,
    undo: vec![UndoVersion {
      tx_id: 60,
      payload: b"live".to_vec(),
      removed: false,
    }],
    write_lock_holder: 5,
  };
  t.chained.insert(b"rh".to_vec(), ct2);
  // 胖：3 版本，中间为 delta 墓碑
  let mut ft = FatTuple {
    creator: 0xAA,
    ..Default::default()
  };
  assert!(ft.append(10, &[0xAA; 4], 4096));
  assert!(ft.append(20, &[], 4096));
  *ft.removed.last_mut().unwrap() = true;
  assert!(ft.append(30, &[0x55; 4], 4096));
  t.fat.insert(b"ft".to_vec(), ft);
  t.graveyard.insert(b"g1".to_vec());
  t.graveyard.insert(b"g2".to_vec());

  let mut b1 = Vec::new();
  t.serialize(&mut b1).await;
  let mut b2 = Vec::new();
  t.serialize(&mut b2).await;
  assert!(!b1.is_empty());
  assert_eq!(b1, b2, "同实例两次序列化必须逐字节一致");

  let mut r = BTreeVi::create(64, 1, 4096);
  assert_eq!(r.deserialize(&b1).await.unwrap(), b1.len());
  for s in [0u64, 5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 80] {
    assert_eq!(
      r.lookup(b"ch", s).await.unwrap(),
      t.lookup(b"ch", s).await.unwrap(),
      "ch@{s}"
    );
    assert_eq!(
      r.lookup(b"rh", s).await.unwrap(),
      t.lookup(b"rh", s).await.unwrap(),
      "rh@{s}"
    );
    assert_eq!(
      r.lookup(b"ft", s).await.unwrap(),
      t.lookup(b"ft", s).await.unwrap(),
      "ft@{s}"
    );
  }
  // 字段级往返
  let rc = r.chained.get(b"ch".as_slice()).unwrap();
  assert_eq!((rc.creator, rc.head_tx, rc.undo.len()), (0xAA, 50, 3));
  assert_eq!(rc.undo[1].tx_id, 10);
  assert!(rc.undo[1].removed, "undo 墓碑往返一致");
  assert_eq!(rc.write_lock_holder, 0, "瞬态闩状态不入盘");
  let rrh = r.chained.get(b"rh".as_slice()).unwrap();
  assert!(rrh.removed_head, "removed_head 往返一致");
  assert_eq!((rrh.head.clone(), rrh.head_tx), (b"gone".to_vec(), 70));
  assert_eq!(rrh.undo[0].payload, b"live".to_vec());
  let rft = r.fat.get(b"ft".as_slice()).unwrap();
  assert_eq!(rft.creator, 0xAA);
  assert_eq!(rft.versions.deltas.len(), 3);
  assert_eq!(
    rft.removed,
    vec![false, true, false],
    "delta 墓碑位往返一致"
  );
  assert_eq!(rft.versions.get_delta(0), &[0xAA; 4]);
  assert_eq!(rft.versions.get_delta(2), &[0x55; 4]);
  assert!(r.graveyard.contains(b"g1".to_vec().as_slice()));
  assert!(r.graveyard.contains(b"g2".to_vec().as_slice()));

  // 空 BTreeVi 往返（0 链 / 0 胖 / 0 墓）
  let e = BTreeVi::create(64, 3, 4096);
  let mut eb = Vec::new();
  e.serialize(&mut eb).await;
  let mut er = BTreeVi::create(64, 3, 4096);
  assert_eq!(er.deserialize(&eb).await.unwrap(), eb.len());
  assert!(er.chained.is_empty() && er.fat.is_empty() && er.graveyard.is_empty());
}

/// P1-14 强化复核：head 为墓碑（`removed_head`）时，链→胖转换必须传播该墓碑，
/// 且 undo 墓碑一并传播——删除窗口不得复活。
#[compio::test]
async fn convert_chained_to_fat_preserves_removed_head() {
  let ct = ChainedTuple {
    creator: 11,
    head: b"gone".to_vec(),
    head_tx: 30,
    removed_head: true,
    undo: vec![UndoVersion {
      tx_id: 20,
      payload: b"live".to_vec(),
      removed: false,
    }],
    write_lock_holder: 0,
  };
  let made = convert_chained_to_fat_tuple_different_attributes(&ct, 4096).await;
  assert!(made.is_some(), "容量充足应成功");
  if let Some(ft) = made {
    assert_eq!(ft.removed, vec![false, true], "head 墓碑位必须传播");
    assert_eq!(ft.reconstruct_tuple(25), Some(b"live".to_vec()));
    assert_eq!(ft.reconstruct_tuple(30), None, "removed_head 窗口不复活");
    assert_eq!(ft.reconstruct_tuple(99), None);
    // 与源链式逐快照一致
    for s in [0u64, 19, 20, 25, 29, 30, 40] {
      assert_eq!(
        ft.reconstruct_tuple(s),
        ct.value_at(s).map(|v| v.to_vec()),
        "转换后可见性 @{s}"
      );
    }
  }
}

/// u16 offset 上限：构造接近 64KiB（65535 字节）的胖元组，offset 表严格累计且
/// 不越界；再追加 1 字节必须被**拒绝**（返回 false），容量再大也受
/// `DELTA_ARRAY_MAX_SIZE` 约束（与上游"页放不下即拒绝/回退"一致）。
#[test]
fn delta_array_u16_offset_boundary() {
  let mut ft = FatTuple::default();
  let block = vec![0xAAu8; 4096];
  for i in 0..15u64 {
    assert!(ft.append(i, &block, DELTA_ARRAY_MAX_SIZE), "第 {i} 块");
  }
  let tail = vec![0x55u8; 4095];
  assert!(ft.append(99, &tail, DELTA_ARRAY_MAX_SIZE));
  assert_eq!(ft.versions.buf.len(), DELTA_ARRAY_MAX_SIZE); // 65535
  for i in 0..15usize {
    assert_eq!(ft.versions.deltas[i].offset, (i * 4096) as u16);
    assert_eq!(ft.versions.get_delta(i).len(), 4096);
  }
  assert_eq!(ft.versions.deltas[15].offset, 61440);
  assert_eq!(ft.versions.get_delta(15), &tail[..]);
  // 超限：拒绝，且缓冲不变
  assert!(
    !ft.append(100, &[1u8], DELTA_ARRAY_MAX_SIZE),
    "超 u16 上限拒绝"
  );
  assert!(!ft.append(100, &[1u8], usize::MAX), "容量再大也受上限约束");
  assert_eq!(
    ft.versions.buf.len(),
    DELTA_ARRAY_MAX_SIZE,
    "拒绝后缓冲不变"
  );

  // 1 字节 ×65535 全铺满：offset ∈ [0,65534]，全部为合法 u16
  let mut one = FatTuple::default();
  for i in 0..DELTA_ARRAY_MAX_SIZE {
    assert!(one.append(i as u64, &[7u8], usize::MAX), "i={i}");
  }
  assert_eq!(one.versions.deltas.len(), 65535);
  assert_eq!(one.versions.deltas[65534].offset, 65534);
  assert!(!one.append(0, &[7u8], usize::MAX), "满后再加拒绝");
}

/// 反序列化对"累计载荷超 u16 上限"的胖元组必须拒绝（不可信输入），
/// 而恰好在 65535 边界内的流必须成功且 offset 正确。
#[compio::test]
async fn deserialize_rejects_oversize_fat() {
  // 0 链 + 1 胖 + 2×40000 = 80000 > u16::MAX → Err
  let mut bad = Vec::new();
  bad.extend_from_slice(&0u32.to_le_bytes());
  bad.extend_from_slice(&1u32.to_le_bytes());
  bad.extend_from_slice(&1u16.to_le_bytes());
  bad.push(b'k');
  bad.extend_from_slice(&1u64.to_le_bytes()); // creator
  bad.extend_from_slice(&2u32.to_le_bytes()); // 2 deltas
  for _ in 0..2 {
    bad.extend_from_slice(&5u64.to_le_bytes());
    bad.push(0);
    bad.extend_from_slice(&40000u32.to_le_bytes());
    bad.extend_from_slice(&vec![0xAAu8; 40000]);
  }
  bad.extend_from_slice(&0u32.to_le_bytes()); // graveyard
  let mut r = BTreeVi::create(64, 3, 4096);
  assert!(r.deserialize(&bad).await.is_err(), "累计 > 64KiB 应拒绝");

  // 恰好 40000 + 25535 = 65535 → Ok，offset 正确
  let mut ok = Vec::new();
  ok.extend_from_slice(&0u32.to_le_bytes());
  ok.extend_from_slice(&1u32.to_le_bytes());
  ok.extend_from_slice(&1u16.to_le_bytes());
  ok.push(b'k');
  ok.extend_from_slice(&1u64.to_le_bytes());
  ok.extend_from_slice(&2u32.to_le_bytes());
  ok.extend_from_slice(&5u64.to_le_bytes());
  ok.push(0);
  ok.extend_from_slice(&40000u32.to_le_bytes());
  ok.extend_from_slice(&vec![0xAAu8; 40000]);
  ok.extend_from_slice(&6u64.to_le_bytes());
  ok.push(0);
  ok.extend_from_slice(&25535u32.to_le_bytes());
  ok.extend_from_slice(&vec![0x55u8; 25535]);
  ok.extend_from_slice(&0u32.to_le_bytes());
  let mut r2 = BTreeVi::create(64, 3, 4096);
  assert_eq!(r2.deserialize(&ok).await.unwrap(), ok.len());
  let ft = r2.fat.get(b"k".as_slice()).unwrap();
  assert_eq!(ft.versions.deltas.len(), 2);
  assert_eq!(ft.versions.deltas[1].offset, 40000);
  assert_eq!(ft.versions.get_delta(1).len(), 25535);
}

/// `scan_olap` 空上界语义：`hi` 为空表示无上界，墓键必须参与（不得退化为空区间）。
#[compio::test]
async fn scan_olap_empty_hi_includes_graveyard() {
  let mut t = vi();
  t.insert(b"dead", b"payload", 10).await.unwrap();
  t.remove(b"dead", 20).await.unwrap();
  t.trigger_page_wise_garbage_collection(30).await;
  assert!(t.graveyard.contains(b"dead".to_vec().as_slice()));

  let mut seen = Vec::new();
  let n = t
    .scan_olap(b"d", &[], 40, |k, _| {
      seen.push(k.to_vec());
      true
    })
    .await;
  assert_eq!((n, seen), (1, vec![b"dead".to_vec()]), "空上界须含墓键");
  // 非空 hi：闭开区间 [lo,hi)
  let mut seen2 = Vec::new();
  let n2 = t
    .scan_olap(b"d", b"e", 40, |k, _| {
      seen2.push(k.to_vec());
      true
    })
    .await;
  assert_eq!((n2, seen2), (1, vec![b"dead".to_vec()]));
  let mut seen3 = Vec::new();
  let n3 = t
    .scan_olap(b"d", b"dea", 40, |k, _| {
      seen3.push(k.to_vec());
      true
    })
    .await;
  assert_eq!(n3, 0, "hi=dea 排除 dead");
}

/// 胖 GC 的锚点必须是"水位内**最新**可见版本"（上游
/// `FatTupleDifferentAttributes.cpp:74-80` 自最新向旧找首个可见者），
/// 净掉的条数与保留的版本序必须精确——用返回的 purged 数钉死（仅校验可见性
/// 不变无法区分 `rposition`/`position`）。
#[test]
fn fat_gc_purges_to_anchor() {
  let mut ft = FatTuple::default();
  for tx in [10u64, 20, 30, 40, 50] {
    assert!(ft.append(tx, &[tx as u8; 2], 4096));
  }
  // watermark=35 → 锚点 = 最新 <=35 = tx30(index2)；其前 2 条（10/20）净化
  assert_eq!(ft.garbage_collection(35), 2, "锚点=tx30，净掉 tx10/tx20");
  assert_eq!(ft.versions.deltas.len(), 3);
  assert_eq!(ft.versions.deltas[0].tx, 30);
  assert_eq!(ft.versions.deltas[0].offset, 0, "偏移重建从 0 起");
  assert_eq!(ft.reconstruct_tuple(35), Some(vec![30u8, 30]));

  // 全部 <= watermark → 锚点 = 最新(index 末)，net len-1
  let mut ft2 = FatTuple::default();
  for tx in [10u64, 20, 30] {
    assert!(ft2.append(tx, &[1, 2], 4096));
  }
  assert_eq!(ft2.garbage_collection(99), 2);
  assert_eq!(ft2.versions.deltas.len(), 1);
  assert_eq!(ft2.versions.deltas[0].tx, 30, "保留最新版本");

  // 无 <= watermark 的版本 → 净 0，全部保留
  let mut ft3 = FatTuple::default();
  for tx in [10u64, 20] {
    assert!(ft3.append(tx, &[1, 2], 4096));
  }
  assert_eq!(ft3.garbage_collection(5), 0);
  assert_eq!(ft3.versions.deltas.len(), 2);
}
