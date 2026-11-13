//! ldb_pack 装箱回归（账本 target `bin_packing_per_block` / `copy_compressed_pages`）。
//!
//! 上游 `io_interface.cc:1168-1221` / `:1223-1321`；4 KiB 不变量见 todo.md:183。

use std::{
  any::Any,
  iter::repeat_n,
  panic::{AssertUnwindSafe, catch_unwind},
};

use ldb_pack::{
  Bin, BinSlot, MAX_PAGES_PER_BIN, ZipEntry, bin_packing_per_block, copy_compressed_pages,
};

/// 构造装箱输入（death_time 取页号便于定位）
fn entries(sizes: &[u16]) -> Vec<ZipEntry> {
  sizes
    .iter()
    .enumerate()
    .map(|(i, &s)| ZipEntry {
      comp_sz: s,
      death_time: i as u64,
      src_off: 0,
    })
    .collect()
}

fn bins(states: &[(u32, u32)]) -> Vec<Bin> {
  states
    .iter()
    .map(|&(sum, count)| Bin { sum, count })
    .collect()
}

/// best-fit：`lower_bound` 取最小可容纳剩余空间（不是首个空 bin）
#[test]
fn best_fit_picks_smallest_remaining() {
  let mut b = bins(&[(0, 0), (2000, 1)]);
  let plan = bin_packing_per_block(&entries(&[1000]), &mut b);

  assert_eq!(plan, vec![Some(BinSlot { bin: 1, off: 2000 })]);
  assert_eq!(b[1].sum, 3000);
  assert_eq!(b[1].count, 2);
  assert_eq!(b[0].sum, 0);
}

/// 剩余空间恰好等于页大小：`lower_bound` 含等号，必须选该 bin（不是空 bin）
#[test]
fn exact_fit_uses_zero_remaining() {
  let mut b = bins(&[(2048, 1), (0, 0)]);
  let plan = bin_packing_per_block(&entries(&[2048]), &mut b);

  assert_eq!(plan, vec![Some(BinSlot { bin: 0, off: 2048 })]);
  assert_eq!((b[0].sum, b[0].count), (4096, 2));
  assert_eq!((b[1].sum, b[1].count), (0, 0));
}

/// 同剩余空间的 bin 按插入序 FIFO（上游 multimap 等价键顺序），不是 LIFO
#[test]
fn fifo_tie_breaking_equal_remaining() {
  let mut b = bins(&[(1000, 1), (1000, 1)]);
  let plan = bin_packing_per_block(&entries(&[1000, 1000]), &mut b);

  assert_eq!(
    plan,
    vec![
      Some(BinSlot { bin: 0, off: 1000 }),
      Some(BinSlot { bin: 1, off: 1000 }),
    ]
  );
  assert_eq!((b[0].sum, b[0].count), (2000, 2));
  assert_eq!((b[1].sum, b[1].count), (2000, 2));
}

/// 回写契约：`plan[i]` 属于入参 `entries[i]`（内部按尺寸降序放置不改变下标对齐）
#[test]
fn plan_aligned_with_input_order() {
  // 输入升序 [500,1000,2000,4000]，降序放置顺序为 4000→2000→1000→500
  let mut b = bins(&[(0, 0); 4]);
  let plan = bin_packing_per_block(&entries(&[500, 1000, 2000, 4000]), &mut b);

  assert_eq!(
    plan,
    vec![
      Some(BinSlot { bin: 2, off: 0 }),
      Some(BinSlot { bin: 1, off: 2000 }),
      Some(BinSlot { bin: 1, off: 0 }),
      Some(BinSlot { bin: 0, off: 0 }),
    ]
  );
}

/// 无可用 bin：留 None（上游会新建 bin，本函数不扩容）
#[test]
fn no_bin_fits_returns_none() {
  let mut b = bins(&[(4096, 0)]);
  assert_eq!(bin_packing_per_block(&entries(&[1000]), &mut b), vec![None]);

  let mut empty: Vec<Bin> = Vec::new();
  assert_eq!(
    bin_packing_per_block(&entries(&[1000, 2000]), &mut empty),
    vec![None, None]
  );

  // 空输入
  assert!(bin_packing_per_block(&[], &mut bins(&[(0, 0)])).is_empty());
}

/// 每 4 KiB 最多 2 页（PAGE_SIZE / MIN_COMP_SIZE = 4096 / 2048）
#[test]
fn page_count_cap_is_two() {
  assert_eq!(MAX_PAGES_PER_BIN, 2);
  let mut b = bins(&[(0, 0)]);
  let plan = bin_packing_per_block(&entries(&[1000, 1000, 1000]), &mut b);

  assert_eq!(
    plan,
    vec![
      Some(BinSlot { bin: 0, off: 0 }),
      Some(BinSlot { bin: 0, off: 1000 }),
      None,
    ]
  );
  assert_eq!((b[0].sum, b[0].count), (2000, 2));

  // count 已达上限的 bin 即使有剩余空间也不得入选
  let mut full = bins(&[(2000, 2)]);
  assert_eq!(
    bin_packing_per_block(&entries(&[1000]), &mut full),
    vec![None]
  );
}

/// 当前页放不下时必须继续尝试后续（更小的）页，不得提前终止
#[test]
fn unfittable_page_does_not_abort_rest() {
  // 唯一 bin 剩 2096B：3000 放不下，100 仍应随后装入
  let mut b = bins(&[(2000, 1)]);
  let plan = bin_packing_per_block(&entries(&[3000, 100]), &mut b);

  assert_eq!(plan, vec![None, Some(BinSlot { bin: 0, off: 2000 })]);
  assert_eq!((b[0].sum, b[0].count), (2100, 2));
}

/// 混合尺寸装箱：bin 内页不跨 4 KiB，sum 不超页大小
#[test]
fn mixed_sizes_never_cross_4k_in_bin() {
  let sizes = [3000u16, 1000, 2000];
  let mut b = bins(&[(0, 0); 3]);
  let plan = bin_packing_per_block(&entries(&sizes), &mut b);

  // 3000→bin0@0，1000 选剩余 1096 的 bin0@3000，2000→bin1@0
  assert_eq!(
    plan,
    vec![
      Some(BinSlot { bin: 0, off: 0 }),
      Some(BinSlot { bin: 0, off: 3000 }),
      Some(BinSlot { bin: 1, off: 0 }),
    ]
  );
  for (e, slot) in entries(&sizes).iter().zip(plan.iter().flatten()) {
    let start = slot.off as usize;
    let end = start + e.comp_sz as usize;
    assert_eq!(start / 4096, (end - 1) / 4096, "页跨 4KiB: {start}..{end}");
  }
  for bin in &b {
    assert!(bin.sum as usize <= 4096);
    assert!(bin.count <= 2);
  }
}

/// 契约违约：`Bin.sum > PAGE_SIZE`（ldb_space stale-bin 实测路径）在 release 也 panic，
/// 不得静默回绕成近 4 GiB 剩余空间并把 `off` 写到槽位外
#[test]
fn bin_sum_over_page_panics() {
  // 上游 uniform 把全局偏移写进 sum 后的实测状态：两个不可压缩页 → [4096, 8192]
  let mut b = bins(&[(4096, 1), (8192, 1)]);
  let err = catch_unwind(AssertUnwindSafe(|| {
    bin_packing_per_block(&entries(&[4096, 4096]), &mut b)
  }))
  .expect_err("Bin.sum=8192 必须 panic");
  assert_eq!(panic_msg(&*err), "Bin.sum 超 PAGE_SIZE（调用方违约）");
}

/// 契约违约：`comp_sz > PAGE_SIZE` 不得静默返回 None（消费者会静默跳过该页）
#[test]
fn bin_entry_comp_sz_over_page_panics() {
  let mut b = bins(&[(0, 0)]);
  let err = catch_unwind(AssertUnwindSafe(|| {
    bin_packing_per_block(&entries(&[4097]), &mut b)
  }))
  .expect_err("comp_sz=4097 必须 panic");
  assert_eq!(panic_msg(&*err), "comp_sz 超 PAGE_SIZE（调用方违约）");
}

/// 契约违约：copy 的 `comp_sz > PAGE_SIZE` 无法满足 4 KiB 不变量，必须 panic
#[test]
fn copy_comp_sz_over_page_panics() {
  let src = vec![ZipEntry {
    comp_sz: 4097,
    death_time: 1,
    src_off: 0,
  }];
  let buf = vec![0u8; 4097];
  let mut dst_buf = vec![0u8; 2 * 4096];
  let err = catch_unwind(AssertUnwindSafe(|| {
    copy_compressed_pages(&src, &buf, &mut dst_buf, 0, 0)
  }))
  .expect_err("comp_sz=4097 必须 panic");
  assert_eq!(panic_msg(&*err), "comp_sz 超 PAGE_SIZE（调用方违约）");
}

/// 构造源缓冲：每页字节 = 页号，便于校验拷贝内容
fn src_buf(sizes: &[u16]) -> Vec<u8> {
  let mut v = Vec::new();
  for (i, &s) in sizes.iter().enumerate() {
    v.extend(repeat_n((i as u8).wrapping_add(1), s as usize));
  }
  v
}

fn with_offsets(sizes: &[u16]) -> Vec<ZipEntry> {
  let mut off = 0u32;
  let mut out = Vec::new();
  for (i, &s) in sizes.iter().enumerate() {
    out.push(ZipEntry {
      comp_sz: s,
      death_time: i as u64 + 1,
      src_off: off,
    });
    off += u32::from(s);
  }
  out
}

/// 段页数上限：第 3 页对齐到下一 4 KiB
#[test]
fn copy_segment_cap_two_pages() {
  let sizes = [1000u16, 1000, 1000];
  let src = with_offsets(&sizes);
  let mut dst_buf = vec![0u8; 3 * 4096];
  let (out, avg) = copy_compressed_pages(&src, &src_buf(&sizes), &mut dst_buf, 0, 2);

  assert_eq!(
    out.iter().map(|e| e.src_off).collect::<Vec<_>>(),
    vec![0, 1000, 4096]
  );
  assert_eq!(avg, 2, "(1+2+3)/3");
  assert_eq!(out.len(), 3);
}

/// 精确填满 4 KiB（2048+2048）与整页 4096：既不换段也不跨界
#[test]
fn copy_exact_fill_pages_no_cross() {
  let sizes = [2048u16, 2048, 4096];
  let src = with_offsets(&sizes);
  let mut dst_buf = vec![0u8; 3 * 4096];
  let (out, avg) = copy_compressed_pages(&src, &src_buf(&sizes), &mut dst_buf, 0, 2);

  assert_eq!(
    out.iter().map(|e| e.src_off).collect::<Vec<_>>(),
    vec![0, 2048, 4096]
  );
  assert_eq!(avg, 2, "(1+2+3)/3");
  for (i, o) in out.iter().enumerate() {
    let start = o.src_off as usize;
    let end = start + sizes[i] as usize;
    assert_eq!(start / 4096, (end - 1) / 4096, "页跨 4KiB: {start}..{end}");
  }
  assert_eq!(&dst_buf[..2048], &[1u8; 2048][..]);
  assert_eq!(&dst_buf[2048..4096], &[2u8; 2048][..]);
  assert_eq!(&dst_buf[4096..8192], &[3u8; 4096][..]);
}

/// 混合尺寸：3000+3000 不得跨界（上游仅按页数对齐会写到 3000，跨 4 KiB）
#[test]
fn copy_mixed_sizes_do_not_cross_4k() {
  let sizes = [3000u16, 3000];
  let src = with_offsets(&sizes);
  let mut dst_buf = vec![0u8; 2 * 4096 * 2];
  let (out, avg) = copy_compressed_pages(&src, &src_buf(&sizes), &mut dst_buf, 0, 1);

  assert_eq!(
    out.iter().map(|e| e.src_off).collect::<Vec<_>>(),
    vec![0, 4096]
  );
  assert_eq!(avg, 1);
  for (i, o) in out.iter().enumerate() {
    let start = o.src_off as usize;
    assert_eq!(&dst_buf[start..start + 3000], &vec![i as u8 + 1; 3000][..]);
  }
}

/// 7×600B：每 2 页一段，全部不跨 4 KiB
#[test]
fn copy_seven_600b_no_cross() {
  let sizes = [600u16; 7];
  let src = with_offsets(&sizes);
  let mut dst_buf = vec![0u8; 7 * 4096];
  let (out, avg) = copy_compressed_pages(&src, &src_buf(&sizes), &mut dst_buf, 0, 6);

  let offs: Vec<u32> = out.iter().map(|e| e.src_off).collect();
  assert_eq!(offs, vec![0, 600, 4096, 4696, 8192, 8792, 12288]);
  assert_eq!(avg, 4);
  for o in &offs {
    assert_eq!(o / 4096, ((*o + 599) / 4096), "页跨 4KiB: {o}");
  }
}

/// avg_edt 整数除法向零截断（上游 `avg_edt / copied_pages_cnt`）
#[test]
fn copy_avg_edt_truncates() {
  let sizes = [100u16, 100];
  let src = vec![
    ZipEntry {
      comp_sz: 100,
      death_time: 1,
      src_off: 0,
    },
    ZipEntry {
      comp_sz: 100,
      death_time: 2,
      src_off: 100,
    },
  ];
  let mut dst_buf = vec![0u8; 4096];
  let (out, avg) = copy_compressed_pages(&src, &src_buf(&sizes), &mut dst_buf, 0, 1);

  assert_eq!(avg, 1, "(1+2)/2 = 1.5 截断为 1");
  assert_eq!(out.len(), 2);
}

/// 区间闭区间 + 跳过前缀：只拷贝 [start,end]，输出 src_off 从 0 重新起算
#[test]
fn copy_range_inclusive_and_skips_prefix() {
  let sizes = [600u16; 5];
  let src = with_offsets(&sizes);
  let mut dst_buf = vec![0u8; 5 * 4096];
  let (out, avg) = copy_compressed_pages(&src, &src_buf(&sizes), &mut dst_buf, 1, 3);

  assert_eq!(
    out.iter().map(|e| e.src_off).collect::<Vec<_>>(),
    vec![0, 600, 4096]
  );
  assert_eq!(avg, 3, "(2+3+4)/3");
  assert_eq!(out[0].death_time, 2);
  assert_eq!(out[2].death_time, 4);
  // 第 1 页（源偏移 600，内容 0x02）落在目标偏移 0
  assert_eq!(&dst_buf[..600], &[2u8; 600][..]);
  // 第 3 页（源偏移 1800，内容 0x04）落在目标偏移 4096
  assert_eq!(&dst_buf[4096..4096 + 600], &[4u8; 600][..]);
}

/// 短目标缓冲：切片越界 panic（调用方契约：按页数 × PAGE_SIZE 预留）
#[test]
fn copy_short_dst_buf_panics() {
  let sizes = [3000u16, 3000];
  let src = with_offsets(&sizes);
  let buf = src_buf(&sizes);
  let mut dst_buf = vec![0u8; 4096];
  assert!(
    catch_unwind(AssertUnwindSafe(|| {
      copy_compressed_pages(&src, &buf, &mut dst_buf, 0, 1)
    }))
    .is_err()
  );
}

/// 非法区间：start>end / end 越界 必须 panic，且命中的是显式契约断言
#[test]
fn copy_invalid_range_panics() {
  let sizes = [600u16; 3];
  let src = with_offsets(&sizes);
  let buf = src_buf(&sizes);
  let mut dst_buf = vec![0u8; 3 * 4096];

  let e1 = catch_unwind(AssertUnwindSafe(|| {
    copy_compressed_pages(&src, &buf, &mut dst_buf, 2, 1)
  }))
  .expect_err("start>end 必须 panic");
  assert_eq!(panic_msg(&*e1), "页区间越界");
  let e2 = catch_unwind(AssertUnwindSafe(|| {
    copy_compressed_pages(&src, &buf, &mut dst_buf, 0, 3)
  }))
  .expect_err("end 越界必须 panic");
  assert_eq!(panic_msg(&*e2), "页区间越界");
}

/// 取 panic 载荷文本（`assert!` 字面量消息为 `&str`，格式化消息为 `String`）
fn panic_msg(err: &(dyn Any + Send)) -> &str {
  err
    .downcast_ref::<&str>()
    .copied()
    .or_else(|| err.downcast_ref::<String>().map(String::as_str))
    .unwrap_or("<无文本>")
}
