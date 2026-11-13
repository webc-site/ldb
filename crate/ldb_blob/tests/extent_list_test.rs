//! ExtentList / TailExtent 回归（上游 `storage/extent/extent_list.{h,cc}`）：
//! tier 定级 / 拆级块尺寸 / 无跨界判定 / 随机 append 序列重建。

use std::thread;

use ldb_blob::{Error, Extent, ExtentList, Iterator, MAX_EXTENTS, extent_size, tier_index};
use ldb_type::{PAGE_SIZE, PageId};

#[test]
fn extent_list_tiers_and_split() {
  assert_eq!(
    (extent_size(0), extent_size(3), extent_size(10)),
    (1, 8, 1024)
  );
  assert_eq!((tier_index(1), tier_index(5), tier_index(1024)), (0, 3, 10));

  let el = ExtentList::new(0, 4);
  // 8 页数据 → 尾容量 4 → 拆出 1 个整级数据块（4 页），余 4 页仍在尾
  let blocks = el.append(&vec![0u8; 8 * 4096]).expect("未越上限");
  assert_eq!(blocks.len(), 1, "8 页 / 尾容量 4 → 恰好拆 1 个整级块");
  assert_eq!(blocks[0].len(), 4 * 4096);
  assert_eq!(el.tail.read().used_pages, 4, "余 4 页留在尾");
  // 调用方（BlobManager）为每块独立分配区段并登记 extent
  el.push_extent(Extent {
    start_pid: 8,
    tier: 2,
  });
  assert_eq!(el.number_of_extents(), 1);
  assert_eq!(el.begin().map(|e| e.tier), Some(2));
  assert_eq!(el.total_size_extents(), 4 * 4096_u64);

  // 迭代快照
  let iter = Iterator::new(el.iter(), Box::new(|_, _, out| out.extend_from_slice(&[1])));
  assert_eq!(iter.tier_index(), Some(2));
}

/// `no_span_extents`（上游 `NoSpanExtents`）：extent 是否都不跨物理块边界。
#[test]
fn no_span_extents_predicate() {
  let el = ExtentList::new(0, 4);
  el.push_extent(Extent {
    start_pid: 0,
    tier: 2,
  }); // [0,4)
  el.push_extent(Extent {
    start_pid: 4,
    tier: 2,
  }); // [4,8)
  assert!(el.no_span_extents(8), "两个 4 页 extent 均落在块内");

  let crossing = ExtentList::new(0, 4);
  crossing.push_extent(Extent {
    start_pid: 6,
    tier: 2,
  }); // [6,10) 跨块
  assert!(!crossing.no_span_extents(8), "越界 extent 必须判为跨界");

  let big = ExtentList::new(0, 64);
  big.push_extent(Extent {
    start_pid: 3,
    tier: 6,
  }); // 64 页 >= 块大小
  assert!(big.no_span_extents(8), "extent 尺寸 >= 块大小则恒不跨界");
}

/// LCG 伪随机（避免引入 dev-dep；测试内自足）。
struct Lcg(u64);

impl Lcg {
  fn below(&mut self, n: u64) -> u64 {
    self.0 = self
      .0
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    (self.0 >> 33) % n
  }
}

/// ExtentList 随机 append 序列：重建流必须与原流逐字节一致；拆出的每块都是
/// 整级尺寸、尾区始终容纳得下（`used_pages <= capacity`）。
fn extent_random_sequence(rounds: usize) {
  let cap = 4u64;
  let tier = tier_index(cap);
  let el = ExtentList::new(0, cap);
  let mut rng = Lcg(0x1234_5678_9abc_def0);
  let mut oracle: Vec<u8> = Vec::new();
  let mut next_pid: PageId = 0;
  let mut ext_pages: u64 = 0;
  let mut ext_bytes: Vec<Vec<u8>> = Vec::new();

  for _ in 0..rounds {
    // 追加 0..3 页的随机字节（覆盖 0 字节、单页、跨页、恰好整级等边界）
    let n = rng.below(3 * PAGE_SIZE as u64 + 1) as usize;
    let chunk: Vec<u8> = (0..n).map(|_| rng.below(256) as u8).collect();
    // 上限语义：`append` 在改动尾缓冲前返回 `TooManyExtents` ⇒ 该 chunk 未生效，
    // oracle 必须同步丢弃（否则重建断言会假失败）
    let blocks = match el.append(&chunk) {
      Ok(blocks) => blocks,
      Err(Error::TooManyExtents) => {
        assert!(
          el.number_of_extents() as usize <= MAX_EXTENTS,
          "拒绝时已登记 extent 数不得越过上限"
        );
        break;
      }
      Err(e) => panic!("意外错误: {e}"),
    };
    oracle.extend_from_slice(&chunk);
    // 模拟 BlobManager：每块独立分配区段（这里用单调 pid），登记 extent
    for block in blocks {
      assert_eq!(
        block.len() as u64,
        extent_size(tier) * PAGE_SIZE as u64,
        "拆出的块必须是整级尺寸"
      );
      let ext = Extent {
        start_pid: next_pid,
        tier,
      };
      el.push_extent(ext);
      next_pid += ext.pages();
      ext_pages += ext.pages();
      ext_bytes.push(block);
    }
    assert_eq!(el.number_of_extents() as usize, ext_bytes.len());
    assert_eq!(el.total_size_extents(), ext_pages * PAGE_SIZE as u64);
    assert!(
      el.tail.read().used_pages <= cap,
      "尾缓冲不得超过容量（否则区段会溢出）"
    );
  }

  // 尾区起始 pid 恒定（拆级不再前移尾段）；重建 == 原流
  assert_eq!(el.tail.read().start_pid, 0);
  let mut recon: Vec<u8> = Vec::with_capacity(oracle.len());
  for b in &ext_bytes {
    recon.extend_from_slice(b);
  }
  recon.extend_from_slice(&el.tail.read().data);
  assert_eq!(recon, oracle, "随机 append 序列重建流必须与原流一致");
}

#[test]
fn extent_random_sequence_short() {
  // 768 轮：算法覆盖充分且 < 100ms（E2）；1e5 轮见下方 #[ignore]
  extent_random_sequence(768);
}

#[test]
#[ignore = "1e5 随机 append 序列（重活，手动 --ignored 运行）"]
fn extent_random_sequence_1e5() {
  extent_random_sequence(100_000);
}

/// 长链 Drop 必须迭代拆解：Arc 链默认递归 Drop 在 1e5 节点时栈溢出（SIGABRT）。
/// 用 256 KiB 小栈线程复现/钉死，防止回归。
#[test]
fn long_chain_drop_does_not_overflow_stack() {
  thread::Builder::new()
    .stack_size(256 * 1024)
    .spawn(|| {
      let el = ExtentList::new(0, 2);
      for i in 0..200_000u64 {
        el.push_extent(Extent {
          start_pid: i * 4,
          tier: 0,
        });
      }
      assert_eq!(el.number_of_extents(), 200_000);
      drop(el);
    })
    .expect("spawn")
    .join()
    .expect("200k 节点长链 Drop 不得栈溢出");
}

/// 上限语义：127 个 extent 可登记，第 128 个在**改动尾缓冲之前**被拒
/// （上游 `ExtentList::EXTENT_CNT_MASK = 127`，BlobState 定长 1100 B 放不下）。
#[test]
fn extent_count_cap_rejects_128th() {
  let cap = 4u64;
  let tier = tier_index(cap);
  let el = ExtentList::new(0, cap);
  let full_pages = extent_size(tier);
  let one_extent = vec![0u8; (full_pages * PAGE_SIZE as u64) as usize];

  // 首个 append 只填尾缓冲（used=full 不满足 >，不拆级），之后每次恰好拆 1 块
  let mut next_pid = 0u64;
  let mut guard = 0;
  while el.number_of_extents() < MAX_EXTENTS as u64 {
    guard += 1;
    assert!(guard <= MAX_EXTENTS + 4, "达到上限所需的 append 次数超预期");
    for block in el.append(&one_extent).expect("127 个以内必须成功") {
      assert_eq!(block.len() as u64, full_pages * PAGE_SIZE as u64);
      el.push_extent(Extent {
        start_pid: next_pid,
        tier,
      });
      next_pid += full_pages;
    }
  }
  assert_eq!(el.number_of_extents() as usize, MAX_EXTENTS);

  let before_len = el.tail.read().data.len();
  let before_bytes = el.total_size_extents();
  let err = el
    .append(&one_extent)
    .expect_err("第 128 个 extent 必须被拒");
  assert!(matches!(err, Error::TooManyExtents));
  assert_eq!(el.tail.read().data.len(), before_len, "拒绝不得改动尾缓冲");
  assert_eq!(el.number_of_extents() as usize, MAX_EXTENTS);
  assert_eq!(el.total_size_extents(), before_bytes);
}
