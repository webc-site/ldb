//! 并发 API 契约（跨三个采样器，非上游单个类）：`Rng` 必须由调用方独占传入，
//! 采样过程不得触碰 fastrand 的线程局部全局 RNG；三个类型必须 `Send + Sync`，
//! 以便 ldb_bench 在多线程压测里共享一个只读采样器、每线程自带 `Rng`。

use std::{sync::Arc, thread};

use fastrand::Rng;
use ldb_zipf::{RejectionInversionZipfSampler, ScrambledZipfGenerator, ZipfGenerator};

const SEED: u64 = 0x1234_5678_9abc_def0;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn samplers_are_send_sync() {
  assert_send_sync::<RejectionInversionZipfSampler>();
  assert_send_sync::<ZipfGenerator>();
  assert_send_sync::<ScrambledZipfGenerator>();
}

/// 固定种子连续跑两次必须逐值相同。若实现偷偷用 `fastrand::f64()`（线程局部全局
/// RNG）而不是传入的 `&mut Rng`，第二次调用会接着全局流走 ⇒ 序列不同。
#[test]
fn explicit_rng_is_the_only_state() {
  let s = RejectionInversionZipfSampler::new(1_000, 0.99);
  let mut a = Vec::with_capacity(64);
  let mut b = Vec::with_capacity(64);
  for half in [&mut a, &mut b] {
    let mut rng = Rng::with_seed(SEED);
    half.extend((0..64).map(|_| s.sample(&mut rng)));
  }
  assert_eq!(a, b);

  let g = ZipfGenerator::new(101, 0.99);
  let take = |g: &ZipfGenerator| {
    let mut rng = Rng::with_seed(SEED);
    (0..64).map(|_| g.rand(&mut rng)).collect::<Vec<_>>()
  };
  assert_eq!(take(&g), take(&g));

  let sc = ScrambledZipfGenerator::new(0, 64, 0.99);
  let take = |sc: &ScrambledZipfGenerator| {
    let mut rng = Rng::with_seed(SEED);
    (0..64).map(|_| sc.rand(&mut rng)).collect::<Vec<_>>()
  };
  assert_eq!(take(&sc), take(&sc));

  // ldb_bench 压测形态：采样器 Arc 共享（只读 &self），每线程一个固定种子 Rng。
  let sampler = Arc::new(ZipfGenerator::new(1_000_001, 0.99));
  let expected = {
    let mut rng = Rng::with_seed(SEED);
    (0..1_000)
      .map(|_| sampler.rand(&mut rng))
      .collect::<Vec<_>>()
  };
  let handles: Vec<_> = (0..4)
    .map(|_| {
      let sampler = Arc::clone(&sampler);
      thread::spawn(move || {
        let mut rng = Rng::with_seed(SEED);
        (0..1_000)
          .map(|_| {
            let v = sampler.rand(&mut rng);
            assert!((1..=1_000_000).contains(&v), "越界: {v}");
            v
          })
          .collect::<Vec<_>>()
      })
    })
    .collect();
  let mut seqs = Vec::with_capacity(handles.len());
  for h in handles {
    let r = h.join();
    assert!(r.is_ok(), "工作线程 panic");
    if let Ok(v) = r {
      seqs.push(v);
    }
  }
  for seq in &seqs {
    assert_eq!(
      seq, &expected,
      "同种子同采样器应逐值相同（线程间无共享状态）"
    );
  }

  // 不同种子必须给出不同序列（否则说明传入的 Rng 被忽略）
  let mut rng = Rng::with_seed(SEED ^ 0xffff);
  let other: Vec<u64> = (0..64).map(|_| sampler.rand(&mut rng)).collect();
  assert_ne!(&other[..], &expected[..64], "不同种子产生了相同前缀");
}
