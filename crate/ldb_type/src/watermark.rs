//! 全局 GSN 落盘水位线。
//!
//! 上游 `ZLeanStore/src/buffer/buffer_manager.cc` 共 12 处直接读
//! `recovery::LogManager::global_min_gsn_flushed`（479/602/898/959/1070/1252/1284/
//! 1316/1319/1322/1326/1537），构成 ldb_buf(L3) → ldb_wal(L3)
//! 环；将该原子量下沉 L0 后两侧共享、互不引用（todo.md §3.0 反向边 1）。

use std::sync::atomic::{AtomicU64, Ordering};

/// 单调递增的「最小已刷盘 GSN」水位线
#[derive(Debug, Default)]
pub struct GsnWatermark {
  inner: AtomicU64,
}

impl GsnWatermark {
  /// 以初始水位构造（`const`：可直接作 `static` 全局水位，对齐上游
  /// `LogManager::global_min_gsn_flushed` 的 `inline static` 形态）
  #[must_use]
  #[inline]
  pub const fn new(init: u64) -> Self {
    Self {
      inner: AtomicU64::new(init),
    }
  }

  /// 读当前水位
  #[inline]
  pub fn load(&self) -> u64 {
    self.inner.load(Ordering::Acquire)
  }

  /// 直接设置（仅恢复 / 测试路径使用）
  #[inline]
  pub fn store(&self, v: u64) {
    self.inner.store(v, Ordering::Release);
  }

  /// 推进到 `v` 与当前水位的较大者（单调不回退），返回推进后的值
  #[inline]
  pub fn advance_min(&self, v: u64) -> u64 {
    let mut cur = self.inner.load(Ordering::Relaxed);
    while cur < v {
      match self
        .inner
        .compare_exchange_weak(cur, v, Ordering::AcqRel, Ordering::Relaxed)
      {
        Ok(_) => return v,
        Err(actual) => cur = actual,
      }
    }
    cur
  }
}
