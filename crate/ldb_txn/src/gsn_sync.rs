//! GSN 依赖同步（上游 `sync/page_guard.*` 的 GSN 半区，经泛型注入 L1）。
//!
//! L1 `ldb_sync::PageGuard` 不能在 L3/L4 固有 impl（孤儿规则）——
//! 本模块以独立函数 + 泛型上下文承载 DetectGSNDependency / AdvanceGSN。

use ldb_type::{LogId, PageId};

/// GSN 上下文（页守卫携带的最小状态）
#[derive(Debug, Clone, Copy, Default)]
pub struct GsnCtx {
  /// 本页 GSN（页头 p_gsn）
  pub page_gsn: LogId,
  /// 已刷盘水位（global_min_gsn_flushed）
  pub flushed: LogId,
}

/// 依赖检测：页 GSN 超前于刷盘水位 → 写回须等待（上游 DetectGSNDependency）
///
/// 返回 true = 检测到依赖（调用方须在提交前等待该页落盘）
#[inline]
pub fn detect_gsn_dependency(ctx: &GsnCtx) -> bool {
  ctx.page_gsn > ctx.flushed
}

/// 推进页 GSN（取观测到的最大值）
#[inline]
pub fn advance_gsn(cur: LogId, observed: LogId) -> LogId {
  cur.max(observed)
}

/// PageGuard 语义包装（账本 fn 名对齐）
#[derive(Debug, Default)]
pub struct PageGuard;

impl PageGuard {
  /// 依赖检测（账本形态）
  pub async fn detect_gsn_dependency(&self, pid: PageId, page_gsn: LogId, flushed: LogId) -> bool {
    let _ = pid;
    page_gsn > flushed
  }

  /// 推进 GSN（账本形态）
  pub async fn advance_gsn(&self, cur: LogId, observed: LogId) -> LogId {
    advance_gsn(cur, observed)
  }
}
