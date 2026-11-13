//! ldb_bench 错误类型。
//!
//! 基准驱动器对外一律返回 [`Result`]，**不以 panic 表达失败**：非法配置在
//! 启动前经 [`Error::InvalidConfig`] 拒绝，后端失败经
//! [`Error::Backend`] 透明转发，相位后置不变量被破坏经
//! [`Error::Invariant`] 报告（取代早期 phase 失败即 panic 的写法）；
//! 外部 `db_bench` 驱动（定位/启动/退出码/超时/输出解析）失败经
//! [`Error::DbBench`] 报告。

use std::result;

use thiserror::Error;

/// ldb_bench 的统一错误枚举。
#[derive(Debug, Error)]
pub enum Error {
  /// 配置非法（替代上游 Ensure/断言式启动失败）。
  ///
  /// 例：`key_len` 不足以容纳 `num_records` 的十进制位数、`zipf` 指数越界。
  #[error("非法配置: {0}")]
  InvalidConfig(String),

  /// 相位执行后置不变量被破坏（例：`bulk_load` 后 `len != num_records`、
  /// 全表扫描条数与 `len` 不一致）。属被测后端的语义违约，不是驱动器故障。
  #[error("基准不变量被破坏: {0}")]
  Invariant(String),

  /// 后端错误（ldb / RocksDB / redb 适配器）透明转发。
  #[error(transparent)]
  Backend(#[from] anyhow::Error),

  /// 外部 `db_bench` 驱动错误：可执行定位失败、启动失败、非零退出码、超时、
  /// 或输出无法解析。**解析失败一律 Err**，绝不静默返回 0 条结果。
  #[error("db_bench 驱动错误: {0}")]
  DbBench(String),
}

/// ldb_bench 结果别名。
pub type Result<T> = result::Result<T, Error>;
