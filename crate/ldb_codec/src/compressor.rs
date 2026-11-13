//! Blob 压缩缓冲管理（上游 `storage/blob/compressor.cc:30-44` 的 mzstd 收敛版）。
//!
//! 修复上游 C-10：`Reset` 的 `Ensure(buffer == nullptr)` 恒真断言 —— 本实现显式
//! 校验 `expect_size` 页对齐；FSST / LZ4 分支整体剔除（`ZSTDCompressBlob` → mzstd）。

use ldb_type::PAGE_SIZE;
use mzstd::Compressor;

use crate::{
  error::{Error, Result},
  page_codec::PageCodec,
};

impl PageCodec {
  /// 重置并预留 `expect_size` 容量（须页对齐，修复 C-10 的 nullptr 恒真断言）
  pub fn reset(&mut self, expect_size: u64) -> Result<()> {
    if expect_size == 0 || !expect_size.is_multiple_of(PAGE_SIZE as u64) {
      return Err(Error::InvalidConfig(format!(
        "expect_size 未按 {PAGE_SIZE} 对齐: {expect_size}"
      )));
    }
    let size = usize::try_from(expect_size)
      .map_err(|_| Error::InvalidConfig(format!("expect_size 超出 usize: {expect_size}")))?;
    self.destroy_buffer();
    self.buffer = vec![0; size];
    Ok(())
  }

  /// 释放缓冲
  pub fn destroy_buffer(&mut self) {
    self.buffer = Vec::new();
    self.size = 0;
  }

  /// 压缩 blob 到内部缓冲（唯一压缩库 mzstd；`Reset` 语义收敛为「容量不足才重建」）
  ///
  /// 缓冲容量取 `max_compressed_size` 的页对齐上界：空 blob 也至少一页，
  /// 不会因 `2 * len` 为 0 而越界写。
  pub fn zstd_compress_blob(&mut self, blob: &[u8]) -> Result<()> {
    let bound = Compressor::max_compressed_size(blob.len())
      .map_err(|_| Error::InvalidConfig(format!("压缩上界溢出: {} 字节", blob.len())))?;
    let need = bound
      .max(1)
      .div_ceil(PAGE_SIZE)
      .checked_mul(PAGE_SIZE)
      .ok_or_else(|| Error::InvalidConfig(format!("压缩缓冲对齐溢出: {bound}")))?;
    if self.buffer.len() < need {
      self.reset(u64::try_from(need).map_err(|_| Error::InvalidConfig("缓冲尺寸溢出".into()))?)?;
    }
    let compressed = Self::mzstd_compress(blob)?;
    if compressed.len() > self.buffer.len() {
      return Err(Error::InvalidConfig(format!(
        "压缩输出 {} 超出缓冲 {}",
        compressed.len(),
        self.buffer.len()
      )));
    }
    self.buffer[..compressed.len()].copy_from_slice(&compressed);
    self.size = compressed.len();
    // 页对齐写出视图的填充位清零：避免把上一轮 blob 的残留字节写盘/泄露
    let up = self.size.div_ceil(PAGE_SIZE) * PAGE_SIZE;
    self.buffer[self.size..up].fill(0);
    Ok(())
  }

  /// 压缩入口（上游 `CompressBlob` 分发器收敛：唯一算法 mzstd）
  pub fn compress_blob(&mut self, blob: &[u8]) -> Result<()> {
    self.zstd_compress_blob(blob)
  }

  /// 有效数据视图（页对齐向上取整，上游 `ToSpan` 语义；I/O 写出用）
  pub fn to_span(&self) -> &[u8] {
    let up = self.size.div_ceil(PAGE_SIZE) * PAGE_SIZE;
    &self.buffer[..up]
  }

  /// 精确压缩数据视图（解码用，不含对齐填充）
  pub fn as_slice(&self) -> &[u8] {
    &self.buffer[..self.size]
  }

  /// 压缩后字节数
  pub fn compressed_len(&self) -> usize {
    self.size
  }
}
