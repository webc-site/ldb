//! 页与 Blob 的 mzstd 编解码 + CRC-32 + SHA-256（唯一压缩库收敛点）。
//!
//! 上游 `io_interface.cc:398-527` 的 LZ4/Zstd/Zlib 三分支已收敛为 mzstd（level 1，
//! 对齐 `:492-494` 的 `ZSTD_compress(..., 1)`）；不可压缩页回退存原文
//! （哨兵 `comp_sz == PAGE_SIZE`，修复上游 C-19 必崩缺陷）；解压以 `comp_sz`
//! 为入参并按其切片，不读映射表（消除 L2 → L3 反向依赖）。
//!
//! 页完整性保持上游语义：**页帧无 checksum**。上游 `CompressPage`
//! （`io_interface.cc:472-527`）只写压缩帧，`DecompressPage` 只断言输出恰为一页
//! （`:469`），`PageHeader`（`page.h:12-20`）也无校验字段。zstd 帧头/块头与
//! 「输出必须恰为 4096 字节」只能拦住结构性损坏：round3 实测单比特翻转在文本/模式页
//! 有 72%～77% 仍解出「长度正确」的错误页，原文回退页 100% 静默；页级校验需
//! `ldb_page`/`ldb_pack`/`ldb_space` 联合改布局，见 round3 报告 NOTES 与 CROSS。

use std::cell::RefCell;

use crc32fast::Hasher;
use ldb_pack::UNCOMPRESSED;
use ldb_type::PAGE_SIZE;
use mzstd::{
  Compressor, DecodeLimits, DecoderConfig, Decompressor, EncodeError, EncoderConfig, Level,
};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// 页压缩级别：上游 `io_interface.cc:494` 固定 level 1（速度优先）。
const PAGE_LEVEL: i32 = 1;

/// 页帧窗口上限：单段帧窗口 = 内容长度 4096 → log2 = 12。
///
/// 恶意帧若声明更大窗口（zstd 参考解码器默认允许 128 MiB），会被解码器在
/// 读取任何块之前拒绝，避免按声明值预分配。
const PAGE_WINDOW_LOG: u8 = 12;

/// Blob 解压窗口上限（zstd 参考解码器默认 2^27 = 128 MiB）。
const BLOB_WINDOW_LOG_MAX: u8 = 27;

/// 合法窗口 log 下限（zstd 最小窗口 1 KiB）。
const WINDOW_LOG_MIN: u8 = 10;

thread_local! {
  /// 页压缩器：线程局部复用（`Compressor` 内含引擎与流状态，逐页重建 = 逐页分配）。
  static ENC: RefCell<Compressor> = RefCell::new(new_page_compressor());
  /// 页解压器：窗口硬限 4 KiB（页帧只可能是单段 4096 字节窗口）。
  static DEC: RefCell<Decompressor> = RefCell::new(Decompressor::new(page_decoder_config()));
  /// Blob 解压器：默认窗口上限，供可信/已知规模数据使用。
  static DEC_BLOB: RefCell<Decompressor> = RefCell::new(Decompressor::default());
}

/// 页压缩器：level 1（上游页压缩固定值）。
fn new_page_compressor() -> Compressor {
  let level = Level::try_from(PAGE_LEVEL).unwrap_or(Level::DEFAULT);
  // level 1 必在合法区间；构造失败只可能是未来 API 变更，回退默认配置而非 panic。
  Compressor::new(EncoderConfig::default().with_level(level)).unwrap_or_default()
}

/// 页解压配置：窗口硬限 4 KiB。`PAGE_WINDOW_LOG` 非法时编译期失败。
const fn page_decoder_config() -> DecoderConfig {
  const LIMITS: DecodeLimits = match DecodeLimits::DEFAULT.with_max_window_log(PAGE_WINDOW_LOG) {
    Ok(limits) => limits,
    Err(_) => panic!("PAGE_WINDOW_LOG 超出 zstd 窗口范围"),
  };
  DecoderConfig::DEFAULT.with_limits(LIMITS)
}

/// 按 `expect_size` 计算解压所需的最小窗口 log，夹在 `[1 KiB, 128 MiB]`。
fn window_log_for(expect_size: usize) -> u8 {
  let ceil_log = if expect_size <= 1 {
    0
  } else {
    (usize::BITS - (expect_size - 1).leading_zeros()) as u8
  };
  ceil_log.clamp(WINDOW_LOG_MIN, BLOB_WINDOW_LOG_MAX)
}

/// 页 / Blob 编解码器（页路径无状态；Blob 路径持有压缩缓冲与解压输出缓冲）
#[derive(Debug, Default)]
pub struct PageCodec {
  pub(crate) buffer: Vec<u8>,
  pub(crate) size: usize,
  /// Blob 解压输出（与压缩缓冲分离，避免 `as_slice` 视图被解压覆盖）
  pub(crate) decomp: Vec<u8>,
}

impl PageCodec {
  /// 构造
  pub fn new() -> Self {
    Self::default()
  }
}

impl PageCodec {
  /// mzstd 压缩（通用字节流；返回新分配的压缩缓冲，供 Blob 路径使用）
  pub fn mzstd_compress(src: &[u8]) -> Result<Vec<u8>> {
    let bound = Compressor::max_compressed_size(src.len())
      .map_err(|_| Error::InvalidConfig(format!("压缩上界溢出: {} 字节", src.len())))?;
    ENC.with(|enc| {
      let mut enc = enc.borrow_mut();
      let mut out = Vec::with_capacity(bound);
      enc.compress_into(src, &mut out).map_err(Error::from)?;
      Ok(out)
    })
  }

  /// mzstd 解压（通用字节流；输出规模由帧声明，仅供可信/已知规模数据）
  ///
  /// 不可信输入请用 [`Self::zstd_decompress_blob`]：输出写入定长切片，
  /// 解压放大在 4 KiB 处即被拒绝。本函数的解码器窗口上限为 2^27，
  /// 4 KiB 输入最多再生 ~128 MiB（zstd 块上限），不会按帧头声明值预分配。
  pub fn mzstd_decompress(src: &[u8]) -> Result<Vec<u8>> {
    DEC_BLOB.with(|dec| dec.borrow_mut().decompress(src).map_err(Error::from))
  }

  /// CRC-32 校验码（IEEE 802.3 / ISO-HDLC，多项式 0x04C11DB7，校验值 0xCBF43926）。
  ///
  /// 与上游 WAL 条目校验逐位一致：`utils.cc:49-50` 调用 `CRC::CRC_32()`，
  /// 该参数集在 `share_headers/crc.h:1523-1540` 定义为多项式 0x04C11DB7、
  /// init/final XOR 0xFFFFFFFF、输入输出反射（check value 0xCBF43926），
  /// 即 IEEE CRC-32，**不是 CRC32C/Castagnoli**；`log_entry.h:31-39` 用它校验
  /// WAL 条目。上游压缩页本身无 checksum（`io_interface.cc:472-527` 的
  /// `CompressPage` 只做 LZ4/Zstd/Zlib 压缩，不写校验字段）。
  /// `crc32fast` 与本参数集字节级兼容。
  pub fn crc32_checksum(data: &[u8]) -> u32 {
    let mut h = Hasher::new();
    h.update(data);
    h.finalize()
  }

  /// SHA-256 内容指纹（Blob 去重索引）
  pub fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
  }

  /// 压缩一个 4 KiB 数据页，返回压缩后长度。
  ///
  /// 膨胀（不可压缩）或恰好一页时回退存原文：返回 `UNCOMPRESSED` 哨兵，
  /// 此时 `dst` 内容无意义，调用方必须存 `src` 原文（上游 C-19 必崩路径的修复）。
  /// 压缩帧直写 `dst`：热路径零分配、零中间拷贝。
  pub fn compress_page(src: &[u8; PAGE_SIZE], dst: &mut [u8; PAGE_SIZE]) -> Result<u16> {
    ENC.with(|enc| {
      let mut enc = enc.borrow_mut();
      match enc.compress_to_slice(src, dst) {
        // n < PAGE_SIZE <= u16::MAX，转换安全
        Ok(n) if n < PAGE_SIZE => Ok(n as u16),
        // 恰好占满一页：与原文同尺寸，无存储收益
        Ok(_) => Ok(UNCOMPRESSED),
        // 目标缓冲装不下：不可压缩，回退原文
        Err(EncodeError::OutputTooSmall { .. }) => Ok(UNCOMPRESSED),
        Err(e) => Err(Error::from(e)),
      }
    })
  }

  /// 解压一个数据页（直写 `dst`，无中间 Vec 分配）。
  ///
  /// `comp_sz` 是权威长度（来自 `Pid2Offset` 的 15-bit 压缩尺寸域）：
  /// - `comp_sz == UNCOMPRESSED`：原文直存，要求 `comp` 至少一页，取前 4096 字节；
  /// - `1..PAGE_SIZE`：只解压 `comp[..comp_sz]`（`comp` 可含后续装箱页，忽略多余字节）。
  ///
  /// 截断、`comp_sz` 越界、解压输出非一页一律返回 `Err`，绝不部分写入后报 Ok。
  pub fn decompress_page(comp: &[u8], comp_sz: u16, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
    if comp_sz == 0 || comp_sz as usize > PAGE_SIZE {
      return Err(Error::InvalidCompressedSize(comp_sz));
    }
    let need = comp_sz as usize;
    if comp.len() < need {
      return Err(Error::Truncated {
        expected: need,
        got: comp.len(),
      });
    }
    let comp = &comp[..need];
    if need == PAGE_SIZE {
      dst.copy_from_slice(comp);
      return Ok(());
    }
    let written = DEC.with(|dec| {
      let mut dec = dec.borrow_mut();
      dec.decompress_to_slice(comp, dst).map_err(Error::from)
    })?;
    if written != PAGE_SIZE {
      return Err(Error::CorruptPage(written));
    }
    Ok(())
  }

  /// 解压 Blob 到内部缓冲，输出上界由 `expect_size`（元数据声明的原长）钉死。
  ///
  /// `max_size` 是调用方给出的**硬上限**（来自记录格式/配置常量，不得由待校验的
  /// Blob 元数据推导）：`expect_size > max_size` 时立即 `InvalidConfig`，不分配任何
  /// 输出缓冲。损坏元数据把原长声明成任意大值时，分配被 `max_size` 截住。
  ///
  /// 与 [`Self::mzstd_decompress`] 的区别：解码器窗口按 `expect_size` 收紧、
  /// 输出写定长切片，压缩炸弹无法放大到 `expect_size` 之外。
  /// 返回解压后的切片（长度恰为 `expect_size`）；输出存于独立缓冲，
  /// 不影响 [`Self::as_slice`] 的压缩视图。`expect_size` 须来自可信元数据
  /// （Blob 记录的原长），`max_size` 须是与之独立的可信上限。
  pub fn zstd_decompress_blob(
    &mut self,
    comp: &[u8],
    expect_size: usize,
    max_size: usize,
  ) -> Result<&[u8]> {
    if expect_size > max_size {
      return Err(Error::InvalidConfig(format!(
        "Blob 声明原长 {expect_size} 超出上限 {max_size}"
      )));
    }
    let limits = DecodeLimits::DEFAULT
      .with_max_window_log(window_log_for(expect_size))
      .map_err(|_| Error::InvalidConfig(format!("非法窗口尺寸: {expect_size}")))?;
    let mut dec = Decompressor::new(DecoderConfig::DEFAULT.with_limits(limits));
    if self.decomp.len() < expect_size {
      self.decomp.resize(expect_size, 0);
    }
    let written = dec
      .decompress_to_slice(comp, &mut self.decomp[..expect_size])
      .map_err(Error::from)?;
    if written != expect_size {
      return Err(Error::BlobSizeMismatch {
        expected: expect_size,
        got: written,
      });
    }
    Ok(&self.decomp[..written])
  }
}
