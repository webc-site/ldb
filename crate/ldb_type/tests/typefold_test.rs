//! TypeFold 字节布局与边界钉死（todo.md §4.1 防线 1）。
//!
//! 本文件只做「具体字节 / 边界值」断言：覆盖上游 `typefold.cc` 全部 10 个函数，
//! 含 `i32::MIN/MAX`、`u32::MAX`、`i64::MIN/MAX`、`u64::MAX`、空 `Varchar`、
//! `length == Leng` 越界写边界、`v.len() + 1` 超出 u16 域边界，以及定宽折叠在
//! 目标缓冲不足时的错误路径（与 `unfold_*` 的截断拒绝对称）。
//!
//! 「性质 / 采样 / 变异」类用例（保序同构、精确逆、位映射、组合键游标、采样全域
//! 保序扫描）一律在 `typefold_property_test.rs`，两文件职责不重叠。

use ldb_type::{Error, typefold::*};

/// 上游 `Varchar<Leng>` 的两个实例化长度（typefold.cc:70-73）
const LENG_16: usize = 16;
const LENG_128: usize = 128;

#[test]
fn fold_unsigned_and_wide_roundtrip() {
  let mut b4 = [0u8; 4];
  fold_uinteger(&mut b4, u32::MAX).unwrap();
  assert_eq!(b4, [0xFF; 4]);
  assert_eq!(unfold_uinteger(&b4).unwrap(), u32::MAX);

  let mut b8 = [0u8; 8];
  // i64::MIN 符号位翻转后全 0，i64::MAX 全 0xFF（字节序 == 数值序的两个端点）
  fold_timestamp(&mut b8, i64::MIN).unwrap();
  assert_eq!(b8, [0x00; 8]);
  assert_eq!(unfold_timestamp(&b8).unwrap(), i64::MIN);
  fold_timestamp(&mut b8, i64::MAX).unwrap();
  assert_eq!(b8, [0xFF; 8]);
  assert_eq!(unfold_timestamp(&b8).unwrap(), i64::MAX);

  fold_unique_id(&mut b8, 0x0123_4567_89AB_CDEF).unwrap();
  assert_eq!(b8, [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]);
  assert_eq!(unfold_unique_id(&b8).unwrap(), 0x0123_4567_89AB_CDEF);
  // u64::MAX 无符号全 1（不做符号位翻转）
  fold_unique_id(&mut b8, u64::MAX).unwrap();
  assert_eq!(b8, [0xFF; 8]);
  assert_eq!(unfold_unique_id(&b8).unwrap(), u64::MAX);
}

/// Varchar：空串 / 常规串 / `length == Leng` 容量边界（Leng ∈ {16,128}）/ 错误路径。
///
/// `x.length == Leng` 边界：上游 `Fold` 写 `writer[x.length] = 0`，
/// 需 `Leng + 1` 容量，否则越界 1 字节（todo.md §1.0 ③ / §9）。
#[test]
fn fold_varchar_boundaries_nul_terminate_and_roundtrip() {
  // 空 Varchar（上游越界缺陷场景，边界必须安全）
  let mut buf = [0u8; 8];
  assert_eq!(fold_varchar(&mut buf, b"").unwrap(), 1);
  assert_eq!(buf[0], 0);

  let n = fold_varchar(&mut buf, b"abc").unwrap();
  assert_eq!(n, 4);
  assert_eq!(&buf[..4], b"abc\0");
  assert_eq!(unfold_varchar(&buf).unwrap(), b"abc");

  // 恰够 Leng+1：成功且 NUL 落在最后一字节；只有 Leng 字节：必须报错且不越界写
  for leng in [LENG_16, LENG_128] {
    let src = vec![b'k'; leng];
    let mut exact = vec![0xFFu8; leng + 1];
    assert_eq!(fold_varchar(&mut exact, &src).unwrap(), (leng + 1) as u16);
    assert_eq!(exact[leng], 0);
    assert_eq!(&exact[..leng], src.as_slice());
    assert_eq!(unfold_varchar(&exact).unwrap(), src.as_slice());

    let mut short = vec![0xFFu8; leng];
    let e = fold_varchar(&mut short, &src).unwrap_err();
    assert!(
      matches!(e, Error::BufferTooSmall(n) if n == leng + 1),
      "{e:?}"
    );
    assert!(short.iter().all(|&b| b == 0xFF));
  }

  // 缓冲不足返回错误而非越界写
  let mut tiny = [0u8; 2];
  let e = fold_varchar(&mut tiny, b"abc").unwrap_err();
  assert!(matches!(e, Error::BufferTooSmall(4)), "{e:?}");

  // 未终止（无 NUL）的输入：解折叠取整个切片，不越界
  assert_eq!(unfold_varchar(b"abcdef").unwrap(), b"abcdef");
  // 解折叠缓冲不足
  let e = unfold_integer(&[0u8; 3]).unwrap_err();
  assert!(matches!(e, Error::BufferTooSmall(4)), "{e:?}");
  let e = unfold_unique_id(&[0u8; 7]).unwrap_err();
  assert!(matches!(e, Error::BufferTooSmall(8)), "{e:?}");
}

/// `v.len() + 1` 超出 u16 域：上游把 `x.length + 1` 截断为 `uint16_t` 返回，
/// 会让调用方游标与实际写入字节数不一致（静默损坏）；本实现必须返回错误。
#[test]
fn fold_varchar_rejects_length_beyond_u16() {
  let over = vec![b'k'; usize::from(u16::MAX)];
  let mut big = vec![0xFFu8; over.len() + 1];
  let e = fold_varchar(&mut big, &over).unwrap_err();
  assert!(
    matches!(e, Error::BufferTooSmall(n) if n == over.len() + 1),
    "{e:?}"
  );
  // 未被写坏
  assert!(big.iter().all(|&b| b == 0xFF));

  // 域内最大值（v.len() == u16::MAX - 1 ⇒ n == u16::MAX）仍必须成功
  let max_in = vec![b'k'; usize::from(u16::MAX) - 1];
  assert_eq!(fold_varchar(&mut big, &max_in).unwrap(), u16::MAX);
  assert_eq!(big[usize::from(u16::MAX) - 1], 0);
  assert_eq!(unfold_varchar(&big).unwrap(), max_in.as_slice());
}

/// 定宽折叠的目标缓冲不足：四个 `fold_*` 必须报 `BufferTooSmall(所需字节数)` 且
/// 不写坏缓冲 —— 与 `unfold_*` 的截断拒绝（`fold_varchar_boundaries_nul_terminate_and_roundtrip`
/// 中的 `unfold_integer(&[0u8; 3])`）对称，补齐错误路径矩阵。
///
/// `put32` / `put64` 是四个定宽折叠共用的写入器，其容量守卫此前只有「容量足够」的
/// 正向用例覆盖，反向（容量不足 + 载荷字节数正确）此前无用例。
#[test]
fn fold_fixed_width_rejects_undersized_writer() {
  // 3 < 4：i32 / u32 折叠必须报 4
  let mut b4 = [0xFFu8; 3];
  assert!(matches!(
    fold_integer(&mut b4, 1).unwrap_err(),
    Error::BufferTooSmall(4)
  ));
  assert!(matches!(
    fold_uinteger(&mut b4, 1).unwrap_err(),
    Error::BufferTooSmall(4)
  ));
  assert_eq!(b4, [0xFF; 3], "i32/u32 短缓冲被写坏");

  // 7 < 8：i64 / u64 折叠必须报 8
  let mut b8 = [0xFFu8; 7];
  assert!(matches!(
    fold_timestamp(&mut b8, 1).unwrap_err(),
    Error::BufferTooSmall(8)
  ));
  assert!(matches!(
    fold_unique_id(&mut b8, 1).unwrap_err(),
    Error::BufferTooSmall(8)
  ));
  assert_eq!(b8, [0xFF; 7], "i64/u64 短缓冲被写坏");

  // 空缓冲：所需字节数仍须如实上报
  let mut empty: [u8; 0] = [];
  assert!(matches!(
    fold_integer(&mut empty, 0).unwrap_err(),
    Error::BufferTooSmall(4)
  ));
  assert!(matches!(
    fold_timestamp(&mut empty, 0).unwrap_err(),
    Error::BufferTooSmall(8)
  ));
}

/// 字节布局钉死：大端 + 符号位翻转（todo.md:351 要求断言布局，不止序关系）。
#[test]
fn fold_byte_layout_matches_big_endian_with_sign_flip() {
  let mut b4 = [0u8; 4];

  // i32 = bswap32(x ^ (1<<31))：零值首字节必为 0x80（只翻符号位，不做整体取反）
  fold_integer(&mut b4, 0).unwrap();
  assert_eq!(b4, [0x80, 0x00, 0x00, 0x00]);
  fold_integer(&mut b4, i32::MIN).unwrap();
  assert_eq!(b4, [0x00; 4]);
  fold_integer(&mut b4, i32::MAX).unwrap();
  assert_eq!(b4, [0xFF; 4]);
  // -1 的补码 0xFFFF_FFFF 翻符号位得 0x7FFF_FFFF（大端）
  fold_integer(&mut b4, -1).unwrap();
  assert_eq!(b4, [0x7F, 0xFF, 0xFF, 0xFF]);

  // u32 = bswap32(x)：纯大端，不翻符号位
  fold_uinteger(&mut b4, 0).unwrap();
  assert_eq!(b4, [0x00; 4]);
  fold_uinteger(&mut b4, 0x0102_0304).unwrap();
  assert_eq!(b4, [0x01, 0x02, 0x03, 0x04]);

  let mut b8 = [0u8; 8];
  fold_timestamp(&mut b8, 0).unwrap();
  assert_eq!(b8, [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
  fold_unique_id(&mut b8, 0).unwrap();
  assert_eq!(b8, [0x00; 8]);
}
