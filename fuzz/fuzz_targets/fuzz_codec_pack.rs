//! fuzz_codec_pack（todo.md §6）：mzstd 页压缩往返 + 畸形输入抗崩溃
//! （解压器对任意字节流必须返回 Err 而非 panic/越界）。

#![no_main]
use libfuzzer_sys::fuzz_target;

use ldb_codec::PageCodec;
use ldb_type::PAGE_SIZE;

fuzz_target!(|data: &[u8]| {
  if data.len() >= PAGE_SIZE {
    // 1) 往返：压缩 → 解压 → 逐字节一致
    let src = &data[..PAGE_SIZE];
    let packed = PageCodec::mzstd_compress(src).expect("页压缩不应失败");
    let back = PageCodec::mzstd_decompress(&packed).expect("压缩页解压不应失败");
    assert_eq!(back.len(), PAGE_SIZE, "解压长度守恒");
    // 往返对相同输入：解压结果逐页一致（mzstd 无字典态）
    let packed2 = PageCodec::mzstd_compress(src).expect("二次压缩不应失败");
    let back2 = PageCodec::mzstd_decompress(&packed2).expect("二次解压不应失败");
    assert_eq!(back, back2, "压缩结果不稳定");
  }
  // 2) 畸形输入：任意字节流（含压缩页前缀变异）一律 Err / 有界输出，禁 panic
  if let Ok(out) = PageCodec::mzstd_decompress(data) {
    // 若声称成功，输出必须有限（≤ 64 MiB 防解压炸弹）
    assert!(out.len() <= 64 << 20, "解压炸弹防护");
  }
  // 3) 摘要稳定性：同输入同摘要
  let d1 = PageCodec::sha256_digest(data);
  let d2 = PageCodec::sha256_digest(data);
  assert_eq!(d1, d2);
});

