//! 与 C 参考实现 zstd 1.5.7 对拍（需要 PATH 中有 `zstd`）。
//!
//! 默认 `#[ignore]`：避免无 zstd 的环境/CI 失败。手动运行：
//! `cargo nextest run -p ldb_codec --run-ignored only --test zstd_differential`

use std::{
  env,
  fs::{self, File},
  io::{BufReader, Read, Write},
  path::PathBuf,
  process::{self, Command, Stdio},
};

use ldb_codec::PageCodec;
use ldb_pack::UNCOMPRESSED;
use ldb_type::PAGE_SIZE;

/// 确定性可压缩页：三种高冗余模式循环，保证每页都产生真实压缩帧。
fn page(i: usize) -> [u8; PAGE_SIZE] {
  let mut p = [0u8; PAGE_SIZE];
  match i % 3 {
    0 => {
      for (j, b) in p.iter_mut().enumerate() {
        *b = ((j + i) % 7) as u8;
      }
    }
    1 => {
      let phrase = b"the quick brown fox jumps over the lazy dog; 4KiB page. ";
      for (j, b) in p.iter_mut().enumerate() {
        *b = phrase[(j + i) % phrase.len()];
      }
    }
    _ => {
      for (j, b) in p.iter_mut().enumerate() {
        *b = ((j / 16 + i) % 13) as u8;
      }
    }
  }
  p
}

fn zstd_version() -> String {
  let out = Command::new("zstd")
    .arg("--version")
    .output()
    .expect("需要 PATH 中的 zstd CLI");
  assert!(out.status.success(), "zstd --version 失败");
  String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn tmp_path(tag: &str) -> PathBuf {
  env::temp_dir().join(format!("ldb_codec_{tag}_{}.bin", process::id()))
}

/// 正向：本 crate 压缩的 1e5 个页帧拼成合法 zstd 流，C zstd 解出的 409.6 MB
/// 必须逐页等于原页（MISMATCH=0）。
#[test]
#[ignore = "外部 zstd CLI 对拍，手动运行"]
fn c_zstd_decodes_100k_page_frames() {
  const N: usize = 100_000;
  println!("对拍参考: {}", zstd_version());
  let path = tmp_path("fwd");
  let mut file = File::create(&path).unwrap();
  let mut dst = [0u8; PAGE_SIZE];
  for i in 0..N {
    let src = page(i);
    let sz = PageCodec::compress_page(&src, &mut dst).unwrap();
    assert_ne!(sz, UNCOMPRESSED, "第 {i} 页意外回退原文，生成器需更可压缩");
    file.write_all(&dst[..sz as usize]).unwrap();
  }
  file.sync_all().unwrap();
  let compressed_bytes = fs::metadata(&path).unwrap().len();

  let mut child = Command::new("zstd")
    .args(["-dc", path.to_str().unwrap()])
    .stdout(Stdio::piped())
    .spawn()
    .unwrap();
  let mut out = BufReader::new(child.stdout.take().unwrap());
  let mut got = [0u8; PAGE_SIZE];
  let mut mismatch = 0usize;
  for i in 0..N {
    out
      .read_exact(&mut got)
      .unwrap_or_else(|e| panic!("第 {i} 页解压输出不足: {e}"));
    if got != page(i) {
      mismatch += 1;
    }
  }
  let mut tail = [0u8; 1];
  assert_eq!(out.read(&mut tail).unwrap(), 0, "解压输出比 1e5 页更长");
  let status = child.wait().unwrap();
  assert!(status.success(), "zstd -d 退出码 {status}");
  fs::remove_file(&path).unwrap();
  println!(
    "FRAMES={N} COMPRESSED_BYTES={compressed_bytes} RATIO={:.4} MISMATCH={mismatch}",
    compressed_bytes as f64 / (N * PAGE_SIZE) as f64
  );
  assert_eq!(mismatch, 0);
}

/// 反向：C zstd level 1 逐页压缩（`--no-check`），本 crate 必须解出原页。
#[test]
#[ignore = "外部 zstd CLI 对拍，手动运行"]
fn page_codec_decodes_c_zstd_frames() {
  const N: usize = 256;
  println!("对拍参考: {}", zstd_version());
  let path = tmp_path("rev");
  let mut mismatch = 0usize;
  for i in 0..N {
    let src = page(i);
    // 走文件而非 stdin：CLI 才知道输入长度，产生与 ZSTD_compress 相同的单段帧
    fs::write(&path, src).unwrap();
    let out = Command::new("zstd")
      .args(["-1", "--no-check", "-c", path.to_str().unwrap()])
      .output()
      .unwrap();
    assert!(out.status.success(), "第 {i} 页 zstd -1 失败");
    let comp = out.stdout;
    assert!(
      comp.len() < PAGE_SIZE,
      "第 {i} 页参考压缩未缩小: {}",
      comp.len()
    );
    let mut back = [0u8; PAGE_SIZE];
    PageCodec::decompress_page(&comp, comp.len() as u16, &mut back)
      .unwrap_or_else(|e| panic!("第 {i} 页参考帧解压失败: {e}"));
    if back != src {
      mismatch += 1;
    }
  }
  fs::remove_file(&path).unwrap();
  println!("FRAMES={N} MISMATCH={mismatch}");
  assert_eq!(mismatch, 0);
}
