//! MemBackend 稀疏/打孔/溢出语义回归（纯内存，无文件系统依赖）。

use std::io;

use ldb_io::MemBackend;
use ldb_type::PAGE_SIZE;

#[compio::test]
async fn mem_backend_sparse_and_punch_clip() {
  let m = MemBackend::new();

  // 从未写过的 pid 读零（稀疏洞），不是 EOF
  let mut out = [0xFFu8; PAGE_SIZE];
  m.read_at_page(5, &mut out).await.unwrap();
  assert!(out.iter().all(|&x| x == 0));

  // 任意长度写（WAL 扇区路径）：长度按实际写入扩展
  m.write_at_page(0, &[9u8; 512]).await.unwrap();
  assert_eq!(m.file_len().await.unwrap(), 512);
  let mut page = [0u8; PAGE_SIZE];
  m.read_at_page(0, &mut page).await.unwrap();
  assert!(page[..512].iter().all(|&x| x == 9));
  assert!(page[512..].iter().all(|&x| x == 0));

  // 整页写后，打孔区间越过 EOF：现有重叠部分必须清零，长度保持
  m.write_at_page(0, &[9u8; PAGE_SIZE]).await.unwrap();
  assert_eq!(m.file_len().await.unwrap(), PAGE_SIZE as u64);
  m.punch_hole((PAGE_SIZE - 16) as u64, 116).await.unwrap();
  let mut out = [1u8; PAGE_SIZE];
  m.read_at_page(0, &mut out).await.unwrap();
  assert!(out[..PAGE_SIZE - 16].iter().all(|&x| x == 9));
  assert!(out[PAGE_SIZE - 16..].iter().all(|&x| x == 0));
  assert_eq!(m.file_len().await.unwrap(), PAGE_SIZE as u64);

  // 溢出拒绝而非 panic
  let err = m.write_at_page(u64::MAX, &[0u8; 8]).await.unwrap_err();
  assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[compio::test]
async fn mem_backend_punch_zero_len_keeps_data() {
  let m = MemBackend::new();
  m.write_at_page(0, &[5u8; PAGE_SIZE]).await.unwrap();
  m.punch_hole(100, 0).await.unwrap();
  let mut out = [0u8; PAGE_SIZE];
  m.read_at_page(0, &mut out).await.unwrap();
  assert!(out.iter().all(|&x| x == 5));
}

/// 页号与字节偏移必须是两套接口：pid=1 指 4096，不是 512（防 8× 错位复发）。
#[compio::test]
async fn mem_backend_page_and_bytes_offsets_do_not_alias() {
  let m = MemBackend::new();
  // 字节偏移 512 写 32 B：文件长度 544，而不是 512*4096
  m.write_bytes(512, &[7u8; 32]).await.unwrap();
  assert_eq!(m.file_len().await.unwrap(), 544);

  // 页号 1 落在 4096；不能覆盖 512 处的数据，也不能被 512 处的写入影响
  m.write_at_page(1, &[9u8; PAGE_SIZE]).await.unwrap();
  assert_eq!(m.file_len().await.unwrap(), 2 * PAGE_SIZE as u64);

  let mut page0 = [0u8; PAGE_SIZE];
  m.read_at_page(0, &mut page0).await.unwrap();
  assert!(page0[..512].iter().all(|&x| x == 0));
  assert!(page0[512..544].iter().all(|&x| x == 7));
  assert!(page0[544..].iter().all(|&x| x == 0));

  // 字节接口读 512 处 = 7；读 4096 处 = 9
  let mut probe = [0u8; 32];
  m.read_bytes(512, &mut probe).await.unwrap();
  assert!(probe.iter().all(|&x| x == 7));
  let mut page1 = [0u8; 64];
  m.read_bytes(PAGE_SIZE as u64, &mut page1).await.unwrap();
  assert!(page1.iter().all(|&x| x == 9));

  // 页接口读 pid=1 也必须落在 4096（防 read_at_page 把页号当字节偏移）
  let mut whole = [0u8; PAGE_SIZE];
  m.read_at_page(1, &mut whole).await.unwrap();
  assert!(whole.iter().all(|&x| x == 9));
}
