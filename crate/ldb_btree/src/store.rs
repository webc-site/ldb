//! 节点页落盘接线（阶段 1 + 阶段 2）：`ldb_buf` 单页 fix/unfix + `ldb_node` 编解码 + 脏页回写。
//!
//! # 现状（阶段 1 + 阶段 2）
//!
//! 主 `BTree` 仍自持进程内 `nodes: RwLock<HashMap<PageId, BTreeNode>>` 节点表，
//! **尚未**把节点访问改走 `ldb_buf`/VMCache。本模块提供**可落地的最小接线**：
//!
//! - [`BufStore::get`]（阶段 1 读路径）：`BufferManager::fix_share` 取一页快照 →
//!   `validate_bytes` 校验（先校验后 cast，绝不把畸形/未加载页当节点）→ 拷贝为拥有
//!   所有权的 `Box<BTreeNode>` → guard Drop 自动 `unfix_share`；`Restart` 由乐观循环重试。
//! - [`BufStore::modify`]（阶段 2 写路径）：`BufferManager::with_page_mut` 持 EXCLUSIVE
//!   闩拿可变页字节 → `validate_bytes` → `from_page_mut` 等价的字节视图 → 闭包**就地编码**
//!   （`insert_key_value`/`remove`）→ 退出统一标脏 + 解锁。畸形页在进入闭包前即被拒。
//! - [`BufStore::modify_with_gsn`]（阶段 2）：就地改并登记该次修改的 WAL GSN
//!   （`PageHeader.p_gsn`，上游 `transaction.cc:119`），使 WAL-before-page 门可达。
//! - [`BufStore::put`]：整页字节镜像经 `with_page_mut`（EXCLUSIVE 就地覆盖）标脏。
//! - [`BufStore::flush`]：`flush_all`（脏页批量写盘 + `PageBackend::sync`）。
//! - [`BufStore::alloc`]/[`BufStore::dealloc`]：页号分配/释放。
//!
//! # 阶段 3+ 阻塞点（本轮不落地，见审查报告残留风险 / CROSS）
//!
//! 1. **缺页 + 逐节点惰性加载**：`BTree` 需把 `find_leaf_*`/`insert`/`remove` 的
//!    `nodes.read()` 改走 `fix_shared`/`with_page_mut`（阶段 2 的 `BufStore::modify`
//!    已把 EXCLUSIVE 就地写通道铺好，尚待在主树路径接入）。
//! 2. **父指针**：`BTreeAttribute`/`OptimisticParentPointer` 未接线，`find_parent_and_slot`
//!    仍是内存表线性扫描；接 VMCache 后须改为上游 `findParent`（`BTreeGeneric.hpp:134`）。
//! 3. **OLC 重启计数 → `faced_contention`**：`contention_split` 的 `faced_contention`
//!    参数当前由 `insert` 的「本轮是否发生过分裂」近似；真实语义需 `ldb_sync`
//!    乐观 guard 的失败标志（上游 `HybridPageGuard::hasFacedContention`）。
//! 4. **WAL/恢复**：`modify`/`put` 直接脏页回写，未接线 WAL（`TxnLogHook`）；崩溃一致性
//!    由后续 `ldb_wal`/`ldb_recovery` 轮次落地。
//! 5. **contention 重置**：页从磁盘载入可写帧时须 `hdr.contention.reset()`
//!    （`ldb_node` R3 契约），阶段 1 的 `get` 是只读快照不触发。

use std::fmt;

use ldb_buf::{BufferManager, Error as LdbBufError, PageBackend};
use ldb_node::BTreeNode;
use ldb_type::{IndexId, PAGE_SIZE, PageId};

use crate::{error, error::Result};

/// 节点页存储适配器（阶段 1：整页读改写 + 显式 flush）。
///
/// 持有缓冲池引用与比较器；`BufferManager` 的方法全部 `&self`（compio 单线程
/// 每核），故本适配器线程/协程模型与缓冲池一致。
pub struct BufStore<'a, B: PageBackend> {
  mgr: &'a BufferManager<B>,
  cmp: ldb_node::Cmp,
}

impl<'a, B: PageBackend> fmt::Debug for BufStore<'a, B> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BufStore").finish_non_exhaustive()
  }
}

impl<'a, B: PageBackend> BufStore<'a, B> {
  /// 绑定缓冲池（比较器用于页字节校验时的槽位有序性）
  pub fn new(mgr: &'a BufferManager<B>, cmp: ldb_node::Cmp) -> Self {
    Self { mgr, cmp }
  }

  /// 读取一个节点页：共享固定 → 校验/解码 → 拷贝为拥有所有权的节点 → unfix。
  ///
  /// `Restart`（乐观快照期间页被改/驱逐）由循环重试；其余错误上抛。
  pub async fn get(&self, pid: PageId) -> Result<Box<BTreeNode>> {
    // 4096 对齐的整页缓冲：BTreeNode 本身 `align(4096)` 且恰为整页，
    // 直接复用它作为 `fix_share` 的目标页，省一次拷贝。
    let mut node = BTreeNode::new(true, 0);
    let buf: &mut [u8; PAGE_SIZE] = unsafe {
      // SAFETY: BTreeNode 为 4096 B、4096 对齐；`&mut Box<BTreeNode>` 独占，
      // 该引用仅在 fix_share 期间借出，不与其它别名引用共存。
      &mut *(node.as_mut() as *mut BTreeNode as *mut [u8; PAGE_SIZE])
    };
    loop {
      match self.mgr.fix_share(pid, buf).await {
        Ok(_version) => break,
        Err(LdbBufError::Restart) => continue,
        Err(e) => return Err(e.into()),
      }
    }
    // 先校验后使用：非法 bool 字节 / 越界槽位 / 有序性破坏都在此处拦截，
    // 绝不把畸形页字节当作有效 BTreeNode。
    let bytes: &[u8; PAGE_SIZE] = unsafe {
      // SAFETY: 同 buf；只读借用，校验函数不构造 BTreeNode 引用。
      &*(node.as_ref() as *const BTreeNode as *const [u8; PAGE_SIZE])
    };
    BTreeNode::validate_bytes(bytes, self.cmp)
      .map_err(|e| error::Error::CorruptTree(format!("页 {pid} 校验失败: {e}")))?;
    Ok(node)
  }

  /// 写回一个节点页（整页字节镜像覆盖页帧并标脏）。
  ///
  /// 阶段 2：改经 [`BufferManager::with_page_mut`]（EXCLUSIVE 就地视图）覆盖，不再走
  /// 通用 `update_pid` 入口 —— 写路径与就地改路径收敛到同一闩语义。
  pub async fn put(&self, pid: PageId, node: &BTreeNode) -> Result<()> {
    let bytes: &[u8; PAGE_SIZE] = unsafe {
      // SAFETY: BTreeNode 恰为 PAGE_SIZE 字节；只读借用 node，不越界。
      &*(node.ptr() as *const [u8; PAGE_SIZE])
    };
    self
      .mgr
      .with_page_mut(pid, |frame| frame.copy_from_slice(bytes))
      .await?;
    Ok(())
  }

  /// 就地修改一个节点页（EXCLUSIVE 闩内；阶段 2 写路径）。
  ///
  /// 与 [`Self::put`]（整页覆盖）互补：先 `validate_bytes` 把页字节校验为合法
  /// B+Tree 节点，再以 [`BTreeNode::from_page_mut`] 等价的字节视图把 `&mut BTreeNode`
  /// 交给闭包**就地编码**（`insert_key_value`/`remove`/空间账更新等），退出由
  /// `with_page_mut` 统一标脏 + 解锁。无需先重建整页再拷贝。
  ///
  /// 畸形/未初始化页在进入闭包前即被拒（`CorruptTree`），绝不把非法节点交给闭包。
  /// 闭包返回 `Err` 时页可能已被部分修改（就地写不可回滚），调用方须自行保证闭包内
  /// 操作的原子性。
  pub async fn modify<F>(&self, pid: PageId, f: F) -> Result<()>
  where
    F: FnOnce(&mut BTreeNode) -> Result<()>,
  {
    let cmp = self.cmp;
    self
      .mgr
      .with_page_mut(pid, move |frame| -> Result<()> {
        // 先校验后使用：与阶段 1 读路径同一 `validate_bytes` 口径。
        BTreeNode::validate_bytes(frame, cmp)
          .map_err(|e| error::Error::CorruptTree(format!("页 {pid} 校验失败: {e}")))?;
        // SAFETY: `BTreeNode` 为 4096 B、4096 对齐的 `#[repr(C)]` 视图，其头部
        // `BTreeNodeHeader` 以 `ldb_page::PageHeader` 起头 ⇒ 与整页 4096 B 字节布局
        // 逐字节兼容（`BTreeNode::from_page_mut` 内部即 `from_mut(page).cast::<Self>()`）。
        // 闭包期间本协程持 EXCLUSIVE 闩、`&mut *frame` 独占该页帧，无别名引用；字节已通过
        // `validate_bytes`（bool 规范化 / 槽位与 fence 在界内 / 空间账自洽）。
        let node: &mut BTreeNode =
          unsafe { &mut *(frame as *mut [u8; PAGE_SIZE] as *mut BTreeNode) };
        f(node)
      })
      .await?
  }

  /// 就地修改并在页头登记该次修改对应的 WAL GSN（上游 `transaction.cc:119`
  /// `page->p_gsn = logger.GetCurrentGSN() + 1`）。
  ///
  /// 用途：写路径在「页的 WAL 记录已产生**之后**」修改页时记下该 GSN，使
  /// [`BufferManager::fuzzy_checkpoint`] 的 WAL-before-page 门（`page.p_gsn <=
  /// flushed_log_gsn`，对齐上游 `buffer_manager.cc:1550-1552`）在**真实生产写路径**
  /// 可达：`p_gsn` 未落 WAL 的脏页在检查点被推后，水位推进后才写盘。
  pub async fn modify_with_gsn<F>(&self, pid: PageId, gsn: ldb_type::LogId, f: F) -> Result<()>
  where
    F: FnOnce(&mut BTreeNode) -> Result<()>,
  {
    self
      .modify(pid, move |node| {
        node.hdr.page.p_gsn = gsn;
        f(node)
      })
      .await
  }

  /// 分配页号并写入一个初始节点，返回 `(页号, 节点)`。
  pub async fn create(&self, is_leaf: bool, index: IndexId) -> Result<(PageId, Box<BTreeNode>)> {
    let pid = self.mgr.alloc_page().await?;
    let node = BTreeNode::new(is_leaf, index);
    self.put(pid, &node).await?;
    Ok((pid, node))
  }

  /// 分配页号（不初始化内容；调用方随后 `put`）。
  pub async fn alloc(&self) -> Result<PageId> {
    Ok(self.mgr.alloc_page().await?)
  }

  /// 释放页号。
  pub async fn dealloc(&self, pid: PageId) -> Result<()> {
    self.mgr.dealloc_page(pid).await?;
    Ok(())
  }

  /// 脏页批量写盘 + 后端 `sync`。
  pub async fn flush(&self) -> Result<()> {
    self.mgr.flush_all().await?;
    Ok(())
  }
}
