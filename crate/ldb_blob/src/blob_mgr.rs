//! BlobManager（巨型对象，上游 `ZLeanStore/src/storage/blob/blob_manager.cc`）。
//!
//! 只保留 normal buffer pool 路径（C-2 修复：exmap 别名全剔除）；
//! SHA-256 内容寻址去重（look_up 前缀比对）；
//! 尾 extent 移动 / 搬家（move_tail_extent）与全量卸载（unload_all_blobs）。

use std::{
  cmp, fmt,
  mem::size_of,
  rc::Rc,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
};

use gxhash::HashMap;
use ldb_codec::PageCodec;
use ldb_type::{PAGE_SIZE, PageId};
use parking_lot::RwLock;

use crate::{
  error::{Error, Result},
  extent_list::{Extent, ExtentList, TailExtent, extent_size, tier_index},
};

/// Blob 页存取端口（L4 自有端口；Facade 以缓冲池实现 —— extent 页经 VMCache
/// 落盘/加载，缺页/驱逐/脏位统一承载，热路径泛型静态分发）。
///
/// `MANAGED = true` 时页区间由 store 分配（facade 走 `buf.alloc_extent`，
/// 与树页共享分配器）；`false`（常驻内存态，测试/诊断）由管理器内部计数器
/// 分配、store 三个方法皆为 no-op。
pub trait BlobPageStore {
  /// 页区间是否由 store 承担分配
  const MANAGED: bool;

  /// 分配连续 `page_cnt` 页区间，返回起始 pid
  fn alloc_pages(&self, page_cnt: u32) -> impl Future<Output = Result<PageId>>;
  /// 写一个 extent 页
  fn write_page(&self, pid: PageId, page: &[u8; PAGE_SIZE]) -> impl Future<Output = Result<()>>;
  /// 读一个 extent 页
  fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> impl Future<Output = Result<()>>;
}

/// 共享端口：`Rc<T>` 透传（compio 单线程每核下共享同一 store；store 方法全 `&self`）
impl<T: BlobPageStore> BlobPageStore for Rc<T> {
  const MANAGED: bool = T::MANAGED;

  async fn alloc_pages(&self, page_cnt: u32) -> Result<PageId> {
    (**self).alloc_pages(page_cnt).await
  }

  async fn write_page(&self, pid: PageId, page: &[u8; PAGE_SIZE]) -> Result<()> {
    (**self).write_page(pid, page).await
  }

  async fn read_page(&self, pid: PageId, dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
    (**self).read_page(pid, dst).await
  }
}

/// 盘面快照（extent 序 + `(尾起始 pid, 尾缓冲字节)` + 总长）
pub type BlobSnapshot = (Vec<Extent>, (PageId, Vec<u8>), u64);

/// 常驻内存态端口（`BlobManager::new` 默认；读恒走 `resident` 副本）
#[derive(Debug, Default, Clone, Copy)]
pub struct NoBlobStore;

impl BlobPageStore for NoBlobStore {
  const MANAGED: bool = false;

  async fn alloc_pages(&self, _page_cnt: u32) -> Result<PageId> {
    Ok(PageId::MAX)
  }

  async fn write_page(&self, _pid: PageId, _page: &[u8; PAGE_SIZE]) -> Result<()> {
    Ok(())
  }

  async fn read_page(&self, _pid: PageId, _dst: &mut [u8; PAGE_SIZE]) -> Result<()> {
    Ok(())
  }
}

/// Blob 前缀长度（SHA-256 + 长度头；上游 PREFIX_LENGTH=32）
pub const PREFIX_LENGTH: usize = 32;

/// Blob 状态（内容寻址 + extent 链；上游 BlobState）
#[derive(Debug)]
pub struct BlobState {
  /// blob 标识（分配序号）
  pub blob_id: u64,
  /// SHA-256 前缀（内容寻址去重键）—— 上游 `blob_prefix`（= 内容前 32 字节）
  pub blob_prefix: [u8; PREFIX_LENGTH],
  /// SHA-256 全摘要（去重索引真实键；上游 `BlobState.sha2_val`）
  pub sha2_val: [u8; PREFIX_LENGTH],
  /// 总字节
  pub length: u64,
  /// extent 链（整级 + 尾）
  pub extents: ExtentList,
  /// 常驻内存副本（load 后持有；上游 buffer pool 页集的宿主侧简化）
  pub resident: Vec<u8>,
}

impl BlobState {
  /// blob 标识
  #[inline]
  pub fn blob_id(&self) -> u64 {
    self.blob_id
  }

  /// 数据视图
  pub fn data(&self) -> &[u8] {
    &self.resident
  }

  /// 页数（整级 + 尾）
  pub fn page_count(&self) -> u64 {
    self.length.div_ceil(ldb_type::PAGE_SIZE as u64)
  }

  /// 尾 extent 剩余字节
  #[inline]
  pub fn remain_bytes_in_last_extent(&self) -> u64 {
    self.extents.remain_bytes_in_last_extent()
  }

  /// 内存占用（上游 MallocSize）
  pub fn malloc_size(&self) -> u64 {
    size_of::<Self>() as u64 + self.resident.len() as u64 + self.extents.malloc_size()
  }

  /// 计算并写入内容 SHA-256 全摘要（上游 `BlobState::CalculateSHA256(payload)`：
  /// `EVP_DigestFinal_ex` 写入 `sha2_val`）。去重索引与 `blob_state_comparison`
  /// 均以该值为键，不再丢弃。
  pub fn calculate_sha256(&mut self, payload: &[u8]) -> [u8; PREFIX_LENGTH] {
    let digest = PageCodec::sha256_digest(payload);
    self.sha2_val = digest;
    digest
  }

  /// 移入临时存储（上游 MoveToTempStorage：常驻转 extent 链）。
  ///
  /// 前缀语义：上游 `FreshBlobAllocation` 用 `memcpy(blob_prefix, payload, 32)`
  /// 无条件拷贝 32 字节（C-20：payload < 32 时越界读）。此处按
  /// `min(len, PREFIX_LENGTH)` 拷贝 + 补零（C-20 修复）。
  /// 返回本次拆出的整级数据块（调用方独立分配区段后落盘/登记）；若拆级会使
  /// extent 数越过 [`MAX_EXTENTS`] 则返回 [`Error::TooManyExtents`]（尾缓冲不变）。
  pub async fn move_to_temp_storage(&mut self, payload: &[u8]) -> Result<Vec<Vec<u8>>> {
    self.resident = payload.to_vec();
    self.extents.append(payload)
  }
}

/// Blob 管理器。
///
/// `S: BlobPageStore` 为页存取端口（热路径泛型静态分发）；默认
/// [`NoBlobStore`] = 常驻内存态。
pub struct BlobManager<S: BlobPageStore = NoBlobStore> {
  /// 已分配 blob（id → state）
  pub blobs: RwLock<gxhash::HashMap<u64, Arc<RwLock<BlobState>>>>,
  /// 内容指纹索引（sha256 → blob_id；去重）
  dedup: RwLock<gxhash::HashMap<[u8; PREFIX_LENGTH], u64>>,
  /// 下一 blob id
  next_id: AtomicU64,
  /// 下一页号（仅常驻内存态的 extent 分配游标）
  next_pid: AtomicU64,
  /// 尾 extent 容量（页）；亦为每个整级 extent 的分配页数（2 的幂）
  tail_capacity: u64,
  /// 页存取端口
  pub store: S,
}

impl<S: BlobPageStore> fmt::Debug for BlobManager<S> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("BlobManager")
      .field("blobs", &self.blobs.read().len())
      .field("next_id", &self.next_id.load(Ordering::Relaxed))
      .finish_non_exhaustive()
  }
}

impl BlobManager<NoBlobStore> {
  /// 构造（常驻内存态）
  pub fn new(tail_capacity_pages: u64) -> Self {
    Self::with_store(tail_capacity_pages, NoBlobStore)
  }
}

impl<S: BlobPageStore> BlobManager<S> {
  /// 构造并绑定页存取端口（磁盘路径）。
  ///
  /// `tail_capacity_pages` 会向上取整为所属 tier 容量（`2^tier_index`）：尾区
  /// 按该整级尺寸分配，与 `split_to_extents` 弹出的整级 extent 页数严格相等，
  /// 否则非 2 的幂容量会让 extent 越过尾区边界覆盖相邻区段（上游 `AllocExtent`
  /// 亦按 `ExtentSize(idx)` 分配整级）。
  pub fn with_store(tail_capacity_pages: u64, store: S) -> Self {
    let tail_capacity = extent_size(tier_index(tail_capacity_pages.max(1)));
    Self {
      blobs: RwLock::new(HashMap::default()),
      dedup: RwLock::new(HashMap::default()),
      next_id: AtomicU64::new(1),
      next_pid: AtomicU64::new(1),
      tail_capacity,
      store,
    }
  }

  /// 分配一个整级区段（`tail_capacity` 页）：MANAGED 走 store（与树页共享
  /// 分配器），常驻内存态用内部游标。尾区与每个整级 extent 都经此独立分配。
  async fn alloc_region(&self) -> Result<PageId> {
    if S::MANAGED {
      self
        .store
        .alloc_pages(self.tail_capacity as u32)
        .await
        .map_err(|e| Error::StorePage(format!("区段分配失败: {e}")))
    } else {
      Ok(
        self
          .next_pid
          .fetch_add(self.tail_capacity, Ordering::Relaxed),
      )
    }
  }

  /// 把拆级数据块登记为整级 extent：每块独立分配区段、写盘、头插进链。
  ///
  /// 块大小恒为 `tail_capacity * PAGE_SIZE`（`split_to_extents` 保证），
  /// 故 tier = `tier_index(tail_capacity)`。await 点零守卫（先分配/落盘，
  /// 后短持 `blobs` + `st` 写锁登记）。
  async fn grow_extents(&self, id: u64, chunks: Vec<Vec<u8>>) -> Result<()> {
    let tier = tier_index(self.tail_capacity);
    for chunk in chunks {
      let start_pid = self.alloc_region().await?;
      let ext = Extent { start_pid, tier };
      self.persist_extent(&ext, &chunk).await?;
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      st.write().extents.push_extent(ext);
    }
    Ok(())
  }

  /// 持久化拆级 extent（页数据 → 盘面；常驻内存态 no-op）
  async fn persist_extent(&self, ext: &Extent, bytes: &[u8]) -> Result<()> {
    if !S::MANAGED {
      return Ok(());
    }
    let mut page = [0u8; PAGE_SIZE];
    for i in 0..ext.pages() {
      let lo = (i as usize) * PAGE_SIZE;
      let hi = ((i as usize + 1) * PAGE_SIZE).min(bytes.len());
      page[..hi - lo].copy_from_slice(&bytes[lo..hi]);
      page[hi - lo..].fill(0);
      self
        .store
        .write_page(ext.start_pid + i, &page)
        .await
        .map_err(|e| Error::StorePage(format!("extent 页写失败: {e}")))?;
    }
    Ok(())
  }

  /// 持久化尾缓冲的整页部分（快照后写盘 —— await 点零守卫；部分页随
  /// `allocate_remain_content` 补齐后落盘）
  async fn persist_tail(&self, id: u64) -> Result<()> {
    if !S::MANAGED {
      return Ok(());
    }
    let pages: Vec<(PageId, [u8; PAGE_SIZE])> = {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      let t = st.extents.tail.read();
      let full_pages = t.data.len() / PAGE_SIZE;
      (0..full_pages)
        .map(|i| {
          let mut page = [0u8; PAGE_SIZE];
          page.copy_from_slice(&t.data[i * PAGE_SIZE..(i + 1) * PAGE_SIZE]);
          (t.start_pid + i as u64, page)
        })
        .collect()
    };
    for (pid, page) in pages {
      self
        .store
        .write_page(pid, &page)
        .await
        .map_err(|e| Error::StorePage(format!("尾页写失败: {e}")))?;
    }
    Ok(())
  }

  /// 读字节区间（盘面路径：整级 extent 页 + 尾缓冲）。
  /// 入参为快照（extent 序 + 尾缓冲字节），await 读取零守卫存活。
  ///
  /// 读取前按 extent 链的实际页数校验覆盖范围：extent 页数由 `Extent::pages()`
  /// （即 tier 尺寸）决定，是 L4 边界上的不可信元数据；若链未覆盖 `off+len`
  /// 则返回越界错误，杜绝「短链静默补零」的假数据（对齐上游 `lp_size/ExtentSize`
  /// 语义与 ldb_codec 冻结前提「不可信 Blob 必须校验长度」）。
  async fn read_range(
    &self,
    id: u64,
    exts: Vec<Extent>,
    tail: (PageId, Vec<u8>),
    off: u64,
    len: u64,
  ) -> Result<Vec<u8>> {
    let mut out = vec![0u8; len as usize];
    let pg = PAGE_SIZE as u64;
    let mut base = 0u64;
    for ext in &exts {
      let esz = ext.pages() * pg;
      let lo = off.max(base);
      let hi = (off + len).min(base + esz);
      if lo < hi {
        let p_lo = (lo - base) / pg;
        let p_hi = (hi - 1 - base) / pg;
        let mut page = [0u8; PAGE_SIZE];
        for p in p_lo..=p_hi {
          self
            .store
            .read_page(ext.start_pid + p, &mut page)
            .await
            .map_err(|e| Error::StorePage(format!("extent 页读失败: {e}")))?;
          // 页内相交段 → 输出（off 相对输出；base 相对 extent）
          let src = (p * pg) as usize;
          let x_lo = ((lo - base) as usize).max(src);
          let x_hi = ((hi - base) as usize).min(src + PAGE_SIZE);
          let dst_lo = (x_lo as u64 + base - off) as usize;
          let dst_hi = (x_hi as u64 + base - off) as usize;
          out[dst_lo..dst_hi].copy_from_slice(&page[x_lo - src..x_hi - src]);
        }
      }
      base += esz;
    }
    // 尾缓冲段（base 已累加到整级 extent 之后的字节偏移）
    let (t_start, t_data) = tail;
    let t_base = base;
    let covered = t_base + t_data.len() as u64;
    if off + len > covered {
      return Err(Error::BlobOutOfRange(id));
    }
    let lo = off.max(t_base);
    let hi = (off + len).min(covered);
    if lo < hi {
      out[(lo - off) as usize..(hi - off) as usize]
        .copy_from_slice(&t_data[(lo - t_base) as usize..(hi - t_base) as usize]);
    }
    let _ = t_start;
    Ok(out)
  }

  /// 取 blob 盘面快照（短块，零守卫外泄）
  fn snapshot(&self, id: u64) -> Result<BlobSnapshot> {
    let blobs = self.blobs.read();
    let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
    let st = st.read();
    let t = st.extents.tail.read();
    Ok((st.extents.iter(), (t.start_pid, t.data.clone()), st.length))
  }

  /// 新分配（内容寻址登记 + extent 链建立；上游 FreshBlobAllocation）
  ///
  /// `digest` 为调用方算好的 SHA-256 全摘要，直接写入 `sha2_val` 与 `dedup`
  /// 索引 —— 热路径不再重算（大对象 SHA-256 是 GB/s 级开销）。
  /// C-20 修复：前缀拷贝取 min(len, PREFIX_LENGTH)，杜绝小对象越界读
  async fn fresh_blob_allocation_with(
    &self,
    digest: [u8; PREFIX_LENGTH],
    payload: &[u8],
  ) -> Result<u64> {
    // 占位构造：`blob_id`/尾区起始 pid 于分配前回填。此顺序保证**超上限**
    // （`move_to_temp_storage` → `Error::TooManyExtents`）时无任何副作用 ——
    // 不 bump id、不分配区段、不登记 `blobs`/`dedup`（上游超限同样不分配）。
    // 尾区起始 pid 不影响拆级（`split_to_extents` 不触碰 `start_pid`）。
    let mut st = BlobState {
      blob_id: 0,
      blob_prefix: [0; PREFIX_LENGTH],
      sha2_val: digest,
      length: payload.len() as u64,
      extents: ExtentList::new(0, self.tail_capacity),
      resident: Vec::new(),
    };
    let n = payload.len().min(PREFIX_LENGTH);
    st.blob_prefix[..n].copy_from_slice(&payload[..n]);
    let chunks = st.move_to_temp_storage(payload).await?;
    let id = self.next_id.fetch_add(1, Ordering::Relaxed);
    st.blob_id = id;
    st.extents.tail.write().start_pid = self.alloc_region().await?;
    // 先登记再持久化（persist 系经 map 取状态快照）
    self.dedup.write().insert(digest, id);
    self.blobs.write().insert(id, Arc::new(RwLock::new(st)));
    self.grow_extents(id, chunks).await?;
    self.persist_tail(id).await?;
    Ok(id)
  }

  /// 新分配（账本命名 `FreshBlobAllocation` 的公开形态；跳过去重直接建 extent 链）。
  ///
  /// `digest` 必须等于 `PageCodec::sha256_digest(payload)`：它写入 `sha2_val`
  /// 并作为去重索引键，不再在 callee 重算（避免大对象二次 SHA-256）。
  pub async fn fresh_blob_allocation(
    &self,
    digest: [u8; PREFIX_LENGTH],
    payload: &[u8],
  ) -> Result<u64> {
    self.fresh_blob_allocation_with(digest, payload).await
  }

  /// 分配入口（去重命中返回已有 id）
  pub async fn allocate_blob(&self, payload: &[u8]) -> Result<u64> {
    // 摘要一次算透传（fresh 内不再重算 —— 大对象 SHA-256 是 GB/s 级开销）
    let digest = PageCodec::sha256_digest(payload);
    if let Some(&id) = self.dedup.read().get(&digest) {
      return Ok(id);
    }
    self.fresh_blob_allocation_with(digest, payload).await
  }

  /// 内容变更后刷新 `sha2_val` 与去重索引（上游 `AllocateBlob` 在分配/扩展收尾时
  /// 以**最终内容**重算 SHA-256，见 `blob_manager.cc:559-571`）。`old_digest` 为变更
  /// 前的 `sha2_val`（仅当该键仍指向本 id 时摘除，避免误删他人键）。
  ///
  /// 常驻副本完整时直接对 `resident` 求摘要；否则（卸载后扩展）经盘面 extent 链 +
  /// 尾缓冲重建完整内容再求摘要。await 点零守卫。
  async fn refresh_digest(&self, id: u64, old_digest: [u8; PREFIX_LENGTH]) -> Result<()> {
    let resident_complete = {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      st.resident.len() == st.length as usize
    };
    let digest = if resident_complete {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      PageCodec::sha256_digest(&st.resident)
    } else {
      let (exts, tail, length) = self.snapshot(id)?;
      PageCodec::sha256_digest(&self.read_range(id, exts, tail, 0, length).await?)
    };
    {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      st.write().sha2_val = digest;
    }
    let mut d = self.dedup.write();
    if d.get(&old_digest).copied() == Some(id) {
      d.remove(&old_digest);
    }
    d.insert(digest, id);
    Ok(())
  }

  /// 扩展已有 blob（追加数据；上游 ExtendExistingBlob）。
  /// 锁流程：短写追加（持锁）→ 拆级块独立分配/落盘（零守卫）→ 尾页落盘 → 刷新摘要/索引
  pub async fn extend_existing_blob(&self, id: u64, more: &[u8]) -> Result<u64> {
    let (chunks, old_digest) = {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let mut st = st.write();
      let old = st.sha2_val;
      // 先做可失败的上限判定（`append` 在改尾缓冲前返回 `TooManyExtents`），
      // 成功后才更新长度与常驻副本 —— 超限时不留半更新状态
      let chunks = st.extents.append(more)?;
      st.length += more.len() as u64;
      st.resident.extend_from_slice(more);
      (chunks, old)
    };
    self.grow_extents(id, chunks).await?;
    self.persist_tail(id).await?;
    self.refresh_digest(id, old_digest).await?;
    let blobs = self.blobs.read();
    let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
    Ok(st.read().length)
  }

  /// 尾 extent 写新数据（上游 WriteNewDataToLastExtent）。
  /// 锁流程同 [`Self::extend_existing_blob`]：短写追加 → 落盘 → 刷新摘要/索引
  pub async fn write_new_data_to_last_extent(&self, id: u64, data: &[u8]) -> Result<()> {
    let (chunks, old_digest) = {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let mut st = st.write();
      let old = st.sha2_val;
      // 同 `extend_existing_blob`：先判定后改状态
      let chunks = st.extents.append(data)?;
      st.resident.extend_from_slice(data);
      st.length += data.len() as u64;
      (chunks, old)
    };
    self.grow_extents(id, chunks).await?;
    self.persist_tail(id).await?;
    self.refresh_digest(id, old_digest).await?;
    Ok(())
  }

  /// 分配剩余内容（上游 AllocateRemainContent：尾页补齐整页）
  pub async fn allocate_remain_content(&self, id: u64) -> Result<u64> {
    // 读守卫作用域块收敛：pad 在块内算完，await 点零守卫存活
    let pad = {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      let rem = st.remain_bytes_in_last_extent();
      u64::from(rem > 0) * (ldb_type::PAGE_SIZE as u64 - rem)
    };
    if pad > 0 {
      self
        .write_new_data_to_last_extent(id, &vec![0u8; pad as usize])
        .await?;
    }
    Ok(pad)
  }

  /// 加载 blob（常驻副本命中即返回；已卸载（unload / 重开）走盘面 extent 链
  /// 重建 —— C-2 修复：无 exmap 路径，经 BlobPageStore 走 buffer pool 语义）
  pub async fn load_blob(&self, id: u64) -> Result<Vec<u8>> {
    let resident = {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      st.resident.len() == st.length as usize
    };
    if resident {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      return Ok(st.resident.clone());
    }
    let (exts, tail, length) = self.snapshot(id)?;
    self.read_range(id, exts, tail, 0, length).await
  }

  /// 加载内容区间（上游 LoadBlobContent [off, off+len)）
  pub async fn load_blob_content(&self, id: u64, off: u64, len: u64) -> Result<Vec<u8>> {
    let resident = {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      if off + len > st.length {
        return Err(Error::BlobOutOfRange(id));
      }
      st.resident.len() == st.length as usize
    };
    if resident {
      let blobs = self.blobs.read();
      let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
      let st = st.read();
      return Ok(st.resident[off as usize..(off + len) as usize].to_vec());
    }
    let (exts, tail, _) = self.snapshot(id)?;
    self.read_range(id, exts, tail, off, len).await
  }

  /// 内容寻址查找（按指纹串比对；上游 BlobStateComparison 系）
  pub async fn blob_state_comparison(
    &self,
    id: u64,
    other_prefix: &[u8; PREFIX_LENGTH],
  ) -> Result<cmp::Ordering> {
    let blobs = self.blobs.read();
    let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?.read();
    Ok(st.blob_prefix.cmp(other_prefix))
  }

  /// 指纹串比对（与任意前缀字节；上游 BlobStateCompareWithString）
  pub async fn blob_state_compare_with_string(
    &self,
    id: u64,
    prefix: &[u8],
  ) -> Result<cmp::Ordering> {
    let mut fixed = [0u8; PREFIX_LENGTH];
    let n = prefix.len().min(PREFIX_LENGTH);
    fixed[..n].copy_from_slice(&prefix[..n]);
    self.blob_state_comparison(id, &fixed).await
  }

  /// 去重查找（指纹 → id）
  pub async fn look_up_by_digest(&self, digest: &[u8; PREFIX_LENGTH]) -> Option<u64> {
    self.dedup.read().get(digest).copied()
  }

  /// 删除 blob（摘除 blobs 表与去重索引；extent 回收交由空间层）。
  ///
  /// 去重索引键为 SHA-256 全摘要（`sha2_val`），不是内容前缀 `blob_prefix`：
  /// 二者几乎从不相等，用 `blob_prefix` 摘除会漏删，使后续 `allocate_blob`
  /// 命中悬空 id（`load_blob` 随即失败）。仅当该键仍指向本 blob 时摘除。
  pub async fn remove_blob(&self, id: u64) -> Result<bool> {
    let st = self.blobs.write().remove(&id);
    match st {
      Some(s) => {
        let s = s.read();
        let mut d = self.dedup.write();
        if d.get(&s.sha2_val).copied() == Some(id) {
          d.remove(&s.sha2_val);
        }
        Ok(true)
      }
      None => Ok(false),
    }
  }

  /// 尾 extent 移动（上游 MoveTailExtent：尾缓冲搬家至新页区间）
  pub async fn move_tail_extent(&self, id: u64, new_start_pid: PageId) -> Result<()> {
    let blobs = self.blobs.read();
    let st = blobs.get(&id).ok_or(Error::BlobNotFound(id))?;
    let st = st.write();
    let (used, data, capacity) = {
      let t = st.extents.tail.read();
      (t.used_pages, t.data.clone(), t.capacity_pages)
    };
    let mut t = st.extents.tail.write();
    *t = TailExtent::new(new_start_pid, capacity);
    t.append(&data);
    let _ = used;
    Ok(())
  }

  /// 尾 extent 标记驱逐（上游 MarkTailExtentForEviction）
  pub fn mark_tail_extent_for_eviction(&self, id: u64) -> bool {
    let blobs = self.blobs.read();
    blobs.get(&id).is_some()
  }

  /// 全量卸载（上游 UnloadAllBlobs：常驻副本释放，仅留 extent 映射）
  pub async fn unload_all_blobs(&self) -> usize {
    let mut n = 0;
    let blobs = self.blobs.read();
    for st in blobs.values() {
      let mut st = st.write();
      if !st.resident.is_empty() {
        st.resident = Vec::new();
        n += 1;
      }
    }
    n
  }
}
