// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `PagedKvCache` methods: pool allocation, the block free list
//! and reference counts, block pointers and strides, host block copies, and
//! BF16 fingerprint diagnostics. The struct is defined in `kv_cache.rs`.
//!
//! Owner: cache.
//! Invariants: see the parent module.

use anyhow::{Result, bail};

use super::block_trace::BlockTrace;
use super::{KvCacheConfig, KvCacheDtype, LayerPool, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

mod diagnostics;

impl PagedKvCache {
    /// 2026-09-25: Allocate a K pool and a V pool of `num_blocks` blocks for
    /// every attention layer. All blocks start free with count 0.
    pub fn new(config: KvCacheConfig, num_blocks: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut total_bytes: usize = 0;
        for i in 0..config.num_layers {
            // 2026-09-25: K and V are sized separately: for `Bf16KTurbo3V` the
            // K pool is BF16-sized and the V pool Turbo3-sized.
            // 2026-10-01: Under `v_aliases_k` the V pool is the K pool with
            // the K stride; nothing is allocated for it.
            let k_block_bytes = config.k_block_bytes_for_layer(i);
            let v_block_bytes = if config.v_aliases_k {
                k_block_bytes
            } else {
                config.v_block_bytes_for_layer(i)
            };
            let k_pool_bytes = num_blocks * k_block_bytes;
            let k_pool = gpu.alloc(k_pool_bytes)?;
            let v_pool = if config.v_aliases_k {
                k_pool
            } else {
                let v_pool_bytes = num_blocks * v_block_bytes;
                total_bytes += v_pool_bytes;
                gpu.alloc(v_pool_bytes)?
            };
            total_bytes += k_pool_bytes;
            layers.push(LayerPool {
                k_pool,
                v_pool,
                k_block_stride: k_block_bytes,
                v_block_stride: v_block_bytes,
                dtype: config.dtype_for_layer(i),
                v_aliased: config.v_aliases_k,
            });
        }

        let free_blocks: Vec<u32> = (0..num_blocks as u32).rev().collect();
        let block_ref_counts = vec![0u32; num_blocks];

        let has_mixed = !config.layer_dtypes.is_empty()
            && config.layer_dtypes.iter().any(|d| *d != config.dtype);
        if has_mixed {
            let hp_count = config
                .layer_dtypes
                .iter()
                .filter(|d| **d != config.dtype)
                .count();
            tracing::info!(
                "KV cache: {} blocks × {} layers ({} high-precision) = {:.1} GB total (mixed dtype)",
                num_blocks,
                config.num_layers,
                hp_count,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        } else if config.v_aliases_k {
            tracing::info!(
                "KV cache: {} blocks × {} layers × {} bytes/block = {:.1} GB total \
                 (V aliased onto K: no V pool)",
                num_blocks,
                config.num_layers,
                config.block_bytes(),
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        } else {
            tracing::info!(
                "KV cache: {} blocks × {} layers × {} bytes/block = {:.1} GB total",
                num_blocks,
                config.num_layers,
                config.block_bytes_kv(),
                (num_blocks * config.num_layers * config.block_bytes_kv()) as f64
                    / (1024.0 * 1024.0 * 1024.0),
            );
        }

        Ok(Self {
            layers,
            num_blocks,
            free_blocks,
            block_ref_counts,
            config,
            trace: BlockTrace::new(num_blocks),
        })
    }

    /// 2026-09-25: Pop a free block and set its count to 1. Errors when the
    /// free list is empty.
    #[track_caller]
    pub fn alloc_block(&mut self) -> Result<u32> {
        let idx = self
            .free_blocks
            .pop()
            .ok_or_else(|| anyhow::anyhow!("KV cache exhausted: no free blocks"))?;
        self.block_ref_counts[idx as usize] = 1;
        if self.trace.is_on() {
            self.trace
                .record(idx as usize, "alloc", 1, std::panic::Location::caller());
        }
        Ok(idx)
    }

    /// 2026-09-25: Zero the block's K and V bytes in every layer, enqueued on
    /// `stream`.
    pub fn zero_block(
        &self,
        block_idx: u32,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        for layer in &self.layers {
            let k_offset = block_idx as usize * layer.k_block_stride;
            let v_offset = block_idx as usize * layer.v_block_stride;
            gpu.memset_async(
                layer.k_pool.offset(k_offset),
                0,
                layer.k_block_stride,
                stream,
            )?;
            if !layer.v_aliased {
                gpu.memset_async(
                    layer.v_pool.offset(v_offset),
                    0,
                    layer.v_block_stride,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: Diagnostic twin of `zero_block` that fills with 0xFF.
    /// `0xFFFF` is a BF16 NaN and `0xFF` an FP8 E4M3 NaN, so attention over a
    /// slot nothing wrote yields NaN instead of zeros. The model engine
    /// calls it instead of `zero_block` under `METRALE_KV_POISON`.
    pub fn poison_block(
        &self,
        block_idx: u32,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        for layer in &self.layers {
            let k_offset = block_idx as usize * layer.k_block_stride;
            let v_offset = block_idx as usize * layer.v_block_stride;
            gpu.memset_async(
                layer.k_pool.offset(k_offset),
                0xFF,
                layer.k_block_stride,
                stream,
            )?;
            if !layer.v_aliased {
                gpu.memset_async(
                    layer.v_pool.offset(v_offset),
                    0xFF,
                    layer.v_block_stride,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: `alloc_block`, returning `None` when the free list is empty.
    #[track_caller]
    pub fn try_alloc_block(&mut self) -> Option<u32> {
        let idx = self.free_blocks.pop()?;
        self.block_ref_counts[idx as usize] = 1;
        if self.trace.is_on() {
            self.trace
                .record(idx as usize, "try_alloc", 1, std::panic::Location::caller());
        }
        Some(idx)
    }

    /// 2026-09-25: Add one reference to a block. There is no check that the
    /// block is allocated.
    #[track_caller]
    pub fn inc_ref(&mut self, block_idx: u32) {
        debug_assert!((block_idx as usize) < self.num_blocks);
        self.block_ref_counts[block_idx as usize] += 1;
        if self.trace.is_on() {
            let after = self.block_ref_counts[block_idx as usize];
            self.trace.record(
                block_idx as usize,
                "inc",
                after,
                std::panic::Location::caller(),
            );
        }
    }

    /// 2026-09-25: Drop one reference. Returns true if the count reached 0 and
    /// the block went back on the free list. At 0 refs a debug build panics;
    /// a release build logs an error and returns false.
    #[track_caller]
    pub fn dec_ref(&mut self, block_idx: u32) -> bool {
        let idx = block_idx as usize;
        debug_assert!(idx < self.num_blocks);
        // 2026-09-25: Refused rather than decremented: the `debug_assert` is
        // compiled out in release, and the workspace sets no `overflow-checks`,
        // so the subtraction would wrap to u32::MAX and the block would never
        // return to the free list.
        debug_assert!(
            self.block_ref_counts[idx] > 0,
            "dec_ref on block with 0 refs"
        );
        if self.block_ref_counts[idx] == 0 {
            let caller = std::panic::Location::caller();
            tracing::error!(
                "dec_ref on block {block_idx} with 0 refs (from {caller}) — refcount bug \
                 (ignoring; would otherwise wrap to u32::MAX and pin the block){}",
                if self.trace.is_on() {
                    format!("\n  history: {}", self.trace.dump(idx))
                } else {
                    String::from(" [set METRALE_KV_TRACE=1 for this block's ref history]")
                }
            );
            return false;
        }
        self.block_ref_counts[idx] -= 1;
        if self.trace.is_on() {
            let after = self.block_ref_counts[idx];
            self.trace
                .record(idx, "dec", after, std::panic::Location::caller());
        }
        if self.block_ref_counts[idx] == 0 {
            self.free_blocks.push(block_idx);
            true
        } else {
            false
        }
    }

    /// 2026-09-25: `dec_ref`, discarding the result.
    #[track_caller]
    pub fn free_block(&mut self, block_idx: u32) {
        self.dec_ref(block_idx);
    }

    /// 2026-09-25: `free_block` on every entry of `block_table`.
    #[track_caller]
    pub fn free_blocks(&mut self, block_table: &[u32]) {
        for &idx in block_table {
            self.free_block(idx);
        }
    }

    /// 2026-09-25: Release the prefix cache's own reference on a block its
    /// eviction returned: one decrement, and a push onto the free list only on
    /// a 1 to 0 transition. At 0 refs it logs a warning and does nothing.
    #[track_caller]
    pub fn return_evicted_block(&mut self, block_idx: u32) {
        let idx = block_idx as usize;
        debug_assert!(idx < self.num_blocks);
        // 2026-09-25: One decrement, not a reset to 0: a sequence can still
        // hold the block in its block table, and freeing it would hand a live
        // block to the next `alloc_block`. The cache's reference is the one
        // `cache_acquires_refs` (model-engine `block_mgmt.rs`) took on insert.
        //
        // At 0 refs the block is already on the free list; a second push would
        // let two `alloc_block` calls hand the same block to two sequences.
        if self.block_ref_counts[idx] == 0 {
            tracing::warn!(
                "return_evicted_block({block_idx}) with 0 refs (from {}) — the prefix cache \
                 returned a block it holds no reference on; ignoring (re-pushing it would \
                 duplicate a free-list entry and alias the block across sequences){}",
                std::panic::Location::caller(),
                if self.trace.is_on() {
                    format!("\n  history: {}", self.trace.dump(idx))
                } else {
                    String::new()
                }
            );
            return;
        }
        self.block_ref_counts[idx] -= 1;
        if self.trace.is_on() {
            let after = self.block_ref_counts[idx];
            self.trace
                .record(idx, "evict_return", after, std::panic::Location::caller());
        }
        if self.block_ref_counts[idx] == 0 {
            self.free_blocks.push(idx as u32);
        }
    }

    /// 2026-09-25: The block's reference count.
    pub fn ref_count(&self, block_idx: u32) -> u32 {
        self.block_ref_counts[block_idx as usize]
    }

    /// 2026-09-25: Length of the free list.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// 2026-09-25: Device address of a block in a layer's K pool.
    pub fn k_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        let layer = &self.layers[layer_idx];
        layer
            .k_pool
            .offset(block_idx as usize * layer.k_block_stride)
    }

    /// 2026-09-25: Device address of a block in a layer's V pool.
    pub fn v_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        let layer = &self.layers[layer_idx];
        layer
            .v_pool
            .offset(block_idx as usize * layer.v_block_stride)
    }

    /// 2026-09-25: Base address of a layer's K pool.
    pub fn k_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].k_pool
    }

    /// 2026-09-25: Base address of a layer's V pool.
    pub fn v_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].v_pool
    }

    /// 2026-09-25: `KvCacheConfig::cache_stride_elements`: elements per block
    /// and side from the global dims, not from `layer_dims`.
    pub fn cache_stride(&self) -> usize {
        self.config.cache_stride_elements()
    }

    /// 2026-09-25: `KvCacheConfig::block_bytes`: uniform dtype, global dims.
    pub fn block_stride_bytes(&self) -> usize {
        self.config.block_bytes()
    }

    /// 2026-09-25: The K-side block stride of attention layer `layer_idx`,
    /// which is also the V-side stride for a symmetric format.
    pub fn block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].k_block_stride
    }

    /// 2026-09-25: The K-side block stride of attention layer `layer_idx`.
    pub fn k_block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].k_block_stride
    }

    /// 2026-09-25: The V-side block stride of attention layer `layer_idx`.
    pub fn v_block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].v_block_stride
    }

    /// 2026-09-25: `KvCacheConfig::nvfp4_data_bytes`.
    pub fn nvfp4_data_bytes(&self) -> usize {
        self.config.nvfp4_data_bytes()
    }

    /// 2026-09-25: `KvCacheConfig::turbo4_data_bytes`.
    pub fn turbo4_data_bytes(&self) -> usize {
        self.config.turbo4_data_bytes()
    }

    /// 2026-09-25: `KvCacheConfig::turbo3_data_bytes`.
    pub fn turbo3_data_bytes(&self) -> usize {
        self.config.turbo3_data_bytes()
    }

    /// 2026-09-25: `KvCacheConfig::turbo2_data_bytes`.
    pub fn turbo2_data_bytes(&self) -> usize {
        self.config.turbo2_data_bytes()
    }

    /// 2026-09-25: `KvCacheConfig::turbo8_data_bytes`.
    pub fn turbo8_data_bytes(&self) -> usize {
        self.config.turbo8_data_bytes()
    }

    /// 2026-09-25: `KvCacheConfig::turbo4_scale_bytes`.
    pub fn turbo4_scale_bytes(&self) -> usize {
        self.config.turbo4_scale_bytes()
    }

    /// 2026-09-25: The configuration the cache was built with.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// 2026-09-25: The format of attention layer `layer_idx`.
    pub fn dtype_for_layer(&self, layer_idx: usize) -> KvCacheDtype {
        self.layers[layer_idx].dtype
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn dtype(&self) -> KvCacheDtype {
        self.config.dtype
    }

    /// 2026-09-25: `KvCacheConfig::num_layers`.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// 2026-09-25: Copy one block of one layer to the host, as `(k, v)`, each
    /// the length of its side's block stride.
    pub fn read_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let k_stride = self.layers[layer_idx].k_block_stride;
        let v_stride = self.layers[layer_idx].v_block_stride;
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);

        let mut k_data = vec![0u8; k_stride];
        let mut v_data = vec![0u8; v_stride];
        gpu.copy_d2h(k_ptr, &mut k_data)?;
        gpu.copy_d2h(v_ptr, &mut v_data)?;

        Ok((k_data, v_data))
    }

    /// 2026-09-25: Copy `k_data` and `v_data` from the host to the start of one
    /// block of one layer.
    pub fn write_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        k_data: &[u8],
        v_data: &[u8],
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);
        gpu.copy_h2d(k_data, k_ptr)?;
        // 2026-10-01: An aliased V is the K block just written.
        if !self.layers[layer_idx].v_aliased {
            gpu.copy_h2d(v_data, v_ptr)?;
        }
        Ok(())
    }

    /// 2026-09-25: How many block indices fit in `available_bytes`, at
    /// `block_bytes_kv_all_layers` each. Errors if that is zero.
    pub fn compute_num_blocks(config: &KvCacheConfig, available_bytes: usize) -> Result<usize> {
        let bytes_per_block = config.block_bytes_kv_all_layers();
        if bytes_per_block == 0 {
            bail!("KV cache block size is zero");
        }
        Ok(available_bytes / bytes_per_block)
    }
}
