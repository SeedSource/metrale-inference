// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `METRALE_GLM_DSA_ROW_BATCH=1`: the per-row part of `decode_k` for all `k` rows
//! of a prefill sub-chunk at once. [`DsaRowBatch`] is the staging it needs;
//! `decode_rows_batched` replaces the row loop, and `indexer_rows_batched` replaces `k` calls
//! of `indexer_forward`.
//!
//! What changes against the row loop, and why the bytes do not (detail in
//! `docs/dsa-rowbatch-NOTES.md`):
//! - The loop's host-to-device copies (slot, `q_pos`, block table, `seq_len`, per row; then
//!   the batched selector's `q_pos` array) become two copies per call from one page-locked
//!   staging buffer. The bytes are the ones the loop computes, from the same formulas.
//! - One `glm5next_mla_latent_write_fp8` launch over `k` blocks: the kernel runs one block
//!   per token and reads only `kv_a[token]` and `slot_mapping[token]`.
//! - `wk`, `compress_gate` through `dense_gemv_bf16_batchm` and `weights_proj` (and the
//!   selector's `wq_b`) through `dense_gemv_bf16_fp32out_batchm`, at most
//!   `DENSE_GEMV_BATCHM_MAX_M` rows per launch: each row's result is bit-identical to the M = 1
//!   GEMV the loop runs. `k_norm` in one launch: `nllb_layernorm_bf16` runs one block per row
//!   and `rows` is only its bound.
//! - 2026-10-01: With `METRALE_GLM_DSA_GEMV_SPLIT=1` those batched GEMVs take one launch per
//!   projection over all `k` rows, `ceil(k / 16)` block rows (`dense_gemv_batchm_split`). Block
//!   row `y` offsets `A` and `C` by its first row and then runs the 16-row body on at most
//!   `DENSE_GEMV_BATCHM_MAX_M` rows; a row's arithmetic reads only its own `A` row, the weight
//!   and `K`, never `blockIdx.y` or which rows share its block, so the bytes do not change.
//! - Nothing in the loop reads the indexer cache, the latent cache or the head weights before
//!   the loop ends when the batched selector is on, so writing every row first changes no
//!   read.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `row_batch_ready` returns `Some` only where `batch_select_enabled` holds (prefill, no
//!   graph capture, `k > 1`), with persistent `bt`/`sl`, and with every GEMV kernel whose
//!   per-row identity the batched path relies on resolved.
//! - The staging buffer is rewritten only after the event recorded behind the previous
//!   call's copies has completed.

use std::sync::atomic::{AtomicPtr, Ordering};

use anyhow::{Result, bail, ensure};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::AttnMetadataDev;
use metrale_model_layers::layers::ops::{
    DENSE_GEMV_BATCHM_MAX_M, dense_gemv_batchm, dense_gemv_batchm_fp32out, dense_gemv_batchm_split,
};
use metrale_model_layers::weight_map::DenseWeight;

use super::super::attend::DsaDecodePaging;
use super::super::select::split::RowSplit;
use super::super::state::Glm5NextDsaState;
use super::decode_k::bt_entries_needed;
use super::{Glm5NextDsaLayer, gemm};

/// 2026-10-01: Staging for `decode_rows_batched`, built by `Glm5NextDsaWorkspace::new` only
/// when `glm5next_layer::dsa_row_batch` is on, `bt`/`sl` persist and `max_rows > 1`.
///
/// Host layout (page-locked on CUDA), mirrored on the device by `meta_dev` for the first
/// part: `[slot i64; max_rows] [seq_len i32; max_rows] [q_pos i32; max_rows]`, then
/// `[block table u32; bt_cap]`, which is copied to the workspace's `bt`.
pub(crate) struct DsaRowBatch {
    host: AtomicPtr<u8>,
    host_bytes: usize,
    meta_dev: DevicePtr,
    max_rows: usize,
    /// 2026-10-01: Recorded after each call's copies; the next call waits on it before it
    /// rewrites `host`, because the copies read `host` after they are enqueued.
    event: u64,
    /// 2026-10-01: `dense_gemv_bf16_fp32out_batchm`; `KernelHandle(0)` when the target lacks
    /// it, and `row_batch_ready` then keeps the row loop.
    gemv_batchm_f32: KernelHandle,
}

impl DsaRowBatch {
    pub(super) fn new(gpu: &dyn GpuBackend, max_rows: usize, bt_cap: usize) -> Result<Self> {
        let host_bytes = 16 * max_rows + 4 * bt_cap;
        Ok(Self {
            host: AtomicPtr::new(gpu.alloc_host_pinned(host_bytes)?),
            host_bytes,
            meta_dev: gpu.alloc(16 * max_rows)?,
            max_rows,
            event: gpu.create_event()?,
            gemv_batchm_f32: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_fp32out_batchm",
            ),
        })
    }

    fn small_bytes(&self) -> usize {
        16 * self.max_rows
    }
    fn sl_dev(&self) -> DevicePtr {
        self.meta_dev.offset(8 * self.max_rows)
    }
    /// 2026-10-01: The `q_pos` array `decode_rows_batched` uploaded, for the selector.
    pub(super) fn q_pos_dev(&self) -> DevicePtr {
        self.meta_dev.offset(12 * self.max_rows)
    }
}

/// 2026-10-01: Block rows of a `METRALE_GLM_DSA_GEMV_SPLIT` launch over `rows` rows:
/// `ceil(rows / DENSE_GEMV_BATCHM_MAX_M)`, so each block row takes `ceil(rows / y)`, at most
/// `DENSE_GEMV_BATCHM_MAX_M`, rows (the kernel's `rows_per_y`).
pub(super) fn gemv_split_blocks(rows: usize) -> u32 {
    rows.div_ceil(DENSE_GEMV_BATCHM_MAX_M as usize) as u32
}

/// 2026-10-01: `rows` rows of `C[t] = A[t] @ B^T` in launches of at most
/// `DENSE_GEMV_BATCHM_MAX_M` rows (never the cuBLASLt arm of `gemm`, which sums in another
/// order). `out_elem` is the output element size: 2 runs `dense_gemv_bf16_batchm`, 4
/// `dense_gemv_bf16_fp32out_batchm`; `kernel` must be the matching one.
/// 2026-10-01: Under `METRALE_GLM_DSA_GEMV_SPLIT=1`, one y-split launch over all `rows`
/// instead (`gemv_split_blocks`; both kernels carry the same y-split code, and
/// `dense_gemv_batchm_split` passes `out_stride` through, so it counts FP32 elements for the
/// FP32-out kernel as `dense_gemv_batchm_fp32out` does).
#[allow(clippy::too_many_arguments)]
fn batchm_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    out_elem: usize,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    rows: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    let w = DenseWeight { weight: b };
    if rows > 0 && crate::glm5next_layer::levers::dsa_gemv_split() {
        let (m, y) = (rows as u32, gemv_split_blocks(rows));
        let (n, kk) = (n as u32, kk as u32);
        return dense_gemv_batchm_split(gpu, kernel, a, &w, c, m, y, n, kk, n, stream);
    }
    let max = DENSE_GEMV_BATCHM_MAX_M as usize;
    let mut r0 = 0;
    while r0 < rows {
        let m = max.min(rows - r0) as u32;
        let a_r = a.offset(r0 * kk * 2);
        let c_r = c.offset(r0 * n * out_elem);
        let (n, kk) = (n as u32, kk as u32);
        if out_elem == 4 {
            dense_gemv_batchm_fp32out(gpu, kernel, a_r, &w, c_r, m, n, kk, n, stream)?;
        } else {
            dense_gemv_batchm(gpu, kernel, a_r, &w, c_r, m, n, kk, n, stream)?;
        }
        r0 += max;
    }
    Ok(())
}

impl Glm5NextDsaLayer {
    /// 2026-10-01: The staging, when this `decode_k` call may take `decode_rows_batched`:
    /// `batch_select` (from `batch_select_enabled`), persistent `bt`/`sl`, the M = 1 GEMVs the
    /// row loop runs plus their batched twins all resolved, and `stream` the backend's default
    /// stream. The row loop's blocking copies run on the default stream; the staged copies run
    /// on `stream`, so only then are the two ordered when a later call takes the loop.
    pub(super) fn row_batch_ready(
        &self,
        gpu: &dyn GpuBackend,
        batch_select: bool,
        stream: u64,
    ) -> Option<&DsaRowBatch> {
        let rb = self.workspace.row_batch.as_ref()?;
        let k = &self.kernels;
        let kernels = k.gemv.0 != 0
            && k.gemv_f32.0 != 0
            && k.gemv_batchm.0 != 0
            && rb.gemv_batchm_f32.0 != 0;
        let same_stream = stream == gpu.default_stream();
        let ok = batch_select && self.persist_bt && kernels && same_stream;
        // 2026-10-01: Once per process, so a serve log proves whether the lever engaged on a
        // prefill (it silently keeps the row loop otherwise).
        if batch_select {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!(
                    "GLM DSA prefill row batch (METRALE_GLM_DSA_ROW_BATCH): {} \
                     (persist_bt {}, kernels {}, default stream {})",
                    if ok { "ENGAGED" } else { "NOT engaged, row loop kept" },
                    self.persist_bt,
                    kernels,
                    same_stream
                );
            });
        }
        ok.then_some(rb)
    }

    /// 2026-10-01: `decode_k` from the latent write to the output projection, for all `k`
    /// rows at once; the caller has run the shared projections and checked `k` and the
    /// lockstep. `meta` is what the row loop would read per row (`None` on a prefill pass).
    /// 2026-10-01: `comm` is the layer's communicator; with two ranks and
    /// `METRALE_GLM_DSA_INDEX_SPLIT` the indexer's query side runs for this rank's rows only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_rows_batched(
        &self,
        rb: &DsaRowBatch,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        st: &mut Glm5NextDsaState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &[u32],
        meta: Option<&AttnMetadataDev>,
        t_proj: Option<std::time::Instant>,
        comm: Option<&dyn metrale_comm::CommBackend>,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let split = comm
            .filter(|c| c.world_size() == 2 && crate::glm5next_layer::dsa_index_split())
            .and_then(|c| RowSplit::new(k, c.rank()).map(|s| (s, c)));
        let w = &self.workspace;
        let block_size = kv_cache.config().block_size;
        let bt_block_size = kv_cache.block_size().max(1);
        let small = rb.small_bytes();
        ensure!(
            k <= rb.max_rows,
            "DSA row batch: {k} rows over {}",
            rb.max_rows
        );

        // 2026-10-01: The host values the row loop computes, by the same formulas: row `r`
        // is position `seq_len + r`, slot `bt[pos / bs] * bs + pos % bs`, `seq_len` entry
        // `pos + 1`, and the shared block-table prefix of `bt_entries_needed` entries.
        let bt_used = match meta {
            Some(_) => &block_table[..0],
            None => {
                let needed = bt_entries_needed(seq_len, k, bt_block_size);
                &block_table[..needed.min(block_table.len())]
            }
        };
        if bt_used.len() > w.bt_cap {
            bail!(
                "DSA layer {}: block table needs {} entries for seq_len {} + {} rows but the \
                 persistent buffer holds {}. This is a BLOCK count against a buffer sized by \
                 max_dsa_context (a TOKEN count); do not write past the allocation.",
                self.layer_idx,
                bt_used.len(),
                seq_len,
                k,
                w.bt_cap
            );
        }
        let used = small + 4 * bt_used.len();
        ensure!(
            used <= rb.host_bytes,
            "DSA row batch: staging {used} B over {}",
            rb.host_bytes
        );
        let mut slots = vec![0i64; k];
        if meta.is_none() {
            for (row, s) in slots.iter_mut().enumerate() {
                let pos = seq_len + row;
                let logical = pos / block_size;
                let physical = *block_table.get(logical).ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSA layer {}: block table has {} entries, needs logical block \
                         {logical} for position {pos}",
                        self.layer_idx,
                        block_table.len()
                    )
                })? as usize;
                *s = (physical * block_size + pos % block_size) as i64;
            }
        }

        // 2026-10-01: The previous call's copies read `host` after their enqueue; wait for
        // them before overwriting it.
        gpu.event_synchronize(rb.event)?;
        let base = rb.host.load(Ordering::Relaxed);
        ensure!(!base.is_null(), "DSA row batch: staging buffer is null");
        // SAFETY: `base` is the non-null `alloc_host_pinned(host_bytes)` region from
        // `DsaRowBatch::new`, which nothing frees, zeroed at allocation, so every byte is
        // initialised; `used <= host_bytes` was checked above and `u8` needs no alignment.
        // `decode_k` runs on the scheduler thread only and this is the region's one
        // reference, dropped before return; the device copies that read it were enqueued by
        // an earlier call and have completed (`event_synchronize` above).
        let host: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(base, used) };
        let m = rb.max_rows;
        for (row, slot_value) in slots.iter().enumerate() {
            let pos = seq_len + row;
            let (slot, sl, qp) = (8 * row, 8 * m + 4 * row, 12 * m + 4 * row);
            host[slot..slot + 8].copy_from_slice(&slot_value.to_le_bytes());
            host[sl..sl + 4].copy_from_slice(&((pos + 1) as i32).to_le_bytes());
            host[qp..qp + 4].copy_from_slice(&(pos as i32).to_le_bytes());
        }
        for (i, b) in bt_used.iter().enumerate() {
            host[small + 4 * i..small + 4 * i + 4].copy_from_slice(&b.to_le_bytes());
        }
        gpu.copy_h2d_async_retained(&host[..small], rb.meta_dev, stream)?;
        if !bt_used.is_empty() {
            gpu.copy_h2d_async_retained(&host[small..used], w.bt, stream)?;
        }
        gpu.record_event(rb.event, stream)?;

        // 2026-10-01: One block per token; block `r` reads `kv_a` row `r` and slot `r`.
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([k as u32, 1, 1])
            .block([self.cfg.kv_lora_rank as u32, 1, 1])
            .arg_ptr(w.kv_a)
            .arg_ptr(self.weights.kv_a_layernorm)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(meta.map_or(rb.meta_dev, |m| m.slot))
            .arg_u32(self.cfg.kv_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .arg_f32(1.0 / self.kv_scale)
            .launch(stream)?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);

        let t = profile::start();
        let own = split.map(|(s, _)| s);
        self.indexer_rows_batched(gpu, rb, hidden, k, own, st, stream)?;
        profile::end(profile::DSA_INDEXER, t, gpu, stream);

        let q_pos_host: Vec<i32> = (0..k).map(|r| (seq_len + r) as i32).collect();
        self.select_rows_batched(gpu, k, st, &q_pos_host, Some(rb), split, stream)?;

        let (bt_dev, sl_dev, max_blocks_per_seq) = match meta {
            Some(m) => (m.block_table, m.seq_len, m.max_blocks_per_seq as usize),
            None => (w.bt, rb.sl_dev(), 0),
        };
        let paging = DsaDecodePaging {
            num_seqs: 1,
            num_q_heads: self.cfg.local_heads,
            num_kv_heads: 1,
            max_blocks_per_seq,
            block_size,
            cache_stride_bytes: (block_size * self.cfg.kv_lora_rank) as u64,
        };
        self.attend_rows(gpu, k, st, kv_cache, bt_dev, sl_dev, &paging, true, stream)?;

        let t_proj = profile::start();
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            w.attn_out,
            self.weights.o_absorb,
            hidden,
            k,
            self.cfg.hidden,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        Ok(())
    }

    /// 2026-10-01: `k` calls of `indexer_forward` (host-offset placement, no `pos_dev`) in
    /// one pass: rows `[len, len + k)` of `k_normed` and `gate`, their validity marks, the
    /// selector head weights straight into `head_weights_rows`, then `advance(k)`.
    /// 2026-10-01: Under `split` only this rank's rows of the head weights (the selector
    /// reads no others); the key side is written for all rows on both ranks.
    #[allow(clippy::too_many_arguments)]
    fn indexer_rows_batched(
        &self,
        gpu: &dyn GpuBackend,
        rb: &DsaRowBatch,
        hidden: DevicePtr,
        k: usize,
        split: Option<RowSplit>,
        st: &mut Glm5NextDsaState,
        stream: u64,
    ) -> Result<()> {
        // 2026-10-01: Checked before any write, as `indexer_forward`'s `ensure_room(1)` is;
        // `k` of those pass exactly when this does.
        st.ensure_room(k)?;
        let d = self.cfg.index_head_dim;
        let h = self.cfg.hidden;
        let pos0 = st.len();
        let off = st.row_offset(pos0);
        let bm = self.kernels.gemv_batchm;
        let k_rows = st.k_normed.offset(off);
        let w_wk = self.weights.wk;
        batchm_rows(gpu, bm, 2, hidden, w_wk, k_rows, k, d, h, stream)?;
        // 2026-10-01: `nllb_layernorm_bf16` normalises row `blockIdx.x` of `x` in place and
        // returns for `row >= rows`; every block does the same arithmetic on its own row at
        // any `rows`, so `k` blocks equal `k` one-block launches.
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([k as u32, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(k_rows)
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(k as u32)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        let gate = st.gate.offset(off);
        let w_gate = self.weights.compress_gate;
        batchm_rows(gpu, bm, 2, hidden, w_gate, gate, k, d, h, stream)?;
        let heads = self.cfg.index_heads;
        let (r0, rows) = split.map_or((0, k), |s| (s.r0, s.rows));
        let hw = self.workspace.head_weights_rows.offset(r0 * heads * 4);
        let x = hidden.offset(r0 * h * 2);
        let f32k = rb.gemv_batchm_f32;
        let w_wp = self.weights.weights_proj;
        batchm_rows(gpu, f32k, 4, x, w_wp, hw, rows, heads, h, stream)?;
        gpu.memset_async(st.valid.offset(pos0), 1, k, stream)?;
        st.advance(k)
    }

    /// 2026-10-01: The selector's `wq_b` for `k` rows, `q_resid` row `r` into `q_idx_rows`
    /// row `r`, through the FP32-out batched GEMV: `select_rows_batched` under the row batch,
    /// unless `METRALE_GLM_DSA_BATCH_QIDX` picked cuBLASLt.
    /// 2026-10-01: Rows `[r0, r0 + rows)` only (all `k` rows unless the index split is on).
    pub(super) fn qidx_rows_batched(
        &self,
        gpu: &dyn GpuBackend,
        rb: &DsaRowBatch,
        r0: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        let n = self.cfg.index_heads * self.cfg.index_head_dim;
        let f32k = rb.gemv_batchm_f32;
        let kk = self.cfg.q_lora_rank;
        let (a, out) = (w.q_resid.offset(r0 * kk * 2), w.q_idx_rows.offset(r0 * n * 4));
        batchm_rows(gpu, f32k, 4, a, self.weights.wq_b, out, rows, n, kk, stream)
    }
}
