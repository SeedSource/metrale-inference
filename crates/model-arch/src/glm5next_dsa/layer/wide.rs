// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1`: `Glm5NextDsaLayer::decode_k_wide`, one
//! prefill window of `k` rows (the staged prefill's `prefill_rows_ffn()`) with every dense
//! projection as ONE GEMM over all `k` rows, and the selection and gather-attend per
//! `core_rows` (`prefill_rows()`) sub-chunk. [`DsaWideArena`] holds the window-wide
//! activations; one arena serves every DSA layer.
//!
//! Against the sliced path (`decode_k` per sub-chunk, with or without the row batch):
//! - `q_a_proj`, `q_absorb`, `kv_a_proj`, `o_absorb` and the indexer's `wk`, `compress_gate`,
//!   `weights_proj` and `wq_b` run once at M = `k` (cuBLASLt above `DENSE_GEMV_BATCHM_MAX_M`
//!   with `METRALE_GLM_CUBLAS_PROJ` on). NOT byte-identical: cuBLASLt picks its algorithm per
//!   shape, and the indexer projections leave the per-row GEMV, so each output element is
//!   summed in another order. Quality is gated at model level.
//! - The latent write covers all `k` rows in one launch, and the indexer keys and gates of all
//!   `k` rows are written into the cache before the first selection. Neither is read early:
//!   sub-chunk `j` advances the indexer length and marks validity only through its own rows,
//!   so its geometry (pool count) is the sliced path's; `dsa_index_scores` takes a pool for
//!   row `r` only when it ends at or before `q_pos[r]`; and the attend reads only the latent
//!   slots the selection names, all at or before the row (`batch_select_enabled` notes).
//! - Per sub-chunk the selection and attend get the same arguments as `decode_rows_batched`,
//!   with the per-row inputs (`q_idx`, head weights, `q_abs`, `q_pos`, `seq_len`) read at the
//!   sub-chunk's offset into the arena.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `decode_k_wide` falls back to `decode_k` per `core_rows` sub-chunk (the sliced path,
//!   unchanged) unless `wide_ready` holds: arena attached and wide enough, persistent
//!   `bt`/`sl`, the batched-selector buffers, `core_rows <= max_rows`, eager prefill with at
//!   most one metadata row, and `stream` the default stream (the host copies below are
//!   blocking copies, ordered only against the default stream, as in the row loop).
//! - The lockstep check, `ensure_room(k)` and the block-table checks run before the first
//!   launch; the indexer cache ends at `seq_len + k`, as after `k` rows of `decode_k`.
//! - 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1`: `ensure_room(k)` bounds the whole window by
//!   the ring; ring writes are `wide_ring.rs`; each sub-chunk compresses only new pools.

use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::{ForwardContext, LayerState};
use metrale_model_layers::layers::ops;

use super::super::Glm5NextDsaConfig;
use super::super::attend::{DsaDecodeInputs, DsaDecodePaging, attention};
use super::super::select::split::{row_split_for, select_tokens_split};
use super::super::select::{DsaSelectInputs, DsaSelectLaunch::Exact, select_tokens};
use super::super::state::Glm5NextDsaState;
use super::decode_k::bt_entries_needed;
use super::{Glm5NextDsaLayer, gemm};
use crate::glm5next_layer::scratch_union::ScratchAlloc;

/// 2026-10-01: Window-wide DSA activations for `decode_k_wide`, `max_rows` rows each. The
/// loader allocates one at `prefill_rows_ffn()` rows when `METRALE_GLM_PREFILL_FULLWIDTH_GEMM`
/// is on and attaches it to every DSA layer's workspace (`with_wide`); the layers run one
/// after another on one stream, so they share it.
///
/// Bytes per row ([`DsaWideArena::bytes_per_row`]): `q_a` and `q_resid` `2 * q_lora_rank`
/// each, `q_abs` and `attn_out` `2 * local_heads * kv_lora_rank` each, `kv_a`
/// `2 * kv_lora_rank`, `q_idx` `4 * index_heads * index_head_dim`, `head_weights`
/// `4 * index_heads`, and 16 of metadata. GLM-5.3 at TP2 (q_lora 1536, 32 local heads x
/// kv_lora 512, 32 x 128 indexer heads): 89,232 B per row, 365.5 MB at 4096 rows, 731 MB at
/// 8192.
pub struct DsaWideArena {
    max_rows: usize,
    /// 2026-10-01: `[max_rows, q_lora_rank]` BF16, `q_a_proj(hidden)` before its RMSNorm.
    q_a: DevicePtr,
    /// 2026-10-01: `[max_rows, q_lora_rank]` BF16, after `q_a_layernorm`.
    q_resid: DevicePtr,
    /// 2026-10-01: `[max_rows, local_heads * kv_lora_rank]` BF16, the absorbed queries.
    q_abs: DevicePtr,
    /// 2026-10-01: `[max_rows, kv_lora_rank]` BF16, `kv_a_proj(hidden)` before the latent write.
    kv_a: DevicePtr,
    /// 2026-10-01: `[max_rows, index_heads * index_head_dim]` FP32, the selector queries.
    q_idx: DevicePtr,
    /// 2026-10-01: `[max_rows, index_heads]` FP32, the selector head weights.
    head_weights: DevicePtr,
    /// 2026-10-01: `[max_rows, local_heads * kv_lora_rank]` BF16, the gather-attend output.
    attn_out: DevicePtr,
    /// 2026-10-01: `[k] i64` KV slots, then `[k] i32` `seq_len` entries, then `[k] i32` query
    /// positions, packed at the call's `k` (16 bytes per row).
    meta: DevicePtr,
}

impl DsaWideArena {
    /// 2026-10-01: Device bytes per row (see the type's doc).
    pub fn bytes_per_row(cfg: &Glm5NextDsaConfig) -> usize {
        let lat = cfg.local_heads * cfg.kv_lora_rank;
        2 * 2 * cfg.q_lora_rank
            + 2 * 2 * lat
            + 2 * cfg.kv_lora_rank
            + 4 * cfg.index_heads * cfg.index_head_dim
            + 4 * cfg.index_heads
            + 16
    }

    /// 2026-10-01: Allocate an arena for windows of up to `max_rows` (at least 1) rows.
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig, max_rows: usize) -> Result<Self> {
        Self::new_in(cfg, max_rows, &mut |b| gpu.alloc(b))
    }

    /// 2026-10-05: [`Self::new`] taking each buffer from `a`, same order and sizes.
    pub fn new_in(cfg: &Glm5NextDsaConfig, max_rows: usize, a: &mut ScratchAlloc) -> Result<Self> {
        let r = max_rows.max(1);
        let lat = cfg.local_heads * cfg.kv_lora_rank;
        Ok(Self {
            max_rows: r,
            q_a: a(r * cfg.q_lora_rank * 2)?,
            q_resid: a(r * cfg.q_lora_rank * 2)?,
            q_abs: a(r * lat * 2)?,
            kv_a: a(r * cfg.kv_lora_rank * 2)?,
            q_idx: a(r * cfg.index_heads * cfg.index_head_dim * 4)?,
            head_weights: a(r * cfg.index_heads * 4)?,
            attn_out: a(r * lat * 2)?,
            meta: a(r * 16)?,
        })
    }

    /// 2026-10-01: The widest window this arena serves.
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
}

impl Glm5NextDsaLayer {
    /// 2026-10-01: The arena, when `decode_k_wide` may take the full-width body for a `k`-row
    /// window at `core` rows per selection (see the module invariants); `None` keeps the
    /// sliced path. Logs once per process which one ran.
    fn wide_ready(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        core: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Option<&Arc<DsaWideArena>> {
        let w = &self.workspace;
        let arena = w.wide.as_ref().filter(|a| k <= a.max_rows);
        let eager = !ctx.graph_capture && !ctx.decode_step;
        let one_seq = ctx.attn_metadata.as_ref().is_none_or(|m| m.num_seqs <= 1);
        let same_stream = stream == gpu.default_stream();
        let selector = w.q_mask_rows.0 != 0 && w.q_idx_rows.0 != 0;
        let ok = arena.is_some()
            && self.persist_bt
            && selector
            && core <= w.max_rows
            && eager
            && one_seq
            && same_stream;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::warn!(
                "GLM DSA full-width prefill (METRALE_GLM_PREFILL_FULLWIDTH_GEMM): {} (arena {}, \
                 persist_bt {}, batched selector {}, core {core} <= {}, eager {eager}, one \
                 sequence {one_seq}, default stream {same_stream})",
                if ok {
                    "ENGAGED"
                } else {
                    "NOT engaged, per-sub-chunk decode_k kept"
                },
                arena.is_some(),
                self.persist_bt,
                selector,
                w.max_rows
            );
        });
        if ok { arena } else { None }
    }

    /// 2026-10-01: `C[M, N] = A[M, K] @ B[N, K]^T`, BF16 in, FP32 out: cuBLASLt above
    /// `DENSE_GEMV_BATCHM_MAX_M` with `cublas_wide_proj` on (as `gemm` does for BF16 out),
    /// otherwise the FP32-out GEMV (M = 1) or tile GEMM. `gemm` itself writes BF16 on its
    /// cuBLASLt arm, so the FP32-out projections cannot go through it at M > 16.
    #[allow(clippy::too_many_arguments)]
    fn gemm_f32_rows(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        kk: usize,
        stream: u64,
    ) -> Result<()> {
        if m > ops::DENSE_GEMV_BATCHM_MAX_M as usize && crate::glm5next_layer::cublas_wide_proj() {
            return ops::cublas_bf16_proj_dense_f32_out(
                a, b, c, m as u32, n as u32, kk as u32, stream,
            );
        }
        let kernels = ops::DenseMmKernels {
            gemm: self.kernels.gemm_f32,
            gemv: self.kernels.gemv_f32,
            // 2026-10-01: No FP32-out arm in `dense_mm_bf16`'s batched GEMV slot.
            batchm: KernelHandle(0),
        };
        ops::dense_mm_bf16(gpu, &kernels, a, b, c, m, n, kk, stream)
    }

    /// 2026-10-01: One prefill window of `k` rows from position `seq_len`, writing the output
    /// projection over `hidden` like `decode_k`: the projections once over all `k` rows, the
    /// selection and attend per `core_rows` rows (module notes). Falls back to `decode_k` over
    /// each `core_rows` sub-chunk when `wide_ready` does not hold, which is the sliced path.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_k_wide(
        &self,
        hidden: DevicePtr,
        k: usize,
        core_rows: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let gpu = ctx.gpu;
        let core = core_rows.max(1);
        let Some(arena) = self.wide_ready(gpu, k, core, ctx, stream) else {
            for (t, n) in crate::glm5next_layer::sub_chunks(k, core) {
                self.decode_k(
                    hidden.offset(t * self.cfg.hidden * 2),
                    n,
                    state,
                    kv_cache,
                    seq_len + t,
                    block_table,
                    ctx,
                    stream,
                    // 2026-10-01: Only the staged prefill reaches `decode_k_wide`.
                    true,
                )?;
            }
            return Ok(());
        };
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        self.check_lockstep(st, seq_len)?;
        if k == 0 {
            bail!("DSA layer {}: a 0-row full-width window", self.layer_idx);
        }
        // 2026-10-01: Checked before any write: the indexer rows of the whole window are
        // written below before the first `advance`.
        st.ensure_room(k)?;

        let c = &self.cfg;
        let w = &self.workspace;
        let (h, ql, kvl) = (c.hidden, c.q_lora_rank, c.kv_lora_rank);
        let lat = c.local_heads * kvl;
        let (heads, d) = (c.index_heads, c.index_head_dim);
        let idx_row = heads * d;
        let block_size = kv_cache.config().block_size;
        let bt_block_size = kv_cache.block_size().max(1);

        // 2026-10-01: The host values the row loop computes, by the same formulas, for all `k`
        // rows: slot `bt[pos / bs] * bs + pos % bs`, `seq_len` entry `pos + 1`, query position
        // `pos`, and the shared block-table prefix of `bt_entries_needed(seq_len, k)` entries
        // (a superset of each sub-chunk's prefix; the gather indexes it by token).
        let needed = bt_entries_needed(seq_len, k, bt_block_size);
        let bt_used = &block_table[..needed.min(block_table.len())];
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
        let mut meta: Vec<u8> = Vec::with_capacity(16 * k);
        for row in 0..k {
            let pos = seq_len + row;
            let logical = pos / block_size;
            let physical = *block_table.get(logical).ok_or_else(|| {
                anyhow::anyhow!(
                    "DSA layer {}: block table has {} entries, needs logical block {logical} \
                     for position {pos}",
                    self.layer_idx,
                    block_table.len()
                )
            })? as usize;
            let slot = (physical * block_size + pos % block_size) as i64;
            meta.extend_from_slice(&slot.to_le_bytes());
        }
        for row in 0..k {
            meta.extend_from_slice(&((seq_len + row + 1) as i32).to_le_bytes());
        }
        for row in 0..k {
            meta.extend_from_slice(&((seq_len + row) as i32).to_le_bytes());
        }
        // 2026-10-01: Blocking copies on the default stream (`wide_ready` requires `stream` to
        // be it), so they land after every launch already enqueued, including the previous
        // layer's reads of this shared arena.
        gpu.copy_h2d(&meta, arena.meta)?;
        if !bt_used.is_empty() {
            let bt: Vec<u8> = bt_used.iter().flat_map(|b| b.to_le_bytes()).collect();
            gpu.copy_h2d(&bt, w.bt)?;
        }
        let slot_dev = arena.meta;
        let sl_dev = arena.meta.offset(8 * k);
        let q_pos_dev = arena.meta.offset(12 * k);

        // 2026-10-01: The shared projections, the q chain and the latent write, all `k` rows.
        let t_proj = profile::start();
        let (g, gv, bm) = (self.kernels.gemm, self.kernels.gemv, self.kernels.gemv_batchm);
        // 2026-10-06: q_a and kv_a both read `hidden`; kv_a is issued right after q_a (see
        // `decode_k`), so under `METRALE_GLM_DENSE_FP8_W8A8_SHARE_QUANT` their W8A8 activation
        // quant runs once. Nothing writes `hidden` until `o_absorb`.
        let share = crate::glm5next_layer::dense_fp8::w8a8_share_input(hidden, k * h * 2);
        gemm(
            gpu,
            g,
            gv,
            bm,
            hidden,
            self.weights.q_a_proj,
            arena.q_a,
            k,
            ql,
            h,
            stream,
        )?;
        gemm(
            gpu,
            g,
            gv,
            bm,
            hidden,
            self.weights.kv_a_proj,
            arena.kv_a,
            k,
            kvl,
            h,
            stream,
        )?;
        drop(share);
        KernelLaunch::new(gpu, self.kernels.rms_norm)
            // 2026-10-01: One block per row, as in `decode_k`.
            .grid([k as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(arena.q_a)
            .arg_ptr(self.weights.q_a_layernorm)
            .arg_ptr(arena.q_resid)
            .arg_u32(ql as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        gemm(
            gpu,
            g,
            gv,
            bm,
            arena.q_resid,
            self.weights.q_absorb,
            arena.q_abs,
            k,
            lat,
            ql,
            stream,
        )?;
        // 2026-10-01: One block per token; block `r` reads `kv_a` row `r` and slot `r`.
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([k as u32, 1, 1])
            .block([kvl as u32, 1, 1])
            .arg_ptr(arena.kv_a)
            .arg_ptr(self.weights.kv_a_layernorm)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(slot_dev)
            .arg_u32(kvl as u32)
            .arg_f32(self.rms_eps)
            .arg_f32(1.0 / self.kv_scale)
            .launch(stream)?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);

        // 2026-10-01: The indexer projections, all `k` rows: keys and gates straight into
        // cache rows `[len, len + k)` (validity and length follow per sub-chunk below), the
        // head weights and the selector queries into the arena.
        let t = profile::start();
        // 2026-10-06: `METRALE_GLM_DSA_POOL_CACHE=1`: ring slots, staged through `arena.q_idx`
        // when the ring wraps inside the window (`wide_ring.rs`). Lever off: the cache rows.
        let win = self.pool_window_begin(gpu, st, arena.q_idx, k, stream)?;
        let (k_rows, gate_rows) = (win.k_rows, win.gate_rows);
        let w_wk = self.weights.wk;
        gemm(gpu, g, gv, bm, hidden, w_wk, k_rows, k, d, h, stream)?;
        // 2026-10-01: `nllb_layernorm_bf16` normalises row `blockIdx.x` in place, as in
        // `indexer_rows_batched`.
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
        let w_gate = self.weights.compress_gate;
        gemm(gpu, g, gv, bm, hidden, w_gate, gate_rows, k, d, h, stream)?;
        win.scatter(gpu, st, d, stream)?;
        let w_wp = self.weights.weights_proj;
        self.gemm_f32_rows(gpu, hidden, w_wp, arena.head_weights, k, heads, h, stream)?;
        let (w_qb, qr, qi) = (self.weights.wq_b, arena.q_resid, arena.q_idx);
        self.gemm_f32_rows(gpu, qr, w_qb, qi, k, idx_row, ql, stream)?;
        profile::end(profile::DSA_INDEXER, t, gpu, stream);

        // 2026-10-01: Per sub-chunk: its rows become valid and the length reaches its end, so
        // the selection plans at the sliced path's length; then the selection and the attend
        // over its rows, inputs at its offset into the arena.
        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        for (t0, n) in crate::glm5next_layer::sub_chunks(k, core) {
            gpu.memset_async(st.valid.offset(st.len()), 1, n, stream)?;
            st.advance(n)?;
            let geom = st.geometry(c, n)?;
            let inputs = DsaSelectInputs {
                k_normed: st.k_normed,
                gate: st.gate,
                valid: st.valid,
                ape: self.weights.ape,
                q: arena.q_idx.offset(t0 * idx_row * 4),
                weights: arena.head_weights.offset(t0 * heads * 4),
                q_pos: q_pos_dev.offset(t0 * 4),
                // 2026-10-01: All 1s for `max_rows >= core >= n` rows.
                q_mask: w.q_mask_rows,
                first_key: 0,
                // 2026-10-01: Host geometry, as `select_rows_batched` passes.
                geom_dev: DevicePtr::NULL,
                // 2026-10-06: Pool cache on: only the pools completed since the last sub-chunk.
                pool_cache: st.pool_select_args()?,
            };
            let t = profile::start();
            // 2026-10-01: Rows `0..n` of the workspace's `[max_rows, out_width]` selection,
            // which the attend below reads before the next sub-chunk overwrites it.
            // 2026-10-04: `METRALE_GLM_DSA_INDEX_SPLIT_WIDE=1` on two ranks: each rank selects
            // half the rows and the ranks swap token rows, the same bytes in rows `0..n`
            // (`select_tokens_split`, the `METRALE_GLM_DSA_INDEX_SPLIT` mechanism).
            let (sk, sel) = (&self.select_kernels, &w.select);
            let on = crate::glm5next_layer::dsa_index_split_wide();
            match row_split_for(ctx.comm, on, n) {
                Some((s, comm)) => {
                    select_tokens_split(gpu, sk, c, &geom, &inputs, sel, s, comm, stream)?
                }
                None => select_tokens(gpu, sk, c, &geom, &inputs, sel, Exact, stream)?,
            }
            st.note_selected();
            profile::end(profile::DSA_SELECT, t, gpu, stream);

            // 2026-10-01: `attend_rows`' launch at offset inputs: all sub-chunk rows share the
            // uploaded block table (row stride 0); row `r` reads its own `seq_len` entry.
            let paging = DsaDecodePaging {
                num_seqs: n,
                num_q_heads: c.local_heads,
                num_kv_heads: 1,
                max_blocks_per_seq: 0,
                block_size,
                cache_stride_bytes: (block_size * kvl) as u64,
            };
            let t = profile::start();
            // 2026-10-03: Through `attention` (prefill rows) so METRALE_GLM_MLA_PREFILL_TC=1 also
            // engages here; before, this arm always took the decode kernel (race #68).
            attention(
                gpu,
                self.decode_kernel,
                c,
                &geom,
                &paging,
                &DsaDecodeInputs {
                    q: arena.q_abs.offset(t0 * lat * 2),
                    k_cache: pool,
                    // 2026-10-01: Absorbed NoPE MLA: K and V are the same latent.
                    v_cache: pool,
                    out: arena.attn_out.offset(t0 * lat * 2),
                    block_tables: w.bt,
                    seq_lens: sl_dev.offset(t0 * 4),
                    sel_indices: w.select.tokens(),
                    k_scale: self.kv_scale,
                    v_scale: self.kv_scale,
                },
                true,
                stream,
            )?;
            profile::end(profile::DSA_ATTEND, t, gpu, stream);
        }

        // 2026-10-01: The output projection, all `k` rows, over the input buffer.
        let t_proj = profile::start();
        gemm(
            gpu,
            g,
            gv,
            bm,
            arena.attn_out,
            self.weights.o_absorb,
            hidden,
            k,
            h,
            lat,
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        Ok(())
    }
}
