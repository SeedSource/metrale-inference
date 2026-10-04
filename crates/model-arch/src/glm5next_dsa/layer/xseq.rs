// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `METRALE_GLM_DSA_XSEQ_BATCH=1`: `Glm5NextDsaLayer::decode_xseq`, the DSA half
//! of a multi-sequence decode (`steps/multi_seq.rs`, every `ks[i] = 1`) or batched verify
//! (`steps/verify_multi.rs`, `ks[i]` draft rows) with the dense projections run ONCE over the
//! rows of all sequences instead of once per sequence. [`DsaXseqArena`] holds those rows; one
//! arena serves every DSA layer.
//!
//! Against the per-sequence loop (`decode_k` per sequence, which re-reads every projection
//! weight once per sequence and the indexer weights once per row):
//! - `q_a_proj`, `q_absorb`, `kv_a_proj` and `o_absorb` run through the same `gemm` wrapper
//!   at M = the group's row count instead of M = `ks[i]`. A group holds at most
//!   `DENSE_GEMV_BATCHM_MAX_M` (16) rows, so every call stays on a GEMV kernel whose rows carry
//!   the one-row GEMV's bits: `dense_gemv_bf16_batchm` against `dense_gemv_bf16` (same kv order,
//!   add order and reduction tree, `--fmad=false`; kernel header), and under
//!   `METRALE_GLM_DENSE_FP8` `dense_gemv_fp8w_batchm` against `dense_gemv_fp8w`
//!   (`glm5next_dense_fp8_microtest`). Wider batches are cut into groups of whole sequences.
//! - The indexer's `wk`, `compress_gate` (BF16 out), `weights_proj` and `wq_b` (FP32 out) run
//!   through the batched GEMVs `decode_rows_batched` uses (`row_batch::batchm_rows`), row for
//!   row the bits of the M = 1 GEMVs `indexer_forward` and `select_row_at` run
//!   (`dsa_rowbatch_bitparity_microtest`); `k_norm` runs one block per row. Each row's key and
//!   gate are then copied into its sequence's indexer cache at the row the loop would write.
//! - `rms_norm_vanilla` runs one block per row, so one launch over the group equals one per
//!   sequence.
//! - Per sequence, in the loop's order: latent write, indexer row, selection, gather-attend
//!   (`decode_k_rows`, the loop's own code), reading the sequence's rows of the arena. No step
//!   reads another sequence's rows, and every input a step reads is complete before it runs
//!   (stream order), so the bytes do not change.
//! - The output projection writes each row over its input row of `hidden`, as the loop does;
//!   here every input row is read before any is written.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `decode_xseq` returns `Ok(false)`, launching nothing, unless `xseq_ready` holds: arena
//!   attached, not capturing a graph (the indexer rows are placed at host offsets), the GEMV
//!   kernels the identity argument relies on resolved, and every `ks[i]` in
//!   `1..=min(workspace rows, arena rows)`. The caller then runs its per-sequence loop.
//! - Per group, the lockstep check, `ensure_room(ks[i])` and the metadata check of every
//!   sequence run before the group's first launch.

use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState};
use metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

use super::super::Glm5NextDsaConfig;
use super::super::state::Glm5NextDsaState;
use super::Glm5NextDsaLayer;

mod group;

/// 2026-10-03: `METRALE_GLM_DSA_XSEQ_BATCH=1`: in the multi-sequence decode
/// (`glm5next_layer::decode_multi_seq`) and the batched verify
/// (`glm5next_layer::batched_verify`), a DSA layer runs its `q_a_proj`, `q_absorb`, `kv_a_proj`
/// and `o_absorb` projections and its indexer projections (`wk`, `compress_gate`,
/// `weights_proj`, `wq_b`) ONCE over the rows of all sequences, at most
/// `DENSE_GEMV_BATCHM_MAX_M` rows per group, instead of once per sequence (per row for the
/// indexer); the latent and indexer writes, the selection and the attend stay per sequence
/// (module notes). Each row keeps its bits: the batched GEMVs give every row the bits of the
/// one-row GEMV (BF16, and FP8 under `METRALE_GLM_DENSE_FP8`). Eager only. Off unless set to
/// `1`; read once; off, the loader attaches no arena and the per-sequence `decode_k` loop runs.
pub fn dsa_xseq_batch() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DSA_XSEQ_BATCH").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_XSEQ_BATCH=1 - GLM DSA projections run once over all sequences \
                 of a multi-sequence decode / batched verify (<= 16 rows per group)"
            );
        }
        on
    })
}

/// 2026-10-03: The cross-sequence rows of `decode_xseq`, `max_rows` (at most
/// `DENSE_GEMV_BATCHM_MAX_M`) rows each. The loader allocates one when
/// `METRALE_GLM_DSA_XSEQ_BATCH` is on and attaches it to every DSA layer's workspace
/// (`with_xseq`); the layers run one after another on one stream, so they share it.
///
/// Bytes per row ([`DsaXseqArena::bytes_per_row`]): `q_a`, `q_resid` `2 * q_lora_rank` each,
/// `q_abs`, `attn_out` `2 * local_heads * kv_lora_rank` each, `kv_a` `2 * kv_lora_rank`,
/// `k_normed`, `gate` `2 * index_head_dim` each, `head_weights` `4 * index_heads`, `q_idx`
/// `4 * index_heads * index_head_dim`. GLM-5.3 at TP2: 89,728 B per row, 1.4 MB at 16 rows.
pub struct DsaXseqArena {
    max_rows: usize,
    q_a: DevicePtr,
    q_resid: DevicePtr,
    q_abs: DevicePtr,
    kv_a: DevicePtr,
    attn_out: DevicePtr,
    k_normed: DevicePtr,
    gate: DevicePtr,
    head_weights: DevicePtr,
    q_idx: DevicePtr,
    /// 2026-10-03: `dense_gemv_bf16_fp32out_batchm`; `KernelHandle(0)` when the target lacks
    /// it, and `xseq_ready` then keeps the per-sequence loop.
    gemv_batchm_f32: KernelHandle,
}

impl DsaXseqArena {
    /// 2026-10-03: Device bytes per row (see the type's doc).
    pub fn bytes_per_row(cfg: &Glm5NextDsaConfig) -> usize {
        let lat = cfg.local_heads * cfg.kv_lora_rank;
        2 * 2 * cfg.q_lora_rank
            + 2 * 2 * lat
            + 2 * cfg.kv_lora_rank
            + 2 * 2 * cfg.index_head_dim
            + 4 * cfg.index_heads
            + 4 * cfg.index_heads * cfg.index_head_dim
    }

    /// 2026-10-03: An arena for groups of up to `max_rows` rows, clamped to
    /// `1..=DENSE_GEMV_BATCHM_MAX_M` (wider groups would leave the row-exact GEMVs).
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig, max_rows: usize) -> Result<Self> {
        let r = max_rows.clamp(1, DENSE_GEMV_BATCHM_MAX_M as usize);
        let lat = cfg.local_heads * cfg.kv_lora_rank;
        let d = cfg.index_head_dim;
        Ok(Self {
            max_rows: r,
            q_a: gpu.alloc(r * cfg.q_lora_rank * 2)?,
            q_resid: gpu.alloc(r * cfg.q_lora_rank * 2)?,
            q_abs: gpu.alloc(r * lat * 2)?,
            kv_a: gpu.alloc(r * cfg.kv_lora_rank * 2)?,
            attn_out: gpu.alloc(r * lat * 2)?,
            k_normed: gpu.alloc(r * d * 2)?,
            gate: gpu.alloc(r * d * 2)?,
            head_weights: gpu.alloc(r * cfg.index_heads * 4)?,
            q_idx: gpu.alloc(r * cfg.index_heads * d * 4)?,
            gemv_batchm_f32: metrale_model_layers::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_fp32out_batchm",
            ),
        })
    }

    /// 2026-10-03: The most rows one group takes.
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
}

/// 2026-10-03: Consecutive groups of whole sequences, each at most `cap` rows: `(first, end,
/// rows)` with sequences `first..end`. Every `ks[i]` must be in `1..=cap`.
pub(super) fn xseq_groups(ks: &[usize], cap: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i0 = 0;
    while i0 < ks.len() {
        let (mut i1, mut rows) = (i0, 0);
        while i1 < ks.len() && rows + ks[i1] <= cap {
            rows += ks[i1];
            i1 += 1;
        }
        if i1 == i0 {
            // A `ks[i]` above `cap`: `xseq_ready` refuses those, so this is unreachable; one
            // sequence per group keeps the loop finite.
            i1 = i0 + 1;
            rows = ks[i0];
        }
        out.push((i0, i1, rows));
        i0 = i1;
    }
    out
}

fn dsa_state(state: &mut dyn LayerState) -> Result<&mut Glm5NextDsaState> {
    state
        .as_any_mut()
        .downcast_mut::<Glm5NextDsaState>()
        .ok_or_else(|| anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState"))
}

impl Glm5NextDsaLayer {
    /// 2026-10-03: Whether the cross-sequence arena is attached (`METRALE_GLM_DSA_XSEQ_BATCH`).
    pub fn xseq_attached(&self) -> bool {
        self.workspace.xseq.is_some()
    }

    /// 2026-10-03: The arena, when `decode_xseq` may batch `ks` (module invariants); `None`
    /// keeps the per-sequence loop. Logs once per process which one ran.
    fn xseq_ready(&self, ks: &[usize], ctx: &ForwardContext) -> Option<&Arc<DsaXseqArena>> {
        let w = &self.workspace;
        let arena = w.xseq.as_ref()?;
        let k = &self.kernels;
        let kernels = k.gemv.0 != 0
            && k.gemv_f32.0 != 0
            && k.gemv_batchm.0 != 0
            && arena.gemv_batchm_f32.0 != 0
            && self.select_kernels.k_norm.0 != 0;
        let eager = !ctx.graph_capture;
        let cap = w.max_rows.min(arena.max_rows);
        let rows_ok = !ks.is_empty() && ks.iter().all(|&k| (1..=cap).contains(&k));
        let ok = kernels && eager && rows_ok;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::warn!(
                "GLM DSA cross-sequence projections (METRALE_GLM_DSA_XSEQ_BATCH): {} (kernels \
                 {kernels}, eager {eager}, ks {ks:?} within 1..={cap})",
                if ok {
                    "ENGAGED"
                } else {
                    "NOT engaged, per-sequence decode_k kept"
                },
            );
        });
        ok.then_some(arena)
    }

    /// 2026-10-03: The DSA mixer for sequences `0..ks.len()`, sequence `i` holding the
    /// `ks[i]` rows of `hidden` from row `Σ ks[..i]`, with state `states[i]`, pre-step length
    /// `seq_lens[i]`, block table `block_tables[i]` and metadata `seq_meta[i]` (what the
    /// per-sequence loop hands that sequence's `decode_k` as `attn_metadata`). Writes the
    /// output projection over `hidden`, as the loop does. Returns `Ok(false)`, launching
    /// nothing, when `xseq_ready` does not hold; the caller then runs its loop.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_xseq(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        states: &mut [&mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        seq_meta: &[Option<AttnMetadataDev>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let n = ks.len();
        if states.len() < n || seq_lens.len() < n || block_tables.len() < n || seq_meta.len() < n {
            bail!(
                "DSA layer {}: decode_xseq of ks={ks:?} got states={} seq_lens={} \
                 block_tables={} metadata={}",
                self.layer_idx,
                states.len(),
                seq_lens.len(),
                block_tables.len(),
                seq_meta.len()
            );
        }
        let Some(arena) = self.xseq_ready(ks, ctx) else {
            return Ok(false);
        };
        let row_bytes = self.cfg.hidden * 2;
        let mut row0 = 0;
        for (i0, i1, rows) in xseq_groups(ks, arena.max_rows) {
            self.xseq_group(
                arena,
                hidden.offset(row0 * row_bytes),
                &ks[i0..i1],
                &mut states[i0..i1],
                kv_cache,
                &seq_lens[i0..i1],
                &block_tables[i0..i1],
                &seq_meta[i0..i1],
                ctx,
                stream,
            )?;
            row0 += rows;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::xseq_groups;

    #[test]
    fn groups_are_whole_sequences_within_the_cap() {
        assert_eq!(xseq_groups(&[3, 3, 3, 3], 16), [(0, 4, 12)]);
        assert_eq!(xseq_groups(&[1, 1, 1, 1], 16), [(0, 4, 4)]);
        assert_eq!(xseq_groups(&[3, 1, 2], 16), [(0, 3, 6)]);
        assert_eq!(
            xseq_groups(&[4, 4, 4, 4, 4, 4, 4, 4], 16),
            [(0, 4, 16), (4, 8, 16)]
        );
        assert_eq!(
            xseq_groups(&[3, 3, 3, 3, 3, 3], 16),
            [(0, 5, 15), (5, 6, 3)]
        );
        assert_eq!(xseq_groups(&[], 16), []);
    }
}
