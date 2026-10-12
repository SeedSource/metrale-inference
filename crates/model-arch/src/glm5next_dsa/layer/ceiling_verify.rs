// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_DSA_CEILING_ROWS=1` (default off, `select::dsa_ceiling_rows`):
//! on the replay-safe (graph-captured) path of `decode_k_rows`, the `k >= 2` verify rows of
//! one sequence are selected by ONE `DsaSelectLaunch::CeilingRows` pass after the row loop,
//! instead of a `dsa_write_geom_pk` and a single-row ceiling selection per row.
//!
//! The pass is the microtest's arm B (`examples/common/dsa_ceiling_rows_rig.rs`, `Arm::Rows`):
//! all `k` indexer stores (the row loop), ONE `dsa_write_geom_pk` from `seq_len[k - 1]`, ONE
//! selection with `q_rows = k` over the contiguous `[k, H, D]` selector queries, `[k, H]` head
//! weights and `positions[k]`, into the base of the `[k, out_width]` selection output. Each
//! row's query and head weights are the bytes the per-row path computes (the same M = 1
//! launches, only the output row differs). The attend after the loop is unchanged.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - 2026-10-12: [`ceiling_rows_group`] is the only gate; every input of the pass is a device
//!   buffer the step metadata or the workspace already holds (no host copy, no sync).

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layer::AttnMetadataDev;

use super::super::select::{
    DsaSelectInputs, DsaSelectLaunch, contiguous_pool_count, dsa_ceiling_rows, select_tokens,
};
use super::super::state::{Glm5NextDsaState, max_dsa_context};
use super::row_src::RowSrc;
use super::{Glm5NextDsaLayer, gemm};

/// 2026-10-12: What [`ceiling_rows_group`] decides from, all known on the host before the
/// row loop's first launch.
#[derive(Debug, Clone, Copy)]
pub(super) struct CeilingRowsFacts {
    /// `METRALE_GLM_DSA_CEILING_ROWS=1`.
    pub(super) lever: bool,
    /// Rows of this `decode_k_rows` call (one state, so one sequence).
    pub(super) k: usize,
    /// The row loop's replay-safe placement (graph capture, metadata, store/geom kernels).
    pub(super) replay_safe: bool,
    /// The state keeps the pool cache (`dsa_write_geom_pk` geometry).
    pub(super) pool_cache: bool,
    /// `dsa_write_geom_pk` resolved.
    pub(super) geom_pk: bool,
    /// The batched prefill selector runs instead (never under capture).
    pub(super) batch_select: bool,
    /// Rows the call's metadata describes (`num_seqs`), `None` without metadata.
    pub(super) meta_rows: Option<usize>,
    /// The `[max_rows]` query / head-weight / mask buffers the pass reads exist.
    pub(super) rows_ready: bool,
}

/// 2026-10-12: Whether `decode_k_rows` runs the one `CeilingRows` pass. All must hold:
/// the lever; at least 2 rows; the replay-safe path with the pool cache and
/// `dsa_write_geom_pk` (the geometry the microtest proved); not the batched selector; and
/// metadata describing exactly these `k` rows, so row `r` of `positions` / `seq_len` is the
/// call's row `r` (a view over several sequences' rows, `num_seqs != k`, falls back).
pub(super) fn ceiling_rows_group(f: &CeilingRowsFacts) -> bool {
    f.lever
        && f.k >= 2
        && f.replay_safe
        && f.pool_cache
        && f.geom_pk
        && !f.batch_select
        && f.meta_rows == Some(f.k)
        && f.rows_ready
}

impl Glm5NextDsaLayer {
    /// 2026-09-25: Replay-safe placement (device-side position and geometry) only while
    /// capturing a graph, with metadata, and with both `dsa_indexer_store` and
    /// `dsa_write_geom` resolved; otherwise the host-offset path. 2026-10-12: moved here from
    /// the row loop unchanged, so the gate below reads the same value.
    pub(super) fn replay_safe(&self, graph_capture: bool, has_meta: bool) -> bool {
        graph_capture
            && has_meta
            && self.select_kernels.indexer_store.0 != 0
            && self.select_kernels.write_geom.0 != 0
    }

    /// 2026-10-12: [`ceiling_rows_group`] for one `decode_k_rows` call of `k` rows.
    pub(super) fn ceiling_rows_for(
        &self,
        st: &Glm5NextDsaState,
        meta: Option<&AttnMetadataDev>,
        replay_safe: bool,
        batch_select: bool,
        src: &RowSrc,
        k: usize,
    ) -> bool {
        let w = &self.workspace;
        let own_rows = w.q_idx_rows.0 != 0 && w.head_weights_rows.0 != 0;
        ceiling_rows_group(&CeilingRowsFacts {
            lever: dsa_ceiling_rows(),
            k,
            replay_safe,
            pool_cache: st.is_pool_cache(),
            geom_pk: self.select_kernels.write_geom_pk.0 != 0,
            batch_select,
            meta_rows: meta.map(|m| m.num_seqs as usize),
            rows_ready: w.q_mask_rows.0 != 0 && (src.pre.is_some() || own_rows),
        })
    }

    /// 2026-10-12: Where `indexer_forward_to` writes row `row`'s head weights: row `row` of
    /// `head_weights_rows` when the call is grouped, else the single-row slot as before.
    pub(super) fn ceiling_rows_hw(&self, group: bool, row: usize) -> DevicePtr {
        let w = &self.workspace;
        if group {
            w.head_weights_rows.offset(row * self.cfg.index_heads * 4)
        } else {
            w.head_weights
        }
    }

    /// 2026-10-12: The one `CeilingRows` selection of the call's `k` rows, after all `k`
    /// indexer stores. Without `src.pre`, `wq_b` runs per row (M = 1, the launch
    /// `select_row_at` makes) into row `r` of `q_idx_rows`; the head weights are already in
    /// `head_weights_rows` ([`Self::ceiling_rows_hw`]). With `src.pre` both are its rows.
    pub(super) fn select_rows_ceiling(
        &self,
        gpu: &dyn GpuBackend,
        st: &mut Glm5NextDsaState,
        src: &RowSrc,
        k: usize,
        meta: &AttnMetadataDev,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        let cfg = &self.cfg;
        let (q, weights) = match &src.pre {
            Some(p) => (p.q_idx, p.head_weights),
            None => {
                let nq = cfg.index_heads * cfg.index_head_dim;
                for row in 0..k {
                    gemm(
                        gpu,
                        self.kernels.gemm_f32,
                        self.kernels.gemv_f32,
                        // 2026-09-25: No FP32-out batched GEMV kernel exists.
                        KernelHandle(0),
                        src.q_resid.offset(row * cfg.q_lora_rank * 2),
                        self.weights.wq_b,
                        w.q_idx_rows.offset(row * nq * 4),
                        1,
                        nq,
                        cfg.q_lora_rank,
                        stream,
                    )?;
                }
                (w.q_idx_rows, w.head_weights_rows)
            }
        };
        // 2026-10-12: The LAST row's `seq_len`: its pool count is the row stride every kernel
        // reads, and each row keeps only pools ending at or before its own `q_pos`.
        let sl_last = meta.seq_len.offset((k - 1) * 4);
        self.pool_write_geom_pk(gpu, st, sl_last, w.geom_dev, stream)?;
        let geom = st.geometry(cfg, k)?;
        let inputs = DsaSelectInputs {
            k_normed: st.k_normed,
            gate: st.gate,
            valid: st.valid,
            ape: self.weights.ape,
            q,
            weights,
            q_pos: meta.positions,
            // 2026-10-12: All 1s, `[max_rows]`; the expand reads `q_mask[r]` per row.
            q_mask: w.q_mask_rows,
            first_key: 0,
            geom_dev: w.geom_dev,
            pool_cache: st.pool_select_args()?,
        };
        let max_pools = contiguous_pool_count(cfg.index_kpool, max_dsa_context(cfg));
        let launch = DsaSelectLaunch::CeilingRows { max_pools };
        let t = crate::glm5next_layer::profile::start();
        select_tokens(
            gpu,
            &self.select_kernels,
            cfg,
            &geom,
            &inputs,
            &w.select,
            launch,
            stream,
        )?;
        st.note_selected();
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "DSA CEILING_ROWS: ENGAGED (layer {}, k = {k})",
                self.layer_idx
            )
        });
        use crate::glm5next_layer::profile;
        profile::end(profile::DSA_SELECT, t, gpu, stream);
        Ok(())
    }
}

#[cfg(test)]
#[path = "ceiling_verify_tests.rs"]
mod tests;
