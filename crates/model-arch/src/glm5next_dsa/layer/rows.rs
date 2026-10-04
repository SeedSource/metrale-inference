// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Three `Glm5NextDsaLayer` methods: `store_indexer_row`, the last step of
//! `indexer_forward`; `select_rows_batched`, the one-pass selection for all rows of a
//! prefill sub-chunk; and `write_kv_row`, one MTP drafter context row.
//! 2026-10-03: Plus `select_row_at`, the per-row selection (moved from `layer.rs`, 500-line
//! cap).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `write_kv_row` writes no KV latent when the indexer cache is behind `seq_len`.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::{ForwardContext, LayerState};

use super::super::select::split::{RowSplit, select_tokens_split};
use super::super::select::{DsaSelectInputs, select_tokens};
use super::super::state::Glm5NextDsaState;
use super::row_batch::DsaRowBatch;
use super::row_src::PreRow;
use super::{Glm5NextDsaLayer, gemm};

impl Glm5NextDsaLayer {
    /// 2026-09-25: The selector query projection and the selection for one query row, written
    /// to row `row` of the selection output.
    ///
    /// `decode_k` calls it right after that row's indexer write, so the geometry is planned
    /// at the row's own cache length; `attend_rows` then attends all rows in one launch.
    ///
    /// 2026-10-03: The row's `q_resid` is at `q_resid_row`. With `pre` (`xseq.rs`), the
    /// selector query and head weights were already computed for this row and are read from
    /// there; without, `wq_b` runs here into the workspace `q_idx` and the head weights are
    /// the ones `indexer_forward` left in the workspace.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_row_at(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &Glm5NextDsaState,
        q_pos_dev: DevicePtr,
        replay_safe: bool,
        q_resid_row: DevicePtr,
        pre: Option<PreRow>,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        let geom = state.geometry(&self.cfg, 1)?;

        let (q_idx, head_weights) = match pre {
            Some(p) => (p.q_idx, p.head_weights),
            None => {
                // 2026-09-25: The selector query `q_idx`, FP32 out.
                gemm(
                    gpu,
                    self.kernels.gemm_f32,
                    self.kernels.gemv_f32,
                    // 2026-09-25: No FP32-out batched GEMV kernel exists.
                    KernelHandle(0),
                    q_resid_row,
                    self.weights.wq_b,
                    w.q_idx,
                    1,
                    self.cfg.index_heads * self.cfg.index_head_dim,
                    self.cfg.q_lora_rank,
                    stream,
                )?;
                // 2026-09-25: `head_weights` comes from `indexer_forward`, which has the layer
                // input.
                (w.q_idx, w.head_weights)
            }
        };

        let inputs = DsaSelectInputs {
            k_normed: state.k_normed,
            gate: state.gate,
            valid: state.valid,
            ape: self.weights.ape,
            q: q_idx,
            weights: head_weights,
            q_pos: q_pos_dev,
            // 2026-09-25: All 1s, set once in `Glm5NextDsaWorkspace::new`.
            q_mask: w.q_mask,
            first_key: 0,
            geom_dev: if replay_safe {
                w.geom_dev
            } else {
                DevicePtr::NULL
            },
        };
        // 2026-09-25: On the replay-safe path the grid and the shared-memory request are set
        // at the context ceiling and the live extents come from `geom_dev`, so one graph
        // serves every context length.
        let launch = if replay_safe {
            super::super::select::DsaSelectLaunch::Ceiling {
                max_pools: super::super::select::contiguous_pool_count(
                    self.cfg.index_kpool,
                    super::super::state::max_dsa_context(&self.cfg),
                ),
            }
        } else {
            super::super::select::DsaSelectLaunch::Exact
        };
        let t = crate::glm5next_layer::profile::start();
        // 2026-09-25: Row `row` of the `[max_rows, out_width]` selection output. The kernels
        // run one query row (`q_rows == 1`) into this row's slot, so `attend_rows` reads all
        // rows in one launch.
        select_tokens(
            gpu,
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &w.select.row(row, &self.cfg),
            launch,
            stream,
        )?;
        use crate::glm5next_layer::profile;
        profile::end(profile::DSA_SELECT, t, gpu, stream);
        Ok(())
    }

    /// 2026-09-26: `indexer_forward`'s last step: places the projected `k_normed` and `gate`
    /// at row `pos` (through `dsa_indexer_store` at `pos_dev` when given), marks the row
    /// valid, and advances `state` by one row.
    pub(super) fn store_indexer_row(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut Glm5NextDsaState,
        pos_dev: Option<DevicePtr>,
        pos: usize,
        d: usize,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        match pos_dev {
            // 2026-09-25: Placement and the validity mark both use the device-side position;
            // a memset at `valid.offset(pos)` would fix a host address in a captured graph.
            Some(pd) => {
                KernelLaunch::new(gpu, self.select_kernels.indexer_store)
                    .grid([1, 1, 1])
                    .block([d.min(1024) as u32, 1, 1])
                    .arg_ptr(w.stage_k)
                    .arg_ptr(w.stage_gate)
                    .arg_ptr(pd)
                    .arg_ptr(state.k_normed)
                    .arg_ptr(state.gate)
                    .arg_ptr(state.valid)
                    .arg_u32(d as u32)
                    .launch(stream)?;
            }
            None => gpu.memset_async(state.valid.offset(pos), 1, 1, stream)?,
        }
        state.advance(1)
    }

    /// 2026-09-25: The selection for all `k` rows in one pass (`q_rows = k`), used when
    /// [`batch_select_enabled`](super::batch_select_enabled) holds; see there for why it
    /// selects what per-row passes do.
    ///
    /// The `wq_b` projection is one M = 1 GEMV per row, or one cuBLASLt GEMM for all rows
    /// when `METRALE_GLM_DSA_BATCH_QIDX=1` (`glm5next_layer::dsa_batch_qidx`, off by
    /// default).
    ///
    /// 2026-10-01: With `row_batch` (`METRALE_GLM_DSA_ROW_BATCH`), `q_pos` is the array
    /// `decode_rows_batched` already uploaded (the same values as `q_pos_host`), and the
    /// per-row GEMVs are the FP32-out batched GEMV, row for row the same bits.
    ///
    /// 2026-10-01: With `split` (`METRALE_GLM_DSA_INDEX_SPLIT`, row batch only), `wq_b` and
    /// the selection run for this rank's rows and the ranks swap their token rows
    /// (`select::split`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn select_rows_batched(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &Glm5NextDsaState,
        q_pos_host: &[i32],
        row_batch: Option<&DsaRowBatch>,
        split: Option<(RowSplit, &dyn metrale_comm::CommBackend)>,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        // 2026-09-25: `decode_k` calls this after all k indexer writes, so `P` is the group's
        // final pool count; each row limits itself to its own position through `q_pos[r]`.
        let geom = state.geometry(&self.cfg, k)?;
        let idx_row = self.cfg.index_heads * self.cfg.index_head_dim;
        // 2026-09-25: `q_resid` is a contiguous `[k, q_lora_rank]` BF16 block and `q_idx_rows`
        // a contiguous `[k, idx_row]` FP32 one, so one GEMM covers all rows. It sums in a
        // different order from the M = 1 GEMV, so near-tied index scores can select other
        // pools; that is why it is opt-in. Decode and the speculative verify never reach this
        // function (`batch_select_enabled`).
        if crate::glm5next_layer::dsa_batch_qidx() {
            metrale_model_layers::layers::ops::cublas_bf16_proj_dense_f32_out(
                w.q_resid,
                self.weights.wq_b,
                w.q_idx_rows,
                k as u32,
                idx_row as u32,
                self.cfg.q_lora_rank as u32,
                stream,
            )?;
        } else if let Some(rb) = row_batch {
            let (r0, rows) = split.map_or((0, k), |(s, _)| (s.r0, s.rows));
            self.qidx_rows_batched(gpu, rb, r0, rows, stream)?;
        } else {
            for row in 0..k {
                gemm(
                    gpu,
                    self.kernels.gemm_f32,
                    self.kernels.gemv_f32,
                    // 2026-09-25: No FP32-out batched GEMV kernel exists.
                    KernelHandle(0),
                    w.q_resid.offset(row * self.cfg.q_lora_rank * 2),
                    self.weights.wq_b,
                    w.q_idx_rows.offset(row * idx_row * 4),
                    1,
                    idx_row,
                    self.cfg.q_lora_rank,
                    stream,
                )?;
            }
        }
        let q_pos = match row_batch {
            Some(rb) => rb.q_pos_dev(),
            None => {
                let bytes: Vec<u8> = q_pos_host.iter().flat_map(|p| p.to_le_bytes()).collect();
                gpu.copy_h2d(&bytes, w.q_pos_rows)?;
                w.q_pos_rows
            }
        };
        let inputs = DsaSelectInputs {
            k_normed: state.k_normed,
            gate: state.gate,
            valid: state.valid,
            ape: self.weights.ape,
            q: w.q_idx_rows,
            weights: w.head_weights_rows,
            q_pos,
            q_mask: w.q_mask_rows,
            first_key: 0,
            // 2026-09-25: Host geometry only: `select_tokens` refuses the device-geometry
            // (ceiling) launch at `q_rows > 1`.
            geom_dev: DevicePtr::NULL,
        };
        let t = crate::glm5next_layer::profile::start();
        // 2026-09-25: The base of the `[max_rows, out_width]` output, not a row slice: the
        // kernels index rows themselves, so rows 0..k land in their own slots.
        let (kernels, cfg) = (&self.select_kernels, &self.cfg);
        match split {
            Some((s, comm)) => select_tokens_split(
                gpu,
                kernels,
                cfg,
                &geom,
                &inputs,
                &w.select,
                s,
                comm,
                stream,
            )?,
            None => select_tokens(
                gpu,
                kernels,
                cfg,
                &geom,
                &inputs,
                &w.select,
                super::super::select::DsaSelectLaunch::Exact,
                stream,
            )?,
        }
        crate::glm5next_layer::profile::end(
            crate::glm5next_layer::profile::DSA_SELECT,
            t,
            gpu,
            stream,
        );
        Ok(())
    }

    /// 2026-09-25: One MTP drafter context row: writes the KV latent and the indexer row, with
    /// no query, selection or attention.
    ///
    /// The row goes to KV slot `seq_len`, and the indexer writes at `state.len()`. An indexer
    /// cache ahead of `seq_len` is rewound; one behind is an error, so rows must be written
    /// densely from 0.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_row(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        match st.len().cmp(&seq_len) {
            std::cmp::Ordering::Greater => st.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA layer {}: indexer cache holds {} rows but the drafter is at {seq_len} — \
                 rows are MISSING, not merely stale.",
                self.layer_idx,
                st.len()
            ),
            std::cmp::Ordering::Equal => {}
        }
        let w = &self.workspace;
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.kv_a_proj,
            w.kv_a,
            1,
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        let block_size = kv_cache.config().block_size;
        let logical = seq_len / block_size;
        let physical = *block_table.get(logical).ok_or_else(|| {
            anyhow::anyhow!(
                "DSA layer {}: block table has {} entries, needs logical block {logical} for \
                 drafter row {seq_len}",
                self.layer_idx,
                block_table.len()
            )
        })? as usize;
        let slot = (physical * block_size + seq_len % block_size) as i64;
        gpu.copy_h2d(&slot.to_le_bytes(), w.slot)?;
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([1, 1, 1])
            .block([self.cfg.kv_lora_rank as u32, 1, 1])
            .arg_ptr(w.kv_a)
            .arg_ptr(self.weights.kv_a_layernorm)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(w.slot)
            .arg_u32(self.cfg.kv_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .arg_f32(1.0 / self.kv_scale)
            .launch(stream)?;
        self.indexer_forward(gpu, hidden, st, None, stream)
    }
}
