// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `Glm5NextDsaLayer::decode_k`, `k` consecutive tokens of one sequence: the
//! shared projections, then per row the latent write, the indexer write and the selection,
//! then one gather-attend and the output projection.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The lockstep check (`check_lockstep`) and the `k` and metadata checks run before the
//!   first launch.

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layer::{ForwardContext, LayerState};

use super::super::attend::DsaDecodePaging;
use super::super::state::Glm5NextDsaState;
use super::{Glm5NextDsaLayer, batch_select_enabled, gemm};

impl Glm5NextDsaLayer {
    /// 2026-09-25: `k` consecutive tokens of one sequence, from position `seq_len`.
    ///
    /// `q_a_proj`, `q_absorb`, `kv_a_proj` and the `o_absorb` output projection each run once
    /// for all `k` rows. Per row, in order: the latent write to the row's paged slot, the
    /// indexer write, and (unless the batched selector runs) that row's selection. After the
    /// rows: the batched selection when `batch_select_enabled` holds, then one `attend_rows`
    /// launch.
    ///
    /// Row `r`'s position, KV slot, `seq_len` and block table come from element `r` of
    /// `ctx.attn_metadata` on a decode step, or when `k > 1` and the metadata has exactly `k`
    /// rows; otherwise they are computed here and copied to the device. On a decode step with
    /// `k > 1`, metadata with another row count is an error.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_k(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        // 2026-09-25: True only for a prefill sub-chunk, as `Glm5NextLayer::forward_k` passes
        // it; see `batch_select_enabled`.
        is_prefill: bool,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let bt_block_size = kv_cache.block_size().max(1);
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        self.check_lockstep(st, seq_len)?;
        if k == 0 || k > self.workspace.max_rows {
            bail!(
                "DSA layer {}: a {k}-token verify does not fit a workspace built for {}",
                self.layer_idx,
                self.workspace.max_rows
            );
        }
        if k > 1
            && let Some(m) = ctx.attn_metadata.as_ref()
            && m.num_seqs as usize != k
            && ctx.decode_step
        {
            bail!(
                "DSA layer {}: a {k}-row pass cannot share attn_metadata describing {} \
                 token(s) — its position and KV slot describe a single token",
                self.layer_idx,
                m.num_seqs
            );
        }
        let rowwise_meta = (k > 1)
            .then_some(ctx.attn_metadata.as_ref())
            .flatten()
            .filter(|m| m.num_seqs as usize == k);
        let gpu = ctx.gpu;
        let w = &self.workspace;
        let t_proj = crate::glm5next_layer::profile::start();

        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.q_a_proj,
            w.q_a,
            k,
            self.cfg.q_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        KernelLaunch::new(gpu, self.kernels.rms_norm)
            // 2026-09-25: `rms_norm_vanilla` runs one block per row, so one launch covers all
            // k rows.
            .grid([k as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.q_a)
            .arg_ptr(self.weights.q_a_layernorm)
            .arg_ptr(w.q_resid)
            .arg_u32(self.cfg.q_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            w.q_resid,
            self.weights.q_absorb,
            w.q_abs,
            k,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            self.cfg.q_lora_rank,
            stream,
        )?;

        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.kv_a_proj,
            w.kv_a,
            k,
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        // 2026-10-01: `METRALE_GLM_DSA_ROW_BATCH=1`: everything below for all rows at once
        // (`row_batch.rs`), only where the batched selector runs; otherwise the row loop.
        let select_rows =
            batch_select_enabled(w.q_idx_rows.0 != 0, is_prefill, ctx.graph_capture, k);
        if let Some(rb) = self.row_batch_ready(gpu, select_rows, stream) {
            let meta = if ctx.decode_step {
                ctx.attn_metadata.as_ref()
            } else {
                rowwise_meta
            };
            return self.decode_rows_batched(
                rb,
                gpu,
                hidden,
                k,
                st,
                kv_cache,
                seq_len,
                block_table,
                meta,
                t_proj,
                ctx.comm,
                stream,
            );
        }
        let mut attend_bt = DevicePtr::NULL;
        let mut attend_sl = DevicePtr::NULL;
        // 2026-09-25: On the host path without `persist_bt`, row 0 allocates the shared bt/sl
        // buffers; they are freed after the attend, which reads them.
        let mut owns_scratch = false;
        let mut attend_paging: Option<DsaDecodePaging> = None;
        // 2026-09-25: `workspace_ready` is whether the batched-selector buffers were
        // allocated (they are NULL under `METRALE_DSA_SELECT_ROWS=0`).
        let batch_select =
            batch_select_enabled(w.q_idx_rows.0 != 0, is_prefill, ctx.graph_capture, k);
        let mut batch_q_pos: Vec<i32> = Vec::with_capacity(if batch_select { k } else { 0 });
        for row in 0..k {
            let pos = seq_len + row;
            let block_size = kv_cache.config().block_size;
            // 2026-09-25: With `meta`, the row's position, KV slot, `seq_len` and block table
            // are read from device arrays and nothing is copied from the host; without it they
            // are computed here and copied host-to-device.
            let meta = if ctx.decode_step {
                ctx.attn_metadata.as_ref()
            } else {
                rowwise_meta
            };
            let bt_stride = meta.map_or(0, |m| m.max_blocks_per_seq as usize) * 4;
            let slot_dev = match meta {
                Some(m) => m.slot.offset(row * 8),
                None => {
                    let logical = pos / block_size;
                    let physical = *block_table.get(logical).ok_or_else(|| {
                        anyhow::anyhow!(
                            "DSA layer {}: block table has {} entries, needs logical block \
                             {logical} for position {pos}",
                            self.layer_idx,
                            block_table.len()
                        )
                    })? as usize;
                    let slot = (physical * block_size + pos % block_size) as i64;
                    gpu.copy_h2d(&slot.to_le_bytes(), w.slot)?;
                    w.slot
                }
            };
            KernelLaunch::new(gpu, self.kernels.latent_write)
                .grid([1, 1, 1])
                .block([self.cfg.kv_lora_rank as u32, 1, 1])
                .arg_ptr(w.kv_a.offset(row * self.cfg.kv_lora_rank * 2))
                .arg_ptr(self.weights.kv_a_layernorm)
                .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
                .arg_ptr(slot_dev)
                .arg_u32(self.cfg.kv_lora_rank as u32)
                .arg_f32(self.rms_eps)
                .arg_f32(1.0 / self.kv_scale)
                .launch(stream)?;

            use crate::glm5next_layer::profile;
            profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
            let t = profile::start();
            // 2026-09-25: Replay-safe placement (device-side position and geometry) only while
            // capturing a graph, with metadata, and with both `dsa_indexer_store` and
            // `dsa_write_geom` resolved; otherwise the host-offset path.
            let replay_safe = ctx.graph_capture
                && meta.is_some()
                && self.select_kernels.indexer_store.0 != 0
                && self.select_kernels.write_geom.0 != 0;
            let pos_dev = if replay_safe {
                meta.map(|m| m.positions.offset(row * 4))
            } else {
                None
            };
            self.indexer_forward(
                gpu,
                hidden.offset(row * self.cfg.hidden * 2),
                st,
                pos_dev,
                stream,
            )?;
            profile::end(profile::DSA_INDEXER, t, gpu, stream);

            let (q_pos_dev, bt_dev_meta, sl_dev_meta) = match meta {
                Some(m) => (
                    m.positions.offset(row * 4),
                    Some(m.block_table.offset(row * bt_stride)),
                    Some(m.seq_len.offset(row * 4)),
                ),
                None => {
                    let qp = pos as i32;
                    gpu.copy_h2d(&qp.to_le_bytes(), w.q_pos)?;
                    (w.q_pos, None, None)
                }
            };
            let (d_bt, d_sl) = match (bt_dev_meta, sl_dev_meta) {
                // 2026-09-25: The metadata already holds both; nothing to copy.
                (Some(b), Some(l)) => (b, l),
                _ => {
                    // 2026-09-25: Upload only the prefix the gather can index
                    // ([`bt_entries_needed`]), not the caller's whole table.
                    let bt_used = {
                        let needed = bt_entries_needed(seq_len, k, bt_block_size);
                        &block_table[..needed.min(block_table.len())]
                    };
                    let bt: Vec<u8> = bt_used.iter().flat_map(|b| b.to_le_bytes()).collect();
                    if bt_used.len() > w.bt_cap {
                        anyhow::bail!(
                            "DSA layer {}: block table needs {} entries for seq_len {} + {} rows \
                     but the persistent buffer holds {}. This is a BLOCK count against a buffer \
                     sized by max_dsa_context (a TOKEN count); do not write past the allocation.",
                            self.layer_idx,
                            bt_used.len(),
                            seq_len,
                            k,
                            w.bt_cap
                        );
                    }
                    // 2026-09-25: `attend_rows` reads these buffers after the row loop, so they
                    // outlive the row that wrote them. Each row writes its own `sl[row]`. The
                    // block table is the same for every row (the rows are one sequence, and
                    // `bt_entries_needed` does not depend on `row`), and it is read with a row
                    // stride of 0 (`max_blocks_per_seq` below).
                    let (d_bt, d_sl) = if self.persist_bt {
                        (w.bt, w.sl)
                    } else if row == 0 {
                        (gpu.alloc(bt.len().max(4))?, gpu.alloc(k * 4)?)
                    } else {
                        // 2026-09-25: Row 0 allocated these; later rows write their own `sl`
                        // slot into them.
                        (attend_bt, attend_sl)
                    };
                    gpu.copy_h2d(&bt, d_bt)?;
                    gpu.copy_h2d(&((pos + 1) as i32).to_le_bytes(), d_sl.offset(row * 4))?;
                    (d_bt, d_sl)
                }
            };
            let owns_bt = bt_dev_meta.is_none();

            let paging = DsaDecodePaging {
                num_seqs: 1,
                num_q_heads: self.cfg.local_heads,
                num_kv_heads: 1,
                // 2026-09-25: The kernel's block-table row stride, and a kernel argument that a
                // captured graph fixes. With metadata it is the metadata's
                // `max_blocks_per_seq`, the stride of its per-row tables; without, all rows
                // share the one table uploaded above, so the stride is 0.
                max_blocks_per_seq: match meta {
                    Some(m) => m.max_blocks_per_seq as usize,
                    None => 0,
                },
                block_size,
                cache_stride_bytes: (block_size * self.cfg.kv_lora_rank) as u64,
            };
            if replay_safe {
                // 2026-09-25: `dsa_write_geom` reads S from `d_sl`, this row's `seq_len` entry
                // in the metadata.
                KernelLaunch::new(gpu, self.select_kernels.write_geom)
                    .grid([1, 1, 1])
                    .block([1, 1, 1])
                    .arg_ptr(d_sl)
                    .arg_ptr(w.geom_dev)
                    .arg_u32(self.cfg.index_kpool as u32)
                    .arg_u32(self.cfg.index_topk as u32)
                    .arg_u32(super::super::select::topk_tile() as u32)
                    .launch(stream)?;
            }
            if batch_select {
                // 2026-09-25: `indexer_forward` left this row's head weights in the single-row
                // slot; copy them to row `row` for the batched pass, which reads `weights[r*H]`.
                gpu.copy_d2d_async(
                    w.head_weights,
                    w.head_weights_rows.offset(row * self.cfg.index_heads * 4),
                    self.cfg.index_heads * 4,
                    stream,
                )?;
                batch_q_pos.push(pos as i32);
            } else {
                self.select_row(gpu, row, st, q_pos_dev, replay_safe, stream)?;
            }
            // 2026-09-25: The attend takes row 0's pointers, the base of the per-row arrays,
            // and indexes rows itself on grid y.
            if row == 0 {
                attend_bt = d_bt;
                attend_sl = d_sl;
                attend_paging = Some(paging);
                owns_scratch = owns_bt && !self.persist_bt;
            }
        }

        // 2026-09-25: The batched selection runs after the row loop, so every row's indexer
        // write is in the cache. Row `r` only takes pools that end at or before `q_pos[r]`,
        // so the rows written after it do not change its selection.
        if batch_select && !batch_q_pos.is_empty() {
            self.select_rows_batched(gpu, k, st, &batch_q_pos, None, None, stream)?;
        }

        if let Some(paging) = attend_paging {
            self.attend_rows(gpu, k, st, kv_cache, attend_bt, attend_sl, &paging, stream)?;
        }
        // 2026-09-25: Freed after the attend, which reads both buffers.
        if owns_scratch {
            gpu.free(attend_bt)?;
            gpu.free(attend_sl)?;
        }

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
}

/// 2026-09-25: Block-table entries to upload for `k` query rows starting at `seq_len`.
///
/// Row `r` attends tokens below its `seq_len` of `seq_len + r + 1`, so the gather reads
/// `block_table[t / block_size]` only for `t < seq_len + k`: at most index
/// `(seq_len + k - 1) / block_size`. This returns at least one entry more. A `block_size`
/// of 0 is treated as 1.
pub(super) fn bt_entries_needed(seq_len: usize, k: usize, block_size: usize) -> usize {
    (seq_len + k) / block_size.max(1) + 2
}
