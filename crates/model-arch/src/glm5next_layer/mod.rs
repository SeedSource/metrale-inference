// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextLayer`, the GLM-5.3 decoder layer behind `TransformerLayer`: a KDA or
//! DSA mixer, a dense or routed-MoE MLP, and the mHC hyper-connection.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every norm launches `rms_norm_vanilla` (`x * w / rms`), never `rms_norm`, which scales by
//!   `1 + w`.
//! - Highway slots are per token: `decode` uses slot 0, `prefill` gives prompt token `t` slot
//!   `t`, and `decode_batched` gives verify row `t` slot `t`.
//! - The last text layer collapses the highway with `hc_head_mean`, which takes no weights.
//! - Any all-reduce of a mixer or MLP output happens before `hc_post` folds that output into
//!   the highway.
//!
//! One mHC site, as `forward_one` and `forward_k` run it:
//!
//! ```text
//! layer 0 only:  hc_expand(hidden) -> streams            [hc_mult, hidden] FP32
//! each site:     hc_pre(streams) -> y (into hidden), post, comb
//!                rms_norm_vanilla(y, site norm) -> normed
//!                sublayer(normed) -> out
//!                hc_post(out, residual = streams, post, comb) -> streams
//! last layer:    hc_head_mean(streams) -> hidden
//! ```
//!
//! `hc_pre` only reads `streams`, so `hc_post` takes the same buffer as its residual and writes
//! its result over it.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_dsa::layer::Glm5NextDsaLayer;
use crate::glm5next_dsa::state::Glm5NextDsaState;
use crate::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaLayer, Glm5NextKdaWorkspace, KdaSeqState};
use crate::glm5next_mlp::forward::{
    Glm5NextMlpWorkspace, forward_dense_sliced, forward_moe_sliced,
};
use crate::glm5next_mlp::weights::{Glm5NextDenseMlpWeights, Glm5NextMoeWeights};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use metrale_model_layers::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use metrale_model_layers::layer::{
    LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
    LayerWriteOnAccept,
};
// 2026-09-25: GLM's own mHC launchers, not DeepSeek-V4's `ops::hc_pre`/`ops::hc_post`.
use crate::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_expand, glm_hc_post, glm_hc_pre,
    hc_head_mean,
};

pub mod state;

pub mod profile;
pub use state::alloc_kda_ssm_state;

mod levers;
mod steps;
mod types;
pub(crate) use levers::{PREFILL_ROWS, cublas_wide_proj, dsa_batch_qidx};
pub use levers::{
    PREFILL_ROWS_FFN_MAX, prefill_rows, prefill_rows_ffn, prefill_staged, staged_merge_signature,
};
pub use steps::staged::{ffn_windows, sub_chunks};
pub use types::{Glm5NextLayer, Glm5NextMhc, Glm5NextMixer, Glm5NextMlpSite};

impl TransformerLayer for Glm5NextLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(match &self.mixer {
            // 2026-09-25: On the model path a KDA layer gets SSM pool addresses instead
            // (`uses_ssm_pool`); this builds a zeroed, pool-free state for any other caller.
            Glm5NextMixer::Kda { cfg, .. } => Box::new(alloc_kda_ssm_state(gpu, cfg)?),
            Glm5NextMixer::Dsa(l) => Box::new(Glm5NextDsaState::alloc(gpu, &l.cfg)?),
        })
    }

    /// 2026-09-25: Frees a `Glm5NextDsaState`'s device buffers; the indexer cache is sized for
    /// the configured maximum context (`max_dsa_context`), not the prompt. Any other state is
    /// left alone, whatever the mixer: on the model path a KDA layer's `SsmLayerState` holds SSM
    /// pool addresses, which this layer does not own.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(dsa) = state.as_any_mut().downcast_mut::<Glm5NextDsaState>() {
            dsa.free(gpu)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_one(
            hidden,
            residual,
            0,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// 2026-09-25: Prefill with the highway indexed by token: prompt token `t` uses highway slot
    /// `t`, so every token's streams are still there when the next layer reads them. The trait's
    /// default runs each token through `decode`, which uses slot 0 for all of them.
    ///
    /// Errors when `num_tokens` exceeds `ctx.buffers.max_batch_tokens()`, the number of highway
    /// slots in the buffer arena.
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let cap = ctx.buffers.max_batch_tokens();
        if num_tokens > cap {
            bail!(
                "GLM layer {}: prefill of {num_tokens} tokens exceeds the {cap}-token mHC \
                 highway the buffer arena was sized for; each token needs its own slot",
                self.layer_idx
            );
        }
        // 2026-09-25: A text layer runs the prompt in sub-chunks of `prefill_rows()` tokens
        // through `forward_k`. The MTP block (`mhc: None`) walks token by token, because
        // `forward_k` refuses a layer without a highway.
        let rows = if self.mhc.is_some() {
            prefill_rows().min(cap)
        } else {
            1
        };
        // 2026-09-29: `METRALE_GLM_PREFILL_STAGED=1`, eager only: attention at `rows`, the FFN in
        // windows of up to `prefill_rows_ffn()` rows (`steps/staged.rs`). Under graph capture,
        // or with the lever off, the loop below runs as it always has.
        if rows > 1 && prefill_staged() && !ctx.graph_capture {
            return self.prefill_staged_run(
                hidden,
                num_tokens,
                rows,
                prefill_rows_ffn().min(cap).max(rows),
                state,
                kv_cache,
                seq_len_start,
                block_table,
                ctx,
                stream,
            );
        }
        if rows > 1 {
            let mut t = 0usize;
            while t < num_tokens {
                let k = rows.min(num_tokens - t);
                self.forward_k(
                    hidden.offset(t * self.hidden * 2),
                    k,
                    state,
                    kv_cache,
                    seq_len_start + t,
                    block_table,
                    ctx,
                    stream,
                    false,
                    t,
                    // 2026-09-25: `is_prefill` is true only for this caller.
                    // This IS the prefill sub-chunk caller.
                    true,
                )?;
                t += k;
            }
            return Ok(());
        }
        for t in 0..num_tokens {
            let off = t * self.hidden * 2;
            self.forward_one(
                hidden.offset(off),
                residual.offset(off),
                t,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: K tokens of one sequence in one call, the speculative-verify body:
    /// `forward_k` over highway slots `0..K`, taking KDA state snapshots. The trait's default
    /// calls `decode` per token, and `decode` uses highway slot 0 for every token.
    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: For a KDA layer, row `t < K - 1` writes intermediate `t`, and
        // `rollback_ssm_states_dispatch` restores intermediate `num_accepted - 1`.
        let kda_bytes = matches!(self.mixer, Glm5NextMixer::Kda { .. }).then_some(());
        if kda_bytes.is_some() && num_tokens > 1 {
            let st = self.kda_state(state)?;
            // 2026-09-25: Error rather than run without the intermediates: a rejected draft
            // could then not be rewound.
            if st.h_state_intermediates.len() + 1 < num_tokens
                || st.conv_state_intermediates.len() + 1 < num_tokens
            {
                bail!(
                    "GLM layer {}: a {num_tokens}-token verify needs {} per-token state \
                     snapshots but the pool has h={} conv={}. With none, this is the \
                     self-speculative / ngram path on a model whose MTP pool was never \
                     sized; with too few, --num-drafts exceeds the pool's tier.",
                    self.layer_idx,
                    num_tokens - 1,
                    st.h_state_intermediates.len(),
                    st.conv_state_intermediates.len(),
                );
            }
        }

        self.forward_k(
            hidden,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            true,
            0,
            // 2026-09-25: `is_prefill` is false here.
            // A speculative verify, NOT a prefill sub-chunk: true here would give an eager
            // verify the prefill-only batched DSA selector (`batch_select_enabled`).
            false,
        )
    }
}

impl LayerCapabilities for Glm5NextLayer {
    /// 2026-09-25: True. `decode` runs every sequence through highway slot 0, and the trait's
    /// default `decode_multi_seq` calls `decode` once per sequence with one shared
    /// `ForwardContext`, so every sequence would read and write the same highway slot. The DSA
    /// mixer's single-row `decode` also reads `attn_metadata` row 0 on a decode step.
    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }

    /// 2026-09-25: True. This layer has no `decode_verify_multi`, and the trait's default
    /// returns an error; answering true makes `can_batch_verify_dispatch` refuse the batched
    /// verify.
    fn decode_verify_multi_unsupported(&self) -> bool {
        true
    }

    /// 2026-09-25: True. A DSA layer's per-sequence state comes from `gpu.alloc` in
    /// `alloc_state`, so the addresses a captured graph holds belong to one sequence.
    fn graph_stale_on_new_sequence(&self) -> bool {
        true
    }

    /// 2026-09-25: True for a KDA layer. Its recurrent and conv state then come from the SSM
    /// pool, and with MTP on, so do the checkpoint and per-token intermediate pointers that a
    /// speculative rollback restores (`meta.rs`).
    fn uses_ssm_pool(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }

    /// 2026-09-25: True for a KDA layer, whose mixer carries recurrent state.
    fn is_ssm_layer(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }
}

impl LayerWeightSetup for Glm5NextLayer {}
impl LayerWriteOnAccept for Glm5NextLayer {}

impl LayerGraphHooks for Glm5NextLayer {
    /// 2026-09-25: A graph replay runs only kernels, so the host-side length of the DSA indexer
    /// cache is reconciled here (`Glm5NextDsaState::sync_to`). The inner `Glm5NextDsaLayer`
    /// implements this too, but the model's layer list holds this composite, so this impl is the
    /// one called. KDA keeps no host-side state.
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => self.dsa_state(state)?.sync_to(seq_len, k),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }

    /// 2026-09-25: Checks before a graph replay that the DSA indexer cache can hold `seq_len + k`
    /// rows. As with `sync_replayed_step`, this composite impl is the one the model calls.
    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => state
                .as_any()
                .downcast_ref::<Glm5NextDsaState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM layer {}: a DSA mixer was handed state that is not a \
                         Glm5NextDsaState",
                        self.layer_idx
                    )
                })?
                .ensure_room_through(seq_len + k)
                // 2026-09-25: `ensure_room_through` raises the same error for every caller;
                // the context names this route.
                .with_context(|| {
                    format!(
                        "DSA replay pre-check (layer {}, before launch_graph, seq_len \
                         {seq_len} + k {k})",
                        self.layer_idx
                    )
                }),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }
}

impl LayerAuxState for Glm5NextLayer {
    /// 2026-09-25: True for a DSA layer: its indexer cache is the state `snapshot_aux` and
    /// `restore_aux` carry. A KDA layer's state is in the SSM pool (`uses_ssm_pool`) and is not
    /// carried here.
    fn has_aux_state(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Dsa(_))
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        if !matches!(self.mixer, Glm5NextMixer::Dsa(_)) {
            return Ok(None);
        }
        let st = state
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a DSA mixer was handed state that is not a Glm5NextDsaState",
                    self.layer_idx
                )
            })?;
        Ok(Some(st.snapshot_blob(gpu, stream)?))
    }

    /// 2026-09-25: Errors on a KDA layer, whose state travels with the SSM snapshot. On a DSA
    /// layer it restores the blob through `Glm5NextDsaState::restore_blob`; `apply_aux_states`
    /// propagates any error.
    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !matches!(self.mixer, Glm5NextMixer::Dsa(_)) {
            bail!(
                "GLM layer {}: restore_aux on a KDA layer — KDA state is pool-backed and \
                 travels with the SSM snapshot, so a blob addressed here is a routing bug",
                self.layer_idx
            );
        }
        self.dsa_state(state)?.restore_blob(blob, gpu, stream)
    }
}

impl LayerSplitPrefill for Glm5NextLayer {}

#[cfg(test)]
mod tests;
