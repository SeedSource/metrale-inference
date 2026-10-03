// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pool-free allocation of a GLM-5.3 KDA layer's per-sequence state.
//!
//! The state is the SSM pool's `SsmLayerState`, not a GLM type:
//! `rollback_ssm_states_dispatch` downcasts the state of every `LinearAttention` layer to that
//! type, and every KDA layer is `LinearAttention`.
//!
//! 2026-10-02: Also `impl LayerAuxState for Glm5NextLayer`, moved here unchanged from `mod.rs` to
//! keep that file under the 500-line cap when the batched verify (`steps/verify_multi.rs`) joined.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: a DSA layer's indexer cache is the only aux state; KDA state travels with the SSM
//! snapshot.

use super::*;
use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::glm5next_kda::Glm5NextKdaConfig;
use metrale_model_layers::layer::SsmLayerState;

/// 2026-09-25: Allocate a KDA layer's recurrent and conv state outside the SSM pool, FP32
/// (`h_is_f16: false`), zeroed, with no checkpoints or intermediates. The model path uses pool
/// slots instead (`Glm5NextLayer::uses_ssm_pool`).
pub fn alloc_kda_ssm_state(gpu: &dyn GpuBackend, cfg: &Glm5NextKdaConfig) -> Result<SsmLayerState> {
    let h_bytes = cfg.recurrent_state_elems() * 4;
    let conv_bytes = cfg.conv_state_elems() * 4;
    let h_state = gpu.alloc(h_bytes)?;
    let conv_state = gpu.alloc(conv_bytes)?;
    gpu.memset_async(h_state, 0, h_bytes, 0)?;
    gpu.memset_async(conv_state, 0, conv_bytes, 0)?;
    gpu.synchronize(0)?;
    Ok(SsmLayerState {
        h_state,
        conv_state,
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    })
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

    /// 2026-10-03: Size of the blob `snapshot_aux_prefix` returns; `0` on a KDA layer.
    fn aux_prefix_bytes(&self, state: &dyn LayerState, rows: usize) -> usize {
        if !matches!(self.mixer, Glm5NextMixer::Dsa(_)) {
            return 0;
        }
        state
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .map_or(0, |st| st.blob_bytes_for_rows(rows))
    }

    /// 2026-10-03: The DSA indexer rows `[0, rows)` (`Glm5NextDsaState::snapshot_blob_prefix`):
    /// row `p` is written once, by the pass over position `p`, so after a pass that ran past
    /// `rows` the first `rows` rows are what a snapshot taken at `rows` holds. `None` on a KDA
    /// layer, as `snapshot_aux`.
    fn snapshot_aux_prefix(
        &self,
        state: &dyn LayerState,
        rows: usize,
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
        Ok(Some(st.snapshot_blob_prefix(rows, gpu, stream)?))
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
