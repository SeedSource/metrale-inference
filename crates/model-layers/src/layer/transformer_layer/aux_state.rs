// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerAuxState`, the host-serialized per-sequence state that travels with an SSM
//! snapshot.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::layer::LayerState;

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerAuxState {
    /// 2026-09-25: Host-serialized per-sequence state that must travel with an SSM
    /// snapshot for a prefix-cache hit to be complete, such as the PLE and QSA carries.
    /// Device-to-host copies inside must be ordered on `stream`. Default `None`.
    fn snapshot_aux(
        &self,
        _state: &dyn LayerState,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    /// 2026-10-03: [`Self::snapshot_aux`] as of the first `rows` positions of the sequence, for
    /// an SSM snapshot captured inside a prefill pass that ran past `rows`
    /// (model-engine `prefill_b/inpass_capture.rs`). The default can only answer for a layer
    /// with no aux blob: it returns `Ok(None)` when `snapshot_aux` does, and an error when
    /// `snapshot_aux` returns a blob it cannot cut to `rows`.
    fn snapshot_aux_prefix(
        &self,
        state: &dyn LayerState,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        match self.snapshot_aux(state, gpu, stream)? {
            None => Ok(None),
            Some(_) => anyhow::bail!(
                "snapshot_aux_prefix({rows}): this layer's aux blob cannot be cut to a prefix"
            ),
        }
    }

    /// 2026-09-25: True when this layer produces aux state. When any layer does, restore
    /// sites decline a snapshot that has no aux blobs (`requires_aux_state`).
    fn has_aux_state(&self) -> bool {
        false
    }

    /// 2026-09-25: Apply a blob from [`Self::snapshot_aux`] to this layer's state
    /// (`apply_aux_states`). The default returns an error.
    fn restore_aux(
        &self,
        _state: &mut dyn LayerState,
        _blob: &[u8],
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("restore_aux on a layer with no aux state")
    }
}
