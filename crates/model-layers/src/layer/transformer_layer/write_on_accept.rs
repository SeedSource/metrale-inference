// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerWriteOnAccept`, the GDN write-on-accept hooks: stash sizing, buffer
//! binding, and the fold of the accepted rows after a batched verify; and the
//! carried-state verify's sizing and binding.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-26: A GDN layer's device buffers for the carried-state verify, set once by the
/// model (`model-engine gdn_carry.rs`) before any capture.
#[derive(Clone, Copy, Debug)]
pub struct GdnCarryBinding {
    /// 2026-09-26: This layer's engaged words, one `u32` per batch position; a carry
    /// kernel sets each position it verified to 1.
    pub flag: DevicePtr,
    /// 2026-09-26: This layer's stash, `slots * seq_floats` f32.
    pub stash: DevicePtr,
    /// 2026-09-26: This layer's pending-row counts, one `u32` per slot.
    pub pend: DevicePtr,
    /// 2026-09-26: Batch position to slot, shared by every layer.
    pub slot_tab: DevicePtr,
    pub seq_floats: usize,
    /// 2026-09-26: This layer's conv stash, `slots * conv_seq_elems` BF16 (four input rows
    /// per slot).
    pub conv_stash: DevicePtr,
    pub conv_seq_elems: usize,
    /// 2026-09-26: This layer's conv-state pointer per batch position, staged per verify;
    /// a declining layer folds through it.
    pub conv_tab: DevicePtr,
}

/// 2026-10-09: `METRALE_GLM_KDA_DEFER_COMMIT=1`: where row 0 of one sequence's K-row verify
/// reads its input recurrent and conv state. A partial-accept commit no longer copies verify
/// intermediate `a - 1` into the live state; it leaves the index in the device `i32` at `idx`
/// (`-1`: the live state is current). The kernel reads that word, so a captured verify graph
/// stays valid; `h_base` / `conv_base` are this layer's intermediate 0 for the slot and the
/// strides the bytes between consecutive intermediates (model-engine `kda_defer.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsmDeferSrc {
    pub idx: DevicePtr,
    pub h_base: DevicePtr,
    pub h_stride: usize,
    pub conv_base: DevicePtr,
    pub conv_stride: usize,
}

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerWriteOnAccept {
    /// 2026-09-25: Per-sequence write-on-accept stash size in f32 elements when this layer
    /// can run write-on-accept, `None` otherwise (the default). The model sizes the stash
    /// from the largest answer and binds each GDN layer with [`Self::gdn_woa_bind`].
    fn gdn_woa_stash_seq_floats(&self) -> Option<usize> {
        None
    }

    /// 2026-09-25: Bind this layer's write-on-accept flag word and stash slab (`seqs`
    /// sequences of [`Self::gdn_woa_stash_seq_floats`] f32 each). The model calls it
    /// once, on the first write-on-accept request and before its graph decision, and
    /// never moves the buffers afterwards (`gdn_woa.rs`).
    fn gdn_woa_bind(&self, _flag: DevicePtr, _stash: DevicePtr, _seqs: usize) {}

    /// 2026-09-26: Per-slot carried-state stash size in f32 elements when this layer can
    /// run the carried-state verify, `None` otherwise (the default).
    fn gdn_carry_seq_floats(&self) -> Option<usize> {
        None
    }

    /// 2026-09-26: Bind this layer's carried-state buffers. Called once, before any
    /// capture; the buffers never move afterwards.
    fn gdn_carry_bind(&self, _binding: GdnCarryBinding) {}

    /// 2026-10-09: Whether this layer's K-row verify honours [`SsmDeferSrc`]
    /// (`METRALE_GLM_KDA_DEFER_COMMIT=1` with the kernels it needs). The model defers a commit
    /// only when every linear-attention layer answers true. Default false.
    fn kda_defer_commit_ready(&self) -> bool {
        false
    }

    /// 2026-09-25: Apply the accepted rows of the last batched verify to this layer's h
    /// states. `h_table` is the layer's WY pointer-table slice, `na_tab` a device `u32[n]`
    /// of accepted row counts in batch order. `Ok(false)` when the layer did nothing, as
    /// the default does.
    fn gdn_fold_accepted(
        &self,
        _gpu: &dyn GpuBackend,
        _h_table: DevicePtr,
        _na_tab: DevicePtr,
        _k_rows: usize,
        _n: usize,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }
}
