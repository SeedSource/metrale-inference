// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `TransformerLayer` trait, the per-layer hooks the model drives (grouped
//! into supertraits in the child modules), and the layout constants of the batched-verify
//! WY pointer tables.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{ForwardContext, GdnPrefillBuffers, LayerState};

mod aux_state;
mod capabilities;
mod default_loops;
mod graph_hooks;
mod split_prefill;
mod weight_setup;
mod write_on_accept;

pub use aux_state::LayerAuxState;
pub use capabilities::LayerCapabilities;
pub use graph_hooks::LayerGraphHooks;
pub use split_prefill::LayerSplitPrefill;
pub use weight_setup::LayerWeightSetup;
pub use write_on_accept::{GdnCarryBinding, LayerWriteOnAccept};

/// 2026-09-25: Batched-verify WY pointer-table layout. `TransformerModel::upload_verify_wy_tables`
/// (model-engine `verify_e2.rs`) writes it; the qwen3_ssm `GdnStates::Multi` arm and the
/// write-on-accept fold read it.
///
/// Per GDN layer, `VERIFY_WY_TABLES_PER_LAYER` tables of `VERIFY_WY_TABLE_SEQS` u64
/// entries lie back to back: table 0 holds each sequence's `h_state`, table `t + 1` its
/// intermediate `t`. A k-row verify writes at most the first k tables and leaves every
/// other entry zero. The strides do not depend on k.
///
/// This constant is the most sequences one batched verify holds: it sizes the tables and
/// the verify hidden and catch-up stashes (`impl_a1.rs`), and `can_batch_verify_dispatch`
/// (`verify_e.rs`) admits at most this many.
pub const VERIFY_WY_TABLE_SEQS: usize = 32;
/// 2026-09-25: Catch-up stash rows per sequence (`ModelLevers::mtp_kv_exact`). The batched
/// verify step stashes at most this many accepted drafts per sequence
/// (`verify_k4_batch_step.rs`).
pub const MTP_CATCHUP_MAX: usize = 4;
/// 2026-09-25: Tables per GDN layer: `h_state` plus up to 15 intermediates.
/// `upload_verify_wy_tables` stages a verify only when k is in
/// `2..=VERIFY_WY_TABLES_PER_LAYER`, and returns NULL otherwise.
pub const VERIFY_WY_TABLES_PER_LAYER: usize = 16;
/// 2026-09-25: Bytes between consecutive tables within a layer slice.
pub const VERIFY_WY_TABLE_STRIDE_BYTES: usize = VERIFY_WY_TABLE_SEQS * 8;
/// 2026-09-25: Bytes between consecutive GDN layers' table slices.
pub const VERIFY_WY_LAYER_STRIDE_BYTES: usize =
    VERIFY_WY_TABLES_PER_LAYER * VERIFY_WY_TABLE_STRIDE_BYTES;

pub trait TransformerLayer:
    Send
    + Sync
    + LayerCapabilities
    + LayerWeightSetup
    + LayerWriteOnAccept
    + LayerGraphHooks
    + LayerAuxState
    + LayerSplitPrefill
{
    /// 2026-09-25: Decode one token through this layer, updating `hidden` in place.
    ///
    /// * `hidden` - `[1, hidden_size]` BF16, read and written
    /// * `residual` - `[1, hidden_size]` BF16 residual stream
    /// * `state` - this layer's state for the sequence
    /// * `seq_len` - the token's position
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        // 2026-09-25: `--high-speed-swap` disk block ids, one per logical block the sequence
        // has had; `block_table` holds the newest of those blocks. Empty without the feature.
        disk_block_ids: &mut Vec<u32>,
        // 2026-09-25: `--high-speed-swap`: how many `disk_block_ids` entries each attention
        // layer has offloaded, indexed by the layer's `attn_layer_idx`.
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()>;

    /// 2026-09-25: Prefill `num_tokens` tokens starting at position `seq_len_start`.
    /// `hidden` and `residual` are `[num_tokens, hidden_size]` BF16. `kv_write_start` is
    /// how many of these tokens already have their KV in the cache; the Qwen3 attention
    /// cache-skip prefill writes KV only for the rest (`prefill/cache_skip.rs`). Default:
    /// `decode` once per token (`default_loops::prefill_default`), which ignores
    /// `kv_write_start`.
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
        default_loops::prefill_default(
            self,
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// 2026-09-25: Split SSM prefill, phase 1: norm, projections, gates, conv1d and L2
    /// norm for `num_tokens` tokens, then the GDN inputs (packed QKV, gate/beta, Z) are
    /// copied into `gdn_bufs` at `token_offset`. The recurrence runs later, in
    /// `prefill_gdn_full`. Default: the whole `prefill`.
    #[allow(clippy::too_many_arguments)]
    fn prefill_phase1(
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
        kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (gdn_bufs, token_offset);
        self.prefill(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            ctx,
            stream,
        )
    }

    /// 2026-09-25: Decode `num_tokens` consecutive tokens of one sequence, starting at
    /// position `seq_len`. `hidden` and `residual` are `[num_tokens, hidden_size]` BF16.
    /// Default: `decode` once per token (`default_loops::decode_batched_default`).
    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        default_loops::decode_batched_default(
            self,
            hidden,
            residual,
            num_tokens,
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

    /// 2026-09-25: Decode one token for each of `num_seqs` rows, each with its own entry in
    /// `seq_lens` and `block_tables`: one row per sequence in a batched decode, one per
    /// verify row in a batched verify (`verify_e.rs`). `hidden` and `residual` are
    /// `[num_seqs, hidden_size]` BF16. Default: `decode` once per row
    /// (`default_loops::decode_multi_seq_default`).
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        default_loops::decode_multi_seq_default(
            self,
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    /// 2026-09-25: Batched verify of `ks[i]` rows for each of `n_seqs` sequences in one
    /// sweep, the rows sequence-major and contiguous in `hidden`/`residual`. The verify
    /// (`verify_e.rs`) runs attention layers through `decode_multi_seq` and every other
    /// layer through this. `wy_tables` is this layer's slice of the staged WY pointer
    /// tables (layout at [`VERIFY_WY_TABLE_SEQS`]), or NULL when none were staged or the
    /// layer is not linear attention. The default returns an error.
    #[allow(clippy::too_many_arguments)]
    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _n_seqs: usize,
        _ks: &[usize],
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _wy_tables: DevicePtr,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("decode_verify_multi: unsupported for this layer type")
    }

    /// 2026-10-02: The batched verify for a layer that answers
    /// [`LayerCapabilities::decode_verify_multi_own_states`]: `ks[i]` rows for each of
    /// `ks.len()` sequences, sequence-major in `hidden`/`residual`, with each sequence's
    /// pre-verify length `seq_lens[i]` and page table `block_tables[i]` (the per-sequence
    /// state such a layer drives itself) and no WY tables. `ctx.attn_metadata` holds one row
    /// per verify row. The default returns an error.
    #[allow(clippy::too_many_arguments)]
    fn decode_verify_multi_seqs<'a, 'b: 'a>(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _ks: &[usize],
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        anyhow::bail!("decode_verify_multi_seqs: unsupported for this layer type")
    }

    /// 2026-09-25: Allocate this layer's per-sequence state. Sequence setup (model-engine
    /// `trait_impl/meta.rs`) calls it for every layer except a linear-attention layer that
    /// uses the SSM pool, which gets pool addresses instead.
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>>;

    /// 2026-09-25: Free the device memory this layer allocated for one sequence, in
    /// `alloc_state` or attached to the state later. `LayerState` holds bare `DevicePtr`s,
    /// so dropping it frees only the host struct.
    ///
    /// The teardown chokepoint `free_sequence_dispatch` calls it for every layer, logs an
    /// error and continues. An implementation must be idempotent, null what it frees,
    /// never free pool addresses, and decide by the state's type rather than rely on a
    /// filter at the call site. The default does nothing, which suits state that owns no
    /// device memory and pool-backed SSM state, released through the pool slot.
    fn release_state(&self, _state: &mut dyn LayerState, _gpu: &dyn GpuBackend) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod verify_wy_table_tests {
    use super::{
        VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS, VERIFY_WY_TABLE_STRIDE_BYTES,
        VERIFY_WY_TABLES_PER_LAYER,
    };

    /// 2026-09-25: `upload_verify_wy_tables` stages only k in
    /// `2..=VERIFY_WY_TABLES_PER_LAYER`; this pins k = 16 as admitted.
    #[test]
    fn tables_per_layer_fund_the_widest_verify_arm() {
        assert!(
            (2..=VERIFY_WY_TABLES_PER_LAYER).contains(&16usize),
            "VERIFY_WY_TABLES_PER_LAYER ({VERIFY_WY_TABLES_PER_LAYER}) must admit k=16"
        );
    }

    /// 2026-09-25: The table-form wyN launch (`trait_decode_batched_conv_gdn_multi.rs`) and
    /// the write-on-accept fold pass `VERIFY_WY_TABLE_SEQS` as the entry count between
    /// slabs, which reaches slab t only when slabs are back to back at that many 8-byte
    /// entries.
    #[test]
    fn slab_stride_matches_the_kernel_contract() {
        assert_eq!(VERIFY_WY_TABLE_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS * 8);
        assert_eq!(
            VERIFY_WY_LAYER_STRIDE_BYTES,
            VERIFY_WY_TABLES_PER_LAYER * VERIFY_WY_TABLE_STRIDE_BYTES
        );
    }

    /// 2026-09-25: A batch of 16 sequences fits one slab.
    #[test]
    fn seqs_capacity_covers_c16() {
        const { assert!(VERIFY_WY_TABLE_SEQS >= 16) };
    }
}
