// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Types every [`TransformerLayer`] shares: per-sequence layer state, the
//! per-pass `ForwardContext`, and the attention and GDN metadata a pass hands its layers.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use std::any::Any;

use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

mod transformer_layer;
pub use transformer_layer::{
    GdnCarryBinding, LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill,
    LayerWeightSetup, LayerWriteOnAccept, MTP_CATCHUP_MAX, TransformerLayer,
    VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS, VERIFY_WY_TABLE_STRIDE_BYTES,
    VERIFY_WY_TABLES_PER_LAYER,
};

/// 2026-09-25: One layer's per-sequence state, reached through `as_any` downcasts.
/// Implementors here: [`EmptyLayerState`], [`AttnLayerState`] and [`SsmLayerState`].
pub trait LayerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// 2026-10-03: Back this state's per-position storage for positions `[0, end)` before a
    /// step writes them (lazily mapped state, `METRALE_DSA_INDEXER_LAZY`). The model calls it
    /// where it grows the KV block table (`block_mgmt::ensure_blocks_through_*`), outside any
    /// graph capture and before any replay. Positions past the state's own capacity are left
    /// to its write-path checks. Default: nothing to map.
    fn map_rows_through(&self, end: usize) -> anyhow::Result<()> {
        let _ = end;
        Ok(())
    }
}

/// 2026-09-25: State of a layer that keeps nothing per sequence.
pub struct EmptyLayerState;

/// 2026-09-25: Per-sequence state of a Qwen3 attention layer. The KV cache lives
/// outside it; the one field is the QSA indexer carry, created on first use and only
/// on a layer that has a QSA indexer.
#[derive(Default)]
pub struct AttnLayerState {
    pub qsa: Option<crate::layers::qsa::QsaSeqState>,
}

impl LayerState for AttnLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl LayerState for EmptyLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// 2026-09-25: Per-sequence state of a recurrent (SSM/GDN) layer: the recurrent h state
/// and the conv1d window, plus the copies speculative decode rolls back to.
pub struct SsmLayerState {
    /// 2026-09-25: Recurrent state; FP32, or FP16 when `h_is_f16`.
    pub h_state: DevicePtr,
    /// 2026-09-25: Conv1d window, FP32 `[conv_dim, d_conv]`.
    pub conv_state: DevicePtr,
    /// 2026-09-25: Copy of `h_state` restored when a verify accepts no draft. A pool
    /// address, or allocated on first use by the checkpoint save
    /// (`checkpoint_ssm_states_dispatch`, `start_checkpoint_async_dispatch`).
    pub h_state_checkpoint: Option<DevicePtr>,
    /// 2026-09-25: Copy of `conv_state`, restored with `h_state_checkpoint`.
    pub conv_state_checkpoint: Option<DevicePtr>,
    /// 2026-09-25: Per-token h states of a verify: element i holds the state after
    /// verify token i. A rollback to `n` accepted tokens copies element `n - 1`.
    pub h_state_intermediates: Vec<DevicePtr>,
    /// 2026-09-25: Per-token conv states of a verify, indexed like
    /// `h_state_intermediates`.
    pub conv_state_intermediates: Vec<DevicePtr>,
    /// 2026-09-25: Storage format of `h_state`: `false` = FP32, `true` = FP16.
    ///
    /// A sequence on an f16-sized pool (`h_prefill_stage` is `Some`) starts `true`: the
    /// slot holds only 2 bytes per element. On an FP32-sized pool it starts `false`, and
    /// `TransformerModel::ssm_h_to_f16_dispatch` converts the state and sets it `true` on
    /// the first decode step when FP16 h-state is enabled.
    pub h_is_f16: bool,
    /// 2026-09-25: FP32 staging blob for this sequence's slot when the pool is f16-sized
    /// (`--ssm-h-dtype f16-pool`). GDN prefill widens `h_state` into it before its FP32
    /// kernels and narrows it back after (`qwen3_ssm::ssm_h_fp16`). `None` means the slot
    /// is FP32 and prefill writes `h_state` in place. One blob per slot, shared by every
    /// layer of the sequence.
    pub h_prefill_stage: Option<DevicePtr>,
    /// 2026-09-25: PLE per-sequence carry (token history and dilated-conv state), created
    /// on first use, only on a layer that hosts a `PleLayer`.
    pub ple: Option<crate::layers::ple::PleSeqState>,
}

impl LayerState for SsmLayerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// 2026-09-25: Device addresses of one pass's attention metadata, uploaded by the model
/// before its layer loop. On a decode pass each row is one sequence, with one entry (or one
/// block-table row) per row in each array. A prefill pass has `num_seqs == 1` and one
/// `positions` and `slot` entry per token.
#[derive(Clone, Copy)]
pub struct AttnMetadataDev {
    /// 2026-09-25: Positions, u32, one per token row. Under MRoPE this is the temporal (T)
    /// stream.
    pub positions: DevicePtr,
    /// 2026-09-25: MRoPE height (H) positions. The same pointer as `positions` except on a
    /// prefill pass that uses MRoPE (`prefill_b/forward_layers.rs`).
    pub positions_h: DevicePtr,
    /// 2026-09-25: MRoPE width (W) positions, set like `positions_h`.
    pub positions_w: DevicePtr,
    /// 2026-09-25: KV write slot of each token row, i64.
    pub slot: DevicePtr,
    /// 2026-09-25: Sequence lengths, one 32-bit entry per row. On decode each is the length
    /// including the token being decoded (`seq_len + 1`).
    pub seq_len: DevicePtr,
    /// 2026-09-25: Block tables, 32-bit entries, row-major.
    pub block_table: DevicePtr,
    /// 2026-09-25: Entries per row of `block_table`.
    pub max_blocks_per_seq: u32,
    /// 2026-09-25: Rows in the batch, padding rows included; 1 for single-sequence decode.
    pub num_seqs: u32,
    /// 2026-09-25: Per-row LoRA adapter slot, i32 (one per token on prefill), at a fixed
    /// address, so the batched bgmv can run inside a captured decode graph. A negative slot
    /// applies no delta (`lora_bgmv.cu`); padding rows are `-1`. `DevicePtr(0)` when no LoRA
    /// weights are loaded, or on a single-request pass whose request uses the active adapter
    /// (`upload_seq_slot_uniform`); the apply sites then skip the bgmv.
    pub seq_slot: DevicePtr,
    /// 2026-09-25: Per-row adapter map for the batched-decode MoE expert fold, `[num_seqs]`
    /// i32 in the fixed-address `TransformerModel::moe_row_adapter_buf`, built by
    /// [`crate::lora::build_moe_row_adapter_decode`]. `< 0` means no fold on that row;
    /// `>= 0` folds the active adapter's per-expert delta. `DevicePtr(0)` when no LoRA
    /// weights are loaded and on every path that does not upload it. Unlike `seq_slot`,
    /// no row resolves a negative slot to the active adapter.
    pub moe_row_adapter: DevicePtr,
}

impl AttnMetadataDev {
    /// 2026-10-01: This metadata seen from decode row `base`: every per-row array advanced by
    /// `base` rows at its own element width (positions u32, slot i64, seq_len i32, block table
    /// `max_blocks_per_seq` i32 entries, seq_slot and moe_row_adapter i32), `num_seqs` reduced to
    /// the rows left from `base`, and every null pointer left null. A layer that serves the rows
    /// of a batched decode one at a time hands row `i` `row_view(i)`; with the unadvanced block
    /// it would attend with row 0's positions, KV slot, length and page table. Ported from
    /// rsafier's Atlas `AttnMetadataDev::row_view` (Atlas e69446eee).
    #[must_use]
    pub fn row_view(&self, base: usize) -> Self {
        let off = |p: DevicePtr, stride: usize| {
            if p.0 == 0 { p } else { p.offset(base * stride) }
        };
        Self {
            positions: off(self.positions, 4),
            positions_h: off(self.positions_h, 4),
            positions_w: off(self.positions_w, 4),
            slot: off(self.slot, 8),
            seq_len: off(self.seq_len, 4),
            block_table: off(self.block_table, self.max_blocks_per_seq as usize * 4),
            max_blocks_per_seq: self.max_blocks_per_seq,
            num_seqs: self.num_seqs.saturating_sub(base as u32),
            seq_slot: off(self.seq_slot, 4),
            moe_row_adapter: off(self.moe_row_adapter, 4),
        }
    }
}

/// 2026-09-25: Device metadata for one kernel-batched prefill chunk over several streams,
/// built by `stage_batched_attn_metadata` (model-engine `prefill_b/stage_batched.rs`) and
/// passed to `prefill_attn_batched_layer` and `prefill_ssm_batched_layer`. Positions and
/// KV slots are stacked in `cu_seqlens` order; block tables and sequence lengths are
/// per-stream pointer arrays.
pub struct BatchedAttnMetadata {
    /// 2026-09-25: Stacked positions, `[total_tokens]` u32.
    pub positions_stacked: DevicePtr,
    /// 2026-09-25: Stacked MRoPE H positions; the same pointer as `positions_stacked`
    /// without MRoPE.
    pub positions_h_stacked: DevicePtr,
    /// 2026-09-25: Stacked MRoPE W positions; the same pointer as `positions_stacked`
    /// without MRoPE.
    pub positions_w_stacked: DevicePtr,
    /// 2026-09-25: Stacked KV write slots, `[total_tokens]` i64.
    pub slot_stacked: DevicePtr,
    /// 2026-09-25: `[batch_size]` device pointers, each to one stream's block table.
    pub block_table_ptrs: DevicePtr,
    /// 2026-09-25: `[batch_size]` device pointers, each to one stream's seq_len.
    pub seq_len_ptrs: DevicePtr,
    // 2026-09-25: The per-stream `h_state` pointers are per layer, so they are not
    // here: `prefill_ssm_batched_layer` stages them for each SSM layer after this block.
    /// 2026-09-25: Number of streams.
    pub batch_size: u32,
    /// 2026-09-25: The largest per-stream token count. Per-stream counts are in
    /// `cu_seqlens`.
    pub chunk_len: u32,
    /// 2026-09-25: Tokens stacked across all streams (`cu_seqlens_host[batch_size]`).
    pub total_tokens: u32,
    /// 2026-09-25: `[batch_size + 1]` i32 prefix sum of per-stream token counts, on the
    /// device.
    pub cu_seqlens: DevicePtr,
    /// 2026-09-25: Host copy of `cu_seqlens`; the model slices per-stream token ranges
    /// with it.
    pub cu_seqlens_host: Vec<i32>,
    /// 2026-09-25: Per-stream KV length, `[batch_size]` i32 on the device: the stream's
    /// chunk start plus its token count. The batched attention needs it per stream; one
    /// value at the maximum would give a short stream the longest stream's causal bound
    /// and block count.
    pub kv_lens: DevicePtr,
    /// 2026-09-25: Host copy of `kv_lens`.
    pub kv_lens_host: Vec<i32>,
    /// 2026-09-25: The largest block count of any stream.
    pub max_blocks_per_seq: u32,
    /// 2026-09-25: Bytes this block occupies in scratch, from the offset it was staged at
    /// through the end of `kv_lens`. The caller advances its scratch cursor by it before
    /// placing the per-layer `h_state` pointer array (`prefill_b/batch_kernel.rs`), so an
    /// undercount would put that array over live metadata.
    pub staged_bytes: usize,
}

/// 2026-09-25: Whole-prompt GDN buffers for the split SSM prefill: `prefill_phase1`
/// writes the inputs, `prefill_gdn_full` runs the recurrence into `output`, and
/// `prefill_phase3` reads `output` and `z`.
///
/// Each token's QKV is packed as `conv_dim` BF16 elements, `[Q(key_dim) | K(key_dim) |
/// V(value_dim)]`, and the GDN kernels read Q, K and V with a stride of `conv_dim`.
pub struct GdnPrefillBuffers {
    /// 2026-09-25: Packed Q/K/V, `[total_len, conv_dim]` BF16.
    pub qkv: DevicePtr,
    /// 2026-09-25: Gate and beta, `[total_len, 2 * num_v_heads]` FP32: per token
    /// `[gate(nv) | beta(nv)]`.
    pub gate_beta: DevicePtr,
    /// 2026-09-25: GDN recurrence output, `[total_len, value_dim]` BF16.
    pub output: DevicePtr,
    /// 2026-09-25: Z gate for the gated RMSNorm, `[total_len, value_dim]` BF16.
    pub z: DevicePtr,
    /// 2026-09-25: Token rows the buffers hold for this call.
    pub total_len: usize,
}

/// 2026-09-25: What one forward pass hands every layer: the GPU, buffers, config, the
/// model's switches and the pass's metadata.
///
/// 2026-10-01: `Copy` (every field is a reference, a scalar or a `Copy` value), so a layer that
/// serves a batched decode row by row can build `ForwardContext { attn_metadata, ..*ctx }`.
#[derive(Clone, Copy)]
pub struct ForwardContext<'a> {
    pub buffers: &'a BufferArena,
    /// 2026-09-25: First mHC highway row of this pass. The prefill chunk of a fused
    /// decode+prefill step uses `padded_n` (`decode_b.rs`), so its highway rows sit above
    /// the decode rows as in hidden/residual. Every other constructor passes 0 or forwards
    /// the value of the context it wraps.
    pub hc_row_offset: usize,
    pub gpu: &'a dyn GpuBackend,
    pub config: &'a ModelConfig,
    /// 2026-09-25: This model's GEMM selection per projection (`layers::ops::GemmDispatch`).
    pub dispatch: &'a crate::layers::ops::GemmDispatch,
    /// 2026-09-25: This model's memo of re-encoded weight copies
    /// (`layers::ops::DerivedWeights`).
    pub derived: &'a crate::layers::ops::DerivedWeights,
    /// 2026-09-25: This model's kernel-path switches other than GEMM selection
    /// (`layers::ops::ModelLevers`).
    pub levers: &'a crate::layers::ops::ModelLevers,
    /// 2026-09-25: This model's diagnostic counters and one-shot latches
    /// (`layers::ops::ModelStats`). They belong to the model, so none spans a model swap.
    pub stats: &'a crate::layers::ops::ModelStats,
    /// 2026-09-25: This pass's attention metadata, or `None`. The kernel-batched prefill
    /// passes `None` and hands its layers a `BatchedAttnMetadata` instead.
    pub attn_metadata: Option<AttnMetadataDev>,
    /// 2026-09-25: Profiling: the layers that read it synchronize after each timed step and
    /// log its time.
    pub profile: bool,
    /// 2026-09-25: Collective backend for cross-rank reductions, such as the MoE
    /// expert-parallel all-reduce; `None` when the pass has none.
    pub comm: Option<&'a dyn metrale_comm::CommBackend>,
    /// 2026-09-25: True when the pass runs under CUDA graphs (`use_graphs` in `decode_a.rs`).
    /// The MoE expert-parallel all-reduce then uses `all_reduce`, and `all_reduce_async`
    /// otherwise (`layers/moe/forward.rs`).
    pub graph_capture: bool,
    /// 2026-09-25: True on decode steps (one token per sequence: `decode_a.rs`,
    /// `decode_a2.rs`), whose `attn_metadata` holds one row per sequence.
    ///
    /// `prefill_default` runs a layer that has no `prefill` of its own by calling its
    /// `decode` once per token with the prefill context, whose `positions` and `slot` are
    /// per-token arrays and whose `block_table` and `seq_len` can be NULL. A layer that
    /// reads them as decode metadata checks this flag, not `attn_metadata.is_some()`.
    pub decode_step: bool,
    /// 2026-09-25: True when this prefill pass continues from a restored SSM snapshot (a
    /// warm prefix-cache hit). The Qwen3 SSM prefill then skips the FLA chunked GDN
    /// kernels (`trait_prefill_recur.rs`, `trait_prefill_gdn.rs`).
    pub gdn_exact_replay: bool,
    /// 2026-09-25: Asks the batched GDN verify for write-on-accept: a layer that engages it
    /// runs `gated_delta_rule_wy4_woa`, and the caller folds the accepted rows through
    /// `Model::gdn_fold_accepted` after the verdict. True only for a batched verify that
    /// requested it (`VerifyBatchedOpts::write_on_accept`, set by the DFlash batch step) and
    /// had the WY tables staged and the stash bound (`verify_e.rs`).
    pub gdn_write_on_accept: bool,
    /// 2026-09-25: Device `[num_tokens]` u32 ids of this pass's tokens, in row order, at a
    /// stable address; single-token decode uploads the id before any graph replay.
    /// Hash-routed MoE layers (`tid2eid`) read it, and PLE does when `host_token_ids` is
    /// `None`. `None` on passes that stage no ids.
    pub token_ids: Option<DevicePtr>,
    /// 2026-09-25: Host copy of this pass's token ids when the caller has them: a decode
    /// step whose tokens are on the host, and the prefill chunk of a fused decode+prefill
    /// step (`decode_b.rs`); `None` on other passes. PLE hashes n-gram ids on the host and
    /// prefers this slice; without it, PLE reads `token_ids` back from the device, which
    /// it refuses during graph capture (`ple/layer.rs`).
    pub host_token_ids: Option<&'a [u32]>,
    /// 2026-09-25: The request slot's LoRA pairs, indexed by global layer
    /// (`len == num_hidden_layers`). `Some` only on a prefill pass whose request routes to
    /// a non-active slot (`TransformerModel::routed_slot_layers`). The K/V/O prefill apply
    /// sites then apply the request slot's pair (`lora::select_routed_pair`) instead of
    /// the per-row bgmv.
    pub routed_lora_layers: Option<&'a [Option<crate::lora::LoraLayerWeights>]>,
    /// 2026-09-25: Mid-chunk SSM tail capture for this prefill pass (model-engine
    /// `prefill_b/midchunk_capture.rs`); `None` when the pass plans none. SSM layers split
    /// their conv and recurrence kernels at `cap_local` and copy the state at that point
    /// into the reserved snapshot slot.
    pub midchunk_capture: Option<MidchunkCapture<'a>>,
    /// 2026-09-25: MoE-LoRA fold decision for this pass. `TransformerModel` passes
    /// `moe_lora_route` for a request's prefill, `decode_moe_route` on decode, verify and
    /// draft passes, and `Refuse` on a kernel-batched multi-stream prefill. Read by the MoE
    /// LoRA fold hooks (`layers/moe/lora.rs`).
    pub moe_lora_route: MoeLoraRoute,
}

/// 2026-09-25: Per-pass descriptor for mid-chunk SSM tail capture: the reserved snapshot
/// slot's per-SSM-layer destinations and the split point in pass-local token coordinates.
///
/// Each SSM layer's prefill takes one ordinal from `ssm_layer_counter`, in model order, and
/// uses it to index `h_dsts`/`conv_dsts`.
///
/// 2026-10-01: `Copy`, because `ForwardContext` carries it and is `Copy`.
#[derive(Clone, Copy)]
pub struct MidchunkCapture<'a> {
    /// 2026-09-25: Capture the state after this many tokens of the pass
    /// (`tb - proc_start`).
    pub cap_local: usize,
    /// 2026-09-25: Per-SSM-layer h_state destination in the reserved slot.
    pub h_dsts: &'a [DevicePtr],
    /// 2026-09-25: Per-SSM-layer conv_state destination in the reserved slot.
    pub conv_dsts: &'a [DevicePtr],
    /// 2026-09-25: Bytes per layer of h_state.
    pub h_bytes: usize,
    /// 2026-09-25: Bytes per layer of conv_state.
    pub conv_bytes: usize,
    /// 2026-09-25: Per-pass SSM-layer ordinal counter, starting at 0.
    pub ssm_layer_counter: &'a std::sync::atomic::AtomicUsize,
    /// 2026-09-25: Split point of a second capture one KV block earlier, at
    /// `tb - block_size`; `Some` only when the pass covers that point and a second snapshot
    /// slot was reserved.
    pub cap_local_early: Option<usize>,
    /// 2026-09-25: Per-SSM-layer h_state destination in the `tb - block_size` slot.
    pub h_dsts_early: &'a [DevicePtr],
    /// 2026-09-25: Per-SSM-layer conv_state destination in the `tb - block_size` slot.
    pub conv_dsts_early: &'a [DevicePtr],
    /// 2026-10-03: Absolute token position of the pass's row 0 (`effective_seq_len_start`),
    /// so a layer that runs its rows in several calls can place `cap_local` from the `seq_len`
    /// each call is handed (`Self::inpass_split`).
    pub seq_pos_start: usize,
    /// 2026-10-03: In-pass capture by pool address (`METRALE_GLM_SSM_INPASS_CAPTURE=1`,
    /// model-engine `prefill_b/inpass_capture.rs`): per SSM ordinal, the sequence's live
    /// h_state pool address. A layer finds its ordinal as the index of its own
    /// `SsmLayerState::h_state` here instead of taking one from `ssm_layer_counter`. Empty for
    /// the tail mid-chunk plan, whose layers ignore the three fields below.
    pub live_h: &'a [DevicePtr],
    /// 2026-10-03: Incremented once by each SSM layer that issued its in-pass capture copies;
    /// the engine registers the snapshot only when every SSM layer did.
    pub captured: &'a std::sync::atomic::AtomicUsize,
}

/// 2026-10-03: Where one call of an SSM layer's prefill captures an in-pass snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InpassSplit {
    /// 2026-10-03: Rows of this call to run before the capture, `1..=k`: the captured state is
    /// the state after row `row - 1` of the call.
    pub row: usize,
    /// 2026-10-03: This layer's h_state destination in the reserved snapshot slot.
    pub h_dst: DevicePtr,
    /// 2026-10-03: This layer's conv_state destination in the reserved snapshot slot.
    pub conv_dst: DevicePtr,
    /// 2026-10-03: Bytes of h_state to copy (the snapshot pool's `h_bytes`).
    pub h_bytes: usize,
    /// 2026-10-03: Bytes of conv_state to copy (the snapshot pool's `conv_bytes`).
    pub conv_bytes: usize,
}

impl MidchunkCapture<'_> {
    /// 2026-10-03: For an in-pass plan (`live_h` non-empty), the split of one prefill call over
    /// `k` rows starting at absolute position `seq_len`, for the layer whose live h_state is
    /// `h_state`: `Some` exactly when the capture point `seq_pos_start + cap_local` lies in
    /// `(seq_len, seq_len + k]` and `h_state` is one of `live_h`. Over calls that tile a pass,
    /// at most one returns `Some` per layer.
    pub fn inpass_split(
        &self,
        seq_len: usize,
        k: usize,
        h_state: DevicePtr,
    ) -> Option<InpassSplit> {
        if self.live_h.is_empty() {
            return None;
        }
        let row = inpass_split_row(self.seq_pos_start + self.cap_local, seq_len, k)?;
        let ord = self.live_h.iter().position(|p| *p == h_state)?;
        Some(InpassSplit {
            row,
            h_dst: *self.h_dsts.get(ord)?,
            conv_dst: *self.conv_dsts.get(ord)?,
            h_bytes: self.h_bytes,
            conv_bytes: self.conv_bytes,
        })
    }
}

/// 2026-10-03: Rows before the capture point `cut` (absolute) in a call over positions
/// `[seq_len, seq_len + k)`: `Some(cut - seq_len)` when `seq_len < cut <= seq_len + k`, else
/// `None`. The state after `cut` tokens is the state after row `cut - seq_len - 1` of the call.
pub fn inpass_split_row(cut: usize, seq_len: usize, k: usize) -> Option<usize> {
    (seq_len < cut && cut <= seq_len + k).then(|| cut - seq_len)
}

/// 2026-09-25: MoE-LoRA fold decision for one forward pass. One MoE adapter is active at
/// a time; `MoeLayer::moe_route_gate` (`layers/moe/lora.rs`) folds on `Fold`, skips on
/// `Skip` and returns an error on `Refuse`. `lora::resolve_moe_lora_route` maps a
/// request's `adapter_slot` to a route.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoeLoraRoute {
    /// 2026-09-25: The request uses the active MoE adapter, or no MoE delta is installed.
    /// The default.
    #[default]
    Fold,
    /// 2026-09-25: No fold. `resolve_moe_lora_route` returns it for a base request
    /// (`adapter_slot < 0`).
    Skip,
    /// 2026-09-25: The request routes to an adapter other than the active one, or the pass
    /// is a kernel-batched multi-stream prefill; the fold returns an error instead of
    /// applying the active adapter to rows it does not own.
    Refuse,
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "layer/release_contract_tests.rs"]
mod release_contract_tests;
