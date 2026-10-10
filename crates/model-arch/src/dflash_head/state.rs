// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `DflashScratch`, a drafter head's device scratch, and
//! `DflashProposerState`, its per-sequence state.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Device scratch for the γ-block forward, allocated once per head
/// in `from_weights.rs`. `nb` below is the head's `max_batch`.
pub struct DflashScratch {
    pub stream_buf: DevicePtr,
    pub norm_buf: DevicePtr,
    pub q_buf: DevicePtr,
    pub k_buf: DevicePtr,
    pub v_buf: DevicePtr,
    pub attn_out: DevicePtr,
    pub mlp_intermediate: DevicePtr,
    pub mlp_up: DevicePtr,
    pub stream_acc: DevicePtr,
    /// 2026-09-25: `[ctx_window, hidden]` BF16: ctx rows after `fc` and
    /// `hidden_norm`.
    pub fc_proj: DevicePtr,
    /// 2026-09-25: `[ctx_window, L * 2 * kv_dim]` BF16: the ctx precompute's KV
    /// GEMM output.
    pub fused_kv_out: DevicePtr,
    /// 2026-09-25: `i64[ctx_window]` paged-cache slot indices for
    /// `reshape_and_cache`.
    pub slot_mapping_dev: DevicePtr,
    /// 2026-09-25: `[PRECOMPUTE_BATCH_ROWS, L_t * h_t]` BF16 staging for the
    /// batched ctx precompute: the uncommitted ctx rows of several sequences,
    /// contiguous.
    pub precompute_in: DevicePtr,
    /// 2026-09-25: `nb` triples `[u32 kv_len, u32 q_offset, u32 q_rope_pos]`,
    /// 12 bytes each, one per band, read by the indirect paged-attention kernel
    /// at entry. The host writes them before the captured region, so a captured
    /// graph sees a fixed pointer.
    pub option_b_indirect_args_dev: DevicePtr,
    /// 2026-09-25: Pinned host buffer, `nb * gamma * 4` bytes, for the
    /// draft-token D2H copy; allocated once in `from_weights.rs`. `AtomicPtr`
    /// keeps `DflashScratch` `Send + Sync`. Nothing stores to it after
    /// construction, so `Relaxed` loads are enough.
    pub draft_tokens_host_pinned: std::sync::atomic::AtomicPtr<u8>,
    /// 2026-09-25: Event recorded after the draft-token D2H copy; the host waits
    /// on it before reading the pinned buffer. Created once in `from_weights.rs`.
    pub draft_tokens_event: u64,
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// 2026-09-25: i32 RoPE positions: the `eff_ctx` ctx rows, then each
    /// sequence's block rows from that sequence's position.
    pub position_ids: DevicePtr,
    /// 2026-09-25: DSpark Markov scratch: `[1, markov_rank]` BF16 latent for the
    /// prev-token gather (`markov_w1[prev]`). `DevicePtr(0)` when the
    /// drafter has no Markov head.
    pub markov_embed: DevicePtr,
    /// 2026-09-25: DSpark Markov scratch: `[vocab]` BF16 bias
    /// (`markov_w2 @ markov_embed`), residual-added onto one logits row
    /// per sequential step. `DevicePtr(0)` when no Markov head.
    pub markov_bias: DevicePtr,
    /// 2026-09-25: DSpark confidence scratch: `[γ]` BF16 per-row acceptance
    /// logits, read back on the host after the draft-token D2H to pick how many
    /// drafts to keep. `DevicePtr(0)` when the drafter has no confidence head.
    pub conf_out: DevicePtr,

    // 2026-09-25: DFlash2 scratch, `DevicePtr(0)` when the checkpoint has no
    // selector.
    /// 2026-09-25: `[nb * γ, 2 * conv_kernel_size * (hidden / conv_group_size)]`
    /// BF16 dynamic conv kernels, reused by each conv site in turn.
    pub conv_dyn: DevicePtr,
    /// 2026-09-25: `[nb * γ, hidden]` BF16 staging for the convolved hidden.
    pub conv_tmp: DevicePtr,
    /// 2026-09-25: `[nb * γ, 16]` f32: the selector's top-16 logits per row.
    pub sel_vals: DevicePtr,
    /// 2026-09-25: `[nb * γ, 16]` u32: the selector's top-16 token ids per row.
    pub sel_idx: DevicePtr,
    /// 2026-09-25: `[nb * γ, selector_rank]` BF16: the selector's hidden
    /// projection.
    pub sel_hproj: DevicePtr,
}

/// 2026-09-25: Per-sequence DFlash drafter state. One block table serves every
/// drafter layer: the paged cache has a pool per layer, addressed by the same
/// slots.
pub struct DflashProposerState {
    /// 2026-09-25: Host copy of the drafter block table.
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    /// 2026-09-25: Drafts the last propose returned; `after_verify` and
    /// `free_state` reset it to 0.
    pub last_num_drafted: usize,
    pub prefill_done: bool,
    /// 2026-09-25: `max_ctx_len` rows of `ctx_slot_bytes`: captured target
    /// hiddens, one row per ctx position, the first `ctx_len` valid. Model-engine
    /// appends rows and slides the window when it is full; `propose_drafts`
    /// appends while there is room.
    pub ctx_hidden_acc: DevicePtr,
    /// 2026-09-25: Valid rows in `ctx_hidden_acc`, at most `max_ctx_len`.
    pub ctx_len: usize,
    pub last_num_accepted: usize,
    /// 2026-09-25: Model-engine has already appended the latest capture; the
    /// next `propose_drafts` skips its own append and clears the flag.
    pub skip_next_decode_append: bool,
    /// 2026-09-25: Rows `ctx_hidden_acc` holds (the `window` of
    /// `alloc_state_windowed`).
    pub max_ctx_len: usize,
    /// 2026-09-25: Bytes per `ctx_hidden_acc` row:
    /// `target_layer_ids.len() * target_hidden_size * 2`.
    pub ctx_slot_bytes: usize,

    /// 2026-09-25: Device copy of `block_table`, made once when the first
    /// propose allocates the table; `None` before that and after `free_state`.
    pub block_table_dev: Option<DevicePtr>,
    /// 2026-09-25: Ctx rows of the paged cache the γ block attends to;
    /// `propose_drafts` sets it to `ctx_len`.
    pub ctx_count_drafter: usize,
    /// 2026-09-25: Slots the allocated block table covers
    /// (`block_table.len() * BLOCK_SIZE`).
    pub max_ctx_count_drafter: usize,
    /// 2026-09-25: Ctx rows `[0, ctx_committed)` whose K/V is already in the
    /// paged cache; the next precompute starts here. Model-engine's window
    /// slides reset it to 0.
    pub ctx_committed: usize,
    /// 2026-09-25: One RoPE position per ctx row (`len() == ctx_len`), stamped
    /// when the row is appended and kept when the window slides. Model-engine's
    /// `update_dflash_ctx_len_after_prefill` seeds prefill row i with i.
    pub ctx_positions: Vec<i32>,
    /// 2026-10-09: Two-rank drafter, rank 0 only: the position of the newest ctx row rank 1
    /// holds (`tp/leader.rs`). Prefill capture sets it; `None` makes the next propose
    /// resend every row.
    pub tp_synced_pos: Option<i32>,
}

impl ProposerState for DflashProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
