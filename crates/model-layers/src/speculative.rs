// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The draft-proposer interface for speculative decoding
//! ([`DraftProposer`], [`ProposerState`]), and the MTP knobs, ladder, tree
//! shape and verify keys in its submodules.
//!
//! Owner: model-layers (speculative).
//! Invariants: none beyond the types.

pub mod glm_dflash;
pub mod ladder;
pub mod tree_shape;
pub mod verify_key;

pub use ladder::{mtp_ladder_disabled, mtp_ladder_drafts, mtp_max_seqs};
mod knobs;
pub use knobs::{
    EP_CMD_MTP_PROPOSE, draft_conf_tau, hidden_fingerprint, mtp_accept_debug, mtp_catchup_enabled,
    mtp_ep_propose_enabled, mtp_multi_seq_mode, mtp_refeed_accepted_enabled, mtp_refeed_debug,
    mtp_refeed_shift, parse_draft_conf_tau, shadow_topk,
};

use std::any::Any;

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::ForwardContext;

/// 2026-09-25: Per-sequence state of a [`DraftProposer`], reached through
/// `as_any` downcasts like `LayerState`.
pub trait ProposerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// 2026-10-03: [`crate::layer::LayerState::map_rows_through`] for a proposer: back the
    /// rows the drafter will write for target positions up to `end` (an implementation adds
    /// its own look-ahead for the rows it drafts past the target). Default: nothing to map.
    fn map_rows_through(&self, end: usize) -> Result<()> {
        let _ = end;
        Ok(())
    }
}

/// 2026-09-25: A draft-token proposer. The model calls `propose` for draft
/// tokens, verifies them with the target, then calls `after_verify` with the
/// accepted count so the proposer can trim its state.
pub trait DraftProposer: Send + Sync {
    /// 2026-09-25: Allocate per-sequence proposer state.
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>>;

    /// 2026-09-25: [`Self::alloc_state`] given the sequence's token budget
    /// (prompt length plus `max_tokens`), so state that scales with context
    /// can be sized to the request instead of `--max-seq-len`. `usize::MAX`
    /// means unknown. The default ignores the budget.
    fn alloc_state_for(
        &self,
        gpu: &dyn GpuBackend,
        budget_tokens: usize,
    ) -> Result<Box<dyn ProposerState>> {
        let _ = budget_tokens;
        self.alloc_state(gpu)
    }

    /// 2026-09-25: The block size γ of a block-diffusion (DFlash) drafter;
    /// `None` for other proposers. With `--dflash`, serve sets the draft count
    /// to γ - 1, at least 1.
    fn block_gamma(&self) -> Option<usize> {
        None
    }

    /// 2026-09-25: Chain confidence of the latest `propose`, when the proposer
    /// computes it; `None` otherwise. The MTP head computes it only while
    /// `draft_conf_tau` is above 0.
    fn last_confidence(&self) -> Option<f32> {
        None
    }

    /// 2026-09-25: Rows of `mtp_prefill_hidden` this proposer can use, given
    /// the served `--max-seq-len`; the model sizes that `[rows, hidden]` BF16
    /// buffer from it. The default is `max_seq_len`.
    ///
    /// Return less only when the proposer can never be given a later
    /// position. A smaller cap corrupts nothing: a prompt longer than the
    /// buffer is not fully captured, so its drafter prefill is skipped, which
    /// costs acceptance.
    fn prefill_hidden_rows(&self, max_seq_len: usize) -> usize {
        max_seq_len
    }

    /// 2026-09-25: Whether this proposer's block is sharded across ranks, so
    /// its forward is given the communicator. The default is false; the
    /// GLM-5.3 MTP head returns [`mtp_ep_propose_enabled`]. When it is true on
    /// a multi-rank serve, the model sends the worker [`EP_CMD_MTP_PROPOSE`]
    /// before each propose, so both ranks run the same forward and
    /// collectives.
    fn needs_comm(&self) -> bool {
        false
    }

    /// 2026-09-25: Whether this proposer's context prefill uses the shared
    /// forward scratch (`ctx.buffers`). If so, the model skips the prefill at
    /// the end of the target's prefill, while the target still owns those
    /// buffers, and it runs at the first `propose` instead. Measured
    /// 2026-08-29 on GLM-5.3 (2x GB10): running it at the end of prefill
    /// changed the target's output on 2 of 6 probes.
    fn prefill_uses_shared_buffers(&self) -> bool {
        false
    }

    /// 2026-09-25: Drafter KV rows held in `state`; 0 when unknown.
    fn drafter_rows(&self, _state: &mut dyn ProposerState) -> usize {
        0
    }

    /// 2026-09-25: Sequence-space pair key of the newest drafter row; `None`
    /// when untracked. Rows are compacted, so the row count does not give the
    /// sequence position.
    fn last_pair_key(&self, _state: &mut dyn ProposerState) -> Option<usize> {
        None
    }

    /// 2026-09-25: Move this sequence's drafter KV blocks out of its state, so
    /// `free_state` does not release them and the model can carry them to the
    /// next turn. Returns `(blocks, rows, last_pair_key)`, or `None` when
    /// unsupported or empty. Afterwards the state must behave as freshly
    /// allocated.
    fn take_drafter_kv(
        &self,
        _state: &mut dyn ProposerState,
    ) -> Option<(Vec<u32>, usize, Option<usize>)> {
        None
    }

    /// 2026-09-25: Install carried blocks into a fresh state, the inverse of
    /// [`Self::take_drafter_kv`]. Returns false when unsupported; the caller
    /// then still owns the blocks and must free them.
    fn install_drafter_kv(
        &self,
        _state: &mut dyn ProposerState,
        _blocks: Vec<u32>,
        _rows: usize,
        _last_pair_key: Option<usize>,
    ) -> bool {
        false
    }

    /// 2026-09-25: Release drafter KV blocks that no proposer state owns, such
    /// as a carried entry being replaced or dropped.
    fn free_drafter_kv(&self, _blocks: &[u32]) {}

    /// 2026-09-25: Append drafter rows at KV slots `row_base ..` with RoPE
    /// positions `pos_base ..` from `(tokens, hiddens)` pairs. Returns the
    /// rows written; the default writes none.
    #[allow(clippy::too_many_arguments)]
    fn catchup_drafter(
        &self,
        _tokens: &[u32],
        _hiddens: DevicePtr,
        _row_base: usize,
        _pos_base: usize,
        _state: &mut dyn ProposerState,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<usize> {
        Ok(0)
    }

    /// 2026-09-25: Propose up to `num_drafts` tokens autoregressively.
    ///
    /// - `last_token`: the last verified token.
    /// - `target_hidden`: the target's `[1, hidden_size]` BF16 hidden for it,
    ///   saved before the final norm.
    /// - `position`: the sequence position, for RoPE.
    /// - `grammar_bitmask`: when `Some`, one bit per token id in i32 words;
    ///   drafts are limited to tokens whose bit is set.
    /// - `target_hidden_stack`: the model's DFlash hidden capture, when it has
    ///   one; the MTP head ignores it.
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>>;

    /// 2026-09-25: Propose `num_drafts` tokens for each of
    /// `last_tokens.len()` sequences at once. Index i of every slice belongs to
    /// sequence i; `target_hiddens[i]` is its `[1, hidden]` BF16 hidden, and
    /// the pointers need not be contiguous. There is no grammar mask.
    ///
    /// `Ok(None)` when unsupported, and the caller proposes per sequence;
    /// otherwise `drafts[i]` holds sequence i's drafts.
    #[allow(clippy::too_many_arguments)]
    fn propose_batch(
        &self,
        _last_tokens: &[u32],
        _target_hiddens: &[DevicePtr],
        _positions: &[usize],
        _num_drafts: usize,
        _states: &mut [&mut dyn ProposerState],
        _ctx: &ForwardContext,
        _stream: u64,
        _out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        Ok(None)
    }

    /// 2026-09-25: The widest batch [`Self::propose_batch`] accepts; 1 means
    /// per-sequence only. The scheduler groups batched proposes by it.
    fn propose_batch_max(&self, _buffers: &BufferArena, _config: &ModelConfig) -> usize {
        1
    }

    /// 2026-09-25: Append one drafter row per accepted draft of the last
    /// verify, for several sequences; the model calls it only with
    /// `ModelLevers::mtp_kv_exact` on. `tokens[i]` are sequence i's accepted
    /// drafts, `hiddens[i][k]` the target hidden paired with `tokens[i][k]`,
    /// and `first_pos[i]` the RoPE position of `tokens[i][0]`. Returns the rows
    /// written; the default writes none.
    fn catchup_batch(
        &self,
        _tokens: &[Vec<u32>],
        _hiddens: &[Vec<DevicePtr>],
        _first_pos: &[usize],
        _states: &mut [&mut dyn ProposerState],
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<usize> {
        Ok(0)
    }

    /// 2026-09-25: Prefill the drafter's KV over the prompt before the
    /// sequence's first `propose`. `hiddens` is `[P, hidden_size]` BF16, row
    /// `i` the target's hidden for `prompt_tokens[i]` before the final norm.
    /// Returns the drafter rows written; the default writes none.
    fn prefill_drafter(
        &self,
        prompt_tokens: &[u32],
        hiddens: DevicePtr,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let _ = (prompt_tokens, hiddens, state, ctx, stream);
        Ok(0)
    }

    /// 2026-09-25: The draft token id the last `propose` with
    /// `draft_embed_target` left on the device; the default returns 0.
    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        let _ = gpu;
        Ok(0)
    }

    /// 2026-09-25: Trim the proposer's state after a verify that accepted
    /// `num_accepted` drafts.
    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()>;

    /// 2026-09-25: Free per-sequence proposer state when a sequence finishes.
    /// `DevicePtr` has no `Drop`, so device memory `alloc_state` allocated
    /// leaks unless it is freed here; the default frees nothing.
    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let _ = (gpu, state);
        Ok(())
    }
}
