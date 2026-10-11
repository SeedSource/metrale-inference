// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelLifecycle`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use crate::traits::SequenceState;
use anyhow::{Result, bail};

/// 2026-09-26: The model's lifecycle and memory: teardown, thread binding, sequence allocation and
/// release, sequence swap, and the KV pool counts.
pub trait ModelLifecycle {
    /// 2026-09-25: Release the device memory this model owns, newest first. The scheduler calls
    /// it once at shutdown (`core/finish.rs`). Default: `Ok(())`, for models that own no pooled
    /// device memory; a model that owns pools and keeps the default leaks them.
    fn teardown(&mut self) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Poll this model's TQ+ InnerQ calibration driver; the scheduler calls it once
    /// per prefill step. Default: no-op.
    fn poll_innerq(&self) {}

    /// 2026-09-25: Model dimensions for the `--high-speed-swap` orchestrator; `None` for a model
    /// that does not support it (the default).
    fn high_speed_swap_dims(&self) -> Option<metrale_storage::ModelDims> {
        None
    }

    /// 2026-09-25: Bind the GPU context to the current thread. The scheduler thread and the EP
    /// worker call it before using the model.
    fn bind_gpu_to_thread(&self) -> Result<()>;

    /// 2026-09-25: A new sequence with its layer states.
    fn alloc_sequence(&self) -> Result<SequenceState>;

    /// 2026-09-25: [`Self::alloc_sequence`] for a request that can reach at most `budget_tokens`
    /// (`prompt_len + max_tokens`), so proposer state that scales with context can be sized to
    /// that (`DraftProposer::alloc_state_for`). Default: [`Self::alloc_sequence`].
    fn alloc_sequence_for(&self, budget_tokens: usize) -> Result<SequenceState> {
        let _ = budget_tokens;
        self.alloc_sequence()
    }

    /// 2026-09-25: Insert the sequence's tokens (prompt and generated) into the prefix cache. Call
    /// before [`Self::free_sequence`], while its blocks are still valid.
    fn cache_sequence(&self, seq: &SequenceState);

    /// 2026-09-25: Release the sequence's KV blocks, prefix-cache refs, SSM pool slot and LoRA
    /// slot ref.
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()>;

    /// 2026-09-25: Move the sequence's SSM state (`h_state` and `conv_state` of every SSM layer)
    /// to pool slot `new_slot`.
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()>;

    /// 2026-10-10: `METRALE_MTP_POOL_CAP`: move a sequence on an SSM slot the MTP verify pools
    /// do not cover onto the lowest free covered slot, on every EP rank. Rank 0 only, between
    /// steps with no step in flight. Returns the new slot, or `None` when there is nothing to
    /// do. Default: `None`.
    fn migrate_to_covered_slot(&self, seq: &mut SequenceState) -> Result<Option<usize>> {
        let _ = seq;
        Ok(None)
    }

    /// 2026-09-25: Give up a retired sequence's SSM pool slot after `compact_sequence` moved a
    /// surviving sequence into it: sets `slot_idx` to `usize::MAX` and empties the slot guard, so
    /// freeing or dropping this sequence does not release the slot. Call it right after that
    /// `compact_sequence`, before any step that can drop the sequence.
    fn detach_slot_for_reuse(&self, seq: &mut SequenceState);

    /// 2026-09-25: Write the sequence's KV blocks and SSM state to `writer`, in a format the
    /// model owns, without freeing anything. Default: an error.
    fn save_sequence_state(
        &self,
        _seq: &SequenceState,
        _writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        bail!("swap not supported by this model")
    }

    /// 2026-09-25: Read state written by [`Self::save_sequence_state`] into an allocated
    /// sequence, allocating `num_blocks` KV blocks. Default: an error.
    fn restore_sequence_state(
        &self,
        _seq: &mut SequenceState,
        _num_blocks: usize,
        _reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        bail!("swap not supported by this model")
    }

    /// 2026-09-25: Free KV blocks. Default `0`.
    fn num_free_blocks(&self) -> usize {
        0
    }

    /// 2026-09-25: Total KV blocks in the paged cache. Default `0`.
    fn num_total_blocks(&self) -> usize {
        0
    }

    /// 2026-09-25: Evict up to `num_blocks` blocks from the prefix cache; returns how many became
    /// free. Swap-in and preemption resume check `num_free_blocks()` before they allocate, so
    /// they call this to reach blocks the cache holds. `0` means nothing was evictable, and the
    /// caller stops asking. Default `0`.
    fn reclaim_prefix_blocks(&self, _num_blocks: usize) -> usize {
        0
    }
}
