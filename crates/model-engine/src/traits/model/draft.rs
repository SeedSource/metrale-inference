// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelDraft`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use crate::traits::SequenceState;
use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-09-26: Draft proposers: MTP and DFlash proposals, the hidden-state saves and stashes they read,
/// and the DFlash context appends.
pub trait ModelDraft {
    /// 2026-09-25: The most drafts a speculative step may send to the sequence in SSM pool slot
    /// `slot_idx` without overflowing that slot's per-token intermediate state. The scheduler
    /// uses the minimum over the active slots. Default `usize::MAX` (no SSM verify pools).
    fn mtp_slot_draft_capacity(&self, _slot_idx: usize) -> usize {
        usize::MAX
    }

    /// 2026-09-25: Generate for one prompt with the model's own draft proposer.
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &metrale_sampling::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult>;

    /// 2026-09-25: Whether a draft proposer or self-speculative decoding is available.
    fn has_proposer(&self) -> bool;

    /// 2026-09-25: The installed DFlash drafter's block size γ, read from the drafter; the serve
    /// layer derives `num_drafts = γ - 1` from it. Default `None`.
    fn dflash_gamma(&self) -> Option<usize> {
        None
    }

    fn has_self_speculative(&self) -> bool;

    /// 2026-09-25: The draft decode step for self-speculative decoding; returns logits.
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr>;

    /// 2026-09-25: Copy hidden row `rows[i]` of the last batched verify into stash slot `i`,
    /// before a propose overwrites the shared hidden buffer. Default: an error.
    fn stash_verify_hidden_rows(&self, rows: &[usize], stream: u64) -> Result<()> {
        let _ = (rows, stream);
        bail!("stash_verify_hidden_rows: unsupported by this model")
    }

    /// 2026-09-25: Copy batched-verify rows into drafter catch-up stash slots,
    /// `slot_rows[j] = (slot, row)`. Default: `Ok(())`.
    fn stash_verify_catchup_rows(&self, slot_rows: &[(usize, usize)]) -> Result<()> {
        let _ = slot_rows;
        Ok(())
    }

    /// 2026-09-25: Append one drafter row per accepted draft before the next propose (see
    /// [`Self::stash_verify_catchup_rows`]); returns the rows written. Default: 0.
    fn run_mtp_catchup_batched(
        &self,
        tokens: &[Vec<u32>],
        first_slot: &[usize],
        first_pos: &[usize],
        seqs: &mut [&mut SequenceState],
    ) -> Result<usize> {
        let _ = (tokens, first_slot, first_pos, seqs);
        Ok(0)
    }

    /// 2026-09-25: [`Self::save_hidden_for_mtp`] from stash slot `idx` (written by
    /// [`Self::stash_verify_hidden_rows`]) instead of the live verify rows, which a propose may
    /// have overwritten. Default: an error.
    fn save_hidden_for_mtp_from_stash(&self, idx: usize, stream: u64) -> Result<()> {
        let _ = (idx, stream);
        bail!("save_hidden_for_mtp_from_stash: unsupported by this model")
    }

    /// 2026-09-25: `num_drafts` MTP drafts for each of `tokens.len()` sequences in one drafter
    /// forward per draft position. `stash_idx[i]` is the verify-stash slot holding sequence
    /// `i`'s hidden, and `positions[i]` its propose position, as for
    /// [`Self::run_mtp_propose_multi`].
    ///
    /// `out_conf`, when `Some`, receives each draft's top-1 log-probability in the shape of the
    /// returned drafts, or zeros when the drafter cannot measure it. `Ok(None)` means
    /// unsupported, and the caller proposes per sequence. Default: `Ok(None)`.
    #[allow(clippy::too_many_arguments)]
    fn run_mtp_propose_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let _ = (
            tokens, positions, stash_idx, num_drafts, seqs, stream, out_conf,
        );
        Ok(None)
    }

    /// 2026-09-25: The widest batch [`Self::run_mtp_propose_batched`] takes in one drafter
    /// forward. Default `1` (per sequence only).
    fn mtp_propose_batch_max(&self) -> usize {
        1
    }

    /// 2026-10-04: True when [`Self::run_mtp_propose_batched`] proposes exactly what the
    /// per-sequence propose would for every sequence (`DraftProposer::propose_batch_mirrors_serial`,
    /// `METRALE_GLM_MTP_BATCH_DRAFT=1`); the scheduler then keeps sequences the per-sequence
    /// propose would treat differently (a grammar) out of the batch. Default false.
    fn mtp_propose_batch_mirrors_serial(&self) -> bool {
        false
    }

    /// 2026-09-25: Copy the hidden state at `token_idx` into the MTP input buffer, which
    /// `run_mtp_propose` and `run_mtp_propose_multi` read. `TransformerModel` copies the hidden
    /// state before the final norm, because the MTP head applies its own norm.
    fn save_hidden_for_mtp(&self, token_idx: usize, stream: u64) -> Result<()>;

    /// 2026-09-25: Store a serially decoded token's final hidden at `pos` in the drafter catch-up
    /// ring. Default: `Ok(())`.
    fn save_hidden_for_catchup(&self, _token_idx: usize, _pos: usize) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Capture row `token_idx` from every DFlash capture layer for the next propose.
    /// Default: `Ok(())`.
    fn save_dflash_hidden_for_propose(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Append row 1 of the DFlash hidden scratch to the proposer context. Default:
    /// `Ok(())`.
    fn dflash_accept_append(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Append scratch rows 0 and 1 to the DFlash context at positions `seq_len - 2`
    /// and `seq_len - 1`, row 1 last, before the propose. Default: `Ok(())`.
    fn dflash_eagle_accept_append(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Append scratch rows `0..=num_accepted` to the DFlash context at positions
    /// `base_pos..=base_pos + num_accepted`, row `num_accepted` last. Default: `Ok(())`.
    fn dflash_eagle_kgamma_append(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _base_pos: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Append the just-decoded token's captured hidden (scratch row 0) to the DFlash
    /// context at position `seq_len - 1`, for a sequence decoded serially while speculation is
    /// suspended; otherwise that capture is overwritten and the context has a hole. It sets
    /// `skip_next_decode_append` so a later propose does not append it again. No-op without a
    /// DFlash proposer state. Default: `Ok(())`.
    fn dflash_serial_ctx_append(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Append `num_committed` DFlash scratch rows, from `scratch_row`, at the end of
    /// the context, stamped with positions `base_pos..base_pos + num_committed`, after sliding
    /// the window if it would overflow. `base_pos` is a position, not a context row: the two
    /// differ after a slide. Default: `Ok(())`.
    fn commit_ctx(
        &self,
        _seq: &mut SequenceState,
        _num_committed: usize,
        _base_pos: usize,
        _scratch_row: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Rows per sequence band in the DFlash hidden scratch: sequence `i` of a batched
    /// verify captures into band `i`, so its `commit_ctx` `scratch_row` is
    /// `i * dflash_capture_band()`. Default `0` (no DFlash drafter).
    fn dflash_capture_band(&self) -> usize {
        0
    }

    /// 2026-09-25: One MTP draft from the saved hidden state; `None` without a proposer.
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Option<u32>>;

    /// 2026-09-25: `num_drafts` MTP drafts from the hidden state saved by `save_hidden_for_mtp`.
    ///
    /// `grammar_bitmask`, when `Some`, limits drafts to a grammar's allowed tokens:
    /// `ceil(vocab_size / 32)` words, bit `tok` set when `tok` is allowed.
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>>;

    /// 2026-09-25: The draft id the proposer left on the device. Default `0`.
    fn read_deferred_draft_token(&self) -> Result<u32> {
        Ok(0)
    }

    /// 2026-09-25: Tell the proposer how many drafts the last verify accepted, so it can trim its
    /// per-sequence state.
    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        stream: u64,
    ) -> Result<()>;
}
