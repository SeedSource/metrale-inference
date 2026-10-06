// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The rank-agreed lazy-map admission of decode and verify steps (race-memory #79,
//! anomaly A168 follow-up; the prefill side is `prefill_b/lazy_agree.rs`).
//!
//! With `METRALE_DSA_INDEXER_LAZY=1` a step maps the DSA indexer rows (target layers and the MTP
//! drafter, `block_mgmt::map_lazy_rows_through`) before it writes them. The VMM lazy-map floor
//! reads this rank's own free device memory, so mid-generation one rank can refuse a granule the
//! other maps. The refusing worker then failed its command (`EpCommandFailed`) while rank 0 went
//! on into the step's first collective and waited forever.
//!
//! So every shared decode and verify step function (the ones the EP worker mirrors: single and
//! batched decode, the graphed K=2/3/4 and K=gamma verifies, the batched verify) calls
//! `agree_decode_lazy_maps` for each of its sequences before its first collective and before any
//! capture or replay. A sequence votes only when the step maps past `lazy_rows_agreed`, the
//! extent every rank is known to have backed:
//!
//! - below it, `map_rows_through` maps nothing on any rank, so the step cannot be refused there
//!   and no collective is spent (the steady state: one compare per sequence per step);
//! - past it (a granule boundary, about every 8,192 rows, plus the first step after prefill),
//!   each rank maps, then votes with one `ep_gather_u32`: 0 = refused, otherwise the rows its
//!   states are now backed through (`lazy_rows_backed`). Every rank refuses unless every vote is
//!   non-zero, and on success every rank sets `lazy_rows_agreed` to the minimum vote.
//!
//! The predicate reads only positions and `lazy_rows_agreed`, which change identically on every
//! rank, never the rank's own mapped extent (a refused step can leave partial maps on one rank,
//! and the head-only `ReserveKv` maps ahead), so every rank makes the same number of gathers.
//! A refusal is a "KV cache exhausted" error on every rank: the scheduler preempts a victim and
//! re-announces the step (decode), or fails the sequence (verify); `ReleaseSeq` frees the
//! sequence on both ranks (`0xFFFFFFF1`); the worker logs `EpCommandFailed` and keeps serving.
//!
//! The MTP propose maps nothing itself: its drafter rows stay within the 256-row look-ahead
//! (`PROPOSER_LOOKAHEAD_ROWS`) of the last agreed step end, and `lazy_rows_backed` nets the
//! look-ahead out, so the agreed extent covers them.
//!
//! Owner: model-engine (memory admission).
//! Invariants:
//! - `lazy_rows_agreed` changes only at a vote, to the minimum of the gathered values, so it is
//!   equal on every rank; it never exceeds what any rank has backed.
//! - Lever off, or no multi-rank protocol: `agree_decode_lazy_maps` returns at once, no
//!   collective, no map.

use anyhow::Result;

use super::super::block_mgmt::{lazy_map_end, map_lazy_states_through};
use super::super::types::TransformerModel;
use super::prefill_b::lazy_agree::agreed_outcome;
use crate::traits::SequenceState;

/// 2026-10-05: The extent through which `map_lazy_states_through(seq, end)` maps nothing new on
/// this rank: the minimum over the layer and proposer states. `None` when no state maps lazily.
pub(in crate::model) fn lazy_rows_backed(seq: &SequenceState) -> Option<usize> {
    seq.layer_states
        .iter()
        .filter_map(|st| st.rows_backed_through())
        .chain(
            seq.proposer_state
                .as_ref()
                .and_then(|p| p.rows_backed_through()),
        )
        .min()
}

/// 2026-10-05: This rank's vote: 0 when its admission failed, else the extent its states are
/// backed through, clamped to `1..=u32::MAX` (`u32::MAX` = fully backed, or nothing lazy).
pub(in crate::model) fn lazy_vote(local: &Result<()>, backed: Option<usize>) -> u32 {
    match local {
        Err(_) => 0,
        Ok(()) => backed.unwrap_or(usize::MAX).clamp(1, u32::MAX as usize) as u32,
    }
}

/// 2026-10-05: The rank-agreed verdict from this rank's admission and every rank's vote: `Ok`
/// with the new agreed extent (the minimum vote; `u32::MAX` widens to `usize::MAX`) only when
/// every vote is non-zero, else the error `prefill_b/lazy_agree::agreed_outcome` gives.
pub(in crate::model) fn agreed_lazy_rows(
    local: Result<()>,
    votes: &[u32],
    what: &str,
) -> Result<usize> {
    let admitted: Vec<u32> = votes.iter().map(|&v| u32::from(v != 0)).collect();
    agreed_outcome(local, &admitted, what)?;
    Ok(match votes.iter().copied().min() {
        Some(u32::MAX) => usize::MAX,
        Some(v) => v as usize,
        None => 0,
    })
}

/// 2026-10-05: One sequence's admission with the gather supplied (`ep_gather_u32` in the model;
/// a two-rank harness in tests). No gather when the step's map end (`lazy_map_end(last_block,
/// block_size)`) is within `lazy_rows_agreed`.
pub(in crate::model) fn agree_lazy_rows_with(
    seq: &mut SequenceState,
    last_block: usize,
    block_size: usize,
    gather: impl FnOnce(u32) -> Result<Vec<u32>>,
) -> Result<()> {
    let end = lazy_map_end(last_block, block_size);
    if end <= seq.lazy_rows_agreed {
        return Ok(());
    }
    let local = map_lazy_states_through(seq, end);
    let votes = gather(lazy_vote(&local, lazy_rows_backed(seq)))?;
    let what = format!(
        "decode/verify step of slot {} through position {end}",
        seq.slot_idx
    );
    seq.lazy_rows_agreed = agreed_lazy_rows(local, &votes, &what)?;
    Ok(())
}

impl TransformerModel {
    /// 2026-10-05: Rank-agreed lazy-map admission of `seq` for a decode or verify step whose
    /// last KV block is `last_block` (module docs). Every rank must call it at the same point of
    /// the same step, for the same sequences in the same order, before the step's first
    /// collective and outside any capture. Lever off or single rank: returns at once.
    pub(in crate::model) fn agree_decode_lazy_maps(
        &self,
        seq: &mut SequenceState,
        last_block: usize,
        block_size: usize,
    ) -> Result<()> {
        if !metrale_model_arch::glm5next_dsa::lazy::dsa_indexer_lazy()
            || !self.multi_rank_protocol_active()
        {
            return Ok(());
        }
        agree_lazy_rows_with(seq, last_block, block_size, |v| self.ep_gather_u32(v))
    }
}
