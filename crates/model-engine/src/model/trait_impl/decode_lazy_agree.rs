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
//!
//! 2026-10-14 (race-memory, A183): the paged KV blocks themselves had the same hole. Decode and
//! verify grow a sequence's blocks per rank (`ensure_blocks_through_decode`) with no vote, each
//! rank sizes its pool from its own free memory (hyb27c 10-11: 81,825 / 81,826 blocks) and only
//! rank 0 inserts into its prefix cache, so one rank could hit "KV cache exhausted" while the
//! other ran on: rank 1 refusing alone left rank 0 in the step's first collective (hang); rank 0
//! refusing alone timed its next broadcast out (exit 75). With `METRALE_KV_DECODE_AGREE=1`
//! every rank caps its usable pool at the minimum block count over the ranks at startup
//! (`TransformerModel::new`), and `agree_decode_kv_blocks` runs right after
//! `agree_decode_lazy_maps` in the same seven step functions: a sequence votes only when the
//! step's last block reaches `kv_blocks_agreed` (once per block of growth, plus the first step
//! after allocation), grows its blocks, and gathers one 0/1 vote; a refusal anywhere is a
//! rank-agreed "KV cache exhausted" on every rank, answered by the scheduler as above.
//! `kv_blocks_agreed` changes only at a vote, so every rank makes the same number of gathers.

use std::sync::OnceLock;

use anyhow::{Result, anyhow};
use metrale_cache::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    ensure_blocks_through_decode, lazy_map_end, map_lazy_states_through,
};
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

/// 2026-10-14 (race-memory, A183): `METRALE_KV_DECODE_AGREE=1` (module docs). Read once.
pub(crate) fn kv_decode_agree() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("METRALE_KV_DECODE_AGREE").as_deref(), Ok("1")))
}

/// 2026-10-14 (race-memory, A183): `METRALE_KV_DECODE_FAIL_ABOVE_BLOCKS` (test-only fault
/// injection, 0 = off): an agreed KV growth whose last block index is at least N refuses on this
/// rank. Read once, per process (per rank); deliberately NOT rank-agreed, so a test can set it on
/// one rank only.
pub(in crate::model) fn kv_decode_fail_above_blocks() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("METRALE_KV_DECODE_FAIL_ABOVE_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// 2026-10-14 (race-memory, A183): One sequence's rank-agreed KV growth through block
/// `last_block`, with the growth and the gather supplied (the model's
/// `ensure_blocks_through_decode` and `ep_gather_u32`; a two-rank harness in tests). No growth
/// and no gather when `last_block < kv_blocks_agreed`. `fail_above` > 0 refuses locally at
/// `last_block >= fail_above` (the test fault-injection lever).
pub(in crate::model) fn agree_kv_blocks_with(
    seq: &mut SequenceState,
    last_block: usize,
    fail_above: usize,
    grow: impl FnOnce(&mut SequenceState) -> Result<()>,
    gather: impl FnOnce(u32) -> Result<Vec<u32>>,
) -> Result<()> {
    if last_block < seq.kv_blocks_agreed {
        return Ok(());
    }
    let local = if fail_above > 0 && last_block >= fail_above {
        Err(anyhow!(
            "KV cache exhausted: test fault-injection lever \
             METRALE_KV_DECODE_FAIL_ABOVE_BLOCKS={fail_above} (block {last_block})"
        ))
    } else {
        grow(seq)
    };
    let votes = gather(u32::from(local.is_ok()))?;
    let what = format!(
        "decode/verify step of slot {} through KV block {last_block}",
        seq.slot_idx
    );
    agreed_outcome(local, &votes, &what)?;
    seq.kv_blocks_agreed = last_block + 1;
    Ok(())
}

impl TransformerModel {
    /// 2026-10-14 (race-memory, A183): Rank-agreed KV block growth of `seq` through `last_block`
    /// for a decode or verify step (module docs). Every rank must call it at the same point of
    /// the same step, for the same sequences in the same order, before the step's first
    /// collective and outside any capture; the step's own `ensure_blocks_through_decode` that
    /// follows then finds the blocks in place. Lever off or single rank: returns at once.
    pub(in crate::model) fn agree_decode_kv_blocks(
        &self,
        seq: &mut SequenceState,
        last_block: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        if !kv_decode_agree() || !self.multi_rank_protocol_active() {
            return Ok(());
        }
        agree_kv_blocks_with(
            seq,
            last_block,
            kv_decode_fail_above_blocks(),
            |s| {
                ensure_blocks_through_decode(
                    s,
                    last_block,
                    kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                    self.levers.kv_poison,
                )
            },
            |v| self.ep_gather_u32(v),
        )
    }

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
