// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The rank-agreed outcome of a prefill chunk's KV-block and lazy-map admission
//! (race-memory #79, anomaly A168).
//!
//! `ensure_blocks_through_prefill` allocates the chunk's KV blocks and, with
//! `METRALE_DSA_INDEXER_LAZY=1`, maps the DSA indexer rows the chunk writes. The VMM map
//! checks the lazy-map floor against this rank's own `cuda_free_memory_bytes()`, so one rank
//! can refuse a chunk the other admits (seen at chunk 499,712 of a 532,195-token prompt:
//! rank 1 refused at 2,798 MiB free against the 2,800 MiB floor). The refusing worker then
//! returned `EpCommandFailed` while rank 0 entered the chunk's first collective and waited
//! forever. KV-block allocation can differ too: each rank evicts from its own prefix cache.
//!
//! So on a multi-rank world every rank votes on its own admission (1 = admitted) with one
//! `ep_gather_u32`, after the admission and before the chunk's first forward collective, and
//! the chunk proceeds only when every rank admitted it. Otherwise every rank returns an error
//! that names "KV cache exhausted" (the scheduler's existing preemption and error paths key on
//! it): the head answers the client and frees the sequence on both ranks (`0xFFFFFFF1`); the
//! worker logs `EpCommandFailed` and keeps serving.
//!
//! Owner: model-engine prefill (memory admission).
//! Invariants:
//! - `agreed_outcome` returns `Ok` on a rank only when every vote is 1, so all ranks reach
//!   the same verdict from the same gathered vector.
//! - A rank that refused returns its own error, with the agreement as context; a rank that
//!   admitted returns an error naming the refusing ranks.
//! - Without the multi-rank protocol `agree_admission` returns the local result unchanged:
//!   no collective, today's single-rank behaviour.

use anyhow::{Result, anyhow};

use super::super::super::types::TransformerModel;

/// 2026-10-05: Ranks whose vote is not 1 (refused), in rank order.
pub(in crate::model) fn refusing_ranks(votes: &[u32]) -> Vec<usize> {
    votes
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 1)
        .map(|(r, _)| r)
        .collect()
}

/// 2026-10-05: This rank's verdict from its own admission result and every rank's vote
/// (`votes[r]` = 1 when rank `r` admitted). An empty vector is a refusal: no rank's admission
/// is known.
pub(in crate::model) fn agreed_outcome(local: Result<()>, votes: &[u32], what: &str) -> Result<()> {
    let refused = refusing_ranks(votes);
    match local {
        Err(e) => Err(e.context(format!(
            "{what}: refused on this rank; rank-agreed refusal (refusing ranks {refused:?})"
        ))),
        Ok(()) if votes.is_empty() || !refused.is_empty() => Err(anyhow!(
            "KV cache exhausted: {what}: rank(s) {refused:?} refused this step's KV/lazy-map \
             admission, so every rank refuses it (rank-agreed)"
        )),
        Ok(()) => Ok(()),
    }
}

impl TransformerModel {
    /// 2026-10-05: Agree the outcome of a per-rank admission (`local`) across every rank of
    /// the communicator: one `ep_gather_u32` of a 0/1 vote, then `agreed_outcome`. Every rank
    /// must call it at the same point, before the step's first forward collective. Single-rank
    /// (no multi-rank protocol): returns `local` with no collective.
    pub(in crate::model) fn agree_admission(&self, local: Result<()>, what: &str) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return local;
        }
        let votes = self.ep_gather_u32(u32::from(local.is_ok()))?;
        agreed_outcome(local, &votes, what)
    }
}
