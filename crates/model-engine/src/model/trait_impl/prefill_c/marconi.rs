// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `prefill_twophase`'s Marconi step: restore a matching SSM snapshot and decide
//! where the KV write and the processed range start.
//!
//! Owner: model-engine.
//! Invariants: `seq.marconi_skip_to` and `seq.reused_prefix_tokens` are set from the returned
//! `kv_write_start` before this returns `Ok`.

use anyhow::Result;
use metrale_telemetry::prefix_cache::PrefixMatch;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-09-26: Returns `(kv_write_start, marconi_skip)`: the token the restored SSM state
    /// is at and `true` after a restore, else `(0, false)`.
    pub(super) fn twophase_marconi_restore(
        &self,
        prefix_match: &PrefixMatch,
        seq: &mut SequenceState,
        matched: usize,
        total_len: usize,
        stream: u64,
    ) -> Result<(usize, bool)> {
        // 2026-09-25: Marconi: restore an SSM snapshot for this session if one matches.
        // `eff_ssm_snapshot` also considers a spilled anchor it faults back in.
        let (eff_snapshot, eff_snapshot_tokens) =
            self.eff_ssm_snapshot(prefix_match, seq.session_hash, stream);
        let (kv_write_start, marconi_skip) = if let Some(snap_id) = eff_snapshot {
            let snap_tok = eff_snapshot_tokens;
            if snap_tok > 0
                && matched <= total_len
                && self
                    .ssm_snapshots
                    .session_matches(snap_id, seq.session_hash)
                // 2026-09-25: Aux-carrying models decline aux-less snapshot slots.
                && (!self.requires_aux_state() || self.ssm_snapshots.has_aux(snap_id))
            {
                self.ssm_snapshots.restore(
                    snap_id,
                    seq.slot_idx,
                    &self.ssm_pool,
                    self.gpu.as_ref(),
                    stream,
                )?;
                if let Some(aux) = self.ssm_snapshots.aux(snap_id) {
                    self.apply_aux_states(seq, &aux, stream)?;
                }
                tracing::info!(target: "metrale_model_engine::model::trait_impl::prefill_c", "Marconi two-phase: restored SSM snapshot at token {snap_tok} \
                         ({matched} KV blocks cached)",
                );
                // 2026-09-25: The restored SSM state is at token `snap_tok`. Skip to
                // `matched` only when the whole prompt matched and the snapshot
                // covers it; otherwise skip to `snap_tok`, so the suffix prefill
                // recomputes the SSM state over [snap_tok, total_len) instead of
                // decoding from a state behind the KV.
                let skip = if matched >= total_len && snap_tok >= matched {
                    matched
                } else {
                    snap_tok
                };
                (skip, true)
            } else {
                (0, false)
            }
        } else {
            if matched > 0 {
                tracing::info!(target: "metrale_model_engine::model::trait_impl::prefill_c", "Prefix cache hit: {} tokens ({} blocks) but no SSM snapshot — \
                         recomputing all KV",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            }
            (0, false)
        };
        seq.marconi_skip_to = kv_write_start;
        // 2026-09-25: The reported cached-token count is the KV actually reused, not
        // the lookup's match (see `prefix_reuse`).
        seq.reused_prefix_tokens = crate::model::trait_impl::prefix_reuse::reused_prefix_tokens(
            matched,
            kv_write_start,
            marconi_skip,
        );
        Ok((kv_write_start, marconi_skip))
    }
}
