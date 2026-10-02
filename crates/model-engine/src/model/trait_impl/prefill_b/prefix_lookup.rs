// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Prefix-cache lookup for a prefill chunk: the match (agreed across ranks
//! on a multi-rank world), the rank-agreed SSM snapshot restore, and the skip point.
//! Returns `(kv_write_start, marconi_skip)`.
//!
//! Owner: model-engine prefill (prefix cache).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_telemetry::prefix_cache::PrefixMatch;

use super::super::super::block_mgmt::reuse_prefix_match_disk_ids;
use super::super::super::types::TransformerModel;
use super::snap_agree;
use crate::traits::{PrefillSlice, SequenceState};

impl TransformerModel {
    pub(in crate::model) fn prefill_b_prefix_lookup(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        total: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
        reserved_match: Option<PrefixMatch>,
    ) -> Result<(usize, bool)> {
        let bs = kv_cache.block_size();
        // 2026-09-25: A second chunk-0 call for the same sequence (a retried prefill)
        // replays the first decision: the first call already took a reference on each
        // matched block and pushed it onto `block_table`, and a rerun would do both again.
        if chunk_start == 0 && seq.prefix_lookup_applied {
            tracing::debug!(
                "prefix lookup: replayed (already applied) slot={}",
                seq.slot_idx
            );
            tracing::debug!(
                "prefix lookup re-entered at chunk 0 (retry): replaying \
                 skip_to={} skip={} without re-acquiring the cached prefix",
                seq.marconi_skip_to,
                seq.prefix_lookup_skip,
            );
            return Ok((seq.marconi_skip_to, seq.prefix_lookup_skip));
        }
        if chunk_start == 0 {
            // 2026-09-25: No match for a vision prompt, for prompt-logprob collection (every
            // position needs a computed hidden row), or when MLA prefill cannot skip a
            // prefix (`mla_prefill_needs_full_recompute`). A batched reservation is used
            // as is.
            let reserved = reserved_match.is_some();
            let mut prefix_match = if self.tokens_have_vision_pad(tokens)
                || seq.collect_prompt_logprobs.is_some()
                || self.mla_prefill_needs_full_recompute()
            {
                PrefixMatch::empty()
            } else if let Some(prefix_match) = reserved_match {
                prefix_match
            } else {
                self.prefix_cache
                    .lookup(tokens, bs, seq.session_hash, seq.adapter_id)
            };
            // 2026-09-25: On a multi-rank world (EP or pure TP) each rank's prefix cache can
            // match a different length, and different lengths give different `proc_count`s,
            // so the collectives would not pair up. Every rank takes the minimum over
            // ranks (`ep_min_u32`), even with nothing matched, because each rooted
            // broadcast needs a receiver on every rank. A rank that matched more releases
            // its match and looks up the agreed prefix again.
            let ep_active = self.multi_rank_protocol_active();
            if ep_active && !reserved {
                let local = prefix_match.matched_tokens as u32;
                let agreed = self.ep_min_u32(local)? as usize;
                if agreed < prefix_match.matched_tokens {
                    self.prefix_cache.release(tokens, bs, seq.adapter_id);
                    if agreed > 0 {
                        prefix_match = self.prefix_cache.lookup(
                            &tokens[..agreed],
                            bs,
                            seq.session_hash,
                            seq.adapter_id,
                        );
                    } else {
                        prefix_match = metrale_telemetry::prefix_cache::PrefixMatch::empty();
                    }
                    tracing::info!(
                        "F83 EP-cache-sync: local_matched={local} agreed_matched={agreed} \
                         (cap to min across ranks)"
                    );
                } else if local > 0 || agreed > 0 {
                    tracing::debug!(
                        "F83 EP-cache-sync: local_matched={local} agreed_matched={agreed} (no cap)"
                    );
                }
            }
            let matched = prefix_match.matched_tokens;
            tracing::debug!(
                "prefix lookup: matched={matched} of {total} (reserved={reserved}) \
                 snapshot_tokens={} tail={} slot={}",
                prefix_match.ssm_snapshot_tokens,
                prefix_match.ssm_snapshot_is_tail,
                seq.slot_idx,
            );
            seq.cached_prefix_tokens = matched;
            seq.cached_prefix_blocks = prefix_match.matched_blocks.len();
            // 2026-09-25: A new prefill: a tail checkpoint recorded by the previous turn
            // belongs to that turn.
            seq.tail_checkpoint_tokens = None;
            // 2026-09-25: Keep the matched prefix tokens so that `free_sequence` can release
            // the lookup's references even when the prefill fails before `seq.tokens`
            // covers them (sequence.rs). Cleared when nothing matched.
            if matched > 0 {
                seq.prefix_ref_tokens = tokens[..matched].to_vec();
            } else {
                seq.prefix_ref_tokens.clear();
            }
            seq.prompt_len = total;
            for &block_idx in &prefix_match.matched_blocks {
                kv_cache.inc_ref(block_idx);
                seq.block_table.push(block_idx);
            }
            reuse_prefix_match_disk_ids(
                &prefix_match.matched_disk_block_ids,
                &mut seq.disk_block_ids,
            );
            // 2026-09-25: Matched blocks bring their disk block ids (above), so raise each
            // layer's offload cursor to at least `disk_block_ids.len()`. The HSS slide in
            // `block_mgmt` refuses to evict a block that a layer's cursor has not passed
            // (`check_safe_to_evict`).
            let new_total = seq.disk_block_ids.len() as u32;
            for cursor in seq.disk_last_offloaded_per_layer.iter_mut() {
                if *cursor < new_total {
                    *cursor = new_total;
                }
            }
            // 2026-09-25: SSM snapshot candidate. Its depth can be below `matched`; the SSM
            // state between it and `matched` is then recomputed, while the K/V write floor
            // in `forward_layers.rs` keeps those shared blocks unwritten.
            // `eff_ssm_snapshot` (`ssm_fault_in.rs`) also considers a snapshot faulted back
            // from the spill tier.
            let (eff_snapshot, eff_snapshot_tokens) =
                self.eff_ssm_snapshot(&prefix_match, seq.session_hash, stream);
            let has_ssm = self.config.num_ssm_layers() > 0;

            // 2026-09-25: Whether and where to restore depends on rank-local state (the
            // snapshot pool and the gates below), so the gates are folded into a proposal
            // and a snapshot is restored only on an all-or-nothing agreement
            // (`snap_agree`). The gates are evaluated before the exchange: the exact
            // full-prompt restore (off unless `METRALE_MARCONI_EXACT=1`, and only with a
            // stashed hidden), the session gate (tail snapshots only), `marconi_min_tokens()`,
            // and the aux gate.
            let gates = snap_agree::LocalGates {
                snap_tok: if eff_snapshot.is_some() {
                    eff_snapshot_tokens
                } else {
                    0
                },
                matched,
                total,
                min_tokens: metrale_model_layers::mtp_carry::marconi_min_tokens(),
                has_hidden: eff_snapshot.is_some_and(|s| self.ssm_snapshots.has_hidden(s)),
                exact_enabled: super::exact_leaf::marconi_exact_enabled(),
                is_tail: prefix_match.ssm_snapshot_is_tail,
                session_ok: eff_snapshot
                    .is_some_and(|s| self.ssm_snapshots.session_matches(s, seq.session_hash)),
                needs_aux: self.requires_aux_state(),
                has_aux: eff_snapshot.is_some_and(|s| self.ssm_snapshots.has_aux(s)),
            };
            let proposal = snap_agree::local_proposal(&gates);
            // 2026-09-25: Every rank of a multi-rank SSM world takes part, even with nothing
            // to propose, or the other ranks' rooted broadcasts have no receiver.
            let agreed = if ep_active && has_ssm {
                let votes = self.ep_gather_u32(proposal)?;
                let agreed = snap_agree::agree(&votes);
                if votes.iter().any(|&v| v != 0) {
                    tracing::info!(
                        "A100 snap-agree: rank={} local_proposal={proposal} votes={votes:?} \
                         agreed={agreed:?} (matched={matched} total={total})",
                        self.comm.as_ref().map_or(0, |c| c.rank()),
                    );
                } else {
                    tracing::debug!(
                        "A100 snap-agree: no rank holds a snapshot (matched={matched}) — recompute"
                    );
                }
                agreed
            } else {
                snap_agree::agree(&[proposal])
            };
            let restore = match (agreed, eff_snapshot) {
                (Some(t), Some(snap_id)) if t as usize == eff_snapshot_tokens => {
                    Some((snap_id, t as usize))
                }
                (Some(t), _) => anyhow::bail!(
                    "A100 invariant violated: ranks agreed to restore token {t} but this rank's \
                     candidate is {eff_snapshot:?}@{eff_snapshot_tokens} (proposal {proposal})"
                ),
                (None, _) => None,
            };

            let mut skip = if let Some((snap_id, snap_tok)) = restore {
                // 2026-09-25: A snapshot save can be in flight on another stream: make this
                // stream wait for every save recorded so far before reading the slot
                // (`wait_snapshot_saves_dispatch`).
                self.wait_snapshot_saves_dispatch(stream)?;
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
                // 2026-09-25: The restored snapshot stays in the pool, so record it as this
                // prefill's tail checkpoint; `finalize_last` then skips an exact leaf within
                // two blocks of it (`exact_leaf`).
                seq.tail_checkpoint_tokens = Some(snap_tok);
                if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
                    self.ssm_pool.debug_state_checksum(
                        seq.slot_idx,
                        self.gpu.as_ref(),
                        stream,
                        &format!("restore@{snap_tok}"),
                    );
                }
                if snap_tok < matched {
                    // 2026-09-25: The suffix prefill resumes at `snap_tok`, so the SSM replay is
                    // `total - snap_tok`; `matched - snap_tok` of it is due to snapshot
                    // granularity. The log prints both.
                    tracing::info!(
                        "Marconi intermediate hit: restored from checkpoint at token {} \
                         (skipping {} tokens, replaying {} SSM tokens to reach {}; \
                         {} of those are the anchor->match gap to {})",
                        snap_tok,
                        snap_tok,
                        total.saturating_sub(snap_tok),
                        total,
                        matched.saturating_sub(snap_tok),
                        matched,
                    );
                } else {
                    tracing::info!(
                        "Marconi SSM cache hit: {} tokens skipped ({} blocks), \
                         snapshot {}, replaying {} SSM tokens to reach {}",
                        matched,
                        prefix_match.matched_blocks.len(),
                        snap_id,
                        total.saturating_sub(snap_tok),
                        total,
                    );
                    // 2026-09-25: Exact full-prompt restore (`snap_tok == matched == total`):
                    // the last prompt token is re-run for logits and applied to the SSM
                    // state a second time. Flag it so `finalize_last` restores the snapshot
                    // again and takes the first token from its stashed hidden. A shorter
                    // match needs no fixup: the suffix pass continues from the snapshot.
                    if matched == total {
                        seq.marconi_exact_snap = Some(snap_id);
                    }
                }
                true
            } else {
                false
            };
            // 2026-09-25: `skip` is true here only when `restore` is `Some`, so this branch
            // cannot fire. The default refusal of an exact full-prompt restore is
            // `bypass_exact` in `snap_agree::local_proposal`.
            if skip
                && restore.is_none()
                && prefix_match.ssm_snapshot_tokens == matched
                && matched == total
                && !super::exact_leaf::marconi_exact_enabled()
            {
                skip = false;
                seq.marconi_exact_snap = None;
                tracing::info!(
                    "exact-leaf snapshot shortcut bypassed (default; METRALE_MARCONI_EXACT=1 re-enables) \
                     for {matched}-token full hit — recomputing all KV+SSM"
                );
            }
            if matched > 0 && !skip && has_ssm {
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) but no SSM snapshot — recomputing all KV",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            } else if matched > 0 && !skip {
                // 2026-09-25: A model without SSM layers skips the matched tokens with no
                // snapshot.
                skip = true;
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) reused (F82+F83: non-SSM cache-hit skip)",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            }
            // 2026-09-25: Skip point (`snap_agree::skip_point`). Without SSM layers a hit
            // skips `matched` tokens. With SSM layers it skips to the restored depth, the
            // agreed `snap_tok` (from `eff_snapshot_tokens`, so a snapshot faulted back from
            // the spill tier counts): that equals `total` only for an exact full-prompt
            // restore, and otherwise the suffix prefill recomputes SSM from `snap_tok`.
            let snap_tok = restore.map_or(0, |(_, t)| t);
            // 2026-09-25: With SSM layers, `skip` is exactly the rank-agreed restore outcome.
            // Without them it is not: a hit sets `skip` with no restore.
            debug_assert!(
                !has_ssm || skip == restore.is_some(),
                "A105: skip ({skip}) diverged from the rank-agreed restore outcome \
                 ({}) for an SSM sequence — something flipped skip after the vote",
                restore.is_some()
            );
            let skip_tokens = snap_agree::skip_point(skip, snap_tok, matched, total, has_ssm);
            seq.marconi_skip_to = skip_tokens;
            // 2026-09-25: Report what was reused, not what matched: an SSM hit without a
            // restore has `matched > 0` but skips nothing (`reused_prefix_tokens`).
            seq.reused_prefix_tokens = crate::model::trait_impl::prefix_reuse::reused_prefix_tokens(
                matched,
                skip_tokens,
                skip,
            );
            seq.prefix_lookup_skip = skip;
            seq.prefix_lookup_applied = true;
            Ok((skip_tokens, skip))
        } else if seq.marconi_skip_to > 0 {
            // 2026-09-25: Later chunks reuse chunk 0's skip point.
            Ok((seq.marconi_skip_to, true))
        } else {
            Ok((0, false))
        }
    }
}
