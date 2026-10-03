// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Absolute-grid prefix-cache restore (`METRALE_PREFIX_GRID_RESTORE=1`, race #69):
//! a prefix-cache hit produces the output of a cold prefill of the same prompt.
//!
//! A hit used to restore at the deepest snapshot (a tail snapshot such as token 31,984) and
//! prefill only the suffix, in a chunk layout no cold prefill runs; outputs then differed from
//! cold (TC-58 at T=0.6: 16/24 vs 21/24 over 24 paired seeds, measured 2026-10-03). Under the
//! grid, with `G` the grid size (`prefill_plan::grid_restore_lever`):
//!
//! - Chunks: every non-last prefill chunk is exactly `[k*G, (k+1)*G)`, whatever the scheduler's
//!   budget (`prefill_plan::grid_chunk_len`, through `Model::prefill_grid`). No tail split, no
//!   in-pass or mid-chunk tail capture.
//! - Snapshots: one at the end of every non-last chunk (`save_checkpoint`: SSM/KDA h and conv
//!   state plus the DSA indexer aux blob, the plain end-of-pass state), and no others: no
//!   prompt-end leaf (`finalize_last`), no decode checkpoints, no finish leaf (`cache_sequence`
//!   is skipped). Eviction is the pool's existing one (`reclaim_from_cache`: stalest session
//!   first, then least recent within it), which evicts the newest save, the deepest boundary of
//!   the latest prompt, last.
//! - K/V provenance: the radix tree only ever receives blocks below the prompt's last chunk
//!   start (`prefill_plan::grid_insert_len`), so every cached block was written by a full `G`-row
//!   non-last chunk at its grid offset, the same pass a cold prefill of any prompt sharing that
//!   prefix runs over it. The last chunk's blocks (a shorter pass) and generated tokens (decode
//!   and verify passes) are never cached.
//! - Restore: the largest `k*G` that is at most the match, strictly below the prompt end, and
//!   holds a snapshot ([`TransformerModel::grid_floor_match`], `prefill_plan::grid_search`). The
//!   match is truncated to exactly that point, so the restored sequence holds blocks `[0, k*G)`
//!   and computes `[k*G, total)` into its own blocks, in the chunks a cold prefill runs from
//!   `k*G`. On a multi-rank world the ranks take the minimum point (`ep_min_u32`) and then the
//!   existing all-or-nothing vote (`snap_agree`) decides the restore.
//! - Ref accounting: the sequence holds radix refs on exactly `prefix_ref_tokens`
//!   (`SequenceState::prefix_grid_refs`): the lookup's `[0, k*G)`, widened by `finalize_last`'s
//!   insert to `[0, grid_insert_len)`. `free_sequence` releases exactly that.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants:
//! - With the lever off, `prefix_grid_active_bs` returns `None` before anything else is read,
//!   and every caller then takes its previous path.
//! - After `grid_floor_match` the match is empty or ends exactly at a grid point below the
//!   prompt end whose deepest snapshot is at that point (on this rank).

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_telemetry::prefix_cache::PrefixMatch;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-10-03: Depth of a lookup's deepest snapshot, resident or spilled; `0` for none.
pub(in crate::model) fn match_snap_depth(m: &PrefixMatch) -> usize {
    if m.ssm_snapshot.is_some() {
        m.ssm_snapshot_tokens
    } else if m.ssm_snapshot_tier_key.is_some() {
        m.ssm_snapshot_tier_tokens
    } else {
        0
    }
}

impl TransformerModel {
    /// 2026-10-03: The grid size when the absolute grid applies to this model, with KV block
    /// size `bs` (passed so callers that hold the `kv_cache` lock need not take it again): the
    /// lever is on, the model has SSM layers, SSM snapshots and the prefix cache are on, the
    /// prefill is chunked (not a single-chunk MLA model), and `G` is valid for the block size and
    /// the prefill arena (`prefill_plan::grid_valid`). A set lever with an invalid `G` warns once
    /// and leaves the previous behaviour.
    pub(in crate::model) fn prefix_grid_active_bs(&self, bs: usize) -> Option<usize> {
        let g = crate::prefill_plan::grid_restore_lever()?;
        if self.config.num_ssm_layers() == 0
            || !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || self.is_mla_dispatch()
        {
            return None;
        }
        let arena = self.buffers.max_batch_tokens();
        if !crate::prefill_plan::grid_valid(g, bs, arena) {
            static W: std::sync::Once = std::sync::Once::new();
            W.call_once(|| {
                tracing::warn!(
                    "METRALE_PREFIX_GRID_RESTORE=1 ignored: grid {g} is not a positive multiple \
                     of the KV block size {bs} (and of 4) no larger than the prefill arena \
                     ({arena} tokens); set METRALE_PREFIX_GRID_TOKENS to --max-prefill-tokens"
                )
            });
            return None;
        }
        Some(g)
    }

    /// 2026-10-03: `prefix_grid_active_bs` for this prompt: `None` as well for a prompt with
    /// vision pads (it gets no prefix match or cache insert at all, so it keeps the previous
    /// plan).
    pub(in crate::model) fn prefix_grid_for_bs(&self, tokens: &[u32], bs: usize) -> Option<usize> {
        let g = self.prefix_grid_active_bs(bs)?;
        (!self.tokens_have_vision_pad(tokens)).then_some(g)
    }

    /// 2026-10-03: `prefix_grid_for_bs` with the block size read under the `kv_cache` lock; for
    /// callers that do not hold it (the scheduler's `Model::prefill_grid`).
    pub(in crate::model) fn prefix_grid_for(&self, tokens: &[u32]) -> Option<usize> {
        // 2026-10-03: Lever off: return before taking the lock.
        crate::prefill_plan::grid_restore_lever()?;
        let bs = self.kv_cache.lock().block_size();
        self.prefix_grid_for_bs(tokens, bs)
    }

    /// 2026-10-03: Release `m`'s radix refs (exactly its matched blocks).
    fn release_prefix_match(&self, tokens: &[u32], bs: usize, m: &PrefixMatch, adapter_id: u64) {
        if m.matched_tokens > 0 {
            self.prefix_cache
                .release_matched(tokens, bs, m.matched_tokens, adapter_id);
        }
    }

    /// 2026-10-03: Truncate a chunk-0 match to the grid restore point (module docs): the
    /// returned match is empty, or holds exactly `[0, k*G)` with a snapshot at `k*G` on this
    /// rank. Every intermediate match is released. On a multi-rank world (`ep_min`) every rank
    /// calls this once per chunk-0 lookup, and they adopt the minimum point; a rank whose own
    /// snapshot is not at that point ends empty and the vote then refuses the restore.
    pub(in crate::model) fn grid_floor_match(
        &self,
        tokens: &[u32],
        bs: usize,
        g: usize,
        seq: &SequenceState,
        first: PrefixMatch,
        ep_min: bool,
    ) -> Result<PrefixMatch> {
        let total = tokens.len();
        let (session, adapter) = (seq.session_hash, seq.adapter_id);
        let mut m = first;
        let start = (m.matched_tokens, match_snap_depth(&m));
        let point = crate::prefill_plan::grid_search(start.0, start.1, total, g, |n| {
            self.release_prefix_match(tokens, bs, &m, adapter);
            m = if n == 0 {
                PrefixMatch::empty()
            } else {
                self.prefix_cache.lookup(&tokens[..n], bs, session, adapter)
            };
            (m.matched_tokens, match_snap_depth(&m))
        });
        if ep_min {
            let agreed = self.ep_min_u32(point as u32)? as usize;
            if agreed < point {
                self.release_prefix_match(tokens, bs, &m, adapter);
                m = if agreed == 0 {
                    PrefixMatch::empty()
                } else {
                    self.prefix_cache
                        .lookup(&tokens[..agreed], bs, session, adapter)
                };
                if crate::prefill_plan::grid_step(
                    m.matched_tokens,
                    match_snap_depth(&m),
                    total,
                    g,
                ) != crate::prefill_plan::GridStep::Accept
                {
                    self.release_prefix_match(tokens, bs, &m, adapter);
                    m = PrefixMatch::empty();
                }
            }
            if agreed != point {
                tracing::info!(
                    "grid restore EP-sync: local={point} agreed={agreed} kept={}",
                    m.matched_tokens
                );
            }
        }
        if start.0 > 0 {
            tracing::info!(
                "grid restore: matched {} (snapshot {}) of {total} -> restore point {} (G={g})",
                start.0,
                start.1,
                m.matched_tokens,
            );
        }
        Ok(m)
    }
}
