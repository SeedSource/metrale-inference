// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Hybrid MTP catch-up at the DFlash -> MTP switch (`METRALE_HYBRID_MTP_CATCHUP=1`,
//! #82 hybrid B2(i)), and the `HYBRID MTP resume` diagnostic every hybrid serve logs.
//!
//! On a hybrid serve the MTP drafter state is untouched while its sequence takes DFlash steps, so
//! at the switch the drafter lacks every token the stretch committed and its next rows land at
//! the stale row count (wrong RoPE positions). With the lever, every hybrid DFlash verify copies
//! its verify rows into a ring ([`TransformerModel::hybrid_mtp_ring_stage`]), the commit marks
//! the accepted ones covered ([`TransformerModel::hybrid_mtp_ring_commit`]), and the sequence's
//! first MTP propose after the stretch feeds them to the drafter
//! ([`TransformerModel::hybrid_mtp_resume`]). The bookkeeping and the plan are
//! `metrale_model_layers::speculative::hybrid_ring` (pure, CPU-tested).
//!
//! Every rank runs all three, with the same inputs: the stage from the K=gamma verify wrapper
//! (`impl_verify.rs`; rank 0 through `decode_verify_dflash`, the worker in
//! `EP_CMD_VERIFY_KGAMMA`), the commit from `commit_accepted_prefix` (`impl_ssm_state.rs`;
//! rank 0 from `step_verify_dflash`, the worker in the same arm), the resume right after
//! `ensure_drafter_context` in every MTP propose path (`run_mtp_propose_inner`, both batched
//! propose paths), which the worker mirrors. A drafter context row needs no collective (a norm,
//! the `eh_proj` GEMV and the DSA cache writes), so each rank feeds its own mirrored drafter.
//!
//! Owner: model-engine (speculative).
//! Invariants:
//! - Off a hybrid serve nothing here runs. With the lever off only the host-side `resume` flag
//!   and the diagnostic run: no device memory, no copy, no drafter call.
//! - Every decision is taken from inputs both ranks hold (the ring meta, `seq.tokens`,
//!   `seq.seq_len`, the drafter's row count), so both ranks end a resume with the same drafter
//!   row count. A device error is returned or logged like any other on that path.

use anyhow::Result;
use metrale_model_layers::speculative::hybrid as hy;
use metrale_model_layers::speculative::hybrid_ring::{self as ring, CatchupPlan, Skip};
use metrale_model_layers::speculative::{DraftProposer, ProposerState};

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-10-14: Where a catch-up step reads its hidden rows.
#[derive(Clone, Copy)]
enum Src {
    /// The ring rows of the step's keys.
    Ring,
    /// The zeroed rows (gap fill).
    Zero,
}

impl TransformerModel {
    /// 2026-10-14: Allocate the ring and the zero rows (`set_hybrid_dflash_proposer`, every rank,
    /// lever on). `kv_budget::dflash_reserve_bytes` reserved the same bytes
    /// (`hy::hybrid_mtp_catchup_bytes`) before KV sizing.
    pub(super) fn hybrid_mtp_ring_alloc(&mut self) -> Result<()> {
        if !self.hybrid_mtp_ring.is_null() {
            return Ok(());
        }
        let rows = hy::hybrid_mtp_catchup_rows();
        let h = self.config.hidden_size;
        let ring_ptr = self.gpu.alloc(rows * h * 2)?;
        let zero_bytes = ring::ZERO_ROWS * h * 2;
        let zero_ptr = self.gpu.alloc(zero_bytes)?;
        self.gpu.memset(zero_ptr, 0, zero_bytes)?;
        self.hybrid_mtp_ring = ring_ptr;
        self.hybrid_mtp_zero_rows = zero_ptr;
        self.hybrid_mtp_ring_rows = rows;
        tracing::info!(
            "HYBRID MTP CATCHUP: ENGAGED — ring of {rows} hidden rows plus {} zero rows \
             ({:.1} MB); the first MTP propose after a DFlash stretch feeds the stretch to the MTP \
             drafter",
            ring::ZERO_ROWS,
            hy::hybrid_mtp_catchup_bytes(true, rows, h) as f64 / 1e6,
        );
        Ok(())
    }

    /// 2026-10-14: A new sequence on `slot`, or a sequence moved off or onto it.
    pub(super) fn hybrid_mtp_ring_reset_slot(&self, slot: usize) {
        self.hybrid_mtp_ring_meta.lock().reset_slot(slot);
    }

    /// 2026-10-14: After a K=gamma verify of `k` rows on a hybrid DFLASH step (every rank, right
    /// after the forward, while `hidden_states` still holds its `k` rows): copy them into the
    /// ring. Lever off: only note the DFlash step for the diagnostic.
    ///
    /// Labels, from the K2 verify's ring write (`verify_k2_step.rs`: row `t` of a verify whose
    /// pre-verify length is `L = seq_len - 2` is labelled `L + t + 1`): the verify forward has
    /// advanced `seq_len` by `k`, so `L = seq_len - k`, and row `t` (the input token at position
    /// `L + t`, whose output is `hidden_{L+t}`) goes to label `L + t + 1`, which is the convention
    /// "label `n` holds `hidden_{n-1}`". The commit then keeps rows `0..=num_accepted`.
    pub(super) fn hybrid_mtp_ring_stage(&self, seq: &SequenceState, k: usize) -> Result<()> {
        if !self.hybrid_on() || !self.dflash_step() {
            return Ok(());
        }
        let slot = seq.slot_idx;
        let mut meta = self.hybrid_mtp_ring_meta.lock();
        if self.hybrid_mtp_ring.is_null() || k == 0 || seq.seq_len < k {
            meta.note_dflash(slot);
            return Ok(());
        }
        let base = seq.seq_len - k + 1;
        let pieces = meta.stage(slot, base, k, self.hybrid_mtp_ring_rows);
        drop(meta);
        let h = self.config.hidden_size;
        let stream = self.gpu.default_stream();
        let src = self.buffers.hidden_states();
        for p in pieces {
            self.gpu.copy_d2d_async(
                src.offset(p.offset * h * 2),
                self.hybrid_mtp_ring.offset(p.ring_row * h * 2),
                p.n * h * 2,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-10-14: `commit_accepted_prefix(seq, num_accepted, k)` on a hybrid DFLASH step (every
    /// rank): the staged verify's first `num_accepted` labels become covered (`RingMeta::commit`).
    pub(super) fn hybrid_mtp_ring_commit(
        &self,
        seq: &SequenceState,
        num_accepted: usize,
        k: usize,
    ) {
        if !self.hybrid_on() || !self.dflash_step() {
            return;
        }
        self.hybrid_mtp_ring_meta
            .lock()
            .commit(seq.slot_idx, seq.seq_len, num_accepted, k);
    }

    /// 2026-10-14: Before an MTP propose at `position` (every rank, after
    /// `ensure_drafter_context`): on the slot's first MTP propose after it took DFlash steps,
    /// catch the MTP drafter up (lever on; `hybrid_ring::plan_catchup`) and log (rank 0) `HYBRID
    /// MTP resume slot=S position=P drafter_rows=R missing=M caught_up=C ...`, where `M` is the
    /// keys below `position - 1` the drafter lacked and `C` the rows fed from the ring (0 with
    /// the lever off). A no-op on any other propose.
    pub(in crate::model) fn hybrid_mtp_resume(
        &self,
        proposer: &dyn DraftProposer,
        seq: &mut SequenceState,
        position: usize,
        stream: u64,
    ) {
        if !self.hybrid_on() || self.dflash_step() {
            return;
        }
        let slot = seq.slot_idx;
        let covered = {
            let mut meta = self.hybrid_mtp_ring_meta.lock();
            if !meta.take_resume(slot) {
                return;
            }
            meta.covered_keys(slot)
        };
        let SequenceState {
            tokens,
            proposer_state,
            ..
        } = seq;
        let Some(state) = proposer_state.as_mut() else {
            return;
        };
        let rows_before = proposer.drafter_rows(state.as_mut());
        let lever = !self.hybrid_mtp_ring.is_null();
        let mut plan = plan_or_skip(lever, rows_before, position, covered, tokens.len());
        let (mut caught_up, mut filled, mut rewound) = (0usize, 0usize, 0usize);
        let mut note = "";
        // 2026-10-14: A plan without a skip passed `plan_or_skip`'s checks, so the no-rewind
        // re-plan may use `covered` as is.
        if plan.skip.is_none()
            && let Some(to) = plan.rewind_to
        {
            match proposer.rewind_drafter_to(state.as_mut(), to) {
                Ok(true) => rewound = rows_before - to,
                Ok(false) => {
                    plan = ring::plan_catchup(rows_before, position, covered, false);
                }
                Err(e) => {
                    tracing::warn!("HYBRID MTP resume: rewind failed ({e:#}); not fed");
                    plan.skip = Some(Skip::Current);
                    note = " rewind_failed";
                }
            }
        }
        if plan.skip.is_none() {
            let ctx = self.mtp_propose_ctx();
            let steps = [
                (plan.head_fill, Src::Zero),
                (plan.feed, Src::Ring),
                (plan.tail_fill, Src::Zero),
            ];
            for (keys, src) in steps {
                let Some((lo, hi)) = keys else {
                    continue;
                };
                let (w, done) = self.hybrid_feed_keys(
                    proposer,
                    state.as_mut(),
                    tokens.as_slice(),
                    lo,
                    hi,
                    src,
                    &ctx,
                    stream,
                );
                match src {
                    Src::Ring => caught_up += w,
                    Src::Zero => filled += w,
                }
                if !done {
                    note = " short_write";
                    break;
                }
            }
        }
        let rank0 = match self.comm.as_ref() {
            Some(c) => c.rank() == 0,
            None => true,
        };
        if rank0 {
            let rows_after = proposer.drafter_rows(state.as_mut());
            tracing::info!(
                "HYBRID MTP resume slot={slot} position={position} drafter_rows={rows_before} \
                 missing={} caught_up={caught_up} rewound={rewound} zero_filled={filled} \
                 rows_after={rows_after} no_drafter_prefill={} lever={} ring_keys={} skip={}{note}",
                plan.missing,
                u8::from(rows_before == 0),
                if lever { "on" } else { "off" },
                covered.map_or("none".to_string(), |(lo, hi)| format!("[{lo},{hi})")),
                plan.skip.map_or("none", skip_name),
            );
        }
    }

    /// 2026-10-14: Feed pair keys `lo..hi` to the drafter, which holds exactly `lo` rows, from
    /// `src`: ring pieces (`split_wrap` of labels `lo + 1 ..`) or zero-row chunks. Row `r` of a
    /// chunk at key `k0` is `(embed(tokens[k0 + r + 1]), hidden row r)`, so the chunk passes
    /// `tokens[k0 ..= k0 + n]`. Returns the rows written and whether all were.
    #[allow(clippy::too_many_arguments)]
    fn hybrid_feed_keys(
        &self,
        proposer: &dyn DraftProposer,
        state: &mut dyn ProposerState,
        tokens: &[u32],
        lo: usize,
        hi: usize,
        src: Src,
        ctx: &metrale_model_layers::layer::ForwardContext,
        stream: u64,
    ) -> (usize, bool) {
        let h = self.config.hidden_size;
        let chunks: Vec<(usize, metrale_gpu_runtime::gpu::DevicePtr, usize)> = match src {
            Src::Ring => ring::split_wrap(lo + 1, hi - lo, self.hybrid_mtp_ring_rows)
                .into_iter()
                .map(|p| {
                    (
                        lo + p.offset,
                        self.hybrid_mtp_ring.offset(p.ring_row * h * 2),
                        p.n,
                    )
                })
                .collect(),
            Src::Zero => (lo..hi)
                .step_by(ring::ZERO_ROWS)
                .map(|k0| {
                    (
                        k0,
                        self.hybrid_mtp_zero_rows,
                        (hi - k0).min(ring::ZERO_ROWS),
                    )
                })
                .collect(),
        };
        let mut written = 0;
        for (k0, hid, n) in chunks {
            if k0 + n >= tokens.len() + 1 {
                return (written, false);
            }
            let toks = &tokens[k0..=k0 + n];
            match proposer.catchup_drafter(toks, hid, k0, k0 + 1, state, ctx, stream) {
                Ok(w) if w == n => written += w,
                Ok(w) => return (written + w, false),
                Err(e) => {
                    tracing::warn!("HYBRID MTP resume: drafter feed failed ({e:#})");
                    return (written, false);
                }
            }
        }
        (written, true)
    }
}

/// 2026-10-14: The resume's plan: [`ring::plan_catchup`] with rewind, or a skip with the lever
/// off (`Skip::NoCoverage`: there is no ring) or when `seq.tokens` (`n_tokens`) lacks a token a
/// fed key needs (key `position - 2` reads `tokens[position - 1]`).
fn plan_or_skip(
    lever: bool,
    drafter_rows: usize,
    position: usize,
    covered: Option<(usize, usize)>,
    n_tokens: usize,
) -> CatchupPlan {
    let covered = if lever && n_tokens >= position {
        covered
    } else {
        None
    };
    ring::plan_catchup(drafter_rows, position, covered, true)
}

/// 2026-10-14: Log name of a [`Skip`].
fn skip_name(s: Skip) -> &'static str {
    match s {
        Skip::NoDrafterPrefill => "no_drafter_prefill",
        Skip::NoCoverage => "no_coverage",
        Skip::FillCap => "fill_cap",
        Skip::Current => "current",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-14: Lever off: the resume plan writes nothing, whatever the drafter and position
    /// (the lever-off path is the diagnostic alone).
    #[test]
    fn lever_off_plans_nothing() {
        for (rows, pos) in [(0usize, 10usize), (5, 500), (499, 500), (102, 151)] {
            let p = plan_or_skip(false, rows, pos, Some((101, 150)), pos);
            assert!(p.skip.is_some(), "{p:?}");
            assert_eq!(
                (p.rewind_to, p.head_fill, p.feed, p.tail_fill),
                (None, None, None, None)
            );
            assert_eq!(p.missing, (pos - 1).saturating_sub(rows));
        }
    }

    /// 2026-10-14: Lever on: the switch plan, and a `seq.tokens` too short to name the last key's
    /// token refuses the feed on every rank alike.
    #[test]
    fn lever_on_plans_the_feed_only_with_the_tokens_it_reads() {
        let p = plan_or_skip(true, 102, 151, Some((101, 150)), 151);
        assert_eq!((p.rewind_to, p.feed), (Some(101), Some((101, 150))));
        let p = plan_or_skip(true, 102, 151, Some((101, 150)), 150);
        assert_eq!(p.skip, Some(Skip::NoCoverage));
    }
}
