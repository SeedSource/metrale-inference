// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Which DFlash verify rows join this step's cross-sequence batched verify
//! (`step_verify_dflash_batched`), and on what basis a batched row's drafts are judged.
//!
//! Owner: scheduler.
//! Invariants:
//! - Without the batched GLM DFlash pin (`DflashRung::multi_batched`, set only by
//!   `METRALE_GLM_DFLASH_BATCHED_VERIFY`) the split is the one `step_mtp` made before: a row
//!   is batched when it has no grammar, at least one draft, may take a raw-argmax verdict
//!   (A144/A146), and has the first such row's draft count; batched rows are judged on the raw
//!   argmax.
//! - Under the pin, every grammarless row with at least `num_drafts` drafts (this step's rung
//!   width) is batched at `gamma = num_drafts`; a longer block is truncated to it, as the MTP
//!   batch truncates to `ladder_nd`. Thinking and biased rows are batched too, and each batched
//!   row is judged on the basis `step_verify_dflash` uses for it ([`verdict_uses_pipeline`]).
//!
//! Measured 2026-10-09 (GLM-5.3 + DFlash2, 2 ranks, C=2/4, run ra-dec-dfbv-n1n2-dfbvval
//! c26dbv rank 0): the raw-argmax-only rule kept every row inside `<think>` out of the batch,
//! so a batch formed only once every row had closed its thinking: 301 batched steps (635 row
//! verifies) against 2985 per-sequence K=4 verifies.

use super::ActiveSeq;
use super::sched_ctx::SchedCtx;

/// 2026-10-09: The rows `step_mtp` verifies in one batch, the rows it verifies one by one, and
/// the batch's draft count (0 when nothing is batchable).
pub(super) struct DflashRowSplit {
    pub batchable: Vec<usize>,
    pub serial: Vec<usize>,
    pub gamma: usize,
}

/// 2026-10-09: Split `verify_idxs` (rows of `active` with pending drafts) for the DFlash batch.
/// Under the batched pin this truncates a batched row's drafts (and their confidences) to
/// `num_drafts`.
pub(super) fn split_verify_rows(
    active: &mut [ActiveSeq],
    verify_idxs: &[usize],
    sched: &SchedCtx,
    num_drafts: usize,
) -> DflashRowSplit {
    let mut out = DflashRowSplit {
        batchable: Vec::new(),
        serial: Vec::new(),
        gamma: 0,
    };
    let pinned = sched.dflash_rung.multi_batched();
    for &idx in verify_idxs {
        let a = &mut active[idx];
        let g = a.pending_drafts.len();
        if pinned {
            if a.grammar_state.is_none() && num_drafts >= 1 && g >= num_drafts {
                a.pending_drafts.truncate(num_drafts);
                a.pending_draft_conf.truncate(num_drafts);
                out.gamma = num_drafts;
                out.batchable.push(idx);
            } else {
                out.serial.push(idx);
            }
            continue;
        }
        // 2026-09-29: A144: the unpinned batched verdict is raw argmax only (no pipeline, no
        // `logit_bias`); a row whose decode-effective bias is non-empty takes the per-sequence
        // step, which routes it through the masked pipeline. A146: so does a thinking row.
        if a.grammar_state.is_some()
            || g < 1
            || crate::scheduler::sample_step::speculative_raw_argmax_forbidden(
                a,
                sched.levers.think_ended_gpu_argmax,
            )
        {
            out.serial.push(idx);
        } else if out.gamma == 0 || g == out.gamma {
            out.gamma = g;
            out.batchable.push(idx);
        } else {
            out.serial.push(idx);
        }
    }
    out
}

/// 2026-10-09: Whether `step_verify_dflash` judges `a`'s drafts through the logits pipeline
/// (`verify_pick_all_with_pipeline`) rather than on the raw argmax: when
/// `dflash_masked_verify` is on (the default), or when the row may not take a raw verdict
/// (A144 bias, A146 thinking). The batched step applies the same rule under the pin.
pub(super) fn verdict_uses_pipeline(sched: &SchedCtx, a: &ActiveSeq) -> bool {
    sched.levers.dflash_masked_verify
        || crate::scheduler::sample_step::speculative_raw_argmax_forbidden(
            a,
            sched.levers.think_ended_gpu_argmax,
        )
}
