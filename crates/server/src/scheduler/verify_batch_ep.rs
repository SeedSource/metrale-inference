// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Rank 0's half of the multi-rank batched-verify handshake after the forward,
//! shared by the batched MTP step (`verify_k4_batch_step`) and the batched DFlash step
//! (`verify_dflash_batch_step`): the verdict words, then the failure-mask gather
//! (model-engine `verify_ep.rs` module docs for the worker's side).
//!
//! Owner: scheduler.
//! Invariants:
//! - Without EP (`Model::is_ep` false) both helpers do nothing and send nothing.
//! - Once [`broadcast_batch_verdict`] returned true, the caller makes exactly one
//!   [`retire_failed_on_workers`] call on every path that follows, before any propose: the
//!   workers block on that gather after real verdict words.

use super::*;

/// 2026-10-09: Multi-rank: send the batch's verdict words (sequence i's accepted drafts,
/// clamped to `ks[i] - 1`, batch order), which the workers wait for after the forward. Returns
/// false, with every sequence finished, when the send fails; the caller then returns without
/// the gather (the workers' read failed with it). True without EP.
pub(super) fn broadcast_batch_verdict(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    accepted: &[usize],
    ks: &[usize],
) -> bool {
    if !model.is_ep() {
        return true;
    }
    let words =
        metrale_model_layers::speculative::glm_dflash_batch::batch_verdict_words(accepted, ks);
    if let Err(e) = model.ep_broadcast_tokens(&words) {
        tracing::error!(
            "EP broadcast batched-verify verdict (n={}): {e:#}",
            batch.len()
        );
        for a in batch.iter_mut() {
            a.finished = true;
        }
        return false;
    }
    true
}

/// 2026-10-02: Multi-rank batched verify: join the workers' commit-failure gather
/// (`Model::ep_agree_verify_failures`, a no-op without EP) with `local_mask`, and finish every
/// sequence whose bit (batch order) any rank set. Must run on every path that follows the
/// verdict broadcast, since the workers block on the gather.
pub(super) fn retire_failed_on_workers(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    local_mask: u32,
) {
    if !model.is_ep() {
        return;
    }
    match model.ep_agree_verify_failures(local_mask) {
        Ok(mask) => {
            for (i, a) in batch.iter_mut().enumerate().take(32) {
                if (mask >> i) & 1 == 1 {
                    tracing::error!("batched verify: worker commit failed, retiring seq {i}");
                    a.finished = true;
                }
            }
        }
        Err(e) => {
            tracing::error!("EP batched-verify failure gather: {e:#}");
            for a in batch.iter_mut() {
                a.finished = true;
            }
        }
    }
}
