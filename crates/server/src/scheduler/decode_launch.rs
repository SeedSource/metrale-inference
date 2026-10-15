// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The launch half of the plain decode step: the KV-exhaustion retry loop
//! (`launch_with_preemption`), which `decode_batch_with_preemption` awaits at
//! once and the pipelined lane keeps in flight, the decode-preemption victim
//! policy, and the readback flavour and DFlash context commit a launch plans.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;
use crate::scheduler::io::{DecodeCtxCommit, DeviceError, Readback, SchedIo, SpillIo, StepPlan};
use metrale_scheduler::Ticket;

/// 2026-09-25: How a plain decode step's argmax readback is planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FeedReadback {
    /// 2026-09-25: The readback `decode_readback_plan` picks, unchanged.
    Plain,
    /// 2026-09-25: The masked feed argmax: the router runs ahead of the host, so a masked
    /// id must be re-picked on the device rather than by a host round trip.
    Masked { think_end_token: Option<u32> },
}

/// 2026-09-25: A resumed sequence is passed over as a decode-preemption victim until it
/// has generated this many more tokens, unless every candidate is immune (see
/// [`choose_decode_victim`]).
pub(super) const PREEMPT_IMMUNITY_TOKENS: usize = 64;

/// 2026-09-25: Pick the decode-preemption victim.
///
/// Policy: fewest generated tokens, ties broken by the lower slot.
///
/// Exclusions:
///   * finished sequences (2026-10-14);
///   * sequences whose tokens contain vision pads, unless they can be spilled (their KV
///     came from image embeddings a token re-prefill cannot reproduce). A grammar-active
///     sequence is never spilled (`SwappedSeq` does not carry its matcher), so it is a
///     victim only when it can be requeued (2026-10-15: before, it was never a victim);
///   * immune sequences (see [`PREEMPT_IMMUNITY_TOKENS`]), unless every
///     eligible candidate is immune.
pub(super) fn choose_decode_victim(
    model: &dyn Model,
    active: &[ActiveSeq],
    can_spill: bool,
) -> Option<usize> {
    // 2026-10-14 (race-memory, A184): a sequence that already finished this step is never a
    // victim (`preempt_after_spec_refusal` runs before finished rows are retired); requeuing
    // it would resume a reply that has ended.
    // 2026-10-15 (race-memory, CAPACITY.md step-2 condition c): a grammar-active row is
    // requeued, never spilled: the requeue keeps the whole `ActiveSeq`, matcher included, and
    // the matcher has accepted exactly the committed tokens (decode advances it after
    // sampling; verify after the KV refusal point).
    let eligible = |a: &ActiveSeq| {
        !a.finished
            && ((can_spill && a.grammar_state.is_none())
                || !model.tokens_contain_vision_pad(&a.seq.tokens))
    };
    let pick = |immune_ok: bool| {
        active
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                eligible(a) && (immune_ok || a.output_tokens.len() >= a.preempt_immune_until_tokens)
            })
            .min_by_key(|(_, a)| (a.output_tokens.len(), a.seq.slot_idx))
            .map(|(i, _)| i)
    };
    pick(false).or_else(|| pick(true))
}

/// 2026-09-25: Send `e` to every row and drain them. Unlike the forward-error path in
/// [`launch_with_preemption`], this does not check the fault latch.
pub(super) fn fail_readback(io: &SchedIo, active: &mut Vec<ActiveSeq>, e: &anyhow::Error) {
    tracing::error!("argmax_batch error: {e:#}");
    for mut a in active.drain(..) {
        send_error(io, &mut a, &format!("{e:#}"));
    }
}

/// 2026-09-25: Which DFlash context commit a plain decode step of `n` rows performs.
pub(super) fn decode_ctx_commit(n: usize, levers: &super::levers::SchedLevers) -> DecodeCtxCommit {
    if levers.dflash_unified_ctx {
        DecodeCtxCommit::Unified
    } else if n == 1 && levers.dflash_serial_append {
        DecodeCtxCommit::SerialAppend
    } else {
        DecodeCtxCommit::None
    }
}

/// 2026-09-25: Launch a plain decode step, preempting a victim and retrying while the
/// pool reports exhaustion. Returns the ticket unawaited, so a pipelining
/// caller can keep the step in flight; `None` when every row was failed
/// instead. `feed` decides the argmax readback's flavour.
pub(super) fn launch_with_preemption(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    active: &mut Vec<ActiveSeq>,
    spill: Option<&dyn SpillIo>,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
    staging: &mut Vec<u8>,
    feed: FeedReadback,
) -> Option<Ticket> {
    let io = &sched.io;
    let model = io.dev.model();
    loop {
        let tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();
        let readback = match (
            super::decode_logits_step::decode_readback_plan(active, sched, staging),
            feed,
        ) {
            (Readback::Argmax, FeedReadback::Masked { think_end_token }) => {
                Readback::ArgmaxMasked {
                    masks: active
                        .iter()
                        .map(|a| super::core::pipeline::row_mask(a, think_end_token))
                        .collect(),
                }
            }
            (planned, _) => planned,
        };
        let ctx = decode_ctx_commit(active.len(), &sched.levers);
        let mut refs: Vec<&mut SequenceState> = active.iter_mut().map(|a| &mut a.seq).collect();
        let launched = io.dev.launch(
            StepPlan::Decode {
                tokens,
                ctx,
                readback,
            },
            &mut refs,
        );
        match launched {
            Ok(t) => return Some(t),
            Err(DeviceError::Readback(e)) | Err(DeviceError::Effect(e)) => {
                drop(refs);
                fail_readback(io, active, &e);
                return None;
            }
            Err(err) => {
                drop(refs);
                let e = err.into_inner();
                let victim = if format!("{e:#}").contains("KV cache exhausted") && active.len() > 1
                {
                    choose_decode_victim(model, active, spill.is_some())
                } else {
                    None
                };
                let Some(vi) = victim else {
                    tracing::error!("decode_batch error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(io, &mut a, &format!("{e:#}"));
                    }
                    // 2026-09-25: If the process-wide fault latch reports a lost CUDA
                    // context, request shutdown: every later request would fail
                    // the same way. `request` is idempotent, so repeated failures
                    // do not repeat it.
                    if let Some(reason) = metrale_core::fault::global().fault() {
                        crate::tui::shutdown::request(reason);
                    }
                    return None;
                };
                // 2026-09-25: `remove`, not `swap_remove`, keeps the ascending-slot order
                // the caller sorted `active` into.
                let v = active.remove(vi);
                tracing::warn!(
                    "KV cache exhausted during decode: preempting slot={} \
                     ({} blocks, {} tokens generated) for later RESUME so the \
                     other {} sequence(s) can continue",
                    v.seq.slot_idx,
                    v.seq.block_table.len(),
                    v.output_tokens.len(),
                    active.len(),
                );
                // 2026-10-15: a grammar-active victim is requeued, never spilled.
                match spill.filter(|_| v.grammar_state.is_none()) {
                    Some(sp) => match super::preempt::spill_out_sequence(io, v, sp) {
                        Ok(s) => swapped.push(s),
                        Err((v, spill_err)) => {
                            tracing::warn!(
                                "decode-preempt spill failed ({spill_err:#}); \
                                 requeuing victim for re-prefill instead"
                            );
                            preempted.push(super::preempt::preempt_requeue(io, v));
                        }
                    },
                    None => preempted.push(super::preempt::preempt_requeue(io, v)),
                }
            }
        }
    }
}
