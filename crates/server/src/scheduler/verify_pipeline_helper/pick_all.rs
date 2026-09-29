// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `verify_pick_all_with_pipeline`: picks every verify position
//! of one sequence, through a GPU-argmax fast path when one applies and the
//! host pipeline otherwise.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-29: the verify-time analogue of `SchedCtx::think_mask_fallbacks`
/// (decode_logits_step.rs): counts calls to
/// [`verify_pick_all_with_pipeline`] where the post-think structural guard
/// (A143 sibling, [`fast_hits_post_think_structural`]) turned off the
/// GPU-argmax fast paths and forced the host pipeline. Write-only
/// diagnostic counter.
static VERIFY_THINK_MASK_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 2026-09-29: A144: counts calls to [`verify_pick_all_with_pipeline`] where
/// a non-empty decode-effective `logit_bias`
/// (`sample_step::speculative_bias_forces_host`) turned off the
/// GPU-argmax fast paths and forced the host pipeline. Write-only
/// diagnostic counter.
pub(crate) static VERIFY_BIAS_HOST_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 2026-09-29: true when thinking has already closed (`think_ended`) and any
/// verify-window GPU argmax is the `</think>` or `<think>` id (A143 sibling).
///
/// Mirrors the single-row decode guard in `decode_logits_step.rs`
/// (`a.think_ended && (Some(tok) == think_end_token || Some(tok) ==
/// a.think_start_token)`): once thinking has ended, the host pipeline's
/// `PostCloseThinkMask` masks both ids so the runner-up wins, while the
/// grammar and grammarless fast paths below return the raw GPU argmax. It
/// scans every position of the verify window rather than one token.
///
/// O(K) over `argmax_ids`, which are already on the host; no extra D2H.
pub(super) fn fast_hits_post_think_structural(
    think_ended: bool,
    argmax_ids: &[u32],
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
) -> bool {
    think_ended
        && argmax_ids
            .iter()
            .any(|&tok| Some(tok) == think_end_token || Some(tok) == think_start_token)
}

/// 2026-09-25: pick a token for each of one sequence's verify positions.
///
/// `argmax_ids` holds the GPU argmax of each position and sets K. The fast
/// paths, tried in order, return `argmax_ids` unchanged without copying the
/// logits rows: the masked chat path (`fast_masked`), the grammar path and
/// the grammarless path below. Otherwise the K rows are copied to host and
/// `pick_positions_from_host` picks them; it returns fewer than K picks when
/// a speculative grammar advance is refused. If that copy fails,
/// `argmax_ids` is returned.
///
/// `row_base` is the sequence's first row in the shared logits buffer: 0 on
/// the single-sequence paths, the prefix sum of the earlier sequences' row
/// counts on `step_verify_k4_batched`.
pub fn verify_pick_all_with_pipeline(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> Vec<u32> {
    use crate::scheduler::mtp_timing::Phase;
    let k = argmax_ids.len();
    if k == 0 {
        return Vec::new();
    }

    // 2026-09-29: A144 logit-bias guard. The GPU-argmax fast paths below
    // never see `logit_bias`, and a bias can raise a competitor above the raw
    // argmax (the tools-active `<tool_call>` +3.0 nudge does). When decode
    // would apply a non-empty bias to this row, force the host pipeline,
    // where `verify_pick_with_pipeline` applies it per position. When decode
    // would itself take its device argmax (bias skipped), the fast paths stay
    // legal: parity with decode, not "always apply".
    let bias_forces_host = crate::scheduler::sample_step::speculative_bias_forces_host(
        a,
        ctx.sampling.think_ended_gpu_argmax,
    );
    if bias_forces_host {
        VERIFY_BIAS_HOST_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // 2026-09-25: the masked chat path; its gates are in `fast_masked.rs`.
    if !bias_forces_host
        && let Some(picks) = fast_masked::try_chat_fast_path(model, argmax_ids, a, ctx, row_base)
    {
        return picks;
    }

    // 2026-09-29: post-think structural guard (A143 sibling). The decode
    // path falls back to the host pipeline when a `think_ended` argmax lands
    // on `</think>`/`<think>`, so `PostCloseThinkMask` masks both ids and the
    // runner-up wins. The grammar and grammarless fast paths below had no
    // such check and returned the structural id unmasked; a hit now forces
    // the host path (`pick_positions::pick_positions_from_host`), which runs
    // `process_position_logits`, `PostCloseThinkMask` included, per
    // position. The masked chat path above already refuses both ids.
    let think_structural_hit = fast_hits_post_think_structural(
        a.think_ended,
        argmax_ids,
        ctx.think_end_token,
        a.think_start_token,
    );
    if think_structural_hit {
        VERIFY_THINK_MASK_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    // 2026-09-25: grammar fast path. Eligible when `fast_greedy_grammar` is on
    // (`METRALE_DISABLE_FAST_GREEDY=1` turns it off), a grammar is active,
    // the sequence is outside thinking, decoding is greedy (temperature 0 or
    // `force_temp_zero`), and the penalties are not `Blocked`. Each position's
    // GPU argmax must be grammar-allowed (a grammar-allowed global maximum is
    // the maximum of the allowed set) and, for `ReduceOnly` penalties,
    // penalty-immune (`fast_greedy`). The matcher is advanced speculatively
    // between positions and rolled back afterwards, as the host path does.
    //
    // The other pipeline stages are not consulted: `MinTokensEosMask`,
    // `PostCloseThinkMask`, `PinToToolCallStart` and the tool-call bias can
    // still apply outside thinking, so this path can emit a token the host
    // path would have masked. Temperature above 0 always takes the host path,
    // where the sampling branch runs.
    let fast_penalty_gate = if ctx.sampling.fast_greedy_grammar
        && a.grammar_state.is_some()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
                ctx.watchdog.min_reasoning_floor,
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if fast_penalty_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked
        && !think_structural_hit
        && !bias_forces_host
    {
        let t_fast = ctx.clock.now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        // 2026-09-25: the same history scope the host path penalises
        // (`penalty_history_scope`), copied before `a.grammar_state` is
        // borrowed mutably.
        let scoped_history: Vec<u32> =
            if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
        let mut fast: Vec<u32> = Vec::with_capacity(k);
        let mut all_allowed = true;
        // 2026-09-25: the block ends `gs`'s mutable borrow before the
        // rollback below borrows `a.grammar_state` again.
        {
            let Some(gs) = a.grammar_state.as_mut() else {
                unreachable!("grammar_state present (gated by is_some above)")
            };
            for (i, &tok) in argmax_ids.iter().enumerate() {
                // 2026-09-25: `ReduceOnly`: the argmax must be absent from the
                // scoped history and have a raw logit above 0.
                if fast_penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly
                    && !crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                        crate::scheduler::fast_greedy::logit_is_positive(
                            model,
                            logits_base,
                            row_base + i,
                            vocab,
                            tok,
                        )
                    })
                {
                    all_allowed = false;
                    break;
                }
                let allowed = if gs.is_terminated() {
                    true
                } else {
                    gs.fill_bitmask();
                    gs.is_token_allowed(tok)
                };
                if !allowed {
                    all_allowed = false;
                    break;
                }
                fast.push(tok);
                // 2026-09-25: advance speculatively so position i+1 is
                // checked against the matcher state after pick i.
                if i + 1 < k && !gs.is_terminated() {
                    let _ = gs.accept_token(tok);
                }
            }
        }
        // 2026-09-25: roll back by the history delta, not by the number of
        // `accept_token` calls: `GrammarState::accept_token` returns true
        // for stop tokens and in the terminated state without adding a
        // history step.
        if let (Some(b), Some(gs)) = (before, a.grammar_state.as_mut()) {
            let adv = gs.num_history_steps().saturating_sub(b);
            if adv > 0 {
                gs.rollback(adv);
            }
        }
        ctx.tel.mark(Phase::FastGreedy, t_fast);
        if all_allowed && fast.len() == k {
            return fast;
        }
    }

    // 2026-09-25: grammarless fast path. Eligible when `fast_greedy_chat` is
    // on (`METRALE_NO_FAST_GREEDY_CHAT=1` turns it off), no grammar is
    // active, the sequence is outside thinking, decoding is greedy, and the
    // penalties are `Neutral`, or `ReduceOnly` with every argmax
    // penalty-immune. No mask stage is consulted (see the grammar path
    // above). The GPU argmax can also break near-ties differently from the
    // host scan, so the picks are not always the host path's.
    let chat_fast_gate = if ctx.sampling.fast_greedy_chat
        && a.grammar_state.is_none()
        && !a.inside_thinking
        && (a.temperature == 0.0 || ctx.sampling.force_temp_zero)
    {
        crate::scheduler::fast_greedy::classify_penalties(
            &crate::scheduler::sample_step::penalty_params_for(
                a,
                crate::scheduler::sample_step::PositionKind::Verify,
                0.0,
                None,
                Vec::new(),
                ctx.watchdog.min_reasoning_floor,
            ),
        )
    } else {
        crate::scheduler::fast_greedy::PenaltyGate::Blocked
    };
    if chat_fast_gate != crate::scheduler::fast_greedy::PenaltyGate::Blocked
        && !think_structural_hit
        && !bias_forces_host
    {
        let t_fast = ctx.clock.now();
        let vocab = model.vocab_size();
        let logits_base = model.logits_buffer_ptr();
        let scoped_history: Vec<u32> =
            if chat_fast_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
                crate::scheduler::sample_step::penalty_history_scope(
                    &a.output_tokens,
                    ctx.tool_call_end_token,
                )
                .to_vec()
            } else {
                Vec::new()
            };
        let all_immune = argmax_ids.iter().enumerate().all(|(i, &tok)| {
            chat_fast_gate == crate::scheduler::fast_greedy::PenaltyGate::Neutral
                || crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                    crate::scheduler::fast_greedy::logit_is_positive(
                        model,
                        logits_base,
                        row_base + i,
                        vocab,
                        tok,
                    )
                })
        });
        ctx.tel.mark(Phase::FastGreedy, t_fast);
        if all_immune {
            return argmax_ids.to_vec();
        }
    }

    let vocab = model.vocab_size();
    // 2026-09-25: the rows are read as BF16; this path does not check
    // `logits_ptr_is_fp32`.
    let elem_bytes = 2usize;
    let total = k * vocab * elem_bytes;
    let t_d2h = ctx.clock.now();
    let mut buf = vec![0u8; total];
    if model
        .copy_logits_to_host(
            model
                .logits_buffer_ptr()
                .offset(row_base * vocab * elem_bytes),
            &mut buf,
        )
        .is_err()
    {
        return argmax_ids.to_vec();
    }
    ctx.tel.mark(Phase::D2h, t_d2h);

    pick_positions::pick_positions_from_host(&buf, vocab, elem_bytes, k, a, ctx)
}

#[cfg(test)]
mod tests {
    use super::fast_hits_post_think_structural;

    const THINK_END: u32 = 100;
    const THINK_START: u32 = 101;
    const HELLO: u32 = 42;

    #[test]
    fn no_hit_when_think_not_ended() {
        // 2026-09-29: inside thinking the guard must not fire even on the
        // structural ids: `PostCloseThinkMask` does not apply yet.
        assert!(!fast_hits_post_think_structural(
            false,
            &[THINK_END, THINK_START],
            Some(THINK_END),
            Some(THINK_START),
        ));
    }

    #[test]
    fn no_hit_when_no_row_is_structural() {
        assert!(!fast_hits_post_think_structural(
            true,
            &[HELLO, HELLO, HELLO],
            Some(THINK_END),
            Some(THINK_START),
        ));
    }

    #[test]
    fn hits_on_think_end_reopen() {
        assert!(fast_hits_post_think_structural(
            true,
            &[HELLO, THINK_END, HELLO],
            Some(THINK_END),
            Some(THINK_START),
        ));
    }

    #[test]
    fn hits_on_think_start_reentry() {
        assert!(fast_hits_post_think_structural(
            true,
            &[THINK_START],
            Some(THINK_END),
            Some(THINK_START),
        ));
    }

    #[test]
    fn no_hit_when_tokens_are_not_configured() {
        // 2026-09-29: with neither id configured nothing can match.
        assert!(!fast_hits_post_think_structural(
            true,
            &[THINK_END, THINK_START],
            None,
            None,
        ));
    }

    #[test]
    fn empty_verify_window_never_hits() {
        assert!(!fast_hits_post_think_structural(
            true,
            &[],
            Some(THINK_END),
            Some(THINK_START),
        ));
    }
}
