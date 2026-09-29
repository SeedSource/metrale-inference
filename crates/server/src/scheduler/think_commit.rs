// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: A146, spec-in-think parity: the per-committed thinking-token
//! transition, shared by the three places that advance it.
//!
//! Tokens are committed on two paths: spec-off decode
//! (`decode_logits_step/per_token.rs`) and the MTP/DFlash verify-accept path
//! (`emit_step/token.rs`, `emit_token`). The K-position verify window
//! (`verify_pipeline_helper/pick_positions.rs`, `pick_positions_from_host`)
//! also advances the same state speculatively per position. The thinking
//! branch had drifted: `emit_token` never toggled `in_code_fence` and never
//! ran the thinking-loop watchdog, and the pick window never advanced
//! `thinking_tokens`. With speculation inside `<think>` each of those gaps
//! changes which token a later position picks, so a temperature-0 spec-on run
//! diverged from spec-off. [`advance_thinking_token`] is the one body all
//! three sites call.
//!
//! Owner: scheduler.
//! Invariants:
//! - `a.spec_think_trail` holds entries only between a verify pick window and
//!   the `emit_token` calls that commit it; any mismatch empties it.

use crate::scheduler::confidence::{MAX_SENTENCE_DEFER_TOKENS, toggle_code_fence};
use crate::scheduler::helpers::{
    THINK_LOOP_CHECK_STRIDE, THINK_LOOP_MIN_TOKENS, THINK_LOOP_PERIOD_MAX, THINK_LOOP_PERIOD_MIN,
    WatchdogParams, detect_thinking_token_loop_with,
};
use crate::scheduler::types::ActiveSeq;

/// 2026-09-29: A146: run-level inputs of [`advance_thinking_token`], the
/// same at every site.
#[derive(Clone, Copy)]
pub(crate) struct ThinkTokenEnv {
    /// 2026-09-29: the tokenizer's atomic ``` token (`None`: fence tracking
    /// is inert).
    pub code_fence_token: Option<u32>,
    /// 2026-09-29: `!disable_watchdogs && watchdog.enable_think_loop_watchdog`.
    pub think_loop_enabled: bool,
    pub watchdog: WatchdogParams,
}

/// 2026-09-29: A146: the commit-time transition for ONE token committed
/// inside `<think>` that is not `</think>`. That includes an EOS sampled
/// inside thinking: decode advances the thinking state for it before
/// dropping it.
///
/// `history_len` is how many `a.output_tokens` entries precede `tok`: the
/// thinking-loop scan must see exactly the history decode sees (decode runs
/// this before pushing `tok`; `emit_token` pushes first and passes
/// `len - 1`). `log` is false on the speculative pick window, which only
/// predicts.
pub(crate) fn advance_thinking_token(
    a: &mut ActiveSeq,
    tok: u32,
    history_len: usize,
    env: ThinkTokenEnv,
    log: bool,
) {
    a.thinking_tokens += 1;
    // 2026-09-29: track ``` code-fence parity inside thinking; a forced
    // `</think>` is deferred inside a fence (`should_inject_think_end`). The
    // thinking-loop watchdog below ignores fences.
    a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, env.code_fence_token);
    // 2026-09-29: budget exhausted: arm the forced `</think>`, which a later
    // position's pipeline injects.
    if let Some(budget) = a.thinking_budget
        && a.thinking_tokens >= budget
        && !a.force_end_thinking
    {
        a.force_end_thinking = true;
        a.sentence_defer_count = 0;
        if log {
            tracing::info!(target: "met::scheduler::decode_logits_step", source = if a.enable_thinking {
                    "request (client budget/effort; scaled by --max-thinking-budget)"
                } else {
                    "spontaneous <think> (--max-thinking-budget / MODEL.toml)"
                },
                "Thinking budget exhausted ({budget} tokens), arming </think>; \
                 deferring up to {MAX_SENTENCE_DEFER_TOKENS} tokens for sentence boundary"
            );
        }
    }
    // 2026-09-29: thinking-loop watchdog: every `THINK_LOOP_CHECK_STRIDE`
    // thinking tokens (from `THINK_LOOP_MIN_TOKENS` on), look for a repeating
    // tail and arm the forced `</think>`.
    if env.think_loop_enabled
        && !a.force_end_thinking
        && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
        && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
        && detect_thinking_token_loop_with(
            &a.output_tokens[..history_len.min(a.output_tokens.len())],
            a.repetition_detection,
            env.watchdog,
        )
    {
        a.force_end_thinking = true;
        a.sentence_defer_count = 0;
        a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
        if log {
            tracing::warn!(target: "met::scheduler::decode_logits_step", thinking_tokens = a.thinking_tokens,
                watchdog_fires = a.think_watchdog_fires,
                "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                THINK_LOOP_PERIOD_MIN,
                THINK_LOOP_PERIOD_MAX,
            );
        }
    }
}

/// 2026-09-29: A146: the thinking budget of a spontaneous `<think>`
/// re-entry: halved per earlier thinking-watchdog fire (at most 4 times, so
/// 1/16), floored at 8. Shared by decode and `emit_token`; the emit twin
/// used to reset to the undecayed budget.
pub(crate) fn spontaneous_think_budget(a: &ActiveSeq) -> u32 {
    let decay_shift = a.think_watchdog_fires.min(4);
    (a.spontaneous_think_budget >> decay_shift).max(8)
}

/// 2026-09-29: A146: the post-pipeline accumulator state the pick window
/// records per verify position, consumed by `emit_token` when that position
/// commits.
///
/// Inside `<think>` the logits pipeline mutates three per-sequence
/// accumulators: `consecutive_confident` and the `force_end_thinking` arming
/// (`F2ConfidenceEarlyStop`), and the `sentence_defer_count` tick
/// (`ForcedThinkEndInjector`). Spec-off decode runs the pipeline then
/// commits, once per token. The verify window runs the pipeline for all K
/// positions before it knows how many commit, so the sequence kept the K-th
/// position's accumulators even when fewer positions were accepted, and the
/// next window double-counted the rejected tail (defer ticks reached the
/// hard override early, F2 streaks over-counted). The window now restores
/// them and leaves this trail; each committing `emit_token` re-applies its
/// own position's entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpecThinkTrail {
    /// 2026-09-29: the pick at this position (the token `emit_token` must be
    /// committing).
    pub tok: u32,
    /// 2026-09-29: `output_tokens.len()` just before this position commits.
    pub out_len: usize,
    pub consecutive_confident: u32,
    pub sentence_defer_count: u32,
    pub force_end_thinking: bool,
}

impl SpecThinkTrail {
    pub(crate) fn capture(a: &ActiveSeq, tok: u32) -> Self {
        Self {
            tok,
            out_len: a.output_tokens.len(),
            consecutive_confident: a.consecutive_confident,
            sentence_defer_count: a.sentence_defer_count,
            force_end_thinking: a.force_end_thinking,
        }
    }
}

/// 2026-09-29: A146: `emit_token`'s entry hook. When `tok` is the next
/// position of the last pick window, apply that position's post-pipeline
/// accumulators (what decode's pipeline would have left before committing
/// `tok`). Any mismatch means the trail is stale (a rejected tail, or an emit
/// outside a window) and it is dropped.
pub(crate) fn apply_spec_think_trail(a: &mut ActiveSeq, tok: u32) {
    let Some(t) = a.spec_think_trail.pop_front() else {
        return;
    };
    if t.tok == tok && t.out_len == a.output_tokens.len() {
        a.consecutive_confident = t.consecutive_confident;
        a.sentence_defer_count = t.sentence_defer_count;
        a.force_end_thinking = t.force_end_thinking;
    } else {
        a.spec_think_trail.clear();
    }
}
