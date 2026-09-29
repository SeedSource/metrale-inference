// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the host-side K-position pick loop of
//! `verify_pick_all_with_pipeline`; it takes host logits rather than a
//! model, so tests drive it directly.
//!
//! Owner: scheduler.
//! Invariants:
//! - On return the grammar matcher is at the history depth it had on entry,
//!   and `inside_thinking`, `think_ended` and `think_just_ended` hold their
//!   entry values. Within the loop, each later position is masked against
//!   the matcher state the earlier picks leave behind, and a `</think>` pick
//!   switches the thinking flags for the remaining positions, so the first
//!   position after it is picked outside thinking.
//! - 2026-09-29: A146: on return `output_tokens` and every commit field in
//!   `SpecThinkState` hold their entry values; `a.spec_think_trail` holds one
//!   entry per pick.

use super::verify_pick_with_pipeline;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::think_commit::{
    SpecThinkTrail, ThinkTokenEnv, advance_thinking_token, spontaneous_think_budget,
};
use crate::scheduler::types::ActiveSeq;

/// 2026-09-29: A146: the step-start snapshot of the commit state the window
/// advances speculatively, restored on exit (the window only picks;
/// `emit_token` commits).
struct SpecThinkState {
    thinking_tokens: u32,
    in_code_fence: bool,
    force_end_thinking: bool,
    sentence_defer_count: u32,
    consecutive_confident: u32,
    think_watchdog_fires: u32,
    thinking_budget: Option<u32>,
    think_skip_count: u32,
    require_tool_call: bool,
    tool_call_opened: bool,
}

impl SpecThinkState {
    fn capture(a: &ActiveSeq) -> Self {
        Self {
            thinking_tokens: a.thinking_tokens,
            in_code_fence: a.in_code_fence,
            force_end_thinking: a.force_end_thinking,
            sentence_defer_count: a.sentence_defer_count,
            consecutive_confident: a.consecutive_confident,
            think_watchdog_fires: a.think_watchdog_fires,
            thinking_budget: a.thinking_budget,
            think_skip_count: a.think_skip_count,
            require_tool_call: a.require_tool_call,
            tool_call_opened: a.tool_call_opened,
        }
    }

    fn restore(&self, a: &mut ActiveSeq) {
        a.thinking_tokens = self.thinking_tokens;
        a.in_code_fence = self.in_code_fence;
        a.force_end_thinking = self.force_end_thinking;
        a.sentence_defer_count = self.sentence_defer_count;
        a.consecutive_confident = self.consecutive_confident;
        a.think_watchdog_fires = self.think_watchdog_fires;
        a.thinking_budget = self.thinking_budget;
        a.think_skip_count = self.think_skip_count;
        a.require_tool_call = self.require_tool_call;
        a.tool_call_opened = self.tool_call_opened;
    }
}

/// 2026-09-25: run `verify_pick_with_pipeline` over `k` host-resident BF16
/// rows of `vocab` logits and return the pick per position. Stops early,
/// returning fewer than `k` picks, when the matcher refuses a pick.
///
/// 2026-09-29: A146, spec-in-think parity: the window advances every piece of
/// commit state a later position's pipeline reads: the picks themselves
/// (pushed onto `output_tokens`, for the mid-word and sentence-boundary
/// previous token and the penalty history), `thinking_tokens`,
/// `in_code_fence`, the budget and thinking-loop arming (shared body
/// `think_commit::advance_thinking_token`, also run by
/// `process_decode_logits` and `emit_token`), the `</think>` resets, a
/// spontaneous `<think>`, and the post-`</think>` pin inputs; all of it is
/// restored on exit. Because a position's pick already reflects any close
/// spec-off would force right after an earlier position, an accepted run
/// never needs truncating: a draft that disagrees with the forced token is
/// rejected there. The pipeline's own accumulators are restored too, and left
/// per position in `a.spec_think_trail` for `emit_token`.
pub(super) fn pick_positions_from_host(
    buf: &[u8],
    vocab: usize,
    elem_bytes: usize,
    k: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // 2026-09-25: the history depth, for the rollback at the end; see there
    // for why it is a depth and not a count of `accept_token` calls.
    let grammar_steps_before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
    let think_flags_before = (a.inside_thinking, a.think_ended, a.think_just_ended);
    // 2026-09-29: A144: tool-body state per position. The `<tool_call>`
    // opener bias is stripped inside a tool body (`strip_in_tool_opener_bias`,
    // via `penalty_params_for`), so a window that opens or closes a call must
    // re-evaluate it at every later position against the picks earlier in the
    // same window, as `emit_token` -> `update_tool_param_state` will on the
    // accept path. Restored on exit.
    let tool_body_before = a.inside_tool_body;
    // 2026-09-29: A146: every piece of per-token commit state the pipeline
    // reads at a later position is advanced here per position, exactly as
    // the commit twins (`process_decode_logits`, `emit_token`) will advance
    // it, and restored on exit. The pipeline reads `thinking_tokens` (the F2
    // 400-token gate, the min-reasoning floor, the forced-`</think>` hard
    // override), `in_code_fence` (fence deferral), `force_end_thinking`
    // (budget and thinking-loop arming), `output_tokens` (the mid-word mask
    // and sentence-boundary gate via `.last()`, the penalty history),
    // `think_just_ended`, `require_tool_call` and `tool_call_opened` (the
    // post-think `<tool_call>` pin), and the spontaneous-`<think>` budget.
    let think_state_before = SpecThinkState::capture(a);
    let out_len_before = a.output_tokens.len();
    let think_env = ThinkTokenEnv {
        code_fence_token: ctx.code_fence_token,
        think_loop_enabled: !ctx.sampling.disable_watchdogs
            && ctx.watchdog.enable_think_loop_watchdog,
        watchdog: ctx.watchdog,
    };
    a.spec_think_trail.clear();

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        // 2026-09-29: A146: the window's earlier picks are pushed onto
        // `output_tokens` below, so every `output_tokens.len() + verify_pos`
        // consumer (the sampling seed, the `min_tokens` checks, the A144 base
        // bias) already sees this position's emitted length: pass 0.
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx, 0);
        picks.push(pick);
        // 2026-09-29: A146: what the pipeline left in its accumulators for
        // THIS position, re-applied by `emit_token` if and only if this
        // position commits.
        a.spec_think_trail
            .push_back(SpecThinkTrail::capture(a, pick));

        // 2026-09-29: A146: mirror of the commit transitions (`emit_token`,
        // decode). A spontaneous `<think>` enters thinking and is never
        // pushed.
        if !a.inside_thinking && a.think_start_token == Some(pick) {
            a.inside_thinking = true;
            a.think_ended = false;
            a.think_skip_count = 0;
            a.thinking_budget = Some(spontaneous_think_budget(a));
            continue;
        }
        // 2026-09-29: A146: a stray `</think>` outside thinking is skipped and
        // never pushed.
        if !a.inside_thinking && ctx.think_end_token == Some(pick) {
            continue;
        }
        if a.require_tool_call && a.tool_call_start_token == Some(pick) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // 2026-09-29: A146: the twin of the 512-token safety clear (decode,
        // `emit_token`).
        if a.require_tool_call && a.output_tokens.len() > 512 {
            a.require_tool_call = false;
        }
        let is_eos = a.eos_tokens.contains(&pick);

        // 2026-09-25: `</think>` closes thinking for the later positions. It
        // is not fed to the matcher. 2026-09-29: A146: the same resets as the
        // commit twins.
        if a.inside_thinking && ctx.think_end_token == Some(pick) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            a.think_just_ended = true;
            a.output_tokens.push(pick);
            continue;
        }
        if a.inside_thinking {
            // 2026-09-29: A146: the shared `advance_thinking_token`
            // (`thinking_tokens`, fence, budget arm, thinking-loop watchdog);
            // decode runs it for a dropped EOS too.
            let history_len = a.output_tokens.len();
            advance_thinking_token(a, pick, history_len, think_env, false);
        } else {
            // 2026-09-29: A146: `emit_token` clears the post-`</think>`
            // one-shot on the first content token; without this a later
            // position would pin again.
            a.think_just_ended = false;
            // 2026-09-29: mirror `update_tool_param_state`'s opener and
            // closer transitions (a no-op inside thinking, like the real
            // one).
            if a.tool_call_start_token == Some(pick) {
                a.inside_tool_body = true;
            } else if a.tool_call_end_token == Some(pick) {
                a.inside_tool_body = false;
            }
        }
        // 2026-09-29: A146: decode never pushes an EOS it keeps generating
        // past (an EOS it honours ends the sequence, so later positions are
        // moot).
        if !is_eos {
            a.output_tokens.push(pick);
        }

        // 2026-09-25: advance the matcher with the pick so the next position
        // is masked against the state after it. Not after the last position,
        // and not inside thinking.
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // 2026-09-25: a refused advance ends the loop; the picks so far
            // are returned.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; the real `accept_token` in emit_token will fail and end the response."
                );
                break;
            }
        }
    }

    // 2026-09-25: roll back by the history delta. Counting `accept_token`
    // calls would over-rewind: `GrammarState::accept_token` returns true for
    // stop tokens and in the terminated state without adding a history step.
    // `emit_token` later advances the matcher for the tokens it emits.
    if let (Some(before), Some(gs)) = (grammar_steps_before, a.grammar_state.as_mut()) {
        let advanced = gs.num_history_steps().saturating_sub(before);
        if advanced > 0 {
            gs.rollback(advanced);
        }
    }
    // 2026-09-25: restore the thinking flags; `emit_token` makes the real
    // `</think>` transition.
    (a.inside_thinking, a.think_ended, a.think_just_ended) = think_flags_before;
    a.inside_tool_body = tool_body_before;
    think_state_before.restore(a);
    a.output_tokens.truncate(out_len_before);

    picks
}
