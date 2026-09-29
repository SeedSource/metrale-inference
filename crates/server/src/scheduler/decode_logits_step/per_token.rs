// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: One decoded token's bookkeeping in
//! `process_decode_logits_skipping`: hard stops, thinking state, grammar,
//! tool-call guards and EOS handling. A token that is kept is handed to
//! `content_emit`.
//!
//! Owner: scheduler.
//! Invariants:
//! - The `<tool_response>` and stray-`</think>` hard stops set
//!   `a.guard_stop` where they set `finished`; `finish_guard_tests.rs`
//!   pins the count over this module's files.

use super::*;

/// 2026-09-26: Apply one sampled token to its sequence. Each early `return`
/// ends this token's processing; the caller moves on to the next row.
pub(super) fn process_decoded_token(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    now: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    a.last_token = tok;
    a.last_token_time = now;

    // 2026-09-25: `<tool_response>` hard stop (`SchedLevers::tool_response_stop`, on
    // unless `METRALE_TOOL_RESPONSE_STOP` is `0` or `false`): the model must not
    // emit this control token; if it does, end the turn before the grammar and
    // EOS handling below.
    if sched.levers.tool_response_stop
        && let Some(trs) = sched.limits.tool_response_hard_stop
        && tok == trs
    {
        a.output_tokens.push(tok);
        a.finished = true;
        // 2026-09-25: Name the cut, or `derive_finish_reason` reports a plain "stop".
        a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);
        tracing::debug!(target: "met::scheduler::decode_logits_step", "<tool_response> hard-stop fired (id={trs}); ending turn");
        return;
    }

    // 2026-09-25: `<think>` outside thinking: enter thinking mode without emitting
    // the token. The thinking budget is the spontaneous budget halved per
    // earlier thinking-watchdog fire (at most 4 times, so 1/16), floored at 8.
    // `PostCloseThinkMask` masks `a.think_start_token` while `think_ended` is
    // set.
    // 2026-09-29: A146: `emit_token` and `pick_positions_from_host` use
    // `think_commit::spontaneous_think_budget`, the same decay and floor.
    if !a.inside_thinking && think_start_token == Some(tok) {
        let decay_shift = a.think_watchdog_fires.min(4);
        let decayed = a.spontaneous_think_budget >> decay_shift;
        a.inside_thinking = true;
        a.think_ended = false;
        a.think_skip_count = 0;
        a.thinking_budget = Some(decayed.max(8));
        if a.think_watchdog_fires > 0 {
            tracing::debug!(target: "met::scheduler::decode_logits_step", fires = a.think_watchdog_fires,
                decayed_budget = decayed,
                "Spontaneous <think> re-entry after watchdog; decayed budget"
            );
        } else {
            tracing::debug!(target: "met::scheduler::decode_logits_step", "Spontaneous <think> detected, entering thinking mode");
        }
        return;
    }

    // 2026-09-25: Drop a `</think>` that arrives outside thinking; the 50th such token
    // (`think_skip_count`) ends the turn.
    if !a.inside_thinking && think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
            // 2026-09-25: Name the cut: the stray token is not pushed, so
            // `derive_finish_reason` has no other sign that a guard ended the turn.
            a.guard_stop = Some(GUARD_STOP_THINK_SKIP);
            tracing::debug!(target: "met::scheduler::decode_logits_step", "</think> think-skip watchdog hard-stop fired (50 consecutive strays); \
                 ending turn"
            );
        }
        return;
    }
    // 2026-09-25: After `</think>`, any other token resets the stray count.
    if a.think_ended {
        a.think_skip_count = 0;
    }

    // 2026-09-25: Advance the grammar matcher only outside thinking; thinking tokens
    // (the closing `</think>` included) are not part of the constrained output.
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        gs.accept_token(tok);
    }

    // 2026-09-25: Thinking tokens, `</think>` included, draw down the same
    // `remaining` budget as content tokens (`handle_content_token`), so thinking
    // cannot run a request past `max_tokens`. `thinking_budget` is the separate
    // per-block cap armed below.
    if a.inside_thinking {
        a.consume_generation_budget();
        if think_end_token == Some(tok) {
            // 2026-09-29: A146: `emit_token`'s `</think>` branch and the
            // speculative flip in `pick_positions_from_host` reset the same
            // fields; keep the three in step.
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            // 2026-09-25: One-shot flag for the first token after `</think>`
            // (`PinToToolCallStart` reads it); the next content token clears it.
            a.think_just_ended = true;
        } else {
            // 2026-09-29: A146: `thinking_tokens`, ``` fence parity, the budget
            // arm and the thinking-loop watchdog, in one body
            // (`think_commit::advance_thinking_token`) that `emit_token` also
            // runs at commit and `pick_positions_from_host` runs per verify
            // position, so a token committed inside `<think>` advances the
            // same state on every path. `tok` is not pushed yet: the history
            // is all of `output_tokens`.
            let history_len = a.output_tokens.len();
            crate::scheduler::think_commit::advance_thinking_token(
                a,
                tok,
                history_len,
                crate::scheduler::think_commit::ThinkTokenEnv {
                    code_fence_token,
                    think_loop_enabled: !sched.levers.disable_watchdogs
                        && sched.watchdog.enable_think_loop_watchdog,
                    watchdog: sched.watchdog,
                },
                true,
            );
        }
    } else {
        handle_content_token(a, model, sched);
    }

    // 2026-09-25: A `<tool_call>` outside thinking satisfies `require_tool_call`; one
    // inside thinking does not.
    if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }
    // 2026-09-25: Every `<tool_call>` opened outside thinking resets the inter-tool
    // prose count.
    if tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
        // 2026-09-25: Tool-call repetition guard: count the `<tool_call>` opens after a
        // call has completed (`tool_call_completed`), and end the response at
        // `MAX_POST_COMPLETION_TOOL_OPENS`.
        if a.tool_call_completed {
            a.post_completion_tool_opens = a.post_completion_tool_opens.saturating_add(1);
            const MAX_POST_COMPLETION_TOOL_OPENS: u32 = 8;
            if a.post_completion_tool_opens >= MAX_POST_COMPLETION_TOOL_OPENS {
                tracing::warn!(target: "met::scheduler::decode_logits_step", opens = a.post_completion_tool_opens,
                    "tool-call repetition runaway: model re-opened {MAX_POST_COMPLETION_TOOL_OPENS}+ tool-call blocks after a completed call on a tool_choice=auto turn; ending response (was burning to max_tokens). Sanitizer keeps the first valid call(s)."
                );
                a.output_tokens.push(tok);
                a.tool_call_opened = true;
                if let Some(ref mut gs) = a.grammar_state {
                    gs.accept_token(tok);
                }
                a.finished = true;
                return;
            }
        }
    }
    // 2026-09-29: A146: `emit_token` (before its push) and
    // `pick_positions_from_host` apply the same 512-token clear.
    // 2026-09-25: `require_tool_call` still set after 512 output tokens: clear it,
    // which lifts its hold on EOS.
    if a.require_tool_call && a.output_tokens.len() > 512 {
        tracing::warn!(target: "met::scheduler::decode_logits_step", "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
        );
        a.require_tool_call = false;
    }

    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    // 2026-09-25: `</tool_call>` outside thinking. A request with an active grammar or
    // declared tools (`tools_present`) keeps generating past a closed call, so
    // the model can emit further calls; any other request ends here.
    if tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.output_tokens.push(tok);
        // 2026-09-25: Read by the EOS escape below and by the repeated-open guard
        // above.
        a.tool_call_completed = true;
        if a.sink.is_streaming() {
            let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                StreamEvent::TokenWithLogprobs(tok, lp)
            } else {
                StreamEvent::Token(tok)
            };
            if !sched.io.req.emit(&a.sink, event, "tool_call_end") {
                tracing::warn!(target: "met::scheduler::decode_logits_step", "Streaming receiver dropped during tool_call_end, finishing sequence"
                );
                a.finished = true;
                return;
            }
        }
        if a.grammar_state.is_none() && !a.tools_present {
            // 2026-09-25: No grammar and no tools declared: end the turn.
            a.finished = true;
        }
        // 2026-09-25: The `continue` below skips `update_tool_param_state`; clear
        // `inside_tool_body` and advance the grammar matcher here.
        a.inside_tool_body = false;
        if let Some(ref mut gs) = a.grammar_state {
            gs.accept_token(tok);
        }
        // 2026-09-25: Clear `think_ended` at `</tool_call>`, so `PostCloseThinkMask` no
        // longer masks `<think>` and the model may think again before its next
        // call. Re-entry still shrinks the budget by `think_watchdog_fires`
        // (the `<think>` branch above).
        a.think_ended = false;
        return;
    }

    // 2026-09-25: EOS handling: each `*_suppress*` flag below can hold a sampled EOS
    // back. EOS escape (`SchedLevers::tool_eos_escape`, on unless
    // `METRALE_TOOL_EOS_ESCAPE` is `0` or `false`): once a tool call has
    // completed, outside a tool body and thinking, the grammar does not hold
    // EOS back.
    let eos_escape = sched.levers.tool_eos_escape
        && a.tool_call_completed
        && !a.inside_tool_body
        && !a.inside_thinking;
    // 2026-09-25: The grammar holds EOS back only when the response cannot legally
    // end at the matcher's position (`grammar_blocks_stop`). That fills a
    // bitmask, so it is evaluated only when the sampled token is an EOS.
    let grammar_suppresses_eos = a.eos_tokens.contains(&tok)
        && !eos_escape
        && crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens);
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    // 2026-09-25: Inside thinking, EOS is held back unless a hard ceiling is hit
    // (`hard_ceiling_hit`): `remaining` is 0 (it already counts this token) or
    // `seq_len` is at the `max_seq_len` ceiling.
    let hard_ceiling = hard_ceiling_hit(a.remaining, a.seq.seq_len, sched.limits.max_seq_len);
    let thinking_suppresses_eos = eos_suppressed_by_thinking(a.inside_thinking, hard_ceiling);
    // 2026-09-25: Post-think EOS guard: on a tool-armed turn (`require_tool_call` or
    // `tool_request`), after `</think>`, hold EOS back until the output holds
    // `POST_THINK_MIN_CONTENT` tokens beyond the thinking ones, so the model
    // has room to open a tool call.
    const POST_THINK_MIN_CONTENT: u32 = 16;
    let post_think_content_tokens =
        (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens);
    let tools_armed = a.require_tool_call || a.tool_request;
    let post_think_suppresses_eos =
        tools_armed && a.think_ended && post_think_content_tokens < POST_THINK_MIN_CONTENT;
    let suppress_eos = grammar_suppresses_eos
        || legacy_suppresses_eos
        || min_tokens_suppresses
        || thinking_suppresses_eos
        || post_think_suppresses_eos;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        // 2026-09-25: An EOS that is not held back: counted in `output_tokens` but not
        // streamed; the sequence finishes.
        a.output_tokens.push(tok);
        crate::scheduler::emit_step::update_tool_param_state(a, tok);
        a.finished = true;
    } else if a.eos_tokens.contains(&tok) && suppress_eos {
        // 2026-09-25: A held-back EOS is dropped (not streamed, not in
        // `output_tokens`) and decoding continues.
        //
        // When thinking is the only reason it is held back and the model sets
        // `honor_eos_inside_thinking`, the EOS closes the thinking block: the
        // same state changes as the `</think>` branch above, plus
        // `think_force_closed`. The EOS itself is still dropped, and thinking no
        // longer holds back a later one.
        let thinking_is_sole_suppressor = thinking_suppresses_eos
            && !grammar_suppresses_eos
            && !legacy_suppresses_eos
            && !post_think_suppresses_eos
            && !min_tokens_suppresses;
        // 2026-09-25: MODEL.toml `[behavior].honor_eos_inside_thinking`, false when
        // unset.
        let honor_eos_in_think = sched.watchdog.honor_eos_inside_thinking;
        if thinking_is_sole_suppressor && honor_eos_in_think {
            a.inside_thinking = false;
            a.think_force_closed = true;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            a.think_just_ended = true;
        }
        tracing::debug!(
            target: "metrale::eos",
            tok,
            implicit_think_close = thinking_is_sole_suppressor && honor_eos_in_think,
            thinking_sole_suppressor = thinking_is_sole_suppressor,
            honor_eos_inside_thinking = honor_eos_in_think,
            inside_thinking = a.inside_thinking,
            think_ended = a.think_ended,
            thinking_tokens = a.thinking_tokens,
            by_thinking = thinking_suppresses_eos,
            by_grammar = grammar_suppresses_eos,
            by_legacy_tool = legacy_suppresses_eos,
            by_post_think = post_think_suppresses_eos,
            by_min_tokens = min_tokens_suppresses,
            "EOS suppressed; model forced to continue"
        );
    } else {
        super::content_emit::emit_content_token(a, tok, model, sched);
    }
}
