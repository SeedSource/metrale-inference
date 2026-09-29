// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `emit_token`: stream and bookkeeping for one token emitted by
//! the MTP, speculative and DFlash steps.
//!
//! Owner: scheduler.
//! Invariants:
//! - The `<tool_response>` and stray-`</think>` hard stops, and the
//!   post-think cap, content-loop and prose-budget stops, set `a.guard_stop`
//!   where they set `finished`. `finish_guard_tests.rs` pins the count.

use super::*;

/// 2026-09-25: Emit a token for an active sequence (stream + bookkeeping).
///
/// EOS and the control tokens that end a turn are not streamed. An EOS is
/// still recorded in `output_tokens`, so it counts as a generated token.
///
/// When `logprobs` is `Some`, it is appended to `a.logprobs_data` and the
/// token is streamed as `StreamEvent::TokenWithLogprobs`.
pub fn emit_token(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    // 2026-09-25: per-token debug line (`slot`, `out_idx`, `tok`), for diffing
    // one stream's tokens between two runs.
    tracing::debug!(
        "TOK slot={} out_idx={} tok={}",
        a.seq.slot_idx,
        a.output_tokens.len(),
        tok,
    );
    // 2026-09-29: A146, spec-in-think parity: re-apply the logits pipeline's
    // accumulators (F2 streak and arming, forced-`</think>` defer ticks) as
    // they stood after THIS position's pipeline in the verify pick window,
    // which is what decode's `process_seq_logits` leaves before its commit.
    // Producer: `pick_positions_from_host`. A no-op outside a verify commit
    // run.
    crate::scheduler::think_commit::apply_spec_think_trail(a, tok);

    // 2026-09-25: the streaming side set the request's cancel flag: finish
    // now, without naming a guard.
    if sched.io.req.is_cancelled(a.cancel_flag.as_ref()) {
        a.finished = true;
        return;
    }

    // 2026-09-25: ChatML role-boundary hard stop (`<|im_start|>`). It runs
    // before the EOS suppression below (grammar, `require_tool_call`,
    // `min_tokens`, thinking), which would otherwise hold the token back and
    // let generation continue.
    if let Some(ims) = sched.limits.im_start_hard_stop
        && tok == ims
    {
        // 2026-09-25: pushed, not streamed. `tokenizer_runtime.rs` adds
        // `<|im_start|>` to `eos_tokens`, so with it as the last token
        // `derive_finish_reason` takes its EOS rung and reports "stop".
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!(
            "<|im_start|> hard-stop fired (id={ims}); ending turn before grammar/suppress_eos"
        );
        return;
    }

    // 2026-09-25: `<tool_response>` hard stop, armed by
    // `SchedLevers::tool_response_stop`: emitting this control token ends
    // the turn.
    if sched.levers.tool_response_stop
        && let Some(trs) = sched.limits.tool_response_hard_stop
        && tok == trs
    {
        a.output_tokens.push(tok);
        a.finished = true;
        // 2026-09-25: named, as in `decode_logits_step`: this token is not in
        // `eos_tokens`, so unnamed the finish would report "stop" while
        // budget remains.
        a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);
        tracing::debug!("<tool_response> hard-stop fired (id={trs}); ending turn");
        return;
    }

    // 2026-09-25: a `<think>` outside thinking enters thinking mode with the
    // spontaneous-think budget; the token itself is not emitted.
    if !a.inside_thinking && a.think_start_token == Some(tok) {
        a.inside_thinking = true;
        // 2026-09-25: restart the post-`</think>` count that
        // `mtp_gate::spec_dispatch_eligible` reads.
        a.post_think_emitted = 0;
        a.think_ended = false;
        a.think_skip_count = 0;
        // 2026-09-29: A146: decayed per earlier thinking-watchdog fire, the
        // same budget as decode's spontaneous `<think>` branch
        // (`think_commit::spontaneous_think_budget`).
        a.thinking_budget = Some(crate::scheduler::think_commit::spontaneous_think_budget(a));
        tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        return;
    }

    // 2026-09-25: a `</think>` outside thinking is skipped, as in
    // `process_decode_logits`; the 50th in a row ends the turn.
    if !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
            // 2026-09-25: named: the skipped `</think>` is not pushed, so
            // unnamed the finish would report "stop" while budget remains.
            a.guard_stop = Some(GUARD_STOP_THINK_SKIP);
            tracing::debug!(
                "</think> think-skip watchdog hard-stop fired (50 consecutive strays); \
                 ending turn"
            );
        }
        return;
    }
    // 2026-09-25: any other token once thinking has ended resets the stray
    // count, as in `decode_logits_step`, so only consecutive strays add up.
    if a.think_ended {
        a.think_skip_count = 0;
    }

    // 2026-09-25: a `<tool_call>` outside thinking satisfies
    // `require_tool_call`.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }
    // 2026-09-29: A146: the twin of decode's 512-token `require_tool_call`
    // safety clear (`decode_logits_step/per_token.rs`, judged on the pre-push
    // length there too). Without it a long speculative `<think>` kept the flag
    // set past 512 tokens, so the post-`</think>` `<tool_call>` pin
    // (`PinToToolCallStart`) and the legacy EOS hold diverged from spec-off.
    if a.require_tool_call && a.output_tokens.len() > 512 {
        tracing::warn!(
            "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
        );
        a.require_tool_call = false;
    }

    // 2026-09-25: tool-body / parameter-body state. `decode_logits_step`
    // calls the same `update_tool_param_state`, so both paths track it.
    update_tool_param_state(a, tok);

    // 2026-09-25: a `</tool_call>` outside thinking marks a tool call
    // complete, which the `tool_eos_escape` gate below reads.
    if a.tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.tool_call_completed = true;
        // 2026-09-29: A143 part B: the twin of the plain-chat hard stop in
        // `decode_logits_step/per_token.rs`, which ends the turn when a
        // request with no grammar and no declared tools emits `</tool_call>`.
        // Without it a spec-on turn (MTP, K2/K3/K4 and DFlash verify-accept)
        // kept generating past the call that spec-off stops at. There is no
        // early return: `tok` falls through to the push and stream below, so
        // `</tool_call>` itself is emitted and nothing after it, because
        // every verify-accept caller stops its window once `a.finished` is
        // set.
        if a.grammar_state.is_none() && !a.tools_present {
            a.finished = true;
        }
    }

    // 2026-09-25: a `<tool_call>` outside thinking resets the inter-tool
    // prose count, as in `decode_logits_step`, so the prose budget below
    // counts only tokens since the last tool call opened.
    if a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
    }

    // 2026-09-25: advance the grammar, except inside thinking: the matcher
    // never sees thinking tokens or the closing `</think>`.
    let mut disengage_grammar = false;
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        let advanced = gs.accept_token(tok);
        if !advanced {
            // 2026-09-25: the matcher refused this token. Rather than end the
            // turn, drop the grammar for the rest of this response and
            // decode unconstrained.
            tracing::warn!(
                tok,
                output_len = a.output_tokens.len(),
                "gs.accept_token returned false — grammar/model disagreement; disengaging grammar for the remainder of this response (free decode + post-hoc tool parse) instead of aborting the turn."
            );
            disengage_grammar = true;
        }
    }
    if disengage_grammar {
        // 2026-09-25: set here, after the `gs` borrow ends.
        a.grammar_state = None;
    }

    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    a.output_tokens.push(tok);

    // 2026-09-25: count tokens emitted after thinking ended, for
    // `mtp_gate::spec_dispatch_eligible`. `</think>` itself is not counted:
    // `think_ended` is still false when it arrives. A request with thinking
    // off on a model with a `</think>` token starts with `think_ended` true
    // (prefill), so it counts from its first token.
    if a.think_ended && !a.inside_thinking {
        a.post_think_emitted += 1;
    }

    // 2026-09-25: thinking tokens draw down `remaining` like content tokens,
    // as in `decode_logits_step`; `thinking_budget` is the separate
    // per-block cap armed below.
    if a.inside_thinking {
        a.consume_generation_budget();
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.think_force_closed = a.force_end_thinking;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            // 2026-09-29: A146: the other resets of decode's `</think>`
            // transition: a later re-entry into `<think>` starts with a fresh
            // F2 streak and outside a fence.
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            // 2026-09-25: one-shot read by `PinToToolCallStart` on the next
            // step; cleared by the next non-thinking token below.
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            // 2026-09-29: A146: the body decode (`decode_logits_step`) and the
            // verify window (`pick_positions_from_host`) also run:
            // `thinking_tokens`, ``` fence parity (never toggled here before),
            // the budget arm, and the thinking-loop watchdog (decode-only
            // before). `tok` is already pushed, so the loop scan gets
            // `len - 1`, the history decode sees.
            let history_len = a.output_tokens.len().saturating_sub(1);
            crate::scheduler::think_commit::advance_thinking_token(
                a,
                tok,
                history_len,
                crate::scheduler::think_commit::ThinkTokenEnv {
                    code_fence_token: sched.limits.code_fence_token,
                    think_loop_enabled: !sched.levers.disable_watchdogs
                        && sched.watchdog.enable_think_loop_watchdog,
                    watchdog: sched.watchdog,
                },
                true,
            );
        }
    } else {
        a.consume_generation_budget();
        // 2026-09-25: the token after `</think>` clears the one-shot.
        a.think_just_ended = false;
        // 2026-09-25: content-phase watchdogs. `handle_content_token`
        // (`decode_logits_content.rs`) runs only on the
        // `process_decode_logits` path; the steps that emit through here
        // get the same checks below, with the same gates. `emit_token` has
        // no `&dyn Model`, which `rollback_to_boundary` needs, so where
        // `handle_content_token` first tries a rollback these stops end the
        // response.
        use crate::scheduler::helpers::{
            CONTENT_LOOP_CHECK_STRIDE, CONTENT_LOOP_MIN_TOKENS, CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_PERIOD_MIN, detect_content_token_loop_normalized_with,
            detect_content_token_loop_with,
        };
        a.content_tokens = a.content_tokens.saturating_add(1);
        // 2026-09-25: post-think content cap, applied whatever
        // `inside_tool_body` says, while a grammar is attached. The cap is
        // MODEL.toml `[behavior].max_post_think_content_tokens`, 100,000
        // when unset.
        if !sched.levers.disable_watchdogs
            && a.grammar_state.is_some()
            && a.content_tokens > sched.watchdog.max_post_think_content_tokens
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                max = sched.watchdog.max_post_think_content_tokens,
                "post-think content cap exceeded in MTP/emit path; ending response (tool-active request would otherwise burn to max_tokens)"
            );
            a.guard_stop = Some(GUARD_STOP_POST_THINK_CAP);
            a.finished = true;
        }
        // 2026-09-25: request `repetition_detection`, then the operator's
        // min-repeats override, then the built-in constants
        // (`WatchdogParams::content_loop_params`).
        let loop_params = sched.watchdog.content_loop_params(a.repetition_detection);
        if !sched.levers.disable_watchdogs
            && sched.levers.loop_watchdog()
            && !a.inside_tool_body
            && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && (detect_content_token_loop_with(&a.output_tokens, loop_params)
                || sched.masks.numeric.as_deref().is_some_and(|m| {
                    detect_content_token_loop_normalized_with(&a.output_tokens, m, loop_params)
                }))
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                "Content-loop watchdog fired in MTP/emit path (period-{}…{} repeat); ending response. \
                 Tune via --content-loop-min-repeats / METRALE_CONTENT_LOOP_MIN_REPEATS, per-request \
                 repetition_detection, or disarm via --content-loop-watchdog off / \
                 METRALE_CONTENT_LOOP_WATCHDOG=0",
                CONTENT_LOOP_PERIOD_MIN,
                CONTENT_LOOP_PERIOD_MAX,
            );
            // 2026-09-25: named, or the finish would report "stop".
            a.guard_stop = Some(GUARD_STOP_CONTENT_LOOP);
            a.finished = true;
        }

        // 2026-09-25: inter-tool prose budget. It counts tokens outside a
        // tool body on requests with `tool_request` set. `tool_request` is
        // fixed at prefill, so the budget still applies after the grammar
        // is dropped.
        if !sched.levers.disable_watchdogs && !a.inside_tool_body && a.tool_request {
            a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
            let max_prose = sched.watchdog.max_inter_tool_prose;
            if a.prose_tokens_since_last_tool > max_prose {
                tracing::warn!(
                    prose_tokens = a.prose_tokens_since_last_tool,
                    max = max_prose,
                    output_len = a.output_tokens.len(),
                    "Inter-tool prose budget exhausted in MTP/emit path; ending response \
                     (no tool call after budget — would otherwise burn to max_tokens); \
                     raise via --max-inter-tool-prose / METRALE_MAX_INTER_TOOL_PROSE / \
                     MODEL.toml [behavior].max_inter_tool_prose (0 disables)"
                );
                a.guard_stop = Some(GUARD_STOP_INTER_TOOL_PROSE);
                a.finished = true;
            }
        }
    }

    // 2026-09-25: EOS escape: once a tool call has completed, outside a
    // tool body and thinking, the grammar does not hold EOS back.
    // `SchedLevers::tool_eos_escape` is on unless
    // `METRALE_TOOL_EOS_ESCAPE` is `0` or `false`.
    let eos_escape = sched.levers.tool_eos_escape
        && a.tool_call_completed
        && !a.inside_tool_body
        && !a.inside_thinking;
    // 2026-09-25: the grammar holds EOS back when stopping is not legal at
    // the matcher's position (`grammar_blocks_stop`, evaluated only for an
    // EOS token because it fills a bitmask). Inside thinking the matcher
    // has not seen the thinking tokens, so a grammar-armed sequence holds
    // EOS back there too.
    let grammar_suppresses_eos = a.eos_tokens.contains(&tok)
        && !eos_escape
        && ((a.inside_thinking && a.grammar_state.is_some())
            || crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens));
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    // 2026-09-25: thinking holds EOS back until a hard ceiling is hit, the
    // same `eos_suppressed_by_thinking` term as `decode_logits_step`.
    let hard_ceiling = crate::scheduler::helpers::hard_ceiling_hit(
        a.remaining,
        a.seq.seq_len,
        sched.limits.max_seq_len,
    );
    let thinking_suppresses_eos =
        crate::scheduler::helpers::eos_suppressed_by_thinking(a.inside_thinking, hard_ceiling);
    let suppress_eos = grammar_suppresses_eos
        || legacy_suppresses_eos
        || min_tokens_suppresses
        || thinking_suppresses_eos;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        a.finished = true;
        return;
    }
    if a.eos_tokens.contains(&tok) && suppress_eos {
        // 2026-09-29: A146: inside `<think>` the dropped EOS must not stay in
        // the history: decode never pushes a held-back EOS, and the history
        // feeds the mid-word `</think>` mask, the sentence-boundary injection
        // gate, the penalties and the thinking-loop scan of every later
        // token. A held-back EOS in the content phase keeps its push, outside
        // the spec-in-think scope.
        if a.inside_thinking && a.output_tokens.last() == Some(&tok) {
            a.output_tokens.pop();
        }
        return;
    }
    // 2026-09-25: thinking tokens of a request without thinking enabled are
    // not streamed, the same gate as `process_decode_logits`.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if !suppress_stream {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        if !sched.io.req.emit(&a.sink, event, "token stream") {
            a.finished = true;
            return;
        }
    }
    // 2026-09-25: length stop: budget used up, or the next step would reach
    // the served `max_seq_len` (`seqlen_force_stop`, never when it is 0).
    if a.remaining == 0 || seqlen_force_stop(a.seq.seq_len, sched.limits.max_seq_len) {
        // 2026-09-25: first try to close an open grammar structure
        // (`emit_grammar_close`).
        emit_grammar_close(a, &sched.io, sched.levers.grammar_budget_close);
        tracing::info!(
            "emit_token: remaining={}, seq_len={}, max_seq_len={}, output_tokens={}, thinking_tokens={}",
            a.remaining,
            a.seq.seq_len,
            sched.limits.max_seq_len,
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}
