// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `StreamEvent::Token` / `TokenWithLogprobs` arm of the streaming
//! chat handler: turn one token into provider-neutral deltas (reasoning, content or
//! tool-call deltas) and run the stream-side loop and leak guards.
//!
//! Owner: server streaming API.
//! Invariants:
//! - Every `cancel_flag` store in this file sets `guard_stop` first
//!   (`cancel_guard_tests` checks it).

use crate::ir::StreamDelta;
use crate::tool_parser;

use super::super::sanitizer::sanitize_content_chunk;
use super::super::stream_guards::bump_f12_tool_call_count;
use super::ctx::StreamCtx;
use super::state::StreamState;

/// 2026-09-26: Read once per process. `METRALE_SIMHASH_LOOP=0` turns the SimHash
/// semantic-loop guard off; unset or any other value leaves it on.
fn simhash_loop_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_SIMHASH_LOOP").as_deref() != Ok("0"))
}

/// 2026-09-26: Read once per process. `METRALE_DISABLE_WATCHDOGS` (`1` or `true`) turns
/// the token loop watchdog off too.
fn watchdogs_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        crate::scheduler::parse_disable_watchdogs(
            std::env::var("METRALE_DISABLE_WATCHDOGS").ok().as_deref(),
        )
    })
}
use super::strip::{
    maybe_log_decode_trace, strip_all_preserving_boundary, strip_preserving_boundary,
};
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

mod detector_content;
#[cfg(test)]
mod role_literal_strip_tests;
mod stop_holdback;
#[cfg(test)]
mod stop_string_holdback_tests;

use detector_content::{detector_content_arm, process_detector_content};
use stop_holdback::apply_stop_string_holdback;

type DeltaVec = Vec<StreamDelta>;

/// 2026-09-26: Most consecutive tokens the stream may spend with
/// `suppressing_param_leak` set (the sanitizer holding back content after an orphan
/// tool-markup opener) before `handle_token` ends the stream.
const MAX_SUPPRESS_STREAK_TOKENS: u32 = 256;

/// 2026-10-03: Clear a delta that is exactly a bare role literal (`user`, `assistant`,
/// `tool`, no surrounding whitespace, under 20 bytes) at the very start of the content
/// or right after a newline (`at_line_start`): the leaked-role-markup case. A spaced
/// mid-sentence token such as `" tool"` is prose and is kept. Nothing is cleared while
/// `inside_tool_call`: there a lone `tool` can be the first fragment of a `tool_*`
/// tool name, which the detector is reassembling.
pub(super) fn strip_bare_role_literal(
    delta: &mut String,
    inside_tool_call: bool,
    at_line_start: bool,
) {
    if inside_tool_call || !at_line_start {
        return;
    }
    if delta.len() < 20 && matches!(delta.as_str(), "user" | "assistant" | "tool") {
        tracing::debug!("role-literal strip: dropped bare '{delta}' delta");
        delta.clear();
    }
}

/// 2026-09-26: Process one token and return the deltas to forward (possibly none). The
/// orphan-suppression streak check runs here, after [`handle_token_inner`], because
/// the inner function has many early returns.
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    // 2026-09-26: Counted as each token arrives, so a rate over DECODED_TOKENS_TOTAL is
    // live rather than one step at completion.
    crate::metrics::DECODED_TOKENS_TOTAL.inc();
    let result = handle_token_inner(state, ctx, tok);
    // 2026-09-26: Debug line for the first non-empty delta batch. With the first
    // stable-content line in `handle_token_inner`, it splits first-delta latency into
    // this function and what runs after it.
    if !state.first_result_logged && !result.is_empty() {
        state.first_result_logged = true;
        tracing::debug!(
            "stream: first delta batch leaves handle_token ({} deltas, {} tokens seen)",
            result.len(),
            state.all_toks.len(),
        );
    }

    // 2026-09-26: Orphan-suppression streak: past `MAX_SUPPRESS_STREAK_TOKENS`
    // consecutive suppressed tokens, end the stream.
    if state.suppressing_param_leak && !state.stop_string_triggered {
        state.suppress_streak_tokens = state.suppress_streak_tokens.saturating_add(1);
        if state.suppress_streak_tokens > MAX_SUPPRESS_STREAK_TOKENS {
            tracing::warn!(
                streak = state.suppress_streak_tokens,
                "orphan tool-call suppression streak exceeded {MAX_SUPPRESS_STREAK_TOKENS} tokens; ending stream",
            );
            state.loop_watchdog_triggered = true;
            state.stop_string_triggered = true;
            // 2026-09-26: Named, so `resolve_wire_finish_reason` reports "length"; a bare
            // cancel with budget left would reach the client as "stop".
            state.guard_stop = Some("suppress_streak");
            state
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
        }
    } else if !state.suppressing_param_leak {
        state.suppress_streak_tokens = 0;
    }

    result
}

fn handle_token_inner(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    let mut deltas: DeltaVec = Vec::new();
    state.all_toks.push(tok);
    // 2026-09-26: One id per streamed token, drained onto the next client-visible chunk
    // when the request asked for `return_token_ids`.
    if ctx.req_return_token_ids {
        state.pending_token_ids.push(tok);
    }

    if !state.thinking_done {
        if let Some(end_id) = ctx.state.think_end_token_id
            && tok == end_id
        {
            state.thinking_done = true;
            // 2026-09-26: Emit only the reasoning bytes past `state.emitted` (for example a
            // held-back incomplete UTF-8 tail); the rest was already streamed.
            if ctx.enable_thinking && state.all_toks.len() > 1 {
                let full = ctx
                    .state
                    .tokenizer
                    .decode(&state.all_toks[..state.all_toks.len() - 1])
                    .unwrap_or_default();
                let stable = full.trim_end_matches('\u{FFFD}');
                if stable.len() > state.emitted {
                    let residual = &stable[state.emitted..];
                    // 2026-09-26: A whitespace-only residual is real text; skip only an
                    // empty one.
                    if !residual.is_empty() {
                        deltas.push(StreamDelta::Reasoning {
                            text: residual.to_string(),
                            token_ids: state.take_ids_if(ctx.req_return_token_ids),
                        });
                    }
                }
            }
            // 2026-09-26: Flush the tail the reasoning sanitizer held back, unless it is
            // suppressing a leak.
            if !state.reasoning_suppressing_leak && !state.reasoning_tag_scan_buf.is_empty() {
                let tail = std::mem::take(&mut state.reasoning_tag_scan_buf);
                if !tail.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: tail,
                        token_ids: Vec::new(),
                    });
                }
            }
            if let Some(ref mut det) = state.detector {
                det.reset();
            }
            state.emitted = 0;
            state.all_toks.clear();
            state.content_decoded.clear();
            state.detok_prefix_offset = 0;
            state.detok_read_offset = 0;
            return deltas;
        }
        if ctx.enable_thinking {
            // 2026-09-26: After the in-think leak scanner cancelled the sequence, emit no
            // more reasoning; tokens already in flight before the scheduler sees
            // `cancel_flag` are dropped here.
            if state.reasoning_xml_leak_detected {
                return deltas;
            }
            // 2026-09-26: Extend the stable decoded text incrementally (O(n) over the
            // stream) rather than decoding all of `all_toks` on every token.
            let delta_stable = ctx.state.tokenizer.incremental_decode(
                &state.all_toks,
                &mut state.detok_prefix_offset,
                &mut state.detok_read_offset,
            );
            state.content_decoded.push_str(&delta_stable);
            let stable_end = state.content_decoded.len();
            if stable_end > state.emitted {
                let raw = state.content_decoded[state.emitted..stable_end].to_string();
                let mut cleaned = raw.clone();
                state.emitted = stable_end;
                cleaned = cleaned.replace("<think>", "");
                if let Some(rest) = cleaned.strip_prefix("assistant\n") {
                    cleaned = rest.to_string();
                } else if let Some(rest) = cleaned.strip_prefix("assistant") {
                    cleaned = rest.to_string();
                }
                // 2026-09-26: Remove complete tool-call blocks without gluing the words on
                // either side (`strip_preserving_boundary`); an unclosed block cuts the rest
                // of the delta, as does `<function=`.
                while let Some(start) = cleaned.find("<tool_call>") {
                    if let Some(end_rel) = cleaned[start..].find("</tool_call>") {
                        let end = start + end_rel + "</tool_call>".len();
                        cleaned = strip_preserving_boundary(&cleaned, start, end);
                    } else {
                        cleaned = cleaned[..start].to_string();
                        break;
                    }
                }
                if let Some(start) = cleaned.find("<function=") {
                    cleaned = cleaned[..start].to_string();
                }
                // 2026-09-26: Remove leaked closing tags the same way.
                for tag in &["</parameter>", "</function>", "</tool_call>"] {
                    cleaned = strip_all_preserving_boundary(&cleaned, tag);
                }
                // 2026-09-26: Remove doubled role words (`useruser`) and role words alone on
                // a line (`\nuser\n` → `\n`).
                for word in &["user", "assistant", "tool"] {
                    let pair = format!("{word}{word}");
                    cleaned = strip_all_preserving_boundary(&cleaned, &pair);
                    let nl_form = format!("\n{word}\n");
                    while cleaned.contains(&nl_form) {
                        cleaned = cleaned.replace(&nl_form, "\n");
                    }
                }
                maybe_log_decode_trace(&raw, &cleaned, stable_end, stable_end - raw.len());
                // 2026-09-26: In-think tool-call leak scanner, for requests with tools. A
                // 256-byte rolling tail of cleaned reasoning finds an opener split across
                // deltas. Each opener counts once; at `ChatLevers::in_think_leak_openers`
                // hits (default 1, 0 = never) the stream is cut and the sequence cancelled.
                let tools_active_request =
                    !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
                if tools_active_request {
                    state.reasoning_xml_scan_buf.push_str(&cleaned);
                    if state.reasoning_xml_scan_buf.len() > 256 {
                        let drop_to = state.reasoning_xml_scan_buf.len() - 256;
                        let cut = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .find(|&(i, _)| i >= drop_to)
                            .map(|(i, _)| i)
                            .unwrap_or(state.reasoning_xml_scan_buf.len());
                        state.reasoning_xml_scan_buf.drain(..cut);
                    }
                    let opener = ["<tool_call>", "<function=", "<parameter=", "<invoke "]
                        .iter()
                        .copied()
                        .filter_map(|m| state.reasoning_xml_scan_buf.find(m).map(|at| (at, m)))
                        .min_by_key(|&(at, _)| at);
                    if let Some((at, op)) = opener {
                        // 2026-09-26: Take the log tail before consuming the match, so the
                        // warning shows the opener in context.
                        let tail_start = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .rev()
                            .nth(63)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let tail = state.reasoning_xml_scan_buf[tail_start..].to_string();
                        // 2026-09-26: Consume the buffer through this opener, so one
                        // occurrence is counted once.
                        state.reasoning_xml_scan_buf.drain(..at + op.len());
                        state.reasoning_xml_opener_hits =
                            state.reasoning_xml_opener_hits.saturating_add(1);
                        let threshold = ctx.state.chat.in_think_leak_openers;
                        if threshold != 0 && state.reasoning_xml_opener_hits >= threshold {
                            state.reasoning_xml_leak_detected = true;
                            // 2026-09-26: `tool_loop_capped` makes the wire reason "length";
                            // `guard_stop` names the cut in the `--dump` record.
                            state.guard_stop = Some("in_think_tool_leak");
                            state.tool_loop_capped = true;
                            state.stop_string_triggered = true;
                            state
                                .cancel_flag
                                .store(true, std::sync::atomic::Ordering::Release);
                            tracing::warn!(
                                model = %ctx.model,
                                request_id = %ctx.id,
                                opener = op,
                                hits = state.reasoning_xml_opener_hits,
                                threshold,
                                tail = %tail,
                                "in-think tool-call leak: opener threshold reached; cancelling \
                                 sequence (finish_reason \"length\", guard in_think_tool_leak). \
                                 Raise/disable via METRALE_INTHINK_TOOL_LEAK_OPENERS (0 = strip-only)"
                            );
                            return deltas;
                        }
                        // 2026-09-26: Below the threshold the stream continues. An opener
                        // split across deltas may already have reached the client's
                        // reasoning in part.
                        tracing::warn!(
                            model = %ctx.model,
                            request_id = %ctx.id,
                            opener = op,
                            hits = state.reasoning_xml_opener_hits,
                            threshold,
                            "in-think tool-call opener observed in reasoning; below \
                             METRALE_INTHINK_TOOL_LEAK_OPENERS threshold, not cancelling"
                        );
                    }
                }
                // 2026-09-26: Then the leak-marker sanitizer, with the reasoning-side state.
                let cleaned = sanitize_content_chunk(
                    &cleaned,
                    &mut state.reasoning_tag_scan_buf,
                    &mut state.reasoning_suppressing_leak,
                    &mut state.reasoning_inside_envelope,
                    &ctx.leak_markers,
                );
                // 2026-09-26: Emit whitespace-only chunks too: `state.emitted` has already
                // moved past these bytes, so a dropped chunk would be lost.
                if !cleaned.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: cleaned,
                        token_ids: state.take_ids_if(ctx.req_return_token_ids),
                    });
                }
            }
        }
        return deltas;
    }

    // 2026-09-26: Content phase, decoded like the reasoning path: extend the stable text
    // with `incremental_decode` (an incomplete UTF-8 tail waits for the next token) and
    // emit the bytes past `state.emitted`. When thinking ran, the think-end token reset
    // `all_toks`, `emitted` and the offsets, so this is post-thinking text only.
    let delta_stable = ctx.state.tokenizer.incremental_decode(
        &state.all_toks,
        &mut state.detok_prefix_offset,
        &mut state.detok_read_offset,
    );
    state.content_decoded.push_str(&delta_stable);
    let stable_end = state.content_decoded.len();
    let _ = tok;
    // 2026-10-03: Whether the content emitted before this token is empty or ends in a
    // newline, for `strip_bare_role_literal`.
    let at_line_start =
        state.content_decoded[..state.emitted].is_empty() || state.content_decoded[..state.emitted].ends_with('\n');
    let mut delta = if stable_end > state.emitted {
        // 2026-09-26: Debug line when the first stable content bytes exist on the stream
        // side.
        if state.emitted == 0 {
            tracing::debug!(
                "stream: first stable content delta ({stable_end} bytes: {:?})",
                &state.content_decoded[..stable_end.min(24)],
            );
        }
        let raw = state.content_decoded[state.emitted..stable_end].to_string();
        state.emitted = stable_end;
        raw
    } else {
        return deltas;
    };
    let _ = &state.content_decoder;

    if state.thinking_done {
        // 2026-09-26: The same think-marker scrub as the blocking path.
        delta = crate::api::strip::scrub_think_markers(&delta);
        // 2026-09-26: A re-opened `<think>` ends this delta there and returns the stream
        // to the thinking phase.
        if let Some(pos) = delta.find("<think>") {
            delta = delta[..pos].to_string();
            state.thinking_done = false;
            state.all_toks.clear();
            state.emitted = 0;
            state.content_decoded.clear();
            state.detok_prefix_offset = 0;
            state.detok_read_offset = 0;
        }
        // 2026-09-26: With no tools, cut this delta at its first tool-call opener
        // (`strip_orphan_tool_markup`, also used by the blocking path). With tools, the
        // detector handles markup.
        let tools_active_request =
            !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
        if !tools_active_request {
            delta = crate::api::strip::strip_orphan_tool_markup(&delta);
        }
    }

    // 2026-09-26: The detector's in-body flag is read before this delta reaches it.
    {
        let inside_tool_call = state
            .detector
            .as_ref()
            .is_some_and(|d| d.inside_tool_call());
        strip_bare_role_literal(&mut delta, inside_tool_call, at_line_start);
    }

    if delta.is_empty() {
        return deltas;
    }

    // 2026-09-26: Client stop sequences are matched on the text, with a held-back tail
    // (`apply_stop_string_holdback`, a pure function so it is tested without a
    // `StreamCtx`).
    if !ctx.stop_strings.is_empty() && !state.stop_string_triggered {
        delta = apply_stop_string_holdback(
            &delta,
            &ctx.stop_strings,
            ctx.stop_string_buffer_len,
            &mut state.accumulated_content,
            &mut state.stop_string_emitted_len,
            &mut state.stop_string_triggered,
        );
        // 2026-09-26: Entry required `!stop_string_triggered`, so a flip here is a client
        // stop-sequence match: record it for the wire finish reason and cancel.
        if state.stop_string_triggered {
            state.note_stop_string_match();
        }
        if delta.is_empty() {
            // 2026-09-26: Everything is held back, or a match left nothing to emit.
            return deltas;
        }
    }

    if state.stop_string_triggered {
        // 2026-09-26: `stop_string_triggered` is also set by the loop and tool guards.
        // Tokens that arrive after it, before the scheduler sees the cancel, still go out
        // as content, but through the buffered sanitizer, which removes tool markup split
        // across deltas; other text is passed on.
        if !delta.is_empty() {
            let cleaned = sanitize_content_chunk(
                &delta,
                &mut state.tag_scan_buf,
                &mut state.suppressing_param_leak,
                &mut state.inside_envelope,
                &ctx.leak_markers,
            );
            if !cleaned.is_empty() {
                deltas.push(StreamDelta::Content {
                    text: cleaned,
                    token_ids: state.take_ids_if(ctx.req_return_token_ids),
                });
            }
        }
        return deltas;
    }

    if state.detector.is_some() {
        // 2026-09-26: Collect the detector outputs first, ending the borrow of
        // `state.detector` before the handlers take `state`.
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.process(&delta)
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                        deltas.extend(events_out);
                        return deltas;
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx);
                }
            }
        }
    } else {
        let sanitized = sanitize_content_chunk(
            &delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &ctx.leak_markers,
        );
        if let Some(events_out) = process_detector_content(state, ctx, &sanitized) {
            deltas.extend(events_out);
            return deltas;
        }
    }

    deltas
}
