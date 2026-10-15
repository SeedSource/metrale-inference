// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Mutable per-stream state of the streaming chat handler. The streaming
//! closure owns it and passes `&mut StreamState` to each `StreamEvent` arm; read-only
//! values are in `StreamCtx` (`ctx.rs`).
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use crate::tool_parser;

pub(super) struct StreamState {
    /// 2026-09-26: Token ids of the current phase; cleared at the think-end token and when
    /// `<think>` re-opens.
    pub(super) all_toks: Vec<u32>,
    /// 2026-09-26: Bytes of `content_decoded` already emitted as reasoning or content.
    pub(super) emitted: usize,
    /// 2026-09-26: Stable decoded text of `all_toks` for the current phase, grown by
    /// `ChatTokenizer::incremental_decode`; reset with `all_toks` and `emitted`.
    pub(super) content_decoded: String,
    /// 2026-09-26: `incremental_decode` offsets into `all_toks`: the decode window is
    /// `all_toks[prefix_offset..]`, and `[prefix_offset..read_offset]` is the decoded
    /// prefix used as left context.
    pub(super) detok_prefix_offset: usize,
    pub(super) detok_read_offset: usize,
    /// 2026-09-26: Always `None`: initialised and never assigned.
    pub(super) content_decoder: Option<crate::tokenizer::StreamingDecoder<'static>>,
    /// 2026-09-26: Content accumulated for stop-string matching across deltas.
    pub(super) accumulated_content: String,
    /// 2026-09-26: Bytes of `accumulated_content` already forwarded. The stop-string
    /// hold-back (`apply_stop_string_holdback`) keeps a tail back, so this can lag the
    /// accumulator until a match or the end of the stream.
    pub(super) stop_string_emitted_len: usize,
    /// 2026-09-26: Post-sanitizer content, appended while shorter than 16 KiB; read by the
    /// refusal classifier and the `--dump` record in `handle_done`.
    pub(super) refusal_scan_buf: String,
    /// 2026-09-26: Set by a stop-string match and by the loop, leak and tool guards. Once
    /// set, content skips the stop-string matcher, the tool-call detector and the loop
    /// watchdogs (it still goes out through the sanitizer), and `handle_done` does not
    /// flush the stop-string tail.
    pub(super) stop_string_triggered: bool,
    /// 2026-09-26: True only when a client stop sequence matched; set by
    /// [`StreamState::note_stop_string_match`]. `handle_done` then reports `"stop"`, even
    /// when the scheduler said `"length"` because the budget ran out before it saw the
    /// cancel.
    pub(super) stop_string_matched: bool,
    /// 2026-09-26: Sanitizer state: content is suppressed after an orphan opener until its
    /// close arrives.
    pub(super) suppressing_param_leak: bool,
    /// 2026-09-26: Consecutive tokens with `suppressing_param_leak` set. Past
    /// `MAX_SUPPRESS_STREAK_TOKENS` (`handle_token.rs`) the stream is ended.
    pub(super) suppress_streak_tokens: u32,
    /// 2026-09-26: Sanitizer state: inside a tool-call envelope (such as
    /// `<minimax:tool_call>`), where inner markup like `<invoke ...>` is passed through.
    pub(super) inside_envelope: bool,
    /// 2026-09-26: `inside_envelope` for the reasoning sanitizer.
    pub(super) reasoning_inside_envelope: bool,
    /// 2026-09-26: Tag-scan buffer for the content sanitizer.
    pub(super) tag_scan_buf: String,
    /// 2026-09-26: `suppressing_param_leak` for the reasoning sanitizer.
    pub(super) reasoning_suppressing_leak: bool,
    /// 2026-09-26: Tag-scan buffer for the reasoning sanitizer.
    pub(super) reasoning_tag_scan_buf: String,
    /// 2026-09-26: Tail buffer of the token loop watchdog (`check_loop_watchdog`).
    pub(super) loop_scan_buf: String,
    /// 2026-09-26: Set when the token loop watchdog, the SimHash guard or the
    /// orphan-suppression streak cuts the stream.
    pub(super) loop_watchdog_triggered: bool,
    /// 2026-10-12: Tokens that have reached `handle_token` (completion tokens, thinking
    /// included, like the scheduler's `output_tokens`). The `min_tokens` floor of
    /// `METRALE_GUARD_HONOR_FORCED_LEN` reads it.
    pub(super) tokens_seen: usize,
    /// 2026-10-12: Content-guard fires that `METRALE_GUARD_HONOR_FORCED_LEN` let pass.
    /// Logged once per request by `handle_done`.
    pub(super) guard_suppressed: u32,
    /// 2026-09-26: Whether `handle_token` has logged the first non-empty delta batch.
    pub(super) first_result_logged: bool,
    /// 2026-09-26: Never set; `handle_done` reads it as a tool-call signal.
    pub(super) salvaged_tool_call: bool,
    /// 2026-09-26: SimHash semantic-loop guard (`process_detector_content`).
    pub(super) simhash_guard: crate::loop_simhash::SimHashLoopGuard,
    /// 2026-09-26: Content waiting for `simhash_guard.check()`, which runs at a sentence
    /// boundary or at 1024 bytes.
    pub(super) simhash_pending: String,
    /// 2026-09-26: Tool-arg dedup for complete tool calls (`ToolArgDedup::new()`).
    pub(super) tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup,
    /// 2026-09-26: Tool-arg dedup for streamed tool calls at `ToolCallEnd`.
    pub(super) tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup,
    /// 2026-09-26: `(name, args so far)` of each streamed tool call, keyed by index, until
    /// `ToolCallEnd` runs the dedup.
    pub(super) streaming_tool_args: HashMap<usize, (String, String)>,
    /// 2026-09-26: Tool calls in this response, counted by `bump_f12_tool_call_count`
    /// against `max_tool_calls_per_response`.
    pub(super) tool_calls_emitted_count: usize,
    /// 2026-09-26: Same-name run guard: `(last name, run length)` of successive tool calls,
    /// whatever their arguments; `None` before the first call. At
    /// `MAX_CONSEC_SAME_NAME_CALLS` the response is ended.
    pub(super) name_run: Option<(String, u32)>,
    /// 2026-09-26: Set when a tool-call loop guard ends the response: the tool-arg dedup,
    /// the within-response dedup, the same-name run cap, or the in-think leak cut.
    /// `handle_done` then reports `"length"` although tool calls were emitted.
    pub(super) tool_loop_capped: bool,
    /// 2026-09-26: The guard that cut the stream: set by the stream-side guards in
    /// `handle_token.rs`, otherwise taken from the scheduler's `Done.guard_stop`
    /// (`mod.rs`). `resolve_wire_finish_reason` maps a value to `"length"` when no
    /// higher rung applies, and the `--dump` record shows it.
    pub(super) guard_stop: Option<&'static str>,
    /// 2026-09-26: Set once the corrective content chunk (empty required parameters, or a
    /// garbled parameter boundary) has been sent; at most one per response.
    pub(super) corrective_hint_sent: bool,
    /// 2026-09-26: Cooperative cancellation shared with the scheduler, which finishes the
    /// sequence at its next token (`emit_token`, `scheduler/emit_step/token.rs`). Stored
    /// by the stop-string match and the sites `cancel_guard_tests` lists; the
    /// within-response dedup and the same-name run cap set `tool_loop_capped` without
    /// storing it.
    pub(super) cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// 2026-09-26: Rolling tail (at most 256 bytes) of cleaned reasoning for the in-think
    /// tool-call leak scanner, so an opener split across deltas is still found. Filled
    /// only in the thinking phase of a request with tools.
    pub(super) reasoning_xml_scan_buf: String,
    /// 2026-09-26: Set when the scanner's opener count reaches the threshold; afterwards
    /// thinking-phase tokens emit nothing.
    pub(super) reasoning_xml_leak_detected: bool,
    /// 2026-09-26: Tool-call openers the scanner has counted in this stream's reasoning,
    /// compared against `ChatLevers::in_think_leak_openers`.
    pub(super) reasoning_xml_opener_hits: u32,
    /// 2026-09-26: Streaming tool-call detector (`Some` iff `tools_active`).
    pub(super) detector: Option<tool_parser::StreamingToolDetector>,
    /// 2026-09-26: True once the think-end token has arrived, or from the start when the
    /// request did not enable thinking; a re-opened `<think>` clears it.
    pub(super) thinking_done: bool,
    /// 2026-09-26: Stays empty: only the `ctx.tool_retry_enabled` branch writes it, and
    /// that flag is always `false`.
    pub(super) buffered_tool_chunks: std::collections::HashMap<usize, Vec<crate::ir::StreamDelta>>,
    /// 2026-09-26: Never set: only the `ctx.tool_retry_enabled` branch sets it.
    pub(super) pending_retry: Option<PendingRetry>,
    /// 2026-09-26: `return_token_ids`: ids of streamed tokens not yet attached to a chunk,
    /// one per `handle_token` call, drained onto the next client-visible chunk or the
    /// `Finish` delta. Empty unless the request opted in.
    pub(super) pending_token_ids: Vec<u32>,
}

/// 2026-09-26: Built only in the `ctx.tool_retry_enabled` branch of `tool_handlers.rs`,
/// which never runs.
pub(super) struct PendingRetry {
    pub(super) errors_summary: String,
    pub(super) failed_idx: usize,
}

impl StreamState {
    pub(super) fn new(
        tools_active: bool,
        enable_thinking: bool,
        cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        tool_defs: Vec<tool_parser::ToolDefinition>,
    ) -> Self {
        Self {
            all_toks: Vec::new(),
            emitted: 0,
            content_decoded: String::new(),
            detok_prefix_offset: 0,
            detok_read_offset: 0,
            content_decoder: None,
            accumulated_content: String::new(),
            stop_string_emitted_len: 0,
            refusal_scan_buf: String::new(),
            stop_string_triggered: false,
            stop_string_matched: false,
            suppressing_param_leak: false,
            suppress_streak_tokens: 0,
            inside_envelope: false,
            reasoning_inside_envelope: false,
            tag_scan_buf: String::new(),
            reasoning_suppressing_leak: false,
            reasoning_tag_scan_buf: String::new(),
            loop_scan_buf: String::new(),
            loop_watchdog_triggered: false,
            tokens_seen: 0,
            guard_suppressed: 0,
            first_result_logged: false,
            salvaged_tool_call: false,
            simhash_guard: crate::loop_simhash::SimHashLoopGuard::new(),
            simhash_pending: String::new(),
            tool_arg_dedup: crate::tool_arg_dedup::ToolArgDedup::new(),
            tool_arg_dedup_within: crate::tool_arg_dedup::ToolArgDedup::with_params(4, 2, 3),
            streaming_tool_args: HashMap::new(),
            tool_calls_emitted_count: 0,
            name_run: None,
            tool_loop_capped: false,
            guard_stop: None,
            corrective_hint_sent: false,
            cancel_flag,
            reasoning_xml_scan_buf: String::new(),
            reasoning_xml_leak_detected: false,
            reasoning_xml_opener_hits: 0,
            detector: if tools_active {
                Some(tool_parser::StreamingToolDetector::new_with_tools(
                    tool_defs,
                ))
            } else {
                None
            },
            thinking_done: !enable_thinking,
            buffered_tool_chunks: HashMap::new(),
            pending_retry: None,
            pending_token_ids: Vec::new(),
        }
    }

    /// 2026-09-26: A client stop sequence matched: record it for `handle_done` and set
    /// `cancel_flag`, so the scheduler stops generating.
    pub(super) fn note_stop_string_match(&mut self) {
        self.stop_string_matched = true;
        self.cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
mod stop_string_match_tests {
    use super::StreamState;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn note_stop_string_match_records_and_cancels() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut s = StreamState::new(false, false, flag.clone(), Vec::new());
        assert!(!s.stop_string_matched, "fresh stream: no match yet");
        assert!(!flag.load(Ordering::Acquire));
        s.note_stop_string_match();
        assert!(
            s.stop_string_matched,
            "match must be recorded for handle_done"
        );
        assert!(
            flag.load(Ordering::Acquire),
            "scheduler cancel flag must flip so generation stops"
        );
    }
}
