// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The streaming translator: `ir::StreamDelta`s → Anthropic SSE
//! events with `message_start` / `content_block_*` / `message_delta` /
//! `message_stop` framing.
//!
//! Owner: server (Anthropic adapter).
//! Invariants:
//! - `message_start` is emitted at most once per translator.
//! - After a Finish or Error delta, `finalize` and a further Finish delta
//!   emit nothing.

use axum::response::sse::Event;

use super::helpers::*;

/// 2026-09-26: One Anthropic SSE event (event name and JSON data), kept typed
/// so tests can inspect it. [`SseEvent::to_axum_event`] converts it for the
/// wire.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SseEvent {
    pub(super) event: String,
    pub(super) data: serde_json::Value,
}

impl SseEvent {
    /// 2026-09-26: The axum event; its data is the JSON text, or empty if
    /// serialization fails.
    pub(super) fn to_axum_event(&self) -> Event {
        Event::default()
            .event(&self.event)
            .data(serde_json::to_string(&self.data).unwrap_or_default())
    }
}

/// 2026-09-26: Which content block the translator has open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OpenBlock {
    /// 2026-09-26: No block is open: before the first block, or after a
    /// `content_block_stop`.
    None,
    /// 2026-09-26: A text block is open at `block_idx`.
    Text,
    /// 2026-09-26: A `thinking` block is open at `block_idx`; reasoning
    /// deltas stream into it as `thinking_delta` events.
    Thinking,
    /// 2026-09-26: A `tool_use` block is open at `block_idx` for this
    /// tool-call `index` of the delta stream. Argument fragments for that
    /// index go into it; a start for another index closes it first.
    ToolUse(usize),
}

/// 2026-09-26: State for one streaming `/v1/messages` response. `on_delta`
/// turns each delta into zero or more events, closing the open block before
/// a block of another kind (or another tool call) opens. A Finish delta
/// closes the open block and emits `message_delta` and `message_stop`.
pub struct AnthropicTranslator {
    model: String,
    msg_started: bool,
    /// 2026-09-26: `msg_<uuid>`, minted in `new`; the delta stream carries
    /// no id.
    msg_id: String,
    block_idx: u32,
    open_block: OpenBlock,
    /// 2026-09-26: Tool-call indexes whose `content_block_start` was sent
    /// and whose block has not been closed. A `ToolCallStart` for an index
    /// in this set opens nothing.
    tool_started: std::collections::HashMap<usize, ()>,
    completion_tokens: usize,
    prompt_tokens: usize,
    cached_prompt_tokens: usize,
    finished: bool,
}

/// 2026-09-26: The message text inside the error payload of a
/// `StreamDelta::Error`.
///
/// The payload is an OpenAI error body (`{"error":{"message":..}}`), which the
/// OpenAI surface sends as SSE data unchanged. Anthropic's `error` event has
/// its own envelope, so only the message is taken. A payload of any other
/// shape is passed through whole.
fn error_detail(payload: &str) -> String {
    serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| payload.to_string())
}

impl AnthropicTranslator {
    pub(super) fn new(model: String) -> Self {
        Self {
            model,
            msg_started: false,
            msg_id: format!("msg_{}", crate::ids::uuid_v4()),
            block_idx: 0,
            open_block: OpenBlock::None,
            tool_started: std::collections::HashMap::new(),
            completion_tokens: 0,
            prompt_tokens: 0,
            cached_prompt_tokens: 0,
            finished: false,
        }
    }

    /// 2026-09-26: The final `message_delta.usage`, with input, cached and
    /// output tokens. Usage arrives only on the Finish delta, and
    /// `message_start` is normally sent before it with `input_tokens: 0`, so
    /// the input count is reported here.
    fn final_usage(&self) -> serde_json::Value {
        serde_json::json!({
            "input_tokens": self.prompt_tokens,
            "cache_read_input_tokens": self.cached_prompt_tokens,
            "output_tokens": self.completion_tokens,
        })
    }

    fn make_event(ev_type: &str, data: serde_json::Value) -> SseEvent {
        SseEvent {
            event: ev_type.to_string(),
            data,
        }
    }

    /// 2026-09-26: Close the open block, if any, advance `block_idx`, and
    /// return its `content_block_stop`; `None` when no block was open.
    /// Closing a tool block also removes its index from `tool_started`, so a
    /// later `ToolCallStart` for that index opens a new block.
    pub(super) fn close_open_block(&mut self) -> Option<SseEvent> {
        match self.open_block {
            OpenBlock::None => None,
            OpenBlock::Text | OpenBlock::Thinking => {
                let ev = Self::make_event(
                    "content_block_stop",
                    serde_json::json!({
                        "type": "content_block_stop",
                        "index": self.block_idx,
                    }),
                );
                self.open_block = OpenBlock::None;
                self.block_idx += 1;
                Some(ev)
            }
            OpenBlock::ToolUse(oa_idx) => {
                self.tool_started.remove(&oa_idx);
                let ev = Self::make_event(
                    "content_block_stop",
                    serde_json::json!({
                        "type": "content_block_stop",
                        "index": self.block_idx,
                    }),
                );
                self.open_block = OpenBlock::None;
                self.block_idx += 1;
                Some(ev)
            }
        }
    }

    pub(super) fn ensure_message_start(&mut self, out: &mut Vec<SseEvent>) {
        if self.msg_started {
            return;
        }
        let id = self.msg_id.clone();
        out.push(Self::make_event(
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": serde_json::Value::Null,
                    "stop_sequence": serde_json::Value::Null,
                    "usage": {
                        "input_tokens": self.prompt_tokens,
                        "output_tokens": 0,
                    },
                },
            }),
        ));
        self.msg_started = true;
    }

    /// 2026-09-26: Translate one delta, pushing the resulting events onto
    /// `out`.
    pub(super) fn on_delta(&mut self, d: &crate::ir::StreamDelta, out: &mut Vec<SseEvent>) {
        use crate::ir::StreamDelta;
        match d {
            StreamDelta::Reasoning { text, .. } if !text.is_empty() => {
                self.ensure_message_start(out);
                if !matches!(self.open_block, OpenBlock::Thinking) {
                    if let Some(stop) = self.close_open_block() {
                        out.push(stop);
                    }
                    out.push(Self::make_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": self.block_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        }),
                    ));
                    self.open_block = OpenBlock::Thinking;
                }
                out.push(Self::make_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.block_idx,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    }),
                ));
            }
            StreamDelta::Content { text, .. } if !text.is_empty() => {
                self.ensure_message_start(out);
                if !matches!(self.open_block, OpenBlock::Text) {
                    if let Some(stop) = self.close_open_block() {
                        out.push(stop);
                    }
                    out.push(Self::make_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": self.block_idx,
                            "content_block": {"type": "text", "text": ""},
                        }),
                    ));
                    self.open_block = OpenBlock::Text;
                }
                out.push(Self::make_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.block_idx,
                        "delta": {"type": "text_delta", "text": text},
                    }),
                ));
            }
            StreamDelta::ToolCallStart { index, id, name } => {
                self.ensure_message_start(out);
                let need_start = !self.tool_started.contains_key(index);
                if need_start
                    && !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == *index)
                {
                    if let Some(stop) = self.close_open_block() {
                        out.push(stop);
                    }
                    out.push(Self::make_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": self.block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": {},
                            },
                        }),
                    ));
                    self.open_block = OpenBlock::ToolUse(*index);
                    self.tool_started.insert(*index, ());
                }
            }
            StreamDelta::ToolCallArgs {
                index, fragment, ..
            } => {
                if fragment.is_empty() {
                    return;
                }
                self.ensure_message_start(out);
                if !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == *index) {
                    // 2026-09-26: A fragment for a tool call whose block is
                    // not open is dropped with a warning. Opening a block
                    // for it would send a second `tool_use` for one call.
                    tracing::warn!(
                        target: "anthropic_translator",
                        oa_idx = index,
                        current_block = ?self.open_block,
                        arg_fragment_len = fragment.len(),
                        "dropping tool-call argument fragment for non-open tool block"
                    );
                    return;
                }
                out.push(Self::make_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.block_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": fragment,
                        },
                    }),
                ));
            }
            StreamDelta::Finish { reason, usage, .. } => {
                if self.finished {
                    return;
                }
                self.prompt_tokens = usage.prompt_tokens;
                self.completion_tokens = usage.completion_tokens;
                self.cached_prompt_tokens = usage.cached_prompt_tokens;
                self.ensure_message_start(out);
                if let Some(stop) = self.close_open_block() {
                    out.push(stop);
                }
                out.push(Self::make_event(
                    "message_delta",
                    serde_json::json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": anthropic_stop_reason(reason),
                            "stop_sequence": serde_json::Value::Null,
                        },
                        "usage": self.final_usage(),
                    }),
                ));
                out.push(Self::make_event(
                    "message_stop",
                    serde_json::json!({"type": "message_stop"}),
                ));
                self.finished = true;
            }
            StreamDelta::Refusal { .. } => {}
            // 2026-09-26: A stream-level failure (`handle_error` sends it as
            // a delta) ends the message with an `error` event. `finished` is
            // set so the `finalize` that `anthropic_sse_from_deltas` runs
            // next adds no `message_delta`/`message_stop`, which would
            // report the partial answer as a completed turn.
            StreamDelta::Error { message } => {
                self.ensure_message_start(out);
                if let Some(stop) = self.close_open_block() {
                    out.push(stop);
                }
                out.push(Self::make_event(
                    "error",
                    serde_json::json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": error_detail(message)},
                    }),
                ));
                self.finished = true;
            }
            StreamDelta::Content { .. } | StreamDelta::Reasoning { .. } => {}
        }
    }

    /// 2026-09-26: Close a stream that ended without a Finish or Error delta.
    /// The open block is closed and the message ends with
    /// `stop_reason: "max_tokens"`, which reports the output as cut short;
    /// `end_turn` would report the turn complete. `convert_stop_reason` maps
    /// the server deadline to the same reason. Does nothing once a Finish or
    /// Error delta has set `finished`.
    pub(super) fn finalize(&mut self, out: &mut Vec<SseEvent>) {
        if self.finished {
            return;
        }
        self.ensure_message_start(out);
        if let Some(stop) = self.close_open_block() {
            out.push(stop);
        }
        out.push(Self::make_event(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "max_tokens",
                    "stop_sequence": serde_json::Value::Null,
                },
                "usage": self.final_usage(),
            }),
        ));
        out.push(Self::make_event(
            "message_stop",
            serde_json::json!({"type": "message_stop"}),
        ));
        self.finished = true;
    }
}
