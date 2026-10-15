// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Read-only per-stream context for the `StreamEvent` arms. The streaming
//! closure owns it and lends it to each handler beside `&mut StreamState`.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use std::collections::HashSet;
use std::sync::Arc;

use crate::AppState;
use crate::tool_parser;

pub(super) struct StreamCtx {
    /// 2026-09-26: Holds the active-requests gauge for the stream's lifetime. `StreamCtx`
    /// is moved into the streaming closure, so the guard drops once, whether the stream
    /// completes, fails, or is dropped when the client disconnects. Never read.
    pub(super) _active_guard: crate::metrics::ActiveRequestGuard,
    pub(super) state: Arc<AppState>,
    pub(super) model: String,
    pub(super) id: String,
    pub(super) prompt_len: usize,
    pub(super) enable_thinking: bool,
    pub(super) tool_defs_for_backfill: Vec<tool_parser::ToolDefinition>,
    pub(super) cwd_for_normalize: Option<String>,
    pub(super) stop_strings: Vec<String>,
    /// 2026-09-26: Always `false` (`mod.rs`), so the retry and chunk-buffering branches in
    /// `tool_handlers.rs` never run.
    pub(super) tool_retry_enabled: bool,
    /// 2026-09-26: The rendered prompt tokens, shared with the scheduler request. Never
    /// read.
    pub(super) prompt_tokens: Arc<Vec<u32>>,
    /// 2026-09-26: Always empty (`mod.rs`). Never read.
    pub(super) prompt_vocab: Arc<HashSet<String>>,
    pub(super) grammar_spec: Option<crate::api::inference_types::GrammarSpec>,
    pub(super) max_tokens: usize,
    pub(super) timeout_at: Option<std::time::Instant>,
    /// 2026-09-26: Bytes held back from each delta for stop-string matching: the longest
    /// stop string's byte length minus one, so a stop string split across two decoded
    /// chunks is never sent in part. 0 when `stop_strings` is empty.
    pub(super) stop_string_buffer_len: usize,
    pub(super) leak_markers: tool_parser::LeakMarkers,
    /// 2026-09-26: Whether the tool parser wants schema-driven coercion of parsed
    /// arguments (`ToolCallParser::wants_typed_arguments`; true for `qwen3_xml`,
    /// `qwen3_coder`, `poolside_v1` and `deepseek_v4_dsml`).
    pub(super) wants_typed_arguments: bool,
    pub(super) max_tool_calls_per_response: usize,
    pub(super) req_return_token_ids: bool,
    pub(super) req_ctx: Option<crate::rate_limiter::RequestContext>,
    pub(super) dump_seq: Option<u64>,
    /// 2026-10-12: The request's `ignore_eos` and `min_tokens`, read by the stream-side
    /// content guards under `METRALE_GUARD_HONOR_FORCED_LEN`.
    pub(super) ignore_eos: bool,
    pub(super) min_tokens: usize,
}
