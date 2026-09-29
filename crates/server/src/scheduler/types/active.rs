// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ActiveSeq`, its guard-stop markers and the budget decrement.
//!
//! Owner: scheduler.
//! Invariants:
//! - `remaining` is decremented only through `consume_budget`, which returns false at 0 instead of wrapping.

#![allow(dead_code)]

use super::*;

/// 2026-09-25: `ActiveSeq::guard_stop` marker for the server-side request deadline
/// (`--request-timeout`, or the request's own timeout). `derive_finish_reason`
/// checks for this exact value first and reports `finish_reason="timeout"`.
pub(in crate::scheduler) const GUARD_STOP_REQUEST_TIMEOUT: &str = "request_timeout";

/// 2026-09-25: The `<tool_response>` hard stop fired: the model emitted that control
/// token (a post-tool-call runaway). Naming the stop makes the wire reason
/// `"length"` instead of `"stop"`, so the client knows the server cut the
/// response short.
pub(in crate::scheduler) const GUARD_STOP_TOOL_RESPONSE: &str = "tool_response_hard_stop";

/// 2026-09-25: The stray-`</think>` watchdog fired: 50 consecutive `</think>` outside a
/// thinking span forced the turn closed. The site skips the token rather than
/// pushing it, so the last token is plain content and, without a name, this
/// server cut would reach `derive_finish_reason`'s final `"stop"`. The agentic
/// harness's `was_cut_off()` grants a recovery turn only on `"length"`.
/// The `<|im_start|>` stop needs no name: `tokenizer_runtime.rs` adds that
/// token to the EOS tokens, the stop pushes it, and `derive_finish_reason`
/// checks the EOS rung before any guard name, so it reports `"stop"`.
pub(in crate::scheduler) const GUARD_STOP_THINK_SKIP: &str = "think_skip_watchdog";

/// 2026-09-25: The inter-tool prose budget ended the turn. Both decode paths
/// (`decode_logits_content` and `emit_step`) stamp it, so the wire reason is
/// `"length"` rather than `"stop"`, and the name reaches
/// `StreamEvent::Done.guard_stop`.
pub(in crate::scheduler) const GUARD_STOP_INTER_TOOL_PROSE: &str = "inter_tool_prose_budget";

/// 2026-09-25: The content-loop watchdog hard-stopped: rollback was declined, or the
/// MTP path (`emit_token`), which does not roll back, caught the loop. A
/// degeneration cut is a server truncation and reports `"length"`.
pub(in crate::scheduler) const GUARD_STOP_CONTENT_LOOP: &str = "content_loop_watchdog";

/// 2026-09-25: The post-`</think>` content cap (`max_post_think_content_tokens`) ended a
/// response with a grammar attached (a tool call, `json_object` or
/// `json_schema` grammar); reported as `"length"`.
pub(in crate::scheduler) const GUARD_STOP_POST_THINK_CAP: &str = "post_think_content_cap";

/// 2026-09-25: A sequence past prefill: its request, sampling, thinking, tool and guard state.
pub(in crate::scheduler) struct ActiveSeq {
    pub seq: SequenceState,
    pub session_hash: u64,
    pub last_token: u32,
    pub output_tokens: Vec<u32>,
    pub remaining: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    pub finished: bool,
    /// 2026-09-25: Set when the sequence is being retired because an inference step failed,
    /// not because the model finished. `finish_sequence` sends this to the client
    /// as an error instead of a normal completion.
    pub error: Option<String>,
    /// 2026-09-25: Which server-side guard force-finished this sequence (e.g.
    /// "fuzzy_repetition"), if any. It sets the wire reason
    /// (`derive_finish_reason`) and reaches `StreamEvent::Done.guard_stop`,
    /// which streaming chat writes into the --dump body.
    pub guard_stop: Option<&'static str>,
    /// 2026-09-25: Provisional `</parameter>` close progress inside a parameter
    /// value body (0 = none, 1 = saw `</`, 2 = saw `</` `parameter`). The body
    /// exits only on the full `</` `parameter` `>` sequence, so close tags such
    /// as `</div>` in a value stay value content instead of turning the rest
    /// of the value into envelope tokens that count toward
    /// `MAX_TOOL_BODY_TOKENS`.
    pub param_close_pending: u8,
    pub sink: ResponseSink,
    /// 2026-09-25: Cooperative cancellation flag, `Some` only for streaming requests.
    /// The chat-stream guards set it; `emit_step::emit_token` checks it on
    /// every call and finishes the sequence when it is set.
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_penalty_window: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub lz_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Vec<u32>,
    pub logit_bias: Vec<(u32, f32)>,
    /// 2026-09-25: Tracks whether the model is inside `<think>...</think>` reasoning.
    pub inside_thinking: bool,
    /// 2026-09-25: Whether the request opted into thinking mode (`enable_thinking=true`).
    /// When false but the model opens `<think>` on its own, the thinking tokens
    /// are not streamed to the client.
    pub enable_thinking: bool,
    /// 2026-09-25: Thinking-token budget: once `thinking_tokens` reaches it,
    /// `force_end_thinking` is armed. `None` means no budget.
    pub thinking_budget: Option<u32>,
    /// 2026-09-25: Per-request override for the content-loop detector. `None` falls back
    /// to the server's settings (`WatchdogParams::content_loop_params`).
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// 2026-09-25: Per-server spontaneous-thinking budget: `--max-thinking-budget`, else
    /// MODEL.toml `[behavior].max_thinking_budget`.
    pub spontaneous_think_budget: u32,
    /// 2026-09-25: Thinking tokens generated so far, not counting the closing `</think>`.
    pub thinking_tokens: u32,
    /// 2026-09-25: Armed when the thinking budget runs out, the thinking-loop watchdog
    /// fires or `F2ConfidenceEarlyStop` trips. `ForcedThinkEndInjector` then
    /// forces `</think>`, deferring to a sentence boundary or a fence close
    /// (`should_inject_think_end`).
    pub force_end_thinking: bool,
    /// 2026-09-25: Whether the server, not a sampled `</think>`, closed the current
    /// thinking span: `emit_token` copies `force_end_thinking` into it at the
    /// `</think>` commit, and the non-MTP path sets it when it honours an EOS
    /// inside thinking. Nothing reads it.
    pub think_force_closed: bool,
    /// 2026-09-25: Steps `ForcedThinkEndInjector` has deferred the forced `</think>`
    /// while waiting for a sentence boundary or a fence close. Reset when
    /// `force_end_thinking` is armed and when thinking closes. At
    /// [`crate::scheduler::confidence::MAX_SENTENCE_DEFER_TOKENS`] the
    /// injector stops deferring.
    pub sentence_defer_count: u32,
    /// 2026-09-25: Consecutive thinking tokens with top-1 softmax probability >= 0.95
    /// (`F2ConfidenceEarlyStop`).
    pub consecutive_confident: u32,
    /// 2026-09-25: True while the current thinking block has an unclosed ``` fence;
    /// the non-MTP decode path toggles it on each sampled fence token. While
    /// set, `ForcedThinkEndInjector` defers a forced `</think>` unless the
    /// defer limit is reached.
    pub in_code_fence: bool,
    /// 2026-09-25: Token id of `</think>`, read by `emit_token`.
    pub think_end_token: Option<u32>,
    /// 2026-09-25: Token id of `<think>` (spontaneous-thinking detection in `emit_token`).
    pub think_start_token: Option<u32>,
    /// 2026-09-25: True once thinking has closed. Its value at birth depends on the
    /// request (see the prefill steps), and re-entering thinking resets it.
    pub think_ended: bool,
    /// 2026-09-25: One-shot flag: set when thinking closes, cleared on the next token.
    pub think_just_ended: bool,
    /// 2026-09-25: Tokens `emit_token` has emitted outside thinking once `think_ended` is
    /// set; reset when the model re-enters thinking. `spec_dispatch_eligible`
    /// reads it to keep the first `METRALE_DFLASH_RESUME_GUARD` tokens on
    /// serial decode.
    pub post_think_emitted: u32,
    /// 2026-09-25: Adaptive speculation (`METRALE_DFLASH_ADAPTIVE=1`) state. Reset on
    /// preemption and on swap-in, so a resumed sequence re-measures. See
    /// `adaptive_spec`.
    pub spec_adapt: crate::scheduler::adaptive_spec::AdaptState,
    /// 2026-09-29: A146, spec-in-think parity: the per-position post-pipeline
    /// accumulators the last verify pick window left, consumed one position
    /// at a time by `emit_token` (see `think_commit::SpecThinkTrail`).
    /// Transient: empty outside a verify commit run, reset on swap-in.
    pub spec_think_trail:
        std::collections::VecDeque<crate::scheduler::think_commit::SpecThinkTrail>,
    /// 2026-09-25: Consecutive `</think>` tokens skipped outside thinking; the 50th ends
    /// the turn.
    pub think_skip_count: u32,
    /// 2026-09-25: Token id of `</tool_call>`. On the non-MTP path, emitting it finishes
    /// the sequence only when there is no grammar and the request declared no
    /// tools.
    pub tool_call_end_token: Option<u32>,
    /// 2026-09-25: While true, EOS is suppressed; a `<tool_call>` outside thinking clears
    /// it.
    pub require_tool_call: bool,
    /// 2026-09-25: Sticky "this is a tool request" flag, set at birth when a grammar is
    /// attached or `use_legacy_tool_call` holds. Unlike
    /// `grammar_state.is_some()` it survives `emit_token` dropping the grammar
    /// mid-response, so the inter-tool prose budget keeps working. False for
    /// plain chat, which is never prose-capped.
    pub tool_request: bool,
    /// 2026-09-25: The request declared tools. Its only reader is the non-MTP
    /// `</tool_call>` check: with no grammar, a closed tool call finishes the
    /// sequence only when this is false, so a tool request can go on to emit
    /// parallel calls.
    pub tools_present: bool,
    /// 2026-09-25: Token id of `<tool_call>`.
    pub tool_call_start_token: Option<u32>,
    /// 2026-09-25: Set when a `<tool_call>` outside thinking clears `require_tool_call`.
    pub tool_call_opened: bool,
    /// 2026-09-25: True between `<tool_call>` and `</tool_call>` outside thinking
    /// (`update_tool_param_state`).
    pub inside_tool_body: bool,
    /// 2026-09-25: True once a `</tool_call>` has been emitted outside thinking; gates the
    /// tool EOS escape (`SchedLevers::tool_eos_escape`) and the post-completion
    /// opener count.
    pub tool_call_completed: bool,
    /// 2026-09-25: `<tool_call>` openers emitted, outside thinking, after a tool call
    /// completed, counted on the non-MTP path. A model that loops whole
    /// `<tool_call>…</tool_call>` blocks never trips the envelope streak,
    /// because each block closes; at `MAX_POST_COMPLETION_TOOL_OPENS`
    /// (`decode_logits_step.rs`) the response is ended.
    pub post_completion_tool_opens: u32,
    /// 2026-09-25: Envelope tokens (inside the tool body but outside a parameter value)
    /// since the tool body opened or the last confirmed `</parameter>`. Above
    /// `MAX_TOOL_BODY_TOKENS` (`emit_step/tool_param.rs`) the response ends
    /// with guard `tool_envelope_stuck`.
    pub tool_body_streak_tokens: u32,
    /// 2026-09-25: True between the model emitting `<parameter=KEY>` and the matching
    /// confirmed `</parameter>` (see `param_close_pending`). Read by
    /// `logit_processors::b1_margin`, the adadec diagnostic and the per-step
    /// logit dump.
    pub inside_parameter_body: bool,
    /// 2026-09-25: Count of body tokens `update_tool_param_state` has committed
    /// inside the current parameter value: +1 for an ordinary body token,
    /// or +2 / +3 when a provisional `</parameter>` close attempt turns
    /// out to be false and the held-back tokens are re-added in bulk (see
    /// `param_close_pending`). Reset to 0 on the opener, on a confirmed
    /// close and on `</tool_call>`. Read by the same consumers as
    /// `inside_parameter_body`.
    pub param_body_chars_emitted: u32,
    /// 2026-09-25: When true, `ToolCallDuringThinkingMask` lowers the `<tool_call>` logit
    /// by 12 outside thinking, and `spec_dispatch_eligible` keeps the sequence
    /// off speculative dispatch.
    pub suppress_tool_call: bool,
    /// 2026-09-25: The request disabled MTP; `spec_dispatch_eligible` keeps the
    /// sequence off speculative dispatch.
    pub disable_mtp: bool,
    /// 2026-09-25: Per-request MTP accept counters
    /// ([`crate::scheduler::mtp_accept_debug::RequestAccept`]); `finish_sequence`
    /// reports `accepted_total()` as the accepted prediction tokens.
    pub mtp_acct: crate::scheduler::mtp_accept_debug::RequestAccept,
    /// 2026-09-25: Set by the non-MTP path's `handle_content_token` on the first content
    /// token. Nothing reads it.
    pub content_started: bool,
    /// 2026-09-25: Content tokens emitted outside thinking, counted by both decode paths
    /// and reduced by a rollback.
    pub content_tokens: u32,
    /// 2026-09-25: Content tokens outside a tool body since the last `<tool_call>` opened,
    /// counted only for tool requests (`tool_request`).
    pub prose_tokens_since_last_tool: u32,
    /// 2026-09-25: How many times the thinking-loop watchdog has fired. The non-MTP path
    /// shifts the budget of each spontaneous `<think>` re-entry right by this
    /// count (at most 4).
    pub think_watchdog_fires: u32,
    /// 2026-09-25: How many times a degeneration watchdog has rolled this sequence
    /// back to a boundary and re-steered. At
    /// [`metrale_kernels::ROLLBACK_RESTEER_CAP`] rollback is declined and the
    /// watchdog hard-stops instead. See
    /// [`crate::scheduler::rollback::rollback_to_boundary`].
    pub rollback_count: u32,
    /// 2026-09-25: Decode-time SSM-snapshot ring that lets a rollback restore a hybrid
    /// model's recurrent state; disabled (`capacity == 0`) for pure-attention
    /// models. See [`crate::scheduler::ssm_decode_ring::SsmDecodeRing`].
    pub ssm_rollback_ring: crate::scheduler::ssm_decode_ring::SsmDecodeRing,
    /// 2026-09-25: Grammar matcher for constrained decoding; `None` when the request has
    /// no grammar or `emit_token` dropped it mid-response.
    pub grammar_state: Option<GrammarState>,
    /// 2026-09-25: Draft tokens awaiting verification.
    pub pending_drafts: Vec<u32>,
    /// 2026-09-25: Top-1 log-probability for each entry of [`Self::pending_drafts`], in
    /// the same order: the D-Cut ranking key (`mtp_dcut`). Only the batched
    /// propose paths fill it; `mtp_dcut` treats a length mismatch as "not
    /// measured" and leaves the sequence at full depth, and truncates it
    /// together with `pending_drafts`.
    pub pending_draft_conf: Vec<f32>,
    /// 2026-09-25: Time of the last token emission; the scheduling policy measures
    /// time-between-tokens from it.
    pub last_token_time: Instant,
    /// 2026-09-25: When the scheduler began the request's prefill; TTFT is
    /// `decode_start - request_start`.
    pub request_start: Instant,
    /// 2026-09-25: When the sequence entered decode, after prefill; decode time is
    /// measured from it.
    pub decode_start: Instant,
    /// 2026-09-25: Sampling seed; each step uses it offset by the output length.
    pub seed: Option<u64>,
    /// 2026-09-25: Number of top logprobs to return per token. `None` means disabled.
    pub top_logprobs: Option<u8>,
    /// 2026-09-25: Per-token logprobs collected so far; `finish_sequence` hands them to
    /// the response, and streaming sends each with its token.
    pub logprobs_data: Vec<crate::api::TokenLogprobs>,
    /// 2026-09-25: Request timeout deadline. `None` means no timeout.
    pub timeout_at: Option<Instant>,
    pub adaptive: crate::adaptive_sampler::AdaptiveSamplingState,
    /// 2026-09-25: Number of prompt tokens served by the prefix cache (no prefill cost).
    pub cached_prompt_tokens: u32,
    /// 2026-09-25: Decode-preemption starvation guard: this sequence is not chosen as a
    /// KV-preemption victim again until `output_tokens.len()` reaches this
    /// threshold. Set on resume (requeue re-prefill and swap-in) to
    /// `output_tokens.len() + preempt::PREEMPT_IMMUNITY_TOKENS`; 0 (fresh
    /// sequences) means no immunity. It is compared against the output length,
    /// so nothing is decremented per step.
    pub preempt_immune_until_tokens: usize,
}

impl ActiveSeq {
    /// 2026-09-25: Consume one token of generation budget, for both decode paths
    /// (`emit_step::emit_token` and the non-MTP `decode_logits_step`).
    ///
    /// A decrement at 0 means a token was processed after the length stop
    /// should have fired. A bare `-= 1` would wrap to usize::MAX in release
    /// builds and panic in debug builds, so this logs and finishes instead.
    pub fn consume_generation_budget(&mut self) {
        if !consume_budget(&mut self.remaining) {
            tracing::warn!(
                output_tokens = self.output_tokens.len(),
                "generation budget decremented at 0 (token processed after \
                 length stop should have fired) — finishing sequence instead \
                 of wrapping"
            );
            self.finished = true;
        }
    }
}

/// 2026-09-25: Decrement `remaining` by one. Returns false (without touching
/// `remaining`) when it is already 0: the caller must finish the sequence
/// rather than wrap.
pub(in crate::scheduler) fn consume_budget(remaining: &mut usize) -> bool {
    if *remaining == 0 {
        return false;
    }
    *remaining -= 1;
    true
}
