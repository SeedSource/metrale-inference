// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Shared fixtures for the scheduler tests: `ActiveSeq` and `PrefillInProgress` builders and the `PreemptStubModel` stub.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::types::{ActiveSeq, ResponseSink};
use super::{DEFAULT_LZ_PENALTY, SsmDecodeRing};
use crate::api::InferenceResponse;
use anyhow::Result;
use metrale_model_engine::traits::SequenceState;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;

mod model_impl;
mod requests;
pub(super) use requests::blocking_request;

pub(super) const EOS: &[u32] = &[151645];
const TOOL_END: Option<u32> = Some(151658);

pub(super) type RespRx = tokio::sync::oneshot::Receiver<Result<InferenceResponse>>;

/// 2026-09-25: A real `ActiveSeq` with a blocking oneshot sink. It starts with
/// `finished: true`; `active_seq` builds an unfinished one.
pub(super) fn test_seq(
    output_tokens: Vec<u32>,
    remaining: usize,
    guard_stop: Option<&'static str>,
    seq_len: usize,
) -> (ActiveSeq, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let now = Instant::now();
    let mut seq = SequenceState::host_only(0);
    seq.seq_len = seq_len;
    let a = ActiveSeq {
        seq,
        session_hash: 0,
        last_token: output_tokens.last().copied().unwrap_or(0),
        output_tokens,
        remaining,
        min_tokens: 7,
        eos_tokens: EOS.to_vec(),
        finished: true,
        error: None,
        guard_stop,
        param_close_pending: 0,
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        inside_thinking: false,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        thinking_tokens: 0,
        force_end_thinking: false,
        think_force_closed: false,
        sentence_defer_count: 0,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: None,
        think_start_token: None,
        think_ended: false,
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        spec_think_trail: Default::default(),
        think_skip_count: 0,
        tool_call_end_token: TOOL_END,
        require_tool_call: false,
        tool_request: false,
        tools_present: false,
        tool_call_start_token: None,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        suppress_tool_call: false,
        disable_mtp: false,
        mtp_acct: Default::default(),
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        last_token_time: now,
        request_start: now,
        decode_start: now,
        seed: None,
        top_logprobs: None,
        logprobs_data: Vec::new(),
        timeout_at: None,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
        preempt_immune_until_tokens: 0,
    };
    (a, rx)
}

/// 2026-09-25: A `PrefillInProgress` with a blocking oneshot sink.
pub(super) fn test_prefill(prompt: Vec<u32>) -> (super::types::PrefillInProgress, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let p = super::types::PrefillInProgress {
        prompt_tokens: std::sync::Arc::new(prompt),
        session_hash: 0,
        seq: SequenceState::host_only(0),
        chunk_offset: 0,
        max_tokens: 16,
        min_tokens: 0,
        eos_tokens: EOS.to_vec(),
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        request_start: Instant::now(),
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_state: None,
        seed: None,
        top_logprobs: None,
        timeout_at: None,
        chunk_cap_logged: false,
    };
    (p, rx)
}

/// 2026-09-25: A real `PrefillInProgress` at `chunk_offset == 0`, identified by
/// `session_hash` so an ordering test can name the request it is tracking.
///
/// `eos_tokens` is empty and `temperature` is 0.0 on purpose: that is the
/// `sample_token` greedy fast path (`argmax_on_device`), which a stub `Model`
/// can answer without a device logits buffer. `max_tokens` is 8 (> 1) so
/// promotion pushes onto `active` instead of finishing immediately.
pub(super) fn test_prefill_ident(
    id: u64,
    prompt_len: usize,
) -> (super::types::PrefillInProgress, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let p = super::types::PrefillInProgress {
        prompt_tokens: std::sync::Arc::new(vec![1u32; prompt_len]),
        session_hash: id,
        seq: SequenceState::host_only(id as usize),
        chunk_offset: 0,
        max_tokens: 8,
        min_tokens: 0,
        eos_tokens: Vec::new(),
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        request_start: Instant::now(),
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        require_tool_call: false,
        tools_present: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_state: None,
        seed: None,
        top_logprobs: None,
        timeout_at: None,
        chunk_cap_logged: false,
    };
    (p, rx)
}

// 2026-09-25: preemption fixtures, shared by preempt_tests,
// prefill_error_delivery_tests, shutdown_drain_tests and swap_out_tests.

/// 2026-09-25: Scripted stub: `decode_batch` fails with the KV-exhausted error for the
/// first `fail_decodes` calls, then succeeds. Records frees, caches, prefills
/// and compactions so the tests can assert the side effects.
#[derive(Default)]
pub(super) struct PreemptStubModel {
    pub(super) fail_decodes: AtomicUsize,
    /// 2026-09-25: When set, `decode_batch` always fails with this message instead.
    pub(super) hard_error: Option<&'static str>,
    /// 2026-09-25: When set, `compact_sequence` always fails with this message.
    pub(super) fail_compact: Option<&'static str>,
    pub(super) compact_calls: AtomicUsize,
    pub(super) decode_calls: AtomicUsize,
    pub(super) freed_slots: Mutex<Vec<usize>>,
    pub(super) cached_seqs: AtomicUsize,
    pub(super) prefilled: Mutex<Vec<Vec<u32>>>,
    pub(super) vision_pad: Option<u32>,
    pub(super) free_blocks: AtomicUsize,
    pub(super) total_blocks: usize,
    pub(super) reclaimable: AtomicUsize,
    /// 2026-10-03: `kv_block_size`; `None` (default) is the trait default.
    pub(super) block_size: Option<usize>,
    /// 2026-10-04: When set, `argmax_on_device` returns it (the greedy
    /// token-0 sample) and `prefill_chunk` records the chunk and succeeds;
    /// unset (default), both fail as before.
    pub(super) first_token: Option<u32>,
    /// 2026-10-04: When non-empty, the logits `decode` / `decode_batch`
    /// return, one row per sequence: `decode` succeeds, `vocab_size` is the
    /// row length, `copy_logits_to_host` reads them as BF16 at the pointer's
    /// offset, and the argmax calls return each row's argmax.
    pub(super) logit_rows: Vec<Vec<f32>>,
}

impl PreemptStubModel {
    pub(super) fn failing(n: usize) -> Self {
        Self {
            fail_decodes: AtomicUsize::new(n),
            ..Default::default()
        }
    }

    /// 2026-09-25: A stub whose `compact_sequence` always fails.
    pub(super) fn failing_compact(msg: &'static str) -> Self {
        Self {
            fail_compact: Some(msg),
            ..Default::default()
        }
    }
}

/// 2026-09-25: An unfinished decode-active sequence at `slot` with `n_out` generated
/// tokens and a known prompt in `seq.tokens`.
pub(super) fn active_seq(slot: usize, n_out: usize) -> (ActiveSeq, super::test_support::RespRx) {
    let out: Vec<u32> = (100..100 + n_out as u32).collect();
    let (mut a, rx) = test_seq(out, 50, None, 4 + n_out);
    a.finished = false;
    a.seq.slot_idx = slot;
    // 2026-09-25: prompt [1,2,3,4] plus all processed outputs (everything but last_token).
    a.seq.tokens = vec![1, 2, 3, 4];
    let n = a.output_tokens.len();
    a.seq
        .tokens
        .extend_from_slice(&a.output_tokens[..n.saturating_sub(1)]);
    (a, rx)
}

pub(super) fn streaming_seq(
    slot: usize,
    n_out: usize,
) -> (
    ActiveSeq,
    tokio::sync::mpsc::Receiver<crate::api::StreamEvent>,
) {
    let (a, _rx) = active_seq(slot, n_out);
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let mut a = a;
    a.sink = ResponseSink::Streaming(tx);
    (a, rx)
}
