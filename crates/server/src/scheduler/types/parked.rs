// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The parked sequence forms: spilled to disk (`SwappedSeq`) and requeued for re-prefill (`PreemptedSeq`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

#![allow(dead_code)]

use super::*;

/// 2026-09-25: A sequence swapped out to the spill pool (its state written by the model's
/// `save_sequence_state`).
pub(in crate::scheduler) struct SwappedSeq {
    pub tokens: Vec<u32>,
    pub session_hash: u64,
    /// 2026-09-25: Per-request LoRA slot, restored with `tokens` so a resumed sequence
    /// keeps its adapter. (`cancel_flag` is not carried: swap-in sets it to
    /// `None`.)
    pub adapter_slot: i32,
    /// 2026-09-25: The adapter id (KV and prefix-cache identity) stamped at prefill.
    /// Stored, not recomputed: for `adapter_slot == -1` the id follows the
    /// installed active adapter, so recomputing after a rotation would give a
    /// different id than the one the sequence's blocks were written under.
    pub adapter_id: u64,
    pub seq_len: usize,
    pub num_blocks: usize,
    pub last_token: u32,
    pub output_tokens: Vec<u32>,
    pub remaining: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    pub sink: ResponseSink,
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
    pub inside_thinking: bool,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    /// 2026-09-25: Per-request override for the content-loop detector, carried across
    /// the spill.
    pub repetition_detection: Option<RepetitionDetectionParams>,
    pub spontaneous_think_budget: u32,
    pub thinking_tokens: u32,
    pub force_end_thinking: bool,
    pub think_force_closed: bool,
    pub sentence_defer_count: u32,
    pub consecutive_confident: u32,
    pub in_code_fence: bool,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub think_ended: bool,
    pub think_just_ended: bool,
    pub post_think_emitted: u32,
    pub think_skip_count: u32,
    pub require_tool_call: bool,
    /// 2026-09-25: Sticky tool-request flag, carried across the spill. A swapped-in
    /// sequence comes back with no grammar, so this is what keeps it marked
    /// as a tool request.
    pub tool_request: bool,
    /// 2026-09-25: Request declared tools, carried across the spill so a resumed
    /// multi-call turn keeps going past `</tool_call>`.
    pub tools_present: bool,
    pub suppress_tool_call: bool,
    /// 2026-09-25: The request disabled MTP; carried across the spill.
    pub disable_mtp: bool,
    /// 2026-09-25: Per-request MTP accept counters, carried across the spill.
    pub mtp_acct: crate::scheduler::mtp_accept_debug::RequestAccept,
    pub content_started: bool,
    pub content_tokens: u32,
    pub prose_tokens_since_last_tool: u32,
    pub think_watchdog_fires: u32,
    /// 2026-09-25: Watchdog rollback counter, carried across the spill.
    pub rollback_count: u32,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_opened: bool,
    pub tool_call_end_token: Option<u32>,
    pub last_token_time: Instant,
    pub request_start: Instant,
    pub decode_start: Instant,
    pub seed: Option<u64>,
    pub top_logprobs: Option<u8>,
    pub logprobs_data: Vec<crate::api::TokenLogprobs>,
    /// 2026-09-25: Number of prompt tokens served by the prefix cache (no prefill cost).
    pub cached_prompt_tokens: u32,
    pub timeout_at: Option<Instant>,
    pub swap_id: u64,
}

/// 2026-09-25: A sequence preempted out of decode when the KV pool ran dry, awaiting a
/// requeue-resume (the counterpart of [`SwappedSeq`] when there is no spill
/// pool).
///
/// Nothing is serialized: the victim's GPU resources are freed (its KV is
/// offered to the prefix cache first, as `finish_sequence` does) and the
/// whole `ActiveSeq` stays on the CPU. Resume re-prefills `tokens` to
/// rebuild KV and SSM state and puts the fresh `SequenceState` back into
/// `a`, so the client's stream continues where it paused.
pub(in crate::scheduler) struct PreemptedSeq {
    /// 2026-09-25: Retained request state. `a.seq` holds no GPU resources (freed at
    /// preemption); everything CPU-side stays live, including `cancel_flag`.
    pub a: ActiveSeq,
    /// 2026-09-25: Token history to re-prefill on resume: prompt plus every processed
    /// output token. Excludes `a.last_token`, the pending decode input, which
    /// resume leaves in place for the first decode step, so no token is
    /// re-sampled or re-emitted.
    pub tokens: Vec<u32>,
    /// 2026-10-15 (race-prefill, A192): The re-prefill in flight under
    /// `METRALE_RESUME_ONE_CHUNK_PER_TICK`; `None` until the resume starts, and always `None`
    /// with the lever off. While it is set the row still counts as preempted everywhere
    /// (admission demand, idle checks, LoRA wait, shutdown drain), and it holds `seq`.
    pub resume: Option<ResumeProgress>,
}

/// 2026-10-15 (race-prefill, A192): A preempt-resume re-prefill part-way through: the freshly
/// allocated sequence and the next chunk's start in `PreemptedSeq::tokens`.
pub(in crate::scheduler) struct ResumeProgress {
    pub seq: SequenceState,
    pub offset: usize,
}
