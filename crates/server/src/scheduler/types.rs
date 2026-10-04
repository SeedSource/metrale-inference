// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Scheduler-internal types: the pending queue, the response sink, the in-progress prefill, and (in `active` and `parked`) the sequence states.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

#![allow(dead_code)]

use std::time::Instant;

use anyhow::Result;
use metrale_model_engine::traits::SequenceState;

use crate::api::inference_types::RepetitionDetectionParams;
use crate::api::{InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::GrammarState;

/// 2026-09-25: The scheduler's own queue of what the inbox has handed over and the
/// loop has not yet admitted.
pub(super) struct PendingQueue {
    pub requests: Vec<InferenceRequest>,
    pub closed: bool,
    /// 2026-09-25: Pending LoRA adapter-rotation control requests, applied by the scheduler
    /// at a quiescent point (see [`super::LoraRotation`]). Kept out of
    /// `requests` so the sequence machinery never sees a control message.
    pub rotations: Vec<super::LoraRotation>,
}

impl PendingQueue {
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            closed: false,
            rotations: Vec::new(),
        }
    }

    /// 2026-09-25: Fold one `recv` batch in, arrivals after what is already queued.
    pub fn absorb(&mut self, arrived: super::io::Arrivals) {
        self.requests.extend(arrived.requests);
        self.rotations.extend(arrived.rotations);
        self.closed |= arrived.closed;
    }

    /// 2026-09-25: Nothing to admit, nothing to apply, and more may still come.
    pub fn is_idle(&self) -> bool {
        self.requests.is_empty() && self.rotations.is_empty() && !self.closed
    }
}

/// 2026-09-25: Per-request slice of a co-dispatched vision encode. With vision
/// co-dispatch on, when two or more image requests that fit one prefill
/// chunk are admitted in the same tick, `prepare_vision_embed_batched`
/// encodes all their images at once and each request gets the offsets it
/// owns in the shared output. `Default` (all zero) means not co-dispatched:
/// the request encodes its own images.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct VisionSlice {
    /// 2026-09-25: First `buf_out` row (post-merge patch) this request owns.
    pub patch_row_offset: usize,
    /// 2026-09-25: First `vision_image_grids` index this request owns.
    pub grid_index_offset: usize,
    /// 2026-09-25: Number of images this request contributed to the batch.
    pub num_images: usize,
    /// 2026-09-25: Total post-merge rows this request owns (Σ merged_p over its images).
    pub patch_row_count: usize,
}

/// 2026-09-25: How to deliver results for an active sequence.
pub(crate) enum ResponseSink {
    Blocking(Option<tokio::sync::oneshot::Sender<Result<InferenceResponse>>>),
    Streaming(tokio::sync::mpsc::Sender<StreamEvent>),
}

impl ResponseSink {
    /// 2026-09-25: Whether mid-stream events reach this client (a blocking client gets
    /// everything in the finish frame instead).
    pub fn is_streaming(&self) -> bool {
        matches!(self, Self::Streaming(_))
    }
}

/// 2026-09-25: An in-progress chunked prefill (prompt being processed in chunks).
pub(super) struct PrefillInProgress {
    /// 2026-09-25: The request's own `Arc`, shared rather than copied.
    pub prompt_tokens: std::sync::Arc<Vec<u32>>,
    pub session_hash: u64,
    pub seq: SequenceState,
    pub chunk_offset: usize,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    pub sink: ResponseSink,
    /// 2026-09-25: Cooperative cancellation flag; see `ActiveSeq::cancel_flag`.
    /// Moved to the `ActiveSeq` on promotion.
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub request_start: Instant,
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
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    /// 2026-09-25: Per-request override for the content-loop detector, moved to the
    /// `ActiveSeq` on promotion. `None` falls back to the server's settings
    /// (`WatchdogParams::content_loop_params`).
    pub repetition_detection: Option<RepetitionDetectionParams>,
    /// 2026-09-25: Per-server spontaneous-thinking budget: `--max-thinking-budget`, else
    /// MODEL.toml `[behavior].max_thinking_budget`. When the model opens
    /// `<think>` on its own, the sequence's thinking budget is set from it.
    pub spontaneous_think_budget: u32,
    pub require_tool_call: bool,
    /// 2026-09-25: The request declared tools; moved to the `ActiveSeq` on promotion.
    pub tools_present: bool,
    pub suppress_tool_call: bool,
    /// 2026-09-25: The request disabled MTP; moved to the `ActiveSeq` on promotion.
    pub disable_mtp: bool,
    pub grammar_state: Option<GrammarState>,
    pub seed: Option<u64>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<Instant>,
    /// 2026-10-03: The "prefill chunk capped" line was logged for this prefill
    /// (`prefill_chunk_cap::log_capped`).
    pub chunk_cap_logged: bool,
}

mod active;
mod parked;
pub(super) use active::*;
pub(super) use parked::*;

#[cfg(test)]
mod budget_tests {
    use super::consume_budget;

    #[test]
    fn consume_budget_decrements_to_zero() {
        let mut r = 2usize;
        assert!(consume_budget(&mut r));
        assert_eq!(r, 1);
        assert!(consume_budget(&mut r));
        assert_eq!(r, 0);
    }

    #[test]
    fn consume_budget_at_zero_signals_finish_and_never_wraps() {
        // 2026-09-25: a bare `remaining -= 1` at 0 would wrap to
        // usize::MAX in release builds, unbounding generation, and
        // panic in debug builds.
        let mut r = 0usize;
        assert!(!consume_budget(&mut r));
        assert_eq!(r, 0);
    }
}
