// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Start this tick's admitted requests, continue in-progress
//! prefills, and the idle-tick requeue resume.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

impl SchedulerCore {
    pub(super) fn start_prefills(&mut self, new_reqs: Vec<InferenceRequest>) -> LaneVerdict {
        let Self {
            ctx: sched,
            eos_tokens,
            grammar_engine,
            active,
            prefilling,
            ..
        } = self;
        let chunked = self.chunked;
        let always_mixed = self.always_mixed;
        let max_prefill_tokens = self.max_prefill_tokens;
        let max_batch_tokens = self.max_batch_tokens;
        let prefill_stream = self.prefill_stream;
        let prefill_event = self.prefill_event;
        let spontaneous_think_budget = self.spontaneous_think_budget;
        let think_end_token = self.think_end_token;
        let think_start_token = self.think_start_token;
        let tool_call_start_token = self.tool_call_start_token;
        let tool_call_end_token = self.tool_call_end_token;
        let t_loop = sched.io.clock.now();
        start_new_requests(
            sched.io.dev.model(),
            sched,
            new_reqs,
            chunked,
            always_mixed,
            max_prefill_tokens,
            max_batch_tokens,
            eos_tokens,
            prefill_stream,
            prefill_event,
            grammar_engine,
            spontaneous_think_budget,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            active,
            prefilling,
        );
        sched.io.tel.mark(mtp_timing::Phase::LoopAdmit, t_loop);

        LaneVerdict::Proceed
    }

    pub(super) fn continue_prefills(&mut self) -> LaneVerdict {
        let Self {
            ctx: sched,
            policy,
            active,
            prefilling,
            preempted,
            did_mixed_step,
            ..
        } = self;
        let max_prefill_tokens = self.max_prefill_tokens;
        let max_batch_tokens = self.max_batch_tokens;
        let always_mixed = self.always_mixed;
        let prefill_stream = self.prefill_stream;
        let prefill_event = self.prefill_event;
        let use_mtp = self.use_mtp;
        let use_self_speculative = self.use_self_speculative;
        let use_ngram_speculative = self.use_ngram_speculative;
        let think_end_token = self.think_end_token;
        let think_start_token = self.think_start_token;
        let code_fence_token = self.code_fence_token;
        let tool_call_start_token = self.tool_call_start_token;
        let tool_call_end_token = self.tool_call_end_token;
        let adaptive_sampling = self.adaptive_sampling;
        let max_batch_size = self.max_batch_size;
        let block_size = self.block_size;
        let t_loop = sched.io.clock.now();
        // 2026-10-03: `METRALE_PREFILL_CHUNK_WHILE_DECODING`: no second prefill chunk in a tick
        // whose StartPrefills ran one while a sequence decodes (`prefill_chunk_cap`).
        let skip = crate::scheduler::prefill_chunk_cap::skip_continue(
            sched.levers.prefill_chunk_while_decoding.is_some(),
            crate::scheduler::prefill_chunk_cap::decoders(active),
            self.started_prefill,
        );
        *did_mixed_step = !skip
            && continue_in_progress_prefills(
            sched.io.dev.model(),
            &**policy,
            active,
            prefilling,
            max_prefill_tokens,
            max_batch_tokens,
            always_mixed,
            prefill_stream,
            prefill_event,
            use_mtp,
            use_self_speculative,
            use_ngram_speculative,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            adaptive_sampling,
            sched,
        );
        sched.io.tel.mark(mtp_timing::Phase::LoopPrefill, t_loop);

        if active.is_empty() {
            // 2026-09-25: `SkipRest` below also skips the end-of-tick resume, so give
            // requeued sequences their resume here first.
            if !preempted.is_empty() {
                preempt::resume_preempted_seqs(
                    sched.io.dev.model(),
                    &sched.io,
                    active,
                    preempted,
                    max_batch_size,
                    block_size,
                );
            }
            if active.is_empty() {
                return LaneVerdict::SkipRest;
            }
        }

        LaneVerdict::Proceed
    }
}
