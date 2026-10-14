// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `F2ConfidenceEarlyStop`: arms the forced `</think>` after a run of confident tokens.
//!
//! The stage acts only when watchdogs are not disabled, the sequence is inside
//! thinking with at least 400 thinking tokens, `force_end_thinking` is not yet
//! set, and the model's `confidence_early_stop` is on. It then counts
//! consecutive tokens whose top-1 softmax probability is >= 0.95
//! ([`crate::scheduler::confidence::confidence_run_step`]). When the count
//! reaches `ctx.watchdog.confidence_run_length` it sets
//! `seq.force_end_thinking`; `forced_think_end` decides when `</think>` is
//! injected.
//!
//! 2026-10-14: It also needs a thinking budget. Thinking with none
//! (`[behavior].effort_prompt_only`, GLM-5.3) ends only when the model closes
//! it, as on the official model, so the stage does not run there.
//!
//! Owner: scheduler.
//! Invariants:
//! - This stage never writes `logits`.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use crate::scheduler::confidence::confidence_run_step;

pub struct F2ConfidenceEarlyStop;

impl LogitsProcessor for F2ConfidenceEarlyStop {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !ctx.sampling.disable_watchdogs
            && a.inside_thinking
            && !a.force_end_thinking
            && a.thinking_tokens >= 400
            && a.thinking_budget.is_some()
            && ctx.watchdog.confidence_early_stop
        {
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
            let confident = sum_exp > 0.0 && 1.0 / sum_exp >= 0.95;
            let (run, force_end) = confidence_run_step(
                confident,
                a.consecutive_confident,
                ctx.watchdog.confidence_run_length,
            );
            a.consecutive_confident = run;
            if force_end {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Confidence early stop armed: top-1 prob >= 0.95 for {} tokens (after {} thinking tokens){}",
                    ctx.watchdog.confidence_run_length,
                    a.thinking_tokens,
                    if a.in_code_fence {
                        " — deferred until ``` fence closes"
                    } else {
                        " — deferring until next sentence boundary"
                    }
                );
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "f2_confidence_early_stop"
    }

    fn is_argmax_invariant(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-14: A run of confident thinking tokens arms the forced `</think>`
    /// only when the sequence has a thinking budget.
    #[test]
    fn confidence_stop_needs_a_thinking_budget() {
        let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
        let io = crate::scheduler::io::SchedIo::for_test();
        let ctx = LogitsContext {
            scratch: &scratch,
            tel: &*io.tel,
            clock: &*io.clock,
            watchdog: crate::scheduler::helpers::WatchdogParams::default(),
            boundary_mask: None,
            mid_word_mask: None,
            sampling: super::super::SamplingLevers::default(),
            think_end_token: Some(1),
            think_start_token: Some(2),
            tool_call_start_token: Some(3),
            tool_call_end_token: Some(4),
            code_fence_token: None,
            verify_pos: 0,
        };
        assert!(ctx.watchdog.confidence_early_stop);
        for (budget, want_armed) in [(Some(1_000_000), true), (None, false)] {
            let (mut a, _rx) = crate::scheduler::test_support::test_seq(Vec::new(), 1000, None, 0);
            a.inside_thinking = true;
            a.enable_thinking = true;
            a.thinking_tokens = 400;
            a.thinking_budget = budget;
            for _ in 0..=ctx.watchdog.confidence_run_length {
                let mut logits = vec![-100.0f32; 8];
                logits[5] = 10.0;
                F2ConfidenceEarlyStop.apply(&mut logits, &mut a, &ctx);
            }
            assert_eq!(a.force_end_thinking, want_armed, "budget {budget:?}");
        }
    }
}
