// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pre-sample logits pipeline: the stages, their driver, and the per-position entry point.
//!
//! Owner: scheduler.
//! Invariants:
//! - `run_pipeline_with_path` runs the stages in the order below and stops at
//!   the first stage that returns [`ProcessorOutcome::EmitToken`].
//! - `process_position_logits` does not call the grammar matcher's
//!   `accept_token` or `rollback`; `pipeline_tests` checks its source for this.
//!
//! ## Stage order
//!
//! 1. [`min_tokens_eos::MinTokensEosMask`] — masks EOS token ids while the
//!    effective output length is below the request's `min_tokens` floor.
//! 2. [`f2_confidence::F2ConfidenceEarlyStop`] — arms `force_end_thinking`
//!    when top-1 probability stays ≥ 0.95 for the configured run.
//! 3. [`mid_word::MidWordThinkEndMask`] — suppresses `</think>` when
//!    the previous token decoded to mid-word text.
//! 4. [`post_close::PostCloseThinkMask`] — while `seq.think_ended` is set,
//!    masks `</think>` and `<think>`.
//! 5. [`tool_during_think::ToolCallDuringThinkingMask`] — masks
//!    `<tool_call>` during thinking; biases it down when the tool-loop flag
//!    is set.
//! 6. [`forced_think_end::ForcedThinkEndInjector`] — once `force_end_thinking`
//!    is armed and the boundary gate allows it, masks every token but `</think>`.
//! 7. [`pin_tool_call::PinToToolCallStart`] — right after `</think>`, when
//!    `require_tool_call` is set, masks every token but `<tool_call>`.
//! 8. [`forced_token::ForcedTokenFastPath`] — when the grammar admits
//!    exactly one next token, returns it and ends the pipeline.
//! 9. [`grammar_bitmask::GrammarBitmaskApply`] — applies the grammar's
//!    next-token bitmask.
//!
//! ## Not stages
//!
//! On the decode path (`decode_logits_seq::process_seq_logits`) the
//! adaptive-sampling entropy observation reads the logits before this pipeline
//! runs, and the final `sample_with_params_history` call comes after it.

use crate::scheduler::ActiveSeq;
use crate::scheduler::sample_step::PositionKind;
use metrale_sampling::{SamplingParams, apply_penalties_and_bias};

pub mod adadec_diag;
mod b1_margin;
pub mod f2_confidence;
pub mod forced_think_end;
pub mod forced_token;
pub mod grammar_bitmask;
pub mod mid_word;
pub mod min_tokens_eos;
pub mod pin_tool_call;
pub mod post_close;
pub mod tool_during_think;

#[cfg(test)]
mod pipeline_tests;

/// 2026-09-25: What every stage receives besides the logits and the sequence:
/// special-token ids, vocab masks, levers, the model's watchdog tunables and
/// the run's I/O routers. `Clone`, not `Copy`, because the masks are `Arc`s.
#[derive(Debug, Clone)]
pub struct LogitsContext<'a> {
    /// 2026-09-25: The run's reusable host decode buffers.
    pub scratch: &'a crate::scheduler::sched_ctx::DecodeScratch,
    /// 2026-09-25: The run's telemetry router: timing marks, counters, diagnostic sinks.
    pub tel: &'a dyn crate::scheduler::io::TelemetryIo,
    pub clock: &'a dyn crate::scheduler::io::ClockIo,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
    /// 2026-09-25: This position's index in the verify window (0 on the decode path).
    /// The `min_tokens` checks count `output_tokens.len() + verify_pos`.
    pub verify_pos: usize,
    /// 2026-09-25: `VocabMasks::boundary`: `mask[id]` iff token `id` decodes to text
    /// ending in a newline or sentence-ending punctuation. Indexed by token id,
    /// so it is valid only for the tokenizer that built it.
    pub boundary_mask: Option<std::sync::Arc<[bool]>>,
    /// 2026-09-25: `VocabMasks::mid_word`: `mask[id]` iff token `id` decodes to text
    /// whose last character is alphanumeric. Same indexing as `boundary_mask`.
    pub mid_word_mask: Option<std::sync::Arc<[bool]>>,
    /// 2026-09-25: This run's sampling levers, from `SchedLevers::sampling`.
    pub sampling: SamplingLevers,
    /// 2026-09-25: This model's watchdog tunables, built at serve load by
    /// `WatchdogParams::from_behavior` from MODEL.toml `[behavior]` and two CLI
    /// overrides. `F2ConfidenceEarlyStop` reads them.
    pub watchdog: crate::scheduler::helpers::WatchdogParams,
}

/// 2026-09-25: The subset of `scheduler::levers::SchedLevers` carried on
/// `LogitsContext` (built by `SchedLevers::sampling`). The pipeline stages and
/// the verify pick helpers read it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplingLevers {
    /// 2026-09-25: `METRALE_FORCE_TEMP_ZERO`: `process_position_logits` returns the
    /// raw-logit argmax, with no pipeline, penalties or bias.
    pub force_temp_zero: bool,
    /// 2026-09-25: Greedy GPU-argmax fast path in `sample_token_with_grammar`, and in
    /// the verify pick for sequences with a grammar
    /// (`METRALE_DISABLE_FAST_GREEDY=1` turns it off).
    pub fast_greedy_grammar: bool,
    /// 2026-09-25: At temperature > 0, `verify_pick_with_pipeline` samples from the
    /// processed logits rather than taking their argmax
    /// (`METRALE_NO_MTP_VERIFY_SAMPLE=1` turns it off).
    pub mtp_verify_sample: bool,
    /// 2026-09-25: `verify_pipeline_helper::fast_masked::try_chat_fast_path` may run
    /// (`METRALE_DISABLE_FAST_MASKED=1` turns it off).
    pub fast_masked: bool,
    /// 2026-09-25: Greedy GPU-argmax verify fast path for sequences without a grammar
    /// (`METRALE_NO_FAST_GREEDY_CHAT=1` turns it off).
    pub fast_greedy_chat: bool,
    /// 2026-09-25: `METRALE_ADADEC_DIAGNOSTIC` is set. `try_chat_fast_path` then
    /// declines, so the diagnostic sees the full pipeline.
    pub adadec_diagnostic: bool,
    /// 2026-09-25: DFlash masked verify (`METRALE_DFLASH_MASKED_VERIFY`).
    /// `try_chat_fast_path` runs only when it is set.
    pub dflash_masked_verify: bool,
    /// 2026-09-25: `METRALE_DISABLE_WATCHDOGS`. When set, `F2ConfidenceEarlyStop` and
    /// `MidWordThinkEndMask` do nothing.
    pub disable_watchdogs: bool,
    /// 2026-09-25: `ForcedTokenFastPath` may fire. On unless
    /// `METRALE_DISABLE_FORCED_TOKEN` is `1` or `true`.
    pub forced_token_fastpath: bool,
    /// 2026-09-25: `sample_step::effective_min_p` passes the sequence's resolved `min_p`
    /// when set and `0.0` when not. Its callers are `sample_token`,
    /// `sample_token_with_grammar` and the temperature > 0 branch of
    /// `verify_pick_with_pipeline`. On unless `METRALE_NO_MTP_MINP=1`.
    pub mtp_minp: bool,
    /// 2026-09-29: `SchedLevers::think_ended_gpu_argmax`
    /// (`METRALE_NO_THINKENDED_GPU_ARGMAX=1` turns it off), read by the
    /// speculative paths to apply `logit_bias` exactly where decode does
    /// (A144, `sample_step::speculative_base_logit_bias`).
    pub think_ended_gpu_argmax: bool,
}

/// 2026-09-25: What a stage tells the driver: keep going, or emit this token and stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorOutcome {
    /// 2026-09-25: Run the next stage. The stage may have changed the logits in place.
    Continue,
    /// 2026-09-25: Stop: the later stages, the penalties and the sample are skipped,
    /// and the caller emits this token.
    EmitToken(u32),
}

/// 2026-09-25: One stage of the pre-sample pipeline.
pub trait LogitsProcessor: Send + Sync {
    /// 2026-09-25: Apply this stage to `logits`. A stage may also change `seq`
    /// (`F2ConfidenceEarlyStop` sets `seq.force_end_thinking`; the grammar
    /// stages fill `seq.grammar_state`'s bitmask).
    fn apply(
        &self,
        logits: &mut [f32],
        seq: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome;

    /// 2026-09-25: The stage's name. `pipeline_tests` pins the values.
    fn name(&self) -> &'static str;

    /// 2026-09-25: `true` if this stage never changes which token wins under argmax.
    /// Nothing outside the tests reads it.
    fn is_argmax_invariant(&self) -> bool {
        false
    }
}

/// 2026-09-25: [`run_pipeline_with_path`] with the `"verify"` label. Returns
/// `Some(token)` when a stage returned [`ProcessorOutcome::EmitToken`], else
/// `None`. Only `pipeline_tests` calls it; the decode and verify paths go
/// through [`process_position_logits`].
pub fn run_pipeline(logits: &mut [f32], seq: &mut ActiveSeq, ctx: &LogitsContext) -> Option<u32> {
    run_pipeline_with_path(logits, seq, ctx, "verify")
}

/// 2026-09-25: The pipeline driver and the single definition of the stage order.
/// `process_position_logits` calls it with `PositionKind::adadec_label()`:
/// `"decode"` for the final decode position, `"verify"` for a verify position.
/// `path` only tags the `METRALE_ADADEC_DIAGNOSTIC` record; it changes no logit.
pub fn run_pipeline_with_path(
    logits: &mut [f32],
    seq: &mut ActiveSeq,
    ctx: &LogitsContext,
    path: &'static str,
) -> Option<u32> {
    let stages: [&dyn LogitsProcessor; 9] = [
        &min_tokens_eos::MinTokensEosMask,
        &f2_confidence::F2ConfidenceEarlyStop,
        &mid_word::MidWordThinkEndMask,
        &post_close::PostCloseThinkMask,
        &tool_during_think::ToolCallDuringThinkingMask,
        &forced_think_end::ForcedThinkEndInjector,
        &pin_tool_call::PinToToolCallStart,
        &forced_token::ForcedTokenFastPath,
        &grammar_bitmask::GrammarBitmaskApply,
    ];
    for stage in stages.iter() {
        match stage.apply(logits, seq, ctx) {
            ProcessorOutcome::Continue => {}
            ProcessorOutcome::EmitToken(tok) => return Some(tok),
        }
    }
    // 2026-09-25: AdaDec diagnostic: reads the logits after the last stage and
    // never writes them. A no-op when the run has no AdaDec sink. Called here
    // rather than as a stage so it carries the caller's `path` label.
    adadec_diag::log_step(ctx.tel.dumps().adadec.as_ref(), logits, seq, path);
    None
}

/// 2026-09-25: Per-position logit processing, called by the final decode position
/// (`decode_logits_seq::process_seq_logits`) and by each verify position
/// (`verify_pipeline_helper::verify_pick_with_pipeline`).
///
/// Steps, in order:
///  1. **`METRALE_FORCE_TEMP_ZERO` bypass** (both kinds): return the raw-logit
///     argmax, with no pipeline, penalties or bias.
///  2. **[`run_pipeline_with_path`]** (the nine stages and the AdaDec
///     diagnostic). A `Some(tok)` is the forced-token fast path; it is
///     returned at once.
///  3. **B1 margin observer**, `FinalDecode` only: records the post-mask
///     top-1/top-2 gap and never writes the logits.
///  4. **`apply_penalties_and_bias`** with `penalties` (repetition, presence,
///     frequency, LZ and DRY penalties, and `logit_bias`, which can include the
///     minimum-reasoning `</think>` floor) over the sequence's history, scoped
///     by `sample_step::penalty_history_scope`.
///
/// Returns `Some(token)` only from steps 1 and 2; the caller emits it without
/// sampling. On `None` the caller samples or takes the argmax of the masked
/// and penalised `logits`.
///
/// This function never calls the grammar matcher's accept or rollback. The
/// callers do: `verify_pipeline_helper` on the verify path and
/// `decode_logits_step` on the decode path. The stages only fill the grammar's
/// bitmask for the current position.
pub fn process_position_logits(
    logits: &mut [f32],
    seq: &mut ActiveSeq,
    ctx: &LogitsContext,
    penalties: &SamplingParams,
    kind: PositionKind,
) -> Option<u32> {
    // 2026-09-25: 1. METRALE_FORCE_TEMP_ZERO: raw-logit argmax on both kinds.
    if ctx.sampling.force_temp_zero {
        let mut best_idx: u32 = 0;
        let mut best_val: f32 = f32::NEG_INFINITY;
        for (j, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = j as u32;
            }
        }
        return Some(best_idx);
    }

    // 2026-09-25: 2. The stages, and the AdaDec diagnostic under this position's label.
    if let Some(forced) = run_pipeline_with_path(logits, seq, ctx, kind.adadec_label()) {
        return Some(forced);
    }

    // 2026-09-25: 3. B1 margin observer, final decode position only. Read-only.
    if kind == PositionKind::FinalDecode {
        b1_margin::observe(logits, seq, ctx.tel.stats());
    }

    // 2026-09-25: 4. Penalties and bias on the masked logits. The history is cut
    //    after the last `tool_call_end_token` (`penalty_history_scope`), so
    //    penalties from finished tool calls do not carry into the next one.
    let t_pen = ctx.clock.now();
    apply_penalties_and_bias(
        logits,
        penalties,
        crate::scheduler::sample_step::penalty_history_scope(
            &seq.output_tokens,
            ctx.tool_call_end_token,
        ),
    );
    if kind == PositionKind::Verify {
        ctx.tel
            .mark(crate::scheduler::mtp_timing::Phase::Penalties, t_pen);
    }

    None
}
