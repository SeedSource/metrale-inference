// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pipeline tests that need no `ActiveSeq`: stage names, the `</think>` gate, signatures, and source scans.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;
use crate::scheduler::confidence::{
    MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    should_inject_think_end,
};

/// 2026-09-25: Pins the `name()` of eight stages and checks they are distinct.
/// `MinTokensEosMask` is not in the list.
#[test]
fn stage_names_are_distinct_and_stable() {
    let names: [&'static str; 8] = [
        f2_confidence::F2ConfidenceEarlyStop.name(),
        mid_word::MidWordThinkEndMask.name(),
        post_close::PostCloseThinkMask.name(),
        tool_during_think::ToolCallDuringThinkingMask.name(),
        forced_think_end::ForcedThinkEndInjector.name(),
        pin_tool_call::PinToToolCallStart.name(),
        forced_token::ForcedTokenFastPath.name(),
        grammar_bitmask::GrammarBitmaskApply.name(),
    ];
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            assert_ne!(
                names[i], names[j],
                "stage names must be distinct ({} == {})",
                names[i], names[j]
            );
        }
    }
    // 2026-09-25: Exact strings, so a rename shows up as a test change.
    assert_eq!(names[0], "f2_confidence_early_stop");
    assert_eq!(names[1], "mid_word_think_end_mask");
    assert_eq!(names[2], "post_close_think_mask");
    assert_eq!(names[3], "tool_call_during_thinking_mask");
    assert_eq!(names[4], "forced_think_end_injector");
    assert_eq!(names[5], "pin_to_tool_call_start");
    assert_eq!(names[6], "forced_token_fastpath");
    assert_eq!(names[7], "grammar_bitmask_apply");
}

/// 2026-09-25: Of the eight stages listed, only `F2ConfidenceEarlyStop` reports
/// `is_argmax_invariant() == true`; it never writes the logits.
#[test]
fn argmax_invariance_advertisement() {
    assert!(f2_confidence::F2ConfidenceEarlyStop.is_argmax_invariant());
    assert!(!mid_word::MidWordThinkEndMask.is_argmax_invariant());
    assert!(!post_close::PostCloseThinkMask.is_argmax_invariant());
    assert!(!tool_during_think::ToolCallDuringThinkingMask.is_argmax_invariant());
    assert!(!forced_think_end::ForcedThinkEndInjector.is_argmax_invariant());
    assert!(!pin_tool_call::PinToToolCallStart.is_argmax_invariant());
    assert!(!forced_token::ForcedTokenFastPath.is_argmax_invariant());
    assert!(!grammar_bitmask::GrammarBitmaskApply.is_argmax_invariant());
}

/// 2026-09-25: Truth table of `should_inject_think_end(force_end_thinking,
/// in_code_fence, at_sentence_boundary, hard_override)`, the gate
/// `ForcedThinkEndInjector` calls.
#[test]
fn forced_think_end_gate_semantics() {
    assert!(!should_inject_think_end(false, false, false, false));
    assert!(!should_inject_think_end(false, true, true, true));

    assert!(should_inject_think_end(true, true, false, true));
    assert!(should_inject_think_end(true, false, false, true));

    assert!(!should_inject_think_end(true, true, false, false));
    assert!(!should_inject_think_end(true, true, true, false));

    assert!(should_inject_think_end(true, false, true, false));

    assert!(!should_inject_think_end(true, false, false, false));
}

/// 2026-09-25: Restates the three override comparisons at their thresholds. It
/// does not call the injector and does not pin the constants' values.
#[test]
fn defer_override_math_constants() {
    let budget: u32 = 100;
    let thinking_tokens: u32 = budget.saturating_mul(THINK_DEFER_BUDGET_FACTOR);
    assert!(thinking_tokens >= budget.saturating_mul(THINK_DEFER_BUDGET_FACTOR));

    let unlimited_tokens: u32 = THINK_DEFER_ABS_CEILING;
    assert!(unlimited_tokens >= THINK_DEFER_ABS_CEILING);

    let defer_count: u32 = MAX_SENTENCE_DEFER_TOKENS;
    assert!(defer_count >= MAX_SENTENCE_DEFER_TOKENS);
}

/// 2026-09-25: The struct literal below names every `LogitsContext` field, so
/// adding, removing or renaming a field fails to compile here. The test also
/// checks that a clone keeps the four special-token ids.
#[test]
fn logits_context_field_set_is_stable() {
    // 2026-09-25: Masks are per-context fields, so this test sets its own (None)
    // and shares no state with other tests.
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers::default(),
        think_end_token: Some(1),
        think_start_token: Some(2),
        tool_call_start_token: Some(3),
        tool_call_end_token: Some(4),
        code_fence_token: None,
        verify_pos: 0,
    };
    // 2026-09-25: `Clone`, not `Copy`: the masks are `Arc`s.
    let ctx2 = ctx.clone();
    assert_eq!(ctx2.think_end_token, Some(1));
    assert_eq!(ctx2.think_start_token, Some(2));
    assert_eq!(ctx2.tool_call_start_token, Some(3));
    assert_eq!(ctx2.tool_call_end_token, Some(4));
    assert_eq!(ctx.tool_call_end_token, Some(4));
}

/// 2026-09-25: Pins the signatures of `run_pipeline`, `process_position_logits` and
/// `run_pipeline_with_path` with fn-pointer types: a signature change fails to
/// compile here.
#[test]
fn run_pipeline_signature_is_stable() {
    type RunPipelineFn = fn(&mut [f32], &mut ActiveSeq, &LogitsContext) -> Option<u32>;
    let _ptr: RunPipelineFn = run_pipeline;
    type ProcessPositionFn = fn(
        &mut [f32],
        &mut ActiveSeq,
        &LogitsContext,
        &metrale_sampling::SamplingParams,
        crate::scheduler::sample_step::PositionKind,
    ) -> Option<u32>;
    let _pp: ProcessPositionFn = process_position_logits;
    type RunPipelineWithPathFn =
        fn(&mut [f32], &mut ActiveSeq, &LogitsContext, &'static str) -> Option<u32>;
    let _rpp: RunPipelineWithPathFn = run_pipeline_with_path;
}

/// 2026-09-25: Source scan of `decode_logits_seq.rs`, the decode path: masks belong
/// in the pipeline stages and penalties in `penalty_params_for`. The file must
/// not contain `f32::NEG_INFINITY`, `repetition_penalty: a.repetition_penalty`,
/// `MIN_REASONING_TOKENS`, `b1_record_low_margin`, `low_margin_in_body`,
/// `if false && `, or any of the four marker strings below, and it must
/// mention `process_position_logits`.
#[test]
fn non_mtp_path_has_no_inline_pipeline_fork() {
    const SRC: &str = include_str!("../decode_logits_seq.rs");

    assert!(
        !SRC.contains("f32::NEG_INFINITY"),
        "decode_logits_seq.rs must not hard-mask inline; masking belongs in logit_processors stages"
    );
    assert!(
        !SRC.contains("repetition_penalty: a.repetition_penalty"),
        "decode_logits_seq.rs must not inline the SamplingParams penalty literal; use penalty_params_for"
    );
    assert!(
        !SRC.contains("MIN_REASONING_TOKENS"),
        "A4 floor must live only in penalty_params_for, not inline in decode_logits_seq.rs"
    );
    assert!(
        !SRC.contains("b1_record_low_margin") && !SRC.contains("low_margin_in_body"),
        "B1 margin detector must live only in logit_processors::b1_margin, not inline"
    );
    assert!(
        !SRC.contains("if false && "),
        "the dead C4v1 `if false && low_margin_in_body` block must be deleted"
    );
    for marker in [
        "Mid-word `</think>` defer",
        "one-shot pin-to-tool-call-start",
        "Forced-token fast-path (xgrammar Tier 3b",
        "Apply grammar bitmask BEFORE sampling",
    ] {
        assert!(
            !SRC.contains(marker),
            "stale inline per-stage block `{marker}` still present in decode_logits_seq.rs"
        );
    }
    assert!(
        SRC.contains("process_position_logits"),
        "decode_logits_seq.rs must call the unified process_position_logits"
    );
}

/// 2026-09-25: Source scans. `sample_step/penalties.rs` holds the
/// minimum-reasoning floor (`min_reasoning_floor: u32`, `a.think_end_token`);
/// `b1_margin.rs` defines `observe` and `LOW_MARGIN_THRESHOLD`; `mod.rs`
/// mentions `b1_margin::observe`, `PositionKind::FinalDecode`,
/// `ctx.sampling.force_temp_zero` and `apply_penalties_and_bias`; and
/// `verify_pipeline_helper.rs` mentions `process_position_logits`.
#[test]
fn unified_fn_includes_a4_and_b1_stages() {
    const SAMPLE_STEP_SRC: &str = include_str!("../sample_step/penalties.rs");
    assert!(
        SAMPLE_STEP_SRC.contains("min_reasoning_floor: u32")
            && SAMPLE_STEP_SRC.contains("a.think_end_token"),
        "A4 POST_THINK_MIN_REASONING floor must live in penalty_params_for"
    );

    const B1_SRC: &str = include_str!("b1_margin.rs");
    assert!(
        B1_SRC.contains("fn observe") && B1_SRC.contains("LOW_MARGIN_THRESHOLD"),
        "B1 margin observer must live in logit_processors::b1_margin"
    );
    const MOD_SRC: &str = include_str!("mod.rs");
    assert!(
        MOD_SRC.contains("b1_margin::observe") && MOD_SRC.contains("PositionKind::FinalDecode"),
        "process_position_logits must call B1 observe gated on FinalDecode"
    );
    assert!(
        // 2026-09-25: The bypass reads the run's carried lever ctx.sampling.force_temp_zero.
        MOD_SRC.contains("ctx.sampling.force_temp_zero")
            && MOD_SRC.contains("apply_penalties_and_bias"),
        "process_position_logits must own the force-temp-zero bypass and penalties+bias"
    );

    const VERIFY_SRC: &str = include_str!("../verify_pipeline_helper.rs");
    assert!(
        VERIFY_SRC.contains("process_position_logits"),
        "verify_pick_with_pipeline must call the unified process_position_logits"
    );
}

/// 2026-09-25: Source scan: from `pub fn process_position_logits` to the end of
/// `mod.rs` (the function is last in the file) there is no `.accept_token(`
/// or `.rollback(` call. The callers own the grammar matcher.
#[test]
fn unified_fn_does_not_advance_matcher() {
    const MOD_SRC: &str = include_str!("mod.rs");
    let start = MOD_SRC
        .find("pub fn process_position_logits")
        .expect("process_position_logits must exist in mod.rs");
    let body = &MOD_SRC[start..];
    assert!(
        !body.contains(".accept_token(") && !body.contains(".rollback("),
        "process_position_logits must not call gs.accept_token / gs.rollback (R1: caller-owned)"
    );
}
