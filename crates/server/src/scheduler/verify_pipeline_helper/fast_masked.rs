// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the masked chat fast path of `verify_pick_all_with_pipeline`:
//! returns the GPU argmax picks without copying the logits rows when the
//! checks below find no stage that would change them.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types. `try_chat_fast_path` does not mutate
//! the sequence.

use crate::scheduler::ActiveSeq;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::mtp_timing::Phase;
use metrale_model_engine::traits::Model;

/// 2026-09-25: `Some(argmax_ids)` when every check passes, `None` when the
/// caller must try its other paths.
///
/// Requires: the `dflash_masked_verify` and `fast_masked` levers on, no
/// grammar, outside thinking (2026-09-29, A146), AdaDec diagnostics off, `F2ConfidenceEarlyStop`, the forced
/// `</think>` and the tool-call pin all inactive, penalties not `Blocked`,
/// and at every position an argmax that is not the think-end, think-start
/// or tool-call-start token and, for `ReduceOnly` penalties, is
/// penalty-immune (`fast_greedy`).
///
/// Not checked: `MinTokensEosMask`, so an EOS argmax below `min_tokens` is
/// returned unmasked; and the temperature, so above 0 this returns the
/// argmax where the host path, with `mtp_verify_sample` on, would sample.
///
/// `row_base` is as for `verify_pick_all_with_pipeline`.
pub(super) fn try_chat_fast_path(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &ActiveSeq,
    ctx: &LogitsContext,
    row_base: usize,
) -> Option<Vec<u32>> {
    // 2026-09-25: the gate is the `dflash_masked_verify` lever itself (on
    // unless `METRALE_DFLASH_MASKED_VERIFY=0`), not whether the serve has a
    // DFlash drafter, so MTP verify steps reach this path too. Where it
    // fires, the GPU argmax replaces the host argmax, which can break
    // near-ties differently.
    if !ctx.sampling.dflash_masked_verify {
        return None;
    }
    let fast_masked_enabled = ctx.sampling.fast_masked;
    let adadec_recording = ctx.sampling.adadec_diagnostic;
    // 2026-09-29: A146, spec-in-think parity: the checks below are judged on
    // the step-start state, but inside `<think>` the window itself can cross
    // the F2 400-token gate, the thinking budget or a thinking-loop stride
    // (arming the forced `</think>` mid-window), and the stateful F2 and
    // defer-tick stages must run per position. Decode never takes a device
    // argmax for a thinking row (`decode_row_uses_gpu_argmax`), so neither
    // does this shortcut.
    if !fast_masked_enabled || a.grammar_state.is_some() || adadec_recording || a.inside_thinking {
        return None;
    }
    use crate::scheduler::confidence::{
        MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    };
    // 2026-09-25: each flag is true whenever `F2ConfidenceEarlyStop`,
    // `ForcedThinkEndInjector` (inject or defer) or `PinToToolCallStart`
    // would act (see their `apply` bodies), and sometimes when it would not.
    let f2_active = !ctx.sampling.disable_watchdogs
        && a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= 400
        && ctx.watchdog.confidence_early_stop;
    let defer_hard_override = match a.thinking_budget {
        Some(b) => a.thinking_tokens >= b.saturating_mul(THINK_DEFER_BUDGET_FACTOR),
        None => a.thinking_tokens >= THINK_DEFER_ABS_CEILING,
    } || a.sentence_defer_count >= MAX_SENTENCE_DEFER_TOKENS;
    let think_end_inject_armed = a.inside_thinking && (a.force_end_thinking || defer_hard_override);
    let pin_tool_armed =
        a.think_just_ended && a.require_tool_call && !a.tool_call_opened && !a.inside_thinking;
    // 2026-09-25: the same `penalty_params_for` call the host path makes for
    // every verify position.
    let penalty_gate = crate::scheduler::fast_greedy::classify_penalties(
        &crate::scheduler::sample_step::penalty_params_for(
            a,
            crate::scheduler::sample_step::PositionKind::Verify,
            0.0,
            None,
            Vec::new(),
            ctx.watchdog.min_reasoning_floor,
        ),
    );
    if f2_active
        || think_end_inject_armed
        || pin_tool_armed
        || penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::Blocked
    {
        return None;
    }
    let t_fast = ctx.clock.now();
    let scoped_history: Vec<u32> =
        if penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly {
            crate::scheduler::sample_step::penalty_history_scope(
                &a.output_tokens,
                ctx.tool_call_end_token,
            )
            .to_vec()
        } else {
            Vec::new()
        };
    let vocab = model.vocab_size();
    let logits_base = model.logits_buffer_ptr();
    let mut all_clear = true;
    for (i, &tok) in argmax_ids.iter().enumerate() {
        // 2026-09-25: the ids `MidWordThinkEndMask`, `PostCloseThinkMask`
        // and `ToolCallDuringThinkingMask` can mask or lower.
        if Some(tok) == ctx.think_end_token
            || Some(tok) == a.think_start_token
            || Some(tok) == ctx.tool_call_start_token
        {
            all_clear = false;
            break;
        }
        if penalty_gate == crate::scheduler::fast_greedy::PenaltyGate::ReduceOnly
            && !crate::scheduler::fast_greedy::argmax_immune(tok, &scoped_history, || {
                crate::scheduler::fast_greedy::logit_is_positive(
                    model,
                    logits_base,
                    row_base + i,
                    vocab,
                    tok,
                )
            })
        {
            all_clear = false;
            break;
        }
    }
    ctx.tel.mark(Phase::FastGreedy, t_fast);
    if all_clear {
        if ctx.tel.stats().once("log:verify_chat_fast_path") {
            tracing::info!(
                "verify chat fast path ACTIVE: masked-greedy == raw argmax, no D2H \
                 (kill-switch: METRALE_DISABLE_FAST_MASKED=1)"
            );
        }
        return Some(argmax_ids.to_vec());
    }
    None
}
