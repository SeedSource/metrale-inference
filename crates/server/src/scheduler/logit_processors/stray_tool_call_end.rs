// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Stray `</tool_call>` mask (MODEL.toml `[behavior].mask_stray_tool_call_end`).
//!
//! GLM-5.3-Flash under load (9+ users at max-batch 16 with the MTP drafter; 4 and 16
//! users with DFlash) sometimes ends a reply that offered no tools with
//! `</tool_call>` (id 154844, not special in its tokenizer): "...careers take root,
//! grow.</tool_call>". With no grammar and no declared tools,
//! `decode_logits_step/per_token.rs` (and `emit_step/token.rs` on the speculative
//! path) end the turn on that token and `lifecycle::derive_finish_reason` reports
//! `tool_calls`. The token is never valid there, so with the flag on this stage masks
//! it to `-inf` and the next-best token wins.
//!
//! The mask applies to a row ([`applies`]) when the flag is on, the request declared
//! no tools (`tools_present` false), it has no grammar and no `<tool_call>` is open
//! (`inside_tool_body` false). A `</tool_call>` that closes a `<tool_call>` the model
//! opened stays legal, so the end-the-turn logic in `per_token.rs` is unchanged; with
//! the mask on, only that legal close can reach it.
//!
//! `tool_choice = "none"`: the chat handler renders the tools into the prompt but
//! sends `tools_active` false, so `tools_present` is false and the mask applies to a
//! stray close there too (an opened call can still close). Left as it is on purpose.
//!
//! The device-argmax paths skip this pipeline, so each one falls back to the host
//! pipeline when its argmax is the masked id on a row where the mask applies
//! ([`hits`]), exactly as they do for the post-`</think>` think ids:
//! `decode_logits_step::process_decode_logits_skipping`,
//! `verify_pipeline_helper::pick_all` (grammar and grammarless fast paths),
//! `verify_pipeline_helper::fast_masked`, the DFlash raw-argmax verdicts (K2/K3/K4,
//! the single DFlash step, the batched DFlash step) and the serial bootstrap picks
//! (`verify_pipeline_helper::repick_stray_tool_call_end`). Token 0 suppresses the id
//! through `FirstTokenPolicy::stray_tool_call_end`; the pipelined (fed) decode lane
//! does not run ahead for such a row (`core/pipeline.rs::ahead_allowed`).
//!
//! Owner: scheduler.
//! Invariants:
//! - With the flag off, nothing here changes a logit or a pick.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

/// 2026-10-15: Times a device argmax (decode, verify window, raw verdict or
/// bootstrap) landed on a masked `</tool_call>` and the pick fell back to the host
/// pipeline. Write-only diagnostic counter.
pub(crate) static STRAY_TOOL_CALL_END_FALLBACKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 2026-10-15: Count one fallback (see [`STRAY_TOOL_CALL_END_FALLBACKS`]).
pub(crate) fn count_fallback() {
    STRAY_TOOL_CALL_END_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// 2026-10-15: Whether the stray `</tool_call>` mask applies to this row now:
/// `enabled` (MODEL.toml flag, `WatchdogParams::mask_stray_tool_call_end`), no
/// declared tools, no grammar, and no `<tool_call>` open.
pub(crate) fn applies(enabled: bool, a: &ActiveSeq) -> bool {
    enabled && !a.tools_present && a.grammar_state.is_none() && !a.inside_tool_body
}

/// 2026-10-15: Whether any of `picks` (device argmax ids) is the tool-call end
/// token on a row where [`applies`] holds, so the caller must take the host
/// pipeline instead. Judged on the step-start state: for a verify window that
/// opens a `<tool_call>` before closing it this is conservative (host path,
/// where `pick_positions_from_host` advances `inside_tool_body` per position);
/// a window that starts inside a tool body ends the turn at its first close
/// (`emit_step/token.rs`), so a later position is never committed.
pub(crate) fn hits(
    enabled: bool,
    a: &ActiveSeq,
    tool_call_end_token: Option<u32>,
    picks: &[u32],
) -> bool {
    match tool_call_end_token {
        Some(t) => applies(enabled, a) && picks.contains(&t),
        None => false,
    }
}

/// 2026-10-15: The id token 0 must suppress for a sequence about to be born
/// with these facts, or `None`. Token 0 has no tool body open yet.
pub(crate) fn first_token_mask_id(
    enabled: bool,
    tools_present: bool,
    has_grammar: bool,
    tool_call_end_token: Option<u32>,
) -> Option<u32> {
    if enabled && !tools_present && !has_grammar {
        tool_call_end_token
    } else {
        None
    }
}

/// 2026-10-15: The pipeline stage: masks `ctx.tool_call_end_token` while
/// [`applies`] holds.
pub struct StrayToolCallEndMask;

impl LogitsProcessor for StrayToolCallEndMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if applies(ctx.watchdog.mask_stray_tool_call_end, a)
            && let Some(end_tok) = ctx.tool_call_end_token
        {
            let idx = end_tok as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "stray_tool_call_end_mask"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::logit_processors::SamplingLevers;
    use crate::scheduler::test_support::test_seq;

    const VOCAB: usize = 16;
    const TOOL_CALL_END: u32 = 9;
    const HELLO: u32 = 3;

    fn plain_seq() -> ActiveSeq {
        let (mut a, _rx) = test_seq(Vec::new(), 100, None, 4);
        a.finished = false;
        a.tools_present = false;
        a.grammar_state = None;
        a.inside_tool_body = false;
        a.tool_call_end_token = Some(TOOL_CALL_END);
        a
    }

    fn run(enabled: bool, a: &mut ActiveSeq, tce: Option<u32>) -> Vec<f32> {
        let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
        let io = crate::scheduler::io::SchedIo::for_test();
        let ctx = LogitsContext {
            scratch: &scratch,
            tel: &*io.tel,
            clock: &*io.clock,
            watchdog: crate::scheduler::helpers::WatchdogParams {
                mask_stray_tool_call_end: enabled,
                ..Default::default()
            },
            boundary_mask: None,
            mid_word_mask: None,
            sampling: SamplingLevers::default(),
            think_end_token: None,
            think_start_token: None,
            tool_call_start_token: None,
            tool_call_end_token: tce,
            code_fence_token: None,
            verify_pos: 0,
        };
        let mut logits: Vec<f32> = (0..VOCAB).map(|i| i as f32).collect();
        logits[TOOL_CALL_END as usize] = 100.0;
        let out = StrayToolCallEndMask.apply(&mut logits, a, &ctx);
        assert_eq!(out, ProcessorOutcome::Continue);
        logits
    }

    fn untouched() -> Vec<f32> {
        let mut l: Vec<f32> = (0..VOCAB).map(|i| i as f32).collect();
        l[TOOL_CALL_END as usize] = 100.0;
        l
    }

    #[test]
    fn masks_only_the_end_token_on_a_plain_row_with_the_flag_on() {
        let mut a = plain_seq();
        let l = run(true, &mut a, Some(TOOL_CALL_END));
        let mut want = untouched();
        want[TOOL_CALL_END as usize] = f32::NEG_INFINITY;
        assert_eq!(
            l, want,
            "only </tool_call> goes to -inf; the runner-up wins"
        );
        assert_eq!(
            metrale_sampling::argmax_first_wins_f32(&l),
            (VOCAB - 1) as u32
        );
    }

    #[test]
    fn flag_off_leaves_the_logits_untouched() {
        let mut a = plain_seq();
        assert_eq!(run(false, &mut a, Some(TOOL_CALL_END)), untouched());
    }

    #[test]
    fn declared_tools_leave_the_logits_untouched() {
        let mut a = plain_seq();
        a.tools_present = true;
        assert_eq!(run(true, &mut a, Some(TOOL_CALL_END)), untouched());
    }

    #[test]
    fn a_grammar_leaves_the_logits_untouched() {
        let vocab = crate::grammar::tests::test_vocab();
        let mut engine = crate::grammar::GrammarEngine::new(&vocab, &[]).unwrap();
        let compiled = engine
            .compile_hermes_tool_grammar(&crate::grammar::tests::test_tool_defs(), false)
            .unwrap();
        let mut a = plain_seq();
        a.grammar_state =
            Some(crate::grammar::GrammarState::new(&compiled, engine.vocab_size()).unwrap());
        assert_eq!(run(true, &mut a, Some(TOOL_CALL_END)), untouched());
    }

    #[test]
    fn an_open_tool_body_leaves_the_logits_untouched() {
        let mut a = plain_seq();
        a.inside_tool_body = true;
        assert_eq!(
            run(true, &mut a, Some(TOOL_CALL_END)),
            untouched(),
            "a </tool_call> closing an opened <tool_call> stays legal"
        );
    }

    #[test]
    fn no_end_token_configured_is_a_no_op() {
        let mut a = plain_seq();
        assert_eq!(run(true, &mut a, None), untouched());
    }

    #[test]
    fn stage_name_is_pinned() {
        assert_eq!(StrayToolCallEndMask.name(), "stray_tool_call_end_mask");
        assert!(!StrayToolCallEndMask.is_argmax_invariant());
    }

    #[test]
    fn hits_needs_the_flag_the_row_conditions_and_the_id() {
        let mut a = plain_seq();
        let t = Some(TOOL_CALL_END);
        assert!(hits(true, &a, t, &[HELLO, TOOL_CALL_END]));
        assert!(hits(true, &a, t, &[TOOL_CALL_END]));
        assert!(!hits(true, &a, t, &[HELLO, HELLO]));
        assert!(!hits(true, &a, t, &[]));
        assert!(!hits(false, &a, t, &[TOOL_CALL_END]));
        assert!(!hits(true, &a, None, &[TOOL_CALL_END]));
        a.tools_present = true;
        assert!(!hits(true, &a, t, &[TOOL_CALL_END]));
        a.tools_present = false;
        a.inside_tool_body = true;
        assert!(!hits(true, &a, t, &[TOOL_CALL_END]));
    }

    #[test]
    fn watchdog_params_carry_the_model_flag_default_off() {
        use crate::scheduler::helpers::WatchdogParams;
        assert!(!WatchdogParams::default().mask_stray_tool_call_end);
        let off = metrale_kernels::ModelBehavior::default();
        assert!(
            !off.mask_stray_tool_call_end,
            "default is off for every model"
        );
        assert!(!WatchdogParams::from_behavior(&off, None, None).mask_stray_tool_call_end);
        let on = metrale_kernels::ModelBehavior {
            mask_stray_tool_call_end: true,
            ..metrale_kernels::ModelBehavior::default()
        };
        assert!(WatchdogParams::from_behavior(&on, None, None).mask_stray_tool_call_end);
    }

    #[test]
    fn first_token_mask_id_follows_the_birth_facts() {
        let t = Some(TOOL_CALL_END);
        assert_eq!(first_token_mask_id(true, false, false, t), t);
        assert_eq!(first_token_mask_id(false, false, false, t), None);
        assert_eq!(first_token_mask_id(true, true, false, t), None);
        assert_eq!(first_token_mask_id(true, false, true, t), None);
        assert_eq!(first_token_mask_id(true, false, false, None), None);
    }
}
