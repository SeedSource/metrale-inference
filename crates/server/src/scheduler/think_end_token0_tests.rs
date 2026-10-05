// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: A95: a `</think>` sampled as token 0 ends thinking in the
//! scheduler, as a later sampled `</think>` does
//! (`first_token_policy::end_thinking_at_token0`, lever
//! `METRALE_THINK_END_AT_TOKEN0`, default on).
//!
//! Births go through the three real birth sites (`start_chunked_prefill`,
//! `prefill_request`, `promote_completed_prefills`) over a
//! `PreemptStubModel` whose greedy token 0 is the given id. Later tokens run
//! the real logits pipeline (`run_pipeline`) and commit through
//! `emit_token`, the MTP / verify commit path. The spec-off decode commit is
//! covered in `decode_logits_step/token0_tests.rs`, which shares the
//! fixtures marked `pub(super)` here.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;

use super::confidence::THINK_DEFER_BUDGET_FACTOR;
use super::emit_step::{StartPrefillResult, emit_token};
use super::levers::SchedLevers;
use super::logit_processors::{LogitsContext, run_pipeline};
use super::phase_promote_prefills::promote_completed_prefills;
use super::prefill_a_step::start_chunked_prefill;
use super::prefill_b_step::prefill_request;
use super::sched_ctx::SchedCtx;
use super::test_support::{PreemptStubModel, blocking_request, test_prefill};
use super::types::ActiveSeq;
use crate::api::InferenceRequest;

/// 2026-10-04: GLM-5.3's `</think>` (the A95 trace) and its EOS ids from
/// `generation_config.json` (the config fixture's `eos_token_id`);
/// `<|user|>` is 154827.
pub(super) const THINK_END: u32 = 154_842;
pub(super) const EOS_IDS: [u32; 3] = [154_820, 154_827, 154_829];
pub(super) const USER: u32 = 154_827;
/// 2026-10-04: Stand-ins for `<think>`, `<tool_call>` and `</tool_call>`;
/// only their distinctness matters here.
pub(super) const THINK_START: u32 = 154_841;
pub(super) const TOOL_CALL: u32 = 154_843;
const TOOL_CALL_END: u32 = 154_844;
/// 2026-10-04: The checkpoint's padded vocab (config fixture `vocab_size`).
pub(super) const VOCAB: usize = 154_880;
/// 2026-10-04: A short thinking budget. With no boundary mask the forced
/// `</think>` lands once `BUDGET * THINK_DEFER_BUDGET_FACTOR` tokens were
/// counted as thinking.
pub(super) const BUDGET: u32 = 4;
/// 2026-10-04: First of the plain content ids the stub model "prefers".
pub(super) const WORD: u32 = 1_000;
/// 2026-10-04: "7 × 8 = **56**", the TEB TC-45 answer, as 8 content ids.
pub(super) const ANSWER: [u32; 8] = [1_100, 1_101, 1_102, 1_103, 1_104, 1_105, 1_106, 1_107];

/// 2026-10-04: The birth site that builds the sequence.
#[derive(Clone, Copy, Debug)]
pub(super) enum Site {
    FirstChunk,
    PrefillRequest,
    Promote,
}

pub(super) const SITES: [Site; 3] = [Site::FirstChunk, Site::PrefillRequest, Site::Promote];

/// 2026-10-04: A test context over `model` with the lever as given.
pub(super) fn sched_with(model: Arc<PreemptStubModel>, lever: bool) -> SchedCtx {
    let mut sched = SchedCtx::for_test_with(model);
    let mut levers = SchedLevers::defaults();
    levers.think_end_at_token0 = lever;
    sched.levers = Arc::new(levers);
    sched
}

/// 2026-10-04: A thinking-on request with the test budget. Nothing in the
/// scheduler reads the response receiver's state, so its drop is harmless.
fn request(legacy_tool: bool) -> InferenceRequest {
    let mut req = blocking_request(None);
    if let InferenceRequest::Blocking {
        max_tokens,
        enable_thinking,
        thinking_budget,
        require_tool_call,
        tools_present,
        ..
    } = &mut req
    {
        *max_tokens = 64;
        *enable_thinking = true;
        *thinking_budget = Some(BUDGET);
        *require_tool_call = legacy_tool;
        *tools_present = legacy_tool;
    }
    req
}

/// 2026-10-04: A sequence born at `site` whose token 0 is `first`, with the
/// lever as given and, when `legacy_tool`, a forced tool call without a
/// grammar (`require_tool_call`). The EOS ids are set after birth: token 0
/// is sampled with them suppressed, which needs host logits the stub lacks.
pub(super) fn born(site: Site, first: u32, lever: bool, legacy_tool: bool) -> ActiveSeq {
    let model = Arc::new(PreemptStubModel {
        first_token: Some(first),
        ..Default::default()
    });
    let sched = sched_with(model.clone(), lever);
    let req = request(legacy_tool);
    let (te, ts, tc, tce) = (
        Some(THINK_END),
        Some(THINK_START),
        Some(TOOL_CALL),
        Some(TOOL_CALL_END),
    );
    let mut a = match site {
        Site::FirstChunk => {
            match start_chunked_prefill(
                &sched,
                te,
                ts,
                tc,
                tce,
                &*model,
                req,
                &[],
                64,
                0,
                0,
                &mut None,
                64,
                false,
                None,
                None,
            )
            .expect("prefill")
            {
                StartPrefillResult::Active(a) => a,
                _ => panic!("a one-chunk prompt joins decode"),
            }
        }
        Site::PrefillRequest => prefill_request(
            &sched,
            te,
            ts,
            tc,
            tce,
            &*model,
            req,
            &[],
            &mut None,
            64,
            None,
        )
        .expect("prefill")
        .expect("joins decode"),
        Site::Promote => {
            let (mut p, _rx) = test_prefill(vec![1, 2, 3]);
            p.enable_thinking = true;
            p.thinking_budget = Some(BUDGET);
            p.max_tokens = 64;
            p.require_tool_call = legacy_tool;
            p.tools_present = legacy_tool;
            let mut prefilling = vec![p];
            let mut active = Vec::new();
            promote_completed_prefills(
                &*model,
                &sched.io,
                &mut prefilling,
                vec![(0, Ok(first))],
                &mut active,
                te,
                ts,
                tc,
                tce,
                4096,
                lever,
            );
            active.pop().expect("joins decode")
        }
    };
    a.eos_tokens = EOS_IDS.to_vec();
    a.min_tokens = 0;
    a
}

/// 2026-10-04: The context the decode pipeline builds for this sequence's
/// special tokens, with no boundary or mid-word mask.
pub(super) fn ctx<'a>(sched: &'a SchedCtx, a: &ActiveSeq) -> LogitsContext<'a> {
    LogitsContext {
        scratch: &sched.scratch,
        tel: &*sched.io.tel,
        clock: &*sched.io.clock,
        watchdog: sched.watchdog,
        boundary_mask: None,
        mid_word_mask: None,
        sampling: sched.levers.sampling(),
        think_end_token: a.think_end_token,
        think_start_token: a.think_start_token,
        tool_call_start_token: a.tool_call_start_token,
        tool_call_end_token: a.tool_call_end_token,
        code_fence_token: None,
        verify_pos: 0,
    }
}

/// 2026-10-04: The pipeline's pick when the model prefers `want` over a
/// flat `vocab`: a forced token, else the processed argmax.
pub(super) fn pick(a: &mut ActiveSeq, sched: &SchedCtx, want: u32, vocab: usize) -> u32 {
    let mut logits = vec![0.0f32; vocab];
    logits[want as usize] = 10.0;
    if let Some(forced) = run_pipeline(&mut logits, a, &ctx(sched, a)) {
        return forced;
    }
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// 2026-10-04: One MTP-path step: pick, then commit through `emit_token`.
fn step(a: &mut ActiveSeq, sched: &SchedCtx, want: u32) -> u32 {
    let tok = pick(a, sched, want, VOCAB);
    emit_token(a, tok, None, sched);
    tok
}

/// 2026-10-04: A context for the emit steps (the lever is read only at
/// birth).
fn emit_sched() -> SchedCtx {
    sched_with(Arc::new(PreemptStubModel::default()), true)
}

/// 2026-10-04: Thinking state after each step, for on/off comparisons.
type Trace = Vec<(u32, bool, bool, u32, bool)>;

fn run(a: &mut ActiveSeq, sched: &SchedCtx, wants: impl IntoIterator<Item = u32>) -> Trace {
    wants
        .into_iter()
        .map(|w| {
            let tok = step(a, sched, w);
            let flags = (a.inside_thinking, a.think_ended, a.force_end_thinking);
            (tok, flags.0, flags.1, a.thinking_tokens, flags.2)
        })
        .collect()
}

/// 2026-10-04: Content ids that never repeat, past the forced-close window.
fn answer_words() -> impl Iterator<Item = u32> {
    WORD..WORD + 4 * BUDGET * THINK_DEFER_BUDGET_FACTOR
}

#[test]
fn lever_on_token0_close_ends_thinking_at_every_birth_site() {
    for site in SITES {
        let a = born(site, THINK_END, true, false);
        assert!(!a.inside_thinking, "{site:?}: still inside thinking");
        assert!(a.think_ended && a.think_just_ended, "{site:?}");
        assert_eq!(a.thinking_tokens, 0, "{site:?}: nothing was thought");
        assert!(!a.force_end_thinking, "{site:?}");
        assert_eq!(a.output_tokens, vec![THINK_END], "{site:?}: token 0 kept");
        assert_eq!(a.remaining, 63, "{site:?}: token 0 drew the budget");
        assert_eq!(a.thinking_budget, Some(BUDGET), "{site:?}");
    }
}

#[test]
fn lever_on_no_forced_close_and_no_budget_count_in_the_budget_window() {
    let sched = emit_sched();
    for site in SITES {
        let mut a = born(site, THINK_END, true, false);
        let trace = run(&mut a, &sched, answer_words());
        assert!(
            trace.iter().all(|t| t.0 != THINK_END),
            "{site:?}: a second </think> was forced mid-answer: {trace:?}"
        );
        assert!(!a.inside_thinking && !a.force_end_thinking, "{site:?}");
        assert_eq!(a.thinking_tokens, 0, "{site:?}: answer counted as thinking");
        let closes = a.output_tokens.iter().filter(|&&t| t == THINK_END).count();
        assert_eq!(closes, 1, "{site:?}");
        assert!(!a.finished, "{site:?}");
    }
}

#[test]
fn lever_off_reproduces_the_old_birth_and_the_forced_mid_answer_close() {
    let sched = emit_sched();
    for site in SITES {
        let mut a = born(site, THINK_END, false, false);
        assert!(a.inside_thinking && !a.think_ended, "{site:?}: old birth");
        assert!(!a.think_just_ended && a.thinking_tokens == 0, "{site:?}");
        let trace = run(&mut a, &sched, answer_words());
        let at = trace.iter().position(|t| t.0 == THINK_END);
        let window = (BUDGET * THINK_DEFER_BUDGET_FACTOR) as usize;
        assert_eq!(at, Some(window), "{site:?}: forced close step: {trace:?}");
        assert_eq!(
            trace[window].3, window as u32,
            "{site:?}: answer as thinking"
        );
        assert!(a.think_force_closed, "{site:?}: the close was forced");
        let closes = a.output_tokens.iter().filter(|&&t| t == THINK_END).count();
        assert_eq!(closes, 2, "{site:?}: the second </think> the stream scrubs");
    }
}

#[test]
fn normal_thinking_is_unchanged_including_the_forced_close_at_the_budget() {
    let sched = emit_sched();
    for site in SITES {
        // 2026-10-04: a natural close at N > 0, then content and EOS.
        let natural = [WORD + 1, WORD + 2, THINK_END, WORD + 3, WORD + 4, USER];
        // 2026-10-04: thinking that never closes by itself.
        let forced = WORD + 10..WORD + 10 + 2 * BUDGET * THINK_DEFER_BUDGET_FACTOR;
        let mut traces = Vec::new();
        for lever in [true, false] {
            let mut a = born(site, WORD, lever, false);
            assert!(a.inside_thinking && !a.think_ended, "{site:?}/{lever}");
            let t1 = run(&mut a, &sched, natural);
            assert!(a.finished && !a.inside_thinking, "{site:?}/{lever}: {t1:?}");
            let mut b = born(site, WORD, lever, false);
            let t2 = run(&mut b, &sched, forced.clone());
            let at = t2.iter().position(|t| t.0 == THINK_END);
            let window = (BUDGET * THINK_DEFER_BUDGET_FACTOR) as usize;
            assert_eq!(at, Some(window), "{site:?}/{lever}: forced close: {t2:?}");
            assert!(
                b.think_force_closed && !b.inside_thinking,
                "{site:?}/{lever}"
            );
            traces.push((t1, t2));
        }
        assert_eq!(
            traces[0], traces[1],
            "{site:?}: the lever changed normal thinking"
        );
    }
}

#[test]
fn eos_and_user_end_the_turn_after_a_token0_close_only_with_the_lever_on() {
    // 2026-10-04: TEB TC-45 / A103: inside "thinking" these EOS ids were held
    // back (and fed back to the model). After a token-0 close each one ends
    // the turn: at once, and after the short answer.
    let sched = emit_sched();
    for site in SITES {
        for eos in EOS_IDS {
            let mut a = born(site, THINK_END, true, false);
            assert_eq!(step(&mut a, &sched, eos), eos);
            assert!(a.finished, "{site:?}: EOS {eos} right after </think>");
            for lever in [true, false] {
                let mut b = born(site, THINK_END, lever, false);
                run(&mut b, &sched, ANSWER);
                assert_eq!(step(&mut b, &sched, eos), eos);
                let kept = b.output_tokens.contains(&eos);
                assert_eq!(
                    b.finished, lever,
                    "{site:?}/{lever}: EOS {eos} after the answer"
                );
                assert_eq!(kept, lever, "{site:?}/{lever}: a held-back EOS is dropped");
                assert_eq!(b.inside_thinking, !lever, "{site:?}/{lever}");
            }
        }
    }
}

#[test]
fn legacy_forced_tool_call_pins_after_a_token0_close_as_after_a_later_one() {
    let sched = emit_sched();
    for site in SITES {
        let mut on = born(site, THINK_END, true, true);
        assert!(on.require_tool_call && on.tool_request, "{site:?}: fixture");
        assert_eq!(step(&mut on, &sched, WORD), TOOL_CALL, "{site:?}: pinned");
        assert!(on.tool_call_opened && !on.require_tool_call, "{site:?}");

        let mut later = born(site, WORD, true, true);
        assert_eq!(step(&mut later, &sched, THINK_END), THINK_END, "{site:?}");
        assert_eq!(step(&mut later, &sched, WORD), TOOL_CALL, "{site:?}");

        let mut off = born(site, THINK_END, false, true);
        assert_eq!(
            step(&mut off, &sched, WORD),
            WORD,
            "{site:?}: lever off: <tool_call> masked inside thinking, no pin"
        );
    }
}

#[test]
fn tool_grammar_engages_on_the_first_token_after_a_token0_close() {
    use crate::grammar::tests::{test_tool_defs, test_vocab};
    use crate::grammar::{GrammarEngine, GrammarState};
    // 2026-10-04: the toy vocab of `grammar::tests`: `<tool_call>` is 128,
    // EOS 130, 131 ids; `</think>` stays GLM's id, outside it.
    const OPEN: u32 = 128;
    const HELLO: u32 = b'h' as u32;
    let grammar = || {
        let vocab = test_vocab();
        let mut engine = GrammarEngine::new(&vocab, &[130]).unwrap();
        let compiled = engine
            .compile_hermes_tool_grammar(&test_tool_defs(), false)
            .unwrap();
        GrammarState::new(&compiled, engine.vocab_size())
            .unwrap()
            .with_stop_tokens(&[130])
    };
    let sched = emit_sched();
    let mut picks = Vec::new();
    for (first, lever) in [(THINK_END, true), (THINK_END, false), (HELLO, true)] {
        let model = Arc::new(PreemptStubModel::default());
        let birth = sched_with(model.clone(), lever);
        let (mut p, _rx) = test_prefill(vec![1, 2, 3]);
        p.enable_thinking = true;
        p.max_tokens = 64;
        p.grammar_state = Some(grammar());
        let (mut prefilling, mut active) = (vec![p], Vec::new());
        promote_completed_prefills(
            &*model,
            &birth.io,
            &mut prefilling,
            vec![(0, Ok(first))],
            &mut active,
            Some(THINK_END),
            Some(THINK_START),
            Some(OPEN),
            None,
            4096,
            lever,
        );
        let mut a = active.pop().expect("joins decode");
        assert!(a.tool_request, "a grammar makes it a tool request");
        if first == HELLO {
            // 2026-10-04: a later close, committed directly (`</think>` is
            // outside the toy vocab).
            emit_token(&mut a, THINK_END, None, &sched);
        }
        let steps = a.grammar_state.as_ref().unwrap().num_history_steps();
        assert_eq!(steps, 0, "the matcher never saw </think>");
        // 2026-10-04: which of the model's pick and the opener survive the
        // pipeline (the grammar also allows spelling the opener byte-wise).
        let mut logits = vec![0.0f32; 131];
        logits[HELLO as usize] = 10.0;
        let c = ctx(&sched, &a);
        let forced = run_pipeline(&mut logits, &mut a, &c);
        let live = |t: u32| forced.map_or(logits[t as usize].is_finite(), |f| f == t);
        picks.push((live(HELLO), live(OPEN)));
    }
    assert_eq!(
        picks[0],
        (false, true),
        "token-0 close: the grammar applies"
    );
    assert_eq!(picks[1], (true, false), "lever off: <tool_call> masked");
    assert_eq!(picks[2], picks[0], "a later close: the same masks");
}

#[test]
fn a_birth_that_is_not_a_token0_close_is_untouched_by_the_lever() {
    for site in SITES {
        for first in [WORD, THINK_START] {
            let on = born(site, first, true, false);
            let off = born(site, first, false, false);
            let state = |a: &ActiveSeq| (a.inside_thinking, a.think_ended, a.think_just_ended);
            assert_eq!(state(&on), state(&off), "{site:?}/{first}");
        }
    }
}

#[test]
fn the_lever_ships_on_and_zero_turns_it_off_in_the_live_resolver() {
    const VAR: &str = "METRALE_THINK_END_AT_TOKEN0";
    assert!(SchedLevers::defaults().think_end_at_token0);
    // 2026-10-04: SAFETY: nothing else in this test binary writes this
    // variable. `cargo test` runs tests on parallel threads, so a
    // concurrent environment access from another test is not excluded.
    let resolve = |v: Option<&str>| {
        unsafe {
            match v {
                Some(v) => std::env::set_var(VAR, v),
                None => std::env::remove_var(VAR),
            }
        }
        SchedLevers::from_env(None).think_end_at_token0
    };
    let (unset, zero, one) = (resolve(None), resolve(Some("0")), resolve(Some("1")));
    resolve(None);
    assert!(unset, "the fix ships on");
    assert!(!zero, "=0 restores the old birth state for an A/B");
    assert!(one);
}
