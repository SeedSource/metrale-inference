// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: tests of `pick_positions_from_host` on verify rows that do
//! and do not cross `</think>`, over synthetic BF16 rows and a real
//! `GrammarState` compiled from a `required` tool grammar.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::pick_positions::pick_positions_from_host;
use super::verify_pick_all_with_pipeline;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::logit_processors::{LogitsContext, SamplingLevers};
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;
use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};

const VOCAB: usize = 131;
const TOOL_CALL_OPEN: u32 = 128;
const TOOL_CALL_CLOSE: u32 = 129;
const EOS: u32 = 130;
/// 2026-09-25: a prose token the tool grammar refuses as the first content
/// token.
const HELLO: u32 = b'h' as u32;
/// 2026-09-25: in-vocab ids standing in for `</think>` and `<think>`.
const THINK_END: u32 = 127;
const THINK_START: u32 = 126;

fn required_tool_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[EOS as i32]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS])
}

/// 2026-09-25: a sequence inside thinking with the tool grammar attached.
fn thinking_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.tool_call_start_token = Some(TOOL_CALL_OPEN);
    a.grammar_state = Some(required_tool_grammar());
    a
}

fn row(hot: &[(u32, f32)]) -> Vec<f32> {
    let mut r = vec![0.0f32; VOCAB];
    for &(id, v) in hot {
        r[id as usize] = v;
    }
    r
}

/// 2026-09-25: rows as the little-endian BF16 `[K, vocab]` buffer
/// `pick_positions_from_host` reads.
fn bf16_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flat_map(|r| {
            r.iter().flat_map(|&v| {
                let b = v.to_bits();
                [(b >> 16) as u8, (b >> 24) as u8]
            })
        })
        .collect()
}

fn with_ctx<R>(f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        // 2026-09-29: `think_ended_gpu_argmax` at its production default
        // (on), so the A144 tests below see decode's real device-argmax
        // regime.
        sampling: SamplingLevers {
            think_ended_gpu_argmax: true,
            ..SamplingLevers::default()
        },
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
        code_fence_token: None,
        // 2026-09-25: unused: `verify_pick_with_pipeline` sets `verify_pos`
        // per position on its own copy of the context.
        verify_pos: 0,
    };
    f(&ctx)
}

#[test]
fn verify_row_crossing_think_end_masks_the_first_post_think_position() {
    let mut a = thinking_seq();
    // 2026-09-25: row 0 picks `</think>`. Row 1's argmax is prose, with
    // `<tool_call>` second; the grammar must win.
    let buf = bf16_rows(&[
        row(&[(THINK_END, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 5.0)]),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(picks[0], THINK_END, "position 0 closes the reasoning span");
    assert_eq!(
        picks[1], TOOL_CALL_OPEN,
        "the first post-think position must be picked under the pristine grammar, not free-run"
    );
    // 2026-09-25: the loop restores the flags and the matcher; it only
    // picks.
    assert!(
        a.inside_thinking && !a.think_ended,
        "sequence state restored after the loop"
    );
    let gs = a
        .grammar_state
        .as_mut()
        .expect("grammar untouched by the loop");
    assert_eq!(
        gs.num_history_steps(),
        0,
        "</think> never fed; speculative advances rolled back"
    );
}

#[test]
fn verify_row_that_stays_inside_thinking_is_not_masked() {
    // 2026-09-25: control: without `</think>` every position stays inside
    // thinking, unmasked by the grammar, and the matcher is not advanced.
    let mut a = thinking_seq();
    let buf = bf16_rows(&[
        row(&[(HELLO, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 5.0)]),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(picks, vec![HELLO, HELLO]);
    assert!(a.inside_thinking);
    assert_eq!(a.grammar_state.as_ref().unwrap().num_history_steps(), 0);
}

// 2026-09-29: post-think structural guard on the fast paths (A143 sibling).
// `verify_pick_all_with_pipeline`'s grammar and grammarless GPU-argmax fast
// paths used to return `argmax_ids` with no post-think check, unlike the
// host path above, which always runs `PostCloseThinkMask`. These tests drive
// the real `verify_pick_all_with_pipeline` entry point, so a regression in
// the guard's wiring, not only in the guard function, fails a test.

/// 2026-09-29: a sequence past `</think>` (`think_ended`) with no grammar:
/// the grammarless fast path's eligibility regime.
fn post_think_grammarless_seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(Vec::new(), 5000, None, 10);
    a.finished = false;
    a.inside_thinking = false;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.think_ended = true;
    a.grammar_state = None;
    a
}

/// 2026-09-29: like [`with_ctx`], with `fast_greedy_chat` on, so the
/// grammarless fast path is eligible absent the guard and the guard test
/// is not vacuous.
fn with_ctx_fast_greedy_chat<R>(f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: None,
        sampling: SamplingLevers {
            fast_greedy_chat: true,
            think_ended_gpu_argmax: true,
            ..SamplingLevers::default()
        },
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
        code_fence_token: None,
        verify_pos: 0,
    };
    f(&ctx)
}

/// 2026-09-29: minimal `Model`: only `vocab_size`, `logits_buffer_ptr` and
/// `copy_logits_to_host` work (the host-path D2H the guard routes to); the
/// rest is unreachable for a grammarless, temperature-0 verify call. Same
/// pattern as `PrefillStubModel` (`prefill_fifo_tests.rs`).
struct FastPathStubModel {
    vocab: usize,
    /// 2026-09-29: `[K, vocab]` BF16 bytes, as `bf16_rows` lays them out.
    buf: Vec<u8>,
}

impl FastPathStubModel {
    fn new(vocab: usize, rows: &[Vec<f32>]) -> Self {
        Self {
            vocab,
            buf: bf16_rows(rows),
        }
    }
}

impl Model for FastPathStubModel {}

impl ModelLifecycle for FastPathStubModel {
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn detach_slot_for_reuse(&self, _seq: &mut SequenceState) {}
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _new_slot: usize) -> Result<()> {
        unreachable!("no compaction in this harness")
    }
}

impl ModelForward for FastPathStubModel {
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _chunk_start: usize,
        _chunk_len: usize,
        _is_last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("no prefill in this harness")
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no prefill in this harness")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no decode in this harness")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        unreachable!("no decode in this harness")
    }
}

impl ModelLogits for FastPathStubModel {
    fn argmax_on_device(&self, _logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        unreachable!("no on-device argmax in this harness")
    }
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        unreachable!("no MTP in this harness")
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let off = logits_ptr.0 as usize;
        dst.copy_from_slice(&self.buf[off..off + dst.len()]);
        Ok(())
    }
    fn argmax_batch(&self, _l: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        unreachable!("not the decode fast path")
    }
}

impl ModelAdapters for FastPathStubModel {}

impl ModelSsmState for FastPathStubModel {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        unreachable!("no SSM in this harness")
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        unreachable!("no SSM in this harness")
    }
}

impl ModelVerify for FastPathStubModel {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        unreachable!("no speculation in this harness")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        unreachable!("no speculation in this harness")
    }
}

impl ModelDraft for FastPathStubModel {
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        unreachable!("no speculation in this harness")
    }
    fn save_hidden_for_mtp(&self, _token_idx: usize, _st: u64) -> Result<()> {
        unreachable!("no MTP in this harness")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        unreachable!("no MTP in this harness")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _mask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!("no MTP in this harness")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        unreachable!("no MTP in this harness")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _n: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        unreachable!("no speculation in this harness")
    }
}

impl ModelVision for FastPathStubModel {}

impl ModelEp for FastPathStubModel {}

impl ModelStreams for FastPathStubModel {}

impl ModelDeviceFeed for FastPathStubModel {}

#[test]
fn verify_fast_path_bails_on_reopened_think_end_after_think_ended() {
    // 2026-09-29: the single-position GPU argmax reopens `</think>`, with
    // `HELLO` the runner-up. With `think_ended` and `fast_greedy_chat` on,
    // the grammarless fast path would, without the guard, return
    // `THINK_END` with no D2H at all.
    let mut a = post_think_grammarless_seq();
    let model = FastPathStubModel::new(VOCAB, &[row(&[(THINK_END, 10.0), (HELLO, 5.0)])]);
    let argmax_ids = [THINK_END];
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &argmax_ids, &mut a, ctx, 0)
    });
    assert_eq!(
        picks,
        vec![HELLO],
        "the post-think structural guard must force the host path so \
         PostCloseThinkMask masks the reopened </think> and the runner-up \
         wins, instead of the fast path returning the raw GPU argmax"
    );
}

#[test]
fn verify_fast_path_takes_the_fast_path_when_no_structural_hit() {
    // 2026-09-29: control: an ordinary content argmax must not force the
    // host path. The stub holds no rows, so a D2H would panic; passing
    // proves the fast path is reachable in this fixture and the guarded
    // test above is not passing through some other ineligibility.
    let mut a = post_think_grammarless_seq();
    let model = FastPathStubModel::new(VOCAB, &[]);
    let argmax_ids = [HELLO];
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &argmax_ids, &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

// 2026-09-29: A144: the speculative paths apply the same `logit_bias` as
// decode. A tools-active request carries `(<tool_call>, +3.0)` in
// `ActiveSeq.logit_bias` (`sampling_setup`). Decode applied it; verify passed
// an empty bias, so spec-on picked the raw prose token where spec-off opened
// a call.

/// 2026-09-29: post-think, grammarless, greedy, tools-active.
/// `repetition_penalty` 1.05 keeps decode on the host pipeline for this row
/// (the `think_ended` device-argmax admission needs neutral penalties), so
/// decode applies the bias.
fn tools_present_seq() -> ActiveSeq {
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    a.repetition_penalty = 1.05;
    a.tool_call_start_token = Some(TOOL_CALL_OPEN);
    a.tool_call_end_token = Some(TOOL_CALL_CLOSE);
    a.logit_bias = vec![(TOOL_CALL_OPEN, 3.0)];
    a
}

/// 2026-09-29: `<tool_call>` 2.0 below the prose argmax: +3.0 flips it, 0.0
/// does not.
fn opener_near_miss_row() -> Vec<f32> {
    row(&[(HELLO, 10.0), (TOOL_CALL_OPEN, 8.0)])
}

#[test]
fn a144_verify_applies_the_same_bias_as_decode_at_a_tools_present_position() {
    use crate::scheduler::sample_step::{
        PositionKind, penalty_params_for, speculative_base_logit_bias,
    };
    let floor = crate::scheduler::helpers::WatchdogParams::default().min_reasoning_floor;
    let mut a = tools_present_seq();
    // 2026-09-29: parameter-level parity: the Verify params carry exactly
    // decode's bias.
    let decode = penalty_params_for(
        &a,
        PositionKind::FinalDecode,
        0.0,
        None,
        a.logit_bias.clone(),
        floor,
    );
    let verify_bias = speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || {
        unreachable!("a host-regime row never probes the raw argmax")
    });
    let verify = penalty_params_for(&a, PositionKind::Verify, 0.0, None, verify_bias, floor);
    assert_eq!(verify.logit_bias, decode.logit_bias);
    assert_eq!(verify.logit_bias, vec![(TOOL_CALL_OPEN, 3.0)]);

    // 2026-09-29: end to end through the host path (fast paths off in
    // `with_ctx`).
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx(|ctx| verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN],
        "verify must pick what decode picks: HELLO 10.0 < <tool_call> 8.0 + 3.0"
    );
}

#[test]
fn a144_opener_bias_is_stripped_per_position_inside_a_tool_body() {
    // 2026-09-29: one window opens a call, stays in its body, closes it, then
    // sits at a fresh opener decision. The +3.0 must be off at position 1
    // (inside the body position 0 opened, else a spurious mid-body re-open)
    // and on again at position 3 (after position 2's `</tool_call>`). The
    // step-start state (outside a body) is wrong for positions 1 and 2.
    let mut a = tools_present_seq();
    let buf = bf16_rows(&[
        row(&[(TOOL_CALL_OPEN, 10.0)]),
        opener_near_miss_row(),
        row(&[(TOOL_CALL_CLOSE, 10.0)]),
        opener_near_miss_row(),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 4, &mut a, ctx));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN, HELLO, TOOL_CALL_CLOSE, TOOL_CALL_OPEN]
    );
    assert!(
        !a.inside_tool_body,
        "tool-body flag restored after the loop"
    );

    // 2026-09-29: starting inside a body: position 0 is stripped; after the
    // close the nudge returns.
    let mut a = tools_present_seq();
    a.inside_tool_body = true;
    let buf = bf16_rows(&[
        opener_near_miss_row(),
        row(&[(TOOL_CALL_CLOSE, 10.0)]),
        opener_near_miss_row(),
    ]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 3, &mut a, ctx));
    assert_eq!(picks, vec![HELLO, TOOL_CALL_CLOSE, TOOL_CALL_OPEN]);
    assert!(a.inside_tool_body, "tool-body flag restored after the loop");
}

#[test]
fn a144_fast_greedy_falls_back_to_host_when_bias_present() {
    // 2026-09-29: `fast_greedy_chat` on and the penalties reduce-only:
    // without the A144 guard the grammarless fast path returns the raw GPU
    // argmax (HELLO) with no D2H, never seeing the bias.
    let mut a = tools_present_seq();
    assert!(crate::scheduler::sample_step::speculative_bias_forces_host(
        &a, true
    ));
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![TOOL_CALL_OPEN]);

    // 2026-09-29: control: without a bias the same fixture takes the fast
    // path.
    let mut a = tools_present_seq();
    a.logit_bias.clear();
    assert!(!crate::scheduler::sample_step::speculative_bias_forces_host(&a, true));
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
}

#[test]
fn a144_bias_skipped_exactly_where_decode_gpu_argmax_skips_it() {
    use crate::scheduler::sample_step::{
        speculative_base_logit_bias, speculative_bias_forces_host,
    };
    // 2026-09-29: neutral penalties, `think_ended`, greedy, no grammar:
    // decode admits the row to its device argmax and never applies
    // `logit_bias`. Parity with decode means verify must not apply it either.
    let mut a = tools_present_seq();
    a.repetition_penalty = 1.0;
    assert!(!speculative_bias_forces_host(&a, true));
    assert!(speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || HELLO).is_empty());
    let model = FastPathStubModel::new(VOCAB, &[opener_near_miss_row()]);
    let picks = with_ctx(|ctx| verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0));
    assert_eq!(
        picks,
        vec![HELLO],
        "decode's GPU argmax emits HELLO; so must verify"
    );

    // 2026-09-29: except when that device argmax lands on a post-think
    // `</think>`/`<think>`: decode then redoes the step on the host, bias
    // included.
    assert_eq!(
        speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || THINK_END),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );
    assert_eq!(
        speculative_base_logit_bias(&a, 0, Some(THINK_END), true, || THINK_START),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );

    // 2026-09-29: a `min_tokens` floor keeps decode on the host until it is
    // met; the floor is judged at `output_len + verify_pos`.
    a.min_tokens = 2;
    assert_eq!(
        speculative_base_logit_bias(&a, 1, Some(THINK_END), true, || HELLO),
        vec![(TOOL_CALL_OPEN, 3.0)]
    );
    assert!(speculative_base_logit_bias(&a, 2, Some(THINK_END), true, || HELLO).is_empty());

    // 2026-09-29: temperature above 0 always runs decode's host sampler.
    a.min_tokens = 0;
    a.temperature = 0.7;
    assert!(speculative_bias_forces_host(&a, true));
}

// 2026-09-29: A144b: verify's final pick uses decode's tie-break. Decode's
// host greedy path (`greedy_pick_last_wins`) and this host path process the
// same dequantised logits (BF16 to F32, no extra rounding either side), so
// an exact tie on a quantised checkpoint is real and common. The host path's
// final argmax used to resolve ties to the first equal id while decode
// resolves them to the last, which produced the K3-vs-spec-off synonym swaps
// (54/60 divergent TEB transcripts at temperature 0).

#[test]
fn a144b_verify_exact_tie_matches_decodes_last_wins_tie_break() {
    // 2026-09-29: HELLO (104) and TOOL_CALL_OPEN (128) tie at the row max,
    // 9.0, which BF16 represents exactly, so the round trip cannot break the
    // tie by accident.
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    let buf = bf16_rows(&[row(&[(HELLO, 9.0), (TOOL_CALL_OPEN, 9.0)])]);
    let picks = with_ctx(|ctx| pick_positions_from_host(&buf, VOCAB, 2, 1, &mut a, ctx));
    assert_eq!(
        picks,
        vec![TOOL_CALL_OPEN],
        "TOOL_CALL_OPEN (id 128) is the last of the two tied ids (104, 128); \
         decode's `greedy_pick_last_wins` must win here, not first-wins' HELLO"
    );
}

// 2026-09-29: A146, spec-in-think parity. With speculation inside `<think>`,
// every committed token must be the token spec-off decode would commit.
// Decode is "pipeline on the live state, then commit" once per token; here
// that is `pick_positions_from_host` with k=1 followed by `emit_token` (the
// k=1 window re-applies its own trail entry, which is exactly decode's
// pipeline-then-commit). The window path picks K positions first, then
// commits an accepted prefix. The tests drive both over the same synthetic
// rows and assert identical picks AND identical commit state.

/// 2026-09-29: a ``` stand-in (any in-vocab id that is neither structural
/// nor HELLO).
const FENCE: u32 = 96;

/// 2026-09-29: mid-reasoning and grammarless (post-think rows then decode
/// free-run).
fn thinking_grammarless_seq() -> ActiveSeq {
    let mut a = thinking_seq();
    a.grammar_state = None;
    a.min_tokens = 0;
    a
}

/// 2026-09-29: the token id marked mid-word (the `MidWordThinkEndMask`
/// input) in every `with_ctx_think` context.
const MID_WORD: u32 = 97;

/// 2026-09-29: [`with_ctx`] plus the tokenizer's ``` id, a boundary mask
/// marking `boundary_ids` and a mid-word mask marking [`MID_WORD`].
fn with_ctx_think<R>(boundary_ids: &[u32], f: impl FnOnce(&LogitsContext) -> R) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let io = crate::scheduler::io::SchedIo::for_test();
    let mut mask = vec![false; VOCAB];
    for &id in boundary_ids {
        mask[id as usize] = true;
    }
    let mut mid = vec![false; VOCAB];
    mid[MID_WORD as usize] = true;
    let ctx = LogitsContext {
        scratch: &scratch,
        tel: &*io.tel,
        clock: &*io.clock,
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: Some(mask.into()),
        mid_word_mask: Some(mid.into()),
        sampling: SamplingLevers {
            think_ended_gpu_argmax: true,
            ..SamplingLevers::default()
        },
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_CALL_OPEN),
        tool_call_end_token: Some(TOOL_CALL_CLOSE),
        code_fence_token: Some(FENCE),
        verify_pos: 0,
    };
    f(&ctx)
}

fn sched_think() -> crate::scheduler::sched_ctx::SchedCtx {
    let mut s = crate::scheduler::sched_ctx::SchedCtx::for_test();
    s.limits.code_fence_token = Some(FENCE);
    s
}

/// 2026-09-29: the commit state every later pick depends on.
type CommitState = (Vec<u32>, bool, bool, u32, bool, u32, u32, bool, u32);

fn commit_state(a: &ActiveSeq) -> CommitState {
    (
        a.output_tokens.clone(),
        a.inside_thinking,
        a.think_ended,
        a.thinking_tokens,
        a.force_end_thinking,
        a.sentence_defer_count,
        a.consecutive_confident,
        a.in_code_fence,
        a.think_watchdog_fires,
    )
}

/// 2026-09-29: the spec-off reference: one position at a time, pipeline
/// then commit.
fn run_serial(mut a: ActiveSeq, rows: &[Vec<f32>], boundary: &[u32]) -> (Vec<u32>, ActiveSeq) {
    let sched = sched_think();
    let mut picks = Vec::new();
    for r in rows {
        let buf = bf16_rows(std::slice::from_ref(r));
        let p = with_ctx_think(boundary, |ctx| {
            pick_positions_from_host(&buf, VOCAB, 2, 1, &mut a, ctx)
        });
        crate::scheduler::emit_step::emit_token(&mut a, p[0], None, &sched);
        picks.push(p[0]);
    }
    (picks, a)
}

/// 2026-09-29: the spec path: windows of `rows`, committing the first
/// `n_commit` picks of each (the accepted prefix plus the bonus). Returns
/// every committed pick.
fn run_windows(
    mut a: ActiveSeq,
    windows: &[(&[Vec<f32>], usize)],
    boundary: &[u32],
) -> (Vec<u32>, ActiveSeq) {
    let sched = sched_think();
    let mut committed = Vec::new();
    for (rows, n_commit) in windows {
        let buf = bf16_rows(rows);
        let picks = with_ctx_think(boundary, |ctx| {
            pick_positions_from_host(&buf, VOCAB, 2, rows.len(), &mut a, ctx)
        });
        for &p in &picks[..*n_commit] {
            crate::scheduler::emit_step::emit_token(&mut a, p, None, &sched);
            committed.push(p);
        }
    }
    (committed, a)
}

fn thinking_grammarless_seq_with(budget: u32, thinking_tokens: u32) -> ActiveSeq {
    let mut a = thinking_grammarless_seq();
    a.thinking_budget = Some(budget);
    a.thinking_tokens = thinking_tokens;
    a
}

#[test]
fn spec_think_budget_arm_mid_window_forces_think_end_at_next_position() {
    // 2026-09-29: the budget is reached by the commit of position 0, so
    // spec-off injects `</think>` as the very next token (hard override:
    // thinking_tokens >= 3 * budget). The window must pick it at position 1
    // (a stale thinking_tokens count picked HELLO), so any draft continuing
    // the reasoning is rejected there and an accepted run never needs
    // truncating.
    let mut a = thinking_grammarless_seq_with(2, 5);
    let rows = vec![row(&[(HELLO, 10.0)]); 3];
    let picks = with_ctx_think(&[], |ctx| {
        pick_positions_from_host(&bf16_rows(&rows), VOCAB, 2, 3, &mut a, ctx)
    });
    assert_eq!(picks[..2], [HELLO, THINK_END]);
    // 2026-09-29: restored: the window only picks.
    assert_eq!(a.thinking_tokens, 5);
    assert!(!a.force_end_thinking && a.inside_thinking && a.output_tokens.is_empty());
    assert_eq!(a.spec_think_trail.len(), 3);

    let (serial_picks, serial) = run_serial(thinking_grammarless_seq_with(2, 5), &rows, &[]);
    let (win_picks, win) = run_windows(thinking_grammarless_seq_with(2, 5), &[(&rows, 3)], &[]);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_code_fence_defers_injection_per_position() {
    // 2026-09-29: `</think>` is armed; FENCE and HELLO are sentence
    // boundaries. Position 0 opens a fence, so position 1 (previous token
    // FENCE, a boundary) must DEFER (in_code_fence); without per-position
    // fence tracking the window injected `</think>` inside the code block.
    // Position 2 closes the fence; position 3 (previous FENCE, fence closed)
    // injects.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.force_end_thinking = true;
        a
    };
    let rows = vec![
        row(&[(FENCE, 10.0)]),
        row(&[(HELLO, 10.0)]),
        row(&[(FENCE, 10.0)]),
        row(&[(HELLO, 10.0)]),
    ];
    let boundary = [FENCE, HELLO];
    let mut a = mk();
    // 2026-09-29: a non-boundary previous token, so position 0 defers.
    a.output_tokens = vec![1];
    let picks = with_ctx_think(&boundary, |ctx| {
        pick_positions_from_host(&bf16_rows(&rows), VOCAB, 2, 4, &mut a, ctx)
    });
    assert_eq!(picks, vec![FENCE, HELLO, FENCE, THINK_END]);
    assert!(!a.in_code_fence, "fence state restored after the loop");

    let mut s0 = mk();
    s0.output_tokens = vec![1];
    let mut w0 = mk();
    w0.output_tokens = vec![1];
    let (serial_picks, serial) = run_serial(s0, &rows, &boundary);
    let (win_picks, win) = run_windows(w0, &[(&rows[..2], 2), (&rows[2..], 2)], &boundary);
    assert_eq!(serial_picks, picks);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
    assert!(!serial.inside_thinking && !serial.in_code_fence);
}

#[test]
fn spec_think_partial_accept_does_not_leak_pipeline_accumulators() {
    // 2026-09-29: the F2 streak arms `force_end_thinking` inside the
    // pipeline. Window 1 (3 positions) arms it at position 2, but only
    // position 0 commits (rejection at 1). The window used to keep position
    // 2's accumulators (streak 60, armed), so window 2 injected `</think>`
    // one token early. Serial: HELLO, HELLO, `</think>` (F2 arms at step 2
    // and the previous HELLO is a boundary), then post-think HELLO.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 400;
        a.consecutive_confident = 57;
        a
    };
    let rows = vec![row(&[(HELLO, 10.0)]); 4];
    let boundary = [HELLO];
    let (serial_picks, serial) = run_serial(mk(), &rows, &boundary);
    assert_eq!(serial_picks, vec![HELLO, HELLO, THINK_END, HELLO]);
    let (win_picks, win) = run_windows(mk(), &[(&rows[..3], 1), (&rows[1..], 3)], &boundary);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_loop_watchdog_parity_window_vs_serial() {
    // 2026-09-29: a period-4 reasoning loop. The thinking-loop watchdog must
    // arm at the same committed token on both paths (it used to run on
    // decode only), and the window must see it arm mid-window (history and
    // thinking_tokens advanced per position) so the injection lands on the
    // same position.
    let ids = [10u32, 11, 12, 13];
    let rows: Vec<Vec<f32>> = (0..72).map(|i| row(&[(ids[i % 4], 10.0)])).collect();
    let boundary = [13u32];
    let (serial_picks, serial) = run_serial(thinking_grammarless_seq(), &rows, &boundary);
    assert!(
        serial.think_watchdog_fires >= 1,
        "the thinking-loop watchdog fired on the serial path"
    );
    assert!(serial_picks.contains(&THINK_END));
    let windows: Vec<(&[Vec<f32>], usize)> = rows.chunks(3).map(|c| (c, c.len())).collect();
    let (win_picks, win) = run_windows(thinking_grammarless_seq(), &windows, &boundary);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_mid_word_mask_reads_the_window_prev_token() {
    // 2026-09-29: "think-close one sentence late": the committed history
    // ends mid-word; position 0 finishes the sentence (HELLO stands in for
    // '.', not mid-word) and position 1's argmax is `</think>`. Spec-off
    // sees the previous token HELLO and closes. The window used to read the
    // STALE committed `output_tokens.last()` (mid-word) at position 1, mask
    // `</think>`, and keep reasoning with the runner-up.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 50;
        a.output_tokens = vec![MID_WORD];
        a
    };
    let rows = vec![
        row(&[(HELLO, 10.0)]),
        row(&[(THINK_END, 10.0), (TOOL_CALL_CLOSE, 9.0)]),
    ];
    let (serial_picks, serial) = run_serial(mk(), &rows, &[]);
    assert_eq!(serial_picks, vec![HELLO, THINK_END]);
    let (win_picks, win) = run_windows(mk(), &[(&rows, 2)], &[]);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

#[test]
fn spec_think_penalty_history_includes_earlier_window_picks() {
    // 2026-09-29: with a history penalty armed, position i must be penalised
    // against picks 0..i-1 exactly as decode (which has committed them)
    // penalises it. Position 1: HELLO 10.0 against a 9.0 runner-up;
    // repetition penalty 2.0 on the already-picked HELLO flips it
    // (10/2 < 9).
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 50;
        a.repetition_penalty = 2.0;
        a
    };
    let rows = vec![
        row(&[(HELLO, 10.0)]),
        row(&[(HELLO, 10.0), (TOOL_CALL_CLOSE, 9.0)]),
    ];
    let (serial_picks, serial) = run_serial(mk(), &rows, &[]);
    assert_eq!(serial_picks, vec![HELLO, TOOL_CALL_CLOSE]);
    let (win_picks, win) = run_windows(mk(), &[(&rows, 2)], &[]);
    assert_eq!(win_picks, serial_picks);
    assert_eq!(commit_state(&win), commit_state(&serial));
}

// 2026-09-29: A146 review fixes.

#[test]
fn fast_path_immunity_sees_earlier_window_picks() {
    // 2026-09-29: `ReduceOnly` (repetition penalty 1.05) grammarless fast
    // path, window argmax [X, X] with X new. Decode and the host path
    // penalise position 1 against position 0's X (9.25 / 1.05 < 9.0, so the
    // runner-up wins). The fast path used to test immunity against the
    // committed history only and returned [X, X].
    const X: u32 = 90;
    const Y: u32 = 91;
    let mk = || {
        let mut a = post_think_grammarless_seq();
        a.min_tokens = 0;
        a.repetition_penalty = 1.05;
        a.logit_bias.clear();
        a
    };
    let rows = [row(&[(X, 9.25)]), row(&[(X, 9.25), (Y, 9.0)])];
    let model = FastPathStubModel::new(VOCAB, &rows);
    let mut a = mk();
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[X, X], &mut a, ctx, 0)
    });
    assert_eq!(
        picks,
        vec![X, Y],
        "the fast path must agree with the host path and decode"
    );
    let mut a = mk();
    let slow =
        with_ctx(|ctx| pick_positions_from_host(&bf16_rows(&rows), VOCAB, 2, 2, &mut a, ctx));
    assert_eq!(slow, vec![X, Y]);
}

#[test]
fn stale_trail_never_reaches_a_windowless_verify_commit() {
    // 2026-09-29: a partial accept leaves trail entries j+1..; a later verify
    // that returns from a fast path (no window) emits a token whose
    // (tok, out_len) can coincide with the stale entry. It must not be
    // applied.
    let mut a = post_think_grammarless_seq();
    a.min_tokens = 0;
    a.logit_bias.clear();
    let stale = crate::scheduler::think_commit::SpecThinkTrail {
        tok: HELLO,
        out_len: a.output_tokens.len(),
        consecutive_confident: 42,
        sentence_defer_count: 7,
        force_end_thinking: true,
    };
    a.spec_think_trail.push_back(stale);
    let model = FastPathStubModel::new(VOCAB, &[row(&[(HELLO, 10.0)])]);
    let picks = with_ctx_fast_greedy_chat(|ctx| {
        verify_pick_all_with_pipeline(&model, &[HELLO], &mut a, ctx, 0)
    });
    assert_eq!(picks, vec![HELLO]);
    crate::scheduler::emit_step::emit_token(&mut a, HELLO, None, &sched_think());
    assert_eq!(
        (
            a.consecutive_confident,
            a.sentence_defer_count,
            a.force_end_thinking
        ),
        (0, 0, false),
        "a stale trail entry leaked into a windowless commit"
    );
}

#[test]
fn self_spec_commits_token_0_before_picking_the_window() {
    // 2026-09-29: verify position 0 is the token AFTER token_0. The history
    // ends mid-word; token_0 (HELLO) finishes the word; position 0's argmax
    // is `</think>`. Spec-off commits HELLO, then sees the previous token
    // HELLO and closes. Picking the window before committing token_0 saw the
    // previous token MID_WORD and masked `</think>`.
    let mk = || {
        let mut a = thinking_grammarless_seq();
        a.thinking_tokens = 50;
        a.output_tokens = vec![MID_WORD];
        a
    };
    let rows = vec![
        row(&[(THINK_END, 10.0), (TOOL_CALL_CLOSE, 9.0)]),
        row(&[(HELLO, 10.0)]),
    ];
    let (serial_picks, serial) = run_serial(
        {
            let mut a = mk();
            crate::scheduler::emit_step::emit_token(&mut a, HELLO, None, &sched_think());
            a
        },
        &rows[..1],
        &[],
    );
    assert_eq!(serial_picks, vec![THINK_END]);

    let mut a = mk();
    let sched = sched_think();
    let buf = bf16_rows(&rows);
    let n = crate::scheduler::spec_step::self_spec_commit(&mut a, HELLO, &[HELLO], &sched, |a| {
        with_ctx_think(&[], |ctx| {
            pick_positions_from_host(&buf, VOCAB, 2, 2, a, ctx)
        })
    });
    assert_eq!(n, Some(0), "draft HELLO rejected by the forced-free close");
    assert_eq!(a.last_token, THINK_END);
    assert_eq!(commit_state(&a), commit_state(&serial));
}
