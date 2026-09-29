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
