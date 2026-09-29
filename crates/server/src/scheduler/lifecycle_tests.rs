// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `lifecycle::derive_finish_reason` and `finish_sequence`.
//!
//! The call-site tests here run `finish_sequence` on a real `ActiveSeq` with a
//! blocking oneshot sink and a stub `Model`, and check the `finish_reason` the
//! response carries, so passing the wrong field fails even when the pure
//! function is right. The pure precedence tests are in
//! `lifecycle_derive_tests.rs`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::io::SchedIo;
use super::lifecycle::{derive_finish_reason, fail_sequence, finish_sequence};
use super::types::{ActiveSeq, GUARD_STOP_REQUEST_TIMEOUT, ResponseSink};
use super::{DEFAULT_LZ_PENALTY, SsmDecodeRing};
use crate::api::InferenceResponse;
use crate::ir::FINISH_REASON_TIMEOUT;
use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    Model, ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle,
    ModelLogits, ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};
use std::time::Instant;

const EOS: &[u32] = &[151645];
const TOOL_END: Option<u32> = Some(151658);
const MAX_SEQ_LEN: usize = 8192;

// 2026-09-25: Call-site tests: `finish_sequence` must pass the sequence's own fields.

/// 2026-09-25: A `Model` stub for `finish_sequence`, which reaches
/// `cache_sequence`, `free_sequence` and `ep_broadcast_cmd_for_seq` (the trait's
/// no-op default). The inference methods bail. `pub(super)` because
/// `prefill_timing_tests` also constructs it.
pub(super) struct StubModel;

impl Model for StubModel {}

impl ModelLifecycle for StubModel {
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn cache_sequence(&self, _s: &SequenceState) {}
    fn free_sequence(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<()> {
        Ok(())
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
}

impl ModelForward for StubModel {
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _cs: usize,
        _cl: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        _s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
}

impl ModelLogits for StubModel {
    fn vocab_size(&self) -> usize {
        0
    }
    fn copy_logits_to_host(&self, _p: DevicePtr, _d: &mut [u8]) -> Result<()> {
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _p: DevicePtr, _st: u64) -> Result<u32> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn argmax_batch(&self, _p: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
}

impl ModelAdapters for StubModel {}

impl ModelSsmState for StubModel {
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
}

impl ModelVerify for StubModel {
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unused in lifecycle tests")
    }
}

impl ModelDraft for StubModel {
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _n: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in lifecycle tests")
    }
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
}

impl ModelVision for StubModel {}

impl ModelEp for StubModel {}

impl ModelStreams for StubModel {}

impl ModelDeviceFeed for StubModel {}

type RespRx = tokio::sync::oneshot::Receiver<Result<InferenceResponse>>;

/// 2026-09-25: A real `ActiveSeq` with a blocking oneshot sink. `min_tokens` is
/// fixed at 7 as a decoy: a call site that passed it where `remaining`
/// belongs fails the tests that use `remaining = 0`.
fn test_seq(
    output_tokens: Vec<u32>,
    remaining: usize,
    guard_stop: Option<&'static str>,
    seq_len: usize,
) -> (ActiveSeq, RespRx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let now = Instant::now();
    let mut seq = SequenceState::host_only(0);
    seq.seq_len = seq_len;
    let a = ActiveSeq {
        seq,
        session_hash: 0,
        last_token: output_tokens.last().copied().unwrap_or(0),
        output_tokens,
        remaining,
        min_tokens: 7,
        eos_tokens: EOS.to_vec(),
        finished: true,
        error: None,
        guard_stop,
        param_close_pending: 0,
        sink: ResponseSink::Blocking(Some(tx)),
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 256,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        inside_thinking: false,
        enable_thinking: false,
        thinking_budget: None,
        repetition_detection: None,
        spontaneous_think_budget: 0,
        thinking_tokens: 0,
        force_end_thinking: false,
        think_force_closed: false,
        sentence_defer_count: 0,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: None,
        think_start_token: None,
        think_ended: false,
        think_just_ended: false,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        spec_think_trail: Default::default(),
        think_skip_count: 0,
        tool_call_end_token: TOOL_END,
        require_tool_call: false,
        tool_request: false,
        tools_present: false,
        tool_call_start_token: None,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        suppress_tool_call: false,
        disable_mtp: false,
        mtp_acct: Default::default(),
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        last_token_time: now,
        request_start: now,
        decode_start: now,
        seed: None,
        top_logprobs: None,
        logprobs_data: Vec::new(),
        timeout_at: None,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
        preempt_immune_until_tokens: 0,
    };
    (a, rx)
}

fn finish_and_recv(mut a: ActiveSeq, mut rx: RespRx) -> InferenceResponse {
    finish_sequence(
        &SchedIo::for_test_with(std::sync::Arc::new(StubModel)),
        &mut a,
        MAX_SEQ_LEN,
    );
    rx.try_recv()
        .expect("finish_sequence must send the blocking response")
        .expect("response must be Ok")
}

#[test]
fn call_site_passes_the_real_budget() {
    // 2026-09-25: Budget exhausted ⇒ "length". Red if finish_sequence stops
    // passing `a.remaining` (e.g. a constant, or the decoy `min_tokens` = 7).
    let (a, rx) = test_seq(vec![5, 6, 42], 0, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
    // 2026-09-25: Budget left ⇒ "stop".
    let (a, rx) = test_seq(vec![5, 6, 42], 7, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "stop");
}

#[test]
fn call_site_passes_the_real_seq_len() {
    // 2026-09-25: Context-ceiling stop with budget left ⇒ "length". Red if
    // the call site stops passing `a.seq.seq_len` / the served ceiling.
    let (a, rx) = test_seq(vec![5, 6, 42], 500, None, MAX_SEQ_LEN - 1);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
}

#[test]
fn call_site_passes_the_real_last_token_and_eos() {
    // 2026-09-25: Red if the call site stops passing `output_tokens.last()`
    // or `a.eos_tokens`.
    let (a, rx) = test_seq(vec![5, 6, 151645], 0, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "stop");
    let (a, rx) = test_seq(vec![5, 6, 151658], 3, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "tool_calls");
}

#[test]
fn call_site_passes_the_real_guard() {
    // 2026-09-25: Timeout is the one guard with a distinct wire reason —
    // proves `a.guard_stop` reaches the decision.
    let (a, rx) = test_seq(vec![5, 6, 42], 3, Some(GUARD_STOP_REQUEST_TIMEOUT), 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, FINISH_REASON_TIMEOUT);
    // 2026-09-25: And a degeneration guard reaches the response as "length"
    // — the truncation signal. Asserted at the call site, not just over the
    // pure function, because that is where the wire value the client
    // actually receives is decided.
    let (a, rx) = test_seq(vec![5, 6, 42], 3, Some("fuzzy_repetition"), 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
}

/// 2026-09-25: `derive_finish_reason` stops on every id in `eos_tokens`, not only the
/// first. `GLM_EOS` is a three-token stop set.
#[test]
fn every_configured_eos_token_stops_generation_independently() {
    const GLM_EOS: [u32; 3] = [154820, 154827, 154829];
    for id in GLM_EOS {
        assert_eq!(
            derive_finish_reason(None, Some(id), &GLM_EOS, None, 10, 5, 100),
            "stop",
            "generation must stop on {id}"
        );
    }
    // 2026-09-25: Non-vacuity: a token that is not in the set must not be
    // read as a stop token. This cannot be shown with budget left: "stop"
    // is also `derive_finish_reason`'s fallthrough when no other reason
    // fires, so comparing against it at a mid-budget `remaining` would pass
    // even on a broken EOS check. Exhaust the budget instead: at the hard
    // ceiling the fallthrough is "length", so the two answers separate and
    // the check means what it says.
    assert_eq!(
        derive_finish_reason(None, Some(154821), &GLM_EOS, None, 0, 5, 100),
        "length",
        "154821 is not a stop token; at the ceiling it must read as truncation"
    );
    for id in GLM_EOS {
        assert_eq!(
            derive_finish_reason(None, Some(id), &GLM_EOS, None, 0, 5, 100),
            "stop",
            "{id} is a stop token and must outrank the ceiling"
        );
    }
    // 2026-09-25: With only the first id in the set (`GLM_EOS[..1]`), the other
    // two are not stop tokens, and at the ceiling they read as "length".
    for id in [GLM_EOS[1], GLM_EOS[2]] {
        assert_eq!(
            derive_finish_reason(None, Some(id), &GLM_EOS[..1], None, 0, 5, 100),
            "length",
            "collapsing to the primary would let {id} through"
        );
    }
}

/// 2026-09-25: A sequence retired by `fail_sequence` must reach the caller as an
/// error that carries the failure text, not as a normal completion.
#[test]
fn a_failed_sequence_is_sent_as_an_error_not_a_normal_finish() {
    let (mut a, mut rx) = test_seq(vec![5, 6, 42], 500, None, 10);
    fail_sequence(
        &mut a,
        "decode_verify_graphed_k3: DSA indexer cache: 16385".into(),
    );
    assert!(a.finished, "fail_sequence must still retire the sequence");
    finish_sequence(
        &SchedIo::for_test_with(std::sync::Arc::new(StubModel)),
        &mut a,
        MAX_SEQ_LEN,
    );
    let got = rx
        .try_recv()
        .expect("a failed sequence must still answer the caller");
    // 2026-09-25: `InferenceResponse` is not Debug, so match rather than
    // expect_err.
    let err = match got {
        Ok(_) => panic!("it must answer with an ERROR, not a completion"),
        Err(e) => e,
    };
    assert!(
        format!("{err:#}").contains("16385"),
        "the client must be told WHY: {err:#}"
    );
}

/// 2026-09-25: The error branch must not swallow ordinary completions — same funnel, no error set.
#[test]
fn a_normal_finish_is_unaffected_by_the_error_branch() {
    let (a, rx) = test_seq(vec![5, 6, 42], 0, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
}

/// 2026-09-26: Source scan of the `verify_k2_step`, `verify_k3_step` and
/// `verify_k4_step` modules, each file of their splits included: no
/// `a.finished = true;` line directly after a `tracing::error!` line. That
/// shape reports a failed generation as a normal completion; `fail_sequence`
/// is the error path.
#[test]
fn no_reachable_verify_error_arm_silently_finishes() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    for (module, split) in [
        ("verify_k2_step", &[][..]),
        ("verify_k3_step", &["stats.rs"][..]),
        ("verify_k4_step", &["stats.rs"][..]),
    ] {
        let sub = dir.join(module);
        let mut found: Vec<String> = match std::fs::read_dir(&sub) {
            Ok(entries) => entries
                .map(|e| e.expect("readable split entry").file_name())
                .filter_map(|n| n.into_string().ok())
                .collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => panic!("{}: {e}", sub.display()),
        };
        found.sort();
        assert_eq!(
            found, split,
            "{module}/ has a file this scan does not include"
        );
    }
    for (name, src) in [
        ("verify_k2_step", include_str!("verify_k2_step.rs")),
        ("verify_k3_step", include_str!("verify_k3_step.rs")),
        (
            "verify_k3_step/stats",
            include_str!("verify_k3_step/stats.rs"),
        ),
        ("verify_k4_step", include_str!("verify_k4_step.rs")),
        (
            "verify_k4_step/stats",
            include_str!("verify_k4_step/stats.rs"),
        ),
    ] {
        for (i, w) in src.lines().collect::<Vec<_>>().windows(2).enumerate() {
            let silent = w[0].contains("tracing::error!") && w[1].trim() == "a.finished = true;";
            assert!(
                !silent,
                "{name}:{}: an error arm still sets `finished = true` without \
                 fail_sequence — that returns HTTP 200 over a failed generation",
                i + 1
            );
        }
    }
}

#[path = "lifecycle_derive_tests.rs"]
mod lifecycle_derive_tests;
