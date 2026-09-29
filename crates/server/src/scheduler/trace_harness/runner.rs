// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Drives the real scheduler loop with a [`RecordingModel`] and turns one scenario into a text trace.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! The trace is the model-call sequence, then the LoRA rotation ack if the
//! scenario queued one, then what every client received.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::model::{ModelCfg, RecordingModel, SeqScript, Shared};
use crate::api::{InferenceRequest, InferenceResponse, StreamEvent};
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::limits::SchedLimits;
use crate::scheduler::vocab_masks::VocabMasks;
use crate::scheduler::{LoraCommand, WatchdogParams};
use crate::scheduling_policy::{FifoPolicy, SchedulingPolicy, SlaiPolicy};

pub(super) const EOS: u32 = 63;

#[derive(Clone, Debug)]
pub(super) struct ReqSpec {
    /// 2026-09-25: Session hash and the first prompt token: how the fake finds the script.
    pub id: u64,
    pub prompt_len: usize,
    /// 2026-09-25: Scripted generated tokens (end with [`EOS`] for a natural stop).
    pub toks: Vec<u32>,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub streaming: bool,
    pub temperature: f32,
    pub top_logprobs: Option<u8>,
    pub num_beams: u32,
    pub beam_hyp: Option<Vec<u32>>,
    pub draft_wrong_every: Option<usize>,
    pub cancel_at: Option<usize>,
    /// 2026-09-25: A deadline set when the request is built, so it has passed by the time
    /// the scheduler checks it.
    pub expired_deadline: bool,
    /// 2026-09-25: `None` means queued before the loop starts; `Some(t)` means it arrives while the
    /// loop is parked at tick `t`.
    pub arrive_at_tick: Option<usize>,
}

impl ReqSpec {
    pub fn new(id: u64, prompt_len: usize, toks: Vec<u32>) -> Self {
        let max_tokens = toks.len();
        Self {
            id,
            prompt_len,
            toks,
            max_tokens,
            min_tokens: 0,
            streaming: true,
            temperature: 0.0,
            top_logprobs: None,
            num_beams: 1,
            beam_hyp: None,
            draft_wrong_every: None,
            cancel_at: None,
            expired_deadline: false,
            arrive_at_tick: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct RunOptions {
    pub max_batch_size: usize,
    pub use_speculative: bool,
    pub dflash: bool,
    pub num_drafts: usize,
    pub max_prefill_tokens: usize,
    pub max_batch_tokens: usize,
    pub self_speculative: bool,
    pub ngram_speculative: bool,
    pub swap_space_gb: usize,
    pub slai_policy: bool,
    pub mtp_gate_force: bool,
    pub loop_watchdog: bool,
    /// 2026-09-25: Token ids the watchdog rollback treats as boundaries.
    pub boundary_tokens: Vec<u32>,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    /// 2026-09-25: A LoRA rotation queued before the loop starts; the scheduler applies it
    /// once nothing is in flight.
    pub lora_rotation: Option<String>,
    /// 2026-09-25: The instrument set the run feeds. The goldens are recorded with a
    /// never-configured (`Off`) one.
    pub telemetry: &'static metrale_telemetry::Telemetry,
    /// 2026-09-25: Fault injection for the pipelined lane (the negative controls).
    pub pipeline_faults: crate::scheduler::PipelineFaults,
}

/// 2026-09-25: Never configured: every feed into it takes the `Off` branch.
static TELEMETRY_OFF: metrale_telemetry::Telemetry =
    metrale_telemetry::Telemetry::new(&metrale_telemetry::clock::MonotonicClock);

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            use_speculative: false,
            dflash: false,
            num_drafts: 1,
            max_prefill_tokens: 0,
            max_batch_tokens: 8192,
            self_speculative: false,
            ngram_speculative: false,
            swap_space_gb: 0,
            slai_policy: false,
            mtp_gate_force: true,
            loop_watchdog: false,
            boundary_tokens: Vec::new(),
            think_end_token: None,
            think_start_token: None,
            lora_rotation: None,
            telemetry: &TELEMETRY_OFF,
            pipeline_faults: crate::scheduler::PipelineFaults::NONE,
        }
    }
}

#[derive(Clone)]
pub(super) struct Scenario {
    pub name: &'static str,
    pub cfg: ModelCfg,
    pub opts: RunOptions,
    pub reqs: Vec<ReqSpec>,
}

enum Sink {
    Blocking(tokio::sync::oneshot::Receiver<anyhow::Result<InferenceResponse>>),
    Streaming(std::thread::JoinHandle<Vec<String>>),
}

fn fmt_event(ev: &StreamEvent) -> String {
    match ev {
        StreamEvent::Token(t) => format!("T{t}"),
        StreamEvent::TokenWithLogprobs(t, lp) => format!("TL{t}:{}", lp.top.len()),
        StreamEvent::PromptLogprobs(v) => format!("PL{}", v.len()),
        StreamEvent::Done {
            finish_reason,
            completion_tokens,
            reasoning_tokens,
            cached_prompt_tokens,
            accepted_prediction_tokens,
            guard_stop,
            ..
        } => format!(
            "Done({finish_reason}, n={completion_tokens}, think={reasoning_tokens}, cached={cached_prompt_tokens}, acc={accepted_prediction_tokens}, guard={guard_stop:?})"
        ),
        StreamEvent::Error(m) => format!("Error({m})"),
    }
}

fn build_request(r: &ReqSpec, shared: &Shared) -> (InferenceRequest, Sink) {
    let prompt: Vec<u32> = (0..r.prompt_len)
        .map(|i| {
            if i == 0 {
                r.id as u32
            } else {
                2 + (i as u32 % 7)
            }
        })
        .collect();
    shared.add_script(
        r.id,
        SeqScript {
            prompt_len: r.prompt_len,
            tokens: r.toks.clone(),
            draft_wrong_every: r.draft_wrong_every,
            beam_hyp: r.beam_hyp.clone(),
            cancel_at: r.cancel_at,
        },
    );
    let timeout_at = r.expired_deadline.then(Instant::now);
    macro_rules! common {
        ($variant:ident, $($extra:tt)*) => {
            InferenceRequest::$variant {
                prompt_tokens: Arc::new(prompt),
                session_hash: r.id,
                adapter_slot: -1,
                src_lang_id: 0,
                tgt_lang_id: 0,
                num_beams: r.num_beams,
                length_penalty: 1.0,
                early_stopping: false,
                image_pixels: vec![],
                max_tokens: r.max_tokens,
                min_tokens: r.min_tokens,
                temperature: r.temperature,
                top_k: 0,
                top_p: 1.0,
                top_n_sigma: 0.0,
                min_p: 0.0,
                repetition_penalty: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                dry_multiplier: 0.0,
                dry_base: 0.0,
                dry_allowed_length: 0,
                lz_penalty: 0.0,
                logit_bias: vec![],
                stop_tokens: vec![],
                enable_thinking: false,
                thinking_budget: None,
                repetition_detection: None,
                require_tool_call: false,
                tools_present: false,
                suppress_tool_call: false,
                disable_mtp: false,
                grammar_spec: None,
                seed: Some(42),
                top_logprobs: r.top_logprobs,
                prompt_logprobs: None,
                echo: false,
                timeout_at,
                $($extra)*
            }
        };
    }
    if r.streaming {
        let (token_tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);
        let cancel_flag = Arc::new(AtomicBool::new(false));
        shared.register_cancel(r.id, Arc::clone(&cancel_flag));
        let collector = std::thread::spawn(move || {
            let mut out = Vec::new();
            while let Some(ev) = rx.blocking_recv() {
                out.push(fmt_event(&ev));
            }
            out
        });
        (
            common!(Streaming, token_tx, cancel_flag),
            Sink::Streaming(collector),
        )
    } else {
        let (response_tx, rx) = tokio::sync::oneshot::channel();
        (common!(Blocking, response_tx), Sink::Blocking(rx))
    }
}

/// 2026-09-25: Wait until the forwarder thread has pulled everything off the request
/// channel, then give it a moment to push into the scheduler's queue.
fn settle(tx: &tokio::sync::mpsc::Sender<InferenceRequest>) {
    while tx.capacity() < tx.max_capacity() {
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(20));
}

/// 2026-09-25: How a run builds its device router over the recording model. The default
/// is the sync router under the effect-trace decorator; the scripted tests
/// insert `ScriptedDeviceIo`, the pipeline tests the asynchronous router.
pub(super) type DeviceBuilder = Arc<
    dyn Fn(
            Arc<dyn metrale_model_engine::traits::Model>,
            metrale_scheduler::TraceSink,
        ) -> Box<crate::scheduler::io::DynDevice>
        + Send
        + Sync,
>;

/// 2026-09-25: The hard deadline for one scenario. A scenario that has not produced its
/// outputs by then is a hang (the loop parked on its inbox or a client
/// stream never closed), and the test fails with the scenario's name
/// instead of blocking the run.
const SCENARIO_DEADLINE: Duration = Duration::from_secs(120);

pub(super) fn run_scenario(sc: &Scenario) -> Vec<String> {
    run_scenario_with(
        sc,
        Arc::new(|model, sink| {
            Box::new(crate::scheduler::io::TracingDeviceIo::new(
                crate::scheduler::io::SyncDeviceIo::new(model),
                sink,
            ))
        }),
    )
}

/// 2026-09-25: `run_scenario` with the device router that `build` makes from the
/// recording model and the trace sink. The run happens on its own thread
/// under [`SCENARIO_DEADLINE`].
pub(super) fn run_scenario_with(sc: &Scenario, build: DeviceBuilder) -> Vec<String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let name = sc.name;
    let sc = sc.clone();
    std::thread::spawn(move || {
        let _ = tx.send(run_scenario_inner(&sc, build));
    });
    match rx.recv_timeout(SCENARIO_DEADLINE) {
        Ok(lines) => lines,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
            "scenario {name} exceeded {SCENARIO_DEADLINE:?}: the loop or a client stream hung"
        ),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("scenario {name} panicked before producing its outputs")
        }
    }
}

fn run_scenario_inner(sc: &Scenario, build: DeviceBuilder) -> Vec<String> {
    let model = RecordingModel::new(sc.cfg.clone());
    let shared = Arc::clone(&model.shared);
    let (request_tx, request_rx) = tokio::sync::mpsc::channel::<InferenceRequest>(1024);
    let (rotation_tx, rotation_rx) = tokio::sync::mpsc::channel(8);

    let mut sinks: Vec<(u64, Sink)> = Vec::new();
    let mut later: Vec<(usize, InferenceRequest)> = Vec::new();
    for r in &sc.reqs {
        let (req, sink) = build_request(r, &shared);
        sinks.push((r.id, sink));
        match r.arrive_at_tick {
            None => request_tx.blocking_send(req).expect("channel open"),
            Some(t) => later.push((t, req)),
        }
    }
    let rotation_ack = sc.opts.lora_rotation.as_ref().map(|name| {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        rotation_tx
            .blocking_send((LoraCommand::Rotate(name.clone()), ack_tx))
            .expect("rotation channel open");
        ack_rx
    });

    let opts = sc.opts.clone();
    let block_size = sc.cfg.block_size;
    let vocab = sc.cfg.vocab;
    let policy: Box<dyn SchedulingPolicy> = if opts.slai_policy {
        Box::new(SlaiPolicy::new(1_000_000))
    } else {
        Box::new(FifoPolicy)
    };
    let mut levers = SchedLevers::defaults();
    levers.mtp_gate_force = opts.mtp_gate_force;
    let levers = Arc::new(levers);
    levers.set_loop_watchdog(opts.loop_watchdog);
    let masks = VocabMasks {
        numeric: None,
        mid_word: None,
        boundary: (!opts.boundary_tokens.is_empty()).then(|| {
            let mut m = vec![false; vocab];
            for &t in &opts.boundary_tokens {
                m[t as usize] = true;
            }
            Arc::from(m.into_boxed_slice())
        }),
    };
    let snapshot = Arc::new(metrale_speculative::snapshot::SnapshotCell::default());
    let trace_shared = Arc::clone(&shared);
    let dev: Box<crate::scheduler::io::DynDevice> = build(
        Arc::new(model),
        Arc::new(move |line| trace_shared.rec(line)),
    );
    let handle = std::thread::spawn(move || {
        crate::scheduler::run(
            dev,
            request_rx,
            rotation_rx,
            crate::scheduler::config::SchedulerConfig {
                eos_tokens: vec![EOS],
                max_batch_size: opts.max_batch_size,
                use_speculative: opts.use_speculative,
                dflash_verify_raw_argmax: opts.dflash,
                // 2026-09-29: A146: the harness's scripted model is not GLM-5.3.
                mtp_spec_think_default: false,
                num_drafts: opts.num_drafts,
                policy,
                max_prefill_tokens: opts.max_prefill_tokens,
                max_batch_tokens: opts.max_batch_tokens,
                use_self_speculative: opts.self_speculative,
                use_ngram_speculative: opts.ngram_speculative,
                swap_space_gb: opts.swap_space_gb,
                high_speed_swap_cfg: None,
                block_size,
                think_end_token: opts.think_end_token,
                think_start_token: opts.think_start_token,
                code_fence_token: None,
                tool_call_start_token: None,
                tool_call_end_token: None,
                grammar_engine: None,
                adaptive_sampling: false,
                session_manager: crate::session_manager::SessionSsmManager::new(3600),
                spontaneous_think_budget: 512,
                vocab_masks: masks,
                limits: SchedLimits {
                    im_start_hard_stop: None,
                    tool_response_hard_stop: None,
                    max_seq_len: 4096,
                    code_fence_token: None,
                },
                watchdog: WatchdogParams::default(),
                levers,
                snapshot,
                dflash_rung: metrale_speculative::dflash_rung::DflashRung::new(),
                telemetry: opts.telemetry,
                pipeline_faults: opts.pipeline_faults,
            },
        );
    });

    later.sort_by_key(|(t, _)| *t);
    settle(&request_tx);
    if let Some((t, _)) = later.first() {
        shared.gate.block_at_tick(*t);
    }
    shared.gate.start();
    let i = 0;
    while i < later.len() {
        let t = later[i].0;
        shared.gate.wait_tick(t);
        while i < later.len() && later[i].0 == t {
            let (_, req) = later.remove(i);
            request_tx.blocking_send(req).expect("channel open");
        }
        settle(&request_tx);
        if i < later.len() {
            shared.gate.block_at_tick(later[i].0);
        }
        shared.gate.release();
    }
    // 2026-09-25: collect every client's output before dropping the
    // request channel: a sink closes when its sequence retires, so the
    // loop sees the channel close only once every request is done, and
    // its shutdown is the same in every run.
    let outputs: Vec<(u64, String)> = sinks
        .into_iter()
        .map(|(id, sink)| {
            let out = match sink {
                Sink::Streaming(h) => h.join().expect("collector").join(" "),
                Sink::Blocking(rx) => match rx.blocking_recv() {
                    Ok(Ok(r)) => format!(
                        "Response(tokens={:?}, {}, think={}, cached={}, acc={}, logprobs={})",
                        r.output_tokens,
                        r.finish_reason,
                        r.reasoning_tokens,
                        r.cached_prompt_tokens,
                        r.accepted_prediction_tokens,
                        r.logprobs.len()
                    ),
                    Ok(Err(e)) => format!("Err({e})"),
                    Err(_) => "dropped".to_string(),
                },
            };
            (id, out)
        })
        .collect();
    let rotation_result = rotation_ack.map(|rx| match rx.blocking_recv() {
        Ok(Ok(ack)) => format!("{ack:?}"),
        Ok(Err(e)) => format!("Err({e})"),
        Err(_) => "dropped".to_string(),
    });
    drop(request_tx);
    drop(rotation_tx);
    handle.join().expect("scheduler thread");

    let mut lines = vec![format!("# scenario {}", sc.name)];
    lines.extend(shared.take_trace());
    if let Some(r) = rotation_result {
        lines.push(format!("rotation ack: {r}"));
    }
    for (id, out) in outputs {
        lines.push(format!("out s{id}: {out}"));
    }
    lines
}
