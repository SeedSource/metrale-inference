// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SchedulerCore`], the scheduler loop's state, and its [`Core`]
//! implementation, run by `metrale_scheduler::run`: `plan` (tick.rs), the
//! lanes (lanes.rs, lane_decode.rs, pipeline_lane.rs), `end_tick`
//! (end_tick.rs) and `finish` (finish.rs).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

mod end_tick;
mod finish;
mod lane_decode;
mod lanes;
pub(super) mod pipeline;
mod pipeline_lane;
mod telemetry;
mod tick;

pub use pipeline::PipelineFaults;

use super::*;
use metrale_cache::kv_spill::KvSpillManager;
use metrale_scheduler::{Core, LaneVerdict, TickPlan};

/// 2026-09-25: One lane of a tick, in execution order.
pub enum Lane {
    /// 2026-09-25: Start this tick's admitted requests (`start_new_requests`).
    StartPrefills(Vec<InferenceRequest>),
    /// 2026-09-25: Continue in-progress prefills; with nothing left active, try to
    /// resume requeued sequences.
    ContinuePrefills,
    /// 2026-09-25: One decode step (`decode_lane`).
    Decode,
}

pub struct SchedulerCore {
    ctx: crate::scheduler::sched_ctx::SchedCtx,
    eos_tokens: Vec<u32>,
    max_batch_size: usize,
    dflash_verify_raw_argmax: bool,
    /// 2026-09-29: A146: see `SchedulerConfig::mtp_spec_think_default`.
    mtp_spec_think_default: bool,
    num_drafts: usize,
    policy: Box<dyn crate::scheduling_policy::SchedulingPolicy>,
    max_prefill_tokens: usize,
    max_batch_tokens: usize,
    use_self_speculative: bool,
    use_ngram_speculative: bool,
    block_size: usize,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    grammar_engine: Option<GrammarEngine>,
    adaptive_sampling: bool,
    session_manager: crate::session_manager::SessionSsmManager,
    spontaneous_think_budget: u32,
    use_mtp: bool,
    chunked: bool,
    mtp_gate: Option<metrale_speculative::mtp_gate::MtpGate>,
    ngram_proposer: Option<NgramProposer>,
    spec_slot_cap: usize,
    always_mixed: bool,
    admit_watermark: usize,
    /// 2026-09-25: What the inbox has handed over and the loop has not yet admitted.
    pending: PendingQueue,
    prefill_stream: u64,
    prefill_event: u64,
    active: Vec<ActiveSeq>,
    prefilling: Vec<PrefillInProgress>,
    swapped: Vec<SwappedSeq>,
    preempted: Vec<PreemptedSeq>,
    snapshot_steps: u64,
    /// 2026-09-25: Set by the prefill-continuation lane, read by the decode lane.
    did_mixed_step: bool,
    /// 2026-10-03: This tick's StartPrefills lane starts at least one request (set in
    /// `plan_tick`); with `METRALE_PREFILL_CHUNK_WHILE_DECODING` on it decides the next tick's
    /// admissions and whether this tick's ContinuePrefills runs (`prefill_chunk_cap`).
    started_prefill: bool,
    /// 2026-09-25: The pipelined decode lane's steps still in flight.
    pipeline: pipeline::Pipeline,
}

impl SchedulerCore {
    /// 2026-09-25: Everything one run needs, built from the config and the routers.
    pub fn new(
        dev: Box<io::DynDevice>,
        request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
        rotation_rx: tokio::sync::mpsc::Receiver<LoraRotation>,
        cfg: config::SchedulerConfig,
    ) -> Self {
        let config::SchedulerConfig {
            eos_tokens,
            max_batch_size,
            use_speculative,
            dflash_verify_raw_argmax,
            mtp_spec_think_default,
            num_drafts,
            policy,
            max_prefill_tokens,
            max_batch_tokens,
            use_self_speculative,
            use_ngram_speculative,
            swap_space_gb,
            high_speed_swap_cfg,
            block_size,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            grammar_engine,
            adaptive_sampling,
            session_manager,
            spontaneous_think_budget,
            vocab_masks,
            limits,
            watchdog,
            levers,
            snapshot,
            dflash_rung,
            telemetry,
            pipeline_faults,
        } = cfg;
        let spill: Option<Box<dyn io::SpillIo>> = if swap_space_gb > 0 {
            let max_bytes = swap_space_gb as u64 * 1024 * 1024 * 1024;
            // 2026-09-25: Per-process directory: `KvSpillManager::new` deletes the
            // `swap_*` files already in its directory, so two processes sharing one
            // would delete each other's live spill files.
            let spill_dir =
                std::env::temp_dir().join(format!("metrale-swap-{}", std::process::id()));
            match KvSpillManager::new(spill_dir.clone(), max_bytes) {
                Ok(mgr) => {
                    tracing::info!("Swap space: {swap_space_gb} GB at {}", spill_dir.display());
                    Some(Box::new(io::FileSpill::new(mgr)))
                }
                Err(e) => {
                    tracing::error!("Failed to initialize swap space: {e:#}");
                    None
                }
            }
        } else {
            None
        };
        let sched = crate::scheduler::sched_ctx::SchedCtx::new(
            vocab_masks,
            levers,
            io::SchedIo::serving(
                dev,
                telemetry,
                snapshot,
                io::TokioRequestIo::new(request_rx, rotation_rx),
                spill,
            ),
            limits,
            watchdog,
            metrale_speculative::adaptive_rung::AdaptiveRung::from_env(),
            dflash_rung,
        );
        sched
            .io
            .dev
            .model()
            .bind_gpu_to_thread()
            .expect("Failed to bind CUDA context to scheduler thread");
        let use_mtp = use_speculative && sched.io.dev.model().has_proposer();
        let num_drafts = if use_mtp || use_self_speculative || use_ngram_speculative {
            num_drafts.max(1)
        } else {
            0
        };
        let chunked = max_prefill_tokens > 0;
        // 2026-09-25: The MTP runtime gate (`metrale_speculative::mtp_gate`), which
        // switches between MTP and plain decode by measured delivered throughput.
        // Armed whenever MTP is on, unless `--mtp-gate force` (or
        // `METRALE_MTP_GATE_FORCE`) disarms it.
        let mtp_gate = if use_mtp && !sched.levers.mtp_gate_force {
            Some(metrale_speculative::mtp_gate::MtpGate::new(num_drafts))
        } else {
            if use_mtp && sched.levers.mtp_gate_force {
                tracing::warn!(
                    "--mtp-gate force: MTP throughput gate DISARMED (diagnostic; \
                     verify runs even where the gate would measure it net-negative)"
                );
            }
            None
        };
        let ngram_proposer = if use_ngram_speculative {
            // 2026-09-25: `NgramProposer::new`'s argument is currently unused (min/max
            // match length are fixed inside the constructor); this is not a 4-gram
            // order.
            Some(NgramProposer::new(4))
        } else {
            None
        };
        tracing::info!(
            "Scheduler started (batched mode, max_batch={max_batch_size}, mtp={}, ngram={}, num_drafts={num_drafts}, policy={}, chunked_prefill={}, max_prefill_tokens={})",
            use_mtp,
            use_ngram_speculative,
            policy.name(),
            chunked,
            if chunked { max_prefill_tokens } else { 0 },
        );
        // 2026-09-25: Speculative dispatch requires every active slot to be below this
        // cap (`decode_lane`). For a model with SSM layers it is
        // `ssm_reserve::mtp_state_slots`, the slot count the MTP verify pools and
        // the preflight reserve are sized to; that equals `max_batch_size` when
        // `max_batch_size` is at most 32 or `METRALE_MTP_POOL_FULL_WIDTH` is set.
        // Other models use `max_batch_size`.
        let spec_slot_cap = if sched.io.dev.model().has_ssm_layers() {
            metrale_model_layers::ssm_reserve::mtp_state_slots(max_batch_size)
        } else {
            max_batch_size
        };
        if spec_slot_cap < max_batch_size {
            tracing::info!(
                "MTP verify pools cover {spec_slot_cap}/{max_batch_size} SSM slots — \
                 sequences on uncovered slots plain-decode until compaction moves them \
                 down (kill switch METRALE_MTP_POOL_FULL_WIDTH restores full width)"
            );
        }

        // 2026-09-25: `METRALE_HOLO_ALWAYS_MIXED` (off unless `1` or `true`): a prefill
        // that can fuse into the active decode takes a mixed step sized by the
        // policy's `prefill_slice_budget`, even when `should_prefill` says wait
        // (see `continue_in_progress_prefills`).
        let always_mixed = sched.levers.holo_always_mixed;
        if always_mixed {
            tracing::info!(
                "METRALE_HOLO_ALWAYS_MIXED=on: fused mixed step always-on (slice-budget)"
            );
        }

        // 2026-09-25: The KV admission watermark; see the `admission` module docs.
        let admit_watermark = admission::resolve_admit_watermark(sched.limits.max_seq_len);

        let pending = PendingQueue::new();

        let prefill_stream = sched
            .io
            .dev
            .model()
            .create_stream()
            .expect("Failed to create prefill CUDA stream");
        let prefill_event = sched
            .io
            .dev
            .model()
            .create_event()
            .expect("Failed to create prefill CUDA event");

        let active: Vec<ActiveSeq> = Vec::new();
        let prefilling: Vec<PrefillInProgress> = Vec::new();
        let swapped: Vec<SwappedSeq> = Vec::new();
        let preempted: Vec<PreemptedSeq> = Vec::new();
        install_high_speed_swap(sched.io.dev.model(), high_speed_swap_cfg);

        Self {
            ctx: sched,
            eos_tokens,
            max_batch_size,
            dflash_verify_raw_argmax,
            mtp_spec_think_default,
            num_drafts,
            policy,
            max_prefill_tokens,
            max_batch_tokens,
            use_self_speculative,
            use_ngram_speculative,
            block_size,
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            grammar_engine,
            adaptive_sampling,
            session_manager,
            spontaneous_think_budget,
            use_mtp,
            chunked,
            mtp_gate,
            ngram_proposer,
            spec_slot_cap,
            always_mixed,
            admit_watermark,
            pending,
            prefill_stream,
            prefill_event,
            active,
            prefilling,
            swapped,
            preempted,
            snapshot_steps: 0,
            did_mixed_step: false,
            started_prefill: false,
            pipeline: pipeline::Pipeline::new(pipeline_faults),
        }
    }

    /// 2026-09-25: The tokenizer ids the commit of a pipelined step needs.
    fn commit_tokens(&self) -> pipeline::CommitTokens {
        pipeline::CommitTokens {
            think_end_token: self.think_end_token,
            think_start_token: self.think_start_token,
            code_fence_token: self.code_fence_token,
            tool_call_start_token: self.tool_call_start_token,
            tool_call_end_token: self.tool_call_end_token,
            adaptive_sampling: self.adaptive_sampling,
        }
    }

    /// 2026-09-25: Settle the pipelined decode steps in flight (`pipeline::drain`).
    fn drain_pipeline(&mut self) {
        let toks = self.commit_tokens();
        pipeline::drain(&mut self.active, &mut self.pipeline, &toks, &self.ctx);
    }
}

impl Core for SchedulerCore {
    type Lane = Lane;

    fn plan(&mut self) -> TickPlan<Lane> {
        self.ctx.io.tel.step_begin();
        self.plan_tick()
    }

    fn execute_lane(&mut self, lane: Lane) -> LaneVerdict {
        self.execute_lane_measured(lane)
    }

    fn end_tick(&mut self) {
        self.end_tick_phases()
    }

    fn finish(self) {
        self.finish_run()
    }
}
