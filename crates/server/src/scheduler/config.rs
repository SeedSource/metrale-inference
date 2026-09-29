// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`SchedulerConfig`]: everything one scheduler run is given at
//! construction. Built by `serve_load` and by the trace harness.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::sync::Arc;

use crate::grammar::GrammarEngine;
use crate::scheduler::helpers::WatchdogParams;
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::limits::SchedLimits;
use crate::scheduler::vocab_masks::VocabMasks;
use crate::scheduling_policy::SchedulingPolicy;
use crate::session_manager::SessionSsmManager;

pub struct SchedulerConfig {
    pub eos_tokens: Vec<u32>,
    pub max_batch_size: usize,
    pub use_speculative: bool,
    pub dflash_verify_raw_argmax: bool,
    /// 2026-09-29: A146: this model's MTP-lane spec-in-think default
    /// (`mtp_gate::mtp_spec_think_default(config.model_type)`, resolved once
    /// at serve load); `METRALE_MTP_SPEC_THINK` / `METRALE_DFLASH_SPEC_THINK`
    /// override it through `levers`.
    pub mtp_spec_think_default: bool,
    pub num_drafts: usize,
    /// 2026-09-25: `--scheduler {fifo|slai}`: admission and ordering.
    pub policy: Box<dyn SchedulingPolicy>,
    pub max_prefill_tokens: usize,
    pub max_batch_tokens: usize,
    pub use_self_speculative: bool,
    pub use_ngram_speculative: bool,
    pub swap_space_gb: usize,
    pub high_speed_swap_cfg: Option<metrale_storage::HighSpeedSwapConfig>,
    pub block_size: usize,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub code_fence_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
    pub grammar_engine: Option<GrammarEngine>,
    pub adaptive_sampling: bool,
    pub session_manager: SessionSsmManager,
    pub spontaneous_think_budget: u32,
    /// 2026-09-25: Per-token masks for this model's vocabulary, indexed by token id.
    pub vocab_masks: VocabMasks,
    /// 2026-09-25: This model's hard stops: two tokenizer-resolved token ids and the
    /// served-context ceiling.
    pub limits: SchedLimits,
    /// 2026-09-25: This model's watchdog tunables (MODEL.toml `[behavior]` and the
    /// overrides that outrank it).
    pub watchdog: WatchdogParams,
    /// 2026-09-25: Shared with the dashboard, whose `/watchdog` command toggles the
    /// loop watchdog mid-run.
    pub levers: Arc<SchedLevers>,
    /// 2026-09-25: The scheduler snapshot cell, shared with the dashboard.
    pub snapshot: Arc<metrale_speculative::snapshot::SnapshotCell>,
    /// 2026-09-25: The DFlash gamma resolver, configured at serve time.
    pub dflash_rung: metrale_speculative::dflash_rung::DflashRung,
    /// 2026-09-25: The instrument set this run feeds (`metrale_telemetry::global()` when
    /// serving).
    pub telemetry: &'static metrale_telemetry::Telemetry,
    /// 2026-09-25: Fault injection for the pipelined decode lane; `PipelineFaults::NONE`
    /// when serving.
    pub pipeline_faults: super::PipelineFaults,
}
