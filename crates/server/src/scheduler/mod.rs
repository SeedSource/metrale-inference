// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The server's scheduler: [`run`] drives `core::SchedulerCore` on the calling thread.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

mod adaptive_spec;
mod admission;
mod beam_prefill;
mod confidence;
pub mod config;
mod core;
pub use core::PipelineFaults;
mod decode_launch;
mod decode_logits_content;
mod decode_logits_seq;
mod decode_logits_step;
mod decode_step;
#[cfg(test)]
mod emit_eos_thinking_tests;
mod emit_step;
#[cfg(test)]
mod emit_tool_call_finish_tests;
mod fast_greedy;
#[cfg(test)]
mod finish_guard_tests;
mod first_token_policy;
#[cfg(test)]
mod first_token_policy_tests;
mod helpers;
pub(crate) mod io;
mod lifecycle;
#[cfg(test)]
mod lifecycle_tests;
mod logit_dump;
mod logit_processors;
mod logprobs;
mod mod_helpers;
pub use mod_helpers::capture_runtime_handle;
pub mod dumps;
pub mod levers;
pub mod limits;
mod mtp_accept_debug;
mod mtp_bootstrap_step;
mod mtp_dcut;
mod mtp_step;
pub(crate) mod mtp_timing;
mod phase_continue_prefills;
mod phase_promote_prefills;
mod phase_start_prefills;
mod preempt;
#[cfg(test)]
mod preempt_tests;
mod prefill_a_step;
mod prefill_a_step_params;
mod prefill_b_step;
#[cfg(test)]
mod prefill_error_delivery_tests;
#[cfg(test)]
mod prefill_fifo_tests;
#[cfg(test)]
mod prefill_timing_tests;
mod repetition;
mod rollback;
mod sample_step;
pub mod sched_ctx;
mod shutdown_drain;
#[cfg(test)]
mod shutdown_drain_tests;
mod spec_step;
mod ssm_decode_ring;
#[cfg(test)]
mod swap_out_tests;
mod teardown;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod think_skip_tests;
#[cfg(test)]
mod trace_harness;
mod types;
mod verify_dflash_batch_step;
mod verify_dflash_step;
mod verify_k2_step;
mod verify_k3_step;
mod verify_k4_batch_step;
mod verify_k4_step;
mod verify_k4_verdict;
mod verify_pipeline_helper;
pub mod vocab_masks;

use beam_prefill::resolve_beam_hyp;
use confidence::*;
use decode_logits_content::*;
use decode_logits_seq::*;
use decode_logits_step::*;
use decode_step::*;
use emit_step::*;
use first_token_policy::*;
pub use helpers::WatchdogParams;
pub(crate) use helpers::parse_disable_watchdogs;
pub use helpers::resolve_content_loop_watchdog;
use helpers::*;
pub use helpers::{CONTENT_LOOP_PERIOD_MAX, CONTENT_LOOP_PERIOD_MIN};
use lifecycle::*;
use logprobs::*;
use mod_helpers::*;
use mtp_bootstrap_step::*;
use mtp_step::*;
use phase_continue_prefills::continue_in_progress_prefills;
use phase_start_prefills::start_new_requests;
use prefill_a_step::*;
use prefill_b_step::*;
use repetition::*;
use rollback::{RollbackOutcome, rollback_to_boundary};
use sample_step::*;
use spec_step::*;
use ssm_decode_ring::SsmDecodeRing;
use types::*;
use verify_dflash_batch_step::*;
use verify_dflash_step::*;
use verify_k2_step::*;
use verify_k3_step::*;
use verify_k4_batch_step::*;
use verify_k4_step::*;
use verify_k4_verdict::*;
// 2026-09-25: No `use` for verify_pipeline_helper: callers name it by its full
// path, `crate::scheduler::verify_pipeline_helper::...`.

// 2026-09-25: Imported for the submodules that `use super::*;`; `run` itself
// does not use all of them.
use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{Model, SequenceState};
use metrale_sampling::{
    SamplingParams, apply_penalties_and_bias, sample_with_params, sample_with_params_history,
};

use crate::api::{GrammarSpec, InferenceRequest, StreamEvent};
use crate::grammar::{GrammarEngine, GrammarState};
use metrale_speculative::ngram::NgramProposer;

/// 2026-09-25: A runtime LoRA adapter command. The scheduler tick applies queued
/// commands only when no sequence is active, prefilling, newly received,
/// swapped or preempted (`core/tick.rs`), so a command never overlaps a decode.
pub enum LoraCommand {
    /// 2026-09-25: Make the resident adapter with this name the active one
    /// (`Model::set_active_lora`).
    Rotate(String),
    /// 2026-09-25: Load the adapter at `dir` into pool `slot` and make it resident
    /// there (`Model::swap_lora_from_disk`).
    LoadIntoSlot {
        name: String,
        dir: std::path::PathBuf,
        slot: usize,
    },
    /// 2026-09-25: Copy the adapter staged on a peer into a cache pool slot and make
    /// it active (`Model::promote_lora_from_peer`). The ack returns the slot and
    /// any evicted adapter's name. `peft` supplies the adapter config the peer
    /// does not send.
    Promote {
        peer_addr: String,
        adapter_id: String,
        name: String,
        peft: metrale_config::PeftAdapterConfig,
    },
    /// 2026-09-25: Like [`Self::Promote`], but loads the adapter from `dir` on local
    /// disk (`Model::promote_lora_from_disk`). No `peft`: the adapter's own
    /// config is read from `dir`.
    PromoteDisk {
        name: String,
        dir: std::path::PathBuf,
    },
}

/// 2026-09-25: Successful result of a [`LoraCommand`]. `Rotate` and `LoadIntoSlot`
/// return [`LoraAck::Done`]; the promotes return the cache slot used and any
/// evicted adapter's name, which `AppState` uses to update its name-to-slot map.
#[derive(Debug, Clone)]
pub enum LoraAck {
    Done,
    Promoted {
        slot: usize,
        evicted: Option<String>,
    },
}

/// 2026-09-25: A LoRA command and the oneshot its sender awaits: `Ok(ack)` on
/// success, `Err(reason)` with the failure text otherwise.
pub type LoraRotation = (
    LoraCommand,
    tokio::sync::oneshot::Sender<Result<LoraAck, String>>,
);

/// 2026-09-25: Run the scheduler loop on the current thread.
pub fn run(
    dev: Box<io::DynDevice>,
    request_rx: tokio::sync::mpsc::Receiver<InferenceRequest>,
    rotation_rx: tokio::sync::mpsc::Receiver<LoraRotation>,
    cfg: config::SchedulerConfig,
) {
    metrale_scheduler::run(core::SchedulerCore::new(dev, request_rx, rotation_rx, cfg));
}
