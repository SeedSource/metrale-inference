// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The plain decode lane one step ahead of the host: the core's half of
//! the asynchronous router.
//!
//! With step `t` in flight, the decode lane launches `t+1` as a fed step (each
//! row's input is `t`'s device argmax, read on the device), then settles `t`:
//! awaits it, commits it, and validates `t+1` against the host state that
//! commit produced (`validate_next`). A row that has finished, or whose next
//! input or position no longer matches, is an over-run: its `t+1` result is
//! skipped at commit, and a `Rollback` releases the block `ReserveKv` added for
//! it. If a surviving row's mask changed, all of `t+1` is redone. The launch
//! bookkeeping (`tokens.push`, `seq_len += 1`) is applied to a fed row only
//! once it is validated.
//!
//! Whether a tick may run ahead is decided per tick by `ahead_allowed`; the
//! router's `max_depth` only caps it.
//!
//! Owner: scheduler.
//! Invariants:
//! - With `PipelineFaults::NONE`, a fed row's result is committed only when its
//!   input and position equal the host state after the previous commit.

use std::time::Instant;

use super::*;
use metrale_scheduler::{FeedSource, RowMask, Ticket};

/// 2026-09-25: Fault injection for the pipeline's tests; serving passes
/// `PipelineFaults::NONE`. It lets the equivalence suite show that it fails
/// when the over-run discard is switched off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineFaults {
    /// 2026-09-25: Commit an over-run row's token as if its input had been valid.
    pub keep_overrun: bool,
}

impl PipelineFaults {
    pub const NONE: Self = Self {
        keep_overrun: false,
    };
}

/// 2026-09-25: A launched, uncommitted decode step.
pub(super) struct Inflight {
    pub(super) ticket: Ticket,
    pub(super) t0: Instant,
    /// 2026-09-25: Per row: the slot and the position the step writes.
    pub(super) rows: Vec<(usize, usize)>,
    pub(super) masks: Vec<RowMask>,
    /// 2026-09-25: What each row was fed, decided at launch; `None` for the pipeline's
    /// first step, whose inputs came from the host and are already booked.
    pub(super) inputs: Option<Vec<FeedSource>>,
    /// 2026-09-25: Rows whose result is an over-run (set by `validate_next`).
    pub(super) discard: Vec<bool>,
    /// 2026-09-25: The batch as launched no longer matches the state its commit needs
    /// (a mask changed): the whole step is redone.
    pub(super) discard_all: bool,
    /// 2026-09-25: The state its commit runs over needs the host path: the block is
    /// read back before the sampler runs.
    pub(super) host_logits: bool,
}

pub(super) struct Pipeline {
    /// 2026-09-25: The step to settle next.
    pub(super) inflight: Option<Inflight>,
    /// 2026-09-25: The step launched ahead of `inflight`, validated once that commits.
    pub(super) next: Option<Inflight>,
    pub(super) faults: PipelineFaults,
}

impl Pipeline {
    pub(super) fn new(faults: PipelineFaults) -> Self {
        Self {
            inflight: None,
            next: None,
            faults,
        }
    }
    pub(super) fn has_inflight(&self) -> bool {
        self.inflight.is_some()
    }
}

/// 2026-09-25: What the settle of an in-flight step leaves for the lane to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Settled {
    /// 2026-09-25: Committed; the lane is done with it.
    Done,
    /// 2026-09-25: The whole step was discarded: the lane must run a step now.
    Redo,
    /// 2026-09-25: The step failed and every row was told.
    Failed,
    /// 2026-09-25: Nothing was in flight.
    Idle,
}

/// 2026-09-25: The two ids the host would re-pick around for a row (`u32::MAX` = none).
pub(crate) fn row_mask(a: &ActiveSeq, think_end_token: Option<u32>) -> RowMask {
    if a.think_ended {
        [
            think_end_token.unwrap_or(u32::MAX),
            a.think_start_token.unwrap_or(u32::MAX),
        ]
    } else {
        [u32::MAX, u32::MAX]
    }
}

/// 2026-09-25: Whether the device's masked argmax (the row's argmax with the two
/// mask ids excluded, as `metrale_sampling::feed_argmax` computes it) is the
/// token the host pipeline would pick. For a `think_ended` row on the argmax
/// fast path, `argmax_readback_eligible` already requires temperature 0, no
/// grammar and neutral penalties; this checks the stages that remain.
fn masked_repick_exact(a: &ActiveSeq, adaptive_sampling: bool) -> bool {
    !a.think_ended
        || (a.logit_bias.is_empty()
            && !a.suppress_tool_call
            && !a.think_just_ended
            && !adaptive_sampling)
}

/// 2026-09-25: Whether this tick may launch a decode step ahead of the host.
pub(super) fn ahead_allowed(
    active: &[ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    adaptive_sampling: bool,
) -> bool {
    let model = sched.io.dev.model();
    let n = active.len();
    // 2026-09-25: The router's cap, and a model that can be fed at all.
    sched.io.dev.max_depth() >= 2
        && model.supports_device_token_feed()
        && n > 0
        // 2026-09-25: Models with SSM layers and EP models stay at depth 1.
        && !model.has_ssm_layers()
        && !model.is_ep()
        // 2026-09-25: Every row takes the device argmax, and its masked re-pick is exact.
        && super::super::decode_logits_step::argmax_readback_eligible(active.iter(), sched)
        && active
            .iter()
            .all(|a| masked_repick_exact(a, adaptive_sampling))
        // 2026-10-15: The device feed masks only the two `row_mask` ids, so it could
        // feed a stray `</tool_call>` the host's `StrayToolCallEndMask` replaces: a row
        // that mask applies to does not run ahead. Inert unless MODEL.toml sets
        // `mask_stray_tool_call_end` (GLM-5.3, whose model has no device token feed).
        && !active.iter().any(|a| {
            crate::scheduler::logit_processors::stray_tool_call_end::applies(
                sched.watchdog.mask_stray_tool_call_end,
                a,
            )
        })
        // 2026-09-25: A row that finishes by length next step would only over-run.
        && active.iter().all(|a| {
            !a.finished
                && a.remaining > 1
                && !seqlen_force_stop(a.seq.seq_len + 1, sched.limits.max_seq_len)
        })
}

impl SchedulerCore {
    /// 2026-09-25: Whether the plain lane may pipeline at all this run.
    pub(super) fn pipelining_configured(&self) -> bool {
        self.ctx.io.dev.max_depth() >= 2
            && !self.use_mtp
            && !self.use_ngram_speculative
            && !self.use_self_speculative
    }
}

/// 2026-09-25: The tokenizer ids and the sampling mode the commit needs.
#[derive(Clone, Copy)]
pub(super) struct CommitTokens {
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub code_fence_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
    pub adaptive_sampling: bool,
}

pub(super) use super::pipeline_lane::{drain, step_decode_pipelined};
