// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Scheduler-loop helpers: request draining, deadlines, retirement, slot compaction.
//!
//! Also `install_high_speed_swap`, called once when `core::SchedulerCore`
//! starts, and the response senders in `send`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use crate::scheduler::io::{SchedIo, WaitPolicy};
use metrale_model_engine::traits::Model;

use super::*;
use crate::api::InferenceRequest;
use crate::scheduler::levers::SchedLevers;
use crate::scheduling_policy::{ActiveSeqTiming, PendingRequestInfo, SchedulingPolicy};

/// 2026-09-25: Install the `--high-speed-swap` orchestrator on this thread when `cfg`
/// is set. `SchedulerCore::new` calls it after `bind_gpu_to_thread`. A failed
/// install, or a model without `high_speed_swap_dims`, is logged and ignored.
pub(super) fn install_high_speed_swap(
    model: &dyn Model,
    cfg: Option<metrale_storage::HighSpeedSwapConfig>,
) {
    let Some(cfg) = cfg else { return };
    match model.high_speed_swap_dims() {
        Some(dims) => {
            tracing::info!(
                "--high-speed-swap installing: dir={}, scratch={} blocks, qd={}, rank={}, \
                 model: {} layers × {}/{} (q/kv) heads × hd={}, bs={}, max_blocks={}",
                cfg.dir.display(),
                cfg.resident_blocks,
                cfg.qd,
                cfg.rank,
                dims.num_layers,
                dims.num_q_heads,
                dims.num_kv_heads,
                dims.head_dim,
                dims.block_size,
                dims.max_blocks_per_layer,
            );
            // 2026-09-25: Stream 0 (the default stream) for the orchestrator.
            if let Err(e) = metrale_storage::install_local(0, cfg, dims) {
                tracing::error!("--high-speed-swap install failed: {e:#}");
            } else {
                tracing::info!("--high-speed-swap orchestrator installed on scheduler thread");
                if std::env::var("METRALE_HIGH_SPEED_SWAP_REPLACE").is_ok() {
                    tracing::warn!(
                        "METRALE_HIGH_SPEED_SWAP_REPLACE=1: per-layer attention will route \
                         through HighSpeedSwap. UNTESTED on real models — requires real-load \
                         validation before production use."
                    );
                }
            }
        }
        None => {
            tracing::warn!(
                "--high-speed-swap requested but model does not expose high_speed_swap_dims; \
                 orchestrator NOT installed"
            );
        }
    }
}

/// 2026-09-25: Co-dispatch admission window: `Some(duration)` when
/// `SchedLevers::prefill_codispatch` is set, else `None`. The length is
/// `METRALE_PREFILL_CODISPATCH_WINDOW_MS` (100 when unset).
fn codispatch_window(levers: &SchedLevers) -> Option<std::time::Duration> {
    if !levers.prefill_codispatch {
        return None;
    }
    Some(std::time::Duration::from_millis(
        levers.codispatch_window_ms,
    ))
}

/// 2026-09-25: Quiet period that ends the co-dispatch window early
/// (`METRALE_PREFILL_CODISPATCH_SETTLE_MS`, 10 when unset). The wait stops
/// only after this long with no new request, so requests that arrive less than
/// this apart are collected together, up to the window's deadline.
fn codispatch_settle(levers: &SchedLevers) -> std::time::Duration {
    std::time::Duration::from_millis(levers.codispatch_settle_ms)
}

/// 2026-09-25: Drain pending request queue and policy-select prefills to start.
pub(super) fn drain_pending_requests(
    io: &SchedIo,
    pending: &mut PendingQueue,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    policy: &dyn SchedulingPolicy,
    levers: &SchedLevers,
    max_batch_size: usize,
    // 2026-09-25: True when swapped or preempted sequences wait to resume. They
    // wait on KV blocks, not on requests, and resume at the end of a tick
    // (`core/end_tick.rs`), so this call must not block indefinitely.
    have_parked: bool,
) -> Vec<InferenceRequest> {
    let g = pending;
    g.absorb(io.req.recv(WaitPolicy::NoWait));
    if active.is_empty() && prefilling.is_empty() && have_parked {
        // 2026-09-25: Only parked sequences: wait at most 10 ms, so the tick
        // reaches the resume pass without spinning.
        if g.is_idle() {
            g.absorb(
                io.req
                    .recv(WaitPolicy::Bounded(std::time::Duration::from_millis(10))),
            );
        }
    } else if active.is_empty() && prefilling.is_empty() {
        // 2026-09-25: Block until a request or a LoRA command arrives, or the
        // inbox closes (`PendingQueue::is_idle` covers all three). A LoRA
        // command alone must end the wait, or the tick's quiescent apply would
        // never run while the scheduler is idle.
        while g.is_idle() {
            g.absorb(io.req.recv(WaitPolicy::Block));
        }
        if g.closed && g.requests.is_empty() {
            return Vec::new();
        }
        // 2026-09-25: Woken by a LoRA command only: return no requests, so the
        // tick reaches its quiescent apply with nothing new admitted.
        if g.requests.is_empty() {
            return Vec::new();
        }
        // 2026-09-25: Co-dispatch window: when idle, keep collecting requests
        // (up to `max_batch_size`) so a concurrent burst is admitted in one
        // tick. A lone request can wait up to `window` longer for its first
        // token.
        if g.requests.len() < max_batch_size
            && let Some(window) = codispatch_window(levers)
        {
            // 2026-09-25: Wait in `settle`-sized slices until the deadline,
            // and stop early after one slice with no arrival.
            let deadline = io.clock.now() + window;
            let settle = window.min(codispatch_settle(levers));
            let mut seen = g.requests.len();
            while g.requests.len() < max_batch_size && !g.closed {
                let now = io.clock.now();
                if now >= deadline {
                    break;
                }
                let slice = (deadline - now).min(settle);
                let arrived = io.req.recv(WaitPolicy::Bounded(slice));
                let timed_out =
                    arrived.requests.is_empty() && arrived.rotations.is_empty() && !arrived.closed;
                g.absorb(arrived);
                if g.requests.len() > seen {
                    // 2026-09-25: Burst still landing — reset the quiet timer.
                    seen = g.requests.len();
                    continue;
                }
                if timed_out {
                    // 2026-09-25: Quiet for a full settle and nothing new: stop waiting.
                    break;
                }
            }
        }
    }

    // 2026-09-25: Ask policy whether to accept prefills this iteration.
    let timings: Vec<ActiveSeqTiming> = active
        .iter()
        .map(|a| ActiveSeqTiming {
            last_token_time: a.last_token_time,
        })
        .collect();

    if g.requests.is_empty() || !policy.should_prefill(io.clock.now(), &timings) {
        return Vec::new();
    }

    // 2026-09-25: Account for both active and in-progress prefilling sequences.
    let cap = max_batch_size.saturating_sub(active.len() + prefilling.len());

    let infos: Vec<PendingRequestInfo> = g
        .requests
        .iter()
        .enumerate()
        .map(|(i, req)| PendingRequestInfo {
            prompt_len: req.prompt_len(),
            index: i,
        })
        .collect();
    let selected = policy.select_prefills(&infos, cap);

    // 2026-09-25: Remove selected indices from pending (reverse order to preserve indices).
    let mut remove_indices = selected.clone();
    remove_indices.sort_unstable_by(|a, b| b.cmp(a));
    let mut taken: Vec<(usize, InferenceRequest)> = Vec::with_capacity(selected.len());
    for idx in remove_indices {
        taken.push((idx, g.requests.remove(idx)));
    }

    // 2026-09-25: Re-sort into policy-selected order.
    let mut result = Vec::with_capacity(selected.len());
    for &sel_idx in &selected {
        let pos = taken.iter().position(|(i, _)| *i == sel_idx).unwrap();
        let (_, req) = taken.swap_remove(pos);
        result.push(req);
    }
    result
}

/// 2026-09-25: Enforce the server-side per-request deadline on every active sequence.
///
/// `core/end_tick.rs` calls it once per tick, before retirement, so it covers
/// every decode path; the MTP path does not call
/// `decode_logits_step::process_decode_logits`.
///
/// A sequence past its deadline gets `guard_stop = GUARD_STOP_REQUEST_TIMEOUT`
/// and is marked finished, so `finish_sequence` reports
/// `finish_reason="timeout"` rather than "length", and it retires like any
/// other finished sequence.
pub(super) fn enforce_request_deadlines(io: &SchedIo, active: &mut [ActiveSeq]) {
    // 2026-09-25: No clock read when no unfinished sequence has a deadline.
    if !active.iter().any(|a| !a.finished && a.timeout_at.is_some()) {
        return;
    }
    let now = io.clock.now();
    for a in active.iter_mut() {
        if a.finished {
            continue;
        }
        let Some(deadline) = a.timeout_at else {
            continue;
        };
        if now < deadline {
            continue;
        }
        let emitted = a.output_tokens.len();
        tracing::warn!(
            slot = a.seq.slot_idx,
            session_hash = a.session_hash,
            elapsed_s = io
                .clock
                .now()
                .saturating_duration_since(a.request_start)
                .as_secs_f64(),
            budget_s = deadline
                .saturating_duration_since(a.request_start)
                .as_secs_f64(),
            emitted_tokens = emitted,
            requested_tokens = emitted + a.remaining,
            "Request TIMEOUT: response TRUNCATED by the server deadline \
             (--request-timeout / per-request `timeout`); \
             reporting finish_reason=\"timeout\", not \"length\""
        );
        a.guard_stop = Some(GUARD_STOP_REQUEST_TIMEOUT);
        a.finished = true;
    }
}

/// 2026-09-25: Finish and remove every finished sequence, in two phases (see the note
/// in the body), then move each survivor whose `slot_idx` is outside `[0..n)`
/// onto a free slot inside it (`compact_survivors_into_range`). A failed move
/// is logged and that survivor keeps its slot.
///
/// When the model uses `ep_protocol_v2`, finished sequences are removed with
/// no compaction, so every survivor keeps its slot.
pub(super) fn retire_finished_sequences(
    io: &SchedIo,
    active: &mut Vec<ActiveSeq>,
    max_seq_len: usize,
) {
    if io.dev.model().ep_protocol_v2() {
        // 2026-09-25: v2 EP: finish and remove, no compaction.
        let mut i = 0;
        let mut retired = false;
        while i < active.len() {
            if active[i].finished {
                let mut a = active.swap_remove(i);
                finish_sequence(io, &mut a, max_seq_len);
                retired = true;
            } else {
                i += 1;
            }
        }
        // 2026-10-10: `METRALE_MTP_POOL_CAP`: a retirement can free a slot the MTP verify pools
        // cover; move survivors on uncovered slots onto those slots, on both ranks, so the
        // batch can speculate again (`Model::migrate_to_covered_slot`, a no-op with the cap
        // off). The caller drained any step in flight before retiring.
        if retired {
            for a in active.iter_mut() {
                if let Err(e) = io.dev.model().migrate_to_covered_slot(&mut a.seq) {
                    tracing::error!("migrate_to_covered_slot (slot {}): {e:#}", a.seq.slot_idx);
                }
            }
        }
        return;
    }

    // 2026-09-25: Two phases, so compaction does not assume slot_idx equals the
    // position in `active`:
    //   Phase 1: finish every finished sequence (`finish_sequence` applies
    //            `Effect::ReleaseSeq`).
    //   Phase 2: move survivors outside [0..n) onto slots in [0..n) that no
    //            survivor holds, each target used once.

    // 2026-09-25: Phase 1.
    let mut survivors: Vec<ActiveSeq> = Vec::with_capacity(active.len());
    for mut a in active.drain(..) {
        if a.finished {
            finish_sequence(io, &mut a, max_seq_len);
        } else {
            survivors.push(a);
        }
    }

    // 2026-09-25: Phase 2.
    compact_survivors_into_range(io, &mut survivors);
    *active = survivors;
}

/// 2026-09-25: Move each sequence whose `slot_idx` is outside `[0..n)` (n = the slice
/// length) onto a slot in `[0..n)` that no sequence in the slice holds, via
/// `Effect::CompactSlot`. Each target is used once. When the slots in the
/// slice are distinct, there are exactly as many targets as sequences to
/// move. A failed move, or a missing target, is logged and the sequence keeps
/// its slot.
///
/// Precondition: the slots of retired sequences are already released, so they
/// can be targets. The only caller, `retire_finished_sequences`, does not call
/// it under `ep_protocol_v2()`.
pub(super) fn compact_survivors_into_range(io: &SchedIo, survivors: &mut [ActiveSeq]) {
    let n = survivors.len();
    let occupied: std::collections::HashSet<usize> =
        survivors.iter().map(|a| a.seq.slot_idx).collect();
    let mut free_targets: Vec<usize> = (0..n).filter(|s| !occupied.contains(s)).collect();
    for a in survivors.iter_mut() {
        if a.seq.slot_idx >= n {
            match free_targets.pop() {
                Some(target) => {
                    if let Err(e) = io.dev.apply(crate::scheduler::io::Effect::CompactSlot {
                        seq: &mut a.seq,
                        target,
                    }) {
                        tracing::error!("compact_sequence: {e}");
                    }
                }
                None => tracing::error!(
                    "compact_survivors_into_range: no free target for out-of-range \
                     slot {} (n={n})",
                    a.seq.slot_idx
                ),
            }
        }
    }
}

mod send;
pub use send::*;
