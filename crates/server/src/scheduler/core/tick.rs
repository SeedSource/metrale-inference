// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The pre-step tick phases: drain, admission, settling a step that
//! runs ahead, the snapshot, quiescent LoRA rotations, the shutdown check and
//! the admission-time swap-out.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

impl SchedulerCore {
    pub(super) fn plan_tick(&mut self) -> TickPlan<Lane> {
        let Self {
            ctx: sched,
            pending,
            active,
            prefilling,
            policy,
            swapped,
            preempted,
            ..
        } = self;
        // 2026-10-15 (race-prefill, A192): an in-flight resume (`METRALE_RESUME_ONE_CHUNK_PER_TICK`)
        // holds one batch slot that `active` / `prefilling` do not show.
        let resume_in_flight = preempted.iter().any(|p| p.resume.is_some());
        let max_batch_size = self.max_batch_size - usize::from(resume_in_flight);
        let admit_watermark = self.admit_watermark;
        let block_size = self.block_size;
        // 2026-09-25: Each `t_loop` read opens one tick section, recorded as a `Loop*`
        // phase (`mtp_timing::Phase::LoopDrain` and the rest).
        let t_loop = sched.io.clock.now();
        let new_reqs = drain_pending_requests(
            &sched.io,
            pending,
            active,
            prefilling,
            &**policy,
            &sched.levers,
            max_batch_size,
            !(swapped.is_empty() && preempted.is_empty()),
        );
        // 2026-09-25: Hold back what does not fit the KV pool or the adapter cohort
        // (see `admission`).
        let new_reqs = admission::gate_admissions(
            sched,
            sched.io.dev.model(),
            pending,
            new_reqs,
            active,
            prefilling,
            swapped,
            preempted,
            admit_watermark,
            sched.limits.max_seq_len,
            block_size,
        );
        // 2026-10-03: `METRALE_PREFILL_CHUNK_WHILE_DECODING`: while a sequence decodes, start
        // at most one request this tick, and none when a prefill is in progress and the last
        // prefill turn started one (`last_turn_started`, A197) (`prefill_chunk_cap::admit_now`); the rest return to the front of
        // the pending queue in arrival order.
        let mut new_reqs = new_reqs;
        // 2026-10-14: `METRALE_DECODE_STEPS_PER_PREFILL_CHUNK`: a held tick (decode only, owed
        // after the last prefill chunk) admits nothing (`decode_slice`).
        self.slice_hold = crate::scheduler::decode_slice::hold(
            sched.levers.decode_steps_per_prefill_chunk,
            crate::scheduler::prefill_chunk_cap::decoders(active),
            self.slice_left,
        );
        // 2026-10-15 (race-prefill, A192): a resume re-prefill in flight
        // (`METRALE_RESUME_ONE_CHUNK_PER_TICK`) holds a sequence the batch-slot count does not
        // see, and runs only under KV pressure: start nothing until it finishes. With
        // `METRALE_RESUME_SLICED` new requests are admitted (the KV gate above already counts
        // the whole resume's demand, and the slot is taken off `max_batch_size`); the resume
        // and new prefills take turns at the one prefill chunk a decoding tick runs.
        let sliced = sched.levers.resume_sliced;
        let admit = if self.slice_hold || (resume_in_flight && !sliced) {
            0
        } else {
            crate::scheduler::prefill_chunk_cap::admit_now(
                sched.levers.prefill_chunk_while_decoding.is_some(),
                crate::scheduler::prefill_chunk_cap::decoders(active),
                new_reqs.len(),
                !prefilling.is_empty(),
                self.last_turn_started,
            )
        };
        if admit < new_reqs.len() {
            for (i, req) in new_reqs.drain(admit..).enumerate() {
                pending.requests.insert(i, req);
            }
        }
        // 2026-10-15 (race-prefill, A192): `METRALE_RESUME_SLICED`: whose turn this tick's
        // prefill chunk is. A held tick gives nobody a turn; with no decoder active the resume
        // runs as without the lever (to the end, at the end of the tick).
        self.prefill_ran = false;
        self.resume_slot = false;
        self.prefill_turn = false;
        if sliced
            && !self.slice_hold
            && crate::scheduler::prefill_chunk_cap::decoders(active) > 0
            && !preempted.is_empty()
        {
            let prefill_waiting = !new_reqs.is_empty() || !prefilling.is_empty();
            let slots_free = active.len() + prefilling.len() + new_reqs.len() < self.max_batch_size;
            let ready = crate::scheduler::preempt::resume_ready(
                sched.io.dev.model(),
                preempted,
                block_size,
                slots_free,
            );
            self.resume_slot = !prefill_waiting || (self.resume_turn && ready);
            self.prefill_turn = !self.resume_slot;
            if self.resume_slot {
                for (i, req) in new_reqs.drain(..).enumerate() {
                    pending.requests.insert(i, req);
                }
            }
        }
        // 2026-10-16 (race-memory, A194 gapF): a request whose chunk 0 was refused for KV waits
        // in `prefilling`. With the while-decoding cap's start/continue alternation, a start turn
        // retries it (new requests wait for the next start turn) and counts as a start, so the
        // next turn continues the long prefill (`prefill_chunk_cap::waiting_head`). With the cap
        // off, or nothing decoding, the order is unchanged.
        let mut waiting_retry = false;
        if sched.levers.prefill_chunk_while_decoding.is_some()
            && crate::scheduler::prefill_chunk_cap::decoders(active) > 0
            && !self.slice_hold
            && !self.resume_slot
        {
            let waiting: Vec<bool> = prefilling
                .iter()
                .map(|p| p.chunk0_refused && p.chunk_offset == 0)
                .collect();
            let retry_turn = !self.last_turn_started && waiting.iter().any(|&w| w);
            if retry_turn {
                for (i, req) in new_reqs.drain(..).enumerate() {
                    pending.requests.insert(i, req);
                }
                waiting_retry = true;
            }
            if let Some(i) = crate::scheduler::prefill_chunk_cap::waiting_head(&waiting, retry_turn)
            {
                let p = prefilling.remove(i);
                prefilling.insert(0, p);
            }
        }
        self.started_prefill = !new_reqs.is_empty();
        // 2026-10-16 (race-prefill, A197): only a prefill turn updates the alternation. Before
        // this, the k-1 held ticks of `METRALE_DECODE_STEPS_PER_PREFILL_CHUNK` and the resume's
        // turns reset it, so every free tick started a new request and ContinuePrefills skipped
        // (`skip_continue`): a steady stream of short requests starved a long prefill (gape E,
        // 2026-10-16: 648 s stall, cut by the server deadline).
        if !self.slice_hold && !self.resume_slot {
            self.last_turn_started = self.started_prefill || waiting_retry;
        }
        sched.io.tel.mark(mtp_timing::Phase::LoopDrain, t_loop);

        // 2026-09-25: Settle a step running ahead of the host before this tick starts
        // or continues a prefill, or may resume a parked sequence.
        if self.pipeline.has_inflight()
            && (!new_reqs.is_empty()
                || !self.prefilling.is_empty()
                || !self.swapped.is_empty()
                || !self.preempted.is_empty())
        {
            self.drain_pipeline();
        }
        let Self {
            ctx: sched,
            pending,
            active,
            prefilling,
            mtp_gate,
            session_manager,
            snapshot_steps,
            swapped,
            preempted,
            ..
        } = self;

        *snapshot_steps += 1;
        let t_loop = sched.io.clock.now();
        {
            let (mtp_mode, delivered_tps) = match mtp_gate.as_ref() {
                Some(g) => g.observe(),
                None => (metrale_speculative::snapshot::MtpModeSnap::Off, 0.0),
            };
            sched
                .io
                .tel
                .publish(metrale_speculative::snapshot::SchedulerSnapshot {
                    active_seqs: active.len() as u32,
                    prefilling_seqs: prefilling.len() as u32,
                    swapped_seqs: (swapped.len() + preempted.len()) as u32,
                    pending_len: new_reqs.len() as u32,
                    kv_blocks_free: sched.io.dev.model().num_free_blocks() as u32,
                    kv_blocks_total: sched.io.dev.model().num_total_blocks() as u32,
                    // 2026-09-25: The model's SSM snapshot pool occupancy.
                    // `SessionSsmManager::save_snapshot` has no production caller,
                    // so the session manager's counters are only a fallback for a
                    // model without a pool.
                    ssm_slots_used: sched
                        .io
                        .dev
                        .model()
                        .ssm_snapshot_occupancy()
                        .map(|(u, _)| u)
                        .unwrap_or(session_manager.session_count() as u32),
                    ssm_slots_total: sched
                        .io
                        .dev
                        .model()
                        .ssm_snapshot_occupancy()
                        .map(|(_, t)| t)
                        .unwrap_or(session_manager.total_slots() as u32),
                    mtp_mode,
                    delivered_tps,
                    steps_total: *snapshot_steps,
                    published_at: sched.io.clock.now(),
                });
        }
        sched.io.tel.mark(mtp_timing::Phase::LoopSnapshot, t_loop);

        // 2026-09-25: Apply queued LoRA rotations only when no sequence is active,
        // prefilling, newly drained, spilled or requeued; otherwise they stay
        // queued. A spilled sequence keeps the adapter slot and id it was
        // prefilled under (`SwappedSeq::adapter_slot`/`adapter_id`), which a
        // rotation could re-point before it resumes.
        if active.is_empty()
            && prefilling.is_empty()
            && new_reqs.is_empty()
            && swapped.is_empty()
            && preempted.is_empty()
        {
            let rotations = std::mem::take(&mut pending.rotations);
            for (cmd, ack) in rotations {
                let res = sched.io.dev.lora(cmd);
                sched.io.req.lora_ack(ack, res);
            }
        }
        if new_reqs.is_empty() && active.is_empty() && prefilling.is_empty() {
            // 2026-09-25: Idle tick: stop once the inbox has closed.
            pending.absorb(sched.io.req.recv(io::WaitPolicy::NoWait));
            if pending.closed {
                return TickPlan {
                    lanes: Vec::new(),
                    shutdown: true,
                };
            }
        }

        // 2026-09-25: Admission-time swap-out: for each new request, spill the active
        // sequence holding the most KV blocks (grammar-active ones excluded) until
        // the request's prompt fits or nothing more can be spilled.
        if let Some(spill) = sched.io.spill.as_deref() {
            for req in &new_reqs {
                let prompt_len = req.prompt_len();
                let blocks_needed = prompt_len / block_size + 1;
                // 2026-09-25: Reclaim prefix-cache blocks before spilling a live sequence.
                // `num_free_blocks()` does not count blocks the cache holds, so it
                // can read near zero while blocks are reclaimable (see
                // `Model::reclaim_prefix_blocks`).
                loop {
                    let free = sched.io.dev.model().num_free_blocks();
                    if free >= blocks_needed
                        || sched
                            .io
                            .dev
                            .model()
                            .reclaim_prefix_blocks(blocks_needed - free)
                            == 0
                    {
                        break;
                    }
                }
                while sched.io.dev.model().num_free_blocks() < blocks_needed && !active.is_empty() {
                    let victim_idx = active
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.grammar_state.is_none())
                        .max_by_key(|(_, a)| a.seq.block_table.len())
                        .map(|(i, _)| i);
                    let Some(victim_idx) = victim_idx else {
                        tracing::warn!("No swappable sequences (all grammar-active)");
                        break;
                    };
                    match swap_out_sequence(&sched.io, active, victim_idx, spill) {
                        Ok(s) => {
                            tracing::info!(
                                "Swap-out: evicted seq (seq_len={}, blocks={}) to disk",
                                s.seq_len,
                                s.num_blocks,
                            );
                            swapped.push(s);
                        }
                        Err(e) => {
                            tracing::error!("Swap-out failed: {e:#}");
                            break;
                        }
                    }
                }
            }
        }

        TickPlan {
            lanes: vec![
                Lane::StartPrefills(new_reqs),
                Lane::ContinuePrefills,
                Lane::Decode,
            ],
            shutdown: false,
        }
    }
}
