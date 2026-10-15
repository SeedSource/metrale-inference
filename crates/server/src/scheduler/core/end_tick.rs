// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The end-of-tick phases: the deadline sweep, settling a step that
//! runs ahead, retirement, swap-in and the requeue resume.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

impl SchedulerCore {
    pub(super) fn end_tick_phases(&mut self) {
        let t_loop = self.ctx.io.clock.now();
        // 2026-09-25: Deadline sweep before retirement, so a timed-out sequence retires
        // in this same tick.
        enforce_request_deadlines(&self.ctx.io, &mut self.active);
        // 2026-09-25: A finishing row releases its slot and blocks at retirement; a step
        // still running ahead may be writing them, so it settles first (its
        // row for the finished sequence is the over-run it discards).
        if self.pipeline.has_inflight() && self.active.iter().any(|a| a.finished) {
            self.drain_pipeline();
        }
        let Self {
            ctx: sched,
            active,
            swapped,
            preempted,
            ..
        } = self;
        let max_batch_size = self.max_batch_size;
        let block_size = self.block_size;
        let max_prefill_tokens = self.max_prefill_tokens;
        let think_end_token = self.think_end_token;
        let think_start_token = self.think_start_token;
        retire_finished_sequences(&sched.io, active, sched.limits.max_seq_len);
        sched.io.tel.mark(mtp_timing::Phase::LoopRetire, t_loop);

        // 2026-09-25: Swap-in: while below `max_batch_size`, resume the first spilled
        // sequence that fits in the free blocks.
        let t_loop = sched.io.clock.now();
        if let Some(spill) = sched.io.spill.as_deref() {
            let mut resumed_any = true;
            while resumed_any && !swapped.is_empty() && active.len() < max_batch_size {
                resumed_any = false;
                let mut free = sched.io.dev.model().num_free_blocks();
                // 2026-09-25: Nothing fits: ask the prefix cache for blocks, which
                // `num_free_blocks()` does not count (see
                // `Model::reclaim_prefix_blocks`).
                if let Some(smallest) = swapped.iter().map(|s| s.num_blocks).min()
                    && smallest > free
                {
                    // 2026-09-25: A pass can free fewer blocks than asked; repeat until
                    // the smallest sequence fits or a pass frees nothing.
                    let was = free;
                    let mut total = 0usize;
                    while free < smallest {
                        let got = sched.io.dev.model().reclaim_prefix_blocks(smallest - free);
                        if got == 0 {
                            break;
                        }
                        total += got;
                        free = sched.io.dev.model().num_free_blocks();
                    }
                    if total > 0 {
                        tracing::info!(
                            "Swap-in: reclaimed {total} block(s) from the prefix cache for a \
                             {smallest}-block sequence (free {was} -> {free})",
                        );
                    }
                }
                if let Some(idx) = swapped.iter().position(|s| s.num_blocks <= free) {
                    let s = swapped.remove(idx);
                    match resume_swapped_seq(
                        think_end_token,
                        think_start_token,
                        sched.io.dev.model(),
                        &sched.io,
                        s,
                        spill,
                    ) {
                        Ok(a) => {
                            tracing::info!(
                                "Swap-in: restored seq (seq_len={}, blocks={})",
                                a.seq.seq_len,
                                a.seq.block_table.len(),
                            );
                            active.push(a);
                            resumed_any = true;
                        }
                        Err(e) => {
                            tracing::error!("Swap-in failed: {e:#}");
                        }
                    }
                }
            }
        }
        // 2026-09-25: Resume requeued (decode-preempted) sequences when blocks free.
        // 2026-10-15 (race-prefill, A192): `METRALE_RESUME_SLICED`: while a sequence decodes the
        // resume runs only on its turn (`plan_tick`), its chunk is capped like a prefill chunk,
        // the decode slice owes `k - 1` decode-only ticks after it, and new prefills (which it
        // no longer blocks) hold batch slots too.
        let sliced = sched.levers.resume_sliced;
        let decoders = crate::scheduler::prefill_chunk_cap::decoders(active);
        let resume_slot = self.resume_slot;
        if !sliced || decoders == 0 || resume_slot {
            let (slots, cap) = if sliced {
                (
                    max_batch_size.saturating_sub(self.prefilling.len()),
                    crate::scheduler::prefill_chunk_cap::active_cap(
                        sched.levers.prefill_chunk_while_decoding,
                        decoders,
                        sched.io.dev.model().kv_block_size(),
                    )
                    .filter(|_| !sched.io.dev.model().is_mla()),
                )
            } else {
                (max_batch_size, None)
            };
            let ran = preempt::resume_preempted_seqs(
                sched.io.dev.model(),
                &sched.io,
                active,
                preempted,
                slots,
                block_size,
                max_prefill_tokens,
                sched.levers.resume_one_chunk_per_tick,
                cap,
            );
            if sliced && ran && decoders > 0 {
                self.slice_left = crate::scheduler::decode_slice::next_left(
                    sched.levers.decode_steps_per_prefill_chunk,
                    decoders,
                    self.slice_left,
                    false,
                    true,
                );
            }
        }
        if sliced {
            if resume_slot {
                self.resume_turn = false;
            } else if self.prefill_ran && !preempted.is_empty() {
                self.resume_turn = true;
            }
        }
        sched.io.tel.mark(mtp_timing::Phase::LoopSwap, t_loop);
    }
}
