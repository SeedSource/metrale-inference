// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Fails every request the scheduler is abandoning at shutdown, one at a time by parked state.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every prefilling, swapped and preempted request passed in is sent an error naming the state it was in.

use super::*;
use crate::scheduler::io::{SchedIo, SpillIo};

/// 2026-09-25: Fails every request the scheduler is abandoning as it exits, across the
/// three parked states: prefilling, swapped and preempted.
///
/// A dropped blocking sink reaches `chat_blocking` and `completions_exec` as a
/// closed channel, which they render as the generic "Inference cancelled".
/// Each request here instead gets a reason naming the state it was in.
pub(super) fn abort_in_flight_on_shutdown(
    model: &dyn Model,
    io: &SchedIo,
    prefilling: Vec<PrefillInProgress>,
    swapped: Vec<SwappedSeq>,
    preempted: Vec<PreemptedSeq>,
    spill: Option<&dyn SpillIo>,
) {
    for mut p in preempted {
        send_error_to_sink(
            io,
            &mut p.a.sink,
            "server shutting down before preempted resume",
        );
        // 2026-10-15 (race-prefill, A192): a re-prefill part-way through holds a sequence;
        // free it as a prefilling one is freed below.
        if let Some(mut r) = p.resume.take() {
            let seq = &mut r.seq;
            let _ = model.free_sequence(seq);
            let _ = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1);
        }
    }
    for mut p in prefilling {
        send_error_to_sink(io, &mut p.sink, "server shut down during prefill");
        let seq = &mut p.seq;
        let _ = model.free_sequence(seq);
        let _ = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1);
    }
    for mut s in swapped {
        send_error_to_sink(
            io,
            &mut s.sink,
            "server shut down while this request was swapped out to disk",
        );
        if let Some(spill) = spill {
            let _ = spill.remove(s.swap_id);
        }
    }
}
