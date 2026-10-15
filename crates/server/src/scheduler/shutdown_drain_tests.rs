// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Proves the shutdown drain names which of the three parked states each abandoned request died in.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::io::{FileSpill, SchedIo};
use super::lifecycle::swap_out_sequence;
// 2026-09-25: PreemptStubModel, not lifecycle_tests::StubModel: the swap-out
// path calls `save_sequence_state`, which lifecycle_tests::StubModel leaves at
// the trait default that fails.
use super::shutdown_drain::abort_in_flight_on_shutdown;
use super::test_support::PreemptStubModel;
use super::test_support::{RespRx, test_prefill, test_seq};
use super::types::PreemptedSeq;
use metrale_cache::kv_spill::KvSpillManager;

fn message(label: &str, mut rx: RespRx) -> String {
    match rx.try_recv() {
        Ok(Err(e)) => format!("{e:#}"),
        Ok(Ok(_)) => panic!("{label}: an abandoned request must not report success"),
        Err(e) => panic!("{label}: the client was dropped instead of told ({e})"),
    }
}

#[test]
fn shutdown_tells_every_parked_request_which_state_it_died_in() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let dir = std::env::temp_dir().join(format!("metrale-shutdown-drain-{}", std::process::id()));
    let spill = FileSpill::new(KvSpillManager::new(dir, 8 * 1024 * 1024).expect("spill manager"));
    let io = SchedIo::for_test_with(model.clone());

    let (prefill, prefill_rx) = test_prefill(vec![1, 2, 3]);

    let (victim, swapped_rx) = test_seq(vec![4, 5], 6, None, 2);
    let mut active = vec![victim];
    let swapped = swap_out_sequence(&io, &mut active, 0, &spill).expect("swap-out writes");

    let (parked, preempted_rx) = test_seq(vec![7], 9, None, 1);
    let preempted = PreemptedSeq {
        a: parked,
        tokens: vec![7],
        resume: None,
    };

    abort_in_flight_on_shutdown(
        &*model,
        &io,
        vec![prefill],
        vec![swapped],
        vec![preempted],
        Some(&spill),
    );

    let m = message("prefilling", prefill_rx);
    assert!(m.contains("prefill"), "prefilling: got {m:?}");
    let m = message("swapped", swapped_rx);
    assert!(m.contains("swapped out"), "swapped: got {m:?}");
    let m = message("preempted", preempted_rx);
    assert!(m.contains("preempted"), "preempted: got {m:?}");
}
