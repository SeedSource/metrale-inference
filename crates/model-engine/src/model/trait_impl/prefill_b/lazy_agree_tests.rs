// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Two-rank tests of the rank-agreed prefill admission in `lazy_agree`
//! (race-memory #79, A168). Each test gives rank 0 and rank 1 their own admission result,
//! gathers the votes with the real rooted-broadcast gather on two threads
//! (`snap_agree_tests::gather_two`), and asserts that both ranks reach the same verdict.
//!
//! Owner: model-engine prefill (memory admission).
//! Invariants: none beyond the types.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::lazy_buffer::{DEFAULT_GRANULE, MapBudget};

use super::lazy_agree::{agreed_outcome, refusing_ranks};
use super::snap_agree_tests::gather_two;

const WHAT: &str = "prefill chunk [499712, +8192) of 532195";

/// 2026-10-05: The error rank 1 raised in the A168 run (`LazyBuffer::map_physical`).
fn floor_refusal() -> anyhow::Error {
    anyhow!(
        "KV cache exhausted: device free memory 2798 MiB would fall below the 2800 MiB \
         lazy-map floor (METRALE_LAZY_MAP_FREE_FLOOR_MB) if 2.0 MiB more were mapped"
    )
}

/// 2026-10-05: Vote with the real gather, then each rank's verdict.
fn decide(local: [Result<()>; 2]) -> [Result<()>; 2] {
    let votes = [u32::from(local[0].is_ok()), u32::from(local[1].is_ok())];
    let [v0, v1] = gather_two(votes);
    assert_eq!(v0, v1, "both ranks see the same votes");
    let [l0, l1] = local;
    [agreed_outcome(l0, &v0, WHAT), agreed_outcome(l1, &v1, WHAT)]
}

fn assert_refused_everywhere(out: &[Result<()>; 2]) {
    for (rank, r) in out.iter().enumerate() {
        let e = format!("{:#}", r.as_ref().expect_err("every rank refuses"));
        // 2026-10-05: The scheduler's preemption and error paths key on this phrase.
        assert!(e.contains("KV cache exhausted"), "rank {rank}: {e}");
    }
}

#[test]
fn refusing_ranks_lists_every_non_one_vote() {
    assert_eq!(refusing_ranks(&[1, 1]), Vec::<usize>::new());
    assert_eq!(refusing_ranks(&[1, 0]), vec![1]);
    assert_eq!(refusing_ranks(&[0, 1, 0]), vec![0, 2]);
}

#[test]
fn both_ranks_admit_and_proceed() {
    let out = decide([Ok(()), Ok(())]);
    assert!(out.iter().all(|r| r.is_ok()));
}

// 2026-10-05: The A168 shape: rank 1 under the lazy-map floor, rank 0 with room. Before the
// fix rank 0 went on into the chunk's collectives and waited forever.
#[test]
fn floor_refusal_on_rank1_refuses_on_both_ranks() {
    let out = decide([Ok(()), Err(floor_refusal())]);
    assert_refused_everywhere(&out);
    let r0 = format!("{:#}", out[0].as_ref().unwrap_err());
    assert!(r0.contains("rank(s) [1]"), "{r0}");
    let r1 = format!("{:#}", out[1].as_ref().unwrap_err());
    assert!(
        r1.contains("2800 MiB lazy-map floor"),
        "keeps its own cause: {r1}"
    );
}

#[test]
fn refusal_on_rank0_refuses_on_both_ranks() {
    let out = decide([Err(floor_refusal()), Ok(())]);
    assert_refused_everywhere(&out);
}

#[test]
fn refusal_on_both_ranks_refuses_on_both_ranks() {
    let out = decide([Err(floor_refusal()), Err(floor_refusal())]);
    assert_refused_everywhere(&out);
}

#[test]
fn empty_vote_vector_is_a_refusal() {
    assert!(agreed_outcome(Ok(()), &[], WHAT).is_err());
}

// 2026-10-05: Single rank (`agree_admission` without the multi-rank protocol returns the
// local result as is); one vote of its own gives the same verdict.
#[test]
fn single_rank_verdict_is_the_local_result() {
    assert!(agreed_outcome(Ok(()), &[1], WHAT).is_ok());
    let e = format!(
        "{:#}",
        agreed_outcome(Err(floor_refusal()), &[0], WHAT).unwrap_err()
    );
    assert!(e.contains("2800 MiB lazy-map floor"), "{e}");
}

// 2026-10-05: The real lazy-map admission on two ranks whose indexer pools differ: rank 1's
// pool is one granule short, so its `ensure_mapped` refuses while rank 0's maps. After the
// vote both refuse. (The mock backend's eager lazy buffers run the same budget refusal as the
// CUDA VMM ones; the VMM floor check is per-rank in the same way.)
#[test]
fn real_lazy_buffers_with_uneven_room_refuse_together() {
    let g = DEFAULT_GRANULE;
    let gpu = MockGpuBackend::new();
    let local: Vec<Result<()>> = [4 * g, g]
        .into_iter()
        .map(|limit| {
            let budget = Arc::new(MapBudget::new("DSA indexer", limit));
            let buf = gpu.alloc_lazy(8 * g, Some(budget)).unwrap();
            let r = buf.ensure_mapped(2 * g);
            buf.release(&gpu).unwrap();
            r
        })
        .collect();
    assert!(local[0].is_ok(), "rank 0 has room");
    assert!(local[1].is_err(), "rank 1 is one granule short");
    let mut it = local.into_iter();
    let out = decide([it.next().unwrap(), it.next().unwrap()]);
    assert_refused_everywhere(&out);
}
