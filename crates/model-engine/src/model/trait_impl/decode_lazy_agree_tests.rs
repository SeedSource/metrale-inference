// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Two-rank tests of the rank-agreed decode/verify lazy-map admission in
//! `decode_lazy_agree` (race-memory #79, A168 follow-up). Each rank runs on its own thread with
//! real lazily mapped GLM-5.3 DSA indexer states (`Glm5NextDsaState::alloc_lazy`, one target
//! layer plus a drafter state with the proposer look-ahead) charged to its own pool, so the
//! ranks' room can differ as the per-rank VMM floor makes it differ. The two ranks exchange
//! votes over a channel pair; a rank whose peer never votes fails after a timeout instead of
//! hanging, and every test checks that both ranks made the same number of gathers.
//!
//! Owner: model-engine (memory admission).
//! Invariants: none beyond the types.

use std::any::Any;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use anyhow::{Result, anyhow};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::lazy_buffer::{DEFAULT_GRANULE, MapBudget};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::lazy::PROPOSER_LOOKAHEAD_ROWS;
use metrale_model_arch::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::speculative::ProposerState;

use super::super::block_mgmt::map_lazy_states_through;
use super::decode_lazy_agree::{
    agree_lazy_rows_with, agreed_lazy_rows, lazy_rows_backed, lazy_vote,
};
use crate::traits::SequenceState;

const G: usize = DEFAULT_GRANULE;
/// 2026-10-05: Rows of 256 B (`index_head_dim` 128, BF16) in one 2 MiB granule.
const ROWS: usize = 8192;
const BS: usize = 16;
const CAP: usize = 65_536;
/// 2026-10-05: The agreed extent after the first vote: the drafter's one granule, net of its
/// look-ahead (the target layer's 8,192 rows are the larger).
const FIRST_AGREED: usize = ROWS - PROPOSER_LOOKAHEAD_ROWS;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: CAP,
    }
}

/// 2026-10-05: The MTP drafter's indexer cache as a proposer state, as
/// `Glm5NextMtpProposerState` delegates to its `dsa`.
struct Drafter(Glm5NextDsaState);

impl ProposerState for Drafter {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn map_rows_through(&self, end: usize) -> Result<()> {
        LayerState::map_rows_through(&self.0, end)
    }
    fn rows_backed_through(&self) -> Option<usize> {
        LayerState::rows_backed_through(&self.0)
    }
}

/// 2026-10-05: One rank's sequence: one lazy target layer and the lazy drafter state, both
/// charged to `pool`.
fn lazy_seq(gpu: &MockGpuBackend, pool: &Arc<MapBudget>) -> SequenceState {
    let mut s = SequenceState::host_only(3);
    s.layer_states.push(Box::new(
        Glm5NextDsaState::alloc_lazy(gpu, &cfg(), 0, pool.clone()).unwrap(),
    ));
    s.proposer_state = Some(Box::new(Drafter(
        Glm5NextDsaState::alloc_lazy(gpu, &cfg(), PROPOSER_LOOKAHEAD_ROWS, pool.clone()).unwrap(),
    )));
    s
}

/// 2026-10-05: One end of the two-rank vote exchange. `gather` returns `[rank 0, rank 1]`, as
/// `ep_gather_u32` does, and counts its calls.
struct Peer {
    rank: usize,
    tx: Sender<u32>,
    rx: Receiver<u32>,
    gathers: Cell<usize>,
}

impl Peer {
    fn gather(&self, v: u32) -> Result<Vec<u32>> {
        self.gathers.set(self.gathers.get() + 1);
        self.tx.send(v).map_err(|e| anyhow!("rank {}: send: {e}", self.rank))?;
        let other = self.rx.recv_timeout(Duration::from_secs(10)).map_err(|e| {
            anyhow!(
                "rank {}: the peer never voted ({e}): the ranks' gather counts differ",
                self.rank
            )
        })?;
        Ok(if self.rank == 0 {
            vec![v, other]
        } else {
            vec![other, v]
        })
    }
}

fn peers() -> [Peer; 2] {
    let (t01, r01) = channel();
    let (t10, r10) = channel();
    let mk = |rank, tx, rx| Peer {
        rank,
        tx,
        rx,
        gathers: Cell::new(0),
    };
    [mk(0, t01, r10), mk(1, t10, r01)]
}

/// 2026-10-05: Run `script(rank, peer)` for both ranks on two threads; each returns its
/// per-step outcomes and its gather count.
fn run_two<T: Send>(script: impl Fn(&Peer) -> T + Sync) -> [(T, usize); 2] {
    let [p0, p1] = peers();
    std::thread::scope(|s| {
        let run = |p: Peer| {
            let script = &script;
            s.spawn(move || {
                let out = script(&p);
                (out, p.gathers.get())
            })
        };
        let h0 = run(p0);
        let h1 = run(p1);
        [h0.join().unwrap(), h1.join().unwrap()]
    })
}

/// 2026-10-05: The admission of one decode step of `seq` at sequence length `seq_len` (last
/// block `seq_len / BS`), as `decode_dispatch_with` makes it.
fn decode_step(seq: &mut SequenceState, seq_len: usize, p: &Peer) -> Result<()> {
    agree_lazy_rows_with(seq, seq_len / BS, BS, |v| p.gather(v))
}

fn kv_exhausted(r: &Result<()>) -> bool {
    r.as_ref()
        .err()
        .is_some_and(|e| format!("{e:#}").contains("KV cache exhausted"))
}

#[test]
fn votes_and_agreed_extent_are_pure_functions_of_the_votes() {
    assert_eq!(lazy_vote(&Err(anyhow!("refused")), Some(5)), 0);
    assert_eq!(lazy_vote(&Ok(()), Some(7936)), 7936);
    assert_eq!(lazy_vote(&Ok(()), None), u32::MAX, "nothing lazy");
    assert_eq!(lazy_vote(&Ok(()), Some(usize::MAX)), u32::MAX, "fully backed");
    assert_eq!(lazy_vote(&Ok(()), Some(0)), 1, "an admitted vote is never 0");
    assert_eq!(agreed_lazy_rows(Ok(()), &[8192, 7936], "t").unwrap(), 7936);
    assert_eq!(
        agreed_lazy_rows(Ok(()), &[u32::MAX, u32::MAX], "t").unwrap(),
        usize::MAX
    );
    assert!(agreed_lazy_rows(Ok(()), &[8192, 0], "t").is_err());
    assert!(agreed_lazy_rows(Ok(()), &[], "t").is_err());
}

/// 2026-10-05: `rows_backed_through` of the real states: the target's mapped rows, the
/// drafter's net of its look-ahead, the minimum over both.
#[test]
fn backed_extent_is_the_minimum_over_target_and_drafter() {
    let gpu = MockGpuBackend::new();
    let pool = Arc::new(MapBudget::new("DSA indexer", usize::MAX));
    let seq = lazy_seq(&gpu, &pool);
    assert_eq!(lazy_rows_backed(&seq), Some(0));
    map_lazy_states_through(&seq, BS).unwrap();
    assert_eq!(seq.layer_states[0].rows_backed_through(), Some(ROWS));
    assert_eq!(lazy_rows_backed(&seq), Some(FIRST_AGREED));
    map_lazy_states_through(&seq, usize::MAX).unwrap();
    assert_eq!(lazy_rows_backed(&seq), Some(usize::MAX), "capacity backed");
    assert_eq!(
        lazy_rows_backed(&SequenceState::host_only(0)),
        None,
        "no lazy state"
    );
}

/// 2026-10-05: Steady state: after the first vote, steps inside the agreed extent make no
/// gather and map nothing; the step whose block crosses it votes again, on both ranks.
#[test]
fn only_boundary_steps_vote_and_both_ranks_vote_at_the_same_steps() {
    let out = run_two(|p| {
        let gpu = MockGpuBackend::new();
        let pool = Arc::new(MapBudget::new("DSA indexer", usize::MAX));
        let mut seq = lazy_seq(&gpu, &pool);
        let mut log = Vec::new();
        for seq_len in [100, 101, FIRST_AGREED - 1, FIRST_AGREED, 2 * ROWS] {
            let before = p.gathers.get();
            decode_step(&mut seq, seq_len, p).unwrap();
            log.push((seq_len, p.gathers.get() - before, seq.lazy_rows_agreed));
        }
        // 2026-10-05: The step's own `ensure_blocks_through_decode` then maps nothing.
        let used = pool.used();
        map_lazy_states_through(&seq, (2 * ROWS / BS + 1) * BS).unwrap();
        assert_eq!(pool.used(), used, "the step maps nothing after the vote");
        log
    });
    assert_eq!(out[0].0, out[1].0, "identical gathers and agreed extents");
    assert_eq!(out[0].1, out[1].1);
    let log = &out[0].0;
    assert_eq!(log[0], (100, 1, FIRST_AGREED), "first step after prefill votes");
    assert_eq!(log[1].1, 0, "steady state: no gather");
    // 2026-10-05: FIRST_AGREED - 1 = 7935 sits in block 495, whose end 7936 is still agreed.
    assert_eq!(log[2].1, 0);
    assert_eq!(log[3].1, 1, "block 496 ends past 7936: vote");
    assert_eq!(log[4].1, 1, "next granule: vote");
    assert_eq!(out[0].1, 3, "three gathers over five steps");
}

/// 2026-10-05: The A168 shape at decode time: rank 1's pool is one granule pair short, so its
/// drafter map refuses at the boundary while rank 0 maps. Both ranks refuse with the
/// "KV cache exhausted" phrase the scheduler preempts on, and neither moves its agreed extent.
#[test]
fn one_rank_refuses_at_the_boundary_so_both_refuse() {
    let out = run_two(|p| {
        let gpu = MockGpuBackend::new();
        // 2026-10-05: The first boundary maps 2 granules (target) + 2 (drafter).
        let limit = if p.rank == 1 { 3 * G } else { usize::MAX };
        let pool = Arc::new(MapBudget::new("DSA indexer", limit));
        let mut seq = lazy_seq(&gpu, &pool);
        let r = decode_step(&mut seq, 100, p);
        (kv_exhausted(&r), format!("{:#}", r.unwrap_err()), seq.lazy_rows_agreed)
    });
    assert_eq!(out[0].1, 1);
    assert_eq!(out[1].1, 1);
    for (rank, ((exhausted, msg, agreed), _)) in out.iter().enumerate() {
        assert!(*exhausted, "rank {rank}: {msg}");
        assert_eq!(*agreed, 0, "rank {rank}: agreed extent unchanged");
    }
    assert!(out[0].0.1.contains("rank(s) [1]"), "{}", out[0].0.1);
    assert!(out[1].0.1.contains("DSA indexer"), "keeps its own cause: {}", out[1].0.1);
}

/// 2026-10-05: After an agreed refusal the ranks' mapped extents differ (rank 0 mapped, rank 1
/// mapped the target only). The retry still votes on both ranks, because the predicate reads
/// the agreed extent, not the rank's own mapping; once the scheduler's victim is freed on
/// rank 1 the retry is admitted on both.
#[test]
fn retry_after_a_refusal_votes_on_both_ranks_and_admits_once_room_is_freed() {
    let out = run_two(|p| {
        let gpu = MockGpuBackend::new();
        let pool = Arc::new(MapBudget::new("DSA indexer", 4 * G));
        // 2026-10-05: A victim holding one granule pair, on rank 1 only (rank 0 had room).
        let mut victim = (p.rank == 1).then(|| {
            let v = Glm5NextDsaState::alloc_lazy(&gpu, &cfg(), 0, pool.clone()).unwrap();
            LayerState::map_rows_through(&v, 1).unwrap();
            v
        });
        let mut seq = lazy_seq(&gpu, &pool);
        let first = decode_step(&mut seq, 100, p);
        if let Some(v) = victim.as_mut() {
            v.free(&gpu).unwrap();
        }
        let retry = decode_step(&mut seq, 100, p);
        (kv_exhausted(&first), retry.is_ok(), seq.lazy_rows_agreed)
    });
    assert_eq!(out[0].1, 2, "rank 0 voted twice although it was mapped after the first");
    assert_eq!(out[1].1, 2);
    for (rank, ((refused, retried, agreed), _)) in out.iter().enumerate() {
        assert!(*refused, "rank {rank}: first attempt refused");
        assert!(*retried, "rank {rank}: retry admitted");
        assert_eq!(*agreed, FIRST_AGREED, "rank {rank}");
    }
}

/// 2026-10-05: A rank that mapped further on its own (the head-only `ReserveKv` path maps
/// through `map_lazy_rows_through` unvoted) votes at the same steps as the other rank, and the
/// agreed extent is the smaller rank's.
#[test]
fn a_rank_mapped_ahead_still_votes_at_the_same_steps() {
    let out = run_two(|p| {
        let gpu = MockGpuBackend::new();
        let pool = Arc::new(MapBudget::new("DSA indexer", usize::MAX));
        let mut seq = lazy_seq(&gpu, &pool);
        if p.rank == 0 {
            map_lazy_states_through(&seq, 3 * ROWS).unwrap();
        }
        decode_step(&mut seq, 100, p).unwrap();
        let first = seq.lazy_rows_agreed;
        decode_step(&mut seq, FIRST_AGREED, p).unwrap();
        (first, seq.lazy_rows_agreed)
    });
    assert_eq!(out[0], out[1]);
    assert_eq!(out[0].0.0, FIRST_AGREED, "the minimum over ranks");
    assert_eq!(out[0].1, 2);
    assert_eq!(out[0].0.1, ROWS, "the target layer's granule is the smaller extent now");
}

/// 2026-10-05: A batched step votes per sequence in batch order, so a refusal at the second
/// sequence stops both ranks at the same gather, and the first sequence's agreed extent moved
/// on both.
#[test]
fn batched_step_refusal_stops_both_ranks_at_the_same_sequence() {
    let out = run_two(|p| {
        let gpu = MockGpuBackend::new();
        // 2026-10-05: Room for one sequence's first boundary (4 granules) on rank 1.
        let limit = if p.rank == 1 { 4 * G } else { usize::MAX };
        let pool = Arc::new(MapBudget::new("DSA indexer", limit));
        let mut a = lazy_seq(&gpu, &pool);
        let mut b = lazy_seq(&gpu, &pool);
        let mut seqs: Vec<&mut SequenceState> = vec![&mut a, &mut b];
        let r = (|| -> Result<()> {
            for seq in seqs.iter_mut() {
                decode_step(seq, 100, p)?;
            }
            Ok(())
        })();
        (kv_exhausted(&r), a.lazy_rows_agreed, b.lazy_rows_agreed)
    });
    assert_eq!(out[0], out[1]);
    assert_eq!(out[0].1, 2);
    assert_eq!(out[0].0, (true, FIRST_AGREED, 0));
}

/// 2026-10-05: Without lazily mapped states (eager buffers, another model) the first vote
/// agrees on `usize::MAX`, and no later step gathers.
#[test]
fn eager_states_vote_once_then_never() {
    let out = run_two(|p| {
        let mut seq = SequenceState::host_only(0);
        for seq_len in [10, 10_000, 60_000] {
            decode_step(&mut seq, seq_len, p).unwrap();
        }
        seq.lazy_rows_agreed
    });
    assert_eq!(out[0], (usize::MAX, 1));
    assert_eq!(out[1], (usize::MAX, 1));
}
