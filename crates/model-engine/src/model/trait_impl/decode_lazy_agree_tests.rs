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

// ---------------------------------------------------------------------------------------------
// 2026-10-14 (race-memory, A183): rank-agreed KV block growth (`agree_kv_blocks_with`) on real
// paged KV pools (`PagedKvCache` on the mock GPU, no prefix cache) of different sizes per rank,
// grown by the real `ensure_blocks_through_decode`.

mod kv {
    use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_telemetry::prefix_cache::NoPrefixCaching;

    use super::super::super::block_mgmt::ensure_blocks_through_decode;
    use super::super::decode_lazy_agree::agree_kv_blocks_with;
    use super::{BS, Peer, kv_exhausted, run_two};
    use crate::traits::SequenceState;
    use anyhow::Result;

    fn pool(gpu: &MockGpuBackend, blocks: usize) -> PagedKvCache {
        let cfg = KvCacheConfig {
            block_size: BS,
            num_kv_heads: 1,
            head_dim: 64,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
            v_aliases_k: false,
        };
        PagedKvCache::new(cfg, blocks, gpu).unwrap()
    }

    /// 2026-10-14: One decode step of `seq` at `seq_len` (last block `seq_len / BS`) with the
    /// vote (`p = Some`) or without it (the 526615e1 behaviour: grow locally, no gather).
    fn step(
        seq: &mut SequenceState,
        seq_len: usize,
        kv: &mut PagedKvCache,
        gpu: &MockGpuBackend,
        fail_above: usize,
        p: Option<&Peer>,
    ) -> Result<()> {
        let last = seq_len / BS;
        let mut grow = |s: &mut SequenceState| {
            ensure_blocks_through_decode(s, last, kv, &NoPrefixCaching, gpu, 0, false)
        };
        match p {
            Some(p) => agree_kv_blocks_with(seq, last, fail_above, grow, |v| p.gather(v)),
            None => grow(seq),
        }
    }

    /// 2026-10-14: Decode one sequence until the first refusal on a pool of `blocks` blocks;
    /// returns (the seq_len of the first refusal or None, whether it was "KV cache exhausted",
    /// blocks held).
    fn decode_until_refused(
        blocks: usize,
        fail_above: usize,
        p: Option<&Peer>,
    ) -> (Option<usize>, bool, usize) {
        let gpu = MockGpuBackend::new();
        let mut kv = pool(&gpu, blocks);
        let mut seq = SequenceState::host_only(0);
        for seq_len in 0..(12 * BS) {
            let r = step(&mut seq, seq_len, &mut kv, &gpu, fail_above, p);
            if r.is_err() {
                return (Some(seq_len), kv_exhausted(&r), seq.block_table.len());
            }
        }
        (None, false, seq.block_table.len())
    }

    /// 2026-10-14: The A183 bug, reproduced on the 526615e1 path: pools of 6 and 5 blocks
    /// (hyb27c had 81,826 / 81,825), no vote. Rank 1 refuses the step that needs block 5 while
    /// rank 0 runs on: in a serve rank 0 then enters the step's collective alone.
    #[test]
    fn without_the_vote_one_block_of_pool_skew_diverges() {
        let r0 = decode_until_refused(6, 0, None);
        let r1 = decode_until_refused(5, 0, None);
        assert_eq!(r1, (Some(5 * BS), true, 5), "rank 1 refuses block 5");
        assert_eq!(
            r0.0,
            Some(6 * BS),
            "rank 0 refuses one block later: the ranks diverge"
        );
    }

    /// 2026-10-14: With the vote both ranks refuse the same step, both report "KV cache
    /// exhausted" (the scheduler's preemption trigger), and both made the same number of
    /// gathers: one per block of growth (the steps inside a block make none).
    #[test]
    fn with_the_vote_one_rank_refusal_is_agreed_on_both_ranks() {
        let out = run_two(|p| decode_until_refused(if p.rank == 0 { 6 } else { 5 }, 0, Some(p)));
        let [((at0, ex0, held0), g0), ((at1, ex1, held1), g1)] = out;
        assert_eq!(
            (at0, at1),
            (Some(5 * BS), Some(5 * BS)),
            "same refused step on both ranks"
        );
        assert!(ex0 && ex1, "both refusals say KV cache exhausted");
        assert_eq!(g0, g1, "same gather count on both ranks");
        assert_eq!(g0, 6, "one gather per block: blocks 0..=5");
        assert_eq!(
            (held0, held1),
            (6, 5),
            "rank 0 keeps its block 5; it is reused on the retry"
        );
    }

    /// 2026-10-14: After the refusal, room appears on the short rank (the scheduler's victim
    /// release, here a freed block): the same step is voted again and agreed, rank 0 does not
    /// allocate a second block 5, and the agreed extent moves past block 5.
    #[test]
    fn the_retry_after_room_appears_is_agreed_without_double_allocation() {
        let out = run_two(|p| {
            let gpu = MockGpuBackend::new();
            let mut kv = pool(&gpu, if p.rank == 0 { 7 } else { 6 });
            let victim = kv.alloc_block().unwrap(); // a second sequence's block
            let mut seq = SequenceState::host_only(0);
            let mut first_refused = None;
            for seq_len in 0..=(5 * BS) {
                if step(&mut seq, seq_len, &mut kv, &gpu, 0, Some(p)).is_err() {
                    first_refused = Some(seq_len);
                    break;
                }
            }
            kv.free_block(victim);
            let retry = step(&mut seq, 5 * BS, &mut kv, &gpu, 0, Some(p));
            (
                first_refused,
                retry.is_ok(),
                seq.block_table.len(),
                seq.kv_blocks_agreed,
            )
        });
        for ((first, retry_ok, held, agreed), _) in &out {
            assert_eq!(*first, Some(5 * BS));
            assert!(*retry_ok, "retry agreed");
            assert_eq!(*held, 6, "blocks 0..=5, no double allocation");
            assert_eq!(*agreed, 6);
        }
        assert_eq!(out[0].1, out[1].1, "same gather count");
    }

    /// 2026-10-14: The test fault-injection lever on ONE rank (the 2-rank GPU check's recipe):
    /// rank 1 refuses at block 3 although both pools are large; the refusal is agreed.
    #[test]
    fn fault_injection_on_one_rank_is_an_agreed_refusal() {
        let out = run_two(|p| decode_until_refused(64, if p.rank == 1 { 3 } else { 0 }, Some(p)));
        for ((at, exhausted, _), _) in &out {
            assert_eq!(*at, Some(3 * BS));
            assert!(*exhausted);
        }
        assert_eq!(out[0].1, out[1].1);
    }

    /// 2026-10-14: The startup cap: blocks at or above the agreed count leave the free list and
    /// are never handed out; capping above the pool retires nothing; a cap below an in-use block
    /// fails.
    #[test]
    fn cap_usable_blocks_retires_the_tail_only() {
        let gpu = MockGpuBackend::new();
        let mut kv = pool(&gpu, 6);
        let dummy = kv.alloc_block().unwrap();
        assert_eq!(dummy, 0, "the free list hands out block 0 first");
        assert_eq!(kv.cap_usable_blocks(9).unwrap(), 0);
        assert_eq!(kv.cap_usable_blocks(5).unwrap(), 1);
        assert_eq!((kv.num_blocks(), kv.num_free_blocks()), (5, 4));
        let got: Vec<u32> = (0..4).map(|_| kv.alloc_block().unwrap()).collect();
        assert!(got.iter().all(|&b| b < 5), "{got:?}");
        assert!(kv.alloc_block().is_err(), "exhausted at the agreed count");
        let mut kv2 = pool(&gpu, 6);
        for _ in 0..4 {
            kv2.alloc_block().unwrap();
        }
        assert!(kv2.cap_usable_blocks(2).is_err(), "block 2 is in use");
    }
}
