// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Row ownership for the sequence-parallel staged prefill
//! (`METRALE_GLM_PREFILL_SEQ_PARALLEL=1`, driver `steps/staged/sp.rs`): of a chunk's
//! sub-chunks, rank 0 owns the first `ceil(n / 2)` and rank 1 the rest, so the boundary is a
//! sub-chunk boundary and every owned launch keeps the width it has unsplit.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The two ranks' rows tile `[0, n)`, with rank 0's first.
//! - `split_item` gives the two parts of any row range; what one rank sends is what the
//!   other receives, so both post matching send/recv pairs (or both skip an empty one).

use anyhow::Result;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-01: A row range, `(first row, rows)`.
pub type Span = (usize, usize);

/// 2026-10-01: The ownership of one chunk of `n` rows: rank 0 owns `[0, split)`, rank 1
/// `[split, n)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpPlan {
    pub split: usize,
    pub n: usize,
}

impl SpPlan {
    /// 2026-10-01: The plan over `subs` (consecutive, as from `sub_chunks`): the boundary is
    /// the first row of sub-chunk `ceil(len / 2)`. `None` below two sub-chunks.
    pub fn new(subs: &[(usize, usize)]) -> Option<Self> {
        if subs.len() < 2 {
            return None;
        }
        let last = subs[subs.len() - 1];
        Some(Self {
            split: subs[subs.len().div_ceil(2)].0,
            n: last.0 + last.1,
        })
    }

    /// 2026-10-01: The rows `rank` (0 or 1) owns.
    pub fn owned(&self, rank: usize) -> Span {
        if rank == 0 {
            (0, self.split)
        } else {
            (self.split, self.n - self.split)
        }
    }

    /// 2026-10-01: Whether `rank` owns row `t`.
    pub fn owns(&self, rank: usize, t: usize) -> bool {
        (t < self.split) == (rank == 0)
    }

    /// 2026-10-01: The rows of `[t, t + k)` that `rank` owns, then those its peer owns.
    pub fn split_item(&self, rank: usize, t: usize, k: usize) -> (Span, Span) {
        let mid = self.split.clamp(t, t + k);
        let (lo, hi) = ((t, mid - t), (mid, t + k - mid));
        if rank == 0 { (lo, hi) } else { (hi, lo) }
    }
}

/// 2026-10-01: One grouped exchange with the peer of a two-rank `comm`: send `send.1` rows of
/// `row_bytes` at `send.0`, receive `recv.1` rows into `recv.0`, on `stream`. An empty side is
/// not posted (the peer's matching side is empty too); the group is closed even when posting
/// fails.
pub fn swap(
    comm: &dyn CommBackend,
    send: (DevicePtr, usize),
    recv: (DevicePtr, usize),
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    if send.1 == 0 && recv.1 == 0 {
        return Ok(());
    }
    let peer = 1 - comm.rank();
    comm.group_start()?;
    let mut posted = Ok(());
    if send.1 > 0 {
        posted = comm.send_to(send.0.0, send.1 * row_bytes, peer, stream);
    }
    if recv.1 > 0 {
        let bytes = recv.1 * row_bytes;
        posted = posted.and_then(|()| comm.recv_from(recv.0.0, bytes, peer, stream));
    }
    let closed = comm.group_end();
    posted.and(closed)
}

/// 2026-10-01: Swap the owned rows of a row-indexed buffer (`row_bytes` per row): send this
/// rank's rows, receive the peer's into their own rows.
pub fn swap_owned(
    comm: &dyn CommBackend,
    plan: SpPlan,
    base: DevicePtr,
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    let (own, peer) = (plan.owned(comm.rank()), plan.owned(1 - comm.rank()));
    let send = (base.offset(own.0 * row_bytes), own.1);
    let recv = (base.offset(peer.0 * row_bytes), peer.1);
    swap(comm, send, recv, row_bytes, stream)
}

/// 2026-10-01: The reduce-scatter of item `[t, t + k)` of `[*, hidden]` BF16 rows, whose
/// partial sits in rows `0..k` of `p`: the rows the peer owns are sent to it, the peer's
/// partial of this rank's rows lands in their rows of `out`, and `bf16_add_inplace` (`add_k`)
/// adds this rank's partial into it, `__hadd(peer, own)`, which equals the all-reduce's
/// `__hadd(own, peer)` because IEEE addition commutes. Without `reduce`, this rank's partial
/// rows are copied to `out`.
#[allow(clippy::too_many_arguments)]
pub fn reduce_item(
    gpu: &dyn GpuBackend,
    add_k: KernelHandle,
    comm: &dyn CommBackend,
    plan: SpPlan,
    p: DevicePtr,
    out: DevicePtr,
    (t, k): Span,
    hidden: usize,
    reduce: bool,
    stream: u64,
) -> Result<()> {
    let rb = hidden * 2;
    let (mine, theirs) = plan.split_item(comm.rank(), t, k);
    // 2026-10-01: `p` row `r - t` holds token `r`.
    let (own_p, own_out) = (p.offset((mine.0 - t) * rb), out.offset(mine.0 * rb));
    if !reduce {
        if mine.1 > 0 {
            gpu.copy_d2d_async(own_p, own_out, mine.1 * rb, stream)?;
        }
        return Ok(());
    }
    let send = (p.offset((theirs.0 - t) * rb), theirs.1);
    swap(comm, send, (own_out, mine.1), rb, stream)?;
    if mine.1 > 0 {
        anyhow::ensure!(
            add_k.0 != 0,
            "sequence-parallel reduce needs bf16_add_inplace"
        );
        let n = mine.1 * hidden;
        KernelLaunch::new(gpu, add_k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(own_out)
            .arg_ptr(own_p)
            .arg_i32(n as i32)
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5next_layer::{ffn_windows, sub_chunks};

    #[test]
    fn halves_tile_the_chunk_on_a_sub_chunk_boundary() {
        for (n, rows) in [(512, 256), (513, 256), (5400, 256), (8192, 256), (8191, 256)] {
            let subs = sub_chunks(n, rows);
            let p = SpPlan::new(&subs).unwrap();
            let (a, b) = (p.owned(0), p.owned(1));
            assert_eq!((a.0, a.0 + a.1, b.0, b.0 + b.1), (0, p.split, p.split, n));
            assert!(subs.iter().any(|s| s.0 == p.split), "n={n}");
            assert!(a.1 >= b.1, "rank 0 takes the larger half n={n}");
            for &(t, k) in &subs {
                assert_eq!(p.owns(0, t), p.owns(0, t + k - 1), "sub-chunk is whole n={n}");
                assert_ne!(p.owns(0, t), p.owns(1, t));
            }
        }
        assert_eq!(SpPlan::new(&sub_chunks(8192, 256)).unwrap().split, 4096);
        assert_eq!(SpPlan::new(&sub_chunks(5400, 256)).unwrap().split, 2816);
        assert_eq!(SpPlan::new(&sub_chunks(256, 256)), None);
    }

    #[test]
    fn item_parts_match_across_ranks() {
        let subs = sub_chunks(5400, 256);
        let p = SpPlan::new(&subs).unwrap();
        let wins = ffn_windows(&subs, 2048, false, |_| true);
        for &(t, k) in subs.iter().chain(&wins) {
            let (m0, t0) = p.split_item(0, t, k);
            let (m1, t1) = p.split_item(1, t, k);
            assert_eq!((m0, t0), (t1, m1), "what 0 sends 1 receives t={t}");
            assert_eq!(m0.1 + m1.1, k);
            for r in [m0, m1] {
                assert!(r.0 >= t && r.0 + r.1 <= t + k);
            }
            assert!(m0.1 == 0 || p.owns(0, m0.0));
            assert!(m1.1 == 0 || p.owns(1, m1.0));
        }
        // 2026-10-01: The window (2048, 2048) straddles the boundary 2816.
        assert_eq!(p.split_item(0, 2048, 2048), ((2048, 768), (2816, 1280)));
        assert_eq!(p.split_item(1, 2048, 2048), ((2816, 1280), (2048, 768)));
    }
}
