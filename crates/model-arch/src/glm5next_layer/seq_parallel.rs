// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Row ownership and the collectives of the sequence-parallel staged prefill
//! (`METRALE_GLM_PREFILL_SEQ_PARALLEL=1`, driver `steps/staged/sp.rs`): in every sub-chunk
//! (`sub_chunks` at the attention width) rank 0 owns the first `ceil(k / 2)` rows and rank 1
//! the rest. Splitting inside each sub-chunk, rather than giving each rank whole sub-chunks,
//! keeps every exchange two-way: a reduce-scatter then moves half an item each way at once,
//! so with the normed-row exchange it costs what the all-reduce did on a full-duplex link.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The two ranks' rows of any item (a sub-chunk or an FFN window, a run of whole
//!   sub-chunks) tile it; `spans` lists them in row order and without empty spans, and both
//!   ranks compute both lists, so the k-th send of one rank meets the k-th receive of the
//!   other with the same size.

use anyhow::Result;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::sub_chunks;

/// 2026-10-01: A row range, `(first row, rows)`.
pub type Span = (usize, usize);

/// 2026-10-01: Send/recv pairs per NCCL group; longer exchanges run as several groups.
const MAX_PAIRS: usize = 32;

/// 2026-10-01: The ownership of a staged prefill chunk whose sub-chunks are `width` rows
/// (the last may be narrower).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpPlan {
    pub width: usize,
}

impl SpPlan {
    /// 2026-10-01: The plan over `subs` (from `sub_chunks`, starting at row 0). `None` for an
    /// empty list.
    pub fn new(subs: &[(usize, usize)]) -> Option<Self> {
        subs.first().map(|s| Self { width: s.1 })
    }

    /// 2026-10-01: `rank`'s rows of sub-chunk `[t, t + k)`: rank 0 the first `ceil(k / 2)`.
    pub fn half(rank: usize, (t, k): Span) -> Span {
        let a = k.div_ceil(2);
        if rank == 0 { (t, a) } else { (t + a, k - a) }
    }

    /// 2026-10-01: `rank`'s non-empty spans of item `[t, t + k)` (which starts on a sub-chunk
    /// boundary and holds whole sub-chunks), in row order.
    pub fn spans(&self, rank: usize, (t, k): Span) -> Vec<Span> {
        sub_chunks(k, self.width)
            .into_iter()
            .map(|(s, n)| Self::half(rank, (t + s, n)))
            .filter(|s| s.1 > 0)
            .collect()
    }
}

/// 2026-10-01: Exchange with the peer of a two-rank `comm`: post a send of each
/// `(ptr, rows)` in `sends` and a receive of each in `recvs` (rows of `row_bytes`), in order,
/// at most `MAX_PAIRS` of each per NCCL group, on `stream`. Each group is closed even when
/// posting fails.
pub fn exchange(
    comm: &dyn CommBackend,
    sends: &[(DevicePtr, usize)],
    recvs: &[(DevicePtr, usize)],
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    let peer = 1 - comm.rank();
    for b in (0..sends.len().max(recvs.len())).step_by(MAX_PAIRS) {
        comm.group_start()?;
        let mut posted = Ok(());
        for &(p, rows) in sends.iter().skip(b).take(MAX_PAIRS) {
            let bytes = rows * row_bytes;
            posted = posted.and_then(|()| comm.send_to(p.0, bytes, peer, stream));
        }
        for &(p, rows) in recvs.iter().skip(b).take(MAX_PAIRS) {
            let bytes = rows * row_bytes;
            posted = posted.and_then(|()| comm.recv_from(p.0, bytes, peer, stream));
        }
        let closed = comm.group_end();
        posted.and(closed)?;
    }
    Ok(())
}

/// 2026-10-01: Swap the owned rows of item `item` in a row-indexed buffer (`row_bytes` per
/// row): send this rank's rows, receive the peer's into their own rows.
pub fn swap_owned(
    comm: &dyn CommBackend,
    plan: SpPlan,
    base: DevicePtr,
    item: Span,
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    let at = |s: Span| (base.offset(s.0 * row_bytes), s.1);
    let sends: Vec<_> = plan.spans(comm.rank(), item).into_iter().map(at).collect();
    let recvs: Vec<_> = plan.spans(1 - comm.rank(), item).into_iter().map(at).collect();
    exchange(comm, &sends, &recvs, row_bytes, stream)
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
    let mine = plan.spans(comm.rank(), (t, k));
    // 2026-10-01: `p` row `r - t` holds token `r`.
    let in_p = |s: Span| p.offset((s.0 - t) * rb);
    let in_out = |s: Span| out.offset(s.0 * rb);
    if !reduce {
        for &s in &mine {
            gpu.copy_d2d_async(in_p(s), in_out(s), s.1 * rb, stream)?;
        }
        return Ok(());
    }
    let theirs = plan.spans(1 - comm.rank(), (t, k));
    let sends: Vec<_> = theirs.iter().map(|&s| (in_p(s), s.1)).collect();
    let recvs: Vec<_> = mine.iter().map(|&s| (in_out(s), s.1)).collect();
    exchange(comm, &sends, &recvs, rb, stream)?;
    anyhow::ensure!(
        add_k.0 != 0,
        "sequence-parallel reduce needs bf16_add_inplace"
    );
    for &s in &mine {
        let n = s.1 * hidden;
        KernelLaunch::new(gpu, add_k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(in_out(s))
            .arg_ptr(in_p(s))
            .arg_i32(n as i32)
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5next_layer::ffn_windows;

    /// 2026-10-01: Rows of `spans`, expanded.
    fn rows(spans: &[Span]) -> Vec<usize> {
        spans.iter().flat_map(|&(t, n)| t..t + n).collect()
    }

    #[test]
    fn halves_tile_every_item_and_match_across_ranks() {
        let cases = [(5400, 256, 2048), (8192, 256, 4096), (8191, 256, 4096), (257, 256, 256)];
        for (n, w, ffn) in cases {
            let subs = sub_chunks(n, w);
            let plan = SpPlan::new(&subs).unwrap();
            let wins = ffn_windows(&subs, ffn, true, |_| true);
            for &(t, k) in subs.iter().chain(&wins).chain(&[(0, n)]) {
                let (a, b) = (plan.spans(0, (t, k)), plan.spans(1, (t, k)));
                let mut all = [rows(&a), rows(&b)].concat();
                all.sort_unstable();
                assert_eq!(all, (t..t + k).collect::<Vec<_>>(), "n={n} item ({t}, {k})");
                assert!(a.iter().chain(&b).all(|s| s.1 > 0));
                assert!(a.windows(2).all(|p| p[0].0 < p[1].0), "row order");
            }
            // 2026-10-01: Each rank's spans of the chunk are its spans of the sub-chunks.
            for r in 0..2 {
                let per_sub: Vec<Span> = subs.iter().flat_map(|&s| plan.spans(r, s)).collect();
                assert_eq!(plan.spans(r, (0, n)), per_sub);
            }
        }
    }

    #[test]
    fn halves_of_a_sub_chunk() {
        assert_eq!(SpPlan::half(0, (512, 256)), (512, 128));
        assert_eq!(SpPlan::half(1, (512, 256)), (640, 128));
        assert_eq!(SpPlan::half(0, (8192, 1)), (8192, 1));
        assert_eq!(SpPlan::half(1, (8192, 1)), (8193, 0));
        let plan = SpPlan::new(&sub_chunks(257, 256)).unwrap();
        assert_eq!(plan.spans(1, (0, 257)), vec![(128, 128)]);
        assert_eq!(SpPlan::new(&[]), None);
    }
}
