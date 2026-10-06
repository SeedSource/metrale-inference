// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Row ownership and the collectives of the sequence-parallel staged prefill
//! (`METRALE_GLM_PREFILL_SEQ_PARALLEL=1`, driver `steps/staged/sp.rs`): in every sub-chunk
//! (`sub_chunks` at the attention width) rank 0 owns the first `ceil(k / 2)` rows and rank 1
//! the rest. Splitting inside each sub-chunk, rather than giving each rank whole sub-chunks,
//! keeps every exchange two-way: a reduce-scatter then moves half an item each way at once,
//! so with the normed-row exchange it costs what the all-reduce did on a full-duplex link.
//!
//! 2026-10-04: Also the pass body both the driver and the two-rank gate run (`sp_pass`): the
//! owned-row fronts (`SpRows::front`), the normed-row swap, the per-call body and
//! reduce-scatter, and the owned-row backs (`SpRows::back`). With
//! `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1` the attention calls are the `rows_ffn` windows, not
//! the sub-chunks; ownership stays per sub-chunk (`SpPlan::spans` clips it to any item), so a
//! row has one owner in both passes and every layer.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A row's owner is a function of the row, the sub-chunk width and the chunk length alone,
//!   whatever item asks (`SpPlan::spans`), so the rank that ran a row's `hc_post` in one pass
//!   runs its `hc_pre` in the next.
//! - The two ranks' rows of any item tile it; `spans` lists them in row order and without
//!   empty spans, and both ranks compute both lists, so the k-th send of one rank meets the
//!   k-th receive of the other with the same size.

use anyhow::{Result, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::profile;
use crate::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_expand, glm_hc_post, glm_hc_post_mix,
    glm_hc_pre_part, glm_hc_pre_part_premixed, hc_head_mean, mix_hc,
};

/// 2026-10-01: A row range, `(first row, rows)`.
pub type Span = (usize, usize);

/// 2026-10-01: Send/recv pairs per NCCL group; longer exchanges run as several groups.
const MAX_PAIRS: usize = 32;

/// 2026-10-01: The ownership of a staged prefill chunk whose sub-chunks are `width` rows
/// (the last may be narrower).
/// 2026-10-04: `total` is the chunk's row count, so a span can be clipped to an item that
/// does not start or end on a sub-chunk boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpPlan {
    pub width: usize,
    pub total: usize,
}

impl SpPlan {
    /// 2026-10-01: The plan over `subs` (from `sub_chunks`, starting at row 0). `None` for an
    /// empty list.
    pub fn new(subs: &[(usize, usize)]) -> Option<Self> {
        let (first, last) = (subs.first()?, subs.last()?);
        Some(Self {
            width: first.1.max(1),
            total: last.0 + last.1,
        })
    }

    /// 2026-10-01: `rank`'s rows of sub-chunk `[t, t + k)`: rank 0 the first `ceil(k / 2)`.
    pub fn half(rank: usize, (t, k): Span) -> Span {
        let a = k.div_ceil(2);
        if rank == 0 { (t, a) } else { (t + a, k - a) }
    }

    /// 2026-10-01: `rank`'s non-empty spans of item `[t, t + k)`, in row order.
    /// 2026-10-04: Its half of every chunk sub-chunk (`sub_chunks(total, width)`) that meets
    /// the item, clipped to the item; rows past `total` belong to no one. For an item of whole
    /// sub-chunks (every item before the full-width lever) these are the halves of its
    /// sub-chunks, as before.
    pub fn spans(&self, rank: usize, (t, k): Span) -> Vec<Span> {
        let end = (t + k).min(self.total);
        let w = self.width.max(1);
        let mut out = Vec::new();
        let mut s = (t / w) * w;
        while s < end {
            let n = w.min(self.total - s);
            let (a, m) = Self::half(rank, (s, n));
            let (lo, hi) = (a.max(t), (a + m).min(end));
            if hi > lo {
                out.push((lo, hi - lo));
            }
            s += n;
        }
        out
    }
}

/// 2026-10-05: The chunks `SpPlan::new` cuts ownership by for a staged prefill of
/// `num_tokens` rows: the `rows` sub-chunks, or with `window` (the full-width arm and
/// `METRALE_GLM_PREFILL_SP_WINDOW_OWNER=1`) the `rows_ffn` windows, which are exactly the
/// full-width attention calls; the FFN windows (`ffn_windows`) are runs of whole sub-chunks of
/// at most `rows_ffn` rows from row 0, so each lies inside one window or, with a merged tail,
/// covers whole windows. Any plan is correct (`spans` clips ownership to any item); the window
/// plan only gives every call one span per rank.
pub fn owner_chunks(num_tokens: usize, rows: usize, rows_ffn: usize, window: bool) -> Vec<Span> {
    let w = if window { rows_ffn.max(rows) } else { rows };
    super::sub_chunks(num_tokens, w)
}

/// 2026-10-04: `rms_norm_vanilla` over `rows` contiguous `[hidden]` BF16 rows in one launch,
/// one block of `min(hidden, 1024)` threads per row (`token = blockIdx.x`; the blocks share
/// nothing and the kernel never reads `gridDim`): the launch `Glm5NextLayer::norm` issues
/// (it calls this), so a row's bytes do not depend on which rows share the launch.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    w: DevicePtr,
    out: DevicePtr,
    rows: usize,
    hidden: usize,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows as u32, 1, 1])
        .block([(hidden.min(1024)) as u32, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(out)
        .arg_u32(hidden as u32)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-10-04: What the owned-row launches need: the layer's mHC kernels and constants, its
/// RMSNorm kernel, and the forward's highway (`streams`, `post`, `comb`) and whole-chunk
/// normed buffer, each at the chunk's row 0 (highway slot `t` is chunk row `t`).
#[derive(Clone, Copy)]
pub struct SpRows<'a> {
    pub mhc: &'a Glm5NextMhcKernels,
    pub rms_norm: KernelHandle,
    pub hidden: usize,
    pub hc_mult: usize,
    pub sinkhorn_iters: u32,
    pub rms_eps: f32,
    pub hc_eps: f32,
    pub streams: DevicePtr,
    pub post: DevicePtr,
    pub comb: DevicePtr,
    pub normed: DevicePtr,
}

/// 2026-10-04: One mHC site of a pass: its weights and norm, whether the pass expands the
/// highway first (layer 0's attention pass) or collapses it last (the last layer's FFN pass),
/// whether the body's partial is all-reduced (`reduce`), and which profile buckets it uses.
#[derive(Clone, Copy)]
pub struct SpSite<'a> {
    pub weights: &'a Glm5NextMhcSiteWeights,
    pub norm: DevicePtr,
    pub expand: bool,
    pub last: bool,
    pub reduce: bool,
    pub attn: bool,
    /// 2026-10-06: `METRALE_GLM_MHC_POST_MIX`: the back of this pass also writes the mix of this
    /// site (the next pass's), at each row's chunk position in its `mix` scratch
    /// (`glm_hc_post_mix`).
    pub post_mix: Option<&'a Glm5NextMhcSiteWeights>,
    /// 2026-10-06: The front finds its mix rows already in `weights.mix` at the chunk positions
    /// (the previous pass's `post_mix`) and runs `hc_finish` only.
    pub premixed: bool,
}

impl SpRows<'_> {
    /// 2026-10-04: The front of rows `[t, t + k)`, part of a `call_rows`-row call: `hc_expand`
    /// when `site.expand`, `hc_pre` of the site into those `hidden` rows (the mix kernel the
    /// whole call takes, `glm_hc_pre_part`), then the norm into the same rows of `normed`:
    /// the launches `attn_half` / `ffn_half` issue for the call, restricted to these rows.
    pub fn front(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        (t, k): Span,
        call_rows: usize,
        site: &SpSite<'_>,
        stream: u64,
    ) -> Result<()> {
        let (h, hc) = (self.hidden, self.hc_mult);
        let streams = self.streams.offset(t * hc * h * 4);
        let post = self.post.offset(t * hc * 4);
        let comb = self.comb.offset(t * hc * hc * 4);
        let x = hidden.offset(t * h * 2);
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);
        let t_mhc = profile::start();
        if site.expand {
            glm_hc_expand(gpu, self.mhc.hc_expand, x, streams, kt, ht, hct, stream)?;
        }
        if site.premixed {
            // 2026-10-06: Rows `[t, t + k)` of the mix sit at rows `t..` of the scratch.
            let mut w = *site.weights;
            w.mix = w.mix.offset(t * mix_hc(hc) * 4);
            glm_hc_pre_part_premixed(
                gpu,
                self.mhc,
                streams,
                &w,
                x,
                post,
                comb,
                kt,
                ht,
                hct,
                self.sinkhorn_iters,
                self.hc_eps,
                stream,
            )?;
        } else {
            glm_hc_pre_part(
                gpu,
                self.mhc,
                streams,
                site.weights,
                x,
                post,
                comb,
                kt,
                call_rows as u32,
                ht,
                hct,
                self.sinkhorn_iters,
                self.rms_eps,
                self.hc_eps,
                stream,
            )?;
        }
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        let out = self.normed.offset(t * h * 2);
        rms_norm_rows(gpu, self.rms_norm, x, site.norm, out, k, h, self.rms_eps, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        Ok(())
    }

    /// 2026-10-04: The back of rows `[t, t + k)`: `hc_post` of the reduced partial in its
    /// `hidden` rows into the highway, then (`last`) `hc_head_mean` into the same rows.
    pub fn back(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        (t, k): Span,
        last: bool,
        post_mix: Option<&Glm5NextMhcSiteWeights>,
        stream: u64,
    ) -> Result<()> {
        let (h, hc) = (self.hidden, self.hc_mult);
        let streams = self.streams.offset(t * hc * h * 4);
        let post = self.post.offset(t * hc * 4);
        let comb = self.comb.offset(t * hc * hc * 4);
        let x = hidden.offset(t * h * 2);
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);
        let t_post = profile::start();
        if let Some(next) = post_mix {
            // 2026-10-06: The next site's mix rows go to rows `t..` of its scratch.
            let mix_out = next.mix.offset(t * mix_hc(hc) * 4);
            glm_hc_post_mix(
                gpu,
                self.mhc,
                x,
                streams,
                post,
                comb,
                streams,
                next.hc_fn,
                mix_out,
                kt,
                ht,
                hct,
                self.rms_eps,
                stream,
            )?;
        } else {
            let post_k = self.mhc.hc_post;
            glm_hc_post(gpu, post_k, x, streams, post, comb, streams, kt, ht, hct, stream)?;
        }
        if last {
            hc_head_mean(gpu, self.mhc.hc_head, streams, x, kt, ht, hct, stream)?;
        }
        profile::end(profile::MHC_POST, t_post, gpu, stream);
        Ok(())
    }
}

/// 2026-10-04: Whether `calls` tile `[0, total)` in order with no empty call.
pub fn calls_tile(calls: &[Span], total: usize) -> bool {
    let mut next = 0;
    for &(t, k) in calls {
        if t != next || k == 0 {
            return false;
        }
        next = t + k;
    }
    next == total
}

/// 2026-10-04: One pass (attention or FFN) of the sequence-parallel staged prefill over
/// `calls`, the attention calls or FFN windows `prefill_staged_run` issues for the chunk (they
/// must tile it):
/// 1. the front of this rank's rows of every call (`SpRows::front`, with the call's width);
/// 2. the normed-row swap (`swap_owned`), so `rows.normed` holds every row's normed input;
/// 3. per call `(t, k)`, `body((t, k), normed rows of the call)`: the mixer or MLP over all
///    `k` rows, returning where its partial's rows `0..k` are; then its reduce-scatter into
///    this rank's rows of `hidden` (`reduce_item`, a copy without `site.reduce`);
/// 4. the back of this rank's rows (`SpRows::back`).
///
/// Both ranks issue the same collectives in the same order: the calls, the plan and the flags
/// are the same on both.
#[allow(clippy::too_many_arguments)]
pub fn sp_pass(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    add_k: KernelHandle,
    plan: SpPlan,
    rows: &SpRows<'_>,
    hidden: DevicePtr,
    calls: &[Span],
    site: SpSite<'_>,
    mut body: impl FnMut(Span, DevicePtr) -> Result<DevicePtr>,
    stream: u64,
) -> Result<()> {
    ensure!(
        calls_tile(calls, plan.total),
        "sequence-parallel pass: calls {calls:?} do not tile the {}-row chunk",
        plan.total
    );
    let (rank, h) = (comm.rank(), rows.hidden);
    let rb = h * 2;
    let bucket = if site.attn {
        profile::REDUCE_ATTN
    } else {
        profile::REDUCE_MLP
    };
    for &c in calls {
        for s in plan.spans(rank, c) {
            rows.front(gpu, hidden, s, c.1, &site, stream)?;
        }
    }
    let t_x = profile::start();
    swap_owned(comm, plan, rows.normed, (0, plan.total), rb, stream)?;
    profile::end(bucket, t_x, gpu, stream);
    for &(t, k) in calls {
        let p = body((t, k), rows.normed.offset(t * rb))?;
        let t_x = profile::start();
        let reduce = site.reduce;
        reduce_item(gpu, add_k, comm, plan, p, hidden, (t, k), h, reduce, stream)?;
        profile::end(bucket, t_x, gpu, stream);
    }
    for &c in calls {
        for s in plan.spans(rank, c) {
            rows.back(gpu, hidden, s, site.last, site.post_mix, stream)?;
        }
    }
    Ok(())
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

// 2026-10-04: In `seq_parallel/tests.rs` (500-line cap).
#[cfg(test)]
mod tests;
