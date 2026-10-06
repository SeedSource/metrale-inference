// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: GPU byte-identity gate for `METRALE_GLM_PREFILL_SEQ_PARALLEL=1` as a whole
//! staged layer pass, including its composition with `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1`:
//! the production sequence-parallel pass (`glm5next_layer::seq_parallel::sp_pass`, the body
//! `steps/staged/sp.rs` runs, then the last layer's `swap_owned`) against the replicated
//! staged pass (`prefill_staged_run`'s launch sequence, transcribed below with the same
//! production launchers: `glm_hc_expand`, `glm_hc_pre`, `rms_norm_rows`, `glm_hc_post`,
//! `hc_head_mean`, and the all-reduce `all_reduce_async`), byte for byte.
//!
//! Two modes:
//! - No arguments (single GPU; what `build-mt.sh` runs): the ROW-LOCAL gate. For every case and
//!   every call of both passes it runs the replicated launches over the whole call, and the
//!   owned-row launches (`SpRows::front` / `SpRows::back`) over BOTH ranks' spans of it, on
//!   copies of the same inputs, and compares every output row (`y` in `hidden`, `post`,
//!   `comb`, the normed rows, the highway after `hc_post`, `hc_head_mean`). This is the
//!   M-invariance gate of the split row-local launches.
//! - `--rank 0|1 --peer <rank0 ip> [--port P]` (one process per node, two nodes): the row-local
//!   gate, then the LAYER gate on a real two-rank `NcclBackend`: two synthetic layers (layer 0
//!   expands, layer 1 collapses) through both arms. The mixer and MLP are stand-ins that are
//!   NOT row-local (each row's partial adds the normed row half a sub-chunk away inside the
//!   call, so a rank's partials read rows the peer normed) and rank-dependent (a seeded
//!   per-rank bias), so the all-reduce / reduce-scatter really sums two different partials.
//!   Gates: the final `hidden` (every row, after the last layer's swap) and this rank's rows of
//!   the highway (`streams`, `post`, `comb`) are bitwise equal across the arms; nothing is
//!   non-finite; the backend reports the send/recv + add all-reduce the lever requires.
//!
//! The real mixers and MLP are not run here: the lever leaves their launches, widths and
//! inputs unchanged (only the normed rows' address moves; see `sp.rs`), which the serve
//! identity run covers end to end.
//!
//! Cases (tokens, ROWS, ROWS_FFN, full width, tail merge): the 5400-token recipe, an odd last
//! window, a one-row tail, 8192-row windows with and without a one-row tail, ROWS 512 / 1024,
//! a window that is not a multiple of ROWS (a `max_batch_tokens` cap), a chunk shorter than
//! one sub-chunk, a one-row chunk, and the sliced (non-full-width) pass.
//!
//! Environment: `glm_hc_pre` refuses more rows than `mhc_mix_max_tokens()`, so when unset this
//! sets `METRALE_GLM_PREFILL_STAGED=1` and `METRALE_GLM_PREFILL_ROWS_FFN=8192` before anything
//! reads them, and `METRALE_GLM_MHC_TOKMAJOR=1` (the shipping mix kernel; run once more with
//! `=0` for the other). Set `SP_GATE_CASES=<n>` to run only the first n cases.
//! 2026-10-05: With `METRALE_GLM_PREFILL_SP_WINDOW_OWNER` (default on; `=0` off) the full-width
//! cases cut ownership per `rows_ffn` window (`seq_parallel::owner_chunks`), as the serve does under that lever.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS: ...`; 1 and a `FAIL` line; 2 when a kernel is absent from this target.
//!
//! Run (single GPU): `cargo run -p metrale-model-arch --release --features cuda,nccl-examples
//! --example glm5next_seq_parallel_layer_2rank`. Two ranks: the same with
//! `-- --rank 0 --peer <rank0-ip> --port 29564` on rank 0's node and `--rank 1` on the other
//! (`scripts/race/run-2rank-example.sh` in spark-bench). Check rank 0's NCCL log for `NET/IB`
//! before trusting the timing (`NET/Socket` = STOP); the bytes do not depend on the transport.

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_comm::{CommBackend, NcclBackend};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_layer::seq_parallel::{
    SpPlan, SpRows, SpSite, Span, owner_chunks, rms_norm_rows, sp_pass, swap_owned,
};
use metrale_model_arch::glm5next_layer::{ffn_windows, sub_chunks};
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_expand, glm_hc_post, glm_hc_pre,
    hc_head_mean, mhc_mix_max_tokens, mhc_tokmajor,
};

/// 2026-10-04: GLM-5.3-Flash geometry: hidden 4096, `hc_mult` 4 (`mix_hc` 24), 20 Sinkhorn
/// iterations, `hc_eps` 1e-6, RMSNorm eps 1e-5.
const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = (2 + HC) * HC;
const SINKHORN: u32 = 20;
const HC_EPS: f32 = 1e-6;
const EPS: f32 = 1e-5;
const POISON: u8 = 0xA5;

/// 2026-10-04: `(tokens, rows, rows_ffn, full width, tail merge)`.
type Case = (usize, usize, usize, bool, bool);
const CASES: [Case; 12] = [
    (5400, 256, 2048, true, true),
    (5401, 256, 2048, true, true),
    (8193, 256, 2048, true, true),
    (8192, 256, 8192, true, true),
    (8193, 256, 8192, true, false),
    (3001, 1024, 2048, true, true),
    (8192, 512, 2048, true, true),
    (8100, 256, 8000, true, true),
    (255, 256, 2048, true, true),
    (1, 256, 2048, true, true),
    (5401, 256, 2048, false, true),
    (2049, 256, 2048, false, false),
];

// 2026-10-01: CUDA driver event API for timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

fn timed(g: &dyn GpuBackend, stream: u64, f: impl FnOnce() -> Result<()>) -> Result<f32> {
    let (mut e0, mut e1) = (0u64, 0u64);
    let mut ms = 0f32;
    // 2026-10-04: SAFETY: plain driver calls on events this function creates and destroys,
    // on the caller's live stream.
    unsafe {
        if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
            bail!("cuEventCreate failed");
        }
        cuEventRecord(e0, stream);
    }
    f()?;
    // 2026-10-04: SAFETY: the two events created above, recorded on the same stream.
    unsafe {
        cuEventRecord(e1, stream);
        cuEventSynchronize(e1);
        cuEventElapsedTime(&mut ms, e0, e1);
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    g.synchronize(stream)?;
    Ok(ms)
}

/// 2026-10-04: Seeded uniform values in `[-a, a)`.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, a: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0) * a
    }
    fn bf16s(&mut self, n: usize, a: f32, center: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(center + self.next(a)).to_le_bytes())
            .collect()
    }
    fn f32s(&mut self, n: usize, a: f32) -> Vec<u8> {
        (0..n).flat_map(|_| self.next(a).to_le_bytes()).collect()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize, stream: u64) -> Result<Vec<u8>> {
    g.synchronize(stream)?;
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// 2026-10-04: Non-finite BF16 values among `b` (exponent all ones).
fn nonfinite_bf16(b: &[u8]) -> usize {
    b.chunks_exact(2)
        .filter(|c| u16::from_le_bytes([c[0], c[1]]) & 0x7F80 == 0x7F80)
        .count()
}

/// 2026-10-04: Bytes in rows `rows` (`row_bytes` each) where `a` and `b` differ.
fn diff_rows(a: &[u8], b: &[u8], rows: &[Span], row_bytes: usize) -> usize {
    rows.iter()
        .flat_map(|&(t, n)| t * row_bytes..(t + n) * row_bytes)
        .filter(|&i| a[i] != b[i])
        .count()
}

/// 2026-10-04: One mHC site's device weights (shared by both arms) and its norm weight.
struct Site {
    w: Glm5NextMhcSiteWeights,
    norm: DevicePtr,
}

fn site(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<Site> {
    // 2026-10-04: `hc_fn` scaled down to keep the mixes out of saturation, as in
    // `glm5next_hc_slice_microtest`; the byte comparison holds at any scale.
    let hc_fn = up(g, &rng.bf16s(MIX * HC * H, 0.02, 0.0))?;
    let scale: Vec<u8> = [0.7f32, 1.3, 0.9]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    Ok(Site {
        w: Glm5NextMhcSiteWeights {
            hc_fn,
            hc_fn_bf16: true,
            hc_scale: up(g, &scale)?,
            hc_base: up(g, &rng.f32s(MIX, 1.0))?,
            mix: g.alloc(mhc_mix_max_tokens() * MIX * 4)?,
        },
        norm: up(g, &rng.bf16s(H, 0.1, 1.0))?,
    })
}

/// 2026-10-04: One arm's buffers for an `n`-row chunk: `hidden`, the highway, the normed rows
/// (whole chunk) and the MLP-output scratch (one window).
struct Arm {
    hidden: DevicePtr,
    streams: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    normed: DevicePtr,
    ffn_out: DevicePtr,
}

impl Arm {
    fn new(g: &dyn GpuBackend, n: usize, window: usize) -> Result<Self> {
        let a = Self {
            hidden: g.alloc(n * H * 2)?,
            streams: g.alloc(n * HC * H * 4)?,
            post: g.alloc(n * HC * 4)?,
            comb: g.alloc(n * HC * HC * 4)?,
            normed: g.alloc(n * H * 2)?,
            ffn_out: g.alloc(window * H * 2)?,
        };
        for (p, b) in a.sized(n) {
            g.memset(p, POISON, b)?;
        }
        g.memset(a.ffn_out, POISON, window * H * 2)?;
        Ok(a)
    }
    /// 2026-10-04: `(buffer, bytes)` of the compared buffers: hidden, streams, post, comb,
    /// normed.
    fn sized(&self, n: usize) -> [(DevicePtr, usize); 5] {
        [
            (self.hidden, n * H * 2),
            (self.streams, n * HC * H * 4),
            (self.post, n * HC * 4),
            (self.comb, n * HC * HC * 4),
            (self.normed, n * H * 2),
        ]
    }
    fn rows<'a>(&self, k: &'a Glm5NextMhcKernels, norm: KernelHandle) -> SpRows<'a> {
        SpRows {
            mhc: k,
            rms_norm: norm,
            hidden: H,
            hc_mult: HC,
            sinkhorn_iters: SINKHORN,
            rms_eps: EPS,
            hc_eps: HC_EPS,
            streams: self.streams,
            post: self.post,
            comb: self.comb,
            normed: self.normed,
        }
    }
    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.hidden,
            self.streams,
            self.post,
            self.comb,
            self.normed,
            self.ffn_out,
        ] {
            g.free(p)?;
        }
        Ok(())
    }
}

/// 2026-10-04: The replicated front of call `(t, k)` (`attn_half_inner` / `ffn_half_inner` up
/// to the norm): `hc_expand`, `hc_pre`, then the norm into `normed` at `normed_at`.
#[allow(clippy::too_many_arguments)]
fn ref_front(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    norm_k: KernelHandle,
    a: &Arm,
    s: &Site,
    (t, n): Span,
    expand: bool,
    normed_at: DevicePtr,
    stream: u64,
) -> Result<()> {
    let x = a.hidden.offset(t * H * 2);
    let streams = a.streams.offset(t * HC * H * 4);
    let (nt, ht, hct) = (n as u32, H as u32, HC as u32);
    if expand {
        glm_hc_expand(g, k.hc_expand, x, streams, nt, ht, hct, stream)?;
    }
    let (post, comb) = (a.post.offset(t * HC * 4), a.comb.offset(t * HC * HC * 4));
    glm_hc_pre(
        g, k, streams, &s.w, x, post, comb, nt, ht, hct, SINKHORN, EPS, HC_EPS, stream,
    )?;
    rms_norm_rows(g, norm_k, x, s.norm, normed_at, n, H, EPS, stream)
}

/// 2026-10-04: The replicated back of call `(t, k)`: `hc_post` of `out` (rows `0..k`), then
/// (`last`) `hc_head_mean` into the call's `hidden` rows.
fn ref_back(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    a: &Arm,
    (t, n): Span,
    out: DevicePtr,
    last: bool,
    stream: u64,
) -> Result<()> {
    let streams = a.streams.offset(t * HC * H * 4);
    let (post, comb) = (a.post.offset(t * HC * 4), a.comb.offset(t * HC * HC * 4));
    let (nt, ht, hct) = (n as u32, H as u32, HC as u32);
    glm_hc_post(
        g, k.hc_post, out, streams, post, comb, streams, nt, ht, hct, stream,
    )?;
    if last {
        let x = a.hidden.offset(t * H * 2);
        hc_head_mean(g, k.hc_head, streams, x, nt, ht, hct, stream)?;
    }
    Ok(())
}

/// 2026-10-04: The attention calls and FFN windows `prefill_staged_run` issues for a case
/// (every sub-chunk mergeable, as for a dense MLP or a grouped-GEMM MoE).
fn calls(&(n, rows, ffn, wide, merge): &Case) -> (Vec<Span>, Vec<Span>) {
    let subs = sub_chunks(n, rows);
    let wide = wide && ffn > rows;
    let attn = if wide {
        sub_chunks(n, ffn)
    } else {
        subs.clone()
    };
    (attn, ffn_windows(&subs, ffn, merge, |_| true))
}

/// 2026-10-05: The ownership chunks of case `c`, as `staged.rs` cuts them: per window under
/// the full-width arm with `METRALE_GLM_PREFILL_SP_WINDOW_OWNER=1`, else per sub-chunk (the
/// lever also needs `METRALE_GLM_PREFILL_SEQ_PARALLEL`, so this reads the variable itself).
fn owners(&(n, rows, ffn, wide, _): &Case) -> Vec<Span> {
    let window = std::env::var("METRALE_GLM_PREFILL_SP_WINDOW_OWNER").as_deref() != Ok("0");
    owner_chunks(n, rows, ffn, wide && window)
}

/// 2026-10-04: The row-local gate (module doc) for one case; returns the differing bytes.
fn row_local_case(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    norm_k: KernelHandle,
    c: &Case,
    stream: u64,
) -> Result<usize> {
    let (n, rows) = (c.0, c.1);
    let plan = SpPlan::new(&owners(c)).context("no sub-chunks")?;
    let (attn_calls, ffn_wins) = calls(c);
    let mut rng = Lcg(0x5EED_5000 ^ (n as u64) << 20 ^ rows as u64);
    let (attn, ffn) = (site(g, &mut rng)?, site(g, &mut rng)?);
    let emb = rng.bf16s(n * H, 1.0, 0.0);
    let (r, s) = (Arm::new(g, n, 1)?, Arm::new(g, n, 1)?);
    g.copy_h2d(&emb, r.hidden)?;
    g.copy_h2d(&emb, s.hidden)?;
    let lanes = s.rows(k, norm_k);
    let mut bad = 0usize;
    // 2026-10-04: The attention site with `hc_expand` (layer 0), then the FFN site with the
    // collapse (last layer). Each back reads the call's `y` rows as its block output.
    let passes = [
        (&attn_calls, &attn, true, false),
        (&ffn_wins, &ffn, false, true),
    ];
    for (cs, st, expand, last) in passes {
        for &(t, kk) in cs.iter() {
            ref_front(
                g,
                k,
                norm_k,
                &r,
                st,
                (t, kk),
                expand,
                r.normed.offset(t * H * 2),
                stream,
            )?;
        }
        let sp_site = SpSite {
            weights: &st.w,
            norm: st.norm,
            expand,
            last,
            reduce: false,
            attn: expand,
            post_mix: None,
            premixed: false,
        };
        for &call in cs.iter() {
            for rank in 0..2 {
                for sp in plan.spans(rank, call) {
                    lanes.front(g, s.hidden, sp, call.1, &sp_site, stream)?;
                }
            }
        }
        let front_diff: usize = r
            .sized(n)
            .iter()
            .zip(s.sized(n))
            .map(|(&(pr, b), (ps, _))| -> Result<usize> {
                let (a, bb) = (down(g, pr, b, stream)?, down(g, ps, b, stream)?);
                Ok(diff_rows(&a, &bb, &[(0, b)], 1))
            })
            .sum::<Result<usize>>()?;
        bad += front_diff;
        for &(t, kk) in cs.iter() {
            let y = r.hidden.offset(t * H * 2);
            ref_back(g, k, &r, (t, kk), y, last, stream)?;
        }
        for &call in cs.iter() {
            for rank in 0..2 {
                for sp in plan.spans(rank, call) {
                    lanes.back(g, s.hidden, sp, last, None, stream)?;
                }
            }
        }
        let back_diff: usize = r
            .sized(n)
            .iter()
            .zip(s.sized(n))
            .map(|(&(pr, b), (ps, _))| -> Result<usize> {
                let (a, bb) = (down(g, pr, b, stream)?, down(g, ps, b, stream)?);
                Ok(diff_rows(&a, &bb, &[(0, b)], 1))
            })
            .sum::<Result<usize>>()?;
        bad += back_diff;
        if front_diff + back_diff > 0 {
            println!(
                "  row-local {c:?} {}: front differs in {front_diff} bytes, back in \
                 {back_diff}",
                if expand { "attn" } else { "ffn" }
            );
        }
    }
    let hr = down(g, r.hidden, n * H * 2, stream)?;
    if nonfinite_bf16(&hr) > 0 {
        println!(
            "  row-local {c:?}: {} non-finite outputs",
            nonfinite_bf16(&hr)
        );
        bad += 1;
    }
    for st in [attn, ffn] {
        for p in [st.w.hc_fn, st.w.hc_scale, st.w.hc_base, st.w.mix, st.norm] {
            g.free(p)?;
        }
    }
    r.free(g)?;
    s.free(g)?;
    Ok(bad)
}

/// 2026-10-04: The stand-in mixer / MLP over call `(t, k)` with normed rows at `x`: each row's
/// partial is its normed row plus the normed row `shift` rows later in the call (wrapping),
/// plus this rank's `bias` row; the attention stand-in writes it over `x` (as DSA does), the
/// MLP one into `out` (as the MoE does). Returns where the partial is.
#[allow(clippy::too_many_arguments)]
fn stand_in(
    g: &dyn GpuBackend,
    add: KernelHandle,
    (t, k): Span,
    x: DevicePtr,
    shift: usize,
    bias: DevicePtr,
    tmp: DevicePtr,
    out: Option<DevicePtr>,
    stream: u64,
) -> Result<DevicePtr> {
    let rb = H * 2;
    let s = shift % k;
    g.copy_d2d_async(x.offset(s * rb), tmp, (k - s) * rb, stream)?;
    if s > 0 {
        g.copy_d2d_async(x, tmp.offset((k - s) * rb), s * rb, stream)?;
    }
    let p = match out {
        Some(o) => {
            g.copy_d2d_async(x, o, k * rb, stream)?;
            o
        }
        None => x,
    };
    for src in [tmp, bias.offset(t * rb)] {
        let e = k * H;
        KernelLaunch::new(g, add)
            .grid([(e as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(p)
            .arg_ptr(src)
            .arg_i32(e as i32)
            .launch(stream)?;
    }
    Ok(p)
}

/// 2026-10-04: The layer gate (module doc) for one case on a two-rank `comm`; returns the
/// differing bytes and the two arms' times (ms).
#[allow(clippy::too_many_arguments)]
fn layer_case(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    norm_k: KernelHandle,
    add: KernelHandle,
    comm: &dyn CommBackend,
    c: &Case,
    stream: u64,
) -> Result<(usize, f32, f32)> {
    let (n, rows) = (c.0, c.1);
    let rank = comm.rank();
    let rb = H * 2;
    let plan = SpPlan::new(&owners(c)).context("no sub-chunks")?;
    let (attn_calls, ffn_wins) = calls(c);
    let window = attn_calls
        .iter()
        .chain(&ffn_wins)
        .map(|s| s.1)
        .max()
        .unwrap_or(1);
    // 2026-10-04: Same seed on both ranks for the weights and the embedding; the bias is
    // per rank.
    let mut rng = Lcg(0x5EED_7000 ^ (n as u64) << 20 ^ rows as u64);
    let layers: Vec<(Site, Site)> = (0..2)
        .map(|_| Ok::<_, anyhow::Error>((site(g, &mut rng)?, site(g, &mut rng)?)))
        .collect::<Result<_>>()?;
    let emb = rng.bf16s(n * H, 1.0, 0.0);
    let bias = up(
        g,
        &Lcg(0xB1A5 ^ (rank as u64 + 1) << 32).bf16s(n * H, 0.5, 0.0),
    )?;
    let tmp = g.alloc(window * rb)?;
    let (r, s) = (Arm::new(g, n, window)?, Arm::new(g, n, window)?);
    let half = rows.div_ceil(2).max(1);

    // 2026-10-04: Arm R, the replicated staged pass: per call the front with the norm into
    // rows `0..k` of `normed`, the stand-in, the all-reduce, the back (`attn_half_inner`,
    // `ffn_half_inner`).
    let arm_r = || -> Result<()> {
        g.copy_h2d(&emb, r.hidden)?;
        for (li, (sa, sf)) in layers.iter().enumerate() {
            let (first, last) = (li == 0, li + 1 == layers.len());
            for &(t, kk) in &attn_calls {
                ref_front(g, k, norm_k, &r, sa, (t, kk), first, r.normed, stream)?;
                let p = stand_in(g, add, (t, kk), r.normed, half, bias, tmp, None, stream)?;
                comm.all_reduce_async(p.0, kk * rb, stream)?;
                ref_back(g, k, &r, (t, kk), p, false, stream)?;
            }
            for &(t, kk) in &ffn_wins {
                ref_front(g, k, norm_k, &r, sf, (t, kk), false, r.normed, stream)?;
                let o = Some(r.ffn_out);
                let p = stand_in(g, add, (t, kk), r.normed, 1, bias, tmp, o, stream)?;
                comm.all_reduce_async(p.0, kk * rb, stream)?;
                ref_back(g, k, &r, (t, kk), p, last, stream)?;
            }
        }
        Ok(())
    };
    // 2026-10-04: Arm S, the production sequence-parallel pass (`prefill_staged_sp`'s body).
    let lanes = s.rows(k, norm_k);
    let arm_s = || -> Result<()> {
        g.copy_h2d(&emb, s.hidden)?;
        for (li, (sa, sf)) in layers.iter().enumerate() {
            let (first, last) = (li == 0, li + 1 == layers.len());
            let attn = SpSite {
                weights: &sa.w,
                norm: sa.norm,
                expand: first,
                last: false,
                reduce: true,
                attn: true,
                post_mix: None,
                premixed: false,
            };
            let mixer =
                |c: Span, x: DevicePtr| stand_in(g, add, c, x, half, bias, tmp, None, stream);
            sp_pass(
                g,
                comm,
                add,
                plan,
                &lanes,
                s.hidden,
                &attn_calls,
                attn,
                mixer,
                stream,
            )?;
            let ffn = SpSite {
                weights: &sf.w,
                norm: sf.norm,
                expand: false,
                last,
                reduce: true,
                attn: false,
                post_mix: None,
                premixed: false,
            };
            let o = Some(s.ffn_out);
            let mlp = |c: Span, x: DevicePtr| stand_in(g, add, c, x, 1, bias, tmp, o, stream);
            sp_pass(
                g, comm, add, plan, &lanes, s.hidden, &ffn_wins, ffn, mlp, stream,
            )?;
            if last {
                swap_owned(comm, plan, s.hidden, (0, n), rb, stream)?;
            }
        }
        Ok(())
    };
    let tr = timed(g, stream, arm_r)?;
    let ts = timed(g, stream, arm_s)?;

    let own = plan.spans(rank, (0, n));
    let all = [(0, n)];
    let (hr, hs) = (
        down(g, r.hidden, n * rb, stream)?,
        down(g, s.hidden, n * rb, stream)?,
    );
    let mut bad = diff_rows(&hr, &hs, &all, rb);
    let hidden_bad = bad;
    for (i, row_bytes) in [(1usize, HC * H * 4), (2, HC * 4), (3, HC * HC * 4)] {
        let (pr, b) = r.sized(n)[i];
        let ps = s.sized(n)[i].0;
        let (a, bb) = (down(g, pr, b, stream)?, down(g, ps, b, stream)?);
        bad += diff_rows(&a, &bb, &own, row_bytes);
    }
    let nonfinite = nonfinite_bf16(&hr);
    println!(
        "  layer {c:?} rank {rank}: hidden differs in {hidden_bad} bytes, own highway in {} \
         bytes; {nonfinite} non-finite; replicated {tr:.3} ms | sequence-parallel {ts:.3} ms \
         (2 layers, stand-in mixers)",
        bad - hidden_bad
    );
    if nonfinite > 0 {
        bad += 1;
    }
    for (sa, sf) in layers {
        for st in [sa, sf] {
            for p in [st.w.hc_fn, st.w.hc_scale, st.w.hc_base, st.w.mix, st.norm] {
                g.free(p)?;
            }
        }
    }
    g.free(bias)?;
    g.free(tmp)?;
    r.free(g)?;
    s.free(g)?;
    Ok((bad, tr, ts))
}

struct Args {
    rank: Option<usize>,
    peer: String,
    port: u16,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        rank: None,
        peer: String::new(),
        port: 29564,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let v = argv
            .get(i + 1)
            .with_context(|| format!("{} needs a value", argv[i]))?;
        match argv[i].as_str() {
            "--rank" => a.rank = Some(v.parse()?),
            "--peer" => a.peer = v.clone(),
            "--port" => a.port = v.parse()?,
            other => bail!("unknown argument {other}"),
        }
        i += 2;
    }
    if a.rank.is_some_and(|r| r > 1) || (a.rank.is_some() && a.peer.is_empty()) {
        bail!("usage: [--rank 0|1 --peer <rank0 address> [--port 29564]] (none: single GPU)");
    }
    Ok(a)
}

/// 2026-10-04: Set `key` to `value` unless the environment already has it, before any reader
/// caches it (the first thing `main` does, single-threaded).
fn default_env(key: &str, value: &str) {
    if std::env::var_os(key).is_none() {
        // SAFETY: called at the top of `main`, before any other thread exists.
        unsafe { std::env::set_var(key, value) };
    }
}

fn main() -> Result<()> {
    default_env("METRALE_GLM_PREFILL_STAGED", "1");
    default_env("METRALE_GLM_PREFILL_ROWS_FFN", "8192");
    default_env("METRALE_GLM_MHC_TOKMAJOR", "1");
    let a = parse_args()?;
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (k, norm_k, add) = match (
        Glm5NextMhcKernels::resolve(g),
        g.kernel("rms_norm_vanilla", "rms_norm_vanilla"),
        g.kernel("bf16_add", "bf16_add_inplace"),
    ) {
        (Ok(k), Ok(n), Ok(a)) => (k, n, a),
        (k, n, a) => {
            println!(
                "a kernel is absent from this target (mhc {:?}, rms_norm_vanilla {:?}, \
                 bf16_add_inplace {:?}) - SKIP",
                k.err(),
                n.err(),
                a.err()
            );
            std::process::exit(2);
        }
    };
    let max_rows = CASES.iter().map(|c| c.0.max(c.2)).max().unwrap_or(1);
    if mhc_mix_max_tokens() < CASES.iter().map(|c| c.2.min(c.0)).max().unwrap_or(1) {
        bail!(
            "mhc_mix_max_tokens() is {}: run with METRALE_GLM_PREFILL_STAGED=1 \
             METRALE_GLM_PREFILL_ROWS_FFN=8192 (or leave both unset)",
            mhc_mix_max_tokens()
        );
    }
    let ncases = std::env::var("SP_GATE_CASES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(CASES.len())
        .min(CASES.len());
    println!(
        "glm5next_seq_parallel_layer_2rank: {ncases} cases, METRALE_GLM_MHC_TOKMAJOR {} \
         (tokmajor kernel resolved: {}), mix scratch {} rows",
        mhc_tokmajor(),
        k.hc_mix_bf16_tokmajor.0 != 0,
        mhc_mix_max_tokens()
    );
    let stream = g.create_stream()?;
    let mut bad_cases = Vec::new();
    for c in &CASES[..ncases] {
        let bad = row_local_case(g, &k, norm_k, c, stream)?;
        println!("row-local {c:?}: {bad} differing bytes");
        if bad > 0 {
            bad_cases.push(format!("row-local {c:?}"));
        }
    }
    let mut mode = "row-local (single GPU)".to_string();
    if let Some(rank) = a.rank {
        let comm = NcclBackend::new(rank, 2, &a.peer, a.port, stream, max_rows * H * 2)?;
        comm.set_add_kernel(add.0);
        if !comm.all_reduce_is_send_recv_add() {
            println!("FAIL: the backend's all-reduce is not the send/recv + add path");
            std::process::exit(1);
        }
        let (mut sum_r, mut sum_s) = (0f32, 0f32);
        for c in &CASES[..ncases] {
            let (bad, tr, ts) = layer_case(g, &k, norm_k, add, &comm, c, stream)?;
            println!("layer {c:?} rank {rank}: {bad} differing bytes");
            if bad > 0 {
                bad_cases.push(format!("layer {c:?}"));
            }
            sum_r += tr;
            sum_s += ts;
        }
        println!(
            "TIMING rank {rank}: replicated {sum_r:.3} ms, sequence-parallel {sum_s:.3} ms over \
             all cases (2 layers each, stand-in mixers; first run, includes warm-up)"
        );
        mode = format!("row-local + two-rank layer, rank {rank}");
    }
    if bad_cases.is_empty() {
        println!("PASS: {mode}: {ncases} cases byte-identical to the replicated staged pass");
        Ok(())
    } else {
        println!("FAIL: {mode}: {} differ: {bad_cases:?}", bad_cases.len());
        std::process::exit(1);
    }
}
