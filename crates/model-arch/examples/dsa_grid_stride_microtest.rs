// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Bit-parity gate and timing for `METRALE_GLM_DSA_GRID_STRIDE`: `dsa_kpool_compress`
//! and `dsa_index_scores` (the pool-indexed kernels of a ceiling, graph-replay decode selection)
//! launched with the ceiling grid the host used to give them (`max_pools + 1` and `max_pools`
//! blocks, one per pool) against the same kernels on stride grids. Identical inputs, the
//! arguments `select_tokens` passes on a ceiling launch (S, pool counts from a device `geom`),
//! GLM-5.3's H = 32, D = 128, KP = 4, one query row.
//!
//! * Capacities `max_pools` in {16,384, 196,608} (`--max-seq-len` 65,536 and 786,432), live
//!   complete pools in {1, 37, 1,024, 16,384}, each with 0 and 3 tokens past the last complete
//!   pool (3 makes a trailing partial pool, which compress writes and nothing reads; at 16,384
//!   live of 16,384 it is the ceiling grid's extra `+ 1` block). The data has invalid tokens,
//!   so some pools are invalid or not visible to the query (the `continue` path of the scores).
//! * Stride arms: the production grid (`ceiling_grids`), and 1, 2, 7, 48 and 576 blocks, the
//!   live pool count, and one fewer, so a block takes 1, 2 or many pools.
//! * Every output buffer is compared as raw bytes over its WHOLE capacity: compress `pool_keys`,
//!   `pool_indices`, `pool_valid`; scores `out`, `valid_cand`. Each arm runs twice, its outputs
//!   first filled with 0xA5 and then 0x5A, and each run is compared with the reference arm run
//!   under the same fill, so an element one arm writes and the other does not cannot match, and
//!   the pools written must be exactly the live ones.
//! * Controls that must fire: one flipped key bit, four negated query heads, and a geometry one
//!   pool short. A run that compared nothing, or whose data has no candidate or no positive
//!   score, fails.
//! * Timing: CUDA events around a replay of a captured 33-call graph (11 layers' caches x 3
//!   verify rows, as in a K = 3 decode step), median of 51 replays after 3 warm-ups, as us per
//!   call. One `TIMING` line per (capacity, live): ceiling grid against the production stride
//!   grid for compress + scores, compress alone and scores alone; one `SWEEP` line per
//!   (capacity, live) over 1..16 blocks per SM, to tune `STRIDE_BLOCKS_PER_SM`. Timing does not
//!   decide the verdict.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and a `PASS` line when everything is bit-identical and the controls fire; 1 and a
//! line saying so otherwise; 2 when a kernel, or `dsa_indexer_grid_stride_v1` (the marker a
//! `dsa_indexer.cu` defines iff both kernels carry the loop), is absent from this target.
//!
//! Run (both kernels are gb10 common kernels):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_grid_stride_microtest

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::DSA_MODULE;
use metrale_model_arch::glm5next_dsa::select::grid_stride::{
    GRID_STRIDE_MARKER, STRIDE_BLOCKS_PER_SM, ceiling_grids, stride_blocks,
};

#[path = "common/dsa_grid_stride_rig.rs"]
mod dsa_grid_stride_rig;
use dsa_grid_stride_rig::*;

// 2026-10-05: CUDA driver event API for kernel-only timing, declared as in
// `dsa_indexer_tiled_bitparity_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-05: Capacities (`max_pools` at `--max-seq-len` 65,536 and 786,432) and the live
/// complete pools and tail tokens tried at each.
const CAPS: &[usize] = &[16_384, 196_608];
const LIVES: &[usize] = &[1, 37, 1_024, 16_384];
const TAILS: &[usize] = &[0, 3];
/// 2026-10-05: Absolute stride grids tried for exactness besides the production one.
const GRIDS: &[usize] = &[1, 2, 7, 48, 576];
/// 2026-10-05: Blocks per SM swept in the timing.
const SWEEP: &[usize] = &[1, 2, 4, 8, 12, 16];
/// 2026-10-05: Selector calls in a K = 3 verify step: one per layer per row.
const CALLS: usize = LAYERS * 3;
const REPLAYS: usize = 51;

/// 2026-10-05: Stride arms of one config: (label, compress grid, scores grid).
fn arms(m: usize, sms: usize, s: usize) -> Vec<(String, usize, usize)> {
    let (npf, np) = (s.div_ceil(KP), s / KP);
    let (pc, ps) = ceiling_grids(m, sms, true);
    let mut v = vec![(format!("production {STRIDE_BLOCKS_PER_SM}/SM"), pc, ps)];
    for &x in GRIDS {
        v.push((format!("{x} blocks"), x.min(m + 1), x.min(m)));
    }
    v.push(("live pools".into(), npf.max(1), np.max(1)));
    v.push((
        "live pools - 1".into(),
        npf.saturating_sub(1).max(1),
        np.saturating_sub(1).max(1),
    ));
    v
}

/// 2026-10-05: One (capacity, live pools, tail tokens) config: every stride arm against the
/// ceiling-grid reference, under both fills.
fn legs(
    g: &dyn GpuBackend,
    d: &Dev,
    a: &Arena,
    live: usize,
    tail: usize,
    t: &mut Tally,
) -> Result<()> {
    let (m, s) = (a.m, live * KP + tail);
    let geom = up(g, &geom_bytes(s))?;
    g.copy_h2d(&i32_bytes(&[q_pos_of(s)]), d.q_pos)?;
    let l = d.layers[0];
    let (cc, cs) = ceiling_grids(m, d.sms, false);
    let arms = arms(m, d.sms, s);
    let mut ok = vec![(true, true); arms.len()];
    let mut refs = Vec::new();
    for poison in [POISON_A, POISON_B] {
        a.fill(g, a.r, a.o_ref, poison)?;
        compress(g, d, m, cc, l, a.r, geom, 0)?;
        let rc = a.read_pools(g, a.r)?;
        scores(g, d, m, cs, (l, d.q), a.r, a.o_ref, geom, 0)?;
        let rs = a.read_scored(g, a.o_ref)?;
        for (i, (_, cx, sx)) in arms.iter().enumerate() {
            a.fill(g, a.n, a.o_new, poison)?;
            compress(g, d, m, *cx, l, a.n, geom, 0)?;
            ok[i].0 &= t.same(&rc, &a.read_pools(g, a.n)?);
            scores(g, d, m, *sx, (l, d.q), a.r, a.o_new, geom, 0)?;
            ok[i].1 &= t.same(&rs, &a.read_scored(g, a.o_new)?);
        }
        refs.push((rc, rs));
    }
    for ((label, cx, sx), (c_ok, s_ok)) in arms.iter().zip(&ok) {
        println!(
            "m={m} live={live} tail={tail} {label:<20} grid {cx}+{sx}: compress identical={c_ok:<5} \
             scores identical={s_ok}"
        );
    }
    // 2026-10-05: The elements a run wrote are those that do not hold its fill, so they are the
    // ones equal across the two fills: exactly the live pools, no fewer and no more.
    let wrote = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(p, q)| p == q).count();
    let (wp, wc) = (
        wrote(&refs[0].0[2], &refs[1].0[2]),
        wrote(&refs[0].1[1], &refs[1].1[1]),
    );
    let (want_p, want_c) = (s.div_ceil(KP), s / KP);
    println!(
        "m={m} live={live} tail={tail} pools written: compress {wp} (want {want_p}), scores {wc} (want {want_c})"
    );
    t.failed += usize::from((wp, wc) != (want_p, want_c));
    t.candidates += refs[0].1[1].iter().filter(|&&v| v == 1).count();
    t.positive += refs[0].1[0]
        .chunks_exact(4)
        .filter(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]) > 0.0)
        .count();
    g.free(geom).ok();
    Ok(())
}

/// 2026-10-05: The three negative controls on one config (16,384 pools of capacity, 37 live):
/// a flipped key bit, four negated query heads, a geometry one pool short. Each must change
/// the stride arm's bytes against the reference. Returns whether each fired.
fn controls(g: &dyn GpuBackend, d: &Dev, a: &Arena) -> Result<[bool; 3]> {
    let (m, live) = (a.m, 37);
    let s = live * KP;
    let (geom, short) = (up(g, &geom_bytes(s))?, up(g, &geom_bytes(s - KP))?);
    g.copy_h2d(&i32_bytes(&[q_pos_of(s)]), d.q_pos)?;
    let l = d.layers[0];
    let (cx, sx) = ceiling_grids(m, d.sms, true);
    a.fill(g, a.r, a.o_ref, POISON_A)?;
    compress(g, d, m, cx, l, a.r, geom, 0)?;
    scores(g, d, m, sx, (l, d.q), a.r, a.o_ref, geom, 0)?;
    let (rc, rs) = (a.read_pools(g, a.r)?, a.read_scored(g, a.o_ref)?);

    // 2026-10-05: Control 1: the top mantissa bit of channel 0 of a valid token of the last
    // complete pool.
    let tok = (0..KP)
        .map(|i| (live - 1) * KP + i)
        .find(|t| t % 23 != 11)
        .unwrap_or(0);
    let mut k = d.k_host.clone();
    k[2 * tok * D] ^= 0x40;
    let pert = Layer { k: up(g, &k)?, ..l };
    a.fill(g, a.n, a.o_new, POISON_A)?;
    compress(g, d, m, cx, pert, a.n, geom, 0)?;
    let c1 = a.read_pools(g, a.n)? != rc;

    // 2026-10-05: Control 2: heads 0..4 negated, so relu clamps the other way for some pools.
    let neg: Vec<f32> = d
        .q_host
        .iter()
        .enumerate()
        .map(|(i, &x)| if i < 4 * D { -x } else { x })
        .collect();
    let q_neg = up(g, &f32_bytes(&neg))?;
    a.fill(g, a.n, a.o_new, POISON_A)?;
    scores(g, d, m, sx, (l, q_neg), a.r, a.o_new, geom, 0)?;
    let c2 = a.read_scored(g, a.o_new)? != rs;

    // 2026-10-05: Control 3: one pool short: that pool stays at its fill in both kernels.
    a.fill(g, a.n, a.o_new, POISON_A)?;
    compress(g, d, m, cx, l, a.n, short, 0)?;
    scores(g, d, m, sx, (l, d.q), a.r, a.o_new, short, 0)?;
    let c3 = a.read_pools(g, a.n)? != rc && a.read_scored(g, a.o_new)? != rs;
    println!(
        "CONTROL key bit flipped detected={c1}; query heads negated detected={c2}; geometry one pool short detected={c3}"
    );
    for p in [geom, short, pert.k, q_neg] {
        g.free(p).ok();
    }
    Ok([c1, c2, c3])
}

/// 2026-10-05: Median ms of one replay of the graph `f` captures on `s`, over `REPLAYS`
/// replays each bracketed by CUDA events, after 3 warm-up replays.
fn time_graph(g: &dyn GpuBackend, s: u64, f: &mut dyn FnMut(u64) -> Result<()>) -> Result<f64> {
    g.begin_capture(s)?;
    if let Err(e) = f(s) {
        g.abort_capture_if_active(s);
        return Err(e);
    }
    let graph = g.end_capture(s)?;
    for _ in 0..3 {
        g.launch_graph(graph, s)?;
    }
    g.synchronize(s)?;
    let mut ev = vec![(0u64, 0u64); REPLAYS];
    // SAFETY: plain CUDA driver event calls; the events are created, used and destroyed here,
    // and the backend made the context current when it was built.
    unsafe {
        for e in ev.iter_mut() {
            check(cuEventCreate(&mut e.0, 0), "cuEventCreate")?;
            check(cuEventCreate(&mut e.1, 0), "cuEventCreate")?;
        }
    }
    for e in &ev {
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.0, s), "cuEventRecord(start)")? };
        g.launch_graph(graph, s)?;
        // SAFETY: as above.
        unsafe { check(cuEventRecord(e.1, s), "cuEventRecord(end)")? };
    }
    g.synchronize(s)?;
    let mut ms = Vec::with_capacity(REPLAYS);
    for e in &ev {
        let mut t: f32 = 0.0;
        // SAFETY: as above; `t` outlives the call and both events have completed.
        unsafe {
            check(cuEventElapsedTime(&mut t, e.0, e.1), "cuEventElapsedTime")?;
            cuEventDestroy_v2(e.0);
            cuEventDestroy_v2(e.1);
        }
        ms.push(f64::from(t));
    }
    g.destroy_graph(graph)?;
    ms.sort_by(|x, y| x.total_cmp(y));
    Ok(ms[ms.len() / 2])
}

/// 2026-10-05: Which kernels a timed graph launches.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Pair,
    Compress,
    Scores,
}

/// 2026-10-05: us per selector call: a graph of `CALLS` calls (call `i` reads layer `i / 3`'s
/// cache) with `grids` = (compress blocks, scores blocks), timed by `time_graph`.
fn us_per_call(
    g: &dyn GpuBackend,
    d: &Dev,
    a: &Arena,
    grids: (usize, usize),
    kind: Kind,
    geom: DevicePtr,
    st: u64,
) -> Result<f64> {
    let ms = time_graph(g, st, &mut |s| {
        for i in 0..CALLS {
            let l = d.layers[(i / 3) % d.layers.len()];
            if kind != Kind::Scores {
                compress(g, d, a.m, grids.0, l, a.r, geom, s)?;
            }
            if kind != Kind::Compress {
                scores(g, d, a.m, grids.1, (l, d.q), a.r, a.o_ref, geom, s)?;
            }
        }
        Ok(())
    })?;
    Ok(ms * 1000.0 / CALLS as f64)
}

/// 2026-10-05: The `TIMING` and `SWEEP` lines of one (capacity, live pools).
fn timing(g: &dyn GpuBackend, d: &Dev, a: &Arena, live: usize) -> Result<()> {
    let (m, s) = (a.m, live * KP);
    let geom = up(g, &geom_bytes(s))?;
    g.copy_h2d(&i32_bytes(&[q_pos_of(s)]), d.q_pos)?;
    let st = g.create_stream()?;
    // 2026-10-05: Fill the pool buffers so a scores-only graph reads live data.
    compress(g, d, m, m + 1, d.layers[0], a.r, geom, 0)?;
    g.synchronize(0)?;
    let (old, new) = (
        ceiling_grids(m, d.sms, false),
        ceiling_grids(m, d.sms, true),
    );
    let t = |grids, kind| us_per_call(g, d, a, grids, kind, geom, st);
    let (po, co, so) = (
        t(old, Kind::Pair)?,
        t(old, Kind::Compress)?,
        t(old, Kind::Scores)?,
    );
    let (pn, cn, sn) = (
        t(new, Kind::Pair)?,
        t(new, Kind::Compress)?,
        t(new, Kind::Scores)?,
    );
    println!(
        "TIMING m={m} live={live}: us per call (compress + scores | compress | scores), median of \
         {REPLAYS} replays of {CALLS} calls: ceiling grid {}+{} blocks {po:.1} | {co:.1} | {so:.1}; \
         stride grid {}+{} blocks {pn:.1} | {cn:.1} | {sn:.1}; compress + scores {:.2}x",
        old.0,
        old.1,
        new.0,
        new.1,
        po / pn
    );
    let mut line =
        format!("SWEEP m={m} live={live}: us per call (compress + scores) by blocks per SM:");
    for &f in SWEEP {
        let grids = ((f * d.sms).min(m + 1), (f * d.sms).min(m));
        line += &format!(" {f}x={:.1}", t(grids, Kind::Pair)?);
    }
    println!("{line} (production {STRIDE_BLOCKS_PER_SM}x, ceiling {po:.1})");
    g.free(geom).ok();
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (Ok(compress_k), Ok(scores_k), Ok(_marker)) = (
        g.kernel(DSA_MODULE, "dsa_kpool_compress"),
        g.kernel(DSA_MODULE, "dsa_index_scores"),
        g.kernel(DSA_MODULE, GRID_STRIDE_MARKER),
    ) else {
        println!(
            "dsa_kpool_compress / dsa_index_scores / {GRID_STRIDE_MARKER} absent from this \
             target (its dsa_indexer.cu has no stride loop) - SKIP"
        );
        std::process::exit(2);
    };
    let d = Dev::new(g, compress_k, scores_k)?;
    println!(
        "device: {} SMs; stride grid {} blocks",
        d.sms,
        stride_blocks(d.sms)
    );

    let mut t = Tally::default();
    let mut control = [false; 3];
    for &m in CAPS {
        let a = Arena::new(g, m)?;
        for &live in LIVES.iter().filter(|&&l| l <= m) {
            for &tail in TAILS {
                legs(g, &d, &a, live, tail, &mut t)?;
            }
        }
        if m == CAPS[0] {
            control = controls(g, &d, &a)?;
        }
        for &live in LIVES.iter().filter(|&&l| l <= m) {
            timing(g, &d, &a, live)?;
        }
    }

    println!(
        "{} legs, {} bytes compared; {} candidate pools, {} positive scores",
        t.legs, t.bytes, t.candidates, t.positive
    );
    if t.legs == 0 || t.bytes == 0 || t.candidates == 0 || t.positive == 0 {
        println!("FAIL - nothing, or only dead data, was compared; this run proves nothing.");
        std::process::exit(1);
    }
    if !control.iter().all(|&c| c) {
        println!("FAIL - a negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if t.failed > 0 {
        println!(
            "FAIL - {} comparison(s) differ. Keep METRALE_GLM_DSA_GRID_STRIDE=0 on this build.",
            t.failed
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {} bytes over {} legs bit-identical: dsa_kpool_compress and dsa_index_scores on \
         stride grids match the one-block-per-pool ceiling grids, and write exactly the live pools.",
        t.bytes, t.legs
    );
    Ok(())
}
