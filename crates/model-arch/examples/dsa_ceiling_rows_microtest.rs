// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Byte-parity gate and timing for ONE ceiling selection over a sequence's `k`
//! verify rows (`DsaSelectLaunch::CeilingRows`, `q_rows = k`) against today's `k` single-row
//! ceiling selections (`decode_k_rows`' per-row loop). Written 2026-10-12, NOT yet run (needs
//! a GB10).
//!
//! Fixture: `examples/common/dsa_ceiling_rows_rig.rs` (two pool-cache states with the same
//! context; arm A = per row store_ring + write_geom_pk + 1-row ceiling select; arm B = k
//! store_rings + one write_geom_pk from the last row's `seq_len` + one k-row ceiling select).
//! The car's DSA selection levers are set here (POOL_CACHE, SCORES_DECODE, TOPK_RADIX,
//! TOPK_RADIX_COMPACT, and the exact-launch-only TC/TC2/TILED/EXPAND_TAIL_COUNT, which a
//! ceiling launch never takes), so the chain is the car's: `dsa_kpool_compress_incr`,
//! `dsa_index_scores_decode`, radix top-k with compaction, `dsa_expand_selection`.
//!
//! Cases: k in {2, 3, 8} x context S in {260, 516, 2044, 4096, 16384, 65536}, the step
//! starting at seq = S - 2, so the rows straddle a pool boundary (row 0 at length S - 1, row 1
//! at S, a multiple of 4). S = 2044 has select_k = P < 512 that differs between rows. At
//! S = 260 and 516, row 0's visible tail (3 keys at slots `row_select_k * 4` = 256 and 512
//! onward) sits across a `glm5next_dsa_mla_decode_fp8` warp-slice edge (slice =
//! ceil(2051 / 8) = 257: slots 257 and 514), while the pass's select_k is the last row's
//! (65 and 129), so a pass-scalar tail base would move it (the Atlas 2026-09-06 defect).
//! Per case `DRAWS` (64) random steps (new staged rows, query, head weights) by graph replay;
//! the first draw also eagerly. Each draw compares every row's `[out_width]` token ids and
//! both states' pools (keys, ids, validity over `[0, ceil((seq + k) / 4))`, and the device
//! watermark) as raw bytes, and checks each arm's rows against a host layout reference (pool
//! slots, tail base `row_select_k * 4`, tail keys; `placement_error`); a row selecting nothing
//! fails the run.
//! Controls, which must fire: arm A row r against arm B row r + 1 differs (the compare can
//! see a row mix-up), and arm B with its geometry written from row 0's `seq_len` differs from
//! arm A at every k = 3 / 8 case (a wrong shared geometry is caught).
//!
//! Timing: CUDA events, median of `REPS` (60) after 5 warm-ups, arms interleaved per rep, for
//! the captured step graph (production verify is graphed) and the same launches enqueued
//! eagerly (includes host launch cost). Arms: A (k rows), B (k rows), ONE (a 1-row chain, A at
//! k = 1). Timing does not decide the verdict.
//!
//! Output: `CASE` per (S, k) (draws, differing draws, per-row select_k and valid counts),
//! `TIMING` per (S, k) and mode with A, B, ONE in us, B/A, A/k/ONE and the primary B/ONE,
//! `CONTROL`, then `PASS ...` when every compare is byte-equal and both controls fired, else
//! `FAIL <which>`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 with `PASS`; 1 with `FAIL`; 2 when a kernel it needs is absent from this target.
//!
//! Run (gb10 common kernels; sets the levers itself):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_ceiling_rows_microtest
//! About 0.2 GB of device memory; a few minutes (context fill to 65,534 rows, 18 x 64 draws).

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, GraphHandle};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;

#[path = "common/dsa_ceiling_rows_rig.rs"]
mod dsa_ceiling_rows_rig;
use dsa_ceiling_rows_rig::*;

// 2026-10-12: CUDA driver event API for kernel-only timing, declared as in
// `dsa_grid_stride_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const KS: &[usize] = &[2, 3, 8];
const CONTEXTS: &[usize] = &[260, 516, 2_044, 4_096, 16_384, 65_536];
const DRAWS: usize = 64;
const REPS: usize = 60;
const WARMUP: usize = 5;
/// 2026-10-12: The car's DSA selection levers (CAR.json c27c env).
const LEVERS: &[(&str, &str)] = &[
    ("METRALE_GLM_DSA_POOL_CACHE", "1"),
    ("METRALE_GLM_DSA_SCORES_DECODE", "1"),
    ("METRALE_GLM_DSA_TOPK_RADIX", "1"),
    ("METRALE_GLM_DSA_TOPK_RADIX_COMPACT", "1"),
    ("METRALE_GLM_DSA_SCORES_TC", "bf16"),
    ("METRALE_GLM_DSA_SCORES_TC2", "1"),
    ("METRALE_GLM_DSA_SCORES_TILED", "1"),
    ("METRALE_GLM_DSA_EXPAND_TAIL_COUNT", "1"),
];

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: CUresult {rc}");
    }
    Ok(())
}

/// 2026-10-12: Event pairs bracketing each timed launch.
struct Events(Vec<(u64, u64)>);
impl Events {
    fn new(n: usize) -> Result<Self> {
        let mut v = vec![(0u64, 0u64); n];
        for e in v.iter_mut() {
            // SAFETY: plain CUDA driver event calls; the backend made the context current.
            unsafe {
                check(cuEventCreate(&mut e.0, 0), "cuEventCreate")?;
                check(cuEventCreate(&mut e.1, 0), "cuEventCreate")?;
            }
        }
        Ok(Self(v))
    }
    fn ms(&self, i: usize) -> Result<f64> {
        let mut t: f32 = 0.0;
        // SAFETY: both events were recorded and the stream synchronised by the caller.
        unsafe {
            check(
                cuEventElapsedTime(&mut t, self.0[i].0, self.0[i].1),
                "elapsed",
            )?
        };
        Ok(f64::from(t))
    }
}
impl Drop for Events {
    fn drop(&mut self) {
        for e in &self.0 {
            // SAFETY: events created in `new`, not used after this.
            unsafe {
                cuEventDestroy_v2(e.0);
                cuEventDestroy_v2(e.1);
            }
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|x, y| x.total_cmp(y));
    v[v.len() / 2]
}

/// 2026-10-12: Median us of each arm (A, B, ONE), arms interleaved per rep; `graph` replays
/// the captured step, else the launches are enqueued eagerly.
fn time_arms(rig: &Rig, k: usize, graphs: Option<[GraphHandle; 3]>) -> Result<[f64; 3]> {
    let (g, s) = (rig.g, rig.s);
    let arms = [(Arm::PerRow, k), (Arm::Rows, k), (Arm::PerRow, 1)];
    let run = |i: usize| -> Result<()> {
        match graphs {
            Some(gr) => g.launch_graph(gr[i], s),
            None => rig.step(arms[i].0, arms[i].1),
        }
    };
    for _ in 0..WARMUP {
        for i in 0..3 {
            run(i)?;
        }
    }
    rig.sync()?;
    let ev = Events::new(REPS * 3)?;
    for rep in 0..REPS {
        for i in 0..3 {
            let e = ev.0[rep * 3 + i];
            // SAFETY: as in `Events::new`.
            unsafe { check(cuEventRecord(e.0, s), "cuEventRecord")? };
            run(i)?;
            // SAFETY: as above.
            unsafe { check(cuEventRecord(e.1, s), "cuEventRecord")? };
        }
    }
    rig.sync()?;
    let mut out = [0.0; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let v: Result<Vec<f64>> = (0..REPS).map(|rep| ev.ms(rep * 3 + i)).collect();
        *o = median(v?) * 1e3;
    }
    Ok(out)
}

/// 2026-10-12: Capture the step graphs of A (k rows), B (k rows) and ONE (A at k = 1).
fn capture(rig: &Rig, k: usize) -> Result<[GraphHandle; 3]> {
    let (g, s) = (rig.g, rig.s);
    let mut out = [GraphHandle(0); 3];
    for (i, (arm, rows)) in [(Arm::PerRow, k), (Arm::Rows, k), (Arm::PerRow, 1)]
        .into_iter()
        .enumerate()
    {
        g.begin_capture(s)?;
        if let Err(e) = rig.step(arm, rows) {
            g.abort_capture_if_active(s);
            return Err(e);
        }
        out[i] = g.end_capture(s)?;
    }
    Ok(out)
}

/// 2026-10-12: Running verdict.
#[derive(Default)]
struct Tally {
    compares: usize,
    differ: usize,
    first: Option<String>,
    row_mixup_seen: bool,
    /// 2026-10-12: Rows whose layout (pool slots, tail base, tail keys) failed the host
    /// reference (`placement_error`), with the first one.
    placement_bad: usize,
    placement_first: Option<String>,
    wrong_geom_missed: Vec<String>,
}

/// 2026-10-12: Compare the outputs of the last A and B steps; returns (equal, A tokens).
fn compare(rig: &Rig, k: usize, label: &str, t: &mut Tally) -> Result<(bool, Vec<u8>)> {
    let ow = rig.cfg.out_width();
    let ta = rig.tokens(&rig.a, k)?;
    let tb = rig.tokens(&rig.b, k)?;
    let end = rig.seq + k;
    let eq = ta == tb && rig.pools(&rig.a, end)? == rig.pools(&rig.b, end)?;
    if let Some(r) = valid_counts(&ta, ow).iter().position(|&c| c == 0) {
        bail!("{label}: row {r} selected nothing");
    }
    // 2026-10-12: Control 1: a row mix-up must be visible to the compare.
    if (0..k - 1)
        .any(|r| ta[r * ow * 4..(r + 1) * ow * 4] != tb[(r + 1) * ow * 4..(r + 2) * ow * 4])
    {
        t.row_mixup_seen = true;
    }
    for (arm, tok) in [("A", &ta), ("B", &tb)] {
        if let Some(e) = placement_error(tok, rig.seq, ow) {
            t.placement_bad += 1;
            t.placement_first
                .get_or_insert_with(|| format!("{label} arm {arm}: {e}"));
        }
    }
    t.compares += 1;
    if !eq {
        t.differ += 1;
        t.first.get_or_insert_with(|| label.to_string());
    }
    Ok((eq, ta))
}

/// 2026-10-12: One (S, k) case: draws, control 2, timing.
fn case(rig: &mut Rig, s_ctx: usize, k: usize, t: &mut Tally) -> Result<()> {
    let seq = rig.seq;
    let graphs = capture(rig, k)?;
    let mut differ = 0;
    let mut counts = Vec::new();
    for d in 0..DRAWS {
        rig.draw(k)?;
        let label = format!("S={s_ctx} k={k} draw={d}");
        if d == 0 {
            rig.step(Arm::PerRow, k)?;
            rig.step(Arm::Rows, k)?;
            let (eq, _) = compare(rig, k, &format!("{label} eager"), t)?;
            differ += usize::from(!eq);
        }
        rig.g.launch_graph(graphs[0], rig.s)?;
        rig.g.launch_graph(graphs[1], rig.s)?;
        let (eq, ta) = compare(rig, k, &format!("{label} graph"), t)?;
        differ += usize::from(!eq);
        counts = valid_counts(&ta, rig.cfg.out_width());
    }
    // 2026-10-12: Control 2: B with the geometry from row 0's length must differ at k >= 3
    // (row 2 has a tail key at seq + 2 >= row 0's length).
    rig.step(Arm::RowsFirstLen, k)?;
    let wrong = rig.tokens(&rig.b, k)?;
    rig.step(Arm::PerRow, k)?;
    let ta = rig.tokens(&rig.a, k)?;
    let caught = wrong != ta;
    if k >= 3 && !caught {
        t.wrong_geom_missed.push(format!("S={s_ctx} k={k}"));
    }
    let selk: Vec<usize> = (0..k).map(|r| ((seq + r + 1) / KP).min(512)).collect();
    println!(
        "CASE S={s_ctx} seq={seq} k={k} draws={DRAWS} differ={differ} bitwise_equal={} \
         select_k_rows={selk:?} valid_counts={counts:?} wrong_geom_caught={caught}",
        if differ == 0 { "yes" } else { "NO" }
    );
    for (mode, gr) in [("graph", Some(graphs)), ("eager", None)] {
        let [a, b, one] = time_arms(rig, k, gr)?;
        println!(
            "TIMING S={s_ctx} k={k} mode={mode} A_us={a:.2} B_us={b:.2} ONE_us={one:.2} \
             B/A={:.3} A/k/ONE={:.3} B/ONE={:.3}",
            b / a,
            a / k as f64 / one,
            b / one
        );
    }
    for gr in graphs {
        rig.g.destroy_graph(gr)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    for (k, v) in LEVERS {
        // SAFETY: set before any thread exists and before any lever is first read.
        unsafe { std::env::set_var(k, v) };
    }
    println!("levers: {LEVERS:?}");
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let need = |k: &Glm5NextDsaKernels| {
        [
            k.kpool_compress_incr,
            k.write_geom_pk,
            k.indexer_store_ring,
            k.pk_len_clamp,
            k.index_scores_decode,
            k.topk_radix_init,
            k.topk_radix_compact,
            k.topk_radix_select0,
            k.grid_stride_marker,
        ]
        .iter()
        .all(|h| h.0 != 0)
    };
    let kernels = match Glm5NextDsaKernels::resolve(g) {
        Ok(k) if need(&k) => k,
        r => {
            let why = r
                .err()
                .map_or("a car entry point".into(), |e| e.to_string());
            println!("a DSA kernel is absent from this target ({why}) - SKIP");
            std::process::exit(2);
        }
    };
    let mut rig = Rig::new(g, kernels, 0xD5A_2026_1012)?;
    let mut t = Tally::default();
    for &s_ctx in CONTEXTS {
        rig.fill_to(s_ctx - 2)?;
        for &k in KS {
            case(&mut rig, s_ctx, k, &mut t)?;
        }
    }
    rig.free()?;
    println!(
        "CONTROL row mix-up visible={}; wrong shared geometry caught at every k>=3 case={} \
         (missed: {:?})",
        t.row_mixup_seen,
        t.wrong_geom_missed.is_empty(),
        t.wrong_geom_missed
    );
    let controls = t.row_mixup_seen && t.wrong_geom_missed.is_empty();
    println!(
        "LAYOUT {} of {} arm outputs failed the host layout reference (first: {})",
        t.placement_bad,
        2 * t.compares,
        t.placement_first.as_deref().unwrap_or("none")
    );
    if t.differ == 0 && t.placement_bad == 0 && controls && t.compares > 0 {
        println!(
            "PASS dsa_ceiling_rows: {} step compares byte-equal (tokens, pools, watermark), \
             controls fired",
            t.compares
        );
        Ok(())
    } else {
        println!(
            "FAIL dsa_ceiling_rows: {} of {} compares differ (first: {}), {} layout \
             failures, controls fired {controls}",
            t.differ,
            t.compares,
            t.first.unwrap_or_else(|| "none".into()),
            t.placement_bad
        );
        std::process::exit(1);
    }
}
