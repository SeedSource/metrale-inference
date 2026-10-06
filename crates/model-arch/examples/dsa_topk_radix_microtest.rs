// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: GPU gate for `METRALE_GLM_DSA_TOPK_RADIX`: the exact radix top-k
//! (`dsa_topk_radix_*`, launched through `select::radix::launch_topk_radix`, the function
//! `select_tokens` calls) against the current `dsa_topk_pools`, byte for byte, plus their time
//! per call and per decode step at long context. One GB10 GPU.
//!
//! * Depth sets: rig-shaped scores (per pool, -FLT_MAX for about 4 in 23 pools, the rate an
//!   invalid token every 23rd gives at `index_kpool` 4, as `dsa_depth_rig`; otherwise a sum of 8
//!   `w * relu(x)` terms, so exact +0.0 occurs) at S = 131,072 / 262,144 / 532,480 tokens
//!   (P = S / 4 pools), 11 layers x 3 verify rows, each row its own seed, select_k 512.
//! * Adversarial sets (3 rows each): all scores equal; all -FLT_MAX; about half -FLT_MAX; +-0.0
//!   mixes, with and without the threshold inside the zeros; exactly select_k finite candidates
//!   and the rest -FLT_MAX; P = select_k; P = select_k + 1; P = 2049 (just above one top-k
//!   tile); P not a multiple of the chunk width (100,003); heavy duplicates at the threshold;
//!   denormals of both signs; a wide spread of both signs.
//! * Launch modes, both kernels the same way: exact (scalars from the host, `geom` NULL, all
//!   rows in one launch) and ceiling (the scalars `select_tokens` derives from a ceiling of
//!   135,168 pools, the production 512K row, with the live P and select_k read from a device
//!   `geom` holding what `dsa_write_geom` writes for S = 4 P; one launch per row, `q_rows` 1).
//!   Each output buffer is filled with a poison byte first. The verdict compares radix against
//!   `dsa_topk_pools` as raw bytes, mode for mode; every entry must also be a pool index in
//!   [0, P). A host sort (score descending, -0.0 == +0.0, then index ascending) is reported as
//!   `host_ref` for information.
//! * Timing (CUDA events around graph replays, median of 51 after 3 warm-ups): per decode step,
//!   33 ceiling calls (11 layers x 3 rows of `q_rows` 1, the production decode shape) and 11
//!   exact 3-row calls; per call, one ceiling call. A speedup under 10x at 532K misses the
//!   design note's bar (`MISS_BAR` line); timing never decides the verdict.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 when every compare is byte-equal, with a final `PASS: dsa_topk_radix_microtest ...`
//! line; 1 with a `FAIL` line otherwise; 2 with a `FAIL` line when the radix entry points are
//! absent from this target.
//!
//! Run (gb10 common kernels):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_topk_radix_microtest
//! Needs well under 1 GB of device memory.

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;
use metrale_model_arch::glm5next_dsa::select::radix::{
    RadixTopk, launch_topk_radix, radix_chunks, radix_resolved, radix_row_bytes,
};
use metrale_model_arch::glm5next_dsa::select::{topk_smem_for_tile, topk_tile};

#[path = "common/dsa_topk_radix_common.rs"]
mod dsa_topk_radix_common;
use dsa_topk_radix_common::{SELECT_K, adversarial, host_reference, rig_scores, time_graph};

/// 2026-10-06: The radix candidate capacity (`radix_kcap` of the GLM-5.3 config).
const KCAP: usize = 512;
/// 2026-10-06: GLM-5.3 `index_kpool`.
const KP: usize = 4;
/// 2026-10-06: Ceiling of every ceiling launch: the pools of the production 512K row
/// (`--max-seq-len` 540,672).
const CEILING_POOLS: usize = 540_672 / KP;
/// 2026-10-06: DSA text layers and verify rows per decode step at K = 3.
const LAYERS: usize = 11;
const ROWS: usize = 3;
/// 2026-10-06: Live contexts timed (tokens).
const CONTEXTS: &[usize] = &[131_072, 262_144, 532_480];
/// 2026-10-06: `ROW_BLOCK` in `glm5next_dsa/select.rs` (private there), the threads per
/// `dsa_topk_pools` block `select_tokens` launches.
const ROW_BLOCK: u32 = 256;
/// 2026-10-06: Fill byte of every output before a launch.
const POISON: u8 = 0xA5;
/// 2026-10-06: The design note's miss bar: under 10x at 532K stops the lane.
const MISS_BAR: f64 = 10.0;

/// 2026-10-06: Little-endian bytes of `v`.
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-10-06: Little-endian bytes of `v`.
fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-10-06: Upload `bytes` to a new device buffer (at least 4 bytes).
fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(4))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

/// 2026-10-06: Download `n` i32 after a device synchronise.
fn down_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect())
}

/// 2026-10-06: `dsa_write_geom`'s top-k tile width for `p` complete pools.
fn np2_of(p: usize) -> usize {
    p.next_power_of_two().max(2).min(topk_tile())
}

/// 2026-10-06: `dsa_write_geom`'s select_k for `p` pools.
fn select_k_of(p: usize) -> usize {
    SELECT_K.min(p)
}

/// 2026-10-06: The `[5]` geometry `dsa_write_geom` writes for S = KP * p tokens (no partial
/// pool): S, pools including a partial one, complete pools, select_k, tile width.
fn write_geom(g: &dyn GpuBackend, geom: DevicePtr, p: usize) -> Result<()> {
    let v = [
        (KP * p) as i32,
        p as i32,
        p as i32,
        select_k_of(p) as i32,
        np2_of(p) as i32,
    ];
    g.copy_h2d(&i32_bytes(&v), geom)
}

/// 2026-10-06: How a call is launched: host scalars, or the ceiling scalars plus `geom`.
#[derive(Clone, Copy)]
enum Mode {
    Exact,
    Ceiling(DevicePtr),
}

/// 2026-10-06: One top-k call: `q_rows` rows of `p` pools at `scores` (row stride `p`) into
/// `selected` (row stride `select_k_of(p)`).
#[derive(Clone, Copy)]
struct Call {
    scores: DevicePtr,
    selected: DevicePtr,
    q_rows: usize,
    p: usize,
}

/// 2026-10-06: `(P, select_k, geom)` arguments of a call: the host's, or those `select_tokens`
/// derives from the ceiling (`select/launch.rs`), with the live values in `geom`.
fn scalars(c: &Call, mode: Mode) -> (usize, usize, DevicePtr) {
    match mode {
        Mode::Exact => (c.p, select_k_of(c.p), DevicePtr::NULL),
        Mode::Ceiling(gd) => (CEILING_POOLS, select_k_of(CEILING_POOLS), gd),
    }
}

/// 2026-10-06: `dsa_topk_pools` as `select_tokens` launches it.
fn old_topk(
    g: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    c: &Call,
    mode: Mode,
    stream: u64,
) -> Result<()> {
    let (p, selk, gd) = scalars(c, mode);
    let np2 = np2_of(p);
    KernelLaunch::new(g, k.topk_pools)
        .grid([c.q_rows as u32, 1, 1])
        .block([ROW_BLOCK, 1, 1])
        .shared_mem(topk_smem_for_tile(np2) as u32)
        .arg_ptr(c.scores)
        .arg_ptr(c.selected)
        .arg_u32(c.q_rows as u32)
        .arg_u32(p as u32)
        .arg_u32(np2 as u32)
        .arg_u32(selk as u32)
        .arg_ptr(gd)
        .launch(stream)
}

/// 2026-10-06: The radix path through the production launcher.
fn radix_topk(
    g: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    c: &Call,
    work: DevicePtr,
    mode: Mode,
    stream: u64,
) -> Result<()> {
    let (p, selk, gd) = scalars(c, mode);
    let t = RadixTopk {
        scores: c.scores,
        selected: c.selected,
        work,
        q_rows: c.q_rows,
        n_pools: p,
        select_k: selk,
        kcap: KCAP,
        geom_dev: gd,
    };
    launch_topk_radix(g, k, &t, stream)
}

/// 2026-10-06: Running verdict.
#[derive(Default)]
struct Tally {
    compares: usize,
    equal: usize,
    bad_entries: usize,
    host_ref_differs: usize,
}

impl Tally {
    /// 2026-10-06: Compare one radix output with the `dsa_topk_pools` one; count entries
    /// outside [0, p) (poison, -1). Returns whether they are byte-equal.
    fn compare(&mut self, what: &str, old: &[i32], radix: &[i32], p: usize) -> bool {
        self.compares += 1;
        let bad = |v: &[i32]| v.iter().filter(|&&x| x < 0 || x as usize >= p).count();
        self.bad_entries += bad(old) + bad(radix);
        let eq = old == radix;
        if eq {
            self.equal += 1;
        } else if let Some(i) = old.iter().zip(radix).position(|(a, b)| a != b) {
            println!(
                "  DIFF {what}: first at slot {i} (row {}, rank {}): dsa_topk_pools {} radix {}",
                i / SELECT_K.min(p).max(1),
                i % SELECT_K.min(p).max(1),
                old[i],
                radix[i]
            );
        }
        eq
    }
}

/// 2026-10-06: Shared device buffers of the run.
struct Bufs {
    work: DevicePtr,
    geom: DevicePtr,
}

/// 2026-10-06: One adversarial set: `rows` (each `p` pools) through both kernels, exact and
/// ceiling.
fn run_case(
    g: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    b: &Bufs,
    name: &str,
    rows: &[Vec<f32>],
    tally: &mut Tally,
) -> Result<()> {
    let (q, p) = (rows.len(), rows[0].len());
    let sk = select_k_of(p);
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let scores = up(g, &f32_bytes(&flat))?;
    let out_bytes = (q * sk * 4).max(4);
    let outs = (0..4)
        .map(|_| g.alloc(out_bytes))
        .collect::<Result<Vec<_>>>()?;
    for &o in &outs {
        g.memset(o, POISON, out_bytes)?;
    }
    write_geom(g, b.geom, p)?;
    let exact = |selected| Call {
        scores,
        selected,
        q_rows: q,
        p,
    };
    old_topk(g, k, &exact(outs[0]), Mode::Exact, 0)?;
    radix_topk(g, k, &exact(outs[1]), b.work, Mode::Exact, 0)?;
    for r in 0..q {
        let row = |selected: DevicePtr| Call {
            scores: scores.offset(r * p * 4),
            selected: selected.offset(r * sk * 4),
            q_rows: 1,
            p,
        };
        old_topk(g, k, &row(outs[2]), Mode::Ceiling(b.geom), 0)?;
        radix_topk(g, k, &row(outs[3]), b.work, Mode::Ceiling(b.geom), 0)?;
    }
    let v = outs
        .iter()
        .map(|&o| down_i32(g, o, q * sk))
        .collect::<Result<Vec<_>>>()?;
    let ex = tally.compare(&format!("{name} exact"), &v[0], &v[1], p);
    let ce = tally.compare(&format!("{name} ceiling"), &v[2], &v[3], p);
    let host: Vec<i32> = rows.iter().flat_map(|r| host_reference(r, sk)).collect();
    let agree = host == v[0];
    tally.host_ref_differs += usize::from(!agree);
    println!(
        "CASE {name}: P={p} rows={q} select_k={sk} exact={} ceiling={} host_ref={}",
        if ex { "EQUAL" } else { "DIFF" },
        if ce { "EQUAL" } else { "DIFF" },
        if agree { "agrees" } else { "differs" }
    );
    for o in outs.into_iter().chain([scores]) {
        g.free(o)?;
    }
    Ok(())
}

/// 2026-10-06: One live context `s`: 11 layers x 3 rows of rig-shaped scores; correctness in both
/// modes, then the timings. Returns the ceiling speedup per decode step (old / radix).
fn run_depth(
    g: &dyn GpuBackend,
    k: &Glm5NextDsaKernels,
    b: &Bufs,
    s: usize,
    st: u64,
    tally: &mut Tally,
) -> Result<f64> {
    let p = s / KP;
    let n = LAYERS * ROWS;
    let mut flat = Vec::with_capacity(n * p);
    for i in 0..n {
        flat.extend(rig_scores(((s as u64) << 8) | i as u64, p));
    }
    let scores = up(g, &f32_bytes(&flat))?;
    let out_bytes = n * SELECT_K * 4;
    let outs = (0..4)
        .map(|_| g.alloc(out_bytes))
        .collect::<Result<Vec<_>>>()?;
    for &o in &outs {
        g.memset(o, POISON, out_bytes)?;
    }
    write_geom(g, b.geom, p)?;
    // 2026-10-06: Call `i` of a step is layer `i / 3`, row `i % 3`: scores row `i`, selected
    // row `i`. An exact call covers a layer's three rows in one launch, same layout.
    let row_call = |i: usize, out: DevicePtr| Call {
        scores: scores.offset(i * p * 4),
        selected: out.offset(i * SELECT_K * 4),
        q_rows: 1,
        p,
    };
    let layer_call = |l: usize, out: DevicePtr| Call {
        scores: scores.offset(l * ROWS * p * 4),
        selected: out.offset(l * ROWS * SELECT_K * 4),
        q_rows: ROWS,
        p,
    };
    let ceil = Mode::Ceiling(b.geom);
    for i in 0..n {
        old_topk(g, k, &row_call(i, outs[0]), ceil, 0)?;
        radix_topk(g, k, &row_call(i, outs[1]), b.work, ceil, 0)?;
    }
    for l in 0..LAYERS {
        old_topk(g, k, &layer_call(l, outs[2]), Mode::Exact, 0)?;
        radix_topk(g, k, &layer_call(l, outs[3]), b.work, Mode::Exact, 0)?;
    }
    let v = outs
        .iter()
        .map(|&o| down_i32(g, o, n * SELECT_K))
        .collect::<Result<Vec<_>>>()?;
    let ce = tally.compare(&format!("S={s} ceiling"), &v[0], &v[1], p);
    let ex = tally.compare(&format!("S={s} exact"), &v[2], &v[3], p);
    let host: Vec<i32> = flat
        .chunks_exact(p)
        .flat_map(|r| host_reference(r, SELECT_K))
        .collect();
    let agree = host == v[0];
    tally.host_ref_differs += usize::from(!agree);
    println!(
        "DEPTH_CHECK S={s} P={p} {n} rows: ceiling={} exact={} host_ref={}",
        if ce { "EQUAL" } else { "DIFF" },
        if ex { "EQUAL" } else { "DIFF" },
        if agree { "agrees" } else { "differs" }
    );

    let old_step = time_graph(g, st, &mut |sm| {
        (0..n).try_for_each(|i| old_topk(g, k, &row_call(i, outs[0]), ceil, sm))
    })?;
    let rad_step = time_graph(g, st, &mut |sm| {
        (0..n).try_for_each(|i| radix_topk(g, k, &row_call(i, outs[1]), b.work, ceil, sm))
    })?;
    let old_call = time_graph(g, st, &mut |sm| {
        old_topk(g, k, &row_call(0, outs[0]), ceil, sm)
    })?;
    let rad_call = time_graph(g, st, &mut |sm| {
        radix_topk(g, k, &row_call(0, outs[1]), b.work, ceil, sm)
    })?;
    let old_ex = time_graph(g, st, &mut |sm| {
        (0..LAYERS).try_for_each(|l| old_topk(g, k, &layer_call(l, outs[2]), Mode::Exact, sm))
    })?;
    let rad_ex = time_graph(g, st, &mut |sm| {
        (0..LAYERS)
            .try_for_each(|l| radix_topk(g, k, &layer_call(l, outs[3]), b.work, Mode::Exact, sm))
    })?;
    let speedup = old_step / rad_step.max(1e-9);
    println!(
        "TIMING S={s} P={p}: per decode step ({n} ceiling calls of 1 row) dsa_topk_pools \
         {old_step:.3} ms, radix {rad_step:.3} ms, {speedup:.1}x; per call {old_call:.4} ms vs \
         {rad_call:.4} ms; exact ({LAYERS} calls of {ROWS} rows) {old_ex:.3} ms vs {rad_ex:.3} ms"
    );
    for o in outs.into_iter().chain([scores]) {
        g.free(o)?;
    }
    Ok(speedup)
}

/// 2026-10-06: Adversarial sets, then the three depths; verdict last.
fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextDsaKernels::resolve(g)?;
    if !radix_resolved(&k) {
        println!(
            "FAIL: dsa_topk_radix_microtest - the dsa_topk_radix_* entry points are absent from \
             this target's dsa_indexer module"
        );
        std::process::exit(2);
    }
    println!(
        "ceiling {CEILING_POOLS} pools; select_k {SELECT_K}; radix chunks per row: {} (1 row), \
         {} ({ROWS} rows); work {} B per row",
        radix_chunks(1),
        radix_chunks(ROWS),
        radix_row_bytes(KCAP)
    );
    let b = Bufs {
        work: g.alloc(ROWS * radix_row_bytes(KCAP))?,
        geom: g.alloc(5 * 4)?,
    };
    // 2026-10-06: Stale work-buffer contents must not matter: start from the poison byte.
    g.memset(b.work, POISON, ROWS * radix_row_bytes(KCAP))?;
    let mut tally = Tally::default();

    for (ci, (name, make)) in adversarial().into_iter().enumerate() {
        let rows: Vec<Vec<f32>> = (0..ROWS)
            .map(|r| make(0xC0FF_EE00 + (ci * ROWS + r) as u64))
            .collect();
        run_case(g, &k, &b, name, &rows, &mut tally)?;
    }

    let st = g.create_stream()?;
    let mut last_speedup = 0.0;
    for &s in CONTEXTS {
        last_speedup = run_depth(g, &k, &b, s, st, &mut tally)?;
    }
    if last_speedup < MISS_BAR {
        println!(
            "MISS_BAR: radix is {last_speedup:.1}x dsa_topk_pools per decode step at S={}, under \
             the design note's {MISS_BAR:.0}x bar",
            CONTEXTS[CONTEXTS.len() - 1]
        );
    }
    g.free(b.work)?;
    g.free(b.geom)?;

    println!(
        "{}/{} compares byte-equal; {} entries outside [0, P); host_ref differs in {} sets \
         (information only)",
        tally.equal, tally.compares, tally.bad_entries, tally.host_ref_differs
    );
    if tally.compares == 0 || tally.equal != tally.compares || tally.bad_entries != 0 {
        println!(
            "FAIL: dsa_topk_radix_microtest - radix and dsa_topk_pools differ, or wrote entries \
             that are not pool indices"
        );
        std::process::exit(1);
    }
    println!(
        "PASS: dsa_topk_radix_microtest - {} compares byte-equal ({} adversarial sets and {} \
         depths, exact and ceiling launches)",
        tally.compares,
        adversarial().len(),
        CONTEXTS.len()
    );
    Ok(())
}
