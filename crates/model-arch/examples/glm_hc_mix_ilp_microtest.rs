// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Microtest for the `METRALE_GLM_HC_MIX_ILP` variants of the decode mHC mix
//! (`glm5next_hc_mix_bf16`, race-pf-mhcilp).
//!
//! - ILP variants `glm5next_hc_mix_bf16_ilp{8,16,32,64}`: BITWISE equal to production. For rows
//!   R in {1,2,3,5,8,9,16}, three highway kinds (normal, heavy-tailed, large-magnitude rows) and
//!   random BF16 `hc_fn` [24, 16384] (normal and heavy-tailed), the `mix_out` u32 bit patterns
//!   are compared: one `PASS`/`FAIL` line per (variant, R).
//! - Split-K variants `glm5next_hc_mix_bf16_splitk{4,8,16}`: NOT bitwise; tolerance lines per
//!   (variant, R) against an FP64 host reference: rel err <= 1e-5 and 0 nonfinite. rel err is
//!   `|got - ref| / max(|ref|, 0.1 * rms(ref over the row's 24 outputs))` (a plain relative
//!   error is meaningless where the dot is near zero). Production's max rel err vs the same
//!   reference is printed for comparison. The split-K kernel is launched twice per case to prove
//!   its in-kernel counter reset (the second launch must give the same bits as the first).
//! - Speed (cold): 96 distinct `hc_fn` copies (75.5 MB, beyond the 24 MB L2) and a matching set
//!   of highways; one CUDA graph of 90 back-to-back launches per arm (launch i uses copy i), the
//!   median of 21 replays; per-launch us and implied hc_fn GB/s (VOID above 255) at R = 1, 3, 8.
//! - Kernel names resolve through `Glm5NextMhcKernels::resolve`, the production resolver, so the
//!   module name (`glm5next_mhc`, a stem not listed in KERNEL.toml `[modules]`) is production's.
//!
//! Final line `PASS all` only if every check passed. Run via build-mt with
//! `EXAMPLES=glm_hc_mix_ilp_microtest` (features cuda,gpu-examples); GB10 only.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{
    GLM5NEXT_MHC_MODULE, Glm5NextMhcKernels, MHC_SPLITK_SCRATCH_BYTES,
};

const H: usize = 4096;
const HC: usize = 4;
const HCD: usize = H * HC;
const MIX: usize = 24;
const NORM_EPS: f32 = 1e-5;
const BIT_ROWS: [usize; 7] = [1, 2, 3, 5, 8, 9, 16];
const TIME_ROWS: [usize; 3] = [1, 3, 8];
const ILP_N: [u32; 4] = [8, 16, 32, 64];
const SPLITK_S: [u32; 3] = [4, 8, 16];
const NCOPY: usize = 96;
const SITES: usize = 90;
const REPLAYS: usize = 21;
const MAX_ROWS: usize = 16;
const TOL: f64 = 1e-5;
const FN_BYTES: usize = MIX * HCD * 2;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32) as f32 / (1u32 << 24) as f32
    }
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
    /// Heavy-tailed: cube of a symmetric value, occasionally x1e4.
    fn heavy(&mut self, a: f32) -> f32 {
        let v = self.sym(1.0);
        let big = if self.unit() < 0.002 { 1e4 } else { 1.0 };
        v * v * v * a * big
    }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bf16s(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect()
}
fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}
fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One mix launch: grid (rows, 24, z); `scratch` is the split-K scratch (an extra argument).
#[allow(clippy::too_many_arguments)]
fn l_mix(
    g: &dyn GpuBackend,
    k: KernelHandle,
    hw: DevicePtr,
    fnw: DevicePtr,
    out: DevicePtr,
    scratch: Option<DevicePtr>,
    rows: usize,
    z: u32,
    stream: u64,
) -> Result<()> {
    let l = KernelLaunch::new(g, k)
        .grid([rows as u32, MIX as u32, z])
        .block([256, 1, 1])
        .arg_ptr(hw)
        .arg_ptr(fnw)
        .arg_ptr(out);
    let l = match scratch {
        Some(s) => l.arg_ptr(s),
        None => l,
    };
    l.arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_f32(NORM_EPS)
        .launch(stream)
}

/// FP64 reference of the mix: out[t][m] = sum_k w * x / sqrt(sum x^2 / n + eps).
fn reference(hw: &[f32], fnw: &[f32], rows: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * MIX];
    for t in 0..rows {
        let x = &hw[t * HCD..(t + 1) * HCD];
        let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
        let rs = 1.0 / (ss / HCD as f64 + NORM_EPS as f64).sqrt();
        for m in 0..MIX {
            let w = &fnw[m * HCD..(m + 1) * HCD];
            let d: f64 = x.iter().zip(w).map(|(&a, &b)| a as f64 * b as f64).sum();
            out[t * MIX + m] = d * rs;
        }
    }
    out
}

/// (max rel err, nonfinite count) of `got` against `want`, per the module doc's definition.
fn rel_err(got: &[f32], want: &[f64], rows: usize) -> (f64, usize) {
    let (mut worst, mut nonfin) = (0f64, 0usize);
    for t in 0..rows {
        let r = &want[t * MIX..(t + 1) * MIX];
        let rms = (r.iter().map(|v| v * v).sum::<f64>() / MIX as f64).sqrt();
        for m in 0..MIX {
            let g = got[t * MIX + m];
            if !g.is_finite() {
                nonfin += 1;
                continue;
            }
            let d = (g as f64 - r[m]).abs() / r[m].abs().max(0.1 * rms).max(1e-300);
            worst = worst.max(d);
        }
    }
    (worst, nonfin)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Capture `f`, replay it `REPLAYS` times between events, median ms per replay.
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
    let mut v = Vec::with_capacity(REPLAYS);
    for _ in 0..REPLAYS {
        let (mut e0, mut e1, mut ms) = (0u64, 0u64, 0f32);
        unsafe {
            if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
                bail!("cuEventCreate failed");
            }
            cuEventRecord(e0, s);
        }
        g.launch_graph(graph, s)?;
        unsafe {
            cuEventRecord(e1, s);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        v.push(ms as f64);
    }
    g.destroy_graph(graph)?;
    Ok(median(&mut v))
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        bail!("glm_hc_mix_ilp_microtest: no glm-5.3-flash PTX set in this build");
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let ks = Glm5NextMhcKernels::resolve(g)?;
    println!("module {GLM5NEXT_MHC_MODULE}");
    if ks.hc_mix_bf16.0 == 0 {
        bail!("glm5next_hc_mix_bf16 not linked");
    }
    for (i, h) in ks.hc_mix_bf16_ilp.iter().enumerate() {
        if h.0 == 0 {
            bail!("ilp{} handle not resolved", ILP_N[i]);
        }
    }
    for (i, h) in ks.hc_mix_bf16_splitk.iter().enumerate() {
        if h.0 == 0 {
            bail!("splitk{} handle not resolved", SPLITK_S[i]);
        }
    }

    let hw_d = g.alloc(MAX_ROWS * HCD * 4)?;
    let out_ref = g.alloc(MAX_ROWS * MIX * 4)?;
    let out_var = g.alloc(MAX_ROWS * MIX * 4)?;
    let scratch = g.alloc(MIX * 4 + MHC_SPLITK_SCRATCH_BYTES)?;
    g.memset(scratch, 0, MIX * 4 + MHC_SPLITK_SCRATCH_BYTES)?;
    let fn_d = g.alloc(FN_BYTES)?;

    let mut fails = 0usize;
    let mut checks = 0usize;
    // (variant name) -> per-R accumulators, in print order.
    for &rows in &BIT_ROWS {
        let mut ilp_bad = [0usize; 4];
        let mut ilp_cmp = 0usize;
        let mut sk_worst = [0f64; 3];
        let mut sk_nonfin = [0usize; 3];
        let mut sk_reset_bad = [0usize; 3];
        let mut prod_worst = 0f64;
        for (case, (hw_kind, fn_kind)) in [
            ("normal", "normal"),
            ("heavy", "normal"),
            ("big_rows", "heavy"),
            ("normal", "heavy"),
        ]
        .into_iter()
        .enumerate()
        {
            let mut rng = Lcg(0xC0FFEE ^ (rows as u64 * 7919) ^ ((case as u64 + 1) << 32));
            let hw: Vec<f32> = (0..rows * HCD)
                .map(|n| match hw_kind {
                    "normal" => rng.sym(8.0),
                    "heavy" => rng.heavy(30.0),
                    // large-magnitude: every third row ~1e12, others normal
                    _ => {
                        if (n / HCD).is_multiple_of(3) {
                            rng.sym(1e12)
                        } else {
                            rng.sym(8.0)
                        }
                    }
                })
                .collect();
            let fnw: Vec<f32> = (0..MIX * HCD)
                .map(|_| {
                    if fn_kind == "normal" {
                        rng.sym(0.05)
                    } else {
                        rng.heavy(0.5)
                    }
                })
                .collect();
            // The device holds BF16; the reference uses the same rounded weights.
            let fnw_r: Vec<f32> = fnw
                .iter()
                .map(|&v| half::bf16::from_f32(v).to_f32())
                .collect();
            g.copy_h2d(&f32s(&hw), hw_d)?;
            g.copy_h2d(&bf16s(&fnw), fn_d)?;
            g.memset_async(out_ref, 0xEE, MAX_ROWS * MIX * 4, 0)?;
            l_mix(g, ks.hc_mix_bf16, hw_d, fn_d, out_ref, None, rows, 1, 0)?;
            let prod = words(&down(g, out_ref, rows * MIX * 4)?);
            ilp_cmp += 1;
            for (i, &h) in ks.hc_mix_bf16_ilp.iter().enumerate() {
                g.memset_async(out_var, 0xEE, MAX_ROWS * MIX * 4, 0)?;
                l_mix(g, h, hw_d, fn_d, out_var, None, rows, 1, 0)?;
                let got = words(&down(g, out_var, rows * MIX * 4)?);
                if got != prod {
                    ilp_bad[i] += 1;
                }
            }
            // Split-K vs FP64 (only where the production result is finite: huge rows can
            // overflow the FP32 sum of squares, and there the FP64 reference is not comparable).
            let want = reference(&hw, &fnw_r, rows);
            let prod_f: Vec<f32> = prod.iter().map(|&b| f32::from_bits(b)).collect();
            let prod_finite = prod_f.iter().all(|v| v.is_finite());
            if prod_finite {
                prod_worst = prod_worst.max(rel_err(&prod_f, &want, rows).0);
            }
            for (i, &h) in ks.hc_mix_bf16_splitk.iter().enumerate() {
                let s = SPLITK_S[i];
                g.memset_async(out_var, 0xEE, MAX_ROWS * MIX * 4, 0)?;
                l_mix(g, h, hw_d, fn_d, out_var, Some(scratch), rows, s, 0)?;
                let a = words(&down(g, out_var, rows * MIX * 4)?);
                g.memset_async(out_var, 0xEE, MAX_ROWS * MIX * 4, 0)?;
                l_mix(g, h, hw_d, fn_d, out_var, Some(scratch), rows, s, 0)?;
                let b = words(&down(g, out_var, rows * MIX * 4)?);
                if a != b {
                    sk_reset_bad[i] += 1;
                }
                if prod_finite {
                    let af: Vec<f32> = a.iter().map(|&w| f32::from_bits(w)).collect();
                    let (w, nf) = rel_err(&af, &want, rows);
                    sk_worst[i] = sk_worst[i].max(w);
                    sk_nonfin[i] += nf;
                }
            }
        }
        for (i, &n) in ILP_N.iter().enumerate() {
            checks += 1;
            let ok = ilp_bad[i] == 0;
            println!(
                "{} ilp{n} R={rows} bitwise vs production: {} of {ilp_cmp} cases differ",
                if ok { "PASS" } else { "FAIL" },
                ilp_bad[i]
            );
            if !ok {
                fails += 1;
            }
        }
        for (i, &s) in SPLITK_S.iter().enumerate() {
            checks += 1;
            let ok = sk_worst[i] <= TOL && sk_nonfin[i] == 0 && sk_reset_bad[i] == 0;
            println!(
                "{} splitk{s} R={rows} tolerance vs FP64: max_rel={:.3e} (production vs FP64 \
                 max_rel={prod_worst:.3e}) nonfinite={} counter_reset_mismatch={} (limit {TOL:e})",
                if ok { "PASS" } else { "FAIL" },
                sk_worst[i],
                sk_nonfin[i],
                sk_reset_bad[i]
            );
            if !ok {
                fails += 1;
            }
        }
    }

    // Cold-pool speed.
    let mut rng = Lcg(0x71E);
    let fn_pool = g.alloc(NCOPY * FN_BYTES)?;
    let one: Vec<f32> = (0..MIX * HCD).map(|_| rng.sym(0.05)).collect();
    let one_b = bf16s(&one);
    for c in 0..NCOPY {
        // Distinct bytes per copy so no copy aliases another in any cache or dedup.
        let mut b = one_b.clone();
        b[0] = c as u8;
        g.copy_h2d(&b, fn_pool.offset(c * FN_BYTES))?;
    }
    let hw_pool = g.alloc(NCOPY * MAX_ROWS * HCD * 4)?;
    let hwv: Vec<f32> = (0..MAX_ROWS * HCD).map(|_| rng.sym(8.0)).collect();
    let hwb = f32s(&hwv);
    for c in 0..NCOPY {
        g.copy_h2d(&hwb, hw_pool.offset(c * MAX_ROWS * HCD * 4))?;
    }
    let out_pool = g.alloc(SITES * MAX_ROWS * MIX * 4)?;
    g.synchronize(0)?;
    let s = g.create_stream()?;
    println!(
        "TIMING cold: {NCOPY} hc_fn copies = {:.1} MB, graph of {SITES} launches, median of {REPLAYS} replays",
        (NCOPY * FN_BYTES) as f64 / 1e6
    );
    for &rows in &TIME_ROWS {
        let mut arms: Vec<(String, KernelHandle, u32, bool)> =
            vec![("production".into(), ks.hc_mix_bf16, 1, false)];
        for (i, &n) in ILP_N.iter().enumerate() {
            arms.push((format!("ilp{n}"), ks.hc_mix_bf16_ilp[i], 1, false));
        }
        for (i, &sk) in SPLITK_S.iter().enumerate() {
            arms.push((format!("splitk{sk}"), ks.hc_mix_bf16_splitk[i], sk, true));
        }
        let mut base = 0f64;
        for (name, h, z, sc) in arms {
            let ms = time_graph(g, s, &mut |st| {
                for i in 0..SITES {
                    l_mix(
                        g,
                        h,
                        hw_pool.offset(i * MAX_ROWS * HCD * 4),
                        fn_pool.offset(i * FN_BYTES),
                        out_pool.offset(i * MAX_ROWS * MIX * 4),
                        if sc { Some(scratch) } else { None },
                        rows,
                        z,
                        st,
                    )?;
                }
                Ok(())
            })?;
            let us = ms * 1000.0 / SITES as f64;
            if name == "production" {
                base = us;
            }
            let gbs = FN_BYTES as f64 / (us * 1e3);
            let flag = if gbs > 255.0 { " VOID" } else { "" };
            println!(
                "SPEED R={rows} {name} us={us:.2} hc_fn_GBs={gbs:.0}{flag} vs_prod={:.2}x",
                base / us
            );
        }
    }

    if fails != 0 {
        println!("FAIL: {fails} of {checks} check(s) did not hold");
        bail!("glm_hc_mix_ilp_microtest: FAIL");
    }
    println!("PASS all ({checks} checks)");
    Ok(())
}
