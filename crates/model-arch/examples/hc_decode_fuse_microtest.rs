// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Gate-0 microtest for the decode-site mHC fusions of race-prefill
//! (SeedHQ/seed-skills#80): bit identity and timing of three new kernels against production.
//!
//! - E1 `glm5next_hc_finish_warp` vs `glm5next_hc_finish`: y, post, comb from the same `mix`.
//! - E2 `glm5next_hc_mix_bf16_1pass` vs `glm5next_hc_mix_bf16`: mix.
//! - E3 `glm5next_hc_mix_finish_1pass` vs mix + finish: mix, y, post, comb in one launch.
//! - Ablation G0a: production `glm5next_hc_finish` already takes `sinkhorn_iters`; it is timed at
//!   1 and 20 (no new kernel); output not compared.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - Shapes are the production decode call's: `glm_hc_pre_sliced_mix` launches the BF16 mix on grid
//!   `(rows, mix_hc = 24)` and `hc_finish` on grid `(rows, 1 + ceil(4096 / 256) = 17)`, block 256,
//!   hidden 4096, hc_mult 4, 20 Sinkhorn iterations, norm_eps 1e-5, hc_eps 1e-6 (the constants of
//!   `hc_post_mix_fast_bitparity_microtest`); the post is `glm_hc_post`.
//! - Every output buffer is compared as raw bits (bytes), after being pre-filled with 0xEE; a case
//!   FAILs on any differing byte. The run FAILs if the number of compared cases is not the
//!   expected one, or if the known-bad (E1 at 19 iterations) is not detected.
//! - Cases: rows {1, 2, 4, 8, 16} x seeds {1, 2, 3}, plus edge inputs at rows 1 and 4: zero,
//!   NaN, +inf, huge, subnormal highways, and `mix` rows of extreme / NaN / inf values with large
//!   scales (the mHC microtests in the tree carry no edge set, so this one is new).
//! - Lines: `HCDEC_CMP rows=R seed=S kernel=K bytes_differ=N`, `HCDEC_T rows=R kernel=K us=X`,
//!   `HCDEC_CUT ...`, final `PASS` / `FAIL`.
//! - Timing: warm (same buffers, the highway L2-resident as after the previous post), per sample
//!   `INNER` back-to-back launches between two events, `SAMPLES` samples, median per launch. The
//!   CUT chains launch mix, finish, post in stream order.
//!
//! Run: `EXAMPLES=hc_decode_fuse_microtest` via build-mt; `cargo run -p metrale-model-arch
//! --release --features cuda,gpu-examples --example hc_decode_fuse_microtest`.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{GLM5NEXT_MHC_MODULE, Glm5NextMhcKernels, glm_hc_post};

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = 24;
const SINKHORN: u32 = 20;
const NORM_EPS: f32 = 1e-5;
const HC_EPS: f32 = 1e-6;
const FIN_GRID_Y: u32 = 1 + (H as u32).div_ceil(256);
const ROWS: [usize; 5] = [1, 2, 4, 8, 16];
const SEEDS: [u64; 3] = [1, 2, 3];
const EDGE_ROWS: [usize; 2] = [1, 4];
const EDGE_KINDS: [&str; 7] = [
    "zero", "nan", "inf", "huge", "subnormal", "mixext", "mixnan",
];
const TIME_ROWS: [usize; 2] = [1, 4];
const SAMPLES: usize = 200;
const INNER: usize = 20;
const MAX_ROWS: usize = 16;

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
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bf16s(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect()
}
fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// Resolved kernels: production ones through the tree's resolver, new ones by name.
struct Ks {
    prod: Glm5NextMhcKernels,
    fin_warp: KernelHandle,
    mix_1pass: KernelHandle,
    fused: KernelHandle,
}

/// Device buffers sized for `MAX_ROWS` rows.
struct Bufs {
    hw: DevicePtr,
    bo: DevicePtr,
    post_in: DevicePtr,
    comb_in: DevicePtr,
    hw_out: DevicePtr,
    fn_w: DevicePtr,
    scale: DevicePtr,
    base: DevicePtr,
    mix: DevicePtr,
    y: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    counters: DevicePtr,
}

const OUT_BYTES: [usize; 4] = [MIX * 4, H * 2, HC * 4, HC * HC * 4]; // per row: mix, y, post, comb
const OUT_NAMES: [&str; 4] = ["mix", "y", "post", "comb"];

fn fill_outs(g: &dyn GpuBackend, b: &Bufs) -> Result<()> {
    for (p, n) in [
        (b.mix, MIX * 4),
        (b.y, H * 2),
        (b.post, HC * 4),
        (b.comb, HC * HC * 4),
    ] {
        g.memset_async(p, 0xEE, n * MAX_ROWS, 0)?;
    }
    Ok(())
}

fn read_outs(g: &dyn GpuBackend, b: &Bufs, t: usize) -> Result<[Vec<u8>; 4]> {
    Ok([
        down(g, b.mix, t * OUT_BYTES[0])?,
        down(g, b.y, t * OUT_BYTES[1])?,
        down(g, b.post, t * OUT_BYTES[2])?,
        down(g, b.comb, t * OUT_BYTES[3])?,
    ])
}

fn l_mix(g: &dyn GpuBackend, k: KernelHandle, b: &Bufs, t: usize, mix_out: DevicePtr) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([t as u32, MIX as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(b.hw)
        .arg_ptr(b.fn_w)
        .arg_ptr(mix_out)
        .arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_f32(NORM_EPS)
        .launch(0)
}

fn l_fin(g: &dyn GpuBackend, k: KernelHandle, b: &Bufs, t: usize, iters: u32) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([t as u32, FIN_GRID_Y, 1])
        .block([256, 1, 1])
        .arg_ptr(b.hw)
        .arg_ptr(b.mix)
        .arg_ptr(b.scale)
        .arg_ptr(b.base)
        .arg_ptr(b.y)
        .arg_ptr(b.post)
        .arg_ptr(b.comb)
        .arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_u32(iters)
        .arg_f32(HC_EPS)
        .launch(0)
}

fn l_fused(g: &dyn GpuBackend, k: &Ks, b: &Bufs, t: usize) -> Result<()> {
    KernelLaunch::new(g, k.fused)
        .grid([t as u32, MIX as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(b.hw)
        .arg_ptr(b.fn_w)
        .arg_ptr(b.mix)
        .arg_ptr(b.counters)
        .arg_ptr(b.scale)
        .arg_ptr(b.base)
        .arg_ptr(b.y)
        .arg_ptr(b.post)
        .arg_ptr(b.comb)
        .arg_u32(H as u32)
        .arg_u32(HC as u32)
        .arg_u32(SINKHORN)
        .arg_f32(NORM_EPS)
        .arg_f32(HC_EPS)
        .launch(0)
}

fn l_post(g: &dyn GpuBackend, k: &Ks, b: &Bufs, t: usize) -> Result<()> {
    glm_hc_post(
        g, k.prod.hc_post, b.bo, b.hw, b.post_in, b.comb_in, b.hw_out, t as u32, H as u32,
        HC as u32, 0,
    )
}

fn ndiff_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// One case's inputs.
struct Case {
    rows: usize,
    label: String,
    hw: Vec<f32>,
    mix_override: Option<Vec<f32>>,
    scale: [f32; 3],
}

fn edge_value(kind: &str, rng: &mut Lcg, n: usize) -> f32 {
    match kind {
        "zero" => 0.0,
        "nan" => {
            if n % 97 == 3 { f32::NAN } else { rng.sym(8.0) }
        }
        "inf" => match n % 211 {
            5 => f32::INFINITY,
            9 => f32::NEG_INFINITY,
            _ => rng.sym(8.0),
        },
        "huge" => rng.sym(1e19),
        "subnormal" => rng.sym(1.0) * 1e-39,
        _ => rng.sym(8.0),
    }
}

fn build_cases() -> Vec<Case> {
    let mut v = Vec::new();
    for &rows in &ROWS {
        for &seed in &SEEDS {
            let mut rng = Lcg(0x5EED0000 + seed * 7919 + rows as u64);
            let hw = (0..rows * HC * H).map(|_| rng.sym(8.0)).collect();
            v.push(Case {
                rows,
                label: seed.to_string(),
                hw,
                mix_override: None,
                scale: [0.8, 0.6, 1.2],
            });
        }
    }
    for &rows in &EDGE_ROWS {
        for kind in EDGE_KINDS {
            let mut rng = Lcg(0xED6E0000 + rows as u64 * 31 + kind.len() as u64);
            let hw: Vec<f32> = (0..rows * HC * H)
                .map(|n| edge_value(kind, &mut rng, n))
                .collect();
            let (mix_override, scale) = match kind {
                "mixext" | "mixnan" => {
                    let vals: [f32; 12] = [
                        0.0, 1e4, -1e4, 88.0, -104.0, 1e30, -0.0, 1e-45, 3.0, -3.0, 20.0, -20.0,
                    ];
                    let m = (0..rows * MIX)
                        .map(|n| {
                            if kind == "mixnan" && n % 11 == 4 {
                                f32::NAN
                            } else if kind == "mixnan" && n % 13 == 6 {
                                f32::INFINITY
                            } else if kind == "mixnan" && n % 17 == 8 {
                                f32::NEG_INFINITY
                            } else {
                                vals[(n * 5 + rows) % vals.len()]
                            }
                        })
                        .collect();
                    (Some(m), [-1e3f32, 1e3, 5.0])
                }
                _ => (None, [0.8f32, 0.6, 1.2]),
            };
            v.push(Case {
                rows,
                label: format!("edge_{kind}"),
                hw,
                mix_override,
                scale,
            });
        }
    }
    v
}

fn median_us(
    g: &dyn GpuBackend,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..20 {
        f()?;
    }
    g.synchronize(0)?;
    let mut v = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let (mut e0, mut e1, mut ms) = (0u64, 0u64, 0f32);
        unsafe {
            if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
                bail!("cuEventCreate returned an error");
            }
            cuEventRecord(e0, 0);
        }
        for _ in 0..INNER {
            f()?;
        }
        unsafe {
            cuEventRecord(e1, 0);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        v.push(ms as f64 * 1000.0 / INNER as f64);
    }
    v.sort_by(f64::total_cmp);
    Ok(v[v.len() / 2])
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        bail!("hc_decode_fuse_microtest: no glm-5.3-flash PTX set in this build");
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let prod = Glm5NextMhcKernels::resolve(g)?;
    if prod.hc_mix_bf16.0 == 0 {
        bail!("hc_decode_fuse_microtest: glm5next_hc_mix_bf16 not linked");
    }
    let ks = Ks {
        prod,
        fin_warp: g.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_finish_warp")?,
        mix_1pass: g.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_mix_bf16_1pass")?,
        fused: g.kernel(GLM5NEXT_MHC_MODULE, "glm5next_hc_mix_finish_1pass")?,
    };

    let mut rng = Lcg(0xFEED);
    let fnv: Vec<f32> = (0..MIX * HC * H).map(|_| rng.sym(0.05)).collect();
    let base: Vec<f32> = (0..MIX).map(|_| rng.sym(1.0)).collect();
    let b = Bufs {
        hw: g.alloc(MAX_ROWS * HC * H * 4)?,
        bo: g.alloc(MAX_ROWS * H * 2)?,
        post_in: g.alloc(MAX_ROWS * HC * 4)?,
        comb_in: g.alloc(MAX_ROWS * HC * HC * 4)?,
        hw_out: g.alloc(MAX_ROWS * HC * H * 4)?,
        fn_w: upload(g, &bf16s(&fnv))?,
        scale: g.alloc(16)?,
        base: upload(g, &f32s(&base))?,
        mix: g.alloc(MAX_ROWS * MIX * 4)?,
        y: g.alloc(MAX_ROWS * H * 2)?,
        post: g.alloc(MAX_ROWS * HC * 4)?,
        comb: g.alloc(MAX_ROWS * HC * HC * 4)?,
        counters: g.alloc(MAX_ROWS * 4)?,
    };
    g.memset_async(b.counters, 0, MAX_ROWS * 4, 0)?;
    let mut rng2 = Lcg(0xB0B0);
    g.copy_h2d(
        &bf16s(&(0..MAX_ROWS * H).map(|_| rng2.sym(3.0)).collect::<Vec<_>>()),
        b.bo,
    )?;
    g.copy_h2d(
        &f32s(&(0..MAX_ROWS * HC).map(|_| rng2.unit() * 2.0).collect::<Vec<_>>()),
        b.post_in,
    )?;
    g.copy_h2d(
        &f32s(&(0..MAX_ROWS * HC * HC).map(|_| rng2.unit()).collect::<Vec<_>>()),
        b.comb_in,
    )?;

    let cases = build_cases();
    let expected_cases = ROWS.len() * SEEDS.len() + EDGE_ROWS.len() * EDGE_KINDS.len();
    if cases.len() != expected_cases {
        bail!("hc_decode_fuse_microtest: built {} cases, expected {expected_cases}", cases.len());
    }
    let mut ran = 0usize;
    let mut bad = 0usize;
    let mut known_bad_seen = false;

    for c in &cases {
        let t = c.rows;
        g.copy_h2d(&f32s(&c.scale), b.scale)?;
        g.copy_h2d(&f32s(&c.hw), b.hw)?;
        // Reference chain: production mix, then production finish. Edge `mix` cases replace the
        // mix by the injected rows before the finish.
        fill_outs(g, &b)?;
        l_mix(g, ks.prod.hc_mix_bf16, &b, t, b.mix)?;
        let ref_mix = down(g, b.mix, t * MIX * 4)?;
        if let Some(m) = &c.mix_override {
            g.copy_h2d(&f32s(m), b.mix)?;
        }
        l_fin(g, ks.prod.hc_finish, &b, t, SINKHORN)?;
        let want = read_outs(g, &b, t)?;
        let fin_mix = down(g, b.mix, t * MIX * 4)?;
        if want[1].iter().all(|&x| x == 0xEE) || want[3].iter().all(|&x| x == 0xEE) {
            println!("BASE did not write y / comb rows={t} case={} (reference is stale)", c.label);
            bad += 1;
        }

        // E2: mix only.
        fill_outs(g, &b)?;
        l_mix(g, ks.mix_1pass, &b, t, b.mix)?;
        let e2 = down(g, b.mix, t * MIX * 4)?;
        let d2 = ndiff_bytes(&ref_mix, &e2);
        println!("HCDEC_CMP rows={t} seed={} kernel=E2_mix_1pass bytes_differ={d2}", c.label);
        ran += 1;
        if d2 != 0 {
            bad += 1;
        }

        // E1: finish from the same mix input as the reference finish.
        fill_outs(g, &b)?;
        g.copy_h2d(&fin_mix, b.mix)?;
        l_fin(g, ks.fin_warp, &b, t, SINKHORN)?;
        let got = read_outs(g, &b, t)?;
        let d1: Vec<usize> = (1..4).map(|i| ndiff_bytes(&want[i], &got[i])).collect();
        let d1s: usize = d1.iter().sum();
        println!(
            "HCDEC_CMP rows={t} seed={} kernel=E1_finish_warp bytes_differ={d1s}",
            c.label
        );
        if d1s != 0 {
            println!("  detail y={} post={} comb={}", d1[0], d1[1], d1[2]);
            bad += 1;
        }
        ran += 1;

        // Known-bad (once, on a normal case): 19 iterations must differ in comb.
        if !known_bad_seen && c.mix_override.is_none() && c.label == "1" {
            fill_outs(g, &b)?;
            g.copy_h2d(&fin_mix, b.mix)?;
            l_fin(g, ks.fin_warp, &b, t, SINKHORN - 1)?;
            let kb = read_outs(g, &b, t)?;
            let n = ndiff_bytes(&want[3], &kb[3]);
            println!("HCDEC_KNOWNBAD rows={t} E1 at 19 iters: comb bytes_differ={n} (must be > 0)");
            if n == 0 {
                bad += 1;
            }
            known_bad_seen = true;
        }

        // E3: the fused launch, compared with the reference chain on the HIGHWAY mix (not the
        // injected one), so the reference is re-run without the override.
        fill_outs(g, &b)?;
        g.memset_async(b.counters, 0, MAX_ROWS * 4, 0)?;
        l_mix(g, ks.prod.hc_mix_bf16, &b, t, b.mix)?;
        l_fin(g, ks.prod.hc_finish, &b, t, SINKHORN)?;
        let want3 = read_outs(g, &b, t)?;
        fill_outs(g, &b)?;
        l_fused(g, &ks, &b, t)?;
        let got3 = read_outs(g, &b, t)?;
        let d3: Vec<usize> = (0..4).map(|i| ndiff_bytes(&want3[i], &got3[i])).collect();
        let d3s: usize = d3.iter().sum();
        println!(
            "HCDEC_CMP rows={t} seed={} kernel=E3_fused bytes_differ={d3s}",
            c.label
        );
        if d3s != 0 {
            println!(
                "  detail {}",
                (0..4).map(|i| format!("{}={}", OUT_NAMES[i], d3[i])).collect::<Vec<_>>().join(" ")
            );
            bad += 1;
        }
        ran += 1;
        // The counters must be back at zero for the next launch.
        let cnt = down(g, b.counters, MAX_ROWS * 4)?;
        if cnt.iter().any(|&x| x != 0) {
            println!("E3 counters not reset rows={t} case={}", c.label);
            bad += 1;
        }
    }

    let expected_cmp = 3 * expected_cases;
    if ran != expected_cmp {
        println!("case count {ran} != expected {expected_cmp}");
        bad += 1;
    }
    if !known_bad_seen {
        println!("known-bad never ran");
        bad += 1;
    }

    // Timing: normal data, seed-1 style highway, rows 1 and 4.
    let mut rng3 = Lcg(0x71E);
    for &t in &TIME_ROWS {
        let hw: Vec<f32> = (0..t * HC * H).map(|_| rng3.sym(8.0)).collect();
        g.copy_h2d(&f32s(&hw), b.hw)?;
        g.copy_h2d(&f32s(&[0.8, 0.6, 1.2]), b.scale)?;
        g.memset_async(b.counters, 0, MAX_ROWS * 4, 0)?;
        l_mix(g, ks.prod.hc_mix_bf16, &b, t, b.mix)?;
        g.synchronize(0)?;
        let tp = median_us(g, || l_fin(g, ks.prod.hc_finish, &b, t, SINKHORN))?;
        let te1 = median_us(g, || l_fin(g, ks.fin_warp, &b, t, SINKHORN))?;
        let tm = median_us(g, || l_mix(g, ks.prod.hc_mix_bf16, &b, t, b.mix))?;
        let te2 = median_us(g, || l_mix(g, ks.mix_1pass, &b, t, b.mix))?;
        let te3 = median_us(g, || l_fused(g, &ks, &b, t))?;
        let ta1 = median_us(g, || l_fin(g, ks.prod.hc_finish, &b, t, 1))?;
        let ta20 = median_us(g, || l_fin(g, ks.prod.hc_finish, &b, t, 20))?;
        let te1_1 = median_us(g, || l_fin(g, ks.fin_warp, &b, t, 1))?;
        let tpost = median_us(g, || l_post(g, &ks, &b, t))?;
        for (name, us) in [
            ("prod_finish", tp),
            ("E1_finish_warp", te1),
            ("prod_mix", tm),
            ("E2_mix_1pass", te2),
            ("E3_fused", te3),
            ("ablate_finish_iters1", ta1),
            ("ablate_finish_iters20", ta20),
            ("E1_finish_warp_iters1", te1_1),
            ("post", tpost),
        ] {
            println!("HCDEC_T rows={t} kernel={name} us={us:.2}");
        }
        let c_prod = median_us(g, || {
            l_mix(g, ks.prod.hc_mix_bf16, &b, t, b.mix)?;
            l_fin(g, ks.prod.hc_finish, &b, t, SINKHORN)?;
            l_post(g, &ks, &b, t)
        })?;
        let c_new = median_us(g, || {
            l_mix(g, ks.mix_1pass, &b, t, b.mix)?;
            l_fin(g, ks.fin_warp, &b, t, SINKHORN)?;
            l_post(g, &ks, &b, t)
        })?;
        let c_new3 = median_us(g, || {
            l_fused(g, &ks, &b, t)?;
            l_post(g, &ks, &b, t)
        })?;
        println!(
            "HCDEC_CUT rows={t} mhc_prod_us={c_prod:.2} mhc_new_us={c_new:.2} cut={:.2}  (E2+E1+post chain)",
            c_prod - c_new
        );
        println!(
            "HCDEC_CUT3 rows={t} mhc_prod_us={c_prod:.2} mhc_new_us={c_new3:.2} cut={:.2}  (E3+post chain)",
            c_prod - c_new3
        );
        println!(
            "HCDEC_SUM rows={t} sum_of_parts_prod_us={:.2} (mix+finish+post timed alone)",
            tm + tp + tpost
        );
    }

    if bad != 0 {
        println!("FAIL: {bad} check(s) did not hold ({ran} of {expected_cmp} comparisons ran)");
        bail!("hc_decode_fuse_microtest: FAIL");
    }
    println!(
        "PASS: E1 glm5next_hc_finish_warp, E2 glm5next_hc_mix_bf16_1pass and E3 \
         glm5next_hc_mix_finish_1pass equal production bit for bit in {ran} comparisons \
         ({expected_cases} cases x 3 kernels); known-bad detected"
    );
    Ok(())
}
