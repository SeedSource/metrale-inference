// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Bitwise gate for `glm5next_hc_post_mix_bf16` (`METRALE_GLM_MHC_POST_MIX`) against
//! the pair it replaces: `glm5next_hc_post` (in place, grid `(T, 16)` as `glm_hc_post` launches
//! it) followed by the next site's `glm5next_hc_mix_bf16_tokmajor` (grid `(T, 1)`).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - Exits 0 only if, for every row count, the fused kernel's whole highway and whole mix output
//!   equal the pair's bit for bit, and both known-bad arms are detected: one `comb` element
//!   nudged by one ULP (highway must differ) and one `hc_fn` element of the next site nudged
//!   (mix must differ while the highway stays equal).
//!
//! Shapes: GLM-5.3 (hidden 4096, hc_mult 4, mix_hc 24) at 4096, 257 and 1 rows. Timing (4096
//! rows, cold: a 64 MiB memset between launches evicts L2): pair vs fused, median of 10.
//!
//! Run: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples --example
//! hc_post_mix_bitparity_microtest`

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{Glm5NextMhcKernels, glm_hc_post, glm_hc_post_mix};

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = 24;
const NORM_EPS: f32 = 1e-5;
const FLUSH_BYTES: usize = 64 << 20;

// 2026-10-06: CUDA driver event API for timing, declared as in `dense_gemm_microtest`.
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

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
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

fn ndiff(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| x != y)
        .count()
}

/// 2026-10-06: Device inputs of one row count.
struct Case {
    t: usize,
    block_out: DevicePtr,
    residual: Vec<u8>,
    post: DevicePtr,
    comb: DevicePtr,
    comb_bad: DevicePtr,
    hc_fn: DevicePtr,
    hc_fn_bad: DevicePtr,
    hw: DevicePtr,
    mix: DevicePtr,
}

/// 2026-10-06: The pair: `hc_post` in place on `hw` (reset to the residual), then the tokmajor
/// mix of the next site into `mix`.
fn run_pair(g: &dyn GpuBackend, k: &Glm5NextMhcKernels, c: &Case, reset: bool) -> Result<()> {
    if reset {
        g.copy_h2d(&c.residual, c.hw)?;
    }
    let (t, h, hc) = (c.t as u32, H as u32, HC as u32);
    glm_hc_post(
        g,
        k.hc_post,
        c.block_out,
        c.hw,
        c.post,
        c.comb,
        c.hw,
        t,
        h,
        hc,
        0,
    )?;
    KernelLaunch::new(g, k.hc_mix_bf16_tokmajor)
        .grid([t, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(c.hw)
        .arg_ptr(c.hc_fn)
        .arg_ptr(c.mix)
        .arg_u32(h)
        .arg_u32(hc)
        .arg_f32(NORM_EPS)
        .launch(0)
}

fn run_fused(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    c: &Case,
    comb: DevicePtr,
    hc_fn: DevicePtr,
    reset: bool,
) -> Result<()> {
    if reset {
        g.copy_h2d(&c.residual, c.hw)?;
    }
    let (t, h, hc) = (c.t as u32, H as u32, HC as u32);
    glm_hc_post_mix(
        g,
        k,
        c.block_out,
        c.hw,
        c.post,
        comb,
        c.hw,
        hc_fn,
        c.mix,
        t,
        h,
        hc,
        NORM_EPS,
        0,
    )
}

fn time_cold(
    g: &dyn GpuBackend,
    flush: DevicePtr,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f32> {
    let mut v = Vec::new();
    for _ in 0..10 {
        g.memset_async(flush, 0x3C, FLUSH_BYTES, 0)?;
        let (mut e0, mut e1, mut ms) = (0u64, 0u64, 0f32);
        unsafe {
            if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
                bail!("cuEventCreate returned an error");
            }
            cuEventRecord(e0, 0);
        }
        f()?;
        unsafe {
            cuEventRecord(e1, 0);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        v.push(ms);
    }
    v.sort_by(f32::total_cmp);
    Ok(v[v.len() / 2])
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        bail!("hc_post_mix_bitparity_microtest: no glm-5.3-flash PTX set in this build");
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextMhcKernels::resolve(g)?;
    if k.hc_post_mix_bf16.0 == 0 || k.hc_mix_bf16_tokmajor.0 == 0 {
        bail!("hc_post_mix_bitparity_microtest: fused or token-major kernel missing from the PTX");
    }
    let flush = g.alloc(FLUSH_BYTES)?;
    let mut bad = 0usize;
    let mut rng = Lcg(0x9C57);
    // 2026-10-06: The next site's hc_fn, shared by every case, and its one-element known-bad.
    let fnv: Vec<f32> = (0..MIX * HC * H).map(|_| rng.sym(0.05)).collect();
    let mut fn_bad = fnv.clone();
    fn_bad[7 * HC * H + 1234] += 0.25;
    let hc_fn = upload(g, &bf16s(&fnv))?;
    let hc_fn_bad = upload(g, &bf16s(&fn_bad))?;

    for t in [4096usize, 257, 1] {
        let bo: Vec<f32> = (0..t * H).map(|_| rng.sym(3.0)).collect();
        let res: Vec<f32> = (0..t * HC * H).map(|_| rng.sym(8.0)).collect();
        let post: Vec<f32> = (0..t * HC).map(|_| rng.unit() * 2.0).collect();
        let comb: Vec<f32> = (0..t * HC * HC).map(|_| rng.unit()).collect();
        let mut comb_bad = comb.clone();
        comb_bad[(t / 2) * HC * HC + 5] =
            f32::from_bits(comb_bad[(t / 2) * HC * HC + 5].to_bits() + 1);
        let c = Case {
            t,
            block_out: upload(g, &bf16s(&bo))?,
            residual: f32s(&res),
            post: upload(g, &f32s(&post))?,
            comb: upload(g, &f32s(&comb))?,
            comb_bad: upload(g, &f32s(&comb_bad))?,
            hc_fn,
            hc_fn_bad,
            hw: g.alloc(t * HC * H * 4)?,
            mix: g.alloc(t * MIX * 4)?,
        };
        let (hw_b, mix_b) = (t * HC * H * 4, t * MIX * 4);
        run_pair(g, &k, &c, true)?;
        let (hw_ref, mix_ref) = (down(g, c.hw, hw_b)?, down(g, c.mix, mix_b)?);
        g.memset_async(c.mix, 0xEE, mix_b, 0)?;
        run_fused(g, &k, &c, c.comb, c.hc_fn, true)?;
        let (hw_new, mix_new) = (down(g, c.hw, hw_b)?, down(g, c.mix, mix_b)?);
        let (dh, dm) = (ndiff(&hw_ref, &hw_new), ndiff(&mix_ref, &mix_new));
        println!(
            "rows={t}: highway {dh} differing floats of {}, mix {dm} differing floats of {}",
            t * HC * H,
            t * MIX
        );
        if dh != 0 || dm != 0 {
            println!("MISMATCH rows={t}");
            bad += 1;
        }
        // 2026-10-06: Known-bad 1: comb nudged by one ULP must change the highway.
        run_fused(g, &k, &c, c.comb_bad, c.hc_fn, true)?;
        let kb1 = ndiff(&hw_ref, &down(g, c.hw, hw_b)?);
        // 2026-10-06: Known-bad 2: the next site's hc_fn nudged must change the mix only.
        run_fused(g, &k, &c, c.comb, c.hc_fn_bad, true)?;
        let (kb2h, kb2m) = (
            ndiff(&hw_ref, &down(g, c.hw, hw_b)?),
            ndiff(&mix_ref, &down(g, c.mix, mix_b)?),
        );
        println!(
            "KNOWN_BAD rows={t}: comb+1ulp highway {kb1} differ; hc_fn nudge mix {kb2m} differ, highway {kb2h}"
        );
        if kb1 == 0 || kb2m == 0 || kb2h != 0 {
            println!("KNOWN_BAD not detected rows={t} (the comparison is blind)");
            bad += 1;
        }
        if t == 4096 {
            let tp = time_cold(g, flush, || run_pair(g, &k, &c, false))?;
            let tf = time_cold(g, flush, || run_fused(g, &k, &c, c.comb, c.hc_fn, false))?;
            println!(
                "TIMING rows={t} cold: post+mix {tp:.3} ms  fused {tf:.3} ms  speedup {:.2}x",
                tp / tf
            );
        }
        for p in [c.block_out, c.post, c.comb, c.comb_bad, c.hw, c.mix] {
            g.free(p)?;
        }
    }
    if bad != 0 {
        bail!("hc_post_mix_bitparity_microtest: {bad} check(s) did not hold");
    }
    println!(
        "PASS: glm5next_hc_post_mix_bf16 equals hc_post + hc_mix_bf16_tokmajor bit for bit \
         (whole highway and mix) at 4096, 257 and 1 rows; both known-bads detected"
    );
    Ok(())
}
