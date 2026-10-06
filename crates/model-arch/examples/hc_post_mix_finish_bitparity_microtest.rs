// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Bitwise gate for `glm5next_hc_post_mix_finish_bf16`
//! (`METRALE_GLM_MHC_POST_MIX_FINISH`) against the pair it replaces on the sequence-parallel
//! back: `glm5next_hc_post_mix_bf16` (`glm_hc_post_mix`, in place on the highway) followed by
//! the next site's `glm5next_hc_finish` (`glm_hc_pre_part_premixed`, 256-row slices, grid
//! `(k, 17)`).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - Both arms alias their buffers as `SpRows::back` / `SpRows::front` do: `y` is written over
//!   `block_out`, `post` / `comb` over the input `post` / `comb`, the highway in place. A third
//!   arm runs the fused kernel with separate output buffers.
//! - Exits 0 only if, at every row count, both fused arms equal the pair bit for bit on every
//!   output (highway, mix, y, post, comb), and both known-bads are detected: the input `comb`
//!   nudged by one ULP (highway and y must differ) and the next site's `hc_base` nudged (post,
//!   comb and y must differ while highway and mix stay equal).
//!
//! Shapes: GLM-5.3 (hidden 4096, hc_mult 4, mix_hc 24, BF16 `hc_fn`, 20 Sinkhorn iterations)
//! at 4096, 257 and 1 rows. Timing (4096 rows, cold: a 64 MiB memset between launches evicts
//! L2): pair vs fused, median of 10.
//!
//! Run: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples --example
//! hc_post_mix_finish_bitparity_microtest`

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_post_mix, glm_hc_post_mix_finish,
    glm_hc_pre_part_premixed,
};

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = 24;
const SINKHORN: u32 = 20;
const NORM_EPS: f32 = 1e-5;
const HC_EPS: f32 = 1e-6;
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

/// 2026-10-06: Elements of `w` bytes that differ.
fn ndiff(a: &[u8], b: &[u8], w: usize) -> usize {
    a.chunks_exact(w)
        .zip(b.chunks_exact(w))
        .filter(|(x, y)| x != y)
        .count()
}

/// 2026-10-06: Host copies of one row count's inputs.
struct Host {
    t: usize,
    bo: Vec<u8>,
    res: Vec<u8>,
    post: Vec<u8>,
    comb: Vec<u8>,
    comb_bad: Vec<u8>,
}

/// 2026-10-06: Device buffers of one row count: the aliased in/out set the back uses, and a
/// separate output set for the third arm.
struct Io {
    bo: DevicePtr,
    hw: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    mix: DevicePtr,
    y2: DevicePtr,
    post2: DevicePtr,
    comb2: DevicePtr,
}

/// 2026-10-06: Every output of one run: highway, mix, y, post, comb.
type Outs = [Vec<u8>; 5];

const NAMES: [&str; 5] = ["highway", "mix", "y", "post", "comb"];
const WIDTHS: [usize; 5] = [4, 4, 2, 4, 4];

fn reset(g: &dyn GpuBackend, io: &Io, h: &Host, comb: &[u8]) -> Result<()> {
    g.copy_h2d(&h.bo, io.bo)?;
    g.copy_h2d(&h.res, io.hw)?;
    g.copy_h2d(&h.post, io.post)?;
    g.copy_h2d(comb, io.comb)?;
    g.memset_async(io.mix, 0xEE, h.t * MIX * 4, 0)
}

fn read(
    g: &dyn GpuBackend,
    t: usize,
    hw: DevicePtr,
    m: DevicePtr,
    o: [DevicePtr; 3],
) -> Result<Outs> {
    Ok([
        down(g, hw, t * HC * H * 4)?,
        down(g, m, t * MIX * 4)?,
        down(g, o[0], t * H * 2)?,
        down(g, o[1], t * HC * 4)?,
        down(g, o[2], t * HC * HC * 4)?,
    ])
}

/// 2026-10-06: The pair as the back and the next front issue it (aliased).
fn run_pair(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    io: &Io,
    next: &Glm5NextMhcSiteWeights,
    t: usize,
) -> Result<()> {
    let (tt, h, hc) = (t as u32, H as u32, HC as u32);
    glm_hc_post_mix(
        g, k, io.bo, io.hw, io.post, io.comb, io.hw, next.hc_fn, io.mix, tt, h, hc, NORM_EPS, 0,
    )?;
    let w = Glm5NextMhcSiteWeights {
        mix: io.mix,
        ..*next
    };
    glm_hc_pre_part_premixed(
        g, k, io.hw, &w, io.bo, io.post, io.comb, tt, h, hc, SINKHORN, HC_EPS, 0,
    )
}

/// 2026-10-06: The fused kernel; `sep` writes y / post / comb to the separate buffers.
fn run_fused(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    io: &Io,
    next: &Glm5NextMhcSiteWeights,
    t: usize,
    sep: bool,
) -> Result<()> {
    let (y, p, c) = if sep {
        (io.y2, io.post2, io.comb2)
    } else {
        (io.bo, io.post, io.comb)
    };
    let (tt, h, hc) = (t as u32, H as u32, HC as u32);
    glm_hc_post_mix_finish(
        g, k, io.bo, io.hw, io.post, io.comb, io.hw, next, io.mix, y, p, c, tt, h, hc, SINKHORN,
        NORM_EPS, HC_EPS, 0,
    )
}

fn diffs(a: &Outs, b: &Outs) -> [usize; 5] {
    std::array::from_fn(|i| ndiff(&a[i], &b[i], WIDTHS[i]))
}

fn show(d: &[usize; 5]) -> String {
    let v: Vec<String> = (0..5).map(|i| format!("{} {}", NAMES[i], d[i])).collect();
    v.join(", ")
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
        bail!("hc_post_mix_finish_bitparity_microtest: no glm-5.3-flash PTX set in this build");
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextMhcKernels::resolve(g)?;
    if k.hc_post_mix_bf16.0 == 0 || k.hc_post_mix_finish_bf16.0 == 0 {
        bail!("hc_post_mix_finish_bitparity_microtest: post_mix or fused kernel missing");
    }
    let flush = g.alloc(FLUSH_BYTES)?;
    let mut bad = 0usize;
    let mut rng = Lcg(0x5C1F);
    // 2026-10-06: The next site's weights, shared by every case: BF16 hc_fn as the car loads
    // it, logit scales of the order of the checkpoint's, and its hc_base known-bad.
    let fnv: Vec<f32> = (0..MIX * HC * H).map(|_| rng.sym(0.05)).collect();
    let base: Vec<f32> = (0..MIX).map(|_| rng.sym(1.0)).collect();
    // 2026-10-06: A different offset per element, so the comb softmax (shift-invariant per
    // row) changes too.
    let base_bad: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, b)| b + 0.01 * (i as f32 + 1.0))
        .collect();
    let mut next = Glm5NextMhcSiteWeights {
        hc_fn: upload(g, &bf16s(&fnv))?,
        hc_fn_bf16: true,
        hc_scale: upload(g, &f32s(&[0.8, 0.6, 1.2]))?,
        hc_base: upload(g, &f32s(&base))?,
        mix: DevicePtr(0),
    };
    let good_base = next.hc_base;
    let bad_base = upload(g, &f32s(&base_bad))?;

    for t in [4096usize, 257, 1] {
        let bo: Vec<f32> = (0..t * H).map(|_| rng.sym(3.0)).collect();
        let res: Vec<f32> = (0..t * HC * H).map(|_| rng.sym(8.0)).collect();
        let post: Vec<f32> = (0..t * HC).map(|_| rng.unit() * 2.0).collect();
        let comb: Vec<f32> = (0..t * HC * HC).map(|_| rng.unit()).collect();
        let mut comb_bad = comb.clone();
        let e = (t / 2) * HC * HC + 5;
        comb_bad[e] = f32::from_bits(comb_bad[e].to_bits() + 1);
        let h = Host {
            t,
            bo: bf16s(&bo),
            res: f32s(&res),
            post: f32s(&post),
            comb: f32s(&comb),
            comb_bad: f32s(&comb_bad),
        };
        let io = Io {
            bo: g.alloc(t * H * 2)?,
            hw: g.alloc(t * HC * H * 4)?,
            post: g.alloc(t * HC * 4)?,
            comb: g.alloc(t * HC * HC * 4)?,
            mix: g.alloc(t * MIX * 4)?,
            y2: g.alloc(t * H * 2)?,
            post2: g.alloc(t * HC * 4)?,
            comb2: g.alloc(t * HC * HC * 4)?,
        };
        let aliased = [io.bo, io.post, io.comb];

        next.hc_base = good_base;
        reset(g, &io, &h, &h.comb)?;
        run_pair(g, &k, &io, &next, t)?;
        let want = read(g, t, io.hw, io.mix, aliased)?;
        if want[2] == h.bo || want[3] == h.post {
            println!("PAIR did not write y / post rows={t} (the reference is stale)");
            bad += 1;
        }

        reset(g, &io, &h, &h.comb)?;
        run_fused(g, &k, &io, &next, t, false)?;
        let d_al = diffs(&want, &read(g, t, io.hw, io.mix, aliased)?);
        reset(g, &io, &h, &h.comb)?;
        for (p, n) in [
            (io.y2, t * H * 2),
            (io.post2, t * HC * 4),
            (io.comb2, t * HC * HC * 4),
        ] {
            g.memset_async(p, 0xEE, n, 0)?;
        }
        run_fused(g, &k, &io, &next, t, true)?;
        let d_sep = diffs(
            &want,
            &read(g, t, io.hw, io.mix, [io.y2, io.post2, io.comb2])?,
        );
        println!("rows={t}: fused aliased differs: {}", show(&d_al));
        println!("rows={t}: fused separate differs: {}", show(&d_sep));
        if d_al.iter().chain(d_sep.iter()).any(|&n| n != 0) {
            println!("MISMATCH rows={t}");
            bad += 1;
        }

        // 2026-10-06: Known-bad 1: the input comb nudged by one ULP must change the highway
        // and, through it, y.
        reset(g, &io, &h, &h.comb_bad)?;
        run_fused(g, &k, &io, &next, t, false)?;
        let kb1 = diffs(&want, &read(g, t, io.hw, io.mix, aliased)?);
        // 2026-10-06: Known-bad 2: the next site's hc_base nudged must change post, comb and y
        // and leave the highway and the mix alone.
        next.hc_base = bad_base;
        reset(g, &io, &h, &h.comb)?;
        run_fused(g, &k, &io, &next, t, false)?;
        let kb2 = diffs(&want, &read(g, t, io.hw, io.mix, aliased)?);
        next.hc_base = good_base;
        println!("KNOWN_BAD rows={t}: comb+1ulp: {}", show(&kb1));
        println!("KNOWN_BAD rows={t}: hc_base nudge: {}", show(&kb2));
        let kb1_ok = kb1[0] != 0 && kb1[2] != 0;
        let kb2_ok = kb2[0] == 0 && kb2[1] == 0 && kb2[2] != 0 && kb2[3] != 0 && kb2[4] != 0;
        if !kb1_ok || !kb2_ok {
            println!("KNOWN_BAD not detected rows={t} (the comparison is blind)");
            bad += 1;
        }

        if t == 4096 {
            reset(g, &io, &h, &h.comb)?;
            let tp = time_cold(g, flush, || run_pair(g, &k, &io, &next, t))?;
            reset(g, &io, &h, &h.comb)?;
            let tf = time_cold(g, flush, || run_fused(g, &k, &io, &next, t, false))?;
            println!(
                "TIMING rows={t} cold: post_mix+finish {tp:.3} ms  fused {tf:.3} ms  \
                 speedup {:.2}x",
                tp / tf
            );
        }
        for p in [
            io.bo, io.hw, io.post, io.comb, io.mix, io.y2, io.post2, io.comb2,
        ] {
            g.free(p)?;
        }
    }
    if bad != 0 {
        bail!("hc_post_mix_finish_bitparity_microtest: {bad} check(s) did not hold");
    }
    println!(
        "PASS: glm5next_hc_post_mix_finish_bf16 equals hc_post_mix_bf16 + hc_finish bit for bit \
         (highway, mix, y, post, comb; aliased and separate outputs) at 4096, 257 and 1 rows; \
         both known-bads detected"
    );
    Ok(())
}
