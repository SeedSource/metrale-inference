// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Byte-identity and timing gate for the tiled CUTLASS weight-SFB swizzle
//! (`METRALE_CUTLASS_SFB_PACK_TILED`, `pack_weight_sfb_batched_tiled_k`) against the scalar
//! kernel (`pack_weight_sfb_batched_k`), through `cutlass::pack_weight_sfb_batched_mode`
//! (mode 1 scalar, mode 2 tiled).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - Exits 0 only if, for every case, the two arms' WHOLE output buffers (every slot, padding
//!   and untouched bytes included; both pre-filled with `SENTINEL`) are byte-equal, the shifted
//!   known-bad arm differs, and the tiled kernel refuses the shapes it does not cover.
//!
//! Cases: the GLM-5.3 W4A4 projections at 144 local experts (gate / up `n = 2048, k = 4096`,
//! down `n = 4096, k = 2048`), a short table with a null slot, and refusals (`n % 128 != 0`,
//! K-major source). Source bytes are uniform random (every E4M3 pattern, NaN included), since
//! both kernels convert each byte with the same expression.
//!
//! Timing: median of `SFB_MT_REPS` (default 20) launches per arm on the GLM shapes, as GB/s of
//! compulsory traffic `2 * count * n * k / 16` bytes (read + write). STREAM on GB10 is about
//! 240 GB/s.
//!
//! Run: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples --example
//! glm_cutlass_sfb_pack_microtest` (a build with CUTLASS_HOME).

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::cutlass::{pack_weight_sfb_batched_mode, sfb_bytes};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

const SENTINEL: u8 = 0xA5;

// 2026-10-05: CUDA driver event API for timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Lcg(u64);

impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

/// 2026-10-05: One case's device inputs: `count` expert scale buffers (`[n, k/16]`, or
/// `[k/16, n]` when not `n_major`), their pointer table (entry `null_slot` zeroed), and the
/// output stride.
struct Case {
    n: usize,
    k: usize,
    count: usize,
    n_major: bool,
    table: DevicePtr,
    stride: usize,
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn make_case(
    g: &dyn GpuBackend,
    n: usize,
    k: usize,
    count: usize,
    n_major: bool,
    null_slot: Option<usize>,
    seed: u64,
) -> Result<Case> {
    let mut rng = Lcg(seed);
    let per = n * k / 16;
    let mut ptrs = Vec::with_capacity(count + 1);
    for s in 0..count + 1 {
        let bytes: Vec<u8> = (0..per).map(|_| rng.byte()).collect();
        let p = upload(g, &bytes)?;
        ptrs.push(if Some(s) == null_slot { 0 } else { p.0 });
    }
    let table: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    Ok(Case {
        n,
        k,
        count,
        n_major,
        table: upload(g, &table)?,
        stride: sfb_bytes(n, k).next_multiple_of(16),
    })
}

/// 2026-10-05: Run `mode` on `c` (experts `first..first + count`) into a fresh
/// SENTINEL-filled buffer and return its bytes.
fn run(g: &dyn GpuBackend, c: &Case, first: usize, count: usize, mode: u32) -> Result<Vec<u8>> {
    let bytes = c.count * c.stride;
    let out = upload(g, &vec![SENTINEL; bytes])?;
    let (n, k) = (c.n as u32, c.k as u32);
    pack_weight_sfb_batched_mode(
        c.table.0, first as u32, count as u32, out.0, c.stride, n, k, c.n_major, mode, 0,
    )?;
    g.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    g.copy_d2h(out, &mut b)?;
    g.free(out)?;
    Ok(b)
}

fn time_ms(c: &Case, mode: u32, reps: usize) -> Result<f32> {
    let (n, k) = (c.n as u32, c.k as u32);
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (mut e0, mut e1) = (0u64, 0u64);
        let mut ms = 0f32;
        unsafe {
            if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
                bail!("cuEventCreate returned an error");
            }
            cuEventRecord(e0, 0);
        }
        pack_weight_sfb_batched_mode(
            c.table.0,
            0,
            c.count as u32,
            TIMING_OUT.with(|o| o.get()),
            c.stride,
            n,
            k,
            c.n_major,
            mode,
            0,
        )?;
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

thread_local! {
    // 2026-10-05: The timing arms' shared output buffer (sized for the largest case).
    static TIMING_OUT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn main() -> Result<()> {
    if !metrale_gpu_runtime::cutlass::available() {
        bail!("glm_cutlass_sfb_pack_microtest: no CUTLASS objects in this build (CUTLASS_HOME)");
    }
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let gb = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gb;
    let reps = std::env::var("SFB_MT_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let mut bad = 0usize;

    // 2026-10-05: (n, k, count, n_major, null slot) for the identity cases.
    let cases = [
        (2048usize, 4096usize, 144usize, true, None),
        (4096, 2048, 144, true, None),
        (2048, 4096, 6, true, Some(2usize)),
        (128, 64, 3, true, None),
    ];
    let mut glm = Vec::new();
    for (i, &(n, k, count, n_major, null_slot)) in cases.iter().enumerate() {
        let c = make_case(g, n, k, count, n_major, null_slot, 0x5FB0 + i as u64)?;
        let a = run(g, &c, 0, count, 1)?;
        let b = run(g, &c, 0, count, 2)?;
        let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        let written = a.iter().filter(|&&x| x != SENTINEL).count();
        println!(
            "case n={n} k={k} experts={count} null_slot={null_slot:?}: {diff} differing bytes of \
             {} ({written} written by the scalar arm)",
            a.len()
        );
        if diff != 0 {
            println!("MISMATCH: n={n} k={k}");
            bad += 1;
        }
        // 2026-10-05: Known-bad: the tiled arm over experts shifted by one must differ.
        let s = run(g, &c, 1, count, 2)?;
        let kd = a.iter().zip(&s).filter(|(x, y)| x != y).count();
        if kd == 0 {
            println!("KNOWN_BAD not detected: n={n} k={k} (the comparison is blind)");
            bad += 1;
        } else {
            println!("KNOWN_BAD detected: shifted experts differ in {kd} bytes");
        }
        if count == 144 {
            glm.push(c);
        }
    }
    // 2026-10-05: Refusals: shapes the tiled kernel does not cover must error, not write.
    for (n, k, n_major) in [(2000usize, 4096usize, true), (2048, 4096, false), (2048, 4048, true)] {
        let c = make_case(g, n, k, 2, n_major, None, 0xBAD0)?;
        match run(g, &c, 0, 2, 2) {
            Err(e) => println!("refusal n={n} k={k} n_major={n_major}: refused ({e:#})"),
            Ok(_) => {
                println!("REFUSAL MISSING: tiled kernel ran on n={n} k={k} n_major={n_major}");
                bad += 1;
            }
        }
        // The scalar arm still covers it.
        run(g, &c, 0, 2, 1)?;
    }
    let big = glm.iter().map(|c| c.count * c.stride).max().unwrap_or(1);
    let out = g.alloc(big)?;
    TIMING_OUT.with(|o| o.set(out.0));
    for c in &glm {
        let traffic = 2.0 * (c.count * c.n * c.k / 16) as f64;
        let (t1, t2) = (time_ms(c, 1, reps)?, time_ms(c, 2, reps)?);
        println!(
            "TIMING n={} k={} experts={}: scalar {:.3} ms ({:.1} GB/s)  tiled {:.3} ms ({:.1} \
             GB/s)  speedup {:.2}x",
            c.n,
            c.k,
            c.count,
            t1,
            traffic / (t1 as f64 * 1e6),
            t2,
            traffic / (t2 as f64 * 1e6),
            t1 / t2
        );
    }
    if bad != 0 {
        bail!("glm_cutlass_sfb_pack_microtest: {bad} check(s) did not hold");
    }
    println!(
        "PASS: tiled SFB pack byte-identical to the scalar kernel on every case (whole buffers), \
         known-bad detected, unsupported shapes refused"
    );
    Ok(())
}
