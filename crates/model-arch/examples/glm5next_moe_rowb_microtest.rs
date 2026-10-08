// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Gate for `METRALE_GLM_MOE_ROUTER_BATCHM` and `METRALE_GLM_MOE_UNION_SCAN`.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Checks:
//! - (1) Router: `dense_gemv_bf16_fp32out_batchm` (one launch of M rows) against M single-row
//!   `dense_gemv_bf16_fp32out` launches, N in {288, 256}, K 4096, M = 2..=16, random BF16
//!   (weights N(0, 0.02), activations N(0, 1)); the FP32 outputs are compared bitwise.
//! - (2) Union: `glm5next_moe_row_union_scan` against `glm5next_moe_row_union`, top_k 8,
//!   rows 2..=8, 288 experts: random routings, heavy overlap (pool of 12 experts), identical
//!   rows, and -1 entries. Both write poisoned buffers; all 64 u_eid + 512 u_slot ints are
//!   compared exactly.
//! - Timing: router at M = 8, N = 288 (8 single-row launches vs one batchm launch, median of
//!   >= 200 reps); union at rows = 8 (median of >= 500 reps); CUDA events, warm caches.
//! - `GATE exact` and `GATE speed` lines; exit 0 on PASS, 1 on FAIL, 2 if a kernel is missing.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_moe_rowb_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

// CUDA driver event API for kernel-only timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const HIDDEN: usize = 4096;
const TOP_K: usize = 8;
const NUM_EXPERTS: usize = 288;
const LAYERS: f64 = 43.0;
const U_EID: usize = 64;
const U_SLOT: usize = 64 * 8;
const POISON: i32 = 0x7f7f_7f7f;
const ROUTER_REPS: usize = 200;
const UNION_REPS: usize = 500;
const GO_MS: f64 = 2.0;
const KILL_MS: f64 = 1.0;

struct Rng(u64);
#[rustfmt::skip]
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn u(&mut self) -> f32 { (self.next() & 0xFF_FFFF) as f32 / 16_777_216.0 }
    /// Box-Muller standard normal.
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.u().max(1e-7), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn le<T: Copy, const B: usize>(v: &[T], f: impl Fn(T) -> [u8; B]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn bf16_normal(rng: &mut Rng, n: usize, sigma: f32) -> Vec<u8> {
    let v: Vec<bf16> = (0..n)
        .map(|_| bf16::from_f32(rng.normal() * sigma))
        .collect();
    le(&v, |x| x.to_bits().to_le_bytes())
}
fn bytes_i32(v: &[u8]) -> Vec<i32> {
    v.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Time `f` (enqueued on stream `s`) between two CUDA events; returns microseconds.
fn time_us(g: &dyn GpuBackend, s: u64, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    let (mut e0, mut e1) = (0u64, 0u64);
    // SAFETY: plain CUDA driver event calls on handles created here and destroyed below.
    unsafe {
        if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
            bail!("cuEventCreate failed");
        }
        cuEventRecord(e0, s);
    }
    f()?;
    let mut ms = 0f32;
    // SAFETY: as above.
    let rc = unsafe {
        cuEventRecord(e1, s);
        let rc = cuEventSynchronize(e1);
        let rc2 = cuEventElapsedTime(&mut ms, e0, e1);
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
        rc | rc2
    };
    g.synchronize(s)?;
    if rc != 0 {
        bail!("CUDA event timing failed: status {rc}");
    }
    Ok(ms as f64 * 1e3)
}

struct Kernels {
    single: KernelHandle,
    batchm: KernelHandle,
    union_old: KernelHandle,
    union_new: KernelHandle,
}

/// One single-row launch of row `r`.
#[allow(clippy::too_many_arguments)]
fn single_row(
    g: &dyn GpuBackend,
    k: &Kernels,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    r: usize,
    n: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k.single)
        .grid([n.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(DevicePtr(a.0 + (r * HIDDEN * 2) as u64))
        .arg_ptr(b)
        .arg_ptr(DevicePtr(c.0 + (r * n * 4) as u64))
        .arg_u32(n as u32)
        .arg_u32(HIDDEN as u32)
        .launch(s)
}

#[allow(clippy::too_many_arguments)]
fn batch_launch(
    g: &dyn GpuBackend,
    k: &Kernels,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k.batchm)
        .grid([n.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(HIDDEN as u32)
        .arg_u32(n as u32)
        .launch(s)
}

/// (1) exactness; returns (compares, fails).
fn router_exact(g: &dyn GpuBackend, k: &Kernels, rng: &mut Rng) -> Result<(usize, usize)> {
    let (mut total, mut fails) = (0, 0);
    for n in [288usize, 256] {
        let d_b = up(g, &bf16_normal(rng, n * HIDDEN, 0.02))?;
        for m in 2..=16usize {
            let d_a = up(g, &bf16_normal(rng, m * HIDDEN, 1.0))?;
            let bytes = m * n * 4;
            let (d_ref, d_new) = (g.alloc(bytes)?, g.alloc(bytes)?);
            g.memset_async(d_ref, 0xA5, bytes, 0)?;
            g.memset_async(d_new, 0xA5, bytes, 0)?;
            for r in 0..m {
                single_row(g, k, d_a, d_b, d_ref, r, n, 0)?;
            }
            batch_launch(g, k, d_a, d_b, d_new, m, n, 0)?;
            g.synchronize(0)?;
            let (x, y) = (dn(g, d_ref, bytes)?, dn(g, d_new, bytes)?);
            let (xi, yi) = (bytes_i32(&x), bytes_i32(&y));
            total += 1;
            let diff = xi.iter().zip(&yi).filter(|(p, q)| p != q).count();
            if diff != 0 {
                fails += 1;
                let first = xi.iter().zip(&yi).position(|(p, q)| p != q);
                println!(
                    "FAIL exact router N={n} M={m}: {diff}/{} u32 differ, first {first:?}",
                    xi.len()
                );
            }
            for p in [d_a, d_ref, d_new] {
                g.free(p)?;
            }
        }
        println!("exact router N={n}: M=2..=16 compared (running fails {fails})");
        g.free(d_b)?;
    }
    Ok((total, fails))
}

/// Distinct random experts, drawn from `pool` (all experts when `None`).
fn pick8(rng: &mut Rng, pool: Option<&[i32]>) -> Vec<i32> {
    let mut p: Vec<i32> = pool.map_or_else(|| (0..NUM_EXPERTS as i32).collect(), <[i32]>::to_vec);
    for i in 0..TOP_K {
        let j = i + rng.next() as usize % (p.len() - i);
        p.swap(i, j);
    }
    p[..TOP_K].to_vec()
}

fn union_cases(rng: &mut Rng, rows: usize) -> Vec<(String, Vec<i32>)> {
    let flat = |v: Vec<Vec<i32>>| v.into_iter().flatten().collect::<Vec<i32>>();
    let mut cases = Vec::new();
    for s in 0..6 {
        cases.push((
            format!("random{s}"),
            flat((0..rows).map(|_| pick8(rng, None)).collect()),
        ));
    }
    for s in 0..6 {
        let pool = pick8(rng, None)
            .into_iter()
            .chain(pick8(rng, None).into_iter().take(4))
            .collect::<Vec<_>>();
        let mut pool = pool;
        pool.sort_unstable();
        pool.dedup();
        cases.push((
            format!("overlap12_{s}"),
            flat((0..rows).map(|_| pick8(rng, Some(&pool))).collect()),
        ));
    }
    let same = pick8(rng, None);
    cases.push((
        "identical".into(),
        flat((0..rows).map(|_| same.clone()).collect()),
    ));
    for s in 0..4 {
        let mut ids = flat((0..rows).map(|_| pick8(rng, None)).collect());
        for v in &mut ids {
            if rng.next().is_multiple_of(4) {
                *v = -1;
            }
        }
        cases.push((format!("neg{s}"), ids));
    }
    let mut ids = flat((0..rows).map(|_| pick8(rng, None)).collect());
    ids[..TOP_K].fill(-1);
    cases.push(("row0_all_neg".into(), ids));
    cases.push(("all_neg".into(), vec![-1; rows * TOP_K]));
    cases
}

#[allow(clippy::too_many_arguments)]
fn union_launch(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    ids: DevicePtr,
    ue: DevicePtr,
    us: DevicePtr,
    rows: usize,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, kern)
        .grid([1, 1, 1])
        .block([(rows * TOP_K) as u32, 1, 1])
        .arg_ptr(ids)
        .arg_ptr(ue)
        .arg_ptr(us)
        .arg_u32(rows as u32)
        .arg_u32(TOP_K as u32)
        .launch(s)
}

fn poison(g: &dyn GpuBackend, ue: DevicePtr, us: DevicePtr) -> Result<()> {
    g.copy_h2d(&le(&[POISON; U_EID], |x: i32| x.to_le_bytes()), ue)?;
    g.copy_h2d(&le(&[POISON; U_SLOT], |x: i32| x.to_le_bytes()), us)?;
    Ok(())
}

/// (2) exactness; returns (compares, fails).
fn union_exact(g: &dyn GpuBackend, k: &Kernels, rng: &mut Rng) -> Result<(usize, usize)> {
    let (mut total, mut fails) = (0, 0);
    let bufs: Vec<DevicePtr> = (0..4)
        .map(|i| g.alloc(if i % 2 == 0 { U_EID * 4 } else { U_SLOT * 4 }))
        .collect::<Result<_>>()?;
    for rows in 2..=8usize {
        for (tag, ids) in union_cases(rng, rows) {
            let d_ids = up(g, &le(&ids, |x: i32| x.to_le_bytes()))?;
            poison(g, bufs[0], bufs[1])?;
            poison(g, bufs[2], bufs[3])?;
            union_launch(g, k.union_old, d_ids, bufs[0], bufs[1], rows, 0)?;
            union_launch(g, k.union_new, d_ids, bufs[2], bufs[3], rows, 0)?;
            g.synchronize(0)?;
            let old = [dn(g, bufs[0], U_EID * 4)?, dn(g, bufs[1], U_SLOT * 4)?].concat();
            let new = [dn(g, bufs[2], U_EID * 4)?, dn(g, bufs[3], U_SLOT * 4)?].concat();
            total += U_EID + U_SLOT;
            let (oi, ni) = (bytes_i32(&old), bytes_i32(&new));
            let diff = oi.iter().zip(&ni).filter(|(a, b)| a != b).count();
            if diff != 0 {
                fails += 1;
                let first = oi.iter().zip(&ni).position(|(a, b)| a != b);
                println!(
                    "FAIL exact union rows={rows} {tag}: {diff}/{} ints differ, first {first:?}",
                    oi.len()
                );
            }
            g.free(d_ids)?;
        }
        println!("exact union rows={rows}: all cases compared (running fails {fails})");
    }
    for p in bufs {
        g.free(p)?;
    }
    Ok((total, fails))
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let names = [
        ("gemv", "dense_gemv_bf16_fp32out"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm"),
        ("w4a16_gemv", "glm5next_moe_row_union"),
        ("w4a16_gemv", "glm5next_moe_row_union_scan"),
    ];
    let mut hs = Vec::new();
    for (module, name) in names {
        match g.kernel(module, name) {
            Ok(h) => hs.push(h),
            Err(_) => println!("MISSING kernel entry {module}::{name}"),
        }
    }
    if hs.len() != names.len() {
        println!("FAIL kernel entries missing ({})", names.len() - hs.len());
        return Ok(2);
    }
    let k = Kernels {
        single: hs[0],
        batchm: hs[1],
        union_old: hs[2],
        union_new: hs[3],
    };
    let mut rng = Rng(0x0072_6f77_6231_0001);

    let (t1, f1) = router_exact(g, &k, &mut rng)?;
    let (t2, f2) = union_exact(g, &k, &mut rng)?;
    let (total, fails) = (t1 + t2, f1 + f2);

    // Router timing at M = 8, N = 288.
    let s = g.create_stream()?;
    let (m, n) = (8usize, 288usize);
    let d_b = up(g, &bf16_normal(&mut rng, n * HIDDEN, 0.02))?;
    let d_a = up(g, &bf16_normal(&mut rng, m * HIDDEN, 1.0))?;
    let d_c = g.alloc(m * n * 4)?;
    let per_row = || -> Result<()> {
        for r in 0..m {
            single_row(g, &k, d_a, d_b, d_c, r, n, s)?;
        }
        Ok(())
    };
    let batched = || batch_launch(g, &k, d_a, d_b, d_c, m, n, s);
    for _ in 0..20 {
        per_row()?;
        batched()?;
    }
    g.synchronize(s)?;
    let (mut ro, mut rn) = (Vec::new(), Vec::new());
    for _ in 0..ROUTER_REPS {
        ro.push(time_us(g, s, &per_row)?);
        rn.push(time_us(g, s, &batched)?);
    }
    let (router_old, router_new) = (median(&mut ro), median(&mut rn));
    println!(
        "TIMING router M={m} N={n}: per-row x{m} {router_old:.2} us, batchm {router_new:.2} us \
         (x{LAYERS} = {:.3} -> {:.3} ms/step), {ROUTER_REPS} reps",
        LAYERS * router_old / 1e3,
        LAYERS * router_new / 1e3
    );

    // Union timing at rows = 8 (random routings, rotated per rep).
    let rows = 8usize;
    let ids: Vec<DevicePtr> = (0..8)
        .map(|_| {
            let v: Vec<i32> = (0..rows).flat_map(|_| pick8(&mut rng, None)).collect();
            up(g, &le(&v, |x: i32| x.to_le_bytes()))
        })
        .collect::<Result<_>>()?;
    let (ue, us) = (g.alloc(U_EID * 4)?, g.alloc(U_SLOT * 4)?);
    let (mut uo, mut un) = (Vec::new(), Vec::new());
    for rep in 0..UNION_REPS + 20 {
        let d_ids = ids[rep % ids.len()];
        let fo = || union_launch(g, k.union_old, d_ids, ue, us, rows, s);
        let fnw = || union_launch(g, k.union_new, d_ids, ue, us, rows, s);
        let (a, b) = (time_us(g, s, &fo)?, time_us(g, s, &fnw)?);
        if rep >= 20 {
            uo.push(a);
            un.push(b);
        }
    }
    let (union_old, union_new) = (median(&mut uo), median(&mut un));
    println!(
        "TIMING union rows={rows}: old {union_old:.2} us, scan {union_new:.2} us \
         (x{LAYERS} = {:.3} -> {:.3} ms/step), {UNION_REPS} reps",
        LAYERS * union_old / 1e3,
        LAYERS * union_new / 1e3
    );

    let exact_ok = fails == 0;
    println!(
        "GATE exact {} ({total} compares)",
        if exact_ok {
            "PASS".to_string()
        } else {
            format!("FAIL {fails} case(s) differ: router {f1}, union {f2}")
        }
    );
    // 2026-10-08: build-mt's verdict line (a line starting "PASS"); the GATE lines carry the detail.
    if exact_ok {
        println!("PASS: glm5next_moe_rowb_microtest {total} compares bit-identical");
    }
    let saves = LAYERS * ((router_old - router_new) + (union_old - union_new)) / 1e3;
    let verdict = if saves >= GO_MS {
        "GO"
    } else if saves < KILL_MS {
        "KILL"
    } else {
        "GREY"
    };
    println!(
        "GATE speed router {router_old:.2} -> {router_new:.2} us, union {union_old:.2} -> \
         {union_new:.2} us, saves {saves:.2} ms/step: {verdict}"
    );
    Ok(if exact_ok { 0 } else { 1 })
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(c) => std::process::exit(c),
        Err(e) => {
            println!("FAIL error: {e:#}");
            std::process::exit(1);
        }
    }
}
