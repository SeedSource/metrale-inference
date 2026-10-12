// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Ceiling for grouping dense NVFP4 GEMV launches: is one launch over weights
//! concatenated along N cheaper than separate launches of the same rows?
//!
//! Owner: model-arch examples.
//! Launch is the production one (`glm5next_layer/dense_fp8.rs` nv4 GEMV): module `w4a16_gemv`,
//! `w4a16_gemv_batch3_staged` at M = 3 and `w4a16_gemv` at M = 1, grid [ceil(N/4),1,1], block
//! [256,1,1], args (a, weight, weight_scale, f32 weight_scale_2, c, n, k). Per group: (S) the
//! member launches back to back on one stream; (G) one launch with N = sum of the members'
//! Ns over a buffer of the members' packed weights and scales appended along N (one
//! weight_scale_2 for all). L2 states: `cold` (72 MiB read-only flush before every rep) and
//! `chain` (no flush; one unrelated 9 MiB GEMV right before). Median of 60 reps after 5
//! warm-ups, S and G interleaved. Weights are random packed bytes, not quantised data.
//! Correctness: G's output rows must be bitwise equal to the members' outputs (rows are
//! independent); output is [M, N] row-major, so G row m = concat of S_i row m.
//!
//! Final line `PASS group_ceiling ...` (exit 0), `FAIL ...` on a bitwise mismatch (exit 1) or
//! a missing kernel (exit 2); timing is information only.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_nv4_group_ceiling_microtest

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

mod data;
mod timing;
use data::{Lcg, dn_u16, gen_act, gen_packed, up_bf16, up_bytes};
use timing::{Events, REPS, time_arms};

/// (name, member Ns, K, uses per decode round).
const GROUPS: &[(&str, &[usize], usize, usize)] = &[
    ("kda_qkv", &[4096, 4096, 4096], 4096, 34),
    ("shared_gateup", &[1024, 1024], 4096, 42),
    ("dsa_qa_kva", &[1536, 512], 4096, 11),
];
const TOKENS_PER_ROUND: f64 = 1.89;
const SCALE2: f32 = 0.01;
/// Flush weight 4096 x 32768 = 64 MiB packed + 8 MiB scales; chain weight 4096 x 4096 = 9 MiB.
const FLUSH: (usize, usize) = (4096, 32768);
const CHAIN: (usize, usize) = (4096, 4096);

/// A packed NVFP4 weight on device.
struct W {
    w: DevicePtr,
    s: DevicePtr,
    n: usize,
    k: usize,
}

fn upload(g: &dyn GpuBackend, w: &[u8], s: &[u8], n: usize, k: usize) -> Result<W> {
    Ok(W {
        w: up_bytes(g, w)?,
        s: up_bytes(g, s)?,
        n,
        k,
    })
}

fn free_w(g: &dyn GpuBackend, w: &W) -> Result<()> {
    g.free(w.w)?;
    g.free(w.s)
}

fn launch(
    g: &dyn GpuBackend,
    h: KernelHandle,
    a: DevicePtr,
    w: &W,
    c: DevicePtr,
    st: u64,
) -> Result<()> {
    KernelLaunch::new(g, h)
        .grid([div_ceil(w.n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.w)
        .arg_ptr(w.s)
        .arg_f32(SCALE2)
        .arg_ptr(c)
        .arg_u32(w.n as u32)
        .arg_u32(w.k as u32)
        .launch(st)
}

/// Zero-weight stand-in (flush / chain): the read is all that matters.
fn zero_w(g: &dyn GpuBackend, (n, k): (usize, usize)) -> Result<W> {
    let w = W {
        w: g.alloc(n * k / 2)?,
        s: g.alloc(n * k / 16)?,
        n,
        k,
    };
    g.memset(w.w, 0, n * k / 2)?;
    g.memset(w.s, 0, n * k / 16)?;
    Ok(w)
}

/// Untimed preludes: the 72 MiB flush weight, the 9 MiB chain weight and their scratch.
struct Pre {
    flush: W,
    chain: W,
    a: DevicePtr,
    c: DevicePtr,
}

/// One group at `m` rows: ([cold, chain] x (S us, G us)), and the differing u16 count.
fn run_group(
    g: &dyn GpuBackend,
    ev: &Events,
    h: KernelHandle,
    pre: &Pre,
    rng: &mut Lcg,
    ns: &[usize],
    k: usize,
    m: usize,
) -> Result<([(f64, f64); 2], usize)> {
    let st = g.default_stream();
    let total: usize = ns.iter().sum();
    let (mut cw, mut cs) = (Vec::new(), Vec::new());
    let mut members = Vec::new();
    for &n in ns {
        let (w, s) = gen_packed(rng, n, k);
        members.push(upload(g, &w, &s, n, k)?);
        cw.extend(w);
        cs.extend(s);
    }
    let cat = upload(g, &cw, &cs, total, k)?;
    let a = up_bf16(g, &gen_act(rng, m, k))?;
    let outs: Vec<DevicePtr> = ns
        .iter()
        .map(|&n| g.alloc(m * n * 2))
        .collect::<Result<_>>()?;
    let cg = g.alloc(m * total * 2)?;

    let arm_s = |s: u64| -> Result<()> {
        for (w, &c) in members.iter().zip(&outs) {
            launch(g, h, a, w, c, s)?;
        }
        Ok(())
    };
    let arm_g = |s: u64| launch(g, h, a, &cat, cg, s);

    // Correctness: poisoned outputs, one run of each arm, compare row by row.
    let poison = vec![0xA5u8; m * total * 2];
    g.copy_h2d(&poison, cg)?;
    for (&c, &n) in outs.iter().zip(ns) {
        g.copy_h2d(&poison[..m * n * 2], c)?;
    }
    arm_s(st)?;
    arm_g(st)?;
    g.synchronize(st)?;
    let yg = dn_u16(g, cg, m * total)?;
    let (mut diff, mut off) = (0usize, 0usize);
    for (&c, &n) in outs.iter().zip(ns) {
        let ys = dn_u16(g, c, m * n)?;
        for r in 0..m {
            for j in 0..n {
                diff += usize::from(ys[r * n + j] != yg[r * total + off + j]);
            }
        }
        off += n;
    }

    // Timing. The prelude kernel is always the M = 1 `w4a16_gemv` over a zero weight.
    let gemv1 = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let cold = time_arms(
        ev,
        st,
        &|s| launch(g, gemv1, pre.a, &pre.flush, pre.c, s),
        &[&arm_s, &arm_g],
    )?;
    let chain = time_arms(
        ev,
        st,
        &|s| launch(g, gemv1, pre.a, &pre.chain, pre.c, s),
        &[&arm_s, &arm_g],
    )?;
    g.synchronize(st)?;
    for w in &members {
        free_w(g, w)?;
    }
    free_w(g, &cat)?;
    for c in outs {
        g.free(c)?;
    }
    g.free(cg)?;
    g.free(a)?;
    Ok(([(cold[0], cold[1]), (chain[0], chain[1])], diff))
}

fn run() -> Result<i32> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (Ok(k1), Ok(k3)) = (
        g.kernel("w4a16_gemv", "w4a16_gemv"),
        g.kernel("w4a16_gemv", "w4a16_gemv_batch3_staged"),
    ) else {
        println!("FAIL missing kernel w4a16_gemv / w4a16_gemv_batch3_staged in this image");
        return Ok(2);
    };
    println!(
        "SETUP reps {REPS} (median), cold = 72 MiB read-only flush, chain = one 9 MiB GEMV before"
    );
    let ev = Events::new()?;
    let pre = Pre {
        flush: zero_w(g, FLUSH)?,
        chain: zero_w(g, CHAIN)?,
        a: g.alloc(FLUSH.1 * 2)?,
        c: g.alloc(FLUSH.0 * 2)?,
    };
    g.memset(pre.a, 0, FLUSH.1 * 2)?;
    g.synchronize(g.default_stream())?;
    let mut rng = Lcg(0x6772_6f75_7063);
    let mut bad = 0usize;
    let mut sums = [0.0f64; 2];
    for (m, h) in [(3usize, k3), (1usize, k1)] {
        for &(name, ns, k, count) in GROUPS {
            let (t, diff) = run_group(g, &ev, h, &pre, &mut rng, ns, k, m)?;
            if diff > 0 {
                bad += 1;
                println!("MISMATCH {name} M={m}: {diff} differing output elements");
            } else {
                println!("BITWISE {name} M={m}: 0 differing output elements");
            }
            for (i, l2) in ["cold", "chain"].into_iter().enumerate() {
                let (s, gg) = t[i];
                let saved = s - gg;
                let per_round_ms = saved * count as f64 / 1000.0;
                if m == 3 {
                    sums[i] += per_round_ms;
                }
                println!(
                    "GROUP {name} M={m} L2={l2} S {s:.2} G {gg:.2} saved {saved:.2} per_round_ms {per_round_ms:.4}"
                );
            }
        }
    }
    for (i, l2) in ["cold", "chain"].into_iter().enumerate() {
        println!(
            "SUM M=3 L2={l2} saved_ms_per_round {:.4} ms_per_token {:.4}",
            sums[i],
            sums[i] / TOKENS_PER_ROUND
        );
    }
    g.free(pre.flush.w)?;
    g.free(pre.flush.s)?;
    g.free(pre.chain.w)?;
    g.free(pre.chain.s)?;
    g.free(pre.a)?;
    g.free(pre.c)?;
    if bad > 0 {
        println!("FAIL group_ceiling: {bad} group(s) with output rows not bitwise equal");
        return Ok(1);
    }
    println!("PASS group_ceiling (bitwise; timing is information)");
    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            println!("ERROR {e:#}");
            std::process::exit(3);
        }
    }
}
