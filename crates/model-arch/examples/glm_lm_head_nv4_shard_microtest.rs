// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate for `METRALE_GLM_LM_HEAD_NV4_SHARD` (`glm5next_lm_head_nv4.rs`): the NVFP4
//! vocab shard of the GLM-5.3 target head against the FP8 shard it replaces on the verify and
//! decode heads (`lm_head_fp8_tc`).
//!
//! Owner: model-arch examples.
//! Shape: this rank's shard at TP2, N = 77440 (vocab 154880 / 2), K = 4096 (GLM-5.3-Flash
//! hidden size; DRAFTER-SURVEY-20261003.md, not the 6144 of the ticket text).
//! Checks:
//! - (a) Exactness of the shard quantizer: `quantize_shard` of rows [S, S+n) of a 2n-row head
//!   is bitwise the same packed bytes, scale bytes and f32 tensor scale as rows [S, S+n) of
//!   `quantize_nvfp4_copy` of the whole head (the fakeq emulation's quantization).
//! - (b) Correctness at M = 1, 3, 8, 9, 16 on BOTH tiers (CUDA-core, tensor-core) and on the
//!   routed `sweep`: logits against an FP64 host reference over the NVFP4 copy's own
//!   (dequantized) weights, per row cosine >= 0.99999, nonfinite 0, argmax identical to the
//!   reference's (a differing argmax passes only when the reference's two values are within
//!   1 % of the row max, a near-tie the BF16 output rounding decides).
//! - (c) Row bits: CUDA-core rows at M = 2, 3 equal the M = 1 launch per row; tensor-core rows at
//!   M = 4..=16 equal the M = 16 launch per row (both documented invariances).
//! - (d) Layout: the 2-row-plus path (sweep into scratch, pitched copy into zeroed `[M, 2n]`
//!   logits at column S) puts every value at its column and leaves the other columns 0.
//! - (e) Speed, M = 1, 2, 3, 4, 5, 7, 8, 9, 16, cold pool (4 rotated copies of each weight,
//!   over 24 MB L2), median of 40 launches: FP8 `dense_gemv_tcm::try_fp8` (needs
//!   `METRALE_GLM_GEMV_TC=1`, as the car runs), NVFP4 CUDA-core tier, NVFP4 tensor-core tier,
//!   and the routed `sweep`. Prints `M=<m> old_us=.. new_us=.. old_GBs=.. new_GBs=..` (old =
//!   FP8, new = routed sweep) and a `TIERS` line with all three; GB/s above 255 prints VOID.
//!   Timing is reported, never gated; `NOTE` lines mark an M where FP8 beats the best NVFP4 tier
//!   (lower `ROUTE_MAX_M` / `TC_FROM_M` in `glm5next_lm_head_nv4.rs` from these).
//!
//! Prints `PASS ...` / `FAIL ...` per check and `PASS all` last when every check passed.
//!
//! Run (GPU):
//!   METRALE_GLM_GEMV_TC=1 METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash \
//!     METRALE_TARGET_QUANT=nvfp4 cargo run -p metrale-model-arch --release \
//!     --features cuda,gpu-examples --example glm_lm_head_nv4_shard_microtest

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_layer::dense_fp8 as df;
use metrale_model_arch::glm5next_lm_head_nv4 as nv4;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight};

#[path = "common/mtp_head_nvfp4_ref.rs"]
mod mtp_head_nvfp4_ref;
use mtp_head_nvfp4_ref::*;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const N: usize = 77440;
const K: usize = 4096;
const SMALL_N: usize = 7744;
const MAX_M: usize = 16;
const COPIES: usize = 4;
const WARMUP: usize = 8;
const TIMED: usize = 40;
const COS_MIN: f64 = 0.99999;
const NEAR_TIE: f64 = 0.01;
const CHECK_MS: [usize; 5] = [1, 3, 8, 9, 16];
const SPEED_MS: [usize; 9] = [1, 2, 3, 4, 5, 7, 8, 9, 16];

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what}: status {rc}");
    }
    Ok(())
}

/// Median microseconds of `f(i)` for launch index `i` over `TIMED` launches between CUDA events.
fn time_us(g: &dyn GpuBackend, s: u64, mut f: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    for i in 0..WARMUP {
        f(i)?;
    }
    g.synchronize(s)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    ck(unsafe { cuEventCreate(&mut e0, 0) }, "cuEventCreate")?;
    ck(unsafe { cuEventCreate(&mut e1, 0) }, "cuEventCreate")?;
    let mut us = Vec::with_capacity(TIMED);
    for i in 0..TIMED {
        ck(unsafe { cuEventRecord(e0, s) }, "cuEventRecord")?;
        f(WARMUP + i)?;
        ck(unsafe { cuEventRecord(e1, s) }, "cuEventRecord")?;
        ck(unsafe { cuEventSynchronize(e1) }, "cuEventSynchronize")?;
        let mut ms = 0f32;
        ck(
            unsafe { cuEventElapsedTime(&mut ms, e0, e1) },
            "cuEventElapsedTime",
        )?;
        us.push(ms as f64 * 1e3);
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    us.sort_by(f64::total_cmp);
    Ok(us[us.len() / 2])
}

fn gbs(bytes: usize, us: f64) -> String {
    let v = bytes as f64 / (us * 1e-6) / 1e9;
    if v > 255.0 {
        format!("VOID({v:.0})")
    } else {
        format!("{v:.1}")
    }
}

struct Report(Vec<String>);

impl Report {
    fn check(&mut self, ok: bool, what: String) {
        if ok {
            println!("PASS {what}");
        } else {
            println!("FAIL {what}");
            self.0.push(what);
        }
    }
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let s = g.default_stream();
    let mut rep = Report(Vec::new());

    // (a) Shard quantizer == rows of the whole-head quantization.
    {
        let vocab = 2 * SMALL_N;
        let wh = gen_weight(0x6e76_3453_4841_5244, vocab, K, true);
        let w = up_u16(g, &wh)?;
        let whole = df::quantize_nvfp4_copy(g, w, vocab, K, s)?;
        let shard = nv4::quantize_shard(g, w, vocab, K, SMALL_N, SMALL_N, s)?;
        let (Some(whole), Some(shard)) = (whole, shard) else {
            bail!("NVFP4 quantizer kernels missing");
        };
        let wp = dn_bytes(g, whole.weight, vocab * K / 2)?;
        let ws = dn_bytes(g, whole.weight_scale, vocab * K / 16)?;
        let sp = dn_bytes(g, shard.weight, SMALL_N * K / 2)?;
        let ss = dn_bytes(g, shard.weight_scale, SMALL_N * K / 16)?;
        let same_p = wp[SMALL_N * K / 2..] == sp[..];
        let same_s = ws[SMALL_N * K / 16..] == ss[..];
        let same_g = whole.weight_scale_2.to_bits() == shard.weight_scale_2.to_bits();
        rep.check(
            same_p && same_s && same_g,
            format!(
                "shard quantizer bitwise == rows of the whole-head quantizer (packed {same_p}, scales \
                 {same_s}, tensor scale {same_g}: {} vs {})",
                shard.weight_scale_2, whole.weight_scale_2
            ),
        );
        for p in [
            w,
            whole.weight,
            whole.weight_scale,
            shard.weight,
            shard.weight_scale,
        ] {
            g.free(p)?;
        }
    }

    // One shard (head == shard, begin 0) for the rest.
    let t0 = std::time::Instant::now();
    let w_host = gen_weight(0x6e76_3453_4841_5244 ^ 7, N, K, false);
    let w = up_u16(g, &w_host)?;
    let mut f8s: Vec<Fp8DenseWeight> = Vec::new();
    let mut q4s: Vec<QuantizedWeight> = Vec::new();
    for _ in 0..COPIES {
        f8s.push(metrale_model_layers::weight_map::quantize_to_fp8(
            &DenseWeight { weight: w },
            N,
            K,
            g,
            g.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
            s,
        )?);
        let Some(q) = nv4::quantize_shard(g, w, N, K, 0, N, s)? else {
            bail!("NVFP4 quantizer kernels missing");
        };
        q4s.push(q);
    }
    println!(
        "SHAPE shard [{N} x {K}] fp8 {} MB, nvfp4 {} MB per copy, {COPIES} copies (setup {:.1} s)",
        (N * K + 4 * N) / (1 << 20),
        (N * K / 2 + N * K / 16) / (1 << 20),
        t0.elapsed().as_secs_f64()
    );
    let q = q4s[0];
    if !ops::dense_gemv_tcm::ready(g, 1, N as u32, K as u32, true) {
        rep.check(
            false,
            "FP8 tensor-core head not routed: run with METRALE_GLM_GEMV_TC=1".into(),
        );
    }

    let mut rng = Lcg(0x6e76_3453_4841_5244 ^ 0x1234_5678);
    let act = gen_act(&mut rng, MAX_M * K);
    let ad = up_u16(g, &act)?;
    let out = g.alloc(MAX_M * N * 2)?;
    let out2 = g.alloc(MAX_M * N * 2)?;
    let qh = Nv4Host {
        packed: dn_bytes(g, q.weight, N * K / 2)?,
        scales: dn_bytes(g, q.weight_scale, N * K / 16)?,
        s2: q.weight_scale_2 as f64,
    };
    let (_rb, ro) = fp64_refs(&w_host, &qh, &act, MAX_M, N, K);

    // (b) Correctness per tier and routed.
    for (name, tier) in [
        ("cuda_core", Some(false)),
        ("tensor_core", Some(true)),
        ("routed", None),
    ] {
        for &m in &CHECK_MS {
            g.memset(out, 0, MAX_M * N * 2)?;
            let launched = match tier {
                Some(tc) => nv4::sweep_tier(g, ad, &q, out, m, N, K, tc, s)?,
                None => {
                    nv4::sweep(g, ad, &q, out, m, N, K, s)?;
                    true
                }
            };
            if !launched {
                rep.check(false, format!("{name} M={m}: tier did not take the shape"));
                continue;
            }
            g.synchronize(s)?;
            let y = dn_u16(g, out, m * N)?;
            let (mut min_cos, mut nf, mut bad_argmax) = (1.0f64, 0usize, 0usize);
            for t in 0..m {
                let r = t * N..(t + 1) * N;
                let yf: Vec<f64> = y[r.clone()].iter().map(|&b| bf(b)).collect();
                let (c, _mr, n1) = row_stats(&yf, &ro[r.clone()]);
                min_cos = min_cos.min(c);
                nf += n1;
                let (ak, ar) = (argmax(&y[r.clone()]), {
                    let row = &ro[r.clone()];
                    (0..N).fold(0, |b, i| if row[i] > row[b] { i } else { b })
                });
                if ak != ar {
                    let row = &ro[r];
                    let top = row[ar].abs().max(1e-30);
                    if (row[ar] - row[ak]).abs() > NEAR_TIE * top {
                        bad_argmax += 1;
                    }
                }
            }
            rep.check(
                min_cos >= COS_MIN && nf == 0 && bad_argmax == 0,
                format!(
                    "{name} M={m}: min_cos={min_cos:.8} (>= {COS_MIN}) nonfinite={nf} argmax_mismatch={bad_argmax}"
                ),
            );
        }
    }

    // (c) Row bits.
    {
        let mut one = Vec::new();
        for t in 0..MAX_M {
            nv4::sweep_tier(
                g,
                ad.offset(t * K * 2),
                &q,
                out2.offset(t * N * 2),
                1,
                N,
                K,
                false,
                s,
            )?;
        }
        g.synchronize(s)?;
        one.extend(dn_u16(g, out2, MAX_M * N)?);
        for m in [2usize, 3] {
            nv4::sweep_tier(g, ad, &q, out, m, N, K, false, s)?;
            g.synchronize(s)?;
            let y = dn_u16(g, out, m * N)?;
            let diff = y.iter().zip(&one[..m * N]).filter(|(a, b)| a != b).count();
            rep.check(
                diff == 0,
                format!("cuda_core rows at M={m} bitwise == the M=1 launch ({diff} diffs)"),
            );
        }
        if nv4::sweep_tier(g, ad, &q, out2, MAX_M, N, K, true, s)? {
            g.synchronize(s)?;
            let full = dn_u16(g, out2, MAX_M * N)?;
            for m in [4usize, 5, 7, 8, 9] {
                nv4::sweep_tier(g, ad, &q, out, m, N, K, true, s)?;
                g.synchronize(s)?;
                let y = dn_u16(g, out, m * N)?;
                let diff = y.iter().zip(&full[..m * N]).filter(|(a, b)| a != b).count();
                rep.check(
                    diff == 0,
                    format!("tensor_core rows at M={m} bitwise == the M=16 launch ({diff} diffs)"),
                );
            }
        }
    }

    // (d) Layout: scratch + pitched copy into [M, 2N] logits at column N.
    {
        let (m, v, begin) = (3usize, 2 * N, N);
        let logits = g.alloc(m * v * 2)?;
        g.memset(logits, 0, m * v * 2)?;
        nv4::sweep(g, ad, &q, out, m, N, K, s)?;
        g.copy_d2d_2d_async(out, N * 2, logits.offset(begin * 2), v * 2, N * 2, m, s)?;
        g.synchronize(s)?;
        let lg = dn_u16(g, logits, m * v)?;
        let sc = dn_u16(g, out, m * N)?;
        let mut bad = 0usize;
        for t in 0..m {
            for j in 0..v {
                let want = if j >= begin {
                    sc[t * N + (j - begin)]
                } else {
                    0
                };
                bad += usize::from(lg[t * v + j] != want);
            }
        }
        rep.check(
            bad == 0,
            format!("pitched scatter into [M, 2N] logits, {bad} wrong elements"),
        );
        g.free(logits)?;
    }

    // (e) Speed, cold pool.
    let (b8, bn) = (N * K + 4 * N, N * K / 2 + N * K / 16);
    for &m in &SPEED_MS {
        let t8 = time_us(g, s, |i| {
            let ok = ops::dense_gemv_tcm::try_fp8(
                g,
                ad,
                &f8s[i % COPIES],
                out,
                m as u32,
                N as u32,
                K as u32,
                N as u32,
                s,
            )?;
            anyhow::ensure!(ok, "FP8 tensor-core head not routed");
            Ok(())
        })?;
        let tcc = time_us(g, s, |i| {
            nv4::sweep_tier(g, ad, &q4s[i % COPIES], out, m, N, K, false, s).map(|_| ())
        })?;
        let ttc = if nv4::sweep_tier(g, ad, &q, out, m, N, K, true, s)? {
            Some(time_us(g, s, |i| {
                nv4::sweep_tier(g, ad, &q4s[i % COPIES], out, m, N, K, true, s).map(|_| ())
            })?)
        } else {
            None
        };
        let tnew = time_us(g, s, |i| {
            nv4::sweep(g, ad, &q4s[i % COPIES], out, m, N, K, s)
        })?;
        println!(
            "M={m} old_us={t8:.1} new_us={tnew:.1} old_GBs={} new_GBs={}",
            gbs(b8, t8),
            gbs(bn, tnew)
        );
        println!(
            "TIERS M={m} fp8_us={t8:.1} nv4_cuda_core_us={tcc:.1} nv4_tensor_core_us={} fp8_GBs={} \
             nv4_cuda_core_GBs={} nv4_tensor_core_GBs={}",
            ttc.map_or("n/a".into(), |t| format!("{t:.1}")),
            gbs(b8, t8),
            gbs(bn, tcc),
            ttc.map_or("n/a".into(), |t| gbs(bn, t)),
        );
        let best = ttc.map_or(tcc, |t| t.min(tcc));
        if t8 < best {
            println!(
                "NOTE M={m}: FP8 {t8:.1} us beats the best NVFP4 tier {best:.1} us; lower \
                 ROUTE_MAX_M / re-tier TC_FROM_M in glm5next_lm_head_nv4.rs"
            );
        }
    }

    let _: DevicePtr = ad;
    if rep.0.is_empty() {
        println!("PASS all");
        Ok(())
    } else {
        println!("FAIL {} check(s)", rep.0.len());
        std::process::exit(1);
    }
}
