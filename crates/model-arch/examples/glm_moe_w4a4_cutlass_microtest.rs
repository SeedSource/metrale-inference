// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: GPU numerics + timing gate for `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1`
//! (`forward_prefill_gemm/cutlass_w4a4.rs` on the CUTLASS Sm120 block-scaled NVFP4 grouped GEMM,
//! `crates/gpu-runtime/cuda/cutlass_nvfp4_grouped_gemm.cu`, NVIDIA CUTLASS, BSD-3-Clause) at the
//! GLM-5.3 EP=2 routed-MoE shapes.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - Shapes: hidden 4096, moe_intermediate 2048, 288 experts, top_k 8, EP=2 (experts 0..144
//!   local, the rest NULL); gate/up K=4096 N=2048 each, down K=2048 N=4096. Each local expert has
//!   its own device copy of one of 4 host templates per projection. Routing: distinct top-8 per
//!   token from a Zipf-like (s = 0.9) popularity. Token windows: 256, 2048, 8192.
//! - Activations: BF16, per-channel lognormal magnitudes with a few outlier channels (heavier
//!   tailed than uniform, like post-norm hidden states).
//! - Arms, all from the same sort output and the same NVFP4 weights:
//!   (a) production W4A16: `_bt_m128_k64_va2` gate and up, `glm5next_swiglu_clamp`, `_va2` down;
//!   (b) CUTLASS W4A4 through the crate's own `cutlass_w4a4::{SfbCache, MoeTables, run}`, with a
//!       STATIC activation scale (stand-in for the checkpoint `input_scale`: amax / (6 * 448) of
//!       this window's x for gate/up, of arm (a)'s activation for down), and again with the
//!       DYNAMIC fallback (`input_scale` 0.0, amax computed on the device);
//!   (c) FP32 reference on sampled local rows: exact dequantized NVFP4 weights, BF16 x, gate/up
//!       and the SwiGLU output rounded to BF16 as the GPU paths store them.
//! - Per case: cosine, max_abs, max_rel (= max_abs / max |ref|), nonfinite, for act and out, each
//!   GPU arm vs (c). Timing per call (ms, TFLOP/s over the local rows) for (a) and (b), plus the
//!   once-per-layer SFB swizzle. Lines are labelled ok / below_screen.
//! - Ends with exactly one verdict line: `PASS: ...` when arm (b) ran, every output is finite
//!   and cos(b, c) >= 0.98 on `out` for both scale modes in every case; otherwise a single
//!   verdict line naming the reason.
//!
//!   cargo run -p metrale-model-arch --release --example glm_moe_w4a4_cutlass_microtest \
//!       --features cuda,gpu-examples      (build with CUTLASS_HOME set, or arm (b) cannot run)
//!   Env: GLM_W4A4_MT_ITERS (timed iterations, 10), GLM_W4A4_MT_REF_ROWS (sampled rows per case
//!   for the FP32 reference, 64).

use std::time::Instant;

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::Glm5NextExpertWeights;
use metrale_model_arch::glm5next_mlp::forward_prefill_gemm::cutlass_w4a4::{
    MoeTables, RunArgs, SfbCache, run,
};
use metrale_model_arch::glm5next_mlp::weights::Nvfp4Proj;

const H: usize = 4096;
const MI: usize = 2048;
const E: usize = 288;
const TOP_K: usize = 8;
const LOCAL: usize = 144;
const LIMIT: f32 = 10.0;
const TOKEN_WINDOWS: [usize; 3] = [256, 2048, 8192];
const COS_MIN: f64 = 0.98;
const VA2: &str = "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va2";
const TEMPLATES: usize = 4;
/// 2026-10-03: The NVFP4 global-scale denominator (E2M1 max 6 x E4M3 max 448).
const FP4_GS_DEN: f32 = 6.0 * 448.0;

struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.bits() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// 2026-10-03: Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn e2m1(code: u8) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = V[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

fn e4m3(code: u8) -> f32 {
    let s = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((code >> 3) & 0xF) as i32;
    let m = (code & 7) as f32;
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// 2026-10-03: One projection template: packed `[n, k/2]` (low nibble = even k), E4M3 scales
/// `[n, k/16]` with codes 0x30..=0x47 (0.5..=3.75), and its exact FP32 dequant `[n, k]`
/// (without the per-expert global scale).
struct Tpl {
    packed: Vec<u8>,
    scale: Vec<u8>,
    deq: Vec<f32>,
}

fn make_tpl(r: &mut Rng, n: usize, k: usize) -> Tpl {
    let mut packed = vec![0u8; n * k / 2];
    for c in packed.chunks_mut(8) {
        let w = r.bits().to_le_bytes();
        c.copy_from_slice(&w[..c.len()]);
    }
    let scale: Vec<u8> = (0..n * k / 16)
        .map(|_| 0x30 + (r.bits() % 0x18) as u8)
        .collect();
    let mut deq = vec![0f32; n * k];
    for row in 0..n {
        for kk in 0..k {
            let b = packed[(row * k + kk) / 2];
            let code = if kk & 1 == 1 { b >> 4 } else { b & 0xF };
            deq[row * k + kk] = e2m1(code) * e4m3(scale[row * (k / 16) + kk / 16]);
        }
    }
    Tpl { packed, scale, deq }
}

/// 2026-10-03: A pointer table over all `E` experts (device tables + the host vectors the
/// CUTLASS arm needs); local expert `e` is a device copy of template `e % TEMPLATES`.
struct Table {
    packed: DevicePtr,
    scale: DevicePtr,
    s2: DevicePtr,
    packed_host: Vec<u64>,
    scale_host: Vec<u64>,
    s2_host: Vec<f32>,
    tpl: Vec<Tpl>,
}

fn build_table(
    g: &dyn GpuBackend,
    r: &mut Rng,
    n: usize,
    k: usize,
    owned: &mut Vec<DevicePtr>,
) -> Result<Table> {
    let tpl: Vec<Tpl> = (0..TEMPLATES).map(|_| make_tpl(r, n, k)).collect();
    let (mut pp, mut sp, mut s2) = (vec![0u64; E], vec![0u64; E], vec![0f32; E]);
    for e in 0..LOCAL {
        let t = &tpl[e % TEMPLATES];
        let (p, s) = (up(g, &t.packed)?, up(g, &t.scale)?);
        owned.extend([p, s]);
        pp[e] = p.0;
        sp[e] = s.0;
        // 2026-10-03: ~1/256 keeps gate/up near unit scale, so the SwiGLU clamp is exercised
        // without saturating every value (as the W4A16 microtests).
        s2[e] = (0.5 + (r.bits() % 64) as f32 / 64.0) / 256.0;
    }
    let t = Table {
        packed: up(g, &le(&pp, u64::to_le_bytes))?,
        scale: up(g, &le(&sp, u64::to_le_bytes))?,
        s2: up(g, &le(&s2, f32::to_le_bytes))?,
        packed_host: pp,
        scale_host: sp,
        s2_host: s2,
        tpl,
    };
    owned.extend([t.packed, t.scale, t.s2]);
    Ok(t)
}

/// 2026-10-03: Distinct top-8 ids per token from a skewed popularity: expert `e` has weight
/// 1 / (rank[e] + 1)^0.9, `rank` a random permutation.
fn make_routing(r: &mut Rng, rows: usize) -> Vec<i32> {
    let mut rank: Vec<usize> = (0..E).collect();
    for i in (1..E).rev() {
        rank.swap(i, (r.bits() % (i as u64 + 1)) as usize);
    }
    let mut acc = 0.0;
    let mut cum = Vec::with_capacity(E);
    for &k in &rank {
        acc += 1.0 / ((k + 1) as f64).powf(0.9);
        cum.push(acc);
    }
    for c in cum.iter_mut() {
        *c /= acc;
    }
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut picked: Vec<i32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let u = r.unit();
            let e = cum.partition_point(|&x| x < u).min(E - 1) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids
}

fn bf(b: &[u8], i: usize) -> f32 {
    bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32()
}

fn round_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

/// 2026-10-03: FP32 reference for one sorted row: (act `[MI]`, out `[H]`).
fn reference_row(
    x: &[f32],
    e: usize,
    gate: &Table,
    upt: &Table,
    down: &Table,
) -> (Vec<f32>, Vec<f32>) {
    let t = e % TEMPLATES;
    let (gd, ud, dd) = (&gate.tpl[t].deq, &upt.tpl[t].deq, &down.tpl[t].deq);
    let mut act = vec![0f32; MI];
    for (n, a) in act.iter_mut().enumerate() {
        let (mut sg, mut su) = (0f64, 0f64);
        let (gr, ur) = (&gd[n * H..(n + 1) * H], &ud[n * H..(n + 1) * H]);
        for k in 0..H {
            sg += (x[k] * gr[k]) as f64;
            su += (x[k] * ur[k]) as f64;
        }
        let gv = round_bf16(sg as f32 * gate.s2_host[e]).min(LIMIT);
        let uv = round_bf16(su as f32 * upt.s2_host[e]).clamp(-LIMIT, LIMIT);
        *a = round_bf16(gv / (1.0 + (-gv).exp()) * uv);
    }
    let mut out = vec![0f32; H];
    for (n, o) in out.iter_mut().enumerate() {
        let dr = &dd[n * MI..(n + 1) * MI];
        let mut s = 0f64;
        for k in 0..MI {
            s += (act[k] * dr[k]) as f64;
        }
        *o = s as f32 * down.s2_host[e];
    }
    (act, out)
}

/// 2026-10-03: Error stats of GPU rows against the reference rows.
#[derive(Default, Clone, Copy)]
struct Stats {
    dot: f64,
    nw: f64,
    ng: f64,
    max_abs: f64,
    max_ref: f64,
    nonfinite: usize,
}

impl Stats {
    fn add(&mut self, want: &[f32], got: impl Iterator<Item = f32>) {
        for (w, g) in want.iter().zip(got) {
            let (w, g) = (*w as f64, g as f64);
            if !g.is_finite() {
                self.nonfinite += 1;
                continue;
            }
            self.dot += w * g;
            self.nw += w * w;
            self.ng += g * g;
            self.max_abs = self.max_abs.max((w - g).abs());
            self.max_ref = self.max_ref.max(w.abs());
        }
    }
    fn cos(&self) -> f64 {
        self.dot / (self.nw.sqrt() * self.ng.sqrt()).max(1e-300)
    }
    fn line(&self) -> String {
        format!(
            "cosine={:.6} max_abs={:.4e} max_rel={:.3e} nonfinite={}",
            self.cos(),
            self.max_abs,
            self.max_abs / self.max_ref.max(1e-30),
            self.nonfinite
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn va2(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    c: DevicePtr,
    off: DevicePtr,
    stid: DevicePtr,
    n: usize,
    kk: usize,
    m_tiles: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(n / 64) as u32, m_tiles, E as u32])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.s2)
        .arg_ptr(c)
        .arg_ptr(off)
        .arg_ptr(stid)
        .arg_u32(E as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(0)
}

fn time_ms(g: &dyn GpuBackend, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    f()?;
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1.0e3 / iters as f64)
}

/// 2026-10-03: Max |v| over `[lo, hi)` rows of a `[*, width]` BF16 buffer.
fn amax_rows(b: &[u8], ranges: &[(usize, usize)], width: usize) -> f32 {
    let mut m = 0f32;
    for &(lo, hi) in ranges {
        for i in lo * width..hi * width {
            let v = bf(b, i).abs();
            if v.is_finite() {
                m = m.max(v);
            }
        }
    }
    m
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    if !metrale_gpu_runtime::cutlass::available() {
        println!(
            "FAIL: arm (b) cannot run: this build has no CUTLASS objects (build with CUTLASS_HOME \
             pointing at CUTLASS v4.6.0)"
        );
        bail!("glm_moe_w4a4_cutlass_microtest: no CUTLASS in this build");
    }
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let gb = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gb;
    let iters = env_usize("GLM_W4A4_MT_ITERS", 10).max(1);
    let ref_rows = env_usize("GLM_W4A4_MT_REF_ROWS", 64).max(1);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    let k_sort = g.kernel("moe", "moe_sort_by_expert")?;
    let k_va2 = g.kernel("moe_w4a16", VA2)?;
    let k_swiglu = g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?;

    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let mut owned = Vec::new();
    let gate = build_table(g, &mut r, MI, H, &mut owned)?;
    let upt = build_table(g, &mut r, MI, H, &mut owned)?;
    let down = build_table(g, &mut r, H, MI, &mut owned)?;

    // 2026-10-03: The production SFB cache, filled once (one layer), timed separately.
    let cache = SfbCache::new(g, LOCAL, H, MI)?;
    let fill = || cache.fill(gate.scale, upt.scale, down.scale, 0, LOCAL, 0);
    let t_swizzle = time_ms(g, iters, fill)?;
    println!(
        "SFB cache {:.1} MB for {LOCAL} experts, swizzle_ms_per_layer={t_swizzle:.3}",
        SfbCache::bytes(LOCAL, H, MI) as f64 / 1e6
    );
    let experts_for = |gu_is: f32, dn_is: f32| -> Vec<Glm5NextExpertWeights> {
        let p = |t: &Table, e: usize, is: f32| Nvfp4Proj {
            packed: DevicePtr(t.packed_host[e]),
            scale: DevicePtr(t.scale_host[e]),
            scale_2: t.s2_host[e],
            input_scale: is,
        };
        (0..LOCAL)
            .map(|e| Glm5NextExpertWeights {
                gate_proj: p(&gate, e, gu_is),
                up_proj: p(&upt, e, gu_is),
                down_proj: p(&down, e, dn_is),
            })
            .collect()
    };

    // 2026-10-03: Heavy-tailed activation channel magnitudes: lognormal(0, 0.5), with 8
    // outlier channels at 12-20x.
    let mut chan: Vec<f32> = (0..H).map(|_| (0.5 * r.normal()).exp() as f32).collect();
    for _ in 0..8 {
        let c = (r.bits() % H as u64) as usize;
        chan[c] *= 12.0 + 8.0 * r.unit() as f32;
    }

    let mut problems: Vec<String> = Vec::new();
    let mut summary: Vec<String> = Vec::new();
    for tokens in TOKEN_WINDOWS {
        let te = tokens * TOP_K;
        let x_f: Vec<f32> = (0..tokens * H)
            .map(|i| round_bf16(r.normal() as f32 * 0.5 * chan[i % H]))
            .collect();
        let x: Vec<u8> = le(&x_f, |v| bf16::from_f32(v).to_bits().to_le_bytes());
        let d_x = up(g, &x)?;
        let d_ids = up(g, &le(&make_routing(&mut r, tokens), i32::to_le_bytes))?;
        let d_stid = g.alloc(te * 4)?;
        let d_seid = g.alloc(te * 4)?;
        let d_t2p = g.alloc(te * 4)?;
        let d_off = g.alloc((E + 1) * 4)?;
        KernelLaunch::new(g, k_sort)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_ids)
            .arg_ptr(d_stid)
            .arg_ptr(d_seid)
            .arg_ptr(d_off)
            .arg_ptr(d_t2p)
            .arg_u32(te as u32)
            .arg_u32(E as u32)
            .arg_u32(TOP_K as u32)
            .launch(0)?;
        g.synchronize(0)?;
        let off_i32: Vec<i32> = dn(g, d_off, (E + 1) * 4)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let off: Vec<usize> = off_i32.iter().map(|&v| v as usize).collect();
        let stid: Vec<usize> = dn(g, d_stid, te * 4)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let local: Vec<(usize, usize)> = (0..LOCAL).map(|e| (off[e], off[e + 1])).collect();
        let r_local: usize = local.iter().map(|(a, b)| b - a).sum();
        let busiest = (0..LOCAL).map(|e| off[e + 1] - off[e]).max().unwrap_or(0);
        let flop = 6.0 * r_local as f64 * (MI * H) as f64;
        println!(
            "tokens={tokens} te={te} local_rows={r_local} busiest_local_expert_rows={busiest} \
             empty_local_experts={}",
            local.iter().filter(|(a, b)| a == b).count()
        );

        let a_gate = g.alloc(te * MI * 2)?;
        let a_up = g.alloc(te * MI * 2)?;
        let act_a = g.alloc(te * MI * 2)?;
        let act_b = g.alloc(te * MI * 2)?;
        let out_a = g.alloc(te * H * 2)?;
        let out_b = g.alloc(te * H * 2)?;

        // 2026-10-03: Arm (a): the production W4A16 prefill path at the VA2 tile.
        let m_tiles = busiest.div_ceil(128).max(1) as u32;
        let arm_a = || -> Result<()> {
            va2(g, k_va2, d_x, &gate, a_gate, d_off, d_stid, MI, H, m_tiles)?;
            va2(g, k_va2, d_x, &upt, a_up, d_off, d_stid, MI, H, m_tiles)?;
            KernelLaunch::new(g, k_swiglu)
                .grid([((te * MI) as u32).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_gate)
                .arg_ptr(a_up)
                .arg_ptr(act_a)
                .arg_u32((te * MI) as u32)
                .arg_f32(LIMIT)
                .launch(0)?;
            va2(
                g,
                k_va2,
                act_a,
                &down,
                out_a,
                d_off,
                DevicePtr(0),
                H,
                MI,
                m_tiles,
            )
        };
        g.memset_async(out_a, 0, te * H * 2, 0)?;
        arm_a()?;
        g.synchronize(0)?;
        let got_act_a = dn(g, act_a, te * MI * 2)?;
        let got_out_a = dn(g, out_a, te * H * 2)?;
        let t_a = time_ms(g, iters, arm_a)?;
        println!(
            "TIMING tokens={tokens} arm=a_w4a16_va2 ms={t_a:.3} TFLOP/s={:.1}",
            flop / (t_a * 1e9)
        );

        // 2026-10-03: Sampled local sorted rows (evenly spaced) and their FP32 reference.
        let local_sorted: Vec<(usize, usize)> = (0..LOCAL)
            .flat_map(|e| (off[e]..off[e + 1]).map(move |i| (i, e)))
            .collect();
        let n_ref = ref_rows.min(local_sorted.len());
        let picks: Vec<(usize, usize)> = (0..n_ref)
            .map(|j| local_sorted[j * local_sorted.len() / n_ref])
            .collect();
        let refs: Vec<(Vec<f32>, Vec<f32>)> = std::thread::scope(|s| {
            let chunk = picks.len().div_ceil(threads).max(1);
            let handles: Vec<_> = picks
                .chunks(chunk)
                .map(|part| {
                    let (x_f, stid, gate, upt, down) = (&x_f, &stid, &gate, &upt, &down);
                    s.spawn(move || {
                        part.iter()
                            .map(|&(i, e)| {
                                let tok = stid[i];
                                reference_row(&x_f[tok * H..(tok + 1) * H], e, gate, upt, down)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("reference thread"))
                .collect()
        });
        let stats = |act: &[u8], out: &[u8]| -> (Stats, Stats) {
            let (mut sa, mut so) = (Stats::default(), Stats::default());
            for ((i, _), (ra, ro)) in picks.iter().zip(&refs) {
                sa.add(ra, (0..MI).map(|c| bf(act, i * MI + c)));
                so.add(ro, (0..H).map(|c| bf(out, i * H + c)));
            }
            (sa, so)
        };
        let (sa_a, so_a) = stats(&got_act_a, &got_out_a);
        println!(
            "  tokens={tokens} a_vs_ref act  {} ref_rows={n_ref}",
            sa_a.line()
        );
        println!("  tokens={tokens} a_vs_ref out  {}", so_a.line());

        // 2026-10-03: Arm (b): static scale = amax / (6 * 448) of this window's x (gate/up) and
        // of arm (a)'s activation over the local rows (down); then the dynamic fallback.
        let x_amax = x_f.iter().fold(0f32, |m, v| m.max(v.abs()));
        let act_amax = amax_rows(&got_act_a, &local, MI);
        let gu_static = x_amax / FP4_GS_DEN;
        let dn_static = act_amax / FP4_GS_DEN;
        for (mode, gu_is, dn_is) in [("static", gu_static, dn_static), ("dynamic", 0.0, 0.0)] {
            let experts = experts_for(gu_is, dn_is);
            let tables = MoeTables::build(E, 0..LOCAL, &experts, &cache.layout());
            let args = RunArgs {
                swiglu: k_swiglu,
                x: d_x,
                sorted_token_ids: d_stid,
                a_gate,
                a_up,
                a_act: act_b,
                expert_out: out_b,
                te,
                hidden: H,
                moe_intermediate: MI,
                swiglu_limit: LIMIT,
                offsets: &off_i32,
                tables: &tables,
            };
            g.memset_async(act_b, 0, te * MI * 2, 0)?;
            g.memset_async(out_b, 0, te * H * 2, 0)?;
            if let Err(e) = run(g, &args, 0).and_then(|_| g.synchronize(0)) {
                problems.push(format!(
                    "tokens={tokens} {mode}: CUTLASS W4A4 run failed: {e:#}"
                ));
                println!("  tokens={tokens} b_{mode} run error: {e:#}");
                continue;
            }
            let got_act_b = dn(g, act_b, te * MI * 2)?;
            let got_out_b = dn(g, out_b, te * H * 2)?;
            let (sa, so) = stats(&got_act_b, &got_out_b);
            let (_, so_ab) = {
                // 2026-10-03: (b) vs (a) over the same sampled rows, for context.
                let want: Vec<(Vec<f32>, Vec<f32>)> = picks
                    .iter()
                    .map(|&(i, _)| {
                        (
                            (0..MI).map(|c| bf(&got_act_a, i * MI + c)).collect(),
                            (0..H).map(|c| bf(&got_out_a, i * H + c)).collect(),
                        )
                    })
                    .collect();
                let (mut x1, mut x2) = (Stats::default(), Stats::default());
                for ((i, _), (wa, wo)) in picks.iter().zip(&want) {
                    x1.add(wa, (0..MI).map(|c| bf(&got_act_b, i * MI + c)));
                    x2.add(wo, (0..H).map(|c| bf(&got_out_b, i * H + c)));
                }
                (x1, x2)
            };
            let label = |s: &Stats| {
                if s.nonfinite == 0 && s.cos() >= COS_MIN {
                    "ok"
                } else {
                    "below_screen"
                }
            };
            println!(
                "  tokens={tokens} b_{mode}_vs_ref act  {} {} gs_gate_up={:.4e} gs_down={:.4e}",
                sa.line(),
                label(&sa),
                gu_is,
                dn_is
            );
            println!(
                "  tokens={tokens} b_{mode}_vs_ref out  {} {}",
                so.line(),
                label(&so)
            );
            println!("  tokens={tokens} b_{mode}_vs_a   out  {}", so_ab.line());
            if so.nonfinite > 0 || sa.nonfinite > 0 {
                problems.push(format!("tokens={tokens} {mode}: non-finite output"));
            }
            if so.cos() < COS_MIN {
                problems.push(format!(
                    "tokens={tokens} {mode}: cos(b,ref) out {:.5} < {COS_MIN}",
                    so.cos()
                ));
            }
            let t_b = time_ms(g, iters, || run(g, &args, 0))?;
            println!(
                "TIMING tokens={tokens} arm=b_cutlass_w4a4_{mode} ms={t_b:.3} TFLOP/s={:.1} \
                 speedup_vs_a={:.3}x saved_ms_per_window={:.3}",
                flop / (t_b * 1e9),
                t_a / t_b,
                t_a - t_b
            );
            summary.push(format!(
                "{tokens}/{mode} cos={:.4} x{:.2}",
                so.cos(),
                t_a / t_b
            ));
        }

        for p in [
            d_x, d_ids, d_stid, d_seid, d_t2p, d_off, a_gate, a_up, act_a, act_b,
        ] {
            g.free(p)?;
        }
        for p in [out_a, out_b] {
            g.free(p)?;
        }
    }
    for p in owned {
        g.free(p)?;
    }
    if !problems.is_empty() {
        println!("FAIL: {}", problems.join("; "));
        bail!(
            "glm_moe_w4a4_cutlass_microtest: {} problem(s)",
            problems.len()
        );
    }
    println!(
        "PASS: CUTLASS W4A4 finite, cos(b,ref) out >= {COS_MIN} in every case ({})",
        summary.join(", ")
    );
    Ok(())
}
