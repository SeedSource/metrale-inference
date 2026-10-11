// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Bit-parity + timing gate for `METRALE_CUTLASS_W4A4_GROUPED_TILE`, the tile / schedule
//! variants of the CUTLASS Sm120 NVFP4 grouped GEMM behind the W4A4 MoE prefill
//! (`crates/gpu-runtime/cuda/cutlass_nvfp4_grouped_gemm.cu`, NVIDIA CUTLASS, BSD-3-Clause;
//! `cutlass::W4a4GemmTile`).
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - Shapes: hidden 4096, 288 experts, top_k 8; `GLM_MT_MI` / `GLM_MT_LOCAL` default to the
//!   expert-TP sizing the car serves (`METRALE_GLM_EXPERT_TP=1`: 1024 / 288, every expert local
//!   at half the intermediate width). Distinct top-8 routing per token from a Zipf-like (s = 0.9)
//!   popularity. Token windows 2048 and 8192. The car's W4A4 sub-levers: dynamic global scales
//!   (`METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE=1` tables), gate/up with pack-once
//!   (`METRALE_CUTLASS_W4A4_PACK_ONCE=1`), down with the SwiGLU amax
//!   (`METRALE_GLM_MOE_SWIGLU_AMAX=1`).
//! - Per window, every variant (`cutlass::set_w4a4_gemm_tile_override`) runs the gate/up call
//!   and the down call; the down input `a_act` (and its amax slot) is computed once, from the
//!   default's gate/up output, so every variant's down reads the same bytes. Outputs are filled
//!   with a 0xA5 sentinel before each call. Each variant's `a_gate`, `a_up` and `expert_out` must
//!   be bitwise equal to the default's (whole buffers), the C side must report that variant as
//!   the one launched with no fallback to the default, and the default must have written its
//!   outputs (not all sentinel).
//! - Timing, per variant, mean over `GLM_W4A4_MT_ITERS` (at least 10) calls after a warm-up:
//!   the gate/up call and the down call (A-side prep + GEMMs, as the serve calls them), and the
//!   same calls with the GEMMs skipped (`cutlass::set_w4a4_gemm_skip`): GEMM time = call minus
//!   prep.
//! - Output lines: `TILE <variant> tokens=<t>: <n> differing bytes ...` per variant and window;
//!   `TILE_TIMING tokens=<t> variant=<v> gateup_ms=.. down_ms=.. total_ms=.. ratio_vs_default=..
//!   gemm_ms=.. gemm_ratio=..` (`ratio_vs_default` on total_ms); `TILE_BEST tokens=<t>
//!   variant=<v> total_ratio=.. ratio=..` (the variant with the lowest GEMM time, default
//!   included; `ratio` is its GEMM ratio and is the last `ratio=` on the line). Ends with one
//!   line starting `PASS` when every variant ran and matched, else a `FAIL:` line and a nonzero
//!   exit.
//!
//!   cargo run -p metrale-model-arch --release --example glm_moe_w4a4_tile_microtest \
//!       --features cuda,gpu-examples      (CUTLASS_HOME set at build time)
//!   Env: GLM_W4A4_MT_ITERS (timed iterations, 10, at least 10), GLM_MT_MI, GLM_MT_LOCAL.

use std::time::Instant;

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::cutlass::{self, W4a4GemmTile};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::Glm5NextExpertWeights;
use metrale_model_arch::glm5next_mlp::forward_prefill_gemm::cutlass_w4a4::{
    MoeTables, SfbCache, swiglu_rows_amax, swiglu_span,
};
use metrale_model_arch::glm5next_mlp::weights::Nvfp4Proj;

const H: usize = 4096;
const E: usize = 288;
const TOP_K: usize = 8;
const LIMIT: f32 = 10.0;
const WINDOWS: [usize; 2] = [2048, 8192];
const SENTINEL: u8 = 0xA5;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

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
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn le_u64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ndiff(x: &[u8], y: &[u8]) -> usize {
    x.iter().zip(y).filter(|(a, b)| a != b).count() + x.len().abs_diff(y.len())
}

/// 2026-10-07: One projection over the local experts: per-expert packed `[n, k/2]` and E4M3
/// scales `[n, k/16]` (codes 0x30..=0x47), device pointer tables over all `E` ids.
struct Proj {
    packed: Vec<u64>,
    scale: Vec<u64>,
    s2: Vec<f32>,
    scale_tab: DevicePtr,
}

fn build_proj(g: &dyn GpuBackend, r: &mut Rng, n: usize, k: usize, local: usize) -> Result<Proj> {
    let (mut packed, mut scale, mut s2) = (vec![0u64; E], vec![0u64; E], vec![0f32; E]);
    for e in 0..local {
        let p: Vec<u8> = (0..n * k / 2).map(|_| r.bits() as u8).collect();
        let s: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x30 + (r.bits() % 0x18) as u8)
            .collect();
        packed[e] = up(g, &p)?.0;
        scale[e] = up(g, &s)?.0;
        s2[e] = (0.5 + (r.bits() % 64) as f32 / 64.0) / 256.0;
    }
    let scale_tab = up(g, &le_u64(&scale))?;
    Ok(Proj {
        packed,
        scale,
        s2,
        scale_tab,
    })
}

/// 2026-10-07: Distinct top-8 ids per token from a Zipf-like (s = 0.9) popularity.
fn routing(r: &mut Rng, rows: usize) -> Vec<u8> {
    let mut cum = Vec::with_capacity(E);
    let mut acc = 0.0;
    for e in 0..E {
        acc += 1.0 / ((e + 1) as f64).powf(0.9);
        cum.push(acc);
    }
    let mut ids = Vec::with_capacity(rows * TOP_K * 4);
    for _ in 0..rows {
        let mut picked: Vec<i32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let u = r.unit() * acc;
            let e = cum.partition_point(|&x| x < u).min(E - 1) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked.iter().flat_map(|v| v.to_le_bytes()));
    }
    ids
}

/// 2026-10-07: Mean ms per call over `iters` calls, after two warm-up calls.
fn time_ms(g: &dyn GpuBackend, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    f()?;
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// 2026-10-07: One variant's timing in a window (ms per call).
struct Timing {
    tile: W4a4GemmTile,
    gate_up: f64,
    down: f64,
    gemm: f64,
}

fn main() -> Result<()> {
    if !cutlass::available() {
        println!("FAIL: this build has no CUTLASS objects (build with CUTLASS_HOME set)");
        bail!("glm_moe_w4a4_tile_microtest: no CUTLASS in this build");
    }
    let mi = env_usize("GLM_MT_MI", 1024);
    let local = env_usize("GLM_MT_LOCAL", 288).min(E);
    let iters = env_usize("GLM_W4A4_MT_ITERS", 10).max(10);
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let gb = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gb;
    let k_sort = g.kernel("moe", "moe_sort_by_expert")?;
    let k_amax = g.kernel("glm5next_ffn", "glm5next_swiglu_clamp_amax")?;
    let (_, ws_size) = cutlass::workspace()?;
    let slot = g.alloc(256)?;
    println!(
        "mi={mi} local_experts={local} workspace={} MiB iters={iters} variants={}",
        ws_size >> 20,
        W4a4GemmTile::ALL.map(|t| t.name()).join(",")
    );

    let mut r = Rng(0x6A09_E667_F3BC_C909);
    let gate = build_proj(g, &mut r, mi, H, local)?;
    let upp = build_proj(g, &mut r, mi, H, local)?;
    let down = build_proj(g, &mut r, H, mi, local)?;
    let cache = SfbCache::new(g, local, H, mi)?;
    cache.fill(gate.scale_tab, upp.scale_tab, down.scale_tab, 0, local, 0)?;
    let nv = |p: &Proj, e: usize| Nvfp4Proj {
        packed: DevicePtr(p.packed[e]),
        scale: DevicePtr(p.scale[e]),
        scale_2: p.s2[e],
        input_scale: 0.0,
    };
    let experts: Vec<Glm5NextExpertWeights> = (0..local)
        .map(|e| Glm5NextExpertWeights {
            gate_proj: nv(&gate, e),
            up_proj: nv(&upp, e),
            down_proj: nv(&down, e),
        })
        .collect();
    let tables = MoeTables::build(E, 0..local, &experts, &cache.layout(), true);
    let mut chan: Vec<f32> = (0..H).map(|_| (0.5 * r.normal()).exp() as f32).collect();
    for _ in 0..8 {
        let c = (r.bits() % H as u64) as usize;
        chan[c] *= 12.0 + 8.0 * r.unit() as f32;
    }

    let mut problems: Vec<String> = Vec::new();
    let mut summary: Vec<String> = Vec::new();
    for tokens in WINDOWS {
        let te = tokens * TOP_K;
        let x: Vec<u8> = (0..tokens * H)
            .flat_map(|i| bf16::from_f32(r.normal() as f32 * 0.5 * chan[i % H]).to_le_bytes())
            .collect();
        let d_x = up(g, &x)?;
        let d_ids = up(g, &routing(&mut r, tokens))?;
        let (d_stid, d_seid, d_t2p) = (g.alloc(te * 4)?, g.alloc(te * 4)?, g.alloc(te * 4)?);
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
        let off: Vec<i32> = dn(g, d_off, (E + 1) * 4)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let span = swiglu_span(&off, 0..local, te, true);
        let rows = span.len();
        let flop = 6.0 * rows as f64 * (mi * H) as f64;
        println!("tokens={tokens} te={te} local_rows={rows}");
        let (a_gate, a_up) = (g.alloc(te * mi * 2)?, g.alloc(te * mi * 2)?);
        let (act, out) = (g.alloc(te * mi * 2)?, g.alloc(te * H * 2)?);
        let t = &tables;
        let gate_up = || -> Result<bool> {
            cutlass::nvfp4_grouped_gate_up_w4a4_ex(
                d_x.0,
                d_stid.0,
                &t.gate.packed,
                &t.gate.sfb,
                &t.gate.scale2,
                &t.up.packed,
                &t.up.sfb,
                &t.up.scale2,
                &t.gate_up_gs,
                a_gate.0,
                a_up.0,
                &off,
                mi as u32,
                H as u32,
                tokens,
                true,
                0,
            )
        };
        let down_call = || -> Result<bool> {
            cutlass::nvfp4_grouped_down_w4a4_ex(
                act.0,
                &t.down.packed,
                &t.down.sfb,
                &t.down.scale2,
                &t.down_gs,
                out.0,
                &off,
                H as u32,
                mi as u32,
                Some((slot.0, span.clone())),
                0,
            )
        };

        // 2026-10-07: One variant's gate/up outputs, and its down output over the shared
        // `a_act`; each output is sentinel-filled before its call.
        let run_gu = |tile: W4a4GemmTile| -> Result<(Vec<u8>, Vec<u8>, bool)> {
            cutlass::set_w4a4_gemm_tile_override(Some(tile));
            g.memset_async(a_gate, SENTINEL, te * mi * 2, 0)?;
            g.memset_async(a_up, SENTINEL, te * mi * 2, 0)?;
            let po = gate_up()?;
            g.synchronize(0)?;
            Ok((dn(g, a_gate, te * mi * 2)?, dn(g, a_up, te * mi * 2)?, po))
        };
        let run_dn = |tile: W4a4GemmTile| -> Result<(Vec<u8>, bool)> {
            cutlass::set_w4a4_gemm_tile_override(Some(tile));
            g.memset_async(out, SENTINEL, te * H * 2, 0)?;
            let pa = down_call()?;
            g.synchronize(0)?;
            Ok((dn(g, out, te * H * 2)?, pa))
        };

        // 2026-10-07: The default's gate/up, then the shared down input (`a_act` and its amax
        // slot, as METRALE_GLM_MOE_SWIGLU_AMAX computes them) from that output, then its down.
        let row = mi * 2;
        let default = run_gu(W4a4GemmTile::Default).and_then(|(g0, u0, po)| {
            g.memset_async(act, 0, te * mi * 2, 0)?;
            g.memset_async(slot, 0, 4, 0)?;
            swiglu_rows_amax(
                g,
                k_amax,
                a_gate.offset(span.start * row),
                a_up.offset(span.start * row),
                act.offset(span.start * row),
                rows * mi,
                LIMIT,
                slot,
                0,
            )?;
            let (o0, pa) = run_dn(W4a4GemmTile::Default)?;
            Ok((g0, u0, o0, po, pa))
        });
        let (g0, u0, o0, po, pa) = match default {
            Ok(v) => v,
            Err(e) => {
                println!("tokens={tokens}: default run error: {e:#}");
                problems.push(format!("tokens={tokens} default: run error {e:#}"));
                continue;
            }
        };
        let written = |b: &[u8]| b.iter().filter(|&&v| v != SENTINEL).count();
        let (wg, wo) = (written(&g0), written(&o0));
        println!(
            "tokens={tokens} default: gate {wg} / out {wo} bytes written (of {} / {}), \
             pack_once engaged={po} swiglu_amax engaged={pa}",
            g0.len(),
            o0.len()
        );
        if wg == 0 || wo == 0 {
            problems.push(format!(
                "tokens={tokens}: the default wrote no output (gate {wg}, out {wo})"
            ));
        }

        let mut timings: Vec<Timing> = Vec::new();
        for tile in W4a4GemmTile::ALL {
            let (_, fb0) = cutlass::w4a4_gemm_tile_stats();
            let res = run_gu(tile).and_then(|(gv, uv, _)| Ok((gv, uv, run_dn(tile)?.0)));
            let (last, fb1) = cutlass::w4a4_gemm_tile_stats();
            let (gv, uv, ov) = match res {
                Ok(v) => v,
                Err(e) => {
                    println!("tokens={tokens} variant={}: run error: {e:#}", tile.name());
                    problems.push(format!("tokens={tokens} {}: run error {e:#}", tile.name()));
                    continue;
                }
            };
            let (dg, du, dd) = (ndiff(&g0, &gv), ndiff(&u0, &uv), ndiff(&o0, &ov));
            println!(
                "TILE {} tokens={tokens}: {} differing bytes (gate {dg}, up {du} of {}; \
                 down {dd} of {}) launched={} fallbacks={}",
                tile.name(),
                dg + du + dd,
                g0.len(),
                o0.len(),
                last.map_or("none", W4a4GemmTile::name),
                fb1 - fb0
            );
            if dg + du + dd != 0 {
                problems.push(format!(
                    "tokens={tokens} {}: {} bytes differ from the default",
                    tile.name(),
                    dg + du + dd
                ));
            }
            if last != Some(tile) || fb1 != fb0 {
                problems.push(format!(
                    "tokens={tokens} {}: launched {:?} with {} fallback(s) to the default",
                    tile.name(),
                    last.map(W4a4GemmTile::name),
                    fb1 - fb0
                ));
            }
            // 2026-10-07: Whole calls, then the same calls with the GEMMs skipped (prep only).
            cutlass::set_w4a4_gemm_tile_override(Some(tile));
            let timed = (|| -> Result<Timing> {
                let t_gu = time_ms(g, iters, || gate_up().map(|_| ()))?;
                let t_dn = time_ms(g, iters, || down_call().map(|_| ()))?;
                cutlass::set_w4a4_gemm_skip(true);
                let p_gu = time_ms(g, iters, || gate_up().map(|_| ()));
                let p_dn = time_ms(g, iters, || down_call().map(|_| ()));
                cutlass::set_w4a4_gemm_skip(false);
                let (p_gu, p_dn) = (p_gu?, p_dn?);
                Ok(Timing {
                    tile,
                    gate_up: t_gu,
                    down: t_dn,
                    gemm: (t_gu - p_gu).max(0.0) + (t_dn - p_dn).max(0.0),
                })
            })();
            match timed {
                Ok(t) => timings.push(t),
                Err(e) => {
                    cutlass::set_w4a4_gemm_skip(false);
                    println!(
                        "tokens={tokens} variant={}: timing error: {e:#}",
                        tile.name()
                    );
                    problems.push(format!(
                        "tokens={tokens} {}: timing error {e:#}",
                        tile.name()
                    ));
                }
            }
        }
        cutlass::set_w4a4_gemm_tile_override(None);

        let base = timings
            .iter()
            .find(|t| t.tile == W4a4GemmTile::Default)
            .map(|t| (t.gate_up + t.down, t.gemm));
        if let Some((base_total, base_gemm)) = base {
            for t in &timings {
                let total = t.gate_up + t.down;
                println!(
                    "TILE_TIMING tokens={tokens} variant={} gateup_ms={:.3} down_ms={:.3} \
                     total_ms={total:.3} ratio_vs_default={:.3} gemm_ms={:.3} gemm_ratio={:.3} \
                     gemm_TFLOP/s={:.1}",
                    t.tile.name(),
                    t.gate_up,
                    t.down,
                    total / base_total.max(1e-9),
                    t.gemm,
                    t.gemm / base_gemm.max(1e-9),
                    flop / (t.gemm.max(1e-9) * 1e9)
                );
            }
            if let Some(best) = timings.iter().min_by(|a, b| a.gemm.total_cmp(&b.gemm)) {
                let total_ratio = (best.gate_up + best.down) / base_total.max(1e-9);
                let ratio = best.gemm / base_gemm.max(1e-9);
                println!(
                    "TILE_BEST tokens={tokens} variant={} total_ratio={total_ratio:.3} \
                     ratio={ratio:.3}",
                    best.tile.name()
                );
                summary.push(format!("{tokens}: best {} x{ratio:.3}", best.tile.name()));
            }
        }
        for p in [
            d_x, d_ids, d_stid, d_seid, d_t2p, d_off, a_gate, a_up, act, out,
        ] {
            g.free(p)?;
        }
    }
    cutlass::set_w4a4_gemm_tile_override(None);
    cutlass::set_w4a4_gemm_skip(false);
    if !problems.is_empty() {
        println!("FAIL: {}", problems.join("; "));
        bail!("glm_moe_w4a4_tile_microtest: {} problem(s)", problems.len());
    }
    println!(
        "PASS: every W4A4 grouped GEMM tile variant ran (no fallback) with gate/up and down \
         outputs bitwise equal to the default ({})",
        summary.join(", ")
    );
    Ok(())
}
