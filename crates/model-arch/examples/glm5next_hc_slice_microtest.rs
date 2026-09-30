// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Byte-identity gate and timing for the sliced GLM mHC pre (`glm_hc_pre` launching
//! `hc_mix` + `hc_finish` per `MHC_SLICE_ROWS`-row slice) against the unsliced launch pair it
//! replaced.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - Exits nonzero unless, for every token count and both `hc_fn` widths (BF16, the production
//!   path through `glm5next_hc_mix_bf16`, and FP32 through `glm5next_hc_mix`), `y`, `post`,
//!   `comb` and the `mix` scratch rows are byte-identical between OLD and NEW, and between OLD
//!   and every swept slice width; and unless OLD wrote every output.
//!
//! Arms, on the same device inputs:
//! - OLD: the launcher as of seed/port4 8e4cf68, transcribed: `hc_mix` on grid `(T, mix_hc)`,
//!   then `hc_finish` on grid `(T, 1 + ceil(H / 256))`, one launch each.
//! - NEW: `glm_hc_pre`, i.e. `glm_hc_pre_sliced` at `MHC_SLICE_ROWS`.
//! - SWEEP: `glm_hc_pre_sliced` at each `GLM_HC_SLICE_SWEEP` width (gated for bytes too).
//!
//! Every output and each arm's own `mix` scratch are filled with 0xAB before its launch, so an
//! unwritten value cannot pass as a match.
//!
//! Timing is wall clock over `GLM_HC_SLICE_REPS` launches, synchronised once at each end:
//! - `hot`: back to back, so a 256-row working set (16.8 MB) stays in GB10's 24 MB L2 across
//!   reps; an upper bound on what a serve sees.
//! - `cold`: a 64 MiB `memset_async` before each launch evicts L2, and the mean time of the
//!   memset alone is subtracted. Closer to a serve, where the MLP runs between two `hc_pre`s.
//!
//! Also timed alone: the mix launches (OLD one grid, NEW the slices), comparable to the nsys
//! `glm5next_hc_mix_bf16` per-call numbers (0.495 ms at 256 rows, about 13.5 ms at 2048 rows,
//! pp8192, 2026-09-29).
//!
//! Geometry: GLM-5.3-Flash's hidden 4096, `hc_mult` 4 (`mix_hc` 24), 20 Sinkhorn iterations,
//! `hc_eps` 1e-6. `glm_hc_pre` refuses more tokens than `mhc_mix_max_tokens()`, which the
//! prefill levers size, so the run sets them as the staged recipe does:
//!
//!   METRALE_GLM_PREFILL_STAGED=1 METRALE_GLM_PREFILL_ROWS=256 METRALE_GLM_PREFILL_ROWS_FFN=2048 \
//!   cargo run -p metrale-model-arch --release --example glm5next_hc_slice_microtest \
//!       --features cuda,gpu-examples
//!
//! Env: `GLM_HC_SLICE_TOKENS` (default `256,2048,1000,1`), `GLM_HC_SLICE_SWEEP` (default
//! `64,128,512,1024`), `GLM_HC_SLICE_REPS` (default `20`).

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, MHC_SLICE_ROWS, glm_hc_pre, glm_hc_pre_sliced,
    mhc_mix_max_tokens, mhc_slices,
};

const HIDDEN: usize = 4096;
const HC: usize = 4;
const SINKHORN_ITERS: u32 = 20;
const HC_EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;
const FLUSH_BYTES: usize = 64 << 20;

fn mix_hc(hc: usize) -> usize {
    (2 + hc) * hc
}

/// 2026-09-29: The launcher's `collapse_blocks`, transcribed for the OLD arm.
fn collapse_blocks(h: usize) -> u32 {
    if h < 256 { 1 } else { h.div_ceil(256) as u32 }
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn f32_bytes(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn env_list(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect()
}

/// 2026-09-29: One arm's outputs and its own `mix` scratch.
struct Outs {
    y: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    mix: DevicePtr,
}

impl Outs {
    fn new(g: &dyn GpuBackend, t: usize) -> Result<Self> {
        Ok(Self {
            y: g.alloc(t * HIDDEN * 2)?,
            post: g.alloc(t * HC * 4)?,
            comb: g.alloc(t * HC * HC * 4)?,
            mix: g.alloc(mhc_mix_max_tokens() * mix_hc(HC) * 4)?,
        })
    }

    fn sizes(t: usize) -> [(&'static str, usize); 4] {
        [
            ("y", t * HIDDEN * 2),
            ("post", t * HC * 4),
            ("comb", t * HC * HC * 4),
            ("mix", t * mix_hc(HC) * 4),
        ]
    }

    fn ptrs(&self) -> [DevicePtr; 4] {
        [self.y, self.post, self.comb, self.mix]
    }

    fn poison(&self, g: &dyn GpuBackend, t: usize) -> Result<()> {
        for (p, (_, n)) in self.ptrs().into_iter().zip(Self::sizes(t)) {
            g.copy_h2d(&vec![0xABu8; n], p)?;
        }
        Ok(())
    }

    fn read(&self, g: &dyn GpuBackend, t: usize) -> Result<Vec<Vec<u8>>> {
        self.ptrs()
            .into_iter()
            .zip(Self::sizes(t))
            .map(|(p, (_, n))| dn(g, p, n))
            .collect()
    }

    fn free(self, g: &dyn GpuBackend) -> Result<()> {
        for p in self.ptrs() {
            g.free(p)?;
        }
        Ok(())
    }
}

/// 2026-09-29: The OLD launcher: one `hc_mix` on grid `(T, mix_hc)`, one `hc_finish` on grid
/// `(T, 1 + ceil(H / 256))`, exactly as `glm_hc_pre` launched them before slicing.
fn old_pre(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    o: &Outs,
    t: u32,
) -> Result<()> {
    old_mix(g, k, w, streams, o.mix, t)?;
    KernelLaunch::new(g, k.hc_finish)
        .grid([t, 1 + collapse_blocks(HIDDEN), 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(o.mix)
        .arg_ptr(w.hc_scale)
        .arg_ptr(w.hc_base)
        .arg_ptr(o.y)
        .arg_ptr(o.post)
        .arg_ptr(o.comb)
        .arg_u32(HIDDEN as u32)
        .arg_u32(HC as u32)
        .arg_u32(SINKHORN_ITERS)
        .arg_f32(HC_EPS)
        .launch(0)
}

/// 2026-09-29: The mix kernel `glm_hc_pre` picks for `w`; errors instead of falling back when the
/// BF16 kernel is missing, so the BF16 arm cannot silently run the FP32 kernel.
fn mix_kernel(k: &Glm5NextMhcKernels, w: &Glm5NextMhcSiteWeights) -> Result<KernelHandle> {
    if w.hc_fn_bf16 {
        if k.hc_mix_bf16.0 == 0 {
            bail!("glm5next_hc_mix_bf16 is not linked; the BF16 arm would silently test FP32");
        }
        Ok(k.hc_mix_bf16)
    } else {
        Ok(k.hc_mix)
    }
}

/// 2026-09-29: The OLD mix launch alone, grid `(T, mix_hc)`.
fn old_mix(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    mix: DevicePtr,
    t: u32,
) -> Result<()> {
    let kern = mix_kernel(k, w)?;
    KernelLaunch::new(g, kern)
        .grid([t, mix_hc(HC) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(w.hc_fn)
        .arg_ptr(mix)
        .arg_u32(HIDDEN as u32)
        .arg_u32(HC as u32)
        .arg_f32(NORM_EPS)
        .launch(0)
}

/// 2026-09-29: The NEW mix launches alone: `mhc_slices` at `MHC_SLICE_ROWS`, pointers advanced
/// per slice as `glm_hc_pre_sliced` advances them.
fn new_mix(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    mix: DevicePtr,
    t: u32,
) -> Result<()> {
    let kern = mix_kernel(k, w)?;
    for (t0, rows) in mhc_slices(t, MHC_SLICE_ROWS) {
        let t0 = t0 as usize;
        KernelLaunch::new(g, kern)
            .grid([rows, mix_hc(HC) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(streams.offset(t0 * HC * HIDDEN * 4))
            .arg_ptr(w.hc_fn)
            .arg_ptr(mix.offset(t0 * mix_hc(HC) * 4))
            .arg_u32(HIDDEN as u32)
            .arg_u32(HC as u32)
            .arg_f32(NORM_EPS)
            .launch(0)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn new_pre(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    o: &Outs,
    t: u32,
    slice: Option<u32>,
) -> Result<()> {
    let wm = Glm5NextMhcSiteWeights { mix: o.mix, ..*w };
    match slice {
        None => glm_hc_pre(
            g,
            k,
            streams,
            &wm,
            o.y,
            o.post,
            o.comb,
            t,
            HIDDEN as u32,
            HC as u32,
            SINKHORN_ITERS,
            NORM_EPS,
            HC_EPS,
            0,
        ),
        Some(s) => glm_hc_pre_sliced(
            g,
            k,
            streams,
            &wm,
            o.y,
            o.post,
            o.comb,
            t,
            HIDDEN as u32,
            HC as u32,
            SINKHORN_ITERS,
            NORM_EPS,
            HC_EPS,
            s,
            0,
        ),
    }
}

/// 2026-09-29: Mean wall milliseconds per call of `f` after 3 warm-up calls. With `flush`, each
/// call is preceded by a `memset_async` of the flush buffer, and the mean time of the memset
/// alone is subtracted.
fn time_ms(
    g: &dyn GpuBackend,
    reps: usize,
    flush: Option<DevicePtr>,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    let mut run = |with_f: bool, n: usize| -> Result<f64> {
        g.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for i in 0..n {
            if let Some(p) = flush {
                g.memset_async(p, (i & 0xff) as u8, FLUSH_BYTES, 0)?;
            }
            if with_f {
                f()?;
            }
        }
        g.synchronize(0)?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / n as f64)
    };
    run(true, 3)?;
    let total = run(true, reps)?;
    if flush.is_none() {
        return Ok(total);
    }
    let flush_only = run(false, reps)?;
    Ok(total - flush_only)
}

fn main() -> Result<()> {
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let gg = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gg;
    let k = Glm5NextMhcKernels::resolve(g)?;

    let tokens = env_list("GLM_HC_SLICE_TOKENS", "256,2048,1000,1");
    let sweep = env_list("GLM_HC_SLICE_SWEEP", "64,128,512,1024");
    let reps = env_list("GLM_HC_SLICE_REPS", "20")
        .first()
        .copied()
        .unwrap_or(20)
        .max(1);
    let t_max = tokens.iter().copied().max().unwrap_or(0);
    if t_max > mhc_mix_max_tokens() {
        bail!(
            "T={t_max} exceeds mhc_mix_max_tokens() = {}: run with METRALE_GLM_PREFILL_STAGED=1 \
             METRALE_GLM_PREFILL_ROWS=256 METRALE_GLM_PREFILL_ROWS_FFN={} (see the header)",
            mhc_mix_max_tokens(),
            t_max.next_multiple_of(256)
        );
    }
    println!(
        "glm5next hc slice microtest: OLD = one (T, mix_hc) mix + one finish; NEW = glm_hc_pre \
         at MHC_SLICE_ROWS={MHC_SLICE_ROWS}; H={HIDDEN} hc={HC} reps={reps}\n"
    );

    let m = mix_hc(HC);
    let hc_dim = HC * HIDDEN;
    let flush = g.alloc(FLUSH_BYTES)?;
    let mut failures = 0usize;
    // 2026-09-29: The T=256 cold times per `hc_fn` width (index 0 BF16, 1 FP32), the base of
    // the linear-from-256 line printed for wider T.
    let mut cold_new_256: [Option<f64>; 2] = [None, None];
    let mut cold_old_256: [Option<f64>; 2] = [None, None];

    for &t in &tokens {
        let mut seed = 0x5eed_0029_u64 ^ (t as u64) << 16;
        let streams: Vec<f32> = (0..t * hc_dim).map(|_| lcg(&mut seed)).collect();
        // 2026-09-29: `hc_fn` scaled down only to keep the mixes out of saturation, as in
        // glm5next_hc_split_gate; the byte comparison holds at any scale.
        let hc_fn: Vec<f32> = (0..m * hc_dim).map(|_| lcg(&mut seed) * 0.02).collect();
        let hc_base: Vec<f32> = (0..m).map(|_| lcg(&mut seed)).collect();
        let d_streams = up(g, &f32_bytes(&streams))?;
        let d_scale = up(g, &f32_bytes(&[0.7, 1.3, 0.9]))?;
        let d_base = up(g, &f32_bytes(&hc_base))?;
        let d_fn_f32 = up(g, &f32_bytes(&hc_fn))?;
        let fn_bf16: Vec<u8> = hc_fn
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
            .collect();
        let d_fn_bf16 = up(g, &fn_bf16)?;

        for (wi, bf) in [(0usize, true), (1usize, false)] {
            let width = if bf { "bf16" } else { "f32 " };
            let w = Glm5NextMhcSiteWeights {
                hc_fn: if bf { d_fn_bf16 } else { d_fn_f32 },
                hc_fn_bf16: bf,
                hc_scale: d_scale,
                hc_base: d_base,
                mix: DevicePtr::NULL,
            };
            let tt = t as u32;

            let old = Outs::new(g, t)?;
            old.poison(g, t)?;
            old_pre(g, &k, &w, d_streams, &old, tt)?;
            let ref_b = old.read(g, t)?;
            for (b, (name, n)) in ref_b.iter().zip(Outs::sizes(t)) {
                if n > 0 && b.iter().all(|&x| x == 0xAB) {
                    bail!("T={t} {width}: OLD never wrote `{name}`");
                }
            }

            let mut arms: Vec<(String, Option<u32>)> = vec![("NEW".into(), None)];
            arms.extend(sweep.iter().map(|&s| (format!("slice{s}"), Some(s as u32))));
            let new = Outs::new(g, t)?;
            for (label, slice) in &arms {
                new.poison(g, t)?;
                new_pre(g, &k, &w, d_streams, &new, tt, *slice)?;
                let got = new.read(g, t)?;
                let mut bad = Vec::new();
                for ((a, b), (name, _)) in ref_b.iter().zip(&got).zip(Outs::sizes(t)) {
                    if a != b {
                        let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
                        bad.push(format!("{name}@byte{i}"));
                    }
                }
                if bad.is_empty() {
                    println!("  T={t:5} {width} {label:10} y/post/comb/mix BYTE-IDENTICAL to OLD");
                } else {
                    println!("  T={t:5} {width} {label:10} DIFFERS: {}", bad.join(", "));
                    failures += 1;
                }
            }

            let pre_old = |fl| time_ms(g, reps, fl, || old_pre(g, &k, &w, d_streams, &old, tt));
            let pre_new = |fl| {
                time_ms(g, reps, fl, || {
                    new_pre(g, &k, &w, d_streams, &new, tt, None)
                })
            };
            let (old_hot, new_hot) = (pre_old(None)?, pre_new(None)?);
            let (old_cold, new_cold) = (pre_old(Some(flush))?, pre_new(Some(flush))?);
            let mix_old_cold = time_ms(g, reps, Some(flush), || {
                old_mix(g, &k, &w, d_streams, old.mix, tt)
            })?;
            let mix_new_cold = time_ms(g, reps, Some(flush), || {
                new_mix(g, &k, &w, d_streams, new.mix, tt)
            })?;
            println!(
                "  T={t:5} {width} ms/call  mix+finish hot OLD {old_hot:8.3} NEW {new_hot:8.3} | \
                 cold OLD {old_cold:8.3} NEW {new_cold:8.3} | mix-only cold OLD \
                 {mix_old_cold:8.3} NEW {mix_new_cold:8.3}"
            );
            for (label, slice) in arms.iter().skip(1) {
                let s = *slice;
                let c = time_ms(g, reps, Some(flush), || {
                    new_pre(g, &k, &w, d_streams, &new, tt, s)
                })?;
                println!("  T={t:5} {width} ms/call  mix+finish cold {label:10} {c:8.3}");
            }
            if t == 256 {
                cold_new_256[wi] = Some(new_cold);
                cold_old_256[wi] = Some(old_cold);
            } else if t > 256
                && let (Some(n256), Some(o256)) = (cold_new_256[wi], cold_old_256[wi])
            {
                let lin = o256 * t as f64 / 256.0;
                println!(
                    "  T={t:5} {width} linear-from-256 {lin:8.3} ms: OLD {:.2}x, NEW {:.2}x \
                     (NEW@256 {n256:.3})",
                    old_cold / lin,
                    new_cold / lin
                );
            }
            old.free(g)?;
            new.free(g)?;
        }
        for p in [d_streams, d_scale, d_base, d_fn_f32, d_fn_bf16] {
            g.free(p)?;
        }
        println!();
    }
    g.free(flush)?;
    if failures != 0 {
        bail!("{failures} sliced arm(s) not byte-identical to the unsliced launch pair");
    }
    println!("PASS: sliced glm_hc_pre byte-identical to the unsliced launch pair");
    Ok(())
}
