// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Byte-identity gate for the staged GLM prefill's FFN pass (whole-chunk prefill
//! L0, `METRALE_GLM_PREFILL_STAGED=1`).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits nonzero unless, for every case, the staged FFN windows write the same bytes as the
//!   unstaged per-sub-chunk calls, for the routed-MoE site and for a dense MLP site.
//!
//! A chunk of `T` random BF16 rows runs through one MLP site twice:
//! - REF, the unstaged loop: `forward_moe` / `forward_dense` once per `sub_chunks(T, W_attn)`
//!   sub-chunk;
//! - STAGED: `forward_moe_sliced` / `forward_dense_sliced` once per `ffn_windows` window of up
//!   to `W_ffn` rows, dense slices of `W_attn`, merging exactly as `Glm5NextLayer::ffn_mergeable`
//!   does (`grouped_prefill_selected`).
//!
//! A third arm, NAIVE, runs the same windows without dense slicing (router and shared expert at
//! M = window). It is reported, never gated: it shows whether cuBLASLt's per-M algorithm choice
//! moves bits, which is why STAGED slices.
//!
//! The geometry is the rank-0 slice of GLM-5.3-Flash at TP=2/EP=2 (hidden 4096, 288 experts of
//! which 144 local, top_k 8, moe_intermediate 2048, shared 1024, dense 6144). Expert weights are
//! random NVFP4, four distinct sets cycled over the local ids. The EP all-reduce is not run
//! here: it is an elementwise two-operand sum (`all_reduce_2rank`: send/recv then
//! `bf16_add_inplace`), so it cannot couple rows.
//!
//! Env: `GLM_STAGED_TOKENS` (default `5400,2072,1000`), `GLM_STAGED_ATTN` (default `256`),
//! `GLM_STAGED_FFN` (default `256,2048,4096`). The production levers apply as in a serve
//! (`METRALE_GLM_MOE_GEMM_TILE`, `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS`, ...).
//!
//!   METRALE_GLM_MOE_GEMM_TILE=bt_m128_k64 METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS=64 \
//!   cargo run -p metrale-model-arch --release --example glm5next_ffn_staged_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_layer::{ffn_windows, sub_chunks};
use metrale_model_arch::glm5next_mlp::build::{build_dense_mlp, build_moe};
use metrale_model_arch::glm5next_mlp::forward::{
    Glm5NextMlpWorkspace, forward_dense, forward_dense_sliced, forward_moe, forward_moe_sliced,
};
use metrale_model_arch::glm5next_mlp::forward_prefill_gemm::grouped_prefill_selected;
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

const FULL_DENSE: usize = 12288;
const FULL_SHARED: usize = 2048;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn rand_f32(s: &mut u64, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| ((lcg(s) % 2001) as f32 / 1000.0 - 1.0) * amp)
        .collect()
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
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

/// 2026-09-29: The rank-0 MLP geometry of GLM-5.3-Flash at TP=2, EP=2.
fn cfg() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: 4096,
        local_dense_intermediate: FULL_DENSE / 2,
        moe_intermediate: 2048,
        local_shared_intermediate: FULL_SHARED / 2,
        num_experts: 288,
        local_experts: 144,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 2,
        ep_world_size: 2,
    }
}

/// 2026-09-29: One random NVFP4 projection `[out, inn]`: packed codes, E4M3 block scales from
/// 0x28..0x38 (finite, about 2^-2..2^0), and a small global scale.
fn nvfp4(g: &dyn GpuBackend, s: &mut u64, out: usize, inn: usize) -> Result<Nvfp4Proj> {
    let packed: Vec<u8> = (0..out * inn / 2).map(|_| lcg(s) as u8).collect();
    let scale: Vec<u8> = (0..out * inn / 16)
        .map(|_| 0x28 + (lcg(s) % 16) as u8)
        .collect();
    Ok(Nvfp4Proj {
        packed: up(g, &packed)?,
        scale: up(g, &scale)?,
        scale_2: 0.02,
    })
}

/// 2026-09-29: First differing row of two `[rows, hidden]` BF16 outputs, and the count.
fn diff_rows(a: &[u8], b: &[u8], hidden: usize) -> (Option<usize>, usize) {
    let row = hidden * 2;
    let bad: Vec<usize> = (0..a.len() / row)
        .filter(|r| a[r * row..(r + 1) * row] != b[r * row..(r + 1) * row])
        .collect();
    (bad.first().copied(), bad.len())
}

fn main() -> Result<()> {
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let g = MetraleCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &g;
    let c = cfg();
    c.validate()?;
    let kern = Glm5NextMlpKernels::resolve(gpu)?;

    let tokens = env_list("GLM_STAGED_TOKENS", "5400,2072,1000,8191");
    let w_attn = env_list("GLM_STAGED_ATTN", "256")
        .first()
        .copied()
        .unwrap_or(256);
    let ffns = env_list("GLM_STAGED_FFN", "256,2048,4096");
    let max_rows = ffns.iter().copied().max().unwrap_or(w_attn).max(w_attn);
    let ws = Glm5NextMlpWorkspace::new_sized(gpu, &c, max_rows, w_attn)?;

    let mut s = 0x5EED_0029_u64;
    let h = c.hidden;
    // 2026-09-29: Host tensors by name for `build_moe` / `build_dense_mlp`.
    let mut host = std::collections::HashMap::<String, Vec<f32>>::new();
    host.insert(
        "mlp.gate.weight".into(),
        rand_f32(&mut s, c.num_experts * h, 0.05),
    );
    host.insert(
        "mlp.gate.e_score_correction_bias".into(),
        rand_f32(&mut s, c.num_experts, 0.01),
    );
    for (prefix, full) in [("mlp.shared_experts", FULL_SHARED), ("mlp", FULL_DENSE)] {
        for n in ["gate_proj", "up_proj", "down_proj"] {
            host.insert(
                format!("{prefix}.{n}.weight"),
                rand_f32(&mut s, full * h, 0.02),
            );
        }
    }
    let load = |n: &str| -> Result<Vec<f32>> {
        host.get(n)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no synthetic tensor {n}"))
    };
    let mi = c.moe_intermediate;
    let sets: Vec<Glm5NextExpertWeights> = (0..4)
        .map(|_| {
            Ok(Glm5NextExpertWeights {
                gate_proj: nvfp4(gpu, &mut s, mi, h)?,
                up_proj: nvfp4(gpu, &mut s, mi, h)?,
                down_proj: nvfp4(gpu, &mut s, h, mi)?,
            })
        })
        .collect::<Result<_>>()?;
    let expert = |id: usize| -> Result<Glm5NextExpertWeights> { Ok(sets[id % sets.len()]) };
    let moe = build_moe(gpu, &c, 0, FULL_SHARED, &load, &expert)?;
    let dense = build_dense_mlp(gpu, &c, 0, FULL_DENSE, "mlp", &load)?;

    let mut failures = 0usize;
    for &t_n in &tokens {
        let x_host: Vec<u8> = rand_f32(&mut s, t_n * h, 1.0)
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect();
        let x = up(gpu, &x_host)?;
        let bytes = t_n * h * 2;
        let subs = sub_chunks(t_n, w_attn);
        // 2026-10-01: Both window rules: the tail alone, and the tail merged into the window
        // before it (`METRALE_GLM_PREFILL_TAIL_MERGE`).
        for (merge_tail, &w_ffn) in [false, true]
            .into_iter()
            .flat_map(|m| ffns.iter().map(move |f| (m, f)))
        {
            let win = ffn_windows(&subs, w_ffn, merge_tail, |k| {
                grouped_prefill_selected(&kern, &c, &ws, k)
            });
            let dense_win = ffn_windows(&subs, w_ffn, merge_tail, |_| true);
            let (r, st, nv) = (g.alloc(bytes)?, g.alloc(bytes)?, g.alloc(bytes)?);
            let off = |p: DevicePtr, t: usize| p.offset(t * h * 2);
            for (site, windows) in [("moe", &win), ("dense", &dense_win)] {
                for &(t, k) in &subs {
                    match site {
                        "moe" => {
                            forward_moe(gpu, &kern, &c, &moe, off(x, t), off(r, t), k, &ws, 0)?
                        }
                        _ => forward_dense(
                            gpu,
                            &kern,
                            &c,
                            &dense,
                            c.local_dense_intermediate,
                            off(x, t),
                            off(r, t),
                            k,
                            &ws,
                            0,
                        )?,
                    }
                }
                for &(t, k) in windows.iter() {
                    for (out, slice) in [(st, w_attn), (nv, k)] {
                        match site {
                            "moe" => forward_moe_sliced(
                                gpu,
                                &kern,
                                &c,
                                &moe,
                                off(x, t),
                                off(out, t),
                                k,
                                slice,
                                &ws,
                                0,
                            )?,
                            _ => forward_dense_sliced(
                                gpu,
                                &kern,
                                &c,
                                &dense,
                                c.local_dense_intermediate,
                                off(x, t),
                                off(out, t),
                                k,
                                slice,
                                &ws,
                                0,
                            )?,
                        }
                    }
                }
                let (rb, sb, nb) = (dn(gpu, r, bytes)?, dn(gpu, st, bytes)?, dn(gpu, nv, bytes)?);
                let nonzero = rb.chunks_exact(2).filter(|v| v != &[0, 0]).count();
                let (first, n_bad) = diff_rows(&rb, &sb, h);
                let (_, n_naive) = diff_rows(&rb, &nb, h);
                let verdict = if n_bad == 0 { "IDENTICAL" } else { "DIFFERS" };
                println!(
                    "{site:5} T={t_n:5} W_attn={w_attn} W_ffn={w_ffn:4} merge_tail={merge_tail} windows={:?} \
                     staged={verdict} ({n_bad} rows, first {first:?}) naive_unsliced={n_naive} \
                     rows differ; ref nonzero={nonzero}/{}",
                    windows.iter().map(|w| w.1).collect::<Vec<_>>(),
                    rb.len() / 2
                );
                if n_bad != 0 || nonzero == 0 {
                    failures += 1;
                }
            }
            for p in [r, st, nv] {
                gpu.free(p)?;
            }
        }
        gpu.free(x)?;
    }
    if failures != 0 {
        bail!("{failures} staged FFN case(s) not byte-identical to the unstaged loop");
    }
    println!("PASS: staged FFN windows byte-identical to the unstaged sub-chunk loop");
    Ok(())
}
