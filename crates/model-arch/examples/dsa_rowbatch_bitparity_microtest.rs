// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Byte-parity gate for the kernels `METRALE_GLM_DSA_ROW_BATCH=1` launches once per
//! prefill sub-chunk instead of once per row (`glm5next_dsa/layer/row_batch.rs`):
//!
//! * `dense_gemv_bf16_fp32out_batchm` against one `dense_gemv_bf16_fp32out` launch per row
//!   (`weights_proj`, `wq_b`), and `dense_gemv_bf16_batchm` against one `dense_gemv_bf16`
//!   launch per row (`wk`, `compress_gate`), at M in {1, 2, 5, 16}, over the GLM-5.3 indexer
//!   shapes and three others, two seeds;
//! * `glm5next_mla_latent_write_fp8` over `k` blocks with a `k`-entry slot array against `k`
//!   one-block launches, each reading a one-entry slot buffer rewritten per row;
//! * `nllb_layernorm_bf16` (`k_norm`) with `rows = k` against `k` launches with `rows = 1`.
//!
//! Every comparison is bitwise. Each arm's output starts filled with its own poison byte, so
//! an element one arm never writes cannot compare equal; the latent-write and `k_norm` legs
//! also check that the bytes outside the written rows still hold the arm's poison. A negative
//! control (one flipped input bit) must be detected, and a run that compared no element fails.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 when every leg is byte-identical, 1 on any difference, a vacuous comparison or a
//! negative control that does not fire, 2 when a kernel is absent from this target.
//!
//! Run (the latent write is a GLM-5.3-Flash target kernel):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_rowbatch_bitparity_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-10-01: (N, K, label). The first four are the GLM-5.3 indexer projections (hidden
/// 4096, q_lora_rank 1536, index_heads 32, index_head_dim 128). K stays a multiple of 8, the
/// kernels' alignment contract.
const SHAPES: &[(usize, usize, &str)] = &[
    (128, 4096, "indexer wk / compress_gate"),
    (32, 4096, "indexer weights_proj"),
    (4096, 1536, "indexer wq_b"),
    (512, 4096, "kv_a width"),
    (77, 520, "N % 4 != 0, short K"),
    (4, 64, "one block, one slab"),
];
const MS: &[usize] = &[1, 2, 5, 16];
const MAX_M: usize = 16;
const POISON_REF: u8 = 0xA5;
const POISON_NEW: u8 = 0x5A;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    fn bf16_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(self.r(lo, hi)).to_bits().to_le_bytes())
            .collect()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-01: Tally of one run: compared elements, differing legs, and the verdict.
#[derive(Default)]
struct Tally {
    compared: usize,
    failed: usize,
}

impl Tally {
    fn leg(&mut self, label: &str, elems: usize, a: &[u8], b: &[u8]) {
        let same = a == b;
        self.compared += elems;
        if !same {
            self.failed += 1;
        }
        let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
        println!("{label:<58} elems={elems:<8} byte-identical={same:<5} diff_bytes={diff}");
    }
}

struct Kernels {
    gemv: KernelHandle,
    gemv_f32: KernelHandle,
    batchm: KernelHandle,
    batchm_f32: KernelHandle,
    latent: KernelHandle,
    k_norm: KernelHandle,
}

/// 2026-10-01: One GEMV leg: M rows through the M = 1 kernel, one launch per row (the row
/// loop), against one batched launch, outputs pre-filled with different poison.
#[allow(clippy::too_many_arguments)]
fn gemv_leg(
    g: &dyn GpuBackend,
    ks: &Kernels,
    f32_out: bool,
    a: DevicePtr,
    w: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let elem = if f32_out { 4 } else { 2 };
    let bytes = m * n * elem;
    let (c_ref, c_new) = (g.alloc(bytes)?, g.alloc(bytes)?);
    g.memset(c_ref, POISON_REF, bytes)?;
    g.memset(c_new, POISON_NEW, bytes)?;
    let wt = DenseWeight { weight: w };
    let (single, batched) = if f32_out {
        (ks.gemv_f32, ks.batchm_f32)
    } else {
        (ks.gemv, ks.batchm)
    };
    for t in 0..m {
        let (a_t, c_t) = (a.offset(t * k * 2), c_ref.offset(t * n * elem));
        ops::dense_gemv(g, single, a_t, &wt, c_t, n as u32, k as u32, 0)?;
    }
    let (m32, n32, k32) = (m as u32, n as u32, k as u32);
    if f32_out {
        ops::dense_gemv_batchm_fp32out(g, batched, a, &wt, c_new, m32, n32, k32, n32, 0)?;
    } else {
        ops::dense_gemv_batchm(g, batched, a, &wt, c_new, m32, n32, k32, n32, 0)?;
    }
    let out = (down(g, c_ref, bytes)?, down(g, c_new, bytes)?);
    g.free(c_ref).ok();
    g.free(c_new).ok();
    Ok(out)
}

fn gemv_legs(g: &dyn GpuBackend, ks: &Kernels, tally: &mut Tally) -> Result<bool> {
    let mut control_ok = true;
    for seed in [7u64, 4242] {
        for &(n, k, label) in SHAPES {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (n * k) as u64);
            let a_bytes = rng.bf16_bytes(MAX_M * k, -1.5, 1.5);
            let a = up(g, &a_bytes)?;
            let w = up(g, &rng.bf16_bytes(n * k, -0.08, 0.08))?;
            for &m in MS {
                for f32_out in [true, false] {
                    let (r, b) = gemv_leg(g, ks, f32_out, a, w, m, n, k)?;
                    let kind = if f32_out { "f32" } else { "bf16" };
                    let name = format!("s{seed} {label} {kind} M={m}");
                    tally.leg(&name, m * n, &r, &b);
                }
            }
            // 2026-10-01: Negative control: one ULP flipped in row 1 of the batched arm's
            // input must change its FP32 output against the unflipped per-row reference.
            let mut pert = a_bytes.clone();
            pert[2 * (k + 3)] ^= 1;
            let a_pert = up(g, &pert)?;
            let (r, _) = gemv_leg(g, ks, true, a, w, 5, n, k)?;
            let (_, b) = gemv_leg(g, ks, true, a_pert, w, 5, n, k)?;
            let fired = r != b;
            control_ok &= fired;
            println!("s{seed} {label} CONTROL 1-ULP flip detected={fired}");
            for p in [a, w, a_pert] {
                g.free(p).ok();
            }
        }
    }
    Ok(control_ok)
}

/// 2026-10-01: `glm5next_mla_latent_write_fp8(kv_a, norm_w, cache, slot_mapping, dim, eps,
/// inv_scale)`, `blocks` blocks of `dim` threads, as `decode_k` launches it.
#[allow(clippy::too_many_arguments)]
fn latent(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    kv_a: DevicePtr,
    norm_w: DevicePtr,
    cache: DevicePtr,
    slots: DevicePtr,
    blocks: usize,
    dim: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([blocks as u32, 1, 1])
        .block([dim as u32, 1, 1])
        .arg_ptr(kv_a)
        .arg_ptr(norm_w)
        .arg_ptr(cache)
        .arg_ptr(slots)
        .arg_u32(dim as u32)
        .arg_f32(1e-6)
        .arg_f32(1.0)
        .launch(0)
}

fn latent_legs(g: &dyn GpuBackend, ks: &Kernels, tally: &mut Tally) -> Result<()> {
    const DIM: usize = 512;
    const SLOTS: usize = 64;
    for k in [2usize, 5, 16] {
        let mut rng = Lcg(0xC0FFEE ^ k as u64);
        let kv_a = up(g, &rng.bf16_bytes(k * DIM, -3.0, 3.0))?;
        let norm_w = up(g, &rng.bf16_bytes(DIM, 0.5, 1.5))?;
        // 2026-10-01: Distinct, non-contiguous slots, as a block table with gaps gives.
        let slots: Vec<i64> = (0..k).map(|r| ((r * 7 + 3) % SLOTS) as i64).collect();
        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let bytes = SLOTS * DIM;
        let (c_ref, c_new) = (g.alloc(bytes)?, g.alloc(bytes)?);
        g.memset(c_ref, POISON_REF, bytes)?;
        g.memset(c_new, POISON_NEW, bytes)?;
        // 2026-10-01: The row loop: rewrite the one-entry slot buffer, launch one block.
        let one = g.alloc(8)?;
        for (r, s) in slots.iter().enumerate() {
            // 2026-10-01: The previous launch reads `one`; let it finish first.
            g.synchronize(0)?;
            g.copy_h2d(&s.to_le_bytes(), one)?;
            let row = kv_a.offset(r * DIM * 2);
            latent(g, ks.latent, row, norm_w, c_ref, one, 1, DIM)?;
        }
        let all = up(g, &slot_bytes)?;
        latent(g, ks.latent, kv_a, norm_w, c_new, all, k, DIM)?;
        let (r, b) = (down(g, c_ref, bytes)?, down(g, c_new, bytes)?);
        let pick = |buf: &[u8]| -> Vec<u8> {
            let rows = slots.iter().map(|&s| s as usize);
            rows.flat_map(|s| buf[s * DIM..(s + 1) * DIM].to_vec()).collect()
        };
        let name = format!("latent_write k={k} written slots");
        tally.leg(&name, k * DIM, &pick(&r), &pick(&b));
        let untouched = |buf: &[u8], poison: u8| {
            (0..SLOTS)
                .filter(|s| !slots.contains(&(*s as i64)))
                .all(|s| buf[s * DIM..(s + 1) * DIM].iter().all(|&x| x == poison))
        };
        let clean = untouched(&r, POISON_REF) && untouched(&b, POISON_NEW);
        println!("latent_write k={k} other slots untouched={clean}");
        if !clean {
            tally.failed += 1;
        }
        for p in [kv_a, norm_w, c_ref, c_new, one, all] {
            g.free(p).ok();
        }
    }
    Ok(())
}

/// 2026-10-01: `nllb_layernorm_bf16(x, w, b, rows, dim, eps)` in place, `blocks` blocks of
/// `dim` threads and `dim * 4` B of shared memory, as `indexer_forward` launches it.
#[allow(clippy::too_many_arguments)]
fn k_norm(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    x: DevicePtr,
    w: DevicePtr,
    b: DevicePtr,
    blocks: usize,
    rows: usize,
    dim: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([blocks as u32, 1, 1])
        .block([dim as u32, 1, 1])
        .shared_mem((dim * 4) as u32)
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(b)
        .arg_u32(rows as u32)
        .arg_u32(dim as u32)
        .arg_f32(1e-6)
        .launch(0)
}

fn k_norm_legs(g: &dyn GpuBackend, ks: &Kernels, tally: &mut Tally) -> Result<()> {
    const DIM: usize = 128;
    for k in [2usize, 5, 16] {
        let mut rng = Lcg(0xBEEF ^ k as u64);
        let x = rng.bf16_bytes(k * DIM, -4.0, 4.0);
        let w = up(g, &rng.bf16_bytes(DIM, 0.5, 1.5))?;
        let b = up(g, &rng.bf16_bytes(DIM, -0.2, 0.2))?;
        // 2026-10-01: One guard row past the k rows, filled with the arm's poison.
        let bytes = (k + 1) * DIM * 2;
        let (x_ref, x_new) = (g.alloc(bytes)?, g.alloc(bytes)?);
        g.memset(x_ref, POISON_REF, bytes)?;
        g.memset(x_new, POISON_NEW, bytes)?;
        g.copy_h2d(&x, x_ref)?;
        g.copy_h2d(&x, x_new)?;
        for r in 0..k {
            k_norm(g, ks.k_norm, x_ref.offset(r * DIM * 2), w, b, 1, 1, DIM)?;
        }
        k_norm(g, ks.k_norm, x_new, w, b, k, k, DIM)?;
        let (r, n) = (down(g, x_ref, bytes)?, down(g, x_new, bytes)?);
        let rows = k * DIM * 2;
        let name = format!("k_norm k={k}");
        tally.leg(&name, k * DIM, &r[..rows], &n[..rows]);
        let clean = r[rows..].iter().all(|&v| v == POISON_REF)
            && n[rows..].iter().all(|&v| v == POISON_NEW);
        let moved = r[..rows] != x[..];
        println!("k_norm k={k} guard row untouched={clean} output differs from input={moved}");
        if !clean || !moved {
            tally.failed += 1;
        }
        for p in [w, b, x_ref, x_new] {
            g.free(p).ok();
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let lookups = [
        ("gemv", "dense_gemv_bf16"),
        ("gemv", "dense_gemv_bf16_fp32out"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        ("dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm"),
        ("glm5next_mla_latent_write", "glm5next_mla_latent_write_fp8"),
        ("nllb_encoder", "nllb_layernorm_bf16"),
    ];
    let mut handles = Vec::new();
    for (module, func) in lookups {
        match g.kernel(module, func) {
            Ok(h) => handles.push(h),
            Err(e) => {
                println!("{module}::{func} absent from this target ({e}) - SKIP");
                std::process::exit(2);
            }
        }
    }
    let ks = Kernels {
        gemv: handles[0],
        gemv_f32: handles[1],
        batchm: handles[2],
        batchm_f32: handles[3],
        latent: handles[4],
        k_norm: handles[5],
    };

    let mut tally = Tally::default();
    let control_ok = gemv_legs(g, &ks, &mut tally)?;
    latent_legs(g, &ks, &mut tally)?;
    k_norm_legs(g, &ks, &mut tally)?;

    if tally.compared == 0 {
        println!("FAIL - no element was compared; this run proves nothing.");
        std::process::exit(1);
    }
    if !control_ok {
        println!("FAIL - a negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if tally.failed > 0 {
        println!(
            "FAIL - {} leg(s) differ. METRALE_GLM_DSA_ROW_BATCH=1 is NOT bit-identical to the \
             per-row walk on this build; keep it off.",
            tally.failed
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {} elements byte-identical: batched fp32out/bf16 GEMVs, the k-block latent \
         write and the rows=k k_norm match their per-row launches.",
        tally.compared
    );
    Ok(())
}
