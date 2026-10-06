// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The fused dense-FFN gate+up decode GEMM (one cuBLASLt W8A8 call
//! over `[gate | up]`, N = 2 x 17408) against the two separate GEMMs, at
//! Qwen3.8-27B shapes, plus its `silu_mul_strided` consumer against
//! `moe_silu_mul`.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, at every M in `ROWS`, the fused output's gate
//!   and up column halves equal the two GEMMs' outputs byte for byte over all
//!   `MAX_M` padded rows and are finite, and `silu_mul_strided` over the fused
//!   rows equals `moe_silu_mul` over the pair for rows 0..M. A changed guard
//!   byte on any output fails the run at read-back.
//!
//! Splitting N gives independent output columns over the same K with the same
//! block scales, so the gate is byte equality, not a tolerance. Four KNOWN_BAD
//! controls (one byte, one row, wrong half, nonfinite) must each be refused.
//!
//! cuBLASLt is given `ceil16(M)` rows (`ops::cublas_fp8_m_pad`), which is
//! `MAX_M` for every M in `ROWS`, and writes rows M..16; both arms are
//! compared over that full extent.
//!
//! Each M is also timed over `REPS` after one warmup run, as µs per gate+up
//! unit and weight bytes per second, for both arms.
//!
//! Run: `METRALE_TARGET_HW=hopper cargo run --release -p metrale-model-arch
//! --features cuda,gpu-examples --example native_fp8_ffn_gateup_fused_microtest`.
//! The target matters: `silu_mul_strided.cu` exists only in `kernels/hopper`,
//! so another target's image fails the kernel lookup. cuBLASLt is called
//! directly, so the serve's route selection (`METRALE_CUBLAS_GEMM`,
//! `ffn_gateup_fused`) is not consulted; the serve spelling is printed at the
//! end.

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};
use std::time::Instant;

const H: usize = 5120;
const INTER: usize = 17408;
const MAX_M: usize = 16;
const ROWS: [usize; 3] = [5, 8, 16];
const REPS: usize = 20;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const BF16: usize = 2;

struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// 2026-09-25: FP8 E4M3 bytes with a random sign and a magnitude code of
    /// 0x00..=0x7E, so never the NaN code 0x7F.
    fn fp8(&mut self, elems: usize) -> Vec<u8> {
        (0..elems)
            .map(|_| {
                let x = self.bits();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect()
    }
    /// 2026-09-25: One FP32 scale per 128x128 weight block.
    fn scales(&mut self, blocks: usize) -> Vec<u8> {
        (0..blocks)
            .flat_map(|_| (((self.bits() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect()
    }
    fn acts(&mut self, elems: usize) -> Vec<u8> {
        (0..elems)
            .flat_map(|_| {
                bf16::from_f32(((self.bits() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// 2026-09-25: A guarded device buffer: `GUARD` sentinel bytes, `len`
/// payload, `GUARD` sentinel bytes. `ptr` is the payload; `base` is what `arm`
/// re-stamps.
struct Guarded {
    base: DevicePtr,
    ptr: DevicePtr,
    sentinel: Vec<u8>,
    len: usize,
}

impl Guarded {
    fn new(gpu: &dyn GpuBackend, len: usize) -> Result<Self> {
        let sentinel = vec![SENTINEL; len + 2 * GUARD];
        let base = upload(gpu, &sentinel)?;
        Ok(Self {
            base,
            ptr: base.offset(GUARD),
            sentinel,
            len,
        })
    }
    fn arm(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.sentinel, self.base)
    }
    /// 2026-09-25: Payload bytes, after checking both guard bands survived.
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut host = vec![0_u8; self.sentinel.len()];
        gpu.copy_d2h(self.base, &mut host)?;
        ensure!(
            host[..GUARD].iter().all(|&b| b == SENTINEL)
                && host[GUARD + self.len..].iter().all(|&b| b == SENTINEL),
            "a launch wrote outside its output extent (guard band clobbered)"
        );
        Ok(host[GUARD..GUARD + self.len].to_vec())
    }
}

/// 2026-09-25: The oracle: `observed` must equal `reference` byte for byte
/// over `spans` (byte offset, byte length) of `observed`, read against
/// `ref_spans` of `reference`. Also refuses any non-finite value inside the
/// `observed` spans.
fn equal_bytes(
    observed: &[u8],
    reference: &[u8],
    spans: &[(usize, usize)],
    ref_spans: &[(usize, usize)],
) -> Result<()> {
    ensure!(spans.len() == ref_spans.len(), "span count mismatch");
    for (&(o, len), &(r, rlen)) in spans.iter().zip(ref_spans) {
        ensure!(len == rlen, "span length mismatch");
        let a = &observed[o..o + len];
        let b = &reference[r..r + rlen];
        ensure!(
            a.chunks_exact(2)
                .all(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).is_finite()),
            "nonfinite value in the fused output"
        );
        if a != b {
            let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
            anyhow::bail!("{bad} of {len} bytes differ (byte equality is the gate)");
        }
    }
    Ok(())
}

/// 2026-09-25: `out[m, n] = a_fp8[m, k] @ w[n, k]ᵀ` through
/// `ops::cublas_fp8_proj_prequant`, the call `DenseFfnLayer::w8a8_gemm` makes on
/// its cuBLASLt arm.
#[allow(clippy::too_many_arguments)]
fn gemm(
    gpu: &dyn GpuBackend,
    kmajor_k: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    a_scale_kmajor: DevicePtr,
    w: &Fp8Weight,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    ops::cublas_fp8_proj_prequant(
        gpu,
        kmajor_k,
        a_fp8,
        a_scale,
        a_scale_kmajor,
        w,
        out,
        m as u32,
        w.n,
        w.k,
        0,
    )
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    // 2026-10-06: `per_token_group_quant_fp8` takes an `Fp8ActQuant` now. Only the shared kernel,
    // as before (no Hopper twin), so this example's numbers are unchanged.
    let quant = ops::Fp8ActQuant {
        shared: gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
        ..Default::default()
    };
    let kmajor = gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?;
    let silu = gpu.kernel("moe_silu_mul", "moe_silu_mul")?;
    let silu_strided = gpu.kernel("silu_mul_strided", "silu_mul_strided")?;
    println!(
        "gate: BYTE EQUALITY (not a tolerance). Splitting N gives independent \
         output columns over the same K with the same block scales, so the \
         fused halves must reproduce the pair exactly."
    );
    println!("shapes: K={H} N={INTER} per half, fused N={}", 2 * INTER);

    let mut rng = Rng(0x0927_2026_F05E_D001);

    // 2026-09-25: One weight allocation holding `[gate | up]` along N, with
    // the two halves as views inside it, as `weight_loader/qwen35_dense.rs`
    // lays out a fused layer, so any difference the comparison sees is the
    // GEMM's and not the operands'.
    let w_bytes = rng.fp8(2 * INTER * H);
    let s_blocks = (2 * INTER / 128) * (H / 128);
    let s_bytes = rng.scales(s_blocks);
    let fused_w = upload(&gpu, &w_bytes)?;
    let fused_s = upload(&gpu, &s_bytes)?;
    let half_scale_bytes = (INTER / 128) * (H / 128) * 4;
    let view = |w_off: usize, s_off: usize, n: usize| Fp8Weight {
        weight: fused_w.offset(w_off),
        row_scale: fused_s.offset(s_off),
        n: n as u32,
        k: H as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    let gate_w = view(0, 0, INTER);
    let up_w = view(INTER * H, half_scale_bytes, INTER);
    let fused = view(0, 0, 2 * INTER);

    let act = upload(&gpu, &rng.acts(MAX_M * H))?;

    // 2026-09-25: Activation-quant scratch for `MAX_M` rows.
    let kg = H / 128;
    let a_fp8 = gpu.alloc(MAX_M * H)?;
    let a_scale = gpu.alloc(MAX_M * kg * 4)?;
    let a_kmajor = gpu.alloc(MAX_M * kg * 4)?;

    let gate_out = Guarded::new(&gpu, MAX_M * INTER * BF16)?;
    let up_out = Guarded::new(&gpu, MAX_M * INTER * BF16)?;
    let fused_out = Guarded::new(&gpu, MAX_M * 2 * INTER * BF16)?;
    let silu_ref = Guarded::new(&gpu, MAX_M * INTER * BF16)?;
    let silu_obs = Guarded::new(&gpu, MAX_M * INTER * BF16)?;

    // 2026-09-25: The whole padded extent, per half. cuBLASLt writes rows
    // `m..16`, so every row of the buffer is compared, pad rows included.
    let half_spans = |row_stride: usize, col_off: usize| -> Vec<(usize, usize)> {
        (0..MAX_M)
            .map(|r| ((r * row_stride + col_off) * BF16, INTER * BF16))
            .collect()
    };
    let contiguous = half_spans(INTER, 0);
    let fused_gate = half_spans(2 * INTER, 0);
    let fused_up = half_spans(2 * INTER, INTER);

    let mut failures = 0usize;
    let mut controls_done = false;

    for m in ROWS {
        println!("\n=== M = {m} (cuBLASLt pad = {MAX_M}) ===");
        // 2026-09-25: Quantize once: gate and up share the input, which is what
        // `cublas_fp8_proj_prequant` exists for.
        ops::per_token_group_quant_fp8(&gpu, quant, act, a_fp8, a_scale, m as u32, H as u32, 0)?;

        // 2026-09-25: The pair.
        gate_out.arm(&gpu)?;
        up_out.arm(&gpu)?;
        gemm(
            &gpu,
            kmajor,
            a_fp8,
            a_scale,
            a_kmajor,
            &gate_w,
            gate_out.ptr,
            m,
        )?;
        gemm(&gpu, kmajor, a_fp8, a_scale, a_kmajor, &up_w, up_out.ptr, m)?;
        gpu.synchronize(0)?;
        let gate_host = gate_out.read(&gpu)?;
        let up_host = up_out.read(&gpu)?;

        // 2026-09-25: The fused arm.
        fused_out.arm(&gpu)?;
        gemm(
            &gpu,
            kmajor,
            a_fp8,
            a_scale,
            a_kmajor,
            &fused,
            fused_out.ptr,
            m,
        )?;
        gpu.synchronize(0)?;
        let fused_host = fused_out.read(&gpu)?;

        for (name, spans, reference) in [
            ("gate half", &fused_gate, &gate_host),
            ("up half", &fused_up, &up_host),
        ] {
            match equal_bytes(&fused_host, reference, spans, &contiguous) {
                Ok(()) => println!("  {name}: BYTE-IDENTICAL over {MAX_M} padded rows"),
                Err(e) => {
                    failures += 1;
                    println!("  {name}: FAIL {e}");
                }
            }
        }

        // 2026-09-25: The SiLU consumer.
        silu_ref.arm(&gpu)?;
        silu_obs.arm(&gpu)?;
        ops::silu_mul(
            &gpu,
            silu,
            gate_out.ptr,
            up_out.ptr,
            silu_ref.ptr,
            (m * INTER) as u32,
            0,
        )?;
        ops::silu_mul_strided(
            &gpu,
            silu_strided,
            fused_out.ptr,
            fused_out.ptr.offset(INTER * BF16),
            silu_obs.ptr,
            m as u32,
            INTER as u32,
            (2 * INTER) as u32,
            INTER as u32,
            0,
        )?;
        gpu.synchronize(0)?;
        let (r, o) = (silu_ref.read(&gpu)?, silu_obs.read(&gpu)?);
        let live: Vec<(usize, usize)> = (0..m)
            .map(|row| (row * INTER * BF16, INTER * BF16))
            .collect();
        match equal_bytes(&o, &r, &live, &live) {
            Ok(()) => println!("  silu_mul_strided: BYTE-IDENTICAL to moe_silu_mul over {m} rows"),
            Err(e) => {
                failures += 1;
                println!("  silu_mul_strided: FAIL {e}");
            }
        }

        if !controls_done {
            // 2026-09-25: A green run has to be able to go red.
            for control in ["one byte", "one row", "wrong half", "nonfinite"] {
                let mut bad = fused_host.clone();
                match control {
                    "one byte" => bad[fused_gate[0].0 + 2] ^= 1,
                    "one row" => {
                        let (o, len) = fused_gate[MAX_M - 1];
                        bad[o..o + len].fill(0);
                    }
                    // 2026-09-25: A layout error: the up half where the
                    // gate half belongs.
                    "wrong half" => {
                        let (g, len) = fused_gate[0];
                        let (u, _) = fused_up[0];
                        bad.copy_within(u..u + len, g);
                    }
                    _ => bad[fused_gate[0].0..fused_gate[0].0 + 2]
                        .copy_from_slice(&0x7fc0_u16.to_le_bytes()),
                }
                let err = equal_bytes(&bad, &gate_host, &fused_gate, &contiguous)
                    .expect_err("known-bad output was admitted by the real oracle");
                println!("  KNOWN_BAD {control}: refused: {err}");
            }
            controls_done = true;
        }

        // 2026-09-25: Time, as weight bytes of both halves per call.
        let bytes = (2 * INTER * H) as f64;
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            f()?;
            gpu.synchronize(0)?;
            let t = Instant::now();
            for _ in 0..REPS {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() / REPS as f64)
        };
        let pair = time(&|| {
            gemm(
                &gpu,
                kmajor,
                a_fp8,
                a_scale,
                a_kmajor,
                &gate_w,
                gate_out.ptr,
                m,
            )?;
            gemm(&gpu, kmajor, a_fp8, a_scale, a_kmajor, &up_w, up_out.ptr, m)
        })?;
        let one = time(&|| {
            gemm(
                &gpu,
                kmajor,
                a_fp8,
                a_scale,
                a_kmajor,
                &fused,
                fused_out.ptr,
                m,
            )
        })?;
        println!(
            "  two GEMMs: {:8.2} us  {:7.0} GB/s   |  fused: {:8.2} us  {:7.0} GB/s   \
             ({:.2}x)",
            pair * 1e6,
            bytes / pair / 1e9,
            one * 1e6,
            bytes / one / 1e9,
            pair / one,
        );
    }

    println!(
        "\nserve spelling: the arm is `[defaults] ffn_gateup_fused` \
         (hopper `true`); `METRALE_FFN_GATEUP_FUSED=0` restores the two-GEMM \
         pair. Round-13 receipt and the round-16 prediction: \
         FFN-GATEUP-FUSION-ATTRIBUTION.md"
    );
    ensure!(failures == 0, "{failures} byte-equality gate(s) failed");
    println!("OK");
    Ok(())
}
