// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bitwise oracle for the Hopper FP8 activation-quantizer twin
//! against the shared `per_token_group_quant_fp8`.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, at every `(M, K)` in `M_DIMS` x `K_DIMS`, both
//!   arms wrote identical FP8 bytes and identical FP32 scale bytes, no guard
//!   byte around either output changed, and KNOWN_BAD differed on row 0.
//!
//! `kernels/hopper/common/fp8_act_quant_hopper.cu` keeps the shared kernel's
//! `amax / 448.0f`, `1e-12f` floor, per-element division by the scale, clamp
//! to +-448 and saturating `__nv_cvt_float_to_fp8`; only the load width, the
//! groups per CTA and the amax reduction differ. So the gate is byte equality,
//! not a tolerance. The worst absolute scale difference is printed only to make
//! a failure diagnosable.
//!
//! Guard bands of `SENTINEL` bytes bracket both output buffers of each arm. The
//! twin derives its K-group span from `gridDim.y`, so a span error can write
//! outside the payload, where a payload comparison would not see it.
//!
//! KNOWN_BAD (`e4m3_trunc`) re-encodes row 0 with round-toward-zero and must
//! differ from the device bytes somewhere; a harness that cannot see a rounding
//! change proves nothing by passing.
//!
//! Each arm's time is also reported as GB/s of compulsory traffic,
//! `M*K*(2 read + 1 write) + M*(K/128)*4` bytes, and as a share of `HBM_GBPS`.
//! An image without the twin (a build other than `kernels/hopper` or
//! `kernels/gb10/glm-5.3-flash`, which stages it through `[sources] use`) fails at
//! start with that message. `HBM_GBPS` is the Hopper figure: on gb10 (273 GB/s) read the
//! GB/s column, not the share.
//!
//! Run: `cargo run -p metrale-model-arch --features cuda,gpu-examples \
//!        --example native_fp8_act_quant_hopper_microtest`

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;

// 2026-10-06: GLM-5.3 shapes added: K 4096 (KDA/DSA/MLP inputs), 1536 (q_absorb), 16384
// (o_absorb), 2048 (shared-expert down) at the 8K prefill's 8191 rows.
const K_DIMS: [u32; 7] = [5120, 6144, 17408, 4096, 1536, 16384, 2048];
const M_DIMS: [u32; 6] = [16, 17, 25, 1168, 4576, 8191];
const M_MAX: u32 = 8191;
const GROUP: u32 = 128;
const E4M3_MAX: f32 = 448.0;
const GUARD: usize = 256;
const SENTINEL: u8 = 0x5a;
const WARMUP: u32 = 3;
const REPS: u32 = 20;
/// 2026-09-25: `memory_bandwidth_gbps` of `kernels/hopper/HARDWARE.toml`,
/// copied here; the harness does not read that file.
const HBM_GBPS: f64 = 3350.0;

/// 2026-09-25: Deterministic LCG with a fixed seed, so a failing byte
/// reproduces run to run.
struct Lcg(u32);
impl Lcg {
    fn next_bits(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn next_unit(&mut self) -> f32 {
        ((self.next_bits() >> 8) & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0
    }
}

/// 2026-09-25: One BF16 value, as the two bytes the kernel reads. The f32 is
/// truncated, not rounded, so the host holds exactly the value the device
/// sees.
fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}
fn bf16_value(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// 2026-09-25: Distinct 128-element groups that `build_rows` tiles across the
/// activation. Groups with different maxima make a reduction that mixes two
/// groups change a scale.
///
/// The first four groups are hand-built, and in row 0 they are K-groups 0 to 3:
/// - 0: all zero, so amax is 0 and the scale takes the `1e-12f` floor;
/// - 1: one 3.5e4 among values of magnitude at most 1e-3, which quantize to 0;
/// - 2: +-(1 + j/16) * 0.0625 and zeros;
/// - 3: +-448 and +-1e6; the +-1e6 elements set the scale and quantize to +-448.
const PATTERN_GROUPS: u32 = 1024;

fn build_pattern(seed: u32) -> Vec<u16> {
    let mut r = Lcg(seed);
    let n = (PATTERN_GROUPS * GROUP) as usize;
    let mut p = vec![0u16; n];
    for i in 0..GROUP as usize {
        let g1 = GROUP as usize + i;
        p[g1] = if i == 7 {
            bf16_bits(3.5e4)
        } else {
            bf16_bits(r.next_unit() * 1e-3)
        };
        let g2 = 2 * GROUP as usize + i;
        p[g2] = bf16_bits((1.0 + (i % 16) as f32 / 16.0) * 0.0625 * ((i % 3) as f32 - 1.0));
        let g3 = 3 * GROUP as usize + i;
        p[g3] = bf16_bits(match i % 4 {
            0 => E4M3_MAX,
            1 => -E4M3_MAX,
            2 => 1.0e6,
            _ => -1.0e6,
        });
    }
    for slot in p.iter_mut().skip(4 * GROUP as usize) {
        let mag = 10f32.powi(((r.next_bits() >> 3) % 7) as i32 - 4);
        *slot = bf16_bits(r.next_unit() * mag);
    }
    p
}

/// 2026-09-25: `M_MAX` rows of `k` BF16 elements, as device-ready bytes. A
/// shorter M uses the first M rows, so one host build serves every M in
/// `M_DIMS`.
fn build_rows(pattern: &[u16], k: u32) -> Vec<u8> {
    let mut out = vec![0u8; M_MAX as usize * k as usize * 2];
    for m in 0..M_MAX as usize {
        for i in 0..k as usize {
            let v = pattern[(m * 7 + i) % pattern.len()];
            let o = (m * k as usize + i) * 2;
            out[o..o + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn guarded(g: &dyn GpuBackend, bytes: usize) -> Result<(DevicePtr, DevicePtr)> {
    let base = g.alloc(bytes + 2 * GUARD)?;
    g.copy_h2d(&vec![SENTINEL; bytes + 2 * GUARD], base)?;
    Ok((base, base.offset(GUARD)))
}

fn guards_intact(g: &dyn GpuBackend, base: DevicePtr, bytes: usize) -> Result<bool> {
    let mut raw = vec![0u8; bytes + 2 * GUARD];
    g.copy_d2h(base, &mut raw)?;
    Ok(raw[..GUARD].iter().all(|b| *b == SENTINEL)
        && raw[GUARD + bytes..].iter().all(|b| *b == SENTINEL))
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn time_us(g: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// 2026-09-25: KNOWN_BAD. The software E4M3 encoder `scl_enc_fp8` of
/// `kernels/gb10/common/per_token_group_quant_fp8.cu` without its two rounding
/// terms (`+ (1u << 19)` on the normal path, `+ 0.5f` on the subnormal path),
/// so it rounds toward zero. NaN and saturation are encoded as there.
fn e4m3_trunc(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let bb = v.to_bits();
    let sign = (bb >> 31) & 1;
    let e = ((bb >> 23) & 0xFF) as i32 - 127;
    let man = bb & 0x7F_FFFF;
    let mut ee = e + 7;
    let em;
    if ee < 1 {
        ee = 0;
        let a = v.abs();
        em = if e >= -10 {
            ((a / 0.001_953_125).floor() as u32).min(7)
        } else {
            0
        };
    } else if ee > 15 {
        ee = 15;
        em = 6;
    } else {
        em = man >> 20;
    }
    ((sign << 7) | ((ee as u32) << 3) | em) as u8
}

/// 2026-09-25: Number of differing bytes between two slices, and the first
/// differing index (`usize::MAX` when none differ).
fn first_diff(a: &[u8], b: &[u8]) -> (usize, usize) {
    let mut n = 0usize;
    let mut at = usize::MAX;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            if at == usize::MAX {
                at = i;
            }
            n += 1;
        }
    }
    (n, at)
}

/// 2026-09-25: Worst absolute difference between two FP32 tensors given as
/// little-endian bytes.
fn worst_f32(a: &[u8], b: &[u8]) -> f64 {
    let mut w = 0.0f64;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let xv = f64::from(f32::from_le_bytes(x.try_into().unwrap()));
        let yv = f64::from(f32::from_le_bytes(y.try_into().unwrap()));
        w = w.max((xv - yv).abs());
    }
    w
}

struct Arm {
    fp8: Vec<u8>,
    scale: Vec<u8>,
    us: f64,
}

fn run_arm(
    g: &dyn GpuBackend,
    quant: ops::Fp8ActQuant,
    input: DevicePtr,
    m: u32,
    k: u32,
) -> Result<(Arm, bool)> {
    let fp8_bytes = m as usize * k as usize;
    let scale_bytes = m as usize * (k / GROUP) as usize * 4;
    let (fp8_base, fp8) = guarded(g, fp8_bytes)?;
    let (scale_base, scale) = guarded(g, scale_bytes)?;

    let us = time_us(g, || {
        ops::per_token_group_quant_fp8(g, quant, input, fp8, scale, m, k, 0)
    })?;
    g.synchronize(0)?;

    let out = Arm {
        fp8: dn(g, fp8, fp8_bytes)?,
        scale: dn(g, scale, scale_bytes)?,
        us,
    };
    let ok = guards_intact(g, fp8_base, fp8_bytes)? && guards_intact(g, scale_base, scale_bytes)?;
    g.free(fp8_base).ok();
    g.free(scale_base).ok();
    Ok((out, ok))
}

/// 2026-09-25: GB/s of compulsory traffic for one launch: read the BF16
/// activation once, write the FP8 bytes once, write the FP32 scales once.
fn gbps(m: u32, k: u32, us: f64) -> f64 {
    let bytes = m as f64 * k as f64 * 3.0 + m as f64 * (k / GROUP) as f64 * 4.0;
    bytes / (us * 1e-6) / 1e9
}

fn leg(
    g: &dyn GpuBackend,
    shared: ops::Fp8ActQuant,
    hopper: ops::Fp8ActQuant,
    host_rows: &[u8],
    input: DevicePtr,
    m: u32,
    k: u32,
) -> Result<()> {
    g.copy_h2d(&host_rows[..m as usize * k as usize * 2], input)?;

    let (a, a_ok) = run_arm(g, shared, input, m, k)?;
    let (b, b_ok) = run_arm(g, hopper, input, m, k)?;

    let (fp8_n, fp8_at) = first_diff(&a.fp8, &b.fp8);
    let (sc_n, sc_at) = first_diff(&a.scale, &b.scale);

    eprintln!(
        "  M={m:<5} K={k:<6} shared {:8.2} us ({:6.1} GB/s, {:4.1}% HBM)  \
         hopper {:8.2} us ({:6.1} GB/s, {:4.1}% HBM)  speedup {:.2}x",
        a.us,
        gbps(m, k, a.us),
        gbps(m, k, a.us) / HBM_GBPS * 100.0,
        b.us,
        gbps(m, k, b.us),
        gbps(m, k, b.us) / HBM_GBPS * 100.0,
        a.us / b.us,
    );

    anyhow::ensure!(a_ok, "M={m} K={k}: the shared kernel wrote past its buffer");
    anyhow::ensure!(b_ok, "M={m} K={k}: the Hopper twin wrote past its buffer");
    anyhow::ensure!(
        fp8_n == 0,
        "M={m} K={k}: {fp8_n} FP8 bytes differ (first at {fp8_at}); the twin must be \
         BIT-identical to `per_token_group_quant_fp8`"
    );
    anyhow::ensure!(
        sc_n == 0,
        "M={m} K={k}: {sc_n} scale bytes differ (first at {sc_at}, worst abs {:.3e})",
        worst_f32(&a.scale, &b.scale)
    );

    // 2026-09-25: KNOWN_BAD: row 0 of the same input, divided by the twin's
    // own scales and re-encoded with round-toward-zero. It must differ from the
    // twin's bytes, or the byte comparison above proves nothing about rounding.
    // One row is K elements, each a chance to round differently.
    let mut bad = 0usize;
    for i in 0..k as usize {
        let scale = f32::from_le_bytes(
            b.scale[(i / GROUP as usize) * 4..(i / GROUP as usize) * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let v = bf16_value(u16::from_le_bytes(
            host_rows[i * 2..i * 2 + 2].try_into().unwrap(),
        ));
        let q = (v / scale).clamp(-E4M3_MAX, E4M3_MAX);
        if e4m3_trunc(q) != b.fp8[i] {
            bad += 1;
        }
    }
    anyhow::ensure!(
        bad > 0,
        "M={m} K={k}: KNOWN_BAD (round-toward-zero) matched the kernel on every byte of \
         row 0 — this harness cannot see a rounding change and its PASS means nothing"
    );
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let resolved = ops::Fp8ActQuant::resolve(g);
    anyhow::ensure!(
        resolved.shared.0 != 0,
        "`per_token_group_quant_fp8` is not in this image; nothing to compare against"
    );
    anyhow::ensure!(
        resolved.hopper.0 != 0,
        "`fp8_act_quant_hopper::per_token_group_quant_fp8_hopper` is not in this image — \
         this microtest is for a build of `kernels/hopper` or gb10 glm-5.3-flash"
    );
    let shared = ops::Fp8ActQuant::shared_only(resolved.shared);
    let hopper = ops::Fp8ActQuant {
        shared: KernelHandle(0),
        hopper: resolved.hopper,
    };

    eprintln!(
        "native_fp8_act_quant_hopper_microtest: groups/CTA={} HBM={HBM_GBPS} GB/s",
        ops::FP8_QUANT_HOPPER_GROUPS_PER_CTA
    );
    let pattern = build_pattern(0x5EED_1234);
    for k in K_DIMS {
        let rows = build_rows(&pattern, k);
        let input = g.alloc(M_MAX as usize * k as usize * 2)?;
        for m in M_DIMS {
            leg(g, shared, hopper, &rows, input, m, k)?;
        }
        g.free(input).ok();
    }
    eprintln!("  ALL LEGS BIT-IDENTICAL (fp8 bytes and fp32 scales), KNOWN_BAD fired on each");
    // 2026-10-06: The build-mt verdict line (scripts/race/build-mt.sh contract).
    println!("PASS: the twin's FP8 bytes and scales equal the shared kernel's on every leg; KNOWN_BAD fired");
    Ok(())
}
