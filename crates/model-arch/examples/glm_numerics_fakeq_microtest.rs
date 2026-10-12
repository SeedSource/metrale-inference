// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: GPU bitwise gate of the two quality-test levers `METRALE_GLM_NUMERICS_FAKEQ`
//! (`glm5next_numerics.rs`) and `METRALE_GLM_KDA_STATE_ROUND` (`glm5next_kda/state_round.rs`,
//! `kernels/gb10/common/kda_state_round.cu`). Neither runs in a serve unless asked; this checks
//! that what they compute is what they claim, before a quality job relies on them.
//!
//! Sections (every comparison is on raw bit patterns, u16 for BF16 and u32 for FP32):
//! - (a) FP8: `fake_quant_inplace(.., Fmt::Fp8)` output equals the host reference of the scheme
//!   (`host_ref::fp8_fakeq_bits`: per-row max/448 scale, `v * (1/scale)` clamped, E4M3 round to
//!   nearest even, decode, times the scale, RNE to BF16), bit for bit.
//! - (b) NVFP4: `fake_quant_inplace(.., Fmt::Nvfp4)` output equals the host dequant
//!   (`host_ref::nvfp4_dequant_bits`) of the bytes `quantize_nvfp4_copy` produces for the same
//!   input (packed E2M1, E4M3 block scales, tensor scale), bit for bit. The returned error
//!   statistics equal the host's over the same sampled rows.
//! - (c) State round: the kernel's output equals `half::{bf16,f16}` round to nearest even and back
//!   to f32 bit for bit; the statistics slot holds the exact absmax (of the finite pre-rounding
//!   values), the exact count of finite values that round to infinity and of non-finite values,
//!   accumulated over two launches, in the decode and in the prefill half of the slot; with
//!   `log_every = 1` the last-block logic resets the right words (and prints its lines).
//! - (d) Known-bad: a deliberately wrong reference (FP8 scale 447 instead of 448; NVFP4 tensor
//!   scale off by 1e-4; the state rounded as the other format; an off-by-one overflow count) IS
//!   detected by the same comparisons.
//!
//! Data per shape: random (per-row gain spread, 0.1 % outliers), and an edge case built row by
//! row: zeros, one repeated value, alternating +-BF16 max, tiny values (1e-30), and rows of exact
//! E4M3 midpoints under a row max of 448 (scale 1, so every tie reaches the encoder).
//! Shapes (rows x cols): 128x4096, 4096x128, 32x4096, 288x4096, 4096x1536, 24x16384.
//!
//! Prints one line starting `PASS:` and exits 0, or a line starting `FAIL` and exits 1.
//!
//! Kernel modules (copied from the production handles): `gemv_fp8w` (`quantize_bf16_to_fp8`),
//! `dequant_fp8_rowscale_bf16`, `quantize_nvfp4` (`nvfp4_global_absmax`,
//! `quantize_bf16_to_nvfp4_mse`), `dequant_nvfp4_bf16` (`dequant_nvfp4_to_bf16`),
//! `kda_state_round` (`kda_state_round`).
//!
//! Run (GPU; build-mt sets the target env):
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm_numerics_fakeq_microtest
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::state_round::{
    Phase, RoundFmt, SLOT_WORDS, launch_round, round_ref_bits,
};
use metrale_model_arch::glm5next_layer::dense_fp8::quantize_nvfp4_copy;
use metrale_model_arch::glm5next_numerics::{
    Fmt, err_stats, fake_quant_inplace, host_ref, sample_rows,
};

const SHAPES: &[(usize, usize)] = &[
    (128, 4096),
    (4096, 128),
    (32, 4096),
    (288, 4096),
    (4096, 1536),
    (24, 16384),
];

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn n(&mut self) -> f64 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
}

fn up_bits(g: &dyn GpuBackend, d: &[u16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_bits(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn dn_u32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
    Ok(dn_bytes(g, p, n * 4)?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn mismatches(a: &[u16], b: &[u16]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Random rows: N(0, 0.02) with a per-row gain spread (x0.25..x4) and 0.1 % outliers at x30.
fn gen_random(rng: &mut Lcg, n: usize, k: usize) -> Vec<u16> {
    let mut w = Vec::with_capacity(n * k);
    for _ in 0..n {
        let gain = 0.25 * 16f64.powf(rng.u());
        for _ in 0..k {
            let mut v = rng.n() * 0.02 * gain;
            if rng.u() < 0.001 {
                v *= 30.0;
            }
            w.push(bf16::from_f64(v).to_bits());
        }
    }
    w
}

/// Midpoints between consecutive finite E4M3 magnitudes, as BF16 bits (every midpoint has at most
/// 4 mantissa bits, so it is exact in BF16), repeated to fill a row of `k`, first element 448.
fn tie_row(k: usize) -> Vec<u16> {
    let mut mids: Vec<f32> = (0u8..0x7E)
        .map(|c| (host_ref::e4m3_decode(c) + host_ref::e4m3_decode(c + 1)) / 2.0)
        .collect();
    let neg: Vec<f32> = mids.iter().map(|m| -m).collect();
    mids.extend(neg);
    let mut row = vec![bf16::from_f32(448.0).to_bits()];
    let mut i = 0;
    while row.len() < k {
        row.push(bf16::from_f32(mids[i % mids.len()]).to_bits());
        i += 1;
    }
    row
}

/// Edge rows cycling through: zeros, one repeated value, alternating +-BF16 max, tiny values,
/// E4M3 ties, then random rows for the rest.
fn gen_edge(rng: &mut Lcg, n: usize, k: usize) -> Vec<u16> {
    let rnd = gen_random(rng, n, k);
    let mut w = rnd.clone();
    let bf_max = bf16::MAX.to_bits();
    for r in 0..n.min(10) {
        let row: Vec<u16> = match r {
            0 => vec![0u16; k],
            1 => vec![bf16::from_f32(3.0).to_bits(); k],
            2 => (0..k)
                .map(|c| if c % 2 == 0 { bf_max } else { bf_max | 0x8000 })
                .collect(),
            3 => (0..k)
                .map(|c| bf16::from_f32(if c % 2 == 0 { 1e-30 } else { -2e-30 }).to_bits())
                .collect(),
            4 => tie_row(k),
            5 => vec![0x8000u16; k],
            6 => (0..k)
                .map(|c| bf16::from_f32(1e-3 * (c as f32 + 1.0)).to_bits())
                .collect(),
            _ => continue,
        };
        w[r * k..(r + 1) * k].copy_from_slice(&row);
    }
    w
}

/// Wrong FP8 reference: the scale divisor 447 instead of 448 (a one-line mistake).
#[allow(clippy::manual_clamp)] // mirrors fmaxf/fminf
fn fp8_wrong_ref(src: &[u16], n: usize, k: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(src.len());
    for r in 0..n {
        let row = &src[r * k..(r + 1) * k];
        let mx = row
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32().abs())
            .fold(0.0f32, f32::max);
        let scale = if mx > 0.0 { mx / 447.0 } else { 1.0 };
        let inv = 1.0f32 / scale;
        for &b in row {
            let v = (bf16::from_bits(b).to_f32() * inv).max(-448.0).min(448.0);
            let q = host_ref::e4m3_encode_rne(v);
            out.push(bf16::from_f32(host_ref::e4m3_decode(q) * scale).to_bits());
        }
    }
    out
}

struct Tally {
    checks: usize,
    fails: Vec<String>,
}
impl Tally {
    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        self.checks += 1;
        if !ok {
            self.fails.push(what());
        }
    }
}

fn fp8_and_nvfp4(g: &dyn GpuBackend, rng: &mut Lcg, t: &mut Tally) -> Result<()> {
    let s = g.default_stream();
    for &(n, k) in SHAPES {
        for (case, src) in [
            ("random", gen_random(rng, n, k)),
            ("edge", gen_edge(rng, n, k)),
        ] {
            let tag = format!("{n}x{k} {case}");
            // (a) FP8.
            let w = up_bits(g, &src)?;
            let st = fake_quant_inplace(g, w, n, k, Fmt::Fp8)?;
            let got = dn_bits(g, w, n * k)?;
            let want = host_ref::fp8_fakeq_bits(&src, n, k);
            let bad = mismatches(&got, &want);
            t.check(bad == 0, || {
                format!("(a) fp8 {tag}: {bad} of {} elements differ", n * k)
            });
            // Known-bad (d): the 447 reference must differ somewhere.
            let wrong = fp8_wrong_ref(&src, n, k);
            let det = mismatches(&got, &wrong);
            t.check(det > 0, || {
                format!("(d) fp8 {tag}: the deliberately wrong reference was NOT detected")
            });
            // Error statistics equal the host's over the same sampled rows.
            let rows = sample_rows(n, k);
            let pick = |v: &[u16]| -> Vec<u16> {
                rows.iter()
                    .flat_map(|&r| v[r * k..(r + 1) * k].iter().copied())
                    .collect()
            };
            let hs = err_stats(&pick(&src), &pick(&want));
            t.check(
                st.max_abs.to_bits() == hs.max_abs.to_bits()
                    && st.sum_sq_err.to_bits() == hs.sum_sq_err.to_bits()
                    && st.elems == hs.elems,
                || format!("(a) fp8 {tag}: error stats {st:?} != host {hs:?}"),
            );
            g.free(w)?;

            // (b) NVFP4: the quantizer's own bytes, dequantized on the host.
            let w = up_bits(g, &src)?;
            let w2 = up_bits(g, &src)?;
            let Some(q) = quantize_nvfp4_copy(g, w2, n, k, s)? else {
                bail!("quantize_nvfp4_copy returned None (NVFP4 kernels missing)");
            };
            let packed = dn_bytes(g, q.weight, n * k / 2)?;
            let scales = dn_bytes(g, q.weight_scale, n * k / 16)?;
            let st = fake_quant_inplace(g, w, n, k, Fmt::Nvfp4)?;
            let got = dn_bits(g, w, n * k)?;
            let want = host_ref::nvfp4_dequant_bits(&packed, &scales, q.weight_scale_2, n, k);
            let bad = mismatches(&got, &want);
            t.check(bad == 0, || {
                format!("(b) nvfp4 {tag}: {bad} of {} elements differ", n * k)
            });
            // 2026-10-12: x1.0001 was invisible: E2M1 x E4M3 products carry <= 6 significant
            // bits, exact in BF16, so a 1e-4 nudge rounded back (N0 FAIL M). 1% crosses BF16
            // rounding (half-ulp 0.2%); a flipped code in a nonzero nibble is the second control.
            let wrong =
                host_ref::nvfp4_dequant_bits(&packed, &scales, q.weight_scale_2 * 1.01, n, k);
            let mut flipped = packed.clone();
            if let Some(b) = flipped.iter_mut().find(|b| (**b & 0x7) != 0) {
                *b ^= 0x1;
            }
            let wrong2 = host_ref::nvfp4_dequant_bits(&flipped, &scales, q.weight_scale_2, n, k);
            let det = mismatches(&got, &wrong).min(mismatches(&got, &wrong2));
            t.check(det > 0, || {
                format!("(d) nvfp4 {tag}: the deliberately wrong reference was NOT detected")
            });
            let hs = err_stats(&pick(&src), &pick(&want));
            t.check(
                st.max_abs.to_bits() == hs.max_abs.to_bits() && st.elems == hs.elems,
                || format!("(b) nvfp4 {tag}: error stats {st:?} != host {hs:?}"),
            );
            for p in [w, w2, q.weight, q.weight_scale] {
                g.free(p)?;
            }
        }
    }
    Ok(())
}

/// Pre-rounding test values: edge list first, then random magnitudes over six decades.
fn gen_state(rng: &mut Lcg, n: usize) -> Vec<f32> {
    let edge = [
        0.0f32,
        -0.0,
        1.0 + 2f32.powi(-11),
        1.0 + 2f32.powi(-8),
        1.0 + 3.0 * 2f32.powi(-8),
        65504.0,
        65519.0,
        65519.99,
        65520.0,
        -65520.0,
        7.0e4,
        -1.0e30,
        f32::MAX,
        -f32::MAX,
        5.96e-8,
        2.98e-8,
        1e-45,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut v: Vec<f32> = edge.iter().copied().take(n).collect();
    while v.len() < n {
        let mag = 10f64.powf(rng.u() * 6.0 - 3.0);
        v.push((rng.n() * mag) as f32);
    }
    v
}

/// Host expectation: (rounded bits, absmax bits over finite values, overflow, nonfinite).
fn state_expect(v: &[f32], fmt: RoundFmt) -> (Vec<u32>, u32, u32, u32) {
    let mut out = Vec::with_capacity(v.len());
    let (mut amax, mut ovf, mut nf) = (0.0f32, 0u32, 0u32);
    for &x in v {
        let r = f32::from_bits(round_ref_bits(x, fmt));
        out.push(if x.is_nan() {
            f32::NAN.to_bits()
        } else {
            r.to_bits()
        });
        if x.is_finite() {
            amax = amax.max(x.abs());
            if !r.is_finite() {
                ovf += 1;
            }
        } else {
            nf += 1;
        }
    }
    (out, amax.to_bits(), ovf, nf)
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn same_f32(got: &[u32], want: &[u32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|&(&a, &b)| {
            if f32::from_bits(b).is_nan() {
                !f32::from_bits(a).is_nan()
            } else {
                a != b
            }
        })
        .count()
}

fn state_round(g: &dyn GpuBackend, rng: &mut Lcg, t: &mut Tally) -> Result<()> {
    let s = g.default_stream();
    let kern = g.kernel("kda_state_round", "kda_state_round")?;
    let slot_bytes = SLOT_WORDS * 4;
    for fmt in [RoundFmt::Bf16, RoundFmt::Fp16] {
        let other = if fmt == RoundFmt::Bf16 {
            RoundFmt::Fp16
        } else {
            RoundFmt::Bf16
        };
        for n in [7usize, 32 * 128 * 128, 1_000_003] {
            let tag = format!("{} n={n}", fmt.name());
            let (a, b) = (gen_state(rng, n), gen_state(rng, n));
            let (ea, eb) = (state_expect(&a, fmt), state_expect(&b, fmt));
            let slot = g.alloc(slot_bytes)?;
            g.memset(slot, 0, slot_bytes)?;
            let (da, db) = (up_f32(g, &a)?, up_f32(g, &b)?);
            // Two decode launches into the decode half, one prefill launch into the prefill half.
            launch_round(g, kern, da, n, fmt, slot, Phase::Decode, 0, 3, s)?;
            launch_round(g, kern, db, n, fmt, slot, Phase::Decode, 0, 3, s)?;
            let dc = up_f32(g, &a)?;
            launch_round(g, kern, dc, n, fmt, slot, Phase::Prefill, 0, 3, s)?;
            g.synchronize(s)?;
            let (ga, gb, gc) = (dn_u32(g, da, n)?, dn_u32(g, db, n)?, dn_u32(g, dc, n)?);
            let (bad_a, bad_b, bad_c) = (
                same_f32(&ga, &ea.0),
                same_f32(&gb, &eb.0),
                same_f32(&gc, &ea.0),
            );
            t.check(bad_a + bad_b + bad_c == 0, || {
                format!("(c) {tag}: rounded values differ ({bad_a}, {bad_b}, {bad_c})")
            });
            let w = dn_u32(g, slot, SLOT_WORDS)?;
            let want_dec = (
                f32::from_bits(ea.1).max(f32::from_bits(eb.1)).to_bits(),
                ea.2 + eb.2,
                ea.3 + eb.3,
            );
            t.check((w[0], w[1], w[2]) == want_dec, || {
                format!(
                    "(c) {tag}: decode stats {:?} != {want_dec:?}",
                    (w[0], w[1], w[2])
                )
            });
            t.check((w[4], w[5], w[6]) == (ea.1, ea.2, ea.3), || {
                format!(
                    "(c) {tag}: prefill stats {:?} != {:?}",
                    (w[4], w[5], w[6]),
                    (ea.1, ea.2, ea.3)
                )
            });
            t.check(w[3] == 0 && w[7] == 0 && w[8] == 0, || {
                format!(
                    "(c) {tag}: log_every=0 touched counters {:?}",
                    (w[3], w[7], w[8])
                )
            });

            // (d) known-bad references.
            let (wo, _, _, _) = state_expect(&a, other);
            t.check(same_f32(&ga, &wo) > 0, || {
                format!("(d) {tag}: rounding as the other format was NOT detected")
            });
            t.check(
                (want_dec.0, want_dec.1 + 1, want_dec.2) != (w[0], w[1], w[2]),
                || format!("(d) {tag}: an off-by-one overflow count was NOT detected"),
            );
            if n == 32 * 128 * 128 {
                t.check(ea.2 >= 1, || {
                    format!("(c) {tag}: edge list produced no overflow ({})", ea.2)
                });
            }

            // log_every = 1: the last block counts, prints and resets. Decode launch with pending
            // prefill statistics: prints the prefill line and the decode line, clears both.
            if n == 32 * 128 * 128 {
                let ls = g.alloc(slot_bytes)?;
                g.memset(ls, 0, slot_bytes)?;
                let pf = up_f32(g, &a)?;
                launch_round(g, kern, pf, n, fmt, ls, Phase::Prefill, 1, 5, s)?;
                g.synchronize(s)?;
                let w = dn_u32(g, ls, SLOT_WORDS)?;
                t.check(
                    (w[4], w[5], w[6], w[7], w[8]) == (ea.1, ea.2, ea.3, 1, 0),
                    || format!("(c) {tag}: prefill log_every=1 words {:?}", &w[4..9]),
                );
                let dd = up_f32(g, &b)?;
                launch_round(g, kern, dd, n, fmt, ls, Phase::Decode, 1, 5, s)?;
                g.synchronize(s)?;
                let w = dn_u32(g, ls, SLOT_WORDS)?;
                t.check(w.iter().all(|&x| x == 0), || {
                    format!("(c) {tag}: decode log_every=1 left words {w:?}")
                });
                let dd2 = up_f32(g, &b)?;
                launch_round(g, kern, dd2, n, fmt, ls, Phase::Decode, 2, 5, s)?;
                g.synchronize(s)?;
                let w = dn_u32(g, ls, SLOT_WORDS)?;
                t.check(w[3] == 1 && w[8] == 0 && w[0] != 0, || {
                    format!("(c) {tag}: log_every=2 after one launch words {w:?}")
                });
                for p in [ls, pf, dd, dd2] {
                    g.free(p)?;
                }
            }
            for p in [slot, da, db, dc] {
                g.free(p)?;
            }
        }
    }
    Ok(())
}

fn run() -> Result<String> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut rng = Lcg(0x6e75_6d66_7131);
    let mut t = Tally {
        checks: 0,
        fails: Vec::new(),
    };
    fp8_and_nvfp4(g, &mut rng, &mut t)?;
    state_round(g, &mut rng, &mut t)?;
    if !t.fails.is_empty() {
        bail!(
            "{} of {} checks failed: {}",
            t.fails.len(),
            t.checks,
            t.fails.join(" | ")
        );
    }
    Ok(format!(
        "PASS: glm_numerics_fakeq {} checks: fp8 and nvfp4 fake-quant bitwise vs host reference \
         ({} shapes x random/edge), state round bf16/fp16 bitwise with exact absmax/overflow/nonfinite \
         stats, known-bad references detected",
        t.checks,
        SHAPES.len()
    ))
}

fn main() {
    match run() {
        Ok(line) => println!("{line}"),
        Err(e) => {
            println!("FAIL glm_numerics_fakeq: {e:#}");
            std::process::exit(1);
        }
    }
}
