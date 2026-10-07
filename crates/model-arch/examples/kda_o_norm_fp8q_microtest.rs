// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Bitwise oracle for `METRALE_GLM_NORM_FP8_QUANT_FUSE`: the KDA sigmoid-gated output
//! norm that also writes the FP8 activation of `o_proj`'s W8A8 GEMM
//! (`kda_o_norm_gated_bf16_fp8q`, `kda_o_norm_gated_bf16in_fp8q`) against the incumbent pair it
//! replaces (`kda_o_norm_gated_bf16` / `kda_o_norm_gated_bf16in`, then
//! `ops::per_token_group_quant_fp8` over the BF16 norm output).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if, for every token count in `T_DIMS` and both input kinds (FP32 core:
//!   the chunked-TC / token-loop prefill; BF16 core: the FlashKDA prefill), the fused kernel's
//!   BF16 norm output, FP8 bytes and FP32 scale bytes equal the incumbent pair's byte for byte
//!   against every quantizer arm in the image (the shared kernel, and the Hopper twin when
//!   staged), no guard byte around a fused output changed, and KNOWN_BAD differed.
//! - Shapes are the GLM-5.3 TP2 KDA layer's: 32 local heads x head_dim 128 (qkv_dim 4096 = the
//!   quantizer's K), at 8192 rows (the full-width window), 257 and 256 (a prefill tail and a
//!   whole sub-chunk).
//!
//! Why byte equality is the right gate: the fused kernel quantizes the BF16-ROUNDED value it
//! stores (a quantizer launch reads that value back), with the quantizer's arithmetic. Any
//! difference, e.g. quantizing the FP32 pre-rounding value, changes amax or a rounding and shows
//! here.
//!
//! The input mixes magnitudes (1e-6 to 1e3 per head), all-zero heads (the 1e-12 scale floor),
//! a lone outlier, tiny values and a wide gate/weight spread. KNOWN_BAD re-encodes row 0 of the
//! fused result with round-toward-zero and must differ somewhere: a harness that cannot see a
//! rounding change proves nothing by passing.
//!
//! Times (host wall, 20 reps): the incumbent pair, as production launches it (the quantizer
//! `Fp8ActQuant::resolve` picks under `METRALE_FP8_ACT_QUANT_HOPPER`), against the fused kernel.
//!
//! Run: `cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!        --example kda_o_norm_fp8q_microtest`

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::{ops, try_kernel};

const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const QKV: usize = HEADS * HEAD_DIM;
const T_DIMS: [usize; 3] = [8192, 257, 256];
const T_MAX: usize = 8192;
const EPS: f32 = 1e-6;
const E4M3_MAX: f32 = 448.0;
const GUARD: usize = 256;
const SENTINEL: u8 = 0x5a;
const WARMUP: u32 = 3;
const REPS: u32 = 20;

/// 2026-10-07: Deterministic LCG with a fixed seed, so a failing byte reproduces run to run.
struct Lcg(u32);
impl Lcg {
    fn next_bits(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn next_unit(&mut self) -> f32 {
        ((self.next_bits() >> 8) & 0xFF_FF_FF) as f32 / 8_388_608.0 - 1.0
    }
}

/// 2026-10-07: BF16 round-to-nearest-even of an f32, as the device `__float2bfloat16` does
/// (finite inputs only).
fn bf16_rne(v: f32) -> u16 {
    let b = v.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    (r >> 16) as u16
}

fn bf16_value(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// 2026-10-07: One head's 128 input values for (token, head): a magnitude class picked from the
/// position, so every class meets every kernel path.
fn head_values(r: &mut Lcg, t: usize, h: usize, out: &mut [f32]) {
    let class = (t * 7 + h * 3) % 11;
    let mag = 10f32.powi(((r.next_bits() >> 3) % 10) as i32 - 6);
    for (i, v) in out.iter_mut().enumerate() {
        *v = match class {
            0 => 0.0,
            1 => {
                if i == 17 {
                    3.5e4
                } else {
                    r.next_unit() * 1e-3
                }
            }
            2 => r.next_unit() * 1e-20,
            3 => r.next_unit() * 1e3,
            _ => r.next_unit() * mag,
        };
    }
}

struct Inputs {
    core_f32: Vec<u8>,
    core_bf16: Vec<u8>,
    gate: Vec<u8>,
    weight: Vec<u8>,
}

fn build_inputs() -> Inputs {
    let mut r = Lcg(0x0DD5_EED1);
    let n = T_MAX * QKV;
    let mut core_f32 = Vec::with_capacity(n * 4);
    let mut core_bf16 = Vec::with_capacity(n * 2);
    let mut gate = Vec::with_capacity(n * 2);
    let mut head = vec![0f32; HEAD_DIM];
    for t in 0..T_MAX {
        for h in 0..HEADS {
            head_values(&mut r, t, h, &mut head);
            for &v in &head {
                core_f32.extend_from_slice(&v.to_le_bytes());
                core_bf16.extend_from_slice(&bf16_rne(v).to_le_bytes());
            }
            // 2026-10-07: Gate in [-8, 8]; a few heads saturate the sigmoid toward 0.
            let spread = if (t + h) % 13 == 0 { 40.0 } else { 8.0 };
            for _ in 0..HEAD_DIM {
                gate.extend_from_slice(&bf16_rne(r.next_unit() * spread).to_le_bytes());
            }
        }
    }
    let mut weight = Vec::with_capacity(HEAD_DIM * 2);
    for _ in 0..HEAD_DIM {
        weight.extend_from_slice(&bf16_rne(0.1 + 3.0 * (r.next_unit() * 0.5 + 0.5)).to_le_bytes());
    }
    Inputs {
        core_f32,
        core_bf16,
        gate,
        weight,
    }
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

/// 2026-10-07: (differing byte count, first differing index or `usize::MAX`).
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

/// 2026-10-07: KNOWN_BAD: E4M3 encode without the two rounding terms (round toward zero); NaN
/// and saturation as `per_token_group_quant_fp8.cu`'s software encoder.
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

/// 2026-10-07: Which input the norm reads.
#[derive(Clone, Copy)]
enum Kind {
    F32,
    Bf16,
}

struct Kernels {
    norm: KernelHandle,
    fused: KernelHandle,
}

struct Bufs {
    core: DevicePtr,
    gate: DevicePtr,
    weight: DevicePtr,
    out_ref: DevicePtr,
    fp8_ref: DevicePtr,
    scale_ref: DevicePtr,
}

/// 2026-10-07: The unfused norm launch, as `back_end_with` issues it.
fn launch_norm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    out: DevicePtr,
    t: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(t * HEADS) as u32, 1, 1])
        .block([HEAD_DIM as u32, 1, 1])
        .arg_ptr(b.core)
        .arg_ptr(b.gate)
        .arg_ptr(b.weight)
        .arg_ptr(out)
        .arg_u32(HEAD_DIM as u32)
        .arg_f32(EPS)
        .launch(0)
}

/// 2026-10-07: The fused launch, as the `fill` closure in `back_end_with` issues it.
fn launch_fused(
    g: &dyn GpuBackend,
    k: KernelHandle,
    b: &Bufs,
    out: DevicePtr,
    fp8: DevicePtr,
    scale: DevicePtr,
    t: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(t * HEADS) as u32, 1, 1])
        .block([HEAD_DIM as u32, 1, 1])
        .arg_ptr(b.core)
        .arg_ptr(b.gate)
        .arg_ptr(b.weight)
        .arg_ptr(out)
        .arg_ptr(fp8)
        .arg_ptr(scale)
        .arg_u32(HEAD_DIM as u32)
        .arg_f32(EPS)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn leg(
    g: &dyn GpuBackend,
    name: &str,
    ks: &Kernels,
    arms: &[(&str, ops::Fp8ActQuant)],
    prod: ops::Fp8ActQuant,
    b: &Bufs,
    t: usize,
) -> Result<()> {
    let n = t * QKV;
    let (ob, out) = guarded(g, n * 2)?;
    let (fb, fp8) = guarded(g, n)?;
    let (sb, scale) = guarded(g, t * HEADS * 4)?;

    launch_fused(g, ks.fused, b, out, fp8, scale, t)?;
    g.synchronize(0)?;
    let (f_out, f_fp8, f_sc) = (
        dn(g, out, n * 2)?,
        dn(g, fp8, n)?,
        dn(g, scale, t * HEADS * 4)?,
    );

    // 2026-10-07: The incumbent norm once; its BF16 output is what every quantizer arm reads.
    launch_norm(g, ks.norm, b, b.out_ref, t)?;
    g.synchronize(0)?;
    let r_out = dn(g, b.out_ref, n * 2)?;
    let (n_out, at_out) = first_diff(&r_out, &f_out);
    anyhow::ensure!(
        n_out == 0,
        "{name} t={t}: {n_out} BF16 norm-output bytes differ (first at {at_out})"
    );

    for (arm, quant) in arms {
        ops::per_token_group_quant_fp8(
            g,
            *quant,
            b.out_ref,
            b.fp8_ref,
            b.scale_ref,
            t as u32,
            QKV as u32,
            0,
        )?;
        g.synchronize(0)?;
        let (r_fp8, r_sc) = (dn(g, b.fp8_ref, n)?, dn(g, b.scale_ref, t * HEADS * 4)?);
        let (n8, at8) = first_diff(&r_fp8, &f_fp8);
        let (ns, ats) = first_diff(&r_sc, &f_sc);
        anyhow::ensure!(
            n8 == 0,
            "{name} t={t} vs {arm}: {n8} FP8 bytes differ (first at {at8})"
        );
        anyhow::ensure!(
            ns == 0,
            "{name} t={t} vs {arm}: {ns} scale bytes differ (first at {ats})"
        );
    }

    let guards = guards_intact(g, ob, n * 2)?
        && guards_intact(g, fb, n)?
        && guards_intact(g, sb, t * HEADS * 4)?;
    anyhow::ensure!(
        guards,
        "{name} t={t}: the fused kernel wrote past an output buffer"
    );

    // 2026-10-07: KNOWN_BAD, row 0: the fused BF16 values over the fused scales, round-toward-zero.
    let mut bad = 0usize;
    for i in 0..QKV {
        let sc = f32::from_le_bytes(
            f_sc[(i / HEAD_DIM) * 4..(i / HEAD_DIM) * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let v = bf16_value(u16::from_le_bytes(
            f_out[i * 2..i * 2 + 2].try_into().unwrap(),
        ));
        if e4m3_trunc((v / sc).clamp(-E4M3_MAX, E4M3_MAX)) != f_fp8[i] {
            bad += 1;
        }
    }
    anyhow::ensure!(
        bad > 0,
        "{name} t={t}: KNOWN_BAD (round-toward-zero) matched on every byte of row 0; this \
         harness cannot see a rounding change"
    );

    // 2026-10-07: Time the production incumbent pair against the fused kernel.
    let pair = time_us(g, || {
        launch_norm(g, ks.norm, b, b.out_ref, t)?;
        ops::per_token_group_quant_fp8(
            g,
            prod,
            b.out_ref,
            b.fp8_ref,
            b.scale_ref,
            t as u32,
            QKV as u32,
            0,
        )
    })?;
    let norm_only = time_us(g, || launch_norm(g, ks.norm, b, b.out_ref, t))?;
    let fused = time_us(g, || launch_fused(g, ks.fused, b, out, fp8, scale, t))?;
    eprintln!(
        "  {name:<5} t={t:<5} norm {norm_only:8.1} us  norm+quant {pair:8.1} us  \
         fused {fused:8.1} us  saves {:7.1} us ({:.2}x)  [bytes equal: BF16 out, FP8, scales; \
         {} quantizer arm(s)]",
        pair - fused,
        pair / fused,
        arms.len()
    );
    for p in [ob, fb, sb] {
        g.free(p).ok();
    }
    Ok(())
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let resolved = ops::Fp8ActQuant::resolve(g);
    anyhow::ensure!(
        resolved.available(),
        "no per_token_group_quant_fp8 in this image"
    );
    let mut arms: Vec<(&str, ops::Fp8ActQuant)> = Vec::new();
    if resolved.shared.0 != 0 {
        arms.push(("shared", ops::Fp8ActQuant::shared_only(resolved.shared)));
    }
    if resolved.hopper.0 != 0 {
        let hopper = ops::Fp8ActQuant {
            shared: KernelHandle(0),
            hopper: resolved.hopper,
        };
        arms.push(("hopper", hopper));
    }

    let kinds: [(&str, Kind, (&str, &str), (&str, &str)); 2] = [
        (
            "f32",
            Kind::F32,
            ("kda_layer_ops", "kda_o_norm_gated_bf16"),
            ("kda_layer_ops", "kda_o_norm_gated_bf16_fp8q"),
        ),
        (
            "bf16",
            Kind::Bf16,
            ("kda_flashkda_glue", "kda_o_norm_gated_bf16in"),
            ("kda_layer_ops", "kda_o_norm_gated_bf16in_fp8q"),
        ),
    ];

    eprintln!(
        "kda_o_norm_fp8q_microtest: {HEADS} heads x {HEAD_DIM} (K={QKV}); quantizer arms: {}",
        arms.iter().map(|a| a.0).collect::<Vec<_>>().join(", ")
    );
    let inputs = build_inputs();
    let gate = g.alloc(inputs.gate.len())?;
    g.copy_h2d(&inputs.gate, gate)?;
    let weight = g.alloc(inputs.weight.len())?;
    g.copy_h2d(&inputs.weight, weight)?;
    let out_ref = g.alloc(T_MAX * QKV * 2)?;
    let fp8_ref = g.alloc(T_MAX * QKV)?;
    let scale_ref = g.alloc(T_MAX * HEADS * 4)?;
    // 2026-10-07: The timing quantizer is the one production would pick (`Fp8ActQuant::pick`
    // reads METRALE_FP8_ACT_QUANT_HOPPER); `resolved` makes that choice per launch.
    let prod = resolved;

    let mut ran = 0usize;
    for (name, kind, norm, fused) in kinds {
        let ks = Kernels {
            norm: try_kernel(g, norm.0, norm.1),
            fused: try_kernel(g, fused.0, fused.1),
        };
        anyhow::ensure!(
            ks.fused.0 != 0,
            "{name}: `{}::{}` is not in this image",
            fused.0,
            fused.1
        );
        if ks.norm.0 == 0 {
            // 2026-10-07: The FlashKDA glue is staged only where FlashKDA is; nothing to compare.
            eprintln!(
                "  {name}: `{}::{}` is not in this image; leg skipped",
                norm.0, norm.1
            );
            continue;
        }
        let bytes = match kind {
            Kind::F32 => &inputs.core_f32,
            Kind::Bf16 => &inputs.core_bf16,
        };
        let core = g.alloc(bytes.len())?;
        g.copy_h2d(bytes, core)?;
        let b = Bufs {
            core,
            gate,
            weight,
            out_ref,
            fp8_ref,
            scale_ref,
        };
        for t in T_DIMS {
            leg(g, name, &ks, &arms, prod, &b, t)?;
            ran += 1;
        }
        g.free(core).ok();
    }
    anyhow::ensure!(ran > 0, "no leg ran");
    eprintln!(
        "  ALL LEGS BIT-IDENTICAL (BF16 norm output, FP8 bytes, FP32 scales), KNOWN_BAD fired"
    );
    // 2026-10-07: The build-mt verdict line (scripts/race/build-mt.sh contract).
    println!(
        "PASS: fused o_norm+quant equals the unfused norm + per_token_group_quant_fp8 byte for \
         byte on every leg; KNOWN_BAD fired"
    );
    Ok(())
}
