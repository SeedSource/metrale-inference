// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Byte-parity gate and timing for the token-major GLM mHC mix
//! (`glm5next_hc_mix_bf16_tokmajor`, `METRALE_GLM_MHC_TOKMAJOR`) against
//! `glm5next_hc_mix_bf16`.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants: none beyond the types.
//!
//! Two comparisons per case, on the same device inputs:
//! - MIX: one unsliced launch each, `hc_mix_bf16` on grid `(T, mix_hc)` and the token-major
//!   kernel on grid `(T, 1)`; the `mix` rows are compared byte for byte.
//! - PRE: the whole `glm_hc_pre_sliced_mix` at `MHC_SLICE_ROWS` (mix + `hc_finish` per slice),
//!   `tokmajor` false then true; `y`, `post`, `comb` and `mix` are compared byte for byte. The
//!   explicit `tokmajor` argument skips the read-once lever and its `MHC_TOKMAJOR_MIN_ROWS`
//!   floor, so 1 and 7 rows run the token-major kernel too.
//!
//! Cases: rows {1, 7, 256, 1000} at GLM-5.3-Flash's hidden 4096, `hc_mult` 4 (`mix_hc` 24);
//! plus 7 rows at hidden 1000 (`hc_dim` 4000, not a multiple of 256, so the last stride is
//! partial) and 7 rows at `hc_mult` 2 (`mix_hc` 8, fewer rows than accumulators). Highway rows
//! cycle through four magnitude classes by token: mixed per-element exponents 1e-8..1e8 with
//! zeros and negative zeros, ~1e-3, ~1e4, and ~1e-21 (squares below FP32's normal range). The
//! BF16 `hc_fn` mixes exponents 1e-4..1e1.
//!
//! Each arm's outputs start filled with its own poison byte (0xAB reference, 0xCD token-major),
//! so a byte either arm leaves unwritten cannot compare equal. A negative control (one flipped
//! highway bit) must change the token-major mix.
//!
//! Timing (rows 1, 7, 256, 1000 at the production geometry): wall clock over
//! `MHC_TOKMAJOR_REPS` (default 20) mix launches after 3 warm-ups, synchronised at both ends;
//! `hot` back to back, `cold` with a 64 MiB `memset_async` before each launch (its own mean time
//! subtracted), as in `glm5next_hc_slice_microtest`. These numbers also place
//! `MHC_TOKMAJOR_MIN_ROWS` (64, PROVISIONAL).
//!
//! Exit: 0 when every case is byte-identical, 1 on any difference or a negative control that does
//! not fire, 2 when a kernel is absent from this target.
//!
//! PRE at 1000 rows needs `mhc_mix_max_tokens() >= 1000`, which the prefill levers size, so:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   METRALE_GLM_PREFILL_STAGED=1 METRALE_GLM_PREFILL_ROWS=256 METRALE_GLM_PREFILL_ROWS_FFN=1024 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example mhc_tokmajor_bitparity_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, MHC_SLICE_ROWS, MHC_TOKMAJOR_MIN_ROWS,
    glm_hc_pre_sliced_mix, mhc_mix_max_tokens,
};

const SINKHORN_ITERS: u32 = 20;
const HC_EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;
const FLUSH_BYTES: usize = 64 << 20;
const POISON_REF: u8 = 0xAB;
const POISON_NEW: u8 = 0xCD;
/// 2026-10-01: `(rows, hidden, hc_mult)`.
const CASES: &[(usize, usize, usize)] = &[
    (1, 4096, 4),
    (7, 4096, 4),
    (256, 4096, 4),
    (1000, 4096, 4),
    (7, 1000, 4),
    (7, 4096, 2),
];

fn mix_hc(hc: usize) -> usize {
    (2 + hc) * hc
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    /// 2026-10-01: Uniform in [-1, 1).
    fn s(&mut self) -> f32 {
        (2.0 * (self.next() as f64 / (1u64 << 53) as f64) - 1.0) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// 2026-10-01: `[rows, hc * hidden]` FP32 highway, magnitude class by token (see the header).
fn highway(r: &mut Lcg, rows: usize, hc_dim: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(rows * hc_dim);
    for t in 0..rows {
        for _ in 0..hc_dim {
            let x = match t % 4 {
                0 => match r.below(16) {
                    0 => 0.0,
                    1 => -0.0,
                    _ => r.s() * 10f32.powi(r.below(17) as i32 - 8),
                },
                1 => r.s() * 1e-3,
                2 => r.s() * 1e4,
                _ => r.s() * 1e-21,
            };
            v.push(x);
        }
    }
    v
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn f32_bytes(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-01: First differing byte of two equal-length buffers, if any.
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// 2026-10-01: One mix launch over `rows` tokens: `hc_mix_bf16` on grid `(rows, mix_hc)`, or the
/// token-major kernel on grid `(rows, 1)`; same arguments.
#[allow(clippy::too_many_arguments)]
fn mix(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    grid_y: u32,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    out: DevicePtr,
    rows: usize,
    hidden: usize,
    hc: usize,
) -> Result<()> {
    KernelLaunch::new(g, kern)
        .grid([rows as u32, grid_y, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(hc_fn)
        .arg_ptr(out)
        .arg_u32(hidden as u32)
        .arg_u32(hc as u32)
        .arg_f32(NORM_EPS)
        .launch(0)
}

/// 2026-10-01: One PRE arm's outputs, `mix` scratch included.
struct Outs {
    bufs: [DevicePtr; 4],
    sizes: [usize; 4],
}

impl Outs {
    fn new(g: &dyn GpuBackend, rows: usize, hidden: usize, hc: usize) -> Result<Self> {
        let sizes = [
            rows * hidden * 2,
            rows * hc * 4,
            rows * hc * hc * 4,
            rows * mix_hc(hc) * 4,
        ];
        let mut bufs = [DevicePtr::NULL; 4];
        for (b, n) in bufs.iter_mut().zip(sizes) {
            *b = g.alloc(n.max(1))?;
        }
        Ok(Self { bufs, sizes })
    }

    fn poison(&self, g: &dyn GpuBackend, byte: u8) -> Result<()> {
        for (p, n) in self.bufs.iter().zip(self.sizes) {
            g.copy_h2d(&vec![byte; n], *p)?;
        }
        Ok(())
    }

    fn read(&self, g: &dyn GpuBackend) -> Result<Vec<Vec<u8>>> {
        self.bufs
            .iter()
            .zip(self.sizes)
            .map(|(p, n)| down(g, *p, n))
            .collect()
    }

    fn free(&self, g: &dyn GpuBackend) {
        for p in self.bufs {
            g.free(p).ok();
        }
    }
}

/// 2026-10-01: `glm_hc_pre_sliced_mix` at `MHC_SLICE_ROWS` into `o`.
#[allow(clippy::too_many_arguments)]
fn pre(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    w: &Glm5NextMhcSiteWeights,
    streams: DevicePtr,
    o: &Outs,
    rows: usize,
    hidden: usize,
    hc: usize,
    tokmajor: bool,
) -> Result<()> {
    let wm = Glm5NextMhcSiteWeights { mix: o.bufs[3], ..*w };
    glm_hc_pre_sliced_mix(
        g,
        k,
        streams,
        &wm,
        o.bufs[0],
        o.bufs[1],
        o.bufs[2],
        rows as u32,
        hidden as u32,
        hc as u32,
        SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
        MHC_SLICE_ROWS,
        tokmajor,
        0,
    )
}

/// 2026-10-01: Mean wall milliseconds per call of `f` after 3 warm-up calls; with `flush`, a
/// `memset_async` of the flush buffer precedes each call and its own mean time is subtracted.
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
    Ok(total - run(false, reps)?)
}

fn main() -> Result<()> {
    // 2026-10-01: `ptx_modules()` aliases the first compiled target, so the backend is built from
    // the glm-5.3-flash set, as in `dsa_mla_headgroup_bitparity_microtest`.
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        println!("glm-5.3-flash kernel target not built - SKIP");
        std::process::exit(2);
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextMhcKernels::resolve(g)?;
    for (h, name) in [
        (k.hc_mix_bf16, "glm5next_hc_mix_bf16"),
        (k.hc_mix_bf16_tokmajor, "glm5next_hc_mix_bf16_tokmajor"),
    ] {
        if h.0 == 0 {
            println!("{name} absent from this target - SKIP");
            std::process::exit(2);
        }
    }
    let max_rows = CASES.iter().map(|c| c.0).max().unwrap_or(0);
    if max_rows > mhc_mix_max_tokens() {
        bail!(
            "rows={max_rows} exceeds mhc_mix_max_tokens() = {}: run with \
             METRALE_GLM_PREFILL_STAGED=1 METRALE_GLM_PREFILL_ROWS=256 \
             METRALE_GLM_PREFILL_ROWS_FFN=1024 (see the header)",
            mhc_mix_max_tokens()
        );
    }
    let reps = std::env::var("MHC_TOKMAJOR_REPS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(20)
        .max(1);
    println!(
        "mhc tokmajor bit-parity: REF = glm5next_hc_mix_bf16 grid (T, mix_hc); NEW = \
         glm5next_hc_mix_bf16_tokmajor grid (T, 1); PRE at MHC_SLICE_ROWS={MHC_SLICE_ROWS}; \
         production floor MHC_TOKMAJOR_MIN_ROWS={MHC_TOKMAJOR_MIN_ROWS}; reps={reps}\n"
    );

    let flush = g.alloc(FLUSH_BYTES)?;
    let (mut compared, mut failed) = (0usize, 0usize);
    let mut control_fired = None;
    for (ci, &(rows, hidden, hc)) in CASES.iter().enumerate() {
        let (hc_dim, m) = (hc * hidden, mix_hc(hc));
        let mut r = Lcg(0x70C3_A70A_0000 + ci as u64);
        let x = highway(&mut r, rows, hc_dim);
        let fn_bytes: Vec<u8> = (0..m * hc_dim)
            .flat_map(|_| {
                let e = r.below(6) as i32 - 4;
                bf16::from_f32(r.s() * 10f32.powi(e)).to_bits().to_le_bytes()
            })
            .collect();
        let base: Vec<f32> = (0..m).map(|_| r.s()).collect();
        let d_x = up(g, &f32_bytes(&x))?;
        let d_fn = up(g, &fn_bytes)?;
        let d_scale = up(g, &f32_bytes(&[0.7, 1.3, 0.9]))?;
        let d_base = up(g, &f32_bytes(&base))?;
        let mix_bytes = rows * m * 4;

        // 2026-10-01: MIX: the two kernels alone, one unsliced launch each.
        let d_ref = up(g, &vec![POISON_REF; mix_bytes])?;
        let d_new = up(g, &vec![POISON_NEW; mix_bytes])?;
        mix(g, k.hc_mix_bf16, m as u32, d_x, d_fn, d_ref, rows, hidden, hc)?;
        mix(g, k.hc_mix_bf16_tokmajor, 1, d_x, d_fn, d_new, rows, hidden, hc)?;
        let (a, b) = (down(g, d_ref, mix_bytes)?, down(g, d_new, mix_bytes)?);
        compared += mix_bytes;
        match first_diff(&a, &b) {
            None => println!(
                "MIX rows={rows:<5} H={hidden:<5} hc={hc} mix_hc={m:<2} bytes={mix_bytes:<7} \
                 BYTE-IDENTICAL"
            ),
            Some(i) => {
                failed += 1;
                println!(
                    "MIX rows={rows:<5} H={hidden:<5} hc={hc} mix_hc={m:<2} DIFFERS at byte {i} \
                     (row {}, mix {})",
                    i / (m * 4),
                    (i / 4) % m
                );
            }
        }

        // 2026-10-01: Negative control on the 7-row production case: one flipped mantissa bit
        // (bit 20) of token 2's element 3, a ~1e4 value, must change the token-major mix.
        if (rows, hidden, hc) == (7, 4096, 4) {
            let mut xp = f32_bytes(&x);
            xp[4 * (2 * hc_dim + 3) + 2] ^= 0x10;
            let d_xp = up(g, &xp)?;
            g.copy_h2d(&vec![POISON_NEW; mix_bytes], d_new)?;
            mix(g, k.hc_mix_bf16_tokmajor, 1, d_xp, d_fn, d_new, rows, hidden, hc)?;
            let c = down(g, d_new, mix_bytes)?;
            control_fired = Some(c != a);
            g.free(d_xp).ok();
        }

        // 2026-10-01: PRE: mix + hc_finish through the launcher, both choices.
        let w = Glm5NextMhcSiteWeights {
            hc_fn: d_fn,
            hc_fn_bf16: true,
            hc_scale: d_scale,
            hc_base: d_base,
            mix: DevicePtr::NULL,
        };
        let (o_ref, o_new) = (Outs::new(g, rows, hidden, hc)?, Outs::new(g, rows, hidden, hc)?);
        o_ref.poison(g, POISON_REF)?;
        o_new.poison(g, POISON_NEW)?;
        pre(g, &k, &w, d_x, &o_ref, rows, hidden, hc, false)?;
        pre(g, &k, &w, d_x, &o_new, rows, hidden, hc, true)?;
        let (ra, rb) = (o_ref.read(g)?, o_new.read(g)?);
        let mut bad = Vec::new();
        for ((a, b), name) in ra.iter().zip(&rb).zip(["y", "post", "comb", "mix"]) {
            compared += a.len();
            if let Some(i) = first_diff(a, b) {
                bad.push(format!("{name}@byte{i}"));
            }
        }
        if bad.is_empty() {
            println!(
                "PRE rows={rows:<5} H={hidden:<5} hc={hc} y/post/comb/mix BYTE-IDENTICAL \
                 (tokmajor vs per-(token, row) mix)"
            );
        } else {
            failed += 1;
            println!("PRE rows={rows:<5} H={hidden:<5} hc={hc} DIFFERS: {}", bad.join(", "));
        }

        // 2026-10-01: Timing at the production geometry.
        if (hidden, hc) == (4096, 4) {
            let t_ref = |fl| {
                time_ms(g, reps, fl, || {
                    mix(g, k.hc_mix_bf16, m as u32, d_x, d_fn, d_ref, rows, hidden, hc)
                })
            };
            let t_new = |fl| {
                time_ms(g, reps, fl, || {
                    mix(g, k.hc_mix_bf16_tokmajor, 1, d_x, d_fn, d_new, rows, hidden, hc)
                })
            };
            let (rh, nh) = (t_ref(None)?, t_new(None)?);
            let (rc, nc) = (t_ref(Some(flush))?, t_new(Some(flush))?);
            println!(
                "TIMING rows={rows:<5} mix ms/launch  hot REF {rh:8.4} NEW {nh:8.4} (x{:.2}) | \
                 cold REF {rc:8.4} NEW {nc:8.4} (x{:.2})",
                rh / nh,
                rc / nc
            );
        }

        o_ref.free(g);
        o_new.free(g);
        for p in [d_x, d_fn, d_scale, d_base, d_ref, d_new] {
            g.free(p).ok();
        }
    }
    g.free(flush).ok();

    println!();
    let fired = control_fired.unwrap_or(false);
    println!("CONTROL 1-bit highway flip detected={fired}");
    if compared == 0 {
        println!("FAIL - no byte was compared; this run proves nothing.");
        std::process::exit(1);
    }
    if !fired {
        println!("FAIL - the negative control did not fire; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if failed > 0 {
        println!(
            "FAIL - {failed} comparison(s) differ. METRALE_GLM_MHC_TOKMAJOR is NOT byte-identical \
             to glm5next_hc_mix_bf16 on this build; keep it off."
        );
        std::process::exit(1);
    }
    println!(
        "PASS - {compared} bytes byte-identical: glm5next_hc_mix_bf16_tokmajor matches \
         glm5next_hc_mix_bf16 (MIX) and glm_hc_pre with it matches glm_hc_pre without (PRE) in \
         every case."
    );
    Ok(())
}
