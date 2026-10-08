// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Bitwise gate and timing for `METRALE_GLM_HC_POST_MIX_FAST`:
//! `glm_hc_post_mix_finish_fast` (`glm5next_hc_post_mix_y_bf16` + `glm5next_hc_comb_from_mix`)
//! against `glm_hc_post_mix_finish_base` (`glm5next_hc_post_mix_finish_bf16`, the kernel the car
//! runs under `METRALE_GLM_MHC_POST_MIX_FINISH=1`).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - Both arms alias their buffers as `SpRows::back` does: `y` over `block_out`, `post` / `comb`
//!   over the input `post` / `comb`, the highway in place. A third run writes the new path's
//!   `y` / `post` / `comb` to separate buffers.
//! - Exits 0 only if, at every row count, both new-path runs equal the base bit for bit on every
//!   output (highway and mix as u32, `y` as u16, post and comb as u32), and both known-bads are
//!   detected: the input `comb` nudged by one ULP (highway, mix and comb must differ) and the
//!   next site's `hc_base` nudged (post, comb and y must differ, highway and mix must not).
//! - The lever environment does not matter: both paths are called explicitly.
//!
//! Shapes: GLM-5.3 (hidden 4096, hc_mult 4, mix_hc 24, BF16 `hc_fn`, 20 Sinkhorn iterations) at
//! 1, 127, 257, 4095, 4096 (the 32K car: 8192-row windows, half per rank), 4097, 8191 and 8192
//! rows. Timing per shape, in place as the car runs it: a 64 MiB memset before every launch and
//! a rotation over enough input sets (at least 2, about 96 MiB) that L2 cannot hold them; median
//! of 9. One line `HCPM_SPEED rows=<n> old_ms=<x> new_ms=<y> ratio=<new/old>` per shape plus the
//! effective GB/s on the minimum bytes (147,712 B per row + 786,432 B of `hc_fn`), and from 4095
//! rows a device-to-device copy of the same bytes (`HCPM_CONTROL`): a read/write-mix control,
//! since the 237-240 GB/s STREAM figure is a read.
//!
//! Run: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples --example
//! hc_post_mix_fast_bitparity_microtest`

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_post_mix_finish_base,
    glm_hc_post_mix_finish_fast, post_mix_fast_usable,
};

const H: usize = 4096;
const HC: usize = 4;
const MIX: usize = 24;
const SINKHORN: u32 = 20;
const NORM_EPS: f32 = 1e-5;
const HC_EPS: f32 = 1e-6;
const FLUSH_BYTES: usize = 64 << 20;
const ROTATE_BYTES: usize = 96 << 20;
const REPS: usize = 9;
/// 2026-10-08: Minimum bytes per row: block_out 8,192 R, residual 65,536 R, out 65,536 W,
/// y 8,192 W, post / comb in 80 R, mix 96 W, post / comb out 80 W.
const ROW_BYTES: usize = 147_712;
const HC_FN_BYTES: usize = MIX * HC * H * 2;
const ROWS: [usize; 8] = [1, 127, 257, 4095, 4096, 4097, 8191, 8192];

// 2026-10-08: CUDA driver event API for timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32) as f32 / (1u32 << 24) as f32
    }
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16s(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect()
}

fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut v = vec![0u8; bytes];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

/// 2026-10-08: Elements of `w` bytes whose bit patterns differ, and the first one as
/// (index, want bits, got bits).
fn ndiff(a: &[u8], b: &[u8], w: usize) -> (usize, Option<(usize, u32, u32)>) {
    let bits = |c: &[u8]| {
        if w == 2 {
            u16::from_le_bytes([c[0], c[1]]) as u32
        } else {
            u32::from_le_bytes([c[0], c[1], c[2], c[3]])
        }
    };
    let mut n = 0;
    let mut first = None;
    for (i, (x, y)) in a.chunks_exact(w).zip(b.chunks_exact(w)).enumerate() {
        let (bx, by) = (bits(x), bits(y));
        if bx != by {
            n += 1;
            first.get_or_insert((i, bx, by));
        }
    }
    (n, first)
}

/// 2026-10-08: Host copies of one row count's inputs.
struct Host {
    t: usize,
    bo: Vec<u8>,
    res: Vec<u8>,
    post: Vec<u8>,
    comb: Vec<u8>,
    comb_bad: Vec<u8>,
}

/// 2026-10-08: Device buffers of one row count: the aliased in/out set the back uses, plus
/// separate y / post / comb outputs.
struct Io {
    bo: DevicePtr,
    hw: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    mix: DevicePtr,
    y2: DevicePtr,
    post2: DevicePtr,
    comb2: DevicePtr,
}

impl Io {
    fn alloc(g: &dyn GpuBackend, t: usize, sep: bool) -> Result<Self> {
        let (y2, post2, comb2) = if sep {
            (
                g.alloc(t * H * 2)?,
                g.alloc(t * HC * 4)?,
                g.alloc(t * HC * HC * 4)?,
            )
        } else {
            (DevicePtr(0), DevicePtr(0), DevicePtr(0))
        };
        Ok(Self {
            bo: g.alloc(t * H * 2)?,
            hw: g.alloc(t * HC * H * 4)?,
            post: g.alloc(t * HC * 4)?,
            comb: g.alloc(t * HC * HC * 4)?,
            mix: g.alloc(t * MIX * 4)?,
            y2,
            post2,
            comb2,
        })
    }

    fn free(&self, g: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.bo, self.hw, self.post, self.comb, self.mix, self.y2, self.post2, self.comb2,
        ] {
            if p.0 != 0 {
                g.free(p)?;
            }
        }
        Ok(())
    }
}

/// 2026-10-08: Every output of one run: highway, mix, y, post, comb.
type Outs = [Vec<u8>; 5];

const NAMES: [&str; 5] = ["highway", "mix", "y", "post", "comb"];
const WIDTHS: [usize; 5] = [4, 4, 2, 4, 4];

fn reset(g: &dyn GpuBackend, io: &Io, h: &Host, comb: &[u8]) -> Result<()> {
    g.copy_h2d(&h.bo, io.bo)?;
    g.copy_h2d(&h.res, io.hw)?;
    g.copy_h2d(&h.post, io.post)?;
    g.copy_h2d(comb, io.comb)?;
    g.memset_async(io.mix, 0xEE, h.t * MIX * 4, 0)
}

fn read(g: &dyn GpuBackend, t: usize, io: &Io, sep: bool) -> Result<Outs> {
    let o = if sep {
        [io.y2, io.post2, io.comb2]
    } else {
        [io.bo, io.post, io.comb]
    };
    Ok([
        down(g, io.hw, t * HC * H * 4)?,
        down(g, io.mix, t * MIX * 4)?,
        down(g, o[0], t * H * 2)?,
        down(g, o[1], t * HC * 4)?,
        down(g, o[2], t * HC * HC * 4)?,
    ])
}

/// 2026-10-08: One path, aliased as the back issues it, or (`sep`) with separate y / post /
/// comb outputs.
fn run(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    io: &Io,
    next: &Glm5NextMhcSiteWeights,
    t: usize,
    fast: bool,
    sep: bool,
) -> Result<()> {
    let (y, p, c) = if sep {
        (io.y2, io.post2, io.comb2)
    } else {
        (io.bo, io.post, io.comb)
    };
    let (tt, h, hc) = (t as u32, H as u32, HC as u32);
    let f = if fast {
        glm_hc_post_mix_finish_fast
    } else {
        glm_hc_post_mix_finish_base
    };
    f(
        g, k, io.bo, io.hw, io.post, io.comb, io.hw, next, io.mix, y, p, c, tt, h, hc, SINKHORN,
        NORM_EPS, HC_EPS, 0,
    )
}

fn diffs(a: &Outs, b: &Outs) -> [(usize, Option<(usize, u32, u32)>); 5] {
    std::array::from_fn(|i| ndiff(&a[i], &b[i], WIDTHS[i]))
}

fn counts(d: &[(usize, Option<(usize, u32, u32)>); 5]) -> [usize; 5] {
    std::array::from_fn(|i| d[i].0)
}

fn show(d: &[(usize, Option<(usize, u32, u32)>); 5]) -> String {
    let v: Vec<String> = (0..5)
        .map(|i| match d[i] {
            (0, _) => format!("{} 0", NAMES[i]),
            (n, Some((e, a, b))) => format!("{} {n} (first #{e}: {a:#x} vs {b:#x})", NAMES[i]),
            (n, None) => format!("{} {n}", NAMES[i]),
        })
        .collect();
    v.join(", ")
}

/// 2026-10-08: Median of `REPS` timed calls of `f(rep)`, a 64 MiB memset before each.
fn time_cold(
    g: &dyn GpuBackend,
    flush: DevicePtr,
    mut f: impl FnMut(usize) -> Result<()>,
) -> Result<f32> {
    let mut v = Vec::new();
    for rep in 0..REPS {
        g.memset_async(flush, 0x3C, FLUSH_BYTES, 0)?;
        let (mut e0, mut e1, mut ms) = (0u64, 0u64, 0f32);
        unsafe {
            if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
                bail!("cuEventCreate returned an error");
            }
            cuEventRecord(e0, 0);
        }
        f(rep)?;
        unsafe {
            cuEventRecord(e1, 0);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        v.push(ms);
    }
    v.sort_by(f32::total_cmp);
    Ok(v[v.len() / 2])
}

fn gbps(bytes: usize, ms: f32) -> f64 {
    bytes as f64 / (ms as f64 * 1e6)
}

/// 2026-10-08: Both paths timed in place over a rotation of input sets, and (from 4095 rows) a
/// device-to-device copy moving the same bytes.
fn time_shape(
    g: &dyn GpuBackend,
    k: &Glm5NextMhcKernels,
    next: &Glm5NextMhcSiteWeights,
    h: &Host,
    flush: DevicePtr,
) -> Result<()> {
    let t = h.t;
    let set_bytes = t * (HC * H * 4 + H * 2 + MIX * 4 + HC * 4 + HC * HC * 4);
    let nset = ROTATE_BYTES.div_ceil(set_bytes).clamp(2, 16);
    let mut sets = Vec::with_capacity(nset);
    for _ in 0..nset {
        let io = Io::alloc(g, t, false)?;
        reset(g, &io, h, &h.comb)?;
        sets.push(io);
    }
    let old = time_cold(g, flush, |r| run(g, k, &sets[r % nset], next, t, false, false))?;
    let new = time_cold(g, flush, |r| run(g, k, &sets[r % nset], next, t, true, false))?;
    let bytes = t * ROW_BYTES + HC_FN_BYTES;
    println!(
        "HCPM_SPEED rows={t} old_ms={old:.4} new_ms={new:.4} ratio={:.4}",
        new / old
    );
    println!(
        "HCPM_GBPS rows={t} min_bytes={bytes} old_gbps={:.1} new_gbps={:.1} sets={nset}",
        gbps(bytes, old),
        gbps(bytes, new)
    );
    if t >= 4095 {
        // 2026-10-08: Half the bytes read, half written, as the kernel's mix (73,856 B read and
        // 73,856 B written per row).
        let half = (t * ROW_BYTES).div_ceil(2);
        let (src, dst) = (g.alloc(half)?, g.alloc(half)?);
        g.memset_async(src, 0x11, half, 0)?;
        let cp = time_cold(g, flush, |_| g.copy_d2d_async(src, dst, half, 0))?;
        println!(
            "HCPM_CONTROL rows={t} d2d_copy_bytes={} ms={cp:.4} gbps={:.1}",
            2 * half,
            gbps(2 * half, cp)
        );
        g.free(src)?;
        g.free(dst)?;
    }
    for io in &sets {
        io.free(g)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        bail!("hc_post_mix_fast_bitparity_microtest: no glm-5.3-flash PTX set in this build");
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextMhcKernels::resolve(g)?;
    if k.hc_post_mix_finish_bf16.0 == 0 || !post_mix_fast_usable(&k) {
        bail!("hc_post_mix_fast_bitparity_microtest: fused or fast-path kernel missing");
    }
    let flush = g.alloc(FLUSH_BYTES)?;
    let mut bad = 0usize;
    let mut rng = Lcg(0x5C1F);
    // 2026-10-08: The next site's weights, shared by every case, as in
    // hc_post_mix_finish_bitparity_microtest.
    let fnv: Vec<f32> = (0..MIX * HC * H).map(|_| rng.sym(0.05)).collect();
    let base: Vec<f32> = (0..MIX).map(|_| rng.sym(1.0)).collect();
    let base_bad: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, b)| b + 0.01 * (i as f32 + 1.0))
        .collect();
    let mut next = Glm5NextMhcSiteWeights {
        hc_fn: upload(g, &bf16s(&fnv))?,
        hc_fn_bf16: true,
        hc_scale: upload(g, &f32s(&[0.8, 0.6, 1.2]))?,
        hc_base: upload(g, &f32s(&base))?,
        mix: DevicePtr(0),
    };
    let good_base = next.hc_base;
    let bad_base = upload(g, &f32s(&base_bad))?;

    for t in ROWS {
        let bo: Vec<f32> = (0..t * H).map(|_| rng.sym(3.0)).collect();
        let res: Vec<f32> = (0..t * HC * H).map(|_| rng.sym(8.0)).collect();
        let post: Vec<f32> = (0..t * HC).map(|_| rng.unit() * 2.0).collect();
        let comb: Vec<f32> = (0..t * HC * HC).map(|_| rng.unit()).collect();
        let mut comb_bad = comb.clone();
        let e = (t / 2) * HC * HC + 5;
        comb_bad[e] = f32::from_bits(comb_bad[e].to_bits() + 1);
        let h = Host {
            t,
            bo: bf16s(&bo),
            res: f32s(&res),
            post: f32s(&post),
            comb: f32s(&comb),
            comb_bad: f32s(&comb_bad),
        };
        drop((bo, res, post, comb, comb_bad));
        let io = Io::alloc(g, t, true)?;

        next.hc_base = good_base;
        reset(g, &io, &h, &h.comb)?;
        run(g, &k, &io, &next, t, false, false)?;
        let want = read(g, t, &io, false)?;
        if want[2] == h.bo || want[3] == h.post {
            println!("BASE did not write y / post rows={t} (the reference is stale)");
            bad += 1;
        }

        reset(g, &io, &h, &h.comb)?;
        run(g, &k, &io, &next, t, true, false)?;
        let d_al = diffs(&want, &read(g, t, &io, false)?);
        reset(g, &io, &h, &h.comb)?;
        for (p, n) in [
            (io.y2, t * H * 2),
            (io.post2, t * HC * 4),
            (io.comb2, t * HC * HC * 4),
        ] {
            g.memset_async(p, 0xEE, n, 0)?;
        }
        run(g, &k, &io, &next, t, true, true)?;
        let d_sep = diffs(&want, &read(g, t, &io, true)?);
        println!("rows={t}: fast aliased differs: {}", show(&d_al));
        println!("rows={t}: fast separate differs: {}", show(&d_sep));
        if d_al.iter().chain(d_sep.iter()).any(|d| d.0 != 0) {
            println!("MISMATCH rows={t}");
            bad += 1;
        }

        // 2026-10-08: Known-bad 1: the input comb nudged by one ULP must change the highway and,
        // through it, the mix and the next comb (y is BF16 and usually absorbs it).
        reset(g, &io, &h, &h.comb_bad)?;
        run(g, &k, &io, &next, t, true, false)?;
        let kb1 = counts(&diffs(&want, &read(g, t, &io, false)?));
        // 2026-10-08: Known-bad 2: the next site's hc_base nudged must change post, comb and y
        // and leave the highway and the mix alone.
        next.hc_base = bad_base;
        reset(g, &io, &h, &h.comb)?;
        run(g, &k, &io, &next, t, true, false)?;
        let kb2 = counts(&diffs(&want, &read(g, t, &io, false)?));
        next.hc_base = good_base;
        println!("KNOWN_BAD rows={t}: comb+1ulp: {kb1:?} (highway, mix, y, post, comb)");
        println!("KNOWN_BAD rows={t}: hc_base nudge: {kb2:?}");
        let kb1_ok = kb1[0] != 0 && kb1[1] != 0 && kb1[4] != 0;
        let kb2_ok = kb2[0] == 0 && kb2[1] == 0 && kb2[2] != 0 && kb2[3] != 0 && kb2[4] != 0;
        if !kb1_ok || !kb2_ok {
            println!("KNOWN_BAD not detected rows={t} (the comparison is blind)");
            bad += 1;
        }
        io.free(g)?;

        time_shape(g, &k, &next, &h, flush)?;
    }
    if bad != 0 {
        bail!("hc_post_mix_fast_bitparity_microtest: {bad} check(s) did not hold");
    }
    println!(
        "PASS: glm5next_hc_post_mix_y_bf16 + glm5next_hc_comb_from_mix equal \
         glm5next_hc_post_mix_finish_bf16 bit for bit (highway, mix, y, post, comb; aliased and \
         separate outputs) at rows {ROWS:?}; both known-bads detected"
    );
    Ok(())
}
