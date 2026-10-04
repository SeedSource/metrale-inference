// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Gate for `METRALE_GLM_MTP_CTX_ROWBATCH=1` (`glm5next_mtp_head.rs` `rows_impl`,
//! `glm5next_dsa/layer/ctx_rows.rs`): the MTP drafter's context rows built in 256-row tiles
//! against the per-row walk, at the real GLM-5.3 MTP shapes (hidden 4096, eh_proj
//! 4096 x 8192, kv_lora 512, indexer wk/compress_gate 128 x 4096) with random weights, for
//! R in {1, 7, 256, 2048, 2500}.
//!
//! Per-row arm = `rows_impl` as written: per row `rms_norm_vanilla` x2 (1 block), `dense_gemv_bf16`
//! eh_proj, `input_norm`, `dense_gemv_bf16` kv_a, a one-block `glm5next_mla_latent_write_fp8`
//! with a one-entry slot buffer, `wk` GEMV + one-block `k_norm`, `compress_gate` GEMV.
//! Batched arm = the tile pipeline: gather, rows-wide norms, 2D copy into the concat, eh_proj
//! as one cuBLASLt GEMM (> 16 rows; else the GEMV), rows-wide `input_norm`, batched GEMVs, the
//! k-block latent write, rows-wide `k_norm`.
//!
//! Legs per R:
//!   A  eh_proj output `x` (the only reduction-order change): per-row cosine and
//!      max|d| / row absmax. Tolerance: cosine >= 0.99999 and max|d|/absmax <= 0.02 (bf16 has
//!      8 mantissa bits, 2^-8 = 0.0039 per rounding; GEMM and GEMV both accumulate fp32 over
//!      K = 8192 and differ by summation order, so a couple of bf16 roundings is the expected
//!      worst case). R <= 16 uses the GEMV and must be byte-identical.
//!   B  DSA write tail from the SAME `x`: written FP8 latent slots, `k_normed` rows and
//!      `gate` rows byte-identical (no change expected: the tail uses the batched GEMV twins).
//!   C  end to end (batched `x` into the batched tail vs per-row everything): fraction of
//!      latent bytes that differ, `k_normed`/`gate` cosine and max|d|/absmax with the same
//!      tolerance as A (informational for the FP8 bytes; FP8 re-quantisation of a 1-ulp
//!      different input flips some bytes).
//!
//! Owner: model-arch examples. Invariants: none beyond the types.
//! Last line: `PASS: ...` or `FAIL: ...`. Exit 0 / 1, 2 when a kernel is absent.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_mtp_ctx_rowbatch_microtest

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

const H: usize = 4096;
const KVR: usize = 512;
const D: usize = 128;
const VOCAB: usize = 3000;
const TILE: usize = 256;
const EPS: f32 = 1e-6;
const MAXM: usize = ops::DENSE_GEMV_BATCHM_MAX_M as usize;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn bf(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(lo + (hi - lo) * self.f()).to_bits().to_le_bytes())
            .collect()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

struct W {
    enorm: DevicePtr,
    hnorm: DevicePtr,
    in_norm: DevicePtr,
    eh: DevicePtr,
    kv_a: DevicePtr,
    kv_ln: DevicePtr,
    wk: DevicePtr,
    gate: DevicePtr,
    kn_w: DevicePtr,
    kn_b: DevicePtr,
    embed: DevicePtr,
    gemv: KernelHandle,
    batchm: KernelHandle,
    rms: KernelHandle,
    latent: KernelHandle,
    k_norm: KernelHandle,
}

fn rms(g: &dyn GpuBackend, w: &W, x: DevicePtr, nw: DevicePtr, out: DevicePtr, rows: usize) -> Result<()> {
    KernelLaunch::new(g, w.rms)
        .grid([rows as u32, 1, 1])
        .block([H.min(1024) as u32, 1, 1])
        .arg_ptr(x)
        .arg_ptr(nw)
        .arg_ptr(out)
        .arg_u32(H as u32)
        .arg_f32(EPS)
        .launch(0)
}

fn gemv1(g: &dyn GpuBackend, w: &W, a: DevicePtr, b: DevicePtr, c: DevicePtr, n: usize, k: usize) -> Result<()> {
    ops::dense_gemv(g, w.gemv, a, &DenseWeight { weight: b }, c, n as u32, k as u32, 0)
}

fn batchm(g: &dyn GpuBackend, w: &W, a: DevicePtr, b: DevicePtr, c: DevicePtr, rows: usize, n: usize, k: usize) -> Result<()> {
    let wt = DenseWeight { weight: b };
    let mut r0 = 0;
    while r0 < rows {
        let m = MAXM.min(rows - r0);
        ops::dense_gemv_batchm(g, w.batchm, a.offset(r0 * k * 2), &wt, c.offset(r0 * n * 2), m as u32, n as u32, k as u32, n as u32, 0)?;
        r0 += m;
    }
    Ok(())
}

fn latent(g: &dyn GpuBackend, w: &W, kv_a: DevicePtr, cache: DevicePtr, slots: DevicePtr, blocks: usize) -> Result<()> {
    KernelLaunch::new(g, w.latent)
        .grid([blocks as u32, 1, 1])
        .block([KVR as u32, 1, 1])
        .arg_ptr(kv_a)
        .arg_ptr(w.kv_ln)
        .arg_ptr(cache)
        .arg_ptr(slots)
        .arg_u32(KVR as u32)
        .arg_f32(EPS)
        .arg_f32(1.0)
        .launch(0)
}

fn k_norm(g: &dyn GpuBackend, w: &W, x: DevicePtr, blocks: usize) -> Result<()> {
    KernelLaunch::new(g, w.k_norm)
        .grid([blocks as u32, 1, 1])
        .block([D as u32, 1, 1])
        .shared_mem((D * 4) as u32)
        .arg_ptr(x)
        .arg_ptr(w.kn_w)
        .arg_ptr(w.kn_b)
        .arg_u32(blocks as u32)
        .arg_u32(D as u32)
        .arg_f32(EPS)
        .launch(0)
}

/// Per-row head: `x[r]` for all rows, as `rows_impl` computes it.
fn head_perrow(g: &dyn GpuBackend, w: &W, toks: &[u32], hid: DevicePtr, x: DevicePtr) -> Result<()> {
    let concat = g.alloc(2 * H * 2)?;
    for r in 0..toks.len() - 1 {
        let e = w.embed.offset(toks[r + 1] as usize * H * 2);
        rms(g, w, e, w.enorm, concat, 1)?;
        rms(g, w, hid.offset(r * H * 2), w.hnorm, concat.offset(H * 2), 1)?;
        gemv1(g, w, concat, w.eh, x.offset(r * H * 2), H, 2 * H)?;
    }
    g.synchronize(0)?;
    g.free(concat).ok();
    Ok(())
}

/// Tiled head, as `rows_impl`'s batched branch computes it.
fn head_batched(g: &dyn GpuBackend, w: &W, toks: &[u32], hid: DevicePtr, x: DevicePtr) -> Result<()> {
    let rows = toks.len() - 1;
    let (gath, nrm, concat) = (g.alloc(TILE * H * 2)?, g.alloc(TILE * H * 2)?, g.alloc(TILE * 2 * H * 2)?);
    let mut r0 = 0;
    while r0 < rows {
        let t = TILE.min(rows - r0);
        for i in 0..t {
            g.copy_d2d_async(w.embed.offset(toks[r0 + i + 1] as usize * H * 2), gath.offset(i * H * 2), H * 2, 0)?;
        }
        rms(g, w, gath, w.enorm, nrm, t)?;
        g.copy_d2d_2d_async(nrm, H * 2, concat, 2 * H * 2, H * 2, t, 0)?;
        rms(g, w, hid.offset(r0 * H * 2), w.hnorm, nrm, t)?;
        g.copy_d2d_2d_async(nrm, H * 2, concat.offset(H * 2), 2 * H * 2, H * 2, t, 0)?;
        let xo = x.offset(r0 * H * 2);
        if t > MAXM {
            ops::cublas_bf16_proj_dense(concat, w.eh, xo, t as u32, H as u32, (2 * H) as u32, 0)?;
        } else {
            for i in 0..t {
                gemv1(g, w, concat.offset(i * 2 * H * 2), w.eh, xo.offset(i * H * 2), H, 2 * H)?;
            }
        }
        g.synchronize(0)?;
        r0 += t;
    }
    for p in [gath, nrm, concat] {
        g.free(p).ok();
    }
    Ok(())
}

struct Out {
    cache: Vec<u8>,
    k: Vec<u8>,
    gate: Vec<u8>,
}

fn tail_perrow(g: &dyn GpuBackend, w: &W, x: DevicePtr, rows: usize) -> Result<Out> {
    let (nm, kva, one) = (g.alloc(H * 2)?, g.alloc(KVR * 2)?, g.alloc(8)?);
    let cache = g.alloc(rows * KVR)?;
    g.memset(cache, 0xA5, rows * KVR)?;
    let (k, gate) = (g.alloc(rows * D * 2)?, g.alloc(rows * D * 2)?);
    for r in 0..rows {
        rms(g, w, x.offset(r * H * 2), w.in_norm, nm, 1)?;
        gemv1(g, w, nm, w.kv_a, kva, KVR, H)?;
        g.synchronize(0)?;
        g.copy_h2d(&(r as i64).to_le_bytes(), one)?;
        latent(g, w, kva, cache, one, 1)?;
        gemv1(g, w, nm, w.wk, k.offset(r * D * 2), D, H)?;
        k_norm_one(g, w, k.offset(r * D * 2))?;
        gemv1(g, w, nm, w.gate, gate.offset(r * D * 2), D, H)?;
    }
    let o = Out { cache: down(g, cache, rows * KVR)?, k: down(g, k, rows * D * 2)?, gate: down(g, gate, rows * D * 2)? };
    for p in [nm, kva, one, cache, k, gate] {
        g.free(p).ok();
    }
    Ok(o)
}

fn k_norm_one(g: &dyn GpuBackend, w: &W, x: DevicePtr) -> Result<()> {
    KernelLaunch::new(g, w.k_norm)
        .grid([1, 1, 1])
        .block([D as u32, 1, 1])
        .shared_mem((D * 4) as u32)
        .arg_ptr(x)
        .arg_ptr(w.kn_w)
        .arg_ptr(w.kn_b)
        .arg_u32(1)
        .arg_u32(D as u32)
        .arg_f32(EPS)
        .launch(0)
}

fn tail_batched(g: &dyn GpuBackend, w: &W, x: DevicePtr, rows: usize) -> Result<Out> {
    let (nm, kva, slots) = (g.alloc(TILE * H * 2)?, g.alloc(TILE * KVR * 2)?, g.alloc(TILE * 8)?);
    let cache = g.alloc(rows * KVR)?;
    g.memset(cache, 0xA5, rows * KVR)?;
    let (k, gate) = (g.alloc(rows * D * 2)?, g.alloc(rows * D * 2)?);
    let mut r0 = 0;
    while r0 < rows {
        let t = TILE.min(rows - r0);
        rms(g, w, x.offset(r0 * H * 2), w.in_norm, nm, t)?;
        batchm(g, w, nm, w.kv_a, kva, t, KVR, H)?;
        let host: Vec<u8> = (0..t).flat_map(|i| ((r0 + i) as i64).to_le_bytes()).collect();
        g.synchronize(0)?;
        g.copy_h2d(&host, slots)?;
        latent(g, w, kva, cache, slots, t)?;
        let kr = k.offset(r0 * D * 2);
        batchm(g, w, nm, w.wk, kr, t, D, H)?;
        k_norm(g, w, kr, t)?;
        batchm(g, w, nm, w.gate, gate.offset(r0 * D * 2), t, D, H)?;
        r0 += t;
    }
    let o = Out { cache: down(g, cache, rows * KVR)?, k: down(g, k, rows * D * 2)?, gate: down(g, gate, rows * D * 2)? };
    for p in [nm, kva, slots, cache, k, gate] {
        g.free(p).ok();
    }
    Ok(o)
}

/// (min cosine, max |d| / row absmax) over `[rows, width]` BF16 byte buffers.
fn stats(a: &[u8], b: &[u8], width: usize) -> (f64, f64) {
    let (fa, fb) = (f32s(a), f32s(b));
    let (mut mc, mut me) = (1.0f64, 0.0f64);
    for (ra, rb) in fa.chunks(width).zip(fb.chunks(width)) {
        let (mut dot, mut na, mut nb, mut mx, mut md) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&x, &y) in ra.iter().zip(rb) {
            let (x, y) = (x as f64, y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            mx = mx.max(x.abs());
            md = md.max((x - y).abs());
        }
        mc = mc.min(dot / (na.sqrt() * nb.sqrt()).max(1e-30));
        me = me.max(md / mx.max(1e-30));
    }
    (mc, me)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kn = |m: &str, f: &str| -> KernelHandle {
        match g.kernel(m, f) {
            Ok(h) => h,
            Err(e) => {
                println!("{m}::{f} absent ({e})");
                println!("FAIL: kernel absent from this target (set METRALE_TARGET_* as in the header)");
                std::process::exit(2);
            }
        }
    };
    let mut rng = Lcg(0x5EED_0003);
    let wbf = |rng: &mut Lcg, n: usize, s: f32| up(g, &rng.bf(n, -s, s));
    let w = W {
        enorm: up(g, &rng.bf(H, 0.5, 1.5))?,
        hnorm: up(g, &rng.bf(H, 0.5, 1.5))?,
        in_norm: up(g, &rng.bf(H, 0.5, 1.5))?,
        eh: wbf(&mut rng, H * 2 * H, 0.03)?,
        kv_a: wbf(&mut rng, KVR * H, 0.08)?,
        kv_ln: up(g, &rng.bf(KVR, 0.5, 1.5))?,
        wk: wbf(&mut rng, D * H, 0.08)?,
        gate: wbf(&mut rng, D * H, 0.08)?,
        kn_w: up(g, &rng.bf(D, 0.5, 1.5))?,
        kn_b: up(g, &rng.bf(D, -0.2, 0.2))?,
        embed: wbf(&mut rng, VOCAB * H, 1.0)?,
        gemv: kn("gemv", "dense_gemv_bf16"),
        batchm: kn("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        rms: kn("rms_norm_vanilla", "rms_norm_vanilla"),
        latent: kn("glm5next_mla_latent_write", "glm5next_mla_latent_write_fp8"),
        k_norm: kn("nllb_encoder", "nllb_layernorm_bf16"),
    };
    let (mut ok, mut cmp) = (true, 0usize);
    for &r in &[1usize, 7, 256, 2048, 2500] {
        let toks: Vec<u32> = (0..=r).map(|_| (rng.f() * (VOCAB - 1) as f32) as u32).collect();
        let hid = up(g, &rng.bf(r * H, -3.0, 3.0))?;
        let (xa, xb) = (g.alloc(r * H * 2)?, g.alloc(r * H * 2)?);
        head_perrow(g, &w, &toks, hid, xa)?;
        head_batched(g, &w, &toks, hid, xb)?;
        let (ba, bb) = (down(g, xa, r * H * 2)?, down(g, xb, r * H * 2)?);
        let (cos, rel) = stats(&ba, &bb, H);
        let ident = ba == bb;
        let a_ok = if r <= MAXM { ident } else { cos >= 0.99999 && rel <= 0.02 };
        println!("R={r:<5} A  eh_proj x: byte-identical={ident} min_cos={cos:.8} max|d|/absmax={rel:.5} {}", if a_ok { "ok" } else { "BAD" });
        let ref_a = tail_perrow(g, &w, xa, r)?;
        let bat_a = tail_batched(g, &w, xa, r)?;
        let b_ok = ref_a.cache == bat_a.cache && ref_a.k == bat_a.k && ref_a.gate == bat_a.gate;
        println!("R={r:<5} B  tail from same x: latent={} k_normed={} gate={} {}", ref_a.cache == bat_a.cache, ref_a.k == bat_a.k, ref_a.gate == bat_a.gate, if b_ok { "ok" } else { "BAD" });
        let bat_b = tail_batched(g, &w, xb, r)?;
        let dcache = ref_a.cache.iter().zip(&bat_b.cache).filter(|(p, q)| p != q).count();
        let (kc, ke) = stats(&ref_a.k, &bat_b.k, D);
        let (gc, ge) = stats(&ref_a.gate, &bat_b.gate, D);
        let c_ok = r <= MAXM || (kc >= 0.9999 && ke <= 0.05 && gc >= 0.9999 && ge <= 0.05);
        println!("R={r:<5} C  end to end: latent bytes differing={dcache}/{} ({:.4}%) k_normed cos={kc:.6} rel={ke:.4} gate cos={gc:.6} rel={ge:.4} {}", r * KVR, 100.0 * dcache as f64 / (r * KVR) as f64, if c_ok { "ok" } else { "BAD" });
        ok &= a_ok && b_ok && c_ok;
        cmp += r;
        for p in [hid, xa, xb] {
            g.free(p).ok();
        }
    }
    if cmp == 0 {
        println!("FAIL: nothing compared");
        std::process::exit(1);
    }
    if ok {
        println!("PASS: batched MTP context rows match the per-row path (eh_proj within bf16 reduction order, DSA write tail byte-identical) for R in 1,7,256,2048,2500");
        Ok(())
    } else {
        println!("FAIL: batched MTP context rows differ from the per-row path beyond tolerance (see BAD lines)");
        std::process::exit(1);
    }
}
