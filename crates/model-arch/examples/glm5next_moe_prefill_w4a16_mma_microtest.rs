// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: GPU gate + timing for `METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1`
//! (`forward_prefill_gemm/w4a16_mma.rs`, `kernels/gb10/common/moe_w4a16_prefill_mma.cu`) against
//! the current production MoE prefill path at the real GLM-5.3 EP=2 shapes.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - Shapes: hidden 4096, moe_intermediate 2048, 288 experts, top_k 8, EP=2 (experts 0..144
//!   local, rest NULL), each local expert its own device weights (4 host templates). Routing:
//!   distinct top-8 per row from a Zipf-like (s = 0.9) popularity; rows in {4096, 8192}.
//! - Reference = production: `moe_sort_by_expert`, `_bt_m128_k64_va2` gate and up,
//!   `glm5next_swiglu_clamp`, `_va2` down. New arm per tile (m128, m64): tile list, fused
//!   gate/up/SwiGLU, down, from the same sort output.
//! - Over LOCAL sorted rows: activation, down output, and down alone on the reference activation.
//!   PASS: cosine > 0.9995 everywhere, no non-finite where the reference is finite, device tile
//!   count == host count. TIMING lines are reported, not gated.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_moe_prefill_w4a16_mma_microtest \
//!       --features cuda,gpu-examples
//!   (GLM_W4A16_MMA_ITERS overrides the 10 timed iterations.)

use std::time::Instant;

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const H: usize = 4096;
const MI: usize = 2048;
const E: usize = 288;
const TOP_K: usize = 8;
const LOCAL: usize = 144;
const LIMIT: f32 = 10.0;
const ROW_COUNTS: [usize; 2] = [4096, 8192];
const COS_MIN: f64 = 0.9995;
const VA2: &str = "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va2";

/// 2026-10-01: (suffix, MT, STAGES) of `w4a16_mma::PREFILL_MMA_M128` / `_M64` (crate-private
/// there); threads = MT / 64 * 128, dynamic smem = STAGES * (MT * 128 + 4608).
const TILES: [(&str, usize, u32); 2] = [("m128", 128, 4), ("m64", 64, 3)];

struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.bits() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-01: One projection's host bytes: packed `[n, k/2]`, E4M3 scales `[n, k/16]` with codes
/// 0x30..=0x47 (0.5..=3.75, no zero or NaN), as the other grouped-GEMM microtests build them.
fn make_proj(r: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut packed = vec![0u8; n * k / 2];
    for c in packed.chunks_mut(8) {
        let w = r.bits().to_le_bytes();
        c.copy_from_slice(&w[..c.len()]);
    }
    let scale = (0..n * k / 16).map(|_| 0x30 + (r.bits() % 0x18) as u8).collect();
    (packed, scale)
}

/// 2026-10-01: A pointer table over all `E` experts; the first `LOCAL` get their own device copy
/// of template `e % 4`, the rest a NULL pointer.
struct Table {
    packed: DevicePtr,
    scale: DevicePtr,
    s2: DevicePtr,
}

fn build_table(
    g: &dyn GpuBackend,
    r: &mut Rng,
    n: usize,
    k: usize,
    owned: &mut Vec<DevicePtr>,
) -> Result<Table> {
    let tpl: Vec<(Vec<u8>, Vec<u8>)> = (0..4).map(|_| make_proj(r, n, k)).collect();
    let (mut pp, mut sp, mut s2) = (vec![0u64; E], vec![0u64; E], vec![0f32; E]);
    for e in 0..LOCAL {
        let (p, s) = (up(g, &tpl[e % 4].0)?, up(g, &tpl[e % 4].1)?);
        owned.extend([p, s]);
        pp[e] = p.0;
        sp[e] = s.0;
        // 2026-10-01: ~1/256 keeps gate/up near unit scale, so the SwiGLU clamp is exercised
        // without saturating every value.
        s2[e] = (0.5 + (r.bits() % 64) as f32 / 64.0) / 256.0;
    }
    let t = Table {
        packed: up(g, &le(&pp, u64::to_le_bytes))?,
        scale: up(g, &le(&sp, u64::to_le_bytes))?,
        s2: up(g, &le(&s2, f32::to_le_bytes))?,
    };
    owned.extend([t.packed, t.scale, t.s2]);
    Ok(t)
}

/// 2026-10-01: Distinct top-8 ids per row from a skewed popularity: expert `e` has weight
/// 1 / (rank[e] + 1)^0.9, `rank` a random permutation.
fn make_routing(r: &mut Rng, rows: usize) -> Vec<i32> {
    let mut rank: Vec<usize> = (0..E).collect();
    for i in (1..E).rev() {
        rank.swap(i, (r.bits() % (i as u64 + 1)) as usize);
    }
    let mut acc = 0.0;
    let mut cum = Vec::with_capacity(E);
    for &k in &rank {
        acc += 1.0 / ((k + 1) as f64).powf(0.9);
        cum.push(acc);
    }
    for c in cum.iter_mut() {
        *c /= acc;
    }
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let mut picked: Vec<i32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let u = r.unit();
            let e = cum.partition_point(|&x| x < u).min(E - 1) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids
}

fn bf(b: &[u8], i: usize) -> f64 {
    bf16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])).to_f32() as f64
}

/// 2026-10-01: (cosine, max abs err, max |ref|, non-finite count where ref is finite) over the
/// rows `ranges` of two `[rows, width]` BF16 buffers.
fn compare(want: &[u8], got: &[u8], ranges: &[(usize, usize)], width: usize) -> [f64; 4] {
    let (mut dot, mut nw, mut ng) = (0.0, 0.0, 0.0);
    let (mut max_err, mut max_ref, mut bad) = (0.0f64, 0.0f64, 0usize);
    for &(lo, hi) in ranges {
        for i in lo * width..hi * width {
            let (w, g) = (bf(want, i), bf(got, i));
            if !w.is_finite() {
                continue;
            }
            if !g.is_finite() {
                bad += 1;
                continue;
            }
            dot += w * g;
            nw += w * w;
            ng += g * g;
            max_err = max_err.max((w - g).abs());
            max_ref = max_ref.max(w.abs());
        }
    }
    [dot / (nw.sqrt() * ng.sqrt()).max(1e-300), max_err, max_ref, bad as f64]
}

#[allow(clippy::too_many_arguments)]
fn va2(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &Table,
    c: DevicePtr,
    off: DevicePtr,
    stid: DevicePtr,
    n: usize,
    kk: usize,
    m_tiles: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(n / 64) as u32, m_tiles, E as u32])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed)
        .arg_ptr(t.scale)
        .arg_ptr(t.s2)
        .arg_ptr(c)
        .arg_ptr(off)
        .arg_ptr(stid)
        .arg_u32(E as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(0)
}

/// 2026-10-01: The production ops in `w4a16_mma::forward`'s order and arguments.
struct NewArm<'a> {
    g: &'a dyn GpuBackend,
    m_tile: usize,
    threads: u32,
    smem: u32,
    k_list: KernelHandle,
    k_gu: KernelHandle,
    k_dn: KernelHandle,
    off: DevicePtr,
    tiles: DevicePtr,
    cap: u32,
}

impl NewArm<'_> {
    fn list(&self, gate: &Table) -> Result<()> {
        KernelLaunch::new(self.g, self.k_list)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.off)
            .arg_ptr(gate.packed)
            .arg_ptr(self.tiles)
            .arg_u32(E as u32)
            .arg_u32(self.m_tile as u32)
            .arg_u32(self.cap)
            .launch(0)
    }
    fn gateup(
        &self,
        x: DevicePtr,
        stid: DevicePtr,
        gt: &Table,
        ut: &Table,
        act: DevicePtr,
    ) -> Result<()> {
        KernelLaunch::new(self.g, self.k_gu)
            .grid([(MI / 64) as u32, self.cap, 1])
            .block([self.threads, 1, 1])
            .shared_mem(self.smem)
            .arg_ptr(x)
            .arg_ptr(stid)
            .arg_ptr(gt.packed)
            .arg_ptr(gt.scale)
            .arg_ptr(gt.s2)
            .arg_ptr(ut.packed)
            .arg_ptr(ut.scale)
            .arg_ptr(ut.s2)
            .arg_ptr(act)
            .arg_ptr(self.off)
            .arg_ptr(self.tiles)
            .arg_u32(MI as u32)
            .arg_u32(H as u32)
            .arg_f32(LIMIT)
            .launch(0)
    }
    fn down(&self, act: DevicePtr, dt: &Table, out: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.g, self.k_dn)
            .grid([(H / 128) as u32, self.cap, 1])
            .block([self.threads, 1, 1])
            .shared_mem(self.smem)
            .arg_ptr(act)
            .arg_ptr(DevicePtr(0))
            .arg_ptr(dt.packed)
            .arg_ptr(dt.scale)
            .arg_ptr(dt.s2)
            .arg_ptr(out)
            .arg_ptr(self.off)
            .arg_ptr(self.tiles)
            .arg_u32(H as u32)
            .arg_u32(MI as u32)
            .launch(0)
    }
}

fn time_ms(g: &dyn GpuBackend, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    f()?;
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1.0e3 / iters as f64)
}

fn main() -> Result<()> {
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let gb = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gb;
    let iters: usize = std::env::var("GLM_W4A16_MMA_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let k_sort = g.kernel("moe", "moe_sort_by_expert")?;
    let k_va2 = g.kernel("moe_w4a16", VA2)?;
    let k_swiglu = g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?;
    let m = "moe_w4a16_prefill_mma";
    let k_list = g.kernel("moe_w4a16_prefill_mma", "moe_w4a16_prefill_tile_list")?;

    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let mut owned = Vec::new();
    let gate = build_table(g, &mut r, MI, H, &mut owned)?;
    let upt = build_table(g, &mut r, MI, H, &mut owned)?;
    let down = build_table(g, &mut r, H, MI, &mut owned)?;
    let mut failures = 0usize;

    for rows in ROW_COUNTS {
        let te = rows * TOP_K;
        let x: Vec<u8> = (0..rows * H)
            .flat_map(|_| bf16::from_f64(r.unit() * 2.0 - 1.0).to_bits().to_le_bytes())
            .collect();
        let d_x = up(g, &x)?;
        let d_ids = up(g, &le(&make_routing(&mut r, rows), i32::to_le_bytes))?;
        let d_stid = g.alloc(te * 4)?;
        let d_seid = g.alloc(te * 4)?;
        let d_t2p = g.alloc(te * 4)?;
        let d_off = g.alloc((E + 1) * 4)?;
        KernelLaunch::new(g, k_sort)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_ids)
            .arg_ptr(d_stid)
            .arg_ptr(d_seid)
            .arg_ptr(d_off)
            .arg_ptr(d_t2p)
            .arg_u32(te as u32)
            .arg_u32(E as u32)
            .arg_u32(TOP_K as u32)
            .launch(0)?;
        g.synchronize(0)?;
        let off: Vec<usize> = dn(g, d_off, (E + 1) * 4)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let local: Vec<(usize, usize)> = (0..LOCAL).map(|e| (off[e], off[e + 1])).collect();
        let r_local: usize = local.iter().map(|(a, b)| b - a).sum();
        let busiest = (0..E).map(|e| off[e + 1] - off[e]).max().unwrap_or(0);
        let flop = 6.0 * r_local as f64 * (MI * H) as f64;
        println!(
            "rows={rows} te={te} local_rows={r_local} busiest_expert_rows={busiest} \
             min_local_rows={}",
            local.iter().map(|(a, b)| b - a).min().unwrap_or(0)
        );

        let a_gate = g.alloc(te * MI * 2)?;
        let a_up = g.alloc(te * MI * 2)?;
        let act_ref = g.alloc(te * MI * 2)?;
        let act_new = g.alloc(te * MI * 2)?;
        let out_ref = g.alloc(te * H * 2)?;
        let out_new = g.alloc(te * H * 2)?;
        let out_iso = g.alloc(te * H * 2)?;
        let tiles = g.alloc((te.div_ceil(64) + LOCAL + 1) * 4)?;

        // 2026-10-01: Reference arm: the production prefill path at the VA2 tile.
        let m_tiles = busiest.div_ceil(128).max(1) as u32;
        let ref_gate = || va2(g, k_va2, d_x, &gate, a_gate, d_off, d_stid, MI, H, m_tiles);
        let ref_up = || va2(g, k_va2, d_x, &upt, a_up, d_off, d_stid, MI, H, m_tiles);
        let ref_act = || {
            KernelLaunch::new(g, k_swiglu)
                .grid([((te * MI) as u32).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_gate)
                .arg_ptr(a_up)
                .arg_ptr(act_ref)
                .arg_u32((te * MI) as u32)
                .arg_f32(LIMIT)
                .launch(0)
        };
        let ref_down =
            || va2(g, k_va2, act_ref, &down, out_ref, d_off, DevicePtr(0), H, MI, m_tiles);
        g.memset_async(out_ref, 0, te * H * 2, 0)?;
        ref_gate()?;
        ref_up()?;
        ref_act()?;
        ref_down()?;
        g.synchronize(0)?;
        let want_act = dn(g, act_ref, te * MI * 2)?;
        let want_out = dn(g, out_ref, te * H * 2)?;
        let t_ref = time_ms(g, iters, || {
            ref_gate()?;
            ref_up()?;
            ref_act()?;
            ref_down()
        })?;
        let t_ref_gu = time_ms(g, iters, || {
            ref_gate()?;
            ref_up()?;
            ref_act()
        })?;
        let t_ref_dn = time_ms(g, iters, ref_down)?;
        println!(
            "TIMING rows={rows} arm=va2_production total_ms={t_ref:.3} \
             gateup_swiglu_ms={t_ref_gu:.3} down_ms={t_ref_dn:.3} TFLOP/s={:.1}",
            flop / (t_ref * 1e9)
        );

        for (suffix, m_tile, stages) in TILES {
            let gu_name = format!("moe_w4a16_prefill_mma_gateup_silu_{suffix}");
            let dn_name = format!("moe_w4a16_prefill_mma_down_{suffix}");
            let arm = NewArm {
                g,
                m_tile,
                threads: (m_tile / 64 * 128) as u32,
                smem: stages * (m_tile as u32 * 128 + 4608),
                k_list,
                k_gu: g.kernel(m, &gu_name)?,
                k_dn: g.kernel(m, &dn_name)?,
                off: d_off,
                tiles,
                cap: (te.div_ceil(m_tile) + LOCAL) as u32,
            };
            g.memset_async(act_new, 0, te * MI * 2, 0)?;
            g.memset_async(out_new, 0, te * H * 2, 0)?;
            g.memset_async(out_iso, 0, te * H * 2, 0)?;
            arm.list(&gate)?;
            arm.gateup(d_x, d_stid, &gate, &upt, act_new)?;
            arm.down(act_new, &down, out_new)?;
            arm.down(act_ref, &down, out_iso)?;
            g.synchronize(0)?;
            let dev_count = u32::from_le_bytes(dn(g, tiles, 4)?[..4].try_into()?) as usize;
            let host_count: usize = local
                .iter()
                .map(|(a, b)| (b - a).div_ceil(m_tile))
                .sum();
            let got_act = dn(g, act_new, te * MI * 2)?;
            let got_out = dn(g, out_new, te * H * 2)?;
            let got_iso = dn(g, out_iso, te * H * 2)?;
            let mut ok = dev_count == host_count;
            for (what, want, got, width) in [
                ("act", &want_act, &got_act, MI),
                ("out", &want_out, &got_out, H),
                ("down_only", &want_out, &got_iso, H),
            ] {
                let [cos, err, max_ref, bad] = compare(want, got, &local, width);
                let pass = cos > COS_MIN && bad == 0.0;
                ok &= pass;
                println!(
                    "  tile={suffix} rows={rows} {what:<9} cosine={cos:.7} max_abs_err={err:.4e} \
                     max_ref={max_ref:.4e} rel={:.3e} nonfinite={bad} {}",
                    err / max_ref.max(1e-30),
                    if pass { "ok" } else { "FAIL" }
                );
            }
            println!("  tile={suffix} rows={rows} tiles device={dev_count} host={host_count}");
            let t_new = time_ms(g, iters, || {
                arm.list(&gate)?;
                arm.gateup(d_x, d_stid, &gate, &upt, act_new)?;
                arm.down(act_new, &down, out_new)
            })?;
            let t_gu = time_ms(g, iters, || arm.gateup(d_x, d_stid, &gate, &upt, act_new))?;
            let t_dn = time_ms(g, iters, || arm.down(act_new, &down, out_new))?;
            println!(
                "TIMING rows={rows} arm=w4a16_mma_{suffix} total_ms={t_new:.3} \
                 gateup_swiglu_ms={t_gu:.3} down_ms={t_dn:.3} TFLOP/s={:.1} \
                 speedup_vs_va2={:.3}x saved_ms_per_layer={:.3}",
                flop / (t_new * 1e9),
                t_ref / t_new,
                t_ref - t_new
            );
            if !ok {
                failures += 1;
            }
        }
        for p in [d_x, d_ids, d_stid, d_seid, d_t2p, d_off, a_gate, a_up, act_ref, act_new] {
            g.free(p)?;
        }
        for p in [out_ref, out_new, out_iso, tiles] {
            g.free(p)?;
        }
    }
    for p in owned {
        g.free(p)?;
    }
    if failures > 0 {
        println!("FAIL: {failures} (rows, tile) combination(s) failed");
        bail!("glm5next_moe_prefill_w4a16_mma_microtest FAILED");
    }
    println!("PASS");
    Ok(())
}
