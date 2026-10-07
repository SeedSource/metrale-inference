// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Bit-parity + timing gate for two exact W4A4 MoE prefill levers
//! (`forward_prefill_gemm/cutlass_w4a4.rs`,
//! `crates/gpu-runtime/cuda/cutlass_nvfp4_grouped_gemm.cu`):
//! `METRALE_CUTLASS_W4A4_PACK_ONCE` (gate/up A quantized once per token, then gathered) and
//! `METRALE_GLM_MOE_SWIGLU_AMAX` (the down call's dynamic amax computed by the SwiGLU).
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - Shapes: hidden 4096, 288 experts, top_k 8; `GLM_MT_MI` / `GLM_MT_LOCAL` default to the
//!   expert-TP sizing 1024 / 288 (every expert local). Dynamic global scales (the car's
//!   `METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE=1` table), SwiGLU over the local rows
//!   (`METRALE_GLM_MOE_SWIGLU_LOCAL_ROWS=1`). Token windows 17, 256, 2048, 8192.
//! - Before every compared call the whole CUTLASS workspace is filled with 0xA5, so bytes a
//!   pack does not write (padding rows of the last SFA tile, 256-byte alignment gaps) read the
//!   same sentinel in both arms and the packed-A + SFA region compares bitwise as a whole.
//! - Per window, lever off vs on, bitwise: (1) gate/up packed A + SFA of every active group and
//!   the per-group gs; (2) through `cutlass_w4a4::run`: the down call's packed A + SFA and gs
//!   bits, `a_act`, `expert_out`; (3) the SwiGLU amax slot equals the host max |a_act| over the
//!   SwiGLU rows. Each lever must report ENGAGED. Known-bads that must be detected: (a) one
//!   staged block-scale byte flipped (`set_w4a4_pack_once_fault`) changes the gate/up SFA;
//!   (b) the amax slot nudged down 16 ULPs changes the down gs bits.
//! - Ends with one line starting `PASS` when everything holds, else a `FAIL:` line and a
//!   nonzero exit. TIMING lines: gate/up call (prep + GEMMs) and the whole `run`, off vs on.
//!
//!   cargo run -p metrale-model-arch --release --example glm_moe_w4a4_pack_once_microtest \
//!       --features cuda,gpu-examples      (CUTLASS_HOME set at build time)
//!   Env: GLM_W4A4_MT_ITERS (timed iterations, 10), GLM_MT_MI, GLM_MT_LOCAL.

use std::time::Instant;

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::cutlass;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::Glm5NextExpertWeights;
use metrale_model_arch::glm5next_mlp::forward_prefill_gemm::cutlass_w4a4::{
    MoeTables, RunArgs, SfbCache, run, swiglu_rows_amax, swiglu_span,
};
use metrale_model_arch::glm5next_mlp::weights::Nvfp4Proj;

const H: usize = 4096;
const E: usize = 288;
const TOP_K: usize = 8;
const LIMIT: f32 = 10.0;
const WINDOWS: [usize; 4] = [17, 256, 2048, 8192];
const SENTINEL: u8 = 0xA5;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

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
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn le_u64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ndiff(x: &[u8], y: &[u8]) -> usize {
    x.iter().zip(y).filter(|(a, b)| a != b).count() + x.len().abs_diff(y.len())
}

/// 2026-10-06: One projection over the local experts: per-expert packed `[n, k/2]` and E4M3
/// scales `[n, k/16]` (codes 0x30..=0x47), device pointer tables over all `E` ids.
struct Proj {
    packed: Vec<u64>,
    scale: Vec<u64>,
    s2: Vec<f32>,
    scale_tab: DevicePtr,
}

fn build_proj(g: &dyn GpuBackend, r: &mut Rng, n: usize, k: usize, local: usize) -> Result<Proj> {
    let (mut packed, mut scale, mut s2) = (vec![0u64; E], vec![0u64; E], vec![0f32; E]);
    for e in 0..local {
        let p: Vec<u8> = (0..n * k / 2).map(|_| r.bits() as u8).collect();
        let s: Vec<u8> = (0..n * k / 16)
            .map(|_| 0x30 + (r.bits() % 0x18) as u8)
            .collect();
        packed[e] = up(g, &p)?.0;
        scale[e] = up(g, &s)?.0;
        s2[e] = (0.5 + (r.bits() % 64) as f32 / 64.0) / 256.0;
    }
    let scale_tab = up(g, &le_u64(&scale))?;
    Ok(Proj {
        packed,
        scale,
        s2,
        scale_tab,
    })
}

/// 2026-10-06: Distinct top-8 ids per token from a Zipf-like (s = 0.9) popularity.
fn routing(r: &mut Rng, rows: usize) -> Vec<u8> {
    let mut cum = Vec::with_capacity(E);
    let mut acc = 0.0;
    for e in 0..E {
        acc += 1.0 / ((e + 1) as f64).powf(0.9);
        cum.push(acc);
    }
    let mut ids = Vec::with_capacity(rows * TOP_K * 4);
    for _ in 0..rows {
        let mut picked: Vec<i32> = Vec::with_capacity(TOP_K);
        while picked.len() < TOP_K {
            let u = r.unit() * acc;
            let e = cum.partition_point(|&x| x < u).min(E - 1) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked.iter().flat_map(|v| v.to_le_bytes()));
    }
    ids
}

fn time_ms(g: &dyn GpuBackend, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// 2026-10-06: The packed A + SFA region and the gs bits the last W4A4 call left behind.
fn last_pack(g: &dyn GpuBackend) -> Result<(Vec<u8>, Vec<u8>, cutlass::W4a4LastPrep)> {
    let Some(p) = cutlass::w4a4_last_prep() else {
        bail!("no CUTLASS last-prep record");
    };
    let a = dn(g, DevicePtr(p.ws_base), (p.sfa_off + p.sfa_bytes) as usize)?;
    let gs = dn(g, DevicePtr(p.ws_base + p.gs_off), p.groups as usize * 4)?;
    Ok((a, gs, p))
}

fn main() -> Result<()> {
    if !cutlass::available() {
        println!("FAIL: this build has no CUTLASS objects (build with CUTLASS_HOME set)");
        bail!("glm_moe_w4a4_pack_once_microtest: no CUTLASS in this build");
    }
    let mi = env_usize("GLM_MT_MI", 1024);
    let local = env_usize("GLM_MT_LOCAL", 288).min(E);
    let iters = env_usize("GLM_W4A4_MT_ITERS", 10).max(1);
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let gb = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gb;
    let k_sort = g.kernel("moe", "moe_sort_by_expert")?;
    let k_swiglu = g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?;
    let k_amax = g.kernel("glm5next_ffn", "glm5next_swiglu_clamp_amax")?;
    let (ws_base, ws_size) = cutlass::workspace()?;
    let fill_ws = || g.memset_async(DevicePtr(ws_base), SENTINEL, ws_size, 0);
    let slot = g.alloc(256)?;
    println!(
        "mi={mi} local_experts={local} workspace={} MiB",
        ws_size >> 20
    );

    let mut r = Rng(0x2545_F491_4F6C_DD1D);
    let gate = build_proj(g, &mut r, mi, H, local)?;
    let upp = build_proj(g, &mut r, mi, H, local)?;
    let down = build_proj(g, &mut r, H, mi, local)?;
    let cache = SfbCache::new(g, local, H, mi)?;
    cache.fill(gate.scale_tab, upp.scale_tab, down.scale_tab, 0, local, 0)?;
    let nv = |p: &Proj, e: usize| Nvfp4Proj {
        packed: DevicePtr(p.packed[e]),
        scale: DevicePtr(p.scale[e]),
        scale_2: p.s2[e],
        input_scale: 0.0,
    };
    let experts: Vec<Glm5NextExpertWeights> = (0..local)
        .map(|e| Glm5NextExpertWeights {
            gate_proj: nv(&gate, e),
            up_proj: nv(&upp, e),
            down_proj: nv(&down, e),
        })
        .collect();
    let tables = MoeTables::build(E, 0..local, &experts, &cache.layout(), true);
    let mut chan: Vec<f32> = (0..H).map(|_| (0.5 * r.normal()).exp() as f32).collect();
    for _ in 0..8 {
        let c = (r.bits() % H as u64) as usize;
        chan[c] *= 12.0 + 8.0 * r.unit() as f32;
    }

    let mut problems: Vec<String> = Vec::new();
    let mut summary: Vec<String> = Vec::new();
    for tokens in WINDOWS {
        let te = tokens * TOP_K;
        let x: Vec<u8> = (0..tokens * H)
            .flat_map(|i| bf16::from_f32(r.normal() as f32 * 0.5 * chan[i % H]).to_le_bytes())
            .collect();
        let d_x = up(g, &x)?;
        let d_ids = up(g, &routing(&mut r, tokens))?;
        let (d_stid, d_seid, d_t2p) = (g.alloc(te * 4)?, g.alloc(te * 4)?, g.alloc(te * 4)?);
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
        let off: Vec<i32> = dn(g, d_off, (E + 1) * 4)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let span = swiglu_span(&off, 0..local, te, true);
        let (a_gate, a_up) = (g.alloc(te * mi * 2)?, g.alloc(te * mi * 2)?);
        let (act, out) = (g.alloc(te * mi * 2)?, g.alloc(te * H * 2)?);
        let mut bad = |what: String| problems.push(format!("tokens={tokens}: {what}"));

        // 2026-10-06: (1) gate/up pack, lever off vs on (and the known-bad (a)).
        let gate_up = |po: bool| -> Result<bool> {
            cutlass::nvfp4_grouped_gate_up_w4a4_ex(
                d_x.0,
                d_stid.0,
                &tables.gate.packed,
                &tables.gate.sfb,
                &tables.gate.scale2,
                &tables.up.packed,
                &tables.up.sfb,
                &tables.up.scale2,
                &tables.gate_up_gs,
                a_gate.0,
                a_up.0,
                &off,
                mi as u32,
                H as u32,
                tokens,
                po,
                0,
            )
        };
        let pack_gu = |po: bool| -> Result<(Vec<u8>, Vec<u8>, bool)> {
            fill_ws()?;
            let eng = gate_up(po)?;
            g.synchronize(0)?;
            let (a, gs, _) = last_pack(g)?;
            Ok((a, gs, eng))
        };
        let (a0, gs0, e0) = pack_gu(false)?;
        let (a1, gs1, e1) = pack_gu(true)?;
        let (d_a, d_gs) = (ndiff(&a0, &a1), ndiff(&gs0, &gs1));
        println!(
            "tokens={tokens} te={te} gate_up: packed A+SFA {d_a} differing bytes of {}, gs {d_gs} \
             of {}, engaged off={e0} on={e1}",
            a0.len(),
            gs0.len()
        );
        if d_a != 0 || d_gs != 0 {
            bad(format!("gate/up pack-once A+SFA/gs differ ({d_a}, {d_gs})"));
        }
        if e0 || !e1 {
            bad(format!(
                "pack-once engagement off={e0} on={e1} (want false/true)"
            ));
        }
        cutlass::set_w4a4_pack_once_fault(true);
        let (a2, _, _) = pack_gu(true)?;
        cutlass::set_w4a4_pack_once_fault(false);
        let kd = ndiff(&a0, &a2);
        if kd == 0 {
            bad("KNOWN_BAD (a) not detected: flipped staged scale left A+SFA unchanged".into());
        } else {
            println!("KNOWN_BAD (a) detected: flipped staged scale changes {kd} A+SFA bytes");
        }

        // 2026-10-06: (2) the whole run, both levers off vs both on.
        let (off_r, tables_r, span_c): (&[i32], &MoeTables, _) = (&off, &tables, span.clone());
        let args = move |on: bool| RunArgs {
            swiglu: k_swiglu,
            x: d_x,
            sorted_token_ids: d_stid,
            a_gate,
            a_up,
            a_act: act,
            expert_out: out,
            te,
            hidden: H,
            moe_intermediate: mi,
            swiglu_limit: LIMIT,
            offsets: off_r,
            tables: tables_r,
            skip_down: false,
            swiglu_rows: span_c.clone(),
            tokens,
            pack_once: on,
            swiglu_amax: if on { k_amax } else { KernelHandle(0) },
            amax_slot: if on { slot } else { DevicePtr(0) },
        };
        type Res = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, cutlass::W4a4LastPrep);
        let run_all = |on: bool| -> Result<Res> {
            fill_ws()?;
            g.memset_async(act, 0, te * mi * 2, 0)?;
            g.memset_async(out, 0, te * H * 2, 0)?;
            run(g, &args(on), 0)?;
            g.synchronize(0)?;
            let (a, gs, p) = last_pack(g)?;
            Ok((a, gs, dn(g, act, te * mi * 2)?, dn(g, out, te * H * 2)?, p))
        };
        let (da0, dg0, ac0, o0, p0) = run_all(false)?;
        let (da1, dg1, ac1, o1, p1) = run_all(true)?;
        let diffs = [
            ndiff(&da0, &da1),
            ndiff(&dg0, &dg1),
            ndiff(&ac0, &ac1),
            ndiff(&o0, &o1),
        ];
        println!(
            "tokens={tokens} run: down A+SFA {} / gs {} / a_act {} / expert_out {} differing \
             bytes (of {} / {} / {} / {}); down pre-amax engaged off={} on={}",
            diffs[0],
            diffs[1],
            diffs[2],
            diffs[3],
            da0.len(),
            dg0.len(),
            ac0.len(),
            o0.len(),
            p0.pre_amax,
            p1.pre_amax
        );
        if diffs.iter().any(|&d| d != 0) {
            bad(format!("levers-on run differs from levers-off {diffs:?}"));
        }
        if p0.pre_amax || !p1.pre_amax {
            bad(format!(
                "SwiGLU amax engagement off={} on={}",
                p0.pre_amax, p1.pre_amax
            ));
        }
        // 2026-10-06: (3) the slot holds exactly max |a_act| over the SwiGLU rows.
        let host_max = (span.start * mi..span.end * mi).fold(0f32, |m, i| {
            let v = bf16::from_le_bytes([ac1[2 * i], ac1[2 * i + 1]])
                .to_f32()
                .abs();
            if v.is_nan() { m } else { m.max(v) }
        });
        let slot_bits = u32::from_le_bytes(dn(g, slot, 4)?.try_into().unwrap_or([0; 4]));
        if slot_bits != host_max.to_bits() {
            bad(format!(
                "amax slot {slot_bits:#x} != host max |a_act| {:#x}",
                host_max.to_bits()
            ));
        }

        // 2026-10-06: Known-bad (b): the slot nudged down 16 ULPs must change the down gs.
        gate_up(true)?;
        let n = span.len() * mi;
        let row = mi * 2;
        g.memset_async(slot, 0, 4, 0)?;
        swiglu_rows_amax(
            g,
            k_amax,
            a_gate.offset(span.start * row),
            a_up.offset(span.start * row),
            act.offset(span.start * row),
            n,
            LIMIT,
            slot,
            0,
        )?;
        g.synchronize(0)?;
        let bits = u32::from_le_bytes(dn(g, slot, 4)?.try_into().unwrap_or([0; 4]));
        g.copy_h2d(&bits.saturating_sub(16).to_le_bytes(), slot)?;
        let t = &tables;
        let used = cutlass::nvfp4_grouped_down_w4a4_ex(
            act.0,
            &t.down.packed,
            &t.down.sfb,
            &t.down.scale2,
            &t.down_gs,
            out.0,
            &off,
            H as u32,
            mi as u32,
            Some((slot.0, span.clone())),
            0,
        )?;
        g.synchronize(0)?;
        let (_, dg2, _) = last_pack(g)?;
        if !used || ndiff(&dg0, &dg2) == 0 {
            bad(format!(
                "KNOWN_BAD (b) not detected (used={used}): nudged amax kept gs"
            ));
        } else {
            println!("KNOWN_BAD (b) detected: nudged amax changes the down gs bits");
        }

        // 2026-10-06: Timing, off vs on: gate/up call (prep + GEMMs), whole run.
        let t_gu0 = time_ms(g, iters, || gate_up(false).map(|_| ()))?;
        let t_gu1 = time_ms(g, iters, || gate_up(true).map(|_| ()))?;
        let t_r0 = time_ms(g, iters, || run(g, &args(false), 0))?;
        let t_r1 = time_ms(g, iters, || run(g, &args(true), 0))?;
        println!(
            "TIMING tokens={tokens} gate_up off_ms={t_gu0:.3} on_ms={t_gu1:.3} saved={:.3} | \
             run off_ms={t_r0:.3} on_ms={t_r1:.3} saved={:.3}",
            t_gu0 - t_gu1,
            t_r0 - t_r1
        );
        summary.push(format!("{tokens}: run -{:.3} ms", t_r0 - t_r1));
        for p in [
            d_x, d_ids, d_stid, d_seid, d_t2p, d_off, a_gate, a_up, act, out,
        ] {
            g.free(p)?;
        }
    }
    if !problems.is_empty() {
        println!("FAIL: {}", problems.join("; "));
        bail!(
            "glm_moe_w4a4_pack_once_microtest: {} problem(s)",
            problems.len()
        );
    }
    println!(
        "PASS: pack-once gate/up A+SFA+gs, SwiGLU-amax down A+SFA+gs, a_act and expert_out \
         bitwise identical, both levers engaged, both known-bads detected ({})",
        summary.join(", ")
    );
    Ok(())
}
