// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Accuracy and timing check for the opt-in FlashKDA KDA prefill
//! (`METRALE_GLM_KDA_PREFILL_FLASHKDA=1`, `Glm5NextKdaLayer::prefill_flashkda`, vendor/flashkda)
//! against `decode_k`, the per-token recurrence prefill runs today (with
//! `METRALE_GLM_KDA_TOKEN_LOOP=1` its two-launch form, byte-identical to the walk).
//!
//! Owner: model-arch examples (GLM-5.3 KDA).
//! Invariants:
//! - Exits nonzero unless, for every case, seed and weight profile:
//!   - `prefill_flashkda` took every call (returned true; a refusal is a FAIL here, not a
//!     silent fallback);
//!   - the final conv state is identical bit for bit (the conv path is the token loop's);
//!   - the core output (FlashKDA's BF16 against the loop's FP32 `[k, heads * 128]`), the layer
//!     output (`final_out`, BF16 `[k, hidden]`) and the final recurrent state (FP32 `[heads,
//!     128, 128]`) are finite with cosine similarity above `COS_MIN`.
//!   `COS_MIN` is a wiring check (a wrong layout, transpose or offset lands far below it), not
//!   the quality bar: FlashKDA keeps the state in BF16 between 16-row chunks, so the recurrence
//!   is not bit-identical by design. Max and mean absolute errors are printed for the quality
//!   call; the model-level gate (TEB, needle, smoke) decides.
//! - Refuses to report PASS when the loop's outputs are all zero or not finite, or the loop left
//!   the recurrent state unchanged.
//!
//! Arms, on the same bound layer, device inputs and starting states, one workspace:
//! - LOOP: `decode_k(.., snapshots = [])` over consecutive calls of `sub` rows.
//! - FLASH: `prefill_flashkda` over the same calls (calls above 4,096 rows are split into
//!   pieces inside, the state carried in place).
//!
//! Inputs: synthetic LCG weights and hidden states at the GLM-5.3-Flash TP=2 per-rank KDA
//! geometry (hidden 4096, 32 heads x 128, conv 4, gate_lower_bound -5), two weight profiles
//! (MILD: slow decay; FULL: decay reaching the -5 bound), a non-zero starting recurrent state so
//! both state transposes are exercised. With `FLASHKDA_PACKET=/path/layer.safetensors` (a KDA
//! layer packet with layer-relative `self_attn.*` names, as `kda_layer_microtest` reads) the
//! cases also run on that layer's checkpoint weights, at the packet's head count.
//!
//! After the check it prints `TIMING` lines at 64, 256, 2,048 and 8,192 rows: the whole KDA
//! layer per arm (one call, and 256-row calls as the staged prefill makes them), and the
//! recurrence alone (`kda_recurrent_prefill_bf16_smem` against one FlashKDA call). Wall clock
//! over repeated calls on stream 0; timing never affects the verdict.
//!
//! Needs a build with FlashKDA (`FLASHKDA_CUTLASS_HOME` set; the GB10 GLM image sets it):
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   FLASHKDA_CUTLASS_HOME=/opt/cutlass-flashkda \
//!   cargo build -p metrale-model-arch --release --example flashkda_prefill_microtest \
//!       --features cuda,gpu-examples
//!   METRALE_GLM_KDA_TOKEN_LOOP=1 ./flashkda_prefill_microtest
use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::flashkda;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda::binding::{self, KdaDtype, KdaTensorSource, RawTensor};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::weight_map::DenseWeight;
use std::collections::BTreeMap;
use std::time::Instant;

const D: usize = 128;
const D_CONV: usize = 4;
const HIDDEN: usize = 4096;
const HEADS: usize = 32;
/// 2026-10-03: `KDA_V_PER_BLOCK` in glm5next_kda/mod.rs, for the loop-recurrence timing.
const VPB: usize = 32;
/// 2026-10-03: Wiring check, see the module doc. PROVISIONAL.
const COS_MIN: f64 = 0.995;
/// 2026-10-03: The workspace's row count; the largest case.
const MAX_ROWS: usize = 8192;
/// 2026-10-03: `(rows, call width)`. 100 / 1000 end on a partial 16-row chunk; 4100 and 8192 in
/// one call split into two pieces inside; 8192 at 256 is the 8K prefill in the staged prefill's
/// 256-row sub-chunks.
const CASES: [(usize, usize); 8] = [
    (64, 64),
    (100, 100),
    (256, 256),
    (1000, 1000),
    (2048, 2048),
    (4100, 4100),
    (8192, 256),
    (8192, 8192),
];
const SEEDS: [u64; 2] = [0xF1A5_4CDA_0001, 0xF1A5_4CDA_0002];
const TIME_ROWS: [usize; 4] = [64, 256, 2048, 8192];
const TIME_SUB: usize = 256;
const TIME_WARMUP: usize = 2;
const TIME_REPS: usize = 5;

#[derive(Clone, Copy, Debug)]
enum Profile {
    Mild,
    Full,
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
    fn vec(&mut self, n: usize, a: f64) -> Vec<f32> {
        (0..n).map(|_| self.r(-a, a) as f32).collect()
    }
}

fn up_bytes(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    up_bytes(g, &b)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    up_bytes(g, &b)
}
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(dn_bytes(g, p, n * 2)?
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(dn_bytes(g, p, n * 4)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn dwt(g: &dyn GpuBackend, v: &[f32]) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: up_bf16(g, v)?,
    })
}

fn cfg_for(heads: usize, hidden: usize) -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden,
        heads,
        head_dim: D,
        conv_kernel: D_CONV,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    }
}

/// 2026-10-03: Synthetic weights at `cfg`'s geometry. Projections are uniform in
/// `±1/sqrt(fan_in)` so activations stay O(1). MILD: `A_log` near 0 and a negative `dt_bias`
/// keep the log-decay small; FULL: larger `A_log` and a wide `f_b` push the gate sigmoid to
/// saturation, so the decay reaches the -5 bound.
fn synthetic_weights(
    g: &dyn GpuBackend,
    cfg: &Glm5NextKdaConfig,
    seed: u64,
    p: Profile,
) -> Result<Glm5NextKdaWeights> {
    let mut r = Lcg(seed ^ 0x57E1_6475);
    let (hid, qkv, hd, h) = (cfg.hidden, cfg.qkv_dim(), cfg.head_dim, cfg.heads);
    let s = |n: usize| 1.0 / (n as f64).sqrt();
    let (a_log_hi, dt_lo, dt_hi, fb_scale) = match p {
        Profile::Mild => (0.3, -4.0, -2.0, 0.5),
        Profile::Full => (2.5, -1.0, 2.0, 3.0),
    };
    Ok(Glm5NextKdaWeights {
        q_proj: dwt(g, &r.vec(qkv * hid, s(hid)))?,
        k_proj: dwt(g, &r.vec(qkv * hid, s(hid)))?,
        v_proj: dwt(g, &r.vec(qkv * hid, s(hid)))?,
        conv: dwt(g, &r.vec(3 * qkv * cfg.conv_kernel, 0.5))?,
        f_a: dwt(g, &r.vec(hd * hid, s(hid)))?,
        f_b: dwt(g, &r.vec(qkv * hd, fb_scale * s(hd)))?,
        dt_bias: up_f32(
            g,
            &(0..qkv)
                .map(|_| r.r(dt_lo, dt_hi) as f32)
                .collect::<Vec<_>>(),
        )?,
        a_log: up_f32(
            g,
            &(0..h)
                .map(|_| r.r(0.0, a_log_hi) as f32)
                .collect::<Vec<_>>(),
        )?,
        b_proj: dwt(g, &r.vec(h * hid, 2.0 * s(hid)))?,
        g_a: dwt(g, &r.vec(hd * hid, s(hid)))?,
        g_b: dwt(g, &r.vec(qkv * hd, s(hd)))?,
        o_norm: dwt(
            g,
            &(0..hd).map(|_| r.r(0.5, 1.5) as f32).collect::<Vec<_>>(),
        )?,
        o_proj: dwt(g, &r.vec(hid * qkv, s(qkv)))?,
    })
}

/// 2026-10-03: A KDA layer packet (safetensors, layer-relative names) as a [`KdaTensorSource`].
struct Packet {
    raw: Vec<u8>,
    base: usize,
    hdr: BTreeMap<String, (KdaDtype, Vec<usize>, usize, usize)>,
}

impl Packet {
    fn open(path: &str) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let hn = u64::from_le_bytes(raw[..8].try_into()?) as usize;
        let j: serde_json::Value = serde_json::from_slice(&raw[8..8 + hn])?;
        let mut hdr = BTreeMap::new();
        for (k, m) in j.as_object().context("packet header is not an object")? {
            if k == "__metadata__" {
                continue;
            }
            let Some(dt) = m["dtype"].as_str().and_then(KdaDtype::parse) else {
                continue;
            };
            let shape = m["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let a = m["data_offsets"][0].as_u64().context("offset")? as usize;
            let b = m["data_offsets"][1].as_u64().context("offset")? as usize;
            hdr.insert(k.clone(), (dt, shape, a, b));
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    /// 2026-10-03: `(heads, hidden)` from `self_attn.A_log` `[H]` and `q_proj` `[H*128, X]`.
    fn geometry(&self) -> Result<(usize, usize)> {
        let h = self
            .hdr
            .get("self_attn.A_log")
            .context("no self_attn.A_log")?;
        let q = self
            .hdr
            .get("self_attn.q_proj.weight")
            .context("no self_attn.q_proj.weight")?;
        Ok((h.1[0], q.1[1]))
    }
}

impl KdaTensorSource for Packet {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        let (dt, shape, a, b) = self.hdr.get(name)?;
        Some(RawTensor {
            dtype: *dt,
            shape: shape.clone(),
            bytes: &self.raw[self.base + a..self.base + b],
        })
    }
    fn names(&self) -> Vec<String> {
        self.hdr.keys().cloned().collect()
    }
}

/// 2026-10-03: One arm's outputs, concatenated over its calls.
struct Captured {
    core: Vec<f32>,
    out: Vec<f32>,
    state: Vec<f32>,
    conv_state: Vec<u8>,
    took_all: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Loop,
    Flash,
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    cfg: Glm5NextKdaConfig,
}

impl Rig<'_> {
    /// 2026-10-03: One arm over `k` rows of `hidden` in calls of `sub` rows, from `conv0` /
    /// `h0`, reading every call's core and layer output back.
    fn run(
        &self,
        arm: Arm,
        hidden: DevicePtr,
        k: usize,
        sub: usize,
        conv0: &[f32],
        h0: &[f32],
    ) -> Result<Captured> {
        let (g, c, ws) = (self.g, &self.cfg, &self.ws);
        let (qkv, hid) = (c.qkv_dim(), c.hidden);
        let st = KdaSeqState {
            conv: up_f32(g, conv0)?,
            recurrent: up_f32(g, h0)?,
        };
        let mut cap = Captured {
            core: Vec::with_capacity(k * qkv),
            out: Vec::with_capacity(k * hid),
            state: Vec::new(),
            conv_state: Vec::new(),
            took_all: true,
        };
        for r0 in (0..k).step_by(sub) {
            let rows = sub.min(k - r0);
            let x = hidden.offset(r0 * hid * 2);
            match arm {
                Arm::Loop => self.layer.decode_k(g, x, rows, &st, ws, &[], 0)?,
                Arm::Flash => {
                    cap.took_all &= self.layer.prefill_flashkda(g, x, rows, &st, ws, 0)?;
                }
            }
            g.synchronize(0)?;
            match arm {
                Arm::Loop => cap.core.extend(dn_f32(g, ws.core, rows * qkv)?),
                Arm::Flash => cap.core.extend(dn_bf16(g, ws.conv_out, rows * qkv)?),
            }
            cap.out.extend(dn_bf16(g, ws.final_out, rows * hid)?);
        }
        cap.state = dn_f32(g, st.recurrent, c.recurrent_state_elems())?;
        cap.conv_state = dn_bytes(g, st.conv, c.conv_state_elems() * 4)?;
        let _ = g.free(st.conv);
        let _ = g.free(st.recurrent);
        Ok(cap)
    }
}

/// 2026-10-03: `(cosine, max |a - b|, mean |a - b|, max |a|, all of b finite)`; a length
/// mismatch gives cosine 0.
fn closeness(a: &[f32], b: &[f32]) -> (f64, f64, f64, f64, bool) {
    if a.len() != b.len() || a.is_empty() {
        return (0.0, f64::INFINITY, f64::INFINITY, 0.0, false);
    }
    let (mut ab, mut aa, mut bb, mut emax, mut esum, mut amax) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        let e = (x - y).abs();
        emax = f64::max(emax, e);
        esum += e;
        amax = f64::max(amax, x.abs());
    }
    let cos = if aa > 0.0 && bb > 0.0 {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    };
    (
        cos,
        emax,
        esum / a.len() as f64,
        amax,
        b.iter().all(|y| y.is_finite()),
    )
}

/// 2026-10-03: The cases for one bound layer; returns whether all passed.
fn run_cases(rig: &Rig, tag: &str) -> Result<bool> {
    let (g, c) = (rig.g, &rig.cfg);
    let mut all_ok = true;
    for &(k, sub) in &CASES {
        for &seed in &SEEDS {
            let mut r = Lcg(seed ^ k as u64);
            let hidden = up_bf16(g, &r.vec(k * c.hidden, 1.0))?;
            let conv0 = r.vec(c.conv_state_elems(), 0.3);
            let h0 = r.vec(c.recurrent_state_elems(), 0.1);
            let lp = rig.run(Arm::Loop, hidden, k, sub, &conv0, &h0)?;
            let fl = rig.run(Arm::Flash, hidden, k, sub, &conv0, &h0)?;
            let _ = g.free(hidden);

            let live = lp.core.iter().all(|x| x.is_finite()) && lp.core.iter().any(|x| *x != 0.0);
            let moved = lp.state != h0;
            let cstate_eq = lp.conv_state == fl.conv_state;
            let (c_cos, c_max, c_mean, c_ref, c_fin) = closeness(&lp.core, &fl.core);
            let (o_cos, o_max, o_mean, o_ref, o_fin) = closeness(&lp.out, &fl.out);
            let (s_cos, s_max, s_mean, s_ref, s_fin) = closeness(&lp.state, &fl.state);
            let ok = live
                && moved
                && fl.took_all
                && cstate_eq
                && c_fin
                && o_fin
                && s_fin
                && c_cos > COS_MIN
                && o_cos > COS_MIN
                && s_cos > COS_MIN;
            all_ok &= ok;
            eprintln!(
                "{tag} rows={k:>4} sub={sub:>4} seed={seed:#x} took={} conv_state={} \
                 core cos={c_cos:.6} max={c_max:.3e} mean={c_mean:.3e} (max|ref|={c_ref:.3e}) \
                 out cos={o_cos:.6} max={o_max:.3e} mean={o_mean:.3e} (max|ref|={o_ref:.3e}) \
                 state cos={s_cos:.6} max={s_max:.3e} mean={s_mean:.3e} (max|ref|={s_ref:.3e}){}  {}",
                fl.took_all,
                if cstate_eq { "eq" } else { "MISMATCH" },
                if live && moved {
                    String::new()
                } else {
                    format!("  VACUOUS(live={live} moved={moved})")
                },
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    Ok(all_ok)
}

/// 2026-10-03: Mean wall-clock milliseconds per call of `f` on stream 0.
fn time_ms(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..TIME_WARMUP {
        f()?;
    }
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..TIME_REPS {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / TIME_REPS as f64)
}

/// 2026-10-03: `TIMING` for `rows` rows. Repeated calls advance the states; the work per call
/// does not change.
fn timing(rig: &Rig, recurrent_rows: KernelHandle, rows: usize) -> Result<()> {
    let (g, c, ws) = (rig.g, &rig.cfg, &rig.ws);
    let (qkv, cd, h) = (c.qkv_dim(), c.conv_dim(), c.heads);
    let mut r = Lcg(SEEDS[0] ^ 0x7131);
    let hidden = up_bf16(g, &r.vec(rows * c.hidden, 1.0))?;
    let st = KdaSeqState {
        conv: up_f32(g, &r.vec(c.conv_state_elems(), 0.3))?,
        recurrent: up_f32(g, &r.vec(c.recurrent_state_elems(), 0.1))?,
    };
    let layer = &rig.layer;
    let calls = |sub: usize, arm: Arm| -> Result<()> {
        for r0 in (0..rows).step_by(sub) {
            let n = sub.min(rows - r0);
            let x = hidden.offset(r0 * c.hidden * 2);
            match arm {
                Arm::Loop => layer.decode_k(g, x, n, &st, ws, &[], 0)?,
                Arm::Flash => {
                    if !layer.prefill_flashkda(g, x, n, &st, ws, 0)? {
                        bail!("prefill_flashkda refused {n} rows");
                    }
                }
            }
        }
        Ok(())
    };
    let loop_layer = time_ms(g, || calls(rows, Arm::Loop))?;
    let flash_layer = time_ms(g, || calls(rows, Arm::Flash))?;
    let (loop_sub, flash_sub) = if rows > TIME_SUB {
        (
            Some(time_ms(g, || calls(TIME_SUB, Arm::Loop))?),
            Some(time_ms(g, || calls(TIME_SUB, Arm::Flash))?),
        )
    } else {
        (None, None)
    };

    // 2026-10-03: The recurrence alone. LOOP: `stateful_rows`' recurrent launch over the conv
    // output, gate and beta a LOOP call just left in the workspace. FLASH: one library call over
    // the contiguous q|k|v, raw gate and transposed beta a FLASH call just left there.
    calls(rows, Arm::Loop)?;
    let loop_recur = time_ms(g, || {
        KernelLaunch::new(g, recurrent_rows)
            .grid([h as u32, (D / VPB) as u32, 1])
            .block([VPB as u32, 1, 1])
            .shared_mem(((3 * D + VPB * (D + 1)) * 4) as u32)
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.conv_out.offset(qkv * 2))
            .arg_ptr(ws.conv_out.offset(qkv * 4))
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(st.recurrent)
            .arg_ptr(ws.core)
            .arg_u32(h as u32)
            .arg_u32(D as u32)
            .arg_f32(1.0 / (D as f32).sqrt())
            .arg_u32(VPB as u32)
            .arg_u32(rows as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(h as u32)
            .arg_u32(qkv as u32)
            .launch(0)
    })?;
    calls(rows, Arm::Flash)?;
    // 2026-10-03: A call above 4,096 rows leaves only its last piece's beta in `ws.beta`; lay
    // all `rows` out head-major again so the timed call reads real logits.
    KernelLaunch::new(g, layer.kernels.flk_beta_t)
        .grid([rows.div_ceil(256) as u32, h as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(ws.beta_bf16_ptr())
        .arg_ptr(ws.beta)
        .arg_u32(rows as u32)
        .arg_u32(h as u32)
        .launch(0)?;
    let fws_bytes = flashkda::workspace_bytes(rows, h);
    let fws = g.alloc(fws_bytes)?;
    let fstate = up_f32(g, &r.vec(c.recurrent_state_elems(), 0.1))?;
    let w = &layer.weights;
    let flash_recur = time_ms(g, || {
        flashkda::fwd_fp32_state(
            &flashkda::FwdArgs {
                q: ws.qkv_parts.0,
                k: ws.qkv_parts.offset(rows * qkv * 2).0,
                v: ws.qkv_parts.offset(2 * rows * qkv * 2).0,
                g: ws.g_raw.0,
                beta_t: ws.beta.0,
                state: fstate.0,
                out: ws.conv_out.0,
                workspace: fws.0,
                workspace_bytes: fws_bytes,
                rows,
                heads: h,
                a_log: w.a_log.0,
                dt_bias: w.dt_bias.0,
                scale: 1.0 / (D as f32).sqrt(),
                lower_bound: c.gate_lower_bound,
            },
            0,
        )
    })?;
    let fmt = |v: Option<f64>| v.map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"));
    let ratio = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (Some(a), Some(b)) => format!("{:.2}x", a / b),
        _ => "n/a".to_string(),
    };
    eprintln!(
        "TIMING rows={rows} heads={h} layer_loop_ms={loop_layer:.3} layer_flash_ms={flash_layer:.3} \
         layer_speedup={:.2}x layer_loop_sub{TIME_SUB}_ms={} layer_flash_sub{TIME_SUB}_ms={} \
         layer_speedup_sub{TIME_SUB}={} recur_loop_ms={loop_recur:.3} \
         recur_flash_ms={flash_recur:.3} recur_speedup={:.2}x",
        loop_layer / flash_layer,
        fmt(loop_sub),
        fmt(flash_sub),
        ratio(loop_sub, flash_sub),
        loop_recur / flash_recur
    );
    for p in [fws, fstate, hidden, st.conv, st.recurrent] {
        let _ = g.free(p);
    }
    Ok(())
}

fn rig_for<'a>(
    g: &'a dyn GpuBackend,
    cfg: Glm5NextKdaConfig,
    weights: Glm5NextKdaWeights,
) -> Result<Rig<'a>> {
    let kernels = Glm5NextKdaKernels::resolve(g)?;
    if !kernels.has_flashkda_glue() {
        bail!("the target lacks a kda_flashkda_glue / kda_chunk_tc kernel");
    }
    let layer = Glm5NextKdaLayer::new(0, cfg, weights, kernels)?;
    let mut ws = Glm5NextKdaWorkspace::new(g, &cfg, MAX_ROWS)?;
    let bytes = ws.alloc_flashkda(g, &cfg)?;
    if bytes == 0 {
        bail!("no FlashKDA scratch allocated (geometry outside the path?)");
    }
    Ok(Rig { g, layer, ws, cfg })
}

fn main() -> Result<()> {
    if !flashkda::available() {
        bail!("this binary was built without FlashKDA; set FLASHKDA_CUTLASS_HOME at build time");
    }
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let recurrent_rows = g.kernel("kda_recurrent", "kda_recurrent_prefill_bf16_smem")?;

    let mut all_ok = true;
    let cfg = cfg_for(HEADS, HIDDEN);
    for (i, profile) in [Profile::Mild, Profile::Full].into_iter().enumerate() {
        let w = synthetic_weights(g, &cfg, SEEDS[i], profile)?;
        let rig = rig_for(g, cfg, w)?;
        all_ok &= run_cases(&rig, &format!("synthetic/{profile:?}"))?;
        if i == 0 {
            for rows in TIME_ROWS {
                timing(&rig, recurrent_rows, rows)?;
            }
        }
    }
    if let Ok(path) = std::env::var("FLASHKDA_PACKET") {
        let p = Packet::open(&path)?;
        let (heads, hidden) = p.geometry()?;
        let pcfg = cfg_for(heads, hidden);
        let (w, _) = binding::bind_kda_weights(g, &pcfg, 0, &p)?;
        let rig = rig_for(g, pcfg, w)?;
        all_ok &= run_cases(&rig, &format!("packet/h{heads}"))?;
    }

    eprintln!(
        "\nFlashKDA prefill CHECK (conv state bitwise; core, layer output and state cosine > \
         {COS_MIN} vs decode_k; errors above are the quality evidence): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
