// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Byte-identity gate for the in-pass SSM capture on the tensor-core chunked KDA
//! prefill (`METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1` with `METRALE_GLM_SSM_INPASS_CAPTURE=1`,
//! `Glm5NextKdaLayer::prefill_chunked_tc(.., Some(&InpassSplit), ..)`, prefill_tc.rs).
//!
//! Owner: model-arch examples (GLM-5.3 KDA).
//! Invariants:
//! - Prints one final line starting `PASS` only when, for every row count `T` and cut:
//!   (a) the captured call reports a capture, and its layer output (`final_out`, BF16), core
//!       output (`core`, FP32), final recurrent state and final conv state equal the uncaptured
//!       call's bit for bit (compared as u16 / u32 words);
//!   (b) the captured h/conv equal, bit for bit, the final states of a separate TC recurrence
//!       over only the first `cut` rows from the same starting states
//!       (`chunked_tc_recurrence_arm`, on the workspace rows the full call's front end wrote, so
//!       the reference shares the projections and differs only in where the recurrence stops);
//!   (c) known-bad: a capture at `cut ± 16` differs from the `cut` reference in both h and conv,
//!       so the comparison in (b) is not blind;
//!   (d) a cut inside a 16-row chunk reports no capture, leaves the slot as poisoned, and
//!       changes no output or state.
//! - Otherwise prints `FAIL ...` lines and exits 1. Refuses PASS when the target lacks the TC
//!   kernels (the prefill would fall back to `decode_k` and (a) would compare `decode_k` with
//!   itself), when the uncaptured outputs are all zero or not finite, or when the reference left
//!   either state unchanged.
//! - 2026-10-10 (FUSECAP): for an 8192-row call and cuts none/2048/4096/6144, the fused front
//!   (`METRALE_GLM_KDA_FRONT_FUSE`) and the unfused front with the same capture give identical
//!   slot conv, slot h, layer output, core and final states; prints `FUSECAP BITWISE cut=<n>
//!   conv=<identical|DIFF> h=<..> out=<..>` per cut and `PASS - FUSECAP` only if all are identical.
//! - Each arm fills its outputs and slots with its own poison byte first (0xAB uncaptured,
//!   0xCD captured), so a byte neither arm writes cannot pass as a match.
//!
//! Geometry: GLM-5.3-Flash KDA at the TP=2 per-rank head count (hidden 6144, 32 heads x 128,
//! conv 4), synthetic LCG weights; `T` in {2048 + 37, 8192}.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example kda_tc_capture_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::layer::InpassSplit;
use metrale_model_layers::weight_map::DenseWeight;

const D: usize = 128;
const D_CONV: usize = 4;
const HIDDEN: usize = 6144;
const HEADS: usize = 32;
/// 2026-10-06: `KDA_TC_C` in glm5next_kda/mod.rs.
const TC_C: usize = 16;
/// 2026-10-06: 2085 ends on a partial chunk; 8192 is the 8K server prefill chunk.
const ROWS: [usize; 2] = [2048 + 37, 8192];
const SEED: u64 = 0x7CCA_0001;
const POISON_PLAIN: u8 = 0xAB;
const POISON_CAP: u8 = 0xCD;

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
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn dwt(g: &dyn GpuBackend, v: &[f32]) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: up_bf16(g, v)?,
    })
}

fn cfg() -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden: HIDDEN,
        heads: HEADS,
        head_dim: D,
        conv_kernel: D_CONV,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    }
}

/// 2026-10-06: Synthetic weights as `kda_snap_fuse_microtest`'s: projections uniform in
/// `±1/sqrt(fan_in)`, a mild decay.
fn synthetic_weights(g: &dyn GpuBackend, c: &Glm5NextKdaConfig) -> Result<Glm5NextKdaWeights> {
    let mut r = Lcg(0x57E1_6475);
    let (hid, qkv, hd, h) = (c.hidden, c.qkv_dim(), c.head_dim, c.heads);
    let s = |n: usize| 1.0 / (n as f64).sqrt();
    Ok(Glm5NextKdaWeights {
        q_proj: dwt(g, &r.vec(qkv * hid, s(hid)))?,
        k_proj: dwt(g, &r.vec(qkv * hid, s(hid)))?,
        v_proj: dwt(g, &r.vec(qkv * hid, s(hid)))?,
        conv: dwt(g, &r.vec(3 * qkv * c.conv_kernel, 0.5))?,
        f_a: dwt(g, &r.vec(hd * hid, s(hid)))?,
        f_b: dwt(g, &r.vec(qkv * hd, 0.5 * s(hd)))?,
        dt_bias: up_f32(g, &(0..qkv).map(|_| r.r(-4.0, -2.0) as f32).collect::<Vec<_>>())?,
        a_log: up_f32(g, &(0..h).map(|_| r.r(0.0, 0.3) as f32).collect::<Vec<_>>())?,
        b_proj: dwt(g, &r.vec(h * hid, 2.0 * s(hid)))?,
        g_a: dwt(g, &r.vec(hd * hid, s(hid)))?,
        g_b: dwt(g, &r.vec(qkv * hd, s(hd)))?,
        o_norm: dwt(g, &(0..hd).map(|_| r.r(0.5, 1.5) as f32).collect::<Vec<_>>())?,
        o_proj: dwt(g, &r.vec(hid * qkv, s(qkv)))?,
    })
}

/// 2026-10-06: The cuts for `t` rows: the first chunk boundary, 2048, the last chunk boundary
/// strictly inside the call, and the call's end.
fn cuts(t: usize) -> Vec<usize> {
    let inside = (t - 1) / TC_C * TC_C;
    let mut v = vec![TC_C, 2048, inside, t];
    v.dedup();
    v
}

/// 2026-10-06: A capture point 16 rows away from `cut` inside `1..=t` that the TC prefill
/// takes (a chunk boundary or `t`), for the known-bad comparison.
fn bad_cut(cut: usize, t: usize) -> usize {
    if cut + TC_C <= t {
        cut + TC_C
    } else {
        (cut - 1) / TC_C * TC_C
    }
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    cfg: Glm5NextKdaConfig,
}

/// 2026-10-06: One `prefill_chunked_tc` call's results as raw bytes.
struct Arm {
    took: bool,
    final_out: Vec<u8>,
    core: Vec<u8>,
    h: Vec<u8>,
    conv: Vec<u8>,
    slot_h: Vec<u8>,
    slot_conv: Vec<u8>,
}

impl Rig<'_> {
    fn bytes(&self) -> (usize, usize) {
        (
            self.cfg.recurrent_state_elems() * 4,
            self.cfg.conv_state_elems() * 4,
        )
    }

    fn states(&self, h0: &[f32], c0: &[f32]) -> Result<KdaSeqState> {
        Ok(KdaSeqState {
            conv: up_f32(self.g, c0)?,
            recurrent: up_f32(self.g, h0)?,
            recurrent_f16: false,
        })
    }

    /// 2026-10-06: One TC prefill of `k` rows of `hidden` from `(h0, c0)`, capturing after
    /// `cut` rows when `Some`. Outputs and the slot start as `poison`.
    fn run(
        &self,
        hidden: DevicePtr,
        k: usize,
        h0: &[f32],
        c0: &[f32],
        cut: Option<usize>,
        poison: u8,
        fuse: bool,
    ) -> Result<Arm> {
        let (g, c, ws) = (self.g, &self.cfg, &self.ws);
        let (hb, cb) = self.bytes();
        let st = self.states(h0, c0)?;
        let slot = (up_bytes(g, &vec![poison; hb])?, up_bytes(g, &vec![poison; cb])?);
        g.copy_h2d(&vec![poison; k * c.hidden * 2], ws.final_out)?;
        g.copy_h2d(&vec![poison; k * c.qkv_dim() * 4], ws.core)?;
        let split = cut.map(|row| InpassSplit {
            row,
            h_dst: slot.0,
            conv_dst: slot.1,
            h_bytes: hb,
            conv_bytes: cb,
        });
        let took = self
            .layer
            .prefill_chunked_tc_inner(g, hidden, k, &st, ws, fuse, split.as_ref(), 0)?;
        g.synchronize(0)?;
        let arm = Arm {
            took,
            final_out: dn_bytes(g, ws.final_out, k * c.hidden * 2)?,
            core: dn_bytes(g, ws.core, k * c.qkv_dim() * 4)?,
            h: dn_bytes(g, st.recurrent, hb)?,
            conv: dn_bytes(g, st.conv, cb)?,
            slot_h: dn_bytes(g, slot.0, hb)?,
            slot_conv: dn_bytes(g, slot.1, cb)?,
        };
        for p in [st.conv, st.recurrent, slot.0, slot.1] {
            let _ = g.free(p);
        }
        Ok(arm)
    }

    /// 2026-10-06: The reference: the TC recurrence over workspace rows `0..cut` from
    /// `(h0, c0)`, on the front end the last `run` of the same hidden left. Returns `(h, conv)`.
    fn reference(&self, cut: usize, h0: &[f32], c0: &[f32]) -> Result<(Vec<u8>, Vec<u8>)> {
        let g = self.g;
        let (hb, cb) = self.bytes();
        let st = self.states(h0, c0)?;
        self.layer
            .chunked_tc_recurrence_arm(g, cut, &st, &self.ws, 0)?;
        g.synchronize(0)?;
        let out = (dn_bytes(g, st.recurrent, hb)?, dn_bytes(g, st.conv, cb)?);
        let _ = g.free(st.conv);
        let _ = g.free(st.recurrent);
        Ok(out)
    }
}

/// 2026-10-06: `(words compared, first differing word)` over `width`-byte little-endian words
/// (2 = BF16 as u16, 4 = FP32 as u32); a length mismatch compares nothing.
fn diff(a: &[u8], b: &[u8], width: usize) -> (usize, Option<usize>) {
    if a.len() != b.len() || a.is_empty() {
        return (0, Some(0));
    }
    let word = |c: &[u8]| -> u32 {
        match width {
            2 => u16::from_le_bytes([c[0], c[1]]) as u32,
            _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        }
    };
    let n = a.len() / width;
    let first = a
        .chunks_exact(width)
        .zip(b.chunks_exact(width))
        .position(|(x, y)| word(x) != word(y));
    (n, first)
}

/// 2026-10-06: All finite and not all zero, read as `width`-byte floats.
fn live(b: &[u8], width: usize) -> bool {
    let v: Vec<f32> = match width {
        2 => b
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    };
    !v.is_empty() && v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0)
}

/// 2026-10-06: Collects one case's failures; `eq` and `ne` add the words they compared.
struct Checks {
    tag: String,
    fails: Vec<String>,
    words: usize,
}

impl Checks {
    fn eq(&mut self, name: &str, a: &[u8], b: &[u8], width: usize) {
        let (n, first) = diff(a, b, width);
        self.words += n;
        if n == 0 || first.is_some() {
            self.fails.push(format!("{name}: {n} words, first mismatch at {first:?}"));
        }
    }
    fn ne(&mut self, name: &str, a: &[u8], b: &[u8], width: usize) {
        let (n, first) = diff(a, b, width);
        self.words += n;
        if n == 0 || first.is_none() {
            self.fails.push(format!("{name}: known-bad comparison found them EQUAL"));
        }
    }
    fn holds(&mut self, name: &str, ok: bool) {
        if !ok {
            self.fails.push(name.to_string());
        }
    }
    /// 2026-10-06: Prints the case's line (and a `FAIL` line per failure); true when it held.
    fn report(self, total: &mut usize) -> bool {
        *total += self.words;
        if self.fails.is_empty() {
            println!("{} ok ({} words)", self.tag, self.words);
            return true;
        }
        for f in &self.fails {
            println!("FAIL {} {f}", self.tag);
        }
        false
    }
}

/// 2026-10-06: All cuts for `t` rows: (a)-(d) of the module doc. Returns whether every case
/// held; adds compared words to `total`.
fn run_rows(rig: &Rig, t: usize, total: &mut usize, cases: &mut usize) -> Result<bool> {
    let (g, c) = (rig.g, &rig.cfg);
    let mut r = Lcg(SEED ^ ((t as u64) << 16));
    let hidden = up_bf16(g, &r.vec(t * c.hidden, 1.0))?;
    let h0 = r.vec(c.recurrent_state_elems(), 0.1);
    let c0 = r.vec(c.conv_state_elems(), 0.3);
    let (h0b, c0b) = (f32_bytes(&h0), f32_bytes(&c0));
    let plain = rig.run(hidden, t, &h0, &c0, None, POISON_PLAIN, false)?;

    let mut all = true;
    let mut base = Checks {
        tag: format!("T={t} uncaptured"),
        fails: Vec::new(),
        words: 0,
    };
    base.holds("uncaptured call reported a capture", !plain.took);
    base.holds("final_out not live", live(&plain.final_out, 2));
    base.holds("core not live", live(&plain.core, 4));
    base.holds("final h_state equals the start", plain.h != h0b);
    base.holds("final conv_state equals the start", plain.conv != c0b);
    all &= base.report(total);

    for cut in cuts(t) {
        let cap = rig.run(hidden, t, &h0, &c0, Some(cut), POISON_CAP, false)?;
        let (ref_h, ref_conv) = rig.reference(cut, &h0, &c0)?;
        let bad = bad_cut(cut, t);
        let worse = rig.run(hidden, t, &h0, &c0, Some(bad), POISON_CAP, false)?;
        let mut k = Checks {
            tag: format!("T={t} cut={cut} bad={bad}"),
            fails: Vec::new(),
            words: 0,
        };
        // 2026-10-06: (a) the capture changes nothing the call computes.
        k.holds("captured call reported no capture", cap.took);
        k.eq("a.final_out", &plain.final_out, &cap.final_out, 2);
        k.eq("a.core", &plain.core, &cap.core, 4);
        k.eq("a.h_state", &plain.h, &cap.h, 4);
        k.eq("a.conv_state", &plain.conv, &cap.conv, 4);
        // 2026-10-06: (b) the slot holds the state after `cut` rows; non-vacuous reference.
        k.holds("reference h_state equals the start", ref_h != h0b);
        k.holds("reference conv_state equals the start", ref_conv != c0b);
        k.eq("b.slot_h", &ref_h, &cap.slot_h, 4);
        k.eq("b.slot_conv", &ref_conv, &cap.slot_conv, 4);
        // 2026-10-06: (c) a capture 16 rows off must not match the `cut` reference.
        k.holds("known-bad call reported no capture", worse.took);
        k.ne("c.slot_h", &ref_h, &worse.slot_h, 4);
        k.ne("c.slot_conv", &ref_conv, &worse.slot_conv, 4);
        all &= k.report(total);
        *cases += 1;
    }

    // 2026-10-06: (d) a cut inside a chunk: no capture, slot untouched, nothing else changed.
    let mid = t - 3;
    let skip = rig.run(hidden, t, &h0, &c0, Some(mid), POISON_CAP, false)?;
    let (hb, cb) = rig.bytes();
    let mut k = Checks {
        tag: format!("T={t} cut={mid} (inside a chunk)"),
        fails: Vec::new(),
        words: 0,
    };
    k.holds("mid-chunk call reported a capture", !skip.took);
    k.eq("d.slot_h untouched", &vec![POISON_CAP; hb], &skip.slot_h, 4);
    k.eq("d.slot_conv untouched", &vec![POISON_CAP; cb], &skip.slot_conv, 4);
    k.eq("d.final_out", &plain.final_out, &skip.final_out, 2);
    k.eq("d.core", &plain.core, &skip.core, 4);
    k.eq("d.h_state", &plain.h, &skip.h, 4);
    k.eq("d.conv_state", &plain.conv, &skip.conv, 4);
    all &= k.report(total);
    *cases += 1;

    let _ = g.free(hidden);
    Ok(all)
}

/// 2026-10-10: Fused front vs unfused front, each with the same in-pass capture, inside one
/// 8192-row call at cuts 2048, 4096 and 6144 (the 2048-mode grid inside an 8192 chunk): the
/// slot's conv state, the slot's h state, the layer output, core and final states must be
/// identical. Prints `FUSECAP BITWISE cut=<n> conv=<identical|DIFF> h=<identical|DIFF> ...`;
/// returns whether every line was identical. Also checks the no-capture fused call.
fn run_fusecap(rig: &Rig, cases: &mut usize) -> Result<bool> {
    let (g, c) = (rig.g, &rig.cfg);
    let t = 8192;
    let mut r = Lcg(SEED ^ 0xF05E);
    let hidden = up_bf16(g, &r.vec(t * c.hidden, 1.0))?;
    let h0 = r.vec(c.recurrent_state_elems(), 0.1);
    let c0 = r.vec(c.conv_state_elems(), 0.3);
    let (h0b, c0b) = (f32_bytes(&h0), f32_bytes(&c0));
    let word = |ok: bool| if ok { "identical" } else { "DIFF" };
    let same = |a: &[u8], b: &[u8], w: usize| {
        let (n, first) = diff(a, b, w);
        n > 0 && first.is_none()
    };
    let mut all = true;
    for cut in [None, Some(2048), Some(4096), Some(6144)] {
        let un = rig.run(hidden, t, &h0, &c0, cut, POISON_PLAIN, false)?;
        let fu = rig.run(hidden, t, &h0, &c0, cut, POISON_PLAIN, true)?;
        let live_ok = live(&un.final_out, 2) && live(&un.core, 4) && un.h != h0b && un.conv != c0b;
        let (conv, h) = match cut {
            Some(_) => (
                same(&un.slot_conv, &fu.slot_conv, 4) && un.took && fu.took,
                same(&un.slot_h, &fu.slot_h, 4),
            ),
            None => (same(&un.conv, &fu.conv, 4), same(&un.h, &fu.h, 4)),
        };
        let out = same(&un.final_out, &fu.final_out, 2)
            && same(&un.core, &fu.core, 4)
            && same(&un.h, &fu.h, 4)
            && same(&un.conv, &fu.conv, 4);
        let ok = conv && h && out && live_ok;
        match cut {
            Some(n) => println!(
                "FUSECAP BITWISE cut={n} conv={} h={} out={}{}",
                word(conv),
                word(h),
                word(out),
                if live_ok { "" } else { " (reference not live)" }
            ),
            None => println!(
                "FUSECAP BITWISE cut=none conv={} h={} out={}{}",
                word(conv),
                word(h),
                word(out),
                if live_ok { "" } else { " (reference not live)" }
            ),
        }
        all &= ok;
        *cases += 1;
    }
    let _ = g.free(hidden);
    Ok(all)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let c = cfg();
    let kernels = Glm5NextKdaKernels::resolve(g)?;
    let layer = Glm5NextKdaLayer::new(0, c, synthetic_weights(g, &c)?, kernels)?;
    // 2026-10-06: Without the TC kernels the prefill falls back to `decode_k`, and (a) would
    // compare the walk with itself.
    if !layer.kernels.has_chunked_tc() {
        println!("FAIL kda_tc_capture_microtest: the target lacks the kda_chunk_tc kernels");
        std::process::exit(1);
    }
    let max_rows = ROWS.iter().copied().max().unwrap_or(0);
    let ws = Glm5NextKdaWorkspace::new(g, &c, max_rows)?;
    let layer_has_fuse = layer.kernels.has_front_fuse();
    let rig = Rig { g, layer, ws, cfg: c };

    let (mut all, mut total, mut cases) = (true, 0usize, 0usize);
    for t in ROWS {
        all &= run_rows(&rig, t, &mut total, &mut cases)?;
    }
    if !layer_has_fuse {
        println!("FAIL FUSECAP: the target lacks the kda_front_fuse kernels");
        std::process::exit(1);
    }
    let fusecap = run_fusecap(&rig, &mut cases)?;
    println!("{}", if fusecap { "PASS - FUSECAP" } else { "FAIL - FUSECAP" });
    all &= fusecap;
    if total == 0 || cases == 0 {
        println!("FAIL kda_tc_capture_microtest: compared nothing; this run proves nothing");
        std::process::exit(1);
    }
    if !all {
        println!(
            "FAIL kda_tc_capture_microtest: the TC in-pass capture is not bit-identical (see the \
             lines above); keep METRALE_GLM_SSM_INPASS_CAPTURE off with the TC prefill"
        );
        std::process::exit(1);
    }
    println!(
        "PASS: kda_tc_capture_microtest {cases} cases, {total} words compared: TC prefill \
         outputs and states unchanged by the capture, slots equal to a TC prefill over the first \
         cut rows, known-bad cut+-16 detected, mid-chunk cut not captured"
    );
    Ok(())
}
