// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate and timing for the KDA recurrent state stored as FP16
//! (`METRALE_GLM_KDA_STATE_F16=1`, crates/model-arch/src/glm5next_kda/state_f16.rs,
//! kernels/gb10/common/kda_state_f16.cu).
//!
//! Owner: model-arch examples (GLM-5.3 KDA).
//! Invariants:
//! - Every check prints one line starting `PASS ` or `FAIL ` with, per output, the cosine, the
//!   max relative error (max |a - b| over max |b|), the nonfinite count of both sides and, where
//!   the claim is exactness, the number of mismatching u16/u32 words. The last line is `PASS all`
//!   only when every check passed; otherwise `FAIL all` and exit 1.
//! - A check refuses PASS when it compared nothing, when its reference output is all zero or not
//!   finite, or when a row left the state unchanged. Each arm's output buffers start at its own
//!   poison byte (0xAB / 0xCD), so a byte either side leaves unwritten cannot pass.
//! - The run refuses (FAIL) when the layer cannot take the FP16 twins (`state_f16_ready`) or the
//!   fused walk (`snap_fuse_ready`, `defer_walk_ready`): a missing kernel must not pass as a
//!   fallback compared with itself.
//!
//! Checks (one bound layer, synthetic weights, GLM-5.3-Flash TP2 per-rank geometry):
//! 1. Single row, each FP16 twin against its FP32 parent: the parent runs on the widened FP16
//!    state, its written state narrowed by `ssm_h_state_f32_to_f16` (RNE) must equal the twin's
//!    written state bit for bit (u16), and the outputs (`core` FP32, `final_out` BF16, conv
//!    state) must be equal bit for bit. `_f16` (in place, `decode`), `_io_f16` (row 0 of a fused
//!    2-row verify, into snapshot 0) and `_defer_f16` (row 0 of `decode_k_defer` reading
//!    intermediate 1). The device narrowing is also checked against a host RNE.
//! 2. W = 3 and W = 9 rows with W - 1 snapshots, FP16 state: the fused walk
//!    (`decode_k_snap_arm(fuse = true)`, `_io_f16` rows) equals the serial walk of single-row
//!    in-place `_f16` launches with snapshot copies (`fuse = false`), bit for bit, on the outputs,
//!    the live states and every snapshot slot.
//! 3. W = 3 and W = 9, FP16 state, deferred source -1, 0 (input and output aliased) and 1:
//!    `decode_k_defer` equals the commit copy (intermediate -> live) followed by the fused walk,
//!    bit for bit, on the outputs, the live states and every intermediate.
//! 4. Speed (never part of correctness): the fused walk alone (`snap_walk_arm`: conv + recurrent
//!    row launches, the verify's W - 1 snapshots) over 34 layers' worth of distinct states, FP32
//!    (old) against FP16 (new), at W = 1, 2, 3, 5, 7, 9 (the DFlash verify width is adaptive:
//!    ~2.6 rows on prose, ~5 on code, ~7 on structured text). Cold: 4 rotating copies of the
//!    34-layer state set (>= 4x the 24 MB L2 at the smallest width). CUDA events around each
//!    34-layer step, 3 warm-up steps, median of 21. Implied GB/s counts the state reads and writes
//!    plus each row's conv and recurrent inputs and outputs; a value above 255 GB/s prints VOID
//!    and fails the check.
//! 5. Nonfinite: every output of checks 1-3 has 0 nonfinite values (counted per line and in
//!    total).
//! Info (no verdict): the FP16 W = 9 fused walk against the FP32 one from the same widened
//! start, the numerics change the lever makes.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm_kda_state_f16_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, bail};
use half::{bf16, f16};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::layer::SsmDeferSrc;
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-12: CUDA driver event API for timing, declared as in `kda_snap_fuse_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const D: usize = 128;
const D_CONV: usize = 4;
const HIDDEN: usize = 6144;
const HEADS: usize = 32;
/// 2026-10-12: The widest walk checked (W = 9).
const MAX_ROWS: usize = 9;
/// 2026-10-12: Verify intermediates per slot (the engine keeps `k - 1`).
const N_INTER: usize = MAX_ROWS - 1;
const SEEDS: [u64; 2] = [0xF16A_0001, 0xF16A_0002];
const BIT_WIDTHS: [usize; 2] = [3, 9];
const SPEED_WIDTHS: [usize; 6] = [1, 2, 3, 5, 7, 9];
const DEFER_IDX: [i32; 3] = [-1, 0, 1];
const POISON_A: u8 = 0xAB;
const POISON_B: u8 = 0xCD;
/// 2026-10-12: `layers_kda` in kernels/gb10/glm-5.3-flash/MODEL.toml.
const KDA_LAYERS: usize = 34;
const COPIES: usize = 4;
const WARMUP: usize = 3;
const TIMED: usize = 21;
const VOID_GBS: f64 = 255.0;

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
fn f32_bytes(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    up_bytes(g, &f32_bytes(d))
}
/// 2026-10-12: Host FP32 -> FP16 as `gdn_f16_store`: clamp to +-65504, round to nearest even.
fn q16(x: f32) -> u16 {
    f16::from_f32(x.clamp(-65504.0, 65504.0)).to_bits()
}
fn f16_bytes(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| q16(*x).to_le_bytes()).collect()
}
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
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

/// 2026-10-12: Synthetic weights, as `kda_snap_fuse_microtest`.
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
        dt_bias: up_f32(
            g,
            &(0..qkv).map(|_| r.r(-4.0, -2.0) as f32).collect::<Vec<_>>(),
        )?,
        a_log: up_f32(g, &(0..h).map(|_| r.r(0.0, 0.3) as f32).collect::<Vec<_>>())?,
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

/// 2026-10-12: Element format of a compared buffer.
#[derive(Clone, Copy)]
enum Fmt {
    Bf16,
    F16,
    F32,
}

fn decode(b: &[u8], f: Fmt) -> Vec<f32> {
    match f {
        Fmt::Bf16 => b
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        Fmt::F16 => b
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        Fmt::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

/// 2026-10-12: Mismatching words (u16 for 2-byte formats, u32 for FP32).
fn word_mismatches(a: &[u8], b: &[u8], f: Fmt) -> usize {
    let w = if matches!(f, Fmt::F32) { 4 } else { 2 };
    a.chunks_exact(w)
        .zip(b.chunks_exact(w))
        .filter(|(x, y)| x != y)
        .count()
}

/// 2026-10-12: The per-output statistics every check line prints.
struct Stats {
    n: usize,
    cos: f64,
    max_rel: f64,
    nonfinite: usize,
    mismatch: usize,
    ref_live: bool,
}

fn stats(a: &[u8], b: &[u8], f: Fmt) -> Stats {
    let (x, y) = (decode(a, f), decode(b, f));
    let nonfinite =
        x.iter().filter(|v| !v.is_finite()).count() + y.iter().filter(|v| !v.is_finite()).count();
    let (mut dot, mut nx, mut ny, mut dmax, mut ymax) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&p, &q) in x.iter().zip(&y) {
        let (p, q) = (p as f64, q as f64);
        dot += p * q;
        nx += p * p;
        ny += q * q;
        dmax = dmax.max((p - q).abs());
        ymax = ymax.max(q.abs());
    }
    let cos = if nx > 0.0 && ny > 0.0 {
        dot / (nx.sqrt() * ny.sqrt())
    } else {
        0.0
    };
    Stats {
        n: if a.len() == b.len() { x.len() } else { 0 },
        cos,
        max_rel: if ymax > 0.0 {
            dmax / ymax
        } else {
            f64::INFINITY
        },
        nonfinite,
        mismatch: if a.len() == b.len() {
            word_mismatches(a, b, f)
        } else {
            usize::MAX
        },
        ref_live: !y.is_empty() && y.iter().all(|v| v.is_finite()) && y.iter().any(|v| *v != 0.0),
    }
}

/// 2026-10-12: Tallies of the run.
#[derive(Default)]
struct Tally {
    checks: usize,
    failed: usize,
    values: usize,
    nonfinite: usize,
}

impl Tally {
    /// 2026-10-12: One bitwise check line: `a` (new) against `b` (reference).
    fn bitwise(&mut self, name: &str, a: &[u8], b: &[u8], f: Fmt) -> bool {
        let s = stats(a, b, f);
        self.checks += 1;
        self.values += s.n;
        self.nonfinite += s.nonfinite;
        let ok = s.n > 0 && s.ref_live && s.nonfinite == 0 && s.mismatch == 0;
        if !ok {
            self.failed += 1;
        }
        println!(
            "{} {name} n={} cos={:.9} max_rel={:.3e} nonfinite={} word_mismatch={} ref_live={}",
            if ok { "PASS" } else { "FAIL" },
            s.n,
            s.cos,
            s.max_rel,
            s.nonfinite,
            s.mismatch,
            s.ref_live
        );
        ok
    }

    /// 2026-10-12: A non-vacuity condition (a state that must have moved, a ready kernel).
    fn require(&mut self, name: &str, ok: bool, detail: &str) -> bool {
        self.checks += 1;
        if !ok {
            self.failed += 1;
        }
        println!("{} {name} {detail}", if ok { "PASS" } else { "FAIL" });
        ok
    }
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    cfg: Glm5NextKdaConfig,
    /// 2026-10-12: Recurrent state bytes FP32 / FP16, conv state bytes.
    hb32: usize,
    hb16: usize,
    cb: usize,
}

/// 2026-10-12: What one arm produced.
struct Out {
    final_out: Vec<u8>,
    core: Vec<u8>,
    h: Vec<u8>,
    conv: Vec<u8>,
    slots_h: Vec<u8>,
    slots_conv: Vec<u8>,
}

impl Rig<'_> {
    fn hb(&self, fp16: bool) -> usize {
        if fp16 { self.hb16 } else { self.hb32 }
    }

    /// 2026-10-12: A state from host values: FP16 bits when `f16`, else FP32 (the values given;
    /// callers pass FP16-representable values when the arms must start equal).
    fn state(&self, h: &[f32], c: &[f32], fp16: bool) -> Result<KdaSeqState> {
        let g = self.g;
        Ok(KdaSeqState {
            conv: up_f32(g, c)?,
            recurrent: if fp16 {
                up_bytes(g, &f16_bytes(h))?
            } else {
                up_f32(g, h)?
            },
            recurrent_f16: fp16,
        })
    }

    fn poison_outputs(&self, rows: usize, poison: u8) -> Result<()> {
        let (g, c, ws) = (self.g, &self.cfg, &self.ws);
        g.copy_h2d(&vec![poison; rows * c.hidden * 2], ws.final_out)?;
        g.copy_h2d(&vec![poison; rows * c.qkv_dim() * 4], ws.core)
    }

    fn free_state(&self, st: &KdaSeqState) {
        let _ = self.g.free(st.conv);
        let _ = self.g.free(st.recurrent);
    }

    /// 2026-10-12: Read back an arm: outputs of `rows` rows, the live state, `nsnap` slots.
    fn read(&self, rows: usize, st: &KdaSeqState, slots: &[(DevicePtr, DevicePtr)]) -> Result<Out> {
        let (g, c, ws) = (self.g, &self.cfg, &self.ws);
        g.synchronize(0)?;
        let hb = self.hb(st.recurrent_f16);
        let mut o = Out {
            final_out: dn_bytes(g, ws.final_out, rows * c.hidden * 2)?,
            core: dn_bytes(g, ws.core, rows * c.qkv_dim() * 4)?,
            h: dn_bytes(g, st.recurrent, hb)?,
            conv: dn_bytes(g, st.conv, self.cb)?,
            slots_h: Vec::new(),
            slots_conv: Vec::new(),
        };
        for &(sh, sc) in slots {
            o.slots_h.extend(dn_bytes(g, sh, hb)?);
            o.slots_conv.extend(dn_bytes(g, sc, self.cb)?);
        }
        Ok(o)
    }

    /// 2026-10-12: Narrow an FP32 state blob (host bytes) on the device
    /// (`ssm_h_state_f32_to_f16`, the prefill staging's narrowing).
    fn narrow_dev(&self, h32: &[u8]) -> Result<Vec<u8>> {
        let g = self.g;
        let src = up_bytes(g, h32)?;
        let dst = up_bytes(g, &vec![0u8; h32.len() / 2])?;
        self.layer.narrow_state(g, src, dst, 0)?;
        g.synchronize(0)?;
        let out = dn_bytes(g, dst, h32.len() / 2)?;
        let _ = g.free(src);
        let _ = g.free(dst);
        Ok(out)
    }

    fn alloc_slots(&self, n: usize, fp16: bool, poison: u8) -> Result<Vec<(DevicePtr, DevicePtr)>> {
        let g = self.g;
        (0..n)
            .map(|_| {
                Ok((
                    up_bytes(g, &vec![poison; self.hb(fp16)])?,
                    up_bytes(g, &vec![poison; self.cb])?,
                ))
            })
            .collect()
    }

    fn free_slots(&self, slots: Vec<(DevicePtr, DevicePtr)>) {
        for (h, c) in slots {
            let _ = self.g.free(h);
            let _ = self.g.free(c);
        }
    }
}

/// 2026-10-12: Starting values: an FP16-representable recurrent state (so the FP32 parent and
/// the FP16 twin start from the same numbers), a conv state, `rows` hidden rows.
fn inputs(rig: &Rig, seed: u64, rows: usize) -> Result<(DevicePtr, Vec<f32>, Vec<f32>)> {
    let c = &rig.cfg;
    let mut r = Lcg(seed);
    let hidden = up_bf16(rig.g, &r.vec(rows * c.hidden, 1.0))?;
    let h0: Vec<f32> = r
        .vec(c.recurrent_state_elems(), 0.1)
        .into_iter()
        .map(|x| f16::from_bits(q16(x)).to_f32())
        .collect();
    let c0 = r.vec(c.conv_state_elems(), 0.3);
    Ok((hidden, h0, c0))
}

/// 2026-10-12: Check 1: each FP16 twin against its FP32 parent plus the RNE narrowing.
fn check_single_row(rig: &Rig, t: &mut Tally, seed: u64) -> Result<()> {
    let (g, c, ws) = (rig.g, &rig.cfg, &rig.ws);
    let core_row = c.qkv_dim() * 4;
    let tag = format!("seed={seed:#x}");

    // 2026-10-12: 1a: `decode` (stateful_row): `_f16` in place against the FP32 smem kernel.
    let (hidden, h0, c0) = inputs(rig, seed ^ 0x1A, 1)?;
    let s32 = rig.state(&h0, &c0, false)?;
    rig.poison_outputs(1, POISON_A)?;
    rig.layer.decode(g, hidden, &s32, ws, 0)?;
    let o32 = rig.read(1, &s32, &[])?;
    let s16 = rig.state(&h0, &c0, true)?;
    rig.poison_outputs(1, POISON_B)?;
    rig.layer.decode(g, hidden, &s16, ws, 0)?;
    let o16 = rig.read(1, &s16, &[])?;
    let want = rig.narrow_dev(&o32.h)?;
    let host: Vec<u8> = decode(&o32.h, Fmt::F32)
        .iter()
        .flat_map(|x| q16(*x).to_le_bytes())
        .collect();
    t.bitwise(
        &format!("1-narrow device_vs_host_rne {tag}"),
        &want,
        &host,
        Fmt::F16,
    );
    t.require(
        &format!("1a-moved {tag}"),
        o32.h != f32_bytes(&h0),
        "(the FP32 row changed the state)",
    );
    t.bitwise(&format!("1a-f16 h_state {tag}"), &o16.h, &want, Fmt::F16);
    t.bitwise(
        &format!("1a-f16 core {tag}"),
        &o16.core,
        &o32.core,
        Fmt::F32,
    );
    t.bitwise(
        &format!("1a-f16 final_out {tag}"),
        &o16.final_out,
        &o32.final_out,
        Fmt::Bf16,
    );
    t.bitwise(
        &format!("1a-f16 conv_state {tag}"),
        &o16.conv,
        &o32.conv,
        Fmt::F32,
    );
    rig.free_state(&s32);
    rig.free_state(&s16);
    let _ = g.free(hidden);

    // 2026-10-12: 1b: row 0 of a fused 2-row verify (`_io_f16` into snapshot 0).
    let (hidden, h0, c0) = inputs(rig, seed ^ 0x1B, 2)?;
    let s32 = rig.state(&h0, &c0, false)?;
    let sl32 = rig.alloc_slots(1, false, POISON_A)?;
    rig.poison_outputs(2, POISON_A)?;
    rig.layer
        .decode_k_snap_arm(g, hidden, 2, &s32, ws, &sl32, true, 0)?;
    let o32 = rig.read(2, &s32, &sl32)?;
    let s16 = rig.state(&h0, &c0, true)?;
    let sl16 = rig.alloc_slots(1, true, POISON_B)?;
    rig.poison_outputs(2, POISON_B)?;
    rig.layer
        .decode_k_snap_arm(g, hidden, 2, &s16, ws, &sl16, true, 0)?;
    let o16 = rig.read(2, &s16, &sl16)?;
    let want = rig.narrow_dev(&o32.slots_h)?;
    t.bitwise(
        &format!("1b-io_f16 snap0_h {tag}"),
        &o16.slots_h,
        &want,
        Fmt::F16,
    );
    t.bitwise(
        &format!("1b-io_f16 core_row0 {tag}"),
        &o16.core[..core_row],
        &o32.core[..core_row],
        Fmt::F32,
    );
    t.bitwise(
        &format!("1b-io_f16 snap0_conv {tag}"),
        &o16.slots_conv,
        &o32.slots_conv,
        Fmt::F32,
    );
    rig.free_state(&s32);
    rig.free_state(&s16);
    rig.free_slots(sl32);
    rig.free_slots(sl16);
    let _ = g.free(hidden);

    // 2026-10-12: 1c: row 0 of `decode_k_defer` reading intermediate 1 (`_defer_f16`), writing
    // snapshot 0 = intermediate 0.
    let (hidden, h0, c0) = inputs(rig, seed ^ 0x1C, 2)?;
    let mut r = Lcg(seed ^ 0x1C1C);
    let live_h: Vec<f32> = r.vec(c.recurrent_state_elems(), 0.1);
    let live_c = r.vec(c.conv_state_elems(), 0.3);
    let mut row0 = Vec::new();
    let mut core0 = Vec::new();
    for fp16 in [false, true] {
        let hb = rig.hb(fp16);
        let live = rig.state(&live_h, &live_c, fp16)?;
        let mut hbuf = vec![if fp16 { POISON_B } else { POISON_A }; hb];
        hbuf.extend(if fp16 { f16_bytes(&h0) } else { f32_bytes(&h0) });
        let h_inter = up_bytes(g, &hbuf)?;
        let mut cbuf = vec![if fp16 { POISON_B } else { POISON_A }; rig.cb];
        cbuf.extend(f32_bytes(&c0));
        let c_inter = up_bytes(g, &cbuf)?;
        let word = up_bytes(g, &1i32.to_le_bytes())?;
        let src = SsmDeferSrc {
            idx: word,
            h_base: h_inter,
            h_stride: hb,
            conv_base: c_inter,
            conv_stride: rig.cb,
        };
        rig.poison_outputs(2, if fp16 { POISON_B } else { POISON_A })?;
        rig.layer
            .decode_k_defer(g, hidden, 2, &live, ws, &[(h_inter, c_inter)], &src, 0)?;
        g.synchronize(0)?;
        row0.push(dn_bytes(g, h_inter, hb)?);
        core0.push(dn_bytes(g, ws.core, core_row)?);
        rig.free_state(&live);
        for p in [h_inter, c_inter, word] {
            let _ = g.free(p);
        }
    }
    let want = rig.narrow_dev(&row0[0])?;
    t.bitwise(
        &format!("1c-defer_f16 snap0_h {tag}"),
        &row0[1],
        &want,
        Fmt::F16,
    );
    t.bitwise(
        &format!("1c-defer_f16 core_row0 {tag}"),
        &core0[1],
        &core0[0],
        Fmt::F32,
    );
    let _ = g.free(hidden);
    Ok(())
}

/// 2026-10-12: Check 2: FP16 fused walk against the serial walk of single-row FP16 launches.
fn check_fused_walk(rig: &Rig, t: &mut Tally, w: usize, seed: u64) -> Result<()> {
    let g = rig.g;
    let tag = format!("W={w} seed={seed:#x}");
    let (hidden, h0, c0) = inputs(rig, seed ^ ((w as u64) << 8), w)?;
    let mut outs = Vec::new();
    for (fuse, poison) in [(false, POISON_A), (true, POISON_B)] {
        let st = rig.state(&h0, &c0, true)?;
        let slots = rig.alloc_slots(w - 1, true, poison)?;
        rig.poison_outputs(w, poison)?;
        rig.layer
            .decode_k_snap_arm(g, hidden, w, &st, &rig.ws, &slots, fuse, 0)?;
        outs.push(rig.read(w, &st, &slots)?);
        rig.free_state(&st);
        rig.free_slots(slots);
    }
    let (serial, fused) = (&outs[0], &outs[1]);
    // 2026-10-12: Every row moved the state: start, slots, live all pairwise distinct in order.
    let start = f16_bytes(&h0);
    let mut seq: Vec<&[u8]> = vec![start.as_slice()];
    seq.extend(serial.slots_h.chunks_exact(rig.hb16));
    seq.push(serial.h.as_slice());
    t.require(
        &format!("2-moved {tag}"),
        seq.windows(2).all(|p| p[0] != p[1]),
        "(each serial row changed the FP16 state)",
    );
    t.bitwise(
        &format!("2-fused final_out {tag}"),
        &fused.final_out,
        &serial.final_out,
        Fmt::Bf16,
    );
    t.bitwise(
        &format!("2-fused core {tag}"),
        &fused.core,
        &serial.core,
        Fmt::F32,
    );
    t.bitwise(
        &format!("2-fused h_state {tag}"),
        &fused.h,
        &serial.h,
        Fmt::F16,
    );
    t.bitwise(
        &format!("2-fused conv_state {tag}"),
        &fused.conv,
        &serial.conv,
        Fmt::F32,
    );
    t.bitwise(
        &format!("2-fused slots_h {tag}"),
        &fused.slots_h,
        &serial.slots_h,
        Fmt::F16,
    );
    t.bitwise(
        &format!("2-fused slots_conv {tag}"),
        &fused.slots_conv,
        &serial.slots_conv,
        Fmt::F32,
    );
    let _ = g.free(hidden);
    Ok(())
}

/// 2026-10-12: Check 3: `decode_k_defer` against the commit copy plus the fused walk, FP16.
fn check_defer(rig: &Rig, t: &mut Tally, w: usize, idx: i32, seed: u64) -> Result<()> {
    let (g, c) = (rig.g, &rig.cfg);
    let tag = format!("W={w} idx={idx} seed={seed:#x}");
    let (hidden, h0, c0) = inputs(rig, seed ^ ((w as u64) << 12) ^ (idx + 2) as u64, w)?;
    let mut r = Lcg(seed ^ 0x3D3D ^ (w as u64));
    let inter_h: Vec<f32> = r.vec(N_INTER * c.recurrent_state_elems(), 0.1);
    let inter_c: Vec<f32> = r.vec(N_INTER * c.conv_state_elems(), 0.3);
    let (hb, cb) = (rig.hb16, rig.cb);
    let mut outs = Vec::new();
    for (defer, poison) in [(false, POISON_A), (true, POISON_B)] {
        let live = rig.state(&h0, &c0, true)?;
        let h_inter = up_bytes(g, &f16_bytes(&inter_h))?;
        let c_inter = up_f32(g, &inter_c)?;
        let snaps: Vec<(DevicePtr, DevicePtr)> = (0..w - 1)
            .map(|t| (h_inter.offset(t * hb), c_inter.offset(t * cb)))
            .collect();
        rig.poison_outputs(w, poison)?;
        let word = up_bytes(g, &idx.to_le_bytes())?;
        if defer {
            let src = SsmDeferSrc {
                idx: word,
                h_base: h_inter,
                h_stride: hb,
                conv_base: c_inter,
                conv_stride: cb,
            };
            rig.layer
                .decode_k_defer(g, hidden, w, &live, &rig.ws, &snaps, &src, 0)?;
        } else {
            if let Ok(i) = usize::try_from(idx) {
                g.copy_d2d_async(h_inter.offset(i * hb), live.recurrent, hb, 0)?;
                g.copy_d2d_async(c_inter.offset(i * cb), live.conv, cb, 0)?;
            }
            rig.layer
                .decode_k_snap_arm(g, hidden, w, &live, &rig.ws, &snaps, true, 0)?;
        }
        let mut o = rig.read(w, &live, &[])?;
        o.slots_h = dn_bytes(g, h_inter, N_INTER * hb)?;
        o.slots_conv = dn_bytes(g, c_inter, N_INTER * cb)?;
        outs.push(o);
        rig.free_state(&live);
        for p in [h_inter, c_inter, word] {
            let _ = g.free(p);
        }
    }
    let (copy, defer) = (&outs[0], &outs[1]);
    t.require(
        &format!("3-moved {tag}"),
        copy.h != f16_bytes(&h0) && copy.slots_h != f16_bytes(&inter_h),
        "(the reference walk changed the live state and the intermediates)",
    );
    t.bitwise(
        &format!("3-defer final_out {tag}"),
        &defer.final_out,
        &copy.final_out,
        Fmt::Bf16,
    );
    t.bitwise(
        &format!("3-defer core {tag}"),
        &defer.core,
        &copy.core,
        Fmt::F32,
    );
    t.bitwise(
        &format!("3-defer h_state {tag}"),
        &defer.h,
        &copy.h,
        Fmt::F16,
    );
    t.bitwise(
        &format!("3-defer conv_state {tag}"),
        &defer.conv,
        &copy.conv,
        Fmt::F32,
    );
    t.bitwise(
        &format!("3-defer inter_h {tag}"),
        &defer.slots_h,
        &copy.slots_h,
        Fmt::F16,
    );
    t.bitwise(
        &format!("3-defer inter_conv {tag}"),
        &defer.slots_conv,
        &copy.slots_conv,
        Fmt::F32,
    );
    let _ = g.free(hidden);
    Ok(())
}

/// 2026-10-12: Info: the FP16 W = 9 fused walk against the FP32 one from the same start.
fn info_drift(rig: &Rig) -> Result<()> {
    let (g, w) = (rig.g, MAX_ROWS);
    let (hidden, h0, c0) = inputs(rig, SEEDS[0] ^ 0x1F0, w)?;
    let mut outs = Vec::new();
    for fp16 in [false, true] {
        let st = rig.state(&h0, &c0, fp16)?;
        let slots = rig.alloc_slots(w - 1, fp16, POISON_A)?;
        rig.layer
            .decode_k_snap_arm(g, hidden, w, &st, &rig.ws, &slots, true, 0)?;
        outs.push(rig.read(w, &st, &slots)?);
        rig.free_state(&st);
        rig.free_slots(slots);
    }
    let h16_wide: Vec<u8> = f32_bytes(&decode(&outs[1].h, Fmt::F16));
    for (name, a, b, f) in [
        ("core", &outs[1].core, &outs[0].core, Fmt::F32),
        (
            "final_out",
            &outs[1].final_out,
            &outs[0].final_out,
            Fmt::Bf16,
        ),
        ("h_state", &h16_wide, &outs[0].h, Fmt::F32),
    ] {
        let s = stats(a, b, f);
        println!(
            "INFO drift W={w} fp16_vs_fp32 {name} n={} cos={:.9} max_rel={:.3e} nonfinite={}",
            s.n, s.cos, s.max_rel, s.nonfinite
        );
    }
    let _ = g.free(hidden);
    Ok(())
}

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what}: CUDA status {rc}");
    }
    Ok(())
}

/// 2026-10-12: Median microseconds of `f(step)` over `TIMED` steps, each between two CUDA events
/// on stream 0, after `WARMUP` steps.
fn time_us(mut f: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    for s in 0..WARMUP {
        f(s)?;
    }
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    ck(unsafe { cuEventCreate(&mut e0, 0) }, "cuEventCreate(start)")?;
    ck(unsafe { cuEventCreate(&mut e1, 0) }, "cuEventCreate(end)")?;
    let mut us = Vec::with_capacity(TIMED);
    for s in 0..TIMED {
        ck(unsafe { cuEventRecord(e0, 0) }, "cuEventRecord(start)")?;
        f(WARMUP + s)?;
        ck(unsafe { cuEventRecord(e1, 0) }, "cuEventRecord(end)")?;
        ck(unsafe { cuEventSynchronize(e1) }, "cuEventSynchronize")?;
        let mut ms = 0f32;
        ck(
            unsafe { cuEventElapsedTime(&mut ms, e0, e1) },
            "cuEventElapsedTime",
        )?;
        us.push(ms as f64 * 1e3);
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    us.sort_by(f64::total_cmp);
    Ok(us[us.len() / 2])
}

/// 2026-10-12: Bytes one W-row fused walk of one layer moves: each row reads and writes the
/// recurrent state (`hb`) and the conv state, reads its q|k|v (BF16), gate and beta (FP32) and
/// writes `core` (FP32); the conv reads the projection row and its weights and writes the
/// packed row. A lower bound (no L2 reuse across rows assumed for inputs only).
fn walk_bytes(c: &Glm5NextKdaConfig, w: usize, hb: usize, cb: usize) -> f64 {
    let (qkv, cd) = (c.qkv_dim(), c.conv_dim());
    let row = 2 * hb
        + 2 * cb
        + 3 * qkv * 2
        + qkv * 4
        + c.heads * 4
        + qkv * 4
        + cd * 2
        + cd * 2
        + cd * c.conv_kernel * 2;
    (w * row) as f64
}

/// 2026-10-12: Check 4: one speed row, old (FP32) against new (FP16), 34 layers per step.
fn speed_row(rig: &Rig, t: &mut Tally, w: usize) -> Result<()> {
    let (g, c) = (rig.g, &rig.cfg);
    let mut r = Lcg(0x5EED ^ w as u64);
    let hidden = up_bf16(g, &r.vec(w * c.hidden, 1.0))?;
    let h0 = r.vec(c.recurrent_state_elems(), 0.1);
    let c0 = r.vec(c.conv_state_elems(), 0.3);
    // 2026-10-12: One full call fills the workspace rows the walk reads (front end).
    let warm = rig.state(&h0, &c0, false)?;
    rig.layer
        .decode_k_snap_arm(g, hidden, w, &warm, &rig.ws, &[], false, 0)?;
    g.synchronize(0)?;
    rig.free_state(&warm);
    let mut res = Vec::new();
    for fp16 in [false, true] {
        let hb = rig.hb(fp16);
        let tmpl_h = rig.state(&h0, &c0, fp16)?;
        let mut sets: Vec<Vec<(KdaSeqState, Vec<(DevicePtr, DevicePtr)>)>> = Vec::new();
        for _ in 0..COPIES {
            let mut layers = Vec::with_capacity(KDA_LAYERS);
            for _ in 0..KDA_LAYERS {
                let st = KdaSeqState {
                    conv: g.alloc(rig.cb)?,
                    recurrent: g.alloc(hb)?,
                    recurrent_f16: fp16,
                };
                g.copy_d2d_async(tmpl_h.recurrent, st.recurrent, hb, 0)?;
                g.copy_d2d_async(tmpl_h.conv, st.conv, rig.cb, 0)?;
                let slots = rig.alloc_slots(w - 1, fp16, 0)?;
                layers.push((st, slots));
            }
            sets.push(layers);
        }
        g.synchronize(0)?;
        let us = time_us(|step| {
            for (st, slots) in &sets[step % COPIES] {
                rig.layer.snap_walk_arm(g, w, st, &rig.ws, slots, 0)?;
            }
            Ok(())
        })?;
        let bytes = walk_bytes(c, w, hb, rig.cb) * KDA_LAYERS as f64;
        res.push((us, bytes / (us * 1e-6) / 1e9));
        for layers in sets {
            for (st, slots) in layers {
                rig.free_state(&st);
                rig.free_slots(slots);
            }
        }
        rig.free_state(&tmpl_h);
    }
    let _ = g.free(hidden);
    let ((old_us, old_gbs), (new_us, new_gbs)) = (res[0], res[1]);
    let void = old_gbs > VOID_GBS || new_gbs > VOID_GBS;
    println!(
        "M={w} old_us={old_us:.1} new_us={new_us:.1} old_GBs={old_gbs:.1} new_GBs={new_gbs:.1} \
         speedup={:.3}x per_layer_old_us={:.2} per_layer_new_us={:.2} (34 KDA layers, fused \
         walk, {} snapshots, {COPIES} rotating copies, median of {TIMED}){}",
        old_us / new_us,
        old_us / KDA_LAYERS as f64,
        new_us / KDA_LAYERS as f64,
        w - 1,
        if void { " VOID" } else { "" }
    );
    t.require(
        &format!("4-speed M={w}"),
        old_us > 0.0 && new_us > 0.0,
        &format!("measured (VOID above {VOID_GBS} GB/s: {void}; a VOID row is excluded from the speed rule, not a failure: rows after the first are L2-warm)"),
    );
    Ok(())
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let c = cfg();
    let kernels = Glm5NextKdaKernels::resolve(g)?;
    let layer = Glm5NextKdaLayer::new(0, c, synthetic_weights(g, &c)?, kernels)?;
    let ws = Glm5NextKdaWorkspace::new(g, &c, MAX_ROWS)?;
    let rig = Rig {
        g,
        layer,
        ws,
        cfg: c,
        hb32: c.recurrent_state_elems() * 4,
        hb16: c.recurrent_state_elems() * 2,
        cb: c.conv_state_elems() * 4,
    };
    let mut t = Tally::default();

    // 2026-10-12: Without the twins or the fused walk the arms would compare a fallback with
    // itself (or bail mid-run); refuse up front.
    let ready = t.require(
        "0-ready",
        rig.layer.state_f16_ready() && rig.layer.snap_fuse_ready() && rig.layer.defer_walk_ready(),
        &format!(
            "state_f16_ready={} snap_fuse_ready={} defer_walk_ready={} (METRALE_GLM_KDA_NO_SMEM \
             must be unset)",
            rig.layer.state_f16_ready(),
            rig.layer.snap_fuse_ready(),
            rig.layer.defer_walk_ready()
        ),
    );
    if ready {
        for &seed in &SEEDS {
            check_single_row(&rig, &mut t, seed)?;
            for &w in &BIT_WIDTHS {
                check_fused_walk(&rig, &mut t, w, seed)?;
                for &idx in &DEFER_IDX {
                    check_defer(&rig, &mut t, w, idx, seed)?;
                }
            }
        }
        info_drift(&rig)?;
        for &w in &SPEED_WIDTHS {
            speed_row(&rig, &mut t, w)?;
        }
    }
    t.require(
        "5-nonfinite",
        t.values > 0 && t.nonfinite == 0,
        &format!("total={} over {} compared values", t.nonfinite, t.values),
    );
    if t.failed == 0 && t.checks > 1 {
        println!("PASS all ({} checks)", t.checks);
        Ok(())
    } else {
        println!("FAIL all ({} of {} checks failed)", t.failed, t.checks);
        std::process::exit(1);
    }
}
