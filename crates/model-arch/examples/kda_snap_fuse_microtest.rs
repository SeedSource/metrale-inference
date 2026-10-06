// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Byte-identity gate and timing for the fused KDA verify snapshot
//! (`METRALE_GLM_KDA_SNAP_FUSE=1`, `Glm5NextKdaLayer::snap_walk`, kernels/gb10/common/
//! kda_snap_fuse.cu) against the walk with snapshot copies it replaces.
//!
//! Owner: model-arch examples (GLM-5.3 KDA).
//! Invariants:
//! - Exits 1, with `FAIL ...` lines, unless for every case and seed the two arms agree bit for
//!   bit on the layer output (`final_out`, BF16 `[k, hidden]`), the recurrent output (`core`,
//!   FP32 `[k, heads * 128]`), the live recurrent and conv states, every snapshot slot, and the
//!   state each accept outcome `n = 0..=k` of a `(k - 1)`-snapshot verify rolls back to
//!   (`n = 0`: the pre-verify checkpoint; `1 <= n < k`: slot `n - 1`; `n = k`: the live state;
//!   the engine's `rollback_ssm_states_dispatch` rule).
//! - Refuses PASS when the layer has no fused path (`snap_fuse_ready` false: a missing twin
//!   kernel would otherwise pass as the copy walk compared with itself), when nothing was
//!   compared, when the reference outputs are all zero or not finite, or when a row left the
//!   state unchanged (each snapshot must differ from the state before it).
//! - Each arm's outputs and slots are filled with its own poison byte first (0xAB reference,
//!   0xCD fused), so a byte either side leaves unwritten cannot pass as a match.
//!
//! Arms, on one bound layer, one workspace, the same hidden rows and starting states:
//! - COPY: `decode_k_snap_arm(.., fuse = false)`: `decode_k` with the lever off, the per-row
//!   walk followed by two device-to-device copies per snapshot row.
//! - FUSED: `decode_k_snap_arm(.., fuse = true)`: the walk with out-of-place row kernels.
//!
//! Timing (never affects the verdict): CUDA events around one `decode_k` call on stream 0,
//! 10 warm-up calls, median of 60, for COPY, FUSED and a snapshot-free call, plus the
//! `2 * (k - 1)` copies alone; reported per layer and scaled to 34 KDA layers per step.
//!
//! Geometry: GLM-5.3-Flash KDA at the TP=2 per-rank head count (hidden 6144, 32 heads x 128,
//! conv 4), synthetic LCG weights.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example kda_snap_fuse_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-06: CUDA driver event API for timing, declared as in `dsa_mla_split_microtest`.
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
const MAX_ROWS: usize = 4;
/// 2026-10-06: `(rows, snapshots)`. `(3, 2)` is the K=3 MTP verify the engine runs (`k - 1`
/// snapshots); `(1, 0)` a single-row verify; the rest cover fewer, more and full-length lists.
const CASES: [(usize, usize); 8] = [
    (1, 0),
    (2, 1),
    (3, 2),
    (4, 3),
    (3, 0),
    (3, 1),
    (3, 3),
    (1, 1),
];
const SEEDS: [u64; 2] = [0x5A4F_0001, 0x5A4F_0002];
const POISON_COPY: u8 = 0xAB;
const POISON_FUSED: u8 = 0xCD;
const WARMUP: usize = 10;
const TIMED: usize = 60;
/// 2026-10-06: `layers_kda` in kernels/gb10/glm-5.3-flash/MODEL.toml.
const KDA_LAYERS: f64 = 34.0;
const TIME_CASES: [usize; 2] = [3, 4];

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

/// 2026-10-06: Synthetic weights: projections uniform in `±1/sqrt(fan_in)`, a mild decay
/// (as `flashkda_prefill_microtest`'s MILD profile).
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

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    cfg: Glm5NextKdaConfig,
}

/// 2026-10-06: One arm's results as raw bytes.
struct Captured {
    final_out: Vec<u8>,
    core: Vec<u8>,
    h: Vec<u8>,
    conv: Vec<u8>,
    ckpt_h: Vec<u8>,
    ckpt_conv: Vec<u8>,
    slots: Vec<(Vec<u8>, Vec<u8>)>,
}

impl Captured {
    /// 2026-10-06: The `(h, conv)` an accept of `n` rows restores, by the engine's rule.
    fn accepted(&self, n: usize, k: usize) -> (&[u8], &[u8]) {
        if n == 0 {
            (self.ckpt_h.as_slice(), self.ckpt_conv.as_slice())
        } else if n < k {
            (self.slots[n - 1].0.as_slice(), self.slots[n - 1].1.as_slice())
        } else {
            (self.h.as_slice(), self.conv.as_slice())
        }
    }
}

impl Rig<'_> {
    fn bytes(&self) -> (usize, usize) {
        (
            self.cfg.recurrent_state_elems() * 4,
            self.cfg.conv_state_elems() * 4,
        )
    }

    /// 2026-10-06: One arm: upload the starting states, poison the outputs and `nsnap` slots,
    /// take the engine's pre-verify checkpoint, run `decode_k_snap_arm`, read everything back.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        hidden: DevicePtr,
        k: usize,
        nsnap: usize,
        h0: &[f32],
        c0: &[f32],
        fuse: bool,
        poison: u8,
    ) -> Result<Captured> {
        let (g, c, ws) = (self.g, &self.cfg, &self.ws);
        let (hb, cb) = self.bytes();
        let st = KdaSeqState {
            conv: up_f32(g, c0)?,
            recurrent: up_f32(g, h0)?,
        };
        let ckpt = (up_bytes(g, &vec![poison; hb])?, up_bytes(g, &vec![poison; cb])?);
        let mut slots = Vec::with_capacity(nsnap);
        for _ in 0..nsnap {
            slots.push((up_bytes(g, &vec![poison; hb])?, up_bytes(g, &vec![poison; cb])?));
        }
        g.copy_h2d(&vec![poison; k * c.hidden * 2], ws.final_out)?;
        g.copy_h2d(&vec![poison; k * c.qkv_dim() * 4], ws.core)?;
        g.copy_d2d_async(st.recurrent, ckpt.0, hb, 0)?;
        g.copy_d2d_async(st.conv, ckpt.1, cb, 0)?;
        self.layer
            .decode_k_snap_arm(g, hidden, k, &st, ws, &slots, fuse, 0)?;
        g.synchronize(0)?;
        let mut cap = Captured {
            final_out: dn_bytes(g, ws.final_out, k * c.hidden * 2)?,
            core: dn_bytes(g, ws.core, k * c.qkv_dim() * 4)?,
            h: dn_bytes(g, st.recurrent, hb)?,
            conv: dn_bytes(g, st.conv, cb)?,
            ckpt_h: dn_bytes(g, ckpt.0, hb)?,
            ckpt_conv: dn_bytes(g, ckpt.1, cb)?,
            slots: Vec::with_capacity(nsnap),
        };
        for &(sh, sc) in &slots {
            cap.slots.push((dn_bytes(g, sh, hb)?, dn_bytes(g, sc, cb)?));
        }
        for p in [st.conv, st.recurrent, ckpt.0, ckpt.1] {
            let _ = g.free(p);
        }
        for (sh, sc) in slots {
            let _ = g.free(sh);
            let _ = g.free(sc);
        }
        Ok(cap)
    }
}

/// 2026-10-06: `(bytes compared, first mismatching byte)`; a length mismatch compares nothing.
fn compare(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    if a.len() != b.len() {
        return (0, Some(0));
    }
    (a.len(), a.iter().zip(b).position(|(x, y)| x != y))
}

/// 2026-10-06: All finite and not all zero, read as `width`-byte floats (2 = BF16, 4 = FP32).
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

/// 2026-10-06: One case and seed; prints one line per check group, returns whether it held
/// and adds the compared bytes to `total`.
fn run_case(rig: &Rig, k: usize, nsnap: usize, seed: u64, total: &mut usize) -> Result<bool> {
    let (g, c) = (rig.g, &rig.cfg);
    let mut r = Lcg(seed ^ ((k as u64) << 8) ^ nsnap as u64);
    let hidden = up_bf16(g, &r.vec(k * c.hidden, 1.0))?;
    let h0 = r.vec(c.recurrent_state_elems(), 0.1);
    let c0 = r.vec(c.conv_state_elems(), 0.3);
    let a = rig.run(hidden, k, nsnap, &h0, &c0, false, POISON_COPY)?;
    let b = rig.run(hidden, k, nsnap, &h0, &c0, true, POISON_FUSED)?;
    let _ = g.free(hidden);

    let mut checks: Vec<(String, &[u8], &[u8])> = vec![
        ("final_out".into(), a.final_out.as_slice(), b.final_out.as_slice()),
        ("core".into(), a.core.as_slice(), b.core.as_slice()),
        ("h_state".into(), a.h.as_slice(), b.h.as_slice()),
        ("conv_state".into(), a.conv.as_slice(), b.conv.as_slice()),
    ];
    for t in 0..nsnap {
        let (x, y) = (&a.slots[t], &b.slots[t]);
        checks.push((format!("slot{t}_h"), x.0.as_slice(), y.0.as_slice()));
        checks.push((format!("slot{t}_conv"), x.1.as_slice(), y.1.as_slice()));
    }
    // 2026-10-06: Accept outcomes only where the engine's rule applies: `k - 1` snapshots.
    if nsnap + 1 == k {
        for n in 0..=k {
            let (ah, ac) = a.accepted(n, k);
            let (bh, bc) = b.accepted(n, k);
            checks.push((format!("accept{n}_h"), ah, bh));
            checks.push((format!("accept{n}_conv"), ac, bc));
        }
    }

    // 2026-10-06: Non-vacuity on the reference arm: live outputs, and every row moved both
    // states (state after row t differs from the state before it).
    let mut seq_h: Vec<&[u8]> = vec![a.ckpt_h.as_slice()];
    let mut seq_c: Vec<&[u8]> = vec![a.ckpt_conv.as_slice()];
    for t in 0..nsnap.min(k.saturating_sub(1)) {
        seq_h.push(a.slots[t].0.as_slice());
        seq_c.push(a.slots[t].1.as_slice());
    }
    if nsnap + 1 >= k {
        seq_h.push(a.h.as_slice());
        seq_c.push(a.conv.as_slice());
    }
    let moved = seq_h.windows(2).all(|w| w[0] != w[1])
        && seq_c.windows(2).all(|w| w[0] != w[1])
        && a.h != a.ckpt_h
        && a.conv != a.ckpt_conv;
    let outs_live = live(&a.final_out, 2) && live(&a.core, 4);
    let vacuous = !(moved && outs_live);

    let mut ok = !vacuous;
    let mut line = format!("k={k} snaps={nsnap} seed={seed:#x}");
    let mut bad = Vec::new();
    for (name, x, y) in &checks {
        let (n, first) = compare(x, y);
        *total += n;
        if n == 0 || first.is_some() {
            ok = false;
            bad.push(format!("{name}: {n} bytes, first mismatch at byte {first:?}"));
        }
    }
    line.push_str(&format!(" groups={} bytes_equal={}", checks.len(), bad.is_empty()));
    if ok {
        println!("{line} ok");
    } else {
        println!("{line}");
        for m in &bad {
            println!("FAIL k={k} snaps={nsnap} seed={seed:#x} {m}");
        }
        if vacuous {
            println!(
                "FAIL k={k} snaps={nsnap} seed={seed:#x} vacuous: moved={moved} \
                 outputs_live={outs_live}"
            );
        }
    }
    Ok(ok)
}

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what}: CUDA status {rc}");
    }
    Ok(())
}

/// 2026-10-06: Median microseconds of `f` over `TIMED` calls, each between two CUDA events on
/// stream 0, after `WARMUP` calls.
fn time_us(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(0)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    ck(unsafe { cuEventCreate(&mut e0, 0) }, "cuEventCreate(start)")?;
    ck(unsafe { cuEventCreate(&mut e1, 0) }, "cuEventCreate(end)")?;
    let mut us = Vec::with_capacity(TIMED);
    for _ in 0..TIMED {
        ck(unsafe { cuEventRecord(e0, 0) }, "cuEventRecord(start)")?;
        f()?;
        ck(unsafe { cuEventRecord(e1, 0) }, "cuEventRecord(end)")?;
        ck(unsafe { cuEventSynchronize(e1) }, "cuEventSynchronize")?;
        let mut ms = 0f32;
        ck(unsafe { cuEventElapsedTime(&mut ms, e0, e1) }, "cuEventElapsedTime")?;
        us.push(ms as f64 * 1e3);
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    us.sort_by(f64::total_cmp);
    Ok(us[us.len() / 2])
}

/// 2026-10-06: `TIMING` for a `k`-row verify with `k - 1` snapshots. Repeated calls advance the
/// states; the work per call does not change.
fn timing(rig: &Rig, k: usize) -> Result<()> {
    let (g, c, ws) = (rig.g, &rig.cfg, &rig.ws);
    let (hb, cb) = rig.bytes();
    let mut r = Lcg(SEEDS[0] ^ 0x7131);
    let hidden = up_bf16(g, &r.vec(k * c.hidden, 1.0))?;
    let st = KdaSeqState {
        conv: up_f32(g, &r.vec(c.conv_state_elems(), 0.3))?,
        recurrent: up_f32(g, &r.vec(c.recurrent_state_elems(), 0.1))?,
    };
    let mut slots = Vec::new();
    for _ in 0..k - 1 {
        slots.push((g.alloc(hb)?, g.alloc(cb)?));
    }
    let layer = &rig.layer;
    let copy = time_us(g, || layer.decode_k_snap_arm(g, hidden, k, &st, ws, &slots, false, 0))?;
    let fused = time_us(g, || layer.decode_k_snap_arm(g, hidden, k, &st, ws, &slots, true, 0))?;
    let none = time_us(g, || layer.decode_k_snap_arm(g, hidden, k, &st, ws, &[], false, 0))?;
    let copies = time_us(g, || {
        for &(sh, sc) in &slots {
            g.copy_d2d_async(st.recurrent, sh, hb, 0)?;
            g.copy_d2d_async(st.conv, sc, cb, 0)?;
        }
        Ok(())
    })?;
    println!(
        "TIMING k={k} snaps={} per_layer_us copy={copy:.1} fused={fused:.1} nosnap={none:.1} \
         copies_alone={copies:.1} | snapshot_cost_us copy={:.1} fused={:.1} | \
         saved_per_layer_us={:.1} saved_per_step_ms(x{KDA_LAYERS})={:.3}",
        k - 1,
        copy - none,
        fused - none,
        copy - fused,
        (copy - fused) * KDA_LAYERS / 1e3
    );
    for (sh, sc) in slots {
        let _ = g.free(sh);
        let _ = g.free(sc);
    }
    for p in [hidden, st.conv, st.recurrent] {
        let _ = g.free(p);
    }
    Ok(())
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let c = cfg();
    let kernels = Glm5NextKdaKernels::resolve(g)?;
    let layer = Glm5NextKdaLayer::new(0, c, synthetic_weights(g, &c)?, kernels)?;
    let ws = Glm5NextKdaWorkspace::new(g, &c, MAX_ROWS)?;
    let rig = Rig { g, layer, ws, cfg: c };

    // 2026-10-06: Without the fused path the FUSED arm silently runs the copy walk.
    if !rig.layer.snap_fuse_ready() {
        println!(
            "FAIL kda_snap_fuse_microtest: no fused path (has_snap_fuse={}, recurrent_smem={}, \
             METRALE_GLM_KDA_NO_SMEM must be unset)",
            rig.layer.kernels.has_snap_fuse(),
            rig.layer.kernels.recurrent_smem.0 != 0
        );
        std::process::exit(1);
    }

    let mut all_ok = true;
    let mut total = 0usize;
    let mut cases = 0usize;
    for &(k, nsnap) in &CASES {
        for &seed in &SEEDS {
            all_ok &= run_case(&rig, k, nsnap, seed, &mut total)?;
            cases += 1;
        }
    }
    for k in TIME_CASES {
        timing(&rig, k)?;
    }

    if total == 0 {
        println!("FAIL kda_snap_fuse_microtest: compared 0 bytes; this run proves nothing");
        std::process::exit(1);
    }
    if !all_ok {
        println!(
            "FAIL kda_snap_fuse_microtest: fused snapshot differs from the copy walk (see the \
             lines above); keep METRALE_GLM_KDA_SNAP_FUSE off"
        );
        std::process::exit(1);
    }
    println!(
        "PASS: kda_snap_fuse_microtest {cases} cases, {total} bytes bit-identical (outputs, live \
         states, snapshot slots, accept outcomes) between the copy walk and the fused snapshot"
    );
    Ok(())
}
