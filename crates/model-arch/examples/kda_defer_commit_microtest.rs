// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Byte-identity gate for the deferred KDA commit (`METRALE_GLM_KDA_DEFER_COMMIT=1`,
//! `Glm5NextKdaLayer::decode_k_defer`, kernels/gb10/common/kda_snap_fuse.cu `_defer` twins)
//! against today's car (fused snapshots, `METRALE_GLM_KDA_SNAP_FUSE=1`, and a commit that copies).
//!
//! Owner: model-arch examples (GLM-5.3 KDA).
//! Invariants:
//! - Exits 1, with `FAIL ...` lines, unless both arms agree bit for bit, for every schedule and
//!   seed, on every op's layer output (`final_out`) and recurrent output (`core`), every saved
//!   snapshot, and at the end the live recurrent and conv states and every intermediate slot.
//! - Refuses PASS when the layer cannot run the deferred walk (`defer_walk_ready` false), when
//!   nothing was compared, when no verify consumed a deferred index `0` (the aliased in-place
//!   case) and `>= 1`, or when the reference outputs are all zero or not finite.
//! - Every arm-owned buffer (outputs, intermediates, snapshot) starts at its own poison byte
//!   (0xAB copy arm, 0xCD deferred arm), so a byte either arm leaves unwritten cannot pass. The
//!   one conv slot no verify uses (MAX_ROWS - 1) is checked to keep its poison instead.
//!
//! Arms, on one bound layer and workspace, one "pool slot": a live state, `MAX_ROWS - 1` h and
//! `MAX_ROWS` conv intermediates at a fixed stride (the engine's layout), a snapshot buffer, and
//! (deferred arm) one device `i32`. Each schedule is a list of ops on the same hidden rows:
//! - `Verify(k, a)`: COPY runs `decode_k_snap_arm(fuse = true)` with snapshots `0..k-1`, then the
//!   commit copies intermediate `a - 1` into the live state when `a < k`. DEFER writes the pending
//!   index (or -1) into the device word, runs `decode_k_defer`, and records `a - 1` as pending
//!   when `a < k` (the engine's `kda_defer_hooks.rs` order).
//! - `Decode`: one `decode` row. DEFER first materialises (copies intermediate `pending` into the
//!   live state and clears it), as every non-verify entry point of the engine does.
//! - `Save` / `Restore`: copy live -> snapshot / snapshot -> live, after the same materialise.
//! At the end DEFER materialises once more and both arms are read back.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example kda_defer_commit_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::layer::SsmDeferSrc;
use metrale_model_layers::weight_map::DenseWeight;

const D: usize = 128;
const D_CONV: usize = 4;
const HIDDEN: usize = 6144;
const HEADS: usize = 32;
const MAX_ROWS: usize = 4;
const SEEDS: [u64; 2] = [0xDEF0_0001, 0xDEF0_0002];
const POISON_COPY: u8 = 0xAB;
const POISON_DEFER: u8 = 0xCD;

#[derive(Clone, Copy, Debug)]
enum Op {
    Verify(usize, usize),
    Decode,
    Save,
    Restore,
}
use Op::{Decode, Restore, Save, Verify};

/// 2026-10-09: The schedules the gate runs: full accepts, minimal accepts (deferred index 0, the
/// in-place aliased row 0), mixed widths (a K=4 verify's index 2 read by a K=2 verify), plain
/// decodes mid-sequence, and a snapshot save and restore mid-sequence.
fn schedules() -> Vec<(&'static str, Vec<Op>)> {
    [
        ("full", "V3.3 V3.3 V4.4 V2.2"),
        ("minimal", "V3.1 V3.1 V2.1 V4.1 V3.1"),
        ("mixed", "V3.2 V3.1 V4.3 V2.1 V4.4 V4.2 V3.3 V3.1 V2.2"),
        ("decode_mid", "V3.1 D V3.2 D D V4.3 V3.3"),
        ("snapshot_mid", "V3.2 S D R V4.1 S R V3.2"),
    ]
    .into_iter()
    .map(|(name, ops)| (name, ops.split(' ').map(parse_op).collect()))
    .collect()
}

/// 2026-10-09: `Vk.a` = `Verify(k, a)`, `D` = `Decode`, `S` = `Save`, `R` = `Restore`.
fn parse_op(t: &str) -> Op {
    match t {
        "D" => Decode,
        "S" => Save,
        "R" => Restore,
        v => {
            let (k, a) = v[1..].split_once('.').expect("Vk.a");
            Verify(k.parse().expect("k"), a.parse().expect("a"))
        }
    }
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

/// 2026-10-09: Synthetic weights as in `kda_snap_fuse_microtest`.
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

/// 2026-10-09: One arm's "pool slot" (module header).
struct Slot {
    live: KdaSeqState,
    h_inter: DevicePtr,
    conv_inter: DevicePtr,
    snap: (DevicePtr, DevicePtr),
    word: DevicePtr,
    /// 2026-10-09: Host mirror of the deferred index and of what the device word holds.
    pending: Option<usize>,
    dev: i32,
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: Glm5NextKdaWorkspace,
    cfg: Glm5NextKdaConfig,
    hb: usize,
    cb: usize,
}

impl Rig<'_> {
    fn slot(&self, h0: &[f32], c0: &[f32], poison: u8) -> Result<Slot> {
        let (g, hb, cb) = (self.g, self.hb, self.cb);
        Ok(Slot {
            live: KdaSeqState {
                conv: up_f32(g, c0)?,
                recurrent: up_f32(g, h0)?,
                recurrent_f16: false,
            },
            h_inter: up_bytes(g, &vec![poison; (MAX_ROWS - 1) * hb])?,
            conv_inter: up_bytes(g, &vec![poison; MAX_ROWS * cb])?,
            snap: (
                up_bytes(g, &vec![poison; hb])?,
                up_bytes(g, &vec![poison; cb])?,
            ),
            word: up_bytes(g, &(-1i32).to_le_bytes())?,
            pending: None,
            dev: -1,
        })
    }

    fn snaps(&self, s: &Slot, k: usize) -> Vec<(DevicePtr, DevicePtr)> {
        (0..k - 1)
            .map(|t| {
                (
                    s.h_inter.offset(t * self.hb),
                    s.conv_inter.offset(t * self.cb),
                )
            })
            .collect()
    }

    /// 2026-10-09: The commit's copy of intermediate `idx` into the live state (both arms; the
    /// deferred arm runs it only to materialise).
    fn copy_inter_to_live(&self, s: &Slot, idx: usize) -> Result<()> {
        let g = self.g;
        g.copy_d2d_async(
            s.h_inter.offset(idx * self.hb),
            s.live.recurrent,
            self.hb,
            0,
        )?;
        g.copy_d2d_async(s.conv_inter.offset(idx * self.cb), s.live.conv, self.cb, 0)
    }

    fn materialise(&self, s: &mut Slot) -> Result<()> {
        if let Some(idx) = s.pending.take() {
            self.copy_inter_to_live(s, idx)?;
        }
        Ok(())
    }

    /// 2026-10-09: Run one op on one arm; appends what it produced to `out`. `consumed` collects
    /// the deferred indices a verify read.
    #[allow(clippy::too_many_arguments)]
    fn op(
        &self,
        s: &mut Slot,
        op: Op,
        hidden: DevicePtr,
        defer: bool,
        poison: u8,
        out: &mut Vec<(String, Vec<u8>)>,
        consumed: &mut Vec<usize>,
    ) -> Result<()> {
        let (g, c, ws) = (self.g, &self.cfg, &self.ws);
        let rows = match op {
            Verify(k, _) => k,
            Decode => 1,
            Save | Restore => 0,
        };
        if rows > 0 {
            g.copy_h2d(&vec![poison; rows * c.hidden * 2], ws.final_out)?;
            g.copy_h2d(&vec![poison; rows * c.qkv_dim() * 4], ws.core)?;
        }
        match op {
            Verify(k, a) => {
                let snaps = self.snaps(s, k);
                if defer {
                    let want = s.pending.map_or(-1, |i| i as i32);
                    if s.dev != want {
                        g.copy_h2d_async(&want.to_le_bytes(), s.word, 0)?;
                        s.dev = want;
                    }
                    let src = SsmDeferSrc {
                        idx: s.word,
                        h_base: s.h_inter,
                        h_stride: self.hb,
                        conv_base: s.conv_inter,
                        conv_stride: self.cb,
                    };
                    self.layer
                        .decode_k_defer(g, hidden, k, &s.live, ws, &snaps, &src, 0)?;
                    consumed.extend(s.pending.take());
                    if a < k {
                        s.pending = Some(a - 1);
                    }
                } else {
                    self.layer
                        .decode_k_snap_arm(g, hidden, k, &s.live, ws, &snaps, true, 0)?;
                    if a < k {
                        self.copy_inter_to_live(s, a - 1)?;
                    }
                }
            }
            Decode => {
                self.materialise(s)?;
                self.layer.decode(g, hidden, &s.live, ws, 0)?;
            }
            Save => {
                self.materialise(s)?;
                g.copy_d2d_async(s.live.recurrent, s.snap.0, self.hb, 0)?;
                g.copy_d2d_async(s.live.conv, s.snap.1, self.cb, 0)?;
            }
            Restore => {
                self.materialise(s)?;
                g.copy_d2d_async(s.snap.0, s.live.recurrent, self.hb, 0)?;
                g.copy_d2d_async(s.snap.1, s.live.conv, self.cb, 0)?;
            }
        }
        g.synchronize(0)?;
        let n = out.len();
        if rows > 0 {
            out.push((
                format!("op{n} {op:?} final_out"),
                dn_bytes(g, ws.final_out, rows * c.hidden * 2)?,
            ));
            out.push((
                format!("op{n} {op:?} core"),
                dn_bytes(g, ws.core, rows * c.qkv_dim() * 4)?,
            ));
        }
        if matches!(op, Save) {
            out.push((format!("op{n} snapshot_h"), dn_bytes(g, s.snap.0, self.hb)?));
            out.push((
                format!("op{n} snapshot_conv"),
                dn_bytes(g, s.snap.1, self.cb)?,
            ));
        }
        Ok(())
    }

    /// 2026-10-09: One schedule on one arm, from the same starting state and hidden rows.
    fn arm(
        &self,
        ops: &[Op],
        hiddens: &[DevicePtr],
        h0: &[f32],
        c0: &[f32],
        defer: bool,
    ) -> Result<(Vec<(String, Vec<u8>)>, Vec<usize>)> {
        let poison = if defer { POISON_DEFER } else { POISON_COPY };
        let mut s = self.slot(h0, c0, poison)?;
        let (mut out, mut consumed) = (Vec::new(), Vec::new());
        for (&op, &hidden) in ops.iter().zip(hiddens) {
            self.op(&mut s, op, hidden, defer, poison, &mut out, &mut consumed)?;
        }
        self.materialise(&mut s)?;
        self.g.synchronize(0)?;
        let g = self.g;
        out.push(("live_h".into(), dn_bytes(g, s.live.recurrent, self.hb)?));
        out.push(("live_conv".into(), dn_bytes(g, s.live.conv, self.cb)?));
        out.push((
            "h_intermediates".into(),
            dn_bytes(g, s.h_inter, (MAX_ROWS - 1) * self.hb)?,
        ));
        // 2026-10-09: The pool holds MAX_ROWS conv intermediates but a K-row verify pairs only
        // rows 0..K-1 with h (mixer.rs / verify_multi.rs `(0..k - 1)`), and every reader takes
        // index a - 1 < K - 1, so slot MAX_ROWS - 1 is never written or read. Compare the written
        // slots; require the dead slot to keep this arm's own poison (neither arm may write it).
        out.push((
            "conv_intermediates".into(),
            dn_bytes(g, s.conv_inter, (MAX_ROWS - 1) * self.cb)?,
        ));
        let dead = dn_bytes(g, s.conv_inter.offset((MAX_ROWS - 1) * self.cb), self.cb)?;
        out.push((
            "conv_dead_slot_untouched".into(),
            vec![u8::from(dead.iter().all(|&b| b == poison))],
        ));
        for p in [
            s.live.conv,
            s.live.recurrent,
            s.h_inter,
            s.conv_inter,
            s.snap.0,
            s.snap.1,
            s.word,
        ] {
            let _ = g.free(p);
        }
        Ok((out, consumed))
    }
}

/// 2026-10-09: All finite and not all zero, read as BF16.
fn live_bf16(b: &[u8]) -> bool {
    let v: Vec<f32> = b
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    !v.is_empty() && v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let c = cfg();
    let kernels = Glm5NextKdaKernels::resolve(g)?;
    let layer = Glm5NextKdaLayer::new(0, c, synthetic_weights(g, &c)?, kernels)?;
    let ws = Glm5NextKdaWorkspace::new(g, &c, MAX_ROWS)?;
    let (hb, cb) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
    let rig = Rig {
        g,
        layer,
        ws,
        cfg: c,
        hb,
        cb,
    };
    if !rig.layer.defer_walk_ready() {
        println!(
            "FAIL kda_defer_commit_microtest: no deferred walk (snap_fuse_ready={}, \
             has_defer_commit={}; METRALE_GLM_KDA_NO_SMEM must be unset)",
            rig.layer.snap_fuse_ready(),
            rig.layer.kernels.has_defer_commit()
        );
        std::process::exit(1);
    }

    let (mut all_ok, mut total, mut cases, mut live_outs) = (true, 0usize, 0usize, true);
    let mut consumed_all = Vec::new();
    for (name, ops) in schedules() {
        for &seed in &SEEDS {
            let mut r = Lcg(seed ^ name.len() as u64);
            let hiddens: Vec<DevicePtr> = ops
                .iter()
                .map(|_| up_bf16(g, &r.vec(MAX_ROWS * c.hidden, 1.0)))
                .collect::<Result<_>>()?;
            let h0 = r.vec(c.recurrent_state_elems(), 0.1);
            let c0 = r.vec(c.conv_state_elems(), 0.3);
            let (a, _) = rig.arm(&ops, &hiddens, &h0, &c0, false)?;
            let (b, consumed) = rig.arm(&ops, &hiddens, &h0, &c0, true)?;
            for h in hiddens {
                let _ = g.free(h);
            }
            consumed_all.extend(consumed);
            let mut bad = Vec::new();
            for ((na, xa), (nb, xb)) in a.iter().zip(&b) {
                if na != nb || xa.len() != xb.len() {
                    bad.push(format!("{na} vs {nb}: shape differs"));
                    continue;
                }
                total += xa.len();
                if let Some(p) = xa.iter().zip(xb).position(|(x, y)| x != y) {
                    bad.push(format!("{na}: first mismatch at byte {p}"));
                }
                if na == "conv_dead_slot_untouched" && (xa[0] != 1 || xb[0] != 1) {
                    bad.push(format!("{na}: dead slot written ({} {})", xa[0], xb[0]));
                }
                if na.ends_with("final_out") {
                    live_outs &= live_bf16(xa);
                }
            }
            if a.len() != b.len() {
                bad.push(format!("{} vs {} groups", a.len(), b.len()));
            }
            cases += 1;
            if bad.is_empty() {
                println!(
                    "{name} seed={seed:#x} groups={} bytes_equal=true ok",
                    a.len()
                );
            } else {
                all_ok = false;
                for m in &bad {
                    println!("FAIL {name} seed={seed:#x} {m}");
                }
            }
        }
    }
    let idx0 = consumed_all.iter().filter(|&&i| i == 0).count();
    let idx_hi = consumed_all.iter().filter(|&&i| i >= 1).count();
    println!("consumed deferred indices: {idx0} at 0 (in place), {idx_hi} at >= 1");
    if total == 0 || idx0 == 0 || idx_hi == 0 || !live_outs {
        println!(
            "FAIL kda_defer_commit_microtest: vacuous (bytes={total}, idx0={idx0}, \
             idx_hi={idx_hi}, outputs_live={live_outs})"
        );
        std::process::exit(1);
    }
    if !all_ok {
        println!(
            "FAIL kda_defer_commit_microtest: the deferred commit differs from the copying \
             commit (see the lines above); keep METRALE_GLM_KDA_DEFER_COMMIT off"
        );
        std::process::exit(1);
    }
    println!(
        "PASS: kda_defer_commit_microtest {cases} schedules x seeds, {total} bytes bit-identical \
         (per-op outputs, snapshots, final live states, intermediates) between the copying commit \
         and the deferred commit"
    );
    Ok(())
}
