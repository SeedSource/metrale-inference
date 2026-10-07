// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Exactness and timing gate for the fused chunked-TC KDA front
//! (`METRALE_GLM_KDA_FRONT_FUSE=1`, kernels/gb10/common/kda_front_fuse.cu,
//! glm5next_kda/prefill_tc_fuse.rs) against the unfused launches it replaces.
//!
//! Owner: model-arch examples (GLM-5.3 KDA kernels).
//! Invariants:
//! - Exits nonzero, and prints a final line starting with `FAIL`, unless every comparison is
//!   bitwise equal and non-vacuous; the final line starts with `PASS` only then.
//! - KERNEL part, per case, seed and gate profile, on the same device inputs and starting
//!   states, over sub-calls of `sub` rows with the states carried:
//!   - UNFUSED: `kda_pack_qkv_bf16`, `kda_gate_bf16`, `kda_sigmoid_bf16_f32`,
//!     `kda_tc_conv_rows`, `kda_tc_conv_state_tail`, `kda_tc_prepare`, `kda_tc_scan` (the
//!     launches `front_end` + `stateful_chunked_tc` make, arguments transcribed);
//!   - FUSED: `kda_ff_prepare`, `kda_ff_conv_state_tail`, `kda_tc_scan`
//!     (`stateful_chunked_tc_fused`, arguments transcribed).
//!   After every sub-call the four chunk records (Q'|W and K'^T as u16, u and M|decay as u32)
//!   are compared; at the end the core output, the conv state and the recurrent state (u32).
//! - LAYER part: one bound layer with synthetic weights at the TP2 per-rank geometry;
//!   `prefill_chunked_tc_arm(.., front_fuse = false)` against `(.., true)` from the same
//!   states: block output (u16), core output, conv state and recurrent state (u32). This runs
//!   the real host dispatch, GEMMs included. FAILs if the target lacks either kernel set
//!   (the fused arm would silently fall back).
//! - Non-vacuity: the unfused core is finite and not all zero, both states moved, and no
//!   record buffer is left all poison.
//!
//! Timing (never affects the verdict): CUDA events on stream 0, median of `TIMED` calls after
//! `WARMUP`, at 8192 rows and 32 heads (one GLM-5.3 TP2 layer call of the 32K staged prefill):
//! each unfused front kernel, the unfused front (six launches), the fused front (two), the
//! scan, and the whole layer call both ways. The projection multiplies by 136 calls (34 KDA
//! layers x 4 windows of 8192 rows at 32K).
//!
//! Geometry: GLM-5.3-Flash KDA (head_dim 128, conv_kernel 4, l2_eps 1e-6, gate_lower_bound
//! -5, hidden 4096) at 32 heads, the TP=2 per-rank layer.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example kda_front_fuse_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use metrale_model_layers::weight_map::DenseWeight;

// 2026-10-07: CUDA driver event API for timing, declared as in `kda_snap_fuse_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const D: usize = 128;
const D_CONV: usize = 4;
const L2_EPS: f32 = 1e-6;
const LOWER_BOUND: f32 = -5.0;
const HEADS: usize = 32;
const HIDDEN: usize = 4096;
/// 2026-10-07: `KDA_TC_C`, `KDA_TC_CONV_ROWS`, `KDA_TC_SCAN_WARPS`, `KDA_TC_SCAN_SMEM` and
/// `KDA_TC_REC_BYTES` in glm5next_kda/mod.rs.
const TC_C: usize = 16;
const TC_CONV_ROWS: usize = 16;
const TC_SCAN_WARPS: usize = 2;
const TC_SCAN_SMEM: u32 = 2 * 16_128;
const REC_BYTES: [usize; 4] = [8192, 4096, 8192, 1024];
const REC_POISON: u8 = 0x7F;
/// 2026-10-07: `(rows, sub-call width)`. 2 / 15 / 17 / 257 end on a partial chunk; 1000 at 256
/// carries both states across four sub-calls; 8192 is one layer call of the 32K prefill.
const CASES: [(usize, usize); 8] = [
    (2, 2),
    (15, 15),
    (16, 16),
    (17, 17),
    (256, 256),
    (257, 257),
    (1000, 256),
    (8192, 8192),
];
const SEEDS: [u64; 2] = [0xF0F0_5E00_0001, 0xF0F0_5E00_0002];
const LAYER_ROWS: [usize; 2] = [257, 8192];
const TIME_ROWS: usize = 8192;
const WARMUP: usize = 5;
const TIMED: usize = 30;
/// 2026-10-07: KDA layer calls of a 32K cold prefill at ROWS_FFN 8192 (34 layers x 4 windows).
const CALLS_32K: f64 = 136.0;

#[derive(Clone, Copy, Debug)]
enum Gates {
    /// 2026-10-07: g + dt_bias in [-6, -1]: small log-decays (long memory).
    Mild,
    /// 2026-10-07: g + dt_bias over [-9, 9]: log-decays across (-5, 0), both ends included.
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
    fn vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f32> {
        (0..n).map(|_| self.r(lo, hi) as f32).collect()
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
fn words_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn words_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn floats(b: &[u8]) -> Vec<f32> {
    words_u32(b).into_iter().map(f32::from_bits).collect()
}
fn bytes_f32(d: &[f32]) -> Vec<u8> {
    d.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what}: CUDA status {rc}");
    }
    Ok(())
}

/// 2026-10-07: Median milliseconds of `f` over `TIMED` calls, each between two CUDA events on
/// stream 0, after `WARMUP` calls.
fn time_ms(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(0)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    ck(unsafe { cuEventCreate(&mut e0, 0) }, "cuEventCreate(start)")?;
    ck(unsafe { cuEventCreate(&mut e1, 0) }, "cuEventCreate(end)")?;
    let mut ms_all = Vec::with_capacity(TIMED);
    for _ in 0..TIMED {
        ck(unsafe { cuEventRecord(e0, 0) }, "cuEventRecord(start)")?;
        f()?;
        ck(unsafe { cuEventRecord(e1, 0) }, "cuEventRecord(end)")?;
        ck(unsafe { cuEventSynchronize(e1) }, "cuEventSynchronize")?;
        let mut ms = 0f32;
        ck(
            unsafe { cuEventElapsedTime(&mut ms, e0, e1) },
            "cuEventElapsedTime",
        )?;
        ms_all.push(ms as f64);
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    ms_all.sort_by(f64::total_cmp);
    Ok(ms_all[ms_all.len() / 2])
}

struct Kernels {
    pack: KernelHandle,
    gate: KernelHandle,
    sigmoid: KernelHandle,
    tc_conv: KernelHandle,
    tc_conv_tail: KernelHandle,
    tc_prepare: KernelHandle,
    tc_scan: KernelHandle,
    ff_prepare: KernelHandle,
    ff_tail: KernelHandle,
}

/// 2026-10-07: One case's host inputs. `parts` is `[3, k, qkv]` (q, k, v projections, the
/// workspace's `qkv_parts`), `g_raw` `[k, qkv]`, `beta_raw` `[k, heads]`.
struct Inputs {
    k: usize,
    parts: Vec<f32>,
    conv_w: Vec<f32>,
    conv_state0: Vec<f32>,
    g_raw: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    beta_raw: Vec<f32>,
    h_state0: Vec<f32>,
}

fn qkv() -> usize {
    HEADS * D
}
fn cd() -> usize {
    3 * qkv()
}

fn gen_inputs(k: usize, seed: u64, gates: Gates) -> Inputs {
    let mut r = Lcg(seed ^ (k as u64).wrapping_mul(0x9E37_79B9));
    let (glo, ghi, dlo, dhi) = match gates {
        Gates::Mild => (-2.0, 1.0, -4.0, -2.0),
        Gates::Full => (-8.0, 8.0, -1.0, 1.0),
    };
    Inputs {
        k,
        parts: r.vec(3 * k * qkv(), -1.0, 1.0),
        conv_w: r.vec(cd() * D_CONV, -0.5, 0.5),
        conv_state0: r.vec(cd() * D_CONV, -0.3, 0.3),
        g_raw: r.vec(k * qkv(), glo, ghi),
        dt_bias: r.vec(qkv(), dlo, dhi),
        a_log: r.vec(HEADS, 0.0, 0.3),
        beta_raw: r.vec(k * HEADS, -4.0, 4.0),
        h_state0: r.vec(HEADS * D * D, -0.1, 0.1),
    }
}

/// 2026-10-07: Device inputs shared by both arms (read only).
struct DevIn {
    parts: DevicePtr,
    conv_w: DevicePtr,
    g_raw: DevicePtr,
    dt_bias: DevicePtr,
    a_log: DevicePtr,
    beta_raw: DevicePtr,
}

impl DevIn {
    fn new(g: &dyn GpuBackend, ins: &Inputs) -> Result<Self> {
        Ok(Self {
            parts: up_bf16(g, &ins.parts)?,
            conv_w: up_bf16(g, &ins.conv_w)?,
            g_raw: up_bf16(g, &ins.g_raw)?,
            dt_bias: up_f32(g, &ins.dt_bias)?,
            a_log: up_f32(g, &ins.a_log)?,
            beta_raw: up_bf16(g, &ins.beta_raw)?,
        })
    }
    /// 2026-10-07: Projection `i` (0 q, 1 k, 2 v) at row `r0`.
    fn part(&self, k: usize, i: usize, r0: usize) -> DevicePtr {
        self.parts.offset((i * k + r0) * qkv() * 2)
    }
    fn free(self, g: &dyn GpuBackend) {
        for p in [
            self.parts,
            self.conv_w,
            self.g_raw,
            self.dt_bias,
            self.a_log,
            self.beta_raw,
        ] {
            let _ = g.free(p);
        }
    }
}

/// 2026-10-07: One arm's states, outputs, chunk records (sized for `sub` rows) and, for the
/// unfused arm, the intermediates the fused arm never writes (sized for `sub` rows).
struct Arm {
    conv_state: DevicePtr,
    h_state: DevicePtr,
    core: DevicePtr,
    rec: [DevicePtr; 4],
    qkv_proj: DevicePtr,
    conv_out: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
}

impl Arm {
    fn new(g: &dyn GpuBackend, ins: &Inputs, sub: usize, fill: u8) -> Result<Self> {
        let recs = sub.div_ceil(TC_C) * HEADS;
        Ok(Self {
            conv_state: up_f32(g, &ins.conv_state0)?,
            h_state: up_f32(g, &ins.h_state0)?,
            core: up_bytes(g, &vec![fill; ins.k * qkv() * 4])?,
            rec: [
                up_bytes(g, &vec![REC_POISON; recs * REC_BYTES[0]])?,
                up_bytes(g, &vec![REC_POISON; recs * REC_BYTES[1]])?,
                up_bytes(g, &vec![REC_POISON; recs * REC_BYTES[2]])?,
                up_bytes(g, &vec![REC_POISON; recs * REC_BYTES[3]])?,
            ],
            qkv_proj: up_bytes(g, &vec![0; sub * cd() * 2])?,
            conv_out: up_bytes(g, &vec![0; sub * cd() * 2])?,
            gate: up_bytes(g, &vec![0; sub * qkv() * 4])?,
            beta: up_bytes(g, &vec![0; sub * HEADS * 4])?,
        })
    }
    /// 2026-10-07: The records of the chunks `rows` rows fill, as raw bytes.
    fn records(&self, g: &dyn GpuBackend, rows: usize) -> Result<[Vec<u8>; 4]> {
        let n = rows.div_ceil(TC_C) * HEADS;
        Ok([
            dn_bytes(g, self.rec[0], n * REC_BYTES[0])?,
            dn_bytes(g, self.rec[1], n * REC_BYTES[1])?,
            dn_bytes(g, self.rec[2], n * REC_BYTES[2])?,
            dn_bytes(g, self.rec[3], n * REC_BYTES[3])?,
        ])
    }
    fn free(self, g: &dyn GpuBackend) {
        for p in [
            self.conv_state,
            self.h_state,
            self.core,
            self.rec[0],
            self.rec[1],
            self.rec[2],
            self.rec[3],
            self.qkv_proj,
            self.conv_out,
            self.gate,
            self.beta,
        ] {
            let _ = g.free(p);
        }
    }
}

/// 2026-10-07: `front_end`'s pack over rows `r0..r0 + rows`.
fn launch_pack(
    g: &dyn GpuBackend,
    kn: &Kernels,
    di: &DevIn,
    a: &Arm,
    k: usize,
    r0: usize,
    rows: usize,
) -> Result<()> {
    KernelLaunch::new(g, kn.pack)
        .grid([div_ceil(qkv() as u32, 256), rows as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(di.part(k, 0, r0))
        .arg_ptr(di.part(k, 1, r0))
        .arg_ptr(di.part(k, 2, r0))
        .arg_ptr(a.qkv_proj)
        .arg_u32(rows as u32)
        .arg_u32(qkv() as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-07: `front_end`'s `kda_gate_bf16` over rows `r0..r0 + rows`.
fn launch_gate(
    g: &dyn GpuBackend,
    kn: &Kernels,
    di: &DevIn,
    a: &Arm,
    r0: usize,
    rows: usize,
) -> Result<()> {
    KernelLaunch::new(g, kn.gate)
        .grid([(rows * HEADS) as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(di.g_raw.offset(r0 * qkv() * 2))
        .arg_ptr(di.dt_bias)
        .arg_ptr(di.a_log)
        .arg_ptr(a.gate)
        .arg_u32(rows as u32)
        .arg_u32(HEADS as u32)
        .arg_u32(D as u32)
        .arg_f32(LOWER_BOUND)
        .launch(0)?;
    Ok(())
}

/// 2026-10-07: `front_end`'s beta sigmoid over rows `r0..r0 + rows`.
fn launch_sigmoid(
    g: &dyn GpuBackend,
    kn: &Kernels,
    di: &DevIn,
    a: &Arm,
    r0: usize,
    rows: usize,
) -> Result<()> {
    let n = rows * HEADS;
    KernelLaunch::new(g, kn.sigmoid)
        .grid([div_ceil(n as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(di.beta_raw.offset(r0 * HEADS * 2))
        .arg_ptr(a.beta)
        .arg_u32(n as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-07: `stateful_chunked_tc`'s conv and conv-state tail over `rows` rows.
fn launch_conv(g: &dyn GpuBackend, kn: &Kernels, di: &DevIn, a: &Arm, rows: usize) -> Result<()> {
    let cd = cd();
    KernelLaunch::new(g, kn.tc_conv)
        .grid([
            div_ceil(cd as u32, 256),
            div_ceil(rows as u32, TC_CONV_ROWS as u32),
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a.conv_state)
        .arg_ptr(a.qkv_proj)
        .arg_ptr(di.conv_w)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(a.conv_out)
        .arg_u32(rows as u32)
        .arg_u32(cd as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32((2 * qkv()) as u32)
        .arg_u32(D as u32)
        .arg_f32(L2_EPS)
        .arg_u32(cd as u32)
        .arg_u32(cd as u32)
        .launch(0)?;
    KernelLaunch::new(g, kn.tc_conv_tail)
        .grid([div_ceil(cd as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.conv_state)
        .arg_ptr(a.qkv_proj)
        .arg_u32(rows as u32)
        .arg_u32(cd as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(cd as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-07: `stateful_chunked_tc`'s prepare over `rows` rows.
fn launch_prepare(g: &dyn GpuBackend, kn: &Kernels, a: &Arm, rows: usize) -> Result<()> {
    let (cd, qkv) = (cd(), qkv());
    KernelLaunch::new(g, kn.tc_prepare)
        .grid([rows.div_ceil(TC_C) as u32, HEADS as u32, 1])
        .block([D as u32, 1, 1])
        .arg_ptr(a.conv_out)
        .arg_ptr(a.conv_out.offset(qkv * 2))
        .arg_ptr(a.conv_out.offset(qkv * 4))
        .arg_ptr(a.gate)
        .arg_ptr(a.beta)
        .arg_ptr(a.rec[0])
        .arg_ptr(a.rec[1])
        .arg_ptr(a.rec[2])
        .arg_ptr(a.rec[3])
        .arg_u32(HEADS as u32)
        .arg_u32(rows as u32)
        .arg_u32(cd as u32)
        .arg_u32(qkv as u32)
        .arg_u32(HEADS as u32)
        .arg_f32(1.0 / (D as f32).sqrt())
        .launch(0)?;
    Ok(())
}

/// 2026-10-07: `stateful_chunked_tc_fused`'s prepare and conv-state tail over rows
/// `r0..r0 + rows`.
#[allow(clippy::too_many_arguments)]
fn launch_fused(
    g: &dyn GpuBackend,
    kn: &Kernels,
    di: &DevIn,
    a: &Arm,
    k: usize,
    r0: usize,
    rows: usize,
) -> Result<()> {
    let qkv = qkv();
    let (q, kk, v) = (di.part(k, 0, r0), di.part(k, 1, r0), di.part(k, 2, r0));
    KernelLaunch::new(g, kn.ff_prepare)
        .grid([rows.div_ceil(TC_C) as u32, HEADS as u32, 1])
        .block([D as u32, 1, 1])
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(v)
        .arg_ptr(a.conv_state)
        .arg_ptr(di.conv_w)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(di.g_raw.offset(r0 * qkv * 2))
        .arg_ptr(di.dt_bias)
        .arg_ptr(di.a_log)
        .arg_ptr(di.beta_raw.offset(r0 * HEADS * 2))
        .arg_ptr(a.rec[0])
        .arg_ptr(a.rec[1])
        .arg_ptr(a.rec[2])
        .arg_ptr(a.rec[3])
        .arg_u32(HEADS as u32)
        .arg_u32(rows as u32)
        .arg_u32(qkv as u32)
        .arg_f32(L2_EPS)
        .arg_f32(LOWER_BOUND)
        .arg_f32(1.0 / (D as f32).sqrt())
        .launch(0)?;
    KernelLaunch::new(g, kn.ff_tail)
        .grid([div_ceil(cd() as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.conv_state)
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(v)
        .arg_u32(rows as u32)
        .arg_u32(qkv as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(qkv as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-07: `launch_tc_scan` over rows `r0..r0 + rows`.
fn launch_scan(g: &dyn GpuBackend, kn: &Kernels, a: &Arm, r0: usize, rows: usize) -> Result<()> {
    KernelLaunch::new(g, kn.tc_scan)
        .grid([HEADS as u32, (D / (16 * TC_SCAN_WARPS)) as u32, 1])
        .block([(32 * TC_SCAN_WARPS) as u32, 1, 1])
        .shared_mem(TC_SCAN_SMEM)
        .arg_ptr(a.rec[0])
        .arg_ptr(a.rec[1])
        .arg_ptr(a.rec[2])
        .arg_ptr(a.rec[3])
        .arg_ptr(a.h_state)
        .arg_ptr(a.core.offset(r0 * qkv() * 4))
        .arg_u32(HEADS as u32)
        .arg_u32(rows as u32)
        .arg_u32(rows.div_ceil(TC_C) as u32)
        .arg_u32(qkv() as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-07: The unfused front over rows `r0..r0 + rows`: the six launches.
#[allow(clippy::too_many_arguments)]
fn unfused_front(
    g: &dyn GpuBackend,
    kn: &Kernels,
    di: &DevIn,
    a: &Arm,
    k: usize,
    r0: usize,
    rows: usize,
) -> Result<()> {
    launch_pack(g, kn, di, a, k, r0, rows)?;
    launch_gate(g, kn, di, a, r0, rows)?;
    launch_sigmoid(g, kn, di, a, r0, rows)?;
    launch_conv(g, kn, di, a, rows)?;
    launch_prepare(g, kn, a, rows)
}

/// 2026-10-07: The KERNEL part for one case: both arms sub-call by sub-call, the records
/// compared after each, the core and the states at the end. Returns the verdict.
fn kernel_case(
    g: &dyn GpuBackend,
    kn: &Kernels,
    k: usize,
    sub: usize,
    seed: u64,
    gates: Gates,
) -> Result<bool> {
    let ins = gen_inputs(k, seed, gates);
    let di = DevIn::new(g, &ins)?;
    let au = Arm::new(g, &ins, sub, 0xAB)?;
    let af = Arm::new(g, &ins, sub, 0xAB)?;
    let mut rec_eq = true;
    let mut rec_live = true;
    for r0 in (0..k).step_by(sub) {
        let rows = sub.min(k - r0);
        unfused_front(g, kn, &di, &au, k, r0, rows)?;
        launch_scan(g, kn, &au, r0, rows)?;
        launch_fused(g, kn, &di, &af, k, r0, rows)?;
        launch_scan(g, kn, &af, r0, rows)?;
        g.synchronize(0)?;
        let (ru, rf) = (au.records(g, rows)?, af.records(g, rows)?);
        rec_eq &= words_u16(&ru[0]) == words_u16(&rf[0])
            && words_u16(&ru[1]) == words_u16(&rf[1])
            && words_u32(&ru[2]) == words_u32(&rf[2])
            && words_u32(&ru[3]) == words_u32(&rf[3]);
        rec_live &= ru.iter().all(|b| b.iter().any(|x| *x != REC_POISON));
    }
    let qkv = qkv();
    let core_u = dn_bytes(g, au.core, k * qkv * 4)?;
    let core_f = dn_bytes(g, af.core, k * qkv * 4)?;
    let hb = HEADS * D * D * 4;
    let cb = cd() * D_CONV * 4;
    let (hu, hf) = (dn_bytes(g, au.h_state, hb)?, dn_bytes(g, af.h_state, hb)?);
    let (cu, cf) = (
        dn_bytes(g, au.conv_state, cb)?,
        dn_bytes(g, af.conv_state, cb)?,
    );
    let core_eq = words_u32(&core_u) == words_u32(&core_f);
    let h_eq = words_u32(&hu) == words_u32(&hf);
    let c_eq = words_u32(&cu) == words_u32(&cf);
    let cu_f = floats(&core_u);
    let live = cu_f.iter().all(|x| x.is_finite()) && cu_f.iter().any(|x| *x != 0.0);
    let moved = hu != bytes_f32(&ins.h_state0) && cu != bytes_f32(&ins.conv_state0);
    let ok = rec_eq && core_eq && h_eq && c_eq && live && moved && rec_live;
    let tag = |b: bool| if b { "eq" } else { "MISMATCH" };
    eprintln!(
        "KERNEL rows={k:>4} sub={sub:>4} gates={gates:?} seed={seed:#x} records={} core={} \
         h_state={} conv_state={}{}  {}",
        tag(rec_eq),
        tag(core_eq),
        tag(h_eq),
        tag(c_eq),
        if live && moved && rec_live {
            String::new()
        } else {
            format!("  VACUOUS(live={live} moved={moved} records_written={rec_live})")
        },
        if ok { "PASS" } else { "FAIL" }
    );
    au.free(g);
    af.free(g);
    di.free(g);
    Ok(ok)
}

/// 2026-10-07: `TIMING` for one 8192-row, 32-head call. Repeated calls advance the conv state;
/// the work per call does not change.
fn kernel_timing(g: &dyn GpuBackend, kn: &Kernels) -> Result<f64> {
    let k = TIME_ROWS;
    let ins = gen_inputs(k, SEEDS[0], Gates::Mild);
    let di = DevIn::new(g, &ins)?;
    let a = Arm::new(g, &ins, k, 0)?;
    let pack = time_ms(g, || launch_pack(g, kn, &di, &a, k, 0, k))?;
    let gate = time_ms(g, || launch_gate(g, kn, &di, &a, 0, k))?;
    let sigm = time_ms(g, || launch_sigmoid(g, kn, &di, &a, 0, k))?;
    let conv = time_ms(g, || launch_conv(g, kn, &di, &a, k))?;
    let prep = time_ms(g, || launch_prepare(g, kn, &a, k))?;
    let unfused = time_ms(g, || unfused_front(g, kn, &di, &a, k, 0, k))?;
    let fused = time_ms(g, || launch_fused(g, kn, &di, &a, k, 0, k))?;
    let scan = time_ms(g, || launch_scan(g, kn, &a, 0, k))?;
    let save = unfused - fused;
    eprintln!(
        "TIMING kernels rows={k} heads={HEADS} pack_ms={pack:.3} gate_ms={gate:.3} \
         sigmoid_ms={sigm:.3} conv+tail_ms={conv:.3} prepare_ms={prep:.3} \
         unfused_front_ms={unfused:.3} fused_front_ms={fused:.3} scan_ms={scan:.3} \
         saved_ms_per_call={save:.3} projected_32k_s={:.3}",
        save * CALLS_32K / 1e3
    );
    a.free(g);
    di.free(g);
    Ok(save)
}

fn dwt(g: &dyn GpuBackend, v: &[f32]) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: up_bf16(g, v)?,
    })
}

fn layer_cfg() -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden: HIDDEN,
        heads: HEADS,
        head_dim: D,
        conv_kernel: D_CONV,
        gate_lower_bound: LOWER_BOUND,
        rms_norm_eps: 1e-5,
        l2_eps: L2_EPS,
        chunk: 32,
    }
}

/// 2026-10-07: Synthetic weights as in `kda_snap_fuse_microtest`: projections uniform in
/// `±1/sqrt(fan_in)`, a mild decay.
fn synthetic_weights(g: &dyn GpuBackend, c: &Glm5NextKdaConfig) -> Result<Glm5NextKdaWeights> {
    let mut r = Lcg(0x57E1_6475);
    let (hid, qkv, hd, h) = (c.hidden, c.qkv_dim(), c.head_dim, c.heads);
    let s = |n: usize| 1.0 / (n as f64).sqrt();
    Ok(Glm5NextKdaWeights {
        q_proj: dwt(g, &r.vec(qkv * hid, -s(hid), s(hid)))?,
        k_proj: dwt(g, &r.vec(qkv * hid, -s(hid), s(hid)))?,
        v_proj: dwt(g, &r.vec(qkv * hid, -s(hid), s(hid)))?,
        conv: dwt(g, &r.vec(3 * qkv * c.conv_kernel, -0.5, 0.5))?,
        f_a: dwt(g, &r.vec(hd * hid, -s(hid), s(hid)))?,
        f_b: dwt(g, &r.vec(qkv * hd, -0.5 * s(hd), 0.5 * s(hd)))?,
        dt_bias: up_f32(g, &r.vec(qkv, -4.0, -2.0))?,
        a_log: up_f32(g, &r.vec(h, 0.0, 0.3))?,
        b_proj: dwt(g, &r.vec(h * hid, -2.0 * s(hid), 2.0 * s(hid)))?,
        g_a: dwt(g, &r.vec(hd * hid, -s(hid), s(hid)))?,
        g_b: dwt(g, &r.vec(qkv * hd, -s(hd), s(hd)))?,
        o_norm: dwt(g, &r.vec(hd, 0.5, 1.5))?,
        o_proj: dwt(g, &r.vec(hid * qkv, -s(qkv), s(qkv)))?,
    })
}

/// 2026-10-07: One LAYER arm's results as raw bytes.
struct LayerOut {
    final_out: Vec<u8>,
    core: Vec<u8>,
    h: Vec<u8>,
    conv: Vec<u8>,
}

/// 2026-10-07: `prefill_chunked_tc_arm` over `k` rows of `hidden` from the given states.
#[allow(clippy::too_many_arguments)]
fn layer_run(
    g: &dyn GpuBackend,
    layer: &Glm5NextKdaLayer,
    ws: &Glm5NextKdaWorkspace,
    hidden: DevicePtr,
    k: usize,
    h0: &[f32],
    c0: &[f32],
    fuse: bool,
) -> Result<LayerOut> {
    let c = &layer.cfg;
    let st = KdaSeqState {
        conv: up_f32(g, c0)?,
        recurrent: up_f32(g, h0)?,
    };
    g.copy_h2d(&vec![0xCD; k * c.hidden * 2], ws.final_out)?;
    g.copy_h2d(&vec![0xCD; k * c.qkv_dim() * 4], ws.core)?;
    // 2026-10-07: The intermediates the fused arm must not read, poisoned before either arm
    // (the unfused arm overwrites them), so a fused read of a stale unfused value cannot pass.
    g.copy_h2d(&vec![0xCD; k * c.conv_dim() * 2], ws.qkv_proj)?;
    g.copy_h2d(&vec![0xCD; k * c.conv_dim() * 2], ws.conv_out)?;
    g.copy_h2d(&vec![0xCD; k * c.qkv_dim() * 4], ws.gate)?;
    g.copy_h2d(&vec![0xCD; k * c.heads * 4], ws.beta)?;
    layer.prefill_chunked_tc_arm(g, hidden, k, &st, ws, fuse, 0)?;
    g.synchronize(0)?;
    let out = LayerOut {
        final_out: dn_bytes(g, ws.final_out, k * c.hidden * 2)?,
        core: dn_bytes(g, ws.core, k * c.qkv_dim() * 4)?,
        h: dn_bytes(g, st.recurrent, c.recurrent_state_elems() * 4)?,
        conv: dn_bytes(g, st.conv, c.conv_state_elems() * 4)?,
    };
    let _ = g.free(st.conv);
    let _ = g.free(st.recurrent);
    Ok(out)
}

/// 2026-10-07: The LAYER part: every row count in `LAYER_ROWS`, then the timing of the whole
/// layer call both ways at the largest. Returns the verdict.
fn layer_part(g: &dyn GpuBackend) -> Result<bool> {
    let cfg = layer_cfg();
    let kernels = Glm5NextKdaKernels::resolve(g)?;
    if !kernels.has_chunked_tc() || !kernels.has_front_fuse() {
        eprintln!(
            "LAYER the target lacks a kernel (chunked_tc={} front_fuse={}); the fused arm \
             would fall back  FAIL",
            kernels.has_chunked_tc(),
            kernels.has_front_fuse()
        );
        return Ok(false);
    }
    let weights = synthetic_weights(g, &cfg)?;
    let layer = Glm5NextKdaLayer::new(0, cfg, weights, kernels)?;
    let max = *LAYER_ROWS.iter().max().unwrap_or(&1);
    let ws = Glm5NextKdaWorkspace::new(g, &cfg, max)?;
    let mut all = true;
    for (i, &k) in LAYER_ROWS.iter().enumerate() {
        let mut r = Lcg(SEEDS[i % SEEDS.len()] ^ 0x1A7E);
        let hidden = up_bf16(g, &r.vec(k * cfg.hidden, -1.0, 1.0))?;
        let h0 = r.vec(cfg.recurrent_state_elems(), -0.1, 0.1);
        let c0 = r.vec(cfg.conv_state_elems(), -0.3, 0.3);
        let u = layer_run(g, &layer, &ws, hidden, k, &h0, &c0, false)?;
        let f = layer_run(g, &layer, &ws, hidden, k, &h0, &c0, true)?;
        let out_eq = words_u16(&u.final_out) == words_u16(&f.final_out);
        let core_eq = words_u32(&u.core) == words_u32(&f.core);
        let h_eq = words_u32(&u.h) == words_u32(&f.h);
        let c_eq = words_u32(&u.conv) == words_u32(&f.conv);
        let uc = floats(&u.core);
        let live = uc.iter().all(|x| x.is_finite()) && uc.iter().any(|x| *x != 0.0);
        let moved = u.h != bytes_f32(&h0) && u.conv != bytes_f32(&c0);
        let ok = out_eq && core_eq && h_eq && c_eq && live && moved;
        all &= ok;
        let tag = |b: bool| if b { "eq" } else { "MISMATCH" };
        eprintln!(
            "LAYER rows={k:>4} final_out={} core={} h_state={} conv_state={}{}  {}",
            tag(out_eq),
            tag(core_eq),
            tag(h_eq),
            tag(c_eq),
            if live && moved {
                String::new()
            } else {
                format!("  VACUOUS(live={live} moved={moved})")
            },
            if ok { "PASS" } else { "FAIL" }
        );
        if k == max {
            let st = KdaSeqState {
                conv: up_f32(g, &c0)?,
                recurrent: up_f32(g, &h0)?,
            };
            let tu = time_ms(g, || {
                layer.prefill_chunked_tc_arm(g, hidden, k, &st, &ws, false, 0)
            })?;
            let tf = time_ms(g, || {
                layer.prefill_chunked_tc_arm(g, hidden, k, &st, &ws, true, 0)
            })?;
            eprintln!(
                "TIMING layer rows={k} heads={HEADS} hidden={HIDDEN} unfused_ms={tu:.3} \
                 fused_ms={tf:.3} saved_ms_per_call={:.3} projected_32k_s={:.3}",
                tu - tf,
                (tu - tf) * CALLS_32K / 1e3
            );
            let _ = g.free(st.conv);
            let _ = g.free(st.recurrent);
        }
        let _ = g.free(hidden);
    }
    Ok(all)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    // 2026-10-07: Hard lookups: a missing kernel is a FAIL here, not a silent fallback.
    let kn = Kernels {
        pack: g.kernel("kda_layer_ops", "kda_pack_qkv_bf16")?,
        gate: g.kernel("kda_gate", "kda_gate_bf16")?,
        sigmoid: g.kernel("kda_layer_ops", "kda_sigmoid_bf16_f32")?,
        tc_conv: g.kernel("kda_chunk_tc", "kda_tc_conv_rows")?,
        tc_conv_tail: g.kernel("kda_chunk_tc", "kda_tc_conv_state_tail")?,
        tc_prepare: g.kernel("kda_chunk_tc", "kda_tc_prepare")?,
        tc_scan: g.kernel("kda_chunk_tc", "kda_tc_scan")?,
        ff_prepare: g.kernel("kda_front_fuse", "kda_ff_prepare")?,
        ff_tail: g.kernel("kda_front_fuse", "kda_ff_conv_state_tail")?,
    };

    let mut all_ok = true;
    let mut cases = 0usize;
    for &(k, sub) in &CASES {
        for gates in [Gates::Mild, Gates::Full] {
            for &seed in &SEEDS {
                all_ok &= kernel_case(g, &kn, k, sub, seed, gates)?;
                cases += 1;
            }
        }
    }
    let layer_ok = layer_part(g)?;
    if cases == 0 {
        bail!("ran 0 cases; refusing to report PASS");
    }

    // 2026-10-07: Timing; reported whatever the verdict.
    let _ = kernel_timing(g, &kn)?;

    let ok = all_ok && layer_ok;
    println!(
        "{} KDA front fuse: {cases} kernel cases (records, core, h_state, conv_state bitwise) \
         {}, layer cases ({LAYER_ROWS:?} rows: final_out, core, states bitwise) {}",
        if ok { "PASS" } else { "FAIL" },
        if all_ok { "pass" } else { "FAIL" },
        if layer_ok { "pass" } else { "FAIL" }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
