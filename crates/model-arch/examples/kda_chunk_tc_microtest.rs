// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Accuracy and timing gate for the opt-in GLM KDA tensor-core chunked prefill
//! (`METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1`, `Glm5NextKdaLayer::prefill_chunked_tc`) against the
//! token loop it replaces in prefill (`METRALE_GLM_KDA_TOKEN_LOOP=1`, `stateful_rows`).
//!
//! Owner: model-arch examples (GLM-5.3 KDA kernels).
//! Invariants:
//! - Exits nonzero unless, for every case, seed and gate profile:
//!   - the conv outputs (`[k, conv_dim]` BF16) and the final conv state (`[conv_dim,
//!     conv_kernel]` FP32) are identical bit for bit (the TC conv is built to be);
//!   - the core outputs (`[k, heads * head_dim]` FP32) and the final recurrent state (`[heads,
//!     head_dim, head_dim]` FP32) are finite and have cosine similarity > `COS_MIN` with the
//!     loop's (the recurrence is not bit-identical by design; max abs error is reported).
//! - Refuses to report PASS when the loop's outputs are all zero or not finite, or when the loop
//!   left either state unchanged.
//!
//! Arms, on the same device inputs and the same starting states:
//! - LOOP: `causal_conv1d_update_l2norm_rows` then `kda_recurrent_prefill_bf16_smem` over all
//!   rows: the two launches `stateful_rows` makes, arguments transcribed.
//! - TC: `kda_tc_conv_rows`, `kda_tc_conv_state_tail`, `kda_tc_prepare`, `kda_tc_scan`: the four
//!   launches `stateful_chunked_tc` makes, arguments transcribed, over sub-calls of `sub` rows
//!   in order (`sub` = rows is one call; `sub` = 256 is a prefill run in 256-row sub-chunks, the
//!   state carried between calls).
//!
//! Gate profiles: MILD draws the log-decay from [-0.3, -0.005] (slow decay, long memory); FULL
//! draws it from (-5, 0), the whole range GLM's `gate_lower_bound = -5` allows, which exercises
//! the exponent range of the chunk factorisation.
//!
//! After the gate it prints `TIMING` lines at 4096 and 8192 rows (32 heads): each launch alone
//! and the whole TC arm as one call and as 256-row sub-calls, against the loop's conv and
//! recurrence (and `kda_recurrent_prefill_bf16_pf`, the `METRALE_GLM_KDA_PREFETCH=1` kernel,
//! when the target has it). Wall clock over repeated launches; timing never affects the verdict.
//!
//! Geometry: GLM-5.3-Flash KDA (head_dim 128, conv_kernel 4, l2_eps 1e-6) at 32 heads, the TP=2
//! per-rank layer the 2x GB10 serve runs.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example kda_chunk_tc_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const D: usize = 128;
const D_CONV: usize = 4;
const L2_EPS: f32 = 1e-6;
const HEADS: usize = 32;
/// 2026-10-01: `KDA_V_PER_BLOCK` in glm5next_kda/mod.rs.
const VPB: usize = 32;
/// 2026-10-01: `KDA_TC_C`, `KDA_TC_CONV_ROWS`, `KDA_TC_SCAN_WARPS`, `KDA_TC_SCAN_SMEM` and
/// `KDA_TC_REC_BYTES` in glm5next_kda/mod.rs.
const TC_C: usize = 16;
const TC_CONV_ROWS: usize = 16;
const TC_SCAN_WARPS: usize = 2;
const TC_SCAN_SMEM: u32 = 2 * 16_128;
const REC_BYTES: [usize; 4] = [8192, 4096, 8192, 1024];

const COS_MIN: f64 = 0.9995;
/// 2026-10-01: `(rows, sub-call width)`. 15 / 17 / 255 / 1000 end on a partial chunk; 8192 at
/// 256 is the 8K prefill in the 256-row sub-chunks the staged prefill runs.
const CASES: [(usize, usize); 10] = [
    (2, 2),
    (15, 15),
    (16, 16),
    (17, 17),
    (255, 255),
    (256, 256),
    (1000, 1000),
    (4096, 4096),
    (8192, 256),
    (8192, 8192),
];
const SEEDS: [u64; 2] = [0x4B_DA7C_0001, 0x4B_DA7C_0002];
const TIME_ROWS: [usize; 2] = [4096, 8192];
const TIME_SUB: usize = 256;
const TIME_WARMUP: usize = 2;
const TIME_REPS: usize = 10;

#[derive(Clone, Copy, Debug)]
enum Gates {
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
}

fn up_bytes(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
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

/// 2026-10-01: One layer's geometry and inputs. Row-major like the workspace:
/// `qkv_proj` `[k, conv_dim]`, `gate` `[k, qkv]`, `beta` `[k, heads]`.
struct Inputs {
    heads: usize,
    k: usize,
    qkv_proj: Vec<bf16>,
    conv_weight: Vec<bf16>,
    conv_state0: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    h_state0: Vec<f32>,
}

impl Inputs {
    fn qkv(&self) -> usize {
        self.heads * D
    }
    fn cd(&self) -> usize {
        3 * self.qkv()
    }
}

fn gen_inputs(heads: usize, k: usize, seed: u64, gates: Gates) -> Inputs {
    let mut r = Lcg(seed ^ ((heads as u64) << 32) ^ (k as u64));
    let qkv = heads * D;
    let cd = 3 * qkv;
    // 2026-10-01: kda_gate's output is a log-decay in (gate_lower_bound, 0) = (-5, 0).
    let (glo, ghi) = match gates {
        Gates::Mild => (-0.3, -0.005),
        Gates::Full => (-4.999, -0.0001),
    };
    Inputs {
        heads,
        k,
        qkv_proj: (0..k * cd).map(|_| bf16::from_f64(r.r(-0.5, 0.5))).collect(),
        conv_weight: (0..cd * D_CONV)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect(),
        conv_state0: (0..cd * D_CONV).map(|_| r.r(-0.3, 0.3) as f32).collect(),
        gate: (0..k * qkv).map(|_| r.r(glo, ghi) as f32).collect(),
        beta: (0..k * heads).map(|_| r.r(0.05, 0.95) as f32).collect(),
        h_state0: (0..heads * D * D).map(|_| r.r(-0.1, 0.1) as f32).collect(),
    }
}

struct Kernels {
    conv_rows: KernelHandle,
    recurrent_rows: KernelHandle,
    /// 2026-10-01: `None` on a target without the prefetching kernel; timing only.
    recurrent_pf: Option<KernelHandle>,
    tc_conv: KernelHandle,
    tc_conv_tail: KernelHandle,
    tc_prepare: KernelHandle,
    tc_scan: KernelHandle,
}

/// 2026-10-01: One arm's results as raw bytes.
struct Captured {
    conv_out: Vec<u8>,
    core: Vec<u8>,
    h_state: Vec<u8>,
    conv_state: Vec<u8>,
}

struct Bufs {
    qkv_proj: DevicePtr,
    conv_w: DevicePtr,
    conv_state: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    h_state: DevicePtr,
    conv_out: DevicePtr,
    core: DevicePtr,
}

impl Bufs {
    fn new(g: &dyn GpuBackend, ins: &Inputs, fill: u8) -> Result<Self> {
        let (k, cd, qkv) = (ins.k, ins.cd(), ins.qkv());
        Ok(Self {
            qkv_proj: up_bf16(g, &ins.qkv_proj)?,
            conv_w: up_bf16(g, &ins.conv_weight)?,
            conv_state: up_f32(g, &ins.conv_state0)?,
            gate: up_f32(g, &ins.gate)?,
            beta: up_f32(g, &ins.beta)?,
            h_state: up_f32(g, &ins.h_state0)?,
            conv_out: up_bytes(g, &vec![fill; k * cd * 2])?,
            core: up_bytes(g, &vec![fill; k * qkv * 4])?,
        })
    }
    fn capture(&self, g: &dyn GpuBackend, ins: &Inputs) -> Result<Captured> {
        let (k, cd, qkv) = (ins.k, ins.cd(), ins.qkv());
        g.synchronize(0)?;
        Ok(Captured {
            conv_out: dn_bytes(g, self.conv_out, k * cd * 2)?,
            core: dn_bytes(g, self.core, k * qkv * 4)?,
            h_state: dn_bytes(g, self.h_state, ins.heads * D * D * 4)?,
            conv_state: dn_bytes(g, self.conv_state, cd * D_CONV * 4)?,
        })
    }
    fn free(self, g: &dyn GpuBackend) {
        for p in [
            self.qkv_proj,
            self.conv_w,
            self.conv_state,
            self.gate,
            self.beta,
            self.h_state,
            self.conv_out,
            self.core,
        ] {
            let _ = g.free(p);
        }
    }
}

/// 2026-10-01: The chunk records for sub-calls of up to `rows` rows: the four workspace buffers
/// `stateful_chunked_tc` borrows (`q_f32`, `k_f32`, `v_f32`, `chunk_gc`), sized to the records.
struct Records {
    qw: DevicePtr,
    kpt: DevicePtr,
    u: DevicePtr,
    md: DevicePtr,
}

impl Records {
    fn new(g: &dyn GpuBackend, heads: usize, rows: usize) -> Result<Self> {
        let recs = rows.div_ceil(TC_C) * heads;
        Ok(Self {
            qw: up_bytes(g, &vec![0x7F; recs * REC_BYTES[0]])?,
            kpt: up_bytes(g, &vec![0x7F; recs * REC_BYTES[1]])?,
            u: up_bytes(g, &vec![0x7F; recs * REC_BYTES[2]])?,
            md: up_bytes(g, &vec![0x7F; recs * REC_BYTES[3]])?,
        })
    }
    fn free(self, g: &dyn GpuBackend) {
        for p in [self.qw, self.kpt, self.u, self.md] {
            let _ = g.free(p);
        }
    }
}

fn smem_bytes() -> u32 {
    ((3 * D + VPB * (D + 1)) * 4) as u32
}

/// 2026-10-01: `kda_pf_smem` in glm5next_kda/mod.rs.
fn pf_smem_bytes() -> u32 {
    ((6 * D + VPB * (D + 1)) * 4) as u32
}

/// 2026-10-01: `stateful_rows`' conv launch, all rows.
fn launch_conv_rows(g: &dyn GpuBackend, kn: &Kernels, ins: &Inputs, b: &Bufs) -> Result<()> {
    let (cd, qkv, k) = (ins.cd(), ins.qkv(), ins.k);
    KernelLaunch::new(g, kn.conv_rows)
        .grid([div_ceil(cd as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.conv_state)
        .arg_ptr(b.qkv_proj)
        .arg_ptr(b.conv_w)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(b.conv_out)
        .arg_u32(k as u32)
        .arg_u32(cd as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32((2 * qkv) as u32)
        .arg_u32(D as u32)
        .arg_f32(L2_EPS)
        .arg_u32(cd as u32)
        .arg_u32(cd as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-01: `stateful_rows`' recurrent launch, all rows, with `kernel` and its request.
fn launch_recurrent_rows(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    smem: u32,
    ins: &Inputs,
    b: &Bufs,
) -> Result<()> {
    let (cd, qkv, h, k) = (ins.cd(), ins.qkv(), ins.heads, ins.k);
    KernelLaunch::new(g, kernel)
        .grid([h as u32, (D / VPB) as u32, 1])
        .block([VPB as u32, 1, 1])
        .shared_mem(smem)
        .arg_ptr(b.conv_out)
        .arg_ptr(b.conv_out.offset(qkv * 2))
        .arg_ptr(b.conv_out.offset(qkv * 4))
        .arg_ptr(b.gate)
        .arg_ptr(b.beta)
        .arg_ptr(b.h_state)
        .arg_ptr(b.core)
        .arg_u32(h as u32)
        .arg_u32(D as u32)
        .arg_f32(1.0 / (D as f32).sqrt())
        .arg_u32(VPB as u32)
        .arg_u32(k as u32)
        .arg_u32(cd as u32)
        .arg_u32(qkv as u32)
        .arg_u32(h as u32)
        .arg_u32(qkv as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-01: `stateful_chunked_tc`'s conv and conv-state tail over rows `r0..r0 + rows`.
fn launch_tc_conv(
    g: &dyn GpuBackend,
    kn: &Kernels,
    ins: &Inputs,
    b: &Bufs,
    r0: usize,
    rows: usize,
) -> Result<()> {
    let (cd, qkv) = (ins.cd(), ins.qkv());
    let proj = b.qkv_proj.offset(r0 * cd * 2);
    KernelLaunch::new(g, kn.tc_conv)
        .grid([
            div_ceil(cd as u32, 256),
            div_ceil(rows as u32, TC_CONV_ROWS as u32),
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(b.conv_state)
        .arg_ptr(proj)
        .arg_ptr(b.conv_w)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(b.conv_out.offset(r0 * cd * 2))
        .arg_u32(rows as u32)
        .arg_u32(cd as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32((2 * qkv) as u32)
        .arg_u32(D as u32)
        .arg_f32(L2_EPS)
        .arg_u32(cd as u32)
        .arg_u32(cd as u32)
        .launch(0)?;
    KernelLaunch::new(g, kn.tc_conv_tail)
        .grid([div_ceil(cd as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b.conv_state)
        .arg_ptr(proj)
        .arg_u32(rows as u32)
        .arg_u32(cd as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(cd as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-01: `stateful_chunked_tc`'s prepare over rows `r0..r0 + rows`.
fn launch_tc_prepare(
    g: &dyn GpuBackend,
    kn: &Kernels,
    ins: &Inputs,
    b: &Bufs,
    rec: &Records,
    r0: usize,
    rows: usize,
) -> Result<()> {
    let (cd, qkv, h) = (ins.cd(), ins.qkv(), ins.heads);
    let conv = b.conv_out.offset(r0 * cd * 2);
    KernelLaunch::new(g, kn.tc_prepare)
        .grid([rows.div_ceil(TC_C) as u32, h as u32, 1])
        .block([D as u32, 1, 1])
        .arg_ptr(conv)
        .arg_ptr(conv.offset(qkv * 2))
        .arg_ptr(conv.offset(qkv * 4))
        .arg_ptr(b.gate.offset(r0 * qkv * 4))
        .arg_ptr(b.beta.offset(r0 * h * 4))
        .arg_ptr(rec.qw)
        .arg_ptr(rec.kpt)
        .arg_ptr(rec.u)
        .arg_ptr(rec.md)
        .arg_u32(h as u32)
        .arg_u32(rows as u32)
        .arg_u32(cd as u32)
        .arg_u32(qkv as u32)
        .arg_u32(h as u32)
        .arg_f32(1.0 / (D as f32).sqrt())
        .launch(0)?;
    Ok(())
}

/// 2026-10-01: `stateful_chunked_tc`'s scan over rows `r0..r0 + rows`.
fn launch_tc_scan(
    g: &dyn GpuBackend,
    kn: &Kernels,
    ins: &Inputs,
    b: &Bufs,
    rec: &Records,
    r0: usize,
    rows: usize,
) -> Result<()> {
    let (qkv, h) = (ins.qkv(), ins.heads);
    KernelLaunch::new(g, kn.tc_scan)
        .grid([h as u32, (D / (16 * TC_SCAN_WARPS)) as u32, 1])
        .block([(32 * TC_SCAN_WARPS) as u32, 1, 1])
        .shared_mem(TC_SCAN_SMEM)
        .arg_ptr(rec.qw)
        .arg_ptr(rec.kpt)
        .arg_ptr(rec.u)
        .arg_ptr(rec.md)
        .arg_ptr(b.h_state)
        .arg_ptr(b.core.offset(r0 * qkv * 4))
        .arg_u32(h as u32)
        .arg_u32(rows as u32)
        .arg_u32(rows.div_ceil(TC_C) as u32)
        .arg_u32(qkv as u32)
        .launch(0)?;
    Ok(())
}

/// 2026-10-01: The whole TC arm over all rows as consecutive sub-calls of `sub` rows.
fn launch_tc(
    g: &dyn GpuBackend,
    kn: &Kernels,
    ins: &Inputs,
    b: &Bufs,
    rec: &Records,
    sub: usize,
) -> Result<()> {
    for r0 in (0..ins.k).step_by(sub.max(1)) {
        let rows = sub.min(ins.k - r0);
        launch_tc_conv(g, kn, ins, b, r0, rows)?;
        launch_tc_prepare(g, kn, ins, b, rec, r0, rows)?;
        launch_tc_scan(g, kn, ins, b, rec, r0, rows)?;
    }
    Ok(())
}

fn run_loop(g: &dyn GpuBackend, kn: &Kernels, ins: &Inputs) -> Result<Captured> {
    let b = Bufs::new(g, ins, 0xAB)?;
    launch_conv_rows(g, kn, ins, &b)?;
    launch_recurrent_rows(g, kn.recurrent_rows, smem_bytes(), ins, &b)?;
    let cap = b.capture(g, ins)?;
    b.free(g);
    Ok(cap)
}

fn run_tc(g: &dyn GpuBackend, kn: &Kernels, ins: &Inputs, sub: usize) -> Result<Captured> {
    let b = Bufs::new(g, ins, 0xCD)?;
    let rec = Records::new(g, ins.heads, sub.min(ins.k))?;
    launch_tc(g, kn, ins, &b, &rec, sub)?;
    let cap = b.capture(g, ins)?;
    rec.free(g);
    b.free(g);
    Ok(cap)
}

/// 2026-10-01: `(cosine, max |a - b|, max |a|, all of b finite)` over two equal-length vectors;
/// a length mismatch gives cosine 0.
fn closeness(a: &[f32], b: &[f32]) -> (f64, f64, f64, bool) {
    if a.len() != b.len() || a.is_empty() {
        return (0.0, f64::INFINITY, 0.0, false);
    }
    let (mut ab, mut aa, mut bb, mut err, mut amax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
        err = err.max((x - y).abs());
        amax = amax.max(x.abs());
    }
    let cos = if aa > 0.0 && bb > 0.0 {
        ab / (aa.sqrt() * bb.sqrt())
    } else {
        0.0
    };
    (cos, err, amax, b.iter().all(|y| y.is_finite()))
}

/// 2026-10-01: Mean wall-clock milliseconds per call of `f`: `TIME_WARMUP` untimed calls, then
/// `TIME_REPS` timed calls back to back on stream 0.
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

/// 2026-10-01: The `TIMING` line for `rows` rows. Repeated launches advance the states; the work
/// per launch does not change.
fn timing(g: &dyn GpuBackend, kn: &Kernels, rows: usize) -> Result<()> {
    let ins = gen_inputs(HEADS, rows, SEEDS[0], Gates::Mild);
    let b = Bufs::new(g, &ins, 0)?;
    let rec = Records::new(g, HEADS, rows)?;
    let loop_conv = time_ms(g, || launch_conv_rows(g, kn, &ins, &b))?;
    let loop_recur = time_ms(g, || {
        launch_recurrent_rows(g, kn.recurrent_rows, smem_bytes(), &ins, &b)
    })?;
    let pf_recur = kn
        .recurrent_pf
        .map(|pf| time_ms(g, || launch_recurrent_rows(g, pf, pf_smem_bytes(), &ins, &b)))
        .transpose()?;
    let tc_conv = time_ms(g, || launch_tc_conv(g, kn, &ins, &b, 0, rows))?;
    let tc_prepare = time_ms(g, || launch_tc_prepare(g, kn, &ins, &b, &rec, 0, rows))?;
    let tc_scan = time_ms(g, || launch_tc_scan(g, kn, &ins, &b, &rec, 0, rows))?;
    let tc_total = time_ms(g, || launch_tc(g, kn, &ins, &b, &rec, rows))?;
    let tc_sub = time_ms(g, || launch_tc(g, kn, &ins, &b, &rec, TIME_SUB))?;
    let loop_total = loop_conv + pf_recur.unwrap_or(loop_recur);
    let pf = pf_recur.map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"));
    eprintln!(
        "TIMING rows={rows} heads={HEADS} loop_conv_ms={loop_conv:.3} \
         loop_recur_ms={loop_recur:.3} pf_recur_ms={pf} tc_conv_ms={tc_conv:.3} \
         tc_prepare_ms={tc_prepare:.3} tc_scan_ms={tc_scan:.3} tc_total_ms={tc_total:.3} \
         tc_sub{TIME_SUB}_total_ms={tc_sub:.3} loop_total_ms={loop_total:.3} \
         speedup={:.2}x speedup_sub{TIME_SUB}={:.2}x",
        loop_total / tc_total,
        loop_total / tc_sub
    );
    rec.free(g);
    b.free(g);
    Ok(())
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    // 2026-10-01: Hard lookups: a missing kernel is a FAIL here, not a silent fallback.
    let kn = Kernels {
        conv_rows: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_rows")?,
        recurrent_rows: g.kernel("kda_recurrent", "kda_recurrent_prefill_bf16_smem")?,
        recurrent_pf: g
            .kernel("kda_recurrent", "kda_recurrent_prefill_bf16_pf")
            .ok(),
        tc_conv: g.kernel("kda_chunk_tc", "kda_tc_conv_rows")?,
        tc_conv_tail: g.kernel("kda_chunk_tc", "kda_tc_conv_state_tail")?,
        tc_prepare: g.kernel("kda_chunk_tc", "kda_tc_prepare")?,
        tc_scan: g.kernel("kda_chunk_tc", "kda_tc_scan")?,
    };

    let mut all_ok = true;
    let mut cases = 0usize;
    for &(k, sub) in &CASES {
        for gates in [Gates::Mild, Gates::Full] {
            for &seed in &SEEDS {
                let ins = gen_inputs(HEADS, k, seed, gates);
                let lp = run_loop(g, &kn, &ins)?;
                let tc = run_tc(g, &kn, &ins, sub)?;

                // 2026-10-01: Non-vacuity: the loop produced finite, non-zero outputs and moved
                // both states.
                let lp_core = floats(&lp.core);
                let live =
                    lp_core.iter().all(|x| x.is_finite()) && lp_core.iter().any(|x| *x != 0.0);
                let h0: Vec<u8> = ins.h_state0.iter().flat_map(|x| x.to_le_bytes()).collect();
                let c0: Vec<u8> = ins.conv_state0.iter().flat_map(|x| x.to_le_bytes()).collect();
                let moved = lp.h_state != h0 && lp.conv_state != c0;

                let conv_eq = words_u16(&lp.conv_out) == words_u16(&tc.conv_out);
                let cstate_eq = words_u32(&lp.conv_state) == words_u32(&tc.conv_state);
                let (c_cos, c_err, c_max, c_fin) = closeness(&lp_core, &floats(&tc.core));
                let (s_cos, s_err, s_max, s_fin) =
                    closeness(&floats(&lp.h_state), &floats(&tc.h_state));
                let ok = live
                    && moved
                    && conv_eq
                    && cstate_eq
                    && c_fin
                    && s_fin
                    && c_cos > COS_MIN
                    && s_cos > COS_MIN;
                all_ok &= ok;
                cases += 1;
                eprintln!(
                    "rows={k:>4} sub={sub:>4} gates={gates:?} seed={seed:#x} conv_out={} \
                     conv_state={} core cos={c_cos:.7} maxerr={c_err:.3e} \
                     (max|ref|={c_max:.3e}) state cos={s_cos:.7} maxerr={s_err:.3e} \
                     (max|ref|={s_max:.3e}){}  {}",
                    if conv_eq { "eq" } else { "MISMATCH" },
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
    }
    if cases == 0 {
        bail!("ran 0 cases; refusing to report PASS");
    }
    eprintln!(
        "\nKDA chunked-TC GATE ({cases} cases; conv bitwise, core and state cosine > {COS_MIN} \
         vs the token loop): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );

    // 2026-10-01: Timing; reported whatever the verdict.
    for rows in TIME_ROWS {
        timing(g, &kn, rows)?;
    }
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
