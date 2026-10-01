// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Byte-identity gate for the opt-in GLM KDA token loop
//! (`METRALE_GLM_KDA_TOKEN_LOOP=1`, `Glm5NextKdaLayer::stateful_rows`) against the per-row walk
//! it replaces (`Glm5NextKdaLayer::stateful_row` once per row).
//!
//! Owner: model-arch examples (GLM-5.3 KDA kernels).
//! Invariants:
//! - Exits nonzero unless, for every head count, row count and seed, the conv outputs
//!   (`[k, conv_dim]` BF16), the recurrent outputs (`[k, heads * head_dim]` FP32), the final
//!   recurrent state (`[heads, head_dim, head_dim]` FP32) and the final conv state
//!   (`[conv_dim, conv_kernel]` FP32) are identical bit for bit (u16 / u32 compares) between the
//!   two arms.
//! - Refuses to report PASS when it compared zero elements, when the walk's outputs are all zero
//!   or not finite, or when the walk left either state unchanged (a kernel that never ran would
//!   otherwise match a twin that never ran).
//!
//! Arms, on the same device inputs and the same starting states:
//! - WALK: for each row, `causal_conv1d_update_l2norm` at batch 1 on that row, then
//!   `kda_recurrent_decode_bf16_smem` on that row: the launches `stateful_row` makes on its
//!   shared-memory branch, with its grid, block, shared memory and per-row pointers transcribed.
//! - LOOP: `causal_conv1d_update_l2norm_rows` over all rows, then
//!   `kda_recurrent_prefill_bf16_smem` over all rows: the two launches `stateful_rows` makes,
//!   with its arguments transcribed.
//!
//! The output buffers are filled before each arm, with 0xAB for WALK and 0xCD for LOOP, so a byte
//! that either side leaves unwritten cannot pass as a match.
//!
//! Geometry: GLM-5.3-Flash KDA (head_dim 128, conv_kernel 4, l2_eps 1e-6), at 32 heads (the TP=2
//! per-rank layer the 2x GB10 serve runs) and 64 heads (TP=1).
//!
//!   cargo run -p metrale-model-arch --release --example kda_tokenloop_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const D: usize = 128;
const D_CONV: usize = 4;
const L2_EPS: f32 = 1e-6;
/// 2026-10-01: `KDA_V_PER_BLOCK` in glm5next_kda/mod.rs.
const VPB: usize = 32;

const HEADS: [usize; 2] = [32, 64];
const ROWS: [usize; 4] = [1, 2, 16, 256];
const SEEDS: [u64; 2] = [0x4B_DA70_0001, 0x4B_DA70_0002];

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

fn gen_inputs(heads: usize, k: usize, seed: u64) -> Inputs {
    let mut r = Lcg(seed ^ ((heads as u64) << 32) ^ (k as u64));
    let qkv = heads * D;
    let cd = 3 * qkv;
    Inputs {
        heads,
        k,
        qkv_proj: (0..k * cd).map(|_| bf16::from_f64(r.r(-0.5, 0.5))).collect(),
        conv_weight: (0..cd * D_CONV)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect(),
        conv_state0: (0..cd * D_CONV).map(|_| r.r(-0.3, 0.3) as f32).collect(),
        // 2026-10-01: kda_gate's output is a bounded log-decay (<= 0); the kernel exponentiates.
        gate: (0..k * qkv).map(|_| r.r(-0.3, -0.005) as f32).collect(),
        beta: (0..k * heads).map(|_| r.r(0.05, 0.95) as f32).collect(),
        h_state0: (0..heads * D * D).map(|_| r.r(-0.1, 0.1) as f32).collect(),
    }
}

struct Kernels {
    conv_decode: KernelHandle,
    recurrent_smem: KernelHandle,
    conv_rows: KernelHandle,
    recurrent_rows: KernelHandle,
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

fn smem_bytes() -> u32 {
    ((3 * D + VPB * (D + 1)) * 4) as u32
}

/// 2026-10-01: WALK: `stateful_row`'s shared-memory branch, once per row, in row order.
fn run_walk(g: &dyn GpuBackend, kn: &Kernels, ins: &Inputs) -> Result<Captured> {
    let b = Bufs::new(g, ins, 0xAB)?;
    let (cd, qkv, h) = (ins.cd(), ins.qkv(), ins.heads);
    for row in 0..ins.k {
        // 2026-10-01: ops::conv1d_update_l2norm at batch_size 1, null bias.
        KernelLaunch::new(g, kn.conv_decode)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(b.conv_state)
            .arg_ptr(b.qkv_proj.offset(row * cd * 2))
            .arg_ptr(b.conv_w)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(b.conv_out.offset(row * cd * 2))
            .arg_u32(1)
            .arg_u32(cd as u32)
            .arg_u32(D_CONV as u32)
            .arg_u32((2 * qkv) as u32)
            .arg_u32(D as u32)
            .arg_f32(L2_EPS)
            .launch(0)?;
        KernelLaunch::new(g, kn.recurrent_smem)
            .grid([h as u32, (D / VPB) as u32, 1])
            .block([VPB as u32, 1, 1])
            .shared_mem(smem_bytes())
            .arg_ptr(b.conv_out.offset(row * cd * 2))
            .arg_ptr(b.conv_out.offset(row * cd * 2 + qkv * 2))
            .arg_ptr(b.conv_out.offset(row * cd * 2 + qkv * 4))
            .arg_ptr(b.gate.offset(row * qkv * 4))
            .arg_ptr(b.beta.offset(row * h * 4))
            .arg_ptr(b.h_state)
            .arg_ptr(b.core.offset(row * qkv * 4))
            .arg_u32(h as u32)
            .arg_u32(D as u32)
            .arg_f32(1.0 / (D as f32).sqrt())
            .arg_u32(VPB as u32)
            .launch(0)?;
    }
    let cap = b.capture(g, ins)?;
    b.free(g);
    Ok(cap)
}

/// 2026-10-01: LOOP: `stateful_rows`, two launches for all rows.
fn run_loop(g: &dyn GpuBackend, kn: &Kernels, ins: &Inputs) -> Result<Captured> {
    let b = Bufs::new(g, ins, 0xCD)?;
    let (cd, qkv, h, k) = (ins.cd(), ins.qkv(), ins.heads, ins.k);
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
    KernelLaunch::new(g, kn.recurrent_rows)
        .grid([h as u32, (D / VPB) as u32, 1])
        .block([VPB as u32, 1, 1])
        .shared_mem(smem_bytes())
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
    let cap = b.capture(g, ins)?;
    b.free(g);
    Ok(cap)
}

/// 2026-10-01: `(elements compared, first mismatching index)`. A length mismatch compares
/// nothing and reports index 0.
fn compare<T: PartialEq>(a: &[T], b: &[T]) -> (usize, Option<usize>) {
    if a.len() != b.len() {
        return (0, Some(0));
    }
    (a.len(), a.iter().zip(b).position(|(x, y)| x != y))
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    // 2026-10-01: Hard lookups: a missing row kernel is a FAIL here, not a silent fallback.
    let kn = Kernels {
        conv_decode: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
        recurrent_smem: g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_smem")?,
        conv_rows: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_rows")?,
        recurrent_rows: g.kernel("kda_recurrent", "kda_recurrent_prefill_bf16_smem")?,
    };

    let mut all_ok = true;
    let mut total_compared = 0usize;
    for &heads in &HEADS {
        for &k in &ROWS {
            for &seed in &SEEDS {
                let ins = gen_inputs(heads, k, seed);
                let walk = run_walk(g, &kn, &ins)?;
                let lp = run_loop(g, &kn, &ins)?;

                let checks = [
                    ("conv_out", compare(&words_u16(&walk.conv_out), &words_u16(&lp.conv_out))),
                    ("core", compare(&words_u32(&walk.core), &words_u32(&lp.core))),
                    ("h_state", compare(&words_u32(&walk.h_state), &words_u32(&lp.h_state))),
                    (
                        "conv_state",
                        compare(&words_u32(&walk.conv_state), &words_u32(&lp.conv_state)),
                    ),
                ];

                // 2026-10-01: Non-vacuity: the walk produced finite, non-zero outputs and moved
                // both states.
                let core_f: Vec<f32> = words_u32(&walk.core)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                let core_live =
                    core_f.iter().all(|x| x.is_finite()) && core_f.iter().any(|x| *x != 0.0);
                let conv_f: Vec<f32> = words_u16(&walk.conv_out)
                    .into_iter()
                    .map(|b| bf16::from_bits(b).to_f32())
                    .collect();
                let conv_live =
                    conv_f.iter().all(|x| x.is_finite()) && conv_f.iter().any(|x| *x != 0.0);
                let h0: Vec<u8> = ins.h_state0.iter().flat_map(|x| x.to_le_bytes()).collect();
                let c0: Vec<u8> = ins.conv_state0.iter().flat_map(|x| x.to_le_bytes()).collect();
                let moved = walk.h_state != h0 && walk.conv_state != c0;

                let mut ok = core_live && conv_live && moved;
                let mut line = format!("heads={heads:>2} k={k:>3} seed={seed:#x} ");
                for (name, (n, first)) in &checks {
                    total_compared += n;
                    ok &= *n > 0 && first.is_none();
                    match first {
                        None => line.push_str(&format!(" {name}={n}/eq")),
                        Some(i) => line.push_str(&format!(" {name}={n}/MISMATCH@{i}")),
                    }
                }
                if !(core_live && conv_live && moved) {
                    line.push_str(&format!(
                        "  VACUOUS(core_live={core_live} conv_live={conv_live} moved={moved})"
                    ));
                }
                all_ok &= ok;
                eprintln!("{line}  {}", if ok { "PASS" } else { "FAIL" });
            }
        }
    }

    if total_compared == 0 {
        bail!("compared 0 elements; refusing to report PASS");
    }
    eprintln!(
        "\nKDA token loop GATE (bitwise vs per-row walk, {total_compared} elements compared): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
