// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Gate G0 for GLM-5.3 routed-expert tensor parallelism (expert-TP): on one GPU,
//! does sweeping the WHOLE expert union at half width (I/2 = 1024) cost no more than an
//! EP-balanced rank sweeping half the union at full width (I = 2048)?
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants: none beyond the types. Timing is reported; only the M = 3 ratio is gated.
//!
//! Design input: `spark-bench/runs/race/nccllat/L12-20261005T1758/EXPERT-TP-DESIGN.md`
//! (standard Megatron / vLLM FusedMoE expert TP: gate/up split by rows of I, down by columns
//! of I; the existing post-MoE all-reduce sums the two partials).
//!
//! Path timed: the decode "row-batched expert union, one sweep" path of `forward_moe`
//! (`glm5next_mlp/forward/moe_experts.rs::row_batched_experts`), launched here with the same
//! kernels, grids and strides: the `expert_out` memset, then per `moe_row_groups` sub-group
//! `glm5next_moe_row_union`, `w4a16_gemv_sw_moe_batchm_m<R>` for gate and up,
//! `glm5next_swiglu_clamp`, and `w4a16_gemv_sw_moe_batchm_m<R>` for down. Kernel handles come
//! from `Glm5NextMlpKernels::resolve`. Router, shared expert and combine are not timed (the
//! design leaves them unchanged). An expert another rank owns is a NULL pointer-table entry,
//! exactly as the EP2 loader leaves it, so the kernels skip it.
//!
//! Arms, per M in {3, 12}, over the same seeded routing (each row: 8 distinct of 288,
//! uniform), `NDRAWS` draws, each call a different draw:
//!   A_bal  I = 2048, ceil(U/2) of the union U local (a perfectly balanced EP rank);
//!   A_max  I = 2048, the busier rank of the contiguous split 0..144 / 144..288 (today's EP2
//!          critical path);
//!   B      I = 1024, the whole union local (an expert-TP rank).
//!
//! Cold pool: all 288 experts are resident at both widths (I = 2048: 288 x 13.5 MiB = 3.9 GB;
//! I = 1024: 1.9 GB), against GB10's 24 MB L2, and consecutive calls take different routing
//! draws, so each call streams its experts from DRAM.
//!
//! Slice check: for draw 0 at M = 3, the full-width routed output against the sum of the two
//! half-width partials, where the half-width weights are true slices of the full-width ones
//! (gate/up rows [0, 1024) and [1024, 2048); down packed-K bytes [0, 512) / [512, 1024) and
//! scale columns [0, 64) / [64, 128) of each row). Cosine >= 0.9999 or exit 1.
//!
//! Output: one `G0 M=...` line per M, then the M = 3 verdict (`PASS: ...` or
//! `G0 RESULT: ... (bar not met)`); exit 0 either way. Nonzero exit only on a real error.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_expert_tp_g0_microtest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mlp::Glm5NextMlpKernels;
use metrale_model_arch::glm5next_mlp::forward::{
    MOE_ROW_BATCH_MAX_ROWS, MOE_ROW_UNION_MAX_IDS, row_batch_max,
};

// 2026-10-05: CUDA driver event API, declared as in `glm5next_l2_prefetch_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const STREAM: u64 = 0;
/// 2026-10-05: GLM-5.3 shapes, `model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json`:
/// hidden_size 4096, moe_intermediate_size 2048, n_routed_experts 288, num_experts_per_tok 8,
/// swiglu_limit 10.0.
const H: usize = 4096;
const NUM_EXPERTS: usize = 288;
const TOP_K: usize = 8;
const I_FULL: usize = 2048;
const I_HALF: usize = I_FULL / 2;
const SWIGLU_LIMIT: f32 = 10.0;
/// 2026-10-05: EP2 contiguous split: rank 0 owns 0..144, rank 1 owns 144..288.
const EP_SPLIT: usize = NUM_EXPERTS / 2;
const MS: [usize; 2] = [3, 12];
const NDRAWS: usize = 256;
/// 2026-10-05: Timed passes over all draws per arm, after one warm-up pass; the arm order
/// rotates each round.
const ROUNDS: usize = 4;
const ROUTE_SEED: u64 = 0x6730_6574_7032_0005;
const BAR: f64 = 1.05;
const COS_MIN: f64 = 0.9999;
/// 2026-10-05: `ACT_BLOCK` in `glm5next_mlp/forward.rs`, the SwiGLU launch block.
const ACT_BLOCK: u32 = 256;
/// 2026-10-05: Bytes of one pointer-table block: packed u64[288], scale u64[288], scale2
/// f32[288] padded to 8 bytes per entry.
const TBL_BYTES: usize = NUM_EXPERTS * 24;
/// 2026-10-05: Three blocks (gate, up, down) per table set.
const SET_BYTES: usize = 3 * TBL_BYTES;

fn check(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} returned CUDA status {rc}");
    }
    Ok(())
}

/// 2026-10-05: Two CUDA events on stream 0.
struct Events([u64; 2]);

impl Events {
    fn new() -> Result<Self> {
        let mut e = [0u64; 2];
        for x in e.iter_mut() {
            // SAFETY: plain CUDA driver call writing one event handle; the backend made the
            // context current when it was built.
            check(unsafe { cuEventCreate(x, 0) }, "cuEventCreate")?;
        }
        Ok(Self(e))
    }

    fn rec(&self, i: usize) -> Result<()> {
        // SAFETY: records an event this struct created on the default stream.
        check(unsafe { cuEventRecord(self.0[i], STREAM) }, "cuEventRecord")
    }

    /// 2026-10-05: Microseconds from event 0 to event 1, after event 1 completes.
    fn us(&self) -> Result<f64> {
        let mut ms: f32 = 0.0;
        // SAFETY: both events were created and recorded by this struct; `ms` outlives the call.
        unsafe {
            check(cuEventSynchronize(self.0[1]), "cuEventSynchronize")?;
            check(
                cuEventElapsedTime(&mut ms, self.0[0], self.0[1]),
                "cuEventElapsedTime",
            )?;
        }
        Ok(ms as f64 * 1e3)
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        for &e in &self.0 {
            // SAFETY: destroys an event this struct created; errors are ignored on drop.
            unsafe {
                cuEventDestroy_v2(e);
            }
        }
    }
}

/// 2026-10-05: Routing RNG.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u32() as u64 * n as u64) >> 32) as usize
    }
    fn unit(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }
}

/// 2026-10-05: Fast weight-byte generator (splitmix64).
fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fill_packed(buf: &mut [u8], seed: u64) {
    let mut s = seed;
    for c in buf.chunks_mut(8) {
        let v = splitmix(&mut s).to_le_bytes();
        let n = c.len();
        c.copy_from_slice(&v[..n]);
    }
}

/// 2026-10-05: E4M3 codes 0x38..=0x3F (1.0..=1.875): no zero, inf or NaN scale.
fn fill_scale(buf: &mut [u8], seed: u64) {
    let mut s = seed;
    for c in buf.chunks_mut(8) {
        let v = splitmix(&mut s);
        for (i, b) in c.iter_mut().enumerate() {
            *b = 0x38 | (((v >> (8 * i)) as u8) & 0x07);
        }
    }
}

/// 2026-10-05: Per-tensor FP32 scale2 of expert `e`, shared by both slices. With E2M1 up to
/// 6, scales up to 1.875 and inputs in [-1, 1], 0.01 keeps the gate projections near unit
/// size, well inside the SwiGLU clamp of 10.
fn scale2_of(e: usize) -> f32 {
    0.01 * (1.0 + (e % 7) as f32 * 0.05)
}

/// 2026-10-05: One expert's full-width (I = 2048) NVFP4 tensors on the host, in the
/// `w4a16_gemv.cu` layout: packed `[N, K/2]` (low nibble = even K), scale `[N, K/16]`.
/// gate/up are `[I, H]`, down is `[H, I]`.
struct ExpertHost {
    p: [Vec<u8>; 3],
    s: [Vec<u8>; 3],
}

fn gen_expert(e: usize) -> ExpertHost {
    let pk = I_FULL * H / 2;
    let sc = I_FULL * H / 16;
    let mk = |proj: u64, kind: u64| ((e as u64) << 8) | (proj << 4) | kind;
    let mut p = [vec![0u8; pk], vec![0u8; pk], vec![0u8; pk]];
    let mut s = [vec![0u8; sc], vec![0u8; sc], vec![0u8; sc]];
    for j in 0..3 {
        fill_packed(&mut p[j], 0xE7_0000_0000 ^ mk(j as u64, 1));
        fill_scale(&mut s[j], 0xE7_0000_0000 ^ mk(j as u64, 2));
    }
    ExpertHost { p, s }
}

/// 2026-10-05: Rank `r`'s expert-TP slice of `x`: gate/up (proj 0, 1) rows
/// `[r * I/2, (r + 1) * I/2)`; down (proj 2) columns `[r * I/2, (r + 1) * I/2)` of every row,
/// i.e. packed bytes `[r * I/4, (r + 1) * I/4)` and scale columns `[r * I/32, (r + 1) * I/32)`.
fn slice_expert(x: &ExpertHost, r: usize) -> ExpertHost {
    let rows = |b: &[u8], row_bytes: usize| -> Vec<u8> {
        b[r * I_HALF * row_bytes..(r + 1) * I_HALF * row_bytes].to_vec()
    };
    let cols = |b: &[u8], row_bytes: usize| -> Vec<u8> {
        let w = row_bytes / 2;
        let mut out = Vec::with_capacity(H * w);
        for n in 0..H {
            let row = &b[n * row_bytes..(n + 1) * row_bytes];
            out.extend_from_slice(&row[r * w..(r + 1) * w]);
        }
        out
    };
    ExpertHost {
        p: [
            rows(x.p[0].as_slice(), H / 2),
            rows(x.p[1].as_slice(), H / 2),
            cols(x.p[2].as_slice(), I_FULL / 2),
        ],
        s: [
            rows(x.s[0].as_slice(), H / 16),
            rows(x.s[1].as_slice(), H / 16),
            cols(x.s[2].as_slice(), I_FULL / 16),
        ],
    }
}

/// 2026-10-05: `count` experts at width `mi` on the device, one allocation per tensor kind;
/// slot `j` of tensor (proj, packed|scale) starts at `j * bytes`.
struct Pool {
    mi: usize,
    count: usize,
    p: [DevicePtr; 3],
    s: [DevicePtr; 3],
}

impl Pool {
    fn new(g: &dyn GpuBackend, mi: usize, count: usize) -> Result<Self> {
        let (pk, sc) = Self::bytes(mi);
        Ok(Self {
            mi,
            count,
            p: [
                g.alloc(count * pk)?,
                g.alloc(count * pk)?,
                g.alloc(count * pk)?,
            ],
            s: [
                g.alloc(count * sc)?,
                g.alloc(count * sc)?,
                g.alloc(count * sc)?,
            ],
        })
    }
    /// 2026-10-05: (packed, scale) bytes of one projection at width `mi`; gate/up `[mi, H]`
    /// and down `[H, mi]` have the same counts.
    fn bytes(mi: usize) -> (usize, usize) {
        (mi * H / 2, mi * H / 16)
    }
    fn total_bytes(&self) -> usize {
        let (pk, sc) = Self::bytes(self.mi);
        self.count * 3 * (pk + sc)
    }
    fn put(&self, g: &dyn GpuBackend, slot: usize, x: &ExpertHost) -> Result<()> {
        let (pk, sc) = Self::bytes(self.mi);
        for j in 0..3 {
            if x.p[j].len() != pk || x.s[j].len() != sc {
                bail!("expert slice size does not match a width-{} pool", self.mi);
            }
            g.copy_h2d(&x.p[j], self.p[j].offset(slot * pk))?;
            g.copy_h2d(&x.s[j], self.s[j].offset(slot * sc))?;
        }
        Ok(())
    }
    fn addr(&self, proj: usize, slot: usize) -> (u64, u64) {
        let (pk, sc) = Self::bytes(self.mi);
        (
            self.p[proj].offset(slot * pk).0,
            self.s[proj].offset(slot * sc).0,
        )
    }
}

/// 2026-10-05: Host bytes of one table set (gate, up, down) over `pool`: `map[e]` is the pool
/// slot of global expert `e`, or `None` for an expert another rank owns (NULL entry).
fn table_set_bytes(pool: &Pool, map: &[Option<usize>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SET_BYTES);
    for proj in 0..3 {
        let mut packed = Vec::with_capacity(NUM_EXPERTS * 8);
        let mut scale = Vec::with_capacity(NUM_EXPERTS * 8);
        let mut s2 = Vec::with_capacity(NUM_EXPERTS * 8);
        for (e, m) in map.iter().enumerate() {
            let (p, s, v) = match m {
                Some(slot) => {
                    let (p, s) = pool.addr(proj, *slot);
                    (p, s, scale2_of(e))
                }
                None => (0u64, 0u64, 0.0f32),
            };
            packed.extend_from_slice(&p.to_le_bytes());
            scale.extend_from_slice(&s.to_le_bytes());
            s2.extend_from_slice(&v.to_le_bytes());
        }
        s2.resize(NUM_EXPERTS * 8, 0);
        out.extend_from_slice(&packed);
        out.extend_from_slice(&scale);
        out.extend_from_slice(&s2);
    }
    out
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(16))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(STREAM)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-10-05: One table set on the device: block `proj` is (packed_ptrs, scale_ptrs,
/// scale2_vals).
#[derive(Clone, Copy)]
struct Tables(DevicePtr);
impl Tables {
    fn proj(self, j: usize) -> (DevicePtr, DevicePtr, DevicePtr) {
        let b = self.0.offset(j * TBL_BYTES);
        (
            b,
            b.offset(NUM_EXPERTS * 8),
            b.offset(NUM_EXPERTS * 16),
        )
    }
}

/// 2026-10-05: The routed-expert scratch `row_batched_experts` uses, sized for 12 rows at
/// I = 2048.
struct Ws {
    a_gate: DevicePtr,
    a_up: DevicePtr,
    a_act: DevicePtr,
    u_eid: DevicePtr,
    u_slot: DevicePtr,
}

/// 2026-10-05: `moe_row_groups` of `glm5next_mlp/forward.rs` (private there): `rows` split
/// into consecutive `(start, width)` sub-groups of at most `cap`, as even as the count allows.
fn moe_row_groups(rows: usize, cap: usize) -> Vec<(usize, usize)> {
    let n = rows.div_ceil(cap.max(1)).max(1);
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    for i in 0..n {
        let w = (rows - start).div_ceil(n - i);
        out.push((start, w));
        start += w;
    }
    out
}

/// 2026-10-05: `moe_row_groups(rows, cap)`, refused unless every sub-group is on the
/// row-batched path `forward_moe` takes (width 2..=8, `width * top_k <= 64`).
fn checked_groups(rows: usize, cap: usize) -> Result<Vec<(usize, usize)>> {
    let groups = moe_row_groups(rows, cap);
    for &(_, w) in &groups {
        if w < 2 || w > MOE_ROW_BATCH_MAX_ROWS || w * TOP_K > MOE_ROW_UNION_MAX_IDS {
            bail!(
                "M={rows}: sub-group width {w} (groups {groups:?}) is not on the row-batched \
                 path; unset METRALE_GLM_MOE_ROW_BATCH_MAX"
            );
        }
    }
    Ok(groups)
}

/// 2026-10-05: The `w4a16_gemv_moe_batchm` launch of `glm5next_mlp/forward/launch.rs`.
#[allow(clippy::too_many_arguments)]
fn batchm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: (DevicePtr, DevicePtr, DevicePtr),
    c: DevicePtr,
    ws: &Ws,
    n: usize,
    kk: usize,
    rows: usize,
    a_row_stride: usize,
    a_slot_stride: usize,
    c_row_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([
            metrale_model_layers::layers::ops::w4a16_gemv_sw_grid_x(n as u32),
            (rows * TOP_K) as u32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.0)
        .arg_ptr(t.1)
        .arg_ptr(t.2)
        .arg_ptr(c)
        .arg_ptr(ws.u_eid)
        .arg_ptr(ws.u_slot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(NUM_EXPERTS as u32)
        .arg_u32(a_row_stride as u32)
        .arg_u32(a_slot_stride as u32)
        .arg_u32(c_row_stride as u32)
        .launch(STREAM)
}

/// 2026-10-05: One routed-expert call as `forward_moe` issues it on the row-batched path:
/// zero `expert_out`, then per sub-group the union build, gate, up, SwiGLU and down
/// (`moe_experts.rs::row_batched_experts`, same strides), at expert width `mi`.
#[allow(clippy::too_many_arguments)]
fn run_moe(
    g: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    ws: &Ws,
    x: DevicePtr,
    ids: DevicePtr,
    rows: usize,
    groups: &[(usize, usize)],
    t: Tables,
    mi: usize,
    expert_out: DevicePtr,
) -> Result<()> {
    g.memset_async(expert_out, 0, rows * TOP_K * H * 2, STREAM)?;
    for &(r0, w_rows) in groups {
        KernelLaunch::new(g, k.moe_row_union)
            .grid([1, 1, 1])
            .block([(w_rows * TOP_K) as u32, 1, 1])
            .arg_ptr(ids.offset(r0 * TOP_K * 4))
            .arg_ptr(ws.u_eid)
            .arg_ptr(ws.u_slot)
            .arg_u32(w_rows as u32)
            .arg_u32(TOP_K as u32)
            .launch(STREAM)?;
        let kb = k.w4a16_gemv_sw_moe_batchm[w_rows - 2];
        let act_off = r0 * TOP_K * mi * 2;
        batchm(
            g,
            kb,
            x.offset(r0 * H * 2),
            t.proj(0),
            ws.a_gate.offset(act_off),
            ws,
            mi,
            H,
            w_rows,
            H,
            0,
            TOP_K * mi,
        )?;
        batchm(
            g,
            kb,
            x.offset(r0 * H * 2),
            t.proj(1),
            ws.a_up.offset(act_off),
            ws,
            mi,
            H,
            w_rows,
            H,
            0,
            TOP_K * mi,
        )?;
        let n_act = w_rows * TOP_K * mi;
        KernelLaunch::new(g, k.swiglu)
            .grid([(n_act as u32).div_ceil(ACT_BLOCK), 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(ws.a_gate.offset(act_off))
            .arg_ptr(ws.a_up.offset(act_off))
            .arg_ptr(ws.a_act.offset(act_off))
            .arg_u32(n_act as u32)
            .arg_f32(SWIGLU_LIMIT)
            .launch(STREAM)?;
        batchm(
            g,
            kb,
            ws.a_act.offset(act_off),
            t.proj(2),
            expert_out.offset(r0 * TOP_K * H * 2),
            ws,
            H,
            mi,
            w_rows,
            TOP_K * mi,
            mi,
            TOP_K * H,
        )?;
    }
    Ok(())
}

/// 2026-10-05: `draws` routing draws of `rows` rows: each row 8 distinct experts of 288.
fn gen_draws(rows: usize, draws: usize, seed: u64) -> Vec<Vec<i32>> {
    let mut rng = Lcg(seed);
    (0..draws)
        .map(|_| {
            let mut ids = Vec::with_capacity(rows * TOP_K);
            for _ in 0..rows {
                let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
                while row.len() < TOP_K {
                    let c = rng.below(NUM_EXPERTS) as i32;
                    if !row.contains(&c) {
                        row.push(c);
                    }
                }
                ids.extend_from_slice(&row);
            }
            ids
        })
        .collect()
}

fn union_of(ids: &[i32]) -> Vec<usize> {
    let mut u: Vec<usize> = ids.iter().map(|&e| e as usize).collect();
    u.sort_unstable();
    u.dedup();
    u
}

fn bf16_vals(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect()
}

fn main() -> Result<()> {
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let backend = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextMlpKernels::resolve(g)?;
    if k.moe_row_union.0 == 0 {
        bail!("glm5next_moe_row_union is absent from this target's PTX");
    }
    if k.w4a16_gemv_sw_moe_batchm.iter().any(|h| h.0 == 0) {
        bail!("a w4a16_gemv_sw_moe_batchm_m<R> tier (2..=8) is absent from this target's PTX");
    }
    let cap = row_batch_max();
    println!(
        "G0 setup: H={H} experts={NUM_EXPERTS} top_k={TOP_K} I={I_FULL} (expert-TP I/2={I_HALF}) \
         draws={NDRAWS} rounds={ROUNDS} row_batch_max={cap} (default {MOE_ROW_BATCH_MAX_ROWS})"
    );

    // 2026-10-05: Cold pools. `full` holds every expert at I = 2048 (the A arms); `half0`
    // holds rank 0's expert-TP slice of the same weights (arm B, and rank 0 of the check).
    let full = Pool::new(g, I_FULL, NUM_EXPERTS)?;
    let half0 = Pool::new(g, I_HALF, NUM_EXPERTS)?;
    for e in 0..NUM_EXPERTS {
        let x = gen_expert(e);
        full.put(g, e, &x)?;
        half0.put(g, e, &slice_expert(&x, 0))?;
    }
    g.synchronize(STREAM)?;
    println!(
        "G0 setup: cold pool I={I_FULL} {:.2} GB, I={I_HALF} {:.2} GB (L2 24 MB); every call \
         takes the next routing draw",
        full.total_bytes() as f64 / 1e9,
        half0.total_bytes() as f64 / 1e9
    );

    let max_rows = *MS.iter().max().unwrap_or(&12);
    let ws = Ws {
        a_gate: g.alloc(max_rows * TOP_K * I_FULL * 2)?,
        a_up: g.alloc(max_rows * TOP_K * I_FULL * 2)?,
        a_act: g.alloc(max_rows * TOP_K * I_FULL * 2)?,
        u_eid: g.alloc(max_rows * TOP_K * 4)?,
        u_slot: g.alloc(max_rows * TOP_K * max_rows * 4)?,
    };
    let eo_bytes = max_rows * TOP_K * H * 2;
    let eo = g.alloc(eo_bytes)?;
    let eo_r1 = g.alloc(eo_bytes)?;

    let mut xr = Lcg(0x6730_0000_0000_00A1);
    let x_host: Vec<u8> = (0..max_rows * H)
        .flat_map(|_| {
            bf16::from_f64(xr.unit() * 2.0 - 1.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let x = upload(g, &x_host)?;

    let all_full: Vec<Option<usize>> = (0..NUM_EXPERTS).map(Some).collect();
    let t_b = Tables(upload(g, &table_set_bytes(&half0, &all_full))?);
    let t_full = Tables(upload(g, &table_set_bytes(&full, &all_full))?);

    // 2026-10-05: Slice check, M = 3, draw 0 of the M = 3 routing.
    {
        let rows = 3usize;
        let groups = checked_groups(rows, cap)?;
        let check_draws = gen_draws(rows, 1, ROUTE_SEED ^ rows as u64);
        let draw = &check_draws[0];
        let un = union_of(draw);
        let ids = upload(
            g,
            &draw
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        // 2026-10-05: Rank 1's slices of the union experts only.
        let half1 = Pool::new(g, I_HALF, un.len())?;
        let mut map1: Vec<Option<usize>> = vec![None; NUM_EXPERTS];
        for (j, &e) in un.iter().enumerate() {
            half1.put(g, j, &slice_expert(&gen_expert(e), 1))?;
            map1[e] = Some(j);
        }
        let t_r1 = Tables(upload(g, &table_set_bytes(&half1, &map1))?);
        let n = rows * TOP_K * H * 2;

        run_moe(g, &k, &ws, x, ids, rows, &groups, t_full, I_FULL, eo)?;
        let full_out = bf16_vals(&download(g, eo, n)?);
        run_moe(g, &k, &ws, x, ids, rows, &groups, t_b, I_HALF, eo)?;
        run_moe(g, &k, &ws, x, ids, rows, &groups, t_r1, I_HALF, eo_r1)?;
        let p0 = bf16_vals(&download(g, eo, n)?);
        let p1 = bf16_vals(&download(g, eo_r1, n)?);

        // 2026-10-05: Combine with seeded routing weights, as `glm5next_moe_combine` would.
        let mut wr = Lcg(0x6730_0000_0000_00C3);
        let wts: Vec<f64> = (0..rows * TOP_K).map(|_| 0.05 + wr.unit()).collect();
        let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        let (mut maxd, mut maxr) = (0.0f64, 0.0f64);
        let mut nonfinite = 0usize;
        for r in 0..rows {
            for h in 0..H {
                let (mut yf, mut yt) = (0.0f64, 0.0f64);
                for s in 0..TOP_K {
                    let i = (r * TOP_K + s) * H + h;
                    let w = wts[r * TOP_K + s];
                    yf += w * full_out[i];
                    yt += w * (p0[i] + p1[i]);
                }
                if !yf.is_finite() || !yt.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                dot += yf * yt;
                aa += yf * yf;
                bb += yt * yt;
                maxd = maxd.max((yf - yt).abs());
                maxr = maxr.max(yf.abs());
            }
        }
        let cos = if aa > 0.0 && bb > 0.0 {
            dot / (aa.sqrt() * bb.sqrt())
        } else {
            0.0
        };
        println!(
            "G0 slice check M=3 union={} cosine={cos:.7} max_rel={:.5} nonfinite={nonfinite} \
             |full|_max={maxr:.4}",
            un.len(),
            maxd / maxr.max(1e-30)
        );
        if cos.is_nan() || cos < COS_MIN || nonfinite > 0 || aa == 0.0 {
            println!("G0 RESULT: slice math mismatch cosine={cos:.7}");
            std::process::exit(1);
        }
    }

    let ev = Events::new()?;
    let mut ratio_m3 = f64::NAN;
    for &rows in &MS {
        let groups = checked_groups(rows, cap)?;
        let draws = gen_draws(rows, NDRAWS, ROUTE_SEED ^ rows as u64);
        let ids_all = upload(
            g,
            &draws
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let ids_of = |d: usize| ids_all.offset(d * rows * TOP_K * 4);

        let mut bal_bytes = Vec::with_capacity(NDRAWS * SET_BYTES);
        let mut max_bytes = Vec::with_capacity(NDRAWS * SET_BYTES);
        let (mut sum_u, mut sum_bal, mut sum_max) = (0usize, 0usize, 0usize);
        for d in &draws {
            let un = union_of(d);
            let half = un.len().div_ceil(2);
            let mut bal: Vec<Option<usize>> = vec![None; NUM_EXPERTS];
            for &e in &un[..half] {
                bal[e] = Some(e);
            }
            let c0 = un.iter().filter(|&&e| e < EP_SPLIT).count();
            let c1 = un.len() - c0;
            let busier: Vec<Option<usize>> = (0..NUM_EXPERTS)
                .map(|e| ((e < EP_SPLIT) == (c0 >= c1)).then_some(e))
                .collect();
            bal_bytes.extend_from_slice(&table_set_bytes(&full, &bal));
            max_bytes.extend_from_slice(&table_set_bytes(&full, &busier));
            sum_u += un.len();
            sum_bal += half;
            sum_max += c0.max(c1);
        }
        let t_bal = upload(g, &bal_bytes)?;
        let t_max = upload(g, &max_bytes)?;
        let mean_u = sum_u as f64 / NDRAWS as f64;
        println!(
            "G0 info M={rows} groups={groups:?} mean_union={mean_u:.2} mean_A_bal_experts={:.2} \
             mean_A_max_experts={:.2} mean_B_experts={mean_u:.2}",
            sum_bal as f64 / NDRAWS as f64,
            sum_max as f64 / NDRAWS as f64
        );

        // 2026-10-05: Arm 0 = A_bal, 1 = A_max, 2 = B; each call takes draw `d`.
        let call = |arm: usize, d: usize| -> Result<()> {
            let (t, mi) = match arm {
                0 => (Tables(t_bal.offset(d * SET_BYTES)), I_FULL),
                1 => (Tables(t_max.offset(d * SET_BYTES)), I_FULL),
                _ => (t_b, I_HALF),
            };
            run_moe(g, &k, &ws, x, ids_of(d), rows, &groups, t, mi, eo)
        };
        for arm in 0..3 {
            for d in 0..NDRAWS {
                call(arm, d)?;
            }
        }
        g.synchronize(STREAM)?;
        let mut tot = [0.0f64; 3];
        for round in 0..ROUNDS {
            for i in 0..3 {
                let arm = (i + round) % 3;
                ev.rec(0)?;
                for d in 0..NDRAWS {
                    call(arm, d)?;
                }
                ev.rec(1)?;
                tot[arm] += ev.us()?;
            }
        }
        g.synchronize(STREAM)?;
        let calls = (ROUNDS * NDRAWS) as f64;
        let (a_bal, a_max, b) = (tot[0] / calls, tot[1] / calls, tot[2] / calls);
        let (r_bal, r_max) = (b / a_bal, b / a_max);
        println!(
            "G0 M={rows} union={mean_u:.2} A_bal_us={a_bal:.2} A_max_us={a_max:.2} B_us={b:.2} \
             ratio_B_over_Abal={r_bal:.4} ratio_B_over_Amax={r_max:.4}"
        );
        if rows == 3 {
            ratio_m3 = r_bal;
        }
        g.free(t_bal)?;
        g.free(t_max)?;
        g.free(ids_all)?;
    }

    if ratio_m3.is_finite() && ratio_m3 <= BAR {
        println!("PASS: G0 expert-TP ratio_B_over_Abal={ratio_m3:.4} <= {BAR:.2} at M=3");
    } else {
        println!("G0 RESULT: ratio_B_over_Abal={ratio_m3:.4} > {BAR:.2} at M=3 (bar not met)");
    }
    Ok(())
}
