// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Gate G1 for GLM-5.3 routed-expert tensor parallelism (expert-TP,
//! `METRALE_GLM_EXPERT_TP`): through the public `build_moe` / `forward_moe`, does the sum of the
//! two expert-TP rank partials equal the sum of the two EP2 rank partials?
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants: none beyond the types. Correctness is gated; timing is information only.
//!
//! Design input: `spark-bench/runs/race/nccllat/L12-20261005T1758/EXPERT-TP-DESIGN.md`
//! (standard Megatron / vLLM FusedMoE expert TP: gate/up split by rows of I, down by columns
//! of I; the existing post-MoE all-reduce sums the two partials).
//!
//! Layouts, at GLM-5.3 shapes (hidden 4096, 288 experts, top-k 8, I = 2048), over the SAME
//! seeded weights, router and input rows:
//!   EP2  rank r binds experts `[144 r, 144 (r + 1))` at I = 2048 (`local_experts` 144);
//!   ETP  rank r binds all 288 experts at I/2 = 1024, each the `expert_tp::slice_nvfp4` of the
//!        full expert the loader's expert-TP bind uploads (gate/up rows, down columns, scale_2
//!        whole).
//! Both are bound by `build_moe` from `Glm5NextMlpConfig` literals that match
//! `Glm5NextMlpConfig::from_config_with(glm53, false | true)` at TP2/EP2. The shared expert is
//! all zeros, so each `forward_moe` output is exactly the rank's routed partial (the shared
//! expert's TP split is the same in both layouts and is not what G1 checks).
//!
//! Paths (`forward_moe` picks them by row count; one child process per lever setting, since the
//! levers are read once per process):
//!   rows 1         per-row device dispatch (`w4a16_gemv_sw_moe`);
//!   rows 3, 12     row-batched union GEMV (`moe_row_groups`);
//!   rows 256       grouped prefill: the tile GEMM (default), the grouped W4A16 MMA
//!                  (`METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1`), and CUTLASS W4A4
//!                  (`METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1`, when this build has CUTLASS; its
//!                  SFB cache is sized for one geometry per process, so EP2 and ETP run in
//!                  separate children).
//!
//! Bars: every W4A16 path (rows 1, 3, 12, 256 tile GEMM, 256 MMA): cosine(ETP sum, EP2 sum)
//! >= 0.9999. CUTLASS W4A4 quantises the activations per rank with a dynamic amax over different
//! row and width sets (ETP down: the I/2 slice), so the two layouts are not expected to agree to
//! 0.9999 with each other; its bar (PROVISIONAL, 2026-10-05) is that ETP's W4A4 error against the
//! W4A16 EP2 sum is at most 1.25x EP2's W4A4 error plus 1e-5. The direct ETP-vs-EP2 W4A4 cosine is
//! printed as information. A W4A4 sum bit-identical to the W4A16 one means the CUTLASS path
//! refused, which misses the bar.
//!
//! Timing (information only): median of `REPS` warm calls of `forward_moe` on the same rows,
//! rank 0 of ETP against the busier EP2 rank.
//!
//! Output: `G1 rows=... path=... cos=...` per check, `G1 TIME ...` lines, then `PASS: ...` when
//! every check meets its bar, else `G1 RESULT: ... (bar not met)` and exit 1.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example glm5next_expert_tp_g1_microtest

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::expert_tp::{Cut, Nvfp4Host, slice_nvfp4};
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::forward_prefill_gemm::{
    cutlass_w4a4, grouped_prefill_selected,
};
use metrale_model_arch::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use metrale_model_arch::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

// 2026-10-05: CUDA driver event API, declared as in `glm5next_expert_tp_g0_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const STREAM: u64 = 0;
/// 2026-10-05: GLM-5.3 shapes, `model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json`.
const H: usize = 4096;
const NUM_EXPERTS: usize = 288;
const TOP_K: usize = 8;
const I_FULL: usize = 2048;
const I_HALF: usize = I_FULL / 2;
const FULL_SHARED: usize = 2048;
const FULL_DENSE: usize = 12288;
const MAX_ROWS: usize = 256;
const COS_MIN: f64 = 0.9999;
/// 2026-10-05: The W4A4 bar: ETP error <= `W4A4_SLACK` x EP2 error + `W4A4_FLOOR`.
const W4A4_SLACK: f64 = 1.25;
const W4A4_FLOOR: f64 = 1e-5;
const WARM: usize = 3;
const REPS: usize = 15;
const SEED_X: u64 = 0x6731_0000_0000_00A1;
const SEED_ROUTER: u64 = 0x6731_0000_0000_00B2;

const LEVER_MMA: &str = "METRALE_GLM_MOE_PREFILL_GROUPED_W4A16";
const LEVER_W4A4: &str = "METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Layout {
    Ep,
    Etp,
}

impl Layout {
    fn name(self) -> &'static str {
        match self {
            Self::Ep => "ep2",
            Self::Etp => "etp",
        }
    }
}

/// 2026-10-05: One child process: a lever setting, the row counts it checks and the layouts it
/// binds.
struct Arm {
    name: &'static str,
    mma: bool,
    w4a4: bool,
    rows: &'static [usize],
    layouts: &'static [Layout],
}

static ARMS: [Arm; 4] = [
    Arm {
        name: "w4a16",
        mma: false,
        w4a4: false,
        rows: &[1, 3, 12, 256],
        layouts: &[Layout::Ep, Layout::Etp],
    },
    Arm {
        name: "mma",
        mma: true,
        w4a4: false,
        rows: &[256],
        layouts: &[Layout::Ep, Layout::Etp],
    },
    Arm {
        name: "w4a4-ep2",
        mma: false,
        w4a4: true,
        rows: &[256],
        layouts: &[Layout::Ep],
    },
    Arm {
        name: "w4a4-etp",
        mma: false,
        w4a4: true,
        rows: &[256],
        layouts: &[Layout::Etp],
    },
];

fn find_arm(name: &str) -> Result<&'static Arm> {
    ARMS.iter()
        .find(|a| a.name == name)
        .with_context(|| format!("unknown G1 arm {name}"))
}

/// 2026-10-05: Rank `rank`'s MLP geometry of GLM-5.3 at TP2/EP2 in `layout`: the fields
/// `Glm5NextMlpConfig::from_config_with` produces (expert-TP: I/2 and every expert local).
fn cfg(layout: Layout, rank: usize) -> Glm5NextMlpConfig {
    let etp = layout == Layout::Etp;
    Glm5NextMlpConfig {
        hidden: H,
        local_dense_intermediate: FULL_DENSE / 2,
        moe_intermediate: if etp { I_HALF } else { I_FULL },
        local_shared_intermediate: FULL_SHARED / 2,
        num_experts: NUM_EXPERTS,
        local_experts: if etp { NUM_EXPERTS } else { NUM_EXPERTS / 2 },
        ep_rank: rank,
        top_k: TOP_K,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 2,
        ep_world_size: 2,
    }
}

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

/// 2026-10-05: Weight-byte generator (splitmix64).
fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fill_packed(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut b = vec![0u8; n];
    for c in b.chunks_mut(8) {
        let v = splitmix(&mut s).to_le_bytes();
        let k = c.len();
        c.copy_from_slice(&v[..k]);
    }
    b
}

/// 2026-10-05: E4M3 codes 0x38..=0x3F (1.0..=1.875): no zero, inf or NaN scale.
fn fill_scale(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut b = vec![0u8; n];
    for c in b.chunks_mut(8) {
        let v = splitmix(&mut s);
        for (i, x) in c.iter_mut().enumerate() {
            *x = 0x38 | (((v >> (8 * i)) as u8) & 0x07);
        }
    }
    b
}

/// 2026-10-05: Per-tensor FP32 scale_2 of expert `e`. With E2M1 up to 6, scales up to 1.875 and
/// inputs in [-1, 1], 0.01 keeps the gate projections near unit size, well inside the SwiGLU
/// clamp of 10 (the G0 values).
fn scale2_of(e: usize) -> f32 {
    0.01 * (1.0 + (e % 7) as f32 * 0.05)
}

/// 2026-10-05: Uniform values in [-amp, amp].
fn rand_f32(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| ((splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amp)
        .collect()
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(16))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn upload_proj(g: &dyn GpuBackend, h: &Nvfp4Host) -> Result<Nvfp4Proj> {
    Ok(Nvfp4Proj {
        packed: upload(g, &h.packed)?,
        scale: upload(g, &h.scale)?,
        scale_2: h.scale_2,
        input_scale: h.input_scale,
    })
}

/// 2026-10-05: The device experts: `full[e]` at I = 2048 (EP2), `half[r][e]` rank r's
/// expert-TP slice at I = 1024. Generated one expert at a time; only that expert's host bytes are
/// held.
struct Experts {
    full: Vec<Glm5NextExpertWeights>,
    half: [Vec<Glm5NextExpertWeights>; 2],
}

fn gen_experts(g: &dyn GpuBackend, want_full: bool, want_half: bool) -> Result<Experts> {
    let mut out = Experts {
        full: Vec::new(),
        half: [Vec::new(), Vec::new()],
    };
    for e in 0..NUM_EXPERTS {
        let mut full = Vec::with_capacity(3);
        let mut half: [Vec<Nvfp4Proj>; 2] = [Vec::with_capacity(3), Vec::with_capacity(3)];
        for j in 0..3usize {
            // 2026-10-05: gate/up are `[I, H]`, down is `[H, I]`.
            let (rows, cols, cut) = if j < 2 {
                (I_FULL, H, Cut::Rows)
            } else {
                (H, I_FULL, Cut::Cols)
            };
            let tag = 0xE7_0000_0000u64 ^ ((e as u64) << 8) ^ ((j as u64) << 4);
            let host = Nvfp4Host {
                packed: fill_packed(rows * cols / 2, tag ^ 1),
                scale: fill_scale(rows * cols / 16, tag ^ 2),
                scale_2: scale2_of(e),
                input_scale: 0.0,
            };
            if want_full {
                full.push(upload_proj(g, &host)?);
            }
            if want_half {
                for (r, h) in half.iter_mut().enumerate() {
                    h.push(upload_proj(g, &slice_nvfp4(&host, rows, cols, cut, r, 2)?)?);
                }
            }
        }
        let pack = |p: &[Nvfp4Proj]| Glm5NextExpertWeights {
            gate_proj: p[0],
            up_proj: p[1],
            down_proj: p[2],
        };
        if want_full {
            out.full.push(pack(&full));
        }
        if want_half {
            for r in 0..2 {
                out.half[r].push(pack(&half[r]));
            }
        }
    }
    g.synchronize(STREAM)?;
    Ok(out)
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// 2026-10-05: The path `forward_moe` takes for `rows` under `arm`.
fn path_name(arm: &Arm, grouped: bool, rows: usize) -> &'static str {
    if grouped {
        if arm.w4a4 {
            "grouped-cutlass-w4a4"
        } else if arm.mma {
            "grouped-w4a16-mma"
        } else {
            "grouped-tile-gemm"
        }
    } else if rows == 1 {
        "per-row-dispatch"
    } else {
        "row-batched-union"
    }
}

/// 2026-10-05: Child: bind `arm.layouts` on this GPU, run `arm.rows`, write the rank-partial
/// sums (f32 LE, `[rows, H]` per (row count, layout) in that order) to `<base>.bin` and the
/// paths and timings to `<base>.txt`.
fn child(name: &str, base: &str) -> Result<()> {
    let arm = find_arm(name)?;
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let backend = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &backend;
    let k = Glm5NextMlpKernels::resolve(g)?;
    let mut txt = String::new();

    if arm.w4a4 {
        let geo = cfg(arm.layouts[0], 0);
        match cutlass_w4a4::prepare_at_load(g, &geo)? {
            Some(b) => println!(
                "G1 child {name}: CUTLASS W4A4 prepared ({:.1} MB, mi {}, {} local experts)",
                b as f64 / 1e6,
                geo.moe_intermediate,
                geo.local_experts
            ),
            None => {
                println!(
                    "G1 child {name}: CUTLASS W4A4 unavailable (no CUTLASS objects in this build, \
                     or the shape was refused); this arm is skipped"
                );
                std::fs::write(format!("{base}.txt"), "skip\n")?;
                return Ok(());
            }
        }
    }

    let has = |l: Layout| arm.layouts.contains(&l);
    let ex = gen_experts(g, has(Layout::Ep), has(Layout::Etp))?;

    let router = rand_f32(SEED_ROUTER, NUM_EXPERTS * H, 0.05);
    let bias = rand_f32(SEED_ROUTER ^ 0xB1A5, NUM_EXPERTS, 0.01);
    // 2026-10-05: A zero shared expert: each output is the rank's routed partial alone.
    let load = |n: &str| -> Result<Vec<f32>> {
        match n {
            "mlp.gate.weight" => Ok(router.clone()),
            "mlp.gate.e_score_correction_bias" => Ok(bias.clone()),
            "mlp.shared_experts.gate_proj.weight"
            | "mlp.shared_experts.up_proj.weight"
            | "mlp.shared_experts.down_proj.weight" => Ok(vec![0.0; FULL_SHARED * H]),
            other => bail!("G1: no synthetic tensor {other}"),
        }
    };

    let x_host: Vec<u8> = rand_f32(SEED_X, MAX_ROWS * H, 1.0)
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect();
    let x = upload(g, &x_host)?;
    let out_bytes = MAX_ROWS * H * 2;

    // 2026-10-05: Per layout: the two ranks' weights, one workspace (same geometry for both
    // ranks; calls are sequential), and one output buffer per rank.
    struct Bound {
        layout: Layout,
        cfgs: [Glm5NextMlpConfig; 2],
        moe: Vec<metrale_model_arch::glm5next_mlp::Glm5NextMoeWeights>,
        ws: Glm5NextMlpWorkspace,
        out: [DevicePtr; 2],
    }
    let mut bound = Vec::new();
    for &layout in arm.layouts {
        let cfgs = [cfg(layout, 0), cfg(layout, 1)];
        let mut moe = Vec::with_capacity(2);
        for (r, c) in cfgs.iter().enumerate() {
            let expert = |id: usize| -> Result<Glm5NextExpertWeights> {
                match layout {
                    Layout::Ep => Ok(ex.full[id]),
                    Layout::Etp => Ok(ex.half[r][id]),
                }
            };
            moe.push(build_moe(g, c, r, FULL_SHARED, &load, &expert)?);
        }
        bound.push(Bound {
            layout,
            cfgs,
            moe,
            ws: Glm5NextMlpWorkspace::new(g, &cfgs[0], MAX_ROWS)?,
            out: [g.alloc(out_bytes)?, g.alloc(out_bytes)?],
        });
    }
    g.synchronize(STREAM)?;

    let ev = Events::new()?;
    let mut bin: Vec<u8> = Vec::new();
    for &rows in arm.rows {
        for b in &bound {
            let grouped = grouped_prefill_selected(&k, &b.cfgs[0], &b.ws, rows);
            let path = path_name(arm, grouped, rows);
            txt.push_str(&format!("path {} {rows} {path}\n", b.layout.name()));
            let mut sum = vec![0f64; rows * H];
            for r in 0..2 {
                forward_moe(
                    g, &k, &b.cfgs[r], &b.moe[r], x, b.out[r], rows, &b.ws, STREAM,
                )?;
                g.synchronize(STREAM)?;
                let mut o = vec![0u8; rows * H * 2];
                g.copy_d2h(b.out[r], &mut o)?;
                for (s, v) in sum.iter_mut().zip(bf16_to_f32(&o)) {
                    *s += v as f64;
                }
                // 2026-10-05: Timing, information only: warm repeats of the same call.
                let mut t = Vec::with_capacity(REPS);
                for i in 0..WARM + REPS {
                    ev.rec(0)?;
                    forward_moe(
                        g, &k, &b.cfgs[r], &b.moe[r], x, b.out[r], rows, &b.ws, STREAM,
                    )?;
                    ev.rec(1)?;
                    let us = ev.us()?;
                    if i >= WARM {
                        t.push(us);
                    }
                }
                txt.push_str(&format!(
                    "time {} {rows} {r} {:.1}\n",
                    b.layout.name(),
                    median(t)
                ));
            }
            for v in &sum {
                bin.extend_from_slice(&(*v as f32).to_le_bytes());
            }
        }
    }
    std::fs::write(format!("{base}.bin"), &bin)?;
    std::fs::write(format!("{base}.txt"), txt)?;
    Ok(())
}

/// 2026-10-05: What one child left: `None` when its arm was skipped.
struct ArmOut {
    sums: std::collections::HashMap<(Layout, usize), Vec<f32>>,
    paths: std::collections::HashMap<(Layout, usize), String>,
    times: std::collections::HashMap<(Layout, usize, usize), f64>,
}

fn parse_layout(s: &str) -> Result<Layout> {
    match s {
        "ep2" => Ok(Layout::Ep),
        "etp" => Ok(Layout::Etp),
        other => bail!("G1: unknown layout {other} in a child record"),
    }
}

fn run_arm(a: &Arm, dir: &std::path::Path) -> Result<Option<ArmOut>> {
    let base = dir.join(format!("g1-{}", a.name));
    let base = base.to_str().context("G1: scratch path is not UTF-8")?.to_string();
    let exe = std::env::current_exe()?;
    let on = |b: bool| if b { "1" } else { "0" };
    let st = std::process::Command::new(exe)
        .arg("--child")
        .arg(a.name)
        .arg(&base)
        .env(LEVER_MMA, on(a.mma))
        .env(LEVER_W4A4, on(a.w4a4))
        .status()?;
    if !st.success() {
        bail!("G1 child {} exited with {st}", a.name);
    }
    let txt = std::fs::read_to_string(format!("{base}.txt"))?;
    if txt.trim() == "skip" {
        return Ok(None);
    }
    let bin = std::fs::read(format!("{base}.bin"))?;
    let vals: Vec<f32> = bin
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut out = ArmOut {
        sums: Default::default(),
        paths: Default::default(),
        times: Default::default(),
    };
    let mut at = 0usize;
    for &rows in a.rows {
        for &l in a.layouts {
            let n = rows * H;
            let v = vals
                .get(at..at + n)
                .with_context(|| format!("G1 child {}: short sums file", a.name))?;
            out.sums.insert((l, rows), v.to_vec());
            at += n;
        }
    }
    for line in txt.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["path", l, rows, p] => {
                out.paths
                    .insert((parse_layout(l)?, rows.parse()?), (*p).to_string());
            }
            ["time", l, rows, r, us] => {
                out.times
                    .insert((parse_layout(l)?, rows.parse()?, r.parse()?), us.parse()?);
            }
            _ => bail!("G1 child {}: unreadable record line {line:?}", a.name),
        }
    }
    let _ = std::fs::remove_file(format!("{base}.bin"));
    let _ = std::fs::remove_file(format!("{base}.txt"));
    Ok(Some(out))
}

/// 2026-10-05: Cosine of two equal-length vectors, and whether both are nonzero and finite.
fn cosine(a: &[f32], b: &[f32]) -> (f64, bool) {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    let ok = aa > 0.0 && bb > 0.0 && aa.is_finite() && bb.is_finite() && a.len() == b.len();
    (ab / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE), ok)
}

fn time_line(label: &str, rows: usize, etp: &ArmOut, ep: &ArmOut) {
    let t = |o: &ArmOut, l: Layout, r: usize| o.times.get(&(l, rows, r)).copied();
    if let (Some(e0), Some(p0), Some(p1)) = (
        t(etp, Layout::Etp, 0),
        t(ep, Layout::Ep, 0),
        t(ep, Layout::Ep, 1),
    ) {
        let busy = p0.max(p1);
        println!(
            "G1 TIME path={label} rows={rows} etp_rank0_us={e0:.1} ep2_rank0_us={p0:.1} \
             ep2_rank1_us={p1:.1} etp/busier_ep2={:.3} (info, warm median of {REPS})",
            e0 / busy
        );
    }
}

fn parent() -> Result<bool> {
    let dir = std::env::temp_dir().join(format!("metrale-g1-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    println!(
        "G1 setup: H={H} experts={NUM_EXPERTS} top_k={TOP_K} I={I_FULL} (expert-TP I/2={I_HALF}) \
         rows=1,3,12,256 bar cos>={COS_MIN} (W4A4: etp_err<={W4A4_SLACK}*ep2_err+{W4A4_FLOOR}); \
         shared expert zero; one child per lever setting"
    );
    let mut outs = Vec::new();
    for a in &ARMS {
        outs.push(run_arm(a, &dir)?);
    }
    let _ = std::fs::remove_dir(&dir);
    let [w16, mma, w4ep, w4etp] = <[Option<ArmOut>; 4]>::try_from(outs)
        .map_err(|_| anyhow::anyhow!("G1: expected four arm results"))?;
    let w16 = w16.context("G1: the W4A16 arm never skips")?;
    let mma = mma.context("G1: the MMA arm never skips")?;

    let mut misses: Vec<String> = Vec::new();
    let mut checks = 0usize;
    let mut eval = |label: &str, rows: usize, o: &ArmOut, misses: &mut Vec<String>| {
        checks += 1;
        let (ep, etp) = (&o.sums[&(Layout::Ep, rows)], &o.sums[&(Layout::Etp, rows)]);
        let (c, ok) = cosine(etp, ep);
        let path = o.paths.get(&(Layout::Etp, rows)).cloned().unwrap_or_default();
        let ep_path = o.paths.get(&(Layout::Ep, rows)).cloned().unwrap_or_default();
        let met = ok && c >= COS_MIN && path == ep_path;
        println!(
            "G1 rows={rows} arm={label} path={path} cos(etp_sum,ep2_sum)={c:.7} bar={COS_MIN} {}",
            if met { "ok" } else { "MISS" }
        );
        if !met {
            misses.push(format!(
                "rows={rows} arm={label} path={path}/{ep_path} cos={c:.7}{}",
                if ok { "" } else { " (degenerate output)" }
            ));
        }
    };
    for rows in w16.rows_list() {
        eval("w4a16", rows, &w16, &mut misses);
    }
    eval("mma", 256, &mma, &mut misses);
    let mma_same = mma.sums[&(Layout::Ep, 256)] == w16.sums[&(Layout::Ep, 256)];
    println!(
        "G1 info: grouped W4A16 MMA EP2 sum is {} the tile GEMM's",
        if mma_same {
            "bit-identical to (check that the MMA path ran)"
        } else {
            "different from (the MMA path ran)"
        }
    );

    match (&w4ep, &w4etp) {
        (Some(ep), Some(etp)) => {
            checks += 1;
            let r = &w16.sums[&(Layout::Ep, 256)];
            let (s_ep, s_etp) = (&ep.sums[&(Layout::Ep, 256)], &etp.sums[&(Layout::Etp, 256)]);
            let (c_ep, ok1) = cosine(s_ep, r);
            let (c_etp, ok2) = cosine(s_etp, r);
            let (c_dir, _) = cosine(s_etp, s_ep);
            let (e_ep, e_etp) = (1.0 - c_ep, 1.0 - c_etp);
            let ran = s_ep != r && s_etp != &w16.sums[&(Layout::Etp, 256)];
            let met = ok1 && ok2 && ran && e_etp <= W4A4_SLACK * e_ep + W4A4_FLOOR;
            println!(
                "G1 rows=256 arm=w4a4 path={} cos(ep2_w4a4,w4a16_ref)={c_ep:.7} \
                 cos(etp_w4a4,w4a16_ref)={c_etp:.7} err_etp={e_etp:.3e} \
                 bar<={:.3e} {} (info: cos(etp_w4a4,ep2_w4a4)={c_dir:.7})",
                etp.paths
                    .get(&(Layout::Etp, 256))
                    .cloned()
                    .unwrap_or_default(),
                W4A4_SLACK * e_ep + W4A4_FLOOR,
                if met { "ok" } else { "MISS" }
            );
            if !met {
                misses.push(format!(
                    "rows=256 arm=w4a4 err_etp={e_etp:.3e} err_ep2={e_ep:.3e}{}",
                    if ran {
                        ""
                    } else {
                        " (W4A4 sum equals W4A16: the CUTLASS path refused)"
                    }
                ));
            }
            time_line("grouped-cutlass-w4a4", 256, etp, ep);
        }
        _ => println!("G1 rows=256 arm=w4a4: skipped (no CUTLASS W4A4 in this build)"),
    }
    for rows in w16.rows_list() {
        let p = w16
            .paths
            .get(&(Layout::Etp, rows))
            .cloned()
            .unwrap_or_default();
        time_line(&p, rows, &w16, &w16);
    }
    time_line("grouped-w4a16-mma", 256, &mma, &mma);

    if misses.is_empty() {
        println!(
            "PASS: G1 expert-TP rank-partial sums match EP2 on {checks} path checks \
             (W4A16 rows 1/3/12/256 + MMA cos>={COS_MIN}{})",
            if w4ep.is_some() && w4etp.is_some() {
                "; CUTLASS W4A4 within 1.25x EP2 error"
            } else {
                "; CUTLASS W4A4 skipped"
            }
        );
        Ok(true)
    } else {
        println!(
            "G1 RESULT: {} of {checks} checks below the bar: {} (bar not met)",
            misses.len(),
            misses.join("; ")
        );
        Ok(false)
    }
}

impl ArmOut {
    /// 2026-10-05: The row counts present, ascending.
    fn rows_list(&self) -> Vec<usize> {
        let mut r: Vec<usize> = self.sums.keys().map(|&(_, rows)| rows).collect();
        r.sort_unstable();
        r.dedup();
        r
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--child" {
        return child(&args[2], &args[3]);
    }
    if !parent()? {
        std::process::exit(1);
    }
    Ok(())
}
