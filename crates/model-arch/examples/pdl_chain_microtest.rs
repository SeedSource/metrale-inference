// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate 1 for programmatic dependent launch (PDL) on decode GEMV chains
//! (SeedHQ/seed-skills#80).
//!
//! Owner: model-arch examples.
//! Three chains, each timed and checked in three arms (A base, B `_pdl`, C `_pdl_touch`):
//! - D (dense NV4, `w4a16_gemv_batch3_staged`, M = 3): per layer KDA in-proj 4096x4096 -> shared
//!   gate 1024x4096 -> shared up 1024x4096 -> shared down 4096x1024 (reads the gate output), 42
//!   layers, distinct weights per layer: 168 launches, 167 boundaries.
//! - G (grouping): per layer one grouped launch N = 12288, K = 4096 (KDA q/k/v concatenated)
//!   then one grouped shared gate+up launch N = 2048, K = 4096, 42 layers: 84 launches, 83
//!   boundaries. Same kernel, so this is "grouping + PDL" measured.
//! - S (small dense, picked from the c27c MTP verify trace, see below): GLM-5.3 has 45 layers,
//!   attention is DSA when `l % 4 == 3` else KDA, the MoE router from layer 3 on.
//!   KDA: `dense_gemv_tcm_bf16` 128x4096 (M = 3) -> `dense_gemv_tcm_bf16` 4096x128 (M = 3, reads
//!   the previous output, writes the next layer's input). DSA, per row of 3: `dense_gemv_tcm_bf16`
//!   128x4096 (indexer wk) and again (compress_gate), both M = 1, then `dense_gemv_bf16_fp32out`
//!   4096x1536 (indexer wq_b, M = 1). Every layer l >= 3: `dense_gemv_bf16_fp32out_batchm` router
//!   288x4096 (M = 3). 209 launches, 208 boundaries. Only the picked kernels are in the chain, so
//!   a boundary here is picked-to-picked, adjacent in the graph, whereas in the real graph other
//!   kernels sit between them.
//!
//! Pick (nsys sqlite `ra-comb27c-n1n2-carmtp`, rank 0, 160 verify rounds, kernels before the
//! round's first argmax; mean launches per round x mean duration): router batchm (72 grid) 42 x
//! 21.6 us = 0.907 ms; fp32out grid 1024 33 x 25.0 us = 0.825 ms; tcm_bf16 grid 8 100 x 8.06 us =
//! 0.806 ms; tcm_bf16 grid 256 34 x 9.11 us = 0.310 ms. Together 2.848 ms = 79.7 % of the 3.575 ms
//! per round of the in-layer small BF16/FP8 dense GEMVs (also: tcm_fp8w grid 256 0.257, tcm_fp8w
//! grid 8 0.215, tcm_bf16 grid 2 0.174, fp32out grid 8 0.081 ms; the one tcm_fp8w grid 4840 launch
//! per round, 1.289 ms, is the LM head: not in-layer, one launch, left out). With the LM head in the
//! denominator the four are 58.6 %. Shapes: tcm grid = ceil(N / 16), other kernels ceil(N / 4).
//!
//! Each arm is captured by stream capture (relaxed mode, as the engine) into one CUDA graph:
//! - A: base kernels, `cuLaunchKernel`.
//! - B: `*_pdl` kernels, `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_
//!   SERIALIZATION = 1` on every launch after the first (kernel: `launch_dependents` at start,
//!   `griddepcontrol.wait` before any read).
//! - C: `*_pdl_touch` kernels, launched like B (kernel: the thread's / warp's first weight vector
//!   is loaded before the wait).
//!
//! Per chain prints `PDL_EDGES chain=X arm=Y total=N programmatic=P ...` (cuGraphGetEdges_v2;
//! `PDL_EDGES_DROPPED chain=X arm=Y` when B or C has no programmatic edge: a finding),
//! `PDL_CMP chain=X arm=Y bytes_differ=N` (all output buffers, raw bytes, vs A), `PDL_RACE`
//! (five more replays each compared with A), `PDL_T chain=X arm=Y ms=M` (median graph replay, 3
//! rounds x 20, order rotated per round, cold L2 by a read-only 72 MiB GEMV before every replay,
//! outside the timed region), `PDL_BOUNDARY chain=X arm=Y saving_us=S boundaries=N` ((A - Y) / N),
//! and `PDL_TRACE` with the real-trace launch counts to scale a per-boundary saving to ms per
//! round. Final line `PASS` iff every chain is bitwise clean in every replay, else `FAIL <why>`
//! (exit 1). Timing is only meaningful when run alone on a GPU.
//!
//! Run (GPU):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example pdl_chain_microtest

use std::ffi::c_void;
use std::ptr::null_mut;

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::weight_map::DenseWeight;

/// `CUlaunchAttribute` (72 bytes): id, 4 bytes padding, a 64-byte value union whose first field
/// is the int `programmaticStreamSerializationAllowed`. Per cuda.h of CUDA 13.x.
#[repr(C)]
#[derive(Clone, Copy)]
struct LaunchAttr {
    id: u32,
    pad: [u8; 4],
    value: [i32; 16],
}
const _: () = assert!(std::mem::size_of::<LaunchAttr>() == 72);
/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`.
const ATTR_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;

/// `CUlaunchConfig`.
#[repr(C)]
struct LaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    stream: u64,
    attrs: *const LaunchAttr,
    num_attrs: u32,
}

/// `CUgraphEdgeData`; `type_` is a `CUgraphDependencyType` (1 = programmatic).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EdgeData {
    from_port: u8,
    to_port: u8,
    type_: u8,
    reserved: [u8; 5],
}
const _: () = assert!(std::mem::size_of::<EdgeData>() == 8);
const DEPENDENCY_PROGRAMMATIC: u8 = 1;

unsafe extern "C" {
    fn cuLaunchKernel(
        f: u64,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        smem: u32,
        stream: u64,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuLaunchKernelEx(
        config: *const LaunchConfig,
        f: u64,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuStreamBeginCapture(stream: u64, mode: u32) -> i32;
    fn cuStreamEndCapture(stream: u64, graph: *mut u64) -> i32;
    fn cuGraphInstantiateWithFlags(exec: *mut u64, graph: u64, flags: u64) -> i32;
    fn cuGraphGetEdges_v2(
        graph: u64,
        from: *mut u64,
        to: *mut u64,
        edge_data: *mut EdgeData,
        num: *mut usize,
    ) -> i32;
    fn cuGraphLaunch(exec: u64, stream: u64) -> i32;
    fn cuGraphDestroy(graph: u64) -> i32;
    fn cuGraphExecDestroy(exec: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
}

fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

const M: usize = 3;
const ROUNDS: usize = 3;
const REPLAYS: usize = 20;
const WARMUP: usize = 3;
const RACE_REPLAYS: usize = 5;
const FLUSH_N: usize = 4096;
const FLUSH_K: usize = 32768;
/// Layers: D and G use 42 (the shared-expert chains of the earlier plan), S uses the model's 45.
const LAYERS_DG: usize = 42;
const LAYERS_S: usize = 45;

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn n(&mut self) -> f64 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
}

/// One NVFP4 weight on the device.
#[derive(Clone, Copy)]
struct W {
    w: u64,
    s: u64,
    s2: f32,
}

/// Kernel families; the index is the row of the kernel table.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// `w4a16_gemv_batch3_staged(a, w, s, s2, c, n, k)`, grid ceil(n/4).
    Nv4,
    /// `dense_gemv_tcm_bf16(a, w, null, c, m, n, k, out_stride)`, grid ceil(n/16).
    Tcm,
    /// `dense_gemv_bf16_fp32out(a, w, c, n, k)`, grid ceil(n/4), FP32 out.
    F32,
    /// `dense_gemv_bf16_fp32out_batchm(a, w, c, m, n, k, out_stride)`, grid ceil(n/4).
    F32B,
}
const KINDS: [Kind; 4] = [Kind::Nv4, Kind::Tcm, Kind::F32, Kind::F32B];

/// One launch of a chain.
struct Step {
    kind: Kind,
    a: u64,
    w: W,
    c: u64,
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    A,
    B,
    C,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::A => "A",
            Arm::B => "B",
            Arm::C => "C",
        }
    }
}

/// Kernel handles: `[kind][arm]`.
type Ks = [[KernelHandle; 3]; 4];

/// Launch `st` on `stream` with the arm's kernel; `pdl` adds the programmatic-serialization attribute.
fn launch(ks: &Ks, arm: Arm, st: &Step, stream: u64, pdl: bool) -> Result<()> {
    let f = ks[KINDS.iter().position(|&k| k == st.kind).unwrap()][arm as usize];
    let (mut a, mut w, mut s, mut s2, mut c, mut nul) = (st.a, st.w.w, st.w.s, st.w.s2, st.c, 0u64);
    let (mut m, mut n, mut k) = (st.m as u32, st.n as u32, st.k as u32);
    let mut out_stride = st.n as u32;
    let p = |x: &mut u64| x as *mut u64 as *mut c_void;
    let q = |x: &mut u32| x as *mut u32 as *mut c_void;
    let mut params: Vec<*mut c_void> = match st.kind {
        Kind::Nv4 => vec![
            p(&mut a),
            p(&mut w),
            p(&mut s),
            &mut s2 as *mut f32 as *mut c_void,
            p(&mut c),
            q(&mut n),
            q(&mut k),
        ],
        Kind::Tcm => vec![
            p(&mut a),
            p(&mut w),
            p(&mut nul),
            p(&mut c),
            q(&mut m),
            q(&mut n),
            q(&mut k),
            q(&mut out_stride),
        ],
        Kind::F32 => vec![p(&mut a), p(&mut w), p(&mut c), q(&mut n), q(&mut k)],
        Kind::F32B => vec![
            p(&mut a),
            p(&mut w),
            p(&mut c),
            q(&mut m),
            q(&mut n),
            q(&mut k),
            q(&mut out_stride),
        ],
    };
    let per_cta = if st.kind == Kind::Tcm { 16 } else { 4 };
    let grid = (st.n as u32).div_ceil(per_cta);
    if !pdl {
        // SAFETY: `params` points at live locals of the kernel's argument types; the driver
        // copies the values during the call (also under stream capture).
        return ck(
            unsafe {
                cuLaunchKernel(
                    f.0,
                    grid,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    null_mut(),
                )
            },
            "cuLaunchKernel",
        );
    }
    let mut attr = LaunchAttr {
        id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
        pad: [0; 4],
        value: [0; 16],
    };
    attr.value[0] = 1;
    let cfg = LaunchConfig {
        grid: [grid, 1, 1],
        block: [256, 1, 1],
        shared_mem: 0,
        stream,
        attrs: &attr,
        num_attrs: 1,
    };
    // SAFETY: `cfg`, `attr` and `params` are live locals laid out as cuda.h declares.
    ck(
        unsafe { cuLaunchKernelEx(&cfg, f.0, params.as_mut_ptr(), null_mut()) },
        "cuLaunchKernelEx",
    )
}

/// Capture the whole chain for `arm` into a graph; returns (graph, exec).
fn capture(ks: &Ks, arm: Arm, steps: &[Step], stream: u64) -> Result<(u64, u64)> {
    // SAFETY: mode 2 = CU_STREAM_CAPTURE_MODE_RELAXED, as the engine's begin_capture.
    ck(
        unsafe { cuStreamBeginCapture(stream, 2) },
        "cuStreamBeginCapture",
    )?;
    let mut res = Ok(());
    for (i, st) in steps.iter().enumerate() {
        res = launch(ks, arm, st, stream, arm != Arm::A && i > 0);
        if res.is_err() {
            break;
        }
    }
    let mut graph = 0u64;
    // SAFETY: ends the capture begun above.
    let end = ck(
        unsafe { cuStreamEndCapture(stream, &mut graph) },
        "cuStreamEndCapture",
    );
    res?;
    end?;
    let mut exec = 0u64;
    // SAFETY: instantiates the graph just captured.
    ck(
        unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) },
        "cuGraphInstantiateWithFlags",
    )?;
    Ok((graph, exec))
}

/// (total edges, programmatic edges, edges leaving a nonzero port).
fn edges(graph: u64) -> Result<(usize, usize, usize)> {
    let mut n = 0usize;
    // SAFETY: count query (null arrays) as the driver documents.
    ck(
        unsafe { cuGraphGetEdges_v2(graph, null_mut(), null_mut(), null_mut(), &mut n) },
        "cuGraphGetEdges_v2 (count)",
    )?;
    let mut from = vec![0u64; n];
    let mut to = vec![0u64; n];
    let mut ed = vec![EdgeData::default(); n];
    let mut m = n;
    // SAFETY: each array holds `n` entries.
    ck(
        unsafe {
            cuGraphGetEdges_v2(
                graph,
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                ed.as_mut_ptr(),
                &mut m,
            )
        },
        "cuGraphGetEdges_v2",
    )?;
    let prog = ed[..m]
        .iter()
        .filter(|e| e.type_ == DEPENDENCY_PROGRAMMATIC)
        .count();
    let port = ed[..m].iter().filter(|e| e.from_port != 0).count();
    Ok((m, prog, port))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

fn quantize(
    g: &dyn GpuBackend,
    absmax: KernelHandle,
    quant: KernelHandle,
    rng: &mut Lcg,
    n: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<u8>, f32)> {
    let host = bf16_bytes(rng, n * k, 0.02);
    let tmp = g.alloc(host.len())?;
    g.copy_h2d(&host, tmp)?;
    let q = metrale_model_layers::weight_map::quantize_to_nvfp4(
        &DenseWeight { weight: tmp },
        n,
        k,
        g,
        absmax,
        quant,
        g.default_stream(),
    )?;
    g.synchronize(g.default_stream())?;
    let mut w = vec![0u8; n * k / 2];
    let mut s = vec![0u8; n * k / 16];
    g.copy_d2h(q.weight, &mut w)?;
    g.copy_d2h(q.weight_scale, &mut s)?;
    g.free(tmp)?;
    g.free(q.weight)?;
    g.free(q.weight_scale)?;
    Ok((w, s, q.weight_scale_2))
}

/// `count` BF16 values ~ N(0, std^2), little-endian bytes.
fn bf16_bytes(rng: &mut Lcg, count: usize, std: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(count * 2);
    for _ in 0..count {
        v.extend_from_slice(&bf16::from_f64(rng.n() * std).to_bits().to_le_bytes());
    }
    v
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<u64> {
    let p = g.alloc(b.len())?;
    g.copy_h2d(b, p)?;
    Ok(p.0)
}

/// A bump allocator over one chain's output arena (base 0 first, to size it).
struct Bump {
    base: u64,
    off: usize,
}
impl Bump {
    fn take(&mut self, bytes: usize) -> u64 {
        let p = self.base + self.off as u64;
        self.off += bytes.div_ceil(256) * 256;
        p
    }
}

/// NVFP4 weights of one layer-slot per layer: the base bytes rotated per (layer, slot).
fn nv4_layers(
    g: &dyn GpuBackend,
    base: &(Vec<u8>, Vec<u8>, f32),
    layers: usize,
    slot: usize,
    slots: usize,
) -> Result<Vec<W>> {
    let mut out = Vec::new();
    for l in 0..layers {
        let mut w = base.0.clone();
        let mut s = base.1.clone();
        let r = (l * slots + slot) * 7919;
        let (rw, rs) = (r % w.len(), r % s.len());
        w.rotate_left(rw);
        s.rotate_left(rs);
        out.push(W {
            w: up(g, &w)?,
            s: up(g, &s)?,
            s2: base.2,
        });
    }
    Ok(out)
}

/// BF16 weights `[n, k]`, one per layer: base bytes rotated by an even offset per (layer, slot).
fn bf16_layers(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    n: usize,
    k: usize,
    layers: usize,
    slot: usize,
) -> Result<Vec<W>> {
    let base = bf16_bytes(rng, n * k, 0.02);
    let mut out = Vec::new();
    for l in 0..layers {
        let mut w = base.clone();
        let len = w.len();
        w.rotate_left(((l * 8 + slot) * 7918) % len & !1);
        out.push(W {
            w: up(g, &w)?,
            s: 0,
            s2: 0.0,
        });
    }
    Ok(out)
}

/// A chain: a name, its launches, and the arena all outputs live in.
struct Chain {
    name: &'static str,
    steps: Vec<Step>,
    arena: u64,
    arena_bytes: usize,
}

/// Build a chain twice: once on a zero-based bump to size the arena, then for real.
fn build(
    g: &dyn GpuBackend,
    name: &'static str,
    f: &dyn Fn(&mut Bump) -> Vec<Step>,
) -> Result<Chain> {
    let mut sizing = Bump { base: 0, off: 0 };
    f(&mut sizing);
    let arena = g.alloc(sizing.off)?.0;
    let mut real = Bump {
        base: arena,
        off: 0,
    };
    let steps = f(&mut real);
    Ok(Chain {
        name,
        steps,
        arena,
        arena_bytes: sizing.off,
    })
}

fn run() -> Result<(bool, String)> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mk = |module: &str, base: &str| -> Result<[KernelHandle; 3]> {
        Ok([
            g.kernel(module, base)?,
            g.kernel(module, &format!("{base}_pdl"))?,
            g.kernel(module, &format!("{base}_pdl_touch"))?,
        ])
    };
    let ks: Ks = [
        mk("w4a16_gemv", "w4a16_gemv_batch3_staged")?,
        mk("dense_gemv_tcm", "dense_gemv_tcm_bf16")?,
        mk("gemv", "dense_gemv_bf16_fp32out")?,
        mk("dense_gemv_bf16_batchm", "dense_gemv_bf16_fp32out_batchm")?,
    ];
    let gemv1 = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let absmax = g.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quant = g.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?;
    let stream = g.default_stream();
    let mut rng = Lcg(0x7064_6c63_6861_696e);

    let act = |g: &dyn GpuBackend, rng: &mut Lcg, count: usize| -> Result<u64> {
        up(g, &bf16_bytes(rng, count, 1.0))
    };
    let mut chains: Vec<Chain> = Vec::new();

    // ---- Chain D: in-proj, shared gate, up, down; 42 layers. ----
    {
        let shapes = [
            (4096usize, 4096usize),
            (1024, 4096),
            (1024, 4096),
            (4096, 1024),
        ];
        let base: Vec<_> = [shapes[0], shapes[1], shapes[3]]
            .iter()
            .map(|&(n, k)| quantize(g, absmax, quant, &mut rng, n, k))
            .collect::<Result<_>>()?;
        let ws: Vec<Vec<W>> = (0..4)
            .map(|slot| nv4_layers(g, &base[[0, 1, 1, 2][slot]], LAYERS_DG, slot, 4))
            .collect::<Result<_>>()?;
        let a0 = act(g, &mut rng, M * 4096)?;
        chains.push(build(g, "D", &|bump| {
            let mut steps = Vec::new();
            let mut x = a0;
            for l in 0..LAYERS_DG {
                let o: Vec<u64> = shapes.iter().map(|&(n, _)| bump.take(M * n * 2)).collect();
                let ins = [x, o[0], o[0], o[1]];
                for i in 0..4 {
                    steps.push(Step {
                        kind: Kind::Nv4,
                        a: ins[i],
                        w: ws[i][l],
                        c: o[i],
                        m: M,
                        n: shapes[i].0,
                        k: shapes[i].1,
                    });
                }
                x = o[3];
            }
            steps
        })?);
    }

    // ---- Chain G: grouped q/k/v (12288 x 4096) then grouped gate+up (2048 x 4096). ----
    {
        let bq = quantize(g, absmax, quant, &mut rng, 12288, 4096)?;
        let bg = quantize(g, absmax, quant, &mut rng, 2048, 4096)?;
        let wq = nv4_layers(g, &bq, LAYERS_DG, 0, 2)?;
        let wg = nv4_layers(g, &bg, LAYERS_DG, 1, 2)?;
        let a0 = act(g, &mut rng, M * 4096)?;
        chains.push(build(g, "G", &|bump| {
            let mut steps = Vec::new();
            let mut x = a0;
            for l in 0..LAYERS_DG {
                let o_qkv = bump.take(M * 12288 * 2);
                // The gate+up output is padded to 3 x 4096 so the next layer reads K = 4096.
                let o_gu = bump.take(M * 4096 * 2);
                steps.push(Step {
                    kind: Kind::Nv4,
                    a: x,
                    w: wq[l],
                    c: o_qkv,
                    m: M,
                    n: 12288,
                    k: 4096,
                });
                steps.push(Step {
                    kind: Kind::Nv4,
                    a: o_qkv,
                    w: wg[l],
                    c: o_gu,
                    m: M,
                    n: 2048,
                    k: 4096,
                });
                x = o_gu;
            }
            steps
        })?);
    }

    // ---- Chain S: the picked small dense GEMVs of the verify round, 45 layers. ----
    {
        let l = LAYERS_S;
        let w_kda_a = bf16_layers(g, &mut rng, 128, 4096, l, 0)?;
        let w_kda_b = bf16_layers(g, &mut rng, 4096, 128, l, 1)?;
        let w_wk = bf16_layers(g, &mut rng, 128, 4096, l, 2)?;
        let w_cg = bf16_layers(g, &mut rng, 128, 4096, l, 3)?;
        let w_wqb = bf16_layers(g, &mut rng, 4096, 1536, l, 4)?;
        let w_rt = bf16_layers(g, &mut rng, 288, 4096, l, 5)?;
        let a0 = act(g, &mut rng, M * 4096)?;
        chains.push(build(g, "S", &|bump| {
            let mut steps = Vec::new();
            let mut x = a0;
            let st = |kind, a, w, c, m, n, k| Step {
                kind,
                a,
                w,
                c,
                m,
                n,
                k,
            };
            for i in 0..l {
                if i % 4 == 3 {
                    // DSA indexer, per row r of 3 (M = 1 launches, as in the trace).
                    for r in 0..M {
                        let xr = x + (r * 4096 * 2) as u64;
                        let o1 = bump.take(128 * 2);
                        let o2 = bump.take(128 * 2);
                        let o3 = bump.take(4096 * 4);
                        steps.push(st(Kind::Tcm, xr, w_wk[i], o1, 1, 128, 4096));
                        steps.push(st(Kind::Tcm, xr, w_cg[i], o2, 1, 128, 4096));
                        steps.push(st(Kind::F32, xr, w_wqb[i], o3, 1, 4096, 1536));
                    }
                } else {
                    // KDA: g_a (128 x 4096) then g_b (4096 x 128); the output is the next input.
                    let o1 = bump.take(M * 128 * 2);
                    let o2 = bump.take(M * 4096 * 2);
                    steps.push(st(Kind::Tcm, x, w_kda_a[i], o1, M, 128, 4096));
                    steps.push(st(Kind::Tcm, o1, w_kda_b[i], o2, M, 4096, 128));
                    x = o2;
                }
                if i >= 3 {
                    let o = bump.take(M * 288 * 4);
                    steps.push(st(Kind::F32B, x, w_rt[i], o, M, 288, 4096));
                }
            }
            steps
        })?);
    }

    // Cold-L2 flush: a read-only M = 1 GEMV over a 72 MiB all-zero weight.
    let fw = W {
        w: g.alloc(FLUSH_N * FLUSH_K / 2)?.0,
        s: g.alloc(FLUSH_N * FLUSH_K / 16)?.0,
        s2: 1.0,
    };
    g.memset(DevicePtr(fw.w), 0, FLUSH_N * FLUSH_K / 2)?;
    g.memset(DevicePtr(fw.s), 0, FLUSH_N * FLUSH_K / 16)?;
    let fa = g.alloc(FLUSH_K * 2)?.0;
    let fc = g.alloc(FLUSH_N * 2)?.0;
    g.memset(DevicePtr(fa), 0, FLUSH_K * 2)?;
    let flush_ks: Ks = [[gemv1; 3]; 4];
    let flush_step = Step {
        kind: Kind::Nv4,
        a: fa,
        w: fw,
        c: fc,
        m: 1,
        n: FLUSH_N,
        k: FLUSH_K,
    };

    let arms = [Arm::A, Arm::B, Arm::C];
    let mut ev = [0u64; 2];
    for e in ev.iter_mut() {
        // SAFETY: creates one event handle into `e`.
        ck(unsafe { cuEventCreate(e, 0) }, "cuEventCreate")?;
    }
    let mut why: Vec<String> = Vec::new();

    for ch in &chains {
        let cn = ch.name;
        let boundaries = ch.steps.len() - 1;
        println!(
            "PDL_CHAIN chain={cn} launches={} boundaries={boundaries}",
            ch.steps.len()
        );
        let mut execs = Vec::new();
        for &arm in &arms {
            let (graph, exec) = capture(&ks, arm, &ch.steps, stream)?;
            let (total, prog, port) = edges(graph)?;
            println!(
                "PDL_EDGES chain={cn} arm={} total={total} programmatic={prog} from_port_nonzero={port}",
                arm.name()
            );
            if arm != Arm::A && prog == 0 {
                println!("PDL_EDGES_DROPPED chain={cn} arm={}", arm.name());
            }
            // SAFETY: the template graph is no longer needed once instantiated.
            unsafe { cuGraphDestroy(graph) };
            execs.push(exec);
        }
        let run_once = |exec: u64| -> Result<Vec<u8>> {
            g.memset(DevicePtr(ch.arena), 0xA5, ch.arena_bytes)?;
            // SAFETY: replays an instantiated graph on the capture stream, then waits.
            ck(unsafe { cuGraphLaunch(exec, stream) }, "cuGraphLaunch")?;
            ck(
                unsafe { cuStreamSynchronize(stream) },
                "cuStreamSynchronize",
            )?;
            let mut out = vec![0u8; ch.arena_bytes];
            g.copy_d2h(DevicePtr(ch.arena), &mut out)?;
            Ok(out)
        };
        let reference = run_once(execs[0])?;
        let diff = |o: &[u8]| o.iter().zip(&reference).filter(|(a, b)| a != b).count();
        if reference.iter().all(|&b| b == 0xA5) {
            why.push(format!("chain {cn}: arm A wrote nothing"));
        }
        for (i, &arm) in arms.iter().enumerate().skip(1) {
            let d = diff(&run_once(execs[i])?);
            println!("PDL_CMP chain={cn} arm={} bytes_differ={d}", arm.name());
            if d != 0 {
                why.push(format!(
                    "chain {cn}: arm {} differs from A in {d} bytes",
                    arm.name()
                ));
            }
            let mut bad = 0;
            for _ in 0..RACE_REPLAYS {
                if diff(&run_once(execs[i])?) != 0 {
                    bad += 1;
                }
            }
            println!(
                "PDL_RACE chain={cn} arm={} replays={RACE_REPLAYS} bad={bad}",
                arm.name()
            );
            if bad != 0 {
                why.push(format!(
                    "chain {cn}: arm {} differed in {bad} of {RACE_REPLAYS} replays",
                    arm.name()
                ));
            }
        }

        let mut times: Vec<Vec<f64>> = vec![Vec::new(); 3];
        let one = |exec: u64| -> Result<f64> {
            launch(&flush_ks, Arm::A, &flush_step, stream, false)?;
            // SAFETY: events created above; the interval holds exactly one graph launch.
            unsafe {
                ck(cuEventRecord(ev[0], stream), "cuEventRecord")?;
                ck(cuGraphLaunch(exec, stream), "cuGraphLaunch")?;
                ck(cuEventRecord(ev[1], stream), "cuEventRecord")?;
                ck(cuEventSynchronize(ev[1]), "cuEventSynchronize")?;
                let mut ms = 0f32;
                ck(
                    cuEventElapsedTime(&mut ms, ev[0], ev[1]),
                    "cuEventElapsedTime",
                )?;
                Ok(ms as f64)
            }
        };
        for round in 0..ROUNDS {
            for j in 0..3 {
                let i = (round + j) % 3;
                for _ in 0..WARMUP {
                    one(execs[i])?;
                }
                let mut t = Vec::with_capacity(REPLAYS);
                for _ in 0..REPLAYS {
                    t.push(one(execs[i])?);
                }
                println!(
                    "PDL_T_ROUND chain={cn} round={round} arm={} ms={:.4}",
                    arms[i].name(),
                    median(t.clone())
                );
                times[i].extend(t);
            }
        }
        let ms: Vec<f64> = times.into_iter().map(median).collect();
        for (i, &arm) in arms.iter().enumerate() {
            println!("PDL_T chain={cn} arm={} ms={:.4}", arm.name(), ms[i]);
        }
        for (i, &arm) in arms.iter().enumerate().skip(1) {
            println!(
                "PDL_BOUNDARY chain={cn} arm={} saving_us={:.3} boundaries={boundaries}",
                arm.name(),
                (ms[0] - ms[i]) * 1e3 / boundaries as f64
            );
        }
        for e in execs {
            // SAFETY: destroys an executable graph created above.
            unsafe { cuGraphExecDestroy(e) };
        }
    }

    // Real-trace launch counts per verify round (nsys sqlite ra-comb27c-n1n2-carmtp, rank 0,
    // mean over 160 rounds, measured 2026-10-12) to scale a per-boundary saving to ms per round.
    println!(
        "PDL_TRACE chain=S launches_per_round=209 (router_batchm 42, fp32out_wq_b 33, tcm_bf16_N128 100, tcm_bf16_N4096_K128 34) coverage_pct_of_small_dense=79.7 ms_per_round=2.848"
    );
    println!(
        "PDL_TRACE chain=D nv4_staged_grid1024_launches_per_round=192 (kda qkvo 136, shared down 42, other 14) nv4_staged_grid256_launches_per_round=84 (shared gate+up)"
    );
    println!(
        "PDL_TRACE chain=G nv4_staged_launches_per_round_replaced=kda q/k/v 3 x 34 = 102 by 34 grouped; shared gate+up 84 by 42 grouped"
    );
    if why.is_empty() {
        Ok((true, "PASS".into()))
    } else {
        Ok((false, format!("FAIL {}", why.join("; "))))
    }
}

fn main() {
    match run() {
        Ok((true, line)) => println!("{line}"),
        Ok((false, line)) => {
            println!("{line}");
            std::process::exit(1);
        }
        Err(e) => {
            println!("FAIL error: {e:#}");
            std::process::exit(1);
        }
    }
}
