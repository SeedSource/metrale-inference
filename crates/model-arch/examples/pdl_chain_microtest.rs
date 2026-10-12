// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Gate 1 for programmatic dependent launch (PDL) on the decode GEMV chain
//! (SeedHQ/seed-skills#80).
//!
//! Owner: model-arch examples.
//! One layer's most numerous NVFP4 launches at M = 3 rows, `w4a16_gemv_batch3_staged`:
//! KDA in-proj 4096x4096 (grid 1024) -> shared gate 1024x4096 (grid 256) -> shared up 1024x4096
//! (grid 256) -> shared down 4096x1024 (grid 1024, reads the gate output). The down output
//! feeds the next layer's in-proj. L = 42 layers with distinct weight buffers (cold-stream, ~0.7
//! GB), so 168 launches and 167 boundaries. Numerics are irrelevant; only that the arms agree.
//!
//! Arms, each captured by stream capture (relaxed mode, as the engine) into one CUDA graph:
//! - A: `w4a16_gemv_batch3_staged`, `cuLaunchKernel`.
//! - B: `w4a16_gemv_batch3_staged_pdl`, `cuLaunchKernelEx` with
//!   `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 1` on every launch after the first
//!   (kernel: `launch_dependents` at start, `griddepcontrol.wait` before any read).
//! - C: `w4a16_gemv_batch3_staged_pdl_touch`, launched like B (kernel: the CTA's first weight word
//!   and scale byte are loaded before the wait).
//!
//! Prints (one per line): `PDL_EDGES arm=X total=N programmatic=P` (cuGraphGetEdges_v2 on the
//! captured graph; `PDL_EDGES_DROPPED arm=X` when B or C has no programmatic edge: a finding,
//! not a crash), `PDL_CMP arm=X bytes_differ=N` (every layer's four output buffers, raw bytes,
//! vs arm A), `PDL_RACE arm=X replays=5 bad=K` (five more replays, each compared with A),
//! `PDL_T arm=X ms=M` (median graph replay, 3 rounds x 20 replays, order rotated per round,
//! cold L2 by a read-only 72 MiB GEMV before every replay, outside the timed region),
//! `PDL_BOUNDARY arm=X saving_us=S` ((A - X) / 167). Final line `PASS` iff every compare is
//! equal and every line above was printed; else `FAIL <why>` and exit 1. Timing is only
//! meaningful when run alone on a GPU.
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

/// Layers, rows, the per-layer GEMV shapes (n, k) in launch order, and the timing plan.
const LAYERS: usize = 42;
const M: usize = 3;
/// (n, k): in-proj, gate, up, down.
const SHAPES: [(usize, usize); 4] = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 1024)];
const ROUNDS: usize = 3;
const REPLAYS: usize = 20;
const WARMUP: usize = 3;
const RACE_REPLAYS: usize = 5;
const FLUSH_N: usize = 4096;
const FLUSH_K: usize = 32768;

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

/// One launch of the chain: input, weight, output, shape.
struct Step {
    a: u64,
    w: W,
    c: u64,
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

/// Launch `step` with kernel `f` on `stream`; `pdl` adds the programmatic-serialization attribute.
fn launch(f: KernelHandle, st: &Step, stream: u64, pdl: bool) -> Result<()> {
    let (mut a, mut w, mut s, mut s2, mut c) = (st.a, st.w.w, st.w.s, st.w.s2, st.c);
    let (mut n, mut k) = (st.n as u32, st.k as u32);
    let mut params: [*mut c_void; 7] = [
        &mut a as *mut u64 as *mut c_void,
        &mut w as *mut u64 as *mut c_void,
        &mut s as *mut u64 as *mut c_void,
        &mut s2 as *mut f32 as *mut c_void,
        &mut c as *mut u64 as *mut c_void,
        &mut n as *mut u32 as *mut c_void,
        &mut k as *mut u32 as *mut c_void,
    ];
    let grid = (st.n as u32).div_ceil(4);
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
fn capture(arm: Arm, ks: [KernelHandle; 3], steps: &[Step], stream: u64) -> Result<(u64, u64)> {
    let f = ks[arm as usize];
    // SAFETY: mode 2 = CU_STREAM_CAPTURE_MODE_RELAXED, as the engine's begin_capture.
    ck(
        unsafe { cuStreamBeginCapture(stream, 2) },
        "cuStreamBeginCapture",
    )?;
    let mut res = Ok(());
    for (i, st) in steps.iter().enumerate() {
        res = launch(f, st, stream, arm != Arm::A && i > 0);
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

/// (total edges, programmatic edges, edges leaving a programmatic port).
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
    let mut host = Vec::with_capacity(n * k * 2);
    for _ in 0..n * k {
        host.extend_from_slice(&bf16::from_f64(rng.n() * 0.02).to_bits().to_le_bytes());
    }
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

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<u64> {
    let p = g.alloc(b.len())?;
    g.copy_h2d(b, p)?;
    Ok(p.0)
}

fn run() -> Result<(bool, String)> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let ks = [
        g.kernel("w4a16_gemv", "w4a16_gemv_batch3_staged")?,
        g.kernel("w4a16_gemv", "w4a16_gemv_batch3_staged_pdl")?,
        g.kernel("w4a16_gemv", "w4a16_gemv_batch3_staged_pdl_touch")?,
    ];
    let gemv1 = g.kernel("w4a16_gemv", "w4a16_gemv")?;
    let absmax = g.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quant = g.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse")?;
    let stream = g.default_stream();
    let mut rng = Lcg(0x7064_6c63_6861_696e);

    // Base quantized weights per shape (gate and up share the shape and the base bytes); each
    // layer's copy is the base rotated by a layer- and slot-dependent byte offset, so the 168
    // buffers differ and are distinct allocations.
    let base: Vec<(Vec<u8>, Vec<u8>, f32)> = [SHAPES[0], SHAPES[1], SHAPES[3]]
        .iter()
        .map(|&(n, k)| quantize(g, absmax, quant, &mut rng, n, k))
        .collect::<Result<_>>()?;
    let mut weights: Vec<[W; 4]> = Vec::new();
    for l in 0..LAYERS {
        let mut row = [W {
            w: 0,
            s: 0,
            s2: 0.0,
        }; 4];
        for (slot, wr) in row.iter_mut().enumerate() {
            let b = &base[[0, 1, 1, 2][slot]];
            let mut w = b.0.clone();
            let mut s = b.1.clone();
            let r = (l * 4 + slot) * 7919;
            let (rw, rs) = (r % w.len(), r % s.len());
            w.rotate_left(rw);
            s.rotate_left(rs);
            *wr = W {
                w: up(g, &w)?,
                s: up(g, &s)?,
                s2: b.2,
            };
        }
        weights.push(row);
    }

    // Activations: layer 0 input, then one output arena [layer][in, gate, up, down] x M rows.
    let mut a0 = Vec::with_capacity(M * 4096 * 2);
    for _ in 0..M * 4096 {
        a0.extend_from_slice(&bf16::from_f64(rng.n()).to_bits().to_le_bytes());
    }
    let a0 = up(g, &a0)?;
    let per_layer: usize = SHAPES.iter().map(|&(n, _)| M * n * 2).sum();
    let arena_bytes = LAYERS * per_layer;
    let arena = g.alloc(arena_bytes)?.0;
    let mut steps = Vec::new();
    let mut x = a0;
    for (l, wl) in weights.iter().enumerate() {
        let o = arena + (l * per_layer) as u64;
        let o_in = o;
        let o_gate = o_in + (M * SHAPES[0].0 * 2) as u64;
        let o_up = o_gate + (M * SHAPES[1].0 * 2) as u64;
        let o_down = o_up + (M * SHAPES[2].0 * 2) as u64;
        let mk = |a: u64, i: usize, c: u64| Step {
            a,
            w: wl[i],
            c,
            n: SHAPES[i].0,
            k: SHAPES[i].1,
        };
        steps.push(mk(x, 0, o_in));
        steps.push(mk(o_in, 1, o_gate));
        steps.push(mk(o_in, 2, o_up));
        steps.push(mk(o_gate, 3, o_down));
        x = o_down;
    }
    let boundaries = steps.len() - 1;

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
    let flush_step = Step {
        a: fa,
        w: fw,
        c: fc,
        n: FLUSH_N,
        k: FLUSH_K,
    };

    let arms = [Arm::A, Arm::B, Arm::C];
    let mut execs = Vec::new();
    let mut why: Vec<String> = Vec::new();
    for &arm in &arms {
        let (graph, exec) = capture(arm, ks, &steps, stream)?;
        let (total, prog, port) = edges(graph)?;
        println!(
            "PDL_EDGES arm={} total={total} programmatic={prog} from_port_nonzero={port}",
            arm.name()
        );
        if arm != Arm::A && prog == 0 {
            println!("PDL_EDGES_DROPPED arm={}", arm.name());
        }
        // SAFETY: the template graph is no longer needed once instantiated.
        unsafe { cuGraphDestroy(graph) };
        execs.push(exec);
    }

    let run_once = |exec: u64| -> Result<Vec<u8>> {
        g.memset(DevicePtr(arena), 0xA5, arena_bytes)?;
        // SAFETY: replays an instantiated graph on the capture stream, then waits.
        ck(unsafe { cuGraphLaunch(exec, stream) }, "cuGraphLaunch")?;
        ck(
            unsafe { cuStreamSynchronize(stream) },
            "cuStreamSynchronize",
        )?;
        let mut out = vec![0u8; arena_bytes];
        g.copy_d2h(DevicePtr(arena), &mut out)?;
        Ok(out)
    };
    let reference = run_once(execs[0])?;
    let diff = |o: &[u8]| o.iter().zip(&reference).filter(|(a, b)| a != b).count();
    if reference.iter().all(|&b| b == 0xA5) {
        why.push("arm A wrote nothing".into());
    }
    for (i, &arm) in arms.iter().enumerate().skip(1) {
        let d = diff(&run_once(execs[i])?);
        println!("PDL_CMP arm={} bytes_differ={d}", arm.name());
        if d != 0 {
            why.push(format!("arm {} differs from A in {d} bytes", arm.name()));
        }
        let mut bad = 0;
        for _ in 0..RACE_REPLAYS {
            if diff(&run_once(execs[i])?) != 0 {
                bad += 1;
            }
        }
        println!(
            "PDL_RACE arm={} replays={RACE_REPLAYS} bad={bad}",
            arm.name()
        );
        if bad != 0 {
            why.push(format!(
                "arm {} differed in {bad} of {RACE_REPLAYS} replays",
                arm.name()
            ));
        }
    }

    // Timing.
    let mut ev = [0u64; 2];
    for e in ev.iter_mut() {
        // SAFETY: creates one event handle into `e`.
        ck(unsafe { cuEventCreate(e, 0) }, "cuEventCreate")?;
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); 3];
    let one = |exec: u64| -> Result<f64> {
        launch(gemv1, &flush_step, stream, false)?;
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
                "PDL_T_ROUND round={round} arm={} ms={:.4}",
                arms[i].name(),
                median(t.clone())
            );
            times[i].extend(t);
        }
    }
    let ms: Vec<f64> = times.into_iter().map(median).collect();
    for (i, &arm) in arms.iter().enumerate() {
        println!("PDL_T arm={} ms={:.4}", arm.name(), ms[i]);
    }
    for (i, &arm) in arms.iter().enumerate().skip(1) {
        println!(
            "PDL_BOUNDARY arm={} saving_us={:.3}",
            arm.name(),
            (ms[0] - ms[i]) * 1e3 / boundaries as f64
        );
    }
    for e in execs {
        // SAFETY: destroys an executable graph created above.
        unsafe { cuGraphExecDestroy(e) };
    }
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
