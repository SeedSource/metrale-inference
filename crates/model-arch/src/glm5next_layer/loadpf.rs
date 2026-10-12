// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Decode L2 LOAD-prefetch on a side stream (`METRALE_GLM_DECODE_L2_LOADPF=1`,
//! default off).
//!
//! On GB10 (24 MB L2) prefetch hints do not reliably leave lines resident, and the hint-based
//! `METRALE_GLM_DECODE_L2_PREFETCH` (prefetch.rs) measured slower. Real loads do allocate in L2.
//! So a use site forks a SIDE stream off the main stream, runs `glm5next_l2_loadpf` there (one
//! u32 load per 32-byte sector of the next GEMV's weight, small grid
//! `METRALE_GLM_DECODE_L2_LOADPF_CTAS`), and the main stream joins it right before that GEMV:
//!
//! * Site 1 (`METRALE_GLM_DECODE_L2_LOADPF`): each KDA layer forks right after its front end
//!   (in-proj and gate GEMVs; `glm5next_kda/decode.rs`) and joins right before `o_proj`
//!   (`glm5next_kda/mod.rs` `back_end_with`). Between them the main stream runs the
//!   latency-bound recurrence (and its snapshot copies on a verify), which leaves the DRAM idle.
//! * Site 2 (`METRALE_GLM_DECODE_L2_LOADPF_SHARED`, also default off): each routed MoE layer
//!   forks right after the router top-k and joins right before the shared-expert gate/up GEMVs
//!   (`glm5next_mlp/forward.rs`). There is NO non-streaming window between those points: the
//!   routed experts (U experts x ~7 MB per rank) stream in between, far more than the L2, and
//!   compete for the DRAM. The switch exists to measure that, not because it is expected to pay.
//!
//! Fork/join is event-based and identical eager and under capture: `record(ev_fork, main)`,
//! `side waits ev_fork`, the load-prefetch on `side`, `record(ev_join, side)`, and at the join
//! `main waits ev_join`. Eager: the side stream runs concurrently with the main stream's next
//! kernels and the main stream blocks (on the device) only at the join. Under stream capture
//! (relaxed mode, decode/verify graphs) the wait on an event recorded in the capturing stream
//! pulls `side` into the capture, the launch becomes a parallel branch of the graph, and the
//! join's wait merges it back before the consuming node; every fork is joined in the same call
//! so `cuStreamEndCapture` never sees an unjoined branch (DeepSeek-V4.1 `step_seg_run.rs` uses
//! the same pattern).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Byte-identical by construction: the kernel writes no device memory (its one store is
//!   guarded by a null `sink`); the join only orders work.
//! - Side streams and events are created once per rank at load ([`LoadPfLanes`]), never per
//!   step. A site shares its lane with every layer: one layer's fork is joined before the next
//!   layer forks.
//! - Steps of 1..=`L2_LOADPF_MAX_ROWS` rows only; the decode callers never fork for a prefill
//!   sub-chunk. A join without a pending fork is a no-op.
//! - Spans are the exact extents the consuming GEMV reads (`gemv_spans`), clipped by
//!   `clip_spans` to `L2_LOADPF_MAX_BYTES` (lengths rounded down to 16 bytes), so no load leaves
//!   an allocation.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::prefetch::{L2_PREFETCH_MAX_SPANS, L2Span, clip_spans};
use super::{Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mlp::Glm5NextMlpConfig;

/// 2026-10-11: The module (`kernels/gb10/glm-5.3-flash/nvfp4/glm5next_l2_loadpf.cu`).
pub const L2_LOADPF_MODULE: &str = "glm5next_l2_loadpf";
/// 2026-10-11: The entry point.
pub const L2_LOADPF_KERNEL: &str = "glm5next_l2_loadpf";
/// 2026-10-11: `GLM_L2LPF_BLOCK` in the kernel.
pub const L2_LOADPF_BLOCK: u32 = 256;
/// 2026-10-11: `GLM_L2LPF_SECTOR`: one u32 load per sector.
pub const L2_LOADPF_SECTOR: usize = 32;
/// 2026-10-11: `GLM_L2LPF_UNROLL`: sectors per thread per grid-stride step.
pub const L2_LOADPF_UNROLL: usize = 8;
/// 2026-10-11: Widest step that forks: decode, multi-sequence decode and every verify width in
/// use (MTP K3, DFlash up to 8 per sequence, batched verify up to 16 rows).
pub const L2_LOADPF_MAX_ROWS: usize = 16;
/// 2026-10-11: Bytes per launch: two thirds of the 24 MB L2 (a KDA `o_proj` is ~9.4 MB per
/// rank as NVFP4 packed + scales).
pub const L2_LOADPF_MAX_BYTES: usize = 16 << 20;
/// 2026-10-11: Default and cap of `METRALE_GLM_DECODE_L2_LOADPF_CTAS` (cap: one per GB10 SM).
pub const L2_LOADPF_CTAS: u32 = 8;
pub const L2_LOADPF_MAX_CTAS: u32 = 48;

/// 2026-10-11: `1` (trimmed) is on; anything else off.
pub(crate) fn parse_switch(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// 2026-10-11: An integer in `1..=L2_LOADPF_MAX_CTAS`, else `None`.
pub(crate) fn parse_ctas(v: Option<&str>) -> Option<u32> {
    v.and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|c| (1..=L2_LOADPF_MAX_CTAS).contains(c))
}

/// 2026-10-11: `METRALE_GLM_DECODE_L2_LOADPF=1` (site 1, KDA `o_proj`). Read once.
pub fn l2_loadpf() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let on = parse_switch(
            std::env::var("METRALE_GLM_DECODE_L2_LOADPF")
                .ok()
                .as_deref(),
        );
        if on {
            tracing::warn!(
                "METRALE_GLM_DECODE_L2_LOADPF=1 - KDA o_proj weights are load-prefetched into L2 \
                 on a side stream during the recurrence (steps of <= {L2_LOADPF_MAX_ROWS} rows, \
                 {} CTAs, byte-identical by construction)",
                l2_loadpf_ctas()
            );
        }
        on
    })
}

/// 2026-10-11: `METRALE_GLM_DECODE_L2_LOADPF_SHARED=1` (site 2, shared expert); needs
/// [`l2_loadpf`]. Read once.
pub fn l2_loadpf_shared() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DECODE_L2_LOADPF_SHARED").ok();
        let set = parse_switch(raw.as_deref());
        if set && !l2_loadpf() {
            tracing::warn!(
                "METRALE_GLM_DECODE_L2_LOADPF_SHARED=1 without METRALE_GLM_DECODE_L2_LOADPF=1 - off"
            );
        }
        set && l2_loadpf()
    })
}

/// 2026-10-11: `METRALE_GLM_DECODE_L2_LOADPF_CTAS` (1..=48), else [`L2_LOADPF_CTAS`]. Read once.
pub fn l2_loadpf_ctas() -> u32 {
    static C: OnceLock<u32> = OnceLock::new();
    *C.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DECODE_L2_LOADPF_CTAS").ok();
        let parsed = parse_ctas(raw.as_deref());
        if let (Some(r), None) = (raw.as_deref(), parsed)
            && !r.trim().is_empty()
        {
            tracing::warn!(
                "METRALE_GLM_DECODE_L2_LOADPF_CTAS={r} is not an integer in \
                 1..={L2_LOADPF_MAX_CTAS} - using {L2_LOADPF_CTAS}"
            );
        }
        parsed.unwrap_or(L2_LOADPF_CTAS)
    })
}

/// 2026-10-11: An NVFP4 `[n, k]` weight as the decode GEMV reads it: packed E2M1 (`n * k / 2`
/// bytes) and the E4M3 block scales (`n * k / 16` bytes, one per 16 K), the layout
/// `dense_fp8::register` allocates and `nv4_gemv` reads.
pub fn nv4_spans(packed: DevicePtr, scales: DevicePtr, n: usize, k: usize) -> [L2Span; 2] {
    [
        L2Span {
            ptr: packed,
            bytes: n * k / 2,
        },
        L2Span {
            ptr: scales,
            bytes: n * k / 16,
        },
    ]
}

/// 2026-10-11: An FP8 `[n, k]` copy: the E4M3 bytes (`n * k`) and its F32 scales, per row
/// (`n`) or 128x128 block-scaled (`ceil(n / 128) * ceil(k / 128)`).
pub fn fp8_spans(w: DevicePtr, scale: DevicePtr, n: usize, k: usize, bs: bool) -> [L2Span; 2] {
    let scales = if bs {
        n.div_ceil(128) * k.div_ceil(128)
    } else {
        n
    };
    [
        L2Span {
            ptr: w,
            bytes: n * k,
        },
        L2Span {
            ptr: scale,
            bytes: scales * 4,
        },
    ]
}

/// 2026-10-11: The bytes a decode GEMV on `[n, k]` weight `ptr` (a `dense_fp8` registry key
/// or a BF16 matrix) streams, data first: the NVFP4 copy when it has one, else the FP8 copy,
/// else the BF16 matrix (`dense_fp8::decode_span`, plus the scales it leaves out).
pub fn gemv_spans(ptr: DevicePtr, n: usize, k: usize) -> Vec<L2Span> {
    use super::dense_fp8::{is_block_scaled, lookup, lookup_nv4};
    if let Some(q) = lookup_nv4(ptr, n, k) {
        return nv4_spans(q.weight, q.weight_scale, n, k).to_vec();
    }
    if let Some(w) = lookup(ptr, n, k) {
        return fp8_spans(w.weight, w.row_scale, n, k, is_block_scaled(ptr, n, k)).to_vec();
    }
    vec![super::prefetch::bf16_matrix_span(ptr, n, k)]
}

/// 2026-10-11: CTAs for one launch: `ctas` (clamped to `1..=L2_LOADPF_MAX_CTAS`), but no more
/// than the spans' sectors need at `L2_LOADPF_UNROLL` sectors per thread.
pub fn loadpf_grid(spans: &[L2Span], ctas: u32) -> u32 {
    let sectors: usize = spans
        .iter()
        .map(|s| s.bytes.div_ceil(L2_LOADPF_SECTOR))
        .sum();
    let per_cta = L2_LOADPF_BLOCK as usize * L2_LOADPF_UNROLL;
    let need = sectors.div_ceil(per_cta).max(1) as u32;
    ctas.clamp(1, L2_LOADPF_MAX_CTAS).min(need)
}

/// 2026-10-11: One `glm5next_l2_loadpf` launch over `spans` (at most `L2_PREFETCH_MAX_SPANS`,
/// already clipped) on `stream`, `sink` null. No launch when `kernel` is unresolved or `spans`
/// is empty. Static arguments: graph-capture and replay safe.
pub fn launch_l2_loadpf(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    spans: &[L2Span],
    ctas: u32,
    stream: u64,
) -> Result<()> {
    if kernel.0 == 0 || spans.is_empty() {
        return Ok(());
    }
    let mut l = KernelLaunch::new(gpu, kernel)
        .grid([loadpf_grid(spans, ctas), 1, 1])
        .block([L2_LOADPF_BLOCK, 1, 1]);
    for i in 0..L2_PREFETCH_MAX_SPANS {
        let (p, n) = spans
            .get(i)
            .map_or((DevicePtr(0), 0u64), |x| (x.ptr, x.bytes as u64));
        l = l.arg_ptr(p).arg_u64(n);
    }
    l.arg_ptr(DevicePtr(0)).launch(stream)
}

/// 2026-10-11: One use site's side stream and fork/join events, created once at load.
#[derive(Debug, Clone, Copy)]
pub struct LoadPfLane {
    pub kernel: KernelHandle,
    pub side: u64,
    pub ev_fork: u64,
    pub ev_join: u64,
}

impl LoadPfLane {
    /// 2026-10-11: A new non-blocking side stream and two timing-disabled events.
    pub fn create(gpu: &dyn GpuBackend, kernel: KernelHandle) -> Result<Self> {
        Ok(Self {
            kernel,
            side: gpu
                .create_stream()
                .context("L2 load-prefetch: the side stream")?,
            ev_fork: gpu
                .create_event()
                .context("L2 load-prefetch: the fork event")?,
            ev_join: gpu
                .create_event()
                .context("L2 load-prefetch: the join event")?,
        })
    }
}

/// 2026-10-11: One layer's use site: its lane, its clipped spans and whether a fork awaits its
/// join (host flag; a graph replays the recorded fork and join together).
#[derive(Debug)]
pub struct LoadPfSite {
    lane: LoadPfLane,
    spans: Vec<L2Span>,
    ctas: u32,
    pending: AtomicBool,
}

impl LoadPfSite {
    /// 2026-10-11: `None` when the kernel is unresolved or nothing survives `clip_spans(spans,
    /// L2_LOADPF_MAX_BYTES)`.
    pub fn new(lane: LoadPfLane, spans: &[L2Span], ctas: u32) -> Option<Self> {
        let spans = clip_spans(spans, L2_LOADPF_MAX_BYTES);
        (lane.kernel.0 != 0 && !spans.is_empty()).then(|| Self {
            lane,
            spans,
            ctas,
            pending: AtomicBool::new(false),
        })
    }

    pub fn spans(&self) -> &[L2Span] {
        &self.spans
    }

    /// 2026-10-11: For a step of 1..=`L2_LOADPF_MAX_ROWS` rows: record the fork on `stream`, let
    /// the side stream wait for it, launch the load-prefetch there and record the join. Returns
    /// whether it forked.
    pub fn fork(&self, gpu: &dyn GpuBackend, rows: usize, stream: u64) -> Result<bool> {
        if rows == 0 || rows > L2_LOADPF_MAX_ROWS {
            return Ok(false);
        }
        let l = &self.lane;
        gpu.record_event(l.ev_fork, stream)?;
        gpu.stream_wait_event(l.side, l.ev_fork)?;
        launch_l2_loadpf(gpu, l.kernel, &self.spans, self.ctas, l.side)?;
        gpu.record_event(l.ev_join, l.side)?;
        self.pending.store(true, Ordering::Release);
        Ok(true)
    }

    /// 2026-10-11: `stream` waits (on the device) for the pending fork's load-prefetch; no-op
    /// without one.
    pub fn join(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if self.pending.swap(false, Ordering::AcqRel) {
            gpu.stream_wait_event(stream, self.lane.ev_join)?;
        }
        Ok(())
    }
}

/// 2026-10-11: [`LoadPfSite::fork`] when `site` is set.
pub fn fork(
    site: Option<&LoadPfSite>,
    gpu: &dyn GpuBackend,
    rows: usize,
    stream: u64,
) -> Result<()> {
    if let Some(s) = site {
        s.fork(gpu, rows, stream)?;
    }
    Ok(())
}

/// 2026-10-11: [`LoadPfSite::join`] when `site` is set.
pub fn join(site: Option<&LoadPfSite>, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    match site {
        Some(s) => s.join(gpu, stream),
        None => Ok(()),
    }
}

/// 2026-10-11: The per-rank lanes, created on first use by [`attach`].
#[derive(Debug, Default)]
pub struct LoadPfLanes {
    kda: Option<LoadPfLane>,
    shared: Option<LoadPfLane>,
    kernel: Option<KernelHandle>,
}

fn total(spans: &[L2Span]) -> usize {
    spans.iter().map(|s| s.bytes).sum()
}

/// 2026-10-11: Loader hook, after `dense_fp8::register_layer` (so the weight fields name the
/// copies the GEMVs read): under the levers, give the KDA layer its `o_proj` site and the MoE
/// site its shared-expert site. No-op with the lever off; a target without the kernel warns
/// once and attaches nothing.
pub fn attach(
    gpu: &dyn GpuBackend,
    lanes: &mut LoadPfLanes,
    mixer: &mut Glm5NextMixer,
    mlp: &mut Glm5NextMlpSite,
    mlp_cfg: &Glm5NextMlpConfig,
) -> Result<()> {
    if !l2_loadpf() {
        return Ok(());
    }
    let kernel = *lanes.kernel.get_or_insert_with(|| {
        let k = metrale_model_layers::layers::try_kernel(gpu, L2_LOADPF_MODULE, L2_LOADPF_KERNEL);
        if k.0 == 0 {
            tracing::warn!(
                "METRALE_GLM_DECODE_L2_LOADPF=1 but {L2_LOADPF_MODULE}::{L2_LOADPF_KERNEL} is not \
                 in this target - no load-prefetch"
            );
        }
        k
    });
    if kernel.0 == 0 {
        return Ok(());
    }
    let ctas = l2_loadpf_ctas();
    if let Glm5NextMixer::Kda { layer, .. } = mixer {
        let (hid, qkv) = (layer.cfg.hidden, layer.cfg.qkv_dim());
        let spans = gemv_spans(layer.weights.o_proj.weight, hid, qkv);
        let first = lanes.kda.is_none();
        let lane = match lanes.kda {
            Some(l) => l,
            None => *lanes.kda.insert(LoadPfLane::create(gpu, kernel)?),
        };
        layer.loadpf = LoadPfSite::new(lane, &spans, ctas);
        if first && let Some(s) = layer.loadpf.as_ref() {
            tracing::info!(
                "METRALE_GLM_DECODE_L2_LOADPF: KDA o_proj [{}, {}] site: {} spans, {} bytes per \
                 launch (of {} the GEMV reads), {} CTAs",
                hid,
                qkv,
                s.spans().len(),
                total(s.spans()),
                total(&spans),
                loadpf_grid(s.spans(), ctas)
            );
        }
    }
    if l2_loadpf_shared()
        && let Glm5NextMlpSite::Moe(w) = mlp
    {
        let (n, k) = (mlp_cfg.local_shared_intermediate, mlp_cfg.hidden);
        let mut spans = gemv_spans(w.shared.gate_proj, n, k);
        spans.extend(gemv_spans(w.shared.up_proj, n, k));
        let first = lanes.shared.is_none();
        let lane = match lanes.shared {
            Some(l) => l,
            None => *lanes.shared.insert(LoadPfLane::create(gpu, kernel)?),
        };
        w.loadpf_shared = LoadPfSite::new(lane, &spans, ctas);
        if first && let Some(s) = w.loadpf_shared.as_ref() {
            tracing::info!(
                "METRALE_GLM_DECODE_L2_LOADPF_SHARED: shared expert gate+up [{n}, {k}] site: {} \
                 spans, {} bytes per launch",
                s.spans().len(),
                total(s.spans())
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "loadpf_tests.rs"]
mod loadpf_tests;
