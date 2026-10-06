// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1`: the GLM-5.3 routed-MoE prefill on the
//! CUTLASS Sm120 block-scaled NVFP4 grouped GEMM (`crates/gpu-runtime/cuda/
//! cutlass_nvfp4_grouped_gemm.cu`, built from NVIDIA CUTLASS, BSD-3-Clause, only when
//! `CUTLASS_HOME` is set), the kernel the Qwen MoE path already uses. After the expert sort,
//! three steps replace the grouped W4A16 gate GEMM, up GEMM, SwiGLU and down GEMM:
//!   1. gate/up: the routed rows of `x` are gathered and quantized to NVFP4 ONCE per call
//!      (shared by gate and up), then one `kGrouped` GEMM per projection into `ws.a_gate` /
//!      `ws.a_up` (BF16, expert-sorted rows);
//!   2. `glm5next_swiglu_clamp` into `ws.a_act`, exactly as the W4A16 path;
//!   3. down: `ws.a_act` quantized to NVFP4, one `kGrouped` GEMM into `ws.expert_out`.
//!
//! Activation scale (the W4A4 export contract). The ModelOpt NVFP4 export carries a static
//! per-expert, per-projection F32 `input_scale` (architecture packet section 2: "`input_scale`
//! F32 static activation scale, i.e. the export is W4A4"), the calibrated activation amax /
//! (6 * 448). It is used as the NVFP4 GLOBAL activation scale `gs`: per 16 values the UE4M3
//! block scale is `amax16 / 6 / gs` (saturated at 448) and the E2M1 codes are `v / (scale *
//! gs)`, rounded to nearest even; the GEMM epilogue multiplies by `weight_scale_2 * gs`. This
//! is how vLLM consumes the same tensors: `a1_gscale = 1 / a13_scale`
//! (`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:526` in the frozen GLM oracle image),
//! and for its CUTLASS backend `a13_scale = w13_input_scale.max(dim=1)`, the max over gate and
//! up per expert (`vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py:382`).
//! So gate and up share `max(gate.input_scale, up.input_scale)` here too, and down uses its
//! own. A projection without a usable `input_scale` (absent, zero, non-finite: e.g. the MTP
//! layer's experts, which Metrale quantizes at load) gets a DYNAMIC global scale: amax / (6 *
//! 448) over every row its experts read in that call, computed on the device.
//!
//! 2026-10-04: two independent opt-in fix levers (each default off, each only with the lever
//! above; lever off = the code above, unchanged), added after paired-seed quality testing
//! showed the static-scale W4A4 degrading one safety scenario:
//!   - `METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE=1`: ignore every static `input_scale`; all gate/up and
//!     down projections get the dynamic global scale (`gs = 0.0` in [`MoeTables`], the C side's
//!     existing amax / 2688 path). The startup log says which mode is on.
//!   - `METRALE_GLM_MOE_W4A4_DOWN_W4A16=1`: gate/up and the clamped SwiGLU stay as above, but
//!     the DOWN projection runs the W4A16 grouped GEMM on BF16 `ws.a_act` instead of quantizing
//!     it to NVFP4 (no second activation quantization). [`forward`] returns
//!     [`Outcome::DownPending`] after the SwiGLU and `dispatch.rs` runs the down kernel the
//!     W4A16 path would have chosen for the call (`w4a16_mma::forward_down` when that lever's
//!     contract holds, else the production `moe_grouped_gemm` tile). `ws.a_act` has the same
//!     layout either way: `swiglu_rows` writes `[te, moe_intermediate]` BF16 in expert-sorted
//!     row order, exactly what the production down launch reads (null gather map).
//!
//! The weight block scales are swizzled into CUTLASS's SFB layout per layer into one cache
//! (`SfbCache`, local experts x (gate + up + down), 216 MB at GLM-5.3 EP=2) and re-swizzled
//! only when the layer changes; the staged prefill runs all FFN windows of a layer back to
//! back, so that is once per layer per prompt chunk.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Lever off (default): [`lever_on`] is false, nothing is allocated, the loader keeps
//!   skipping `*.input_scale`, and `forward_moe_grouped_prefill` runs exactly as before.
//! - Lever on: outputs land in the same buffers and expert-sorted row order as the W4A16 path
//!   (`glm5next_moe_combine_indexed` reads them unchanged); rows of a remote expert are not
//!   written (the caller pre-zeroes `expert_out`).
//! - Refusals never produce output: a build without CUTLASS, a shape outside
//!   [`shape_contract`], no cache, or a CUTLASS error makes [`forward`] return
//!   [`Outcome::Refused`] with a logged reason and the caller runs the W4A16 path, which
//!   rewrites every local row.
//! - Numerics: W4A4, NOT bit-identical to any W4A16 path; quality is gated at model level.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::forward::Glm5NextMlpWorkspace;
use super::super::weights::{Glm5NextExpertWeights, Glm5NextMoeWeights};
use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-03: `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1` (read once, `metrale_config`).
pub fn lever_on() -> bool {
    metrale_config::glm_moe_prefill_cutlass_w4a4()
}

/// 2026-10-04: `METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE=1` (read once, `metrale_config`).
pub fn dynamic_scale_on() -> bool {
    metrale_config::glm_moe_w4a4_dynamic_scale()
}

/// 2026-10-04: `METRALE_GLM_MOE_W4A4_DOWN_W4A16=1` (read once, `metrale_config`).
pub fn down_w4a16_on() -> bool {
    metrale_config::glm_moe_w4a4_down_w4a16()
}

/// 2026-10-06: `METRALE_GLM_MOE_SWIGLU_LOCAL_ROWS=1` (read once, `metrale_config`).
pub fn swiglu_local_rows_on() -> bool {
    metrale_config::glm_moe_swiglu_local_rows()
}

/// 2026-10-06: The sorted rows the SwiGLU must cover: the local experts' span
/// `offsets[local.start] .. offsets[local.end]` when `local_only`, else every routed row `0..te`.
/// Rows are sorted by global expert id, so the local experts' rows are contiguous.
pub fn swiglu_span(
    offsets: &[i32],
    local: std::ops::Range<usize>,
    te: usize,
    local_only: bool,
) -> std::ops::Range<usize> {
    if !local_only {
        return 0..te;
    }
    let at = |e: usize| offsets.get(e).map_or(te, |&o| (o.max(0) as usize).min(te));
    let (lo, hi) = (at(local.start), at(local.end));
    lo..hi.max(lo)
}

/// 2026-10-03: The CUTLASS tile's K and the TMA row alignment: both GEMM dims a multiple of
/// 128 (GLM-5.3: hidden 4096, moe_intermediate 2048), at least one local expert, and at most
/// 65535 of them (the batched swizzle's grid.y). `Err` carries the refusal reason.
pub fn shape_contract(
    hidden: usize,
    moe_intermediate: usize,
    num_experts: usize,
    local_experts: usize,
) -> std::result::Result<(), String> {
    if hidden == 0 || !hidden.is_multiple_of(128) {
        return Err(format!("hidden {hidden} is not a positive multiple of 128"));
    }
    if moe_intermediate == 0 || !moe_intermediate.is_multiple_of(128) {
        return Err(format!(
            "moe_intermediate {moe_intermediate} is not a positive multiple of 128"
        ));
    }
    if local_experts == 0 || local_experts > 65_535 || local_experts > num_experts {
        return Err(format!(
            "{local_experts} local experts of {num_experts} is outside 1..=65535"
        ));
    }
    if num_experts > i32::MAX as usize {
        return Err(format!("{num_experts} experts overflow the C ABI's int"));
    }
    Ok(())
}

/// 2026-10-03: A usable static activation scale: finite and positive, else 0.0 ("dynamic").
pub fn sanitize_input_scale(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 { v } else { 0.0 }
}

/// 2026-10-03: The global scale gate and up share (one quantized A feeds both): the larger
/// usable one of the two `input_scale`s, as vLLM's `w13_input_scale.max(dim=1)`; 0.0 (dynamic)
/// when neither is usable.
pub fn gate_up_global_scale(gate: f32, up: f32) -> f32 {
    sanitize_input_scale(gate).max(sanitize_input_scale(up))
}

/// 2026-10-03: Where expert slot `s`'s swizzled SFB lives: `base + off_p + s * stride_p`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SfbLayout {
    pub base: u64,
    pub gate_off: u64,
    pub up_off: u64,
    pub down_off: u64,
    /// 2026-10-03: `sfb_bytes(moe_intermediate, hidden)`, gate and up.
    pub gu_stride: u64,
    /// 2026-10-03: `sfb_bytes(hidden, moe_intermediate)`.
    pub down_stride: u64,
}

impl SfbLayout {
    /// 2026-10-03: Gate | up | down regions for `slots` experts at `base`.
    pub fn new(base: u64, slots: usize, hidden: usize, moe_intermediate: usize) -> Self {
        let gu = metrale_gpu_runtime::cutlass::sfb_bytes(moe_intermediate, hidden) as u64;
        let dn = metrale_gpu_runtime::cutlass::sfb_bytes(hidden, moe_intermediate) as u64;
        let s = slots as u64;
        Self {
            base,
            gate_off: 0,
            up_off: gu * s,
            down_off: 2 * gu * s,
            gu_stride: gu,
            down_stride: dn,
        }
    }

    /// 2026-10-03: Bytes of the three regions.
    pub fn total_bytes(&self, slots: usize) -> usize {
        self.down_off as usize + self.down_stride as usize * slots
    }
}

/// 2026-10-03: One projection's host tables over all GLOBAL expert ids: device pointers to the
/// packed `[N, K/2]` E2M1 weight and the swizzled SFB, and the weight global scale. A remote id
/// has null pointers (the C side then gives it no group).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjTables {
    pub packed: Vec<u64>,
    pub sfb: Vec<u64>,
    pub scale2: Vec<f32>,
}

/// 2026-10-03: Everything the two CUTLASS calls take from the weights, per global id.
/// `gate_up_gs` / `down_gs`: the activation global scale per expert (0.0 = dynamic).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MoeTables {
    pub gate: ProjTables,
    pub up: ProjTables,
    pub down: ProjTables,
    pub gate_up_gs: Vec<f32>,
    pub down_gs: Vec<f32>,
}

impl MoeTables {
    /// 2026-10-03: Tables for `num_experts` global ids of which `local` (ascending, slot `s` =
    /// id `local.start + s`) are `experts`, with SFBs at `sfb`. 2026-10-04: `dynamic_scale` (the
    /// `METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE` lever) leaves every `gate_up_gs` / `down_gs` at 0.0,
    /// the dynamic global scale, whatever the checkpoint's `input_scale`s are.
    pub fn build(
        num_experts: usize,
        local: std::ops::Range<usize>,
        experts: &[Glm5NextExpertWeights],
        sfb: &SfbLayout,
        dynamic_scale: bool,
    ) -> Self {
        let mut t = Self {
            gate: ProjTables::zeroed(num_experts),
            up: ProjTables::zeroed(num_experts),
            down: ProjTables::zeroed(num_experts),
            gate_up_gs: vec![0.0; num_experts],
            down_gs: vec![0.0; num_experts],
        };
        for (s, e) in experts.iter().enumerate() {
            let id = local.start + s;
            if id >= num_experts || id >= local.end {
                break;
            }
            let s64 = s as u64;
            t.gate.set(
                id,
                e.gate_proj.packed.0,
                sfb.base + sfb.gate_off + s64 * sfb.gu_stride,
                e.gate_proj.scale_2,
            );
            t.up.set(
                id,
                e.up_proj.packed.0,
                sfb.base + sfb.up_off + s64 * sfb.gu_stride,
                e.up_proj.scale_2,
            );
            t.down.set(
                id,
                e.down_proj.packed.0,
                sfb.base + sfb.down_off + s64 * sfb.down_stride,
                e.down_proj.scale_2,
            );
            if !dynamic_scale {
                t.gate_up_gs[id] =
                    gate_up_global_scale(e.gate_proj.input_scale, e.up_proj.input_scale);
                t.down_gs[id] = sanitize_input_scale(e.down_proj.input_scale);
            }
        }
        t
    }

    /// 2026-10-03: (local experts with a static gate/up scale, with a static down scale).
    pub fn static_scale_counts(&self) -> (usize, usize) {
        let live = |i: usize| self.gate.packed[i] != 0;
        let n = self.gate.packed.len();
        (
            (0..n)
                .filter(|&i| live(i) && self.gate_up_gs[i] > 0.0)
                .count(),
            (0..n).filter(|&i| live(i) && self.down_gs[i] > 0.0).count(),
        )
    }
}

impl ProjTables {
    fn zeroed(n: usize) -> Self {
        Self {
            packed: vec![0; n],
            sfb: vec![0; n],
            scale2: vec![0.0; n],
        }
    }
    fn set(&mut self, id: usize, packed: u64, sfb: u64, scale2: f32) {
        self.packed[id] = packed;
        self.sfb[id] = if packed == 0 { 0 } else { sfb };
        self.scale2[id] = scale2;
    }
}

/// 2026-10-03: The swizzled-SFB cache for one layer's local experts. `key` is the layer it
/// holds (the device address of that layer's gate scale-pointer table, unique per layer while
/// the model lives), 0 when empty.
pub struct SfbCache {
    slots: usize,
    hidden: usize,
    moe_intermediate: usize,
    layout: SfbLayout,
    key: Mutex<u64>,
}

impl SfbCache {
    /// 2026-10-03: Bytes for `slots` experts at these dims (216 MB for GLM-5.3 EP=2: 144 x
    /// (2 x 512 KiB + 512 KiB)).
    pub fn bytes(slots: usize, hidden: usize, moe_intermediate: usize) -> usize {
        SfbLayout::new(0, slots, hidden, moe_intermediate).total_bytes(slots)
    }

    /// 2026-10-03: Allocate the cache (device memory, never freed while the cache lives).
    pub fn new(
        gpu: &dyn GpuBackend,
        slots: usize,
        hidden: usize,
        moe_intermediate: usize,
    ) -> Result<Self> {
        let bytes = Self::bytes(slots, hidden, moe_intermediate);
        let buf = gpu.alloc(bytes.max(1))?;
        Ok(Self {
            slots,
            hidden,
            moe_intermediate,
            layout: SfbLayout::new(buf.0, slots, hidden, moe_intermediate),
            key: Mutex::new(0),
        })
    }

    pub fn layout(&self) -> SfbLayout {
        self.layout
    }

    /// 2026-10-03: Swizzle the experts `first..first + count` of three device scale-pointer
    /// tables (indexed by global id, `[N, K/16]` E4M3 each; a null entry is skipped) into the
    /// cache, one batched launch per projection on `stream`. Forgets the held layer first, so a
    /// failed launch never leaves a stale key.
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &self,
        gate_scale_ptrs: DevicePtr,
        up_scale_ptrs: DevicePtr,
        down_scale_ptrs: DevicePtr,
        first: usize,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        if count > self.slots {
            bail!("SFB cache holds {} experts, {count} asked", self.slots);
        }
        *self.key.lock().unwrap_or_else(|p| p.into_inner()) = 0;
        let (h, mi) = (self.hidden as u32, self.moe_intermediate as u32);
        let l = self.layout;
        use metrale_gpu_runtime::cutlass::pack_weight_sfb_batched as swz;
        swz(
            gate_scale_ptrs.0,
            first as u32,
            count as u32,
            l.base + l.gate_off,
            l.gu_stride as usize,
            mi,
            h,
            true,
            stream,
        )?;
        swz(
            up_scale_ptrs.0,
            first as u32,
            count as u32,
            l.base + l.up_off,
            l.gu_stride as usize,
            mi,
            h,
            true,
            stream,
        )?;
        swz(
            down_scale_ptrs.0,
            first as u32,
            count as u32,
            l.base + l.down_off,
            l.down_stride as usize,
            h,
            mi,
            true,
            stream,
        )?;
        Ok(())
    }

    /// 2026-10-03: Make the cache hold `w`'s layer: a no-op when it already does, else
    /// [`Self::fill`] from `w.ptrs` and remember the layer. Returns whether it swizzled.
    pub fn ensure(
        &self,
        cfg: &Glm5NextMlpConfig,
        w: &Glm5NextMoeWeights,
        stream: u64,
    ) -> Result<bool> {
        let key = w.ptrs.gate.scale_ptrs.0;
        if *self.key.lock().unwrap_or_else(|p| p.into_inner()) == key {
            return Ok(false);
        }
        self.fill(
            w.ptrs.gate.scale_ptrs,
            w.ptrs.up.scale_ptrs,
            w.ptrs.down.scale_ptrs,
            cfg.local_expert_range().start,
            cfg.local_experts,
            stream,
        )?;
        *self.key.lock().unwrap_or_else(|p| p.into_inner()) = key;
        Ok(true)
    }
}

/// 2026-10-03: The device buffers and sizes of one routed-MoE prefill call.
pub struct RunArgs<'a> {
    /// 2026-10-03: `glm5next_swiglu_clamp`.
    pub swiglu: KernelHandle,
    /// 2026-10-03: Token-major `[tokens, hidden]` BF16 MoE input.
    pub x: DevicePtr,
    /// 2026-10-03: `[te]` sorted row -> token (`moe_sort_by_expert`).
    pub sorted_token_ids: DevicePtr,
    pub a_gate: DevicePtr,
    pub a_up: DevicePtr,
    pub a_act: DevicePtr,
    pub expert_out: DevicePtr,
    /// 2026-10-03: Routed rows (`rows * top_k`).
    pub te: usize,
    pub hidden: usize,
    pub moe_intermediate: usize,
    pub swiglu_limit: f32,
    /// 2026-10-03: Host copy of `expert_offsets`, `num_experts + 1` entries.
    pub offsets: &'a [i32],
    pub tables: &'a MoeTables,
    /// 2026-10-04: Stop after the SwiGLU: the caller runs the down projection on W4A16
    /// (`METRALE_GLM_MOE_W4A4_DOWN_W4A16=1`).
    pub skip_down: bool,
    /// 2026-10-06: Sorted rows the SwiGLU covers ([`swiglu_span`]); `0..te` writes every row.
    pub swiglu_rows: std::ops::Range<usize>,
}

/// 2026-10-03: CUTLASS W4A4 gate/up, the clamped SwiGLU, CUTLASS W4A4 down, all on `stream`.
/// 2026-10-04: With `skip_down`, only gate/up and the SwiGLU (`a_act` is then ready for a W4A16
/// down; `expert_out` is not written).
pub fn run(gpu: &dyn GpuBackend, a: &RunArgs<'_>, stream: u64) -> Result<()> {
    let t = a.tables;
    let (h, mi) = (a.hidden as u32, a.moe_intermediate as u32);
    metrale_gpu_runtime::cutlass::nvfp4_grouped_gate_up_w4a4(
        a.x.0,
        a.sorted_token_ids.0,
        &t.gate.packed,
        &t.gate.sfb,
        &t.gate.scale2,
        &t.up.packed,
        &t.up.sfb,
        &t.up.scale2,
        &t.gate_up_gs,
        a.a_gate.0,
        a.a_up.0,
        a.offsets,
        mi,
        h,
        stream,
    )?;
    // 2026-10-06: Elementwise over `swiglu_rows` only (BF16 `[te, moe_intermediate]` buffers).
    let (r, row_bytes) = (&a.swiglu_rows, a.moe_intermediate * 2);
    if !r.is_empty() {
        super::super::forward::swiglu_rows(
            gpu,
            a.swiglu,
            a.a_gate.offset(r.start * row_bytes),
            a.a_up.offset(r.start * row_bytes),
            a.a_act.offset(r.start * row_bytes),
            r.len() * a.moe_intermediate,
            a.swiglu_limit,
            stream,
        )?;
    }
    if a.skip_down {
        return Ok(());
    }
    metrale_gpu_runtime::cutlass::nvfp4_grouped_down_w4a4(
        a.a_act.0,
        &t.down.packed,
        &t.down.sfb,
        &t.down.scale2,
        &t.down_gs,
        a.expert_out.0,
        a.offsets,
        h,
        mi,
        stream,
    )
}

/// 2026-10-03: The process's cache, set by [`prepare_at_load`].
static CACHE: OnceLock<SfbCache> = OnceLock::new();

/// 2026-10-03: At load, before the KV pool is sized: with the lever on and a usable build and
/// shape, allocate the SFB cache and the CUTLASS workspace and return their bytes; `Ok(None)`
/// (logged) when the lever is off or refused. An allocation failure is an error.
pub fn prepare_at_load(gpu: &dyn GpuBackend, cfg: &Glm5NextMlpConfig) -> Result<Option<usize>> {
    if !lever_on() {
        return Ok(None);
    }
    if !metrale_gpu_runtime::cutlass::available() {
        tracing::error!(
            "METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1 REFUSED: this build has no CUTLASS objects \
             (CUTLASS_HOME unset at build time) — routed-MoE prefill stays W4A16"
        );
        return Ok(None);
    }
    if let Err(why) = shape_contract(
        cfg.hidden,
        cfg.moe_intermediate,
        cfg.num_experts,
        cfg.local_experts,
    ) {
        tracing::error!(
            "METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1 REFUSED: {why} — routed-MoE prefill stays W4A16"
        );
        return Ok(None);
    }
    if let Some(c) = CACHE.get() {
        return Ok(Some(SfbCache::bytes(c.slots, c.hidden, c.moe_intermediate)));
    }
    let cache = SfbCache::new(gpu, cfg.local_experts, cfg.hidden, cfg.moe_intermediate)?;
    let sfb = SfbCache::bytes(cfg.local_experts, cfg.hidden, cfg.moe_intermediate);
    let ws = metrale_gpu_runtime::cutlass::warm_workspace()?;
    let _ = CACHE.set(cache);
    tracing::warn!(
        "GLM routed-MoE prefill: CUTLASS W4A4 ON (METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1) — NVFP4 \
         activations against {}, down projection {}; NOT bit-identical to W4A16. SFB cache \
         {:.1} MB ({} local experts), CUTLASS workspace {:.1} MB",
        scale_mode_text(dynamic_scale_on()),
        down_mode_text(down_w4a16_on()),
        sfb as f64 / 1e6,
        cfg.local_experts,
        ws as f64 / 1e6
    );
    Ok(Some(sfb + ws))
}

/// 2026-10-04: Startup-log wording of the activation-scale mode.
fn scale_mode_text(dynamic: bool) -> &'static str {
    if dynamic {
        "the DYNAMIC per-call global scale (METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE=1; static \
         input_scales ignored)"
    } else {
        "the checkpoint input_scale (dynamic only where one is missing)"
    }
}

/// 2026-10-04: Startup-log wording of the down-projection mode.
fn down_mode_text(down_w4a16: bool) -> &'static str {
    if down_w4a16 {
        "W4A16 on BF16 activations (METRALE_GLM_MOE_W4A4_DOWN_W4A16=1)"
    } else {
        "W4A4"
    }
}

/// 2026-10-04: What [`forward`] did with a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// 2026-10-04: Refused before any output was trusted; the caller runs the W4A16 path.
    Refused,
    /// 2026-10-04: Gate/up, SwiGLU and down done; `expert_out` written.
    Done,
    /// 2026-10-04: Gate/up and SwiGLU done (`ws.a_act` ready); the caller runs the W4A16 down
    /// (`METRALE_GLM_MOE_W4A4_DOWN_W4A16=1`).
    DownPending,
}

/// 2026-10-03: Log the first few refusals of a call, then count silently.
fn refuse(why: &str) -> Result<Outcome> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 4 {
        tracing::error!(
            "GLM routed-MoE prefill CUTLASS W4A4: {why} — this call runs W4A16{}",
            if n == 3 {
                " (further refusals not logged)"
            } else {
                ""
            }
        );
    }
    Ok(Outcome::Refused)
}

/// 2026-10-03: The dispatch arm: after `moe_sort_by_expert`, run the routed experts of `te`
/// slots through CUTLASS W4A4. [`Outcome::Done`]: outputs written. [`Outcome::Refused`]: refused
/// with a logged reason before any output was trusted; the caller runs the W4A16 path.
/// 2026-10-04: [`Outcome::DownPending`] with `METRALE_GLM_MOE_W4A4_DOWN_W4A16=1`: only the down
/// projection is left to the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    te: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<Outcome> {
    let Some(cache) = CACHE.get() else {
        return refuse("not prepared at load (no CUTLASS build, or a refused shape)");
    };
    if cache.hidden != cfg.hidden
        || cache.moe_intermediate != cfg.moe_intermediate
        || cfg.local_experts > cache.slots
        || w.experts.len() != cfg.local_experts
    {
        return refuse("this MoE site's geometry differs from the prepared cache");
    }
    if k.swiglu.0 == 0 {
        return refuse("glm5next_swiglu_clamp is missing");
    }
    // 2026-10-03: The C side needs the expert offsets on the host (a blocking D2H), which a
    // CUDA graph capture cannot hold.
    if gpu.stream_is_capturing(stream) {
        return refuse("the stream is being captured into a CUDA graph");
    }

    let mut raw = vec![0u8; (cfg.num_experts + 1) * 4];
    gpu.copy_d2h_on_stream(ws.expert_offsets(), &mut raw, stream)?;
    let offsets: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if offsets.last().copied().unwrap_or(-1) != te as i32 {
        return refuse("expert_offsets does not end at the routed row count");
    }

    if let Err(e) = cache.ensure(cfg, w, stream) {
        return refuse(&format!("SFB swizzle failed: {e:#}"));
    }
    let tables = MoeTables::build(
        cfg.num_experts,
        cfg.local_expert_range(),
        &w.experts,
        &cache.layout(),
        dynamic_scale_on(),
    );
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let (gu, dn) = tables.static_scale_counts();
            tracing::warn!(
                "GLM routed-MoE prefill CUTLASS W4A4: first layer uses a static input_scale for \
                 {gu}/{n} local gate/up and {dn}/{n} local down projections (the rest: dynamic \
                 per-tensor amax / 2688; METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE {}); down projection \
                 {}",
                if dynamic_scale_on() {
                    "=1: all dynamic"
                } else {
                    "off"
                },
                down_mode_text(down_w4a16_on()),
                n = cfg.local_experts
            );
        });
    }
    let args = RunArgs {
        swiglu: k.swiglu,
        x,
        sorted_token_ids: ws.sorted_token_ids(),
        a_gate: ws.a_gate(),
        a_up: ws.a_up(),
        a_act: ws.a_act(),
        expert_out: ws.expert_out(),
        te,
        hidden: cfg.hidden,
        moe_intermediate: cfg.moe_intermediate,
        swiglu_limit: cfg.swiglu_limit,
        offsets: &offsets,
        tables: &tables,
        skip_down: down_w4a16_on(),
        swiglu_rows: swiglu_span(
            &offsets,
            cfg.local_expert_range(),
            te,
            swiglu_local_rows_on(),
        ),
    };
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        if swiglu_local_rows_on() {
            ONCE.call_once(|| {
                tracing::warn!(
                    "GLM routed-MoE prefill CUTLASS W4A4: SwiGLU over local rows only \
                     (METRALE_GLM_MOE_SWIGLU_LOCAL_ROWS=1): first layer {}..{} of {te} routed rows",
                    args.swiglu_rows.start,
                    args.swiglu_rows.end
                );
            });
        }
    }
    match run(gpu, &args, stream) {
        Ok(()) if args.skip_down => Ok(Outcome::DownPending),
        Ok(()) => Ok(Outcome::Done),
        Err(e) => refuse(&format!("{e:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swiglu_span_covers_local_experts_or_everything() {
        // 4 experts, 2 per rank: rows 0..3 e0, 3..3 e1, 3..7 e2, 7..10 e3.
        let off = [0, 3, 3, 7, 10];
        assert_eq!(swiglu_span(&off, 0..2, 10, false), 0..10);
        assert_eq!(swiglu_span(&off, 0..2, 10, true), 0..3);
        assert_eq!(swiglu_span(&off, 2..4, 10, true), 3..10);
        assert_eq!(swiglu_span(&off, 1..2, 10, true), 3..3);
        // A short or bad table never reaches past `te` or runs backwards.
        assert_eq!(swiglu_span(&off[..3], 2..4, 10, true), 3..10);
        assert_eq!(swiglu_span(&[0, 9, 4], 1..2, 10, true), 9..9);
    }
    use crate::glm5next_mlp::weights::Nvfp4Proj;

    fn proj(packed: u64, s2: f32, is: f32) -> Nvfp4Proj {
        Nvfp4Proj {
            packed: DevicePtr(packed),
            scale: DevicePtr(packed + 1),
            scale_2: s2,
            input_scale: is,
        }
    }

    #[test]
    fn glm53_shapes_pass_the_contract_and_odd_ones_do_not() {
        assert!(shape_contract(4096, 2048, 288, 144).is_ok());
        assert!(shape_contract(4096, 2048, 288, 288).is_ok());
        assert!(shape_contract(4000, 2048, 288, 144).is_err());
        assert!(shape_contract(4096, 2000, 288, 144).is_err());
        assert!(shape_contract(4096, 2048, 288, 0).is_err());
        assert!(shape_contract(4096, 2048, 8, 16).is_err());
        assert!(shape_contract(0, 2048, 288, 144).is_err());
    }

    #[test]
    fn unusable_input_scales_become_dynamic_and_gate_up_takes_the_max() {
        for bad in [0.0f32, -1e-3, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(sanitize_input_scale(bad), 0.0, "{bad}");
        }
        assert_eq!(sanitize_input_scale(3e-4), 3e-4);
        assert_eq!(gate_up_global_scale(1e-4, 3e-4), 3e-4);
        assert_eq!(gate_up_global_scale(3e-4, 1e-4), 3e-4);
        assert_eq!(gate_up_global_scale(f32::NAN, 2e-4), 2e-4);
        assert_eq!(gate_up_global_scale(0.0, 0.0), 0.0);
    }

    #[test]
    fn sfb_layout_regions_do_not_overlap_and_match_glm53_bytes() {
        let l = SfbLayout::new(0x1000, 144, 4096, 2048);
        assert_eq!(l.gu_stride, 2048 * 256);
        assert_eq!(l.down_stride, 4096 * 128);
        assert_eq!(l.up_off, l.gate_off + 144 * l.gu_stride);
        assert_eq!(l.down_off, l.up_off + 144 * l.gu_stride);
        assert_eq!(l.total_bytes(144), 144 * (2 * 524_288 + 524_288));
        assert_eq!(SfbCache::bytes(144, 4096, 2048), 226_492_416);
    }

    /// 2026-10-03: Local ids get their weights, cache SFBs and scales; remote ids stay null
    /// (no CUTLASS group) and dynamic.
    #[test]
    fn tables_index_by_global_id_and_leave_remote_ids_null() {
        let experts: Vec<Glm5NextExpertWeights> = (0..3u64)
            .map(|s| Glm5NextExpertWeights {
                gate_proj: proj(0x10_000 + s * 0x100, 0.5, 1e-4 * (s + 1) as f32),
                up_proj: proj(0x20_000 + s * 0x100, 0.25, 2e-4),
                down_proj: proj(0x30_000 + s * 0x100, 0.125, if s == 1 { 0.0 } else { 5e-4 }),
            })
            .collect();
        let l = SfbLayout::new(0x9000_0000, 3, 256, 128);
        let t = MoeTables::build(8, 3..6, &experts, &l, false);
        for id in [0usize, 1, 2, 6, 7] {
            assert_eq!(t.gate.packed[id], 0);
            assert_eq!(t.gate.sfb[id], 0);
            assert_eq!(t.down.sfb[id], 0);
            assert_eq!(t.gate_up_gs[id], 0.0);
        }
        for s in 0..3usize {
            let id = 3 + s;
            assert_eq!(t.gate.packed[id], 0x10_000 + s as u64 * 0x100);
            assert_eq!(t.up.packed[id], 0x20_000 + s as u64 * 0x100);
            assert_eq!(t.gate.sfb[id], l.base + s as u64 * l.gu_stride);
            assert_eq!(t.up.sfb[id], l.base + l.up_off + s as u64 * l.gu_stride);
            assert_eq!(
                t.down.sfb[id],
                l.base + l.down_off + s as u64 * l.down_stride
            );
            assert_eq!(t.gate.scale2[id], 0.5);
            assert_eq!(t.down.scale2[id], 0.125);
            assert_eq!(
                t.gate_up_gs[id],
                gate_up_global_scale(1e-4 * (s + 1) as f32, 2e-4)
            );
        }
        assert_eq!(t.down_gs[4], 0.0);
        assert_eq!(t.down_gs[5], 5e-4);
        assert_eq!(t.static_scale_counts(), (3, 2));
    }

    /// 2026-10-04: `METRALE_GLM_MOE_W4A4_DYNAMIC_SCALE=1`: every global scale is 0.0 (dynamic),
    /// while weights, SFBs and weight scales are exactly what the static build produces.
    #[test]
    fn dynamic_scale_zeroes_every_activation_scale_and_nothing_else() {
        let experts: Vec<Glm5NextExpertWeights> = (0..3u64)
            .map(|s| Glm5NextExpertWeights {
                gate_proj: proj(0x10_000 + s * 0x100, 0.5, 1e-4 * (s + 1) as f32),
                up_proj: proj(0x20_000 + s * 0x100, 0.25, 2e-4),
                down_proj: proj(0x30_000 + s * 0x100, 0.125, 5e-4),
            })
            .collect();
        let l = SfbLayout::new(0x9000_0000, 3, 256, 128);
        let stat = MoeTables::build(8, 3..6, &experts, &l, false);
        let dynm = MoeTables::build(8, 3..6, &experts, &l, true);
        assert_eq!(stat.static_scale_counts(), (3, 3));
        assert_eq!(dynm.static_scale_counts(), (0, 0));
        assert_eq!(dynm.gate_up_gs, vec![0.0; 8]);
        assert_eq!(dynm.down_gs, vec![0.0; 8]);
        // Only the two activation-scale vectors differ.
        assert_eq!(
            MoeTables {
                gate_up_gs: vec![],
                down_gs: vec![],
                ..stat
            },
            MoeTables {
                gate_up_gs: vec![],
                down_gs: vec![],
                ..dynm
            }
        );
    }
}
