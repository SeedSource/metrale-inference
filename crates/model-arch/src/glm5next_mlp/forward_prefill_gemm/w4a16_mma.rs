// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1`: the GLM-5.3 routed-MoE prefill through
//! `kernels/gb10/common/moe_w4a16_prefill_mma.cu`. After the expert sort, three launches replace
//! the production gate GEMM, up GEMM, SwiGLU and down GEMM:
//!   1. `moe_w4a16_prefill_tile_list`: a compact (expert, M tile) list over the LOCAL experts;
//!   2. `moe_w4a16_prefill_mma_gateup_silu_<tile>`: gate and up for the same 64 columns in one
//!      CTA, GLM's clamped SwiGLU in the epilogue, the activation written to `ws.a_act`;
//!   3. `moe_w4a16_prefill_mma_down_<tile>`: down into `ws.expert_out`.
//! The weights stay packed in shared memory (cp.async ring) and are dequantised in registers.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Lever off (default): `PrefillMmaKernels::resolve` looks nothing up, `usable` is false, and
//!   `forward_moe_grouped_prefill` runs exactly as before.
//! - Lever on: the outputs land in the same buffers and expert-sorted row order as the
//!   production path, so `glm5next_moe_combine_indexed` reads them unchanged; rows of a remote
//!   expert are not written (the caller pre-zeroes `expert_out`).
//! - Numerics: NOT byte-identical to `moe_w4a16_grouped_gemm_ptrtable_*` (the k order inside
//!   each MMA differs). The dequantised BF16 weights and the SwiGLU expression are the same.
//! - The tile list lives in `ws.prefill_tile_scratch()` (`u_eid`); `usable` checks it fits.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::Glm5NextMlpConfig;
use super::super::forward::Glm5NextMlpWorkspace;
use super::super::weights::Glm5NextMoeWeights;

/// 2026-10-01: Module of `moe_w4a16_prefill_mma.cu` (not in `common/KERNEL.toml`'s
/// `[modules]`, so its file stem).
pub(crate) const PREFILL_MMA_MODULE: &str = "moe_w4a16_prefill_mma";
/// 2026-10-01: The kernel's `PMMA_KS`: K must be a multiple of it.
pub(crate) const PREFILL_MMA_K_STEP: usize = 64;
/// 2026-10-01: Activation columns per gate/up CTA (64 gate + 64 up weight rows).
pub(crate) const PREFILL_MMA_GATEUP_COLS: usize = 64;
/// 2026-10-01: Output columns per down CTA.
pub(crate) const PREFILL_MMA_DOWN_COLS: usize = 128;
/// 2026-10-01: The kernel's `PMMA_MAX_EXPERTS` (tile-list shared array).
pub(crate) const PREFILL_MMA_MAX_EXPERTS: usize = 1024;
/// 2026-10-01: GB10's opt-in dynamic shared memory per block (99 KB, kernel-engineering ledger).
pub(crate) const PREFILL_MMA_SMEM_LIMIT: u32 = 101_376;

/// 2026-10-01: One instantiation of the kernel template (`PMMA_GATEUP` / `PMMA_DOWN`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillMmaTile {
    /// 2026-10-01: Entry-point suffix: `moe_w4a16_prefill_mma_{gateup_silu,down}_<suffix>`.
    pub suffix: &'static str,
    /// 2026-10-01: Rows per CTA (`MT`); the tile list counts M tiles in it.
    pub m_tile: usize,
    /// 2026-10-01: `(MT / 64) * 128`, the kernel's `__launch_bounds__`.
    pub threads: u32,
    /// 2026-10-01: cp.async ring depth (`STAGES`).
    pub stages: u32,
}

impl PrefillMmaTile {
    /// 2026-10-01: Dynamic shared memory the kernel indexes: `STAGES * (MT * 128 + 4608)`.
    pub(crate) const fn smem_bytes(&self) -> u32 {
        self.stages * (self.m_tile as u32 * 128 + 4608)
    }
    pub(crate) fn gateup_name(&self) -> String {
        format!("moe_w4a16_prefill_mma_gateup_silu_{}", self.suffix)
    }
    pub(crate) fn down_name(&self) -> String {
        format!("moe_w4a16_prefill_mma_down_{}", self.suffix)
    }
}

/// 2026-10-01: The tile the lever runs. `PREFILL_MMA_M64` is built and benched
/// (`examples/glm5next_moe_prefill_w4a16_mma_microtest.rs`) but not selected.
pub(crate) const PREFILL_MMA_M128: PrefillMmaTile = PrefillMmaTile {
    suffix: "m128",
    m_tile: 128,
    threads: 256,
    stages: 4,
};
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const PREFILL_MMA_M64: PrefillMmaTile = PrefillMmaTile {
    suffix: "m64",
    m_tile: 64,
    threads: 128,
    stages: 3,
};

/// 2026-10-01: The three entry points, all `KernelHandle(0)` when the lever is off or the PTX
/// lacks one of them.
#[derive(Debug, Clone, Copy)]
pub struct PrefillMmaKernels {
    pub(crate) tile_list: KernelHandle,
    pub(crate) gateup: KernelHandle,
    pub(crate) down: KernelHandle,
    pub(crate) tile: PrefillMmaTile,
}

impl PrefillMmaKernels {
    /// 2026-10-01: Looks the kernels up only when `METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1`.
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Self {
        let tile = PREFILL_MMA_M128;
        let none = KernelHandle(0);
        if !super::tile::prefill_gemm_grouped_w4a16() {
            return Self {
                tile_list: none,
                gateup: none,
                down: none,
                tile,
            };
        }
        let get =
            |name: &str| metrale_model_layers::layers::try_kernel(gpu, PREFILL_MMA_MODULE, name);
        let k = Self {
            tile_list: get("moe_w4a16_prefill_tile_list"),
            gateup: get(&tile.gateup_name()),
            down: get(&tile.down_name()),
            tile,
        };
        if !k.ready() {
            tracing::warn!(
                "METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1 but `{PREFILL_MMA_MODULE}` lacks an \
                 entry point (tile list {}, gateup {}, down {}) — production grouped GEMM",
                k.tile_list.0 != 0,
                k.gateup.0 != 0,
                k.down.0 != 0
            );
        }
        k
    }

    pub(crate) fn ready(&self) -> bool {
        self.tile_list.0 != 0 && self.gateup.0 != 0 && self.down.0 != 0
    }
}

/// 2026-10-01: Tile-list capacity for `te` routed rows: every local expert adds at most one
/// partial tile, so `ceil(te / m_tile) + local_experts` always holds the real count.
pub(crate) fn tile_cap(te: usize, local_experts: usize, tile: PrefillMmaTile) -> usize {
    te.div_ceil(tile.m_tile.max(1)) + local_experts
}

/// 2026-10-01: The shape contract of `moe_w4a16_prefill_mma.cu`, host side. Pure, so tests can
/// probe it. `x_addr` is the gate/up A buffer, `act_addr` the down A buffer.
pub(crate) fn shape_ok(
    cfg: &Glm5NextMlpConfig,
    te: usize,
    scratch_entries: usize,
    x_addr: u64,
    act_addr: u64,
    tile: PrefillMmaTile,
) -> bool {
    let cap = tile_cap(te, cfg.local_experts, tile);
    cfg.hidden.is_multiple_of(PREFILL_MMA_K_STEP)
        && cfg.moe_intermediate.is_multiple_of(PREFILL_MMA_K_STEP)
        && cfg.moe_intermediate.is_multiple_of(PREFILL_MMA_GATEUP_COLS)
        && cfg.hidden.is_multiple_of(PREFILL_MMA_DOWN_COLS)
        && cfg.num_experts <= PREFILL_MMA_MAX_EXPERTS
        && x_addr.is_multiple_of(16)
        && act_addr.is_multiple_of(16)
        && cap < 65_536
        && te.div_ceil(tile.m_tile.max(1)) < 65_536
        && cap < scratch_entries
        && tile.smem_bytes() <= PREFILL_MMA_SMEM_LIMIT
}

/// 2026-10-01: Whether `forward_moe_grouped_prefill` takes this path for `te` routed rows: the
/// lever's kernels resolved, the shape contract holds, and every local expert's packed weights
/// are 4-byte aligned (the kernel's minimum; 16-byte alignment takes its fast copy).
pub(crate) fn usable(
    k: &PrefillMmaKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    te: usize,
    ws: &Glm5NextMlpWorkspace,
) -> bool {
    if !k.ready() {
        return false;
    }
    let ok = shape_ok(cfg, te, ws.max_total_expanded(), x.0, ws.a_act().0, k.tile)
        && w.experts.iter().all(|e| {
            [e.gate_proj, e.up_proj, e.down_proj]
                .iter()
                .all(|p| p.packed.0.is_multiple_of(4))
        });
    if !ok {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1: shape/alignment contract not met \
                 (hidden {}, moe_intermediate {}, experts {}, {te} routed rows) — production \
                 grouped GEMM for such calls",
                cfg.hidden,
                cfg.moe_intermediate,
                cfg.num_experts
            );
        });
    }
    ok
}

/// 2026-10-01: Tile list, fused gate/up/SwiGLU into `ws.a_act`, down into `ws.expert_out`.
/// `stid` is `ws.sorted_token_ids()`: gate/up gather their A rows of `x` through it; down reads
/// `ws.a_act` in sorted order. Call only after `moe_sort_by_expert` and when `usable` holds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    gpu: &dyn GpuBackend,
    k: &PrefillMmaKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    stid: DevicePtr,
    te: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    let tile = k.tile;
    let cap = tile_cap(te, cfg.local_experts, tile) as u32;
    let tiles = ws.prefill_tile_scratch();
    let mi = cfg.moe_intermediate;

    KernelLaunch::new(gpu, k.tile_list)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ws.expert_offsets())
        .arg_ptr(w.ptrs.gate.packed_ptrs)
        .arg_ptr(tiles)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(tile.m_tile as u32)
        .arg_u32(cap)
        .launch(stream)?;

    KernelLaunch::new(gpu, k.gateup)
        .grid([(mi / PREFILL_MMA_GATEUP_COLS) as u32, cap, 1])
        .block([tile.threads, 1, 1])
        .shared_mem(tile.smem_bytes())
        .arg_ptr(x)
        .arg_ptr(stid)
        .arg_ptr(w.ptrs.gate.packed_ptrs)
        .arg_ptr(w.ptrs.gate.scale_ptrs)
        .arg_ptr(w.ptrs.gate.scale2_vals)
        .arg_ptr(w.ptrs.up.packed_ptrs)
        .arg_ptr(w.ptrs.up.scale_ptrs)
        .arg_ptr(w.ptrs.up.scale2_vals)
        .arg_ptr(ws.a_act())
        .arg_ptr(ws.expert_offsets())
        .arg_ptr(tiles)
        .arg_u32(mi as u32)
        .arg_u32(cfg.hidden as u32)
        .arg_f32(cfg.swiglu_limit)
        .launch(stream)?;

    KernelLaunch::new(gpu, k.down)
        .grid([(cfg.hidden / PREFILL_MMA_DOWN_COLS) as u32, cap, 1])
        .block([tile.threads, 1, 1])
        .shared_mem(tile.smem_bytes())
        .arg_ptr(ws.a_act())
        .arg_ptr(DevicePtr(0))
        .arg_ptr(w.ptrs.down.packed_ptrs)
        .arg_ptr(w.ptrs.down.scale_ptrs)
        .arg_ptr(w.ptrs.down.scale2_vals)
        .arg_ptr(ws.expert_out())
        .arg_ptr(ws.expert_offsets())
        .arg_ptr(tiles)
        .arg_u32(cfg.hidden as u32)
        .arg_u32(mi as u32)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = include_str!("../../../../../kernels/gb10/common/moe_w4a16_prefill_mma.cu");

    /// 2026-10-01: Both tiles' entry points exist in the source, and the host's geometry matches
    /// the instantiation lines (MT, STAGES) and the kernel's smem formula.
    #[test]
    fn tiles_match_their_kernel_instantiations() {
        assert!(SRC.contains("extern \"C\" __global__ void moe_w4a16_prefill_tile_list("));
        for (t, line_gu, line_dn) in [
            (PREFILL_MMA_M128, "PMMA_GATEUP(m128, 128, 4)", "PMMA_DOWN(m128, 128, 4)"),
            (PREFILL_MMA_M64, "PMMA_GATEUP(m64, 64, 3)", "PMMA_DOWN(m64, 64, 3)"),
        ] {
            assert!(SRC.contains(line_gu) && SRC.contains(line_dn), "{t:?}");
            assert_eq!(t.threads as usize, (t.m_tile / 64) * 128, "{t:?}");
            assert!(t.smem_bytes() <= PREFILL_MMA_SMEM_LIMIT, "{t:?}");
            assert_eq!(t.gateup_name(), format!("moe_w4a16_prefill_mma_gateup_silu_{}", t.suffix));
        }
        assert_eq!(PREFILL_MMA_M128.smem_bytes(), 83_968);
        assert_eq!(PREFILL_MMA_M64.smem_bytes(), 38_400);
    }

    /// 2026-10-01: The capacity bound holds the exact tile count for skewed and uniform
    /// histograms (every expert local, the worst case for the `+ local_experts` term).
    #[test]
    fn tile_cap_never_undercounts() {
        for tile in [PREFILL_MMA_M128, PREFILL_MMA_M64] {
            for (te, experts) in [(1024usize, 288usize), (32_768, 144), (65_536, 288), (77, 8)] {
                let mut counts = vec![0usize; experts];
                let mut s = 0x1234_5678u64;
                for i in 0..te {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    // 2026-10-01: Half the rows on expert 0, the rest spread: a hot expert.
                    let e = if i % 2 == 0 { 0 } else { (s >> 33) as usize % experts };
                    counts[e] += 1;
                }
                let exact: usize = counts.iter().map(|c| c.div_ceil(tile.m_tile)).sum();
                assert!(exact <= tile_cap(te, experts, tile), "{tile:?} te={te}");
            }
        }
    }
}
