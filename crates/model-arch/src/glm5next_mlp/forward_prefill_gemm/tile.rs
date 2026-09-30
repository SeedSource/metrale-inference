// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The grouped-GEMM tiles the GLM-5.3 routed-MoE prefill can launch, and the env
//! levers of that path: tile, width floor, exact grid height, and on/off.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Each lever is read once per process.
//! - `GEMM_TILES[0]` is the base kernel, `moe_w4a16_grouped_gemm_ptrtable`.

/// 2026-09-25: `M_TILE` of the base kernel (`#define M_TILE 64` in
/// `kernels/gb10/common/moe_w4a16_grouped_gemm.cu`). Tests only; the dispatch uses
/// `GemmTile::m_tile`.
#[cfg(test)]
pub(crate) const GROUPED_M_TILE: usize = 64;

/// 2026-09-25: One grouped-GEMM kernel entry point and the launch geometry it is built for.
/// `METRALE_GLM_MOE_GEMM_TILE` picks one (`gemm_tile`).
///
/// Measured 2026-09-22 with `examples/glm5next_moe_grouped_tile_bench` on one GB10, at the
/// EP=2 production shape and 256 rows: `bt_m16_k128` moved 156.2 GB/s on gate/up and 161.8 on
/// down, against 43.2 and 46.1 for `base`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmTile {
    /// 2026-09-25: Kernel entry point in the `moe_w4a16` module.
    pub name: &'static str,
    /// 2026-09-25: Rows per CTA, the kernel's `M_TILE`; the grid height is counted in it.
    pub m_tile: usize,
    /// 2026-09-25: Output columns per CTA; grid.x is `ceil(n_out / n_tile)`.
    pub n_tile: u32,
    /// 2026-09-25: Threads per block, 32 per warp.
    pub threads: u32,
    /// 2026-09-29: The kernel is an `_mfast` variant: grid x counts M tiles and grid y N tiles
    /// (`grid_dims`), so the CTAs of one (expert, N tile) launch adjacently.
    pub m_fast: bool,
}

impl GemmTile {
    /// 2026-09-29: Launch grid for `n_out` output columns, `max_m_tiles` M tiles and
    /// `num_experts` experts: `(n tiles, m tiles, experts)`, or `(m tiles, n tiles, experts)`
    /// for an `m_fast` tile.
    pub(crate) fn grid_dims(&self, n_out: usize, max_m_tiles: u32, num_experts: usize) -> [u32; 3] {
        let n_tiles = (n_out as u32).div_ceil(self.n_tile);
        if self.m_fast {
            [max_m_tiles, n_tiles, num_experts as u32]
        } else {
            [n_tiles, max_m_tiles, num_experts as u32]
        }
    }
}

/// 2026-09-25: The tile used when `METRALE_GLM_MOE_GEMM_TILE` is unset or names no known tile.
pub(crate) const DEFAULT_GEMM_TILE: GemmTile = GemmTile {
    name: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128",
    m_tile: 16,
    n_tile: 64,
    threads: 128,
    m_fast: false,
};

/// 2026-09-25: Every tile `select_gemm_tile` accepts.
pub(crate) const GEMM_TILES: &[GemmTile] = &[
    // 2026-09-25: Index 0 is the base tile: `select_gemm_tile("base")` returns it, and
    // `Glm5NextMlpKernels::resolve` falls back to it when a tile is missing from the PTX.
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
        m_fast: false,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_k32",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
        m_fast: false,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_k64",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
        m_fast: false,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_m16_k64",
        m_tile: 16,
        n_tile: 64,
        threads: 128,
        m_fast: false,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_k128",
        m_tile: 16,
        n_tile: 64,
        threads: 128,
        m_fast: false,
    },
    DEFAULT_GEMM_TILE,
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_n128_k128",
        m_tile: 16,
        n_tile: 128,
        threads: 256,
        m_fast: false,
    },
    // 2026-09-29: Whole-chunk prefill M1. Each computes every element with the same MMAs in the
    // same k order as `bt_m16_k128` (see the kernel file), so the output is byte-identical to
    // it (INFERRED; `glm5next_moe_grouped_prefill_microtest` asserts it). `bt_k128` is the M64
    // tile; `_mfast` tiles put the M tile in grid x for L2 reuse of one expert's weight columns.
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128_mfast",
        m_tile: 16,
        n_tile: 64,
        threads: 128,
        m_fast: true,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_k128",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
        m_fast: false,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m64_k128_mfast",
        m_tile: 64,
        n_tile: 64,
        threads: 128,
        m_fast: true,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: false,
    },
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_mfast",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: true,
    },    // 2026-09-30: `bt_m128_k64` with coalesced 16-byte A staging (VEC_A): the same smem_A tile
    // bit for bit, so the same MMAs in the same k order (met-moediag2: A staging is the
    // bt_m128_k64 limiter). Byte-identical to `bt_m16_k128` (tile bench + prefill microtest).
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: false,
    },    // 2026-09-30: VA2 = VEC_A plus a cp.async double-buffered A pipeline: the same smem_A bits
    // and the same MMAs in the same k order as `bt_m128_k64`, byte-identical to `bt_m16_k128`
    // (met-va2bench 16/16; 17-40 % below `_va` at production shapes).
    GemmTile {
        name: "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va2",
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: false,
    },
];

/// 2026-09-25: Resolve a tile name: `base` or an empty value is `GEMM_TILES[0]`; otherwise the
/// full entry-point name, or its suffix after `moe_w4a16_grouped_gemm_ptrtable_`. `None` for any
/// other value.
pub(crate) fn select_gemm_tile(v: &str) -> Option<GemmTile> {
    let v = v.trim();
    if v.eq_ignore_ascii_case("base") || v.is_empty() {
        return Some(GEMM_TILES[0]);
    }
    GEMM_TILES
        .iter()
        .copied()
        .find(|t| t.name == v || t.name.strip_prefix("moe_w4a16_grouped_gemm_ptrtable_") == Some(v))
}

/// 2026-09-25: The tile for this process, from `METRALE_GLM_MOE_GEMM_TILE`, read once: unset
/// gives `DEFAULT_GEMM_TILE`, and an unknown name logs an error and gives `DEFAULT_GEMM_TILE`.
pub(crate) fn gemm_tile() -> GemmTile {
    static T: std::sync::OnceLock<GemmTile> = std::sync::OnceLock::new();
    *T.get_or_init(|| match std::env::var("METRALE_GLM_MOE_GEMM_TILE") {
        Ok(v) => match select_gemm_tile(&v) {
            Some(t) => {
                tracing::warn!(
                    "GLM routed-MoE prefill grouped GEMM tile overridden to `{}` \
                     (M_TILE {}, N_TILE {}, {} threads); default is `{}`",
                    t.name,
                    t.m_tile,
                    t.n_tile,
                    t.threads,
                    DEFAULT_GEMM_TILE.name
                );
                t
            }
            None => {
                tracing::error!(
                    "METRALE_GLM_MOE_GEMM_TILE=`{v}` is not a known tile — using the default \
                     `{}`. Known: {:?}",
                    DEFAULT_GEMM_TILE.name,
                    GEMM_TILES.iter().map(|t| t.name).collect::<Vec<_>>()
                );
                DEFAULT_GEMM_TILE
            }
        },
        Err(_) => DEFAULT_GEMM_TILE,
    })
}

/// 2026-09-25: Whether the routed-expert prefill may take the grouped GEMM: yes unless
/// `METRALE_GLM_MOE_PREFILL_GEMM=0`. Read once.
pub(crate) fn prefill_gemm_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let off = std::env::var("METRALE_GLM_MOE_PREFILL_GEMM").as_deref() == Ok("0");
        if off {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM DISABLED \
                 (METRALE_GLM_MOE_PREFILL_GEMM=0) — row-batched GEMV path"
            );
        }
        !off
    })
}

/// 2026-09-25: Narrowest row group the grouped GEMM takes: `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS`,
/// or `DEFAULT_GEMM_MIN_ROWS` when unset or not a number; at least 1. Read once.
pub(crate) fn prefill_gemm_min_rows() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let m = std::env::var("METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_GEMM_MIN_ROWS)
            .max(1);
        if m != DEFAULT_GEMM_MIN_ROWS {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM width floor overridden to {m} rows \
                 (default {DEFAULT_GEMM_MIN_ROWS}; MEASURED 0.40x at 16, 0.74x at 64, 1.09x at 128, 1.56x at 256)"
            );
        }
        m
    })
}

/// 2026-09-25: The default of `prefill_gemm_min_rows`.
pub(crate) const DEFAULT_GEMM_MIN_ROWS: usize = 128;

/// 2026-09-25: Whether the grid height comes from the real expert histogram, one
/// stream-synchronising read of `expert_offsets` per call, rather than the worst case: yes
/// unless `METRALE_GLM_MOE_PREFILL_GEMM_EXACT_TILES=0`. Read once.
pub(crate) fn prefill_gemm_exact_tiles() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let off = std::env::var("METRALE_GLM_MOE_PREFILL_GEMM_EXACT_TILES").as_deref() == Ok("0");
        if off {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM: WORST-CASE grid height \
                 (METRALE_GLM_MOE_PREFILL_GEMM_EXACT_TILES=0) — no per-layer expert_offsets D2H"
            );
        }
        !off
    })
}

/// 2026-09-30: Whether the routed-expert prefill gathers `x` into expert-sorted order once,
/// into the workspace's own scratch buffer (`moe_permute_tokens`), then runs gate and up with
/// `sorted_token_ids` NULL, instead of each of those two GEMMs gathering its own A rows through
/// `sorted_token_ids`: `METRALE_GLM_MOE_PREFILL_PERMUTE=1`. Byte-identical by construction — the
/// permuted buffer holds exactly the BF16 bits `A[sorted_token_ids[row]]` would have read, so a
/// GEMM tile computes the same `mma.sync` sums in the same order either way
/// (`glm5next_moe_prefill_permute_microtest` asserts it row-for-row at rows in
/// {1, 17, 256, 1000, 2048, 4096}). MEASURED 2026-09-30 with
/// `examples/glm5next_moe_grouped_tile_bench` (`GLM_TILE_BENCH_GATHER_X=1`; see
/// `~/lazarus/spark-bench/runs/metrale/moeprobe/RESULT.md`): the embedded `sorted_token_ids`
/// gather costs gate/up ~10% at the production window (2048 rows) and 44% at 4096 rows, so
/// paying the gather once instead of twice is worth it in principle — but the permute pass's
/// own GPU cost, and the net effect, are UNMEASURED (no GPU in this change). Off by default
/// until measured. Read once.
pub(crate) fn prefill_gemm_permute() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_MOE_PREFILL_PERMUTE").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM: GATHER-ONCE gate/up A \
                 (METRALE_GLM_MOE_PREFILL_PERMUTE=1) — permutes x into expert-sorted order once \
                 per layer via moe_permute_tokens, then runs gate and up with sorted_token_ids \
                 NULL; UNMEASURED on GPU as of 2026-09-30"
            );
        }
        on
    })
}

/// 2026-09-25: Grid height of the grouped GEMM: tiles of `m_tile` rows that the busiest expert
/// in the host copy of `expert_offsets` needs, at least 1 and at most `worst_case`.
pub(crate) fn max_m_tiles_from_offsets(offsets: &[i32], worst_case: u32, m_tile: usize) -> u32 {
    let mut prev = 0i32;
    let mut max_rows = 0i32;
    for &cur in offsets.iter().skip(1) {
        max_rows = max_rows.max(cur - prev);
        prev = cur;
    }
    (max_rows.max(0) as u32)
        .div_ceil(m_tile.max(1) as u32)
        .max(1)
        .min(worst_case.max(1))
}
