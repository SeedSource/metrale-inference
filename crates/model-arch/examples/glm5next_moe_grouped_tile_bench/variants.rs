// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The tile-geometry variants of `moe_w4a16_grouped_gemm_ptrtable` that the bench
//! times against the base kernel. Data only, no GPU calls.
//!
//! Owner: model-arch examples (GLM-5.3 routed MoE).
//! Invariants:
//! - Each entry's `m_tile`, `n_tile` and `threads` are the `MT`, `NTILE` and `WARPS * 32` of
//!   the kernel it names.

/// 2026-09-25: One kernel under test. `m_tile` sets the grid height and `n_tile` the grid
/// width; either value wrong is silent (dropped rows or unwritten columns), not an error.
pub(crate) struct Variant {
    pub name: &'static str,
    pub kernel: &'static str,
    pub m_tile: u32,
    pub n_tile: u32,
    pub threads: u32,
    /// 2026-09-29: `_mfast` kernel: grid x counts M tiles and grid y N tiles.
    pub m_fast: bool,
    /// 2026-09-29: Whole-chunk prefill M1 tile: its output must be byte-identical to
    /// `REFERENCE`'s, and the bench exits nonzero if it is not.
    pub must_match_ref: bool,
    /// 2026-09-30: A bench-only pipeline-stage isolator of `bt_m128_k64` (see `diag()`
    /// below). Excluded from every run unless `GLM_TILE_BENCH_DIAG=1`; reported as DIAG, not
    /// compared for byte identity, and never pushed to the M1 failure list.
    pub is_diag: bool,
}

/// 2026-09-29: The production default tile, the byte-identity reference of the M1 tiles.
pub(crate) const REFERENCE: &str = "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128";

const fn v(
    name: &'static str,
    kernel: &'static str,
    m_tile: u32,
    n_tile: u32,
    threads: u32,
) -> Variant {
    Variant {
        name,
        kernel,
        m_tile,
        n_tile,
        threads,
        m_fast: false,
        must_match_ref: false,
        is_diag: false,
    }
}

/// 2026-09-29: An M1 tile, checked byte for byte against `REFERENCE`.
const fn m1(
    name: &'static str,
    kernel: &'static str,
    m_tile: u32,
    threads: u32,
    m_fast: bool,
) -> Variant {
    Variant {
        name,
        kernel,
        m_tile,
        n_tile: 64,
        threads,
        m_fast,
        must_match_ref: true,
        is_diag: false,
    }
}

/// 2026-09-30: A bench-only diagnostic tile that isolates one pipeline stage of
/// `bt_m128_k64` (`moe_w4a16_grouped_core`'s DIAG template parameter: NODEQ, NOMMA, NOLOAD-W,
/// NOLOAD-A or NOLOAD-AW). Same MT=128, NTILE=64, KS=64, WARPS=8 grid/block/smem footprint as
/// `bt_m128_k64`, so its timing is directly comparable. Its output differs from every other
/// variant's by construction, so `must_match_ref` is false (never an M1 failure) and the
/// bench reports it as DIAG rather than comparing it for byte identity. Excluded from every
/// run unless `GLM_TILE_BENCH_DIAG=1`.
const fn diag(name: &'static str, kernel: &'static str) -> Variant {
    Variant {
        name,
        kernel,
        m_tile: 128,
        n_tile: 64,
        threads: 256,
        m_fast: false,
        must_match_ref: false,
        is_diag: true,
    }
}

pub(crate) const VARIANTS: &[Variant] = &[
    v(
        "base M64 N64 K16",
        "moe_w4a16_grouped_gemm_ptrtable",
        64,
        64,
        128,
    ),
    v(
        "     M64 N64 K32",
        "moe_w4a16_grouped_gemm_ptrtable_k32",
        64,
        64,
        128,
    ),
    v(
        "     M64 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_k64",
        64,
        64,
        128,
    ),
    v(
        "     M64 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_k128",
        64,
        64,
        128,
    ),
    v(
        "     M16 N64 K32",
        "moe_w4a16_grouped_gemm_ptrtable_m16_k32",
        16,
        64,
        128,
    ),
    v(
        "     M16 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_m16_k64",
        16,
        64,
        128,
    ),
    v(
        "     M16 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_m16_k128",
        16,
        64,
        128,
    ),
    v(
        "     M16 N64 K256",
        "moe_w4a16_grouped_gemm_ptrtable_m16_k256",
        16,
        64,
        128,
    ),
    v(
        "     M16 N128 K32",
        "moe_w4a16_grouped_gemm_ptrtable_m16_n128_k32",
        16,
        128,
        256,
    ),
    v(
        "     M16 N128 K64",
        "moe_w4a16_grouped_gemm_ptrtable_m16_n128_k64",
        16,
        128,
        256,
    ),
    v(
        "     M16 N128 K128",
        "moe_w4a16_grouped_gemm_ptrtable_m16_n128_k128",
        16,
        128,
        256,
    ),
    v(
        "kmaj M64 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_km_k64",
        64,
        64,
        128,
    ),
    v(
        "kmaj M16 N64 K32",
        "moe_w4a16_grouped_gemm_ptrtable_km_m16_k32",
        16,
        64,
        128,
    ),
    v(
        "kmaj M16 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_km_m16_k64",
        16,
        64,
        128,
    ),
    v(
        "kmaj M16 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_km_m16_k128",
        16,
        64,
        128,
    ),
    v(
        "kmaj M16 N128 K64",
        "moe_w4a16_grouped_gemm_ptrtable_km_m16_n128_k64",
        16,
        128,
        256,
    ),
    v(
        "kmaj M16 N128 K128",
        "moe_w4a16_grouped_gemm_ptrtable_km_m16_n128_k128",
        16,
        128,
        256,
    ),
    v(
        "alut M64 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_al_k64",
        64,
        64,
        128,
    ),
    v(
        "alut M16 N64 K32",
        "moe_w4a16_grouped_gemm_ptrtable_al_m16_k32",
        16,
        64,
        128,
    ),
    v(
        "alut M16 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_al_m16_k64",
        16,
        64,
        128,
    ),
    v(
        "alut M16 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_al_m16_k128",
        16,
        64,
        128,
    ),
    v(
        "alut M16 N128 K64",
        "moe_w4a16_grouped_gemm_ptrtable_al_m16_n128_k64",
        16,
        128,
        256,
    ),
    v(
        "al+km M64 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_k64",
        64,
        64,
        128,
    ),
    v(
        "al+km M64 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_k128",
        64,
        64,
        128,
    ),
    v(
        "al+km M16 N64 K32",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_k32",
        16,
        64,
        128,
    ),
    v(
        "al+km M16 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_k64",
        16,
        64,
        128,
    ),
    v(
        "al+km M16 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_k128",
        16,
        64,
        128,
    ),
    v(
        "al+km M16 N64 K256",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_k256",
        16,
        64,
        128,
    ),
    v(
        "al+km M16 N128 K64",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_n128_k64",
        16,
        128,
        256,
    ),
    v(
        "al+km M16 N128 K128",
        "moe_w4a16_grouped_gemm_ptrtable_alkm_m16_n128_k128",
        16,
        128,
        256,
    ),
    v(
        "BT M16 N64 K64",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k64",
        16,
        64,
        128,
    ),
    v(
        "BT M16 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128",
        16,
        64,
        128,
    ),
    v(
        "BT M16 N64 K256",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k256",
        16,
        64,
        128,
    ),
    v(
        "BT M16 N128 K128",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m16_n128_k128",
        16,
        128,
        256,
    ),
    m1(
        "BT M64 N64 K128",
        "moe_w4a16_grouped_gemm_ptrtable_bt_k128",
        64,
        128,
        false,
    ),
    m1(
        "BT M16 K128 mfast",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128_mfast",
        16,
        128,
        true,
    ),
    m1(
        "BT M64 K128 mfast",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m64_k128_mfast",
        64,
        128,
        true,
    ),
    m1(
        "BT M128 K64",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64",
        128,
        256,
        false,
    ),
    m1(
        "BT M128 K64 mfast",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_mfast",
        128,
        256,
        true,
    ),
    // 2026-09-30: `bt_m128_k64` with coalesced 16-byte A staging (VEC_A); same smem_A bits, so
    // it must stay byte-identical to REFERENCE.
    m1(
        "BT M128 K64 VA",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va",
        128,
        256,
        false,
    ),
    // 2026-09-30: VA with the A tile double-buffered and the next K-step prefetched by 16-byte
    // cp.async (VA2); each K-step's smem_A bits are the scalar path's, so it must stay
    // byte-identical to REFERENCE.
    m1(
        "BT M128 K64 VA2",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_va2",
        128,
        256,
        false,
    ),
    // 2026-09-30: Bench-only pipeline-stage isolators of `bt_m128_k64`, gated behind
    // GLM_TILE_BENCH_DIAG=1 (see `diag()` and `moe_w4a16_grouped_core`'s DIAG doc comment in
    // moe_w4a16_grouped_gemm.cu). Never used in production.
    diag(
        "DIAG bt_m128_k64 NODEQ",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_diag_nodeq",
    ),
    diag(
        "DIAG bt_m128_k64 NOMMA",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_diag_nomma",
    ),
    diag(
        "DIAG bt_m128_k64 NOLOAD-W",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_diag_noload",
    ),
    diag(
        "DIAG bt_m128_k64 NOLOAD-A",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_diag_noloada",
    ),
    diag(
        "DIAG bt_m128_k64 NOLOAD-AW",
        "moe_w4a16_grouped_gemm_ptrtable_bt_m128_k64_diag_noloadaw",
    ),
];
