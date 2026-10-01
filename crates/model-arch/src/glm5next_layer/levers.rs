// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash layer launch levers: the prefill sub-chunk width, cuBLASLt for
//! wide projections, and the batched DSA indexer query.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Each function reads its environment variable once per process and caches the result.

/// 2026-09-25: Default tokens per batched prefill sub-chunk, the width `Glm5NextLayer::prefill`
/// hands `Glm5NextLayer::forward_k` (overridable through [`prefill_rows`]).
///
/// 16 is `DENSE_GEMV_BATCHM_MAX_M`, the widest M that `ops::dense_mm_bf16` sends to the batched
/// GEMV (`dense_gemv_bf16_batchm`), whose rows carry the same bits as the M = 1 GEMV. A wider
/// sub-chunk moves the dense projections to cuBLASLt or the tile GEMM, which accumulate in a
/// different order. Measured 2026-09-02: the 12 GLM prefill projection shapes cost 1.36-1.98x less
/// per token at M = 16 than at M = 8 in 11 of 12 (N4096 K128: 0.77x).
///
/// The routed MoE does not follow the width: `glm5next_mlp::forward::forward_moe` splits a wider
/// row group into even sub-groups of at most `row_batch_max()`, whose default and ceiling is
/// `MOE_ROW_BATCH_MAX_ROWS` (8).
pub(crate) const PREFILL_ROWS: usize = 16;

/// 2026-09-25: Send GLM projections with M above `DENSE_GEMV_BATCHM_MAX_M` to cuBLASLt BF16
/// instead of the tile GEMM. Read by the KDA, DSA and MLP blocks, always behind a
/// `> DENSE_GEMV_BATCHM_MAX_M` row test, so the M = 1 GEMV and the batched GEMV are never
/// replaced. On unless `METRALE_GLM_CUBLAS_PROJ=0`.
pub(crate) fn cublas_wide_proj() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_CUBLAS_PROJ").as_deref() != Ok("0");
        tracing::warn!(
            "METRALE_GLM_CUBLAS_PROJ: wide GLM projections (M > {}) use {}",
            metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M,
            if on {
                "cuBLASLt BF16"
            } else {
                "the scalar tile GEMM"
            }
        );
        on
    })
}

/// 2026-09-25: Compute the DSA indexer `wq_b` projection for all prefill rows in one cuBLASLt
/// GEMM instead of one M = 1 GEMV per row. Off unless `METRALE_GLM_DSA_BATCH_QIDX=1`.
///
/// Prefill only: it is read in `select_rows_batched`, which runs only when
/// `batch_select_enabled` holds (`is_prefill && k > 1`). Decode and the speculative verify keep
/// the per-row GEMV. The GEMM accumulates in a different order from the GEMV.
pub(crate) fn dsa_batch_qidx() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DSA_BATCH_QIDX").as_deref() == Ok("1");
        tracing::warn!(
            "METRALE_GLM_DSA_BATCH_QIDX: prefill DSA indexer q uses {}",
            if on {
                "ONE batched cuBLASLt GEMM"
            } else {
                "one M=1 GEMV per row"
            }
        );
        on
    })
}

/// 2026-10-01: `METRALE_GLM_DSA_ROW_BATCH=1` runs the per-row part of a prefill sub-chunk's DSA
/// block (`Glm5NextDsaLayer::decode_k`) for all rows at once: one metadata upload from pinned
/// staging, one latent-write launch, the indexer projections through the batched GEMVs and
/// one `k_norm` launch. Bit-identical to the per-row walk by construction
/// (`glm5next_dsa/layer/row_batch.rs`, `docs/dsa-rowbatch-NOTES.md`). Off unless set to `1`;
/// read once. Prefill only: it applies only where `batch_select_enabled` holds.
pub(crate) fn dsa_row_batch() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_DSA_ROW_BATCH").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_ROW_BATCH=1 - prefill DSA latent/indexer writes run once per \
                 sub-chunk instead of once per row"
            );
        }
        on
    })
}

/// 2026-09-25: `PREFILL_ROWS`, overridable at launch with `METRALE_GLM_PREFILL_ROWS` (values
/// below 1 or unparsable are ignored). `1` selects the per-token walk: `Glm5NextLayer::prefill`
/// takes the batched sub-chunk path only when `rows > 1`. A width above
/// `DENSE_GEMV_BATCHM_MAX_M` changes the projection numerics as well as the speed.
pub fn prefill_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let r = std::env::var("METRALE_GLM_PREFILL_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|r| *r >= 1)
            .unwrap_or(PREFILL_ROWS);
        if r != PREFILL_ROWS {
            tracing::warn!("GLM prefill sub-chunk overridden to {r} rows (default {PREFILL_ROWS})");
        }
        r
    })
}

/// 2026-09-29: Ceiling of the staged prefill's FFN window (`prefill_rows_ffn`), in rows.
pub const PREFILL_ROWS_FFN_MAX: usize = 4096;

/// 2026-09-29: `METRALE_GLM_PREFILL_STAGED=1` runs a prefill chunk as two passes per layer: the
/// attention half over every `prefill_rows()` sub-chunk, then the FFN half over windows of up to
/// `prefill_rows_ffn()` rows (`Glm5NextLayer::prefill_staged_run`). Off unless set to `1`; read
/// once. Eager prefill only: a call under graph capture takes the unstaged loop.
pub fn prefill_staged() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_STAGED=1 - GLM prefill runs attention at {} rows, then the \
                 FFN in windows of up to {} rows (byte-identical to the unstaged loop by \
                 construction; see steps/staged.rs)",
                prefill_rows(),
                prefill_rows_ffn()
            );
        }
        on
    })
}

/// 2026-09-29: The FFN window for attention width `attn` and a requested width: the request
/// (default `attn`) capped at `PREFILL_ROWS_FFN_MAX`, rounded down to a multiple of `attn`, and
/// never below `attn`.
pub(crate) fn resolve_rows_ffn(attn: usize, requested: Option<usize>) -> usize {
    let attn = attn.max(1);
    let want = requested
        .filter(|r| *r >= 1)
        .unwrap_or(attn)
        .min(PREFILL_ROWS_FFN_MAX);
    ((want / attn) * attn).max(attn)
}

/// 2026-09-29: The staged prefill's FFN window, from `METRALE_GLM_PREFILL_ROWS_FFN` through
/// `resolve_rows_ffn`; `prefill_rows()` when `prefill_staged()` is off, so every sizing that
/// reads it is unchanged then. Read once.
pub fn prefill_rows_ffn() -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        let attn = prefill_rows();
        let staged = std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if !staged {
            return attn;
        }
        let req = std::env::var("METRALE_GLM_PREFILL_ROWS_FFN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let r = resolve_rows_ffn(attn, req);
        if req.is_some_and(|q| q != r) {
            tracing::warn!(
                "METRALE_GLM_PREFILL_ROWS_FFN={} resolved to {r} (a multiple of the {attn}-row \
                 attention width, at most {PREFILL_ROWS_FFN_MAX})",
                req.unwrap_or_default()
            );
        }
        r
    })
}

/// 2026-10-01: `METRALE_GLM_PREFILL_TAIL_MERGE=1` (with `METRALE_GLM_PREFILL_STAGED=1`): the
/// staged FFN pass lets a mergeable tail sub-chunk (narrower than `prefill_rows()`) join the
/// window before it instead of taking a window, a routed grouped GEMM and an all-reduce of its
/// own (`ffn_windows`). Byte-identical by the same argument as the staged pass. Off unless set
/// to `1`; read once; inert when staging is off.
pub fn prefill_tail_merge() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_TAIL_MERGE").as_deref() == Ok("1")
            && std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_TAIL_MERGE=1 - a mergeable prefill tail joins the last                  staged FFN window (byte-identical by construction; see steps/staged.rs)"
            );
        }
        on
    })
}

/// 2026-09-29: The env-read inputs that decide which sub-chunks the staged FFN pass merges
/// (`grouped_prefill_selected`): the grouped-GEMM minimum rows and its switch, forced host
/// dispatch and route tracing, packed into one value for the rank-agreement check. A skew
/// would give the ranks different window counts, and each window issues its own all-reduce.
pub fn staged_merge_signature() -> u64 {
    use crate::glm5next_mlp::forward_prefill_gemm::{prefill_gemm_enabled, prefill_gemm_min_rows};
    ((prefill_gemm_min_rows() as u64) << 3)
        | (u64::from(prefill_gemm_enabled()) << 2)
        | (u64::from(crate::glm5next_mlp::forward::host_dispatch_forced()) << 1)
        | u64::from(super::profile::trace_on())
}
