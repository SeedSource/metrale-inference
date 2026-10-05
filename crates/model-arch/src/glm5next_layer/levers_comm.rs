// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The GLM-5.3 two-rank communication levers (split out of `levers.rs` for the
//! 500-line cap): the DSA index-selection row split (sliced and full-width), the
//! staged-prefill all-reduce overlap and the staged-prefill sequence parallelism. All
//! default-off, all byte-identical by construction.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Each function reads its environment variable once per process and caches the result.

use super::{dsa_batch_qidx, dsa_row_batch, parse_dsa_switch, warn_unparsed_dsa_switch};

/// 2026-10-01: `METRALE_GLM_DSA_INDEX_SPLIT=1` (with `METRALE_GLM_DSA_ROW_BATCH=1`, and
/// `METRALE_GLM_DSA_BATCH_QIDX` off): on two tensor-parallel ranks the replicated DSA indexer's
/// query side of a prefill sub-chunk (the `wq_b` and `weights_proj` GEMVs, index scores, pool
/// top-k, expansion) runs for half the rows on each rank, and the ranks swap their halves of
/// the token output (`glm5next_dsa::select::split`). Byte-identical by construction (each row
/// gets the inputs and output slot it had in the full pass); adds one grouped send/recv per
/// DSA sub-chunk, so the ranks must agree on it (startup check). Inert without a two-rank
/// communicator or below two rows. Off unless set to `1`; read once.
pub fn dsa_index_split() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_INDEX_SPLIT").ok();
        let asked = parse_dsa_switch(raw.as_deref());
        warn_unparsed_dsa_switch("METRALE_GLM_DSA_INDEX_SPLIT", raw.as_deref());
        let on = asked && dsa_row_batch() && !dsa_batch_qidx();
        if asked {
            tracing::warn!(
                "METRALE_GLM_DSA_INDEX_SPLIT=1 - {} (needs METRALE_GLM_DSA_ROW_BATCH=1 and \
                 METRALE_GLM_DSA_BATCH_QIDX off; byte-identical by construction)",
                if on {
                    "two ranks split the DSA prefill index selection by rows"
                } else {
                    "NOT engaged"
                }
            );
        }
        on
    })
}

/// 2026-10-04: `METRALE_GLM_DSA_INDEX_SPLIT_WIDE=1`: the `METRALE_GLM_DSA_INDEX_SPLIT` row
/// split on the full-width staged prefill (`METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1`,
/// `Glm5NextDsaLayer::decode_k_wide`), which the base lever does not reach: on two
/// tensor-parallel ranks each `core_rows` sub-chunk's selection kernels (index scores, pool
/// top-k, expansion) run for half its rows on each rank, and the ranks swap their token rows
/// (`select_tokens_split`). The query projections (`wq_b`, `weights_proj`) still run over the
/// whole window on both ranks: they are one cuBLASLt GEMM at M = window, whose per-row bytes
/// depend on M. Byte-identical by construction (the base lever's argument); adds one grouped
/// send/recv per DSA sub-chunk of two or more rows, so the ranks must agree on it (startup
/// check). Independent of `METRALE_GLM_DSA_ROW_BATCH` / `METRALE_GLM_DSA_BATCH_QIDX`. Off
/// unless set to `1`; read once.
pub fn dsa_index_split_wide() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DSA_INDEX_SPLIT_WIDE").ok();
        let on = parse_dsa_switch(raw.as_deref());
        warn_unparsed_dsa_switch("METRALE_GLM_DSA_INDEX_SPLIT_WIDE", raw.as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_DSA_INDEX_SPLIT_WIDE=1 - two ranks split each full-width DSA \
                 prefill sub-chunk's index selection by rows (byte-identical by construction)"
            );
        }
        on
    })
}

/// 2026-10-01: `METRALE_GLM_PREFILL_COMM_OVERLAP=1` (with `METRALE_GLM_PREFILL_STAGED=1`): the
/// staged prefill overlaps each sub-chunk's mixer all-reduce with the next sub-chunk's attention
/// compute, and each FFN window's MLP all-reduce with the next window's FFN compute
/// (`steps/staged.rs`, schedule `comm_overlap::overlap_schedule`). The partial is copied into
/// the sub-chunk's own (dead after the norm) `hidden` rows and reduced there on the comm stream
/// (`CommBackend::all_reduce_deferred`); the compute stream waits for it only right before the
/// `hc_post` that folds it in. Byte-identical by construction: the same operands go through the
/// same `bf16_add_inplace`, every launch keeps its inputs, and the collective sequence (count,
/// sizes, order) is unchanged, so the ranks need not agree on it. Inert under graph capture,
/// with `METRALE_GLM_PROFILE` set, without a communicator or with staging off. Off unless set to
/// `1`; read once.
pub fn prefill_comm_overlap() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_COMM_OVERLAP").as_deref() == Ok("1")
            && std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_COMM_OVERLAP=1 - staged GLM prefill all-reduces overlap \
                 the next sub-chunk / FFN window (byte-identical by construction; see \
                 steps/staged.rs)"
            );
        }
        on
    })
}

/// 2026-10-01: `METRALE_GLM_PREFILL_SEQ_PARALLEL=1` (with `METRALE_GLM_PREFILL_STAGED=1`):
/// on two tensor-parallel ranks the staged prefill splits the replicated row-local work by
/// rows: rank 0 runs the hyper-connection (`hc_expand`, `hc_pre`, `hc_post`, `hc_head_mean`)
/// and the RMSNorms for the first half of every sub-chunk's rows, rank 1 for the rest. The
/// normed rows are exchanged before each mixer / MLP pass, and each partial goes only to the
/// owner of its rows, which adds it with the all-reduce's own kernel (a reduce-scatter in
/// place of the all-reduce); the last layer exchanges the final hidden rows
/// (`steps/staged/sp.rs`). Byte-identical by construction; changes the collective sequence,
/// so the ranks must agree on it (startup check). Supersedes
/// `METRALE_GLM_PREFILL_COMM_OVERLAP` while engaged. Inert under graph capture, without a
/// two-rank send/recv all-reduce, or when the chunk outgrows the norm buffer. Off unless set
/// to `1`; read once. Off under `METRALE_GLM_DFLASH=1` (startup warning): rank 0's DFlash
/// capture reads every row of a tap layer's `hidden`, and `SpRows::back` collapses only the
/// rank's own rows.
/// 2026-10-04: Composes with `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1` (before, inert under it):
/// the attention pass then runs per `rows_ffn` window and the MLP's dense GEMMs per window,
/// exactly the full-width arm's calls; only the row-local mHC / norm work is split (ownership
/// stays per sub-chunk). Splits nothing else: the router, MoE sort / combine and DSA indexer
/// projections stay on every row (see the lever doc in `table_a.rs`).
pub fn prefill_seq_parallel() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_SEQ_PARALLEL").as_deref() == Ok("1")
            && std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on && metrale_model_layers::speculative::glm_dflash::glm_dflash_enabled() {
            tracing::warn!(
                "METRALE_GLM_PREFILL_SEQ_PARALLEL=1 IGNORED: METRALE_GLM_DFLASH=1 is set and \
                 the DFlash tap capture needs every row of a tap layer on rank 0 (mutually \
                 exclusive)"
            );
            return false;
        }
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_SEQ_PARALLEL=1 - staged GLM prefill splits mHC and norms \
                 by rows across the two ranks, full-width arm included (byte-identical by \
                 construction; see steps/staged/sp.rs)"
            );
        }
        on
    })
}

/// 2026-10-05: `METRALE_GLM_PREFILL_SP_WINDOW_OWNER=1` (with
/// `METRALE_GLM_PREFILL_SEQ_PARALLEL=1` and `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1`): the
/// sequence-parallel ownership is cut per `rows_ffn` window instead of per `rows` sub-chunk
/// (`seq_parallel::owner_chunks`), so rank 0 owns the first half of every window. Every
/// sequence-parallel call is then one span per rank: one send/recv pair per exchange instead
/// of one per sub-chunk (an 8192-row window at `rows` 256: 1 x 32 MiB instead of 32 x 1 MiB),
/// and one owned mHC / norm / add launch per call instead of one per sub-chunk.
/// Byte-identical by the same argument as the base lever (every owned-row launch is per
/// token; `steps/staged/sp.rs`); changes the collective sequence, so the ranks must agree
/// (startup check). Off unless set to `1`; read once; inert without the two levers above.
pub fn prefill_sp_window_owner() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_SP_WINDOW_OWNER").as_deref() == Ok("1")
            && prefill_seq_parallel();
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_SP_WINDOW_OWNER=1 - sequence-parallel row ownership per \
                 full-width window (one send/recv pair per exchange; byte-identical)"
            );
        }
        on
    })
}
