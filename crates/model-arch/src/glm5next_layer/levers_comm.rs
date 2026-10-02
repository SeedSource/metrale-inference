// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The GLM-5.3 two-rank communication levers (split out of `levers.rs` for the
//! 500-line cap): the DSA index-selection row split, the staged-prefill all-reduce overlap
//! and the staged-prefill sequence parallelism. All default-off, all byte-identical by
//! construction.
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
/// to `1`; read once.
pub fn prefill_seq_parallel() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_PREFILL_SEQ_PARALLEL").as_deref() == Ok("1")
            && std::env::var("METRALE_GLM_PREFILL_STAGED").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_PREFILL_SEQ_PARALLEL=1 - staged GLM prefill splits mHC and norms \
                 by rows across the two ranks (byte-identical by construction; see \
                 steps/staged/sp.rs)"
            );
        }
        on
    })
}
