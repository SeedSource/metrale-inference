// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_GLM_MS_DECODE_GRAPHS`: the batched multi-sequence decode step
//! (`METRALE_GLM_DECODE_MULTI_SEQ=1`) as a CUDA graph.
//!
//! Owner: model-arch (GLM-5.3).
//! With `METRALE_GLM_MS_DECODE_GRAPHS=1`, `decode_multi_seq_eager_only` answers false, so
//! model-engine `decode_a2.rs` captures and replays the width-n step. The step stays unpadded
//! (`decode_multi_seq_unpadded`: a DSA padding row would allocate state inside the capture),
//! so each exact width gets its own graph, keyed by the rows' slots. On a replay the engine runs
//! each layer's `check_replay_room` before and `sync_replayed_step` after, which advance the
//! DSA indexer's host length as the C=1 verify graphs do. The DSA cross-sequence projections
//! (`METRALE_GLM_DSA_XSEQ_BATCH`) stay engaged under capture: their indexer rows are stored
//! from the device position (`store_pre_indexer_row`). Off unless `1`; read once.
//!
//! 2026-10-07: `METRALE_GLM_BATCHED_VERIFY_GRAPHS`: the batched MTP verify
//! (`METRALE_GLM_BATCHED_VERIFY=1`, `steps/verify_multi.rs`) as a CUDA graph.
//! `decode_verify_multi_graphable` answers true, so model-engine `verify_e.rs` captures one graph
//! per exact `(slot, k)` key (no ghost-row borrow) and runs each sequence's `check_replay_room`
//! before and `sync_replayed_step` after a replay, as the C=1 verify graphs do; `free_sequence`
//! drops every batched key holding a freed slot. Each row's DSA position comes from the staged
//! verify metadata (`verify_rows_view`), so `decode_k` and the xseq group take the replay-safe
//! store. Off unless `1`; read once.

use std::sync::OnceLock;

/// Whether `METRALE_GLM_MS_DECODE_GRAPHS=1` is set (read once; logged when on).
pub fn ms_decode_graphs() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_MS_DECODE_GRAPHS").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_MS_DECODE_GRAPHS=1: ENGAGED - batched multi-sequence decode steps \
                 run as CUDA graphs (unpadded, one per exact width and slot set)"
            );
        }
        on
    })
}

/// Whether `METRALE_GLM_BATCHED_VERIFY_GRAPHS=1` is set (read once; logged when on).
pub fn bv_graphs() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_BATCHED_VERIFY_GRAPHS").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_BATCHED_VERIFY_GRAPHS=1: ENGAGED - batched MTP verify steps run as \
                 CUDA graphs (one per exact slot and row-count set)"
            );
        }
        on
    })
}

/// Whether the DSA cross-sequence projections may run under capture: either graph lever stores
/// their indexer rows from the device position.
pub fn xseq_capture_ok() -> bool {
    ms_decode_graphs() || bv_graphs()
}
