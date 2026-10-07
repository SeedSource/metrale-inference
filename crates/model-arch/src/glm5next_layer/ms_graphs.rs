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
