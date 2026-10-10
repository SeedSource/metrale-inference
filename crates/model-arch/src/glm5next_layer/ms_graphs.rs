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

/// 2026-10-10: Most rows a captured batched verify may hold (`decode_verify_multi_graph_max_rows`;
/// a wider one runs eagerly): `graph_max_rows(prefill_gemm_min_rows())`. Read once.
pub fn bv_graph_max_rows() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        graph_max_rows(crate::glm5next_mlp::forward_prefill_gemm::prefill_gemm_min_rows())
    })
}

/// 2026-10-10: The rows a batched-verify forward keeps on capture-safe kernels: at most
/// `dense_fp8::NV4_MAX_M` (16; wider dense GEMMs take the prefill dequant path, whose eager
/// dequant cache a capture turns off for the rest of the process), and below the routed MoE's
/// grouped GEMM (`grouped_prefill_selected`: more than `MOE_ROW_BATCH_MAX_ROWS` rows and at least
/// `moe_gemm_min_rows`). That route reads the expert histogram on the host (`dispatch.rs`
/// `grouped_max_m_tiles`, and the W4A4 arm), a stream sync no capture can hold: on 2026-10-10 an
/// 18-row capture (n=9, k=2, `METRALE_GLM_MOE_PREFILL_GEMM_MIN_ROWS=17`) was invalidated and the
/// EP communicator poisoned (ladg2m, MB=16).
pub fn graph_max_rows(moe_gemm_min_rows: usize) -> usize {
    use crate::glm5next_mlp::forward::MOE_ROW_BATCH_MAX_ROWS;
    let moe = MOE_ROW_BATCH_MAX_ROWS.max(moe_gemm_min_rows.saturating_sub(1));
    super::dense_fp8::NV4_MAX_M.min(moe)
}

/// Whether the DSA cross-sequence projections may run under capture: either graph lever stores
/// their indexer rows from the device position.
pub fn xseq_capture_ok() -> bool {
    ms_decode_graphs() || bv_graphs()
}

#[cfg(test)]
mod tests {
    use super::graph_max_rows;
    use crate::glm5next_mlp::forward::MOE_ROW_BATCH_MAX_ROWS;

    /// 2026-10-10: No row count within the cap reaches the grouped MoE GEMM or a dense GEMM wider
    /// than the decode GEMVs, for every MoE floor; the car's 17 and the default 128 cap at 16.
    #[test]
    fn graph_cap_stays_below_the_prefill_routes() {
        assert_eq!(graph_max_rows(17), 16);
        assert_eq!(graph_max_rows(128), 16);
        assert_eq!(graph_max_rows(12), 11);
        assert_eq!(graph_max_rows(1), MOE_ROW_BATCH_MAX_ROWS);
        for floor in 1..=200 {
            let cap = graph_max_rows(floor);
            assert!(cap <= super::super::dense_fp8::NV4_MAX_M);
            for r in 1..=cap {
                assert!(
                    !(r > MOE_ROW_BATCH_MAX_ROWS && r >= floor),
                    "floor {floor} r {r}"
                );
            }
        }
    }
}
