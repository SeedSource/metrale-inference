// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `LayerCapabilities`, the side-effect-free questions the model asks a layer when it
//! picks a route: graph eligibility, batched decode and verify support, rollback, SSM
//! pool use and the MLA prefill mode.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

/// 2026-09-26: A supertrait of `TransformerLayer`; see the module header.
pub trait LayerCapabilities {
    /// 2026-09-25: True when this layer's prefill attends only over the tokens it is
    /// handed, so skipping a cached prefix would hide it from attention. The Qwen3
    /// attention layer returns true when it runs MLA; the prefix lookup then matches
    /// nothing unless `METRALE_MLA_PREFIX_SKIP=1` (`mla_prefill_needs_full_recompute`).
    fn uses_local_mla_prefill(&self) -> bool {
        false
    }

    /// 2026-09-25: Whether this layer's online FP8-KV calibration has frozen its scale;
    /// `None` when the layer runs no online calibration (the default). The model lifts its
    /// CUDA-graph suppression once no layer reports `Some(false)`
    /// (`graphs_ready_after_fp8_kv_cal`).
    fn fp8_calibration_frozen(&self) -> Option<bool> {
        None
    }

    /// 2026-09-25: True when this layer's decode cannot run inside a CUDA graph, such as
    /// the QSA indexer's host top-k. The model ORs it across layers into
    /// `decode_graph_veto` when it is built.
    fn decode_graph_unsupported(&self) -> bool {
        false
    }

    /// 2026-09-25: True when this layer cannot serve a batched multi-sequence decode step.
    /// The batched decode ORs it across layers into `hc_perseq` (`decode_a2.rs`) and then
    /// runs each sequence through `decode`; the single-GPU fused decode+prefill ORs it into
    /// `hc_qsa_perseq` (`decode_b.rs`) and then runs the batched decode and the prefill
    /// separately.
    fn decode_multi_seq_unsupported(&self) -> bool {
        false
    }

    /// 2026-10-01: True when this layer's batched `decode_multi_seq` runs sparse-attention
    /// selection per sequence (each row against its own indexer state, page table and length),
    /// so a batch may stay batched once some sequence passes the index budget. `decode_a2.rs`
    /// requires it of every layer before ignoring `qsa_active`. Default false: past the budget
    /// the batch runs per sequence, because a batched path that shared one selection would
    /// attend with another sequence's chosen blocks. From rsafier's Atlas 04beaac1f.
    fn decode_multi_seq_selection_per_seq(&self) -> bool {
        false
    }

    /// 2026-10-01: True when a batched multi-sequence decode step must run eagerly and unpadded.
    /// `decode_a2.rs` then captures no graph and runs exactly `n` rows. GLM-5.3 answers true:
    /// its DSA state is allocated per sequence (a padding row would allocate one every step),
    /// and the indexer's host-side length advances only when the layer code runs, not on a
    /// graph replay of the batch.
    fn decode_multi_seq_eager_only(&self) -> bool {
        false
    }

    /// 2026-10-07: True when a batched multi-sequence decode step must run exactly `n` rows even
    /// when it is graphed: `decode_a2.rs` then skips the padding ladder and graph borrowing, and
    /// captures one graph per exact width. GLM-5.3 answers true (per-sequence DSA state).
    fn decode_multi_seq_unpadded(&self) -> bool {
        false
    }

    /// 2026-10-01: True when this layer cannot share the fused decode + prefill forward
    /// (`decode_b.rs`), where the prefill chunk's highway rows start at `hc_row_offset`. GLM-5.3
    /// answers true: its prefill numbers highway slots from 0, over the decode rows' slots.
    fn fused_decode_prefill_unsupported(&self) -> bool {
        false
    }

    /// 2026-09-25: True when this layer keeps per-sequence state that lowering the
    /// sequence's KV cursor does not rewind, such as a monotonic cache count or an n-gram
    /// history. The model ORs it across layers, and the scheduler's `rollback_to_boundary`
    /// then declines the rollback (`LayerStateNotRewindable`).
    fn decode_rollback_unsupported(&self) -> bool {
        false
    }

    /// 2026-09-25: True when this layer cannot serve a batched multi-sequence verify
    /// (`decode_verify_multi`); `can_batch_verify_dispatch` then refuses the batch. It is
    /// separate from [`Self::decode_multi_seq_unsupported`] because the answers can
    /// differ.
    fn decode_verify_multi_unsupported(&self) -> bool {
        false
    }

    /// 2026-10-02: True when this layer serves the batched verify through
    /// `TransformerLayer::decode_verify_multi_seqs` instead of `decode_multi_seq` /
    /// `decode_verify_multi`: it owns per-sequence state the engine cannot stage (GLM-5.3's DSA
    /// indexer cache), runs eager, reads no WY pointer tables and takes no write-on-accept
    /// carry. The batched verify (`verify_e.rs`) then stages no WY tables, captures no graph,
    /// and routes the layer to `decode_verify_multi_seqs` whatever its `LayerType`. When every
    /// layer answers true, the multi-rank batched verify (`verify_ep.rs`) may run: the layer's
    /// collectives depend only on the batch shape and per-sequence lengths, which every rank
    /// holds identically. Consulted only where `decode_verify_multi_unsupported` is false.
    fn decode_verify_multi_own_states(&self) -> bool {
        false
    }

    /// 2026-10-07: True when this own-state layer's `decode_verify_multi_seqs` is replay-safe
    /// under capture: every position it writes comes from the staged metadata, and
    /// `check_replay_room` / `sync_replayed_step` reconcile its host bookkeeping per sequence.
    /// When every own-state layer answers true, the batched verify may capture a graph keyed by
    /// the exact `(slot, k)` pairs, with no ghost-row borrow. Consulted only where
    /// [`Self::decode_verify_multi_own_states`] is true.
    fn decode_verify_multi_graphable(&self) -> bool {
        false
    }

    /// 2026-10-02: Most verify rows (`R = Σ ks`) one `decode_verify_multi_seqs` call takes;
    /// `can_batch_verify_dispatch` refuses a wider batch, which the scheduler then verifies in
    /// narrower chunks. Unbounded unless the layer's scratch is sized for fewer rows.
    fn decode_verify_multi_max_rows(&self) -> usize {
        usize::MAX
    }

    #[allow(clippy::too_many_arguments)]
    /// 2026-09-25: True when a captured decode graph goes stale once a new sequence takes
    /// this slot. Decode graphs are keyed by `slot_idx`, which is safe only while every
    /// per-sequence address a capture bakes lives in the slot-addressed SSM pool. A layer
    /// that allocates its own per-sequence buffers (GLM-5.3's DSA indexer cache) returns
    /// true, and `free_sequence_dispatch` then drops that slot's graphs, except for a slot
    /// index past the SSM pool's `max_slots`.
    fn graph_stale_on_new_sequence(&self) -> bool {
        false
    }

    /// 2026-09-25: True for an SSM layer. The model's split prefill then runs
    /// `prefill_phase1`, `prefill_gdn_full` and `prefill_phase3` instead of `prefill`.
    fn is_ssm_layer(&self) -> bool {
        false
    }

    /// 2026-09-25: Whether this layer's recurrent state lives in the shared SSM pool. When
    /// true (the default), sequence setup hands a linear-attention layer an `SsmLayerState`
    /// with pool addresses and never calls its `alloc_state`. A linear-attention layer with
    /// its own state type returns false; the Kimi K3 layer (`kimi_k3/bound.rs`) does, and
    /// downcasts its state to `K3CpuFallbackState`.
    fn uses_ssm_pool(&self) -> bool {
        true
    }

    /// 2026-10-03: True when this SSM layer's prefill honours an in-pass capture
    /// (`MidchunkCapture::inpass_split`): it writes its recurrent and conv state as of the
    /// capture point into the reserved snapshot slot, inside the pass, bit-identical to the
    /// state the pass itself carries there, and counts itself in `MidchunkCapture::captured`.
    /// The model enables `METRALE_GLM_SSM_INPASS_CAPTURE` only when every SSM layer answers
    /// true (model-engine `prefill_b/inpass_capture.rs`). Default false.
    fn inpass_ssm_capture_supported(&self) -> bool {
        false
    }
}
