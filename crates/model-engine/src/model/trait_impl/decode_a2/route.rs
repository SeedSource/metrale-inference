// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The layer-declared inputs of `decode_batch_dispatch`'s multi-sequence route and
//! the one-shot line that names the route a batch took.
//!
//! Owner: model-engine (decode).
//! Invariants: none beyond the types.

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-01: True when every layer runs sparse-attention selection per sequence inside its
    /// batched `decode_multi_seq` (`decode_multi_seq_selection_per_seq`), so `qsa_active` need
    /// not send the batch to the per-sequence loop.
    pub(super) fn ms_selection_per_seq(&self) -> bool {
        self.layers
            .iter()
            .all(|l| l.decode_multi_seq_selection_per_seq())
    }

    /// 2026-10-01: True when some layer needs the batched multi-sequence step eager and unpadded
    /// (`decode_multi_seq_eager_only`): no graph capture or replay, exactly `n` rows.
    pub(super) fn ms_eager_only(&self) -> bool {
        self.layers.iter().any(|l| l.decode_multi_seq_eager_only())
    }

    /// 2026-10-01: Log, once per process, which route the first multi-sequence step took. The
    /// two routes differ in correctness as well as speed, and the boot-time concurrency note is
    /// printed from `max_batch_size` alone, so it does not say which one ran. From rsafier's
    /// Atlas e69446eee.
    pub(super) fn log_multi_seq_route(
        &self,
        n: usize,
        hc_perseq: bool,
        ms_layer_veto: bool,
        qsa_active: bool,
    ) {
        static ROUTE: std::sync::Once = std::sync::Once::new();
        ROUTE.call_once(|| {
            tracing::info!(
                n_seqs = n,
                ms_layer_veto,
                qsa_active,
                hc_mult = self.config.hc_mult,
                eager_only = self.ms_eager_only(),
                "multi-seq decode route: {}",
                if hc_perseq {
                    "PER-SEQ loop (one sequence at a time)"
                } else {
                    "BATCHED decode_multi_seq"
                }
            );
        });
    }
}
