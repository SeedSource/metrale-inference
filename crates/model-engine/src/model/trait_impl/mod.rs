// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `Model` and its supertraits for `TransformerModel`: one `impl_<supertrait>.rs`
//! file per supertrait, mostly delegating to `<method>_dispatch` helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants:
//! - `prefill`, `prefill_chunk`, `prefill_twophase` and `mixed_forward` call
//!   `try_eager_drafter_prefill` only after their forward returns `Ok`.
//! - When the forward returns `Err`, `decode_batch`, `mixed_forward` and every
//!   verify method that goes through `release_verify_capture_on_err` end any
//!   capture still open on the default stream.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{
    ChunkedPrefillPageMetadata, FeedSource, Model, PrefillSlice, RowMask, SequenceState,
};
use metrale_model_layers::layer::{AttnMetadataDev, LayerState};
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights};

mod async_chkpt;
mod borrow_streak;
mod decode_a;
mod decode_a2;
mod decode_a3;
mod decode_a_diag;
mod decode_b;
mod decode_b2;
pub(in crate::model) mod decode_checkpoint;
// 2026-10-05: The rank-agreed lazy-map admission of decode and verify steps (A168).
mod decode_lazy_agree;
#[cfg(test)]
mod decode_lazy_agree_tests;
mod decode_graph_key;
mod decode_multi_seq_gate;
mod drafter_prefill;
mod ep_misc;
mod feed;
mod gdn_woa;
mod graph_borrow;
mod impl_adapters;
mod impl_device_feed;
mod impl_draft;
mod impl_ep;
mod impl_forward;
mod impl_lifecycle;
mod impl_logits;
mod impl_ssm_state;
mod impl_streams;
mod impl_verify;
mod impl_vision;
mod lm_head_batched;
mod meta;
mod meta_argmax;
mod prefill_a;
mod prefill_b;
mod prefill_c;
mod prefill_d;
mod prefix_reuse;
mod sequence;
mod sequence_compact;
mod speculative;
mod speculative_mtp;
pub(in crate::model) mod ssm_fault_in;
mod verify_a;
mod verify_a_ssm;
mod verify_b;
mod verify_c;
mod verify_c2;
mod verify_d;
mod verify_e;
pub(in crate::model) mod verify_e2;
// 2026-10-02: The multi-rank batched MTP verify (`EP_CMD_VERIFY_BATCH`).
pub(in crate::model) mod verify_ep;
// 2026-10-04: The batched MTP propose that mirrors the per-sequence one
// (`EP_CMD_MTP_PROPOSE_BATCH`).
pub(in crate::model) mod propose_batch_ep;
mod verify_fused;
mod worker_commit;

impl Model for TransformerModel {}

impl TransformerModel {
    /// 2026-09-25: On `Err`, end and discard any capture still open on the
    /// default stream, as `decode_batch` does, so a verify that fails during
    /// capture does not leave the stream capturing. A no-op when the stream is
    /// not capturing. The verify modules that capture (verify_b, verify_c,
    /// verify_c2, verify_d, verify_e, verify_fused) capture on the default
    /// stream.
    fn release_verify_capture_on_err<T>(&self, r: Result<T>) -> Result<T> {
        if r.is_err() {
            self.gpu.abort_capture_if_active(self.gpu.default_stream());
        }
        r
    }

    /// 2026-09-26: Collect each layer's aux sequence state
    /// (`LayerAuxState::snapshot_aux`) as `(layer index, blob)` pairs to
    /// store with an SSM snapshot. Empty when no layer returns a blob.
    pub(in crate::model) fn collect_aux_states(
        &self,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<Vec<(u32, Vec<u8>)>> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(blob) =
                l.snapshot_aux(seq.layer_states[i].as_ref(), self.gpu.as_ref(), stream)?
            {
                out.push((i as u32, blob));
            }
        }
        Ok(out)
    }

    /// 2026-10-03: Host bytes `collect_aux_states` would copy for a snapshot of the first `rows`
    /// positions, without allocating them (the aux byte budget sizes a grid save first).
    pub(in crate::model) fn aux_prefix_bytes_total(
        &self,
        seq: &SequenceState,
        rows: usize,
    ) -> usize {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| l.aux_prefix_bytes(seq.layer_states[i].as_ref(), rows))
            .sum()
    }

    /// 2026-09-25: True when some layer reports `has_aux_state`; restore sites
    /// then decline a snapshot that has no aux blobs.
    pub(in crate::model) fn requires_aux_state(&self) -> bool {
        self.layers.iter().any(|l| l.has_aux_state())
    }

    /// 2026-09-25: Apply a snapshot's aux blobs to the layers they came from.
    /// The first layer error is returned.
    pub(in crate::model) fn apply_aux_states(
        &self,
        seq: &mut SequenceState,
        blobs: &[(u32, Vec<u8>)],
        stream: u64,
    ) -> Result<()> {
        for (i, blob) in blobs {
            self.layers[*i as usize].restore_aux(
                seq.layer_states[*i as usize].as_mut(),
                blob,
                self.gpu.as_ref(),
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod gdn_woa_folded_slots_tests {
    /// 2026-09-25: The folded-slot record must hold only the slots folded by
    /// the most recent fold.
    ///
    /// The logic is modelled directly because the real path needs a GPU.
    /// `record_new` is the recording in `gdn_fold_accepted_dispatch` (clear,
    /// then extend only when a layer folded). `record_old` clears only when a
    /// layer folded; it is the negative control.
    fn record_old(list: &mut Vec<usize>, any: bool, slots: &[usize]) {
        if any {
            list.clear();
            list.extend_from_slice(slots);
        }
    }
    fn record_new(list: &mut Vec<usize>, any: bool, slots: &[usize]) {
        list.clear();
        if any {
            list.extend_from_slice(slots);
        }
    }
    /// 2026-09-25: The reader in async_chkpt.rs: removes `slot` and returns
    /// true when it was listed, which skips that slot's `h` restore.
    fn consume(list: &mut Vec<usize>, slot: usize) -> bool {
        match list.iter().position(|&s| s == slot) {
            Some(p) => {
                list.swap_remove(p);
                true
            }
            None => false,
        }
    }

    /// 2026-09-25: A fold that folds nothing must not leave a previous batch's
    /// slots listed. The reader is keyed by slot index, so a stale entry makes
    /// the slot's next occupant skip its `h` restore.
    #[test]
    fn a_no_op_fold_does_not_leave_a_stale_claim() {
        let mut new = vec![];
        record_new(&mut new, true, &[1, 2, 3, 4]);
        for s in [1, 2, 3, 4] {
            assert!(consume(&mut new, s), "batch A's own slots are folded");
        }
        record_new(&mut new, false, &[7]);
        assert!(
            !consume(&mut new, 7),
            "slot 7 was NOT folded and must restore h"
        );

        // 2026-09-25: Control: `record_old` fails this. The no-op fold leaves
        // batch A's entry for slot 3 in the list.
        let mut old = vec![];
        record_old(&mut old, true, &[1, 2, 3, 4]);
        record_old(&mut old, false, &[7]);
        assert!(
            consume(&mut old, 3),
            "the shipped behaviour leaves slot 3 claimed — this assertion \
             documents the bug, and its inverse is the fix above"
        );
    }
}
