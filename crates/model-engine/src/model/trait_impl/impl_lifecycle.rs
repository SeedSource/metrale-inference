// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelLifecycle for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;

use crate::model::types::TransformerModel;
use crate::traits::{ModelLifecycle, SequenceState};

impl ModelLifecycle for TransformerModel {
    fn teardown(&mut self) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.release_pools()
    }

    /// 2026-09-25: Poll this model's InnerQ calibration driver, if it has one.
    /// A `maybe_finalize` error is logged as a warning and not returned.
    #[cfg(feature = "cuda")]
    fn poll_innerq(&self) {
        if let Some(driver) = self.innerq.as_ref()
            && let Err(e) = driver.maybe_finalize(128)
        {
            tracing::warn!(target: "metrale_model_engine::model::trait_impl", "InnerQ maybe_finalize failed: {e:#}");
        }
    }

    fn high_speed_swap_dims(&self) -> Option<metrale_storage::ModelDims> {
        self.high_speed_swap_dims_dispatch()
    }

    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.bind_gpu_to_thread_dispatch()
    }

    fn alloc_sequence(&self) -> Result<SequenceState> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        let seq = self.alloc_sequence_dispatch(usize::MAX)?;
        // 2026-10-09: Slot reuse: no deferred commit of a previous occupant survives.
        self.kda_defer_reset_slot(seq.slot_idx)?;
        Ok(seq)
    }

    fn alloc_sequence_for(&self, budget_tokens: usize) -> Result<SequenceState> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        let seq = self.alloc_sequence_dispatch(budget_tokens)?;
        self.kda_defer_reset_slot(seq.slot_idx)?;
        Ok(seq)
    }

    fn cache_sequence(&self, seq: &SequenceState) {
        if let Err(e) = self.gdn_carry_flush_pending() {
            tracing::error!("gdn carry flush before {}: {e:#}", "cache_sequence");
        }
        if let Err(e) = self.kda_defer_flush(self.gpu.default_stream()) {
            tracing::error!("KDA deferred-commit flush before cache_sequence: {e:#}");
        }
        self.cache_sequence_dispatch(seq)
    }

    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        let slot = seq.slot_idx;
        self.free_sequence_dispatch(seq)?;
        self.kda_defer_reset_slot(slot)
    }

    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        let old = seq.slot_idx;
        self.compact_sequence_dispatch(seq, new_slot)?;
        self.kda_defer_reset_slot(old)?;
        self.kda_defer_reset_slot(new_slot)
    }

    fn migrate_to_covered_slot(&self, seq: &mut SequenceState) -> Result<Option<usize>> {
        self.migrate_to_covered_slot_dispatch(seq)
    }

    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        if let Err(e) = self.gdn_carry_flush_pending() {
            tracing::error!("gdn carry flush before {}: {e:#}", "detach_slot_for_reuse");
        }
        if let Err(e) = self.kda_defer_flush(self.gpu.default_stream()) {
            tracing::error!("KDA deferred-commit flush before detach_slot_for_reuse: {e:#}");
        }
        let slot = seq.slot_idx;
        self.detach_slot_for_reuse_dispatch(seq);
        if let Err(e) = self.kda_defer_reset_slot(slot) {
            tracing::error!("KDA deferred-commit reset after detach_slot_for_reuse: {e:#}");
        }
    }

    fn save_sequence_state(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.save_sequence_state_dispatch(seq, writer)
    }

    fn restore_sequence_state(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(self.gpu.default_stream())?;
        self.restore_sequence_state_dispatch(seq, num_blocks, reader)
    }

    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks_dispatch()
    }

    fn num_total_blocks(&self) -> usize {
        self.num_total_blocks_dispatch()
    }

    fn reclaim_prefix_blocks(&self, num_blocks: usize) -> usize {
        self.reclaim_prefix_blocks_dispatch(num_blocks)
    }
}
