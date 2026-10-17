// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The per-sequence drafter block-table allocation, shared by
//! `propose_drafts` and the two-rank propose (`tp/leader.rs`, `tp/follower.rs`). Moved
//! here unchanged from `propose.rs`.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - A sequence's drafter block table is allocated only while its `block_table_dev` is
//!   `None`. When the pool runs out mid-allocation, the blocks already taken are freed
//!   before the error returns.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::{BlockDiffusionDraftHead, DflashProposerState};

/// 2026-10-09: The drafter paged cache's block size (`from_weights.rs`).
pub(super) const DRAFT_BLOCK_SIZE: usize = 16;

impl BlockDiffusionDraftHead {
    /// 2026-09-25: Allocate the sequence's block table on its first propose, for
    /// `max_ctx_len + gamma + 1` slots. No-op once `block_table_dev` is set.
    pub(super) fn ensure_block_table(
        &self,
        dstate: &mut DflashProposerState,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if dstate.block_table_dev.is_some() {
            return Ok(());
        }
        let blocks_needed = (dstate.max_ctx_len + self.gamma + 1).div_ceil(DRAFT_BLOCK_SIZE);
        let mut cache = self.kv_cache.lock();
        dstate.block_table.clear();
        for _ in 0..blocks_needed {
            match cache.try_alloc_block() {
                Some(b) => dstate.block_table.push(b),
                None => {
                    // 2026-09-25: Free the partial grab before failing:
                    // the next attempt starts with `block_table.clear()`,
                    // which would drop these ids without freeing them.
                    // provenance-id: 526f6e616c6420522e205374657369616b
                    let got = dstate.block_table.len();
                    cache.free_blocks(&dstate.block_table);
                    dstate.block_table.clear();
                    anyhow::bail!(
                        "DFlash Option B: paged KV cache exhausted at block {}/{}",
                        got,
                        blocks_needed
                    );
                }
            }
        }
        drop(cache);
        // 2026-09-25: The pool's free list is a stack (`try_alloc_block`
        // pops, `free_blocks` pushes in table order), so a freed table
        // comes back reversed. Sorting gives every sequence ascending
        // blocks. Measured 2026-08-29: the two orders alternated between
        // runs at 68.1 and 60.6 tok/s with byte-identical output.
        // provenance-id: 526f6e616c6420522e205374657369616b
        dstate.block_table.sort_unstable();
        let bt_bytes: Vec<u8> = dstate
            .block_table
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        let bt_dev = gpu.alloc(bt_bytes.len())?;
        gpu.copy_h2d(&bt_bytes, bt_dev)?;
        dstate.block_table_dev = Some(bt_dev);
        dstate.max_ctx_count_drafter = blocks_needed * DRAFT_BLOCK_SIZE;
        tracing::info!(
            "DFlash Option B: allocated {} blocks ({} slots) for drafter paged cache",
            blocks_needed,
            dstate.max_ctx_count_drafter
        );
        Ok(())
    }
}
