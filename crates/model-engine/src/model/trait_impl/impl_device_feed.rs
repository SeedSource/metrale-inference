// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelDeviceFeed for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::model::types::TransformerModel;
use crate::traits::{FeedSource, ModelDeviceFeed, RowMask, SequenceState};

impl ModelDeviceFeed for TransformerModel {
    // 2026-09-25: Device token feed. The `_dispatch` helpers down to
    // `fed_step_settled_dispatch` are defined in `feed.rs`.
    fn supports_device_token_feed(&self) -> bool {
        self.supports_device_token_feed_dispatch()
    }

    fn decode_batch_fed(
        &self,
        sources: &[FeedSource],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.gdn_carry_flush_pending()?;
        self.kda_defer_flush(stream)?;
        self.decode_batch_fed_dispatch(sources, seqs, stream)
    }

    fn argmax_batch_to_feed(
        &self,
        logits: DevicePtr,
        masks: &[RowMask],
        dst: *mut u32,
        event: u64,
        stream: u64,
    ) -> Result<()> {
        self.argmax_batch_to_feed_dispatch(logits, masks, dst, event, stream)
    }

    fn reserve_decode_block(&self, seq: &mut SequenceState) -> Result<usize> {
        self.reserve_decode_block_dispatch(seq)
    }

    fn release_decode_blocks(&self, seq: &mut SequenceState, blocks: usize) -> Result<()> {
        self.release_decode_blocks_dispatch(seq, blocks)
    }

    fn fed_step_settled(&self) {
        self.fed_step_settled_dispatch()
    }

    fn event_query(&self, event: u64) -> Result<bool> {
        self.gpu.event_query(event)
    }

    fn event_synchronize(&self, event: u64) -> Result<()> {
        self.gpu.event_synchronize(event)
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        self.gpu.destroy_event(event)
    }

    fn copy_d2h_async(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
        self.gpu.copy_d2h_async(src, dst, stream)
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        self.gpu.alloc_host_pinned(bytes)
    }

    fn free_host_pinned(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        self.gpu.free_host_pinned(ptr, bytes)
    }
}
