// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Sequence retirement: insert a finished sequence into the prefix cache
//! (`cache_sequence_dispatch`), release everything it owns (`free_sequence_dispatch`), and
//! the KV block counters.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

mod state_io;

impl TransformerModel {
    pub(super) fn cache_sequence_dispatch(&self, seq: &SequenceState) {
        // 2026-10-03: Under the absolute grid (`prefill_b/grid_restore.rs`) the prefill-end insert
        // is the whole contribution: generated tokens' blocks come from decode and verify passes,
        // which no cold prefill runs, so they are not cached, and no finish-leaf snapshot is
        // saved (the restore uses grid points only). The sequence's refs stay as the prefill
        // left them (`prefix_ref_tokens`).
        if seq.prefix_grid_refs {
            return;
        }
        let bs = self.kv_cache.lock().block_size();
        if seq.tokens.len() >= bs && !seq.block_table.is_empty() {
            // 2026-09-25: Prefill already inserted the prompt, so `prompt_len` is passed as
            // `matched_tokens` and `insert` treats only the generated tokens as sequence-owned.
            // Skipped when the prefix cache is inactive, for vision prompts, and once the HSS
            // window has slid (`block_table` then no longer parallels `tokens`).
            if self.prefix_cache.is_active()
                && !self.tokens_have_vision_pad(&seq.tokens)
                && seq.hss_window_start() == 0
            {
                // 2026-09-25: A leaf SSM snapshot at the full length (prompt + generated), so
                // the next warm hit restores at this turn's end (`finish_leaf_snapshot`).
                let finish_snap = self.finish_leaf_snapshot(seq);
                let acquired = if let Some(snap_id) = finish_snap {
                    let (displaced, acquired) = self.prefix_cache.insert_with_snapshot(
                        &seq.tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        snap_id,
                        seq.session_hash,
                        seq.prompt_len,
                        seq.adapter_id,
                    );
                    if let Some(old) = displaced {
                        self.ssm_snapshots.free(old);
                    }
                    acquired
                } else {
                    self.prefix_cache.insert(
                        &seq.tokens,
                        &seq.block_table,
                        &seq.disk_block_ids,
                        bs,
                        seq.prompt_len,
                        seq.adapter_id,
                    )
                };
                // 2026-09-25: Take the cache's refs on exactly the blocks the insert reports,
                // not on this sequence's `block_table` (see `cache_acquires_refs`).
                super::super::block_mgmt::cache_acquires_refs(&acquired, &mut self.kv_cache.lock());
            }
        }
    }

    pub(super) fn free_sequence_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        // 2026-09-25: The SSM slot is released first, and its zeroing errors are only
        // logged, so a failing GPU cannot keep the slot from returning to the free list.
        //
        // `slot_idx >= max_slots` is the sentinel `detach_slot_for_reuse_dispatch` sets
        // when another sequence took this slot over: the guard is emptied but the index is
        // not released. Otherwise `take()` yields the owned index, which is released once,
        // and leaves the guard's `Drop` with nothing to release.
        let slot_reused_by_compact = seq.slot_idx >= self.ssm_pool.max_slots;
        let taken = seq.ssm_slot.as_mut().and_then(|g| g.take());
        let slot_to_release = if slot_reused_by_compact { None } else { taken };
        if let Some(slot) = slot_to_release {
            let stream = self.gpu.default_stream();
            if let Err(e) = self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream) {
                tracing::error!("free_sequence: ssm_pool.zero_slot({slot}): {e:#}");
            }
            if let Err(e) = self.gpu.synchronize(stream) {
                tracing::error!("free_sequence: gpu.synchronize after zero_slot({slot}): {e:#}");
            }
            self.ssm_pool.release_slot(slot);
        }

        // 2026-09-25: Layer state holds bare `DevicePtr`s, so dropping `layer_states` frees
        // only the host structs; each layer frees the device memory it allocated per
        // sequence. Errors are logged, not propagated, so the KV blocks and prefix refs
        // below are still released.
        for (layer_idx, ls) in seq.layer_states.iter_mut().enumerate() {
            if let Some(layer) = self.layers.get(layer_idx)
                && let Err(e) = layer.release_state(ls.as_mut(), self.gpu.as_ref())
            {
                tracing::error!("free_sequence: release_state(layer {layer_idx}): {e:#}");
            }
        }

        // 2026-09-25: Release the LoRA slot ref this sequence acquired. `-1` means none,
        // and resetting it to `-1` makes a second free release nothing.
        if seq.acquired_adapter_slot >= 0 {
            self.release_adapter_slot(seq.acquired_adapter_slot);
            seq.acquired_adapter_slot = -1;
        }

        // 2026-09-25: Release the lookup's radix refs before freeing the blocks. A prefill
        // that matched a prefix and then failed never filled `seq.tokens`, so when
        // `tokens` is shorter than the matched prefix the release runs over the prefix
        // tokens stashed at lookup (`prefix_ref_tokens`). Only one of the two is used.
        // 2026-10-03: Under the absolute grid the sequence holds refs on exactly
        // `prefix_ref_tokens` (`SequenceState::prefix_grid_refs`), whatever stage it failed or
        // finished at; releasing `seq.tokens` would also decrement nodes other sequences inserted
        // past it.
        let release_tokens = if seq.prefix_grid_refs {
            &seq.prefix_ref_tokens
        } else if seq.tokens.len() >= seq.cached_prefix_tokens {
            &seq.tokens
        } else {
            &seq.prefix_ref_tokens
        };
        self.prefix_cache.release(
            release_tokens,
            self.kv_cache.lock().block_size(),
            seq.adapter_id,
        );
        if !seq.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&seq.block_table);
            seq.block_table.clear();
        }

        // 2026-09-25: `--high-speed-swap`: drop one disk ref per disk block id. An id
        // addresses the same slot in every layer's file, so one ref covers all layers, and
        // the id returns to the free list only when its refcount reaches 0.
        if !seq.disk_block_ids.is_empty() {
            // 2026-09-25: `None` when no HSS instance is installed on this thread.
            if let Some(Err(e)) = metrale_storage::with_local(|hss| {
                for &disk_id in &seq.disk_block_ids {
                    hss.dec_disk_ref(disk_id);
                }
                Ok(())
            }) {
                tracing::error!("free_sequence: metrale_storage dec_disk_ref batch: {e:#}");
            }
            seq.disk_block_ids.clear();
            for v in seq.disk_last_offloaded_per_layer.iter_mut() {
                *v = 0;
            }
        }

        // 2026-09-25: Drop this slot's graphs when a layer owns per-sequence device state.
        // The graph caches are slot-keyed because the per-sequence addresses a capture
        // bakes are normally in the slot-addressed SSM pool; a layer that allocates state per
        // sequence (the GLM-5.3 DSA mixer, `graph_stale_on_new_sequence`) would have the
        // next request replay graphs that write this one's freed buffers. Only this slot's
        // entries are dropped, and only then (`decode_graph_key` tests pin it).
        if !slot_reused_by_compact && self.layers.iter().any(|l| l.graph_stale_on_new_sequence()) {
            let slot = seq.slot_idx as u32;
            let mut stale: Vec<metrale_gpu_runtime::gpu::GraphHandle> = self
                .decode_graph
                .lock()
                .remove(&seq.slot_idx)
                .into_iter()
                .collect();
            // 2026-09-25: The K-row verify graphs are slot-keyed too and bake the same
            // per-sequence state (`Glm5NextDsaState::{k_normed, gate, valid}`).
            for m in [
                &self.verify2_graph,
                &self.verify3_graph,
                &self.verify4_graph,
            ] {
                stale.extend(m.lock().remove(&seq.slot_idx));
            }
            {
                // 2026-09-25: A batched graph bakes every row's state pointers, so every key
                // that contains this slot is dropped.
                let mut batch = self.batch_decode_graphs.lock();
                let keys: Vec<Vec<u32>> = batch
                    .0
                    .keys()
                    .filter(|k| k.contains(&slot))
                    .cloned()
                    .collect();
                for k in keys {
                    if let Some((g, _)) = batch.0.remove(&k) {
                        stale.push(g);
                    }
                }
            }
            for g in stale {
                if g.0 != 0
                    && let Err(e) = self.gpu.destroy_graph(g)
                {
                    tracing::warn!("free_sequence: destroy graph for slot {slot}: {e:#}");
                }
            }
        }

        // 2026-09-25: The SSM buffers belong to the pool, so the references are cleared,
        // not freed.
        for state in &mut seq.layer_states {
            if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                ssm.h_state = DevicePtr(0);
                ssm.conv_state = DevicePtr(0);
                ssm.h_prefill_stage = None;
                ssm.h_state_checkpoint = None;
                ssm.conv_state_checkpoint = None;
                ssm.h_state_intermediates.clear();
                ssm.conv_state_intermediates.clear();
            }
        }

        // 2026-09-25: `verify_kgamma_graph` and `fused_graph` are keyed by (slot, K) and bake
        // the request's LoRA routing, so every K of this slot is dropped.
        for graph_map in [&self.verify_kgamma_graph, &self.fused_graph] {
            let mut cache = graph_map.lock();
            let keys: Vec<(usize, usize)> = cache
                .keys()
                .filter(|k| k.0 == seq.slot_idx)
                .copied()
                .collect();
            for k in keys {
                if let Some(graph) = cache.remove(&k)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(kgamma/fused[{},{}]): {e:#}",
                        k.0,
                        k.1
                    );
                }
            }
        }

        // 2026-09-25: MTP drafter carry: move this turn's drafter KV into the model's single
        // carry slot before the proposer state is freed, so the next turn of the same session
        // can adopt it. `take_drafter_kv` empties the proposer state, so the blocks are owned
        // by the carry slot or by a sequence, never both; a displaced carry is freed.
        if metrale_model_layers::mtp_carry::mtp_carry_drafter_enabled(&self.levers)
            && let Some(ref proposer) = self.proposer
            && let Some(ref mut pstate) = seq.proposer_state
            && let Some((blocks, rows, last_pair_key)) = proposer.take_drafter_kv(pstate.as_mut())
        {
            let entry = metrale_model_layers::mtp_carry::CarriedDrafter {
                block_table: blocks,
                rows,
                last_pair_key,
                tokens: seq.tokens.clone(),
                // 2026-09-25: `CarriedDrafter::usable_by` accepts only a matching non-zero session.
                session_hash: seq.session_hash,
            };
            let previous = self.mtp_carry.lock().replace(entry);
            if let Some(old) = previous {
                proposer.free_drafter_kv(&old.block_table);
            }
            if metrale_model_layers::mtp_carry::mtp_carry_debug() {
                tracing::info!(
                    "MTP_CARRY store: rows={rows} last_pair_key={last_pair_key:?} \
                     seq_tokens={}",
                    seq.tokens.len(),
                );
            }
        }

        if let Some(ref proposer) = self.proposer
            && let Some(ref mut pstate) = seq.proposer_state
        {
            proposer.free_state(self.gpu.as_ref(), pstate.as_mut())?;
        }

        self.free_chunked_prefill_meta(seq)?;

        // 2026-09-25: `METRALE_SEQ_MEMTRACE`: the closing half of this sequence's memory
        // bracket, after everything it owns has been released.
        crate::model::seq_memtrace::trace(self.gpu.as_ref(), "free");

        Ok(())
    }

    /// 2026-10-03: Free KV blocks, capped (lever `METRALE_DSA_INDEXER_LAZY` on) by the blocks'
    /// worth of tokens the DSA indexer pool can still give a new sequence, so prefill
    /// admission, swap-in and preemption-resume wait for indexer room as they do for KV room.
    pub(super) fn num_free_blocks_dispatch(&self) -> usize {
        let kv = self.kv_cache.lock();
        let free = kv.num_free_blocks();
        match metrale_model_arch::glm5next_dsa::lazy::pool_free_tokens() {
            Some(tokens) => free.min(tokens / kv.config().block_size.max(1)),
            None => free,
        }
    }

    /// 2026-10-03: Total KV blocks, capped (lever `METRALE_DSA_INDEXER_LAZY` on) by the blocks'
    /// worth of tokens the whole DSA indexer pool holds, so admission's reservation and the
    /// preemption-resume "can never fit" check see the indexer limit too.
    pub(super) fn num_total_blocks_dispatch(&self) -> usize {
        let kv = self.kv_cache.lock();
        let total = kv.num_blocks();
        match metrale_model_arch::glm5next_dsa::lazy::pool_total_tokens() {
            Some(tokens) => total.min(tokens / kv.config().block_size.max(1)),
            None => total,
        }
    }

    pub(super) fn reclaim_prefix_blocks_dispatch(&self, num_blocks: usize) -> usize {
        if num_blocks == 0 || !self.prefix_cache.is_active() {
            return 0;
        }
        let evicted = self.prefix_cache.evict(num_blocks);
        if evicted.is_empty() {
            return 0;
        }
        let mut kv = self.kv_cache.lock();
        let before = kv.num_free_blocks();
        super::super::block_mgmt::apply_evicted_blocks(evicted, &mut kv);
        kv.num_free_blocks().saturating_sub(before)
    }
}
