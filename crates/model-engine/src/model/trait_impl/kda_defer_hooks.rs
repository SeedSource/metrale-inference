// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The model side of the deferred KDA commit (`METRALE_GLM_KDA_DEFER_COMMIT=1`;
//! bookkeeping in `model/kda_defer.rs`, the verify's row 0 in model-arch
//! `glm5next_kda/snap_fuse/defer.rs`).
//!
//! Owner: model-engine (SSM state).
//! Invariants:
//! - Capability is computed once: every linear-attention layer answers `kda_defer_commit_ready`
//!   (the levers on, the kernels resolved), the pool keeps FP32 per-token intermediates, and the
//!   pool's layer count matches. Not capable (the lever off included): every hook here is a
//!   no-op, nothing is ever pending, and the commit copies exactly as before.
//! - Only `commit_accepted_prefix_dispatch` creates a deferral, only for a slot whose last verify
//!   was a consumer (`DeferBook::arm`), only for `0 < a < k` with no GDN fold.
//! - A consumer is a single-sequence verify of `k >= 2` rows (`decode_verify_graphed`, `_k3`,
//!   `_k4`, `_kgamma`, `decode_and_verify_fused`, on both ranks): every KDA layer of the slot gets
//!   its `SsmDeferSrc` and the device word is synced to the pending index first.
//! - Every other entry point that can read or write a sequence's live state calls
//!   [`TransformerModel::kda_defer_flush`] first, next to `gdn_carry_flush_pending` (the same set
//!   of `Model` entries), and the EP worker flushes before every command except the
//!   single-sequence verifies (their wrappers decide) and the drafter proposals (no KDA state).
//! - A flush is today's commit copy (`kda_defer_plans`), on `secondary_stream`, then the default
//!   stream (and the caller's) waits for it, as after `sync_secondary`.

use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{LayerWriteOnAccept, SsmDeferSrc, SsmLayerState};

use super::super::kda_defer::{self, CONSUMED, DEFERRED, MATERIALISED};
use super::super::ssm_batched_copy::{StateCopy, run_ssm_state_copies};
use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-10-09: EP worker commands that must not materialise a deferred commit first: the
/// single-sequence verifies (their `Model` wrappers handle the slot) and the drafter proposals,
/// which never touch KDA state and run between every commit and the next verify.
pub(in crate::model) fn worker_cmd_keeps_deferral(cmd: u32) -> bool {
    matches!(cmd, 0xFFFF_FFF2..=0xFFFF_FFF4)
        || cmd == metrale_model_layers::speculative::glm_dflash::EP_CMD_VERIFY_KGAMMA
        || cmd == metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE
        || cmd == super::propose_batch_ep::EP_CMD_MTP_PROPOSE_BATCH
}

impl TransformerModel {
    /// 2026-10-09: Whether commits may be deferred on this model (module header). Computed once.
    pub(in crate::model) fn kda_defer_capable(&self) -> bool {
        *self.ssm_pool.kda_defer.capable.get_or_init(|| {
            let kda: Vec<usize> = (0..self.layers.len())
                .filter(|&i| self.config.layer_type(i) == LayerType::LinearAttention)
                .collect();
            let ready = kda
                .iter()
                .filter(|&&i| self.layers[i].kda_defer_commit_ready())
                .count();
            if ready == 0 {
                return false;
            }
            let p = &self.ssm_pool;
            // 2026-10-12: FP32 intermediates, or FP16 ones under `METRALE_GLM_KDA_STATE_F16=1`
            // (the `_defer_f16` twin reads them; glm5next_kda/state_f16.rs).
            let width_ok = p.h_stored_bytes == p.h_bytes
                || (metrale_model_layers::ssm_reserve::glm_kda_state_f16_for(
                    &self.config.model_type,
                ) && p.h_stored_bytes * 2 == p.h_bytes);
            let pool_ok = !p.h_intermediate_pools.is_empty()
                && width_ok
                && p.conv_bytes == self.kda_defer_conv_bytes()
                && p.num_ssm_layers == kda.len()
                && p.kda_defer.book.lock().slots() > 0;
            let ok = ready == kda.len() && pool_ok;
            if ok {
                tracing::info!(
                    "METRALE_GLM_KDA_DEFER_COMMIT: capable ({} KDA layers, {} slots)",
                    kda.len(),
                    p.kda_defer.book.lock().slots()
                );
            } else {
                tracing::warn!(
                    "METRALE_GLM_KDA_DEFER_COMMIT ignored: {ready}/{} KDA layers ready, pool \
                     ok={pool_ok} (FP32 intermediates, or FP16 under METRALE_GLM_KDA_STATE_F16=1, \
                     needed); every commit keeps its copy",
                    kda.len()
                );
            }
            ok
        })
    }

    /// 2026-10-09: The conv bytes `commit_accepted_prefix_dispatch` copies per layer.
    fn kda_defer_conv_bytes(&self) -> usize {
        let c = &self.config;
        (c.linear_num_key_heads * c.linear_key_head_dim * 2
            + c.linear_num_value_heads * c.linear_value_head_dim)
            * c.linear_conv_kernel_dim
            * 4
    }

    /// 2026-10-09: The plans `commit_accepted_prefix_dispatch` builds for a partial accept of
    /// `idx + 1` rows of `slot` with no GDN fold: intermediate `idx` into the live state of every
    /// SSM layer, in layer order. The live state is the pool's slot (`SsmLayerState::h_state` is
    /// set from `ssm_pool.h_state(l, slot)` at alloc and compaction).
    fn kda_defer_plans(&self, slot: usize, idx: usize) -> (Vec<StateCopy>, Vec<StateCopy>) {
        let p = &self.ssm_pool;
        let n = p.num_ssm_layers;
        let h = (0..n)
            .map(|l| StateCopy {
                src: p.h_intermediate(l, slot, idx),
                dst: p.h_state(l, slot),
                bytes: p.h_stored_bytes,
            })
            .collect();
        let conv = (0..n)
            .map(|l| StateCopy {
                src: p.conv_intermediate(l, slot, idx),
                dst: p.conv_state(l, slot),
                bytes: self.kda_defer_conv_bytes(),
            })
            .collect();
        (h, conv)
    }

    /// 2026-10-09: Materialise every pending deferral: today's commit copy, then the default stream
    /// and `stream` wait for it. A no-op (one lock) when nothing is pending.
    pub(in crate::model) fn kda_defer_flush(&self, stream: u64) -> Result<()> {
        self.kda_defer_flush_except(None, stream)
    }

    fn kda_defer_flush_except(&self, keep: Option<usize>, stream: u64) -> Result<()> {
        let todo = {
            let mut b = self.ssm_pool.kda_defer.book.lock();
            if !b.any_pending() {
                return Ok(());
            }
            b.take_all_except(keep)
        };
        if todo.is_empty() {
            return Ok(());
        }
        for &(slot, idx) in &todo {
            let (h, conv) = self.kda_defer_plans(slot, idx);
            run_ssm_state_copies(self.gpu.as_ref(), &h, &conv, self.secondary_stream)?;
        }
        self.gpu
            .record_event(self.secondary_event, self.secondary_stream)?;
        let d = self.gpu.default_stream();
        self.gpu.stream_wait_event(d, self.secondary_event)?;
        if stream != d {
            self.gpu.stream_wait_event(stream, self.secondary_event)?;
        }
        MATERIALISED.fetch_add(todo.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// 2026-10-09: The device words, allocated (`-1` filled) on first use, before any capture.
    fn kda_defer_words(&self) -> Result<DevicePtr> {
        let mut d = self.ssm_pool.kda_defer.dev.lock();
        if d.0 == 0 {
            let bytes = self.ssm_pool.kda_defer.book.lock().slots() * 4;
            let p = self.gpu.alloc(bytes)?;
            self.gpu.memset(p, 0xFF, bytes)?;
            *d = p;
        }
        Ok(*d)
    }

    /// 2026-10-09: Write `v` into `slot`'s device word on the default stream, ahead of the verify.
    fn kda_defer_write_word(&self, words: DevicePtr, slot: usize, v: i32) -> Result<()> {
        self.gpu.copy_h2d_async(
            &v.to_le_bytes(),
            words.offset(slot * 4),
            self.gpu.default_stream(),
        )
    }

    /// 2026-10-09: Before a single-sequence verify of `k` rows: flush every other slot; then either
    /// set every KDA layer's `SsmDeferSrc` (a consumer, `k >= 2`) or flush this slot too; sync the
    /// slot's device word to its pending index. Returns whether this verify is a consumer.
    pub(in crate::model) fn kda_defer_pre_verify(
        &self,
        seq: &mut SequenceState,
        k: usize,
    ) -> Result<bool> {
        let stream = self.gpu.default_stream();
        if !self.kda_defer_capable() {
            return Ok(false);
        }
        let slot = seq.slot_idx;
        let covered = self.ssm_pool.kda_defer.book.lock().covers(slot);
        let consumer = covered && k >= 2;
        self.kda_defer_flush_except(consumer.then_some(slot), stream)?;
        let words = if covered {
            self.kda_defer_words()?
        } else {
            DevicePtr::NULL
        };
        let (write, pending) = {
            let mut b = self.ssm_pool.kda_defer.book.lock();
            if !consumer {
                b.disarm(slot);
            }
            (b.sync_dev(slot), b.pending(slot))
        };
        if let Some(v) = write {
            self.kda_defer_write_word(words, slot, v)?;
        }
        let mut ssm_layer = 0usize;
        for (i, ls) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = ls
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            ssm.kda_defer = consumer.then(|| SsmDeferSrc {
                idx: words.offset(slot * 4),
                h_base: self.ssm_pool.h_intermediate(ssm_layer, slot, 0),
                h_stride: self.ssm_pool.h_stored_bytes,
                conv_base: self.ssm_pool.conv_intermediate(ssm_layer, slot, 0),
                conv_stride: self.ssm_pool.conv_bytes,
            });
            ssm_layer += 1;
        }
        if consumer
            && let Some(idx) = pending
            && CONSUMED.fetch_add(1, Ordering::Relaxed) == 0
        {
            tracing::info!(
                "METRALE_GLM_KDA_DEFER_COMMIT: ENGAGED (first verify reading a deferred \
                 commit: slot {slot}, intermediate {idx}, {ssm_layer} KDA layers, k={k})"
            );
        }
        Ok(consumer)
    }

    /// 2026-10-09: Before a batched verify or any EP worker command other than a single-sequence
    /// verify or a proposal: materialise everything and disarm every slot, so the next commits
    /// copy as before (a commit defers only right after its own consumer verify).
    pub(in crate::model) fn kda_defer_flush_disarm(&self, stream: u64) -> Result<()> {
        if !self.kda_defer_capable() {
            return Ok(());
        }
        self.kda_defer_flush(stream)?;
        self.ssm_pool.kda_defer.book.lock().disarm_all();
        Ok(())
    }

    /// 2026-10-09: After a consumer verify returned `Ok`: its row 0 read the pending state, so the
    /// live state is current and its commit may defer.
    pub(in crate::model) fn kda_defer_post_verify(&self, seq: &SequenceState, consumer: bool) {
        if consumer {
            let mut b = self.ssm_pool.kda_defer.book.lock();
            b.clear(seq.slot_idx);
            b.arm(seq.slot_idx);
        }
    }

    /// 2026-10-09: The commit's decision. `Ok(true)`: the partial accept is deferred and nothing
    /// is copied. `Ok(false)`: today's commit runs. Refuses a commit that finds its slot still
    /// pending: a verify ran without consuming or flushing it, so the live state is wrong.
    pub(in crate::model) fn kda_defer_commit(
        &self,
        seq: &SequenceState,
        num_accepted: usize,
        k: usize,
        h_folded: bool,
    ) -> Result<bool> {
        if !self.kda_defer_capable() {
            return Ok(false);
        }
        let slot = seq.slot_idx;
        let mut b = self.ssm_pool.kda_defer.book.lock();
        if let Some(idx) = b.pending(slot) {
            bail!(
                "METRALE_GLM_KDA_DEFER_COMMIT: slot {slot} still has deferred intermediate {idx} \
                 at its next commit; a verify ran without the deferred-commit hook"
            );
        }
        let armed = b.take_armed(slot);
        kda_defer::count_commit();
        let idx = num_accepted.wrapping_sub(1);
        if !armed
            || h_folded
            || num_accepted == 0
            || num_accepted >= k
            || !b.covers(slot)
            || idx >= self.ssm_pool.h_inter_count(slot)
        {
            return Ok(false);
        }
        b.defer(slot, idx);
        if DEFERRED.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::info!(
                "METRALE_GLM_KDA_DEFER_COMMIT: ENGAGED (first deferred commit: slot {slot}, \
                 {num_accepted}/{k} rows, intermediate {idx}; the state copy is skipped)"
            );
        }
        Ok(true)
    }

    /// 2026-10-09: Slot reuse (alloc, free, compaction): nothing pending, device word `-1`.
    pub(in crate::model) fn kda_defer_reset_slot(&self, slot: usize) -> Result<()> {
        let write = self.ssm_pool.kda_defer.book.lock().reset_slot(slot);
        let words = *self.ssm_pool.kda_defer.dev.lock();
        match write {
            Some(v) if words.0 != 0 => self.kda_defer_write_word(words, slot, v),
            _ => Ok(()),
        }
    }
}
