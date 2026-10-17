// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `BlockDiffusionDraftHead::propose_drafts`, the body behind
//! `DraftProposer::propose` and the per-sequence prep of `propose_batch`.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - The ctx append pushes one `ctx_positions` entry per `ctx_len` increment.
//! - A sequence's drafter block table is allocated only while its
//!   `block_table_dev` is `None`. When the pool runs out mid-allocation, the
//!   blocks already taken are freed before the error returns.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{BlockDiffusionDraftHead, DflashProposerState};
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::ProposerState;

impl BlockDiffusionDraftHead {
    pub(super) fn propose_drafts(
        &self,
        last_token: u32,
        _target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        _stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
        // 2026-09-25: `Some`: run only the per-sequence prep (ctx append,
        // drafter block allocation, and the ctx precompute unless the batch
        // defers it), push `(block_table_dev, ctx_count)` and return an empty
        // Vec without a forward. `propose_batch` uses this, so both paths run
        // one copy of the prep.
        collect_prep: Option<&mut Vec<(DevicePtr, u32)>>,
    ) -> Result<Vec<u32>> {
        // 2026-09-25: The block width is the draft count plus the anchor row,
        // clamped by `set_block_g` to the head's gamma. `propose_batch` sets
        // the same value before its prep loop.
        self.set_block_g(num_drafts);
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;

        // 2026-09-25: `METRALE_DFLASH_CTX_PARITY_DUMP=1`, one-shot per
        // `ctx.stats`: the whole ctx accumulator (`ctx_len` rows of
        // `ctx_slot_bytes`) to `/tmp/metrale_ctx_parity.bin`, and its shape to
        // `/tmp/metrale_ctx_parity.json`.
        {
            if self.levers.ctx_parity_dump
                && dstate.ctx_len > 0
                && ctx.stats.dumped.keyed("dflash_ctx_parity")
            {
                let n_bytes = dstate.ctx_len * dstate.ctx_slot_bytes;
                let mut buf = vec![0u8; n_bytes];
                ctx.gpu.synchronize(_stream)?;
                ctx.gpu.copy_d2h(dstate.ctx_hidden_acc, &mut buf)?;
                match std::fs::write("/tmp/metrale_ctx_parity.bin", &buf) {
                    Ok(()) => {
                        let elems_per_slot = dstate.ctx_slot_bytes / 2;
                        let meta = format!(
                            "{{\"ctx_len\":{},\"ctx_slot_bytes\":{},\"elems_per_slot\":{},\"position\":{},\"last_token\":{},\"n_bytes\":{}}}",
                            dstate.ctx_len,
                            dstate.ctx_slot_bytes,
                            elems_per_slot,
                            position,
                            last_token,
                            n_bytes,
                        );
                        let _ = std::fs::write("/tmp/metrale_ctx_parity.json", meta);
                        tracing::info!(
                            "DFLASH CTX_PARITY: wrote {} bytes — ctx_len={} slots × {} BF16 elems/slot (position={}, last_token={}) to /tmp/metrale_ctx_parity.bin",
                            n_bytes,
                            dstate.ctx_len,
                            dstate.ctx_slot_bytes / 2,
                            position,
                            last_token,
                        );
                    }
                    Err(e) => {
                        tracing::warn!("DFLASH CTX_PARITY: write failed: {e}");
                    }
                }
            }
        }

        let _ = (ctx, position, last_token);

        // 2026-09-25: Append the latest captured target hiddens as one ctx row,
        // unless `METRALE_DFLASH_DEBUG_NO_DECODE_APPEND=1`, there is no capture,
        // the accumulator is full, or the model already appended this capture
        // (`skip_next_decode_append`, consumed here).
        let eagle_skip = dstate.skip_next_decode_append;
        dstate.skip_next_decode_append = false;
        let skip_decode_append = self.levers.no_decode_append;
        if !skip_decode_append
            && !eagle_skip
            && let Some(latest_ctx) = target_hidden_stack
            && dstate.ctx_len < dstate.max_ctx_len
        {
            let dst_offset = dstate.ctx_len * dstate.ctx_slot_bytes;
            ctx.gpu.copy_d2d_async(
                latest_ctx,
                dstate.ctx_hidden_acc.offset(dst_offset),
                dstate.ctx_slot_bytes,
                _stream,
            )?;
            // 2026-09-25: The row's RoPE position, stamped once as
            // `position - 1`.
            debug_assert_eq!(dstate.ctx_positions.len(), dstate.ctx_len);
            dstate.ctx_positions.push(position.saturating_sub(1) as i32);
            dstate.ctx_len += 1;
        }

        // 2026-09-25: The paged drafter cache (`METRALE_DFLASH_OPTION_B`, on
        // unless `0`). A sequence's block table is allocated on its first
        // propose, for `max_ctx_len + gamma + 1` slots. `BLOCK_SIZE` is the
        // drafter cache's block size (`from_weights.rs`).
        let option_b_enabled = self.levers.option_b;
        let option_b_arg: Option<(DevicePtr, u32)> = if option_b_enabled {
            const BLOCK_SIZE: usize = 16;
            self.ensure_block_table(dstate, ctx.gpu)?;
            // 2026-09-25: Only the tail `[ctx_committed, ctx_len)` needs K/V:
            // each committed row was roped at its own stamped position, so it
            // stays valid as `position` moves.
            let force_full = self.levers.full_precompute;
            let committed = if force_full {
                0
            } else {
                dstate.ctx_committed.min(dstate.ctx_len)
            };
            let new_count = dstate.ctx_len - committed;
            // 2026-09-25: In a batched prep with batched precompute on,
            // `precompute_ctx_kv_batched` covers this tail after every
            // sequence's prep; `ctx_committed` stays put so that pass sees it.
            let defer_to_batch = collect_prep.is_some() && super::batched_precompute_enabled();
            if !defer_to_batch && dstate.ctx_len > 0 && new_count > 0 {
                // 2026-09-25: The precompute scratch holds `ctx_window` rows
                // and the tail can be longer, so it runs in chunks of at most
                // `ctx_window` rows. The paged cache covers every slot up to
                // `max_ctx_len + gamma + 1`, so only the scratch bounds a chunk.
                anyhow::ensure!(
                    self.ctx_window > 0,
                    "DFlash precompute: ctx_window=0 but ctx tail of {} slots \
                     needs precompute — scratch has no capacity",
                    new_count,
                );
                let slot_mapping = &self.scratch.slot_mapping_dev;
                let mut chunk_start = committed;
                while chunk_start < dstate.ctx_len {
                    let chunk_count = (dstate.ctx_len - chunk_start).min(self.ctx_window);
                    metrale_model_layers::layers::ops::fill_slots_from_block_table(
                        ctx.gpu,
                        self.kernels.fill_slots,
                        *slot_mapping,
                        dstate.block_table_dev.unwrap(),
                        chunk_start as u32,
                        chunk_count as u32,
                        BLOCK_SIZE as u32,
                        _stream,
                    )?;
                    let slot_positions =
                        &dstate.ctx_positions[chunk_start..chunk_start + chunk_count];
                    self.precompute_ctx_kv(
                        dstate.ctx_hidden_acc,
                        chunk_start,
                        chunk_count,
                        slot_positions,
                        *slot_mapping,
                        ctx,
                        _stream,
                        true,
                    )?;
                    chunk_start += chunk_count;
                }
                dstate.ctx_committed = dstate.ctx_len;
            }
            dstate.ctx_count_drafter = dstate.ctx_len;
            // 2026-09-25: `METRALE_DFLASH_CTXLEN_PROBE=1`: warn at the first
            // `ctx_positions` pair that is not strictly increasing.
            if self.levers.ctxlen_probe
                && let Some(i) = dstate.ctx_positions.windows(2).position(|w| w[1] <= w[0])
            {
                tracing::warn!(
                    "DFLASH CTX_POSITIONS VIOLATION: slot {} pos {} -> slot {} pos {} \
                     (not strictly increasing: double-append or seam hole)",
                    i,
                    dstate.ctx_positions[i],
                    i + 1,
                    dstate.ctx_positions[i + 1],
                );
            }
            // 2026-09-25: The same probe logs `ctx_len` against `position`
            // when `position` is a multiple of 16.
            if self.levers.ctxlen_probe && position.is_multiple_of(16) {
                tracing::info!(
                    "DFLASH CTXLEN_PROBE: position={} ctx_len={} q_offset(=ctx_len)={} GAP={} (position - ctx_len; healthy≈prompt_len, BUG if grows unbounded)",
                    position,
                    dstate.ctx_len,
                    dstate.ctx_len,
                    position.saturating_sub(dstate.ctx_len),
                );
            }
            let ablate_no_ctx = self.levers.option_b_no_ctx;
            let effective_ctx_count = if ablate_no_ctx {
                0
            } else {
                dstate.ctx_count_drafter as u32
            };
            Some((dstate.block_table_dev.unwrap(), effective_ctx_count))
        } else {
            None
        };

        if let Some(sink) = collect_prep {
            let arg = option_b_arg.ok_or_else(|| {
                anyhow::anyhow!(
                    "batched DFlash propose requires Option B (paged drafter KV); \
                     set METRALE_DFLASH_OPTION_B=1 or let the batched path decline"
                )
            })?;
            sink.push(arg);
            return Ok(Vec::new());
        }

        let drafts = self
            .forward_block(
                last_token,
                position,
                ctx,
                _stream,
                if dstate.ctx_len > 0 {
                    Some((dstate.ctx_hidden_acc, dstate.ctx_len))
                } else {
                    None
                },
                option_b_arg,
                None,
            )
            .map_err(|e| {
                tracing::warn!("DFlash forward_block failed, falling back to no-spec: {e:#}");
                e
            })?;
        let cap = self.levers.draft_cap.unwrap_or(self.block_g());

        if self.levers.verify_trace {
            tracing::info!(
                "DFLASH TRACE drafts: token_in={} position={} γ={} drafts_pre_cap={:?}",
                last_token,
                position,
                drafts.len(),
                drafts,
            );
        }

        // 2026-09-25: Row 0's input is `last_token` (the anchor), so the drafts
        // start at row 1. A drafter without a mask token (`mask_token_id == 0`)
        // keeps row 0.
        let drafts = if self.mask_token_id != 0 && drafts.len() > 1 {
            drafts[1..].to_vec()
        } else {
            drafts
        };

        let drafts = drafts.into_iter().take(cap).collect::<Vec<_>>();
        dstate.last_num_drafted = drafts.len();
        Ok(drafts)
    }
}
