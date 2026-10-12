// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Wire helpers of the EP/TP command protocol: the rank-0
//! broadcasts (`ep_broadcast_*`), the worker's `ep_recv_seq_and_cmd`, and
//! `ep_min_u32`.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    /// 2026-09-25: Send `tokens` from rank 0 to every rank in one broadcast.
    /// Rank 0 returns its input; the other ranks pass a vector of the same
    /// length and get rank 0's values. Without a comm backend it returns
    /// `tokens` unchanged. Errors when the payload exceeds the scratch buffer.
    pub(super) fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        let n = tokens.len();
        if self.comm.is_none() {
            return Ok(tokens.to_vec());
        }
        let comm = self.comm.as_ref().unwrap();
        let byte_len = n * 4;
        let stream = self.gpu.default_stream();

        // 2026-09-25: The scratch buffer is the device staging. It is not sized
        // from the prompt length, so a long prompt's payload can exceed it:
        // bound-check before the copy and the broadcast rather than overrun
        // adjacent device buffers.
        let scratch_bytes = self.buffers.sizes().scratch;
        if byte_len > scratch_bytes {
            anyhow::bail!(
                "ep_broadcast_tokens: token payload {byte_len} bytes (n={n}) \
                 exceeds scratch capacity {scratch_bytes} bytes",
            );
        }
        let dev_buf = self.buffers.scratch();

        if comm.rank() == 0 {
            // 2026-09-25: SAFETY: `byte_len = n * 4` and `n = tokens.len()`, so the
            // reinterpreted span is exactly `tokens.len() * size_of::<u32>()`
            // bytes, the whole of `tokens`. `tokens: &[u32]` is a live shared
            // borrow, so every byte is initialised and no `&mut` to it can exist.
            // u8 has alignment 1, so the cast cannot under-align.
            let token_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, byte_len) };
            self.gpu.copy_h2d(token_bytes, dev_buf)?;
        }

        comm.broadcast(dev_buf.0, byte_len, 0)?;

        if comm.rank() != 0 {
            self.gpu.synchronize(stream)?;
            let mut result = vec![0u32; n];
            // 2026-09-25: SAFETY: `result` was just built by `vec![0u32; n]`, so
            // its length is `n` and every element is initialised; `byte_len =
            // n * 4 = result.len() * size_of::<u32>()`, so the span is exactly the
            // Vec's buffer. `result_bytes` is the only reference derived from
            // `result` while it is live (the next use of `result` is the
            // `Ok(result)` move), so the `&mut` is unaliased.
            let result_bytes =
                unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, byte_len) };
            self.gpu.copy_d2h(dev_buf, result_bytes)?;
            Ok(result)
        } else {
            Ok(tokens.to_vec())
        }
    }

    /// 2026-09-25: Minimum of `val` over all ranks, via `ep_gather_u32` (one
    /// broadcast rooted at each rank); the comm trait's all-reduce only sums.
    /// The prefix-cache lookup uses it so head and worker agree on
    /// `matched_tokens` when their local caches disagree. Every rank must call
    /// it at the same point, or the broadcasts have no partner.
    pub(super) fn ep_min_u32(&self, val: u32) -> Result<u32> {
        let votes = self.ep_gather_u32(val)?;
        Ok(votes.into_iter().min().unwrap_or(val))
    }

    /// 2026-09-25: Whether the head-worker command protocol is live: a comm
    /// backend exists and the EP or the TP world is larger than 1, so pure TP
    /// counts as well as EP.
    pub(crate) fn multi_rank_protocol_active(&self) -> bool {
        self.comm.is_some() && (self.config.ep_world_size > 1 || self.config.tp_world_size > 1)
    }

    /// 2026-09-25: Broadcast a `(seq_id, cmd)` pair from rank 0: the `seq_id`
    /// preamble only when `v2`, then `cmd`. Workers read it with
    /// [`Self::ep_recv_seq_and_cmd`]. A no-op unless
    /// `multi_rank_protocol_active()`, so single-GPU callers may call it
    /// unconditionally (`ep_broadcast_u32` panics without a comm backend).
    ///
    /// Both ranks must agree on `v2`: a worker expecting the other shape reads
    /// the next u32 as the wrong field.
    pub(super) fn ep_broadcast_seq_and_cmd(&self, seq_id: u32, cmd: u32, v2: bool) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        if v2 {
            self.ep_broadcast_u32(seq_id)?;
        }
        self.ep_broadcast_u32(cmd)?;
        Ok(())
    }

    /// 2026-09-25: Broadcast a batched decode step (`0xFFFFFFE0`) from rank 0:
    ///
    /// ```text
    /// preamble seq_id = 0  (ignored — cmd routes the whole batch)
    /// cmd = 0xFFFFFFE0
    /// N (u32)
    /// seq_ids[N]  (one bulk broadcast)
    /// tokens[N]   (one bulk broadcast)
    /// ```
    ///
    /// The worker's matching receive is `ep_worker_decode_batch`, and both ranks
    /// then run `decode_batch_compute_main`, so their per-layer collectives
    /// pair up. A no-op unless `multi_rank_protocol_active()`.
    ///
    /// Requires `ep_protocol_v2`: without the preamble the worker would read
    /// the `seq_id` word as the command. Only a debug assertion checks it.
    pub(super) fn ep_broadcast_decode_batch_dispatch(
        &self,
        seq_ids: &[u32],
        tokens: &[u32],
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        debug_assert!(
            self.ep_protocol_v2,
            "ep_broadcast_decode_batch_dispatch called without METRALE_EP_PROTOCOL=v2"
        );
        debug_assert_eq!(
            seq_ids.len(),
            tokens.len(),
            "seq_ids and tokens length mismatch"
        );
        self.ep_broadcast_seq_and_cmd(0, 0xFFFFFFE0, true)?;
        self.ep_broadcast_u32(seq_ids.len() as u32)?;
        self.ep_broadcast_tokens(seq_ids)?;
        self.ep_broadcast_tokens(tokens)?;
        Ok(())
    }

    /// 2026-09-25: Receive a `(seq_id, cmd)` pair from rank 0; the worker side
    /// of [`Self::ep_broadcast_seq_and_cmd`]. With `v2` the `seq_id` is the
    /// slot to run the command in; without it the `seq_id` is always 0.
    pub(super) fn ep_recv_seq_and_cmd(&self, v2: bool) -> Result<(u32, u32)> {
        // 2026-09-25: Only the first word waits through server idle time. Once
        // it arrives the command is in flight, and the remaining words use the
        // bounded `broadcast`.
        let comm = self
            .comm
            .as_ref()
            .expect("worker command receive without comm");
        comm.recv_command_u32(self.ep_cmd_buf.0, 0)?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        let mut buf = [0u8; 4];
        self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
        let first = u32::from_le_bytes(buf);
        if v2 {
            Ok((first, self.ep_broadcast_u32(0)?))
        } else {
            Ok((0, first))
        }
    }

    /// 2026-10-12: Broadcast `vals` from rank 0 as ONE packet through `ep_cmd_buf` (at most
    /// `EP_PACK_MAX_WORDS` words). Rank 0 returns `vals`; the other ranks pass a slice of the same
    /// length and get rank 0's values. Panics without a comm backend.
    pub(super) fn ep_broadcast_packed(&self, vals: &[u32]) -> Result<Vec<u32>> {
        let n = vals.len();
        anyhow::ensure!(
            (1..=super::ep_c1_pack::EP_PACK_MAX_WORDS).contains(&n),
            "ep_broadcast_packed: {n} words outside 1..={}",
            super::ep_c1_pack::EP_PACK_MAX_WORDS
        );
        let comm = self.comm.as_ref().expect("ep_broadcast_packed without comm");
        let stream = self.gpu.default_stream();
        let mut bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        if comm.rank() == 0 {
            self.gpu.copy_h2d(&bytes, self.ep_cmd_buf)?;
            comm.broadcast(self.ep_cmd_buf.0, 4 * n, 0)?;
            Ok(vals.to_vec())
        } else {
            comm.broadcast(self.ep_cmd_buf.0, 4 * n, 0)?;
            self.gpu.synchronize(stream)?;
            self.gpu.copy_d2h(self.ep_cmd_buf, &mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
    }

    /// 2026-10-12: A command's follow-up words: one packet under `METRALE_EP_C1_PACK=1`, else one
    /// `ep_broadcast_u32` per word (the wire both ranks used before). Rank 0 passes the values; the
    /// other ranks pass placeholders of the same length and get rank 0's values.
    pub(super) fn ep_broadcast_cmd_words(&self, vals: &[u32]) -> Result<Vec<u32>> {
        if super::ep_c1_pack::ep_c1_pack() {
            return self.ep_broadcast_packed(vals);
        }
        vals.iter().map(|&v| self.ep_broadcast_u32(v)).collect()
    }

    /// 2026-09-25: Broadcast one u32 from rank 0 through `ep_cmd_buf`. Rank 0
    /// returns `val`; the other ranks return the received value. Panics without
    /// a comm backend.
    pub(super) fn ep_broadcast_u32(&self, val: u32) -> Result<u32> {
        let comm = self.comm.as_ref().expect("ep_broadcast_u32 without comm");
        let stream = self.gpu.default_stream();
        if comm.rank() == 0 {
            self.gpu.copy_h2d(&val.to_le_bytes(), self.ep_cmd_buf)?;
            comm.broadcast(self.ep_cmd_buf.0, 4, 0)?;
            Ok(val)
        } else {
            comm.broadcast(self.ep_cmd_buf.0, 4, 0)?;
            self.gpu.synchronize(stream)?;
            let mut buf = [0u8; 4];
            self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
            Ok(u32::from_le_bytes(buf))
        }
    }
}
