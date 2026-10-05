// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The LM head projections: `lm_head` for one row and
//! `lm_head_batched` for K rows, with its wide arm and the vocab-parallel BF16
//! split.
//!
//! Owner: model-engine.
//! Invariants:
//! - Except on the Q6_K head, which returns right after its projection, a
//!   successful projection applies the token overlay and then, when a softcap
//!   kernel is loaded, the softcap.

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
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, MtpWeights, QuantizedWeight};

/// 2026-09-25: Whether the wide batched LM head arm runs. Setting
/// `METRALE_NO_LMHEAD_BATCHED_WIDE` to any value, `0` included, turns it off
/// and leaves those row counts on `w4a16_gemm`. Read once per process.
fn lmhead_batched_wide_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_LMHEAD_BATCHED_WIDE").is_err())
}

impl TransformerModel {
    /// 2026-09-25: Wide batched LM head for the batched-verify row counts,
    /// called for `num_tokens` in 9..=`VERIFY_ROW_CAP`. It tries the same kernels
    /// in the same order as the decode head (`trait_impl/lm_head_batched.rs`):
    /// the BF16-MMA twin GEMM, the twin tile GEMM, then for up to 16 rows the
    /// batch16 GEMV.
    ///
    /// Returns `false`, having launched nothing, when none applies (no NVFP4
    /// head, or no usable twin and more than 16 rows or no batch16 kernel).
    fn lm_head_batched_wide(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        let Some(ref nvfp4) = self.lm_head_nvfp4 else {
            return Ok(false);
        };
        if let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t {
            // 2026-09-25: The BF16-MMA variant (`METRALE_LMHEAD_LOSSLESS`) first,
            // as in the decode head.
            if self.w4a16_gemm_t_bf16_kernel.0 != 0 {
                ops::w4a16_gemm_n128_m128_bf16_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_bf16_kernel,
                    hidden,
                    nvfp4_t,
                    logits,
                    num_tokens,
                    v,
                    h,
                    ldb,
                    stream,
                )?;
                return Ok(true);
            }
            if self.w4a16_gemm_t_kernel.0 != 0 {
                ops::w4a16_gemm_n128_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_kernel,
                    hidden,
                    nvfp4_t,
                    logits,
                    num_tokens,
                    v,
                    h,
                    ldb,
                    stream,
                )?;
                return Ok(true);
            }
        }
        // 2026-09-25: Without a usable twin, up to 16 rows take the batch16
        // GEMV; wider row counts return `false` and keep `w4a16_gemm`.
        if num_tokens <= 16 && self.w4a16_gemv_batch16_kernel.0 != 0 {
            ops::w4a16_gemv_batchm(
                self.gpu.as_ref(),
                self.w4a16_gemv_batch16_kernel,
                hidden,
                nvfp4,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
            return Ok(true);
        }
        Ok(false)
    }

    /// 2026-09-25: LM head for K rows: `hidden[K, H]` → `logits_dst[K, V]`,
    /// which it returns.
    pub(super) fn lm_head_batched(
        &self,
        hidden: DevicePtr,
        num_tokens: u32,
        logits_dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        // 2026-09-25: The caller picks the destination, so co-dispatched
        // prefill streams can each write their own logits rows.
        let logits = logits_dst;
        if self.lm_head_q6k_run(hidden, num_tokens, logits, stream)? {
            return Ok(logits);
        }
        // 2026-09-27: Under a row-invariant tier policy an NVFP4 head takes the
        // wide arm's tile GEMM at every row count, the order the batched verify
        // uses from 9 rows.
        if metrale_model_layers::layers::row_invariant()
            && self.lm_head_nvfp4_t.is_some()
            && self.lm_head_batched_wide(hidden, num_tokens, logits, stream)?
        {
            return Ok(logits);
        }
        if self.lm_head_bf16_tc(hidden, num_tokens, logits, stream)?
            || self.lm_head_fp8_tc(hidden, num_tokens, logits, stream)?
        {
            // 2026-10-04: `METRALE_GLM_GEMV_TC=1` ran the BF16 head (`lm_head_bf16_tc`) or
            // the FP8 head (`lm_head_fp8_tc`) on the row-invariant tensor-core GEMV.
        } else if let Some(ref fp8) = self.lm_head_fp8 {
            // 2026-09-25: FP8 E4M3 head. Two rows use the dual GEMV; any other
            // row count, or a missing dual kernel, runs one GEMV per row.
            let bf16 = 2usize;
            if num_tokens == 2 && self.dense_gemv_fp8w_batch2_kernel.0 != 0 {
                ops::dense_gemv_fp8w_batch2(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_batch2_kernel,
                    hidden,
                    fp8,
                    logits,
                    v,
                    h,
                    stream,
                )?;
            } else {
                for i in 0..num_tokens as usize {
                    ops::dense_gemv_fp8w(
                        self.gpu.as_ref(),
                        self.dense_gemv_fp8w_kernel,
                        hidden.offset(i * h as usize * bf16),
                        fp8,
                        logits.offset(i * v as usize * bf16),
                        v,
                        h,
                        stream,
                    )?;
                }
            }
        } else if self.lm_head_nvfp4.is_none()
            && (2..=ops::DENSE_GEMV_BATCHM_DECODE_MAX_M).contains(&num_tokens)
            && self.dense_gemv_batchm_kernel.0 != 0
        {
            // 2026-09-25: BF16 head, 2..=`DENSE_GEMV_BATCHM_DECODE_MAX_M` rows: one
            // batched GEMV for every row, since the NVFP4 tiers below do not
            // cover a BF16 head.
            let (w, n, dst) = match self.lmhead_vocab_shard(v) {
                // 2026-09-25: Vocab-parallel, as in `lm_head`: this rank computes its
                // row range into a zeroed buffer, and the all-reduce below sums the
                // ranks' pieces.
                Some((begin, len)) => {
                    self.gpu.memset_async(
                        logits,
                        0,
                        num_tokens as usize * v as usize * 2,
                        stream,
                    )?;
                    (
                        metrale_model_layers::weight_map::DenseWeight {
                            weight: self.lm_head_weight.weight.offset(begin * h as usize * 2),
                        },
                        len as u32,
                        logits.offset(begin * 2),
                    )
                }
                None => (
                    metrale_model_layers::weight_map::DenseWeight {
                        weight: self.lm_head_weight.weight,
                    },
                    v,
                    logits,
                ),
            };
            ops::dense_gemv_batchm(
                self.gpu.as_ref(),
                self.dense_gemv_batchm_kernel,
                hidden,
                &w,
                dst,
                num_tokens,
                n,
                h,
                // 2026-09-25: Rows of `logits` are a full vocab apart even when this
                // rank writes a slice.
                v,
                stream,
            )?;
            if n != v
                && let Some(comm) = self.comm_ref()
            {
                comm.all_reduce_async(logits.0, num_tokens as usize * v as usize * 2, stream)?;
            }
        } else if num_tokens == 2 {
            if let Some(ref nvfp4) = self.lm_head_nvfp4 {
                ops::w4a16_gemv_batch2(
                    self.gpu.as_ref(),
                    self.w4a16_gemv_batch2_kernel,
                    hidden,
                    nvfp4,
                    logits,
                    v,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    hidden,
                    &self.lm_head_weight,
                    logits,
                    v,
                    h,
                    stream,
                )?;
                ops::dense_gemv(
                    self.gpu.as_ref(),
                    self.dense_gemv_kernel,
                    hidden.offset(h as usize * 2),
                    &self.lm_head_weight,
                    logits.offset(v as usize * 2),
                    v,
                    h,
                    stream,
                )?;
            }
        } else if (3..=ops::gemv_tc::narrow_gemv_max_rows()).contains(&num_tokens)
            && self.w4a16_batchm.kernel(num_tokens).0 != 0
            && let Some(ref nvfp4) = self.lm_head_nvfp4
        {
            // 2026-09-25: The narrowest loaded batched-GEMV tier that covers the
            // row count.
            ops::w4a16_gemv_batchm(
                self.gpu.as_ref(),
                self.w4a16_batchm.kernel(num_tokens),
                hidden,
                nvfp4,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        } else if (9..=super::trait_impl::verify_e2::VERIFY_ROW_CAP as u32).contains(&num_tokens)
            && lmhead_batched_wide_enabled()
            && self.lm_head_batched_wide(hidden, num_tokens, logits, stream)?
        {
            // 2026-09-25: The wide arm launched. When it returns `false` it
            // launched nothing and the chain continues to `w4a16_gemm`.
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            ops::w4a16_gemm(
                self.gpu.as_ref(),
                self.w4a16_gemm_kernel,
                hidden,
                nvfp4,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        }
        // 2026-09-25: Token overlay: overwrite the overridden logit columns
        // after the base projection and before the softcap. A null `seq_slot`
        // applies the active overlay slot to every row; a no-op without
        // overlays.
        self.apply_lmhead_overlay(hidden, DevicePtr(0), logits, num_tokens, false, stream)?;
        if self.logit_softcap_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            let total = num_tokens * v;
            self.apply_logit_softcap(logits, total, cap, stream)?;
        }
        Ok(logits)
    }

    /// 2026-10-04: `METRALE_GLM_GEMV_TC=1`: the BF16 head (no FP8 / NVFP4 / Q6_K head) for
    /// `rows` in 1..=16 on the row-invariant tensor-core GEMV (`ops::dense_gemv_tcm`): this
    /// rank's vocab slice into zeroed logits plus the all-reduce when `lmhead_vocab_shard`
    /// splits the vocab, else the whole vocab. A row's logits do not depend on `rows`, so the
    /// 1-row `lm_head` and `lm_head_batched` agree. Returns `false` having launched nothing
    /// when the lever is off, the head is not BF16, or the shape or entry does not route.
    fn lm_head_bf16_tc(
        &self,
        hidden: DevicePtr,
        rows: u32,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        if self.lm_head_fp8.is_some()
            || self.lm_head_nvfp4.is_some()
            || self.lm_head_q6k.is_some()
            || !ops::dense_gemv_tcm::ready(self.gpu.as_ref(), rows, v, h, false)
        {
            return Ok(false);
        }
        let Some((begin, len)) = self.lmhead_vocab_shard(v) else {
            return ops::dense_gemv_tcm::try_bf16(
                self.gpu.as_ref(),
                hidden,
                &self.lm_head_weight,
                logits,
                rows,
                v,
                h,
                v,
                stream,
            );
        };
        self.gpu
            .memset_async(logits, 0, rows as usize * v as usize * 2, stream)?;
        let launched = ops::dense_gemv_tcm::try_bf16(
            self.gpu.as_ref(),
            hidden,
            &DenseWeight {
                weight: self.lm_head_weight.weight.offset(begin * h as usize * 2),
            },
            logits.offset(begin * 2),
            rows,
            len as u32,
            h,
            // Rows of `logits` are a full vocab apart even when this rank writes a slice.
            v,
            stream,
        )?;
        anyhow::ensure!(
            launched,
            "lm_head_bf16_tc: the vocab slice [{begin}, +{len}) did not route"
        );
        if let Some(comm) = self.comm_ref() {
            comm.all_reduce_async(logits.0, rows as usize * v as usize * 2, stream)?;
        }
        Ok(true)
    }

    /// 2026-10-05: `METRALE_GLM_GEMV_TC=1`: the FP8 head (`--lm-head-dtype fp8`, `[V, H]`
    /// E4M3 plus one f32 scale per row, `Fp8DenseWeight`) for `rows` in 1..=16 on the
    /// row-invariant tensor-core FP8 GEMV (`ops::dense_gemv_tcm::try_fp8`), laid out as
    /// `lm_head_bf16_tc`: this rank's vocab slice (weight offset `begin * h` bytes, scale
    /// offset `begin * 4` bytes) into zeroed logits plus the all-reduce when
    /// `lmhead_vocab_shard` splits the vocab, else the whole vocab. A row's logits do not
    /// depend on `rows`, so the 1-row `lm_head` and `lm_head_batched` agree. Returns `false`
    /// having launched nothing when the lever is off, there is no FP8 head, a Q6_K head is
    /// installed, or the shape or entry does not route.
    fn lm_head_fp8_tc(
        &self,
        hidden: DevicePtr,
        rows: u32,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        let Some(ref fp8) = self.lm_head_fp8 else {
            return Ok(false);
        };
        if self.lm_head_q6k.is_some()
            || !ops::dense_gemv_tcm::ready(self.gpu.as_ref(), rows, v, h, true)
        {
            return Ok(false);
        }
        let Some((begin, len)) = self.lmhead_vocab_shard(v) else {
            return ops::dense_gemv_tcm::try_fp8(
                self.gpu.as_ref(),
                hidden,
                fp8,
                logits,
                rows,
                v,
                h,
                v,
                stream,
            );
        };
        self.gpu
            .memset_async(logits, 0, rows as usize * v as usize * 2, stream)?;
        let launched = ops::dense_gemv_tcm::try_fp8(
            self.gpu.as_ref(),
            hidden,
            &Fp8DenseWeight {
                // 1 byte per FP8 element, 4 per f32 row scale.
                weight: fp8.weight.offset(begin * h as usize),
                row_scale: fp8.row_scale.offset(begin * 4),
            },
            logits.offset(begin * 2),
            rows,
            len as u32,
            h,
            // Rows of `logits` are a full vocab apart even when this rank writes a slice.
            v,
            stream,
        )?;
        anyhow::ensure!(
            launched,
            "lm_head_fp8_tc: the vocab slice [{begin}, +{len}) did not route"
        );
        if let Some(comm) = self.comm_ref() {
            comm.all_reduce_async(logits.0, rows as usize * v as usize * 2, stream)?;
        }
        Ok(true)
    }

    /// 2026-09-25: `(begin, len)` rows of the BF16 LM head this rank computes,
    /// or `None` for the whole-vocab projection.
    ///
    /// `None` unless there is a communicator with at least 2 ranks and the vocab
    /// divides evenly by the world size, or when `METRALE_NO_LMHEAD_VOCAB_TP=1`
    /// (read once per process). Only the BF16 dense-head paths call it.
    fn lmhead_vocab_shard(&self, v: u32) -> Option<(usize, usize)> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("METRALE_NO_LMHEAD_VOCAB_TP").as_deref() == Ok("1")) {
            return None;
        }
        let comm = self.comm_ref()?;
        let ws = comm.world_size();
        if ws < 2 || !(v as usize).is_multiple_of(ws) {
            return None;
        }
        let len = v as usize / ws;
        Some((comm.rank() * len, len))
    }

    pub(super) fn lm_head(&self, hidden: DevicePtr, stream: u64) -> Result<DevicePtr> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        // 2026-09-25: The FP32 scratch when `use_fp32_logits`, else the shared
        // BF16 buffer; readers take the dtype from `decode_logits_fp32`.
        let (logits, fp32) = if self.use_fp32_logits {
            (self.logits_fp32_buf, true)
        } else {
            (self.buffers.logits(), false)
        };
        if self.lm_head_q6k.is_some() {
            // 2026-09-25: The Q6_K head, installed when the checkpoint stores
            // `lm_head.weight` as Q6_K. It writes BF16 logits only, so an FP32
            // destination is an error.
            anyhow::ensure!(!fp32, "Q6_K lm_head has no FP32-logits variant");
            self.lm_head_q6k_run(hidden, 1, logits, stream)?;
            return Ok(logits);
        }
        if !fp32 && self.lm_head_fp8_tc(hidden, 1, logits, stream)? {
            // 2026-10-05: `METRALE_GLM_GEMV_TC=1`: the same tensor-core FP8 GEMV as the
            // batched head, so a decode row and a verify row get the same logits.
        } else if let Some(ref fp8) = self.lm_head_fp8 {
            // 2026-09-25: FP8 E4M3 head (`--lm-head-dtype fp8`). It has no
            // FP32-output variant; with `use_fp32_logits` false, `logits` is the
            // BF16 buffer.
            ops::dense_gemv_fp8w(
                self.gpu.as_ref(),
                self.dense_gemv_fp8w_kernel,
                hidden,
                fp8,
                logits,
                v,
                h,
                stream,
            )?;
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // 2026-09-25: The FP32-output variant when the destination is the
            // FP32 buffer, which does not happen while `use_fp32_logits` is false.
            let kernel = if fp32 {
                self.w4a16_gemv_logits_kernel
            } else {
                self.w4a16_gemv_kernel
            };
            ops::w4a16_gemv(
                self.gpu.as_ref(),
                kernel,
                hidden,
                nvfp4,
                logits,
                v,
                h,
                stream,
            )?;
        } else if fp32 {
            // 2026-09-25: Unreachable while `use_fp32_logits` is false, and
            // `dense_gemv_fp32out_kernel` is 0.
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_fp32out_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        } else if self.lm_head_bf16_tc(hidden, 1, logits, stream)? {
            // 2026-10-04: `METRALE_GLM_GEMV_TC=1`: the same tensor-core GEMV as the batched
            // head, so a decode row and a verify row get the same logits.
        } else if let Some((begin, len)) = self.lmhead_vocab_shard(v) {
            // 2026-09-25: Vocab-parallel BF16 head. Each rank holds the whole head
            // but computes only its row range (`lmhead_vocab_shard`) into a zeroed
            // buffer, and the all-reduce sums the ranks' pieces into full logits.
            self.gpu.memset_async(logits, 0, v as usize * 2, stream)?;
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                hidden,
                &metrale_model_layers::weight_map::DenseWeight {
                    weight: self.lm_head_weight.weight.offset(begin * h as usize * 2),
                },
                logits.offset(begin * 2),
                len as u32,
                h,
                stream,
            )?;
            if let Some(comm) = self.comm_ref() {
                comm.all_reduce_async(logits.0, v as usize * 2, stream)?;
            }
        } else {
            ops::dense_gemv(
                self.gpu.as_ref(),
                self.dense_gemv_kernel,
                hidden,
                &self.lm_head_weight,
                logits,
                v,
                h,
                stream,
            )?;
        }
        // 2026-09-25: Token overlay, as in `lm_head_batched`; `fp32` picks the
        // FP32-logits overlay kernel.
        self.apply_lmhead_overlay(hidden, DevicePtr(0), logits, 1, fp32, stream)?;
        if self.logit_softcap_kernel.0 != 0 || self.logit_softcap_fp32_kernel.0 != 0 {
            let cap = self.config.final_logit_softcapping;
            self.apply_logit_softcap_dtype(logits, v, cap, fp32, stream)?;
        }
        Ok(logits)
    }
}
