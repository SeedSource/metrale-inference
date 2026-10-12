// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_LM_HEAD_NV4_SHARD=1`: the sharded NVFP4 target head.
//!
//! Owner: model-engine.
//! Invariants:
//! - Off (the default) the field is `None`, `install_lm_head_nv4_shard` returns before touching
//!   the GPU, and `lm_head_fp8_tc` runs exactly as before.
//! - On, it applies only to a vocab-sharded FP8 head (`lm_head_fp8` set, no Q6_K head,
//!   `lmhead_vocab_shard` splits the vocab, a BF16 head to quantize from): the calls that take
//!   `lm_head_fp8_tc` (the 1-row decode head and `lm_head_batched`, 1..=16 rows) take the NVFP4
//!   shard instead when `glm5next_lm_head_nv4::routes(rows)`. Any other head, or a failed
//!   quantization, keeps the FP8 route (logged, never fatal).
//! - The FP8 copy stays: `lm_head_project_batched` (C > 1 pure decode) and the `decode_b2`
//!   prefill row read it over the full vocab.
//! - The logits layout and combine are the FP8 route's: zeroed `[rows, vocab]` BF16 logits, this
//!   rank's columns `[begin, begin + len)` written, then the cross-rank all-reduce.
//! - The shard sweep writes rows at stride `len`, so 2+ rows go through one shared scratch and a
//!   pitched copy; that scratch assumes the 2+-row callers (verify heads, prompt logprobs) share
//!   one stream. The 1-row call writes the logits directly.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_arch::glm5next_lm_head_nv4 as nv4;
use metrale_model_layers::weight_map::QuantizedWeight;

use super::types::TransformerModel;

/// This rank's NVFP4 vocab shard and the scatter scratch.
pub(crate) struct LmHeadNv4Shard {
    q: QuantizedWeight,
    begin: usize,
    len: usize,
    /// `[ROUTE_MAX_M, len]` BF16.
    scratch: DevicePtr,
}

impl TransformerModel {
    /// Build the NVFP4 shard when the lever is on and the head qualifies (module invariants).
    pub fn install_lm_head_nv4_shard(&mut self) -> Result<()> {
        if !nv4::enabled() {
            return Ok(());
        }
        let v = self.config.vocab_size;
        let h = self.config.hidden_size;
        let Some((begin, len)) = self.lmhead_vocab_shard(v as u32) else {
            tracing::warn!(
                "METRALE_GLM_LM_HEAD_NV4_SHARD=1 is ignored: the head is not vocab-sharded"
            );
            return Ok(());
        };
        if self.lm_head_fp8.is_none()
            || self.lm_head_q6k.is_some()
            || self.lm_head_weight.weight.is_null()
        {
            tracing::warn!(
                "METRALE_GLM_LM_HEAD_NV4_SHARD=1 is ignored: it needs the FP8 head with a BF16 \
                 head to quantize from"
            );
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        match nv4::quantize_shard(
            self.gpu.as_ref(),
            self.lm_head_weight.weight,
            v,
            h,
            begin,
            len,
            stream,
        ) {
            Ok(Some(q)) => {
                let scratch = self.gpu.alloc(nv4::ROUTE_MAX_M * len * 2)?;
                let mb = |b: usize| b / (1024 * 1024);
                tracing::warn!(
                    "METRALE_GLM_LM_HEAD_NV4_SHARD=1 - GLM target head: rows [{begin}, +{len}) x \
                     {h} quantised to NVFP4 ({} MB + {} MB scratch, tensor scale {} from the whole \
                     head); 1..={} row heads read it (CUDA-core below {} rows, tensor-core from \
                     there); the FP8 head stays for C>1 decode; NOT byte-identical to the FP8 head",
                    mb(len * h / 2 + len * h / 16),
                    mb(nv4::ROUTE_MAX_M * len * 2),
                    q.weight_scale_2,
                    nv4::ROUTE_MAX_M,
                    nv4::TC_FROM_M,
                );
                self.lm_head_nv4_shard = Some(LmHeadNv4Shard {
                    q,
                    begin,
                    len,
                    scratch,
                });
            }
            Ok(None) => tracing::warn!(
                "METRALE_GLM_LM_HEAD_NV4_SHARD=1 but the NVFP4 quantizer is unavailable or the \
                 shape is not quantizable; keeping the FP8 head"
            ),
            Err(e) => tracing::warn!("NVFP4 target head unavailable ({e:#}); keeping the FP8 head"),
        }
        Ok(())
    }

    /// `Ok(true)` when the NVFP4 shard served this `rows`-row head call (logits written, combine
    /// done); `Ok(false)` having launched nothing (no shard, or `rows` not routed).
    pub(super) fn lm_head_nv4_shard_run(
        &self,
        hidden: DevicePtr,
        rows: u32,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let Some(ref s) = self.lm_head_nv4_shard else {
            return Ok(false);
        };
        let rows = rows as usize;
        if !nv4::routes(rows) {
            return Ok(false);
        }
        let v = self.config.vocab_size;
        let h = self.config.hidden_size;
        self.gpu.memset_async(logits, 0, rows * v * 2, stream)?;
        if rows == 1 {
            nv4::sweep(
                self.gpu.as_ref(),
                hidden,
                &s.q,
                logits.offset(s.begin * 2),
                1,
                s.len,
                h,
                stream,
            )?;
        } else {
            nv4::sweep(
                self.gpu.as_ref(),
                hidden,
                &s.q,
                s.scratch,
                rows,
                s.len,
                h,
                stream,
            )?;
            self.gpu.copy_d2d_2d_async(
                s.scratch,
                s.len * 2,
                logits.offset(s.begin * 2),
                v * 2,
                s.len * 2,
                rows,
                stream,
            )?;
        }
        if let Some(comm) = self.comm_ref() {
            comm.all_reduce_async(logits.0, rows * v * 2, stream)?;
        }
        Ok(true)
    }
}
