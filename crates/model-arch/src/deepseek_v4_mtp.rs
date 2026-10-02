// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DeepSeek-V4-Flash multi-token-prediction (MTP) draft proposer.
//!
//! Implements [`DraftProposer`] over the `DeepseekV4MtpModule` loaded by
//! `load_v4_mtp_module`. Where
//! [`metrale_model_layers::layers::MtpHead`] builds its own attention + MoE
//! block, the V4 MTP module's body is a reused V4 layer (MLA + mHC
//! hyper-connections + MoE), so the proposer delegates the bulk of the
//! forward to `body.decode()` and wraps it with the MTP-specific pieces.
//!
//! One draft (`forward_one`; `propose` runs it `num_drafts` times):
//!
//! ```text
//!   embed   = embed_tokens[token]                            // [hidden] bf16
//!   h_in    = e_proj · rms_norm(embed,  enorm)
//!           + h_proj · rms_norm(hidden, hnorm)               // combiner
//!   hc_expand(h_in → hc_streams)                             // first mHC stage
//!   body.decode(hc_streams, …, mtp_kv_cache, state.seq_len)  // middle mHC + MLA + MoE
//!   hc_head(hc_streams → h_out)                              // last mHC stage
//!   logits  = lm_head(rms_norm(h_out, norm))
//!   draft   = argmax(logits)                                 // grammar-masked when Some
//! ```
//!
//! The body was assembled with `layer_idx = num_hidden_layers`, so its
//! `decode_inner_hc` is neither the first nor the last model layer: it runs
//! the middle mHC mixing (hc_pre → attn → hc_post → hc_pre → ffn → hc_post)
//! on `hc_streams` and calls neither `hc_expand` nor `hc_head`. The proposer
//! supplies both ends.
//!
//! ## Separate KV cache and metadata
//!
//! The MTP attention writes into its own MLA-shaped [`PagedKvCache`]
//! (num_kv_heads = 1, head_dim = kv_lora_rank + qk_rope_head_dim), never the
//! target's. The decode attention reads positions / slot / seq_len /
//! block_table from `ctx.attn_metadata`, so the proposer uploads its own
//! metadata to `scratch().offset(MTP_META_OFFSET)`, away from the target's
//! at 32768, and passes it in a derived [`ForwardContext`].
//!
//! Owner: model-arch, DeepSeek-V4 MTP.
//! Invariants: none beyond the types.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::weight_loader::deepseek_v4::DeepseekV4MtpModule;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState};
use metrale_model_layers::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::{DraftProposer, ProposerState};
use metrale_model_layers::weight_map::DenseWeight;

mod proposer;

/// 2026-09-25: Per-sequence state for the DeepSeek-V4 MTP proposer: the block
/// table and length of the sequence in the MTP KV cache, the draft count of
/// the last `propose` (for `after_verify`), and the body's layer state from
/// `body.alloc_state`.
pub struct DeepseekV4MtpProposerState {
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub last_num_drafted: usize,
    pub body_state: Box<dyn LayerState>,
}

impl ProposerState for DeepseekV4MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// 2026-09-25: DeepSeek-V4 MTP draft proposer. `embed_tokens` and `lm_head`
/// are the target model's bf16 tables; `mtp_vocab_size` limits the draft
/// LM-head GEMV (0 = full vocab); `kv_cache` is the MTP attention's own cache.
pub struct DeepseekV4MtpHead {
    module: DeepseekV4MtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    mtp_vocab_size: u32,
    kv_cache: Mutex<PagedKvCache>,

    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    residual_add_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_head_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl DeepseekV4MtpHead {
    /// 2026-09-25: Build the proposer from a loaded `DeepseekV4MtpModule` and the
    /// target's bf16 embedding and LM head.
    pub fn new(
        module: DeepseekV4MtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        // 2026-09-25: The target's MLA cache shape (num_kv_heads = 1, head_dim
        // = kv_lora_rank + qk_rope_head_dim), so the reused V4 body writes and
        // reads it at the right strides; bf16.
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        // 2026-09-25: The body was built with layer index `num_hidden_layers`
        // and its decode indexes the cache at that layer, so the cache has
        // `num_hidden_layers + 1` layer slots of which only the last is used.
        let num_layers = config.num_hidden_layers + 1;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: mla_cache_dim,
            num_layers,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
            v_aliases_k: false,
        };
        let mtp_num_blocks = max_seq_len / kv_config.block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            // 2026-09-25: The norm weights are plain (`w * x_normed`); the shared
            // `rms_norm` kernel would apply `1 + w`.
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    /// 2026-09-25: Allocate per-sequence state; the body's part comes from
    /// `body.alloc_state`.
    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<DeepseekV4MtpProposerState> {
        Ok(DeepseekV4MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
            body_state: self.module.body.alloc_state(gpu)?,
        })
    }

    /// 2026-09-25: One MTP draft step at `position`. Returns the drafted token id.
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut DeepseekV4MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc_mult = ctx.config.hc_mult as u32;
        let row_bytes = h as usize * 2;

        // 2026-09-25: 1. Embed the input token (a row copy from the shared table).
        let embed_out = ctx.buffers.ssm_qkvz();
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        // 2026-09-25: 2. Combiner: h_in = e_proj·rms_norm(embed, enorm)
        // + h_proj·rms_norm(target_hidden, hnorm).
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.module.enorm,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;
        let normed_hidden = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.module.hnorm,
            normed_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        // 2026-09-25: The embedding branch goes into `h_in`, the hidden branch
        // into a temp, then `bf16_residual_add` does h_in += temp in place.
        let h_in = ctx.buffers.hidden_states();
        let h_branch = ctx.buffers.norm_output();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed_embed,
            &self.module.e_proj,
            h_in,
            h,
            h,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed_hidden,
            &self.module.h_proj,
            h_branch,
            h,
            h,
            stream,
        )?;
        ops::residual_add(ctx.gpu, self.residual_add_k, h_in, h_branch, h, stream)?;

        // 2026-09-25: 3. mHC expand: replicate h_in into hc_mult streams.
        let hc_streams = ctx.buffers.hc_streams();
        ops::hc_expand(
            ctx.gpu,
            self.hc_expand_k,
            h_in,
            hc_streams,
            1,
            h,
            hc_mult,
            stream,
        )?;

        // 2026-09-25: 4. Body decode: the middle mHC + MLA attention (writing
        // the MTP KV cache) + MoE, on `hc_streams`.
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        // 2026-09-25: The MTP attention metadata goes to `MTP_META_OFFSET`, away
        // from the target's at 32768: pos(u32)@0, slot(i64)@8, seq_len(i32)@16,
        // block_table(i32[])@256 (`pack_mtp_attn_meta`).
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let max_blocks = state.block_table.len() as u32;
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;

        // 2026-09-25: `pack_mtp_attn_meta` refuses a block table that does not
        // fit the scratch after `MTP_META_OFFSET`.
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            actual_seq_len,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

        let mtp_meta = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(8),
            seq_len: meta_base.offset(16),
            block_table: meta_base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
            seq_slot: metrale_gpu_runtime::gpu::DevicePtr(0),
            moe_row_adapter: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        // 2026-09-25: The body's hash-routed MoE (if any) reads the decode token
        // id from `token_ids[0]`, so this draft's input token goes there.
        if let Some(tid_buf) = ctx.token_ids {
            ctx.gpu
                .copy_h2d_async(&token.to_le_bytes(), tid_buf, stream)?;
        }

        // 2026-09-25: A ForwardContext carrying the MTP metadata, with graph
        // capture off: the metadata is built on the host for every call.
        let mtp_ctx = ForwardContext {
            buffers: ctx.buffers,
            hc_row_offset: ctx.hc_row_offset,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(mtp_meta),
            profile: ctx.profile,
            // 2026-09-25: No communicator: the MTP draft runs only on rank 0, so
            // its MoE must not issue an EP all-reduce. The body is loaded with
            // every expert local (`force_all_experts` in `load_v4_mtp_module`).
            comm: None,
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: ctx.token_ids,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: metrale_model_layers::layer::MoeLoraRoute::Skip,
        };

        // 2026-09-25: `decode_inner_hc` reads the streams from
        // `ctx.buffers.hc_streams()` and collapses into its `hidden` argument,
        // so `hidden` must not alias `hc_streams`; `hidden_states()` held
        // `h_in`, which `hc_expand` has already consumed.
        let body_scratch = ctx.buffers.hidden_states();
        let mut disk_block_ids: Vec<u32> = Vec::new();
        let mut disk_last_offloaded: Vec<u32> = vec![0u32; 1];
        let residual = ctx.buffers.residual();
        self.module.body.decode(
            body_scratch,
            residual,
            state.body_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut disk_block_ids,
            &mut disk_last_offloaded,
            &mtp_ctx,
            stream,
        )?;
        drop(kv_cache);

        // 2026-09-25: 5. mHC head: collapse the hc_mult streams into h_out.
        let h_out = ctx.buffers.hidden_states();
        if let Some(ref head) = self.module.hc_head {
            // 2026-09-25: Only the Sinkhorn `hc_head` launch is here; a low-rank
            // head takes other arguments (`hc_head_site` dispatches on
            // `lowrank`), so it is refused rather than launched as Sinkhorn.
            anyhow::ensure!(
                head.lowrank.is_none(),
                "deepseek_v4_mtp: low-rank mHC head reached the Sinkhorn MTP \
                 path; this module has no low-rank dispatch"
            );
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                h_out,
                1,
                h,
                hc_mult,
                eps,
                ctx.config.hc_eps,
                stream,
            )?;
        } else {
            // 2026-09-25: No head weights (no mHC): take the first stream's row.
            ctx.gpu
                .copy_d2d_async(hc_streams, h_out, row_bytes, stream)?;
        }

        // 2026-09-25: 6. Final norm and the shared LM head -> logits.
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            h_out,
            &self.module.norm,
            final_normed,
            1,
            h,
            eps,
            stream,
        )?;
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            final_normed,
            &self.lm_head,
            logits,
            v,
            h,
            stream,
        )?;

        // 2026-09-25: 7. Argmax, grammar-masked when a bitmask is given.
        let out_ptr = ctx.buffers.scratch();
        let token_id = if let Some(bitmask) = grammar_bitmask {
            argmax_grammar_masked(ctx.gpu, logits, v as usize, bitmask, position)?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
            u32::from_le_bytes(buf)
        };

        state.seq_len += 1;
        Ok(token_id)
    }
}

/// 2026-09-25: Grammar-masked argmax over the bf16 logits on the CPU: copy the
/// logits to the host, skip the tokens the bitmask rejects, take the argmax.
/// Returns 0 (with a warning) when the bitmask allows no token.
fn argmax_grammar_masked(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    vocab: usize,
    bitmask: &[i32],
    position: usize,
) -> Result<u32> {
    let mut bf16_buf = vec![0u8; vocab * 2];
    gpu.copy_d2h(logits, &mut bf16_buf)?;

    let mut best_tok = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    let mut any_allowed = false;
    for tok in 0..vocab {
        let word = tok / 32;
        let bit = tok % 32;
        let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0;
        if !allowed {
            continue;
        }
        any_allowed = true;
        // 2026-09-25: bf16 -> f32: bf16 is the upper 16 bits of an f32.
        let hi = u16::from_le_bytes([bf16_buf[2 * tok], bf16_buf[2 * tok + 1]]);
        let val = f32::from_bits((hi as u32) << 16);
        if val > best_val {
            best_val = val;
            best_tok = tok as u32;
        }
    }
    if !any_allowed {
        tracing::warn!(
            "V4 MTP grammar mask allowed zero tokens at pos {position}; \
             returning 0 as pad-draft (will be rejected at verify)."
        );
        return Ok(0);
    }
    Ok(best_tok)
}
