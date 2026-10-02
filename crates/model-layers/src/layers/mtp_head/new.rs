// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `MtpHead::new`: quantizes the head's projections, builds its
//! FFN (dense, NVFP4 MoE, native-FP8 MoE or per-expert MoE), and allocates
//! its KV pool, kernel handles and scratch.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - A head has one FFN form: the MoE parts are built only when the weights
//!   carry no dense FFN.
//! - The drafter KV pool and the `propose_meta` stride use the same block
//!   size.

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::GpuBackend;
use parking_lot::Mutex;

use super::{MtpHead, MtpQuantization, ProjectionWeight};
use crate::layers::MoeLayer;
use crate::weight_map::{DenseWeight, MoeWeights, MtpWeights, QuantizedWeight, quantize_to_nvfp4};

impl MtpHead {
    pub fn new(
        weights: MtpWeights,
        embed_tokens: DenseWeight,
        target_final_norm: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        // 2026-09-25: Transposed twin of the main LM head; the caller passes
        // `None` when the drafter has a dedicated draft head.
        lm_head_nvfp4_t: Option<(QuantizedWeight, u32)>,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
        quant: MtpQuantization,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        main_kv_blocks: usize,
        levers: &crate::layers::ops::ModelLevers,
    ) -> Result<Self> {
        let stream = gpu.default_stream();
        // 2026-09-25: A dense-FFN head under `--mtp-quantization nvfp4` runs
        // the BF16 forward (`effective_for_head`): q/o and the dense
        // gate/up/down are stored NVFP4 and run W4A16, while fc/k/v stay BF16,
        // which the drafter prefill requires.
        let wquant = quant;
        let quant = wquant.effective_for_head(weights.dense_ffn.is_some());
        let dense_nvfp4 = quant != wquant;
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let nvfp4_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let fp8_k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;

        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let inter = if config.moe_intermediate_size > 0 {
            config.moe_intermediate_size
        } else {
            config.intermediate_size
        };

        let q = |bf16: &DenseWeight, n: usize, k: usize| -> Result<ProjectionWeight> {
            Self::quantize_proj(bf16, n, k, quant, gpu, absmax_k, nvfp4_k, fp8_k, stream)
        };
        // 2026-09-25: The requested precision, for the weight-only NVFP4
        // projections of a dense head; identical to `q` for every other head.
        let qw = |bf16: &DenseWeight, n: usize, k: usize| -> Result<ProjectionWeight> {
            Self::quantize_proj(bf16, n, k, wquant, gpu, absmax_k, nvfp4_k, fp8_k, stream)
        };

        let fc = q(&weights.fc, h, h * 2)?;
        let q_proj = qw(&weights.q_proj, nq * hd * 2, h)?;
        let k_proj = q(&weights.k_proj, nkv * hd, h)?;
        let v_proj = q(&weights.v_proj, nkv * hd, h)?;
        let o_proj = qw(&weights.o_proj, h, nq * hd)?;

        let dense_ffn_generic = if let Some(dense_ffn) = weights.dense_ffn.as_ref() {
            Some((
                qw(&dense_ffn.gate_proj, inter, h)?,
                qw(&dense_ffn.up_proj, inter, h)?,
                qw(&dense_ffn.down_proj, h, inter)?,
            ))
        } else {
            None
        };

        // 2026-09-25: MoE: an NVFP4 head gets a fused `MoeLayer`; an FP8 or
        // BF16 head gets a `MoeLayer` on the checkpoint's FP8 tables when the
        // weights carry them (`moe_fp8`), else per-expert weights.
        let mut weights = weights;
        let moe_parts = if dense_ffn_generic.is_some() {
            (None, None, None, None)
        } else {
            match quant {
                MtpQuantization::Nvfp4 => {
                    let gate_nvfp4 = quantize_to_nvfp4(
                        &weights.moe_gate,
                        config.num_experts,
                        h,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let mut experts = Vec::with_capacity(weights.experts.len());
                    for (i, de) in weights.experts.iter().enumerate() {
                        let gate_proj = quantize_to_nvfp4(
                            &de.gate_proj,
                            inter,
                            h,
                            gpu,
                            absmax_k,
                            nvfp4_k,
                            stream,
                        )?;
                        let up_proj = quantize_to_nvfp4(
                            &de.up_proj,
                            inter,
                            h,
                            gpu,
                            absmax_k,
                            nvfp4_k,
                            stream,
                        )?;
                        let down_proj = quantize_to_nvfp4(
                            &de.down_proj,
                            h,
                            inter,
                            gpu,
                            absmax_k,
                            nvfp4_k,
                            stream,
                        )?;
                        experts.push(crate::weight_map::ExpertWeight {
                            gate_proj,
                            up_proj,
                            down_proj,
                        });
                        if (i + 1) % 128 == 0 {
                            tracing::info!(
                                "  MTP experts quantized: {}/{}",
                                i + 1,
                                weights.experts.len()
                            );
                        }
                    }
                    let shared_gate = quantize_to_nvfp4(
                        &weights.shared_expert.gate_proj,
                        inter,
                        h,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let shared_up = quantize_to_nvfp4(
                        &weights.shared_expert.up_proj,
                        inter,
                        h,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let shared_down = quantize_to_nvfp4(
                        &weights.shared_expert.down_proj,
                        h,
                        inter,
                        gpu,
                        absmax_k,
                        nvfp4_k,
                        stream,
                    )?;
                    let moe_weights = MoeWeights {
                        gate: weights.moe_gate,
                        shared_expert: crate::weight_map::ExpertWeight {
                            gate_proj: shared_gate,
                            up_proj: shared_up,
                            down_proj: shared_down,
                        },
                        shared_expert_gate: weights.shared_expert_gate,
                        experts,
                        router_pre_norm: None,
                        correction_bias: None,
                    };
                    let moe = MoeLayer::new(
                        moe_weights,
                        config.num_experts,
                        Some(gate_nvfp4),
                        gpu,
                        config,
                    )?;
                    (Some(moe), None, None, None)
                }
                MtpQuantization::Fp8 | MtpQuantization::Bf16 if weights.fp8_experts.is_some() => {
                    let moe = Self::new_native_fp8_moe(&mut weights, config, gpu)?;
                    (None, None, None, Some(moe))
                }
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                    let mut experts_g = Vec::with_capacity(weights.experts.len());
                    for (i, de) in weights.experts.iter().enumerate() {
                        let gate_proj = q(&de.gate_proj, inter, h)?;
                        let up_proj = q(&de.up_proj, inter, h)?;
                        let down_proj = q(&de.down_proj, h, inter)?;
                        experts_g.push((gate_proj, up_proj, down_proj));
                        if (i + 1) % 128 == 0 {
                            tracing::info!(
                                "  MTP experts quantized: {}/{}",
                                i + 1,
                                weights.experts.len()
                            );
                        }
                    }
                    let shared = (
                        q(&weights.shared_expert.gate_proj, inter, h)?,
                        q(&weights.shared_expert.up_proj, inter, h)?,
                        q(&weights.shared_expert.down_proj, h, inter)?,
                    );
                    (None, Some(experts_g), Some(shared), None)
                }
            }
        };
        let (moe_nvfp4, moe_experts_generic, moe_shared_generic, moe_fp8) = moe_parts;

        // 2026-09-25: One attention layer. BF16 KV for BF16 and FP8 heads, FP8
        // KV for NVFP4 heads; the FP8 KV forward passes unit K/V scales
        // (`forward.rs`).
        let kv_bf16 = matches!(quant, MtpQuantization::Bf16 | MtpQuantization::Fp8);
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: nkv,
            head_dim: hd,
            num_layers: 1,
            dtype: if kv_bf16 {
                KvCacheDtype::Bf16
            } else {
                KvCacheDtype::Fp8
            },
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
            v_aliases_k: false,
        };
        // 2026-09-26: `max_seq_len / block + 1` blocks for each of
        // `mtp_max_seqs()` sequences, capped at the main pool's block count
        // (and never below one sequence). model-engine `factory/build/kv_budget.rs`
        // (`mtp_pool_reserve_bytes`) reserves the same arithmetic from the KV
        // budget; change both together.
        let per_seq_blocks = max_seq_len / kv_config.block_size + 1;
        let mtp_num_blocks = per_seq_blocks
            .saturating_mul(crate::speculative::mtp_max_seqs())
            .min(main_kv_blocks.max(per_seq_blocks));
        // 2026-09-25: The per-sequence `propose_meta` stride, from the drafter
        // pool's block size (`batch_caps::propose_meta_stride_env`: computed
        // from `max_seq_len`, floor 2048, override
        // `METRALE_PROPOSE_META_STRIDE`).
        let propose_meta_stride =
            super::batch_caps::propose_meta_stride_env(max_seq_len, kv_config.block_size);
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        let (
            dense_gemv_k,
            dense_gemv_fp8w_k,
            deinterleave_qg_k,
            moe_topk_k,
            moe_silu_mul_k,
            moe_weighted_sum_blend_k,
        ) = match quant {
            MtpQuantization::Nvfp4 => (None, None, None, None, None, None),
            MtpQuantization::Fp8 => (
                // 2026-09-25: BF16 GEMV for the router (`moe_gate` is BF16).
                Some(gpu.kernel("gemv", "dense_gemv_bf16")?),
                Some(gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?),
                Some(gpu.kernel("ssm_preprocess", "deinterleave_qg")?),
                Some(gpu.kernel("moe_topk", "moe_topk_softmax")?),
                Some(gpu.kernel("moe_silu_mul", "moe_silu_mul")?),
                Some(gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?),
            ),
            MtpQuantization::Bf16 => (
                Some(gpu.kernel("gemv", "dense_gemv_bf16")?),
                None,
                Some(gpu.kernel("ssm_preprocess", "deinterleave_qg")?),
                Some(gpu.kernel("moe_topk", "moe_topk_softmax")?),
                Some(gpu.kernel("moe_silu_mul", "moe_silu_mul")?),
                Some(gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?),
            ),
        };

        let effective_vocab = if mtp_vocab_size > 0 {
            (mtp_vocab_size as usize).min(config.vocab_size)
        } else {
            config.vocab_size
        };
        let ffn_kind: &str = if dense_ffn_generic.is_some() {
            "dense FFN"
        } else if moe_nvfp4.is_some() {
            "MoE (NVFP4 fused)"
        } else if moe_fp8.is_some() {
            "MoE (native FP8 tables; batched propose grouped)"
        } else {
            "MoE (per-expert)"
        };
        tracing::info!(
            "MTP head: quant={:?}{wo}, fc=[{h},{h2}], attn Q=[{qd},{h}], ffn={ffn}, \
             {ne} experts, vocab={ev}/{fv} (LM head {lm:.1} MB)",
            quant,
            wo = if dense_nvfp4 {
                " (weight-only NVFP4 q/o + dense FFN)"
            } else {
                ""
            },
            h2 = h * 2,
            qd = nq * hd * 2,
            ffn = ffn_kind,
            ne = if dense_ffn_generic.is_some() {
                0
            } else {
                config.num_experts
            },
            ev = effective_vocab,
            fv = config.vocab_size,
            lm = (effective_vocab * h / 2) as f64 / (1024.0 * 1024.0),
        );

        // 2026-09-25: Dedicated `PREFILL_CHUNK`-row scratch for the batched
        // row writer (`drafter_rows_impl`), allocated only when the drafter
        // prefill or the catch-up feed (`METRALE_MTP_CATCHUP`) is on.
        let prefill_scratch = if super::mtp_drafter_prefill_enabled(levers)
            || crate::speculative::mtp_catchup_enabled()
        {
            let c = super::prefill::PREFILL_CHUNK;
            let bf16 = 2usize;
            Some(super::MtpPrefillScratch {
                embed: gpu.alloc(c * h * bf16)?,
                normed_embed: gpu.alloc(c * h * bf16)?,
                normed_hidden: gpu.alloc(c * h * bf16)?,
                concat: gpu.alloc(c * 2 * h * bf16)?,
                fc_out: gpu.alloc(c * h * bf16)?,
                normed2: gpu.alloc(c * h * bf16)?,
                k_out: gpu.alloc(c * nkv * hd * bf16)?,
                v_out: gpu.alloc(c * nkv * hd * bf16)?,
                q_scratch: gpu.alloc(c * nq * hd * bf16)?,
                pos_dev: gpu.alloc(c * 4)?,
                slot_dev: gpu.alloc(c * 8)?,
            })
        } else {
            None
        };

        Ok(Self {
            pre_fc_norm_embedding: weights.pre_fc_norm_embedding,
            pre_fc_norm_hidden: weights.pre_fc_norm_hidden,
            input_layernorm: weights.input_layernorm,
            post_attn_layernorm: weights.post_attn_layernorm,
            norm: weights.norm,
            fc,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
            moe_nvfp4,
            moe_experts_generic,
            moe_shared_generic,
            moe_fp8,
            moe_gate: weights.moe_gate,
            shared_expert_gate: weights.shared_expert_gate,
            dense_ffn_generic,
            quant,
            mtp_vocab_size,
            embed_tokens,
            target_final_norm,
            lm_head_nvfp4,
            kv_cache: Mutex::new(kv_cache),
            attn_layer_idx: 0,
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            gemv_sw: crate::layers::ops::gemv_sw_from(
                std::env::var("METRALE_NO_GEMV_SW").ok().as_deref(),
            ),
            w4a16_gemv_qg_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
            w4a16_gemv_dual_k: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            rope_k: gpu.kernel("rope", "rope_forward")?,
            reshape_cache_k: if kv_bf16 {
                gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?
            } else {
                gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?
            },
            paged_decode_k: if kv_bf16 {
                gpu.kernel("paged_decode", "paged_decode_attn")?
            } else {
                gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")?
            },
            kv_bf16,
            kv_exact: levers.mtp_kv_exact,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            residual_add_rms_norm_k: gpu.kernel("norm", "residual_add_rms_norm")?,
            sigmoid_gate_mul_k: gpu.kernel("residual_add", "sigmoid_gate_mul")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            embed_from_argmax_k: gpu.kernel("embed_from_argmax", "embed_from_argmax")?,
            draft_token_id_dev: gpu.alloc(4)?,
            last_conf_bits: std::sync::atomic::AtomicU32::new(1.0f32.to_bits()),
            dense_gemv_k,
            dense_gemv_fp8w_k,
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv").ok(),
            deinterleave_qg_k,
            moe_topk_k,
            moe_silu_mul_k,
            moe_weighted_sum_blend_k,
            // 2026-09-25: 0 when absent; the drafter prefill then writes no
            // rows.
            dense_gemm_k: crate::layers::try_kernel(gpu, "gemm", "dense_gemm_bf16"),
            batched_embed_k: crate::layers::try_kernel(gpu, "embed_from_argmax", "batched_embed"),
            dense_gemm_pipelined_k: crate::layers::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            // 2026-09-25: 0 when absent; `row_dispatch::drafter_row_kernel`
            // then picks the pipelined GEMM or the per-row loop.
            dense_gemv_batchm_k: crate::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            w4a16_batchm: crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(gpu),
            w4a16_gemv_batch16_k: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch16",
            ),
            w4a16_gemv_batch32_k: crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch32",
            ),
            // 2026-09-25: Dropped when `METRALE_NO_MTP_LMHEAD_TGEMM` is set to
            // any value, `0` included.
            lm_head_nvfp4_t: lm_head_nvfp4_t
                .filter(|_| std::env::var_os("METRALE_NO_MTP_LMHEAD_TGEMM").is_none()),
            w4a16_gemm_t_k: crate::layers::tgemm_kernel(gpu),
            argmax_batch_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch"),
            argmax_batch_lp_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_batch_lp"),
            // 2026-09-25: `PROPOSE_META_SEQS` slabs of `propose_meta_stride`
            // bytes, a dedicated allocation rather than an offset into
            // `scratch`.
            propose_meta: gpu.alloc(super::batch_caps::PROPOSE_META_SEQS * propose_meta_stride)?,
            propose_meta_stride,
            prefill_scratch,
        })
    }
}
