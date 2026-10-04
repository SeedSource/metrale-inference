// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Byte size of every buffer in the GPU buffer arena, from the model config, the serve limits and five environment levers.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - [`BufferSizes::from_config`] allocates nothing; besides its arguments it
//!   reads only `METRALE_GGUF_NATIVE_Q2`, `METRALE_GGUF_NATIVE_Q2_MMQ`,
//!   `METRALE_FP8_ROWWISE`, `METRALE_GLM_ARENA_TRIM` (`sizes_glm_trim.rs`) and,
//!   through `attn_splitk::policy_from_env`, `METRALE_ATTN_DECODE_SPLITK`.

use metrale_config::ModelConfig;
use metrale_kernels::attn_splitk;

mod regions;

/// 2026-09-25: The widest `M` the fused dense-FFN gate+up decode GEMM serves, and so
/// the row extent of `ffn_gate_up_fused`. Declared here because the arena is sized
/// in this crate; the dispatch (`layers/dense_ffn_gateup_fused.rs` in
/// metrale-model-layers) reads it for its `5..=GATEUP_FUSED_MAX_M` band, so the band
/// and the buffer cannot disagree.
pub const GATEUP_FUSED_MAX_M: usize = 16;

/// 2026-09-25: Byte size of each arena buffer. `M` is `max_batch_tokens` and
/// `ceil16(M)` is M rounded up to a multiple of 16.
#[derive(Debug, Clone)]
pub struct BufferSizes {
    pub hidden_states: usize,
    pub residual: usize,
    pub norm_output: usize,
    pub qkv_output: usize,
    pub attn_output: usize,
    pub gate_logits: usize,
    /// 2026-09-25: FP32 router logits `[M, num_experts + zero_expert_num]` for
    /// the `METRALE_FP32_GATE` and `METRALE_FP32_ROUTING` paths, which run the
    /// top-K on unrounded logits. 256 bytes for a model without experts.
    pub gate_logits_f32: usize,
    /// 2026-09-25: FP32 MoE-input norm output `[M, hidden]` that the router GEMM
    /// reads under `METRALE_FP32_ROUTING`. 256 bytes for a model without experts.
    pub moe_router_in_f32: usize,
    pub moe_output: usize,
    pub logits: usize,
    pub ssm_qkvz: usize,
    pub ssm_ba: usize,
    pub ssm_deinterleaved: usize,
    pub ssm_gates: usize,
    pub ssm_conv_out_f32: usize,
    pub scratch: usize,
    pub expert_gate_out: usize,
    pub expert_up_out: usize,
    pub expert_down_out: usize,
    pub splitk_workspace: usize,
    /// 2026-09-25: GDN FLA chunked-prefill scratch, W | U | S | uc | gc back to
    /// back. 0 unless the linear-attention key and value head dims are both 128,
    /// the only shape the FLA prefill kernels run.
    pub gdn_fla_scratch: usize,
    /// 2026-09-25: Mamba-2 SSD chunked-scan scratch, dt | dA_cumsum | CB. 0
    /// unless the model has Mamba-2 heads and a state size.
    pub ssd_scratch: usize,
    /// 2026-09-25: Grouped O-projection latent `[M, o_groups * o_lora_rank]`
    /// BF16, at least 256 bytes.
    pub o_latent: usize,
    /// 2026-09-25: The zero weight `norm_unit_w`, `max(hidden, Mamba-2 d_inner)`
    /// BF16 values.
    pub norm_unit_w: usize,
    /// 2026-09-25: Hyper-connection residual streams `[M, hc_mult, hidden]`,
    /// FP32. 256 bytes when `hc_mult == 0`.
    pub hc_streams: usize,
    /// 2026-09-25: Hyper-connection `post` weights `[M, hc_mult]` F32.
    pub hc_post: usize,
    /// 2026-09-25: Hyper-connection `comb` matrix `[M, hc_mult, hc_mult]` F32.
    pub hc_comb: usize,
    /// 2026-09-25: Low-rank hyper-connection scratch: the larger of the decode
    /// split layout (T <= 64, FP32) and the prefill GEMM layout (slabs of at
    /// most 2048 tokens, BF16); see the sizing in `from_config`. 256 bytes
    /// without a low-rank hyper-connection.
    pub hc_lowrank_scratch: usize,
    /// 2026-09-25: QSA prefill-selection scratch for 2048-row slabs, shared by
    /// the indexer layers: qk `[2048, (index_n_heads + 1) * index_head_dim]`
    /// BF16, q_post `[2048, index_n_heads, index_head_dim]` F32, scores
    /// `[2048, ceil(max_seq_len / index_compress_ratio)]` F32, lists
    /// `[2048, index_topk / index_compress_ratio]` i32. 256 bytes without an
    /// indexer. `layers/qsa_select.rs` carves the same layout.
    pub qsa_select_scratch: usize,
    /// 2026-09-25: Token ids `[M]` u32 of the current pass, read by the
    /// DeepSeek-V4 hash-routed MoE layers (`tid2eid[token_id]`). At least 256
    /// bytes.
    pub token_ids: usize,
    /// 2026-09-25: Dense-FFN activation-quant scratch shared by every layer's
    /// prefill, sized for K = max(hidden, intermediate); all 0 for a MoE model.
    /// `ffn_act_q8`: q8_1_mmq activations for the Q4_K MMQ path.
    /// `ffn_act_a`: int8 `[ceil16(M), K]`, also NVFP4 packed and FP8 activations.
    /// `ffn_act_scale`: `[ceil16(M), K/32]` 4-byte int8 scales, also the NVFP4
    /// and FP8 scales.
    pub ffn_act_q8: usize,
    pub ffn_act_a: usize,
    pub ffn_act_scale: usize,
    /// 2026-09-25: `[K/128, ceil16(M)]` f32 copy of the FP8 activation scales,
    /// token index contiguous, for the cuBLASLt block-scaled FFN GEMM; the
    /// quantizer writes `[M, K/128]` into `ffn_act_scale`. 0 for a MoE model.
    pub ffn_act_scale_kmajor: usize,
    /// 2026-09-25: `[ceil16(GATEUP_FUSED_MAX_M), 2 * intermediate]` BF16 output of
    /// the fused dense-FFN gate+up decode GEMM; a row is `[gate | up]`. A buffer
    /// of its own because the fused arm serves only the decode band, and a
    /// widened `expert_gate_out` would carry `max_batch_tokens` rows. Sized for
    /// every dense model, whatever the lever says: the levers are resolved above
    /// this crate. 0 for a MoE model.
    pub ffn_gate_up_fused: usize,
    /// 2026-09-25: FP8 activation scratch for the prefill projections, 1 byte per
    /// element, `[ceil16(M), K]` for the widest K among hidden,
    /// `q_heads * head_dim`, Mamba-2 d_inner and the GDN value dim.
    pub fp8_act: usize,
    /// 2026-09-25: The grouped FP8 MoE scratch slab (`moe_fp8_scratch::Layout`); 0 for a
    /// dense model.
    pub moe_fp8_scratch: usize,
    /// 2026-09-25: One f32 scale per 128 elements of `fp8_act`.
    pub fp8_act_scale: usize,
    /// 2026-09-25: `[K/128, ceil16(M)]` f32 transpose of `fp8_act_scale`, token
    /// index contiguous, for the cuBLASLt block-scaled projections. Sized for
    /// every model, the SSM `in_proj_qkvz` arm included.
    pub fp8_act_scale_kmajor: usize,
    /// 2026-09-25: LoRA shrink output `xa = x @ Aᵀ`, `[M, adapter_max_rank]`
    /// BF16. This and the other `lora_*` entries are 0 when
    /// `adapter_max_rank == 0`.
    pub lora_xa: usize,
    /// 2026-09-25: LoRA expand output `delta = xa @ Bᵀ`, `[M, n]` BF16 with n the
    /// widest target: max(hidden, intermediate, the q projection width).
    pub lora_delta: usize,
    /// 2026-09-25: LoRA hidden-activation scratch `[M, intermediate_size]` BF16.
    pub lora_hact: usize,
    /// 2026-09-25: LoRA adapter slot per prefill token, `[M]` i32. A buffer of
    /// its own, so it cannot overlap the metadata regions of `scratch`.
    pub lora_seq_slot: usize,
    /// 2026-09-25: Keep-packed Q2_0 prefill dequant scratch
    /// (`METRALE_GGUF_NATIVE_Q2=1`): one BF16 buffer for the largest keep-packed
    /// projection, reused by each projection's dequant. Sized by
    /// [`crate::buffers::q2_dequant_scratch_bytes`]; 0 unless the lever is `1`.
    pub q2_dequant_scratch: usize,
    /// 2026-09-25: Keep-packed Q2_0 MMQ prefill q8_1 activation scratch
    /// (`METRALE_GGUF_NATIVE_Q2_MMQ=1`), shared by every keep-packed projection:
    /// each quantizes its BF16 activation here and runs the packed MMQ GEMM.
    /// 0 unless the lever is `1`.
    pub q2_act_q8: usize,
    /// 2026-09-25: Row-wise FP8 GDN prefill BF16-weight slab
    /// (`METRALE_FP8_ROWWISE=1`): every GDN layer's BF16 `in_proj_qkvz` and
    /// `out_proj`, one slice pair per layer carved on its first prefill and kept
    /// for the arena's life. Sized in `sizes_rowwise.rs`; 0 unless the lever is
    /// `1`.
    pub ssm_rowwise_w_bf16: usize,
    /// 2026-10-04: Bytes the GLM-5.3 trim zeroed above (`sizes_glm_trim.rs`); not a buffer.
    pub glm_trimmed: usize,
}

impl BufferSizes {
    /// 2026-09-25: Every buffer size, in bytes. `max_seq_len` and `kv_block_size`
    /// set the block-table width in `scratch` (`max_seq_len / kv_block_size + 1`
    /// blocks, 256 when `kv_block_size` is 0) and the QSA score width;
    /// `max_batch_size` sets the decode-metadata rows.
    pub fn from_config(
        config: &ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        kv_block_size: usize,
        max_batch_size: usize,
    ) -> Self {
        let decode_meta = super::DecodeMetaLayout::for_max_batch_size(max_batch_size);
        let bf16 = 2;
        let m = max_batch_tokens;
        let h = config.hidden_size;

        // 2026-09-25: A gated q projection (`attn_gated`) writes [Q | gate], twice
        // `q_heads * head_dim` wide.
        let q_heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let hd = config.head_dim;
        let q_proj_mul = if config.attn_gated { 2 } else { 1 };
        let qkv_dim = (q_heads * q_proj_mul + 2 * kv_heads) * hd;

        let scratch = regions::scratch_bytes(config, m, max_seq_len, kv_block_size, &decode_meta);

        // 2026-09-25: The row extent of every buffer a cuBLASLt block-scaled FP8
        // GEMM touches. `ops::cublas_fp8_proj_prequant` (metrale-model-layers)
        // hands cuBLASLt `ceil16(M)` rows: the pad rows of the activation and its
        // scales are read, and the pad rows of the output are written, so a
        // buffer sized for M rows would be overrun by up to 15 rows.
        let m_pad = m.div_ceil(16) * 16;
        // 2026-09-25: Expert outputs: `ceil16(max(M, 3))` rows of top_k experts
        // (or of the dense FFN).
        let k_max = m.max(3).div_ceil(16) * 16;
        let expert_inter = if config.num_experts > 0 {
            let routed = config.num_experts_per_tok * config.moe_intermediate_size;
            k_max * routed.max(config.intermediate_size)
        } else {
            k_max * config.intermediate_size
        };
        let expert_gate_out = expert_inter * bf16;
        let expert_up_out = expert_inter * bf16;
        // 2026-09-25: A LatentMoE model's routed experts write `moe_latent_size`
        // wide rows (`moe_input_size`).
        let moe_out_dim = config.moe_input_size();
        let expert_down_out = if config.num_experts > 0 {
            k_max * config.num_experts_per_tok * moe_out_dim * bf16
        } else {
            k_max * h * bf16
        };

        // 2026-09-25: Logit rows: min(M, max(160, decode rows + 1)). 160 is
        // `VERIFY_ROW_CAP`, the most rows a batched MTP verify scores. The mixed
        // decode step (`decode_b2.rs`) writes a prefill row after its `padded_n`
        // decode rows, and `padded_n` can reach `decode_meta.rows()`, hence + 1.
        let logits_tokens = m.min(160.max(decode_meta.rows() + 1));

        // 2026-09-25: Mamba-2 d_inner may exceed hidden_size; norm_output and
        // attn_output hold rows of either.
        let mamba2_d_inner = config.mamba2_d_inner();
        let max_dim = h.max(mamba2_d_inner);

        // 2026-09-25: Split-K decode workspace: one `[o[head_dim], m, l]` F32 slot
        // per (sequence, q head, split). A short workspace is an out-of-bounds
        // device write, so the slot count comes from `attn_splitk::workspace_slots`
        // under the policy the dispatch resolves too (`policy_from_env`), for
        // `decode_meta.rows()` sequences, the widest batch the metadata upload
        // accepts.
        let splitk_slots = attn_splitk::workspace_slots(
            attn_splitk::policy_from_env(),
            metrale_kernels::TARGET_SM_COUNT,
            q_heads as u32,
            decode_meta.rows() as u32,
            (max_batch_size as u32).max(1),
        ) as usize;
        let splitk_workspace = splitk_slots * (hd + 2) * 4;

        let residual_elem = bf16;

        // 2026-09-25: The widest K of the prefill projections that quantize into
        // `fp8_act`: hidden (qkv, ssm qkvz), `q_heads * head_dim` (o_proj), Mamba-2
        // d_inner (its out_proj) and the GDN value dim (its out_proj).
        let max_proj_k = h
            .max(q_heads * hd)
            .max(mamba2_d_inner)
            .max(config.linear_num_value_heads * config.linear_value_head_dim);
        let fp8_act = m_pad * max_proj_k;
        let fp8_act_scale = m_pad * max_proj_k.div_ceil(128) * 4;
        // 2026-09-25: The transpose is written from `fp8_act_scale` while that is
        // still live, so the two cannot share a buffer.
        let fp8_act_scale_kmajor = fp8_act_scale;
        // 2026-09-25: LoRA scratch, only when `adapter_max_rank > 0`. `max_n` is
        // the widest projection output an adapter can target.
        let (lora_xa, lora_delta, lora_hact, lora_seq_slot) = if config.adapter_max_rank > 0 {
            let max_n = h
                .max(config.intermediate_size)
                .max(q_proj_mul * q_heads * hd);
            (
                m * config.adapter_max_rank * bf16,
                m * max_n * bf16,
                m * config.intermediate_size * bf16,
                m * 4,
            )
        } else {
            (0, 0, 0, 0)
        };

        let (ssd_scratch, gdn_fla_scratch) = regions::ssm_scratch_bytes(config, m, bf16);

        let (q2_dequant_scratch, q2_act_q8) = super::sizes_q2::q2_scratch_sizes(config, m, h, hd);

        let ssm_rowwise_w_bf16 = super::sizes_rowwise::ssm_rowwise_w_bf16_bytes(config);

        // 2026-09-25: `ceil16` rows for the same cuBLASLt pad as `m_pad`.
        let ffn_gate_up_fused = if config.num_experts == 0 {
            let rows = GATEUP_FUSED_MAX_M.div_ceil(16) * 16;
            rows * 2 * config.intermediate_size * bf16
        } else {
            0
        };

        let (ffn_act_q8, ffn_act_a, ffn_act_scale, ffn_act_scale_kmajor) =
            if config.num_experts == 0 {
                let kmax = h.max(config.intermediate_size);
                let kpad = kmax.div_ceil(256) * 256;
                // 2026-09-25: The q8_1 term is `q8_1_scratch_bytes` in
                // metrale-model-layers (`ops/q4k_mmq.rs`). The int8 activation
                // `[ceil16(M), K]` and scale `[ceil16(M), K/32]` terms also hold
                // the NVFP4 (`[M, K/2]`, `[M, K/16]`) and FP8 (`[M, K]`,
                // `[M, K/128]`) forms.
                (
                    m * kpad * 4 + (1 << 20),
                    m_pad * kmax,
                    m_pad * (kmax / 32) * 4,
                    m_pad * (kmax / 128) * 4,
                )
            } else {
                (0, 0, 0, 0)
            };

        let mut sizes = Self {
            hidden_states: m * h * residual_elem,
            residual: m * h * residual_elem,
            norm_output: m * max_dim * bf16,
            // 2026-09-25: `m_pad` rows: the cuBLASLt Q/K/V prefill arm
            // (`prefill_qkv_w8a8.rs`) writes `ceil16(M)` rows of Q here.
            qkv_output: m_pad * qkv_dim * bf16,
            attn_output: (m * config.num_attention_heads * config.head_dim * bf16)
                .max(m * mamba2_d_inner * bf16)
                // 2026-09-25: Absorbed MLA writes [M, q_heads, kv_lora_rank + rope dim].
                .max(if config.kv_lora_rank > 0 {
                    m * config.num_attention_heads
                        * (config.kv_lora_rank + config.qk_rope_head_dim)
                        * bf16
                } else {
                    0
                }),
            gate_logits: if config.num_experts > 0 {
                // 2026-09-25: The router also scores the `zero_expert_num` zero
                // experts, which have no FFN.
                m * (config.num_experts + config.zero_expert_num) * bf16
            } else {
                256
            },
            gate_logits_f32: if config.num_experts > 0 {
                m * (config.num_experts + config.zero_expert_num) * 4
            } else {
                256
            },
            moe_router_in_f32: if config.num_experts > 0 {
                m * h * 4
            } else {
                256
            },
            // 2026-09-25: `ceil16(M)` rows: the cuBLASLt dense-FFN down
            // projection writes its output here.
            moe_output: m.div_ceil(16) * 16 * h * bf16,
            logits: logits_tokens * config.vocab_size * bf16,
            // 2026-09-25: The SSM buffers are also scratch for other layers, so
            // each is the largest of its uses and at least 256 bytes. `ssm_qkvz`
            // holds the QKVZ projection (`ceil16(M)` rows for its cuBLASLt arm),
            // the Mamba-2 in_proj output, prefill K and V, and the MoE shared-expert
            // up output.
            ssm_qkvz: (m_pad * config.ssm_qkvz_size() * bf16)
                .max(m * config.mamba2_in_proj_size() * bf16)
                // 2026-09-25: K at row 0 and V at row M, V `ceil16(M)` rows tall
                // on the cuBLASLt arm (`prefill_qkv_w8a8.rs`).
                .max((m + m_pad) * kv_heads * hd * bf16)
                .max(m * config.shared_expert_intermediate_size * bf16)
                .max(256),
            ssm_ba: (m * config.ssm_ba_size() * bf16)
                .max(m * config.moe_latent_size * bf16)
                // 2026-09-25: MLA uses `ssm_ba` for q_latent [M, q_lora_rank] and
                // later for the K rope rows [M, qk_rope_head_dim], one at a time.
                .max(if config.kv_lora_rank > 0 {
                    (m * config.qk_rope_head_dim * bf16).max(m * config.q_lora_rank * bf16)
                } else {
                    0
                })
                .max(256),
            // 2026-09-25: `ceil16(M)` rows as in `ssm_qkvz`: on a
            // `sequential_qkvz` model the QKVZ projection writes here.
            ssm_deinterleaved: (m_pad * config.ssm_qkvz_size() * bf16)
                .max(m * config.mamba2_d_xbc() * bf16)
                .max(m * q_heads * hd * bf16)
                // 2026-09-25: Absorbed MLA's Q [M, q_heads, kv_lora_rank + rope dim].
                .max(if config.kv_lora_rank > 0 {
                    m * q_heads * (config.kv_lora_rank + config.qk_rope_head_dim) * bf16
                } else {
                    0
                })
                .max(256),
            ssm_gates: (m * config.linear_num_value_heads * 2 * 4).max(256),
            // 2026-09-25: FP32 conv1d output, bounded by the QKVZ width. MLA also
            // uses it for its Q rope rows [M, q_heads * qk_rope_head_dim] BF16.
            ssm_conv_out_f32: (m * config.ssm_qkvz_size() * 4)
                .max(if config.kv_lora_rank > 0 {
                    m * q_heads * config.qk_rope_head_dim * bf16
                } else {
                    0
                })
                .max(256),
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            splitk_workspace,
            gdn_fla_scratch,
            ssd_scratch,
            o_latent: (m * config.o_groups * config.o_lora_rank * bf16).max(256),
            norm_unit_w: max_dim * bf16,
            hc_streams: if config.hc_mult > 0 {
                m * config.hc_mult * h * 4
            } else {
                256
            },
            hc_post: if config.hc_mult > 0 {
                (m * config.hc_mult * 4).max(256)
            } else {
                256
            },
            hc_comb: if config.hc_mult > 0 {
                (m * config.hc_mult * config.hc_mult * 4).max(256)
            } else {
                256
            },
            hc_lowrank_scratch: if config.hc_mult > 0 && config.hc_lowrank > 0 {
                // 2026-09-25: Two layouts share this region
                // (`hyper_connection_lowrank.rs`, `hyper_connection_lowrank_gemm.rs`):
                // - T <= 64: normed FP32 [64, hc*H], then low FP32 [64, rank];
                // - T > 64, in slabs of Ts <= 2048 tokens: normed BF16 [Ts, hc*H],
                //   up_pre BF16 [Ts, hc*H], low BF16 [Ts, rank], inj_pre BF16 [Ts, hc].
                let t = m.min(64);
                let split = t * (config.hc_mult * h + config.hc_lowrank) * 4;
                let ts = m.min(2048);
                let gemm = ts * (2 * config.hc_mult * h + config.hc_lowrank + config.hc_mult) * 2;
                split.max(gemm)
            } else {
                256
            },
            qsa_select_scratch: if config.index_topk > 0 && config.index_compress_ratio > 0 {
                const ROWS: usize = 2048;
                let qkw = (config.index_n_heads + 1) * config.index_head_dim;
                let n_blocks = max_seq_len.div_ceil(config.index_compress_ratio);
                let topk = config.index_topk / config.index_compress_ratio;
                ROWS * qkw * 2
                    + ROWS * config.index_n_heads * config.index_head_dim * 4
                    + ROWS * n_blocks * 4
                    + ROWS * topk * 4
            } else {
                256
            },
            token_ids: (m * 4).max(256),
            ffn_act_q8,
            ffn_act_a,
            ffn_act_scale,
            ffn_act_scale_kmajor,
            ffn_gate_up_fused,
            fp8_act,
            moe_fp8_scratch: super::moe_fp8_scratch::Layout::new(config, m).bytes,
            fp8_act_scale,
            fp8_act_scale_kmajor,
            lora_xa,
            lora_delta,
            lora_hact,
            lora_seq_slot,
            q2_dequant_scratch,
            q2_act_q8,
            ssm_rowwise_w_bf16,
            glm_trimmed: 0,
        };
        sizes.glm_trimmed = super::sizes_glm_trim::trim_for_glm(&mut sizes, config);
        sizes
    }

    /// 2026-09-25: The sum of the sizes, which preflight reserves for the arena.
    /// `o_latent` and `norm_unit_w` are not in it.
    pub fn total_bytes(&self) -> usize {
        self.hidden_states
            + self.residual
            + self.norm_output
            + self.qkv_output
            + self.attn_output
            + self.gate_logits
            + self.gate_logits_f32
            + self.moe_router_in_f32
            + self.moe_output
            + self.logits
            + self.ssm_qkvz
            + self.ssm_ba
            + self.ssm_deinterleaved
            + self.ssm_gates
            + self.ssm_conv_out_f32
            + self.scratch
            + self.expert_gate_out
            + self.expert_up_out
            + self.hc_lowrank_scratch
            + self.qsa_select_scratch
            + self.expert_down_out
            + self.splitk_workspace
            + self.gdn_fla_scratch
            + self.ssd_scratch
            + self.hc_streams
            + self.hc_post
            + self.hc_comb
            + self.token_ids
            + self.ffn_act_q8
            + self.ffn_act_a
            + self.ffn_gate_up_fused
            + self.ffn_act_scale
            + self.ffn_act_scale_kmajor
            + self.fp8_act
            + self.moe_fp8_scratch
            + self.fp8_act_scale
            + self.fp8_act_scale_kmajor
            + self.lora_xa
            + self.lora_delta
            + self.lora_hact
            + self.lora_seq_slot
            + self.q2_dequant_scratch
            + self.q2_act_q8
            + self.ssm_rowwise_w_bf16
    }
}
