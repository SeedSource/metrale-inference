// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: `BufferArena` accessors (device pointers and allocated byte sizes) and the zeroing helpers.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Every `*_bytes` accessor returns the allocated size of its buffer, from
//!   `BufferSizes`.
//! - The zeroing helpers only enqueue `memset_async` on the given stream; the
//!   first failed enqueue is returned and later buffers are not zeroed.
//! - The zeroing helpers skip a `DevicePtr::NULL` or 0-byte buffer (an entry the
//!   GLM-5.3 trim left unallocated, `sizes_glm_trim.rs`) and enqueue nothing for it.

use super::{BufferArena, sizes::BufferSizes};
use crate::gpu::{DevicePtr, GpuBackend};

/// 2026-10-04: Zero `bytes` at `ptr` on `stream`; nothing for a NULL pointer or
/// 0 bytes, since `cuMemsetD8Async` has no NULL guard.
fn memset_live(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    bytes: usize,
    stream: u64,
) -> anyhow::Result<()> {
    if ptr.is_null() || bytes == 0 {
        return Ok(());
    }
    gpu.memset_async(ptr, 0, bytes, stream)
}

impl BufferArena {
    pub fn hidden_states(&self) -> DevicePtr {
        self.hidden_states
    }
    pub fn residual(&self) -> DevicePtr {
        self.residual
    }
    pub fn norm_output(&self) -> DevicePtr {
        self.norm_output
    }
    /// 2026-09-25: Allocated byte size of `norm_output`, the bound the attention
    /// prefill o_proj's cuBLASLt arm (`paged_oproj.rs`) checks its `ceil16(M)`
    /// rows against.
    pub fn norm_output_bytes(&self) -> usize {
        self.sizes.norm_output
    }
    pub fn qkv_output(&self) -> DevicePtr {
        self.qkv_output
    }
    /// 2026-09-25: Allocated byte size of `qkv_output`, the bound the multi-seq
    /// decode W8A8 arm (`multi_seq/w8a8_decode.rs`) and the cuBLASLt Q/K/V
    /// prefill arm check their `ceil16(M)` rows against.
    pub fn qkv_output_bytes(&self) -> usize {
        self.sizes.qkv_output
    }
    pub fn attn_output(&self) -> DevicePtr {
        self.attn_output
    }
    pub fn gate_logits(&self) -> DevicePtr {
        self.gate_logits
    }
    pub fn gate_logits_f32(&self) -> DevicePtr {
        self.gate_logits_f32
    }
    pub fn moe_router_in_f32(&self) -> DevicePtr {
        self.moe_router_in_f32
    }
    pub fn moe_output(&self) -> DevicePtr {
        self.moe_output
    }
    pub fn logits(&self) -> DevicePtr {
        self.logits
    }
    pub fn ssm_qkvz(&self) -> DevicePtr {
        self.ssm_qkvz
    }
    /// 2026-09-25: Allocated byte size of `ssm_qkvz`, the QKVZ projection's
    /// destination on an interleaved model, checked by the cuBLASLt arm against
    /// its `ceil16(M)` rows.
    pub fn ssm_qkvz_bytes(&self) -> usize {
        self.sizes.ssm_qkvz
    }
    pub fn ssm_ba(&self) -> DevicePtr {
        self.ssm_ba
    }
    /// 2026-09-25: Sequential `[Q|K|V|Z]` rows after deinterleaving.
    pub fn ssm_deinterleaved(&self) -> DevicePtr {
        self.ssm_deinterleaved
    }
    /// 2026-09-25: Allocated byte size of `ssm_deinterleaved`, the QKVZ
    /// projection's destination on a sequential model; same padded-M check.
    pub fn ssm_deinterleaved_bytes(&self) -> usize {
        self.sizes.ssm_deinterleaved
    }
    /// 2026-09-25: GDN gates in FP32: per token, `linear_num_value_heads` gate
    /// values, then as many beta values.
    pub fn ssm_gates(&self) -> DevicePtr {
        self.ssm_gates
    }
    /// 2026-09-25: FP32 conv1d output of the SSM layers.
    pub fn ssm_conv_out_f32(&self) -> DevicePtr {
        self.ssm_conv_out_f32
    }
    /// 2026-09-25: Scratch for MoE routing and the metadata uploads (layouts in `sizes.rs`).
    pub fn scratch(&self) -> DevicePtr {
        self.scratch
    }
    /// 2026-09-25: Mamba-2 SSD chunked-scan scratch (dt | dA_cumsum | CB); NULL without Mamba-2 layers.
    pub fn ssd_scratch(&self) -> DevicePtr {
        self.ssd_scratch
    }
    /// 2026-09-25: Token ids `[M]` u32 of the pass, read by the DeepSeek-V4
    /// hash-routed MoE layers. The caller uploads them before the layer loop,
    /// and under CUDA-graph decode before each replay.
    pub fn token_ids(&self) -> DevicePtr {
        self.token_ids
    }
    /// 2026-09-25: Allocated byte size of `scratch`, the bound for metadata staging.
    pub fn scratch_bytes(&self) -> usize {
        self.sizes.scratch
    }
    pub fn expert_gate_out(&self) -> DevicePtr {
        self.expert_gate_out
    }
    pub fn expert_up_out(&self) -> DevicePtr {
        self.expert_up_out
    }
    /// 2026-09-25: Allocated byte size of `expert_gate_out`, equal to that of
    /// `expert_up_out` (debug-asserted).
    pub fn expert_gate_out_bytes(&self) -> usize {
        debug_assert_eq!(self.sizes.expert_gate_out, self.sizes.expert_up_out);
        self.sizes.expert_gate_out
    }
    pub fn moe_output_bytes(&self) -> usize {
        self.sizes.moe_output
    }
    pub fn expert_down_out(&self) -> DevicePtr {
        self.expert_down_out
    }
    pub fn expert_down_out_bytes(&self) -> usize {
        self.sizes.expert_down_out
    }
    pub fn gate_logits_bytes(&self) -> usize {
        self.sizes.gate_logits
    }
    pub fn logits_bytes(&self) -> usize {
        self.sizes.logits
    }
    pub fn attn_output_bytes(&self) -> usize {
        self.sizes.attn_output
    }
    /// 2026-09-25: GDN FLA prefill scratch (W | U | S | uc | gc, carved by the
    /// caller); `DevicePtr::NULL` unless both linear-attention head dims are 128.
    pub fn gdn_fla_scratch(&self) -> DevicePtr {
        self.gdn_fla_scratch
    }
    /// 2026-09-25: Shared dense-FFN q8_1 activation scratch (Q4_K MMQ); NULL for MoE.
    pub fn ffn_act_q8(&self) -> DevicePtr {
        self.ffn_act_q8
    }
    /// 2026-09-25: Shared dense-FFN int8 / NVFP4 / FP8 activation scratch; NULL for MoE.
    pub fn ffn_act_a(&self) -> DevicePtr {
        self.ffn_act_a
    }
    /// 2026-09-25: The scales paired with `ffn_act_a`; NULL for MoE.
    pub fn ffn_act_scale(&self) -> DevicePtr {
        self.ffn_act_scale
    }
    /// 2026-09-25: `[ceil16(GATEUP_FUSED_MAX_M), 2 * intermediate]` BF16 output
    /// of the fused dense-FFN gate+up decode GEMM: gate at column 0, up at
    /// column `intermediate`. NULL for MoE.
    pub fn ffn_gate_up_fused(&self) -> DevicePtr {
        self.ffn_gate_up_fused
    }
    /// 2026-09-25: Allocated byte size of `ffn_gate_up_fused`; the fused arm
    /// selects itself only when its `[ceil16(m), 2 * intermediate]` output fits.
    pub fn ffn_gate_up_fused_bytes(&self) -> usize {
        self.sizes.ffn_gate_up_fused
    }
    pub fn ffn_act_a_bytes(&self) -> usize {
        self.sizes.ffn_act_a
    }
    pub fn ffn_act_scale_bytes(&self) -> usize {
        self.sizes.ffn_act_scale
    }
    /// 2026-09-25: `[K/128, ceil16(M)]` transpose of the dense-FFN FP8 activation
    /// scales for the cuBLASLt block-scaled GEMM; NULL for MoE.
    pub fn ffn_act_scale_kmajor(&self) -> DevicePtr {
        self.ffn_act_scale_kmajor
    }
    pub fn ffn_act_scale_kmajor_bytes(&self) -> usize {
        self.sizes.ffn_act_scale_kmajor
    }
    /// 2026-09-25: FP8 activation scratch for the prefill projections.
    pub fn fp8_act(&self) -> DevicePtr {
        self.fp8_act
    }
    pub fn fp8_act_bytes(&self) -> usize {
        self.sizes.fp8_act
    }
    /// 2026-09-25: One FP32 scale per 128 elements of `fp8_act`.
    pub fn fp8_act_scale(&self) -> DevicePtr {
        self.fp8_act_scale
    }
    pub fn fp8_act_scale_bytes(&self) -> usize {
        self.sizes.fp8_act_scale
    }
    /// 2026-09-25: `[K/128, ceil16(M)]` transpose of `fp8_act_scale` for the
    /// cuBLASLt block-scaled projections.
    pub fn fp8_act_scale_kmajor(&self) -> DevicePtr {
        self.fp8_act_scale_kmajor
    }
    pub fn fp8_act_scale_kmajor_bytes(&self) -> usize {
        self.sizes.fp8_act_scale_kmajor
    }
    /// 2026-09-25: BF16 dequant scratch for keep-packed Q2_0 prefill, reused by
    /// every projection; NULL unless `METRALE_GGUF_NATIVE_Q2=1`.
    pub fn q2_dequant_scratch(&self) -> DevicePtr {
        self.q2_dequant_scratch
    }
    pub fn q2_dequant_scratch_bytes(&self) -> usize {
        self.sizes.q2_dequant_scratch
    }
    /// 2026-09-25: q8_1 activation scratch for keep-packed Q2_0 MMQ prefill; NULL
    /// unless `METRALE_GGUF_NATIVE_Q2_MMQ=1`.
    pub fn q2_act_q8(&self) -> DevicePtr {
        self.q2_act_q8
    }
    pub fn q2_act_q8_bytes(&self) -> usize {
        self.sizes.q2_act_q8
    }
    pub fn splitk_workspace(&self) -> DevicePtr {
        self.splitk_workspace
    }
    /// 2026-09-25: Grouped O-projection latent `[M, o_groups * o_lora_rank]` BF16.
    pub fn o_latent(&self) -> DevicePtr {
        self.o_latent
    }
    /// 2026-09-25: All-zero BF16 weight: `rms_norm` with it is a plain normalize.
    pub fn norm_unit_w(&self) -> DevicePtr {
        self.norm_unit_w
    }
    /// 2026-09-25: Hyper-connection residual streams `[M, hc_mult, hidden]` FP32.
    pub fn hc_streams(&self) -> DevicePtr {
        self.hc_streams
    }

    /// 2026-09-25: Low-rank hyper-connection scratch; its two layouts are
    /// described in `sizes.rs`.
    pub fn hc_lowrank_scratch(&self) -> DevicePtr {
        self.hc_lowrank_scratch
    }
    /// 2026-09-25: QSA prefill-selection scratch shared by the indexer layers;
    /// `layers/qsa_select.rs` carves it as `sizes.rs` sizes it.
    pub fn qsa_select_scratch(&self) -> DevicePtr {
        self.qsa_select_scratch
    }
    /// 2026-09-25: Hyper-connection `post` weights `[M, hc_mult]` F32.
    pub fn hc_post(&self) -> DevicePtr {
        self.hc_post
    }
    /// 2026-09-25: Hyper-connection `comb` matrix `[M, hc_mult, hc_mult]` F32.
    pub fn hc_comb(&self) -> DevicePtr {
        self.hc_comb
    }
    pub fn max_batch_tokens(&self) -> usize {
        self.max_batch_tokens
    }
    /// 2026-09-25: The batched-decode metadata layout.
    pub fn decode_meta(&self) -> super::DecodeMetaLayout {
        self.decode_meta
    }
    pub fn sizes(&self) -> &BufferSizes {
        &self.sizes
    }

    /// 2026-09-25: LoRA shrink scratch `xa = x @ Aᵀ`, `[M, adapter_max_rank]`
    /// BF16; `DevicePtr::NULL` without an adapter.
    pub fn lora_xa(&self) -> DevicePtr {
        self.lora_xa
    }
    pub fn lora_xa_bytes(&self) -> usize {
        self.sizes.lora_xa
    }
    /// 2026-09-25: LoRA expand scratch `delta = xa @ Bᵀ` BF16; `DevicePtr::NULL`
    /// without an adapter.
    pub fn lora_delta(&self) -> DevicePtr {
        self.lora_delta
    }
    pub fn lora_delta_bytes(&self) -> usize {
        self.sizes.lora_delta
    }
    /// 2026-09-25: LoRA hidden-activation scratch `[M, intermediate_size]` BF16;
    /// `DevicePtr::NULL` without an adapter.
    pub fn lora_hact(&self) -> DevicePtr {
        self.lora_hact
    }
    pub fn lora_hact_bytes(&self) -> usize {
        self.sizes.lora_hact
    }
    /// 2026-09-25: LoRA adapter slot per prefill token, `[M]` i32;
    /// `DevicePtr::NULL` without an adapter.
    pub fn lora_seq_slot(&self) -> DevicePtr {
        self.lora_seq_slot
    }

    /// 2026-09-25: Zero `hidden_states`, `residual`, `gate_logits`, the three
    /// expert outputs and `moe_output`: seven memsets, where `zero_all` issues
    /// eighteen. The prefill paths call this on a sequence's first chunk when
    /// they do not call `zero_all`.
    pub fn zero_prefill_essentials(&self, gpu: &dyn GpuBackend, stream: u64) -> anyhow::Result<()> {
        memset_live(gpu, self.hidden_states, self.sizes.hidden_states, stream)?;
        memset_live(gpu, self.residual, self.sizes.residual, stream)?;
        memset_live(gpu, self.gate_logits, self.sizes.gate_logits, stream)?;
        memset_live(
            gpu,
            self.expert_gate_out,
            self.sizes.expert_gate_out,
            stream,
        )?;
        memset_live(gpu, self.expert_up_out, self.sizes.expert_up_out, stream)?;
        memset_live(
            gpu,
            self.expert_down_out,
            self.sizes.expert_down_out,
            stream,
        )?;
        memset_live(gpu, self.moe_output, self.sizes.moe_output, stream)?;
        Ok(())
    }

    /// 2026-09-25: `zero_all` limited to the first `tokens` rows of each
    /// token-major buffer: for a buffer whose size is a multiple of
    /// `max_batch_tokens`, `size / max_batch_tokens * tokens` bytes; any other
    /// buffer, and every buffer when `tokens >= max_batch_tokens`, is zeroed
    /// whole. `splitk_workspace`, `logits` and `scratch` are zeroed whole. The
    /// caller must read only rows `0..tokens`. 2026-10-04: buffers sized on
    /// `ceil16(max_batch_tokens)` rows are trimmed too (see `zero_head_bytes`).
    /// Measured 2026-08-28 on
    /// GLM-5.3-Flash (nsys, `max_batch_tokens = 4096`): `zero_all` took 8.01 ms
    /// of an 85 ms decode step.
    pub fn zero_all_rows(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        tokens: usize,
    ) -> anyhow::Result<()> {
        let m = self.max_batch_tokens.max(1);
        let head = |n: usize| zero_head_bytes(n, m, tokens);
        for (ptr, n) in [
            (self.hidden_states, self.sizes.hidden_states),
            (self.residual, self.sizes.residual),
            (self.norm_output, self.sizes.norm_output),
            (self.qkv_output, self.sizes.qkv_output),
            (self.attn_output, self.sizes.attn_output),
            (self.gate_logits, self.sizes.gate_logits),
            (self.moe_output, self.sizes.moe_output),
            (self.ssm_qkvz, self.sizes.ssm_qkvz),
            (self.ssm_ba, self.sizes.ssm_ba),
            (self.ssm_deinterleaved, self.sizes.ssm_deinterleaved),
            (self.ssm_gates, self.sizes.ssm_gates),
            (self.ssm_conv_out_f32, self.sizes.ssm_conv_out_f32),
            (self.expert_gate_out, self.sizes.expert_gate_out),
            (self.expert_up_out, self.sizes.expert_up_out),
            (self.expert_down_out, self.sizes.expert_down_out),
        ] {
            memset_live(gpu, ptr, head(n), stream)?;
        }
        memset_live(
            gpu,
            self.splitk_workspace,
            self.sizes.splitk_workspace,
            stream,
        )?;
        memset_live(gpu, self.logits, self.sizes.logits, stream)?;
        memset_live(gpu, self.scratch, self.sizes.scratch, stream)?;
        Ok(())
    }

    /// 2026-09-25: Zero, in full, the eighteen buffers `zero_all_rows` covers.
    /// The others (`gate_logits_f32`, `moe_router_in_f32`, `o_latent`,
    /// `norm_unit_w`, the hyper-connection, QSA, FLA and SSD scratch,
    /// `token_ids`, `ffn_*`, `fp8_*`, `q2_*`, `lora_*` and the row-wise slab)
    /// keep their contents.
    pub fn zero_all(&self, gpu: &dyn GpuBackend, stream: u64) -> anyhow::Result<()> {
        memset_live(gpu, self.hidden_states, self.sizes.hidden_states, stream)?;
        memset_live(gpu, self.residual, self.sizes.residual, stream)?;
        memset_live(gpu, self.norm_output, self.sizes.norm_output, stream)?;
        memset_live(gpu, self.qkv_output, self.sizes.qkv_output, stream)?;
        memset_live(gpu, self.attn_output, self.sizes.attn_output, stream)?;
        memset_live(gpu, self.gate_logits, self.sizes.gate_logits, stream)?;
        memset_live(gpu, self.moe_output, self.sizes.moe_output, stream)?;
        memset_live(gpu, self.ssm_qkvz, self.sizes.ssm_qkvz, stream)?;
        memset_live(gpu, self.ssm_ba, self.sizes.ssm_ba, stream)?;
        memset_live(
            gpu,
            self.ssm_deinterleaved,
            self.sizes.ssm_deinterleaved,
            stream,
        )?;
        memset_live(gpu, self.ssm_gates, self.sizes.ssm_gates, stream)?;
        memset_live(
            gpu,
            self.ssm_conv_out_f32,
            self.sizes.ssm_conv_out_f32,
            stream,
        )?;
        memset_live(
            gpu,
            self.splitk_workspace,
            self.sizes.splitk_workspace,
            stream,
        )?;
        memset_live(
            gpu,
            self.expert_gate_out,
            self.sizes.expert_gate_out,
            stream,
        )?;
        memset_live(gpu, self.expert_up_out, self.sizes.expert_up_out, stream)?;
        memset_live(
            gpu,
            self.expert_down_out,
            self.sizes.expert_down_out,
            stream,
        )?;
        memset_live(gpu, self.logits, self.sizes.logits, stream)?;
        memset_live(gpu, self.scratch, self.sizes.scratch, stream)?;
        Ok(())
    }
}

/// 2026-10-04: Bytes of a token-major buffer of `n` bytes that `zero_all_rows`
/// clears for `tokens` rows, with `m = max_batch_tokens`. A buffer of `m` rows
/// keeps the first `tokens` rows. A buffer of `ceil16(m)` rows (`m_pad` and
/// `k_max` in `sizes.rs`, which the cuBLASLt arms address in 16-row tiles)
/// keeps the first `ceil16(tokens)` rows. Anything else, and any buffer when
/// `tokens >= m`, is cleared whole. Before this, `m_pad` buffers were cleared
/// whole whenever `m` was not a multiple of 16: on GLM-5.3 with
/// `max_batch_tokens = 8193` that was ~2.1 GB per spec-off decode step,
/// 10.8 ms of 62.7 (race-decode 2026-10-04, nsys on `ra-r3`).
pub(crate) fn zero_head_bytes(n: usize, m: usize, tokens: usize) -> usize {
    let m = m.max(1);
    if tokens >= m {
        return n;
    }
    if n.is_multiple_of(m) {
        return n / m * tokens;
    }
    let m16 = m.next_multiple_of(16);
    if n.is_multiple_of(m16) {
        return n / m16 * tokens.next_multiple_of(16).min(m16);
    }
    n
}
