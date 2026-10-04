// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The per-model GPU buffer arena: every intermediate tensor of a forward pass, allocated once and reused by every step.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Every buffer is allocated at its [`BufferSizes`] size. The gated ones
//!   (`ssd_scratch`, `gdn_fla_scratch`, `ffn_*`, `moe_fp8_scratch`, `q2_*`,
//!   `lora_*`, `ssm_rowwise_w_bf16`, and the entries the GLM-5.3 trim can
//!   zero, `sizes_glm_trim.rs`) are `DevicePtr::NULL` when that size is 0.
//! - No device memory is allocated after construction, and `release` frees
//!   every pointer field.

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;
use metrale_config::ModelConfig;

mod accessors;
mod debug_checksum;
pub mod decode_meta;
mod moe_fp8_scratch;
mod release;
mod rowwise_slab;
mod sizes;
mod sizes_glm_trim;
mod sizes_q12;
mod sizes_q2;
mod sizes_rowwise;
pub use decode_meta::{DECODE_META_MAX_ROWS, DECODE_META_MIN_ROWS, DecodeMetaLayout};
pub use moe_fp8_scratch::MoeFp8Scratch;
pub use sizes::{BufferSizes, GATEUP_FUSED_MAX_M};
pub use sizes_glm_trim::glm_arena_trim_active;
pub use sizes_q2::q2_dequant_scratch_bytes;
pub use sizes_q12::{
    Q12_SIZING_STREAMS, q12_batched_scratch_bytes, q12_batched_scratch_bytes_varlen,
};
pub use sizes_rowwise::{
    ssm_rowwise_w_bf16_bytes, ssm_rowwise_w_bf16_bytes_for, ssm_rowwise_w_bf16_layer_bytes,
};

/// 2026-09-25: The GPU buffers of one forward pass, `M = max_batch_tokens`.
/// [`BufferSizes::from_config`] holds each buffer's size formula.
pub struct BufferArena {
    hidden_states: DevicePtr,
    residual: DevicePtr,
    norm_output: DevicePtr,
    qkv_output: DevicePtr,
    attn_output: DevicePtr,
    gate_logits: DevicePtr,
    gate_logits_f32: DevicePtr,
    moe_router_in_f32: DevicePtr,
    moe_output: DevicePtr,
    logits: DevicePtr,
    ssm_qkvz: DevicePtr,
    ssm_ba: DevicePtr,
    ssm_deinterleaved: DevicePtr,
    ssm_gates: DevicePtr,
    ssm_conv_out_f32: DevicePtr,
    scratch: DevicePtr,
    expert_gate_out: DevicePtr,
    expert_up_out: DevicePtr,
    expert_down_out: DevicePtr,
    splitk_workspace: DevicePtr,
    o_latent: DevicePtr,
    norm_unit_w: DevicePtr,
    hc_streams: DevicePtr,
    hc_post: DevicePtr,
    hc_comb: DevicePtr,
    hc_lowrank_scratch: DevicePtr,
    qsa_select_scratch: DevicePtr,
    gdn_fla_scratch: DevicePtr,
    ssd_scratch: DevicePtr,
    token_ids: DevicePtr,
    ffn_act_q8: DevicePtr,
    ffn_act_a: DevicePtr,
    ffn_act_scale: DevicePtr,
    ffn_act_scale_kmajor: DevicePtr,
    ffn_gate_up_fused: DevicePtr,
    fp8_act: DevicePtr,
    /// 2026-09-25: The grouped FP8 MoE activation, scale and tile-worklist slab that every
    /// MoE layer reuses in turn; see [`BufferArena::moe_fp8_scratch`].
    moe_fp8_scratch: DevicePtr,
    fp8_act_scale: DevicePtr,
    fp8_act_scale_kmajor: DevicePtr,
    q2_dequant_scratch: DevicePtr,
    lora_xa: DevicePtr,
    lora_delta: DevicePtr,
    lora_hact: DevicePtr,
    lora_seq_slot: DevicePtr,
    q2_act_q8: DevicePtr,
    ssm_rowwise_w_bf16: DevicePtr,
    /// 2026-09-25: Bytes carved from `ssm_rowwise_w_bf16` so far; it only grows.
    ssm_rowwise_w_bf16_used: std::sync::atomic::AtomicUsize,
    max_batch_tokens: usize,
    /// 2026-09-25: Decode-metadata layout for the serve `max_batch_size`; the
    /// batched-decode upload (`upload_batch_metadata_fixed` / `_at` in
    /// metrale-model-engine) takes its offsets from it.
    decode_meta: DecodeMetaLayout,
    sizes: BufferSizes,
}

impl BufferArena {
    /// 2026-09-25: Size the buffers with [`BufferSizes::from_config`] and allocate them.
    pub fn new(
        config: &ModelConfig,
        max_batch_tokens: usize,
        max_seq_len: usize,
        kv_block_size: usize,
        max_batch_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let sizes = BufferSizes::from_config(
            config,
            max_batch_tokens,
            max_seq_len,
            kv_block_size,
            max_batch_size,
        );
        Self::from_sizes(config, sizes, max_batch_tokens, max_batch_size, gpu)
    }

    /// 2026-09-25: [`BufferArena::new`] with the sizes passed in.
    ///
    /// `BufferSizes::from_config` reads the environment for the env-gated
    /// entries (`q2_*`, `ssm_rowwise_w_bf16`), so a test that needs one of
    /// them sized builds the sizes itself and calls this instead of setting a
    /// process-global variable. On an allocation error the buffers already
    /// allocated are not freed.
    pub fn from_sizes(
        config: &ModelConfig,
        sizes: BufferSizes,
        max_batch_tokens: usize,
        max_batch_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let decode_meta = DecodeMetaLayout::for_max_batch_size(max_batch_size);

        let hidden_states = gpu.alloc(sizes.hidden_states)?;
        let residual = gpu.alloc(sizes.residual)?;
        let norm_output = gpu.alloc(sizes.norm_output)?;
        let qkv_output = alloc_or_null(gpu, sizes.qkv_output)?;
        let attn_output = alloc_or_null(gpu, sizes.attn_output)?;
        let gate_logits = alloc_or_null(gpu, sizes.gate_logits)?;
        let gate_logits_f32 = alloc_or_null(gpu, sizes.gate_logits_f32)?;
        let moe_router_in_f32 = alloc_or_null(gpu, sizes.moe_router_in_f32)?;
        let moe_output = gpu.alloc(sizes.moe_output)?;
        let logits = gpu.alloc(sizes.logits)?;
        let ssm_qkvz = alloc_or_null(gpu, sizes.ssm_qkvz)?;
        let ssm_ba = alloc_or_null(gpu, sizes.ssm_ba)?;
        let ssm_deinterleaved = alloc_or_null(gpu, sizes.ssm_deinterleaved)?;
        let ssm_gates = alloc_or_null(gpu, sizes.ssm_gates)?;
        let ssm_conv_out_f32 = alloc_or_null(gpu, sizes.ssm_conv_out_f32)?;
        let scratch = gpu.alloc(sizes.scratch)?;
        let expert_gate_out = alloc_or_null(gpu, sizes.expert_gate_out)?;
        let expert_up_out = alloc_or_null(gpu, sizes.expert_up_out)?;
        let expert_down_out = alloc_or_null(gpu, sizes.expert_down_out)?;
        let splitk_workspace = gpu.alloc(sizes.splitk_workspace)?;
        let o_latent = gpu.alloc(sizes.o_latent)?;
        // 2026-09-25: All zeros: the `rms_norm` kernel scales by 1 + weight, so a
        // zero weight is a plain normalize.
        let norm_unit_w = gpu.alloc(sizes.norm_unit_w)?;
        gpu.memset(norm_unit_w, 0, sizes.norm_unit_w)?;
        let hc_streams = gpu.alloc(sizes.hc_streams)?;
        let hc_post = gpu.alloc(sizes.hc_post)?;
        let hc_comb = gpu.alloc(sizes.hc_comb)?;
        let hc_lowrank_scratch = gpu.alloc(sizes.hc_lowrank_scratch)?;
        let qsa_select_scratch = gpu.alloc(sizes.qsa_select_scratch)?;
        // 2026-09-25: A gated entry of 0 bytes is left NULL, which its callers
        // read as "not available".
        let ssd_scratch = if sizes.ssd_scratch > 0 {
            gpu.alloc(sizes.ssd_scratch)?
        } else {
            DevicePtr::NULL
        };
        let gdn_fla_scratch = if sizes.gdn_fla_scratch > 0 {
            gpu.alloc(sizes.gdn_fla_scratch)?
        } else {
            DevicePtr::NULL
        };
        let token_ids = gpu.alloc(sizes.token_ids)?;
        let ffn_act_q8 = if sizes.ffn_act_q8 > 0 {
            gpu.alloc(sizes.ffn_act_q8)?
        } else {
            DevicePtr::NULL
        };
        let ffn_act_a = if sizes.ffn_act_a > 0 {
            gpu.alloc(sizes.ffn_act_a)?
        } else {
            DevicePtr::NULL
        };
        let ffn_act_scale = if sizes.ffn_act_scale > 0 {
            gpu.alloc(sizes.ffn_act_scale)?
        } else {
            DevicePtr::NULL
        };
        let ffn_act_scale_kmajor = if sizes.ffn_act_scale_kmajor > 0 {
            gpu.alloc(sizes.ffn_act_scale_kmajor)?
        } else {
            DevicePtr::NULL
        };
        let ffn_gate_up_fused = if sizes.ffn_gate_up_fused > 0 {
            gpu.alloc(sizes.ffn_gate_up_fused)?
        } else {
            DevicePtr::NULL
        };
        let fp8_act = alloc_or_null(gpu, sizes.fp8_act)?;
        let moe_fp8_scratch = if sizes.moe_fp8_scratch > 0 {
            gpu.alloc(sizes.moe_fp8_scratch)?
        } else {
            DevicePtr::NULL
        };
        let fp8_act_scale = alloc_or_null(gpu, sizes.fp8_act_scale)?;
        let fp8_act_scale_kmajor = alloc_or_null(gpu, sizes.fp8_act_scale_kmajor)?;
        let q2_dequant_scratch = if sizes.q2_dequant_scratch > 0 {
            gpu.alloc(sizes.q2_dequant_scratch)?
        } else {
            DevicePtr::NULL
        };
        let lora_xa = if sizes.lora_xa > 0 {
            gpu.alloc(sizes.lora_xa)?
        } else {
            DevicePtr::NULL
        };
        let lora_delta = if sizes.lora_delta > 0 {
            gpu.alloc(sizes.lora_delta)?
        } else {
            DevicePtr::NULL
        };
        let lora_hact = if sizes.lora_hact > 0 {
            gpu.alloc(sizes.lora_hact)?
        } else {
            DevicePtr::NULL
        };
        let lora_seq_slot = if sizes.lora_seq_slot > 0 {
            gpu.alloc(sizes.lora_seq_slot)?
        } else {
            DevicePtr::NULL
        };
        let q2_act_q8 = if sizes.q2_act_q8 > 0 {
            gpu.alloc(sizes.q2_act_q8)?
        } else {
            DevicePtr::NULL
        };
        let ssm_rowwise_w_bf16 = if sizes.ssm_rowwise_w_bf16 > 0 {
            gpu.alloc(sizes.ssm_rowwise_w_bf16)?
        } else {
            DevicePtr::NULL
        };

        tracing::info!(
            "Buffer arena: {} tokens × {:.1} MB total (attn_out={:.1}MB, ssm_deint={:.1}MB, kv_lora_rank={})",
            max_batch_tokens,
            sizes.total_bytes() as f64 / (1024.0 * 1024.0),
            sizes.attn_output as f64 / (1024.0 * 1024.0),
            sizes.ssm_deinterleaved as f64 / (1024.0 * 1024.0),
            config.kv_lora_rank,
        );
        if sizes.glm_trimmed > 0 {
            tracing::info!(
                "GLM-5.3 arena trim ENGAGED (METRALE_GLM_ARENA_TRIM): {} bytes ({:.1} MiB) of \
                 arena buffers this model never reads left unallocated; the GDN two-phase \
                 prefill scratch is skipped too",
                sizes.glm_trimmed,
                sizes.glm_trimmed as f64 / (1024.0 * 1024.0),
            );
        }

        Ok(Self {
            hidden_states,
            residual,
            norm_output,
            qkv_output,
            attn_output,
            gate_logits,
            gate_logits_f32,
            moe_router_in_f32,
            moe_output,
            logits,
            ssm_qkvz,
            ssm_ba,
            ssm_deinterleaved,
            ssm_gates,
            ssm_conv_out_f32,
            scratch,
            expert_gate_out,
            expert_up_out,
            expert_down_out,
            splitk_workspace,
            o_latent,
            norm_unit_w,
            hc_streams,
            hc_post,
            hc_comb,
            hc_lowrank_scratch,
            qsa_select_scratch,
            gdn_fla_scratch,
            ssd_scratch,
            token_ids,
            ffn_act_q8,
            ffn_act_a,
            ffn_act_scale,
            ffn_act_scale_kmajor,
            ffn_gate_up_fused,
            fp8_act,
            moe_fp8_scratch,
            fp8_act_scale,
            fp8_act_scale_kmajor,
            q2_dequant_scratch,
            lora_xa,
            lora_delta,
            lora_hact,
            lora_seq_slot,
            q2_act_q8,
            ssm_rowwise_w_bf16,
            ssm_rowwise_w_bf16_used: std::sync::atomic::AtomicUsize::new(0),
            max_batch_tokens,
            decode_meta,
            sizes,
        })
    }
}

/// 2026-10-04: `gpu.alloc(bytes)`, or `DevicePtr::NULL` for 0 bytes:
/// `cuMemAlloc_v2` refuses a zero-byte request.
fn alloc_or_null(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    if bytes > 0 {
        gpu.alloc(bytes)
    } else {
        Ok(DevicePtr::NULL)
    }
}

#[cfg(test)]
mod glm_trim_tests;
#[cfg(test)]
mod tests;
