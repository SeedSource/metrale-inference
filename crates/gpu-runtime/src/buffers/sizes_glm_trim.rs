// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The GLM-5.3 arena trim: the arena entries a GLM-5.3-family model never reads are sized 0, so the arena leaves them `DevicePtr::NULL` and the KV pool, sized after the arena, gets the memory.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The gate is the model family, `model_type` `glm5_next` or `glm5_next_text`
//!   (the two types `Glm5NextWeightLoader` loads), plus `METRALE_GLM_ARENA_TRIM`
//!   (on unless `0`). It is never a config field the family shares:
//!   DeepSeek-V4 and Kimi-K3 also have `kv_lora_rank`, `hc_mult`, `index_topk`
//!   and routed experts, and read these entries.
//! - Only the entries [`trim`] lists change, and only to 0.
//! - The readers of those entries are the Qwen3 attention and SSM layers, the
//!   generic MoE, dense-FFN and MTP head (metrale-model-layers), Nemotron and the
//!   DeepSeek-V4 MTP head (metrale-model-arch), and the arena's own zeroing and
//!   debug helpers, which skip a NULL entry. A GLM-5.3 model builds none of
//!   those layers: every layer is a `Glm5NextLayer`, the drafter is
//!   `Glm5NextMtpHead` or the DFlash head, and that code reads only
//!   `norm_output`, `moe_output` and the hyper-connection buffers, while the
//!   engine reads `hidden_states`, `residual`, `norm_output`, `logits`,
//!   `scratch`, `token_ids` and `hc_streams`.

use super::sizes::BufferSizes;
use metrale_config::ModelConfig;

/// 2026-10-04: Whether the trim applies to `config`: a GLM-5.3-family model and
/// `METRALE_GLM_ARENA_TRIM` not `0`. The arena sizing, the GDN two-phase
/// prefill scratch (metrale-model-engine) and its preflight reserve (`met`) all
/// read this, so the three agree.
pub fn glm_arena_trim_active(config: &ModelConfig) -> bool {
    applies(
        config,
        std::env::var("METRALE_GLM_ARENA_TRIM").ok().as_deref(),
    )
}

/// 2026-10-04: [`glm_arena_trim_active`] with the lever value passed in.
pub(super) fn applies(config: &ModelConfig, lever: Option<&str>) -> bool {
    matches!(config.model_type.as_str(), "glm5_next" | "glm5_next_text") && lever != Some("0")
}

/// 2026-10-04: [`trim`] when [`glm_arena_trim_active`]; returns the bytes it
/// removed, 0 when the trim does not apply.
pub(super) fn trim_for_glm(sizes: &mut BufferSizes, config: &ModelConfig) -> usize {
    if glm_arena_trim_active(config) {
        trim(sizes)
    } else {
        0
    }
}

/// 2026-10-04: Set the eighteen entries no GLM-5.3 code reads to 0 and return
/// the bytes removed. For the GLM-5.3-Flash config at the ship limits
/// (`max_batch_tokens` 8196, TP2) that is 3,590,249,220 bytes (3,423.9 MiB) of
/// a 4,242.9 MiB arena.
pub(super) fn trim(s: &mut BufferSizes) -> usize {
    [
        &mut s.qkv_output,
        &mut s.attn_output,
        &mut s.gate_logits,
        &mut s.gate_logits_f32,
        &mut s.moe_router_in_f32,
        &mut s.ssm_qkvz,
        &mut s.ssm_ba,
        &mut s.ssm_deinterleaved,
        &mut s.ssm_gates,
        &mut s.ssm_conv_out_f32,
        &mut s.expert_gate_out,
        &mut s.expert_up_out,
        &mut s.expert_down_out,
        &mut s.gdn_fla_scratch,
        &mut s.fp8_act,
        &mut s.fp8_act_scale,
        &mut s.fp8_act_scale_kmajor,
        &mut s.moe_fp8_scratch,
    ]
    .into_iter()
    .map(std::mem::take)
    .sum()
}
