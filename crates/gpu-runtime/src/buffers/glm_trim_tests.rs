// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Tests for the GLM-5.3 arena trim (`sizes_glm_trim.rs`): the family gate, the trimmed sizes against an untrimmed twin, and a trimmed arena on the mock backend (NULL entries, zeroing, debug checksum, release).
//!
//! Owner: gpu-runtime (buffer arena).
//! Invariants: none beyond the types.

use super::sizes_glm_trim::{applies, trim};
use super::*;
use crate::gpu::mock::MockGpuBackend;
use metrale_core::scope::ModelResource;

/// 2026-10-04: One TP2 rank of GLM-5.3-Flash, built on the Qwen3-Next preset.
/// Every field the arena sizing reads matches `parse_config` of the checkpoint's
/// `config.json` with the TP2 head split: the two gave identical `BufferSizes` at
/// four sets of serve limits, the ship limits among them (checked 2026-10-04).
fn glm53_tp2() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".to_string();
    c.attn_gated = false;
    c.hidden_size = 4096;
    c.intermediate_size = 12_288;
    c.vocab_size = 154_880;
    c.num_attention_heads = 32;
    c.num_key_value_heads = 32;
    c.linear_num_key_heads = 32;
    c.num_experts = 288;
    c.num_experts_per_tok = 8;
    c.moe_intermediate_size = 2048;
    c.shared_expert_intermediate_size = 2048;
    c.kv_lora_rank = 512;
    c.q_lora_rank = 1536;
    c.hc_mult = 4;
    c.index_n_heads = 32;
    c.index_head_dim = 128;
    c.index_topk = 2048;
    c
}

/// 2026-10-04: The same shape under another model type.
fn twin(model_type: &str) -> ModelConfig {
    let mut c = glm53_tp2();
    c.model_type = model_type.to_string();
    c
}

/// 2026-10-04: The eighteen entries the trim zeroes, listed apart from `trim`
/// so a change to either list fails here.
fn trimmed(s: &BufferSizes) -> [(&'static str, usize); 18] {
    [
        ("qkv_output", s.qkv_output),
        ("attn_output", s.attn_output),
        ("gate_logits", s.gate_logits),
        ("gate_logits_f32", s.gate_logits_f32),
        ("moe_router_in_f32", s.moe_router_in_f32),
        ("ssm_qkvz", s.ssm_qkvz),
        ("ssm_ba", s.ssm_ba),
        ("ssm_deinterleaved", s.ssm_deinterleaved),
        ("ssm_gates", s.ssm_gates),
        ("ssm_conv_out_f32", s.ssm_conv_out_f32),
        ("expert_gate_out", s.expert_gate_out),
        ("expert_up_out", s.expert_up_out),
        ("expert_down_out", s.expert_down_out),
        ("gdn_fla_scratch", s.gdn_fla_scratch),
        ("fp8_act", s.fp8_act),
        ("fp8_act_scale", s.fp8_act_scale),
        ("fp8_act_scale_kmajor", s.fp8_act_scale_kmajor),
        ("moe_fp8_scratch", s.moe_fp8_scratch),
    ]
}

#[test]
fn the_gate_is_the_glm_family_and_the_lever_not_a_shared_field() {
    let glm = glm53_tp2();
    assert!(applies(&glm, None));
    assert!(applies(&glm, Some("1")));
    assert!(!applies(&glm, Some("0")), "`0` turns the trim off");
    assert!(applies(&twin("glm5_next_text"), None));
    // 2026-10-04: These share `kv_lora_rank`, `hc_mult`, `index_topk` and the
    // routed experts with the GLM shape, and read the trimmed entries.
    for other in ["deepseek_v4", "deepseek_v41", "kimi_k3", "qwen3_next"] {
        assert!(!applies(&twin(other), None), "{other}");
    }
}

#[test]
fn glm_sizes_drop_exactly_the_trimmed_entries() {
    // 2026-10-04: The ship limits: max_batch_tokens 8196, max_seq_len 65536,
    // 16-token blocks, 4 sequences. Assumes `METRALE_GLM_ARENA_TRIM` unset.
    let glm = BufferSizes::from_config(&glm53_tp2(), 8196, 65_536, 16, 4);
    let full = BufferSizes::from_config(&twin("deepseek_v4"), 8196, 65_536, 16, 4);
    assert_eq!(full.glm_trimmed, 0, "no trim outside the GLM-5.3 family");
    let mut sum = 0;
    for ((name, kept), (_, untrimmed)) in trimmed(&glm).into_iter().zip(trimmed(&full)) {
        assert_eq!(kept, 0, "{name}");
        assert!(untrimmed > 0, "{name} is sized without the trim");
        sum += untrimmed;
    }
    assert_eq!(glm.glm_trimmed, sum);
    assert_eq!(glm.total_bytes(), full.total_bytes() - sum);
    // 2026-10-04: Every other entry is unchanged.
    let mut expect = full.clone();
    assert_eq!(trim(&mut expect), sum);
    expect.glm_trimmed = sum;
    assert_eq!(format!("{glm:?}"), format!("{expect:?}"));
    // 2026-10-04: 3,423.9 MiB of the 4,242.9 MiB arena the ship config logs.
    assert_eq!(full.total_bytes(), 4_449_005_908);
    assert_eq!(sum, 3_590_249_220);
}

#[test]
fn a_trimmed_arena_holds_null_entries_that_the_helpers_skip() {
    let gpu = MockGpuBackend::new();
    let mut arena = BufferArena::new(&glm53_tp2(), 16, 4096, 16, 4, &gpu).unwrap();
    let s = arena.sizes().clone();
    assert!(s.glm_trimmed > 0);
    let null = [
        arena.qkv_output(),
        arena.attn_output(),
        arena.gate_logits(),
        arena.gate_logits_f32(),
        arena.moe_router_in_f32(),
        arena.ssm_qkvz(),
        arena.ssm_ba(),
        arena.ssm_deinterleaved(),
        arena.ssm_gates(),
        arena.ssm_conv_out_f32(),
        arena.expert_gate_out(),
        arena.expert_up_out(),
        arena.expert_down_out(),
        arena.gdn_fla_scratch(),
        arena.fp8_act(),
        arena.fp8_act_scale(),
        arena.fp8_act_scale_kmajor(),
        arena.moe_fp8_scratch,
    ];
    assert!(null.iter().all(|p| p.is_null()));
    // 2026-10-04: Nothing else is allocated: `total_bytes` leaves out only
    // `o_latent` and `norm_unit_w`.
    assert_eq!(
        gpu.live_bytes(),
        Some(s.total_bytes() + s.o_latent + s.norm_unit_w)
    );

    // 2026-10-04: The mock refuses a memset or copy on a NULL pointer, so each
    // helper returning Ok means the NULL entries were skipped; the live ones
    // are still cleared.
    let ones = vec![1u8; s.hidden_states];
    gpu.copy_h2d(&ones, arena.hidden_states()).unwrap();
    arena.zero_all(&gpu, 0).unwrap();
    let hidden = gpu.read_alloc(arena.hidden_states()).unwrap();
    assert!(hidden.iter().all(|b| *b == 0));
    arena.zero_all_rows(&gpu, 0, 1).unwrap();
    arena.zero_prefill_essentials(&gpu, 0).unwrap();
    // 2026-10-04: Of the sixteen probed buffers, five are live.
    let before = gpu.d2h_blocking_count();
    arena.debug_buffer_checksum(&gpu, 0, "trim-test");
    assert_eq!(gpu.d2h_blocking_count() - before, 5);

    arena.release(&gpu).unwrap();
    assert_eq!(gpu.live_bytes(), Some(0));
}
