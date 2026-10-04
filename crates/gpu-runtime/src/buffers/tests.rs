// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the buffer arena's sizing (`BufferSizes`) and
//! allocation (`BufferArena`), on `ModelConfig::qwen3_next_80b_nvfp4` and a
//! Qwen3.8-27B GDN fixture, with the mock backend.
//!
//! Owner: gpu-runtime (buffer arena).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn mixed_dense_moe_sizes_for_widest_ffn() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.intermediate_size = 12_288;
    cfg.num_experts = 256;
    cfg.num_experts_per_tok = 10;
    cfg.moe_intermediate_size = 1_024;

    let sizes = BufferSizes::from_config(&cfg, 4, 4096, 16, 32);
    // 2026-09-25: The dense intermediate (12288) is wider than the routed
    // width (top_k 10 x 1024 = 10240), and the expert buffers take the wider.
    // Rows are `k_max` in `sizes.rs`: max(M, 3) rounded up to 16.
    let rows = 4_usize.div_ceil(16) * 16;
    assert!(cfg.intermediate_size > cfg.num_experts_per_tok * cfg.moe_intermediate_size);
    assert_eq!(sizes.expert_gate_out, rows * 12_288 * 2);
    assert_eq!(sizes.expert_up_out, rows * 12_288 * 2);
}
use crate::gpu::mock::MockGpuBackend;
use std::collections::HashSet;

#[test]
fn test_buffer_sizes_qwen3() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let sizes = BufferSizes::from_config(&cfg, 1, 4096, 16, 32);

    // 2026-09-25: M = 1, BF16 (2 bytes). `qkv_output`, `ssm_qkvz` and
    // `ssm_deinterleaved` are sized for `m_pad` = ceil16(M) = 16 rows
    // (`sizes.rs`), the rows a cuBLASLt arm writes. qkv: 16 * (16*2 + 2*2)
    // * 256 * 2 (gated Q, K, V). ssm_qkvz: 16 * (16*128 + 16*128 + 32*128 +
    // 32*128) * 2. `ssm_ba` and `ssm_gates` are at the 256-byte floor.
    assert_eq!(sizes.hidden_states, 4096);
    assert_eq!(sizes.qkv_output, 294912);
    assert_eq!(sizes.attn_output, 8192);
    assert_eq!(sizes.gate_logits, 1024);
    assert_eq!(sizes.logits, 303872);
    assert_eq!(sizes.ssm_qkvz, 393216);
    assert_eq!(sizes.ssm_ba, 256);
    assert_eq!(sizes.ssm_deinterleaved, 393216);
    assert_eq!(sizes.ssm_gates, 256);
}

#[test]
fn test_buffer_arena_alloc() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let arena = BufferArena::new(&cfg, 128, 4096, 16, 32, &gpu).unwrap();

    assert_eq!(arena.max_batch_tokens(), 128);
    let sizes = arena.sizes();
    let buffers = [
        ("hidden_states", arena.hidden_states(), sizes.hidden_states),
        ("residual", arena.residual(), sizes.residual),
        ("norm_output", arena.norm_output(), sizes.norm_output),
        ("qkv_output", arena.qkv_output(), sizes.qkv_output),
        ("attn_output", arena.attn_output(), sizes.attn_output),
        ("gate_logits", arena.gate_logits(), sizes.gate_logits),
        (
            "gate_logits_f32",
            arena.gate_logits_f32(),
            sizes.gate_logits_f32,
        ),
        (
            "moe_router_in_f32",
            arena.moe_router_in_f32(),
            sizes.moe_router_in_f32,
        ),
        ("moe_output", arena.moe_output(), sizes.moe_output),
        ("logits", arena.logits(), sizes.logits),
        ("ssm_qkvz", arena.ssm_qkvz(), sizes.ssm_qkvz),
        ("ssm_ba", arena.ssm_ba(), sizes.ssm_ba),
        (
            "ssm_deinterleaved",
            arena.ssm_deinterleaved(),
            sizes.ssm_deinterleaved,
        ),
        ("ssm_gates", arena.ssm_gates(), sizes.ssm_gates),
        (
            "ssm_conv_out_f32",
            arena.ssm_conv_out_f32(),
            sizes.ssm_conv_out_f32,
        ),
        ("scratch", arena.scratch(), sizes.scratch),
        (
            "expert_gate_out",
            arena.expert_gate_out(),
            sizes.expert_gate_out,
        ),
        ("expert_up_out", arena.expert_up_out(), sizes.expert_up_out),
        (
            "expert_down_out",
            arena.expert_down_out(),
            sizes.expert_down_out,
        ),
        (
            "splitk_workspace",
            arena.splitk_workspace(),
            sizes.splitk_workspace,
        ),
        ("o_latent", arena.o_latent(), sizes.o_latent),
        ("norm_unit_w", arena.norm_unit_w(), sizes.norm_unit_w),
        ("hc_streams", arena.hc_streams(), sizes.hc_streams),
        ("hc_post", arena.hc_post(), sizes.hc_post),
        ("hc_comb", arena.hc_comb(), sizes.hc_comb),
        ("ssd_scratch", arena.ssd_scratch(), sizes.ssd_scratch),
        (
            "gdn_fla_scratch",
            arena.gdn_fla_scratch(),
            sizes.gdn_fla_scratch,
        ),
        ("token_ids", arena.token_ids(), sizes.token_ids),
        ("ffn_act_q8", arena.ffn_act_q8(), sizes.ffn_act_q8),
        ("ffn_act_a", arena.ffn_act_a(), sizes.ffn_act_a),
        ("ffn_act_scale", arena.ffn_act_scale(), sizes.ffn_act_scale),
        (
            "ffn_act_scale_kmajor",
            arena.ffn_act_scale_kmajor(),
            sizes.ffn_act_scale_kmajor,
        ),
        (
            "ffn_gate_up_fused",
            arena.ffn_gate_up_fused(),
            sizes.ffn_gate_up_fused,
        ),
        ("fp8_act", arena.fp8_act(), sizes.fp8_act),
        (
            "moe_fp8_scratch",
            arena.moe_fp8_scratch,
            sizes.moe_fp8_scratch,
        ),
        ("fp8_act_scale", arena.fp8_act_scale(), sizes.fp8_act_scale),
        (
            "fp8_act_scale_kmajor",
            arena.fp8_act_scale_kmajor(),
            sizes.fp8_act_scale_kmajor,
        ),
        (
            "q2_dequant_scratch",
            arena.q2_dequant_scratch(),
            sizes.q2_dequant_scratch,
        ),
        ("q2_act_q8", arena.q2_act_q8(), sizes.q2_act_q8),
        ("lora_xa", arena.lora_xa(), sizes.lora_xa),
        ("lora_delta", arena.lora_delta(), sizes.lora_delta),
        ("lora_hact", arena.lora_hact(), sizes.lora_hact),
        ("lora_seq_slot", arena.lora_seq_slot(), sizes.lora_seq_slot),
        // 2026-09-25: The list must name every arena buffer: the final
        // assert is `alloc_count() == allocated.len()`.
        (
            "hc_lowrank_scratch",
            arena.hc_lowrank_scratch(),
            sizes.hc_lowrank_scratch,
        ),
        (
            "qsa_select_scratch",
            arena.qsa_select_scratch(),
            sizes.qsa_select_scratch,
        ),
    ];
    let mut allocated = HashSet::new();
    for (name, ptr, bytes) in buffers {
        if bytes == 0 {
            assert!(ptr.is_null(), "{name} must be null when disabled");
        } else {
            assert!(!ptr.is_null(), "{name} must be allocated");
            assert!(
                allocated.insert(ptr.0),
                "{name} aliases another arena buffer"
            );
            assert_eq!(gpu.read_alloc(ptr).unwrap().len(), bytes, "{name} size");
        }
    }
    assert_eq!(gpu.alloc_count(), allocated.len());
    assert!(
        gpu.read_alloc(arena.norm_unit_w())
            .unwrap()
            .iter()
            .all(|byte| *byte == 0),
        "unit RMSNorm weight must be zero-initialized"
    );
}

#[test]
fn q2_dequant_scratch_covers_largest_projection() {
    // 2026-09-25: One BF16 dequant scratch serves every keep-packed
    // projection (`q2_dequant_scratch_bytes`), so it must hold the largest
    // `[N, K]` among them: FFN, fused qkvz, attention q and k/v.
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let bytes = q2_dequant_scratch_bytes(&cfg);
    let h = cfg.hidden_size;
    let ffn = cfg.intermediate_size * h * 2;
    let qkvz = cfg.ssm_qkvz_size() * h * 2;
    let q_mul = if cfg.attn_gated { 2 } else { 1 };
    let q = cfg.num_attention_heads * q_mul * cfg.head_dim * h * 2;
    let kv = cfg.num_key_value_heads * cfg.head_dim * h * 2;
    assert!(bytes >= ffn, "scratch {bytes} < FFN {ffn}");
    assert!(bytes >= qkvz, "scratch {bytes} < qkvz {qkvz}");
    assert!(bytes >= q, "scratch {bytes} < q_proj {q}");
    assert!(bytes >= kv, "scratch {bytes} < kv_proj {kv}");
    assert!(bytes > 0);
}

#[test]
fn q2_scratch_flags_are_explicit_partitions() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let m = 3;
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let dequant_bytes = q2_dequant_scratch_bytes(&cfg);
    let kmax = h
        .max(cfg.intermediate_size)
        .max(cfg.num_attention_heads * hd);
    let mmq_bytes = m * kmax.div_ceil(256) * 256 * 4 + (1 << 20);

    assert_eq!(
        sizes_q2::q2_scratch_sizes_for(&cfg, m, h, hd, false, false),
        (0, 0)
    );
    assert_eq!(
        sizes_q2::q2_scratch_sizes_for(&cfg, m, h, hd, true, false),
        (dequant_bytes, 0)
    );
    assert_eq!(
        sizes_q2::q2_scratch_sizes_for(&cfg, m, h, hd, false, true),
        (0, mmq_bytes)
    );
}

#[test]
fn test_buffer_sizes_scale_with_batch() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let s1 = BufferSizes::from_config(&cfg, 1, 4096, 16, 32);
    let s128 = BufferSizes::from_config(&cfg, 128, 4096, 16, 32);
    assert_eq!(s128.hidden_states, s1.hidden_states * 128);
    // 2026-09-25: Logits rows are `m.min(160.max(rows + 1))` (`sizes.rs`
    // `logits_tokens`); at m = 128 that is 128.
    assert_eq!(s128.logits, 128 * cfg.vocab_size * 2);
}

/// 2026-09-25: Sizing does not change with `max_batch_size` while the
/// decode-meta rows fit the 160-row verify envelope in `sizes.rs` (`bt_rows`
/// for scratch, the 160-row floor of `logits_tokens`). Logits grow from
/// bs = 160 (161 rows); at bs = 192 the decode-meta block
/// (`DecodeMetaLayout::meta_bytes`) is larger than the verify block and
/// scratch grows with it.
#[test]
fn test_buffer_sizes_decode_meta_widening() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let s32 = BufferSizes::from_config(&cfg, 8192, 4096, 16, 32);
    // 2026-09-25: Up to 32, the decode-meta rows are the 32-row floor.
    for bs in [1usize, 31, 32] {
        let s = BufferSizes::from_config(&cfg, 8192, 4096, 16, bs);
        assert_eq!(s.total_bytes(), s32.total_bytes(), "bs={bs}");
        assert_eq!(s.scratch, s32.scratch, "bs={bs}");
        assert_eq!(s.logits, s32.logits, "bs={bs}");
    }
    // 2026-09-25: 33 to 159 rows stay inside the 160-row envelope.
    for bs in [33usize, 64, 128, 159] {
        let s = BufferSizes::from_config(&cfg, 8192, 4096, 16, bs);
        assert_eq!(s.total_bytes(), s32.total_bytes(), "bs={bs}");
    }
    let s160 = BufferSizes::from_config(&cfg, 8192, 4096, 16, 160);
    assert_eq!(s160.logits, 161 * cfg.vocab_size * 2);
    // 2026-09-25: Decode-meta block at R = 192: 24R + R * max_blocks * 4,
    // after the 32768-byte fixed region.
    let s192 = BufferSizes::from_config(&cfg, 8192, 4096, 16, 192);
    assert_eq!(s192.logits, 193 * cfg.vocab_size * 2);
    let max_blocks = 4096 / 16 + 1;
    assert!(s192.scratch >= 32768 + 24 * 192 + 192 * max_blocks * 4);
}

// 2026-09-25: The row-wise FP8 GDN prefill BF16-weight slab
// (`ssm_rowwise_w_bf16`): zero unless `METRALE_FP8_ROWWISE` is armed, and
// exact at the 27B geometry. The lever is passed to
// `ssm_rowwise_w_bf16_bytes_for` rather than set with `set_var`, which is
// process-global and would race the other tests in this binary.

/// 2026-09-25: Qwen3.8-27B's GDN geometry: hidden 5120 and 64 layers, every
/// fourth full attention, so 48 GDN layers (`kernels/gb10/qwen3.8-27b/MODEL.toml`
/// `hidden_size`, `layers_total`, `layers_linear_attention`), with 16x128 key
/// heads and 48x128 value heads. The MODEL.toml has no GDN head fields; these
/// match `qwen38_27b` in model-arch's `predicted_residency_tests.rs`. Other
/// fields stay at `qwen3_next_80b_nvfp4`'s values.
fn qwen38_27b() -> ModelConfig {
    use metrale_config::LayerType;
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 5120;
    c.num_hidden_layers = 64;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c.full_attention_interval = 4;
    c.layer_types = (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c
}

#[test]
fn rowwise_bf16_slab_is_sized_only_when_the_lever_is_armed() {
    let cfg = qwen38_27b();
    assert_eq!(
        ssm_rowwise_w_bf16_bytes_for(&cfg, false),
        0,
        "an unarmed METRALE_FP8_ROWWISE must leave the default recipe's ledger \
         byte-identical — the arena allocates NULL for a 0-byte entry"
    );

    // 2026-09-25: in_proj_qkvz: (16*128 + 16*128 + 48*128 + 48*128) = 16384
    // rows x 5120 x 2 B. out_proj: [5120, 48*128] x 2 B.
    assert_eq!(cfg.ssm_qkvz_size(), 16_384);
    let qkvz = 16_384 * 5_120 * 2;
    let out_proj = 5_120 * (48 * 128) * 2;
    assert_eq!(qkvz, 167_772_160);
    assert_eq!(ssm_rowwise_w_bf16_layer_bytes(&cfg), qkvz + out_proj);
    assert_eq!(cfg.num_ssm_layers(), 48);
    assert_eq!(
        ssm_rowwise_w_bf16_bytes_for(&cfg, true),
        48 * (qkvz + out_proj),
        "48 GDN layers x (in_proj_qkvz + out_proj) — 10.31 GiB, which is what \
         the preflight ring fitter now prices instead of discovering at \
         layer 36"
    );
}

/// 2026-09-25: The slab is counted in `total_bytes()`, which the server's
/// preflight passes to `headroom::post_load_yardstick` as its `arena` term
/// (`preflight.rs` `buffer_arena_bytes`).
#[test]
fn rowwise_bf16_slab_is_counted_in_total_bytes() {
    let cfg = qwen38_27b();
    let mut sizes = BufferSizes::from_config(&cfg, 64, 4096, 16, 32);
    // 2026-09-25: Set to 0 first: `from_config` reads METRALE_FP8_ROWWISE
    // from the environment.
    sizes.ssm_rowwise_w_bf16 = 0;
    let before = sizes.total_bytes();
    // 2026-09-25: A sentinel, not the sizing function's value, so a sizing
    // function that returned 0 could not make this pass.
    sizes.ssm_rowwise_w_bf16 = 4096;
    assert_eq!(sizes.total_bytes(), before + 4096);
}

#[test]
fn zero_head_bytes_trims_m_and_ceil16_buffers() {
    use super::accessors::zero_head_bytes;
    // 2026-10-04: GLM-5.3 serve: max_batch_tokens 8193, m_pad = 8208 rows.
    let (m, w) = (8193, 16_384 * 2);
    assert_eq!(zero_head_bytes(m * w, m, 1), w);
    assert_eq!(zero_head_bytes(8208 * w, m, 1), 16 * w);
    assert_eq!(zero_head_bytes(8208 * w, m, 17), 32 * w);
    // whole buffer when every row is in use or the size matches neither layout
    assert_eq!(zero_head_bytes(8208 * w, m, m), 8208 * w);
    assert_eq!(zero_head_bytes(1_000_003, m, 1), 1_000_003);
    // m already a multiple of 16: unchanged behaviour
    assert_eq!(zero_head_bytes(4096 * w, 4096, 1), w);
}
