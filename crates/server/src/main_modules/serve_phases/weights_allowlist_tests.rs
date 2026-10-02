// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the per-model lists in [`super`] that leave checkpoint
//! tensors unloaded, asserting both members and non-members.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use super::{skip_activation_scales, skip_activation_scales_for, skip_mtp};
use metrale_config::ModelConfig;

fn cfg(model_type: &str) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = model_type.to_string();
    c
}

#[test]
fn activation_scales_are_skipped_only_for_the_listed_models() {
    assert!(skip_activation_scales(&cfg("glm5_next")));
    assert!(skip_activation_scales(&cfg("qwen4_exp")));

    // 2026-09-26: `step3p7` reads `input_scale` on its own loader path.
    for keep in ["step3p7", "qwen3_5_moe", "minimax_m2", "kimi_k3", "llama"] {
        assert!(
            !skip_activation_scales(&cfg(keep)),
            "{keep} must keep its activation scales"
        );
    }
}

/// 2026-09-26: The two lists are independent: `glm5_next` skips activation
/// scales but keeps `mtp.*`.
#[test]
fn the_mtp_skip_list_is_unchanged_by_the_activation_scale_list() {
    assert!(skip_mtp(&cfg("qwen4_exp")));
    assert!(!skip_mtp(&cfg("glm5_next")));
}

/// 2026-10-03: `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1` keeps GLM's activation scales (its
/// loader defers them instead) and changes no other model.
#[test]
fn the_glm_cutlass_w4a4_lever_keeps_only_glm_activation_scales() {
    assert!(skip_activation_scales_for("glm5_next", false));
    assert!(!skip_activation_scales_for("glm5_next", true));
    for lever in [false, true] {
        assert!(skip_activation_scales_for("qwen4_exp", lever));
        for keep in ["step3p7", "qwen3_5_moe", "minimax_m2", "kimi_k3", "llama"] {
            assert!(!skip_activation_scales_for(keep, lever), "{keep} lever={lever}");
        }
    }
}

/// 2026-10-02: The drafter's own tables are left unloaded only when every one
/// is `[vocab, hidden]`; the estimate then drops exactly their bytes.
#[test]
fn drafter_tables_are_shared_only_when_every_table_has_the_expected_shape() {
    use super::shareable_bytes_from_headers as shareable;
    let (v, h) = (154_880u64, 2_048u64);
    let tbl = v * h * 2;
    let hdrs = |lm_shape: Vec<u64>| {
        vec![
            ("embed_tokens.weight".to_string(), vec![v, h], tbl),
            ("lm_head.weight".to_string(), lm_shape, tbl),
            ("layers.0.q_proj.weight".to_string(), vec![4096, h], 4096 * h * 2),
        ]
    };
    // both tables match: both counted, the layer tensor is not
    assert_eq!(shareable(&hdrs(vec![v, h]), v as usize, h as usize), 2 * tbl);
    // one table differs: nothing is shared
    assert_eq!(shareable(&hdrs(vec![v, h + 1]), v as usize, h as usize), 0);
    // wrong target hidden size: nothing is shared
    assert_eq!(shareable(&hdrs(vec![v, h]), v as usize, h as usize + 8), 0);
    // no tables in the checkpoint: nothing to subtract
    assert_eq!(shareable(&[], v as usize, h as usize), 0);
}
