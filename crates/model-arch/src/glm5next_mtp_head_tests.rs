// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `drafter_context_rows`: the GLM drafter's row cap follows the DSA indexer reservation.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: none beyond the types.

use super::drafter_context_rows;
use crate::glm5next_dsa::Glm5NextDsaConfig;

/// 2026-09-25: GLM-5.3's DSA shape; the same values as `glm5next_dsa::state::tests::cfg`.
fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

/// 2026-09-25: A context longer than the indexer reservation is clamped to the reservation.
#[test]
fn a_declared_context_past_the_dsa_reservation_does_not_size_the_drafter() {
    let c = cfg();
    assert_eq!(drafter_context_rows(524_288, &c), 16_384);
    assert_eq!(drafter_context_rows(262_144, &c), 16_384);
}

/// 2026-09-25: A context at or below the reservation passes through unchanged.
#[test]
fn a_context_under_the_ceiling_is_untouched() {
    let c = cfg();
    assert_eq!(drafter_context_rows(8_192, &c), 8_192);
    assert_eq!(drafter_context_rows(16_384, &c), 16_384);
}

/// 2026-09-25: The cap is `max_dsa_context`, not a literal: it follows `max_context` and
/// rounds down to whole `index_kpool` pools. A hardcoded 16,384 passes the two tests above
/// and fails this one.
#[test]
fn the_cap_tracks_the_indexer_reservation_not_a_constant() {
    let mut c = cfg();
    c.max_context = 65_536;
    assert_eq!(drafter_context_rows(524_288, &c), 65_536);
    assert_eq!(drafter_context_rows(32_768, &c), 32_768);
    c.max_context = 65_538;
    assert_eq!(
        drafter_context_rows(524_288, &c),
        65_536,
        "whole pools only"
    );
}

/// 2026-10-05: `METRALE_GLM_DENSE_FP8=1` converts `eh_proj` (`dense_fp8::register_mtp`), so no
/// head code may read `module.eh_proj` through a direct GEMV or GEMM: every read goes through
/// `dense_fp8::route`, and the registered pointer never reaches a BF16 kernel.
#[test]
fn eh_proj_is_read_only_through_route() {
    let files = [
        ("glm5next_mtp_head.rs", include_str!("glm5next_mtp_head.rs")),
        (
            "glm5next_mtp_head/batch.rs",
            include_str!("glm5next_mtp_head/batch.rs"),
        ),
        (
            "glm5next_mtp_head/proposer.rs",
            include_str!("glm5next_mtp_head/proposer.rs"),
        ),
    ];
    let mut routed = 0;
    for (name, src) in files {
        for (i, line) in src.lines().enumerate() {
            if line.contains("module.eh_proj") {
                // The only allowed mentions: the pointer handed to `route` (the `let` in
                // `eh_proj_one` / `eh_proj_wide`, or batch.rs' `route` argument).
                assert!(
                    line.contains("let (h, w) =") || line.trim() == "self.module.eh_proj.weight,",
                    "{name}:{}: direct eh_proj read: {line}",
                    i + 1
                );
                routed += 1;
            }
        }
    }
    assert_eq!(routed, 3, "eh_proj_one, eh_proj_wide and the batched route");
}
