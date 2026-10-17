// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Tests for the GLM DFlash batched-verify predicates (`glm_dflash_batch.rs`).
//!
//! Owner: model-layers (speculative).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn ep_admits_truth_table() {
    // 2026-10-09: (protocol, v2, dflash, lever, own_states) -> admitted. Every row of the 32.
    for bits in 0u32..32 {
        let [p, v2, df, lever, own] = [0, 1, 2, 3, 4].map(|i| (bits >> i) & 1 == 1);
        let want = p && v2 && own && (!df || lever);
        assert_eq!(
            batched_verify_ep_admits(p, v2, df, lever, own),
            want,
            "protocol={p} v2={v2} dflash={df} lever={lever} own_states={own}"
        );
    }
    // 2026-10-09: The two rows the lever changes: DFlash on a live v2 GLM serve.
    assert!(!batched_verify_ep_admits(true, true, true, false, true));
    assert!(batched_verify_ep_admits(true, true, true, true, true));
    // 2026-10-09: The lever never admits without the protocol, v2 or own-state layers.
    assert!(!batched_verify_ep_admits(false, true, true, true, true));
    assert!(!batched_verify_ep_admits(true, false, true, true, true));
    assert!(!batched_verify_ep_admits(true, true, true, true, false));
}

#[test]
fn multi_k_pins_four_only_on_glm_with_a_lever() {
    for mt in ["glm5_next", "glm5_next_text"] {
        assert_eq!(glm_dflash_multi_k(mt, true, false), Some(4));
        assert_eq!(glm_dflash_multi_k(mt, false, true), Some(4));
        assert_eq!(glm_dflash_multi_k(mt, true, true), Some(4));
        assert_eq!(glm_dflash_multi_k(mt, false, false), None);
    }
    assert_eq!(glm_dflash_multi_k("qwen3_5", true, true), None);
}

#[test]
fn shape_caps_k_at_the_ep_meta_bound_only_with_comm() {
    // 2026-10-09: gamma 8, 4 sequences, 8 x 4 capture rows.
    for k in 2..=4 {
        assert!(dflash_batch_shape_ok(&[k; 4], 8, 32, true), "k={k} ep");
        assert!(dflash_batch_shape_ok(&[k; 4], 8, 32, false), "k={k} single");
    }
    // 2026-10-09: k=8 (the C=1 width, still pending after a C 1 -> 2 step) batches on one GPU
    // but not across ranks: the EP meta refuses k > 4 at send.
    assert!(dflash_batch_shape_ok(&[8; 4], 8, 32, false));
    assert!(!dflash_batch_shape_ok(&[8; 4], 8, 32, true));
    assert!(!dflash_batch_shape_ok(&[5, 5], 8, 32, true));
    // 2026-10-09: ragged, k=1, gamma<2, too few capture rows.
    assert!(!dflash_batch_shape_ok(&[4, 3], 8, 32, true));
    assert!(!dflash_batch_shape_ok(&[1, 1], 8, 32, true));
    assert!(!dflash_batch_shape_ok(&[2, 2], 1, 32, false));
    assert!(!dflash_batch_shape_ok(&[4; 5], 8, 32, true));
}

#[test]
fn worker_skips_the_stash_only_for_a_rank0_proposer_under_the_lever() {
    assert!(worker_skips_verify_stash(true, Some(false)));
    assert!(!worker_skips_verify_stash(true, Some(true)));
    assert!(!worker_skips_verify_stash(true, None));
    assert!(!worker_skips_verify_stash(false, Some(false)));
}

#[test]
fn verdict_words_clamp_to_k_minus_one() {
    assert_eq!(batch_verdict_words(&[3, 0, 9], &[4, 4, 4]), [3, 0, 3]);
    assert_eq!(batch_verdict_words(&[1, 2], &[2, 3]), [1, 2]);
    assert_eq!(batch_verdict_words(&[5, 5], &[2, 3]), [1, 2]);
}
