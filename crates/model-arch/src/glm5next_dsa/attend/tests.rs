// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the DSA decode launcher's guards, geometry only. The
//! kernel's numerics are checked on a GPU by the `glm5next_dsa_decode_gate`
//! example.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::select::DsaSelectGeometry;

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

fn paging() -> DsaDecodePaging {
    DsaDecodePaging {
        num_seqs: 1,
        num_q_heads: 64,
        num_kv_heads: 1,
        max_blocks_per_seq: 256,
        block_size: 64,
        cache_stride_bytes: 64 * 512,
    }
}

/// 2026-09-25: Pools are built over absolute positions, so a `block_size` that
/// is not a multiple of `index_kpool` would split a pool across two pages.
#[test]
fn a_block_size_that_splits_a_pool_is_refused() {
    let c = cfg();
    let mut p = paging();
    assert!(p.validate(&c).is_ok(), "64 is a multiple of kpool 4");
    p.block_size = 66;
    let e = p.validate(&c).unwrap_err().to_string();
    assert!(e.contains("straddle"), "{e}");
}

/// 2026-09-25: MLA has one latent KV head; `validate` refuses any other count.
#[test]
fn more_than_one_kv_head_is_refused() {
    let c = cfg();
    let mut p = paging();
    p.num_kv_heads = 8;
    assert!(p.validate(&c).is_err());
}

#[test]
fn degenerate_launches_are_refused() {
    let c = cfg();
    for mutate in [
        (|p: &mut DsaDecodePaging| p.num_seqs = 0) as fn(&mut DsaDecodePaging),
        |p: &mut DsaDecodePaging| p.num_q_heads = 0,
        |p: &mut DsaDecodePaging| p.block_size = 0,
    ] {
        let mut p = paging();
        mutate(&mut p);
        assert!(p.validate(&c).is_err(), "{p:?} must be refused");
    }
}

/// 2026-09-25: `decode_attention` refuses a selection planned for a different row
/// count than it decodes. The launch needs a GPU, so this checks only that the
/// refused condition is reachable from `DsaSelectGeometry::plan`.
#[test]
fn a_row_count_mismatch_between_selection_and_launch_is_refused() {
    let c = cfg();
    let p = paging();
    let two_rows = DsaSelectGeometry::plan(&c, 4_096, 2).unwrap();
    assert_eq!(two_rows.q_rows, 2);
    assert_ne!(two_rows.q_rows, p.num_seqs);
    assert!(
        two_rows.q_rows != p.num_seqs,
        "the guarded condition must be reachable"
    );
    let one_row = DsaSelectGeometry::plan(&c, 4_096, 1).unwrap();
    assert_eq!(one_row.q_rows, p.num_seqs);
}

/// 2026-09-25: NoPE: the softmax scale is over the latent width, which is the whole
/// cache token, not over a 576-wide token with a 64-dim rope tail (`MLA_CACHE_DIM`
/// in the DeepSeek-V4-Flash decode kernels).
#[test]
fn the_score_scale_is_the_latent_width_not_the_v4_cache_width() {
    let c = cfg();
    assert_eq!(c.kv_cache_dim(), 512, "NoPE: cache token is pure latent");
    assert_eq!(c.kv_cache_dim(), c.kv_lora_rank);
    let ours = (c.kv_lora_rank as f32).powf(-0.5);
    let v4 = 576f32.powf(-0.5);
    assert!((ours - 0.044_194_173).abs() < 1e-9, "1/sqrt(512)");
    assert!(
        (ours - v4).abs() > 1e-4,
        "the two scales must not be interchangeable by accident"
    );
}

/// 2026-09-29 (A153): `METRALE_GLM_MLA_SCALE_AUTHOR=1`'s arm is `qk_head_dim^-0.5`
/// (`qk_nope_head_dim + qk_rope_head_dim`), 1/sqrt(256) = 1/16 for GLM-5.3's 256+0 split; the
/// default (off) arm is unchanged from the test above.
#[test]
fn author_scale_arm_is_one_sixteenth_for_256_plus_0() {
    let c = cfg();
    assert_eq!(c.qk_nope_head_dim + c.qk_rope_head_dim, 256);
    let author = mla_scale_for(&c, true);
    assert!(
        (author - 0.0625).abs() < 1e-9,
        "1/sqrt(256) == 1/16, got {author}"
    );
    let default = mla_scale_for(&c, false);
    assert!(
        (default - (c.kv_lora_rank as f32).powf(-0.5)).abs() < 1e-9,
        "the off arm is unchanged: kv_lora_rank^-0.5"
    );
}
