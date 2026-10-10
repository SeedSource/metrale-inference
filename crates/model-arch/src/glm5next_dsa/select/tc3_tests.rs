// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Host tests of `METRALE_GLM_DSA_SCORES_TC3`: the parse, the dispatch envelope, the
//! grid, the BF16 q region size, and the `.cu` entry points and defines.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::select::tc2::{SCORES_TC2_ROWS, scores_tc2_smem};
use crate::glm5next_dsa::{DSA_MODULE, Glm5NextDsaConfig};

#[test]
fn dsa_scores_tc3_parse() {
    assert!(parse_scores_tc3(Some("1")));
    assert!(parse_scores_tc3(Some(" 1 ")));
    for v in [
        None,
        Some(""),
        Some("0"),
        Some("on"),
        Some("true"),
        Some("2"),
    ] {
        assert!(!parse_scores_tc3(v), "{v:?}");
    }
}

/// 2026-10-10: Engaged only on top of a tc2 launch with `>= 32` rows and a BF16 q region.
#[test]
fn dsa_scores_tc3_selection_envelope() {
    let ok = scores_tc3_for;
    assert!(ok(true, true, true, 128, true));
    assert!(ok(true, true, true, SCORES_TC3_MIN_ROWS, true));
    assert!(!ok(false, true, true, 128, true), "lever off");
    assert!(!ok(true, false, true, 128, true), "entry points absent");
    assert!(!ok(true, true, false, 128, true), "tc2 not selected");
    assert!(!ok(true, true, true, 31, true), "fewer than 32 rows");
    assert!(!ok(true, true, true, 128, false), "no scratch region");
}

#[test]
fn dsa_scores_tc3_grid_and_region() {
    assert_eq!(scores_tc3_grid(1, 1), [1, 1, 1]);
    assert_eq!(scores_tc3_grid(32, 256), [1, 1, 1]);
    assert_eq!(scores_tc3_grid(33, 257), [2, 2, 1]);
    assert_eq!(scores_tc3_grid(128, 65_536), [256, 4, 1]);
    assert_eq!(SCORES_TC3_POOLS, 8 * 32, "32 pools (four n-tiles) per warp");
    assert_eq!(SCORES_TC3_ROWS, SCORES_TC2_ROWS);
    // 2026-10-10: tc3 stages q exactly as tc2 does, so it shares tc2's shared-memory size.
    assert_eq!(scores_tc2_smem(128, 32), 21_632);
    assert_eq!(q_to_bf16_blocks(1), 1);
    assert_eq!(q_to_bf16_blocks(128 * 32 * 128 / 4), 512);
    assert_eq!(q_to_bf16_blocks(1 << 30), 4096);
    assert_eq!(q_to_bf16_blocks(257), 2);
}

/// 2026-10-10: GLM-5.3 DSA geometry (`select/tests.rs`).
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

/// 2026-10-10: 128 rows x 32 heads x 128 dims x 2 B = 1 MiB; 0 with the lever off or under
/// 32 rows.
#[test]
fn dsa_scores_tc3_qbf_region_bytes() {
    let cfg = cfg();
    let g = |rows| DsaSelectGeometry::plan(&cfg, 16_384, rows).unwrap();
    assert_eq!(qbf_region_bytes(&g(128), true), 128 * 32 * 128 * 2);
    assert_eq!(qbf_region_bytes(&g(128), false), 0);
    assert_eq!(qbf_region_bytes(&g(31), true), 0);
    assert_eq!(qbf_region_bytes(&g(32), true), 32 * 32 * 128 * 2);
}

/// 2026-10-10: `dsa_indexer.cu` defines the tc3 entry points with tc2's parameter list (q the
/// BF16 buffer), and its defines mirror the Rust constants.
#[test]
fn dsa_tc3_entry_points_and_tile_match_the_kernel_file() {
    let cu = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(format!("{DSA_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).expect("dsa_indexer.cu readable");
    let params = |head: &str| -> String {
        let Some(at) = src.find(head) else {
            panic!("{cu:?} lacks `{head}`");
        };
        let start = at + head.len();
        let len = src[start..].find(')').expect("parameter list closes");
        src[start..start + len]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let tc2 = params(
        "extern \"C\" __global__ void __launch_bounds__(DSA_TC2_THREADS, 2) \
         dsa_index_scores_tc2(",
    );
    let tc3 = params(
        "extern \"C\" __global__ void __launch_bounds__(DSA_TC3_THREADS, DSA_TC3_MINB) \
         dsa_index_scores_tc3(",
    );
    assert_eq!(
        tc3,
        tc2.replacen(
            "const float* __restrict__ q,",
            "const __nv_bfloat16* __restrict__ qb,",
            1
        ),
        "the launcher passes tc2's arguments with q the BF16 buffer"
    );
    assert!(
        src.contains("extern \"C\" __global__ void dsa_q_to_bf16("),
        "dsa_q_to_bf16 entry point"
    );
    for (define, v) in [
        ("DSA_TC3_THREADS", format!("{SCORES_TC3_BLOCK}u")),
        ("DSA_TC3_NJ", (SCORES_TC3_POOLS / 64).to_string()),
        ("DSA_TC2_ROWS", format!("{SCORES_TC3_ROWS}u")),
    ] {
        let line = format!("#define {define} {v}");
        assert!(
            src.lines().any(|l| l.trim() == line),
            "{cu:?}: `{line}` does not match the Rust constant"
        );
    }
}
