// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Host tests of `METRALE_GLM_DSA_SCORES_TC2`: the parse, the dispatch envelope, the
//! grid and shared-memory sizing, and the `.cu` entry point and defines.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::DSA_MODULE;

#[test]
fn dsa_scores_tc2_parse() {
    assert!(parse_scores_tc2(Some("1")));
    assert!(parse_scores_tc2(Some(" 1 ")));
    for v in [
        None,
        Some(""),
        Some("0"),
        Some("on"),
        Some("true"),
        Some("2"),
    ] {
        assert!(!parse_scores_tc2(v), "{v:?}");
    }
}

/// 2026-10-07: Engaged only on top of a `dsa_index_scores_tc` launch in mode 1.
#[test]
fn dsa_scores_tc2_selection_envelope() {
    let ok = scores_tc2_for;
    assert!(ok(true, true, true, 1, 32));
    assert!(ok(true, true, true, 1, SCORES_TC2_MAX_H));
    assert!(!ok(false, true, true, 1, 32), "lever off");
    assert!(!ok(true, false, true, 1, 32), "entry point absent");
    assert!(
        !ok(true, true, false, 1, 32),
        "tensor-core path not selected"
    );
    for mode in [0, 2, 3] {
        assert!(!ok(true, true, true, mode, 32), "mode {mode}");
    }
    assert!(
        !ok(true, true, true, 1, SCORES_TC2_MAX_H + 1),
        "too many heads"
    );
}

#[test]
fn dsa_scores_tc2_grid_and_smem() {
    assert_eq!(scores_tc2_grid(1, 1), [1, 1, 1]);
    assert_eq!(scores_tc2_grid(32, 128), [1, 1, 1]);
    assert_eq!(scores_tc2_grid(33, 129), [2, 2, 1]);
    assert_eq!(scores_tc2_grid(128, 32_768), [256, 4, 1]);
    assert_eq!(SCORES_TC2_BLOCK as usize, 8 * 32, "eight warps");
    assert_eq!(SCORES_TC2_POOLS, 8 * 16, "16 pools (two n-tiles) per warp");
    assert_eq!(scores_tc2_smem(128, 32), 21_504);
    // 2026-10-07: Within the default 48 KiB at the envelope's corner, so no opt-in is needed.
    assert!(scores_tc2_smem(128, SCORES_TC2_MAX_H) <= 49_152);
}

/// 2026-10-07: `dsa_indexer.cu` defines `dsa_index_scores_tc2` with `dsa_index_scores_tc`'s
/// parameter list (the launcher passes the same arguments), and its defines mirror the Rust
/// constants; a rename or a resize in the `.cu` alone fails here instead of mis-launching.
#[test]
fn dsa_tc2_entry_point_and_tile_match_the_kernel_file() {
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
    let tc = params(
        "extern \"C\" __global__ void __launch_bounds__(DSA_TC_THREADS) \
         dsa_index_scores_tc(",
    );
    let tc2 = params(
        "extern \"C\" __global__ void __launch_bounds__(DSA_TC2_THREADS, 2) \
         dsa_index_scores_tc2(",
    );
    assert_eq!(
        tc2, tc,
        "the launcher passes dsa_index_scores_tc's arguments"
    );
    for (define, v) in [
        ("DSA_TC2_ROWS", SCORES_TC2_ROWS),
        ("DSA_TC2_POOLS", SCORES_TC2_POOLS),
        ("DSA_TC2_THREADS", SCORES_TC2_BLOCK as usize),
        ("DSA_TC2_WARPS", SCORES_TC2_POOLS / 16),
    ] {
        let line = format!("#define {define} {v}u");
        assert!(
            src.lines().any(|l| l.trim() == line),
            "{cu:?}: `{line}` does not match the Rust constant"
        );
    }
}
