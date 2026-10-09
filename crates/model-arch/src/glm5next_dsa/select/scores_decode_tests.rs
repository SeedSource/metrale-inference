// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Host tests of `METRALE_GLM_DSA_SCORES_DECODE`: the parse, the dispatch envelope,
//! the grid and shared-memory sizing, and the `.cu` entry point and defines.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::DSA_MODULE;

#[test]
fn dsa_scores_decode_parse() {
    assert!(parse_scores_decode(Some("1")));
    assert!(parse_scores_decode(Some(" 1 ")));
    for v in [
        None,
        Some(""),
        Some("0"),
        Some("on"),
        Some("true"),
        Some("2"),
    ] {
        assert!(!parse_scores_decode(v), "{v:?}");
    }
}

/// 2026-10-08: Engaged only on a ceiling launch at D 128 with a bounded head count and aligned
/// keys.
#[test]
fn dsa_scores_decode_selection_envelope() {
    let ok = scores_decode_for;
    assert!(ok(true, true, true, 128, 32, true));
    assert!(ok(true, true, true, 128, 1, true));
    assert!(ok(true, true, true, 128, SCORES_DECODE_MAX_H, true));
    assert!(!ok(false, true, true, 128, 32, true), "lever off");
    assert!(!ok(true, false, true, 128, 32, true), "entry point absent");
    assert!(!ok(true, true, false, 128, 32, true), "exact launch");
    for d in [0, 64, 96, 127, 129, 256] {
        assert!(!ok(true, true, true, d, 32, true), "d {d}");
    }
    assert!(!ok(true, true, true, 128, 0, true), "no heads");
    assert!(
        !ok(true, true, true, 128, SCORES_DECODE_MAX_H + 1, true),
        "too many heads"
    );
    assert!(!ok(true, true, true, 128, 32, false), "unaligned keys");
}

#[test]
fn dsa_scores_decode_grid_and_smem() {
    assert_eq!(scores_decode_block_pools(), 128);
    assert_eq!(SCORES_DECODE_BLOCK as usize, SCORES_DECODE_WARPS * 32);
    // 2026-10-08: GB10 (48 SMs) at the 512K ceiling: 96 blocks; a tiny ceiling: one per step.
    assert_eq!(scores_decode_grid_x(135_168, 48), 96);
    assert_eq!(scores_decode_grid_x(129, 48), 2);
    assert_eq!(scores_decode_grid_x(1, 48), 1);
    assert_eq!(scores_decode_grid_x(0, 48), 1, "never a zero-extent grid");
    assert_eq!(
        scores_decode_grid_x(135_168, 0),
        SCORES_DECODE_BLOCKS_PER_SM
    );
    // 2026-10-08: Independent of the live context by construction: only the ceiling enters.
    assert_eq!(scores_decode_smem(32), 33_408);
    assert!(scores_decode_smem(SCORES_DECODE_MAX_H) <= 49_152);
}

/// 2026-10-08: `dsa_indexer.cu` defines `dsa_index_scores_decode` with `dsa_index_scores`'
/// parameter list (the launcher passes the same arguments), and its defines mirror the Rust
/// constants; a rename or a resize in the `.cu` alone fails here instead of mis-launching.
#[test]
fn dsa_scores_decode_entry_point_matches_the_kernel_file() {
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
    let plain = params("extern \"C\" __global__ void dsa_index_scores(");
    let dec = params(
        "extern \"C\" __global__ void __launch_bounds__(DSA_DEC_THREADS, 2) \
         dsa_index_scores_decode(",
    );
    assert_eq!(
        dec, plain,
        "the launcher passes dsa_index_scores' arguments"
    );
    for (define, v) in [
        ("DSA_DEC_THREADS", SCORES_DECODE_BLOCK as usize),
        ("DSA_DEC_D", SCORES_DECODE_D),
        ("DSA_DEC_LD", SCORES_DECODE_LD),
    ] {
        let line = format!("#define {define} {v}u");
        assert!(
            src.lines().any(|l| l.trim() == line),
            "{cu:?}: `{line}` does not match the Rust constant"
        );
    }
    assert!(
        src.lines()
            .any(|l| l.trim() == "#define DSA_DEC_WARPS (DSA_DEC_THREADS / 32u)"),
        "DSA_DEC_WARPS is the block's warp count"
    );
}

#[test]
fn the_lever_is_declared_default_off_and_read_here() {
    use metrale_config::levers::{Class, Ty, lookup};
    let l = lookup("METRALE_GLM_DSA_SCORES_DECODE").expect("declared in the lever table");
    assert_eq!(l.default, "off");
    assert_eq!((l.ty, l.class), (Ty::Switch, Class::Runtime));
    assert_eq!(
        l.reader,
        "crates/model-arch/src/glm5next_dsa/select/scores_decode.rs"
    );
    assert!(include_str!("scores_decode.rs").contains(l.env));
    // 2026-10-08: The table text names the head cap, so it is changed with the constant.
    assert!(
        l.doc
            .contains(&format!("more than {SCORES_DECODE_MAX_H} index heads")),
        "{}",
        l.doc
    );
}
