// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Host tests of `METRALE_GLM_DSA_EXPAND_TAIL_COUNT`: the parse, the dispatch
//! envelope, the kernel file's entry points and chunk define, a host model of the chunked
//! count against the original byte scan, and the lever declaration.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::DSA_MODULE;

#[test]
fn expand_tail_count_parse() {
    assert!(parse_expand_tail_count(Some("1")));
    assert!(parse_expand_tail_count(Some(" 1 ")));
    for v in [
        None,
        Some(""),
        Some("0"),
        Some("on"),
        Some("true"),
        Some("2"),
    ] {
        assert!(!parse_expand_tail_count(v), "{v:?}");
    }
}

#[test]
fn expand_tail_count_selection_envelope() {
    let ok = expand_tail_count_for;
    let need = chunk_count(65_536) * 4;
    assert!(ok(true, true, true, true, 65_536, need));
    assert!(!ok(false, true, true, true, 65_536, need), "lever off");
    assert!(
        !ok(true, false, true, true, 65_536, need),
        "entry point absent"
    );
    assert!(
        !ok(true, true, false, true, 65_536, need),
        "ceiling or device geometry"
    );
    assert!(
        !ok(true, true, true, false, 65_536, need),
        "no always-tail count"
    );
    assert!(!ok(true, true, true, true, 0, need), "no keys");
    assert!(
        !ok(true, true, true, true, 65_536, need - 4),
        "scores region too small"
    );
    assert_eq!(chunk_count(1), 1);
    assert_eq!(chunk_count(1024), 1);
    assert_eq!(chunk_count(1025), 2);
    assert_eq!(chunk_count(262_144), 256);
}

/// 2026-10-10: The kernel file defines the chunk size the launcher uses, the cc entry point
/// takes the plain entry point's parameter list plus `chunk_cnt`, and the chunk-count kernel
/// has the argument list `launch_chunk_counts` passes.
#[test]
fn expand_tail_count_entry_points_match_the_kernel_file() {
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
    let plain = params("extern \"C\" __global__ void dsa_expand_selection(");
    let cc = params("extern \"C\" __global__ void dsa_expand_selection_cc(");
    assert_eq!(
        cc,
        format!("{plain}, const int* __restrict__ chunk_cnt"),
        "the cc entry point is the plain one plus chunk_cnt"
    );
    let counts = params("extern \"C\" __global__ void dsa_valid_chunk_counts(");
    assert_eq!(
        counts,
        "const unsigned char* __restrict__ valid_keys, int* __restrict__ chunk_cnt, \
         unsigned int S"
    );
    let line = format!("#define DSA_CC_CHUNK {EXPAND_TAIL_COUNT_CHUNK}");
    assert!(src.lines().any(|l| l.trim() == line), "{cu:?}: `{line}`");
    assert_eq!(
        CHUNK_COUNTS_BLOCK, 256,
        "warp_sum[8] in dsa_valid_chunk_counts"
    );
}

/// 2026-10-10: A host model of the device arithmetic: the original per-row byte scan against
/// the chunk-count form, over masks with holes, `q_pos` below 0, at 0, at chunk edges, at
/// `S - 1` and beyond `S`, and an `S` that is not a chunk multiple.
#[test]
fn chunked_visible_count_equals_the_byte_scan() {
    let scan = |valid: &[u8], s: usize, qp: i32| -> i32 {
        (0..s)
            .filter(|&t| (t as i64) <= qp as i64 && valid[t] != 0)
            .count() as i32
    };
    let chunked = |valid: &[u8], counts: &[i32], s: usize, qp: i32| -> i32 {
        let lim: usize = if qp < 0 {
            0
        } else {
            ((qp as u64 + 1).min(s as u64)) as usize
        };
        let nfull = lim / EXPAND_TAIL_COUNT_CHUNK;
        let mut n: i32 = counts[..nfull].iter().sum();
        for t in nfull * EXPAND_TAIL_COUNT_CHUNK..lim {
            n += (valid[t] != 0) as i32;
        }
        n
    };
    for s in [1usize, 1023, 1024, 1025, 3000, 4096, 5000] {
        for pattern in 0..4u32 {
            let valid: Vec<u8> = (0..s)
                .map(|t| match pattern {
                    0 => 1,
                    1 => (t % 7 != 3) as u8,
                    2 => (t >= 1500 && t % 2 == 0) as u8,
                    _ => {
                        if t % 5 == 0 {
                            2
                        } else {
                            0
                        }
                    }
                })
                .collect();
            let counts: Vec<i32> = (0..chunk_count(s))
                .map(|c| {
                    (c * EXPAND_TAIL_COUNT_CHUNK..((c + 1) * EXPAND_TAIL_COUNT_CHUNK).min(s))
                        .filter(|&t| valid[t] != 0)
                        .count() as i32
                })
                .collect();
            let mut qps = vec![
                -5,
                -1,
                0,
                1,
                1022,
                1023,
                1024,
                1025,
                2047,
                2048,
                s as i32 - 1,
            ];
            qps.extend([s as i32, s as i32 + 7, i32::MAX]);
            for qp in qps {
                assert_eq!(
                    chunked(&valid, &counts, s, qp),
                    scan(&valid, s, qp),
                    "s {s} qp {qp} p {pattern}"
                );
            }
        }
    }
}

#[test]
fn the_lever_is_declared_default_off_and_read_here() {
    use metrale_config::levers::{Class, Ty, lookup};
    let l = lookup("METRALE_GLM_DSA_EXPAND_TAIL_COUNT").expect("declared in the lever table");
    assert_eq!(l.default, "off");
    assert_eq!((l.ty, l.class), (Ty::Switch, Class::Runtime));
    assert_eq!(
        l.reader,
        "crates/model-arch/src/glm5next_dsa/select/tail_count.rs"
    );
    assert!(include_str!("tail_count.rs").contains(l.env));
    assert!(include_str!("tail_count.rs").contains("METRALE_GLM_DSA_EXPAND_TAIL_COUNT ENGAGED"));
}
