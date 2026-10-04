// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: CPU tests of the `dense_gemv_tcm` routing rules and the kernel source's
//! entry list.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn lever_value_rule() {
    assert!(gemv_tc_from(Some("1")));
    for v in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some("on"),
        Some(" 1"),
    ] {
        assert!(!gemv_tc_from(v), "{v:?}");
    }
}

#[test]
fn shape_gate() {
    for m in 1..=TCM_MAX_M {
        assert!(routes(m, 4096, 4096, false));
        assert!(routes(m, 4096, 4096, true));
    }
    assert!(!routes(0, 4096, 4096, false));
    assert!(!routes(TCM_MAX_M + 1, 4096, 4096, false));
    assert!(!routes(4, 0, 4096, false));
    assert!(!routes(4, 4096, 0, false));
    // BF16 needs k % 64, FP8 k % 128.
    assert!(routes(4, 4096, 64, false));
    assert!(!routes(4, 4096, 64, true));
    assert!(routes(4, 4096, 128, true));
    assert!(!routes(4, 4096, 96, false));
    // Any N routes (rows past N load zeros and are never stored).
    assert!(routes(4, 37, 128, true));
}

#[test]
fn variant_ignores_m_and_matches_kind() {
    // The rule takes no M; this pins the kind for the GLM-5.3 shapes and the LM head.
    for (n, k) in [
        (4096, 4096),
        (128, 4096),
        (4096, 128),
        (1536, 4096),
        (16384, 1536),
        (512, 4096),
        (4096, 16384),
        (6144, 4096),
        (4096, 6144),
        (1024, 4096),
        (4096, 1024),
        (77440, 4096),
    ] {
        assert!(!variant_for(n, k, false).is_fp8());
        assert!(variant_for(n, k, true).is_fp8());
    }
}

#[test]
fn entries_exist_in_kernel_source() {
    let src = include_str!("../../../../../kernels/gb10/common/dense_gemv_tcm.cu");
    for v in TcmVariant::ALL {
        assert!(
            src.contains(&format!("TCM_ENTRY({}, ", v.entry())),
            "{} missing from dense_gemv_tcm.cu",
            v.entry()
        );
        let nt = if v.rows_per_cta() == 32 { 2 } else { 1 };
        let fp8 = if v.is_fp8() { "true" } else { "false" };
        assert!(
            src.contains(&format!("TCM_ENTRY({}, {fp8}, {nt}, ", v.entry())),
            "{}: kind/NT disagree with the kernel source",
            v.entry()
        );
    }
    assert!(
        src.contains("#define TCM_NB 2"),
        "TCM_MAX_M = 16 needs TCM_NB 2"
    );
}
