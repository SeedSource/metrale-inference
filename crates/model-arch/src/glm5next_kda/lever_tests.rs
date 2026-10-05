// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests of the KDA prefetch and chunked-TC levers' parsing and their geometry
//! gates: no GPU.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn prefetch_lever_is_on_only_for_one() {
    assert!(prefetch_requested(Some("1")));
    for v in [None, Some(""), Some("0"), Some("true"), Some("on"), Some(" 1"), Some("2")] {
        assert!(!prefetch_requested(v), "{v:?}");
    }
}

/// 2026-10-01: The GLM-5.3-Flash geometry (head_dim 128, `KDA_V_PER_BLOCK` 32) fits, with the
/// request the kernel comment gives (under the ~32 KB that keeps all 128 blocks resident).
#[test]
fn glm_geometry_fits_the_prefetch_kernel() {
    assert_eq!(kda_pf_smem(128, KDA_V_PER_BLOCK), Some(19_584));
}

#[test]
fn prefetch_geometry_outside_the_contract_is_refused() {
    // 2026-10-01: More staged elements per thread than KDA_PF_E_MAX.
    assert_eq!(kda_pf_smem(288, 32), None);
    // 2026-10-01: VPB does not divide D.
    assert_eq!(kda_pf_smem(100, 32), None);
    // 2026-10-01: More threads than __launch_bounds__(32).
    assert_eq!(kda_pf_smem(128, 64), None);
    assert_eq!(kda_pf_smem(128, 0), None);
    assert_eq!(kda_pf_smem(64, 4), None);
}

/// 2026-10-01: The largest geometry the contract admits (8 elements per thread at 32 threads)
/// stays under `KDA_SMEM_BUDGET` (`Some`), and a smaller VPB that still fits is accepted.
#[test]
fn prefetch_geometry_inside_the_contract_is_accepted() {
    assert_eq!(kda_pf_smem(256, 32), Some(39_040));
    assert_eq!(kda_pf_smem(64, 8), Some((6 * 64 + 8 * 65) * 4));
}

#[test]
fn chunked_tc_lever_is_on_only_for_one() {
    assert!(chunked_tc_requested(Some("1")));
    for v in [None, Some(""), Some("0"), Some("true"), Some("on"), Some(" 1"), Some("2")] {
        assert!(!chunked_tc_requested(v), "{v:?}");
    }
}

/// 2026-10-01: The GLM-5.3-Flash KDA geometry at the TP=2 per-rank head count, with the loader's
/// prefill chunk of 32.
fn glm_cfg() -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden: 6144,
        heads: 32,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    }
}

/// 2026-10-01: The GLM geometry runs the tensor-core prefill at every row count the workspace
/// holds, including a tail chunk shorter than 16 rows.
#[test]
fn glm_geometry_takes_the_chunked_tc_prefill() {
    let cfg = glm_cfg();
    for (k, t_pad) in [(2, 32), (17, 32), (256, 256), (1000, 1024), (8192, 8192)] {
        assert_eq!(chunked_tc_refusal(&cfg, k, t_pad, true), None, "k={k}");
    }
}

#[test]
fn chunked_tc_prefill_outside_its_contract_is_refused() {
    let cfg = glm_cfg();
    assert!(chunked_tc_refusal(&cfg, 256, 256, false).is_some());
    assert!(chunked_tc_refusal(&cfg, 0, 256, true).is_some());
    // 2026-10-01: More rows than the borrowed buffers were sized for.
    assert!(chunked_tc_refusal(&cfg, 300, 256, true).is_some());
    // 2026-10-04: FULLWIDTH without CHUNK_PREFILL used to size the chunk buffers to verify_k (padded to one
    // chunk) while the caller passed t_pad; an 8192-row prefill must be refused against those buffers.
    assert!(chunked_tc_refusal(&cfg, 8192, cfg.chunk, true).is_some());
    let mut c = cfg;
    c.head_dim = 64;
    assert!(chunked_tc_refusal(&c, 256, 256, true).is_some());
    let mut c = cfg;
    c.gate_lower_bound = -10.0;
    assert!(chunked_tc_refusal(&c, 256, 256, true).is_some());
    let mut c = cfg;
    c.gate_lower_bound = 0.5;
    assert!(chunked_tc_refusal(&c, 256, 256, true).is_some());
}

#[test]
fn flashkda_lever_is_on_only_for_one() {
    assert!(flashkda_requested(Some("1")));
    for v in [None, Some(""), Some("0"), Some("true"), Some("on"), Some(" 1"), Some("2")] {
        assert!(!flashkda_requested(v), "{v:?}");
    }
}

/// 2026-10-03: The GLM geometry with the library, the glue kernels and the scratch takes the
/// FlashKDA prefill; each missing piece is refused with a reason.
#[test]
fn flashkda_refusals() {
    use prefill_flashkda::flashkda_refusal;
    let cfg = glm_cfg();
    assert_eq!(flashkda_refusal(&cfg, true, true, true), None);
    assert!(flashkda_refusal(&cfg, false, true, true).is_some());
    assert!(flashkda_refusal(&cfg, true, false, true).is_some());
    assert!(flashkda_refusal(&cfg, true, true, false).is_some());
    let mut c = cfg;
    c.head_dim = 64;
    assert!(flashkda_refusal(&c, true, true, true).is_some());
    let mut c = cfg;
    c.gate_lower_bound = -5.5;
    assert!(flashkda_refusal(&c, true, true, true).is_some());
    let mut c = cfg;
    c.gate_lower_bound = 0.5;
    assert!(flashkda_refusal(&c, true, true, true).is_some());
    let mut c = cfg;
    c.conv_kernel = 9;
    assert!(flashkda_refusal(&c, true, true, true).is_some());
}

/// 2026-10-03: Pieces cover `0..k` in order without gaps, each at most the cap, all but the
/// last a multiple of 16, and the count is the fewest the cap allows.
#[test]
fn flashkda_pieces_cover_the_call() {
    use prefill_flashkda::{FLASHKDA_PIECE_ROWS, flashkda_pieces};
    assert!(flashkda_pieces(0, 4096).is_empty());
    assert_eq!(flashkda_pieces(256, 4096), vec![(0, 256)]);
    assert_eq!(flashkda_pieces(4096, 4096), vec![(0, 4096)]);
    assert_eq!(flashkda_pieces(4100, 4096), vec![(0, 2064), (2064, 2036)]);
    assert_eq!(flashkda_pieces(8192, 4096), vec![(0, 4096), (4096, 4096)]);
    for k in [1usize, 15, 17, 100, 255, 1000, 4097, 8191, 8193, 12_289, 16_384] {
        for cap in [16usize, 256, FLASHKDA_PIECE_ROWS] {
            let p = flashkda_pieces(k, cap);
            assert_eq!(p.len(), k.div_ceil(cap), "k={k} cap={cap}");
            let mut next = 0;
            for (i, &(s, rows)) in p.iter().enumerate() {
                assert_eq!(s, next, "k={k} cap={cap}");
                assert!(rows >= 1 && rows <= cap, "k={k} cap={cap}");
                if i + 1 < p.len() {
                    assert!(rows.is_multiple_of(16), "k={k} cap={cap}");
                }
                next += rows;
            }
            assert_eq!(next, k, "k={k} cap={cap}");
        }
    }
}

/// 2026-10-03: The scratch the loader logs: 9.2 MB at the staged prefill's 256-row KDA calls,
/// ~115 MB at 4,096 rows (the cap), for GLM-5.3 TP2 (32 heads).
#[test]
fn flashkda_scratch_bytes() {
    let cfg = glm_cfg();
    let state = 32 * 128 * 128 * 4;
    assert_eq!(
        Glm5NextKdaWorkspace::bytes_flashkda(&cfg, 256),
        32 * 16 * 13_824 + 128 + state
    );
    assert_eq!(
        Glm5NextKdaWorkspace::bytes_flashkda(&cfg, 4096),
        Glm5NextKdaWorkspace::bytes_flashkda(&cfg, 8192)
    );
    assert_eq!(
        Glm5NextKdaWorkspace::bytes_flashkda(&cfg, 100),
        32 * 7 * 13_824 + 128 + state
    );
}
