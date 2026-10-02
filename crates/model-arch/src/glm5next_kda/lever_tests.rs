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
