// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Tests of the KDA prefetch lever's parsing and its geometry gate: no GPU.
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
