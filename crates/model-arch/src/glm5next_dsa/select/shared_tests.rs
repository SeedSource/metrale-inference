// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Host tests of `METRALE_GLM_DSA_SELECT_SCRATCH_SHARED`: the shared scratch's
//! sizing (per-region maximum over the layer and MTP geometries), the saved-bytes formula
//! against the 2026-10-05 alloc ledger, and that the workspace keeps its own allocation when
//! no shared scratch is passed.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::layer::Glm5NextDsaWorkspace;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-10-05: GLM-5.3 DSA geometry (`select/tests.rs`) at `max_context` tokens.
fn cfg(max_context: usize) -> Glm5NextDsaConfig {
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
        max_context,
    }
}

/// 2026-10-05: `plan_bytes` of one geometry is what `alloc` always reserved; of several, the
/// per-region maximum (here regions come from different geometries: a longer context with
/// fewer rows against a shorter one with more rows).
#[test]
fn plan_bytes_is_the_per_region_max() {
    let c = cfg(16_384);
    let long = DsaSelectGeometry::plan(&c, 16_384, 1).unwrap();
    let wide = DsaSelectGeometry::plan(&c, 4_096, 64).unwrap();
    let (one, t1) = DsaSelectScratch::plan_bytes(&c, &[long]);
    // 2026-10-06: Regions 0 to 5; region 6 (radix top-k) follows the lever, off here.
    assert_eq!(&one[..6], &long.scratch_bytes()[..]);
    assert_eq!(t1, c.out_width() * 4);
    let (both, tb) = DsaSelectScratch::plan_bytes(&c, &[long, wide]);
    for i in 0..6 {
        assert_eq!(
            both[i],
            long.scratch_bytes()[i].max(wide.scratch_bytes()[i]),
            "region {i}"
        );
    }
    // 2026-10-05: The pool regions follow the longer context, the per-row ones the wider pass.
    assert_eq!(both[0], long.scratch_bytes()[0]);
    assert!(wide.scratch_bytes()[3] > long.scratch_bytes()[3]);
    assert_eq!(both[3], wide.scratch_bytes()[3]);
    assert_eq!(tb, 64 * c.out_width() * 4);
    assert_eq!(
        DsaSelectScratch::plan_bytes(&c, &[wide, long]),
        (both, tb),
        "order-independent"
    );
}

/// 2026-10-05: The layer geometry dominates the 1-row MTP one, so the shared scratch is one
/// layer's; saved = 11 layers + MTP - 1 layer. Pinned against the 2026-10-05 alloc ledger
/// (GLM-5.3 TP2, 256-row sub-chunks): 3,859 MiB for the twelve at msl 786,432 and 347 MiB at
/// 65,536.
#[test]
fn saved_bytes_match_the_alloc_ledger() {
    for (msl, total, saved) in [
        (65_536usize, 363_654_156usize, 331_390_988usize),
        (557_056, 2_874_461_196, 2_619_908_108),
        (786_432, 4_046_171_148, 3_687_882_764),
    ] {
        let p = SharedPlan::new(&cfg(msl), 256).unwrap();
        assert_eq!(p.shared_bytes(), p.layer_bytes, "msl {msl}");
        assert_eq!(11 * p.layer_bytes + p.mtp_bytes, total, "msl {msl}");
        assert_eq!(p.saved_bytes(11, true), saved, "msl {msl}");
        assert_eq!(
            p.saved_bytes(11, false),
            10 * p.layer_bytes,
            "msl {msl} without MTP"
        );
    }
    let p = SharedPlan::new(&cfg(16_384), 16).unwrap();
    assert_eq!(p.saved_bytes(0, false), 0);
    assert_eq!(p.saved_bytes(1, false), 0);
}

/// 2026-10-05: Lever-off construction (`new`, and `new_with_select(None)`) allocates its own
/// select scratch in every workspace; a shared one is not allocated again, and one that does
/// not fit the workspace's largest pass is refused.
#[test]
fn workspace_keeps_its_own_scratch_unless_given_one() {
    let gpu = MockGpuBackend::new();
    let c = cfg(4_096);
    let rows = 16;
    let geom = DsaSelectGeometry::plan(&c, 4_096, rows).unwrap();
    let sel_bytes = DsaSelectScratch::planned_total(&DsaSelectScratch::plan_bytes(&c, &[geom]));

    let b0 = gpu.live_bytes().unwrap();
    let n0 = gpu.live_alloc_count();
    let _a = Glm5NextDsaWorkspace::new(&gpu, &c, rows).unwrap();
    let own = gpu.live_bytes().unwrap() - b0;
    let own_n = gpu.live_alloc_count() - n0;
    let b1 = gpu.live_bytes().unwrap();
    let _b = Glm5NextDsaWorkspace::new_with_select(&gpu, &c, rows, None).unwrap();
    assert_eq!(gpu.live_bytes().unwrap() - b1, own, "None is exactly `new`");

    let shared = SharedPlan::new(&c, rows).unwrap();
    let s = DsaSelectScratch::alloc_sized(&gpu, shared.plan).unwrap();
    assert_eq!(s.bytes(), sel_bytes);
    let (b2, n2) = (gpu.live_bytes().unwrap(), gpu.live_alloc_count());
    let _c = Glm5NextDsaWorkspace::new_with_select(&gpu, &c, rows, Some(s)).unwrap();
    assert_eq!(gpu.live_bytes().unwrap() - b2, own - sel_bytes);
    assert_eq!(
        gpu.live_alloc_count() - n2,
        own_n - 7,
        "the 7 scratch regions"
    );
    // 2026-10-05: The 1-row MTP workspace takes the same scratch.
    assert!(Glm5NextDsaWorkspace::new_with_select(&gpu, &c, MTP_ROWS, Some(s)).is_ok());

    // 2026-10-05: A scratch planned for fewer rows does not fit a wider workspace.
    let small = DsaSelectScratch::alloc_sized(&gpu, SharedPlan::new(&c, 1).unwrap().plan).unwrap();
    assert!(Glm5NextDsaWorkspace::new_with_select(&gpu, &c, rows, Some(small)).is_err());
}
