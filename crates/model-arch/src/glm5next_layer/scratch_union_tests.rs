// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Tests for `scratch_union`: the bump allocator, measuring vs placing passes of
//! the three member constructors, and the lever-off constructors' `gpu.alloc` calls.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::Glm5NextDsaConfig;
use crate::glm5next_dsa::layer::DsaWideArena;
use crate::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaWorkspace};
use crate::glm5next_mlp::Glm5NextMlpConfig;
use crate::glm5next_mlp::forward::{Glm5NextMlpWorkspace, mlp_ws_bytes_sized};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-10-05: GLM-5.3 TP2 KDA geometry (as `kda_split_workspace_saves_only_the_chunk_buffers`).
fn kda_cfg() -> Glm5NextKdaConfig {
    Glm5NextKdaConfig {
        hidden: 4096,
        heads: 32,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-5,
        l2_eps: 1e-6,
        chunk: 32,
    }
}

/// 2026-10-05: GLM-5.3 MLP geometry at TP=2, EP=2 (as `forward::tests::ws_sizing::cfg`).
fn mlp_cfg() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: 4096,
        local_dense_intermediate: 12288 / 2,
        moe_intermediate: 2048,
        local_shared_intermediate: 2048 / 2,
        num_experts: 288,
        local_experts: 144,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 2,
        ep_world_size: 2,
    }
}

/// 2026-10-05: GLM-5.3 DSA geometry at TP2 (32 local heads).
fn dsa_cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 32,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

/// 2026-10-05: The sizes a constructor asks for, in call order.
fn record<T>(build: impl FnOnce(&mut ScratchAlloc) -> Result<T>) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut b = Bump::new(DevicePtr(0));
    build(&mut |n| {
        sizes.push(n);
        b.alloc(n)
    })
    .expect("build");
    sizes
}

#[test]
fn bump_aligns_every_sub_allocation_and_tracks_the_high_water_mark() {
    let mut b = Bump::new(DevicePtr(0x1000));
    let p: Vec<u64> = [1usize, 300, 0, 513, 256]
        .iter()
        .map(|&n| b.alloc(n).unwrap().0)
        .collect();
    assert_eq!(
        p,
        vec![
            0x1000,
            0x1000 + 256,
            0x1000 + 768,
            0x1000 + 768,
            0x1000 + 1536
        ]
    );
    assert!(p.iter().all(|x| (x - 0x1000) % ALIGN as u64 == 0));
    assert_eq!(b.used(), 1536 + 256);
}

#[test]
fn the_lever_parses_only_one() {
    assert!(parse_switch(Some("1")));
    assert!(parse_switch(Some(" 1 ")));
    for v in [None, Some(""), Some("0"), Some("true"), Some("on")] {
        assert!(!parse_switch(v), "{v:?}");
    }
}

/// 2026-10-05: Each member's measuring pass and placing pass (from a real base) agree, and the
/// placed members all start at the union base.
#[test]
fn measuring_and_placing_passes_agree_for_every_member() {
    let gpu = MockGpuBackend::new();
    let (k, m, d) = (kda_cfg(), mlp_cfg(), dsa_cfg());
    for (rows, chunk) in [(16usize, 16usize), (40, 16), (64, 64)] {
        let want_max = [
            measure(|a| Glm5NextKdaWorkspace::new_split_in(&k, rows, chunk, a)).unwrap(),
            measure(|a| Glm5NextMlpWorkspace::new_sized_in(&m, rows, 16, a)).unwrap(),
            measure(|a| DsaWideArena::new_in(&d, rows, a)).unwrap(),
        ]
        .into_iter()
        .max()
        .unwrap();
        let before = gpu.alloc_count();
        let u = plan_glm(
            &gpu,
            (&k, rows, chunk),
            Some((&m, rows, 16)),
            Some((&d, rows)),
        )
        .unwrap()
        .expect("three members");
        assert_eq!(gpu.alloc_count(), before + 1, "one real allocation");
        assert_eq!(gpu.read_alloc(u.base()).unwrap().len(), want_max);
        let kws = u
            .place(KDA, |a| {
                Glm5NextKdaWorkspace::new_split_in(&k, rows, chunk, a)
            })
            .unwrap();
        assert_eq!(kws.qkv_parts, u.base(), "KDA starts at the base");
        u.place(MLP, |a| Glm5NextMlpWorkspace::new_sized_in(&m, rows, 16, a))
            .unwrap();
        u.place(DSA_WIDE, |a| DsaWideArena::new_in(&d, rows, a))
            .unwrap();
        assert!(u.place("nope", |a| a(1)).is_err());
    }
}

/// 2026-10-05: A placing pass that asks for more than was measured is refused.
#[test]
fn a_placing_pass_that_disagrees_with_its_measure_is_refused() {
    let gpu = MockGpuBackend::new();
    let u = ScratchUnion::alloc(&gpu, vec![("a", 512), ("b", 100)])
        .unwrap()
        .unwrap();
    assert!(u.place("a", |a| a(512)).is_ok());
    assert!(u.place("a", |a| a(513)).is_err());
    assert!(u.place("b", |a| a(50)).is_err());
}

/// 2026-10-05: Fewer than two members (per-layer MLP, no wide arena): no union, nothing
/// allocated.
#[test]
fn one_member_keeps_separate_allocations() {
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    assert!(
        plan_glm(&gpu, (&kda_cfg(), 16, 16), None, None)
            .unwrap()
            .is_none()
    );
    assert!(
        ScratchUnion::alloc(&gpu, vec![(KDA, 4096)])
            .unwrap()
            .is_none()
    );
    assert_eq!(gpu.alloc_count(), before);
}

/// 2026-10-05: Lever off, the plain constructors make the `gpu.alloc` calls they made before:
/// the sizes match the existing byte formulas (MLP: in `mlp_ws_bytes_sized` order), the call
/// count matches, and the mock (which packs allocations at 256 B like `Bump`) spans exactly
/// the measured union bytes.
#[test]
fn lever_off_constructors_make_the_same_alloc_calls() {
    let (k, m, d) = (kda_cfg(), mlp_cfg(), dsa_cfg());
    for (rows, chunk) in [(16usize, 16usize), (40, 16)] {
        let kda = record(|a| Glm5NextKdaWorkspace::new_split_in(&k, rows, chunk, a));
        assert_eq!(kda.len(), 18);
        assert_eq!(
            kda.iter().sum::<usize>(),
            Glm5NextKdaWorkspace::bytes_split(&k, rows, chunk)
        );
        // 2026-10-05: `METRALE_GLM_MOE_PREFILL_PERMUTE` is unset in tests: no `moe_perm`.
        let mlp = record(|a| Glm5NextMlpWorkspace::new_sized_in(&m, rows, 16, a));
        assert_eq!(mlp, mlp_ws_bytes_sized(&m, rows, 16).to_vec());
        let dsa = record(|a| DsaWideArena::new_in(&d, rows, a));
        assert_eq!(dsa.len(), 8);
        assert_eq!(
            dsa.iter().sum::<usize>(),
            DsaWideArena::bytes_per_row(&d) * rows
        );

        let span = |n: usize, build: &dyn Fn(&MockGpuBackend)| {
            let gpu = MockGpuBackend::new();
            let p0 = gpu.alloc(1).unwrap().0;
            let c0 = gpu.alloc_count();
            build(&gpu);
            assert_eq!(gpu.alloc_count() - c0, n, "alloc calls");
            (gpu.alloc(1).unwrap().0 - p0 - ALIGN as u64) as usize
        };
        let up = |b: usize| b.div_ceil(ALIGN) * ALIGN;
        let km = measure(|a| Glm5NextKdaWorkspace::new_split_in(&k, rows, chunk, a)).unwrap();
        let s = span(kda.len(), &|g| {
            Glm5NextKdaWorkspace::new_split(g, &k, rows, chunk).unwrap();
        });
        assert_eq!(s, up(km));
        let mm = measure(|a| Glm5NextMlpWorkspace::new_sized_in(&m, rows, 16, a)).unwrap();
        let s = span(mlp.len(), &|g| {
            Glm5NextMlpWorkspace::new_sized(g, &m, rows, 16).unwrap();
        });
        assert_eq!(s, up(mm));
        let dm = measure(|a| DsaWideArena::new_in(&d, rows, a)).unwrap();
        let s = span(dsa.len(), &|g| {
            DsaWideArena::new(g, &d, rows).unwrap();
        });
        assert_eq!(s, up(dm));
    }
}
