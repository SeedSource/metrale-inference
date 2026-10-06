// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Host tests of `METRALE_GLM_DSA_TOPK_RADIX`: the key transform, the planning and
//! sizing functions, the dispatch table, the launch sequence on a mock, and the `.cu` defines.
//! The host model of the kernels is in `radix_model_tests.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::DSA_MODULE;
use crate::glm5next_dsa::select::DsaSelectScratch;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

/// 2026-10-06: GLM-5.3 DSA geometry (`select/tests.rs`) at `max_context` tokens.
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

/// 2026-10-06: Finite floats in ascending order, with -0.0 and +0.0 adjacent.
const ASCENDING: &[f32] = &[
    f32::NEG_INFINITY,
    -f32::MAX,
    -1.0e30,
    -2.0,
    -1.0,
    -f32::MIN_POSITIVE,
    -1.0e-45,
    -0.0,
    0.0,
    1.0e-45,
    f32::MIN_POSITIVE,
    0.5,
    1.0,
    1.0e30,
    f32::MAX,
    f32::INFINITY,
];

/// 2026-10-06: The key preserves the float order, ties -0.0 with +0.0, and puts -FLT_MAX below
/// every other finite score (only -inf is lower).
#[test]
fn radix_key_orders_like_the_comparator() {
    for w in ASCENDING.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a == b {
            assert_eq!(radix_key(a), radix_key(b), "{a:e} == {b:e}");
        } else {
            assert!(radix_key(a) < radix_key(b), "{a:e} < {b:e}");
        }
    }
    assert_eq!(radix_key(-0.0), radix_key(0.0));
    assert_eq!(radix_key(0.0), 0x8000_0000);
    assert_eq!(radix_key(-f32::MAX), 0x0080_0000);
    assert!(radix_key(f32::NEG_INFINITY) < radix_key(-f32::MAX));
    // 2026-10-06: Every pair of a pseudo-random sample, both signs and many exponents.
    let mut x = 0x1234_5678_u32;
    let mut v = Vec::new();
    for _ in 0..400 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let f = f32::from_bits(x);
        if f.is_finite() {
            v.push(f);
        }
    }
    v.extend_from_slice(&[-f32::MAX, -0.0, 0.0]);
    for &a in &v {
        for &b in &v {
            assert_eq!(a > b, radix_key(a) > radix_key(b), "{a:e} vs {b:e}");
            assert_eq!(a == b, radix_key(a) == radix_key(b), "{a:e} vs {b:e}");
        }
    }
}

/// 2026-10-06: `radix_chunks` spreads `RADIX_TARGET_BLOCKS` over the rows, within 1..=32.
#[test]
fn radix_chunks_stay_within_the_tie_base_array() {
    assert_eq!(radix_chunks(0), RADIX_MAX_CHUNKS);
    assert_eq!(radix_chunks(1), 32);
    assert_eq!(radix_chunks(3), 32);
    assert_eq!(radix_chunks(4), 24);
    assert_eq!(radix_chunks(96), 1);
    assert_eq!(radix_chunks(4096), 1);
    for q in 1..300 {
        assert!(
            (1..=RADIX_MAX_CHUNKS).contains(&radix_chunks(q)),
            "q_rows {q}"
        );
    }
    // 2026-10-06: The pass-2 per-chunk histograms reuse the two 4096-bin histograms.
    const { assert!(RADIX_MAX_CHUNKS * RADIX_BINS_LAST <= RADIX_STATE) };
}

/// 2026-10-06: Work-buffer sizing at GLM-5.3's shape: kcap 512, 9,264 words a row.
#[test]
fn radix_work_buffer_sizes() {
    let c = cfg(540_672);
    assert_eq!(radix_kcap(&c), 512);
    assert_eq!(RADIX_CAND, 8_240);
    assert_eq!(radix_row_words(512), 9_264);
    assert_eq!(radix_row_bytes(512), 37_056);
    assert_eq!(radix_work_bytes(&c, 3), 3 * 37_056);
    // 2026-10-06: kcap never exceeds the sort kernel's shared arrays.
    let mut wide = c;
    wide.index_topk = 65_536;
    assert_eq!(radix_kcap(&wide), RADIX_SORT_MAX);
}

/// 2026-10-06: Region 6 is planned only with the lever on and above one top-k tile; lever off,
/// the plan and the allocations are what they were.
#[test]
fn the_radix_region_follows_the_lever_and_the_tile() {
    let c = cfg(540_672);
    let long = DsaSelectGeometry::plan(&c, 540_672, 3).unwrap();
    let short = DsaSelectGeometry::plan(&c, 4 * topk_tile(), 3).unwrap();
    assert_eq!(radix_region_bytes(&long, &c, false), 0);
    assert_eq!(radix_region_bytes(&long, &c, true), 3 * 37_056);
    assert_eq!(
        radix_region_bytes(&short, &c, true),
        0,
        "exactly one tile of pools"
    );
    let (off, _) = DsaSelectScratch::plan_bytes_for(&c, &[long], false);
    let (on, _) = DsaSelectScratch::plan_bytes_for(&c, &[long], true);
    assert_eq!(off[..6], on[..6]);
    assert_eq!((off[6], on[6]), (0, 3 * 37_056));

    let gpu = MockGpuBackend::new();
    let n0 = gpu.live_alloc_count();
    let plan_off = DsaSelectScratch::plan_bytes_for(&c, &[long], false);
    let a = DsaSelectScratch::alloc_sized(&gpu, plan_off).unwrap();
    assert_eq!(
        gpu.live_alloc_count() - n0,
        7,
        "lever off: the seven regions as before"
    );
    let n1 = gpu.live_alloc_count();
    let plan_on = DsaSelectScratch::plan_bytes_for(&c, &[long], true);
    let b = DsaSelectScratch::alloc_sized(&gpu, plan_on).unwrap();
    assert_eq!(
        gpu.live_alloc_count() - n1,
        8,
        "lever on: plus the radix region"
    );
    a.free(&gpu).unwrap();
    b.free(&gpu).unwrap();
    assert_eq!(gpu.live_alloc_count(), n0);
}

/// 2026-10-06: The dispatch table: the lever first, then the entry points (warned even at a
/// short context), then the tile, then the scratch.
#[test]
fn radix_mode_table() {
    let big = topk_tile() + 1;
    assert_eq!(radix_mode(false, true, true, big), RadixMode::Off);
    assert_eq!(
        radix_mode(true, false, true, big),
        RadixMode::MissingKernels
    );
    assert_eq!(radix_mode(true, false, true, 1), RadixMode::MissingKernels);
    assert_eq!(radix_mode(true, true, true, topk_tile()), RadixMode::Small);
    assert_eq!(radix_mode(true, true, false, big), RadixMode::NoScratch);
    assert_eq!(radix_mode(true, true, true, big), RadixMode::Engaged);
    assert_eq!(
        RADIX_ENGAGED_LINE,
        "METRALE_GLM_DSA_TOPK_RADIX=1: ENGAGED - radix top-k"
    );
}

/// 2026-10-06: `1` (blanks ignored) is on; anything else is off.
#[test]
fn the_lever_parses_as_a_switch() {
    assert!(parse_topk_radix(Some("1")));
    assert!(parse_topk_radix(Some(" 1 ")));
    for v in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some("on"),
        Some("2"),
    ] {
        assert!(!parse_topk_radix(v), "{v:?}");
    }
}

/// 2026-10-06: The five entry points are looked up when the kernels are resolved and are
/// optional: a module without one resolves, and the radix path reads as unresolved.
#[test]
fn the_radix_entry_points_are_optional() {
    let gpu = MockGpuBackend::new();
    let k = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    assert!(radix_resolved(&k));
    for name in ENTRY_POINTS {
        let want = (DSA_MODULE.to_string(), name.to_string());
        assert!(gpu.kernel_lookups_snapshot().contains(&want), "{name}");
    }
    for name in ENTRY_POINTS {
        let gpu = MockGpuBackend::new();
        gpu.deny_kernel(DSA_MODULE, name);
        let k = Glm5NextDsaKernels::resolve(&gpu).expect("resolves without it");
        assert!(!radix_resolved(&k), "{name} denied");
        assert_ne!(k.topk_pools.0, 0);
    }
}

const ENTRY_POINTS: [&str; 5] = [
    "dsa_topk_radix_init",
    "dsa_topk_radix_hist",
    "dsa_topk_radix_find",
    "dsa_topk_radix_gather",
    "dsa_topk_radix_sort",
];

/// 2026-10-06: One call is nine launches in order (init; hist, find for each pass; gather;
/// sort) with the grids and argument counts of the kernels' parameter lists.
#[test]
fn launch_sequence_and_grids() {
    let gpu = MockGpuBackend::new();
    let k = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    let t = RadixTopk {
        scores: DevicePtr(0x1000),
        selected: DevicePtr(0x2000),
        work: DevicePtr(0x3000),
        q_rows: 3,
        n_pools: 133_120,
        select_k: 512,
        kcap: 512,
        geom_dev: DevicePtr::NULL,
    };
    launch_topk_radix(&gpu, &k, &t, 7).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 9);
    let rows: [u32; 3] = [3, 1, 1];
    let chunked: [u32; 3] = [32, 3, 1];
    let want = [
        (rows, 4),
        (chunked, 7),
        (rows, 6),
        (chunked, 7),
        (rows, 6),
        (chunked, 7),
        (rows, 6),
        (chunked, 6),
        (rows, 5),
    ];
    for (i, (launch, (grid, args))) in l.iter().zip(want).enumerate() {
        assert_eq!(launch.grid, grid, "launch {i}");
        assert_eq!(launch.block, [RADIX_THREADS, 1, 1], "launch {i}");
        assert_eq!(
            launch.shared_mem, 0,
            "launch {i}: static shared memory only"
        );
        assert_eq!(launch.stream, 7);
        assert_eq!(launch.args.len(), args, "launch {i}");
    }
    // 2026-10-06: The pass argument of the three hist launches.
    for (i, pass) in [(1usize, 0u32), (3, 1), (5, 2)] {
        assert!(
            matches!(&l[i].args[5], MockArg::Bytes(b) if b[..] == pass.to_le_bytes()),
            "hist {i}"
        );
    }
    // 2026-10-06: A scalar select_k above kcap is refused before any launch.
    let gpu = MockGpuBackend::new();
    let bad = RadixTopk { kcap: 256, ..t };
    assert!(launch_topk_radix(&gpu, &k, &bad, 0).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}

/// 2026-10-06: `dsa_indexer.cu` defines the five entry points with the parameter lists the
/// launcher passes, and its layout defines equal the Rust constants.
#[test]
fn radix_defines_and_entry_points_match_the_kernel_file() {
    let cu = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(format!("{DSA_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).expect("dsa_indexer.cu readable");
    let params = |name: &str| -> String {
        let head =
            format!("extern \"C\" __global__ void __launch_bounds__(DSA_RADIX_THREADS) {name}(");
        let Some(at) = src.find(&head) else {
            panic!("{cu:?} lacks `{head}`");
        };
        let start = at + head.len();
        let len = src[start..].find(')').expect("parameter list closes");
        src[start..start + len]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let geom = "const int* __restrict__ geom";
    let work = "unsigned int* __restrict__ work";
    let scores = "const float* __restrict__ scores";
    for (name, want) in [
        (
            "dsa_topk_radix_init",
            format!("{work}, unsigned int select_k, unsigned int kcap, {geom}"),
        ),
        (
            "dsa_topk_radix_hist",
            format!(
                "{scores}, {work}, unsigned int P, unsigned int select_k, unsigned int kcap, \
                 unsigned int pass, {geom}"
            ),
        ),
        (
            "dsa_topk_radix_find",
            format!(
                "{work}, unsigned int select_k, unsigned int kcap, unsigned int pass, \
                 unsigned int chunks, {geom}"
            ),
        ),
        (
            "dsa_topk_radix_gather",
            format!(
                "{scores}, {work}, unsigned int P, unsigned int select_k, unsigned int kcap, \
                 {geom}"
            ),
        ),
        (
            "dsa_topk_radix_sort",
            format!(
                "const unsigned int* __restrict__ work, int* __restrict__ selected, \
                 unsigned int select_k, unsigned int kcap, {geom}"
            ),
        ),
    ] {
        assert_eq!(params(name), want, "{name}");
    }
    for (define, v) in [
        ("DSA_RADIX_THREADS", RADIX_THREADS as usize),
        ("DSA_RADIX_MAX_CHUNKS", RADIX_MAX_CHUNKS),
        ("DSA_RADIX_BINS", RADIX_BINS),
        ("DSA_RADIX_BINS_LAST", RADIX_BINS_LAST),
        ("DSA_RADIX_STATE", RADIX_STATE),
        ("DSA_RADIX_STATE_WORDS", RADIX_STATE_WORDS),
        ("DSA_RADIX_TIE", RADIX_TIE),
        ("DSA_RADIX_CAND", RADIX_CAND),
        ("DSA_RADIX_SORT_MAX", RADIX_SORT_MAX),
    ] {
        let line = format!("#define {define} {v}u");
        assert!(
            src.lines().any(|l| l.trim() == line),
            "{cu:?}: `{line}` does not match the Rust constant"
        );
    }
}
