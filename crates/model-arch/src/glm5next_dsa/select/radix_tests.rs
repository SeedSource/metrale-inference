// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Host tests of `METRALE_GLM_DSA_TOPK_RADIX`: the key transform, a host model of
//! the five radix kernels against a plain sort on adversarial score sets, the planning and
//! sizing functions, the dispatch table, the launch sequence on a mock, and the `.cu` defines.
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

/// 2026-10-06: The order `dsa_topk_pools` selects in: score descending (float compare, so -0.0
/// ties +0.0), then pool index ascending; the first `k` indices.
fn reference(scores: &[f32], k: usize) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&i, &j| {
        scores[j]
            .partial_cmp(&scores[i])
            .expect("finite scores")
            .then(i.cmp(&j))
    });
    idx.into_iter().take(k).map(|i| i as i32).collect()
}

fn matches(key: u32, prefix: u32, pass: u32) -> bool {
    match pass {
        0 => true,
        1 => key >> 20 == prefix >> 20,
        _ => key >> 8 == prefix >> 8,
    }
}

fn digit(key: u32, pass: u32) -> usize {
    (match pass {
        0 => key >> 20,
        1 => (key >> 8) & 0xFFF,
        _ => key & 0xFF,
    }) as usize
}

/// 2026-10-06: Host model of one row of the radix kernels, step for step: chunking, the three
/// hist/find passes with the per-thread bin split and exclusive scan (asserting exactly one
/// thread finds the bin), the tie bases, the gather and the final sort. The kernels' passes 0
/// and 1 add the chunk histograms into one with atomics; summing per-chunk counts is the same.
fn model(scores: &[f32], select_k: usize, chunks: usize) -> Vec<i32> {
    if select_k == 0 {
        return Vec::new();
    }
    let p = scores.len();
    let keys: Vec<u32> = scores.iter().map(|&v| radix_key(v)).collect();
    let chunk = p.div_ceil(chunks);
    let range = |c: usize| {
        let lo = (c * chunk).min(p);
        lo..(lo + chunk).min(p)
    };
    let threads = RADIX_THREADS as usize;
    let (mut prefix, mut need) = (0u32, select_k as u32);
    let mut tie_base = vec![0u32; chunks];
    let mut ngt = 0u32;
    for pass in 0..RADIX_PASSES {
        let nbins = if pass == 2 { RADIX_BINS_LAST } else { RADIX_BINS };
        let mut ch = vec![vec![0u32; nbins]; chunks];
        for (c, h) in ch.iter_mut().enumerate() {
            for i in range(c) {
                if matches(keys[i], prefix, pass) {
                    h[digit(keys[i], pass)] += 1;
                }
            }
        }
        let count = |b: usize| ch.iter().map(|h| h[b]).sum::<u32>();
        let per = nbins / threads;
        let (mut excl, mut found, mut next) = (0u32, 0, (prefix, need));
        for t in 0..threads {
            let top = nbins - 1 - t * per;
            let local: u32 = (0..per).map(|j| count(top - j)).sum();
            if excl < need && need <= excl + local {
                found += 1;
                let (mut cum, mut b) = (excl, top);
                for j in 0..per {
                    b = top - j;
                    let h = count(b);
                    if cum + h >= need {
                        break;
                    }
                    cum += h;
                }
                let shift = [20, 8, 0][pass as usize];
                next = (prefix | ((b as u32) << shift), need - cum);
                if pass == 2 {
                    ngt = select_k as u32 - (need - cum);
                    let mut run = 0;
                    for (c, base) in tie_base.iter_mut().enumerate() {
                        *base = run;
                        run += ch[c][b];
                    }
                }
            }
            excl += local;
        }
        assert_eq!(found, 1, "pass {pass}: exactly one thread holds the threshold bin");
        (prefix, need) = next;
    }
    let (tau, m) = (prefix, need);
    let mut cand = vec![None; select_k];
    let mut gt = 0u32;
    for (c, &base) in tie_base.iter().enumerate() {
        let mut run = base;
        for i in range(c) {
            if keys[i] > tau {
                cand[gt as usize] = Some((keys[i], i));
                gt += 1;
            } else if keys[i] == tau {
                if run < m {
                    cand[(ngt + run) as usize] = Some((keys[i], i));
                }
                run += 1;
            }
        }
    }
    assert_eq!(gt, ngt, "count(key > tau) is n_gt");
    let mut cand: Vec<(u32, usize)> = cand.into_iter().map(|c| c.expect("filled")).collect();
    cand.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    cand.into_iter().map(|(_, i)| i as i32).collect()
}

/// 2026-10-06: Deterministic generator for the score sets.
struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn pick(&mut self, from: &[f32]) -> f32 {
        from[((self.f() * from.len() as f32) as usize).min(from.len() - 1)]
    }
}

/// 2026-10-06: The adversarial sets of the GPU microtest, at host-test sizes.
fn cases() -> Vec<(&'static str, Vec<f32>)> {
    let mut g = Lcg(0xD5A_70C);
    let m = -f32::MAX;
    let mut exact_k = vec![m; 6_000];
    for i in 0..512 {
        exact_k[(i * 11 + 3) % 6_000] = 0.25 + g.f();
    }
    vec![
        ("all equal", vec![1.0; 3_000]),
        ("all -FLT_MAX", vec![m; 2_500]),
        ("half -FLT_MAX", (0..5_000).map(|_| if g.f() < 0.5 { m } else { g.f() }).collect()),
        ("+-0 mix", (0..4_000).map(|_| g.pick(&[0.0, -0.0, 1e-3, m])).collect()),
        ("exactly select_k candidates", exact_k),
        ("P = select_k", (0..512).map(|_| g.f()).collect()),
        ("P = select_k + 1", (0..513).map(|_| g.f()).collect()),
        ("P = 2049", (0..2_049).map(|_| g.pick(&[0.5, 0.25, 0.125])).collect()),
        ("threshold duplicates", (0..7_777).map(|_| g.pick(&[0.1, 0.2, 0.3, 0.4, m])).collect()),
        ("spread", (0..20_011).map(|_| (g.f() - 0.5) * 1e4).collect()),
        ("denormals", (0..3_001).map(|_| g.pick(&[1e-45, -1e-45, 0.0, -0.0, 2e-45])).collect()),
    ]
}

/// 2026-10-06: The host model of the kernels selects exactly what the comparator does, for
/// every adversarial set, at several chunk counts (one, uneven, the 1-row production 32, and
/// more chunks than some sets have pools per chunk).
#[test]
fn the_radix_model_selects_exactly_the_first_select_k() {
    for (name, scores) in cases() {
        let k = 512.min(scores.len());
        let want = reference(&scores, k);
        for chunks in [1, 3, 7, radix_chunks(1), RADIX_MAX_CHUNKS] {
            assert_eq!(model(&scores, k, chunks), want, "{name}, {chunks} chunks");
        }
    }
    // 2026-10-06: Tiny rows, every select_k, chunks past the pool count (empty chunks).
    let tiny = [0.0, -0.0, -f32::MAX, 1.0, -f32::MAX, 1.0, 0.0];
    for k in 0..=tiny.len() {
        for chunks in [1, 2, 32] {
            assert_eq!(model(&tiny, k, chunks), reference(&tiny, k), "k {k}, {chunks} chunks");
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
        assert!((1..=RADIX_MAX_CHUNKS).contains(&radix_chunks(q)), "q_rows {q}");
    }
    // 2026-10-06: The pass-2 per-chunk histograms reuse the two 4096-bin histograms.
    assert!(RADIX_MAX_CHUNKS * RADIX_BINS_LAST <= RADIX_STATE);
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
    assert_eq!(radix_region_bytes(&short, &c, true), 0, "exactly one tile of pools");
    let (off, _) = DsaSelectScratch::plan_bytes_for(&c, &[long], false);
    let (on, _) = DsaSelectScratch::plan_bytes_for(&c, &[long], true);
    assert_eq!(off[..6], on[..6]);
    assert_eq!((off[6], on[6]), (0, 3 * 37_056));

    let gpu = MockGpuBackend::new();
    let n0 = gpu.live_alloc_count();
    let plan_off = DsaSelectScratch::plan_bytes_for(&c, &[long], false);
    let a = DsaSelectScratch::alloc_sized(&gpu, plan_off).unwrap();
    assert_eq!(gpu.live_alloc_count() - n0, 7, "lever off: the seven regions as before");
    let n1 = gpu.live_alloc_count();
    let plan_on = DsaSelectScratch::plan_bytes_for(&c, &[long], true);
    let b = DsaSelectScratch::alloc_sized(&gpu, plan_on).unwrap();
    assert_eq!(gpu.live_alloc_count() - n1, 8, "lever on: plus the radix region");
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
    assert_eq!(radix_mode(true, false, true, big), RadixMode::MissingKernels);
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
    for v in [None, Some(""), Some("0"), Some("true"), Some("on"), Some("2")] {
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
    let rows = [3, 1, 1];
    let chunked = [32, 3, 1];
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
        assert_eq!(launch.shared_mem, 0, "launch {i}: static shared memory only");
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
        let head = format!(
            "extern \"C\" __global__ void __launch_bounds__(DSA_RADIX_THREADS) {name}("
        );
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
