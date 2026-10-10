// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Host tests of `METRALE_GLM_DSA_TOPK_RADIX_COMPACT`: a host model of
//! `dsa_topk_radix_compact` + `dsa_topk_radix_select0` + the final sort against the full order,
//! the sizing and dispatch, the launch sequence on a mock, and the `.cu` defines and entry
//! points (including that `dsa_topk_radix_hist_rc` is `dsa_topk_radix_hist` plus its early
//! return).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::select::radix::{
    RADIX_BINS, RADIX_SORT_MAX, RadixTopk, launch_topk_radix, radix_chunks, radix_key,
    radix_region_bytes_with, radix_work_bytes,
};
use crate::glm5next_dsa::select::{DsaSelectGeometry, DsaSelectScratch, topk_tile};
use crate::glm5next_dsa::{DSA_MODULE, Glm5NextDsaConfig};
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

/// 2026-10-10: GLM-5.3 DSA geometry (as `radix_tests.rs`).
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

/// 2026-10-10: The order `dsa_topk_pools` selects in: score descending (float compare), then
/// pool index ascending; the first `k` indices.
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

/// 2026-10-10: The kernels' bitonic network over `n2` (a power of two) pairs, step for step:
/// key descending, then index ascending (`dsa_topk_radix_select0` and `dsa_topk_radix_sort`).
fn bitonic(sk: &mut [u32], si: &mut [u32]) {
    let n2 = sk.len();
    assert!(n2.is_power_of_two() && si.len() == n2);
    let mut k = 2;
    while k <= n2 {
        let mut j = k >> 1;
        while j > 0 {
            for i in 0..n2 {
                let l = i ^ j;
                if l > i {
                    let gt = sk[i] > sk[l] || (sk[i] == sk[l] && si[i] < si[l]);
                    let want_desc = (i & k) == 0;
                    if want_desc != gt {
                        sk.swap(i, l);
                        si.swap(i, l);
                    }
                }
            }
            j >>= 1;
        }
        k <<= 1;
    }
}

/// 2026-10-10: Deterministic generator.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn f(&mut self) -> f32 {
        ((self.next() >> 7) as f32) / ((1u64 << 24) as f32)
    }
    fn pick(&mut self, from: &[f32]) -> f32 {
        from[(self.next() as usize) % from.len()]
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next() as usize) % (i + 1));
        }
    }
}

/// 2026-10-10: Host model of one row of the lever-on sequence up to `selected`: hist 0 and
/// find 0 (the bin b0 and need0), then, for a COMPACT row, the compaction with the atomic
/// slot order scrambled by `seed` (chunks visited and keys within a chunk in a shuffled
/// order), select0 (padded bitonic sort, first need0) and the final sort of the select_k
/// candidates. `None` for a FALLBACK row (hist0[b0] > COMPACT_CAP), which runs the lever-off
/// passes unchanged.
fn compact_model(scores: &[f32], select_k: usize, chunks: usize, seed: u64) -> Option<Vec<i32>> {
    let p = scores.len();
    assert!(select_k >= 1 && select_k <= p);
    let keys: Vec<u32> = scores.iter().map(|&v| radix_key(v)).collect();
    let mut hist0 = vec![0usize; RADIX_BINS];
    for &k in &keys {
        hist0[(k >> 20) as usize] += 1;
    }
    let (mut cum, mut b0) = (0usize, usize::MAX);
    for b in (0..RADIX_BINS).rev() {
        if cum + hist0[b] >= select_k {
            b0 = b;
            break;
        }
        cum += hist0[b];
    }
    assert_ne!(b0, usize::MAX, "find 0 reaches select_k");
    let need0 = select_k - cum;
    assert!(need0 >= 1 && need0 <= hist0[b0]);
    if hist0[b0] > COMPACT_CAP {
        return None;
    }
    let ngt0 = select_k - need0;

    // compact: GT_CNT and BINCNT slots in a scrambled (atomic) order.
    let mut g = Lcg(seed);
    let chunk = p.div_ceil(chunks);
    let mut order: Vec<usize> = (0..chunks).collect();
    g.shuffle(&mut order);
    let (mut gt, mut bin) = (Vec::new(), Vec::new());
    for c in order {
        let lo = (c * chunk).min(p);
        let mut idx: Vec<usize> = (lo..(lo + chunk).min(p)).collect();
        g.shuffle(&mut idx);
        for i in idx {
            let top = (keys[i] >> 20) as usize;
            if top > b0 {
                gt.push((keys[i], i as u32));
            } else if top == b0 {
                bin.push((keys[i], i as u32));
            }
        }
    }
    assert_eq!(gt.len(), ngt0, "exactly select_k - need0 keys above b0");
    assert_eq!(bin.len(), hist0[b0], "BINCNT equals hist0[b0]");

    // select0: pad with (0, 0xFFFFFFFF), sort, take need0.
    let n2 = bin.len().next_power_of_two().max(2);
    let mut sk = vec![0u32; n2];
    let mut si = vec![u32::MAX; n2];
    for (j, &(k, i)) in bin.iter().enumerate() {
        sk[j] = k;
        si[j] = i;
    }
    bitonic(&mut sk, &mut si);
    assert!(
        si[..need0].iter().all(|&i| i != u32::MAX),
        "no pad is taken"
    );
    let mut cand = gt;
    cand.extend((0..need0).map(|j| (sk[j], si[j])));
    assert_eq!(cand.len(), select_k);
    assert!(select_k <= RADIX_SORT_MAX);

    // sort (`dsa_topk_radix_sort`, padded with key 0 and INT_MAX).
    let n2 = select_k.next_power_of_two().max(2);
    let mut sk = vec![0u32; n2];
    let mut si = vec![i32::MAX as u32; n2];
    for (j, &(k, i)) in cand.iter().enumerate() {
        sk[j] = k;
        si[j] = i;
    }
    bitonic(&mut sk, &mut si);
    Some(si[..select_k].iter().map(|&i| i as i32).collect())
}

/// 2026-10-10: Named rows and whether each is expected COMPACT.
fn cases() -> Vec<(&'static str, Vec<f32>, bool)> {
    let mut g = Lcg(0x5EED_C0DE);
    let m = -f32::MAX;
    // Rig-like: about 4 in 23 pools -FLT_MAX, the rest sums of 8 w * relu(x).
    let rig = |g: &mut Lcg, p: usize| -> Vec<f32> {
        (0..p)
            .map(|_| {
                if g.f() < 4.0 / 23.0 {
                    return m;
                }
                (0..8).fold(0.0f32, |acc, _| {
                    let w = g.f() * 0.3;
                    let x = g.f() * 2.0 - 1.0;
                    acc + w * x.max(0.0)
                })
            })
            .collect()
    };
    // need0 == hist0[b0]: 500 keys above, 12 distinct keys in [1.0, 1.125), the rest 0.1.
    let mut exact = vec![0.1f32; 9_000];
    for i in 0..500 {
        exact[(i * 17 + 5) % 9_000] = 2.0 + (i as f32) / 256.0;
    }
    let mut placed = 0;
    for i in (0..9_000).rev() {
        if exact[i] == 0.1 && placed < 12 {
            exact[i] = 1.0 + (placed as f32) / 128.0;
            placed += 1;
        }
    }
    let ties = |real: usize, neg: usize, g: &mut Lcg| -> Vec<f32> {
        let mut v: Vec<f32> = (0..real)
            .map(|_| g.f())
            .chain((0..neg).map(|_| m))
            .collect();
        g.shuffle(&mut v);
        v
    };
    vec![
        ("rig 32768", rig(&mut g, 32_768), true),
        ("rig 9001 (ragged)", rig(&mut g, 9_001), true),
        (
            "spread",
            (0..20_011).map(|_| (g.f() - 0.5) * 1e4).collect(),
            true,
        ),
        ("-FLT_MAX bin <= cap", ties(300, 2_000, &mut g), true),
        ("-FLT_MAX bin > cap", ties(300, 2_249, &mut g), false),
        ("all -FLT_MAX, P 2048", vec![m; 2_048], true),
        ("all equal, P 2000", vec![0.75; 2_000], true),
        ("all equal, P 3000", vec![0.75; 3_000], false),
        ("need0 == bin count", exact, true),
        ("P = select_k", (0..512).map(|_| g.f()).collect(), true),
        (
            "P = select_k, ties",
            (0..512).map(|_| g.pick(&[0.5, m])).collect(),
            true,
        ),
        // About 1,600 zeros of both signs in b0 (COMPACT); next, about 2,400 (FALLBACK).
        (
            "+-0 at the threshold",
            (0..12_000)
                .map(|i| {
                    if i % 120 == 7 {
                        0.5 + g.f()
                    } else {
                        g.pick(&[0.0, -0.0, m, m, m, m, m, m, m, m, m, m, m, m, m])
                    }
                })
                .collect(),
            true,
        ),
        (
            "+-0 at the threshold, > cap",
            (0..12_000)
                .map(|i| {
                    if i % 120 == 7 {
                        0.5 + g.f()
                    } else {
                        g.pick(&[0.0, -0.0, m, m, m, m, m, m, m, m])
                    }
                })
                .collect(),
            false,
        ),
        (
            "denormals",
            (0..1_500)
                .map(|_| g.pick(&[1e-45, -1e-45, 0.0, -0.0, 2e-45]))
                .collect(),
            true,
        ),
        // About 1,555 copies of the threshold value 0.4.
        (
            "threshold duplicates",
            (0..7_777)
                .map(|_| g.pick(&[0.1, 0.2, 0.3, 0.4, m]))
                .collect(),
            true,
        ),
    ]
}

/// 2026-10-10: For every COMPACT row the compaction + select0 + sort model selects exactly the
/// first select_k pools of the full order, whatever order the atomics hand out slots in and at
/// several chunk counts; FALLBACK rows are classified as expected.
#[test]
fn the_compaction_model_selects_exactly_the_first_select_k() {
    for (name, scores, compact) in cases() {
        for select_k in [512.min(scores.len()), 1, 37.min(scores.len())] {
            let want = reference(&scores, select_k);
            for chunks in [1, 3, 7, radix_chunks(1)] {
                for seed in [1u64, 2, 3] {
                    match compact_model(&scores, select_k, chunks, seed) {
                        Some(got) => assert_eq!(got, want, "{name}, k {select_k}, c {chunks}"),
                        None => assert!(
                            select_k != 512 || !compact,
                            "{name}: expected COMPACT at select_k 512"
                        ),
                    }
                }
            }
            if select_k == 512.min(scores.len()) {
                assert_eq!(
                    compact_model(&scores, select_k, 1, 0).is_some(),
                    compact,
                    "{name}: COMPACT/FALLBACK classification"
                );
            }
        }
    }
}

/// 2026-10-10: The pad (key 0, index 0xFFFFFFFF) sorts after every real pair: every finite
/// float (and -inf) keys above 0, and a pool index is below u32::MAX.
#[test]
fn the_select0_pad_sorts_last() {
    assert_eq!(radix_key(-f32::MAX), 0x0080_0000);
    assert_eq!(radix_key(f32::NEG_INFINITY), 0x007F_FFFF);
    let mut x = 0x9E37_79B9_u32;
    for _ in 0..100_000 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let f = f32::from_bits(x);
        if !f.is_nan() {
            assert!(radix_key(f) > 0, "{f:e}");
        }
    }
    // A sort of real pairs and pads leaves every pad after every real pair.
    let mut sk = vec![0x0080_0000, 0, 5, 0, 0x0080_0000, 0, 0, 0];
    let mut si = vec![9, u32::MAX, 3, u32::MAX, 2, u32::MAX, u32::MAX, u32::MAX];
    bitonic(&mut sk, &mut si);
    assert_eq!(&si[..3], &[2, 9, 3]);
    assert!(si[3..].iter().all(|&i| i == u32::MAX));
}

/// 2026-10-10: `1` (blanks ignored) is on; anything else is off.
#[test]
fn the_lever_parses_as_a_switch() {
    assert!(parse_radix_compact(Some("1")));
    assert!(parse_radix_compact(Some(" 1 ")));
    for v in [None, Some(""), Some("0"), Some("true"), Some("2")] {
        assert!(!parse_radix_compact(v), "{v:?}");
    }
    assert_eq!(
        RADIX_COMPACT_ENGAGED_LINE,
        "METRALE_GLM_DSA_TOPK_RADIX_COMPACT ENGAGED"
    );
}

/// 2026-10-10: The pair buffer adds 16 KiB a row to region 6, only with both levers on and a
/// radix region; the dispatch needs the lever, the entry points and that many bytes.
#[test]
fn the_pair_buffer_extends_region_6_only_with_the_lever() {
    let c = cfg(540_672);
    assert_eq!(compact_bytes(1), 16_384);
    assert_eq!(compact_bytes(128), 2 << 20);
    let long = DsaSelectGeometry::plan(&c, 540_672, 3).unwrap();
    let short = DsaSelectGeometry::plan(&c, 4 * topk_tile(), 3).unwrap();
    let work = radix_work_bytes(&c, 3);
    assert_eq!(radix_region_bytes_with(&long, &c, true, false), work);
    assert_eq!(
        radix_region_bytes_with(&long, &c, true, true),
        work + 3 * 16_384
    );
    assert_eq!(radix_region_bytes_with(&long, &c, false, true), 0);
    assert_eq!(radix_region_bytes_with(&short, &c, true, true), 0);
    // The environment leaves the lever off in tests: plans are unchanged.
    let (on, _) = DsaSelectScratch::plan_bytes_for(&c, &[long], true);
    assert_eq!(on[6], work);

    let need = work + compact_bytes(3);
    assert!(radix_compact_for(true, true, need, work, 3));
    assert!(!radix_compact_for(false, true, need, work, 3));
    assert!(!radix_compact_for(true, false, need, work, 3));
    assert!(!radix_compact_for(true, true, need - 1, work, 3));
}

/// 2026-10-10: The five entry points are optional: a module without one resolves, the radix
/// path still reads as resolved, the compaction does not, and a call with `bins` set fails
/// before any launch.
#[test]
fn the_compaction_entry_points_are_optional() {
    let gpu = MockGpuBackend::new();
    let k = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    assert!(radix_compact_resolved(&k));
    for name in ENTRY_POINTS {
        let gpu = MockGpuBackend::new();
        gpu.deny_kernel(DSA_MODULE, name);
        let k = Glm5NextDsaKernels::resolve(&gpu).expect("resolves without it");
        assert!(super::super::radix::radix_resolved(&k), "{name} denied");
        assert!(!radix_compact_resolved(&k), "{name} denied");
        assert!(launch_topk_radix(&gpu, &k, &topk(DevicePtr(0x4000)), 0).is_err());
        assert!(gpu.launches_snapshot().is_empty(), "{name} denied");
    }
}

const ENTRY_POINTS: [&str; 5] = [
    "dsa_topk_radix_compact",
    "dsa_topk_radix_select0",
    "dsa_topk_radix_hist_rc",
    "dsa_topk_radix_find_rc",
    "dsa_topk_radix_gather_rc",
];

fn topk(bins: DevicePtr) -> RadixTopk {
    RadixTopk {
        scores: DevicePtr(0x1000),
        selected: DevicePtr(0x2000),
        work: DevicePtr(0x3000),
        q_rows: 3,
        n_pools: 133_120,
        select_k: 512,
        kcap: 512,
        geom_dev: DevicePtr::NULL,
        bins,
    }
}

/// 2026-10-10: With `bins` set one call is eleven launches: init, hist 0, find 0, compact,
/// select0, hist_rc 1, find_rc 1, hist_rc 2, find_rc 2, gather_rc, sort, with the grids and
/// argument counts of the kernels' parameter lists; with `bins` NULL the lever-off nine.
#[test]
fn compaction_launch_sequence() {
    let gpu = MockGpuBackend::new();
    let k = Glm5NextDsaKernels::resolve(&gpu).unwrap();
    let bins = DevicePtr(0x4000);
    launch_topk_radix(&gpu, &k, &topk(bins), 7).unwrap();
    let l = gpu.launches_snapshot();
    let rows: [u32; 3] = [3, 1, 1];
    let chunked: [u32; 3] = [32, 3, 1];
    let want = [
        (k.topk_radix_init, rows, 4),
        (k.topk_radix_hist, chunked, 7),
        (k.topk_radix_find, rows, 6),
        (k.topk_radix_compact, chunked, 7),
        (k.topk_radix_select0, rows, 5),
        (k.topk_radix_hist_rc, chunked, 7),
        (k.topk_radix_find_rc, rows, 6),
        (k.topk_radix_hist_rc, chunked, 7),
        (k.topk_radix_find_rc, rows, 6),
        (k.topk_radix_gather_rc, chunked, 6),
        (k.topk_radix_sort, rows, 5),
    ];
    assert_eq!(l.len(), want.len());
    for (i, (launch, (h, grid, args))) in l.iter().zip(want).enumerate() {
        assert_eq!(launch.func, h.0, "launch {i}");
        assert_eq!(launch.grid, grid, "launch {i}");
        assert_eq!(launch.block, [RADIX_THREADS, 1, 1], "launch {i}");
        assert_eq!(launch.shared_mem, 0, "launch {i}");
        assert_eq!(launch.args.len(), args, "launch {i}");
    }
    assert_eq!(l[3].args[2], MockArg::Buffer(bins), "compact bins");
    assert_eq!(l[4].args[1], MockArg::Buffer(bins), "select0 bins");

    let gpu = MockGpuBackend::new();
    launch_topk_radix(&gpu, &k, &topk(DevicePtr::NULL), 7).unwrap();
    let l = gpu.launches_snapshot();
    let off = [
        k.topk_radix_init,
        k.topk_radix_hist,
        k.topk_radix_find,
        k.topk_radix_hist,
        k.topk_radix_find,
        k.topk_radix_hist,
        k.topk_radix_find,
        k.topk_radix_gather,
        k.topk_radix_sort,
    ];
    assert_eq!(
        l.iter().map(|x| x.func).collect::<Vec<_>>(),
        off.iter().map(|h| h.0).collect::<Vec<_>>(),
        "lever off: the nine launches as before"
    );
}

/// 2026-10-10: `dsa_indexer.cu`: the defines equal the Rust constants; compact and select0 take
/// the parameters the launchers pass; each `_rc` entry point takes its lever-off twin's
/// parameters; and `dsa_topk_radix_hist_rc`'s body is `dsa_topk_radix_hist`'s plus the early
/// return (the copy kept so the lever-off hist's SASS is unchanged).
#[test]
fn compaction_defines_and_entry_points_match_the_kernel_file() {
    let cu = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .join(format!("{DSA_MODULE}.cu"));
    let src = std::fs::read_to_string(&cu).expect("dsa_indexer.cu readable");
    let head = |name: &str| {
        format!("extern \"C\" __global__ void __launch_bounds__(DSA_RADIX_THREADS) {name}(")
    };
    let params = |name: &str| -> String {
        let h = head(name);
        let at = src.find(&h).unwrap_or_else(|| panic!("{cu:?} lacks `{h}`"));
        let start = at + h.len();
        let len = src[start..].find(')').expect("parameter list closes");
        src[start..start + len]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };
    let body = |name: &str| -> String {
        let at = src.find(&head(name)).expect("entry point");
        let open = at + src[at..].find(") {\n").expect("body opens");
        let close = open + src[open..].find("\n}\n").expect("body closes");
        src[open..close].to_string()
    };
    let geom = "const int* __restrict__ geom";
    assert_eq!(
        params("dsa_topk_radix_compact"),
        format!(
            "const float* __restrict__ scores, unsigned int* __restrict__ work, \
             unsigned int* __restrict__ bins, unsigned int P, unsigned int select_k, \
             unsigned int kcap, {geom}"
        )
    );
    assert_eq!(
        params("dsa_topk_radix_select0"),
        format!(
            "unsigned int* __restrict__ work, const unsigned int* __restrict__ bins, \
             unsigned int select_k, unsigned int kcap, {geom}"
        )
    );
    for twin in ["hist", "find", "gather"] {
        let off = format!("dsa_topk_radix_{twin}");
        assert_eq!(params(&format!("{off}_rc")), params(&off), "{twin}");
    }
    let early = "    // 2026-10-10: A COMPACT row (dsa_topk_radix_select0 set its mode word) is already\n    \
                 // selected; every thread reads the same word, so the whole block returns.\n    \
                 if (w[DSA_RADIX_STATE + DSA_RS_MODE] != 0u) return;\n";
    let rc = body("dsa_topk_radix_hist_rc");
    assert_eq!(rc.matches(early).count(), 1, "hist_rc has its early return");
    assert_eq!(
        rc.replacen(early, "", 1),
        body("dsa_topk_radix_hist"),
        "hist_rc is hist plus the early return"
    );
    let defined = |name: &str| -> usize {
        let line = src
            .lines()
            .find(|l| l.split_whitespace().take(2).eq(["#define", name]))
            .unwrap_or_else(|| panic!("{cu:?} lacks #define {name}"));
        line.split_whitespace()
            .nth(2)
            .unwrap()
            .trim_end_matches('u')
            .parse()
            .unwrap()
    };
    assert_eq!(defined("DSA_COMPACT_CAP"), COMPACT_CAP);
    assert_eq!(defined("DSA_RS_BINCNT"), RS_BINCNT);
    assert_eq!(defined("DSA_RS_MODE"), RS_MODE);
    assert_eq!(defined("DSA_RS_B0CNT"), super::super::binstat::RS_B0CNT);
    assert_eq!(defined("DSA_RS_BINP"), super::super::binstat::RS_BINP);
    // Slots 4 to 7 are distinct and inside the zeroed state words.
    let slots = [4usize, 5, RS_BINCNT, RS_MODE];
    assert!(
        slots
            .iter()
            .all(|&s| s > 3 && s < super::super::radix::RADIX_STATE_WORDS)
    );
    assert_eq!(COMPACT_CAP, RADIX_SORT_MAX);
}
