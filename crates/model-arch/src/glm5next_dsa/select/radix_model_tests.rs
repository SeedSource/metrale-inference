// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: A host model of the five radix top-k kernels (`dsa_topk_radix_*`), step for step,
//! checked against a plain sort on adversarial score sets at several chunk counts. It checks the
//! algorithm (passes, bin split, tie bases, gather slots); the GPU gate is
//! `examples/dsa_topk_radix_microtest.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;

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

/// 2026-10-06: `dsa_radix_match`.
fn matches(key: u32, prefix: u32, pass: u32) -> bool {
    match pass {
        0 => true,
        1 => key >> 20 == prefix >> 20,
        _ => key >> 8 == prefix >> 8,
    }
}

/// 2026-10-06: `dsa_radix_digit`.
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
