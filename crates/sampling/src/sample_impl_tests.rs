// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `sample_top_p_prefix` (top-p with no top-k cap and no min-p) against
//! the fully sorted path it replaces for that case.
//!
//! Owner: sampling.
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-14: Deterministic logits with no ties: a shuffled even grid over
/// [-scale, scale). `scale` sets how flat the distribution is (0.05 needs most of the
/// vocabulary for the nucleus, 12.0 a few tokens).
fn logits(n: usize, scale: f32, salt: u64) -> Vec<u8> {
    let mut grid: Vec<f32> = (0..n)
        .map(|i| (2.0 * i as f32 / n as f32 - 1.0) * scale)
        .collect();
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
    for i in (1..n).rev() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        grid.swap(i, ((x >> 33) as usize) % (i + 1));
    }
    grid.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn params(temperature: f32, top_p: f32, min_p: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_p,
        min_p,
        ..SamplingParams::greedy(16)
    }
}

/// 2026-10-14: Each token's cumulative share (f64, descending logit, after the
/// sampler's own f32 temperature scaling) up to and including it.
fn quantiles(data: &[u8], t: f32) -> std::collections::HashMap<u32, f64> {
    let mut v: Vec<(u32, f32)> = data
        .chunks_exact(4)
        .enumerate()
        .map(|(i, c)| (i as u32, f32::from_le_bytes([c[0], c[1], c[2], c[3]]) / t))
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let max = v[0].1 as f64;
    let w: Vec<f64> = v.iter().map(|&(_, l)| (l as f64 - max).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut cum = 0.0;
    v.iter()
        .zip(&w)
        .map(|(&(i, _), &x)| {
            cum += x / total;
            (i, cum)
        })
        .collect()
}

/// 2026-10-14: The prefix path (min_p 0) and the sorted path (min_p 1e-30, which only
/// drops weights below 1e-30 of the maximum) draw at the same quantile of the same
/// nucleus. On a 1,000-token vocabulary they draw the same token for every seed. On
/// a 154,880-token one the total weight is summed in a different order (vocabulary
/// vs sorted), so a seeded draw can land on a nearby token: an f32 total over 154,880
/// terms carries up to ~1e-3 relative error on either path (1.1e-4 at quantile 0.26
/// measured), so the quantiles must agree within 1e-3 and both draws stay inside the
/// f64 nucleus.
#[test]
fn top_p_prefix_draws_what_the_sorted_path_draws() {
    for (n, scale) in [
        (1000usize, 4.0f32),
        (1000, 0.05),
        (154_880, 12.0),
        (154_880, 0.3),
    ] {
        let data = logits(n, scale, n as u64 ^ scale.to_bits() as u64);
        for (t, p) in [(1.0f32, 0.95f32), (0.6, 0.5), (1.0, 0.999)] {
            let q = quantiles(&data, t);
            for seed in 0..64u64 {
                let fast = sample_with_params_seeded(&data, &params(t, p, 0.0), &[], Some(seed));
                let sorted =
                    sample_with_params_seeded(&data, &params(t, p, 1e-30), &[], Some(seed));
                let at = format!("n={n} scale={scale} t={t} top_p={p} seed={seed}");
                if n == 1000 {
                    assert_eq!(fast, sorted, "{at}");
                }
                let (qf, qs) = (q[&fast], q[&sorted]);
                assert!((qf - qs).abs() <= 1e-3, "{at}: quantile {qf} vs {qs}");
                assert!(
                    qf.max(qs) <= p as f64 + 1e-3,
                    "{at}: {qf} outside the nucleus"
                );
            }
        }
    }
}

/// 2026-10-14: A peaked distribution keeps only the dominant token at top_p 0.5.
#[test]
fn top_p_prefix_keeps_the_nucleus_only() {
    let mut v = vec![0.0f32; 5000];
    v[1234] = 20.0;
    let data: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
    for seed in 0..32u64 {
        assert_eq!(
            sample_with_params_seeded(&data, &params(1.0, 0.5, 0.0), &[], Some(seed)),
            1234
        );
    }
}

/// 2026-10-14: Temperature 0 stays the greedy argmax: the new path is never reached.
#[test]
fn greedy_is_unchanged() {
    let data = logits(154_880, 3.0, 7);
    let f: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let want = greedy_pick_last_wins(&f);
    for p in [0.95f32, 1.0] {
        assert_eq!(
            sample_with_params_seeded(&data, &params(0.0, p, 0.0), &[], Some(1)),
            want
        );
    }
}
