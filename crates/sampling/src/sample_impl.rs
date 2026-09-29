// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The host sampling pipeline over little-endian f32 logits:
//! penalties and logit bias, then either greedy argmax or top-n-sigma,
//! temperature, top-k, min-p, top-p and a multinomial draw.
//!
//! Owner: sampling.
//! Invariants:
//! - `apply_penalties_and_bias` runs before the greedy bypass, so penalties
//!   and bias also apply at temperature 0.

use super::{SamplingParams, apply_penalties_and_bias, record_entropy};

pub fn sample_with_params_history(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
) -> u32 {
    sample_with_params_seeded(data, params, token_history, params.seed)
}

/// 2026-09-25: Samples one token id from the little-endian f32 logits in
/// `data`. `temperature <= 0.0` returns the greedy argmax. Otherwise
/// `Some(seed)` draws from `StdRng::seed_from_u64(seed)` and `None` from
/// `rand::random`.
pub fn sample_with_params_seeded(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
) -> u32 {
    let n = data.len() / 4;
    let top_k = params.top_k as usize;
    let top_p = params.top_p;
    let top_n_sigma = params.top_n_sigma;
    let min_p = params.min_p;

    let mut raw_logits: Vec<f32> = data
        .chunks_exact(4)
        .take(n)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // 2026-09-25: Penalties and logit bias go first, before the greedy
    // bypass, so they apply at temperature 0 too.
    apply_penalties_and_bias(&mut raw_logits, params, token_history);

    // 2026-09-25: Greedy returns the argmax of the penalised logits and skips
    // every later stage. Those stages only drop tokens below the maximum or
    // apply a monotonic scale, so none of them can move the argmax.
    if params.temperature <= 0.0 {
        return greedy_pick_last_wins(&raw_logits);
    }
    let temperature = params.temperature;

    // 2026-09-25: Top-n-sigma, before temperature scaling: logits below
    // mean - n*sigma become -inf.
    if top_n_sigma > 0.0 {
        let sum: f32 = raw_logits.iter().sum();
        let mean = sum / n as f32;
        let var: f32 = raw_logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let sigma = var.sqrt();
        if sigma > 0.0 {
            let threshold = mean - top_n_sigma * sigma;
            for logit in raw_logits.iter_mut() {
                if *logit < threshold {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    let mut logits: Vec<(u32, f32)> = raw_logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (i as u32, v / temperature))
        .collect();

    if logits.is_empty() {
        return raw_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // 2026-09-25: Top-k, min-p and top-p need descending order. With a top-k
    // cap, quickselect isolates the k largest in O(n) and only those k are
    // sorted. With none of the three, nothing is sorted: the distribution of
    // the multinomial draw does not depend on order.
    let cmp_desc =
        |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
    let sorted = if top_k > 0 && top_k < logits.len() {
        logits.select_nth_unstable_by(top_k, cmp_desc);
        logits.truncate(top_k);
        logits.sort_unstable_by(cmp_desc);
        true
    } else if min_p > 0.0 || top_p < 1.0 {
        logits.sort_unstable_by(cmp_desc);
        true
    } else {
        false
    };

    // 2026-09-25: `sorted` means logits[0] is the maximum; otherwise it is
    // reduced for. The min-p and top-p blocks below run only when `sorted` is
    // true, so they can rely on descending order.
    let max_val = if sorted {
        logits[0].1
    } else {
        logits
            .iter()
            .map(|&(_, v)| v)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_val).exp()))
        .collect();

    // 2026-09-25: Entropy H = -Σ p·ln(p), in nats, of the softmax over the
    // candidates left after top-k and before min-p and top-p.
    {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let h: f32 = probs
                .iter()
                .map(|&(_, w)| {
                    let p = w * inv;
                    if p > 1e-10 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            record_entropy(h);
        }
    }

    if min_p > 0.0 {
        let max_prob = probs[0].1;
        let threshold = min_p * max_prob;
        probs.retain(|p| p.1 >= threshold);
    }

    if top_p < 1.0 {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumsum += prob / sum;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    let sum: f32 = probs.iter().map(|p| p.1).sum();
    let random_val: f32 = if let Some(s) = seed {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        rng.random::<f32>()
    } else {
        rand::random::<f32>()
    };
    let threshold = random_val * sum;
    let mut cumsum = 0.0f32;
    for &(idx, prob) in &probs {
        cumsum += prob;
        if cumsum >= threshold {
            return idx;
        }
    }
    probs.last().map_or(0, |p| p.0)
}

/// 2026-09-25: Returns what `max_by(partial_cmp.unwrap_or(Equal))` returns:
/// the last of several equal maxima. The verify path's `argmax_first_wins`
/// takes the first, so the two are not interchangeable on exact ties.
///
/// A NaN compares `Equal`, and under `max_by` an equal later element
/// displaces the current maximum, so a NaN after the maximum changes the
/// result. A max-then-find pass cannot reproduce that. Pass 1 therefore takes
/// the maximum over 8 lanes while detecting NaN. With any NaN present the
/// `max_by` expression itself runs; otherwise the result is the last index
/// equal to the maximum, which is what `max_by` returns on NaN-free input,
/// -0.0/+0.0 ties included.
///
/// 2026-09-29: A144b: `pub`, re-exported from the crate root, as the one
/// tie-break for every other host-side greedy pick that must match what
/// this (decode's) host path would emit at the same position: the server's
/// verify final pick (`verify_pipeline_helper::verify_pick_with_pipeline`)
/// calls it rather than a second last-wins implementation.
pub fn greedy_pick_last_wins(v: &[f32]) -> u32 {
    const LANES: usize = 8;
    let mut acc = [f32::NEG_INFINITY; LANES];
    let mut any_nan = false;
    let mut chunks = v.chunks_exact(LANES);
    for c in &mut chunks {
        for (a, &x) in acc.iter_mut().zip(c) {
            any_nan |= x.is_nan();
            if x > *a {
                *a = x;
            }
        }
    }
    let mut best = f32::NEG_INFINITY;
    for &a in acc.iter() {
        if a > best {
            best = a;
        }
    }
    for &x in chunks.remainder() {
        any_nan |= x.is_nan();
        if x > best {
            best = x;
        }
    }
    if any_nan {
        return v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    v.iter()
        .rposition(|&x| x == best)
        .unwrap_or(0)
        .try_into()
        .unwrap_or(0)
}

#[cfg(test)]
mod greedy_tests {
    use super::greedy_pick_last_wins;

    fn reference(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    fn agree(v: &[f32]) {
        assert_eq!(greedy_pick_last_wins(v), reference(v), "diverged on {v:?}");
    }

    #[test]
    fn matches_max_by_reference() {
        agree(&[]);
        agree(&[1.0]);
        agree(&[1.0, 3.0, 2.0]);
        agree(&[1.0, 5.0, 5.0, 5.0, 2.0]);
        agree(&[5.0, 1.0, 5.0]);
        // 2026-09-25: a tie across the 8-lane boundary (index 7 in a lane,
        // index 8 in the remainder).
        agree(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0]);
        agree(&[-5.0, -1.0, -3.0]);
        agree(&[-0.0, 0.0, -0.0]);
        agree(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        agree(&[f32::INFINITY, 1.0, f32::INFINITY]);
        agree(&[f32::NAN, 1.0, 2.0]);
        agree(&[1.0, 2.0, f32::NAN]);
        agree(&[f32::NAN]);
    }

    #[test]
    fn vocab_sized_last_wins() {
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[100_000] = 999.0;
        v[200_000] = 999.0;
        assert_eq!(greedy_pick_last_wins(&v), reference(&v));
        assert_eq!(greedy_pick_last_wins(&v), 200_000);
    }
}
