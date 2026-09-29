// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: the verify path's first-index-wins argmax.
//! 2026-09-29: and (A144b) the last-index-wins greedy pick its final pick
//! shares with decode's host path.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

/// 2026-09-25: index of the first maximum under strict `>`: NaN never wins,
/// and empty or all-NaN input returns 0. -0.0 and +0.0 compare equal, so
/// the first of them wins. Not the sampler's greedy tie-break, which takes
/// the last maximum (`greedy_pick_last_wins`).
///
/// `metrale_sampling::argmax_first_wins_f32` computes it in two passes (the
/// maximum over 8 lanes, then the first index equal to it) so the scan
/// vectorises; the tests below check it against the one-loop form.
pub(super) fn argmax_first_wins(logits: &[f32]) -> u32 {
    metrale_sampling::argmax_first_wins_f32(logits)
}

/// 2026-09-29: A144b: the verify host path's final pick, the pick decode's
/// own host pipeline would make at this position, must use decode's tie
/// rule: last-index-wins (`metrale_sampling::greedy_pick_last_wins`), not
/// [`argmax_first_wins`] above.
///
/// Decode's single-row host greedy path (`sample_with_params_history` ->
/// `greedy_pick_last_wins`) and the verify host path process the same
/// penalised, biased and masked logits at the same FP32 precision (both
/// dequantise BF16 to F32 with `bf16_to_f32`), so exact ties are common on
/// quantised checkpoints. Verify was picking the first tied index and decode
/// the last: a tie-break divergence, not a precision or masking defect. This
/// wrapper delegates to the function decode uses.
pub(super) fn greedy_pick_last_wins(logits: &[f32]) -> u32 {
    metrale_sampling::greedy_pick_last_wins(logits)
}

#[cfg(test)]
mod argmax_tests {
    use super::argmax_first_wins;

    /// 2026-09-25: the one-loop reference form.
    fn reference(logits: &[f32]) -> u32 {
        let mut best_id: u32 = 0;
        let mut best_val: f32 = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i as u32;
            }
        }
        best_id
    }

    fn agree(v: &[f32]) {
        assert_eq!(
            argmax_first_wins(v),
            reference(v),
            "diverged from the original loop on {v:?}"
        );
    }

    #[test]
    fn matches_reference_on_edge_cases() {
        agree(&[]);
        agree(&[1.0]);
        agree(&[1.0, 2.0, 3.0]);
        agree(&[3.0, 2.0, 1.0]);
        agree(&[1.0, 5.0, 5.0, 5.0, 2.0]);
        // 2026-09-25: ties across the 8-lane chunk boundary and in the
        // remainder tail.
        agree(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0]);
        agree(&[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]);
        agree(&[-5.0, -1.0, -3.0]);
        agree(&[-0.0, 0.0]);
        agree(&[0.0, -0.0]);
        agree(&[-1.0, -0.0, 0.0, -1.0]);
        agree(&[f32::NAN, 1.0, 2.0]);
        agree(&[1.0, f32::NAN, 2.0]);
        agree(&[1.0, 2.0, f32::NAN]);
        agree(&[f32::NAN, f32::NAN]);
        agree(&[f32::NEG_INFINITY, -1.0]);
        agree(&[f32::INFINITY, 1.0]);
        agree(&[1.0, f32::INFINITY, f32::INFINITY]);
        agree(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    }

    #[test]
    fn matches_reference_on_vocab_sized_input() {
        // 2026-09-25: deterministic pseudo-random vocab-sized input with a
        // duplicated maximum.
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[123_457] = 999.0;
        v[200_003] = 999.0;
        assert_eq!(argmax_first_wins(&v), reference(&v));
        assert_eq!(argmax_first_wins(&v), 123_457);
    }
}

#[cfg(test)]
mod greedy_pick_last_wins_tests {
    use super::{argmax_first_wins, greedy_pick_last_wins};

    /// 2026-09-29: the tie rule decode's host path has always used:
    /// `max_by(partial_cmp.unwrap_or(Equal))`, last wins.
    fn decode_reference(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// 2026-09-29: A144b: on an exact tie, verify's final pick agrees with
    /// decode's host greedy pick.
    #[test]
    fn exact_tie_matches_decodes_host_pick() {
        let v = [1.0f32, 5.0, 5.0, 5.0, 2.0];
        assert_eq!(greedy_pick_last_wins(&v), decode_reference(&v));
        assert_eq!(
            greedy_pick_last_wins(&v),
            3,
            "the last of the tied indices (1, 2, 3) must win"
        );
    }

    /// 2026-09-29: the two functions must disagree on a real tie, or the
    /// A144b fix is a no-op. If they start agreeing, one of them changed its
    /// tie-break.
    #[test]
    fn diverges_from_first_wins_on_a_real_tie() {
        let v = [1.0f32, 5.0, 5.0, 5.0, 2.0];
        assert_eq!(argmax_first_wins(&v), 1, "first tied index");
        assert_eq!(greedy_pick_last_wins(&v), 3, "last tied index");
        assert_ne!(argmax_first_wins(&v), greedy_pick_last_wins(&v));
    }

    #[test]
    fn vocab_sized_last_wins_matches_decode() {
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[100_000] = 999.0;
        // 2026-09-29: a duplicated maximum: the last must win, as in decode.
        v[200_000] = 999.0;
        assert_eq!(greedy_pick_last_wins(&v), decode_reference(&v));
        assert_eq!(greedy_pick_last_wins(&v), 200_000);
    }
}
