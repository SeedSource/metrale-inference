// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Chunked MTP hidden capture for GLM-5.3 (`METRALE_GLM_MTP_CHUNKED_CAPTURE`),
//! the A59 fix: the prompt-hidden capture shrinks from `max_seq_len` rows (4 GiB at 524K
//! for hidden 4096) to one arena's worth of rows, `BufferArena::max_batch_tokens`.
//!
//! The capture becomes a staging window: rows `[stage_start, captured)` of the prompt sit at
//! staging rows `0..captured - stage_start`. After every non-last `prefill_chunk` the window is
//! drained into the drafter (`catchup_drafter` at `row_base = stage_start`), so each chunk's
//! rows go to staging row 0 again. The last chunk's rows stay staged until the first propose,
//! which drains the rest, because the caller still samples from the shared forward buffers
//! after the last chunk. Pair key `k` is `(tokens[k + 1], hidden_k)` either way, so the
//! drafter rows match the whole-prompt `prefill_drafter` row for row.
//!
//! A chunk that would overflow the window, or a drafter that is not at `stage_start`, leaves
//! the capture short: the drafter then has fewer rows, which costs acceptance, never
//! correctness (the target verifies every draft).
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - These functions are pure; the callers own the atomics and the device copies.

/// 2026-10-01: Whether the chunked capture is on for `model_type`: `glm5_next` with
/// `METRALE_GLM_MTP_CHUNKED_CAPTURE=1`, and not with the drafter carry
/// (`mtp_carry_drafter_enabled`), which addresses the capture by absolute row. Read once.
pub(crate) fn glm_mtp_chunked_capture(model_type: &str, carry_on: bool) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = || std::env::var("METRALE_GLM_MTP_CHUNKED_CAPTURE").as_deref() == Ok("1");
    model_type == "glm5_next" && !carry_on && *ON.get_or_init(on)
}

/// 2026-10-01: Staging row at which a chunk of `proc_count` rows starting at prompt row
/// `chunk_start` is written, or `None` when it starts before the window or overflows its
/// `capacity` rows.
pub(crate) fn stage_offset(
    chunk_start: usize,
    proc_count: usize,
    stage_start: usize,
    capacity: usize,
) -> Option<usize> {
    let off = chunk_start.checked_sub(stage_start)?;
    (off + proc_count <= capacity).then_some(off)
}

/// 2026-10-01: The token range `[lo, hi)` to hand `catchup_drafter` at `row_base = lo` so it
/// drains the staged rows, or `None` when there is nothing to drain.
///
/// - `prompt_len == None`: after a non-last chunk. Every staged key `stage_start..captured`
///   is drained, which needs `tokens[captured]`, so `captured < tokens_len`.
/// - `prompt_len == Some(p)`: the first propose. The capture must cover the prompt; keys
///   `stage_start..p - 1` are drained (key `p - 1` pairs with the first sampled token, which
///   the propose itself writes), as `prefill_drafter(&tokens[..p])` would.
///
/// The drafter must be exactly at `stage_start`; `catchup_drafter` writes nothing otherwise.
pub(crate) fn drain_range(
    stage_start: usize,
    captured: usize,
    tokens_len: usize,
    drafter_rows: usize,
    prompt_len: Option<usize>,
) -> Option<(usize, usize)> {
    if drafter_rows != stage_start || captured <= stage_start {
        return None;
    }
    let hi = match prompt_len {
        None if captured < tokens_len => captured + 1,
        Some(p) if captured >= p && p <= tokens_len => p,
        _ => return None,
    };
    // 2026-10-01: `rows_impl` needs two tokens for one row.
    (hi >= stage_start + 2).then_some((stage_start, hi))
}

/// 2026-10-01: Bytes the GLM MTP head allocates after the KV pool is sized, under the
/// chunked capture, reserved before sizing (A59): the staging capture (`stage_rows` x
/// `hidden` BF16), the drafter's private KV pool (`max_seq_len / 16 + 2` blocks of 16 FP8
/// latents of `kv_lora_rank`, twice that without the V alias) and one drafter indexer cache
/// (`index_head_dim` BF16 twice plus a validity byte per row). `max_seq_len` bounds the
/// drafter's rows, so this can only over-reserve.
pub(crate) fn glm_mtp_reserve_bytes(
    hidden: usize,
    kv_lora_rank: usize,
    index_head_dim: usize,
    max_seq_len: usize,
    stage_rows: usize,
    v_alias: bool,
) -> usize {
    let capture = stage_rows.min(max_seq_len) * hidden * 2;
    let sides = if v_alias { 1 } else { 2 };
    let pool = (max_seq_len / 16 + 2) * 16 * kv_lora_rank * sides;
    let indexer = max_seq_len * (index_head_dim * 4 + 1);
    capture + pool + indexer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-01: A drafter model: appends `tokens.len() - 1` pairs `(tokens[r + 1], hidden
    /// row)` when `row_base` equals its length, as `rows_impl` does.
    struct Drafter(Vec<(u32, usize)>);

    impl Drafter {
        fn catchup(&mut self, tokens: &[u32], hidden_rows: &[usize], row_base: usize) {
            if self.0.len() != row_base || tokens.len() < 2 {
                return;
            }
            let pairs = tokens[1..].iter().copied().zip(hidden_rows.iter().copied());
            self.0.extend(pairs);
        }
    }

    /// 2026-10-01: Runs a chunked prefill of `p` tokens in chunks of `chunk` rows through
    /// a staging window of `cap` rows and the first-propose drain; returns the drafter rows.
    fn chunked(p: usize, chunk: usize, cap: usize) -> Vec<(u32, usize)> {
        let tokens: Vec<u32> = (0..p as u32).map(|t| 1000 + t).collect();
        let mut staging = vec![usize::MAX; cap];
        let (mut stage_start, mut captured) = (0usize, 0usize);
        let mut d = Drafter(Vec::new());
        let mut s = 0;
        while s < p {
            let n = chunk.min(p - s);
            let off = stage_offset(s, n, stage_start, cap).filter(|_| s == captured);
            if let Some(off) = off {
                for (i, row) in staging[off..off + n].iter_mut().enumerate() {
                    *row = s + i;
                }
                captured = s + n;
            }
            let drain = drain_range(stage_start, captured, p, d.0.len(), None);
            if let Some((lo, hi)) = drain.filter(|_| s + n < p) {
                d.catchup(&tokens[lo..hi], &staging, lo);
                stage_start = captured;
            }
            s += n;
        }
        if let Some((lo, hi)) = drain_range(stage_start, captured, p, d.0.len(), Some(p)) {
            d.catchup(&tokens[lo..hi], &staging, lo);
        }
        d.0
    }

    fn whole(p: usize) -> Vec<(u32, usize)> {
        (0..p - 1).map(|r| (1000 + r as u32 + 1, r)).collect()
    }

    #[test]
    fn chunked_drafter_rows_match_the_whole_prompt_prefill() {
        for &(p, chunk) in &[(10, 4), (9, 3), (8192, 2048), (5, 8), (2, 1), (17, 16)] {
            assert_eq!(chunked(p, chunk, chunk), whole(p), "p={p} chunk={chunk}");
        }
    }

    #[test]
    fn overflow_leaves_the_drafter_short_never_wrong() {
        // 2026-10-01: A window narrower than a chunk captures nothing; the drafter
        // has no rows rather than misaligned ones.
        assert!(chunked(10, 4, 3).is_empty());
    }

    #[test]
    fn stage_offset_bounds() {
        assert_eq!(stage_offset(8, 4, 8, 4), Some(0));
        assert_eq!(stage_offset(8, 5, 8, 4), None);
        assert_eq!(stage_offset(4, 2, 8, 4), None);
        assert_eq!(stage_offset(10, 2, 8, 4), Some(2));
    }

    #[test]
    fn drain_range_needs_the_next_token_and_an_aligned_drafter() {
        assert_eq!(drain_range(0, 4, 10, 0, None), Some((0, 5)));
        assert_eq!(drain_range(0, 10, 10, 0, None), None);
        assert_eq!(drain_range(4, 8, 10, 3, None), None);
        assert_eq!(drain_range(4, 10, 10, 4, Some(10)), Some((4, 10)));
        assert_eq!(drain_range(4, 9, 10, 4, Some(10)), None);
        assert_eq!(drain_range(9, 10, 10, 9, Some(10)), None);
    }

    #[test]
    fn reserve_at_524k_is_a_fraction_of_the_full_capture() {
        let full_capture = 524_288usize * 4096 * 2;
        let r = glm_mtp_reserve_bytes(4096, 512, 128, 524_288, 8192, true);
        // 2026-10-01: 64 MiB staging + 256 MiB pool + ~257 MiB indexer.
        assert_eq!(r, 8192 * 8192 + (32_768 + 2) * 16 * 512 + 524_288 * 513);
        assert!(r < full_capture / 6);
    }

    #[test]
    fn lever_is_glm_only_and_off_with_the_carry() {
        assert!(!glm_mtp_chunked_capture("deepseek_v3", false));
        assert!(!glm_mtp_chunked_capture("glm5_next", true));
    }
}
