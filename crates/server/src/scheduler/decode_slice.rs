// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_DECODE_STEPS_PER_PREFILL_CHUNK=<k>` (race-prefill, SeedHQ/seed-skills#68):
//! while another sequence decodes, follow every tick that ran a prefill chunk with `k - 1`
//! decode-only ticks (no admission, no ContinuePrefills), so the decoders take `k` steps per
//! prefill chunk instead of one. On a multi-rank serve the fused mixed step is off
//! (`run_standard.rs` `can_mix`), so a tick is one prefill chunk then one decode step, and the
//! decoders' inter-step gap is the chunk's wall time. Measured on the c29 car (RULER 128K, C=4,
//! r128car3 2026-10-14): a 1024-token chunk (`METRALE_PREFILL_CHUNK_WHILE_DECODING=1024`) took
//! ~0.87 s at any context depth (~0.5 s of it fixed per chunk), a decode step ~71 ms, so decode
//! ran ~2.6 tok/s while another user prefilled vs ~30 tok/s otherwise. With `k` steps per chunk
//! a larger chunk amortises the fixed part: TPOT ~ (chunk + k x step) / (k x tokens per step).
//!
//! Rank agreement: only rank 0 runs the scheduler; a held tick simply sends no prefill command,
//! so the worker sees the same decode/verify commands it always does.
//!
//! Owner: scheduler.
//! Invariants:
//! - Unset, empty, `0`, `1` or unparsable: off; every tick is planned exactly as without it.
//! - With no decoder active nothing is held, so a single-user serve is unchanged.
//! - A held tick admits nothing (new requests return to the front of the pending queue in
//!   arrival order) and runs no prefill chunk; the Decode lane runs as usual.
//! - The hold only follows a tick that ran a prefill chunk, and lasts at most `k - 1` ticks.

/// 2026-10-14: The lever's environment name.
pub(crate) const LEVER: &str = "METRALE_DECODE_STEPS_PER_PREFILL_CHUNK";

/// 2026-10-14: The lever from its raw value: `None` (off) for unset, empty, `0`, `1` (one decode
/// step per chunk is the behaviour without the lever) and unparsable values (with a warning);
/// otherwise `Some(k)` with `k >= 2`.
pub(crate) fn resolve(raw: Option<&str>) -> Option<usize> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    let Ok(k) = s.parse::<usize>() else {
        tracing::warn!("{LEVER}={s:?} is not a step count; decode slicing stays off");
        return None;
    };
    if k <= 1 {
        return None;
    }
    tracing::warn!(
        "{LEVER}={k}: while a sequence decodes, each prefill chunk is followed by {} decode-only \
         ticks",
        k - 1
    );
    Some(k)
}

/// 2026-10-14: Whether this tick is held (decode only): the lever is on, a decoder is active and
/// the hold left by earlier ticks is not used up.
pub(crate) fn hold(lever: Option<usize>, decoders: usize, left: usize) -> bool {
    lever.is_some() && decoders > 0 && left > 0
}

/// 2026-10-14: The hold left for the ticks after this one. `prefill_ran`: this tick ran a prefill
/// chunk (a chunk 0 in StartPrefills or a continued chunk); `held`: this tick was held.
/// Off, or no decoder active: 0. A chunk ran: `k - 1`. A held tick: one less. Otherwise unchanged.
pub(crate) fn next_left(
    lever: Option<usize>,
    decoders: usize,
    left: usize,
    held: bool,
    prefill_ran: bool,
) -> usize {
    let Some(k) = lever else { return 0 };
    if decoders == 0 {
        0
    } else if prefill_ran {
        k - 1
    } else if held {
        left.saturating_sub(1)
    } else {
        left
    }
}

#[cfg(test)]
#[path = "decode_slice_tests.rs"]
mod tests;
