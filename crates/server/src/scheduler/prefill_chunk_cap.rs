// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `METRALE_PREFILL_CHUNK_WHILE_DECODING=<tokens>` (race-decode, SeedHQ/seed-skills#80):
//! while another sequence is decoding, cap every prefill chunk at the lever, so the decoders
//! wait one capped chunk per tick instead of a whole `--max-prefill-tokens` chunk. On a
//! multi-rank serve (`Model::is_ep`) the fused mixed step is off (`phase_continue_prefills`), so
//! a tick runs StartPrefills, one ContinuePrefills chunk, then Decode, and the decoders'
//! inter-token gap is the prefill chunk's wall time (~12-13 s per 8192-token GLM-5.3 chunk).
//!
//! Rank agreement: only rank 0 runs the scheduler. The chunk length reaches the worker in the
//! `0xFFFFFFF0` broadcast (`run_standard.rs`, `prefill_a_step.rs`; worker arm in model-engine
//! `model/impl_a2.rs`), so both ranks run every chunk at the same length by construction and the
//! lever is never read on a worker (it is not a rank-agreed scalar).
//!
//! Owner: scheduler.
//! Invariants:
//! - Unset, empty, `0` or unparsable: off; every plan is the uncapped one.
//! - On: the cap is a positive multiple of [`GRANULE`], and at plan time of the KV block size.
//! - The cap only shortens a chunk, and only while a decoder is active.
//! - Under `METRALE_PREFIX_GRID_RESTORE` a capped chunk never crosses a grid point: every grid
//!   point is still a chunk end, so the grid snapshots are saved where they always were.

use metrale_model_engine::prefill_plan::plan_chunk_len_grid;

/// 2026-10-03: The lever's environment name.
pub(crate) const LEVER: &str = "METRALE_PREFILL_CHUNK_WHILE_DECODING";

/// 2026-10-03: The cap's granule and minimum, in tokens. 256 is the race stack's GLM attention
/// sub-chunk (`METRALE_GLM_PREFILL_ROWS=256`), twice the MoE GEMM tile M (128), a multiple of
/// the KV block (16) and of 4 (the GDN/KDA WY4 rule for intermediate chunks). Any chunk length
/// is valid on the GLM prefill path; the granule keeps chunk ends on absolute multiples of the
/// sub-chunk width, so the attention sub-chunks see the same rows as uncapped chunks.
pub(crate) const GRANULE: usize = 256;

/// 2026-10-03: The lever from its raw value: `None` (off) for unset, empty or `0`, and for an
/// unparsable value (with a warning); otherwise the value rounded down to a multiple of
/// [`GRANULE`], at least [`GRANULE`] (with a warning when that changed it).
pub(crate) fn resolve(raw: Option<&str>) -> Option<usize> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    let Ok(v) = s.parse::<usize>() else {
        tracing::warn!("{LEVER}={s:?} is not a token count; the prefill chunk cap stays off");
        return None;
    };
    if v == 0 {
        return None;
    }
    let r = round_to(v, GRANULE);
    if r != v {
        tracing::warn!(
            "{LEVER}={v} rounded to {r}: the cap is a multiple of {GRANULE} tokens, at least \
             {GRANULE}"
        );
    }
    tracing::warn!(
        "{LEVER}={r}: a prefill chunk planned while another sequence decodes is capped at {r} \
         tokens"
    );
    Some(r)
}

/// 2026-10-03: `v` rounded down to a multiple of `g`, at least `g` (`g > 0`).
fn round_to(v: usize, g: usize) -> usize {
    (v / g * g).max(g)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// 2026-10-03: The cap aligned to the KV block size too: rounded down to a multiple of
/// `lcm(GRANULE, block_size)`, at least that. With the default 16-token block it is `cap`.
pub(crate) fn aligned_cap(cap: usize, block_size: Option<usize>) -> usize {
    let g = match block_size.filter(|&b| b > 0) {
        Some(b) => GRANULE / gcd(GRANULE, b) * b,
        None => GRANULE,
    };
    round_to(cap, g)
}

/// 2026-10-03: The cap in force for a chunk planned now: the aligned lever when it is on and
/// `decoders` (other sequences in the decode phase) is at least one, else `None`.
pub(crate) fn active_cap(
    lever: Option<usize>,
    decoders: usize,
    block_size: Option<usize>,
) -> Option<usize> {
    if decoders == 0 {
        return None;
    }
    lever.map(|c| aligned_cap(c, block_size))
}

/// 2026-10-03: Sequences in the decode phase: `active` entries that have not finished.
pub(crate) fn decoders(active: &[super::types::ActiveSeq]) -> usize {
    active.iter().filter(|a| !a.finished).count()
}

/// 2026-10-03: The chunk at `offset` of a `total`-token prompt, as `(capped, uncapped)`.
/// `uncapped` is `plan_chunk_len_grid(offset, total, proposed, ..)`, the plan without the lever.
/// With `cap = Some(c)` (from [`active_cap`]) and `uncapped > c`, `capped` is the plan of a
/// `c`-token proposal (so it ends on a block boundary, or at the tail split point when it would
/// span it, exactly as an uncapped chunk does); under the grid (which ignores the proposal) it is
/// `c`, which stays below the next grid point. Otherwise `capped == uncapped`.
pub(crate) fn plan_capped(
    offset: usize,
    total: usize,
    proposed: usize,
    block_size: Option<usize>,
    split: Option<usize>,
    grid: Option<usize>,
    cap: Option<usize>,
) -> (usize, usize) {
    let uncapped = plan_chunk_len_grid(offset, total, proposed, block_size, split, grid);
    let capped = match cap {
        Some(c) if uncapped > c && grid.is_some() => c,
        Some(c) if uncapped > c => {
            plan_chunk_len_grid(offset, total, c, block_size, split, grid).min(uncapped)
        }
        _ => uncapped,
    };
    (capped, uncapped)
}

/// 2026-10-03: Log, once per prefill (`logged`), that a chunk was capped.
pub(crate) fn log_capped(logged: &mut bool, len: usize, decoders: usize) {
    if !*logged {
        tracing::info!("prefill chunk capped: {len} (decoders active={decoders})");
        *logged = true;
    }
}

#[cfg(test)]
#[path = "prefill_chunk_cap_tests.rs"]
mod tests;
