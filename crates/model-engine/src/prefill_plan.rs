// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Where prefill chunks end, and which chunk ends carry an SSM snapshot.
//!
//! A later request's prefix match is block-floored, and a snapshot deeper than the
//! match cannot be restored. So a non-last chunk ends on a KV block boundary, and the
//! chunk that would span the tail split point ends there. A last chunk that spans the
//! split point is split by `TransformerModel::prefill_chunk_dispatch` instead, inside
//! one scheduler tick.
//!
//! Before 2026-09-27 only the last chunk was split. A dense serve's first chunk is
//! `prefill_budget + max_batch_size` = 8193 tokens, so every later chunk end sat one
//! token past a block boundary, and a 32772-token prompt's split point (32752) fell
//! inside the second-to-last chunk: no tail snapshot was saved, and the byte-identical
//! warm request restored at 24577 and replayed 8195 tokens (12.5 s instead of ~0.2 s).
//!
//! 2026-10-03: Under `METRALE_PREFIX_GRID_RESTORE=1` (race #69) the chunk ends come from the
//! absolute grid instead (`grid_chunk_len`): every non-last chunk ends at a multiple of `G`,
//! whatever the scheduler's budget. The idle first chunk of `prefill_budget + max_batch_size`
//! tokens (`phase_start_prefills.rs`, `budget = max_batch_tokens` when nothing else runs) is why
//! the budget-derived plan can make an 8193-row single chunk, or, with `max_batch_size >= 16`,
//! a first chunk that is a block longer than the budget, shifting every later chunk off the
//! `G` grid; the grid ignores the budget.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants:
//! - `plan_chunk_len` returns a length in `1..=proposed` for a non-empty `proposed`,
//!   and returns `proposed` unchanged for a last chunk.
//! - `grid_chunk_len(offset, total, g)` ends at `min(next multiple of g above offset, total)`.

/// 2026-10-02: Default minimum rows in the last prefill pass after a tail split
/// (race #69: a 27-row last pass after a 32K history lost the final question).
pub const DEFAULT_MIN_TAIL_ROWS: usize = 256;

/// 2026-10-02: Minimum last-pass rows, `METRALE_PREFIX_MIN_TAIL_ROWS` (default 256).
pub fn min_tail_rows() -> usize {
    std::env::var("METRALE_PREFIX_MIN_TAIL_ROWS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_MIN_TAIL_ROWS)
}

/// 2026-10-03: Whether a `METRALE_GLM_SSM_INPASS_CAPTURE` value asks for the in-pass tail
/// snapshot: `1` only.
pub fn inpass_capture_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-03: `METRALE_GLM_SSM_INPASS_CAPTURE=1` (race #69): instead of splitting the prefill at
/// `tail_split_point`, take the SSM snapshot there inside the one pass, so a prompt runs the
/// cache-off pass sequence with the prefix cache on (`TransformerModel::inpass_ssm_capture_active`
/// adds the per-model condition). Off unless set to `1`. Read once per process.
pub fn inpass_capture_lever() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = inpass_capture_requested(
            std::env::var("METRALE_GLM_SSM_INPASS_CAPTURE")
                .ok()
                .as_deref(),
        );
        if on {
            tracing::warn!(
                "METRALE_GLM_SSM_INPASS_CAPTURE=1 - the prefix-cache tail snapshot is captured \
                 inside the prefill pass; prompts are no longer split at the tail split point"
            );
        }
        on
    })
}

/// 2026-10-03: The split point the dispatcher and the scheduler use: `cut` (the model's
/// `tail_split_point` for this prompt, `None` when it does not split), or `None` when the
/// in-pass capture replaces the split. With the split gone, `plan_chunk_len` and
/// `prefill_chunk_dispatch` follow the cache-off chunk grid exactly.
pub fn effective_split(cut: Option<usize>, inpass_capture: bool) -> Option<usize> {
    if inpass_capture { None } else { cut }
}

/// 2026-10-03: Whether a prefill pass over `[proc_start, proc_start + proc_count)` must capture
/// the in-pass snapshot at `cut`: only when it strictly spans `cut`. A pass that ends at `cut` is
/// a non-last chunk whose end `save_checkpoint` saves (`is_prompt_tail_end`), and one that starts
/// at or after `cut` begins from a state at or past it.
pub fn inpass_capture_spans(cut: usize, proc_start: usize, proc_count: usize) -> bool {
    proc_start < cut && cut < proc_start + proc_count
}

/// 2026-10-03: Default grid size `G` of `METRALE_PREFIX_GRID_RESTORE`
/// (`METRALE_PREFIX_GRID_TOKENS` unset): the serve's 8192-token prefill chunk.
pub const DEFAULT_GRID_TOKENS: usize = 8192;

/// 2026-10-03: Whether a `METRALE_PREFIX_GRID_RESTORE` value asks for the absolute chunk grid:
/// `1` only.
pub fn grid_restore_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-03: The grid size for a `METRALE_PREFIX_GRID_TOKENS` value: a positive integer, else
/// [`DEFAULT_GRID_TOKENS`] (unset, empty, `0` or unparsable).
pub fn grid_tokens_from(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&g| g > 0)
        .unwrap_or(DEFAULT_GRID_TOKENS)
}

/// 2026-10-03: `METRALE_PREFIX_GRID_RESTORE=1` (race #69): `Some(G)` with `G` from
/// `METRALE_PREFIX_GRID_TOKENS` (default 8192), `None` when off. Read once per process. The model
/// adds the per-model conditions and validates `G` (`TransformerModel::prefix_grid_active_bs`).
///
/// With the grid on, every prefill chunk ends at an absolute multiple of `G` or at the prompt end
/// ([`grid_chunk_len`]), SSM snapshots are saved only at those chunk ends, and a prefix-cache hit
/// restores only at one ([`grid_step`]), so the suffix prefill runs the same chunk sequence a cold
/// prefill of the same prompt runs from that point.
pub fn grid_restore_lever() -> Option<usize> {
    static G: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        let on = grid_restore_requested(
            std::env::var("METRALE_PREFIX_GRID_RESTORE")
                .ok()
                .as_deref(),
        );
        if !on {
            return None;
        }
        let g = grid_tokens_from(std::env::var("METRALE_PREFIX_GRID_TOKENS").ok().as_deref());
        tracing::warn!(
            "METRALE_PREFIX_GRID_RESTORE=1 - prefill chunks end on the absolute {g}-token grid, \
             SSM snapshots are saved and restored only at grid points, and the prefix-cache \
             tail split and the in-pass / mid-chunk tail captures are off"
        );
        Some(g)
    })
}

/// 2026-10-03: Whether `g` can be the prefill grid: positive, a multiple of the KV block size
/// (so every grid point is a block boundary a block-floored match can reach) and of 4 (the GDN
/// WY4 rule for intermediate chunks), and no larger than the prefill arena (`arena_cap` tokens),
/// since every non-last chunk is exactly `g` rows.
pub fn grid_valid(g: usize, block_size: usize, arena_cap: usize) -> bool {
    g > 0
        && block_size > 0
        && g.is_multiple_of(block_size)
        && g.is_multiple_of(4)
        && g <= arena_cap
}

/// 2026-10-03: Length of the grid chunk at `offset` of a `total`-token prompt: up to the next
/// multiple of `g` above `offset`, or to `total` when that comes first. A function of
/// `(offset, total, g)` only (not of the scheduler's budget or occupancy), so a prompt is cut
/// into the same chunks however busy the scheduler is, and from offset `k*g` on a warm prefill
/// runs exactly the chunks a cold one runs. `0` at or past `total`, or for `g == 0`.
pub fn grid_chunk_len(offset: usize, total: usize, g: usize) -> usize {
    if g == 0 || offset >= total {
        return 0;
    }
    let next = (offset / g + 1) * g;
    next.min(total) - offset
}

/// 2026-10-03: `plan_chunk_len`, or [`grid_chunk_len`] when `grid` is `Some(G)`: the grid
/// replaces the budget-derived length (`proposed`), the block alignment and the tail split. An
/// empty proposal stays empty. With `grid == None` this is `plan_chunk_len` exactly.
pub fn plan_chunk_len_grid(
    offset: usize,
    total: usize,
    proposed: usize,
    block_size: Option<usize>,
    split: Option<usize>,
    grid: Option<usize>,
) -> usize {
    match grid {
        Some(g) if proposed > 0 && g > 0 => grid_chunk_len(offset, total, g),
        _ => plan_chunk_len(offset, total, proposed, block_size, split),
    }
}

/// 2026-10-03: Whether a non-last chunk ending at `end_token` saves a grid snapshot: a positive
/// exact multiple of both `g` and the KV block size strictly below `total` (a chunk ending at
/// `total` is the last one). No flooring: unlike `is_checkpoint_chunk_end`, which classifies by
/// `end_token / block_size`, an end off the grid never saves.
pub fn is_grid_checkpoint_end(end_token: usize, total: usize, g: usize, block_size: usize) -> bool {
    g > 0
        && block_size > 0
        && end_token > 0
        && end_token < total
        && end_token.is_multiple_of(g)
        && end_token.is_multiple_of(block_size)
}

/// 2026-10-03: The deepest grid point a hit on a `total`-token prompt may restore, given a
/// `matched`-token radix match: the largest multiple of `g` that is at most `matched` and
/// strictly below `total` (at least one row is always computed, so the last chunk produces the
/// logits as in a cold prefill). `0` when there is none.
pub fn grid_restore_cap(matched: usize, total: usize, g: usize) -> usize {
    if g == 0 || total == 0 {
        return 0;
    }
    matched.min(total - 1) / g * g
}

/// 2026-10-03: The prompt length the grid prefix-cache insert covers: the largest multiple of `g`
/// strictly below `total`, where the prompt's last chunk starts. Every block below it was written
/// by a full `g`-row non-last chunk, the pass a cold prefill of any prompt sharing that prefix
/// runs over it; the last chunk's blocks are not inserted (`finalize_last`).
pub fn grid_insert_len(total: usize, g: usize) -> usize {
    grid_restore_cap(total, total, g)
}

/// 2026-10-03: One step of the grid restore-point search (`prefill_b/grid_restore.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridStep {
    /// 2026-10-03: The match ends at a grid point and its deepest snapshot is exactly there:
    /// keep the match and let the rank vote restore that snapshot.
    Accept,
    /// 2026-10-03: Release the match and look up `tokens[..n]` again (`n` a grid point below the
    /// current match), which matches at most `n` tokens and reports the deepest snapshot at or
    /// below `n`.
    Probe(usize),
    /// 2026-10-03: No grid point can be restored: release the match and prefill cold.
    Miss,
}

/// 2026-10-03: Decide from a lookup that matched `matched` tokens and whose deepest snapshot is
/// `snap` tokens deep (`0` for none) on a `total`-token prompt. Every `Probe(n)` has `n` below
/// `matched`, and a probe's lookup matches at most `n`, so successive probes strictly decrease
/// and the search ends.
pub fn grid_step(matched: usize, snap: usize, total: usize, g: usize) -> GridStep {
    let cap = grid_restore_cap(matched, total, g);
    if cap == 0 {
        return GridStep::Miss;
    }
    if matched == cap && snap == cap {
        return GridStep::Accept;
    }
    if snap >= cap {
        // 2026-10-03: The deepest snapshot is at or past the cap (off the grid, or at a point
        // the restore may not use); one at the cap itself may still exist. `snap > matched`
        // never comes out of a lookup, so `matched > cap` here.
        return if matched > cap {
            GridStep::Probe(cap)
        } else {
            GridStep::Miss
        };
    }
    let below = snap / g * g;
    if below == 0 {
        GridStep::Miss
    } else {
        GridStep::Probe(below)
    }
}

/// 2026-10-03: Upper bound on [`grid_search`] steps; each probe is at least one grid point
/// shallower, so only a lookup that broke its contract reaches it.
pub const GRID_SEARCH_MAX_STEPS: usize = 64;

/// 2026-10-03: The grid restore-point search: starting from a lookup that matched `matched`
/// tokens with its deepest snapshot `snap` tokens deep, apply [`grid_step`] until it accepts or
/// misses. `relookup(n)` releases the current match and, for `n > 0`, looks up the first `n`
/// prompt tokens and returns that lookup's `(matched, snap)`; `relookup(0)` only releases (the
/// caller then holds an empty match). Returns the accepted grid point, with the caller holding a
/// match of exactly that many tokens, or `0` after a miss, with the caller holding nothing.
pub fn grid_search(
    matched: usize,
    snap: usize,
    total: usize,
    g: usize,
    mut relookup: impl FnMut(usize) -> (usize, usize),
) -> usize {
    let (mut matched, mut snap) = (matched, snap);
    for _ in 0..GRID_SEARCH_MAX_STEPS {
        match grid_step(matched, snap, total, g) {
            GridStep::Accept => return matched,
            GridStep::Probe(n) => (matched, snap) = relookup(n),
            GridStep::Miss => {
                relookup(0);
                return 0;
            }
        }
    }
    relookup(0);
    0
}

/// 2026-10-02: The tail split point of a `total`-token prompt with the env minimum tail.
pub fn tail_split_point(total: usize, block_size: usize) -> Option<usize> {
    tail_split_point_min(total, block_size, min_tail_rows())
}

/// 2026-10-02: The tail split point: the largest block-aligned position that leaves at
/// least `min_tail` rows in the last pass, and is never above the old point (one block
/// below the last block boundary strictly under `total`, so a warm next turn's
/// block-floored match still restores it). `None` when that is not above token 0, i.e.
/// the prompt is too short to split without a short last pass.
pub fn tail_split_point_min(total: usize, block_size: usize, min_tail: usize) -> Option<usize> {
    let tb = metrale_gpu_runtime::ssm_tail_boundary(total, block_size)?;
    let old = tb - block_size;
    let by_min = total.checked_sub(min_tail)? / block_size * block_size;
    let cut = old.min(by_min);
    (cut > 0).then_some(cut)
}

/// 2026-09-27: Length of the prefill chunk at `offset` of a `total`-token prompt, given
/// the capped `proposed` length.
///
/// A last chunk (`offset + proposed >= total`) is returned unchanged. A non-last chunk
/// ends at the last `block_size` boundary inside it (when that is past `offset`), then
/// at `split` when it would still span it, and is finally rounded down to a multiple of
/// 4 (the GDN WY4 rule for intermediate chunks). With chunk ends on block boundaries
/// every later offset is block-aligned, so the rounding changes nothing when
/// `block_size` is a multiple of 4.
pub fn plan_chunk_len(
    offset: usize,
    total: usize,
    proposed: usize,
    block_size: Option<usize>,
    split: Option<usize>,
) -> usize {
    if proposed == 0 || offset + proposed >= total {
        return proposed;
    }
    let mut end = offset + proposed;
    if let Some(bs) = block_size.filter(|&bs| bs > 0) {
        let aligned = (end / bs) * bs;
        if aligned > offset {
            end = aligned;
        }
    }
    if let Some(cut) = split
        && offset < cut
        && cut < end
    {
        end = cut;
    }
    let len = end - offset;
    if len >= 4 { (len / 4) * 4 } else { len }
}

/// 2026-09-27: Whether a non-last chunk ending at `end_token` saves an SSM snapshot: a
/// prompt-tail end (the last block boundary under `total`, or one block below it), or a
/// block boundary whose block index is a multiple of `interval` (0 turns interval
/// snapshots off). Block 0 never does.
pub fn is_checkpoint_chunk_end(
    end_token: usize,
    total: usize,
    block_size: usize,
    interval: usize,
) -> bool {
    if block_size == 0 {
        return false;
    }
    let end_block = end_token / block_size;
    let on_interval = interval > 0 && end_block.is_multiple_of(interval);
    end_block != 0 && (is_prompt_tail_end(end_token, total, block_size) || on_interval)
}

/// 2026-09-27: Whether a chunk end is a prompt-tail checkpoint: the last block boundary
/// under `total`, or one block below it (`tail_split_point`). A warm next turn's
/// block-floored match lands at one of the two.
pub fn is_prompt_tail_end(end_token: usize, total: usize, block_size: usize) -> bool {
    if block_size == 0 {
        return false;
    }
    let tail = (total.saturating_sub(1) / block_size) * block_size;
    end_token == tail
        || (tail >= block_size && end_token == tail - block_size)
        || tail_split_point(total, block_size) == Some(end_token)
}

#[cfg(test)]
#[path = "prefill_plan_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "prefill_plan_grid_tests.rs"]
mod grid_tests;
