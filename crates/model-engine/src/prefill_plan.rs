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
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants:
//! - `plan_chunk_len` returns a length in `1..=proposed` for a non-empty `proposed`,
//!   and returns `proposed` unchanged for a last chunk.

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
