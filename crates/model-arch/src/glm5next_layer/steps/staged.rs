// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The staged prefill (`METRALE_GLM_PREFILL_STAGED=1`): one layer's prompt chunk as
//! an attention pass over `prefill_rows()`-row sub-chunks, then an FFN pass over windows of up to
//! `prefill_rows_ffn()` rows, so the routed-MoE grouped GEMM sees wide M while every attention
//! launch keeps the width it has unstaged.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The attention pass visits exactly the sub-chunks the unstaged loop visits (`sub_chunks`).
//! - Every FFN window is a run of whole consecutive sub-chunks (`ffn_windows`), and inside a
//!   window the MLP's dense GEMMs run one sub-chunk at a time (`dense_slice = prefill_rows()`),
//!   so each dense GEMM sees the rows and M it sees unstaged.
//! - A sub-chunk joins a multi-sub-chunk window only when the routed MoE would already take the
//!   grouped GEMM for it alone (`grouped_prefill_selected`), so no row changes routed path.
//!
//! Why this is byte-identical to the unstaged loop (the `forward_k` sequence per sub-chunk):
//! - Unstaged, sub-chunk `j`'s FFN half runs before sub-chunk `j + 1`'s attention half. The FFN
//!   half writes only its own rows' highway slots, `post`/`comb` slots and `hidden` rows, plus
//!   scratch (`norm_output`, `moe_output`, the MLP workspace, the FFN site's `mix`) that the
//!   attention half writes before it reads and never carries across calls. The attention half
//!   reads its own rows' slots (written by the previous layer), its mixer state and KV (written
//!   only by earlier attention halves). So the attention of `j + 1` reads nothing the FFN of `j`
//!   wrote, and moving every FFN half after every attention half changes no input.
//! - The FFN half of row `t` reads highway slot `t` and the weights. `hc_pre`, the norm,
//!   `hc_post`, `hc_head_mean`, the router top-k, the SwiGLU and the combine are one block (or
//!   element) per row. The routed grouped GEMM computes each output element from its own row
//!   and the expert weight in a fixed k order (no split-K); the sort's order among an
//!   expert's rows and the M-tile a row lands in do not enter the sum. The router and shared
//!   expert GEMMs are M-dependent (cuBLASLt picks its algorithm per shape), which is why they
//!   run per sub-chunk. The all-reduce is elementwise across the two ranks.

use super::*;
use crate::glm5next_mlp::forward_prefill_gemm::grouped_prefill_selected;

/// 2026-09-29: The unstaged loop's sub-chunks of a `num_tokens`-token chunk at width `rows`,
/// as `(first token, rows)`.
pub fn sub_chunks(num_tokens: usize, rows: usize) -> Vec<(usize, usize)> {
    let rows = rows.max(1);
    (0..num_tokens)
        .step_by(rows)
        .map(|t| (t, rows.min(num_tokens - t)))
        .collect()
}

/// 2026-09-29: The FFN windows over `subs` (consecutive, as from `sub_chunks`), each at most
/// `max_rows` rows (a single sub-chunk wider than that stays whole). A sub-chunk for which
/// `mergeable(rows)` is false gets a window of its own, and so does any sub-chunk narrower
/// than the first one (the tail), so a window's inner slices at the first sub-chunk's width
/// line up with its sub-chunks.
///
/// 2026-10-01: With `merge_tail`, a mergeable tail may join the window before it. Its slices
/// still line up: every earlier sub-chunk in that window is full width, so `row_slices` at the
/// first sub-chunk's width ends on exactly the tail's rows.
pub fn ffn_windows(
    subs: &[(usize, usize)],
    max_rows: usize,
    merge_tail: bool,
    mergeable: impl Fn(usize) -> bool,
) -> Vec<(usize, usize)> {
    let width = subs.first().map_or(0, |s| s.1);
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(subs.len());
    let mut open = false;
    for (i, &(t, k)) in subs.iter().enumerate() {
        let tail = merge_tail && i + 1 == subs.len() && k < width;
        let joins = (k == width || tail) && mergeable(k);
        match out.last_mut() {
            Some(last) if open && joins && last.1 + k <= max_rows && last.0 + last.1 == t => {
                last.1 += k;
            }
            _ => out.push((t, k)),
        }
        open = joins;
    }
    out
}

impl Glm5NextLayer {
    /// 2026-09-29: Whether a sub-chunk of `rows` rows may share an FFN window with its
    /// neighbours: always for a dense MLP (its GEMMs run per sub-chunk inside the window), and
    /// for a routed MoE only when `forward_moe` would take the grouped GEMM for it alone.
    fn ffn_mergeable(&self, rows: usize) -> bool {
        match &self.mlp {
            Glm5NextMlpSite::Dense(_) => true,
            Glm5NextMlpSite::Moe(_) => {
                grouped_prefill_selected(&self.mlp_kernels, &self.mlp_cfg, &self.mlp_ws, rows)
            }
        }
    }

    /// 2026-09-29: One prefill chunk of `num_tokens` tokens, staged: `attn_half` over each
    /// `sub_chunks(num_tokens, rows)` sub-chunk, then `ffn_half` over each `ffn_windows` window
    /// of up to `rows_ffn` rows with dense slices of `rows`. See the module notes for why the
    /// result equals the unstaged loop's.
    ///
    /// 2026-10-01: With `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1` (`full_width_attn`), the
    /// attention pass instead calls `attn_half` once per `sub_chunks(num_tokens, rows_ffn)`
    /// window with the DSA core at `rows` rows, and the FFN pass hands each window's MLP a
    /// dense slice as wide as the window. NOT byte-identical to the sliced pass: every dense
    /// projection runs at another M (see `levers::prefill_fullwidth_gemm`). The row order is
    /// the same: a window's rows go through the attention half in order (KDA walks its
    /// recurrence row by row; DSA writes the window's latents and indexer keys first, which no
    /// earlier row reads, then selects and attends sub-chunk by sub-chunk).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn prefill_staged_run(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        rows: usize,
        rows_ffn: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let subs = sub_chunks(num_tokens, rows);
        let wide = full_width_attn(prefill_fullwidth_gemm(), rows, rows_ffn);
        // 2026-10-01: The attention calls: the sub-chunks, or under the full-width lever the
        // windows of `rows_ffn` rows (whole sub-chunks, since `rows_ffn` is a multiple of
        // `rows`), each with its DSA core at `rows`.
        let attn_calls = if wide {
            sub_chunks(num_tokens, rows_ffn)
        } else {
            subs.clone()
        };
        for &(t, k) in &attn_calls {
            self.attn_half(
                hidden.offset(t * self.hidden * 2),
                k,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                ctx,
                stream,
                // 2026-09-29: No KDA snapshots: a prefill, as the unstaged loop passes.
                false,
                t,
                // This IS a prefill sub-chunk (staged attention pass).
                true,
                // 2026-10-01: `core_rows`: the DSA selection and attend width.
                if wide { rows.min(k) } else { k },
            )?;
        }
        let merge_tail = prefill_tail_merge();
        for (t, k) in ffn_windows(&subs, rows_ffn, merge_tail, |k| self.ffn_mergeable(k)) {
            let x = hidden.offset(t * self.hidden * 2);
            // 2026-10-01: Full width: the window's dense GEMMs in one slice.
            let dense_slice = if wide { k } else { rows };
            self.ffn_half(x, k, ctx, stream, t, dense_slice)?;
        }
        Ok(())
    }
}

/// 2026-10-01: Whether the staged pass takes the full-width arm: the lever is on and the FFN
/// window is wider than the attention sub-chunk (at equal widths there is nothing to widen).
pub(crate) fn full_width_attn(lever: bool, rows: usize, rows_ffn: usize) -> bool {
    lever && rows_ffn > rows
}

#[cfg(test)]
mod tests {
    use super::{ffn_windows, sub_chunks};

    /// 2026-09-29: The unstaged `prefill` loop, transcribed: `k = rows.min(num_tokens - t)`.
    fn unstaged(num_tokens: usize, rows: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut t = 0usize;
        while t < num_tokens {
            let k = rows.min(num_tokens - t);
            out.push((t, k));
            t += k;
        }
        out
    }

    const CASES: [(usize, usize); 9] = [
        (1, 256),
        (255, 256),
        (256, 256),
        (257, 256),
        (5400, 256),
        (8192, 256),
        (100, 16),
        (4097, 2048),
        (33, 1),
    ];

    /// 2026-09-29: At `W_ffn == W_attn` both passes visit exactly the unstaged row ranges.
    #[test]
    fn equal_widths_visit_the_unstaged_ranges() {
        for (n, rows) in CASES {
            let subs = sub_chunks(n, rows);
            assert_eq!(subs, unstaged(n, rows), "attention n={n} rows={rows}");
            let win = ffn_windows(&subs, rows, false, |_| true);
            assert_eq!(win, unstaged(n, rows), "ffn n={n} rows={rows}");
        }
    }

    /// 2026-09-29: Windows tile the chunk, start and end on sub-chunk boundaries, stay within
    /// the cap, and leave the tail and any unmergeable sub-chunk alone.
    #[test]
    fn wide_windows_are_whole_sub_chunks() {
        for (n, rows) in CASES {
            for ffn in [rows, 2 * rows, 2048, 4096] {
                let subs = sub_chunks(n, rows);
                let bounds: Vec<usize> = subs.iter().map(|s| s.0).chain([n]).collect();
                let win = ffn_windows(&subs, ffn, false, |_| true);
                let mut next = 0;
                for &(t, k) in &win {
                    assert_eq!(t, next, "contiguous n={n} rows={rows} ffn={ffn}");
                    assert!(bounds.contains(&t) && bounds.contains(&(t + k)));
                    assert!(k <= ffn.max(rows));
                    next = t + k;
                }
                assert_eq!(next, n);
                let tail = *subs.last().unwrap();
                if tail.1 < rows {
                    assert_eq!(*win.last().unwrap(), tail, "tail alone n={n} rows={rows}");
                }
                let alone = ffn_windows(&subs, ffn, false, |_| false);
                assert_eq!(alone, subs, "unmergeable n={n} rows={rows} ffn={ffn}");
            }
        }
    }

    /// 2026-10-01: `merge_tail`: a mergeable tail joins the window before it when that window
    /// has room, and the windows still tile the chunk on sub-chunk boundaries; an unmergeable
    /// tail, or one with no room, stays alone; nothing else changes.
    #[test]
    fn merged_tail_joins_the_last_window() {
        let subs = sub_chunks(8191, 256);
        assert_eq!(
            ffn_windows(&subs, 4096, false, |_| true),
            vec![(0, 4096), (4096, 3840), (7936, 255)]
        );
        assert_eq!(
            ffn_windows(&subs, 4096, true, |_| true),
            vec![(0, 4096), (4096, 4095)]
        );
        assert_eq!(
            ffn_windows(&subs, 4096, true, |k| k >= 256),
            vec![(0, 4096), (4096, 3840), (7936, 255)]
        );
        let full = sub_chunks(8192 + 100, 256);
        assert_eq!(
            ffn_windows(&full, 4096, true, |_| true),
            vec![(0, 4096), (4096, 4096), (8192, 100)]
        );
        for (n, rows) in CASES {
            for ffn in [rows, 2 * rows, 2048, 4096] {
                let subs = sub_chunks(n, rows);
                let a = ffn_windows(&subs, ffn, false, |_| true);
                let b = ffn_windows(&subs, ffn, true, |_| true);
                let bounds: Vec<usize> = subs.iter().map(|s| s.0).chain([n]).collect();
                let mut next = 0;
                for &(t, k) in &b {
                    assert_eq!(t, next, "contiguous n={n} rows={rows} ffn={ffn}");
                    assert!(bounds.contains(&t) && bounds.contains(&(t + k)));
                    assert!(k <= ffn.max(rows));
                    next = t + k;
                }
                assert_eq!(next, n);
                assert!(b.len() == a.len() || b.len() + 1 == a.len());
                assert_eq!(a[..a.len() - 2.min(a.len())], b[..a.len() - 2.min(a.len())]);
            }
        }
    }

    /// 2026-09-29: The production recipe's shape: 5400 tokens at 256/2048.
    #[test]
    fn the_5400_token_recipe() {
        let subs = sub_chunks(5400, 256);
        assert_eq!(subs.len(), 22);
        let win = ffn_windows(&subs, 2048, false, |_| true);
        assert_eq!(win, vec![(0, 2048), (2048, 2048), (4096, 1280), (5376, 24)]);
    }

    /// 2026-09-29: `resolve_rows_ffn`: default, rounding down to a multiple, the 8192 cap and the
    /// attention-width floor.
    #[test]
    fn ffn_width_resolution() {
        use crate::glm5next_layer::levers::resolve_rows_ffn;
        assert_eq!(resolve_rows_ffn(256, None), 256);
        assert_eq!(resolve_rows_ffn(256, Some(2048)), 2048);
        assert_eq!(resolve_rows_ffn(256, Some(2000)), 1792);
        assert_eq!(resolve_rows_ffn(256, Some(100)), 256);
        assert_eq!(resolve_rows_ffn(256, Some(0)), 256);
        assert_eq!(resolve_rows_ffn(256, Some(9000)), 8192);
        assert_eq!(resolve_rows_ffn(256, Some(4096)), 4096);
        assert_eq!(resolve_rows_ffn(3000, Some(9000)), 6000);
        assert_eq!(resolve_rows_ffn(8192, Some(2048)), 8192);
    }
}
