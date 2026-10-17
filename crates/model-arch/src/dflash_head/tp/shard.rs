// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Pure index maths of the two-rank DFlash drafter (`METRALE_DFLASH_DRAFT_TP2`):
//! which slice of every drafter weight a rank holds, the top-16 merge, and the ctx-row
//! bookkeeping both ranks agree on. No GPU here, so `tests.rs` checks all of it on the host.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - For every weight, the rank slices of [`weight_parts`] cover the full `[n, k]` matrix
//!   exactly once (checked by `tests.rs`).
//! - [`merge_top16_rows`] equals the single-rank top-16 under the order "higher value, then
//!   lower token id" whenever each rank's input is its own shard's top-16 under that order.

use anyhow::{Result, ensure};

/// 2026-10-09: Candidates per row in the DFlash2 selector (`dflash2_topk16`).
pub const TOP_K: usize = 16;

/// 2026-10-09: Split `total` into `world` contiguous parts in units of `align`. The first
/// `(total / align) % world` ranks take one extra unit. Returns `(start, len)` of `rank`.
pub fn shard_range(
    total: usize,
    world: usize,
    rank: usize,
    align: usize,
) -> Result<(usize, usize)> {
    ensure!(
        world >= 1 && rank < world,
        "shard_range: rank {rank} of world {world}"
    );
    ensure!(align >= 1, "shard_range: align 0");
    ensure!(
        total.is_multiple_of(align),
        "shard_range: {total} is not a multiple of {align}"
    );
    let units = total / align;
    let base = units / world;
    let extra = units % world;
    let start_units = rank * base + rank.min(extra);
    let len_units = base + usize::from(rank < extra);
    Ok((start_units * align, len_units * align))
}

/// 2026-10-09: The drafter shapes the TP split is derived from (full, unsharded values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpGeometry {
    pub world: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub inter: usize,
    pub vocab: usize,
    /// 2026-10-09: `fc` input width (`target_layer_ids.len() * target_hidden_size`).
    pub fc_in: usize,
}

/// 2026-10-09: Alignment of the intermediate, vocab and `fc` output splits: the row-scaled
/// FP8 GEMM tiles N by 128, and 128 keeps every K slice a multiple of 16 for the rt2 GEMVs.
pub const SPLIT_ALIGN: usize = 128;

/// 2026-10-09: One rank's slice of each sharded dimension, as `(start, len)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankLayout {
    pub q_heads: (usize, usize),
    pub kv_heads: (usize, usize),
    pub inter: (usize, usize),
    pub vocab: (usize, usize),
    pub fc_out: (usize, usize),
}

impl TpGeometry {
    /// 2026-10-09: Whether this geometry can be split: every KV head group stays on one
    /// rank (`kv_heads % world == 0`, `q_heads % kv_heads == 0`), each rank gets at least
    /// one 128-wide unit of the intermediate and vocab, and `fc`'s output splits evenly
    /// (its all-gather needs equal parts). The error says which condition failed.
    pub fn check(&self) -> Result<()> {
        ensure!(self.world >= 2, "TP needs world >= 2, got {}", self.world);
        ensure!(
            self.kv_heads > 0 && self.kv_heads.is_multiple_of(self.world),
            "num_key_value_heads {} not divisible by {}",
            self.kv_heads,
            self.world
        );
        ensure!(
            self.q_heads.is_multiple_of(self.kv_heads),
            "num_attention_heads {} not a multiple of num_key_value_heads {}",
            self.q_heads,
            self.kv_heads
        );
        for (name, v) in [
            ("intermediate_size", self.inter),
            ("vocab_size", self.vocab),
        ] {
            ensure!(
                v.is_multiple_of(SPLIT_ALIGN) && v / SPLIT_ALIGN >= self.world,
                "{name} {v} not splittable in {SPLIT_ALIGN}-wide units over {} ranks",
                self.world
            );
        }
        ensure!(
            self.hidden.is_multiple_of(SPLIT_ALIGN * self.world),
            "hidden_size {} not divisible by {} (fc all-gather needs equal parts)",
            self.hidden,
            SPLIT_ALIGN * self.world
        );
        Ok(())
    }

    /// 2026-10-14: The whole drafter as one rank holds it (no TP): every slice is the full
    /// dimension. [`weight_parts_with`] over it gives the unsharded weights.
    pub fn full_layout(&self) -> RankLayout {
        RankLayout {
            q_heads: (0, self.q_heads),
            kv_heads: (0, self.kv_heads),
            inter: (0, self.inter),
            vocab: (0, self.vocab),
            fc_out: (0, self.hidden),
        }
    }

    /// 2026-10-09: `rank`'s slices. Errors on a geometry [`Self::check`] rejects.
    pub fn layout(&self, rank: usize) -> Result<RankLayout> {
        self.check()?;
        let kv = shard_range(self.kv_heads, self.world, rank, 1)?;
        let ratio = self.q_heads / self.kv_heads;
        Ok(RankLayout {
            q_heads: (kv.0 * ratio, kv.1 * ratio),
            kv_heads: kv,
            inter: shard_range(self.inter, self.world, rank, SPLIT_ALIGN)?,
            vocab: shard_range(self.vocab, self.world, rank, SPLIT_ALIGN)?,
            fc_out: shard_range(self.hidden, self.world, rank, SPLIT_ALIGN)?,
        })
    }
}

/// 2026-10-09: How a weight `W[n, k]` (`y = x W^T`) is split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// 2026-10-09: Column-parallel: rows `[start, start + len)` of `W` (output features).
    Rows,
    /// 2026-10-09: Row-parallel: columns `[start, start + len)` of `W` (input features);
    /// the partial outputs are summed across ranks.
    Cols,
}

/// 2026-10-09: One rank's part of one weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightPart {
    pub name: &'static str,
    pub n: usize,
    pub k: usize,
    pub split: Split,
    pub start: usize,
    pub len: usize,
}

/// 2026-10-09: Every sharded drafter weight's part on `rank`, in the order
/// `shard_weights.rs` applies them. q/k/v/gate/up/lm_head/fc are row slices (contiguous,
/// so a pointer offset); o/down are column slices (a pitched copy).
pub fn weight_parts(g: &TpGeometry, rank: usize) -> Result<Vec<WeightPart>> {
    Ok(weight_parts_with(g, &g.layout(rank)?))
}

/// 2026-10-14: [`weight_parts`] for an explicit layout: one rank's slices, or
/// [`TpGeometry::full_layout`] for the unsharded drafter. The FP8 mirror prediction
/// (`from_weights/fp8_weights.rs`) reads both through this one rule.
pub fn weight_parts_with(g: &TpGeometry, l: &RankLayout) -> Vec<WeightPart> {
    let hd = g.head_dim;
    let q_dim = g.q_heads * hd;
    let kv_dim = g.kv_heads * hd;
    let rows = |name, n, k, r: (usize, usize)| WeightPart {
        name,
        n,
        k,
        split: Split::Rows,
        start: r.0,
        len: r.1,
    };
    let cols = |name, n, k, r: (usize, usize)| WeightPart {
        name,
        n,
        k,
        split: Split::Cols,
        start: r.0,
        len: r.1,
    };
    let q = (l.q_heads.0 * hd, l.q_heads.1 * hd);
    let kv = (l.kv_heads.0 * hd, l.kv_heads.1 * hd);
    vec![
        rows("q_proj", q_dim, g.hidden, q),
        rows("k_proj", kv_dim, g.hidden, kv),
        rows("v_proj", kv_dim, g.hidden, kv),
        cols("o_proj", g.hidden, q_dim, q),
        rows("gate_proj", g.inter, g.hidden, l.inter),
        rows("up_proj", g.inter, g.hidden, l.inter),
        cols("down_proj", g.hidden, g.inter, l.inter),
        rows("lm_head", g.vocab, g.hidden, l.vocab),
        rows("fc", g.hidden, g.fc_in, l.fc_out),
    ]
}

/// 2026-10-09: The candidate order of the merge: higher value first, then lower token id.
/// A NaN value sorts as negative infinity.
fn cand_order(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    let key = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
    key(b.0).total_cmp(&key(a.0)).then(a.1.cmp(&b.1))
}

/// 2026-10-09: One rank's top-16 for `rows` rows: values and shard-local ids, row-major
/// `[rows, 16]`, plus the first global id of the rank's vocab slice.
pub struct RankTop<'a> {
    pub vals: &'a [f32],
    pub ids: &'a [u32],
    pub vocab_start: u32,
}

/// 2026-10-09: Merge every rank's per-row top-16 into the global top-16 (ids made global by
/// adding the rank's `vocab_start`), sorted by "higher value, then lower id". Each global
/// top-16 member is in its own shard's top-16 under the same order, so the result equals the
/// single-rank top-16 under that order. `dflash2_topk16` breaks ties by thread layout, not by
/// id, so on exact ties the single-rank kernel and this merge may pick different ids.
pub fn merge_top16_rows(rows: usize, parts: &[RankTop<'_>]) -> Result<(Vec<f32>, Vec<u32>)> {
    for p in parts {
        ensure!(
            p.vals.len() == rows * TOP_K && p.ids.len() == rows * TOP_K,
            "merge_top16_rows: part has {} values / {} ids for {rows} rows",
            p.vals.len(),
            p.ids.len()
        );
    }
    let mut out_v = Vec::with_capacity(rows * TOP_K);
    let mut out_i = Vec::with_capacity(rows * TOP_K);
    let mut cands: Vec<(f32, u32)> = Vec::with_capacity(parts.len() * TOP_K);
    for r in 0..rows {
        cands.clear();
        for p in parts {
            let row = r * TOP_K..(r + 1) * TOP_K;
            cands.extend(
                p.vals[row.clone()]
                    .iter()
                    .zip(&p.ids[row])
                    .map(|(&v, &i)| (v, i + p.vocab_start)),
            );
        }
        cands.sort_by(cand_order);
        for &(v, i) in cands.iter().take(TOP_K) {
            out_v.push(v);
            out_i.push(i);
        }
    }
    Ok((out_v, out_i))
}

/// 2026-10-09: The reference: top-16 of one full row under the merge order (the merge
/// contract, checked by `tests.rs`).
#[cfg(test)]
pub fn top16_reference(row: &[f32], id_base: u32) -> Vec<(f32, u32)> {
    let mut c: Vec<(f32, u32)> = row
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32 + id_base))
        .collect();
    c.sort_by(cand_order);
    c.truncate(TOP_K);
    c
}

/// 2026-10-09: Rows of rank 0's ctx accumulator the follower already holds: those whose
/// position is at most `synced_pos`. `None` (nothing synced) or positions that are not
/// strictly increasing give 0, which makes the leader send every row.
pub fn leader_keep(positions: &[i32], synced_pos: Option<i32>) -> usize {
    let Some(s) = synced_pos else { return 0 };
    if positions.windows(2).any(|w| w[1] <= w[0]) {
        return 0;
    }
    positions.partition_point(|&p| p <= s)
}

/// 2026-10-09: How the follower turns its ctx rows into rank 0's: drop its first `drop`
/// rows, keep the next `keep`, then append the rows the header carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conform {
    pub drop: usize,
    pub keep: usize,
}

/// 2026-10-09: The follower's conform plan, or an error naming the mismatch (the follower
/// then runs the propose's collectives on scratch and reports the desync).
/// `first_pos`/`keep_last_pos` are rank 0's positions of its row 0 and row `keep - 1`.
pub fn plan_conform(
    follower_positions: &[i32],
    follower_cap: usize,
    keep: usize,
    first_pos: i32,
    keep_last_pos: i32,
    ctx_len: usize,
) -> Result<Conform> {
    ensure!(
        ctx_len <= follower_cap,
        "ctx_len {ctx_len} exceeds the follower's {follower_cap} rows"
    );
    ensure!(keep <= ctx_len, "keep {keep} > ctx_len {ctx_len}");
    if keep == 0 {
        return Ok(Conform { drop: 0, keep: 0 });
    }
    let drop = follower_positions
        .iter()
        .position(|&p| p == first_pos)
        .ok_or_else(|| anyhow::anyhow!("follower has no row at position {first_pos}"))?;
    ensure!(
        drop + keep <= follower_positions.len(),
        "follower holds {} rows, needs {} from row {drop}",
        follower_positions.len(),
        keep
    );
    ensure!(
        follower_positions[drop + keep - 1] == keep_last_pos,
        "follower row {} is at position {}, rank 0 kept {keep_last_pos}",
        drop + keep - 1,
        follower_positions[drop + keep - 1]
    );
    Ok(Conform { drop, keep })
}

/// 2026-10-09: The precompute chunks `[committed, ctx_len)` in pieces of at most `window`
/// rows, as `(start, len)`: one `fc` all-gather each. Both ranks derive the propose's
/// collective schedule from it.
pub fn precompute_chunks(committed: usize, ctx_len: usize, window: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if window == 0 {
        return out;
    }
    let mut s = committed.min(ctx_len);
    while s < ctx_len {
        let n = (ctx_len - s).min(window);
        out.push((s, n));
        s += n;
    }
    out
}
