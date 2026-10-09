// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `METRALE_GLM_DFLASH_KSEL=1`: a per-step, batch-wide verify width for the GLM
//! DFlash batched verify (`step_verify_dflash_batched` under `DflashRung::multi_batched`).
//!
//! For a batch of `n` streams this picks one K (verify rows per stream, drafts + 1, the
//! `METRALE_GLM_DFLASH_MULTI_K` sense) in `2..=k_max`, `k_max` being the step's rung width
//! (drafts + 1, at most `EP_BATCH_VERIFY_MAX_K`), that maximises
//! `sum_s E_s(K) / cost(n, K)`, with `E_s(K) = 1 + sum_{j=1..K-1} prod_{i<=j} p_{s,i}` (expected
//! tokens committed per stream: accepted drafts plus the bonus). `p_{s,i}` is the drafter's
//! top-1 probability (`conf_log::p_for`); a stream without a record uses
//! `METRALE_GLM_DFLASH_KSEL_PDEF` (default 0.6) at every position. `cost(n, K)` is the measured
//! verify + propose ms from `METRALE_GLM_DFLASH_KCOST="n:K:ms,..."`, linearly interpolated in K
//! between the entries present for that `n` (never extrapolated).
//!
//! Owner: scheduler.
//! Invariants:
//! - Exact under the batched verify: only how many drafts are checked changes, never a drafted,
//!   verified or emitted token's value (as `METRALE_DFLASH_CONF_TAU` at C=1).
//! - Rank 0 only. The chosen K reaches the EP worker in the existing `EP_CMD_VERIFY_BATCH` meta
//!   (`ks[n]`, `verify_ep.rs`), which the worker verifies, commits and stashes by; no new words.
//! - One K per batch (`dflash_batch_shape_ok` needs a uniform k); only a K the model accepts
//!   for this batch (`can_batch_verify`) is a candidate.
//! - No table, no entry for `n`, or no costed candidate: the step keeps the fixed width (the
//!   rung's, i.e. `METRALE_GLM_DFLASH_MULTI_K` capped by the drafts).
//! - The levers are read once per process ([`init`], called at serve setup on rank 0).
//!
//! p at C>=2 (2026-10-09): `conf_log` records only a single-sequence DFlash2 propose, and only
//! with `METRALE_DFLASH_CONF_LOG` or `METRALE_DFLASH_CONF_TAU` on; the batched propose records
//! nothing, and the one global record matches at most the stream proposed last. So at C>=2 most
//! streams score on the PDEF prior and the choice is mostly the cost table's per-n optimum.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::ActiveSeq;
use metrale_model_engine::traits::Model;

/// 2026-10-09: Log the 1st selection and then every this many (as `DFLASH ACCEPT SUMMARY`).
const LOG_EVERY: u64 = 32;
/// 2026-10-09: The prior top-1 probability for a stream without a `conf_log` record.
pub(crate) const PDEF_DEFAULT: f32 = 0.6;

/// 2026-10-09: Measured verify + propose ms per `(n, K)`, sorted by `(n, K)`, no duplicates.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CostTable {
    entries: Vec<(usize, usize, f64)>,
}

impl CostTable {
    /// 2026-10-09: The cost of a batch of `n` at width `k`: the entry, or the linear
    /// interpolation between the nearest entries below and above `k` for that `n`; `None`
    /// outside the measured range or with no entry for `n`.
    pub(crate) fn cost(&self, n: usize, k: usize) -> Option<f64> {
        let row: Vec<(usize, f64)> = self
            .entries
            .iter()
            .filter(|e| e.0 == n)
            .map(|e| (e.1, e.2))
            .collect();
        if let Some(&(_, ms)) = row.iter().find(|e| e.0 == k) {
            return Some(ms);
        }
        let lo = row.iter().rev().find(|e| e.0 < k)?;
        let hi = row.iter().find(|e| e.0 > k)?;
        let t = (k - lo.0) as f64 / (hi.0 - lo.0) as f64;
        Some(lo.1 + t * (hi.1 - lo.1))
    }

    /// 2026-10-09: Whether any entry is for a batch of `n`.
    pub(crate) fn has_n(&self, n: usize) -> bool {
        self.entries.iter().any(|e| e.0 == n)
    }
}

/// 2026-10-09: Parse `"n:K:ms,n:K:ms,..."` (whitespace around fields allowed). Every entry
/// needs `n >= 2`, `K >= 2`, and a finite `ms > 0`; a repeated `(n, K)` or an empty list is
/// malformed. `Err` names the first problem.
pub(crate) fn parse_cost_table(s: &str) -> Result<CostTable, String> {
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    for item in s.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return Err(format!("empty entry in {s:?}"));
        }
        let f: Vec<&str> = item.split(':').map(str::trim).collect();
        let [n, k, ms] = f[..] else {
            return Err(format!("entry {item:?} is not n:K:ms"));
        };
        let n: usize = n.parse().map_err(|_| format!("entry {item:?}: bad n"))?;
        let k: usize = k.parse().map_err(|_| format!("entry {item:?}: bad K"))?;
        let ms: f64 = ms.parse().map_err(|_| format!("entry {item:?}: bad ms"))?;
        if n < 2 || k < 2 || !ms.is_finite() || ms <= 0.0 {
            return Err(format!("entry {item:?}: needs n>=2, K>=2, ms>0"));
        }
        if entries.iter().any(|e| e.0 == n && e.1 == k) {
            return Err(format!("entry {item:?}: (n, K) repeated"));
        }
        entries.push((n, k, ms));
    }
    entries.sort_by_key(|e| (e.0, e.1));
    Ok(CostTable { entries })
}

/// 2026-10-09: [`parse_cost_table`], warning and returning `None` on a malformed value.
pub(crate) fn parse_kcost(s: &str) -> Option<CostTable> {
    match parse_cost_table(s) {
        Ok(t) => Some(t),
        Err(e) => {
            tracing::warn!(
                "METRALE_GLM_DFLASH_KCOST malformed ({e}); KSEL keeps the fixed C>=2 width"
            );
            None
        }
    }
}

/// 2026-10-09: `E(K) = 1 + sum_{j=1..K-1} prod_{i<=j} p_i`, each p clamped to `[0, 1]`;
/// positions past `p.len()` use `pdef`.
pub(crate) fn expected_tokens(p: &[f32], pdef: f32, k: usize) -> f64 {
    let mut e = 1.0f64;
    let mut prod = 1.0f64;
    for j in 0..k.saturating_sub(1) {
        let pj = p.get(j).copied().unwrap_or(pdef).clamp(0.0, 1.0);
        prod *= f64::from(pj);
        e += prod;
    }
    e
}

/// 2026-10-09: The chosen width, its score (expected tokens per ms over the batch) and every
/// costed candidate's `(K, score)`, ascending in K.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Selection {
    pub k: usize,
    pub score: f64,
    pub alts: Vec<(usize, f64)>,
}

/// 2026-10-09: The argmax over K in `2..=k_max` (`allowed(K)` and costed) of
/// `sum_s E_s(K) / cost(n, K)`, `n = streams.len()`; a tie keeps the wider K (the fixed
/// width's side). `None` (the caller keeps the fixed width) when no candidate is costed.
pub(crate) fn select_k(
    streams: &[Option<Vec<f32>>],
    k_max: usize,
    pdef: f32,
    table: &CostTable,
    allowed: impl Fn(usize) -> bool,
) -> Option<Selection> {
    let n = streams.len();
    if n < 2 || !table.has_n(n) {
        return None;
    }
    let mut alts: Vec<(usize, f64)> = Vec::new();
    for k in 2..=k_max {
        let Some(ms) = table.cost(n, k) else {
            continue;
        };
        if !allowed(k) {
            continue;
        }
        let tokens: f64 = streams
            .iter()
            .map(|p| expected_tokens(p.as_deref().unwrap_or(&[]), pdef, k))
            .sum();
        alts.push((k, tokens / ms));
    }
    let &(k, score) = alts
        .iter()
        .fold(None::<&(usize, f64)>, |best, c| match best {
            Some(b) if b.1 > c.1 => Some(b),
            _ => Some(c),
        })?;
    Some(Selection { k, score, alts })
}

/// 2026-10-09: The resolved levers.
struct Ksel {
    on: bool,
    pdef: f32,
    table: Option<CostTable>,
}

static CFG: OnceLock<Ksel> = OnceLock::new();
static SELECTIONS: AtomicU64 = AtomicU64::new(0);

/// 2026-10-09: `METRALE_GLM_DFLASH_KSEL_PDEF`: a float in `[0, 1]`; anything else warns and
/// keeps [`PDEF_DEFAULT`].
pub(crate) fn parse_pdef(v: Option<&str>) -> f32 {
    match v.map(|s| s.trim().parse::<f32>()) {
        None => PDEF_DEFAULT,
        Some(Ok(p)) if (0.0..=1.0).contains(&p) => p,
        Some(_) => {
            tracing::warn!(
                "METRALE_GLM_DFLASH_KSEL_PDEF={v:?} not a float in [0, 1]; keeping {PDEF_DEFAULT}"
            );
            PDEF_DEFAULT
        }
    }
}

fn cfg() -> &'static Ksel {
    CFG.get_or_init(|| {
        let on = metrale_model_layers::speculative::glm_dflash::parse_switch(
            std::env::var("METRALE_GLM_DFLASH_KSEL").ok().as_deref(),
        );
        if !on {
            return Ksel {
                on,
                pdef: PDEF_DEFAULT,
                table: None,
            };
        }
        let pdef = parse_pdef(
            std::env::var("METRALE_GLM_DFLASH_KSEL_PDEF")
                .ok()
                .as_deref(),
        );
        let table = match std::env::var("METRALE_GLM_DFLASH_KCOST") {
            Ok(s) => parse_kcost(&s),
            Err(_) => {
                tracing::warn!(
                    "METRALE_GLM_DFLASH_KSEL: METRALE_GLM_DFLASH_KCOST unset; fixed C>=2 width"
                );
                None
            }
        };
        tracing::warn!(
            "METRALE_GLM_DFLASH_KSEL: ENGAGED (pdef={pdef}, cost entries={}) - GLM DFlash \
             batched verify picks one K in 2..=rung per step by E[tokens]/cost(n,K)",
            table.as_ref().map_or(0, |t| t.entries.len()),
        );
        Ksel { on, pdef, table }
    })
}

/// 2026-10-09: Resolve the levers (first call only) and log `ENGAGED` once when on.
pub(crate) fn init() {
    let _ = cfg();
}

/// 2026-10-09: Under the lever and the batched GLM pin, choose this batch's width and truncate
/// every row's drafts (and confidences) to `K - 1`; returns the drafts per stream the verify
/// runs with (`drafts_per_seq` unchanged when the step keeps the fixed width). Every row
/// carries exactly `drafts_per_seq` drafts on entry (`split_verify_rows`).
pub(super) fn select_and_truncate(
    model: &dyn Model,
    batch: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts_per_seq: usize,
) -> usize {
    let c = cfg();
    if !c.on || !sched.dflash_rung.multi_batched() || batch.len() < 2 || drafts_per_seq < 1 {
        return drafts_per_seq;
    }
    let Some(table) = c.table.as_ref() else {
        return drafts_per_seq;
    };
    let n = batch.len();
    let k_max = (drafts_per_seq + 1)
        .min(metrale_model_layers::speculative::glm_dflash_batch::EP_BATCH_VERIFY_MAX_K);
    let streams: Vec<Option<Vec<f32>>> = batch
        .iter()
        .map(|a| metrale_model_arch::dflash_head::conf_log::p_for(&a.pending_drafts))
        .collect();
    let Some(sel) = select_k(&streams, k_max, c.pdef, table, |k| {
        model.can_batch_verify(&vec![k; n])
    }) else {
        if sched.io.tel.stats().once("log:dflash_ksel_fallback") {
            tracing::info!(
                "DFLASH KSEL: n={n} no costed K in 2..={k_max}; fixed K={}",
                drafts_per_seq + 1
            );
        }
        return drafts_per_seq;
    };
    for a in batch.iter_mut() {
        a.pending_drafts.truncate(sel.k - 1);
        a.pending_draft_conf.truncate(sel.k - 1);
    }
    let count = SELECTIONS.fetch_add(1, Ordering::Relaxed);
    if count.is_multiple_of(LOG_EVERY) {
        let alt: Vec<String> = sel
            .alts
            .iter()
            .map(|(k, s)| format!("{k}:{s:.4}"))
            .collect();
        tracing::info!(
            "DFLASH KSEL: n={n} K={} score={:.4} alt={} p_streams={}/{n} sel={}",
            sel.k,
            sel.score,
            alt.join(","),
            streams.iter().filter(|p| p.is_some()).count(),
            count + 1,
        );
    }
    sel.k - 1
}

#[cfg(test)]
#[path = "dflash_ksel_tests.rs"]
mod tests;
