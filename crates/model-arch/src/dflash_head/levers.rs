// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DFlash drafter levers, resolved from the environment once per head and carried on it.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - `DFlashLevers::from_env` is called only while a head is built (`from_weights.rs`),
//!   so a head's levers are fixed for its lifetime.
//! - Each field reads one `METRALE_*` variable, except `any_diagnostic_armed`, which
//!   reads every name in `GRAPH_SUPPRESSING_DIAGNOSTICS`.
//!
//! The levers live on the head rather than in a process-wide `OnceLock`, so a second
//! model built in the same process resolves the environment again.

/// 2026-09-25: Diagnostic and A/B levers for one loaded DFlash drafter.
// 2026-09-25: No `Eq`: `conf_tau` is an `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DFlashLevers {
    /// 2026-09-25: A variable in `GRAPH_SUPPRESSING_DIAGNOSTICS` is set, at any
    /// value. `forward_block` then runs eagerly, without CUDA graphs. This is
    /// presence, not truth: `METRALE_DFLASH_BLOCK_DUMP=0` suppresses graphs and
    /// arms no dump.
    pub any_diagnostic_armed: bool,

    /// 2026-09-25: `METRALE_DFLASH_DEBUG_DUMP=1`: log the first 10 BF16 values
    /// of key `forward_block` intermediates.
    pub debug_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_DUMP_FULL=1`: one-shot whole-tensor dumps
    /// to `/tmp`, and the first propose's drafts to the log.
    pub debug_dump_full: bool,
    /// 2026-09-25: `METRALE_DFLASH_LOG_DRAFTS=1`: log the first propose's drafts
    /// (one-shot per model).
    pub log_drafts: bool,
    /// 2026-09-25: `METRALE_DFLASH_BLOCK_DUMP=1`: one-shot `/tmp` dumps of the
    /// block logits and per-layer intermediates, gated by `block_dump_armed_at`.
    pub block_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_BLOCK_DUMP_AT_POS=<n>`: arm the block dump only
    /// at decode position >= n. Unset or unparseable is 0.
    pub block_dump_at_pos: usize,
    /// 2026-09-25: `METRALE_DFLASH_OPTION_B_DIAG=1`: one-shot read-back of the
    /// K row layer 0 just wrote to the paged drafter cache, compared with its
    /// source.
    pub option_b_diag: bool,

    /// 2026-09-25: `METRALE_DFLASH_DEBUG_FORCE_PATTERN=1`: overwrite the first ctx
    /// slot `forward_block` reads with a deterministic pattern.
    pub force_pattern: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN=1`: the same, for the
    /// noise rows.
    pub force_noise_pattern: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_CTX_OFF=1`: drop ctx conditioning
    /// (`eff_ctx = 0`).
    pub force_no_ctx: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_CTX_USED=<n>`: cap `eff_ctx` at n (it
    /// stays bounded by the ctx rows present and `ctx_window`).
    pub force_ctx_used: Option<usize>,

    /// 2026-09-25: `METRALE_DFLASH_PRECOMPUTE=1`: also run the ctx K/V precompute
    /// from `forward_block`; `propose_drafts` runs it either way.
    pub precompute: bool,
    /// 2026-09-25: `METRALE_DFLASH_PRECOMPUTE_COMMIT=1`: let that diagnostic run
    /// write to the paged cache (its `commit` argument).
    pub precompute_commit: bool,

    /// 2026-09-25: `METRALE_DFLASH_PROPOSE_WARMUP_N=<n>`: eager passes per block
    /// width before graph capture. Unset or unparseable is 2.
    pub propose_warmup_n: usize,

    /// 2026-09-25: The paged drafter cache for ctx K/V ("Option B"). On unless
    /// `METRALE_DFLASH_OPTION_B=0`; resolved by `option_b_from` in `dflash_head.rs`.
    pub option_b: bool,
    /// 2026-09-25: `METRALE_DFLASH_OPTION_B_NO_CTX=1`: pass `ctx_count = 0`, so
    /// paged attention sees only the γ K/V written in-layer.
    pub option_b_no_ctx: bool,
    /// 2026-09-25: The DFlash2 conv and selector path. Runs when the checkpoint
    /// carries its components, unless `METRALE_DFLASH2=0`.
    pub dflash2: bool,
    /// 2026-09-25: `METRALE_DFLASH_BATCH_PROPOSE=<width>` caps the cross-sequence
    /// batch. Unset is `usize::MAX` (as wide as the scratch bands allow); `0` or
    /// `1` makes `propose_batch_max` return 1.
    pub batch_propose_width: usize,
    /// 2026-09-25: `METRALE_DFLASH_DRAFT_CAP=<n>`: return at most n drafts per
    /// propose. `None` caps at the block width in flight (`block_g`).
    pub draft_cap: Option<usize>,
    /// 2026-10-08: `METRALE_DFLASH_DRAFT_GEMV=1`: the drafter's BF16 projections of
    /// at most 16 rows run on `dense_gemv_bf16_batchm` instead of the 128x128-tile
    /// pipelined GEMM (`draft_gemm.rs`, SeedHQ/seed-skills#71 S1a).
    pub draft_gemv: bool,

    /// 2026-10-08: `METRALE_DFLASH_CONF_LOG=1`: after a single-sequence DFlash2 propose, read
    /// the selector's top-16 logits back (one blocking D2H) for the verify step's
    /// `DFLASH CONF` line. Log-only; does not suppress CUDA graphs.
    pub conf_log: bool,
    /// 2026-10-09: `METRALE_DFLASH_CONF_TAU=<t>`, `t` in (0, 1): the scheduler keeps a C=1
    /// DFlash2 block's drafts while their top-1 probability (the `conf_log` p) is at least
    /// `t`. Here it only makes a single-sequence propose compute p even with `conf_log` off.
    /// Unset, unparseable or outside (0, 1) is `0.0` (off); see [`parse_conf_tau`].
    pub dflash_conf_tau: f32,
    /// 2026-09-25: `METRALE_DFLASH_VERIFY_TRACE=1`: log every propose's drafts
    /// before the row-0 drop and the cap.
    pub verify_trace: bool,
    /// 2026-09-25: `METRALE_DFLASH_PRECOMPUTE_DUMP=1`: one-shot `/tmp` dumps of
    /// the ctx K/V precompute intermediates.
    pub precompute_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_CTX_PARITY_DUMP=1`: one-shot `/tmp` dump of the
    /// accumulated ctx hidden rows.
    pub ctx_parity_dump: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_NO_DECODE_APPEND=1`: skip the ctx append
    /// at the start of `propose_drafts`.
    pub no_decode_append: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_FULL_PRECOMPUTE=1`: the non-batched
    /// propose recomputes the whole ctx prefix (`committed = 0`) each step.
    pub full_precompute: bool,
    /// 2026-09-25: `METRALE_DFLASH_CTXLEN_PROBE=1`: warn when `ctx_positions` is
    /// not strictly increasing, and log `ctx_len` against the position when the
    /// position is a multiple of 16.
    pub ctxlen_probe: bool,

    /// 2026-09-25: The sequential Markov fixup. Runs when the drafter carries
    /// the head, unless `METRALE_DSPARK_MARKOV=0`, which takes the plain
    /// per-row argmax loop.
    pub dspark_markov: bool,
    /// 2026-09-25: `METRALE_DSPARK_CONF_TAU=<t>`: sigmoid-space acceptance
    /// threshold for the confidence head. Unset or unparseable is `0.0`, which
    /// leaves the head off.
    pub conf_tau: f32,
    /// 2026-09-25: `METRALE_DSPARK_SHIFT=1|0` forces the shifted-row convention
    /// on or off; any other value (`None`) defers to the drafter config.
    pub dspark_shift: Option<bool>,
    /// 2026-09-25: Row 0 carries the Markov bias. On unless
    /// `METRALE_DSPARK_ANCHOR_BIAS=0`. Confidence truncation also requires it:
    /// an unbiased row never writes its confidence slot.
    pub dspark_anchor_bias: bool,
    /// 2026-09-25: `METRALE_DSPARK_CONF_TRACE=1`: log the confidence logits and
    /// sigmoids.
    pub dspark_conf_trace: bool,
}

/// 2026-09-25: Variables whose presence, at any value, sets `any_diagnostic_armed`
/// and so keeps `forward_block` off CUDA graphs.
const GRAPH_SUPPRESSING_DIAGNOSTICS: [&str; 11] = [
    "METRALE_DFLASH_PROPOSE_NO_GRAPH",
    "METRALE_DFLASH_DEBUG_DUMP_FULL",
    "METRALE_DFLASH_OPTION_B_DIAG",
    "METRALE_DFLASH_PRECOMPUTE_DUMP",
    "METRALE_DFLASH_VERIFY_TRACE",
    "METRALE_DFLASH_LOG_DRAFTS",
    "METRALE_DFLASH_DEBUG_FORCE_PATTERN",
    "METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
    "METRALE_DFLASH_DEBUG_CTX_OFF",
    "METRALE_DFLASH_DEBUG_CTX_USED",
    "METRALE_DFLASH_BLOCK_DUMP",
];

fn from_values(
    mut value: impl FnMut(&str) -> Option<String>,
    mut present: impl FnMut(&str) -> bool,
) -> DFlashLevers {
    fn opt_in(value: Option<&str>) -> bool {
        value == Some("1")
    }
    fn parsed<T: std::str::FromStr>(value: Option<&str>) -> Option<T> {
        value.and_then(|v| v.parse().ok())
    }

    DFlashLevers {
        any_diagnostic_armed: GRAPH_SUPPRESSING_DIAGNOSTICS.iter().any(|var| present(var)),

        debug_dump: opt_in(value("METRALE_DFLASH_DEBUG_DUMP").as_deref()),
        debug_dump_full: opt_in(value("METRALE_DFLASH_DEBUG_DUMP_FULL").as_deref()),
        log_drafts: opt_in(value("METRALE_DFLASH_LOG_DRAFTS").as_deref()),
        block_dump: opt_in(value("METRALE_DFLASH_BLOCK_DUMP").as_deref()),
        block_dump_at_pos: parsed(value("METRALE_DFLASH_BLOCK_DUMP_AT_POS").as_deref())
            .unwrap_or(0),
        option_b_diag: opt_in(value("METRALE_DFLASH_OPTION_B_DIAG").as_deref()),

        force_pattern: opt_in(value("METRALE_DFLASH_DEBUG_FORCE_PATTERN").as_deref()),
        force_noise_pattern: opt_in(value("METRALE_DFLASH_DEBUG_FORCE_NOISE_PATTERN").as_deref()),
        force_no_ctx: opt_in(value("METRALE_DFLASH_DEBUG_CTX_OFF").as_deref()),
        force_ctx_used: parsed(value("METRALE_DFLASH_DEBUG_CTX_USED").as_deref()),

        precompute: opt_in(value("METRALE_DFLASH_PRECOMPUTE").as_deref()),
        precompute_commit: opt_in(value("METRALE_DFLASH_PRECOMPUTE_COMMIT").as_deref()),

        option_b: super::option_b_from(value("METRALE_DFLASH_OPTION_B").as_deref()),
        option_b_no_ctx: opt_in(value("METRALE_DFLASH_OPTION_B_NO_CTX").as_deref()),
        dflash2: value("METRALE_DFLASH2").as_deref() != Some("0"),
        batch_propose_width: parsed(value("METRALE_DFLASH_BATCH_PROPOSE").as_deref())
            .unwrap_or(usize::MAX),
        draft_cap: parsed(value("METRALE_DFLASH_DRAFT_CAP").as_deref()),
        draft_gemv: opt_in(value("METRALE_DFLASH_DRAFT_GEMV").as_deref()),

        conf_log: opt_in(value("METRALE_DFLASH_CONF_LOG").as_deref()),
        dflash_conf_tau: parse_conf_tau(value("METRALE_DFLASH_CONF_TAU").as_deref()),
        verify_trace: opt_in(value("METRALE_DFLASH_VERIFY_TRACE").as_deref()),
        precompute_dump: opt_in(value("METRALE_DFLASH_PRECOMPUTE_DUMP").as_deref()),
        ctx_parity_dump: opt_in(value("METRALE_DFLASH_CTX_PARITY_DUMP").as_deref()),
        no_decode_append: opt_in(value("METRALE_DFLASH_DEBUG_NO_DECODE_APPEND").as_deref()),
        full_precompute: opt_in(value("METRALE_DFLASH_DEBUG_FULL_PRECOMPUTE").as_deref()),
        ctxlen_probe: opt_in(value("METRALE_DFLASH_CTXLEN_PROBE").as_deref()),

        dspark_markov: value("METRALE_DSPARK_MARKOV").as_deref() != Some("0"),
        conf_tau: parsed(value("METRALE_DSPARK_CONF_TAU").as_deref()).unwrap_or(0.0),

        propose_warmup_n: parsed(value("METRALE_DFLASH_PROPOSE_WARMUP_N").as_deref()).unwrap_or(2),

        dspark_shift: match value("METRALE_DSPARK_SHIFT").as_deref() {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        },
        dspark_anchor_bias: value("METRALE_DSPARK_ANCHOR_BIAS").as_deref() != Some("0"),
        dspark_conf_trace: opt_in(value("METRALE_DSPARK_CONF_TRACE").as_deref()),
    }
}

/// 2026-10-09: `METRALE_DFLASH_CONF_TAU`: a finite `t` with `0 < t < 1`, else `0.0` (off).
pub fn parse_conf_tau(value: Option<&str>) -> f32 {
    value
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|t| t.is_finite() && *t > 0.0 && *t < 1.0)
        .unwrap_or(0.0)
}

impl DFlashLevers {
    /// 2026-09-25: Resolve from the environment. The head calls it once, in
    /// `from_weights.rs`; per-step code reads the head's `levers` field.
    pub fn from_env() -> Self {
        from_values(metrale_config::levers::var, |var| {
            metrale_config::levers::var_os(var).is_some()
        })
    }

    /// 2026-09-25: What `from_env` resolves to with none of these variables set
    /// (checked by `nothing_set_resolves_to_defaults`).
    pub fn defaults() -> Self {
        Self {
            // 2026-09-25: Every field whose unset value is not its type's
            // `Default` is spelled out here.
            option_b: true,
            dflash2: true,
            dspark_markov: true,
            batch_propose_width: usize::MAX,
            dspark_anchor_bias: true,
            propose_warmup_n: 2,
            ..Self::default()
        }
    }

    /// 2026-09-25: The block dump is armed for this decode position.
    pub fn block_dump_armed_at(&self, position: usize) -> bool {
        self.block_dump && position >= self.block_dump_at_pos
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
