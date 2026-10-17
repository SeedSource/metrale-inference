// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Hybrid speculative decode (`METRALE_HYBRID_DFLASH=1`, issue #82): one serve holds
//! the GLM MTP head and a DFlash drafter, and rank 0 picks the proposer for each step. This
//! module holds the pieces every crate shares: the two levers, the per-step mode, and the EP
//! command that carries the mode to the worker ranks. Host-side and pure apart from the lever
//! reads, so it is CPU-tested (from the server crate's `hybrid_mode_tests.rs`).
//!
//! Owner: model-layers (speculative).
//! Invariants:
//! - The mode is decided on rank 0 alone and reaches every other rank only through
//!   [`EP_CMD_SPEC_MODE`]; a worker never derives it from its own state.
//! - [`decode_spec_mode`] refuses any word [`encode_spec_mode`] cannot produce, so a
//!   desynchronised command stream fails loudly instead of switching a proposer.
//! - Both levers are read once per process.

use anyhow::{Result, bail};

/// 2026-10-10: The proposer one speculative step runs. `Plain` steps propose nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SpecStepMode {
    Plain = 0,
    Mtp = 1,
    Dflash = 2,
}

impl SpecStepMode {
    /// 2026-10-10: Short name for logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Plain => "PLAIN",
            Self::Mtp => "MTP",
            Self::Dflash => "DFLASH",
        }
    }

    /// 2026-10-10: Inverse of `as u8`; `None` for any other byte.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Plain),
            1 => Some(Self::Mtp),
            2 => Some(Self::Dflash),
            _ => None,
        }
    }
}

/// 2026-10-10: EP worker command: set the worker's step mode. Payload: one word,
/// [`encode_spec_mode`]. Rank 0 sends it (preamble `seq_id` 0, ignored) whenever its mode
/// changes, before the first command of the step that runs in the new mode, so the worker's
/// proposer routing (`EP_CMD_MTP_PROPOSE`, verify trims) always matches rank 0's.
/// 2026-10-11: 0xFFFF_FFFC; it was 0xFFFF_FFFB, which the car's `EP_CMD_MIGRATE_SLOT`
/// (METRALE_MTP_POOL_CAP EP v2, 198cb05d) also took.
pub const EP_CMD_SPEC_MODE: u32 = 0xFFFF_FFFC;

/// 2026-10-10: High half of a mode word: `"HY"`. A word without it is not a mode word.
const SPEC_MODE_MAGIC: u32 = 0x4859_0000;

/// 2026-10-10: The payload word of [`EP_CMD_SPEC_MODE`].
pub fn encode_spec_mode(mode: SpecStepMode) -> u32 {
    SPEC_MODE_MAGIC | mode as u32
}

/// 2026-10-10: Decode a [`EP_CMD_SPEC_MODE`] payload word. Errors on a missing magic or an
/// unknown mode value.
pub fn decode_spec_mode(word: u32) -> Result<SpecStepMode> {
    if word & 0xFFFF_0000 != SPEC_MODE_MAGIC || word & 0xFF00 != 0 {
        bail!("EP_CMD_SPEC_MODE: word {word:#010x} is not a mode word (desynchronised stream?)");
    }
    match SpecStepMode::from_u8((word & 0xFF) as u8) {
        Some(m) => Ok(m),
        None => bail!(
            "EP_CMD_SPEC_MODE: unknown mode {} in {word:#010x}",
            word & 0xFF
        ),
    }
}

/// 2026-10-10: `METRALE_HYBRID_DFLASH=1`, given together with `--dflash --speculative`: keep the
/// MTP head and the DFlash drafter both resident and pick the proposer per step (DFlash while
/// one grammarless sequence is active, MTP up to `METRALE_MTP_MAX_SEQS`, plain above). Both ranks
/// must set it. Read once.
pub fn hybrid_dflash_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::glm_dflash::parse_switch(std::env::var("METRALE_HYBRID_DFLASH").ok().as_deref())
    })
}

/// 2026-10-10: Default of [`hybrid_dflash_dwell_ms`]: policy D2s of the design note (e).
pub const HYBRID_DWELL_MS_DEFAULT: u64 = 2000;

/// 2026-10-10: `METRALE_HYBRID_DFLASH_DWELL_MS=<ms>` (default 2000): how long a single sequence
/// must have been alone, continuously, before the hybrid serve returns it to DFlash. Leaving
/// DFlash is immediate. Unparseable values keep the default. Read once.
pub fn hybrid_dflash_dwell_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        parse_dwell_ms(
            std::env::var("METRALE_HYBRID_DFLASH_DWELL_MS")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-10: The dwell value parser: a non-negative integer of milliseconds, else
/// [`HYBRID_DWELL_MS_DEFAULT`].
pub fn parse_dwell_ms(v: Option<&str>) -> u64 {
    v.and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(HYBRID_DWELL_MS_DEFAULT)
}

/// 2026-10-11: `METRALE_HYBRID_RATE_GATE=1` (policy D2sR, design note (f)): on top of D2s, demote a
/// sole sequence back to MTP when its DFlash wall rate falls below its own MTP wall rate, and bar
/// it from DFlash for a doubling cool-down. Inert without `METRALE_HYBRID_DFLASH=1`. Rank 0 only
/// (the mode is decided there). Read once.
pub fn hybrid_rate_gate_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::glm_dflash::parse_switch(std::env::var("METRALE_HYBRID_RATE_GATE").ok().as_deref())
    })
}

/// 2026-10-11: Default of [`hybrid_rate_margin_pct`].
pub const HYBRID_RATE_MARGIN_PCT_DEFAULT: u64 = 0;
/// 2026-10-11: Largest accepted [`hybrid_rate_margin_pct`].
pub const HYBRID_RATE_MARGIN_PCT_MAX: u64 = 50;

/// 2026-10-11: `METRALE_HYBRID_RATE_MARGIN_PCT=<0..50>` (default 0): the rate gate demotes when
/// `r_D < r_M * (1 - pct/100)`. Unparseable or out-of-range values keep the default. Read once.
pub fn hybrid_rate_margin_pct() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        parse_rate_margin_pct(
            std::env::var("METRALE_HYBRID_RATE_MARGIN_PCT")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-11: The margin parser: an integer in `0..=50`, else
/// [`HYBRID_RATE_MARGIN_PCT_DEFAULT`].
pub fn parse_rate_margin_pct(v: Option<&str>) -> u64 {
    v.and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|p| *p <= HYBRID_RATE_MARGIN_PCT_MAX)
        .unwrap_or(HYBRID_RATE_MARGIN_PCT_DEFAULT)
}

/// 2026-10-11: Default of [`hybrid_rate_cooldown_ms`].
pub const HYBRID_RATE_COOLDOWN_MS_DEFAULT: u64 = 8000;

/// 2026-10-11: `METRALE_HYBRID_RATE_COOLDOWN_MS=<ms>` (default 8000): the base cool-down `C_0` of
/// a demoted sequence; its k-th demotion bars DFlash for `C_0 * 2^k` (k capped at 6). Unparseable
/// values keep the default. Read once.
pub fn hybrid_rate_cooldown_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        parse_rate_cooldown_ms(
            std::env::var("METRALE_HYBRID_RATE_COOLDOWN_MS")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-11: The cool-down parser: a non-negative integer of milliseconds, else
/// [`HYBRID_RATE_COOLDOWN_MS_DEFAULT`].
pub fn parse_rate_cooldown_ms(v: Option<&str>) -> u64 {
    v.and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(HYBRID_RATE_COOLDOWN_MS_DEFAULT)
}

/// 2026-10-10: Bytes of one hybrid DFlash ctx accumulator (`DflashProposerState::ctx_hidden_acc`):
/// `window` rows of `taps` target hidden rows, BF16. A hybrid serve allocates exactly one, at the
/// full window on every rank (review F1), and reserves it before KV sizing.
pub fn hybrid_ctx_state_bytes(taps: usize, window: usize, hidden: usize) -> usize {
    taps * window * hidden * 2
}

/// 2026-10-10: The token window a hybrid serve sizes the DFlash drafter's paged KV and its KV
/// reserve by (design note (b)): `--max-seq-len` capped at `METRALE_DFLASH_CTX_CAP`
/// (`ctx_cap`, `0` = uncapped). The drafter attends over at most its ctx window, which the same
/// cap bounds, so the rest of a 512K `--max-seq-len` would be dead memory.
pub fn hybrid_drafter_window(max_seq_len: usize, ctx_cap: usize) -> usize {
    if ctx_cap == 0 {
        max_seq_len
    } else {
        max_seq_len.min(ctx_cap)
    }
}

/// 2026-10-14: `METRALE_HYBRID_MTP_CATCHUP=1` (#82 hybrid B2(i); inert without
/// `METRALE_HYBRID_DFLASH=1`): every hybrid DFlash verify also copies its committed target hidden
/// rows into a ring (`hybrid_ring`), on every rank, and the first MTP propose of the sequence
/// after its DFlash stretch feeds them to the MTP drafter, so the drafter resumes with the
/// stretch's context at the right positions. Both ranks must set it. Read once.
pub fn hybrid_mtp_catchup_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::glm_dflash::parse_switch(std::env::var("METRALE_HYBRID_MTP_CATCHUP").ok().as_deref())
    })
}

/// 2026-10-14: `METRALE_HYBRID_MTP_CATCHUP_ROWS=<rows>` (default
/// [`super::hybrid_ring::DEFAULT_RING_ROWS`]): the ring's row count, the longest DFlash stretch
/// whose every row reaches the MTP drafter. Read once.
pub fn hybrid_mtp_catchup_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        parse_catchup_rows(
            std::env::var("METRALE_HYBRID_MTP_CATCHUP_ROWS")
                .ok()
                .as_deref(),
        )
    })
}

/// 2026-10-14: The ring row parser: an integer in
/// `MIN_RING_ROWS..=MAX_RING_ROWS` (`hybrid_ring`), else the default.
pub fn parse_catchup_rows(v: Option<&str>) -> usize {
    use super::hybrid_ring::{DEFAULT_RING_ROWS, MAX_RING_ROWS, MIN_RING_ROWS};
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|r| (MIN_RING_ROWS..=MAX_RING_ROWS).contains(r))
        .unwrap_or(DEFAULT_RING_ROWS)
}

/// 2026-10-14: Device bytes of the hybrid MTP catch-up buffers a hybrid serve allocates after KV
/// sizing when the lever is on: the ring (`rows` BF16 hidden rows) and the zero rows the gap
/// fill reads (`hybrid_ring::ZERO_ROWS`). `0` with the lever off.
pub fn hybrid_mtp_catchup_bytes(lever_on: bool, rows: usize, hidden: usize) -> usize {
    if lever_on {
        (rows + super::hybrid_ring::ZERO_ROWS) * hidden * 2
    } else {
        0
    }
}
