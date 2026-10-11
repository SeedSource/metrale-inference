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
