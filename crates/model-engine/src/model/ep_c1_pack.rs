// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_EP_C1_PACK=1` (race-memory #80, C1 round): the per-command follow-up words of the two
//! commands every C=1 MTP round sends go as ONE broadcast each instead of one 4-byte broadcast per word:
//! - verify K=3/K=4 (`0xFFFFFFF3`/`0xFFFFFFF4`): the K tokens;
//! - MTP propose (`EP_CMD_MTP_PROPOSE`): token, position, num_drafts, hidden row.
//!
//! epdval (#80 6121666624) measured 34 us of round time per extra command word; this removes 5 words per round
//! (2 verify + 3 propose). The `(seq_id, cmd)` pair and `num_accepted` are unchanged.
//!
//! Owner: model-engine.
//! Invariants:
//! - Off unless the variable is exactly `1` (after trimming); read once per process.
//! - Both ranks must run the same setting (the serve passes one env to both containers), or the wire
//!   desynchronises: the head sends one `4 * n`-byte broadcast where the worker waits for n 4-byte ones.
//! - The words carry the same values either way, so model outputs are unchanged.

use std::sync::OnceLock;

/// 2026-10-12: Whether the variable's text turns the packing on.
pub(super) fn parse_c1_pack(v: Option<&str>) -> bool {
    v.is_some_and(|s| s.trim() == "1")
}

/// 2026-10-12: The process-wide setting.
pub(crate) fn ep_c1_pack() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_c1_pack(std::env::var("METRALE_EP_C1_PACK").ok().as_deref());
        if on {
            tracing::warn!(
                "METRALE_EP_C1_PACK=1: ENGAGED - verify tokens and MTP propose words go as one broadcast \
                 per command"
            );
        }
        on
    })
}

/// 2026-10-12: Most words one packed broadcast carries (`ep_cmd_buf` holds 64 bytes).
pub(super) const EP_PACK_MAX_WORDS: usize = 16;

#[cfg(test)]
mod tests {
    use super::{EP_PACK_MAX_WORDS, parse_c1_pack};

    #[test]
    fn parse_is_off_unless_one() {
        assert!(!parse_c1_pack(None));
        assert!(!parse_c1_pack(Some("")));
        assert!(!parse_c1_pack(Some("0")));
        assert!(!parse_c1_pack(Some("true")));
        assert!(!parse_c1_pack(Some("2")));
        assert!(parse_c1_pack(Some("1")));
        assert!(parse_c1_pack(Some(" 1 ")));
    }

    #[test]
    fn packet_fits_the_command_buffer() {
        assert_eq!(EP_PACK_MAX_WORDS * 4, 64);
    }
}
