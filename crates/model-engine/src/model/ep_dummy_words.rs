// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_EP_DUMMY_WORDS=N`: a slope test, not a serving path (race-memory #80, C1 round). After
//! every worker command (`ep_broadcast_seq_and_cmd` on rank 0, `ep_recv_seq_and_cmd` on the worker), both ranks
//! exchange N extra 4-byte broadcasts whose value is ignored. The round time per added word says whether the
//! command-word chain sits on the C1 critical path, before any protocol packing is built.
//!
//! Owner: model-engine.
//! Invariants:
//! - Off (0) unless the variable parses as an integer > 0; the value is read once per process.
//! - Both ranks must run the same N (the serve passes one env to both containers), or the wire desynchronises.
//! - The words carry 0 and nothing reads them, so model outputs are unchanged.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// 2026-10-12: N from the variable's text; anything that is not an integer > 0 is 0 (off). Capped at 64.
pub(super) fn parse_dummy_words(v: Option<&str>) -> u32 {
    v.and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0)
        .min(64)
}

/// 2026-10-12: The process-wide N (0 = off).
pub(super) fn ep_dummy_words() -> u32 {
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        let n = parse_dummy_words(std::env::var("METRALE_EP_DUMMY_WORDS").ok().as_deref());
        if n > 0 {
            tracing::warn!(
                "METRALE_EP_DUMMY_WORDS={n}: ENGAGED - {n} extra 4-byte broadcasts after every worker command \
                 (slope test, not a serving path)"
            );
        }
        n
    })
}

/// 2026-10-12: Counts commands that carried dummy words; logs every 1024th so a run can derive commands per
/// round (`EP_DUMMY_WORDS commands=...`). Called only when N > 0.
pub(super) fn note_command() {
    static C: AtomicU64 = AtomicU64::new(0);
    let c = C.fetch_add(1, Ordering::Relaxed) + 1;
    if c % 1024 == 0 {
        tracing::info!(
            "EP_DUMMY_WORDS commands={c} words={}",
            c * u64::from(ep_dummy_words())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::parse_dummy_words;

    #[test]
    fn parse_is_off_unless_positive_integer() {
        assert_eq!(parse_dummy_words(None), 0);
        assert_eq!(parse_dummy_words(Some("")), 0);
        assert_eq!(parse_dummy_words(Some("0")), 0);
        assert_eq!(parse_dummy_words(Some("abc")), 0);
        assert_eq!(parse_dummy_words(Some("-3")), 0);
        assert_eq!(parse_dummy_words(Some("4")), 4);
        assert_eq!(parse_dummy_words(Some(" 8 ")), 8);
        assert_eq!(parse_dummy_words(Some("1000")), 64);
    }
}
