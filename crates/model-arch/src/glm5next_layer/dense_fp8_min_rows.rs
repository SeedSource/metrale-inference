// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: `METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS`: the fewest rows a
//! `METRALE_GLM_DENSE_FP8_W8A8` GEMM takes (`dense_fp8::w8a8`); narrower calls keep the dequant
//! path (FP8 copy dequantized to BF16 in the arena, then cuBLASLt).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset, the minimum is [`DEFAULT_MIN_ROWS`] (64), which is what `dense_fp8` used before
//!   this lever, so nothing changes.
//! - A value is clamped up to [`FLOOR_MIN_ROWS`] (17): 1..=16 rows belong to the decode and
//!   verify GEMVs (`dense_fp8::NV4_MAX_M`, `dense_gemv_tcm::TCM_MAX_M`,
//!   `DENSE_GEMV_FP8W_BATCHM_MAX_M`), which read the same per-row bits at every M; W8A8 there
//!   would make a verify row differ from the decode row. A value above 64 is allowed (it narrows
//!   W8A8); a non-integer warns once and keeps the default.
//! - Why lower it: a 17..=63-row GEMM (a prefix-cached follow-up turn, a short prompt, a prefill
//!   tail chunk, the MTP `eh_proj` in the drafter's row batch) on the dequant path moves about
//!   five bytes per weight element (FP8 read, BF16 write, BF16 read) against one for W8A8.
//!   NOT byte-identical for those calls (activation quantization, the same scheme as the
//!   64-row-and-up calls).

use std::sync::OnceLock;

/// 2026-10-06: The minimum when the lever is unset; the pre-lever `W8A8_MIN_ROWS`.
pub const DEFAULT_MIN_ROWS: usize = 64;

/// 2026-10-06: The lowest minimum the lever accepts: one past the widest decode/verify GEMV.
pub const FLOOR_MIN_ROWS: usize = 17;

/// 2026-10-06: Parse a lever value: `None` (unset or blank) is the default; an integer is
/// clamped to at least [`FLOOR_MIN_ROWS`]; anything else is `Err` (the caller warns and uses the
/// default).
pub fn parse_min_rows(raw: Option<&str>) -> Result<usize, ()> {
    match raw.map(str::trim) {
        None | Some("") => Ok(DEFAULT_MIN_ROWS),
        Some(v) => v
            .parse::<usize>()
            .map(|r| r.max(FLOOR_MIN_ROWS))
            .map_err(|_| ()),
    }
}

/// 2026-10-06: The W8A8 minimum row count for this process (read once).
pub fn w8a8_min_rows() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        let raw = std::env::var("METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS").ok();
        match parse_min_rows(raw.as_deref()) {
            Ok(r) => {
                if r != DEFAULT_MIN_ROWS {
                    tracing::warn!(
                        "METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS={} - W8A8 prefill GEMMs from {r} \
                         rows (default {DEFAULT_MIN_ROWS}, floor {FLOOR_MIN_ROWS}); NOT \
                         byte-identical for the calls it moves",
                        raw.as_deref().unwrap_or("").trim()
                    );
                }
                r
            }
            Err(()) => {
                tracing::warn!(
                    "METRALE_GLM_DENSE_FP8_W8A8_MIN_ROWS={} is not an integer - using \
                     {DEFAULT_MIN_ROWS}",
                    raw.as_deref().unwrap_or("")
                );
                DEFAULT_MIN_ROWS
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w8min_parse() {
        assert_eq!(parse_min_rows(None), Ok(64));
        assert_eq!(parse_min_rows(Some("  ")), Ok(64));
        assert_eq!(parse_min_rows(Some("17")), Ok(17));
        assert_eq!(parse_min_rows(Some("32")), Ok(32));
        assert_eq!(parse_min_rows(Some("1")), Ok(17));
        assert_eq!(parse_min_rows(Some("0")), Ok(17));
        assert_eq!(parse_min_rows(Some("128")), Ok(128));
        assert_eq!(parse_min_rows(Some("x")), Err(()));
        assert_eq!(parse_min_rows(Some("-5")), Err(()));
    }

    #[test]
    fn w8min_floor_is_past_every_gemv_tier() {
        assert!(FLOOR_MIN_ROWS > super::super::dense_fp8::NV4_MAX_M);
        assert!(
            FLOOR_MIN_ROWS > metrale_model_layers::layers::ops::dense_gemv_tcm::TCM_MAX_M as usize
        );
        assert!(
            FLOOR_MIN_ROWS
                > metrale_model_layers::layers::ops::DENSE_GEMV_FP8W_BATCHM_MAX_M as usize
        );
    }
}
