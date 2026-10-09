// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `DFlashLevers` resolution: each switch's polarity, the
//! numeric fallbacks, and presence-not-truth for the graph kill switch.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::*;
use std::collections::HashMap;

/// 2026-09-25: Resolve against a fixed map instead of the process environment.
/// `set_var` is `unsafe` and process-global. `from_env` is `from_values` with
/// closures over `metrale_config::levers::{var, var_os}`, so this runs the
/// production resolution.
fn resolve(vars: &[(&str, &str)]) -> DFlashLevers {
    let map: HashMap<&str, &str> = vars.iter().copied().collect();
    from_values(
        |var| map.get(var).map(|v| (*v).to_string()),
        |var| map.contains_key(var),
    )
}

#[test]
fn nothing_set_resolves_to_defaults() {
    assert_eq!(resolve(&[]), DFlashLevers::defaults());
}

#[test]
fn defaults_are_spelled_out_not_derived() {
    let d = DFlashLevers::defaults();
    // 2026-09-25: Every field whose unset value is not its type's `Default`,
    // then `any_diagnostic_armed`.
    assert_eq!(d.propose_warmup_n, 2);
    assert_eq!(d.batch_propose_width, usize::MAX);
    assert!(d.dspark_anchor_bias);
    assert!(d.option_b);
    assert!(d.dflash2);
    assert!(d.dspark_markov);
    assert!(!d.any_diagnostic_armed);
}

#[test]
fn every_opt_in_is_off_until_it_is_exactly_one() {
    // 2026-09-25: `=1` arms and no other spelling does. One row per field, so
    // a field read from the wrong variable fails its own row.
    let cases: [(&str, fn(&DFlashLevers) -> bool); 17] = [
        ("METRALE_DFLASH_DEBUG_DUMP", |l| l.debug_dump),
        ("METRALE_DFLASH_DEBUG_DUMP_FULL", |l| l.debug_dump_full),
        ("METRALE_DFLASH_LOG_DRAFTS", |l| l.log_drafts),
        ("METRALE_DFLASH_BLOCK_DUMP", |l| l.block_dump),
        ("METRALE_DFLASH_OPTION_B_DIAG", |l| l.option_b_diag),
        ("METRALE_DFLASH_DEBUG_FORCE_PATTERN", |l| l.force_pattern),
        ("METRALE_DFLASH_PRECOMPUTE", |l| l.precompute),
        ("METRALE_DSPARK_CONF_TRACE", |l| l.dspark_conf_trace),
        ("METRALE_DFLASH_OPTION_B_NO_CTX", |l| l.option_b_no_ctx),
        ("METRALE_DFLASH_VERIFY_TRACE", |l| l.verify_trace),
        ("METRALE_DFLASH_PRECOMPUTE_DUMP", |l| l.precompute_dump),
        ("METRALE_DFLASH_CTX_PARITY_DUMP", |l| l.ctx_parity_dump),
        ("METRALE_DFLASH_DEBUG_FULL_PRECOMPUTE", |l| {
            l.full_precompute
        }),
        ("METRALE_DFLASH_CTXLEN_PROBE", |l| l.ctxlen_probe),
        ("METRALE_DFLASH_CONF_LOG", |l| l.conf_log),
        ("METRALE_DFLASH_DRAFT_GEMV", |l| l.draft_gemv),
        ("METRALE_DFLASH_DRAFT_TP2", |l| l.draft_tp2),
    ];
    for (var, read) in cases {
        assert!(!read(&resolve(&[])), "{var} armed with nothing set");
        assert!(read(&resolve(&[(var, "1")])), "{var} did not arm at =1");
        assert!(!read(&resolve(&[(var, "0")])), "{var} armed at =0");
        assert!(
            !read(&resolve(&[(var, "true")])),
            "{var} armed at =true; these levers are strict `1`"
        );
    }
}

/// 2026-09-25: Every opt-out lever is on when unset and only an exact `0`
/// turns it off.
#[test]
fn every_opt_out_ships_on_and_only_zero_disables_it() {
    let cases: [(&str, fn(&DFlashLevers) -> bool); 4] = [
        ("METRALE_DSPARK_ANCHOR_BIAS", |l| l.dspark_anchor_bias),
        ("METRALE_DFLASH_OPTION_B", |l| l.option_b),
        ("METRALE_DFLASH2", |l| l.dflash2),
        ("METRALE_DSPARK_MARKOV", |l| l.dspark_markov),
    ];
    for (var, read) in cases {
        assert!(read(&resolve(&[])), "{var} must ship ON");
        assert!(read(&resolve(&[(var, "1")])), "{var} off at =1");
        assert!(
            !read(&resolve(&[(var, "0")])),
            "{var} is not disabled at =0"
        );
        assert!(
            read(&resolve(&[(var, "")])),
            "{var}: empty is not a kill switch"
        );
    }
}

#[test]
fn the_numeric_path_levers_default_to_unbounded() {
    // 2026-09-25: Unset is unbounded (`usize::MAX`, `None`), not zero.
    assert_eq!(resolve(&[]).batch_propose_width, usize::MAX);
    assert_eq!(resolve(&[]).draft_cap, None);
    assert_eq!(
        resolve(&[("METRALE_DFLASH_BATCH_PROPOSE", "2")]).batch_propose_width,
        2
    );
    assert_eq!(
        resolve(&[("METRALE_DFLASH_DRAFT_CAP", "1")]).draft_cap,
        Some(1)
    );
}

#[test]
fn conf_tau_is_off_until_a_positive_threshold_is_given() {
    // 2026-09-25: An unparseable value falls back to 0.0, which leaves the
    // confidence head off (`confidence_active` needs a positive threshold).
    assert_eq!(resolve(&[]).conf_tau, 0.0);
    assert_eq!(
        resolve(&[("METRALE_DSPARK_CONF_TAU", "junk")]).conf_tau,
        0.0
    );
    assert_eq!(resolve(&[("METRALE_DSPARK_CONF_TAU", "0.7")]).conf_tau, 0.7);
}

#[test]
fn dflash_conf_tau_is_off_outside_the_open_unit_interval() {
    // 2026-10-09: Only a finite `t` with 0 < t < 1 engages; everything else is 0.0 (off).
    assert_eq!(resolve(&[]).dflash_conf_tau, 0.0);
    for bad in ["junk", "0", "1", "1.5", "-0.2", "NaN", "inf", ""] {
        assert_eq!(
            resolve(&[("METRALE_DFLASH_CONF_TAU", bad)]).dflash_conf_tau,
            0.0,
            "{bad:?} must leave the lever off"
        );
    }
    assert_eq!(
        resolve(&[("METRALE_DFLASH_CONF_TAU", "0.4")]).dflash_conf_tau,
        0.4
    );
    // 2026-10-09: Independent of the log lever.
    assert!(!resolve(&[("METRALE_DFLASH_CONF_TAU", "0.4")]).conf_log);
}

#[test]
fn dspark_shift_defers_to_the_checkpoint_unless_spelled() {
    assert_eq!(resolve(&[]).dspark_shift, None);
    assert_eq!(
        resolve(&[("METRALE_DSPARK_SHIFT", "1")]).dspark_shift,
        Some(true)
    );
    assert_eq!(
        resolve(&[("METRALE_DSPARK_SHIFT", "0")]).dspark_shift,
        Some(false)
    );
    // 2026-09-25: Any other value is not an override; the drafter config decides.
    assert_eq!(
        resolve(&[("METRALE_DSPARK_SHIFT", "yes")]).dspark_shift,
        None
    );
}

#[test]
fn numeric_levers_fall_back_when_unparseable() {
    assert_eq!(
        resolve(&[("METRALE_DFLASH_PROPOSE_WARMUP_N", "5")]).propose_warmup_n,
        5
    );
    assert_eq!(
        resolve(&[("METRALE_DFLASH_PROPOSE_WARMUP_N", "x")]).propose_warmup_n,
        2
    );
    assert_eq!(
        resolve(&[("METRALE_DFLASH_BLOCK_DUMP_AT_POS", "64")]).block_dump_at_pos,
        64
    );
    assert_eq!(
        resolve(&[("METRALE_DFLASH_DEBUG_CTX_USED", "7")]).force_ctx_used,
        Some(7)
    );
    assert_eq!(
        resolve(&[("METRALE_DFLASH_DEBUG_CTX_USED", "-1")]).force_ctx_used,
        None
    );
}

/// 2026-09-25: A graph-suppressing variable set to any value, `0` and empty
/// included, sets `any_diagnostic_armed` although it arms no diagnostic.
#[test]
fn a_diagnostic_set_to_zero_still_suppresses_graphs() {
    for var in GRAPH_SUPPRESSING_DIAGNOSTICS {
        assert!(
            resolve(&[(var, "0")]).any_diagnostic_armed,
            "{var}=0 must still force the eager path"
        );
        assert!(
            resolve(&[(var, "")]).any_diagnostic_armed,
            "{var}= (empty) must still force the eager path"
        );
    }
    // 2026-09-25: Variables outside the list do not: `METRALE_DFLASH2`,
    // `METRALE_DFLASH_OPTION_B` and the warm-up count are not diagnostics.
    assert!(!resolve(&[("METRALE_DFLASH2", "0")]).any_diagnostic_armed);
    assert!(!resolve(&[("METRALE_DFLASH_OPTION_B", "0")]).any_diagnostic_armed);
    assert!(!resolve(&[("METRALE_DFLASH_PROPOSE_WARMUP_N", "4")]).any_diagnostic_armed);
}

#[test]
fn the_block_dump_arms_only_at_or_past_its_position() {
    let armed = resolve(&[
        ("METRALE_DFLASH_BLOCK_DUMP", "1"),
        ("METRALE_DFLASH_BLOCK_DUMP_AT_POS", "64"),
    ]);
    assert!(!armed.block_dump_armed_at(63));
    assert!(armed.block_dump_armed_at(64));
    assert!(armed.block_dump_armed_at(65));
    let off = resolve(&[("METRALE_DFLASH_BLOCK_DUMP_AT_POS", "0")]);
    assert!(!off.block_dump_armed_at(1_000_000));
}
