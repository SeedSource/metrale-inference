// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for the drafter carry: session admission, prefix
//! truncation, append plans, the ticketed hidden-row interval, carry arming,
//! and the snapshot restore threshold.
//!
//! Owner: model-layers (MTP drafter).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: The session of every `carried` fixture.
const SESSION: u64 = 0x5E55_1014;
/// 2026-09-25: Another session.
const OTHER: u64 = 0x0DD0_0DD0;

fn carried(tokens: &[u32], rows: usize, last_pair_key: Option<usize>) -> CarriedDrafter {
    CarriedDrafter {
        block_table: vec![1, 2, 3],
        rows,
        last_pair_key,
        tokens: tokens.to_vec(),
        session_hash: SESSION,
    }
}

/// 2026-09-25: Another session cannot adopt the entry even when the entry's
/// tokens are a full prefix of its prompt.
#[test]
fn a_foreign_session_cannot_adopt_the_slot() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), Some((4, 3)));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], OTHER), None);
}

/// 2026-09-25: A request with session 0 cannot adopt, and an entry left with
/// session 0 cannot be adopted by anyone.
#[test]
fn an_unstamped_request_cannot_adopt_the_slot() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], 0), None);
    let mut unstamped = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    unstamped.session_hash = 0;
    assert_eq!(unstamped.usable_by(&[1, 2, 3, 4, 5, 6, 7], 0), None);
    assert_eq!(unstamped.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), None);
}

/// 2026-09-25: Two requests share a 64-token template prefix. The prefix
/// alone would allow adoption, the session check refuses the other session,
/// and the same session still adopts the whole entry.
#[test]
fn the_shared_template_prefix_is_not_an_identity() {
    let template: Vec<u32> = (0..64).collect();
    let mut mine = template.clone();
    mine.extend_from_slice(&[900, 901, 902]);
    let mut theirs = template.clone();
    theirs.extend_from_slice(&[700, 701, 702]);

    let c = CarriedDrafter {
        block_table: vec![1, 2, 3],
        rows: 60,
        last_pair_key: Some(59),
        tokens: mine.clone(),
        session_hash: SESSION,
    };
    assert_eq!(c.common_prefix_len(&theirs), 64);
    assert_eq!(c.usable_by(&theirs, OTHER), None);
    assert_eq!(c.usable_by(&mine, SESSION), Some((60, 59)));
}

/// 2026-09-25: A configured carry is armed only without multi-sequence
/// dispatch, and a carry configured off is never armed.
#[test]
fn configured_carry_is_not_armed_carry_under_a_multi_seq_cap() {
    use crate::drafter_context::DrafterContext;
    let both = DrafterContext {
        prefill: true,
        carry: true,
    };
    let prefill_only = DrafterContext {
        prefill: true,
        carry: false,
    };
    assert!(carry_armed_with(both, false));
    assert!(
        !carry_armed_with(both, true),
        "a >1 dispatch cap must force the carry off"
    );
    assert!(!carry_armed_with(prefill_only, false));
    assert!(!carry_armed_with(prefill_only, true));
}

/// 2026-09-25: A's last prefill chunk lands after B has written its own rows.
/// The late write takes the interval over instead of merging into B's, and A
/// then reads only its own rows.
#[test]
fn a_late_chunk_takes_the_interval_over_instead_of_claiming_foreign_rows() {
    // 2026-09-25: Negative control: `merge_interval` alone would merge A's
    // write into B's rows 0..600.
    let unstamped = merge_interval(merge_interval((0, 0), 0, 600), 500, 500);
    assert_eq!(
        unstamped,
        (0, 1000),
        "control: without a stamp the merge claims B's rows 0..600"
    );

    let a0 = stamped_merge(StoreRange::EMPTY, 1, 0, 500);
    assert_eq!(
        a0,
        StoreRange {
            owner: 1,
            lo: 0,
            hi: 500
        }
    );
    let b0 = stamped_merge(StoreRange::EMPTY, 2, 0, 600);
    assert_eq!(
        b0,
        StoreRange {
            owner: 2,
            lo: 0,
            hi: 600
        }
    );
    let a1 = stamped_merge(b0, 1, 500, 500);
    assert_eq!(
        a1,
        StoreRange {
            owner: 1,
            lo: 500,
            hi: 1000
        },
        "a foreign interval must be taken over, never extended"
    );
    assert_eq!(a1.visible_to(1), (500, 1000));
}

/// 2026-09-25: A wrote rows [12000, 12100), then B wrote [0, 13000) and owns
/// the interval. A's append plan is refused.
#[test]
fn a_foreign_interval_cannot_satisfy_an_append_plan() {
    let a = stamped_merge(StoreRange::EMPTY, 1, 12000, 100);
    let b = stamped_merge(a, 2, 0, 13000);
    assert_eq!(
        b,
        StoreRange {
            owner: 2,
            lo: 0,
            hi: 13000
        }
    );

    // 2026-09-25: Negative control: B's whole interval would satisfy the plan.
    assert_eq!(
        plan_append(11999, 12100, 0, 13000),
        Some(AppendPlan {
            first_key: 12000,
            rows: 99
        }),
        "control: the unstamped read satisfies the plan with B's rows"
    );

    assert_eq!(b.visible_to(1), (0, 0));
    let (lo, hi) = b.visible_to(1);
    assert_eq!(
        plan_append(11999, 12100, lo, hi),
        None,
        "a foreign interval must not satisfy an append"
    );
}

/// 2026-09-25: The owner's own append still succeeds, with the numbers of the
/// test above.
#[test]
fn a_sequence_still_appends_from_its_own_rows() {
    let a = stamped_merge(StoreRange::EMPTY, 1, 12000, 100);
    let (lo, hi) = a.visible_to(1);
    assert_eq!((lo, hi), (12000, 12100));
    assert_eq!(
        plan_append(11999, 12100, lo, hi),
        Some(AppendPlan {
            first_key: 12000,
            rows: 99
        }),
        "the owner's own append must be unaffected"
    );
}

/// 2026-09-25: One sequence's consecutive chunks merge into one interval.
#[test]
fn consecutive_chunks_of_one_sequence_still_merge() {
    let r = stamped_merge(stamped_merge(StoreRange::EMPTY, 1, 0, 500), 1, 500, 500);
    assert_eq!(
        r,
        StoreRange {
            owner: 1,
            lo: 0,
            hi: 1000
        }
    );
}

/// 2026-09-25: Ticket 0 matches nothing: a reader with ticket 0 sees nothing,
/// nobody reads an unclaimed range, and a writer with ticket 0 leaves the
/// range unclaimed.
#[test]
fn generation_zero_never_matches_anything() {
    let claimed = StoreRange {
        owner: 7,
        lo: 0,
        hi: 9999,
    };
    assert_eq!(
        claimed.visible_to(0),
        (0, 0),
        "a ticketless reader sees nothing"
    );
    let unclaimed = StoreRange {
        owner: 0,
        lo: 0,
        hi: 9999,
    };
    assert_eq!(unclaimed.visible_to(0), (0, 0), "0 does not match 0");
    assert_eq!(
        unclaimed.visible_to(7),
        (0, 0),
        "and nobody owns an unclaimed range"
    );
    assert_eq!(stamped_merge(StoreRange::EMPTY, 0, 0, 10).owner, 0);
}

/// 2026-09-25: `ForeignHiddens` and `NoHiddens` print differently.
#[test]
fn foreign_hiddens_does_not_read_like_a_coverage_miss() {
    let foreign = CarryOutcome::ForeignHiddens {
        owner: 2,
        expected: 1,
    }
    .to_string();
    assert!(foreign.contains("belong to sequence gen 2"), "{foreign}");
    assert_ne!(foreign, CarryOutcome::NoHiddens.to_string());
}

#[test]
fn session_matches_is_equality_and_refuses_zero() {
    let c = carried(&[1, 2, 3], 2, Some(1));
    assert!(c.session_matches(SESSION));
    assert!(!c.session_matches(OTHER));
    assert!(!c.session_matches(0), "zero is unknown, not wildcard");
}

/// 2026-09-25: `ForeignSession` and `PrefixMismatch` print differently.
#[test]
fn the_two_refusals_do_not_read_alike() {
    let foreign = CarryOutcome::ForeignSession {
        entry_session: SESSION,
        prompt_session: OTHER,
    }
    .to_string();
    let mismatch = CarryOutcome::PrefixMismatch {
        common: 4,
        entry_rows: 9,
    }
    .to_string();
    assert!(foreign.contains("foreign session"), "{foreign}");
    assert!(!foreign.contains("prefix mismatch"), "{foreign}");
    assert!(mismatch.contains("prefix mismatch"), "{mismatch}");
    assert_ne!(foreign, mismatch);
}

#[test]
fn usable_by_keeps_everything_when_the_whole_entry_matches() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    // 2026-09-25: Pair key 3 consumed tokens[0..=4], and all 5 match.
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 5, 6, 7], SESSION), Some((4, 3)));
}

#[test]
fn usable_by_truncates_the_tail_instead_of_refusing() {
    // 2026-09-25: Divergence at index 4 gives common 4, so the highest usable
    // key is 2 and one row is dropped.
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    assert_eq!(c.usable_by(&[1, 2, 3, 4, 9, 6, 7], SESSION), Some((3, 2)));
    // 2026-09-25: Divergence at index 2 gives common 2: only key 0 survives.
    assert_eq!(c.usable_by(&[1, 2, 9, 9], SESSION), Some((1, 0)));
}

#[test]
fn usable_by_declines_when_nothing_survives() {
    let c = carried(&[1, 2, 3, 4, 5], 4, Some(3));
    // 2026-09-25: Fewer than 2 common tokens: not even pair key 0 is usable.
    assert_eq!(c.usable_by(&[1, 9, 9], SESSION), None);
    assert_eq!(c.usable_by(&[], SESSION), None);
    assert_eq!(
        carried(&[1, 2, 3], 0, Some(1)).usable_by(&[1, 2, 3, 4], SESSION),
        None
    );
    assert_eq!(
        carried(&[1, 2, 3], 2, None).usable_by(&[1, 2, 3, 4], SESSION),
        None
    );
}

#[test]
fn usable_by_never_drops_more_rows_than_exist() {
    // 2026-09-25: 2 rows but key 4. Truncating to a low common prefix drops
    // more keys than rows, so it declines instead of underflowing.
    let c = carried(&[1, 2, 3, 4, 5, 6], 2, Some(4));
    assert_eq!(c.usable_by(&[1, 2, 9], SESSION), None);
}

#[test]
fn append_plan_covers_exactly_the_missing_pair_keys() {
    // 2026-09-25: The drafter holds keys up to 97; a 200-token prompt needs
    // keys up to 198; the hidden store covers [97, 200).
    let p = plan_append(97, 200, 97, 200).unwrap();
    assert_eq!(
        p,
        AppendPlan {
            first_key: 98,
            rows: 101
        }
    );
}

#[test]
fn append_plan_clamps_up_to_the_hidden_store_floor() {
    // 2026-09-25: The store starts at 150, so keys 98..149 are skipped.
    let p = plan_append(97, 200, 150, 200).unwrap();
    assert_eq!(
        p,
        AppendPlan {
            first_key: 150,
            rows: 49
        }
    );
}

#[test]
fn append_plan_declines_when_the_store_stops_short_of_the_last_key() {
    // 2026-09-25: Hidden row 198 is needed; the store ends before it.
    assert_eq!(plan_append(97, 200, 97, 190), None);
    assert_eq!(plan_append(97, 200, 97, 198), None);
}

#[test]
fn append_plan_declines_when_nothing_is_missing() {
    assert_eq!(plan_append(198, 200, 0, 200), None);
    assert_eq!(plan_append(250, 200, 0, 200), None);
}

#[test]
fn append_plan_declines_on_a_degenerate_prompt() {
    assert_eq!(plan_append(0, 1, 0, 8), None);
    assert_eq!(plan_append(0, 0, 0, 8), None);
}

#[test]
fn merge_interval_extends_on_overlap_and_abut() {
    assert_eq!(merge_interval((10, 20), 20, 5), (10, 25));
    assert_eq!(merge_interval((10, 20), 15, 10), (10, 25));
    assert_eq!(merge_interval((10, 20), 5, 6), (5, 20));
}

#[test]
fn merge_interval_replaces_on_a_gap() {
    // 2026-09-25: A disjoint write does not claim the gap.
    assert_eq!(merge_interval((10, 20), 30, 5), (30, 35));
    assert_eq!(merge_interval((10, 20), 0, 5), (0, 5));
    assert_eq!(merge_interval((0, 0), 30, 5), (30, 35));
}

#[test]
fn common_prefix_len_is_the_validity_primitive() {
    let c = carried(&[1, 2, 3, 4], 3, Some(2));
    assert_eq!(c.common_prefix_len(&[1, 2, 3, 4, 5]), 4);
    assert_eq!(c.common_prefix_len(&[1, 2, 9, 4, 5]), 2);
    assert_eq!(c.common_prefix_len(&[]), 0);
}

/// 2026-09-25: The store ticket (`SequenceState::mtp_store_gen`) is drawn from
/// its own counter, not from `mtp_prefill_capture_gen`. `owns_capture`
/// requires the capture generation to be unchanged between a sequence's
/// capture and its propose, so drawing from it at every admission would skip
/// the drafter prefill of a sequence when another is admitted in between.
/// Checked against the source of the model's `meta.rs`, since running
/// `alloc_sequence` needs a whole model.
#[test]
fn the_store_ticket_never_draws_from_the_capture_generation() {
    let meta = include_str!("../../model-engine/src/model/trait_impl/meta.rs");
    let draw = meta
        .lines()
        .find(|l| l.contains("let store_gen ="))
        .expect("alloc_sequence must draw a store ticket");
    assert!(
        draw.contains("mtp_store_gen_seq"),
        "the store ticket must come from its own dispenser, got: {draw}"
    );
    assert!(
        !draw.contains("mtp_prefill_capture_gen"),
        "sharing the capture counter disables drafter prefill for any sequence \
         admitted between a capture and its propose: {draw}"
    );
}

/// 2026-09-25: `DEFAULT_MARCONI_MIN_TOKENS` is 256, and the test process sets
/// no `METRALE_MARCONI_MIN_TOKENS`. It repeats the reader's fallback expression
/// instead of calling the reader, whose value another test may already have
/// fixed.
#[test]
fn the_advertised_default_is_the_one_the_reader_falls_back_to() {
    assert_eq!(DEFAULT_MARCONI_MIN_TOKENS, 256);
    let fallback = std::env::var("METRALE_MARCONI_MIN_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MARCONI_MIN_TOKENS);
    assert_eq!(
        fallback, DEFAULT_MARCONI_MIN_TOKENS,
        "no env override in tests"
    );
}

/// 2026-09-25: A second `set_marconi_min_tokens` returns false without
/// panicking, and the value is stable once read. Which call wins depends on
/// test order.
#[test]
fn setting_the_threshold_twice_reports_the_loss_rather_than_panicking() {
    let first = set_marconi_min_tokens(4096);
    assert!(
        !set_marconi_min_tokens(8192),
        "a second set must report the loss"
    );
    let resolved = marconi_min_tokens();
    if first {
        assert_eq!(resolved, 4096, "the winner's value is what readers see");
    }
    assert_eq!(resolved, marconi_min_tokens(), "stable once read");
}

#[test]
fn cap_adopted_drops_keys_past_the_grid_restore_point() {
    // 2026-10-03: Under the cap: unchanged.
    assert_eq!(cap_adopted(100, 99, 200), Some((100, 99)));
    assert_eq!(cap_adopted(100, 99, 99), Some((100, 99)));
    // 2026-10-03: 32250 carried keys (rows 32251), restore at 24576: keys up to 24575 survive.
    assert_eq!(cap_adopted(32251, 32250, 24575), Some((24576, 24575)));
    // 2026-10-03: Nothing survives.
    assert_eq!(cap_adopted(10, 50, 40), None);
    assert_eq!(cap_adopted(10, 50, 39), None);
}
