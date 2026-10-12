// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Host tests of `ceiling_verify.rs`: the grouping condition table (one sequence's
//! rows on the replay-safe pool-cache path group; anything else falls back to the per-row
//! loop), and source checks that the row loop skips its per-row geometry write and selection
//! only under `group`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::{CeilingRowsFacts, ceiling_rows_group};

/// 2026-10-12: A C1 verify of 3 rows of one sequence, graph-captured, pool cache on.
fn one_seq(k: usize) -> CeilingRowsFacts {
    CeilingRowsFacts {
        lever: true,
        k,
        replay_safe: true,
        pool_cache: true,
        geom_pk: true,
        batch_select: false,
        meta_rows: Some(k),
        rows_ready: true,
    }
}

#[test]
fn one_sequences_verify_rows_group() {
    for k in [2, 3, 8, 16] {
        assert!(ceiling_rows_group(&one_seq(k)), "k = {k}");
    }
}

#[test]
fn a_single_row_or_the_lever_off_keeps_the_loop() {
    assert!(!ceiling_rows_group(&one_seq(1)));
    assert!(!ceiling_rows_group(&CeilingRowsFacts {
        lever: false,
        ..one_seq(3)
    }));
}

/// 2026-10-12: Metadata that describes more rows than this call (rows of several sequences,
/// e.g. a batched view whose `num_seqs` is not narrowed to the call), fewer rows, or none at
/// all: the call's row `r` is not provably row `r` of `positions` / `seq_len`, so no group.
#[test]
fn metadata_not_describing_exactly_these_rows_falls_back() {
    for meta_rows in [Some(6), Some(2), Some(1), None] {
        let f = CeilingRowsFacts {
            meta_rows,
            ..one_seq(3)
        };
        assert!(!ceiling_rows_group(&f), "{meta_rows:?}");
    }
}

#[test]
fn every_other_term_is_required() {
    let off: [fn(&mut CeilingRowsFacts); 5] = [
        |f| f.replay_safe = false,
        |f| f.pool_cache = false,
        |f| f.geom_pk = false,
        |f| f.batch_select = true,
        |f| f.rows_ready = false,
    ];
    for (i, clear) in off.iter().enumerate() {
        let mut f = one_seq(3);
        clear(&mut f);
        assert!(!ceiling_rows_group(&f), "term {i}");
    }
}

/// 2026-10-12: In the row loop, the per-row `dsa_write_geom_pk` and `select_row_at` are
/// skipped only under `group`, and the one pass runs after the loop, before the attend.
#[test]
fn the_row_loop_skips_per_row_selection_only_under_group() {
    let src = include_str!("decode_k.rs");
    let gate =
        "let group = self.ceiling_rows_for(st, call_meta, replay_safe, batch_select, src, k);";
    let loop_start = src.find("for row in 0..k {").expect("row loop");
    assert!(src.find(gate).expect("group gate") < loop_start);
    assert!(
        src.contains("            if group {\n                // 2026-10-12: The one geometry")
    );
    assert!(src.contains("} else if !group {\n                let q_resid_row"));
    let pass = src
        .find("if let Some(m) = call_meta.filter(|_| group) {")
        .expect("one pass after the loop");
    let attend = src.find("self.attend_rows_at(").expect("attend");
    assert!(loop_start < pass && pass < attend);
    assert_eq!(src.matches("select_rows_ceiling(").count(), 1);
}
