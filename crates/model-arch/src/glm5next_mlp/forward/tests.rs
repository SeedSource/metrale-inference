// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `forward`: the row-group split and the workspace sizing.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::{MOE_ROW_BATCH_MAX_ROWS, moe_row_groups};

/// 2026-09-25: For every cap up to `MOE_ROW_BATCH_MAX_ROWS` and 1 to 64 rows, the groups cover
/// every row once, in order, each at most `cap` wide, and at the full cap none is one row wide
/// when `rows >= 2`.
#[test]
fn row_groups_cover_and_never_orphan_a_row() {
    for cap in 1..=MOE_ROW_BATCH_MAX_ROWS {
        for rows in 1..=64 {
            let g = moe_row_groups(rows, cap);
            assert_eq!(g[0].0, 0, "rows={rows} cap={cap}: does not start at 0");
            let mut next = 0;
            for &(start, w) in &g {
                assert_eq!(start, next, "rows={rows} cap={cap}: gap or overlap");
                assert!(w >= 1 && w <= cap, "rows={rows} cap={cap}: width {w}");
                if rows >= 2 && cap == MOE_ROW_BATCH_MAX_ROWS {
                    assert!(w >= 2, "rows={rows} cap={cap}: orphaned a single row");
                }
                next += w;
            }
            assert_eq!(next, rows, "rows={rows} cap={cap}: {next} rows covered");
        }
    }
}

#[test]
fn row_groups_at_the_shipping_widths() {
    assert_eq!(moe_row_groups(16, 8), vec![(0, 8), (8, 8)]);
    assert_eq!(moe_row_groups(8, 8), vec![(0, 8)]);
    assert_eq!(moe_row_groups(9, 8), vec![(0, 5), (5, 4)]);
}

mod ws_sizing {
    use crate::glm5next_mlp::Glm5NextMlpConfig;
    use crate::glm5next_mlp::forward::{mlp_ws_bytes, mlp_ws_permute_bytes, mlp_ws_total_bytes};

    /// 2026-09-25: The config fixture's MLP geometry (hidden 4096, 288 experts, top_k 8) at
    /// TP=2, EP=2.
    fn cfg() -> Glm5NextMlpConfig {
        Glm5NextMlpConfig {
            hidden: 4096,
            local_dense_intermediate: 12288 / 2,
            moe_intermediate: 2048,
            local_shared_intermediate: 2048 / 2,
            num_experts: 288,
            local_experts: 144,
            ep_rank: 0,
            top_k: 8,
            routed_scale: 2.5,
            renormalize: true,
            swiglu_limit: 10.0,
            router_bf16_ladder: false,
            tp_world_size: 2,
            ep_world_size: 2,
        }
    }

    /// 2026-09-25: `mlp_ws_bytes` against hand-computed sizes at 16 to 1024 rows. `new` is
    /// not called, so a change to `new` alone does not fail this test.
    #[test]
    fn matches_the_hand_computed_campaign_footprint() {
        let c = cfg();
        for rows in [16usize, 64, 128, 256, 512, 1024] {
            let b = mlp_ws_bytes(&c, rows);
            assert_eq!(b[0], rows * 16384 * 2, "a_gate at {rows}");
            assert_eq!(b[1], b[0], "a_up at {rows}");
            assert_eq!(b[2], b[0], "a_act at {rows}");
            assert_eq!(b[3], rows * 288 * 4, "logits at {rows}");
            assert_eq!(b[6], rows * 8 * 4096 * 2, "expert_out at {rows}");
            assert_eq!(b[7], rows * 4096 * 2, "shared_out at {rows}");
            assert_eq!(b[9], rows * 8 * rows * 4, "u_slot at {rows}");
            assert_eq!(b[12], 289 * 4, "expert_offsets is row-independent");
            assert_eq!(
                mlp_ws_total_bytes(&c, rows),
                rows * 173_376 + 32 * rows * rows + 1156
            );
        }
    }

    /// 2026-09-25: One workspace against 45 per-layer copies at 256, 512 and 1024 rows, in
    /// decimal MB and GB.
    #[test]
    fn sharing_one_workspace_saves_44_of_45_copies() {
        let c = cfg();
        // 2026-09-25: Decimal units, as the loader's log prints them.
        let mb = |n: usize| n as f64 / 1e6;
        for (rows, per_layer_mb, stack_gb) in [
            (256usize, 46.48, 2.092),
            (512, 97.16, 4.372),
            (1024, 211.09, 9.499),
        ] {
            let one = mlp_ws_total_bytes(&c, rows);
            assert!(
                (mb(one) - per_layer_mb).abs() < 0.05,
                "rows={rows}: {:.2} MB per workspace, expected ≈{per_layer_mb}",
                mb(one)
            );
            assert!(
                (mb(one * 45) / 1000.0 - stack_gb).abs() < 0.01,
                "rows={rows}: {:.3} GB for 45 private copies, expected ≈{stack_gb}",
                mb(one * 45) / 1000.0
            );
        }
    }

    /// 2026-09-25: `mlp_ws_bytes` treats 0 rows as 1, as `new` does, so no buffer size is 0.
    #[test]
    fn zero_rows_clamps_to_one() {
        let c = cfg();
        assert_eq!(mlp_ws_total_bytes(&c, 0), mlp_ws_total_bytes(&c, 1));
        assert!(mlp_ws_bytes(&c, 0).iter().all(|&b| b > 0));
    }

    /// 2026-09-30: `mlp_ws_permute_bytes` (the `METRALE_GLM_MOE_PREFILL_PERMUTE=1` gather-once
    /// buffer): 0 with `on = false` at every row count including 0, and `rows * top_k * hidden *
    /// 2` with `on = true`, 0 rows clamped to 1 like every other buffer in this module.
    #[test]
    fn permute_bytes_is_zero_off_and_rows_top_k_hidden_times_two_on() {
        let c = cfg();
        for rows in [0usize, 1, 16, 64, 128, 256, 1024] {
            assert_eq!(
                mlp_ws_permute_bytes(&c, rows, false),
                0,
                "lever off must allocate nothing at {rows} rows"
            );
            assert_eq!(
                mlp_ws_permute_bytes(&c, rows, true),
                rows.max(1) * c.top_k * c.hidden * 2,
                "lever on at {rows} rows"
            );
        }
    }
}
