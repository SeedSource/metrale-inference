// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The minimum-reasoning floor (`</think>` biased -8 for the first
//! `min_reasoning_floor_tokens` thinking tokens): 16 by default, 0 (off) for GLM-5.3, whose
//! official serve has no such floor (spark-bench c30gate TEBGAP-TRACE.md E).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{PositionKind, penalty_params_for};
use crate::scheduler::helpers::WatchdogParams;
use metrale_kernels::ModelBehavior;

const THINK_END: u32 = 4242;

/// 2026-10-15: The `</think>` bias `penalty_params_for` gives a sequence 3 tokens into
/// thinking (no budget) under `behavior`'s floor, as decode passes it (from
/// `WatchdogParams::from_behavior`).
fn think_end_bias(behavior: &ModelBehavior) -> Vec<f32> {
    let floor = WatchdogParams::from_behavior(behavior, None, None).min_reasoning_floor;
    let (mut a, _rx) = crate::scheduler::test_support::test_seq(vec![1, 2, 3], 100, None, 4);
    a.finished = false;
    a.inside_thinking = true;
    a.thinking_tokens = 3;
    a.thinking_budget = None;
    a.think_end_token = Some(THINK_END);
    penalty_params_for(&a, PositionKind::FinalDecode, 0.0, None, Vec::new(), floor)
        .logit_bias
        .iter()
        .filter(|(t, _)| *t == THINK_END)
        .map(|(_, b)| *b)
        .collect()
}

#[test]
fn the_floor_is_16_by_default_and_off_for_glm53() {
    let default = ModelBehavior::default();
    assert_eq!(default.min_reasoning_floor_tokens, 16);
    assert_eq!(think_end_bias(&default), [-8.0]);
    let glm = ModelBehavior {
        min_reasoning_floor_tokens: 0,
        ..ModelBehavior::default()
    };
    assert_eq!(
        WatchdogParams::from_behavior(&glm, None, None).min_reasoning_floor,
        0
    );
    assert!(think_end_bias(&glm).is_empty());
}
