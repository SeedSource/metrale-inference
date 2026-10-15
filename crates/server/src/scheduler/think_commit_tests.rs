// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The thinking-loop watchdog under MODEL.toml
//! `[behavior].enable_think_loop_watchdog`: on by default, off for GLM-5.3, whose official
//! serve has no such watchdog (TEB TC-88: c30 forced `</think>` twice per trial on the
//! model's own digit list; spark-bench c30gate TEBGAP-TRACE.md B).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::{ThinkTokenEnv, advance_thinking_token};
use crate::scheduler::helpers::WatchdogParams;
use metrale_kernels::ModelBehavior;

/// 2026-10-15: GLM-5.3's watchdog setting as `kernels/gb10/glm-5.3-flash/MODEL.toml` sets it.
fn glm53_behavior() -> ModelBehavior {
    ModelBehavior {
        enable_think_loop_watchdog: false,
        ..ModelBehavior::default()
    }
}

/// 2026-10-15: A TC-88 style thinking tail: the model writing out its own long digit list,
/// which repeats one 12-token window ("1, 2, 3, ..."-like ids) 40 times. Commits each token
/// as decode does (advance, then push) and returns (`</think>` armed, watchdog fires).
fn think_digit_list(behavior: &ModelBehavior, disable_watchdogs: bool) -> (bool, u32) {
    let wp = WatchdogParams::from_behavior(behavior, None, None);
    let env = ThinkTokenEnv::new(None, disable_watchdogs, wp);
    let (mut a, _rx) = crate::scheduler::test_support::test_seq(Vec::new(), 100_000, None, 4);
    a.finished = false;
    a.thinking_budget = None;
    let window: Vec<u32> = (0..12).map(|i| 1000 + i).collect();
    for _ in 0..40 {
        for &t in &window {
            let history_len = a.output_tokens.len();
            advance_thinking_token(&mut a, t, history_len, env, false);
            a.output_tokens.push(t);
        }
    }
    (a.force_end_thinking, a.think_watchdog_fires)
}

#[test]
fn glm53_behavior_turns_the_think_loop_watchdog_off() {
    let wp = WatchdogParams::from_behavior(&glm53_behavior(), None, None);
    assert!(!wp.enable_think_loop_watchdog);
    assert!(!ThinkTokenEnv::new(None, false, wp).think_loop_enabled);
    let default = WatchdogParams::from_behavior(&ModelBehavior::default(), None, None);
    assert!(default.enable_think_loop_watchdog);
    assert!(ThinkTokenEnv::new(None, false, default).think_loop_enabled);
    assert!(!ThinkTokenEnv::new(None, true, default).think_loop_enabled);
}

#[test]
fn a_repeated_digit_list_stays_unforced_with_the_watchdog_off() {
    // 2026-10-15: The same tail arms a forced `</think>` with the default watchdog, so the
    // fixture is one the watchdog does catch.
    let (armed, fires) = think_digit_list(&ModelBehavior::default(), false);
    assert!(
        armed && fires == 1,
        "default watchdog: armed={armed} fires={fires}"
    );
    assert_eq!(think_digit_list(&glm53_behavior(), false), (false, 0));
    assert_eq!(
        think_digit_list(&ModelBehavior::default(), true),
        (false, 0)
    );
}
