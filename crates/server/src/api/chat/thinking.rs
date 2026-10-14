// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Resolve `(enable_thinking, thinking_budget)` for one request from its
//! `ir::ThinkingDirective`. Precedence (highest wins):
//!   1. `--disable-thinking` (thinking off for every request);
//!   2. the request directive (the client's, or `--default-chat-template-kwargs` when
//!      the client is silent; `prepare.rs` makes that choice);
//!   3. MODEL.toml `[behavior].thinking_default`, which a tool turn overrides to off
//!      when `thinking_in_tools` is false.
//!
//! 2026-10-14: With MODEL.toml `[behavior].effort_prompt_only` (GLM-5.3), thinking that is
//! on has no budget unless the client names one: effort reaches the model only through the
//! chat template's prompt line, as on the official model.
//!
//! Owner: server chat API.
//! Invariants:
//! - A disabled result carries no budget: `enable_thinking == false` implies `None`.
//! - `enable_thinking == true` with `None` means no budget: only `effort_prompt_only` gives it.

use std::sync::Arc;

use crate::AppState;
use crate::ir::ThinkingDirective;

pub(super) fn resolve_thinking(
    state: &Arc<AppState>,
    directive: ThinkingDirective,
    max_tokens: u32,
    tools_active: bool,
) -> (bool, Option<u32>) {
    resolve(
        directive,
        Policy {
            disable_thinking: state.disable_thinking,
            model_default: state.behavior.thinking_default,
            thinking_in_tools: state.behavior.thinking_in_tools,
            max_thinking_budget: state.behavior.max_thinking_budget,
            effort_capped_at_ceiling: state.behavior.effort_capped_at_ceiling,
            effort_prompt_only: state.behavior.effort_prompt_only,
            cap_at_max_tokens: state.behavior.cap_thinking_at_max_tokens,
        },
        max_tokens,
        tools_active,
    )
}

/// 2026-09-26: Generation allowance after the tools-active `--tool-max-tokens` shrink.
/// Both the thinking-budget resolution (`prepare.rs`) and the sampling `max_tokens`
/// (`sampling_setup.rs`) read it, so a tool turn's thinking budget is sized to the
/// tokens it may emit.
pub(super) fn generation_max_tokens(
    max_tokens: usize,
    tools_active: bool,
    tool_max_tokens: usize,
) -> usize {
    if tools_active {
        max_tokens.min(tool_max_tokens)
    } else {
        max_tokens
    }
}

/// 2026-09-26: Server and model policy inputs, copied from `AppState` so `resolve` is a
/// pure function.
#[derive(Clone, Copy)]
struct Policy {
    disable_thinking: bool,
    model_default: bool,
    thinking_in_tools: bool,
    max_thinking_budget: u32,
    /// 2026-09-26: MODEL.toml `[behavior].effort_capped_at_ceiling`: clamp the effort
    /// ladder at E, so high and xhigh give E. An explicit client token budget is not
    /// affected.
    effort_capped_at_ceiling: bool,
    /// 2026-10-14: MODEL.toml `[behavior].effort_prompt_only` (after the
    /// `--max-thinking-budget` / `METRALE_EFFORT_THINKING_BUDGET` override): thinking
    /// without a client budget gets none, at every effort level.
    effort_prompt_only: bool,
    /// 2026-09-26: MODEL.toml `[behavior].cap_thinking_at_max_tokens`. When false, the
    /// budget (the client's, or `max_thinking_budget`) is not clamped to 90% of
    /// `max_tokens`.
    cap_at_max_tokens: bool,
}

fn resolve(
    directive: ThinkingDirective,
    policy: Policy,
    max_tokens: u32,
    tools_active: bool,
) -> (bool, Option<u32>) {
    if policy.disable_thinking {
        return (false, None);
    }
    let (et, tb) = match directive {
        // 2026-09-26: A `None` budget resolves to `max_thinking_budget` below.
        ThinkingDirective::Unspecified => (policy.model_default, None),
        ThinkingDirective::Off => (false, None),
        ThinkingDirective::On { budget } => (true, budget),
        // 2026-10-14: The official GLM-5.3 template turns effort into the prompt line
        // "Reasoning Effort: Low|High|Max" and nothing else; no budget here.
        ThinkingDirective::OnEffort(_) if policy.effort_prompt_only => (true, None),
        // 2026-09-26: An effort level scales with `max_thinking_budget`, which MODEL.toml
        // or `--max-thinking-budget` sets.
        ThinkingDirective::OnEffort(level) => (
            true,
            Some(effort_budget(
                level,
                policy.max_thinking_budget,
                policy.effort_capped_at_ceiling,
            )),
        ),
    };
    // 2026-09-26: `thinking_in_tools = false` turns thinking off on a tool turn only
    // when the directive is `Unspecified`. Any explicit directive, including the
    // server-level default one, wins.
    let et = if tools_active && !policy.thinking_in_tools && !directive.is_explicit() {
        false
    } else {
        et
    };
    let budget = if et && tb.is_none() && policy.effort_prompt_only {
        // 2026-10-14: No budget: no forced `</think>`, no confidence early stop and no
        // 90% `max_tokens` clamp; thinking ends when the model closes it or at
        // `max_tokens`.
        None
    } else if et {
        let b = tb.unwrap_or(policy.max_thinking_budget);
        if !policy.cap_at_max_tokens {
            Some(b)
        } else {
            // 2026-09-26: With `cap_at_max_tokens`, the thinking budget is also capped at
            // 90% of `max_tokens` (at least 1), leaving room for content and tool
            // arguments after the reasoning.
            let safety_cap_pct = 9;
            let max = ((max_tokens * safety_cap_pct) / 10).max(1);
            Some(b.min(max))
        }
    } else {
        None
    };
    (et, budget)
}

/// 2026-09-26: Token budget for an effort level, as a ratio of the model's
/// `max_thinking_budget` E: minimal=E/4, low=E/2, medium=E, high=2E, xhigh=4E
/// (minimal and low at least 1). At the default E=256 (`DEFAULT_MAX_THINKING_BUDGET`)
/// that is 64/128/256/512/1024. `resolve` applies the `cap_at_max_tokens` clamp
/// afterwards.
///
/// `capped_at_ceiling` (MODEL.toml `[behavior].effort_capped_at_ceiling`, default
/// false) clamps every level at E. `kernels/gb10/qwen3.5-397b-a17b/MODEL.toml` sets
/// it, and records its reason: a budget sweep there (2026-05-07) scored 256 thinking
/// tokens worse than 128. Explicit client budgets never pass through here.
fn effort_budget(
    level: crate::ir::EffortLevel,
    max_thinking_budget: u32,
    capped_at_ceiling: bool,
) -> u32 {
    use crate::ir::EffortLevel;
    let e = max_thinking_budget.max(1);
    let b = match level {
        EffortLevel::Minimal => (e / 4).max(1),
        EffortLevel::Low => (e / 2).max(1),
        EffortLevel::Medium => e,
        EffortLevel::High => e.saturating_mul(2),
        EffortLevel::XHigh => e.saturating_mul(4),
    };
    if capped_at_ceiling { b.min(e) } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy {
            disable_thinking: false,
            model_default: false,
            thinking_in_tools: true,
            max_thinking_budget: 2048,
            effort_capped_at_ceiling: false,
            effort_prompt_only: false,
            cap_at_max_tokens: true,
        }
    }

    #[test]
    fn kill_switch_overrides_everything() {
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: Some(512) },
            Policy {
                disable_thinking: true,
                ..policy()
            },
            4096,
            false,
        );
        assert!(!et);
        assert!(tb.is_none());
    }

    #[test]
    fn unspecified_falls_to_model_default() {
        let (et, tb) = resolve(
            ThinkingDirective::Unspecified,
            Policy {
                model_default: true,
                ..policy()
            },
            4096,
            false,
        );
        assert!(et);
        // 2026-09-26: `max_thinking_budget` (2048), below 90% of 4096.
        assert_eq!(tb, Some(2048));

        let (et, tb) = resolve(ThinkingDirective::Unspecified, policy(), 4096, false);
        assert!(!et);
        assert!(tb.is_none());
    }

    #[test]
    fn explicit_budget_capped_at_90_pct_of_max_tokens() {
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: Some(4096) },
            policy(),
            1000,
            false,
        );
        assert!(et);
        assert_eq!(tb, Some(900));
    }

    #[test]
    fn budgetless_on_defers_to_model_cap() {
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: None },
            policy(),
            4096,
            false,
        );
        assert!(et);
        assert_eq!(tb, Some(2048));
    }

    #[test]
    fn effort_ladder_at_default_ceiling_matches_the_historical_absolutes() {
        // 2026-09-26: At the default ceiling (256, `ModelBehavior::default()`) the levels
        // resolve to 64/128/256/512/1024. Fails if the ratios change.
        use crate::ir::EffortLevel::*;
        let default_ceiling = Policy {
            max_thinking_budget: 256,
            cap_at_max_tokens: false,
            ..policy()
        };
        for (level, historical) in [
            (Minimal, 64),
            (Low, 128),
            (Medium, 256),
            (High, 512),
            (XHigh, 1024),
        ] {
            let (et, tb) = resolve(
                ThinkingDirective::OnEffort(level),
                default_ceiling,
                4096,
                false,
            );
            assert!(et);
            assert_eq!(tb, Some(historical), "level={level:?}");
        }
    }

    #[test]
    fn effort_ladder_scales_with_the_operator_ceiling() {
        // 2026-09-26: `--max-thinking-budget` (folded into `max_thinking_budget` by
        // serve) sets what an effort level means.
        use crate::ir::EffortLevel::*;
        let operator = Policy {
            max_thinking_budget: 16256,
            cap_at_max_tokens: false,
            ..policy()
        };
        let (et, tb) = resolve(ThinkingDirective::OnEffort(Medium), operator, 32768, false);
        assert!(et);
        assert_eq!(tb, Some(16256));
        let (_, tb) = resolve(ThinkingDirective::OnEffort(Minimal), operator, 32768, false);
        assert_eq!(tb, Some(4064));
        // 2026-09-26: With `cap_thinking_at_max_tokens`, the 90%-of-max_tokens clamp
        // still applies.
        let capped = Policy {
            max_thinking_budget: 16256,
            ..policy()
        };
        let (_, tb) = resolve(ThinkingDirective::OnEffort(Medium), capped, 1000, false);
        assert_eq!(tb, Some(900));
    }

    #[test]
    fn effort_cap_defaults_off_so_shipping_it_changes_nothing() {
        // 2026-09-26: The built-in default is `false`, read from
        // `ModelBehavior::default()` rather than a copy of the literal.
        assert!(!metrale_kernels::ModelBehavior::default().effort_capped_at_ceiling);
    }

    #[test]
    fn effort_cap_clamps_the_ladder_at_the_ceiling() {
        // 2026-09-26: With the cap, high and xhigh stop at E.
        use crate::ir::EffortLevel::*;
        let p = Policy {
            max_thinking_budget: 128,
            effort_capped_at_ceiling: true,
            effort_prompt_only: false,
            cap_at_max_tokens: false,
            ..policy()
        };
        for (level, expected) in [
            (Minimal, 32),
            (Low, 64),
            (Medium, 128),
            (High, 128),
            (XHigh, 128),
        ] {
            let (et, tb) = resolve(ThinkingDirective::OnEffort(level), p, 4096, false);
            assert!(et);
            assert_eq!(tb, Some(expected), "level={level:?}");
        }
    }

    #[test]
    fn effort_cap_never_touches_an_explicit_client_budget() {
        // 2026-09-26: The cap binds only the effort ladder; a client that states a
        // number gets that number.
        let p = Policy {
            max_thinking_budget: 128,
            effort_capped_at_ceiling: true,
            effort_prompt_only: false,
            cap_at_max_tokens: false,
            ..policy()
        };
        let (et, tb) = resolve(ThinkingDirective::On { budget: Some(4096) }, p, 8192, false);
        assert!(et);
        assert_eq!(tb, Some(4096));
    }

    #[test]
    fn effort_is_explicit_for_thinking_in_tools_purposes() {
        // 2026-09-26: An effort level is an explicit directive, so the
        // `thinking_in_tools = false` suppression does not apply.
        let p = Policy {
            thinking_in_tools: false,
            ..policy()
        };
        let (et, _) = resolve(
            ThinkingDirective::OnEffort(crate::ir::EffortLevel::Medium),
            p,
            4096,
            true,
        );
        assert!(et);
    }

    #[test]
    fn cap_at_max_tokens_false_skips_the_90pct_clamp() {
        // 2026-09-26: Without `cap_at_max_tokens`, a small `max_tokens` does not clamp the
        // budget.
        let no_cap = Policy {
            cap_at_max_tokens: false,
            ..policy()
        };
        let (et, tb) = resolve(
            ThinkingDirective::On { budget: Some(4096) },
            no_cap,
            1000,
            false,
        );
        assert!(et);
        assert_eq!(tb, Some(4096));
        let (_, tb) = resolve(ThinkingDirective::On { budget: None }, no_cap, 1000, false);
        assert_eq!(tb, Some(2048));
    }

    #[test]
    fn tools_suppression_only_when_client_silent() {
        let no_tools_thinking = Policy {
            model_default: true,
            thinking_in_tools: false,
            ..policy()
        };
        let (et, _) = resolve(
            ThinkingDirective::Unspecified,
            Policy {
                ..no_tools_thinking
            },
            4096,
            true,
        );
        assert!(!et);
        let (et, _) = resolve(
            ThinkingDirective::On { budget: None },
            Policy {
                model_default: true,
                thinking_in_tools: false,
                ..policy()
            },
            4096,
            true,
        );
        assert!(et);
        let (et, _) = resolve(
            ThinkingDirective::Off,
            Policy {
                model_default: true,
                thinking_in_tools: false,
                ..policy()
            },
            4096,
            true,
        );
        assert!(!et);
    }

    #[test]
    fn explicit_off_wins_over_model_default() {
        let (et, tb) = resolve(
            ThinkingDirective::Off,
            Policy {
                model_default: true,
                ..policy()
            },
            4096,
            false,
        );
        assert!(!et);
        assert!(tb.is_none());
    }

    #[test]
    fn issue_517_tool_capped_ceiling_leaves_room_for_tool_call() {
        // 2026-09-26: The caller passes `generation_max_tokens` (here 256), so the
        // budget is 90% of 256 = 230.
        let (et, tb) = resolve(ThinkingDirective::On { budget: None }, policy(), 256, true);
        assert!(et);
        assert_eq!(tb, Some(230));
        assert!(tb.unwrap() < 256);
    }

    fn prompt_only() -> Policy {
        // 2026-10-14: GLM-5.3: E = 256 (the default), the 90% clamp on, effort a prompt
        // line only.
        Policy {
            max_thinking_budget: 256,
            effort_prompt_only: true,
            ..policy()
        }
    }

    #[test]
    fn prompt_only_gives_every_effort_level_no_budget() {
        use crate::ir::EffortLevel;
        for level in [
            EffortLevel::Minimal,
            EffortLevel::Low,
            EffortLevel::Medium,
            EffortLevel::High,
            EffortLevel::XHigh,
        ] {
            for tools in [false, true] {
                let r = resolve(ThinkingDirective::OnEffort(level), prompt_only(), 65536, tools);
                assert_eq!(r, (true, None), "{level:?} tools={tools}");
            }
            // 2026-10-14: not the 90% clamp either: a small max_tokens still gives None.
            let r = resolve(ThinkingDirective::OnEffort(level), prompt_only(), 100, false);
            assert_eq!(r, (true, None), "{level:?} max_tokens 100");
        }
    }

    #[test]
    fn prompt_only_thinking_on_without_a_budget_has_none() {
        let r = resolve(ThinkingDirective::On { budget: None }, prompt_only(), 65536, false);
        assert_eq!(r, (true, None));
        let r = resolve(
            ThinkingDirective::Unspecified,
            Policy {
                model_default: true,
                ..prompt_only()
            },
            65536,
            false,
        );
        assert_eq!(r, (true, None));
    }

    #[test]
    fn prompt_only_keeps_an_explicit_client_budget_and_off() {
        let r = resolve(ThinkingDirective::On { budget: Some(700) }, prompt_only(), 65536, false);
        assert_eq!(r, (true, Some(700)));
        // 2026-10-14: the client's budget keeps the 90% clamp it always had.
        let r = resolve(ThinkingDirective::On { budget: Some(700) }, prompt_only(), 100, false);
        assert_eq!(r, (true, Some(90)));
        assert_eq!(resolve(ThinkingDirective::Off, prompt_only(), 65536, false), (false, None));
        let off = Policy {
            disable_thinking: true,
            ..prompt_only()
        };
        let r = resolve(ThinkingDirective::OnEffort(crate::ir::EffortLevel::XHigh), off, 65536, false);
        assert_eq!(r, (false, None));
    }

    #[test]
    fn without_prompt_only_the_ladder_is_unchanged() {
        // 2026-10-14: the lever-on / other-model path: 128 / 256 / 512 / 1024 at E = 256.
        use crate::ir::EffortLevel;
        let ladder = Policy {
            effort_prompt_only: false,
            ..prompt_only()
        };
        for (level, want) in [
            (EffortLevel::Low, 128),
            (EffortLevel::Medium, 256),
            (EffortLevel::High, 512),
            (EffortLevel::XHigh, 1024),
        ] {
            let r = resolve(ThinkingDirective::OnEffort(level), ladder, 65536, false);
            assert_eq!(r, (true, Some(want)), "{level:?}");
        }
    }
}
