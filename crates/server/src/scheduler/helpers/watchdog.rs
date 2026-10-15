// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-model watchdog tunables (`WatchdogParams`) and their
//! env/CLI resolution.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Per-model tunables for the decode-time watchdogs. A served
/// run builds them with [`WatchdogParams::from_behavior`] from the model's
/// MODEL.toml `[behavior]` table; `Default` returns
/// [`DEFAULT_WATCHDOG_PARAMS`]. The two differ for an unset table:
/// `confidence_run_length` is 30 from `ModelBehavior` and
/// `CONFIDENCE_RUN_LIMIT` (60) from `Default`.
#[derive(Debug, Clone, Copy)]
pub struct WatchdogParams {
    /// 2026-09-25: Thinking-loop watchdog: how many end-anchored copies of
    /// a period trip a forced `</think>`.
    pub think_loop_min_repeats: usize,
    /// 2026-09-25: Passed to [`super::detectors::detect_token_loop`], which
    /// ignores it, so this field has no effect.
    pub think_loop_scan_window: usize,
    /// 2026-09-25: Whether `F2ConfidenceEarlyStop` may arm a forced
    /// `</think>`.
    pub confidence_early_stop: bool,
    /// 2026-09-25: Consecutive confident thinking tokens after which
    /// `F2ConfidenceEarlyStop` arms a forced `</think>`.
    pub confidence_run_length: u32,
    /// 2026-09-25: Fuzzy-repetition tolerance divisor: a `pattern_len`
    /// window tolerates `max(pattern_len / div, 1)` mismatches.
    pub fuzzy_repeat_tolerance_div: usize,
    /// 2026-09-25: Cap on free-text tokens since the last `<tool_call>`
    /// open on a tool request. `u32::MAX` means the guard is disabled (see
    /// [`resolve_max_inter_tool_prose`]).
    pub max_inter_tool_prose: u32,
    /// 2026-09-25: Cap on content tokens per generation for a sequence
    /// with a grammar attached.
    pub max_post_think_content_tokens: u32,
    /// 2026-09-25: When thinking is the only reason an EOS is suppressed,
    /// close the thinking block instead of discarding the EOS.
    pub honor_eos_inside_thinking: bool,
    /// 2026-09-25: Per-model gate for the thinking-loop watchdog.
    pub enable_think_loop_watchdog: bool,
    /// 2026-09-25: Thinking tokens during which `</think>` gets a negative
    /// logit bias (`[behavior].min_reasoning_floor_tokens`); 0 disables it.
    pub min_reasoning_floor: u32,
    /// 2026-09-25: Allow a firing watchdog to roll back to the last
    /// boundary and re-steer; false makes
    /// [`crate::scheduler::rollback::rollback_to_boundary`] decline.
    pub rollback_resteer: bool,
    /// 2026-09-25: Operator override for the content-loop detector's
    /// repeat threshold (`--content-loop-min-repeats` /
    /// `METRALE_CONTENT_LOOP_MIN_REPEATS`). `None` = the built-in
    /// [`CONTENT_LOOP_MIN_REPEATS`]. A per-request `repetition_detection`
    /// object outranks it (see [`WatchdogParams::content_loop_params`]).
    pub content_loop_min_repeats: Option<u32>,
    /// 2026-10-15: MODEL.toml `[behavior].mask_stray_tool_call_end`: mask the
    /// tool-call end token on a row with no tools, no grammar and no open
    /// `<tool_call>` (`logit_processors::stray_tool_call_end`). No env lever and
    /// no CLI override: the MODEL.toml flag is the switch.
    pub mask_stray_tool_call_end: bool,
}

/// 2026-09-25: What `WatchdogParams::default()` returns. Served runs use
/// [`WatchdogParams::from_behavior`] instead.
const DEFAULT_WATCHDOG_PARAMS: WatchdogParams = WatchdogParams {
    think_loop_min_repeats: THINK_LOOP_MIN_REPEATS,
    think_loop_scan_window: THINK_LOOP_SCAN_WINDOW,
    confidence_early_stop: true,
    confidence_run_length: crate::scheduler::confidence::CONFIDENCE_RUN_LIMIT,
    fuzzy_repeat_tolerance_div: 12,
    max_inter_tool_prose: MAX_INTER_TOOL_PROSE,
    max_post_think_content_tokens: MAX_POST_THINK_CONTENT_TOKENS,
    rollback_resteer: true,
    honor_eos_inside_thinking: false,
    enable_think_loop_watchdog: true,
    content_loop_min_repeats: None,
    min_reasoning_floor: 16,
    mask_stray_tool_call_end: metrale_kernels::DEFAULT_MASK_STRAY_TOOL_CALL_END,
};

impl Default for WatchdogParams {
    fn default() -> Self {
        DEFAULT_WATCHDOG_PARAMS
    }
}

impl WatchdogParams {
    /// 2026-09-25: The default fuzzy-repeat tolerance divisor, exposed so
    /// the `repetition` tests can name it.
    pub const DEFAULT_FUZZY_TOLERANCE_DIV: usize =
        DEFAULT_WATCHDOG_PARAMS.fuzzy_repeat_tolerance_div;

    /// 2026-09-25: Resolve this model's watchdog tunables from its
    /// MODEL.toml `[behavior]` table, then the overrides that outrank it.
    ///
    /// `max_inter_tool_prose_cli` is `--max-inter-tool-prose`; see
    /// [`resolve_max_inter_tool_prose`] for the precedence chain.
    /// `content_loop_min_repeats_cli` is `--content-loop-min-repeats`;
    /// precedence CLI -> `METRALE_CONTENT_LOOP_MIN_REPEATS` -> `None` (the
    /// built-in [`CONTENT_LOOP_MIN_REPEATS`]).
    pub fn from_behavior(
        b: &metrale_kernels::ModelBehavior,
        max_inter_tool_prose_cli: Option<u32>,
        content_loop_min_repeats_cli: Option<u32>,
    ) -> Self {
        let mut p = Self {
            min_reasoning_floor: b.min_reasoning_floor_tokens,
            think_loop_min_repeats: b.think_loop_min_repeats as usize,
            think_loop_scan_window: b.think_loop_scan_window as usize,
            confidence_early_stop: b.confidence_early_stop,
            confidence_run_length: b.confidence_run_length,
            fuzzy_repeat_tolerance_div: b.fuzzy_repeat_tolerance_div as usize,
            max_inter_tool_prose: b.max_inter_tool_prose,
            max_post_think_content_tokens: b.max_post_think_content_tokens,
            rollback_resteer: b.rollback_resteer,
            honor_eos_inside_thinking: b.honor_eos_inside_thinking,
            enable_think_loop_watchdog: b.enable_think_loop_watchdog,
            content_loop_min_repeats: None,
            mask_stray_tool_call_end: b.mask_stray_tool_call_end,
        };
        let env = match std::env::var("METRALE_MAX_INTER_TOOL_PROSE") {
            Ok(v) => match v.parse::<u32>() {
                Ok(n) => Some(n),
                Err(_) => {
                    // 2026-09-25: A set but unparseable override is a config
                    // error: warn, so the fallback to the lower-precedence
                    // value is visible.
                    tracing::warn!(
                        value = %v,
                        "METRALE_MAX_INTER_TOOL_PROSE is set but not a u32; ignoring it"
                    );
                    None
                }
            },
            Err(_) => None,
        };
        p.max_inter_tool_prose =
            resolve_max_inter_tool_prose(p.max_inter_tool_prose, env, max_inter_tool_prose_cli);
        p.content_loop_min_repeats = content_loop_min_repeats_cli.or(parse_env_u32(
            "METRALE_CONTENT_LOOP_MIN_REPEATS",
            std::env::var("METRALE_CONTENT_LOOP_MIN_REPEATS")
                .ok()
                .as_deref(),
        ));
        p
    }

    /// 2026-09-25: The content-loop detector params in force for one
    /// sequence: the request's own `repetition_detection` object outranks
    /// the operator override; `None` = the built-in constants. The operator
    /// override sets only the repeat threshold; its periods are the built-in
    /// range.
    pub fn content_loop_params(
        &self,
        request: Option<crate::api::inference_types::RepetitionDetectionParams>,
    ) -> Option<crate::api::inference_types::RepetitionDetectionParams> {
        request.or_else(|| {
            self.content_loop_min_repeats.map(|n| {
                crate::api::inference_types::RepetitionDetectionParams {
                    min_pattern_size: CONTENT_LOOP_PERIOD_MIN as u32,
                    max_pattern_size: CONTENT_LOOP_PERIOD_MAX as u32,
                    min_count: n,
                }
            })
        })
    }
}

/// 2026-09-25: Parse an optional numeric env override. A set but
/// unparseable value logs a warning and returns `None`.
fn parse_env_u32(name: &str, v: Option<&str>) -> Option<u32> {
    let v = v?;
    match v.trim().parse::<u32>() {
        Ok(n) => Some(n),
        Err(_) => {
            tracing::warn!(value = %v, "{name} is set but not a u32; ignoring it");
            None
        }
    }
}

/// 2026-09-25: Resolve whether the content-loop watchdog is armed for this
/// run.
///
/// Precedence, highest wins: `--content-loop-watchdog` (CLI) ->
/// `METRALE_CONTENT_LOOP_WATCHDOG` (env, `1`/`true`/`0`/`false`) ->
/// MODEL.toml `[behavior].enable_loop_watchdog`. Any other env value logs a
/// warning and keeps the MODEL.toml value.
pub fn resolve_content_loop_watchdog(toml: bool, env: Option<&str>, cli: Option<bool>) -> bool {
    if let Some(cli) = cli {
        return cli;
    }
    match env {
        None => toml,
        Some(v) => {
            let v = v.trim();
            if v == "1" || v.eq_ignore_ascii_case("true") {
                true
            } else if v == "0" || v.eq_ignore_ascii_case("false") {
                false
            } else {
                tracing::warn!(
                    value = %v,
                    "METRALE_CONTENT_LOOP_WATCHDOG is set but not 1/true/0/false; \
                     keeping the MODEL.toml value"
                );
                toml
            }
        }
    }
}

/// 2026-09-25: Resolve the effective inter-tool prose budget.
///
/// Precedence, highest wins: `--max-inter-tool-prose` (CLI) ->
/// `METRALE_MAX_INTER_TOOL_PROSE` (env) -> MODEL.toml `[behavior].max_inter_tool_prose` -> the shared default
/// (`metrale_kernels::DEFAULT_MAX_INTER_TOOL_PROSE`, already folded into
/// `toml` by the build-time parse).
///
/// 0 means "guard disabled" and maps to `u32::MAX`: the check sites fire
/// on `prose_tokens_since_last_tool > max`, so a literal 0 would end a tool
/// request at its first free-text token.
pub fn resolve_max_inter_tool_prose(toml: u32, env: Option<u32>, cli: Option<u32>) -> u32 {
    let v = cli.or(env).unwrap_or(toml);
    if v == 0 { u32::MAX } else { v }
}

/// 2026-09-25: The default inter-tool prose budget. With
/// `tool_choice="auto"` the tool grammar compiles with
/// `stop_after_first=false`, so it does not end the turn after a tool call;
/// this budget bounds the free text in between. It aliases
/// `metrale_kernels::DEFAULT_MAX_INTER_TOOL_PROSE`, the value the build-time
/// `[behavior]` parse uses for an unset key, so `Default` and an unset
/// MODEL.toml key agree.
pub const MAX_INTER_TOOL_PROSE: u32 = metrale_kernels::DEFAULT_MAX_INTER_TOOL_PROSE;

/// 2026-09-25: `Default`'s cap on content tokens per generation for a
/// sequence with a grammar attached (`grammar_state.is_some()`); the check
/// ignores `inside_tool_body`. The build-time parse of an unset
/// `[behavior].max_post_think_content_tokens` also gives 100_000; the
/// qwen3.6-35b-a3b MODEL.tomls set 8192.
pub const MAX_POST_THINK_CONTENT_TOKENS: u32 = 100_000;
