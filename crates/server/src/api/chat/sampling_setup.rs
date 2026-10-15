// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-request sampling setup: sampling values (request, then
//! preset or server default, then MODEL.toml floors and ceilings), the
//! `<tool_call>` bias, the token cap, stop tokens, the grammar, the deadline
//! and `top_logprobs`.
//!
//! Owner: server (chat API).
//! Invariants:
//! - Under `METRALE_FORCE_TEMP_ZERO`, temperature, top_k, top_n_sigma, min_p,
//!   the DRY multiplier and the LZ penalty are 0, top_p and the repetition
//!   penalty are 1, presence and frequency penalties are 0, and no logit bias
//!   is applied.

use axum::http::StatusCode;
use axum::response::Response;
use std::sync::Arc;

use crate::AppState;
use crate::ir::ChatRequest;
use crate::tool_parser;

use super::super::compact::openai_error_response;
use super::super::inference_impl::tokenize_stop_sequences;
use super::super::inference_types::GrammarSpec;
use super::thinking;

pub(super) struct SamplingSetup {
    pub(super) temperature: f32,
    pub(super) top_k: u32,
    pub(super) top_p: f32,
    pub(super) top_n_sigma: f32,
    pub(super) min_p: f32,
    pub(super) repetition_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) dry_multiplier: f32,
    pub(super) dry_base: f32,
    pub(super) dry_allowed_length: u32,
    pub(super) lz_penalty: f32,
    pub(super) logit_bias: Vec<(u32, f32)>,
    pub(super) max_tokens: usize,
    pub(super) stop_tokens: Vec<u32>,
    pub(super) tool_choice_required: bool,
    pub(super) grammar_spec: Option<GrammarSpec>,
    pub(super) timeout_at: Option<std::time::Instant>,
    pub(super) top_logprobs: Option<u8>,
}

fn tool_choice_required_for_parser(
    tools_active: bool,
    tool_choice: Option<&tool_parser::ToolChoice>,
    parser_name: Option<&str>,
) -> bool {
    if !tools_active {
        return false;
    }

    let explicit_required = tool_choice.is_some_and(|tc| {
        matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "required")
            || matches!(tc, tool_parser::ToolChoice::Specific { .. })
    });
    let parser_required = matches!(parser_name, Some("minimax_xml"));

    explicit_required || parser_required
}

/// 2026-09-26: Whether the model's tool-grammar opt-out
/// (`disable_tool_grammar`: MODEL.toml `[behavior]`, or `--tool-grammar`)
/// applies to this request. It applies only when the request does not require
/// a tool call, so `tool_choice` `required`, a named tool, and the
/// `minimax_xml` parser keep the grammar. Without it, `required` falls back to
/// the scheduler's EOS suppression (`prefill_a_step.rs`), which needs a
/// `tool_call_start_token`.
fn tool_grammar_escape_applies(disable_tool_grammar: bool, tool_choice_required: bool) -> bool {
    disable_tool_grammar && !tool_choice_required
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::result_large_err)]
pub(super) fn build_sampling(
    state: &Arc<AppState>,
    req: &ChatRequest,
    enable_thinking: bool,
    tools_active: bool,
    suppress_tool_call: bool,
    tool_call_repeat_count: usize,
) -> Result<SamplingSetup, Response> {
    // 2026-09-26: Preset: `tools` when tools are active, else
    // `thinking_text` or `non_thinking`.
    let preset = if tools_active {
        &state.sampling_presets.tools
    } else if enable_thinking {
        &state.sampling_presets.thinking_text
    } else {
        &state.sampling_presets.non_thinking
    };
    // 2026-09-26: `METRALE_FORCE_TEMP_ZERO` (`1` or `true`): greedy decoding
    // whatever the request, preset or MODEL.toml floors say.
    let force_temp_zero = std::env::var("METRALE_FORCE_TEMP_ZERO")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // 2026-09-26: A request value wins. Otherwise temperature, top_k and
    // top_p come from the preset when MODEL.toml sets
    // `use_sampling_presets_for_core`, and from the server defaults when it
    // does not.
    let core_preset = state.behavior.use_sampling_presets_for_core;
    let temperature = if force_temp_zero {
        0.0
    } else {
        req.sampling.temperature.unwrap_or(if core_preset {
            preset.temperature
        } else {
            state.default_temperature
        })
    };
    let top_k = if force_temp_zero {
        0
    } else {
        req.sampling.top_k.unwrap_or(if core_preset {
            preset.top_k
        } else {
            state.default_top_k
        })
    };
    let top_p = if force_temp_zero {
        1.0
    } else {
        req.sampling.top_p.unwrap_or(if core_preset {
            preset.top_p
        } else {
            state.default_top_p
        })
    };
    // 2026-09-26: top_n_sigma and min_p: the request, then (with
    // `use_sampling_presets_for_core`) the preset when it sets one, then the
    // server default.
    let top_n_sigma = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .top_n_sigma
            .or(if core_preset {
                preset.top_n_sigma
            } else {
                None
            })
            .unwrap_or(state.default_top_n_sigma)
    };
    let min_p = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .min_p
            .or(if core_preset { preset.min_p } else { None })
            .unwrap_or(state.default_min_p)
    };
    let repetition_penalty = if force_temp_zero {
        1.0
    } else {
        req.sampling
            .repetition_penalty
            .unwrap_or(preset.repetition_penalty)
    };
    let presence_penalty = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .presence_penalty
            .unwrap_or(preset.presence_penalty)
    };
    let frequency_penalty = if force_temp_zero {
        0.0
    } else {
        req.sampling
            .frequency_penalty
            .unwrap_or(preset.frequency_penalty)
    };
    // 2026-09-26: MODEL.toml `[behavior]` `min_p_floor` and `temperature_max`
    // apply after the values above, so they bind whatever the client sends.
    // 0.0 turns each off; neither applies under `METRALE_FORCE_TEMP_ZERO`.
    let min_p = if !force_temp_zero && state.behavior.min_p_floor > 0.0 {
        min_p.max(state.behavior.min_p_floor)
    } else {
        min_p
    };
    let temperature = if !force_temp_zero && state.behavior.temperature_max > 0.0 {
        temperature.min(state.behavior.temperature_max)
    } else {
        temperature
    };
    // 2026-09-26: Logs the resolved values of the first request in the
    // process.
    {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "sampling resolved (first request): temp={temperature:.3} top_p={top_p:.3} \
                 top_k={top_k} min_p={min_p:.4} top_n_sigma={top_n_sigma:.4} \
                 rep_pen={repetition_penalty:.3} (core_preset={core_preset})"
            );
        });
    }
    let dry_multiplier = if force_temp_zero {
        0.0
    } else {
        preset.dry_multiplier
    };
    let dry_base = preset.dry_base;
    let dry_allowed_length = preset.dry_allowed_length;
    let lz_penalty = if force_temp_zero {
        0.0
    } else {
        preset.lz_penalty
    };

    if !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("presence_penalty must be between -2.0 and 2.0, got {presence_penalty}"),
        ));
    }
    if !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("frequency_penalty must be between -2.0 and 2.0, got {frequency_penalty}"),
        ));
    }

    let mut logit_bias: Vec<(u32, f32)> = if force_temp_zero {
        Vec::new()
    } else {
        req.logit_bias.clone()
    };

    if let Some(tc_id) = state.tool_call_start_token_id
        && let Some(bias) = tool_call_start_bias(
            force_temp_zero,
            tools_active,
            suppress_tool_call,
            tool_call_repeat_count,
            &state.behavior,
        )
    {
        logit_bias.push((tc_id, bias));
    }

    // 2026-09-26: The same `generation_max_tokens` cap `prepare_chat_prompt`
    // gives thinking resolution.
    let max_tokens =
        thinking::generation_max_tokens(req.max_tokens, tools_active, state.tool_max_tokens);
    if tools_active && max_tokens < req.max_tokens {
        tracing::info!(
            "Tool max_tokens cap: {} → {} (tool_max_tokens={})",
            req.max_tokens,
            max_tokens,
            state.tool_max_tokens
        );
    }

    // 2026-09-26: Only the request's stop sequences. `</tool_call>` is not a
    // stop token, so generation continues past a closed call and the model can
    // emit several.
    let stop_tokens = tokenize_stop_sequences(&state.tokenizer, &req.stop);

    let tool_choice_required = tool_choice_required_for_parser(
        tools_active,
        req.tool_choice.as_ref(),
        state.tool_call_parser.as_ref().map(|p| p.name()),
    );

    let grammar_spec = request_grammar(
        req,
        tools_active,
        tool_choice_required,
        &state.behavior,
        state.tool_call_parser.as_ref(),
    );

    let timeout_at = state.request_deadline(req.timeout_secs);

    let top_logprobs = req.top_logprobs;

    Ok(SamplingSetup {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    })
}

/// 2026-09-26: `<tool_call>` bias by `tool_call_repeat_count`: +3 at 0 or
/// 1, none at 2, -5 at 3, -10 beyond. Only with tools active, the token not
/// hard-masked, and `METRALE_FORCE_TEMP_ZERO` off.
///
/// 2026-10-15: Moved out of `build_sampling` for the tests; none at all under MODEL.toml
/// `[behavior].tool_call_start_bias = false` (GLM-5.3, as on the official serve).
fn tool_call_start_bias(
    force_temp_zero: bool,
    tools_active: bool,
    suppress_tool_call: bool,
    tool_call_repeat_count: usize,
    behavior: &metrale_kernels::ModelBehavior,
) -> Option<f32> {
    if force_temp_zero || !tools_active || suppress_tool_call || !behavior.tool_call_start_bias {
        return None;
    }
    let bias = match tool_call_repeat_count {
        0 | 1 => 3.0,
        2 => 0.0,
        3 => -5.0,
        _ => -10.0,
    };
    (bias != 0.0).then_some(bias)
}

/// 2026-09-26: One grammar per request. `response_format` is enforced when
/// tools are inactive or `tool_choice` is `none`; otherwise the tool-call
/// grammar, if any, is used and the response format is left to the model.
/// The wire's `{"type":"text"}` is lowered to `None`, so presence means a
/// constraint.
///
/// 2026-10-15: Moved out of `build_sampling` so the tests run it. With MODEL.toml
/// `[behavior].response_format_with_tools` (GLM-5.3) see [`response_format_enforced`].
fn request_grammar(
    req: &ChatRequest,
    tools_active: bool,
    tool_choice_required: bool,
    behavior: &metrale_kernels::ModelBehavior,
    parser: Option<&Arc<dyn tool_parser::ToolCallParser>>,
) -> Option<GrammarSpec> {
    let has_response_format = req.response_format.is_some();
    let tool_choice_none = req
        .tool_choice
        .as_ref()
        .is_some_and(|tc| matches!(tc, tool_parser::ToolChoice::Mode(m) if m == "none"));
    let response_format_only = response_format_enforced(
        has_response_format,
        tools_active,
        tool_choice_none,
        tool_choice_required,
        behavior.response_format_with_tools,
    );

    let use_triggers = !tool_choice_required;
    if response_format_only {
        match req.response_format.as_ref().unwrap() {
            crate::ir::ResponseFormat::JsonObject => Some(GrammarSpec::JsonObject),
            crate::ir::ResponseFormat::JsonSchema { schema, .. } => Some(GrammarSpec::JsonSchema {
                schema: schema.to_string(),
            }),
        }
    } else if tools_active
        && tool_grammar_escape_applies(behavior.disable_tool_grammar, tool_choice_required)
    {
        // 2026-09-26: No grammar; tool calls are still parsed from the output.
        tracing::info!("MODEL.toml [behavior].disable_tool_grammar=true — tool-call grammar OFF");
        None
    } else if tools_active {
        if has_response_format {
            tracing::info!(
                "response_format + tools both set; enforcing tool-call grammar. \
                 Schema-shape compliance falls to the model (embed schema text in \
                 the user/system message for best results)."
            );
        }
        let parser = parser.map(std::sync::Arc::clone);
        let mut tools = req.tools.clone();
        if let Some(tool_parser::ToolChoice::Specific { ref function }) = req.tool_choice {
            tools.retain(|t| t.function.name == function.name);
        }
        parser.map(|p| GrammarSpec::ToolCall {
            tools,
            parser: p,
            use_triggers,
        })
    } else {
        None
    }
}

/// 2026-10-15: Whether `response_format` is the request's grammar. Without
/// `response_format_with_tools` (every model but GLM-5.3): only when tools are inactive or
/// `tool_choice` is `none`. With it: unless `tool_choice` is `required` or names a function
/// (`tool_choice_required`), which keep the tool-call grammar. That is vLLM's precedence,
/// which applies the response_format structured output with `tool_choice` auto; the JSON
/// grammar then masks `<tool_call>`, so the turn answers in JSON and calls no tool.
fn response_format_enforced(
    has_response_format: bool,
    tools_active: bool,
    tool_choice_none: bool,
    tool_choice_required: bool,
    with_tools: bool,
) -> bool {
    if with_tools {
        has_response_format && !tool_choice_required
    } else {
        has_response_format && (!tools_active || tool_choice_none)
    }
}

#[cfg(test)]
#[path = "sampling_setup_tests.rs"]
mod sampling_setup_tests;
