// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: `[behavior]` defaults that the build-time MODEL.toml parse and
// `ModelBehavior::default()` both read.
//
// Owner: kernels crate.
// Invariants:
// - `lib.rs` compiles this file as a module and `build_parse_behavior.rs`
//   `include!`s it, so both sides use the same constants. `//!` cannot appear
//   at an `include!` site, which is why this header uses `//`.

/// 2026-09-25: Default `[behavior].max_inter_tool_prose`: the cap on free-text
/// tokens between tool calls on a tool request. 0 is reserved: the runtime
/// resolver (`resolve_max_inter_tool_prose`) maps it to "guard disabled". A
/// model can pin a tighter bound in its MODEL.toml, as
/// `kernels/strix/qwen3.6-35b-a3b` does with 384.
pub const DEFAULT_MAX_INTER_TOOL_PROSE: u32 = 3072;

/// 2026-09-25: Default `[behavior].max_thinking_budget`: the anchor E of the
/// `reasoning_effort` ladder and the budget of a thinking-on request that
/// names none.
pub const DEFAULT_MAX_THINKING_BUDGET: u32 = 256;

/// 2026-09-25: Default `[behavior].effort_capped_at_ceiling`: whether the
/// qualitative `reasoning_effort` levels are clamped at E
/// (`max_thinking_budget`, or `--max-thinking-budget`).
///
/// `false` keeps high = 2E and xhigh = 4E, above the ceiling; at the default
/// E that is 512 and 1024, which
/// `effort_ladder_at_default_ceiling_matches_the_historical_absolutes` pins.
/// `true` clamps every level at E, for a model whose MODEL.toml records that
/// a larger budget scores worse (`kernels/gb10/qwen3.5-397b-a17b` sets it).
/// The clamp applies only to the effort ladder; an explicit client token
/// budget is not clamped.
pub const DEFAULT_EFFORT_CAPPED_AT_CEILING: bool = false;

/// 2026-10-14: Default `[behavior].effort_prompt_only`: whether `reasoning_effort`
/// reaches the model only through the chat template, with no thinking token budget.
///
/// `false` keeps the budget ladder above. `true` gives every effort level, and a
/// thinking-on request that names no budget (including template-forced thinking), no
/// budget: no forced `</think>`, no `F2ConfidenceEarlyStop`, and no 90% `max_tokens`
/// clamp, so thinking ends when the model closes it or at `max_tokens`. That is the
/// official GLM-5.3 semantics: its `chat_template.jinja` (nvidia/GLM-5.3-Flash-NVFP4
/// @09b04e5e, lines 2-3) renders effort as the prompt line "Reasoning Effort:
/// Low|High|Max" and vLLM serves it with no budget. An explicit client token budget
/// still applies. `kernels/gb10/glm-5.3-flash` sets it.
pub const DEFAULT_EFFORT_PROMPT_ONLY: bool = false;

/// 2026-10-15: Default `[behavior].response_format_with_tools`: which grammar a request with
/// both `response_format` and active tools gets.
///
/// `false` keeps the rule every other model has: `response_format` only when tools are
/// inactive or `tool_choice` is `none`, else the tool-call grammar. `true` enforces
/// `response_format` unless `tool_choice` is `required` or names a function (those keep the
/// tool-call grammar), as vLLM applies the response_format structured output with
/// `tool_choice` auto. The JSON grammar then masks `<tool_call>`, so such a turn answers in
/// JSON and cannot call a tool. TEB TC-65/66/67/69 (spark-bench c30gate TEBGAP-TRACE.md): the
/// official serve answers bare JSON 24/24, c30 wraps it in prose 12/12.
/// `kernels/gb10/glm-5.3-flash` sets it.
pub const DEFAULT_RESPONSE_FORMAT_WITH_TOOLS: bool = false;

/// 2026-10-15: Default `[behavior].tool_call_start_bias`: whether a tool request gets the
/// `<tool_call>` start-token logit bias by `tool_call_repeat_count` (+3 at 0 or 1, none at
/// 2, -5 at 3, -10 beyond; `api/chat/sampling_setup.rs`).
///
/// `true` keeps it. `false` adds no bias, as the official serve (vLLM) adds none.
/// `kernels/gb10/glm-5.3-flash` sets it (spark-bench c30gate TEBGAP-TRACE.md D).
pub const DEFAULT_TOOL_CALL_START_BIAS: bool = true;

/// 2026-10-15: Default `[behavior].hf_tool_json`: the `tojson` filter style of this model's
/// chat template (`tokenizer/jinja_helpers.rs`).
///
/// `false` keeps minijinja's compact `tojson` (`{"a":1}`) unless
/// `METRALE_USE_HF_REF_JSON_DUMPS=1`. `true` always uses the spaced Python `json.dumps`
/// form (`{"a": 1}`, `ToolJsonStyle::HfSpaced`), which renders tool schemas byte for byte
/// as the official HF / vLLM template does; the env var still works for every model.
/// `kernels/gb10/glm-5.3-flash` sets it (TEB turn-1 prompts: 86/88 differ from the official
/// render with compact, 87/88 byte-identical with HfSpaced; spark-bench c30gate
/// TEBGAP-TRACE.md C).
pub const DEFAULT_HF_TOOL_JSON: bool = false;

/// 2026-10-15: Default `[behavior].render_tools_on_tool_choice_none`: whether a request with
/// tools and `tool_choice` `none` passes them to the chat template.
///
/// `false` keeps the rule every other model has: tools reach the template only when tools
/// are active. `true` renders them for `tool_choice` `none` too, as the official template
/// (HF `apply_chat_template`) and vLLM (default `exclude_tools_when_tool_choice_none` false)
/// do. Tools stay inactive for parsing, the grammar and the `<tool_call>` bias.
/// `kernels/gb10/glm-5.3-flash` sets it (TEB TC-44; spark-bench c30gate TEBGAP-TRACE.md C).
pub const DEFAULT_RENDER_TOOLS_ON_TOOL_CHOICE_NONE: bool = false;

/// 2026-10-15: Default `[behavior].mask_stray_tool_call_end`: whether the scheduler
/// masks the tool-call end token (`</tool_call>`) on a sequence that offered no tools,
/// has no grammar and has no `<tool_call>` open (server
/// `scheduler/logit_processors/stray_tool_call_end.rs`), so the next-best token wins.
///
/// `false` leaves the token free, as before. `true` is for a model that, under load,
/// ends a plain answer with a stray `</tool_call>`, which the scheduler reads as the end
/// of a tool call (finish_reason `tool_calls`). `kernels/gb10/glm-5.3-flash` sets it.
pub const DEFAULT_MASK_STRAY_TOOL_CALL_END: bool = false;
