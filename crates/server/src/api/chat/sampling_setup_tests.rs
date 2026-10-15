// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `sampling_setup`: which requests count as tool-required, and
//! when the model-level tool-grammar escape hatch applies.
//!
//! Owner: server chat API.
//! Invariants: none beyond the types.

#[cfg(test)]
mod tests {
    // 2026-09-26: `tests` sits inside `sampling_setup_tests`, a child module of
    // `sampling_setup`, so `super::super` is `sampling_setup`.
    use super::super::{
        request_grammar, response_format_enforced, tool_call_start_bias,
        tool_choice_required_for_parser, tool_grammar_escape_applies,
    };

    use crate::tool_parser::{ToolChoice, ToolChoiceFunction};

    /// 2026-09-26: The `--tool-grammar` help scopes the hatch to `tool_choice="auto"`.
    #[test]
    fn the_escape_hatch_applies_in_auto_mode() {
        let required = tool_choice_required_for_parser(true, None, Some("qwen3_coder"));

        assert!(!required);
        assert!(tool_grammar_escape_applies(true, required));
    }

    /// 2026-09-26: The grammar is what enforces `required`. The hatch is a model
    /// property and `required` a request property; the request wins.
    #[test]
    fn required_mode_keeps_the_grammar_despite_the_escape_hatch() {
        let choice = ToolChoice::Mode("required".to_string());
        let required = tool_choice_required_for_parser(true, Some(&choice), Some("qwen3_coder"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    #[test]
    fn specific_function_keeps_the_grammar_despite_the_escape_hatch() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "memory".to_string(),
            },
        };
        let required = tool_choice_required_for_parser(true, Some(&choice), Some("qwen3_coder"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    /// 2026-09-26: `minimax_xml` counts as tool-required whatever `tool_choice` says,
    /// so the hatch keeps its grammar too.
    #[test]
    fn minimax_xml_keeps_the_grammar_despite_the_escape_hatch() {
        let required = tool_choice_required_for_parser(true, None, Some("minimax_xml"));

        assert!(required);
        assert!(!tool_grammar_escape_applies(true, required));
    }

    /// 2026-09-26: With the hatch off it never applies, whatever `tool_choice` says.
    #[test]
    fn the_hatch_being_off_is_unaffected_by_tool_choice() {
        assert!(!tool_grammar_escape_applies(false, false));
        assert!(!tool_grammar_escape_applies(false, true));
    }

    #[test]
    fn bare_json_auto_uses_triggered_grammar() {
        assert!(!tool_choice_required_for_parser(
            true,
            None,
            Some("bare_json")
        ));
    }

    #[test]
    fn bare_json_required_mode_enforces_from_first_token() {
        let choice = ToolChoice::Mode("required".to_string());

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn specific_function_enforces_from_first_token() {
        let choice = ToolChoice::Specific {
            function: ToolChoiceFunction {
                name: "memory".to_string(),
            },
        };

        assert!(tool_choice_required_for_parser(
            true,
            Some(&choice),
            Some("bare_json")
        ));
    }

    #[test]
    fn minimax_xml_remains_parser_required() {
        assert!(tool_choice_required_for_parser(
            true,
            None,
            Some("minimax_xml")
        ));
    }

    #[test]
    fn inactive_tools_are_not_required() {
        assert!(!tool_choice_required_for_parser(
            false,
            None,
            Some("bare_json")
        ));
    }

    /// 2026-10-15: TEB TC-65 turn 2 as the bench sends it (tool_eval_bench orchestrator:
    /// `response_format` only after the first tool call): the tool result is in, tools are
    /// still declared, `tool_choice` auto, and the weather_report json_schema.
    fn tc65_turn2(tool_choice: serde_json::Value) -> crate::ir::ChatRequest {
        let body = serde_json::json!({
            "model": "glm",
            "messages": [
                {"role": "user", "content": "Get the current weather in Tokyo and output it as JSON matching this schema. Include a recommendation for what to wear."},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": "{\"location\": \"Tokyo\", \"temperature\": 28, \"units\": \"celsius\", \"condition\": \"Sunny\"}"}
            ],
            "tools": [
                {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a specific location",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}},
                        "required": ["location"], "additionalProperties": false}}},
                {"type": "function", "function": {"name": "web_search", "description": "Search the web for current information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
            ],
            "tool_choice": tool_choice,
            "response_format": {"type": "json_schema", "json_schema": {"name": "weather_report", "strict": true,
                "schema": {"type": "object", "properties": {"location": {"type": "string"},
                    "temperature_celsius": {"type": "number"}, "condition": {"type": "string"},
                    "recommendation": {"type": "string"}},
                    "required": ["location", "temperature_celsius", "condition", "recommendation"],
                    "additionalProperties": false}}}
        });
        let req: crate::openai::ChatCompletionRequest =
            serde_json::from_value(body).expect("chat request");
        req.into()
    }

    /// 2026-10-15: `request_grammar` as `build_sampling` calls it, with GLM-5.3's tool
    /// parser (`poolside_v1`, tool_defaults.toml) and tools active.
    fn grammar_for(
        req: &crate::ir::ChatRequest,
        with_tools: bool,
    ) -> Option<crate::api::inference_types::GrammarSpec> {
        let parser: std::sync::Arc<dyn crate::tool_parser::ToolCallParser> =
            crate::tool_parser::ToolCallFormat::PoolsideV1
                .into_parser()
                .into();
        let behavior = metrale_kernels::ModelBehavior {
            response_format_with_tools: with_tools,
            ..metrale_kernels::ModelBehavior::default()
        };
        let required =
            tool_choice_required_for_parser(true, req.tool_choice.as_ref(), Some(parser.name()));
        request_grammar(req, true, required, &behavior, Some(&parser))
    }

    fn named(name: &str) -> serde_json::Value {
        serde_json::json!({"type": "function", "function": {"name": name}})
    }

    #[test]
    fn with_the_flag_tools_auto_json_schema_gets_the_json_schema_grammar() {
        use crate::api::inference_types::GrammarSpec;
        let req = tc65_turn2(serde_json::json!("auto"));
        match grammar_for(&req, true) {
            Some(GrammarSpec::JsonSchema { schema }) => {
                assert!(schema.contains("temperature_celsius"), "{schema}");
            }
            other => panic!("want JsonSchema, got {:?}", other.map(|g| kind(&g))),
        }
    }

    #[test]
    fn with_the_flag_required_and_named_keep_the_tool_call_grammar() {
        use crate::api::inference_types::GrammarSpec;
        let req = tc65_turn2(serde_json::json!("required"));
        assert!(
            matches!(grammar_for(&req, true), Some(GrammarSpec::ToolCall { tools, use_triggers: false, .. }) if tools.len() == 2)
        );
        let req = tc65_turn2(named("get_weather"));
        match grammar_for(&req, true) {
            Some(GrammarSpec::ToolCall {
                tools,
                use_triggers,
                ..
            }) => {
                assert!(!use_triggers);
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0].function.name, "get_weather");
            }
            other => panic!("want ToolCall, got {:?}", other.map(|g| kind(&g))),
        }
    }

    #[test]
    fn without_the_flag_tools_keep_the_tool_call_grammar() {
        // 2026-10-15: Every other model: today's rule, the response format left to the model
        // while tools are active, enforced when `tool_choice` is `none`.
        use crate::api::inference_types::GrammarSpec;
        for choice in [
            serde_json::json!("auto"),
            serde_json::json!("required"),
            named("get_weather"),
        ] {
            let req = tc65_turn2(choice.clone());
            assert!(
                matches!(grammar_for(&req, false), Some(GrammarSpec::ToolCall { .. })),
                "{choice}"
            );
        }
        for with_tools in [false, true] {
            assert!(response_format_enforced(
                true, false, false, false, with_tools
            ));
            assert!(response_format_enforced(
                true, true, true, false, with_tools
            ));
            assert!(!response_format_enforced(
                false, true, false, false, with_tools
            ));
            assert!(!response_format_enforced(
                true, true, false, true, with_tools
            ));
        }
        assert!(!response_format_enforced(true, true, false, false, false));
        assert!(response_format_enforced(true, true, false, false, true));
    }

    fn kind(g: &crate::api::inference_types::GrammarSpec) -> &'static str {
        use crate::api::inference_types::GrammarSpec;
        match g {
            GrammarSpec::ToolCall { .. } => "ToolCall",
            GrammarSpec::JsonObject => "JsonObject",
            GrammarSpec::JsonSchema { .. } => "JsonSchema",
        }
    }

    #[test]
    fn tool_call_start_bias_is_unchanged_by_default_and_off_for_glm53() {
        let default = metrale_kernels::ModelBehavior::default();
        assert!(default.tool_call_start_bias);
        let ladder: Vec<Option<f32>> = (0..6)
            .map(|n| tool_call_start_bias(false, true, false, n, &default))
            .collect();
        assert_eq!(
            ladder,
            [
                Some(3.0),
                Some(3.0),
                None,
                Some(-5.0),
                Some(-10.0),
                Some(-10.0)
            ]
        );
        assert_eq!(tool_call_start_bias(true, true, false, 0, &default), None);
        assert_eq!(tool_call_start_bias(false, false, false, 0, &default), None);
        assert_eq!(tool_call_start_bias(false, true, true, 0, &default), None);
        // 2026-10-15: GLM-5.3 (`tool_call_start_bias = false`): no bias at any count.
        let glm = metrale_kernels::ModelBehavior {
            tool_call_start_bias: false,
            ..default
        };
        for n in 0..6 {
            assert_eq!(
                tool_call_start_bias(false, true, false, n, &glm),
                None,
                "{n}"
            );
        }
    }
}
