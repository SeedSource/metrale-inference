// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Lowers an OpenAI Responses-API request into a `ChatCompletionRequest`.
//!
//! Owner: server (OpenAI API layer).
//! Invariants: none beyond the types.

use super::*;

#[derive(Debug)]
pub enum LowerResponsesError {
    /// 2026-09-26: The request cannot be lowered: `input` is neither a string nor an array,
    /// a tool definition or `tool_choice` does not parse, or a tool type is a built-in
    /// hosted tool or unknown.
    BadRequest(String),
    /// 2026-09-26: `previous_response_id` was set but the resolver returned nothing. The
    /// Responses handler answers 400 with `param=previous_response_id` and
    /// `code=response_not_found`.
    PriorNotFound(String),
}

impl LowerResponsesError {
    pub fn message(&self) -> &str {
        match self {
            Self::BadRequest(m) | Self::PriorNotFound(m) => m.as_str(),
        }
    }
}

/// 2026-09-26: Lower an OpenAI Responses-API request into a `ChatCompletionRequest` for
/// the chat-completions pipeline.
///
/// When `previous_response_id` is set, `resolve_prior` is called with it (the Responses
/// handler passes a closure over [`crate::response_store::ResponseStore`]) and the
/// returned transcript goes before the current input. `None` from the resolver yields
/// [`LowerResponsesError::PriorNotFound`].
pub fn lower_responses_to_chat(
    r: ResponsesRequest,
    resolve_prior: impl FnOnce(&str) -> Option<Vec<IncomingMessage>>,
) -> Result<ChatCompletionRequest, LowerResponsesError> {
    let mut messages: Vec<IncomingMessage> = Vec::new();

    if let Some(prior_id) = r.previous_response_id.as_deref() {
        match resolve_prior(prior_id) {
            Some(prior) => messages.extend(prior),
            None => {
                return Err(LowerResponsesError::PriorNotFound(format!(
                    "previous_response_id '{prior_id}' not found or expired"
                )));
            }
        }
    }

    if let Some(instr) = r.instructions.clone() {
        // 2026-09-26: `instructions` becomes a system message at index 0, ahead of the
        // resumed transcript; system messages inside the transcript are kept.
        messages.insert(0, IncomingMessage::synthetic_system(instr));
    }
    match &r.input {
        serde_json::Value::String(s) => {
            messages.push(IncomingMessage::synthetic_user_text(s.clone()));
        }
        serde_json::Value::Array(items) => {
            for it in items {
                if let Some(m) = IncomingMessage::from_responses_input_item(it) {
                    messages.push(m);
                }
            }
        }
        _ => {
            return Err(LowerResponsesError::BadRequest(
                "`input` must be a string or array of input items".into(),
            ));
        }
    }

    // 2026-09-26: Function tools (type `function`, `namespace` or absent) are parsed into
    // `ToolDefinition`; a built-in hosted tool type or any other type fails the request.
    // A list that parses to no tools becomes `None`.
    let tools: Option<Vec<crate::tool_parser::ToolDefinition>> = match r.tools {
        None => None,
        Some(list) => {
            // 2026-09-26: Not pre-sized from `list.len()`: that count comes from the request
            // body, and an allocation sized by it is an uncontrolled-allocation sink (CWE-789).
            let mut parsed = Vec::new();
            for raw in list {
                let ty = raw.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "function" | "" | "namespace" => {
                        // 2026-09-26: A tool without a nested `function` object is in the flat
                        // Responses form; its `name`, `description`, `parameters` and `strict`
                        // are moved into one so the chat-format `ToolDefinition` accepts it.
                        let normalized = if raw.get("function").is_some() {
                            raw
                        } else if let Some(obj) = raw.as_object() {
                            let mut function = serde_json::Map::new();
                            for key in ["name", "description", "parameters", "strict"] {
                                if let Some(v) = obj.get(key) {
                                    function.insert(key.to_string(), v.clone());
                                }
                            }
                            serde_json::json!({
                                "type": "function",
                                "function": serde_json::Value::Object(function),
                            })
                        } else {
                            raw
                        };
                        match serde_json::from_value::<crate::tool_parser::ToolDefinition>(
                            normalized,
                        ) {
                            Ok(td) => parsed.push(td),
                            Err(e) => {
                                return Err(LowerResponsesError::BadRequest(format!(
                                    "invalid tool definition: {e}"
                                )));
                            }
                        }
                    }
                    builtin @ ("web_search"
                    | "web_search_preview"
                    | "file_search"
                    | "computer_use_preview"
                    | "code_interpreter"
                    | "image_generation"
                    | "mcp"
                    | "local_shell"
                    | "custom_tool") => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "built-in tool '{builtin}' is not supported by this server. Metrale Engine serves inference only and does not ship hosted tools (web search, file search, code interpreter, computer use, image generation, MCP). Provide your own `function`-type tools instead."
                        )));
                    }
                    other => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "unknown tool type '{other}'. Supported types: 'function'."
                        )));
                    }
                }
            }
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        }
    };
    Ok(ChatCompletionRequest {
        model: r.model,
        adapter: None,
        src_lang: None,
        tgt_lang: None,
        num_beams: None,
        length_penalty: None,
        early_stopping: None,
        messages,
        max_tokens: r.max_output_tokens.unwrap_or_else(default_max_tokens),
        temperature: r.temperature,
        top_k: None,
        top_p: r.top_p,
        top_n_sigma: None,
        min_p: None,
        repetition_penalty: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        stream: r.stream,
        return_token_ids: false,
        enable_thinking: None,
        thinking: None,
        thinking_token_budget: None,
        repetition_detection: None,
        reasoning: r.reasoning,
        chat_template_kwargs: None,
        tools,
        tool_choice: match r.tool_choice {
            None => None,
            Some(raw) => {
                let normalized = match raw.as_str() {
                    Some(_) => raw,
                    None => match raw.as_object() {
                        Some(obj) if obj.get("function").is_some() => raw,
                        Some(obj)
                            if obj.get("type").and_then(|v| v.as_str()) == Some("function")
                                && obj.get("name").is_some() =>
                        {
                            let mut function = serde_json::Map::new();
                            function.insert(
                                "name".to_string(),
                                obj.get("name").cloned().unwrap_or(serde_json::Value::Null),
                            );
                            serde_json::json!({
                                "type": "function",
                                "function": serde_json::Value::Object(function),
                            })
                        }
                        _ => raw,
                    },
                };
                match serde_json::from_value::<crate::tool_parser::ToolChoice>(normalized) {
                    Ok(tc) => Some(tc),
                    Err(e) => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "invalid tool_choice: {e}"
                        )));
                    }
                }
            }
        },
        stop: Vec::new(),
        response_format: None,
        min_tokens: 0,
        ignore_eos: false,
        seed: None,
        logprobs: None,
        top_logprobs: None,
        timeout: None,
        n: 1,
        stream_options: None,
        parallel_tool_calls: None,
        verbosity: None,
        service_tier: r.service_tier,
        store: r.store,
        metadata: r.metadata,
        safety_identifier: None,
        prompt_cache_key: None,
        user: None,
        modalities: None,
        audio: None,
        prediction: None,
        web_search_options: None,
        reasoning_effort: None,
    })
}
