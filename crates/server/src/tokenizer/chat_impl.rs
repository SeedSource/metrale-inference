// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ChatTokenizer` construction (template selection), encode and decode, and the
//! chat-template apply paths.
//!
//! Owner: server (tokenizer).
//! Invariants: none beyond the types.

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

#[path = "chat_impl/vision_pads.rs"]
mod vision_pads;

use super::{
    ChatEncoding, ChatTokenizer, StreamingDecoder, autoclose_assistant_think,
    normalize_tool_call_arguments, remap_developer_role, resolve_think_control,
};

/// 2026-09-26: Message rewrites applied before every Jinja render (`render_chat`), in order:
/// parse string tool-call arguments (`normalize_tool_call_arguments`), turn `developer`
/// messages into `system` (`remap_developer_role`), close an open `<think>` before a
/// `<tool_call>` in assistant history (`autoclose_assistant_think`), and strip inline
/// `<|think_on|>`/`<|think_off|>` tokens (`resolve_think_control`).
///
/// Returns the rewritten messages and the thinking flag to render with: the last inline
/// control token when there is one, else `enable_thinking`.
pub(crate) fn preprocess_for_render(
    messages: &[serde_json::Value],
    enable_thinking: bool,
) -> (Vec<serde_json::Value>, bool) {
    let prepared = normalize_tool_call_arguments(messages);
    let mut prepared = remap_developer_role(prepared);
    autoclose_assistant_think(&mut prepared);
    let (prepared, control_override) = resolve_think_control(&prepared);
    let effective_thinking = control_override.unwrap_or(enable_thinking);
    (prepared, effective_thinking)
}

impl ChatTokenizer {
    pub fn from_model_dir(
        model_dir: &Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        repo_root: Option<&Path>,
        disable_template_overrides: bool,
    ) -> Result<Self> {
        Self::from_model_dir_with(
            model_dir,
            eos_token_id,
            supports_thinking,
            model_type,
            repo_root,
            disable_template_overrides,
            false,
        )
    }

    /// 2026-10-15: [`Self::from_model_dir`] with MODEL.toml `[behavior].hf_tool_json`:
    /// `true` renders the templates' `tojson` in the spaced HF form whatever
    /// `METRALE_USE_HF_REF_JSON_DUMPS` says (`jinja_helpers::build_jinja_env_for`).
    pub fn from_model_dir_with(
        model_dir: &Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        repo_root: Option<&Path>,
        disable_template_overrides: bool,
        hf_tool_json: bool,
    ) -> Result<Self> {
        let official_k3 = super::kimi_k3::uses_xtml(model_dir, model_type)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("Failed to disable tokenizer truncation: {e}"))?;

        // 2026-09-26: Template source, first match wins: an official Kimi K3 checkpoint gets
        // no chat template; then `jinja-templates/{model_type}.jinja` when the file exists and
        // `--disable-template-overrides` is off; then the model's own template
        // (`load_config_template`); then the ChatML default.
        let override_tmpl = if disable_template_overrides {
            None
        } else {
            super::jinja_helpers::load_override_template(model_type, repo_root)
        };
        let (chat_template, checkpoint_template) = if official_k3 {
            tracing::warn!("Official Kimi K3: raw completions only; XTML chat is unavailable");
            (String::new(), false)
        } else if let Some(override_tmpl) = override_tmpl {
            (override_tmpl, false)
        } else if let Some(config_tmpl) = super::jinja_helpers::load_config_template(model_dir)? {
            (config_tmpl, true)
        } else {
            tracing::warn!("No chat template found — using default ChatML");
            (
                super::jinja_helpers::default_chatml_template(supports_thinking),
                false,
            )
        };

        let jinja_env = super::jinja_helpers::build_jinja_env_for(&chat_template, hf_tool_json)?;

        // 2026-09-26: A variant template that fails to compile is dropped without a log line.
        let openai_jinja_env = super::jinja_helpers::load_openai_template(model_type, repo_root)
            .and_then(|tmpl| {
                tracing::info!("Loaded OpenAI-variant Jinja template for {model_type}");
                super::jinja_helpers::build_jinja_env_for(&tmpl, hf_tool_json).ok()
            });
        let chat_encoding = if official_k3 {
            ChatEncoding::KimiK3XtmlUnsupported
        } else if model_type == "deepseek_v4" || model_type == "deepseek_v41" {
            tracing::info!("Using checkpoint-native DeepSeek-V4 message encoding");
            ChatEncoding::DeepseekV4
        } else {
            ChatEncoding::Jinja
        };

        let native_qwen_tool_template = checkpoint_owns_qwen_tool_prompt(
            model_type,
            checkpoint_template,
            openai_jinja_env.is_some(),
        ) && jinja_env
            .get_template("chat")?
            .undeclared_variables(false)
            .contains("tools");
        tracing::info!("Loaded tokenizer from {}", tokenizer_path.display());
        Ok(Self {
            tokenizer,
            eos_token_id,
            supports_thinking,
            chat_encoding,
            native_qwen_tool_template,
            chat_template,
            jinja_env,
            openai_jinja_env,
        })
    }

    pub(crate) fn uses_native_qwen_tool_template(&self) -> bool {
        self.native_qwen_tool_template
    }

    /// 2026-09-26: The underlying Hugging Face tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
    }

    /// 2026-09-26: Decode without skipping special tokens; `decode` skips them.
    pub fn decode_with_special(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {e}"))
    }

    /// 2026-09-26: Incremental detokenizer. Returns the text `toks` added since the last call
    /// and advances the offsets.
    ///
    /// Each call decodes only `toks[prefix_offset..read_offset]` and `toks[prefix_offset..]`
    /// (with `decode`, so special tokens are skipped) and returns what the second adds to the
    /// first. It returns an empty string and leaves the offsets alone when the second decode
    /// is not longer, ends in U+FFFD, or does not split at a char boundary. Offsets that no
    /// longer fit `toks` are reset to 0 first.
    pub fn incremental_decode(
        &self,
        toks: &[u32],
        prefix_offset: &mut usize,
        read_offset: &mut usize,
    ) -> String {
        if *read_offset > toks.len() || *prefix_offset > *read_offset {
            *prefix_offset = 0;
            *read_offset = 0;
        }
        let prefix_text = self
            .decode(&toks[*prefix_offset..*read_offset])
            .unwrap_or_default();
        let new_text = self.decode(&toks[*prefix_offset..]).unwrap_or_default();
        if new_text.len() > prefix_text.len()
            && !new_text.ends_with('\u{FFFD}')
            && let Some(delta) = new_text.get(prefix_text.len()..)
        {
            let delta = delta.to_string();
            *prefix_offset = *read_offset;
            *read_offset = toks.len();
            return delta;
        }
        String::new()
    }

    /// 2026-09-26: A new [`StreamingDecoder`] over this tokenizer.
    pub fn streaming_decoder(&self, skip_special_tokens: bool) -> StreamingDecoder<'_> {
        StreamingDecoder {
            inner: self.tokenizer.decode_stream(skip_special_tokens),
        }
    }

    /// 2026-09-26: `apply_chat_template_jinja_with_effort` with no reasoning effort and
    /// `preserve_thinking` unset.
    pub fn apply_chat_template_jinja(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
    ) -> Result<Vec<u32>> {
        self.apply_chat_template_jinja_with_effort(
            messages,
            tools,
            enable_thinking,
            disable_tool_steering,
            None,
            None,
        )
    }

    pub fn apply_chat_template_jinja_with_effort(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: Option<bool>,
    ) -> Result<Vec<u32>> {
        super::kimi_k3::require_chat_support(self.chat_encoding)?;
        if self.chat_encoding == ChatEncoding::DeepseekV4 {
            let rendered = super::deepseek_v4::encode_messages(
                messages,
                tools,
                enable_thinking,
                reasoning_effort,
            )?;
            return self.encode(&rendered);
        }

        let rendered = super::chat_render::render_chat(
            &self.jinja_env,
            messages,
            tools,
            super::chat_render::RenderFlags {
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
                allow_continue_final: true,
            },
        )?;

        if rendered.len() < 2000 {
            let tail_start = rendered.floor_char_boundary(rendered.len().saturating_sub(200));
            tracing::info!(
                "Jinja rendered ({} chars): {:?}",
                rendered.len(),
                &rendered[tail_start..]
            );
        }

        self.encode(&rendered)
    }

    /// 2026-09-26: `apply_chat_template_openai_with_effort` with no reasoning effort and
    /// `preserve_thinking` unset.
    pub fn apply_chat_template_openai(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
    ) -> Result<Vec<u32>> {
        self.apply_chat_template_openai_with_effort(
            messages,
            tools,
            enable_thinking,
            disable_tool_steering,
            None,
            None,
        )
    }

    pub fn apply_chat_template_openai_with_effort(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
        disable_tool_steering: bool,
        reasoning_effort: Option<&str>,
        preserve_thinking: Option<bool>,
    ) -> Result<Vec<u32>> {
        super::kimi_k3::require_chat_support(self.chat_encoding)?;
        if self.chat_encoding == ChatEncoding::DeepseekV4 {
            return self.apply_chat_template_jinja_with_effort(
                messages,
                tools,
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
            );
        }
        if let Some(ref env) = self.openai_jinja_env {
            let rendered = super::chat_render::render_chat(
                env,
                messages,
                tools,
                super::chat_render::RenderFlags {
                    enable_thinking,
                    disable_tool_steering,
                    reasoning_effort,
                    preserve_thinking,
                    allow_continue_final: false,
                },
            )
            .map_err(|e| anyhow::anyhow!("Failed to render OpenAI Jinja template: {e}"))?;
            self.encode(&rendered)
        } else {
            self.apply_chat_template_jinja_with_effort(
                messages,
                tools,
                enable_thinking,
                disable_tool_steering,
                reasoning_effort,
                preserve_thinking,
            )
        }
    }

    /// 2026-09-26: Render `(role, content)` pairs through `apply_chat_template_jinja`, with no
    /// tools; `_image_pad_counts` is unused.
    pub fn apply_chat_template(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
        _image_pad_counts: &[usize],
    ) -> Result<Vec<u32>> {
        let json_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| {
                serde_json::json!({
                    "role": role,
                    "content": content,
                })
            })
            .collect();

        self.apply_chat_template_jinja(&json_messages, None, enable_thinking, false)
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn think_end_token_id(&self) -> Option<u32> {
        if !self.supports_thinking {
            return None;
        }
        match self.encode("</think>") {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        }
    }

    pub fn supports_thinking(&self) -> bool {
        self.supports_thinking
    }

    pub fn uses_deepseek_v4_encoding(&self) -> bool {
        self.chat_encoding == ChatEncoding::DeepseekV4
    }
}

/// 2026-09-26: True only for the checkpoint's own template (not an override or the ChatML
/// default), with no OpenAI variant, on model types `qwen3_5`, `qwen3_5_moe`, `qwen3_6` and
/// `qwen3_6_moe`. `from_model_dir` also requires the template to use `tools`.
fn checkpoint_owns_qwen_tool_prompt(
    model_type: &str,
    checkpoint: bool,
    openai_override: bool,
) -> bool {
    checkpoint
        && !openai_override
        && matches!(
            model_type,
            "qwen3_5" | "qwen3_5_moe" | "qwen3_6" | "qwen3_6_moe"
        )
}
