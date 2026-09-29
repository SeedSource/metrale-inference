// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What the serve derives from the tokenizer: the vocab cap, the
//! reasoning parser, special token ids, the per-token vocab masks, the
//! hard-stop ids and the grammar engine.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;

use crate::cli;

pub(crate) struct TokenizerRuntime {
    pub(crate) reasoning_parser_box: Option<Box<dyn crate::reasoning_parser::ReasoningParser>>,
    pub(crate) think_end_token: Option<u32>,
    pub(crate) think_start_token: Option<u32>,
    /// 2026-09-26: Id of ``` when it is a single token. The scheduler tracks
    /// fences with it (`confidence::toggle_code_fence`), and a forced
    /// `</think>` waits while inside one (`should_inject_think_end`). `None`
    /// when the tokenizer splits ```; the fence state then never changes.
    pub(crate) code_fence_token: Option<u32>,
    pub(crate) tool_call_start_token: Option<u32>,
    pub(crate) tool_call_end_token: Option<u32>,
    pub(crate) grammar_engine: Option<crate::grammar::GrammarEngine>,
    /// 2026-09-26: Per-token classification masks for this tokenizer's
    /// vocabulary.
    pub(crate) vocab_masks: crate::scheduler::vocab_masks::VocabMasks,
    /// 2026-09-26: This tokenizer's hard-stop token ids. `max_seq_len` is 0
    /// here; the caller fills it in.
    pub(crate) limits: crate::scheduler::limits::SchedLimits,
}

pub(crate) fn resolve_tokenizer_runtime(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
    tokenizer: &crate::tokenizer::ChatTokenizer,
    eos_tokens: &mut Vec<u32>,
    supports_thinking: bool,
    // 2026-09-26: The checkpoint directory; the grammar mask-cache snapshot
    // is kept under it unless `METRALE_GRAMMAR_CACHE_DIR` is set.
    model_dir: &std::path::Path,
) -> TokenizerRuntime {
    use crate::{grammar, reasoning_parser};

    let mut vocab_masks = crate::scheduler::vocab_masks::VocabMasks::default();

    let tokenizer_vocab = tokenizer.inner().get_vocab_size(true);
    if tokenizer_vocab > 0 && tokenizer_vocab < config.vocab_size {
        tracing::info!(
            "Capping vocab_size from {} (config) to {} (tokenizer incl. special tokens)",
            config.vocab_size,
            tokenizer_vocab,
        );
        config.vocab_size = tokenizer_vocab;
    }

    let reasoning_parser_box: Option<Box<dyn reasoning_parser::ReasoningParser>> = {
        let defaults_toml = include_str!("../../../tool_defaults.toml");
        let defaults: toml::Value =
            toml::from_str(defaults_toml).unwrap_or(toml::Value::Table(Default::default()));
        let auto_format = defaults
            .get("reasoning")
            .and_then(|t| t.get(config.model_type.as_str()))
            .and_then(|s| s.as_str())
            .and_then(|s| s.parse::<reasoning_parser::ReasoningFormat>().ok());
        if let Some(fmt) = auto_format {
            let p = fmt.into_parser();
            tracing::info!(
                "Reasoning parser: {} (auto-detected from model_type '{}')",
                p.name(),
                config.model_type
            );
            Some(p)
        } else if supports_thinking {
            let p = reasoning_parser::ReasoningFormat::Qwen.into_parser();
            tracing::info!(
                "Reasoning parser: {} (default for thinking-capable model)",
                p.name()
            );
            Some(p)
        } else {
            None
        }
    };
    let think_end_token = reasoning_parser_box
        .as_ref()
        .and_then(|p| p.end_token_id(tokenizer));
    let think_start_token: Option<u32> = tokenizer
        .encode("<think>")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    let code_fence_token: Option<u32> = tokenizer
        .encode("```")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(fid) = code_fence_token {
        tracing::info!("Code-fence token: {} (``` — F2 fence guard active)", fid);
    }

    // 2026-09-26: Numeric mask: true when the token decodes to ASCII digits
    // with at most one leading space. The content-loop watchdog reads it to
    // match loops that differ only in their numbers. A decode error leaves
    // the id false.
    {
        let vocab_size = tokenizer.inner().get_vocab_size(true);
        let mut mask: Vec<bool> = vec![false; vocab_size];
        let mut numeric_count = 0usize;
        for (id, slot) in mask.iter_mut().enumerate() {
            if let Ok(s) = tokenizer.decode_with_special(&[id as u32]) {
                let body = s.strip_prefix(' ').unwrap_or(&s);
                if !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit()) {
                    *slot = true;
                    numeric_count += 1;
                }
            }
        }
        vocab_masks.numeric = Some(std::sync::Arc::from(mask));
        tracing::info!(
            "Numeric-token mask: {numeric_count}/{vocab_size} ids classified \
             as digit-runs (digit-normalized content-loop path active)"
        );
    }

    // 2026-09-26: Boundary mask: true when the decoded text ends in a
    // newline, or in `.`, `!` or `?` followed only by spaces, tabs, `\r`,
    // quotes or closing brackets. Read by the rollback and by the forced
    // `</think>` injection. A decode error leaves the id false.
    {
        let vocab_size = tokenizer.inner().get_vocab_size(true);
        let mut mask: Vec<bool> = vec![false; vocab_size];
        let mut boundary_count = 0usize;
        let is_boundary = |s: &str| -> bool {
            let trimmed = s.trim_end_matches([' ', '\t', '"', '\'', ')', ']', '}', '\r']);
            match trimmed.chars().last() {
                Some('\n') => true,
                Some('.') | Some('!') | Some('?') => true,
                _ => s.ends_with('\n'),
            }
        };
        for (id, slot) in mask.iter_mut().enumerate() {
            if let Ok(s) = tokenizer.decode_with_special(&[id as u32])
                && !s.is_empty()
                && is_boundary(&s)
            {
                *slot = true;
                boundary_count += 1;
            }
        }
        vocab_masks.boundary = Some(std::sync::Arc::from(mask));
        tracing::info!(
            "Boundary-token mask: {boundary_count}/{vocab_size} ids end in a \
             newline / sentence boundary (Phase-C rollback-to-boundary active)"
        );
    }

    // 2026-09-26: Mid-word mask: true when the decoded text ends in an
    // alphanumeric character. `MidWordThinkEndMask` masks `</think>` after
    // such a token while thinking. A decode error leaves the id false.
    {
        let vocab_size = tokenizer.inner().get_vocab_size(true);
        let mut mask: Vec<bool> = vec![false; vocab_size];
        let mut mid_word_count = 0usize;
        for (id, slot) in mask.iter_mut().enumerate() {
            if let Ok(s) = tokenizer.decode_with_special(&[id as u32])
                && !s.is_empty()
                && let Some(last_ch) = s.chars().last()
                && last_ch.is_alphanumeric()
            {
                *slot = true;
                mid_word_count += 1;
            }
        }
        vocab_masks.mid_word = Some(std::sync::Arc::from(mask));
        tracing::info!(
            "Mid-word token mask: {mid_word_count}/{vocab_size} ids end in alphanumeric \
             (mid-word </think> defer active during thinking)"
        );
    }

    if let Some(tid) = think_end_token {
        tracing::info!(
            "Thinking end token: {} ({})",
            tid,
            reasoning_parser_box.as_ref().unwrap().end_tag()
        );
    }
    if let Some(tid) = think_start_token {
        tracing::info!("Thinking start token: {tid} (<think>)");
    }

    let im_start_id: Option<u32> = tokenizer
        .encode("<|im_start|>")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(id) = im_start_id {
        if !eos_tokens.contains(&id) {
            eos_tokens.push(id);
        }
        tracing::info!("ChatML role-boundary hard stop: <|im_start|> (id {id}) registered");
    }

    // 2026-09-26: `<tool_response>`, when it is a single token, is kept out of
    // `eos_tokens`: it stops a sequence only through
    // `SchedLimits::tool_response_hard_stop` while the `tool_response_stop`
    // lever (`METRALE_TOOL_RESPONSE_STOP`, on by default) is set.
    let tool_response_id: Option<u32> = tokenizer
        .encode("<tool_response>")
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(id) = tool_response_id {
        tracing::info!("Tool-response hard stop: <tool_response> (id {id}) registered");
    }

    let tool_call_format_name: Option<String> = args.tool_call_parser.clone().or_else(|| {
        let defaults: toml::Table = toml::from_str(include_str!("../../../tool_defaults.toml"))
            .expect("invalid tool_defaults.toml");
        defaults
            .get("model_type")
            .and_then(|t| t.as_table())
            .and_then(|t| t.get(config.model_type.as_str()))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    });
    let (tc_start_str, tc_end_str): (&str, &str) = match tool_call_format_name.as_deref() {
        Some("minimax_xml") => ("<minimax:tool_call>", "</minimax:tool_call>"),
        _ => ("<tool_call>", "</tool_call>"),
    };
    let tool_call_start_token = tokenizer
        .encode(tc_start_str)
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(tid) = tool_call_start_token {
        tracing::info!("Tool call start token: {} ({})", tid, tc_start_str);
    } else {
        tracing::warn!(
            "Tool call start token unresolved for {tc_start_str} — \
             require_tool_call / suppress / force-emit-after-think will be no-ops"
        );
    }
    let tool_call_end_token = tokenizer
        .encode(tc_end_str)
        .ok()
        .and_then(|ids| if ids.len() == 1 { Some(ids[0]) } else { None });
    if let Some(tid) = tool_call_end_token {
        tracing::info!("Tool call end token: {} ({})", tid, tc_end_str);
    }

    let grammar_engine = {
        let stop_ids: Vec<i32> = eos_tokens.iter().map(|&id| id as i32).collect();
        let model_vocab_size = Some(config.vocab_size);
        match grammar::GrammarEngine::from_tokenizer(tokenizer.inner(), model_vocab_size, &stop_ids)
        {
            Ok(mut engine) => {
                tracing::info!(
                    "Grammar engine initialized (vocab_size={}, vocab_type=auto-detected from tokenizer)",
                    engine.vocab_size()
                );
                // 2026-09-26: Load the mask cache snapshot and arm its saver.
                // This runs on every model load, a model swap included.
                engine.attach_mask_cache(model_dir);
                Some(engine)
            }
            Err(e) => {
                tracing::warn!("Grammar engine init failed (constrained decoding disabled): {e}");
                None
            }
        }
    };

    TokenizerRuntime {
        limits: crate::scheduler::limits::SchedLimits {
            im_start_hard_stop: im_start_id,
            tool_response_hard_stop: tool_response_id,
            max_seq_len: 0,
            code_fence_token,
        },
        vocab_masks,
        reasoning_parser_box,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
    }
}
