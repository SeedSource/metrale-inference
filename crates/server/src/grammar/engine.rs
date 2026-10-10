// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The server-wide grammar engine: one `GrammarCompiler` bound
//! to the model's tokenizer.
//!
//! Owner: server (grammar).
//! Invariants: none beyond the types.

use metrale_grammar::{GrammarCompiler, TokenizerInfo, VocabType, detect_metadata_from_hf};

/// 2026-09-26: Cache budget passed to `GrammarCompiler::new`. The compiler
/// gives `limit / 3 * 2` bytes to its compiled-grammar cache and the rest to
/// its rule-level mask cache, and evicts least-recently-used entries from
/// each to stay inside its share.
const GRAMMAR_CACHE_BUDGET_BYTES: isize = 1024 * 1024 * 1024;

/// 2026-10-10: `METRALE_GRAMMAR_CACHE_MB` overrides [`GRAMMAR_CACHE_BUDGET_BYTES`]
/// in MiB (at least 16). The whole cache lives in the host memory of the rank
/// that runs grammar, and one compiled grammar over a ~155K-token vocab is
/// ~20 MB, so the budget sets how many distinct tool sets stay cached; a miss
/// recompiles. Unset or blank keeps 1 GiB; anything else that does not parse
/// is an error, so a typo stops model load instead of silently using 1 GiB.
pub(crate) fn cache_budget_bytes(raw: Option<&str>) -> Result<isize, String> {
    let mb = crate::env_config::parse_min::<isize>(
        "METRALE_GRAMMAR_CACHE_MB",
        raw,
        16,
        "the grammar cache budget in MiB",
    )?;
    Ok(mb.map_or(GRAMMAR_CACHE_BUDGET_BYTES, |mb| mb * 1024 * 1024))
}

use super::extract_ordered_vocab;

/// 2026-09-25: The grammar compiler shared by every request to a loaded model.
/// Serving builds one with [`Self::from_tokenizer`] on each model load
/// (`serve_phases/tokenizer_runtime.rs`: at startup and on a model swap).
/// Compiled grammars are cached by their full request text (schema, EBNF or
/// structural-tag JSON).
pub struct GrammarEngine {
    pub(super) compiler: GrammarCompiler,
    vocab_size: usize,
    /// 2026-09-26: Where the prewarm hook saves the rule-level mask cache.
    /// `None` until [`Self::attach_mask_cache`] runs, and when that finds
    /// persistence off or the rule cache absent. See [`super::mask_cache`].
    pub(super) snapshot: Option<super::mask_cache::MaskSnapshot>,
    /// 2026-10-10: Compile calls so far; drives the periodic cache-stats log.
    compiles: std::sync::atomic::AtomicU64,
}

// 2026-09-26: SAFETY: every field is already `Send` (`GrammarCompiler`,
// `usize`, and `MaskSnapshot`'s `PathBuf`, `SnapshotIdentity`,
// `RuleLevelCache` and `Arc` atomics), so this asserts nothing the auto
// trait does not.
unsafe impl Send for GrammarEngine {}

/// 2026-09-26: Errors from building an engine, compiling a grammar or
/// creating a matcher.
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("grammar compilation failed: {0}")]
    Compilation(String),
    #[error("invalid JSON schema: {0}")]
    InvalidSchema(String),
    #[error("no tools provided")]
    NoTools,
}

impl GrammarEngine {
    /// 2026-09-26: Build an engine from a vocabulary ordered by token id
    /// (`vocab[i]` is token `i`) and the model's stop token ids.
    ///
    /// Uses [`VocabType::RAW`]: each string is taken as the token's literal
    /// bytes. That is wrong for a vocabulary stored in GPT-2 ByteLevel form
    /// (`Ġ` for a space, `Ċ` for `\n`) or with `<0xHH>` / `▁` byte-fallback
    /// pieces, which need decoding first. For a HuggingFace tokenizer use
    /// [`Self::from_tokenizer`], which detects the vocabulary type.
    pub fn new(vocab: &[String], stop_token_ids: &[i32]) -> Result<Self, GrammarError> {
        let stop: Option<Box<[i32]>> = if stop_token_ids.is_empty() {
            None
        } else {
            Some(stop_token_ids.to_vec().into_boxed_slice())
        };
        let tokenizer_info = TokenizerInfo::new(vocab, VocabType::RAW, &stop, false)
            .map_err(GrammarError::Compilation)?;
        Self::from_tokenizer_info(tokenizer_info)
    }

    /// 2026-09-26: Build an engine from a HuggingFace tokenizer. The
    /// vocabulary type (raw, byte-fallback or ByteLevel) and
    /// `add_prefix_space` come from [`detect_metadata_from_hf`], which reads
    /// the tokenizer's serialized JSON: `metrale-grammar` does not depend on
    /// the `tokenizers` crate.
    ///
    /// `vocab_size`, when given (serving passes the model config's), wins
    /// over the tokenizer's count: the vocabulary is padded with empty
    /// strings or truncated to it.
    pub fn from_tokenizer(
        tokenizer: &tokenizers::Tokenizer,
        vocab_size: Option<usize>,
        stop_token_ids: &[i32],
    ) -> Result<Self, GrammarError> {
        let backend_str = tokenizer
            .to_string(false)
            .map_err(|e| GrammarError::Compilation(format!("tokenizer serialize failed: {e}")))?;
        let metadata = detect_metadata_from_hf(&backend_str).map_err(GrammarError::Compilation)?;
        let detected_label = match metadata.vocab_type {
            VocabType::RAW => "raw",
            VocabType::BYTE_FALLBACK => "byte_fallback",
            VocabType::BYTE_LEVEL => "byte_level",
        };
        tracing::info!(
            "Grammar: detected tokenizer vocab_type={detected_label}, add_prefix_space={}",
            metadata.add_prefix_space,
        );

        let ordered_vocab = extract_ordered_vocab(tokenizer);
        let model_vocab_size = vocab_size.unwrap_or(ordered_vocab.len());
        let mut sized_vocab = if model_vocab_size > ordered_vocab.len() {
            let mut v = ordered_vocab;
            v.resize(model_vocab_size, String::new());
            v
        } else if model_vocab_size < ordered_vocab.len() {
            let mut v = ordered_vocab;
            v.truncate(model_vocab_size);
            v
        } else {
            ordered_vocab
        };
        // 2026-09-26: This borrow is the only use of the `mut` binding;
        // without it rustc warns `unused_mut`.
        let _ = &mut sized_vocab;

        let stop: Option<Box<[i32]>> = if stop_token_ids.is_empty() {
            None
        } else {
            Some(stop_token_ids.to_vec().into_boxed_slice())
        };
        let tokenizer_info = TokenizerInfo::new(
            &sized_vocab,
            metadata.vocab_type,
            &stop,
            metadata.add_prefix_space,
        )
        .map_err(GrammarError::Compilation)?;
        Self::from_tokenizer_info(tokenizer_info)
    }

    fn from_tokenizer_info(tokenizer_info: TokenizerInfo) -> Result<Self, GrammarError> {
        let vocab_size = tokenizer_info.vocab_size();
        // 2026-09-26: `max_threads = 1`, so `compile_top_k_masks` warms
        // masks on one thread. The budget is finite, not `-1` (unlimited),
        // because the compiled-grammar cache keys by the full request text:
        // every distinct tool set adds an entry, and only the budget's LRU
        // eviction removes them.
        let budget = cache_budget_bytes(std::env::var("METRALE_GRAMMAR_CACHE_MB").ok().as_deref())
            .map_err(GrammarError::Compilation)?;
        let compiler = GrammarCompiler::new(&tokenizer_info, 1, true, budget)
            .map_err(GrammarError::Compilation)?;
        Ok(Self {
            compiler,
            vocab_size,
            snapshot: None,
            compiles: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// 2026-10-10: A177: count a compile call and, every 50th, log what the
    /// grammar cache accounts for, so a rank-0 host-memory climb can be
    /// compared with the accounted bytes.
    pub(super) fn note_compile(&self) {
        let n = self
            .compiles
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if n % 50 == 0 {
            tracing::info!(
                "Grammar cache: compiles={n} accounted_bytes={} vocab_shared_bytes={} entries={}",
                self.compiler.get_cache_size_bytes(),
                self.compiler.vocab_shared_bytes(),
                self.compiler.cache_len(),
            );
        }
    }

    /// 2026-09-26: Vocabulary size the grammar was compiled against; a
    /// bitmask holds `ceil(vocab_size / 32)` `i32` words.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
