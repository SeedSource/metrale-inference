// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Grammar compilation other than tool calls: raw structural tags, JSON
//! schema, any JSON, and EBNF.
//!
//! Owner: server grammar.
//! Invariants: none beyond the types.

use metrale_grammar::CompiledGrammar;

use super::engine::{GrammarEngine, GrammarError};

impl GrammarEngine {
    /// 2026-09-26: Compile a `triggered_tags` structural tag built from its parts:
    /// `triggers`, `tags`, `at_least_one` and `stop_after_first` go into the JSON document
    /// as given.
    pub(super) fn compile_structural_tag_raw(
        &mut self,
        triggers: &[String],
        tags: &[serde_json::Value],
        at_least_one: bool,
        stop_after_first: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        let structural_tag_json = serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags,
                "at_least_one": at_least_one,
                "stop_after_first": stop_after_first,
            }
        })
        .to_string();

        let grammar = metrale_grammar::Grammar::from_structural_tag(&structural_tag_json)
            .map_err(GrammarError::Compilation)?;
        let compiled = self
            .compiler
            .compile_grammar(&grammar)
            .map_err(GrammarError::Compilation);
        self.note_compile();
        compiled
    }

    /// 2026-09-26: Most space, newline and tab characters allowed in a row between JSON
    /// tokens in a `GrammarSpec::JsonSchema` grammar (`max_whitespace_cnt`). With `None`
    /// the run is unbounded, so the grammar lets a model emit whitespace until max_tokens
    /// and never close the JSON. 8 allows a newline followed by up to 7 indent characters.
    const MAX_JSON_SCHEMA_WHITESPACE: i32 = 8;

    /// 2026-09-26: Compile a grammar that enforces a JSON schema.
    pub fn compile_json_schema(&mut self, schema: &str) -> Result<CompiledGrammar, GrammarError> {
        let compiled = self
            .compiler
            .compile_json_schema(
                schema,
                true,
                None,
                None::<(&str, &str)>,
                true,
                Some(Self::MAX_JSON_SCHEMA_WHITESPACE),
            )
            .map_err(GrammarError::Compilation);
        self.note_compile();
        compiled
    }

    /// 2026-09-26: Compile the built-in JSON grammar (any valid JSON).
    pub fn compile_json_grammar(&mut self) -> Result<CompiledGrammar, GrammarError> {
        self.compiler
            .compile_builtin_json_grammar()
            .map_err(GrammarError::Compilation)
    }

    /// 2026-09-26: Compile a grammar from an EBNF string.
    pub fn compile_ebnf(
        &mut self,
        ebnf: &str,
        root_rule: &str,
    ) -> Result<CompiledGrammar, GrammarError> {
        let compiled = self
            .compiler
            .compile_grammar_from_ebnf(ebnf, root_rule)
            .map_err(GrammarError::Compilation);
        self.note_compile();
        compiled
    }
}
