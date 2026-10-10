// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Cold grammar preparation for a new tool schema: reuse of
//! rule-level masks across schemas, and the on-disk mask snapshot.
//!
//! Two ignored diagnostics print phase timings for a real tokenizer named
//! by `QWEN_TOKENIZER_JSON`. The other tests assert cache counters, bitmask
//! equality, a timing ratio and a file count, never an absolute time.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use std::time::{Duration, Instant};

use super::*;
use crate::grammar::GrammarState;

/// 2026-09-26: A deterministic `n`-token vocabulary (printable ASCII, then
/// three-character strings), followed by `<tool_call>`, `</tool_call>` and
/// `<eos>`.
fn wide_vocab(n: usize) -> Vec<String> {
    let alphabet: Vec<char> = (b' '..=b'~').map(|c| c as char).collect();
    let mut vocab: Vec<String> = alphabet.iter().map(|c| c.to_string()).collect();
    let mut i = 0usize;
    while vocab.len() < n {
        let a = alphabet[i % alphabet.len()];
        let b = alphabet[(i / alphabet.len()) % alphabet.len()];
        let c = alphabet[(i / (alphabet.len() * alphabet.len())) % alphabet.len()];
        vocab.push(format!("{a}{b}{c}"));
        i += 1;
    }
    vocab.truncate(n);
    vocab.extend(["<tool_call>", "</tool_call>", "<eos>"].map(String::from));
    vocab
}

const VOCAB: usize = 1024;

fn engine() -> GrammarEngine {
    let vocab = wide_vocab(VOCAB);
    let eos = (vocab.len() - 1) as i32;
    GrammarEngine::new(&vocab, &[eos]).expect("engine builds")
}

/// 2026-09-26: A one-tool list: `name` with a required string `a` and a
/// required integer `b`.
fn tool(name: &str, a: &str, b: &str) -> Vec<ToolDefinition> {
    serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up the current weather for a city.",
            "parameters": {"type": "object", "properties": {
                a: {"type": "string", "description": "City name."},
                b: {"type": "integer", "description": "Forecast horizon in days."}},
                "required": [a, b]},
        }
    }]))
    .unwrap()
}

/// 2026-09-26: Timings of one request's grammar preparation, and the state
/// it produced.
struct Prepared {
    /// 2026-09-26: `compile_qwen3_coder_tool_grammar`. Not part of the
    /// asserted ratio.
    construct: Duration,
    /// 2026-09-26: State construction, the prewarm and the first
    /// `fill_bitmask`, which joins the prewarm.
    masks: Duration,
    state: GrammarState,
}

fn prepare(engine: &mut GrammarEngine, tools: &[ToolDefinition]) -> Prepared {
    let started = Instant::now();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(tools, true, "</parameter>")
        .expect("tool grammar compiles");
    let construct = started.elapsed();

    let started = Instant::now();
    let hook = engine.mask_snapshot_hook();
    let mut state =
        GrammarState::new_with_hook(&compiled, engine.vocab_size(), hook).expect("grammar state");
    state.fill_bitmask();
    Prepared {
        construct,
        masks: started.elapsed(),
        state,
    }
}

/// 2026-09-26: The size of the first non-empty `masks-*.bin` under
/// `dir/.metrale-grammar-cache`, polled for up to 30 s because the save
/// runs on its own thread; `None` on timeout.
fn await_snapshot(dir: &std::path::Path) -> Option<u64> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(entries) = std::fs::read_dir(dir.join(".metrale-grammar-cache")) {
            for entry in entries.filter_map(Result::ok) {
                // 2026-09-26: Match the `.bin` suffix too: the writer's temp
                // file, `masks-<fp>.tmp<pid>`, also starts with "masks-" and
                // exists before the rename makes the `.bin` file.
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("masks-") && name.ends_with(".bin") {
                    let len = entry.metadata().ok()?.len();
                    if len > 0 {
                        return Some(len);
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    None
}

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "metrale-918-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

#[test]
fn a_second_distinct_schema_reuses_the_first_schemas_masks() {
    // 2026-09-26: The rule-level cache keys masks by rule structure, so a
    // second schema with different names still hits masks the first one
    // computed. Asserted on the cache's hit/miss counters, not on time.
    let mut engine = engine();
    let cache = engine
        .compiler
        .rule_cache_handle()
        .expect("serve compilers enable the rule cache");

    prepare(&mut engine, &tool("get_weather", "city", "days"));
    let (hits_after_first, misses_after_first) = cache.hit_miss();
    assert!(
        misses_after_first > 0,
        "the first schema should have computed masks, not found them"
    );

    prepare(&mut engine, &tool("search_docs", "query", "limit"));
    let (hits, misses) = cache.hit_miss();
    let (new_hits, new_misses) = (hits - hits_after_first, misses - misses_after_first);
    assert!(
        new_hits > 0,
        "cross-schema mask reuse lost: the second schema hit {new_hits} / missed {new_misses}"
    );
}

#[test]
fn a_persisted_snapshot_removes_the_cold_prewarm_for_the_next_process() {
    let dir = scratch("hit");
    let tools = tool("get_weather", "city", "days");

    // 2026-09-26: First engine: nothing on disk; its prewarm hook saves
    // the snapshot.
    let mut first = engine();
    first.attach_mask_cache(&dir);
    let cold = prepare(&mut first, &tools);
    let snapshot = await_snapshot(&dir).expect("snapshot written by the background prewarm");
    assert!(snapshot > 0, "snapshot file is empty");

    // 2026-09-26: Second engine on the same directory loads it.
    let mut second = engine();
    second.attach_mask_cache(&dir);
    let warm = prepare(&mut second, &tools);

    assert_eq!(
        cold.state.bitmask_data(),
        warm.state.bitmask_data(),
        "snapshot-warmed grammar admits a different first token set"
    );
    assert!(
        cold.masks > warm.masks * 3,
        "the snapshot did not remove the cold prewarm: cold={:?} warm={:?} \
         (construct {:?} / {:?})",
        cold.masks,
        warm.masks,
        cold.construct,
        warm.construct,
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_snapshot_from_a_different_tokenizer_is_a_miss() {
    let dir = scratch("miss");
    let tools = tool("get_weather", "city", "days");

    let mut writer = engine();
    writer.attach_mask_cache(&dir);
    prepare(&mut writer, &tools);
    await_snapshot(&dir).expect("the writing engine persisted its masks");

    // 2026-09-26: Same vocabulary size, different strings, so a different
    // fingerprint and file name: the directory still holds one file.
    let other_vocab: Vec<String> = wide_vocab(VOCAB).iter().map(|t| format!("z{t}")).collect();
    let eos = (other_vocab.len() - 1) as i32;
    let mut reader = GrammarEngine::new(&other_vocab, &[eos]).expect("engine builds");
    reader.attach_mask_cache(&dir);
    let names: Vec<String> = std::fs::read_dir(dir.join(".metrale-grammar-cache"))
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names.len(),
        1,
        "a second tokenizer must not reuse the first's file: {names:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2026-09-26: Prints per-schema construction, prewarm and first-fill
/// times for the tokenizer at `QWEN_TOKENIZER_JSON`. Ignored: it needs that
/// file and asserts nothing.
#[test]
#[ignore = "CPU diagnostic: set QWEN_TOKENIZER_JSON to an existing tokenizer.json"]
fn phase_timings_with_a_real_tokenizer() {
    let path = std::env::var("QWEN_TOKENIZER_JSON").expect("QWEN_TOKENIZER_JSON");
    let tokenizer = tokenizers::Tokenizer::from_file(path).expect("tokenizer loads");
    let stop = tokenizer.token_to_id("<|im_end|>").expect("<|im_end|>") as i32;
    let started = Instant::now();
    let mut engine =
        GrammarEngine::from_tokenizer(&tokenizer, None, &[stop]).expect("engine builds");
    println!(
        "vocab={} engine_build_ms={:.1}",
        engine.vocab_size(),
        started.elapsed().as_secs_f64() * 1000.0
    );
    let cases = [
        (
            "get_weather(city, days)",
            tool("get_weather", "city", "days"),
        ),
        (
            "get_weather(city, days) again",
            tool("get_weather", "city", "days"),
        ),
        (
            "search_docs(query, limit) NEW",
            tool("search_docs", "query", "limit"),
        ),
        (
            "run_cmd(command, timeout) NEW",
            tool("run_cmd", "command", "timeout"),
        ),
    ];
    for (label, tools) in cases {
        let t = Instant::now();
        let compiled = engine
            .compile_qwen3_coder_tool_grammar(&tools, true, "</parameter>")
            .expect("tool grammar compiles");
        let construct = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let masks = compiled.compile_top_k_masks(512);
        let prewarm = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let mut state = GrammarState::new(&compiled, engine.vocab_size()).expect("grammar state");
        state.fill_bitmask();
        let first_fill = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "[{label}] construct_ms={construct:.1} masks={masks} prewarm_ms={prewarm:.1} \
             matcher_and_first_fill_ms={first_fill:.1} mask_bytes={}",
            compiled.memory_size_bytes(),
        );
    }
}

/// 2026-09-26: Prints the time from compiling a tool grammar to the first
/// filled mask, cold and with a snapshot on disk, for the tokenizer at
/// `QWEN_TOKENIZER_JSON`. Ignored for the same reason as
/// [`phase_timings_with_a_real_tokenizer`].
#[test]
#[ignore = "CPU diagnostic: set QWEN_TOKENIZER_JSON to an existing tokenizer.json"]
fn cold_vs_snapshot_warm_with_a_real_tokenizer() {
    let path = std::env::var("QWEN_TOKENIZER_JSON").expect("QWEN_TOKENIZER_JSON");
    let tokenizer = tokenizers::Tokenizer::from_file(path).expect("tokenizer loads");
    let stop = tokenizer.token_to_id("<|im_end|>").expect("<|im_end|>") as i32;
    let dir = scratch("real");
    let tools = tool("get_weather", "city", "days");
    let build = || GrammarEngine::from_tokenizer(&tokenizer, None, &[stop]).expect("engine");

    let mut cold_engine = build();
    cold_engine.attach_mask_cache(&dir);
    let cold = prepare(&mut cold_engine, &tools);
    await_snapshot(&dir).expect("snapshot persisted");

    for rep in 0..3 {
        let mut warm_engine = build();
        warm_engine.attach_mask_cache(&dir);
        let warm = prepare(&mut warm_engine, &tools);
        let mut unseen_engine = build();
        unseen_engine.attach_mask_cache(&dir);
        let unseen = prepare(&mut unseen_engine, &tool("run_cmd", "command", "timeout"));
        await_snapshot(&dir);
        // 2026-09-26: `admit_ms` is compile plus state construction, before
        // the first fill joins the prewarm.
        let mut admit_engine = build();
        let admitted = Instant::now();
        let compiled = admit_engine
            .compile_qwen3_coder_tool_grammar(&tool("ping", "host", "count"), true, "</parameter>")
            .unwrap();
        let mut state =
            GrammarState::new_with_hook(&compiled, admit_engine.vocab_size(), None).unwrap();
        let admit_ms = admitted.elapsed().as_secs_f64() * 1000.0;
        state.fill_bitmask();
        let joined_ms = admitted.elapsed().as_secs_f64() * 1000.0;
        println!(
            "rep={rep} cold_ms={:.1} warm_same_schema_ms={:.1} warm_unseen_schema_ms={:.1} \
             cold_admit_ms={admit_ms:.1} cold_admit_to_first_fill_ms={joined_ms:.1}",
            (cold.construct + cold.masks).as_secs_f64() * 1000.0,
            (warm.construct + warm.masks).as_secs_f64() * 1000.0,
            (unseen.construct + unseen.masks).as_secs_f64() * 1000.0,
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2026-10-10: A177: a failing save is attempted once, not per prewarm.
#[test]
fn a_failed_snapshot_save_is_latched_not_retried() {
    let dir = scratch("failsave");
    // A file where the cache directory should be makes every save fail.
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(".metrale-grammar-cache"), b"not a dir").unwrap();
    let tools = tool("get_weather", "city", "days");
    let mut e = engine();
    e.attach_mask_cache(&dir);
    prepare(&mut e, &tools);
    let deadline = Instant::now() + Duration::from_secs(10);
    while e.snapshot_save_state() != Some((1, true)) {
        assert!(
            Instant::now() < deadline,
            "save never failed: {:?}",
            e.snapshot_save_state()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let hook = e.mask_snapshot_hook().expect("armed");
    for _ in 0..5 {
        hook(1);
    }
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(e.snapshot_save_state(), Some((1, true)));
    let _ = std::fs::remove_dir_all(&dir);
}

/// 2026-10-10: `METRALE_GRAMMAR_CACHE_MB`: unset or blank keeps 1 GiB, a value
/// sets MiB, and a typo or a value below 16 is an error, never the default.
#[test]
fn grammar_cache_mb_parses_strictly() {
    use crate::grammar::engine::cache_budget_bytes;
    assert_eq!(cache_budget_bytes(None), Ok(1024 * 1024 * 1024));
    assert_eq!(cache_budget_bytes(Some("  ")), Ok(1024 * 1024 * 1024));
    assert_eq!(cache_budget_bytes(Some("256")), Ok(256 * 1024 * 1024));
    assert!(cache_budget_bytes(Some("256MB")).is_err());
    assert!(cache_budget_bytes(Some("8")).is_err());
}
