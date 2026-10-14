// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: CPU replay of the streaming loop guards over saved streams. It reads
//! fixture JSON from `GUARD_REPLAY_DIR` and feeds each fixture's chunks through the
//! guard sequence of `process_detector_content`. No guard logic is changed here.
//!
//! Replica, not the real `StreamState`: `StreamState::new` and its fields are
//! `pub(super)` to `api::chat_stream`, and `process_detector_content` needs a
//! `StreamCtx` that holds an `Arc<AppState>`. So `Replay` below copies the sequence of
//! `api/chat_stream/handle_token/detector_content.rs` and calls the same functions:
//! - `sanitize_content_chunk` with the poolside_v1 (GLM-5.3) markers
//!   (`detector_content.rs:100-106`, markers from `chat_stream/mod.rs:171-175`);
//! - SimHash: `simhash_pending.push_str`, `ends_at_sentence_boundary(..).is_some() ||
//!   len >= 1024` then `SimHashLoopGuard::check` and `clear`, then the >4096 halving
//!   (`detector_content.rs:34-50`);
//! - the line watchdog, `check_loop_watchdog(sanitized, &mut loop_scan_buf, triggered)`
//!   (`detector_content.rs:52-57`), run on every chunk even when SimHash fired.
//! The detector-less path is used (`handle_token.rs:493-498`): the replayed requests
//! carried no tools. `loop_watchdog_triggered` stops further SimHash pushes as in the
//! original (`detector_content.rs:34`).
//!
//! The server drops the chunk that trips a guard (`detector_content.rs:77`), so a saved client
//! text has a hole at the trip and ends within a few in-flight tokens of it. Two opt-in
//! probes stand in for the missing chunk (they are marked in the report, not exact
//! replays): `GUARD_REPLAY_TAIL_PROBE=1` runs `SimHashLoopGuard::check` on the unchecked
//! pending tail at the end, as the next boundary chunk would; `GUARD_REPLAY_HOLES=1`
//! inserts a fixture's `hole` chunk (text inferred from the rank0.log needle) before its
//! `before_chunk`.
//!
//! Env: `GUARD_REPLAY_DIR` (fixture dir; test skips with a message when unset);
//! `GUARD_REPLAY_EXPECT` = `report` (default, never fails) | `fire` (every fixture must
//! fire a guard) | `nofire` (none may fire); `GUARD_REPLAY_ONLY` = comma list of labels
//! to run.
//!
//! Owner: server streaming API.
//! Invariants: none beyond the types.

use crate::api::sanitizer::sanitize_content_chunk;
use crate::api::stream_guards::check_loop_watchdog;
use crate::loop_simhash::{SimHashLoopGuard, ends_at_sentence_boundary};
use crate::tool_parser::{PoolsideV1Parser, ToolCallParser};

/// 2026-10-14: The guard-relevant part of `StreamState` (`chat_stream/state.rs:66-79`).
struct Replay {
    tag_scan_buf: String,
    suppressing_param_leak: bool,
    inside_envelope: bool,
    loop_scan_buf: String,
    loop_watchdog_triggered: bool,
    simhash_guard: SimHashLoopGuard,
    simhash_pending: String,
}

/// 2026-10-14: Which guard fired, as named by `detector_content.rs:68-72`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Fired {
    SimHash,
    LineWatchdog,
}

impl Replay {
    fn new() -> Self {
        Self {
            tag_scan_buf: String::new(),
            suppressing_param_leak: false,
            inside_envelope: false,
            loop_scan_buf: String::new(),
            loop_watchdog_triggered: false,
            simhash_guard: SimHashLoopGuard::new(),
            simhash_pending: String::new(),
        }
    }

    /// 2026-10-14: One chunk through sanitize, then the `process_detector_content`
    /// guard sequence. Returns the guard that tripped, if any.
    fn feed(&mut self, text: &str) -> Option<Fired> {
        let markers = PoolsideV1Parser.leak_markers();
        let sanitized = sanitize_content_chunk(
            text,
            &mut self.tag_scan_buf,
            &mut self.suppressing_param_leak,
            &mut self.inside_envelope,
            &markers,
        );
        let semantic_trip = if !self.loop_watchdog_triggered {
            self.simhash_pending.push_str(&sanitized);
            let mut dup = false;
            if ends_at_sentence_boundary(&self.simhash_pending).is_some()
                || self.simhash_pending.len() >= 1024
            {
                dup = self.simhash_guard.check(&self.simhash_pending);
                self.simhash_pending.clear();
            }
            if self.simhash_pending.len() > 4096 {
                let drop_to = self.simhash_pending.len() / 2;
                self.simhash_pending.drain(..drop_to);
            }
            dup
        } else {
            false
        };
        let token_trip =
            check_loop_watchdog(&sanitized, &mut self.loop_scan_buf, self.loop_watchdog_triggered);
        if semantic_trip || token_trip {
            self.loop_watchdog_triggered = true;
            // 2026-10-14: The original names SimHash first when both trip.
            return Some(if semantic_trip { Fired::SimHash } else { Fired::LineWatchdog });
        }
        None
    }
}

struct Outcome {
    label: String,
    fired: Option<Fired>,
    probe: &'static str,
    chunk_idx: usize,
    n_chunks: usize,
    tok_idx: u64,
    n_tokens: u64,
    ring_len: usize,
    expected: String,
    completion_tokens: u64,
}

fn replay_fixture(v: &serde_json::Value) -> Outcome {
    let mut chunks: Vec<&str> = v["chunks"]
        .as_array()
        .expect("chunks")
        .iter()
        .map(|c| c.as_str().expect("chunk str"))
        .collect();
    let mut ctok: Vec<u64> = v["chunk_token_index"]
        .as_array()
        .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0)).collect())
        .unwrap_or_default();
    let mut probe = "";
    if std::env::var("GUARD_REPLAY_HOLES").as_deref() == Ok("1") {
        if let (Some(at), Some(text)) = (v["hole"]["before_chunk"].as_u64(), v["hole"]["text"].as_str()) {
            let at = (at as usize).min(chunks.len());
            let tok = ctok.get(at).copied().unwrap_or(0);
            chunks.insert(at, text);
            ctok.insert(at, tok);
            probe = "hole";
        }
    }
    let mut r = Replay::new();
    let mut fired = None;
    let mut chunk_idx = 0;
    for (i, c) in chunks.iter().enumerate() {
        if let Some(f) = r.feed(c) {
            fired = Some(f);
            chunk_idx = i;
            break;
        }
    }
    if fired.is_none()
        && std::env::var("GUARD_REPLAY_TAIL_PROBE").as_deref() == Ok("1")
        && !r.simhash_pending.is_empty()
        && r.simhash_guard.check(&r.simhash_pending)
    {
        fired = Some(Fired::SimHash);
        chunk_idx = chunks.len();
        probe = "tail";
    }
    Outcome {
        label: v["label"].as_str().unwrap_or("?").to_string(),
        fired,
        probe,
        chunk_idx,
        n_chunks: chunks.len(),
        tok_idx: ctok.get(chunk_idx).copied().unwrap_or(ctok.last().copied().unwrap_or(0)),
        n_tokens: v["retokenized_tokens"].as_u64().unwrap_or(chunks.len() as u64),
        ring_len: r.simhash_guard.len(),
        expected: format!(
            "{} {}",
            v["expected_guard_rank0"].as_str().unwrap_or("?"),
            v["rank0_warn"]
        ),
        completion_tokens: v["completion_tokens"].as_u64().unwrap_or(0),
    }
}

#[test]
fn replay_saved_streams_through_the_loop_guards() {
    let Ok(dir) = std::env::var("GUARD_REPLAY_DIR") else {
        eprintln!("guard_replay: GUARD_REPLAY_DIR is unset; skipping (no fixtures to replay)");
        return;
    };
    let mode = std::env::var("GUARD_REPLAY_EXPECT").unwrap_or_else(|_| "report".into());
    assert!(
        matches!(mode.as_str(), "report" | "fire" | "nofire"),
        "GUARD_REPLAY_EXPECT must be report, fire or nofire; got {mode}"
    );
    let only: Option<Vec<String>> = std::env::var("GUARD_REPLAY_ONLY")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read GUARD_REPLAY_DIR {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    let mut bad = Vec::new();
    println!("GUARD_REPLAY_EXPECT={mode} dir={dir}");
    println!("label\tfired\tchunk\ttoken(of retok/completion)\tring_len\texpected_rank0");
    for p in paths {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).expect("read fixture"))
                .expect("fixture json");
        let o = replay_fixture(&v);
        if let Some(only) = &only {
            if !only.contains(&o.label) {
                continue;
            }
        }
        let f = match o.fired {
            Some(Fired::SimHash) => "A",
            Some(Fired::LineWatchdog) => "B",
            None => "none",
        };
        println!(
            "{}\t{}{}\t{}/{}\t{}/{}/{}\t{}\t{}",
            o.label, f, if o.probe.is_empty() { String::new() } else { format!("({})", o.probe) }, o.chunk_idx, o.n_chunks, o.tok_idx, o.n_tokens, o.completion_tokens,
            o.ring_len, o.expected
        );
        match mode.as_str() {
            "fire" if o.fired.is_none() => bad.push(format!("{} did not fire", o.label)),
            "nofire" if o.fired.is_some() => bad.push(format!("{} fired {f}", o.label)),
            _ => {}
        }
    }
    assert!(bad.is_empty(), "guard replay expectation ({mode}) failed: {bad:?}");
}
