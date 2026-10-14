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
//! fire a guard) | `nofire` (none may fire) | `bound` (2026-10-14: each fixture's own
//! `expected`: `fire` must fire, by `live_guard_token + 64` when the fixture has
//! `live_guard_token` and by `fire_by_token` when it has that; `nofire` or no `expected`
//! must not fire); `GUARD_REPLAY_ONLY` = comma list of labels to run.
//!
//! 2026-10-14: `synthetic` below holds hard-loop positives and legit-structure negatives
//! as unit tests through the same `Replay`; `GUARD_SYNTH_WRITE_DIR=<dir>` makes
//! `write_synthetic_fixtures` write them as fixtures for `GUARD_REPLAY_EXPECT=bound`.
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
    /// 2026-10-14: The fixture's own `expected` (`fire`/`nofire`), if any.
    want: Option<String>,
    /// 2026-10-14: Latest token a `fire` fixture may fire at, if bounded.
    fire_by: Option<u64>,
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
    if std::env::var("GUARD_REPLAY_HOLES").as_deref() == Ok("1")
        && let (Some(at), Some(text)) = (
            v["hole"]["before_chunk"].as_u64(),
            v["hole"]["text"].as_str(),
        )
    {
        let at = (at as usize).min(chunks.len());
        let tok = ctok.get(at).copied().unwrap_or(0);
        chunks.insert(at, text);
        ctok.insert(at, tok);
        probe = "hole";
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
        want: v["expected"].as_str().map(str::to_string),
        fire_by: v["live_guard_token"]
            .as_u64()
            .map(|t| t + 64)
            .or_else(|| v["fire_by_token"].as_u64()),
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
        matches!(mode.as_str(), "report" | "fire" | "nofire" | "bound"),
        "GUARD_REPLAY_EXPECT must be report, fire, nofire or bound; got {mode}"
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
    println!(
        "label\tfired\tchunk\ttoken(of retok/completion)\tring_len\texpected_rank0\twant/fire_by"
    );
    for p in paths {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).expect("read fixture"))
                .expect("fixture json");
        // 2026-10-14: filter before replaying, so a skipped fixture is never run.
        if let Some(only) = &only
            && !only.iter().any(|l| v["label"].as_str() == Some(l.as_str()))
        {
            continue;
        }
        let o = replay_fixture(&v);
        let f = match o.fired {
            Some(Fired::SimHash) => "A",
            Some(Fired::LineWatchdog) => "B",
            None => "none",
        };
        println!(
            "{}\t{}{}\t{}/{}\t{}/{}/{}\t{}\t{}\t{}/{}",
            o.label,
            f,
            if o.probe.is_empty() {
                String::new()
            } else {
                format!("({})", o.probe)
            },
            o.chunk_idx,
            o.n_chunks,
            o.tok_idx,
            o.n_tokens,
            o.completion_tokens,
            o.ring_len,
            o.expected,
            o.want.as_deref().unwrap_or("-"),
            o.fire_by.map_or("-".to_string(), |t| t.to_string())
        );
        match mode.as_str() {
            "fire" if o.fired.is_none() => bad.push(format!("{} did not fire", o.label)),
            "nofire" if o.fired.is_some() => bad.push(format!("{} fired {f}", o.label)),
            "bound" => match (o.want.as_deref(), o.fired) {
                (Some("fire"), None) => bad.push(format!("{} did not fire", o.label)),
                (Some("fire"), Some(_)) if o.fire_by.is_some_and(|b| o.tok_idx > b) => {
                    bad.push(format!(
                        "{} fired {f} at token {} > bound {}",
                        o.label,
                        o.tok_idx,
                        o.fire_by.unwrap_or(0)
                    ))
                }
                (Some("fire"), Some(_)) => {}
                (_, Some(_)) => bad.push(format!("{} fired {f}", o.label)),
                (_, None) => {}
            },
            _ => {}
        }
    }
    assert!(bad.is_empty(), "guard replay expectation ({mode}) failed: {bad:?}");
}

/// 2026-10-14: Synthetic hard loops (must fire) and legit structure (must not), fed
/// through `Replay` in pseudo-tokens, plus the fixture writer for the replay harness.
mod synthetic {
    use super::{Fired, Replay};

    /// 2026-10-14: A synthetic stream: `prefix` then `unit` `copies` times, then `suffix`.
    pub(super) struct Case {
        pub label: &'static str,
        pub text: String,
        pub fire: bool,
        /// Token index at the end of the 2nd full copy of `unit` (fire cases).
        pub second_end_tok: usize,
        /// Tokens in one copy of `unit`.
        pub unit_toks: usize,
    }

    /// 2026-10-14: BPE-like pseudo-tokens: a newline alone; otherwise optional leading
    /// spaces plus up to 4 alphanumeric chars, or plus one other char.
    pub(super) fn pseudo_tokens(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut alnum = 0usize;
        for ch in text.chars() {
            let starts_new = if ch == '\n' {
                true
            } else if ch == ' ' || ch == '\t' {
                !cur.trim().is_empty()
            } else if ch.is_alphanumeric() {
                alnum >= 4 || (!cur.trim().is_empty() && alnum == 0)
            } else {
                !cur.trim().is_empty()
            };
            if starts_new && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                alnum = 0;
            }
            cur.push(ch);
            if ch == '\n' {
                out.push(std::mem::take(&mut cur));
                alnum = 0;
            } else if ch.is_alphanumeric() {
                alnum += 1;
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    fn case(
        label: &'static str,
        prefix: &str,
        unit: &str,
        copies: usize,
        suffix: &str,
        fire: bool,
    ) -> Case {
        let mut text = String::from(prefix);
        for _ in 0..copies {
            text.push_str(unit);
        }
        text.push_str(suffix);
        let second_end_tok = pseudo_tokens(&format!("{prefix}{unit}{unit}")).len();
        Case {
            label,
            text,
            fire,
            second_end_tok,
            unit_toks: pseudo_tokens(unit).len(),
        }
    }

    /// 2026-10-14: Tokens after the end of the 2nd full repeat within which a hard loop
    /// must fire: two more copies of the repeating unit.
    pub(super) fn fire_by(c: &Case) -> usize {
        c.second_end_tok + 2 * c.unit_toks
    }

    pub(super) fn cases() -> Vec<Case> {
        let mut v = vec![case(
            "syn_paragraph_x6",
            "Here is the summary you asked for.\n\n",
            "The service reads the queue, batches the records, and writes them to the \
             warehouse every five minutes. If a batch fails, it is retried twice before \
             the job alerts the on-call engineer. Logs for every run are kept for thirty days.\n\n",
            6,
            "",
            true,
        )];
        v.push(case(
            "syn_sentence_x10",
            "Sure, let me look into that.\n",
            "I will now check the configuration file for the missing database setting.\n",
            10,
            "",
            true,
        ));
        v.push(case(
            "syn_block3_x5",
            "Polling the deployment API until the rollout finishes:\n\n",
            "status pending for deployment web-frontend\nretrying request against the cluster api\nwaiting for a response from the controller\n",
            5,
            "",
            true,
        ));
        v.push(case(
            "syn_phrase_noline",
            "The final answer is ",
            "and the result stays exactly the same as before ",
            12,
            "",
            true,
        ));
        v.push(case(
            "syn_code_tail4_x6",
            "```python\nimport time\n\ndef wait_until_done(client):\n    while True:\n",
            "        status = client.get_status()\n        if status == 'done':\n            break\n        time.sleep(1)\n",
            6,
            "",
            true,
        ));
        let mut table = String::from(
            "| # | Item | Unit price | Stock | Location |\n|---|------|-----------:|-------|----------|\n",
        );
        for i in 1..=20usize {
            table.push_str(&format!(
                "| {i} | Widget {} | ${}.{:02} | {} | Warehouse {} |\n",
                (b'A' + (i % 26) as u8) as char,
                5 + i * 3 % 17,
                (i * 37) % 100,
                if i.is_multiple_of(4) {
                    "Out of stock"
                } else {
                    "In stock"
                },
                if i.is_multiple_of(2) {
                    "North"
                } else {
                    "South"
                }
            ));
        }
        v.push(case(
            "syn_md_table20",
            "Here is the inventory:\n\n",
            &table,
            1,
            "\nLet me know if you need totals.",
            false,
        ));
        let diagram = "```\n\
┌──────────────┐      ┌──────────────┐      ┌──────────────┐\n\
│    Client    │─────▶│   Gateway    │─────▶│   Service    │\n\
└──────────────┘      └──────────────┘      └──────────────┘\n\
        │                     │                     │\n\
        ▼                     ▼                     ▼\n\
┌──────────────┐      ┌──────────────┐      ┌──────────────┐\n\
│    Cache     │      │   Metrics    │      │   Database   │\n\
└──────────────┘      └──────────────┘      └──────────────┘\n\
        │                     │                     │\n\
        ▼                     ▼                     ▼\n\
┌──────────────┐      ┌──────────────┐      ┌──────────────┐\n\
│   Replica    │      │  Dashboard   │      │   Backups    │\n\
└──────────────┘      └──────────────┘      └──────────────┘\n\
```\n";
        v.push(case(
            "syn_ascii_box",
            "The request path looks like this:\n\n",
            diagram,
            1,
            "\nEach arrow is one network hop.",
            false,
        ));
        let bullets = [
            (
                "Timeouts",
                "set a timeout on every outbound HTTP call so a slow dependency cannot hold a worker forever.",
            ),
            (
                "Retries",
                "retry idempotent requests with exponential backoff and a cap on the total attempts.",
            ),
            (
                "Circuit breakers",
                "stop calling a dependency that keeps failing and probe it again after a cool-down.",
            ),
            (
                "Bulkheads",
                "give each dependency its own connection pool so one outage cannot drain the others.",
            ),
            (
                "Rate limits",
                "limit how many requests each client may send per second to protect shared capacity.",
            ),
            (
                "Health checks",
                "expose a cheap endpoint the load balancer can poll to take sick instances out.",
            ),
            (
                "Graceful shutdown",
                "stop accepting new work on SIGTERM and finish the requests already in flight.",
            ),
            (
                "Idempotency keys",
                "let clients send a key so a retried payment is not charged twice.",
            ),
            (
                "Dead-letter queues",
                "move messages that keep failing aside so they do not block the rest of the queue.",
            ),
            (
                "Structured logs",
                "log one JSON object per event with a request id so traces can be joined later.",
            ),
            (
                "Metrics",
                "record latency percentiles and error rates for every endpoint and dependency.",
            ),
            (
                "Alerts",
                "page on symptoms users feel, such as error rate and latency, not on every CPU spike.",
            ),
        ];
        let mut list = String::new();
        for (k, d) in bullets {
            list.push_str(&format!("- **{k}**: {d}\n"));
        }
        v.push(case(
            "syn_bullets12",
            "Here are the resilience patterns to apply:\n\n",
            &list,
            1,
            "\nStart with timeouts and retries.",
            false,
        ));
        let mut points = String::from("```json\n[");
        for i in 0..10 {
            points.push_str(&format!("{{\"type\":\"point\",\"x\":{i},\"y\":0}}, "));
        }
        points.push_str("]\n```\n");
        v.push(case(
            "syn_json_line10",
            "Here is the payload:\n\n",
            &points,
            1,
            "\nEach point has y = 0.",
            false,
        ));
        let mut svc_list = String::new();
        for svc in [
            "nginx",
            "postgres",
            "redis",
            "kafka",
            "grafana",
            "prometheus",
            "jenkins",
            "rabbitmq",
            "elasticsearch",
            "memcached",
            "haproxy",
            "vault",
        ] {
            svc_list.push_str(&format!(
                "- Restart the {svc} service and verify that it comes back up cleanly.\n"
            ));
        }
        v.push(case(
            "syn_templated_list12",
            "Restart each service in order:\n\n",
            &svc_list,
            1,
            "\nThen run the smoke tests.",
            false,
        ));
        let mut code = String::from("```python\ndef weight_for(x):\n    total = 0\n");
        for (c, w, name) in [
            ('A', 10, "alpha"),
            ('B', 20, "beta"),
            ('C', 30, "gamma"),
            ('D', 40, "delta"),
            ('E', 50, "epsilon"),
            ('F', 60, "zeta"),
            ('G', 70, "eta"),
            ('H', 80, "theta"),
        ] {
            code.push_str(&format!(
                "    if x == '{c}':\n        print(f'Processing category {c} ({name})')\n        total += {w}\n"
            ));
        }
        code.push_str("    return total\n```\n");
        v.push(case(
            "syn_code_if8",
            "Here is the function:\n\n",
            &code,
            1,
            "\nEach branch adds its own weight.",
            false,
        ));
        v
    }

    /// 2026-10-14: (guard, token index) of the first trip, if any.
    pub(super) fn run(text: &str) -> Option<(Fired, usize)> {
        let mut r = Replay::new();
        pseudo_tokens(text)
            .iter()
            .enumerate()
            .find_map(|(i, t)| r.feed(t).map(|f| (f, i)))
    }

    #[test]
    fn hard_loops_fire_within_two_units_after_the_second_repeat() {
        for c in cases().into_iter().filter(|c| c.fire) {
            let got = run(&c.text);
            let (f, at) = got.unwrap_or_else(|| panic!("{} did not fire", c.label));
            println!(
                "{}\t{f:?}\t{at}\t2nd repeat ends {}\tunit {}\tbound {}",
                c.label,
                c.second_end_tok,
                c.unit_toks,
                fire_by(&c)
            );
            assert!(
                at <= fire_by(&c),
                "{} fired at {at} > bound {}",
                c.label,
                fire_by(&c)
            );
        }
    }

    #[test]
    fn legit_structure_does_not_fire() {
        for c in cases().into_iter().filter(|c| !c.fire) {
            assert_eq!(run(&c.text), None, "{} fired", c.label);
        }
    }

    /// 2026-10-14: `GUARD_SYNTH_WRITE_DIR=<dir>` writes every case as a replay fixture
    /// (`expected`, and `fire_by_token` for fire cases); skipped when unset.
    #[test]
    fn write_synthetic_fixtures() {
        let Ok(dir) = std::env::var("GUARD_SYNTH_WRITE_DIR") else {
            return;
        };
        std::fs::create_dir_all(&dir).expect("create GUARD_SYNTH_WRITE_DIR");
        for c in cases() {
            let toks = pseudo_tokens(&c.text);
            let mut v = serde_json::json!({
                "label": c.label,
                "source": "guard_replay.rs synthetic::cases (metrale #82 guard fix, 2026-10-14)",
                "expected": if c.fire { "fire" } else { "nofire" },
                "chunks": toks,
                "chunk_token_index": (0..toks.len()).collect::<Vec<_>>(),
                "retokenized_tokens": toks.len(),
                "completion_tokens": toks.len(),
            });
            if c.fire {
                v["second_repeat_end_token"] = c.second_end_tok.into();
                v["unit_tokens"] = c.unit_toks.into();
                v["fire_by_token"] = fire_by(&c).into();
            }
            std::fs::write(
                format!("{dir}/{}.json", c.label),
                serde_json::to_string_pretty(&v).expect("json"),
            )
            .expect("write fixture");
        }
    }
}
