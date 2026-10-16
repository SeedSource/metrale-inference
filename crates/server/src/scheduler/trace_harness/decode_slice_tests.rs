// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_DECODE_STEPS_PER_PREFILL_CHUNK` (`decode_slice`) through the real
//! scheduler loop: one sequence decodes while a second, long prompt prefills in chunks, with the
//! fused mixed step off (`no_mix`, the multi-rank shape) and the chunk-while-decoding cap on (the
//! race car's setting). Off: one decode step per prefill chunk. `k`: `k` decode steps per chunk.
//! Either way every client receives the same bytes, and a lone request runs the same model calls.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};

fn reply(n: usize, base: u32) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n as u32).map(|i| base + i).collect();
    v.push(EOS);
    v
}

fn scenario(k: Option<usize>, with_decoder: bool) -> Scenario {
    let mut reqs = Vec::new();
    if with_decoder {
        // 2026-10-14: the decoder: a short prompt and a long reply, so it decodes throughout.
        reqs.push(ReqSpec::new(1, 4, reply(60, 100)));
    }
    // 2026-10-14: the long prompt: 64 tokens in 8-token chunks, arriving once seq 1 decodes.
    let mut long = ReqSpec::new(2, 64, reply(3, 500));
    long.arrive_at_tick = with_decoder.then_some(3);
    reqs.push(long);
    Scenario {
        name: "decode_slice",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            max_prefill_tokens: 8,
            no_mix: true,
            prefill_chunk_while_decoding: Some(256),
            decode_steps_per_prefill_chunk: k,
            ..RunOptions::default()
        },
        reqs,
    }
}

/// Per tick: (ran a prefill chunk of seq 2, ran a decode step).
fn ticks(lines: &[String]) -> Vec<(bool, bool)> {
    let mut out: Vec<(bool, bool)> = Vec::new();
    for l in lines {
        if l.starts_with("-- tick ") {
            out.push((false, false));
        } else if let Some(t) = out.last_mut() {
            if l.starts_with("prefill_chunk(") && l.contains(" s2@") {
                t.0 = true;
            }
            if l.starts_with("decode_batch(") || l.starts_with("decode(") {
                t.1 = true;
            }
        }
    }
    out
}

/// The decode-only ticks between consecutive seq-2 chunks, while seq 1 is still decoding.
fn gaps(t: &[(bool, bool)]) -> Vec<usize> {
    let chunk_ticks: Vec<usize> = t.iter().enumerate().filter(|(_, x)| x.0).map(|(i, _)| i).collect();
    chunk_ticks
        .windows(2)
        .map(|w| (w[0] + 1..w[1]).filter(|&i| t[i].1 && !t[i].0).count())
        .collect()
}

fn outputs(lines: &[String]) -> Vec<String> {
    lines.iter().filter(|l| l.starts_with("out s")).cloned().collect()
}

#[test]
fn off_runs_one_decode_step_per_prefill_chunk() {
    let lines = run_scenario(&scenario(None, true));
    let t = ticks(&lines);
    let g = gaps(&t);
    assert!(g.len() >= 6, "seq 2 ran {} chunk ticks: {lines:#?}", g.len() + 1);
    assert!(g.iter().all(|&n| n == 0), "off: chunks back to back, got gaps {g:?}");
    assert!(t.iter().filter(|x| x.0).all(|x| x.1), "off: every chunk tick also decodes");
}

#[test]
fn k_runs_k_decode_steps_per_prefill_chunk() {
    for k in [2usize, 3, 5] {
        let lines = run_scenario(&scenario(Some(k), true));
        let t = ticks(&lines);
        let g = gaps(&t);
        assert!(g.len() >= 6, "k={k}: seq 2 ran {} chunk ticks", g.len() + 1);
        assert!(g.iter().all(|&n| n == k - 1), "k={k}: want {} decode-only ticks per gap, got {g:?}", k - 1);
        assert!(t.iter().filter(|x| x.0).all(|x| x.1), "k={k}: every chunk tick also decodes");
    }
}

#[test]
fn client_bytes_do_not_depend_on_the_lever() {
    let off = outputs(&run_scenario(&scenario(None, true)));
    assert_eq!(off.len(), 2, "{off:?}");
    for k in [2usize, 3, 5] {
        assert_eq!(outputs(&run_scenario(&scenario(Some(k), true))), off, "k={k}");
    }
}

#[test]
fn a_lone_request_runs_the_same_model_calls() {
    // 2026-10-14: no decoder while seq 2 prefills, and no prefill while it decodes: the lever
    // never holds a tick, so the whole trace is identical (the C=1 byte-identity case).
    let off = run_scenario(&scenario(None, false));
    for k in [2usize, 5] {
        assert_eq!(run_scenario(&scenario(Some(k), false)), off, "k={k}");
    }
}

/// 2026-10-16 (race-prefill, A197): seq 1 decodes, seq 2's 64-token prompt prefills in 8-token
/// chunks, and 40 short requests (one chunk, a 1-token reply) wait behind them: more than the
/// one start per prefill turn can serve before seq 2 is done. All are queued before the loop
/// starts (the gate parks only at the first `arrive_at_tick`, so timed arrivals after it race the
/// loop): tick 1 has no decoder and starts the first `max_batch_size`, the rest stay pending.
fn stream_scenario(k: Option<usize>) -> Scenario {
    let mut sc = scenario(k, true);
    sc.reqs[0] = ReqSpec::new(1, 4, reply(200, 100));
    sc.reqs[1].arrive_at_tick = None;
    for i in 0..40u64 {
        sc.reqs.push(ReqSpec::new(10 + i, 4, reply(1, 40)));
    }
    sc
}

#[test]
fn a_steady_stream_of_short_requests_does_not_starve_a_long_prefill() {
    // 2026-10-16 (race-prefill, A197): with `k` the held ticks reset the start/continue
    // alternation, so every free tick started a short request and seq 2 got no chunk until the
    // stream ended (gape E on GPU: 648 s). Starts and continues must alternate across the held
    // ticks: a seq-2 chunk at least every second prefill turn, i.e. within 2k ticks. `None`
    // (no held ticks, the c29 car's setting) alternates every tick.
    for k in [None, Some(2usize), Some(3)] {
        let lines = run_scenario(&stream_scenario(k));
        let t = ticks(&lines);
        let chunk_ticks: Vec<usize> = t.iter().enumerate().filter(|(_, x)| x.0).map(|(i, _)| i).collect();
        assert_eq!(chunk_ticks.len(), 8, "k={k:?}: seq 2 ran {} chunks", chunk_ticks.len());
        let worst = chunk_ticks.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        let bound = 2 * k.unwrap_or(1);
        assert!(worst <= bound, "k={k:?}: seq 2 waited {worst} ticks between chunks (bound {bound}): {chunk_ticks:?}");
        let outs = outputs(&lines);
        assert_eq!(outs.len(), 42, "k={k:?}: {outs:?}");
        assert!(outs.iter().all(|o| o.contains("Done(stop")), "k={k:?}: {outs:?}");
    }
}

/// 2026-10-16 (race-memory, A194 gapF): seq 1 decodes; seq 2 and seq 3 (64-token prompts, 8-token
/// chunks) and seq 4 (a 4-token prompt) arrive together, queued at one park. seq 4's first
/// chunk-0 attempt is refused for KV (`chunk0_kv_refuse`), so it waits as a prefill at offset 0.
fn refused_short_scenario(k: Option<usize>) -> Scenario {
    let mut sc = scenario(k, true);
    sc.reqs[0] = ReqSpec::new(1, 4, reply(200, 100));
    sc.reqs[1].arrive_at_tick = Some(3);
    let mut s3 = ReqSpec::new(3, 64, reply(3, 600));
    s3.arrive_at_tick = Some(3);
    sc.reqs.push(s3);
    let mut s4 = ReqSpec::new(4, 4, reply(2, 40));
    s4.arrive_at_tick = Some(3);
    sc.reqs.push(s4);
    sc.cfg.chunk0_kv_refuse = vec![(4, 1)];
    sc.opts.single_stream_prefill = true;
    sc
}

#[test]
fn a_refused_short_request_is_retried_at_the_next_start_turn_not_behind_the_long_prefills() {
    // 2026-10-16 (race-memory, A194 gapF on GPU): the refused request was pushed behind both long
    // prefills, and only the head advances on the single-stream path (the multi-rank shape), so a
    // 25-token request waited 518 s behind two 324K prefills. It is retried at the next start
    // turn: within two prefill turns (2k ticks) of the refusal, while seq 2 still gets a chunk at
    // least every second prefill turn.
    for k in [None, Some(2usize), Some(3)] {
        let lines = run_scenario(&refused_short_scenario(k));
        let mut tick = 0usize;
        let (mut refused_at, mut ran_at, mut s2) = (None, None, Vec::new());
        for l in &lines {
            if l.starts_with("-- tick ") {
                tick += 1;
            } else if l.starts_with("prefill_chunk(") && l.contains(" s4@") {
                if l.contains("-> ERR") {
                    refused_at.get_or_insert(tick);
                } else {
                    ran_at.get_or_insert(tick);
                }
            } else if l.starts_with("prefill_chunk(") && l.contains(" s2@") {
                s2.push(tick);
            }
        }
        let s4_lines: Vec<&String> = lines
            .iter()
            .filter(|l| l.contains("s4@") || l.contains("out s4") || l.starts_with("-- tick"))
            .collect();
        let (Some(r), Some(a)) = (refused_at, ran_at) else {
            panic!("k={k:?}: s4 refused at {refused_at:?}, ran at {ran_at:?}: {s4_lines:#?}");
        };
        let bound = 2 * k.unwrap_or(1);
        let s2_last = *s2.last().unwrap();
        assert!(
            a <= r + bound,
            "k={k:?}: s4 refused at tick {r}, ran at {a} (bound {bound}); s2 chunks {s2:?}"
        );
        assert!(
            a < s2_last,
            "k={k:?}: s4 ran at {a}, after seq 2's prefill ended ({s2_last})"
        );
        let worst = s2.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        assert!(
            worst <= bound,
            "k={k:?}: seq 2 waited {worst} ticks between chunks (bound {bound}): {s2:?}"
        );
        let outs = outputs(&lines);
        assert_eq!(outs.len(), 4, "k={k:?}: {outs:?}");
        assert!(
            outs.iter().all(|o| o.contains("Done(stop")),
            "k={k:?}: {outs:?}"
        );
    }
}
