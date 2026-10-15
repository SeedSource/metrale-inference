// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Success, edge-case and device-error tests (the `a_`, `b_` and `c_` prefixes), driven through `ScriptedDeviceIo` against the real core (`SchedulerCore` under `metrale_scheduler::run`).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! The scripted router sits under the effect-trace decorator, so each
//! run yields the `fx` plan/effect lines beside the client outputs;
//! the assertions read those.
//!
//! These runs use max depth 1 (`ScriptedDeviceIo::new(.., 1)`);
//! `pipeline_tests.rs` covers depth 2.

use std::sync::{Arc, Mutex};

use metrale_scheduler::{Fault, ScriptedDeviceIo};

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario, run_scenario_with};
use super::scenarios;
use crate::scheduler::io::{SyncDeviceIo, TracingDeviceIo};

static SERIAL: Mutex<()> = Mutex::new(());

fn scripted(sc: &Scenario, setup: fn(&ScriptedDeviceIo<SyncDeviceIo>)) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let vocab = sc.cfg.vocab;
    run_scenario_with(
        sc,
        Arc::new(move |model, sink| {
            // 2026-09-25: the harness keys its model scripts the same
            // way: the session hash, or the prompt's first token for
            // a row that came back without one.
            let dev = ScriptedDeviceIo::new(
                SyncDeviceIo::new(model),
                |seq: &metrale_model_engine::traits::SequenceState| {
                    if seq.session_hash != 0 {
                        seq.session_hash
                    } else {
                        u64::from(seq.tokens.first().copied().unwrap_or(0))
                    }
                },
                vocab,
                1,
            );
            setup(&dev);
            Box::new(TracingDeviceIo::new(dev, sink))
        }),
    )
}

fn traced(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

fn output(lines: &[String], id: u64) -> String {
    let key = format!("out s{id}: ");
    lines
        .iter()
        .find_map(|l| l.strip_prefix(&key))
        .unwrap_or_else(|| panic!("no output for s{id}"))
        .to_string()
}

fn positions(lines: &[String], needle: &str) -> Vec<usize> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains(needle))
        .map(|(i, _)| i)
        .collect()
}

fn model_toks(n: usize, base: u32) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

fn three_streams() -> Scenario {
    Scenario {
        name: "scripted_three_streams",
        cfg: ModelCfg::default(),
        opts: RunOptions::default(),
        reqs: vec![
            ReqSpec::new(1, 5, model_toks(8, 10)),
            ReqSpec::new(2, 5, model_toks(8, 20)),
            ReqSpec::new(3, 7, model_toks(8, 30)),
        ],
    }
}

// 2026-09-25: success cases.

#[test]
fn a_decode_greedy_emits_the_device_script_in_order() {
    let lines = scripted(&three_streams(), |d| {
        d.script(1, [70, 71, 72, EOS]);
        d.script(2, [80, 81, EOS]);
        d.script(3, [90, 91, 92, 93, EOS]);
    });
    // 2026-09-25: the first token is the prefill's (the prefill lane
    // still samples from the model); every decode token is the
    // router's answer, and the stop comes from the script's EOS.
    assert_eq!(
        output(&lines, 1),
        "T10 T70 T71 T72 Done(stop, n=5, think=0, cached=0, acc=0, guard=None)"
    );
    assert_eq!(
        output(&lines, 2),
        "T20 T80 T81 Done(stop, n=4, think=0, cached=0, acc=0, guard=None)"
    );
    assert_eq!(
        output(&lines, 3),
        "T30 T90 T91 T92 T93 Done(stop, n=6, think=0, cached=0, acc=0, guard=None)"
    );
    // 2026-09-25: the model's own continuation never reaches a client.
    for l in lines.iter().filter(|l| l.starts_with("out s")) {
        assert!(
            !l.contains(" T11 ") && !l.contains(" T21 ") && !l.contains(" T31 "),
            "{l}"
        );
    }
}

#[test]
fn a_a_finished_sequence_is_released_before_the_survivors_are_compacted() {
    let lines = scripted(&three_streams(), |d| {
        d.script(1, [70, EOS]);
        d.script(2, [80, 81, 82, EOS]);
        d.script(3, [90, 91, 92, 93, 94, EOS]);
    });
    let releases = positions(&lines, "fx apply ReleaseSeq{cache=true, finish_sequence}");
    assert_eq!(releases.len(), 3, "one release per finished sequence");
    let compactions = positions(&lines, "fx apply CompactSlot");
    // 2026-09-25: s1 (slot 0) retires first; the survivors are
    // compacted down right after its release and before the next
    // decode step.
    let first_compaction = compactions
        .first()
        .copied()
        .expect("a compaction after the first retire");
    assert!(releases[0] < first_compaction);
    let next_launch = positions(&lines, "fx launch Decode")
        .into_iter()
        .find(|&i| i > releases[0])
        .expect("a decode step after the first retire");
    assert!(first_compaction < next_launch);
    assert!(lines.iter().any(|l| l == "fx apply EpShutdown"));
    assert!(
        releases
            .iter()
            .all(|&r| r < positions(&lines, "fx apply EpShutdown")[0])
    );
}

// 2026-09-25: edge cases.

#[test]
fn b_a_deadline_finishes_the_sequence_in_the_tick_it_expires() {
    let mut expired = ReqSpec::new(1, 5, model_toks(8, 10));
    expired.expired_deadline = true;
    let sc = Scenario {
        name: "scripted_deadline",
        cfg: ModelCfg::default(),
        opts: RunOptions::default(),
        reqs: vec![expired, ReqSpec::new(2, 5, model_toks(4, 20))],
    };
    let lines = scripted(&sc, |d| {
        d.script(1, [70, 71, 72, 73, 74, 75]);
        d.script(2, [80, 81, EOS]);
    });
    let out1 = output(&lines, 1);
    assert!(
        out1.ends_with(
            "Done(timeout, n=2, think=0, cached=0, acc=0, guard=Some(\"request_timeout\"))"
        ),
        "{out1}"
    );
    let release = positions(&lines, "fx apply ReleaseSeq{cache=true, finish_sequence}")[0];
    let launches = positions(&lines, "fx launch Decode");
    let before = launches.iter().filter(|&&i| i < release).count();
    assert_eq!(
        before, 1,
        "exactly one decode step ran before the timeout retire"
    );
    assert!(output(&lines, 2).ends_with("Done(stop, n=4, think=0, cached=0, acc=0, guard=None)"));
}

#[test]
fn b_a_lora_rotation_waits_for_quiescence() {
    let sc = Scenario {
        name: "scripted_lora",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            lora_rotation: Some("adapter-y".into()),
            ..RunOptions::default()
        },
        reqs: vec![ReqSpec::new(1, 5, model_toks(6, 10))],
    };
    let lines = scripted(&sc, |d| d.script(1, [70, 71, 72, EOS]));
    let lora = positions(&lines, "fx lora Rotate(adapter-y)");
    assert_eq!(lora.len(), 1);
    let release = positions(&lines, "fx apply ReleaseSeq{cache=true, finish_sequence}");
    assert_eq!(release.len(), 1);
    assert!(
        release[0] < lora[0],
        "the rotation waits for the active sequence to retire"
    );
    assert!(lines.iter().any(|l| l == "rotation ack: Done"));
}

#[test]
fn b_the_watchdog_restores_the_boundary_snapshot_before_resteering() {
    let sc = scenarios::all()
        .into_iter()
        .find(|s| s.name == "watchdog_rollback")
        .expect("scenario");
    let lines = traced(&sc);
    let saves = positions(&lines, "fx apply SsmSnapshotSave{slot=0}");
    let restores = positions(&lines, "fx apply SsmSnapshotRestore{slot=0}");
    assert!(!restores.is_empty(), "the content-loop watchdog re-steered");
    assert!(
        saves[0] < restores[0],
        "a boundary snapshot exists before it is restored"
    );
    let out = output(&lines, 1);
    assert!(out.contains(", 12, "), "{out}");
}

// 2026-09-25: device errors.

#[test]
fn c_kv_exhaustion_preempts_one_victim_and_the_rest_continue() {
    let lines = scripted(&three_streams(), |d| {
        d.script(1, [70, 71, 72, EOS]);
        d.script(2, [80, 81, EOS]);
        d.script(3, [90, 91, 92, EOS]);
        d.fail_next(Fault::KvExhausted);
    });
    let requeues = positions(&lines, "fx apply ReleaseSeq{cache=true, preempt_requeue}");
    assert_eq!(requeues.len(), 1, "exactly one victim is requeued");
    let launches = positions(&lines, "fx launch Decode{n=");
    let retry = launches
        .iter()
        .find(|&&i| i > requeues[0])
        .expect("a retry after the preemption");
    assert!(
        lines[*retry].starts_with("fx launch Decode{n=2"),
        "{}",
        lines[*retry]
    );
    // 2026-09-25: nobody is told about the preemption and everybody finishes.
    for id in 1..=3 {
        let out = output(&lines, id);
        assert!(!out.contains("Error"), "{out}");
        assert!(out.contains("Done(stop"), "{out}");
    }
}

/// 2026-10-15 (race-prefill, A192): the KV-exhaustion run with 4-token prefill chunks, the
/// resume lever on or off.
fn preempt_run(one_chunk_per_tick: bool) -> Vec<String> {
    let mut sc = three_streams();
    sc.opts.max_prefill_tokens = 4;
    sc.opts.resume_one_chunk_per_tick = one_chunk_per_tick;
    scripted(&sc, |d| {
        d.script(1, [70, 71, 72, EOS]);
        d.script(2, [80, 81, EOS]);
        d.script(3, [90, 91, 92, EOS]);
        d.fail_next(Fault::KvExhausted);
    })
}

/// 2026-10-15: the resume's chunks: every `prefill_chunk` after the requeue.
fn resume_chunks(lines: &[String]) -> Vec<usize> {
    let requeue = positions(lines, "fx apply ReleaseSeq{cache=true, preempt_requeue}")[0];
    positions(lines, "prefill_chunk(")
        .into_iter()
        .filter(|&i| i > requeue)
        .collect()
}

// 2026-10-15 (race-prefill, A192): `METRALE_RESUME_ONE_CHUNK_PER_TICK` through the real
// scheduler loop. Off, the victim's whole re-prefill runs inside one tick (no decode between
// its chunks: every other row waits). On, the other rows decode between its chunks, and every
// reply is the same.
#[test]
fn c_chunked_resume_lets_the_other_rows_decode_between_its_chunks_with_the_same_replies() {
    let off = preempt_run(false);
    let on = preempt_run(true);
    for lines in [&off, &on] {
        assert_eq!(
            positions(lines, "fx apply ReleaseSeq{cache=true, preempt_requeue}").len(),
            1,
            "exactly one victim is requeued"
        );
    }
    let decodes_between = |lines: &[String]| {
        let c = resume_chunks(lines);
        assert!(c.len() >= 2, "a multi-chunk resume: {c:?}");
        positions(lines, "fx launch Decode{n=")
            .into_iter()
            .filter(|&i| i > c[0] && i < c[c.len() - 1])
            .count()
    };
    assert_eq!(decodes_between(&off), 0, "off: the resume runs inside one tick");
    assert!(decodes_between(&on) >= 1, "on: the other rows decode between chunks");
    // 2026-10-15: the same chunk plan in both runs.
    let plan = |lines: &[String]| -> Vec<String> {
        resume_chunks(lines)
            .into_iter()
            .map(|i| lines[i].split(", stream=").next().unwrap_or("").to_string())
            .collect()
    };
    assert_eq!(plan(&off), plan(&on));
    for id in 1..=3 {
        let (a, b) = (output(&off, id), output(&on, id));
        assert!(!b.contains("Error"), "{b}");
        assert!(b.contains("Done(stop"), "{b}");
        assert_eq!(reply_tokens(&a), reply_tokens(&b), "s{id} reply");
    }
}

/// 2026-10-15 (race-prefill, A192): the KV-exhaustion run for `METRALE_RESUME_SLICED`. s1-s3
/// decode long replies from the first tick; the ninth decode step is refused and s1 (fewest
/// tokens out, lowest slot) is the victim: a 4-token-chunk resume of its 13-token history. s4
/// arrives as it starts to resume. The multi-rank shape: no fused mixed step, the
/// while-decoding cap on (256, above every chunk here, so the chunk plan is the same with the
/// lever on or off).
fn sliced_run(sliced: bool, k: Option<usize>, exhaust_at: usize, s4_at: usize) -> Vec<String> {
    let reply = |n: u32, base: u32| -> Vec<u32> {
        let mut v: Vec<u32> = (0..n).map(|i| base + i).collect();
        v.push(EOS);
        v
    };
    let s3 = ReqSpec::new(3, 4, reply(40, 300));
    let mut s4 = ReqSpec::new(4, 8, reply(4, 400));
    s4.arrive_at_tick = Some(s4_at);
    let mut cfg = ModelCfg::default();
    cfg.kv_exhaust_at = vec![exhaust_at];
    let sc = Scenario {
        name: "sliced_resume",
        cfg,
        opts: RunOptions {
            max_prefill_tokens: 4,
            no_mix: true,
            prefill_chunk_while_decoding: Some(256),
            decode_steps_per_prefill_chunk: k,
            resume_one_chunk_per_tick: true,
            resume_sliced: sliced,
            ..RunOptions::default()
        },
        reqs: vec![
            ReqSpec::new(1, 4, reply(40, 100)),
            ReqSpec::new(2, 4, reply(40, 200)),
            s3,
            s4,
        ],
    };
    run_scenario(&sc)
}

/// 2026-10-15: per tick after the requeue: (ran a resume chunk of s1, ran a chunk of s4, ran a
/// decode step).
fn resume_ticks(lines: &[String]) -> Vec<(bool, bool, bool)> {
    let requeue = positions(lines, "fx apply ReleaseSeq{cache=true, preempt_requeue}")[0];
    let mut out: Vec<(bool, bool, bool)> = Vec::new();
    for l in &lines[requeue..] {
        if l.starts_with("-- tick ") {
            out.push((false, false, false));
        } else if let Some(t) = out.last_mut() {
            if l.starts_with("prefill_chunk(") && l.contains(" s1@") {
                t.0 = true;
            }
            if l.starts_with("prefill_chunk(") && l.contains(" s4@") {
                t.1 = true;
            }
            if l.starts_with("decode_batch(") || l.starts_with("decode(") {
                t.2 = true;
            }
        }
    }
    out
}

// 2026-10-15 (race-prefill, A192): `METRALE_RESUME_SLICED` through the real scheduler loop.
// Without it (one chunk per tick alone) the late request waits for the whole resume. With it the
// late request is admitted while the resume is in flight, its chunks and the resume's take turns
// (never two prefill chunks in one tick), with `k` the decoders take `k` steps per resume chunk,
// and every reply is the same.
const EXHAUST_AT: usize = 8;
const S4_AT: usize = 10;

#[test]
fn c_sliced_resume_admits_new_requests_and_takes_k_decode_steps_per_chunk() {
    let base = sliced_run(false, None, EXHAUST_AT, S4_AT);
    let t = resume_ticks(&base);
    let s1: Vec<usize> = t.iter().enumerate().filter(|(_, x)| x.0).map(|(i, _)| i).collect();
    let s4: Vec<usize> = t.iter().enumerate().filter(|(_, x)| x.1).map(|(i, _)| i).collect();
    assert!(s1.len() >= 2, "a multi-chunk resume: {t:?}");
    assert!(!s4.is_empty(), "s4 prefilled: {t:?}");
    assert!(
        s4[0] > s1[s1.len() - 1],
        "one chunk per tick alone: s4 waits for the whole resume: {t:?}"
    );
    for k in [None, Some(2), Some(3)] {
        let lines = sliced_run(true, k, EXHAUST_AT, S4_AT);
        let t = resume_ticks(&lines);
        let s1: Vec<usize> = t.iter().enumerate().filter(|(_, x)| x.0).map(|(i, _)| i).collect();
        let s4: Vec<usize> = t.iter().enumerate().filter(|(_, x)| x.1).map(|(i, _)| i).collect();
        assert!(s1.len() >= 2 && !s4.is_empty(), "k={k:?}: {t:?}");
        assert!(s4[0] < s1[s1.len() - 1], "k={k:?}: s4 starts during the resume: {t:?}");
        assert!(t.iter().all(|x| !(x.0 && x.1)), "k={k:?}: one prefill chunk per tick: {t:?}");
        // 2026-10-15: between two resume chunks: the other prefill's chunk ticks plus k - 1
        // decode-only ticks after each chunk, every one with a decode step.
        let want = k.unwrap_or(1) - 1;
        for w in s1.windows(2) {
            let between = &t[w[0] + 1..w[1]];
            assert!(between.iter().all(|x| x.2), "k={k:?}: decoders step every tick: {t:?}");
            let s4_chunks = between.iter().filter(|x| x.1).count();
            assert_eq!(
                between.len(),
                want + s4_chunks * (1 + want),
                "k={k:?}: decode-only ticks between resume chunks: {t:?}"
            );
        }
        for id in 1..=4 {
            let (a, b) = (output(&base, id), output(&lines, id));
            assert!(!b.contains("Error"), "k={k:?} s{id}: {b}");
            assert!(b.contains("Done(stop"), "k={k:?} s{id}: {b}");
            assert_eq!(reply_tokens(&a), reply_tokens(&b), "k={k:?} s{id} reply");
        }
    }
}

#[test]
fn c_resume_sliced_alone_stays_off() {
    // 2026-10-15 (race-prefill, A192): without one chunk per tick the lever resolves off, so
    // the run is the synchronous resume's, model call for model call.
    let mut sc = three_streams();
    sc.opts.max_prefill_tokens = 4;
    let off = scripted(&sc, |d| {
        d.script(1, [70, 71, 72, EOS]);
        d.fail_next(Fault::KvExhausted);
    });
    sc.opts.resume_sliced = true;
    let on = scripted(&sc, |d| {
        d.script(1, [70, 71, 72, EOS]);
        d.fail_next(Fault::KvExhausted);
    });
    assert_eq!(off, on);
}

/// 2026-10-14 (race-memory, A184): the reply tokens of `out` without the closing
/// `Done(...)` stats (acceptance counts differ when drafts are dropped).
fn reply_tokens(out: &str) -> &str {
    out.split(" Done(").next().unwrap_or(out)
}

#[test]
fn c_kv_refusal_in_a_batched_mtp_verify_preempts_one_victim_and_no_reply_is_cut() {
    // 2026-10-14 (race-memory, A184): kvagcheck2 (n3n4) saw one KV refusal in a C=4
    // batched MTP verify end all four replies early with finish "stop". Here the second
    // batched verify of three MTP streams is refused: one victim is requeued, and every
    // reply still carries the same tokens as the fault-free run and ends without an error.
    let reference = run_scenario(&scenarios::batched_verify());
    let mut sc = scenarios::batched_verify();
    sc.name = "batched_verify_kv_refusal";
    sc.cfg.verify_kv_exhaust_at = vec![1];
    let lines = run_scenario(&sc);
    assert_eq!(
        positions(&lines, "-> ERR KV cache exhausted").len(),
        1,
        "the scripted refusal ran once"
    );
    assert_eq!(
        positions(&lines, "fx apply ReleaseSeq{cache=true, preempt_requeue}").len(),
        1,
        "exactly one victim is requeued"
    );
    for id in 1..=3 {
        let (want, got) = (output(&reference, id), output(&lines, id));
        assert!(!got.contains("Error"), "s{id}: {got}");
        assert!(got.contains("Done(stop"), "s{id}: {got}");
        assert_eq!(
            reply_tokens(&got),
            reply_tokens(&want),
            "s{id} reply was cut or changed"
        );
    }
}

#[test]
fn c_a_fatal_device_error_fails_every_row_in_the_batch_and_only_them() {
    // 2026-09-25: request 3 arrives at tick 1, while requests 1 and 2
    // are active; the scripted fault fails their decode, and request 3
    // still completes.
    let mut late = ReqSpec::new(3, 5, model_toks(6, 30));
    late.arrive_at_tick = Some(1);
    let sc = Scenario {
        name: "scripted_fatal",
        cfg: ModelCfg::default(),
        opts: RunOptions::default(),
        reqs: vec![
            ReqSpec::new(1, 5, model_toks(6, 10)),
            ReqSpec::new(2, 5, model_toks(6, 20)),
            late,
        ],
    };
    let lines = scripted(&sc, |d| {
        d.script(3, [90, 91, EOS]);
        d.fail_next(Fault::Fatal);
    });
    assert_eq!(output(&lines, 1), "T10 Error(scripted device fault)");
    assert_eq!(output(&lines, 2), "T20 Error(scripted device fault)");
    // 2026-09-25: the rows that failed are released without a cache
    // offer; the late arrival is untouched and completes.
    assert_eq!(
        positions(&lines, "fx apply ReleaseSeq{cache=false, send_error}").len(),
        2
    );
    assert_eq!(
        output(&lines, 3),
        "T30 T90 T91 Done(stop, n=4, think=0, cached=0, acc=0, guard=None)"
    );
}

#[test]
fn c_kv_refusal_in_a_batched_dflash_verify_preempts_one_victim_and_no_reply_is_cut() {
    // 2026-10-14 (race-memory, A184): the DFlash batched verify used to put its drafts back
    // on a KV refusal and return, so a lasting refusal repeated every step with no
    // preemption (a stalled batch). Now it preempts one victim, as the MTP verify does.
    let reference = run_scenario(&scenarios::dflash("dflash_batched3", 3));
    let mut sc = scenarios::dflash("dflash_batched3_kv_refusal", 3);
    sc.cfg.verify_kv_exhaust_at = vec![1];
    let lines = run_scenario(&sc);
    assert_eq!(
        positions(&lines, "-> ERR KV cache exhausted").len(),
        1,
        "the scripted refusal ran once"
    );
    assert_eq!(
        positions(&lines, "fx apply ReleaseSeq{cache=true, preempt_requeue}").len(),
        1,
        "exactly one victim is requeued"
    );
    for id in 1..=3 {
        let (want, got) = (output(&reference, id), output(&lines, id));
        assert!(!got.contains("Error"), "s{id}: {got}");
        assert!(got.contains("Done(stop"), "s{id}: {got}");
        assert_eq!(
            reply_tokens(&got),
            reply_tokens(&want),
            "s{id} reply was cut or changed"
        );
    }
}
