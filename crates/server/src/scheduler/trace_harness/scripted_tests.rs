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
