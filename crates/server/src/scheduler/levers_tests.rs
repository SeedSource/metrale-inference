// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Scheduler lever resolution tests: lever polarities in
//! `defaults()` and in `from_env()`, and the guard against environment
//! reads on the per-token path.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn the_five_opt_out_levers_ship_on() {
    // 2026-09-25: Each is spelled as a negative env var; an opt-in resolver
    // would turn all five off.
    let d = SchedLevers::defaults();
    assert!(d.fast_greedy_grammar, "METRALE_DISABLE_FAST_GREEDY");
    assert!(d.fast_masked, "METRALE_DISABLE_FAST_MASKED");
    assert!(d.mtp_minp, "METRALE_NO_MTP_MINP");
    assert!(d.mtp_verify_sample, "METRALE_NO_MTP_VERIFY_SAMPLE");
    assert!(d.forced_token_fastpath, "METRALE_DISABLE_FORCED_TOKEN");
}

/// 2026-09-25: The two turn-termination levers ship ON, and they accept
/// `false` as well as `0`.
///
/// Pinned in `from_env()`, not just `defaults()`, for the reason
/// `spec_think_per_lane_in_the_resolver_the_server_actually_uses` exists:
/// `defaults()` is a hand-written literal and cannot catch a change to
/// what the server resolves.
#[test]
fn the_turn_termination_levers_ship_on_in_the_live_resolver() {
    // 2026-09-25: SAFETY: nothing else in this test binary writes these
    // variables. `cargo test` runs tests on parallel threads, so a
    // concurrent environment access from another test is not excluded.
    unsafe {
        std::env::remove_var("METRALE_TOOL_RESPONSE_STOP");
        std::env::remove_var("METRALE_TOOL_EOS_ESCAPE");
    }
    let live = SchedLevers::from_env(None);
    assert!(live.tool_response_stop, "METRALE_TOOL_RESPONSE_STOP");
    assert!(live.tool_eos_escape, "METRALE_TOOL_EOS_ESCAPE");
    assert!(SchedLevers::defaults().tool_response_stop);
    assert!(SchedLevers::defaults().tool_eos_escape);

    // 2026-09-25: The rule they resolve through, exercised directly.
    // `!= "1"` would pass the `0` case and silently ignore every other
    // spelling.
    use crate::scheduler::helpers::parse_flag_default_on as f;
    assert!(f(None));
    assert!(f(Some("1")));
    assert!(f(Some("true")));
    assert!(f(Some("junk")), "unknown values keep the shipped behaviour");
    assert!(!f(Some("0")));
    assert!(!f(Some("false")));
    assert!(!f(Some("FALSE")), "case-insensitive");
    assert!(!f(Some("  0  ")), "trimmed");
}

#[test]
fn every_opt_in_lever_ships_off() {
    let d = SchedLevers::defaults();
    assert!(!d.force_temp_zero);
    assert!(!d.dflash_masked_verify && !d.dflash_adaptive && !d.dflash_spec_think);
    // 2026-09-29: A146: no explicit MTP spec-in-think choice: the model's
    // default applies.
    assert_eq!(d.mtp_spec_think_env, None);
    assert!(!d.disable_watchdogs);
    assert!(!d.decode_timing && !d.mtp_timing && !d.adadec_diagnostic);
}

/// 2026-09-25: `defaults()` cannot catch a change to what the server
/// resolves.
///
/// It is a hand-written struct literal; `from_env()` is the constructor
/// `met serve` actually calls. A resolver change that touches only
/// `from_env()` can leave `defaults()` — and so
/// `every_opt_in_lever_ships_off` above — unchanged and green while
/// production resolves a different value.
///
/// So this asserts the resolver itself. 2026-09-29: A146: the contract
/// changed: spec-in-think is split per lane, and the MTP lane's default is
/// per model, on only for GLM-5.3, whose parity chain made spec-in-think
/// commit exactly what spec-off decode would (Atlas, 2026-09-26: K=3
/// byte-identical 6/6; TEB 156/176 identical to spec-off per scenario).
/// Every other model, and the DFlash lane on every model, stays opt-in until
/// it passes the same gates. With no env set the resolver must therefore
/// express NO opinion for MTP (`None`, so the model default applies) and keep
/// DFlash off.
///
/// One test on purpose: every case mutates the same process env, and the
/// harness runs tests on parallel threads.
#[test]
fn spec_think_per_lane_in_the_resolver_the_server_actually_uses() {
    use metrale_speculative::mtp_gate::mtp_spec_think_default;
    const MTP: &str = "METRALE_MTP_SPEC_THINK";
    const DFLASH: &str = "METRALE_DFLASH_SPEC_THINK";
    let glm = mtp_spec_think_default("glm5_next");
    let other = mtp_spec_think_default("qwen3_next");
    // 2026-09-29: SAFETY: nothing else in this test binary writes these
    // variables. `cargo test` runs tests on parallel threads, so a
    // concurrent environment access from another test is not excluded.
    let set = |mtp: Option<&str>, dflash: Option<&str>| unsafe {
        match mtp {
            Some(v) => std::env::set_var(MTP, v),
            None => std::env::remove_var(MTP),
        }
        match dflash {
            Some(v) => std::env::set_var(DFLASH, v),
            None => std::env::remove_var(DFLASH),
        }
    };

    set(None, None);
    let live = SchedLevers::from_env(None);
    assert_eq!(
        live.mtp_spec_think_env, None,
        "with no env set the resolver must defer to the model default"
    );
    assert!(
        live.mtp_spec_think(glm),
        "GLM-5.3 MTP spec-in-think ships ON"
    );
    assert!(
        !live.mtp_spec_think(other),
        "non-GLM MTP spec-in-think must stay OPT-IN until independently qualified"
    );
    assert!(
        !live.dflash_spec_think,
        "METRALE_DFLASH_SPEC_THINK must stay OPT-IN for the DFlash lane: from_env() \
         resolved it ON. DFlash-in-think has no GPU + TEB qualification yet."
    );
    // 2026-09-29: the two DFlash levers graduated earlier stay graduated.
    assert!(
        live.dflash_masked_verify,
        "masked_verify is intentionally default-ON"
    );
    assert!(
        live.dflash_seam_serial,
        "seam_serial is intentionally default-ON"
    );

    set(Some("0"), None);
    let l = SchedLevers::from_env(None);
    assert!(
        !l.mtp_spec_think(glm),
        "METRALE_MTP_SPEC_THINK=0 turns GLM off"
    );

    set(None, Some("0"));
    let l = SchedLevers::from_env(None);
    assert!(
        !l.mtp_spec_think(glm),
        "METRALE_DFLASH_SPEC_THINK=0 must keep working as an off switch for MTP"
    );
    assert!(!l.dflash_spec_think);

    set(None, Some("1"));
    let l = SchedLevers::from_env(None);
    assert!(
        l.mtp_spec_think(glm) && l.mtp_spec_think(other),
        "METRALE_DFLASH_SPEC_THINK=1 keeps its pre-split MTP opt-in on every model"
    );
    assert!(
        l.dflash_spec_think,
        "METRALE_DFLASH_SPEC_THINK=1 opts DFlash in"
    );

    set(Some("1"), None);
    let l = SchedLevers::from_env(None);
    assert!(
        l.mtp_spec_think(other),
        "METRALE_MTP_SPEC_THINK=1 is the per-lane opt-in for qualification runs"
    );
    assert!(
        !l.dflash_spec_think,
        "the MTP switch must not opt the DFlash lane in"
    );

    set(Some("1"), Some("0"));
    assert!(
        !SchedLevers::from_env(None).mtp_spec_think(glm),
        "an explicit =0 on either variable beats an opt-in"
    );

    set(None, None);
}

#[test]
fn mtp_spec_think_env_resolution_table() {
    use crate::scheduler::levers::resolve_mtp_spec_think_env as r;
    assert_eq!(r(None, None), None);
    assert_eq!(r(Some("0"), None), Some(false));
    assert_eq!(r(None, Some("0")), Some(false));
    assert_eq!(r(Some("1"), None), Some(true));
    assert_eq!(r(None, Some("1")), Some(true));
    assert_eq!(r(Some("1"), Some("0")), Some(false));
    assert_eq!(r(Some("0"), Some("1")), Some(false));
    // 2026-09-29: anything but an exact 0/1 is no opinion, like `opt_in`.
    assert_eq!(r(Some("true"), Some("")), None);
}

/// 2026-09-29: A146: the lane and model split, through the levers the
/// server resolves with no env set (`defaults()`: no explicit choice) and
/// the per-model default resolved at serve load.
fn in_think_eligible(levers: &SchedLevers, model_type: &str, dflash_lane: bool) -> bool {
    use metrale_speculative::mtp_gate::{
        mtp_spec_think_default, spec_dispatch_eligible, spec_think_for_lane,
    };
    let spec_think = spec_think_for_lane(
        dflash_lane,
        levers.mtp_spec_think(mtp_spec_think_default(model_type)),
        levers.dflash_spec_think,
    );
    spec_dispatch_eligible(true, 0, 50, false, false, spec_think, 0, dflash_lane)
}

#[test]
fn glm53_mtp_speculates_in_think_by_default() {
    let d = SchedLevers::defaults();
    assert!(in_think_eligible(&d, "glm5_next", false));
    assert!(in_think_eligible(&d, "glm5_next_text", false));
}

#[test]
fn other_models_mtp_stays_serial_in_think_by_default() {
    let d = SchedLevers::defaults();
    assert!(!in_think_eligible(&d, "qwen3_next", false));
    // 2026-09-29: METRALE_DFLASH_SPEC_THINK=1 (or METRALE_MTP_SPEC_THINK=1)
    // opts it in.
    let mut opted = SchedLevers::defaults();
    opted.mtp_spec_think_env = Some(true);
    assert!(in_think_eligible(&opted, "qwen3_next", false));
}

#[test]
fn glm53_mtp_spec_think_zero_turns_it_off() {
    let mut off = SchedLevers::defaults();
    // 2026-09-29: METRALE_MTP_SPEC_THINK=0.
    off.mtp_spec_think_env = Some(false);
    assert!(!in_think_eligible(&off, "glm5_next", false));
}

#[test]
fn dflash_lane_unchanged_opt_in_on_every_model() {
    let d = SchedLevers::defaults();
    for m in ["glm5_next", "qwen3_next"] {
        assert!(!in_think_eligible(&d, m, true), "{m}: DFlash stays opt-in");
    }
    // 2026-09-29: the MTP override never reaches the DFlash lane.
    let mut mtp_on = SchedLevers::defaults();
    mtp_on.mtp_spec_think_env = Some(true);
    assert!(!in_think_eligible(&mtp_on, "glm5_next", true));
    let mut df_on = SchedLevers::defaults();
    // 2026-09-29: METRALE_DFLASH_SPEC_THINK=1.
    df_on.dflash_spec_think = true;
    assert!(in_think_eligible(&df_on, "qwen3_next", true));
}

#[test]
fn the_loop_watchdog_is_toggleable_at_runtime() {
    // 2026-09-25: The one lever that changes mid-run: the TUI ops REPL
    // flips it.
    let d = SchedLevers::defaults();
    assert!(!d.loop_watchdog());
    d.set_loop_watchdog(true);
    assert!(d.loop_watchdog());
    d.set_loop_watchdog(false);
    assert!(!d.loop_watchdog());
}

#[test]
fn an_absent_mtp_gate_flag_leaves_the_legacy_variable_reachable() {
    // 2026-09-25: `None` (no `--mtp-gate` flag) must leave
    // `METRALE_MTP_GATE_FORCE` in charge.
    //
    // 2026-09-25: The env var is process-global with no reset, so this is
    // the only test in this binary that may write it — a second writer
    // would make both order-dependent.
    unsafe { std::env::remove_var("METRALE_MTP_GATE_FORCE") };
    assert!(
        !resolve_mtp_gate_force(None),
        "an absent flag leaves the decision to the variable, which is unset"
    );
    assert!(resolve_mtp_gate_force(Some(true)));
    assert!(!resolve_mtp_gate_force(Some(false)));
    assert!(
        SchedLevers::from_env(Some(true)).mtp_gate_force,
        "and the carried levers read the same resolution — one rule, not two"
    );
}

#[test]
fn two_runs_hold_independent_levers() {
    let a = SchedLevers::defaults();
    let b = SchedLevers {
        dflash_adaptive: true,
        ..SchedLevers::defaults()
    };
    assert!(!a.dflash_adaptive && b.dflash_adaptive);
    a.set_loop_watchdog(true);
    assert!(!b.loop_watchdog(), "and independent runtime state");
}

/// 2026-09-25: The scheduler thread must not read the environment per token.
///
/// `emit_step` and `decode_logits_step` run once per generated token per
/// sequence. An `std::env::var` call there allocates and takes the
/// process-wide environment lock, once per token per sequence instead of
/// once per run. The levers this path needs are `SchedLevers` fields.
///
/// A source-level check because the property is "who may read the
/// environment", which no runtime assertion can observe. The sibling table
/// for `metrale-model-layers` lives in `layers/ops/hot_path_env_guards.rs`.
#[test]
fn the_per_token_scheduler_path_does_not_read_the_environment() {
    // 2026-09-26: (module, functions still allowed to read). A module is
    // its `<module>.rs` plus every non-test `.rs` file under `<module>/`, so
    // code that a split moves into a child file stays scanned.
    const GUARDED: [(&str, &[&str]); 7] = [
        ("emit_step", &[]),
        // 2026-09-25: Per verify step.
        ("verify_dflash_step", &[]),
        ("verify_k2_step", &[]),
        // 2026-09-25: Per prefill chunk.
        ("prefill_a_step", &[]),
        (
            // 2026-09-25: Per sequence per decode step.
            "decode_logits_seq",
            &["process_seq_logits"],
        ),
        // 2026-09-25: Every lever this module needs is a `SchedLevers`
        // field, so the whole module is on the no-read list.
        ("decode_logits_step", &[]),
        (
            "helpers",
            // 2026-09-26: `WatchdogParams::from_behavior` in
            // `helpers/watchdog.rs` runs once per model load.
            &["from_behavior"],
        ),
    ];
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let mut offenders = Vec::new();
    let mut scanned = 0usize;
    for (module, allowed) in GUARDED {
        for path in module_files(&dir, module) {
            scanned += 1;
            let file = path.strip_prefix(&dir).unwrap_or(&path).display();
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{} is a guarded path: {e}", path.display()));
            let mut current = "<file scope>".to_string();
            for (i, line) in text.lines().enumerate() {
                let trimmed = line.trim_start();
                for prefix in ["pub(crate) fn ", "pub(super) fn ", "pub fn ", "fn "] {
                    if let Some(rest) = trimmed.strip_prefix(prefix) {
                        current = rest
                            .split(['(', '<'])
                            .next()
                            .unwrap_or("?")
                            .trim()
                            .to_string();
                        break;
                    }
                }
                // 2026-09-25: Comments name these variables throughout; only
                // code counts.
                let code = line.split("//").next().unwrap_or("");
                if code.contains("std::env::var") && !allowed.contains(&current.as_str()) {
                    offenders.push(format!("{file}:{} in `{current}`", i + 1));
                }
            }
        }
    }
    // 2026-09-26: 16 files hold these modules today. A lower count means a
    // module moved out from under this table, not that the path got smaller.
    assert!(scanned >= 16, "only {scanned} guarded files were found");
    assert!(
        offenders.is_empty(),
        "environment reads on the per-token scheduler path. Resolve the \
         variable ONCE into `SchedLevers` and read `sched.levers` instead — \
         and reuse the parser in `helpers`, do not re-spell the rule: \
         {offenders:?}"
    );
}

/// 2026-09-25: The verify-step levers' polarities. `dflash_eagle_fix` is
/// asserted against `from_env()` as well as `defaults()`, for the reason
/// `spec_think_per_lane_in_the_resolver_the_server_actually_uses` exists.
#[test]
fn the_verify_step_levers_hold_their_polarities() {
    let d = SchedLevers::defaults();
    assert!(d.dflash_eagle_fix, "the EAGLE append fix ships ON");
    assert!(!d.dflash_step_timing);
    assert!(!d.vision_timing);

    // 2026-09-25: SAFETY: nothing else in this test binary writes these
    // variables. `cargo test` runs tests on parallel threads, so a
    // concurrent environment access from another test is not excluded.
    unsafe { std::env::remove_var("METRALE_DFLASH_EAGLE_FIX") };
    assert!(
        SchedLevers::from_env(None).dflash_eagle_fix,
        "METRALE_DFLASH_EAGLE_FIX must stay DEFAULT-ON in the resolver the \
         server actually uses — `defaults()` is a hand-written literal and \
         cannot catch a change here"
    );
}

/// 2026-09-26: `<module>.rs` and every `.rs` file below `<module>/` whose
/// name does not end in `tests.rs`. The module file itself must exist.
fn module_files(dir: &std::path::Path, module: &str) -> Vec<std::path::PathBuf> {
    let root = dir.join(format!("{module}.rs"));
    assert!(root.is_file(), "{} is a guarded path", root.display());
    let mut files = vec![root];
    let mut pending = vec![dir.join(module)];
    while let Some(d) = pending.pop() {
        if !d.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&d).expect("readable scheduler directory") {
            let path = entry.expect("readable scheduler directory").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                pending.push(path);
            } else if name.ends_with(".rs") && !name.ends_with("tests.rs") {
                files.push(path);
            }
        }
    }
    files
}
