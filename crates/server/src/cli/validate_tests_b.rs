// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `validate_serve_args`, second file (`validate_tests.rs` holds the first).
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.
use super::*;
use clap::Parser;

fn parse(extra: &[&str]) -> ServeArgs {
    let mut argv = vec!["met", "serve", "dummy/model", "--model-name", "dummy"];
    argv.extend_from_slice(extra);
    match super::super::Cli::parse_from(argv).command {
        super::super::Command::Serve(a) => a,
        super::super::Command::Benchmark(_)
        | super::super::Command::DumpServeOptions
        | super::super::Command::SyncRecipes
        | super::super::Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
    }
}

#[test]
fn flagship_recipe_is_accepted() {
    // 2026-09-26: bf16 with `--kv-high-precision-layers auto` is redundant but
    // valid.
    assert!(
        validate_serve_args(&parse(&[
            "--kv-cache-dtype",
            "bf16",
            "--lm-head-dtype",
            "nvfp4",
            "--kv-high-precision-layers",
            "auto",
            "--scheduler",
            "slai",
            "--speculative",
            "--num-drafts",
            "1",
            "--mtp-quantization",
            "bf16",
            "--enable-prefix-caching",
        ]))
        .is_ok()
    );
}

#[test]
fn enum_typos_are_rejected() {
    let err = validate_serve_args(&parse(&["--scheduler", "fifoo"])).unwrap_err();
    assert!(err.contains("--scheduler"));
    assert!(err.contains("fifo, slai"));
}

#[test]
fn multiple_violations_all_reported() {
    let err = validate_serve_args(&parse(&[
        "--require-auth",
        "--num-drafts",
        "3",
        "--rank",
        "5",
        "--world-size",
        "2",
    ]))
    .unwrap_err();
    assert!(err.contains("[1]"));
    assert!(err.contains("[2]"));
    assert!(err.contains("[3]"));
}

#[test]
fn gpu_mem_util_range_enforced() {
    assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "1.5"])).is_err());
    assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "0.0"])).is_err());
    assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "0.9"])).is_ok());
}

/// 2026-09-26: These flags fall back to MODEL.toml only when omitted, so an
/// explicit value, even one equal to the engine default, must parse as `Some`.
#[test]
fn model_toml_backed_flags_distinguish_omitted_from_explicit() {
    let omitted = parse(&[]);
    assert_eq!(omitted.num_drafts, None);
    assert_eq!(omitted.kv_cache_dtype, None);
    assert_eq!(omitted.fp8_kv_calibration_tokens, None);

    let explicit = parse(&[
        "--num-drafts",
        "1",
        "--kv-cache-dtype",
        "fp8",
        "--fp8-kv-calibration-tokens",
        "0",
    ]);
    assert_eq!(explicit.num_drafts, Some(1));
    assert_eq!(explicit.kv_cache_dtype.as_deref(), Some("fp8"));
    // 2026-09-26: An explicit 0 turns calibration off even when MODEL.toml
    // enables it.
    assert_eq!(explicit.fp8_kv_calibration_tokens, Some(0));
}

#[test]
fn f16_pool_is_a_published_dtype_and_inherits_every_f16_rule() {
    use metrale_model_layers::layers::qwen3_ssm::ssm_h_dtype_bits;

    // 2026-09-26: The validator and `KernelFlagPlan` both decode through this
    // function, so a spelling accepted here publishes exactly these bits.
    assert_eq!(ssm_h_dtype_bits(None), (false, false));
    assert_eq!(ssm_h_dtype_bits(Some("f32")), (false, false));
    assert_eq!(ssm_h_dtype_bits(Some("f16")), (true, false));
    // 2026-09-26: Never the pool without `h_f16`: an FP32 h-state in a pool
    // sized at 2 bytes per element would be written out of bounds.
    assert_eq!(ssm_h_dtype_bits(Some("f16-pool")), (true, true));

    assert!(
        validate_serve_args(&parse(&["--ssm-h-dtype", "f16-pool", "--gdn-fused-norm"])).is_ok(),
        "the supported stage-3 pairing"
    );
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16-pool"])).is_err());
    assert!(
        validate_serve_args(&parse(&[
            "--exact-verify",
            "--ssm-h-dtype",
            "f16-pool",
            "--gdn-fused-norm",
        ]))
        .is_err()
    );
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16pool"])).is_err());
    assert_eq!(ssm_h_dtype_bits(Some("f16pool")), (false, false));
}

#[test]
fn dflash_refuses_the_f16_h_state_only_at_the_untwinned_width() {
    for dtype in ["f16", "f16-pool"] {
        let err = validate_serve_args(&parse(&[
            "--dflash",
            "--dflash-gamma",
            "16",
            "--ssm-h-dtype",
            dtype,
            "--gdn-fused-norm",
        ]))
        .unwrap_err();
        assert!(err.contains("dflash-gamma"), "{dtype}: {err}");
        assert!(
            validate_serve_args(&parse(&[
                "--dflash",
                "--dflash-gamma",
                "10",
                "--ssm-h-dtype",
                dtype,
                "--gdn-fused-norm",
            ]))
            .is_ok(),
            "{dtype}: gamma 10 (width 11) has a twin and must be allowed"
        );
    }
    assert!(validate_serve_args(&parse(&["--dflash"])).is_ok());
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"])).is_ok());
    assert!(
        validate_serve_args(&parse(&["--dflash", "--ssm-h-dtype", "f32"])).is_ok(),
        "the FP32 h-state has always been DFlash's supported pairing"
    );
}

// 2026-09-26: DFlash under the f16 h-state pool. The verify width is
// gamma + 1 and the FP16 twins reach width `MAX_F16_TWIN_K` (16), so the last
// allowed gamma is 15.

fn f16_dflash(gamma: Option<&str>) -> ServeArgs {
    let mut argv = vec![
        "--ssm-h-dtype",
        "f16-pool",
        "--gdn-fused-norm",
        "--dflash",
        "--draft-model",
        "some/drafter",
    ];
    if let Some(g) = gamma {
        argv.push("--dflash-gamma");
        argv.push(g);
    }
    parse(&argv)
}

#[test]
fn dflash_under_the_f16_pool_is_allowed_up_to_the_last_twinned_width() {
    for g in ["4", "8", "10", "15"] {
        assert!(
            validate_serve_args(&f16_dflash(Some(g))).is_ok(),
            "gamma {g} (verify width {}) has an FP16 twin and must be allowed",
            g.parse::<usize>().unwrap() + 1
        );
    }
}

#[test]
fn dflash_gamma_16_is_refused_because_width_17_has_no_twin() {
    let v = validate_serve_args(&f16_dflash(Some("16")));
    let msg = format!(
        "{:#}",
        v.expect_err("gamma 16 is verify width 17 — no FP16 twin")
    );
    assert!(
        msg.contains("f16") && msg.contains("dflash-gamma"),
        "the refusal must name both halves of the incompatible pair: {msg}"
    );
}

#[test]
fn an_unset_gamma_is_left_to_the_runtime_backstop() {
    // 2026-09-26: An unset gamma resolves from the drafter checkpoint through
    // `default_dflash_gamma`, which clamps it to `MAX_F16_TWIN_DFLASH_GAMMA`.
    assert!(validate_serve_args(&f16_dflash(None)).is_ok());
}

#[test]
fn hermetic_closes_both_channels_through_the_resolvers() {
    let a = parse(&["--hermetic"]);
    assert!(
        !a.prefix_caching_enabled(),
        "--hermetic must close the radix KV prefix cache"
    );
    assert_eq!(
        a.mtp_gate_force(),
        Some(true),
        "--hermetic must PIN the MTP gate rather than leave it to the environment"
    );
}

#[test]
fn without_hermetic_neither_channel_is_touched() {
    let a = parse(&[]);
    assert!(!a.prefix_caching_enabled(), "the flag's own default is off");
    assert_eq!(
        a.mtp_gate_force(),
        None,
        "absent means the documented METRALE_MTP_GATE_FORCE fallback decides"
    );
    let on = parse(&["--enable-prefix-caching"]);
    assert!(
        on.prefix_caching_enabled(),
        "asking for the cache without --hermetic must still get it"
    );
}

#[test]
fn hermetic_beside_a_flag_it_closes_is_refused() {
    let err = validate_serve_args(&parse(&["--hermetic", "--enable-prefix-caching"])).unwrap_err();
    assert!(err.contains("--hermetic"), "{err}");
    assert!(err.contains("--enable-prefix-caching"), "{err}");
    assert!(
        err.contains("known-answer test"),
        "the message must say WHY, not just that: {err}"
    );

    let err = validate_serve_args(&parse(&["--hermetic", "--mtp-gate", "auto"])).unwrap_err();
    assert!(err.contains("--mtp-gate auto"), "{err}");
}

#[test]
fn hermetic_beside_an_agreeing_flag_is_accepted() {
    assert!(validate_serve_args(&parse(&["--hermetic", "--mtp-gate", "force"])).is_ok());
    assert!(validate_serve_args(&parse(&["--hermetic"])).is_ok());
}

/// 2026-09-26: A serve override `hermetic = "true"` renders to the bare
/// `--hermetic`, and `hermetic = "false"` renders to nothing.
#[test]
fn the_serve_override_spelling_of_hermetic_round_trips() {
    use crate::recipe::schema::argv_for;
    assert_eq!(
        argv_for("hermetic", "true"),
        Some(vec!["--hermetic".into()])
    );
    assert_eq!(argv_for("hermetic", "false"), None);
    assert!(parse(&["--hermetic"]).hermetic, "the bare flag enables it");
    assert_eq!(parse(&["--hermetic"]).mtp_gate_force(), Some(true));
    assert_eq!(parse(&[]).mtp_gate_force(), None);
}

#[test]
fn both_scheduler_routers_are_accepted_and_the_default_stays_sync() {
    assert!(validate_serve_args(&parse(&["--scheduler-config", "async"])).is_ok());
    assert!(validate_serve_args(&parse(&["--scheduler-config", "sync"])).is_ok());
    assert_eq!(parse(&[]).scheduler_config, "sync", "the default router");
    let err = validate_serve_args(&parse(&["--scheduler-config", "nonsense"])).unwrap_err();
    assert!(err.contains("--scheduler-config"), "{err}");
}

/// 2026-10-10: clap no longer rejects `--dflash --speculative`; without
/// `METRALE_HYBRID_DFLASH` (unset in tests) `validate_serve_args` still does.
#[test]
fn dflash_with_speculative_is_refused_without_the_hybrid_lever() {
    let err = validate_serve_args(&parse(&["--dflash", "--speculative"]))
        .expect_err("one proposer per serve unless hybrid");
    assert!(err.contains("--speculative"), "{err}");
    assert!(err.contains("METRALE_HYBRID_DFLASH"), "{err}");
    assert!(validate_serve_args(&parse(&["--speculative"])).is_ok());
}

/// 2026-10-10: The hybrid rules, with the lever passed in.
#[test]
fn hybrid_lever_flag_rules() {
    let check = |extra: &[&str], hybrid: bool| {
        let mut v = Vec::new();
        check_hybrid_dflash(&mut v, &parse(extra), hybrid);
        v.len()
    };
    assert_eq!(check(&["--dflash", "--speculative"], false), 1);
    assert_eq!(check(&["--dflash"], false), 0);
    assert_eq!(check(&["--speculative"], false), 0);
    assert_eq!(check(&["--dflash", "--speculative"], true), 0);
    assert_eq!(check(&["--dflash"], true), 1);
    assert_eq!(check(&["--speculative"], true), 1);
    assert_eq!(check(&[], true), 1);
    assert_eq!(
        check(
            &["--dflash", "--speculative", "--ssm-rollback-mode", "replay"],
            true
        ),
        1
    );
}
