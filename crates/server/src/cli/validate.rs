// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Validation of `met serve` arguments that clap cannot do alone:
//! flag combinations that contradict each other, and the enumerated values.
//!
//! [`validate_serve_args`] reports every violation in one error, each with what
//! is wrong, why, and the fix. `serve` calls it before the model load, and a
//! model swap calls it before loading the next model.
//!
//! Owner: server CLI.
//! Invariants: `validate_serve_args` returns `Ok` only when no rule fired.

use super::ServeArgs;
// 2026-09-26: The value lists live in `flag_values`, which the dashboard's
// option picker also reads (`options_for_flag`).
use super::flag_values::{
    KvHighPrecisionLayers, LM_HEAD_DTYPES, MTP_GATES, MTP_QUANTS, SCHEDULER_CONFIGS, SCHEDULERS,
    SSM_H_DTYPES, TELEMETRY_LEVELS, TOOL_CALL_PARSERS, TRISTATES,
};

mod violation;
use violation::{Violation, check_enum, format_violations};

/// 2026-09-26: Validate a `met serve` command line.
///
/// # Errors
/// One formatted string listing every violation, so the operator can fix them
/// in one pass.
pub fn validate_serve_args(args: &ServeArgs) -> Result<(), String> {
    let mut v: Vec<Violation> = Vec::new();

    check_enum(
        &mut v,
        "--lm-head-dtype",
        &args.lm_head_dtype,
        LM_HEAD_DTYPES,
    );
    if let Some(dtype) = &args.ssm_h_dtype {
        check_enum(&mut v, "--ssm-h-dtype", dtype, SSM_H_DTYPES);
    }
    // 2026-09-26: Only when given: absent lets `METRALE_MTP_GATE_FORCE` decide.
    if let Some(gate) = &args.mtp_gate {
        check_enum(&mut v, "--mtp-gate", gate, MTP_GATES);
    }
    for (flag, value) in [
        ("--ssm-batched-recurrent", &args.ssm_batched_recurrent),
        ("--content-loop-watchdog", &args.content_loop_watchdog),
        ("--tool-grammar", &args.tool_grammar),
    ] {
        check_enum(&mut v, flag, value, TRISTATES);
    }

    // 2026-09-26: `--hermetic` closes these channels, and the resolvers in
    // `cli::hermetic` would pick hermetic's answer without saying so. Refuse
    // the pair so the command line never names an override that is ignored.
    if args.hermetic && args.enable_prefix_caching {
        v.push(Violation::new(
            "--hermetic with --enable-prefix-caching",
            "--hermetic closes the radix KV prefix cache: it is keyed on token content \
             with no session component, so one request's KV blocks are reachable by any \
             later request sharing a prefix — exactly the cross-request channel a \
             known-answer test must not have",
            "drop --enable-prefix-caching (--hermetic already implies it is off), or drop \
             --hermetic if you meant to measure WITH the cache",
        ));
    }
    if args.hermetic && args.mtp_gate.as_deref() == Some("auto") {
        v.push(Violation::new(
            "--hermetic with --mtp-gate auto",
            "--hermetic pins the MTP gate to `force`, which disarms the arbiter. Under \
             `auto` the gate PROBES: its token counter is cumulative and is never reset \
             at a request boundary, so which request gets served by the serial arm — \
             which is not byte-equal to the batch arm even at temperature 0 — depends on \
             everything served before it",
            "drop --mtp-gate auto (--hermetic already forces the gate), or drop --hermetic",
        ));
    }
    // 2026-09-26: The FP16 h-state twins exist only on the fused-norm decode
    // arm; without it an FP32-only kernel reads the FP16 pool and produces
    // wrong output, not an error. An absent `--gdn-fused-norm` beside
    // `--ssm-h-dtype f16` publishes `fused_norm` off (`KernelFlagPlan`).
    // `h_f16` comes from `ssm_h_dtype_bits`, the decode the published cell
    // uses, so `f16-pool` binds every f16 rule below too.
    let h_f16 =
        metrale_model_layers::layers::qwen3_ssm::ssm_h_dtype_bits(args.ssm_h_dtype.as_deref()).0;
    if h_f16 && !args.gdn_fused_norm {
        v.push(Violation::new(
            "--ssm-h-dtype f16 without --gdn-fused-norm",
            "the FP16 h-state twins exist only on the fused-norm decode arm; the unfused \
             arms (gated_delta_rule_decode, ..._decode_f32_strided) are FP32-only and \
             would read the FP16 pool as FP32 — fluent garbage, not an error",
            "add --gdn-fused-norm, or use --ssm-h-dtype f32",
        ));
    }
    // 2026-09-26: The DFlash verify width is gamma + 1. Above
    // `MAX_F16_TWIN_DFLASH_GAMMA` it dispatches `gated_delta_rule_wy17`, which
    // has no FP16 h-state twin. Only an explicit `--dflash-gamma` is checked:
    // an unset gamma resolves through `default_dflash_gamma`, which clamps to
    // the same bound.
    if h_f16
        && args.dflash
        && args
            .dflash_gamma
            .is_some_and(|g| g > metrale_model_layers::layers::qwen3_ssm::MAX_F16_TWIN_DFLASH_GAMMA)
    {
        v.push(Violation::new(
            "--dflash with a verify width above the FP16 twin range, together with \
             --ssm-h-dtype f16",
            "DFlash verify widths above 16 dispatch gated_delta_rule_wy17, which has \
             no FP16 h-state twin — an FP32 kernel over an FP16 h-state emits fluent \
             garbage. Widths 5..16 are covered by the wyN _f16 twins",
            "use --dflash-gamma <= 15 (verify width 16), drop --dflash, or use \
             --ssm-h-dtype f32",
        ));
    }

    // 2026-09-26: The same `FromStr` that `publish_kernel_flags` publishes
    // through.
    if let Err(why) = args
        .ssm_rollback_mode
        .parse::<metrale_model_layers::ssm_reserve::SsmRollbackMode>()
    {
        v.push(Violation::new(
            format!(
                "--ssm-rollback-mode '{}' is not a valid value.",
                args.ssm_rollback_mode
            ),
            why,
            "use snapshot (default, wired) or replay (experimental scaffold).",
        ));
    }
    // 2026-09-26: The same parse that `publish_kernel_flags` publishes through.
    if let Err(why) =
        metrale_model_layers::ssm_reserve::parse_decode_ring_slots(&args.ssm_decode_ring_slots)
    {
        v.push(Violation::new(
            format!(
                "--ssm-decode-ring-slots '{}' is not a valid value.",
                args.ssm_decode_ring_slots
            ),
            why,
            "use auto (default: preflight sizes the ring from free memory) or 0..=8.",
        ));
    }
    // 2026-09-26: An FP16 h-state turns the exact verify chain off
    // (`GdnFlags::verify_exact_active`), so the pair would drop an explicit
    // `--exact-verify` without saying so.
    if args.exact_verify && h_f16 {
        v.push(Violation::new(
            "--exact-verify together with --ssm-h-dtype f16",
            "the exact MTP-verify chain (issue #435) runs FP32-reader kernels and must \
             never read the FP16 h-state pool, so with f16 the exact request would be \
             silently dropped and spec-on output would NOT equal spec-off",
            "drop --ssm-h-dtype f16 (f32 is the default), or drop --exact-verify",
        ));
    }
    check_enum(
        &mut v,
        "--mtp-quantization",
        &args.mtp_quantization,
        MTP_QUANTS,
    );
    check_enum(&mut v, "--scheduler", &args.scheduler, SCHEDULERS);
    check_enum(
        &mut v,
        "--scheduler-config",
        &args.scheduler_config,
        SCHEDULER_CONFIGS,
    );
    check_enum(&mut v, "--telemetry", &args.telemetry, TELEMETRY_LEVELS);
    if let Some(parser) = args.tool_call_parser.as_deref() {
        check_enum(&mut v, "--tool-call-parser", parser, TOOL_CALL_PARSERS);
    }
    // 2026-09-26: Parsed with `KvCacheDtype`'s own `FromStr`. Only an explicit
    // value can be checked here: an omitted flag resolves later against
    // MODEL.toml (`resolve_kv_dtype_str`). The build embeds MODEL.toml's
    // `default_kv_dtype` unchecked (`build_parse_behavior.rs`), so a bad value
    // there fails only at the same parse in `serve_phases/kv_cache.rs`, during
    // the model load.
    if let Some(kv_dtype) = args.kv_cache_dtype.as_deref()
        && kv_dtype
            .parse::<metrale_cache::kv_cache::KvCacheDtype>()
            .is_err()
    {
        v.push(Violation::new(
            format!("--kv-cache-dtype '{kv_dtype}' is not a known KV cache dtype."),
            "the value does not parse to any supported KV cache format.",
            "use one of: fp8, bf16, nvfp4 (or a turbo* TurboQuant-Plus variant).",
        ));
    }

    // 2026-09-26: The frozen FP8 KV scale is amax × headroom / 448
    // (`fp8_calibration.rs`), so a headroom below 1.0 clips the values it was
    // measured from.
    if args.fp8_kv_headroom < 1.0 {
        v.push(Violation::new(
            format!("--fp8-kv-headroom {} is below 1.0.", args.fp8_kv_headroom),
            "the frozen FP8 KV scale covers headroom× the calibration-window absmax; \
             a multiplier under 1.0 clips the very values it was measured from.",
            "use a value ≥ 1.0 (default 2.0).",
        ));
    }

    // 2026-09-26: Only `fp8` runs online FP8 KV calibration. Both flags must be
    // explicit: an omitted --kv-cache-dtype resolves against MODEL.toml later.
    if let (Some(calib), Some(kv_dtype)) = (
        args.fp8_kv_calibration_tokens,
        args.kv_cache_dtype.as_deref(),
    ) && calib > 0
        && kv_dtype != "fp8"
    {
        v.push(Violation::new(
            format!(
                "--fp8-kv-calibration-tokens {calib} has no effect with --kv-cache-dtype {kv_dtype}.",
            ),
            "online FP8 KV-scale calibration only feeds an FP8 KV cache; with a \
             bf16/nvfp4 cache the calibrated scales are never read.",
            "set --kv-cache-dtype fp8, or drop --fp8-kv-calibration-tokens (0 = off).",
        ));
    }
    // 2026-09-26: --kv-high-precision-layers over a bf16 cache is a no-op, and
    // is accepted: recipes pass `kv_cache_dtype: bf16` with
    // `kv_high_precision_layers: auto` (e.g. the fixture
    // `qwen3.6-35b-a3b-fp8-mtp.yaml`).

    if args.require_auth && args.auth_tokens_file.is_none() && args.auth_token.is_none() {
        v.push(Violation::new(
            "--require-auth is set but no bearer tokens were provided.",
            "with auth enforced and no tokens loaded, EVERY request is rejected 401.",
            "pass --auth-tokens-file <path> (preferred, 0600) or --auth-token <token>.",
        ));
    }

    // 2026-09-26: Only an explicit --num-drafts is checked: an omitted one
    // resolves against MODEL.toml later, and a model default without a
    // speculative method is not a user error.
    let any_spec = args.speculative || args.self_speculative || args.ngram_speculative;
    if let Some(num_drafts) = args.num_drafts
        && num_drafts > 1
        && !any_spec
    {
        // 2026-09-26: Under --dflash the draft count is the drafter's gamma
        // minus 1 (`serve_load`), so the flag is ignored there too; only the
        // remedy differs.
        if args.dflash {
            v.push(Violation::new(
                format!("--num-drafts {num_drafts} is ignored under --dflash."),
                "a DFlash serve drafts at the drafter checkpoint's trained block size \
                 (γ); the scheduler overrides --num-drafts with γ - 1.",
                "drop --num-drafts, or use --dflash-gamma to override the drafter's γ \
                 (block-diffusion drafters are trained at ONE block size — expect \
                 acceptance collapse away from it).",
            ));
        } else {
            v.push(Violation::new(
                format!("--num-drafts {num_drafts} is set but no speculative method is enabled.",),
                "the draft count only applies when speculative decoding proposes drafts; \
                 without it the flag is ignored.",
                "add --speculative (MTP), --self-speculative, or --ngram-speculative — or \
                 drop --num-drafts.",
            ));
        }
    }

    // 2026-10-10: `--dflash` with `--speculative` (clap's conflict until 2026-10-10) only on a
    // hybrid serve, which needs both (`check_hybrid_dflash`).
    check_hybrid_dflash(
        &mut v,
        args,
        metrale_model_layers::speculative::hybrid::hybrid_dflash_enabled(),
    );

    // 2026-10-10: `METRALE_SSM_NO_CHECKPOINT=1` allocates no pre-verify SSM checkpoint; ngram
    // and self-speculative verify roll back to it (`num_accepted == 0`), MTP never does.
    if metrale_model_layers::ssm_reserve::ssm_no_checkpoint()
        && (args.self_speculative || args.ngram_speculative)
    {
        v.push(Violation::new(
            "METRALE_SSM_NO_CHECKPOINT=1 is set with --self-speculative / --ngram-speculative.",
            "those paths restore the per-slot pre-verify SSM checkpoint, which the lever does \
             not allocate; only the MTP path never reads it.",
            "unset METRALE_SSM_NO_CHECKPOINT, or serve with --speculative (MTP).",
        ));
    }

    if args.disable_thinking && args.max_thinking_budget.is_some() {
        v.push(Violation::new(
            "--max-thinking-budget is set together with --disable-thinking.",
            "--disable-thinking strips reasoning entirely, so there is nothing for the \
             budget to cap.",
            "drop one: keep --disable-thinking for no reasoning, or drop it and keep the \
             budget to bound reasoning length.",
        ));
    }

    if args.rank >= args.world_size {
        v.push(Violation::new(
            format!(
                "--rank {} is out of range for --world-size {}.",
                args.rank, args.world_size
            ),
            "ranks are 0-indexed, so a valid rank is in 0..world_size.",
            format!(
                "set --rank in 0..={} (or raise --world-size).",
                args.world_size.saturating_sub(1)
            ),
        ));
    }
    if args.ep_size > args.world_size {
        v.push(Violation::new(
            format!(
                "--ep-size {} exceeds --world-size {}.",
                args.ep_size, args.world_size
            ),
            "expert parallelism cannot span more ranks than exist.",
            "raise --world-size to at least --ep-size, or lower --ep-size.",
        ));
    }
    if args.tp_size > args.world_size {
        v.push(Violation::new(
            format!(
                "--tp-size {} exceeds --world-size {}.",
                args.tp_size, args.world_size
            ),
            "tensor parallelism cannot span more ranks than exist.",
            "raise --world-size to at least --tp-size, or lower --tp-size.",
        ));
    }

    if !args.high_speed_swap {
        let mut orphaned: Vec<&str> = Vec::new();
        if args.high_speed_swap_dir.is_some() {
            orphaned.push("--high-speed-swap-dir");
        }
        if args.high_speed_swap_gb.is_some() {
            orphaned.push("--high-speed-swap-gb");
        }
        if args.high_speed_swap_resident_blocks.is_some() {
            orphaned.push("--high-speed-swap-resident-blocks");
        }
        if args.no_high_speed_swap_graph {
            orphaned.push("--no-high-speed-swap-graph");
        }
        if !orphaned.is_empty() {
            v.push(Violation::new(
                format!("{} set without --high-speed-swap.", orphaned.join(", ")),
                "high-speed-swap tuning options are ignored unless the feature is on.",
                "add --high-speed-swap, or drop the tuning option(s).",
            ));
        }
    }

    if !(args.gpu_memory_utilization > 0.0 && args.gpu_memory_utilization <= 1.0) {
        v.push(Violation::new(
            format!(
                "--gpu-memory-utilization {} is outside (0.0, 1.0].",
                args.gpu_memory_utilization
            ),
            "the value is the fraction of total GPU memory Metrale Engine may claim.",
            "use a fraction in (0.0, 1.0], e.g. 0.90.",
        ));
    }

    // 2026-09-26: The same `KvHighPrecisionLayers` parse the resolve site in
    // `serve_phases/kv_cache.rs` uses, run here before the weight load.
    if let Err(why) = args
        .kv_high_precision_layers
        .parse::<KvHighPrecisionLayers>()
    {
        v.push(Violation::new(
            format!(
                "--kv-high-precision-layers '{}' is not a valid value.",
                args.kv_high_precision_layers
            ),
            why,
            "use a layer count (0 = let the KV dtype decide), auto (2, recommended), \
             or max/all (every attention layer).",
        ));
    }

    // 2026-09-26: Nothing else reads `args.warmup_prompt`. The flag stays on the
    // CLI so that a copied command line gets this explanation rather than
    // clap's "unexpected argument".
    if args.warmup_prompt.is_some() {
        v.push(Violation::new(
            "--warmup-prompt is not implemented.",
            "nothing reads it: no prompt is tokenized, no prefill runs and nothing enters \
             the prefix cache, so the cold-start TTFT its help text promises to remove is \
             still paid on the first real request.",
            "drop --warmup-prompt, and warm the server yourself by sending one throwaway \
             request after startup — that populates the prefix cache through the same path \
             a real request does.",
        ));
    }

    if v.is_empty() {
        return Ok(());
    }
    Err(format_violations(&v))
}

#[cfg(test)]
#[path = "validate_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "validate_tests_b.rs"]
mod validate_tests_b;

/// 2026-10-10: The flag rules of the hybrid serve (`METRALE_HYBRID_DFLASH=1`, `hybrid`). Without
/// the lever `--dflash` and `--speculative` stay mutually exclusive, as clap enforced before.
/// With it both are required, and the SSM rollback mode must be the snapshot mode (the DFlash
/// slot's verify pools are per-token intermediates).
fn check_hybrid_dflash(v: &mut Vec<Violation>, args: &ServeArgs, hybrid: bool) {
    if !hybrid {
        if args.dflash && args.speculative {
            v.push(Violation::new(
                "--dflash is set together with --speculative.",
                "a serve holds one proposer: the DFlash drafter or the MTP head.",
                "drop one of the two, or set METRALE_HYBRID_DFLASH=1 (on every rank) to hold \
                 both and pick per step.",
            ));
        }
        return;
    }
    if !(args.dflash && args.speculative) {
        v.push(Violation::new(
            "METRALE_HYBRID_DFLASH=1 is set without both --dflash and --speculative.",
            "the hybrid serve holds the MTP head (--speculative) and the DFlash drafter \
             (--dflash) and picks one per step.",
            "add --dflash --speculative, or unset METRALE_HYBRID_DFLASH.",
        ));
    }
    if args
        .ssm_rollback_mode
        .parse::<metrale_model_layers::ssm_reserve::SsmRollbackMode>()
        .is_ok_and(|m| m == metrale_model_layers::ssm_reserve::SsmRollbackMode::Replay)
    {
        v.push(Violation::new(
            "METRALE_HYBRID_DFLASH=1 is set with a non-snapshot --ssm-rollback-mode.",
            "the hybrid serve sizes the DFlash slot's per-token SSM intermediates, which only \
             the snapshot rollback mode allocates.",
            "drop --ssm-rollback-mode, or unset METRALE_HYBRID_DFLASH.",
        ));
    }
    if std::env::var("METRALE_DFLASH_UNIFIED_CTX").as_deref() == Ok("0") {
        v.push(Violation::new(
            "METRALE_HYBRID_DFLASH=1 is set with METRALE_DFLASH_UNIFIED_CTX=0.",
            "the hybrid serve appends the DFlash drafter context only through the unified \
             commit (`commit_ctx`) on every step type; the older append paths are off on it.",
            "unset METRALE_DFLASH_UNIFIED_CTX, or unset METRALE_HYBRID_DFLASH.",
        ));
    }
}
