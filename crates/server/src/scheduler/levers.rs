// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Scheduler levers, resolved from the environment when a run
//! starts and carried on `SchedCtx`; the scheduler's counterpart to
//! `metrale_model_layers::layers::ops::ModelLevers`.
//!
//! The loop watchdog is the one lever that changes mid-run: the TUI's ops
//! REPL toggles it with `/watchdog on|off`, so it is an [`AtomicBool`].
//!
//! Owner: scheduler.
//! Invariants:
//! - A run holds its levers behind an `Arc` (`SchedCtx::levers`), so only
//!   `loop_watchdog` changes after construction.

use std::sync::atomic::{AtomicBool, Ordering};

/// 2026-09-25: Decode / verify / speculation levers for one run.
pub struct SchedLevers {
    // 2026-09-25: Grammar and sampling.
    /// 2026-09-25: Greedy rows may take the GPU argmax instead of the host
    /// masked pick (MTP bootstrap sample; grammar rows at verify).
    /// `METRALE_DISABLE_FAST_GREEDY=1` turns it off.
    pub fast_greedy_grammar: bool,
    /// 2026-09-25: The grammarless fast masked verify pick
    /// (`verify_pipeline_helper::fast_masked`), which also requires
    /// `dflash_masked_verify`. `METRALE_DISABLE_FAST_MASKED=1` turns it off.
    pub fast_masked: bool,
    /// 2026-09-25: Greedy grammarless rows at verify may take the GPU
    /// argmax, whose tie-breaking at near-equal logits can differ from the
    /// host pick. `METRALE_NO_FAST_GREEDY_CHAT=1` turns it off.
    pub fast_greedy_chat: bool,
    /// 2026-09-25: Treat every row as greedy regardless of the request
    /// (`METRALE_FORCE_TEMP_ZERO=1`).
    pub force_temp_zero: bool,

    // 2026-09-25: Tool-call turn termination. Resolved once: their readers
    // run per token, where an environment read would take the process-wide
    // environment lock.
    /// 2026-09-25: End the turn when the model samples the `<tool_response>`
    /// control token. `METRALE_TOOL_RESPONSE_STOP=0`/`false` disables.
    pub tool_response_stop: bool,
    /// 2026-09-25: Once a tool call has completed, and the row is not
    /// inside a tool body or thinking, lift the grammar's EOS suppression so
    /// the model's EOS ends the turn. `METRALE_TOOL_EOS_ESCAPE=0`/`false`
    /// disables.
    pub tool_eos_escape: bool,
    /// 2026-09-25: `sample_step::effective_min_p` passes the row's min_p;
    /// `METRALE_NO_MTP_MINP=1` makes it pass 0.0.
    pub mtp_minp: bool,
    /// 2026-09-25: At temperature > 0 a verify position samples from the
    /// processed logits instead of taking their argmax.
    /// `METRALE_NO_MTP_VERIFY_SAMPLE=1` turns it off.
    pub mtp_verify_sample: bool,

    // 2026-09-25: DFlash speculation.
    /// 2026-09-25: Append the accepted verify rows to the DFlash drafter
    /// context: `dflash_eagle_kgamma_append` in `verify_dflash_step.rs`
    /// (when `dflash_unified_ctx` is off) and `dflash_eagle_accept_append`
    /// on the K=2 accept in `verify_k2_step.rs`. `METRALE_DFLASH_EAGLE_FIX=0`
    /// turns it off.
    pub dflash_eagle_fix: bool,
    /// 2026-09-25: `METRALE_DFLASH_STEP_TIMING=1`: time a DFlash step's
    /// verify and propose separately.
    pub dflash_step_timing: bool,
    /// 2026-09-25: `METRALE_VISION_TIMING` (presence): synchronize after
    /// each prefill chunk and log its wall time.
    pub vision_timing: bool,
    pub dflash_masked_verify: bool,
    pub dflash_seam_serial: bool,
    pub dflash_adaptive: bool,
    pub dflash_serial_append: bool,
    pub dflash_unified_ctx: bool,
    pub dflash_spec_think: bool,
    /// 2026-09-25: Pin the MTP throughput gate to the verify arm for DFlash
    /// at `active.len() <= 2` (`METRALE_DFLASH_GATE_PIN_C2=0` turns it
    /// off). Measured 2026-08-19 (qwen3.8-27B+DFlash2, C=2):
    /// arbitration was par on tok/s (25.4 vs 24.4) but its serial/batch-K
    /// forward flips fork the temp-0 token stream mid-answer, and the bad
    /// attractor degenerates into repetition (content-loop watchdog kills,
    /// 300-cap rambles); pinned verify holds C1-parity accept (75% vs 38%)
    /// and completions EOS normally. At C>=3 arbitration wins instead —
    /// per-seq serial verify genuinely loses there (22.0 vs 28.1 tok/s at
    /// C=4).
    pub dflash_gate_pin_c2: bool,
    /// 2026-09-25: Cross-sequence batched DFlash verify when two or more
    /// rows verify (also needs `mtp_batch_verify`).
    /// `METRALE_DFLASH_BATCH_VERIFY=0` forces the per-sequence loop.
    pub dflash_batch_verify: bool,
    /// 2026-09-25: Mean accepted drafts below which adaptive speculation
    /// suspends.
    pub dflash_adaptive_min: f32,
    /// 2026-09-25: Serially-decoded tokens between adaptive re-probes.
    pub dflash_adaptive_reprobe: u32,
    /// 2026-09-25: `METRALE_DFLASH_RESUME_GUARD=N` (0 = off): the number of
    /// post-`</think>` tokens kept on serial decode.
    pub dflash_resume_guard: u32,
    /// 2026-09-25: `METRALE_MTP_SHADOW_TOPK` — the verify side of the
    /// drafter top-k probe. Parsed by
    /// `metrale_model_layers::speculative::shadow_topk`, the SSOT.
    pub shadow_topk: usize,

    // 2026-09-25: Watchdogs.
    /// 2026-09-25: `METRALE_DISABLE_WATCHDOGS=1`/`true`: disarm the
    /// generation watchdogs that check it.
    pub disable_watchdogs: bool,
    /// 2026-09-25: Parsed from `METRALE_EOS_SUPPRESS_THINKING=1` but read
    /// nowhere; EOS suppression inside thinking is
    /// `helpers::eos_suppressed_by_thinking`.
    pub eos_suppressed_by_thinking: bool,
    /// 2026-09-25: The forced-token fast path (`logit_processors::forced_token`).
    /// `METRALE_DISABLE_FORCED_TOKEN=1`/`true` turns it off.
    pub forced_token_fastpath: bool,

    // 2026-09-25: Diagnostics and the MTP gate override.
    pub decode_timing: bool,
    pub mtp_timing: bool,
    pub mtp_gate_force: bool,
    pub adadec_diagnostic: bool,

    // 2026-09-25: Loop-shape levers.
    /// 2026-09-25: `METRALE_HOLO_ALWAYS_MIXED=1|true`: an active decode and
    /// an in-progress prefill always take a fused mixed step.
    pub holo_always_mixed: bool,
    /// 2026-09-25: `--prefill-codispatch` / `METRALE_PREFILL_CODISPATCH`,
    /// as the model layer resolved it (`prefill_codispatch_enabled`).
    pub prefill_codispatch: bool,
    /// 2026-09-25: `METRALE_PREFILL_VARLEN`, as the model layer resolved it
    /// (`prefill_varlen_enabled`).
    pub prefill_varlen: bool,
    /// 2026-09-25: `METRALE_PREFILL_CODISPATCH_WINDOW_MS` (default 100).
    pub codispatch_window_ms: u64,
    /// 2026-09-25: `METRALE_PREFILL_CODISPATCH_SETTLE_MS` (default 10).
    pub codispatch_settle_ms: u64,
    /// 2026-09-25: `METRALE_VISION_CODISPATCH=1|true` (default off).
    pub vision_codispatch: bool,
    /// 2026-09-25: `METRALE_BEAM_CODISPATCH` (default on; `0`/`false`
    /// disables).
    pub beam_codispatch: bool,
    /// 2026-09-25: `METRALE_BISECT_Q12_DISABLE=1|true`: turn off the
    /// batched prefill dispatch in `phase_continue_prefills`.
    pub bisect_q12_disable: bool,
    /// 2026-09-25: `METRALE_BISECT_NO_MIX=1|true`: never fuse prefill with
    /// decode in `run_standard`.
    pub bisect_no_mix: bool,
    /// 2026-09-25: `METRALE_MIXED_SLICE_TOKENS`: a cap on the mixed step's
    /// prefill slice (0 = the policy's budget).
    pub mixed_slice_tokens: usize,
    /// 2026-09-25: `METRALE_GRAMMAR_BUDGET_CLOSE` (default on; `0`/`false`
    /// disables).
    pub grammar_budget_close: bool,
    /// 2026-09-25: Rows that ended thinking may take the GPU argmax readback
    /// when `think_ended_gpu_ok` allows it; `METRALE_NO_THINKENDED_GPU_ARGMAX=1`
    /// turns it off.
    pub think_ended_gpu_argmax: bool,
    /// 2026-09-25: `METRALE_PARALLEL_SAMPLE` (default on; `0` disables).
    pub parallel_sample: bool,
    /// 2026-09-25: Batched MTP bootstrap; kill switch
    /// `METRALE_NO_MTP_BATCH_BOOTSTRAP` (presence).
    pub mtp_batch_bootstrap: bool,
    /// 2026-09-25: Batched bootstrap argmax; kill switch
    /// `METRALE_NO_MTP_BOOT_ARGMAX` (presence).
    pub mtp_boot_argmax: bool,
    /// 2026-09-25: Batched K-row verify; kill switch
    /// `METRALE_NO_MTP_BATCH_VERIFY` (presence).
    pub mtp_batch_verify: bool,
    /// 2026-09-25: Batched propose; kill switch
    /// `METRALE_NO_MTP_BATCH_PROPOSE` (presence).
    pub mtp_batch_propose: bool,
    /// 2026-09-25: D-Cut pruning; kill switch `METRALE_NO_MTP_DCUT`
    /// (presence).
    pub dcut_enabled: bool,
    /// 2026-09-25: `METRALE_MTP_DCUT_MAX_SEQS` (default 8): the widest
    /// verify batch D-Cut prunes.
    pub dcut_width_cap: usize,
    /// 2026-09-25: `METRALE_MTP_DCUT_RATIO` snapped to the nearest D-Cut
    /// bucket (default 0.75).
    pub dcut_ratio: f32,
    /// 2026-09-25: `METRALE_MTP_ACCEPT_FOLD_AT_16` (presence): batch widths
    /// above 16 share the accept-telemetry bucket of width 16.
    pub mtp_accept_fold_at_16: bool,
    /// 2026-09-25: `METRALE_MTP_ACCEPT_DEBUG`: the accept-telemetry log lines.
    pub mtp_accept_debug: bool,
    /// 2026-09-25: `METRALE_MTP_MAX_SEQS`: the speculation width cap, as
    /// the model layer resolved it (the model factory sizes the MTP KV pool
    /// with the same value).
    pub mtp_max_seqs: usize,
    /// 2026-09-25: `METRALE_SPEC_ENTRY_PIN` (default 8): while a batch row
    /// has emitted fewer post-`</think>` tokens than this, the MTP gate is
    /// pinned to verify (`mtp_gate::entry_pin_forces_verify`).
    pub spec_entry_pin_tokens: u32,
    /// 2026-09-25: `METRALE_SSM_TAIL_CKPT=1`, as the runtime resolved it.
    pub ssm_tail_ckpt: bool,
    /// 2026-09-25: `--no-ssm-tail-midchunk` / `METRALE_SSM_TAIL_MIDCHUNK`,
    /// as the runtime resolved it (`ssm_tail_midchunk_enabled`).
    pub ssm_tail_midchunk: bool,

    /// 2026-09-25: Loop watchdog; the TUI ops REPL toggles it while
    /// serving.
    loop_watchdog: AtomicBool,
}

/// 2026-09-25: `METRALE_FOO=1` enables.
fn opt_in(var: &str) -> bool {
    metrale_config::levers::var(var).as_deref() == Some("1")
}

/// 2026-09-25: `METRALE_FOO=0` disables: a default-on lever whose off
/// switch is an explicit zero. Not interchangeable with [`on_unless`]:
/// swapping them inverts the switch, so `=0` would leave the lever on and
/// `=1` would turn it off.
fn on_unless_zero(var: &str) -> bool {
    metrale_config::levers::var(var).as_deref() != Some("0")
}

/// 2026-09-25: `METRALE_FOO=1` disables — the flag names a negative, the
/// field stores the positive, so the inversion happens here instead of at
/// every read site.
fn on_unless(var: &str) -> bool {
    metrale_config::levers::var(var).as_deref() != Some("1")
}

/// 2026-09-25: A numeric tunable: the parsed value, or `default` when
/// unset or unparsable.
fn num<T: std::str::FromStr>(var: &str, default: T) -> T {
    metrale_config::levers::var(var)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 2026-09-25: Presence-gated: any value enables, including `0`.
fn present(var: &str) -> bool {
    metrale_config::levers::var(var).is_some()
}

/// 2026-09-25: The `--mtp-gate force` decision in force: the flag when it
/// was given (`Some`), the `METRALE_MTP_GATE_FORCE` variable otherwise.
///
/// `None` means the flag was not given, which keeps the variable reachable;
/// an explicit `--mtp-gate auto` (`Some(false)`) overrides it.
/// `SchedLevers::from_env` and the startup log (`serve_flags.rs`) both call
/// it, so the log prints the value in force.
pub fn resolve_mtp_gate_force(cli: Option<bool>) -> bool {
    cli.unwrap_or_else(|| opt_in("METRALE_MTP_GATE_FORCE"))
}

/// 2026-09-25: `METRALE_FOO=1|true` enables (the loop-shape levers' parse).
fn opt_in_word(var: &str) -> bool {
    metrale_config::levers::var(var).is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// 2026-09-25: The bisect levers' parse: `1`, or `true` after a Unicode
/// lowercase. Kept distinct from [`opt_in_word`] so neither lever changes
/// what it accepts.
fn opt_in_lowercase(var: &str) -> bool {
    metrale_config::levers::var(var).is_some_and(|v| v == "1" || v.to_lowercase() == "true")
}

impl SchedLevers {
    /// 2026-09-25: Resolve from the environment; serve calls it once per
    /// run. `mtp_gate_force_cli` is the command line's `--mtp-gate` as
    /// `ServeArgs::mtp_gate_force` resolved it.
    pub fn from_env(mtp_gate_force_cli: Option<bool>) -> Self {
        Self {
            fast_greedy_grammar: on_unless("METRALE_DISABLE_FAST_GREEDY"),
            fast_masked: on_unless("METRALE_DISABLE_FAST_MASKED"),
            fast_greedy_chat: on_unless("METRALE_NO_FAST_GREEDY_CHAT"),
            force_temp_zero: opt_in("METRALE_FORCE_TEMP_ZERO"),
            // 2026-09-25: `parse_flag_default_on` turns these off on "0" or
            // "false" (trimmed, any case); a bare `!= "1"` test would
            // ignore `=false`.
            tool_response_stop: crate::scheduler::helpers::parse_flag_default_on(
                metrale_config::levers::var("METRALE_TOOL_RESPONSE_STOP").as_deref(),
            ),
            tool_eos_escape: crate::scheduler::helpers::parse_flag_default_on(
                metrale_config::levers::var("METRALE_TOOL_EOS_ESCAPE").as_deref(),
            ),
            mtp_minp: on_unless("METRALE_NO_MTP_MINP"),
            mtp_verify_sample: on_unless("METRALE_NO_MTP_VERIFY_SAMPLE"),

            dflash_eagle_fix: on_unless_zero("METRALE_DFLASH_EAGLE_FIX"),
            dflash_step_timing: opt_in("METRALE_DFLASH_STEP_TIMING"),
            vision_timing: present("METRALE_VISION_TIMING"),
            dflash_masked_verify: on_unless_zero("METRALE_DFLASH_MASKED_VERIFY"),
            dflash_seam_serial: on_unless_zero("METRALE_DFLASH_SEAM_SERIAL"),
            dflash_adaptive: opt_in("METRALE_DFLASH_ADAPTIVE"),
            dflash_serial_append: opt_in("METRALE_DFLASH_SERIAL_APPEND"),
            dflash_unified_ctx: on_unless_zero("METRALE_DFLASH_UNIFIED_CTX"),
            // 2026-09-25: Opt-in. `mtp_gate::spec_dispatch_eligible` reads it
            // on both lanes (`if inside_thinking && !spec_think { return
            // false; }`), so turning it on lets plain MTP speculate inside
            // `<think>` too.
            dflash_spec_think: opt_in("METRALE_DFLASH_SPEC_THINK"),
            dflash_gate_pin_c2: on_unless_zero("METRALE_DFLASH_GATE_PIN_C2"),
            dflash_batch_verify: on_unless_zero("METRALE_DFLASH_BATCH_VERIFY"),
            dflash_adaptive_min: num("METRALE_DFLASH_ADAPTIVE_MIN", 2.0),
            dflash_adaptive_reprobe: num("METRALE_DFLASH_ADAPTIVE_REPROBE", 256),
            dflash_resume_guard: num("METRALE_DFLASH_RESUME_GUARD", 0),
            shadow_topk: metrale_model_layers::speculative::shadow_topk(),

            // 2026-09-25: The `helpers` parsers accept "1" or "true"
            // (trimmed, any case); a bare `== "1"` test would ignore
            // `=true`.
            disable_watchdogs: crate::scheduler::helpers::parse_disable_watchdogs(
                metrale_config::levers::var("METRALE_DISABLE_WATCHDOGS").as_deref(),
            ),
            eos_suppressed_by_thinking: opt_in("METRALE_EOS_SUPPRESS_THINKING"),
            forced_token_fastpath: crate::scheduler::helpers::parse_forced_token_fastpath(
                metrale_config::levers::var("METRALE_DISABLE_FORCED_TOKEN").as_deref(),
            ),

            // 2026-09-25: Presence-gated, not value-gated.
            decode_timing: present("METRALE_DECODE_TIMING"),
            mtp_timing: opt_in("METRALE_MTP_TIMING"),
            // 2026-09-25: `--mtp-gate`, falling back to
            // `METRALE_MTP_GATE_FORCE` when the flag is absent.
            mtp_gate_force: resolve_mtp_gate_force(mtp_gate_force_cli),
            adadec_diagnostic: present("METRALE_ADADEC_DIAGNOSTIC"),

            holo_always_mixed: opt_in_word("METRALE_HOLO_ALWAYS_MIXED"),
            prefill_codispatch: metrale_model_layers::layers::ops::prefill_codispatch_enabled(),
            prefill_varlen: metrale_model_layers::layers::ops::prefill_varlen_enabled(),
            codispatch_window_ms: num("METRALE_PREFILL_CODISPATCH_WINDOW_MS", 100),
            codispatch_settle_ms: num("METRALE_PREFILL_CODISPATCH_SETTLE_MS", 10),
            vision_codispatch: opt_in_word("METRALE_VISION_CODISPATCH"),
            beam_codispatch: metrale_config::levers::var("METRALE_BEAM_CODISPATCH")
                .is_none_or(|v| v != "0" && !v.eq_ignore_ascii_case("false")),
            bisect_q12_disable: opt_in_lowercase("METRALE_BISECT_Q12_DISABLE"),
            bisect_no_mix: opt_in_lowercase("METRALE_BISECT_NO_MIX"),
            mixed_slice_tokens: num("METRALE_MIXED_SLICE_TOKENS", 0),
            grammar_budget_close: crate::scheduler::helpers::parse_flag_default_on(
                metrale_config::levers::var("METRALE_GRAMMAR_BUDGET_CLOSE").as_deref(),
            ),
            think_ended_gpu_argmax: on_unless("METRALE_NO_THINKENDED_GPU_ARGMAX"),
            parallel_sample: on_unless_zero("METRALE_PARALLEL_SAMPLE"),
            mtp_batch_bootstrap: !present("METRALE_NO_MTP_BATCH_BOOTSTRAP"),
            mtp_boot_argmax: !present("METRALE_NO_MTP_BOOT_ARGMAX"),
            mtp_batch_verify: !present("METRALE_NO_MTP_BATCH_VERIFY"),
            mtp_batch_propose: !present("METRALE_NO_MTP_BATCH_PROPOSE"),
            dcut_enabled: !present("METRALE_NO_MTP_DCUT"),
            dcut_width_cap: num("METRALE_MTP_DCUT_MAX_SEQS", 8),
            dcut_ratio: crate::scheduler::mtp_dcut::dcut_ratio_from_env(),
            mtp_accept_fold_at_16: present("METRALE_MTP_ACCEPT_FOLD_AT_16"),
            mtp_accept_debug: metrale_model_layers::speculative::mtp_accept_debug(),
            mtp_max_seqs: metrale_model_layers::speculative::mtp_max_seqs(),
            spec_entry_pin_tokens: metrale_speculative::mtp_gate::entry_pin_tokens_from_env(),
            ssm_tail_ckpt: metrale_gpu_runtime::ssm_tail_ckpt_enabled(),
            ssm_tail_midchunk: metrale_gpu_runtime::ssm_tail_midchunk_enabled(),

            loop_watchdog: AtomicBool::new(false),
        }
    }

    /// 2026-09-25: Lever values for tests, without reading the environment.
    /// They match `from_env` with no `METRALE_*` set except
    /// `dflash_masked_verify` and `dflash_seam_serial`, which are off here
    /// and on in `from_env`.
    pub fn defaults() -> Self {
        Self {
            fast_greedy_grammar: true,
            fast_masked: true,
            fast_greedy_chat: true,
            force_temp_zero: false,
            tool_response_stop: true,
            tool_eos_escape: true,
            mtp_minp: true,
            mtp_verify_sample: true,
            dflash_eagle_fix: true,
            dflash_step_timing: false,
            vision_timing: false,
            dflash_masked_verify: false,
            dflash_seam_serial: false,
            dflash_adaptive: false,
            dflash_serial_append: false,
            dflash_unified_ctx: true,
            dflash_spec_think: false,
            dflash_gate_pin_c2: true,
            dflash_batch_verify: true,
            dflash_adaptive_min: 2.0,
            dflash_adaptive_reprobe: 256,
            dflash_resume_guard: 0,
            shadow_topk: 0,
            disable_watchdogs: false,
            eos_suppressed_by_thinking: false,
            forced_token_fastpath: true,
            decode_timing: false,
            mtp_timing: false,
            mtp_gate_force: false,
            adadec_diagnostic: false,
            holo_always_mixed: false,
            prefill_codispatch: false,
            prefill_varlen: false,
            codispatch_window_ms: 100,
            codispatch_settle_ms: 10,
            vision_codispatch: false,
            beam_codispatch: true,
            bisect_q12_disable: false,
            bisect_no_mix: false,
            mixed_slice_tokens: 0,
            grammar_budget_close: true,
            think_ended_gpu_argmax: true,
            parallel_sample: true,
            mtp_batch_bootstrap: true,
            mtp_boot_argmax: true,
            mtp_batch_verify: true,
            mtp_batch_propose: true,
            dcut_enabled: true,
            dcut_width_cap: 8,
            dcut_ratio: 0.75,
            mtp_accept_fold_at_16: false,
            mtp_accept_debug: false,
            // 2026-09-25: `mtp_max_seqs()` with the variable unset.
            mtp_max_seqs: 32,
            spec_entry_pin_tokens: 8,
            ssm_tail_ckpt: false,
            ssm_tail_midchunk: true,
            loop_watchdog: AtomicBool::new(false),
        }
    }

    /// 2026-09-25: Is the loop watchdog armed?
    pub fn loop_watchdog(&self) -> bool {
        self.loop_watchdog.load(Ordering::Relaxed)
    }

    /// 2026-09-25: The subset the pre-sample pipeline reads, for `LogitsContext`.
    pub fn sampling(&self) -> crate::scheduler::logit_processors::SamplingLevers {
        crate::scheduler::logit_processors::SamplingLevers {
            force_temp_zero: self.force_temp_zero,
            fast_greedy_grammar: self.fast_greedy_grammar,
            mtp_verify_sample: self.mtp_verify_sample,
            fast_masked: self.fast_masked,
            fast_greedy_chat: self.fast_greedy_chat,
            adadec_diagnostic: self.adadec_diagnostic,
            dflash_masked_verify: self.dflash_masked_verify,
            disable_watchdogs: self.disable_watchdogs,
            forced_token_fastpath: self.forced_token_fastpath,
            mtp_minp: self.mtp_minp,
            think_ended_gpu_argmax: self.think_ended_gpu_argmax,
        }
    }

    /// 2026-09-25: Arm or disarm the loop watchdog (serve startup, and the
    /// TUI ops REPL mid-run).
    pub fn set_loop_watchdog(&self, on: bool) {
        self.loop_watchdog.store(on, Ordering::Relaxed);
    }
}

impl Default for SchedLevers {
    fn default() -> Self {
        Self::defaults()
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
