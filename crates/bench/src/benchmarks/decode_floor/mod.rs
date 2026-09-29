// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Decode Floor Gate: the pinned single-user decode throughput
//! floor. Every generation knob is pinned so that two runs are comparable, and
//! the one metric, the median server decode rate over `RUNS` runs, is judged
//! against the served variant's BENCH.toml `server_decode_tok_s` bound under
//! `--pull-request-gate`. `decode-floor` is listed in `gate::coverage::REQUIRED`.
//!
//! The pins (the benchmark's definition, not parameters):
//! - Prompt: `MINHEAP_PROMPT`. A different prompt is a different benchmark.
//! - Request: temperature 0, seed 0, `max_tokens` = `MAX_TOKENS`, and
//!   `reasoning_effort: "none"` in the body, which the server maps to thinking
//!   off, so the gate serve needs no thinking flag. Thinking off matters:
//!   speculative dispatch is off inside `<think>` unless the lane's
//!   spec-in-think lever is on (`spec_dispatch_eligible`; 2026-09-29, A146:
//!   MTP on GLM-5.3 by default, otherwise `METRALE_MTP_SPEC_THINK=1` or
//!   `METRALE_DFLASH_SPEC_THINK=1`), so a thinking-on run could measure
//!   serial decode.
//! - Runs: `RUNS`, with no warmup parameter. The metric is the median
//!   `usage."response_token/s"`.
//!
//! Vacuity pins (`score::evaluate`): the run is inconclusive, rendered as a
//! failing verdict, unless every run emits at least `MIN_OUTPUT_TOKENS`,
//! reports a finite positive server decode rate and a nonzero
//! `usage.completion_tokens_details.accepted_prediction_tokens`, and the mean
//! of the per-run accept lengths (`stats::accept_len`: emitted tokens per
//! decode step, `completion / (completion - accepted)`) is at least
//! `MIN_ACCEPT_LEN`.
//!
//! Owner: bench, decode_floor.
//! Invariants: the run verdict is a pass only for `Evaluation::Measured` with
//! `min_tok_s > 0` and a median at or above it (`verdict_for`).

use crate::hardware::Sensitivity;
use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;

use crate::benchmark::{Benchmark, BenchmarkDescriptor, ModelExpectation};
use crate::hardware::energy_sampler::EnergyMeter;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{
    BenchmarkResult, Cell, CellStyle, Column, LogLine, ResultTable, RunStatus, Stat,
};

const SUMMARY: &str = "Pinned decode-rate floor: 3 fixed runs of one code prompt, \
                       median server decode tok/s vs a committed threshold";
pub const METADATA: PluginMetadata = PluginMetadata::metrale(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "decode-floor",
    name: "Decode Floor Gate",
    summary: SUMMARY,
    detail: "Three timed streaming runs of one committed MinHeap-class code prompt, with every \
             generation knob pinned (temperature 0, seed 0, max_tokens 1500, thinking off via \
             a per-request reasoning_effort \"none\" — no serve flag needed). The metric is the \
             MEDIAN server decode rate (usage.\"response_token/s\"), judged against the \
             BENCH.toml floor under --pull-request-gate. Vacuity pins make the run \
             INCONCLUSIVE rather than PASS when it measured nothing: every run must emit \
             >=750 of the 1500-token budget (the calibrated instrument's natural stop is a \
             deterministic 915), report the server rate, and show accept_len_mean >= 1.5 \
             derived from usage.completion_tokens_details.accepted_prediction_tokens \
             (requires the accept-stats instrumentation; a serve that is not speculating \
             cannot pass this gate's floor honestly). REQUIRED since 2026-08-15; the \
             floor in force is 22.7, from a 10-run set on the gate's own serve (2026-09-06).",
    duration_hint: "~3–6 min",
    expected_secs: 180,
    updated: "2026-08-15",
    // 2026-09-26: The only `gate = "decode-floor"` BENCH.toml entry is for
    // unsloth/Qwen3.8-27B-NVFP4 (`kernels/gb10/qwen3.8-27b/BENCH.toml`); the
    // driver measures whatever it is pointed at.
    intended_for: Some(ModelExpectation {
        families: &["qwen3.8-27b"],
        note: "The decode floor is recorded for unsloth/Qwen3.8-27B-NVFP4 (10-run \
               calibration on the gate's own serve, 2026-09-06). Other checkpoints run fine \
               but have no committed floor to be judged against — a number with no baseline \
               gates nothing.",
    }),
    // 2026-09-26: Under --pull-request-gate, `min_tok_s` is filled from the
    // served variant's `server_decode_tok_s` bound (min minus noise) unless
    // `--param` sets it, so a Measured run gives a pass or fail run verdict
    // (`verdict_for`).
    threshold_params: &[("min_tok_s", "server_decode_tok_s")],
    needs_confirmation: false,
    // 2026-09-26: The metric is a rate: a box that throttled during the run
    // reports a floor miss that the code did not cause.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(DecodeFloor::default()),
};

mod score;
pub(crate) use score::{
    Evaluation, MAX_TOKENS, MINHEAP_PROMPT, RUNS, RunObs, evaluate, instrument_metrics, verdict_for,
};

#[derive(Default)]
pub struct DecodeFloor {
    handle: Option<PluginHandle>,
    timeout: Duration,
    /// 2026-09-26: Verdict floor (tok/s); 0.0 gives an info verdict.
    min_tok_s: f64,
    samples: Vec<RunObs>,
    started: Option<Instant>,
    probed: bool,
    /// 2026-09-26: GPU-rail power sampling: started after the probe (idle
    /// baseline), one window per pinned run.
    energy: EnergyMeter,
}

impl DecodeFloor {
    fn request_body(model: &str) -> serde_json::Value {
        json!({
            "model": model,
            "stream": true,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": MINHEAP_PROMPT}],
        })
    }

    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    async fn one_run(&self) -> Result<http::ChatOutcome> {
        let handle = self.handle()?;
        let target = handle.target();
        let body = Self::request_body(&target.model);
        http::chat_stream(target, &body, self.timeout).await
    }

    fn table(&self) -> ResultTable {
        let mut t = ResultTable::new(
            "DECODE FLOOR",
            vec![
                Column::right("Run", 4),
                Column::right("Out tok", 8),
                Column::right("Decode tok/s (srv)", 18),
                Column::right("Accepted", 9),
                Column::right("E2E ms", 9),
            ],
        );
        for (i, s) in self.samples.iter().enumerate() {
            t.push(vec![
                Cell::new((i + 1).to_string()),
                Cell::new(s.completion_tokens.to_string()),
                Cell::styled(
                    s.server_tps
                        .map(|v| format!("{v:.1}"))
                        .unwrap_or_else(|| "—".into()),
                    CellStyle::Accent,
                ),
                Cell::new(
                    s.accepted_prediction_tokens
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "—".into()),
                ),
                Cell::new(format!("{:.0}", s.e2e_ms)),
            ]);
        }
        t
    }
}

impl Plugin for DecodeFloor {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send {
        self.handle = Some(handle);
        self.started = Some(Instant::now());
        async { Ok(()) }
    }
}

impl Benchmark for DecodeFloor {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        // 2026-09-26: The generation knobs are pins, not parameters (module
        // docs). Only the transport timeout and the verdict floor are tunable.
        vec![
            ParamSpec::new(
                "request_timeout_s",
                "Request timeout",
                "Seconds before a single request is abandoned. Transport-side only — it \
                 cannot change the measured decode rate.",
                ParamKind::Int { min: 30, max: 3600 },
                ParamValue::Int(300),
            ),
            ParamSpec::new(
                "min_tok_s",
                "Decode floor",
                "Run-verdict floor on the median server decode rate. 0 disables (a \
                 standalone run reports an info verdict); under --pull-request-gate this \
                 is auto-filled from the variant's BENCH.toml server_decode_tok_s `min` \
                 bound. Vacuous runs stay INCONCLUSIVE regardless.",
                ParamKind::Float {
                    min: 0.0,
                    max: 10_000.0,
                },
                // 2026-09-26: 0.0 is the documented off state, not an implicit
                // bar (PCND).
                ParamValue::Float(0.0),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        let specs = self.parameters();
        values.validate_against(&specs)?;
        self.timeout = Duration::from_secs(values.usize("request_timeout_s")? as u64);
        self.min_tok_s = values.float("min_tok_s")?;
        self.samples.clear();
        self.probed = false;
        self.energy = EnergyMeter::default();
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        let total = RUNS as u64;

        if !self.probed {
            self.probed = true;
            http::probe(handle.target(), Duration::from_secs(10))
                .await
                .context("endpoint probe failed — check the target URL and port")?;
            // 2026-09-26: The probe answered and no run is in flight: the idle
            // baseline is taken now.
            for line in self.energy.start(handle.target()).await {
                handle.log(line.level, line.text);
            }
            return Ok(BenchmarkResult::running("probe", self.elapsed())
                .with_progress(0, total)
                .log_line(LogLine::info(format!(
                    "{} · MinHeap code prompt · max_tokens {MAX_TOKENS} · temp 0 · seed 0 · \
                     reasoning_effort none · {RUNS} pinned runs",
                    handle.target().base_url
                ))));
        }

        if self.samples.len() < RUNS {
            handle.status(format!("run {}/{RUNS}", self.samples.len() + 1));
            let window_start = Instant::now();
            let outcome = self.one_run().await?;
            let window_end = Instant::now();
            let mut obs = RunObs::from_outcome(&outcome);
            obs.energy = self.energy.window(window_start, window_end);
            let fmt_ms =
                |v: Option<f64>| v.map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
            let mut line = LogLine::info(format!(
                "run {}/{RUNS}: {} tok · decode {} tok/s (server) · itl {} ms server / {} ms \
                 client · accepted {} · E2E {:.0} ms",
                self.samples.len() + 1,
                obs.completion_tokens,
                obs.server_tps
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "—".into()),
                fmt_ms(obs.server_tpot_ms),
                fmt_ms(obs.client_tpot_ms),
                obs.accepted_prediction_tokens
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "—".into()),
                obs.e2e_ms,
            ));
            if let Some(e) = &obs.energy {
                line.text
                    .push_str(&format!(" · {}", e.one_line(self.energy.idle())));
            }
            self.samples.push(obs);
            let done = self.samples.len() as u64;
            handle.progress(done, total);
            return Ok(BenchmarkResult::running("timed", self.elapsed())
                .with_progress(done, total)
                .with_table(self.table())
                .log_line(line));
        }

        if self.samples.iter().all(|s| s.completion_tokens == 0) {
            bail!("no run produced any output token — nothing to measure");
        }

        let sampler_cost_line = self.energy.stop().await;
        let mut metrics = BTreeMap::new();
        metrics.insert("runs".to_string(), self.samples.len() as f64);
        instrument_metrics(&self.samples, self.energy.idle(), &mut metrics);
        self.energy.metrics(&mut metrics);
        let eval = evaluate(&self.samples);
        let verdict = verdict_for(&eval, self.min_tok_s);
        let summary = match eval {
            Evaluation::Inconclusive(_) => Vec::new(),
            Evaluation::Measured {
                median_decode_tok_s,
                min_output_tokens,
                accept_len_mean,
            } => {
                metrics.insert("server_decode_tok_s".to_string(), median_decode_tok_s);
                metrics.insert("output_tokens".to_string(), min_output_tokens as f64);
                metrics.insert("accept_len_mean".to_string(), accept_len_mean);
                vec![
                    Stat::new(
                        "Decode tok/s (server, median)",
                        format!("{median_decode_tok_s:.1}"),
                        "tok/s",
                    )
                    .with_style(CellStyle::Good),
                    Stat::new("Accept len (mean)", format!("{accept_len_mean:.2}"), ""),
                    Stat::new(
                        "Output tok (min run)",
                        format!("{min_output_tokens} / {MAX_TOKENS} cap"),
                        "",
                    ),
                ]
            }
        };
        let mut frame = BenchmarkResult {
            status: RunStatus::Completed,
            ..BenchmarkResult::running("done", self.elapsed())
        }
        .with_progress(total, total)
        .with_summary(summary)
        .with_table(self.table())
        .with_metrics(metrics)
        .with_verdict(verdict);
        if let Some(line) = sampler_cost_line {
            frame = frame.log_line(line);
        }
        Ok(frame)
    }
}

#[cfg(test)]
#[path = "decode_floor_tests.rs"]
mod tests;
