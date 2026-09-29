// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-side token sampling: `SamplingParams`, the `Sampler` that
//! copies device logits to the host, the penalty stage, and the argmax helpers.
//!
//! Owner: metrale-sampling.
//! Invariants:
//! - In this crate, only the non-greedy path of `sample_with_params_seeded`
//!   writes the entropy gauges (through `record_entropy`).

use std::sync::atomic::Ordering;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

// 2026-09-25: The entropy gauges are fields of
// `metrale_telemetry::run_metrics::RunMetrics`; `reset_for_new_run` zeroes
// them when a run starts.

/// 2026-09-25: Entropy (nats) of the last non-greedy sample in this run.
pub fn last_entropy() -> f32 {
    f32::from_bits(
        metrale_telemetry::run_metrics::metrics()
            .last_entropy
            .load(Ordering::Relaxed),
    )
}

/// 2026-09-25: Non-greedy samples in this run whose entropy was below 0.3 nats.
pub fn low_entropy_token_count() -> u64 {
    metrale_telemetry::run_metrics::metrics()
        .low_entropy_tokens
        .load(Ordering::Relaxed)
}

/// 2026-09-25: Non-greedy samples in this run that recorded an entropy.
pub fn total_sampled_token_count() -> u64 {
    metrale_telemetry::run_metrics::metrics()
        .total_sampled_tokens
        .load(Ordering::Relaxed)
}

pub(crate) fn record_entropy(entropy: f32) {
    let m = metrale_telemetry::run_metrics::metrics();
    m.last_entropy.store(entropy.to_bits(), Ordering::Relaxed);
    m.total_sampled_tokens.fetch_add(1, Ordering::Relaxed);
    if entropy < 0.3 {
        m.low_entropy_tokens.fetch_add(1, Ordering::Relaxed);
    }
}

/// 2026-09-25: Sampling parameters for a request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 2026-09-25: Temperature. 0.0 is greedy (`is_greedy`).
    pub temperature: f32,
    /// 2026-09-25: Top-k: keep only the k highest-probability tokens.
    /// 0 disables it.
    pub top_k: u32,
    /// 2026-09-25: Top-p (nucleus): keep the smallest set of tokens whose
    /// cumulative probability reaches p. 1.0 or more disables it.
    pub top_p: f32,
    /// 2026-09-25: Top-n-sigma: before temperature scaling, drop tokens whose
    /// logit is below mean - n*sigma. 0.0 disables it.
    pub top_n_sigma: f32,
    /// 2026-09-25: Min-p: keep tokens with prob >= min_p * max_prob
    /// (post-softmax). 0.0 disables it.
    pub min_p: f32,
    /// 2026-09-25: Per-token logit bias as (token_id, bias) pairs, added after
    /// the penalties and before any filter.
    pub logit_bias: Vec<(u32, f32)>,
    /// 2026-09-25: Repetition penalty: for every occurrence of a token in the
    /// window, a positive logit is divided by it and a non-positive one
    /// multiplied. 1.0 and values <= 0.0 disable it.
    pub repetition_penalty: f32,
    /// 2026-09-25: Window, in tokens, for the repetition, presence and
    /// frequency penalties. 0 uses the whole history.
    pub repetition_penalty_window: u32,
    /// 2026-09-25: Presence penalty: subtracted once from the logit of each
    /// token that appears in the window. 0.0 disables it.
    pub presence_penalty: f32,
    /// 2026-09-25: Frequency penalty: subtracted from a token's logit once per
    /// occurrence in the window. 0.0 disables it.
    pub frequency_penalty: f32,
    /// 2026-09-25: Strength of `apply_lz_penalty`. Values <= 0.0 disable it.
    pub lz_penalty: f32,
    /// 2026-09-25: Multiplier of the DRY ("don't repeat yourself") penalty,
    /// `apply_dry_penalty`. Values <= 0.0 disable it.
    pub dry_multiplier: f32,
    /// 2026-09-25: DRY base: penalty = multiplier * base^(match_len - allowed_len).
    pub dry_base: f32,
    /// 2026-09-25: DRY allowed length: only matches longer than this are penalised.
    pub dry_allowed_length: u32,
    /// 2026-09-25: DRY sequence breakers: token ids at which a match stops.
    pub dry_sequence_breakers: Vec<u32>,
    /// 2026-09-25: Maximum tokens to generate.
    pub max_tokens: usize,
    /// 2026-09-25: Stop token ids.
    pub stop_token_ids: Vec<u32>,
    /// 2026-09-25: RNG seed. `Some` seeds a `StdRng` for each draw; `None`
    /// uses the thread RNG.
    pub seed: Option<u64>,
}

impl SamplingParams {
    /// 2026-09-25: Greedy sampling with every filter and penalty neutral.
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_sequence_breakers: Vec::new(),
            max_tokens,
            stop_token_ids: Vec::new(),
            seed: None,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// 2026-09-25: Copies device logits to the host and picks tokens there.
pub struct Sampler {
    /// 2026-09-25: Host buffer for the BF16 logits copy, reused across calls.
    logits_host: Vec<u8>,
    /// 2026-09-25: FP32 expansion of one row of `logits_host`.
    logits_f32: Vec<f32>,
    vocab_size: usize,
}

impl Sampler {
    pub fn new(vocab_size: usize) -> Self {
        let logits_host = vec![0u8; vocab_size * 2];
        let logits_f32 = vec![0.0f32; vocab_size];
        Self {
            logits_host,
            logits_f32,
            vocab_size,
        }
    }

    /// 2026-09-25: Copy one row of BF16 logits from the device, expand it to FP32 and return it.
    fn fetch_logits_f32(&mut self, logits_ptr: DevicePtr, gpu: &dyn GpuBackend) -> Result<&[f32]> {
        let byte_len = self.vocab_size * 2;
        gpu.copy_d2h(logits_ptr, &mut self.logits_host[..byte_len])?;
        for i in 0..self.vocab_size {
            self.logits_f32[i] = bf16_to_f32(self.logits_host[i * 2], self.logits_host[i * 2 + 1]);
        }
        // 2026-09-25: With `METRALE_DUMP_LOGITS_PATH=<dir>`, append this row's
        // FP32 logits to `<dir>/logits_fetch.bin`. The variable is read once per
        // process. The server's `RunDumps` (scheduler/dumps.rs) reads the same
        // variable and writes `logits_seq.bin` and `logits_stok.bin` in the same
        // directory. Open and write errors are ignored.
        static DUMP_DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        if let Some(dir) = DUMP_DIR.get_or_init(|| std::env::var("METRALE_DUMP_LOGITS_PATH").ok()) {
            use std::io::Write;
            let path = std::path::Path::new(&dir).join("logits_fetch.bin");
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        self.logits_f32.as_ptr() as *const u8,
                        self.vocab_size * 4,
                    )
                };
                let _ = f.write_all(bytes);
            }
        }
        Ok(&self.logits_f32[..self.vocab_size])
    }

    /// 2026-09-25: Pick one token from `[vocab_size]` BF16 logits on the device.
    ///
    /// The pick runs on the host. Greedy params take `argmax_bf16` of the raw
    /// row, with no penalty or bias. Otherwise the row is expanded to FP32 and
    /// passed to `sample_with_params` with an empty history.
    pub fn sample(
        &mut self,
        logits_ptr: DevicePtr,
        params: &SamplingParams,
        gpu: &dyn GpuBackend,
    ) -> Result<u32> {
        if params.is_greedy() {
            let byte_len = self.vocab_size * 2;
            gpu.copy_d2h(logits_ptr, &mut self.logits_host[..byte_len])?;
            return Ok(argmax_bf16(&self.logits_host[..byte_len]));
        }
        let f32_logits = self.fetch_logits_f32(logits_ptr, gpu)?;
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, f32_logits.len() * 4)
        };
        Ok(sample_with_params(f32_bytes, params))
    }

    /// 2026-09-25: Pick one token per row of `[batch_size, vocab_size]` BF16
    /// logits on the device, as `sample` does for one row.
    ///
    /// Row `i` uses `params[i]`, or `params[0]` when `params` is shorter; it
    /// panics when `params` is empty.
    pub fn sample_batch(
        &mut self,
        logits_ptr: DevicePtr,
        batch_size: usize,
        params: &[&SamplingParams],
        gpu: &dyn GpuBackend,
    ) -> Result<Vec<u32>> {
        let total_bytes = batch_size * self.vocab_size * 2;
        if self.logits_host.len() < total_bytes {
            self.logits_host.resize(total_bytes, 0);
        }
        gpu.copy_d2h(logits_ptr, &mut self.logits_host[..total_bytes])?;

        let stride_bf16 = self.vocab_size * 2;
        let mut tokens = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * stride_bf16;
            let end = start + stride_bf16;
            let p = params.get(i).copied().unwrap_or(params[0]);
            tokens.push(if p.is_greedy() {
                argmax_bf16(&self.logits_host[start..end])
            } else {
                if self.logits_f32.len() < self.vocab_size {
                    self.logits_f32.resize(self.vocab_size, 0.0);
                }
                for j in 0..self.vocab_size {
                    self.logits_f32[j] = bf16_to_f32(
                        self.logits_host[start + j * 2],
                        self.logits_host[start + j * 2 + 1],
                    );
                }
                let f32_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        self.logits_f32.as_ptr() as *const u8,
                        self.vocab_size * 4,
                    )
                };
                sample_with_params(f32_bytes, p)
            });
        }
        Ok(tokens)
    }
}

pub mod feed_argmax;
#[cfg(test)]
#[path = "feed_argmax_tests.rs"]
mod feed_argmax_tests;
mod penalties;
pub use penalties::{apply_dry_penalty, apply_lz_penalty, apply_penalties_and_bias};
mod sample_impl;
pub use sample_impl::{
    greedy_pick_last_wins, sample_with_params_history, sample_with_params_seeded,
};

/// 2026-09-25: `sample_with_params_history` with an empty history, so of the
/// penalty stage only the logit bias applies.
pub fn sample_with_params(data: &[u8], params: &SamplingParams) -> u32 {
    sample_with_params_history(data, params, &[])
}

/// 2026-09-25: Index of the first maximum of `v` under strict `>`.
///
/// Two passes: the maximum over 8 lanes plus the remainder, then the first
/// index equal to it. NaN never wins, and empty or all-NaN input returns 0.
/// -0.0 and +0.0 compare equal, so the first of them wins. The server's verify
/// path calls this (`verify_pipeline_helper/argmax.rs`), and its tests compare
/// it with a one-loop form.
pub fn argmax_first_wins_f32(v: &[f32]) -> u32 {
    const LANES: usize = 8;
    let mut acc = [f32::NEG_INFINITY; LANES];
    let mut chunks = v.chunks_exact(LANES);
    for c in &mut chunks {
        for (a, &x) in acc.iter_mut().zip(c) {
            if x > *a {
                *a = x;
            }
        }
    }
    let mut best = f32::NEG_INFINITY;
    for &a in acc.iter() {
        if a > best {
            best = a;
        }
    }
    for &x in chunks.remainder() {
        if x > best {
            best = x;
        }
    }
    v.iter()
        .position(|&x| x == best)
        .unwrap_or(0)
        .try_into()
        .unwrap_or(0)
}

/// 2026-09-25: Argmax over little-endian FP32 bytes, with the same two passes
/// and first-index tie-break as [`argmax_first_wins_f32`], reading the byte
/// chunks directly instead of building a `Vec<f32>`.
pub fn argmax_f32(data: &[u8]) -> u32 {
    debug_assert!(data.len().is_multiple_of(4));
    let vals = || {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
    };
    const LANES: usize = 8;
    let mut acc = [f32::NEG_INFINITY; LANES];
    let mut it = data.chunks_exact(4 * LANES);
    for block in &mut it {
        for (a, c) in acc.iter_mut().zip(block.chunks_exact(4)) {
            let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            if x > *a {
                *a = x;
            }
        }
    }
    let mut best = f32::NEG_INFINITY;
    for &a in acc.iter() {
        if a > best {
            best = a;
        }
    }
    for c in it.remainder().chunks_exact(4) {
        let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        if x > best {
            best = x;
        }
    }
    vals()
        .position(|x| x == best)
        .unwrap_or(0)
        .try_into()
        .unwrap_or(0)
}

/// 2026-09-25: Argmax over little-endian BF16 bytes; the first maximum wins, and
/// empty input returns 0. Used by `Sampler`.
pub fn argmax_bf16(data: &[u8]) -> u32 {
    debug_assert!(data.len().is_multiple_of(2));
    let n = data.len() / 2;
    if n == 0 {
        return 0;
    }
    let mut best_idx: u32 = 0;
    let mut best_val = bf16_to_f32(data[0], data[1]);
    for i in 1..n {
        let val = bf16_to_f32(data[i * 2], data[i * 2 + 1]);
        if val > best_val {
            best_val = val;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// 2026-09-25: Convert one little-endian BF16 value to f32 (exact).
#[inline]
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = (lo as u32) | ((hi as u32) << 8);
    f32::from_bits(bits << 16)
}

#[cfg(test)]
mod tests;
