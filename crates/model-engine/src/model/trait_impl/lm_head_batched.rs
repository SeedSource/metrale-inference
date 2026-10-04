// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The batched decode LM head, shared by the pure-decode batch and the mixed decode + prefill head.
//!
//! `decode_batch_compute_main_with` (decode_a2.rs) and `mixed_final_norm_lm_head`
//! (decode_b2.rs) both call [`TransformerModel::lm_head_project_batched`], so the two heads
//! pick the same kernel, and produce the same bits, at a given `padded_n`.
//!
//! Owner: model-engine decode.
//! Invariants:
//! - The route depends only on `padded_n`, the model's weights and kernel handles, and values
//!   resolved once per process, so it is the same at CUDA-graph capture and replay.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::weight_map::DenseWeight;

use super::super::types::TransformerModel;
use metrale_model_layers::layers::ops;

/// 2026-09-25: Batched GEMV for the NVFP4 decode head: on unless
/// `METRALE_NO_LM_HEAD_BATCH_GEMV` is exactly `1`. Read once per process.
pub(super) fn lm_head_batch_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_LM_HEAD_BATCH_GEMV").as_deref() != Ok("1"))
}

/// 2026-09-25: `METRALE_LMHEAD_BATCH_GEMV` for the BF16 head: only the exact value `0`
/// disables the batched GEMV.
fn bf16_batch_gemv_from_value(value: Option<&str>) -> bool {
    value != Some("0")
}

/// 2026-09-25: The BF16 decode head's batched-GEMV band, the widest `m` that takes
/// `dense_gemv_batchm`; wider batches take a tile GEMM that reassociates the reduction.
/// Only this head reads it.
///
/// The default is the compiled target's `[defaults] lm_head_batchm_max`
/// (`kernels/<hw>/HARDWARE.toml`); `METRALE_LM_HEAD_BATCHM_MAX` overrides it. Resolution and
/// clamping are `ops::target_defaults::resolve_batchm_max`, cached once by `resolved()` so
/// the route is constant across CUDA-graph replays.
fn lm_head_batchm_max() -> u32 {
    ops::target_defaults::resolved().lm_head_batchm_max.value
}

/// 2026-09-25: The BF16 head's tensor-core arm: which kernels this target carries, whether
/// the arm is on, and at what CTA width. [`lm_head_m16_tc_route`] is the rule that reads it.
#[derive(Debug, Clone, Copy)]
pub(super) struct LmHeadM16Tc {
    /// 2026-09-25: `dense_gemm_m16_bf16` (N_TILE=32). 0 when the kernel set lacks it.
    pub narrow: KernelHandle,
    /// 2026-09-25: `dense_gemm_m16_bf16_n64` (N_TILE=64). 0 when absent.
    pub wide: KernelHandle,
    /// 2026-09-25: The target's `lm_head_m16_tc` default, overridden by `METRALE_LM_HEAD_M16_TC`
    /// (`0`, `false`, `off` or `no` turn it off; any other value turns it on).
    pub enabled: bool,
    /// 2026-09-25: Requested CTA width: 32 (default) or 64.
    pub n_tile: u32,
}

/// 2026-09-25: `METRALE_LM_HEAD_M16_TC_NTILE`: 32 (default) or 64. Any other value
/// falls back to 32 rather than failing the boot; the route log names the tile that ran.
fn m16_tc_n_tile_from_value(value: Option<&str>) -> u32 {
    match value.map(str::trim) {
        Some("64") => ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE,
        _ => ops::DENSE_GEMM_M16_BF16_N_TILE,
    }
}

/// 2026-09-25: Process-wide resolution of both, `OnceLock`-cached: the route must be constant
/// across CUDA-graph replays, and a per-call `env::var` could change the launch set between
/// capture and replay.
fn lm_head_m16_tc_env() -> (bool, u32) {
    static ENV: std::sync::OnceLock<(bool, u32)> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        let n_tile = std::env::var("METRALE_LM_HEAD_M16_TC_NTILE").ok();
        (
            // 2026-09-25: The target's `[defaults] lm_head_m16_tc` row, then
            // `METRALE_LM_HEAD_M16_TC`; `=0` keeps the bit-exact tiers.
            ops::target_defaults::resolved().lm_head_m16_tc.value,
            m16_tc_n_tile_from_value(n_tile.as_deref()),
        )
    })
}

/// 2026-09-25: The whole selection rule for the tensor-core head arm, as a pure function of
/// the row count, the reduction depth and the resolved lever and handles.
///
/// The band is `5..=DENSE_GEMM_M16_BF16_MAX_M` (16) rows. Below it the batched GEMV or the
/// GEMM serves the head, `m == 1` included; above it is past the kernel's M tile. `k` must
/// be a multiple of `DENSE_GEMM_M16_BF16_K_STEP` (64), the kernel's per-step advance; any
/// other K declines here rather than launching.
///
/// Returns the launcher, its handle and the CTA width that will run: a target without the
/// `_n64` kernel serves a 64 request with the 32-wide kernel.
fn lm_head_m16_tc_route(
    tc: LmHeadM16Tc,
    m: u32,
    k: u32,
) -> Option<(ops::DenseM16Bf16Gemm, KernelHandle, u32)> {
    if !tc.enabled
        || !(5..=ops::DENSE_GEMM_M16_BF16_MAX_M).contains(&m)
        || !k.is_multiple_of(ops::DENSE_GEMM_M16_BF16_K_STEP)
    {
        return None;
    }
    if tc.n_tile == ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE && tc.wide.0 != 0 {
        return Some((
            ops::dense_gemm_m16_bf16_n64,
            tc.wide,
            ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE,
        ));
    }
    if tc.narrow.0 != 0 {
        return Some((
            ops::dense_gemm_m16_bf16,
            tc.narrow,
            ops::DENSE_GEMM_M16_BF16_N_TILE,
        ));
    }
    None
}

fn lmhead_batch_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        bf16_batch_gemv_from_value(std::env::var("METRALE_LMHEAD_BATCH_GEMV").ok().as_deref())
    })
}

/// 2026-09-25: Shared BF16-head dispatch for ordinary and mixed multi-sequence decode.
#[allow(clippy::too_many_arguments)]
fn project_bf16_lm_head(
    gpu: &dyn GpuBackend,
    fallback: KernelHandle,
    batch_gemv: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    [m, n, k]: [u32; 3],
    batch_enabled: bool,
    batchm_max: u32,
    m16_tc: LmHeadM16Tc,
    stream: u64,
) -> Result<()> {
    // 2026-10-04: `METRALE_GLM_GEMV_TC=1` goes before every other arm: 1..=16 rows on the
    // row-invariant tensor-core GEMV (`ops::dense_gemv_tcm`), the kernel the 1-row and
    // verify heads also take under that lever.
    if ops::dense_gemv_tcm::try_bf16(gpu, input, weight, output, m, n, k, n, stream)? {
        return Ok(());
    }
    // 2026-09-25: The tensor-core arm goes first, at 5..=16 rows (`lm_head_m16_tc_route`). It
    // reassociates the K reduction, so it is not bit-identical to the batched GEMV; it runs
    // only where the target declares `lm_head_m16_tc` or `METRALE_LM_HEAD_M16_TC` turns it on.
    //
    // `dense_gemv_batchm` shares one BF16 weight read across up to `batchm_max` rows and
    // needs `k % 8 == 0` for its 16-byte loads. `batchm_max` is the decode band
    // (`lm_head_batchm_max`), not the kernel's `DENSE_GEMV_BATCHM_MAX_M`.
    if let Some((gemm, kernel, n_tile)) = lm_head_m16_tc_route(m16_tc, m, k) {
        log_m16_tc_head_route(n_tile, m16_tc.n_tile);
        return gemm(gpu, kernel, input, weight, output, m, n, k, k, n, stream);
    }
    if batch_enabled && batch_gemv.0 != 0 && (1..=batchm_max).contains(&m) && k.is_multiple_of(8) {
        ops::dense_gemv_batchm(gpu, batch_gemv, input, weight, output, m, n, k, n, stream)
    } else {
        ops::dense_gemm(gpu, fallback, input, weight, output, m, n, k, stream)
    }
}

/// 2026-09-25: The route line's text, separate from the `Once` latch below so a test can
/// pin the wording without tripping a process-global latch.
///
/// The ULP clause states the tier's contract,
/// `layers::dense_ffn::m16_tc::within_m16_tc_budget`: within 2 ordinal BF16 ULP, or under
/// the FP32 accumulation floor for an output that has cancelled; not a bare 2-ULP bound on
/// every element.
fn m16_tc_head_route_message(n_tile: u32, asked: u32) -> String {
    format!(
        "[metrale] BF16 lm_head decode: METRALE_LM_HEAD_M16_TC — tensor-core \
         dense_gemm_m16_bf16 N_TILE={n_tile} (asked {asked}) for 5..=16 rows, ahead of \
         dense_gemv_bf16_batchm. One weight pass, m16n8k16 MMA, so logits are \
         REASSOCIATED vs the scalar dense_gemv_bf16 — within 2 ordinal BF16 ULP, OR the \
         FP32 accumulation floor for outputs that have catastrophically cancelled (the \
         contract is layers::dense_ffn::m16_tc::within_m16_tc_budget, not a bare 2-ULP \
         bound; H100 round 9 measured up to 100 ordinal ULP on logits cancelled to \
         4.9e-6..2.6e-4 of the row RMS) — unlike the batched GEMV. Unset it to restore the \
         bit-exact tier (#927/#928)."
    )
}

/// 2026-09-25: Log-once latch for the tensor-core head arm. This arm reassociates the
/// reduction, so a report at 5..=16 rows needs to say which tier ran.
fn log_m16_tc_head_route(n_tile: u32, asked: u32) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!("{}", m16_tc_head_route_message(n_tile, asked));
    });
}

impl TransformerModel {
    /// 2026-09-25: This model's tensor-core head arm: the handles this target carries, plus
    /// the process-wide lever. Assembled per call so the env resolution stays in one
    /// `OnceLock`.
    fn lm_head_m16_tc(&self) -> LmHeadM16Tc {
        let (enabled, n_tile) = lm_head_m16_tc_env();
        LmHeadM16Tc {
            narrow: self.lm_head_m16_tc_kernel,
            wide: self.lm_head_m16_tc_n64_kernel,
            enabled,
            n_tile,
        }
    }

    /// 2026-09-25: Project `normed` [padded_n, H] into `logits` [padded_n, V].
    ///
    /// `v` is read from `self.config.vocab_size` rather than passed: it is the
    /// same number at both call sites and a parameter would be a second place
    /// for it to be wrong.
    pub(super) fn lm_head_project_batched(
        &self,
        normed: DevicePtr,
        padded_n: usize,
        h: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let logits = self.buffers.logits();
        let v = self.config.vocab_size;
        if let Some(ref fp8) = self.lm_head_fp8 {
            for i in 0..padded_n {
                ops::dense_gemv_fp8w(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_kernel,
                    normed.offset(i * h * bf16),
                    fp8,
                    logits.offset(i * v * bf16),
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // 2026-09-25: At `padded_n >= 5` a tile GEMM over the padded transposed twin
            // (`lm_head_nvfp4_t`) serves the head; below that, or without it, the batched GEMV
            // for `padded_n`; `w4a16_gemm` when no GEMV kernel applies or the batched GEMV is off.
            // 2026-09-27: Under a row-invariant tier policy the tile GEMM serves every
            // `padded_n`.
            let tile_rows = padded_n >= 5 || metrale_model_layers::layers::row_invariant();
            if tile_rows
                && self.w4a16_gemm_t_bf16_kernel.0 != 0
                && let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t
            {
                // 2026-09-25: Lossless path: BF16 MMA, no activation downcast.
                ops::w4a16_gemm_n128_m128_bf16_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_bf16_kernel,
                    normed,
                    nvfp4_t,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    ldb,
                    stream,
                )?;
            } else if tile_rows
                && self.w4a16_gemm_t_kernel.0 != 0
                && let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t
            {
                ops::w4a16_gemm_n128_ldb(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_t_kernel,
                    normed,
                    nvfp4_t,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    ldb,
                    stream,
                )?;
            } else {
                let narrow = self.w4a16_batchm.kernel(padded_n as u32);
                let gemv_k = if narrow.0 != 0 {
                    narrow
                } else if padded_n <= 16 {
                    self.w4a16_gemv_batch16_kernel
                } else {
                    metrale_gpu_runtime::gpu::KernelHandle(0)
                };
                if gemv_k.0 != 0 && lm_head_batch_gemv_enabled() {
                    ops::w4a16_gemv_batchm(
                        self.gpu.as_ref(),
                        gemv_k,
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        self.gpu.as_ref(),
                        self.w4a16_gemm_kernel,
                        normed,
                        nvfp4,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else {
            project_bf16_lm_head(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                self.dense_gemv_batchm_kernel,
                normed,
                &self.lm_head_weight,
                logits,
                [padded_n as u32, v as u32, h as u32],
                lmhead_batch_gemv_enabled(),
                lm_head_batchm_max(),
                self.lm_head_m16_tc(),
                stream,
            )?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
#[path = "lm_head_bf16_tests.rs"]
mod bf16_tests;
