// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_KDA_STATE_ROUND=bf16|fp16`: a quality-test lever that rounds every
//! KDA recurrent state, in place, to BF16 or FP16 precision (and back to FP32) after each write,
//! so a later job can measure what a half-width state costs in output quality before anyone
//! writes a half-width-state kernel. Speed does not matter here; correctness of the emulation
//! and "off = nothing changes" do.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Unset, empty, `0` or `off`: [`state_round`] is `None`, nothing is allocated or launched, and
//!   no decode path changes (the token loop and the fused snapshot walk are gated on
//!   `state_round().is_none()`, which is true).
//! - Any other value than `bf16` / `fp16` fails the load ([`validate`], called by
//!   `Glm5NextKdaKernels::resolve`).
//! - The rounding is a SEPARATE kernel (`kernels/gb10/common/kda_state_round.cu`), launched on the
//!   caller's stream right after the kernel that wrote the state, so a CUDA graph captures it like
//!   its neighbour. The production recurrent kernels are not modified.
//! - Decode: `Glm5NextKdaLayer::stateful_row` rounds the `[heads, head_dim, head_dim]` FP32
//!   state it just advanced (so also every row of `decode_k`'s walk, `decode_n_seqs` and the
//!   batched verify walk, and every snapshot copied after a row). While the lever is on, the
//!   token loop (`stateful_rows`) and the fused snapshot walk (`snap_walk`) do not run, because
//!   they keep the state in FP32 across rows; the per-row walk runs instead.
//! - Prefill: the chunked arms (FlashKDA, chunked TC, FP32 chunked scan) advance the state
//!   inside one scan in FP32 and write the final state once; [`Glm5NextKdaLayer::state_round_prefill`]
//!   rounds that final state after the scan (called from `steps/mixer.rs`). So prefill rounding
//!   is once per sub-chunk call, not once per token. An in-pass snapshot captured by the chunked
//!   TC arm inside the call is copied from the un-rounded state.
//! - Statistics and logging happen on the device (kernel comment): per layer, the max |value|
//!   before rounding, the count of finite values that round to infinity (FP16 above 65504) and the
//!   count of non-finite values. The kernel's last block prints
//!   `KDA_STATE_ROUND fmt=<f> step=<n> absmax=<v> overflow=<count> phase=decode|prefill layer=<i>
//!   nonfinite=<count>` every [`LOG_EVERY`] decode launches of a layer (`step` = that layer's
//!   decode launches since the lever was armed) and once per prefill, at the first decode launch
//!   after it (`step` = prefill sub-chunk calls), then clears the counters. Device printf needs
//!   no host readback, so nothing syncs and nothing breaks graph capture; the text appears
//!   when the host next synchronizes. A run that prefills and never decodes prints no prefill
//!   line. A decode launch is one state update, so a multi-row verify counts one per row.

use super::*;
use std::sync::OnceLock;

/// 2026-10-12: Decode launches of one layer between log lines.
pub const LOG_EVERY: u32 = 256;
/// 2026-10-12: Layer slots in the statistics buffer; a layer index at or above it shares the last.
pub const SLOTS: usize = 128;
/// 2026-10-12: u32 words in one layer's statistics slot (kernel comment).
pub const SLOT_WORDS: usize = 16;
const BLOCK: u32 = 256;
const MAX_BLOCKS: u64 = 2048;

/// 2026-10-12: The rounding format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundFmt {
    Bf16,
    Fp16,
}

impl RoundFmt {
    /// The lever value / log name.
    pub fn name(self) -> &'static str {
        match self {
            RoundFmt::Bf16 => "bf16",
            RoundFmt::Fp16 => "fp16",
        }
    }
    /// The kernel's `mode` argument.
    pub fn mode(self) -> u32 {
        match self {
            RoundFmt::Bf16 => 0,
            RoundFmt::Fp16 => 1,
        }
    }
}

/// 2026-10-12: Parse a `METRALE_GLM_KDA_STATE_ROUND` value: unset, empty, `0` or `off` = off;
/// `bf16` or `fp16` = that format (case-insensitive); anything else is an error.
pub fn parse_state_round(raw: Option<&str>) -> Result<Option<RoundFmt>> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "" | "0" | "off" => Ok(None),
        "bf16" => Ok(Some(RoundFmt::Bf16)),
        "fp16" => Ok(Some(RoundFmt::Fp16)),
        other => {
            bail!("METRALE_GLM_KDA_STATE_ROUND: unknown format {other:?} (bf16, fp16, or unset)")
        }
    }
}

fn parsed() -> &'static Result<Option<RoundFmt>, String> {
    static P: OnceLock<Result<Option<RoundFmt>, String>> = OnceLock::new();
    P.get_or_init(|| {
        parse_state_round(std::env::var("METRALE_GLM_KDA_STATE_ROUND").ok().as_deref())
            .map_err(|e| format!("{e:#}"))
    })
}

/// 2026-10-12: The rounding format the process runs with; `None` when the lever is off (or its
/// value is invalid, which [`validate`] reports). Read once.
pub fn state_round() -> Option<RoundFmt> {
    parsed().as_ref().ok().copied().flatten()
}

/// 2026-10-12: An error naming an invalid `METRALE_GLM_KDA_STATE_ROUND` value.
pub fn validate() -> Result<()> {
    match parsed() {
        Ok(_) => Ok(()),
        Err(e) => bail!("{e}"),
    }
}

/// 2026-10-12: Which launch a rounding follows (the statistics halves of the slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Decode,
    Prefill,
}

struct Shared {
    kernel: KernelHandle,
    stats: DevicePtr,
}

static SHARED: OnceLock<Shared> = OnceLock::new();

/// 2026-10-12: Resolve the kernel and allocate the zeroed statistics buffer
/// (`SLOTS * SLOT_WORDS` u32), once per process, when the lever is on; a no-op when it is off.
/// Called by `Glm5NextKdaKernels::resolve`, so a missing kernel fails the load, not a decode.
pub fn init(gpu: &dyn GpuBackend) -> Result<()> {
    validate()?;
    let Some(fmt) = state_round() else {
        return Ok(());
    };
    if SHARED.get().is_some() {
        return Ok(());
    }
    let kernel = gpu
        .kernel("kda_state_round", "kda_state_round")
        .map_err(|e| anyhow::anyhow!("METRALE_GLM_KDA_STATE_ROUND={}: {e:#}", fmt.name()))?;
    let bytes = SLOTS * SLOT_WORDS * 4;
    let stats = gpu.alloc(bytes)?;
    gpu.memset(stats, 0, bytes)?;
    let _ = SHARED.set(Shared { kernel, stats });
    tracing::warn!(
        "METRALE_GLM_KDA_STATE_ROUND={} - every GLM KDA recurrent state is rounded in place to {} \
         after each write (decode: per state update; prefill: once per chunked call); the token \
         loop and the fused snapshot walk are bypassed. Quality-test lever, not a product mode.",
        fmt.name(),
        fmt.name()
    );
    Ok(())
}

/// 2026-10-12: The statistics buffer (for the microtest); `None` when [`init`] has not run.
pub fn stats_buffer() -> Option<DevicePtr> {
    SHARED.get().map(|s| s.stats)
}

/// 2026-10-12: Launch the rounding kernel on `state` (`n` FP32 elements) with `slot` as the
/// layer's statistics slot, `log_every` as the kernel comment defines it. Public for the
/// microtest, which passes its own kernel handle and slot.
#[allow(clippy::too_many_arguments)]
pub fn launch_round(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    state: DevicePtr,
    n: usize,
    fmt: RoundFmt,
    slot: DevicePtr,
    phase: Phase,
    log_every: u32,
    layer: u32,
    stream: u64,
) -> Result<()> {
    let blocks = (n as u64).div_ceil(BLOCK as u64).clamp(1, MAX_BLOCKS) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(state)
        .arg_u64(n as u64)
        .arg_u32(fmt.mode())
        .arg_ptr(slot)
        .arg_u32(u32::from(phase == Phase::Prefill))
        .arg_u32(log_every)
        .arg_u32(layer)
        .launch(stream)
}

impl Glm5NextKdaLayer {
    /// 2026-10-12: Round `state.recurrent` (this layer's `[heads, head_dim, head_dim]` FP32 state)
    /// to the lever's format, on `stream`, right after the launch that wrote it. A no-op, with no
    /// launch, when the lever is off.
    pub(super) fn state_round_after(
        &self,
        gpu: &dyn GpuBackend,
        state: &KdaSeqState,
        phase: Phase,
        stream: u64,
    ) -> Result<()> {
        let Some(fmt) = state_round() else {
            return Ok(());
        };
        let Some(sh) = SHARED.get() else {
            bail!(
                "METRALE_GLM_KDA_STATE_ROUND={} but the rounding kernel was never initialised \
                 (Glm5NextKdaKernels::resolve)",
                fmt.name()
            );
        };
        let slot = sh
            .stats
            .offset(self.layer_idx.min(SLOTS - 1) * SLOT_WORDS * 4);
        launch_round(
            gpu,
            sh.kernel,
            state.recurrent,
            self.cfg.recurrent_state_elems(),
            fmt,
            slot,
            phase,
            LOG_EVERY,
            self.layer_idx as u32,
            stream,
        )
    }

    /// 2026-10-12: [`Self::state_round_after`] for the state a chunked prefill arm (FlashKDA,
    /// chunked TC, FP32 chunked scan) just wrote. Called by `steps/mixer.rs` after those arms.
    pub fn state_round_prefill(
        &self,
        gpu: &dyn GpuBackend,
        state: &KdaSeqState,
        stream: u64,
    ) -> Result<()> {
        self.state_round_after(gpu, state, Phase::Prefill, stream)
    }
}

/// 2026-10-12: Host reference of the kernel's rounding of one value: round to nearest even to
/// BF16 / FP16 and back to f32 (what `__float2bfloat16_rn` / `__float2half_rn` do), as bits.
pub fn round_ref_bits(v: f32, fmt: RoundFmt) -> u32 {
    match fmt {
        RoundFmt::Bf16 => half::bf16::from_f32(v).to_f32().to_bits(),
        RoundFmt::Fp16 => half::f16::from_f32(v).to_f32().to_bits(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_off_forms() {
        for v in [
            None,
            Some(""),
            Some("  "),
            Some("0"),
            Some("off"),
            Some("OFF"),
        ] {
            assert_eq!(parse_state_round(v).unwrap(), None, "{v:?}");
        }
    }

    #[test]
    fn parse_formats() {
        assert_eq!(
            parse_state_round(Some("bf16")).unwrap(),
            Some(RoundFmt::Bf16)
        );
        assert_eq!(
            parse_state_round(Some(" FP16 ")).unwrap(),
            Some(RoundFmt::Fp16)
        );
    }

    #[test]
    fn parse_rejects_anything_else() {
        for v in ["1", "fp8", "bfloat16", "bf16,fp16", "half", "true"] {
            let e = parse_state_round(Some(v)).unwrap_err().to_string();
            assert!(e.contains("METRALE_GLM_KDA_STATE_ROUND"), "{e}");
        }
    }

    #[test]
    fn format_names_and_modes_match_the_kernel() {
        assert_eq!((RoundFmt::Bf16.name(), RoundFmt::Bf16.mode()), ("bf16", 0));
        assert_eq!((RoundFmt::Fp16.name(), RoundFmt::Fp16.mode()), ("fp16", 1));
    }

    #[test]
    fn host_reference_rounds_to_nearest_even() {
        // 1 + 2^-8 is the BF16 tie between 1.0 and 1 + 2^-7: even mantissa (1.0) wins.
        assert_eq!(
            round_ref_bits(1.0 + 2f32.powi(-8), RoundFmt::Bf16),
            1.0f32.to_bits()
        );
        // 1 + 3 * 2^-8 ties between 1 + 2^-7 and 1 + 2^-6: even is 1 + 2^-6.
        assert_eq!(
            round_ref_bits(1.0 + 3.0 * 2f32.powi(-8), RoundFmt::Bf16),
            (1.0f32 + 2f32.powi(-6)).to_bits()
        );
        // FP16 overflows to infinity at 65520 and keeps 65504.
        assert_eq!(
            round_ref_bits(65504.0, RoundFmt::Fp16),
            65504.0f32.to_bits()
        );
        assert_eq!(
            round_ref_bits(65520.0, RoundFmt::Fp16),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            round_ref_bits(65519.0, RoundFmt::Fp16),
            65504.0f32.to_bits()
        );
    }
}
