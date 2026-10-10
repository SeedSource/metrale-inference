// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: FP16 storage of the GDN h-state (`--ssm-h-dtype f16` and `f16-pool`): the
//! decode-side guard and the prefill widen/narrow pair.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - Under `--ssm-h-dtype f16` the pool stays FP32-sized and the FP16 state occupies the
//!   first half of each slot, so slots are `h_state_bytes` apart and the FP16 decode kernel
//!   is given that pitch as `h_seq_stride`.
//! - `SsmLayerState::h_is_f16` records the slot's current format. The layer never converts
//!   it on the decode path: it selects a kernel, and [`require_h_f16`] returns an error for
//!   a state that is still FP32. The conversion is `TransformerModel::ssm_h_to_f16_dispatch`,
//!   called from the model's decode entry points outside the CUDA-graph region, because a
//!   conversion captured into a graph would re-run on every replay over state that is
//!   already FP16.
//! - Marconi snapshots are always FP32: `ssm_snapshot::save` widens an FP16 source, and
//!   `restore` narrows into an f16-sized pool.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, SsmLayerState};

/// 2026-09-25: Error when an FP16 decode kernel would run over a state still flagged
/// FP32. Reading FP32 bits as FP16 does not fault, so a missed conversion would
/// otherwise give wrong numbers silently.
pub(super) fn require_h_f16(state: &SsmLayerState) -> Result<()> {
    if !state.h_is_f16 {
        bail!(
            "METRALE_SSM_H_FP16: decode reached an SSM layer whose h-state is still FP32. \
             `ssm_h_to_f16_dispatch` must run at the top of every decode entry point, \
             OUTSIDE the CUDA-graph region."
        );
    }
    Ok(())
}

// 2026-09-25: Under `--ssm-h-dtype f16-pool` a pool slot holds 2 bytes per element, but
// the GDN prefill kernels read and write FP32 in place. Each slot therefore has an FP32
// staging blob (`SsmStatePool::h_prefill_stage`, shared by all layers of the slot):
// prefill widens the slot into it, runs the FP32 kernels over it, and narrows it back.
// Outside a layer's prefill call the slot holds FP16, and `h_is_f16` is set when the
// slot is allocated.

/// 2026-09-25: The h-state pointer a prefill pass runs over, and what [`prefill_h_end`]
/// must do afterwards. `Staged` carries both the slot and the stage, so the end call
/// cannot be handed a mismatched pair.
#[derive(Debug)]
pub(super) enum PrefillH {
    /// 2026-09-25: FP32-sized pool: the kernels run over the slot itself, and the end
    /// call moves nothing.
    InPlace(DevicePtr),
    /// 2026-09-25: f16-sized pool: the kernels run over `stage`, which is narrowed back
    /// into `slot`.
    Staged { slot: DevicePtr, stage: DevicePtr },
}

impl PrefillH {
    /// 2026-09-25: The pointer the FP32 prefill kernels run over.
    pub(super) fn ptr(&self) -> DevicePtr {
        match self {
            Self::InPlace(p) => *p,
            Self::Staged { stage, .. } => *stage,
        }
    }
}

/// 2026-09-25: Widen this sequence's h slot into its FP32 staging blob when the pool is
/// f16-sized (`h_prefill_stage` is set), and return the pointer the prefill kernels run
/// over.
///
/// `h_f32_bytes` is the FP32 size of one layer's h blob; the conversion covers
/// `h_f32_bytes / 4` elements. Errors, launching nothing, when the state is not flagged
/// FP16 or the widening kernel did not resolve.
pub(super) fn prefill_h_begin(
    gpu: &dyn GpuBackend,
    f16_to_f32_k: KernelHandle,
    state: &SsmLayerState,
    h_f32_bytes: usize,
    stream: u64,
) -> Result<PrefillH> {
    let Some(stage) = state.h_prefill_stage else {
        return Ok(PrefillH::InPlace(state.h_state));
    };
    if !state.h_is_f16 {
        bail!(
            "--ssm-h-dtype f16-pool: prefill reached an SSM layer whose h-state is flagged FP32 \
             over a 2-byte-sized pool slot. The slot cannot physically hold FP32; \
             `h_is_f16` must be set at slot allocation under the f16-sized pool."
        );
    }
    if f16_to_f32_k.0 == 0 {
        bail!(
            "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f16_to_f32 did not resolve on this \
             target — refusing to run the FP32 GDN prefill kernels over a 2-byte-sized pool slot \
             (that is an out-of-bounds write into the neighbouring slot, not a precision loss)."
        );
    }
    crate::layers::ops::ssm_h_state_f16_to_f32(
        gpu,
        f16_to_f32_k,
        state.h_state,
        stage,
        (h_f32_bytes / 4) as u64,
        stream,
    )?;
    Ok(PrefillH::Staged {
        slot: state.h_state,
        stage,
    })
}

/// 2026-09-25: Narrow the FP32 staging blob back into the h slot; a no-op for
/// `InPlace`. Errors when the narrowing kernel did not resolve.
///
/// Call it on the stream the prefill kernels ran on: only stream order separates the
/// narrow from those kernels, and the slot's next layer reuses the same staging blob.
pub(super) fn prefill_h_end(
    gpu: &dyn GpuBackend,
    f32_to_f16_k: KernelHandle,
    h: PrefillH,
    h_f32_bytes: usize,
    stream: u64,
) -> Result<()> {
    let PrefillH::Staged { slot, stage } = h else {
        return Ok(());
    };
    if f32_to_f16_k.0 == 0 {
        bail!(
            "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f32_to_f16 did not resolve on this \
             target — the prefill h-state cannot be narrowed back into its 2-byte-sized slot."
        );
    }
    crate::layers::ops::ssm_h_state_f32_to_f16(
        gpu,
        f32_to_f16_k,
        stage,
        slot,
        (h_f32_bytes / 4) as u64,
        stream,
    )
}

impl Qwen3SsmLayer {
    /// 2026-09-25: The h pool's per-slot byte pitch: `h_state_bytes` on an FP32-sized
    /// pool, half that under `--ssm-h-dtype f16-pool`.
    ///
    /// The pool computes `SsmStatePool::h_stored_bytes` with the same
    /// [`crate::ssm_reserve::ssm_h_stored_bytes`] and the same flag, so the layer and the
    /// allocator agree on how far apart two slots are.
    pub(super) fn h_slot_stride_bytes(&self) -> usize {
        crate::ssm_reserve::ssm_h_stored_bytes(
            self.h_state_bytes,
            super::gdn_flags::ssm_h_f16_pool_enabled(),
        )
    }

    /// 2026-09-25: [`Self::prefill_gdn_recurrence`] with the `--ssm-h-dtype f16-pool`
    /// widen/narrow around it.
    ///
    /// On an FP32-sized pool (`h_prefill_stage == None`) this is
    /// `prefill_gdn_recurrence(ssm_state.h_state, ..)`: `prefill_h_begin` returns the
    /// slot pointer and `prefill_h_end` launches nothing. On an f16-sized pool it widens
    /// the slot into the staging blob, runs the FP32 recurrence over the blob, and
    /// narrows the result back. If the recurrence returns an error the narrow is
    /// skipped, so the slot keeps its value from before the call.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_recurrence_staged(
        &self,
        ssm_state: &SsmLayerState,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gates_buf: DevicePtr,
        gdn_out_buf: DevicePtr,
        k: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
        midcap_idx: Option<usize>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = prefill_h_begin(
            ctx.gpu,
            self.ssm_h_f16_to_f32_k,
            ssm_state,
            self.h_state_bytes,
            stream,
        )?;
        self.prefill_gdn_recurrence(
            h.ptr(),
            q_ptr,
            k_ptr,
            v_ptr,
            gates_buf,
            gdn_out_buf,
            k,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            midcap_idx,
            ctx,
            stream,
        )?;
        prefill_h_end(
            ctx.gpu,
            self.ssm_h_f32_to_f16_k,
            h,
            self.h_state_bytes,
            stream,
        )
    }
}

#[cfg(test)]
mod prefill_narrowing_tests {
    use super::*;
    use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

    /// 2026-09-25: One layer's h blob at the Qwen3.6-27B GDN shape (48 v-heads × 128 ×
    /// 128 FP32).
    const H_F32: usize = 48 * 128 * 128 * 4;
    const K: KernelHandle = KernelHandle(0xDEAD);

    fn state(stage: Option<DevicePtr>, h_is_f16: bool) -> SsmLayerState {
        SsmLayerState {
            kda_defer: None,
            h_state: DevicePtr(0x1000),
            conv_state: DevicePtr(0x2000),
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            h_is_f16,
            h_prefill_stage: stage,
            ple: None,
        }
    }

    /// 2026-09-25: FP32-sized pool: the kernels are handed the slot itself and neither
    /// end launches a conversion.
    #[test]
    fn fp32_sized_pool_launches_nothing_and_hands_back_the_slot() {
        let gpu = MockGpuBackend::new();
        let st = state(None, false);
        let h = prefill_h_begin(&gpu, K, &st, H_F32, 0).unwrap();
        assert_eq!(h.ptr().0, st.h_state.0);
        assert!(matches!(h, PrefillH::InPlace(_)));
        prefill_h_end(&gpu, K, h, H_F32, 0).unwrap();
        assert!(
            gpu.launches_snapshot().is_empty(),
            "the FP32-sized pool must not launch a conversion"
        );
        // 2026-09-25: A null converter handle is not consulted on this pool, so a
        // target without `ssm_h_dtype.cu` still runs.
        let gpu2 = MockGpuBackend::new();
        let st2 = state(None, false);
        let h2 = prefill_h_begin(&gpu2, KernelHandle(0), &st2, H_F32, 0).unwrap();
        prefill_h_end(&gpu2, KernelHandle(0), h2, H_F32, 0).unwrap();
        assert!(gpu2.launches_snapshot().is_empty());
    }

    /// 2026-09-25: f16-sized pool: one widen on the way in and one narrow on the way
    /// out, the kernels run over the staging blob, and both launches carry the FP32
    /// element count (`h_bytes / 4`), not the byte count or the halved storage width.
    #[test]
    fn f16_sized_pool_widens_in_and_narrows_out_over_the_stage() {
        let gpu = MockGpuBackend::new();
        let stage = DevicePtr(0x9000);
        let st = state(Some(stage), true);
        let stream = 0xCAFE;

        let h = prefill_h_begin(&gpu, K, &st, H_F32, stream).unwrap();
        assert_eq!(h.ptr().0, stage.0, "the kernels must run over the stage");
        assert!(matches!(
            h,
            PrefillH::Staged { slot, stage: s } if slot.0 == st.h_state.0 && s.0 == stage.0
        ));
        let after_begin = gpu.launches_snapshot();
        assert_eq!(after_begin.len(), 1, "one widen");

        prefill_h_end(&gpu, K, h, H_F32, stream).unwrap();
        let all = gpu.launches_snapshot();
        assert_eq!(all.len(), 2, "one widen + one narrow, no more");

        // 2026-09-25: 256-thread blocks over `n = h_bytes / 4` elements, capped at 4096
        // blocks (the kernel is grid-stride). Both halves convert the same count.
        let n = (H_F32 / 4) as u32;
        assert_eq!(n, 786_432);
        for (launch, src, dst) in [(&all[0], st.h_state, stage), (&all[1], stage, st.h_state)] {
            let expected_n = (n as u64).to_ne_bytes().to_vec();
            assert_eq!(
                launch.args,
                vec![
                    MockArg::Buffer(src),
                    MockArg::Buffer(dst),
                    MockArg::Bytes(expected_n),
                ]
            );
            assert_eq!(launch.stream, stream);
            assert_eq!(launch.shared_mem, 0);
            assert_eq!(launch.block, [256, 1, 1]);
            assert_eq!(launch.grid, [n.div_ceil(256).clamp(1, 4096), 1, 1]);
        }
    }

    /// 2026-09-25: A null converter handle is an error at both ends: running the FP32
    /// kernels over a 2-byte-per-element slot would write past the slot.
    #[test]
    fn a_null_converter_refuses_rather_than_running_fp32_over_a_narrow_slot() {
        let gpu = MockGpuBackend::new();
        let st = state(Some(DevicePtr(0x9000)), true);
        let e = prefill_h_begin(&gpu, KernelHandle(0), &st, H_F32, 0).unwrap_err();
        assert!(e.to_string().contains("ssm_h_state_f16_to_f32"), "{e}");
        assert!(e.to_string().contains("out-of-bounds"), "{e}");
        assert!(gpu.launches_snapshot().is_empty());

        let staged = PrefillH::Staged {
            slot: DevicePtr(0x1000),
            stage: DevicePtr(0x9000),
        };
        let e2 = prefill_h_end(&gpu, KernelHandle(0), staged, H_F32, 0).unwrap_err();
        assert!(e2.to_string().contains("ssm_h_state_f32_to_f16"), "{e2}");
    }

    /// 2026-09-25: A staged slot flagged FP32 is refused before any launch: an
    /// f16-sized slot cannot hold FP32.
    #[test]
    fn an_fp32_flagged_state_over_a_narrow_slot_is_refused() {
        let gpu = MockGpuBackend::new();
        let st = state(Some(DevicePtr(0x9000)), false);
        let e = prefill_h_begin(&gpu, K, &st, H_F32, 0).unwrap_err();
        assert!(e.to_string().contains("flagged FP32"), "{e}");
        assert!(gpu.launches_snapshot().is_empty());
    }
}
