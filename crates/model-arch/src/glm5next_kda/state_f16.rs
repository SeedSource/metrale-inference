// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: The KDA recurrent state stored as FP16 (`METRALE_GLM_KDA_STATE_F16=1`; lever and
//! pool width in model-layers `ssm_reserve/kda_state_f16.rs`), the math kept FP32.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Off (the default) nothing here runs: every `KdaSeqState` the model builds has
//!   `recurrent_f16 == false` and every launch is the one it was.
//! - A state with `recurrent_f16` reaches only the decode and verify entry points (`decode`,
//!   `decode_k` with snapshots, `decode_k_defer`, `decode_n_seqs`, `decode_verify_n_seqs`). Their
//!   recurrent launches are the FP16-storage twins (kernels/gb10/common/kda_state_f16.cu) of the
//!   shared-memory row kernels, with the same grid, block, shared memory and per-row pointers;
//!   every state copy they make moves `recurrent_state_elems() * 2` bytes. The conv state stays
//!   FP32 and keeps its kernels. The two-pass `kda_recurrent_decode_bf16` and the token loop have
//!   no twin: a state they would get is refused (`stateful_row`) or walked per row
//!   (`stateful_rows` declines), never handed to an FP32 kernel.
//! - The prefill arms (`prefill`, `prefill_chunked_tc*`, `prefill_flashkda`, the in-pass capture)
//!   refuse an FP16 state ([`KdaSeqState::require_fp32`]). The model runs them on the slot's
//!   FP32 staging blob instead: [`Glm5NextKdaLayer::widen_state`] before,
//!   [`Glm5NextKdaLayer::narrow_state`] after (glm5next_layer `steps/mixer.rs`).
//! - The loader refuses the lever at boot ([`state_f16_refusal`]) when a twin or a conversion
//!   did not resolve or the row walk would not take the shared-memory kernel.

use super::*;

/// 2026-10-12: `METRALE_GLM_KDA_STATE_F16=1` (read once per process, model-layers
/// `ssm_reserve::glm_kda_state_f16`).
pub fn kda_state_f16() -> bool {
    metrale_model_layers::ssm_reserve::glm_kda_state_f16()
}

/// 2026-10-12: Why the FP16 state cannot run with these kernels on this geometry, or `None`
/// when it can: every decode and verify row must take a shared-memory row kernel, which has a
/// twin, and the prefill staging needs both conversions.
pub fn state_f16_refusal(cfg: &Glm5NextKdaConfig, k: &Glm5NextKdaKernels) -> Option<&'static str> {
    let d = cfg.head_dim;
    let vpb = KDA_V_PER_BLOCK.min(d);
    let smem = (3 * d + vpb * (d + 1)) * 4;
    if !k.has_state_f16() {
        Some("the target lacks a kda_state_f16 twin or an ssm_h_dtype conversion")
    } else if kda_no_smem() {
        Some("METRALE_GLM_KDA_NO_SMEM=1 selects the two-pass kernel, which has no FP16 twin")
    } else if !d.is_multiple_of(vpb) || smem > KDA_SMEM_BUDGET {
        Some("the geometry would take the two-pass kernel, which has no FP16 twin")
    } else {
        None
    }
}

impl KdaSeqState {
    /// 2026-10-12: Bytes of one recurrent state (or snapshot slot) in this state's format.
    pub fn recurrent_bytes(&self, cfg: &Glm5NextKdaConfig) -> usize {
        cfg.recurrent_state_elems() * if self.recurrent_f16 { 2 } else { 4 }
    }

    /// 2026-10-12: Errors for an FP16 state: `what` runs FP32 kernels over `recurrent`.
    pub fn require_fp32(&self, what: &str) -> Result<()> {
        if self.recurrent_f16 {
            bail!(
                "KDA {what}: the recurrent state is FP16 (METRALE_GLM_KDA_STATE_F16=1) and this \
                 path runs FP32 kernels; the model must stage it into the FP32 prefill blob"
            );
        }
        Ok(())
    }
}

impl Glm5NextKdaLayer {
    /// 2026-10-12: Widen the FP16 state at `slot` into the FP32 blob `stage`
    /// (`ssm_h_state_f16_to_f32`, exact).
    pub fn widen_state(
        &self,
        gpu: &dyn GpuBackend,
        slot: DevicePtr,
        stage: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.kernels.h_widen.0 == 0 {
            bail!("METRALE_GLM_KDA_STATE_F16: ssm_h_dtype::ssm_h_state_f16_to_f32 did not resolve");
        }
        metrale_model_layers::layers::ops::ssm_h_state_f16_to_f32(
            gpu,
            self.kernels.h_widen,
            slot,
            stage,
            self.cfg.recurrent_state_elems() as u64,
            stream,
        )
    }

    /// 2026-10-12: Narrow the FP32 blob `stage` back into the FP16 state at `slot`
    /// (`ssm_h_state_f32_to_f16`, round to nearest even).
    pub fn narrow_state(
        &self,
        gpu: &dyn GpuBackend,
        stage: DevicePtr,
        slot: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.kernels.h_narrow.0 == 0 {
            bail!("METRALE_GLM_KDA_STATE_F16: ssm_h_dtype::ssm_h_state_f32_to_f16 did not resolve");
        }
        metrale_model_layers::layers::ops::ssm_h_state_f32_to_f16(
            gpu,
            self.kernels.h_narrow,
            stage,
            slot,
            self.cfg.recurrent_state_elems() as u64,
            stream,
        )
    }

    /// 2026-10-12: Whether this layer can run an FP16 state ([`state_f16_refusal`] is `None`).
    /// Independent of the lever (the microtest's arm).
    pub fn state_f16_ready(&self) -> bool {
        state_f16_refusal(&self.cfg, &self.kernels).is_none()
    }

    /// 2026-10-12: Microtest arm (`glm_kda_state_f16_microtest` timing): the fused-snapshot walk
    /// alone (`snap_walk`: conv and recurrent row launches, no front or back end) over workspace
    /// rows `0..k` as a previous call's front end left them, on `state` in its own format.
    /// Refuses where `snap_walk` would decline, so a timing cannot measure a fallback.
    pub fn snap_walk_arm(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        stream: u64,
    ) -> Result<()> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA snap-walk arm of {k} rows: outside 1..={}",
                ws.max_tokens
            );
        }
        if !self.snap_walk(gpu, 0, k, state, ws, snapshots, stream)? {
            bail!("KDA snap-walk arm: the fused walk is not ready on this layer");
        }
        Ok(())
    }

    /// 2026-10-12: The recurrent launch of one decode/verify row over an FP16 state: the twin of
    /// `kda_recurrent_decode_bf16_smem` (in place, `h_in == h_out`) or of the `_io` kernel
    /// (`h_in != h_out`), with `stateful_row`'s grid, block, shared memory and per-row pointers.
    /// Refuses, launching nothing, where the twin cannot run.
    pub(super) fn recurrent_row_f16(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        h_in: DevicePtr,
        h_out: DevicePtr,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        if let Some(why) = state_f16_refusal(&self.cfg, &self.kernels) {
            bail!(
                "KDA layer {}: FP16 recurrent state refused: {why}",
                self.layer_idx
            );
        }
        let c = &self.cfg;
        let qkv = c.qkv_dim();
        let cd = c.conv_dim();
        let d = c.head_dim;
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem = (3 * d + vpb * (d + 1)) * 4;
        let kernel = if h_in == h_out {
            self.kernels.recurrent_smem_f16
        } else {
            self.kernels.recurrent_io_f16
        };
        let mut l = KernelLaunch::new(gpu, kernel)
            .grid([c.heads as u32, (d / vpb) as u32, 1])
            .block([vpb as u32, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(ws.conv_out.offset(row * cd * 2))
            .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
            .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
            .arg_ptr(ws.gate.offset(row * qkv * 4))
            .arg_ptr(ws.beta.offset(row * c.heads * 4))
            .arg_ptr(h_in);
        if h_in != h_out {
            l = l.arg_ptr(h_out);
        }
        l.arg_ptr(ws.core.offset(row * qkv * 4))
            .arg_u32(c.heads as u32)
            .arg_u32(d as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .arg_u32(vpb as u32)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Glm5NextKdaConfig {
        Glm5NextKdaConfig {
            hidden: 6144,
            heads: 32,
            head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            rms_norm_eps: 1e-5,
            l2_eps: 1e-6,
            chunk: 32,
        }
    }

    const fn st(f16: bool) -> KdaSeqState {
        KdaSeqState {
            conv: DevicePtr(0x1000),
            recurrent: DevicePtr(0x2000),
            recurrent_f16: f16,
        }
    }

    /// 2026-10-12: GLM-5.3 TP2: 2 MiB FP32, 1 MiB FP16 per layer.
    #[test]
    fn recurrent_bytes_follow_the_format() {
        let c = cfg();
        assert_eq!(st(false).recurrent_bytes(&c), 2 << 20);
        assert_eq!(st(true).recurrent_bytes(&c), 1 << 20);
        assert_eq!(st(false).recurrent_bytes(&c), c.recurrent_state_elems() * 4);
    }

    #[test]
    fn prefill_arms_refuse_an_f16_state() {
        assert!(st(false).require_fp32("prefill").is_ok());
        let e = st(true).require_fp32("prefill").unwrap_err().to_string();
        assert!(e.contains("FP16") && e.contains("prefill"), "{e}");
    }
}
