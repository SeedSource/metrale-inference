// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: The fused front of the tensor-core chunked KDA prefill
//! (`METRALE_GLM_KDA_FRONT_FUSE=1`, inert unless `METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1` runs).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Byte-identical to the unfused chunked-TC prefill: the chunk records, the core output, the
//!   block output and both states carry the same bits (kernels/gb10/common/kda_front_fuse.cu
//!   says why; `kda_front_fuse_microtest` checks it).
//! - Only [`Glm5NextKdaLayer::prefill_chunked_tc_arm`] takes this path, after
//!   `chunked_tc_refusal` passed; decode, verify, snapshot and capture walks never do.
//! - Allocates nothing; `ws.qkv_proj`, `ws.conv_out`, `ws.gate` and `ws.beta` are left
//!   unwritten (nothing on this path reads them).
//!
//! Unfused (per call):
//!
//! ```text
//! front_end: q|k|v GEMMs -> pack -> f GEMMs -> kda_gate -> b_proj -> sigmoid -> g GEMMs
//! kda_tc_conv_rows -> kda_tc_conv_state_tail -> kda_tc_prepare -> kda_tc_scan
//! ```
//!
//! Fused:
//!
//! ```text
//! front_end: q|k|v GEMMs -> f GEMMs -> b_proj -> g GEMMs
//! kda_ff_prepare -> kda_ff_conv_state_tail -> kda_tc_scan
//! ```
//!
//! At GLM-5.3 TP2 (32 heads per rank) and 8192 rows that drops the passes that wrote and
//! re-read `qkv_proj` (192 MiB), `conv_out` (192 MiB) and `gate` (128 MiB): about 1.0 GiB of
//! DRAM traffic per layer call (estimate, not measured).

use super::*;

/// 2026-10-07: `KDA_FF_DCONV` in kda_front_fuse.cu: the conv taps `kda_ff_prepare` is compiled
/// for.
const KDA_FF_DCONV: usize = 4;

/// 2026-10-07: Whether a `METRALE_GLM_KDA_FRONT_FUSE` value asks for the fused front: `1` only.
pub(super) fn front_fuse_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-07: `METRALE_GLM_KDA_FRONT_FUSE=1`: the tensor-core chunked prefill
/// ([`Glm5NextKdaLayer::prefill_chunked_tc`]) skips the pack, the gate and the beta sigmoid in
/// its front end and replaces `kda_tc_conv_rows`, `kda_tc_conv_state_tail` and
/// `kda_tc_prepare` with `kda_ff_prepare` and `kda_ff_conv_state_tail`. Byte-identical. Off
/// unless set to `1`. Read once per process.
pub(super) fn kda_front_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let on = front_fuse_requested(std::env::var("METRALE_GLM_KDA_FRONT_FUSE").ok().as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_FRONT_FUSE=1 - the KDA chunked-TC prefill fuses pack, conv, \
                 gate and beta into its prepare (byte-identical by construction; see \
                 kernels/gb10/common/kda_front_fuse.cu). Inert without \
                 METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1"
            );
        }
        on
    })
}

/// 2026-10-07: Why the fused front cannot run for this geometry, or `None` when it can.
/// `kernels` is [`Glm5NextKdaKernels::has_front_fuse`]. The caller has already passed
/// `chunked_tc_refusal` (head_dim 128, q|k width a multiple of 256).
pub(super) fn front_fuse_refusal(cfg: &Glm5NextKdaConfig, kernels: bool) -> Option<&'static str> {
    if !kernels {
        Some("the target lacks a kda_front_fuse kernel")
    } else if cfg.conv_kernel != KDA_FF_DCONV {
        Some("conv_kernel is not 4, the taps kda_ff_prepare is compiled for")
    } else {
        None
    }
}

/// 2026-10-07: The one warning when `METRALE_GLM_KDA_FRONT_FUSE=1` cannot be honoured and the
/// chunked-TC prefill keeps its unfused launches.
pub(super) fn kda_front_fuse_fallback(why: &str) {
    static W: std::sync::Once = std::sync::Once::new();
    W.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_KDA_FRONT_FUSE=1 ignored ({why}); the KDA chunked-TC prefill keeps \
             its unfused front"
        )
    });
}

impl Glm5NextKdaLayer {
    /// 2026-10-07: `stateful_chunked_tc` with the fused front: over rows `0..k`, reading the
    /// three projections in `ws.qkv_parts` (part `i` at `i * k * qkv` elements, as
    /// `front_end_opts` wrote them), `ws.g_raw` and `ws.beta_bf16`. The conv state and the
    /// recurrent state advance in place; the core output lands in `ws.core`. The caller has
    /// checked `chunked_tc_refusal` and [`front_fuse_refusal`].
    pub(super) fn stateful_chunked_tc_fused(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let (qkv, cd, d) = (c.qkv_dim(), c.conv_dim(), c.head_dim);
        let nchunks = k.div_ceil(KDA_TC_C);
        // 2026-10-07: logged once when the fused front first runs (gate scripts grep it).
        static ENGAGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        ENGAGED.get_or_init(|| tracing::warn!("METRALE_GLM_KDA_FRONT_FUSE=1: ENGAGED"));
        let q = ws.qkv_parts;
        let kp = ws.qkv_parts.offset(k * qkv * 2);
        let v = ws.qkv_parts.offset(2 * k * qkv * 2);

        // 2026-10-07: Reads the incoming conv state; the tail below then writes it.
        KernelLaunch::new(gpu, self.kernels.ff_prepare)
            .grid([nchunks as u32, c.heads as u32, 1])
            .block([KDA_TC_D as u32, 1, 1])
            .arg_ptr(q)
            .arg_ptr(kp)
            .arg_ptr(v)
            .arg_ptr(state.conv)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.g_raw)
            .arg_ptr(self.weights.dt_bias)
            .arg_ptr(self.weights.a_log)
            .arg_ptr(ws.beta_bf16)
            .arg_ptr(ws.q_f32)
            .arg_ptr(ws.k_f32)
            .arg_ptr(ws.v_f32)
            .arg_ptr(ws.chunk_gc)
            .arg_u32(c.heads as u32)
            .arg_u32(k as u32)
            .arg_u32(qkv as u32)
            .arg_f32(c.l2_eps)
            .arg_f32(c.gate_lower_bound)
            .arg_f32(1.0 / (d as f32).sqrt())
            .launch(stream)?;
        KernelLaunch::new(gpu, self.kernels.ff_tail)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state.conv)
            .arg_ptr(q)
            .arg_ptr(kp)
            .arg_ptr(v)
            .arg_u32(k as u32)
            .arg_u32(qkv as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(qkv as u32)
            .launch(stream)?;
        self.launch_tc_scan(gpu, k, state, ws, stream)
    }
}
