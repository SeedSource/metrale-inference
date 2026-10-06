// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: `METRALE_GLM_MTP_HEAD_NVFP4=1`: an NVFP4 copy of this rank's vocab shard of the
//! draft `lm_head`, read by the draft-head sweep in place of the FP8 copy.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Off (the default), `build` returns `None` without launching or allocating, and the head
//!   builds and reads its FP8 copy exactly as before.
//! - On, it applies only where the FP8 copy would have been built (`METRALE_GLM_MTP_HEAD_FP8`
//!   not `0`) and only on a vocab-sharded head (`head_n != vocab`): the FP8 copy and this one are
//!   read only on the sharded sweep, so an unsharded head reads the BF16 `lm_head` either way.
//! - The copy is quantized from the BF16 shard (`dense_fp8::quantize_nvfp4_copy`, the dense
//!   NVFP4 lever's MSE quantizer and layout), never from the FP8 copy. The BF16 `lm_head` is the
//!   target's own head and stays.
//! - When the copy is built, the FP8 copy is not: the draft head has exactly two readers,
//!   `forward_one` (1 row, the per-sequence propose) and `propose_batch_impl` (2..=16 rows,
//!   `MTP_BATCH_DRAFT_MAX`), both on the sharded sweep and both inside the NVFP4 tiers'
//!   1..=16 rows. The context prefill and catch-up (`rows_impl`) never run the head. Audited
//!   2026-10-06; a reader of more than 16 rows would need the FP8 copy back.
//! - The sweep (`dense_fp8::nv4_gemv_uncounted`) writes `[rows, n]` BF16 logits at row stride
//!   `n`, the layout and dtype the FP8 sweep writes and `argmax_bf16` / `argmax_bf16_batch` read.
//!   Each row's bits equal the one-row launch on it, so the batched propose still drafts what
//!   the per-sequence propose drafts. It allocates and synchronizes nothing.
//! - NOT byte-identical to the FP8 head. The draft head only proposes and the target's own
//!   head verifies, so emitted tokens are unchanged (greedy); the acceptance rate can move.

use super::*;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-10-06: `METRALE_GLM_MTP_HEAD_NVFP4=1` opts in; read once.
pub(super) fn mtp_head_nvfp4() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_MTP_HEAD_NVFP4").as_deref() == Ok("1"))
}

/// 2026-10-06: The NVFP4 copy of the `[head_n, hidden]` BF16 shard at `shard`, or `None` (the
/// FP8 path stays) when the lever is off, the FP8 head is turned off, the head is not
/// vocab-sharded, or the kernels or the quantization fail (logged; never fatal).
pub(super) fn build(
    gpu: &dyn GpuBackend,
    shard: DevicePtr,
    head_n: usize,
    vocab: usize,
    hidden: usize,
) -> Option<QuantizedWeight> {
    if !mtp_head_nvfp4() {
        return None;
    }
    if std::env::var("METRALE_GLM_MTP_HEAD_FP8").as_deref() == Ok("0") {
        tracing::warn!(
            "METRALE_GLM_MTP_HEAD_NVFP4=1 is ignored with METRALE_GLM_MTP_HEAD_FP8=0 (it replaces \
             the FP8 draft head)"
        );
        return None;
    }
    if head_n == vocab {
        tracing::warn!(
            "METRALE_GLM_MTP_HEAD_NVFP4=1 is ignored: the draft head is not vocab-sharded, so its \
             sweep reads the BF16 lm_head"
        );
        return None;
    }
    let s = gpu.default_stream();
    match crate::glm5next_layer::dense_fp8::quantize_nvfp4_copy(gpu, shard, head_n, hidden, s) {
        Ok(Some(q)) => {
            let mib = |b: usize| b / (1024 * 1024);
            let (nv4, fp8) = (
                head_n * hidden / 2 + head_n * hidden / 16,
                head_n * hidden + head_n * 4,
            );
            tracing::warn!(
                "METRALE_GLM_MTP_HEAD_NVFP4=1 - GLM MTP: draft lm_head shard quantised to NVFP4 \
                 ({head_n} rows x {hidden}, {} MB; the {} MB FP8 copy is not built, saves {} MB); \
                 1..=16-row draft sweeps read w4a16_gemv / w4a16_gemv_batch*; NOT byte-identical \
                 to the FP8 head (drafts only)",
                mib(nv4),
                mib(fp8),
                mib(fp8.saturating_sub(nv4)),
            );
            Some(q)
        }
        Ok(None) => {
            tracing::warn!(
                "METRALE_GLM_MTP_HEAD_NVFP4=1 but the NVFP4 kernels are unavailable or the shape \
                 ({head_n} x {hidden}) is not quantizable; keeping the FP8 draft head"
            );
            None
        }
        Err(e) => {
            tracing::warn!("GLM MTP: NVFP4 draft head unavailable ({e:#}); keeping the FP8 head");
            None
        }
    }
}

impl Glm5NextMtpHead {
    /// 2026-10-06: The sharded draft-head sweep on the NVFP4 copy: `logits[rows, n] = x[rows,
    /// hidden] @ shard^T`, `x` packed at stride `hidden`, `logits` at stride `n`. Returns
    /// `false` without launching when this sweep does not take it (no copy, not sharded, `n` not
    /// the shard, or `rows` outside 1..=16); the caller then runs its FP8 / BF16 sweep.
    pub(super) fn head_sweep_nv4(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        logits: DevicePtr,
        rows: usize,
        sharded: bool,
        n: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(q) = self
            .head_nv4
            .filter(|_| sharded && n == self.head_n && (1..=MTP_BATCH_DRAFT_MAX).contains(&rows))
        else {
            return Ok(false);
        };
        crate::glm5next_layer::dense_fp8::nv4_gemv_uncounted(
            gpu,
            x,
            &q,
            logits,
            rows,
            n,
            self.hidden,
            stream,
        )?;
        Ok(true)
    }
}
