// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Speculative-decoding buffers for `TransformerModel::new`: the transposed
//! lm_head twin, the verify stashes and WY tables, the drafter's prompt-hidden
//! capture and the DFlash hidden-state capture.
//!
//! Owner: model-engine.
//! Invariants:
//! - A buffer whose feature is off is `DevicePtr::NULL` (or `None`), never a zero-byte allocation.

use std::sync::Arc;

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::MtpQuantization;
use metrale_model_layers::layers::ops::ModelLevers;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::QuantizedWeight;

/// 2026-09-25: Whether to build the transposed lm_head twin for the tile-GEMM
/// decode path: on unless `METRALE_NO_LMHEAD_TGEMM` is exactly `1`. Read once,
/// in `TransformerModel::new`, so changing it later has no effect.
fn lmhead_tgemm_enabled() -> bool {
    std::env::var("METRALE_NO_LMHEAD_TGEMM").ok().as_deref() != Some("1")
}

/// 2026-09-26: The padded transposed lm_head twin and its row stride, or `None`.
pub(super) fn build_lm_head_nvfp4_t(
    lm_head_nvfp4: &Option<QuantizedWeight>,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<(QuantizedWeight, u32)>> {
    Ok(match (lm_head_nvfp4, lmhead_tgemm_enabled()) {
        (Some(w), true) => {
            let (t, stride) =
                    metrale_model_layers::weight_map::QuantizedWeight::transpose_concat_for_gemm_padded(
                        gpu,
                        &[(w, config.vocab_size)],
                        config.hidden_size,
                        16,
                        128,
                    )?;
            // 2026-09-25: A padded stride is correct only on targets whose
            // `w4a16_gemm_t` takes `ldb`; a kernel without it steps rows by N
            // and reads sheared rows without faulting. When the stride equals
            // the vocab size, `ldb` changes nothing.
            if stride != config.vocab_size {
                tracing::warn!(target: "metrale_model_engine::model::impl_a1", "lm_head twin uses a PADDED stride ({} != vocab {}): this target's \
                     w4a16_gemm_t MUST accept the `ldb` argument, or decode at padded_n>=5 \
                     will read sheared rows. Disable with METRALE_NO_LMHEAD_TGEMM=1.",
                    stride,
                    config.vocab_size
                );
            }
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "lm_head transposed twin: vocab={} -> padded stride={} (vocab%16={}), tile GEMM active",
                config.vocab_size,
                stride,
                config.vocab_size % 16
            );
            Some((t, stride as u32))
        }
        _ => None,
    })
}

/// 2026-09-26: Returns `(mtp_hidden_save, verify_hidden_stash, verify_catchup_stash,
/// verify_wy_tables, gdn_woa_na_tab, mtp_catchup_ring)`.
pub(super) fn alloc_verify_buffers(
    config: &ModelConfig,
    proposer: &Option<Arc<dyn DraftProposer>>,
    levers: &ModelLevers,
    gpu: &dyn GpuBackend,
) -> Result<(
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
)> {
    let mtp_hidden_save = gpu.alloc(config.hidden_size * 4)?;
    // 2026-09-25: Batched-verify hidden stash, `[VERIFY_WY_TABLE_SEQS,
    // hidden_size]` BF16: one row per sequence of the widest batched
    // verify. NULL without an MTP proposer.
    let verify_hidden_stash = if proposer.is_some() {
        gpu.alloc(metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS * config.hidden_size * 2)?
    } else {
        DevicePtr::NULL
    };
    let verify_catchup_stash = if proposer.is_some() && levers.mtp_kv_exact {
        gpu.alloc(
            metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS
                * metrale_model_layers::layer::MTP_CATCHUP_MAX
                * config.hidden_size
                * 2,
        )?
    } else {
        DevicePtr::NULL
    };
    // 2026-09-25: Batched-verify WY pointer tables at a fixed address, so
    // captured CUDA graphs stay valid: one `VERIFY_WY_LAYER_STRIDE_BYTES`
    // slice per SSM layer. NULL without an MTP proposer or SSM layers.
    let verify_wy_tables = if proposer.is_some() && config.num_ssm_layers() > 0 {
        let bytes =
            config.num_ssm_layers() * metrale_model_layers::layer::VERIFY_WY_LAYER_STRIDE_BYTES;
        let buf = gpu.alloc(bytes)?;
        gpu.memset(buf, 0, bytes)?;
        buf
    } else {
        DevicePtr::NULL
    };
    let gdn_woa_na_tab = if verify_wy_tables.is_null() {
        DevicePtr::NULL
    } else {
        let b = gpu.alloc(metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS * 4)?;
        gpu.memset(b, 0, metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS * 4)?;
        b
    };
    // 2026-09-25: Allocated only when `mtp_catchup_enabled()`
    // (`METRALE_MTP_CATCHUP=1` outside multi-sequence MTP mode).
    let mtp_catchup_ring = if metrale_model_layers::speculative::mtp_catchup_enabled() {
        gpu.alloc(super::super::types::MTP_CATCHUP_RING_ROWS * config.hidden_size * 2)?
    } else {
        DevicePtr::NULL
    };
    Ok((
        mtp_hidden_save,
        verify_hidden_stash,
        verify_catchup_stash,
        verify_wy_tables,
        gdn_woa_na_tab,
        mtp_catchup_ring,
    ))
}

/// 2026-09-26: Returns `(capture_rows, mtp_prefill_hidden)`.
pub(super) fn alloc_prefill_capture(
    config: &ModelConfig,
    proposer: &Option<Arc<dyn DraftProposer>>,
    max_seq_len: usize,
    dflash_kgamma: usize,
    has_mtp: bool,
    mtp_quant_fwd: MtpQuantization,
    levers: &ModelLevers,
    mtp_quant: MtpQuantization,
    stage_rows: usize,
    gpu: &dyn GpuBackend,
) -> Result<(usize, DevicePtr)> {
    // 2026-09-25: Whole-prompt hidden capture buffer, `[rows, hidden_size]`
    // BF16, for the drafter prefill. It is allocated only when MTP is
    // active, the drafter-prefill lever is on, and the MTP head runs a BF16
    // forward (`supports_drafter_prefill`).
    //
    // Rows: the proposer's reachable context (`prefill_hidden_rows`, at
    // most `max_seq_len`), further capped at `dflash_ctx_cap()` when DFlash
    // is on (a cap of 0 means none). `capture_rows` is computed outside the
    // branch because it is also `mtp_prefill_capacity`, the bound the
    // capture guard checks.
    let mtp_prefill_rows = proposer
        .as_ref()
        .map_or(max_seq_len, |p| p.prefill_hidden_rows(max_seq_len))
        .min(max_seq_len);
    let capture_rows = if dflash_kgamma > 0 {
        let cap = metrale_model_arch::dflash_head::dflash_ctx_cap();
        if cap == 0 {
            mtp_prefill_rows
        } else {
            mtp_prefill_rows.min(cap)
        }
    } else {
        mtp_prefill_rows
    };
    // 2026-10-01: Under the chunked capture (`mtp_stage`, A59) the buffer is a staging
    // window of one arena (`stage_rows`), drained into the drafter after every chunk.
    let carry_on = metrale_model_layers::mtp_carry::mtp_carry_drafter_enabled(levers);
    let chunked = crate::model::mtp_stage::glm_mtp_chunked_capture(&config.model_type, carry_on);
    let capture_rows = if chunked {
        capture_rows.min(stage_rows)
    } else {
        capture_rows
    };
    let mtp_prefill_hidden = if has_mtp
        && mtp_quant_fwd.supports_drafter_prefill()
        && metrale_model_layers::layers::mtp_drafter_prefill_enabled(levers)
    {
        // 2026-09-25: The DFlash cap exists because the drafter cannot use
        // more prompt than its context window stores, so capture past it
        // is dead memory. Pure-MTP serves keep the full ceiling.
        let bytes = capture_rows * config.hidden_size * 2;
        tracing::info!(target: "metrale_model_engine::model::impl_a1", "MTP drafter context: allocating {:.0} MB prompt-hidden capture \
             ({} x {} BF16){}",
            bytes as f64 / 1e6,
            capture_rows,
            config.hidden_size,
            if chunked {
                " — chunked staging window (METRALE_GLM_MTP_CHUNKED_CAPTURE, A59)".to_string()
            } else if capture_rows < max_seq_len {
                format!(
                    " — capped from --max-seq-len {max_seq_len} to the reachable \
                     context (proposer ceiling A59 and/or the DFlash ctx cap)"
                )
            } else {
                String::new()
            },
        );
        gpu.alloc(bytes)?
    } else {
        if has_mtp
            && !mtp_quant_fwd.supports_drafter_prefill()
            && metrale_model_layers::layers::mtp_drafter_prefill_enabled(levers)
        {
            tracing::info!(target: "metrale_model_engine::model::impl_a1", "MTP drafter context: INACTIVE — the batched drafter prefill \
                 needs a BF16 MTP head (--mtp-quantization bf16); this head is \
                 {mtp_quant:?}. No prompt-hidden capture allocated.",
            );
        }
        DevicePtr::NULL
    };
    Ok((capture_rows, mtp_prefill_hidden))
}

/// 2026-09-26: Returns `(dflash_capture_layers, dflash_hidden_save_rows,
/// dflash_hidden_save)`.
pub(super) fn alloc_dflash_capture(
    config: &ModelConfig,
    dflash_kgamma: usize,
    max_batch_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(Vec<usize>, usize, Option<DevicePtr>)> {
    // 2026-09-25: DFlash hidden-state capture, allocated only when
    // `config.dflash_capture_layers` is non-empty (the factory fills it from
    // the drafter's target layer ids).
    let dflash_capture_layers: Vec<usize> = config.dflash_capture_layers.clone();
    // 2026-09-25: Row capacity of the capture buffer: `max_batch_size`
    // bands of `max(dflash_kgamma, 2)` rows. In a batched K=γ verify,
    // sequence `i` writes the band starting at row `i * dflash_kgamma`, and
    // the scheduler passes that row to `commit_ctx` as `scratch_row`;
    // single-sequence paths use band 0. `try_dflash_capture_all_at` clamps
    // its writes to this capacity.
    let dflash_hidden_save_rows = if dflash_capture_layers.is_empty() {
        0
    } else {
        dflash_kgamma.max(2) * max_batch_size.max(1)
    };
    let dflash_hidden_save = if dflash_capture_layers.is_empty() {
        None
    } else {
        let n = dflash_capture_layers.len();
        // 2026-09-25: Row-major; each row is `n_capture * hidden_size` BF16.
        Some(gpu.alloc(dflash_hidden_save_rows * n * config.hidden_size * 2)?)
    };
    Ok((
        dflash_capture_layers,
        dflash_hidden_save_rows,
        dflash_hidden_save,
    ))
}
