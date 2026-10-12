// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LM-head representation at load: the main head (pre-packed NVFP4, runtime NVFP4,
//! runtime or checkpoint FP8, or BF16) and the draft-only NVFP4 head for MTP.
//!
//! Owner: model-engine factory.
//! Invariants:
//! - `setup_lm_heads` returns `(lm_head_nvfp4, lm_head_fp8, mtp_lm_head_nvfp4)`; the first two
//!   are never both `Some`, and both are `None` for a BF16 main head.
//! - `mtp_lm_head_nvfp4` is `Some` only when `lm_head_nvfp4` is `None`.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::weight_map::{Fp8DenseWeight, quantize_to_fp8, quantize_to_nvfp4};

#[allow(clippy::type_complexity)]
pub(super) fn setup_lm_heads(
    store: &WeightStore,
    lm_head: &metrale_model_layers::weight_map::DenseWeight,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    use_speculative: bool,
    have_mtp_weights: bool,
) -> Result<(
    Option<metrale_model_layers::weight_map::QuantizedWeight>,
    Option<Fp8DenseWeight>,
    Option<metrale_model_layers::weight_map::QuantizedWeight>,
)> {
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    // 2026-09-25: A U8 lm_head weight is already NVFP4-packed. Quantizing it as BF16 would read
    // `vocab * hidden` BF16 values from a smaller packed buffer, so it is bound as-is below.
    let lm_head_key = [
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ]
    .into_iter()
    .find(|k| store.contains(k));
    let lm_head_prepacked_nvfp4 = lm_head_key
        .and_then(|k| store.get(k).ok())
        .is_some_and(|w| w.dtype == metrale_model_weights::weights::WeightDtype::UInt8);

    // 2026-10-12: `METRALE_GLM_NUMERICS_FAKEQ`: the `lm_head` family fake-quantizes the BF16 head in
    // place, which only makes sense when the head stays BF16 (`--lm-head-dtype bf16`); any other
    // head would be quantized a second time below, so it fails the load. By now every layer and
    // the MTP block have been through the other families, so the per-family summary prints here.
    // Unset: `validate` and `fakeq` return at once and nothing else runs.
    {
        use metrale_model_arch::glm5next_numerics as nq;
        nq::validate()?;
        if !nq::fakeq().is_empty() {
            let kind = if lm_head_prepacked_nvfp4 {
                nq::HeadKind::PrepackedNvfp4
            } else if config.skip_lm_head_quantization() {
                nq::HeadKind::Bf16
            } else if config.lm_head_fp8 {
                nq::HeadKind::Fp8
            } else {
                nq::HeadKind::Nvfp4
            };
            if config.tie_word_embeddings && nq::lists(nq::Family::LmHead) {
                anyhow::bail!(
                    "METRALE_GLM_NUMERICS_FAKEQ lists lm_head, but this model ties lm_head to the                      embedding; the in-place fake-quant would change the embedding too"
                );
            }
            nq::apply_lm_head(gpu, lm_head.weight, config.vocab_size, config.hidden_size, kind)?;
            nq::log_summary();
        }
    }

    let mut lm_head_fp8: Option<Fp8DenseWeight> = None;
    let lm_head_nvfp4 = if lm_head_prepacked_nvfp4 {
        // 2026-09-25: No BF16 lm_head exists in the checkpoint, so a BF16 or FP8 request cannot
        // be honoured; it is warned about and the packed head is used.
        if config.skip_lm_head_quantization() || config.lm_head_fp8 {
            tracing::warn!(
                "--lm-head-dtype override ignored: this checkpoint ships lm_head                  pre-packed as NVFP4 (no BF16 tensor exists to keep or requantize);                  using the packed NVFP4 head"
            );
        }
        let prefix = lm_head_key.unwrap().strip_suffix(".weight").unwrap();
        let q = metrale_model_layers::weight_map::quantized(store, prefix, gpu)?;
        tracing::info!(
            "LM head loaded as pre-packed NVFP4 (vocab={}, skipped requantize)",
            config.vocab_size
        );
        Some(q)
    } else if config.skip_lm_head_quantization() {
        tracing::info!("LM head kept as BF16 (skip NVFP4 quantization per model config)");
        None
    } else if config.lm_head_fp8 {
        // 2026-09-25: Prefer the checkpoint's own FP8 lm_head. Quantizing the BF16 dequant of
        // it again would be a lossy FP8 -> BF16 -> FP8 round trip and a second copy. Otherwise
        // quantize the BF16 head to FP8 E4M3 with a per-row f32 scale (`quantize_to_fp8`).
        // The share may have more rows than `vocab_size`; the head is row-major `[rows, hidden]`
        // and its readers project `config.vocab_size` rows, so the extra rows are never read.
        let native = native_fp8_lm_head_share(store, config, gpu)?;
        let q = if let Some((shared, rows)) = native {
            tracing::info!(
                "LM head served from the checkpoint's NATIVE FP8 (w8a16, vocab={}, \
                 tensor rows={rows}) — no requantize, no second copy",
                config.vocab_size
            );
            shared
        } else {
            let quantize_fp8_k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
            let q = quantize_to_fp8(
                lm_head,
                config.vocab_size,
                config.hidden_size,
                gpu,
                quantize_fp8_k,
                stream,
            )?;
            tracing::info!(
                "LM head quantized to FP8 (w8a16, vocab={}) — checkpoint is not \
                 natively FP8, so this is a runtime mirror",
                config.vocab_size
            );
            q
        };
        lm_head_fp8 = Some(q);
        None
    } else {
        let q = quantize_to_nvfp4(
            lm_head,
            config.vocab_size,
            config.hidden_size,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        tracing::info!("LM head quantized to NVFP4 (vocab={})", config.vocab_size);
        Some(q)
    };

    // 2026-09-25: The MTP proposer's vocab projection is an NVFP4 `QuantizedWeight`
    // (`MtpHead::lm_head_nvfp4`). When the main head is not NVFP4 and speculative decoding with
    // an MTP head is active, build a separate NVFP4 copy used only for drafting. When the main
    // head is NVFP4 this stays `None` and `TransformerModel::new` drafts with the main head.
    let mtp_lm_head_nvfp4 = if lm_head_nvfp4.is_none() && use_speculative && have_mtp_weights {
        let q = quantize_to_nvfp4(
            lm_head,
            config.vocab_size,
            config.hidden_size,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        tracing::info!(
            "Draft-only NVFP4 LM head built for MTP (main head stays BF16, vocab={})",
            config.vocab_size,
        );
        Some(q)
    } else {
        None
    };
    Ok((lm_head_nvfp4, lm_head_fp8, mtp_lm_head_nvfp4))
}

/// 2026-09-25: An `Fp8DenseWeight` that views the checkpoint's own FP8 E4M3 lm_head, and its row
/// count. Used for the main FP8 head and for the DFlash drafter tail.
///
/// The weight pointer is the store's tensor, not a copy; the per-row BF16 scale is converted to
/// a new f32 buffer. The local unsloth/Qwen3.8-27B-NVFP4 checkpoint has this layout:
/// `lm_head.weight` F8_E4M3 `[248320, 5120]` and `lm_head.weight_scale` BF16 `[248320, 1]`.
///
/// Returns `Ok(None)` when there is no lm_head, it is not FP8 E4M3, its shape is not
/// `[rows >= vocab_size, hidden_size]`, or the scale is missing or not one BF16 value per row.
pub(super) fn native_fp8_lm_head_share(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<(Fp8DenseWeight, usize)>> {
    use metrale_model_weights::weights::WeightDtype;
    let Some(key) = [
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ]
    .into_iter()
    .find(|k| store.contains(k)) else {
        return Ok(None);
    };
    let w = store.get(key)?;
    if w.dtype != WeightDtype::FP8E4M3 {
        return Ok(None);
    }
    // 2026-09-25: Rows may exceed `vocab_size`, so the row count is returned for the caller to
    // check; the DFlash drafter requires it to equal its own vocab (`from_weights`).
    if w.shape.len() != 2 || w.shape[1] != config.hidden_size || w.shape[0] < config.vocab_size {
        tracing::warn!(
            "native FP8 lm_head share declined: shape {:?} vs hidden {} / vocab >= {}",
            w.shape,
            config.hidden_size,
            config.vocab_size
        );
        return Ok(None);
    }
    let rows = w.shape[0];
    let scale_key = format!("{key}_scale");
    let Ok(s) = store.get(&scale_key) else {
        return Ok(None);
    };
    if s.dtype != WeightDtype::BF16 || s.num_elements() != rows {
        tracing::warn!(
            "native FP8 lm_head share declined: {scale_key} dtype {:?} shape {:?} \
             is not a per-row BF16 scale",
            s.dtype,
            s.shape
        );
        return Ok(None);
    }
    // 2026-09-25: BF16 -> f32 on the host, once at load: `rows * 2` bytes down, `rows * 4` up.
    let n = rows;
    let mut host_bf16 = vec![0u8; n * 2];
    gpu.copy_d2h(s.ptr, &mut host_bf16)?;
    let host_f32: Vec<u8> = host_bf16
        .chunks_exact(2)
        .flat_map(|c| {
            let bits = (u16::from_le_bytes([c[0], c[1]]) as u32) << 16;
            f32::from_bits(bits).to_le_bytes()
        })
        .collect();
    let row_scale = gpu.alloc(n * 4)?;
    gpu.copy_h2d(&host_f32, row_scale)?;
    tracing::info!(
        "Native FP8 lm_head share ready for the DFlash drafter tail \
         ([{rows} x {}] E4M3 + per-row scale; skips the 1.27 GB runtime mirror)",
        config.hidden_size
    );
    Ok(Some((
        Fp8DenseWeight {
            weight: w.ptr,
            row_scale,
        },
        rows,
    )))
}
