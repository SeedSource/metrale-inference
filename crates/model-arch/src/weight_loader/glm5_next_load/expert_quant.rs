// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The quantise-at-load arms of `bind_expert`: a deferred or
//! resident BF16 routed-expert projection becomes an NVFP4 triple.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Quantise one expert projection that was deferred: its BF16
/// bytes are read from the shard into host memory, and only the NVFP4 triple
/// is uploaded. A deferred tensor that is not 2-D BF16 is an error. The raw
/// bytes are dropped before quantising. `build_moe` binds only
/// `local_expert_range()`, so only this rank's experts are read.
pub(super) fn quantize_deferred_expert_proj(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    d: &metrale_model_weights::weights::DeferredTensor,
    base: &str,
) -> Result<Nvfp4Proj> {
    let [rows, cols] = d.shape[..] else {
        bail!(
            "{base}.weight must be 2-D to quantise, got shape {:?}",
            d.shape
        );
    };
    if d.dtype != WeightDtype::BF16 {
        bail!(
            "{base}.weight was deferred as {:?}; only a full-width BF16 expert is \
             quantised at load",
            d.dtype
        );
    }
    let bytes = d
        .read_host_bytes()
        .with_context(|| format!("{base}.weight: reading the deferred expert from its shard"))?;
    let values: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    drop(bytes);
    quantize_and_upload(gpu, store, base, &values, rows, cols)
}

/// 2026-09-25: Quantise one resident BF16 expert projection, read back from
/// the device. The source stays in the store until
/// [`Glm5NextWeightLoader::prune_after_load`] frees it.
pub(super) fn quantize_expert_proj(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    w: &WeightTensor,
    base: &str,
) -> Result<Nvfp4Proj> {
    let [rows, cols] = w.shape[..] else {
        bail!(
            "{base}.weight must be 2-D to quantise, got shape {:?}",
            w.shape
        );
    };
    let values = host_f32(gpu, w, &format!("{base}.weight"))?;
    quantize_and_upload(gpu, store, base, &values, rows, cols)
}

/// 2026-09-25: Encode `values` as NVFP4 and upload the packed codes and block
/// scales; both buffers are adopted by the store's derived-weight ledger, so
/// the store owns them.
fn quantize_and_upload(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    base: &str,
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4Proj> {
    let blob = nvfp4_quant::quantize_to_nvfp4(base, values, rows, cols)?;
    let packed = upload_bytes(gpu, store, &blob.packed)?;
    let scale = upload_bytes(gpu, store, &blob.scales)?;
    // 2026-10-03: A projection quantised at load has no calibrated activation scale.
    Ok(Nvfp4Proj {
        packed,
        scale,
        scale_2: blob.scale_2,
        input_scale: 0.0,
    })
}

/// 2026-09-25: The ledger label of every buffer `upload_bytes` adopts.
const QUANTIZED_EXPERT_LABEL: &str = "glm5_next routed expert, NVFP4 at load";

fn upload_bytes(gpu: &dyn GpuBackend, store: &WeightStore, b: &[u8]) -> Result<DevicePtr> {
    // 2026-10-06: Into the derived weight arena when this layer planned it
    // (`METRALE_GLM_WEIGHT_ARENA=1`, `expert_arena::plan_expert_layer`); else one allocation.
    if let Some(p) = store
        .derived()
        .upload_in_arena(gpu, QUANTIZED_EXPERT_LABEL, b)?
    {
        return Ok(p);
    }
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(b, p)?;
    store.derived().adopt(QUANTIZED_EXPERT_LABEL, p, b.len());
    Ok(p)
}
