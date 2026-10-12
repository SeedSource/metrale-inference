// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_LM_HEAD_NV4_SHARD=1`: an NVFP4 copy of this rank's vocab shard of the
//! GLM-5.3 target `lm_head`, read by the 1-row decode head and the 1..=16-row verify head in
//! place of the FP8 shard sweep (`lm_head_fp8_tc`).
//!
//! Owner: model-arch (GLM-5.3 target head); wired in `model-engine` `model/lm_head_nv4_shard.rs`.
//! Invariants:
//! - Off (the default), nothing here runs: no kernel is resolved, nothing is allocated.
//! - The quantized values equal what `METRALE_GLM_NUMERICS_FAKEQ=lm_head:nvfp4` (f9338cce)
//!   emulated: `quantize_nvfp4_copy` of the WHOLE `[vocab, hidden]` head has ONE FP32 tensor
//!   scale (global absmax / (6 * 448)) and every row is quantized independently of the others, so
//!   [`quantize_shard`] takes the global absmax over the whole BF16 head and quantizes only the
//!   shard's rows with that scale. A shard-only absmax would be a different tensor scale and
//!   different E4M3 block scales; the N0 quality evidence would not transfer.
//! - [`sweep`] writes `[rows, n]` BF16 logits at row stride `n` (the NVFP4 kernels have no
//!   output stride); the caller scatters them into the full-vocab logits.
//! - Tier by row count: rows < [`TC_FROM_M`] on the CUDA-core `w4a16_gemv` / `batch2` / `batch3`
//!   tiers (rows bit-identical across M), rows >= [`TC_FROM_M`] on the tensor-core tiers
//!   `w4a16_gemv_tc8` / `tc16` (flat in M; bits differ from the CUDA-core tiers), because the
//!   CUDA-core batch4+ tiers are compute-bound on GB10 (115-152 GB/s at M = 8 against 195-233 GB/s
//!   for FP8, dfnv4mt 2026-10-12). Without the tensor-core kernels the CUDA-core tiers serve.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::weight_map::QuantizedWeight;

use crate::glm5next_layer::{dense_fp8, dense_nv4_tc};

/// First row count that takes the tensor-core tier. PROVISIONAL: the CUDA-core batch4+ tiers are
/// compute-bound (dfnv4mt 2026-10-12, M = 8); the break-even against the CUDA-core tier at M = 4..=8
/// is not measured on this shape until `glm_lm_head_nv4_shard_microtest` runs on GB10.
pub const TC_FROM_M: usize = 4;

/// Largest row count routed to the NVFP4 shard; larger counts keep the FP8 shard sweep.
/// PROVISIONAL: 16 (every row count the NVFP4 tiers cover, `dense_fp8::NV4_MAX_M`) until the
/// microtest's per-M `old_us` / `new_us` rows show a row count where FP8 wins; then lower this to
/// the last M where NVFP4 is faster. Both copies are resident either way.
pub const ROUTE_MAX_M: usize = 16;

/// `METRALE_GLM_LM_HEAD_NV4_SHARD=1` opts in; read once.
pub fn enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_LM_HEAD_NV4_SHARD").as_deref() == Ok("1"))
}

/// Whether a `rows`-row head call takes the NVFP4 shard (see [`ROUTE_MAX_M`]).
pub fn routes(rows: usize) -> bool {
    (1..=ROUTE_MAX_M).contains(&rows)
}

/// The NVFP4 copy of rows `[begin, begin + len)` of the BF16 `[vocab, hidden]` head at `head`,
/// with the tensor scale of the WHOLE head (see the module invariants). `Ok(None)` when the NVFP4
/// quantizer kernels are missing or the shape is not quantizable (`hidden % 16 != 0`, empty
/// shard, shard outside the head). Synchronizes `stream`.
pub fn quantize_shard(
    gpu: &dyn GpuBackend,
    head: DevicePtr,
    vocab: usize,
    hidden: usize,
    begin: usize,
    len: usize,
    stream: u64,
) -> Result<Option<QuantizedWeight>> {
    if len == 0
        || hidden == 0
        || !hidden.is_multiple_of(16)
        || begin + len > vocab
        || head.is_null()
    {
        return Ok(None);
    }
    let (Ok(absmax), Ok(quant)) = (
        gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax"),
        gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_mse"),
    ) else {
        return Ok(None);
    };
    let total = vocab * hidden;
    let max_buf = gpu.alloc(4)?;
    gpu.memset(max_buf, 0, 4)?;
    KernelLaunch::new(gpu, absmax)
        .grid([(total / 256).clamp(1, 1024) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(head)
        .arg_ptr(max_buf)
        .arg_u32(total as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut b = [0u8; 4];
    gpu.copy_d2h(max_buf, &mut b)?;
    gpu.free(max_buf)?;
    let global_max = f32::from_le_bytes(b);
    // Same arithmetic as `quantize_to_nvfp4`: 6.0 = E2M1 max, 448.0 = E4M3 max.
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };
    let packed = gpu.alloc(len * hidden / 2)?;
    let scales = gpu.alloc(len * hidden / 16)?;
    KernelLaunch::new(gpu, quant)
        .grid([len as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(head.offset(begin * hidden * 2))
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_f32(scale2)
        .arg_u32(len as u32)
        .arg_u32(hidden as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(Some(QuantizedWeight {
        weight: packed,
        weight_scale: scales,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    }))
}

/// `logits[rows, n] = x[rows, k] @ shard^T` (BF16, row stride `n`), 1..=16 rows, tier by row
/// count (module invariants). No allocation or sync, so it is capture-safe.
#[allow(clippy::too_many_arguments)]
pub fn sweep(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    q: &QuantizedWeight,
    out: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    if rows >= TC_FROM_M && dense_nv4_tc::launch_tc(gpu, x, q, out, rows, n, k, stream)? {
        return Ok(());
    }
    dense_fp8::nv4_gemv_cuda_core(gpu, x, q, out, rows, n, k, stream)
}

/// [`sweep`] pinned to one tier, for the microtest: `tensor_core` runs the tc8/tc16 tier
/// (`Ok(false)` when it does not take the shape), else the CUDA-core tier.
#[allow(clippy::too_many_arguments)]
pub fn sweep_tier(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    q: &QuantizedWeight,
    out: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
    tensor_core: bool,
    stream: u64,
) -> Result<bool> {
    if tensor_core {
        return dense_nv4_tc::launch_tc(gpu, x, q, out, rows, n, k, stream);
    }
    dense_fp8::nv4_gemv_cuda_core(gpu, x, q, out, rows, n, k, stream)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_constants_are_consistent() {
        const { assert!(TC_FROM_M >= 1 && TC_FROM_M <= ROUTE_MAX_M) };
        const { assert!(ROUTE_MAX_M <= dense_fp8::NV4_MAX_M) };
        assert!(!routes(0) && routes(1) && routes(ROUTE_MAX_M) && !routes(ROUTE_MAX_M + 1));
    }
}
