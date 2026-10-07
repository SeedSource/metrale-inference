// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_GLM_NV4_BATCHM_STAGED`: the lever behind `w4a16_gemv_batch{4..8,16}_staged`.
//!
//! Owner: model-arch (GLM-5.3).
//! `METRALE_GLM_NV4_BATCHM_STAGED=1` routes the 4..=16-row NVFP4 GEMVs (`dense_fp8::nv4_gemv_uncounted`)
//! through the staged tiers (activation rows staged in shared memory per K window, next weight word
//! prefetched; same arithmetic order, so the same bits as `w4a16_gemv_batch{4..8,16}` and per row
//! as the M = 1 kernel). Read once. Gate `examples/glm5next_nv4_bm_staged_microtest.rs`.

use std::sync::{Once, OnceLock};

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// Rows of the six tiers, in `Nv4Kernels::tiers` order.
pub const TIER_ROWS: [u32; 6] = [4, 5, 6, 7, 8, 16];

/// Whether `METRALE_GLM_NV4_BATCHM_STAGED=1` is set (read once).
pub fn nv4_bm_staged() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_NV4_BATCHM_STAGED").as_deref() == Ok("1"))
}

/// The staged tiers, `None` when the loaded image lacks any of them.
pub fn load_staged(gpu: &dyn GpuBackend) -> Option<[KernelHandle; 6]> {
    let mut out = Vec::with_capacity(TIER_ROWS.len());
    for m in TIER_ROWS {
        out.push(
            gpu.kernel("w4a16_gemv", &format!("w4a16_gemv_batch{m}_staged"))
                .ok()?,
        );
    }
    out.try_into().ok()
}

/// Tier `i`: the staged kernel when the lever is on and the image has it (logged once), else `base`.
pub fn pick_tier(staged: Option<[KernelHandle; 6]>, i: usize, base: KernelHandle) -> KernelHandle {
    match staged {
        Some(s) if nv4_bm_staged() => {
            static ONCE: Once = Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "METRALE_GLM_NV4_BATCHM_STAGED=1: ENGAGED - 4..16-row NVFP4 dense GEMVs run \
                     w4a16_gemv_batch{{4..8,16}}_staged"
                );
            });
            s[i]
        }
        _ => base,
    }
}
