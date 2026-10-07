// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `METRALE_GLM_NV4_B3_STAGED`: the lever behind `w4a16_gemv_batch3_staged`.
//!
//! Owner: model-arch (GLM-5.3).
//! `METRALE_GLM_NV4_B3_STAGED=1` routes the 3-row NVFP4 GEMV (`dense_fp8::nv4_gemv_uncounted`, M = 3)
//! through `w4a16_gemv_batch3_staged` (activations staged in shared memory, next weight word
//! prefetched; same arithmetic order, so the same bits as `w4a16_gemv_batch3`). Read once.
//! Serve A/B 2026-10-07 (race-dec-b3val-L12): texts identical, -0.99 ms/step.

use std::sync::{Once, OnceLock};

use metrale_gpu_runtime::gpu::KernelHandle;

/// Whether `METRALE_GLM_NV4_B3_STAGED=1` is set (read once).
pub fn nv4_b3_staged() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_NV4_B3_STAGED").as_deref() == Ok("1"))
}

/// The 3-row kernel: `staged` when the lever is on and the image has it (logged once), else `base`.
pub fn pick_b3(staged: Option<KernelHandle>, base: KernelHandle) -> KernelHandle {
    match staged {
        Some(h) if nv4_b3_staged() => {
            static ONCE: Once = Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "METRALE_GLM_NV4_B3_STAGED=1: ENGAGED - 3-row NVFP4 dense GEMVs run \
                     w4a16_gemv_batch3_staged"
                );
            });
            h
        }
        _ => base,
    }
}
