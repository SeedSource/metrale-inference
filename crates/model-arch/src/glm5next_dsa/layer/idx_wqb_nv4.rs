// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_IDX_WQB_NV4=1` (default off): the DSA indexer's `wq_b`
//! (`[index_heads * index_head_dim, q_lora_rank]` = 4096 x 1536, TP-replicated) reads an NVFP4
//! copy on decode and verify rows instead of the BF16 weight.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - The copy is made at load by `dense_fp8::quantize_nvfp4_copy`, the quantizer the
//!   `METRALE_GLM_NUMERICS_FAKEQ` family `wq_b:nvfp4` used (f9338cce), so the quantized values
//!   are the ones the N0 falsifier measured. The BF16 weight is KEPT: prefill (`wide.rs`,
//!   `row_batch.rs`, `rows.rs` batched selector, cuBLASLt arms) still reads it. Extra memory per
//!   layer: `n k / 2 + n k / 16` = 3.54 MB at 4096 x 1536.
//! - The copy sits in a registry keyed by the BF16 weight's device address, so the struct
//!   `Glm5NextDsaWeights` is unchanged. Lever off: no kernel resolved, no allocation, no lookup.
//! - Output is FP32 (`kernels/gb10/common/w4a16_gemv_f32out.cu`), the type the selector reads. The
//!   CUDA-core kernels keep the production NVFP4 GEMVs' accumulation order, so BF16(output) is
//!   bit-identical to their BF16 output (gate `glm_idx_wqb_nv4_microtest`); the tensor-core pair
//!   does the same against `w4a16_gemv_tc8` / `tc16`.
//! - Sites routed (decode / verify rows): `indexer_project_rows` (single-sequence
//!   `IDX_ROWBATCH` and the cross-sequence group), `indexer_qidx_one` (CeilingRows without a
//!   precomputed indexer) and `select_row_at` (the per-row loop). Rows past 16 are launched in
//!   chunks of 16. `METRALE_GLM_DSA_GEMV_SPLIT` does not apply to the NVFP4 launches.
//! - 2026-10-12 PROVISIONAL (no measurement yet; `glm_idx_wqb_nv4_microtest` prints the
//!   per-M winner): which kernel a chunk of `m` rows takes is [`wqb_route`], from
//!   [`BF16_MAX_M`], [`CORE_MAX_M`]. CUDA-core NVFP4 batch GEMVs are compute-bound at M >= 4
//!   (115-152 GB/s against 195-233 GB/s for FP8 rowscale at M = 8, dfnv4mt 2026-10-12); the
//!   tensor-core kernel does the same work per weight byte at every M.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::weight_map::QuantizedWeight;

use super::row_batch::batchm_rows;
use super::{Glm5NextDsaLayer, gemm};

/// Most rows one NVFP4 launch takes (`w4a16_gemv_f32out_b16`, `w4a16_gemv_tc16_f32out`).
pub const NV4_MAX_M: usize = 16;
/// 2026-10-12 PROVISIONAL: chunks of 1..=this many rows keep the BF16 FP32-out GEMV. 0 = none.
pub const BF16_MAX_M: usize = 0;
/// 2026-10-12 phcmt2 (race-lead: as written): chunks of `BF16_MAX_M + 1..=this` rows take the CUDA-core NVFP4
/// kernel (bit-identical per row to the production NVFP4 tiers); larger chunks take the
/// tensor-core kernel (needs `K % 128 == 0`, else CUDA-core).
pub const CORE_MAX_M: usize = 9;

/// Which kernel a chunk of rows takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WqbRoute {
    Bf16,
    Core,
    Tc,
}

/// 2026-10-12: The kernel for a chunk of `m` rows (`1..=16`; anything else is BF16) at inner size
/// `k`, from the PROVISIONAL constants above.
#[allow(clippy::absurd_extreme_comparisons)] // BF16_MAX_M is 0 until the microtest says otherwise
pub fn wqb_route(m: usize, k: usize) -> WqbRoute {
    if m == 0 || m > NV4_MAX_M || m <= BF16_MAX_M {
        WqbRoute::Bf16
    } else if m <= CORE_MAX_M || !k.is_multiple_of(128) {
        WqbRoute::Core
    } else {
        WqbRoute::Tc
    }
}

/// `METRALE_GLM_IDX_WQB_NV4=1` (read once; any other value is off).
pub fn idx_wqb_nv4() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_IDX_WQB_NV4").as_deref() == Ok("1"))
}

#[derive(Clone, Copy)]
struct Kernels {
    /// `w4a16_gemv_f32out_b{1,2,3,4,5,6,7,8,16}`.
    core: [KernelHandle; 9],
    tc8: KernelHandle,
    tc16: KernelHandle,
}

fn kernels(gpu: &dyn GpuBackend) -> Option<Kernels> {
    static K: OnceLock<Option<Kernels>> = OnceLock::new();
    *K.get_or_init(|| {
        let r = (|| -> Result<Kernels> {
            let k = |n: &str| gpu.kernel("w4a16_gemv_f32out", n);
            Ok(Kernels {
                core: [
                    k("w4a16_gemv_f32out_b1")?,
                    k("w4a16_gemv_f32out_b2")?,
                    k("w4a16_gemv_f32out_b3")?,
                    k("w4a16_gemv_f32out_b4")?,
                    k("w4a16_gemv_f32out_b5")?,
                    k("w4a16_gemv_f32out_b6")?,
                    k("w4a16_gemv_f32out_b7")?,
                    k("w4a16_gemv_f32out_b8")?,
                    k("w4a16_gemv_f32out_b16")?,
                ],
                tc8: k("w4a16_gemv_tc8_f32out")?,
                tc16: k("w4a16_gemv_tc16_f32out")?,
            })
        })();
        match r {
            Ok(k) => Some(k),
            Err(e) => {
                tracing::warn!(
                    "METRALE_GLM_IDX_WQB_NV4=1: NOT ENGAGED (w4a16_gemv_f32out kernels: {e:#})"
                );
                None
            }
        }
    })
}

fn registry() -> &'static RwLock<HashMap<u64, QuantizedWeight>> {
    static R: OnceLock<RwLock<HashMap<u64, QuantizedWeight>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(HashMap::new()))
}

static COPY_BYTES: AtomicU64 = AtomicU64::new(0);
static COPIES: AtomicU64 = AtomicU64::new(0);

/// The core kernel for `m` rows: b1, b2, b3, b4..=b8, then b16 for 9..=16.
fn core_index(m: usize) -> usize {
    match m {
        1..=8 => m - 1,
        _ => 8,
    }
}

/// Launches the FP32-out NVFP4 kernel for `route` (never `Bf16`) on `m` rows.
#[allow(clippy::too_many_arguments)]
fn launch_nv4(
    gpu: &dyn GpuBackend,
    kn: &Kernels,
    route: WqbRoute,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let (h, cols) = match route {
        WqbRoute::Core => (kn.core[core_index(m)], 4u32),
        WqbRoute::Tc if m <= 8 => (kn.tc8, 8),
        WqbRoute::Tc => (kn.tc16, 16),
        WqbRoute::Bf16 => bail!("launch_nv4: BF16 route"),
    };
    KernelLaunch::new(gpu, h)
        .grid([(n as u32).div_ceil(cols), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(q.weight)
        .arg_ptr(q.weight_scale)
        .arg_f32(q.weight_scale_2)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

impl Glm5NextDsaLayer {
    /// 2026-10-12: `METRALE_GLM_IDX_WQB_NV4=1`: quantize this layer's `wq_b` to NVFP4 (the
    /// BF16 weight stays) and register the copy. No-op with the lever off.
    pub fn register_wq_b_nv4(&self, gpu: &dyn GpuBackend) -> Result<()> {
        if !idx_wqb_nv4() {
            return Ok(());
        }
        let (n, k) = (
            self.cfg.index_heads * self.cfg.index_head_dim,
            self.cfg.q_lora_rank,
        );
        if kernels(gpu).is_none() {
            return Ok(());
        }
        let Some(q) = crate::glm5next_layer::dense_fp8::quantize_nvfp4_copy(
            gpu,
            self.weights.wq_b,
            n,
            k,
            gpu.default_stream(),
        )?
        else {
            tracing::warn!(
                "METRALE_GLM_IDX_WQB_NV4=1: layer {} wq_b [{n}, {k}] not quantized (NVFP4 \
                 quantizer kernels missing or shape unsupported); it keeps BF16",
                self.layer_idx
            );
            return Ok(());
        };
        registry().write().unwrap().insert(self.weights.wq_b.0, q);
        let bytes = (n * k / 2 + n * k / 16) as u64;
        let total = COPY_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let count = COPIES.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::info!(
            "METRALE_GLM_IDX_WQB_NV4=1: layer {} wq_b [{n}, {k}] NVFP4 copy {:.2} MB (BF16 kept for \
             prefill); {count} copies, {:.1} MB total",
            self.layer_idx,
            bytes as f64 / 1e6,
            total as f64 / 1e6
        );
        Ok(())
    }

    /// 2026-10-12: `out[r] = wq_b(a[r])` (FP32) for `rows` contiguous rows of `a`
    /// (`[rows, q_lora_rank]` BF16) on the NVFP4 copy, in launches of at most 16 rows, each on
    /// the kernel [`wqb_route`] picks. `Ok(false)` launches nothing (lever off, no copy for this
    /// layer, kernels missing): the caller runs its BF16 path. `f32k` is the BF16 FP32-out
    /// batched GEMV a BF16-routed chunk of 2+ rows uses (the sites that pass one row may pass
    /// `KernelHandle(0)`).
    pub(super) fn wq_b_nv4_rows(
        &self,
        gpu: &dyn GpuBackend,
        f32k: KernelHandle,
        a: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<bool> {
        if !idx_wqb_nv4() || rows == 0 {
            return Ok(false);
        }
        let Some(q) = registry()
            .read()
            .unwrap()
            .get(&self.weights.wq_b.0)
            .copied()
        else {
            return Ok(false);
        };
        let Some(kn) = kernels(gpu) else {
            return Ok(false);
        };
        let (n, k) = (
            self.cfg.index_heads * self.cfg.index_head_dim,
            self.cfg.q_lora_rank,
        );
        let mut r0 = 0;
        while r0 < rows {
            let m = NV4_MAX_M.min(rows - r0);
            let (a_r, c_r) = (a.offset(r0 * k * 2), out.offset(r0 * n * 4));
            match wqb_route(m, k) {
                WqbRoute::Bf16 if m == 1 => gemm(
                    gpu,
                    self.kernels.gemm_f32,
                    self.kernels.gemv_f32,
                    KernelHandle(0),
                    a_r,
                    self.weights.wq_b,
                    c_r,
                    1,
                    n,
                    k,
                    stream,
                )?,
                WqbRoute::Bf16 => {
                    if f32k.0 == 0 {
                        bail!(
                            "wq_b: BF16-routed chunk of {m} rows without an FP32-out batched GEMV"
                        );
                    }
                    batchm_rows(gpu, f32k, 4, a_r, self.weights.wq_b, c_r, m, n, k, stream)?
                }
                route => launch_nv4(gpu, &kn, route, a_r, &q, c_r, m, n, k, stream)?,
            }
            r0 += NV4_MAX_M;
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "METRALE_GLM_IDX_WQB_NV4=1: ENGAGED (first: {rows} rows; decode / verify wq_b on the \
                 NVFP4 copy, FP32 out; PROVISIONAL map: BF16 <= {BF16_MAX_M}, CUDA-core <= \
                 {CORE_MAX_M}, tensor-core above)"
            )
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::absurd_extreme_comparisons)]
    fn route_map_is_a_partition_of_1_to_16() {
        for m in 1..=16 {
            let r = wqb_route(m, 1536);
            if m <= BF16_MAX_M {
                assert_eq!(r, WqbRoute::Bf16);
            } else if m <= CORE_MAX_M {
                assert_eq!(r, WqbRoute::Core);
            } else {
                assert_eq!(r, WqbRoute::Tc);
            }
        }
        assert_eq!(wqb_route(0, 1536), WqbRoute::Bf16);
        assert_eq!(wqb_route(17, 1536), WqbRoute::Bf16);
        // The tensor-core kernel needs K % 128 == 0.
        assert_eq!(wqb_route(8, 1500), WqbRoute::Core);
        assert_eq!(
            wqb_route(8, 1536),
            if CORE_MAX_M >= 8 {
                WqbRoute::Core
            } else {
                WqbRoute::Tc
            }
        );
    }

    #[test]
    fn core_kernel_index_matches_the_tiers() {
        assert_eq!(core_index(1), 0);
        assert_eq!(core_index(3), 2);
        assert_eq!(core_index(8), 7);
        assert_eq!(core_index(9), 8);
        assert_eq!(core_index(16), 8);
    }

    #[test]
    fn every_site_is_gated_and_prefill_keeps_bf16() {
        let ib = include_str!("idx_rowbatch.rs");
        assert_eq!(ib.matches("wq_b_nv4_rows(").count(), 2);
        let rows = include_str!("rows.rs");
        assert_eq!(rows.matches("wq_b_nv4_rows(").count(), 1);
        // Prefill / wide arms never route to NVFP4.
        for f in [include_str!("wide.rs"), include_str!("row_batch.rs")] {
            assert!(!f.contains("wq_b_nv4_rows("));
        }
    }
}
