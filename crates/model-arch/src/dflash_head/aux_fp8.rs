// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_DFLASH_AUX_FP8=1` (default off): FP8 row-scaled mirrors of the
//! drafter's BF16 projections outside the seven layer GEMMs, on the FP8 GEMV ladder the
//! layer GEMMs use.
//!
//! Why: in the TP2 C=1 propose these four still read BF16 through
//! `dense_gemv_bf16_batchm`, 1.26 ms of a ~8.2 ms window (rank-0 nsys, cardf arm,
//! PROPOSE-SPLIT 2026-10-12):
//! - `attention_conv.kernel_projection` / `mlp_conv.kernel_projection`
//!   `[2 * conv_kernel_size * hidden / conv_group_size, hidden]` per layer (`conv_prepare`);
//! - `fc` `[hidden (this rank's fc_out under TP), n_targets * target_hidden]`
//!   (`fc_project`);
//! - the fused ctx K/V `[layers * 2 * kv_dim, hidden]` (`precompute_ctx_kv`);
//! - `candidate_selector.hidden_projection` `[selector_rank, hidden]` (`dflash2_walk_block`).
//!
//! Same quantizer and layout as the layer mirrors (`DenseWeight::quantize_to_fp8`: E4M3,
//! one f32 scale per output row) and the same kernels (`fp8_gemv_rowscale_batch8_rt2` up to
//! 8 rows, `_batch16_rt2` up to 16). Above 16 rows the conv and selector projections take
//! `fp8_gemm_n128_row_scaled` (their BF16 copies are freed, see
//! `release_bf16_layer_weights`); `fc` and the fused K/V keep their BF16 copies and their
//! BF16 GEMM for those (prompt-sized ctx precompute chunks, and `fc` is also read by the
//! non-paged `block_ctx_project`).
//!
//! Numerics: drafter logits only. Every draft is verified by the target, so served tokens
//! do not change; drafts (acceptance) may.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Armed only on an `Fp8Weights` head (the FP8 kernels and the quantizer are present).
//! - A mirror is `None` unless the lever armed it; `None` routes to the BF16 call as before.

use std::sync::Once;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight};

use super::{BlockDiffusionDraftHead, DflashQuantization};

/// 2026-10-12: The engaged line, printed once per process (one per rank).
static ENGAGED: Once = Once::new();

/// 2026-10-12: The kernel one auxiliary projection takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AuxRoute {
    /// 2026-10-12: `fp8_gemv_rowscale_batch8_rt2` (1..=8 rows).
    Rt2x8,
    /// 2026-10-12: `fp8_gemv_rowscale_batch16_rt2` (9..=16 rows).
    Rt2x16,
    /// 2026-10-12: `fp8_gemm_n128_row_scaled`, any row count.
    RowScaled,
    /// 2026-10-12: The BF16 `draft_gemm_bf16`, as without the lever.
    Bf16,
}

/// 2026-10-12: Route a projection of `m` rows and depth `k`. `mirror`: the FP8 copy exists;
/// `bf16_kept`: the BF16 copy stays live, so rows the GEMVs do not cover go back to BF16;
/// `rt2`/`rt2_16`: those kernels resolved; `rt_on`: `fp8_rt_enabled()`.
pub(super) fn aux_route(
    mirror: bool,
    bf16_kept: bool,
    (rt2, rt2_16, rt_on): (bool, bool, bool),
    m: u32,
    k: u32,
) -> AuxRoute {
    if !mirror {
        return AuxRoute::Bf16;
    }
    if rt_on && k.is_multiple_of(16) && m >= 1 {
        if rt2 && m <= 8 {
            return AuxRoute::Rt2x8;
        }
        if rt2_16 && m <= 16 {
            return AuxRoute::Rt2x16;
        }
    }
    if bf16_kept {
        AuxRoute::Bf16
    } else {
        AuxRoute::RowScaled
    }
}

/// 2026-10-12: One mirrored tensor: name, `[n, k]`, how many (per layer or one), and
/// whether its BF16 copy is freed once the mirror exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AuxShape {
    pub name: &'static str,
    pub n: usize,
    pub k: usize,
    pub count: usize,
    pub replaces_bf16: bool,
}

impl AuxShape {
    /// 2026-10-12: Bytes of all `count` FP8 mirrors (1 B per weight + one f32 per row).
    pub fn fp8_bytes(&self) -> usize {
        self.count * (self.n * self.k + 4 * self.n)
    }

    /// 2026-10-12: Bytes of the BF16 tensors they mirror.
    pub fn bf16_bytes(&self) -> usize {
        self.count * self.n * self.k * 2
    }
}

/// 2026-10-12: Inputs of [`aux_shapes`]: this rank's geometry.
#[derive(Clone, Copy, Debug)]
pub(super) struct AuxGeom {
    pub hidden: usize,
    pub layers: usize,
    pub conv_kernel_size: usize,
    pub conv_group_size: usize,
    pub selector_rank: usize,
    pub fc_rows: usize,
    pub fc_in: usize,
    pub kv_dim: usize,
}

/// 2026-10-12: The mirrors the lever builds for geometry `g`, in arm order.
pub(super) fn aux_shapes(g: AuxGeom) -> Vec<AuxShape> {
    let mut v = Vec::with_capacity(5);
    if g.conv_kernel_size > 0 && g.conv_group_size > 0 {
        let dyn_cols = 2 * g.conv_kernel_size * (g.hidden / g.conv_group_size);
        for name in ["attention_conv_proj", "mlp_conv_proj"] {
            v.push(AuxShape {
                name,
                n: dyn_cols,
                k: g.hidden,
                count: g.layers,
                replaces_bf16: true,
            });
        }
    }
    v.push(AuxShape {
        name: "fc",
        n: g.fc_rows,
        k: g.fc_in,
        count: 1,
        replaces_bf16: false,
    });
    v.push(AuxShape {
        name: "fused_kv",
        n: g.layers * 2 * g.kv_dim,
        k: g.hidden,
        count: 1,
        replaces_bf16: false,
    });
    if g.selector_rank > 0 {
        v.push(AuxShape {
            name: "selector_hidden_proj",
            n: g.selector_rank,
            k: g.hidden,
            count: 1,
            replaces_bf16: true,
        });
    }
    v
}

impl BlockDiffusionDraftHead {
    /// 2026-10-12: This head's [`AuxGeom`] (local dims under TP).
    fn aux_geom(&self) -> AuxGeom {
        AuxGeom {
            hidden: self.hidden_size,
            layers: self.layers.len(),
            conv_kernel_size: self.conv_kernel_size,
            conv_group_size: self.conv_group_size,
            selector_rank: if self.selector_hidden_proj.is_some() {
                self.selector_rank
            } else {
                0
            },
            fc_rows: self.tp.as_ref().map_or(self.hidden_size, |t| t.fc_len),
            fc_in: self.target_layer_ids.len() * self.target_hidden_size,
            kv_dim: self.num_kv_heads * self.head_dim,
        }
    }

    /// 2026-10-12: Build the mirrors under `METRALE_DFLASH_AUX_FP8=1` on an `Fp8Weights`
    /// head; otherwise warn (lever on) or do nothing.
    pub(super) fn arm_aux_fp8(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !self.levers.aux_fp8 {
            return Ok(());
        }
        if self.quant != DflashQuantization::Fp8Weights || self.kernels.quantize_bf16_to_fp8.0 == 0
        {
            tracing::warn!(
                "METRALE_DFLASH_AUX_FP8=1: IDLE (the drafter is not on FP8 weights; needs \
                 METRALE_DFLASH_DRAFTER_FP8 and its kernels): conv, fc, fused K/V and selector \
                 projections stay BF16"
            );
            return Ok(());
        }
        let quant_k = self.kernels.quantize_bf16_to_fp8;
        let stream = 0u64;
        let g = self.aux_geom();
        let shapes = aux_shapes(g);
        let shape = |name: &str| *shapes.iter().find(|s| s.name == name).expect("aux shape");
        if g.conv_kernel_size > 0 && g.conv_group_size > 0 {
            let s = shape("attention_conv_proj");
            for layer in self.layers.iter_mut() {
                if let Some(w) = layer.attention_conv_proj.as_ref() {
                    layer.attention_conv_proj_fp8 =
                        Some(w.quantize_to_fp8(gpu, quant_k, s.n, s.k, stream)?);
                }
                if let Some(w) = layer.mlp_conv_proj.as_ref() {
                    layer.mlp_conv_proj_fp8 =
                        Some(w.quantize_to_fp8(gpu, quant_k, s.n, s.k, stream)?);
                }
            }
        }
        let s = shape("fc");
        self.fc_fp8 = Some(self.fc.quantize_to_fp8(gpu, quant_k, s.n, s.k, stream)?);
        if let Some(fused) = self.fused_kv_weight {
            let s = shape("fused_kv");
            self.fused_kv_fp8 = Some(
                DenseWeight { weight: fused }.quantize_to_fp8(gpu, quant_k, s.n, s.k, stream)?,
            );
        }
        if let Some(w) = self.selector_hidden_proj.as_ref() {
            let s = shape("selector_hidden_proj");
            self.selector_hidden_proj_fp8 =
                Some(w.quantize_to_fp8(gpu, quant_k, s.n, s.k, stream)?);
        }
        let mirror_mb = shapes.iter().map(AuxShape::fp8_bytes).sum::<usize>() as f64 / 1e6;
        let freed_mb = shapes
            .iter()
            .filter(|s| s.replaces_bf16)
            .map(AuxShape::bf16_bytes)
            .sum::<usize>() as f64
            / 1e6;
        let list: Vec<String> = shapes
            .iter()
            .map(|s| format!("{}x{} [{}x{}]", s.name, s.count, s.n, s.k))
            .collect();
        let rank = self.tp.as_ref().map_or(0, |t| t.rank());
        ENGAGED.call_once(|| {
            tracing::warn!(
                "METRALE_DFLASH_AUX_FP8=1: ENGAGED (rank {rank}: {}; +{mirror_mb:.1} MB FP8 \
                 mirrors, {freed_mb:.1} MB BF16 conv/selector freed after load; <=16 rows on \
                 fp8_gemv_rowscale rt2)",
                list.join(", ")
            );
        });
        Ok(())
    }

    /// 2026-10-12: `C[m, n] = A[m, k] @ W[n, k]^T` for an auxiliary projection: on its FP8
    /// mirror when [`aux_route`] says so, else `draft_gemm_bf16` on `bf16`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_gemm_aux(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        (bf16, fp8, bf16_kept): (&DenseWeight, Option<&Fp8DenseWeight>, bool),
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let kernels = (
            self.kernels.fp8_gemv_rt2.0 != 0,
            self.kernels.fp8_gemv_rt2_16.0 != 0,
            super::fp8_rt_enabled(),
        );
        match (aux_route(fp8.is_some(), bf16_kept, kernels, m, k), fp8) {
            (AuxRoute::Rt2x8, Some(w)) => ops::fp8_gemv_rowscale_batch8_rt2(
                gpu,
                self.kernels.fp8_gemv_rt2,
                input,
                w,
                output,
                m,
                n,
                k,
                stream,
            ),
            (AuxRoute::Rt2x16, Some(w)) => ops::fp8_gemv_rowscale_batch16_rt2(
                gpu,
                self.kernels.fp8_gemv_rt2_16,
                input,
                w,
                output,
                m,
                n,
                k,
                stream,
            ),
            (AuxRoute::RowScaled, Some(w)) => ops::fp8_gemm_n128_row_scaled(
                gpu,
                self.kernels.fp8_gemm_n128_row_scaled,
                input,
                w,
                output,
                m,
                n,
                k,
                stream,
            ),
            _ => self.draft_gemm_bf16(gpu, input, bf16, output, m, n, k, stream),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: (bool, bool, bool) = (true, true, true);

    #[test]
    fn routes_by_rows_and_mirror() {
        assert_eq!(aux_route(false, true, ALL, 4, 4096), AuxRoute::Bf16);
        assert_eq!(aux_route(false, false, ALL, 4, 4096), AuxRoute::Bf16);
        assert_eq!(aux_route(true, true, ALL, 1, 4096), AuxRoute::Rt2x8);
        assert_eq!(aux_route(true, true, ALL, 8, 20480), AuxRoute::Rt2x8);
        assert_eq!(aux_route(true, true, ALL, 9, 4096), AuxRoute::Rt2x16);
        assert_eq!(aux_route(true, true, ALL, 16, 4096), AuxRoute::Rt2x16);
        assert_eq!(aux_route(true, true, ALL, 17, 4096), AuxRoute::Bf16);
        assert_eq!(aux_route(true, false, ALL, 17, 4096), AuxRoute::RowScaled);
        assert_eq!(aux_route(true, true, ALL, 0, 4096), AuxRoute::Bf16);
        assert_eq!(
            aux_route(true, false, (true, true, false), 4, 4096),
            AuxRoute::RowScaled,
            "METRALE_NO_DFLASH_FP8_RT=1 skips the GEMVs"
        );
        assert_eq!(
            aux_route(true, true, (false, true, true), 4, 4096),
            AuxRoute::Rt2x16,
            "8-row kernel absent"
        );
        assert_eq!(
            aux_route(true, true, ALL, 4, 4104),
            AuxRoute::Bf16,
            "K % 16"
        );
    }

    /// 2026-10-12: The GLM-5.3 DFlash2 drafter on one of two ranks (DRAFT_TP2 log line:
    /// fc(0,2048), kv 4 heads x 128; 5 layers, hidden 4096, group 16, selector rank 256):
    /// the shapes PROPOSE-SPLIT measured on `dense_gemv_bf16_batchm`.
    #[test]
    fn glm_tp2_shapes_and_bytes() {
        let g = AuxGeom {
            hidden: 4096,
            layers: 5,
            conv_kernel_size: 2,
            conv_group_size: 16,
            selector_rank: 256,
            fc_rows: 2048,
            fc_in: 5 * 4096,
            kv_dim: 4 * 128,
        };
        let s = aux_shapes(g);
        let dims: Vec<(&str, usize, usize, usize, bool)> = s
            .iter()
            .map(|a| (a.name, a.n, a.k, a.count, a.replaces_bf16))
            .collect();
        assert_eq!(
            dims,
            vec![
                ("attention_conv_proj", 1024, 4096, 5, true),
                ("mlp_conv_proj", 1024, 4096, 5, true),
                ("fc", 2048, 20480, 1, false),
                ("fused_kv", 5120, 4096, 1, false),
                ("selector_hidden_proj", 256, 4096, 1, true),
            ]
        );
        let bf16: usize = s.iter().map(AuxShape::bf16_bytes).sum();
        let fp8: usize = s.iter().map(AuxShape::fp8_bytes).sum();
        // 2026-10-12: 211.8 MB BF16 read per round today; FP8 halves it plus row scales.
        assert_eq!(bf16, 211_812_352);
        assert_eq!(fp8, 105_906_176 + 4 * (10 * 1024 + 2048 + 5120 + 256));
        let extra: usize = s
            .iter()
            .filter(|a| !a.replaces_bf16)
            .map(AuxShape::fp8_bytes)
            .sum();
        let freed: usize = s
            .iter()
            .filter(|a| a.replaces_bf16)
            .map(AuxShape::bf16_bytes)
            .sum();
        let freed_fp8: usize = s
            .iter()
            .filter(|a| a.replaces_bf16)
            .map(AuxShape::fp8_bytes)
            .sum();
        assert_eq!(extra, 2048 * 20480 + 4 * 2048 + 5120 * 4096 + 4 * 5120);
        assert_eq!(freed, 10 * 1024 * 4096 * 2 + 256 * 4096 * 2);
        assert!(freed > freed_fp8);
    }

    #[test]
    fn no_conv_or_selector_shapes_without_dflash2() {
        let g = AuxGeom {
            hidden: 4096,
            layers: 5,
            conv_kernel_size: 0,
            conv_group_size: 0,
            selector_rank: 0,
            fc_rows: 4096,
            fc_in: 5 * 4096,
            kv_dim: 1024,
        };
        let names: Vec<&str> = aux_shapes(g).iter().map(|a| a.name).collect();
        assert_eq!(names, vec!["fc", "fused_kv"]);
    }
}
