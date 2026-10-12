// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: NVFP4 copies of the DFlash drafter's weights, three default-off levers:
//! - `METRALE_DFLASH_NV4_HEAD=1`: the lm_head (this rank's vocab shard under
//!   `METRALE_DFLASH_DRAFT_TP2`) in place of its runtime FP8 copy;
//! - `METRALE_DFLASH_NV4_LAYERS=1`: the seven per-layer GEMMs (q, k, v, o, gate, up, down) in
//!   place of their FP8 row-scaled mirrors;
//! - `METRALE_DFLASH_NV4_AUX=1`: the auxiliary projections (`aux_fp8.rs`: both conv kernel
//!   projections, `fc`, the fused ctx K/V, the selector's hidden projection) in place of the
//!   `METRALE_DFLASH_AUX_FP8` mirrors.
//!
//! Why: the TP2 C=1 propose is a weight-read stream (906 MB per rank per round with AUX_FP8,
//! P3 desk study 2026-10-12, spark-bench `.planning/targets/DFLASH-P3-DRAFTER-BYTES-20261012.md`).
//! NVFP4 is 0.5625 B per weight against FP8's 1 B + 4 B per row: head 317.5 -> 178.4 MB,
//! layers 482.9 -> 271.4 MB, aux 106.0 -> 59.6 MB per rank per round.
//!
//! Quantizer and kernels are the ones the MTP draft head uses (`METRALE_GLM_MTP_HEAD_NVFP4`,
//! `glm5next_mtp_head/head_nv4.rs`): `dense_fp8::quantize_nvfp4_copy` (packed E2M1, one E4M3
//! scale per 16 K, FP32 tensor scale, MSE quantizer) from the BF16 weight, and
//! `dense_fp8::nv4_gemv_cuda_core` (the CUDA-core `w4a16_gemv` / `_batch2..8,16` tiers; never the
//! tensor-core route nor the batchm staged tiers, whatever the target's levers say). Design
//! reference only: no code copied or ported from TensorFold/Mia.
//!
//! Numerics: drafter logits only. Every draft is verified by the target, so served tokens do
//! not change; drafts (acceptance) may. Each output row's bits equal the one-row launch on that
//! row. Under TP2 the o/down copies are quantized from this rank's contiguous column slice
//! (their own tensor scale), so the ranks' partial sums are not those of one-rank NVFP4.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - A lever arms only on the FP8 drafter (`Fp8Weights`) whose block rows never exceed
//!   [`NV4_ROWS_MAX`] ([`block_rows_max`]: the two-rank drafter proposes one sequence; a
//!   one-rank drafter may batch `max_batch` sequences). Otherwise it logs NOT ENGAGED and
//!   every weight keeps today's copy and kernel.
//! - A weight with an NVFP4 copy has no FP8 copy built for it (memory does not grow), so its
//!   readers take the NVFP4 GEMV at every row count they can see; [`gemv`] refuses a row
//!   count outside 1..=16 rather than fall back to a freed or absent copy.
//! - Off (the default), nothing here allocates, launches or logs.

use std::sync::Once;

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::weight_map::{DenseWeight, QuantizedWeight};

use super::{BlockDiffusionDraftHead, DflashLayer, DflashQuantization};
use crate::glm5next_layer::dense_fp8;

/// 2026-10-12: The widest row count the NVFP4 GEMV tiers take.
pub(super) const NV4_ROWS_MAX: usize = dense_fp8::NV4_MAX_M;

/// 2026-10-12: The engaged lines, printed once per process (one per rank).
static HEAD_LOG: Once = Once::new();
static LAYERS_LOG: Once = Once::new();
static AUX_LOG: Once = Once::new();

/// 2026-10-12: One layer's NVFP4 copies (`METRALE_DFLASH_NV4_LAYERS` for the seven GEMMs,
/// `METRALE_DFLASH_NV4_AUX` for the two conv kernel projections); `None` = not built.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerNv4 {
    pub q: Option<QuantizedWeight>,
    pub k: Option<QuantizedWeight>,
    pub v: Option<QuantizedWeight>,
    pub o: Option<QuantizedWeight>,
    pub gate: Option<QuantizedWeight>,
    pub up: Option<QuantizedWeight>,
    pub down: Option<QuantizedWeight>,
    pub attention_conv_proj: Option<QuantizedWeight>,
    pub mlp_conv_proj: Option<QuantizedWeight>,
}

impl LayerNv4 {
    /// 2026-10-12: The seven GEMM copies in [`layer_shapes`] order.
    fn gemms(&self) -> [&Option<QuantizedWeight>; 7] {
        [
            &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
        ]
    }
}

/// 2026-10-12: The head's NVFP4 copies: the lm_head (`METRALE_DFLASH_NV4_HEAD`) and the
/// head-level auxiliary projections (`METRALE_DFLASH_NV4_AUX`).
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadNv4 {
    pub lm_head: Option<QuantizedWeight>,
    pub fc: Option<QuantizedWeight>,
    pub fused_kv: Option<QuantizedWeight>,
    pub selector_hidden_proj: Option<QuantizedWeight>,
    /// 2026-10-12: Why the head lever did not build its copy, when the FP8 arm knew it.
    pub head_skip: Option<&'static str>,
}

/// 2026-10-12: Bytes of an NVFP4 copy of `[n, k]`: packed E2M1 (k/2 per row) plus one E4M3
/// scale per 16 K (the FP32 tensor scale lives on the host).
pub(super) const fn nv4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / 16
}

/// 2026-10-12: Bytes of the FP8 row-scaled copy of `[n, k]` (1 B per weight + one f32 per row).
pub(super) const fn fp8_bytes(n: usize, k: usize) -> usize {
    n * k + 4 * n
}

/// 2026-10-12: The widest block a drafter GEMM sees: `gamma` rows (`set_block_g` clamps the
/// block to `2..=gamma.max(2)`) times the sequences one propose packs (one under TP2,
/// `propose_batch_max`; up to `max_batch` otherwise).
pub(super) fn block_rows_max(gamma: usize, max_batch: usize, tp: bool) -> usize {
    let g = gamma.max(2);
    if tp { g } else { g * max_batch.max(1) }
}

/// 2026-10-12: Why no NVFP4 lever can arm on this drafter, or `None` when they can.
pub(super) fn blocker(fp8_drafter: bool, rows_max: usize) -> Option<String> {
    if !fp8_drafter {
        return Some(
            "the drafter is not on FP8 weights (needs METRALE_DFLASH_DRAFTER_FP8 not 0 and its \
             kernels)"
                .to_string(),
        );
    }
    if rows_max > NV4_ROWS_MAX {
        return Some(format!(
            "drafter blocks reach {rows_max} rows > {NV4_ROWS_MAX} (the NVFP4 GEMV tiers take \
             1..={NV4_ROWS_MAX}; the two-rank drafter METRALE_DFLASH_DRAFT_TP2 proposes one \
             sequence)"
        ));
    }
    None
}

/// 2026-10-12: One per-layer GEMM: `[n, k]` as read, and for a row-parallel slice the full
/// depth and the slice's first column (`(k_full, c0)`; then `k` is the slice width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Nv4Shape {
    pub name: &'static str,
    pub n: usize,
    pub k: usize,
    pub cols: Option<(usize, usize)>,
}

impl Nv4Shape {
    const fn rows(name: &'static str, n: usize, k: usize) -> Self {
        Self {
            name,
            n,
            k,
            cols: None,
        }
    }

    /// 2026-10-12: `[n, k]` as read: a column slice `(k_full, (c0, len))` of a row-parallel
    /// weight (TP2), else the whole `[n, k_local]`.
    fn sliced(
        name: &'static str,
        n: usize,
        k_local: usize,
        slice: Option<(usize, (usize, usize))>,
    ) -> Self {
        match slice {
            Some((k_full, (c0, len))) => Self {
                name,
                n,
                k: len,
                cols: Some((k_full, c0)),
            },
            None => Self::rows(name, n, k_local),
        }
    }
}

/// 2026-10-12: The seven per-layer GEMMs of one rank, in q, k, v, o, gate, up, down order.
/// `q_dim` / `kv_dim` / `inter` are this rank's; `o_cols` / `down_cols` are the TP2
/// row-parallel slices (`TpPlan::o_cols` / `down_cols`), `None` on one rank.
pub(super) fn layer_shapes(
    (h, q_dim, kv_dim, inter): (usize, usize, usize, usize),
    o_cols: Option<(usize, (usize, usize))>,
    down_cols: Option<(usize, (usize, usize))>,
) -> [Nv4Shape; 7] {
    [
        Nv4Shape::rows("q", q_dim, h),
        Nv4Shape::rows("k", kv_dim, h),
        Nv4Shape::rows("v", kv_dim, h),
        Nv4Shape::sliced("o", h, q_dim, o_cols),
        Nv4Shape::rows("gate", inter, h),
        Nv4Shape::rows("up", inter, h),
        Nv4Shape::sliced("down", h, inter, down_cols),
    ]
}

/// 2026-10-12: An NVFP4 copy of the BF16 `[n, k]` at `w`; `None` when the NVFP4 kernels are
/// missing, the pointer is null or `k % 16 != 0`.
pub(super) fn quantize_rows(
    gpu: &dyn GpuBackend,
    w: DevicePtr,
    n: usize,
    k: usize,
) -> Result<Option<QuantizedWeight>> {
    dense_fp8::quantize_nvfp4_copy(gpu, w, n, k, gpu.default_stream())
}

/// 2026-10-12: An NVFP4 copy of `s` read from the BF16 weight `w`: the whole rows, or for a
/// column slice a contiguous BF16 `[n, len]` temporary (freed before returning) quantized on
/// its own.
fn quantize_shape(
    gpu: &dyn GpuBackend,
    w: &DenseWeight,
    s: &Nv4Shape,
) -> Result<Option<QuantizedWeight>> {
    let Some((k_full, c0)) = s.cols else {
        return quantize_rows(gpu, w.weight, s.n, s.k);
    };
    ensure!(
        c0 + s.k <= k_full && s.k > 0,
        "DFlash NV4 slice {}: [{c0}, +{}) of {k_full}",
        s.name,
        s.k
    );
    if w.weight.is_null() || !s.k.is_multiple_of(16) {
        return Ok(None);
    }
    let stream = gpu.default_stream();
    let tmp = gpu.alloc(s.n * s.k * 2)?;
    // 2026-10-12: BF16 is two bytes per element; pitches and width are bytes.
    let copied = gpu
        .copy_d2d_2d_async(
            w.weight.offset(c0 * 2),
            k_full * 2,
            tmp,
            s.k * 2,
            s.k * 2,
            s.n,
            stream,
        )
        .and_then(|()| gpu.synchronize(stream));
    // 2026-10-12: `quantize_nvfp4_copy` synchronizes the stream, so the temporary is free to go.
    let q = copied.and_then(|()| quantize_rows(gpu, tmp, s.n, s.k));
    gpu.free(tmp)?;
    q
}

/// 2026-10-12: Builds `layer`'s NVFP4 copies of the seven GEMMs (`shapes` from
/// [`layer_shapes`]); returns which were built, in the same order. The FP8 arm builds its
/// mirror for every weight left `false`.
pub(super) fn quantize_layer(
    gpu: &dyn GpuBackend,
    layer: &mut DflashLayer,
    shapes: &[Nv4Shape; 7],
) -> Result<[bool; 7]> {
    let srcs = [
        layer.q_proj,
        layer.k_proj,
        layer.v_proj,
        layer.o_proj,
        layer.gate_proj,
        layer.up_proj,
        layer.down_proj,
    ];
    let mut out = [None; 7];
    for ((slot, w), s) in out.iter_mut().zip(srcs.iter()).zip(shapes) {
        *slot = quantize_shape(gpu, w, s)?;
    }
    let n = &mut layer.nv4;
    [n.q, n.k, n.v, n.o, n.gate, n.up, n.down] = out;
    Ok(out.map(|q| q.is_some()))
}

/// 2026-10-12: `C[m, n] = A[m, k] @ W[n, k]^T` over the NVFP4 copy `q` on the CUDA-core tiers
/// (`dense_fp8::nv4_gemv_cuda_core`), `A` packed at stride `k`, `C` (BF16) at stride `n`.
/// Errors outside 1..=[`NV4_ROWS_MAX`] rows: the weight may have no other copy.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemv(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    q: &QuantizedWeight,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if !(1..=NV4_ROWS_MAX).contains(&(m as usize)) {
        bail!(
            "DFlash NVFP4 GEMV: {m} rows outside 1..={NV4_ROWS_MAX} ({n}x{k}); the lever's \
             row bound (nv4::block_rows_max) was violated"
        );
    }
    dense_fp8::nv4_gemv_cuda_core(gpu, a, q, c, m as usize, n as usize, k as usize, stream)
}

/// 2026-10-12: Megabytes, for the log lines.
fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

impl BlockDiffusionDraftHead {
    /// 2026-10-12: This head's [`block_rows_max`].
    pub(super) fn nv4_rows_max(&self) -> usize {
        block_rows_max(self.gamma, self.max_batch, self.tp.is_some())
    }

    /// 2026-10-12: Whether `lever` may build NVFP4 copies on this head, called from the FP8
    /// arm (so the head is about to be `Fp8Weights`). No log: [`Self::report_nv4`] explains.
    pub(super) fn nv4_armed(&self, lever: bool) -> bool {
        lever && blocker(true, self.nv4_rows_max()).is_none()
    }

    /// 2026-10-12: The reason `lever`'s copies were not built, from the head's final state.
    fn nv4_reason(&self, head_lever: bool) -> String {
        let fp8 = self.quant == DflashQuantization::Fp8Weights;
        if let Some(r) = blocker(fp8, self.nv4_rows_max()) {
            return r;
        }
        if head_lever {
            if self.lm_head_nvfp4.is_some() {
                return "the target ships an NVFP4 lm_head, which the drafter already reads \
                        (w4a16_gemm)"
                    .to_string();
            }
            if let Some(r) = self.nv4.head_skip {
                return r.to_string();
            }
        }
        "quantize_nvfp4_copy built nothing (NVFP4 kernels quantize_nvfp4 / w4a16_gemv missing \
         from this image, a null BF16 source, or K % 16 != 0)"
            .to_string()
    }

    /// 2026-10-12: One line per lever that is set: ENGAGED (shapes, MB, delta against the FP8
    /// copies not built) once per rank, or a NOT ENGAGED warning with the reason. Runs after the
    /// FP8 and aux arms in `from_weights`.
    pub(super) fn report_nv4(&self) {
        let rank = self.tp.as_ref().map_or(0, |t| t.rank());
        if self.levers.nv4_head {
            if self.nv4.lm_head.is_some() {
                let (n, k) = (self.vocab_size, self.hidden_size);
                let (nv, f8) = (nv4_bytes(n, k), fp8_bytes(n, k));
                HEAD_LOG.call_once(|| {
                    tracing::warn!(
                        "METRALE_DFLASH_NV4_HEAD=1: ENGAGED (rank {rank}: lm_head [{n}x{k}] NVFP4 \
                         {:.1} MB; the {:.1} MB FP8 copy is not built, {:+.1} MB; 1..=16 rows on \
                         the CUDA-core w4a16_gemv tiers; drafts only)",
                        mb(nv),
                        mb(f8),
                        mb(nv) - mb(f8),
                    );
                });
            } else {
                tracing::warn!(
                    "METRALE_DFLASH_NV4_HEAD=1: NOT ENGAGED: {}; the drafter head keeps its \
                     FP8/BF16 copy",
                    self.nv4_reason(true)
                );
            }
        }
        if self.levers.nv4_layers {
            self.report_nv4_layers(rank);
        }
        if self.levers.nv4_aux {
            self.report_nv4_aux(rank);
        }
    }

    /// 2026-10-12: The `METRALE_DFLASH_NV4_LAYERS` line (see [`Self::report_nv4`]).
    fn report_nv4_layers(&self, rank: usize) {
        let built: usize = self
            .layers
            .iter()
            .map(|l| l.nv4.gemms().iter().filter(|q| q.is_some()).count())
            .sum();
        if built == 0 {
            tracing::warn!(
                "METRALE_DFLASH_NV4_LAYERS=1: NOT ENGAGED: {}; the layer GEMMs keep their FP8/BF16 \
                 copies",
                self.nv4_reason(false)
            );
            return;
        }
        let shapes = self.nv4_layer_shapes();
        let (mut nv, mut f8) = (0usize, 0usize);
        for l in &self.layers {
            for (q, s) in l.nv4.gemms().iter().zip(&shapes) {
                if q.is_some() {
                    nv += nv4_bytes(s.n, s.k);
                    f8 += fp8_bytes(s.n, s.k);
                }
            }
        }
        let list: Vec<String> = shapes
            .iter()
            .map(|s| format!("{} [{}x{}]", s.name, s.n, s.k))
            .collect();
        let total = 7 * self.layers.len();
        LAYERS_LOG.call_once(|| {
            tracing::warn!(
                "METRALE_DFLASH_NV4_LAYERS=1: ENGAGED (rank {rank}: {built}/{total} layer GEMMs \
                 NVFP4, per layer {}; {:.1} MB NVFP4, the {:.1} MB FP8 mirrors are not built, \
                 {:+.1} MB; 1..=16 rows on the CUDA-core w4a16_gemv tiers; drafts only)",
                list.join(", "),
                mb(nv),
                mb(f8),
                mb(nv) - mb(f8),
            );
        });
    }

    /// 2026-10-12: The `METRALE_DFLASH_NV4_AUX` line (see [`Self::report_nv4`]).
    fn report_nv4_aux(&self, rank: usize) {
        let shapes = super::aux_fp8::aux_shapes(self.aux_geom());
        let mut parts = Vec::new();
        let (mut nv, mut f8) = (0usize, 0usize);
        for s in &shapes {
            let built = match s.name {
                "attention_conv_proj" => self
                    .layers
                    .iter()
                    .filter(|l| l.nv4.attention_conv_proj.is_some())
                    .count(),
                "mlp_conv_proj" => self
                    .layers
                    .iter()
                    .filter(|l| l.nv4.mlp_conv_proj.is_some())
                    .count(),
                "fc" => usize::from(self.nv4.fc.is_some()),
                "fused_kv" => usize::from(self.nv4.fused_kv.is_some()),
                _ => usize::from(self.nv4.selector_hidden_proj.is_some()),
            };
            if built > 0 {
                nv += built * nv4_bytes(s.n, s.k);
                f8 += built * fp8_bytes(s.n, s.k);
                parts.push(format!("{}x{built} [{}x{}]", s.name, s.n, s.k));
            }
        }
        if parts.is_empty() {
            tracing::warn!(
                "METRALE_DFLASH_NV4_AUX=1: NOT ENGAGED: {}; the auxiliary projections keep \
                 their FP8 (METRALE_DFLASH_AUX_FP8) or BF16 copies",
                self.nv4_reason(false)
            );
            return;
        }
        AUX_LOG.call_once(|| {
            tracing::warn!(
                "METRALE_DFLASH_NV4_AUX=1: ENGAGED (rank {rank}: {}; {:.1} MB NVFP4 in place of \
                 {:.1} MB AUX_FP8 mirrors (not built), {:+.1} MB against them; <=16 rows on the \
                 CUDA-core w4a16_gemv tiers, fc / fused K/V keep BF16 for wider precompute \
                 chunks; drafts only)",
                parts.join(", "),
                mb(nv),
                mb(f8),
                mb(nv) - mb(f8),
            );
        });
    }

    /// 2026-10-12: [`layer_shapes`] of this head (local dims, TP2 slices from the plan).
    pub(super) fn nv4_layer_shapes(&self) -> [Nv4Shape; 7] {
        let dims = (
            self.hidden_size,
            self.num_q_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.intermediate_size,
        );
        let o = self.tp.as_ref().map(|t| t.plan.o_cols());
        let down = self.tp.as_ref().map(|t| t.plan.down_cols());
        layer_shapes(dims, o, down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-12: GLM-5.3 DFlash2 drafter on one of two ranks (DRAFT_TP2 log: q 16 heads,
    /// kv 4, inter 6144; hidden 4096, head_dim 128): the shapes the P3 desk study lists.
    #[test]
    fn glm_tp2_layer_shapes_and_bytes() {
        let s = layer_shapes(
            (4096, 2048, 512, 6144),
            Some((4096, (2048, 2048))),
            Some((12288, (6144, 6144))),
        );
        let dims: Vec<(&str, usize, usize, Option<(usize, usize)>)> =
            s.iter().map(|x| (x.name, x.n, x.k, x.cols)).collect();
        assert_eq!(
            dims,
            vec![
                ("q", 2048, 4096, None),
                ("k", 512, 4096, None),
                ("v", 512, 4096, None),
                ("o", 4096, 2048, Some((4096, 2048))),
                ("gate", 6144, 4096, None),
                ("up", 6144, 4096, None),
                ("down", 4096, 6144, Some((12288, 6144))),
            ]
        );
        assert!(s.iter().all(|x| x.k.is_multiple_of(16)));
        let nv: usize = s.iter().map(|x| 5 * nv4_bytes(x.n, x.k)).sum();
        let f8: usize = s.iter().map(|x| 5 * fp8_bytes(x.n, x.k)).sum();
        // 2026-10-12: P3 §4: layers 482.9 -> 271.4 MB per rank per round.
        assert_eq!(
            f8,
            482_344_960 + 5 * 4 * (2048 + 512 * 2 + 4096 + 6144 * 2 + 4096)
        );
        assert_eq!(nv, 482_344_960 / 16 * 9);
        assert!((271.0..271.5).contains(&mb(nv)), "{}", mb(nv));
        assert!((482.5..483.0).contains(&mb(f8)), "{}", mb(f8));
        // 2026-10-12: The head shard [77440, 4096]: 317.5 -> 178.4 MB.
        assert!((178.3..178.5).contains(&mb(nv4_bytes(77440, 4096))));
        assert!((317.4..317.6).contains(&mb(fp8_bytes(77440, 4096))));
    }

    #[test]
    fn one_rank_layer_shapes_are_whole() {
        let s = layer_shapes((4096, 4096, 1024, 12288), None, None);
        assert_eq!(s[3], Nv4Shape::rows("o", 4096, 4096));
        assert_eq!(s[6], Nv4Shape::rows("down", 4096, 12288));
    }

    #[test]
    fn rows_bound_and_blockers() {
        assert_eq!(block_rows_max(8, 4, true), 8, "TP2 proposes one sequence");
        assert_eq!(block_rows_max(8, 1, false), 8);
        assert_eq!(block_rows_max(8, 4, false), 32);
        assert_eq!(block_rows_max(0, 0, false), 2, "set_block_g clamps to >= 2");
        assert_eq!(block_rows_max(16, 1, true), 16);
        assert!(blocker(true, 8).is_none());
        assert!(blocker(true, 16).is_none());
        assert!(blocker(true, 17).unwrap().contains("17 rows"));
        assert!(blocker(false, 8).unwrap().contains("not on FP8"));
    }

    /// 2026-10-12: The NVFP4 byte formula is the quantizer's layout (`dense_fp8.rs`: packed
    /// E2M1 `[n, k/2]`, one E4M3 scale per 16 K) and beats FP8 + row scales at every K % 16.
    #[test]
    fn nv4_is_0_5625_bytes_per_weight() {
        assert_eq!(nv4_bytes(16, 16), 16 * 8 + 16);
        for (n, k) in [(1usize, 16usize), (512, 4096), (4096, 20480)] {
            assert_eq!(nv4_bytes(n, k) * 16, n * k * 9);
            assert!(nv4_bytes(n, k) < fp8_bytes(n, k));
        }
    }
}
