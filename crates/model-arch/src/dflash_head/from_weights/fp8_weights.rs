// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The DFlash drafter's FP8 weight copies: each layer's seven dense GEMM
//! weights and the lm_head.
//!
//! 2026-10-08: `arm_drafter_fp8` decides whether the head goes FP8 (moved here from
//! `from_weights.rs`) and, on targets whose `w4a16` module lacks the row-scaled FP8 GEMMs
//! (GLM-5.3), takes them from the target's `dflash_fp8_gemm` module under an explicit
//! METRALE_DFLASH_DRAFTER_FP8=1 (SeedHQ/seed-skills#71 S1b).
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Unset or any value but `1` on a target without the `w4a16` kernels keeps the BF16
//!   drafter, as before 2026-10-08; `0` keeps BF16 everywhere.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::kernel_handles;
use crate::dflash_head::{BlockDiffusionDraftHead, DflashQuantization};

/// 2026-09-25: FP8 drafter weights, on unless METRALE_DFLASH_DRAFTER_FP8=0: quantize each
/// layer's seven dense GEMM weights (q/k/v/o/gate/up/down) to FP8 E4M3 with one f32 scale
/// per row, give the lm_head an FP8 copy, and set `quant = Fp8Weights`. Needs the
/// row-scaled FP8 GEMM and its m16 variant; without them the drafter stays on BF16 with a
/// warning.
///
/// 2026-10-08: When the `w4a16` module lacks them and the variable is exactly `1`, the
/// pair comes from `dflash_fp8_gemm` (GLM-5.3's own port). The serve's footprint
/// pre-flight and KV reserve already count the mirrors for any value but `0`.
pub(super) fn arm_drafter_fp8(
    head: &mut BlockDiffusionDraftHead,
    gpu: &dyn GpuBackend,
    q_dim: usize,
    kv_dim: usize,
    lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
) -> Result<()> {
    let fp8_env = std::env::var("METRALE_DFLASH_DRAFTER_FP8").ok();
    let fp8_requested = fp8_env.as_deref() != Some("0");
    let present = |h: &BlockDiffusionDraftHead| {
        h.kernels.fp8_gemm_n128_row_scaled.0 != 0 && h.kernels.fp8_gemm_n128_row_scaled_m16.0 != 0
    };
    let mut source = "w4a16";
    if fp8_requested && !present(head) && fp8_env.as_deref() == Some("1") {
        let (k, k16) = kernel_handles::load_fp8_row_scaled_fallback(gpu);
        if k.0 != 0 && k16.0 != 0 {
            head.kernels.fp8_gemm_n128_row_scaled = k;
            head.kernels.fp8_gemm_n128_row_scaled_m16 = k16;
            source = kernel_handles::GLM_FP8_GEMM_MODULE;
        }
    }
    if !fp8_requested {
        return Ok(());
    }
    if !present(head) {
        tracing::warn!(
            "METRALE_DFLASH_DRAFTER_FP8 is not 0 but fp8_gemm_t_row_scaled(_m16) are in \
             neither this target's w4a16 module nor (with =1) its dflash_fp8_gemm module \
             — staying on the BF16 drafter path."
        );
        return Ok(());
    }
    quantize_drafter_fp8(head, gpu, q_dim, kv_dim, lm_head_native_fp8)?;
    // 2026-10-08: The kernels the FP8 projections take, in the layers' and tail's order.
    let rt2 = crate::dflash_head::fp8_rt_enabled();
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "METRALE_DFLASH_DRAFTER_FP8: ENGAGED - drafter weights FP8 E4M3 row-scaled: \
         <=8 rows fp8_gemv_rowscale_batch8_rt2 (present {}), <=16 rows \
         fp8_gemv_rowscale_batch16_rt2 (present {}), rt2 enabled {}, else \
         {source}::fp8_gemm_t_row_scaled(_m16)",
        head.kernels.fp8_gemv_rt2.0 != 0,
        head.kernels.fp8_gemv_rt2_16.0 != 0,
        rt2,
    );
    Ok(())
}

/// 2026-09-26: Gives `head` FP8 copies of its layer weights and lm_head, and sets
/// `quant = Fp8Weights`.
fn quantize_drafter_fp8(
    head: &mut BlockDiffusionDraftHead,
    gpu: &dyn GpuBackend,
    q_dim: usize,
    kv_dim: usize,
    lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
) -> Result<()> {
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: quantizing drafter weights to FP8 E4M3 ({} layers × 7 GEMMs)",
        head.num_layers
    );
    let stream = 0u64;
    let q_dim_local = q_dim;
    let kv_dim_local = kv_dim;
    let h = head.hidden_size;
    let inter = head.intermediate_size;
    let quant_k = head.kernels.quantize_bf16_to_fp8;
    let mut mirror_bytes = 0usize;
    // 2026-10-09: METRALE_DFLASH_DRAFT_TP2: o/down are row-parallel, so each rank keeps
    // a column slice of the full quantization (`tp/shard_weights.rs`).
    let o_cols = head.tp.as_ref().map(|t| t.plan.o_cols());
    let down_cols = head.tp.as_ref().map(|t| t.plan.down_cols());
    let quantize_cols = crate::dflash_head::tp::shard_weights::quantize_cols;
    // 2026-10-12: METRALE_DFLASH_NV4_LAYERS (`nv4.rs`): NVFP4 copies first, from the BF16
    // weights (still live: they are released after this arm); the FP8 mirror is built only
    // for a weight that got none.
    let nv4_shapes = head
        .nv4_armed(head.levers.nv4_layers)
        .then(|| head.nv4_layer_shapes());
    for (layer_idx, layer) in head.layers.iter_mut().enumerate() {
        let nv4 = match nv4_shapes.as_ref() {
            Some(shapes) => crate::dflash_head::nv4::quantize_layer(gpu, layer, shapes)?,
            None => [false; 7],
        };
        // 2026-10-14: What each mirror below allocates, by the rule the KV reserve predicts with.
        let mb = fp8_mirror_alloc_bytes;
        let (o_n, o_k) = o_cols.map_or((h, q_dim_local), |(_, (_, cols))| (h, cols));
        let (d_n, d_k) = down_cols.map_or((h, inter), |(_, (_, cols))| (h, cols));
        for (built, bytes) in [
            (!nv4[0], mb(q_dim_local, h)),
            (!nv4[1], mb(kv_dim_local, h)),
            (!nv4[2], mb(kv_dim_local, h)),
            (!nv4[3], mb(o_n, o_k)),
            (!nv4[4], mb(inter, h)),
            (!nv4[5], mb(inter, h)),
            (!nv4[6], mb(d_n, d_k)),
        ] {
            if built {
                mirror_bytes += bytes;
            }
        }
        if !nv4[0] {
            layer.q_proj_fp8 = Some(
                layer
                    .q_proj
                    .quantize_to_fp8(gpu, quant_k, q_dim_local, h, stream)?,
            );
        }
        if !nv4[1] {
            layer.k_proj_fp8 = Some(
                layer
                    .k_proj
                    .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
            );
        }
        if !nv4[2] {
            layer.v_proj_fp8 = Some(
                layer
                    .v_proj
                    .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
            );
        }
        if !nv4[3] {
            layer.o_proj_fp8 = Some(match o_cols {
                Some((k_full, cols)) => {
                    quantize_cols(gpu, quant_k, &layer.o_proj, h, k_full, cols)?
                }
                None => layer
                    .o_proj
                    .quantize_to_fp8(gpu, quant_k, h, q_dim_local, stream)?,
            });
        }
        if !nv4[4] {
            layer.gate_proj_fp8 = Some(
                layer
                    .gate_proj
                    .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
            );
        }
        if !nv4[5] {
            layer.up_proj_fp8 = Some(
                layer
                    .up_proj
                    .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
            );
        }
        if !nv4[6] {
            layer.down_proj_fp8 = Some(match down_cols {
                Some((k_full, cols)) => {
                    quantize_cols(gpu, quant_k, &layer.down_proj, h, k_full, cols)?
                }
                None => layer
                    .down_proj
                    .quantize_to_fp8(gpu, quant_k, h, inter, stream)?,
            });
        }
        tracing::debug!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: layer {} quantized", layer_idx
        );
    }
    // 2026-09-25: The lm_head's FP8 copy: the checkpoint's own FP8 lm_head when the
    // model factory passes one (`lm_head_native_fp8`, an `Fp8DenseWeight` like the
    // layer copies, so the same tail kernels read it), else a new FP8
    // quantization of the shared BF16 lm_head in its own buffer, which leaves the
    // target's weight untouched.
    if let Some((shared, rows)) = lm_head_native_fp8 {
        // 2026-10-09: Under TP the share is the full head and this rank takes its rows.
        let full_vocab = head
            .tp
            .as_ref()
            .map_or(head.vocab_size, |t| t.plan.geom.vocab);
        let shared = head
            .tp
            .as_ref()
            .map_or(shared, |t| t.plan.lm_head_fp8_rows(&shared));
        anyhow::ensure!(
            rows == full_vocab,
            "native FP8 lm_head share rows ({rows}) != drafter vocab ({}) — \
                     the drafter's tail GEMM iterates head.vocab_size rows",
            full_vocab
        );
        tracing::info!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: sharing the checkpoint's NATIVE FP8 lm_head \
             [{} × {}] (1.27 GB runtime mirror skipped)",
            head.vocab_size,
            head.hidden_size
        );
        head.lm_head_shared_fp8 = Some(shared);
        head.nv4.head_skip = Some(
            "the drafter shares the checkpoint's native FP8 lm_head (no BF16 rows to quantize)",
        );
    } else if let Some(q) = head
        .nv4_armed(head.levers.nv4_head && head.lm_head_nvfp4.is_none())
        .then(|| {
            crate::dflash_head::nv4::quantize_rows(
                gpu,
                head.lm_head_shared,
                head.vocab_size,
                head.hidden_size,
            )
        })
        .transpose()?
        .flatten()
    {
        // 2026-10-12: METRALE_DFLASH_NV4_HEAD (`nv4.rs`): an NVFP4 copy of this rank's BF16
        // head rows in place of the FP8 copy, which is not built. The target's head is
        // untouched.
        head.nv4.lm_head = Some(q);
    } else {
        tracing::info!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash Phase G: quantizing shared lm_head [{} × {}]",
            head.vocab_size,
            head.hidden_size
        );
        let lm_head_bf16 = metrale_model_layers::weight_map::DenseWeight {
            weight: head.lm_head_shared,
        };
        head.lm_head_shared_fp8 = Some(lm_head_bf16.quantize_to_fp8(
            gpu,
            quant_k,
            head.vocab_size,
            head.hidden_size,
            stream,
        )?);
        mirror_bytes += fp8_mirror_alloc_bytes(head.vocab_size, head.hidden_size);
    }
    head.fp8_mirror_bytes = mirror_bytes;
    head.quant = DflashQuantization::Fp8Weights;
    tracing::info!(
        target: "metrale_model_arch::dflash_head::from_weights",
        "DFlash Phase G: drafter weights ready as FP8 (quant = Fp8Weights). \
         Set METRALE_DFLASH_DRAFTER_FP8=0 to revert to BF16."
    );
    Ok(())
}

/// 2026-10-10: Store-name suffixes of the seven dense GEMM weights per layer whose BF16
/// copies [`BlockDiffusionDraftHead::release_bf16_layer_weights`] frees.
const BF16_LAYER_SUFFIXES: [&str; 7] = [
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
];

/// 2026-10-12: Store-name suffixes of the BF16 tensors `METRALE_DFLASH_AUX_FP8` replaces
/// (`aux_fp8.rs`): the two conv kernel projections per layer, then the selector's hidden
/// projection.
const AUX_CONV_SUFFIXES: [&str; 2] = [
    ".attention_conv.kernel_projection.weight",
    ".mlp_conv.kernel_projection.weight",
];
const AUX_SELECTOR_SUFFIX: &str = "candidate_selector.hidden_projection.weight";

/// 2026-10-14: Bytes one per-row FP8 mirror of a `[n, k]` weight allocates:
/// `Fp8DenseWeight` is `n * k` E4M3 bytes plus `n` f32 row scales (`quantize_to_fp8`; the
/// column slice of `quantize_cols` allocates the same for its `[n, len]`).
pub(crate) fn fp8_mirror_alloc_bytes(n: usize, k: usize) -> usize {
    n * k + n * std::mem::size_of::<f32>()
}

impl BlockDiffusionDraftHead {
    /// 2026-10-14: The bytes `quantize_drafter_fp8` allocates on one rank for the seven layer
    /// GEMM mirrors and the lm_head mirror, predicted from the drafter config before the head
    /// exists: each weight's part from `tp::shard::weight_parts_with` (the rule the install's
    /// slices come from; `full_layout` without TP), priced by [`fp8_mirror_alloc_bytes`].
    /// `0` when `fp8_requested` is false (`METRALE_DFLASH_DRAFTER_FP8=0`). `draft_tp2` is
    /// `METRALE_DFLASH_DRAFT_TP2`; `rank_world` the serve's `(rank, world)`.
    /// Counts the lm_head mirror always: a checkpoint-native FP8 lm_head is shared, so that
    /// case over-reserves it. `fc` / fused K/V / conv / selector mirrors
    /// (`METRALE_DFLASH_AUX_FP8`) and NVFP4 copies are not counted. If the geometry cannot be
    /// split the full (unsharded) size is returned, the safe side.
    pub fn predicted_fp8_mirror_bytes(
        c: &crate::weight_loader::DflashConfig,
        draft_tp2: bool,
        rank_world: Option<(usize, usize)>,
        fp8_requested: bool,
    ) -> usize {
        use crate::dflash_head::tp::shard::{Split, TpGeometry, weight_parts_with};
        if !fp8_requested {
            return 0;
        }
        let g = TpGeometry {
            world: 2,
            q_heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            hidden: c.hidden_size,
            inter: c.intermediate_size,
            vocab: c.vocab_size,
            fc_in: 0,
        };
        let layout = draft_tp2
            .then(|| crate::dflash_head::tp::TpPlan::two_rank(rank_world))
            .flatten()
            .and_then(|(rank, _)| g.layout(rank).ok())
            .unwrap_or_else(|| g.full_layout());
        let bytes = |p: &crate::dflash_head::tp::shard::WeightPart| match p.split {
            Split::Rows => fp8_mirror_alloc_bytes(p.len, p.k),
            Split::Cols => fp8_mirror_alloc_bytes(p.n, p.len),
        };
        weight_parts_with(&g, &layout)
            .iter()
            .map(|p| match p.name {
                "fc" => 0,
                "lm_head" => bytes(p),
                _ => c.num_hidden_layers * bytes(p),
            })
            .sum()
    }

    /// 2026-10-14: The bytes [`Self::release_bf16_layer_weights`] frees under its layer rule,
    /// predicted from the drafter store before the head exists: the BF16 store copies of the
    /// seven layer GEMM weights, when the FP8 drafter is requested (`fp8_requested`,
    /// `METRALE_DFLASH_DRAFTER_FP8` not `0`) and the paged body is on (`option_b`).
    /// The KV sizing credits these bytes, which are freed only after it (#82 hybrid M1),
    /// and the install checks that the release freed at least this much. The conv and
    /// selector frees under `METRALE_DFLASH_AUX_FP8` are not counted.
    pub fn predicted_bf16_layer_release_bytes(
        store: &metrale_model_weights::weights::WeightStore,
        fp8_requested: bool,
        option_b: bool,
    ) -> usize {
        if !(fp8_requested && option_b) {
            return 0;
        }
        store.freeable_bytes_matching(|n, t| {
            BF16_LAYER_SUFFIXES.iter().any(|s| n.ends_with(s))
                && t.dtype == metrale_model_weights::weights::WeightDtype::BF16
        })
    }

    /// 2026-10-10: Frees the BF16 copies of the layers' seven dense GEMM weights once the
    /// head runs on their FP8 mirrors (#82 hybrid memory, P2). Returns the freed (tensors,
    /// bytes), or `None` when it keeps them.
    ///
    /// Kept unless `quant` is `Fp8Weights`, every layer holds all seven mirrors, and the
    /// paged layer body is on (`levers.option_b`): the non-paged body
    /// (`forward_block_layer.rs`, METRALE_DFLASH_OPTION_B=0) reads the BF16 weights.
    /// The paged body passes them to `gemm_swap`, which reads the mirror when it is
    /// present. The fused ctx K/V weight is its own copy, made in `from_weights`.
    /// Only BF16 store tensors are freed: an NVFP4-packed tensor was dequantized into
    /// another allocation.
    ///
    /// 2026-10-12: Under `METRALE_DFLASH_AUX_FP8` it also frees the conv kernel projections
    /// (when every layer that has them holds both mirrors) and the selector's hidden
    /// projection (when mirrored): their only readers, `conv_prepare` and
    /// `dflash2_walk_block`, route a mirrored projection to FP8 at every row count
    /// (`aux_fp8::aux_route` with the BF16 copy not kept). `fc` and the fused K/V keep
    /// their BF16 copies.
    ///
    /// The head's BF16 pointers are set to null first, so a read that this audit missed
    /// faults with an illegal address instead of reading freed memory.
    pub fn release_bf16_layer_weights(
        &mut self,
        store: &mut metrale_model_weights::weights::WeightStore,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<(usize, usize)>> {
        // 2026-10-12: A weight is mirrored by its FP8 copy or its NVFP4 copy
        // (METRALE_DFLASH_NV4_LAYERS / _AUX, `nv4.rs`), whose readers take it at every row
        // count they see.
        let mirrored = self.layers.iter().all(|l| {
            (l.q_proj_fp8.is_some() || l.nv4.q.is_some())
                && (l.k_proj_fp8.is_some() || l.nv4.k.is_some())
                && (l.v_proj_fp8.is_some() || l.nv4.v.is_some())
                && (l.o_proj_fp8.is_some() || l.nv4.o.is_some())
                && (l.gate_proj_fp8.is_some() || l.nv4.gate.is_some())
                && (l.up_proj_fp8.is_some() || l.nv4.up.is_some())
                && (l.down_proj_fp8.is_some() || l.nv4.down.is_some())
        });
        let layers_go =
            self.quant == DflashQuantization::Fp8Weights && self.levers.option_b && mirrored;
        let att = |l: &crate::dflash_head::DflashLayer| {
            l.attention_conv_proj_fp8.is_some() || l.nv4.attention_conv_proj.is_some()
        };
        let mlp = |l: &crate::dflash_head::DflashLayer| {
            l.mlp_conv_proj_fp8.is_some() || l.nv4.mlp_conv_proj.is_some()
        };
        let conv_go = self.layers.iter().any(att)
            && self.layers.iter().all(|l| {
                (l.attention_conv_proj.is_none() || att(l))
                    && (l.mlp_conv_proj.is_none() || mlp(l))
            });
        let selector_go =
            self.selector_hidden_proj_fp8.is_some() || self.nv4.selector_hidden_proj.is_some();
        let doomed: std::collections::HashSet<String> = store
            .names()
            .filter(|n| {
                (layers_go && BF16_LAYER_SUFFIXES.iter().any(|s| n.ends_with(s)))
                    || (conv_go && AUX_CONV_SUFFIXES.iter().any(|s| n.ends_with(s)))
                    || (selector_go && n.ends_with(AUX_SELECTOR_SUFFIX))
            })
            .filter(|n| {
                store
                    .get(n)
                    .is_ok_and(|t| t.dtype == metrale_model_weights::weights::WeightDtype::BF16)
            })
            .map(str::to_owned)
            .collect();
        if doomed.is_empty() {
            return Ok(None);
        }
        // The quantize kernels and the fused K/V copies read these on stream 0.
        gpu.synchronize(0)?;
        let null = metrale_model_layers::weight_map::DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };
        for l in &mut self.layers {
            if layers_go {
                l.q_proj = null;
                l.k_proj = null;
                l.v_proj = null;
                l.o_proj = null;
                l.gate_proj = null;
                l.up_proj = null;
                l.down_proj = null;
            }
            if conv_go {
                for p in [&mut l.attention_conv_proj, &mut l.mlp_conv_proj] {
                    if let Some(w) = p.as_mut() {
                        *w = null;
                    }
                }
            }
        }
        if selector_go && let Some(w) = self.selector_hidden_proj.as_mut() {
            *w = null;
        }
        store.free_matching(gpu, |n| doomed.contains(n)).map(Some)
    }
}

#[cfg(test)]
mod release_credit_tests {
    use crate::dflash_head::BlockDiffusionDraftHead;
    use metrale_gpu_runtime::gpu::GpuBackend;
    use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
    use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

    /// 2026-10-14: The KV credit counts the BF16 copies of the seven layer GEMM weights only,
    /// and nothing when the FP8 drafter or the paged body is off (#82 hybrid M1).
    #[test]
    fn credit_counts_the_bf16_layer_gemms_the_release_frees() {
        let gpu = MockGpuBackend::new();
        let mut map = std::collections::HashMap::new();
        let mut want = 0;
        for l in 0..2 {
            for s in super::BF16_LAYER_SUFFIXES {
                map.insert(
                    format!("layers.{l}{s}"),
                    WeightTensor {
                        ptr: gpu.alloc(256).expect("alloc"),
                        shape: vec![8, 16],
                        dtype: WeightDtype::BF16,
                    },
                );
                want += 8 * 16 * 2;
            }
        }
        for (name, dtype) in [
            ("fc.weight", WeightDtype::BF16),
            (
                "layers.0.attention_conv.kernel_projection.weight",
                WeightDtype::BF16,
            ),
            ("layers.9.self_attn.q_proj.weight", WeightDtype::FP8E4M3),
        ] {
            map.insert(
                name.to_string(),
                WeightTensor {
                    ptr: gpu.alloc(256).expect("alloc"),
                    shape: vec![8, 16],
                    dtype,
                },
            );
        }
        let store = WeightStore::from_map(map);
        let p = BlockDiffusionDraftHead::predicted_bf16_layer_release_bytes;
        assert_eq!(p(&store, true, true), want);
        assert_eq!(
            p(&store, false, true),
            0,
            "METRALE_DFLASH_DRAFTER_FP8=0 keeps BF16"
        );
        assert_eq!(p(&store, true, false), 0, "the non-paged body reads BF16");
    }
}

#[cfg(test)]
mod mirror_prediction_tests {
    use super::fp8_mirror_alloc_bytes as mb;
    use crate::dflash_head::BlockDiffusionDraftHead;
    use crate::weight_loader::dflash_loader::parse_dflash_config;

    /// 2026-10-14: The GLM-5.3 DFlash2 drafter's shapes (serve log 2026-10-14: 5 layers,
    /// hidden 4096, inter 12288 = 2 x 6144, GQA 32/8 = 2 x 16/4, head_dim 128, vocab 154880).
    fn glm() -> crate::weight_loader::DflashConfig {
        parse_dflash_config(
            r#"{"hidden_size": 4096, "num_hidden_layers": 5, "intermediate_size": 12288,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 154880}"#,
        )
        .unwrap()
    }

    /// 2026-10-14: TP1 counts every shape whole; a TP2 rank counts its shard (column-parallel
    /// rows halved, row-parallel columns halved, lm_head vocab halved) with the full rows'
    /// scales for o/down (#82 hybrid M1b).
    #[test]
    fn tp2_rank_predicts_its_shard_and_tp1_the_whole() {
        let c = glm();
        let p = BlockDiffusionDraftHead::predicted_fp8_mirror_bytes;
        let tp1 = p(&c, false, None, true);
        let layer1 = mb(4096, 4096)
            + 2 * mb(1024, 4096)
            + mb(4096, 4096)
            + 2 * mb(12288, 4096)
            + mb(4096, 12288);
        assert_eq!(tp1, 5 * layer1 + mb(154_880, 4096));
        for rank in 0..2 {
            let tp2 = p(&c, true, Some((rank, 2)), true);
            let layer2 = mb(2048, 4096)
                + 2 * mb(512, 4096)
                + mb(4096, 2048)
                + 2 * mb(6144, 4096)
                + mb(4096, 6144);
            assert_eq!(tp2, 5 * layer2 + mb(77_440, 4096), "rank {rank}");
            // Halved up to the unsharded scales of o/down (4 bytes x 4096 rows x 2 x 5).
            assert!(tp2 * 2 >= tp1 && tp2 * 2 <= tp1 + 5 * 2 * 4096 * 4);
        }
        // 2026-10-14: about 763 MiB per rank for the GLM drafter, against 603 MiB for the
        // five q/k/v/gate/up mirrors and the lm_head the ledger lists (o/down are the rest).
        assert_eq!(p(&c, true, Some((0, 2)), true), 800_320_000);
    }

    /// 2026-10-14: Without the TP lever, or off a two-rank serve, the whole drafter is
    /// predicted; METRALE_DFLASH_DRAFTER_FP8=0 predicts nothing.
    #[test]
    fn lever_off_world_one_and_fp8_off() {
        let c = glm();
        let p = BlockDiffusionDraftHead::predicted_fp8_mirror_bytes;
        let whole = p(&c, false, None, true);
        assert_eq!(p(&c, false, Some((0, 2)), true), whole, "lever off");
        assert_eq!(p(&c, true, Some((0, 1)), true), whole, "one rank");
        assert_eq!(p(&c, true, None, true), whole, "no communicator");
        assert_eq!(p(&c, true, Some((1, 2)), false), 0);
        assert_eq!(p(&c, false, None, false), 0);
    }
}
