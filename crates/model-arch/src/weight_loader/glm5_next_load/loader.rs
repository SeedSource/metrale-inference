// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl ModelWeightLoader for Glm5NextWeightLoader`.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use super::*;

impl ModelWeightLoader for Glm5NextWeightLoader {
    /// 2026-09-25: True only when `metrale_config::glm_vision_enabled()`
    /// (`METRALE_GLM_VISION`). The GLM-5.3 config parser reads the same function
    /// before it parses `vision_config`; this method takes no config, so the
    /// gate is the environment.
    fn binds_vision_encoder(&self) -> bool {
        metrale_config::glm_vision_enabled()
    }

    /// 2026-09-25: Bind the `model.visual.*` tower; see
    /// [`crate::weight_loader::glm5_next_vision`].
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<metrale_model_layers::layers::VisionTower>> {
        crate::weight_loader::glm5_next_vision::load_glm5_next_vision(store, config, gpu)
    }

    /// 2026-09-25: Keep the MTP layer's BF16 routed-expert weights off the
    /// device (`is_full_width_mtp_expert`); `bind_expert` quantises them from
    /// disk. The layer is `num_hidden_layers`. A checkpoint whose MTP experts
    /// are U8 defers nothing.
    /// 2026-10-03: Under `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1` also every `*.input_scale`
    /// (`defer_rule`); `bind_expert` reads the routed experts' from disk. Off, the rule is the
    /// one above.
    fn defer_predicate(
        &self,
        config: &ModelConfig,
    ) -> Option<metrale_model_weights::weights::DeferHook> {
        let num_layers = config.num_hidden_layers;
        let defer_scales = metrale_config::glm_moe_prefill_cutlass_w4a4();
        Some(std::sync::Arc::new(move |name: &str, dtype: WeightDtype| {
            defer_rule(name, dtype, num_layers, defer_scales)
        }))
    }

    /// 2026-09-25: DSA, KDA and the MLP all shard under TP (see `glm5_next_load.rs`);
    /// routed experts are also split by EP (`local_expert_range`).
    fn supports_tp(&self) -> bool {
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        self.build_layers(StoreAccess::Shared(store), config, gpu, layer_kv_dtypes)
    }

    /// 2026-10-02: `load_layers` with a mutable store, so each text layer's raw store
    /// tensors are freed right after the layer is built (`METRALE_LOAD_EARLY_FREE`,
    /// default on). See [`StoreAccess::release_layer`] for the invariant.
    fn load_layers_releasing(
        &self,
        store: &mut WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let access = if early_free_enabled() {
            StoreAccess::Exclusive(store)
        } else {
            StoreAccess::Shared(&*store)
        };
        self.build_layers(access, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.language_model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.language_model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    /// 2026-09-25: `None`: the GLM-5.3 MTP layer is loaded by
    /// `glm5_next_mtp::load_glm5next_mtp_module`, not as `MtpWeights`.
    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::weight_loader::MtpWeights>> {
        Ok(None)
    }

    /// 2026-09-25: Free the store tensors `is_reuploaded` matches, then those
    /// `is_quantized_expert_weight` matches.
    fn prune_after_load(
        &self,
        store: &mut WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let n = config.num_hidden_layers;
        let (count, bytes) = store.free_matching(gpu, |name| is_reuploaded(name, n))?;
        tracing::info!(
            "glm5_next: released {count} store tensors ({:.2} GB) already re-uploaded by the \
             binders; routed experts and the MTP block kept",
            bytes as f64 / 1e9,
        );
        // 2026-10-05: The MTP `eh_proj` BF16 original, when `METRALE_GLM_DENSE_FP8=1` kept only an
        // FP8 copy of it (`dense_fp8::register_mtp`); the store owns the buffer, so it is freed here.
        let eh_name = format!("model.language_model.layers.{n}.eh_proj.weight");
        if let Ok(t) = store.get(&eh_name)
            && crate::glm5next_layer::dense_fp8::take_deferred_free(t.ptr)
        {
            let (c, b) = store.free_matching(gpu, |name| name == eh_name)?;
            tracing::info!(
                "glm5_next: released {c} MTP eh_proj tensor ({:.1} MB) replaced by its FP8 copy",
                b as f64 / 1e6
            );
        }
        let quantized: std::collections::BTreeSet<String> = store
            .names()
            .filter(|n| {
                store
                    .get(n)
                    .is_ok_and(|t| is_quantized_expert_weight(n, t.dtype))
            })
            .map(str::to_string)
            .collect();
        if !quantized.is_empty() {
            let (qcount, qbytes) = store.free_matching(gpu, |name| quantized.contains(name))?;
            tracing::info!(
                "glm5_next: released {qcount} full-width BF16 routed-expert tensors \
                 ({:.2} GB) quantised to NVFP4 at bind time",
                qbytes as f64 / 1e9,
            );
        }
        Ok(())
    }
}

/// 2026-10-02: `METRALE_LOAD_EARLY_FREE`: default on; `0` restores freeing everything in
/// `prune_after_load`.
fn early_free_enabled() -> bool {
    std::env::var("METRALE_LOAD_EARLY_FREE").as_deref() != Ok("0")
}

/// 2026-10-02: How `build_layers` sees the store: shared (`load_layers`) or exclusive
/// (`load_layers_releasing`), so one body serves both without changing the trait's
/// `load_layers(&WeightStore)` signature.
enum StoreAccess<'a> {
    Shared(&'a WeightStore),
    Exclusive(&'a mut WeightStore),
}

impl StoreAccess<'_> {
    fn shared(&self) -> &WeightStore {
        match self {
            StoreAccess::Shared(s) => *s,
            StoreAccess::Exclusive(s) => &**s,
        }
    }

    fn can_release(&self) -> bool {
        matches!(self, StoreAccess::Exclusive(_))
    }

    /// Free layer `idx`'s store tensors that `prune_after_load` would free anyway: the
    /// `is_reuploaded` set (non-expert tensors of a text layer) and its BF16 routed-expert
    /// projections that `bind_expert` quantised into separate buffers. Returns (count, bytes).
    ///
    /// Invariant (checked by reading the code, 2026-10-02): once layer `idx` is built, nothing
    /// reads these tensors. `LayerSource::collect` copied them to the host; the binders upload
    /// their own buffers; layer `idx`'s U8 experts (`mlp.experts.*`, zero-copy) and the MTP layer
    /// (`idx >= num_layers`) are never matched, and the MTP loader, embedding, final norm,
    /// lm_head and vision tower read other names only. The only cross-layer read, layer 0's
    /// `f_a_proj` shape, happens before the loop. Callers must not hold a shared borrow from
    /// [`Self::shared`] across this call. A no-op on a shared store.
    fn release_layer(
        &mut self,
        gpu: &dyn GpuBackend,
        idx: usize,
        num_layers: usize,
    ) -> Result<(usize, usize)> {
        let StoreAccess::Exclusive(store) = self else {
            return Ok((0, 0));
        };
        let prefix = format!("model.language_model.layers.{idx}.");
        let quantized: std::collections::BTreeSet<String> = store
            .names()
            .filter(|n| n.starts_with(&prefix))
            .filter(|n| {
                store
                    .get(n)
                    .is_ok_and(|t| is_quantized_expert_weight(n, t.dtype))
            })
            .map(str::to_string)
            .collect();
        store.free_matching(gpu, |name| {
            name.starts_with(&prefix)
                && (is_reuploaded(name, num_layers) || quantized.contains(name))
        })
    }
}

impl Glm5NextWeightLoader {
    fn build_layers(
        &self,
        mut access: StoreAccess<'_>,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let skeleton = Glm5NextTextSkeleton::from_config(config)?;
        // 2026-09-25: `l2_eps` and `chunk` are fixed here; every other field is
        // read from the config.
        let kda_cfg = Glm5NextKdaConfig {
            hidden: config.hidden_size,
            heads: config.linear_num_value_heads,
            head_dim: config.linear_value_head_dim,
            conv_kernel: config.linear_conv_kernel_dim,
            gate_lower_bound: config.linear_gate_lower_bound,
            rms_norm_eps: config.rms_norm_eps as f32,
            l2_eps: 1e-6,
            chunk: 32,
        };
        kda_cfg.validate()?;
        // 2026-09-25: `gate_rank` is the row count of layer 0's `f_a_proj`; the
        // config has no such key.
        let gate_rank = {
            let n = qualify(0, "self_attn.f_a_proj.weight");
            let t = access.shared().get(&n).with_context(|| {
                format!("glm5_next: {n} is needed to size the KDA gate bottleneck")
            })?;
            *t.shape.first().context("f_a_proj has no rows")?
        };
        let kda_plan = KdaTpPlan::from_config(config, gate_rank)?;
        let dsa_cfg = Glm5NextDsaConfig::from_config(config)?;
        let mlp_cfg = Glm5NextMlpConfig::from_config(config)?;

        let kda_kernels = Glm5NextKdaKernels::resolve(gpu)?;
        let dsa_kernels = Glm5NextDsaKernels::resolve(gpu)?;
        let dsa_layer_kernels = Glm5NextDsaLayerKernels::resolve(gpu)?;
        let mlp_kernels = Glm5NextMlpKernels::resolve(gpu)?;
        let mhc_kernels_probe = Glm5NextMhcKernels::resolve(gpu)?;
        let rms_norm_k = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
        // 2026-09-25: Only a layer without mHC (the MTP layer) uses `add_k`, and it
        // fails there if the kernel is missing; `try_kernel` lets a target without
        // it still build the text stack.
        let add_k = metrale_model_layers::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace");

        // 2026-09-25: Every KDA layer shares one workspace (one `kda_cfg`). The
        // workspaces are sized for `verify_k` rows, the largest of the batched
        // GEMV's `DENSE_GEMV_BATCHM_MAX_M`, `PREFILL_ROWS` and `prefill_rows()`
        // (`METRALE_GLM_PREFILL_ROWS`).
        let verify_k = (metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize)
            .max(crate::glm5next_layer::PREFILL_ROWS)
            .max(crate::glm5next_layer::prefill_rows());
        // 2026-10-01: `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1` runs a KDA layer's `decode_k`
        // over a whole staged window (`fullwidth_rows()`, the FFN window), so the workspace
        // holds that many rows. Its chunked-scan buffers stay at `verify_k` rows unless
        // `METRALE_GLM_KDA_CHUNK_PREFILL=1` sends the window through the chunked scan. GLM-5.3
        // TP2 (32 heads x 128, hidden 4096): 139,712 B per row without the chunk buffers (98,304 B
        // more with them), so ~0.54 GB more than at 256 rows at a 4096-row window. Lever off,
        // `kda_rows == kda_chunk_rows == verify_k`: the allocation `new` made before.
        let wide_rows = crate::glm5next_layer::fullwidth_rows();
        // 2026-10-02: `METRALE_GLM_BATCHED_VERIFY=1` also widens the KDA and MLP scratch to
        // `batched_verify_rows()` (64 rows; KDA 139,712 B per row at TP2, plus the MLP rows);
        // off it adds 0.
        let bv_rows = crate::glm5next_layer::levers::batched_verify_rows();
        let kda_rows = verify_k.max(wide_rows.unwrap_or(0)).max(bv_rows);
        let kda_chunk_rows = if crate::glm5next_layer::kda_chunk_prefill() {
            kda_rows
        } else {
            verify_k
        };
        if wide_rows.is_some() {
            let b = |t: usize, c: usize| {
                crate::glm5next_kda::Glm5NextKdaWorkspace::bytes_split(&kda_cfg, t, c) as f64
            };
            tracing::warn!(
                "GLM KDA workspace (METRALE_GLM_PREFILL_FULLWIDTH_GEMM): {kda_rows} rows, chunk \
                 buffers {kda_chunk_rows} rows, {:.1} MB ({:+.1} MB against {verify_k} rows)",
                b(kda_rows, kda_chunk_rows) / 1e6,
                (b(kda_rows, kda_chunk_rows) - b(verify_k, verify_k)) / 1e6,
            );
        }
        // 2026-09-29: The staged prefill (`METRALE_GLM_PREFILL_STAGED=1`) runs the MLP over
        // windows of `prefill_rows_ffn()` rows, so the MLP scratch holds that many; `u_slot`
        // stays at `verify_k` rows (`mlp_ws_bytes_sized`). Staged off, `mlp_rows == verify_k`
        // and the allocation is the unstaged one.
        let mlp_rows = verify_k.max(crate::glm5next_layer::prefill_rows_ffn()).max(bv_rows);
        // 2026-10-05: `METRALE_GLM_PREFILL_SCRATCH_UNION=1`: the KDA workspace, the shared MLP
        // workspace and the DSA wide arena start at one base in ONE allocation sized to the
        // largest. Invariant: all three are pure per-call scratch on one stream (nothing carried
        // across sublayers/windows/layers/requests); decode/verify graphs bake KDA/MLP pointers,
        // so the union is allocated here, at load, and never moved, freed or resized. Out of the
        // union: FlashKDA scratch, CUTLASS workspace/SFB cache, W8A8 scratch, DSA xseq arena,
        // per-layer DSA workspaces, the MTP head's workspaces (`glm5next_layer::scratch_union`).
        // Off: `None`, and every member below is built by its plain constructor as before.
        let scratch = if crate::glm5next_layer::scratch_union::scratch_union() {
            crate::glm5next_layer::scratch_union::plan_glm(
                gpu,
                (&kda_cfg, kda_rows, kda_chunk_rows),
                crate::glm5next_mlp::forward::mlp_ws_shared()
                    .then_some((&mlp_cfg, mlp_rows, verify_k)),
                wide_rows.map(|r| (&dsa_cfg, r)),
            )?
        } else {
            None
        };
        let mut kda_ws_inner = match &scratch {
            Some(u) => u.place(crate::glm5next_layer::scratch_union::KDA, |a| {
                crate::glm5next_kda::Glm5NextKdaWorkspace::new_split_in(
                    &kda_cfg,
                    kda_rows,
                    kda_chunk_rows,
                    a,
                )
            })?,
            None => crate::glm5next_kda::Glm5NextKdaWorkspace::new_split(
                gpu,
                &kda_cfg,
                kda_rows,
                kda_chunk_rows,
            )?,
        };
        // 2026-10-03: `METRALE_GLM_KDA_PREFILL_FLASHKDA=1` adds the FlashKDA scratch (library
        // workspace for min(kda_rows, 4096) rows plus one transposed recurrent state; 9.2 MB at
        // 256 rows, 115 MB at 4096 rows for GLM-5.3 TP2), here at load before the KV pool is
        // sized. Off, nothing is allocated.
        if crate::glm5next_kda::kda_prefill_flashkda() {
            let bytes = kda_ws_inner.alloc_flashkda(gpu, &kda_cfg)?;
            if bytes > 0 {
                tracing::warn!(
                    "GLM KDA FlashKDA scratch: {:.1} MB for {kda_rows}-row KDA calls",
                    bytes as f64 / 1e6
                );
            } else {
                tracing::warn!(
                    "METRALE_GLM_KDA_PREFILL_FLASHKDA=1 but no FlashKDA scratch was allocated \
                     (library built: {}); the KDA prefill keeps its other arms",
                    metrale_gpu_runtime::flashkda::available()
                );
            }
        }
        let kda_ws = std::sync::Arc::new(kda_ws_inner);

        // 2026-10-03: `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1` adds the per-layer swizzled
        // weight-scale cache (216 MB at GLM-5.3 EP=2) and the CUTLASS workspace, here at load
        // before the KV pool is sized. Off, nothing is allocated; a build without CUTLASS or an
        // unsupported shape logs the refusal and the routed-MoE prefill stays W4A16.
        crate::glm5next_mlp::forward_prefill_gemm::cutlass_w4a4::prepare_at_load(gpu, &mlp_cfg)?;

        // 2026-09-25: Unless `METRALE_GLM_MLP_WS_SHARED=0`, one MLP workspace serves
        // every layer; otherwise each layer allocates its own. Either way it is
        // allocated here, at load, before the KV pool is sized.
        // 2026-09-29: `mlp_rows` (computed above the KDA workspace) is the staged-prefill width.
        let mlp_ws_bytes =
            crate::glm5next_mlp::forward::mlp_ws_total_bytes_sized(&mlp_cfg, mlp_rows, verify_k);
        let shared_mlp_ws = if crate::glm5next_mlp::forward::mlp_ws_shared() {
            tracing::info!(
                "GLM MLP workspace: SHARED, 1 x {:.1} MB for {} layers at {mlp_rows} rows \
                 (per-layer would be {:.1} MB)",
                mlp_ws_bytes as f64 / 1e6,
                skeleton.layers.len(),
                (mlp_ws_bytes * skeleton.layers.len()) as f64 / 1e6,
            );
            Some(std::sync::Arc::new(match &scratch {
                Some(u) => u.place(crate::glm5next_layer::scratch_union::MLP, |a| {
                    crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new_sized_in(
                        &mlp_cfg, mlp_rows, verify_k, a,
                    )
                })?,
                None => crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new_sized(
                    gpu, &mlp_cfg, mlp_rows, verify_k,
                )?,
            }))
        } else {
            tracing::warn!(
                "GLM MLP workspace: PER-LAYER, {} x {:.1} MB at {mlp_rows} rows",
                skeleton.layers.len(),
                mlp_ws_bytes as f64 / 1e6,
            );
            None
        };

        let dsa_plan = crate::glm5next_dsa::tp::DsaTpPlan::new(
            config.tp_rank,
            config.tp_world_size.max(1),
            &dsa_cfg,
        )?;
        // 2026-10-01: `METRALE_GLM_PREFILL_FULLWIDTH_GEMM=1`: one window-wide DSA arena for
        // `decode_k_wide`, shared by every DSA layer (they run one after another on one
        // stream), allocated here, at load, before the KV pool is sized. GLM-5.3 TP2:
        // 89,232 B per row, 365.5 MB at a 4096-row window (`DsaWideArena` doc).
        let dsa_wide = match wide_rows {
            Some(r) => {
                let per_row = crate::glm5next_dsa::layer::DsaWideArena::bytes_per_row(&dsa_cfg);
                tracing::warn!(
                    "GLM DSA wide arena (METRALE_GLM_PREFILL_FULLWIDTH_GEMM): 1 x {:.1} MB for \
                     {r} rows, shared by the DSA layers",
                    (per_row * r) as f64 / 1e6
                );
                Some(std::sync::Arc::new(match &scratch {
                    Some(u) => u.place(crate::glm5next_layer::scratch_union::DSA_WIDE, |a| {
                        crate::glm5next_dsa::layer::DsaWideArena::new_in(&dsa_cfg, r, a)
                    })?,
                    None => crate::glm5next_dsa::layer::DsaWideArena::new(gpu, &dsa_cfg, r)?,
                }))
            }
            None => None,
        };
        // 2026-10-03: `METRALE_GLM_DSA_XSEQ_BATCH=1`: one cross-sequence DSA arena for
        // `decode_xseq` (`DENSE_GEMV_BATCHM_MAX_M` rows), shared by every DSA layer like the
        // wide arena. GLM-5.3 TP2: 89,728 B per row, 1.4 MB (`DsaXseqArena` doc).
        let dsa_xseq = if crate::glm5next_dsa::layer::dsa_xseq_batch() {
            let r = metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize;
            let a = crate::glm5next_dsa::layer::DsaXseqArena::new(gpu, &dsa_cfg, r)?;
            tracing::warn!(
                "GLM DSA cross-sequence arena (METRALE_GLM_DSA_XSEQ_BATCH): 1 x {:.2} MB for \
                 {} rows, shared by the DSA layers",
                (crate::glm5next_dsa::layer::DsaXseqArena::bytes_per_row(&dsa_cfg) * a.max_rows())
                    as f64
                    / 1e6,
                a.max_rows()
            );
            Some(std::sync::Arc::new(a))
        } else {
            None
        };
        let last = skeleton.layers.len() - 1;
        let mut out: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(skeleton.layers.len());
        // 2026-10-01: Decode L2 prefetch (`METRALE_GLM_DECODE_L2_PREFETCH`): the kernel, optional
        // (0 on a target without it), and each built layer with its attention-head spans, so a
        // layer can be handed the NEXT layer's head once every layer exists.
        let l2pf_kernel = metrale_model_layers::layers::try_kernel(
            gpu,
            crate::glm5next_layer::prefetch::L2_PREFETCH_MODULE,
            crate::glm5next_layer::prefetch::L2_PREFETCH_KERNEL,
        );
        let mut built: Vec<(Glm5NextLayer, Vec<crate::glm5next_layer::L2Span>)> =
            Vec::with_capacity(skeleton.layers.len());

        // 2026-10-01: DFlash tap layers (`config.dflash_capture_layers`, the drafter's
        // `target_layer_ids`) collapse their highway into `hidden` for the engine's capture.
        // Only under `METRALE_GLM_DFLASH=1`; the factory refuses a GLM drafter without it
        // (`glm_dflash_gate`). An out-of-range tap fails the load.
        let dflash_taps = if metrale_model_layers::speculative::glm_dflash::glm_dflash_enabled() {
            metrale_model_layers::speculative::glm_dflash::glm_dflash_tap_flags(
                &config.dflash_capture_layers,
                skeleton.layers.len(),
            )?
        } else {
            vec![false; skeleton.layers.len()]
        };
        if dflash_taps.iter().any(|&t| t) {
            tracing::info!(
                "glm5_next: DFlash tap collapse (hc_head_mean) at layers {:?}",
                config.dflash_capture_layers
            );
        }

        // 2026-09-25: The KV pool has `num_attention_layers()` slots, which counts
        // the sparse-attention layers only, so a DSA layer addresses it by its
        // ordinal among DSA layers, not by its model index.
        let mut attn_layer_idx = 0usize;

        let early_free = access.can_release();
        let mut fp8_total = crate::glm5next_layer::dense_fp8::LayerFp8::default();
        let (mut freed_count, mut freed_bytes) = (0usize, 0usize);
        for sl in &skeleton.layers {
            // 2026-10-02: Shared borrow of the store for this iteration only (shadows the
            // name the binders use); it must be dead before `release_layer` below.
            let store = access.shared();
            let idx = sl.index;
            let t_layer = std::time::Instant::now();
            let src = LayerSource::collect(gpu, store, idx)
                .with_context(|| format!("glm5_next: collecting layer {idx}"))?;
            let t_collect = t_layer.elapsed();

            let mut mixer = match sl.mixer {
                Mixer::Kda => {
                    let sharded = KdaShardedSource::new(&src, &kda_plan)?;
                    let (w, _report) = bind_kda_weights(gpu, &kda_cfg, idx, &sharded)?;
                    Glm5NextMixer::Kda {
                        layer: Box::new(Glm5NextKdaLayer::new(idx, kda_cfg, w, kda_kernels)?),
                        ws: kda_ws.clone(),
                        cfg: kda_cfg,
                    }
                }
                Mixer::Dsa => {
                    let load = |n: &str| src.f32(n);
                    let w = build_dsa_weights(gpu, &dsa_cfg, &dsa_plan, &load)?;
                    Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
                        persist_bt: std::env::var("METRALE_GLM_DSA_ALLOC_PER_STEP").as_deref()
                            != Ok("1"),
                        cfg: dsa_cfg,
                        weights: w,
                        kernels: dsa_layer_kernels,
                        select_kernels: dsa_kernels,
                        decode_kernel:
                            crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(gpu)?,
                        workspace: {
                            let ws = crate::glm5next_dsa::layer::Glm5NextDsaWorkspace::new(
                                gpu, &dsa_cfg, verify_k,
                            )?;
                            let ws = match &dsa_wide {
                                Some(a) => ws.with_wide(a.clone()),
                                None => ws,
                            };
                            match &dsa_xseq {
                                Some(a) => ws.with_xseq(a.clone()),
                                None => ws,
                            }
                        },
                        layer_idx: idx,
                        attn_layer_idx: {
                            let a = attn_layer_idx;
                            attn_layer_idx += 1;
                            a
                        },
                        rms_eps: config.rms_norm_eps as f32,
                        kv_scale: 1.0,
                    }))
                }
            };

            let t_mixer = t_layer.elapsed();

            let load = |n: &str| src.f32(n);
            let mut mlp = match sl.mlp {
                Mlp::Dense => Glm5NextMlpSite::Dense(mlp_build::build_dense_mlp(
                    gpu,
                    &mlp_cfg,
                    config.tp_rank,
                    config.intermediate_size,
                    "mlp",
                    &load,
                )?),
                Mlp::RoutedMoe => {
                    let expert = |id: usize| bind_expert(gpu, store, idx, id);
                    Glm5NextMlpSite::Moe(Box::new(mlp_build::build_moe(
                        gpu,
                        &mlp_cfg,
                        config.tp_rank,
                        config.shared_expert_intermediate_size,
                        &load,
                        &expert,
                    )?))
                }
            };

            let t_mlp = t_layer.elapsed();
            // 2026-10-03: `METRALE_GLM_DENSE_FP8=1`: FP8 copies of this text layer's dense
            // projections replace the BF16 originals, which are freed and whose fields now name
            // the copies (`glm5next_layer::dense_fp8`); a no-op when the lever is off.
            let fp8 = crate::glm5next_layer::dense_fp8::register_layer(
                gpu, &mut mixer, &mut mlp, &mlp_cfg,
            )
            .with_context(|| format!("glm5_next: FP8 dense copies of layer {idx}"))?;
            fp8_total.fp8_bytes += fp8.fp8_bytes;
            fp8_total.bf16_freed += fp8.bf16_freed;
            // 2026-10-01: `mixer_head`: the mixer's FIRST decode projection as the `[n, k]`
            // extent its GEMV reads (KDA `front_end` `q_proj`, DSA `decode_k` `q_a_proj`). Only
            // the first: the projection after it streams more than the 24 MB L2 and would evict
            // anything prefetched further ahead. 2026-10-03: read from the field after
            // `register_layer`, so under the FP8 lever it names the FP8 copy (the BF16 original
            // is freed).
            let mixer_head = match &mixer {
                Glm5NextMixer::Kda { layer, .. } => crate::glm5next_layer::dense_fp8::decode_span(
                    layer.weights.q_proj.weight,
                    kda_cfg.qkv_dim(),
                    kda_cfg.hidden,
                ),
                Glm5NextMixer::Dsa(l) => crate::glm5next_layer::dense_fp8::decode_span(
                    l.weights.q_a_proj,
                    dsa_cfg.q_lora_rank,
                    dsa_cfg.hidden,
                ),
            };
            tracing::info!(
                "glm5_next layer {idx} built: collect {:.2}s mixer {:.2}s mlp {:.2}s \
                 (mixer={:?} mlp={:?})",
                t_collect.as_secs_f64(),
                (t_mixer - t_collect).as_secs_f64(),
                (t_mlp - t_mixer).as_secs_f64(),
                sl.mixer,
                sl.mlp,
            );

            let mhc = if sl.hyper_connection {
                Some(Glm5NextMhc {
                    kernels: mhc_kernels_probe,
                    attn: bind_mhc_site(gpu, &src, "attn", config.hc_mult, config.hidden_size)?,
                    ffn: bind_mhc_site(gpu, &src, "ffn", config.hc_mult, config.hidden_size)?,
                    hc_mult: config.hc_mult,
                    sinkhorn_iters: config.hc_sinkhorn_iters,
                    hc_eps: config.hc_eps,
                })
            } else {
                None
            };

            // 2026-10-01: Prefetch heads. Attention: the attention-site `hc_fn`, then the mixer
            // head. FFN: the FFN-site `hc_fn`, then the router `[num_experts, hidden]` (MoE) or
            // the dense `gate_proj` `[local_dense_intermediate, hidden]`, all BF16.
            let (attn_head, ffn_head) = {
                use crate::glm5next_layer::prefetch::{bf16_matrix_span, mhc_site_span};
                let mut attn = Vec::new();
                let mut ffn = Vec::new();
                if let Some(m) = mhc.as_ref() {
                    attn.push(mhc_site_span(&m.attn, m.hc_mult, config.hidden_size));
                    ffn.push(mhc_site_span(&m.ffn, m.hc_mult, config.hidden_size));
                }
                attn.push(mixer_head);
                ffn.push(match &mlp {
                    Glm5NextMlpSite::Moe(w) => {
                        bf16_matrix_span(w.router, mlp_cfg.num_experts, mlp_cfg.hidden)
                    }
                    Glm5NextMlpSite::Dense(w) => crate::glm5next_layer::dense_fp8::decode_span(
                        w.gate_proj,
                        mlp_cfg.local_dense_intermediate,
                        mlp_cfg.hidden,
                    ),
                });
                (attn, ffn)
            };

            let layer = Glm5NextLayer {
                layer_idx: idx,
                mixer,
                mlp,
                mlp_cfg,
                mlp_kernels,
                mlp_ws: match &shared_mlp_ws {
                    Some(ws) => ws.clone(),
                    None => std::sync::Arc::new(
                        crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new_sized(
                            gpu, &mlp_cfg, mlp_rows, verify_k,
                        )?,
                    ),
                },
                mhc,
                input_norm: upload_f32_as_bf16(gpu, &src.f32("input_layernorm.weight")?)?,
                post_attn_norm: upload_f32_as_bf16(
                    gpu,
                    &src.f32("post_attention_layernorm.weight")?,
                )?,
                rms_norm_k,
                add_k,
                rms_eps: config.rms_norm_eps as f32,
                hidden: config.hidden_size,
                mixer_all_reduce: match sl.mixer {
                    Mixer::Kda => kda_plan.needs_output_all_reduce(),
                    Mixer::Dsa => dsa_plan.needs_output_all_reduce(),
                },
                is_first: idx == 0,
                is_last: idx == last,
                dflash_tap: dflash_taps.get(idx).copied().unwrap_or(false),
                prefetch: crate::glm5next_layer::Glm5NextPrefetch {
                    kernel: l2pf_kernel,
                    ffn_head,
                    next_attn_head: Vec::new(),
                },
            };
            // 2026-10-02: Layer `idx` is built: every binder above read its own layer's store
            // tensors (host copy in `src`, or the routed experts via `bind_expert`), and
            // nothing later reads them (see `StoreAccess::release_layer`).
            if early_free {
                let (c, b) = access
                    .release_layer(gpu, idx, config.num_hidden_layers)
                    .with_context(|| format!("glm5_next: early-free of layer {idx}"))?;
                freed_count += c;
                freed_bytes += b;
                tracing::info!(
                    "glm5_next early-free layer {idx}: released {c} raw store tensors \
                     ({:.1} MB)",
                    b as f64 / 1e6,
                );
            }
            built.push((layer, attn_head));
        }
        // 2026-10-03: The prefill dequant arena, allocated here so the KV budget counts it.
        let arena = crate::glm5next_layer::dense_fp8::finish_load(gpu)
            .context("glm5_next: METRALE_GLM_DENSE_FP8 dequant arena")?;
        if crate::glm5next_layer::dense_fp8::dense_fp8() {
            tracing::info!(
                "METRALE_GLM_DENSE_FP8: {:.2} GB of BF16 dense weights per rank freed, replaced by \
                 {:.2} GB of FP8 copies; prefill dequant arena {:.1} MB; net {:+.2} GB/rank vs BF16",
                fp8_total.bf16_freed as f64 / 1e9,
                fp8_total.fp8_bytes as f64 / 1e9,
                arena as f64 / 1e6,
                (fp8_total.fp8_bytes + arena) as f64 / 1e9 - fp8_total.bf16_freed as f64 / 1e9,
            );
        }
        if early_free {
            tracing::info!(
                "glm5_next early-free total: released {freed_count} raw store tensors \
                 ({:.2} GB) during load_layers; prune_after_load sweeps the rest",
                freed_bytes as f64 / 1e9,
            );
        } else {
            tracing::info!(
                "glm5_next early-free: OFF (METRALE_LOAD_EARLY_FREE=0 or immutable store); raw \
                 store copies stay until prune_after_load"
            );
        }
        // 2026-10-01: Layer i prefetches layer i + 1's attention head; the last layer none.
        let heads: Vec<Vec<crate::glm5next_layer::L2Span>> =
            built.iter().map(|(_, h)| h.clone()).collect();
        for (i, (mut layer, _)) in built.into_iter().enumerate() {
            if let Some(next) = heads.get(i + 1) {
                layer.prefetch.next_attn_head = next.clone();
            }
            out.push(Box::new(layer));
        }
        Ok(out)
    }
}
