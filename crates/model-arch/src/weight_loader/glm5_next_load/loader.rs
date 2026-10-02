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
    fn defer_predicate(
        &self,
        config: &ModelConfig,
    ) -> Option<metrale_model_weights::weights::DeferHook> {
        let num_layers = config.num_hidden_layers;
        Some(std::sync::Arc::new(
            move |name: &str, dtype: WeightDtype| is_full_width_mtp_expert(name, dtype, num_layers),
        ))
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
            let t = store.get(&n).with_context(|| {
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
        let kda_ws = std::sync::Arc::new(crate::glm5next_kda::Glm5NextKdaWorkspace::new(
            gpu, &kda_cfg, verify_k,
        )?);

        // 2026-09-25: Unless `METRALE_GLM_MLP_WS_SHARED=0`, one MLP workspace serves
        // every layer; otherwise each layer allocates its own. Either way it is
        // allocated here, at load, before the KV pool is sized.
        // 2026-09-29: The staged prefill (`METRALE_GLM_PREFILL_STAGED=1`) runs the MLP over
        // windows of `prefill_rows_ffn()` rows, so the MLP scratch holds that many; `u_slot`
        // stays at `verify_k` rows (`mlp_ws_bytes_sized`). Staged off, `mlp_rows == verify_k`
        // and the allocation is the unstaged one.
        let mlp_rows = verify_k.max(crate::glm5next_layer::prefill_rows_ffn());
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
            Some(std::sync::Arc::new(
                crate::glm5next_mlp::forward::Glm5NextMlpWorkspace::new_sized(
                    gpu, &mlp_cfg, mlp_rows, verify_k,
                )?,
            ))
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

        // 2026-09-25: The KV pool has `num_attention_layers()` slots, which counts
        // the sparse-attention layers only, so a DSA layer addresses it by its
        // ordinal among DSA layers, not by its model index.
        let mut attn_layer_idx = 0usize;

        for sl in &skeleton.layers {
            let idx = sl.index;
            let t_layer = std::time::Instant::now();
            let src = LayerSource::collect(gpu, store, idx)
                .with_context(|| format!("glm5_next: collecting layer {idx}"))?;
            let t_collect = t_layer.elapsed();

            // 2026-10-01: `mixer_head`: the mixer's FIRST decode projection as the `[n, k]` BF16
            // extent its GEMV reads (KDA `front_end` `q_proj`, DSA `decode_k` `q_a_proj`). Only
            // the first: the projection after it streams more than the 24 MB L2 and would evict
            // anything prefetched further ahead.
            let (mixer, mixer_head) = match sl.mixer {
                Mixer::Kda => {
                    let sharded = KdaShardedSource::new(&src, &kda_plan)?;
                    let (w, _report) = bind_kda_weights(gpu, &kda_cfg, idx, &sharded)?;
                    let head = crate::glm5next_layer::prefetch::bf16_matrix_span(
                        w.q_proj.weight,
                        kda_cfg.qkv_dim(),
                        kda_cfg.hidden,
                    );
                    let mixer = Glm5NextMixer::Kda {
                        layer: Box::new(Glm5NextKdaLayer::new(idx, kda_cfg, w, kda_kernels)?),
                        ws: kda_ws.clone(),
                        cfg: kda_cfg,
                    };
                    (mixer, head)
                }
                Mixer::Dsa => {
                    let load = |n: &str| src.f32(n);
                    let w = build_dsa_weights(gpu, &dsa_cfg, &dsa_plan, &load)?;
                    let head = crate::glm5next_layer::prefetch::bf16_matrix_span(
                        w.q_a_proj,
                        dsa_cfg.q_lora_rank,
                        dsa_cfg.hidden,
                    );
                    let mixer = Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
                        persist_bt: std::env::var("METRALE_GLM_DSA_ALLOC_PER_STEP").as_deref()
                            != Ok("1"),
                        cfg: dsa_cfg,
                        weights: w,
                        kernels: dsa_layer_kernels,
                        select_kernels: dsa_kernels,
                        decode_kernel:
                            crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(gpu)?,
                        workspace: crate::glm5next_dsa::layer::Glm5NextDsaWorkspace::new(
                            gpu, &dsa_cfg, verify_k,
                        )?,
                        layer_idx: idx,
                        attn_layer_idx: {
                            let a = attn_layer_idx;
                            attn_layer_idx += 1;
                            a
                        },
                        rms_eps: config.rms_norm_eps as f32,
                        kv_scale: 1.0,
                    }));
                    (mixer, head)
                }
            };

            let t_mixer = t_layer.elapsed();

            let load = |n: &str| src.f32(n);
            let mlp = match sl.mlp {
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
                    Glm5NextMlpSite::Dense(w) => bf16_matrix_span(
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
                prefetch: crate::glm5next_layer::Glm5NextPrefetch {
                    kernel: l2pf_kernel,
                    ffn_head,
                    next_attn_head: Vec::new(),
                },
            };
            built.push((layer, attn_head));
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
