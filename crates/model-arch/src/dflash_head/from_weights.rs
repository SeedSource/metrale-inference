// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Construction of the DFlash drafter head from its loaded weights: kernel
//! handles, the paged BF16 drafter KV pool, scratch, the RoPE table, the fused K/V
//! weight and, by default, FP8 weight copies.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

mod fp8_weights;
mod kernel_handles;
mod rope_table;

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use parking_lot::Mutex;

use super::{BlockDiffusionDraftHead, DflashLayer, DflashQuantization, DflashScratch};
use crate::weight_loader::DflashWeights;

impl BlockDiffusionDraftHead {
    pub fn from_weights(
        weights: DflashWeights,
        embed_tokens_shared: DevicePtr,
        lm_head_shared: DevicePtr,
        lm_head_nvfp4: Option<metrale_model_layers::weight_map::QuantizedWeight>,
        lm_head_native_fp8: Option<(metrale_model_layers::weight_map::Fp8DenseWeight, usize)>,
        target_hidden_size: usize,
        gamma: Option<usize>,
        window_size: Option<usize>,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
        // 2026-09-25: The widest cross-sequence batch one forward serves. The
        // gamma-sized scratch holds this many bands of gamma rows, and the drafter KV
        // pool holds this many sequences' blocks.
        max_batch_size: usize,
    ) -> Result<Self> {
        let nb = max_batch_size.max(1);
        let target_layer_ids = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.clone())
            .unwrap_or_default();
        let mask_token_id = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.mask_token_id)
            .unwrap_or(0);

        if target_layer_ids.is_empty() {
            anyhow::bail!(
                "DFlash drafter config.json has no `dflash_config.target_layer_ids` — \
                 cannot determine which target hidden states to capture"
            );
        }

        let _ = target_hidden_size;

        let num_layers = weights.config.num_hidden_layers;
        let hidden_size = weights.config.hidden_size;
        let intermediate_size = weights.config.intermediate_size;
        let num_q_heads = weights.config.num_attention_heads;
        let num_kv_heads = weights.config.num_key_value_heads;
        let head_dim = weights.config.head_dim;
        let vocab_size = weights.config.vocab_size;
        // 2026-09-25: `gamma` is `--dflash-gamma` when given, else `default_dflash_gamma`
        // of the drafter's `effective_block_size()` (`dflash_config.block_size`, else the
        // top-level `block_size`, which defaults to 16). The model factory sizes its
        // pools with the same helper (factory/build.rs), so this must use it too. The
        // serve reads the head's gamma back through `block_gamma()`.
        let gamma_val = gamma.unwrap_or_else(|| {
            metrale_model_layers::layers::qwen3_ssm::default_dflash_gamma(
                weights.config.effective_block_size(),
            )
        });

        // 2026-09-25: The drafter's paged KV cache: one cache for all drafter layers,
        // in 16-token blocks.
        let block_size = 16;
        let kv_config = KvCacheConfig {
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            // 2026-09-25: BF16, the type the drafter's paged attention
            // (`attn_prefill_paged_indirect`) reads and `reshape_and_cache_flash` writes.
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
            v_aliases_k: false,
        };
        // 2026-09-25: Each sequence's lazy block allocation in propose.rs takes
        // `ceil((max_ctx_len + gamma + 1) / 16)` blocks, and `max_ctx_len <=
        // max_seq_len`, so `per_seq_blocks` covers one sequence. The pool holds
        // `max_batch_size` of them plus one spare block.
        // provenance-id: 526f6e616c6420522e205374657369616b
        let per_seq_blocks = (max_seq_len + gamma_val + 1).div_ceil(block_size);
        let num_blocks = per_seq_blocks * max_batch_size.max(1) + 1;
        tracing::info!(
            "DFlash drafter paged KV pool: {} blocks ({} per-seq x max_batch_size {})",
            num_blocks,
            per_seq_blocks,
            max_batch_size.max(1)
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;

        // 2026-09-25: A `gpu.kernel(..)?` handle is required and fails construction
        // when absent; a `try_kernel` handle is `KernelHandle(0)` when absent.
        let kernels = kernel_handles::load_kernels(gpu)?;

        // 2026-09-25: Scratch, allocated once. The row buffers hold `rows_max` rows: the
        // non-paged path's `ctx_window` ctx rows plus `gamma` block rows, or `nb` bands of
        // `gamma` rows for a batched propose, whichever is more.
        let bf16 = 2usize;
        let g = gamma_val;
        // 2026-09-25: `ctx_window` bounds the ctx rows the non-paged path attends to and
        // the rows one ctx precompute call takes. METRALE_DFLASH_CTX_WINDOW (default
        // 4096), read here at construction. The row scratch grows linearly with it.
        let ctx_window: usize = std::env::var("METRALE_DFLASH_CTX_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        tracing::info!(
            "DFlash ctx_window = {} (set METRALE_DFLASH_CTX_WINDOW to override; \
             drafter trained on full captured prefix — larger is better, \
             scratch grows linearly)",
            ctx_window
        );
        let n_attn = g + ctx_window;
        let rows_max = n_attn.max(nb * gamma_val);
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let scratch = DflashScratch {
            stream_buf: gpu.alloc(rows_max * hidden_size * bf16)?,
            norm_buf: gpu.alloc(rows_max * hidden_size * bf16)?,
            q_buf: gpu.alloc(rows_max * q_dim * bf16)?,
            k_buf: gpu.alloc(rows_max * kv_dim * bf16)?,
            v_buf: gpu.alloc(rows_max * kv_dim * bf16)?,
            attn_out: gpu.alloc(rows_max * q_dim * bf16)?,
            mlp_intermediate: gpu.alloc(rows_max * intermediate_size * bf16)?,
            mlp_up: gpu.alloc(rows_max * intermediate_size * bf16)?,
            stream_acc: gpu.alloc(rows_max * hidden_size * bf16)?,
            fc_proj: gpu.alloc(ctx_window * hidden_size * bf16)?,
            // 2026-09-25: The precompute's fused K/V output: up to `ctx_window` rows
            // (`precompute_ctx_kv` refuses more), `L * 2 * kv_dim` BF16 per row.
            fused_kv_out: gpu
                .alloc(ctx_window * num_layers * 2 * num_kv_heads * head_dim * bf16)?,
            // 2026-09-25: `ctx_window` i64 cache slots (`long long*` in
            // `fill_slots_from_block_table`): the precompute's ctx rows, and the block
            // rows in `forward_block`.
            slot_mapping_dev: gpu.alloc(ctx_window * 8)?,
            precompute_in: gpu.alloc(
                super::PRECOMPUTE_BATCH_ROWS * target_layer_ids.len() * target_hidden_size * bf16,
            )?,
            // 2026-09-25: One `[kv_len, q_offset, q_rope_pos]` u32 triple (12 bytes) per
            // band, read by the indirect paged attention at kernel entry.
            option_b_indirect_args_dev: gpu.alloc(nb * 12)?,
            // 2026-09-25: Host staging for the drafted tokens (`nb * gamma` u32; page-locked
            // on CUDA), and the event `forward_block` records after copying into it.
            draft_tokens_host_pinned: std::sync::atomic::AtomicPtr::new(
                gpu.alloc_host_pinned(nb * gamma_val * 4)?,
            ),
            draft_tokens_event: gpu.create_event()?,
            // 2026-09-25: `nb * gamma` rows, not `rows_max`: the lm_head writes one row per
            // block row, and no path indexes the logits by a ctx offset.
            logits: gpu.alloc(nb * g * vocab_size * bf16)?,
            draft_tokens_dev: gpu.alloc(rows_max * 4)?,
            position_ids: gpu.alloc(rows_max * 4)?,
            // 2026-09-25: DSpark Markov scratch, allocated only when the drafter config
            // declares a Markov head (`markov_rank > 0`); `DevicePtr(0)` otherwise.
            markov_embed: if weights.config.markov_rank > 0 {
                gpu.alloc(weights.config.markov_rank * bf16)?
            } else {
                DevicePtr(0)
            },
            markov_bias: if weights.config.markov_rank > 0 {
                gpu.alloc(vocab_size * bf16)?
            } else {
                DevicePtr(0)
            },
            conf_out: if weights.confidence_proj.is_some() {
                gpu.alloc(g * bf16)?
            } else {
                DevicePtr(0)
            },
            // 2026-09-25: DFlash2 scratch, allocated only when the checkpoint ships the
            // selector. `dflash2_active` requires the selector, so the convs and the walk
            // never run without these buffers.
            conv_dyn: if weights.selector_pred.is_some() {
                let cfg = weights.config.dflash_config.as_ref();
                let ksz = cfg.map(|c| c.conv_kernel_size).unwrap_or(0).max(1);
                let gsz = cfg.map(|c| c.conv_group_size).unwrap_or(0).max(1);
                gpu.alloc(nb * g * 2 * ksz * (hidden_size / gsz) * bf16)?
            } else {
                DevicePtr(0)
            },
            conv_tmp: if weights.selector_pred.is_some() {
                gpu.alloc(nb * g * hidden_size * bf16)?
            } else {
                DevicePtr(0)
            },
            sel_vals: if weights.selector_pred.is_some() {
                gpu.alloc(nb * g * 16 * 4)?
            } else {
                DevicePtr(0)
            },
            sel_idx: if weights.selector_pred.is_some() {
                gpu.alloc(nb * g * 16 * 4)?
            } else {
                DevicePtr(0)
            },
            sel_hproj: if weights.selector_pred.is_some() {
                let rank = weights
                    .config
                    .dflash_config
                    .as_ref()
                    .map(|c| c.selector_rank)
                    .unwrap_or(256)
                    .max(1);
                gpu.alloc(nb * g * rank * bf16)?
            } else {
                DevicePtr(0)
            },
        };

        // 2026-09-25: The RoPE inverse-frequency table, from the drafter config's
        // `rope_scaling`: absent gives plain RoPE (`1 / theta^(2j / dim)`),
        // `rope_type = "yarn"` the YaRN-blended table, and any other value plain RoPE
        // with a warning. RoPE rotates the whole head (`rotary_dim = head_dim`).
        let rope_theta = weights.config.rope_theta;
        let rotary_dim = head_dim;
        let inv_freq_table = rope_table::rope_inv_freq_table(&weights, rope_theta, rotary_dim);

        let inv_freq_bytes: Vec<u8> = inv_freq_table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let yarn_inv_freq = gpu.alloc(inv_freq_bytes.len())?;
        gpu.copy_h2d(&inv_freq_bytes, yarn_inv_freq)?;

        // 2026-09-25: The fused K/V weight `[L * 2 * kv_dim, hidden]` BF16, laid out
        // `[K_0, V_0, K_1, V_1, ...]` and copied device to device from each layer's
        // `k_proj` and `v_proj`, so `precompute_ctx_kv` computes every layer's ctx K/V in
        // one GEMM. `kv_dim_bytes` is the size of one layer's K (or V) weight.
        let kv_dim_bytes = num_kv_heads * head_dim * hidden_size * bf16;
        let fused_total_bytes = num_layers * 2 * kv_dim_bytes;
        let fused_kv_weight = gpu.alloc(fused_total_bytes)?;
        for (l, layer) in weights.layers.iter().enumerate() {
            let layer_base = l * 2 * kv_dim_bytes;
            gpu.copy_d2d(
                layer.k_proj.weight,
                fused_kv_weight.offset(layer_base),
                kv_dim_bytes,
            )?;
            gpu.copy_d2d(
                layer.v_proj.weight,
                fused_kv_weight.offset(layer_base + kv_dim_bytes),
                kv_dim_bytes,
            )?;
        }
        tracing::info!(
            "DFlash fused_kv_weight: {} bytes ({} layers × 2 × kv_dim × h × bf16), \
             layout [K0,V0,K1,V1,…] to match vLLM precompute_and_store_context_kv",
            fused_total_bytes,
            num_layers,
        );

        let mut head = Self {
            // 2026-09-25: The drafter's `DFlashLevers`, resolved from the environment once,
            // here.
            levers: super::levers::DFlashLevers::from_env(),
            num_layers,
            hidden_size,
            intermediate_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            draft_vocab_size: weights.config.draft_vocab_size.unwrap_or(vocab_size),
            gamma: gamma_val,
            block_gamma: std::sync::atomic::AtomicUsize::new(gamma_val),
            max_batch: nb,
            mask_token_id,
            window_size,
            target_layer_ids,
            target_hidden_size,

            embed_tokens_shared,
            lm_head_shared,
            lm_head_nvfp4,
            lm_head_shared_fp8: None,
            hidden_norm: weights.hidden_norm,
            norm: weights.norm,
            fc: weights.fc,
            draft_id_to_target_id: None,
            layers: weights
                .layers
                .into_iter()
                .map(|l| DflashLayer {
                    input_layernorm: l.input_layernorm,
                    post_attention_layernorm: l.post_attention_layernorm,
                    q_proj: l.q_proj,
                    k_proj: l.k_proj,
                    v_proj: l.v_proj,
                    o_proj: l.o_proj,
                    q_norm: l.q_norm,
                    k_norm: l.k_norm,
                    gate_proj: l.gate_proj,
                    up_proj: l.up_proj,
                    down_proj: l.down_proj,
                    // 2026-09-25: Filled below when the FP8 drafter path is on.
                    q_proj_fp8: None,
                    k_proj_fp8: None,
                    v_proj_fp8: None,
                    o_proj_fp8: None,
                    gate_proj_fp8: None,
                    up_proj_fp8: None,
                    down_proj_fp8: None,
                    attention_conv_base: l.attention_conv_base,
                    attention_conv_proj: l.attention_conv_proj,
                    mlp_conv_base: l.mlp_conv_base,
                    mlp_conv_proj: l.mlp_conv_proj,
                })
                .collect(),
            fused_kv_weight: Some(fused_kv_weight),
            kv_cache: Mutex::new(kv_cache),
            scratch,
            kernels,
            max_seq_len,
            yarn_inv_freq,
            rope_theta,
            rotary_dim,
            rms_norm_eps: 1e-6,
            ctx_window,
            propose_graphs: parking_lot::Mutex::new(super::ProposeGraphs::default()),
            suppress_graphs: std::sync::atomic::AtomicBool::new(false),
            propose_warmup_count: std::sync::atomic::AtomicUsize::new(0),
            quant: DflashQuantization::Bf16,
            // 2026-09-25: `markov_rank` is 0 when the checkpoint has no `markov_w1`.
            markov_rank: if weights.markov_w1.is_some() {
                weights.config.markov_rank
            } else {
                0
            },
            markov_w1: weights.markov_w1,
            markov_w2: weights.markov_w2,
            confidence_proj: weights.confidence_proj,
            confidence_bias: weights.confidence_bias,
            confidence_with_markov: weights.config.confidence_head_with_markov,
            shifted_rows: weights
                .config
                .dflash_config
                .as_ref()
                .and_then(|c| c.projector_type.as_deref())
                == Some("dspark"),
            conv_kernel_size: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_kernel_size)
                .unwrap_or(0),
            conv_group_size: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.conv_group_size)
                .unwrap_or(0),
            selector_rank: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_rank)
                .unwrap_or(0),
            selector_top_k: weights
                .config
                .dflash_config
                .as_ref()
                .map(|c| c.selector_top_k)
                .unwrap_or(0),
            selector_pred: weights.selector_pred,
            selector_succ: weights.selector_succ,
            selector_hidden_proj: weights.selector_hidden_proj,
        };
        if head.selector_pred.is_some() {
            tracing::info!(
                "DFlash2 armed: conv k={} group={} selector rank={} top_k={} \
                 (kernels present: conv={} topk={} walk={})",
                head.conv_kernel_size,
                head.conv_group_size,
                head.selector_rank,
                head.selector_top_k,
                head.kernels.dflash2_conv2.0 != 0,
                head.kernels.dflash2_topk16.0 != 0,
                head.kernels.dflash2_selector_walk.0 != 0,
            );
        }
        if head.shifted_rows {
            tracing::info!(
                "DSpark drafter: SpecForge shifted row convention active \
                 (projector_type=dspark) — draft vector rotates right by 1"
            );
        }

        tracing::info!(
            "BlockDiffusionDraftHead loaded: {} layers, hidden={}, intermediate={}, \
             GQA {}/{}, head_dim={}, γ={}, vocab={}, mask_token_id={}, target_layers={:?}",
            head.num_layers,
            head.hidden_size,
            head.intermediate_size,
            head.num_q_heads,
            head.num_kv_heads,
            head.head_dim,
            head.gamma,
            head.vocab_size,
            head.mask_token_id,
            head.target_layer_ids,
        );

        // 2026-09-25: FP8 drafter weights, on unless METRALE_DFLASH_DRAFTER_FP8=0:
        // quantize each layer's seven dense GEMM weights (q/k/v/o/gate/up/down) to FP8
        // E4M3 with one f32 scale per row, give the lm_head an FP8 copy, and set
        // `quant = Fp8Weights`. Needs the row-scaled FP8 GEMM and its m16 variant;
        // without them the drafter stays on BF16 with a warning.
        let fp8_requested =
            std::env::var("METRALE_DFLASH_DRAFTER_FP8").ok().as_deref() != Some("0");
        let fp8_kernels_present = head.kernels.fp8_gemm_n128_row_scaled.0 != 0
            && head.kernels.fp8_gemm_n128_row_scaled_m16.0 != 0;
        if fp8_requested && !fp8_kernels_present {
            tracing::warn!(
                "METRALE_DFLASH_DRAFTER_FP8=1 but fp8_gemm_t_row_scaled(_m16) kernels are \
                 not in this target's w4a16 PTX module — staying on the BF16 drafter path. \
                 Port the Phase G kernels from kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu."
            );
        }
        if fp8_requested && fp8_kernels_present {
            fp8_weights::quantize_drafter_fp8(&mut head, gpu, q_dim, kv_dim, lm_head_native_fp8)?;
        }

        Ok(head)
    }

    /// 2026-09-25: Errors when `target_hidden_size` differs from the value the head was
    /// built with, which sets the `fc` input width
    /// (`target_layer_ids.len() * target_hidden_size`). Nothing calls it.
    pub fn validate_against_target(&self, target_hidden_size: usize) -> Result<()> {
        if self.target_hidden_size != target_hidden_size {
            anyhow::bail!(
                "DFlash drafter target_hidden_size mismatch: drafter expects {}, target is {}",
                self.target_hidden_size,
                target_hidden_size
            );
        }
        Ok(())
    }
}
