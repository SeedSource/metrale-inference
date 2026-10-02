// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `TransformerModel::new`: resolves the model-level kernels and
//! levers, sizes the SSM, snapshot, speculative and capture buffers, and builds
//! the model around the loaded layers.
//!
//! Owner: model-engine.
//! Invariants:
//! - The constructed model's `mtp_prefill_capacity` is the row count allocated
//!   for `mtp_prefill_hidden`, and 0 when that buffer is NULL.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

mod comm_setup;
mod kernels;
mod spec_buffers;
mod ssm_setup;

use kernels::{AuxKernels, ModelKernels};

impl TransformerModel {
    pub fn new(
        config: ModelConfig,
        embed_tokens: DenseWeight,
        final_norm: DenseWeight,
        lm_head_weight: DenseWeight,
        lm_head_nvfp4: Option<QuantizedWeight>,
        // 2026-09-25: FP8 LM head, `Some` only under `--lm-head-dtype fp8`, and
        // then `lm_head_nvfp4` is `None`.
        lm_head_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
        // 2026-09-25: NVFP4 head used only by the MTP draft proposer. The
        // factory builds it when the main head is not NVFP4 and speculative
        // decoding has MTP weights; otherwise it is `None` and the proposer uses
        // `lm_head_nvfp4` (see `draft_lm_head_nvfp4`).
        mtp_lm_head_nvfp4: Option<QuantizedWeight>,
        layers: Vec<Box<dyn TransformerLayer>>,
        buffers: BufferArena,
        kv_cache: PagedKvCache,
        mtp_weights: Vec<MtpWeights>,
        gpu: Box<dyn GpuBackend>,
        max_seq_len: usize,
        max_batch_size: usize,
        mtp_quant: metrale_model_layers::layers::MtpQuantization,
        use_speculative: bool,
        prefix_cache: Box<dyn metrale_telemetry::prefix_cache::PrefixCache>,
        mtp_vocab_size: u32,
        comm: Option<std::sync::Arc<dyn metrale_comm::CommBackend>>,
        self_speculative: bool,
        num_drafts: usize,
        vision_encoder: Option<metrale_model_layers::layers::VisionTower>,
        ssm_cache_slots: usize,
        ssm_checkpoint_interval: usize,
    ) -> Result<Self> {
        // 2026-09-25: `rms_norm_kernel` is used only for `final_norm`, in
        // `final_norm_apply`.
        let rms_norm_kernel = if metrale_model_layers::ships_vanilla_norm_weights(&config) {
            gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?
        } else {
            gpu.kernel("norm", "rms_norm")?
        };
        let ModelKernels {
            dense_gemv_kernel,
            dense_gemv_fp32out_kernel,
            w4a16_gemv_kernel,
            w4a16_gemv_logits_kernel,
            w4a16_gemm_t_kernel,
            w4a16_gemm_t_bf16_kernel,
            w4a16_gemm_kernel,
            w4a16_gemv_batch2_kernel,
            w4a16_batchm,
            w4a16_gemv_batch16_kernel,
            dense_gemv_fp8w_kernel,
            dense_gemv_fp8w_batch2_kernel,
            dense_gemm_kernel,
            dense_gemv_batchm_kernel,
            lm_head_m16_tc_kernel,
            lm_head_m16_tc_n64_kernel,
            argmax_kernel,
            argmax_batch_kernel,
            argmax_logits_kernel,
            batched_embed_kernel,
            fill_slots_kernel,
            argmax_feed_kernel,
            feed_resolve_kernel,
        } = kernels::resolve_model_kernels(&config, gpu.as_ref())?;
        let profile = config.profile;
        let profile_first = std::env::var("METRALE_PROFILE_FIRST").is_ok();

        // 2026-09-25: A fresh `ModelLevers::from_env()`, not the process-cached
        // `ModelLevers::get()`, so a second model built in the same process
        // resolves its own levers. `max_decode_seqs` pins the split-K attention
        // split count to the configured max batch, so a sequence's attention
        // reduction does not depend on how many others share its batch.
        let mut levers = ops::ModelLevers::from_env();
        levers.max_decode_seqs = (max_batch_size as u32).max(1);

        tracing::info!(
            "TransformerModel: {} layers, vocab={}, hidden={}{}{}",
            layers.len(),
            config.vocab_size,
            config.hidden_size,
            if profile { " [PROFILE MODE]" } else { "" },
            if profile_first {
                " [PROFILE_FIRST]"
            } else {
                ""
            },
        );

        let (dflash_kgamma, draft_lm_head_nvfp4, has_mtp, mtp_quant_fwd, num_intermediates) =
            ssm_setup::size_rollback_pools(
                &config,
                mtp_lm_head_nvfp4,
                lm_head_nvfp4,
                self_speculative,
                use_speculative,
                &mtp_weights,
                mtp_quant,
                num_drafts,
            );
        let ssm_pool = std::sync::Arc::new(SsmStatePool::new(
            &config,
            max_batch_size,
            has_mtp,
            num_intermediates,
            num_drafts,
            // 2026-09-25: h pools sized at 2 bytes per element
            // (`--ssm-h-dtype f16-pool`).
            metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
            // 2026-09-25: `--ssm-rollback-mode`; `Snapshot` unless the server
            // published another mode.
            metrale_model_layers::ssm_reserve::ssm_rollback_mode(),
            gpu.as_ref(),
        )?);

        let (ssm_snapshots, ssm_tier_store) = ssm_setup::build_ssm_snapshots(
            &config,
            &ssm_pool,
            use_speculative,
            ssm_cache_slots,
            prefix_cache.as_ref(),
            max_batch_size,
            ssm_checkpoint_interval,
            &kv_cache,
            gpu.as_ref(),
        )?;

        // 2026-09-25: Fixed per-sequence stride of the attention block-table
        // metadata, so captured CUDA graphs stay valid.
        let max_blocks_per_seq = (max_seq_len / kv_cache.block_size() + 1) as u32;

        // 2026-09-25: Permanent KV block for padding rows and for block-table
        // default fill, zeroed so a read of it yields zeros rather than stale
        // or uninitialized bytes.
        let mut kv_cache = kv_cache;
        let dummy_kv_block = kv_cache.alloc_block()?;
        kv_cache.zero_block(dummy_kv_block, gpu.as_ref(), gpu.default_stream())?;
        gpu.synchronize(gpu.default_stream())?;

        // 2026-09-25: Transposed lm_head twin with its row stride padded to a
        // multiple of 128, so the tile GEMM's 16-byte `cp.async` B loads stay
        // aligned whatever the vocab size. Built unless
        // `METRALE_NO_LMHEAD_TGEMM=1`.
        let lm_head_nvfp4_t =
            spec_buffers::build_lm_head_nvfp4_t(&lm_head_nvfp4, &config, gpu.as_ref())?;
        // 2026-09-25: The drafter may use the twin only when its head is the
        // main head (`mtp_lm_head_nvfp4` absent): the twin is a transpose of
        // `lm_head_nvfp4`, so a dedicated draft head would score drafts against
        // the wrong weight. The shared case copies the handle, not the memory.
        let draft_lm_head_nvfp4_t = if mtp_lm_head_nvfp4.is_none() {
            lm_head_nvfp4_t
        } else {
            None
        };
        let proposer: Option<Arc<dyn DraftProposer>> = super::impl_a1_init::build_mtp_proposer(
            use_speculative,
            mtp_weights,
            embed_tokens,
            final_norm,
            draft_lm_head_nvfp4,
            draft_lm_head_nvfp4_t,
            &config,
            gpu.as_ref(),
            mtp_quant,
            mtp_vocab_size,
            max_seq_len,
            kv_cache.num_blocks(),
            &levers,
        )?;

        ssm_setup::log_self_speculative(&config, self_speculative);

        let (
            mtp_hidden_save,
            verify_hidden_stash,
            verify_catchup_stash,
            verify_wy_tables,
            gdn_woa_na_tab,
            mtp_catchup_ring,
        ) = spec_buffers::alloc_verify_buffers(&config, &proposer, &levers, gpu.as_ref())?;

        let (capture_rows, mtp_prefill_hidden) = spec_buffers::alloc_prefill_capture(
            &config,
            &proposer,
            max_seq_len,
            dflash_kgamma,
            has_mtp,
            mtp_quant_fwd,
            &levers,
            mtp_quant,
            buffers.max_batch_tokens(),
            gpu.as_ref(),
        )?;

        let (dflash_capture_layers, dflash_hidden_save_rows, dflash_hidden_save) =
            spec_buffers::alloc_dflash_capture(
                &config,
                dflash_kgamma,
                max_batch_size,
                gpu.as_ref(),
            )?;

        // 2026-09-25: One u32 for the EP command broadcast.
        let ep_cmd_buf = gpu.alloc(4)?;

        // 2026-09-25: Fixed-address buffer (CUDA-graph safe) for the batched
        // decode MoE per-row adapter map, 4 bytes per batch row, uploaded each
        // decode step. Allocated unconditionally because LoRA weights arrive
        // after construction (`set_lora_weights`).
        let moe_row_adapter_buf = gpu.alloc(max_batch_size.max(1) * 4)?;

        // 2026-09-25: Secondary stream and its event, used by the asynchronous
        // checkpoint copies (`trait_impl/async_chkpt.rs`).
        let secondary_stream = gpu.create_stream()?;
        let secondary_event = gpu.create_event()?;
        // 2026-09-25: Orders SSM snapshot saves (default stream) before a warm
        // Marconi restore (prefill stream); see the `snapshot_event` field doc.
        let snapshot_event = gpu.create_event()?;

        // 2026-09-25: Levers that shape the collective schedule are read per rank
        // from the environment, so ranks that disagree can hang or reduce the
        // wrong extent. `rank_agree::assert_ranks_agree` checks them before the
        // first token.
        if let Some(ref comm) = comm {
            comm_setup::assert_rank_levers_agree(gpu.as_ref(), comm)?;
        }

        // 2026-09-25: At world size 2, register the reduce targets
        // (`moe_output`, `norm_output`, and `logits` for the vocab-parallel
        // lm_head) with the comm backend and give it the `bf16_add_inplace`
        // kernel for its send/recv path. Each step is best-effort: a failure is
        // logged and construction continues.
        if let Some(ref comm) = comm
            && comm.world_size() == 2
        {
            comm_setup::register_reduce_buffers(comm, &buffers, gpu.as_ref());
        }

        // 2026-09-25: Device token feed buffers: one u32 per decode-metadata row
        // for cells, ids and sources, two for masks. Allocated only when both
        // feed kernels are loaded.
        let (feed_rows, feed_cells, feed_ids, feed_sources, feed_masks) =
            kernels::alloc_feed_buffers(
                &buffers,
                argmax_feed_kernel,
                feed_resolve_kernel,
                gpu.as_ref(),
            )?;
        let pinned_bytes = buffers.sizes().scratch.max(64 * 1024);
        let pinned_ptr = gpu.alloc_host_pinned(pinned_bytes)?;
        tracing::info!("Pinned metadata staging: {} KB", pinned_bytes / 1024);
        let max_batch_tokens = buffers.max_batch_tokens();
        let pinned_staging = std::cell::UnsafeCell::new(PinnedMetaStaging {
            ptr: pinned_ptr,
            bytes: pinned_bytes,
            positions: Vec::with_capacity(max_batch_tokens),
            positions_h: Vec::with_capacity(max_batch_tokens),
            positions_w: Vec::with_capacity(max_batch_tokens),
            slots: Vec::with_capacity(max_batch_tokens),
        });

        let AuxKernels {
            ssm_norm_k,
            ssm_norm_f16_k,
            ssm_h_f32_to_f16_k,
            ssm_h_f16_to_f32_k,
            logit_softcap_kernel,
            logit_softcap_fp32_kernel,
            use_fp32_logits,
            logits_fp32_buf,
            embed_scale_kernel,
            ssm_norm_ptrs,
        } = kernels::resolve_aux_kernels(&config, &ssm_pool, gpu.as_ref())?;

        // 2026-09-25: Sized for `min(max_batch_tokens, max_seq_len)` tokens;
        // `prefill_twophase` falls back to `prefill_chunk` for a longer prompt.
        let (gdn_qkv, gdn_gate_beta, gdn_out, gdn_z, gdn_buf_len) =
            super::impl_a1_init::build_gdn_prefill_buffers(
                &config,
                max_batch_tokens,
                max_seq_len,
                gpu.as_ref(),
            )?;

        // 2026-09-25: FP8 KV calibration observes only the FP8 cache write
        // (`write_kv_cache_fp8.rs`), so `fp8_kv_calibration_tokens` suppresses
        // CUDA graphs only when the cache is FP8.
        let has_fp8_calibration = config.fp8_kv_calibration_tokens > 0
            && kv_cache.dtype() == metrale_cache::kv_cache::KvCacheDtype::Fp8;
        // 2026-09-25: Resolve the token-overlay kernels before `gpu` moves into
        // `Self`.
        let overlay_kernels =
            metrale_model_layers::layers::ops::token_overlay::OverlayKernels::new(gpu.as_ref());
        Ok(Self {
            // 2026-09-25: Installed by the factory after construction
            // (`adopt_weight_store`).
            weight_store: None,
            config,
            dispatch: metrale_model_layers::layers::ops::GemmDispatch::from_env(),
            derived: metrale_model_layers::layers::ops::DerivedWeights::new(),
            levers,
            stats: ops::ModelStats::new(),
            #[cfg(feature = "cuda")]
            innerq: kernels::start_innerq(gpu.as_ref()),
            embed_tokens,
            ngram_embed: None,
            final_norm,
            lm_head_weight,
            lm_head_nvfp4,
            lm_head_nvfp4_t,
            lm_head_fp8,
            lm_head_q6k: None,
            // 2026-09-25: Computed before `layers` moves into the struct, since
            // the veto is a fold over the layers.
            decode_graph_veto: layers.iter().any(|l| l.decode_graph_unsupported()),
            layers,
            buffers,
            lora: None,
            lora_rotatable: false,
            kv_cache: Mutex::new(kv_cache),
            gpu,
            rms_norm_kernel,
            dense_gemv_kernel,
            dense_gemv_fp32out_kernel,
            w4a16_gemv_kernel,
            w4a16_gemv_logits_kernel,
            w4a16_gemm_t_kernel,
            w4a16_gemm_t_bf16_kernel,
            w4a16_gemm_kernel,
            w4a16_gemv_batch2_kernel,
            w4a16_batchm,
            w4a16_gemv_batch16_kernel,
            dense_gemv_fp8w_kernel,
            dense_gemv_fp8w_batch2_kernel,
            dense_gemm_kernel,
            dense_gemv_batchm_kernel,
            lm_head_m16_tc_kernel,
            lm_head_m16_tc_n64_kernel,
            argmax_kernel,
            argmax_batch_kernel,
            argmax_logits_kernel,
            batched_embed_kernel,
            fill_slots_kernel,
            argmax_feed_kernel,
            feed_resolve_kernel,
            feed_cells,
            feed_ids,
            feed_sources,
            feed_masks,
            feed_rows,
            inflight_fed_graph_keys: Mutex::new(std::collections::VecDeque::new()),
            decode_graph: Mutex::new(std::collections::HashMap::new()),
            batch_decode_graphs: Mutex::new((HashMap::new(), 0)),
            suppress_graphs: std::sync::atomic::AtomicBool::new(
                has_fp8_calibration
                    || std::env::var("METRALE_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true")
                    // 2026-09-25: Eager decode, so `METRALE_DEBUG_SYNC_KERNELS` can
                    // synchronize after each launch and report an async fault at
                    // the kernel that caused it.
                    || std::env::var("METRALE_DEBUG_NO_GRAPH").as_deref() == Ok("1"),
            ),
            ssm_pool,
            ssm_snapshots,
            ssm_tier_store,
            max_blocks_per_seq,
            dummy_kv_block,
            profile,
            profile_first_pending: std::sync::atomic::AtomicBool::new(profile_first),
            proposer,
            mtp_hidden_save,
            verify_hidden_stash,
            verify_catchup_stash,
            mtp_catchup_ring,
            mtp_catchup_meta: parking_lot::Mutex::new((0, 0)),
            mtp_prefill_hidden,
            // 2026-09-25: The rows actually allocated (both caps applied), which
            // the capture guard in `drafter_prefill.rs` checks against.
            mtp_prefill_capacity: if mtp_prefill_hidden.is_null() {
                0
            } else {
                capture_rows
            },
            mtp_prefill_capture_len: std::sync::atomic::AtomicUsize::new(0),
            mtp_stage_start: std::sync::atomic::AtomicUsize::new(0),
            mtp_prefill_capture_gen: std::sync::atomic::AtomicU64::new(0),
            mtp_store_gen_seq: std::sync::atomic::AtomicU64::new(0),
            mtp_carry: parking_lot::Mutex::new(None),
            mtp_store_range: parking_lot::Mutex::new(
                metrale_model_layers::mtp_carry::StoreRange::EMPTY,
            ),
            dflash_hidden_save,
            verify_ptok_meta: std::sync::OnceLock::new(),
            dflash_hidden_save_rows,
            dflash_kgamma,
            dflash_capture_layers,
            verify2_graph: Mutex::new(std::collections::HashMap::new()),
            verify3_graph: Mutex::new(std::collections::HashMap::new()),
            verify4_graph: Mutex::new(std::collections::HashMap::new()),
            verify_batched_graphs: Mutex::new((std::collections::HashMap::new(), 0)),
            verify_wy_tables,
            gdn_woa_na_tab,
            gdn_woa_folded_slots: parking_lot::Mutex::new(Vec::new()),
            gdn_woa_eligible: std::sync::atomic::AtomicBool::new(false),
            gdn_woa_bound: parking_lot::Mutex::new((DevicePtr::NULL, DevicePtr::NULL, 0)),
            gdn_carry: Default::default(),
            // 2026-09-25: Nothing staged yet, so the first batched verify step
            // uploads the tables.
            verify_wy_cache: Mutex::new(None),
            verify_kgamma_graph: Mutex::new(std::collections::HashMap::new()),
            fused_graph: Mutex::new(std::collections::HashMap::new()),
            prefix_cache,
            secondary_stream,
            secondary_event,
            snapshot_event,
            comm,
            ep_cmd_buf,
            ep_protocol_v2: matches!(std::env::var("METRALE_EP_PROTOCOL").as_deref(), Ok("v2")),
            self_speculative,
            last_mtp_hidden_idx: std::sync::atomic::AtomicUsize::new(0),
            vision_encoder,
            vision_embed_patches: Mutex::new(0),
            vision_image_grids: Mutex::new(Vec::new()),
            vision_row_base: Mutex::new(0),
            vision_grid_base: Mutex::new(0),
            vision_owned_images: Mutex::new(0),
            pinned_staging,
            ssm_checkpoint_interval,
            ssm_state_norm_kernel: ssm_norm_k,
            ssm_state_norm_f16_kernel: ssm_norm_f16_k,
            ssm_h_f32_to_f16_kernel: ssm_h_f32_to_f16_k,
            ssm_h_f16_to_f32_kernel: ssm_h_f16_to_f32_k,
            ssm_h_f16_scratch: std::sync::OnceLock::new(),
            ssm_norm_ptrs_buf: ssm_norm_ptrs,
            moe_row_adapter_buf,
            gdn_buf_qkv: gdn_qkv,
            gdn_buf_gate_beta: gdn_gate_beta,
            gdn_buf_out: gdn_out,
            gdn_buf_z: gdn_z,
            gdn_buf_max_len: gdn_buf_len,
            logit_softcap_kernel,
            logit_softcap_fp32_kernel,
            use_fp32_logits,
            logits_fp32_buf,
            embed_scale_kernel,
            overlays: None,
            overlay_kernels,
            overlay_route_slot: std::sync::atomic::AtomicI32::new(-1),
            decode_moe_route: std::sync::atomic::AtomicI32::new(1), // 2026-09-25: `MoeLoraRoute::Fold`
        })
    }
}
