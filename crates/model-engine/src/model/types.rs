// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `TransformerModel`, the state one loaded model owns, and its teardown (`release_pools`).
//!
//! Owner: model-engine.
//! Invariants:
//! - `release_pools` attempts every release even after an earlier one fails, and returns the first error.

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};

use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::DraftProposer;
use metrale_model_layers::weight_map::{DenseWeight, Fp8DenseWeight, MtpWeights, QuantizedWeight};

mod staging;
mod teardown;

#[allow(dead_code)]
/// 2026-09-25: Rows in `mtp_catchup_ring`. Position `p` is stored in ring row
/// `p % MTP_CATCHUP_RING_ROWS`.
pub(super) const MTP_CATCHUP_RING_ROWS: usize = 512;

/// 2026-09-25: One loaded model: its layers, weights, kernel handles, KV and
/// SSM pools and CUDA-graph caches. `trait_impl` implements `Model` for it.
pub struct TransformerModel {
    pub(super) config: ModelConfig,
    /// 2026-09-25: Which GEMM implementation each projection takes, resolved
    /// from the environment when this model was built
    /// (`GemmDispatch::from_env`). Every `ForwardContext` this model creates
    /// borrows it.
    pub(super) dispatch: metrale_model_layers::layers::ops::GemmDispatch,
    /// 2026-09-25: Per-model memo of derived weight encodings, released by
    /// `release_pools`.
    pub(super) derived: metrale_model_layers::layers::ops::DerivedWeights,
    /// 2026-09-25: The weight ledger this model was built from, installed by
    /// `adopt_weight_store` after construction and held for teardown:
    /// `release_pools` releases it after every other pool. `None` before
    /// adoption and after release.
    pub(super) weight_store: Option<metrale_model_weights::weights::WeightStore>,
    pub(super) levers: metrale_model_layers::layers::ops::ModelLevers,
    /// 2026-09-25: Diagnostic counters and one-shot dump latches for this model.
    pub(super) stats: metrale_model_layers::layers::ops::ModelStats,
    pub(super) embed_tokens: DenseWeight,
    /// 2026-09-25: Fused n-gram input embedding, installed by
    /// `set_ngram_embedding`. A `Mutex` because the forward path has `&self`
    /// while `NgramEmbedding::embed` takes `&mut self`.
    pub(super) ngram_embed:
        Option<std::sync::Mutex<metrale_model_layers::layers::ngram_embed::NgramEmbedding>>,
    pub(super) final_norm: DenseWeight,
    pub(super) lm_head_weight: DenseWeight,
    pub(super) lm_head_nvfp4: Option<QuantizedWeight>,
    /// 2026-09-25: Transposed `[K/2, ldb]` copy of `lm_head_nvfp4` for the tile
    /// GEMM, and its row stride.
    ///
    /// The stride is the vocab rounded up to a multiple of 128
    /// (`transpose_concat_for_gemm_padded(.., 16, 128)`) so that every row
    /// starts 16-byte aligned for the GEMM's B loads, including for an odd
    /// vocab. It is an additional buffer: `lm_head_nvfp4` keeps its row-major
    /// layout. `None` when there is no NVFP4 head or `METRALE_NO_LMHEAD_TGEMM=1`.
    pub(super) lm_head_nvfp4_t: Option<(QuantizedWeight, u32)>,
    /// 2026-09-25: Runtime FP8 E4M3 LM head with per-row scales, read by the
    /// `dense_gemv_fp8w` kernels. `Some` only on the `--lm-head-dtype fp8`
    /// path of `setup_lm_heads`, which returns `lm_head_nvfp4 = None`.
    pub(super) lm_head_fp8: Option<Fp8DenseWeight>,
    /// 2026-09-25: The output head as raw Q6_K blocks, run by the K-quant GEMV.
    /// Installed by `set_lm_head_q6k` when the store's `lm_head.weight` is
    /// Q6_K; `None` otherwise.
    pub(super) lm_head_q6k: Option<super::lm_head_q6k::LmHeadQ6k>,
    pub(super) layers: Vec<Box<dyn TransformerLayer>>,
    /// 2026-09-25: `true` when any layer's `decode_graph_unsupported()` is true,
    /// so decode stays eager. Computed once at construction from `layers`.
    pub(super) decode_graph_veto: bool,
    pub(super) buffers: BufferArena,
    /// 2026-09-25: The LoRA adapter pool and pointer tables, installed by
    /// `set_lora_weights`, which also installs the active adapter's per-layer
    /// pairs into the layers. `None` = no adapter.
    pub(super) lora: Option<metrale_model_layers::lora::LoraWeights>,
    /// 2026-09-25: Runtime adapter rotation is armed: `METRALE_LORA_ROTATE` is
    /// on, or `METRALE_LORA_PEER` is non-empty. Set by `set_lora_weights`; the
    /// rotate and swap entry points refuse while it is false.
    pub(super) lora_rotatable: bool,
    pub(super) kv_cache: Mutex<PagedKvCache>,
    pub(super) gpu: Box<dyn GpuBackend>,
    /// 2026-09-25: TQ+ InnerQ calibration driver, when `TURBO_INNERQ` is a
    /// positive integer. The scheduler reaches it through `Model::poll_innerq`.
    #[cfg(feature = "cuda")]
    pub(super) innerq: Option<metrale_model_layers::layers::qwen3_attention::InnerQDriver>,
    pub(super) rms_norm_kernel: KernelHandle,
    pub(super) dense_gemv_kernel: KernelHandle,
    /// 2026-09-25: Always `KernelHandle(0)`: `use_fp32_logits` is always false.
    pub(super) dense_gemv_fp32out_kernel: KernelHandle,
    pub(super) w4a16_gemv_kernel: KernelHandle,
    pub(super) w4a16_gemv_logits_kernel: KernelHandle,
    /// 2026-09-25: Tile GEMM over `lm_head_nvfp4_t`. 0 when `tgemm_probe_ok`
    /// rejects the model type.
    pub(super) w4a16_gemm_t_kernel: KernelHandle,
    /// 2026-09-25: BF16-MMA tile GEMM over `lm_head_nvfp4_t`
    /// (`w4a16_gemm_t_m128_bf16_v2`). 0 unless `METRALE_LMHEAD_LOSSLESS` is set
    /// (any value) and the target has the kernel.
    pub(super) w4a16_gemm_t_bf16_kernel: KernelHandle,
    pub(super) w4a16_gemm_kernel: KernelHandle,
    pub(super) w4a16_gemv_batch2_kernel: KernelHandle,
    /// 2026-09-25: The narrow batched-GEMV tiers; a tier the target lacks is a
    /// 0 handle.
    pub(super) w4a16_batchm: metrale_model_layers::layers::w4a16_gemv_tiers::W4a16BatchmTiers,
    pub(super) w4a16_gemv_batch16_kernel: KernelHandle,
    /// 2026-09-25: FP8 E4M3 GEMV for `lm_head_fp8`. Loaded unconditionally;
    /// launched only on the FP8-head path.
    pub(super) dense_gemv_fp8w_kernel: KernelHandle,
    /// 2026-09-25: FP8-weight GEMV over two rows in one weight pass. 0 when the
    /// target lacks the kernel.
    pub(super) dense_gemv_fp8w_batch2_kernel: KernelHandle,
    pub(super) dense_gemm_kernel: KernelHandle,
    /// 2026-09-25: Batched BF16 GEMV (M rows, one weight pass). 0 = absent.
    pub(super) dense_gemv_batchm_kernel: KernelHandle,
    /// 2026-09-25: Tensor-core BF16 GEMM with a 16-row M tile
    /// (`dense_gemm_m16_bf16`), the BF16 LM-head arm that
    /// `trait_impl/lm_head_batched.rs::lm_head_m16_tc_route` selects when the
    /// `lm_head_m16_tc` lever is on. 0 when the target lacks the kernel.
    pub(super) lm_head_m16_tc_kernel: KernelHandle,
    /// 2026-09-25: The `N_TILE=64` variant (`dense_gemm_m16_bf16_n64`), selected
    /// by `METRALE_LM_HEAD_M16_TC_NTILE=64`. 0 when the target lacks it.
    pub(super) lm_head_m16_tc_n64_kernel: KernelHandle,
    pub(super) argmax_kernel: KernelHandle,
    /// 2026-09-25: Batched argmax (one block per row). 0 when the kernel set lacks it.
    pub(super) argmax_batch_kernel: KernelHandle,
    pub(super) argmax_logits_kernel: KernelHandle,
    pub(super) batched_embed_kernel: KernelHandle,
    pub(super) fill_slots_kernel: KernelHandle,
    /// 2026-09-25: The device token feed (`trait_impl/feed.rs`): the masked feed
    /// argmax and the source resolver of the `argmax_feed` module. Both are 0
    /// when the target lacks the module, and then the feed is not used.
    pub(super) argmax_feed_kernel: KernelHandle,
    pub(super) feed_resolve_kernel: KernelHandle,
    /// 2026-09-25: Feed buffers, `[feed_rows]` u32 each: the cells a step's
    /// argmax writes and the next step reads, the resolved input ids, and the
    /// per-row source words. The masks are `[feed_rows * 2]` u32. All four are
    /// NULL when either feed kernel is 0.
    pub(super) feed_cells: DevicePtr,
    pub(super) feed_ids: DevicePtr,
    pub(super) feed_sources: DevicePtr,
    pub(super) feed_masks: DevicePtr,
    pub(super) feed_rows: usize,
    /// 2026-09-25: Batched-graph keys of fed steps still in flight, in launch
    /// order (`pin_fed_graph_key` / `fed_step_settled_dispatch`). The batched
    /// decode graph LRU never evicts these keys.
    pub(super) inflight_fed_graph_keys: Mutex<std::collections::VecDeque<Vec<u32>>>,
    /// 2026-09-25: CUDA graphs for n=1 decode, keyed by `seq.slot_idx`. A
    /// captured graph has the slot's SSM `h_state`/`conv_state` pointers baked
    /// in as kernel arguments, so it is only replayed for the same slot.
    pub(super) decode_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// 2026-09-25: CUDA graphs for batched decode, keyed by the per-row SSM pool
    /// slot vector (`trait_impl/decode_graph_key.rs`). Value =
    /// `(graph, last_use_tick)`; the `u64` beside the map is the tick counter.
    /// At `batch_decode_graph_cap` entries the least-recently-used graph is
    /// destroyed.
    pub(super) batch_decode_graphs: Mutex<(HashMap<Vec<u32>, (GraphHandle, u64)>, u64)>,
    /// 2026-09-25: Pre-allocated SSM state pool, so state addresses stay fixed
    /// across graph replays. `Arc` so each `SequenceState` can hold the
    /// `SlotGuard` from `SsmStatePool::claim_guarded`, which returns the slot
    /// to the free list when the sequence is dropped.
    pub(super) ssm_pool: Arc<SsmStatePool>,
    /// 2026-09-25: SSM state snapshots for Marconi prefix caching.
    pub(super) ssm_snapshots: SsmSnapshotPool,
    /// 2026-09-25: SSM snapshot spill tier. `Some` only when `METRALE_SSM_TIER`
    /// is set and the model has SSM layers (`build_ssm_tier_store`).
    pub(super) ssm_tier_store: Option<Arc<dyn super::ssm_tier::SnapshotBlobStore>>,
    /// 2026-09-25: `max_seq_len / block_size + 1`, the fixed block-table stride
    /// in attention metadata, so a captured graph's stride stays valid.
    pub(super) max_blocks_per_seq: u32,
    /// 2026-09-25: A KV block allocated and zeroed at construction, never freed,
    /// used for padding rows in batched decode and verify.
    pub(super) dummy_kv_block: u32,
    pub(super) profile: bool,
    /// 2026-09-25: One-shot profile of the next prefill, armed by
    /// `METRALE_PROFILE_FIRST` (any value). The prefill layer loop
    /// (`prefill_b/forward_layers.rs`) swaps it to false when it consumes it.
    pub(super) profile_first_pending: std::sync::atomic::AtomicBool,
    /// 2026-09-25: When true, CUDA graph capture and replay are skipped. Set at
    /// construction during FP8 KV calibration or under a graph-disabling
    /// diagnostic env, and around the per-sequence decode loop in `decode_a2`.
    pub(super) suppress_graphs: std::sync::atomic::AtomicBool,
    /// 2026-09-25: Draft proposer: built in `new` from the MTP weights, or
    /// installed later by `set_dflash_proposer`.
    pub(super) proposer: Option<Arc<dyn DraftProposer>>,
    /// 2026-09-25: One FP32 hidden row (`hidden_size * 4` bytes): the target
    /// hidden that `save_hidden_for_mtp` copies out before the MTP head runs.
    pub(super) mtp_hidden_save: DevicePtr,
    /// 2026-09-25: Batched-verify hidden stash, `[VERIFY_WY_TABLE_SEQS,
    /// hidden_size]` BF16, one row per sequence of a batched verify.
    /// `stash_verify_hidden_rows` copies each sequence's accepted-row hidden
    /// here and `save_hidden_for_mtp_from_stash` reads it back for the drafter.
    /// NULL without a proposer at construction.
    pub(super) verify_hidden_stash: DevicePtr,
    /// 2026-09-25: `ModelLevers::mtp_kv_exact`: verify-forward rows of accepted
    /// drafts, `[VERIFY_WY_TABLE_SEQS * MTP_CATCHUP_MAX, hidden_size]` BF16.
    /// NULL unless the lever is on and a proposer exists at construction.
    pub(super) verify_catchup_stash: DevicePtr,
    /// 2026-09-25: `METRALE_MTP_CATCHUP`: ring of per-position final hiddens
    /// captured during serial decode, BF16 rows, row = position %
    /// `MTP_CATCHUP_RING_ROWS`. Feeds the drafter catch-up on the next propose.
    /// NULL when `mtp_catchup_enabled()` is false at construction.
    pub(super) mtp_catchup_ring: DevicePtr,
    /// 2026-09-25: `(first_position, count)` of the contiguous position range in
    /// the ring. A write that does not extend the range restarts it at that
    /// position; `count` never exceeds `MTP_CATCHUP_RING_ROWS`.
    pub(super) mtp_catchup_meta: parking_lot::Mutex<(usize, usize)>,
    /// 2026-09-25: Whole-prompt final-layer hidden capture, `[rows,
    /// hidden_size]` BF16 with `rows = mtp_prefill_capacity`. Written per
    /// prefill chunk and read by the drafter prefill and the drafter carry
    /// (`drafter_prefill.rs`). NULL unless an MTP proposer exists, its head supports the
    /// drafter prefill, and the drafter-prefill lever is on.
    pub(super) mtp_prefill_hidden: DevicePtr,
    /// 2026-09-25: Row capacity of `mtp_prefill_hidden`: `max_seq_len` capped by
    /// the proposer's `prefill_hidden_rows` and, with DFlash, by the DFlash
    /// context cap. 0 when the buffer is NULL. The capture bounds check
    /// compares against this value.
    pub(super) mtp_prefill_capacity: usize,
    /// 2026-09-25: Rows of `mtp_prefill_hidden` captured contiguously from
    /// position 0 for the sequence that owns the capture. Reset to 0 by
    /// `alloc_sequence`. A chunk that does not extend the contiguous range
    /// leaves it short, and the propose-site coverage check then skips the
    /// drafter prefill.
    pub(super) mtp_prefill_capture_len: std::sync::atomic::AtomicUsize,
    /// 2026-10-01: Prompt row at staging row 0 under the chunked capture (`mtp_stage`).
    pub(super) mtp_stage_start: std::sync::atomic::AtomicUsize,
    /// 2026-09-25: Generation of the single-slot capture above. A chunk-0
    /// prefill bumps it and stamps the new value on its sequence
    /// (`SequenceState::mtp_capture_gen`). Appends and the drafter-prefill
    /// consume require the sequence's stamp to equal the current value and to
    /// be non-zero, so a sequence whose capture another sequence's prefill has
    /// restarted skips the drafter prefill.
    pub(super) mtp_prefill_capture_gen: std::sync::atomic::AtomicU64,
    /// 2026-09-25: Ticket counter for `mtp_store_range` ownership
    /// (`SequenceState::mtp_store_gen`), drawn once per `alloc_sequence`.
    ///
    /// It is separate from `mtp_prefill_capture_gen` because a draw on every
    /// admission would advance that counter, and `owns_capture` requires the
    /// capture generation to be unchanged between a sequence's capture and its
    /// propose.
    pub(super) mtp_store_gen_seq: std::sync::atomic::AtomicU64,
    /// 2026-09-25: The previous turn's drafter KV, held so the next turn of the
    /// same session can adopt it. Single slot: `mtp_carry::carry_armed_with`
    /// turns the carry off in multi-sequence MTP mode. `None` when the carry is
    /// off or nothing has been carried.
    pub(super) mtp_carry:
        parking_lot::Mutex<Option<metrale_model_layers::mtp_carry::CarriedDrafter>>,
    /// 2026-09-25: Absolute position interval of `mtp_prefill_hidden` rows and
    /// the sequence ticket that wrote them. Updated only while the drafter
    /// carry is armed.
    ///
    /// The ticket (`owner`) is what ties the interval to one sequence:
    /// `mtp_carry::stamped_merge` replaces rather than extends the interval
    /// when a different owner writes. `alloc_sequence` also resets it.
    pub(super) mtp_store_range: parking_lot::Mutex<metrale_model_layers::mtp_carry::StoreRange>,
    /// 2026-09-25: DFlash per-layer hidden capture, allocated when the config
    /// names DFlash capture layers. Row-major: `dflash_hidden_save_rows` rows,
    /// each `dflash_capture_layers.len() * hidden_size` BF16, with capture
    /// layer `s` at element offset `s * hidden_size` inside a row. `None` for
    /// non-DFlash runs.
    pub(super) dflash_hidden_save: Option<DevicePtr>,
    /// 2026-09-25: Lazily allocated 16 KiB metadata staging for the
    /// sliding-attention per-token verify loop (`verify_attention_per_token`).
    /// The loop cannot stage at `scratch + 32768`: the verify bodies'
    /// multi-seq metadata overlay lives there, and the interleaved
    /// FullAttention layers still read it between the per-token calls. Layout:
    /// position at +0, slot at +8, seq_len at +16, seq_slot at +128, block
    /// table from +256; the loop refuses a block table that would not fit.
    pub(super) verify_ptok_meta: std::sync::OnceLock<DevicePtr>,
    /// 2026-09-25: Layer indices to capture for DFlash, from
    /// `config.dflash_capture_layers`. Empty when DFlash is off.
    pub(super) dflash_capture_layers: Vec<usize>,
    /// 2026-09-25: Row capacity of `dflash_hidden_save`,
    /// `dflash_kgamma.max(2) * max_batch_size.max(1)`, or 0 without DFlash.
    /// `try_dflash_capture_all_at` clamps its writes to it.
    pub(super) dflash_hidden_save_rows: usize,
    /// 2026-09-25: Rows per per-sequence capture band in `dflash_hidden_save`:
    /// the drafter γ + 1, or 17 when γ is unknown, or 0 without DFlash.
    /// Sequence `i` of a batched verify writes rows from `i * dflash_kgamma`;
    /// single-sequence paths use band 0. The scheduler passes this stride to
    /// `commit_ctx` as `scratch_row`.
    pub(super) dflash_kgamma: usize,
    /// 2026-09-25: CUDA graphs for K=2 verify, keyed by `seq.slot_idx` for the
    /// same reason as `decode_graph`.
    pub(super) verify2_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// 2026-09-25: CUDA graphs for K=3 verify, keyed by `seq.slot_idx`.
    pub(super) verify3_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// 2026-09-25: CUDA graphs for K=4 verify, keyed by `seq.slot_idx`.
    pub(super) verify4_graph: Mutex<std::collections::HashMap<usize, GraphHandle>>,
    /// 2026-09-25: CUDA graphs for the batched verify (`verify_e.rs`), keyed by
    /// `verify_key::verify_graph_key`: each sequence's `(ssm slot, row count)`
    /// in dispatch order, then a sentinel word for the WY-tables-present and
    /// write-on-accept bits. A graph bakes every sequence's state pointers and
    /// the row counts, so it replays only for an equal key. Value =
    /// `(graph, last_use_tick)`; at `VERIFY_BATCHED_GRAPH_CAP` entries the
    /// least-recently-used graph is destroyed.
    pub(super) verify_batched_graphs:
        Mutex<(std::collections::HashMap<Vec<u32>, (GraphHandle, u64)>, u64)>,
    /// 2026-09-25: Batched-verify WY pointer tables at a fixed device address:
    /// one `VERIFY_WY_LAYER_STRIDE_BYTES` slice per GDN layer, holding
    /// `VERIFY_WY_TABLES_PER_LAYER` tables of `VERIFY_WY_TABLE_SEQS` u64
    /// entries. Zeroed at allocation and written by `upload_verify_wy_tables`.
    /// NULL without a proposer at construction or without SSM layers.
    pub(super) verify_wy_tables: DevicePtr,
    /// 2026-09-25: Write-on-accept: device `u32[VERIFY_WY_TABLE_SEQS]` of
    /// accepted row counts for the post-verdict fold, and the SSM slots the
    /// most recent fold committed (`async_chkpt.rs` consumes an entry to skip
    /// that slot's h restore). The table is NULL when `verify_wy_tables` is.
    pub(super) gdn_woa_na_tab: DevicePtr,
    pub(super) gdn_woa_folded_slots: Mutex<Vec<usize>>,
    /// 2026-09-25: Cleared at the start of every batched verify, set at the end
    /// of one that ran under a write-on-accept request with its tables staged,
    /// and consumed by the fold (`gdn_fold_accepted_dispatch`), which declines
    /// unless it was set.
    pub(super) gdn_woa_eligible: std::sync::atomic::AtomicBool,
    /// 2026-09-25: Write-on-accept `(flags, stash, seqs)` for every GDN layer,
    /// allocated and bound by `gdn_woa_bind` on the first write-on-accept
    /// request, before any capture, and never moved. `(NULL, NULL, 0)` until
    /// then.
    pub(super) gdn_woa_bound: Mutex<(DevicePtr, DevicePtr, usize)>,
    /// 2026-09-26: Carried-state GDN verify buffers and pending rows (`gdn_carry.rs`).
    pub(super) gdn_carry: super::gdn_carry::GdnCarry,
    /// 2026-09-25: Key of the bytes currently staged in `verify_wy_tables`
    /// (`verify_wy_cache_key`), or `None` when nothing has been staged. A step
    /// whose key matches skips the host build and the H2D. Presence of
    /// `METRALE_NO_VERIFY_WY_CACHE` disables the cache.
    pub(super) verify_wy_cache: Mutex<Option<Vec<u64>>>,
    /// 2026-09-25: CUDA graphs for the DFlash K=γ verify, keyed by
    /// `(seq.slot_idx, tokens.len())`.
    pub(super) verify_kgamma_graph: Mutex<std::collections::HashMap<(usize, usize), GraphHandle>>,
    /// 2026-09-25: CUDA graphs for the DFlash decode+verify fused pass
    /// (`verify_fused.rs`), keyed by `(seq.slot_idx, M)` with
    /// `M = tokens.len()`.
    pub(super) fused_graph: Mutex<std::collections::HashMap<(usize, usize), GraphHandle>>,
    pub(super) prefix_cache: Box<dyn metrale_telemetry::prefix_cache::PrefixCache>,
    /// 2026-09-25: Second CUDA stream and its event, used by the asynchronous
    /// SSM checkpoint copies (`async_chkpt.rs`).
    pub(super) secondary_stream: u64,
    pub(super) secondary_event: u64,
    /// 2026-09-25: Orders SSM-snapshot saves before a later warm Marconi
    /// restore that runs on another stream. A save records it after enqueueing
    /// its D2D copies (`record_snapshot_save_dispatch`), and a restore waits on
    /// it before reading the snapshot region (`wait_snapshot_saves_dispatch`),
    /// so a restore never reads a slot whose save copy is still in flight.
    pub(super) snapshot_event: u64,
    /// 2026-09-25: Communication backend for multi-rank (EP/TP) collectives.
    /// `None` on a single GPU.
    pub(super) comm: Option<std::sync::Arc<dyn metrale_comm::CommBackend>>,
    /// 2026-09-25: 4-byte device buffer for the EP token broadcast.
    pub(super) ep_cmd_buf: DevicePtr,
    /// 2026-09-25: EP wire protocol: true when `METRALE_EP_PROTOCOL=v2`, which
    /// precedes every command broadcast with a `seq_id` broadcast so the
    /// worker can route slot-bound work to the right `SequenceState`. With a
    /// comm backend, `new` checks that the ranks agree on it
    /// (`rank_agree::assert_ranks_agree`).
    pub(super) ep_protocol_v2: bool,
    pub(super) self_speculative: bool,
    /// 2026-09-25: Last token index `save_hidden_for_mtp` copied from, broadcast
    /// to the other EP rank before an MTP propose.
    pub(super) last_mtp_hidden_idx: std::sync::atomic::AtomicUsize,
    pub(super) vision_encoder: Option<metrale_model_layers::layers::VisionTower>,
    /// 2026-09-25: Number of patches encoded by the last `prepare_vision_embed`
    /// call. 0 means no vision embeddings are pending.
    pub(super) vision_embed_patches: Mutex<usize>,
    /// 2026-09-25: Per-item `(t_len, grid_h_post_merge, grid_w_post_merge)` from
    /// the most recent vision prepare, used by MRoPE prefill to assign
    /// `(t, h, w)` position ids to each vision pad token. Empty when no vision
    /// input is pending. `t_len` is the number of temporal groups the item
    /// spans (1 for a still image), because a video occupies one contiguous
    /// pad run whose T advances.
    pub(super) vision_image_grids: Mutex<Vec<(usize, usize, usize)>>,
    /// 2026-09-25: Where the next prefill's vision slice starts inside a batched
    /// vision prepare: the first row of the packed output, the first
    /// `vision_image_grids` index, and how many items it owns. Set by
    /// `Model::set_vision_slice_base`; all 0 reads from row 0 and grid 0.
    pub(super) vision_row_base: Mutex<usize>,
    pub(super) vision_grid_base: Mutex<usize>,
    pub(super) vision_owned_images: Mutex<usize>,
    /// 2026-09-25: Page-locked host staging for batched metadata H2D transfers,
    /// allocated in `new` and freed on drop (`drop_pinned_staging`).
    ///
    /// An `UnsafeCell`, not a `Mutex`, because the model is used only from the
    /// scheduler thread after construction (see the `Sync` impl below).
    pub(super) pinned_staging: std::cell::UnsafeCell<PinnedMetaStaging>,
    /// 2026-09-25: Save an SSM snapshot every N blocks during chunked prefill;
    /// 0 disables the intermediate checkpoints (`prefill_b/save_checkpoint.rs`).
    pub(super) ssm_checkpoint_interval: usize,
    /// 2026-09-25: `ssm_state_clamp_norm_fused`, 0 when the target lacks it.
    pub(super) ssm_state_norm_kernel: KernelHandle,
    /// 2026-09-25: The FP16 h-state variant (`ssm_state_clamp_norm_fused_f16`),
    /// for sequences whose `SsmLayerState::h_is_f16` is set.
    pub(super) ssm_state_norm_f16_kernel: KernelHandle,
    /// 2026-09-25: Device pointer table `[num_ssm_layers]` for
    /// `ssm_state_clamp_norm_fused`.
    pub(super) ssm_norm_ptrs_buf: DevicePtr,
    /// 2026-09-25: FP32 → FP16 h-state converter (`METRALE_SSM_H_FP16`).
    pub(super) ssm_h_f32_to_f16_kernel: KernelHandle,
    /// 2026-09-25: FP16 → FP32 h-state converter, used by the batched prefill
    /// path under the FP16-sized pool (`--ssm-h-dtype f16-pool`).
    pub(super) ssm_h_f16_to_f32_kernel: KernelHandle,
    /// 2026-09-25: Staging for the FP32 → FP16 conversion
    /// (`ssm_h_to_f16_dispatch`), one layer wide (`h_bytes / 2`), allocated on
    /// first use. The narrowing cannot run in place: output element `i` is
    /// written into the bytes that hold input element `i / 2`.
    pub(super) ssm_h_f16_scratch: std::sync::OnceLock<DevicePtr>,

    /// 2026-09-25: Fixed-address device buffer for the batched-decode MoE
    /// per-row adapter map, `[max_batch_size]` i32 (`< 0` = base row, `>= 0` =
    /// fold that adapter). Allocated in `new` and refreshed per decode step by
    /// `upload_moe_row_adapter`, which does not touch it when `lora` is `None`.
    pub(super) moe_row_adapter_buf: DevicePtr,

    // 2026-09-25: Two-phase SSM prefill buffers, `gdn_buf_max_len` rows each,
    // shared by every layer and sequence. NULL when the model has no GDN
    // layers (`build_gdn_prefill_buffers`).
    /// 2026-09-25: Packed QKV, `[rows, conv_dim]` BF16, per token
    /// `[Q(key_dim) | K(key_dim) | V(value_dim)]`.
    pub(super) gdn_buf_qkv: DevicePtr,
    /// 2026-09-25: Gate and beta, `[rows, 2 * num_v_heads]` FP32.
    pub(super) gdn_buf_gate_beta: DevicePtr,
    /// 2026-09-25: GDN output, `[rows, value_dim]` BF16.
    pub(super) gdn_buf_out: DevicePtr,
    /// 2026-09-25: Z gate for the gated RMS norm, `[rows, value_dim]` BF16.
    pub(super) gdn_buf_z: DevicePtr,
    /// 2026-09-25: Row count of the buffers above:
    /// `min(max_batch_tokens, max_seq_len)`.
    pub(super) gdn_buf_max_len: usize,

    /// 2026-09-25: Logit softcapping, `logits = cap * tanh(logits / cap)`.
    /// `KernelHandle(0)` when the model has no softcapping.
    pub(super) logit_softcap_kernel: KernelHandle,
    /// 2026-09-25: Always `KernelHandle(0)`: `use_fp32_logits` is always false.
    pub(super) logit_softcap_fp32_kernel: KernelHandle,
    /// 2026-09-25: Always false: `new` sets it so, and the BF16 logits path is
    /// always taken.
    pub(super) use_fp32_logits: bool,
    /// 2026-09-25: FP32 logits scratch, allocated only when `use_fp32_logits`;
    /// NULL otherwise.
    pub(super) logits_fp32_buf: DevicePtr,
    /// 2026-09-25: Embedding scale by `config.embed_scale`.
    /// `KernelHandle(0)` when the model does not scale embeddings.
    pub(super) embed_scale_kernel: KernelHandle,
    /// 2026-09-25: Per-adapter-slot token overlay tables (embed and lm_head row
    /// overrides), built by `set_lora_weights` only when some adapter
    /// overrides a row. `None` makes every overlay hook return early.
    pub(super) overlays: Option<metrale_model_layers::lora::TokenOverlaySet>,
    /// 2026-09-25: Token overlay kernels, resolved in `new`.
    pub(super) overlay_kernels: metrale_model_layers::layers::ops::token_overlay::OverlayKernels,
    /// 2026-09-25: The overlay route of the current forward: the request's
    /// `adapter_slot`, stamped at each `Model` entry
    /// (`stamp_overlay_route` / `stamp_overlay_route_batch`). `i32::MIN` marks
    /// a batch that mixes adapters; the hooks then skip.
    pub(super) overlay_route_slot: std::sync::atomic::AtomicI32,
    /// 2026-09-25: MoE LoRA route of the current decode batch, read through
    /// `decode_moe_route()`: 0 = Skip, 1 = Fold, 2 = Refuse (any other value
    /// reads as Fold). Initialised to 1.
    pub(super) decode_moe_route: std::sync::atomic::AtomicI32,
}

/// 2026-09-25: Pinned host staging buffer plus reusable metadata `Vec`s.
pub(crate) struct PinnedMetaStaging {
    /// 2026-09-25: Page-locked host buffer from `alloc_host_pinned`.
    pub(super) ptr: *mut u8,
    /// 2026-09-25: Size of `ptr`'s region in bytes.
    pub(super) bytes: usize,
    pub(super) positions: Vec<u32>,
    pub(super) positions_h: Vec<u32>,
    pub(super) positions_w: Vec<u32>,
    pub(super) slots: Vec<i64>,
}

// 2026-09-25: SAFETY: TransformerModel is constructed on one thread and then
// moved to the scheduler thread as a `Box<dyn Model>`; every later call
// (prefill, decode, verify) is made from that thread. The `Model` trait
// requires Send + Sync for the move. `UnsafeCell<PinnedMetaStaging>` is not
// Sync, so single-thread access is an assumption of the scheduler design, not
// something the types enforce. The pinned pointer is valid from any thread.
unsafe impl Send for TransformerModel {}
// 2026-09-25: SAFETY: Model methods are only called from the scheduler thread; there is no concurrent `&self` access.
unsafe impl Sync for TransformerModel {}
