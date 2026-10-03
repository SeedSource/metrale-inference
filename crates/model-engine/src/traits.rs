// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `Model` trait, the per-sequence state it works on (`SequenceState`), and the
//! input and result types of its forward passes.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;

use metrale_model_layers::layer::LayerState;
use metrale_model_layers::speculative::ProposerState;

/// 2026-09-25: Result of a mixed forward pass (decode rows plus one prefill chunk).
pub struct MixedForwardResult {
    /// 2026-09-25: Logits for the decode sequences, `[N, vocab_size]`; NULL when there are none.
    pub decode_logits: DevicePtr,
    /// 2026-09-25: Logits for the prefill sequence's last token, `[1, vocab_size]`; NULL unless
    /// the chunk was its last.
    pub prefill_logits: DevicePtr,
}

/// 2026-09-25: One prefilling stream's chunk. `prefill_batch_chunk` and `mixed_forward_batch`
/// take a `&mut [PrefillSlice<'_>]` and run every stream's chunk in one forward pass.
pub struct PrefillSlice<'a> {
    /// 2026-09-25: The stream's full prompt.
    pub prompt_tokens: &'a [u32],
    pub seq: &'a mut SequenceState,
    /// 2026-09-25: Offset into `prompt_tokens` where this chunk starts.
    pub chunk_start: usize,
    pub chunk_len: usize,
    /// 2026-09-25: Whether this is the stream's final chunk, which produces its last-token logits.
    pub is_last_chunk: bool,
}

/// 2026-09-25: Result of a batched mixed forward pass: M decode rows plus N prefill chunks.
pub struct MixedBatchResult {
    /// 2026-09-25: Logits for the decode rows, `[M, vocab]`; NULL when there are none.
    pub decode_logits: DevicePtr,
    /// 2026-09-25: One entry per prefill stream, in input order: `[1, vocab]` logits when that
    /// stream's chunk was its last, NULL otherwise.
    pub prefill_logits: Vec<DevicePtr>,
}

/// 2026-09-25: Per-sequence device copies of the paged-attention metadata for chunked prefill,
/// kept across chunks so each chunk uploads only the block-table entries it added.
pub struct ChunkedPrefillPageMetadata {
    /// 2026-09-25: The block table as 32-bit entries.
    pub block_table: DevicePtr,
    /// 2026-09-25: The running sequence length, one `u32`.
    pub seq_len: DevicePtr,
    /// 2026-09-25: Block-table entries allocated, sized for the whole prompt.
    pub block_capacity: usize,
    /// 2026-09-25: Block-table entries already uploaded.
    pub uploaded_blocks: usize,
}

/// 2026-09-25: One sequence's state across prefill and decode.
pub struct SequenceState {
    /// 2026-09-25: Prompt and generated token ids processed so far.
    pub tokens: Vec<u32>,
    /// 2026-09-25: Physical KV blocks of the paged cache, in logical order.
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub layer_states: Vec<Box<dyn LayerState>>,
    /// 2026-09-25: The draft proposer's per-sequence state; `None` without a proposer.
    pub proposer_state: Option<Box<dyn ProposerState>>,
    /// 2026-09-25: Pool slot of this sequence, taken from `ssm_slot` at `alloc_sequence`.
    /// `compact_sequence` moves it, and `detach_slot_for_reuse` sets it to `usize::MAX`.
    pub slot_idx: usize,
    /// 2026-09-25: Guard that returns the SSM pool slot to the free list on drop, so the slot is
    /// released on every exit path, including abort and unwind. The explicit free and compact
    /// paths `take()` or `migrate()` it so the slot is released once. `None` for models
    /// without an SSM pool.
    pub(crate) ssm_slot: Option<crate::model::ssm_pool::SlotGuard>,
    /// 2026-09-25: Prompt position covered by the restored SSM snapshot and cached KV, set by the
    /// chunk-0 prefix lookup and returned again when that lookup is replayed.
    pub marconi_skip_to: usize,
    /// 2026-09-25: Snapshot slot when the whole prompt matched a snapshot. The last prompt token
    /// is then re-run for logits, which advances the SSM state a second time, so
    /// `finalize_last` restores the state from this snapshot again and takes the first token's
    /// hidden row from it. `None` otherwise.
    pub marconi_exact_snap: Option<usize>,
    /// 2026-09-25: Session identity, set by the scheduler before prefill. Saved SSM snapshots are
    /// tagged with it and a snapshot is restored only for a matching tag
    /// (`SsmSnapshotPool::session_matches`); `0` skips the check.
    pub session_hash: u64,
    /// 2026-09-25: This sequence's claim on the single shared whole-prompt hidden capture
    /// (`mtp_prefill_hidden`). `try_mtp_prefill_capture` sets it to a new capture generation when
    /// it writes this sequence's rows from row 0. `ensure_drafter_context` prefills the drafter
    /// only while it equals the model's current generation, since another sequence's chunk 0 may
    /// have restarted the capture in between. `0` means it never owned a capture.
    pub mtp_capture_gen: u64,
    /// 2026-09-25: This sequence's ticket for the shared hidden-row interval (`mtp_store_range`),
    /// drawn in `alloc_sequence` from the model's counter, and separate from `mtp_capture_gen`,
    /// which is set only by a capture that starts at row 0. `0` means it was not drawn in
    /// `alloc_sequence`, and matches no owner.
    pub mtp_store_gen: u64,
    /// 2026-09-25: Prefix-cache namespace of the sequence's LoRA adapter; `0` is the base model.
    /// Each id has its own radix root, so adapters never reuse each other's blocks. The
    /// scheduler sets it from `Model::adapter_id_for`.
    pub adapter_id: u64,
    /// 2026-09-25: Paged metadata for chunked prefill, allocated by the first chunk that needs it.
    pub chunked_prefill_meta: Option<ChunkedPrefillPageMetadata>,
    /// 2026-09-25: Prompt tokens matched by the chunk-0 prefix-cache lookup. The block-ref
    /// accounting reads it (the release in `free_sequence`, the prefill-end cache insert). It is
    /// not what clients are told: a match can be found and then recomputed
    /// (`reused_prefix_tokens`).
    pub cached_prefix_tokens: usize,
    /// 2026-09-25: Prompt tokens whose KV this request read from the prefix cache instead of
    /// recomputing (`prefix_reuse::reused_prefix_tokens`). The scheduler reports it as
    /// `usage.prompt_tokens_details.cached_tokens`.
    pub reused_prefix_tokens: usize,
    /// 2026-09-25: Number of `block_table` entries that came from the prefix-cache lookup
    /// (`matched_blocks.len()`); 0 without a hit.
    pub cached_prefix_blocks: usize,
    /// 2026-09-25: The matched prefix tokens (`tokens[..cached_prefix_tokens]`), stored at lookup.
    /// When a prefill matched a prefix and then failed before `tokens` was filled,
    /// `free_sequence` releases the lookup's radix refs over these instead. Empty when nothing
    /// matched.
    pub prefix_ref_tokens: Vec<u32>,
    /// 2026-09-25: Whether the chunk-0 prefix lookup has run for this sequence.
    ///
    /// The lookup is not idempotent: it takes radix refs, `inc_ref`s each matched block and
    /// pushes it onto `block_table`, all before the suffix is allocated. When that allocation
    /// fails, the scheduler's preempt-and-retry re-enters chunk 0; with this flag set the lookup
    /// replays its first result instead of pushing the blocks and taking the refs again.
    pub prefix_lookup_applied: bool,
    /// 2026-09-25: The `skip` result of the chunk-0 lookup, returned again on a replay.
    pub prefix_lookup_skip: bool,
    /// 2026-09-25: Token count of an SSM checkpoint near the prompt's end that the pool holds for
    /// this sequence: saved during this prefill, or the one a warm prefill restored from. Cleared
    /// at chunk 0; `finalize_last` reads it to decide whether the prompt-end leaf gets its own
    /// snapshot (`prefill_b::exact_leaf`).
    pub tail_checkpoint_tokens: Option<usize>,
    /// 2026-10-03: Set by the chunk-0 prefix lookup when the absolute prefill grid applies to
    /// this prompt (`METRALE_PREFIX_GRID_RESTORE=1`, `prefill_b/grid_restore.rs`). The sequence
    /// then holds radix refs on exactly `prefix_ref_tokens` (the lookup's match, widened by the
    /// prefill-end grid insert), and `free_sequence` releases those; `cache_sequence` and decode
    /// checkpoints are skipped. `false` keeps the previous accounting.
    pub prefix_grid_refs: bool,
    /// 2026-09-25: Length of the prefix, from position 0, whose paged KV this sequence has written
    /// or validly reused, updated per chunk by `prefill_b_proc_range`. The prefill-end cache
    /// insert and checkpoint save cover at most `kv_valid_tokens / block_size` complete blocks, so
    /// a block whose KV was never written is not cached.
    pub kv_valid_tokens: usize,
    /// 2026-09-25: Block index (`len / block_size`) of the latest decode-time SSM checkpoint, so a
    /// boundary is not saved twice. Prefill sets it to the prompt's block count.
    pub last_decode_ckpt_block: usize,
    /// 2026-09-25: Prompt length, set by prefill. `cache_sequence` passes it to the prefix-cache
    /// insert as `matched_tokens`, so the insert takes refs only for the generated tokens.
    pub prompt_len: usize,
    /// 2026-09-25: `--high-speed-swap` disk block ids, one per logical block the sequence has had,
    /// in order. When the HBM window is at its cap, the oldest `block_table` entry is freed and
    /// its disk id stays (`ensure_blocks_through_decode`), so `block_table` holds the newest
    /// blocks and `hss_window_start()` is the logical index of `block_table[0]`. Empty without
    /// `--high-speed-swap`.
    pub disk_block_ids: Vec<u32>,
    /// 2026-09-25: For each attention layer, how many `disk_block_ids` entries it has offloaded.
    /// A block leaves the HBM window only when every layer has offloaded it
    /// (`check_safe_to_evict`). `TransformerModel::alloc_sequence` sizes it to the attention-layer
    /// count whether or not `--high-speed-swap` is on.
    pub disk_last_offloaded_per_layer: Vec<u32>,
    /// 2026-09-25: `Some(k)` makes prefill score every prompt position with its top-`k`
    /// alternatives (`/v1/completions` with `echo` and `logprobs`). Set by the scheduler before
    /// prefill. Such a request bypasses the prefix cache, so every position has a hidden row.
    pub collect_prompt_logprobs: Option<u8>,
    /// 2026-09-25: Filled across prefill chunks: one entry per prompt position `i` scoring
    /// `tokens[i + 1]`. The last prompt position, whose target is the first generated token, has
    /// no entry.
    pub prompt_logprobs: Vec<PromptTokenLogprob>,
    /// 2026-09-25: The LoRA pool slot this sequence's requests select (not `slot_idx`, which is
    /// the KV/SSM pool slot); `-1` selects the installed active adapter. Set by the scheduler at
    /// prefill; the prefill, decode and verify paths route LoRA by it (`moe_lora_route`,
    /// `upload_seq_slot_uniform`).
    pub adapter_slot: i32,
    /// 2026-09-25: The resolved LoRA pool slot this sequence holds a ref on, as returned by
    /// `Model::acquire_adapter_slot`; `-1` when none. `free_sequence` releases exactly this index
    /// and resets it to `-1`, so a rotation of the active adapter between prefill and finish
    /// cannot release a different slot.
    pub acquired_adapter_slot: i32,
    /// 2026-09-25: NLLB per-request source-language token id; `0` uses the deployment default.
    pub src_lang_id: u32,
    /// 2026-09-25: NLLB per-request target-language token id; `0` uses the deployment default.
    pub tgt_lang_id: u32,
    /// 2026-09-25: NLLB beam count; `1` turns beam search off.
    pub num_beams: u32,
    /// 2026-09-25: NLLB length penalty: a hypothesis scores `sum_logprob / len^length_penalty`.
    pub length_penalty: f32,
    /// 2026-09-25: NLLB beam search: `true` stops once `num_beams` hypotheses have finished;
    /// `false` also waits until no running beam can beat the worst finished one.
    pub early_stopping: bool,
}

impl SequenceState {
    /// 2026-09-25: A host-only sequence state: no GPU resources, no SSM slot, no layer states,
    /// every counter zeroed. NLLB's `alloc_sequence` and the test models build on it;
    /// `TransformerModel::alloc_sequence` writes its own literal. Other crates can construct a
    /// `SequenceState` only through this, because `ssm_slot` is crate-private.
    pub fn host_only(slot_idx: usize) -> Self {
        SequenceState {
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states: Vec::new(),
            proposer_state: None,
            slot_idx,
            ssm_slot: None,
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            mtp_capture_gen: 0,
            mtp_store_gen: 0,
            adapter_id: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            reused_prefix_tokens: 0,
            cached_prefix_blocks: 0,
            prefix_ref_tokens: Vec::new(),
            prefix_lookup_applied: false,
            tail_checkpoint_tokens: None,
            prefix_grid_refs: false,
            prefix_lookup_skip: false,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: Vec::new(),
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            adapter_slot: -1,
            acquired_adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
        }
    }

    /// 2026-09-25: The SSM pool slot this sequence holds, or `None` without one. The scheduler
    /// sorts decode batches by it.
    #[inline]
    pub fn ssm_slot_idx(&self) -> Option<usize> {
        self.ssm_slot.as_ref().and_then(|g| g.idx())
    }

    /// 2026-09-25: Logical block index of `block_table[0]`: `disk_block_ids.len()` minus
    /// `block_table.len()`, saturating. 0 without `--high-speed-swap`, where `disk_block_ids` is
    /// empty.
    #[inline]
    pub fn hss_window_start(&self) -> usize {
        self.disk_block_ids
            .len()
            .saturating_sub(self.block_table.len())
    }

    /// 2026-09-25: Physical HBM block of a logical block index, or `None` when that block has left
    /// the HBM window (or is past the end of `block_table`).
    #[inline]
    pub fn physical_block_for(&self, abs_block_idx: usize) -> Option<u32> {
        let ws = self.hss_window_start();
        if abs_block_idx < ws {
            return None;
        }
        self.block_table.get(abs_block_idx - ws).copied()
    }
}

mod logprobs;
mod model;
pub use logprobs::*;
pub use model::{
    BeamReq, EpCommandFailed, FeedSource, Model, ModelAdapters, ModelDeviceFeed, ModelDraft,
    ModelEp, ModelForward, ModelLifecycle, ModelLogits, ModelSsmState, ModelStreams, ModelVerify,
    ModelVision, RowMask, VerifyBatchedOpts, padded_batch_n,
};
