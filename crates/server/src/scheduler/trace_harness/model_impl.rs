// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `impl Model for RecordingModel`: the fake's answers at the scheduler-model boundary, most of them recorded as trace lines.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! Bodies with bookkeeping live in `model_forward.rs` and `model_feed.rs`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    BeamReq, FeedSource, MixedBatchResult, MixedForwardResult, Model, ModelAdapters,
    ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle, ModelLogits, ModelSsmState,
    ModelStreams, ModelVerify, ModelVision, PrefillSlice, RowMask, SequenceState,
    VerifyBatchedOpts,
};

use super::model::RecordingModel;
use super::model_forward::sid;

macro_rules! rec {
    ($m:expr, $($arg:tt)*) => { $m.rec(format!($($arg)*)) };
}

/// 2026-09-25: Methods that record one trace line and return the given value.
macro_rules! recorded {
    ($( fn $name:ident(&$s:ident $(, $arg:ident : $ty:ty)*) -> $ret:ty = $val:expr => $fmt:literal $(, $e:expr)*; )*) => {
        $( fn $name(&$s $(, $arg: $ty)*) -> $ret { rec!($s, $fmt $(, $e)*); $val } )*
    };
}

/// 2026-09-25: `&mut self` variants of the same.
macro_rules! recorded_mut {
    ($( fn $name:ident(&mut $s:ident $(, $arg:ident : $ty:ty)*) -> $ret:ty = $val:expr => $fmt:literal $(, $e:expr)*; )*) => {
        $( fn $name(&mut $s $(, $arg: $ty)*) -> $ret { rec!($s, $fmt $(, $e)*); $val } )*
    };
}

/// 2026-09-25: Methods whose body adds no trace line.
macro_rules! facts {
    ($( fn $name:ident(&$s:ident $(, $arg:ident : $ty:ty)*) -> $ret:ty = $val:expr; )*) => {
        $( fn $name(&$s $(, $arg: $ty)*) -> $ret { $val } )*
    };
}

/// 2026-09-25: Verify entry points that take a token vector and answer per position.
macro_rules! verify_vec {
    ($( fn $name:ident => $label:literal; )*) => {
        $( fn $name(&self, t: &[u32], seq: &mut SequenceState, _s: u64) -> Result<Vec<u32>> {
            Ok(self.verify_k($label, t, seq))
        } )*
    };
}

/// 2026-09-25: The fixed-width graphed verifies.
macro_rules! verify_fixed {
    ($( fn $name:ident => $label:literal, $k:literal; )*) => {
        $( fn $name(&self, t: &[u32; $k], seq: &mut SequenceState, _s: u64) -> Result<[u32; $k]> {
            let v = self.verify_k($label, t, seq);
            Ok(std::array::from_fn(|i| v[i]))
        } )*
    };
}

impl Model for RecordingModel {}

impl ModelLifecycle for RecordingModel {
    recorded_mut! {
        fn teardown(&mut self) -> Result<()> = Ok(()) => "teardown()";
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        self.alloc("alloc_sequence()".to_string())
    }
    fn alloc_sequence_for(&self, budget_tokens: usize) -> Result<SequenceState> {
        self.alloc(format!("alloc_sequence_for(budget={budget_tokens})"))
    }
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        self.fx_free_sequence(seq);
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        self.fx_compact(seq, new_slot);
        Ok(())
    }
    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        rec!(self, "detach_slot_for_reuse({})", sid(seq));
        seq.slot_idx = usize::MAX;
    }
    fn restore_sequence_state(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        _r: &mut dyn std::io::Read,
    ) -> Result<()> {
        self.fx_restore(seq, num_blocks);
        Ok(())
    }
    fn reclaim_prefix_blocks(&self, num_blocks: usize) -> usize {
        let got = self.reclaim(num_blocks);
        rec!(self, "reclaim_prefix_blocks({num_blocks}) -> {got}");
        got
    }
    facts! {
        fn num_total_blocks(&self) -> usize = self.cfg.total_blocks;
    }
    recorded! {
        fn poll_innerq(&self) -> () = () => "poll_innerq()";
        fn cache_sequence(&self, seq: &SequenceState) -> () = () => "cache_sequence({})", sid(seq);
        fn high_speed_swap_dims(&self) -> Option<metrale_storage::ModelDims> = None => "high_speed_swap_dims()";
        fn bind_gpu_to_thread(&self) -> Result<()> = Ok(()) => "bind_gpu_to_thread()";
        fn save_sequence_state(&self, seq: &SequenceState, _w: &mut dyn std::io::Write) -> Result<()> = Ok(()) => "save_sequence_state({}, blocks={})", sid(seq), seq.block_table.len();
        fn num_free_blocks(&self) -> usize = self.free_blocks() => "num_free_blocks() -> {}", self.free_blocks();
    }
}

impl ModelForward for RecordingModel {
    fn generate_beam_batch(&self, reqs: &[BeamReq]) -> Result<Vec<Vec<u32>>> {
        let hyps: Vec<Vec<u32>> = reqs
            .iter()
            .map(|r| self.beam_hyp(u64::from(r.prompt_tokens[0])))
            .collect();
        rec!(self, "generate_beam_batch(n={}) -> {hyps:?}", reqs.len());
        Ok(hyps)
    }
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        Ok(self.fx_prefill(tokens, seq, stream))
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        Ok(self.fx_prefill_chunk(tokens, seq, chunk_start, chunk_len, is_last_chunk, stream))
    }
    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        Ok(self.fx_decode("decode", token, seq, stream))
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.fx_decode_batch(tokens, seqs, stream)
    }
    fn mixed_forward(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<MixedForwardResult> {
        Ok(self.fx_mixed_forward(
            decode_tokens,
            decode_seqs,
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        ))
    }
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        Ok(self.fx_prefill_batch_chunk(streams, stream, None))
    }
    fn prefill_batch_chunk_rows(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        Ok(self.fx_prefill_batch_chunk(streams, stream, Some(row_base)))
    }
    fn mixed_forward_batch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<MixedBatchResult> {
        Ok(self.fx_mixed_forward_batch(decode_tokens, decode_seqs, prefill_streams, stream))
    }
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.fx_prefill_twophase(tokens, seq, chunk_size, stream)
    }
    facts! {
        fn supports_beam(&self) -> bool = self.cfg.supports_beam;
        fn is_mla(&self) -> bool = false;
        fn hc_mult(&self) -> usize = 1;
        fn kv_block_size(&self) -> Option<usize> = Some(self.cfg.block_size);
    }
    recorded! {
        fn normalize_ssm_states(&self, seq: &SequenceState, stream: u64) -> Result<()> = Ok(()) => "normalize_ssm_states({}, stream={stream})", sid(seq);
    }
}

impl ModelLogits for RecordingModel {
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let (row, off) = self.locate(logits_ptr);
        self.fill_slab(row, off, dst);
        rec!(
            self,
            "copy_logits_to_host(row={row}, off={off}, bytes={})",
            dst.len()
        );
        Ok(())
    }
    fn argmax_on_device(&self, logits_ptr: DevicePtr, stream: u64) -> Result<u32> {
        let (row, _) = self.locate(logits_ptr);
        let t = self.row(row);
        rec!(self, "argmax_on_device(row={row}, stream={stream}) -> {t}");
        Ok(t)
    }
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
        let (row, _) = self.locate(logits_ptr);
        let out: Vec<u32> = (row..row + n).map(|r| self.row(r)).collect();
        rec!(
            self,
            "argmax_batch(row={row}, n={n}, stream={stream}) -> {out:?}"
        );
        Ok(out)
    }
    facts! {
        fn vocab_size(&self) -> usize = self.cfg.vocab;
        fn logits_ptr_is_fp32(&self, _p: DevicePtr) -> bool = false;
        fn logits_buffer_ptr(&self) -> DevicePtr = self.row_ptr(0);
        fn hidden_after_norm(&self) -> DevicePtr = DevicePtr::NULL;
        fn decode_logits_fp32(&self) -> bool = false;
        fn decode_logits_ptr(&self) -> DevicePtr = self.row_ptr(0);
    }
}

impl ModelAdapters for RecordingModel {
    recorded_mut! {
        fn set_active_lora(&mut self, name: &str) -> Result<()> = Ok(()) => "set_active_lora({name})";
        fn swap_lora_from_disk(&mut self, dir: &std::path::Path, name: &str, slot: usize) -> Result<()> = Ok(()) => "swap_lora_from_disk({}, {name}, {slot})", dir.display();
        fn promote_lora_from_peer(&mut self, peer: &str, adapter_id: &str, name: &str, _peft: metrale_config::PeftAdapterConfig) -> Result<(usize, Option<String>)> = Ok((0, None)) => "promote_lora_from_peer({peer}, {adapter_id}, {name})";
        fn promote_lora_from_disk(&mut self, dir: &std::path::Path, name: &str) -> Result<(usize, Option<String>)> = Ok((0, None)) => "promote_lora_from_disk({}, {name})", dir.display();
    }
    recorded! {
        fn release_adapter_slot(&self, resolved: i32) -> () = () => "release_adapter_slot({resolved})";
        fn adapter_id_for(&self, slot: i32) -> u64 = 0 => "adapter_id_for({slot})";
        fn acquire_adapter_slot(&self, slot: i32) -> i32 = slot => "acquire_adapter_slot({slot})";
    }
}

impl ModelSsmState for RecordingModel {
    fn ssm_snapshot_occupancy(&self) -> Option<(u32, u32)> {
        if let Some(t) = self.snapshot_tick() {
            rec!(self, "-- tick {t}");
        }
        None
    }
    facts! {
        fn has_ssm_layers(&self) -> bool = self.cfg.has_ssm_layers;
        fn decode_rollback_unsupported(&self) -> bool = false;
        fn decode_rollback_ring_slots(&self) -> usize = self.cfg.ring_slots;
    }
    recorded! {
        fn decode_marconi_checkpoint(&self, seq: &mut SequenceState) -> () = () => "decode_marconi_checkpoint({})", sid(seq);
        fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()> = Ok(()) => "checkpoint_ssm_states({})", sid(seq);
        fn rollback_ssm_states(&self, seq: &mut SequenceState, na: usize) -> Result<()> = Ok(()) => "rollback_ssm_states({}, n={na})", sid(seq);
        fn save_decode_ssm_snapshot(&self, seq: &SequenceState, ring: usize) -> Result<()> = Ok(()) => "save_decode_ssm_snapshot({}, ring={ring})", sid(seq);
        fn restore_decode_ssm_snapshot(&self, seq: &SequenceState, ring: usize) -> Result<()> = Ok(()) => "restore_decode_ssm_snapshot({}, ring={ring})", sid(seq);
        fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> = Ok(()) => "start_checkpoint_async({})", sid(seq);
        fn start_rollback_and_checkpoint_async(&self, seq: &mut SequenceState, na: usize) -> Result<()> = Ok(()) => "start_rollback_and_checkpoint_async({}, na={na})", sid(seq);
        fn sync_secondary(&self) -> Result<()> = Ok(()) => "sync_secondary()";
        fn commit_accepted_prefix(&self, seq: &mut SequenceState, na: usize, k: usize) -> Result<()> = Ok(()) => "commit_accepted_prefix({}, na={na}, k={k})", sid(seq);
    }
}

impl ModelVerify for RecordingModel {
    verify_fixed! {
        fn decode_verify_graphed => "decode_verify_graphed", 2;
        fn decode_verify_graphed_k3 => "decode_verify_graphed_k3", 3;
        fn decode_verify_graphed_k4 => "decode_verify_graphed_k4", 4;
    }
    verify_vec! {
        fn decode_verify => "decode_verify";
        fn decode_verify_graphed_kgamma => "decode_verify_graphed_kgamma";
        fn decode_verify_dflash => "decode_verify_dflash";
        fn decode_and_verify_fused => "decode_and_verify_fused";
    }
    fn decode_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        stream: u64,
        opts: VerifyBatchedOpts,
    ) -> Result<Vec<u32>> {
        self.fx_verify_batched(tokens, ks, seqs, stream, opts.write_on_accept)
    }
    recorded! {
        fn can_batch_verify(&self, ks: &[usize]) -> bool = self.cfg.can_batch_verify => "can_batch_verify({ks:?}) -> {}", self.cfg.can_batch_verify;
        fn gdn_fold_accepted(&self, slots: &[usize], accepted: &[u32], k: usize) -> Result<bool> = Ok(true) => "gdn_fold_accepted(slots={slots:?}, accepted={accepted:?}, k={k}) -> true";
    }
}

impl ModelDraft for RecordingModel {
    fn generate_speculative(
        &self,
        _prompt: &[u32],
        _params: &metrale_sampling::SamplingParams,
        _num_drafts: usize,
    ) -> Result<metrale_model_engine::engine::GenerateResult> {
        anyhow::bail!("generate_speculative is not a scheduler path")
    }
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        Ok(self.fx_decode("decode_draft", token, seq, stream))
    }
    fn run_mtp_catchup_batched(
        &self,
        tokens: &[Vec<u32>],
        first_slot: &[usize],
        first_pos: &[usize],
        _seqs: &mut [&mut SequenceState],
    ) -> Result<usize> {
        let (s, p) = (first_slot, first_pos);
        rec!(
            self,
            "run_mtp_catchup_batched(tokens={tokens:?}, slots={s:?}, pos={p:?}) -> 0"
        );
        Ok(0)
    }
    fn run_mtp_propose_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash_idx: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let d = self.fx_propose_batched(
            tokens, positions, stash_idx, num_drafts, seqs, stream, out_conf,
        );
        Ok(Some(d))
    }
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Option<u32>> {
        let d = self.drafts_for(seq, position, 1);
        let id = sid(seq);
        rec!(
            self,
            "run_mtp_propose(token={token}, pos={position}, {id}, stream={stream}) -> {d:?}"
        );
        Ok(d.first().copied())
    }
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        let masked = grammar_bitmask.is_some();
        Ok(self.fx_propose_multi(token, position, num_drafts, seq, stream, masked))
    }
    fn read_deferred_draft_token(&self) -> Result<u32> {
        anyhow::bail!("read_deferred_draft_token is not a scheduler path")
    }
    facts! {
        fn has_proposer(&self) -> bool = self.cfg.has_proposer;
        fn dflash_gamma(&self) -> Option<usize> = self.cfg.dflash_gamma;
        fn has_self_speculative(&self) -> bool = self.cfg.has_self_speculative;
        fn mtp_propose_batch_max(&self) -> usize = 32;
        fn dflash_capture_band(&self) -> usize = 1;
    }
    recorded! {
        fn mtp_slot_draft_capacity(&self, slot: usize) -> usize = self.cfg.slot_draft_capacity => "mtp_slot_draft_capacity({slot})";
        fn stash_verify_hidden_rows(&self, rows: &[usize], stream: u64) -> Result<()> = Ok(()) => "stash_verify_hidden_rows({rows:?}, stream={stream})";
        fn stash_verify_catchup_rows(&self, slot_rows: &[(usize, usize)]) -> Result<()> = Ok(()) => "stash_verify_catchup_rows({slot_rows:?})";
        fn save_hidden_for_mtp_from_stash(&self, idx: usize, stream: u64) -> Result<()> = Ok(()) => "save_hidden_for_mtp_from_stash({idx}, stream={stream})";
        fn save_hidden_for_mtp(&self, idx: usize, stream: u64) -> Result<()> = Ok(()) => "save_hidden_for_mtp({idx}, stream={stream})";
        fn save_hidden_for_catchup(&self, idx: usize, pos: usize) -> Result<()> = Ok(()) => "save_hidden_for_catchup({idx}, pos={pos})";
        fn save_dflash_hidden_for_propose(&self, idx: usize, stream: u64) -> Result<()> = Ok(()) => "save_dflash_hidden_for_propose({idx}, stream={stream})";
        fn dflash_accept_append(&self, seq: &mut SequenceState) -> Result<()> = Ok(()) => "dflash_accept_append({})", sid(seq);
        fn dflash_eagle_accept_append(&self, seq: &mut SequenceState) -> Result<()> = Ok(()) => "dflash_eagle_accept_append({})", sid(seq);
        fn dflash_eagle_kgamma_append(&self, seq: &mut SequenceState, na: usize, base: usize) -> Result<()> = Ok(()) => "dflash_eagle_kgamma_append({}, na={na}, base={base})", sid(seq);
        fn dflash_serial_ctx_append(&self, seq: &mut SequenceState) -> Result<()> = Ok(()) => "dflash_serial_ctx_append({})", sid(seq);
        fn commit_ctx(&self, seq: &mut SequenceState, n: usize, base: usize, row: usize) -> Result<()> = Ok(()) => "commit_ctx({}, n={n}, base={base}, row={row})", sid(seq);
        fn trim_proposer_state(&self, seq: &mut SequenceState, na: usize, stream: u64) -> Result<()> = Ok(()) => "trim_proposer_state({}, na={na}, stream={stream})", sid(seq);
    }
}

impl ModelVision for RecordingModel {
    facts! {
        fn tokens_contain_vision_pad(&self, _t: &[u32]) -> bool = false;
    }
    recorded! {
        fn set_vision_slice_base(&self, row_base: usize, grid_base: usize, owned: usize) -> () = () => "set_vision_slice_base({row_base}, {grid_base}, {owned})";
        fn prepare_vision_embed(&self, images: &[metrale_model_layers::VisionItem]) -> Result<()> = Ok(()) => "prepare_vision_embed(n={})", images.len();
        fn prepare_vision_embed_batched(&self, per_request: &[Vec<metrale_model_layers::VisionItem>]) -> Result<Vec<(usize, usize, usize, usize)>> = Ok(Vec::new()) => "prepare_vision_embed_batched(n={})", per_request.len();
        fn ep_sync_vision_embeds(&self, tokens: &[u32]) -> Result<()> = Ok(()) => "ep_sync_vision_embeds(n={})", tokens.len();
    }
}

impl ModelEp for RecordingModel {
    facts! {
        fn is_ep(&self) -> bool = false;
        fn ep_protocol_v2(&self) -> bool = false;
    }
    recorded! {
        fn ep_worker_step(&self, _slots: &mut [Option<SequenceState>]) -> Result<bool> = Ok(false) => "ep_worker_step()";
        fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> = Ok(()) => "ep_broadcast_cmd({cmd:#x})";
        fn ep_broadcast_cmd_for_seq(&self, seq_id: u32, cmd: u32) -> Result<()> = Ok(()) => "ep_broadcast_cmd_for_seq(seq={seq_id}, cmd={cmd:#x})";
        fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> = Ok(tokens.to_vec()) => "ep_broadcast_tokens(n={})", tokens.len();
    }
}

impl ModelStreams for RecordingModel {
    fn create_stream(&self) -> Result<u64> {
        rec!(self, "create_stream() -> 1");
        self.gate.wait_start();
        Ok(1)
    }
    facts! {
        fn default_stream(&self) -> u64 = 0;
    }
    recorded! {
        fn create_event(&self) -> Result<u64> = Ok(self.fx_create_event()) => "create_event() -> {}", self.state.lock().unwrap().events_created + 2;
        fn record_event(&self, event: u64, stream: u64) -> Result<()> = { self.fx_record_event(event); Ok(()) } => "record_event(event={event}, stream={stream})";
        fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> = Ok(()) => "stream_wait_event(stream={stream}, event={event})";
        fn synchronize(&self, stream: u64) -> Result<()> = Ok(()) => "synchronize(stream={stream})";
    }
}

impl ModelDeviceFeed for RecordingModel {
    fn decode_batch_fed(
        &self,
        sources: &[FeedSource],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.fx_decode_batch_fed(sources, seqs, stream)
    }
    fn argmax_batch_to_feed(
        &self,
        logits_ptr: DevicePtr,
        masks: &[RowMask],
        dst: *mut u32,
        event: u64,
        stream: u64,
    ) -> Result<()> {
        self.fx_argmax_batch_to_feed(logits_ptr, masks, dst, event, stream)
    }
    fn reserve_decode_block(&self, seq: &mut SequenceState) -> Result<usize> {
        self.fx_reserve_decode_block(seq)
    }
    fn release_decode_blocks(&self, seq: &mut SequenceState, blocks: usize) -> Result<()> {
        self.fx_release_decode_blocks(seq, blocks)
    }
    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        metrale_gpu_runtime::host_heap::alloc_zeroed(bytes)
    }
    fn free_host_pinned(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        metrale_gpu_runtime::host_heap::free(ptr, bytes)
    }
    facts! {
        fn supports_device_token_feed(&self) -> bool = self.cfg.device_token_feed;
        fn event_query(&self, event: u64) -> Result<bool> = self.fx_event_query(event);
    }
    recorded! {
        fn event_synchronize(&self, event: u64) -> Result<()> = { self.fx_event_synchronize(event); Ok(()) } => "event_synchronize(event={event})";
        fn destroy_event(&self, event: u64) -> Result<()> = Ok(()) => "destroy_event(event={event})";
        fn fed_step_settled(&self) -> () = () => "fed_step_settled()";
    }
}
