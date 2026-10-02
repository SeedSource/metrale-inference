// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests `engine::generate` with a mock `Model` whose logits make a
//! fixed token list the argmax.
//!
//! Owner: metrale-model-engine.
//! Invariants: none beyond the types.

use super::*;

fn argmax_bf16(data: &[u8]) -> u32 {
    debug_assert!(data.len().is_multiple_of(2));
    let n = data.len() / 2;
    if n == 0 {
        return 0;
    }
    let mut best_idx = 0u32;
    let mut best_val = bf16_to_f32(data[0], data[1]);
    for i in 1..n {
        let val = bf16_to_f32(data[i * 2], data[i * 2 + 1]);
        if val > best_val {
            best_val = val;
            best_idx = i as u32;
        }
    }
    best_idx
}

#[inline]
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = (lo as u32) | ((hi as u32) << 8);
    f32::from_bits(bits << 16)
}
use crate::traits::{
    ModelAdapters, ModelDeviceFeed, ModelDraft, ModelEp, ModelForward, ModelLifecycle, ModelLogits,
    ModelSsmState, ModelStreams, ModelVerify, ModelVision, SequenceState,
};
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-25: Mock model whose logits make `output_sequence` the argmax.
struct MockModel {
    gpu: MockGpuBackend,
    vocab_size: usize,
    /// 2026-09-25: Prefill yields entry 0; decode yields entry `seq_len`,
    /// clamped to the last entry.
    output_sequence: Vec<u32>,
    buffers: BufferArena,
    _kv_cache: parking_lot::Mutex<PagedKvCache>,
}

impl MockModel {
    fn new(output_sequence: Vec<u32>) -> Self {
        let config = ModelConfig::qwen3_next_80b_nvfp4();
        let gpu = MockGpuBackend::new();
        let buffers = BufferArena::new(&config, 1, 4096, 16, 1, &gpu).unwrap();
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: config.num_attention_layers(),
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
            v_aliases_k: false,
        };
        let kv_cache = PagedKvCache::new(kv_config, 10, &gpu).unwrap();

        Self {
            gpu,
            vocab_size: config.vocab_size,
            output_sequence,
            buffers,
            _kv_cache: parking_lot::Mutex::new(kv_cache),
        }
    }

    /// 2026-09-25: Writes BF16 logits that make `token_id` the argmax.
    fn write_logits_for_token(&self, token_id: u32) -> DevicePtr {
        let ptr = self.buffers.logits();
        let byte_len = self.vocab_size * 2;
        let mut logits = vec![0u8; byte_len];
        // 2026-09-25: Every logit -1.0 (BF16 0xBF80, little-endian).
        for i in 0..self.vocab_size {
            logits[i * 2] = 0x80;
            logits[i * 2 + 1] = 0xBF;
        }
        // 2026-09-25: The target token 10.0 (BF16 0x4120).
        let idx = token_id as usize;
        if idx < self.vocab_size {
            logits[idx * 2] = 0x20;
            logits[idx * 2 + 1] = 0x41;
        }
        self.gpu.copy_h2d(&logits, ptr).unwrap();
        ptr
    }
}

impl Model for MockModel {}

impl ModelLifecycle for MockModel {
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        seq.slot_idx = new_slot;
        Ok(())
    }
    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        // 2026-09-25: Same as the production `detach_slot_for_reuse_dispatch`
        // (model/trait_impl/sequence_compact.rs).
        if let Some(g) = seq.ssm_slot.as_mut() {
            let _ = g.take();
        }
        seq.slot_idx = usize::MAX;
    }
}

impl ModelForward for MockModel {
    fn prefill(
        &self,
        _tokens: &[u32],
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        let token_id = self.output_sequence[0];
        Ok(self.write_logits_for_token(token_id))
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _chunk_start: usize,
        _chunk_len: usize,
        is_last_chunk: bool,
        _stream: u64,
    ) -> Result<DevicePtr> {
        if is_last_chunk {
            self.prefill(tokens, seq, 0)
        } else {
            Ok(DevicePtr::NULL)
        }
    }
    fn decode(&self, _token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        seq.seq_len += 1;
        let idx = seq.seq_len.min(self.output_sequence.len() - 1);
        let token_id = self.output_sequence[idx];
        Ok(self.write_logits_for_token(token_id))
    }
    fn decode_batch(
        &self,
        _tokens: &[u32],
        _seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<DevicePtr> {
        Ok(self.buffers.logits())
    }
}

impl ModelLogits for MockModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.gpu.copy_d2h(logits_ptr, dst)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr(0)
    }
    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        // 2026-09-25: Copies to the host and takes the argmax on the CPU.
        let mut buf = vec![0u8; self.vocab_size * 2];
        self.gpu.copy_d2h(logits_ptr, &mut buf)?;
        Ok(argmax_bf16(&buf))
    }
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
        let mut results = Vec::with_capacity(n);
        let v = self.vocab_size;
        for i in 0..n {
            let ptr = DevicePtr(logits_ptr.0 + (i * v * 2) as u64);
            results.push(self.argmax_on_device(ptr, stream)?);
        }
        Ok(results)
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        self.buffers.norm_output()
    }
}

impl ModelAdapters for MockModel {}

impl ModelSsmState for MockModel {
    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _seq: &mut SequenceState, _num_accepted: usize) -> Result<()> {
        Ok(())
    }
}

impl ModelVerify for MockModel {
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let mut results = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let logits = self.decode(token, seq, stream)?;
            let tok = self.argmax_on_device(logits, stream)?;
            results.push(tok);
        }
        Ok(results)
    }
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 2]> {
        let r = self.decode_verify(&tokens[..], seq, stream)?;
        Ok([r[0], r[1]])
    }
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 3]> {
        let r = self.decode_verify(&tokens[..], seq, stream)?;
        Ok([r[0], r[1], r[2]])
    }
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 4]> {
        let r = self.decode_verify(&tokens[..], seq, stream)?;
        Ok([r[0], r[1], r[2], r[3]])
    }
}

impl ModelDraft for MockModel {
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &metrale_sampling::SamplingParams,
        _num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        crate::engine::generate(self, prompt_tokens, params)
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.decode(token, seq, stream)
    }
    fn save_hidden_for_mtp(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }
    fn run_mtp_propose(
        &self,
        _token: u32,
        _position: usize,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        Ok(None)
    }
    fn run_mtp_propose_multi(
        &self,
        _token: u32,
        _position: usize,
        _num_drafts: usize,
        _seq: &mut SequenceState,
        _stream: u64,
        _grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }
    fn trim_proposer_state(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }
}

impl ModelVision for MockModel {}

impl ModelEp for MockModel {}

impl ModelStreams for MockModel {}

impl ModelDeviceFeed for MockModel {}

#[test]
fn test_generate_stops_on_eos() {
    // 2026-09-25: Tokens 42, 99, then stop token 2.
    let model = MockModel::new(vec![42, 99, 2]);

    let params = SamplingParams {
        stop_token_ids: vec![2],
        ..SamplingParams::greedy(10)
    };

    let result = generate(&model, &[1, 2, 3], &params).unwrap();
    assert_eq!(result.output_tokens, vec![42, 99, 2]);
    assert_eq!(result.finish_reason, "stop");
}

#[test]
fn test_generate_stops_on_max_tokens() {
    let model = MockModel::new(vec![10, 10, 10, 10, 10]);

    let params = SamplingParams {
        stop_token_ids: vec![2],
        ..SamplingParams::greedy(3)
    };

    let result = generate(&model, &[1], &params).unwrap();
    assert_eq!(result.output_tokens, vec![10, 10, 10]);
    assert_eq!(result.finish_reason, "length");
}

#[test]
fn test_generate_immediate_stop() {
    // 2026-09-25: The prefill token is the stop token.
    let model = MockModel::new(vec![2]);

    let params = SamplingParams {
        stop_token_ids: vec![2],
        ..SamplingParams::greedy(10)
    };

    let result = generate(&model, &[1], &params).unwrap();
    assert_eq!(result.output_tokens, vec![2]);
    assert_eq!(result.finish_reason, "stop");
}

#[test]
fn test_argmax_bf16_basic() {
    let data: Vec<u8> = vec![
        0x80, 0x3F, // 2026-09-25: BF16 1.0, little-endian.
        0x40, 0x40, // 2026-09-25: BF16 3.0.
        0x00, 0x40, // 2026-09-25: BF16 2.0.
    ];
    assert_eq!(argmax_bf16(&data), 1);
}
