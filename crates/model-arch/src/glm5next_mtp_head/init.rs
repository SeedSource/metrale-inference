// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextMtpHead::new`: the drafter's private KV pool and its vocab-sharded draft head.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - A constructed head's `max_seq_len` is at most `max_dsa_context` of its DSA block.
//! - `head_n` is `vocab_size / tp_world_size` when `tp_world_size > 1` divides
//!   `vocab_size`, and `vocab_size` otherwise.

use super::*;

impl Glm5NextMtpHead {
    pub fn new(
        module: Glm5NextMtpModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
    ) -> Result<Self> {
        let dsa = match &module.layer.mixer {
            crate::glm5next_layer::Glm5NextMixer::Dsa(l) => l,
            _ => bail!("GLM MTP block is not a DSA layer"),
        };
        // 2026-09-25: The drafter block is a DSA layer, so it can never reach a position past
        // `max_dsa_context`: that is the indexer cache `Glm5NextDsaState::alloc` reserves, and
        // `advance` refuses to grow past it. Everything sized from `max_seq_len` here uses the
        // capped value: the private KV pool below, the block table claimed in `alloc_state`,
        // the bounds checks in `forward_one` and `rows_impl`, and, through
        // `prefill_hidden_rows`, the model's `mtp_prefill_hidden` capture.
        let max_seq_len = drafter_context_rows(max_seq_len, &dsa.cfg);
        let kv_config = drafter_kv_config(dsa.cfg.kv_lora_rank);
        // 2026-10-04: Under `METRALE_GLM_MTP_SEQ_KV` every state allocates its own pool
        // (`alloc_state_for`) and nothing reads this one, so it shrinks to two blocks.
        let seq_kv = batch::mtp_seq_kv();
        let blocks = if seq_kv {
            2
        } else {
            max_seq_len / kv_config.block_size + 2
        };
        let kv_cache = PagedKvCache::new(kv_config, blocks, gpu)?;
        // 2026-09-25: The draft `lm_head` sweep is most of the drafter's time. Measured
        // 2026-08-29 with `METRALE_GLM_MTP_SKIP=head`: propose 8.84 ms -> 1.52 ms. It reads
        // `[vocab, hidden]` BF16, 1.27 GB at GLM-5.3's 154,880 x 4,096.
        //
        // So the sweep is split by vocab across the ranks that run propose together: each
        // rank sweeps its `head_n` rows over the full hidden dimension, and the ranks exchange
        // (max, argmax) through one 16-byte all-reduce (`forward_one`).
        //
        // Vocab, not hidden: rows of `[vocab, hidden]` are contiguous, so a vocab shard is a
        // base-pointer offset and a smaller `n`. A hidden shard would need a row stride, and
        // the gemv takes none; it assumes rows packed at K.
        let head_world = config.tp_world_size.max(1);
        let head_rank = config.tp_rank;
        let head_n = if head_world > 1 && config.vocab_size.is_multiple_of(head_world) {
            config.vocab_size / head_world
        } else {
            config.vocab_size
        };
        // 2026-09-25: Quantise only the rows this rank sweeps: `head_n * hidden` bytes, not
        // the whole vocab. A failure here is not fatal; the head falls back to the BF16 sweep.
        let gemv_fp8w_k =
            metrale_model_layers::layers::try_kernel(gpu, "gemv_fp8w", "dense_gemv_fp8w");
        let head_fp8 = if std::env::var("METRALE_GLM_MTP_HEAD_FP8").as_deref() == Ok("0")
            || gemv_fp8w_k.0 == 0
        {
            None
        } else {
            let shard = DenseWeight {
                weight: lm_head
                    .weight
                    .offset(head_rank * head_n * config.hidden_size * 2),
            };
            match gpu
                .kernel("gemv_fp8w", "quantize_bf16_to_fp8")
                .and_then(|qk| {
                    metrale_model_layers::weight_map::quantize_to_fp8(
                        &shard,
                        head_n,
                        config.hidden_size,
                        gpu,
                        qk,
                        gpu.default_stream(),
                    )
                }) {
                Ok(q) => {
                    tracing::info!(
                        "GLM MTP: draft lm_head shard quantised to FP8 ({} rows x {}, {} MB)",
                        head_n,
                        config.hidden_size,
                        head_n * config.hidden_size / (1024 * 1024),
                    );
                    Some(q)
                }
                Err(e) => {
                    tracing::warn!("GLM MTP: FP8 draft head unavailable ({e:#}); staying BF16");
                    None
                }
            }
        };

        // 2026-10-03: Scratch for `METRALE_GLM_MTP_CTX_ROWBATCH=1` (about 10 MB at 256 rows).
        let ctx_scratch = if ctx_rowbatch() && module.layer.can_drafter_write_kv_rows() {
            let (t, h) = (CTX_TILE, config.hidden_size);
            Some(Mutex::new(CtxScratch {
                gath: gpu.alloc(t * h * 2)?,
                nrm: gpu.alloc(t * h * 2)?,
                concat: gpu.alloc(t * 2 * h * 2)?,
                xo: gpu.alloc(t * h * 2)?,
                kv_a: gpu.alloc(t * dsa.cfg.kv_lora_rank * 2)?,
                slots: gpu.alloc(t * 8)?,
            }))
        } else {
            None
        };

        // 2026-10-04: `METRALE_GLM_MTP_BATCH_DRAFT=1`: the batched propose's kernels and
        // scratch, sized once here; `None` (per-sequence proposes only) when the lever is off
        // or a kernel is missing.
        let batch =
            batch::BatchScratch::new(gpu, config.hidden_size, config.vocab_size)?.map(Mutex::new);
        let kv_lora_rank = dsa.cfg.kv_lora_rank;

        Ok(Self {
            ctx_scratch,
            module,
            embed_tokens,
            lm_head,
            kv_cache: Mutex::new(kv_cache),
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            hidden: config.hidden_size,
            vocab: config.vocab_size,
            max_seq_len,
            head_rank,
            head_v0: head_rank * head_n,
            head_n,
            head_fp8,
            gemv_fp8w_k,
            kv_lora_rank,
            seq_kv,
            batch,
        })
    }
}

/// 2026-10-04: The drafter's latent pool geometry: one KV head of `kv_lora_rank` FP8 per token,
/// 16-token blocks, one layer. The head's shared pool and every per-sequence pool
/// (`METRALE_GLM_MTP_SEQ_KV`) use it.
pub(super) fn drafter_kv_config(kv_lora_rank: usize) -> KvCacheConfig {
    // 2026-09-25: One KV head of `kv_lora_rank` per token, the absorbed-MLA latent that
    // the block's `latent_write` and paged gather address.
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: kv_lora_rank,
        num_layers: 1,
        dtype: KvCacheDtype::Fp8,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        // 2026-10-01: The drafter block, like the target's DSA layers,
        // touches only the K pool (`write_kv_row`, the paged gather).
        v_aliases_k: metrale_cache::kv_cache::glm_kv_v_alias("glm5_next"),
    }
}
