// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Post-construction wiring for [`TransformerModel`]: borrow the GPU and
//! config, install a proposer or an n-gram embedding, and the MLA prefix-skip check.
//!
//! Owner: model-engine.
//! Invariants:
//! - `set_dflash_proposer` never grows the MTP prefill capture buffer. If shrinking it
//!   fails, the buffer is left NULL with capacity 0.

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::types::TransformerModel;
use metrale_model_layers::speculative::DraftProposer;

impl TransformerModel {
    /// 2026-09-25: Borrow the GPU backend. The factory uses it to build the
    /// proposers installed after construction and to read allocation reports.
    pub fn gpu_backend(&self) -> &dyn GpuBackend {
        self.gpu.as_ref()
    }

    /// 2026-09-25: Borrow the model config, for building the proposers that
    /// are installed after construction.
    pub fn config_ref(&self) -> &ModelConfig {
        &self.config
    }

    /// 2026-09-25: Install a proposer built after construction (the
    /// DeepSeek-V4 or GLM-5.3 MTP head, or a DFlash drafter), replacing any
    /// MTP proposer `TransformerModel::new` built. It also shrinks the MTP
    /// prefill capture buffer to the rows this proposer can use.
    ///
    /// It does not check flag combinations; clap rejects `--dflash` with
    /// `--speculative`.
    pub fn set_dflash_proposer(&mut self, proposer: std::sync::Arc<dyn DraftProposer>) {
        if self.proposer.is_some() {
            tracing::info!("DFlash: replacing existing MTP proposer with BlockDiffusionDraftHead");
        }
        // 2026-09-25: `new()` sized `mtp_prefill_hidden` before this proposer
        // existed. Ask the proposer how many rows it can be handed
        // (`DraftProposer::prefill_hidden_rows`; the default keeps the size) and
        // shrink to that; never grow. The old buffer is freed before the smaller
        // one is allocated, so the two are never held at once.
        let rows = proposer.prefill_hidden_rows(self.mtp_prefill_capacity);
        if !self.mtp_prefill_hidden.is_null() && rows < self.mtp_prefill_capacity {
            let was = self.mtp_prefill_capacity;
            let bytes = rows * self.config.hidden_size * 2;
            let old = std::mem::replace(
                &mut self.mtp_prefill_hidden,
                metrale_gpu_runtime::gpu::DevicePtr::NULL,
            );
            self.mtp_prefill_capacity = 0;
            match self.gpu.free(old).and_then(|_| self.gpu.alloc(bytes)) {
                Ok(smaller) => {
                    self.mtp_prefill_hidden = smaller;
                    self.mtp_prefill_capacity = rows;
                    tracing::info!(
                        "MTP drafter context: capture buffer rightsized {was} -> {rows} rows \
                         ({:.0} -> {:.0} MB) — the proposer cannot be handed a position past \
                         {rows} (A59)",
                        (was * self.config.hidden_size * 2) as f64 / 1e6,
                        bytes as f64 / 1e6,
                    );
                }
                // 2026-09-25: NULL with capacity 0 is the feature's off state: the
                // capture epilogue and the propose site both check it, so drafter
                // prefill is disabled and the serve keeps running.
                Err(e) => tracing::warn!(
                    "MTP drafter context: rightsizing the capture buffer failed ({e:#}) — \
                     drafter prefill and carry are DISABLED for this serve"
                ),
            }
        }
        // 2026-10-02: `TransformerModel::new` allocates the batched-verify hidden stash
        // (and MTP-catchup stash) only when it built a proposer itself
        // (`spec_buffers::alloc_verify_buffers`). GLM-5.3 / V4 install their MTP head here,
        // after construction, so the stash stayed NULL and `can_batch_verify_dispatch`
        // (`!verify_hidden_stash.is_null()`) refused every batch. Allocate them now.
        // A failure leaves them NULL: the batched route stays refused, the serve runs.
        let hs = self.config.hidden_size;
        let seqs = metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS;
        if self.verify_hidden_stash.is_null() {
            match self.gpu.alloc(seqs * hs * 2) {
                Ok(p) => self.verify_hidden_stash = p,
                Err(e) => tracing::warn!("verify hidden stash alloc failed ({e:#}); batched verify stays off"),
            }
        }
        if self.verify_catchup_stash.is_null() && self.levers.mtp_kv_exact {
            match self.gpu.alloc(seqs * metrale_model_layers::layer::MTP_CATCHUP_MAX * hs * 2) {
                Ok(p) => self.verify_catchup_stash = p,
                Err(e) => tracing::warn!("verify catchup stash alloc failed ({e:#})"),
            }
        }
        self.proposer = Some(proposer);
    }

    /// 2026-09-25: Install the fused n-gram input embedding (only the LongCat
    /// loader builds one). Once set, `embed_ctx` embeds through it and the
    /// bare-token `embed` refuses to run.
    pub fn set_ngram_embedding(
        &mut self,
        ngram: metrale_model_layers::layers::ngram_embed::NgramEmbedding,
    ) {
        tracing::info!("set_ngram_embedding: installed on the served model");
        self.ngram_embed = Some(std::sync::Mutex::new(ngram));
    }

    /// 2026-09-25: True when this model fuses n-gram lookups into its input embedding.
    pub fn has_ngram_embedding(&self) -> bool {
        self.ngram_embed.is_some()
    }

    /// 2026-09-25: True when prefill must not skip a cached prefix because a
    /// layer's local MLA prefill (`uses_local_mla_prefill`) attends only over
    /// the tokens it processes: the skipped prefix would be absent from
    /// attention. The prefix lookup then uses an empty match and prefill
    /// recomputes the whole prompt. `METRALE_MLA_PREFIX_SKIP=1` makes it
    /// return false.
    pub(crate) fn mla_prefill_needs_full_recompute(&self) -> bool {
        if std::env::var("METRALE_MLA_PREFIX_SKIP").as_deref() == Ok("1") {
            return false;
        }
        self.layers.iter().any(|l| l.uses_local_mla_prefill())
    }
}
