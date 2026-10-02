// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `impl ModelVerify for TransformerModel`, mostly delegating to `<method>_dispatch`
//! helpers in the sibling modules.
//!
//! Owner: model-engine.
//! Invariants: the ones in `trait_impl/mod.rs`.

use anyhow::Result;

use crate::model::types::TransformerModel;
use crate::traits::{ModelVerify, SequenceState};

impl ModelVerify for TransformerModel {
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.gdn_carry_flush_pending()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        let r = self.decode_verify_dispatch(tokens, seq, stream);
        self.release_verify_capture_on_err(r)
    }

    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        self.gdn_carry_flush_pending()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        let r = self.decode_verify_graphed_dispatch(tokens, seq, _stream);
        self.release_verify_capture_on_err(r)
    }

    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 3]> {
        self.gdn_carry_flush_pending()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        let r = self.decode_verify_graphed_k3_dispatch(tokens, seq, _stream);
        self.release_verify_capture_on_err(r)
    }

    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 4]> {
        self.gdn_carry_flush_pending()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        let r = self.decode_verify_graphed_k4_dispatch(tokens, seq, _stream);
        self.release_verify_capture_on_err(r)
    }

    fn can_batch_verify(&self, ks: &[usize]) -> bool {
        let ok = self.can_batch_verify_dispatch(ks);
        self.log_batch_verify_gate_once(ks, ok);
        ok
    }

    fn decode_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
        opts: crate::traits::VerifyBatchedOpts,
    ) -> Result<Vec<u32>> {
        self.ssm_pool.require_verify_rollback_supported()?;
        // 2026-10-02: Multi-rank (`verify_ep.rs`): the workers get the whole batch first, and
        // after a failure the abort verdict they wait for; a no-op on one GPU.
        let sent = self.ep_send_verify_batch(tokens, ks, &*seqs)?;
        let r = self.decode_verify_batched_dispatch(tokens, ks, seqs, _stream, opts);
        if sent && r.is_err() {
            self.ep_send_verify_batch_abort(ks.len());
        }
        self.release_verify_capture_on_err(r)
    }

    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        self.gdn_carry_flush_pending()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        let r = self.decode_verify_graphed_kgamma_dispatch(tokens, seq, _stream);
        self.release_verify_capture_on_err(r)
    }

    fn decode_and_verify_fused(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        self.gdn_carry_flush_pending()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        let r = self.decode_and_verify_fused_dispatch(tokens, seq, _stream);
        self.release_verify_capture_on_err(r)
    }

    fn gdn_fold_accepted(
        &self,
        slots: &[usize],
        accepted_rows: &[u32],
        k_rows: usize,
    ) -> Result<bool> {
        self.gdn_fold_accepted_dispatch(slots, accepted_rows, k_rows)
    }
}

impl TransformerModel {
    /// 2026-10-02: Logs once at INFO the first batched-verify gate answer, and once the first
    /// refusal, with every clause of `can_batch_verify_dispatch`, so a run's log shows whether
    /// the batched route was taken and which clause refused it.
    fn log_batch_verify_gate_once(&self, ks: &[usize], ok: bool) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static SEEN_OK: AtomicBool = AtomicBool::new(false);
        static SEEN_NO: AtomicBool = AtomicBool::new(false);
        if SEEN_OK.load(Ordering::Relaxed) && (ok || SEEN_NO.load(Ordering::Relaxed)) {
            return;
        }
        let seen = if ok { &SEEN_OK } else { &SEEN_NO };
        if seen.swap(true, Ordering::Relaxed) {
            return;
        }
        let rows: usize = ks.iter().sum();
        let min_cap = self
            .layers
            .iter()
            .filter(|l| l.decode_verify_multi_own_states())
            .map(|l| l.decode_verify_multi_max_rows())
            .min();
        tracing::info!(
            "batched MTP verify gate {}: ks={ks:?} rows={rows} comm={} ep_ok={} v2={} dflash_save={} \
             stash={} layers_declining={} layers_own_states={}/{} min_row_cap={min_cap:?} \
             lora={} hss={}",
            if ok { "ADMITS (route engaged)" } else { "REFUSES (per-sequence verify)" },
            self.comm.is_some(),
            self.batched_verify_ep_ok(),
            self.ep_protocol_v2,
            self.dflash_hidden_save.is_some(),
            !self.verify_hidden_stash.is_null(),
            self.layers.iter().filter(|l| l.decode_verify_multi_unsupported()).count(),
            self.layers.iter().filter(|l| l.decode_verify_multi_own_states()).count(),
            self.layers.len(),
            self.lora.is_some(),
            self.kv_cache.lock().config().cache_blocks_per_seq.is_some(),
        );
    }
}
