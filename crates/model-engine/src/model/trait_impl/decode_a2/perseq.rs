// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-sequence arm of `decode_batch_dispatch`: each row runs `decode()` and
//! its logits row is staged through the host into the batched logits buffer.
//!
//! Owner: model-engine (decode).
//! Invariants: `suppress_graphs` holds its previous value again when this returns.
//! Known gap (2026-10-05, A168 review): each row's lazy-map vote runs inside its own `decode()`,
//! so a refusal at row i > 0 comes after rows 0..i-1 advanced on both ranks, and the scheduler's
//! preempt-and-relaunch decodes those rows again (a duplicated token). The same holds for a KV
//! exhaustion mid-loop before the vote. The GLM ship envs set `METRALE_GLM_DECODE_MULTI_SEQ=1`,
//! whose batched path votes every row before any advances (`decode_a2.rs`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::{ModelEp, ModelForward, SequenceState};

impl TransformerModel {
    /// 2026-09-26: Run `decode()` for each of the `n` sequences with CUDA graphs suppressed,
    /// collecting their logits rows, and return the batched logits pointer.
    pub(super) fn decode_batch_perseq(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        n: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        use std::sync::atomic::Ordering;
        let logits = self.decode_logits_ptr();
        let v = self.config.vocab_size;
        let elem = if self.decode_logits_fp32() { 4 } else { 2 };
        let row_bytes = v * elem;
        // 2026-09-25: No CUDA graphs inside the loop; the previous setting is
        // restored after it.
        let prev_suppress = self.suppress_graphs.swap(true, Ordering::Relaxed);
        // 2026-09-25: `decode()` runs on the backend's `default_stream()` whatever
        // `stream` is, so stage the rows on that stream.
        let copy_stream = self.gpu.default_stream();
        let result = (|| -> Result<()> {
            let mut staged = vec![0u8; n * row_bytes];
            for i in 0..n {
                // 2026-09-25: The same announcement as the `n == 1` path, so an
                // EP worker runs this sequence's forward; no-op without a
                // communicator.
                self.ep_broadcast_cmd_for_seq(seqs[i].slot_idx as u32, tokens[i])?;
                self.decode(tokens[i], seqs[i], stream)?;
                // 2026-09-25: `decode()` wrote this sequence's logits to row 0;
                // copy them out before the next `decode()` overwrites them.
                // `copy_d2h_on_stream` orders the copy after the kernels on
                // `copy_stream` and waits for it.
                self.gpu.copy_d2h_on_stream(
                    logits,
                    &mut staged[i * row_bytes..(i + 1) * row_bytes],
                    copy_stream,
                )?;
            }
            // 2026-09-25: Upload the assembled [n, vocab] logits.
            self.gpu.copy_h2d_async(&staged, logits, copy_stream)?;
            self.gpu.synchronize(copy_stream)?;
            Ok(())
        })();
        self.suppress_graphs.store(prev_suppress, Ordering::Relaxed);
        result?;
        Ok(logits)
    }
}
