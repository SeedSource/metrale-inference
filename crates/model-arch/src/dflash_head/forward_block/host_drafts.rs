// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The drafted tokens of `forward_block`: the copy to the host staging buffer, the
//! DSpark row shift and the DSpark confidence truncation.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;

use super::dims::BlockDims;
use crate::dflash_head::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// 2026-09-26: Copies the `rows_total` drafted tokens to the host and applies the DSpark
    /// shift.
    pub(super) fn copy_drafts_out(&self, d: &BlockDims<'_>, stream: u64) -> Result<Vec<u32>> {
        let BlockDims {
            gpu,
            width,
            rows_total,
            levers,
            ..
        } = *d;
        // 2026-09-25: Copy the drafted tokens (`rows_total` u32) into the host staging
        // buffer. `copy_d2h_on_stream` returns only after the copy has finished.
        let pinned_ptr = self
            .scratch
            .draft_tokens_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        // 2026-09-25: `from_weights` is the only writer of this pointer, and a failed
        // allocation fails construction, so null here means it was never set.
        anyhow::ensure!(
            !pinned_ptr.is_null(),
            "DFlash draft-token pinned staging buffer is null (γ={}, rows={rows_total})",
            width
        );
        // 2026-09-25: SAFETY: `pinned_ptr` is non-null (checked above) and is the
        // `alloc_host_pinned(nb * gamma * 4)` region from `from_weights`, which nothing
        // frees. The span is `rows_total * 4 = n_seq * width * 4` bytes. `n_seq <= nb`
        // (`propose_batch` declines above `max_batch`), and `width <= gamma` whenever
        // `gamma >= 2` (`set_block_g` clamps to `2..=gamma.max(2)`), so it fits. The
        // region is zeroed at allocation (`GpuBackend::alloc_host_pinned`), so every byte
        // is initialised, and `u8` needs no alignment. Only this path reaches the
        // buffer, `host_buf` is its only reference and is dropped before this function
        // returns, and `copy_d2h_on_stream` returns after the copy has finished.
        let host_buf: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(pinned_ptr, rows_total * 4) };
        gpu.copy_d2h_on_stream(self.scratch.draft_tokens_dev, host_buf, stream)?;
        gpu.record_event(self.scratch.draft_tokens_event, stream)?;
        gpu.event_synchronize(self.scratch.draft_tokens_event)?;
        let mut drafts: Vec<u32> = host_buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // 2026-10-08: `METRALE_DFLASH_CONF_LOG=1`, single-sequence DFlash2 propose only: read
        // the selector's top-16 logits back for the verify step's `DFLASH CONF` line. This
        // is the lever's only sync: a blocking D2H on `stream`, after the graph replay and
        // the draft copy above, so the drafted tokens are already fixed. Lever off: skipped.
        // 2026-10-09: `METRALE_DFLASH_CONF_TAU` needs the same p, so it also arms the read.
        if (levers.conf_log || levers.dflash_conf_tau > 0.0)
            && d.n_seq == 1
            && self.dflash2_active()
        {
            self.record_conf(gpu, width, stream, &drafts)?;
        }
        // 2026-09-25: Shifted-row drafters (`shifted_rows`, set when the config's
        // `projector_type` is "dspark") predict position j + 1 in row j. Rotating the
        // vector right by one puts row j at index j + 1, so once propose drops index 0
        // the drafts are rows 0..width-2; the last row lands in index 0.
        // METRALE_DSPARK_SHIFT=1 or 0 overrides the config.
        let shift = levers.dspark_shift.unwrap_or(self.shifted_rows);
        if shift {
            drafts.rotate_right(1);
        }
        Ok(drafts)
    }
}

impl BlockDiffusionDraftHead {
    /// 2026-10-08: Softmax the `width` rows of `scratch.sel_vals` (top-16 logits, descending)
    /// and store `(p1, p1 - p2)` per draft. Drafts start at row 1 when `mask_token_id != 0`
    /// (propose drops row 0), else at row 0. 2026-10-09: `rows_drafted` (the unshifted drafted
    /// tokens) are stored from the same row, so `conf_log::p_for` matches what propose returns.
    fn record_conf(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        width: usize,
        stream: u64,
        rows_drafted: &[u32],
    ) -> Result<()> {
        let mut raw = vec![0u8; width * 16 * 4];
        gpu.copy_d2h_on_stream(self.scratch.sel_vals, &mut raw, stream)?;
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let skip = usize::from(self.mask_token_id != 0);
        let rows = vals
            .chunks_exact(16)
            .skip(skip)
            .map(crate::dflash_head::conf_log::row_conf)
            .collect();
        let tokens = rows_drafted.iter().skip(skip).copied().collect();
        crate::dflash_head::conf_log::record(rows, tokens);
        Ok(())
    }
}

/// 2026-09-26: DSpark confidence truncation of `drafts`, given the `width` BF16 confidence
/// logits in `cbuf` and the threshold `tau`.
pub(super) fn conf_truncate(drafts: &mut Vec<u32>, cbuf: &[u8], width: usize, tau: f32) {
    // 2026-09-25: sigmoid(x) < τ ⇔ x < logit(τ), so compare logits.
    let tau_logit = (tau / (1.0 - tau)).ln();
    let mut keep = width;
    for j in 0..width.saturating_sub(1) {
        let bits = u16::from_le_bytes([cbuf[j * 2], cbuf[j * 2 + 1]]);
        let logit = f32::from_bits((bits as u32) << 16);
        if logit < tau_logit {
            keep = j + 1;
            break;
        }
    }
    drafts.truncate(keep.max(1));
}
