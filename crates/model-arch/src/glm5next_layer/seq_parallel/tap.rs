// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: DFlash tap capture under the sequence-parallel staged prefill, for the hybrid
//! MTP + DFlash serve (`METRALE_HYBRID_DFLASH=1`, which keeps `METRALE_GLM_PREFILL_SEQ_PARALLEL`
//! on; a pure DFlash serve still turns it off).
//!
//! Under SP a tap layer's FFN back runs `hc_head_mean` on this rank's own rows only
//! (`SpRows::back` with `last`), so after the layer each rank holds the tap rows it owns. The
//! DFlash capture (model-engine `try_dflash_prefill_capture_layer`) reads chunk rows from
//! `hidden`; before it does, both ranks call `TransformerLayer::prefill_tap_gather` with the
//! chunk rows the capture keeps (`tap_window_rows`), and the layer swaps exactly those rows
//! with the peer (`swap_owned`), so every rank leaves with all of them. Rows outside that
//! window are not moved.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The record (`SpTapRecord`) is cleared at the top of every prefill of the layer and set only
//!   by an SP prefill of a tap layer that is not the last layer (the last layer's final swap
//!   already leaves every row on both ranks). Every input to it is the same on both ranks, so
//!   both ranks swap, or neither does.
//! - A gather consumes the record. A record for another chunk is stale (that chunk's sequence
//!   had no DFlash state) and is dropped; one for this chunk but another `hidden` buffer is an
//!   error, since the capture would read rows this rank does not hold.

use anyhow::{Result, bail};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{SpPlan, Span, swap_owned};

/// 2026-10-10: The SP prefill of one chunk of a tap layer: where it ran and its ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpTapTicket {
    pub seq_len_start: usize,
    pub total: usize,
    pub hidden: u64,
    pub plan: SpPlan,
}

/// 2026-10-10: The layer's last SP tap prefill, if its rows are still split across the ranks.
#[derive(Debug, Default)]
pub struct SpTapRecord(std::sync::Mutex<Option<SpTapTicket>>);

impl SpTapRecord {
    /// 2026-10-10: Replace the record (`None` at the top of every prefill).
    pub fn set(&self, t: Option<SpTapTicket>) {
        if let Ok(mut g) = self.0.lock() {
            *g = t;
        }
    }

    /// 2026-10-10: Take the record. `Ok(Some(plan))` when it is this chunk's; `Ok(None)` when
    /// there is none or it is another chunk's; an error when it is this chunk's on another
    /// `hidden` buffer.
    pub fn take_for(
        &self,
        seq_len_start: usize,
        total: usize,
        hidden: u64,
    ) -> Result<Option<SpPlan>> {
        let Some(t) = self.0.lock().ok().and_then(|mut g| g.take()) else {
            return Ok(None);
        };
        if t.seq_len_start != seq_len_start || t.total != total {
            return Ok(None);
        }
        if t.hidden != hidden {
            bail!(
                "SP tap gather: the prefill of rows {seq_len_start}..+{total} ran on hidden \
                 {:#x}, the capture reads {hidden:#x}",
                t.hidden
            );
        }
        Ok(Some(t.plan))
    }
}

/// 2026-10-10: The rows of a chunk (absolute `[chunk_start, chunk_start + chunk_len)`) inside
/// the absolute window `[lo, hi)`, as a chunk-local span. Empty (`(_, 0)`) when they do not
/// meet. The engine passes the window the DFlash prefill capture keeps,
/// `PrefillCtxWindow::span` (the last `max_ctx_len` prompt positions), so these are the rows
/// `PrefillCtxWindow::chunk_rows` captures.
pub fn tap_window_rows(chunk_start: usize, chunk_len: usize, (lo, hi): Span) -> Span {
    let a = chunk_start.max(lo);
    let b = (chunk_start + chunk_len).min(hi);
    if b > a {
        (a - chunk_start, b - a)
    } else {
        (0, 0)
    }
}

/// 2026-10-10: Swap the tap rows `rows` (chunk-local) of `hidden` with the peer under `plan`:
/// send this rank's rows, receive the peer's. Both ranks must call it with the same arguments.
pub fn gather_tap_rows(
    comm: &dyn CommBackend,
    plan: SpPlan,
    hidden: DevicePtr,
    rows: Span,
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    if rows.1 == 0 {
        return Ok(());
    }
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::warn!(
            "HYBRID SP TAP CAPTURE: ENGAGED (first gather: {} tap rows of a {}-row SP chunk, \
             each rank sends the rows it owns)",
            rows.1,
            plan.total
        );
    });
    swap_owned(comm, plan, hidden, rows, row_bytes, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticket(seq_len_start: usize, hidden: u64) -> SpTapTicket {
        SpTapTicket {
            seq_len_start,
            total: 8192,
            hidden,
            plan: SpPlan {
                width: 8192,
                total: 8192,
            },
        }
    }

    #[test]
    fn a_record_is_taken_once_and_only_for_its_own_chunk() {
        let r = SpTapRecord::default();
        assert_eq!(r.take_for(0, 8192, 0x1000).unwrap(), None);
        r.set(Some(ticket(0, 0x1000)));
        let plan = r.take_for(0, 8192, 0x1000).unwrap();
        assert_eq!(plan, Some(ticket(0, 0x1000).plan));
        assert_eq!(r.take_for(0, 8192, 0x1000).unwrap(), None, "taken once");
        // A stale record (another chunk) is dropped, not used.
        r.set(Some(ticket(8192, 0x1000)));
        assert_eq!(r.take_for(0, 8192, 0x1000).unwrap(), None);
        assert_eq!(r.take_for(8192, 8192, 0x1000).unwrap(), None);
        // This chunk on another buffer: the capture would read rows this rank lacks.
        r.set(Some(ticket(0, 0x2000)));
        assert!(r.take_for(0, 8192, 0x1000).is_err());
        // Cleared by the next prefill.
        r.set(Some(ticket(0, 0x1000)));
        r.set(None);
        assert_eq!(r.take_for(0, 8192, 0x1000).unwrap(), None);
    }
}
