// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `DflashScratch`, a drafter head's device scratch, and
//! `DflashProposerState`, its per-sequence state.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Device scratch for the γ-block forward, allocated once per head
/// in `from_weights.rs`. `nb` below is the head's `max_batch`.
pub struct DflashScratch {
    pub stream_buf: DevicePtr,
    pub norm_buf: DevicePtr,
    pub q_buf: DevicePtr,
    pub k_buf: DevicePtr,
    pub v_buf: DevicePtr,
    pub attn_out: DevicePtr,
    pub mlp_intermediate: DevicePtr,
    pub mlp_up: DevicePtr,
    pub stream_acc: DevicePtr,
    /// 2026-09-25: `[ctx_window, hidden]` BF16: ctx rows after `fc` and
    /// `hidden_norm`.
    pub fc_proj: DevicePtr,
    /// 2026-09-25: `[ctx_window, L * 2 * kv_dim]` BF16: the ctx precompute's KV
    /// GEMM output.
    pub fused_kv_out: DevicePtr,
    /// 2026-09-25: `i64[ctx_window]` paged-cache slot indices for
    /// `reshape_and_cache`.
    pub slot_mapping_dev: DevicePtr,
    /// 2026-09-25: `[PRECOMPUTE_BATCH_ROWS, L_t * h_t]` BF16 staging for the
    /// batched ctx precompute: the uncommitted ctx rows of several sequences,
    /// contiguous.
    pub precompute_in: DevicePtr,
    /// 2026-09-25: `nb` triples `[u32 kv_len, u32 q_offset, u32 q_rope_pos]`,
    /// 12 bytes each, one per band, read by the indirect paged-attention kernel
    /// at entry. The host writes them before the captured region, so a captured
    /// graph sees a fixed pointer.
    pub option_b_indirect_args_dev: DevicePtr,
    /// 2026-09-25: Pinned host buffer, `nb * gamma * 4` bytes, for the
    /// draft-token D2H copy; allocated once in `from_weights.rs`. `AtomicPtr`
    /// keeps `DflashScratch` `Send + Sync`. Nothing stores to it after
    /// construction, so `Relaxed` loads are enough.
    pub draft_tokens_host_pinned: std::sync::atomic::AtomicPtr<u8>,
    /// 2026-09-25: Event recorded after the draft-token D2H copy; the host waits
    /// on it before reading the pinned buffer. Created once in `from_weights.rs`.
    pub draft_tokens_event: u64,
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// 2026-09-25: i32 RoPE positions: the `eff_ctx` ctx rows, then each
    /// sequence's block rows from that sequence's position.
    pub position_ids: DevicePtr,
    /// 2026-09-25: DSpark Markov scratch: `[1, markov_rank]` BF16 latent for the
    /// prev-token gather (`markov_w1[prev]`). `DevicePtr(0)` when the
    /// drafter has no Markov head.
    pub markov_embed: DevicePtr,
    /// 2026-09-25: DSpark Markov scratch: `[vocab]` BF16 bias
    /// (`markov_w2 @ markov_embed`), residual-added onto one logits row
    /// per sequential step. `DevicePtr(0)` when no Markov head.
    pub markov_bias: DevicePtr,
    /// 2026-09-25: DSpark confidence scratch: `[γ]` BF16 per-row acceptance
    /// logits, read back on the host after the draft-token D2H to pick how many
    /// drafts to keep. `DevicePtr(0)` when the drafter has no confidence head.
    pub conf_out: DevicePtr,

    // 2026-09-25: DFlash2 scratch, `DevicePtr(0)` when the checkpoint has no
    // selector.
    /// 2026-09-25: `[nb * γ, 2 * conv_kernel_size * (hidden / conv_group_size)]`
    /// BF16 dynamic conv kernels, reused by each conv site in turn.
    pub conv_dyn: DevicePtr,
    /// 2026-09-25: `[nb * γ, hidden]` BF16 staging for the convolved hidden.
    pub conv_tmp: DevicePtr,
    /// 2026-09-25: `[nb * γ, 16]` f32: the selector's top-16 logits per row.
    pub sel_vals: DevicePtr,
    /// 2026-09-25: `[nb * γ, 16]` u32: the selector's top-16 token ids per row.
    pub sel_idx: DevicePtr,
    /// 2026-09-25: `[nb * γ, selector_rank]` BF16: the selector's hidden
    /// projection.
    pub sel_hproj: DevicePtr,
}

/// 2026-09-25: Per-sequence DFlash drafter state. One block table serves every
/// drafter layer: the paged cache has a pool per layer, addressed by the same
/// slots.
pub struct DflashProposerState {
    /// 2026-09-25: Host copy of the drafter block table.
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    /// 2026-09-25: Drafts the last propose returned; `after_verify` and
    /// `free_state` reset it to 0.
    pub last_num_drafted: usize,
    pub prefill_done: bool,
    /// 2026-09-25: `max_ctx_len` rows of `ctx_slot_bytes`: captured target
    /// hiddens, one row per ctx position, the first `ctx_len` valid. Model-engine
    /// appends rows and slides the window when it is full; `propose_drafts`
    /// appends while there is room.
    pub ctx_hidden_acc: DevicePtr,
    /// 2026-09-25: Valid rows in `ctx_hidden_acc`, at most `max_ctx_len`.
    pub ctx_len: usize,
    pub last_num_accepted: usize,
    /// 2026-09-25: Model-engine has already appended the latest capture; the
    /// next `propose_drafts` skips its own append and clears the flag.
    pub skip_next_decode_append: bool,
    /// 2026-09-25: Rows `ctx_hidden_acc` holds (the `window` of
    /// `alloc_state_windowed`).
    pub max_ctx_len: usize,
    /// 2026-09-25: Bytes per `ctx_hidden_acc` row:
    /// `target_layer_ids.len() * target_hidden_size * 2`.
    pub ctx_slot_bytes: usize,

    /// 2026-09-25: Device copy of `block_table`, made once when the first
    /// propose allocates the table; `None` before that and after `free_state`.
    pub block_table_dev: Option<DevicePtr>,
    /// 2026-09-25: Ctx rows of the paged cache the γ block attends to;
    /// `propose_drafts` sets it to `ctx_len`.
    pub ctx_count_drafter: usize,
    /// 2026-09-25: Slots the allocated block table covers
    /// (`block_table.len() * BLOCK_SIZE`).
    pub max_ctx_count_drafter: usize,
    /// 2026-09-25: Ctx rows `[0, ctx_committed)` whose K/V is already in the
    /// paged cache; the next precompute starts here. Model-engine's window
    /// slides reset it to 0.
    pub ctx_committed: usize,
    /// 2026-09-25: One RoPE position per ctx row (`len() == ctx_len`), stamped
    /// when the row is appended and kept when the window slides. Model-engine's
    /// `update_dflash_ctx_len_after_prefill` seeds prefill row i with i.
    pub ctx_positions: Vec<i32>,
    /// 2026-10-09: Two-rank drafter, rank 0 only: the position of the newest ctx row rank 1
    /// holds (`tp/leader.rs`). Prefill capture sets it; `None` makes the next propose
    /// resend every row.
    pub tp_synced_pos: Option<i32>,
}

impl ProposerState for DflashProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// 2026-10-10: The prompt positions a DFlash prefill capture keeps: the last `max_ctx_len` of
/// the prompt, absolute `[start, end)` with `end` the prompt length and
/// `start = end - max_ctx_len` (saturating). The drafter attends over a sliding window of the
/// most recent rows, so the tail next to the decode position must be in the context. Before
/// 2026-10-10 the capture kept the first `max_ctx_len` positions, which left a prompt longer
/// than the window without its tail. For a prompt within the window `start` is 0 and every
/// value below is what the first-W capture gave.
///
/// Accumulator slot `i` holds position `start + i`; the slots do not move between chunks,
/// since `start` depends on the prompt length alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillCtxWindow {
    pub start: usize,
    pub end: usize,
}

impl PrefillCtxWindow {
    /// 2026-10-10: The window of a prompt of `prompt_len` tokens. `chunk_end` (the end of the
    /// chunk just run) raises `end` if `prompt_len` is stale, so a captured row never falls
    /// past the window.
    pub fn new(prompt_len: usize, chunk_end: usize, max_ctx_len: usize) -> Self {
        let end = prompt_len.max(chunk_end);
        Self {
            start: end.saturating_sub(max_ctx_len),
            end,
        }
    }

    /// 2026-10-10: `(start, end)`.
    pub fn span(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    /// 2026-10-10: The rows of the chunk `[chunk_start, chunk_start + chunk_len)` inside the
    /// window, as a chunk-local `(first row, rows)`; `(0, 0)` when none.
    pub fn chunk_rows(&self, chunk_start: usize, chunk_len: usize) -> (usize, usize) {
        let a = chunk_start.max(self.start);
        let b = (chunk_start + chunk_len).min(self.end);
        if b > a {
            (a - chunk_start, b - a)
        } else {
            (0, 0)
        }
    }

    /// 2026-10-10: Captured rows (`ctx_len`) once the prompt has run up to `chunk_end`.
    pub fn len_after(&self, chunk_end: usize) -> usize {
        chunk_end.min(self.end).saturating_sub(self.start)
    }

    /// 2026-10-10: The absolute positions of those rows, one per slot.
    pub fn positions_after(&self, chunk_end: usize) -> Vec<i32> {
        (self.start..self.start + self.len_after(chunk_end))
            .map(|p| p as i32)
            .collect()
    }
}

#[cfg(test)]
mod prefill_window_tests {
    use super::PrefillCtxWindow;

    /// 2026-10-10: The rows the first-W capture (before 2026-10-10) kept from a chunk, and
    /// its `ctx_len` after it: the reference for prompts within the window.
    fn first_w(chunk_start: usize, chunk_len: usize, w: usize) -> (Vec<usize>, usize) {
        let rows = (0..chunk_len)
            .take_while(|&t| chunk_start + t < w)
            .collect();
        (rows, (chunk_start + chunk_len).min(w))
    }

    fn kept(win: &PrefillCtxWindow, cs: usize, n: usize) -> Vec<usize> {
        let (t, k) = win.chunk_rows(cs, n);
        (t..t + k).collect()
    }

    /// Prompts within the window (P < W and P = W) keep exactly what the first-W capture kept,
    /// chunk by chunk, with the same `ctx_len` and positions 0..len.
    #[test]
    fn within_the_window_is_identical_to_first_w() {
        let w = 2048;
        for p in [1usize, 7, 1000, 2047, 2048] {
            for chunk in [1usize, 256, 1000, 2048, 4096] {
                let mut cs = 0;
                while cs < p {
                    let n = chunk.min(p - cs);
                    let win = PrefillCtxWindow::new(p, cs + n, w);
                    assert_eq!(win.start, 0);
                    let (rows, len) = first_w(cs, n, w);
                    assert_eq!(kept(&win, cs, n), rows, "p={p} chunk={chunk} cs={cs}");
                    assert_eq!(win.len_after(cs + n), len, "p={p} chunk={chunk} cs={cs}");
                    let want: Vec<i32> = (0..len as i32).collect();
                    assert_eq!(win.positions_after(cs + n), want);
                    cs += n;
                }
            }
        }
    }

    /// P > W: the last W positions, with chunk boundaries before, at and after `start`; each
    /// position lands in slot `pos - start` once, and the positions are absolute.
    #[test]
    fn past_the_window_keeps_the_tail() {
        let (p, w) = (5000usize, 2048usize);
        let s = p - w;
        for chunk in [1000usize, s, s - 1, s + 1, 2048, 5000] {
            let mut slots = vec![usize::MAX; w];
            let mut cs = 0;
            while cs < p {
                let n = chunk.min(p - cs);
                let win = PrefillCtxWindow::new(p, cs + n, w);
                assert_eq!(win.span(), (s, p));
                for t in kept(&win, cs, n) {
                    let pos = cs + t;
                    assert!(pos >= s, "chunk={chunk}: kept position {pos} before {s}");
                    assert_eq!(slots[pos - s], usize::MAX, "slot written twice");
                    slots[pos - s] = pos;
                }
                let len = win.len_after(cs + n);
                assert_eq!(len, (cs + n).saturating_sub(s), "chunk={chunk} cs={cs}");
                let pos = win.positions_after(cs + n);
                assert_eq!(pos.len(), len);
                assert_eq!(pos.first().copied(), (len > 0).then_some(s as i32));
                cs += n;
            }
            assert_eq!(slots, (s..p).collect::<Vec<_>>(), "chunk={chunk}");
        }
        // A chunk ending exactly at `start` keeps nothing; one starting there keeps all.
        let win = PrefillCtxWindow::new(p, s, w);
        assert_eq!(win.chunk_rows(s - 100, 100), (0, 0));
        assert_eq!(win.chunk_rows(s, 100), (0, 100));
        assert_eq!(win.chunk_rows(s - 1, 2), (1, 1));
    }

    /// A stale (short) prompt length never lets a row past the window.
    #[test]
    fn a_stale_prompt_len_is_raised_to_the_chunk_end() {
        let win = PrefillCtxWindow::new(0, 300, 2048);
        assert_eq!(win.span(), (0, 300));
        let win = PrefillCtxWindow::new(10, 3000, 2048);
        assert_eq!(win.span(), (952, 3000));
    }
}
