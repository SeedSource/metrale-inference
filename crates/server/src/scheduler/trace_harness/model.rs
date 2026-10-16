// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The recording fake model: state, scripts, gates and the trace buffer.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! The `Model` trait impl lives in `model_impl.rs`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::SequenceState;

/// 2026-09-25: Device address the fake reports as its logits buffer. Rows are `vocab *
/// 2` bytes apart, so a pointer handed back by the scheduler decodes to a
/// (row, byte offset) pair.
pub(super) const LOGITS_BASE: u64 = 0x1000_0000;

/// 2026-09-25: The argmax the fake returns at a verify position whose input, or an
/// earlier input, was a wrong draft. The prefix-accept verdict stops at the
/// first mismatch, before these positions, so any fixed id works; a fixed
/// one keeps the trace stable.
pub(super) const GARBAGE: u32 = 1;

/// 2026-09-25: Per-request script: what the model "wants" to generate after the prompt.
#[derive(Clone, Debug)]
pub(super) struct SeqScript {
    /// 2026-09-25: Prompt length, so a generated index can be derived from `seq.tokens`.
    pub prompt_len: usize,
    /// 2026-09-25: Tokens generated in order. Past the end the script repeats its last
    /// token.
    pub tokens: Vec<u32>,
    /// 2026-09-25: A draft for a generated index that is a multiple of `n` is wrong, so
    /// verify rejects it and the partial-accept paths run. `None` is a perfect
    /// drafter.
    pub draft_wrong_every: Option<usize>,
    /// 2026-09-25: Beam search output for a beam request.
    pub beam_hyp: Option<Vec<u32>>,
    /// 2026-09-25: Flip the request's cancel flag when `maybe_cancel` sees this many
    /// generated tokens in `seq.tokens`.
    pub cancel_at: Option<usize>,
}

/// 2026-09-25: The fake's configuration: the facts it reports and the faults it injects.
#[derive(Clone, Debug)]
pub(super) struct ModelCfg {
    pub vocab: usize,
    pub block_size: usize,
    pub total_blocks: usize,
    pub has_proposer: bool,
    pub has_self_speculative: bool,
    pub supports_beam: bool,
    pub has_ssm_layers: bool,
    pub ring_slots: usize,
    pub dflash_gamma: Option<usize>,
    pub twophase_ok: bool,
    pub can_batch_verify: bool,
    pub slot_draft_capacity: usize,
    /// 2026-09-25: Indexes of `decode_batch` / `decode_batch_fed` calls (0-based, counted
    /// per model) that fail with "KV cache exhausted" instead of running.
    pub kv_exhaust_at: Vec<usize>,
    /// 2026-10-14 (race-memory, A184): Indexes of `decode_verify_batched` calls (0-based,
    /// counted per model) that fail with "KV cache exhausted" instead of running.
    pub verify_kv_exhaust_at: Vec<usize>,
    /// 2026-10-15 (race-prefill, A194): (session, n): the first n `prefill_chunk` calls at
    /// offset 0 for that session fail with a rank-agreed "KV cache exhausted" (a new request
    /// refused at chunk 0, as fchr arm A's DSA indexer pool refusals).
    pub chunk0_kv_refuse: Option<(u64, usize)>,
    /// 2026-09-25: Blocks the prefix cache can give back on `reclaim_prefix_blocks`.
    pub reclaimable: usize,
    /// 2026-09-25: Blocks pinned outside any live sequence (a warm prefix cache): they
    /// lower `num_free_blocks` without being reclaimable.
    pub held_blocks: usize,
    /// 2026-09-25: What `supports_device_token_feed` reports.
    pub device_token_feed: bool,
    /// 2026-09-25: `event_query` answers "pending" this many times per event before the
    /// event completes (0 = complete at once).
    pub event_lag: usize,
    /// 2026-09-25: An event only completes once every event recorded after it has
    /// completed: the newest step is ready first.
    pub reversed_completion: bool,
    /// 2026-09-25: The n-th `event_query` (1-based, counted per model) fails with a
    /// device fault instead of answering.
    pub event_query_fault_at: Option<usize>,
}

impl Default for ModelCfg {
    fn default() -> Self {
        Self {
            vocab: 64,
            block_size: 16,
            total_blocks: 1_000,
            has_proposer: false,
            has_self_speculative: false,
            supports_beam: false,
            has_ssm_layers: false,
            ring_slots: 0,
            dflash_gamma: None,
            twophase_ok: false,
            can_batch_verify: true,
            slot_draft_capacity: 8,
            kv_exhaust_at: Vec::new(),
            verify_kv_exhaust_at: Vec::new(),
            chunk0_kv_refuse: None,
            reclaimable: 0,
            held_blocks: 0,
            device_token_feed: true,
            event_lag: 0,
            reversed_completion: false,
            event_query_fault_at: None,
        }
    }
}

#[derive(Default)]
pub(super) struct State {
    pub free_slots: Vec<usize>,
    pub next_slot: usize,
    pub used_blocks: usize,
    pub reclaimable: usize,
    /// 2026-09-25: Next token per row of the most recent forward.
    pub rows: Vec<u32>,
    pub decode_batch_calls: usize,
    /// 2026-10-14: `decode_verify_batched` calls so far (`verify_kv_exhaust_at`).
    pub verify_batched_calls: usize,
    /// 2026-10-15: chunk-0 refusals so far (`chunk0_kv_refuse`).
    pub chunk0_refused: usize,
    pub ticks: usize,
    /// 2026-09-25: What the last feed argmax wrote, per row.
    pub feed_cells: Vec<u32>,
    /// 2026-09-25: Events handed out so far; the first event id is 2, as the goldens
    /// record.
    pub events_created: u64,
    /// 2026-09-25: Recorded, not yet completed events, in record order, with the
    /// number of times each was queried.
    pub pending_events: Vec<(u64, usize)>,
    pub event_queries: usize,
}

/// 2026-09-25: A barrier the harness uses to order request arrival against the loop:
/// the scheduler thread parks in a model call until `release`, and the
/// harness can wait for the loop to reach a tick.
#[derive(Default)]
pub(super) struct Gate {
    state: Mutex<GateState>,
    cv: Condvar,
}

#[derive(Default)]
struct GateState {
    started: bool,
    ticks: usize,
    block_at_tick: Option<usize>,
    released: bool,
}

impl Gate {
    /// 2026-09-25: Called by the model on the scheduler thread (from `create_stream`)
    /// before its first tick.
    pub fn wait_start(&self) {
        let mut g = self.state.lock().unwrap();
        while !g.started {
            g = self.cv.wait(g).unwrap();
        }
    }
    pub fn start(&self) {
        self.state.lock().unwrap().started = true;
        self.cv.notify_all();
    }
    /// 2026-09-25: Called by the model once per tick. Parks when the harness asked to
    /// block at this tick, until [`Gate::release`].
    pub fn tick(&self) -> usize {
        let mut g = self.state.lock().unwrap();
        g.ticks += 1;
        let t = g.ticks;
        self.cv.notify_all();
        if g.block_at_tick == Some(t) {
            g.released = false;
            while !g.released {
                g = self.cv.wait(g).unwrap();
            }
        }
        t
    }
    pub fn block_at_tick(&self, t: usize) {
        self.state.lock().unwrap().block_at_tick = Some(t);
    }
    pub fn wait_tick(&self, t: usize) {
        let mut g = self.state.lock().unwrap();
        while g.ticks < t {
            g = self.cv.wait(g).unwrap();
        }
    }
    pub fn release(&self) {
        let mut g = self.state.lock().unwrap();
        g.released = true;
        g.block_at_tick = None;
        self.cv.notify_all();
    }
}

/// 2026-09-25: Everything the harness must still reach after the model has been moved
/// into the scheduler loop: the trace, the scripts, the gate.
#[derive(Default)]
pub(super) struct Shared {
    pub trace: Mutex<Vec<String>>,
    pub scripts: Mutex<HashMap<u64, SeqScript>>,
    pub cancel_flags: Mutex<HashMap<u64, Arc<AtomicBool>>>,
    pub state: Mutex<State>,
    pub gate: Gate,
}

pub(super) struct RecordingModel {
    pub cfg: ModelCfg,
    pub shared: Arc<Shared>,
}

impl std::ops::Deref for RecordingModel {
    type Target = Shared;
    fn deref(&self) -> &Shared {
        &self.shared
    }
}

impl Shared {
    pub fn add_script(&self, hash: u64, script: SeqScript) {
        self.scripts.lock().unwrap().insert(hash, script);
    }

    pub fn register_cancel(&self, hash: u64, flag: Arc<AtomicBool>) {
        self.cancel_flags.lock().unwrap().insert(hash, flag);
    }

    pub fn rec(&self, line: String) {
        self.trace.lock().unwrap().push(line);
    }

    pub fn take_trace(&self) -> Vec<String> {
        std::mem::take(&mut *self.trace.lock().unwrap())
    }
}

impl RecordingModel {
    pub fn new(cfg: ModelCfg) -> Self {
        let shared = Shared::default();
        shared.state.lock().unwrap().reclaimable = cfg.reclaimable;
        Self {
            cfg,
            shared: Arc::new(shared),
        }
    }

    /// 2026-09-25: `s<hash>@<slot>:len=<seq_len>` — how a sequence is spelled in a trace.
    pub fn seq_id(seq: &SequenceState) -> String {
        format!("s{}@{}:len={}", seq.session_hash, seq.slot_idx, seq.seq_len)
    }

    pub fn row_ptr(&self, row: usize) -> DevicePtr {
        DevicePtr(LOGITS_BASE + (row * self.cfg.vocab * 2) as u64)
    }

    /// 2026-09-25: (row, byte offset within the row) for a pointer into the logits slab.
    pub fn locate(&self, p: DevicePtr) -> (usize, usize) {
        let off = (p.0 - LOGITS_BASE) as usize;
        let row_bytes = self.cfg.vocab * 2;
        (off / row_bytes, off % row_bytes)
    }

    /// 2026-09-25: The scheduler stamps `session_hash` before prefill; a swapped-in
    /// sequence comes back without it, so the prompt's first token (also
    /// the request id) is the fallback key.
    fn script_for(&self, seq: &SequenceState) -> SeqScript {
        let key = if seq.session_hash != 0 {
            seq.session_hash
        } else {
            u64::from(seq.tokens.first().copied().unwrap_or(0))
        };
        self.scripts
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
            .unwrap_or_else(|| panic!("no script for sequence key {key}"))
    }

    /// 2026-09-25: The token the script generates at generated-index `g`.
    fn script_token(s: &SeqScript, g: usize) -> u32 {
        s.tokens
            .get(g)
            .or(s.tokens.last())
            .copied()
            .expect("scripts hold at least one token")
    }

    /// 2026-09-25: Output of a forward whose last input sits at absolute position
    /// `last_pos`: the script token after it. Position `p` holds generated
    /// token `p - prompt_len`, so the output is generated index
    /// `last_pos - prompt_len + 1`.
    pub fn out_after(&self, seq: &SequenceState, last_pos: usize) -> u32 {
        let s = self.script_for(seq);
        let g = (last_pos + 1).saturating_sub(s.prompt_len);
        Self::script_token(&s, g)
    }

    /// 2026-09-25: The drafter's proposal: the `n` script tokens after the one at absolute
    /// position `position` (the scheduler passes `seq_len`, where its
    /// not-yet-processed `last_token` goes), with `draft_wrong_every` applied.
    pub fn drafts_for(&self, seq: &SequenceState, position: usize, n: usize) -> Vec<u32> {
        let s = self.script_for(seq);
        let base = position.saturating_sub(s.prompt_len);
        (1..=n)
            .map(|j| {
                let t = Self::script_token(&s, base + j);
                match s.draft_wrong_every {
                    Some(m) if m > 0 && (base + j).is_multiple_of(m) => {
                        (t + 1) % (self.cfg.vocab as u32 - 1)
                    }
                    _ => t,
                }
            })
            .collect()
    }

    /// 2026-09-25: Verify a row of `k` inputs whose first input lands at absolute
    /// position `first_pos`: the argmax at each position. An input after the
    /// first that is not the scripted continuation makes its own output and
    /// every later one [`GARBAGE`].
    pub fn verify_row(&self, seq: &SequenceState, first_pos: usize, inputs: &[u32]) -> Vec<u32> {
        let s = self.script_for(seq);
        let mut out = Vec::with_capacity(inputs.len());
        let mut poisoned = false;
        for (t, &tok) in inputs.iter().enumerate() {
            let pos = first_pos + t;
            let g_in = pos.saturating_sub(s.prompt_len);
            if t > 0 && Self::script_token(&s, g_in) != tok {
                poisoned = true;
            }
            out.push(if poisoned {
                GARBAGE
            } else {
                Self::script_token(&s, g_in + 1)
            });
        }
        out
    }

    pub fn beam_hyp(&self, prompt_key: u64) -> Vec<u32> {
        self.scripts
            .lock()
            .unwrap()
            .get(&prompt_key)
            .and_then(|s| s.beam_hyp.clone())
            .unwrap_or_default()
    }

    /// 2026-09-25: Fire the scripted cancel when the request has `cancel_at` generated
    /// tokens in `seq.tokens`. Called from the decode and verify forwards.
    pub fn maybe_cancel(&self, seq: &SequenceState) {
        let s = self.script_for(seq);
        let g = seq.tokens.len().saturating_sub(s.prompt_len);
        if s.cancel_at == Some(g)
            && let Some(f) = self.cancel_flags.lock().unwrap().get(&seq.session_hash)
        {
            f.store(true, Ordering::Release);
        }
    }

    // 2026-09-25: slots and blocks.

    pub fn alloc_slot(&self) -> usize {
        let mut st = self.state.lock().unwrap();
        if let Some(s) = st.free_slots.pop() {
            return s;
        }
        let s = st.next_slot;
        st.next_slot += 1;
        s
    }

    pub fn release_slot(&self, slot: usize) {
        if slot == usize::MAX {
            return;
        }
        let mut st = self.state.lock().unwrap();
        st.free_slots.push(slot);
        st.free_slots.sort_unstable_by(|a, b| b.cmp(a));
    }

    pub fn claim_slot(&self, slot: usize) {
        let mut st = self.state.lock().unwrap();
        st.free_slots.retain(|&s| s != slot);
    }

    /// 2026-09-25: `tokens / block_size + 1` blocks: enough for `tokens` tokens and the
    /// next position, as the engine sizes a decode's blocks.
    pub fn blocks_for(&self, tokens: usize) -> usize {
        tokens / self.cfg.block_size + 1
    }

    pub fn take_blocks(&self, n: usize) {
        self.state.lock().unwrap().used_blocks += n;
    }

    pub fn give_blocks(&self, n: usize) {
        let mut st = self.state.lock().unwrap();
        st.used_blocks = st.used_blocks.saturating_sub(n);
    }

    pub fn free_blocks(&self) -> usize {
        let used = self.state.lock().unwrap().used_blocks + self.cfg.held_blocks;
        self.cfg.total_blocks.saturating_sub(used)
    }

    /// 2026-09-25: Grow `seq.block_table` to cover `seq.seq_len` tokens, drawing from
    /// the free pool.
    pub fn grow_blocks(&self, seq: &mut SequenceState) {
        let want = self.blocks_for(seq.seq_len);
        let have = seq.block_table.len();
        if want > have {
            self.take_blocks(want - have);
            seq.block_table.extend((have..want).map(|b| b as u32));
        }
    }

    pub fn next_decode_batch_call(&self) -> usize {
        let mut st = self.state.lock().unwrap();
        let c = st.decode_batch_calls;
        st.decode_batch_calls += 1;
        c
    }

    /// 2026-10-15 (race-prefill, A194): whether this chunk-0 `prefill_chunk` of `session`
    /// is refused (`chunk0_kv_refuse`); counts the refusal.
    pub fn refuse_chunk0(&self, session: u64, start: usize) -> bool {
        let Some((s, n)) = self.cfg.chunk0_kv_refuse else { return false };
        let mut st = self.state.lock().unwrap();
        if s != session || start != 0 || st.chunk0_refused >= n {
            return false;
        }
        st.chunk0_refused += 1;
        true
    }

    /// 2026-10-14 (race-memory, A184): the index of this `decode_verify_batched` call.
    pub fn next_verify_batched_call(&self) -> usize {
        let mut st = self.state.lock().unwrap();
        let c = st.verify_batched_calls;
        st.verify_batched_calls += 1;
        c
    }

    /// 2026-09-25: The snapshot publish reads `ssm_snapshot_occupancy` twice per tick;
    /// the first read of each pair marks the tick at the gate, which parks it
    /// if the harness asked to block there.
    pub fn snapshot_tick(&self) -> Option<usize> {
        let mut st = self.state.lock().unwrap();
        st.ticks += 1;
        let first_of_pair = st.ticks % 2 == 1;
        drop(st);
        first_of_pair.then(|| self.gate.tick())
    }

    pub fn reclaim(&self, num_blocks: usize) -> usize {
        let mut st = self.state.lock().unwrap();
        let got = num_blocks.min(st.reclaimable);
        st.reclaimable -= got;
        st.used_blocks = st.used_blocks.saturating_sub(got);
        got
    }

    pub fn set_rows(&self, rows: Vec<u32>) {
        self.state.lock().unwrap().rows = rows;
    }

    pub fn row(&self, r: usize) -> u32 {
        let st = self.state.lock().unwrap();
        st.rows.get(r).copied().unwrap_or_else(|| {
            panic!(
                "logits row {r} read, but the last forward produced {}",
                st.rows.len()
            )
        })
    }

    pub fn rows(&self) -> Vec<u32> {
        self.state.lock().unwrap().rows.clone()
    }

    /// 2026-09-25: BF16 bytes for `n` bytes of the slab starting at (row, byte_off): the
    /// scripted token of that row is 30.0, everything else 0.0.
    pub fn fill_slab(&self, row: usize, byte_off: usize, dst: &mut [u8]) {
        let vocab = self.cfg.vocab;
        let row_bytes = vocab * 2;
        for (i, b) in dst.iter_mut().enumerate() {
            let abs = byte_off + i;
            let r = row + abs / row_bytes;
            let within = abs % row_bytes;
            let tok = (within / 2) as u32;
            let hi_byte = within % 2 == 1;
            let hit = tok == self.row(r);
            // 2026-09-25: 30.0 in bf16 is 0x41F0; 0.0 is 0x0000. Little-endian: lo, hi.
            // A 30-logit margin leaves the other tokens about 1e-17 of the
            // mass at temperature 0.7 (`host_logits_paths`).
            *b = if hit && hi_byte {
                0x41
            } else if hit {
                0xF0
            } else {
                0
            };
        }
    }
}

pub(super) fn fmt_ids(ids: &[u32]) -> String {
    format!("{ids:?}")
}
