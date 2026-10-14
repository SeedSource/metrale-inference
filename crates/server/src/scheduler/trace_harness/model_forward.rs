// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The recording fake's forward-pass and bookkeeping bodies.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! Each records a trace line and applies its effect to the host-side
//! `SequenceState` (`tokens`, `seq_len`, `block_table`, `slot_idx`) and to
//! the fake's slot and block pools.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{
    MixedBatchResult, MixedForwardResult, PrefillSlice, SequenceState,
};

use super::model::{RecordingModel, fmt_ids};

macro_rules! rec {
    ($m:expr, $($arg:tt)*) => { $m.rec(format!($($arg)*)) };
}

pub(super) fn sid(s: &SequenceState) -> String {
    RecordingModel::seq_id(s)
}

impl RecordingModel {
    fn push_all(seq: &mut SequenceState, toks: &[u32]) {
        seq.tokens.extend_from_slice(toks);
        seq.seq_len += toks.len();
    }
    fn prefill_into(&self, seq: &mut SequenceState, toks: &[u32], start: usize, len: usize) {
        seq.tokens.extend_from_slice(&toks[start..start + len]);
        seq.seq_len = start + len;
        if start == 0 {
            seq.prompt_len = toks.len();
        }
        self.grow_blocks(seq);
    }
    /// 2026-09-25: One decode-row forward: push the input, produce the next token.
    fn decode_row(&self, seq: &mut SequenceState, tok: u32) -> u32 {
        Self::push_all(seq, &[tok]);
        self.grow_blocks(seq);
        self.out_after(seq, seq.seq_len - 1)
    }
    fn prefill_row(&self, sl: &mut PrefillSlice<'_>) -> u32 {
        self.prefill_into(sl.seq, sl.prompt_tokens, sl.chunk_start, sl.chunk_len);
        self.out_after(sl.seq, sl.seq.seq_len - 1)
    }
    fn slice_desc(sl: &PrefillSlice<'_>) -> String {
        let (a, b, c) = (sl.chunk_start, sl.chunk_len, sl.is_last_chunk);
        format!("{} start={a} len={b} last={c}", sid(sl.seq))
    }

    pub(super) fn verify_k(&self, name: &str, toks: &[u32], seq: &mut SequenceState) -> Vec<u32> {
        self.maybe_cancel(seq);
        let out = self.verify_row(seq, seq.seq_len, toks);
        Self::push_all(seq, toks);
        self.grow_blocks(seq);
        self.set_rows(out.clone());
        rec!(
            self,
            "{name}(tokens={}, {}) -> {}",
            fmt_ids(toks),
            sid(seq),
            fmt_ids(&out)
        );
        out
    }
    pub(super) fn alloc(&self, what: String) -> Result<SequenceState> {
        let slot = self.alloc_slot();
        rec!(self, "{what} -> slot {slot}");
        Ok(SequenceState::host_only(slot))
    }
    pub(super) fn fx_prefill(&self, tokens: &[u32], seq: &mut SequenceState, st: u64) -> DevicePtr {
        self.prefill_into(seq, tokens, 0, tokens.len());
        self.set_rows(vec![self.out_after(seq, seq.seq_len - 1)]);
        rec!(
            self,
            "prefill(n={}, {}, stream={st})",
            tokens.len(),
            sid(seq)
        );
        self.row_ptr(0)
    }
    pub(super) fn fx_prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        start: usize,
        len: usize,
        last: bool,
        st: u64,
    ) -> DevicePtr {
        self.prefill_into(seq, tokens, start, len);
        self.set_rows(vec![self.out_after(seq, seq.seq_len - 1)]);
        let id = sid(seq);
        rec!(
            self,
            "prefill_chunk(start={start}, len={len}, last={last}, {id}, stream={st})"
        );
        self.row_ptr(0)
    }
    pub(super) fn fx_prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk: usize,
        st: u64,
    ) -> Result<DevicePtr> {
        let (n, id) = (tokens.len(), sid(seq));
        if !self.cfg.twophase_ok {
            rec!(
                self,
                "prefill_twophase(n={n}, chunk={chunk}, {id}) -> ERR unsupported"
            );
            anyhow::bail!("two-phase prefill unsupported (scripted)");
        }
        self.prefill_into(seq, tokens, 0, tokens.len());
        self.set_rows(vec![self.out_after(seq, seq.seq_len - 1)]);
        rec!(
            self,
            "prefill_twophase(n={n}, chunk={chunk}, {id}, stream={st})"
        );
        Ok(self.row_ptr(0))
    }
    pub(super) fn fx_decode(
        &self,
        name: &str,
        token: u32,
        seq: &mut SequenceState,
        st: u64,
    ) -> DevicePtr {
        if name == "decode" {
            self.maybe_cancel(seq);
        }
        let out = self.decode_row(seq, token);
        self.set_rows(vec![out]);
        rec!(self, "{name}(token={token}, {}, stream={st})", sid(seq));
        self.row_ptr(0)
    }
    pub(super) fn fx_decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        st: u64,
    ) -> Result<DevicePtr> {
        let call = self.next_decode_batch_call();
        let ids: Vec<String> = seqs.iter().map(|s| sid(s)).collect();
        let head = format!(
            "decode_batch(tokens={}, seqs={ids:?}, stream={st})",
            fmt_ids(tokens)
        );
        if self.cfg.kv_exhaust_at.contains(&call) {
            rec!(self, "{head} -> ERR KV cache exhausted");
            anyhow::bail!("KV cache exhausted (scripted at decode_batch #{call})");
        }
        let mut rows = Vec::with_capacity(seqs.len());
        for (s, &t) in seqs.iter_mut().zip(tokens) {
            self.maybe_cancel(s);
            rows.push(self.decode_row(s, t));
        }
        self.set_rows(rows);
        self.rec(head);
        Ok(self.row_ptr(0))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fx_mixed_forward(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        p: &mut SequenceState,
        start: usize,
        len: usize,
        last: bool,
        st: u64,
    ) -> MixedForwardResult {
        let mut rows: Vec<u32> = decode_seqs
            .iter_mut()
            .zip(decode_tokens)
            .map(|(s, &t)| self.decode_row(s, t))
            .collect();
        self.prefill_into(p, prefill_tokens, start, len);
        rows.push(self.out_after(p, p.seq_len - 1));
        let n = decode_seqs.len();
        self.set_rows(rows);
        let (d, id) = (fmt_ids(decode_tokens), sid(p));
        rec!(
            self,
            "mixed_forward(decode={d}, prefill={id} start={start} len={len} last={last}, stream={st})"
        );
        MixedForwardResult {
            decode_logits: self.row_ptr(0),
            prefill_logits: self.row_ptr(n),
        }
    }
    pub(super) fn fx_prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        st: u64,
        row_base: Option<usize>,
    ) -> Vec<DevicePtr> {
        let base = row_base.unwrap_or(0);
        let mut rows = self.rows();
        rows.truncate(base);
        rows.extend(streams.iter_mut().map(|sl| self.prefill_row(sl)));
        let desc: Vec<String> = streams.iter().map(Self::slice_desc).collect();
        let n = streams.len();
        self.set_rows(rows);
        match row_base {
            None => rec!(self, "prefill_batch_chunk(streams={desc:?}, stream={st})"),
            Some(b) => rec!(
                self,
                "prefill_batch_chunk_rows(streams={desc:?}, stream={st}, row_base={b})"
            ),
        }
        (0..n).map(|r| self.row_ptr(base + r)).collect()
    }
    pub(super) fn fx_mixed_forward_batch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefills: &mut [PrefillSlice<'_>],
        st: u64,
    ) -> MixedBatchResult {
        let mut rows: Vec<u32> = decode_seqs
            .iter_mut()
            .zip(decode_tokens)
            .map(|(s, &t)| self.decode_row(s, t))
            .collect();
        let n = rows.len();
        rows.extend(prefills.iter_mut().map(|sl| self.prefill_row(sl)));
        let desc: Vec<String> = prefills.iter().map(Self::slice_desc).collect();
        let m = prefills.len();
        self.set_rows(rows);
        let d = fmt_ids(decode_tokens);
        rec!(
            self,
            "mixed_forward_batch(decode={d}, prefills={desc:?}, stream={st})"
        );
        MixedBatchResult {
            decode_logits: self.row_ptr(0),
            prefill_logits: (0..m).map(|r| self.row_ptr(n + r)).collect(),
        }
    }
    pub(super) fn fx_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        st: u64,
        woa: bool,
    ) -> Result<Vec<u32>> {
        let call = self.next_verify_batched_call();
        if self.cfg.verify_kv_exhaust_at.contains(&call) {
            let ids: Vec<String> = seqs.iter().map(|s| sid(s)).collect();
            let t = fmt_ids(tokens);
            rec!(
                self,
                "decode_verify_batched(tokens={t}, ks={ks:?}, seqs={ids:?}, stream={st}, woa={woa}) -> ERR KV cache exhausted"
            );
            anyhow::bail!("KV cache exhausted (scripted at decode_verify_batched #{call})");
        }
        let mut out = Vec::with_capacity(tokens.len());
        let mut off = 0usize;
        let mut ids = Vec::new();
        for (s, &k) in seqs.iter_mut().zip(ks) {
            let row = &tokens[off..off + k];
            out.extend(self.verify_row(s, s.seq_len, row));
            Self::push_all(s, row);
            self.grow_blocks(s);
            ids.push(sid(s));
            off += k;
        }
        self.set_rows(out.clone());
        let (t, o) = (fmt_ids(tokens), fmt_ids(&out));
        rec!(
            self,
            "decode_verify_batched(tokens={t}, ks={ks:?}, seqs={ids:?}, stream={st}, woa={woa}) -> {o}"
        );
        Ok(out)
    }
    pub(super) fn fx_propose_batched(
        &self,
        tokens: &[u32],
        positions: &[usize],
        stash: &[usize],
        n: usize,
        seqs: &mut [&mut SequenceState],
        st: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<Vec<u32>> {
        let drafts: Vec<Vec<u32>> = seqs
            .iter()
            .zip(positions)
            .map(|(s, &p)| self.drafts_for(s, p, n))
            .collect();
        if let Some(c) = out_conf {
            *c = drafts.iter().map(|d| vec![0.0; d.len()]).collect();
        }
        let t = fmt_ids(tokens);
        rec!(
            self,
            "run_mtp_propose_batched(tokens={t}, positions={positions:?}, stash={stash:?}, n={n}, stream={st}) -> {drafts:?}"
        );
        drafts
    }
    pub(super) fn fx_propose_multi(
        &self,
        token: u32,
        position: usize,
        n: usize,
        seq: &mut SequenceState,
        st: u64,
        masked: bool,
    ) -> Vec<u32> {
        let d = self.drafts_for(seq, position, n);
        let (id, ds) = (sid(seq), fmt_ids(&d));
        rec!(
            self,
            "run_mtp_propose_multi(token={token}, pos={position}, n={n}, {id}, stream={st}, mask={masked}) -> {ds}"
        );
        d
    }
    pub(super) fn fx_free_sequence(&self, seq: &mut SequenceState) {
        rec!(
            self,
            "free_sequence({}, blocks={})",
            sid(seq),
            seq.block_table.len()
        );
        self.give_blocks(seq.block_table.len());
        seq.block_table.clear();
        self.release_slot(seq.slot_idx);
    }
    pub(super) fn fx_compact(&self, seq: &mut SequenceState, new_slot: usize) {
        rec!(self, "compact_sequence({}, new_slot={new_slot})", sid(seq));
        if seq.slot_idx != new_slot {
            self.claim_slot(new_slot);
            self.release_slot(seq.slot_idx);
            seq.slot_idx = new_slot;
        }
    }
    pub(super) fn fx_restore(&self, seq: &mut SequenceState, num_blocks: usize) {
        self.take_blocks(num_blocks);
        seq.block_table = (0..num_blocks as u32).collect();
        rec!(
            self,
            "restore_sequence_state({}, blocks={num_blocks})",
            sid(seq)
        );
    }
}
