// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched verify: `n` sequences with `ks[i]` rows each in one forward pass.
//!
//! The `R = Σ ks` rows are seq-major: sequence `i` owns rows
//! `[off_i, off_i + ks[i])`. Every weight-bearing op runs once over all R
//! rows; attention runs through `decode_multi_seq` with per-row block tables
//! and lengths, and every other layer through `decode_verify_multi`.
//!
//! The layer loop, final norm, LM head and argmax are captured as one CUDA
//! graph per `verify_batched_graph_key`. Embeds, KV-block allocation, the
//! metadata, block-table and WY-table uploads, and the argmax readback run
//! outside the graph; the uploads go to fixed addresses before every replay.
//! `METRALE_NO_MTP_VERIFY_GRAPHS` (presence) or `METRALE_K4_DIAG=1` makes the
//! forward eager.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - `tokens` and `seq_len` of the sequences change only when
//!   `decode_verify_batched_dispatch` returns `Ok`. An `Err` can leave KV
//!   blocks allocated and recurrent state written.
//! - After a call, `gdn_woa_eligible` is true only if that call returned `Ok`
//!   under a write-on-accept request with the WY tables staged.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, bail, ensure};

/// 2026-09-25: Byte offsets of the batched-verify metadata from `meta_base`,
/// all derived from `R = verify_e2::VERIFY_ROW_CAP`:
///   positions  u32 × R at [0, 4R)
///   seq_slot   u32 × R at [4R, 8R)
///   slots      i64 × R at [8R, 16R)
///   seq_lens   i32 × R at [16R, 20R)
///   bt         i32 × R × max_blocks from 24R
const VMETA_R: usize = super::verify_e2::VERIFY_ROW_CAP;
const VMETA_SEQ_SLOT: usize = VMETA_R * 4;
const VMETA_SLOTS: usize = VMETA_R * 8;
const VMETA_SEQ_LENS: usize = VMETA_R * 16;
const VMETA_BT: usize = VMETA_R * 24;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState};
use metrale_model_layers::layers::ops;

mod eager;
mod graphs;
mod readback;
mod stage;

impl TransformerModel {
    /// 2026-09-25: Whether `decode_verify_batched_dispatch` can take `ks.len()`
    /// sequences with `ks[i]` rows each. The scheduler verifies the sequences
    /// one at a time when it returns false.
    ///
    /// It requires `2 <= n <= VERIFY_WY_TABLE_SEQS`, `Σ ks <= VERIFY_ROW_CAP`,
    /// no comm backend unless `batched_verify_ep_ok`, a verify hidden stash (allocated only with a
    /// proposer), no layer that declines `decode_verify_multi`, no HSS
    /// (`cache_blocks_per_seq`), and, with an adapter loaded, no
    /// `METRALE_LORA_NO_BATCH_VERIFY=1`. Without DFlash every `ks[i]` must be
    /// in 2..=4. With DFlash the batch must be uniform with k in
    /// 2..=`dflash_kgamma`, and `n * dflash_kgamma` must fit
    /// `dflash_hidden_save_rows`.
    pub(super) fn can_batch_verify_dispatch(&self, ks: &[usize]) -> bool {
        let n = ks.len();
        let shape_ok = if self.dflash_hidden_save.is_some() {
            // 2026-09-25: DFlash: one k for the whole batch, any value a
            // capture band holds, and a full band per sequence.
            self.dflash_kgamma >= 2
                && ks.iter().all(|&k| (2..=self.dflash_kgamma).contains(&k))
                && ks.iter().all(|&k| k == ks[0])
                && n * self.dflash_kgamma <= self.dflash_hidden_save_rows
        } else {
            ks.iter().all(|k| (2..=4).contains(k))
        };
        (2..=metrale_model_layers::layer::VERIFY_WY_TABLE_SEQS).contains(&n)
            && shape_ok
            && ks.iter().sum::<usize>() <= super::verify_e2::VERIFY_ROW_CAP
            // 2026-10-02: With a comm backend only through the multi-rank batched verify
            // (`verify_ep.rs`): EP protocol v2, every layer `decode_verify_multi_own_states`.
            && (self.comm.is_none() || self.batched_verify_ep_ok())
            && !(self.lora.is_some() && metrale_model_layers::lora::no_batch_verify())
            && !self.verify_hidden_stash.is_null()
            && !self
                .layers
                .iter()
                .any(|l| l.decode_verify_multi_unsupported())
            // 2026-10-02: A GLM-5.3 layer takes as many rows as its KDA and MLP scratch hold.
            && self.layers.iter().all(|l| {
                !l.decode_verify_multi_own_states()
                    || ks.iter().sum::<usize>() <= l.decode_verify_multi_max_rows()
            })
            // 2026-09-25: HSS: `decode_multi_seq` takes no disk block ids, so
            // history offloaded to disk would be missing.
            && self
                .kv_cache
                .lock()
                .config()
                .cache_blocks_per_seq
                .is_none()
    }

    /// 2026-09-25: Verify `n = seqs.len()` sequences in one forward over
    /// `R = Σ ks` rows. `tokens` is flat and seq-major: row `off_i + j` is
    /// sequence `i`'s token `j`. Returns the argmax token of each row.
    ///
    /// On `Ok` each sequence's `tokens` gains its `ks[i]` tokens and `seq_len`
    /// grows by `ks[i]`; rolling back rejected rows is the caller's job. On
    /// `Err` neither changes. The R logits rows stay in `buffers.logits()`
    /// until the next forward overwrites them.
    pub(super) fn decode_verify_batched_dispatch(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
        opts: crate::traits::VerifyBatchedOpts,
    ) -> Result<Vec<u32>> {
        // 2026-09-25: Nothing is foldable until this call says so at its end:
        // an Err anywhere below leaves the fold declined and the host restore
        // on.
        self.gdn_woa_eligible
            .store(false, std::sync::atomic::Ordering::Release);
        let t_launch = std::time::Instant::now();
        let mapped_argmax = mapped_argmax_host_dev(self.gpu.as_ref());
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n = seqs.len();
        // 2026-09-25: Sequence i owns rows [off[i], off[i+1]).
        let mut off: Vec<usize> = Vec::with_capacity(n + 1);
        let mut acc = 0usize;
        for &k in ks {
            off.push(acc);
            acc += k;
        }
        off.push(acc);
        let r_total = acc;
        let k_max = ks.iter().copied().max().unwrap_or(0);
        // 2026-09-25: Re-checks the row-count range `can_batch_verify_dispatch`
        // admits: 2..=4 without DFlash, 2..=dflash_kgamma with it. DFlash
        // uniformity is not re-checked here.
        let dflash_k = self
            .dflash_hidden_save
            .is_some()
            .then_some(self.dflash_kgamma);
        ensure!(
            n >= 2
                && ks.len() == n
                && ks
                    .iter()
                    .all(|&k| dflash_k.map_or((2..=4).contains(&k), |g| (2..=g).contains(&k)))
                && tokens.len() == r_total,
            "batched verify: n={n} ks={ks:?} tokens={} dflash_k={dflash_k:?}",
            tokens.len()
        );
        // 2026-09-25: R ≤ VERIFY_ROW_CAP bounds the VMETA_* layout above and
        // the block-table staging `sizes.rs` sizes for 160 rows (`bt_rows`).
        // The logits arena there holds
        // `min(max_batch_tokens, max(160, decode rows + 1))` rows
        // (`logits_tokens`).
        ensure!(
            r_total <= super::verify_e2::VERIFY_ROW_CAP,
            "batched verify: R={r_total} exceeds the {}-row buffer capacity",
            super::verify_e2::VERIFY_ROW_CAP
        );

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        for (r, &t) in tokens.iter().enumerate() {
            self.embed(t, hidden.offset(r * h * bf16), stream)?;
        }

        let bs = kv_cache.block_size();
        for (i, seq) in seqs.iter_mut().enumerate() {
            let last_pos = seq.seq_len + ks[i] - 1;
            ensure_blocks_through_decode(
                seq,
                last_pos / bs,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;
        }

        // 2026-09-25: METRALE_K4_DIAG=1 synchronises the stream after every
        // layer so a fault names its layer. It also turns graphs off
        // (`graphs_on` below), because a synchronise cannot be captured.
        let k4_diag = super::verify_e2::k4_diag_enabled();

        // 2026-09-25: Stage the per-GDN-layer WY pointer tables before any
        // replay, at the batch's widest row count `k_max`. Table strides do
        // not depend on k (`VERIFY_WY_LAYER_STRIDE_BYTES`). NULL means no
        // tables were staged, and the layers get a NULL slice below.
        // 2026-10-02: A layer that owns its verify state (`decode_verify_multi_own_states`,
        // GLM-5.3) reads no tables: none are staged, so no carry, write-on-accept or graph.
        let own_states = self.any_verify_own_states();
        let wy_tables_base = if own_states {
            DevicePtr::NULL
        } else {
            self.upload_verify_wy_tables(&*seqs, k_max, &[], stream)?
        };
        // 2026-09-25: A write-on-accept request is honoured only with the WY
        // tables staged, since the fold reads them. `gdn_woa_bind` allocates
        // and binds the stash on the first request, before any capture.
        // 2026-09-26: The carried-state verify takes the request first
        // (`gdn_carry.rs`); DFlash and a verify without tables never carry.
        let carry = self.gdn_carry_begin(
            opts.write_on_accept && dflash_k.is_none() && !wy_tables_base.is_null(),
            &*seqs,
            ks,
            stream,
        )?;
        let write_on_accept = !carry
            && opts.write_on_accept
            && ks.iter().all(|&k| k == k_max)
            && !wy_tables_base.is_null()
            && self.gdn_woa_bind()?;

        // 2026-09-25: Graph decision. An exact key hit replays. On a miss,
        // with `graph_borrow_enabled()`, `find_borrowable_verify_key` may pick
        // a wider captured key whose first `(slot, k)` pairs equal this
        // batch's, so the active rows keep their `off[i]`. Its extra pairs
        // become ghost rows this step must feed: pad metadata, pad embeds and
        // WY entries. Each ghost slot must be free and, with WY tables, its
        // intermediate pool must cover the ghost's depth (the closure in
        // `pick_verify_graph`).
        let graphs_on = super::verify_e2::verify_graphs_enabled() && !k4_diag && !own_states;
        let graph_key = if graphs_on {
            self.verify_batched_graph_key(
                &*seqs,
                ks,
                wy_tables_base.is_null(),
                write_on_accept || carry,
            )
        } else {
            None
        };
        let mut graphs = graph_key
            .as_ref()
            .map(|_| self.verify_batched_graphs.lock());
        let (replay, ghosts, mut outcome) =
            self.pick_verify_graph(&mut graphs, &graph_key, wy_tables_base, r_total, n);
        // 2026-09-25: Rows this step must prepare: active rows plus any ghost
        // rows. `r_up <= VERIFY_ROW_CAP` holds on every path: without ghosts
        // by the `ensure!` above, with ghosts by the borrow veto.
        let r_ghost: usize = ghosts.iter().map(|&(_, k)| k as usize).sum();
        let r_up = r_total + r_ghost;
        if !ghosts.is_empty() {
            // 2026-09-25: Ghost rows embed token 0 so they hold finite values;
            // only rows 0..r_total are read back.
            for r in r_total..r_up {
                self.embed(0, hidden.offset(r * h * bf16), stream)?;
            }
            // 2026-09-25: Re-stage the WY tables with ghost entries appended,
            // built from the SSM pool (`upload_verify_wy_tables`).
            let k_ghost = ghosts.iter().map(|&(_, k)| k as usize).max().unwrap_or(0);
            let restaged =
                self.upload_verify_wy_tables(&*seqs, k_max.max(k_ghost), &ghosts, stream)?;
            // 2026-09-25: A change in table presence would replay the graph
            // against tables without its ghost entries.
            ensure!(
                restaged.is_null() == wy_tables_base.is_null(),
                "verify graph borrow: ghost WY restage flipped table presence"
            );
        }

        if carry {
            self.gdn_carry_stage(&*seqs, &ghosts, stream)?;
        }
        let metadata = self.stage_verify_metadata(&*seqs, ks, &off, bs, r_total, r_up, stream)?;

        if let Some(graph) = replay {
            // 2026-09-25: The graph reads this step's metadata and WY tables
            // from the fixed addresses refreshed above.
            if graph.0 != 0 {
                self.gpu.launch_graph(graph, stream)?;
            }
        } else {
            // 2026-09-25: No graph to replay: run the forward, under capture
            // when graphs are on. A full cache still captures: the insert
            // below evicts the least recently used entry.
            let capture = graphs.is_some();

            let ctx = ForwardContext {
                buffers: &self.buffers,
                hc_row_offset: 0,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                dispatch: &self.dispatch,
                moe_lora_route: self.decode_moe_route(),
                derived: &self.derived,
                levers: &self.levers,
                stats: &self.stats,
                attn_metadata: Some(metadata),
                profile: false,
                comm: self.comm_ref(),
                graph_capture: capture,
                decode_step: false,
                gdn_exact_replay: false,
                gdn_write_on_accept: write_on_accept || carry,
                token_ids: None,
                host_token_ids: None,
                routed_lora_layers: None,
                midchunk_capture: None,
            };

            let mut seq_lens_vec: Vec<usize> = Vec::with_capacity(r_total);
            let mut block_tables_vec: Vec<Vec<u32>> = Vec::with_capacity(r_total);
            for (i, seq) in seqs.iter().enumerate() {
                for j in 0..ks[i] {
                    seq_lens_vec.push(seq.seq_len + j);
                    block_tables_vec.push(seq.block_table.clone());
                }
            }

            // 2026-09-25: Attention layer states for `decode_multi_seq`,
            // allocated before `begin_capture`.
            let mut attn_dummy_states: Vec<Vec<Box<dyn LayerState>>> = Vec::new();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(layer_idx) == LayerType::FullAttention
                    && layer.decode_verify_multi_own_states()
                {
                    // 2026-10-02: Keeps `attn_idx` aligned; this layer never reads it.
                    attn_dummy_states.push(Vec::new());
                } else if self.config.layer_type(layer_idx) == LayerType::FullAttention {
                    attn_dummy_states.push(
                        (0..r_total)
                            .map(|_| layer.alloc_state(self.gpu.as_ref()))
                            .collect::<Result<_>>()?,
                    );
                }
            }

            if capture {
                self.gpu.begin_capture(stream)?;
            }

            self.run_verify_layers(
                &mut attn_dummy_states,
                seqs,
                &mut kv_cache,
                &seq_lens_vec,
                &block_tables_vec,
                hidden,
                residual,
                r_total,
                n,
                ks,
                &off,
                wy_tables_base,
                k4_diag,
                &ctx,
                stream,
            )?;

            let normed = self.buffers.norm_output();
            self.final_norm_apply(
                hidden,
                normed,
                r_total as u32,
                h as u32,
                self.config.rms_norm_eps as f32,
                stream,
            )?;

            if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
                anyhow::bail!("K4_DIAG(batched): CUDA error after final norm: {e:#}");
            }

            self.lm_head_batched(normed, r_total as u32, self.buffers.logits(), stream)?;

            if k4_diag && let Err(e) = self.gpu.synchronize(stream) {
                anyhow::bail!("K4_DIAG(batched): CUDA error after lm_head_batched: {e:#}");
            }

            self.verify_rows_argmax(mapped_argmax, r_total, bf16, stream)?;

            if capture {
                self.finish_verify_capture(&mut graphs, graph_key, &mut outcome, ks, n, stream)?;
            }
        }
        // 2026-09-25: The live key count, read while the lock is held, is
        // reported beside the outcome.
        let live_keys = graphs.as_ref().map(|g| g.0.len()).unwrap_or(0);
        drop(graphs);
        super::verify_e2::record_verify_graph_outcome(n, live_keys, outcome);

        // 2026-09-25: The argmax rows are in the mapped blob or at scratch
        // offset 0. `METRALE_MTP_TIMING=1` reports the time before this point
        // (launch) and after it (wait plus copy) separately.
        let t_d2h = std::time::Instant::now();
        let launch_us = t_d2h.duration_since(t_launch).as_micros() as u64;
        let buf = self.read_verify_argmax(mapped_argmax, r_total, stream)?;
        if carry {
            self.gdn_carry_end(&*seqs, ks, stream)?;
        }
        {
            // 2026-09-25: `METRALE_MTP_TIMING=1`: sum the launch and readback
            // times and log their per-call means once per 100 batched
            // verifies. It lives here because the scheduler's `mtp_timing` is
            // in the server crate.
            use std::sync::atomic::{AtomicU64, Ordering};
            static LAUNCH_US: AtomicU64 = AtomicU64::new(0);
            static D2H_US: AtomicU64 = AtomicU64::new(0);
            static N: AtomicU64 = AtomicU64::new(0);
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *ON.get_or_init(|| std::env::var("METRALE_MTP_TIMING").as_deref() == Ok("1")) {
                let d2h_us = t_d2h.elapsed().as_micros() as u64;
                LAUNCH_US.fetch_add(launch_us, Ordering::Relaxed);
                D2H_US.fetch_add(d2h_us, Ordering::Relaxed);
                let n_done = N.fetch_add(1, Ordering::Relaxed) + 1;
                if n_done.is_multiple_of(100) {
                    let l = LAUNCH_US.swap(0, Ordering::Relaxed);
                    let d = D2H_US.swap(0, Ordering::Relaxed);
                    tracing::info!(
                        "batched-verify fwd split [100 calls]: launch={:.2}ms d2h_wait={:.2}ms",
                        l as f64 / 100_000.0,
                        d as f64 / 100_000.0,
                    );
                }
            }
        }

        let mut out = Vec::with_capacity(r_total);
        for r in 0..r_total {
            let o = r * 4;
            out.push(u32::from_le_bytes([
                buf[o],
                buf[o + 1],
                buf[o + 2],
                buf[o + 3],
            ]));
        }

        for (i, seq) in seqs.iter_mut().enumerate() {
            for &t in &tokens[off[i]..off[i + 1]] {
                seq.tokens.push(t);
            }
            seq.seq_len += ks[i];
        }

        // 2026-09-25: The verify completed: the fold may run once if this was a
        // write-on-accept request.
        self.gdn_woa_eligible
            .store(write_on_accept, std::sync::atomic::Ordering::Release);
        Ok(out)
    }
}

/// 2026-09-25: A 65_536-byte page-locked host blob and its device alias
/// (`host_ptr_to_device`) for the mapped argmax. After the first success the
/// same pair is returned for the rest of the process, so a captured graph
/// keeps a valid address. Returns `None` when `METRALE_NO_MAPPED_ARGMAX=1`, or
/// when the allocation or the mapping fails; callers then use scratch and a
/// copy.
fn mapped_argmax_host_dev(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
) -> Option<(*mut u8, metrale_gpu_runtime::gpu::DevicePtr)> {
    use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
    static HOST: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
    static DEV: AtomicU64 = AtomicU64::new(0);
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var("METRALE_NO_MAPPED_ARGMAX").as_deref() == Ok("1")) {
        return None;
    }
    let mut h = HOST.load(Ordering::Acquire);
    if h.is_null() {
        // 2026-09-25: Only the scheduler thread calls this. A failure stores
        // nothing, so the next call tries again.
        h = gpu.alloc_host_pinned(65_536).ok()?;
        let d = gpu.host_ptr_to_device(h).ok()?;
        DEV.store(d.0, Ordering::Release);
        HOST.store(h, Ordering::Release);
    }
    let d = DEV.load(Ordering::Acquire);
    if d == 0 {
        return None;
    }
    Some((h, metrale_gpu_runtime::gpu::DevicePtr(d)))
}
