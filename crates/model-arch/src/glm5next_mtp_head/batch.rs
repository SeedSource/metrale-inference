// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The batched GLM drafter (`METRALE_GLM_MTP_BATCH_DRAFT=1`) and the
//! per-sequence drafter KV pool it needs (`METRALE_GLM_MTP_SEQ_KV=1`).
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - Both levers default off; off, `new` allocates no scratch, `alloc_state` no pool, and
//!   `propose_batch` answers `Ok(None)`, so every propose runs `propose` as before.
//! - `propose_batch` runs draft step `d` of all `n` sequences as one pass: the two concat
//!   norms per row (the embedding rows are not contiguous), then `eh_proj`, the block
//!   (`Glm5NextLayer::decode_rows_for_drafter`: DSA per sequence, everything else over the `n`
//!   rows), `shared_head.norm`, the `lm_head` sweep and the argmax over all rows, and, on a
//!   vocab-sharded head, one exchange of `16 n` bytes. Every launch it adds is row-local and
//!   each row equals the one-row launch `forward_one` makes (`dense_gemv_bf16_batchm` and
//!   `dense_gemv_fp8w_batchm` per their kernel headers, `argmax_bf16_batch` runs the
//!   `argmax_bf16` body per block, `rms_norm_vanilla` one block per row), so each sequence's
//!   drafts, drafter rows and indexer rows equal what `propose` writes for it alone, given a
//!   pool of its own.
//! - With the head's one shared pool that equality cannot hold: a draft's attention would read
//!   the latent rows other sequences wrote at the same positions in a different order. So the
//!   batch lever implies `METRALE_GLM_MTP_SEQ_KV`, and `propose_batch_ready` refuses a state
//!   without its own pool.

use super::*;
use metrale_gpu_runtime::buffers::BufferArena;

/// 2026-10-04: Widest batch `propose_batch` takes: `DENSE_GEMV_BATCHM_MAX_M` and
/// `DENSE_GEMV_FP8W_BATCHM_MAX_M` (one block row of the batched GEMVs), the scratch rows.
pub const MTP_BATCH_DRAFT_MAX: usize = 16;

/// 2026-10-04: `METRALE_GLM_MTP_BATCH_DRAFT=1`: batched drafter passes across sequences
/// (`propose_batch`). Off unless set to `1`; read once.
pub fn mtp_batch_draft() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_MTP_BATCH_DRAFT").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_MTP_BATCH_DRAFT=1 - the GLM MTP drafter runs draft step d of every \
                 batched sequence in one pass (implies METRALE_GLM_MTP_SEQ_KV=1)"
            );
        }
        on
    })
}

/// 2026-10-04: `METRALE_GLM_MTP_SEQ_KV=1`, or the batch lever: each sequence's drafter gets its
/// own latent KV pool instead of the head's shared one. Off unless set; read once. It changes
/// the drafts of concurrent sequences (each stops reading the others' rows), never the tokens
/// a verify emits.
pub fn mtp_seq_kv() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("METRALE_GLM_MTP_SEQ_KV").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "METRALE_GLM_MTP_SEQ_KV=1 - each GLM MTP drafter state owns its latent KV pool"
            );
        }
        on || mtp_batch_draft()
    })
}

/// 2026-10-04: Bytes of one per-sequence drafter pool for `rows` tokens: `rows / 16 + 2`
/// blocks of 16 FP8 `[kv_lora_rank]` latents (`drafter_kv_config`), twice that when the V
/// pool is not aliased onto K. The serve's per-sequence reserve charges it.
pub fn seq_kv_pool_bytes(rows: usize, kv_lora_rank: usize) -> usize {
    let k = (rows / 16 + 2) * 16 * kv_lora_rank;
    if metrale_cache::kv_cache::glm_kv_v_alias("glm5_next") {
        k
    } else {
        2 * k
    }
}

/// 2026-10-04: The batched propose's kernels and `[MTP_BATCH_DRAFT_MAX, ..]` scratch: concat
/// `[2 hidden]`, block input/output `[hidden]`, logits `[vocab]` (BF16), argmax `[1]` u32 and
/// the head exchange `[8]` BF16 per row.
pub(super) struct BatchScratch {
    gemv_batchm: KernelHandle,
    gemv_fp8w_batchm: KernelHandle,
    argmax_batch: KernelHandle,
    concat: DevicePtr,
    x: DevicePtr,
    logits: DevicePtr,
    arg: DevicePtr,
    xchg: DevicePtr,
}

impl BatchScratch {
    /// 2026-10-04: `None` with the lever off or any of the three kernels missing; the head then
    /// proposes per sequence.
    pub(super) fn new(gpu: &dyn GpuBackend, hidden: usize, vocab: usize) -> Result<Option<Self>> {
        if !mtp_batch_draft() {
            return Ok(None);
        }
        let probe = metrale_model_layers::layers::try_kernel;
        let gemv_batchm = probe(gpu, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
        let gemv_fp8w_batchm = probe(gpu, "dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm");
        let argmax_batch = probe(gpu, "argmax", "argmax_bf16_batch");
        if gemv_batchm.0 == 0 || gemv_fp8w_batchm.0 == 0 || argmax_batch.0 == 0 {
            tracing::warn!(
                "METRALE_GLM_MTP_BATCH_DRAFT=1 but a batched kernel is missing \
                 (bf16_batchm={} fp8w_batchm={} argmax_batch={}); proposing per sequence",
                gemv_batchm.0 != 0,
                gemv_fp8w_batchm.0 != 0,
                argmax_batch.0 != 0
            );
            return Ok(None);
        }
        let m = MTP_BATCH_DRAFT_MAX;
        Ok(Some(Self {
            gemv_batchm,
            gemv_fp8w_batchm,
            argmax_batch,
            concat: gpu.alloc(m * 2 * hidden * 2)?,
            x: gpu.alloc(m * hidden * 2)?,
            logits: gpu.alloc(m * vocab * 2)?,
            arg: gpu.alloc(m * 4)?,
            xchg: gpu.alloc(m * 16)?,
        }))
    }
}

/// 2026-10-04: Where one drafter state keeps what a propose leaves behind, for the microtest
/// (`glm5next_mtp_batch_draft_microtest`): the last draft's block output `x` (after the final
/// norm) and logits, its own pool's K base and block table, its length, and its DSA indexer
/// rows (`k_normed`, `gate`, `valid`).
#[doc(hidden)]
pub struct DebugStateView {
    pub x: DevicePtr,
    pub logits: DevicePtr,
    pub own_k_pool: Option<DevicePtr>,
    pub block_table: Vec<u32>,
    pub seq_len: usize,
    pub idx_k_normed: DevicePtr,
    pub idx_gate: DevicePtr,
    pub idx_valid: DevicePtr,
}

impl Glm5NextMtpHead {
    /// 2026-10-04: `DebugStateView` of a GLM drafter state; `None` for any other state.
    #[doc(hidden)]
    pub fn debug_state_view(&self, state: &mut dyn ProposerState) -> Option<DebugStateView> {
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()?;
        Some(DebugStateView {
            x: st.x,
            logits: st.logits,
            own_k_pool: st.own_kv.as_ref().map(|k| k.k_pool_ptr(0)),
            block_table: st.block_table.clone(),
            seq_len: st.seq_len,
            idx_k_normed: st.dsa.k_normed,
            idx_gate: st.dsa.gate,
            idx_valid: st.dsa.valid,
        })
    }

    /// 2026-10-04: The batched propose's `[MTP_BATCH_DRAFT_MAX, hidden]` block output and
    /// `[MTP_BATCH_DRAFT_MAX, sweep columns]` logits (row stride = the sweep's column count,
    /// `vocab` or `head_n`); `None` without the scratch.
    #[doc(hidden)]
    pub fn debug_batch_bufs(&self) -> Option<(DevicePtr, DevicePtr)> {
        self.batch.as_ref().map(|b| {
            let b = b.lock();
            (b.x, b.logits)
        })
    }

    /// 2026-10-04: Whether a context write takes the row-batched tile path
    /// (`METRALE_GLM_MTP_CTX_ROWBATCH=1` and its scratch allocated), so a gate can prove the
    /// path it compares actually ran.
    #[doc(hidden)]
    pub fn debug_ctx_rowbatch(&self) -> bool {
        self.ctx_scratch.is_some() && ctx_rowbatch()
    }

    /// 2026-10-04: Columns of the draft-head sweep: `head_n` when sharded (a communicator and
    /// a vocab split), else `vocab`.
    #[doc(hidden)]
    pub fn debug_sweep_cols(&self, sharded: bool) -> usize {
        if sharded && self.head_n != self.vocab {
            self.head_n
        } else {
            self.vocab
        }
    }

    // 2026-10-04: `norm_rows` (one block per row, each row's bytes equal `norm` on that row)
    // is `Glm5NextMtpHead::norm_rows` in `glm5next_mtp_head.rs`, shared with the row-batched
    // context write (`METRALE_GLM_MTP_CTX_ROWBATCH`).

    /// 2026-10-04: `propose_batch_max`: `MTP_BATCH_DRAFT_MAX` capped by the shared forward
    /// buffers' rows and the block's MLP workspace; 1 (per sequence) without the scratch.
    pub(super) fn batch_rows_max(&self, buffers: &BufferArena) -> usize {
        if self.batch.is_none() {
            return 1;
        }
        MTP_BATCH_DRAFT_MAX
            .min(buffers.max_batch_tokens())
            .min(self.module.layer.mlp_ws.max_rows())
    }

    /// 2026-10-04: Whether `propose_batch` runs for these sequences: the scratch exists, 2 to
    /// `MTP_BATCH_DRAFT_MAX` sequences, at least one draft, neither timing arm
    /// (`METRALE_GLM_MTP_SKIP`), and every state a GLM drafter state with its own pool whose
    /// block table and the drafter's row cap cover every row this propose writes. Reads only
    /// host state, so ranks holding the same states answer the same.
    pub(super) fn batch_ready(
        &self,
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
    ) -> bool {
        let n = states.len();
        if self.batch.is_none()
            || !(2..=MTP_BATCH_DRAFT_MAX).contains(&n)
            || positions.len() != n
            || num_drafts == 0
            || skip_head()
            || skip_block()
        {
            return false;
        }
        states.iter_mut().zip(positions).all(|(s, &p)| {
            let Some(st) = s.as_any_mut().downcast_mut::<Glm5NextMtpProposerState>() else {
                return false;
            };
            // 2026-10-04: `propose` writes rows `start..start + num_drafts`, `start` the
            // drafter length after its rewind to `p`; the DSA gather indexes the block table
            // through `(row + 1) / 16 + 1` (`bt_entries_needed`).
            let start = st.seq_len.min(p);
            st.own_kv.is_some()
                && !st.released
                && p + num_drafts <= self.max_seq_len
                && (start + num_drafts) / 16 + 2 <= st.block_table.len()
        })
    }

    /// 2026-10-04: The batched propose (module docs). The caller checked `batch_ready`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_batch_impl(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let n = states.len();
        if last_tokens.len() != n || target_hiddens.len() != n || positions.len() != n {
            bail!(
                "GLM MTP batched propose: {n} states, {} tokens, {} hiddens, {} positions",
                last_tokens.len(),
                target_hiddens.len(),
                positions.len()
            );
        }
        let scratch = self
            .batch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GLM MTP batched propose without its scratch"))?
            .lock();
        let sc = &*scratch;
        let gpu = ctx.gpu;
        let h = self.hidden;
        let mut sts: Vec<&mut Glm5NextMtpProposerState> = states
            .iter_mut()
            .map(|s| {
                s.as_any_mut()
                    .downcast_mut::<Glm5NextMtpProposerState>()
                    .ok_or_else(|| anyhow::anyhow!("not a GLM MTP proposer state"))
            })
            .collect::<Result<_>>()?;
        // 2026-10-04: `propose`'s rewind and debug line, per sequence, before any launch.
        for (i, st) in sts.iter_mut().enumerate() {
            let position = positions[i];
            if st.seq_len > position {
                st.dsa.rewind_to(position)?;
                st.seq_len = position;
            }
            if metrale_model_layers::speculative::mtp_refeed_debug() {
                let fp = metrale_model_layers::speculative::hidden_fingerprint(
                    gpu,
                    target_hiddens[i],
                    h,
                );
                tracing::info!(
                    "GLM_MTP_DBG propose position={position} drafter_rows={} tok={} \
                     fp_target={fp:016x} (batched)",
                    st.seq_len,
                    last_tokens[i],
                );
            }
        }
        // 2026-10-04: The head sweep `forward_one` picks: sharded only with a communicator and
        // a vocab split; the FP8 copy only on the sharded sweep.
        let sharded = ctx.comm.is_some() && self.head_n != self.vocab;
        let (w, nv, v0) = if sharded {
            (
                DenseWeight {
                    weight: self.lm_head.weight.offset(self.head_v0 * h * 2),
                },
                self.head_n,
                self.head_v0,
            )
        } else {
            (self.lm_head, self.vocab, 0)
        };
        let fp8 = self.head_fp8.filter(|_| sharded && nv == self.head_n);

        let mut drafts: Vec<Vec<u32>> = (0..n).map(|_| Vec::with_capacity(num_drafts)).collect();
        let mut tokens = last_tokens.to_vec();
        for d in 0..num_drafts {
            for (i, &tok) in tokens.iter().enumerate() {
                let concat = sc.concat.offset(i * 2 * h * 2);
                let embed_row = self.embed_tokens.weight.offset(tok as usize * h * 2);
                self.norm(gpu, embed_row, self.module.enorm, concat, h, stream)?;
                // 2026-10-04: Draft 1 reads the target's verified hidden, every later draft
                // the drafter's own previous output (`propose`'s `hidden = st.x`).
                let hidden_in = if d == 0 {
                    target_hiddens[i]
                } else {
                    sc.x.offset(i * h * 2)
                };
                self.norm(
                    gpu,
                    hidden_in,
                    self.module.hnorm,
                    concat.offset(h * 2),
                    h,
                    stream,
                )?;
            }
            ops::dense_gemv_batchm(
                gpu,
                sc.gemv_batchm,
                sc.concat,
                &self.module.eh_proj,
                sc.x,
                n as u32,
                h as u32,
                (2 * h) as u32,
                h as u32,
                stream,
            )?;
            {
                let mut dsa_refs: Vec<&mut dyn LayerState> = Vec::with_capacity(n);
                let mut kv_refs: Vec<&mut PagedKvCache> = Vec::with_capacity(n);
                let mut bt_refs: Vec<&mut Vec<u32>> = Vec::with_capacity(n);
                let mut lens: Vec<usize> = Vec::with_capacity(n);
                for st in sts.iter_mut() {
                    let Glm5NextMtpProposerState {
                        dsa,
                        own_kv,
                        seq_len,
                        block_table,
                        ..
                    } = &mut **st;
                    let kv = own_kv.as_mut().ok_or_else(|| {
                        anyhow::anyhow!("GLM MTP batched propose: a state without its own pool")
                    })?;
                    dsa_refs.push(dsa);
                    kv_refs.push(kv);
                    bt_refs.push(block_table);
                    lens.push(*seq_len);
                }
                self.module.layer.decode_rows_for_drafter(
                    sc.x,
                    n,
                    &mut dsa_refs,
                    &mut kv_refs,
                    &lens,
                    &mut bt_refs,
                    ctx,
                    stream,
                )?;
            }
            for st in sts.iter_mut() {
                st.seq_len += 1;
            }
            self.norm_rows(gpu, sc.x, self.module.final_norm, sc.x, n, stream)?;
            match fp8 {
                Some(q) => ops::dense_gemv_fp8w_batchm(
                    gpu,
                    sc.gemv_fp8w_batchm,
                    sc.x,
                    &q,
                    sc.logits,
                    n as u32,
                    1,
                    nv as u32,
                    h as u32,
                    nv as u32,
                    stream,
                )?,
                None => ops::dense_gemv_batchm(
                    gpu,
                    sc.gemv_batchm,
                    sc.x,
                    &w,
                    sc.logits,
                    n as u32,
                    nv as u32,
                    h as u32,
                    nv as u32,
                    stream,
                )?,
            }
            ops::argmax_bf16_batch(
                gpu,
                sc.argmax_batch,
                sc.logits,
                sc.arg,
                nv as u32,
                n as u32,
                nv as u32,
                stream,
            )?;
            let mut out = vec![0u8; n * 4];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(sc.arg, &mut out)?;
            let local: Vec<usize> = out
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                .collect();
            let picks: Vec<u32> = match ctx.comm.filter(|_| sharded) {
                None => local.iter().map(|&l| (v0 + l) as u32).collect(),
                Some(comm) => self.exchange_picks(gpu, comm, sc, &local, nv, v0, stream)?,
            };
            for (i, &p) in picks.iter().enumerate() {
                drafts[i].push(p);
                tokens[i] = p;
            }
        }
        for st in sts.iter_mut() {
            st.last_drafted = num_drafts;
        }
        Ok(drafts)
    }

    /// 2026-10-04: `forward_one`'s cross-rank pick for every row in one all-reduce: row `i`
    /// packs the same 8 BF16 lanes at bytes `16 i..16 i + 16` (this rank's max logit, then
    /// three base-256 digits of its global index; the other ranks' lanes zero), so the SUM
    /// delivers each row's lanes as `forward_one`'s 16-byte all-reduce does, and the same
    /// lower-rank-wins rule picks.
    #[allow(clippy::too_many_arguments)]
    fn exchange_picks(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn metrale_comm::CommBackend,
        sc: &BatchScratch,
        local: &[usize],
        nv: usize,
        v0: usize,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let n = local.len();
        let mut lbs = vec![[0u8; 2]; n];
        for (i, (&l, lb)) in local.iter().zip(lbs.iter_mut()).enumerate() {
            gpu.copy_d2h_async(sc.logits.offset((i * nv + l) * 2), lb, stream)?;
        }
        gpu.synchronize(stream)?;
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        let mut pack = vec![0u8; 16 * n];
        for (i, (&l, lb)) in local.iter().zip(&lbs).enumerate() {
            let row = &mut pack[16 * i..16 * i + 16];
            let g = v0 + l;
            row[self.head_rank * 2..][..2].copy_from_slice(lb);
            for d in 0..3 {
                let digit = ((g >> (8 * d)) & 0xFF) as f32;
                row[4 + (self.head_rank * 3 + d) * 2..][..2].copy_from_slice(&bf(digit));
            }
        }
        gpu.copy_h2d(&pack, sc.xchg)?;
        comm.all_reduce_async(sc.xchg.0, 16 * n, stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(sc.xchg, &mut pack)?;
        Ok((0..n)
            .map(|i| {
                let row = &pack[16 * i..16 * i + 16];
                let lane = |j: usize| {
                    f32::from_bits((u16::from_le_bytes([row[j * 2], row[j * 2 + 1]]) as u32) << 16)
                };
                let win = if lane(0) >= lane(1) { 0 } else { 1 };
                (0..3).fold(0usize, |a, d| {
                    a + ((lane(2 + win * 3 + d) as usize) << (8 * d))
                }) as u32
            })
            .collect())
    }
}
