// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextMtpHead`: GLM-5.3's MTP block as a [`DraftProposer`], one draft token per `forward_one`.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - The FP8 copy of the draft head is read only on the vocab-sharded sweep; every other
//!   sweep reads the BF16 `lm_head`.
//!
//! ```text
//! x = eh_proj( concat( enorm(embed[token]), hnorm(target_hidden) ) )   [1, 2H] -> [1, H]
//! x = layers.{num_hidden_layers}(x)      DSA + routed MoE, plain residual (no mHC)
//! logits = lm_head( shared_head.norm(x) )              the target's own head
//! draft  = argmax(logits)
//! ```
//!
//! The block is sharded like a text layer: its routed experts are split across EP ranks and
//! its DSA `o_proj` is row-parallel, so its forward needs the communicator (`needs_comm`).
//!
//! 2026-10-04: Two default-off levers (`batch.rs`). `METRALE_GLM_MTP_SEQ_KV=1` gives every
//! sequence its own drafter latent KV pool: off, every state's block table is the whole of the
//! head's one private pool, so concurrent sequences write and read each other's latent rows
//! (a row's slot is its position, and positions overlap). `METRALE_GLM_MTP_BATCH_DRAFT=1`
//! (implies the first) runs draft step `d` of every batched sequence in one pass
//! (`propose_batch`).

use anyhow::{Result, bail};
use parking_lot::Mutex;
use std::any::Any;

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::{ForwardContext, LayerState};

use crate::weight_loader::Glm5NextMtpModule;
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::{DraftProposer, ProposerState};
use metrale_model_layers::weight_map::DenseWeight;

mod batch;
mod init;
mod proposer;

pub use batch::{
    DebugStateView, MTP_BATCH_DRAFT_MAX, mtp_batch_draft, mtp_seq_kv, seq_kv_pool_bytes,
};

/// 2026-09-25: Per-sequence drafter state: the block's own indexer cache and KV block table.
pub struct Glm5NextMtpProposerState {
    dsa: Glm5NextDsaState,
    /// 2026-10-04: This sequence's own drafter latent pool (`METRALE_GLM_MTP_SEQ_KV=1`); `None`
    /// reads and writes the head's shared pool, as before the lever.
    own_kv: Option<PagedKvCache>,
    /// 2026-09-25: Rows the drafter has written. `propose` and `after_verify` roll it back.
    seq_len: usize,
    block_table: Vec<u32>,
    /// 2026-09-25: How many drafts the last `propose` wrote, so `after_verify` knows what
    /// to trim.
    last_drafted: usize,
    /// 2026-09-25: Scratch: `[2, hidden]` BF16 concat, `[hidden]` BF16 block input,
    /// `[vocab]` BF16 logits, `[1]` u32 argmax.
    concat: DevicePtr,
    x: DevicePtr,
    logits: DevicePtr,
    arg: DevicePtr,
    /// 2026-09-25: 8 BF16 lanes for the vocab-sharded head's cross-rank pick: each rank's
    /// max logit, then three base-256 digits of each rank's token id.
    head_xchg: DevicePtr,
    /// 2026-09-25: Set by `free_state`, which is a no-op once it is set.
    released: bool,
}

impl ProposerState for Glm5NextMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    /// 2026-10-03: The drafter's DSA indexer cache, mapped with its look-ahead.
    fn map_rows_through(&self, end: usize) -> Result<()> {
        metrale_model_layers::layer::LayerState::map_rows_through(&self.dsa, end)
    }
}

pub struct Glm5NextMtpHead {
    module: Glm5NextMtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    /// 2026-09-25: A one-layer pool of the drafter's own, so its entries can be trimmed
    /// independently of the target's.
    kv_cache: Mutex<PagedKvCache>,
    rms_norm_k: KernelHandle,
    gemv_k: KernelHandle,
    argmax_k: KernelHandle,
    hidden: usize,
    vocab: usize,
    max_seq_len: usize,
    /// 2026-09-25: Vocab shard of the shared `lm_head` this rank sweeps:
    /// `[head_v0, head_v0 + head_n)`. `head_n == vocab` on a single rank or when `vocab_size`
    /// does not divide by `tp_world_size`.
    head_rank: usize,
    head_v0: usize,
    head_n: usize,
    /// 2026-09-25: FP8 E4M3 copy of this rank's vocab shard of `lm_head`, for drafting only.
    /// The target verifies every draft with its own head, so this copy can change the
    /// acceptance rate but not an emitted token. `None` when `METRALE_GLM_MTP_HEAD_FP8=0`,
    /// when the `gemv_fp8w` kernel is absent, or when quantisation fails.
    /// Measured 2026-08-29 (nsys): 2.66 ms -> ~1.33 ms per draft sweep.
    head_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    gemv_fp8w_k: KernelHandle,
    /// 2026-10-04: `kv_lora_rank` of the block, the head dim of a per-sequence pool.
    kv_lora_rank: usize,
    /// 2026-10-04: `METRALE_GLM_MTP_SEQ_KV` (or `METRALE_GLM_MTP_BATCH_DRAFT`) at construction:
    /// `alloc_state` gives each sequence its own latent pool.
    seq_kv: bool,
    /// 2026-10-04: The batched propose's kernels and scratch; `Some` only under
    /// `METRALE_GLM_MTP_BATCH_DRAFT=1` with every kernel resolved (`batch.rs`).
    batch: Option<Mutex<batch::BatchScratch>>,
    /// 2026-10-03: Scratch for the row-batched context write; `Some` only with
    /// `METRALE_GLM_MTP_CTX_ROWBATCH=1` and a block that can run it. Allocated at load, so it
    /// sits before KV sizing. One tile buffer is enough because its own mutex is held for the
    /// whole write (2026-10-04: under `METRALE_GLM_MTP_SEQ_KV` a context write uses the
    /// sequence's own pool and no longer holds the head's `kv_cache` lock, which used to
    /// serialise it).
    ctx_scratch: Option<Mutex<CtxScratch>>,
}

/// 2026-10-03: Rows per tile of the batched context write.
const CTX_TILE: usize = 256;

/// 2026-10-03: `METRALE_GLM_MTP_CTX_ROWBATCH=1`; read once per process.
fn ctx_rowbatch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_CTX_ROWBATCH").ok().as_deref() == Some("1"))
}

/// 2026-10-03: Device scratch of one `CTX_TILE`: `gath` and `nrm` are `[T, hidden]`,
/// `concat` `[T, 2 hidden]`, `xo` `[T, hidden]`, `kv_a` `[T, kv_lora_rank]` (all BF16),
/// `slots` `[T]` i64.
struct CtxScratch {
    gath: DevicePtr,
    nrm: DevicePtr,
    concat: DevicePtr,
    xo: DevicePtr,
    kv_a: DevicePtr,
    slots: DevicePtr,
}

/// 2026-09-25: Rows the GLM drafter can ever be asked for: the served context, clamped to
/// the rows its DSA indexer cache reserves. `max_dsa_context` is `max_context` (the serve's
/// `--max-seq-len`) rounded down to whole `index_kpool` pools, so the cap follows the config.
fn drafter_context_rows(max_seq_len: usize, cfg: &crate::glm5next_dsa::Glm5NextDsaConfig) -> usize {
    max_seq_len.min(crate::glm5next_dsa::state::max_dsa_context(cfg))
}

impl Glm5NextMtpHead {
    /// 2026-10-05: `eh_proj` on one `[2 * hidden]` row: `dense_fp8::route` (the FP8 copy under
    /// `METRALE_GLM_DENSE_FP8=1`, the tensor-core GEMV under `METRALE_GLM_GEMV_TC=1`), else the
    /// BF16 `dense_gemv_bf16` it always ran.
    fn eh_proj_one(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        c: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::dense_fp8::{Route, route};
        let (h, w) = (self.hidden, self.module.eh_proj.weight);
        match route(gpu, self.gemv_k, a, w, c, 1, h, 2 * h, stream)? {
            Route::Done => Ok(()),
            Route::Weight(p) => ops::dense_gemv(
                gpu,
                self.gemv_k,
                a,
                &DenseWeight { weight: p },
                c,
                h as u32,
                (2 * h) as u32,
                stream,
            ),
        }
    }

    /// 2026-10-05: `eh_proj` on a tile of `t > 16` rows as one cuBLASLt GEMM; under
    /// `METRALE_GLM_DENSE_FP8=1` it reads the BF16 dequant of the FP8 copy (`route`, dequant arena).
    fn eh_proj_wide(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        c: DevicePtr,
        t: usize,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::dense_fp8::{Route, route};
        let (h, w) = (self.hidden, self.module.eh_proj.weight);
        let w = match route(gpu, self.gemv_k, a, w, c, t, h, 2 * h, stream)? {
            Route::Done => return Ok(()),
            Route::Weight(p) => p,
        };
        ops::cublas_bf16_proj_dense(a, w, c, t as u32, h as u32, (2 * h) as u32, stream)
    }

    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([1, 1, 1])
            .block([(n.min(1024)) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .arg_f32(self.module.layer.rms_eps)
            .launch(stream)
    }

    /// 2026-10-03: `norm` over `rows` rows (one block per row, same block size as `norm`).
    fn norm_rows(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([rows as u32, 1, 1])
            .block([(self.hidden.min(1024)) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(self.hidden as u32)
            .arg_f32(self.module.layer.rms_eps)
            .launch(stream)
    }

    /// 2026-09-25: One draft token. Advances `seq_len` by one.
    fn forward_one(
        &self,
        token: u32,
        hidden_in: DevicePtr,
        position: usize,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<u32> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        if position >= self.max_seq_len {
            bail!(
                "GLM MTP drafter: position {position} is past the {} it was sized for",
                self.max_seq_len
            );
        }
        // 2026-09-25: The embedding row is read by pointer: `embed_tokens` is `[vocab, hidden]`
        // BF16 and a row is contiguous, so there is nothing to gather.
        let embed_row = self.embed_tokens.weight.offset(token as usize * h * 2);
        self.norm(gpu, embed_row, self.module.enorm, st.concat, h, stream)?;
        self.norm(
            gpu,
            hidden_in,
            self.module.hnorm,
            st.concat.offset(h * 2),
            h,
            stream,
        )?;
        self.eh_proj_one(gpu, st.concat, st.x, stream)?;

        // 2026-09-25: The block writes its output back over `st.x` (plain residual, in place).
        if skip_block() {
            st.seq_len += 1;
        } else {
            // 2026-10-04: The sequence's own pool under `METRALE_GLM_MTP_SEQ_KV`, else the
            // head's shared one.
            let mut shared = None;
            let kv: &mut PagedKvCache = match st.own_kv.as_mut() {
                Some(own) => own,
                None => &mut *shared.insert(self.kv_cache.lock()),
            };
            let dsa_state: &mut dyn LayerState = &mut st.dsa;
            self.module.layer.decode_one_for_drafter(
                st.x,
                dsa_state,
                kv,
                st.seq_len,
                &mut st.block_table,
                ctx,
                stream,
            )?;
            drop(shared);
            st.seq_len += 1;
        }

        // 2026-09-25: Timing arm `METRALE_GLM_MTP_SKIP=head`: the head is skipped and the
        // draft is token 0, so `propose` times the block alone. `=block` skips the block instead.
        if skip_head() {
            return Ok(0);
        }
        // 2026-09-25: `shared_head.norm`, then the target's own `lm_head`: the MTP layer carries
        // no head of its own.
        self.norm(gpu, st.x, self.module.final_norm, st.x, h, stream)?;
        // 2026-09-25: Sharded only with a communicator and a vocab split across ranks;
        // otherwise the sweep covers the whole vocab.
        let sharded = ctx.comm.is_some() && self.head_n != self.vocab;
        let (w, n, v0) = if sharded {
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
        // 2026-09-25: The FP8 copy covers only `[head_v0, head_v0 + head_n)`, so it serves the
        // sharded sweep alone; an unsharded sweep reads the BF16 head.
        match self.head_fp8.filter(|_| sharded && n == self.head_n) {
            Some(q) => ops::dense_gemv_fp8w(
                gpu,
                self.gemv_fp8w_k,
                st.x,
                &q,
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
            None => ops::dense_gemv(
                gpu,
                self.gemv_k,
                st.x,
                &w,
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
        }
        ops::argmax_bf16(gpu, self.argmax_k, st.logits, st.arg, n as u32, stream)?;
        let mut out = [0u8; 4];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.arg, &mut out)?;
        let local = u32::from_le_bytes(out) as usize;
        let Some(comm) = ctx.comm.filter(|_| sharded) else {
            return Ok((v0 + local) as u32);
        };
        // 2026-09-25: Exchange (max, argmax) in 8 BF16 lanes: `[val_r0, val_r1, then 3 base-256
        // digits of each rank's global index]`. Each rank writes only its own lanes and leaves
        // the others zero, so a SUM all-reduce delivers both ranks' values untouched (`x + 0.0`
        // is exact).
        //
        // The all-reduce sums its bytes as BF16 elements (`NcclDataType::Bfloat16`), so an f32
        // packed here would be summed as two BF16 halves. A token id needs 18 bits and BF16
        // holds integers exactly only through 256, hence the digits; the logit lane is already
        // BF16, so it round-trips bit for bit.
        let mut lb = [0u8; 2];
        gpu.copy_d2h(st.logits.offset(local * 2), &mut lb)?;
        let g = v0 + local;
        let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes();
        let mut pack = [0u8; 16];
        pack[self.head_rank * 2..][..2].copy_from_slice(&lb);
        for d in 0..3 {
            let digit = ((g >> (8 * d)) & 0xFF) as f32;
            pack[4 + (self.head_rank * 3 + d) * 2..][..2].copy_from_slice(&bf(digit));
        }
        gpu.copy_h2d(&pack, st.head_xchg)?;
        comm.all_reduce_async(st.head_xchg.0, 16, stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.head_xchg, &mut pack)?;
        let lane = |i: usize| {
            f32::from_bits(
                (u16::from_le_bytes(pack[i * 2..][..2].try_into().unwrap()) as u32) << 16,
            )
        };
        // 2026-09-25: `>=` makes the lower rank win a tie, identically on both ranks, so the
        // two ranks' drafter KV streams cannot diverge on a tie.
        let win = if lane(0) >= lane(1) { 0 } else { 1 };
        let idx = (0..3).fold(0usize, |a, d| {
            a + ((lane(2 + win * 3 + d) as usize) << (8 * d))
        });
        Ok(idx as u32)
    }

    /// 2026-09-25: Append `tokens.len() - 1` drafter context rows: row `r` is pair key
    /// `row_base + r` = `(embed(tokens[r + 1]), hiddens row r)`. Used for both the whole-prompt
    /// prefill and the catch-up feed; the only difference between them is `row_base`.
    ///
    /// The row space is dense: a row's KV slot, indexer row and RoPE position are all
    /// `seq_len` (see `Glm5NextDsaLayer::write_kv_row`), so every pair key from 0 up is
    /// written and slot == key.
    ///
    /// Unless `METRALE_GLM_MTP_PREFILL_FULL=1`, a context row runs only `input_norm` and the
    /// DSA cache writes (`drafter_write_kv_row`): no MoE, no attention, no `lm_head`, because
    /// its block output is discarded.
    #[allow(clippy::too_many_arguments)]
    fn rows_impl(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let st = match state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
        {
            Some(s) => s,
            None => return Ok(0),
        };
        // 2026-09-25: Rows must append exactly at the drafter's current length, or the dense
        // row space grows a hole and every later RoPE position is wrong.
        if st.seq_len != row_base || tokens.len() < 2 {
            return Ok(0);
        }
        let h = self.hidden;
        let rows = tokens.len() - 1;
        if row_base + rows > self.max_seq_len {
            return Ok(0);
        }
        let gpu = ctx.gpu;
        let dbg = metrale_model_layers::speculative::mtp_refeed_debug();
        let prefill_full = std::env::var("METRALE_GLM_MTP_PREFILL_FULL")
            .ok()
            .as_deref()
            == Some("1");
        let Glm5NextMtpProposerState {
            dsa,
            own_kv,
            seq_len,
            block_table,
            concat,
            x,
            ..
        } = st;
        // 2026-10-04: The sequence's own pool under `METRALE_GLM_MTP_SEQ_KV`, else the head's.
        // Both context paths below (the row-batched tile write and the per-row loop) write
        // through this one `kv`.
        let mut shared = None;
        let kv: &mut PagedKvCache = match own_kv.as_mut() {
            Some(own) => own,
            None => &mut *shared.insert(self.kv_cache.lock()),
        };
        if let Some(sc_lock) = self
            .ctx_scratch
            .as_ref()
            .filter(|_| !prefill_full && !dbg && ctx_rowbatch())
        {
            let sc = sc_lock.lock();
            let dsa_state: &mut dyn LayerState = dsa;
            let mut r0 = 0;
            while r0 < rows {
                let t = CTX_TILE.min(rows - r0);
                if r0 > 0 {
                    // The tile's blocking slot copy overwrites `slots`; the previous tile's
                    // latent write may still be reading it.
                    gpu.synchronize(stream)?;
                }
                for i in 0..t {
                    gpu.copy_d2d_async(
                        self.embed_tokens
                            .weight
                            .offset(tokens[r0 + i + 1] as usize * h * 2),
                        sc.gath.offset(i * h * 2),
                        h * 2,
                        stream,
                    )?;
                }
                // 2026-10-03: Same one-block-per-row kernel and block size as the per-row
                // `norm`, so each row's bytes are the per-row path's.
                self.norm_rows(gpu, sc.gath, self.module.enorm, sc.nrm, t, stream)?;
                gpu.copy_d2d_2d_async(sc.nrm, h * 2, sc.concat, 2 * h * 2, h * 2, t, stream)?;
                self.norm_rows(
                    gpu,
                    hiddens.offset(r0 * h * 2),
                    self.module.hnorm,
                    sc.nrm,
                    t,
                    stream,
                )?;
                gpu.copy_d2d_2d_async(
                    sc.nrm,
                    h * 2,
                    sc.concat.offset(h * 2),
                    2 * h * 2,
                    h * 2,
                    t,
                    stream,
                )?;
                // 2026-10-03: eh_proj as one GEMM over the tile (GEMV order differs from
                // cuBLASLt's: bf16-level only). Tiles of up to 16 rows keep the GEMV.
                if t > ops::DENSE_GEMV_BATCHM_MAX_M as usize {
                    self.eh_proj_wide(gpu, sc.concat, sc.xo, t, stream)?;
                } else {
                    for i in 0..t {
                        self.eh_proj_one(
                            gpu,
                            sc.concat.offset(i * 2 * h * 2),
                            sc.xo.offset(i * h * 2),
                            stream,
                        )?;
                    }
                }
                self.module.layer.drafter_write_kv_rows(
                    sc.xo,
                    t,
                    sc.nrm,
                    sc.kv_a,
                    sc.slots,
                    dsa_state,
                    kv,
                    *seq_len,
                    &block_table[..],
                    ctx,
                    stream,
                )?;
                *seq_len += t;
                r0 += t;
            }
            return Ok(rows);
        }
        for r in 0..rows {
            let embed_row = self
                .embed_tokens
                .weight
                .offset(tokens[r + 1] as usize * h * 2);
            self.norm(gpu, embed_row, self.module.enorm, *concat, h, stream)?;
            self.norm(
                gpu,
                hiddens.offset(r * h * 2),
                self.module.hnorm,
                concat.offset(h * 2),
                h,
                stream,
            )?;
            self.eh_proj_one(gpu, *concat, *x, stream)?;
            let dsa_state: &mut dyn LayerState = dsa;
            // 2026-09-25: Diagnostic arm `METRALE_GLM_MTP_PREFILL_FULL=1`: build the row through
            // the full block path a propose uses (`decode_one_for_drafter`) instead of the
            // cache-only write.
            if prefill_full {
                self.module.layer.decode_one_for_drafter(
                    *x,
                    dsa_state,
                    kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            } else {
                self.module.layer.drafter_write_kv_row(
                    *x,
                    dsa_state,
                    kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            }
            if dbg {
                let fp = metrale_model_layers::speculative::hidden_fingerprint(
                    gpu,
                    hiddens.offset(r * h * 2),
                    h,
                );
                tracing::info!(
                    "GLM_MTP_DBG ctx row slot={} key={} tok={} fp_hidden={fp:016x}",
                    *seq_len,
                    row_base + r,
                    tokens[r + 1],
                );
            }
            *seq_len += 1;
        }
        Ok(rows)
    }
}

/// 2026-09-25: `METRALE_GLM_MTP_SKIP=head`: stop the drafter after the block, before
/// `shared_head.norm`, the `lm_head` gemv, the argmax and the D2H. A timing arm; read once
/// per process.
fn skip_head() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_SKIP").ok().as_deref() == Some("head"))
}

/// 2026-09-25: `METRALE_GLM_MTP_SKIP=block`: skip the MTP block itself and run only the
/// head. A timing arm; read once per process.
fn skip_block() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_SKIP").ok().as_deref() == Some("block"))
}

#[cfg(test)]
#[path = "glm5next_mtp_head_tests.rs"]
mod a59_sizing_tests;
