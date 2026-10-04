// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: GPU check of the GLM-5.3 batched MTP propose (`METRALE_GLM_MTP_BATCH_DRAFT=1`,
//! `glm5next_mtp_head/batch.rs`): draft step `d` of every sequence in one pass.
//!
//! The MTP head is built as the serve builds it (`Glm5NextMtpHead::new`): one DSA + routed MoE
//! block with no hyper-connection and a row-parallel `o_proj` at the rank-0 TP=2/EP=2 geometry
//! (random weights, `common/glm5next_multiseq_stack.rs`), the real vocab, a random BF16
//! `embed_tokens` and `lm_head`, and the FP8 shard of the draft head. Four sequences are
//! prefilled to `MTPB_LENS` (default 37, 2300, 120, 2600: two past the DSA top-k of 2048), then
//! run six rounds of propose + `after_verify`, draft counts 1, 2, 3, 2, 1, 3 and a random
//! accepted count per sequence and round (so positions and drafter lengths diverge).
//!
//! Three arms, each under two contexts (no communicator: the BF16 full-vocab sweep; a one-rank
//! communicator: the vocab-sharded FP8 sweep and the cross-rank pick):
//! - REF: a child process with both levers off (the head's shared drafter pool, one-row MLP
//!   workspace), each sequence alone and in turn, per-sequence `propose`: the incumbent.
//! - SERIAL: this process, levers on (each sequence its own pool), per-sequence `propose`.
//! - BATCHED: this process, `propose_batch` over sequences 0..2 (N = 2) and 0..4 (N = 4).
//!
//! Bitwise, every round: drafted ids, the last draft's logits row and block output `x`, BATCHED
//! vs SERIAL and SERIAL vs REF. After the last round, BATCHED vs SERIAL: every drafter latent KV
//! row and DSA indexer row (`k_normed`, `gate`, `valid`) in `[0, seq_len)` and `seq_len`.
//!
//! CTX leg (2026-10-04, the comb10 port): the drafter context written by the row-batched tile
//! path (`METRALE_GLM_MTP_CTX_ROWBATCH=1`, `Glm5NextDsaLayer::write_kv_rows`) into each
//! sequence's own pool (`METRALE_GLM_MTP_SEQ_KV`, implied by the batch lever), fed as the
//! chunked-capture drain feeds it (`METRALE_GLM_MTP_CHUNKED_CAPTURE`: the first chunk through
//! `prefill_drafter`, every later one through `catchup_drafter` at `row_base` = the drafter's
//! rows). It runs in a third child (the lever is read once per process and decides the tile
//! scratch at head construction) and is compared with this process's per-row walk:
//! - A: drain chunks of at most 16 rows (eh_proj stays the GEMV): every drafter latent KV row and
//!   DSA indexer row bitwise equal to the per-row walk, per sequence.
//! - B: 1000-row drain chunks (256-row tiles, eh_proj as one cuBLASLt GEMM, a bf16
//!   reduction-order change): within the tolerance of `glm5next_mtp_ctx_rowbatch_microtest` leg
//!   C (`ctx_tolerance`). A write that went to any pool but the sequence's own fails it.
//! - C / D: the six serial rounds, and the batched rounds over all four sequences, after the A
//!   drain, bitwise equal to the serial rounds after the per-row walk (sharded context).
//!
//! Timing, N = 2 and 4, two drafts, the sharded context: median wall time of the
//! per-sequence loop and of one `propose_batch` (`MTPB_REPS`, default 20). A one-rank
//! communicator does no transfer, so the per-row collectives the batch saves on a TP=2 serve
//! are not in these numbers.
//!
//! Every launch, sync and readback uses `gpu.default_stream()`, as the serve's propose does:
//! the backend's `copy_h2d` runs on that stream, so launching on stream 0 would let a later
//! upload into a shared one-row buffer (the DSA block table and lengths) overtake kernels still
//! queued for the previous sequence.
//!
//! Run on one GB10: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples
//! --example glm5next_mtp_batch_draft_microtest`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#[path = "common/glm5next_multiseq_stack.rs"]
mod glm5next_multiseq_stack;

use anyhow::{Context, Result, bail};
use glm5next_multiseq_stack as st;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype};
use metrale_comm::{CommBackend, SingleGpuBackend};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::layer::Glm5NextDsaWorkspace;
use metrale_model_arch::glm5next_layer::Glm5NextMixer;
use metrale_model_arch::glm5next_mlp::forward::Glm5NextMlpWorkspace;
use metrale_model_arch::glm5next_mtp_head::{Glm5NextMtpHead, MTP_BATCH_DRAFT_MAX};
use metrale_model_arch::weight_loader::glm5_next_mtp::Glm5NextMtpModule;
use metrale_model_layers::layer::{ForwardContext, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use metrale_model_layers::speculative::{DraftProposer, ProposerState};
use metrale_model_layers::weight_map::DenseWeight;
use std::time::Instant;

const ND_SCHEDULE: [usize; 6] = [1, 2, 3, 2, 1, 3];
const LEVERS: [&str; 2] = ["METRALE_GLM_MTP_BATCH_DRAFT", "METRALE_GLM_MTP_SEQ_KV"];
/// 2026-10-04: The context-write lever the CTX leg turns on (in its own child: it is read once
/// per process and decides at head construction whether the tile scratch exists).
const CTX_LEVER: &str = "METRALE_GLM_MTP_CTX_ROWBATCH";
/// 2026-10-04: CTX leg A: drain chunk sizes, cycled, each at most 16 rows so every tile keeps
/// the eh_proj GEMV (`rows_impl`), which is byte-identical to the per-row walk. The first
/// chunk goes through `prefill_drafter`, the rest through `catchup_drafter` at
/// `row_base = drafter rows`, as `METRALE_GLM_MTP_CHUNKED_CAPTURE`'s drain feeds it.
const SMALL_CHUNKS: [usize; 7] = [16, 5, 16, 11, 1, 16, 9];
/// 2026-10-04: CTX leg B: drain chunks of this many rows (256-row tiles, eh_proj as one GEMM).
const BIG_CHUNK: usize = 1000;

/// 2026-10-04: What every `ForwardContext` borrows.
struct Env<'g> {
    gpu: &'g dyn GpuBackend,
    config: ModelConfig,
    buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    levers: ModelLevers,
    stats: ModelStats,
    single: SingleGpuBackend,
}

impl Env<'_> {
    /// 2026-10-04: The propose context of `run_mtp_propose_inner`, with the communicator when
    /// `sharded`.
    fn ctx(&self, sharded: bool) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: if sharded {
                Some(&self.single as &dyn CommBackend)
            } else {
                None
            },
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }
}

/// 2026-10-04: One round of one sequence: the committed token and the target hidden the
/// propose reads, its draft count, and the accepted count `after_verify` gets.
struct Round {
    token: u32,
    hidden: Vec<u8>,
    nd: usize,
    accepted: usize,
}

/// 2026-10-04: One sequence's prompt (`len + 1` tokens: `prefill_drafter` writes `len` rows) and
/// rounds. Deterministic from `seed`.
struct Plan {
    prompt: Vec<u32>,
    prompt_hidden: Vec<u8>,
    rounds: Vec<Round>,
}

fn plan(len: usize, h: usize, vocab: usize, seed: u64) -> Plan {
    let mut r = st::Lcg(seed);
    let prompt = (0..=len)
        .map(|_| (r.next() % vocab as u64) as u32)
        .collect();
    let prompt_hidden = st::bf16_bytes(&r.vec(len * h, 1.0));
    let rounds = ND_SCHEDULE
        .iter()
        .map(|&nd| Round {
            token: (r.next() % vocab as u64) as u32,
            hidden: st::bf16_bytes(&r.vec(h, 1.0)),
            nd,
            accepted: (r.next() % (nd as u64 + 1)) as usize,
        })
        .collect();
    Plan {
        prompt,
        prompt_hidden,
        rounds,
    }
}

/// 2026-10-04: A `[rows, cols]` BF16 weight, uniform in `[-amp, amp]`, generated and uploaded
/// in 64 MiB pieces (the real vocab makes `lm_head` 1.9 GB).
fn big_bf16(
    g: &dyn GpuBackend,
    rows: usize,
    cols: usize,
    amp: f32,
    seed: u64,
) -> Result<DevicePtr> {
    let total = rows * cols;
    let p = g.alloc(total * 2)?;
    let mut x = seed | 1;
    let piece = 32 << 20;
    let mut buf = Vec::with_capacity(piece * 2);
    let mut done = 0usize;
    while done < total {
        let n = piece.min(total - done);
        buf.clear();
        for _ in 0..n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = ((x >> 40) & 0xFFFF) as f32 / 65535.0;
            buf.extend_from_slice(&half::bf16::from_f32((u * 2.0 - 1.0) * amp).to_le_bytes());
        }
        g.copy_h2d(&buf, p.offset(done * 2))?;
        done += n;
    }
    Ok(p)
}

/// 2026-10-04: The MTP head as `load_glm5next_mtp_module` and `Glm5NextMtpHead::new` build it,
/// from the synthetic stack's DSA + routed MoE layer.
fn build_head(gpu: &dyn GpuBackend, geo: &st::Geometry, sets: usize) -> Result<Glm5NextMtpHead> {
    let cfg = &geo.config;
    let h = cfg.hidden_size;
    let mut rng = st::Lcg(0x3770_BA7C);
    let mut stack = st::build_stack(gpu, geo, sets, &mut rng)?;
    let mut layer = stack.swap_remove(1);
    let Glm5NextMixer::Dsa(dsa) = &mut layer.mixer else {
        bail!("stack layer 1 is not the DSA layer");
    };
    // 2026-10-04: As the loader: a one-row DSA workspace, its own one-layer pool.
    dsa.workspace = Glm5NextDsaWorkspace::new(gpu, &dsa.cfg, 1)?;
    dsa.layer_idx = cfg.num_hidden_layers;
    dsa.attn_layer_idx = 0;
    dsa.persist_bt = true;
    layer.layer_idx = cfg.num_hidden_layers;
    layer.mhc = None;
    layer.mixer_all_reduce = true;
    layer.is_first = false;
    layer.is_last = false;
    let ws_rows = if metrale_model_arch::glm5next_mtp_head::mtp_batch_draft() {
        MTP_BATCH_DRAFT_MAX
    } else {
        1
    };
    layer.mlp_ws = std::sync::Arc::new(Glm5NextMlpWorkspace::new(gpu, &geo.mlp, ws_rows)?);
    let norm = |r: &mut st::Lcg| -> Result<DevicePtr> {
        let v: Vec<f32> = r.vec(h, 0.1).iter().map(|x| 1.0 + x).collect();
        st::up_bf16(gpu, &v)
    };
    let module = Glm5NextMtpModule {
        layer,
        eh_proj: DenseWeight {
            weight: st::up_bf16(gpu, &rng.vec(h * 2 * h, 0.02))?,
        },
        enorm: norm(&mut rng)?,
        hnorm: norm(&mut rng)?,
        final_norm: norm(&mut rng)?,
    };
    let embed = DenseWeight {
        weight: big_bf16(gpu, cfg.vocab_size, h, 0.05, 0xE3BE_D001)?,
    };
    let lm_head = DenseWeight {
        weight: big_bf16(gpu, cfg.vocab_size, h, 0.02, 0x13AD_0002)?,
    };
    Glm5NextMtpHead::new(module, embed, lm_head, cfg, gpu, 8192)
}

/// 2026-10-04: What one round leaves for one sequence: drafts, the last draft's logits row and
/// block output.
#[derive(PartialEq, Clone)]
struct Out {
    drafts: Vec<u32>,
    logits: Vec<u8>,
    x: Vec<u8>,
}

impl Out {
    fn bytes(&self) -> Vec<u8> {
        let mut b: Vec<u8> = self.drafts.iter().flat_map(|d| d.to_le_bytes()).collect();
        b.extend_from_slice(&self.logits);
        b.extend_from_slice(&self.x);
        b
    }
}

struct Rig<'g> {
    env: Env<'g>,
    head: Glm5NextMtpHead,
    plans: Vec<Plan>,
    h: usize,
    kv_lora: usize,
    index_dim: usize,
    hid_in: DevicePtr,
    prompt_hid: DevicePtr,
}

impl Rig<'_> {
    fn g(&self) -> &dyn GpuBackend {
        self.env.gpu
    }

    /// 2026-10-04: A fresh drafter state for sequence `u` with its prompt rows written.
    fn fresh(&self, u: usize, sharded: bool) -> Result<Box<dyn ProposerState>> {
        self.fresh_with(u, sharded, None)
    }

    /// 2026-10-04: `fresh`, with the prompt rows written whole (`chunks` `None`) or as a
    /// chunked-capture drain: the first chunk through `prefill_drafter`, every later one through
    /// `catchup_drafter` at `row_base` = the drafter's rows, hiddens from that prompt row.
    fn fresh_with(
        &self,
        u: usize,
        sharded: bool,
        chunks: Option<&[usize]>,
    ) -> Result<Box<dyn ProposerState>> {
        self.fresh_timed(u, sharded, chunks).map(|(s, _)| s)
    }

    /// 2026-10-04: `fresh_with`, plus the wall time of the context writes alone (default
    /// stream drained before and after; state allocation and the hidden upload excluded).
    fn fresh_timed(
        &self,
        u: usize,
        sharded: bool,
        chunks: Option<&[usize]>,
    ) -> Result<(Box<dyn ProposerState>, f64)> {
        let g = self.g();
        let p = &self.plans[u];
        let len = p.prompt.len() - 1;
        let mut s = self.head.alloc_state_for(g, len + 512)?;
        g.synchronize(g.default_stream())?;
        g.copy_h2d(&p.prompt_hidden, self.prompt_hid)?;
        let ctx = self.env.ctx(sharded);
        let stream = g.default_stream();
        g.synchronize(stream)?;
        let t = Instant::now();
        let rows = match chunks {
            None => {
                self.head
                    .prefill_drafter(&p.prompt, self.prompt_hid, s.as_mut(), &ctx, stream)?
            }
            Some(cs) => {
                let (mut lo, mut i) = (0usize, 0usize);
                while lo < len {
                    let c = cs[i % cs.len()].min(len - lo);
                    let toks = &p.prompt[lo..lo + c + 1];
                    let hid = self.prompt_hid.offset(lo * self.h * 2);
                    let n = if lo == 0 {
                        self.head
                            .prefill_drafter(toks, hid, s.as_mut(), &ctx, stream)?
                    } else {
                        self.head.catchup_drafter(
                            toks,
                            hid,
                            lo,
                            lo + 1,
                            s.as_mut(),
                            &ctx,
                            stream,
                        )?
                    };
                    if n != c {
                        bail!("drain chunk at row {lo} wrote {n} rows, expected {c}");
                    }
                    lo += c;
                    i += 1;
                }
                lo
            }
        };
        g.synchronize(stream)?;
        let secs = t.elapsed().as_secs_f64();
        if rows != len {
            bail!("prefill wrote {rows} rows, expected {len}");
        }
        Ok((s, secs))
    }

    /// 2026-10-04: Bytes of one drafter latent row in a pool (`drafter_kv_config`).
    fn latent_row(&self) -> usize {
        let kv_cfg = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: self.kv_lora,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
            v_aliases_k: metrale_cache::kv_cache::glm_kv_v_alias("glm5_next"),
        };
        kv_cfg.k_block_bytes_for_layer(0) / 16
    }

    /// 2026-10-04: `n` bytes at `p` after the default stream drains (`st::down` syncs stream 0,
    /// which the rig does not use).
    fn down(&self, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
        let g = self.g();
        g.synchronize(g.default_stream())?;
        let mut b = vec![0u8; n];
        g.copy_d2h(p, &mut b)?;
        Ok(b)
    }

    /// 2026-10-04: One per-sequence propose of round `r` for sequence `u` at `pos`.
    fn serial_round(
        &self,
        u: usize,
        r: usize,
        pos: usize,
        s: &mut dyn ProposerState,
        sharded: bool,
    ) -> Result<Out> {
        let g = self.g();
        let rd = &self.plans[u].rounds[r];
        g.synchronize(g.default_stream())?;
        g.copy_h2d(&rd.hidden, self.hid_in)?;
        let ctx = self.env.ctx(sharded);
        let drafts = self.head.propose(
            rd.token,
            self.hid_in,
            pos,
            rd.nd,
            s,
            &ctx,
            g.default_stream(),
            None,
            None,
            None,
        )?;
        let v = self.head.debug_state_view(s).context("not a GLM state")?;
        let cols = self.head.debug_sweep_cols(sharded);
        Ok(Out {
            drafts,
            logits: self.down(v.logits, cols * 2)?,
            x: self.down(v.x, self.h * 2)?,
        })
    }

    /// 2026-10-04: Round `r` for sequences `0..n` in one `propose_batch`.
    fn batch_round(
        &self,
        r: usize,
        pos: &[usize],
        states: &mut [Box<dyn ProposerState>],
        sharded: bool,
    ) -> Result<Vec<Out>> {
        let g = self.g();
        let n = states.len();
        let h = self.h;
        let nd = self.plans[0].rounds[r].nd;
        g.synchronize(g.default_stream())?;
        for u in 0..n {
            g.copy_h2d(
                &self.plans[u].rounds[r].hidden,
                self.hid_in.offset(u * h * 2),
            )?;
        }
        let tokens: Vec<u32> = (0..n).map(|u| self.plans[u].rounds[r].token).collect();
        let hiddens: Vec<DevicePtr> = (0..n).map(|u| self.hid_in.offset(u * h * 2)).collect();
        let mut refs: Vec<&mut dyn ProposerState> = Vec::with_capacity(n);
        for s in states.iter_mut() {
            refs.push(s.as_mut());
        }
        let ctx = self.env.ctx(sharded);
        let drafts = self
            .head
            .propose_batch(
                &tokens,
                &hiddens,
                pos,
                nd,
                &mut refs,
                &ctx,
                g.default_stream(),
                None,
            )?
            .context("propose_batch declined (propose_batch_ready false)")?;
        let (bx, bl) = self.head.debug_batch_bufs().context("no batch scratch")?;
        let cols = self.head.debug_sweep_cols(sharded);
        (0..n)
            .map(|u| {
                Ok(Out {
                    drafts: drafts[u].clone(),
                    logits: self.down(bl.offset(u * cols * 2), cols * 2)?,
                    x: self.down(bx.offset(u * h * 2), h * 2)?,
                })
            })
            .collect()
    }

    /// 2026-10-04: A state's drafter rows `[0, seq_len)`: latent KV (own pool) and indexer
    /// `k_normed`, `gate`, `valid`, with `seq_len` first.
    fn state_rows(&self, s: &mut dyn ProposerState) -> Result<Vec<u8>> {
        let v = self.head.debug_state_view(s).context("not a GLM state")?;
        let kp = v.own_k_pool.context("no own pool")?;
        let row = self.latent_row();
        let mut out = (v.seq_len as u64).to_le_bytes().to_vec();
        for p in 0..v.seq_len {
            let blk = v.block_table[p / 16] as usize;
            out.extend(self.down(kp.offset((blk * 16 + p % 16) * row), row)?);
        }
        let d = self.index_dim;
        out.extend(self.down(v.idx_k_normed, v.seq_len * d * 2)?);
        out.extend(self.down(v.idx_gate, v.seq_len * d * 2)?);
        out.extend(self.down(v.idx_valid, v.seq_len)?);
        Ok(out)
    }
}

/// 2026-10-04: Per-sequence rounds for every sequence in turn: one `Out` per (sequence, round),
/// and with `rows` each sequence's final `state_rows`.
fn serial_arm(
    rig: &Rig,
    sharded: bool,
    rows: bool,
    chunks: Option<&[usize]>,
) -> Result<(Vec<Vec<Out>>, Vec<Vec<u8>>)> {
    let mut outs = Vec::new();
    let mut finals = Vec::new();
    for u in 0..rig.plans.len() {
        let mut s = rig.fresh_with(u, sharded, chunks)?;
        let mut pos = rig.plans[u].prompt.len();
        let mut v = Vec::new();
        for r in 0..ND_SCHEDULE.len() {
            v.push(rig.serial_round(u, r, pos, s.as_mut(), sharded)?);
            let a = rig.plans[u].rounds[r].accepted;
            rig.head
                .after_verify(a, s.as_mut(), rig.g().default_stream())?;
            pos += a + 1;
        }
        if rows {
            finals.push(rig.state_rows(s.as_mut())?);
        }
        rig.head.free_state(rig.g(), s.as_mut())?;
        outs.push(v);
    }
    Ok((outs, finals))
}

fn ref_bytes(outs: &[Vec<Out>]) -> Vec<u8> {
    outs.iter().flatten().flat_map(|o| o.bytes()).collect()
}

/// 2026-10-04: The REF arm (module docs), run in a child with both levers off.
fn child(out_path: &str) -> Result<()> {
    let modules = ptx();
    let backend = MetraleCudaBackend::new(0, &modules)?;
    let rig = rig(&backend)?;
    let mut all = Vec::new();
    for sharded in [false, true] {
        let (outs, _) = serial_arm(&rig, sharded, false, None)?;
        all.extend(ref_bytes(&outs));
    }
    std::fs::write(out_path, all)?;
    Ok(())
}

fn ptx() -> Vec<(&'static str, &'static [u8])> {
    metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules)
}

fn rig(backend: &MetraleCudaBackend) -> Result<Rig<'_>> {
    let gpu: &dyn GpuBackend = backend;
    let geo = st::geometry(include_str!(
        "../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json"
    ))?;
    let lens = st::env_list("MTPB_LENS", "37,2300,120,2600");
    if lens.len() != 4 {
        bail!("MTPB_LENS needs four lengths, got {lens:?}");
    }
    let sets = st::env_usize("MTPB_EXPERT_SETS", 144);
    let h = geo.config.hidden_size;
    let head = build_head(gpu, &geo, sets)?;
    let plans: Vec<Plan> = lens
        .iter()
        .enumerate()
        .map(|(u, &l)| plan(l, h, geo.config.vocab_size, 0x5EED_0000 + u as u64))
        .collect();
    let max_len = *lens.iter().max().unwrap_or(&1);
    Ok(Rig {
        env: Env {
            gpu,
            buffers: BufferArena::new(&geo.config, 64, 8192, 16, 8, gpu)?,
            config: geo.config.clone(),
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            levers: ModelLevers::defaults(),
            stats: ModelStats::new(),
            single: SingleGpuBackend,
        },
        head,
        plans,
        h,
        kv_lora: geo.dsa.kv_lora_rank,
        index_dim: geo.dsa.index_head_dim,
        hid_in: gpu.alloc(MTP_BATCH_DRAFT_MAX * h * 2)?,
        prompt_hid: gpu.alloc(max_len * h * 2)?,
    })
}

/// 2026-10-04: `a == b` bytewise; prints the first differing index otherwise.
fn same(what: &str, a: &[u8], b: &[u8]) -> bool {
    if a == b {
        return true;
    }
    let i = a
        .iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()));
    println!(
        "  MISMATCH {what}: len {} vs {}, first differing byte {i}",
        a.len(),
        b.len()
    );
    false
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// 2026-10-04: Append `b` to `out` as a u64-LE-length-prefixed section.
fn put(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u64).to_le_bytes());
    out.extend_from_slice(b);
}

/// 2026-10-04: The next section `put` wrote.
fn take(inp: &mut &[u8]) -> Result<Vec<u8>> {
    let Some(head) = inp.get(..8) else {
        bail!("CTX output truncated (no section header)");
    };
    let n = u64::from_le_bytes(head.try_into()?) as usize;
    let Some(body) = inp.get(8..8 + n) else {
        bail!("CTX output truncated (section of {n} bytes)");
    };
    let v = body.to_vec();
    *inp = &inp[8 + n..];
    Ok(v)
}

/// 2026-10-04: Median wall ms of `reps` context writes of sequence `u` (`fresh_timed`).
fn prefill_ms(rig: &Rig, u: usize, chunks: Option<&[usize]>, reps: usize) -> Result<f64> {
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (mut s, secs) = rig.fresh_timed(u, true, chunks)?;
        v.push(secs * 1e3);
        rig.head.free_state(rig.g(), s.as_mut())?;
    }
    Ok(median(v))
}

/// 2026-10-04: The CTX leg (module docs), in a child with `METRALE_GLM_MTP_BATCH_DRAFT=1`
/// (so `METRALE_GLM_MTP_SEQ_KV`: every state its own pool) and `METRALE_GLM_MTP_CTX_ROWBATCH=1`.
/// Sections, in order: A (state rows after the <=16-row drain) and B (state rows after the
/// 1000-row drain) per sequence; C, the serial rounds after the A drain (sharded context) and
/// each sequence's final rows; D, the batched rounds (N = all sequences) after the A drain, one
/// section per (round, sequence), then each final; T, the median ms of the B and A drains of
/// the longest sequence.
fn ctx_child(out_path: &str) -> Result<()> {
    let modules = ptx();
    let backend = MetraleCudaBackend::new(0, &modules)?;
    let rig = rig(&backend)?;
    if !rig.head.debug_ctx_rowbatch() {
        bail!("CTX child: row-batched context scratch missing ({CTX_LEVER} or its kernel)");
    }
    if rig.head.debug_batch_bufs().is_none() {
        bail!("CTX child: batched-propose scratch missing (lever or kernels)");
    }
    let n = rig.plans.len();
    let mut out = Vec::new();
    let big = [BIG_CHUNK];
    for chunks in [&SMALL_CHUNKS[..], &big[..]] {
        for u in 0..n {
            let mut s = rig.fresh_with(u, true, Some(chunks))?;
            put(&mut out, &rig.state_rows(s.as_mut())?);
            rig.head.free_state(rig.g(), s.as_mut())?;
        }
    }
    let (outs, finals) = serial_arm(&rig, true, true, Some(&SMALL_CHUNKS[..]))?;
    put(&mut out, &ref_bytes(&outs));
    for f in &finals {
        put(&mut out, f);
    }
    let mut states: Vec<Box<dyn ProposerState>> = (0..n)
        .map(|u| rig.fresh_with(u, true, Some(&SMALL_CHUNKS[..])))
        .collect::<Result<_>>()?;
    let mut pos: Vec<usize> = (0..n).map(|u| rig.plans[u].prompt.len()).collect();
    for r in 0..ND_SCHEDULE.len() {
        for o in rig.batch_round(r, &pos, &mut states, true)? {
            put(&mut out, &o.bytes());
        }
        for (u, s) in states.iter_mut().enumerate() {
            let a = rig.plans[u].rounds[r].accepted;
            rig.head
                .after_verify(a, s.as_mut(), rig.g().default_stream())?;
            pos[u] += a + 1;
        }
    }
    for s in states.iter_mut() {
        put(&mut out, &rig.state_rows(s.as_mut())?);
        rig.head.free_state(rig.g(), s.as_mut())?;
    }
    let longest = longest_seq(&rig);
    put(
        &mut out,
        &prefill_ms(&rig, longest, Some(&[BIG_CHUNK][..]), 3)?.to_le_bytes(),
    );
    put(
        &mut out,
        &prefill_ms(&rig, longest, Some(&SMALL_CHUNKS[..]), 3)?.to_le_bytes(),
    );
    std::fs::write(out_path, out)?;
    Ok(())
}

fn longest_seq(rig: &Rig) -> usize {
    (0..rig.plans.len())
        .max_by_key(|&u| rig.plans[u].prompt.len())
        .unwrap_or(0)
}

/// 2026-10-04: E4M3 (finite variant) byte to f32.
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    if e == 0xF && b & 0x7 == 0x7 {
        return f32::NAN;
    }
    if e == 0 {
        sign * (m / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// 2026-10-04: Per-row (min cosine, max |d| / row absmax) of two `[rows, width]` buffers decoded
/// by `dec`; a row that is zero in both counts as cosine 1, a nonfinite value as cosine 0.
fn rows_stats(a: &[f32], b: &[f32], width: usize) -> (f64, f64) {
    let (mut cmin, mut rmax) = (1.0f64, 0.0f64);
    for (ra, rb) in a.chunks(width).zip(b.chunks(width)) {
        let (mut d, mut na, mut nb, mut amax, mut dmax) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (&x, &y) in ra.iter().zip(rb) {
            let (x, y) = (x as f64, y as f64);
            if !x.is_finite() || !y.is_finite() {
                return (0.0, f64::INFINITY);
            }
            d += x * y;
            na += x * x;
            nb += y * y;
            amax = amax.max(x.abs());
            dmax = dmax.max((x - y).abs());
        }
        let c = if na == 0.0 && nb == 0.0 {
            1.0
        } else if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            d / (na.sqrt() * nb.sqrt())
        };
        cmin = cmin.min(c);
        if amax > 0.0 {
            rmax = rmax.max(dmax / amax);
        } else if dmax > 0.0 {
            rmax = f64::INFINITY;
        }
    }
    (cmin, rmax)
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// 2026-10-04: CTX leg B: the 256-row-tile drafter rows `got` against the per-row walk's
/// `want` (both `state_rows`). Exact: length and `valid`. Tolerance (eh_proj runs as one GEMM
/// over a tile, a bf16 reduction-order change; the bounds of
/// `glm5next_mtp_ctx_rowbatch_microtest` leg C): `k_normed` and `gate` per-row cosine >= 0.9999
/// and max|d|/absmax <= 0.05; latent FP8 bytes differing <= 20 % and per-row cosine >= 0.99.
/// A context write that went to any pool but the sequence's own leaves its own rows empty, which
/// fails every bound.
fn ctx_tolerance(
    what: &str,
    want: &[u8],
    got: &[u8],
    row: usize,
    kv_lora: usize,
    d: usize,
) -> bool {
    let split = |b: &[u8]| -> Option<(usize, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
        let l = u64::from_le_bytes(b.get(..8)?.try_into().ok()?) as usize;
        let mut o = 8;
        let mut cut = |n: usize| -> Option<Vec<u8>> {
            let v = b.get(o..o + n)?.to_vec();
            o += n;
            Some(v)
        };
        let lat = cut(l * row)?;
        let kn = cut(l * d * 2)?;
        let gt = cut(l * d * 2)?;
        let va = cut(l)?;
        (o == b.len()).then_some((l, lat, kn, gt, va))
    };
    let (Some(w), Some(g)) = (split(want), split(got)) else {
        println!("  MISMATCH {what}: unparsable state rows");
        return false;
    };
    if w.0 != g.0 || w.4 != g.4 {
        println!(
            "  MISMATCH {what}: rows {} vs {} or valid bytes differ",
            w.0, g.0
        );
        return false;
    }
    let (kc, ke) = rows_stats(&bf16s(&w.2), &bf16s(&g.2), d);
    let (gc, ge) = rows_stats(&bf16s(&w.3), &bf16s(&g.3), d);
    let diff = w.1.iter().zip(&g.1).filter(|(a, b)| a != b).count();
    let frac = diff as f64 / w.1.len().max(1) as f64;
    let lc = if row == kv_lora {
        let f = |b: &[u8]| b.iter().map(|&x| e4m3(x)).collect::<Vec<f32>>();
        rows_stats(&f(&w.1), &f(&g.1), row).0
    } else {
        f64::NAN
    };
    let ok = kc >= 0.9999
        && ke <= 0.05
        && gc >= 0.9999
        && ge <= 0.05
        && frac <= 0.20
        && (row != kv_lora || lc >= 0.99);
    println!(
        "  {what}: rows={} latent bytes differing={diff}/{} ({:.4}%) latent min_cos={lc:.6} \
         k_normed min_cos={kc:.6} rel={ke:.4} gate min_cos={gc:.6} rel={ge:.4} {}",
        w.0,
        w.1.len(),
        100.0 * frac,
        if ok { "ok" } else { "BAD" }
    );
    ok
}

fn main() -> Result<()> {
    if let Ok(path) = std::env::var("MTPB_REF_OUT") {
        return child(&path);
    }
    if let Ok(path) = std::env::var("MTPB_CTX_OUT") {
        return ctx_child(&path);
    }
    // 2026-10-04: REF first, in a child with both levers off, before this process holds any
    // device memory.
    let ref_path = std::env::temp_dir().join(format!("mtpb-ref-{}.bin", std::process::id()));
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.env("MTPB_REF_OUT", &ref_path);
    for l in LEVERS {
        cmd.env_remove(l);
    }
    cmd.env_remove(CTX_LEVER);
    let t0 = Instant::now();
    let status = cmd.status()?;
    if !status.success() {
        bail!("REF child exited {status}");
    }
    let ref_all = std::fs::read(&ref_path)?;
    let _ = std::fs::remove_file(&ref_path);
    println!(
        "REF arm (levers off, shared pool, per-sequence) done in {:.1} s, {} bytes",
        t0.elapsed().as_secs_f64(),
        ref_all.len()
    );

    // 2026-10-04: CTX leg (module docs): a child with SEQ_KV (via the batch lever) and the
    // row-batched context write on, also before this process holds device memory.
    let ctx_path = std::env::temp_dir().join(format!("mtpb-ctx-{}.bin", std::process::id()));
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.env("MTPB_CTX_OUT", &ctx_path);
    cmd.env("METRALE_GLM_MTP_BATCH_DRAFT", "1");
    cmd.env(CTX_LEVER, "1");
    let t0 = Instant::now();
    let status = cmd.status()?;
    if !status.success() {
        bail!("CTX child exited {status}");
    }
    let ctx_all = std::fs::read(&ctx_path)?;
    let _ = std::fs::remove_file(&ctx_path);
    println!(
        "CTX arm (levers on + {CTX_LEVER}=1, own pools, chunked drains) done in {:.1} s, {} bytes",
        t0.elapsed().as_secs_f64(),
        ctx_all.len()
    );

    // 2026-10-04: SAFETY: single-threaded here, before anything reads the environment. This
    // process is the per-row context walk, so the row-batch lever is cleared.
    unsafe {
        std::env::set_var("METRALE_GLM_MTP_BATCH_DRAFT", "1");
        std::env::remove_var(CTX_LEVER);
    }
    let modules = ptx();
    let backend = MetraleCudaBackend::new(0, &modules)?;
    let rig = rig(&backend)?;
    if rig.head.debug_batch_bufs().is_none() {
        bail!("batched-propose scratch missing (lever or kernels)");
    }
    if rig.head.debug_ctx_rowbatch() {
        bail!("{CTX_LEVER} is on in the per-row process");
    }
    let lens: Vec<usize> = rig.plans.iter().map(|p| p.prompt.len() - 1).collect();
    println!(
        "prefill lens={lens:?}, rounds nd={ND_SCHEDULE:?}, vocab={} head cols sharded={}",
        rig.env.config.vocab_size,
        rig.head.debug_sweep_cols(true)
    );

    let mut bad = 0usize;
    let mut ref_off = 0usize;
    let mut sharded_serial: Option<(Vec<Vec<Out>>, Vec<Vec<u8>>)> = None;
    for sharded in [false, true] {
        let arm = if sharded { "sharded-fp8" } else { "bf16-full" };
        let (serial, finals) = serial_arm(&rig, sharded, true, None)?;
        if sharded {
            sharded_serial = Some((serial.clone(), finals.clone()));
        }
        // 2026-10-04: SERIAL vs REF.
        let mine = ref_bytes(&serial);
        let theirs = ref_all.get(ref_off..ref_off + mine.len()).unwrap_or(&[]);
        ref_off += mine.len();
        let ok = same(&format!("{arm} serial-vs-ref"), &mine, theirs);
        bad += usize::from(!ok);
        println!("  ctx={arm} SERIAL(levers on) vs REF(levers off): bitwise_equal={ok}");
        let accepted_total: usize = rig
            .plans
            .iter()
            .flat_map(|p| p.rounds.iter().map(|r| r.accepted))
            .sum();
        println!("  ctx={arm} accepted counts total={accepted_total} (varied per sequence)");

        for n in [2usize, 4] {
            let mut states: Vec<Box<dyn ProposerState>> = (0..n)
                .map(|u| rig.fresh(u, sharded))
                .collect::<Result<_>>()?;
            let mut pos: Vec<usize> = (0..n).map(|u| rig.plans[u].prompt.len()).collect();
            let mut eq_rounds = 0usize;
            for r in 0..ND_SCHEDULE.len() {
                let outs = rig.batch_round(r, &pos, &mut states, sharded)?;
                let mut round_ok = true;
                for (u, o) in outs.iter().enumerate() {
                    let s = &serial[u][r];
                    let ok = o.drafts == s.drafts
                        && same(
                            &format!("{arm} n={n} r={r} u={u} logits"),
                            &o.logits,
                            &s.logits,
                        )
                        && same(&format!("{arm} n={n} r={r} u={u} x"), &o.x, &s.x);
                    if !ok {
                        println!(
                            "  MISMATCH {arm} n={n} round={r} seq={u}: drafts batched={:?} serial={:?}",
                            o.drafts, s.drafts
                        );
                    }
                    round_ok &= ok;
                }
                bad += usize::from(!round_ok);
                eq_rounds += usize::from(round_ok);
                for (u, s) in states.iter_mut().enumerate() {
                    let a = rig.plans[u].rounds[r].accepted;
                    rig.head
                        .after_verify(a, s.as_mut(), rig.g().default_stream())?;
                    pos[u] += a + 1;
                }
            }
            let mut rows_ok = 0usize;
            for (u, s) in states.iter_mut().enumerate() {
                let got = rig.state_rows(s.as_mut())?;
                if same(
                    &format!("{arm} n={n} seq={u} drafter rows"),
                    &got,
                    &finals[u],
                ) {
                    rows_ok += 1;
                } else {
                    bad += 1;
                }
                rig.head.free_state(rig.g(), s.as_mut())?;
            }
            println!(
                "  ctx={arm} n={n} BATCHED vs SERIAL: rounds bitwise_equal {eq_rounds}/{} \
                 (drafts+logits+x per sequence), drafter KV+indexer rows equal {rows_ok}/{n}",
                ND_SCHEDULE.len()
            );
        }
    }
    if ref_off != ref_all.len() {
        println!(
            "  MISMATCH REF length {} vs consumed {ref_off}",
            ref_all.len()
        );
        bad += 1;
    }

    // 2026-10-04: CTX leg comparisons (module docs) against this process's per-row context
    // walk under the same levers.
    {
        let (serial, finals) = sharded_serial.context("no sharded serial arm")?;
        let n = rig.plans.len();
        let mut inp: &[u8] = &ctx_all;
        let mut pre = Vec::with_capacity(n);
        for u in 0..n {
            let mut s = rig.fresh(u, true)?;
            pre.push(rig.state_rows(s.as_mut())?);
            rig.head.free_state(rig.g(), s.as_mut())?;
        }
        let mut a_ok = 0usize;
        for (u, want) in pre.iter().enumerate() {
            let got = take(&mut inp)?;
            if same(&format!("CTX A seq={u} drafter rows"), &got, want) {
                a_ok += 1;
            } else {
                bad += 1;
            }
        }
        println!(
            "  CTX A ({CTX_LEVER}=1, drain chunks <=16 rows via prefill_drafter +              catchup_drafter, own pools) vs per-row walk: drafter KV+indexer rows bitwise equal              {a_ok}/{n}"
        );
        let (row, kvr, d) = (rig.latent_row(), rig.kv_lora, rig.index_dim);
        let mut b_ok = 0usize;
        for (u, want) in pre.iter().enumerate() {
            let got = take(&mut inp)?;
            if ctx_tolerance(&format!("CTX B seq={u}"), want, &got, row, kvr, d) {
                b_ok += 1;
            } else {
                bad += 1;
            }
        }
        println!(
            "  CTX B ({CTX_LEVER}=1, {BIG_CHUNK}-row drain chunks, 256-row tiles, eh_proj GEMM)              vs per-row walk: within tolerance {b_ok}/{n}"
        );
        let c = take(&mut inp)?;
        let mut c_ok = same("CTX C serial rounds", &c, &ref_bytes(&serial));
        for (u, f) in finals.iter().enumerate() {
            c_ok &= same(&format!("CTX C seq={u} final rows"), &take(&mut inp)?, f);
        }
        bad += usize::from(!c_ok);
        println!(
            "  CTX C serial propose after the A drain vs after the per-row walk (sharded):              bitwise_equal={c_ok}"
        );
        let mut d_ok = true;
        for r in 0..ND_SCHEDULE.len() {
            for (u, ser) in serial.iter().enumerate() {
                let got = take(&mut inp)?;
                d_ok &= same(&format!("CTX D r={r} seq={u}"), &got, &ser[r].bytes());
            }
        }
        for (u, f) in finals.iter().enumerate() {
            d_ok &= same(&format!("CTX D seq={u} final rows"), &take(&mut inp)?, f);
        }
        bad += usize::from(!d_ok);
        println!(
            "  CTX D batched propose (N={n}) after the A drain vs serial after the per-row walk              (sharded): bitwise_equal={d_ok}"
        );
        let f = |b: Vec<u8>| -> Result<f64> {
            Ok(f64::from_le_bytes(
                b.as_slice().try_into().context("timing section")?,
            ))
        };
        let (ms_big, ms_small) = (f(take(&mut inp)?)?, f(take(&mut inp)?)?);
        if !inp.is_empty() {
            println!("  MISMATCH CTX output: {} trailing bytes", inp.len());
            bad += 1;
        }
        let longest = longest_seq(&rig);
        let rows = rig.plans[longest].prompt.len() - 1;
        let ms_row = prefill_ms(&rig, longest, None, 3)?;
        println!(
            "TIMING ctx rows={rows} perrow_ms={ms_row:.3} rowbatch_{BIG_CHUNK}chunk_ms={ms_big:.3}              speedup={:.2}x rowbatch_le16chunk_ms={ms_small:.3} (median of 3, context writes only,              separate processes)",
            ms_row / ms_big
        );
    }

    // 2026-10-04: Timing, sharded context, two drafts.
    let reps = st::env_usize("MTPB_REPS", 20).max(1);
    let nd = 2usize;
    for n in [2usize, 4] {
        let mut ser: Vec<Box<dyn ProposerState>> =
            (0..n).map(|u| rig.fresh(u, true)).collect::<Result<_>>()?;
        let mut bat: Vec<Box<dyn ProposerState>> =
            (0..n).map(|u| rig.fresh(u, true)).collect::<Result<_>>()?;
        let mut pos_s: Vec<usize> = (0..n).map(|u| rig.plans[u].prompt.len()).collect();
        let mut pos_b = pos_s.clone();
        let tokens: Vec<u32> = (0..n).map(|u| rig.plans[u].rounds[0].token).collect();
        let (g, h) = (rig.g(), rig.h);
        g.synchronize(g.default_stream())?;
        for u in 0..n {
            g.copy_h2d(&rig.plans[u].rounds[0].hidden, rig.hid_in.offset(u * h * 2))?;
        }
        let hiddens: Vec<DevicePtr> = (0..n).map(|u| rig.hid_in.offset(u * h * 2)).collect();
        let ctx = rig.env.ctx(true);
        let (mut ts, mut tb) = (Vec::new(), Vec::new());
        for rep in 0..reps + 2 {
            g.synchronize(g.default_stream())?;
            let t = Instant::now();
            for u in 0..n {
                rig.head.propose(
                    tokens[u],
                    hiddens[u],
                    pos_s[u],
                    nd,
                    ser[u].as_mut(),
                    &ctx,
                    g.default_stream(),
                    None,
                    None,
                    None,
                )?;
            }
            g.synchronize(g.default_stream())?;
            let dt_s = t.elapsed().as_secs_f64();
            let mut refs: Vec<&mut dyn ProposerState> = Vec::with_capacity(n);
            for s in bat.iter_mut() {
                refs.push(s.as_mut());
            }
            g.synchronize(g.default_stream())?;
            let t = Instant::now();
            let got = rig.head.propose_batch(
                &tokens,
                &hiddens,
                &pos_b,
                nd,
                &mut refs,
                &ctx,
                g.default_stream(),
                None,
            )?;
            g.synchronize(g.default_stream())?;
            let dt_b = t.elapsed().as_secs_f64();
            if got.is_none() {
                bail!("timing: propose_batch declined at rep {rep}");
            }
            // 2026-10-04: Keep both drafts each rep, so both arms advance alike.
            for u in 0..n {
                rig.head
                    .after_verify(nd - 1, ser[u].as_mut(), g.default_stream())?;
                rig.head
                    .after_verify(nd - 1, bat[u].as_mut(), g.default_stream())?;
                pos_s[u] += nd;
                pos_b[u] += nd;
            }
            if rep >= 2 {
                ts.push(dt_s);
                tb.push(dt_b);
            }
        }
        let (ms_s, ms_b) = (median(ts) * 1e3, median(tb) * 1e3);
        println!(
            "TIMING n={n} nd={nd} perseq_ms={ms_s:.3} batched_ms={ms_b:.3} speedup={:.2}x \
             perseq_per_pass_ms={:.3} batched_per_pass_ms={:.3} (median of {reps}; one-rank comm, \
             no transfer)",
            ms_s / ms_b,
            ms_s / (n * nd) as f64,
            ms_b / nd as f64
        );
        for s in ser.iter_mut().chain(bat.iter_mut()) {
            rig.head.free_state(g, s.as_mut())?;
        }
    }

    if bad > 0 {
        println!("FAIL: {bad} comparison(s) differ");
        bail!("glm5next_mtp_batch_draft_microtest: {bad} mismatch(es)");
    }
    println!(
        "PASS: batched MTP propose bitwise equal to per-sequence propose (N=2, N=4, nd 1..3, \
         both head sweeps), per-sequence propose with levers on equal to levers off, and the \
         {CTX_LEVER}=1 context write into own pools equal to the per-row walk (bitwise at <=16-row \
         drains incl. batched propose after it, within tolerance at 256-row tiles)"
    );
    Ok(())
}
