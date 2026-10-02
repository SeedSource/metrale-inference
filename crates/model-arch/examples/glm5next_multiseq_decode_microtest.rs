// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: GPU check of the GLM-5.3 batched multi-sequence decode
//! (`METRALE_GLM_DECODE_MULTI_SEQ`, `glm5next_layer/steps/multi_seq.rs`; ported from rsafier's
//! Atlas e69446eee, acf792e28 and 04beaac1f).
//!
//! A three-layer stack at the rank-0 TP=2/EP=2 geometry (KDA + dense MLP, DSA + routed MoE,
//! KDA + routed MoE; random weights, `common/glm5next_multiseq_stack.rs`) holds four sequences,
//! prefilled to `GLM_MS_LENS` (default 37, 2300, 120, 2600: two past the DSA top-k of 2048).
//!
//! Correctness, N = 2 (sequences 0 and 1) and N = 4, for `GLM_MS_STEPS` decode steps each:
//! - SOLO: each sequence alone through `TransformerLayer::decode` with its own one-row
//!   metadata (the per-sequence route the lever replaces).
//! - BATCHED: the same inputs and starting states through `decode_multi_seq` in one call.
//! Per row it compares the output hidden, the DSA latent KV slot and the KDA states, and prints
//! IDENTICAL (bytes equal), DRIFT (close to its own solo row, rel L2 < 1e-2) or CONTAMINATED
//! (closer to another sequence's solo row than to its own, or far from its own). DRIFT fails
//! unless `GLM_MS_ALLOW_DRIFT=1`; CONTAMINATED and non-finite output always fail.
//!
//! Timing, N = 1, 2, 4: per-layer synchronized time of the per-sequence loop and of the batched
//! call (median of `GLM_MS_REPS`), projected over the checkpoint's layer mix to one rank's
//! decode step. The projection leaves out the TP all-reduces, embedding, LM head and sampling,
//! and both arms run eager (production single-sequence decode replays CUDA graphs).
//!
//! Run on one GB10: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples
//! --example glm5next_multiseq_decode_microtest`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#[path = "common/glm5next_multiseq_stack.rs"]
mod glm5next_multiseq_stack;

use anyhow::{Result, bail};
use glm5next_multiseq_stack as st;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_layer::Glm5NextLayer;
use metrale_model_arch::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer, Mlp};
use metrale_model_layers::layer::{
    AttnMetadataDev, ForwardContext, LayerState, MoeLoraRoute, SsmLayerState, TransformerLayer,
};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use std::time::Instant;

/// 2026-10-01: Prompt tokens per prefill call; the arena's highway holds this many slots.
const CHUNK: usize = 64;

/// 2026-10-01: What every `ForwardContext` borrows.
struct Env<'g> {
    gpu: &'g dyn GpuBackend,
    config: ModelConfig,
    buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    levers: ModelLevers,
    stats: ModelStats,
}

impl Env<'_> {
    fn ctx(
        &self,
        attn_metadata: Option<AttnMetadataDev>,
        decode_step: bool,
    ) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step,
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

/// 2026-10-01: Each KDA layer's `(h_state, conv_state)` of `seq`, with their byte sizes.
fn kda_bufs(seq: &mut st::Seq, h_bytes: usize, c_bytes: usize) -> Vec<(DevicePtr, usize)> {
    let mut v = Vec::new();
    for s in seq.states.iter_mut() {
        if let Some(x) = s.as_any_mut().downcast_mut::<SsmLayerState>() {
            v.push((x.h_state, h_bytes));
            v.push((x.conv_state, c_bytes));
        }
    }
    v
}

/// 2026-10-01: Everything one decode step leaves behind for one sequence.
#[derive(PartialEq)]
struct Trace {
    out: Vec<u8>,
    kv: Vec<u8>,
    kda: Vec<Vec<u8>>,
}

struct Rig<'g> {
    env: Env<'g>,
    stack: Vec<Glm5NextLayer>,
    kv: PagedKvCache,
    meta: st::MetaBufs,
    hid: DevicePtr,
    h: usize,
    kv_lora: usize,
    h_bytes: usize,
    c_bytes: usize,
}

impl Rig<'_> {
    fn g(&self) -> &dyn GpuBackend {
        self.env.gpu
    }

    fn kv_slot(&self, seq: &st::Seq) -> DevicePtr {
        let p = seq.len;
        let slot = seq.blocks[p / st::BLOCK] as usize * st::BLOCK + p % st::BLOCK;
        self.kv.k_pool_ptr(0).offset(slot * self.kv_lora)
    }

    fn trace(&self, seq: &mut st::Seq, row: usize) -> Result<Trace> {
        let g = self.g();
        let out = st::down(g, self.hid.offset(row * self.h * 2), self.h * 2)?;
        let kv = st::down(g, self.kv_slot(seq), self.kv_lora)?;
        let kda = kda_bufs(seq, self.h_bytes, self.c_bytes)
            .into_iter()
            .map(|(p, n)| st::down(g, p, n))
            .collect::<Result<_>>()?;
        Ok(Trace { out, kv, kda })
    }

    fn restore(&self, seq: &mut st::Seq, kda: &[Vec<u8>]) -> Result<()> {
        let g = self.g();
        g.synchronize(0)?;
        for ((p, _), b) in kda_bufs(seq, self.h_bytes, self.c_bytes).iter().zip(kda) {
            g.copy_h2d(b, *p)?;
        }
        Ok(())
    }

    fn put_rows(&self, rows: &[Vec<u8>]) -> Result<()> {
        self.g().synchronize(0)?;
        for (r, b) in rows.iter().enumerate() {
            self.g().copy_h2d(b, self.hid.offset(r * self.h * 2))?;
        }
        Ok(())
    }

    /// 2026-10-01: Layer `l` for `seq` alone, as the per-sequence route runs it.
    fn solo_layer(&mut self, l: usize, seq: &mut st::Seq, meta: AttnMetadataDev) -> Result<()> {
        let ctx = self.env.ctx(Some(meta), true);
        let mut bt = seq.blocks.clone();
        let (mut d0, mut d1) = (Vec::new(), Vec::new());
        self.stack[l].decode(
            self.hid,
            self.env.buffers.residual(),
            seq.states[l].as_mut(),
            &mut self.kv,
            seq.len,
            &mut bt,
            &mut d0,
            &mut d1,
            &ctx,
            0,
        )
    }

    /// 2026-10-01: Layer `l` for `seqs` in one `decode_multi_seq` call.
    fn batch_layer(
        &mut self,
        l: usize,
        seqs: &mut [st::Seq],
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let ctx = self.env.ctx(Some(meta), true);
        let lens: Vec<usize> = seqs.iter().map(|s| s.len).collect();
        let bts: Vec<Vec<u32>> = seqs.iter().map(|s| s.blocks.clone()).collect();
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            seqs.iter_mut().map(|s| &mut *s.states[l]).collect();
        let n = refs.len();
        self.stack[l].decode_multi_seq(
            self.hid,
            self.env.buffers.residual(),
            n,
            &mut refs,
            &mut self.kv,
            &lens,
            &bts,
            &ctx,
            0,
        )
    }

    fn prefill(&mut self, seq: &mut st::Seq, len: usize, rng: &mut st::Lcg) -> Result<()> {
        let mut t = 0;
        while t < len {
            let k = CHUNK.min(len - t);
            self.put_rows(&[st::bf16_bytes(&rng.vec(k * self.h, 1.0))])?;
            let ctx = self.env.ctx(None, false);
            for l in 0..self.stack.len() {
                let (mut d0, mut d1) = (Vec::new(), Vec::new());
                self.stack[l].prefill(
                    self.hid,
                    self.env.buffers.residual(),
                    k,
                    seq.states[l].as_mut(),
                    &mut self.kv,
                    t,
                    &mut seq.blocks,
                    &mut d0,
                    &mut d1,
                    0,
                    &ctx,
                    0,
                )?;
            }
            t += k;
        }
        seq.len = len;
        Ok(())
    }
}

/// 2026-10-01: One correctness step over the first `n` sequences. Returns the failures.
fn check_step(
    rig: &mut Rig,
    seqs: &mut [st::Seq],
    n: usize,
    step: usize,
    rng: &mut st::Lcg,
) -> Result<usize> {
    let inputs: Vec<Vec<u8>> = (0..n)
        .map(|_| st::bf16_bytes(&rng.vec(rig.h, 1.0)))
        .collect();
    let mut solo = Vec::with_capacity(n);
    for (i, seq) in seqs.iter_mut().take(n).enumerate() {
        let snap = kda_bufs(seq, rig.h_bytes, rig.c_bytes)
            .into_iter()
            .map(|(p, b)| st::down(rig.g(), p, b))
            .collect::<Result<Vec<_>>>()?;
        rig.put_rows(std::slice::from_ref(&inputs[i]))?;
        let meta = rig.meta.upload(rig.g(), &[&*seq])?;
        for l in 0..rig.stack.len() {
            rig.solo_layer(l, seq, meta)?;
        }
        solo.push(rig.trace(seq, 0)?);
        rig.restore(seq, &snap)?;
    }
    // 2026-10-01: Zero each row's KV slot so a batched DSA arm that skipped its latent write
    // cannot pass on the solo arm's bytes.
    for seq in seqs.iter().take(n) {
        rig.g().synchronize(0)?;
        rig.g().copy_h2d(&vec![0u8; rig.kv_lora], rig.kv_slot(seq))?;
    }
    rig.put_rows(&inputs)?;
    let refs: Vec<&st::Seq> = seqs.iter().take(n).collect();
    let meta = rig.meta.upload(rig.g(), &refs)?;
    for l in 0..rig.stack.len() {
        rig.batch_layer(l, &mut seqs[..n], meta)?;
    }
    let allow_drift = std::env::var("GLM_MS_ALLOW_DRIFT").as_deref() == Ok("1");
    let solo_f: Vec<Vec<f32>> = solo.iter().map(|t| st::row_f32(&t.out)).collect();
    let mut fails = 0;
    for (r, seq) in seqs.iter_mut().take(n).enumerate() {
        let got = rig.trace(seq, r)?;
        let f = st::row_f32(&got.out);
        let own = st::rel_l2(&f, &solo_f[r]);
        let nearest_other = (0..n)
            .filter(|&j| j != r)
            .map(|j| st::rel_l2(&f, &solo_f[j]))
            .fold(f64::INFINITY, f64::min);
        let finite = f.iter().all(|x| x.is_finite()) && f.iter().any(|x| *x != 0.0);
        let verdict = if !finite {
            "NONFINITE"
        } else if got == solo[r] {
            "IDENTICAL"
        } else if own < 1e-2 && own < nearest_other {
            "DRIFT"
        } else {
            "CONTAMINATED"
        };
        let bad = match verdict {
            "IDENTICAL" => false,
            "DRIFT" => !allow_drift,
            _ => true,
        };
        fails += usize::from(bad);
        println!(
            "  n={n} step={step} seq={r} len={} {verdict} out_rel_l2={own:.3e} \
             nearest_other={nearest_other:.3e} kv_equal={} kda_equal={}",
            seq.len,
            got.kv == solo[r].kv,
            got.kda == solo[r].kda
        );
        seq.len += 1;
    }
    Ok(fails)
}

/// 2026-10-01: Median per-layer seconds of `reps` steps over the first `n` sequences, per
/// sequence loop (`batched` false) or in one call. DSA states rewind on the next call
/// (`check_lockstep`), so the lengths stay put.
fn time_arm(
    rig: &mut Rig,
    seqs: &mut [st::Seq],
    n: usize,
    batched: bool,
    reps: usize,
) -> Result<Vec<f64>> {
    let layers = rig.stack.len();
    let refs: Vec<&st::Seq> = seqs.iter().take(n).collect();
    let meta = rig.meta.upload(rig.g(), &refs)?;
    let mut samples = vec![Vec::with_capacity(reps); layers];
    for rep in 0..reps + 1 {
        let mut rng = st::Lcg(0xD15C + rep as u64);
        let rows: Vec<Vec<u8>> = (0..n)
            .map(|_| st::bf16_bytes(&rng.vec(rig.h, 1.0)))
            .collect();
        rig.put_rows(&rows)?;
        for (l, sample) in samples.iter_mut().enumerate() {
            rig.g().synchronize(0)?;
            let t0 = Instant::now();
            if batched {
                rig.batch_layer(l, &mut seqs[..n], meta)?;
            } else {
                for (i, seq) in seqs.iter_mut().take(n).enumerate() {
                    rig.solo_layer(l, seq, meta.row_view(i))?;
                }
            }
            rig.g().synchronize(0)?;
            // 2026-10-01: Rep 0 is warm-up.
            if rep > 0 {
                sample.push(t0.elapsed().as_secs_f64());
            }
        }
    }
    Ok(samples
        .into_iter()
        .map(|mut v| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        })
        .collect())
}

fn main() -> Result<()> {
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let backend = MetraleCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let geo = st::geometry(include_str!(
        "../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json"
    ))?;
    let lens = st::env_list("GLM_MS_LENS", "37,2300,120,2600");
    if lens.len() != 4 {
        bail!("GLM_MS_LENS needs four lengths, got {lens:?}");
    }
    let steps = st::env_usize("GLM_MS_STEPS", 3);
    let reps = st::env_usize("GLM_MS_REPS", 10).max(1);
    let sets = st::env_usize("GLM_MS_EXPERT_SETS", 144);
    let mut rng = st::Lcg(0x5EED_0A11);
    let stack = st::build_stack(gpu, &geo, sets, &mut rng)?;

    // 2026-10-01: Each sequence owns a disjoint block range covering its prompt, the
    // correctness steps (sequences 0 and 1 step in both the N=2 and N=4 rounds) and a spare.
    let need: Vec<usize> = lens
        .iter()
        .map(|l| (l + 2 * steps + 2).div_ceil(st::BLOCK))
        .collect();
    if need.iter().any(|n| *n > st::MAX_BLOCKS) {
        bail!("a length in {lens:?} needs more than {} blocks", st::MAX_BLOCKS);
    }
    let mut seqs = Vec::with_capacity(4);
    let mut first = 0u32;
    for n in &need {
        seqs.push(st::new_seq(gpu, &stack, first, *n)?);
        first += *n as u32;
    }
    let env = Env {
        gpu,
        buffers: BufferArena::new(&geo.config, CHUNK, 4096, st::BLOCK, 8, gpu)?,
        config: geo.config.clone(),
        dispatch: GemmDispatch::defaults(),
        derived: DerivedWeights::new(),
        levers: ModelLevers::defaults(),
        stats: ModelStats::new(),
    };
    let h = geo.config.hidden_size;
    let mut rig = Rig {
        env,
        stack,
        kv: st::kv_cache(gpu, &geo.dsa, first as usize)?,
        meta: st::MetaBufs::new(gpu, st::rows())?,
        hid: gpu.alloc(CHUNK * h * 2)?,
        h,
        kv_lora: geo.dsa.kv_lora_rank,
        h_bytes: geo.kda.recurrent_state_elems() * 4,
        c_bytes: geo.kda.conv_state_elems() * 4,
    };
    for (seq, len) in seqs.iter_mut().zip(&lens) {
        rig.prefill(seq, *len, &mut rng)?;
    }
    println!(
        "prefilled lens={lens:?} (DSA index_topk={}), {sets} expert weight sets",
        geo.dsa.index_topk
    );

    let mut fails = 0;
    for n in [2usize, 4] {
        for step in 0..steps {
            fails += check_step(&mut rig, &mut seqs, n, step, &mut rng)?;
        }
    }

    // 2026-10-01: Layer mix of the checkpoint for the projection: layer 0 of the stack times
    // KDA + dense, layer 1 DSA + MoE (and DSA + dense, if any), layer 2 KDA + MoE.
    let sk = Glm5NextTextSkeleton::from_config(&geo.config)?;
    let count = |mx: Mixer, ml: Mlp| {
        sk.layers
            .iter()
            .filter(|l| l.mixer == mx && l.mlp == ml)
            .count()
    };
    let (kd, km) = (count(Mixer::Kda, Mlp::Dense), count(Mixer::Kda, Mlp::RoutedMoe));
    let dsa = count(Mixer::Dsa, Mlp::Dense) + count(Mixer::Dsa, Mlp::RoutedMoe);
    let mut base_tps = 0.0;
    for n in [1usize, 2, 4] {
        let mut step_s = [0.0f64; 2];
        for (a, batched) in [false, true].into_iter().enumerate() {
            let t = time_arm(&mut rig, &mut seqs, n, batched, reps)?;
            step_s[a] = kd as f64 * t[0] + dsa as f64 * t[1] + km as f64 * t[2];
            println!(
                "TIMING n={n} arm={} kda_dense_us={:.1} dsa_moe_us={:.1} kda_moe_us={:.1} \
                 projected_step_ms={:.2} projected_tok_s={:.2}",
                if batched { "batched" } else { "perseq" },
                t[0] * 1e6,
                t[1] * 1e6,
                t[2] * 1e6,
                step_s[a] * 1e3,
                n as f64 / step_s[a]
            );
        }
        let tps = n as f64 / step_s[1];
        if n == 1 {
            base_tps = tps;
        }
        println!(
            "TIMING n={n} batched_vs_perseq={:.2}x aggregate_vs_n1={:.2}x \
             (layers kda+dense={kd} dsa={dsa} kda+moe={km}; one rank, no all-reduce/LM head)",
            step_s[0] / step_s[1],
            tps / base_tps
        );
    }
    if fails > 0 {
        println!("FAIL: {fails} row-step(s) diverged from their solo run");
        bail!("glm5next_multiseq_decode_microtest: {fails} failure(s)");
    }
    println!("PASS: batched decode matches each sequence run alone (N=2, N=4, {steps} steps)");
    Ok(())
}
