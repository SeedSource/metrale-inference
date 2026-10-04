// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: GPU A/B gate of `METRALE_GLM_DSA_XSEQ_BATCH` (`glm5next_dsa/layer/xseq.rs`):
//! one GLM-5.3 DSA layer at the rank-0 TP=2 geometry (random weights,
//! `common/glm5next_multiseq_stack.rs`), sequences prefilled to `GLM_XS_LENS` (default 37,
//! 2300, 120, 2600, 700, 1500: three past the DSA top-k of 2048).
//!
//! Per case, the SAME inputs and starting states go through
//! - OFF: the per-sequence loop the two call sites run (`decode_k` per sequence, each with
//!   the loop's metadata: `row_view(i)` for the multi-sequence decode, `verify_rows_view` for
//!   the batched verify), and
//! - ON: `Glm5NextDsaLayer::decode_xseq` with the arena attached,
//!
//! and every byte each arm leaves behind is compared raw (memcmp): the output rows (BF16),
//! the FP8 latent KV slot of every new row, and the indexer cache rows (`k_normed`, `gate`,
//! `valid`). Before each arm the new KV slots and indexer rows are filled with that arm's own
//! poison byte, so a row one arm never writes cannot compare equal. Cases: decode ks
//! [1,1,1,1]; verify ks [3,3,3,3], [3,1,2], [1,1,1,1]; verify [3,1,2] with `decode_step`
//! true; verify [3,3,3,3,3,3] (two groups: 15 + 3 rows). Two steps each.
//!
//! KERNEL leg: each of the four projections (`q_a_proj`, `q_absorb`, `kv_a_proj`, `o_absorb`)
//! through the layer's own dispatch at M in {2, 4, 6, 12, 16} rows against one M = 1 call per
//! row, bitwise per row.
//!
//! The process runs the whole gate with BF16 weights, then re-runs itself with
//! `METRALE_GLM_DENSE_FP8=1` (the four projections converted to FP8 copies, as the loader's
//! `register_layer` does), so one example covers both weight formats.
//!
//! TIMING: the DSA layer alone, eager, median of `GLM_XS_REPS` synchronized runs per arm, at
//! C = 4 (decode ks [1;4], verify ks [3;4]), and the projection over GLM-5.3's DSA layer count
//! for one rank's step.
//!
//! Run on one GB10: `cargo run -p metrale-model-arch --release --features cuda,gpu-examples
//! --example glm5next_dsa_xseq_microtest`.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#[path = "common/glm5next_dsa_xseq_rig.rs"]
mod glm5next_dsa_xseq_rig;
#[path = "common/glm5next_multiseq_stack.rs"]
mod glm5next_multiseq_stack;

use anyhow::{Result, bail};
use glm5next_dsa_xseq_rig::{
    CHUNK, Env, META_ROWS, Meta, Mode, POISON_OFF, POISON_ON, Rig, Seq, bf16_diff,
};
use glm5next_multiseq_stack as st;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;
use metrale_model_arch::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use metrale_model_arch::glm5next_dsa::layer::{
    DsaXseqArena, Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWorkspace,
};
use metrale_model_arch::glm5next_layer::dense_fp8 as df;
use metrale_model_arch::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::ops::{
    self, DenseMmKernels, DerivedWeights, GemmDispatch, ModelLevers, ModelStats,
};
use std::sync::Arc;
use std::time::Instant;

/// 2026-10-03: One case, `steps` steps over the first `ks.len()` sequences. Returns failures.
fn check_case(
    rig: &mut Rig,
    seqs: &mut [Seq],
    ks: &[usize],
    mode: Mode,
    decode_step: bool,
    steps: usize,
    rng: &mut st::Lcg,
    label: &str,
) -> Result<usize> {
    let n = ks.len();
    let rows: usize = ks.iter().sum();
    let mut fails = 0;
    for step in 0..steps {
        let seqs = &mut seqs[..n];
        let input = st::bf16_bytes(&rng.vec(rows * rig.h, 1.0));
        let m = rig.upload_meta(seqs, ks)?;
        let metas = Rig::seq_metas(&m, ks, mode);
        let hits0 = df::hits();

        rig.poison(seqs, ks, POISON_OFF)?;
        rig.g().copy_h2d(&input, rig.hid)?;
        rig.run_off(seqs, ks, &metas, decode_step)?;
        let off = rig.capture(seqs, ks)?;
        let hits1 = df::hits();

        rig.poison(seqs, ks, POISON_ON)?;
        rig.g().copy_h2d(&input, rig.hid)?;
        rig.run_on(seqs, ks, &metas, decode_step)?;
        let on = rig.capture(seqs, ks)?;
        let hits2 = df::hits();

        for (i, (a, b)) in off.iter().zip(&on).enumerate() {
            let out_eq = a.0 == b.0;
            let side_eq = a.1 == b.1;
            let finite = st::row_f32(&a.0).iter().all(|x| x.is_finite());
            let (nd, mx) = bf16_diff(&a.0, &b.0);
            let ok = out_eq && side_eq && finite;
            fails += usize::from(!ok);
            println!(
                "  {label} step={step} seq={i} len={} k={} {} out_bitwise_equal={out_eq} \
                 ({} B, {nd} differing bf16, max_abs {mx:.3e}) kv+indexer_bitwise_equal={side_eq} \
                 ({} B) finite={finite}",
                seqs[i].len,
                ks[i],
                if ok { "IDENTICAL" } else { "MISMATCH" },
                a.0.len(),
                a.1.len()
            );
        }
        if df::dense_fp8() {
            println!(
                "  {label} step={step} fp8 GEMV launches: off arm {} on arm {}",
                hits1 - hits0,
                hits2 - hits1
            );
            if hits2 == hits1 || hits1 == hits0 {
                println!("  {label}: an arm launched no FP8 GEMV under METRALE_GLM_DENSE_FP8=1");
                fails += 1;
            }
        }
        for (s, &k) in seqs.iter_mut().zip(ks) {
            s.len += k;
        }
    }
    Ok(fails)
}

/// 2026-10-03: The four projections through the layer's dispatch at M rows against M one-row
/// calls, bitwise per row. Returns failures.
fn kernel_leg(rig: &Rig, rng: &mut st::Lcg) -> Result<usize> {
    let g = rig.g();
    let c = rig.layer.cfg;
    let lat = c.local_heads * c.kv_lora_rank;
    let w = &rig.layer.weights;
    let kn = &rig.layer.kernels;
    let shapes = [
        ("q_a_proj", w.q_a_proj, c.q_lora_rank, c.hidden),
        ("q_absorb", w.q_absorb, lat, c.q_lora_rank),
        ("kv_a_proj", w.kv_a_proj, c.kv_lora_rank, c.hidden),
        ("o_absorb", w.o_absorb, c.hidden, lat),
    ];
    let proj =
        |a: DevicePtr, b: DevicePtr, out: DevicePtr, m: usize, n: usize, k: usize| match df::route(
            g, kn.gemv, a, b, out, m, n, k, 0,
        )? {
            df::Route::Done => Ok(()),
            df::Route::Weight(b) => ops::dense_mm_bf16(
                g,
                &DenseMmKernels {
                    gemm: kn.gemm,
                    gemv: kn.gemv,
                    batchm: kn.gemv_batchm,
                },
                a,
                b,
                out,
                m,
                n,
                k,
                0,
            ),
        };
    let mut fails = 0;
    for (name, wp, n, k) in shapes {
        let a = g.alloc(16 * k * 2)?;
        let (c1, cm) = (g.alloc(16 * n * 2)?, g.alloc(16 * n * 2)?);
        for m in [2usize, 4, 6, 12, 16] {
            g.synchronize(0)?;
            g.copy_h2d(&st::bf16_bytes(&rng.vec(m * k, 1.0)), a)?;
            g.copy_h2d(&vec![POISON_OFF; m * n * 2], c1)?;
            g.copy_h2d(&vec![POISON_ON; m * n * 2], cm)?;
            for r in 0..m {
                proj(a.offset(r * k * 2), wp, c1.offset(r * n * 2), 1, n, k)?;
            }
            proj(a, wp, cm, m, n, k)?;
            let (x, y) = (st::down(g, c1, m * n * 2)?, st::down(g, cm, m * n * 2)?);
            let same = (0..m)
                .filter(|r| x[r * n * 2..(r + 1) * n * 2] == y[r * n * 2..(r + 1) * n * 2])
                .count();
            let (nd, mx) = bf16_diff(&x, &y);
            let ok = same == m;
            fails += usize::from(!ok);
            println!(
                "  KERNEL {name} [{n}x{k}] M={m}: rows bitwise equal to the M=1 call {same}/{m} \
                 ({nd} differing bf16, max_abs {mx:.3e}) {}",
                if ok { "ok" } else { "MISMATCH" }
            );
        }
        for p in [a, c1, cm] {
            g.free(p)?;
        }
    }
    Ok(fails)
}

/// 2026-10-03: Median seconds of `reps` runs of one arm (rep 0 warm-up). The lengths stay put:
/// each run rewinds the indexer caches (`check_lockstep`) and rewrites the same KV slots.
fn time_arm(
    rig: &mut Rig,
    seqs: &mut [Seq],
    ks: &[usize],
    mode: Mode,
    on: bool,
    reps: usize,
) -> Result<f64> {
    let seqs = &mut seqs[..ks.len()];
    let m = rig.upload_meta(seqs, ks)?;
    let metas = Rig::seq_metas(&m, ks, mode);
    let decode_step = mode == Mode::Decode;
    let rows: usize = ks.iter().sum();
    let mut v = Vec::with_capacity(reps);
    let mut rng = st::Lcg(0x7135);
    for rep in 0..reps + 1 {
        rig.g().synchronize(0)?;
        rig.g()
            .copy_h2d(&st::bf16_bytes(&rng.vec(rows * rig.h, 1.0)), rig.hid)?;
        rig.g().synchronize(0)?;
        let t0 = Instant::now();
        if on {
            rig.run_on(seqs, ks, &metas, decode_step)?;
        } else {
            rig.run_off(seqs, ks, &metas, decode_step)?;
        }
        rig.g().synchronize(0)?;
        if rep > 0 {
            v.push(t0.elapsed().as_secs_f64());
        }
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Ok(v[v.len() / 2])
}

fn run_all() -> Result<usize> {
    let fp8 = df::dense_fp8();
    let tag = if fp8 { "fp8" } else { "bf16" };
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
    let lens = st::env_list("GLM_XS_LENS", "37,2300,120,2600,700,1500");
    if lens.len() != 6 {
        bail!("GLM_XS_LENS needs six lengths, got {lens:?}");
    }
    let steps = st::env_usize("GLM_XS_STEPS", 2);
    let reps = st::env_usize("GLM_XS_REPS", 20).max(1);
    let mut rng = st::Lcg(0x0D5A_75E0);
    let cfg = &geo.config;
    let mut layer = Glm5NextDsaLayer {
        persist_bt: true,
        cfg: geo.dsa,
        weights: st::dsa_weights(gpu, &geo.dsa, &mut rng)?,
        kernels: Glm5NextDsaLayerKernels::resolve(gpu)?,
        select_kernels: Glm5NextDsaKernels::resolve(gpu)?,
        decode_kernel: Glm5NextDsaDecodeKernel::resolve(gpu)?,
        workspace: Glm5NextDsaWorkspace::new(gpu, &geo.dsa, st::rows())?,
        layer_idx: 1,
        attn_layer_idx: 0,
        rms_eps: cfg.rms_norm_eps as f32,
        kv_scale: 1.0,
    };
    if fp8 {
        // 2026-10-03: The four projections `register_layer` converts for a DSA layer.
        let c = layer.cfg;
        let lat = c.local_heads * c.kv_lora_rank;
        let w = &mut layer.weights;
        let mut acc = df::LayerFp8::default();
        let mut all = true;
        all &= df::convert_weight(gpu, &mut w.q_a_proj, c.q_lora_rank, c.hidden, &mut acc)?;
        all &= df::convert_weight(gpu, &mut w.q_absorb, lat, c.q_lora_rank, &mut acc)?;
        all &= df::convert_weight(gpu, &mut w.kv_a_proj, c.kv_lora_rank, c.hidden, &mut acc)?;
        all &= df::convert_weight(gpu, &mut w.o_absorb, c.hidden, lat, &mut acc)?;
        let arena = df::finish_load(gpu)?;
        if !all {
            bail!("METRALE_GLM_DENSE_FP8=1 but a DSA projection was not converted");
        }
        println!(
            "[{tag}] converted q_a_proj, q_absorb, kv_a_proj, o_absorb to FP8 ({} B FP8, \
             dequant arena {arena} B)",
            acc.fp8_bytes
        );
    }
    let arena = Arc::new(DsaXseqArena::new(gpu, &geo.dsa, 16)?);
    // 2026-10-03: Each sequence owns a disjoint block range: its prompt plus every case's
    // steps (at most 3 rows a step, six cases) and a spare block.
    let growth = 6 * steps * 3 + st::BLOCK;
    let need: Vec<usize> = lens
        .iter()
        .map(|l| (l + growth).div_ceil(st::BLOCK))
        .collect();
    if need.iter().any(|n| *n > st::MAX_BLOCKS) {
        bail!(
            "a length in {lens:?} needs more than {} blocks",
            st::MAX_BLOCKS
        );
    }
    let total: usize = need.iter().sum();
    let h = cfg.hidden_size;
    let mut rig = Rig {
        env: Env {
            gpu,
            buffers: BufferArena::new(cfg, 64, 4096, st::BLOCK, 8, gpu)?,
            config: cfg.clone(),
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            levers: ModelLevers::defaults(),
            stats: ModelStats::new(),
        },
        kv: st::kv_cache(gpu, &geo.dsa, total)?,
        meta: Meta {
            pos: gpu.alloc(META_ROWS * 4)?,
            slot: gpu.alloc(META_ROWS * 8)?,
            sl: gpu.alloc(META_ROWS * 4)?,
            bt: gpu.alloc(META_ROWS * st::MAX_BLOCKS * 4)?,
        },
        hid: gpu.alloc(META_ROWS.max(CHUNK) * h * 2)?,
        h,
        kv_lora: geo.dsa.kv_lora_rank,
        d: geo.dsa.index_head_dim,
        arena,
        layer,
    };
    let mut seqs = Vec::with_capacity(6);
    let mut first = 0u32;
    for n in &need {
        seqs.push(Seq {
            len: 0,
            blocks: (first..first + *n as u32).collect(),
            state: rig.layer.alloc_state(gpu)?,
        });
        first += *n as u32;
    }
    for (s, len) in seqs.iter_mut().zip(&lens) {
        rig.prefill(s, *len, &mut rng)?;
    }
    println!(
        "[{tag}] prefilled lens={lens:?} (DSA index_topk={}, hidden {h}, local heads {}, \
         workspace rows {})",
        geo.dsa.index_topk,
        geo.dsa.local_heads,
        st::rows()
    );

    let mut fails = kernel_leg(&rig, &mut rng)?;
    let cases: [(&str, &[usize], Mode, bool); 6] = [
        ("decode ks=[1,1,1,1]", &[1, 1, 1, 1], Mode::Decode, true),
        ("verify ks=[3,3,3,3]", &[3, 3, 3, 3], Mode::Verify, false),
        ("verify ks=[3,1,2]", &[3, 1, 2], Mode::Verify, false),
        ("verify ks=[1,1,1,1]", &[1, 1, 1, 1], Mode::Verify, false),
        (
            "verify(decode_step) ks=[3,1,2]",
            &[3, 1, 2],
            Mode::Verify,
            true,
        ),
        (
            "verify ks=[3;6] (groups 15+3)",
            &[3, 3, 3, 3, 3, 3],
            Mode::Verify,
            false,
        ),
    ];
    for (label, ks, mode, ds) in cases {
        let label = format!("[{tag}] {label}");
        fails += check_case(&mut rig, &mut seqs, ks, mode, ds, steps, &mut rng, &label)?;
    }

    let sk = Glm5NextTextSkeleton::from_config(cfg)?;
    let dsa_layers = sk.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count();
    for (label, ks, mode) in [
        ("decode C=4 ks=[1;4]", &[1usize, 1, 1, 1][..], Mode::Decode),
        ("verify C=4 ks=[3;4]", &[3, 3, 3, 3][..], Mode::Verify),
    ] {
        let t_off = time_arm(&mut rig, &mut seqs, ks, mode, false, reps)?;
        let t_on = time_arm(&mut rig, &mut seqs, ks, mode, true, reps)?;
        println!(
            "TIMING [{tag}] {label}: DSA layer perseq_us={:.1} xseq_us={:.1} speedup={:.2}x \
             saved_us={:.1}/layer -> {:.2} ms per step over {dsa_layers} DSA layers (one rank, \
             eager, median of {reps})",
            t_off * 1e6,
            t_on * 1e6,
            t_off / t_on,
            (t_off - t_on) * 1e6,
            (t_off - t_on) * 1e3 * dsa_layers as f64
        );
    }
    Ok(fails)
}

fn main() -> Result<()> {
    let child = std::env::var("GLM_XS_FP8_CHILD").as_deref() == Ok("1");
    let fails = run_all()?;
    if child {
        if fails > 0 {
            println!("FAIL: [fp8] {fails} comparison(s) differ");
            std::process::exit(1);
        }
        println!("[fp8] all comparisons bitwise equal");
        return Ok(());
    }
    let mut total = fails;
    if df::dense_fp8() {
        println!("METRALE_GLM_DENSE_FP8 already on in the parent; no separate FP8 run");
    } else {
        let status = std::process::Command::new(std::env::current_exe()?)
            .env("METRALE_GLM_DENSE_FP8", "1")
            .env("GLM_XS_FP8_CHILD", "1")
            .status()?;
        if !status.success() {
            println!("FAIL: the METRALE_GLM_DENSE_FP8=1 run exited {status}");
            total += 1;
        }
    }
    if total > 0 {
        println!("FAIL: {total} failure(s)");
        bail!("glm5next_dsa_xseq_microtest: {total} failure(s)");
    }
    println!(
        "PASS: METRALE_GLM_DSA_XSEQ_BATCH on == off bitwise (outputs, KV latents, indexer rows) \
         in every case, BF16 and FP8 weights; batched projections row-exact"
    );
    Ok(())
}
