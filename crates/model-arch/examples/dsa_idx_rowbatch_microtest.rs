// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: GPU bitwise gate of `METRALE_GLM_DSA_IDX_ROWBATCH`
//! (`glm5next_dsa/layer/idx_rowbatch.rs`): the DSA indexer's four projections over `k` rows in
//! one batched call (`Glm5NextDsaLayer::indexer_project_rows`, the block `decode_k_rows` runs
//! for a single sequence's verify and `xseq_group` for a cross-sequence group) against the
//! per-row production path, row by row at M = 1:
//! - `Glm5NextDsaLayer::indexer_project_one` (what `indexer_forward_to` launches per row: `wk`,
//!   `k_norm`, `compress_gate`, `weights_proj` FP32), and
//! - `Glm5NextDsaLayer::indexer_qidx_one` (the per-row `wq_b` of `select_rows_ceiling`).
//!
//! GLM-5.3 shapes (hidden 4096, q_lora 1536, 32 index heads, head dim 128), random
//! deterministic weights and activations. Cases k = 1, 2, 3, 5, 16. k = 1 is kept to prove the
//! factored function at one row; in a serve the lever's `group` never engages at k = 1
//! (`ceiling_rows_group` needs k >= 2).
//!
//! Every output element (`k_normed`, `gate`, `head_weights`, `q_idx`) is compared with memcmp
//! semantics (raw bytes), each arm's outputs poisoned first with a different byte, element
//! counts asserted > 0. Negative control: one flipped exponent bit of element 0 of each of the
//! four weights, applied after the per-row reference ran, must change the matching batched
//! output (then the bit is restored).
//!
//! Environment: the car's `METRALE_GLM_GEMV_TC=1` and `METRALE_GLM_DSA_GEMV_SPLIT=1` are set in
//! process (they are read once, before any launch). The gate fails when the tensor-core GEMV
//! does not route `wk` at the case's k (the lever would not engage in a serve).
//!
//! Kernel modules (copied from the production handles): `gemv` (`dense_gemv_bf16`,
//! `dense_gemv_bf16_fp32out`), `dense_gemv_bf16_batchm` (`dense_gemv_bf16_batchm`,
//! `dense_gemv_bf16_fp32out_batchm`), `dense_gemv_tcm` (`dense_gemv_tcm::MODULE`, entries
//! `dense_gemv_tcm_bf16*`), `gemm` (`dense_gemm_bf16`), `nllb_encoder` (`LAYERNORM_MODULE`,
//! `nllb_layernorm_bf16`), `rms_norm_vanilla`, `glm5next_mla_latent_write`, and the DSA select
//! kernels of `Glm5NextDsaKernels::resolve`. All come from the layer's own `resolve` calls.
//!
//! Prints `PASS dsa_idx_rowbatch (...)` or `FAIL dsa_idx_rowbatch: <why>` (exit 1).
//!
//! Run (GPU; build-mt sets the target env):
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_idx_rowbatch_microtest
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

#[path = "common/glm5next_multiseq_stack.rs"]
mod glm5next_multiseq_stack;

use anyhow::{Result, bail};
use glm5next_multiseq_stack as st;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::Glm5NextDsaKernels;
use metrale_model_arch::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use metrale_model_arch::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWorkspace, IndexerRowsOut,
    idx_batchm_f32_kernel,
};
use metrale_model_layers::layers::ops::dense_gemv_tcm;

const LEVERS: [(&str, &str); 2] = [
    ("METRALE_GLM_GEMV_TC", "1"),
    ("METRALE_GLM_DSA_GEMV_SPLIT", "1"),
];
const KS: [usize; 5] = [1, 2, 3, 5, 16];
const POISON_REF: u8 = 0xA5;
const POISON_BAT: u8 = 0x5A;

/// Device buffers of one arm's four outputs for up to `rows` rows.
struct Outs {
    k_normed: DevicePtr,
    gate: DevicePtr,
    head_weights: DevicePtr,
    q_idx: DevicePtr,
}

/// Byte sizes of the four outputs for `rows` rows: BF16, BF16, FP32, FP32.
fn sizes(l: &Glm5NextDsaLayer, rows: usize) -> [usize; 4] {
    let c = &l.cfg;
    [
        rows * c.index_head_dim * 2,
        rows * c.index_head_dim * 2,
        rows * c.index_heads * 4,
        rows * c.index_heads * c.index_head_dim * 4,
    ]
}

impl Outs {
    fn new(g: &dyn GpuBackend, l: &Glm5NextDsaLayer, rows: usize) -> Result<Self> {
        let s = sizes(l, rows);
        Ok(Self {
            k_normed: g.alloc(s[0])?,
            gate: g.alloc(s[1])?,
            head_weights: g.alloc(s[2])?,
            q_idx: g.alloc(s[3])?,
        })
    }
    fn ptrs(&self) -> [DevicePtr; 4] {
        [self.k_normed, self.gate, self.head_weights, self.q_idx]
    }
    fn poison(&self, g: &dyn GpuBackend, l: &Glm5NextDsaLayer, rows: usize, b: u8) -> Result<()> {
        for (p, n) in self.ptrs().iter().zip(sizes(l, rows)) {
            g.memset_async(*p, b, n, 0)?;
        }
        g.synchronize(0)
    }
    fn download(
        &self,
        g: &dyn GpuBackend,
        l: &Glm5NextDsaLayer,
        rows: usize,
    ) -> Result<Vec<Vec<u8>>> {
        self.ptrs()
            .iter()
            .zip(sizes(l, rows))
            .map(|(p, n)| st::down(g, *p, n))
            .collect()
    }
}

/// Differing elements and total elements between two raw buffers of `esz`-byte elements.
fn diff(a: &[u8], b: &[u8], esz: usize) -> (usize, usize) {
    let n = a.len() / esz;
    let d = (0..n)
        .filter(|i| a[i * esz..(i + 1) * esz] != b[i * esz..(i + 1) * esz])
        .count();
    (d, n)
}

const NAMES: [&str; 4] = ["k_normed", "gate", "head_weights", "q_idx"];
const ESZ: [usize; 4] = [2, 2, 4, 4];

/// The per-row production path: `indexer_project_one` + `indexer_qidx_one` for each row.
fn run_per_row(
    g: &dyn GpuBackend,
    l: &Glm5NextDsaLayer,
    hidden: DevicePtr,
    q_resid: DevicePtr,
    o: &Outs,
    k: usize,
) -> Result<()> {
    let c = &l.cfg;
    let (hid, ql, d, heads) = (c.hidden, c.q_lora_rank, c.index_head_dim, c.index_heads);
    for r in 0..k {
        l.indexer_project_one(
            g,
            hidden.offset(r * hid * 2),
            o.k_normed.offset(r * d * 2),
            o.gate.offset(r * d * 2),
            o.head_weights.offset(r * heads * 4),
            0,
        )?;
        l.indexer_qidx_one(
            g,
            q_resid.offset(r * ql * 2),
            o.q_idx.offset(r * heads * d * 4),
            0,
        )?;
    }
    g.synchronize(0)
}

fn run_batched(
    g: &dyn GpuBackend,
    l: &Glm5NextDsaLayer,
    hidden: DevicePtr,
    q_resid: DevicePtr,
    o: &Outs,
    k: usize,
) -> Result<()> {
    let out = IndexerRowsOut {
        k_normed: o.k_normed,
        gate: o.gate,
        head_weights: o.head_weights,
        q_idx: o.q_idx,
    };
    l.indexer_project_rows(g, idx_batchm_f32_kernel(g), hidden, q_resid, &out, k, 0)?;
    g.synchronize(0)
}

/// Flip an exponent bit of element 0 of the BF16 weight at `w`; returns the original byte.
fn flip(g: &dyn GpuBackend, w: DevicePtr) -> Result<u8> {
    let mut b = [0u8; 2];
    g.synchronize(0)?;
    g.copy_d2h(w, &mut b)?;
    let orig = b[1];
    // Bit 14 is the top exponent bit; the weights are well below 1, so it is 0 and the flip
    // makes the element astronomically large.
    g.copy_h2d(&[b[0], orig ^ 0x40], w)?;
    g.synchronize(0)?;
    Ok(orig)
}

fn restore(g: &dyn GpuBackend, w: DevicePtr, orig_hi: u8, lo: u8) -> Result<()> {
    g.copy_h2d(&[lo, orig_hi], w)?;
    g.synchronize(0)
}

fn run() -> Result<String> {
    for (k, v) in LEVERS {
        // SAFETY: set before any thread exists and before any lever is first read.
        unsafe { std::env::set_var(k, v) };
    }
    println!("levers: {LEVERS:?}");
    let modules = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "glm-5.3-flash" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(metrale_kernels::ptx_modules);
    let backend = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &backend;
    let geo = st::geometry(include_str!(
        "../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json"
    ))?;
    let mut rng = st::Lcg(0x1D8B_0A7C_5EED_0012);
    let cfg = &geo.config;
    let layer = Glm5NextDsaLayer {
        persist_bt: true,
        cfg: geo.dsa,
        weights: st::dsa_weights(g, &geo.dsa, &mut rng)?,
        kernels: Glm5NextDsaLayerKernels::resolve(g)?,
        select_kernels: Glm5NextDsaKernels::resolve(g)?,
        decode_kernel: Glm5NextDsaDecodeKernel::resolve(g)?,
        workspace: Glm5NextDsaWorkspace::new(g, &geo.dsa, 1)?,
        layer_idx: 1,
        attn_layer_idx: 0,
        rms_eps: cfg.rms_norm_eps as f32,
        kv_scale: 1.0,
    };
    if idx_batchm_f32_kernel(g).0 == 0 {
        bail!("dense_gemv_bf16_fp32out_batchm is not loaded");
    }
    let c = layer.cfg;
    println!(
        "shapes: hidden {} q_lora {} index_heads {} index_head_dim {}",
        c.hidden, c.q_lora_rank, c.index_heads, c.index_head_dim
    );
    let kmax = *KS.iter().max().unwrap_or(&1);
    let hidden_h = rng.vec(kmax * c.hidden, 1.0);
    let qres_h = rng.vec(kmax * c.q_lora_rank, 1.0);
    let hidden = st::up_bf16(g, &hidden_h)?;
    let q_resid = st::up_bf16(g, &qres_h)?;
    let o_ref = Outs::new(g, &layer, kmax)?;
    let o_bat = Outs::new(g, &layer, kmax)?;
    let (mut total, mut differ) = (0usize, 0usize);
    let mut fails = Vec::new();
    for &k in &KS {
        if !dense_gemv_tcm::ready(g, k as u32, c.index_head_dim as u32, c.hidden as u32, false) {
            bail!(
                "tensor-core GEMV does not route wk at k={k} (METRALE_GLM_GEMV_TC not on, or \
                 dense_gemv_tcm missing): the lever would not engage"
            );
        }
        o_ref.poison(g, &layer, k, POISON_REF)?;
        o_bat.poison(g, &layer, k, POISON_BAT)?;
        run_per_row(g, &layer, hidden, q_resid, &o_ref, k)?;
        run_batched(g, &layer, hidden, q_resid, &o_bat, k)?;
        let (r, b) = (o_ref.download(g, &layer, k)?, o_bat.download(g, &layer, k)?);
        for i in 0..4 {
            let (d, n) = diff(&r[i], &b[i], ESZ[i]);
            println!("k={k} {:<13} differ={d} of={n}", NAMES[i]);
            if n == 0 {
                fails.push(format!("k={k} {} has no elements", NAMES[i]));
            }
            // A poisoned output nobody wrote would equal its own poison in neither arm.
            if r[i].iter().all(|x| *x == POISON_REF) || b[i].iter().all(|x| *x == POISON_BAT) {
                fails.push(format!("k={k} {} was never written", NAMES[i]));
            }
            total += n;
            differ += d;
            if d != 0 {
                fails.push(format!("k={k} {}: {d} of {n} elements differ", NAMES[i]));
            }
        }
    }
    // Negative control: flip one weight bit after the per-row reference ran at k = 3.
    let k = 3;
    o_ref.poison(g, &layer, k, POISON_REF)?;
    run_per_row(g, &layer, hidden, q_resid, &o_ref, k)?;
    let r = o_ref.download(g, &layer, k)?;
    let w = &layer.weights;
    for (i, wp) in [w.wk, w.compress_gate, w.weights_proj, w.wq_b]
        .into_iter()
        .enumerate()
    {
        let mut lo = [0u8; 2];
        g.synchronize(0)?;
        g.copy_d2h(wp, &mut lo)?;
        let hi = flip(g, wp)?;
        o_bat.poison(g, &layer, k, POISON_BAT)?;
        run_batched(g, &layer, hidden, q_resid, &o_bat, k)?;
        let b = o_bat.download(g, &layer, k)?;
        restore(g, wp, hi, lo[0])?;
        let (d, n) = diff(&r[i], &b[i], ESZ[i]);
        println!("negative control {}: differ={d} of={n}", NAMES[i]);
        if d == 0 {
            fails.push(format!("negative control {} not detected", NAMES[i]));
        }
    }
    // After restoring the weights the batched arm must match again.
    o_bat.poison(g, &layer, k, POISON_BAT)?;
    run_batched(g, &layer, hidden, q_resid, &o_bat, k)?;
    let b = o_bat.download(g, &layer, k)?;
    for i in 0..4 {
        if diff(&r[i], &b[i], ESZ[i]).0 != 0 {
            fails.push(format!("{} differs after restoring the weights", NAMES[i]));
        }
    }
    if total == 0 {
        fails.push("no elements compared".to_string());
    }
    if fails.is_empty() {
        Ok(format!(
            "PASS dsa_idx_rowbatch ({differ} of {total} elements differ over k in {KS:?}; \
             negative controls detected 4/4)"
        ))
    } else {
        bail!("{}", fails.join("; "))
    }
}

fn main() {
    match run() {
        Ok(line) => println!("{line}"),
        Err(e) => {
            println!("FAIL dsa_idx_rowbatch: {e}");
            std::process::exit(1);
        }
    }
}
