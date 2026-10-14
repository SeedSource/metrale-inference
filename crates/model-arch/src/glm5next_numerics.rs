// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_NUMERICS_FAKEQ=<family>:<fmt>[,<family>:<fmt>...]`: a quality-test
//! lever that emulates a lower-precision weight format at load, so a later job can measure what
//! the format costs in output quality before anyone writes a fast kernel for it. For each listed
//! family the BF16 weight is quantized to the format and dequantized back into the SAME BF16
//! buffer, and the existing BF16 kernels then read the emulated values. Speed does not matter;
//! the emulation's correctness and "unset = nothing runs" do.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset, empty, `0` or `off`: [`fakeq`] is empty, every entry point returns at once, no kernel
//!   resolves, nothing is allocated, nothing is printed.
//! - Families (the weights they cover, `[rows, cols]` BF16):
//!   `kda_fb` KDA `f_a` `[head_dim, hidden]`, `f_b` `[qkv, head_dim]`, `b_proj` `[heads, hidden]`;
//!   `kda_g` KDA `g_a` `[head_dim, hidden]`, `g_b` `[qkv, head_dim]`;
//!   `lm_head` the main head `[vocab, hidden]` (only with `--lm-head-dtype bf16`; any other head
//!   dtype fails the load);
//!   `wq_b` DSA indexer `wq_b` `[index_heads * index_head_dim, q_lora_rank]`;
//!   `router` MoE router `mlp.gate` `[num_experts, hidden]`, every MoE layer including the MTP block;
//!   `idx_small` DSA indexer `wk`, `compress_gate` `[index_head_dim, hidden]` and `weights_proj`
//!   `[index_heads, hidden]` (the BF16 copy already carries the `index_heads^-0.5` fold);
//!   `hc_fn` mHC `hc_fn` of both sites `[mix_hc, hc_mult * hidden]` (BF16 in this model; an FP32
//!   `hc_fn` fails the load rather than being emulated on the wrong buffer type).
//! - Formats: `fp8` is the `METRALE_GLM_DENSE_FP8` scheme exactly (`quantize_bf16_to_fp8`: one FP32
//!   scale per row, max|row| / 448, E4M3 RNE, then `dequant_fp8_rowscale_bf16`); `nvfp4` is
//!   `dense_fp8::quantize_nvfp4_copy` exactly (`quantize_bf16_to_nvfp4_mse`: MSE-chosen E4M3
//!   scale per 16 K, FP32 tensor scale), dequantized by the existing
//!   `dequant_nvfp4_to_bf16`. The dequantized value is rounded to BF16 (relative 2^-9), below
//!   either format's own error, so the emulation is close to a real kernel, not bit-exact.
//! - An unknown family or format, a repeated family, a malformed item, or a shape with `k % 16 !=
//!   0` is a load error.
//! - A fake-quantized tensor is never converted afterwards by `METRALE_GLM_DENSE_FP8` /
//!   `METRALE_GLM_DENSE_NVFP4` (that would stack a second quantization): `dense_fp8::register_layer`
//!   skips `g_a` / `g_b` when `kda_g` is listed ([`lists`]); the other families are not in the
//!   dense lever's list; `lm_head` is only accepted for a head that `setup_lm_heads` keeps BF16.
//! - Per family the load logs `NUMERICS_FAKEQ family=<f> fmt=<fmt> tensors=<count>
//!   max_abs_err=<v> rel_rms_err=<v>` ([`log_summary`], from `setup_lm_heads`, after every layer
//!   and the MTP block loaded) and once `NUMERICS_FAKEQ ENGAGED families=...`. The error is
//!   measured on a sample of at most [`SAMPLE_ELEMS`] elements per tensor (whole rows) against the
//!   original values: `max_abs_err` the largest |new - old| over every sampled element of every
//!   tensor of the family, `rel_rms_err` = sqrt(sum err^2 / sum old^2) over all of them.

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use crate::glm5next_layer::{Glm5NextMhc, Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mlp::Glm5NextMlpConfig;

/// 2026-10-12: Elements per tensor the error statistics look at (whole rows, evenly strided).
pub const SAMPLE_ELEMS: usize = 1 << 20;

/// 2026-10-12: A weight family the lever can fake-quantize.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    KdaFb,
    KdaG,
    LmHead,
    WqB,
    Router,
    IdxSmall,
    HcFn,
}

impl Family {
    pub const ALL: [Family; 7] = [
        Family::KdaFb,
        Family::KdaG,
        Family::LmHead,
        Family::WqB,
        Family::Router,
        Family::IdxSmall,
        Family::HcFn,
    ];
    /// The name in the lever syntax and the log lines.
    pub fn name(self) -> &'static str {
        match self {
            Family::KdaFb => "kda_fb",
            Family::KdaG => "kda_g",
            Family::LmHead => "lm_head",
            Family::WqB => "wq_b",
            Family::Router => "router",
            Family::IdxSmall => "idx_small",
            Family::HcFn => "hc_fn",
        }
    }
    fn parse(s: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.name() == s)
    }
}

/// 2026-10-12: The emulated format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    Fp8,
    Nvfp4,
    /// 2026-10-14: deliberately coarse 2-bit symmetric positive control (host-side).
    Int2,
}

impl Fmt {
    pub fn name(self) -> &'static str {
        match self {
            Fmt::Fp8 => "fp8",
            Fmt::Nvfp4 => "nvfp4",
            Fmt::Int2 => "int2",
        }
    }
    fn parse(s: &str) -> Option<Fmt> {
        match s {
            "fp8" => Some(Fmt::Fp8),
            "nvfp4" => Some(Fmt::Nvfp4),
            "int2" => Some(Fmt::Int2),
            _ => None,
        }
    }
}

/// 2026-10-12: The parsed lever: `(family, format)` in the order written.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FakeqSpec(pub Vec<(Family, Fmt)>);

impl FakeqSpec {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn fmt_of(&self, f: Family) -> Option<Fmt> {
        self.0.iter().find(|(g, _)| *g == f).map(|(_, m)| *m)
    }
}

/// 2026-10-12: Parse a `METRALE_GLM_NUMERICS_FAKEQ` value. Unset, empty, `0`, `off` = none.
/// Items are `family:fmt`, comma separated, whitespace trimmed; an unknown family or format, an
/// item without a colon, an empty item or a repeated family is an error.
pub fn parse_fakeq(raw: Option<&str>) -> Result<FakeqSpec> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(FakeqSpec::default());
    };
    if matches!(raw, "" | "0" | "off") {
        return Ok(FakeqSpec::default());
    }
    let mut out: Vec<(Family, Fmt)> = Vec::new();
    for item in raw.split(',').map(str::trim) {
        let Some((fam, fmt)) = item.split_once(':') else {
            bail!(
                "METRALE_GLM_NUMERICS_FAKEQ: item {item:?} is not <family>:<fmt> (families: {}; \
                 formats: fp8, nvfp4, int2)",
                family_list()
            );
        };
        let (fam, fmt) = (fam.trim(), fmt.trim());
        let Some(f) = Family::parse(fam) else {
            bail!(
                "METRALE_GLM_NUMERICS_FAKEQ: unknown family {fam:?} (families: {})",
                family_list()
            );
        };
        let Some(m) = Fmt::parse(fmt) else {
            bail!(
                "METRALE_GLM_NUMERICS_FAKEQ: unknown format {fmt:?} for {fam} (fp8, nvfp4, int2)"
            );
        };
        if out.iter().any(|(g, _)| *g == f) {
            bail!("METRALE_GLM_NUMERICS_FAKEQ: family {fam} listed twice");
        }
        out.push((f, m));
    }
    Ok(FakeqSpec(out))
}

fn family_list() -> String {
    Family::ALL.map(|f| f.name()).join(", ")
}

fn parsed() -> &'static Result<FakeqSpec, String> {
    static P: OnceLock<Result<FakeqSpec, String>> = OnceLock::new();
    P.get_or_init(|| {
        parse_fakeq(std::env::var("METRALE_GLM_NUMERICS_FAKEQ").ok().as_deref())
            .map_err(|e| format!("{e:#}"))
    })
}

/// 2026-10-12: The lever as the process runs it; empty when off (or invalid, which [`validate`]
/// reports). Read once.
pub fn fakeq() -> FakeqSpec {
    parsed().as_ref().cloned().unwrap_or_default()
}

/// 2026-10-12: An error naming an invalid `METRALE_GLM_NUMERICS_FAKEQ` value. Called at the start
/// of every entry point that can run, so a typo fails the load instead of silently running a
/// reference arm.
pub fn validate() -> Result<()> {
    match parsed() {
        Ok(_) => Ok(()),
        Err(e) => bail!("{e}"),
    }
}

/// 2026-10-12: Whether family `f` is listed.
pub fn lists(f: Family) -> bool {
    parsed().as_ref().is_ok_and(|s| s.fmt_of(f).is_some())
}

/// 2026-10-12: Error statistics of one fake-quantized tensor (a sample of whole rows).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrStats {
    pub max_abs: f64,
    pub sum_sq_err: f64,
    pub sum_sq_ref: f64,
    pub elems: u64,
}

impl ErrStats {
    fn add(&mut self, o: &ErrStats) {
        self.max_abs = self.max_abs.max(o.max_abs);
        self.sum_sq_err += o.sum_sq_err;
        self.sum_sq_ref += o.sum_sq_ref;
        self.elems += o.elems;
    }
    /// sqrt(sum err^2 / sum ref^2); 0 for an all-zero reference with zero error.
    pub fn rel_rms(&self) -> f64 {
        if self.sum_sq_ref > 0.0 {
            (self.sum_sq_err / self.sum_sq_ref).sqrt()
        } else if self.sum_sq_err > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

/// 2026-10-12: Row indices the error statistics read: every `step`-th row, `step` chosen so the
/// sample is at most [`SAMPLE_ELEMS`] elements (at least one row).
pub fn sample_rows(n: usize, k: usize) -> Vec<usize> {
    let rows_budget = (SAMPLE_ELEMS / k.max(1)).max(1);
    let step = n.div_ceil(rows_budget).max(1);
    (0..n).step_by(step).collect()
}

fn read_rows(gpu: &dyn GpuBackend, w: DevicePtr, k: usize, rows: &[usize]) -> Result<Vec<u16>> {
    let mut out = Vec::with_capacity(rows.len() * k);
    let mut buf = vec![0u8; k * 2];
    for &r in rows {
        gpu.copy_d2h(w.offset(r * k * 2), &mut buf)?;
        out.extend(
            buf.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]])),
        );
    }
    Ok(out)
}

/// 2026-10-12: Error statistics of `new` against `old` (BF16 bit patterns, same length).
pub fn err_stats(old: &[u16], new: &[u16]) -> ErrStats {
    let mut s = ErrStats::default();
    for (&o, &n) in old.iter().zip(new) {
        let (o, n) = (bf16::from_bits(o).to_f64(), bf16::from_bits(n).to_f64());
        let e = (n - o).abs();
        if e > s.max_abs || e.is_nan() {
            s.max_abs = e;
        }
        s.sum_sq_err += e * e;
        s.sum_sq_ref += o * o;
        s.elems += 1;
    }
    s
}

/// 2026-10-12: Quantize the BF16 `[n, k]` weight at `w` to `fmt` and write the dequantized values
/// back into the same buffer, on the default stream (every launch is followed by a sync; this is a
/// load-time operation). Frees the temporaries. Returns the error statistics of a row sample
/// against the original values.
pub fn fake_quant_inplace(
    gpu: &dyn GpuBackend,
    w: DevicePtr,
    n: usize,
    k: usize,
    fmt: Fmt,
) -> Result<ErrStats> {
    if n == 0 || k == 0 || !k.is_multiple_of(16) || w.is_null() {
        bail!(
            "METRALE_GLM_NUMERICS_FAKEQ: cannot fake-quantize a [{n}, {k}] weight (needs n, k > 0 and \
             k a multiple of 16)"
        );
    }
    let s = gpu.default_stream();
    let rows = sample_rows(n, k);
    let old = read_rows(gpu, w, k, &rows)?;
    match fmt {
        Fmt::Fp8 => {
            let kq = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
            let kd = gpu.kernel("dequant_fp8_rowscale_bf16", "dequant_fp8_rowscale_bf16")?;
            let q = metrale_model_layers::weight_map::quantize_to_fp8(
                &DenseWeight { weight: w },
                n,
                k,
                gpu,
                kq,
                s,
            )?;
            let r = ops::dequant_fp8_rowscale_bf16(gpu, kd, &q, w, n as u32, k as u32, s)
                .and_then(|()| gpu.synchronize(s));
            gpu.free(q.weight)?;
            gpu.free(q.row_scale)?;
            r?;
        }
        Fmt::Nvfp4 => {
            let kd = gpu.kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16")?;
            let Some(q) = crate::glm5next_layer::dense_fp8::quantize_nvfp4_copy(gpu, w, n, k, s)?
            else {
                bail!(
                    "METRALE_GLM_NUMERICS_FAKEQ: the NVFP4 quantizer kernels are missing from this \
                     target"
                );
            };
            let r = KernelLaunch::new(gpu, kd)
                .grid([n as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(q.weight)
                .arg_ptr(q.weight_scale)
                .arg_ptr(w)
                .arg_f32(q.weight_scale_2)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(s)
                .and_then(|()| gpu.synchronize(s));
            gpu.free(q.weight)?;
            gpu.free(q.weight_scale)?;
            r?;
        }
        Fmt::Int2 => {
            // Host-side, in row chunks (<= 64 MiB of BF16 each): int2 is a positive control, so
            // no device kernel (and no image change) is needed. Same arithmetic as
            // `host_ref::int2_fakeq_bits`.
            let rows_per = ((32usize << 20) / k).max(1);
            let mut buf = Vec::new();
            let mut r0 = 0;
            while r0 < n {
                let m = rows_per.min(n - r0);
                buf.resize(m * k * 2, 0u8);
                let at = w.offset(r0 * k * 2);
                gpu.copy_d2h(at, &mut buf)?;
                let src: Vec<u16> = buf
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let out = host_ref::int2_fakeq_bits(&src, m, k);
                for (c, v) in buf.chunks_exact_mut(2).zip(out) {
                    c.copy_from_slice(&v.to_le_bytes());
                }
                gpu.copy_h2d(&buf, at)?;
                r0 += m;
            }
            gpu.synchronize(s)?;
        }
    }
    let new = read_rows(gpu, w, k, &rows)?;
    Ok(err_stats(&old, &new))
}

#[derive(Default)]
struct Agg {
    fmt: Option<Fmt>,
    tensors: usize,
    stats: ErrStats,
}

fn agg() -> &'static Mutex<BTreeMap<Family, Agg>> {
    static A: OnceLock<Mutex<BTreeMap<Family, Agg>>> = OnceLock::new();
    A.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// 2026-10-12: Fake-quantize one weight of family `f` when the lever lists it, and fold its error
/// statistics into the family's load line. A no-op when `f` is not listed.
pub fn apply(
    gpu: &dyn GpuBackend,
    f: Family,
    w: DevicePtr,
    n: usize,
    k: usize,
    what: &str,
) -> Result<()> {
    let Some(fmt) = parsed().as_ref().ok().and_then(|s| s.fmt_of(f)) else {
        return Ok(());
    };
    let st = fake_quant_inplace(gpu, w, n, k, fmt).map_err(|e| {
        anyhow::anyhow!(
            "METRALE_GLM_NUMERICS_FAKEQ {}:{} on {what} [{n}, {k}]: {e:#}",
            f.name(),
            fmt.name()
        )
    })?;
    let mut m = agg().lock().unwrap();
    let a = m.entry(f).or_default();
    a.fmt = Some(fmt);
    a.tensors += 1;
    a.stats.add(&st);
    Ok(())
}

/// 2026-10-12: Fake-quantize one layer's mixer and MLP weights (a text layer or the MTP block).
/// Runs before `dense_fp8::register_layer`, which skips `g_a` / `g_b` when `kda_g` is listed.
pub fn apply_layer(
    gpu: &dyn GpuBackend,
    mixer: &Glm5NextMixer,
    mlp: &Glm5NextMlpSite,
    mlp_cfg: &Glm5NextMlpConfig,
    layer_idx: usize,
) -> Result<()> {
    validate()?;
    if fakeq().is_empty() {
        return Ok(());
    }
    match mixer {
        Glm5NextMixer::Kda { layer, cfg, .. } => {
            let w = &layer.weights;
            let (hid, qkv, hd, heads) = (cfg.hidden, cfg.qkv_dim(), cfg.head_dim, cfg.heads);
            let at = |s: &str| format!("layer {layer_idx} {s}");
            apply(gpu, Family::KdaFb, w.f_a.weight, hd, hid, &at("f_a"))?;
            apply(gpu, Family::KdaFb, w.f_b.weight, qkv, hd, &at("f_b"))?;
            apply(
                gpu,
                Family::KdaFb,
                w.b_proj.weight,
                heads,
                hid,
                &at("b_proj"),
            )?;
            apply(gpu, Family::KdaG, w.g_a.weight, hd, hid, &at("g_a"))?;
            apply(gpu, Family::KdaG, w.g_b.weight, qkv, hd, &at("g_b"))?;
        }
        Glm5NextMixer::Dsa(l) => {
            let (c, w) = (&l.cfg, &l.weights);
            let at = |s: &str| format!("layer {layer_idx} {s}");
            let (ih, ihd) = (c.index_heads, c.index_head_dim);
            apply(
                gpu,
                Family::WqB,
                w.wq_b,
                ih * ihd,
                c.q_lora_rank,
                &at("indexer.wq_b"),
            )?;
            apply(
                gpu,
                Family::IdxSmall,
                w.wk,
                ihd,
                c.hidden,
                &at("indexer.wk"),
            )?;
            apply(
                gpu,
                Family::IdxSmall,
                w.compress_gate,
                ihd,
                c.hidden,
                &at("indexer.compress_gate"),
            )?;
            apply(
                gpu,
                Family::IdxSmall,
                w.weights_proj,
                ih,
                c.hidden,
                &at("indexer.weights_proj"),
            )?;
        }
    }
    if let Glm5NextMlpSite::Moe(w) = mlp {
        apply(
            gpu,
            Family::Router,
            w.router,
            mlp_cfg.num_experts,
            mlp_cfg.hidden,
            &format!("layer {layer_idx} mlp.gate"),
        )?;
    }
    Ok(())
}

/// 2026-10-12: Fake-quantize a text layer's two mHC `hc_fn` matrices. Fails for an FP32 `hc_fn`.
pub fn apply_mhc(
    gpu: &dyn GpuBackend,
    mhc: &Glm5NextMhc,
    hidden: usize,
    layer_idx: usize,
) -> Result<()> {
    validate()?;
    if !lists(Family::HcFn) {
        return Ok(());
    }
    let rows = crate::glm5next_mhc::mix_hc(mhc.hc_mult);
    for (site, s) in [("attn", &mhc.attn), ("ffn", &mhc.ffn)] {
        if !s.hc_fn_bf16 {
            bail!(
                "METRALE_GLM_NUMERICS_FAKEQ hc_fn: layer {layer_idx} {site} hc_fn is FP32; only a BF16 \
                 hc_fn is emulated (the loader uploads it as BF16, so this should not happen)"
            );
        }
        apply(
            gpu,
            Family::HcFn,
            s.hc_fn,
            rows,
            mhc.hc_mult * hidden,
            &format!("layer {layer_idx} hc_{site}_fn"),
        )?;
    }
    Ok(())
}

/// 2026-10-12: What `setup_lm_heads` decided for the main head, for [`apply_lm_head`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadKind {
    /// Kept BF16 (`--lm-head-dtype bf16`, or an MLA model's default).
    Bf16,
    /// Pre-packed NVFP4 in the checkpoint.
    PrepackedNvfp4,
    /// Quantized to FP8 at load.
    Fp8,
    /// Quantized to NVFP4 at load.
    Nvfp4,
}

/// 2026-10-12: The error text for `lm_head` listed with a head that is not BF16.
pub fn lm_head_dtype_error(kind: HeadKind) -> String {
    format!(
        "METRALE_GLM_NUMERICS_FAKEQ lists lm_head, which needs the main LM head to stay BF16 \
         (--lm-head-dtype bf16), but this load would use {kind:?}; fake-quantizing a head that is \
         quantized again afterwards would stack two quantizations"
    )
}

/// 2026-10-12: Fake-quantize the main LM head `[vocab, hidden]` when `lm_head` is listed; fails
/// the load when the head is not kept BF16. A no-op when `lm_head` is not listed.
pub fn apply_lm_head(
    gpu: &dyn GpuBackend,
    head: DevicePtr,
    vocab: usize,
    hidden: usize,
    kind: HeadKind,
) -> Result<()> {
    validate()?;
    if !lists(Family::LmHead) {
        return Ok(());
    }
    if kind != HeadKind::Bf16 {
        bail!("{}", lm_head_dtype_error(kind));
    }
    apply(gpu, Family::LmHead, head, vocab, hidden, "lm_head")
}

/// 2026-10-12: Log one `NUMERICS_FAKEQ family=...` line per listed family and, once,
/// `NUMERICS_FAKEQ ENGAGED families=...`. Called after every layer, the MTP block and the LM head
/// were processed (`setup_lm_heads`). Silent when the lever is off.
pub fn log_summary() {
    static ONCE: OnceLock<()> = OnceLock::new();
    let spec = fakeq();
    if spec.is_empty() || ONCE.set(()).is_err() {
        return;
    }
    let m = agg().lock().unwrap();
    for (f, fmt) in &spec.0 {
        match m.get(f) {
            Some(a) => tracing::info!(
                "NUMERICS_FAKEQ family={} fmt={} tensors={} max_abs_err={:.6e} rel_rms_err={:.6e}",
                f.name(),
                fmt.name(),
                a.tensors,
                a.stats.max_abs,
                a.stats.rel_rms()
            ),
            None => tracing::warn!(
                "NUMERICS_FAKEQ family={} fmt={} tensors=0 max_abs_err=n/a rel_rms_err=n/a (the \
                 family matched no tensor in this model)",
                f.name(),
                fmt.name()
            ),
        }
    }
    tracing::info!(
        "NUMERICS_FAKEQ ENGAGED families={}",
        spec.0
            .iter()
            .map(|(f, m)| format!("{}:{}", f.name(), m.name()))
            .collect::<Vec<_>>()
            .join(",")
    );
}

/// 2026-10-12: Host references of the schemes above, for the unit tests and
/// `examples/glm_numerics_fakeq_microtest.rs`. They mirror the kernels' arithmetic step by step.
pub mod host_ref {
    use half::bf16;

    /// The E2M1 magnitudes by code, with the sign in bit 3 (`DQ_E2M1_LUT` in dequant_nvfp4_bf16.cu).
    pub const E2M1_LUT: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    /// E4M3 byte -> f32 as `dq8_dec` / `fp8_e4m3_to_f32` do: bias 7, subnormal `m * 2^-9`, NaN
    /// code (exp 15, m 7) = NaN.
    pub fn e4m3_decode(b: u8) -> f32 {
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let e = ((b >> 3) & 0xF) as i32;
        let m = (b & 7) as f32;
        if e == 15 && (b & 7) == 7 {
            return f32::NAN;
        }
        if e == 0 {
            sign * m * 0.001953125
        } else {
            sign * (1.0 + m * 0.125) * 2f32.powi(e - 7)
        }
    }

    /// E4M3 byte -> f32 as `dq_fp8_e4m3_decode` of dequant_nvfp4_bf16.cu does (the NaN code reads
    /// as 0, the block scale byte decode).
    pub fn e4m3_scale_decode(b: u8) -> f32 {
        let e = ((b >> 3) & 0xF) as i32;
        let m = b & 7;
        if e == 15 && m == 7 {
            return 0.0;
        }
        e4m3_decode(b)
    }

    /// f32 -> E4M3 byte, round to nearest even on the encoding, saturating at +-448, sign kept
    /// (`-0.0` and tiny negatives give `0x80`): what `(__nv_fp8_e4m3)val` does for a finite `val`
    /// already clamped to +-448.
    pub fn e4m3_encode_rne(v: f32) -> u8 {
        let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
        let a = (v.abs() as f64).min(448.0);
        if a == 0.0 {
            return sign;
        }
        let table: Vec<f64> = (0u8..=0x7E).map(|c| e4m3_decode(c) as f64).collect();
        let hi = table.partition_point(|&t| t < a);
        if hi >= table.len() {
            return sign | 0x7E;
        }
        if table[hi] == a {
            return sign | hi as u8;
        }
        if hi == 0 {
            return sign;
        }
        let lo = hi - 1;
        let (dl, dh) = (a - table[lo], table[hi] - a);
        let c = if dl < dh {
            lo
        } else if dh < dl {
            hi
        } else if lo % 2 == 0 {
            lo
        } else {
            hi
        };
        sign | c as u8
    }

    /// The per-row FP8 fake-quant of `quantize_bf16_to_fp8` + `dequant_fp8_rowscale_bf16` on a
    /// BF16 `[n, k]` matrix (bit patterns): scale = max|row| / 448 (1 for a zero row), `v * (1 /
    /// scale)` clamped to +-448, E4M3 RNE, decoded, times the scale in f32, RNE to BF16.
    #[allow(clippy::manual_clamp)] // fmaxf/fminf NaN semantics, not f32::clamp
    pub fn fp8_fakeq_bits(src: &[u16], n: usize, k: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(src.len());
        for r in 0..n {
            let row = &src[r * k..(r + 1) * k];
            let mut mx = 0.0f32;
            for &b in row {
                let a = bf16::from_bits(b).to_f32().abs();
                if a > mx {
                    mx = a;
                }
            }
            let scale = if mx > 0.0 { mx / 448.0 } else { 1.0 };
            let inv = 1.0f32 / scale;
            for &b in row {
                // fmaxf / fminf: a NaN operand yields the other one (0 * inf = NaN gives -448).
                let v = (bf16::from_bits(b).to_f32() * inv).max(-448.0).min(448.0);
                let q = e4m3_encode_rne(v);
                out.push(bf16::from_f32(e4m3_decode(q) * scale).to_bits());
            }
        }
        out
    }

    /// 2026-10-14: the `int2` positive-control fake-quant on a BF16 `[n, k]` matrix (bit patterns):
    /// per group of 16 consecutive values along K, scale = max|x| / 1.5 (f32; a zero group stays
    /// zero), code = clamp(floor(x / scale + 2), 0, 3) (round to nearest, ties toward +), value =
    /// (code - 1.5) * scale in f32, RNE to BF16. Values are {-1.5, -0.5, 0.5, 1.5} * scale.
    pub fn int2_fakeq_bits(src: &[u16], n: usize, k: usize) -> Vec<u16> {
        assert!(k % 16 == 0 && src.len() == n * k);
        let mut out = Vec::with_capacity(src.len());
        for grp in src.chunks_exact(16) {
            let mx = grp
                .iter()
                .map(|&b| bf16::from_bits(b).to_f32().abs())
                .filter(|a| !a.is_nan())
                .fold(0.0f32, f32::max);
            let scale = mx / 1.5;
            for &b in grp {
                let v = bf16::from_bits(b).to_f32();
                if scale == 0.0 || !scale.is_finite() || v.is_nan() {
                    out.push(if scale == 0.0 { 0 } else { b });
                    continue;
                }
                let code = (v / scale + 2.0).floor().clamp(0.0, 3.0);
                out.push(bf16::from_f32((code - 1.5) * scale).to_bits());
            }
        }
        out
    }

    /// `dequant_nvfp4_to_bf16` on `packed` `[n, k/2]`, `scales` `[n, k/16]`, tensor scale `g`:
    /// `E2M1(nibble) * (E4M3(scale) * g)` in f32, RNE to BF16 (bit patterns).
    pub fn nvfp4_dequant_bits(
        packed: &[u8],
        scales: &[u8],
        g: f32,
        n: usize,
        k: usize,
    ) -> Vec<u16> {
        let mut out = Vec::with_capacity(n * k);
        for r in 0..n {
            for c in 0..k {
                let s = e4m3_scale_decode(scales[r * (k / 16) + c / 16]) * g;
                let byte = packed[r * (k / 2) + c / 2];
                let nib = if c % 2 == 1 { byte >> 4 } else { byte & 0xF };
                out.push(bf16::from_f32(E2M1_LUT[nib as usize] * s).to_bits());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::host_ref::*;
    use super::*;

    #[test]
    fn parse_off_forms() {
        for v in [None, Some(""), Some("  "), Some("0"), Some("off")] {
            assert!(parse_fakeq(v).unwrap().is_empty(), "{v:?}");
        }
    }

    #[test]
    fn parse_good_values() {
        let s = parse_fakeq(Some("kda_fb:fp8, router:nvfp4,lm_head:nvfp4")).unwrap();
        assert_eq!(
            s.0,
            vec![
                (Family::KdaFb, Fmt::Fp8),
                (Family::Router, Fmt::Nvfp4),
                (Family::LmHead, Fmt::Nvfp4)
            ]
        );
        assert_eq!(s.fmt_of(Family::Router), Some(Fmt::Nvfp4));
        assert_eq!(s.fmt_of(Family::HcFn), None);
        for f in Family::ALL {
            let one = parse_fakeq(Some(&format!("{}:fp8", f.name()))).unwrap();
            assert_eq!(one.0, vec![(f, Fmt::Fp8)]);
        }
    }

    #[test]
    fn parse_bad_values_name_the_lever() {
        for v in [
            "kda_fb",
            "kda_fb:",
            ":fp8",
            "kda_fb:fp4",
            "kda_x:fp8",
            "kda_fb:fp8,kda_fb:nvfp4",
            "kda_fb:fp8,,router:fp8",
            "kda_fb:fp8,",
            "1",
            "all:fp8",
            "KDA_FB:FP8",
        ] {
            let e = parse_fakeq(Some(v)).unwrap_err().to_string();
            assert!(e.contains("METRALE_GLM_NUMERICS_FAKEQ"), "{v}: {e}");
        }
    }

    #[test]
    fn lm_head_error_names_the_dtype() {
        let e = lm_head_dtype_error(HeadKind::Nvfp4);
        assert!(
            e.contains("--lm-head-dtype bf16") && e.contains("Nvfp4"),
            "{e}"
        );
    }

    /// Every finite code decodes and re-encodes to itself; negative zero keeps its sign.
    #[test]
    fn e4m3_encode_inverts_decode() {
        for c in 0u8..=0x7E {
            for s in [0u8, 0x80] {
                let b = c | s;
                assert_eq!(e4m3_encode_rne(e4m3_decode(b)), b, "code {b:#x}");
            }
        }
        assert_eq!(e4m3_encode_rne(-0.0), 0x80);
        assert_eq!(e4m3_encode_rne(0.0), 0x00);
        assert_eq!(e4m3_encode_rne(448.0), 0x7E);
        assert_eq!(e4m3_encode_rne(-448.0), 0xFE);
    }

    #[test]
    fn e4m3_encode_rounds_ties_to_even_and_underflows() {
        // 1.0 (0x38) and 1.125 (0x39): the midpoint 1.0625 ties to the even code 0x38.
        assert_eq!(e4m3_encode_rne(1.0625), 0x38);
        // 1.125 (0x39) and 1.25 (0x3A): the midpoint 1.1875 ties to the even code 0x3A.
        assert_eq!(e4m3_encode_rne(1.1875), 0x3A);
        assert_eq!(e4m3_encode_rne(1.0626), 0x39);
        // 2^-9 is the smallest subnormal (0x01); half of it ties to 0 (even).
        assert_eq!(e4m3_encode_rne(2f32.powi(-10)), 0x00);
        assert_eq!(e4m3_encode_rne(2f32.powi(-10) * 1.01), 0x01);
        assert_eq!(e4m3_encode_rne(-(2f32.powi(-10))), 0x80);
        // 416 (0x7D) and 448 (0x7E): midpoint 432 ties to 0x7E (even).
        assert_eq!(e4m3_encode_rne(432.0), 0x7E);
        assert_eq!(e4m3_encode_rne(1000.0), 0x7E);
    }

    #[test]
    fn fp8_reference_reproduces_representable_rows_and_bounds_the_error() {
        let k = 32;
        let row: Vec<u16> = (0..k)
            .map(|i| bf16::from_f32((i as f32 - 15.5) * 0.37).to_bits())
            .collect();
        let out = fp8_fakeq_bits(&row, 1, k);
        let st = err_stats(&row, &out);
        // E4M3 has a 3-bit mantissa: relative step 2^-3, so rms error stays well under 2^-3.
        assert!(st.rel_rms() < 0.07, "{}", st.rel_rms());
        assert!(st.max_abs > 0.0);
        // A row of one value is exactly representable: scale = v / 448, code 448.
        let ones = vec![bf16::from_f32(3.0).to_bits(); k];
        let o = fp8_fakeq_bits(&ones, 1, k);
        assert!(
            o.iter().all(|&b| bf16::from_bits(b).to_f32() == 3.0),
            "{o:?}"
        );
        // A zero row stays zero (scale 1).
        let z = fp8_fakeq_bits(&vec![0u16; k], 1, k);
        assert!(z.iter().all(|&b| b == 0));
    }

    #[test]
    fn nvfp4_reference_decodes_codes() {
        // One block: scale byte 0x38 = 1.0, tensor scale 0.5; nibbles 0..15 over 16 columns.
        let packed: Vec<u8> = (0..8).map(|i| (2 * i + 1) << 4 | (2 * i)).collect();
        let out = nvfp4_dequant_bits(&packed, &[0x38], 0.5, 1, 16);
        for (c, &b) in out.iter().enumerate() {
            let want = bf16::from_f32(E2M1_LUT[c] * 0.5).to_bits();
            assert_eq!(b, want, "col {c}");
        }
    }

    #[test]
    fn parse_accepts_int2() {
        let s = parse_fakeq(Some("router:int2")).unwrap();
        assert_eq!(s.0, vec![(Family::Router, Fmt::Int2)]);
        assert_eq!(Fmt::Int2.name(), "int2");
        assert!(parse_fakeq(Some("router:int3")).is_err());
        assert!(parse_fakeq(Some("router:int2,router:fp8")).is_err());
    }

    fn bf(v: f32) -> u16 {
        bf16::from_f32(v).to_bits()
    }

    #[test]
    fn int2_reference_known_vector() {
        // max|x| = 3.0 -> scale 2.0, levels {-3,-1,1,3}. Boundaries at -2, 0, 2 (x/scale -1,0,1).
        let mut g = vec![0.0f32; 16];
        g[..8].copy_from_slice(&[3.0, -3.0, 0.9, -0.9, 1.1, -1.1, 2.1, -2.1]);
        g[8] = 0.0; // ties toward +: 0 -> +1
        g[9] = 1.9;
        g[10] = -1.9;
        let src: Vec<u16> = g.iter().map(|&v| bf(v)).collect();
        let out = int2_fakeq_bits(&src, 1, 16);
        let want = [3.0, -3.0, 1.0, -1.0, 1.0, -1.0, 3.0, -3.0, 1.0, 1.0, -1.0];
        for (i, w) in want.iter().enumerate() {
            assert_eq!(bf16::from_bits(out[i]).to_f32(), *w, "idx {i}");
        }
        // A zero group stays zero; groups are independent (second group has its own scale).
        let mut two = vec![0u16; 32];
        two[16] = bf(0.3);
        let o = int2_fakeq_bits(&two, 1, 32);
        assert!(o[..16].iter().all(|&b| b == 0));
        assert_eq!(
            bf16::from_bits(o[16]).to_f32(),
            bf16::from_f32(1.5 * (0.3f32 / 1.5)).to_f32()
        );
        // Every output is one of 4 levels per group.
        let lv: std::collections::BTreeSet<u32> = out
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32().to_bits())
            .collect();
        assert!(lv.len() <= 4, "{lv:?}");
    }

    #[test]
    fn int2_error_much_larger_than_nvfp4_scale_on_random_data() {
        // Gaussian-ish data via an LCG; compare with a 4-bit-like grid (fp8 reference as the
        // fine format) and the analytic 2-bit floor.
        let (n, k) = (8, 256);
        let mut x = 0x2545F4914F6CDD1Du64;
        let mut u = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as f64 / (1u64 << 53) as f64
        };
        let src: Vec<u16> = (0..n * k)
            .map(|_| bf((((u() + u() + u() + u()) - 2.0) * 1.7) as f32))
            .collect();
        let e2 = err_stats(&src, &int2_fakeq_bits(&src, n, k)).rel_rms();
        let e8 = err_stats(&src, &fp8_fakeq_bits(&src, n, k)).rel_rms();
        assert!(e2 > 0.2, "int2 rel_rms {e2}");
        assert!(e2 > 10.0 * e8, "int2 {e2} vs fp8 {e8}");
    }

    #[test]
    fn sample_rows_respects_the_budget() {
        let r = sample_rows(24, 16384);
        assert_eq!(r.len(), 24);
        let r = sample_rows(154_880, 4096);
        assert!(r.len() * 4096 <= SAMPLE_ELEMS + 4096 && !r.is_empty());
        assert_eq!(r[0], 0);
        let r = sample_rows(1, 1 << 24);
        assert_eq!(r, vec![0]);
    }

    #[test]
    fn err_stats_counts_and_scales() {
        let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(), bf16::from_f32(2.0).to_bits()];
        let b: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(), bf16::from_f32(2.5).to_bits()];
        let s = err_stats(&a, &b);
        assert_eq!(s.elems, 2);
        assert!((s.max_abs - 0.5).abs() < 1e-12);
        assert!((s.rel_rms() - (0.25f64 / 5.0).sqrt()).abs() < 1e-12);
        assert_eq!(err_stats(&a, &a).rel_rms(), 0.0);
    }
}
