// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextWeightLoader`: builds the GLM-5.3 text stack from a
//! `WeightStore`, plus the helpers the MTP loader shares.
//!
//! `Glm5NextTextSkeleton::from_config` fixes each layer's mixer (KDA or DSA)
//! and MLP (dense or routed MoE) from the config's `layer_types` and
//! `mlp_only_layers`; the loader builds what it lists.
//!
//! Under TP, DSA shards through `DsaTpPlan`, the MLP through the rank passed to
//! `glm5next_mlp::build`, and KDA through [`KdaShardedSource`], which slices
//! the host bytes before `bind_kda_weights` sees them. At TP > 1 both mixer
//! plans report `needs_output_all_reduce`, and the layer all-reduces the mixer
//! output (`Glm5NextLayer::mixer_all_reduce`).
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Context, Result, bail};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::ModelWeightLoader;
use crate::glm5next_dsa::build::build_dsa_weights;
use crate::glm5next_dsa::layer::{Glm5NextDsaLayer, Glm5NextDsaLayerKernels};
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::glm5next_kda::binding::{KdaDtype, KdaTensorSource, RawTensor, bind_kda_weights};
use crate::glm5next_kda::tp::KdaTpPlan;
use crate::glm5next_kda::tp_bind::KdaShardedSource;
use crate::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer};
use crate::glm5next_layer::{Glm5NextLayer, Glm5NextMhc, Glm5NextMixer, Glm5NextMlpSite};
use crate::glm5next_mhc::{Glm5NextMhcKernels, Glm5NextMhcSiteWeights, mhc_mix_max_tokens, mix_hc};
use crate::glm5next_mlp::weights::{Glm5NextExpertWeights, Nvfp4Proj};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels, build as mlp_build};
use crate::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer, Mlp};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::DenseWeight;

#[cfg(test)]
mod defer_hook_tests;
pub mod expert_arena;
mod expert_quant;
mod expert_tp_bind;
#[cfg(test)]
mod expert_tp_bind_tests;
#[cfg(test)]
mod export_layout_tests;
mod loader;
mod nvfp4_dequant;
mod nvfp4_quant;
#[cfg(test)]
mod plan_cast_tests;
mod plan_dtype;

use expert_quant::{quantize_deferred_expert_proj, quantize_expert_proj};

pub struct Glm5NextWeightLoader;

/// 2026-09-25: The full name of layer `layer`'s tensor `leaf`, under
/// `model.language_model.layers.`.
fn qualify(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

/// 2026-09-25: Whether a store tensor belongs to a text layer (index below
/// `num_layers`) and is not a routed expert. `load_layers` copies those to the
/// host (`LayerSource::collect`) and the binders upload their own buffers, so
/// `prune_after_load` frees the store's copies.
///
/// Two families are never matched:
/// - `mlp.experts.*`: `bind_expert` binds U8 experts zero-copy from the store's
///   pointers.
/// - layer `num_layers`, the MTP layer: `glm5_next_mtp::load_glm5next_mtp_module`
///   reads it.
fn is_reuploaded(name: &str, num_layers: usize) -> bool {
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, rel)) = rest.split_once('.') else {
        return false;
    };
    let Ok(idx) = idx.parse::<usize>() else {
        return false;
    };
    idx < num_layers && !rel.starts_with("mlp.experts.")
}

/// 2026-09-25: Whether a store tensor is a BF16 routed-expert projection
/// weight, which [`bind_expert`] quantised to NVFP4 rather than binding. The
/// dtype is the whole test: a U8 expert is the kernel's operand and is never
/// matched. `prune_after_load` frees the matches; `factory::build` calls it
/// after the MTP module is bound. With a store that honoured
/// [`Glm5NextWeightLoader::defer_predicate`], the MTP layer's BF16 experts are
/// never resident.
fn is_quantized_expert_weight(name: &str, dtype: WeightDtype) -> bool {
    dtype == WeightDtype::BF16
        && name.starts_with("model.language_model.layers.")
        && name.contains(".mlp.experts.")
        && name.ends_with("_proj.weight")
}

/// 2026-09-25: Whether a store tensor is a BF16 `mlp.experts.*_proj.weight` of
/// layer `num_layers`, the MTP layer: the family
/// [`Glm5NextWeightLoader::defer_predicate`] keeps off the device. The shared
/// expert (`mlp.shared_experts.*`) does not match; it is read through
/// [`LayerSource`].
fn is_full_width_mtp_expert(name: &str, dtype: WeightDtype, num_layers: usize) -> bool {
    if dtype != WeightDtype::BF16 || !name.ends_with("_proj.weight") {
        return false;
    }
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, rel)) = rest.split_once('.') else {
        return false;
    };
    idx.parse::<usize>().ok() == Some(num_layers) && rel.starts_with("mlp.experts.")
}

/// 2026-10-03: Whether a store tensor is an NVFP4 activation scale (`*.input_scale`). Under
/// `METRALE_GLM_MOE_PREFILL_CUTLASS_W4A4=1` the server keeps these names and
/// [`Glm5NextWeightLoader::defer_predicate`] defers every one of them: about 37k F32 scalars
/// across the checkpoint, each of which would otherwise be its own padded device allocation.
fn is_activation_scale(name: &str) -> bool {
    name.ends_with(".input_scale")
}

/// 2026-10-03: The defer rule: the MTP layer's full-width routed experts always, plus every
/// activation scale when `defer_activation_scales` (the CUTLASS W4A4 lever).
fn defer_rule(
    name: &str,
    dtype: WeightDtype,
    num_layers: usize,
    defer_activation_scales: bool,
) -> bool {
    is_full_width_mtp_expert(name, dtype, num_layers)
        || (defer_activation_scales && is_activation_scale(name))
}

/// 2026-10-05: Whether a store tensor belongs to a routed expert of any layer, the MTP layer
/// included (`model.language_model.layers.{L}.mlp.experts.{id}.*`: weight, block scale, global
/// scale, activation scale). Under `METRALE_GLM_EXPERT_TP=1`
/// [`Glm5NextWeightLoader::defer_predicate`] defers every one of them, and
/// `expert_tp_bind::bind_expert_tp` reads back only this rank's slice of each.
fn is_routed_expert_tensor(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return false;
    };
    let Some((idx, rel)) = rest.split_once('.') else {
        return false;
    };
    let Some(tail) = rel.strip_prefix("mlp.experts.") else {
        return false;
    };
    let Some((id, _)) = tail.split_once('.') else {
        return false;
    };
    idx.parse::<usize>().is_ok() && id.parse::<usize>().is_ok()
}

/// 2026-10-05: [`bind_expert`], or under expert-TP (`cfg.is_expert_tp()`) this rank's slice of
/// the expert (`expert_tp_bind::bind_expert_tp`). The text and MTP loaders both bind through
/// here with the same `cfg` layout, so the two never mix layouts. Lever off, exactly
/// [`bind_expert`].
/// 2026-10-06: `pub` for `examples/glm5next_weight_arena_microtest.rs`, which binds through it.
pub fn bind_expert_cfg(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
    cfg: &Glm5NextMlpConfig,
) -> Result<Glm5NextExpertWeights> {
    if cfg.is_expert_tp() {
        expert_tp_bind::bind_expert_tp(gpu, store, layer, id, cfg)
    } else {
        bind_expert(gpu, store, layer, id)
    }
}

/// 2026-10-03: The value of an activation-scale tensor's bytes: one F32 (any shape with one
/// element) that is finite and positive; `None` otherwise.
fn input_scale_value(bytes: &[u8], dtype: WeightDtype, numel: usize) -> Option<f32> {
    if dtype != WeightDtype::FP32 || numel != 1 || bytes.len() != 4 {
        return None;
    }
    let v = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    (v.is_finite() && v > 0.0).then_some(v)
}

/// 2026-10-03: Routed expert projection `base`'s (`mlp.experts.{id}.{proj}`) `input_scale` of
/// layer `layer`, read from its shard when deferred (the lever's normal case) or back from the
/// device when resident. 0.0 when absent or not one finite positive F32, which the CUTLASS W4A4
/// prefill treats as "no static scale" (dynamic per-tensor amax); a present but unusable tensor
/// is logged.
fn read_input_scale(gpu: &dyn GpuBackend, store: &WeightStore, layer: usize, base: &str) -> f32 {
    let name = qualify(layer, &format!("{base}.input_scale"));
    let got = if let Some(d) = store.deferred(&name) {
        let numel = d.shape.iter().product::<usize>();
        match d.read_host_bytes() {
            Ok(b) => Some(input_scale_value(&b, d.dtype, numel)),
            Err(e) => {
                tracing::warn!("glm5_next: {name}: reading the deferred input_scale failed: {e:#}");
                Some(None)
            }
        }
    } else if let Ok(t) = store.get(&name) {
        let numel = t.shape.iter().product::<usize>();
        Some(host_bytes(gpu, t).ok().and_then(|b| input_scale_value(&b, t.dtype, numel)))
    } else {
        None
    };
    match got {
        Some(Some(v)) => v,
        Some(None) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "glm5_next: {name} is present but not one finite positive F32 — that \
                     projection's CUTLASS W4A4 prefill uses a dynamic activation scale (first \
                     such tensor; later ones are not logged)"
                );
            });
            0.0
        }
        None => 0.0,
    }
}

/// 2026-09-25: Read a device tensor back as host bytes.
fn host_bytes(gpu: &dyn GpuBackend, t: &WeightTensor) -> Result<Vec<u8>> {
    let mut b = vec![0u8; t.byte_size()];
    gpu.copy_d2h(t.ptr, &mut b)?;
    Ok(b)
}

/// 2026-09-25: Read a BF16 or FP32 device tensor back as host `f32`, by the
/// tensor's own dtype; any other dtype is an error.
fn host_f32(gpu: &dyn GpuBackend, t: &WeightTensor, what: &str) -> Result<Vec<f32>> {
    let b = host_bytes(gpu, t)?;
    match t.dtype {
        WeightDtype::BF16 => Ok(b
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        WeightDtype::FP32 => Ok(b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        other => bail!(
            "{what}: dtype {other:?} cannot be read as f32 without a conversion this loader refuses to guess"
        ),
    }
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: One layer's tensors under layer-relative names, copied to the
/// host (routed experts by name only).
pub(super) struct LayerSource {
    names: Vec<String>,
    tensors: std::collections::BTreeMap<String, (WeightDtype, Vec<usize>, Vec<u8>)>,
    /// 2026-09-25: Copies, at the width the binders bind at, of the tensors
    /// stored at another width (see [`plan_dtype`]). Only
    /// [`KdaTensorSource::get`] reads them; [`Self::f32`] reads the stored bytes.
    plan_cast: std::collections::BTreeMap<String, (WeightDtype, Vec<u8>)>,
}

impl LayerSource {
    pub(super) fn collect(gpu: &dyn GpuBackend, store: &WeightStore, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.");
        let mut names = Vec::new();
        let mut tensors = std::collections::BTreeMap::new();
        let mut plan_cast = std::collections::BTreeMap::new();
        let rels: Vec<String> = store
            .names()
            .filter_map(|n| n.strip_prefix(&prefix).map(|r| r.to_string()))
            .collect();
        for rel in rels {
            let rel = rel.as_str();
            // 2026-09-25: Routed experts are listed but not copied: `bind_expert` reads
            // them from the store.
            if rel.starts_with("mlp.experts.") {
                names.push(rel.to_string());
                continue;
            }
            names.push(rel.to_string());
            let t = store.get(&format!("{prefix}{rel}"))?;
            let bytes = host_bytes(gpu, t)?;
            if let Some(cast) = plan_dtype::cast_to_plan_dtype(rel, t.dtype, &bytes)? {
                plan_cast.insert(rel.to_string(), cast);
            }
            tensors.insert(rel.to_string(), (t.dtype, t.shape.clone(), bytes));
        }
        Ok(Self {
            names,
            tensors,
            plan_cast,
        })
    }

    pub(super) fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, shape, bytes) = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        match dtype {
            WeightDtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()),
            WeightDtype::FP32 => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            // 2026-09-25: A packed-NVFP4 tensor is dequantized, so `f32` accepts
            // it as well as BF16 and FP32.
            WeightDtype::UInt8 => self.dequant_packed_nvfp4(name, shape, bytes),
            other => bail!("{name}: dtype {other:?} is not a plain float tensor"),
        }
    }

    /// 2026-09-25: One packed-NVFP4 `.weight` of this layer as `f32`, scaled by
    /// its `.weight_scale` (F8_E4M3 blocks) and scalar `.weight_scale_2`
    /// siblings. A missing or differently typed sibling is an error. Reached
    /// only from the `UInt8` arm of [`Self::f32`].
    fn dequant_packed_nvfp4(&self, name: &str, shape: &[usize], bytes: &[u8]) -> Result<Vec<f32>> {
        let base = name.strip_suffix(".weight").with_context(|| {
            format!("{name}: packed NVFP4 must be a `.weight`, with scale siblings beside it")
        })?;
        let (scale_dtype, _, scale_bytes) = self
            .tensors
            .get(&format!("{base}.weight_scale"))
            .with_context(|| format!("{name} is packed NVFP4 but {base}.weight_scale is absent"))?;
        if *scale_dtype != WeightDtype::FP8E4M3 {
            bail!("{base}.weight_scale is {scale_dtype:?}, expected F8_E4M3 block scales");
        }
        let s2 = self.f32(&format!("{base}.weight_scale_2"))?;
        let [scale_2] = s2[..] else {
            bail!("{base}.weight_scale_2 is not a scalar");
        };
        nvfp4_dequant::dequant_nvfp4_to_f32(name, bytes, shape, scale_bytes, scale_2)
    }
}

impl KdaTensorSource for LayerSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        let (dtype, shape, bytes) = match self.plan_cast.get(name) {
            // 2026-09-25: A plan-width copy, when one exists, replaces the stored bytes.
            Some((dtype, bytes)) => (dtype, &self.tensors.get(name)?.1, bytes),
            None => {
                let (dtype, shape, bytes) = self.tensors.get(name)?;
                (dtype, shape, bytes)
            }
        };
        // 2026-09-25: Only BF16 and FP32 tensors are offered; any other dtype reads
        // as absent rather than cast.
        let dtype = match dtype {
            WeightDtype::BF16 => KdaDtype::Bf16,
            WeightDtype::FP32 => KdaDtype::F32,
            _ => return None,
        };
        Some(RawTensor {
            dtype,
            shape: shape.clone(),
            bytes,
        })
    }
    fn names(&self) -> Vec<String> {
        self.names.clone()
    }
}

/// 2026-09-25: Bind one mHC site (`attn` or `ffn`) of a layer. `hc_{site}_fn`
/// is uploaded as BF16 with `hc_fn_bf16` set; `base` and `scale` as F32. Each
/// tensor's length is checked first.
fn bind_mhc_site(
    gpu: &dyn GpuBackend,
    src: &LayerSource,
    site: &str,
    hc_mult: usize,
    hidden: usize,
) -> Result<Glm5NextMhcSiteWeights> {
    let f = src.f32(&format!("hc_{site}_fn"))?;
    let want = mix_hc(hc_mult) * hc_mult * hidden;
    if f.len() != want {
        bail!(
            "hc_{site}_fn has {} elements, expected mix_hc({hc_mult}) * {hc_mult} * {hidden} = {want}",
            f.len()
        );
    }
    let base = src.f32(&format!("hc_{site}_base"))?;
    if base.len() != mix_hc(hc_mult) {
        bail!(
            "hc_{site}_base has {} entries, expected mix_hc({hc_mult}) = {}",
            base.len(),
            mix_hc(hc_mult)
        );
    }
    let scale = src.f32(&format!("hc_{site}_scale"))?;
    if scale.len() != 3 {
        bail!(
            "hc_{site}_scale has {} entries, expected 3 (pre, post, comb)",
            scale.len()
        );
    }
    Ok(Glm5NextMhcSiteWeights {
        hc_fn: upload_f32_as_bf16(gpu, &f)?,
        hc_fn_bf16: true,
        hc_scale: upload_f32(gpu, &scale)?,
        hc_base: upload_f32(gpu, &base)?,
        // 2026-09-25: The `hc_mix` to `hc_finish` scratch; one per site, so a
        // layer's two sites never share it.
        mix: gpu.alloc(mhc_mix_max_tokens() * mix_hc(hc_mult) * 4)?,
    })
}

/// 2026-09-25: One routed expert's gate/up/down as NVFP4 triples. Each
/// projection takes the first arm that applies:
///
/// 1. deferred by [`Glm5NextWeightLoader::defer_predicate`]: read from disk,
///    quantised, uploaded;
/// 2. resident U8: bound zero-copy with its `weight_scale` and scalar
///    `weight_scale_2`;
/// 3. resident BF16: read back and quantised; `prune_after_load` frees the
///    source.
///
/// Any other dtype is an error.
pub(super) fn bind_expert(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
) -> Result<Glm5NextExpertWeights> {
    let proj = |p: &str| -> Result<Nvfp4Proj> {
        let base = format!("mlp.experts.{id}.{p}");
        let wname = qualify(layer, &format!("{base}.weight"));
        // 2026-09-25: Checked first: a deferred tensor has no store entry.
        if let Some(d) = store.deferred(&wname) {
            return quantize_deferred_expert_proj(gpu, store, d, &base);
        }
        let w = store.get(&wname)?;
        match w.dtype {
            WeightDtype::UInt8 => {
                let scale = store.get(&qualify(layer, &format!("{base}.weight_scale")))?;
                let s2 = store.get(&qualify(layer, &format!("{base}.weight_scale_2")))?;
                let s2 = host_f32(gpu, s2, &format!("{base}.weight_scale_2"))?;
                let [s2] = s2[..] else {
                    bail!("{base}.weight_scale_2 is not a scalar");
                };
                // 2026-10-03: Lever off: never looked up (the server skipped the tensors).
                let input_scale = if metrale_config::glm_moe_prefill_cutlass_w4a4() {
                    read_input_scale(gpu, store, layer, &base)
                } else {
                    0.0
                };
                Ok(Nvfp4Proj {
                    packed: w.ptr,
                    scale: scale.ptr,
                    scale_2: s2,
                    input_scale,
                })
            }
            WeightDtype::BF16 => quantize_expert_proj(gpu, store, w, &base),
            other => bail!("{base}.weight is {other:?}, expected packed U8 NVFP4 or BF16"),
        }
    };
    Ok(Glm5NextExpertWeights {
        gate_proj: proj("gate_proj")?,
        up_proj: proj("up_proj")?,
        down_proj: proj("down_proj")?,
    })
}

fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: store.get(name)?.ptr,
    })
}

pub(super) fn upload_f32_as_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

#[cfg(test)]
mod prune_tests;

/// 2026-09-25: [`LayerSource::collect`], for the MTP loader.
pub(super) fn layer_source(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
) -> Result<LayerSource> {
    LayerSource::collect(gpu, store, layer)
}

/// 2026-09-25: `upload_f32_as_bf16`, for the MTP loader.
pub fn upload_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    upload_f32_as_bf16(gpu, v)
}

#[cfg(test)]
mod vision_capability_tests;
