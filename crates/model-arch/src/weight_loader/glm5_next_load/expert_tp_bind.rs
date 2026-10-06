// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The expert-TP arm of the routed-expert bind (`METRALE_GLM_EXPERT_TP=1`): this
//! rank's slice of one routed expert, read from its shard and uploaded, so the whole expert set
//! never reaches the device.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Every tensor of a routed expert is deferred (`is_routed_expert_tensor`); a resident one is
//!   refused, not sliced from the device, since holding every expert resident is the cost
//!   expert-TP exists to avoid.
//! - Packed U8 NVFP4: gate_proj/up_proj read only this rank's rows (packed and block scale);
//!   down_proj is read whole and cut by columns on 16-element scale-group boundaries. Global
//!   (`weight_scale_2`) and activation (`input_scale`) scales are kept whole on every rank.
//! - Full-width BF16 (the official export's MTP experts): the whole tensor is quantised first,
//!   exactly as `bind_expert` does, and the NVFP4 result is cut the same way, so the two ranks
//!   share one `scale_2`.
//! - At most one projection's host bytes are held at a time; each is dropped once uploaded.

use super::*;
use crate::glm5next_mlp::expert_tp::{Cut, Nvfp4Host, row_range, slice_nvfp4};

/// 2026-10-05: The ledger label of every buffer [`upload_slice`] adopts.
const EXPERT_TP_LABEL: &str = "glm5_next routed expert, expert-TP slice";

/// 2026-10-05: Routed expert `id` of layer `layer`, rank `cfg.ep_rank`'s expert-TP slice: width
/// `cfg.moe_intermediate` (half of I) of each of gate_proj, up_proj and down_proj.
pub(super) fn bind_expert_tp(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
    cfg: &Glm5NextMlpConfig,
) -> Result<Glm5NextExpertWeights> {
    let proj = |p: &str| -> Result<Nvfp4Proj> {
        let base = format!("mlp.experts.{id}.{p}");
        let host = slice_proj(gpu, store, layer, &base, Cut::of(p), cfg)?;
        upload_slice(gpu, store, host)
    };
    Ok(Glm5NextExpertWeights {
        gate_proj: proj("gate_proj")?,
        up_proj: proj("up_proj")?,
        down_proj: proj("down_proj")?,
    })
}

/// 2026-10-05: This rank's NVFP4 slice of projection `base` (`mlp.experts.{id}.{proj}`) on the
/// host.
fn slice_proj(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    base: &str,
    cut: Cut,
    cfg: &Glm5NextMlpConfig,
) -> Result<Nvfp4Host> {
    let (rank, world) = (cfg.ep_rank, cfg.ep_world_size);
    let wname = qualify(layer, &format!("{base}.weight"));
    let Some(d) = store.deferred(&wname) else {
        bail!(
            "{wname}: METRALE_GLM_EXPERT_TP=1 slices every routed expert from its shard, but \
             this tensor is not deferred (an F16 checkpoint, which is never deferred, or a store \
             loaded without the expert-TP defer rule). Unset the lever"
        );
    };
    match d.dtype {
        WeightDtype::UInt8 => {
            let [rows, half_k] = d.shape[..] else {
                bail!("{wname}: packed NVFP4 must be 2-D, got shape {:?}", d.shape);
            };
            let cols = half_k * 2;
            check_geometry(&wname, rows, cols, cut, cfg)?;
            let sname = qualify(layer, &format!("{base}.weight_scale"));
            let s = store.deferred(&sname).with_context(|| {
                format!("{sname}: expert-TP needs the block scale deferred with its weight")
            })?;
            if s.byte_size() != rows * cols / 16 {
                bail!(
                    "{sname}: {} bytes, expected {} for a [{rows}, {cols}] NVFP4 projection",
                    s.byte_size(),
                    rows * cols / 16
                );
            }
            let scale_2 = scalar_f32(
                gpu,
                store,
                &qualify(layer, &format!("{base}.weight_scale_2")),
            )?;
            // 2026-10-05: Lever off, never looked up (the server skipped the tensors), as in
            // `bind_expert`.
            let input_scale = if metrale_config::glm_moe_prefill_cutlass_w4a4() {
                read_input_scale(gpu, store, layer, base)
            } else {
                0.0
            };
            match cut {
                Cut::Rows => {
                    let pr = row_range(d.byte_size(), rows, rank, world)?;
                    let sr = row_range(s.byte_size(), rows, rank, world)?;
                    Ok(Nvfp4Host {
                        packed: d.read_host_range(pr.start, pr.len())?,
                        scale: s.read_host_range(sr.start, sr.len())?,
                        scale_2,
                        input_scale,
                    })
                }
                Cut::Cols => {
                    let full = Nvfp4Host {
                        packed: d.read_host_range(0, d.byte_size())?,
                        scale: s.read_host_range(0, s.byte_size())?,
                        scale_2,
                        input_scale,
                    };
                    slice_nvfp4(&full, rows, cols, cut, rank, world)
                }
            }
        }
        WeightDtype::BF16 => {
            let [rows, cols] = d.shape[..] else {
                bail!("{wname}: must be 2-D to quantise, got shape {:?}", d.shape);
            };
            check_geometry(&wname, rows, cols, cut, cfg)?;
            let bytes = d
                .read_host_range(0, d.byte_size())
                .with_context(|| format!("{wname}: reading the deferred expert from its shard"))?;
            let values: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            drop(bytes);
            let blob = nvfp4_quant::quantize_to_nvfp4(base, &values, rows, cols)?;
            drop(values);
            // 2026-10-05: As in `bind_expert`: a projection quantised at load has no calibrated
            // activation scale.
            let full = Nvfp4Host {
                packed: blob.packed,
                scale: blob.scales,
                scale_2: blob.scale_2,
                input_scale: 0.0,
            };
            slice_nvfp4(&full, rows, cols, cut, rank, world)
        }
        other => bail!(
            "{wname} was deferred as {other:?}; expert-TP slices packed U8 NVFP4 or BF16 only"
        ),
    }
}

/// 2026-10-05: The whole projection's `[rows, cols]` is `[I, hidden]` for a row cut and
/// `[hidden, I]` for a column cut, with `I = ep_world_size * moe_intermediate`.
fn check_geometry(
    name: &str,
    rows: usize,
    cols: usize,
    cut: Cut,
    cfg: &Glm5NextMlpConfig,
) -> Result<()> {
    let (i_dim, h_dim) = match cut {
        Cut::Rows => (rows, cols),
        Cut::Cols => (cols, rows),
    };
    let full = cfg.ep_world_size * cfg.moe_intermediate;
    if i_dim != full || h_dim != cfg.hidden {
        bail!(
            "{name}: shape [{rows}, {cols}] is not the expert-TP {cut:?} cut of I {full} x \
             hidden {}",
            cfg.hidden
        );
    }
    Ok(())
}

/// 2026-10-05: One F32 or BF16 scalar, read from its shard when deferred (expert-TP's normal
/// case) or back from the device when resident.
fn scalar_f32(gpu: &dyn GpuBackend, store: &WeightStore, name: &str) -> Result<f32> {
    if let Some(d) = store.deferred(name) {
        let numel: usize = d.shape.iter().product();
        let b = d.read_host_bytes()?;
        return match (d.dtype, numel, b.len()) {
            (WeightDtype::FP32, 1, 4) => Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            (WeightDtype::BF16, 1, 2) => {
                Ok(half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            }
            _ => bail!(
                "{name}: expected one F32 or BF16 value, got {:?} of shape {:?}",
                d.dtype,
                d.shape
            ),
        };
    }
    let v = host_f32(gpu, store.get(name)?, name)?;
    let [v] = v[..] else {
        bail!("{name} is not a scalar");
    };
    Ok(v)
}

/// 2026-10-05: Upload one slice's packed codes and block scales; the store's derived-weight
/// ledger adopts both buffers, so the store owns them. The host bytes are dropped on return.
fn upload_slice(gpu: &dyn GpuBackend, store: &WeightStore, h: Nvfp4Host) -> Result<Nvfp4Proj> {
    let up = |b: &[u8]| -> Result<DevicePtr> {
        // 2026-10-06: Into the derived weight arena when this layer planned it
        // (`METRALE_GLM_WEIGHT_ARENA=1`, `expert_arena::plan_expert_layer`); else one allocation.
        if let Some(p) = store.derived().upload_in_arena(gpu, EXPERT_TP_LABEL, b)? {
            return Ok(p);
        }
        let p = gpu.alloc(b.len().max(1))?;
        gpu.copy_h2d(b, p)?;
        store.derived().adopt(EXPERT_TP_LABEL, p, b.len());
        Ok(p)
    };
    Ok(Nvfp4Proj {
        packed: up(&h.packed)?,
        scale: up(&h.scale)?,
        scale_2: h.scale_2,
        input_scale: h.input_scale,
    })
}
