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
use crate::glm5next_mlp::expert_tp::{Cut, Nvfp4Host, col_part, row_range, slice_nvfp4};
use crate::glm5next_mlp::scale_rows::{self, ScaleRows};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    bind_expert_tp_csf(gpu, store, layer, id, cfg, csf_scales_enabled(cfg)?)
}

/// 2026-10-17: [`bind_expert_tp`] with the csfcar choice made by the caller (tests drive both
/// arms without the process-wide lever).
pub(super) fn bind_expert_tp_csf(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
    cfg: &Glm5NextMlpConfig,
    csf: bool,
) -> Result<Glm5NextExpertWeights> {
    let proj = |p: &str| -> Result<Nvfp4Proj> {
        let base = format!("mlp.experts.{id}.{p}");
        let host = slice_proj(gpu, store, layer, &base, Cut::of(p), cfg)?;
        // 2026-10-17: `METRALE_GLM_MOE_CSF_SCALES=1`: a checkpoint NVFP4 scale (not one
        // quantised here from BF16) is held as scale rows: the encoding `precode_csf_layer` made
        // for the arena plan, or encoded now.
        let rows = if csf && is_u8_checkpoint(store, layer, &base) {
            Some(match take_pending(layer, id, p) {
                Some(r) => r,
                None => encode_rank_scale(&host.scale, Cut::of(p), cfg)
                    .with_context(|| qualify(layer, &base))?,
            })
        } else {
            None
        };
        upload_slice(gpu, store, host, rows)
    };
    Ok(Glm5NextExpertWeights {
        gate_proj: proj("gate_proj")?,
        up_proj: proj("up_proj")?,
        down_proj: proj("down_proj")?,
    })
}

/// 2026-10-16: A projection's block scale: the deferred plain tensor (read by range, as before),
/// or the whole decoded csf_32x2_v1 scale.
enum ScaleSrc<'a> {
    Plain(&'a metrale_model_weights::weights::DeferredTensor),
    Decoded(Vec<u8>),
}

impl ScaleSrc<'_> {
    fn byte_size(&self) -> usize {
        match self {
            Self::Plain(d) => d.byte_size(),
            Self::Decoded(v) => v.len(),
        }
    }
    fn read_host_range(&self, start: usize, len: usize) -> Result<Vec<u8>> {
        match self {
            Self::Plain(d) => d.read_host_range(start, len),
            Self::Decoded(v) => v
                .get(start..start + len)
                .map(<[u8]>::to_vec)
                .with_context(|| {
                    format!(
                        "decoded scale range {start}+{len} outside {} bytes",
                        v.len()
                    )
                }),
        }
    }
}

/// 2026-10-16: `base`'s csf_32x2_v1 scale decoded to plain `[rows, c]` E4M3 bytes, or `None` when
/// the projection has no `.weight_scale.csf32_bases` (every checkpoint but FP4-Local). All five
/// parts are routed-expert tensors, so deferred with the weight; a missing part is an error.
fn csf32_scale(
    store: &WeightStore,
    layer: usize,
    base: &str,
    rows: usize,
    c: usize,
) -> Result<Option<Vec<u8>>> {
    let part = |p: &str| qualify(layer, &format!("{base}.weight_scale.csf32_{p}"));
    if store.deferred(&part("bases")).is_none() {
        return Ok(None);
    }
    let read = |p: &str| -> Result<Vec<u8>> {
        let n = part(p);
        let d = store
            .deferred(&n)
            .with_context(|| format!("{n}: csf_32x2_v1 part missing or not deferred"))?;
        d.read_host_range(0, d.byte_size())
            .with_context(|| format!("{n}: reading from its shard"))
    };
    let u16s = |v: Vec<u8>| {
        v.chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect::<Vec<_>>()
    };
    let (bases, codes) = (read("bases")?, read("codes")?);
    let offsets = u16s(read("offsets")?);
    let page_bases: Vec<u32> = read("page_bases")?
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect();
    let exceptions = u16s(read("exceptions")?);
    let parts = super::csf32::Csf32Parts {
        bases: &bases,
        codes: &codes,
        offsets: &offsets,
        page_bases: &page_bases,
        exceptions: &exceptions,
    };
    super::csf32::decode(&qualify(layer, base), rows, c, &parts).map(Some)
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
            // 2026-10-16: FP4-Local ships the block scale as csf_32x2_v1 parts: decoded whole here
            // (before the row cut), then sliced like a plain scale. A plain scale is read exactly
            // as before (only this rank's rows).
            let s = match csf32_scale(store, layer, base, rows, cols / 16)? {
                Some(full) => ScaleSrc::Decoded(full),
                None => ScaleSrc::Plain(store.deferred(&sname).with_context(|| {
                    format!("{sname}: expert-TP needs the block scale deferred with its weight")
                })?),
            };
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
/// 2026-10-17: With `rows` (csfcar), the block scale uploaded is that encoding, and the
/// projection's scale pointer is its tagged row section (`ScaleRows::device_ptr`).
fn upload_slice(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    h: Nvfp4Host,
    rows: Option<ScaleRows>,
) -> Result<Nvfp4Proj> {
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
    let scale = match rows {
        Some(r) => DevicePtr(r.device_ptr(up(&r.bytes)?.0)),
        None => up(&h.scale)?,
    };
    Ok(Nvfp4Proj {
        packed: up(&h.packed)?,
        scale,
        scale_2: h.scale_2,
        input_scale: h.input_scale,
    })
}

// ---------------------------------------------------------------------------------------------
// 2026-10-17: csfcar (`METRALE_GLM_MOE_CSF_SCALES=1`): routed-expert block scales held as scale
// rows (`glm5next_mlp::scale_rows`). The arena plan (`expert_arena::expert_upload_sizes`) needs
// each buffer's exact size before the bind, and an encoded size depends on the bytes, so
// `precode_csf_layer` encodes this rank's scale slices of a layer first and parks them here;
// the bind takes them (`take_pending`). Without the arena the bind encodes on its own.
// Invariants: the parked map is empty after each layer's bind (`precode_csf_layer` refuses a
// leftover); an encoding is of exactly the bytes `slice_proj` uploads plain (same `ScaleSrc`,
// same row range or `col_part`), so the plan and the bind agree.
// ---------------------------------------------------------------------------------------------

type PendingKey = (usize, usize, String);
static PENDING: Mutex<BTreeMap<PendingKey, ScaleRows>> = Mutex::new(BTreeMap::new());
static CSF_PLAIN: AtomicUsize = AtomicUsize::new(0);
static CSF_HELD: AtomicUsize = AtomicUsize::new(0);
static CSF_EXC: AtomicUsize = AtomicUsize::new(0);

/// 2026-10-17: Whether the bind holds checkpoint scales as scale rows: the lever, checked
/// against what it needs (expert-TP) and what it cannot serve (the W4A16 prefill-MMA kernel
/// reads plain scales).
pub(super) fn csf_scales_enabled(cfg: &Glm5NextMlpConfig) -> Result<bool> {
    if !metrale_config::glm_moe_csf_scales() {
        return Ok(false);
    }
    if !cfg.is_expert_tp() {
        bail!(
            "METRALE_GLM_MOE_CSF_SCALES=1 needs METRALE_GLM_EXPERT_TP=1 (it encodes the expert-TP slices); unset one"
        );
    }
    if std::env::var("METRALE_GLM_MOE_PREFILL_GROUPED_W4A16").as_deref() == Ok("1") {
        bail!(
            "METRALE_GLM_MOE_CSF_SCALES=1 with METRALE_GLM_MOE_PREFILL_GROUPED_W4A16=1: moe_w4a16_prefill_mma reads plain scales; unset one"
        );
    }
    Ok(true)
}

/// Whether `base`'s weight is a packed U8 NVFP4 checkpoint tensor (its scale comes from the
/// checkpoint; a BF16 one is quantised at bind and keeps plain scales).
fn is_u8_checkpoint(store: &WeightStore, layer: usize, base: &str) -> bool {
    store
        .deferred(&qualify(layer, &format!("{base}.weight")))
        .is_some_and(|d| d.dtype == WeightDtype::UInt8)
}

/// 2026-10-17: Encode this rank's plain scale slice (`[rows, c]` after the cut).
fn encode_rank_scale(scale: &[u8], cut: Cut, cfg: &Glm5NextMlpConfig) -> Result<ScaleRows> {
    let (rows, c) = match cut {
        Cut::Rows => (cfg.moe_intermediate, cfg.hidden / 16),
        Cut::Cols => (cfg.hidden, cfg.moe_intermediate / 16),
    };
    let r = scale_rows::encode(rows, c, scale)?;
    CSF_PLAIN.fetch_add(scale.len(), Ordering::Relaxed);
    CSF_HELD.fetch_add(r.bytes.len(), Ordering::Relaxed);
    CSF_EXC.fetch_add(r.exceptions, Ordering::Relaxed);
    Ok(r)
}

/// This rank's plain scale slice of projection `base`, read exactly as `slice_proj` reads it.
fn rank_scale(
    store: &WeightStore,
    layer: usize,
    base: &str,
    cut: Cut,
    cfg: &Glm5NextMlpConfig,
) -> Result<Vec<u8>> {
    let wname = qualify(layer, &format!("{base}.weight"));
    let d = store
        .deferred(&wname)
        .with_context(|| format!("{wname}: not deferred"))?;
    let [rows, half_k] = d.shape[..] else {
        bail!("{wname}: packed NVFP4 must be 2-D, got shape {:?}", d.shape);
    };
    let cols = half_k * 2;
    let sname = qualify(layer, &format!("{base}.weight_scale"));
    let s = match csf32_scale(store, layer, base, rows, cols / 16)? {
        Some(full) => ScaleSrc::Decoded(full),
        None => ScaleSrc::Plain(
            store
                .deferred(&sname)
                .with_context(|| format!("{sname}: not deferred"))?,
        ),
    };
    let (rank, world) = (cfg.ep_rank, cfg.ep_world_size);
    match cut {
        Cut::Rows => {
            let sr = row_range(s.byte_size(), rows, rank, world)?;
            s.read_host_range(sr.start, sr.len())
        }
        Cut::Cols => col_part(&s.read_host_range(0, s.byte_size())?, rows, rank, world),
    }
}

/// 2026-10-17: Encode and park every checkpoint-NVFP4 scale slice layer `layer` will bind on
/// this rank (lever on); the arena plan reads their sizes ([`pending_len`]). Logs the layer's
/// totals. A no-op with the lever off.
pub(super) fn precode_csf_layer(
    store: &WeightStore,
    layer: usize,
    cfg: &Glm5NextMlpConfig,
) -> Result<()> {
    if !csf_scales_enabled(cfg)? {
        return Ok(());
    }
    precode_csf_layer_on(store, layer, cfg)
}

/// 2026-10-17: [`precode_csf_layer`]'s body, lever check left to the caller (tests).
pub(super) fn precode_csf_layer_on(
    store: &WeightStore,
    layer: usize,
    cfg: &Glm5NextMlpConfig,
) -> Result<()> {
    let mut map = PENDING
        .lock()
        .map_err(|_| anyhow::anyhow!("csf scale cache poisoned"))?;
    if !map.is_empty() {
        bail!(
            "glm5_next layer {layer}: {} encoded scales left over from an earlier layer's bind",
            map.len()
        );
    }
    let (p0, h0, e0) = (
        CSF_PLAIN.load(Ordering::Relaxed),
        CSF_HELD.load(Ordering::Relaxed),
        CSF_EXC.load(Ordering::Relaxed),
    );
    for id in cfg.local_expert_range() {
        for p in ["gate_proj", "up_proj", "down_proj"] {
            let base = format!("mlp.experts.{id}.{p}");
            if !is_u8_checkpoint(store, layer, &base) {
                continue;
            }
            let cut = Cut::of(p);
            let plain =
                rank_scale(store, layer, &base, cut, cfg).with_context(|| qualify(layer, &base))?;
            let r = encode_rank_scale(&plain, cut, cfg).with_context(|| qualify(layer, &base))?;
            map.insert((layer, id, p.to_string()), r);
        }
    }
    let (p1, h1, e1) = (
        CSF_PLAIN.load(Ordering::Relaxed),
        CSF_HELD.load(Ordering::Relaxed),
        CSF_EXC.load(Ordering::Relaxed),
    );
    if p0 == 0 && p1 > 0 {
        tracing::warn!(
            "METRALE_GLM_MOE_CSF_SCALES=1: ENGAGED - routed-expert block scales held as scale rows (moe_scale_byte)"
        );
    }
    if p1 > p0 {
        tracing::info!(
            "glm5_next layer {layer}: csf scales {} projections, {} -> {} bytes ({:.2}%), {} exceptions; rank total {} -> {} bytes",
            map.len(),
            p1 - p0,
            h1 - h0,
            100.0 * (h1 - h0) as f64 / (p1 - p0) as f64,
            e1 - e0,
            p1,
            h1
        );
    }
    Ok(())
}

/// 2026-10-17: The parked encoding's size for (layer, expert, projection), if any.
pub(super) fn pending_len(layer: usize, id: usize, proj: &str) -> Option<usize> {
    PENDING
        .lock()
        .ok()?
        .get(&(layer, id, proj.to_string()))
        .map(|r| r.bytes.len())
}

fn take_pending(layer: usize, id: usize, proj: &str) -> Option<ScaleRows> {
    PENDING.lock().ok()?.remove(&(layer, id, proj.to_string()))
}

/// 2026-10-17: Plain vs held scale bytes and exceptions encoded so far on this rank (boot log,
/// tests).
#[allow(dead_code)]
pub(super) fn csf_totals() -> (usize, usize, usize) {
    (
        CSF_PLAIN.load(Ordering::Relaxed),
        CSF_HELD.load(Ordering::Relaxed),
        CSF_EXC.load(Ordering::Relaxed),
    )
}
