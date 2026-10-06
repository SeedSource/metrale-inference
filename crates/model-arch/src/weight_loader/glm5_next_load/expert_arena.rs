// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: GLM-5.3's use of the weight arena (`METRALE_GLM_WEIGHT_ARENA=1`): which
//! checkpoint tensors the fast loader may place in the store's arena, and the per-layer plan of
//! the derived arena the routed-expert binders upload into.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - [`arena_rule`] claims only routed-expert tensors that are not BF16: the U8 packed
//!   weights, their E4M3 block scales and F32 scalars, which `bind_expert` binds zero-copy or
//!   reads on the host and which nothing frees before teardown (`is_reuploaded` never matches
//!   `mlp.experts.*`; `is_quantized_expert_weight`, the only expert family freed after load,
//!   is BF16).
//! - [`plan_expert_layer`] lists, in bind order (ascending local expert id; gate, up, down;
//!   packed then block scale), the exact byte size of every buffer `upload_bytes`
//!   (`expert_quant.rs`) or `upload_slice` (`expert_tp_bind.rs`) will upload for the layer, so
//!   its chunk is filled exactly. A projection bound zero-copy (resident U8) uploads nothing
//!   and is not listed.
//! - Lever off, nothing is planned: the derived arena stays disabled and both upload sites run
//!   their per-buffer `alloc` + `adopt` unchanged.

use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::{is_routed_expert_tensor, qualify};
use crate::glm5next_mlp::Glm5NextMlpConfig;
use crate::glm5next_mlp::expert_tp::Cut;

/// 2026-10-06: The boot-log label of the derived arena.
pub const DERIVED_EXPERT_ARENA_LABEL: &str = "glm5_next routed experts uploaded at bind";

/// 2026-10-06: The fast loader's arena rule: a routed-expert tensor of any layer that is not
/// full-width BF16 (see the module invariants).
pub fn arena_rule(name: &str, dtype: WeightDtype) -> bool {
    dtype != WeightDtype::BF16 && is_routed_expert_tensor(name)
}

/// 2026-10-06: The `[rows, cols]` of the NVFP4 projection `proj` of expert `id` that this rank
/// uploads at bind time, or `None` when it uploads nothing for it (resident U8, bound
/// zero-copy) or the tensor is absent or not 2-D (the binder reports that).
fn uploaded_shape(
    store: &WeightStore,
    layer: usize,
    id: usize,
    proj: &str,
    cfg: &Glm5NextMlpConfig,
) -> Option<(usize, usize)> {
    if cfg.is_expert_tp() {
        // `bind_expert_tp` uploads this rank's slice, width `moe_intermediate`, whatever the
        // stored dtype: gate/up `[I/2, hidden]` by rows, down `[hidden, I/2]` by columns.
        let (i, h) = (cfg.moe_intermediate, cfg.hidden);
        return Some(match Cut::of(proj) {
            Cut::Rows => (i, h),
            Cut::Cols => (h, i),
        });
    }
    let wname = qualify(layer, &format!("mlp.experts.{id}.{proj}.weight"));
    // `bind_expert`: deferred BF16 and resident BF16 are quantised and uploaded whole.
    let (dtype, shape) = match store.deferred(&wname) {
        Some(d) => (d.dtype, d.shape.as_slice()),
        None => {
            let t = store.get(&wname).ok()?;
            (t.dtype, t.shape.as_slice())
        }
    };
    match (dtype, shape) {
        (WeightDtype::BF16, &[rows, cols]) => Some((rows, cols)),
        _ => None,
    }
}

/// 2026-10-06: The byte sizes, in upload order, of the buffers layer `layer`'s routed-expert
/// bind will upload on this rank: per projection, `rows * cols / 2` packed codes then
/// `rows * cols / 16` E4M3 block scales (`Nvfp4Blob`, `Nvfp4Host`).
pub fn expert_upload_sizes(
    store: &WeightStore,
    layer: usize,
    cfg: &Glm5NextMlpConfig,
) -> Vec<usize> {
    let mut sizes = Vec::new();
    for id in cfg.local_expert_range() {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            if let Some((rows, cols)) = uploaded_shape(store, layer, id, proj, cfg) {
                sizes.push(rows * cols / 2);
                sizes.push(rows * cols / 16);
            }
        }
    }
    sizes
}

/// 2026-10-06: Plan the store's derived arena for layer `layer`'s routed experts when `enabled`
/// (production passes `metrale_config::glm_weight_arena()`); call right before the layer's
/// `build_moe`. Returns the planned bytes; 0, planning nothing, when disabled or when the layer
/// uploads nothing (every expert bound zero-copy).
pub fn plan_expert_layer(
    store: &WeightStore,
    layer: usize,
    cfg: &Glm5NextMlpConfig,
    enabled: bool,
) -> usize {
    if !enabled {
        return 0;
    }
    let sizes = expert_upload_sizes(store, layer, cfg);
    if sizes.is_empty() {
        return 0;
    }
    store
        .derived()
        .arena()
        .plan(DERIVED_EXPERT_ARENA_LABEL, sizes)
}
