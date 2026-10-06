// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The fast loader's half of the weight arena (`METRALE_GLM_WEIGHT_ARENA=1`): which
//! of a shard's tensors go into the arena, the shard's exact chunk plan, and their slots.
//!
//! Owner: model-weights (fast loader).
//! Invariants:
//! - A tensor goes into the arena only when the model's [`crate::weights::ArenaHook`] claims it
//!   (asked with the store dtype, as the defer hook is).
//! - The plan lists the claimed tensors' sizes in the order the copier uploads them (the
//!   shard's retained order), so every chunk is exactly filled.
//! - Without an arena, or when the arena declines (`Ok(None)`), the tensor takes the
//!   unchanged per-tensor path in `load_shard_fast`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::header::TensorMeta;
use crate::weights::{WeightArena, WeightDtype};

/// 2026-10-06: The boot-log label of the fast loader's arena.
pub(super) const FAST_ARENA_LABEL: &str = "checkpoint tensors claimed by the model loader";

/// 2026-10-06: An arena and the predicate that picks its tensors, or `None` (lever off).
pub(super) type ArenaSel<'a> = Option<(&'a WeightArena, &'a dyn Fn(&str, WeightDtype) -> bool)>;

/// 2026-10-06: Which of `tensors` (the shard's retained list, in copy order) go into the arena;
/// plans the arena for exactly those. All false, planning nothing, without an arena.
pub(super) fn plan_shard(arena: ArenaSel<'_>, tensors: &[TensorMeta]) -> Vec<bool> {
    let Some((arena, pick)) = arena else {
        return vec![false; tensors.len()];
    };
    let flags: Vec<bool> = tensors.iter().map(|t| pick(&t.name, t.dtype)).collect();
    let sizes = tensors
        .iter()
        .zip(&flags)
        .filter(|(_, f)| **f)
        .map(|(t, _)| t.len);
    arena.plan(FAST_ARENA_LABEL, sizes);
    flags
}

/// 2026-10-06: An arena slot of `bytes` when `in_arena`; `Ok(None)` when not, or when the arena
/// declines (a failed chunk), so the caller allocates the tensor alone. The caller copies.
pub(super) fn alloc(
    arena: ArenaSel<'_>,
    in_arena: bool,
    gpu: &dyn GpuBackend,
    bytes: usize,
) -> Result<Option<DevicePtr>> {
    match arena {
        Some((a, _)) if in_arena => a.alloc(gpu, bytes),
        _ => Ok(None),
    }
}
