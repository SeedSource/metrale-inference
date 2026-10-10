// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `cutlass` module of a build without the `cuda` feature. It
//! keeps the entry points metrale-model-layers calls without a `cfg`, so that
//! build compiles.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Every function body is `unreachable!`: reaching one is a bug.

use anyhow::Result;

pub fn bf16_gemm_act_weight_t(
    _act: u64,
    _weight: u64,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::bf16_gemm_act_weight_t is cuda-only (not built for metal)")
}

pub fn nvfp4_gemm_bf16_act_weight_t(
    _act: u64,
    _weight_packed_t: u64,
    _weight_scale_t: u64,
    _weight_scale_2: f32,
    _out: u64,
    _m: u32,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_gemm_bf16_act_weight_t is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_fused(
    _a: u64,
    _sorted_token_ids: u64,
    _gate_packed_ptrs: &[u64],
    _gate_sfb_ptrs: &[u64],
    _gate_scale2_vals: &[f32],
    _up_packed_ptrs: &[u64],
    _up_sfb_ptrs: &[u64],
    _up_scale2_vals: &[f32],
    _c_gate: u64,
    _c_up: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_gate_up_fused is cuda-only (not built for metal)")
}

pub fn nvfp4_grouped_down(
    _a: u64,
    _packed_ptrs: &[u64],
    _sfb_ptrs: &[u64],
    _scale2_vals: &[f32],
    _c: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_down is cuda-only (not built for metal)")
}

pub fn pack_bf16_weight_to_nvfp4_t(
    _weight_bf16: u64,
    _packed_t: u64,
    _scale_t: u64,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_bf16_weight_to_nvfp4_t is cuda-only (not built for metal)")
}

pub fn pack_weight_sfb(
    _scale_in: u64,
    _scale_out: u64,
    _n: u32,
    _k: u32,
    _src_n_major: bool,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_weight_sfb is cuda-only (not built for metal)")
}

pub fn transpose_nvfp4_packed_kton(
    _src_packed_t: u64,
    _dst_packed: u64,
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::transpose_nvfp4_packed_kton is cuda-only (not built for metal)")
}

/// 2026-10-03: No CUTLASS objects without the `cuda` feature.
pub fn available() -> bool {
    false
}

/// 2026-10-03: As the cuda module's: there is no workspace without CUTLASS.
pub fn warm_workspace() -> Result<usize> {
    anyhow::bail!("CUTLASS support was not built (no `cuda` feature)")
}

/// 2026-10-03: The same formula as the cuda module's `sfb_bytes`.
pub fn sfb_bytes(n: usize, k: usize) -> usize {
    n.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4
}

#[allow(clippy::too_many_arguments)]
pub fn pack_weight_sfb_batched(
    _scale_ptrs_dev: u64,
    _first: u32,
    _count: u32,
    _out_base: u64,
    _out_stride: usize,
    _n: u32,
    _k: u32,
    _src_n_major: bool,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_weight_sfb_batched is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn pack_weight_sfb_batched_mode(
    _scale_ptrs_dev: u64,
    _first: u32,
    _count: u32,
    _out_base: u64,
    _out_stride: usize,
    _n: u32,
    _k: u32,
    _src_n_major: bool,
    _mode: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::pack_weight_sfb_batched_mode is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_w4a4(
    _a: u64,
    _sorted_token_ids: u64,
    _gate_packed_ptrs: &[u64],
    _gate_sfb_ptrs: &[u64],
    _gate_scale2_vals: &[f32],
    _up_packed_ptrs: &[u64],
    _up_sfb_ptrs: &[u64],
    _up_scale2_vals: &[f32],
    _act_gscale_vals: &[f32],
    _c_gate: u64,
    _c_up: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_gate_up_w4a4 is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_down_w4a4(
    _a: u64,
    _packed_ptrs: &[u64],
    _sfb_ptrs: &[u64],
    _scale2_vals: &[f32],
    _act_gscale_vals: &[f32],
    _c: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _stream: u64,
) -> Result<()> {
    unreachable!("cutlass::nvfp4_grouped_down_w4a4 is cuda-only (not built for metal)")
}

pub fn set_w4a4_amax_dedup_override(_force: Option<bool>) {
    unreachable!("cutlass::set_w4a4_amax_dedup_override is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_w4a4_ex(
    _a: u64,
    _sorted_token_ids: u64,
    _gate_packed_ptrs: &[u64],
    _gate_sfb_ptrs: &[u64],
    _gate_scale2_vals: &[f32],
    _up_packed_ptrs: &[u64],
    _up_sfb_ptrs: &[u64],
    _up_scale2_vals: &[f32],
    _act_gscale_vals: &[f32],
    _c_gate: u64,
    _c_up: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _num_tokens: usize,
    _pack_once: bool,
    _stream: u64,
) -> Result<bool> {
    unreachable!("cutlass::nvfp4_grouped_gate_up_w4a4_ex is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_down_w4a4_ex(
    _a: u64,
    _packed_ptrs: &[u64],
    _sfb_ptrs: &[u64],
    _scale2_vals: &[f32],
    _act_gscale_vals: &[f32],
    _c: u64,
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _pre_amax: Option<(u64, std::ops::Range<usize>)>,
    _stream: u64,
) -> Result<bool> {
    unreachable!("cutlass::nvfp4_grouped_down_w4a4_ex is cuda-only (not built for metal)")
}

pub fn set_w4a4_pack_compact_override(_force: Option<bool>) {
    unreachable!("cutlass::set_w4a4_pack_compact_override is cuda-only (not built for metal)")
}

pub fn w4a4_last_pack_compact() -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
pub fn w4a4_pack_only(
    _a: u64,
    _sorted_token_ids: u64,
    _valid_ptrs: &[u64],
    _act_gscale_vals: &[f32],
    _expert_offsets_host: &[i32],
    _n: u32,
    _k: u32,
    _num_tokens: usize,
    _pack_once: bool,
    _stream: u64,
) -> Result<bool> {
    unreachable!("cutlass::w4a4_pack_only is cuda-only (not built for metal)")
}

pub fn w4a4_pack_replay(_compact: bool, _reps: usize, _stream: u64) -> Result<()> {
    unreachable!("cutlass::w4a4_pack_replay is cuda-only (not built for metal)")
}

pub fn set_w4a4_pack_once_fault(_on: bool) {
    unreachable!("cutlass::set_w4a4_pack_once_fault is cuda-only (not built for metal)")
}

/// 2026-10-06: Mirrors the cuda build's `W4a4LastPrep`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct W4a4LastPrep {
    pub ws_base: u64,
    pub sfa_off: u64,
    pub sfa_bytes: u64,
    pub gs_off: u64,
    pub groups: u64,
    pub pack_once: bool,
    pub pre_amax: bool,
}

pub fn w4a4_last_prep() -> Option<W4a4LastPrep> {
    None
}

pub fn workspace() -> Result<(u64, usize)> {
    unreachable!("cutlass::workspace is cuda-only (not built for metal)")
}
