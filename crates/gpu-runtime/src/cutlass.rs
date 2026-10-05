// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-side FFI to the CUTLASS GEMM wrappers in `cuda/cutlass_*.cu`:
//! the shared `extern` block and workspace here; dense BF16 and NVFP4 GEMMs in
//! `gemm`, grouped per-expert NVFP4 MoE GEMMs in `grouped`, and NVFP4 weight
//! pack, SFB swizzle and transpose in `pack`.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The FFI and the workspace exist only under `cfg(metrale_cutlass)`, which
//!   `build.rs` sets only when `CUTLASS_HOME` is set at build time. Without it
//!   every public wrapper returns an error.
//! - A wrapper that returns `Ok` got status 0 from its C entry point.

#[cfg(metrale_cutlass)]
use anyhow::{Result, bail};

#[cfg(metrale_cutlass)]
use std::ffi::c_void;
#[cfg(metrale_cutlass)]
use std::sync::OnceLock;

mod gemm;
mod grouped;
mod pack;

pub use gemm::{bf16_gemm_act_weight_t, nvfp4_gemm_bf16_act_weight_t};
pub use grouped::{
    nvfp4_grouped_down, nvfp4_grouped_down_w4a4, nvfp4_grouped_gate_up,
    nvfp4_grouped_gate_up_fused, nvfp4_grouped_gate_up_w4a4,
};
pub use pack::{
    pack_bf16_weight_to_nvfp4_t, pack_weight_sfb, pack_weight_sfb_batched, pack_weight_sfb_batched_mode, sfb_bytes,
    transpose_nvfp4_packed_kton,
};

/// 2026-10-03: Whether this build carries the CUTLASS objects (`CUTLASS_HOME` was set at build
/// time, `cfg(metrale_cutlass)`). Without them every wrapper here returns an error.
pub fn available() -> bool {
    cfg!(metrale_cutlass)
}

/// 2026-10-03: Allocate the shared CUTLASS workspace now (it is otherwise allocated on first
/// use) and return its size in bytes, so a caller can reserve it at load, before the KV pool is
/// sized. Errors without CUTLASS or when the allocation fails.
pub fn warm_workspace() -> anyhow::Result<usize> {
    #[cfg(metrale_cutlass)]
    {
        Ok(ctx()?.ws_size)
    }
    #[cfg(not(metrale_cutlass))]
    {
        anyhow::bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(all(test, metrale_cutlass))]
mod tests;

#[cfg(metrale_cutlass)]
unsafe extern "C" {
    pub(crate) fn metrale_cutlass_bf16_gemm_act_weight_t(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_nvfp4_gemm_bf16_act_weight_t(
        act: *const c_void,
        weight_packed_t: *const c_void,
        weight_scale_t: *const c_void,
        weight_scale_2: f32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_pack_bf16_weight_to_nvfp4_t(
        weight_bf16: *const c_void,
        packed_t: *mut c_void,
        scale_t: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_nvfp4_grouped_gate_up(
        a_bf16: *const c_void,
        gate_packed_ptrs: *const u64,
        gate_scale_ptrs: *const u64,
        gate_scale2_vals: *const f32,
        up_packed_ptrs: *const u64,
        up_scale_ptrs: *const u64,
        up_scale2_vals: *const f32,
        c_gate_bf16: *mut c_void,
        c_up_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_nvfp4_grouped_gate_up_fused(
        a_bf16: *const c_void,
        sorted_token_ids: *const i32,
        gate_packed_ptrs: *const u64,
        gate_sfb_ptrs: *const u64,
        gate_scale2_vals: *const f32,
        up_packed_ptrs: *const u64,
        up_sfb_ptrs: *const u64,
        up_scale2_vals: *const f32,
        c_gate_bf16: *mut c_void,
        c_up_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_nvfp4_grouped_down(
        a_bf16: *const c_void,
        packed_ptrs: *const u64,
        sfb_ptrs: *const u64,
        scale2_vals: *const f32,
        c_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_pack_weight_sfb(
        scale_in: *const c_void,
        scale_out: *mut c_void,
        n: i32,
        k: i32,
        src_n_major: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_pack_weight_sfb_batched(
        scale_ptrs_dev: *const u64,
        first: i32,
        count: i32,
        out_base: *mut c_void,
        out_stride: u64,
        n: i32,
        k: i32,
        src_n_major: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_pack_weight_sfb_batched_mode(
        scale_ptrs_dev: *const u64,
        first: i32,
        count: i32,
        out_base: *mut c_void,
        out_stride: u64,
        n: i32,
        k: i32,
        src_n_major: i32,
        mode: i32,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_nvfp4_grouped_gate_up_w4a4(
        a_bf16: *const c_void,
        sorted_token_ids: *const i32,
        gate_packed_ptrs: *const u64,
        gate_sfb_ptrs: *const u64,
        gate_scale2_vals: *const f32,
        up_packed_ptrs: *const u64,
        up_sfb_ptrs: *const u64,
        up_scale2_vals: *const f32,
        act_gscale_vals: *const f32,
        c_gate_bf16: *mut c_void,
        c_up_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_nvfp4_grouped_down_w4a4(
        a_bf16: *const c_void,
        packed_ptrs: *const u64,
        sfb_ptrs: *const u64,
        scale2_vals: *const f32,
        act_gscale_vals: *const f32,
        c_bf16: *mut c_void,
        expert_offsets_host: *const i32,
        num_experts: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    pub(crate) fn metrale_cutlass_transpose_nvfp4_packed_kton(
        src_packed_t: *const c_void,
        dst_packed: *mut c_void,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn metrale_cutlass_bf16_gemm_act_weight_t_128x256(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn metrale_cutlass_bf16_gemm_act_weight_t_256x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn metrale_cutlass_bf16_gemm_act_weight_t_64x128(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn metrale_cutlass_bf16_gemm_act_weight_t_128x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn metrale_cutlass_bf16_gemm_act_weight_t_64x64(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn metrale_cublaslt_bf16_gemm_act_weight_t_algo(
        act: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
        algo_index: i32,
        returned_count: *mut i32,
    ) -> i32;
    pub(crate) fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
}

#[cfg(metrale_cutlass)]
pub(crate) struct Ctx {
    pub(crate) workspace: u64,
    pub(crate) ws_size: usize,
}

#[cfg(metrale_cutlass)]
unsafe impl Send for Ctx {}
#[cfg(metrale_cutlass)]
unsafe impl Sync for Ctx {}

#[cfg(metrale_cutlass)]
/// 2026-09-25: The shared CUTLASS workspace, allocated on first use and never
/// freed. Static like the process CUDA context (`crate::cuda_host`): its size
/// comes from a fixed budget, not from any model, so it is kept across model
/// loads.
static CTX: OnceLock<Ctx> = OnceLock::new();

#[cfg(metrale_cutlass)]
pub(crate) fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    // 2026-09-25: Scratch for every CUTLASS wrapper; the grouped NVFP4 path
    // carves packed A, SFA, per-group arrays and the GEMM workspace from it.
    // 512 MiB unless `METRALE_CUTLASS_WORKSPACE_MB` parses as a number.
    let ws_size = std::env::var("METRALE_CUTLASS_WORKSPACE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        * 1024
        * 1024;
    let mut workspace = 0u64;
    let status = unsafe { cuMemAlloc_v2(&mut workspace, ws_size) };
    if status != 0 {
        bail!("cuMemAlloc CUTLASS workspace failed: {status}");
    }
    let _ = CTX.set(Ctx { workspace, ws_size });
    Ok(CTX.get().unwrap())
}
