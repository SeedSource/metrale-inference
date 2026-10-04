// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Grouped (per-expert) CUTLASS NVFP4 MoE GEMM wrappers.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Before any call into C, every wrapper checks that each per-expert host
//!   array has `num_experts` entries and that `expert_offsets` has
//!   `num_experts + 1` non-negative, non-decreasing entries.

use anyhow::{Result, bail};

#[cfg(metrale_cutlass)]
use std::ffi::c_void;

#[cfg(metrale_cutlass)]
use super::*;

/// 2026-09-25: Check the host arrays a grouped launch passes to C as bare
/// pointers. The C side reads `num_experts` entries from each per-expert array
/// and `num_experts + 1` from `expert_offsets`, and the lengths do not cross the
/// FFI, so a short slice would be read past its end.
///
/// Outside the `cfg(metrale_cutlass)` arms, so it runs, and is tested, without
/// CUTLASS. `offsets` must also be non-negative and non-decreasing. Its upper
/// bound (`offsets[num_experts] <= M_total`) is the caller's to check: no
/// wrapper takes `M_total`.
fn ensure_group_arrays(
    who: &str,
    num_experts: usize,
    per_expert: &[(&str, usize)],
    offsets: &[i32],
) -> Result<()> {
    if offsets.len() != num_experts + 1 {
        bail!(
            "{who}: expert_offsets len {} != num_experts+1 {}",
            offsets.len(),
            num_experts + 1
        );
    }
    for (name, len) in per_expert {
        if *len != num_experts {
            bail!("{who}: {name} len {len} != num_experts {num_experts}");
        }
    }
    if offsets[0] < 0 {
        bail!("{who}: expert_offsets[0] = {} is negative", offsets[0]);
    }
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            bail!(
                "{who}: expert_offsets is not non-decreasing ({} then {}) — a group would \
                 have a negative row count",
                w[0],
                w[1]
            );
        }
    }
    Ok(())
}

/// 2026-09-25: Per-expert NVFP4 gate and up GEMMs: for each expert with rows,
/// the C side calls `metrale_cutlass_nvfp4_gemm_bf16_act_weight_t` once for gate
/// and once for up, so the result matches `nvfp4_gemm_bf16_act_weight_t` on each
/// expert's rows.
///
/// `a` is bf16 `[M_total, K]`; expert `e` owns rows
/// `[expert_offsets[e], expert_offsets[e+1])`. `*_packed_ptrs` and `*_scale_ptrs`
/// hold one device pointer per expert, in the `pack_bf16_weight_to_nvfp4_t`
/// layout (`[N,K/2]` and `[K/16,N]`). The `*_scale2_vals` and `expert_offsets`
/// slices are host arrays.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up(
    a: u64,
    gate_packed_ptrs: &[u64],
    gate_scale_ptrs: &[u64],
    gate_scale2_vals: &[f32],
    up_packed_ptrs: &[u64],
    up_scale_ptrs: &[u64],
    up_scale2_vals: &[f32],
    c_gate: u64,
    c_up: u64,
    expert_offsets: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let num_experts = gate_packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_gate_up",
        num_experts,
        &[
            ("gate_scale_ptrs", gate_scale_ptrs.len()),
            ("gate_scale2_vals", gate_scale2_vals.len()),
            ("up_packed_ptrs", up_packed_ptrs.len()),
            ("up_scale_ptrs", up_scale_ptrs.len()),
            ("up_scale2_vals", up_scale2_vals.len()),
        ],
        expert_offsets,
    )?;
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_nvfp4_grouped_gate_up(
                a as *const c_void,
                gate_packed_ptrs.as_ptr(),
                gate_scale_ptrs.as_ptr(),
                gate_scale2_vals.as_ptr(),
                up_packed_ptrs.as_ptr(),
                up_scale_ptrs.as_ptr(),
                up_scale2_vals.as_ptr(),
                c_gate as *mut c_void,
                c_up as *mut c_void,
                expert_offsets.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 grouped gate_up failed: status {status}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (
            a,
            gate_packed_ptrs,
            gate_scale_ptrs,
            gate_scale2_vals,
            up_packed_ptrs,
            up_scale_ptrs,
            up_scale2_vals,
            c_gate,
            c_up,
            expert_offsets,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-09-25: Grouped NVFP4 gate and up GEMMs, one `GemmUniversalMode::kGrouped`
/// launch per projection over the active experts. `a` is packed once and
/// shared by both.
///
/// `a` is bf16 and token-major. Group row `i` reads token `sorted_token_ids[i]`,
/// or token `i` when `sorted_token_ids` is 0 (null); expert `e` owns group rows
/// `[expert_offsets_host[e], expert_offsets_host[e+1])`, and `c_gate`/`c_up` are
/// written in that order. `*_packed_ptrs` hold one device pointer per expert to
/// a CUTLASS `[N,K/2]` weight; `*_sfb_ptrs` to its swizzled SFB scales (see
/// `pack_weight_sfb`). `*_scale2_vals` and `expert_offsets_host` are host arrays.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_fused(
    a: u64,
    sorted_token_ids: u64,
    gate_packed_ptrs: &[u64],
    gate_sfb_ptrs: &[u64],
    gate_scale2_vals: &[f32],
    up_packed_ptrs: &[u64],
    up_sfb_ptrs: &[u64],
    up_scale2_vals: &[f32],
    c_gate: u64,
    c_up: u64,
    expert_offsets_host: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let num_experts = gate_packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_gate_up_fused",
        num_experts,
        &[
            ("gate_sfb_ptrs", gate_sfb_ptrs.len()),
            ("gate_scale2_vals", gate_scale2_vals.len()),
            ("up_packed_ptrs", up_packed_ptrs.len()),
            ("up_sfb_ptrs", up_sfb_ptrs.len()),
            ("up_scale2_vals", up_scale2_vals.len()),
        ],
        expert_offsets_host,
    )?;
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_nvfp4_grouped_gate_up_fused(
                a as *const c_void,
                sorted_token_ids as *const i32,
                gate_packed_ptrs.as_ptr(),
                gate_sfb_ptrs.as_ptr(),
                gate_scale2_vals.as_ptr(),
                up_packed_ptrs.as_ptr(),
                up_sfb_ptrs.as_ptr(),
                up_scale2_vals.as_ptr(),
                c_gate as *mut c_void,
                c_up as *mut c_void,
                expert_offsets_host.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 grouped(fused) gate_up failed: status {status}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (
            a,
            sorted_token_ids,
            gate_packed_ptrs,
            gate_sfb_ptrs,
            gate_scale2_vals,
            up_packed_ptrs,
            up_sfb_ptrs,
            up_scale2_vals,
            c_gate,
            c_up,
            expert_offsets_host,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-09-25: Grouped NVFP4 down projection, one `kGrouped` launch. `a` is the
/// bf16 intermediate `[M_total, K]`, already in expert order (no gather).
/// `packed_ptrs` and `sfb_ptrs` hold one device pointer per expert to a
/// `[N,K/2]` weight and its swizzled SFB scales; `scale2_vals` and
/// `expert_offsets_host` are host arrays. Writes `c` `[M_total, N]`.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_down(
    a: u64,
    packed_ptrs: &[u64],
    sfb_ptrs: &[u64],
    scale2_vals: &[f32],
    c: u64,
    expert_offsets_host: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let num_experts = packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_down",
        num_experts,
        &[
            ("sfb_ptrs", sfb_ptrs.len()),
            ("scale2_vals", scale2_vals.len()),
        ],
        expert_offsets_host,
    )?;
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_nvfp4_grouped_down(
                a as *const c_void,
                packed_ptrs.as_ptr(),
                sfb_ptrs.as_ptr(),
                scale2_vals.as_ptr(),
                c as *mut c_void,
                expert_offsets_host.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!("CUTLASS nvfp4 grouped down failed: status {status}");
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (
            a,
            packed_ptrs,
            sfb_ptrs,
            scale2_vals,
            c,
            expert_offsets_host,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-10-03: [`nvfp4_grouped_gate_up_fused`] in W4A4 mode with an NVFP4 global activation
/// scale per expert: `act_gscale_vals[e] > 0` (finite) is expert `e`'s static scale (the
/// checkpoint's `input_scale`, calibrated amax / (6 * 448)), shared by gate and up; any other
/// value makes that expert dynamic (amax / (6 * 448) over the rows of the dynamic experts, on
/// the device). Each 16-value block: UE4M3 scale `amax / 6 / gs` (saturated at 448), E2M1 codes
/// rounded to nearest even; epilogue alpha `scale2[e] * gs`. Not bit-identical to any W4A16 path.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_gate_up_w4a4(
    a: u64,
    sorted_token_ids: u64,
    gate_packed_ptrs: &[u64],
    gate_sfb_ptrs: &[u64],
    gate_scale2_vals: &[f32],
    up_packed_ptrs: &[u64],
    up_sfb_ptrs: &[u64],
    up_scale2_vals: &[f32],
    act_gscale_vals: &[f32],
    c_gate: u64,
    c_up: u64,
    expert_offsets_host: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let num_experts = gate_packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_gate_up_w4a4",
        num_experts,
        &[
            ("gate_sfb_ptrs", gate_sfb_ptrs.len()),
            ("gate_scale2_vals", gate_scale2_vals.len()),
            ("up_packed_ptrs", up_packed_ptrs.len()),
            ("up_sfb_ptrs", up_sfb_ptrs.len()),
            ("up_scale2_vals", up_scale2_vals.len()),
            ("act_gscale_vals", act_gscale_vals.len()),
        ],
        expert_offsets_host,
    )?;
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_nvfp4_grouped_gate_up_w4a4(
                a as *const c_void,
                sorted_token_ids as *const i32,
                gate_packed_ptrs.as_ptr(),
                gate_sfb_ptrs.as_ptr(),
                gate_scale2_vals.as_ptr(),
                up_packed_ptrs.as_ptr(),
                up_sfb_ptrs.as_ptr(),
                up_scale2_vals.as_ptr(),
                act_gscale_vals.as_ptr(),
                c_gate as *mut c_void,
                c_up as *mut c_void,
                expert_offsets_host.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!(
                "CUTLASS nvfp4 grouped W4A4 gate_up failed: status {status} (-2: workspace \
                 {} MiB too small, METRALE_CUTLASS_WORKSPACE_MB)",
                ctx.ws_size >> 20
            );
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (
            a,
            sorted_token_ids,
            gate_packed_ptrs,
            gate_sfb_ptrs,
            gate_scale2_vals,
            up_packed_ptrs,
            up_sfb_ptrs,
            up_scale2_vals,
            act_gscale_vals,
            c_gate,
            c_up,
            expert_offsets_host,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// 2026-10-03: [`nvfp4_grouped_down`] in W4A4 mode, `act_gscale_vals` as in
/// [`nvfp4_grouped_gate_up_w4a4`] (the down projection's own `input_scale`).
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_grouped_down_w4a4(
    a: u64,
    packed_ptrs: &[u64],
    sfb_ptrs: &[u64],
    scale2_vals: &[f32],
    act_gscale_vals: &[f32],
    c: u64,
    expert_offsets_host: &[i32],
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let num_experts = packed_ptrs.len();
    ensure_group_arrays(
        "nvfp4_grouped_down_w4a4",
        num_experts,
        &[
            ("sfb_ptrs", sfb_ptrs.len()),
            ("scale2_vals", scale2_vals.len()),
            ("act_gscale_vals", act_gscale_vals.len()),
        ],
        expert_offsets_host,
    )?;
    #[cfg(metrale_cutlass)]
    {
        let ctx = ctx()?;
        let status = unsafe {
            metrale_cutlass_nvfp4_grouped_down_w4a4(
                a as *const c_void,
                packed_ptrs.as_ptr(),
                sfb_ptrs.as_ptr(),
                scale2_vals.as_ptr(),
                act_gscale_vals.as_ptr(),
                c as *mut c_void,
                expert_offsets_host.as_ptr(),
                num_experts as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            bail!(
                "CUTLASS nvfp4 grouped W4A4 down failed: status {status} (-2: workspace \
                 {} MiB too small, METRALE_CUTLASS_WORKSPACE_MB)",
                ctx.ws_size >> 20
            );
        }
        Ok(())
    }
    #[cfg(not(metrale_cutlass))]
    {
        let _ = (
            a,
            packed_ptrs,
            sfb_ptrs,
            scale2_vals,
            act_gscale_vals,
            c,
            expert_offsets_host,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(test)]
mod tests {
    use super::ensure_group_arrays;

    /// 2026-09-25: Every per-expert array is checked, not only the first.
    #[test]
    fn rejects_any_short_per_expert_array() {
        let offsets = [0i32, 4, 9];
        ensure_group_arrays("t", 2, &[("a", 2), ("b", 2)], &offsets).unwrap();
        // 2026-09-25: Each array one entry short is rejected, and the message
        // names the array.
        let e = ensure_group_arrays("t", 2, &[("a", 1), ("b", 2)], &offsets).unwrap_err();
        assert!(e.to_string().contains("a len 1"), "{e}");
        let e = ensure_group_arrays("t", 2, &[("a", 2), ("b", 1)], &offsets).unwrap_err();
        assert!(e.to_string().contains("b len 1"), "{e}");
        // 2026-09-25: Too long is rejected too: the caller disagrees with
        // `num_experts`.
        assert!(ensure_group_arrays("t", 2, &[("a", 3), ("b", 2)], &offsets).is_err());
    }

    #[test]
    fn rejects_bad_expert_offsets() {
        assert!(
            ensure_group_arrays("t", 2, &[], &[0i32, 4]).is_err(),
            "short"
        );
        assert!(
            ensure_group_arrays("t", 2, &[], &[0i32, 4, 9, 12]).is_err(),
            "long"
        );
        assert!(
            ensure_group_arrays("t", 2, &[], &[-1i32, 4, 9]).is_err(),
            "negative base"
        );
        assert!(
            ensure_group_arrays("t", 2, &[], &[0i32, 9, 4]).is_err(),
            "decreasing => negative group row count"
        );
        // 2026-09-25: Equal consecutive offsets are an expert with no rows, which
        // is accepted.
        ensure_group_arrays("t", 2, &[], &[0i32, 4, 4]).unwrap();
    }

    /// 2026-09-25: Zero experts with a one-entry `offsets` passes the check
    /// without a panic.
    #[test]
    fn zero_experts_is_not_a_panic() {
        ensure_group_arrays("t", 0, &[], &[0i32]).unwrap();
    }
}
