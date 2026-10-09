// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The routed-expert paths of `forward_moe` that run expert by expert: the per-row
//! sweeps (device- or host-dispatched) and the row-batched union GEMV.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Both paths write only the routed outputs of this rank's experts; `forward_moe` zeroes
//!   `expert_out` before either runs.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::launch::{
    swiglu, union_kernel, w4a16_gemv, w4a16_gemv_moe, w4a16_gemv_moe_batchm,
    w4a16_gemv_moe_batchm_gateup,
};
use super::{Glm5NextMlpWorkspace, announce_dispatch, host_dispatch_forced};
use crate::glm5next_layer::profile;
use crate::glm5next_mlp::weights::Glm5NextMoeWeights;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-09-26: The arguments of one `forward_moe` call that its routed-expert paths read.
pub(super) struct MoeSite<'a> {
    pub(super) gpu: &'a dyn GpuBackend,
    pub(super) k: &'a Glm5NextMlpKernels,
    pub(super) cfg: &'a Glm5NextMlpConfig,
    pub(super) w: &'a Glm5NextMoeWeights,
    pub(super) x: DevicePtr,
    pub(super) rows: usize,
    pub(super) ws: &'a Glm5NextMlpWorkspace,
    pub(super) stream: u64,
}

/// 2026-09-26: The per-row routed experts of `forward_moe`: nothing when `batched` or
/// `grouped_prefill` is set, else each row's device- or host-dispatched sweep.
pub(super) fn per_row_experts(
    site: &MoeSite<'_>,
    batched: bool,
    grouped_prefill: bool,
) -> Result<()> {
    let MoeSite {
        gpu,
        k,
        cfg,
        w,
        x,
        rows,
        ws,
        stream,
    } = *site;
    for r in 0..rows {
        if batched || grouped_prefill {
            break;
        }
        let xr = x.offset(r * cfg.hidden * 2);
        let ids_r = ws.ids.offset(r * cfg.top_k * 4);
        let expert_out_r = ws.expert_out.offset(r * cfg.top_k * cfg.hidden * 2);

        let grouped = !host_dispatch_forced() && k.w4a16_gemv_sw_moe.0 != 0;
        announce_dispatch(grouped);
        if grouped {
            // 2026-09-25: Device-dispatched: the kernels read `ids` on the GPU, so nothing
            // here synchronises except the route trace below.
            let t = profile::start();
            let mi = cfg.moe_intermediate;
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                xr,
                &w.ptrs.gate,
                ws.a_gate,
                ids_r,
                mi,
                cfg.hidden,
                cfg.top_k,
                cfg.num_experts,
                0,
                stream,
            )?;
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                xr,
                &w.ptrs.up,
                ws.a_up,
                ids_r,
                mi,
                cfg.hidden,
                cfg.top_k,
                cfg.num_experts,
                0,
                stream,
            )?;
            // 2026-09-25: All slots at once. A remote slot's rows hold stale values; the down
            // projection skips that slot.
            swiglu(
                gpu,
                k.swiglu,
                ws.a_gate,
                ws.a_up,
                ws.a_act,
                cfg.top_k * mi,
                cfg.swiglu_limit,
                stream,
            )?;
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                ws.a_act,
                &w.ptrs.down,
                expert_out_r,
                ids_r,
                cfg.hidden,
                mi,
                cfg.top_k,
                cfg.num_experts,
                mi,
                stream,
            )?;
            profile::end(profile::MOE_EXPERTS, t, gpu, stream);

            if profile::trace_on() {
                let mut ids = vec![0u8; cfg.top_k * 4];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(ids_r, &mut ids)?;
                let decoded: Vec<i32> = (0..cfg.top_k)
                    .map(|k| {
                        i32::from_le_bytes([
                            ids[k * 4],
                            ids[k * 4 + 1],
                            ids[k * 4 + 2],
                            ids[k * 4 + 3],
                        ])
                    })
                    .collect();
                profile::stash_route(&decoded);
            }
        } else {
            // 2026-09-25: Host-dispatched: synchronise, read this row's `ids` back, and launch
            // the GEMVs of each local expert. The read is timed on its own (`MOE_HOSTSYNC`).
            let t = profile::start();
            let mut ids = vec![0u8; cfg.top_k * 4];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ids_r, &mut ids)?;
            profile::end(profile::MOE_HOSTSYNC, t, gpu, stream);

            if profile::trace_on() {
                let decoded: Vec<i32> = (0..cfg.top_k)
                    .map(|k| {
                        i32::from_le_bytes([
                            ids[k * 4],
                            ids[k * 4 + 1],
                            ids[k * 4 + 2],
                            ids[k * 4 + 3],
                        ])
                    })
                    .collect();
                profile::stash_route(&decoded);
            }

            let t = profile::start();
            for slot in 0..cfg.top_k {
                let id = i32::from_le_bytes([
                    ids[slot * 4],
                    ids[slot * 4 + 1],
                    ids[slot * 4 + 2],
                    ids[slot * 4 + 3],
                ]);
                // 2026-09-25: -1 is `glm5next_router_topk`'s unfilled-slot id, possible only when
                // `top_k > num_experts`, which `validate` refuses; it adds nothing.
                if id < 0 {
                    continue;
                }
                let id = id as usize;
                if id >= cfg.num_experts {
                    bail!(
                        "GLM MoE: router selected expert {id} of {}",
                        cfg.num_experts
                    );
                }
                let Some(local) = cfg.local_slot(id) else {
                    continue;
                };
                let e = &w.experts[local];
                let dst = expert_out_r.offset(slot * cfg.hidden * 2);
                let mi = cfg.moe_intermediate;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    xr,
                    &e.gate_proj,
                    ws.a_gate,
                    mi,
                    cfg.hidden,
                    stream,
                )?;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    xr,
                    &e.up_proj,
                    ws.a_up,
                    mi,
                    cfg.hidden,
                    stream,
                )?;
                swiglu(
                    gpu,
                    k.swiglu,
                    ws.a_gate,
                    ws.a_up,
                    ws.a_act,
                    mi,
                    cfg.swiglu_limit,
                    stream,
                )?;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    ws.a_act,
                    &e.down_proj,
                    dst,
                    cfg.hidden,
                    mi,
                    stream,
                )?;
            }

            profile::end(profile::MOE_EXPERTS, t, gpu, stream);
        }
    }
    Ok(())
}

/// 2026-09-26: The row-batched routed experts of `forward_moe`: one union build and one sweep
/// per `(start, width)` sub-group of `groups`.
pub(super) fn row_batched_experts(site: &MoeSite<'_>, groups: &[(usize, usize)]) -> Result<()> {
    let MoeSite {
        gpu,
        k,
        cfg,
        w,
        x,
        ws,
        stream,
        ..
    } = *site;
    let t = profile::start();
    let mi = cfg.moe_intermediate;
    // 2026-09-25: One union build and one sweep per `moe_row_groups` sub-group; the
    // sub-groups cover disjoint row ranges.
    for &(r0, w_rows) in groups {
        // 2026-09-25: Each sub-group rebuilds the union tables over its slice of `ids`,
        // overwriting the previous sub-group's; stream order puts that after the previous
        // sweeps.
        KernelLaunch::new(gpu, union_kernel(gpu, k.moe_row_union))
            .grid([1, 1, 1])
            .block([(w_rows * cfg.top_k) as u32, 1, 1])
            .arg_ptr(ws.ids.offset(r0 * cfg.top_k * 4))
            .arg_ptr(ws.u_eid)
            .arg_ptr(ws.u_slot)
            .arg_u32(w_rows as u32)
            .arg_u32(cfg.top_k as u32)
            .launch(stream)?;

        let kb = k.w4a16_gemv_sw_moe_batchm[w_rows - 2];
        // 2026-10-09: METRALE_GLM_MOE_GATEUP_FAST=1 runs gate and up as one launch
        // (bit-identical); off or unavailable, the two launches below run as before.
        let fused = w4a16_gemv_moe_batchm_gateup(
            gpu,
            x.offset(r0 * cfg.hidden * 2),
            &w.ptrs.gate,
            &w.ptrs.up,
            ws.a_gate.offset(r0 * cfg.top_k * mi * 2),
            ws.a_up.offset(r0 * cfg.top_k * mi * 2),
            ws.u_eid,
            ws.u_slot,
            mi,
            cfg.hidden,
            w_rows,
            cfg.top_k,
            cfg.num_experts,
            cfg.hidden,
            0,
            cfg.top_k * mi,
            stream,
        )?;
        if !fused {
            w4a16_gemv_moe_batchm(
                gpu,
                false,
                kb,
                x.offset(r0 * cfg.hidden * 2),
                &w.ptrs.gate,
                ws.a_gate.offset(r0 * cfg.top_k * mi * 2),
                ws.u_eid,
                ws.u_slot,
                mi,
                cfg.hidden,
                w_rows,
                cfg.top_k,
                cfg.num_experts,
                cfg.hidden,
                0,
                cfg.top_k * mi,
                stream,
            )?;
            w4a16_gemv_moe_batchm(
                gpu,
                false,
                kb,
                x.offset(r0 * cfg.hidden * 2),
                &w.ptrs.up,
                ws.a_up.offset(r0 * cfg.top_k * mi * 2),
                ws.u_eid,
                ws.u_slot,
                mi,
                cfg.hidden,
                w_rows,
                cfg.top_k,
                cfg.num_experts,
                cfg.hidden,
                0,
                cfg.top_k * mi,
                stream,
            )?;
        }
        // 2026-09-25: Every (row, slot) at once. A remote slot's rows hold stale values; the
        // down projection skips that slot.
        swiglu(
            gpu,
            k.swiglu,
            ws.a_gate.offset(r0 * cfg.top_k * mi * 2),
            ws.a_up.offset(r0 * cfg.top_k * mi * 2),
            ws.a_act.offset(r0 * cfg.top_k * mi * 2),
            w_rows * cfg.top_k * mi,
            cfg.swiglu_limit,
            stream,
        )?;
        w4a16_gemv_moe_batchm(
            gpu,
            true,
            kb,
            ws.a_act.offset(r0 * cfg.top_k * mi * 2),
            &w.ptrs.down,
            ws.expert_out.offset(r0 * cfg.top_k * cfg.hidden * 2),
            ws.u_eid,
            ws.u_slot,
            cfg.hidden,
            mi,
            w_rows,
            cfg.top_k,
            cfg.num_experts,
            cfg.top_k * mi,
            mi,
            cfg.top_k * cfg.hidden,
            stream,
        )?;
    }
    profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    Ok(())
}
