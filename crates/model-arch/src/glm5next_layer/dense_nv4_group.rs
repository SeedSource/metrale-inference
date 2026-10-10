// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `METRALE_GLM_NV4_TC_GROUP`: GLM NVFP4 projections that read the same input in
//! one grouped, persistent tensor-core launch.
//!
//! Owner: model-arch (GLM-5.3).
//! `METRALE_GLM_NV4_TC_GROUP=1` (read once; off by default) lets a call site hand
//! [`try_group`] two or three projections of the same activation `a` `[m, k]`. When every
//! member would run `w4a16_gemv_tc8` / `w4a16_gemv_tc16` on its own (`dense_fp8::route`'s
//! NVFP4 arm, then `dense_nv4_tc::try_launch`: `METRALE_GLM_NV4_TC` takes the launch, e.g.
//! inside a `MultiSeqScope` under `multi`, and the shape routes to the same entry), they run as
//! one `w4a16_gemv_tc{8,16}_group` launch over all their 8- (16-) column tiles, with a grid of
//! `min(tiles, SMs x resident CTAs per SM)` CTAs that loop tiles. Each tile runs the per-CTA
//! entry's body on the same operands, so every output byte equals the per-projection launch
//! (gate `examples/glm5next_nv4_tc_microtest.rs`, GROUP checks). Otherwise `Ok(false)` and the
//! caller issues its per-projection calls unchanged.
//!
//! Groups (each reads `a`, nothing between the calls writes `a`, and no member reads another
//! member's output): KDA `q/k/v_proj` (`glm5next_kda/mod.rs` `front_end_opts`), dense-MLP and
//! shared-expert `gate/up_proj` (`glm5next_mlp/forward/dense.rs`), DSA `q_a_proj/kv_a_proj`
//! (`glm5next_dsa/layer/decode_k.rs`, `xseq/group.rs`).

use std::sync::{Once, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::gemv_tc::{self, TcGroupMember};
use metrale_model_layers::weight_map::QuantizedWeight;

use super::{dense_fp8, dense_nv4_tc};

/// `METRALE_GLM_NV4_TC_GROUP=1`, read once.
pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("METRALE_GLM_NV4_TC_GROUP").is_ok_and(|v| v.trim() == "1"))
}

/// One projection of a group: the BF16 weight pointer the caller would pass to its `gemm`
/// wrapper (`dense_fp8::route` maps it to the NVFP4 copy), its output and its `n`.
#[derive(Clone, Copy)]
pub struct Proj {
    pub b: DevicePtr,
    pub c: DevicePtr,
    pub n: usize,
}

/// Runs `c_i[m, n_i] = a[m, k] @ W_i^T` for every member as one grouped launch and returns
/// `Ok(true)`, or launches nothing and returns `Ok(false)` (lever off, a member that would not
/// take the tensor-core NVFP4 path on its own, or the grouped entry missing). `gemv` is the
/// GEMV handle the caller's `gemm` wrapper passes to `dense_fp8::route`. No allocation or
/// sync, so it is capture-safe.
pub fn try_group(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    a: DevicePtr,
    members: &[Proj],
    m: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if !enabled()
        || members.len() < 2
        || members.len() > gemv_tc::TC_GROUP_MAX
        || !dense_nv4_tc::takes(m)
    {
        return Ok(false);
    }
    let mut qs: [Option<QuantizedWeight>; gemv_tc::TC_GROUP_MAX] = [None; gemv_tc::TC_GROUP_MAX];
    for (slot, p) in qs.iter_mut().zip(members) {
        match dense_fp8::nv4_copy_routed(gpu, gemv, p.b, m, p.n, k) {
            Some(q) => *slot = Some(q),
            None => return Ok(false),
        }
    }
    let group: Vec<TcGroupMember> = members
        .iter()
        .zip(qs.iter().flatten())
        .map(|(p, q)| TcGroupMember {
            weight: q,
            out: p.c,
            n: p.n as u32,
        })
        .collect();
    if !gemv_tc::tc_group_launch(gpu, a, &group, m as u32, k as u32, stream)? {
        return Ok(false);
    }
    dense_fp8::nv4_group_hits(members.len());
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_NV4_TC_GROUP=1: ENGAGED - {} NVFP4 projections of one input ({m} rows, \
             K {k}) in one persistent w4a16_gemv_tc*_group launch; bytes equal to the \
             per-projection tc8/tc16 launches",
            members.len()
        );
    });
    Ok(true)
}
