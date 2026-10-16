// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_NV4_B3_GROUP`: GLM NVFP4 projections of the same three rows in one
//! `w4a16_gemv_batch3_staged_group` launch instead of one `w4a16_gemv_batch3_staged` launch each.
//!
//! Owner: model-arch (GLM-5.3).
//! `METRALE_GLM_NV4_B3_GROUP=1` (read once; off by default) is the CUDA-core twin of
//! `METRALE_GLM_NV4_TC_GROUP` (`dense_nv4_group`): the same call sites hand
//! `dense_nv4_group::try_group` the same groups, and a call the tensor-core group does not take
//! comes here. Groups (each reads one activation `a`, nothing between the calls writes `a`, no
//! member reads another member's output; the call sites' contract):
//! - `kda_qkv`: KDA `q/k/v_proj` (`glm5next_kda/mod.rs` `front_end_opts`), 3 x `[qkv, hidden]`.
//! - `shared_gateup`: `gate/up_proj` of `forward_dense_sliced` (`glm5next_mlp/forward/dense.rs`):
//!   the shared expert of every MoE layer and the dense-MLP layers, 2 x `[inter, hidden]`.
//! - `dsa_qa_kva`: DSA `q_a_proj` + `kv_a_proj` (`glm5next_dsa/layer/decode_k.rs`,
//!   `xseq/group.rs`, via `proj_gemm.rs` `try_group_q_a_kv_a`).
//!
//! Engaged only where today's per-projection launch is `w4a16_gemv_batch3_staged` ([`decline`]):
//! exactly 3 rows (the C1 MTP K3 verify), `METRALE_GLM_NV4_B3_STAGED=1` with that kernel loaded,
//! `METRALE_GLM_NV4_TC` not taking the launch (`dense_nv4_tc::takes`), every member a weight
//! with an NVFP4 copy that `dense_fp8::route` would run on the NVFP4 GEMV
//! (`dense_fp8::nv4_copy_routed`), 2..=3 members, and the image has the grouped entry. Anything
//! else returns `Ok(false)` without launching and the caller runs its per-projection calls.
//!
//! Exactness: the grouped kernel runs, for each block, the per-projection kernel's body on that
//! member's own weight, scales, FP32 tensor scale and output (`kernels/gb10/common/w4a16_gemv.cu`
//! `w4a16_b3s_block`), at the block index the member's own launch would give it, so every output
//! byte equals the separate launch (same operands, same FP operations in the same order).
//! Weights are NOT concatenated along N: each NVFP4 copy has its own `weight_scale_2`
//! (`quantize_to_nvfp4`: absmax / (6 x 448) per tensor), so one launch over appended weights
//! with one tensor scale would change the bits. The kernel selects the member per block
//! instead; nothing is copied or allocated at load (0 extra bytes). Outputs land in the
//! caller's buffers at row stride `n_j`, as the separate launches write them.
//!
//! No allocation or sync per call; the kernel handle resolves at load ([`finish_load`], from
//! `dense_fp8::finish_load`) and only with the lever on, so the launch is graph-capture safe
//! with fixed pointers (weights never move; outputs are the callers' workspace buffers).

use std::sync::{Once, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::weight_map::QuantizedWeight;

use super::dense_nv4_group::Proj;
use super::{dense_fp8, dense_nv4_tc};

/// Rows the grouped entry runs (the `w4a16_gemv_batch3_staged` tier).
pub const ROWS: usize = 3;
/// Most members one launch takes.
pub const MAX_MEMBERS: usize = 3;
/// Outputs per 256-thread block (`N_PER_BLOCK` in `w4a16_gemv.cu`).
pub const OUTS_PER_BLOCK: usize = 4;
/// Module and entry point of the grouped kernel.
pub const MODULE: &str = "w4a16_gemv";
pub const KERNEL: &str = "w4a16_gemv_batch3_staged_group";

/// `METRALE_GLM_NV4_B3_GROUP` parsed: `1` (trimmed) is on, anything else (or unset) off.
fn parse(v: Option<&str>) -> bool {
    v.map(str::trim) == Some("1")
}

/// `METRALE_GLM_NV4_B3_GROUP=1`, read once.
pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| parse(std::env::var("METRALE_GLM_NV4_B3_GROUP").ok().as_deref()))
}

/// Why a call does not take the grouped launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// The lever is off.
    Off,
    /// Not exactly [`ROWS`] rows: today's launch is not `w4a16_gemv_batch3_staged`.
    Rows,
    /// Fewer than 2 or more than [`MAX_MEMBERS`] members.
    Members,
    /// `METRALE_GLM_NV4_TC` takes this launch (tensor-core tier; `dense_nv4_group` handles it).
    TcTakes,
    /// `METRALE_GLM_NV4_B3_STAGED=1` is off or its kernel is missing.
    NotStaged,
    /// The image lacks `w4a16_gemv_batch3_staged_group`.
    NoKernel,
    /// A member has no NVFP4 copy the dense route would run.
    NoCopy,
}

/// The per-call facts [`decline`] judges (cheapest first).
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    pub lever: bool,
    pub m: usize,
    pub members: usize,
    pub tc_takes: bool,
    pub staged_live: bool,
    pub kernel: bool,
}

/// The first reason a call with these facts keeps its per-projection launches, `None` when it
/// takes the grouped launch (member copies are checked after this, as [`Decline::NoCopy`]).
pub fn decline(g: Gate) -> Option<Decline> {
    if !g.lever {
        Some(Decline::Off)
    } else if g.m != ROWS {
        Some(Decline::Rows)
    } else if !(2..=MAX_MEMBERS).contains(&g.members) {
        Some(Decline::Members)
    } else if g.tc_takes {
        Some(Decline::TcTakes)
    } else if !g.staged_live {
        Some(Decline::NotStaged)
    } else if !g.kernel {
        Some(Decline::NoKernel)
    } else {
        None
    }
}

/// Blocks member `n` outputs take (ceil(n / 4)).
pub fn member_blocks(n: usize) -> usize {
    n.div_ceil(OUTS_PER_BLOCK)
}

/// The grouped launch's grid: the sum of the members' block counts.
pub fn grid_blocks(ns: &[usize]) -> usize {
    ns.iter().map(|&n| member_blocks(n)).sum()
}

/// Host mirror of the kernel's block-to-member map: block `blk` of the grouped launch runs
/// block `.1` of member `.0` (that member's own launch at `blockIdx.x = .1`); `None` past the
/// grid. Unit tests check it against [`grid_blocks`].
pub fn member_of_block(ns: &[usize], mut blk: usize) -> Option<(usize, usize)> {
    for (j, &n) in ns.iter().enumerate() {
        let b = member_blocks(n);
        if blk < b {
            return Some((j, blk));
        }
        blk -= b;
    }
    None
}

/// The grouped kernel, resolved once and only with the lever on (`None` when absent).
fn kernel(gpu: &dyn GpuBackend) -> Option<KernelHandle> {
    static H: OnceLock<Option<KernelHandle>> = OnceLock::new();
    *H.get_or_init(|| {
        if !enabled() {
            return None;
        }
        gpu.kernel(MODULE, KERNEL).ok()
    })
}

/// The wired groups whose weights `METRALE_GLM_DENSE_NVFP4` gives NVFP4 copies, for the
/// ENGAGED line.
fn wired_groups() -> String {
    let c = dense_fp8::dense_nvfp4();
    let mut v = Vec::new();
    if c.kda {
        v.push("kda_qkv");
    }
    if c.shared || c.mlp || c.mtp || dense_fp8::dense_mlp_ckpt() {
        v.push("shared_gateup");
    }
    if c.dsa {
        v.push("dsa_qa_kva");
    }
    v.join(", ")
}

fn warn_not_engaged(why: Decline) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let what = match why {
            Decline::NoKernel => "the image lacks w4a16_gemv_batch3_staged_group",
            _ => "METRALE_GLM_NV4_B3_STAGED=1 is off or w4a16_gemv_batch3_staged is missing",
        };
        tracing::warn!(
            "METRALE_GLM_NV4_B3_GROUP=1: NOT engaged ({what}); projections run per launch"
        );
    });
}

/// 2026-10-12: At load (`dense_fp8::finish_load`): resolve the kernel while nothing captures,
/// and warn once when the lever is on but cannot engage.
pub fn finish_load(gpu: &dyn GpuBackend) {
    if !enabled() {
        return;
    }
    if !dense_fp8::nv4_b3_staged_live(gpu) {
        warn_not_engaged(Decline::NotStaged);
    } else if kernel(gpu).is_none() {
        warn_not_engaged(Decline::NoKernel);
    }
}

/// Runs `c_i[3, n_i] = a[3, k] @ W_i^T` for every member as one `w4a16_gemv_batch3_staged_group`
/// launch and returns `Ok(true)`, or launches nothing and returns `Ok(false)` (module doc).
/// `gemv` is the GEMV handle the caller's `gemm` wrapper passes to `dense_fp8::route`.
pub fn try_group(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    a: DevicePtr,
    members: &[Proj],
    m: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    let lever = enabled();
    if !lever || m != ROWS {
        return Ok(false);
    }
    let gate = Gate {
        lever,
        m,
        members: members.len(),
        tc_takes: dense_nv4_tc::takes(m),
        staged_live: dense_fp8::nv4_b3_staged_live(gpu),
        kernel: kernel(gpu).is_some(),
    };
    if let Some(why) = decline(gate) {
        if matches!(why, Decline::NotStaged | Decline::NoKernel) {
            warn_not_engaged(why);
        }
        return Ok(false);
    }
    let Some(h) = kernel(gpu) else {
        return Ok(false);
    };
    let mut qs: [Option<QuantizedWeight>; MAX_MEMBERS] = [None; MAX_MEMBERS];
    for (slot, p) in qs.iter_mut().zip(members) {
        match dense_fp8::nv4_copy_routed(gpu, gemv, p.b, m, p.n, k) {
            Some(q) => *slot = Some(q),
            None => return Ok(false),
        }
    }
    let ns: Vec<usize> = members.iter().map(|p| p.n).collect();
    let mut l = KernelLaunch::new(gpu, h)
        .grid([grid_blocks(&ns) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a);
    for j in 0..MAX_MEMBERS {
        l = match (qs[j], members.get(j)) {
            (Some(q), Some(p)) => l
                .arg_ptr(q.weight)
                .arg_ptr(q.weight_scale)
                .arg_f32(q.weight_scale_2)
                .arg_ptr(p.c)
                .arg_u32(p.n as u32),
            _ => l
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(DevicePtr::NULL)
                .arg_f32(1.0)
                .arg_ptr(DevicePtr::NULL)
                .arg_u32(0),
        };
    }
    l.arg_u32(k as u32).launch(stream)?;
    dense_fp8::nv4_group_hits(members.len());
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_NV4_B3_GROUP=1: ENGAGED ({}) - same-input 3-row NVFP4 projections run as \
             one w4a16_gemv_batch3_staged_group launch; bytes equal to the per-projection \
             w4a16_gemv_batch3_staged launches",
            wired_groups()
        );
    });
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lever_parses_only_one_as_on() {
        assert!(parse(Some("1")));
        assert!(parse(Some(" 1 ")));
        for v in [None, Some(""), Some("0"), Some("true"), Some("11")] {
            assert!(!parse(v), "{v:?}");
        }
    }

    #[test]
    fn grid_and_block_map_cover_each_member_exactly() {
        // The three groups' per-rank shapes (TP2), a partial last block, and an empty slot.
        for ns in [
            vec![4096usize, 4096, 4096],
            vec![1024, 1024],
            vec![1536, 512],
            vec![5, 3, 9],
            vec![8, 0, 4],
        ] {
            let grid = grid_blocks(&ns);
            assert_eq!(
                grid,
                ns.iter().map(|n| n.div_ceil(4)).sum::<usize>(),
                "{ns:?}"
            );
            let mut seen = vec![0usize; ns.len()];
            let mut covered: Vec<Vec<bool>> = ns.iter().map(|&n| vec![false; n]).collect();
            for b in 0..grid {
                let (j, local) = member_of_block(&ns, b).expect("block inside the grid");
                assert_eq!(local, seen[j], "{ns:?}: member {j} blocks in order");
                seen[j] += 1;
                for o in 0..OUTS_PER_BLOCK {
                    let col = local * OUTS_PER_BLOCK + o;
                    if col < ns[j] {
                        assert!(!covered[j][col], "{ns:?}: column {col} twice");
                        covered[j][col] = true;
                    }
                }
            }
            assert_eq!(member_of_block(&ns, grid), None, "{ns:?}: past the grid");
            for (j, c) in covered.iter().enumerate() {
                assert!(c.iter().all(|&x| x), "{ns:?}: member {j} not fully covered");
                assert_eq!(seen[j], member_blocks(ns[j]));
            }
        }
    }

    #[test]
    fn grouped_offsets_match_the_kda_qkv_parts_layout() {
        // KDA writes q/k/v into `qkv_parts` at i * t * qkv * 2 bytes; each member's C is its own
        // part at row stride qkv, which is what the separate launch writes (C[t * N + n]).
        let (t, qkv) = (ROWS, 4096usize);
        let part = |i: usize| i * t * qkv * 2;
        assert_eq!(part(1) - part(0), t * qkv * 2);
        assert_eq!(part(2) - part(1), t * qkv * 2);
        // Grouped block of member 1, local block 0 starts at grid block 1024.
        assert_eq!(member_of_block(&[qkv, qkv, qkv], 1024), Some((1, 0)));
        assert_eq!(member_of_block(&[qkv, qkv, qkv], 2047), Some((1, 1023)));
        assert_eq!(member_of_block(&[1536, 512], 384), Some((1, 0)));
    }

    #[test]
    fn falls_back_unless_every_condition_holds() {
        let ok = Gate {
            lever: true,
            m: 3,
            members: 3,
            tc_takes: false,
            staged_live: true,
            kernel: true,
        };
        assert_eq!(decline(ok), None);
        assert_eq!(decline(Gate { members: 2, ..ok }), None);
        assert_eq!(decline(Gate { lever: false, ..ok }), Some(Decline::Off));
        for m in [0usize, 1, 2, 4, 8, 16, 17] {
            assert_eq!(decline(Gate { m, ..ok }), Some(Decline::Rows), "m={m}");
        }
        for n in [0usize, 1, 4] {
            assert_eq!(
                decline(Gate { members: n, ..ok }),
                Some(Decline::Members),
                "{n}"
            );
        }
        assert_eq!(
            decline(Gate {
                tc_takes: true,
                ..ok
            }),
            Some(Decline::TcTakes)
        );
        assert_eq!(
            decline(Gate {
                staged_live: false,
                ..ok
            }),
            Some(Decline::NotStaged)
        );
        assert_eq!(
            decline(Gate {
                kernel: false,
                ..ok
            }),
            Some(Decline::NoKernel)
        );
    }
}
