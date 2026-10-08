// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The fused verify snapshot (`METRALE_GLM_KDA_SNAP_FUSE=1`): a `decode_k` walk
//! that takes snapshots writes each one from its own kernels instead of copying it out.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Off (the default) nothing here runs and `decode_k` is unchanged.
//! - On, the walk launches the same two kernels per row as `stateful_row`'s shared-memory
//!   branch, in the same order, with the same grid, block, shared memory and per-row pointers;
//!   only the state pointers change. Row `t` reads the state the previous row wrote and writes
//!   to `snapshots[t]` if it has one and is not the last row, else to the live state
//!   ([`snap_fuse_plan`]). A row whose read and write pointers are equal launches the in-place
//!   kernel exactly as `stateful_row` does; otherwise the out-of-place twin
//!   (kernels/gb10/common/kda_snap_fuse.cu, same expressions on the same floats).
//! - So every snapshot slot ends holding the state after its row, the live state ends holding
//!   the state after the last row, and every output is the walk's, bit for bit; only the
//!   `2 * (k - 1)` device-to-device copies are gone. `kda_snap_fuse_microtest` checks it.
//! - Runs only where `stateful_row` would launch `kda_recurrent_decode_bf16_smem` and both twin
//!   kernels resolved ([`Glm5NextKdaLayer::snap_fuse_ready`]); otherwise the walk with its
//!   copies runs, with one warning per process while the lever is on.
//! - No host synchronisation, no allocation, fixed pointers: valid under CUDA graph capture,
//!   like the copies it replaces.
//!
//! Measured 2026-10-06 (nsys, GLM-5.3 TP2/EP2 decode, K=3 verify, 34 KDA layers): the copies
//! were 68 x 2 MiB + 68 x 192 KiB per step, about 0.55 ms of a 72 ms step.

use super::*;

/// 2026-10-06: Whether a `METRALE_GLM_KDA_SNAP_FUSE` value asks for the fused snapshot: `1`
/// only.
pub(super) fn snap_fuse_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-06: `METRALE_GLM_KDA_SNAP_FUSE=1`: a KDA `decode_k` (and each sequence of
/// `decode_verify_n_seqs`) that takes snapshots writes them from the row kernels instead of
/// copying them. Off unless set to `1`. Read once per process.
pub(super) fn kda_snap_fuse() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let on = snap_fuse_requested(std::env::var("METRALE_GLM_KDA_SNAP_FUSE").ok().as_deref());
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_SNAP_FUSE=1 - the KDA verify walk writes its state snapshots \
                 from the row kernels instead of copying them (byte-identical by construction; \
                 see kernels/gb10/common/kda_snap_fuse.cu)"
            );
        }
        on
    })
}

/// 2026-10-06: The one warning when `METRALE_GLM_KDA_SNAP_FUSE=1` cannot be honoured.
fn kda_snap_fuse_fallback() {
    static W: std::sync::Once = std::sync::Once::new();
    W.call_once(|| {
        tracing::warn!(
            "METRALE_GLM_KDA_SNAP_FUSE=1 ignored (the target lacks a kda_snap_fuse kernel or the \
             walk would not take the shared-memory recurrent kernel); the KDA verify keeps its \
             snapshot copies"
        )
    });
}

/// 2026-10-06: Where one row of the fused walk reads and writes the two states, and the copy
/// it still makes afterwards (only a last row that has a snapshot: its state must land in the
/// live buffer and in the slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SnapRowIo {
    pub h_in: DevicePtr,
    pub h_out: DevicePtr,
    pub c_in: DevicePtr,
    pub c_out: DevicePtr,
    /// 2026-10-06: `(h_dst, conv_dst)` to copy the live state into after the row.
    pub copy_to: Option<(DevicePtr, DevicePtr)>,
}

/// 2026-10-06: The fused walk's pointers for `k` rows. Row `t` reads what row `t - 1` wrote
/// (row 0 the live state) and writes `snapshots[t]` when it has one and `t < k - 1`, else the
/// live state. Equal snapshot pointers stay correct: a later row overwrites the slot, as the
/// later copy did.
pub(super) fn snap_fuse_plan(
    k: usize,
    state: &KdaSeqState,
    snapshots: &[(DevicePtr, DevicePtr)],
) -> Vec<SnapRowIo> {
    let mut plan = Vec::with_capacity(k);
    let (mut h_in, mut c_in) = (state.recurrent, state.conv);
    for t in 0..k {
        let last = t + 1 == k;
        let (h_out, c_out) = match snapshots.get(t) {
            Some(&(h, c)) if !last => (h, c),
            _ => (state.recurrent, state.conv),
        };
        let copy_to = if last { snapshots.get(t).copied() } else { None };
        plan.push(SnapRowIo {
            h_in,
            h_out,
            c_in,
            c_out,
            copy_to,
        });
        h_in = h_out;
        c_in = c_out;
    }
    plan
}

impl Glm5NextKdaLayer {
    /// 2026-10-06: Whether the fused walk can run: both twin kernels resolved and
    /// `stateful_row` would take its shared-memory recurrent branch.
    pub fn snap_fuse_ready(&self) -> bool {
        let d = self.cfg.head_dim;
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem = (3 * d + vpb * (d + 1)) * 4;
        self.kernels.has_snap_fuse()
            && self.kernels.recurrent_smem.0 != 0
            && d.is_multiple_of(vpb)
            && smem <= KDA_SMEM_BUDGET
            && !kda_no_smem()
    }

    /// 2026-10-06: [`Self::decode_k`] with the fused snapshot chosen by `fuse` instead of by
    /// `METRALE_GLM_KDA_SNAP_FUSE`, for `kda_snap_fuse_microtest` to run both arms in one
    /// process. `fuse = false` is today's `decode_k` with the lever off.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_k_snap_arm(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        fuse: bool,
        stream: u64,
    ) -> Result<()> {
        self.decode_k_impl(gpu, hidden, k, state, ws, snapshots, None, fuse, stream)
    }

    /// 2026-10-06: The fused walk over workspace rows `row0..row0 + k` on `state`, writing the
    /// state after row `t` to `snapshots[t]`. Returns `Ok(false)`, having launched nothing,
    /// when [`Self::snap_fuse_ready`] is false; the caller then walks with copies.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn snap_walk(
        &self,
        gpu: &dyn GpuBackend,
        row0: usize,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        stream: u64,
    ) -> Result<bool> {
        if !self.snap_fuse_ready() {
            kda_snap_fuse_fallback();
            return Ok(false);
        }
        let c = &self.cfg;
        let (h_bytes, conv_bytes) = (c.recurrent_state_elems() * 4, c.conv_state_elems() * 4);
        for (t, io) in snap_fuse_plan(k, state, snapshots).iter().enumerate() {
            self.conv_row_io(gpu, row0 + t, io.c_in, io.c_out, ws, stream)?;
            self.recurrent_row_io(gpu, row0 + t, io.h_in, io.h_out, ws, stream)?;
            if let Some((h_dst, conv_dst)) = io.copy_to {
                gpu.copy_d2d_async(state.recurrent, h_dst, h_bytes, stream)?;
                gpu.copy_d2d_async(state.conv, conv_dst, conv_bytes, stream)?;
            }
        }
        Ok(true)
    }

    /// 2026-10-06: `stateful_row`'s conv launch on workspace row `row`, reading the conv state
    /// from `c_in` and writing it to `c_out`: the in-place `causal_conv1d_update_l2norm` when
    /// they are equal, its out-of-place twin otherwise, with the same grid and arguments.
    fn conv_row_io(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        c_in: DevicePtr,
        c_out: DevicePtr,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let cd = c.conv_dim();
        if c_in == c_out {
            return ops::conv1d_update_l2norm(
                gpu,
                self.kernels.conv_decode,
                c_in,
                ws.qkv_proj.offset(row * cd * 2),
                &self.weights.conv,
                ws.conv_out.offset(row * cd * 2),
                cd as u32,
                c.conv_kernel as u32,
                1,
                c.qk_channels() as u32,
                c.head_dim as u32,
                c.l2_eps,
                stream,
            );
        }
        // 2026-10-06: `ops::conv1d_update_l2norm`'s launch at batch 1, null bias, with the
        // state pointer split in two.
        KernelLaunch::new(gpu, self.kernels.conv_io)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(c_in)
            .arg_ptr(c_out)
            .arg_ptr(ws.qkv_proj.offset(row * cd * 2))
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.conv_out.offset(row * cd * 2))
            .arg_u32(1)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(c.qk_channels() as u32)
            .arg_u32(c.head_dim as u32)
            .arg_f32(c.l2_eps)
            .launch(stream)
    }

    /// 2026-10-06: `stateful_row`'s shared-memory recurrent launch on workspace row `row`,
    /// reading the state from `h_in` and writing it to `h_out`: `kda_recurrent_decode_bf16_smem`
    /// when they are equal, its out-of-place twin otherwise, same grid, block, shared memory and
    /// per-row pointers.
    fn recurrent_row_io(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        h_in: DevicePtr,
        h_out: DevicePtr,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let qkv = c.qkv_dim();
        let cd = c.conv_dim();
        let d = c.head_dim;
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem = (3 * d + vpb * (d + 1)) * 4;
        let kernel = if h_in == h_out {
            self.kernels.recurrent_smem
        } else {
            self.kernels.recurrent_io
        };
        let mut l = KernelLaunch::new(gpu, kernel)
            .grid([c.heads as u32, (d / vpb) as u32, 1])
            .block([vpb as u32, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(ws.conv_out.offset(row * cd * 2))
            .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 2))
            .arg_ptr(ws.conv_out.offset(row * cd * 2 + qkv * 4))
            .arg_ptr(ws.gate.offset(row * qkv * 4))
            .arg_ptr(ws.beta.offset(row * c.heads * 4))
            .arg_ptr(h_in);
        if h_in != h_out {
            l = l.arg_ptr(h_out);
        }
        l.arg_ptr(ws.core.offset(row * qkv * 4))
            .arg_u32(c.heads as u32)
            .arg_u32(d as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .arg_u32(vpb as u32)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lever_is_on_only_for_one() {
        assert!(snap_fuse_requested(Some("1")));
        for v in [None, Some(""), Some("0"), Some("true"), Some("on"), Some(" 1"), Some("2")] {
            assert!(!snap_fuse_requested(v), "{v:?}");
        }
    }

    const S: KdaSeqState = KdaSeqState {
        conv: DevicePtr(0x1000),
        recurrent: DevicePtr(0x2000),
    };

    fn snaps(n: usize) -> Vec<(DevicePtr, DevicePtr)> {
        (0..n)
            .map(|t| {
                (
                    DevicePtr(0x10_000 + t as u64 * 0x100),
                    DevicePtr(0x20_000 + t as u64 * 0x100),
                )
            })
            .collect()
    }

    /// 2026-10-06: Replays a plan on version counters: a row reads its input buffer's version
    /// and writes version + 1 to its output (a copy carries the version). The walk with copies
    /// leaves the live state at version `k` and slot `t` at version `t + 1`, every row reading
    /// version `t`; the plan must reproduce all three.
    fn replay(k: usize, sn: &[(DevicePtr, DevicePtr)]) {
        use std::collections::HashMap;
        let plan = snap_fuse_plan(k, &S, sn);
        assert_eq!(plan.len(), k);
        let mut ver: HashMap<DevicePtr, usize> = HashMap::new();
        ver.insert(S.recurrent, 0);
        ver.insert(S.conv, 0);
        for (t, io) in plan.iter().enumerate() {
            let (h, c) = (ver[&io.h_in], ver[&io.c_in]);
            assert_eq!((h, c), (t, t), "k={k} snaps={} row {t} reads the wrong state", sn.len());
            ver.insert(io.h_out, h + 1);
            ver.insert(io.c_out, c + 1);
            if let Some((hd, cd)) = io.copy_to {
                assert_eq!((io.h_out, io.c_out), (S.recurrent, S.conv));
                let (hv, cv) = (ver[&S.recurrent], ver[&S.conv]);
                ver.insert(hd, hv);
                ver.insert(cd, cv);
            }
        }
        assert_eq!((ver[&S.recurrent], ver[&S.conv]), (k, k));
        for (t, (hd, cd)) in sn.iter().enumerate().take(k) {
            assert_eq!((ver[hd], ver[cd]), (t + 1, t + 1), "k={k} slot {t}");
        }
    }

    #[test]
    fn plan_matches_the_walk_with_copies() {
        for k in 1..=5 {
            for n in 0..=k + 1 {
                replay(k, &snaps(n));
            }
        }
    }

    /// 2026-10-06: The K=3 verify the engine runs: two interior snapshots, no copy, the last
    /// row back to the live state.
    #[test]
    fn k3_verify_plan_ping_pongs_through_the_slots() {
        let sn = snaps(2);
        let p = snap_fuse_plan(3, &S, &sn);
        assert_eq!((p[0].h_in, p[0].h_out), (S.recurrent, sn[0].0));
        assert_eq!((p[1].h_in, p[1].h_out), (sn[0].0, sn[1].0));
        assert_eq!((p[2].h_in, p[2].h_out), (sn[1].0, S.recurrent));
        assert_eq!((p[2].c_in, p[2].c_out), (sn[1].1, S.conv));
        assert!(p.iter().all(|io| io.copy_to.is_none()));
    }

    /// 2026-10-06: Aliased slots (all equal, or equal to the live state) still replay right.
    #[test]
    fn aliased_slots_replay_right() {
        let same = vec![(DevicePtr(0x9000), DevicePtr(0xA000)); 3];
        for k in 1..=4 {
            let p = snap_fuse_plan(k, &S, &same);
            let last = p.last().unwrap();
            assert_eq!((last.h_out, last.c_out), (S.recurrent, S.conv));
        }
        let live = vec![(S.recurrent, S.conv); 2];
        let p = snap_fuse_plan(3, &S, &live);
        assert!(p.iter().all(|io| io.h_in == io.h_out && io.c_in == io.c_out));
    }
}
