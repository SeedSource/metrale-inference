// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The deferred commit (`METRALE_GLM_KDA_DEFER_COMMIT=1`): a K-row verify whose row 0
//! reads its input state from the verify intermediate a partial-accept commit left in place, so
//! the commit's copy of that intermediate into the live state (68 MiB + 6.4 MiB per step on
//! GLM-5.3 TP2, serial on both ranks) is not made.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Off (the default) nothing here runs: [`Glm5NextKdaLayer::defer_commit_ready`] is false, so
//!   the model never sets an `SsmDeferSrc` and every commit copies as before.
//! - [`Glm5NextKdaLayer::decode_k_defer`] is the fused-snapshot walk (`snap_walk`, snap_fuse.rs)
//!   with one change: row 0's conv and recurrent launches are the `_defer` twins
//!   (kernels/gb10/common/kda_snap_fuse.cu), which read their input from intermediate
//!   `*defer.idx` of this slot when that device word is `>= 0`, else from the live state, and
//!   write snapshot 0 as the fused walk's row 0 does. Rows `>= 1`, the last row's copy, the
//!   grid, block, shared memory and every other pointer are the fused walk's.
//! - So with `*idx = -1` every byte is the fused walk's, and with `*idx = i` every byte is the
//!   fused walk's after the copy `intermediate i -> live` the commit used to make (the kernel
//!   comment gives the aliasing argument for `i == 0`, where input and output are one buffer).
//! - Refuses, launching nothing, `k < 2`, no snapshots, a snapshot 0 equal to the live state, or a
//!   layer without the kernels; the model sets a source only when every KDA layer is ready.
//! - The index is read on the device and every pointer is fixed per slot and layer: valid under
//!   CUDA graph capture, like the walk it extends.

use super::*;
use metrale_model_layers::layer::SsmDeferSrc;

/// 2026-10-09: Whether a `METRALE_GLM_KDA_DEFER_COMMIT` value asks for the deferred commit: `1`
/// only.
pub(super) fn defer_commit_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-09: `METRALE_GLM_KDA_DEFER_COMMIT=1`, read once per process.
fn kda_defer_commit() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let on = defer_commit_requested(
            std::env::var("METRALE_GLM_KDA_DEFER_COMMIT")
                .ok()
                .as_deref(),
        );
        if on {
            tracing::warn!(
                "METRALE_GLM_KDA_DEFER_COMMIT=1 - a partial-accept commit leaves the accepted KDA \
                 state in its verify intermediate and the next fused-snapshot verify reads it \
                 there (byte-identical by construction; see kernels/gb10/common/kda_snap_fuse.cu)"
            );
        }
        on
    })
}

impl Glm5NextKdaLayer {
    /// 2026-10-09: Whether [`Self::decode_k_defer`] can run on this layer: the fused walk is ready
    /// and both `_defer` kernels resolved. Independent of the levers (the microtest's arm).
    pub fn defer_walk_ready(&self) -> bool {
        self.snap_fuse_ready() && self.kernels.has_defer_commit()
    }

    /// 2026-10-09: Whether the model may defer this layer's commit: `METRALE_GLM_KDA_DEFER_COMMIT=1`
    /// and `METRALE_GLM_KDA_SNAP_FUSE=1` with [`Self::defer_walk_ready`]. Warns once when the
    /// lever is on and cannot be honoured.
    pub fn defer_commit_ready(&self) -> bool {
        if !kda_defer_commit() {
            return false;
        }
        // 2026-10-12: Under `METRALE_GLM_KDA_STATE_F16=1` also the FP16 twins (state_f16.rs).
        let ok = kda_snap_fuse()
            && self.defer_walk_ready()
            && (!super::super::state_f16::kda_state_f16() || self.state_f16_ready());
        if !ok {
            static W: std::sync::Once = std::sync::Once::new();
            W.call_once(|| {
                tracing::warn!(
                    "METRALE_GLM_KDA_DEFER_COMMIT=1 ignored (needs METRALE_GLM_KDA_SNAP_FUSE=1, the \
                     shared-memory recurrent kernel and both kda_snap_fuse _defer kernels; \
                     has_defer_commit={}); every commit keeps its state copy",
                    self.kernels.has_defer_commit()
                )
            });
        }
        ok
    }

    /// 2026-10-09: [`Self::decode_k`] with fused snapshots whose row 0 reads its input state where
    /// `defer` says (module header). `snapshots[t]` receives the state after row `t`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_k_defer(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        defer: &SsmDeferSrc,
        stream: u64,
    ) -> Result<()> {
        if k < 2 || k > ws.max_tokens || snapshots.is_empty() || !self.defer_walk_ready() {
            bail!(
                "KDA decode_k_defer: k={k} (workspace {}), {} snapshots, ready={}: the model set a \
                 deferred source this walk cannot honour",
                ws.max_tokens,
                snapshots.len(),
                self.defer_walk_ready()
            );
        }
        if snapshots[0].0 == state.recurrent || snapshots[0].1 == state.conv {
            bail!("KDA decode_k_defer: snapshot 0 is the live state");
        }
        // 2026-10-12: The h intermediates hold the state's format: whole halves when FP16.
        let h_elem = if state.recurrent_f16 { 2 } else { 4 };
        if !defer.h_stride.is_multiple_of(h_elem) || !defer.conv_stride.is_multiple_of(4) {
            bail!("KDA decode_k_defer: intermediate strides {defer:?} are not whole elements");
        }
        if state.recurrent_f16 && !self.state_f16_ready() {
            bail!("KDA decode_k_defer: FP16 recurrent state without the kda_state_f16 twins");
        }
        use crate::glm5next_layer::profile;
        let c = &self.cfg;
        let (h_bytes, conv_bytes) = (state.recurrent_bytes(c), c.conv_state_elems() * 4);
        let t_front = profile::start();
        self.front_end(gpu, hidden, k, ws, stream)?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = profile::start();
        for (t, io) in snap_fuse_plan(k, state, snapshots).iter().enumerate() {
            if t == 0 {
                self.conv_row_defer(gpu, state.conv, io.c_out, defer, ws, stream)?;
                self.recurrent_row_defer(gpu, state, io.h_out, defer, ws, stream)?;
            } else {
                self.conv_row_io(gpu, t, io.c_in, io.c_out, ws, stream)?;
                // 2026-10-12: An FP16 state takes the FP16 twins.
                if state.recurrent_f16 {
                    self.recurrent_row_f16(gpu, t, io.h_in, io.h_out, ws, stream)?;
                } else {
                    self.recurrent_row_io(gpu, t, io.h_in, io.h_out, ws, stream)?;
                }
            }
            if let Some((h_dst, conv_dst)) = io.copy_to {
                gpu.copy_d2d_async(state.recurrent, h_dst, h_bytes, stream)?;
                gpu.copy_d2d_async(state.conv, conv_dst, conv_bytes, stream)?;
            }
        }
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = profile::start();
        let r = self.back_end(gpu, k, ws, stream);
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r
    }

    /// 2026-10-09: `conv_row_io`'s out-of-place launch on row 0 with the input chosen on the
    /// device: `causal_conv1d_update_l2norm_defer`, same grid and the same trailing arguments.
    fn conv_row_defer(
        &self,
        gpu: &dyn GpuBackend,
        live: DevicePtr,
        c_out: DevicePtr,
        defer: &SsmDeferSrc,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let cd = c.conv_dim();
        KernelLaunch::new(gpu, self.kernels.conv_defer)
            .grid([div_ceil(cd as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(live)
            .arg_ptr(defer.conv_base)
            .arg_u64((defer.conv_stride / 4) as u64)
            .arg_ptr(defer.idx)
            .arg_ptr(c_out)
            .arg_ptr(ws.qkv_proj)
            .arg_ptr(self.weights.conv.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(ws.conv_out)
            .arg_u32(1)
            .arg_u32(cd as u32)
            .arg_u32(c.conv_kernel as u32)
            .arg_u32(c.qk_channels() as u32)
            .arg_u32(c.head_dim as u32)
            .arg_f32(c.l2_eps)
            .launch(stream)
    }

    /// 2026-10-09: `recurrent_row_io`'s out-of-place launch on row 0 with the input chosen on the
    /// device: `kda_recurrent_decode_bf16_smem_defer`, same grid, block and shared memory.
    /// 2026-10-12: Its FP16 twin `kda_recurrent_decode_bf16_smem_defer_f16` for an FP16 `state`,
    /// the intermediate stride then in halves.
    fn recurrent_row_defer(
        &self,
        gpu: &dyn GpuBackend,
        state: &KdaSeqState,
        h_out: DevicePtr,
        defer: &SsmDeferSrc,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        let qkv = c.qkv_dim();
        let d = c.head_dim;
        let vpb = KDA_V_PER_BLOCK.min(d);
        let smem = (3 * d + vpb * (d + 1)) * 4;
        let (kernel, h_elem) = if state.recurrent_f16 {
            (self.kernels.recurrent_defer_f16, 2)
        } else {
            (self.kernels.recurrent_defer, 4)
        };
        KernelLaunch::new(gpu, kernel)
            .grid([c.heads as u32, (d / vpb) as u32, 1])
            .block([vpb as u32, 1, 1])
            .shared_mem(smem as u32)
            .arg_ptr(ws.conv_out)
            .arg_ptr(ws.conv_out.offset(qkv * 2))
            .arg_ptr(ws.conv_out.offset(qkv * 4))
            .arg_ptr(ws.gate)
            .arg_ptr(ws.beta)
            .arg_ptr(state.recurrent)
            .arg_ptr(defer.h_base)
            .arg_u64((defer.h_stride / h_elem) as u64)
            .arg_ptr(defer.idx)
            .arg_ptr(h_out)
            .arg_ptr(ws.core)
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
        assert!(defer_commit_requested(Some("1")));
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some(" 1"),
            Some("on"),
            Some("2"),
        ] {
            assert!(!defer_commit_requested(v), "{v:?}");
        }
    }

    const S: KdaSeqState = KdaSeqState {
        conv: DevicePtr(0x1000),
        recurrent: DevicePtr(0x2000),
        recurrent_f16: false,
    };

    /// 2026-10-09: `n` distinct snapshot slots, as the engine's per-slot intermediates.
    fn slots(n: usize) -> Vec<(DevicePtr, DevicePtr)> {
        (0..n)
            .map(|t| {
                let o = t as u64 * 0x100;
                (DevicePtr(0x10_000 + o), DevicePtr(0x20_000 + o))
            })
            .collect()
    }

    /// 2026-10-09: Replays a walk on host values: row `t` maps its input `x` to `x * 31 + t + 1`
    /// (any injective row function serves) and writes it to its output; the last row's copy
    /// carries the value. `row0_in` overrides row 0's input buffer (the device's choice).
    fn replay(
        k: usize,
        sn: &[(DevicePtr, DevicePtr)],
        mem: &mut std::collections::HashMap<DevicePtr, u64>,
        row0_in: Option<(DevicePtr, DevicePtr)>,
    ) {
        for (t, io) in snap_fuse_plan(k, &S, sn).iter().enumerate() {
            let (hi, ci) = match (t, row0_in) {
                (0, Some(src)) => src,
                _ => (io.h_in, io.c_in),
            };
            let (h, c) = (mem[&hi], mem[&ci]);
            mem.insert(io.h_out, h * 31 + t as u64 + 1);
            mem.insert(io.c_out, c * 31 + t as u64 + 1);
            if let Some((hd, cd)) = io.copy_to {
                let (hv, cv) = (mem[&S.recurrent], mem[&S.conv]);
                mem.insert(hd, hv);
                mem.insert(cd, cv);
            }
        }
    }

    /// 2026-10-09: For every deferred index `i` (including `i == 0`, where row 0 reads and writes
    /// slot 0, and indices from a wider previous verify), the walk whose row 0 reads slot `i`
    /// leaves the live state and every slot equal to "copy slot `i` to live, then walk".
    #[test]
    fn deferred_row0_equals_copy_then_walk() {
        for k in 2..=5 {
            for prev_slots in 1..=4usize {
                let sn = slots(prev_slots.max(k - 1));
                for i in 0..prev_slots {
                    let mut base = std::collections::HashMap::new();
                    base.insert(S.recurrent, 7u64);
                    base.insert(S.conv, 9u64);
                    for (t, (h, c)) in sn.iter().enumerate() {
                        base.insert(*h, 100 + t as u64);
                        base.insert(*c, 200 + t as u64);
                    }
                    let mut want = base.clone();
                    let (sh, sc) = (want[&sn[i].0], want[&sn[i].1]);
                    want.insert(S.recurrent, sh);
                    want.insert(S.conv, sc);
                    replay(k, &sn[..k - 1], &mut want, None);
                    let mut got = base.clone();
                    replay(k, &sn[..k - 1], &mut got, Some(sn[i]));
                    // 2026-10-09: The commit's copy also left slot i's value in live; the
                    // deferred arm never wrote live before the walk, which the walk overwrites.
                    assert_eq!(got, want, "k={k} prev_slots={prev_slots} i={i}");
                }
            }
        }
    }
}
