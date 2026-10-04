// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `BufferArena::debug_buffer_checksum`, a debug fingerprint of the arena's buffers.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use super::BufferArena;
use crate::gpu::{DevicePtr, GpuBackend};

impl BufferArena {
    /// 2026-10-04: Synchronize `stream`, then log at warn level the sum, sum of
    /// squares and sum of absolute values of each listed buffer read as f32
    /// lanes over its whole allocation (non-finite lanes skipped), tagged with
    /// `tag`. A failed synchronize or copy is ignored; a failed copy skips that
    /// buffer, and so does a NULL or 0-byte one (an entry the GLM-5.3 trim left
    /// unallocated). The model engine calls it at the first decode step when
    /// `ssm_save_dump` is set.
    pub fn debug_buffer_checksum(&self, gpu: &dyn GpuBackend, stream: u64, tag: &str) {
        gpu.synchronize(stream).ok();
        let probe = |name: &str, ptr: DevicePtr, bytes: usize| {
            if ptr.is_null() || bytes == 0 {
                return;
            }
            let mut hb = vec![0u8; bytes];
            if gpu.copy_d2h(ptr, &mut hb).is_err() {
                return;
            }
            let (mut sum, mut ssq, mut sabs) = (0f64, 0f64, 0f64);
            for c in hb.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
                if v.is_finite() {
                    sum += v;
                    ssq += v * v;
                    sabs += v.abs();
                }
            }
            tracing::warn!(
                "METRALE_BUF_CKSUM[{tag}] {name} bytes={bytes} sum={sum:.6} ssq={ssq:.6} sabs={sabs:.6}"
            );
        };
        probe(
            "hidden_states",
            self.hidden_states,
            self.sizes.hidden_states,
        );
        probe("residual", self.residual, self.sizes.residual);
        probe("norm_output", self.norm_output, self.sizes.norm_output);
        probe("qkv_output", self.qkv_output, self.sizes.qkv_output);
        probe("attn_output", self.attn_output, self.sizes.attn_output);
        probe("gate_logits", self.gate_logits, self.sizes.gate_logits);
        probe("moe_output", self.moe_output, self.sizes.moe_output);
        probe("ssm_qkvz", self.ssm_qkvz, self.sizes.ssm_qkvz);
        probe("ssm_ba", self.ssm_ba, self.sizes.ssm_ba);
        probe(
            "ssm_deinterleaved",
            self.ssm_deinterleaved,
            self.sizes.ssm_deinterleaved,
        );
        probe("ssm_gates", self.ssm_gates, self.sizes.ssm_gates);
        probe(
            "ssm_conv_out_f32",
            self.ssm_conv_out_f32,
            self.sizes.ssm_conv_out_f32,
        );
        probe(
            "expert_gate_out",
            self.expert_gate_out,
            self.sizes.expert_gate_out,
        );
        probe(
            "expert_up_out",
            self.expert_up_out,
            self.sizes.expert_up_out,
        );
        probe(
            "expert_down_out",
            self.expert_down_out,
            self.sizes.expert_down_out,
        );
        probe(
            "splitk_workspace",
            self.splitk_workspace,
            self.sizes.splitk_workspace,
        );
    }
}
