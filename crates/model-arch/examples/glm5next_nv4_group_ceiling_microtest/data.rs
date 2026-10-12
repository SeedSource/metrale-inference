// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: Random packed NVFP4 weights, activations and host <-> device copies for
//! `glm5next_nv4_group_ceiling_microtest`.
//!
//! Owner: model-arch examples.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

pub struct Lcg(pub u64);
impl Lcg {
    pub fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    pub fn n(&mut self) -> f64 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
    pub fn byte(&mut self) -> u8 {
        (self.u() * 256.0) as u8
    }
}

/// Packed E2M1 `[n, k/2]` (any byte is valid) and E4M3 scales `[n, k/16]` (finite, near 1.0:
/// bytes 0x30..=0x3F). Row-major, so members concatenate along N by plain append.
pub fn gen_packed(rng: &mut Lcg, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let w = (0..n * k / 2).map(|_| rng.byte()).collect();
    let s = (0..n * k / 16)
        .map(|_| 0x30 + (rng.byte() & 0x0F))
        .collect();
    (w, s)
}

/// `rows` N(0,1) activation rows as BF16.
pub fn gen_act(rng: &mut Lcg, rows: usize, k: usize) -> Vec<bf16> {
    (0..rows * k).map(|_| bf16::from_f64(rng.n())).collect()
}

pub fn up_bytes(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

pub fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    up_bytes(g, &b)
}

pub fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}
