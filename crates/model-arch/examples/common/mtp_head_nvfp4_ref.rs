// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Host side of `glm5next_mtp_head_nvfp4_microtest`: random head weights and hidden
//! states, device up/down helpers, the NVFP4 / E4M3 decodes, and the threaded FP64 references
//! over the BF16 weights and over the NVFP4 copy's own weights.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types. The decodes follow `w4a16_gemv.cu` (low nibble = even K;
//! one E4M3 scale per 16 K; FP32 tensor scale), as in `glm5next_dense_nvfp4_microtest.rs`.

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-10-06: 64-bit LCG (as the dense NVFP4 microtest).
pub(crate) struct Lcg(pub(crate) u64);
impl Lcg {
    pub(crate) fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Approximately N(0, 1) (Irwin-Hall of 4).
    pub(crate) fn n(&mut self) -> f64 {
        (self.u() + self.u() + self.u() + self.u() - 2.0) * 1.7320508
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get()).clamp(1, 32)
}

/// 2026-10-06: `[n, k]` BF16 bits, N(0, 0.02^2); `heavy` = per-row gain x0.25..x4 and 0.1 %
/// outliers at x30 (the dense microtest's generator). Rows are generated in parallel, row `r`
/// from seed `seed ^ r`, so the weights do not depend on the thread count.
pub(crate) fn gen_weight(seed: u64, n: usize, k: usize, heavy: bool) -> Vec<u16> {
    let mut w = vec![0u16; n * k];
    let per = n.div_ceil(threads());
    std::thread::scope(|sc| {
        for (ci, chunk) in w.chunks_mut(per * k).enumerate() {
            sc.spawn(move || {
                for (ri, row) in chunk.chunks_mut(k).enumerate() {
                    let r = ci * per + ri;
                    let mut rng = Lcg(seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let gain = if heavy {
                        0.25 * 16f64.powf(rng.u())
                    } else {
                        1.0
                    };
                    for x in row.iter_mut() {
                        let mut v = rng.n() * 0.02 * gain;
                        if heavy && rng.u() < 0.001 {
                            v *= 30.0;
                        }
                        *x = bf16::from_f64(v).to_bits();
                    }
                }
            });
        }
    });
    w
}

/// 2026-10-06: Random hidden states, N(0, 1) BF16 (the head input is RMS-normed).
pub(crate) fn gen_act(rng: &mut Lcg, len: usize) -> Vec<u16> {
    (0..len).map(|_| bf16::from_f64(rng.n()).to_bits()).collect()
}

pub(crate) fn up_u16(g: &dyn GpuBackend, d: &[u16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

pub(crate) fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

pub(crate) fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    Ok(dn_bytes(g, p, n * 2)?
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

pub(crate) fn bf(b: u16) -> f64 {
    bf16::from_bits(b).to_f64()
}

/// 2026-10-06: OCP E4M3 decode (bias 7, subnormal m * 2^-9; 0x7F/0xFF are NaN).
pub(crate) fn e4m3(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f64;
    if e == 15 && (b & 7) == 7 {
        return f64::NAN;
    }
    if e == 0 {
        s * m * 2f64.powi(-9)
    } else {
        s * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-10-06: The NVFP4 copy as downloaded: packed `[n, k/2]`, scales `[n, k/16]`, `s2`.
pub(crate) struct Nv4Host {
    pub(crate) packed: Vec<u8>,
    pub(crate) scales: Vec<u8>,
    pub(crate) s2: f64,
}

/// 2026-10-06: FP64 references for `rows` hidden states `act` (`[rows, k]` BF16 bits) over
/// the `[n, k]` BF16 weights `w` and over the NVFP4 copy's own weights `q`: returns
/// `(ref_bf16w, ref_own)`, each `[rows, n]`. Threaded over output columns.
pub(crate) fn fp64_refs(
    w: &[u16],
    q: &Nv4Host,
    act: &[u16],
    rows: usize,
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = act[..rows * k].iter().map(|&b| bf(b)).collect();
    let per = n.div_ceil(threads());
    let parts: Vec<Vec<(f64, f64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n.div_ceil(per))
            .map(|ci| {
                let a = &a;
                sc.spawn(move || {
                    let (j0, j1) = (ci * per, ((ci + 1) * per).min(n));
                    let (mut wb, mut wq) = (vec![0.0f64; k], vec![0.0f64; k]);
                    let mut out = Vec::with_capacity((j1 - j0) * rows);
                    for j in j0..j1 {
                        for (i, (b, v)) in wb.iter_mut().zip(wq.iter_mut()).enumerate() {
                            *b = bf(w[j * k + i]);
                            let byte = q.packed[(j * k + i) / 2];
                            let nib = if i.is_multiple_of(2) { byte & 0xF } else { byte >> 4 };
                            *v = E2M1[nib as usize] * e4m3(q.scales[j * (k / 16) + i / 16]) * q.s2;
                        }
                        for at in a.chunks_exact(k) {
                            let (mut s1, mut s2) = (0.0, 0.0);
                            for ((&x, &y1), &y2) in at.iter().zip(&wb).zip(&wq) {
                                s1 += x * y1;
                                s2 += x * y2;
                            }
                            out.push((s1, s2));
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("reference thread")).collect()
    });
    let (mut rb, mut ro) = (vec![0.0f64; rows * n], vec![0.0f64; rows * n]);
    let mut j = 0;
    for part in parts {
        for col in part.chunks_exact(rows) {
            for (t, &(s1, s2)) in col.iter().enumerate() {
                rb[t * n + j] = s1;
                ro[t * n + j] = s2;
            }
            j += 1;
        }
    }
    (rb, ro)
}

/// 2026-10-06: Per-row (cos, max|y - r| / max|r|, nonfinite count) of `y` against `r`.
pub(crate) fn row_stats(y: &[f64], r: &[f64]) -> (f64, f64, usize) {
    let (mut dot, mut yy, mut rr, mut maxd, mut maxr) = (0.0, 0.0, 0.0, 0.0f64, 0.0f64);
    let mut nonfinite = 0;
    for (&a, &b) in y.iter().zip(r) {
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        dot += a * b;
        yy += a * a;
        rr += b * b;
        maxd = maxd.max((a - b).abs());
        maxr = maxr.max(b.abs());
    }
    let cos = if yy > 0.0 && rr > 0.0 {
        dot / (yy.sqrt() * rr.sqrt())
    } else {
        1.0
    };
    (cos, maxd / maxr.max(1e-30), nonfinite)
}

/// 2026-10-06: Index of the largest BF16 value of `row` (the lowest index on a tie); NaN never
/// wins.
pub(crate) fn argmax(row: &[u16]) -> usize {
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, &b) in row.iter().enumerate() {
        let v = bf(b);
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0
}
