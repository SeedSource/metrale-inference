// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: CPU emulation of the row-batched MoE gate/up sweep: the car entry
//! `w4a16_gemv_sw_moe_batchm_m<R>` (per-lane chains + shuffle-down tree, per matrix) against
//! `w4a16_gemv_sw_moe_batchm_gateup_m<R>` (one launch for both matrices, per-chunk dequant
//! shared across rows, transposed tree), both in kernels/gb10/common/w4a16_gemv.cu. Every FP32
//! operation is emulated with the same rounding (`f32::mul_add` = fmaf, `+` = __fadd_rn; the
//! kernels build with --fmad=false), so equal bits here mean the two kernels evaluate the same
//! rounded operations on the same operands.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

const LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// E4M3 byte -> f32 (exact), as `(float)__nv_fp8_e4m3`.
fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// One expert matrix [N, K]: nibbles (row-major, element e of chunk kk is nibble e of its
/// 8-byte word) and the per-chunk scale `fp8 * scale2`.
struct Mat {
    nib: Vec<u8>,
    sc: Vec<f32>,
}

fn gen_mat(rng: &mut Rng, n: usize, k: usize) -> Mat {
    let scale2 = 0.75 + (rng.next() % 1000) as f32 / 997.0;
    let nib = (0..n * k).map(|_| (rng.next() & 0xF) as u8).collect();
    let sc = (0..n * k / 16)
        .map(|_| {
            let mut b = (rng.next() & 0xFF) as u8;
            if b & 0x7F == 0x7F {
                b ^= 1; // never NaN
            }
            e4m3(b) * scale2
        })
        .collect();
    Mat { nib, sc }
}

/// BF16-representable activations with magnitudes over 2^-8 .. 2^8 (rounding-order sensitive).
fn gen_act(rng: &mut Rng, k: usize) -> Vec<f32> {
    (0..k)
        .map(|_| {
            let bits = (rng.next() & 0xFFFF) as u16;
            let exp = 119 + (rng.next() % 17) as u32; // 2^-8 .. 2^8
            let f = f32::from_bits(((bits as u32 & 0x807F) << 16) | (exp << 23));
            f32::from_bits(f.to_bits() & 0xFFFF_0000)
        })
        .collect()
}

/// 16-step part chain of chunk kk of column n (fmaf(af, lut, part) from 0.0f).
fn part(a: &[f32], m: &Mat, n: usize, k: usize, kk: usize) -> f32 {
    let mut p = 0.0f32;
    for e in 0..16 {
        let w = LUT[m.nib[n * k + kk * 16 + e] as usize];
        p = a[kk * 16 + e].mul_add(w, p);
    }
    p
}

/// The car: w4a16_gemv_partial_rows (orig lanes 0..63, its exact loop) + shuffle-down tree on
/// each chain + acc_a + acc_b.
fn car(a: &[f32], m: &Mat, n: usize, k: usize) -> f32 {
    let k16 = k / 16;
    let mut lanes = [[0.0f32; 32]; 2];
    for o in 0..64usize {
        let (mut acc0, mut acc1) = (0.0f32, 0.0f32);
        let mut k16i = o * 2;
        while k16i < k16 + 1 {
            for c in 0..2 {
                let kk = k16i + c;
                if kk >= k16 {
                    break;
                }
                let p = part(a, m, n, k, kk);
                if c == 0 {
                    acc0 = m.sc[n * k16 + kk].mul_add(p, acc0);
                } else {
                    acc1 = m.sc[n * k16 + kk].mul_add(p, acc1);
                }
            }
            k16i += 128;
        }
        lanes[o / 32][o % 32] = acc0 + acc1;
    }
    let mut t = [0.0f32; 2];
    for (ch, v) in lanes.iter_mut().enumerate() {
        let mut off = 16;
        while off > 0 {
            let old = *v;
            for i in 0..32 {
                // __shfl_down_sync: an out-of-range source returns the lane's own value.
                let src = if i + off < 32 { old[i + off] } else { old[i] };
                v[i] = old[i] + src;
            }
            off >>= 1;
        }
        t[ch] = v[0];
    }
    t[0] + t[1]
}

/// The gateup entry for one column n of both matrices over R rows (`live[r]` false = slot -1):
/// per-lane loops exactly as the kernel, then the transposed tree and the butterfly simulated
/// across 32 lanes. Returns the stored f32 per (mat, row), None where nothing is stored.
fn gateup(
    acts: &[Vec<f32>],
    live: &[bool],
    m: [&Mat; 2],
    n: usize,
    k: usize,
) -> Vec<[Option<f32>; 2]> {
    let r_n = acts.len();
    let rp: usize = if r_n <= 2 {
        2
    } else if r_n <= 4 {
        4
    } else {
        8
    };
    let v_n = 2 * rp;
    let logv = v_n.trailing_zeros() as usize;
    let k16 = k / 16;
    let mut t = vec![[0.0f32; 2]; 32];
    for ch in 0..2usize {
        // v[lane][mat * RP + r]
        let mut v = vec![vec![0.0f32; v_n]; 32];
        for (lane, vl) in v.iter_mut().enumerate() {
            for h in 0..2usize {
                let mut acc = [vec![0.0f32; r_n], vec![0.0f32; r_n]];
                let mut kk = 2 * (lane + 32 * ch) + h;
                while kk < k16 {
                    for r in 0..r_n {
                        if !live[r] {
                            continue;
                        }
                        let pg = part(&acts[r], m[0], n, k, kk);
                        let pu = part(&acts[r], m[1], n, k, kk);
                        acc[0][r] = m[0].sc[n * k16 + kk].mul_add(pg, acc[0][r]);
                        acc[1][r] = m[1].sc[n * k16 + kk].mul_add(pu, acc[1][r]);
                    }
                    kk += 128;
                }
                for r in 0..r_n {
                    if h == 0 {
                        vl[r] = acc[0][r];
                        vl[rp + r] = acc[1][r];
                    } else {
                        vl[r] += acc[0][r];
                        vl[rp + r] += acc[1][r];
                    }
                }
            }
        }
        // Transposed levels.
        for s in 0..logv {
            let o = 16usize >> s;
            let hh = v_n >> (s + 1);
            let old = v.clone();
            for lane in 0..32 {
                let hi = lane & o != 0;
                let partner = lane ^ o;
                for kq in 0..hh {
                    let keep = if hi {
                        old[lane][kq + hh]
                    } else {
                        old[lane][kq]
                    };
                    // The partner sends what it does not keep: v[k] if it is hi, else v[k + hh].
                    let recv = if hi {
                        old[partner][kq + hh]
                    } else {
                        old[partner][kq]
                    };
                    v[lane][kq] = keep + recv;
                }
            }
        }
        // Butterfly.
        let mut tv: Vec<f32> = (0..32).map(|l| v[l][0]).collect();
        let mut o = 16usize >> logv;
        while o > 0 {
            let old = tv.clone();
            for lane in 0..32 {
                tv[lane] = old[lane] + old[lane ^ o];
            }
            o >>= 1;
        }
        for lane in 0..32 {
            t[lane][ch] = tv[lane];
        }
    }
    let mut out = vec![[None; 2]; r_n];
    for lane in 0..32usize {
        let m_i = lane >> (5 - logv);
        let holder = lane & ((32 >> logv) - 1) == 0;
        let (mat, r) = (m_i / rp, m_i % rp);
        if holder && r < r_n && live[r] {
            assert!(
                out[r][mat].is_none(),
                "two lanes store (mat {mat}, row {r})"
            );
            out[r][mat] = Some(t[lane][0] + t[lane][1]);
        }
    }
    out
}

/// 2026-10-09: For R = 2..=8 and K = 4096 (GLM-5.3 hidden, K16 = 256) plus K = 1024, 3072 and
/// 2080 (lanes with fewer or no chunks), every live (mat, row) output of the gateup entry has
/// the car's f32 bits, and nothing is stored for a dead row.
#[test]
fn gateup_fast_matches_car_bits() {
    let mut rng = Rng(0x6761_7465_7570);
    let mut outputs = 0usize;
    for k in [4096usize, 1024, 3072, 2080] {
        let n_cols = 3;
        let (mg, mu) = (gen_mat(&mut rng, n_cols, k), gen_mat(&mut rng, n_cols, k));
        for r_n in 2..=8usize {
            let acts: Vec<Vec<f32>> = (0..r_n).map(|_| gen_act(&mut rng, k)).collect();
            let live: Vec<bool> = (0..r_n).map(|r| r_n < 4 || r % 3 != 1).collect();
            for n in 0..n_cols {
                let got = gateup(&acts, &live, [&mg, &mu], n, k);
                for r in 0..r_n {
                    for (mat, mm) in [&mg, &mu].into_iter().enumerate() {
                        match (live[r], got[r][mat]) {
                            (false, None) => {}
                            (true, Some(x)) => {
                                let want = car(&acts[r], mm, n, k);
                                assert_eq!(
                                    x.to_bits(),
                                    want.to_bits(),
                                    "K={k} R={r_n} n={n} mat={mat} row={r}: {x} vs car {want}"
                                );
                                outputs += 1;
                            }
                            (l, g) => {
                                panic!("K={k} R={r_n} row={r} mat={mat}: live {l} stored {g:?}")
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(outputs > 500, "only {outputs} outputs compared");
}

/// 2026-10-09: Sensitivity control: the same data summed in a different order (one sequential
/// FP32 sum over the 64 lane values) differs from the car on some outputs, so the bit match
/// above is not an artefact of order-insensitive data.
#[test]
fn emulation_detects_a_reordered_sum() {
    let mut rng = Rng(0x006f_7264_6572);
    let k = 4096;
    let m = gen_mat(&mut rng, 8, k);
    let mut differ = 0;
    for n in 0..8 {
        let a = gen_act(&mut rng, k);
        let k16 = k / 16;
        let mut seq = 0.0f32;
        for o in 0..64usize {
            let (mut acc0, mut acc1) = (0.0f32, 0.0f32);
            for kk in [o * 2, o * 2 + 128] {
                acc0 = m.sc[n * k16 + kk].mul_add(part(&a, &m, n, k, kk), acc0);
                acc1 = m.sc[n * k16 + kk + 1].mul_add(part(&a, &m, n, k, kk + 1), acc1);
            }
            seq += acc0 + acc1;
        }
        if seq.to_bits() != car(&a, &m, n, k).to_bits() {
            differ += 1;
        }
    }
    assert!(
        differ > 0,
        "a sequential sum matched the tree on every column"
    );
}
