// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: [`super::csf32`]: round trips through the test encoder over the FP4-Local shapes
//! and the geometry edges, structural refusals, and (when `METRALE_TEST_CSF32_DIR` names a dump)
//! byte agreement with the Python decoder on real FP4-Local tensors. The real data is never
//! committed (NVIDIA Open Model Agreement; weights stay on the nodes).
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: none beyond the types.

use super::csf32::{Csf32Parts, decode, encode};

/// Deterministic scale bytes in the E4M3 range FP4-Local shows (97..=126), with a share of
/// outliers more than 15 above the row minimum so exceptions occur.
fn scales(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..rows * cols)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (x >> 33) as u32;
            if r % 97 == 0 { 126 } else { 97 + (r % 12) as u8 }
        })
        .collect()
}

fn round_trip(rows: usize, cols: usize, seed: u64) -> usize {
    let s = scales(rows, cols, seed);
    let (b, c, o, pb, e) = encode(rows, cols, &s);
    let p = Csf32Parts { bases: &b, codes: &c, offsets: &o, page_bases: &pb, exceptions: &e };
    assert_eq!(decode("t", rows, cols, &p).unwrap(), s, "[{rows}, {cols}]");
    e.len()
}

#[test]
fn round_trips_on_fp4_local_shapes() {
    // gate/up [2048 x 256 scales], down [4096 x 128]; real tensors had 0.25-0.33% exceptions.
    assert!(round_trip(2048, 256, 1) > 0);
    assert!(round_trip(4096, 128, 2) > 0);
}

#[test]
fn round_trips_on_geometry_edges() {
    // one block; odd column count (a 1-wide last pair); row blocks not a multiple of 4;
    // more than one page (32 groups) with a partial last page.
    for (r, c) in [(32, 2), (32, 1), (64, 3), (96, 6), (32 * 4 * 9, 2 * 2 * 4 + 1)] {
        round_trip(r, c, (r * 31 + c) as u64);
    }
}

#[test]
fn every_value_reachable_through_exceptions() {
    // a row min of 0 with 255 present forces exceptions carrying the full byte range
    let mut s = vec![0u8; 32 * 2];
    for (i, v) in s.iter_mut().enumerate() {
        *v = (i * 4) as u8;
    }
    s[63] = 255;
    let (b, c, o, pb, e) = encode(32, 2, &s);
    let p = Csf32Parts { bases: &b, codes: &c, offsets: &o, page_bases: &pb, exceptions: &e };
    assert_eq!(decode("t", 32, 2, &p).unwrap(), s);
}

#[test]
fn structural_faults_are_refused() {
    let s = scales(64, 4, 7);
    let (b, c, o, pb, e) = encode(64, 4, &s);
    let ok = |b: &[u8], c: &[u8], o: &[u16], pb: &[u32], e: &[u16]| {
        decode("t", 64, 4, &Csf32Parts { bases: b, codes: c, offsets: o, page_bases: pb, exceptions: e })
    };
    assert!(ok(&b, &c, &o, &pb, &e).is_ok());
    assert!(ok(&b[1..], &c, &o, &pb, &e).unwrap_err().to_string().contains("csf32_bases"));
    assert!(ok(&b, &c[1..], &o, &pb, &e).unwrap_err().to_string().contains("csf32_codes"));
    assert!(ok(&b, &c, &o[1..], &pb, &e).unwrap_err().to_string().contains("csf32_offsets"));
    let mut o2 = o.clone();
    o2[0] = 1;
    assert!(ok(&b, &c, &o2, &pb, &e).unwrap_err().to_string().contains("starts at"));
    let mut e2 = e.clone();
    e2.push(0);
    assert!(ok(&b, &c, &o, &pb, &e2).unwrap_err().to_string().contains("consumed"));
    assert!(decode("t", 48, 4, &Csf32Parts { bases: &b, codes: &c, offsets: &o, page_bases: &pb, exceptions: &e })
        .unwrap_err()
        .to_string()
        .contains("multiple of 32"));
}

/// Real FP4-Local tensors (layers.10.mlp.experts.0 gate_proj [2048 x 256], down_proj [4096 x 128]),
/// dumped by the Python checker (runs/race/pf-fp4l/csf32_decode_check.py) as raw little-endian
/// `<proj>.<part>.bin` plus its decoded `<proj>.plain.bin`. Skipped when the env var is unset.
#[test]
fn agrees_with_the_python_decoder_on_real_tensors() {
    let Ok(dir) = std::env::var("METRALE_TEST_CSF32_DIR") else {
        eprintln!("METRALE_TEST_CSF32_DIR unset: real-tensor agreement skipped");
        return;
    };
    let rd = |p: &str, part: &str| std::fs::read(format!("{dir}/{p}.{part}.bin")).unwrap();
    let u16s = |v: Vec<u8>| v.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>();
    let u32s = |v: Vec<u8>| v.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>();
    for (proj, rows, cols) in [("gate_proj", 2048, 256), ("down_proj", 4096, 128)] {
        let (b, c) = (rd(proj, "bases"), rd(proj, "codes"));
        let (o, pb, e) = (u16s(rd(proj, "offsets")), u32s(rd(proj, "page_bases")), u16s(rd(proj, "exceptions")));
        let p = Csf32Parts { bases: &b, codes: &c, offsets: &o, page_bases: &pb, exceptions: &e };
        let got = decode(proj, rows, cols, &p).unwrap();
        assert_eq!(got, rd(proj, "plain"), "{proj}: Rust and Python decoders disagree");
        eprintln!("{proj}: {} scales, {} exceptions, byte-identical to the Python decoder", got.len(), e.len());
    }
}
