// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`super`] (scale rows with escape): round trips over the GLM-5.3 expert-TP shapes
//! and the edges, the host reader against decode at every position, base choice, header words,
//! and refusals.
//!
//! Owner: model-arch GLM-5.3 MoE.
//! Invariants: none beyond the types.

use super::*;

/// Deterministic scale bytes like the official expert scales (a narrow band near 0x60..0x70 per
/// row, the band moving with the row) plus outliers on both sides, so escapes occur.
fn scales(rows: usize, c: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    let mut out = Vec::with_capacity(rows * c);
    for n in 0..rows {
        let band = 92 + (n % 13) as u8;
        for _ in 0..c {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (x >> 33) as u32;
            out.push(match r % 211 {
                0 => band.wrapping_add(15 + (r % 20) as u8),
                1 => band.wrapping_sub(1 + (r % 30) as u8),
                2 => 0,
                3 => 255,
                _ => band + (r % 14) as u8,
            });
        }
    }
    out
}

fn round_trip(rows: usize, c: usize, plain: &[u8]) -> ScaleRows {
    let s = encode(rows, c, plain).unwrap();
    assert_eq!(decode(rows, c, &s).unwrap(), plain, "[{rows}, {c}]");
    for n in 0..rows {
        for kk in 0..c {
            assert_eq!(
                scale_byte(&s.bytes, s.head, c, n, kk),
                plain[n * c + kk],
                "({n}, {kk}) of [{rows}, {c}]"
            );
        }
    }
    assert_eq!(s.bytes.len(), encoded_len(rows, c, s.exceptions));
    s
}

#[test]
fn round_trips_glm_expert_tp_shapes() {
    // gate/up per rank [I/2 = 1024, hidden/16 = 256]; down per rank [hidden 4096, (I/2)/16 = 64];
    // unsliced [2048, 256] and [4096, 128].
    for (rows, c, seed) in [
        (1024, 256, 1),
        (4096, 64, 2),
        (2048, 256, 3),
        (4096, 128, 4),
    ] {
        let s = round_trip(rows, c, &scales(rows, c, seed));
        assert!(s.exceptions > 0, "[{rows}, {c}] fixture has no escapes");
        assert!(
            s.bytes.len() * 100 < rows * c * 60,
            "[{rows}, {c}]: {} bytes",
            s.bytes.len()
        );
    }
}

#[test]
fn round_trips_edges() {
    round_trip(1, 2, &[7, 200]);
    round_trip(1, 256, &(0..=255u8).collect::<Vec<_>>());
    round_trip(3, 4, &[255; 12]);
    round_trip(3, 4, &[0; 12]);
    // every byte an escape but the window
    let mut wide = vec![0u8; 2 * 256];
    for (i, v) in wide.iter_mut().enumerate() {
        *v = (i * 37 % 256) as u8;
    }
    round_trip(2, 256, &wide);
}

#[test]
fn base_is_the_fullest_window_and_codes_stay_below_escape() {
    // 14 bytes at 100..=113, 3 at 40: base 99 or 100 both hold 14; the smallest wins.
    let mut row: Vec<u8> = (100..=113).collect();
    row.extend([40, 40]);
    assert_eq!(row_base(&row), 99);
    let s = encode(1, row.len(), &row).unwrap();
    assert_eq!(s.exceptions, 2);
    for kk in 0..row.len() {
        let b = s.bytes[s.head + 1 + kk / 2];
        let code = if kk % 2 == 0 { b & 15 } else { b >> 4 };
        assert_eq!(code == ESCAPE, row[kk] == 40, "kk {kk}");
    }
}

#[test]
fn header_words_and_alignment() {
    let s = encode(5, 8, &scales(5, 8, 9)).unwrap();
    assert_eq!(s.head % 4, 0);
    assert_eq!(u32_at(&s.bytes, s.head - 8) as usize, s.head);
    assert_eq!(u32_at(&s.bytes, s.head - 4), 5);
    assert_eq!(s.device_ptr(0x1000) & 1, 1);
    assert_eq!(s.device_ptr(0x1000) & !1, 0x1000 + s.head as u64);
}

#[test]
fn refuses_what_it_cannot_hold() {
    assert!(encode(1, 3, &[0; 3]).is_err());
    assert!(encode(1, 258, &[0; 258]).is_err());
    assert!(encode(0, 2, &[]).is_err());
    assert!(encode(2, 2, &[0; 3]).is_err());
    // 65536 escapes: every byte of a [256, 256] scale alternates far apart in each row pair
    let mut many = vec![0u8; 512 * 256];
    for (i, v) in many.iter_mut().enumerate() {
        *v = if (i / 16) % 2 == 0 {
            (i % 16) as u8 * 16
        } else {
            255 - (i % 16) as u8 * 16
        };
    }
    assert!(encode(512, 256, &many).is_err());
}

#[test]
fn decode_refuses_damage() {
    let plain = scales(64, 64, 5);
    let s = encode(64, 64, &plain).unwrap();
    assert!(s.exceptions > 0);
    // drop one record from the last row that has one: an escape without a record
    let mut d = s.clone();
    let rows = 64;
    let n = (0..rows)
        .rev()
        .find(|&n| u16_at(&d.bytes, 2 * (n + 1)) > u16_at(&d.bytes, 2 * n))
        .unwrap();
    let v = u16_at(&d.bytes, 2 * (n + 1)) - 1;
    d.bytes[2 * (n + 1)..2 * (n + 1) + 2].copy_from_slice(&v.to_le_bytes());
    assert!(decode(rows, 64, &d).is_err());
    // wrong header word
    let mut h = s.clone();
    h.bytes[h.head - 4] ^= 1;
    assert!(decode(rows, 64, &h).is_err());
}
