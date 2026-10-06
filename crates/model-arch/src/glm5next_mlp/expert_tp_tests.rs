// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The expert-TP slices of one NVFP4 projection (`expert_tp::slice_nvfp4`) against
//! the whole tensor, on pseudo-random bytes whose position is checked exactly: the two ranks'
//! slices reassemble into the original packed and scale bytes, a down_proj column cut lands on
//! 16-element scale-group boundaries, and `scale_2` / `input_scale` stay whole on both ranks.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::expert_tp::{Cut, Nvfp4Host, col_part, row_range, slice_nvfp4};

fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

fn proj(rows: usize, cols: usize, seed: u64) -> Nvfp4Host {
    Nvfp4Host {
        packed: bytes(seed, rows * cols / 2),
        scale: bytes(seed ^ 0x5ca1e, rows * cols / 16),
        scale_2: 0.0123,
        input_scale: 0.75,
    }
}

/// 2026-10-05: gate/up (`[I, H]`, cut by rows): rank 0 then rank 1 is the whole tensor.
#[test]
fn expert_tp_row_slices_reassemble_the_whole_tensor() {
    for (rows, cols) in [(8usize, 64usize), (2048, 4096)] {
        let full = proj(rows, cols, 7);
        let p: Vec<Nvfp4Host> = (0..2)
            .map(|r| slice_nvfp4(&full, rows, cols, Cut::Rows, r, 2).unwrap())
            .collect();
        for s in &p {
            assert_eq!(s.packed.len(), rows / 2 * cols / 2);
            assert_eq!(s.scale.len(), rows / 2 * cols / 16);
            assert_eq!(s.scale_2, full.scale_2, "scale_2 must stay whole");
            assert_eq!(s.input_scale, full.input_scale, "input_scale must stay whole");
        }
        assert_eq!([p[0].packed.clone(), p[1].packed.clone()].concat(), full.packed);
        assert_eq!([p[0].scale.clone(), p[1].scale.clone()].concat(), full.scale);
        // 2026-10-05: The range the loader reads from disk is the same rows.
        let rr = row_range(full.packed.len(), rows, 1, 2).unwrap();
        assert_eq!(&full.packed[rr], p[1].packed.as_slice());
    }
}

/// 2026-10-05: down (`[H, I]`, cut by columns): per row, rank 0's bytes then rank 1's are the
/// whole row; scale group `j` of rank `r`'s slice is group `r * G/2 + j` of the whole row, with
/// its 8 packed bytes, so no group straddles the cut.
#[test]
fn expert_tp_column_slices_reassemble_on_scale_group_boundaries() {
    for (rows, cols) in [(4usize, 64usize), (4096, 2048)] {
        let full = proj(rows, cols, 11);
        let p: Vec<Nvfp4Host> = (0..2)
            .map(|r| slice_nvfp4(&full, rows, cols, Cut::Cols, r, 2).unwrap())
            .collect();
        let (pb, sb) = (cols / 2, cols / 16);
        let (hp, hs) = (pb / 2, sb / 2);
        for (r, s) in p.iter().enumerate() {
            assert_eq!(s.packed.len(), rows * hp);
            assert_eq!(s.scale.len(), rows * hs);
            assert_eq!(s.scale_2, full.scale_2, "scale_2 must stay whole");
            assert_eq!(s.input_scale, full.input_scale, "input_scale must stay whole");
            for n in 0..rows {
                for j in 0..hs {
                    let g = r * hs + j;
                    assert_eq!(s.scale[n * hs + j], full.scale[n * sb + g], "row {n} group {j}");
                    assert_eq!(
                        &s.packed[n * hp + j * 8..n * hp + j * 8 + 8],
                        &full.packed[n * pb + g * 8..n * pb + g * 8 + 8],
                        "rank {r} row {n} group {j}"
                    );
                }
            }
        }
        for n in 0..rows {
            let row = [
                &p[0].packed[n * hp..(n + 1) * hp],
                &p[1].packed[n * hp..(n + 1) * hp],
            ]
            .concat();
            assert_eq!(row.as_slice(), &full.packed[n * pb..(n + 1) * pb]);
            let srow = [
                &p[0].scale[n * hs..(n + 1) * hs],
                &p[1].scale[n * hs..(n + 1) * hs],
            ]
            .concat();
            assert_eq!(srow.as_slice(), &full.scale[n * sb..(n + 1) * sb]);
        }
    }
}

/// 2026-10-05: The leaf name picks the cut.
#[test]
fn expert_tp_cut_follows_the_projection() {
    assert_eq!(Cut::of("gate_proj"), Cut::Rows);
    assert_eq!(Cut::of("up_proj"), Cut::Rows);
    assert_eq!(Cut::of("down_proj"), Cut::Cols);
}

/// 2026-10-05: A cut that would split a scale group, a ragged row count or a wrong byte count
/// is refused, not sliced.
#[test]
fn expert_tp_refuses_cuts_that_do_not_split_cleanly() {
    // 2026-10-05: 48 columns halve to 24, which splits a 16-element group.
    let p = proj(4, 48, 3);
    assert!(slice_nvfp4(&p, 4, 48, Cut::Cols, 0, 2).is_err());
    // 2026-10-05: 3 rows do not halve.
    let p = proj(3, 64, 3);
    assert!(slice_nvfp4(&p, 3, 64, Cut::Rows, 0, 2).is_err());
    // 2026-10-05: The buffers do not match the stated shape.
    let p = proj(4, 64, 3);
    assert!(slice_nvfp4(&p, 8, 64, Cut::Rows, 0, 2).is_err());
    // 2026-10-05: A rank outside the world.
    assert!(slice_nvfp4(&p, 4, 64, Cut::Rows, 2, 2).is_err());
    assert!(col_part(&p.packed, 4, 2, 2).is_err());
    assert!(row_range(p.packed.len(), 4, 2, 2).is_err());
}
