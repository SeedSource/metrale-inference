// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the DSA decode launcher's guards, geometry only. The
//! kernel's numerics are checked on a GPU by the `glm5next_dsa_decode_gate`
//! example.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::select::DsaSelectGeometry;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

fn paging() -> DsaDecodePaging {
    DsaDecodePaging {
        num_seqs: 1,
        num_q_heads: 64,
        num_kv_heads: 1,
        max_blocks_per_seq: 256,
        block_size: 64,
        cache_stride_bytes: 64 * 512,
    }
}

/// 2026-09-25: Pools are built over absolute positions, so a `block_size` that
/// is not a multiple of `index_kpool` would split a pool across two pages.
#[test]
fn a_block_size_that_splits_a_pool_is_refused() {
    let c = cfg();
    let mut p = paging();
    assert!(p.validate(&c).is_ok(), "64 is a multiple of kpool 4");
    p.block_size = 66;
    let e = p.validate(&c).unwrap_err().to_string();
    assert!(e.contains("straddle"), "{e}");
}

/// 2026-09-25: MLA has one latent KV head; `validate` refuses any other count.
#[test]
fn more_than_one_kv_head_is_refused() {
    let c = cfg();
    let mut p = paging();
    p.num_kv_heads = 8;
    assert!(p.validate(&c).is_err());
}

#[test]
fn degenerate_launches_are_refused() {
    let c = cfg();
    for mutate in [
        (|p: &mut DsaDecodePaging| p.num_seqs = 0) as fn(&mut DsaDecodePaging),
        |p: &mut DsaDecodePaging| p.num_q_heads = 0,
        |p: &mut DsaDecodePaging| p.block_size = 0,
    ] {
        let mut p = paging();
        mutate(&mut p);
        assert!(p.validate(&c).is_err(), "{p:?} must be refused");
    }
}

/// 2026-09-25: `decode_attention` refuses a selection planned for a different row
/// count than it decodes. The launch needs a GPU, so this checks only that the
/// refused condition is reachable from `DsaSelectGeometry::plan`.
#[test]
fn a_row_count_mismatch_between_selection_and_launch_is_refused() {
    let c = cfg();
    let p = paging();
    let two_rows = DsaSelectGeometry::plan(&c, 4_096, 2).unwrap();
    assert_eq!(two_rows.q_rows, 2);
    assert_ne!(two_rows.q_rows, p.num_seqs);
    assert!(
        two_rows.q_rows != p.num_seqs,
        "the guarded condition must be reachable"
    );
    let one_row = DsaSelectGeometry::plan(&c, 4_096, 1).unwrap();
    assert_eq!(one_row.q_rows, p.num_seqs);
}

/// 2026-09-25: NoPE: the softmax scale is over the latent width, which is the whole
/// cache token, not over a 576-wide token with a 64-dim rope tail (`MLA_CACHE_DIM`
/// in the DeepSeek-V4-Flash decode kernels).
#[test]
fn the_score_scale_is_the_latent_width_not_the_v4_cache_width() {
    let c = cfg();
    assert_eq!(c.kv_cache_dim(), 512, "NoPE: cache token is pure latent");
    assert_eq!(c.kv_cache_dim(), c.kv_lora_rank);
    let ours = (c.kv_lora_rank as f32).powf(-0.5);
    let v4 = 576f32.powf(-0.5);
    assert!((ours - 0.044_194_173).abs() < 1e-9, "1/sqrt(512)");
    assert!(
        (ours - v4).abs() > 1e-4,
        "the two scales must not be interchangeable by accident"
    );
}

/// 2026-09-29 (A153): `METRALE_GLM_MLA_SCALE_AUTHOR=1`'s arm is `qk_head_dim^-0.5`
/// (`qk_nope_head_dim + qk_rope_head_dim`), 1/sqrt(256) = 1/16 for GLM-5.3's 256+0 split; the
/// default (off) arm is unchanged from the test above.
#[test]
fn author_scale_arm_is_one_sixteenth_for_256_plus_0() {
    let c = cfg();
    assert_eq!(c.qk_nope_head_dim + c.qk_rope_head_dim, 256);
    let author = mla_scale_for(&c, true);
    assert!(
        (author - 0.0625).abs() < 1e-9,
        "1/sqrt(256) == 1/16, got {author}"
    );
    let default = mla_scale_for(&c, false);
    assert!(
        (default - (c.kv_lora_rank as f32).powf(-0.5)).abs() < 1e-9,
        "the off arm is unchanged: kv_lora_rank^-0.5"
    );
}

/// 2026-10-01: `METRALE_GLM_DSA_MLA_HEADGROUP` selects a head group only for 2, 4 or 8; unset,
/// 0 and any other value keep the per-head kernel.
#[test]
fn headgroup_lever_parses_only_two_four_eight() {
    assert_eq!(parse_mla_headgroup(None), 0);
    for (v, g) in [("2", 2), ("4", 4), ("8", 8), (" 4 ", 4)] {
        assert_eq!(parse_mla_headgroup(Some(v)), g, "{v:?}");
    }
    for v in ["", "0", "1", "3", "16", "32", "-2", "two", "4x"] {
        assert_eq!(parse_mla_headgroup(Some(v)), 0, "{v:?} must keep the per-head kernel");
    }
}

/// 2026-10-01: A requested head group runs only when it divides the per-rank head count and its
/// entry point resolved; otherwise the launch falls back to the per-head kernel (0).
#[test]
fn headgroup_falls_back_unless_it_divides_the_heads_and_resolved() {
    assert_eq!(headgroup_for(0, 32, true), 0, "off stays off");
    assert_eq!(headgroup_for(0, 32, false), 0);
    for g in DSA_MLA_HEADGROUPS {
        assert_eq!(headgroup_for(g, 32, true), g, "32 heads per rank take G = {g}");
        assert_eq!(headgroup_for(g, 64, true), g);
        assert_eq!(headgroup_for(g, 32, false), 0, "an unresolved G = {g} falls back");
    }
    assert_eq!(headgroup_for(4, 6, true), 0, "4 does not divide 6");
    assert_eq!(headgroup_for(8, 12, true), 0, "8 does not divide 12");
    assert_eq!(headgroup_for(2, 6, true), 2);
}

/// 2026-10-01: Inputs the tensor-core prefill kernel takes: one 16-byte-aligned pool as K and V,
/// one scale, aligned Q and O.
fn tc_inputs() -> DsaDecodeInputs {
    let pool = DevicePtr(0x10_0000);
    DsaDecodeInputs {
        q: DevicePtr(0x20_0000),
        k_cache: pool,
        v_cache: pool,
        out: DevicePtr(0x30_0000),
        block_tables: DevicePtr(0x40_0000),
        seq_lens: DevicePtr(0x50_0000),
        sel_indices: DevicePtr(0x60_0000),
        k_scale: 0.0173,
        v_scale: 0.0173,
    }
}

/// 2026-10-01: Whether a resolved tensor-core kernel refuses this launch.
fn tc_refused(
    c: &Glm5NextDsaConfig,
    g: &DsaSelectGeometry,
    p: &DsaDecodePaging,
    i: &DsaDecodeInputs,
) -> bool {
    super::prefill_tc::prefill_tc_refusal(true, c, g, p, i).is_some()
}

/// 2026-10-01: The GLM-5.3 TP2 prefill launch (32 heads per rank, selection 2051) and the TP1 one
/// (64 heads) take the tensor-core kernel once it resolved.
#[test]
fn prefill_tc_takes_the_glm_shapes() {
    use super::prefill_tc::prefill_tc_refusal;
    let (c, g, i) = (
        cfg(),
        DsaSelectGeometry::plan(&cfg(), 8_192, 1).unwrap(),
        tc_inputs(),
    );
    assert!(g.out_width <= MLA_PREFILL_TC_MAX_SEL, "width {}", g.out_width);
    for heads in [32, 64] {
        let p = DsaDecodePaging {
            num_q_heads: heads,
            ..paging()
        };
        assert_eq!(prefill_tc_refusal(true, &c, &g, &p, &i), None, "{heads} heads");
    }
    let unresolved = prefill_tc_refusal(false, &c, &g, &paging(), &i).unwrap();
    assert!(unresolved.contains("did not resolve"), "{unresolved}");
}

/// 2026-10-01: Every shape or layout the tensor-core kernel does not handle is refused, so the
/// launch falls back to the decode kernel instead of reading past a buffer.
#[test]
fn prefill_tc_refuses_what_the_kernel_does_not_handle() {
    let (c, g, i, p) = (
        cfg(),
        DsaSelectGeometry::plan(&cfg(), 8_192, 1).unwrap(),
        tc_inputs(),
        paging(),
    );
    for heads in [16, 48, 6] {
        let p = DsaDecodePaging {
            num_q_heads: heads,
            ..paging()
        };
        assert!(tc_refused(&c, &g, &p, &i), "{heads} heads is not a multiple of 32");
    }
    let kv8 = DsaDecodePaging {
        num_kv_heads: 8,
        ..paging()
    };
    assert!(tc_refused(&c, &g, &kv8, &i));
    let mut wide = g;
    wide.out_width = MLA_PREFILL_TC_MAX_SEL + 1;
    assert!(tc_refused(&c, &wide, &p, &i));
    let mut kvl = cfg();
    kvl.kv_lora_rank = 256;
    assert!(tc_refused(&kvl, &g, &p, &i));
    let distinct_v = DsaDecodeInputs {
        v_cache: DevicePtr(0x70_0000),
        ..i
    };
    assert!(tc_refused(&c, &g, &p, &distinct_v));
    let distinct_scale = DsaDecodeInputs {
        v_scale: 0.0291,
        ..i
    };
    assert!(tc_refused(&c, &g, &p, &distinct_scale));
    let misaligned = DsaDecodeInputs {
        k_cache: i.k_cache.offset(1),
        v_cache: i.k_cache.offset(1),
        ..i
    };
    assert!(tc_refused(&c, &g, &p, &misaligned));
    let odd_stride = DsaDecodePaging {
        cache_stride_bytes: 64 * 512 + 8,
        ..paging()
    };
    assert!(tc_refused(&c, &g, &odd_stride, &i));
}

/// 2026-10-01: The tensor-core module and entry names are the ones the `.cu` file defines (the
/// module is its file stem), and the shared-memory mirror covers the kernel's layout.
#[test]
fn prefill_tc_names_and_mirrors_are_consistent() {
    assert_eq!(MLA_PREFILL_TC_MODULE, "glm5next_dsa_mla_prefill_tc");
    assert!(MLA_PREFILL_TC_ENTRY.starts_with(MLA_PREFILL_TC_MODULE));
    // 2026-10-01: Q and K tiles (32 x 512 BF16 each), the four FP32 S partials and the BF16 P
    // tile (row stride 40), the compacted index list, alpha, 1/l and the warp counts.
    let layout = 32 * 512 * 2 * 2
        + 4 * 32 * 40 * 4
        + 32 * 40 * 2
        + MLA_PREFILL_TC_MAX_SEL * 4
        + 32 * 4 * 2
        + 8 * 4;
    assert_eq!(MLA_PREFILL_TC_SMEM_BYTES as usize, layout);
    // 2026-10-01: GB10's shared memory per SM (metrale_core::device::sm121::SMEM_PER_SM).
    const _: () = assert!(MLA_PREFILL_TC_SMEM_BYTES <= 101_376);
    const _: () = assert!(MLA_PREFILL_TC_HEADS == 32);
}

/// 2026-10-08: `METRALE_GLM_MLA_PREFILL_TC2` takes a launch only where the tensor-core kernel
/// would (lever on, launch not refused) and its own entry point resolved; every other case is
/// ignored with a reason naming the cause.
#[test]
fn prefill_tc2_replaces_only_tensor_core_launches() {
    use super::prefill_tc::prefill_tc2_ignored;
    assert_eq!(prefill_tc2_ignored(true, None, None, true), None);
    let off = prefill_tc2_ignored(false, None, None, true).unwrap();
    assert!(off.contains("METRALE_GLM_MLA_PREFILL_TC is off"), "{off}");
    // 2026-10-08: TC off wins over everything else: nothing the TC2 kernel could replace.
    assert_eq!(
        prefill_tc2_ignored(false, Some("x"), Some("y"), false),
        Some(off)
    );
    let refused =
        prefill_tc2_ignored(true, Some("selection width 4000 > 2560"), None, true).unwrap();
    assert!(
        refused.contains(MLA_PREFILL_TC_ENTRY) && refused.contains("selection width 4000"),
        "{refused}"
    );
    let unresolved = prefill_tc2_ignored(true, None, None, false).unwrap();
    assert!(
        unresolved.contains(MLA_PREFILL_TC2_ENTRY) && unresolved.contains("did not resolve"),
        "{unresolved}"
    );
    // 2026-10-09: A6: a launch the tensor-core kernel takes but the rewrite refuses goes to the
    // tensor-core kernel, with the rewrite named as the refuser.
    let own = prefill_tc2_ignored(true, None, Some("page stride"), true).unwrap();
    assert!(
        own.contains(MLA_PREFILL_TC2_ENTRY) && own.contains("page stride"),
        "{own}"
    );
}

/// 2026-10-09: A6 addresses keys as cache rows x 512 bytes, so it takes only pages of
/// `block_size` contiguous rows; a padded page (16-byte aligned, so the tensor-core kernel takes
/// it) is refused by the rewrite only.
#[test]
fn prefill_tc2_refuses_a_page_stride_that_is_not_block_size_rows() {
    use super::prefill_tc::prefill_tc2_refusal;
    assert_eq!(prefill_tc2_refusal(&paging()), None);
    let padded = DsaDecodePaging {
        cache_stride_bytes: 64 * 512 + 16,
        ..paging()
    };
    let why = prefill_tc2_refusal(&padded).unwrap();
    assert!(why.contains("page stride 32784"), "{why}");
    let g = DsaSelectGeometry::plan(&cfg(), 8_192, 1).unwrap();
    assert!(!tc_refused(&cfg(), &g, &padded, &tc_inputs()));
}

/// 2026-10-08: The rewrite launches with the tensor-core kernel's geometry: same heads per block
/// and the same shared-memory size (the Q tile's 32 KB became the second key buffer), and its
/// entry names carry the module name (the `.cu` file stem).
#[test]
fn prefill_tc2_names_and_mirrors_match_the_tensor_core_kernel() {
    assert_eq!(MLA_PREFILL_TC2_MODULE, "glm5next_dsa_mla_prefill_tc2");
    for e in [
        MLA_PREFILL_TC2_ENTRY,
        MLA_PREFILL_TC2_HWCVT_ENTRY,
        MLA_PREFILL_TC2_CVT_CHECK_ENTRY,
        MLA_PREFILL_TC2_L2_FLUSH_ENTRY,
    ] {
        assert!(e.starts_with(MLA_PREFILL_TC2_MODULE), "{e}");
    }
    assert_eq!(MLA_PREFILL_TC2_HEADS, MLA_PREFILL_TC_HEADS);
    assert_eq!(MLA_PREFILL_TC2_SMEM_BYTES, MLA_PREFILL_TC_SMEM_BYTES);
    assert!(prefill_tc::MLA_PREFILL_TC2_ENGAGED_LINE.starts_with("METRALE_GLM_MLA_PREFILL_TC2"));
}

/// 2026-10-05: `METRALE_GLM_DSA_MLA_SPLIT`: `1` is auto, an integer of 2 or more forces S (at
/// most 16), anything else is off.
#[test]
fn split_lever_parses_auto_forced_and_off() {
    assert_eq!(split::parse_mla_split(None), MlaSplit::Off);
    for v in ["", "0", " 0 ", "auto", "-3", "2x"] {
        assert_eq!(split::parse_mla_split(Some(v)), MlaSplit::Off, "{v:?}");
    }
    assert_eq!(split::parse_mla_split(Some("1")), MlaSplit::Auto);
    assert_eq!(split::parse_mla_split(Some(" 1 ")), MlaSplit::Auto);
    for (v, s) in [("2", 2), ("8", 8), ("16", 16), ("40", DSA_MLA_SPLIT_MAX)] {
        assert_eq!(split::parse_mla_split(Some(v)), MlaSplit::Fixed(s), "{v:?}");
    }
}

/// 2026-10-05: `split::split_count` on 48 SMs.
fn split_s(mode: MlaSplit, rows: usize, heads: usize, width: usize, cap: usize) -> usize {
    split::split_count(mode, rows, heads, width, 48, cap)
}

/// 2026-10-05: A scratch for 32 heads, in partials.
const CAP32: usize = DSA_MLA_SPLIT_MAX_ROWS * 32 * DSA_MLA_SPLIT_MAX;

/// 2026-10-05: The auto rule at GLM-5.3 TP2 shapes (32 heads, width 2051, 48 SMs), forced S,
/// and the caps (16, the scratch, 64 slots per block).
#[test]
fn split_count_follows_the_documented_rule() {
    let auto = MlaSplit::Auto;
    assert_eq!(split_s(auto, 1, 32, 2051, CAP32), 16);
    assert_eq!(split_s(auto, 3, 32, 2051, CAP32), 8);
    assert_eq!(split_s(auto, 12, 32, 2051, CAP32), 2);
    assert_eq!(split_s(auto, 16, 32, 2051, CAP32), 2);
    assert_eq!(split_s(auto, 3, 32, 600, CAP32), 8);
    // 2026-10-05: Fewer than two 64-slot blocks stays unsplit.
    assert_eq!(split_s(auto, 3, 32, 127, CAP32), 1);
    assert_eq!(split_s(auto, 3, 32, 128, CAP32), 2);
    assert_eq!(split_s(auto, 3, 32, 1, CAP32), 1);
    assert_eq!(split_s(MlaSplit::Fixed(5), 3, 32, 2051, CAP32), 5);
    assert_eq!(split_s(MlaSplit::Fixed(16), 16, 32, 2051, CAP32), 16);
    // 2026-10-05: A 32-head scratch holds 8 splits of 16 rows x 64 heads.
    assert_eq!(split_s(MlaSplit::Fixed(16), 16, 64, 2051, CAP32), 8);
    assert_eq!(split_s(MlaSplit::Off, 3, 32, 2051, CAP32), 1);
    assert_eq!(split_s(auto, 3, 32, 2051, 0), 1);
}

/// 2026-10-05: The documented scratch size: 16 rows x heads x 16 splits x (512 + 2) FP32,
/// 16.8 MB at 32 heads per rank.
#[test]
fn split_scratch_is_sixteen_rows_by_heads_by_sixteen_splits() {
    assert_eq!(DSA_MLA_SPLIT_PARTIAL_FLOATS, 514);
    assert_eq!(split_scratch_bytes(32), 16_842_752);
    assert_eq!(split_scratch_bytes(64), 33_685_504);
    assert_eq!(DSA_MLA_SPLIT_ENTRY, "glm5next_dsa_mla_decode_fp8_hg8_split");
    assert_eq!(DSA_MLA_SPLIT_HEADGROUP, 8);
}
