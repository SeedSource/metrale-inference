// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Accuracy and timing gate for the split-key DSA MLA decode
//! (`glm5next_dsa_mla_decode_fp8_hg8_split` + `glm5next_dsa_mla_split_merge`,
//! `METRALE_GLM_DSA_MLA_SPLIT`) against the head-grouped `glm5next_dsa_mla_decode_fp8_hg8`,
//! launched through `attend::decode_attention_split` and `attend::decode_attention_headgroup`.
//!
//! Inputs (`common/dsa_mla_split_fixture.rs`), as `dsa_mla_headgroup_bitparity_microtest`: 32
//! heads, random BF16 q, a random FP8 pool (no NaN codes) with a non-power-of-two k_scale and
//! random paged block tables. Rows {1, 3, 12, 16} x selection widths {1, 7, 64, 600, 2048}
//! with `-1` holes, other negative and past-`seq_len` indices and duplicates; when there are 3
//! or more rows, row 1 has `seq_len` 0
//! and must stay untouched and row 2 has no valid index and must come out zero. Plus the
//! production shape (width 2051, a 600-token valid prefix then `-1`), a 1-byte-offset pool
//! (byte loads) and a distinct V buffer and scale. Each case runs the split pair at S = 1, 2,
//! 5 and 16 (S = 1 still interleaves the keys over the block's warps; 16 leaves most blocks
//! empty at the small widths).
//!
//! Pass, per (row, head) of a written row: cosine >= 0.99999 and max |split - hg8| <= 2 BF16
//! ULP of the head's output magnitude (ULP = 2^(floor(log2(max |hg8|)) - 7)). Why 2: both
//! kernels accumulate in FP32 and differ only in summation order and in the split's extra
//! `__expf` rescale, an FP32 difference of order n * 2^-24 * max |v| (about 1e-4 relative at
//! n = 2048, worst case), far below one BF16 ULP (2^-8 relative); each side then rounds to BF16
//! once (at most 0.5 ULP of the element, so at most 0.5 ULP of the magnitude). 1 ULP plus the
//! FP32 term covers it; 2 leaves margin without admitting a wrong key or weight. The all-invalid
//! row must be +0.0 bits in both arms and the `seq_len` 0 row must keep each arm's own poison,
//! bit for bit. A negative control (one head's q negated) must fail the bound, and a run that
//! compared nothing fails.
//!
//! Timing: CUDA events around each launch (the split arm times split + merge), 5 warm-up
//! launches, median of 60, at rows 3 and 12 for width 600 and 2048 (all valid) and the 2051 /
//! 600-prefix production shape, split at the auto S (`attend::split_count`).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 on PASS, 1 on any failure, a vacuous comparison or a negative control that does not
//! fire, 2 when a kernel is absent from this target. The last line is `RESULT: ...`.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_mla_split_microtest

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::attend::{
    DsaDecodeInputs, Glm5NextDsaDecodeKernel, MlaSplit, decode_attention_headgroup,
    decode_attention_split, split_count,
};

#[path = "common/dsa_mla_split_fixture.rs"]
mod dsa_mla_split_fixture;
use dsa_mla_split_fixture::*;

// 2026-10-05: CUDA driver event API for kernel-only timing, declared as in
// `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const ROWS: &[usize] = &[1, 3, 12, 16];
const WIDTHS: &[usize] = &[1, 7, 64, 600, 2048];
const SPLITS: &[usize] = &[1, 2, 5, 16];
const POISON_REF: u8 = 0xA5;
const POISON_NEW: u8 = 0x5A;
const COS_MIN: f64 = 0.99999;
const MAX_ULP: f32 = 2.0;
const WARMUP: usize = 5;
const TIMED: usize = 60;

/// 2026-10-05: One launch: `s` = 0 is the `_hg8` reference, else the split pair at S = `s`.
fn launch(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    inputs: &DsaDecodeInputs,
    s: usize,
) -> Result<()> {
    let (c, gm, pg) = (cfg(), geom(case.rows, case.width), paging(case.rows));
    if s == 0 {
        decode_attention_headgroup(g, kernel, 8, &c, &gm, &pg, inputs, 0)
    } else {
        decode_attention_split(g, kernel, s, &c, &gm, &pg, inputs, 0)
    }
}

/// 2026-10-05: One arm with q from `q` and the output poisoned with `poison`.
fn run(
    g: &dyn GpuBackend,
    kernel: Glm5NextDsaDecodeKernel,
    case: &Case,
    q: DevicePtr,
    s: usize,
    poison: u8,
) -> Result<Vec<u8>> {
    let bytes = case.rows * HEADS * KVL * 2;
    g.memset(case.inputs.out, poison, bytes)?;
    launch(g, kernel, case, &DsaDecodeInputs { q, ..case.inputs }, s)?;
    down(g, case.inputs.out, bytes)
}

/// 2026-10-05: BF16 ULP at magnitude `m`: 2^(floor(log2 m) - 7); 0 for 0.
fn bf16_ulp(m: f32) -> f32 {
    if m <= 0.0 {
        return 0.0;
    }
    let e = ((m.to_bits() >> 23) & 0xFF) as i32 - 127;
    2f32.powi(e - 7)
}

#[derive(Default)]
struct Stats {
    compared: usize,
    worst_cos: f64,
    worst_ulp: f32,
}

/// 2026-10-05: The pass rule of the module header for reference `r` and split output `n`;
/// `st` collects the compared elements and the worst cosine and error.
fn compare(case: &Case, r: &[u8], n: &[u8], st: &mut Stats) -> bool {
    let row_bytes = HEADS * KVL * 2;
    let mut ok = true;
    for (row, &sl) in case.seq_lens.iter().enumerate() {
        let (a, b) = (
            &r[row * row_bytes..(row + 1) * row_bytes],
            &n[row * row_bytes..(row + 1) * row_bytes],
        );
        if sl == 0 {
            ok &= a.iter().all(|&x| x == POISON_REF) && b.iter().all(|&x| x == POISON_NEW);
            continue;
        }
        st.compared += HEADS * KVL;
        if case.invalid_row == Some(row) {
            ok &= a.iter().all(|&x| x == 0) && b.iter().all(|&x| x == 0);
            continue;
        }
        ok &= !b.iter().all(|&x| x == POISON_NEW);
        let val = |buf: &[u8], i: usize| bf16::from_le_bytes([buf[2 * i], buf[2 * i + 1]]).to_f32();
        for h in 0..HEADS {
            let (mut dot, mut nx, mut ny, mut err, mut mag) = (0f64, 0f64, 0f64, 0f32, 0f32);
            for i in h * KVL..(h + 1) * KVL {
                let (x, y) = (val(a, i), val(b, i));
                dot += x as f64 * y as f64;
                nx += x as f64 * x as f64;
                ny += y as f64 * y as f64;
                err = err.max((x - y).abs());
                mag = mag.max(x.abs());
            }
            let cos = if nx == 0.0 && ny == 0.0 {
                1.0
            } else {
                dot / (nx.sqrt() * ny.sqrt())
            };
            let ulp = bf16_ulp(mag);
            let e_ulp = if err == 0.0 { 0.0 } else { err / ulp };
            // 2026-10-05: NaN fails both tests (cosine NaN, err/0 = inf).
            ok &= cos >= COS_MIN && e_ulp <= MAX_ULP;
            st.worst_cos = st.worst_cos.min(cos);
            st.worst_ulp = st.worst_ulp.max(e_ulp);
        }
    }
    ok
}

/// 2026-10-05: `rc` as a Result.
fn ck(rc: i32, what: &str) -> Result<()> {
    if rc != 0 {
        bail!("{what} failed: status {rc}");
    }
    Ok(())
}

/// 2026-10-05: Median microseconds of `f` over `TIMED` launches, each between two CUDA events
/// on stream 0, after `WARMUP` launches.
fn time_us(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(0)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    ck(unsafe { cuEventCreate(&mut e0, 0) }, "cuEventCreate(start)")?;
    ck(unsafe { cuEventCreate(&mut e1, 0) }, "cuEventCreate(end)")?;
    let mut us = Vec::with_capacity(TIMED);
    for _ in 0..TIMED {
        ck(unsafe { cuEventRecord(e0, 0) }, "cuEventRecord(start)")?;
        f()?;
        ck(unsafe { cuEventRecord(e1, 0) }, "cuEventRecord(end)")?;
        ck(unsafe { cuEventSynchronize(e1) }, "cuEventSynchronize")?;
        let mut ms = 0f32;
        ck(unsafe { cuEventElapsedTime(&mut ms, e0, e1) }, "cuEventElapsedTime")?;
        us.push(ms as f64 * 1e3);
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    us.sort_by(f64::total_cmp);
    Ok(us[us.len() / 2])
}

fn main() -> Result<()> {
    // 2026-10-05: The kernels are in the glm-5.3-flash target; build the backend from that set
    // (as `dsa_mla_headgroup_bitparity_microtest`).
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        println!("glm-5.3-flash kernel target not built - SKIP");
        std::process::exit(2);
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let kernel = Glm5NextDsaDecodeKernel::resolve(g).context("Glm5NextDsaDecodeKernel")?;
    if !kernel.has_headgroup(8) || !kernel.has_split_kernels() {
        println!(
            "glm5next_dsa_mla_decode_fp8_hg8 or the split pair absent from this target - SKIP"
        );
        std::process::exit(2);
    }
    let (kernel, scratch) = kernel.with_split_scratch(g, HEADS)?;
    let sms = g.sm_count()?;

    let mut plan: Vec<(usize, usize, Sel, Pool)> = Vec::new();
    for &w in WIDTHS {
        for &rows in ROWS {
            plan.push((rows, w, Sel::Mixed, Pool::Aligned));
        }
    }
    for &rows in ROWS {
        plan.push((rows, 2051, Sel::Prefix(600), Pool::Aligned));
    }
    plan.push((3, 600, Sel::Mixed, Pool::Misaligned));
    plan.push((3, 600, Sel::Mixed, Pool::DistinctV));

    let mut st = Stats {
        worst_cos: 1.0,
        ..Stats::default()
    };
    let (mut arms, mut failed) = (0usize, 0usize);
    for (i, &(rows, w, sel_kind, pool)) in plan.iter().enumerate() {
        let case = build_case(g, rows, w, sel_kind, pool, 0x5EED_1000 + i as u64)?;
        let r = run(g, kernel, &case, case.inputs.q, 0, POISON_REF)?;
        for &s in SPLITS {
            let n = run(g, kernel, &case, case.inputs.q, s, POISON_NEW)?;
            let mut one = Stats {
                worst_cos: 1.0,
                ..Stats::default()
            };
            let ok = compare(&case, &r, &n, &mut one);
            arms += 1;
            failed += usize::from(!ok);
            st.compared += one.compared;
            st.worst_cos = st.worst_cos.min(one.worst_cos);
            st.worst_ulp = st.worst_ulp.max(one.worst_ulp);
            let kind = match (sel_kind, pool) {
                (Sel::Prefix(_), _) => "prefix600",
                (_, Pool::Misaligned) => "misaligned",
                (_, Pool::DistinctV) => "distinct-V",
                _ => "mixed",
            };
            println!(
                "W={w:<5} rows={rows:<3} {kind:<10} S={s:<3} elems={:<8} min_cos={:.7} \
                 max_err={:.3} ulp pass={ok}",
                one.compared, one.worst_cos, one.worst_ulp
            );
        }
        free_case(g, &case);
    }

    // 2026-10-05: Negative control: head 5 of row 0 with q negated must fail the bound.
    let case = build_case(g, 3, 600, Sel::Mixed, Pool::Aligned, 0xC047_8002)?;
    let r = run(g, kernel, &case, case.inputs.q, 0, POISON_REF)?;
    let mut pert = case.q_bytes.clone();
    for i in 5 * KVL..6 * KVL {
        pert[2 * i + 1] ^= 0x80;
    }
    let q_pert = up(g, &pert)?;
    let n = run(g, kernel, &case, q_pert, 4, POISON_NEW)?;
    let fired = !compare(&case, &r, &n, &mut Stats::default());
    println!("CONTROL row 0 head 5 q negated: detected={fired}");
    g.free(q_pert).ok();
    free_case(g, &case);

    // 2026-10-05: Timing at the verify shapes, split at the auto S.
    let mut timing = Vec::new();
    for &rows in &[3usize, 12] {
        for (w, sel_kind, label) in [
            (600, Sel::Prefix(600), "W600"),
            (2048, Sel::Prefix(2048), "W2048"),
            (2051, Sel::Prefix(600), "W2051/600"),
        ] {
            let case = build_case(g, rows, w, sel_kind, Pool::Aligned, 0x7135 + rows as u64)?;
            let s = split_count(MlaSplit::Auto, rows, HEADS, w, sms, usize::MAX);
            let hg8 = time_us(g, || launch(g, kernel, &case, &case.inputs, 0))?;
            let split = time_us(g, || launch(g, kernel, &case, &case.inputs, s))?;
            println!(
                "TIMING rows={rows:<3} {label:<10} hg8 {hg8:>8.1} us  split(S={s}) {split:>8.1} \
                 us  x{:.2}",
                hg8 / split
            );
            timing.push(format!("r{rows}/{label} hg8={hg8:.1}us split(S={s})={split:.1}us"));
            free_case(g, &case);
        }
    }
    g.free(scratch).ok();

    let verdict = if st.compared == 0 {
        println!("FAIL - no element was compared; this run proves nothing.");
        "FAIL"
    } else if !fired {
        println!("FAIL - the negative control did not fire; this harness is VACUOUS.");
        "FAIL"
    } else if failed > 0 {
        println!(
            "FAIL - {failed} of {arms} arm(s) outside cos >= {COS_MIN} / err <= {MAX_ULP} BF16 \
             ULP or the zero / no-write contract. Keep METRALE_GLM_DSA_MLA_SPLIT off."
        );
        "FAIL"
    } else {
        println!(
            "PASS: {arms} split arm(s) within cos >= {COS_MIN} / err <= {MAX_ULP} BF16 ULP of hg8; \
             zero / no-write contract held"
        );
        "PASS"
    };
    println!(
        "RESULT: dsa_mla_split {verdict} arms={arms} failed={failed} elems={} min_cos={:.7} \
         max_err_ulp={:.3} control={fired} sms={sms} | {}",
        st.compared,
        st.worst_cos,
        st.worst_ulp,
        timing.join(" | ")
    );
    if verdict != "PASS" {
        std::process::exit(1);
    }
    Ok(())
}
