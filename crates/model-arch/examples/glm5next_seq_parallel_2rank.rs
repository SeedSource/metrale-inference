// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Two-rank GPU gate (and timing) for the communication of
//! `METRALE_GLM_PREFILL_SEQ_PARALLEL=1`: the production reduce-scatter
//! (`glm5next_layer::seq_parallel::reduce_item`) and owned-row exchange (`swap_owned`) against
//! today's all-reduce (`all_reduce_async` on the two-rank send/recv + `bf16_add_inplace` path),
//! on a real `NcclBackend`, over the item sequence one staged layer issues: every attention
//! sub-chunk, then every FFN window (`sub_chunks`, `ffn_windows`; the default 5400 tokens at
//! 256 / 2048 has a window straddling the ownership boundary, so both directions run).
//!
//! Each rank holds a seeded BF16 partial for all `tokens` rows of `[tokens, 4096]`. Per item
//! both arms first copy the item's partial rows into a work buffer (where a mixer or the MLP
//! leaves them). Arm A all-reduces the work buffer and copies it to its output (today). Arm B
//! runs `reduce_item` into a poisoned output. Then arm A's output with the peer's rows
//! poisoned goes through `swap_owned`.
//!
//! Gates (PASS needs all, on each rank): arm B equals arm A on every byte of this rank's rows
//! and leaves the peer's rows untouched; the swapped buffer equals arm A on every row; a
//! negative control (rank 1 flips the sign of one element of a row rank 0 owns, arm B again)
//! changes exactly that element on rank 0 and nothing on rank 1, so rank 0's sum provably
//! includes the peer's data; the backend reports the send/recv + add all-reduce
//! (`all_reduce_is_send_recv_add`), the path the lever requires. The mHC / norm launches the
//! lever moves are row-local (`glm_hc_pre_sliced` gate and the per-token grids); the full
//! layer is gated by the serve identity run. Timing (CUDA events, median of `reps`) is
//! informational.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS`; 1 and `FAIL`; 2 when `bf16_add_inplace` is absent from this target.
//!
//! Run, one process per node, same arguments except `--rank` (rank 0 listens on `--port`):
//!   cargo run -p metrale-model-arch --release --features cuda,nccl-examples \
//!     --example glm5next_seq_parallel_2rank -- --rank 0 --peer <rank0-ip> --port 29563
//! (or `scripts/race/run-2rank-example.sh` in spark-bench). Check rank 0's NCCL log for
//! `NET/IB` before trusting the timing (`NET/Socket` = STOP).

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_comm::{CommBackend, NcclBackend};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_layer::seq_parallel::{SpPlan, reduce_item, swap_owned};
use metrale_model_arch::glm5next_layer::{ffn_windows, sub_chunks};

// 2026-10-01: CUDA driver event API for timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// 2026-10-01: GLM-5.3 hidden size.
const H: usize = 4096;
const POISON: u8 = 0xA5;

struct Args {
    rank: usize,
    peer: String,
    port: u16,
    tokens: usize,
    rows: usize,
    ffn: usize,
    reps: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        rank: usize::MAX,
        peer: String::new(),
        port: 29563,
        tokens: 5400,
        rows: 256,
        ffn: 2048,
        reps: 3,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let v = argv
            .get(i + 1)
            .with_context(|| format!("{} needs a value", argv[i]))?;
        match argv[i].as_str() {
            "--rank" => a.rank = v.parse()?,
            "--peer" => a.peer = v.clone(),
            "--port" => a.port = v.parse()?,
            "--tokens" => a.tokens = v.parse()?,
            "--rows" => a.rows = v.parse()?,
            "--ffn" => a.ffn = v.parse()?,
            "--reps" => a.reps = v.parse()?,
            other => bail!("unknown argument {other}"),
        }
        i += 2;
    }
    if a.rank > 1 || a.peer.is_empty() || a.rows == 0 || a.tokens <= a.rows || a.reps == 0 {
        bail!(
            "usage: --rank 0|1 --peer <rank0 address> [--port P] [--tokens 5400 (> rows)] \
             [--rows 256] [--ffn 2048] [--reps 3]"
        );
    }
    Ok(a)
}

/// 2026-10-01: `rank`'s seeded BF16 partial, `n` elements in [-2, 2).
fn partial(rank: usize, n: usize) -> Vec<u8> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15 ^ ((rank as u64) << 40);
    (0..n)
        .flat_map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            bf16::from_f64(4.0 * u - 2.0).to_bits().to_le_bytes()
        })
        .collect()
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn timed(g: &dyn GpuBackend, stream: u64, f: impl Fn() -> Result<()>) -> Result<f32> {
    let (mut e0, mut e1) = (0u64, 0u64);
    let mut ms = 0f32;
    // 2026-10-01: SAFETY: plain driver calls on events this function creates and destroys,
    // on the caller's live stream.
    unsafe {
        if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
            bail!("cuEventCreate failed");
        }
        cuEventRecord(e0, stream);
    }
    f()?;
    // 2026-10-01: SAFETY: the two events created above, recorded on the same stream.
    unsafe {
        cuEventRecord(e1, stream);
        cuEventSynchronize(e1);
        cuEventElapsedTime(&mut ms, e0, e1);
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    g.synchronize(stream)?;
    Ok(ms)
}

/// 2026-10-01: Byte offsets in `a` and `b` (same length) where they differ.
fn diffs(a: &[u8], b: &[u8], range: std::ops::Range<usize>) -> Vec<usize> {
    range.filter(|&i| a[i] != b[i]).collect()
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let add = match g.kernel("bf16_add", "bf16_add_inplace") {
        Ok(k) => k,
        Err(e) => {
            println!("bf16_add::bf16_add_inplace absent from this target ({e}) - SKIP");
            std::process::exit(2);
        }
    };
    let stream = g.create_stream()?;
    let rb = H * 2;
    let total = a.tokens * rb;
    let comm = NcclBackend::new(a.rank, 2, &a.peer, a.port, stream, total)?;
    comm.set_add_kernel(add.0);
    let send_recv_add = comm.all_reduce_is_send_recv_add();

    let subs = sub_chunks(a.tokens, a.rows);
    let items: Vec<(usize, usize)> = subs
        .iter()
        .copied()
        .chain(ffn_windows(&subs, a.ffn.max(a.rows), false, |_| true))
        .collect();
    let plan = SpPlan::new(&subs).context("needs at least two sub-chunks")?;
    let (own, peer) = (plan.owned(a.rank), plan.owned(1 - a.rank));
    let own_bytes = own.0 * rb..(own.0 + own.1) * rb;

    // 2026-10-01: Control: rank 1 flips the sign of element 7 of the middle row rank 0 owns.
    let ctrl_row = plan.owned(0).1 / 2;
    let ctrl_elem = ctrl_row * rb + 7 * 2;
    let ctrl_byte = ctrl_elem + 1;
    let mine = partial(a.rank, a.tokens * H);
    let mut ctrl = mine.clone();
    if a.rank == 1 {
        ctrl[ctrl_byte] ^= 0x80;
    }
    let up = |bytes: &[u8]| -> Result<DevicePtr> {
        let p = g.alloc(bytes.len())?;
        g.copy_h2d(bytes, p)?;
        Ok(p)
    };
    let (src, src_ctrl) = (up(&mine)?, up(&ctrl)?);
    let work = g.alloc(items.iter().map(|i| i.1).max().unwrap_or(1) * rb)?;
    let (out_a, out_b) = (g.alloc(total)?, g.alloc(total)?);
    let (out_c, swapped) = (g.alloc(total)?, g.alloc(total)?);

    let arm_a = |out: DevicePtr| -> Result<()> {
        for &(t, k) in &items {
            g.copy_d2d_async(src.offset(t * rb), work, k * rb, stream)?;
            comm.all_reduce_async(work.0, k * rb, stream)?;
            g.copy_d2d_async(work, out.offset(t * rb), k * rb, stream)?;
        }
        Ok(())
    };
    let arm_b = |from: DevicePtr, out: DevicePtr| -> Result<()> {
        for &(t, k) in &items {
            g.copy_d2d_async(from.offset(t * rb), work, k * rb, stream)?;
            reduce_item(
                g,
                add,
                &comm,
                plan,
                work,
                out,
                (t, k),
                H,
                true,
                stream,
            )?;
        }
        Ok(())
    };

    g.memset(out_a, POISON, total)?;
    g.memset(out_b, POISON, total)?;
    g.memset(out_c, POISON, total)?;
    timed(g, stream, || arm_a(out_a))?;
    timed(g, stream, || arm_b(src, out_b))?;
    timed(g, stream, || arm_b(src_ctrl, out_c))?;
    let (ha, hb, hc) = (down(g, out_a, total)?, down(g, out_b, total)?, down(g, out_c, total)?);

    // 2026-10-01: The swap: arm A's own rows, the peer's rows poisoned.
    g.memset(swapped, POISON, total)?;
    let (from, to) = (out_a.offset(own.0 * rb), swapped.offset(own.0 * rb));
    g.copy_d2d_async(from, to, own.1 * rb, stream)?;
    timed(g, stream, || swap_owned(&comm, plan, swapped, rb, stream))?;
    let hs = down(g, swapped, total)?;

    let ab_own = diffs(&ha, &hb, own_bytes.clone()).len();
    let b_peer_written = (peer.0 * rb..(peer.0 + peer.1) * rb)
        .filter(|&i| hb[i] != POISON)
        .count();
    let swap_diff = diffs(&ha, &hs, 0..total).len();
    let c_diff = diffs(&ha, &hc, own_bytes);
    let control_fired = if a.rank == 0 {
        !c_diff.is_empty() && c_diff.iter().all(|&i| i == ctrl_elem || i == ctrl_byte)
    } else {
        c_diff.is_empty()
    };

    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for _ in 0..a.reps {
        ta.push(timed(g, stream, || arm_a(out_a))?);
        tb.push(timed(g, stream, || arm_b(src, out_b))?);
    }
    let (ma, mb) = (median(ta), median(tb));

    println!(
        "rank {} tokens {} rows {} ffn {}: {} items, split at row {}, own rows {:?}; \
         send/recv+add all-reduce: {send_recv_add}",
        a.rank,
        a.tokens,
        a.rows,
        a.ffn,
        items.len(),
        plan.split,
        own
    );
    println!(
        "timing (median of {}): all-reduce {ma:.3} ms | reduce-scatter {mb:.3} ms per layer's \
         items (copies included in both)",
        a.reps
    );
    println!(
        "own bytes {}: A vs B differ {ab_own}; B wrote {b_peer_written} peer bytes; swap vs A \
         differ {swap_diff} of {total}; control differs at {c_diff:?} (fired: {control_fired})",
        own.1 * rb
    );
    let ok = send_recv_add && ab_own == 0 && b_peer_written == 0 && swap_diff == 0;
    if ok && control_fired && own.1 > 0 {
        println!("PASS");
        Ok(())
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
