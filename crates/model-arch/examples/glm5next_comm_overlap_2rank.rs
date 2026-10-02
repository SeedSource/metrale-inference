// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Two-rank GPU gate (and timing) for `METRALE_GLM_PREFILL_COMM_OVERLAP=1`: the
//! deferred all-reduce (`CommBackend::all_reduce_deferred` / `all_reduce_join`) driven by the
//! production schedule (`glm5next_layer::comm_overlap::overlap_schedule`) against today's
//! joined `all_reduce_async`, on a real `NcclBackend` over the two-node RoCE link.
//!
//! Each arm walks `items` items of `rows x 4096` BF16 (the served 256-row sub-chunk by
//! default). Per item it runs a stand-in for the mixer (`busy` passes of `bf16_add_inplace`
//! over a private 32 MiB scratch, so the compute stream has work that does not touch the
//! reduce buffers), copies the item's partial into a work slot (the production copy into the
//! `hidden` rows), all-reduces it, and copies the reduced slot to the arm's output (the
//! stand-in for `hc_post`). Arm A joins every all-reduce at once (`all_reduce_async`, the
//! default path). Arm B follows `overlap_schedule`: issue item `i + 1` before joining item `i`.
//! Partials are seeded per (rank, item), so each rank can also rebuild the expected sum.
//!
//! Gates (PASS needs all): arm B's output is bitwise identical to arm A's on every item; a
//! negative control (rank 0 flips one bit of one item's partial, arm B again) must differ from
//! arm A exactly inside that item; arm A agrees with the host oracle (round-to-nearest-even of
//! the exact sum of both ranks' partials) on at least 99.99% of elements (a sanity gate that
//! the peer's partial really arrived; `__hadd` rounding is the reference, so the oracle is not
//! the byte gate). Timing (CUDA events, median of `reps` alternating runs) is informational.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS`; 1 and `FAIL`; 2 when `bf16_add_inplace` is absent from this target.
//!
//! Run, one process per node, same arguments except `--rank` (rank 0 listens on `--port`):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
//!     cargo run -p metrale-model-arch --release --features cuda,nccl-examples \
//!     --example glm5next_comm_overlap_2rank -- --rank 0 --peer <rank0-ip> --port 29561
//! Check rank 0's NCCL log for `NET/IB` before trusting the timing (`NET/Socket` = STOP).

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_comm::{CommBackend, NcclBackend};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_layer::comm_overlap::{OverlapStep, overlap_schedule};

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
/// 2026-10-01: Busy scratch, BF16 elements (32 MiB).
const SCRATCH_ELEMS: usize = 16 << 20;
const POISON_A: u8 = 0xA5;
const POISON_B: u8 = 0x5A;

struct Args {
    rank: usize,
    peer: String,
    port: u16,
    rows: usize,
    items: usize,
    busy: usize,
    reps: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        rank: usize::MAX,
        peer: String::new(),
        port: 29561,
        rows: 256,
        items: 32,
        busy: 4,
        reps: 5,
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
            "--rows" => a.rows = v.parse()?,
            "--items" => a.items = v.parse()?,
            "--busy" => a.busy = v.parse()?,
            "--reps" => a.reps = v.parse()?,
            other => bail!("unknown argument {other}"),
        }
        i += 2;
    }
    if a.rank > 1 || a.peer.is_empty() || a.rows == 0 || a.items == 0 || a.reps == 0 {
        bail!(
            "usage: --rank 0|1 --peer <rank0 address> [--port P] [--rows 256] [--items 32] \
             [--busy 4] [--reps 5]"
        );
    }
    Ok(a)
}

/// 2026-10-01: The BF16 partial of `rank`'s item `item`, `n` elements in [-2, 2).
fn partial(rank: usize, item: usize, n: usize) -> Vec<bf16> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15 ^ ((rank as u64) << 40) ^ (item as u64);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            bf16::from_f64(4.0 * u - 2.0)
        })
        .collect()
}

fn bf16_bytes(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    comm: &'a NcclBackend,
    add: KernelHandle,
    stream: u64,
    scratch: DevicePtr,
    scratch_src: DevicePtr,
    /// 2026-10-01: Two work slots (the production `hidden` rows), each `bytes` long.
    work: [DevicePtr; 2],
    bytes: usize,
    items: usize,
    busy: usize,
}

impl Rig<'_> {
    fn add_launch(&self, dst: DevicePtr, src: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(self.g, self.add)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_i32(n as i32)
            .launch(self.stream)
    }

    /// 2026-10-01: The mixer stand-in: `busy` passes over the private scratch.
    fn busy_work(&self) -> Result<()> {
        for _ in 0..self.busy {
            self.add_launch(self.scratch, self.scratch_src, SCRATCH_ELEMS)?;
        }
        Ok(())
    }

    /// 2026-10-01: Busy work, then item `i`'s partial (at `src`) copied into work slot `w`.
    fn front(&self, src: DevicePtr, i: usize, w: usize) -> Result<()> {
        self.busy_work()?;
        self.g
            .copy_d2d_async(src.offset(i * self.bytes), self.work[w], self.bytes, self.stream)
    }

    /// 2026-10-01: Work slot `w` copied to item `i` of `out` (the `hc_post` stand-in).
    fn back(&self, out: DevicePtr, i: usize, w: usize) -> Result<()> {
        self.g
            .copy_d2d_async(self.work[w], out.offset(i * self.bytes), self.bytes, self.stream)
    }

    /// 2026-10-01: Arm A, the default path: every all-reduce joined at once.
    fn arm_joined(&self, src: DevicePtr, out: DevicePtr) -> Result<()> {
        for i in 0..self.items {
            self.front(src, i, 0)?;
            self.comm.all_reduce_async(self.work[0].0, self.bytes, self.stream)?;
            self.back(out, i, 0)?;
        }
        Ok(())
    }

    /// 2026-10-01: Arm B: the production overlap schedule, work slot = deferred slot.
    fn arm_overlapped(&self, src: DevicePtr, out: DevicePtr) -> Result<()> {
        for step in overlap_schedule(self.items) {
            match step {
                OverlapStep::Issue { item, slot } => {
                    self.front(src, item, slot)?;
                    self.comm
                        .all_reduce_deferred(self.work[slot].0, self.bytes, self.stream, slot)?;
                }
                OverlapStep::Finish { item, slot } => {
                    self.comm.all_reduce_join(self.stream, slot)?;
                    self.back(out, item, slot)?;
                }
            }
        }
        Ok(())
    }

    fn timed(&self, f: impl Fn() -> Result<()>) -> Result<f32> {
        let (mut e0, mut e1) = (0u64, 0u64);
        let mut ms = 0f32;
        // 2026-10-01: SAFETY: plain driver calls on events this function creates and destroys,
        // on the rig's live stream.
        unsafe {
            if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
                bail!("cuEventCreate failed");
            }
            cuEventRecord(e0, self.stream);
        }
        f()?;
        // 2026-10-01: SAFETY: the two events created above, recorded on the same stream.
        unsafe {
            cuEventRecord(e1, self.stream);
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        self.g.synchronize(self.stream)?;
        Ok(ms)
    }
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
    let elems = a.rows * H;
    let bytes = elems * 2;
    let comm = NcclBackend::new(a.rank, 2, &a.peer, a.port, stream, bytes)?;
    comm.set_add_kernel(add.0);

    // 2026-10-01: This rank's partials, item after item, plus the control copy (rank 0 flips
    // the sign of element 7 of the middle item, a change no rounding can absorb; rank 1's
    // control equals its partials).
    let mine: Vec<Vec<bf16>> = (0..a.items).map(|i| partial(a.rank, i, elems)).collect();
    let src_host: Vec<u8> = mine.iter().flat_map(|p| bf16_bytes(p)).collect();
    let mut ctrl_host = src_host.clone();
    let ctrl_item = a.items / 2;
    if a.rank == 0 {
        ctrl_host[ctrl_item * bytes + 7 * 2 + 1] ^= 0x80;
    }
    let total = a.items * bytes;
    let src = g.alloc(total)?;
    g.copy_h2d(&src_host, src)?;
    let ctrl = g.alloc(total)?;
    g.copy_h2d(&ctrl_host, ctrl)?;
    let out_a = g.alloc(total)?;
    let out_b = g.alloc(total)?;
    let out_c = g.alloc(total)?;
    let scratch = g.alloc(SCRATCH_ELEMS * 2)?;
    let scratch_src = g.alloc(SCRATCH_ELEMS * 2)?;
    g.memset(scratch, 0, SCRATCH_ELEMS * 2)?;
    g.memset(scratch_src, 0, SCRATCH_ELEMS * 2)?;
    let work = [g.alloc(bytes)?, g.alloc(bytes)?];
    let rig = Rig {
        g,
        comm: &comm,
        add,
        stream,
        scratch,
        scratch_src,
        work,
        bytes,
        items: a.items,
        busy: a.busy,
    };

    // 2026-10-01: Correctness runs on poisoned outputs, then the control.
    g.memset(out_a, POISON_A, total)?;
    g.memset(out_b, POISON_B, total)?;
    g.memset(out_c, POISON_B, total)?;
    rig.timed(|| rig.arm_joined(src, out_a))?;
    rig.timed(|| rig.arm_overlapped(src, out_b))?;
    rig.timed(|| rig.arm_overlapped(ctrl, out_c))?;
    let ha = down(g, out_a, total)?;
    let hb = down(g, out_b, total)?;
    let hc = down(g, out_c, total)?;

    let ab_diff = ha.iter().zip(&hb).filter(|(x, y)| x != y).count();
    let c_diff: Vec<usize> = ha
        .iter()
        .zip(&hc)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i)
        .collect();
    let control_fired = !c_diff.is_empty()
        && c_diff
            .iter()
            .all(|&i| i / bytes == ctrl_item && (i % bytes) / 2 == 7);

    // 2026-10-01: Host oracle: RNE of the exact sum of both ranks' partials.
    let peer = 1 - a.rank;
    let mut oracle_bad = 0usize;
    for (i, own) in mine.iter().enumerate() {
        let theirs = partial(peer, i, elems);
        for e in 0..elems {
            let want = bf16::from_f64(own[e].to_f64() + theirs[e].to_f64()).to_bits();
            let off = i * bytes + 2 * e;
            let got = u16::from_le_bytes([ha[off], ha[off + 1]]);
            if got != want {
                oracle_bad += 1;
            }
        }
    }
    let compared = total;
    let oracle_ok = (oracle_bad as f64) <= 1e-4 * ((a.items * elems) as f64);

    // 2026-10-01: Timing, arms alternated.
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for _ in 0..a.reps {
        ta.push(rig.timed(|| rig.arm_joined(src, out_a))?);
        tb.push(rig.timed(|| rig.arm_overlapped(src, out_b))?);
    }
    let (ma, mb) = (median(ta), median(tb));
    let busy_only = median(
        (0..a.reps)
            .map(|_| {
                rig.timed(|| {
                    for _ in 0..a.items {
                        rig.busy_work()?;
                    }
                    Ok(())
                })
            })
            .collect::<Result<Vec<f32>>>()?,
    );

    println!(
        "rank {} rows {} items {} busy {}: {} bytes per all-reduce",
        a.rank, a.rows, a.items, a.busy, bytes
    );
    println!(
        "timing (median of {}): joined {ma:.3} ms | overlapped {mb:.3} ms | saved {:.3} ms \
         ({:.1} us per item) | busy-only floor {busy_only:.3} ms",
        a.reps,
        ma - mb,
        1000.0 * (ma - mb) / a.items as f32
    );
    println!(
        "bytes compared {compared}: A vs B differ {ab_diff}; control differs {} (fired: \
         {control_fired}); oracle mismatches {oracle_bad} (ok: {oracle_ok})",
        c_diff.len()
    );
    if compared > 0 && ab_diff == 0 && control_fired && oracle_ok {
        println!("PASS");
        Ok(())
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
