// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: Two-rank GPU gate (and timing) for `METRALE_GLM_DSA_INDEX_SPLIT=1`: the
//! production split selection (`glm5next_dsa::select::split::select_tokens_split`, each rank
//! selects half the query rows and the ranks swap token rows over a real `NcclBackend`)
//! against today's replicated pass (`select_tokens` over all rows), byte for byte.
//!
//! One exact batched selection at GLM-5.3's indexer shape (32 heads x 128, kpool 4, top-k
//! 2048): `ctx` keys of seeded BF16 `k_normed` / `gate`, seeded f32 APE, `q` and head weights,
//! and the `rows` query rows at the end of the context (`q_pos = ctx - rows + r`, a prefill
//! sub-chunk's last pass). Both ranks build the same inputs from the same seed.
//!
//! Gates (PASS needs all, on each rank): the split output equals the full pass on every byte
//! of the `[rows, out_width]` token rows (half of them received from the peer); a negative
//! control (rank 1 negates the head weights of one of its own rows, then the split again)
//! differs from the full pass inside that row only, on BOTH ranks, so rank 0 proves the row
//! it compares really came over the link. Timing (CUDA events, median of `reps` alternating
//! runs) is informational. `METRALE_GLM_DSA_SCORES_TILED=1` in the environment applies to
//! both arms as in production.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 and `PASS`; 1 and `FAIL`.
//!
//! Run, one process per node, same arguments except `--rank` (rank 0 listens on `--port`):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
//!     cargo run -p metrale-model-arch --release --features cuda,nccl-examples \
//!     --example glm5next_dsa_index_split_2rank -- --rank 0 --peer <rank0-ip> --port 29562
//! (or `scripts/race/run-2rank-example.sh` in spark-bench). Check rank 0's NCCL log for
//! `NET/IB` before trusting the timing (`NET/Socket` = STOP).

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_comm::NcclBackend;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::select::split::{RowSplit, select_tokens_split};
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, select_tokens,
};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

// 2026-10-01: CUDA driver event API for timing, declared as in `dense_gemm_microtest`.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const POISON_A: u8 = 0xA5;
const POISON_B: u8 = 0x5A;

struct Args {
    rank: usize,
    peer: String,
    port: u16,
    rows: usize,
    ctx: usize,
    reps: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        rank: usize::MAX,
        peer: String::new(),
        port: 29562,
        rows: 256,
        ctx: 8192,
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
            "--ctx" => a.ctx = v.parse()?,
            "--reps" => a.reps = v.parse()?,
            other => bail!("unknown argument {other}"),
        }
        i += 2;
    }
    if a.rank > 1 || a.peer.is_empty() || a.rows < 2 || a.ctx < a.rows || a.reps == 0 {
        bail!(
            "usage: --rank 0|1 --peer <rank0 address> [--port P] [--rows 256 (>= 2)] \
             [--ctx 8192 (>= rows)] [--reps 5]"
        );
    }
    Ok(a)
}

/// 2026-10-01: GLM-5.3's DSA indexer shape (`max_context` only bounds the planner).
fn cfg(ctx: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 32,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: ctx,
    }
}

/// 2026-10-01: `n` seeded values in `[-scale, scale)`.
fn seeded(seed: u64, n: usize, scale: f64) -> Vec<f64> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            scale * (2.0 * u - 1.0)
        })
        .collect()
}

fn f32_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect()
}

fn bf16_bytes(v: &[f64]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| bf16::from_f64(*x).to_bits().to_le_bytes())
        .collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
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

fn main() -> Result<()> {
    let a = parse_args()?;
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kernels = Glm5NextDsaKernels::resolve(g)?;
    let cfg = cfg(a.ctx);
    let (heads, d, kp) = (cfg.index_heads, cfg.index_head_dim, cfg.index_kpool);
    let geom = DsaSelectGeometry::plan(&cfg, a.ctx, a.rows)?;
    let scratch = DsaSelectScratch::alloc(g, &cfg, &geom)?;
    let row_bytes = cfg.out_width() * 4;
    let total = a.rows * row_bytes;
    let stream = g.create_stream()?;
    let comm = NcclBackend::new(a.rank, 2, &a.peer, a.port, stream, total)?;
    let split = RowSplit::new(a.rows, a.rank).context("split needs >= 2 rows")?;

    // 2026-10-01: The same seeded inputs on both ranks.
    let k_normed = up(g, &bf16_bytes(&seeded(1, a.ctx * d, 2.0)))?;
    let gate = up(g, &bf16_bytes(&seeded(2, a.ctx * d, 1.0)))?;
    let valid = up(g, &vec![1u8; a.ctx])?;
    let ape = up(g, &f32_bytes(&seeded(3, kp * d, 0.25)))?;
    let q = up(g, &f32_bytes(&seeded(4, a.rows * heads * d, 1.0)))?;
    let w_host = seeded(5, a.rows * heads, (heads as f64).powf(-0.5));
    let weights = up(g, &f32_bytes(&w_host))?;
    let q_pos_host: Vec<u8> = (0..a.rows)
        .flat_map(|r| ((a.ctx - a.rows + r) as i32).to_le_bytes())
        .collect();
    let q_pos = up(g, &q_pos_host)?;
    let q_mask = up(g, &vec![1u8; a.rows])?;
    // 2026-10-01: Control weights: rank 1 negates the head weights of its own row 3 (or its
    // last row); rank 0's copy is unchanged.
    let peer1 = RowSplit::new(a.rows, 1).context("split")?;
    let ctrl_row = peer1.r0 + 3.min(peer1.rows - 1);
    let mut w_ctrl = w_host.clone();
    if a.rank == 1 {
        for h in 0..heads {
            w_ctrl[ctrl_row * heads + h] = -w_ctrl[ctrl_row * heads + h];
        }
    }
    let weights_ctrl = up(g, &f32_bytes(&w_ctrl))?;
    let inputs = DsaSelectInputs {
        k_normed,
        gate,
        valid,
        ape,
        q,
        weights,
        q_pos,
        q_mask,
        first_key: 0,
        geom_dev: DevicePtr::NULL,
    };
    let ctrl_inputs = DsaSelectInputs {
        weights: weights_ctrl,
        ..inputs
    };

    let full = || {
        select_tokens(
            g,
            &kernels,
            &cfg,
            &geom,
            &inputs,
            &scratch,
            DsaSelectLaunch::Exact,
            stream,
        )
    };
    let split_run = |inp: &DsaSelectInputs| {
        select_tokens_split(g, &kernels, &cfg, &geom, inp, &scratch, split, &comm, stream)
    };

    // 2026-10-01: Correctness on poisoned outputs: full, split, control.
    g.memset(scratch.tokens(), POISON_A, total)?;
    timed(g, stream, &full)?;
    let ha = down(g, scratch.tokens(), total)?;
    g.memset(scratch.tokens(), POISON_B, total)?;
    timed(g, stream, || split_run(&inputs))?;
    let hb = down(g, scratch.tokens(), total)?;
    g.memset(scratch.tokens(), POISON_B, total)?;
    timed(g, stream, || split_run(&ctrl_inputs))?;
    let hc = down(g, scratch.tokens(), total)?;

    let ab_diff = ha.iter().zip(&hb).filter(|(x, y)| x != y).count();
    let c_rows: std::collections::BTreeSet<usize> = ha
        .iter()
        .zip(&hc)
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(i, _)| i / row_bytes)
        .collect();
    let control_fired = c_rows.len() == 1 && c_rows.contains(&ctrl_row);
    let (own, peer) = (split.rows, a.rows - split.rows);
    let selected = ha
        .chunks_exact(4)
        .filter(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) >= 0)
        .count();

    // 2026-10-01: Timing, arms alternated.
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for _ in 0..a.reps {
        ta.push(timed(g, stream, &full)?);
        tb.push(timed(g, stream, || split_run(&inputs))?);
    }
    let (ma, mb) = (median(ta), median(tb));

    println!(
        "rank {} ctx {} rows {} (own {own} from row {}, peer {peer}); pools {} select_k {} \
         out_width {}; {selected} selected token slots in the full pass",
        a.rank,
        a.ctx,
        a.rows,
        split.r0,
        geom.n_pools,
        geom.select_k,
        cfg.out_width()
    );
    println!(
        "timing (median of {}): full {ma:.3} ms | split+exchange {mb:.3} ms | saved {:.3} ms \
         per DSA sub-chunk",
        a.reps,
        ma - mb
    );
    println!(
        "bytes compared {total}: full vs split differ {ab_diff}; control differs in rows \
         {c_rows:?} (want [{ctrl_row}], fired: {control_fired})"
    );
    if total > 0 && selected > 0 && ab_diff == 0 && control_fired {
        println!("PASS");
        Ok(())
    } else {
        println!("FAIL");
        std::process::exit(1);
    }
}
