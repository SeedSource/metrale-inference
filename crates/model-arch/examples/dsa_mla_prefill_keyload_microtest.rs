// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: What the remaining key-load cost of the DSA MLA prefill kernel (TC2 with A5) is
//! made of, TIMING ONLY. The phase ablation (`dsa_mla_prefill_ablation_microtest`) put the
//! gather at 33-35% of TC2; A5 (block-table page kept in registers one tile) recovered about half
//! (TC2 x0.775 of the TC kernel). TC2 issues tile t+2's key loads into registers during tile t
//! and stores them at tile t+1, so a load has about a tile to land; what is left is either load
//! latency tails stalling the CTA at its barriers (1 CTA of 8 warps per SM) or load issue cost.
//! The two call for different fixes (deeper prefetch / cp.async staging vs fewer, wider loads),
//! so this example times the production TC2 entry and the ablation module's `nogather` arm (no
//! key loads; the floor) on the same rows with four key layouts:
//!   `real`  the ablation example's ctx generator (neighbouring rows share ~95% of their 512
//!           selected pools; unique footprint ~14 MB, under the 24 MB L2);
//!   `l2`    the same draws restricted to the first `L2_TOKENS` tokens (4 MB: L2-resident after
//!           first touch, larger than L1);
//!   `l1`    every row selects the same `L1_POOLS` pools (32 KB: L1-resident, minimal latency);
//!   `dram`  every row draws fresh pools (no reuse between rows: DRAM-bound).
//! Cold L2 before each sample (64 MB read pass outside the events), only `cuLaunchKernel` between
//! the CUDA events, launches alternated, median of 15. Reading (pre-registered on #68): with
//! gap(x) = tc2(x) - nogather(x), gap(l1) <= 0.3 gap(real) says latency/memory (deeper
//! prefetch); gap(l1) >= 0.7 gap(real) says issue cost (wider loads); between is mixed.
//!
//! Output: per case and layout `KEYLOAD <case> <layout> tc2 <ms> ms nogather <ms> ms gap <ms> ms`,
//! then `KEYLOAD_GAP <case> l1/real <r> l2/real <r> dram/real <r>`. The final line starts `PASS`
//! when every launch succeeded and every time is finite; otherwise `FAIL` lines name the cause.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 on PASS, 1 on FAIL, 2 when the kernels are absent from this target.
//!
//! Run:
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!     cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example dsa_mla_prefill_keyload_microtest

use anyhow::{Context, Result};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::Glm5NextDsaConfig;
use metrale_model_arch::glm5next_dsa::attend::{
    DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, MLA_PREFILL_TC2_L2_FLUSH_ENTRY,
    MLA_PREFILL_TC2_MODULE, MLA_PREFILL_TC2_SMEM_BYTES, mla_scale, prefill_attention_tc2,
};
use metrale_model_arch::glm5next_dsa::select::DsaSelectGeometry;

const HEADS: usize = 32;
const KVL: usize = 512;
const BLOCK: usize = 64;
const KPOOL: usize = 4;
const TAIL: usize = 3;
const WIDTH: usize = 2051;
const K_SCALE: f32 = 0.0173;
const CTX_ROWS: usize = 256;
const EVENT_ITERS: usize = 15;
const CTX_POOL_MIN_BYTES: usize = 48 << 20;
const FLUSH_BYTES: usize = 64 << 20;
// 2026-10-09: The `l2` layout's token range (4 MB of keys) and the `l1` layout's pool count
// (64 keys, 32 KB).
const L2_TOKENS: usize = 8192;
const L1_POOLS: usize = 16;

/// 2026-10-09: The ablation module (the `.cu` file stem), its `nogather` entry and that entry's
/// dynamic shared memory (`AB_SMEM_NK32` in the kernel file).
const AB_MODULE: &str = "glm5next_dsa_mla_prefill_ablate";
const AB_NOGATHER: &str = "nogather";
const AB_SMEM_NK32: u32 = 99_104;

/// 2026-10-09: Where the rows' selected keys come from (module doc).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Real,
    L2,
    L1,
    Dram,
}

impl Layout {
    const ALL: [Layout; 4] = [Layout::Real, Layout::L2, Layout::L1, Layout::Dram];

    fn name(self) -> &'static str {
        match self {
            Layout::Real => "real",
            Layout::L2 => "l2",
            Layout::L1 => "l1",
            Layout::Dram => "dram",
        }
    }
}

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
    fn cuFuncSetAttribute(func: u64, attrib: i32, value: i32) -> i32;
    fn cuFuncGetAttribute(value: *mut i32, attrib: i32, func: u64) -> i32;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks: *mut i32,
        func: u64,
        block_size: i32,
        smem: usize,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        func: u64,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        smem: u32,
        stream: u64,
        params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> i32;
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f(&mut self) -> f32 {
        ((self.next() as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: KPOOL,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: HEADS,
        q_lora_rank: 1536,
        kv_lora_rank: KVL,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 131_072,
    }
}

/// 2026-10-09: One ctx case on the device.
struct Case {
    name: String,
    rows: usize,
    paging: DsaDecodePaging,
    inputs: DsaDecodeInputs,
    owned: Vec<DevicePtr>,
}

/// 2026-10-09: `dsa_mla_prefill_ablation_microtest`'s `ctx_case` with a key layout: the last
/// `CTX_ROWS` rows of a `seq`-token prefill on one shared, shuffled block table over a pool of
/// at least `CTX_POOL_MIN_BYTES`, 512 selected pools plus the tail per row, drawn per `layout`.
fn ctx_case(g: &dyn GpuBackend, seq: usize, seed: u64, layout: Layout) -> Result<Case> {
    let mut rng = Lcg(seed);
    let page = BLOCK * KVL;
    let n_logical = seq.div_ceil(BLOCK);
    let n_phys = n_logical.max(CTX_POOL_MIN_BYTES / page) + 8;
    let mut phys: Vec<i32> = (0..n_phys as i32).collect();
    for i in (1..phys.len()).rev() {
        phys.swap(i, rng.below(i + 1));
    }
    let bt: Vec<i32> = phys[..n_logical].to_vec();
    let n_sel_pools = (WIDTH - TAIL) / KPOOL;
    let first_sl = seq - CTX_ROWS + 1;
    let lim = if layout == Layout::L2 {
        L2_TOKENS / KPOOL
    } else {
        usize::MAX
    };
    let n_pools0 = ((first_sl - TAIL) / KPOOL).min(lim);
    let mut pools: Vec<usize> = Vec::with_capacity(n_sel_pools);
    let mut taken = std::collections::HashSet::new();
    if layout == Layout::L1 {
        pools.extend((0..n_sel_pools).map(|s| s % L1_POOLS));
    }
    while pools.len() < n_sel_pools {
        let p = rng.below(n_pools0);
        if taken.insert(p) {
            pools.push(p);
        }
    }
    let (mut seq_lens, mut sel) = (Vec::with_capacity(CTX_ROWS), Vec::new());
    for r in 0..CTX_ROWS {
        let sl = first_sl + r;
        seq_lens.push(sl as i32);
        let swap_pct = match layout {
            Layout::Real | Layout::L2 => 5,
            Layout::L1 => 0,
            Layout::Dram => 100,
        };
        if r > 0 {
            let n_pools = ((sl - TAIL) / KPOOL).min(lim);
            for slot in 0..n_sel_pools {
                if rng.below(100) < swap_pct {
                    let p = loop {
                        let p = rng.below(n_pools);
                        if !taken.contains(&p) {
                            break p;
                        }
                    };
                    taken.remove(&pools[slot]);
                    taken.insert(p);
                    pools[slot] = p;
                }
            }
        }
        for &p in &pools {
            sel.extend((p * KPOOL..p * KPOOL + KPOOL).map(|t| t as i32));
        }
        sel.extend((0..TAIL).map(|d| (sl - TAIL + d) as i32));
    }
    let q_bytes: Vec<u8> = (0..CTX_ROWS * HEADS * KVL)
        .flat_map(|_| bf16::from_f32(2.0 * rng.f() - 1.0).to_bits().to_le_bytes())
        .collect();
    let pool_bytes: Vec<u8> = (0..n_phys * page)
        .map(|_| {
            loop {
                let b = (rng.next() & 0xFF) as u8;
                if (b & 0x7F) != 0x7F {
                    break b;
                }
            }
        })
        .collect();
    let q = up(g, &q_bytes)?;
    let pool = up(g, &pool_bytes)?;
    let out = g.alloc(CTX_ROWS * HEADS * KVL * 2)?;
    let block_tables = up(g, &i32_bytes(&bt))?;
    let seq_lens_d = up(g, &i32_bytes(&seq_lens))?;
    let sel_indices = up(g, &i32_bytes(&sel))?;
    Ok(Case {
        name: format!("ctx{}k", seq / 1024),
        rows: CTX_ROWS,
        paging: DsaDecodePaging {
            num_seqs: CTX_ROWS,
            num_q_heads: HEADS,
            num_kv_heads: 1,
            max_blocks_per_seq: 0,
            block_size: BLOCK,
            cache_stride_bytes: (BLOCK * KVL) as u64,
        },
        inputs: DsaDecodeInputs {
            q,
            k_cache: pool,
            v_cache: pool,
            out,
            block_tables,
            seq_lens: seq_lens_d,
            sel_indices,
            k_scale: K_SCALE,
            v_scale: K_SCALE,
        },
        owned: vec![q, pool, out, block_tables, seq_lens_d, sel_indices],
    })
}

/// 2026-10-09: The checked TC2 launch (validation and warm-up).
fn launch_tc2(g: &dyn GpuBackend, kernel: Glm5NextDsaDecodeKernel, case: &Case) -> Result<()> {
    let geom = DsaSelectGeometry {
        seq: 131_072,
        q_rows: case.rows,
        n_pools_full: 0,
        n_pools: 0,
        select_k: 0,
        out_width: WIDTH,
        topk_np2: 0,
        topk_smem: 0,
        index_head_dim: 128,
        index_heads: 32,
        index_kpool: KPOOL,
    };
    prefill_attention_tc2(g, kernel, &cfg(), &geom, &case.paging, &case.inputs, 0)
}

/// 2026-10-09: The 64 MB L2 eviction read pass (TC2 module's `_l2_flush`).
struct L2Flush {
    func: KernelHandle,
    src: DevicePtr,
    sink: DevicePtr,
    blocks: u32,
}

impl L2Flush {
    fn new(g: &dyn GpuBackend) -> Result<Self> {
        let func = g
            .kernel(MLA_PREFILL_TC2_MODULE, MLA_PREFILL_TC2_L2_FLUSH_ENTRY)
            .context("L2 flush kernel")?;
        let src = g.alloc(FLUSH_BYTES)?;
        g.memset(src, 0x5A, FLUSH_BYTES)?;
        let sink = g.alloc(4)?;
        let blocks = 8 * g.sm_count().unwrap_or(48);
        g.synchronize(0)?;
        Ok(Self {
            func,
            src,
            sink,
            blocks,
        })
    }

    fn run(&self, g: &dyn GpuBackend) -> Result<()> {
        KernelLaunch::new(g, self.func)
            .grid([self.blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.src)
            .arg_u64((FLUSH_BYTES / 16) as u64)
            .arg_ptr(self.sink)
            .launch(0)
    }
}

/// 2026-10-09: One entry's launch with its arguments packed once (the TC2 argument list).
struct RawLaunch {
    func: u64,
    grid: [u32; 2],
    smem: u32,
    ptrs: [u64; 6],
    u32s: [u32; 4],
    f32s: [f32; 2],
    stride: u64,
}

impl RawLaunch {
    fn new(func: KernelHandle, smem: u32, case: &Case) -> Result<Self> {
        if func.0 == 0 {
            anyhow::bail!("entry point absent");
        }
        // 2026-10-09: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
        let rc = unsafe { cuFuncSetAttribute(func.0, 8, smem as i32) };
        if rc != 0 {
            anyhow::bail!("cuFuncSetAttribute failed: status {rc}");
        }
        let (p, i) = (&case.paging, &case.inputs);
        Ok(Self {
            func: func.0,
            grid: [(HEADS / 32) as u32, case.rows as u32],
            smem,
            ptrs: [
                i.q.0,
                i.k_cache.0,
                i.out.0,
                i.block_tables.0,
                i.seq_lens.0,
                i.sel_indices.0,
            ],
            u32s: [
                WIDTH as u32,
                p.max_blocks_per_seq as u32,
                p.num_q_heads as u32,
                p.block_size as u32,
            ],
            f32s: [mla_scale(&cfg()), i.k_scale],
            stride: p.cache_stride_bytes,
        })
    }

    fn launch(&mut self) -> i32 {
        use std::ffi::c_void;
        let mut params: [*mut c_void; 13] = [std::ptr::null_mut(); 13];
        for (k, v) in self.ptrs.iter_mut().enumerate() {
            params[k] = v as *mut u64 as *mut c_void;
        }
        for (k, v) in self.u32s.iter_mut().enumerate() {
            params[6 + k] = v as *mut u32 as *mut c_void;
        }
        for (k, v) in self.f32s.iter_mut().enumerate() {
            params[10 + k] = v as *mut f32 as *mut c_void;
        }
        params[12] = &mut self.stride as *mut u64 as *mut c_void;
        let [gx, gy] = self.grid;
        unsafe {
            cuLaunchKernel(
                self.func,
                gx,
                gy,
                1,
                256,
                1,
                1,
                self.smem,
                0,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
    }

    /// 2026-10-09: (registers, local bytes, CTAs per SM at this launch's shared memory).
    fn resources(&self) -> (i32, i32, i32) {
        let (mut regs, mut local, mut ctas) = (-1i32, -1i32, -1i32);
        unsafe {
            cuFuncGetAttribute(&mut regs, 4, self.func);
            cuFuncGetAttribute(&mut local, 3, self.func);
            cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut ctas,
                self.func,
                256,
                self.smem as usize,
            );
        }
        (regs, local, ctas)
    }
}

/// 2026-10-09: Median kernel milliseconds per launch, cold L2: `EVENT_ITERS` rounds over the
/// launches (order reversed on odd rounds), an L2 pass before each sample outside the events.
/// Launches that fail are reported in `failed` (index) and get no time.
fn event_times(
    g: &dyn GpuBackend,
    flush: &L2Flush,
    raws: &mut [RawLaunch],
    failed: &mut [bool],
) -> Result<Vec<f64>> {
    let (mut e0, mut e1) = (0u64, 0u64);
    let rc = unsafe { cuEventCreate(&mut e0, 0) } | unsafe { cuEventCreate(&mut e1, 0) };
    if rc != 0 {
        anyhow::bail!("cuEventCreate failed: status {rc}");
    }
    // 2026-10-09: One warm launch each.
    for (k, r) in raws.iter_mut().enumerate() {
        failed[k] |= r.launch() != 0;
    }
    g.synchronize(0)?;
    let n = raws.len();
    let mut samples = vec![Vec::with_capacity(EVENT_ITERS); n];
    for it in 0..EVENT_ITERS {
        for k in 0..n {
            let a = if it % 2 == 0 { k } else { n - 1 - k };
            if failed[a] {
                continue;
            }
            flush.run(g)?;
            let mut ms = 0f32;
            let rc = unsafe { cuEventRecord(e0, 0) };
            let lrc = raws[a].launch();
            let rc = rc
                | unsafe { cuEventRecord(e1, 0) }
                | unsafe { cuEventSynchronize(e1) }
                | unsafe { cuEventElapsedTime(&mut ms, e0, e1) };
            if lrc != 0 {
                failed[a] = true;
                g.synchronize(0).ok();
                continue;
            }
            if rc != 0 {
                anyhow::bail!("CUDA event timing failed: status {rc}");
            }
            samples[a].push(ms as f64);
        }
    }
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    Ok(samples
        .into_iter()
        .map(|mut v| {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        })
        .collect())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let Some(glm) = sets.iter().find(|s| s.target.model == "glm-5.3-flash") else {
        println!("glm-5.3-flash kernel target not built - SKIP");
        std::process::exit(2);
    };
    let backend = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &backend;
    let kernel = Glm5NextDsaDecodeKernel::resolve(g)
        .context("Glm5NextDsaDecodeKernel")?
        .with_prefill_tc2(g);
    if !kernel.has_prefill_tc2() {
        println!("glm5next_dsa_mla_prefill_tc2_fp8 absent from this target - SKIP");
        std::process::exit(2);
    }
    let entry = format!("{AB_MODULE}_{AB_NOGATHER}");
    let nogather = g
        .kernel(AB_MODULE, &entry)
        .with_context(|| format!("{entry} did not resolve"))?;
    let flush = L2Flush::new(g)?;
    let mut fails = 0usize;

    for (seq, seed) in [(131_072usize, 0xC7C_0131u64), (32_768, 0xC7C_0032)] {
        let mut gaps = Vec::with_capacity(Layout::ALL.len());
        for layout in Layout::ALL {
            let case = ctx_case(g, seq, seed, layout)?;
            launch_tc2(g, kernel, &case)?;
            g.synchronize(0)?;
            // 2026-10-09: Index 0 is TC2 (A5, the production entry), 1 the nogather floor.
            let mut raws = vec![
                RawLaunch::new(
                    kernel.prefill_tc2_handle(false),
                    MLA_PREFILL_TC2_SMEM_BYTES,
                    &case,
                )?,
                RawLaunch::new(nogather, AB_SMEM_NK32, &case)?,
            ];
            let mut failed = vec![false; 2];
            let ms = event_times(g, &flush, &mut raws, &mut failed)?;
            if failed.iter().any(|f| *f) || ms.iter().any(|m| !m.is_finite()) {
                println!(
                    "FAIL {} {}: launch failed or no time (tc2 {:.4}, nogather {:.4})",
                    case.name,
                    layout.name(),
                    ms[0],
                    ms[1]
                );
                fails += 1;
            }
            let gap = ms[0] - ms[1];
            println!(
                "KEYLOAD {} {} tc2 {:.4} ms nogather {:.4} ms gap {gap:.4} ms",
                case.name,
                layout.name(),
                ms[0],
                ms[1]
            );
            gaps.push(gap);
            let ((r0, l0, c0), (r1, l1, c1)) = (raws[0].resources(), raws[1].resources());
            println!(
                "INFO {} {} tc2 regs={r0} local_bytes={l0} ctas_per_sm={c0} nogather regs={r1} \
                 local_bytes={l1} ctas_per_sm={c1}",
                case.name,
                layout.name()
            );
            for p in &case.owned {
                g.free(*p).ok();
            }
        }
        println!(
            "KEYLOAD_GAP ctx{}k l1/real {:.3} l2/real {:.3} dram/real {:.3}",
            seq / 1024,
            gaps[2] / gaps[0],
            gaps[1] / gaps[0],
            gaps[3] / gaps[0]
        );
    }
    g.free(flush.src).ok();
    g.free(flush.sink).ok();

    if fails > 0 {
        println!("FAIL - {fails} key-load check(s) failed (see FAIL lines above)");
        std::process::exit(1);
    }
    println!(
        "PASS - TC2 and the nogather floor launched and timed in every case and layout (cold L2, \
         median of {EVENT_ITERS})."
    );
    Ok(())
}
