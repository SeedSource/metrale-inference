# Kernel performance map

The living map for taking every GPU kernel of the engine to its hardware limit. For each kernel
entry point it records where it lives, which models launch it, which architectural component it
serves, what was traded away to get it where it is (with the pull requests behind it), and how far
its measured time is from the roofline floor of the hardware it runs on.

Most of this file is **generated** by [`scripts/kernel_perf.py`](scripts/kernel_perf.py) from the
tree itself plus three curated inputs under [`docs/kernel-perf/`](docs/kernel-perf/). CI runs
`python3 scripts/kernel_perf.py --check` in the cheap-checks job, so a kernel added, removed,
renamed or rewired without regenerating this file fails the PR. Edit the inputs, never the
generated block.

**Where to look**

| You want | Section |
|---|---|
| What is in the tree, in numbers | [Inventory at a glance](#inventory-at-a-glance) |
| Which checkpoints form each family | [Architecture families](#architecture-families) |
| Per-component counts (primary, launched from, unique, dead, measured) | [Components](#components) |
| Kernels every decoder architecture runs | [Shared by all LLM architectures](#shared-by-all-llm-architectures) |
| Every kernel a component launches | [Kernels by component](#kernels-by-component) |
| Kernels only one component launches | [Unique kernels by component](#unique-kernels-by-component) |
| Compiled code no engine path launches | [Compiled but not launched](#compiled-but-not-launched) |
| The trade-off notes behind the "Trade-offs" cells | [`docs/kernel-perf/TRADEOFFS.md`](docs/kernel-perf/TRADEOFFS.md) |
| Measured regimes at a glance | [Measurements](#measurements) |
| Every measured row behind a "% of floor" cell | [`docs/kernel-perf/MEASUREMENTS.md`](docs/kernel-perf/MEASUREMENTS.md) |

## Methodology

### 1. Inventory: derived from the tree, never typed in

1. **Targets.** Every `(hardware, model, quant)` target is enumerated and resolved with
   [`scripts/lib/kernel_layout.py`](scripts/lib/kernel_layout.py), the Python mirror of the
   resolver the build uses (`crates/closure/src/layout.rs`, held to the same answer on the real
   tree by `crates/kernels/tests/layout_mirror.rs`). `[sources] use`, `[hardware] inherits`,
   `[model] kernel_source` redirects and declared `[shadow]` forks are therefore applied exactly as
   `crates/kernels/build.rs` applies them. A source file that many model directories or hardware
   trees compile is **one row** listing all of them; a fork (a same-stem file with different bytes,
   see [`kernels/FORKS.md`](kernels/FORKS.md)) is a separate row because it is separate code.
2. **Modules.** A compiled file's module is its stem, renamed by the `[modules]` tables of the
   target's `KERNEL.toml`s merged least-specific first, as the build does. The table shows
   `module::function`, the pair the engine passes to `gpu.kernel(module, function)`.
3. **Entry points.** [`scripts/lib/kernel_perf_scan.py`](scripts/lib/kernel_perf_scan.py) reads
   each compiled source and the local headers it includes, with comments stripped, and takes every
   `extern "C" __global__` function (CUDA/HIP) or `kernel` function (Metal). Names built by the
   preprocessor are expanded: function-like macros (`WYN_INSTANTIATE(K)` → `gated_delta_rule_wy##K`),
   object-like names defined by the including file before a shared body (`#define KERNEL_NAME` +
   `prefill_paged_compute.cuh`), and pasting helpers (`PAGED_CONCAT(KERNEL_NAME, _64)`). A kernel
   whose body lives in a header is listed under the header, with the compiled `.cu` recorded as
   its source. `static` and template `__global__` functions are not loader-visible and are not
   rows. The scan was checked on 2026-09-27 against the `.entry` directives of a full nvcc PTX
   build of all 32 gb10 targets: the only differences were entry points added or removed in
   commits after that build and the six mangled internal kernels of the vendored MMQ code.
4. **Call sites.** Every string literal in `crates/**/*.rs` equal to an entry-point name is a call
   site; a `{}`-templated literal (`format!("w4a16_gemv_sw_moe_batchm_m{r}")`) counts for the names
   it matches when no exact literal exists. When the call spells its module (a literal or a
   `const X: &str`), the site is attached only to the entry points of that module, which
   separates same-named kernels in different files (the ten `paged_decode_attn_splitk_nvfp4`s).
   Sites in `tests/`, `*_tests.rs`, inline `#[cfg(test)] mod` blocks and `examples/` are counted
   but never make a kernel "used"; an entry point with no other site is listed under
   [Compiled but not launched](#compiled-but-not-launched).

### 2. Families and components: curated, and validated against the tree

[`docs/kernel-perf/taxonomy.toml`](docs/kernel-perf/taxonomy.toml) holds the only hand-written
classification:

- **Architecture families** group model directories by architecture (all model directories of the
  same name on every hardware tree belong to the same family). Each lists the
  architecture-specific components its layers have.
- **Components** are the architectural pieces the code is organised around: attention, MLA,
  sparse attention, GDN, KDA, Mamba2, causal conv1d, MoE, dense FFN, the projection GEMM/GEMV
  formats, norms, elementwise, RoPE, KV cache, quantization, embedding/LM head, sampling,
  speculative decoding, hyper-connections, n-gram memory, vision, NLLB, LoRA, one-time weight
  load, and diagnostics. A component is *generic* when every decoder family has it (attention,
  norms, projections, ...). Each component names the Rust paths that implement it (`sites`), e.g.
  `crates/model-layers/src/layers/qwen3_ssm/` for GDN.
- **Rules** give each entry point its *primary* component and a short *kind* (first matching rule
  by name/path glob wins).

The generator refuses to render while any entry point matches no rule, a rule matches nothing, a
model directory belongs to no family, or a `sites` path matches no file, so the taxonomy cannot
drift silently from `kernels/` and `crates/`.

### 3. "LLMs" column: which checkpoints launch a kernel

For each engine call site of a kernel, a family is admitted when all four hold:

1. a target of the family **compiles** the file (from the resolver, step 1.1);
2. the kernel's primary component is generic or **one the family has**;
3. the component owning the call-site path is generic or one the family has (a GEMV launched from
   the GDN layer counts only for GDN families, the same GEMV launched from the attention layer for
   every family);
4. when the path is **family-exclusive** (`[[family_site]]`, e.g. `crates/model-arch/src/kimi_k3`),
   the family is one of those.

A family whose forward pass does not use the shared transformer path (`shared_engine = false`:
NLLB) is admitted only through its exclusive paths. The cell names the families and the number of
checkpoints they contain; the [families table](#architecture-families) maps each to its model
directories and checkpoint ids. This is **reachability**, not a launch trace: a kernel is listed
for every family that compiles it and reaches its call site. Which checkpoint actually launches it
in a given run can narrow further by runtime choices (KV-cache dtype, head_dim, batch width, the
`METRALE_*` levers and `[defaults]` rows of `HARDWARE.toml`), which a profile of that run shows.
"none — its callers' targets compile another copy" marks a file whose call sites exist but whose
every caller resolves the same name to a different (forked) file.

### 4. Shared and unique

- **Shared by all LLM architectures** = used (step 3) by *every* decoder family (every family with
  `shared_engine = true`; there are 14). NLLB is an encoder-decoder with its own self-contained
  kernel set, so it is outside the quorum; a row also names it when NLLB uses the kernel.
- **Kernels by component** lists, under each component, a full row for every entry point whose
  *primary* component it is, plus the names of other components' entry points that the
  component's code launches (their full rows are under their own component).
- **Unique to a component** = every engine call site of the entry point belongs to that one
  component (the owner of the call-site path, or the primary component where no component owns
  the path).

### 5. Trade-offs and PRs

[`docs/kernel-perf/tradeoffs.toml`](docs/kernel-perf/tradeoffs.toml) holds one entry per known
trade-off, keyed by source file (applies to every entry point the file defines or compiles) or by
`file::function`: what was given up for what, known limits (registers, occupancy, spills, shape
restrictions, precision and bit-exactness across batch widths), and dated measurements, each with
its source (a code comment by `file:line`, a pull request, a perf document, or a dated
measurement note). PR numbers are pull requests of this repository only. The generator rejects an
entry whose key no longer names a kernel, so deleting or renaming a kernel forces its notes to be
moved or dropped. The notes are rendered to
[`docs/kernel-perf/TRADEOFFS.md`](docs/kernel-perf/TRADEOFFS.md); table cells link there and to the
PRs.

### 6. "% of floor": how far a kernel is from the hardware limit

For one kernel call in one regime:

```
floor_us     = max(bytes / BW_peak, flops / FLOP_peak(dtype))
pct_of_floor = floor_us / time_us * 100        (100 % = at the roofline floor)
```

- `time_us` is the **measured** median device time of the call (Nsight Systems kernel trace of a
  real serve run in the named regime, one row per kernel per regime).
- `bytes` is the **minimum** DRAM traffic the call must move, from the call's shapes: every weight
  byte it reads once (packed format including block scales), every activation / KV / state byte it
  must read or write once. Re-reads a better kernel could avoid are *not* counted, so cache misses
  and redundant passes show up as distance from the floor.
- `flops` is the **useful** arithmetic (2·M·N·K for a GEMM, the attention and recurrence FLOPs the
  math requires), at the precision the tensor cores execute.
- `bound` names the term that sets the floor. The peaks are the ones in
  [Peaks (GB10)](#peaks-gb10) below.
- **Above 100 %** means the kernel beat the modelled floor: the byte model counts traffic the
  hardware served from L2 (data reused across back-to-back calls), or the kernel streams faster
  than the peak in use. Treat it as a prompt to re-check that row's byte model, not as headroom
  below zero.
- **"not measured"** means no row exists for that kernel yet. It is never estimated, interpolated
  or copied from another kernel, and a kernel measured in one regime only shows that regime.

The summary cell shows the lowest and highest % over the kernel's rows and the regime of the
lowest; [`docs/kernel-perf/MEASUREMENTS.md`](docs/kernel-perf/MEASUREMENTS.md) (generated) has every
row with its shape, source (trace, commit, box, date) and notes. The generator re-derives `pct_of_floor` from `floor_us / time_us` and fails if
a stored value disagrees by more than 0.5 points, and fails on a row naming a kernel the tree does
not have.

#### Peaks (GB10)

Measured on one GB10 with the GPU otherwise idle, 2026-09-28, SM clock 2405–2496 MHz under load,
microbenchmarks built for `sm_121a` (plain `sm_121` rejects the block-scaled FP4 MMA). The floor uses
the **measured achievable** peak, not the datasheet, so 100 % is reachable; the datasheet column
converts (a row's % of the 273 GB/s datasheet bandwidth is `pct × 249/273 = pct × 0.912`).

| Resource | Floor uses | How it was measured | Datasheet / nominal |
|---|---|---|---|
| DRAM read | **249.0 GB/s** | one CTA streams one contiguous 8 KiB tile (512 threads, 16 B `ld.global.nc` per lane) over a 4 GiB buffer, median of 7; in-model GEMVs reach 246–253 GB/s | 273 GB/s (LPDDR5X-8533, 256-bit) |
| DRAM read, grid-stride / write / copy | 236.0 / 196.6 / 214.3 GB/s | streaming `__ldcs` / `__stcs` kernels; not used as the floor | |
| BF16 `mma.sync.m16n8k16`, FP32 acc | **123.7 TFLOPS** | registers only, 8 independent accumulators per warp, best of 48×{1,2,4} CTAs × {4,8} warps | 125 |
| E4M3 `mma.sync.m16n8k32`, FP32 acc | **243.6 TFLOPS** | same | 250 |
| NVFP4 `mma.sync…mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` | **490.8 TFLOPS** | same (SASS `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`) | 500 (dense) |
| FP32 FFMA (CUDA cores) | **30.0 TFLOPS** | 8 independent FFMA chains; FMUL+FADD (the `--fmad=false` form) reaches 14.9 | 30.1 |

One bandwidth covers reads and writes, which flatters write-heavy kernels: a pure copy tops out at
86 % of its floor and a pure write at 79 %, so an elementwise kernel near 85 % is at the practical
limit. Tensor-core peaks need at least 8 resident warps per SM (4 warps on one CTA per SM reach 115
BF16 / 448 FP4 TFLOPS). Hopper, B200/B300, Strix and Metal have no measured peaks yet, so every
kernel on those targets is "not measured".

**Peak class.** `PEAK[class]` is the tensor-core rate of the operand formats the kernel
*implements*, not of the instruction it happens to issue: `fp8` for W8A8 and W4A8 (NVFP4 weight
dequantized to E4M3 × E4M3 activations), `fp4` for W4A4 NVFP4 × NVFP4, `bf16` for W8A16 / W4A16 /
BF16 GEMMs, attention and GDN matmuls, `fp32` only for the FP32-state GDN recurrent decode. So a W8A8
kernel that decodes E4M3 to BF16 in software and issues BF16 MMAs is judged against the FP8 peak,
and the exact-order FP32 router GEMMs against the BF16 peak (their notes say so). A floor never
assumes a lower precision than the kernel implements; a lever that changes the number format
(NVFP4 experts, W4A4 downcast) changes `bytes` itself, not the floor of the old kernel.

**No latency term.** Launch-latency-bound kernels (single-CTA sorts and worklists, top-k, argmax,
1–4-row norms) read near 0 %: their gap is recovered by fusing or removing the launch, not by
bandwidth. A row's `notes` give its share of the regime's GPU time, which says whether it matters.

#### Byte and FLOP models

The shape of each call is recovered from its launch grid through the launcher's grid formula
(`crates/model-layers/src/layers/ops/*.rs`), then:

| Family | Bytes of one call | FLOPs, class |
|---|---|---|
| FP8 block-scaled / unscaled FP8 GEMM | N·K weight + block scales + M·K activations (+ scales) + 2·M·N out | 2MNK, fp8 |
| W8A16 GEMM/GEMV | E4M3 weight + scales + 2·M·K + 2·M·N | 2MNK, bf16 |
| NVFP4 W4A16 / W4A8 GEMM/GEMV | 0.5625·N·K (E2M1 + UE4M3 per 16) + 2·M·K + 2·M·N | 2MNK, bf16 or fp8 |
| NVFP4 W4A4 MMQ | 0.5625·(N·K + M·K) + 2·M·N | 2MNK, fp4 |
| BF16 GEMV/GEMM | 2·N·K + 2·M·K + 2·M·N | 2MNK, bf16 |
| MoE grouped prefill (W8A8) | all routed experts' weights + scales, activations once, outputs once | 2·(top-k·T)·N·K, fp8 |
| MoE grouped decode | (distinct experts + shared) × expert bytes + activations; the distinct-expert count D comes from a routing log (R = 2 → 14.5, 8 → 45, 16 → 72, 32 → 121.5, 64 → 153) because the grid is a capacity | 2·rows·N·K, bf16 |
| Attention prefill | Q + O + K/V once ((prefix + chunk) · kv heads · head_dim · 2 · 2 B, BF16 KV) | 4·hd·nq·(T·P + T(T+1)/2), bf16 |
| Attention decode | each sequence's K and V once, plus q and o | 4·rows·nq·L·hd, bf16 |
| GDN chunked prefill (chunk 64) | per stage: its inputs once, its outputs (W/U, per-chunk states, o) once | per (chunk, v-head) matmul FLOPs, bf16 |
| GDN decode | recurrent state read + write per sequence (FP32 or FP16 pool), read-only for the lazy carried-state kernels whose write-back is deferred, plus q/k/v/o | 8·rows·nv·dk², fp32 |
| Conv, norms, RoPE, residual, cache writes | one read and one write of each tensor (RoPE: rotary dims only) | — |

`bytes` counts every weight byte (scales included) once, every activation input once, every output
once, and recurrent/KV state once (read, and written where the step must persist it). Re-reads a
kernel actually does (per-M-tile weight re-streaming, activation re-reads per N tile, L2 misses) are
not counted, so they show up as distance from the floor. `flops` excludes padding (a 64-row MMA
tile carrying 32 live rows), dequantization arithmetic and softmax work.

#### Regimes

A regime names the serving situation the time was taken in: `decode C=<concurrent sequences>`
with the speculative verify rows `R`, `prefill <prompt tokens>` for a single cold request, and the
model. A kernel's efficiency depends strongly on the regime (a decode GEMV at C=1 and C=16 moves the
same weights for up to 16x the useful work), so rows are never merged across regimes; a kernel that
runs at several shapes in one regime gets one row per shape (`regime · <shape>`). Measured so far,
all on GB10 at main `e37e3cb2` (kernel-equivalent binary), `nsys --trace=cuda --cuda-graph-trace=node`:

| Model | Regime | Serving configuration |
|---|---|---|
| Qwen/Qwen3.6-35B-A3B-FP8 | decode C=1, R=2 | concurrency-sweep flags (BF16 KV, MTP 1 draft, forced), 3 s window of a 1000-token greedy essay, KV ≈ 585 |
| Qwen/Qwen3.6-35B-A3B-FP8 | decode C=16, R=32 | same, 16 concurrent, KV ≈ 273 |
| Qwen/Qwen3.6-35B-A3B-FP8 | prefill 4k (4549 tok) and 32k (32772 tok), cold | TTFT-gate recipe (FP8 weights, BF16 LM head, BF16 KV) |
| unsloth/Qwen3.8-27B-NVFP4 | decode C=1, R=4 | decode-floor recipe (BF16 KV and LM head, 3 drafts), KV ≈ 245 |
| unsloth/Qwen3.8-27B-NVFP4 | decode C=16, R=32 | concurrency-sweep recipe (FP8 KV, FP16 SSM pool, batched recurrent), KV ≈ 308 |
| unsloth/Qwen3.8-27B-NVFP4 | prefill 4k (4103 tok) and 32k (32772 tok), cold | decode-floor recipe flags |

Between 99.8 % and 99.9 % of each regime's GPU kernel time has a floor model; a (kernel, shape) is
written as a row when it is at least 0.3 % of its regime's kernel time. Each window's purity is
checked from the grids (for example grouped `gridY = 264` means R = 32).

**Caveats.** The profiler inflates decode steps by about 3 % at C=1, about 8 % (dense) and 12–19 %
(MoE) at C=16, mostly inside kernel durations, so C=16 percentages are *understated* by up to that
factor; prefill inflation is 1–2 %. Per-call time is the median over the window's calls of that
(kernel, shape). Decode KV lengths are short (245–585 tokens), so long-context decode attention is a
separate, not yet measured regime. Byte counts are modelled, not counted; an `ncu` DRAM-bytes pass
over the largest rows would validate them.

### 7. Updating

```bash
python3 scripts/kernel_perf.py            # regenerate KERNEL-PERF.md, docs/kernel-perf/{TRADEOFFS,MEASUREMENTS}.md
python3 scripts/kernel_perf.py --check    # what CI runs: stale file or taxonomy drift -> exit 1
python3 scripts/kernel_perf.py --json     # the joined inventory (targets, call sites, families) as JSON
python3 scripts/kernel_perf_test.py       # generator self-test on a fixture tree (also run by CI)
```

- **A kernel was added, removed or renamed:** run the generator. If it reports `no rule
  classifies`, add or widen a rule in `taxonomy.toml`; if it reports a trade-off or measurement
  naming nothing, move or delete that entry.
- **A new model directory:** add it to its family in `taxonomy.toml` (or add a family).
- **A trade-off was made or learned:** add a `[[t]]` entry to `tradeoffs.toml` with its source
  and PR.
- **A kernel was measured:** append `[[m]]` rows to `measurements.toml` (schema below) and
  regenerate. Replace a kernel's rows for a regime when it is re-measured; keep the source line
  exact (trace, commit, box, date) so the number can be reproduced.

```toml
[[m]]
kernel = "<module>::<function>"        # as the loader names it
file = "kernels/gb10/common/x.cu"      # the defining file, or the .cu that compiles a header body
hardware = "gb10"
model = "Qwen/Qwen3.6-35B-A3B-FP8"
regime = "decode C=16 (R=32)"           # or "prefill 32k", "decode C=1", ...
time_us = 0.0                           # measured, per call (median)
bytes = 0                               # minimum DRAM bytes the call must move (model)
flops = 0                               # useful FLOPs (model)
bound = "memory"                        # or "compute"
floor_us = 0.0                          # max(bytes/BW_peak, flops/FLOP_peak) with the peaks above
pct_of_floor = 0.0                      # floor_us / time_us * 100
source = "nsys <file> @ <commit>, <box>, <date>"
notes = ""
```

<!-- kernel_perf.py: BEGIN GENERATED (edit the inputs, then run scripts/kernel_perf.py) -->

## Inventory at a glance

- **1333 kernel entry points** in **341 source files** across 7 hardware trees (b200, b300, gb10, hopper, metal, strix, strix-hip), compiled into 58 (hardware, model, quant) targets.
- **1076** have at least one engine call site; **257** are compiled but launched only from tests, examples or not at all (see [Compiled but not launched](#compiled-but-not-launched)).
- **15 architecture families**, **29 components**.
- **61** entry points have a measured % of floor; every other row reads “not measured”.

## Architecture families

| Label | Family | Checkpoints (model directory → checkpoint) | Components | Entry points used |
|---|---|---|---|---|
| Qwen-GDN | Qwen3.x GDN hybrid, dense FFN | `qwen3.5-27b` → Kbenkhaled/Qwen3.5-27B-NVFP4<br>`qwen3.6-27b` → Qwen/Qwen3.6-27B<br>`qwen3.8-27b` → Qwen/Qwen3.8-27B<br>`holo-3.1-0.8b` → Hcompany/Holo-3.1-0.8B<br>`holo-3.1-4b` → Hcompany/Holo-3.1-4B<br>`ornith-1.0-9b` → deepreinforce-ai/Ornith-1.0-9B<br>`qwen3-5-4b-vlm-mlx-int8` → mlx-community/Qwen3.5-4B-MLX-8bit | GDN, Causal conv1d, Dense FFN, Vision encoder | 557 |
| Qwen-GDN-MoE | Qwen3.x GDN hybrid, MoE (incl. Qwen3-Next) | `qwen3.5-35b-a3b` → Sehyo/Qwen3.5-35B-A3B-NVFP4<br>`qwen3.5-122b-a10b` → Sehyo/Qwen3.5-122B-A10B-NVFP4<br>`qwen3.5-397b-a17b` → nvidia/Qwen3.5-397B-A17B-NVFP4<br>`qwen3.6-35b-a3b` → Qwen/Qwen3.6-35B-A3B-FP8<br>`holo-3.1-35b-a3b` → Hcompany/Holo-3.1-35B-A3B-NVFP4<br>`qwen3-next-80b-a3b` → nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 | GDN, Causal conv1d, MoE, Dense FFN, Vision encoder | 598 |
| Qwen3.8-FN | Qwen3.8-Flash-Next (GDN + QSA sparse attention + mHC + PLE + MoE) | `qwen3.8-flash-next` → Qwen/Qwen3.8-Flash-Next | GDN, Causal conv1d, Sparse / compressed attention, Hyper-connections, N-gram and memory embeddings, MoE, Dense FFN, Vision encoder | 510 |
| Qwen3-VL | Qwen3-VL MoE (full attention) | `qwen3-vl-30b-a3b` → ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 | MoE, Vision encoder | 338 |
| Gemma4 | Gemma 4 (sliding/full attention, dense and MoE) | `gemma-4-26b-a4b` → bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16<br>`gemma-4-31b` → nvidia/Gemma-4-31B-IT-NVFP4 | MoE, Dense FFN, Vision encoder | 364 |
| Nemotron-H | Nemotron-H (Mamba2 hybrid + MoE) | `nemotron-3-nano-30b-a3b` → nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4<br>`nemotron-super-120b-a12b` → nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4<br>`nemotron-labs-3-puzzle-75b-a9b` → nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 | Mamba2, Causal conv1d, MoE, Dense FFN | 394 |
| DeepSeek-V4 | DeepSeek-V4 (MLA + CSA/HCA + mHC + MoE + Engram) | `deepseek-v4-flash` → RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8<br>`deepseek-v4.1-flash` → deepseek-ai/DeepSeek-V4.1-Flash | MLA, Sparse / compressed attention, Hyper-connections, N-gram and memory embeddings, MoE, Dense FFN | 436 |
| Mistral4 | Mistral Small 4 (MLA + MoE) | `mistral-small-4` → mistralai/Mistral-Small-4-119B-2603-NVFP4 | MLA, MoE, Dense FFN | 346 |
| GLM-5.3 | GLM-5.3-Flash (KDA + DSA sparse MLA + mHC + MoE) | `glm-5.3-flash` → LibertAIDAI/GLM-5.3-Flash-NVFP4 | KDA, Causal conv1d, MLA, Sparse / compressed attention, Hyper-connections, MoE, Dense FFN, Vision encoder | 394 |
| Kimi-K3 | Kimi K3 (KDA + gated MLA + LatentMoE) | `kimi-k3` → inference-optimization/Kimi-K3-0.40B | KDA, Causal conv1d, MLA, MoE, Dense FFN | 347 |
| Laguna | Laguna (full/sliding attention + MoE) | `laguna-s-2.1` → poolside/Laguna-S-2.1-NVFP4<br>`laguna-xs-2.1` → poolside/Laguna-XS-2.1-NVFP4 | MoE, Dense FFN | 339 |
| MiniMax-M2 | MiniMax-M2 (full attention + sigmoid MoE) | `minimax-m2-229b` → MiniMaxAI/MiniMax-M2.7 | MoE, Dense FFN | 339 |
| Step-3.7 | Step-3.7-Flash (full/sliding attention + sigmoid MoE) | `step3p7-flash` → stepfun-ai/Step-3.7-Flash-NVFP4 | MoE, Dense FFN, Vision encoder | 338 |
| LongCat | LongCat-Flash-Lite (MLA + MoE + n-gram embeddings) | `longcat-flash-lite` → meituan-longcat/LongCat-Flash-Lite | MLA, MoE, N-gram and memory embeddings, Dense FFN | 359 |
| NLLB | NLLB-200 (encoder-decoder translation) | `nllb-200-3.3b` → facebook/nllb-200-3.3B | Encoder-decoder translation | 28 |

## Components

| Component | Scope | Primary entry points | Launched from it (incl. other primaries) | Unique to it | Not launched | Measured |
|---|---|---|---|---|---|---|
| Attention (GQA/MHA: paged decode, split-K, prefill/flash) | every family | 104 | 336 | 193 | 36 | 20 |
| MLA (multi-head latent attention) | families listing it | 34 | 34 | 0 | 7 | 0 |
| Sparse / compressed attention (DSA, CSA/HCA, QSA) | families listing it | 43 | 61 | 44 | 0 | 3 |
| GDN (gated delta rule linear attention) | families listing it | 187 | 335 | 217 | 11 | 27 |
| KDA (Kimi delta attention, linear attention) | families listing it | 11 | 24 | 10 | 3 | 5 |
| Mamba2 (selective state-space scan) | families listing it | 6 | 63 | 7 | 1 | 8 |
| Causal conv1d (short convolution of GDN/KDA/Mamba2) | families listing it | 9 | 9 | 0 | 3 | 2 |
| MoE (routing, dispatch, expert GEMM/GEMV, combine) | families listing it | 187 | 284 | 196 | 44 | 22 |
| Dense FFN (gate/up/down projections of non-MoE layers) | families listing it | 0 | 88 | 34 | 0 | 12 |
| Projection GEMM/GEMV — BF16/F32 | every family | 26 | 26 | 6 | 1 | 4 |
| Projection GEMM/GEMV — FP8 (W8A16, W8A8, block-scaled) | every family | 73 | 73 | 7 | 1 | 6 |
| Projection GEMM/GEMV — NVFP4 W4A16 | every family | 60 | 60 | 14 | 10 | 7 |
| Projection GEMM/GEMV — W4A4 (FP4 activations) | every family | 22 | 22 | 11 | 0 | 2 |
| Projection GEMM/GEMV — integer / K-quant (Q2_0, Q2_K..Q6_K, INT8, MLX INT8) | every family | 23 | 23 | 6 | 31 | 0 |
| Normalization (RMSNorm, LayerNorm, L2, gated norms) | every family | 64 | 64 | 0 | 37 | 4 |
| Activations and elementwise (SiLU/GELU/ReLU², residual, gates, scale) | every family | 16 | 16 | 0 | 16 | 3 |
| Positional encoding (RoPE, YaRN, MRoPE) | every family | 12 | 12 | 0 | 0 | 1 |
| KV cache (write, quantize, TurboQuant rotation, slot metadata) | every family | 36 | 36 | 1 | 2 | 0 |
| Quantization and format conversion | every family | 47 | 47 | 4 | 7 | 4 |
| Embedding and LM head (lookup, overlays, softcap, scale) | every family | 11 | 16 | 7 | 6 | 0 |
| Sampling (argmax, top-p, feed-forward of the chosen token) | every family | 6 | 6 | 3 | 2 | 1 |
| Speculative decoding (MTP heads, DFlash drafter, verify helpers) | every family | 3 | 89 | 16 | 0 | 12 |
| Hyper-connections (mHC) | families listing it | 27 | 27 | 13 | 2 | 0 |
| N-gram and memory embeddings (Engram, PLE, n-gram tables) | families listing it | 5 | 16 | 6 | 0 | 2 |
| Vision encoder (ViT towers) | families listing it | 32 | 36 | 32 | 3 | 0 |
| Encoder-decoder translation (NLLB, self-contained kernel set) | families listing it | 26 | 29 | 24 | 28 | 1 |
| LoRA adapters (BGMV shrink/expand) | every family | 6 | 6 | 6 | 0 | 0 |
| Weight load and repack (one-time, not per token) | every family | 0 | 62 | 15 | 0 | 2 |
| Diagnostics and microtests (no serving path) | every family | 0 | 0 | 0 | 6 | 0 |

## Shared by all LLM architectures

Entry points used by **every one of the 14 decoder families** (every family with `shared_engine = true`; NLLB, the self-contained encoder-decoder, is excluded from the quorum and named when it also uses the kernel). 215 entry points qualify; 2 of them are used by all 15 families.

| Kernel (module::function) | File | Component · kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| argmax::`argmax_bf16` | [gb10/common/argmax_bf16.cu:14][f5] | Sampling · argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families + NLLB (31 ckpts) | [1 note][t5] | [2%][m5.argmax_bf16] (decode C=1 (R=4, MTP k=3)) |
| argmax::`argmax_{bf16_batch, bf16_batch_lp, fp32}` (3) | [gb10/common/argmax_bf16.cu:68][f5] | Sampling · argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t5] | not measured |
| argmax_feed::`argmax_bf16_batch_feed`, `feed_resolve` | [gb10/common/argmax_feed.cu:44][f6] | Sampling · argmax / top-p | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t6] | not measured |
| attn_prefill::`attn_prefill` | [gb10/common/attn_prefill.cu:78][f7] | Attention · prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | not measured |
| attn_prefill::`attn_prefill_64` | [gb10/common/attn_prefill.cu:562][f7] | Attention · prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | [27%][m7.attn_prefill_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_fa128::`attn_prefill_{fa128, fa128_paged}` (2) | [gb10/common/attn_prefill_fa128.cu:356][f8] | Attention · prefill (flash) | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [gb10/common/attn_prefill_h128.cu:46][f10] | Attention · prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t10] | not measured |
| bf16_add::`bf16_add_inplace` | [gb10/common/bf16_add.cu:8][f12] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t12] | not measured |
| gemm::`dense_gemm_bf16` | [gb10/common/dense_gemm_bf16.cu:26][f14] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | [1–2%][m14.dense_gemm_bf16] (prefill 32k (cold, 32772 tok)) |
| gemm::`dense_gemm_bf16_f32out` | [gb10/common/dense_gemm_bf16.cu:85][f14] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | not measured |
| gemm::`dense_gemm_bf16_pipelined` | [gb10/common/dense_gemm_bf16.cu:446][f14] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families + NLLB (31 ckpts) | [1 note][t14] | not measured |
| gemm_splitk::`dense_gemm_splitk_{partial, reduce}` (2) | [gb10/common/dense_gemm_splitk.cu:27][f15] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t15] | not measured |
| gemm_tc::`dense_gemm_{tc, tc_scaled_acc}` (2) | [gb10/common/dense_gemm_tc.cu:185][f16] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t16] | not measured |
| gemv::`dense_gemv_bf16` | [gb10/common/dense_gemv_bf16.cu:33][f17] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t17] | [94–100%][m17.dense_gemv_bf16] (decode C=1 (R=4, MTP k=3)) |
| dense_gemv_bf16_batchm::`dense_gemv_bf16_batchm` | [gb10/common/dense_gemv_bf16_batchm.cu:86][f19] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t19] | [16–92%][m19.dense_gemv_bf16_batchm] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_tc16` | [gb10/common/dense_gemv_bf16_tc.cu:251][f20] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | [86–95%][m20.dense_gemv_bf16_tc16] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_{tc32, tc8}` (2) | [gb10/common/dense_gemv_bf16_tc.cu:250][f20] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | not measured |
| gemv_fp8w::`dense_gemv_fp8w` | [gb10/common/dense_gemv_fp8w.cu:131][f21] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| gemv_fp8w::`quantize_bf16_to_fp8` | [gb10/common/dense_gemv_fp8w.cu:65][f21] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| dense_gemv_fp8w_batch2::`dense_gemv_fp8w_batch2` | [gb10/common/dense_gemv_fp8w_batch2.cu:72][f22] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t22] | not measured |
| dequant_fp8_blockscaled_bf16::`dequant_fp8_blockscaled_bf16` | [gb10/common/dequant_fp8_blockscaled_bf16.cu:89][f23] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t23] | not measured |
| dequant_gguf_bf16::`dequant_{q2_0_gn_to_bf16, q2_k_to_bf16, q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (6) | [gb10/common/dequant_gguf_bf16.cu:43][f24] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t24] | not measured |
| dequant_nvfp4_bf16::`dequant_nvfp4_to_bf16` | [gb10/common/dequant_nvfp4_bf16.cu:50][f25] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t25] | not measured |
| dflash2::`dflash2_{conv2, selector_walk, topk16}` (3) | [gb10/common/dflash2.cu:31][f26] | Speculative decoding · DFlash drafter | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [3 notes][t26] | not measured |
| embed_from_argmax::`batched_embed`, `embed_from_argmax` | [gb10/common/embed_from_argmax.cu:17][f29] | Embedding and LM head · embedding / LM head | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t29] | not measured |
| fp8_gemm_blockscaled_pipe::`fp8_gemm_blockscaled_pipe_128x64` | [gb10/common/fp8_gemm_blockscaled_pipe.cu:63][f30] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| fp8_gemm_t_blockscaled::`fp8_gemm_t_blockscaled` | [gb10/common/fp8_gemm_t_blockscaled.cu:113][f31] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [3 notes][t31] | [14–49%][m31.fp8_gemm_t_blockscaled] (prefill 32k (cold, 32772 tok)) |
| fp8_gemv_rt::`fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2) | [gb10/common/fp8_gemv_rt.cu:156][f32] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t32] | not measured |
| fp8_scale_transpose::`fp8_act_scale_to_kmajor` | [gb10/common/fp8_scale_transpose.cu:35][f33] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t33] | not measured |
| fused_k_norm_rope_cache::`fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2) | [gb10/common/fused_k_norm_rope_cache.cu:53][f34] | KV cache · cache write | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t34] | not measured |
| lora_bgmv::`lora_bgmv_{expand_fold, shrink}` (2) | [gb10/common/lora_bgmv.cu:50][f62] | LoRA adapters · BGMV shrink/expand | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t62] | not measured |
| metadata_fill::`fill_slots_from_block_table` | [gb10/common/metadata_fill.cu:5][f65] | KV cache · cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t65] | not measured |
| moe_lora_gather_bgmv::`moe_lora_gather_bgmv_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_gather_bgmv.cu:57][f76] | LoRA adapters · BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t76] | not measured |
| moe_lora_grouped_down::`moe_lora_grouped_down_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_grouped_down.cu:67][f77] | LoRA adapters · BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t77] | not measured |
| paged_decode::`paged_decode_attn` | [gb10/common/paged_decode_attn.cu:54][f111] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t111] | [11–44%][m111.paged_decode_attn] (decode C=1 (R=2, MTP k=1)) |
| paged_decode_attn_bf16_gqa::`paged_decode_attn_bf16_gqa` | [gb10/common/paged_decode_attn_bf16_gqa.cu:46][f112] | Attention · paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t112] | not measured |
| paged_decode_bf16k_turbo2v::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v.cu:85][f113] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t113] | not measured |
| paged_decode_bf16k_turbo2v_128::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu:85][f114] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t114] | not measured |
| paged_decode_bf16k_turbo3v::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v.cu:91][f115] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t115] | not measured |
| paged_decode_bf16k_turbo3v_128::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu:91][f116] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t116] | not measured |
| paged_decode_bf16k_turbo4v::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v.cu:84][f117] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t117] | not measured |
| paged_decode_bf16k_turbo4v_128::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu:84][f118] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t118] | not measured |
| paged_decode_fp8::`paged_decode_attn_fp8` | [gb10/common/paged_decode_attn_fp8.cu:72][f119] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t119] | [45%][m119.paged_decode_attn_fp8] (decode C=16 (R=32, MTP k=1)) |
| paged_decode_fp8::`paged_decode_attn_{reduce_fp8, splitk_fp8}` (2) | [gb10/common/paged_decode_attn_fp8.cu:346][f119] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t119] | not measured |
| paged_decode_attn_fp8_gqa::`paged_decode_attn_fp8_gqa` | [gb10/common/paged_decode_attn_fp8_gqa.cu:112][f120] | Attention · paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t120] | not measured |
| paged_decode_fp8k_turbo2v::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v.cu:101][f121] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t121] | not measured |
| paged_decode_fp8k_turbo2v_128::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu:101][f122] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t122] | not measured |
| paged_decode_fp8k_turbo3v::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v.cu:112][f123] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t123] | not measured |
| paged_decode_fp8k_turbo3v_128::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu:112][f124] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t124] | not measured |
| paged_decode_fp8k_turbo4v::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v.cu:99][f125] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t125] | not measured |
| paged_decode_fp8k_turbo4v_128::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu:99][f126] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t126] | not measured |
| paged_decode_turbo2::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2.cu:80][f128] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t128] | not measured |
| paged_decode_attn_turbo2_128::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2_128.cu:80][f129] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t129] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3.cu:110][f130] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t130] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3_128.cu:110][f131] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t131] | not measured |
| paged_decode_turbo3k_turbo8v::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v.cu:119][f132] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t132] | not measured |
| paged_decode_turbo3k_turbo8v_128::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu:119][f133] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t133] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4.cu:95][f134] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t134] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_128.cu:95][f135] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t135] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_512.cu:119][f136] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t136] | not measured |
| paged_decode_turbo4k_turbo3v::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v.cu:122][f137] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t137] | not measured |
| paged_decode_turbo4k_turbo3v_128::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu:122][f138] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t138] | not measured |
| paged_decode_turbo4k_turbo8v::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v.cu:114][f139] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t139] | not measured |
| paged_decode_turbo4k_turbo8v_128::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu:114][f140] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t140] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8.cu:100][f141] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t141] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_128.cu:98][f142] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t142] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_512.cu:125][f143] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t143] | not measured |
| per_token_group_quant_fp8::`per_token_group_quant_fp8` | [gb10/common/per_token_group_quant_fp8.cu:39][f144] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t144] | [81–83%][m144.per_token_group_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| prefill_paged::`attn_prefill_{paged, paged_batched, paged_batched_64, paged_fp8, paged_fp8_64, paged_fp8_batched, paged_fp8_batched_64, paged_nvfp4, paged_nvfp4_64, paged_nvfp4_batched, paged_nvfp4_batched_64}` (11) | [gb10/common/prefill_paged_compute.cuh:162][f145] | Attention · prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [8 notes][t145] | not measured |
| prefill_paged::`attn_prefill_paged_64` | [gb10/common/prefill_paged_compute.cuh:644][f145] | Attention · prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t145] | [23–24%][m145.attn_prefill_paged_64] (prefill 32k (cold, 32772 tok)) |
| prefill_paged_indirect::`attn_prefill_paged_{indirect, turbo2, turbo3_64, turbo4, turbo4_64, turbo8_64}` (6) | [gb10/common/prefill_paged_compute.cuh:162][f145] | Attention · prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t145] | not measured |
| attn_prefill_paged_512::`attn_prefill_paged_512` | [gb10/common/prefill_paged_compute_512.cuh:83][f146] | Attention · prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t146] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v_64, bf16k_turbo3v_64, bf16k_turbo4v_64, fp8k_turbo2v_64, fp8k_turbo3v_64, fp8k_turbo4v_64, turbo3k_turbo8v_64, turbo4k_turbo3v_64, turbo4k_turbo8v_64}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:455][f147] | Attention · prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t147] | not measured |
| q2_0_gemv_vec::`q2_0_gemv_vec` | [gb10/common/q2_0_gemv_vec.cu:80][f149] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t149] | not measured |
| quant_rowwise_fp8::`quant_rowwise_fp8` | [gb10/common/quant_rowwise_fp8.cu:38][f150] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t150] | not measured |
| quantize_nvfp4::`f32_to_bf16_trunc` | [gb10/common/quantize_bf16_to_nvfp4.cu:29][f152] | Quantization and format conversion · dtype conversion | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t152] | not measured |
| quantize_nvfp4::`nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `quantize_bf16_to_nvfp4_mse` | [gb10/common/quantize_bf16_to_nvfp4.cu:133][f152] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t152] · [#34][pr34] | not measured |
| reshape_and_cache::`bf16_absmax`, `reshape_and_cache_flash`, `reshape_and_cache_flash_fp8`, `reshape_and_cache_flash_nvfp4`, `reshape_and_cache_flash_v_only` | [gb10/common/reshape_and_cache.cu:30][f154] | KV cache · cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t154] | not measured |
| reshape_and_cache_fused_k_fp8::`fused_k_norm_rope_cache_write_fp8_kv` | [gb10/common/reshape_and_cache_fused_k_fp8.cu:131][f155] | KV cache · cache write | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t155] | not measured |
| reshape_and_cache_turbo::`reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13) | [gb10/common/reshape_and_cache_turbo.cu:179][f156] | KV cache · cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t156] | not measured |
| residual_add::`bf16_concat` | [gb10/common/residual_add.cu:142][f157] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | [4%][m157.bf16_concat] (prefill 4k (cold, 4549 tok)) |
| residual_add::`bf16_residual_add` | [gb10/common/residual_add.cu:10][f157] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | [97–100%][m157.bf16_residual_add] (prefill 4k (cold, 4103 tok)) |
| residual_add::`bf16_scaled_add`, `sigmoid_gate_mul`, `sigmoid_gate_mul_batched`, `sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast` | [gb10/common/residual_add.cu:60][f157] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t157] | not measured |
| rms_norm_vanilla::`rms_norm_{vanilla, vanilla_warp_row}` (2) | [gb10/common/rms_norm_vanilla.cu:38][f159] | Normalization · normalization | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t159] | not measured |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved` | [gb10/common/rope_mrope_interleaved.cu:34][f161] | Positional encoding · rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | [6%][m161.rope_forward_mrope_interleaved] (prefill 32k (cold, 32772 tok)) |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved_k_only` | [gb10/common/rope_mrope_interleaved.cu:108][f161] | Positional encoding · rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | not measured |
| token_overlay::`embed_overlay_routed_bf16`, `embed_rowdiff_bf16`, `lmhead_overlay_routed_bf16`, `lmhead_overlay_routed_f32` | [gb10/common/token_overlay.cu:20][f167] | Embedding and LM head · embedding / LM head | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t167] | not measured |
| tq_plus_innerq_apply::`tq_plus_innerq_apply_{k, q}` (2) | [gb10/common/tq_plus_innerq_apply.cu:71][f168] | KV cache · TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t168] | not measured |
| transpose_u8::`transpose_u8` | [gb10/common/transpose_u8.cu:15][f169] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_fp8_ldmab::`fp8_fp8_gemm_ldmab` | [gb10/common/w4a16_fp8_ldmab.cu:65][f171] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t171] | [11–43%][m171.fp8_fp8_gemm_ldmab] (prefill 32k (cold, 32772 tok)) |
| w4a16_fp8_ldmab::`fp8_predequant_nvfp4_t` | [gb10/common/w4a16_fp8_ldmab.cu:193][f171] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t171] | not measured |
| w4a16_gemv::`w4a16_{gemv, gemv_batch16, gemv_batch2, gemv_batch3, gemv_batch32, gemv_batch8, gemv_batch8_rt2, gemv_dual_batch2, gemv_dual_batch3, gemv_logits, gemv_qg, gemv_qg_batch2, gemv_qg_batch3}` (13) | [gb10/common/w4a16_gemv.cu:167][f173] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t173] | not measured |
| w4a16_gemv::`w4a16_gemv_sw` | [gb10/common/w4a16_gemv.cu:233][f173] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t173] | [96–99%][m173.w4a16_gemv_sw] (decode C=1 (R=4, MTP k=3)) |
| w4a16_gemv_fused::`w4a16_gemv_dual` | [gb10/common/w4a16_gemv_fused.cu:144][f174] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t174] | not measured |
| w4a16_gemv_tc::`w4a16_gemv_tc16` | [gb10/common/w4a16_gemv_tc.cu:256][f175] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t175] · [#1][pr1] | [86–92%][m175.w4a16_gemv_tc16] (decode C=16 (R=32, MTP k=1)) |
| w4a16_gemv_tc::`w4a16_gemv_tc8` | [gb10/common/w4a16_gemv_tc.cu:255][f175] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t175] · [#1][pr1] | [59–90%][m175.w4a16_gemv_tc8] (decode C=1 (R=4, MTP k=3)) |
| w4a4_gemv_mx::`w4a4_{gemv_mx16, gemv_mx16_nt2, gemv_mx16_ps, gemv_mx32, gemv_mx32_nt4, gemv_mx32_ps, gemv_mx64, gemv_mx64_nt2, gemv_mx8, quant_rows}` (10) | [gb10/common/w4a4_gemv_mx.cu:360][f176] | Projection GEMM/GEMV — W4A4 · W4A4 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [23 notes][t176] · [#1][pr1] [#14][pr14] [#18][pr18] | not measured |
| w8a16_gemm::`w8a16_gemm` | [gb10/common/w8a16_gemm.cu:86][f177] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t177] | not measured |
| w8a16_gemm_pipe128::`w8a16_gemm_pipe128` | [gb10/common/w8a16_gemm_pipe128.cu:58][f178] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w8a16_gemm_pipelined::`w8a16_gemm_pipelined` | [gb10/common/w8a16_gemm_pipelined.cu:174][f179] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t179] | [23–24%][m179.w8a16_gemm_pipelined] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m32` | [gb10/common/w8a16_gemm_pipelined_m32.cu:159][f180] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [7 notes][t180] · [#4][pr4] [#34][pr34] | [20–95%][m180.w8a16_gemm_pipelined_m32] (decode C=16 (R=32, MTP k=1)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m64` | [gb10/common/w8a16_gemm_pipelined_m32.cu:318][f180] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t180] · [#4][pr4] [#34][pr34] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [gb10/common/w8a16_gemm_t.cu:607][f181] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t181] | not measured |
| w8a16_gemm_t::`w8a16_gemm_{t, t_pipelined}` (2) | [gb10/common/w8a16_gemm_t.cu:151][f181] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t181] | not measured |
| w8a16_gemm_t_m128::`w8a16_gemm_t_m128` | [gb10/common/w8a16_gemm_t_m128.cu:62][f182] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t182] | [28%][m182.w8a16_gemm_t_m128] (prefill 4k (cold, 4549 tok)) |
| w8a16_gemv::`w8a16_gemv` | [gb10/common/w8a16_gemv.cu:110][f183] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 strix hip | all 14 decoder families (30 ckpts) | [2 notes][t183] | not measured |
| w8a16_gemv_batch4::`w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4) | [gb10/common/w8a16_gemv_batch4.cu:234][f184] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t184] | not measured |
| wht_bf16::`wht_bf16_{inplace, inplace_inv}` (2) | [gb10/common/wht_bf16.cu:51][f186] | KV cache · TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t186] | not measured |
| widen_block_scale_f32::`widen_block_scale_f32` | [gb10/common/widen_block_scale_f32.cu:21][f187] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t187] | not measured |

## Kernels by component

Every entry point with an engine call site, under each component that launches it: a full row under its primary component, and a name under every other component whose code launches it.

### Attention (GQA/MHA: paged decode, split-K, prefill/flash)

336 entry points: 104 primary here (full rows), 232 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| attn_prefill::`attn_prefill` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | not measured |
| attn_prefill_512tc::`attn_prefill_512tc` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [3 notes][t7] | not measured |
| attn_prefill::`attn_prefill_64` | [gb10/common/attn_prefill.cu:562][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | [27%][m7.attn_prefill_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_fa128::`attn_prefill_{fa128, fa128_paged}` (2) | [gb10/common/attn_prefill_fa128.cu:356][f8] | prefill (flash) | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [gb10/common/attn_prefill_h128.cu:46][f10] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t10] | not measured |
| paged_decode::`paged_decode_attn` | [gb10/common/paged_decode_attn.cu:54][f111] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t111] | [11–44%][m111.paged_decode_attn] (decode C=1 (R=2, MTP k=1)) |
| paged_decode_attn_bf16_gqa::`paged_decode_attn_bf16_gqa` | [gb10/common/paged_decode_attn_bf16_gqa.cu:46][f112] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t112] | not measured |
| paged_decode_bf16k_turbo2v::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v.cu:85][f113] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t113] | not measured |
| paged_decode_bf16k_turbo2v_128::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu:85][f114] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t114] | not measured |
| paged_decode_bf16k_turbo3v::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v.cu:91][f115] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t115] | not measured |
| paged_decode_bf16k_turbo3v_128::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu:91][f116] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t116] | not measured |
| paged_decode_bf16k_turbo4v::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v.cu:84][f117] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t117] | not measured |
| paged_decode_bf16k_turbo4v_128::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu:84][f118] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t118] | not measured |
| paged_decode_fp8::`paged_decode_attn_fp8` | [gb10/common/paged_decode_attn_fp8.cu:72][f119] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t119] | [45%][m119.paged_decode_attn_fp8] (decode C=16 (R=32, MTP k=1)) |
| paged_decode_fp8::`paged_decode_attn_{reduce_fp8, splitk_fp8}` (2) | [gb10/common/paged_decode_attn_fp8.cu:346][f119] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t119] | not measured |
| paged_decode_attn_fp8_gqa::`paged_decode_attn_fp8_gqa` | [gb10/common/paged_decode_attn_fp8_gqa.cu:112][f120] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t120] | not measured |
| paged_decode_fp8k_turbo2v::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v.cu:101][f121] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t121] | not measured |
| paged_decode_fp8k_turbo2v_128::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu:101][f122] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t122] | not measured |
| paged_decode_fp8k_turbo3v::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v.cu:112][f123] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t123] | not measured |
| paged_decode_fp8k_turbo3v_128::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu:112][f124] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t124] | not measured |
| paged_decode_fp8k_turbo4v::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v.cu:99][f125] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t125] | not measured |
| paged_decode_fp8k_turbo4v_128::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu:99][f126] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t126] | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/common/paged_decode_attn_nvfp4.cu:87][f127] | paged decode | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (28 ckpts) | [1 note][t127] | not measured |
| paged_decode_turbo2::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2.cu:80][f128] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t128] | not measured |
| paged_decode_attn_turbo2_128::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2_128.cu:80][f129] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t129] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3.cu:110][f130] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t130] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3_128.cu:110][f131] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t131] | not measured |
| paged_decode_turbo3k_turbo8v::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v.cu:119][f132] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t132] | not measured |
| paged_decode_turbo3k_turbo8v_128::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu:119][f133] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t133] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4.cu:95][f134] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t134] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_128.cu:95][f135] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t135] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_512.cu:119][f136] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t136] | not measured |
| paged_decode_turbo4k_turbo3v::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v.cu:122][f137] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t137] | not measured |
| paged_decode_turbo4k_turbo3v_128::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu:122][f138] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t138] | not measured |
| paged_decode_turbo4k_turbo8v::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v.cu:114][f139] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t139] | not measured |
| paged_decode_turbo4k_turbo8v_128::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu:114][f140] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t140] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8.cu:100][f141] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t141] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_128.cu:98][f142] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t142] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_512.cu:125][f143] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t143] | not measured |
| prefill_paged::`attn_prefill_{paged, paged_batched, paged_batched_64, paged_fp8, paged_fp8_64, paged_fp8_batched, paged_fp8_batched_64, paged_nvfp4, paged_nvfp4_64, paged_nvfp4_batched, paged_nvfp4_batched_64}` (11) | [gb10/common/prefill_paged_compute.cuh:162][f145] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [8 notes][t145] | not measured |
| prefill_paged::`attn_prefill_paged_64` | [gb10/common/prefill_paged_compute.cuh:644][f145] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t145] | [23–24%][m145.attn_prefill_paged_64] (prefill 32k (cold, 32772 tok)) |
| prefill_paged_indirect::`attn_prefill_paged_{indirect, turbo2, turbo3_64, turbo4, turbo4_64, turbo8_64}` (6) | [gb10/common/prefill_paged_compute.cuh:162][f145] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t145] | not measured |
| attn_prefill_paged_512::`attn_prefill_paged_512` | [gb10/common/prefill_paged_compute_512.cuh:83][f146] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t146] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v_64, bf16k_turbo3v_64, bf16k_turbo4v_64, fp8k_turbo2v_64, fp8k_turbo3v_64, fp8k_turbo4v_64, turbo3k_turbo8v_64, turbo4k_turbo3v_64, turbo4k_turbo8v_64}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:455][f147] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t147] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/deepseek-v4-flash/nvfp4/attn_prefill_512.cu:13][f188] | prefill (flash) | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t188] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu:43][f205] | paged decode | b200 gb10 hop | DeepSeek-V4, Gemma4 (4 ckpts) | — | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu:87][f208] | paged decode | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t208] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512.cu:13][f211] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [2 notes][t211] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu:43][f220] | paged decode | gb10 | Gemma4 (2 ckpts) | — | not measured |
| paged_decode_attn_fp8_512::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu:46][f221] | paged decode | gb10 | Gemma4 (2 ckpts) | [1 note][t221] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu:12][f223] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [1 note][t223] | not measured |
| paged_decode_bf16_splitk_hopper::`paged_decode_attn_{reduce_bf16_hopper, splitk_bf16_hopper}` (2) | [hopper/common/paged_decode_bf16_splitk_hopper.cu:30][f280] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t280] | not measured |
| paged_decode_fp8_splitk_hopper::`paged_decode_attn_{reduce_fp8_hopper, splitk_fp8_hopper}` (2) | [hopper/common/paged_decode_fp8_splitk_hopper.cu:47][f281] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t281] | not measured |
| attention_decode::`attention_decode` | [metal/common/attention_decode.metal:30][f289] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t289] | not measured |
| attention_decode_bf16k_turbov::`attention_decode_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/attention_decode_bf16k_turbov.metal:113][f290] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t290] | not measured |
| attention_decode_turbo2::`attention_decode_turbo2` | [metal/common/attention_decode_turbo2.metal:39][f291] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t291] | not measured |
| attention_decode_turbo3::`attention_decode_turbo3` | [metal/common/attention_decode_turbo3.metal:53][f292] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t292] | not measured |
| attention_decode_turbo4::`attention_decode_turbo4` | [metal/common/attention_decode_turbo4.metal:41][f293] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t293] | not measured |
| attention_decode_turbo8::`attention_decode_turbo8` | [metal/common/attention_decode_turbo8.metal:38][f294] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t294] | not measured |
| attn_prefill::`attn_{prefill, prefill_64}` (2) | [strix-hip/common/attn_prefill.cu:69][f330] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t330] | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [strix-hip/common/attn_prefill_h128.cu:185][f332] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t332] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_residual_add`, `sigmoid_gate_mul`, `sigmoid_gate_mul_batched`, `sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast`, `sigmoid_gate`; [Embedding and LM head](#embedding-and-lm-head-lookup-overlays-softcap-scale): `bf16_scale_inplace`, `bf16_scale_inplace`; [GDN](#gdn-gated-delta-rule-linear-attention): `deinterleave_qg`, `deinterleave_qg_{split, split_qnorm_mrope}` (2), `deinterleave_qg_split_qnorm`; [Hyper-connections](#hyper-connections-mhc): `hc_expand`, `hc_head`, `hc_post`, `hc_pre`, `hc_expand`, `hc_head`, `hc_post`, `hc_pre`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2), `reshape_and_cache_{flash, flash_fp8, flash_nvfp4, flash_v_only}` (4), `fused_k_norm_rope_cache_write_fp8_kv`, `reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13), `tq_plus_innerq_apply_{k, q}` (2), `wht_bf16_{inplace, inplace_inv}` (2), `wht_bf16_{inplace, inplace_inv}` (2); [MLA](#mla-multi-head-latent-attention): `grouped_gemm_mla`, `mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9), `mla_fused_prefill`, `mla_paged_decode_nvfp4`, `mla_paged_decode_fp8`, `mla_prefill_attn_320`, `paged_decode_attn_{fp8, splitk_fp8}` (2), `paged_decode_attn`, `mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9), `mla_fused_prefill`, `mla_prefill_attn_320`, `paged_decode_attn_{fp8, splitk_fp8}` (2), `paged_decode_attn`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `residual_add_rms_norm_vanilla`, `rms_norm`, `rms_norm_residual_vanilla`, `rms_norm_strided`, `rms_norm_residual`, `rms_norm_{vanilla, vanilla_warp_row}` (2), `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`; [Positional encoding](#positional-encoding-rope-yarn-mrope): `rope_{forward, forward_proportional, forward_strided, forward_yarn, forward_yarn_interleaved, forward_yarn_interleaved_inv, forward_yarn_scaled}` (7), `rope_forward_mrope_interleaved`, `rope_forward_mrope_interleaved_k_only`, `rope_{forward, forward_yarn}` (2); [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemm_splitk_{partial, reduce}` (2), `dense_gemm_tc`, `dense_gemv_bf16`, `dense_gemv_bf16_batchm`, `dense_gemm_{bf16, bf16_pipelined}` (2), `dense_gemm_tc`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4), `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemm_pipelined_m32`, `w8a16_gemm_pipelined_m64`, `w8a16_gemm_{t, t_pipelined}` (2), `w8a16_gemm_t_m128`, `w8a16_gemv`, `w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t}` (3), `fp8_gemm_t_m128`, `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `w8a16_gemm_{m16, m16_strided}` (2), `w8a16_gemv`, `w8a16_gemv_batch16_{ncol2, ncol2_strided, ncol4, ncol4_strided}` (4), `w8a16_gemm`, `w8a16_gemm_t`, `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4); [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch2, gemv_batch3, gemv_dual_batch2, gemv_dual_batch3, gemv_qg, gemv_qg_batch2, gemv_qg_batch3}` (8), `w4a16_gemv_sw`, `w4a16_gemv_dual`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128_bf16}` (2), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm_mfast`; [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `q2_0_gemv_vec`, `metrale_q2_0_mmq128_{nc, wc}` (2); [Quantization and format conversion](#quantization-and-format-conversion): `dequant_q2_0_gn_to_bf16`, `fp8_act_scale_to_kmajor`, `quantize_bf16_to_nvfp4`, `transpose_{block_scale, fp8}` (2), `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `metrale_q8_1_quantize_ds4_bf16`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `transpose_{block_scale, fp8}` (2), `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`; [Sparse / compressed attention](#sparse-compressed-attention-dsa-csa-hca-qsa): `csa_compress`, `prefill_attn_compressed`.

### MLA (multi-head latent attention)

34 entry points: 34 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| grouped_gemm_mla::`grouped_gemm_mla` | [gb10/deepseek-v4-flash/nvfp4/grouped_gemm_mla.cu:35][f192] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat, Mistral4 (4 ckpts) | [2 notes][t192] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:33][f196] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t196] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu:21][f198] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t198] | not measured |
| mla_paged_decode::`mla_paged_decode_nvfp4` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu:78][f199] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t199] | not measured |
| mla_paged_decode_fp8::`mla_paged_decode_fp8` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu:38][f200] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t200] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu:26][f201] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t201] | not measured |
| paged_decode_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu:66][f206] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t206] | not measured |
| paged_decode_mla::`paged_decode_attn` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu:59][f207] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t207] | not measured |
| glm5next_mla_latent_write::`glm5next_mla_latent_write_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_mla_latent_write.cu:30][f228] | MLA decode/prefill | gb10 | GLM-5.3 (1 ckpts) | [1 note][t228] | not measured |
| mla_decode::`k3_mla_{maybe_rope_f32, sdpa_gate_f32}` (2) | [gb10/kimi-k3/bf16/mla_decode.cu:41][f232] | MLA decode/prefill | b200 b300 gb10 | Kimi-K3 (1 ckpts) | [1 note][t232] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/mistral-small-4/nvfp4/mla_absorbed.cu:33][f237] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t237] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu:21][f238] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t238] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu:24][f239] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t239] | not measured |
| paged_decode_attn_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu:66][f240] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t240] | not measured |
| paged_decode_mla::`paged_decode_attn` | [gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu:59][f241] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t241] | not measured |

### Sparse / compressed attention (DSA, CSA/HCA, QSA)

61 entry points: 43 primary here (full rows), 18 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [b300/common/dsa_indexer.cu:74][f2] | DSA indexer / sparse MLA | b300 | none — its callers' targets compile another copy | [1 note][t2] | not measured |
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [gb10/common/dsa_indexer.cu:74][f27] | DSA indexer / sparse MLA | b200 gb10 hop | GLM-5.3 (1 ckpts) | [5 notes][t27] | not measured |
| attn_v41::`attn_v41_{act_quant_fp8, fp4_quant, gemm_f32, gemv_f32_staged, index_score, pool, ring_put, rmsnorm_bf16, rmsnorm_f32, rope, scale_bf16, scatter_cols, slice_cols, sparse_attn}` (14) | [gb10/deepseek-v4-flash/nvfp4/attn_v41.cu:100][f189] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [6 notes][t189] | not measured |
| csa_compress::`csa_compress` | [gb10/deepseek-v4-flash/nvfp4/csa_compress.cu:20][f190] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t190] | not measured |
| prefill_attn_compressed::`prefill_attn_compressed` | [gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu:23][f209] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t209] | not measured |
| glm5next_dsa_mla_decode::`glm5next_dsa_mla_decode_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu:94][f227] | DSA indexer / sparse MLA | gb10 | GLM-5.3 (1 ckpts) | [1 note][t227] | not measured |
| qsa_indexer::`qsa_{block_pool, gather, prefill_attn, qprep, qprep_rows, score, score_rows, score_rows_tc}` (8) | [gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu:70][f271] | QSA sparse attention | gb10 | Qwen3.8-FN (1 ckpts) | [3 notes][t271] | not measured |

Also launched here: [Encoder-decoder translation](#encoder-decoder-translation-nllb-self-contained-kernel-set): `nllb_layernorm_bf16`, `nllb_layernorm_bf16`; [MLA](#mla-multi-head-latent-attention): `glm5next_mla_latent_write_fp8`; [MoE](#moe-routing-dispatch-expert-gemm-gemv-combine): `kquant_mmvq_q2_k_{groups_w, pair_w}` (2); [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `rms_norm_vanilla`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_f32out`, `dense_gemv_bf16`, `dense_gemv_bf16_fp32out`, `dense_gemv_bf16_batchm`, `dense_gemm_{bf16, bf16_f32out}` (2); [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `kquant_mmvq_q2_k_w`, `metrale_q2_k_mmq128_nc`, `metrale_q2_k_mmq128_wc`; [Quantization and format conversion](#quantization-and-format-conversion): `kquant_q8_1_rows_bf16`, `metrale_q8_1_quantize_d2s6_bf16`.

### GDN (gated delta rule linear attention)

335 entry points: 187 primary here (full rows), 148 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill}` (9) | [gb10/common/gated_delta_rule.cu:78][f35] | delta-rule recurrence | b200 b300 gb10 hop | none — its callers' targets compile another copy | [6 notes][t35] | not measured |
| gated_delta_rule_carry::`gdn_carry_conv` | [gb10/common/gated_delta_rule_carry.cu:343][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [33%][m36.gdn_carry_conv] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_carry::`gdn_carry_{conv_flush, flush, wy2, wy3, wy3_lazy, wy4, wy4_lazy}` (7) | [gb10/common/gated_delta_rule_carry.cu:283][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | not measured |
| gated_delta_rule_carry::`gdn_carry_wy2_lazy` | [gb10/common/gated_delta_rule_carry.cu:286][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [51%][m36.gdn_carry_wy2_lazy] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_{ksplit, pipe, tc_vblock, tma, vtile}` (5) | [gb10/common/gated_delta_rule_fla.cu:857][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [10 notes][t37] | not measured |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_vfused` | [gb10/common/gated_delta_rule_fla.cu:1085][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [25–36%][m37.gated_delta_rule_chunk_delta_h_vfused] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_fwd_o` | [gb10/common/gated_delta_rule_fla.cu:2003][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [32–33%][m37.gated_delta_rule_chunk_fwd_o] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_recompute_wu` | [gb10/common/gated_delta_rule_fla.cu:262][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [11 notes][t37] | [34–40%][m37.gated_delta_rule_recompute_wu] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_persistent::`gated_delta_rule_prefill_{persistent, persistent_batched, persistent_wy4, persistent_wy4_batched}` (4) | [gb10/common/gated_delta_rule_persistent.cu:54][f38] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t38] | not measured |
| gated_delta_rule_regresident::`gated_delta_rule_prefill_regresident` | [gb10/common/gated_delta_rule_regresident.cu:46][f39] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t39] | not measured |
| gated_delta_rule_wy::`gated_delta_rule_wy2` | [gb10/common/gated_delta_rule_wy.cu:30][f40] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t40] | [82%][m40.gated_delta_rule_wy2] (decode C=1 (R=2, MTP k=1)) |
| gated_delta_rule_wy2_resident::`gated_delta_rule_wy2_resident` | [gb10/common/gated_delta_rule_wy2_resident.cu:58][f41] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t41] | not measured |
| gated_delta_rule_wy2_resident_f16::`gated_delta_rule_wy2_resident_f16` | [gb10/common/gated_delta_rule_wy2_resident_f16.cu:65][f42] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t42] | [58%][m42.gated_delta_rule_wy2_resident_f16] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_wy3::`gated_delta_rule_wy3` | [gb10/common/gated_delta_rule_wy3.cu:18][f43] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t43] | not measured |
| gated_delta_rule_wy3_f16::`gated_delta_rule_wy3_f16` | [gb10/common/gated_delta_rule_wy3_f16.cu:40][f44] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t44] | not measured |
| gated_delta_rule_wy3_resident::`gated_delta_rule_wy3_resident` | [gb10/common/gated_delta_rule_wy3_resident.cu:59][f45] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t45] | not measured |
| gated_delta_rule_wy3_resident_f16::`gated_delta_rule_wy3_resident_f16` | [gb10/common/gated_delta_rule_wy3_resident_f16.cu:48][f46] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t46] | not measured |
| gated_delta_rule_wy4::`gated_delta_rule_wy4` | [gb10/common/gated_delta_rule_wy4.cu:18][f47] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t47] | [58%][m47.gated_delta_rule_wy4] (decode C=1 (R=4, MTP k=3)) |
| gated_delta_rule_wy4_f16::`gated_delta_rule_wy4_f16` | [gb10/common/gated_delta_rule_wy4_f16.cu:40][f48] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t48] | not measured |
| gated_delta_rule_wy4_woa::`gated_delta_rule_wy4_{flag_clear, fold, woa}` (3) | [gb10/common/gated_delta_rule_wy4_woa.cu:55][f49] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t49] | not measured |
| gated_delta_rule_wy64_prefill::`gated_delta_rule_prefill_{wy64, wy64_batched}` (2) | [gb10/common/gated_delta_rule_wy64_prefill.cu:41][f50] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t50] | not measured |
| gated_delta_rule_wy_f16::`gated_delta_rule_wy2_f16` | [gb10/common/gated_delta_rule_wy_f16.cu:45][f51] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t51] | not measured |
| gated_delta_rule_wyn::`gated_delta_rule_{wy10, wy10_f16, wy10_f16_table, wy10_table, wy11, wy11_f16, wy11_f16_table, wy11_table, wy12, wy12_f16, wy12_f16_table, wy12_table, wy13, wy13_f16, wy13_f16_table, wy13_table, wy14, wy14_f16, wy14_f16_table, wy14_table, wy15, wy15_f16, wy15_f16_table, wy15_table, wy16, wy16_f16, wy16_f16_table, wy16_table, wy5, wy5_f16, wy5_f16_table, wy5_table, wy6, wy6_f16, wy6_f16_table, wy6_table, wy7, wy7_f16, wy7_f16_table, wy7_table, wy8, wy8_f16, wy8_f16_table, wy8_table, wy9, wy9_f16, wy9_f16_table, wy9_table}` (48) | [gb10/common/gated_delta_rule_wyn.cu:293][f52] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t52] | not measured |
| gdn_chunk_fwd_o_mma8::`gated_delta_rule_chunk_fwd_o_mma8` | [gb10/common/gdn_chunk_fwd_o_mma8.cu:96][f53] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn` | [gb10/common/gdn_verify_fused_conv_kn.cu:50][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn_batched` | [gb10/common/gdn_verify_fused_conv_kn.cu:157][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | [35%][m54.gdn_verify_fused_conv_kn_batched] (decode C=16 (R=32, MTP k=1)) |
| gdn_verify_fused_k2::`gdn_verify_fused_{conv_k2, norm_k2}` (2) | [gb10/common/gdn_verify_fused_k2.cu:60][f55] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t55] | not measured |
| ssm_ba_gates_hopper::`dense_gemm_ba_gates_prefill_hopper` | [gb10/common/ssm_ba_gates_hopper.cu:115][f162] | GDN pre/post-processing | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t162] | not measured |
| ssm_h_dtype::`ssm_h_state_{f16_to_f32, f32_to_f16}` (2) | [gb10/common/ssm_h_dtype.cu:25][f164] | GDN pre/post-processing | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t164] | not measured |
| ssm_preprocess::`compute_gdn_gates`, `deinterleave_qg_split`, `deinterleave_qg_split_qnorm_mrope`, `deinterleave_qkvz`, `dense_gemv_ba_gates` | [gb10/common/ssm_preprocess.cu:35][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| ssm_preprocess::`deinterleave_qg` | [gb10/common/ssm_preprocess.cu:90][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | [3–50%][m165.deinterleave_qg] (decode C=1 (R=2, MTP k=1)) |
| ssm_preprocess::`deinterleave_qg_split_qnorm` | [gb10/common/ssm_preprocess.cu:190][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | [88–90%][m165.deinterleave_qg_split_qnorm] (prefill 4k (cold, 4103 tok)) |
| ssm_preprocess::`dense_gemm_ba_gates_prefill` | [gb10/common/ssm_preprocess.cu:482][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t165] | [8–43%][m165.dense_gemm_ba_gates_prefill] (prefill 32k (cold, 32772 tok)) |
| ssm_state_norm::`ssm_state_clamp_norm_{fused, fused_f16}` (2) | [gb10/common/ssm_state_norm.cu:32][f166] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t166] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/gemma-4-26b-a4b/nvfp4/gated_delta_rule.cu:24][f213] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t213] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/qwen3-next-80b-a3b/nvfp4/gated_delta_rule.cu:24][f248] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t248] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/qwen3.5-122b-a10b/nvfp4/gated_delta_rule.cu:24][f251] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t251] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f16_norm, decode_f16_strided_norm_half, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, decode_f32_strided_norm_half, decode_f32_strided_norm_smem, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (16) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu:24][f252] | delta-rule recurrence | gb10 hop strix hip | Qwen-GDN (7 ckpts) | [6 notes][t252] | not measured |
| gated_delta_rule_snap::`gated_delta_rule_decode_f32_{norm_snap, strided_norm_snap}` (2) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule_snap.cu:66][f253] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t253] | not measured |
| gdn_verify_fused_conv_kn_f32::`gdn_verify_fused_conv_kn_f32` | [gb10/qwen3.6-27b/nvfp4/gdn_verify_fused_conv_kn_f32.cu:33][f254] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t254] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu:59][f263] | delta-rule recurrence | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t263] | not measured |
| gated_delta_rule_wy17::`gated_delta_rule_wy17` | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule_wy17.cu:41][f264] | delta-rule recurrence | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t264] | not measured |
| gated_delta_rule_chunk_tc::`gated_delta_rule_chunk_delta_h_tcfuse_x2` | [hopper/common/gated_delta_rule_chunk_tc.cu:409][f275] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [5 notes][t275] | not measured |
| gdn_fwd_o_hopper::`gated_delta_rule_chunk_fwd_o_hopper` | [hopper/common/gdn_fwd_o_hopper.cu:94][f276] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t276] | not measured |
| gdn_recompute_wu_hopper::`gated_delta_rule_recompute_wu_hopper` | [hopper/common/gdn_recompute_wu_hopper.cu:138][f277] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t277] | not measured |
| gated_delta_rule_decode::`gated_delta_rule_decode` | [metal/common/gated_delta_rule_decode.metal:38][f304] | delta-rule recurrence | metal | Qwen-GDN (7 ckpts) | [1 note][t304] | not measured |
| gdn_helpers::`gdn_compute_gate` | [metal/common/gdn_helpers.metal:28][f305] | delta-rule recurrence | metal | Qwen-GDN (7 ckpts) | — | not measured |
| gdn_helpers::`sigmoid_bf16_to_f32` | [metal/common/gdn_helpers.metal:56][f305] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |
| qwen35_qkv_split::`qwen35_qkv_split` | [metal/common/qwen35_qkv_split.metal:20][f322] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_residual_add`, `sigmoid_gate`; [Attention](#attention-gqa-mha-paged-decode-split-k-prefill-flash): `attention_decode`, `attention_decode_bf16k_{turbo2v, turbo3v, turbo4v}` (3), `attention_decode_turbo2`, `attention_decode_turbo3`, `attention_decode_turbo4`, `attention_decode_turbo8`; [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `causal_conv1d_update`, `causal_conv1d_update_{chunk2, l2norm_f32, l2norm_f32_strided}` (3), `causal_conv1d_update_l2norm`, `causal_conv1d_update_prefill`, `causal_conv1d_update_prefill_tp`, `causal_conv1d_update_l2norm`; [Hyper-connections](#hyper-connections-mhc): `hc_expand`, `hc_post`, `hc_pre`, `hc_expand`, `hc_post`, `hc_pre`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `wht_bf16_{inplace, inplace_inv}` (2), `kv_cache_append`, `kv_cache_append_bf16k_{turbo2v, turbo3v, turbo4v}` (3), `kv_cache_append_turbo2`, `kv_cache_append_turbo3`, `kv_cache_append_turbo4`, `kv_cache_append_turbo8`, `wht_bf16_{inplace, inplace_inv}` (2); [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `gated_rms_{norm, norm_f32_input, norm_f32_input_strided}` (3), `gated_rms_norm_prefill`, `l2_norm_bf16`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `gated_rms_norm`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `gated_rms_norm_{f32_input_sigmoid, prefill_sigmoid, sigmoid}` (3), `add_rms_norm`, `rms_norm`; [Positional encoding](#positional-encoding-rope-yarn-mrope): `rope_apply`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16_batch2`, `dense_gemm_{bf16, bf16_pipelined}` (2); [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemv_{batch16, batch4}` (2), `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemm_pipelined_m32`, `w8a16_gemm_pipelined_m64`, `w8a16_gemm_t`, `w8a16_gemv`, `w8a16_gemv_{batch16, batch4}` (2), `fp8_gemm_{t, t_m128}` (2), `fp8_gemm_{t, t_m128}` (2), `fp8_gemm_t`, `fp8_gemm_t_m128`, `fp8_gemm_{t, t_m128}` (2), `w8a16_gemv`, `w8a16_gemm`, `w8a16_gemm_t`, `fp8_gemm_{t, t_m128}` (2), `fp8_gemm_{t, t_m128}` (2); [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch16, gemv_batch2, gemv_batch3}` (4), `w4a16_gemv_qkvz`, `w4a16_gemv_sw`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_gemm`, `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2); [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `q2_0_gemv_vec`, `metrale_q2_0_mmq128_{nc, wc}` (2); [Quantization and format conversion](#quantization-and-format-conversion): `dequant_q2_0_gn_to_bf16`, `fp8_act_scale_to_kmajor`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`, `metrale_q8_1_quantize_ds4_bf16`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`.

### KDA (Kimi delta attention, linear attention)

24 entry points: 11 primary here (full rows), 13 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| kda_chunk::`kda_chunk_{prepare, scan}` (2) | [gb10/common/kda_chunk.cu:95][f58] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t58] | not measured |
| kda_gate::`kda_gate_bf16` | [gb10/common/kda_gate.cu:74][f59] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | — | not measured |
| kda_layer_ops::`kda_{fill_f32, o_norm_gated_bf16, pack_qkv_bf16, sigmoid_bf16_f32, split_widen}` (5) | [gb10/common/kda_layer_ops.cu:50][f60] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t60] | not measured |
| kda_recurrent::`kda_recurrent_decode_{bf16, bf16_smem}` (2) | [gb10/common/kda_recurrent.cu:144][f61] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t61] | not measured |
| kda_decode::`k3_kda_recurrent_step_f32` | [gb10/kimi-k3/bf16/kda_decode.cu:60][f231] | KDA op | b200 b300 gb10 | Kimi-K3 (1 ckpts) | [1 note][t231] | not measured |

Also launched here: [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `causal_conv1d_update_l2norm`, `causal_conv1d_update_prefill`, `k3_kda_conv_update_f32`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemv_bf16`, `dense_gemv_bf16_batchm`, `dense_gemm_bf16`.

### Mamba2 (selective state-space scan)

63 entry points: 6 primary here (full rows), 57 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| mamba2_ssd_chunk::`mamba2_ssd_{bmm, cumsum, scan}` (3) | [gb10/common/mamba2_ssd_chunk.cu:44][f63] | SSD / selective scan | b200 b300 gb10 hop | Nemotron-H (3 ckpts) | [2 notes][t63] | not measured |
| mamba2_ssm::`mamba2_ssm_{decode, prefill, prefill_persistent}` (3) | [gb10/common/mamba2_ssm_decode.cu:28][f64] | SSD / selective scan | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [2 notes][t64] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_residual_add`; [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `causal_conv1d_update`, `causal_conv1d_update_prefill`, `causal_conv1d_update_prefill_tp`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16`, `dense_gemm_bf16_pipelined`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemv`, `fp8_{fp8_gemm_t_m128_mfast, gemm_t_m128_mfast}` (2), `w8a16_gemv`, `w8a16_gemm`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_{gemm, gemm_t}` (2), `w4a16_gemv`, `w4a16_gemv_sw`, `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t}` (2), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_gemm_t`, `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm_mfast`; [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_nvfp4`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`.

### Causal conv1d (short convolution of GDN/KDA/Mamba2)

9 entry points: 9 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| causal_conv1d::`causal_conv1d_update` | [gb10/common/causal_conv1d.cu:96][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (17 ckpts) | [1 note][t13] | not measured |
| causal_conv1d::`causal_conv1d_update_{chunk2, l2norm_f32, l2norm_f32_strided}` (3) | [gb10/common/causal_conv1d.cu:247][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t13] | not measured |
| causal_conv1d::`causal_conv1d_update_l2norm` | [gb10/common/causal_conv1d.cu:327][f13] | causal conv1d | b200 b300 gb10 hop strix hip | GLM-5.3, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (15 ckpts) | [1 note][t13] | [33–54%][m13.causal_conv1d_update_l2norm] (decode C=1 (R=2, MTP k=1)) |
| causal_conv1d::`causal_conv1d_update_prefill` | [gb10/common/causal_conv1d.cu:184][f13] | causal conv1d | b200 b300 gb10 hop strix hip | GLM-5.3, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (18 ckpts) | [2 notes][t13] | not measured |
| causal_conv1d::`causal_conv1d_update_prefill_tp` | [gb10/common/causal_conv1d.cu:581][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (17 ckpts) | [3 notes][t13] | [94–96%][m13.causal_conv1d_update_prefill_tp] (prefill 32k (cold, 32772 tok)) |
| kda_decode::`k3_kda_conv_update_f32` | [gb10/kimi-k3/bf16/kda_decode.cu:21][f231] | causal conv1d | b200 b300 gb10 | Kimi-K3 (1 ckpts) | [1 note][t231] | not measured |
| causal_conv1d_update_l2norm::`causal_conv1d_update_l2norm` | [metal/common/causal_conv1d_update_l2norm.metal:41][f299] | causal conv1d | metal | Qwen-GDN (7 ckpts) | [1 note][t299] | not measured |

### MoE (routing, dispatch, expert GEMM/GEMV, combine)

284 entry points: 187 primary here (full rows), 97 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [b300/common/moe_shared_expert_fused.cu:48][f3] | expert GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [2 notes][t3] | not measured |
| gemm::`dense_gemm_bf16_router` | [gb10/common/dense_gemm_bf16.cu:204][f14] | routing / top-k | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t14] | [6%][m14.dense_gemm_bf16_router] (prefill 32k (cold, 32772 tok)) |
| glm5next_ffn::`glm5next_moe_{combine, combine_indexed}` (2) | [gb10/common/glm5next_ffn.cu:213][f56] | dispatch / combine | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_router_topk` | [gb10/common/glm5next_ffn.cu:102][f56] | routing / top-k | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_swiglu_clamp` | [gb10/common/glm5next_ffn.cu:31][f56] | expert activation | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| moe_bf16_grouped_gemm::`moe_bf16_grouped_gemm` | [gb10/common/moe_bf16_grouped_gemm.cu:86][f66] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t66] | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_finalize` | [gb10/common/moe_decode_atomic_c4.cu:183][f67] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t67] | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_silu_down_accum` | [gb10/common/moe_decode_atomic_c4.cu:37][f67] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t67] | not measured |
| moe_expert_gemv::`moe_expert_gemv` | [gb10/common/moe_expert_gemv.cu:57][f68] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t68] | not measured |
| moe_expert_gemv::`moe_weighted_sum_blend` | [gb10/common/moe_expert_gemv.cu:195][f68] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t68] | not measured |
| moe_relu2_fused::`moe_expert_relu2_down_shared` | [gb10/common/moe_expert_relu2_down_shared.cu:53][f70] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t70] | not measured |
| moe_fp8_grouped_blend::`moe_weighted_sum_blend_fp8_grouped` | [gb10/common/moe_fp8_grouped_blend.cu:17][f71] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t71] · [#4][pr4] | [16–77%][m71.moe_weighted_sum_blend_fp8_grouped] (decode C=1 (R=2, MTP k=1)) |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [gb10/common/moe_fp8_grouped_gemm.cu:281][f72] | expert GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t72] | not measured |
| moe_fp8_grouped_sort::`moe_fp8_grouped_sort` | [gb10/common/moe_fp8_grouped_sort.cu:24][f73] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t73] · [#34][pr34] | [1%][m73.moe_fp8_grouped_sort] (decode C=1 (R=2, MTP k=1)) |
| moe_gate_topk::`moe_gate_topk_fused` | [gb10/common/moe_gate_topk.cu:46][f74] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t74] | not measured |
| moe_hash_route::`moe_hash_{route, route_batched}` (2) | [gb10/common/moe_hash_route.cu:25][f75] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t75] | not measured |
| moe_nvfp4_grouped::`moe_expert_{down_act_nvfp4_grouped, gate_up_act_nvfp4_grouped}` (2) | [gb10/common/moe_nvfp4_grouped.cu:72][f78] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [5 notes][t78] · [#34][pr34] | not measured |
| moe::`moe_batched_blend` | [gb10/common/moe_permute.cu:126][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | [97–100%][m79.moe_batched_blend] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_build_tile_worklist` | [gb10/common/moe_permute.cu:276][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | [0%][m79.moe_build_tile_worklist] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_{permute_tokens, sort_by_expert}` (2) | [gb10/common/moe_permute.cu:20][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | not measured |
| moe::`moe_unpermute_reduce_indexed` | [gb10/common/moe_permute.cu:95][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | [87–88%][m79.moe_unpermute_reduce_indexed] (prefill 32k (cold, 32772 tok)) |
| moe_prefill::`moe_expert_{gate_up_shared_prefill, silu_down_shared_prefill}` (2) | [gb10/common/moe_prefill.cu:59][f80] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t80] | not measured |
| moe_prefill::`moe_weighted_sum_blend_prefill` | [gb10/common/moe_prefill.cu:360][f80] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t80] | not measured |
| moe_router_gemm::`moe_router_gemm_bf16` | [gb10/common/moe_router_gemm.cu:32][f81] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t81] · [#34][pr34] | not measured |
| moe_router_gemm_prefill::`moe_router_gemm_rt` | [gb10/common/moe_router_gemm_prefill.cu:24][f82] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/common/moe_shared_expert_fused.cu:48][f83] | expert GEMM/GEMV | b200 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t83] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/common/moe_shared_expert_fused_batch2.cu:292][f84] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t84] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/common/moe_shared_expert_fused_batch2.cu:497][f84] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t84] | not measured |
| moe_shared_expert_fused_batch2_t::`moe_expert_{gate_up_shared_batch2_t, silu_down_shared_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_batch2_t.cu:41][f85] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t85] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/common/moe_shared_expert_fused_batch3.cu:50][f86] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t86] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/common/moe_shared_expert_fused_batch3.cu:335][f86] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t86] | not measured |
| moe_shared_expert_fused_batch3_t::`moe_expert_{gate_up_shared_batch3_t, silu_down_shared_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_batch3_t.cu:36][f87] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | not measured |
| moe_shared_expert_fused_bf16::`moe_expert_{gate_up_shared_bf16, silu_down_shared_bf16}` (2) | [gb10/common/moe_shared_expert_fused_bf16.cu:26][f88] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t88] | not measured |
| moe_shared_expert_fused_bf16_batch2::`moe_expert_{gate_up_shared_bf16_batch2, silu_down_shared_bf16_batch2}` (2) | [gb10/common/moe_shared_expert_fused_bf16_batch2.cu:39][f89] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t89] | not measured |
| moe_shared_expert_fused_fp8::`moe_expert_gate_up_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:98][f90] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t90] | [85%][m90.moe_expert_gate_up_shared_fp8] (decode C=1 (R=2, MTP k=1)) |
| moe_shared_expert_fused_fp8::`moe_expert_silu_down_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:262][f90] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t90] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_expert_{gate_up_shared_fp8_batch2, silu_down_shared_fp8_batch2}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:97][f91] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t91] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_weighted_sum_blend_fp8_batch2` | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:410][f91] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t91] | not measured |
| moe_shared_expert_fused_fp8_batch2_t::`moe_expert_{gate_up_shared_fp8_batch2_t, silu_down_shared_fp8_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2_t.cu:28][f92] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t92] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_expert_{gate_up_shared_fp8_batch3, silu_down_shared_fp8_batch3}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:97][f93] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t93] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_weighted_sum_blend_fp8_batch3` | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:408][f93] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t93] | not measured |
| moe_shared_expert_fused_fp8_batch3_t::`moe_expert_{gate_up_shared_fp8_batch3_t, silu_down_shared_fp8_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3_t.cu:28][f94] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t94] | not measured |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_down_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:331][f95] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [12 notes][t95] · [#4][pr4] [#34][pr34] | [89–102%][m95.moe_expert_down_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_gate_up_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:153][f95] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [11 notes][t95] · [#4][pr4] [#34][pr34] | [88–97%][m95.moe_expert_gate_up_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_t::`moe_expert_{gate_up_shared_fp8_t, silu_down_shared_fp8_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_t.cu:34][f96] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t96] | not measured |
| moe_shared_expert_fused_t::`moe_expert_{gate_up_shared_t, gate_up_shared_t_e8m0, silu_down_shared_t, silu_down_shared_t_e8m0}` (4) | [gb10/common/moe_shared_expert_fused_t.cu:187][f97] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t97] | not measured |
| moe_silu_mul::`moe_silu_mul` | [gb10/common/moe_silu_mul.cu:38][f98] | expert activation | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN (19 ckpts) | — | not measured |
| moe_sorted::`moe_sorted_{gate_up, silu_down}` (2) | [gb10/common/moe_sorted_prefill.cu:54][f99] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t99] | not measured |
| moe_topk::`moe_topk_{softmax, softmax_batched, softmax_f32}` (3) | [gb10/common/moe_topk.cu:182][f100] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t100] | not measured |
| moe_topk::`moe_topk_softmax_rows` | [gb10/common/moe_topk.cu:197][f100] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t100] · [#34][pr34] | [0%][m100.moe_topk_softmax_rows] (decode C=1 (R=2, MTP k=1)) |
| moe_topk_sig::`moe_topk_{sigmoid, sigmoid_batched}` (2) | [gb10/common/moe_topk_sigmoid.cu:22][f101] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t101] | not measured |
| moe_topk_softmax_bias::`moe_topk_softmax_{bias, bias_batched}` (2) | [gb10/common/moe_topk_softmax_bias.cu:189][f102] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t102] | not measured |
| moe_topk_softmax_bias::`moe_zero_expert_add` | [gb10/common/moe_topk_softmax_bias.cu:233][f102] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t102] | not measured |
| moe_topk_sqrt::`moe_topk_{sqrtsoftplus, sqrtsoftplus_batched}` (2) | [gb10/common/moe_topk_sqrtsoftplus.cu:22][f103] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t103] | not measured |
| moe_transpose_batched::`moe_transpose_u8_batched` | [gb10/common/moe_transpose_batched.cu:21][f104] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t104] | not measured |
| moe_unpermute_blend::`moe_unpermute_blend` | [gb10/common/moe_unpermute_blend.cu:17][f105] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_k32, ptrtable_t}` (3) | [gb10/common/moe_w4a16_grouped_gemm.cu:231][f106] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3, Gemma4, Nemotron-H (6 ckpts) | [4 notes][t106] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_ptrtable_{alkm_m16_k128, bt_k128, bt_m128_k64, bt_m128_k64_mfast, bt_m16_k128, bt_m16_k128_mfast, bt_m16_n128_k128, bt_m64_k128_mfast, k64, m16_k64}` (10) | [gb10/common/moe_w4a16_grouped_gemm.cu:973][f106] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t106] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm` | [gb10/common/moe_w8a8_grouped_gemm.cu:93][f107] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t107] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm_pm4` | [gb10/common/moe_w8a8_grouped_gemm.cu:396][f107] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [4 notes][t107] | [18–32%][m107.moe_w8a8_grouped_gemm_pm4] (prefill 32k (cold, 32772 tok)) |
| moe_w8a8_grouped_gemm_e4m3::`moe_w8a8_{gateup_silu_e4m3_w1, gateup_silu_e4m3_w2, grouped_gemm_e4m3_dn, grouped_gemm_e4m3_gu}` (4) | [gb10/common/moe_w8a8_grouped_gemm_e4m3.cu:101][f108] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_{relu2_down_prefill, up_prefill}` (2) | [gb10/common/nemotron_moe_prefill.cu:161][f109] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_topk_sigmoid_batched` | [gb10/common/nemotron_moe_prefill.cu:53][f109] | routing / top-k | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t109] | not measured |
| nemotron_moe_prefill::`nemotron_moe_weighted_sum_prefill` | [gb10/common/nemotron_moe_prefill.cu:444][f109] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| relu2::`moe_weighted_sum_scale` | [gb10/common/relu_squared.cu:57][f153] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t153] | not measured |
| w4a16_gemv::`glm5next_moe_row_union` | [gb10/common/w4a16_gemv.cu:2201][f173] | dispatch / combine | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [2 notes][t173] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k_experts_w2, q2_k_experts_w8, q2_k_groups_w, q2_k_pair_w, q3_k_experts_w2, q3_k_experts_w8}` (6) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:321][f195] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [7 notes][t195] | not measured |
| moe_silu_mul::`moe_silu_mul` | [gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu:40][f202] | expert activation | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t202] | not measured |
| moe_v41::`moe_v41_{accumulate, finish, gather_rows, scatter_add, slot_table_set, sum_rows, swiglu}` (7) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:16][f203] | dispatch / combine | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t203] | not measured |
| moe_v41::`moe_v41_{route_select, router_gemv_f32out, router_gemv_f32out_products}` (3) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:145][f203] | routing / top-k | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t203] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_e8m0, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_e8m0, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_e8m0, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_e8m0, w4a16_grouped_gemm_ptrtable_t_k64, w4a16_grouped_gemm_ptrtable_t_k64_e8m0}` (11) | [gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu:188][f204] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, Kimi-K3, LongCat (4 ckpts) | [1 note][t204] | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused.cu:31][f216] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t216] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:35][f217] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t217] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:327][f217] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t217] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:33][f218] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t218] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:323][f218] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t218] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/gemma-4-26b-a4b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f219] | expert GEMM/GEMV | b200 gb10 hop | Gemma4, Mistral4, Qwen-GDN-MoE, Qwen3-VL (10 ckpts) | [1 note][t219] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_m128, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (7) | [gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f233] | expert GEMM/GEMV | gb10 | Laguna, MiniMax-M2, Step-3.7 (4 ckpts) | [1 note][t233] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_relu2, ptrtable_t}` (3) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu:590][f243] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [2 notes][t243] | not measured |
| moe_w4a4::`moe_w4a4_grouped_gemm_relu2` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu:49][f244] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t244] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:135][f255] | expert GEMM/GEMV | gb10 hop strix | none — its callers' targets compile another copy | [1 note][t255] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_down_t_k64_fp4, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_fp4, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_k32, w4a16_grouped_gemm_ptrtable_m256, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (10) | [gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f265] | expert GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN-MoE, Qwen3.8-FN (7 ckpts) | [9 notes][t265] | not measured |
| moe_silu_mul::`moe_silu_mul` | [gb10/step3p7-flash/nvfp4/moe_silu_mul.cu:31][f272] | expert activation | gb10 | Step-3.7 (1 ckpts) | [1 note][t272] | not measured |
| moe_bucket_builder::`bucket_builder` | [hopper/common/moe_bucket_builder.cu:5][f278] | dispatch / combine | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [2 notes][t278] · [#25][pr25] | not measured |
| moe_w8a8_m16::`pm4_m16` | [hopper/common/moe_w8a8_m16.cu:106][f279] | expert GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [3 notes][t279] · [#25][pr25] | not measured |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [strix-hip/common/moe_fp8_grouped_gemm.cu:249][f336] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t336] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [strix-hip/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:61][f339] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [4 notes][t339] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `relu_squared_inplace`, `bf16_residual_add`, `gelu_mul`, `gelu`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `rms_norm`, `rms_norm_residual`, `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2); [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_f32out`, `dense_gemm_bf16_pipelined`, `dense_gemm_f32in_f32out`, `dense_gemv_bf16`, `dense_gemv_bf16_fp32out`, `dense_gemv_bf16_batchm`, `dense_gemm_{bf16, bf16_f32out, bf16_pipelined}` (3), `dense_gemm_f32in_f32out`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemv`, `fp8_gemm_t`, `fp8_gemm_{t, t_m128_mfast}` (2), `fp8_gemm_t`, `fp8_gemm_t`, `w8a16_gemv`, `w8a16_gemm`, `fp8_gemm_t`, `fp8_gemm_t`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_{gemm, gemm_t}` (2), `w4a16_{gemv, gemv_batch2, gemv_batch3}` (3), `w4a16_gemv_sw`, `w4a16_gemv_sw_{moe, moe_batchm_m2, moe_batchm_m3, moe_batchm_m4, moe_batchm_m5, moe_batchm_m6, moe_batchm_m7, moe_batchm_m8}` (8), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t}` (2), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_gemm_t`, `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm_mfast`; [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `kquant_mmvq_q2_k_w`, `kquant_mmvq_q3_k_w`, `metrale_q2_k_mmq128_nc`, `metrale_q2_k_mmq128_wc`, `metrale_q3_k_mmq128_nc`, `metrale_q3_k_mmq128_wc`; [Quantization and format conversion](#quantization-and-format-conversion): `silu_mul_quant_fp8`, `quantize_bf16_to_nvfp4`, `kquant_q8_1_rows_bf16`, `kquant_swiglu_q8_1_rows_bf16`, `metrale_q8_1_quantize_d2s6_bf16`, `metrale_q8_1_quantize_d4_bf16`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`.

### Dense FFN (gate/up/down projections of non-MoE layers)

88 entry points: 0 primary here (full rows), 88 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `gelu_mul`, `metrale_nvfp4_silu_mul_quant`, `metrale_nvfp4_silu_mul_scaled`, `silu_mul_strided`, `gelu`; [MoE](#moe-routing-dispatch-expert-gemm-gemv-combine): `moe_silu_mul`, `moe_silu_mul`, `moe_silu_mul`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `k3_dense_{down_f32io, gate_up_situ_f32io}` (2), `dense_gemm_bf16`, `dense_gemm_tc`, `dense_gemv_bf16`, `dense_gemm_bf16`, `dense_gemv_bf16`, `dense_gemm_bf16`, `dense_gemm_tc`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemv_{batch16, batch4}` (2), `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemm_t_m128`, `w8a16_gemv`, `w8a16_gemv_{batch16, batch4}` (2), `w8a16_gemv_{dual, silu_input}` (2), `w8a16_gemm_{m16, m16_n64}` (2), `w8a16_gemv`, `w8a16_gemv_{dual, silu_input}` (2), `w8a16_gemm`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch2, gemv_batch3, gemv_dual_batch2, gemv_dual_batch3}` (5), `w4a16_gemv_sw`, `w4a16_gemv_dual`, `w4a16_gemv_{dual_sw, silu_input, silu_input_sw}` (3), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128_bf16, gemm_t_m128_bf16_v2}` (3), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm`, `metrale_nvfp4_mmq128_nc`, `metrale_nvfp4_{mmq128_wc, mmq16_nc, mmq16_wc, mmq32_wc, mmq64_nc, mmq64_wc}` (6), `metrale_nvfp4_mmq32_nc`, `w4a4_gemm`; [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `q2_0_gemv_vec`, `q2_0_gemv_vec_batchm`, `metrale_q2_0_mmq128_{nc, wc}` (2), `metrale_q4k_mmq128_{nc, wc}` (2), `int8_gemm_faith2`, `int8_gemm_i32acc`, `requant_a_bf16_int8`, `requant_w_nvfp4_int8`; [Quantization and format conversion](#quantization-and-format-conversion): `dequant_q2_0_gn_to_bf16`, `dequant_nvfp4_to_bf16`, `fp8_act_scale_to_kmajor`, `quantize_bf16_to_nvfp4`, `metrale_nvfp4_quantize_bf16`, `metrale_nvfp4_repack`, `metrale_nvfp4_scale_bf16`, `metrale_q8_1_quantize_ds4_bf16`, `q4k_quantize`.

### Projection GEMM/GEMV — BF16/F32

26 entry points: 26 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dense_f32io::`k3_dense_{down_f32io, gate_up_situ_f32io}` (2) | [b200/kimi-k3/bf16/dense_f32io.cu:12][f1] | BF16/F32 GEMM/GEMV | b200 | Kimi-K3 (1 ckpts) | [1 note][t1] | not measured |
| gemm::`dense_gemm_bf16` | [gb10/common/dense_gemm_bf16.cu:26][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | [1–2%][m14.dense_gemm_bf16] (prefill 32k (cold, 32772 tok)) |
| gemm::`dense_gemm_bf16_f32out` | [gb10/common/dense_gemm_bf16.cu:85][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | not measured |
| gemm::`dense_gemm_bf16_pipelined` | [gb10/common/dense_gemm_bf16.cu:446][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families + NLLB (31 ckpts) | [1 note][t14] | not measured |
| gemm::`dense_gemm_f32in_f32out` | [gb10/common/dense_gemm_bf16.cu:131][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| gemm_splitk::`dense_gemm_splitk_{partial, reduce}` (2) | [gb10/common/dense_gemm_splitk.cu:27][f15] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t15] | not measured |
| gemm_tc::`dense_gemm_{tc, tc_scaled_acc}` (2) | [gb10/common/dense_gemm_tc.cu:185][f16] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t16] | not measured |
| gemv::`dense_gemv_bf16` | [gb10/common/dense_gemv_bf16.cu:33][f17] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t17] | [94–100%][m17.dense_gemv_bf16] (decode C=1 (R=4, MTP k=3)) |
| gemv::`dense_gemv_bf16_fp32out` | [gb10/common/dense_gemv_bf16.cu:120][f17] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [3 notes][t17] | not measured |
| dense_gemv_bf16_batch2::`dense_gemv_bf16_batch2` | [gb10/common/dense_gemv_bf16_batch2.cu:32][f18] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t18] | not measured |
| dense_gemv_bf16_batchm::`dense_gemv_bf16_batchm` | [gb10/common/dense_gemv_bf16_batchm.cu:86][f19] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t19] | [16–92%][m19.dense_gemv_bf16_batchm] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_tc16` | [gb10/common/dense_gemv_bf16_tc.cu:251][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | [86–95%][m20.dense_gemv_bf16_tc16] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_{tc32, tc8}` (2) | [gb10/common/dense_gemv_bf16_tc.cu:250][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | not measured |
| dense_gemm_m16_bf16::`dense_gemm_m16_{bf16, bf16_n64}` (2) | [hopper/common/dense_gemm_m16_bf16.cu:339][f273] | BF16/F32 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t273] | not measured |
| dense_gemm_bf16::`dense_gemm_bf16` | [metal/common/dense_gemm_bf16.metal:23][f301] | BF16/F32 GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t301] | not measured |
| dense_gemv_bf16::`dense_gemv_bf16` | [metal/common/dense_gemv_bf16.metal:27][f302] | BF16/F32 GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t302] | not measured |
| gemm::`dense_gemm_{bf16, bf16_f32out, bf16_pipelined}` (3) | [strix-hip/common/dense_gemm_bf16.cu:26][f334] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [2 notes][t334] | not measured |
| gemm::`dense_gemm_f32in_f32out` | [strix-hip/common/dense_gemm_bf16.cu:129][f334] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [1 note][t334] | not measured |
| gemm_tc::`dense_gemm_tc` | [strix-hip/common/dense_gemm_tc.cu:31][f335] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t335] | not measured |

### Projection GEMM/GEMV — FP8 (W8A16, W8A8, block-scaled)

73 entry points: 73 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w8a16_gemv_batch4::`w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4) | [b300/common/w8a16_gemv_batch4.cu:213][f4] | FP8 GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [1 note][t4] | not measured |
| gemv_fp8w::`dense_gemv_fp8w` | [gb10/common/dense_gemv_fp8w.cu:131][f21] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| dense_gemv_fp8w_batch2::`dense_gemv_fp8w_batch2` | [gb10/common/dense_gemv_fp8w_batch2.cu:72][f22] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t22] | not measured |
| fp8_gemm_blockscaled_pipe::`fp8_gemm_blockscaled_pipe_128x64` | [gb10/common/fp8_gemm_blockscaled_pipe.cu:63][f30] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| fp8_gemm_t_blockscaled::`fp8_gemm_t_blockscaled` | [gb10/common/fp8_gemm_t_blockscaled.cu:113][f31] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [3 notes][t31] | [14–49%][m31.fp8_gemm_t_blockscaled] (prefill 32k (cold, 32772 tok)) |
| fp8_gemv_rt::`fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2) | [gb10/common/fp8_gemv_rt.cu:156][f32] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t32] | not measured |
| w4a16_fp8_ldmab::`fp8_fp8_gemm_ldmab` | [gb10/common/w4a16_fp8_ldmab.cu:65][f171] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t171] | [11–43%][m171.fp8_fp8_gemm_ldmab] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm::`w8a16_gemm` | [gb10/common/w8a16_gemm.cu:86][f177] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t177] | not measured |
| w8a16_gemm_pipe128::`w8a16_gemm_pipe128` | [gb10/common/w8a16_gemm_pipe128.cu:58][f178] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w8a16_gemm_pipelined::`w8a16_gemm_pipelined` | [gb10/common/w8a16_gemm_pipelined.cu:174][f179] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t179] | [23–24%][m179.w8a16_gemm_pipelined] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m32` | [gb10/common/w8a16_gemm_pipelined_m32.cu:159][f180] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [7 notes][t180] · [#4][pr4] [#34][pr34] | [20–95%][m180.w8a16_gemm_pipelined_m32] (decode C=16 (R=32, MTP k=1)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m64` | [gb10/common/w8a16_gemm_pipelined_m32.cu:318][f180] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t180] · [#4][pr4] [#34][pr34] | not measured |
| w8a16_gemm_t::`w8a16_gemm_{t, t_pipelined}` (2) | [gb10/common/w8a16_gemm_t.cu:151][f181] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t181] | not measured |
| w8a16_gemm_t_m128::`w8a16_gemm_t_m128` | [gb10/common/w8a16_gemm_t_m128.cu:62][f182] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t182] | [28%][m182.w8a16_gemm_t_m128] (prefill 4k (cold, 4549 tok)) |
| w8a16_gemv::`w8a16_gemv` | [gb10/common/w8a16_gemv.cu:110][f183] | FP8 GEMM/GEMV | b200 b300 gb10 strix hip | all 14 decoder families (30 ckpts) | [2 notes][t183] | not measured |
| w8a16_gemv_batch4::`w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4) | [gb10/common/w8a16_gemv_batch4.cu:234][f184] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t184] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [gb10/common/w8a16_gemv_fused.cu:123][f185] | FP8 GEMM/GEMV | b200 b300 gb10 | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t185] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4) | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:382][f210] | FP8 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t210] | not measured |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu:72][f235] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t235] | not measured |
| w4a16_v3::`w4a16_gemm_t_m128_v3` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v3.cu:73][f236] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t236] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, fp8_gemm_t_m128_mfast, gemm_t, gemm_t_m128, gemm_t_m128_mfast}` (6) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:555][f246] | FP8 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t246] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_row_scaled, gemm_t_row_scaled_k64, gemm_t_row_scaled_m16, gemm_t_row_scaled_p4}` (7) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:820][f260] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [4 notes][t260] | not measured |
| w4a16::`fp8_gemm_t_m128` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:2745][f260] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t260] | [14–17%][m260.fp8_gemm_t_m128] (prefill 32k (cold, 32772 tok)) |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm_v2.cu:100][f261] | FP8 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t261] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128, gemm_t_row_scaled, gemm_t_row_scaled_m16}` (6) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:630][f267] | FP8 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t267] · [#34][pr34] | not measured |
| w8a16_gemm_m16::`w8a16_gemm_{m16, m16_n64, m16_strided}` (3) | [hopper/common/w8a16_gemm_m16.cu:382][f283] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t283] | not measured |
| w8a16_gemv::`w8a16_gemv` | [hopper/common/w8a16_gemv.cu:54][f284] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t284] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [hopper/common/w8a16_gemv_fused.cu:66][f285] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t285] | not measured |
| w8a16_gemv_ncol::`w8a16_gemv_batch16_{ncol2, ncol2_strided, ncol4, ncol4_strided}` (4) | [hopper/common/w8a16_gemv_ncol.cu:199][f286] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t286] | not measured |
| w8a16_gemm::`w8a16_gemm` | [strix-hip/common/w8a16_gemm.cu:230][f337] | FP8 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t337] | not measured |
| w8a16_gemm_t::`w8a16_gemm_t` | [strix-hip/common/w8a16_gemm_t.cu:119][f338] | FP8 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t338] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4) | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:324][f340] | FP8 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [1 note][t340] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4) | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:323][f341] | FP8 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t341] | not measured |

### Projection GEMM/GEMV — NVFP4 W4A16

60 entry points: 60 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a16::`w4a16_{gemm, gemm_t}` (2) | [gb10/common/w4a16_gemm.cu:87][f172] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 | GLM-5.3, Kimi-K3 (2 ckpts) | [1 note][t172] | not measured |
| w4a16_gemv::`w4a16_{gemv, gemv_batch16, gemv_batch2, gemv_batch3, gemv_batch32, gemv_batch8, gemv_batch8_rt2, gemv_dual_batch2, gemv_dual_batch3, gemv_logits, gemv_qg, gemv_qg_batch2, gemv_qg_batch3}` (13) | [gb10/common/w4a16_gemv.cu:167][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t173] | not measured |
| w4a16_gemv::`w4a16_gemv_qkvz` | [gb10/common/w4a16_gemv.cu:1557][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t173] | not measured |
| w4a16_gemv::`w4a16_gemv_sw` | [gb10/common/w4a16_gemv.cu:233][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t173] | [96–99%][m173.w4a16_gemv_sw] (decode C=1 (R=4, MTP k=3)) |
| w4a16_gemv::`w4a16_gemv_sw_{moe, moe_batchm_m2, moe_batchm_m3, moe_batchm_m4, moe_batchm_m5, moe_batchm_m6, moe_batchm_m7, moe_batchm_m8}` (8) | [gb10/common/w4a16_gemv.cu:300][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [1 note][t173] | not measured |
| w4a16_gemv_fused::`w4a16_gemv_dual` | [gb10/common/w4a16_gemv_fused.cu:144][f174] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t174] | not measured |
| w4a16_gemv_fused::`w4a16_gemv_{dual_sw, silu_input, silu_input_sw}` (3) | [gb10/common/w4a16_gemv_fused.cu:209][f174] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [2 notes][t174] | not measured |
| w4a16_gemv_tc::`w4a16_gemv_tc16` | [gb10/common/w4a16_gemv_tc.cu:256][f175] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t175] · [#1][pr1] | [86–92%][m175.w4a16_gemv_tc16] (decode C=16 (R=32, MTP k=1)) |
| w4a16_gemv_tc::`w4a16_gemv_tc8` | [gb10/common/w4a16_gemv_tc.cu:255][f175] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t175] · [#1][pr1] | [59–90%][m175.w4a16_gemv_tc8] (decode C=1 (R=4, MTP k=3)) |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:32][f210] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t210] | not measured |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:32][f246] | NVFP4 W4A16 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t246] | not measured |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_k64_p3, gemm_t_m128_bf16, gemm_t_m128_bf16_v2}` (6) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:151][f260] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [6 notes][t260] | not measured |
| w4a16::`w4a16_gemm_t_k64_n64_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1648][f260] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t260] | [54%][m260.w4a16_gemm_t_k64_n64_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_gemm_t_m128` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1860][f260] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t260] | [28–32%][m260.w4a16_gemm_t_m128] (prefill 4k (cold, 4103 tok)) |
| w4a16::`w4a16_gemm_t_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:585][f260] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t260] | [8–78%][m260.w4a16_gemm_t_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_{gemm, gemm_t_k64, gemm_t_m128}` (3) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:151][f267] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [4 notes][t267] · [#34][pr34] | not measured |
| w4a16::`w4a16_gemm_t` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:345][f267] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [6 notes][t267] · [#34][pr34] | [73–87%][m267.w4a16_gemm_t] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:91][f340] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [4 notes][t340] | not measured |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:91][f341] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [4 notes][t341] | not measured |

### Projection GEMM/GEMV — W4A4 (FP4 activations)

22 entry points: 22 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a4_gemv_mx::`w4a4_{gemv_mx16, gemv_mx16_nt2, gemv_mx16_ps, gemv_mx32, gemv_mx32_nt4, gemv_mx32_ps, gemv_mx64, gemv_mx64_nt2, gemv_mx8, quant_rows}` (10) | [gb10/common/w4a4_gemv_mx.cu:360][f176] | W4A4 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [23 notes][t176] · [#1][pr1] [#14][pr14] [#18][pr18] | not measured |
| w4a4::`w4a4_{gemm, gemm_mfast}` (2) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a4_gemm.cu:115][f247] | W4A4 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t247] | not measured |
| nvfp4_mmq::`metrale_nvfp4_{gemm_pipe, mmq128_wc, mmq16_nc, mmq16_wc, mmq32_wc, mmq64_nc, mmq64_wc}` (7) | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:72][f256] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | not measured |
| nvfp4_mmq::`metrale_nvfp4_mmq128_nc` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:67][f256] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [17–18%][m256.metrale_nvfp4_mmq128_nc] (prefill 4k (cold, 4103 tok)) |
| nvfp4_mmq::`metrale_nvfp4_mmq32_nc` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:97][f256] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [82–87%][m256.metrale_nvfp4_mmq32_nc] (decode C=16 (R=32, MTP k=1)) |
| w4a4::`w4a4_gemm` | [gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu:50][f262] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t262] | not measured |

### Projection GEMM/GEMV — integer / K-quant (Q2_0, Q2_K..Q6_K, INT8, MLX INT8)

23 entry points: 23 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| q2_0_gemv_vec::`q2_0_gemv_vec` | [gb10/common/q2_0_gemv_vec.cu:80][f149] | integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t149] | not measured |
| q2_0_gemv_vec::`q2_0_gemv_vec_batchm` | [gb10/common/q2_0_gemv_vec.cu:162][f149] | integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t149] | not measured |
| kquant_moe::`kquant_mmvq_q2_k_w`, `kquant_mmvq_q3_k_w`, `kquant_mmvq_q6_k_w`, `metrale_q2_k_mmq128_nc`, `metrale_q2_k_mmq128_wc`, `metrale_q3_k_mmq128_nc`, `metrale_q3_k_mmq128_wc` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:76][f195] | integer / K-quant GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t195] | not measured |
| q2_0_mmq::`metrale_q2_0_mmq128_{nc, wc}` (2) | [gb10/qwen3.6-27b/nvfp4/q2_0_mmq.cu:60][f257] | integer / K-quant GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t257] | not measured |
| q4k_mmq::`metrale_q4k_mmq128_{nc, wc}` (2) | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:50][f258] | integer / K-quant GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t258] | not measured |
| w4a16::`int8_gemm_faith2`, `int8_gemm_i32acc`, `requant_a_bf16_int8`, `requant_w_nvfp4_int8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:4569][f260] | integer / K-quant GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t260] | not measured |
| mlx_int8_dequant::`mlx_int8_dequant` | [metal/common/mlx_int8_dequant.metal:21][f315] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | — | not measured |
| mlx_int8_gemm::`mlx_int8_gemm` | [metal/common/mlx_int8_gemm.metal:23][f316] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t316] | not measured |
| mlx_int8_gemv::`mlx_int8_gemv` | [metal/common/mlx_int8_gemv.metal:39][f317] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t317] | not measured |
| mlx_int8_gemv_gate_up::`mlx_int8_gemv_gate_up` | [metal/common/mlx_int8_gemv_gate_up.metal:41][f318] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t318] | not measured |
| mlx_int8_gemv_silu_gate::`mlx_int8_gemv_silu_{gate, gate_resid}` (2) | [metal/common/mlx_int8_gemv_silu_gate.metal:29][f319] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [2 notes][t319] | not measured |

### Normalization (RMSNorm, LayerNorm, L2, gated norms)

64 entry points: 64 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| norm::`gated_rms_{norm, norm_f32_input, norm_f32_input_strided}` (3) | [gb10/common/rms_norm.cu:1003][f158] | normalization | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [4 notes][t158] | not measured |
| norm::`gated_rms_norm_prefill` | [gb10/common/rms_norm.cu:1277][f158] | normalization | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t158] | [69–96%][m158.gated_rms_norm_prefill] (decode C=16 (R=32, MTP k=1)) |
| norm::`l2_norm_bf16` | [gb10/common/rms_norm.cu:1376][f158] | normalization | b200 b300 gb10 hop strix hip | GLM-5.3, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (15 ckpts) | [2 notes][t158] | [81–90%][m158.l2_norm_bf16] (prefill 4k (cold, 4103 tok)) |
| norm::`residual_add_rms_norm` | [gb10/common/rms_norm.cu:382][f158] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [3 notes][t158] | [6–80%][m158.residual_add_rms_norm] (decode C=1 (R=2, MTP k=1)) |
| norm::`residual_add_rms_norm_gatef32`, `residual_add_rms_norm_vanilla`, `rms_norm`, `rms_norm_residual_vanilla`, `rms_norm_strided` | [gb10/common/rms_norm.cu:45][f158] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [3 notes][t158] | not measured |
| norm::`rms_norm_residual` | [gb10/common/rms_norm.cu:259][f158] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [2 notes][t158] | [4–90%][m158.rms_norm_residual] (decode C=1 (R=2, MTP k=1)) |
| rms_norm_vanilla::`rms_norm_{vanilla, vanilla_warp_row}` (2) | [gb10/common/rms_norm_vanilla.cu:38][f159] | normalization | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t159] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:499][f222] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t222] | not measured |
| norm::`residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:40][f222] | normalization | gb10 | Gemma4 (2 ckpts) | [1 note][t222] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:520][f226] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t226] | not measured |
| norm::`residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:40][f226] | normalization | gb10 | Gemma4 (2 ckpts) | [1 note][t226] | not measured |
| norm::`gated_rms_norm` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:296][f234] | normalization | b200 gb10 hop | Nemotron-H (3 ckpts) | [2 notes][t234] | not measured |
| norm::`gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:419][f234] | normalization | b200 gb10 hop | none — its callers' targets compile another copy | [1 note][t234] | not measured |
| norm::`residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:42][f234] | normalization | b200 gb10 hop | LongCat, MiniMax-M2, Mistral4, Nemotron-H, Step-3.7 (7 ckpts) | [1 note][t234] | not measured |
| norm::`gated_rms_norm`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:42][f245] | normalization | b200 gb10 hop | Nemotron-H (3 ckpts) | [2 notes][t245] | not measured |
| norm::`gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:427][f245] | normalization | b200 gb10 hop | none — its callers' targets compile another copy | [1 note][t245] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `l2_norm_bf16` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:285][f249] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t249] | not measured |
| norm::`residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:42][f249] | normalization | gb10 | Qwen3-VL (1 ckpts) | [1 note][t249] | not measured |
| gated_norm_sigmoid::`gated_rms_norm_{f32_input_sigmoid, prefill_sigmoid, sigmoid}` (3) | [gb10/qwen3.8-flash-next/nvfp4/gated_norm_sigmoid.cu:50][f268] | normalization | gb10 | Qwen3.8-FN (1 ckpts) | [1 note][t268] | not measured |
| add_rms_norm::`add_rms_norm` | [metal/common/add_rms_norm.metal:32][f287] | normalization | metal | Qwen-GDN (7 ckpts) | [1 note][t287] | not measured |
| rms_norm::`rms_norm` | [metal/common/rms_norm.metal:21][f323] | normalization | metal | Qwen-GDN (7 ckpts) | [1 note][t323] | not measured |

### Activations and elementwise (SiLU/GELU/ReLU², residual, gates, scale)

16 entry points: 16 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| bf16_add::`bf16_add_inplace` | [gb10/common/bf16_add.cu:8][f12] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t12] | not measured |
| relu2::`relu_squared_inplace` | [gb10/common/relu_squared.cu:26][f153] | activation / gate / residual | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| residual_add::`bf16_concat` | [gb10/common/residual_add.cu:142][f157] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | [4%][m157.bf16_concat] (prefill 4k (cold, 4549 tok)) |
| residual_add::`bf16_residual_add` | [gb10/common/residual_add.cu:10][f157] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | [97–100%][m157.bf16_residual_add] (prefill 4k (cold, 4103 tok)) |
| residual_add::`bf16_scaled_add`, `sigmoid_gate_mul`, `sigmoid_gate_mul_batched`, `sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast` | [gb10/common/residual_add.cu:60][f157] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t157] | not measured |
| gelu::`gelu_mul` | [gb10/gemma-4-26b-a4b/nvfp4/gelu.cu:43][f214] | activation / gate / residual | gb10 | Gemma4 (2 ckpts) | [1 note][t214] | not measured |
| nvfp4_mmq::`metrale_nvfp4_silu_mul_quant` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:242][f256] | activation / gate / residual | gb10 hop | Qwen-GDN (7 ckpts) | [4 notes][t256] | [57–87%][m256.metrale_nvfp4_silu_mul_quant] (decode C=16 (R=32, MTP k=1)) |
| nvfp4_mmq::`metrale_nvfp4_silu_mul_scaled` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:221][f256] | activation / gate / residual | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | not measured |
| silu_mul_strided::`silu_mul_strided` | [hopper/common/silu_mul_strided.cu:44][f282] | activation / gate / residual | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t282] | not measured |
| bf16_add::`bf16_add` | [metal/common/bf16_add.metal:19][f297] | activation / gate / residual | metal | Qwen-GDN (7 ckpts) | — | not measured |
| gelu::`gelu` | [metal/common/gelu.metal:22][f306] | activation / gate / residual | metal | Qwen-GDN (7 ckpts) | — | not measured |
| sigmoid_gate::`sigmoid_gate` | [metal/common/sigmoid_gate.metal:16][f326] | activation / gate / residual | metal | Qwen-GDN (7 ckpts) | — | not measured |

### Positional encoding (RoPE, YaRN, MRoPE)

12 entry points: 12 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| rope::`rope_{forward, forward_proportional, forward_strided, forward_yarn, forward_yarn_interleaved, forward_yarn_interleaved_inv, forward_yarn_scaled}` (7) | [gb10/common/rope.cu:29][f160] | rotary | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (29 ckpts) | [3 notes][t160] | not measured |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved` | [gb10/common/rope_mrope_interleaved.cu:34][f161] | rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | [6%][m161.rope_forward_mrope_interleaved] (prefill 32k (cold, 32772 tok)) |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved_k_only` | [gb10/common/rope_mrope_interleaved.cu:108][f161] | rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | not measured |
| rope::`rope_{forward, forward_yarn}` (2) | [gb10/mistral-small-4/nvfp4/rope.cu:29][f242] | rotary | gb10 | Mistral4 (1 ckpts) | [1 note][t242] | not measured |
| rope_apply::`rope_apply` | [metal/common/rope_apply.metal:33][f324] | rotary | metal | Qwen-GDN (7 ckpts) | — | not measured |

### KV cache (write, quantize, TurboQuant rotation, slot metadata)

36 entry points: 36 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| fused_k_norm_rope_cache::`fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2) | [gb10/common/fused_k_norm_rope_cache.cu:53][f34] | cache write | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t34] | not measured |
| metadata_fill::`fill_slots_from_block_table` | [gb10/common/metadata_fill.cu:5][f65] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t65] | not measured |
| reshape_and_cache::`bf16_absmax`, `reshape_and_cache_flash`, `reshape_and_cache_flash_fp8`, `reshape_and_cache_flash_nvfp4`, `reshape_and_cache_flash_v_only` | [gb10/common/reshape_and_cache.cu:30][f154] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t154] | not measured |
| reshape_and_cache_fused_k_fp8::`fused_k_norm_rope_cache_write_fp8_kv` | [gb10/common/reshape_and_cache_fused_k_fp8.cu:131][f155] | cache write | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t155] | not measured |
| reshape_and_cache_turbo::`reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13) | [gb10/common/reshape_and_cache_turbo.cu:179][f156] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t156] | not measured |
| tq_plus_innerq_apply::`tq_plus_innerq_apply_{k, q}` (2) | [gb10/common/tq_plus_innerq_apply.cu:71][f168] | TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t168] | not measured |
| wht_bf16::`wht_bf16_{inplace, inplace_inv}` (2) | [gb10/common/wht_bf16.cu:51][f186] | TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t186] | not measured |
| kv_cache_append::`kv_cache_append` | [metal/common/kv_cache_append.metal:19][f307] | cache write | metal | Qwen-GDN (7 ckpts) | — | not measured |
| kv_cache_append_bf16k_turbov::`kv_cache_append_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/kv_cache_append_bf16k_turbov.metal:129][f308] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t308] | not measured |
| kv_cache_append_turbo2::`kv_cache_append_turbo2` | [metal/common/kv_cache_append_turbo2.metal:53][f309] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t309] | not measured |
| kv_cache_append_turbo3::`kv_cache_append_turbo3` | [metal/common/kv_cache_append_turbo3.metal:65][f310] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t310] | not measured |
| kv_cache_append_turbo4::`kv_cache_append_turbo4` | [metal/common/kv_cache_append_turbo4.metal:82][f311] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t311] | not measured |
| kv_cache_append_turbo8::`kv_cache_append_turbo8` | [metal/common/kv_cache_append_turbo8.metal:47][f312] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t312] | not measured |
| wht_bf16::`wht_bf16_{inplace, inplace_inv}` (2) | [metal/common/wht_bf16.metal:142][f329] | TurboQuant rotation | metal | Qwen-GDN (7 ckpts) | [1 note][t329] | not measured |

### Quantization and format conversion

47 entry points: 47 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| gemv_fp8w::`quantize_bf16_to_fp8` | [gb10/common/dense_gemv_fp8w.cu:65][f21] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| dequant_fp8_blockscaled_bf16::`dequant_fp8_blockscaled_bf16` | [gb10/common/dequant_fp8_blockscaled_bf16.cu:89][f23] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t23] | not measured |
| dequant_gguf_bf16::`dequant_{q2_0_gn_to_bf16, q2_k_to_bf16, q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (6) | [gb10/common/dequant_gguf_bf16.cu:43][f24] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t24] | not measured |
| dequant_nvfp4_bf16::`dequant_nvfp4_to_bf16` | [gb10/common/dequant_nvfp4_bf16.cu:50][f25] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t25] | not measured |
| fp8_scale_transpose::`fp8_act_scale_to_kmajor` | [gb10/common/fp8_scale_transpose.cu:35][f33] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t33] | not measured |
| moe_silu_mul::`silu_mul_quant_fp8` | [gb10/common/moe_silu_mul.cu:107][f98] | activation quantize | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN (19 ckpts) | [1 note][t98] | [21–24%][m98.silu_mul_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| per_token_group_quant_fp8::`per_token_group_quant_fp8` | [gb10/common/per_token_group_quant_fp8.cu:39][f144] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t144] | [81–83%][m144.per_token_group_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| quant_rowwise_fp8::`quant_rowwise_fp8` | [gb10/common/quant_rowwise_fp8.cu:38][f150] | activation quantize | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t150] | not measured |
| quantize_bf16_to_fp8_blockscaled::`quantize_bf16_to_fp8_blockscaled` | [gb10/common/quantize_bf16_to_fp8_blockscaled.cu:54][f151] | activation quantize | b200 b300 gb10 hop | LongCat (1 ckpts) | [1 note][t151] | not measured |
| quantize_nvfp4::`f32_to_bf16_trunc` | [gb10/common/quantize_bf16_to_nvfp4.cu:29][f152] | dtype conversion | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t152] | not measured |
| quantize_nvfp4::`nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `quantize_bf16_to_nvfp4_mse` | [gb10/common/quantize_bf16_to_nvfp4.cu:133][f152] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t152] · [#34][pr34] | not measured |
| transpose_u8::`transpose_u8` | [gb10/common/transpose_u8.cu:15][f169] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_fp8_ldmab::`fp8_predequant_nvfp4_t` | [gb10/common/w4a16_fp8_ldmab.cu:193][f171] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t171] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [gb10/common/w8a16_gemm_t.cu:607][f181] | dequant / repack / transpose | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t181] | not measured |
| widen_block_scale_f32::`widen_block_scale_f32` | [gb10/common/widen_block_scale_f32.cu:21][f187] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t187] | not measured |
| kquant_moe::`kquant_q8_1_rows_bf16`, `kquant_swiglu_q8_1_rows_bf16`, `metrale_q8_1_quantize_d2s6_bf16`, `metrale_q8_1_quantize_d4_bf16` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:82][f195] | activation quantize | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t195] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:540][f210] | activation quantize | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t210] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:501][f210] | dequant / repack / transpose | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t210] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:623][f246] | activation quantize | gb10 | Nemotron-H (3 ckpts) | [1 note][t246] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:584][f246] | dequant / repack / transpose | gb10 | Nemotron-H (3 ckpts) | [1 note][t246] | not measured |
| nvfp4_mmq::`metrale_nvfp4_quantize_bf16` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:132][f256] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [28–54%][m256.metrale_nvfp4_quantize_bf16] (decode C=16 (R=32, MTP k=1)) |
| nvfp4_mmq::`metrale_nvfp4_repack` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:142][f256] | dequant / repack / transpose | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | not measured |
| nvfp4_mmq::`metrale_nvfp4_scale_bf16` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:214][f256] | dequant / repack / transpose | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [99–103%][m256.metrale_nvfp4_scale_bf16] (prefill 32k (cold, 32772 tok)) |
| q4k_mmq::`metrale_q8_1_quantize_ds4_bf16` | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:68][f258] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t258] | not measured |
| q4k_quantize::`q4k_quantize` | [gb10/qwen3.6-27b/nvfp4/q4k_quantize.cu:81][f259] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t259] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:974][f260] | activation quantize | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t260] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:932][f260] | dequant / repack / transpose | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t260] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:784][f267] | activation quantize | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t267] · [#34][pr34] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:742][f267] | dequant / repack / transpose | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t267] · [#34][pr34] | not measured |
| fp8_act_quant_hopper::`per_token_group_quant_fp8_hopper` | [hopper/common/fp8_act_quant_hopper.cu:94][f274] | activation quantize | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t274] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [strix-hip/common/w8a16_gemm_t.cu:198][f338] | dequant / repack / transpose | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t338] | not measured |
| w4a16::`bf16_to_fp8` | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:451][f340] | activation quantize | hip | Qwen-GDN (7 ckpts) | [1 note][t340] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:419][f340] | dequant / repack / transpose | hip | Qwen-GDN (7 ckpts) | [1 note][t340] | not measured |
| w4a16::`bf16_to_fp8` | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:450][f341] | activation quantize | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t341] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:418][f341] | dequant / repack / transpose | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t341] | not measured |

### Embedding and LM head (lookup, overlays, softcap, scale)

16 entry points: 11 primary here (full rows), 5 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| embed_from_argmax::`batched_embed`, `embed_from_argmax` | [gb10/common/embed_from_argmax.cu:17][f29] | embedding / LM head | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t29] | not measured |
| embed_from_argmax::`batched_embed_fp8` | [gb10/common/embed_from_argmax.cu:110][f29] | embedding / LM head | b200 b300 gb10 hop strix hip | LongCat (1 ckpts) | [2 notes][t29] | not measured |
| token_overlay::`embed_overlay_routed_bf16`, `embed_rowdiff_bf16`, `lmhead_overlay_routed_bf16`, `lmhead_overlay_routed_f32` | [gb10/common/token_overlay.cu:20][f167] | embedding / LM head | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t167] | not measured |
| embed_scale::`bf16_scale_inplace` | [gb10/gemma-4-26b-a4b/nvfp4/embed_scale.cu:12][f212] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | — | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-26b-a4b/nvfp4/logit_softcap.cu:12][f215] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t215] | not measured |
| embed_scale::`bf16_scale_inplace` | [gb10/gemma-4-31b/nvfp4/embed_scale.cu:12][f224] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t224] | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-31b/nvfp4/logit_softcap.cu:12][f225] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t225] | not measured |

Also launched here: [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `kquant_mmvq_q6_k_w`; [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_fp8`, `nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `kquant_q8_1_rows_bf16`.

### Sampling (argmax, top-p, feed-forward of the chosen token)

6 entry points: 6 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| argmax::`argmax_bf16` | [gb10/common/argmax_bf16.cu:14][f5] | argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families + NLLB (31 ckpts) | [1 note][t5] | [2%][m5.argmax_bf16] (decode C=1 (R=4, MTP k=3)) |
| argmax::`argmax_{bf16_batch, bf16_batch_lp, fp32}` (3) | [gb10/common/argmax_bf16.cu:68][f5] | argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t5] | not measured |
| argmax_feed::`argmax_bf16_batch_feed`, `feed_resolve` | [gb10/common/argmax_feed.cu:44][f6] | argmax / top-p | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t6] | not measured |

### Speculative decoding (MTP heads, DFlash drafter, verify helpers)

89 entry points: 3 primary here (full rows), 86 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dflash2::`dflash2_{conv2, selector_walk, topk16}` (3) | [gb10/common/dflash2.cu:31][f26] | DFlash drafter | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [3 notes][t26] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_concat`, `bf16_residual_add`, `sigmoid_gate_mul`; [Attention](#attention-gqa-mha-paged-decode-split-k-prefill-flash): `attn_prefill_h128`, `paged_decode_attn`, `paged_decode_attn_fp8`, `attn_prefill_{paged, paged_fp8}` (2), `attn_prefill_paged_indirect`, `attn_prefill_h128`; [Embedding and LM head](#embedding-and-lm-head-lookup-overlays-softcap-scale): `batched_embed`, `embed_from_argmax`; [GDN](#gdn-gated-delta-rule-linear-attention): `deinterleave_qg`; [Hyper-connections](#hyper-connections-mhc): `hc_expand`, `hc_head`, `hc_expand`, `hc_head`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `fill_slots_from_block_table`, `reshape_and_cache_{flash, flash_fp8}` (2); [MoE](#moe-routing-dispatch-expert-gemm-gemv-combine): `moe_expert_gemv`, `moe_weighted_sum_blend`, `moe_silu_mul`, `moe_topk_softmax`, `moe_silu_mul`, `moe_silu_mul`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `rms_norm_vanilla`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`; [Positional encoding](#positional-encoding-rope-yarn-mrope): `rope_{forward, forward_yarn}` (2), `rope_{forward, forward_yarn}` (2); [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16_batchm`, `dense_gemm_bf16`, `dense_gemm_{bf16, bf16_pipelined}` (2); [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `dense_gemv_fp8w`, `fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2), `w8a16_gemv`, `fp8_gemm_t_row_{scaled, scaled_k64, scaled_m16, scaled_p4}` (4), `fp8_gemm_t_row_{scaled, scaled_m16}` (2), `w8a16_gemv`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch16, gemv_batch32, gemv_qg}` (4), `w4a16_gemv_sw`, `w4a16_gemv_dual`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`; [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_fp8`, `nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`; [Sampling](#sampling-argmax-top-p-feed-forward-of-the-chosen-token): `argmax_bf16`, `argmax_bf16_{batch, batch_lp}` (2).

### Hyper-connections (mHC)

27 entry points: 27 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm5next_mhc::`glm5next_hc_{expand, finish, head, mix, mix_bf16, post, pre}` (7) | [gb10/common/glm5next_mhc.cu:64][f57] | hyper-connection mix | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t57] | not measured |
| hc_v41::`hc_v41_{collapse, collapse_wide, finish_collapse, mixes_dot, mixes_finish, post_wide}` (6) | [gb10/deepseek-v4-flash/nvfp4/hc_v41.cu:118][f193] | hyper-connection mix | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t193] | not measured |
| hyper_connection::`hc_expand`, `hc_head`, `hc_post`, `hc_pre` | [gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:38][f194] | hyper-connection mix | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [2 notes][t194] | not measured |
| hyper_connection::`hc_expand`, `hc_head`, `hc_post`, `hc_pre`, `hc_pre_down`, `hc_pre_finish`, `hc_pre_mix`, `hc_pre_stage`, `hc_pre_stage_bf16`, `hc_silu_scale` | [gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu:96][f269] | hyper-connection mix | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t269] | not measured |

### N-gram and memory embeddings (Engram, PLE, n-gram tables)

16 entry points: 5 primary here (full rows), 11 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| engram_v41::`engram_v41_{gate, wkv_q2k_gemv}` (2) | [gb10/deepseek-v4-flash/nvfp4/engram_v41.cu:43][f191] | memory embedding | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t191] | not measured |
| ple::`ple_{add_highway, conv, gate}` (3) | [gb10/qwen3.8-flash-next/nvfp4/ple.cu:90][f270] | memory embedding | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t270] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_scaled_add`; [Embedding and LM head](#embedding-and-lm-head-lookup-overlays-softcap-scale): `batched_embed`, `embed_from_argmax`, `batched_embed_fp8`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemm_{bf16, bf16_pipelined}` (2); [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_fp8`, `dequant_q2_k_to_bf16`.

### Vision encoder (ViT towers)

36 entry points: 32 primary here (full rows), 4 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm_vit::`glm_vit_{add_bias, add_inplace, copy, f32_to_bf16, gelu_erf, im2col_2x2, layernorm, qknorm_rope_deint, rmsnorm, scatter_head, softmax_rows, swiglu_clamp}` (12) | [gb10/glm-5.3-flash/nvfp4/glm_vit.cu:42][f229] | ViT op | gb10 | GLM-5.3 (1 ckpts) | [2 notes][t229] | not measured |
| vision_encoder::`vision_{add_inplace, attention_rope, bf16_copy, f32_to_bf16, gelu, gemm_bias, layer_norm, spatial_merge}` (8) | [gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu:23][f250] | ViT op | gb10 | Qwen-GDN-MoE, Qwen3-VL (7 ckpts) | [2 notes][t250] | not measured |
| vision_encoder::`vision_add_bias`, `vision_add_inplace`, `vision_attention_rope`, `vision_bf16_copy`, `vision_f32_to_bf16`, `vision_gelu`, `vision_gemm_bias`, `vision_layer_norm`, `vision_spatial_merge`, `vit_rope_deinterleave`, `vit_scatter_head`, `vit_softmax_rows` | [gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu:23][f266] | ViT op | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t266] | not measured |

Also launched here: [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_f32out`, `dense_gemm_bf16_pipelined`, `dense_gemm_bf16_{f32out, pipelined}` (2).

### Encoder-decoder translation (NLLB, self-contained kernel set)

29 entry points: 26 primary here (full rows), 3 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, beam_topk, bias_bf16, embed_bf16, gather_batched, gemv_bf16, layernorm_bf16, layernorm_oop_bf16, relu_bf16, scale_bf16, scatter_batched}` (14) | [gb10/common/nllb_encoder.cu:203][f110] | NLLB encoder/decoder op | b200 b300 gb10 hop | NLLB (1 ckpts) | [3 notes][t110] | not measured |
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, bias_bf16, embed_bf16, gather_batched, gemv_bf16, layernorm_bf16, relu_bf16, scale_bf16, scatter_batched}` (12) | [metal/common/nllb_encoder.metal:281][f320] | NLLB encoder/decoder op | metal | NLLB (1 ckpts) | [1 note][t320] | not measured |

Also launched here: [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_pipelined`, `dense_gemm_bf16_pipelined`; [Sampling](#sampling-argmax-top-p-feed-forward-of-the-chosen-token): `argmax_bf16`.

### LoRA adapters (BGMV shrink/expand)

6 entry points: 6 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| lora_bgmv::`lora_bgmv_{expand_fold, shrink}` (2) | [gb10/common/lora_bgmv.cu:50][f62] | BGMV shrink/expand | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t62] | not measured |
| moe_lora_gather_bgmv::`moe_lora_gather_bgmv_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_gather_bgmv.cu:57][f76] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t76] | not measured |
| moe_lora_grouped_down::`moe_lora_grouped_down_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_grouped_down.cu:67][f77] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t77] | not measured |

### Weight load and repack (one-time, not per token)

62 entry points: 0 primary here (full rows), 62 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_add_inplace`, `bf16_add`; [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `k3_kda_conv_update_f32`; [GDN](#gdn-gated-delta-rule-linear-attention): `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`; [Hyper-connections](#hyper-connections-mhc): `hc_v41_{collapse, collapse_wide, finish_collapse, mixes_dot, mixes_finish, post_wide}` (6), `hc_expand`, `hc_post`, `hc_expand`, `hc_post`; [KDA](#kda-kimi-delta-attention-linear-attention): `k3_kda_recurrent_step_f32`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `wht_bf16_inplace`, `wht_bf16_inplace`; [MLA](#mla-multi-head-latent-attention): `k3_mla_{maybe_rope_f32, sdpa_gate_f32}` (2); [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `rms_norm_vanilla`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16`, `dense_gemm_bf16_pipelined`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemv`, `w8a16_gemv`, `w8a16_gemm`; [Quantization and format conversion](#quantization-and-format-conversion): `dequant_fp8_blockscaled_bf16`, `dequant_{q2_0_gn_to_bf16, q2_k_to_bf16, q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (6), `dequant_nvfp4_to_bf16`, `quantize_bf16_to_fp8_blockscaled`, `f32_to_bf16_trunc`, `nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `quantize_bf16_to_nvfp4_mse`, `transpose_u8`, `widen_block_scale_f32`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`.

## Unique kernels by component

Entry points whose every engine call site belongs to one component.

### Unique to Attention (GQA/MHA: paged decode, split-K, prefill/flash)

193 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w8a16_gemv_batch4::`w8a16_gemv_{batch16_strided, batch4_strided}` (2) | [b300/common/w8a16_gemv_batch4.cu:255][f4] | FP8 GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [1 note][t4] | not measured |
| attn_prefill::`attn_prefill` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | not measured |
| attn_prefill_512tc::`attn_prefill_512tc` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [3 notes][t7] | not measured |
| attn_prefill::`attn_prefill_64` | [gb10/common/attn_prefill.cu:562][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | [27%][m7.attn_prefill_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_fa128::`attn_prefill_{fa128, fa128_paged}` (2) | [gb10/common/attn_prefill_fa128.cu:356][f8] | prefill (flash) | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| gemm_splitk::`dense_gemm_splitk_{partial, reduce}` (2) | [gb10/common/dense_gemm_splitk.cu:27][f15] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t15] | not measured |
| fused_k_norm_rope_cache::`fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2) | [gb10/common/fused_k_norm_rope_cache.cu:53][f34] | cache write | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t34] | not measured |
| paged_decode_attn_bf16_gqa::`paged_decode_attn_bf16_gqa` | [gb10/common/paged_decode_attn_bf16_gqa.cu:46][f112] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t112] | not measured |
| paged_decode_bf16k_turbo2v::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v.cu:85][f113] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t113] | not measured |
| paged_decode_bf16k_turbo2v_128::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu:85][f114] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t114] | not measured |
| paged_decode_bf16k_turbo3v::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v.cu:91][f115] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t115] | not measured |
| paged_decode_bf16k_turbo3v_128::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu:91][f116] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t116] | not measured |
| paged_decode_bf16k_turbo4v::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v.cu:84][f117] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t117] | not measured |
| paged_decode_bf16k_turbo4v_128::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu:84][f118] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t118] | not measured |
| paged_decode_fp8::`paged_decode_attn_{reduce_fp8, splitk_fp8}` (2) | [gb10/common/paged_decode_attn_fp8.cu:346][f119] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t119] | not measured |
| paged_decode_attn_fp8_gqa::`paged_decode_attn_fp8_gqa` | [gb10/common/paged_decode_attn_fp8_gqa.cu:112][f120] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t120] | not measured |
| paged_decode_fp8k_turbo2v::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v.cu:101][f121] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t121] | not measured |
| paged_decode_fp8k_turbo2v_128::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu:101][f122] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t122] | not measured |
| paged_decode_fp8k_turbo3v::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v.cu:112][f123] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t123] | not measured |
| paged_decode_fp8k_turbo3v_128::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu:112][f124] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t124] | not measured |
| paged_decode_fp8k_turbo4v::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v.cu:99][f125] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t125] | not measured |
| paged_decode_fp8k_turbo4v_128::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu:99][f126] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t126] | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/common/paged_decode_attn_nvfp4.cu:87][f127] | paged decode | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (28 ckpts) | [1 note][t127] | not measured |
| paged_decode_turbo2::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2.cu:80][f128] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t128] | not measured |
| paged_decode_attn_turbo2_128::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2_128.cu:80][f129] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t129] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3.cu:110][f130] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t130] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3_128.cu:110][f131] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t131] | not measured |
| paged_decode_turbo3k_turbo8v::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v.cu:119][f132] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t132] | not measured |
| paged_decode_turbo3k_turbo8v_128::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu:119][f133] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t133] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4.cu:95][f134] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t134] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_128.cu:95][f135] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t135] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_512.cu:119][f136] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t136] | not measured |
| paged_decode_turbo4k_turbo3v::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v.cu:122][f137] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t137] | not measured |
| paged_decode_turbo4k_turbo3v_128::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu:122][f138] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t138] | not measured |
| paged_decode_turbo4k_turbo8v::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v.cu:114][f139] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t139] | not measured |
| paged_decode_turbo4k_turbo8v_128::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu:114][f140] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t140] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8.cu:100][f141] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t141] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_128.cu:98][f142] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t142] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_512.cu:125][f143] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t143] | not measured |
| prefill_paged::`attn_prefill_paged_64` | [gb10/common/prefill_paged_compute.cuh:644][f145] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t145] | [23–24%][m145.attn_prefill_paged_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_paged_batched::`attn_prefill_paged_{batched, batched_64, fp8_64, fp8_batched, fp8_batched_64, nvfp4, nvfp4_64, nvfp4_batched, nvfp4_batched_64}` (9) | [gb10/common/prefill_paged_compute.cuh:162][f145] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [5 notes][t145] | not measured |
| prefill_paged_turbo2::`attn_prefill_paged_{turbo2, turbo3_64, turbo4, turbo4_64, turbo8_64}` (5) | [gb10/common/prefill_paged_compute.cuh:162][f145] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [5 notes][t145] | not measured |
| attn_prefill_paged_512::`attn_prefill_paged_512` | [gb10/common/prefill_paged_compute_512.cuh:83][f146] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t146] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v_64, bf16k_turbo3v_64, bf16k_turbo4v_64, fp8k_turbo2v_64, fp8k_turbo3v_64, fp8k_turbo4v_64, turbo3k_turbo8v_64, turbo4k_turbo3v_64, turbo4k_turbo8v_64}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:455][f147] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t147] | not measured |
| reshape_and_cache::`reshape_and_cache_flash_{nvfp4, v_only}` (2) | [gb10/common/reshape_and_cache.cu:30][f154] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t154] | not measured |
| reshape_and_cache_fused_k_fp8::`fused_k_norm_rope_cache_write_fp8_kv` | [gb10/common/reshape_and_cache_fused_k_fp8.cu:131][f155] | cache write | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t155] | not measured |
| reshape_and_cache_turbo::`reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13) | [gb10/common/reshape_and_cache_turbo.cu:179][f156] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t156] | not measured |
| residual_add::`sigmoid_gate_mul_batched`, `sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast` | [gb10/common/residual_add.cu:119][f157] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t157] | not measured |
| norm::`residual_add_rms_norm_vanilla`, `rms_norm_residual_vanilla`, `rms_norm_strided` | [gb10/common/rms_norm.cu:131][f158] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [2 notes][t158] | not measured |
| rms_norm_vanilla::`rms_norm_vanilla_warp_row` | [gb10/common/rms_norm_vanilla.cu:120][f159] | normalization | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t159] | not measured |
| rope::`rope_forward_{proportional, strided, yarn_interleaved, yarn_interleaved_inv, yarn_scaled}` (5) | [gb10/common/rope.cu:147][f160] | rotary | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (29 ckpts) | [3 notes][t160] | not measured |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved` | [gb10/common/rope_mrope_interleaved.cu:34][f161] | rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | [6%][m161.rope_forward_mrope_interleaved] (prefill 32k (cold, 32772 tok)) |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved_k_only` | [gb10/common/rope_mrope_interleaved.cu:108][f161] | rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | not measured |
| ssm_preprocess::`deinterleave_qg_{split, split_qnorm_mrope}` (2) | [gb10/common/ssm_preprocess.cu:137][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| ssm_preprocess::`deinterleave_qg_split_qnorm` | [gb10/common/ssm_preprocess.cu:190][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | [88–90%][m165.deinterleave_qg_split_qnorm] (prefill 4k (cold, 4103 tok)) |
| tq_plus_innerq_apply::`tq_plus_innerq_apply_{k, q}` (2) | [gb10/common/tq_plus_innerq_apply.cu:71][f168] | TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t168] | not measured |
| w4a16_gemv::`w4a16_gemv_qg_{batch2, batch3}` (2) | [gb10/common/w4a16_gemv.cu:1679][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t173] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [gb10/common/w8a16_gemm_t.cu:607][f181] | dequant / repack / transpose | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t181] | not measured |
| w8a16_gemv_batch4::`w8a16_gemv_{batch16_strided, batch4_strided}` (2) | [gb10/common/w8a16_gemv_batch4.cu:276][f184] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t184] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/deepseek-v4-flash/nvfp4/attn_prefill_512.cu:13][f188] | prefill (flash) | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t188] | not measured |
| csa_compress::`csa_compress` | [gb10/deepseek-v4-flash/nvfp4/csa_compress.cu:20][f190] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t190] | not measured |
| grouped_gemm_mla::`grouped_gemm_mla` | [gb10/deepseek-v4-flash/nvfp4/grouped_gemm_mla.cu:35][f192] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat, Mistral4 (4 ckpts) | [2 notes][t192] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:33][f196] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t196] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu:21][f198] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t198] | not measured |
| mla_paged_decode::`mla_paged_decode_nvfp4` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu:78][f199] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t199] | not measured |
| mla_paged_decode_fp8::`mla_paged_decode_fp8` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu:38][f200] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t200] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu:26][f201] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t201] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu:43][f205] | paged decode | b200 gb10 hop | DeepSeek-V4, Gemma4 (4 ckpts) | — | not measured |
| paged_decode_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu:66][f206] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t206] | not measured |
| paged_decode_mla::`paged_decode_attn` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu:59][f207] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t207] | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu:87][f208] | paged decode | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t208] | not measured |
| prefill_attn_compressed::`prefill_attn_compressed` | [gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu:23][f209] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t209] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:571][f210] | FP8 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t210] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512.cu:13][f211] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [2 notes][t211] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu:43][f220] | paged decode | gb10 | Gemma4 (2 ckpts) | — | not measured |
| paged_decode_attn_fp8_512::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu:46][f221] | paged decode | gb10 | Gemma4 (2 ckpts) | [1 note][t221] | not measured |
| norm::`rms_norm_strided` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:1170][f222] | normalization | gb10 | Gemma4 (2 ckpts) | [1 note][t222] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu:12][f223] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [1 note][t223] | not measured |
| norm::`rms_norm_strided` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:1191][f226] | normalization | gb10 | Gemma4 (2 ckpts) | [1 note][t226] | not measured |
| norm::`rms_norm_strided` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:1200][f234] | normalization | b200 gb10 hop | LongCat, MiniMax-M2, Mistral4, Nemotron-H, Step-3.7 (7 ckpts) | [1 note][t234] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/mistral-small-4/nvfp4/mla_absorbed.cu:33][f237] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t237] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu:21][f238] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t238] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu:24][f239] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t239] | not measured |
| paged_decode_attn_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu:66][f240] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t240] | not measured |
| paged_decode_mla::`paged_decode_attn` | [gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu:59][f241] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t241] | not measured |
| norm::`rms_norm_strided` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:1208][f245] | normalization | b200 gb10 hop | Nemotron-H (3 ckpts) | [1 note][t245] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:654][f246] | FP8 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t246] | not measured |
| norm::`rms_norm_strided` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:1160][f249] | normalization | gb10 | Qwen3-VL (1 ckpts) | [1 note][t249] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1004][f260] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t260] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:814][f267] | FP8 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t267] · [#34][pr34] | not measured |
| paged_decode_bf16_splitk_hopper::`paged_decode_attn_{reduce_bf16_hopper, splitk_bf16_hopper}` (2) | [hopper/common/paged_decode_bf16_splitk_hopper.cu:30][f280] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t280] | not measured |
| paged_decode_fp8_splitk_hopper::`paged_decode_attn_{reduce_fp8_hopper, splitk_fp8_hopper}` (2) | [hopper/common/paged_decode_fp8_splitk_hopper.cu:47][f281] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t281] | not measured |
| w8a16_gemm_m16::`w8a16_gemm_m16_strided` | [hopper/common/w8a16_gemm_m16.cu:426][f283] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t283] | not measured |
| w8a16_gemv_ncol::`w8a16_gemv_batch16_{ncol2, ncol2_strided, ncol4, ncol4_strided}` (4) | [hopper/common/w8a16_gemv_ncol.cu:199][f286] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t286] | not measured |
| attn_prefill::`attn_{prefill, prefill_64}` (2) | [strix-hip/common/attn_prefill.cu:69][f330] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t330] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [strix-hip/common/w8a16_gemm_t.cu:198][f338] | dequant / repack / transpose | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t338] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:475][f340] | FP8 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [1 note][t340] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:474][f341] | FP8 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t341] | not measured |

### Unique to Sparse / compressed attention (DSA, CSA/HCA, QSA)

44 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [b300/common/dsa_indexer.cu:74][f2] | DSA indexer / sparse MLA | b300 | none — its callers' targets compile another copy | [1 note][t2] | not measured |
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [gb10/common/dsa_indexer.cu:74][f27] | DSA indexer / sparse MLA | b200 gb10 hop | GLM-5.3 (1 ckpts) | [5 notes][t27] | not measured |
| attn_v41::`attn_v41_{act_quant_fp8, fp4_quant, gemm_f32, gemv_f32_staged, index_score, pool, ring_put, rmsnorm_bf16, rmsnorm_f32, rope, scale_bf16, scatter_cols, slice_cols, sparse_attn}` (14) | [gb10/deepseek-v4-flash/nvfp4/attn_v41.cu:100][f189] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [6 notes][t189] | not measured |
| kquant_moe::`kquant_mmvq_q2_k_{groups_w, pair_w}` (2) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:321][f195] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [5 notes][t195] | not measured |
| glm5next_dsa_mla_decode::`glm5next_dsa_mla_decode_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu:94][f227] | DSA indexer / sparse MLA | gb10 | GLM-5.3 (1 ckpts) | [1 note][t227] | not measured |
| glm5next_mla_latent_write::`glm5next_mla_latent_write_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_mla_latent_write.cu:30][f228] | MLA decode/prefill | gb10 | GLM-5.3 (1 ckpts) | [1 note][t228] | not measured |
| qsa_indexer::`qsa_{block_pool, gather, prefill_attn, qprep, qprep_rows, score, score_rows, score_rows_tc}` (8) | [gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu:70][f271] | QSA sparse attention | gb10 | Qwen3.8-FN (1 ckpts) | [3 notes][t271] | not measured |

### Unique to GDN (gated delta rule linear attention)

217 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| causal_conv1d::`causal_conv1d_update_{chunk2, l2norm_f32, l2norm_f32_strided}` (3) | [gb10/common/causal_conv1d.cu:247][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t13] | not measured |
| dense_gemv_bf16_batch2::`dense_gemv_bf16_batch2` | [gb10/common/dense_gemv_bf16_batch2.cu:32][f18] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t18] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill}` (8) | [gb10/common/gated_delta_rule.cu:233][f35] | delta-rule recurrence | b200 b300 gb10 hop | none — its callers' targets compile another copy | [6 notes][t35] | not measured |
| gated_delta_rule_carry::`gdn_carry_conv` | [gb10/common/gated_delta_rule_carry.cu:343][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [33%][m36.gdn_carry_conv] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_carry::`gdn_carry_{conv_flush, flush, wy2, wy3, wy3_lazy, wy4, wy4_lazy}` (7) | [gb10/common/gated_delta_rule_carry.cu:283][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | not measured |
| gated_delta_rule_carry::`gdn_carry_wy2_lazy` | [gb10/common/gated_delta_rule_carry.cu:286][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [51%][m36.gdn_carry_wy2_lazy] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_{ksplit, pipe, tc_vblock, tma, vtile}` (5) | [gb10/common/gated_delta_rule_fla.cu:857][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [10 notes][t37] | not measured |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_vfused` | [gb10/common/gated_delta_rule_fla.cu:1085][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [25–36%][m37.gated_delta_rule_chunk_delta_h_vfused] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_fwd_o` | [gb10/common/gated_delta_rule_fla.cu:2003][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [32–33%][m37.gated_delta_rule_chunk_fwd_o] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_recompute_wu` | [gb10/common/gated_delta_rule_fla.cu:262][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [11 notes][t37] | [34–40%][m37.gated_delta_rule_recompute_wu] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_persistent::`gated_delta_rule_prefill_{persistent, persistent_batched, persistent_wy4, persistent_wy4_batched}` (4) | [gb10/common/gated_delta_rule_persistent.cu:54][f38] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t38] | not measured |
| gated_delta_rule_regresident::`gated_delta_rule_prefill_regresident` | [gb10/common/gated_delta_rule_regresident.cu:46][f39] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t39] | not measured |
| gated_delta_rule_wy::`gated_delta_rule_wy2` | [gb10/common/gated_delta_rule_wy.cu:30][f40] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t40] | [82%][m40.gated_delta_rule_wy2] (decode C=1 (R=2, MTP k=1)) |
| gated_delta_rule_wy2_resident::`gated_delta_rule_wy2_resident` | [gb10/common/gated_delta_rule_wy2_resident.cu:58][f41] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t41] | not measured |
| gated_delta_rule_wy2_resident_f16::`gated_delta_rule_wy2_resident_f16` | [gb10/common/gated_delta_rule_wy2_resident_f16.cu:65][f42] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t42] | [58%][m42.gated_delta_rule_wy2_resident_f16] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_wy3::`gated_delta_rule_wy3` | [gb10/common/gated_delta_rule_wy3.cu:18][f43] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t43] | not measured |
| gated_delta_rule_wy3_f16::`gated_delta_rule_wy3_f16` | [gb10/common/gated_delta_rule_wy3_f16.cu:40][f44] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t44] | not measured |
| gated_delta_rule_wy3_resident::`gated_delta_rule_wy3_resident` | [gb10/common/gated_delta_rule_wy3_resident.cu:59][f45] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t45] | not measured |
| gated_delta_rule_wy3_resident_f16::`gated_delta_rule_wy3_resident_f16` | [gb10/common/gated_delta_rule_wy3_resident_f16.cu:48][f46] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t46] | not measured |
| gated_delta_rule_wy4::`gated_delta_rule_wy4` | [gb10/common/gated_delta_rule_wy4.cu:18][f47] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t47] | [58%][m47.gated_delta_rule_wy4] (decode C=1 (R=4, MTP k=3)) |
| gated_delta_rule_wy4_f16::`gated_delta_rule_wy4_f16` | [gb10/common/gated_delta_rule_wy4_f16.cu:40][f48] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t48] | not measured |
| gated_delta_rule_wy4_woa::`gated_delta_rule_wy4_{flag_clear, fold, woa}` (3) | [gb10/common/gated_delta_rule_wy4_woa.cu:55][f49] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t49] | not measured |
| gated_delta_rule_wy64_prefill::`gated_delta_rule_prefill_{wy64, wy64_batched}` (2) | [gb10/common/gated_delta_rule_wy64_prefill.cu:41][f50] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t50] | not measured |
| gated_delta_rule_wy_f16::`gated_delta_rule_wy2_f16` | [gb10/common/gated_delta_rule_wy_f16.cu:45][f51] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t51] | not measured |
| gated_delta_rule_wyn::`gated_delta_rule_{wy10, wy10_f16, wy10_f16_table, wy10_table, wy11, wy11_f16, wy11_f16_table, wy11_table, wy12, wy12_f16, wy12_f16_table, wy12_table, wy13, wy13_f16, wy13_f16_table, wy13_table, wy14, wy14_f16, wy14_f16_table, wy14_table, wy15, wy15_f16, wy15_f16_table, wy15_table, wy16, wy16_f16, wy16_f16_table, wy16_table, wy5, wy5_f16, wy5_f16_table, wy5_table, wy6, wy6_f16, wy6_f16_table, wy6_table, wy7, wy7_f16, wy7_f16_table, wy7_table, wy8, wy8_f16, wy8_f16_table, wy8_table, wy9, wy9_f16, wy9_f16_table, wy9_table}` (48) | [gb10/common/gated_delta_rule_wyn.cu:293][f52] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t52] | not measured |
| gdn_chunk_fwd_o_mma8::`gated_delta_rule_chunk_fwd_o_mma8` | [gb10/common/gdn_chunk_fwd_o_mma8.cu:96][f53] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn` | [gb10/common/gdn_verify_fused_conv_kn.cu:50][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn_batched` | [gb10/common/gdn_verify_fused_conv_kn.cu:157][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | [35%][m54.gdn_verify_fused_conv_kn_batched] (decode C=16 (R=32, MTP k=1)) |
| gdn_verify_fused_k2::`gdn_verify_fused_{conv_k2, norm_k2}` (2) | [gb10/common/gdn_verify_fused_k2.cu:60][f55] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t55] | not measured |
| norm::`gated_rms_norm_f32_{input, input_strided}` (2) | [gb10/common/rms_norm.cu:1101][f158] | normalization | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t158] | not measured |
| norm::`gated_rms_norm_prefill` | [gb10/common/rms_norm.cu:1277][f158] | normalization | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t158] | [69–96%][m158.gated_rms_norm_prefill] (decode C=16 (R=32, MTP k=1)) |
| ssm_ba_gates_hopper::`dense_gemm_ba_gates_prefill_hopper` | [gb10/common/ssm_ba_gates_hopper.cu:115][f162] | GDN pre/post-processing | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t162] | not measured |
| ssm_h_dtype::`ssm_h_state_{f16_to_f32, f32_to_f16}` (2) | [gb10/common/ssm_h_dtype.cu:25][f164] | GDN pre/post-processing | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t164] | not measured |
| ssm_preprocess::`compute_gdn_gates`, `deinterleave_qkvz`, `dense_gemv_ba_gates` | [gb10/common/ssm_preprocess.cu:35][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| ssm_preprocess::`dense_gemm_ba_gates_prefill` | [gb10/common/ssm_preprocess.cu:482][f165] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t165] | [8–43%][m165.dense_gemm_ba_gates_prefill] (prefill 32k (cold, 32772 tok)) |
| ssm_state_norm::`ssm_state_clamp_norm_{fused, fused_f16}` (2) | [gb10/common/ssm_state_norm.cu:32][f166] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t166] | not measured |
| w4a16_gemv::`w4a16_gemv_qkvz` | [gb10/common/w4a16_gemv.cu:1557][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t173] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/gemma-4-26b-a4b/nvfp4/gated_delta_rule.cu:24][f213] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t213] | not measured |
| norm::`gated_rms_norm_{f32_input, prefill}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:597][f222] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t222] | not measured |
| norm::`gated_rms_norm_{f32_input, prefill}` (2) | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:618][f226] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t226] | not measured |
| norm::`gated_rms_norm_{f32_input, prefill}` (2) | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:508][f234] | normalization | b200 gb10 hop | none — its callers' targets compile another copy | [1 note][t234] | not measured |
| norm::`gated_rms_norm_{f32_input, prefill}` (2) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:516][f245] | normalization | b200 gb10 hop | none — its callers' targets compile another copy | [1 note][t245] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/qwen3-next-80b-a3b/nvfp4/gated_delta_rule.cu:24][f248] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t248] | not measured |
| norm::`gated_rms_norm_{f32_input, prefill}` (2) | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:437][f249] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t249] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/qwen3.5-122b-a10b/nvfp4/gated_delta_rule.cu:24][f251] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t251] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f16_norm, decode_f16_strided_norm_half, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, decode_f32_strided_norm_half, decode_f32_strided_norm_smem, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (15) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu:24][f252] | delta-rule recurrence | gb10 hop strix hip | Qwen-GDN (7 ckpts) | [6 notes][t252] | not measured |
| gated_delta_rule_snap::`gated_delta_rule_decode_f32_{norm_snap, strided_norm_snap}` (2) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule_snap.cu:66][f253] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t253] | not measured |
| gdn_verify_fused_conv_kn_f32::`gdn_verify_fused_conv_kn_f32` | [gb10/qwen3.6-27b/nvfp4/gdn_verify_fused_conv_kn_f32.cu:33][f254] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t254] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu:59][f263] | delta-rule recurrence | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t263] | not measured |
| gated_delta_rule_wy17::`gated_delta_rule_wy17` | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule_wy17.cu:41][f264] | delta-rule recurrence | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t264] | not measured |
| gated_norm_sigmoid::`gated_rms_norm_{f32_input_sigmoid, prefill_sigmoid, sigmoid}` (3) | [gb10/qwen3.8-flash-next/nvfp4/gated_norm_sigmoid.cu:50][f268] | normalization | gb10 | Qwen3.8-FN (1 ckpts) | [1 note][t268] | not measured |
| gated_delta_rule_chunk_tc::`gated_delta_rule_chunk_delta_h_tcfuse_x2` | [hopper/common/gated_delta_rule_chunk_tc.cu:409][f275] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [5 notes][t275] | not measured |
| gdn_fwd_o_hopper::`gated_delta_rule_chunk_fwd_o_hopper` | [hopper/common/gdn_fwd_o_hopper.cu:94][f276] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t276] | not measured |
| gdn_recompute_wu_hopper::`gated_delta_rule_recompute_wu_hopper` | [hopper/common/gdn_recompute_wu_hopper.cu:138][f277] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t277] | not measured |
| add_rms_norm::`add_rms_norm` | [metal/common/add_rms_norm.metal:32][f287] | normalization | metal | Qwen-GDN (7 ckpts) | [1 note][t287] | not measured |
| attention_decode::`attention_decode` | [metal/common/attention_decode.metal:30][f289] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t289] | not measured |
| attention_decode_bf16k_turbov::`attention_decode_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/attention_decode_bf16k_turbov.metal:113][f290] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t290] | not measured |
| attention_decode_turbo2::`attention_decode_turbo2` | [metal/common/attention_decode_turbo2.metal:39][f291] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t291] | not measured |
| attention_decode_turbo3::`attention_decode_turbo3` | [metal/common/attention_decode_turbo3.metal:53][f292] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t292] | not measured |
| attention_decode_turbo4::`attention_decode_turbo4` | [metal/common/attention_decode_turbo4.metal:41][f293] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t293] | not measured |
| attention_decode_turbo8::`attention_decode_turbo8` | [metal/common/attention_decode_turbo8.metal:38][f294] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t294] | not measured |
| causal_conv1d_update_l2norm::`causal_conv1d_update_l2norm` | [metal/common/causal_conv1d_update_l2norm.metal:41][f299] | causal conv1d | metal | Qwen-GDN (7 ckpts) | [1 note][t299] | not measured |
| gdn_helpers::`gdn_compute_gate` | [metal/common/gdn_helpers.metal:28][f305] | delta-rule recurrence | metal | Qwen-GDN (7 ckpts) | — | not measured |
| gdn_helpers::`sigmoid_bf16_to_f32` | [metal/common/gdn_helpers.metal:56][f305] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |
| kv_cache_append::`kv_cache_append` | [metal/common/kv_cache_append.metal:19][f307] | cache write | metal | Qwen-GDN (7 ckpts) | — | not measured |
| kv_cache_append_bf16k_turbov::`kv_cache_append_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/kv_cache_append_bf16k_turbov.metal:129][f308] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t308] | not measured |
| kv_cache_append_turbo2::`kv_cache_append_turbo2` | [metal/common/kv_cache_append_turbo2.metal:53][f309] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t309] | not measured |
| kv_cache_append_turbo3::`kv_cache_append_turbo3` | [metal/common/kv_cache_append_turbo3.metal:65][f310] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t310] | not measured |
| kv_cache_append_turbo4::`kv_cache_append_turbo4` | [metal/common/kv_cache_append_turbo4.metal:82][f311] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t311] | not measured |
| kv_cache_append_turbo8::`kv_cache_append_turbo8` | [metal/common/kv_cache_append_turbo8.metal:47][f312] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t312] | not measured |
| qwen35_qkv_split::`qwen35_qkv_split` | [metal/common/qwen35_qkv_split.metal:20][f322] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |
| rms_norm::`rms_norm` | [metal/common/rms_norm.metal:21][f323] | normalization | metal | Qwen-GDN (7 ckpts) | [1 note][t323] | not measured |
| rope_apply::`rope_apply` | [metal/common/rope_apply.metal:33][f324] | rotary | metal | Qwen-GDN (7 ckpts) | — | not measured |

### Unique to KDA (Kimi delta attention, linear attention)

10 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| kda_chunk::`kda_chunk_{prepare, scan}` (2) | [gb10/common/kda_chunk.cu:95][f58] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t58] | not measured |
| kda_gate::`kda_gate_bf16` | [gb10/common/kda_gate.cu:74][f59] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | — | not measured |
| kda_layer_ops::`kda_{fill_f32, o_norm_gated_bf16, pack_qkv_bf16, sigmoid_bf16_f32, split_widen}` (5) | [gb10/common/kda_layer_ops.cu:50][f60] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t60] | not measured |
| kda_recurrent::`kda_recurrent_decode_{bf16, bf16_smem}` (2) | [gb10/common/kda_recurrent.cu:144][f61] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t61] | not measured |

### Unique to Mamba2 (selective state-space scan)

7 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| mamba2_ssd_chunk::`mamba2_ssd_{bmm, cumsum, scan}` (3) | [gb10/common/mamba2_ssd_chunk.cu:44][f63] | SSD / selective scan | b200 b300 gb10 hop | Nemotron-H (3 ckpts) | [2 notes][t63] | not measured |
| mamba2_ssm::`mamba2_ssm_{decode, prefill, prefill_persistent}` (3) | [gb10/common/mamba2_ssm_decode.cu:28][f64] | SSD / selective scan | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [2 notes][t64] | not measured |
| w4a16::`fp8_fp8_gemm_t_m128_mfast` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:1678][f246] | FP8 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t246] | not measured |

### Unique to MoE (routing, dispatch, expert GEMM/GEMV, combine)

196 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [b300/common/moe_shared_expert_fused.cu:48][f3] | expert GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [2 notes][t3] | not measured |
| gemm::`dense_gemm_bf16_router` | [gb10/common/dense_gemm_bf16.cu:204][f14] | routing / top-k | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t14] | [6%][m14.dense_gemm_bf16_router] (prefill 32k (cold, 32772 tok)) |
| gemm::`dense_gemm_f32in_f32out` | [gb10/common/dense_gemm_bf16.cu:131][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| glm5next_ffn::`glm5next_moe_{combine, combine_indexed}` (2) | [gb10/common/glm5next_ffn.cu:213][f56] | dispatch / combine | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_router_topk` | [gb10/common/glm5next_ffn.cu:102][f56] | routing / top-k | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_swiglu_clamp` | [gb10/common/glm5next_ffn.cu:31][f56] | expert activation | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| moe_bf16_grouped_gemm::`moe_bf16_grouped_gemm` | [gb10/common/moe_bf16_grouped_gemm.cu:86][f66] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t66] | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_finalize` | [gb10/common/moe_decode_atomic_c4.cu:183][f67] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t67] | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_silu_down_accum` | [gb10/common/moe_decode_atomic_c4.cu:37][f67] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t67] | not measured |
| moe_relu2_fused::`moe_expert_relu2_down_shared` | [gb10/common/moe_expert_relu2_down_shared.cu:53][f70] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t70] | not measured |
| moe_fp8_grouped_blend::`moe_weighted_sum_blend_fp8_grouped` | [gb10/common/moe_fp8_grouped_blend.cu:17][f71] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t71] · [#4][pr4] | [16–77%][m71.moe_weighted_sum_blend_fp8_grouped] (decode C=1 (R=2, MTP k=1)) |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [gb10/common/moe_fp8_grouped_gemm.cu:281][f72] | expert GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t72] | not measured |
| moe_fp8_grouped_sort::`moe_fp8_grouped_sort` | [gb10/common/moe_fp8_grouped_sort.cu:24][f73] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t73] · [#34][pr34] | [1%][m73.moe_fp8_grouped_sort] (decode C=1 (R=2, MTP k=1)) |
| moe_gate_topk::`moe_gate_topk_fused` | [gb10/common/moe_gate_topk.cu:46][f74] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t74] | not measured |
| moe_hash_route::`moe_hash_{route, route_batched}` (2) | [gb10/common/moe_hash_route.cu:25][f75] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t75] | not measured |
| moe_nvfp4_grouped::`moe_expert_{down_act_nvfp4_grouped, gate_up_act_nvfp4_grouped}` (2) | [gb10/common/moe_nvfp4_grouped.cu:72][f78] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [5 notes][t78] · [#34][pr34] | not measured |
| moe::`moe_batched_blend` | [gb10/common/moe_permute.cu:126][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | [97–100%][m79.moe_batched_blend] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_build_tile_worklist` | [gb10/common/moe_permute.cu:276][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | [0%][m79.moe_build_tile_worklist] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_{permute_tokens, sort_by_expert}` (2) | [gb10/common/moe_permute.cu:20][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | not measured |
| moe::`moe_unpermute_reduce_indexed` | [gb10/common/moe_permute.cu:95][f79] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t79] | [87–88%][m79.moe_unpermute_reduce_indexed] (prefill 32k (cold, 32772 tok)) |
| moe_prefill::`moe_expert_{gate_up_shared_prefill, silu_down_shared_prefill}` (2) | [gb10/common/moe_prefill.cu:59][f80] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t80] | not measured |
| moe_prefill::`moe_weighted_sum_blend_prefill` | [gb10/common/moe_prefill.cu:360][f80] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t80] | not measured |
| moe_router_gemm::`moe_router_gemm_bf16` | [gb10/common/moe_router_gemm.cu:32][f81] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t81] · [#34][pr34] | not measured |
| moe_router_gemm_prefill::`moe_router_gemm_rt` | [gb10/common/moe_router_gemm_prefill.cu:24][f82] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/common/moe_shared_expert_fused.cu:48][f83] | expert GEMM/GEMV | b200 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t83] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/common/moe_shared_expert_fused_batch2.cu:292][f84] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t84] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/common/moe_shared_expert_fused_batch2.cu:497][f84] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t84] | not measured |
| moe_shared_expert_fused_batch2_t::`moe_expert_{gate_up_shared_batch2_t, silu_down_shared_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_batch2_t.cu:41][f85] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t85] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/common/moe_shared_expert_fused_batch3.cu:50][f86] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t86] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/common/moe_shared_expert_fused_batch3.cu:335][f86] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t86] | not measured |
| moe_shared_expert_fused_batch3_t::`moe_expert_{gate_up_shared_batch3_t, silu_down_shared_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_batch3_t.cu:36][f87] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | not measured |
| moe_shared_expert_fused_bf16::`moe_expert_{gate_up_shared_bf16, silu_down_shared_bf16}` (2) | [gb10/common/moe_shared_expert_fused_bf16.cu:26][f88] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t88] | not measured |
| moe_shared_expert_fused_bf16_batch2::`moe_expert_{gate_up_shared_bf16_batch2, silu_down_shared_bf16_batch2}` (2) | [gb10/common/moe_shared_expert_fused_bf16_batch2.cu:39][f89] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t89] | not measured |
| moe_shared_expert_fused_fp8::`moe_expert_gate_up_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:98][f90] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t90] | [85%][m90.moe_expert_gate_up_shared_fp8] (decode C=1 (R=2, MTP k=1)) |
| moe_shared_expert_fused_fp8::`moe_expert_silu_down_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:262][f90] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t90] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_expert_{gate_up_shared_fp8_batch2, silu_down_shared_fp8_batch2}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:97][f91] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t91] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_weighted_sum_blend_fp8_batch2` | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:410][f91] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t91] | not measured |
| moe_shared_expert_fused_fp8_batch2_t::`moe_expert_{gate_up_shared_fp8_batch2_t, silu_down_shared_fp8_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2_t.cu:28][f92] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t92] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_expert_{gate_up_shared_fp8_batch3, silu_down_shared_fp8_batch3}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:97][f93] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t93] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_weighted_sum_blend_fp8_batch3` | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:408][f93] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t93] | not measured |
| moe_shared_expert_fused_fp8_batch3_t::`moe_expert_{gate_up_shared_fp8_batch3_t, silu_down_shared_fp8_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3_t.cu:28][f94] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t94] | not measured |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_down_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:331][f95] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [12 notes][t95] · [#4][pr4] [#34][pr34] | [89–102%][m95.moe_expert_down_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_gate_up_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:153][f95] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [11 notes][t95] · [#4][pr4] [#34][pr34] | [88–97%][m95.moe_expert_gate_up_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_t::`moe_expert_{gate_up_shared_fp8_t, silu_down_shared_fp8_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_t.cu:34][f96] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t96] | not measured |
| moe_shared_expert_fused_t::`moe_expert_{gate_up_shared_t, gate_up_shared_t_e8m0, silu_down_shared_t, silu_down_shared_t_e8m0}` (4) | [gb10/common/moe_shared_expert_fused_t.cu:187][f97] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t97] | not measured |
| moe_silu_mul::`silu_mul_quant_fp8` | [gb10/common/moe_silu_mul.cu:107][f98] | activation quantize | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN (19 ckpts) | [1 note][t98] | [21–24%][m98.silu_mul_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| moe_sorted::`moe_sorted_{gate_up, silu_down}` (2) | [gb10/common/moe_sorted_prefill.cu:54][f99] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t99] | not measured |
| moe_topk::`moe_topk_softmax_{batched, f32}` (2) | [gb10/common/moe_topk.cu:219][f100] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t100] | not measured |
| moe_topk::`moe_topk_softmax_rows` | [gb10/common/moe_topk.cu:197][f100] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t100] · [#34][pr34] | [0%][m100.moe_topk_softmax_rows] (decode C=1 (R=2, MTP k=1)) |
| moe_topk_sig::`moe_topk_{sigmoid, sigmoid_batched}` (2) | [gb10/common/moe_topk_sigmoid.cu:22][f101] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t101] | not measured |
| moe_topk_softmax_bias::`moe_topk_softmax_{bias, bias_batched}` (2) | [gb10/common/moe_topk_softmax_bias.cu:189][f102] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t102] | not measured |
| moe_topk_softmax_bias::`moe_zero_expert_add` | [gb10/common/moe_topk_softmax_bias.cu:233][f102] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t102] | not measured |
| moe_topk_sqrt::`moe_topk_{sqrtsoftplus, sqrtsoftplus_batched}` (2) | [gb10/common/moe_topk_sqrtsoftplus.cu:22][f103] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t103] | not measured |
| moe_transpose_batched::`moe_transpose_u8_batched` | [gb10/common/moe_transpose_batched.cu:21][f104] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t104] | not measured |
| moe_unpermute_blend::`moe_unpermute_blend` | [gb10/common/moe_unpermute_blend.cu:17][f105] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_k32, ptrtable_t}` (3) | [gb10/common/moe_w4a16_grouped_gemm.cu:231][f106] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3, Gemma4, Nemotron-H (6 ckpts) | [4 notes][t106] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_ptrtable_{alkm_m16_k128, bt_k128, bt_m128_k64, bt_m128_k64_mfast, bt_m16_k128, bt_m16_k128_mfast, bt_m16_n128_k128, bt_m64_k128_mfast, k64, m16_k64}` (10) | [gb10/common/moe_w4a16_grouped_gemm.cu:973][f106] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t106] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm` | [gb10/common/moe_w8a8_grouped_gemm.cu:93][f107] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t107] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm_pm4` | [gb10/common/moe_w8a8_grouped_gemm.cu:396][f107] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [4 notes][t107] | [18–32%][m107.moe_w8a8_grouped_gemm_pm4] (prefill 32k (cold, 32772 tok)) |
| moe_w8a8_grouped_gemm_e4m3::`moe_w8a8_{gateup_silu_e4m3_w1, gateup_silu_e4m3_w2, grouped_gemm_e4m3_dn, grouped_gemm_e4m3_gu}` (4) | [gb10/common/moe_w8a8_grouped_gemm_e4m3.cu:101][f108] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_{relu2_down_prefill, up_prefill}` (2) | [gb10/common/nemotron_moe_prefill.cu:161][f109] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_topk_sigmoid_batched` | [gb10/common/nemotron_moe_prefill.cu:53][f109] | routing / top-k | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t109] | not measured |
| nemotron_moe_prefill::`nemotron_moe_weighted_sum_prefill` | [gb10/common/nemotron_moe_prefill.cu:444][f109] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| relu2::`moe_weighted_sum_scale` | [gb10/common/relu_squared.cu:57][f153] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t153] | not measured |
| relu2::`relu_squared_inplace` | [gb10/common/relu_squared.cu:26][f153] | activation / gate / residual | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| w4a16_gemv::`glm5next_moe_row_union` | [gb10/common/w4a16_gemv.cu:2201][f173] | dispatch / combine | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [2 notes][t173] | not measured |
| w4a16_gemv::`w4a16_gemv_sw_{moe, moe_batchm_m2, moe_batchm_m3, moe_batchm_m4, moe_batchm_m5, moe_batchm_m6, moe_batchm_m7, moe_batchm_m8}` (8) | [gb10/common/w4a16_gemv.cu:300][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [1 note][t173] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k_experts_w2, q2_k_experts_w8, q3_k_experts_w2, q3_k_experts_w8}` (4) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:397][f195] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [6 notes][t195] | not measured |
| kquant_moe::`kquant_mmvq_q3_k_w`, `metrale_q3_k_mmq128_nc`, `metrale_q3_k_mmq128_wc` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:78][f195] | integer / K-quant GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t195] | not measured |
| kquant_moe::`kquant_swiglu_q8_1_rows_bf16`, `metrale_q8_1_quantize_d4_bf16` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:86][f195] | activation quantize | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t195] | not measured |
| moe_v41::`moe_v41_{accumulate, finish, gather_rows, scatter_add, slot_table_set, sum_rows, swiglu}` (7) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:16][f203] | dispatch / combine | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t203] | not measured |
| moe_v41::`moe_v41_{route_select, router_gemv_f32out, router_gemv_f32out_products}` (3) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:145][f203] | routing / top-k | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t203] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_e8m0, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_e8m0, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_e8m0, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_e8m0, w4a16_grouped_gemm_ptrtable_t_k64, w4a16_grouped_gemm_ptrtable_t_k64_e8m0}` (11) | [gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu:188][f204] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, Kimi-K3, LongCat (4 ckpts) | [1 note][t204] | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused.cu:31][f216] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t216] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:35][f217] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t217] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:327][f217] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t217] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:33][f218] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t218] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:323][f218] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t218] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/gemma-4-26b-a4b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f219] | expert GEMM/GEMV | b200 gb10 hop | Gemma4, Mistral4, Qwen-GDN-MoE, Qwen3-VL (10 ckpts) | [1 note][t219] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_m128, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (7) | [gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f233] | expert GEMM/GEMV | gb10 | Laguna, MiniMax-M2, Step-3.7 (4 ckpts) | [1 note][t233] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_relu2, ptrtable_t}` (3) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu:590][f243] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [2 notes][t243] | not measured |
| moe_w4a4::`moe_w4a4_grouped_gemm_relu2` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu:49][f244] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t244] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:135][f255] | expert GEMM/GEMV | gb10 hop strix | none — its callers' targets compile another copy | [1 note][t255] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_down_t_k64_fp4, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_fp4, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_k32, w4a16_grouped_gemm_ptrtable_m256, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (10) | [gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f265] | expert GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN-MoE, Qwen3.8-FN (7 ckpts) | [9 notes][t265] | not measured |
| moe_bucket_builder::`bucket_builder` | [hopper/common/moe_bucket_builder.cu:5][f278] | dispatch / combine | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [2 notes][t278] · [#25][pr25] | not measured |
| moe_w8a8_m16::`pm4_m16` | [hopper/common/moe_w8a8_m16.cu:106][f279] | expert GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [3 notes][t279] · [#25][pr25] | not measured |
| gemm::`dense_gemm_f32in_f32out` | [strix-hip/common/dense_gemm_bf16.cu:129][f334] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [1 note][t334] | not measured |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [strix-hip/common/moe_fp8_grouped_gemm.cu:249][f336] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t336] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [strix-hip/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:61][f339] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [4 notes][t339] | not measured |

### Unique to Dense FFN (gate/up/down projections of non-MoE layers)

34 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dense_f32io::`k3_dense_{down_f32io, gate_up_situ_f32io}` (2) | [b200/kimi-k3/bf16/dense_f32io.cu:12][f1] | BF16/F32 GEMM/GEMV | b200 | Kimi-K3 (1 ckpts) | [1 note][t1] | not measured |
| q2_0_gemv_vec::`q2_0_gemv_vec_batchm` | [gb10/common/q2_0_gemv_vec.cu:162][f149] | integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t149] | not measured |
| w4a16_gemv_fused::`w4a16_gemv_{dual_sw, silu_input, silu_input_sw}` (3) | [gb10/common/w4a16_gemv_fused.cu:209][f174] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [2 notes][t174] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [gb10/common/w8a16_gemv_fused.cu:123][f185] | FP8 GEMM/GEMV | b200 b300 gb10 | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t185] | not measured |
| w4a4::`w4a4_gemm` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a4_gemm.cu:115][f247] | W4A4 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | — | not measured |
| nvfp4_mmq::`metrale_nvfp4_mmq128_nc` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:67][f256] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [17–18%][m256.metrale_nvfp4_mmq128_nc] (prefill 4k (cold, 4103 tok)) |
| nvfp4_mmq::`metrale_nvfp4_{mmq128_wc, mmq16_nc, mmq16_wc, mmq32_wc, mmq64_nc, mmq64_wc}` (6) | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:72][f256] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | not measured |
| nvfp4_mmq::`metrale_nvfp4_mmq32_nc` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:97][f256] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [82–87%][m256.metrale_nvfp4_mmq32_nc] (decode C=16 (R=32, MTP k=1)) |
| nvfp4_mmq::`metrale_nvfp4_quantize_bf16` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:132][f256] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [28–54%][m256.metrale_nvfp4_quantize_bf16] (decode C=16 (R=32, MTP k=1)) |
| nvfp4_mmq::`metrale_nvfp4_repack` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:142][f256] | dequant / repack / transpose | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | not measured |
| nvfp4_mmq::`metrale_nvfp4_scale_bf16` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:214][f256] | dequant / repack / transpose | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | [99–103%][m256.metrale_nvfp4_scale_bf16] (prefill 32k (cold, 32772 tok)) |
| nvfp4_mmq::`metrale_nvfp4_silu_mul_quant` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:242][f256] | activation / gate / residual | gb10 hop | Qwen-GDN (7 ckpts) | [4 notes][t256] | [57–87%][m256.metrale_nvfp4_silu_mul_quant] (decode C=16 (R=32, MTP k=1)) |
| nvfp4_mmq::`metrale_nvfp4_silu_mul_scaled` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:221][f256] | activation / gate / residual | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | not measured |
| q4k_mmq::`metrale_q4k_mmq128_{nc, wc}` (2) | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:50][f258] | integer / K-quant GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t258] | not measured |
| q4k_quantize::`q4k_quantize` | [gb10/qwen3.6-27b/nvfp4/q4k_quantize.cu:81][f259] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t259] | not measured |
| w4a16::`int8_gemm_faith2`, `int8_gemm_i32acc`, `requant_a_bf16_int8`, `requant_w_nvfp4_int8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:4569][f260] | integer / K-quant GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t260] | not measured |
| w4a4::`w4a4_gemm` | [gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu:50][f262] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t262] | not measured |
| silu_mul_strided::`silu_mul_strided` | [hopper/common/silu_mul_strided.cu:44][f282] | activation / gate / residual | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t282] | not measured |
| w8a16_gemm_m16::`w8a16_gemm_m16_n64` | [hopper/common/w8a16_gemm_m16.cu:407][f283] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t283] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [hopper/common/w8a16_gemv_fused.cu:66][f285] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t285] | not measured |

### Unique to Projection GEMM/GEMV — BF16/F32

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| gemm_tc::`dense_gemm_tc_scaled_acc` | [gb10/common/dense_gemm_tc.cu:197][f16] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t16] | not measured |
| dense_gemv_bf16_tc::`dense_gemv_bf16_tc16` | [gb10/common/dense_gemv_bf16_tc.cu:251][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | [86–95%][m20.dense_gemv_bf16_tc16] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_{tc32, tc8}` (2) | [gb10/common/dense_gemv_bf16_tc.cu:250][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | not measured |
| dense_gemm_m16_bf16::`dense_gemm_m16_{bf16, bf16_n64}` (2) | [hopper/common/dense_gemm_m16_bf16.cu:339][f273] | BF16/F32 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t273] | not measured |

### Unique to Projection GEMM/GEMV — FP8 (W8A16, W8A8, block-scaled)

7 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dense_gemv_fp8w_batch2::`dense_gemv_fp8w_batch2` | [gb10/common/dense_gemv_fp8w_batch2.cu:72][f22] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t22] | not measured |
| fp8_gemm_blockscaled_pipe::`fp8_gemm_blockscaled_pipe_128x64` | [gb10/common/fp8_gemm_blockscaled_pipe.cu:63][f30] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_fp8_ldmab::`fp8_fp8_gemm_ldmab` | [gb10/common/w4a16_fp8_ldmab.cu:65][f171] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t171] | [11–43%][m171.fp8_fp8_gemm_ldmab] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm_pipe128::`w8a16_gemm_pipe128` | [gb10/common/w8a16_gemm_pipe128.cu:58][f178] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu:72][f235] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t235] | not measured |
| w4a16_v3::`w4a16_gemm_t_m128_v3` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v3.cu:73][f236] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t236] | not measured |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm_v2.cu:100][f261] | FP8 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t261] | not measured |

### Unique to Projection GEMM/GEMV — NVFP4 W4A16

14 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a16_gemv::`w4a16_gemv_{batch8, batch8_rt2, logits}` (3) | [gb10/common/w4a16_gemv.cu:363][f173] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t173] | not measured |
| w4a16_gemv_tc::`w4a16_gemv_tc16` | [gb10/common/w4a16_gemv_tc.cu:256][f175] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t175] · [#1][pr1] | [86–92%][m175.w4a16_gemv_tc16] (decode C=16 (R=32, MTP k=1)) |
| w4a16_gemv_tc::`w4a16_gemv_tc8` | [gb10/common/w4a16_gemv_tc.cu:255][f175] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t175] · [#1][pr1] | [59–90%][m175.w4a16_gemv_tc8] (decode C=1 (R=4, MTP k=3)) |
| w4a16::`w4a16_gemm_t_k64` | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:695][f210] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t210] | not measured |
| w4a16::`w4a16_gemm_t_k64` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:778][f246] | NVFP4 W4A16 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t246] | not measured |
| w4a16::`w4a16_gemm_t_{k64, k64_p3}` (2) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1121][f260] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t260] | not measured |
| w4a16::`w4a16_gemm_t_k64_n64_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1648][f260] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t260] | [54%][m260.w4a16_gemm_t_k64_n64_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_gemm_t_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:585][f260] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t260] | [8–78%][m260.w4a16_gemm_t_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_gemm_t_k64` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:931][f267] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t267] · [#34][pr34] | not measured |
| w4a16::`w4a16_gemm_t_k64` | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:570][f340] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [2 notes][t340] | not measured |
| w4a16::`w4a16_gemm_t_k64` | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:569][f341] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [3 notes][t341] | not measured |

### Unique to Projection GEMM/GEMV — W4A4 (FP4 activations)

11 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a4_gemv_mx::`w4a4_{gemv_mx16, gemv_mx16_nt2, gemv_mx16_ps, gemv_mx32, gemv_mx32_nt4, gemv_mx32_ps, gemv_mx64, gemv_mx64_nt2, gemv_mx8, quant_rows}` (10) | [gb10/common/w4a4_gemv_mx.cu:360][f176] | W4A4 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [23 notes][t176] · [#1][pr1] [#14][pr14] [#18][pr18] | not measured |
| nvfp4_mmq::`metrale_nvfp4_gemm_pipe` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:356][f256] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t256] | not measured |

### Unique to Projection GEMM/GEMV — integer / K-quant (Q2_0, Q2_K..Q6_K, INT8, MLX INT8)

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| mlx_int8_dequant::`mlx_int8_dequant` | [metal/common/mlx_int8_dequant.metal:21][f315] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | — | not measured |
| mlx_int8_gemm::`mlx_int8_gemm` | [metal/common/mlx_int8_gemm.metal:23][f316] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t316] | not measured |
| mlx_int8_gemv::`mlx_int8_gemv` | [metal/common/mlx_int8_gemv.metal:39][f317] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t317] | not measured |
| mlx_int8_gemv_gate_up::`mlx_int8_gemv_gate_up` | [metal/common/mlx_int8_gemv_gate_up.metal:41][f318] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t318] | not measured |
| mlx_int8_gemv_silu_gate::`mlx_int8_gemv_silu_{gate, gate_resid}` (2) | [metal/common/mlx_int8_gemv_silu_gate.metal:29][f319] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [2 notes][t319] | not measured |

### Unique to KV cache (write, quantize, TurboQuant rotation, slot metadata)

1 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| reshape_and_cache::`bf16_absmax` | [gb10/common/reshape_and_cache.cu:443][f154] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t154] | not measured |

### Unique to Quantization and format conversion

4 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| per_token_group_quant_fp8::`per_token_group_quant_fp8` | [gb10/common/per_token_group_quant_fp8.cu:39][f144] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t144] | [81–83%][m144.per_token_group_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| quant_rowwise_fp8::`quant_rowwise_fp8` | [gb10/common/quant_rowwise_fp8.cu:38][f150] | activation quantize | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t150] | not measured |
| w4a16_fp8_ldmab::`fp8_predequant_nvfp4_t` | [gb10/common/w4a16_fp8_ldmab.cu:193][f171] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t171] | not measured |
| fp8_act_quant_hopper::`per_token_group_quant_fp8_hopper` | [hopper/common/fp8_act_quant_hopper.cu:94][f274] | activation quantize | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t274] | not measured |

### Unique to Embedding and LM head (lookup, overlays, softcap, scale)

7 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| token_overlay::`embed_overlay_routed_bf16`, `embed_rowdiff_bf16`, `lmhead_overlay_routed_bf16`, `lmhead_overlay_routed_f32` | [gb10/common/token_overlay.cu:20][f167] | embedding / LM head | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t167] | not measured |
| kquant_moe::`kquant_mmvq_q6_k_w` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:356][f195] | integer / K-quant GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t195] | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-26b-a4b/nvfp4/logit_softcap.cu:12][f215] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t215] | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-31b/nvfp4/logit_softcap.cu:12][f225] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t225] | not measured |

### Unique to Sampling (argmax, top-p, feed-forward of the chosen token)

3 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| argmax::`argmax_fp32` | [gb10/common/argmax_bf16.cu:194][f5] | argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t5] | not measured |
| argmax_feed::`argmax_bf16_batch_feed`, `feed_resolve` | [gb10/common/argmax_feed.cu:44][f6] | argmax / top-p | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t6] | not measured |

### Unique to Speculative decoding (MTP heads, DFlash drafter, verify helpers)

16 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| argmax::`argmax_bf16_batch_lp` | [gb10/common/argmax_bf16.cu:128][f5] | argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t5] | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [gb10/common/attn_prefill_h128.cu:46][f10] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t10] | not measured |
| dflash2::`dflash2_{conv2, selector_walk, topk16}` (3) | [gb10/common/dflash2.cu:31][f26] | DFlash drafter | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [3 notes][t26] | not measured |
| fp8_gemv_rt::`fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2) | [gb10/common/fp8_gemv_rt.cu:156][f32] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t32] | not measured |
| prefill_paged_indirect::`attn_prefill_paged_indirect` | [gb10/common/prefill_paged_compute.cuh:162][f145] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t145] | not measured |
| residual_add::`bf16_concat` | [gb10/common/residual_add.cu:142][f157] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | [4%][m157.bf16_concat] (prefill 4k (cold, 4549 tok)) |
| w4a16::`fp8_gemm_t_row_{scaled, scaled_k64, scaled_m16, scaled_p4}` (4) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:6843][f260] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [4 notes][t260] | not measured |
| w4a16::`fp8_gemm_t_row_{scaled, scaled_m16}` (2) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:1748][f267] | FP8 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t267] · [#34][pr34] | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [strix-hip/common/attn_prefill_h128.cu:185][f332] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t332] | not measured |

### Unique to Hyper-connections (mHC)

13 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm5next_mhc::`glm5next_hc_{expand, finish, head, mix, mix_bf16, post, pre}` (7) | [gb10/common/glm5next_mhc.cu:64][f57] | hyper-connection mix | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t57] | not measured |
| hyper_connection::`hc_pre_down`, `hc_pre_finish`, `hc_pre_mix`, `hc_pre_stage`, `hc_pre_stage_bf16`, `hc_silu_scale` | [gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu:329][f269] | hyper-connection mix | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t269] | not measured |

### Unique to N-gram and memory embeddings (Engram, PLE, n-gram tables)

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| embed_from_argmax::`batched_embed_fp8` | [gb10/common/embed_from_argmax.cu:110][f29] | embedding / LM head | b200 b300 gb10 hop strix hip | LongCat (1 ckpts) | [2 notes][t29] | not measured |
| engram_v41::`engram_v41_{gate, wkv_q2k_gemv}` (2) | [gb10/deepseek-v4-flash/nvfp4/engram_v41.cu:43][f191] | memory embedding | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t191] | not measured |
| ple::`ple_{add_highway, conv, gate}` (3) | [gb10/qwen3.8-flash-next/nvfp4/ple.cu:90][f270] | memory embedding | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t270] | not measured |

### Unique to Vision encoder (ViT towers)

32 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm_vit::`glm_vit_{add_bias, add_inplace, copy, f32_to_bf16, gelu_erf, im2col_2x2, layernorm, qknorm_rope_deint, rmsnorm, scatter_head, softmax_rows, swiglu_clamp}` (12) | [gb10/glm-5.3-flash/nvfp4/glm_vit.cu:42][f229] | ViT op | gb10 | GLM-5.3 (1 ckpts) | [2 notes][t229] | not measured |
| vision_encoder::`vision_{add_inplace, attention_rope, bf16_copy, f32_to_bf16, gelu, gemm_bias, layer_norm, spatial_merge}` (8) | [gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu:23][f250] | ViT op | gb10 | Qwen-GDN-MoE, Qwen3-VL (7 ckpts) | [2 notes][t250] | not measured |
| vision_encoder::`vision_add_bias`, `vision_add_inplace`, `vision_attention_rope`, `vision_bf16_copy`, `vision_f32_to_bf16`, `vision_gelu`, `vision_gemm_bias`, `vision_layer_norm`, `vision_spatial_merge`, `vit_rope_deinterleave`, `vit_scatter_head`, `vit_softmax_rows` | [gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu:23][f266] | ViT op | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t266] | not measured |

### Unique to Encoder-decoder translation (NLLB, self-contained kernel set)

24 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, beam_topk, bias_bf16, embed_bf16, gather_batched, gemv_bf16, layernorm_oop_bf16, relu_bf16, scale_bf16, scatter_batched}` (13) | [gb10/common/nllb_encoder.cu:203][f110] | NLLB encoder/decoder op | b200 b300 gb10 hop | NLLB (1 ckpts) | [3 notes][t110] | not measured |
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, bias_bf16, embed_bf16, gather_batched, gemv_bf16, relu_bf16, scale_bf16, scatter_batched}` (11) | [metal/common/nllb_encoder.metal:281][f320] | NLLB encoder/decoder op | metal | NLLB (1 ckpts) | [1 note][t320] | not measured |

### Unique to LoRA adapters (BGMV shrink/expand)

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| lora_bgmv::`lora_bgmv_{expand_fold, shrink}` (2) | [gb10/common/lora_bgmv.cu:50][f62] | BGMV shrink/expand | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t62] | not measured |
| moe_lora_gather_bgmv::`moe_lora_gather_bgmv_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_gather_bgmv.cu:57][f76] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t76] | not measured |
| moe_lora_grouped_down::`moe_lora_grouped_down_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_grouped_down.cu:67][f77] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t77] | not measured |

### Unique to Weight load and repack (one-time, not per token)

15 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dequant_gguf_bf16::`dequant_{q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (4) | [gb10/common/dequant_gguf_bf16.cu:43][f24] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t24] | not measured |
| quantize_bf16_to_fp8_blockscaled::`quantize_bf16_to_fp8_blockscaled` | [gb10/common/quantize_bf16_to_fp8_blockscaled.cu:54][f151] | activation quantize | b200 b300 gb10 hop | LongCat (1 ckpts) | [1 note][t151] | not measured |
| quantize_nvfp4::`f32_to_bf16_trunc` | [gb10/common/quantize_bf16_to_nvfp4.cu:29][f152] | dtype conversion | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t152] | not measured |
| quantize_nvfp4::`quantize_bf16_to_nvfp4_mse` | [gb10/common/quantize_bf16_to_nvfp4.cu:265][f152] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t152] · [#34][pr34] | not measured |
| transpose_u8::`transpose_u8` | [gb10/common/transpose_u8.cu:15][f169] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | not measured |
| widen_block_scale_f32::`widen_block_scale_f32` | [gb10/common/widen_block_scale_f32.cu:21][f187] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t187] | not measured |
| hc_v41::`hc_v41_{collapse, collapse_wide, finish_collapse, mixes_dot, mixes_finish, post_wide}` (6) | [gb10/deepseek-v4-flash/nvfp4/hc_v41.cu:118][f193] | hyper-connection mix | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t193] | not measured |

## Compiled but not launched

No engine call site names these entry points: they are reached only from tests or `examples/` (microbenchmarks, the Metal Qwen3.5 driver), or not at all. They cost build time and are candidates for removal or for wiring up.

| Kernel (module::function) | File | Component · kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| prefill_fp8kv::`attn_prefill_fp8kv_64` | [gb10/common/attn_prefill_fp8kv.cu:94][f9] | Attention · prefill (flash) | b200 b300 gb10 hop strix | — | [1 note][t9] | not measured |
| attn_prefill_h128::`attn_prefill_h128_64` | [gb10/common/attn_prefill_h128.cu:509][f10] | Attention · prefill (flash) | b200 b300 gb10 hop strix | — | [1 note][t10] | not measured |
| attn_prefill_v47::`attn_prefill_v47` | [gb10/common/attn_prefill_v47.cu:30][f11] | Attention · prefill (flash) | b200 b300 gb10 hop strix | — | [1 note][t11] | not measured |
| causal_conv1d::`causal_conv1d_{fwd, update_f32}` (2) | [gb10/common/causal_conv1d.cu:30][f13] | Causal conv1d · causal conv1d | b200 b300 gb10 hop strix hip | — | [1 note][t13] | not measured |
| gemm::`fused_silu_mul` | [gb10/common/dense_gemm_bf16.cu:590][f14] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix | — | — | not measured |
| e2m1::`e2m1_quantize` | [gb10/common/e2m1_branchless.cu:48][f28] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | — | [1 note][t28] | not measured |
| embed_from_argmax::`batched_embed_f32`, `embed_from_argmax_f32` | [gb10/common/embed_from_argmax.cu:57][f29] | Embedding and LM head · embedding / LM head | b200 b300 gb10 hop strix hip | — | [1 note][t29] | not measured |
| fused_k_norm_rope_cache::`fused_k_norm_rope_cache_write_fp8` | [gb10/common/fused_k_norm_rope_cache.cu:252][f34] | KV cache · cache write | b200 b300 gb10 hop | — | [1 note][t34] | not measured |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_{h, h_dvsplit, h_ksplit_vblock2, h_ksplit_vblock4, h_ksplit_vblock8, h_tc}` (6) | [gb10/common/gated_delta_rule_fla.cu:532][f37] | GDN · delta-rule recurrence | b200 b300 gb10 hop | — | [10 notes][t37] | not measured |
| gated_delta_rule_persistent::`gated_delta_rule_prefill_persistent_{multihead, regtile}` (2) | [gb10/common/gated_delta_rule_persistent.cu:498][f38] | GDN · delta-rule recurrence | b200 b300 gb10 hop strix hip | — | [3 notes][t38] | not measured |
| glm5next_ffn::`glm5next_swiglu_clamp_f32out` | [gb10/common/glm5next_ffn.cu:50][f56] | MoE · expert activation | b200 b300 gb10 hop | — | [1 note][t56] | not measured |
| glm5next_mhc::`glm5next_hc_post_ref` | [gb10/common/glm5next_mhc.cu:497][f57] | Hyper-connections · hyper-connection mix | b200 b300 gb10 hop | — | [2 notes][t57] | not measured |
| kda_gate::`kda_gate_f32` | [gb10/common/kda_gate.cu:101][f59] | KDA · KDA op | b200 b300 gb10 hop | — | — | not measured |
| kda_layer_ops::`kda_o_norm_gated_f32` | [gb10/common/kda_layer_ops.cu:66][f60] | KDA · KDA op | b200 b300 gb10 hop | — | — | not measured |
| kda_recurrent::`kda_recurrent_decode_f32` | [gb10/common/kda_recurrent.cu:125][f61] | KDA · KDA op | b200 b300 gb10 hop | — | [1 note][t61] | not measured |
| moe_expert_gemv::`moe_weighted_sum` | [gb10/common/moe_expert_gemv.cu:164][f68] | MoE · dispatch / combine | b200 b300 gb10 hop strix hip | — | [2 notes][t68] | not measured |
| moe_expert_gemv_fused::`moe_expert_gemv_{gate_up, gate_up_2x, silu_down, silu_down_2x, silu_down_wide}` (5) | [gb10/common/moe_expert_gemv_fused.cu:53][f69] | MoE · expert GEMM/GEMV | b200 b300 gb10 hop strix hip | — | [1 note][t69] | not measured |
| moe::`moe_{count_experts, unpermute_reduce}` (2) | [gb10/common/moe_permute.cu:45][f79] | MoE · dispatch / combine | b200 b300 gb10 hop strix hip | — | [1 note][t79] | not measured |
| moe_w4a16::`moe_w4a16_grouped_{gemm, gemm_ptrtable_al_k64, gemm_ptrtable_al_m16_k128, gemm_ptrtable_al_m16_k32, gemm_ptrtable_al_m16_k64, gemm_ptrtable_al_m16_n128_k64, gemm_ptrtable_alkm_k128, gemm_ptrtable_alkm_k64, gemm_ptrtable_alkm_m16_k256, gemm_ptrtable_alkm_m16_k32, gemm_ptrtable_alkm_m16_k64, gemm_ptrtable_alkm_m16_n128_k128, gemm_ptrtable_alkm_m16_n128_k64, gemm_ptrtable_bt_m16_k256, gemm_ptrtable_bt_m16_k64, gemm_ptrtable_k128, gemm_ptrtable_km_k64, gemm_ptrtable_km_m16_k128, gemm_ptrtable_km_m16_k32, gemm_ptrtable_km_m16_k64, gemm_ptrtable_km_m16_n128_k128, gemm_ptrtable_km_m16_n128_k64, gemm_ptrtable_m16_k128, gemm_ptrtable_m16_k256, gemm_ptrtable_m16_k32, gemm_ptrtable_m16_n128_k128, gemm_ptrtable_m16_n128_k32, gemm_ptrtable_m16_n128_k64}` (28) | [gb10/common/moe_w4a16_grouped_gemm.cu:37][f106] | MoE · expert GEMM/GEMV | b200 gb10 hop | — | [4 notes][t106] | not measured |
| moe_w4a16::`moe_w4a16_grouped_stream_probe` | [gb10/common/moe_w4a16_grouped_gemm.cu:1006][f106] | Diagnostics and microtests · microtest / smoke | b200 gb10 hop | — | [5 notes][t106] | not measured |
| nllb_encoder::`nllb_{add_inplace, argmax_batched, attention, attn_kv, embed, layernorm, linear, relu_inplace, scale_inplace}` (9) | [gb10/common/nllb_encoder.cu:18][f110] | Encoder-decoder translation · NLLB encoder/decoder op | b200 b300 gb10 hop | — | [1 note][t110] | not measured |
| paged_decode::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/common/paged_decode_attn.cu:322][f111] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t111] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo3.cu:561][f130] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t130] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo3_128.cu:551][f131] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t131] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo4.cu:536][f134] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t134] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo4_128.cu:536][f135] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t135] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo4_512.cu:560][f136] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t136] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo8.cu:543][f141] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t141] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo8_128.cu:543][f142] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t142] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo8_512.cu:566][f143] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t143] | not measured |
| prefill_paged_indirect::`attn_prefill_paged_{indirect_64, turbo2_64, turbo3, turbo8}` (4) | [gb10/common/prefill_paged_compute.cuh:162][f145] | Attention · prefill (flash) | b200 b300 gb10 hop | — | [5 notes][t145] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo3k_turbo8v, turbo4k_turbo3v, turbo4k_turbo8v}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:99][f147] | Attention · prefill (flash) | b200 b300 gb10 hop | — | [2 notes][t147] | not measured |
| q2_0_gemv::`q2_0_{gemv, gemv_batchm}` (2) | [gb10/common/q2_0_gemv.cu:46][f148] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | — | [1 note][t148] | not measured |
| relu2::`bias_add_bf16_f32`, `relu_squared` | [gb10/common/relu_squared.cu:12][f153] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | — | not measured |
| relu2::`convert_f32_to_bf16` | [gb10/common/relu_squared.cu:80][f153] | Quantization and format conversion · dtype conversion | b200 b300 gb10 hop strix hip | — | — | not measured |
| reshape_and_cache::`bf16_absmax_per_head` | [gb10/common/reshape_and_cache.cu:395][f154] | KV cache · cache write | b200 b300 gb10 hop strix hip | — | [1 note][t154] | not measured |
| residual_add::`bf16_sigmoid_blend`, `bf16_sigmoid_blend_device`, `silu_mul_separate` | [gb10/common/residual_add.cu:41][f157] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | [1 note][t157] | not measured |
| residual_add::`bf16_to_f32` | [gb10/common/residual_add.cu:25][f157] | Quantization and format conversion · dtype conversion | b200 b300 gb10 hop strix hip | — | — | not measured |
| norm::`f32_residual_add` | [gb10/common/rms_norm.cu:987][f158] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | [2 notes][t158] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/common/rms_norm.cu:199][f158] | Normalization · normalization | b200 b300 gb10 hop strix hip | — | [2 notes][t158] | not measured |
| ssm_ba_gates_tiled::`dense_gemm_ba_gates_prefill_tiled` | [gb10/common/ssm_ba_gates_tiled.cu:23][f163] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 gb10 hop | — | — | not measured |
| vector_add::`vector_add` | [gb10/common/vector_add.cu:6][f170] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | — | not measured |
| w4a16::`w4a16_dequant` | [gb10/common/w4a16_gemm.cu:292][f172] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 | — | [1 note][t172] | not measured |
| w4a16_gemv::`w4a16_gemv_{batch4, batch5, batch6, batch7, batch8_pf, batch8_pf2, batch8_pf3, batch8_pf_free, batch8_rt4}` (9) | [gb10/common/w4a16_gemv.cu:720][f173] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | — | [3 notes][t173] | not measured |
| w8a16_gemm::`w8a16_dequant` | [gb10/common/w8a16_gemm.cu:224][f177] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix | — | [1 note][t177] | not measured |
| hc_v41::`hc_v41_mixes` | [gb10/deepseek-v4-flash/nvfp4/hc_v41.cu:46][f193] | Hyper-connections · hyper-connection mix | b200 gb10 hop | — | [2 notes][t193] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k, q3_k}` (2) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:210][f195] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | b200 gb10 hop | — | [4 notes][t195] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k_experts, q2_k_experts_w, q3_k_experts, q3_k_experts_w}` (4) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:231][f195] | MoE · expert GEMM/GEMV | b200 gb10 hop | — | [4 notes][t195] | not measured |
| mla_cache_assemble_fp8::`mla_cache_assemble_fp8_batched` | [gb10/deepseek-v4-flash/nvfp4/mla_cache_assemble_fp8.cu:32][f197] | MLA · MLA decode/prefill | b200 gb10 hop | — | [1 note][t197] | not measured |
| moe_v41::`moe_v41_router_gemv_f32out_staged` | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:171][f203] | MoE · routing / top-k | b200 gb10 hop | — | [1 note][t203] | not measured |
| paged_decode_attn_512::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu:305][f205] | Attention · paged decode | b200 gb10 hop | — | [1 note][t205] | not measured |
| paged_decode_fp8_mla::`paged_decode_attn_reduce_fp8` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu:507][f206] | MLA · MLA decode/prefill | b200 gb10 hop | — | [1 note][t206] | not measured |
| paged_decode_mla::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu:316][f207] | MLA · MLA decode/prefill | b200 gb10 hop | — | [1 note][t207] | not measured |
| embed_scale::`f32_scale_inplace` | [gb10/gemma-4-26b-a4b/nvfp4/embed_scale.cu:25][f212] | Embedding and LM head · embedding / LM head | gb10 | — | — | not measured |
| gelu::`gelu_tanh` | [gb10/gemma-4-26b-a4b/nvfp4/gelu.cu:21][f214] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t214] | not measured |
| paged_decode_attn_512::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu:305][f220] | Attention · paged decode | gb10 | — | [1 note][t220] | not measured |
| paged_decode_attn_fp8_512::`paged_decode_attn_reduce_fp8` | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu:487][f221] | Attention · paged decode | gb10 | — | [1 note][t221] | not measured |
| norm::`f32_residual_add` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:471][f222] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t222] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:282][f222] | Normalization · normalization | gb10 | — | [1 note][t222] | not measured |
| embed_scale::`f32_scale_inplace` | [gb10/gemma-4-31b/nvfp4/embed_scale.cu:28][f224] | Embedding and LM head · embedding / LM head | gb10 | — | [1 note][t224] | not measured |
| logit_softcap::`logit_softcap_fp32` | [gb10/gemma-4-31b/nvfp4/logit_softcap.cu:31][f225] | Embedding and LM head · embedding / LM head | gb10 | — | [1 note][t225] | not measured |
| norm::`f32_residual_add` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:487][f226] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t226] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:294][f226] | Normalization · normalization | gb10 | — | [1 note][t226] | not measured |
| fp4_mma_microtest::`fp4_microtest_{mma, pack}` (2) | [gb10/holo-3.1-0.8b/nvfp4/fp4_mma_microtest.cu:87][f230] | Diagnostics and microtests · microtest / smoke | gb10 | — | [1 note][t230] | not measured |
| norm::`f32_residual_add` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:494][f234] | Activations and elementwise · activation / gate / residual | b200 gb10 hop | — | [1 note][t234] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:684][f234] | Normalization · normalization | b200 gb10 hop | — | [1 note][t234] | not measured |
| paged_decode_attn_fp8_mla::`paged_decode_attn_reduce_fp8` | [gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu:507][f240] | MLA · MLA decode/prefill | gb10 | — | [1 note][t240] | not measured |
| paged_decode_mla::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu:316][f241] | MLA · MLA decode/prefill | gb10 | — | [1 note][t241] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu:91][f243] | MoE · expert GEMM/GEMV | gb10 | — | — | not measured |
| norm::`f32_residual_add` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:502][f245] | Activations and elementwise · activation / gate / residual | b200 gb10 hop | — | [1 note][t245] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:692][f245] | Normalization · normalization | b200 gb10 hop | — | [1 note][t245] | not measured |
| w4a16::`fp8_gemm_t_mfast` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:565][f246] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | gb10 | — | [2 notes][t246] | not measured |
| norm::`f32_residual_add` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:531][f249] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t249] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:644][f249] | Normalization · normalization | gb10 | — | [1 note][t249] | not measured |
| vision_encoder::`vision_gemm_bias_nn` | [gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu:45][f250] | Vision encoder · ViT op | gb10 | — | [1 note][t250] | not measured |
| q4k_mmq::`metrale_q8_1_quantize_ds4` | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:63][f258] | Quantization and format conversion · activation quantize | gb10 hop | — | [1 note][t258] | not measured |
| w4a16::`int8_gemm_8w`, `int8_gemm_8w3`, `int8_gemm_8w_ilp`, `int8_gemm_8w_ldm`, `int8_gemm_8w_ldmab`, `int8_gemm_8w_pipe`, `int8_gemm_faith`, `int8_gemm_faith10`, `int8_gemm_faith3`, `int8_gemm_faith4`, `int8_gemm_faith5`, `int8_gemm_faith6`, `int8_gemm_faith7`, `int8_gemm_faith8`, `int8_gemm_faith9`, `int8_gemm_mmq`, `int8_gemm_mmq2`, `int8_gemm_mmqf`, `int8_gemm_mmqf2`, `int8_gemm_mmqf3`, `int8_gemm_padA`, `int8_gemm_splitk`, `int8_gemm_t_m128`, `int8_gemm_t_m128_k64`, `int8_gemm_t_m64`, `int8_splitk_reduce`, `requant_a_bf16_int8_il` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:3043][f260] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | gb10 hop strix | — | [4 notes][t260] | not measured |
| w4a16::`w4a16_gemm_t_m64_bf16` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:2584][f260] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | — | [2 notes][t260] | not measured |
| vision_encoder::`vision_gemm_bias_nn` | [gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu:45][f266] | Vision encoder · ViT op | b200 gb10 hop strix hip | — | [1 note][t266] | not measured |
| gated_delta_rule_chunk_tc::`gated_delta_rule_chunk_delta_h_tcfuse` | [hopper/common/gated_delta_rule_chunk_tc.cu:408][f275] | GDN · delta-rule recurrence | hop | — | [3 notes][t275] | not measured |
| argmax_bf16::`argmax_bf16` | [metal/common/argmax_bf16.metal:22][f288] | Sampling · argmax / top-p | metal | — | [1 note][t288] | not measured |
| attention_full::`attention_full` | [metal/common/attention_full.metal:22][f295] | Attention · prefill (flash) | metal | — | [1 note][t295] | not measured |
| attention_prefill::`attention_prefill` | [metal/common/attention_prefill.metal:27][f296] | Attention · prefill (flash) | metal | — | [1 note][t296] | not measured |
| causal_conv1d_decode::`causal_conv1d_decode` | [metal/common/causal_conv1d_decode.metal:28][f298] | Causal conv1d · causal conv1d | metal | — | [1 note][t298] | not measured |
| conv3d_patch_embed::`conv3d_patch_embed` | [metal/common/conv3d_patch_embed.metal:24][f300] | Vision encoder · ViT op | metal | — | [1 note][t300] | not measured |
| embed_lookup::`embed_lookup` | [metal/common/embed_lookup.metal:17][f303] | Embedding and LM head · embedding / LM head | metal | — | [1 note][t303] | not measured |
| gdn_helpers::`bf16_mul`, `silu_apply` | [metal/common/gdn_helpers.metal:79][f305] | GDN · GDN helper | metal | — | — | not measured |
| layer_norm::`layer_norm` | [metal/common/layer_norm.metal:27][f313] | Normalization · normalization | metal | — | [1 note][t313] | not measured |
| lora_bgmv::`lora_bgmv_{expand_fold_stub, shrink_stub}` (2) | [metal/common/lora_bgmv.metal:12][f314] | Diagnostics and microtests · microtest / smoke | metal | — | [1 note][t314] | not measured |
| nllb_encoder::`nllb_{add_inplace, add_position_bf16, argmax_batched, argmax_bf16_rows, attention, attn_kv, attn_kv_batched_bf16, cache_write_bf16, embed, gemv_batched_bf16, gemv_bf16_no_bias, layernorm, linear, linear_bf16, linear_no_bias, linear_no_bias_bf16, relu_inplace, scale_inplace, topk_lse_bf16}` (19) | [metal/common/nllb_encoder.metal:13][f320] | Encoder-decoder translation · NLLB encoder/decoder op | metal | — | [2 notes][t320] | not measured |
| noop_smoke::`noop_smoke` | [metal/common/noop_smoke.metal:11][f321] | Diagnostics and microtests · microtest / smoke | metal | — | — | not measured |
| selective_scan_decode::`selective_scan_decode` | [metal/common/selective_scan_decode.metal:40][f325] | Mamba2 · SSD / selective scan | metal | — | [1 note][t325] | not measured |
| silu_gate::`silu_gate` | [metal/common/silu_gate.metal:20][f327] | Activations and elementwise · activation / gate / residual | metal | — | — | not measured |
| softmax_topp::`softmax_topp` | [metal/common/softmax_topp.metal:32][f328] | Sampling · argmax / top-p | metal | — | [1 note][t328] | not measured |
| prefill_fp8kv::`attn_prefill_fp8kv_64` | [strix-hip/common/attn_prefill_fp8kv.cu:63][f331] | Attention · prefill (flash) | hip | — | [1 note][t331] | not measured |
| attn_prefill_h128::`attn_prefill_h128_64` | [strix-hip/common/attn_prefill_h128.cu:207][f332] | Attention · prefill (flash) | hip | — | [1 note][t332] | not measured |
| attn_prefill_v47::`attn_prefill_v47` | [strix-hip/common/attn_prefill_v47.cu:46][f333] | Attention · prefill (flash) | hip | — | [1 note][t333] | not measured |
| gemm::`fused_silu_mul` | [strix-hip/common/dense_gemm_bf16.cu:322][f334] | Activations and elementwise · activation / gate / residual | hip | — | [1 note][t334] | not measured |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm_v2` | [strix-hip/common/moe_fp8_grouped_gemm.cu:394][f336] | MoE · expert GEMM/GEMV | hip | — | [3 notes][t336] | not measured |
| w8a16_gemm::`w8a16_dequant` | [strix-hip/common/w8a16_gemm.cu:296][f337] | Quantization and format conversion · dequant / repack / transpose | hip | — | [2 notes][t337] | not measured |

## Measurements

199 rows over 61 entry points; every row, with its shape, time, floor, source and notes, is in [`docs/kernel-perf/MEASUREMENTS.md`](docs/kernel-perf/MEASUREMENTS.md). Per regime (median over the regime's rows, unweighted):

| Hardware | Model | Regime | Rows | Entry points | Median % of floor |
|---|---|---|---|---|---|
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | decode C=1 (R=2, MTP k=1) | 23 | 19 | 55% |
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | decode C=16 (R=32, MTP k=1) | 15 | 10 | 73% |
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | prefill 32k (cold, 32772 tok) | 33 | 24 | 25% |
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | prefill 4k (cold, 4549 tok) | 35 | 27 | 33% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | decode C=1 (R=4, MTP k=3) | 19 | 10 | 89% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | decode C=16 (R=32, MTP k=1) | 20 | 13 | 68% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | prefill 32k (cold, 32772 tok) | 26 | 20 | 29% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | prefill 4k (cold, 4103 tok) | 28 | 21 | 36% |

[f1]: kernels/b200/kimi-k3/bf16/dense_f32io.cu
[f2]: kernels/b300/common/dsa_indexer.cu
[f3]: kernels/b300/common/moe_shared_expert_fused.cu
[f4]: kernels/b300/common/w8a16_gemv_batch4.cu
[f5]: kernels/gb10/common/argmax_bf16.cu
[f6]: kernels/gb10/common/argmax_feed.cu
[f7]: kernels/gb10/common/attn_prefill.cu
[f8]: kernels/gb10/common/attn_prefill_fa128.cu
[f9]: kernels/gb10/common/attn_prefill_fp8kv.cu
[f10]: kernels/gb10/common/attn_prefill_h128.cu
[f11]: kernels/gb10/common/attn_prefill_v47.cu
[f12]: kernels/gb10/common/bf16_add.cu
[f13]: kernels/gb10/common/causal_conv1d.cu
[f14]: kernels/gb10/common/dense_gemm_bf16.cu
[f15]: kernels/gb10/common/dense_gemm_splitk.cu
[f16]: kernels/gb10/common/dense_gemm_tc.cu
[f17]: kernels/gb10/common/dense_gemv_bf16.cu
[f18]: kernels/gb10/common/dense_gemv_bf16_batch2.cu
[f19]: kernels/gb10/common/dense_gemv_bf16_batchm.cu
[f20]: kernels/gb10/common/dense_gemv_bf16_tc.cu
[f21]: kernels/gb10/common/dense_gemv_fp8w.cu
[f22]: kernels/gb10/common/dense_gemv_fp8w_batch2.cu
[f23]: kernels/gb10/common/dequant_fp8_blockscaled_bf16.cu
[f24]: kernels/gb10/common/dequant_gguf_bf16.cu
[f25]: kernels/gb10/common/dequant_nvfp4_bf16.cu
[f26]: kernels/gb10/common/dflash2.cu
[f27]: kernels/gb10/common/dsa_indexer.cu
[f28]: kernels/gb10/common/e2m1_branchless.cu
[f29]: kernels/gb10/common/embed_from_argmax.cu
[f30]: kernels/gb10/common/fp8_gemm_blockscaled_pipe.cu
[f31]: kernels/gb10/common/fp8_gemm_t_blockscaled.cu
[f32]: kernels/gb10/common/fp8_gemv_rt.cu
[f33]: kernels/gb10/common/fp8_scale_transpose.cu
[f34]: kernels/gb10/common/fused_k_norm_rope_cache.cu
[f35]: kernels/gb10/common/gated_delta_rule.cu
[f36]: kernels/gb10/common/gated_delta_rule_carry.cu
[f37]: kernels/gb10/common/gated_delta_rule_fla.cu
[f38]: kernels/gb10/common/gated_delta_rule_persistent.cu
[f39]: kernels/gb10/common/gated_delta_rule_regresident.cu
[f40]: kernels/gb10/common/gated_delta_rule_wy.cu
[f41]: kernels/gb10/common/gated_delta_rule_wy2_resident.cu
[f42]: kernels/gb10/common/gated_delta_rule_wy2_resident_f16.cu
[f43]: kernels/gb10/common/gated_delta_rule_wy3.cu
[f44]: kernels/gb10/common/gated_delta_rule_wy3_f16.cu
[f45]: kernels/gb10/common/gated_delta_rule_wy3_resident.cu
[f46]: kernels/gb10/common/gated_delta_rule_wy3_resident_f16.cu
[f47]: kernels/gb10/common/gated_delta_rule_wy4.cu
[f48]: kernels/gb10/common/gated_delta_rule_wy4_f16.cu
[f49]: kernels/gb10/common/gated_delta_rule_wy4_woa.cu
[f50]: kernels/gb10/common/gated_delta_rule_wy64_prefill.cu
[f51]: kernels/gb10/common/gated_delta_rule_wy_f16.cu
[f52]: kernels/gb10/common/gated_delta_rule_wyn.cu
[f53]: kernels/gb10/common/gdn_chunk_fwd_o_mma8.cu
[f54]: kernels/gb10/common/gdn_verify_fused_conv_kn.cu
[f55]: kernels/gb10/common/gdn_verify_fused_k2.cu
[f56]: kernels/gb10/common/glm5next_ffn.cu
[f57]: kernels/gb10/common/glm5next_mhc.cu
[f58]: kernels/gb10/common/kda_chunk.cu
[f59]: kernels/gb10/common/kda_gate.cu
[f60]: kernels/gb10/common/kda_layer_ops.cu
[f61]: kernels/gb10/common/kda_recurrent.cu
[f62]: kernels/gb10/common/lora_bgmv.cu
[f63]: kernels/gb10/common/mamba2_ssd_chunk.cu
[f64]: kernels/gb10/common/mamba2_ssm_decode.cu
[f65]: kernels/gb10/common/metadata_fill.cu
[f66]: kernels/gb10/common/moe_bf16_grouped_gemm.cu
[f67]: kernels/gb10/common/moe_decode_atomic_c4.cu
[f68]: kernels/gb10/common/moe_expert_gemv.cu
[f69]: kernels/gb10/common/moe_expert_gemv_fused.cu
[f70]: kernels/gb10/common/moe_expert_relu2_down_shared.cu
[f71]: kernels/gb10/common/moe_fp8_grouped_blend.cu
[f72]: kernels/gb10/common/moe_fp8_grouped_gemm.cu
[f73]: kernels/gb10/common/moe_fp8_grouped_sort.cu
[f74]: kernels/gb10/common/moe_gate_topk.cu
[f75]: kernels/gb10/common/moe_hash_route.cu
[f76]: kernels/gb10/common/moe_lora_gather_bgmv.cu
[f77]: kernels/gb10/common/moe_lora_grouped_down.cu
[f78]: kernels/gb10/common/moe_nvfp4_grouped.cu
[f79]: kernels/gb10/common/moe_permute.cu
[f80]: kernels/gb10/common/moe_prefill.cu
[f81]: kernels/gb10/common/moe_router_gemm.cu
[f82]: kernels/gb10/common/moe_router_gemm_prefill.cu
[f83]: kernels/gb10/common/moe_shared_expert_fused.cu
[f84]: kernels/gb10/common/moe_shared_expert_fused_batch2.cu
[f85]: kernels/gb10/common/moe_shared_expert_fused_batch2_t.cu
[f86]: kernels/gb10/common/moe_shared_expert_fused_batch3.cu
[f87]: kernels/gb10/common/moe_shared_expert_fused_batch3_t.cu
[f88]: kernels/gb10/common/moe_shared_expert_fused_bf16.cu
[f89]: kernels/gb10/common/moe_shared_expert_fused_bf16_batch2.cu
[f90]: kernels/gb10/common/moe_shared_expert_fused_fp8.cu
[f91]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch2.cu
[f92]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch2_t.cu
[f93]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch3.cu
[f94]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch3_t.cu
[f95]: kernels/gb10/common/moe_shared_expert_fused_fp8_grouped.cu
[f96]: kernels/gb10/common/moe_shared_expert_fused_fp8_t.cu
[f97]: kernels/gb10/common/moe_shared_expert_fused_t.cu
[f98]: kernels/gb10/common/moe_silu_mul.cu
[f99]: kernels/gb10/common/moe_sorted_prefill.cu
[f100]: kernels/gb10/common/moe_topk.cu
[f101]: kernels/gb10/common/moe_topk_sigmoid.cu
[f102]: kernels/gb10/common/moe_topk_softmax_bias.cu
[f103]: kernels/gb10/common/moe_topk_sqrtsoftplus.cu
[f104]: kernels/gb10/common/moe_transpose_batched.cu
[f105]: kernels/gb10/common/moe_unpermute_blend.cu
[f106]: kernels/gb10/common/moe_w4a16_grouped_gemm.cu
[f107]: kernels/gb10/common/moe_w8a8_grouped_gemm.cu
[f108]: kernels/gb10/common/moe_w8a8_grouped_gemm_e4m3.cu
[f109]: kernels/gb10/common/nemotron_moe_prefill.cu
[f110]: kernels/gb10/common/nllb_encoder.cu
[f111]: kernels/gb10/common/paged_decode_attn.cu
[f112]: kernels/gb10/common/paged_decode_attn_bf16_gqa.cu
[f113]: kernels/gb10/common/paged_decode_attn_bf16k_turbo2v.cu
[f114]: kernels/gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu
[f115]: kernels/gb10/common/paged_decode_attn_bf16k_turbo3v.cu
[f116]: kernels/gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu
[f117]: kernels/gb10/common/paged_decode_attn_bf16k_turbo4v.cu
[f118]: kernels/gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu
[f119]: kernels/gb10/common/paged_decode_attn_fp8.cu
[f120]: kernels/gb10/common/paged_decode_attn_fp8_gqa.cu
[f121]: kernels/gb10/common/paged_decode_attn_fp8k_turbo2v.cu
[f122]: kernels/gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu
[f123]: kernels/gb10/common/paged_decode_attn_fp8k_turbo3v.cu
[f124]: kernels/gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu
[f125]: kernels/gb10/common/paged_decode_attn_fp8k_turbo4v.cu
[f126]: kernels/gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu
[f127]: kernels/gb10/common/paged_decode_attn_nvfp4.cu
[f128]: kernels/gb10/common/paged_decode_attn_turbo2.cu
[f129]: kernels/gb10/common/paged_decode_attn_turbo2_128.cu
[f130]: kernels/gb10/common/paged_decode_attn_turbo3.cu
[f131]: kernels/gb10/common/paged_decode_attn_turbo3_128.cu
[f132]: kernels/gb10/common/paged_decode_attn_turbo3k_turbo8v.cu
[f133]: kernels/gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu
[f134]: kernels/gb10/common/paged_decode_attn_turbo4.cu
[f135]: kernels/gb10/common/paged_decode_attn_turbo4_128.cu
[f136]: kernels/gb10/common/paged_decode_attn_turbo4_512.cu
[f137]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo3v.cu
[f138]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu
[f139]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo8v.cu
[f140]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu
[f141]: kernels/gb10/common/paged_decode_attn_turbo8.cu
[f142]: kernels/gb10/common/paged_decode_attn_turbo8_128.cu
[f143]: kernels/gb10/common/paged_decode_attn_turbo8_512.cu
[f144]: kernels/gb10/common/per_token_group_quant_fp8.cu
[f145]: kernels/gb10/common/prefill_paged_compute.cuh
[f146]: kernels/gb10/common/prefill_paged_compute_512.cuh
[f147]: kernels/gb10/common/prefill_paged_compute_asym.cuh
[f148]: kernels/gb10/common/q2_0_gemv.cu
[f149]: kernels/gb10/common/q2_0_gemv_vec.cu
[f150]: kernels/gb10/common/quant_rowwise_fp8.cu
[f151]: kernels/gb10/common/quantize_bf16_to_fp8_blockscaled.cu
[f152]: kernels/gb10/common/quantize_bf16_to_nvfp4.cu
[f153]: kernels/gb10/common/relu_squared.cu
[f154]: kernels/gb10/common/reshape_and_cache.cu
[f155]: kernels/gb10/common/reshape_and_cache_fused_k_fp8.cu
[f156]: kernels/gb10/common/reshape_and_cache_turbo.cu
[f157]: kernels/gb10/common/residual_add.cu
[f158]: kernels/gb10/common/rms_norm.cu
[f159]: kernels/gb10/common/rms_norm_vanilla.cu
[f160]: kernels/gb10/common/rope.cu
[f161]: kernels/gb10/common/rope_mrope_interleaved.cu
[f162]: kernels/gb10/common/ssm_ba_gates_hopper.cu
[f163]: kernels/gb10/common/ssm_ba_gates_tiled.cu
[f164]: kernels/gb10/common/ssm_h_dtype.cu
[f165]: kernels/gb10/common/ssm_preprocess.cu
[f166]: kernels/gb10/common/ssm_state_norm.cu
[f167]: kernels/gb10/common/token_overlay.cu
[f168]: kernels/gb10/common/tq_plus_innerq_apply.cu
[f169]: kernels/gb10/common/transpose_u8.cu
[f170]: kernels/gb10/common/vector_add.cu
[f171]: kernels/gb10/common/w4a16_fp8_ldmab.cu
[f172]: kernels/gb10/common/w4a16_gemm.cu
[f173]: kernels/gb10/common/w4a16_gemv.cu
[f174]: kernels/gb10/common/w4a16_gemv_fused.cu
[f175]: kernels/gb10/common/w4a16_gemv_tc.cu
[f176]: kernels/gb10/common/w4a4_gemv_mx.cu
[f177]: kernels/gb10/common/w8a16_gemm.cu
[f178]: kernels/gb10/common/w8a16_gemm_pipe128.cu
[f179]: kernels/gb10/common/w8a16_gemm_pipelined.cu
[f180]: kernels/gb10/common/w8a16_gemm_pipelined_m32.cu
[f181]: kernels/gb10/common/w8a16_gemm_t.cu
[f182]: kernels/gb10/common/w8a16_gemm_t_m128.cu
[f183]: kernels/gb10/common/w8a16_gemv.cu
[f184]: kernels/gb10/common/w8a16_gemv_batch4.cu
[f185]: kernels/gb10/common/w8a16_gemv_fused.cu
[f186]: kernels/gb10/common/wht_bf16.cu
[f187]: kernels/gb10/common/widen_block_scale_f32.cu
[f188]: kernels/gb10/deepseek-v4-flash/nvfp4/attn_prefill_512.cu
[f189]: kernels/gb10/deepseek-v4-flash/nvfp4/attn_v41.cu
[f190]: kernels/gb10/deepseek-v4-flash/nvfp4/csa_compress.cu
[f191]: kernels/gb10/deepseek-v4-flash/nvfp4/engram_v41.cu
[f192]: kernels/gb10/deepseek-v4-flash/nvfp4/grouped_gemm_mla.cu
[f193]: kernels/gb10/deepseek-v4-flash/nvfp4/hc_v41.cu
[f194]: kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu
[f195]: kernels/gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu
[f196]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu
[f197]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_cache_assemble_fp8.cu
[f198]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu
[f199]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu
[f200]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu
[f201]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu
[f202]: kernels/gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu
[f203]: kernels/gb10/deepseek-v4-flash/nvfp4/moe_v41.cu
[f204]: kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu
[f205]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu
[f206]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu
[f207]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu
[f208]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu
[f209]: kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu
[f210]: kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu
[f211]: kernels/gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512.cu
[f212]: kernels/gb10/gemma-4-26b-a4b/nvfp4/embed_scale.cu
[f213]: kernels/gb10/gemma-4-26b-a4b/nvfp4/gated_delta_rule.cu
[f214]: kernels/gb10/gemma-4-26b-a4b/nvfp4/gelu.cu
[f215]: kernels/gb10/gemma-4-26b-a4b/nvfp4/logit_softcap.cu
[f216]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused.cu
[f217]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu
[f218]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu
[f219]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_w4a16_grouped_gemm.cu
[f220]: kernels/gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu
[f221]: kernels/gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu
[f222]: kernels/gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu
[f223]: kernels/gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu
[f224]: kernels/gb10/gemma-4-31b/nvfp4/embed_scale.cu
[f225]: kernels/gb10/gemma-4-31b/nvfp4/logit_softcap.cu
[f226]: kernels/gb10/gemma-4-31b/nvfp4/rms_norm.cu
[f227]: kernels/gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu
[f228]: kernels/gb10/glm-5.3-flash/nvfp4/glm5next_mla_latent_write.cu
[f229]: kernels/gb10/glm-5.3-flash/nvfp4/glm_vit.cu
[f230]: kernels/gb10/holo-3.1-0.8b/nvfp4/fp4_mma_microtest.cu
[f231]: kernels/gb10/kimi-k3/bf16/kda_decode.cu
[f232]: kernels/gb10/kimi-k3/bf16/mla_decode.cu
[f233]: kernels/gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu
[f234]: kernels/gb10/minimax-m2-229b/nvfp4/rms_norm.cu
[f235]: kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu
[f236]: kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v3.cu
[f237]: kernels/gb10/mistral-small-4/nvfp4/mla_absorbed.cu
[f238]: kernels/gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu
[f239]: kernels/gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu
[f240]: kernels/gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu
[f241]: kernels/gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu
[f242]: kernels/gb10/mistral-small-4/nvfp4/rope.cu
[f243]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu
[f244]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu
[f245]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu
[f246]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu
[f247]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a4_gemm.cu
[f248]: kernels/gb10/qwen3-next-80b-a3b/nvfp4/gated_delta_rule.cu
[f249]: kernels/gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu
[f250]: kernels/gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu
[f251]: kernels/gb10/qwen3.5-122b-a10b/nvfp4/gated_delta_rule.cu
[f252]: kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu
[f253]: kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule_snap.cu
[f254]: kernels/gb10/qwen3.6-27b/nvfp4/gdn_verify_fused_conv_kn_f32.cu
[f255]: kernels/gb10/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu
[f256]: kernels/gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu
[f257]: kernels/gb10/qwen3.6-27b/nvfp4/q2_0_mmq.cu
[f258]: kernels/gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu
[f259]: kernels/gb10/qwen3.6-27b/nvfp4/q4k_quantize.cu
[f260]: kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu
[f261]: kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm_v2.cu
[f262]: kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu
[f263]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu
[f264]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule_wy17.cu
[f265]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu
[f266]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu
[f267]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu
[f268]: kernels/gb10/qwen3.8-flash-next/nvfp4/gated_norm_sigmoid.cu
[f269]: kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu
[f270]: kernels/gb10/qwen3.8-flash-next/nvfp4/ple.cu
[f271]: kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu
[f272]: kernels/gb10/step3p7-flash/nvfp4/moe_silu_mul.cu
[f273]: kernels/hopper/common/dense_gemm_m16_bf16.cu
[f274]: kernels/hopper/common/fp8_act_quant_hopper.cu
[f275]: kernels/hopper/common/gated_delta_rule_chunk_tc.cu
[f276]: kernels/hopper/common/gdn_fwd_o_hopper.cu
[f277]: kernels/hopper/common/gdn_recompute_wu_hopper.cu
[f278]: kernels/hopper/common/moe_bucket_builder.cu
[f279]: kernels/hopper/common/moe_w8a8_m16.cu
[f280]: kernels/hopper/common/paged_decode_bf16_splitk_hopper.cu
[f281]: kernels/hopper/common/paged_decode_fp8_splitk_hopper.cu
[f282]: kernels/hopper/common/silu_mul_strided.cu
[f283]: kernels/hopper/common/w8a16_gemm_m16.cu
[f284]: kernels/hopper/common/w8a16_gemv.cu
[f285]: kernels/hopper/common/w8a16_gemv_fused.cu
[f286]: kernels/hopper/common/w8a16_gemv_ncol.cu
[f287]: kernels/metal/common/add_rms_norm.metal
[f288]: kernels/metal/common/argmax_bf16.metal
[f289]: kernels/metal/common/attention_decode.metal
[f290]: kernels/metal/common/attention_decode_bf16k_turbov.metal
[f291]: kernels/metal/common/attention_decode_turbo2.metal
[f292]: kernels/metal/common/attention_decode_turbo3.metal
[f293]: kernels/metal/common/attention_decode_turbo4.metal
[f294]: kernels/metal/common/attention_decode_turbo8.metal
[f295]: kernels/metal/common/attention_full.metal
[f296]: kernels/metal/common/attention_prefill.metal
[f297]: kernels/metal/common/bf16_add.metal
[f298]: kernels/metal/common/causal_conv1d_decode.metal
[f299]: kernels/metal/common/causal_conv1d_update_l2norm.metal
[f300]: kernels/metal/common/conv3d_patch_embed.metal
[f301]: kernels/metal/common/dense_gemm_bf16.metal
[f302]: kernels/metal/common/dense_gemv_bf16.metal
[f303]: kernels/metal/common/embed_lookup.metal
[f304]: kernels/metal/common/gated_delta_rule_decode.metal
[f305]: kernels/metal/common/gdn_helpers.metal
[f306]: kernels/metal/common/gelu.metal
[f307]: kernels/metal/common/kv_cache_append.metal
[f308]: kernels/metal/common/kv_cache_append_bf16k_turbov.metal
[f309]: kernels/metal/common/kv_cache_append_turbo2.metal
[f310]: kernels/metal/common/kv_cache_append_turbo3.metal
[f311]: kernels/metal/common/kv_cache_append_turbo4.metal
[f312]: kernels/metal/common/kv_cache_append_turbo8.metal
[f313]: kernels/metal/common/layer_norm.metal
[f314]: kernels/metal/common/lora_bgmv.metal
[f315]: kernels/metal/common/mlx_int8_dequant.metal
[f316]: kernels/metal/common/mlx_int8_gemm.metal
[f317]: kernels/metal/common/mlx_int8_gemv.metal
[f318]: kernels/metal/common/mlx_int8_gemv_gate_up.metal
[f319]: kernels/metal/common/mlx_int8_gemv_silu_gate.metal
[f320]: kernels/metal/common/nllb_encoder.metal
[f321]: kernels/metal/common/noop_smoke.metal
[f322]: kernels/metal/common/qwen35_qkv_split.metal
[f323]: kernels/metal/common/rms_norm.metal
[f324]: kernels/metal/common/rope_apply.metal
[f325]: kernels/metal/common/selective_scan_decode.metal
[f326]: kernels/metal/common/sigmoid_gate.metal
[f327]: kernels/metal/common/silu_gate.metal
[f328]: kernels/metal/common/softmax_topp.metal
[f329]: kernels/metal/common/wht_bf16.metal
[f330]: kernels/strix-hip/common/attn_prefill.cu
[f331]: kernels/strix-hip/common/attn_prefill_fp8kv.cu
[f332]: kernels/strix-hip/common/attn_prefill_h128.cu
[f333]: kernels/strix-hip/common/attn_prefill_v47.cu
[f334]: kernels/strix-hip/common/dense_gemm_bf16.cu
[f335]: kernels/strix-hip/common/dense_gemm_tc.cu
[f336]: kernels/strix-hip/common/moe_fp8_grouped_gemm.cu
[f337]: kernels/strix-hip/common/w8a16_gemm.cu
[f338]: kernels/strix-hip/common/w8a16_gemm_t.cu
[f339]: kernels/strix-hip/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu
[f340]: kernels/strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu
[f341]: kernels/strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu
[m5.argmax_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-argmax-bf16-cu-argmax-bf16
[m157.bf16_concat]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-residual-add-cu-bf16-concat
[m158.l2_norm_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-l2-norm-bf16
[m267.w4a16_gemm_t]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-35b-a3b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t
[m173.w4a16_gemv_sw]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-gemv-cu-w4a16-gemv-sw
[m36.gdn_carry_conv]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-carry-cu-gdn-carry-conv
[m7.attn_prefill_64]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-attn-prefill-cu-attn-prefill-64
[m14.dense_gemm_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemm-bf16-cu-dense-gemm-bf16
[m17.dense_gemv_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemv-bf16-cu-dense-gemv-bf16
[m175.w4a16_gemv_tc8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-gemv-tc-cu-w4a16-gemv-tc8
[m165.deinterleave_qg]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-ssm-preprocess-cu-deinterleave-qg
[m175.w4a16_gemv_tc16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-gemv-tc-cu-w4a16-gemv-tc16
[m260.fp8_gemm_t_m128]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-fp8-gemm-t-m128
[m260.w4a16_gemm_t_p3]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t-p3
[m79.moe_batched_blend]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-permute-cu-moe-batched-blend
[m111.paged_decode_attn]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-paged-decode-attn-cu-paged-decode-attn
[m157.bf16_residual_add]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-residual-add-cu-bf16-residual-add
[m158.rms_norm_residual]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-rms-norm-residual
[m182.w8a16_gemm_t_m128]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w8a16-gemm-t-m128-cu-w8a16-gemm-t-m128
[m260.w4a16_gemm_t_m128]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t-m128
[m36.gdn_carry_wy2_lazy]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-carry-cu-gdn-carry-wy2-lazy
[m98.silu_mul_quant_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-silu-mul-cu-silu-mul-quant-fp8
[m171.fp8_fp8_gemm_ldmab]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-fp8-ldmab-cu-fp8-fp8-gemm-ldmab
[m20.dense_gemv_bf16_tc16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemv-bf16-tc-cu-dense-gemv-bf16-tc16
[m40.gated_delta_rule_wy2]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-wy-cu-gated-delta-rule-wy2
[m47.gated_delta_rule_wy4]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-wy4-cu-gated-delta-rule-wy4
[m73.moe_fp8_grouped_sort]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-fp8-grouped-sort-cu-moe-fp8-grouped-sort
[m179.w8a16_gemm_pipelined]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w8a16-gemm-pipelined-cu-w8a16-gemm-pipelined
[m100.moe_topk_softmax_rows]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-topk-cu-moe-topk-softmax-rows
[m119.paged_decode_attn_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-paged-decode-attn-fp8-cu-paged-decode-attn-fp8
[m14.dense_gemm_bf16_router]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemm-bf16-cu-dense-gemm-bf16-router
[m145.attn_prefill_paged_64]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-prefill-paged-compute-cuh-attn-prefill-paged-64
[m158.residual_add_rms_norm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-residual-add-rms-norm
[m19.dense_gemv_bf16_batchm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemv-bf16-batchm-cu-dense-gemv-bf16-batchm
[m31.fp8_gemm_t_blockscaled]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-fp8-gemm-t-blockscaled-cu-fp8-gemm-t-blockscaled
[m158.gated_rms_norm_prefill]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-gated-rms-norm-prefill
[m256.metrale_nvfp4_mmq32_nc]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-mmq32-nc
[m79.moe_build_tile_worklist]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-permute-cu-moe-build-tile-worklist
[m256.metrale_nvfp4_mmq128_nc]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-mmq128-nc
[m260.w4a16_gemm_t_k64_n64_p3]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t-k64-n64-p3
[m180.w8a16_gemm_pipelined_m32]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w8a16-gemm-pipelined-m32-cu-w8a16-gemm-pipelined-m32
[m256.metrale_nvfp4_scale_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-scale-bf16
[m107.moe_w8a8_grouped_gemm_pm4]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-w8a8-grouped-gemm-cu-moe-w8a8-grouped-gemm-pm4
[m144.per_token_group_quant_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-per-token-group-quant-fp8-cu-per-token-group-quant-fp8
[m13.causal_conv1d_update_l2norm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-causal-conv1d-cu-causal-conv1d-update-l2norm
[m165.deinterleave_qg_split_qnorm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-ssm-preprocess-cu-deinterleave-qg-split-qnorm
[m165.dense_gemm_ba_gates_prefill]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-ssm-preprocess-cu-dense-gemm-ba-gates-prefill
[m256.metrale_nvfp4_quantize_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-quantize-bf16
[m37.gated_delta_rule_chunk_fwd_o]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-fla-cu-gated-delta-rule-chunk-fwd-o
[m79.moe_unpermute_reduce_indexed]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-permute-cu-moe-unpermute-reduce-indexed
[m256.metrale_nvfp4_silu_mul_quant]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-silu-mul-quant
[m37.gated_delta_rule_recompute_wu]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-fla-cu-gated-delta-rule-recompute-wu
[m90.moe_expert_gate_up_shared_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-shared-expert-fused-fp8-cu-moe-expert-gate-up-shared-fp8
[m13.causal_conv1d_update_prefill_tp]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-causal-conv1d-cu-causal-conv1d-update-prefill-tp
[m161.rope_forward_mrope_interleaved]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rope-mrope-interleaved-cu-rope-forward-mrope-interleaved
[m95.moe_expert_down_act_fp8_grouped]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-shared-expert-fused-fp8-grouped-cu-moe-expert-down-act-fp8-grouped
[m54.gdn_verify_fused_conv_kn_batched]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gdn-verify-fused-conv-kn-cu-gdn-verify-fused-conv-kn-batched
[m42.gated_delta_rule_wy2_resident_f16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-wy2-resident-f16-cu-gated-delta-rule-wy2-resident-f16
[m71.moe_weighted_sum_blend_fp8_grouped]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-fp8-grouped-blend-cu-moe-weighted-sum-blend-fp8-grouped
[m95.moe_expert_gate_up_act_fp8_grouped]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-shared-expert-fused-fp8-grouped-cu-moe-expert-gate-up-act-fp8-grouped
[m37.gated_delta_rule_chunk_delta_h_vfused]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-fla-cu-gated-delta-rule-chunk-delta-h-vfused
[pr1]: https://github.com/Metrale/metrale-inference/pull/1
[pr4]: https://github.com/Metrale/metrale-inference/pull/4
[pr14]: https://github.com/Metrale/metrale-inference/pull/14
[pr18]: https://github.com/Metrale/metrale-inference/pull/18
[pr25]: https://github.com/Metrale/metrale-inference/pull/25
[pr34]: https://github.com/Metrale/metrale-inference/pull/34
[t1]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b200-kimi-k3-bf16-dense-f32io-cu
[t2]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b300-common-dsa-indexer-cu
[t3]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b300-common-moe-shared-expert-fused-cu
[t4]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b300-common-w8a16-gemv-batch4-cu
[t5]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-argmax-bf16-cu
[t6]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-argmax-feed-cu
[t7]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-cu
[t9]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-fp8kv-cu
[t10]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-h128-cu
[t11]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-v47-cu
[t12]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-bf16-add-cu
[t13]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-causal-conv1d-cu
[t14]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemm-bf16-cu
[t15]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemm-splitk-cu
[t16]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemm-tc-cu
[t17]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-cu
[t18]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-batch2-cu
[t19]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-batchm-cu
[t20]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-tc-cu
[t21]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-fp8w-cu
[t22]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-fp8w-batch2-cu
[t23]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dequant-fp8-blockscaled-bf16-cu
[t24]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dequant-gguf-bf16-cu
[t25]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dequant-nvfp4-bf16-cu
[t26]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dflash2-cu
[t27]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dsa-indexer-cu
[t28]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-e2m1-branchless-cu
[t29]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-embed-from-argmax-cu
[t31]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fp8-gemm-t-blockscaled-cu
[t32]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fp8-gemv-rt-cu
[t33]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fp8-scale-transpose-cu
[t34]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fused-k-norm-rope-cache-cu
[t35]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-cu
[t36]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-carry-cu
[t37]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-fla-cu
[t38]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-persistent-cu
[t39]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-regresident-cu
[t40]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy-cu
[t41]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy2-resident-cu
[t42]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy2-resident-f16-cu
[t43]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-cu
[t44]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-f16-cu
[t45]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-resident-cu
[t46]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-resident-f16-cu
[t47]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy4-cu
[t48]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy4-f16-cu
[t49]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy4-woa-cu
[t50]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy64-prefill-cu
[t51]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy-f16-cu
[t52]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wyn-cu
[t54]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gdn-verify-fused-conv-kn-cu
[t55]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gdn-verify-fused-k2-cu
[t56]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-glm5next-ffn-cu
[t57]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-glm5next-mhc-cu
[t58]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-kda-chunk-cu
[t60]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-kda-layer-ops-cu
[t61]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-kda-recurrent-cu
[t62]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-lora-bgmv-cu
[t63]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-mamba2-ssd-chunk-cu
[t64]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-mamba2-ssm-decode-cu
[t65]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-metadata-fill-cu
[t66]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-bf16-grouped-gemm-cu
[t67]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-decode-atomic-c4-cu
[t68]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-expert-gemv-cu
[t69]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-expert-gemv-fused-cu
[t70]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-expert-relu2-down-shared-cu
[t71]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-blend-cu
[t72]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-gemm-cu
[t73]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-sort-cu
[t74]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-gate-topk-cu
[t75]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-hash-route-cu
[t76]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-lora-gather-bgmv-cu
[t77]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-lora-grouped-down-cu
[t78]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-nvfp4-grouped-cu
[t79]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-permute-cu
[t80]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-prefill-cu
[t81]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-router-gemm-cu
[t83]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-cu
[t84]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch2-cu
[t85]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch2-t-cu
[t86]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch3-cu
[t87]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch3-t-cu
[t88]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-bf16-cu
[t89]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-bf16-batch2-cu
[t90]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-cu
[t91]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch2-cu
[t92]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch2-t-cu
[t93]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch3-cu
[t94]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch3-t-cu
[t95]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-grouped-cu
[t96]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-t-cu
[t97]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-t-cu
[t98]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-silu-mul-cu
[t99]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-sorted-prefill-cu
[t100]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-cu
[t101]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-sigmoid-cu
[t102]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-softmax-bias-cu
[t103]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-sqrtsoftplus-cu
[t104]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-transpose-batched-cu
[t106]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-w4a16-grouped-gemm-cu
[t107]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-w8a8-grouped-gemm-cu
[t109]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-nemotron-moe-prefill-cu
[t110]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-nllb-encoder-cu
[t111]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-cu
[t112]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16-gqa-cu
[t113]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo2v-cu
[t114]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo2v-128-cu
[t115]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo3v-cu
[t116]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo3v-128-cu
[t117]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo4v-cu
[t118]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo4v-128-cu
[t119]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8-cu
[t120]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8-gqa-cu
[t121]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo2v-cu
[t122]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo2v-128-cu
[t123]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo3v-cu
[t124]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo3v-128-cu
[t125]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo4v-cu
[t126]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo4v-128-cu
[t127]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-nvfp4-cu
[t128]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo2-cu
[t129]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo2-128-cu
[t130]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3-cu
[t131]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3-128-cu
[t132]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3k-turbo8v-cu
[t133]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3k-turbo8v-128-cu
[t134]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4-cu
[t135]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4-128-cu
[t136]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4-512-cu
[t137]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo3v-cu
[t138]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo3v-128-cu
[t139]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo8v-cu
[t140]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo8v-128-cu
[t141]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo8-cu
[t142]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo8-128-cu
[t143]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo8-512-cu
[t144]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-per-token-group-quant-fp8-cu
[t145]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-prefill-paged-compute-cuh
[t146]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-prefill-paged-compute-512-cuh
[t147]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-prefill-paged-compute-asym-cuh
[t148]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-q2-0-gemv-cu
[t149]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-q2-0-gemv-vec-cu
[t150]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-quant-rowwise-fp8-cu
[t151]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-quantize-bf16-to-fp8-blockscaled-cu
[t152]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-quantize-bf16-to-nvfp4-cu
[t153]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-relu-squared-cu
[t154]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-reshape-and-cache-cu
[t155]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-reshape-and-cache-fused-k-fp8-cu
[t156]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-reshape-and-cache-turbo-cu
[t157]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-residual-add-cu
[t158]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rms-norm-cu
[t159]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rms-norm-vanilla-cu
[t160]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rope-cu
[t161]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rope-mrope-interleaved-cu
[t162]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-ba-gates-hopper-cu
[t164]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-h-dtype-cu
[t165]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-preprocess-cu
[t166]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-state-norm-cu
[t167]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-token-overlay-cu
[t168]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-tq-plus-innerq-apply-cu
[t171]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-fp8-ldmab-cu
[t172]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemm-cu
[t173]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemv-cu
[t174]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemv-fused-cu
[t175]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemv-tc-cu
[t176]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a4-gemv-mx-cu
[t177]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-cu
[t179]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-pipelined-cu
[t180]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-pipelined-m32-cu
[t181]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-t-cu
[t182]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-t-m128-cu
[t183]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemv-cu
[t184]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemv-batch4-cu
[t185]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemv-fused-cu
[t186]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-wht-bf16-cu
[t187]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-widen-block-scale-f32-cu
[t188]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-attn-prefill-512-cu
[t189]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-attn-v41-cu
[t190]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-csa-compress-cu
[t191]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-engram-v41-cu
[t192]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-grouped-gemm-mla-cu
[t193]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-hc-v41-cu
[t194]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-hyper-connection-cu
[t195]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-kquant-moe-cu
[t196]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-absorbed-cu
[t197]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-cache-assemble-fp8-cu
[t198]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-fused-prefill-cu
[t199]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-paged-decode-cu
[t200]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-paged-decode-fp8-cu
[t201]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-prefill-attn-cu
[t202]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-moe-silu-mul-cu
[t203]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-moe-v41-cu
[t204]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-moe-w4a16-grouped-gemm-cu
[t205]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-512-cu
[t206]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-fp8-mla-cu
[t207]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-mla-cu
[t208]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-nvfp4-cu
[t209]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-prefill-attn-compressed-cu
[t210]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-w4a16-gemm-cu
[t211]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-attn-prefill-512-cu
[t213]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-gated-delta-rule-cu
[t214]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-gelu-cu
[t215]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-logit-softcap-cu
[t216]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-shared-expert-fused-cu
[t217]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-shared-expert-fused-batch2-cu
[t218]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-shared-expert-fused-batch3-cu
[t219]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-w4a16-grouped-gemm-cu
[t220]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-paged-decode-attn-512-cu
[t221]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-paged-decode-attn-fp8-512-cu
[t222]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-rms-norm-cu
[t223]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-attn-prefill-512-cu
[t224]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-embed-scale-cu
[t225]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-logit-softcap-cu
[t226]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-rms-norm-cu
[t227]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-glm-5-3-flash-nvfp4-glm5next-dsa-mla-decode-cu
[t228]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-glm-5-3-flash-nvfp4-glm5next-mla-latent-write-cu
[t229]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-glm-5-3-flash-nvfp4-glm-vit-cu
[t230]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-holo-3-1-0-8b-nvfp4-fp4-mma-microtest-cu
[t231]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-kimi-k3-bf16-kda-decode-cu
[t232]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-kimi-k3-bf16-mla-decode-cu
[t233]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-moe-w4a16-grouped-gemm-cu
[t234]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-rms-norm-cu
[t235]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-w4a16-gemm-v2-cu
[t236]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-w4a16-gemm-v3-cu
[t237]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-mla-absorbed-cu
[t238]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-mla-fused-prefill-cu
[t239]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-mla-prefill-attn-cu
[t240]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-paged-decode-attn-fp8-mla-cu
[t241]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-paged-decode-attn-mla-cu
[t242]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-rope-cu
[t243]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-moe-w4a16-grouped-gemm-cu
[t244]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-moe-w4a4-grouped-cu
[t245]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-rms-norm-cu
[t246]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-w4a16-gemm-cu
[t247]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-w4a4-gemm-cu
[t248]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-next-80b-a3b-nvfp4-gated-delta-rule-cu
[t249]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-vl-30b-a3b-nvfp4-rms-norm-cu
[t250]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-vl-30b-a3b-nvfp4-vision-encoder-cu
[t251]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-5-122b-a10b-nvfp4-gated-delta-rule-cu
[t252]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-gated-delta-rule-cu
[t253]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-gated-delta-rule-snap-cu
[t254]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-gdn-verify-fused-conv-kn-f32-cu
[t255]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-moe-w4a16-grouped-gemm-cu
[t256]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu
[t257]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-q2-0-mmq-cu
[t258]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-q4k-mmq-cu
[t259]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-q4k-quantize-cu
[t260]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu
[t261]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-v2-cu
[t262]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-w4a4-gemm-cu
[t263]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-gated-delta-rule-cu
[t264]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-gated-delta-rule-wy17-cu
[t265]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-moe-w4a16-grouped-gemm-cu
[t266]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-vision-encoder-cu
[t267]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-w4a16-gemm-cu
[t268]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-gated-norm-sigmoid-cu
[t269]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-hyper-connection-cu
[t270]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-ple-cu
[t271]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-qsa-indexer-cu
[t272]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-step3p7-flash-nvfp4-moe-silu-mul-cu
[t273]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-dense-gemm-m16-bf16-cu
[t274]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-fp8-act-quant-hopper-cu
[t275]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-gated-delta-rule-chunk-tc-cu
[t276]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-gdn-fwd-o-hopper-cu
[t277]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-gdn-recompute-wu-hopper-cu
[t278]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-moe-bucket-builder-cu
[t279]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-moe-w8a8-m16-cu
[t280]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-paged-decode-bf16-splitk-hopper-cu
[t281]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-paged-decode-fp8-splitk-hopper-cu
[t282]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-silu-mul-strided-cu
[t283]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemm-m16-cu
[t284]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemv-cu
[t285]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemv-fused-cu
[t286]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemv-ncol-cu
[t287]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-add-rms-norm-metal
[t288]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-argmax-bf16-metal
[t289]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-metal
[t290]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-bf16k-turbov-metal
[t291]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo2-metal
[t292]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo3-metal
[t293]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo4-metal
[t294]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo8-metal
[t295]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-full-metal
[t296]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-prefill-metal
[t298]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-causal-conv1d-decode-metal
[t299]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-causal-conv1d-update-l2norm-metal
[t300]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-conv3d-patch-embed-metal
[t301]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-dense-gemm-bf16-metal
[t302]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-dense-gemv-bf16-metal
[t303]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-embed-lookup-metal
[t304]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-gated-delta-rule-decode-metal
[t308]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-bf16k-turbov-metal
[t309]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo2-metal
[t310]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo3-metal
[t311]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo4-metal
[t312]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo8-metal
[t313]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-layer-norm-metal
[t314]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-lora-bgmv-metal
[t316]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemm-metal
[t317]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemv-metal
[t318]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemv-gate-up-metal
[t319]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemv-silu-gate-metal
[t320]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-nllb-encoder-metal
[t323]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-rms-norm-metal
[t325]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-selective-scan-decode-metal
[t328]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-softmax-topp-metal
[t329]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-wht-bf16-metal
[t330]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-cu
[t331]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-fp8kv-cu
[t332]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-h128-cu
[t333]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-v47-cu
[t334]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-dense-gemm-bf16-cu
[t335]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-dense-gemm-tc-cu
[t336]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-moe-fp8-grouped-gemm-cu
[t337]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-w8a16-gemm-cu
[t338]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-w8a16-gemm-t-cu
[t339]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-qwen3-6-27b-nvfp4-moe-w4a16-grouped-gemm-cu
[t340]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-qwen3-6-27b-nvfp4-w4a16-gemm-cu
[t341]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-qwen3-6-35b-a3b-nvfp4-w4a16-gemm-cu

<!-- kernel_perf.py: END GENERATED -->
