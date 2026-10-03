# FlashKDA (vendored, unmodified)

MoonshotAI's FlashKDA forward kernels (Kimi Delta Attention chunked prefill), MIT.

- Upstream: https://github.com/MoonshotAI/FlashKDA
- Commit: `7afb9f454f160a6c4bbc0999beca0a8c40a38934` (2026-09-01, "replace fp16 Neumann inverse
  with 8x8 fp32 forward substitution + 16x16 bf16 merge")
- License: MIT, `Copyright (c) 2026 MoonshotAI`, shipped in place as [`LICENSE`](LICENSE).
- Dependency: CUTLASS (BSD-3-Clause), headers only, at upstream's submodule pin
  `5c149f52a436782210263fb2f19b354443a61c6a` (v4.3.2). It is not vendored: the GB10 Dockerfiles
  clone it at that commit and point `FLASHKDA_CUTLASS_HOME` at it.

## What is here

Only the CUDA sources the forward launcher needs, byte-identical to upstream at the commit
above (sha256 below). Upstream's PyTorch binding (`csrc/flash_kda.cpp`), Python package, tests
and benchmarks are not vendored.

| File | sha256 |
|---|---|
| `csrc/fwd.h` | `462afefb5c1bfe89428166cd81fc70954606049a41a4b67376025a2b2f91d7cc` |
| `csrc/smxx/fwd_launch.cu` | `dbaf9ba867fc8c799cf147b7e17bd2aba4a08b1b63c5f7df0cca1173a0df5562` |
| `csrc/smxx/fwd_kernel1.cuh` | `5ae34613a42e0c7056708bdd52096e65db240532f96f1d2b1353ecfc798fd265` |
| `csrc/smxx/fwd_kernel2.cuh` | `f8a634154ca26a18855adbd3db17d6208a874b16f173d0eb9814bfefff03b9b9` |
| `csrc/smxx/utils.cuh` | `97db8d9e0e55ba7ce3be60f2ba81ce3e90e03878b7d728c67bf661581d6d769b` |
| `LICENSE` | `05f1750624d6ab5f6dd59dea79e3156f4fec6b3065c94aeeaf265df677f9e6e8` |

## How Metrale uses it

`crates/gpu-runtime/build.rs` compiles `csrc/smxx/fwd_launch.cu` together with Metrale's own
C-ABI wrapper `crates/gpu-runtime/cuda/flashkda_kda_fwd.cu` into the static library
`metrale_flashkda` when `FLASHKDA_CUTLASS_HOME` is set at build time, with upstream's nvcc flags
(`setup.py`: `-O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda` and the
`-U__CUDA_NO_*` set) for the target's arch. Upstream lists sm_90a/100a/103a/120a; the source
compiles unmodified for sm_121a/sm_121f with CUDA 13.0 (checked 2026-10-03: no spills, TMA and
HMMA instructions present, the same SASS instruction counts as the sm_120a build).

The only caller is the opt-in GLM-5.3 KDA prefill (`METRALE_GLM_KDA_PREFILL_FLASHKDA=1`,
`crates/model-arch/src/glm5next_kda/prefill_flashkda.rs`).

## Updating

Copy the five files from a new upstream commit, update the commit, CUTLASS pin and hashes above
and in `THIRD_PARTY_NOTICES.md`, and rerun `flashkda_prefill_microtest` on a GB10.
