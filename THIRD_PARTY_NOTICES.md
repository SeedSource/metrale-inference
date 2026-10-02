# Third-Party Notices

This repository incorporates, or is derived in part from, the third-party
components listed below. Each keeps its own upstream license. Full license
texts are collected under [`LICENSES/`](LICENSES/) by SPDX identifier; where a
component ships its own license file in place, the entry points to that copy
instead.

Files listed here as third-party keep their original headers, and the
repository's own SPDX header is never stamped over them (see `paths-ignore` in
`.licenserc.yaml`).

---

## 1. cudarc — MIT OR Apache-2.0

Rust bindings to the CUDA driver and NVRTC APIs, vendored at upstream version
0.19.2. The root `Cargo.toml` `[patch.crates-io]` entry explains the one local
change: CUDA-driver symbols that a non-NVIDIA runtime lacks resolve to a stub
instead of panicking at init.

- **License**: MIT OR Apache-2.0, at the recipient's option.
- **License texts**: shipped in place at
  [`vendor/cudarc/LICENSE-MIT`](vendor/cudarc/LICENSE-MIT) and
  [`vendor/cudarc/LICENSE-APACHE`](vendor/cudarc/LICENSE-APACHE).
- **Copyright**: the upstream license files carry no copyright line; the
  crate's `Cargo.toml` names `Chelsea Lowman <clowman1993@gmail.com>` as author.
- **Upstream**: https://github.com/chelsea0x3b/cudarc
- **In-repo path**: `vendor/cudarc/`

---

## 2. llama.cpp / ggml — MIT

Vendored CUDA MMQ (matrix-multiply-quantized) kernel sources and their
supporting ggml/GGUF headers (`mmq.cuh`, `mma.cuh`, `vecdotq.cuh`,
`quantize.cuh`, `quantize_impl.cuh`, `common.cuh`, `gguf.h`, `ggml.h`,
`ggml-impl.h`, `ggml-common.h`, `ggml-cuda.h`, `ggml-backend.h`,
`ggml-alloc.h`, `vendors/cuda.h`), used by the dense Qwen3.6-27B Q4_K decode
path.

- **License**: MIT. Text at [`LICENSES/MIT.txt`](LICENSES/MIT.txt).
- **Copyright**: `Copyright (c) 2023-2026 The ggml authors` (from upstream's
  root `LICENSE`; the vendored files carry no per-file copyright line).
- **Upstream**: https://github.com/ggml-org/llama.cpp (`ggml/src/ggml-cuda/`)
- **In-repo path**: `kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/`
- **Also derived from llama.cpp**: the DRY repetition penalty in
  `crates/sampling/src/lib.rs` is a Rust port of llama.cpp PR #9702.

---

## 3. vLLM — Apache-2.0

Reference copies of vLLM's FP8/W8A8 quantization, MoE dispatch and Qwen3.5 /
Qwen3-Next / gated-delta-net model sources (`fp8.py`, `fp8_utils.py`,
`input_quant_fp8.py`, `moe_oracle_fp8.py`, `scaled_mm_*.py`, `w8a8_utils.py`,
`qwen3_5.py`, `qwen3_next.py`, `gdn_linear_attn.py`), used as a
drift-comparison oracle by an FP8 bench harness.

- **License**: Apache-2.0. Text at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- **Copyright**: every file carries its upstream header,
  `SPDX-License-Identifier: Apache-2.0` /
  `SPDX-FileCopyrightText: Copyright contributors to the vLLM project`.
- **Upstream**: https://github.com/vllm-project/vllm
- **In-repo path**: `bench/fp8_dgx2_drift/vllm_src/`

---

## 4. Hugging Face Transformers model references

### 4a. Qwen4-exp modeling files — Apache-2.0

Generated `transformers` modeling and configuration files
(`modeling_qwen4_exp.py`, `configuration_qwen4_exp.py`), used as a
correctness oracle by the Qwen4-exp bench harness.

- **License**: Apache-2.0. Text at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- **Copyright**: `Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
  All rights reserved.` (verbatim, both files).
- **Upstream**: https://github.com/huggingface/transformers
  (`src/transformers/models/qwen4_exp/`)
- **In-repo path**: `bench/qwen4_exp/ref/`

### 4b. LongCat-Flash n-gram modeling file — MIT

`modeling_longcat_ngram.py`, used as a correctness oracle by the n-gram
speculative-decoding bench harness. It subclasses `transformers`'
`LongcatFlashForCausalLM`, but the file itself is licensed MIT by its own
header.

- **License**: MIT. Text at [`LICENSES/MIT.txt`](LICENSES/MIT.txt).
- **Copyright**: `Copyright (c) 2025 Meituan` (verbatim from the file header,
  matching upstream's `LICENSE`).
- **Upstream**: https://github.com/meituan-longcat/LongCat-Flash-Chat
- **In-repo path**: `bench/ngram_ref/modeling_longcat_ngram.py`. The other
  files in `bench/ngram_ref/` carry no third-party header and are project code.

---

## 5. XGrammar — Apache-2.0 (design port)

`crates/grammar/` is a from-scratch pure-Rust reimplementation of XGrammar's
grammar-constrained decoding engine, following mlc-ai/xgrammar v0.1.32 (see
`crates/grammar/DESIGN.md` and
[`docs/adr/0010-vendor-xgrammar.md`](docs/adr/0010-vendor-xgrammar.md)). No
upstream source file is vendored; several test suites port upstream's test
cases and EBNF fixtures. Upstream's notice is recorded here for attribution.

- **License**: Apache-2.0. Text at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- **Copyright**: `Copyright (c) 2024 by XGrammar Contributors` (upstream
  `NOTICE`).
- **Upstream**: https://github.com/mlc-ai/xgrammar
- **In-repo path**: `crates/grammar/`

---

## 6. MLCommons endpoints scorers — Apache-2.0 (ports)

The BFCL v4 dataset provisioner and scorer are Python ports of the MLCommons
endpoints adapter and scorer, kept row-for-row and aggregation-for-aggregation
identical so their scores stay comparable with recorded baselines. The MLPerf
agentic scoring rules are a Rust port of the upstream
`AgenticInferenceInlineScorer` at commit `7935df4`; its parity-fixture
generator executes that upstream class from a local checkout and vendors none
of it.

- **License**: Apache-2.0. Text at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- **Copyright**: `Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
  rights reserved.` (upstream per-file header of all three source files).
- **Upstream**: https://github.com/mlcommons/endpoints
  (`src/inference_endpoint/dataset_manager/predefined/bfcl_v4/__init__.py`,
  `src/inference_endpoint/evaluation/bfcl_v4_scorer.py`,
  `src/inference_endpoint/evaluation/scoring.py`)
- **In-repo paths**: `crates/bench/assets/bfcl/provision.py`,
  `crates/bench/assets/bfcl/score.py`,
  `crates/bench/src/benchmarks/mlperf_agentic/scoring.rs`

---

## 7. TurboQuant and TurboQuant+

The TurboQuant+ KV-cache and weight-rotation work follows the prior-art chain
recorded in [`CITATIONS.md`](CITATIONS.md) and
[`docs/turboquant-plus.md`](docs/turboquant-plus.md).

- **Paper**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni,
  *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*,
  arXiv:2504.19874 (2025). A citation, not vendored source.
- **`TheTom/llama-cpp-turboquant`** — MIT, `Copyright (c) 2023-2026 The ggml
  authors` (the fork keeps llama.cpp's license). The Rademacher sign arrays in
  `kernels/gb10/common/tq_plus_signs.cuh` are byte-identical to
  `TURBO_WHT_SIGNS1`/`TURBO_WHT_SIGNS2` in its `turbo-quant.cuh`, and the
  sparse-V dequant is a port of one of its commits.
  Upstream: https://github.com/TheTom/llama-cpp-turboquant
- **`TheTom/turboquant_plus`** — Apache-2.0. The research repository whose
  papers document the per-feature designs (asymmetric K/V, sparse V, InnerQ,
  layer-aware V, weight rotation). Upstream: https://github.com/TheTom/turboquant_plus
- **In-repo paths**: `kernels/gb10/common/tq_plus_signs.cuh`,
  `kernels/gb10/common/tq_plus_innerq.cuh`,
  `kernels/gb10/common/tq_plus_innerq_apply.cu`,
  `kernels/gb10/common/paged_decode_attn_bf16k_turbo2v.cu` (sparse V),
  `crates/model-arch/src/weight_loader/qwen35/load_layers/tq_plus_weight_rotation.rs`,
  `crates/cache/src/kv_cache/tests_tq_plus.rs`

---

## 8. Build-time dependencies fetched, not vendored

The GB10 builder image (`docker/gb10/Dockerfile.builder`) clones these at
pinned commits and compiles against their headers. Their source is not
committed here, but binaries built with them incorporate it.

### 8a. CUTLASS — BSD-3-Clause

- **License**: BSD-3-Clause. Text at [`LICENSES/BSD-3-Clause.txt`](LICENSES/BSD-3-Clause.txt).
- **Copyright**: `Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES.
  All rights reserved.` (upstream `LICENSE.txt`).
- **Upstream**: https://github.com/NVIDIA/cutlass, pinned by `CUTLASS_SHA`.
- **Used by**: `crates/gpu-runtime/cuda/cutlass_*.cu`, `crates/gpu-runtime/src/cutlass.rs`
  and `crates/gpu-runtime/src/cutlass/`.

### 8b. FlashInfer and its pinned CCCL — Apache-2.0

- **License**: Apache-2.0. Text at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
  FlashInfer's pinned CCCL submodule is Apache-2.0 (Thrust), Apache-2.0 WITH
  LLVM-exception (libcu++) and BSD-3-Clause (CUB).
- **Copyright**: `Copyright 2025-2026 NVIDIA` / `Copyright 2023-2026 FlashInfer
  community (https://flashinfer.ai/)` (upstream `NOTICE`).
- **Upstream**: https://github.com/flashinfer-ai/flashinfer, pinned by
  `FLASHINFER_SHA`.
- **Used by**: `crates/gpu-runtime/cuda/flashinfer_ragged_prefill.cu`, a
  host-callable wrapper that mirrors FlashInfer's `csrc/batch_prefill.cu`
  caller, and `crates/gpu-runtime/src/flashinfer.rs`.

---

## 9. Fonts

Font files served by the book, each under its own font license.

### 9a. Urbanist — OFL-1.1

- **Files**: `book/theme/fonts/urbanist-*.woff2`
- **License text**: shipped in place at
  [`book/theme/fonts/URBANIST-LICENSE.txt`](book/theme/fonts/URBANIST-LICENSE.txt);
  canonical text at [`LICENSES/OFL-1.1.txt`](LICENSES/OFL-1.1.txt).
- **Copyright**: `Copyright 2021 The Urbanist Project Authors
  (https://github.com/coreyhu/Urbanist)`.

### 9b. IBM Plex Mono — OFL-1.1

- **Files**: `book/theme/fonts/ibm-plex-mono-*.woff2`
- **License text**: shipped in place at
  [`book/theme/fonts/IBM-PLEX-MONO-LICENSE.txt`](book/theme/fonts/IBM-PLEX-MONO-LICENSE.txt);
  canonical text at [`LICENSES/OFL-1.1.txt`](LICENSES/OFL-1.1.txt).
- **Copyright**: `Copyright 2017 IBM Corp. All rights reserved.`

---

## 10. Public-domain text

### 10a. *Moby-Dick; or, The Whale*, Herman Melville (1851) — public domain

The opening chapters, up to part of chapter 9, as the prompt of the high-ISL
TTFT gates. The file is compiled into the bench binary with `include_str!`.

- **License**: none. The work is in the public domain in the United States.
- **Source**: Project Gutenberg eBook #2701,
  https://www.gutenberg.org/ebooks/2701. The Project Gutenberg header, footer,
  licence and trademark text and the transcriber's note are removed, so no
  Project Gutenberg licence term applies to the file.
- **In-repo path**: `crates/bench/src/benchmarks/ttft/prompts/long-32k.txt`;
  [`NOTICE.md`](crates/bench/src/benchmarks/ttft/prompts/NOTICE.md) beside it
  records the download's sha256 and the trimming, and
  `scripts/make_long_prompt.py` rebuilds the file from the download.

---

## 11. FlashKDA — MIT (design reference)

`kernels/gb10/common/kda_chunk_tc.cu`, the opt-in tensor-core chunked KDA
prefill (`METRALE_GLM_KDA_PREFILL_CHUNKED_TC=1`), follows two design choices
described in FlashKDA's v1 design note: a chunk of 16 tokens (with
`gate_lower_bound = -5` the in-chunk decay range fits FP32/BF16 without
rescaling) and a split into a token-parallel per-chunk kernel and a
head-parallel recurrence kernel. No upstream source file is vendored or
ported; the kernels were written from the delta-rule math already in
`kernels/gb10/common/kda_chunk.cu`. Upstream's notice is recorded here for
attribution. The general chunk-parallel (WY) form of the delta rule is the one
flash-linear-attention (MIT, `Copyright (c) 2023-2026 Songlin Yang, Yu Zhang,
Zhiyuan Li`, https://github.com/fla-org/flash-linear-attention) implements;
no code was taken from it either.

- **License**: MIT. Text at [`LICENSES/MIT.txt`](LICENSES/MIT.txt).
- **Copyright**: `Copyright (c) 2026 MoonshotAI` (upstream root `LICENSE`).
- **Upstream**: https://github.com/MoonshotAI/FlashKDA
  (`docs/20260420-flashkda-v1-deep-dive.md`)
- **In-repo path**: `kernels/gb10/common/kda_chunk_tc.cu`

---

## Keeping this file current

Re-run these after any change that adds a vendored directory, a build-time
fetch, a ported file or a static asset:

1. `git grep -l 'SPDX-License-Identifier' | xargs grep -L 'SPDX-License-Identifier: MIT OR Apache-2.0'`
   lists files whose header declares a license other than the project's own.
2. `git grep -n -E '^\W*Copyright'` finds foreign copyright lines.
3. `git grep -n -i -E '(ported|adapted|derived|copied) from'` finds ports.
4. `docker/**/Dockerfile*` shows build-time fetches.
