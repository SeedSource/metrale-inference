# Metrale Engine — All models on GB10 (multi-target binary)
#
# Build from repo root:
#   docker build -f docker/gb10/Dockerfile -t metrale-gb10 .
#
# Run (auto-selects kernels based on model config):
#   docker run --gpus all --ipc=host -p 127.0.0.1:8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     metrale-gb10 serve --bind 0.0.0.0 nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
#
#   docker run --gpus all --ipc=host -p 127.0.0.1:8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     metrale-gb10 serve --bind 0.0.0.0 Kbenkhaled/Qwen3.5-35B-A3B-NVFP4

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
# Build with the stable toolchain rustup installed above rather than the
# channel rust-toolchain.toml pins.
ENV RUSTUP_TOOLCHAIN=stable

# 2026-10-03: CUTLASS headers for the vendored FlashKDA (vendor/flashkda, MIT) at FlashKDA's own
# submodule pin (v4.3.2). crates/gpu-runtime/build.rs compiles FlashKDA only when
# FLASHKDA_CUTLASS_HOME is set; it is reached at run time only through the opt-in
# METRALE_GLM_KDA_PREFILL_FLASHKDA=1. CUTLASS is BSD-3-Clause (THIRD_PARTY_NOTICES.md 8a).
ARG FLASHKDA_CUTLASS_SHA=5c149f52a436782210263fb2f19b354443a61c6a
ENV FLASHKDA_CUTLASS_HOME=/opt/cutlass-flashkda
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git ${FLASHKDA_CUTLASS_HOME} && \
    git -C ${FLASHKDA_CUTLASS_HOME} checkout ${FLASHKDA_CUTLASS_SHA}

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/
# quick-speed-bench (metrale-bench) include_str!'s its ISL prompt fixtures from
# tests/fixtures — without this line the release build fails in-container
# ("couldn't read ../../tests/fixtures/bench_prompt_128.txt") while every
# full-tree local build succeeds. Copy just the fixtures, not all of tests/.
COPY tests/fixtures/ tests/fixtures/

ENV METRALE_TARGET_HW=gb10
ENV METRALE_TARGET_MODEL=*
ENV METRALE_TARGET_QUANT=*

RUN cargo build --release -p metrale-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"

# Stamp the source commit into the image so ":latest" provenance is answerable:
#   docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' <tag>
# Pass it with --build-arg METRALE_GIT_SHA=$(git rev-parse HEAD).
ARG METRALE_GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${METRALE_GIT_SHA}"

# NCCL for multi-GPU (EP=2), RDMA userspace for RoCE transport.
# NCCL ≥ 2.28 is required for `ncclMemAlloc`/`ncclMemFree` symmetric memory
# windows (substrate for device-API fused allreduce kernels). The base
# `nvidia/cuda:13.0.0-runtime-ubuntu24.04` image has the NVIDIA CUDA APT
# repo enabled, which carries libnccl2 2.29+ (>=2.28 satisfied). Install
# the latest available, then verify the resolved version meets the floor.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    # Sanity-check the linked NCCL is ≥ 2.28 — fail-fast if a future repo
    # change downgrades the available version below the symmetric-memory floor.
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" >&2; exit 1; }

COPY --from=builder /build/target/release/met /usr/local/bin/met
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE-MIT /LICENSE-MIT
COPY LICENSE-APACHE /LICENSE-APACHE
COPY THIRD_PARTY_NOTICES.md /THIRD_PARTY_NOTICES.md
COPY LICENSES/ /LICENSES/
COPY README.md /README.md

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["met"]
