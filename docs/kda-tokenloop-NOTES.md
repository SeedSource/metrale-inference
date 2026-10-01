# KDA token loop (seed/kda-tokenloop): notes

2026-10-01. Branch `seed/kda-tokenloop` from `abf7cce4`. LOCAL ONLY (quiet mode): not pushed,
not built and not run on a GPU yet. Every claim below about GPU behaviour is **UNVERIFIED** until
the microtest has passed on a GB10 node.

## What it does

`Glm5NextKdaLayer::decode_k` used to make 2 launches per row (`causal_conv1d_update_l2norm`,
then `kda_recurrent_decode_bf16_smem`). With `METRALE_GLM_KDA_TOKEN_LOOP=1` (default OFF), a
`decode_k` with `k > 1` and no snapshots makes 2 launches in total:

1. `causal_conv1d_update_l2norm_rows`: conv + SiLU + L2 over all `k` rows, in row order.
2. `kda_recurrent_prefill_bf16_smem`: the recurrence over all `k` rows, in row order. Each thread
   keeps its FP32 state column in shared memory for the whole launch and writes it to global once.

Anything else (verify with snapshots, `k == 1`, lever off, either kernel missing, the smem kernel
not selectable, `METRALE_GLM_KDA_NO_SMEM=1`) takes the old per-row walk, unchanged.

## Files changed

| File | Change |
|---|---|
| `kernels/gb10/common/kda_recurrent.cu` | + `kda_recurrent_prefill_bf16_smem`. The decode kernel is untouched. |
| `kernels/gb10/common/causal_conv1d.cu` | + `causal_conv1d_update_l2norm_rows`. Existing kernels are untouched. |
| `crates/model-arch/src/glm5next_kda/kernels.rs` | + `conv_rows`, `recurrent_rows` handles, resolved with `try_kernel` (0 when absent). `ENTRY_POINTS` (the required count) is still 14. |
| `crates/model-arch/src/glm5next_kda/mod.rs` | + `kda_token_loop()` OnceLock lever reader, the same pattern as `kda_no_smem()`. |
| `crates/model-arch/src/glm5next_kda/decode.rs` | + `stateful_rows()`; `decode_k` calls it when the conditions hold and walks the rows otherwise. |
| `crates/config/src/levers/table_a.rs` | + `METRALE_GLM_KDA_TOKEN_LOOP` row (Switch, off, Runtime, reader `glm5next_kda/mod.rs`), in name order. |
| `crates/model-arch/examples/kda_tokenloop_microtest.rs` + `Cargo.toml` `[[example]]` | Bitwise A/B gate. |

Kernel registration needs nothing beyond the `.cu` entry point: modules are the `.cu` files in
`kernels/gb10/common/`, and lookups are `gpu.kernel("<module>", "<entry>")`. No model shadow
replaces `causal_conv1d.cu` or `kda_recurrent.cu`, and only gb10 serves `glm5_next`.

## Why it is bit-identical (the argument; the microtest is the proof)

- **No contraction anywhere.** `kernels/gb10/common/KERNEL.toml` builds the whole tree with
  `--fmad=false`, so neither the old nor the new kernels fuse a multiply and an add. Every
  operation is a separately rounded IEEE op. Same file, same flags, same intrinsics (`expf`,
  `__expf`, `rsqrtf`, `__shfl_down_sync`, `__bfloat162float`, `__float2bfloat16`).
- **Recurrence.** Per token, the new kernel uses the decode kernel's expressions in the same order:
  stage `expf(gate)`, `k`, `q * scale`; pass 1 `s = x * decay[kk]; kv += s * k[kk]` serially over
  kk = 0..D-1; `delta = (v - kv) * beta`; pass 2 `s = col[kk] + k[kk] * delta; o += s * q[kk]`
  serially. The only operand that changes is where the pre-decay state is read from: `col[kk]`
  (shared memory) instead of `S[kk*D+vi]` (global). Both hold the same FP32 value, because the
  previous token stored exactly that float. Loading and storing FP32 does no rounding, so even
  FTZ cannot change it (FTZ applies to arithmetic results, which are the same in both kernels).
  Thread to column mapping, grid, block and `#pragma unroll 8` are the same.
- **Conv.** Per token: shift the window, insert the new BF16 input widened to FP32, start from
  bias (null, so 0.0f) then add `win[k] * w[k]` serially, SiLU `acc * (1/(1+__expf(-acc)))`,
  then the L2: the warp `__shfl_down` tree at offsets 16..1, the 4 warp partials added
  left to right, `rsqrtf(total + eps)`, `silu *= r`. The window lives in registers and is written
  back once. It is FP32 in both places, and the weights are widened once rather than per token,
  which gives the same floats. This is the structure of the existing `gdn_verify_fused_conv_kn`,
  which `gdn_conv_kn_microtest` already checks byte for byte against the per-token kernel.
- **Reordering conv and recurrence.** The new path runs the conv for every row, then the
  recurrence for every row. Conv row t+1 reads `qkv_proj[t+1]` and the conv state, and writes
  `conv_out[t+1]`. Recurrent row t reads `conv_out[t]`, `gate[t]` and `beta[t]`, and writes the
  recurrent state and `core[t]`. The two sets do not overlap, so this order gives the same result
  as the interleaved walk.
- **Pointers.** Row t uses `conv_out + t*cd` for q, with k and v at `+qkv` and `+2*qkv` (BF16
  elements), `gate + t*qkv`, `beta + t*heads` and `core + t*qkv` (FP32). These are
  `stateful_row`'s per-row offsets with a row stride added.

## Shared memory and synchronisation

- Recurrent: `(3*D + VPB*(D+1)) * 4` = 18,048 B per block at D=128, VPB=32, the same request as
  the decode smem kernel and under `KDA_SMEM_BUDGET` (48 KiB). Grid (H, D/VPB), block 32.
  Per token: a `__syncthreads` before staging (t>0), so token t-1's readers finish before
  `sh_decay/sh_k/sh_q` is overwritten, and one after staging. Each thread's `col` is private, so
  it needs no barrier. Thread-level early returns were replaced by an `owns` guard so every
  thread reaches every barrier.
- Conv: 32 B static (`warp_sums[8]`), block 256. It adds a 3rd `__syncthreads` per token after
  the L2 apply, so the next token's lane-0 write cannot race the current token's read (the
  same fix as `gdn_verify_fused_conv_kn`).

## Uncertain / risks

1. **Not compiled.** This x86 box has no `cargo`, `rustc`, `rustfmt` or `nvcc` (checked
   2026-10-01). Both the Rust and the CUDA were reviewed by hand only. The first step on a build
   host is `cargo fmt` and `cargo check -p metrale-model-arch -p metrale-config`, then the kernel
   build.
2. **The microtest repeats the launches; it does not call them.** `stateful_row` and
   `stateful_rows` are private, so the example writes out their arguments. Drift between
   `decode.rs` and the example would not be caught. An end-to-end check (two processes, because
   the lever is read once per process) is the second gate: prefill the same prompt with
   `METRALE_GLM_KDA_TOKEN_LOOP=0` and `=1` and compare greedy tokens and logits byte for byte.
3. The conv window takes up to 8 taps (`float win[8]`); `Glm5NextKdaConfig::validate` already
   requires `conv_kernel <= 4`, and `stateful_rows` refuses anything above 8.
4. Row offsets are 32-bit `unsigned int` strides widened to `size_t` per token, so there is no
   overflow for any workspace size.
5. Performance is unmeasured. Expected gains: 2k-2 fewer launches per KDA layer per sub-chunk,
   and the recurrent state (H*D*D*4 B, 2 MiB at H=32) is read and written once per launch
   instead of once per token. Parallelism per token is unchanged (H*D/VPB blocks of 32 threads),
   so a long sub-chunk is now one long-running launch. **PROVISIONAL** until measured.

## Run on a GB10 node (build per the usual node recipe; quiet mode, local only)

```bash
# 1) Bitwise kernel gate (exits 1 on any mismatch, or on a vacuous run)
cargo run -p metrale-model-arch --release --example kda_tokenloop_microtest \
    --features cuda,gpu-examples

# 2) Existing gates must still pass (the per-row path is untouched)
cargo run -p metrale-model-arch --release --example kda_recurrent_microtest --features cuda,gpu-examples
cargo run -p metrale-model-arch --release --example kda_layer_microtest     --features cuda,gpu-examples
cargo test -p metrale-config levers

# 3) End-to-end A/B: two serves of the same image, the lever OFF vs ON, same prompt, greedy
METRALE_GLM_KDA_TOKEN_LOOP=0 met serve ...   # baseline
METRALE_GLM_KDA_TOKEN_LOOP=1 met serve ...   # token loop
```

PASS looks like: every line ends in `PASS`, and the last line reads
`KDA token loop GATE (bitwise vs per-row walk, N elements compared): PASS` with N > 0.
Coverage is heads {32, 64} x k {1, 2, 16, 256} x 2 seeds. Each case compares `conv_out`
(u16), `core`, the final recurrent state and the final conv state (u32).
