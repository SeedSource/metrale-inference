# GLM DSA prefill row batch (`METRALE_GLM_DSA_ROW_BATCH=1`)

2026-10-01. Branch `seed/dsa-rowbatch` from `abf7cce4`. Status: **written and hand-reviewed, not
compiled and not run**. The authoring host had no cargo, rustc or nvcc. Nothing below is measured.
Before anyone relies on the lever, build it, run the microtest on a GB10 and run the end-to-end byte
check (see "Run").

## What it does

The lever is off by default. When it is on, `Glm5NextDsaLayer::decode_k` takes
`decode_rows_batched` (`crates/model-arch/src/glm5next_dsa/layer/row_batch.rs`) in place of the
per-row loop, but only when all of these hold:

- `batch_select_enabled(..)` is true: the workspace has the batched-selector buffers
  (`METRALE_DSA_SELECT_ROWS` is not `0`), `is_prefill`, `!ctx.graph_capture` and `k > 1`. These are
  the conditions the existing batched selection already needs, so this path is never reached by
  decode, by the speculative verify (eager or captured) or by any graph capture.
- `persist_bt` is true (`METRALE_GLM_DSA_ALLOC_PER_STEP` is not `1`).
- `gemv`, `gemv_f32`, `gemv_batchm` and the new `dense_gemv_bf16_fp32out_batchm` all resolved. If
  `gemv_f32` were missing, the row loop would use the tile GEMM, which the batched path does not
  match, so the batched path refuses.
- `stream == gpu.default_stream()` (see risk 2).

The staging (`DsaRowBatch`) is allocated in `Glm5NextDsaWorkspace::new` only when the lever is on,
`persist`, the batched selector is on and `max_rows > 1`. So when the lever is off, there is no
allocation and no behaviour change. The only changes the old path sees are mechanical:
`select_rows_batched` takes one extra `None` argument, and a local in it was renamed.

## Files changed

| File | Change |
|---|---|
| `kernels/gb10/common/dense_gemv_bf16_batchm.cu` | New entry point `dense_gemv_bf16_fp32out_batchm`. Its diff against `dense_gemv_bf16_batchm` is 3 lines: the name, `float* C` and the unrounded store. |
| `crates/model-layers/src/layers/ops/gemm_quant_gemv.rs`, `gemm_quant.rs` | Launcher `dense_gemv_batchm_fp32out`. It uses the same launch contract as `dense_gemv_batchm`, and the same refusal for `m` outside 1..=16. |
| `crates/model-arch/src/glm5next_dsa/layer/row_batch.rs` (new) | `DsaRowBatch` staging, `row_batch_ready`, `decode_rows_batched`, `indexer_rows_batched`, `qidx_rows_batched`. |
| `crates/model-arch/src/glm5next_dsa/layer/decode_k.rs` | Early branch after the shared projections. The loop is textually untouched. |
| `crates/model-arch/src/glm5next_dsa/layer/rows.rs` | `select_rows_batched(.., row_batch: Option<&DsaRowBatch>, ..)`. With `Some`, it uses the already-uploaded `q_pos` and the batched FP32 `wq_b`. |
| `crates/model-arch/src/glm5next_dsa/layer/workspace.rs` | `row_batch: Option<DsaRowBatch>` field. |
| `crates/model-arch/src/glm5next_dsa/layer.rs` | `mod row_batch;` (496 lines, under the 500 cap). |
| `crates/model-arch/src/glm5next_layer/levers.rs`, `mod.rs` | Reader `dsa_row_batch()`: a OnceLock, `== Ok("1")`. |
| `crates/config/src/levers/table_a.rs` | Lever row, in name order after `METRALE_GLM_DSA_BATCH_QIDX`. |
| `crates/model-arch/src/glm5next_dsa/layer/tests.rs` | Source-check test (prefill-only entry, `gemm` used once), and `row_batch.rs` joins the RMSNorm scan. |
| `crates/model-arch/examples/dsa_rowbatch_bitparity_microtest.rs` (new), `crates/model-arch/Cargo.toml` | GPU bit-parity gate. |

## Equivalence, piece by piece

For each piece, the claim is that the batched path writes the same bytes as the loop to every
output, every cache and every piece of state.

**1. Host metadata.** The loop computes values per row from the same formulas, and the batched
path computes the same values:

- `slot = bt[pos/bs]*bs + pos%bs` (i64)
- `q_pos = pos` (i32)
- `seq_len entry = pos+1` (i32)
- the block-table prefix of `bt_entries_needed(seq_len, k, bs)` entries. This is the same for
  every row, because it does not depend on `row`.

The loop issued about 5 blocking copies per row, each from pageable memory. Each one also drained
the default stream, through the `cuStreamSynchronize` in `copy_h2d`. It then issued the selector's
`q_pos` array copy. The batched path makes one call per `decode_k`, which does three things:

- `event_synchronize`, so the previous call's DMA has finished before the staging is rewritten;
- it fills the page-locked staging and issues two `copy_h2d_async_retained` on `stream`: the
  `[slot|seq_len|q_pos]` block goes to `meta_dev`, and the block table goes to the workspace `bt`
  as before;
- it calls `record_event`.

The consumers are the latent write, the selector and the attend. They run on the same stream after
the copies, so they read identical bytes. The same block-table capacity check runs with the same
message. Under `rowwise_meta` (never on a prefill pass, where `num_seqs == 1`), the slot, block
table and `seq_len` come from the metadata exactly as in the loop.

**2. Latent write.** `glm5next_mla_latent_write_fp8` uses `token = blockIdx.x`. It reads
`kv_a + token*dim` and `slot_mapping[token]`, and nothing else depends on the grid. The loop's
launch for row `r` (one block, with `kv_a.offset(r*dim*2)` and a one-entry slot buffer) is
therefore the same computation as block `r` of a `k`-block launch over the base `kv_a` with the
`k`-entry slot array. The rows write disjoint slots.

**3. Indexer projections.**

- `wk` and `compress_gate` run through `dense_gemv_bf16_batchm` with `out_stride = d`, straight into
  `k_normed` and `gate` rows `[len, len+k)`. Each launch covers at most 16 rows. These rows are
  contiguous, because `row_offset(pos) = pos*d*2`.
- `weights_proj` runs through `dense_gemv_bf16_fp32out_batchm` straight into `head_weights_rows`.
  This drops the loop's single-row slot and its D2D copy.
- `wq_b` (in `select_rows_batched`) runs through the FP32 batched GEMV into `q_idx_rows`.

The batched kernels follow the M = 1 kernels' order exactly: the same per-row kv stride-64 order,
lo-then-hi adds, `__shfl_down` tree and two-warp smem sum, and the build uses `--fmad=false`.
`m` enters no row's operand order, so each row's result is bit-identical to the M = 1 GEMV the loop
launches (`dense_mm_bf16` at m = 1 picks `gemv` or `gemv_f32`). The FP32 kernel is the BF16
batchm body with only the store changed. This identity is the kernel header's claim, and the
microtest checks it.

The cuBLASLt arm of `gemm` (M > 16, not bit-identical) is never used for these projections. Groups
are at most `DENSE_GEMV_BATCHM_MAX_M`, and a host test pins `gemm(` to the single `o_absorb` call
that the loop also makes. If `METRALE_GLM_DSA_BATCH_QIDX=1`, cuBLASLt `wq_b` still takes
precedence, as before. That lever was never bit-identical.

**4. `k_norm`.** `nllb_layernorm_bf16` sets `row = blockIdx.x` and returns if `row >= rows`. Each
block reduces only its own row. It uses the same `blockDim` (`d.min(1024)`), the same shared
memory and the same tree, and `rows` is only the bound. So `rows = k` with `k` blocks over the base
equals `k` launches with `rows = 1` at the row offsets. Every `wk` row is written by earlier launches
on the same stream before the one `k_norm` launch reads it.

**5. Validity, length, room.**

- `memset_async(valid + len, 1, k)` writes the same bytes as `k` one-byte memsets.
- `ensure_room(k)` passes exactly when `k` successive `ensure_room(1)` checks with advancing would
  pass.
- `advance(k)` gives the same `len` as `k` calls of `advance(1)`.

**6. Ordering and causality.** With the batched selector on (a precondition), nothing inside the
loop reads the indexer cache (`k_normed`, `gate`, `valid`), the paged latent cache or
`head_weights_rows`. Selection (`select_rows_batched`) and the attend (`attend_rows`) already run
after the loop. Consider what they read:

- `state.geometry(cfg, k)` reads `state.len()` after all `k` advances, in both paths.
- Row `r` limits itself through `q_pos[r]`, which is the per-row value, the same bytes.
- `seq_lens[r] = seq_len + r + 1` is per row, the same bytes.

`replay_safe` needs `graph_capture`, so it is false, and `indexer_forward`'s host-offset placement
(no `pos_dev`) is what the batched path reproduces. So moving every row's writes ahead of the first
read changes no value that any reader sees. Nothing that would change meaning reads `state.len()`
mid-loop.

**7. Output projection.** It is the same `gemm` call with the same arguments.

## What stays per-row, and what differs only in scratch

- Selection and attend were already batched, and they are unchanged.
- The row loop is untouched, and it still serves decode, the speculative verify, graph capture,
  `k == 1`, `METRALE_DSA_SELECT_ROWS=0`, `METRALE_GLM_DSA_ALLOC_PER_STEP=1` and a non-default stream.
- Some workspace scratch is no longer written: `w.slot`, `w.q_pos`, `w.head_weights`, `w.sl` and
  `w.q_pos_rows`. The loop would leave the last row's values there. Every reader of these buffers
  writes them first in the same call:
  - `w.slot`: written before the latent write in `decode_k` and in `write_kv_row`;
  - `w.q_pos` and `w.head_weights`: written before `select_row`;
  - `w.sl`: all `k` entries written before the attend;
  - `w.q_pos_rows`: written before the selector.

  None of these buffers is an output, cache, state or aux-state input.
- If the call fails, it fails before any launch (a short block table, the `bt` capacity check or a
  full cache). The loop could fail part-way, with earlier rows already written and advanced. Results
  on success are unaffected.

## Risks

1. **Unverified build.** The code has not been compiled, formatted or clippy-checked. It was laid out
   by hand to rustfmt's defaults. Run `cargo fmt --all -- --check` and `scripts/check.sh clippy
   --tests` before anything else.
2. **Stream ordering.** The staged copies are stream-ordered on `stream`. The loop's copies are
   blocking on the backend's default stream. Mixing them is ordered only when the two streams are
   the same, so `row_batch_ready` requires `stream == gpu.default_stream()`. The multi-rank (TP2)
   prefill passes the default stream. A single-rank prefill on another stream silently keeps the
   loop, so check that the lever actually engages before reading a timing.
3. **The BF16 batchm identity at M = 16.** `bf16_batch_bitparity_microtest` grades only M <= 8.
   The GLM prefill already uses batchm at M = 16 for `q_a`, `kv_a` and the other projections on the
   same claim. The new microtest grades M = 16 at the indexer shapes.
4. **Pinned memory.** The staging holds `16*max_rows + 4*bt_cap` bytes of page-locked memory per DSA
   layer. That is about 4 B per context token per layer: at `max_context = 262144` it is about
   1 MiB per layer, or 11 MiB over 11 layers. Nothing frees it, as with DFlash's staging. Device
   memory adds `16*max_rows` bytes per layer.
5. **Event semantics.** The design relies on `cuEventSynchronize` on a never-recorded event
   returning at once (CUDA driver API). The default backends make events no-ops and
   `copy_h2d_async_retained` blocking, which is still correct.
6. **KERNEL-PERF.md.** `scripts/kernel_perf.py --check` was already stale at `abf7cce4`. The new
   entry point adds to that, and the file was not regenerated here.

## Run

```bash
# host checks
scripts/check.sh fmt --all -- --check
scripts/check.sh clippy -p metrale-model-arch --tests
cargo test -p metrale-model-arch glm5next_dsa::layer::tests
cargo test -p metrale-config levers

# GPU bit-parity gate (GB10; the latent write is a GLM target kernel). Exit 0 = PASS.
METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
  cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
  --example dsa_rowbatch_bitparity_microtest

# end to end: two fresh serves, identical except the lever; compare the hash column
#   arm A: (unset)                      arm B: METRALE_GLM_DSA_ROW_BATCH=1
scripts/glm53-byte-identity-probe.sh    # first six requests after each start
# plus an 8192-token prefill under nsys: per-row cudaMemcpy HtoD and the 1-block
# glm5next_mla_latent_write_fp8 launches should be gone in arm B
```
