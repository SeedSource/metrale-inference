// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Mock-GPU tests of the Qwen3 SSM layer: state allocation, the batched-verify
//! QKVZ/out_proj dispatch on a native-FP8 GDN layer, and `qkvz_verify_nvfp4_wins`.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: none beyond the types.

use super::*;
use crate::weight_map::Fp8Weight;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

#[test]
fn ssm_state_allocation_uses_layer_sizes_and_defaults() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    let before = gpu.alloc_count();

    let state = layer.alloc_state(&gpu).unwrap();
    let state = state
        .as_any()
        .downcast_ref::<SsmLayerState>()
        .expect("Qwen3 SSM allocation must return SsmLayerState");

    assert_eq!(gpu.alloc_count(), before + 2);
    assert_eq!(
        gpu.read_alloc(state.h_state).unwrap().len(),
        layer.h_state_bytes
    );
    assert_eq!(
        gpu.read_alloc(state.conv_state).unwrap().len(),
        layer.conv_state_bytes
    );
    assert_eq!(layer.h_state_bytes, 32 * 128 * 128 * 4);
    assert_eq!(layer.conv_state_bytes, 8192 * 4 * 4);
    assert!(state.h_state_checkpoint.is_none());
    assert!(state.conv_state_checkpoint.is_none());
    assert!(state.h_state_intermediates.is_empty());
    assert!(state.conv_state_intermediates.is_empty());
    assert!(!state.h_is_f16);
    assert!(state.h_prefill_stage.is_none());
}

// 2026-09-25: Batched-verify QKVZ/out_proj dispatch on a native-FP8 GDN layer, whose
// dense and NVFP4 slots are null and whose only projection weights are the block-scaled
// FP8 pair (`qkvz_fp8w`, `out_proj_fp8w`).

use crate::layer::TransformerLayer;
use crate::weight_map::WeightQuantFormat;
use metrale_gpu_runtime::buffers::BufferArena;

/// 2026-09-25: A layer wired like the native-FP8 GDN arm of the qwen35_dense loader: dense
/// QKVZ slot null, out_proj a null `QuantizedWeight`, no NVFP4 weights, no FFN.
/// `with_qkvz_fp8w` and `with_out_fp8w` choose which block-scaled FP8 weights it gets.
pub(super) fn native_fp8_gdn_layer(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    with_qkvz_fp8w: bool,
    with_out_fp8w: bool,
) -> Qwen3SsmLayer {
    let h = config.hidden_size;
    let qkvz_size = config.ssm_qkvz_size();
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: DevicePtr::NULL,
        },
        in_proj_ba: dw(config.ssm_ba_size() * h * 2),
        conv1d: dw(
            (2 * config.linear_num_key_heads * config.linear_key_head_dim + value_dim)
                * config.linear_conv_kernel_dim
                * 2,
        ),
        a_log: dw(config.linear_num_value_heads * 4),
        dt_bias: dw(config.linear_num_value_heads * 4),
        norm: dw(config.linear_value_head_dim * 2),
        out_proj: QuantizedWeight::null(),
    };
    let mut layer = Qwen3SsmLayer::new_sequential(
        dw(h * 2),
        ssm,
        dw(h * 2),
        FfnComponent::None,
        None,
        None,
        None,
        config,
        gpu,
    )
    .unwrap();
    let fp8 = |n: usize, k: usize| Fp8Weight {
        weight: gpu.alloc(n * k).unwrap(),
        row_scale: gpu.alloc((n / 128) * (k / 128) * 4).unwrap(),
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    layer.set_fp8_decode_weights(
        with_qkvz_fp8w.then(|| fp8(qkvz_size, h)),
        with_out_fp8w.then(|| fp8(h, value_dim)),
    );
    layer
}

/// 2026-09-25: An SSM state with `n_inter` h and conv intermediates laid out like a pool
/// slot.
fn mk_state(gpu: &MockGpuBackend, layer: &Qwen3SsmLayer, n_inter: usize) -> SsmLayerState {
    let h_bytes = layer.h_state_bytes;
    let conv_bytes = layer.conv_state_bytes;
    // 2026-09-25: One slab per family, entries `h_bytes` / `conv_bytes` apart, as a pool
    // slot's intermediates are.
    let h_slab = gpu.alloc(h_bytes * n_inter).unwrap();
    let conv_slab = gpu.alloc(conv_bytes * n_inter).unwrap();
    SsmLayerState {
        h_state: gpu.alloc(h_bytes).unwrap(),
        conv_state: gpu.alloc(conv_bytes).unwrap(),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: (0..n_inter).map(|i| h_slab.offset(i * h_bytes)).collect(),
        conv_state_intermediates: (0..n_inter)
            .map(|i| conv_slab.offset(i * conv_bytes))
            .collect(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    }
}

/// 2026-09-25: Run the batched verify for per-sequence widths `ks` through the layer's
/// `decode_verify_multi` (the entry point model-engine `verify_e.rs` calls) and return
/// its result.
fn run_batched_verify(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &Qwen3SsmLayer,
    ks: &[usize],
) -> anyhow::Result<()> {
    let buffers = BufferArena::new(config, 64, 4096, 16, 32, gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        hc_row_offset: 0,
        gpu,
        config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        // 2026-09-25: The default route; the layer has no FFN, so nothing reads it.
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };
    let kv_config = metrale_cache::kv_cache::KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: config.num_hidden_layers,
        dtype: metrale_cache::kv_cache::KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        v_aliases_k: false,
    };
    let mut kv = metrale_cache::kv_cache::PagedKvCache::new(kv_config, 8, gpu).unwrap();
    let mut states_own: Vec<SsmLayerState> = ks.iter().map(|_| mk_state(gpu, layer, 4)).collect();
    let mut states: Vec<&mut (dyn LayerState + 'static)> = states_own
        .iter_mut()
        .map(|s| s as &mut (dyn LayerState + 'static))
        .collect();
    layer.decode_verify_multi(
        buffers.hidden_states(),
        buffers.residual(),
        ks.len(),
        ks,
        &mut states,
        &mut kv,
        DevicePtr::NULL,
        &ctx,
        0,
    )
}

/// 2026-09-25: R = 4 + 3 = 7 rows on a layer holding only the FP8 pair: both projections
/// run `w8a16_gemv_batch16` (grid ceil(N/4)), and the call succeeds, so no null-slot
/// arm was reached (those return an error). The arm keys on the SSM layer's own
/// `w8a16_gemv_batch16_k`, not on `METRALE_FFN_BATCH16`.
#[test]
fn native_fp8_gdn_batched_verify_r7_dispatches_batch16_gemv() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[4, 3]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    assert!(
        has_fp8_projection(&gpu, qkvz, 7, 12_288, 2_048, [3_072, 1, 1]),
        "QKVZ must consume its block-scaled FP8 pair at M=7"
    );
    assert!(
        has_fp8_projection(&gpu, out, 7, 2_048, 4_096, [512, 1, 1]),
        "out_proj must consume its block-scaled FP8 pair at M=7"
    );
}

/// 2026-09-25: Without `w8a16_gemv_batch16` and without the 32-row twin, R = 7 runs the
/// 128-row `w8a16_gemm_pipelined` (grid ceil(N/32) × ceil(M/128)). The twin's cases are
/// in `tests_m32_tile.rs`.
#[test]
fn native_fp8_gdn_batched_verify_r7_without_batch16_keeps_the_tile_gemm() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    layer.w8a16_gemv_batch16_k = metrale_gpu_runtime::gpu::KernelHandle(0);
    layer.w8a16_gemm_pipelined_m32_k = metrale_gpu_runtime::gpu::KernelHandle(0);
    run_batched_verify(&gpu, &config, &layer, &[4, 3]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    assert!(
        has_fp8_projection(&gpu, qkvz, 7, 12_288, 2_048, [384, 1, 1]),
        "QKVZ must fall back to w8a16_gemm_pipelined at M=7"
    );
    assert!(
        has_fp8_projection(&gpu, out, 7, 2_048, 4_096, [64, 1, 1]),
        "out_proj must fall back to w8a16_gemm_pipelined at M=7"
    );
}

/// 2026-09-25: R = 2 + 2 = 4 runs `w8a16_gemv_batch4` for both projections.
#[test]
fn native_fp8_gdn_batched_verify_r4_still_ok() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[2, 2]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    assert!(has_fp8_projection(
        &gpu,
        qkvz,
        4,
        12_288,
        2_048,
        [3_072, 1, 1]
    ));
    assert!(has_fp8_projection(&gpu, out, 4, 2_048, 4_096, [512, 1, 1]));
}

/// 2026-09-25: With no QKVZ weight in any form, R = 7 returns the QKVZ dispatch error
/// instead of launching `dense_gemm` on the null dense slot.
#[test]
fn batched_verify_null_qkvz_fails_fast() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, false, false);
    let err = run_batched_verify(&gpu, &config, &layer, &[4, 3])
        .expect_err("NULL QKVZ slot must be refused, not launched");
    assert!(
        format!("{err:#}").contains("batched GDN QKVZ dispatch"),
        "wrong error: {err:#}"
    );
}

/// 2026-09-25: With the FP8 QKVZ but no out_proj weight, R = 7 returns the out_proj
/// dispatch error.
#[test]
fn batched_verify_null_out_proj_fails_fast() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, false);
    let err = run_batched_verify(&gpu, &config, &layer, &[4, 3])
        .expect_err("null out_proj must be refused, not launched");
    assert!(
        format!("{err:#}").contains("batched GDN out_proj dispatch"),
        "wrong error: {err:#}"
    );
}

fn scalar_u32(arg: &MockArg) -> Option<u32> {
    let MockArg::Bytes(bytes) = arg else {
        return None;
    };
    (bytes.len() == 4).then(|| u32::from_ne_bytes(bytes.as_slice().try_into().unwrap()))
}

/// 2026-09-25: Whether any launch used `weight`'s block-scaled pair with these M, N, K
/// and grid, a 256-thread block and seven arguments. Input and output buffers are not
/// compared.
fn has_fp8_projection(
    gpu: &MockGpuBackend,
    weight: &Fp8Weight,
    m: u32,
    n: u32,
    k: u32,
    grid: [u32; 3],
) -> bool {
    gpu.launches_snapshot().iter().any(|launch: &MockLaunch| {
        launch.grid == grid
            && launch.block == [256, 1, 1]
            && launch.args.len() == 7
            && launch.args[1] == MockArg::Buffer(weight.weight)
            && launch.args[2] == MockArg::Buffer(weight.row_scale)
            && scalar_u32(&launch.args[4]) == Some(m)
            && scalar_u32(&launch.args[5]) == Some(n)
            && scalar_u32(&launch.args[6]) == Some(k)
    })
}

// 2026-09-25: `qkvz_verify_nvfp4_wins` is a pure function of the row count, the weight
// copies the layer holds and the kill switch, so it is tested directly: the mock gives
// every kernel the same handle and cannot tell the two GEMMs apart.

use super::trait_decode_batched::qkvz_verify_nvfp4_wins;

/// 2026-09-25: At and below `VERIFY_TGEMM_MIN_TOKENS` (8) rows the predicate is false.
#[test]
fn qkvz_verify_keeps_fp8_at_and_below_the_threshold() {
    for m in 1..=8 {
        assert!(
            !qkvz_verify_nvfp4_wins(m, true, true, true, false),
            "M={m} must keep the FP8 arm"
        );
    }
}

/// 2026-09-25: Above 8 rows, with every precondition met, the predicate is true.
#[test]
fn qkvz_verify_takes_nvfp4_above_the_threshold() {
    for m in [9, 12, 16, 32, 64, 65, 128, 512] {
        assert!(
            qkvz_verify_nvfp4_wins(m, true, true, true, false),
            "M={m} must take the NVFP4 arm"
        );
    }
}

/// 2026-09-25: The kill switch (`METRALE_NO_QKVZ_NVFP4_DECODE`) makes it false at every
/// row count.
#[test]
fn qkvz_verify_kill_switch_restores_the_fp8_arm() {
    for m in [9, 16, 32, 128, 512] {
        assert!(
            !qkvz_verify_nvfp4_wins(m, true, true, true, true),
            "kill switch must restore the FP8 arm at M={m}"
        );
    }
}

/// 2026-09-25: Without the NVFP4 twin (`qkvz_nvfp4_t` is `None`) it is false.
#[test]
fn qkvz_verify_declines_without_the_nvfp4_twin() {
    for m in [9, 32, 512] {
        assert!(!qkvz_verify_nvfp4_wins(m, true, false, true, false));
    }
}

/// 2026-09-25: Without a tile GEMM (`deep_k_gemm` gives handle 0) it is false, so
/// `ms_proj_gemm` is never handed a zero handle.
#[test]
fn qkvz_verify_declines_without_a_tile_gemm() {
    for m in [9, 32, 512] {
        assert!(!qkvz_verify_nvfp4_wins(m, true, true, false, false));
    }
}

/// 2026-09-25: Without the FP8 prefill copy it is false even with NVFP4 present: the arm
/// replaces the FP8 arm and no other.
#[test]
fn qkvz_verify_declines_without_the_fp8_copy() {
    for m in [9, 17, 32, 512] {
        assert!(
            !qkvz_verify_nvfp4_wins(m, false, true, true, false),
            "M={m}: no FP8 copy means nothing to divert"
        );
    }
}

/// 2026-09-25: Tests of the 32-row M-tile twin, a child module that uses this file's layer
/// and verify helpers.
#[path = "tests_m32_tile.rs"]
mod m32_tile;
