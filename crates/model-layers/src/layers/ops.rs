// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel launch wrappers that the layers compose into forward
//! passes, grouped into the submodules under `ops/` and re-exported here
//! (most by glob).
//!
//! Owner: model-layers (ops).
//! Invariants: none beyond the types.

#[path = "ops/activations.rs"]
mod activations;
#[path = "ops/derived_weights.rs"]
mod derived_weights;
#[path = "ops/dispatch_config.rs"]
mod dispatch_config;
#[cfg(test)]
#[path = "ops/dispatch_config_routing_tests.rs"]
mod dispatch_config_routing_tests;
#[path = "ops/dispatch_helpers.rs"]
mod dispatch_helpers;
#[path = "ops/dispatch_proj.rs"]
mod dispatch_proj;
#[cfg(test)]
#[path = "ops/kernel_tree_tests_util.rs"]
mod kernel_tree_tests_util;
// 2026-09-25: W8A8 routing of the decode projections at `DECODE_W8A8_ROWS` rows.
#[path = "ops/dispatch_proj_decode.rs"]
mod dispatch_proj_decode;
// 2026-09-25: Resolution of the compiled target's serving defaults (`TargetLevers`).
#[path = "ops/target_defaults.rs"]
pub mod target_defaults;
// 2026-09-25: Row-wise FP8 projection routing (`cublas_fp8_rowwise_proj`).
#[path = "ops/dispatch_proj_rowwise.rs"]
mod dispatch_proj_rowwise;
#[path = "ops/embeddings.rs"]
mod embeddings;
#[path = "ops/fp8_act_quant.rs"]
mod fp8_act_quant;
// 2026-09-25: The CTA-count floor and the lever of the Hopper FP8 activation-quant launch.
#[path = "ops/fp8_act_quant_floor.rs"]
mod fp8_act_quant_floor;
#[path = "ops/fp8_gemv_batch.rs"]
mod fp8_gemv_batch;
// 2026-09-25: Launchers including the batch-2 and batch-3 FP8 MoE expert kernels.
#[path = "ops/fp8_moe.rs"]
mod fp8_moe;
#[path = "ops/fp8_moe_batch_a.rs"]
mod fp8_moe_batch_a;
#[path = "ops/fp8_moe_batch_b.rs"]
mod fp8_moe_batch_b;
mod fp8_moe_grouped;
// 2026-09-27: The grouped NVFP4 MoE decode kernels (`moe/forward_nvfp4_grouped_decode.rs`).
mod nvfp4_moe_grouped;
// 2026-09-25: The 32-row M-tile twin of `w8a16_gemm_pipelined` and its by-M selector.
mod w8a16_gemm_pipelined_m32;
// 2026-09-25: Tensor-core BF16 GEMM with a 16-row M tile (`dense_gemm_m16_bf16`).
#[path = "ops/dense_gemm_m16_bf16.rs"]
mod dense_gemm_m16_bf16;
#[path = "ops/gemm_dense.rs"]
mod gemm_dense;
#[path = "ops/gemm_dense_int8.rs"]
mod gemm_dense_int8;
#[path = "ops/gemm_fp4.rs"]
mod gemm_fp4;
#[path = "ops/model_stats.rs"]
pub mod model_stats;
#[path = "ops/w8a16_gemm_m16.rs"]
mod w8a16_gemm_m16;
// 2026-09-25: N-column-blocked W8A16 batch-16 GEMVs (`w8a16_gemv_batch16_ncol2`/`ncol4`).
#[path = "ops/w8a16_gemv_ncol.rs"]
mod w8a16_gemv_ncol;
pub use model_stats::ModelStats;

#[path = "ops/gemm_fp8_prefill.rs"]
mod gemm_fp8_prefill;
#[path = "ops/gemm_quant.rs"]
mod gemm_quant;
#[path = "ops/gemv_q2.rs"]
mod gemv_q2;
#[path = "ops/gemv_q2_vec.rs"]
mod gemv_q2_vec;
#[path = "ops/gemv_sw.rs"]
mod gemv_sw;
#[path = "ops/hyper_connection.rs"]
mod hyper_connection;
#[path = "ops/hyper_connection_dispatch.rs"]
mod hyper_connection_dispatch;
#[path = "ops/hyper_connection_lowrank.rs"]
mod hyper_connection_lowrank;
#[cfg(test)]
#[path = "ops/hyper_connection_lowrank_tests.rs"]
mod hyper_connection_lowrank_tests;
#[path = "ops/kv_cache.rs"]
mod kv_cache;
#[path = "ops/kv_cache_fp8k.rs"]
mod kv_cache_fp8k;
#[path = "ops/kv_cache_turbok.rs"]
mod kv_cache_turbok;
#[path = "ops/lora_delta.rs"]
pub mod lora_delta;
#[path = "ops/model_levers.rs"]
mod model_levers;
#[path = "ops/moe_atomic_c4.rs"]
mod moe_atomic_c4;
#[path = "ops/moe_expert.rs"]
mod moe_expert;
#[path = "ops/moe_expert_more.rs"]
mod moe_expert_more;
#[path = "ops/moe_gate.rs"]
mod moe_gate;
#[path = "ops/moe_grouped_a.rs"]
mod moe_grouped_a;
#[path = "ops/moe_grouped_a2.rs"]
mod moe_grouped_a2;
#[path = "ops/moe_grouped_b.rs"]
mod moe_grouped_b;
#[path = "ops/moe_grouped_fp4.rs"]
mod moe_grouped_fp4;
#[path = "ops/moe_lora_grouped.rs"]
pub mod moe_lora_grouped;
#[path = "ops/moe_prefill.rs"]
mod moe_prefill;
#[path = "ops/norm.rs"]
mod norm;
// 2026-09-25: Tensor-core routing of the MTP drafter's BF16 GEMV.
#[path = "ops/dense_gemv_tc.rs"]
pub mod dense_gemv_tc;
#[path = "ops/gemv_tc.rs"]
pub mod gemv_tc;
// 2026-10-04: METRALE_GLM_GEMV_TC: small-M tensor-core dense GEMVs (BF16 and FP8 weights).
#[path = "ops/dense_gemv_tcm.rs"]
pub mod dense_gemv_tcm;
#[cfg(test)]
#[path = "ops/kquant_fold_tests.rs"]
mod kquant_fold_tests;
mod kquant_mmq;
#[cfg(test)]
#[path = "ops/kquant_mmq_tests.rs"]
mod kquant_mmq_tests;
#[cfg(test)]
#[path = "ops/norm_gated_rms_strided_tests.rs"]
mod norm_gated_rms_strided_tests;
mod nvfp4_mmq;
#[path = "ops/ple.rs"]
mod ple;
#[path = "ops/prefill_attn_a.rs"]
mod prefill_attn_a;
#[path = "ops/prefill_attn_b.rs"]
mod prefill_attn_b;
#[path = "ops/prefill_attn_batched.rs"]
mod prefill_attn_batched;
#[path = "ops/prefill_attn_fa128.rs"]
mod prefill_attn_fa128;
#[path = "ops/prefill_attn_fp8k.rs"]
mod prefill_attn_fp8k;
#[path = "ops/prefill_attn_main_a.rs"]
mod prefill_attn_main_a;
#[path = "ops/prefill_attn_main_b.rs"]
mod prefill_attn_main_b;
#[path = "ops/prefill_attn_turbok.rs"]
mod prefill_attn_turbok;
mod q2_0_mmq;
mod q4k_mmq;
#[path = "ops/qsa.rs"]
mod qsa;
#[path = "ops/quant_dispatch.rs"]
mod quant_dispatch;
#[path = "ops/sampling.rs"]
mod sampling;
#[path = "ops/ssm_ba_gates_hopper.rs"]
mod ssm_ba_gates_hopper;
#[path = "ops/ssm_gdn_a.rs"]
mod ssm_gdn_a;
#[path = "ops/ssm_gdn_a2.rs"]
mod ssm_gdn_a2;
#[path = "ops/ssm_gdn_a3.rs"]
mod ssm_gdn_a3;
#[path = "ops/ssm_gdn_b.rs"]
mod ssm_gdn_b;
#[path = "ops/ssm_gdn_batched.rs"]
mod ssm_gdn_batched;
#[path = "ops/ssm_gdn_carry.rs"]
mod ssm_gdn_carry;
#[path = "ops/ssm_gdn_hopper_prefill.rs"]
mod ssm_gdn_hopper_prefill;
#[path = "ops/ssm_gdn_snap.rs"]
mod ssm_gdn_snap;
#[path = "ops/ssm_gdn_tc_route.rs"]
mod ssm_gdn_tc_route;
#[path = "ops/ssm_gdn_woa.rs"]
mod ssm_gdn_woa;
#[path = "ops/ssm_gdn_wyn.rs"]
mod ssm_gdn_wyn;
#[path = "ops/ssm_mamba.rs"]
mod ssm_mamba;
#[path = "ops/ssm_preproc.rs"]
mod ssm_preproc;
#[path = "ops/ssm_ssd.rs"]
mod ssm_ssd;
pub mod token_overlay;
#[path = "ops/w4a4_proj.rs"]
pub mod w4a4_proj;
// 2026-09-25: Host-side tests of the Hopper `w8a16_gemv` override's index math.
#[cfg(test)]
#[path = "ops/w8a16_gemv_hopper_tests.rs"]
mod w8a16_gemv_hopper_tests;
#[path = "ops/wide_prefill.rs"]
mod wide_prefill;

pub use activations::*;
pub use dense_gemm_m16_bf16::*;
pub use derived_weights::{Derivation, DerivedWeights};
pub use dispatch_config::{CublasScope, GemmDispatch, parse_cublas_scope};
pub use dispatch_helpers::*;
pub use dispatch_proj::*;
pub use dispatch_proj_decode::*;
pub use dispatch_proj_rowwise::*;
pub use embeddings::*;
pub use fp8_act_quant::*;
pub use fp8_act_quant_floor::*;
pub use fp8_gemv_batch::*;
pub use fp8_moe::*;
pub use fp8_moe_batch_a::*;
pub use fp8_moe_batch_b::*;
pub use fp8_moe_grouped::*;
pub use gemm_dense::*;
pub use gemm_dense_int8::*;
pub use gemm_fp4::*;
pub use gemm_fp8_prefill::*;
pub use gemm_quant::*;
pub use gemv_q2::*;
pub use gemv_q2_vec::*;
pub use gemv_sw::*;
pub use nvfp4_moe_grouped::*;

pub use hyper_connection::*;
pub use hyper_connection_dispatch::*;
pub use hyper_connection_lowrank::*;
pub use kquant_mmq::*;
pub use kv_cache::*;
pub use kv_cache_fp8k::*;
pub use kv_cache_turbok::*;
pub use model_levers::ModelLevers;
pub use moe_atomic_c4::*;
pub use moe_expert::*;
pub use moe_expert_more::*;
pub use moe_gate::*;
pub use moe_grouped_a::*;
pub use moe_grouped_a2::*;
#[allow(unused_imports)]
pub(crate) use moe_grouped_b::*;
pub use moe_grouped_fp4::*;
pub use moe_lora_grouped::*;
pub use moe_prefill::*;
pub use norm::*;
pub use nvfp4_mmq::*;
pub use ple::*;
pub use prefill_attn_a::*;
pub use prefill_attn_b::*;
pub use prefill_attn_batched::*;
pub use prefill_attn_fa128::AttnFa128Kernels;
pub use prefill_attn_fp8k::*;
pub use prefill_attn_main_a::*;
pub use prefill_attn_main_b::*;
pub use prefill_attn_turbok::*;
pub use q2_0_mmq::*;
pub use q4k_mmq::*;
pub use qsa::*;
pub use quant_dispatch::*;
pub use sampling::*;
pub use ssm_ba_gates_hopper::*;
pub use ssm_gdn_a::*;
pub use ssm_gdn_a2::*;
pub use ssm_gdn_a3::*;
pub use ssm_gdn_b::*;
pub use ssm_gdn_batched::*;
pub use ssm_gdn_carry::*;
pub(crate) use ssm_gdn_hopper_prefill::*;
pub use ssm_gdn_snap::*;
pub use ssm_gdn_tc_route::*;
pub use ssm_gdn_woa::*;
pub use ssm_gdn_wyn::*;
pub use ssm_mamba::*;
pub use ssm_preproc::*;
pub use ssm_ssd::*;
pub use w8a16_gemm_m16::*;
pub use w8a16_gemm_pipelined_m32::*;
pub use w8a16_gemv_ncol::*;
pub use wide_prefill::*;
