// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-family model architectures, their weight loaders (`weight_loader`, `mistral_loader`) and the DFlash and MTP draft heads.
//!
//! Owner: model-arch.
//! Invariants: none beyond the types.

pub mod attn_v41;
pub mod deepseek_v41_layer;
pub mod deepseek_v41_ref;
pub mod deepseek_v4_mtp;
pub mod dflash_head;
pub mod engram_v41;
pub mod glm5next_dsa;
pub mod glm5next_dsa_ref;
pub mod glm5next_kda;
pub mod glm5next_kda_ref;
pub mod glm5next_layer;
pub mod glm5next_mhc;
pub mod glm5next_mlp;
pub mod glm5next_mtp_head;
pub mod glm5next_numerics;
pub mod glm5next_skeleton;
pub mod kimi_k3;
pub mod mistral_loader;
pub mod moe_v41;
pub mod nemotron_mamba2;
pub mod nemotron_moe;
#[cfg(test)]
pub mod ple_tests;
pub mod precision_schedule;
pub mod seq_state_reserve;
pub mod tp_shard;
pub mod weight_loader;
