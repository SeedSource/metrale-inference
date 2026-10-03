// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The generic transformer model, [`TransformerModel`], and the served NLLB model.
//!
//! Layer-specific logic lives behind the `TransformerLayer` trait objects in
//! `TransformerModel::layers`; `trait_impl` implements `Model` for it.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

pub(crate) mod block_mgmt;
pub(crate) mod drop;
mod gdn_carry;
pub(crate) mod impl_a1;
pub(crate) mod impl_a1_init;
pub(crate) mod impl_a2;
mod impl_a2_ep_broadcast;
pub(crate) mod impl_a3;
mod impl_a3_embed;
mod impl_a3_lm_head;
mod impl_a3_norm;
pub(crate) mod impl_b1;
mod impl_b1_decode;
pub(crate) mod impl_b2;
pub(crate) mod impl_b3;
pub(crate) mod impl_b3_accessors;
pub(crate) mod impl_lora;
mod impl_lora_rotate;
pub(crate) mod impl_lora_swap;
mod impl_ngram;
pub(crate) mod lm_head_q6k;
pub(crate) mod mtp_stage;
pub(crate) mod pinned_pack;
pub(crate) mod seq_memtrace;
pub(crate) mod ssm_batched_copy;
pub(crate) mod ssm_pool;
mod ssm_pool_slots;
pub(crate) mod ssm_snapshot;
mod ssm_snapshot_auxbudget;
mod ssm_snapshot_decode;
pub(crate) mod ssm_snapshot_faultin;
mod ssm_snapshot_init;
pub(crate) mod ssm_snapshot_spill;
mod ssm_snapshot_teardown;
pub(crate) mod ssm_spill_gate;
pub(crate) mod ssm_spill_staging;
pub(crate) mod ssm_tier;
pub(crate) mod token_overlay;
pub(crate) mod trait_impl;
pub(crate) mod types;

// 2026-09-25: Served NLLB-200 / M2M-100 encoder-decoder model (cuda feature only).
#[cfg(feature = "cuda")]
pub mod nllb;
#[cfg(all(test, not(feature = "cuda")))]
#[path = "nllb/host_tests.rs"]
mod nllb_host_tests;

pub use types::TransformerModel;
