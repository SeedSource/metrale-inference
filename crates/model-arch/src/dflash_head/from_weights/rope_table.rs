// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The DFlash drafter's RoPE inverse-frequency table, from its config's
//! `rope_scaling`.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use crate::weight_loader::DflashWeights;

/// 2026-09-26: The `rotary_dim / 2` inverse frequencies for `rope_theta`; logs the RoPE kind.
pub(super) fn rope_inv_freq_table(
    weights: &DflashWeights,
    rope_theta: f32,
    rotary_dim: usize,
) -> Vec<f32> {
    let dim_f = rotary_dim as f32;
    let n_pairs = rotary_dim / 2;
    let mut inv_freq_table = vec![0.0f32; n_pairs];

    for j in 0..n_pairs {
        inv_freq_table[j] = 1.0 / rope_theta.powf((2 * j) as f32 / dim_f);
    }

    let rope_kind: &str;
    if let Some(scaling) = weights.config.rope_scaling.as_ref() {
        match scaling.rope_type.as_deref() {
            Some("yarn") => {
                let factor = scaling.factor.unwrap_or(1.0);
                let beta_fast = scaling.beta_fast.unwrap_or(32.0);
                let beta_slow = scaling.beta_slow.unwrap_or(1.0);
                let orig_max_pos = scaling.original_max_position_embeddings.unwrap_or(4096.0);
                let find_correction_dim = |num_rot: f32| -> f32 {
                    (dim_f * (orig_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
                        / (2.0 * rope_theta.ln())
                };
                let low = find_correction_dim(beta_fast).floor().max(0.0);
                let high = find_correction_dim(beta_slow)
                    .ceil()
                    .min((rotary_dim - 1) as f32);
                let ramp_denom = if (high - low).abs() < 1e-6 {
                    high - low + 0.001
                } else {
                    high - low
                };
                for j in 0..n_pairs {
                    let pos_freq = rope_theta.powf((2 * j) as f32 / dim_f);
                    let inv_freq_extrap = 1.0 / pos_freq;
                    let inv_freq_interp = 1.0 / (factor * pos_freq);
                    let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
                    let extrap_factor = 1.0 - ramp;
                    inv_freq_table[j] =
                        inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor;
                }
                tracing::info!(
                    target: "metrale_model_arch::dflash_head::from_weights",
                    "DFlash RoPE = YaRN: theta={rope_theta}, factor={factor}, \
                     beta_fast={beta_fast}, beta_slow={beta_slow}, \
                     max_pos={orig_max_pos}, low_dim={low:.1}, high_dim={high:.1}",
                );
                rope_kind = "yarn";
            }
            // 2026-10-01: transformers 5.x writes `rope_type: "default"` for plain
            // RoPE (both GLM-5.3 DFlash2 drafters); the table above is already plain.
            Some("default") => {
                tracing::info!(
                    target: "metrale_model_arch::dflash_head::from_weights",
                    "DFlash RoPE = plain (rope_type \"default\"), theta={rope_theta}, \
                     {n_pairs} pairs",
                );
                rope_kind = "plain";
            }
            Some(other) => {
                tracing::warn!(
                    target: "metrale_model_arch::dflash_head::from_weights",
                    "DFlash drafter config has rope_scaling.rope_type={other:?} which Metrale Engine \
                     doesn't recognise — falling back to plain RoPE (theta={rope_theta})."
                );
                rope_kind = "plain (unknown rope_type)";
            }
            None => {
                tracing::warn!(
                    target: "metrale_model_arch::dflash_head::from_weights",
                    "DFlash drafter config has rope_scaling without rope_type — \
                     falling back to plain RoPE (theta={rope_theta})."
                );
                rope_kind = "plain (no rope_type)";
            }
        }
    } else {
        tracing::info!(
            target: "metrale_model_arch::dflash_head::from_weights",
            "DFlash RoPE = plain (no rope_scaling in drafter config), theta={rope_theta}, \
             {n_pairs} pairs",
        );
        rope_kind = "plain";
    }
    let _ = rope_kind;
    inv_freq_table
}
