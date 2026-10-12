// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: `METRALE_GLM_KDA_STATE_F16=1`: the GLM-5.3 KDA recurrent state stored as FP16 in
//! the SSM pool (live slots and verify intermediates), the math staying FP32 (model-arch
//! `glm5next_kda/state_f16.rs`, kernels/gb10/common/kda_state_f16.cu).
//!
//! Owner: model-layers (SSM reserve).
//! Invariants:
//! - The pool, the preflight reserve and the model all decide the h pool's width through
//!   [`ssm_h_pool_f16`], so allocation and reserve agree.
//! - The lever narrows the pool of a GLM-5.3 model (`glm5_next` / `glm5_next_text`) only; any
//!   other model keeps exactly `--ssm-h-dtype f16-pool` (`ssm_h_f16_pool_enabled`). It never
//!   touches the Qwen GDN flag cell, so no GDN kernel choice changes.
//! - Lever off (the default): [`ssm_h_pool_f16`] is `ssm_h_f16_pool_enabled()`, the value every
//!   caller passed before.
//!
//! The narrowed pool reuses the `f16-pool` machinery as is: `h_stored_bytes = h_bytes / 2` for
//! the live slots, the verify intermediates and the checkpoints, one FP32 prefill staging blob
//! per slot, FP32 snapshots (save widens, restore narrows), and slot states allocated with
//! `h_is_f16` set.

/// 2026-10-12: Whether a `METRALE_GLM_KDA_STATE_F16` value asks for the FP16 state: `1` only.
pub fn glm_kda_state_f16_requested(v: Option<&str>) -> bool {
    v == Some("1")
}

/// 2026-10-12: `METRALE_GLM_KDA_STATE_F16=1`, read once per process.
pub fn glm_kda_state_f16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        glm_kda_state_f16_requested(std::env::var("METRALE_GLM_KDA_STATE_F16").ok().as_deref())
    })
}

/// 2026-10-12: Whether `model_type` is a GLM-5.3 model, whose linear-attention layers are KDA.
pub fn is_glm_kda_model(model_type: &str) -> bool {
    matches!(model_type, "glm5_next" | "glm5_next_text")
}

/// 2026-10-12: Whether the KDA FP16 state is in force for a model of `model_type`: the lever on
/// and a GLM-5.3 model.
pub fn glm_kda_state_f16_for(model_type: &str) -> bool {
    glm_kda_state_f16() && is_glm_kda_model(model_type)
}

/// 2026-10-12: Env-free core of [`ssm_h_pool_f16`].
pub fn ssm_h_pool_f16_with(qwen_f16_pool: bool, glm_lever: bool, model_type: &str) -> bool {
    qwen_f16_pool || (glm_lever && is_glm_kda_model(model_type))
}

/// 2026-10-12: Whether the SSM h pool of a model of `model_type` is FP16-sized: the Qwen
/// `--ssm-h-dtype f16-pool`, or a GLM-5.3 model under `METRALE_GLM_KDA_STATE_F16=1`.
/// `SsmStatePool::new`'s production call and the server's preflight reserve take the width from
/// here.
pub fn ssm_h_pool_f16(model_type: &str) -> bool {
    ssm_h_pool_f16_with(
        crate::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
        glm_kda_state_f16(),
        model_type,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lever_is_on_only_for_one() {
        assert!(glm_kda_state_f16_requested(Some("1")));
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some(" 1"),
            Some("on"),
            Some("2"),
        ] {
            assert!(!glm_kda_state_f16_requested(v), "{v:?}");
        }
    }

    #[test]
    fn lever_narrows_only_glm() {
        for mt in ["glm5_next", "glm5_next_text"] {
            assert!(ssm_h_pool_f16_with(false, true, mt), "{mt}");
            assert!(!ssm_h_pool_f16_with(false, false, mt), "{mt}");
        }
        for mt in ["qwen3_5", "qwen3_5_moe", "nemotron_h", "kimi_k3", ""] {
            assert!(!ssm_h_pool_f16_with(false, true, mt), "{mt}");
        }
    }

    /// 2026-10-12: Lever off, the width is the Qwen flag, whatever the model.
    #[test]
    fn lever_off_is_the_qwen_flag() {
        for mt in ["glm5_next", "qwen3_5", ""] {
            for q in [false, true] {
                assert_eq!(ssm_h_pool_f16_with(q, false, mt), q, "{mt} {q}");
            }
        }
    }

    /// 2026-10-12: At GLM-5.3 TP2 (32 heads x 128 x 128 per rank) the stored blob is 1 MiB and
    /// the staging blob 2 MiB.
    #[test]
    fn glm_tp2_blob_sizes() {
        let h = 32 * 128 * 128 * 4;
        assert_eq!(crate::ssm_reserve::ssm_h_stored_bytes(h, true), 1 << 20);
        assert_eq!(
            crate::ssm_reserve::ssm_h_prefill_stage_bytes(5, h, true),
            5 << 21
        );
        assert_eq!(
            crate::ssm_reserve::ssm_h_prefill_stage_bytes(5, h, false),
            0
        );
    }
}
