// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `METRALE_GLM_MOE_CSF_SCALES` lever (`glm_moe_csf_scales`, csfcar).
//!
//! Owner: config.
//! Invariants: read once per process; only `1` (trimmed) turns it on.

/// 2026-10-17: Whether GLM-5.3 holds its routed-expert NVFP4 block scales losslessly compressed
/// on the device ("scale rows with escape", model-arch `glm5next_mlp/scale_rows.rs`; the MoE
/// kernels read them through `moe_scale_byte`) instead of plain E4M3 bytes. Off by default; read
/// once per process. Read by the GLM-5.3 weight loader.
pub fn glm_moe_csf_scales() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        glm_moe_csf_scales_from(std::env::var("METRALE_GLM_MOE_CSF_SCALES").ok().as_deref())
    })
}

/// 2026-10-17: [`glm_moe_csf_scales`] from a raw value: `1` (trimmed) is on, anything else off.
pub fn glm_moe_csf_scales_from(value: Option<&str>) -> bool {
    value.map(str::trim) == Some("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_turns_it_on() {
        assert!(glm_moe_csf_scales_from(Some("1")));
        assert!(glm_moe_csf_scales_from(Some(" 1 ")));
        for v in [None, Some(""), Some("0"), Some("on"), Some("true")] {
            assert!(!glm_moe_csf_scales_from(v), "{v:?}");
        }
    }
}
