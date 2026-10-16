// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `METRALE_GLM_NV4_PLAIN_SCALE`: which kernel quantizes the GLM-5.3 dense NVFP4
//! decode copies (`dense_fp8::nv4_kernels`; also the MTP draft-head copy, which goes through
//! `quantize_nvfp4_copy`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset, `0` or anything but `1`: `quantize_bf16_to_nvfp4_mse` (the block scale chosen by
//!   squared error), byte-identical to before this lever.
//! - `1`: `quantize_bf16_to_nvfp4` (plain block scale E4M3(block_max / (6 * scale2)), no
//!   search), same signature and layout. NOT byte-identical. Source idea: plain absmax/6 block
//!   scale as SparkGLM/tbraun (no code copied).
//! - Routed-expert quantization and the other model loaders resolve their own kernels.

use std::sync::OnceLock;

/// 2026-10-16: A `METRALE_GLM_NV4_PLAIN_SCALE` value: `1` is on; unset or anything else off.
pub fn parse_plain_scale(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// 2026-10-16: The quantizer kernel name for the lever state.
pub fn quant_kernel_name(plain: bool) -> &'static str {
    if plain {
        "quantize_bf16_to_nvfp4"
    } else {
        "quantize_bf16_to_nvfp4_mse"
    }
}

/// 2026-10-16: `METRALE_GLM_NV4_PLAIN_SCALE=1`, read once.
pub fn nv4_plain_scale() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        parse_plain_scale(std::env::var("METRALE_GLM_NV4_PLAIN_SCALE").ok().as_deref())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_only_one_is_on() {
        assert!(parse_plain_scale(Some("1")));
        assert!(parse_plain_scale(Some(" 1 ")));
        assert!(!parse_plain_scale(None));
        assert!(!parse_plain_scale(Some("0")));
        assert!(!parse_plain_scale(Some("")));
        assert!(!parse_plain_scale(Some("true")));
    }

    #[test]
    fn kernel_name_follows_lever() {
        assert_eq!(quant_kernel_name(true), "quantize_bf16_to_nvfp4");
        assert_eq!(quant_kernel_name(false), "quantize_bf16_to_nvfp4_mse");
    }
}
