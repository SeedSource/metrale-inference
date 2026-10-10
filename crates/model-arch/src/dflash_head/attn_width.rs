// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The DFlash drafter's head width → its γ-block paged-indirect attention module (A93).
//!
//! `attn_prefill_paged_indirect` is compiled with a fixed tile width `HDIM`, and the drafter passes
//! its runtime `head_dim` to it. The common module is built with HDIM 256. With a head_dim-128
//! drafter (incoai GLM-5.3-Flash-DFlash2) every tile row then loads heads `h` and `h+1`, and each
//! head's scores become `Q_h·K_h + Q_{h+1}·K_{h+1}`: in bounds, wrong weights. Measured on GB10
//! 2026-09-07 (spark-bench HANDOFF-08): count_pin acceptance 4.55/7 → 6.17/7 once the width matched
//! (vLLM DFlash2 6.10/7). Ported from our own Atlas fix ea0e51269 (Seed-Cluster, charter L2).
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - This is the one place that maps a drafter width to a module; a width with no build is an error.
//! - [`assert_width`] runs before the handle is bound, so a mismatched module cannot be armed silently.

use anyhow::{Result, bail};

/// 2026-10-10: One compiled width of the paged-indirect γ-block attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagedIndirectSpec {
    /// `kernels/gb10/common/KERNEL.toml` `[modules]` name.
    pub module: &'static str,
    /// `extern "C"` entry symbol.
    pub func: &'static str,
    /// Compile-time `HDIM` of the module.
    pub hdim: usize,
}

/// 2026-10-10: The common build (`attn_prefill_paged_indirect.cu`, the .cuh default HDIM 256).
pub(crate) const PAGED_INDIRECT_H256: PagedIndirectSpec = PagedIndirectSpec {
    module: "prefill_paged_indirect",
    func: "attn_prefill_paged_indirect",
    hdim: 256,
};

/// 2026-10-10: The HDIM 128 build (`attn_prefill_paged_indirect_h128.cu`).
pub(crate) const PAGED_INDIRECT_H128: PagedIndirectSpec = PagedIndirectSpec {
    module: "prefill_paged_indirect_h128",
    func: "attn_prefill_paged_indirect_h128",
    hdim: 128,
};

/// 2026-10-10: The build whose width equals the drafter's `head_dim`; any other width is an error.
pub(crate) fn paged_indirect_spec_for(head_dim: usize) -> Result<PagedIndirectSpec> {
    match head_dim {
        128 => Ok(PAGED_INDIRECT_H128),
        256 => Ok(PAGED_INDIRECT_H256),
        other => bail!(
            "DFlash drafter head_dim={other} has no paged-indirect attention build (compiled \
             widths: 128, 256); refusing to run the γ-block attention at a mismatched tile width \
             (A93). Add an attn_prefill_paged_indirect_h{other} module and map it in \
             dflash_head::attn_width."
        ),
    }
}

/// 2026-10-10: The module about to be bound must be compiled for exactly the drafter's width.
pub(crate) fn assert_width(spec: PagedIndirectSpec, head_dim: usize) -> Result<()> {
    if spec.hdim != head_dim {
        bail!(
            "DFlash paged-indirect attention width mismatch: `{}` is compiled HDIM={} but the \
             drafter head_dim={head_dim}; refusing to arm the drafter (A93: the mismatch mixes \
             adjacent heads' scores while staying in bounds)",
            spec.func,
            spec.hdim
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_128_resolves_the_h128_build() {
        let s = paged_indirect_spec_for(128).unwrap();
        assert_eq!(s, PAGED_INDIRECT_H128);
        assert_eq!(s.hdim, 128);
    }

    #[test]
    fn head_dim_256_resolves_the_common_build() {
        assert_eq!(paged_indirect_spec_for(256).unwrap(), PAGED_INDIRECT_H256);
    }

    #[test]
    fn widths_without_a_build_are_errors() {
        for w in [0usize, 64, 96, 192, 512] {
            let e = paged_indirect_spec_for(w).unwrap_err().to_string();
            assert!(e.contains(&format!("head_dim={w}")), "{e}");
        }
    }

    #[test]
    fn the_guard_refuses_the_a93_pairing() {
        let e = assert_width(PAGED_INDIRECT_H256, 128)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("compiled HDIM=256") && e.contains("head_dim=128"),
            "{e}"
        );
        assert!(assert_width(PAGED_INDIRECT_H128, 256).is_err());
        for w in [128usize, 256] {
            assert_width(paged_indirect_spec_for(w).unwrap(), w).unwrap();
        }
    }

    /// 2026-10-10: The Rust names match the kernel sources and the KERNEL.toml alias, the h128 file
    /// pins HDIM 128 under its own symbol, and the drafter resolves the kernel through this module
    /// (no fixed `prefill_paged_indirect` lookup is left in `kernel_handles.rs`).
    #[test]
    fn sources_match_the_specs() {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels/gb10/common/");
        let h128 = std::fs::read_to_string(format!("{root}attn_prefill_paged_indirect_h128.cu"))
            .expect("h128 source");
        assert!(h128.contains("#define HDIM 128"));
        assert!(h128.contains(&format!("#define KERNEL_NAME {}", PAGED_INDIRECT_H128.func)));
        assert!(h128.contains("#include \"attn_prefill_paged_indirect.cu\""));
        let common = std::fs::read_to_string(format!("{root}attn_prefill_paged_indirect.cu"))
            .expect("common source");
        assert!(common.contains(
            "#ifndef KERNEL_NAME\n#define KERNEL_NAME attn_prefill_paged_indirect\n#endif"
        ));
        assert!(
            !common.contains("#define HDIM"),
            "256 comes from the .cuh default"
        );
        let cuh = std::fs::read_to_string(format!("{root}prefill_paged_compute.cuh")).unwrap();
        assert!(cuh.contains("#ifndef HDIM\n#define HDIM 256\n#endif"));
        let toml = std::fs::read_to_string(format!("{root}KERNEL.toml")).expect("KERNEL.toml");
        for s in [PAGED_INDIRECT_H128, PAGED_INDIRECT_H256] {
            let line = format!("{} = \"{}\"", s.func, s.module);
            assert!(toml.contains(&line), "KERNEL.toml must alias `{line}`");
        }
        let handles = include_str!("from_weights/kernel_handles.rs");
        assert!(
            !handles.contains("\"prefill_paged_indirect\""),
            "fixed-width lookup is back"
        );
        assert!(handles.contains("attn_width::paged_indirect_spec_for(head_dim)"));
        assert!(handles.contains("attn_width::assert_width("));
    }
}
