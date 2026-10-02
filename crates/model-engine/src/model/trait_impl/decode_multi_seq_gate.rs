// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that a layer declining the batched multi-sequence path is routed around it.
//!
//! The default `decode_multi_seq` loop passes one `ForwardContext` to every sequence in the
//! batch, so a layer that indexes per-sequence state by a fixed row would compute with another
//! sequence's state. A layer returning true from `decode_multi_seq_unsupported()` is routed per
//! sequence by `decode_a2`'s `hc_perseq` and `decode_b`'s `hc_qsa_perseq`; one returning true
//! from `decode_verify_multi_unsupported()` makes `can_batch_verify_dispatch` refuse the
//! batched verify.
//!
//! The veto is the first disjunct, outside the `hc_mult > 0` conjunction. Inside it, the veto
//! would also need `seq_len >= index_topk + index_compress_ratio - 1` and would miss every
//! shorter sequence. The tests pin that placement, not only the presence of the term.
//!
//! Owner: model-engine decode.
//! Invariants: none beyond the types.

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use metrale_cache::kv_cache::PagedKvCache;
    use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
    use metrale_model_layers::layer::{ForwardContext, LayerState, TransformerLayer};
    use metrale_model_layers::layer::{
        LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
        LayerWriteOnAccept,
    };

    /// 2026-09-25: The two required `TransformerLayer` methods, stubbed. These tests only
    /// call the capability predicates, never a forward.
    macro_rules! stub_forward {
        () => {
            #[allow(clippy::too_many_arguments)]
            fn decode(
                &self,
                _hidden: DevicePtr,
                _residual: DevicePtr,
                _state: &mut dyn LayerState,
                _kv_cache: &mut PagedKvCache,
                _seq_len: usize,
                _block_table: &mut Vec<u32>,
                _disk_block_ids: &mut Vec<u32>,
                _disk_last_offloaded_per_layer: &mut Vec<u32>,
                _ctx: &ForwardContext,
                _stream: u64,
            ) -> Result<()> {
                unreachable!("capability-predicate test never runs a forward")
            }
            fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
                unreachable!("capability-predicate test never allocates state")
            }
        };
    }

    fn src(rel: &str) -> String {
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"))
    }

    /// 2026-09-25: Slice `s` from the first `from` to the next `to` after it, or to the end.
    fn block<'a>(s: &'a str, from: &str, to: &str) -> &'a str {
        let start = s
            .find(from)
            .unwrap_or_else(|| panic!("missing anchor {from:?}"));
        let rest = &s[start..];
        let end = rest.find(to).unwrap_or(rest.len());
        &rest[..end]
    }

    /// 2026-09-25: Both trait defaults are false, so a layer that does not override them keeps
    /// the batched route.
    #[test]
    fn defaults_are_false_so_no_existing_layer_changes_route() {
        struct Plain;
        impl TransformerLayer for Plain {
            stub_forward!();
        }
        impl LayerCapabilities for Plain {}
        impl LayerWeightSetup for Plain {}
        impl LayerWriteOnAccept for Plain {}
        impl LayerGraphHooks for Plain {}
        impl LayerAuxState for Plain {}
        impl LayerSplitPrefill for Plain {}
        assert!(
            !Plain.decode_multi_seq_unsupported(),
            "default must be false — a new predicate may not re-route existing models"
        );
        assert!(
            !Plain.decode_verify_multi_unsupported(),
            "default must be false"
        );
        // 2026-10-01: The three predicates the batched GLM decode added keep every other
        // model on its old route.
        assert!(!Plain.decode_multi_seq_selection_per_seq());
        assert!(!Plain.decode_multi_seq_eager_only());
        assert!(!Plain.fused_decode_prefill_unsupported());
    }

    #[test]
    fn a_declining_layer_is_honoured_on_both_axes_independently() {
        struct DeclinesDecode;
        impl TransformerLayer for DeclinesDecode {
            stub_forward!();
        }
        impl LayerCapabilities for DeclinesDecode {
            fn decode_multi_seq_unsupported(&self) -> bool {
                true
            }
        }
        impl LayerWeightSetup for DeclinesDecode {}
        impl LayerWriteOnAccept for DeclinesDecode {}
        impl LayerGraphHooks for DeclinesDecode {}
        impl LayerAuxState for DeclinesDecode {}
        impl LayerSplitPrefill for DeclinesDecode {}
        struct DeclinesVerify;
        impl TransformerLayer for DeclinesVerify {
            stub_forward!();
        }
        impl LayerCapabilities for DeclinesVerify {
            fn decode_verify_multi_unsupported(&self) -> bool {
                true
            }
        }
        impl LayerWeightSetup for DeclinesVerify {}
        impl LayerWriteOnAccept for DeclinesVerify {}
        impl LayerGraphHooks for DeclinesVerify {}
        impl LayerAuxState for DeclinesVerify {}
        impl LayerSplitPrefill for DeclinesVerify {}
        assert!(DeclinesDecode.decode_multi_seq_unsupported());
        assert!(
            !DeclinesDecode.decode_verify_multi_unsupported(),
            "decode and verify answers must be independent"
        );
        assert!(DeclinesVerify.decode_verify_multi_unsupported());
        assert!(!DeclinesVerify.decode_multi_seq_unsupported());
    }

    /// 2026-09-25: `decode_a2`, the batched decode dispatcher with and without EP. Deleting the
    /// veto term, or moving it inside the `hc_mult > 0` conjunction, fails this test.
    #[test]
    fn decode_a2_routes_a_declining_layer_per_sequence_at_every_length() {
        let s = src("src/model/trait_impl/decode_a2.rs");
        let b = block(&s, "let ms_layer_veto", "if self.comm.is_some()");
        assert!(
            b.contains("decode_multi_seq_unsupported()"),
            "decode_a2 must consult the layer predicate"
        );
        // 2026-09-25: The veto is the first disjunct of hc_perseq, so it is outside `hc_mult > 0`.
        assert!(
            b.contains("let hc_perseq = ms_layer_veto\n            || ("),
            "the veto must be hoisted OUT of the hc_mult/qsa_active conjunction; \
             folded inside, it would only fire at seq_len >= index_topk - 1"
        );
    }

    /// 2026-09-25: `decode_b`, the fused decode + prefill path, which runs only without a comm.
    /// A veto checked only in `decode_a2` would leave this path open.
    #[test]
    fn decode_b_routes_a_declining_layer_per_sequence_at_every_length() {
        let s = src("src/model/trait_impl/decode_b.rs");
        let b = block(&s, "let ms_layer_veto", "if self.comm.is_some()");
        assert!(
            b.contains("decode_multi_seq_unsupported()"),
            "decode_b must consult the layer predicate — it is the single-GPU path"
        );
        assert!(
            b.contains("let hc_qsa_perseq = ms_layer_veto\n            || ("),
            "the veto must be hoisted OUT of the hc_mult/index_topk conjunction"
        );
    }

    /// 2026-09-25: `can_batch_verify_dispatch` refuses the batched verify sweep for a declining
    /// layer as a routing decision, not a mid-request `bail!`.
    #[test]
    fn can_batch_verify_dispatch_consults_the_verify_predicate() {
        let s = src("src/model/trait_impl/verify_e.rs");
        let b = block(&s, "fn can_batch_verify_dispatch", "\n    pub(super) fn ");
        assert!(
            b.contains("decode_verify_multi_unsupported()"),
            "can_batch_verify_dispatch must consult the layer predicate"
        );
        assert!(
            b.contains("&& !self"),
            "the term must be a NEGATED conjunct of the existing self-gate"
        );
    }

    /// 2026-09-25: The veto is consumed at dispatch, not as a serve-time `max_batch_size` clamp
    /// in `serve_load.rs`, which would lower concurrency for every model.
    #[test]
    fn stage0_did_not_reintroduce_a_serve_time_clamp() {
        // 2026-09-26: `load_model`'s steps live in `serve_load.rs` and its child modules;
        // `serve_load/scheduler_setup.rs` sizes `max_batch_size`.
        for rel in [
            "serve_load.rs",
            "serve_load/adapters.rs",
            "serve_load/carried.rs",
            "serve_load/load_phases.rs",
            "serve_load/model_setup.rs",
            "serve_load/scheduler_setup.rs",
        ] {
            let s = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../server/src/main_modules")
                    .join(rel),
            )
            .unwrap_or_else(|e| panic!("read {rel}: {e}"));
            assert!(
                !s.contains("decode_multi_seq_unsupported"),
                "the concurrency capability must be consumed at the DISPATCH site, \
                 never as a serve-time max_batch_size clamp ({rel})"
            );
        }
    }
}
