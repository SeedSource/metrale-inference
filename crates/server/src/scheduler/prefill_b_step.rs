// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `prefill_request`: non-chunked prefill of one new request.
//!
//! Owner: scheduler.
//! Invariants:
//! - Every `Err` returned after the response sink is bound is preceded by
//!   `send_error_to_sink`, so the client gets an error frame rather than a
//!   dropped channel.

use super::*;

mod first_token;

/// 2026-09-25: Prefill a new request and return its `ActiveSeq` for decode.
/// Returns `None` when the request already finished: a beam search, or a
/// first token that is EOS or reaches `max_tokens`.
pub fn prefill_request(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    mut req: InferenceRequest,
    eos_tokens: &[u32],
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    // 2026-09-25: `Some` when this beam request's hypothesis was already
    // computed by a batched search; `None` runs the search here.
    precomputed_beam_hyp: Option<Vec<u32>>,
) -> Result<Option<ActiveSeq>> {
    let stop_tokens = req.take_stop_tokens();
    let eos_tokens = if stop_tokens.is_empty() {
        eos_tokens.to_vec()
    } else {
        let mut merged = eos_tokens.to_vec();
        merged.extend(stop_tokens);
        merged.sort_unstable();
        merged.dedup();
        merged
    };
    let eos_tokens = &eos_tokens;

    let top_k = req.top_k();
    let top_p = req.top_p();
    let top_n_sigma = req.top_n_sigma();
    let min_p = req.min_p();
    let repetition_penalty = req.repetition_penalty();
    let presence_penalty = req.presence_penalty();
    let frequency_penalty = req.frequency_penalty();
    let _dry_multiplier = req.dry_multiplier();
    let _dry_base = req.dry_base();
    let _dry_allowed_length = req.dry_allowed_length();
    let req_lz_penalty = req.lz_penalty();
    let logit_bias = req.logit_bias().to_vec();
    let req_min_tokens = req.min_tokens();
    let req_session_hash = req.session_hash();
    let req_adapter_slot = req.adapter_slot();
    let req_src_lang = req.src_lang_id();
    let req_tgt_lang = req.tgt_lang_id();
    let req_num_beams = req.num_beams();
    let req_length_penalty = req.length_penalty();
    let req_early_stopping = req.early_stopping();
    let req_enable_thinking = req.enable_thinking();
    let req_thinking_budget = req.thinking_budget();
    let req_repetition_detection = req.repetition_detection();
    if req_enable_thinking {
        tracing::info!("Thinking enabled, budget={:?}", req_thinking_budget);
    }
    let req_require_tool_call = req.require_tool_call();
    let req_tools_present = req.tools_present();
    let req_suppress_tool_call = req.suppress_tool_call();
    let req_disable_mtp = req.disable_mtp();
    let req_seed = req.seed();
    let req_top_logprobs = req.top_logprobs();
    let req_timeout_at = req.timeout_at();
    let grammar_spec = req.take_grammar_spec();
    // 2026-09-25: Taken before grammar compilation, so the TTFT measured
    // from it (`decode_start - request_start`) includes compilation.
    let request_start = sched.io.clock.now();
    let mut grammar_state = compile_grammar_state(grammar_engine, &grammar_spec, eos_tokens);
    let (prompt_tokens, max_tokens, mut sink, image_pixels, temperature, cancel_flag) = match req {
        InferenceRequest::Streaming {
            prompt_tokens,
            max_tokens,
            temperature,
            token_tx,
            image_pixels,
            cancel_flag,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Streaming(token_tx),
            image_pixels,
            temperature,
            Some(cancel_flag),
        ),
        InferenceRequest::Blocking {
            prompt_tokens,
            max_tokens,
            temperature,
            response_tx,
            image_pixels,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Blocking(Some(response_tx)),
            image_pixels,
            temperature,
            None,
        ),
    };

    tracing::info!(
        "Prefilling: {} prompt tokens, max_tokens={max_tokens}",
        prompt_tokens.len(),
    );
    let mut seq = match model.alloc_sequence() {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("alloc_sequence failed: {e:#}");
            send_error_to_sink(&sched.io, &mut sink, &msg);
            return Err(e);
        }
    };
    seq.session_hash = req_session_hash;
    seq.adapter_slot = req_adapter_slot;
    seq.src_lang_id = req_src_lang;
    seq.tgt_lang_id = req_tgt_lang;
    seq.num_beams = req_num_beams;
    seq.length_penalty = req_length_penalty;
    seq.early_stopping = req_early_stopping;
    // 2026-09-25: The adapter id is the prefix-cache identity; slot `-1`
    // resolves to the active adapter.
    seq.adapter_id = model.adapter_id_for(req_adapter_slot);
    // 2026-09-25: Take a ref on the resolved LoRA slot, released at the
    // sequence's terminal free.
    seq.acquired_adapter_slot = model.acquire_adapter_slot(req_adapter_slot);

    // 2026-09-25: Beam search runs to completion in the model, and the
    // request finishes here with the winning hypothesis. The chat and
    // completions APIs reject streaming beam requests.
    if model.supports_beam() && seq.num_beams > 1 {
        let hyp = match resolve_beam_hyp(
            model,
            precomputed_beam_hyp,
            &seq,
            &prompt_tokens,
            max_tokens,
        ) {
            Ok(h) => h,
            Err(e) => {
                send_error_to_sink(&sched.io, &mut sink, &format!("beam search failed: {e:#}"));
                let _ = model.free_sequence(&mut seq);
                return Err(e);
            }
        };
        let last = hyp.last().copied().unwrap_or(0);
        let use_legacy_tool_call =
            req_require_tool_call && grammar_state.is_none() && tool_call_start_token.is_some();
        let tool_request = grammar_state.is_some() || use_legacy_tool_call;
        let now = sched.io.clock.now();
        let cached_prompt_tok = seq.reused_prefix_tokens as u32;
        let mut a = ActiveSeq {
            seq,
            session_hash: req_session_hash,
            last_token: last,
            output_tokens: hyp,
            remaining: 0,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            finished: true,
            error: None,
            guard_stop: None,
            param_close_pending: 0,
            sink,
            cancel_flag: cancel_flag.clone(),
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            repetition_penalty_window: 256,
            presence_penalty,
            frequency_penalty,
            lz_penalty: req_lz_penalty,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            logit_bias: logit_bias.clone(),
            pending_drafts: Vec::new(),
            pending_draft_conf: Vec::new(),
            inside_thinking: born_inside_thinking(req_enable_thinking, think_end_token),
            enable_thinking: req_enable_thinking,
            thinking_budget: req_thinking_budget,
            repetition_detection: req_repetition_detection,
            spontaneous_think_budget,
            thinking_tokens: 0,
            cached_prompt_tokens: cached_prompt_tok,
            preempt_immune_until_tokens: 0,
            force_end_thinking: false,
            think_force_closed: false,
            sentence_defer_count: 0,
            consecutive_confident: 0,
            in_code_fence: false,
            think_end_token,
            think_start_token,
            think_ended: !req_enable_thinking && think_end_token.is_some(),
            think_just_ended: false,
            post_think_emitted: 0,
            spec_adapt: Default::default(),
            spec_think_trail: Default::default(),
            think_skip_count: 0,
            require_tool_call: use_legacy_tool_call,
            tool_request,
            tools_present: req_tools_present,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            mtp_acct: Default::default(),
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            tool_call_completed: false,
            post_completion_tool_opens: 0,
            tool_body_streak_tokens: 0,
            inside_parameter_body: false,
            param_body_chars_emitted: 0,
            tool_call_end_token,
            grammar_state,
            last_token_time: now,
            request_start,
            decode_start: now,
            seed: req_seed,
            top_logprobs: req_top_logprobs,
            logprobs_data: Vec::new(),
            timeout_at: req_timeout_at,
            adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
        };
        finish_sequence(&sched.io, &mut a, sched.limits.max_seq_len);
        return Ok(None);
    }

    // 2026-09-25: An error from this block frees the sequence below.
    let prefill_result = (|| -> Result<u32> {
        if !image_pixels.is_empty() {
            model.prepare_vision_embed(&image_pixels)?;
        }

        model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?;
        model.ep_broadcast_cmd(0)?;
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?;
        model.ep_broadcast_tokens(&prompt_tokens)?;
        // 2026-09-25: The worker makes the matching call in its
        // `0xFFFFFFF0` handler.
        model.ep_sync_vision_embeds(&prompt_tokens)?;

        let logits = model.prefill(&prompt_tokens, &mut seq, 0)?;
        sample_first_token(
            model,
            logits,
            temperature,
            top_k,
            top_p,
            min_p,
            eos_tokens,
            grammar_state.as_mut(),
            FirstTokenPolicy::for_birth(
                req_enable_thinking,
                think_end_token,
                tool_call_start_token,
            ),
            &sched.levers.sampling(),
            sched.io.tel.dumps(),
        )
    })();

    let first = match prefill_result {
        Ok(token) => token,
        Err(e) => {
            let msg = format!("prefill failed: {e:#}");
            send_error_to_sink(&sched.io, &mut sink, &msg);
            if let Err(free_err) = model.free_sequence(&mut seq) {
                tracing::error!(
                    "prefill_b_step: free_sequence (after prefill error): {free_err:#}"
                );
            }
            if let Err(bcast_err) = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1)
            {
                tracing::error!(
                    "prefill_b_step: ep_broadcast (after prefill error): {bcast_err:#}"
                );
            }
            return Err(e);
        }
    };

    first_token::finish_first_token(
        sched,
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
        model,
        eos_tokens,
        spontaneous_think_budget,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        req_lz_penalty,
        logit_bias,
        req_min_tokens,
        req_session_hash,
        req_enable_thinking,
        req_thinking_budget,
        req_repetition_detection,
        req_require_tool_call,
        req_tools_present,
        req_suppress_tool_call,
        req_disable_mtp,
        req_seed,
        req_top_logprobs,
        req_timeout_at,
        request_start,
        grammar_state,
        max_tokens,
        sink,
        temperature,
        cancel_flag,
        seq,
        first,
    )
}
