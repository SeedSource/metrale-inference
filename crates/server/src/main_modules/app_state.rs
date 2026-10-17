// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `AppState`, the per-model state the HTTP handlers read, and the
//! LoRA adapter routing and promotion it carries.
//!
//! Owner: server (HTTP layer).
//! Invariants: none beyond the types.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::api::InferenceRequest;
use crate::tokenizer::ChatTokenizer;
use crate::{
    conversation_store, rate_limiter, reasoning_parser, request_dumper, response_store, tool_parser,
};

/// 2026-09-26: Map a request's adapter selector to a LoRA pool slot: `None`
/// gives `Some(-1)` (the model's active adapter), a name in `adapter_names`
/// gives `Some(its index)`, and any other name gives `None`.
pub fn resolve_adapter_slot(adapter_names: &[String], adapter: Option<&str>) -> Option<i32> {
    match adapter {
        None => Some(-1),
        Some(name) => adapter_names
            .iter()
            .position(|n| n == name)
            .map(|i| i as i32),
    }
}

pub struct AppState {
    pub tokenizer: ChatTokenizer,
    pub model_name: String,
    /// 2026-09-26: The first `--lora-adapter`'s name, if any. No handler reads
    /// it.
    pub adapter_name: Option<String>,
    /// 2026-09-26: Resident adapter names; index i is the slot
    /// `resolve_adapter_slot` gives that name. Listed by /v1/models, and the
    /// only names `POST /v1/lora/active` accepts.
    pub adapter_names: Vec<String>,
    /// 2026-09-26: The adapter last made active through the LoRA control
    /// routes, starting at the LoRA pool's first adapter, if any. Written,
    /// never read; the model holds the active slot.
    pub active_adapter: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    pub max_seq_len: usize,
    pub request_tx: mpsc::Sender<InferenceRequest>,
    /// 2026-09-26: LoRA control channel to the scheduler: a `LoraCommand` and
    /// its ack. `None` when no `--lora-adapter` pool is loaded.
    pub rotation_tx: Option<mpsc::Sender<crate::scheduler::LoraRotation>>,
    /// 2026-09-26: The model config's `vision` section (`serve_load.rs`).
    pub vision_config: Option<metrale_config::VisionConfig>,
    /// 2026-09-26: Image area bound in pixels (`resolve_vision_max_pixels`);
    /// `None` when none is declared.
    pub vision_max_pixels: Option<usize>,
    /// 2026-09-26: Whether and how `image_url` parts with an http(s) URL are
    /// fetched; off unless `--vision-allow-remote-images`.
    pub remote_image_policy: crate::api::chat::remote_image::RemoteImagePolicy,
    /// 2026-09-26: ffmpeg subprocess policy for video parts; off unless
    /// `--video-allow-ffmpeg`.
    pub video_ffmpeg: metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy,
    /// 2026-09-26: `--video-fps`: frames per second a video is sampled at.
    pub video_fps: f32,
    /// 2026-09-26: Default temperature, top-k and top-p: `generation_config.json`,
    /// else MODEL.toml `[sampling.non_thinking]` (`resolve_sampling_defaults`).
    pub default_temperature: f32,
    pub default_top_k: u32,
    pub default_top_p: f32,
    /// 2026-09-26: `generation_config.json`, else `--default-top-n-sigma`.
    pub default_top_n_sigma: f32,
    /// 2026-09-26: `generation_config.json`, else MODEL.toml
    /// `[sampling.non_thinking]`, else `--default-min-p`.
    pub default_min_p: f32,
    /// 2026-09-26: Tool call parser; `None` turns tool handling off
    /// (`api/chat/prepare.rs`). An `Arc`, so each request's
    /// `GrammarSpec::ToolCall` can share it.
    pub tool_call_parser: Option<std::sync::Arc<dyn tool_parser::ToolCallParser>>,
    /// 2026-09-26: Chat-path levers, resolved at load from MODEL.toml
    /// `[behavior]` and `METRALE_*` variables (`ChatLevers::resolve`).
    pub chat: crate::api::chat::levers::ChatLevers,
    pub reasoning_parser: Option<Box<dyn reasoning_parser::ReasoningParser>>,
    /// 2026-09-26: End-of-thinking token id, from the reasoning parser's
    /// `end_token_id`.
    pub think_end_token_id: Option<u32>,
    /// 2026-09-26: `<think>` token id, when it encodes to one token. An
    /// unclosed `<think>` in the prompt's last 8 tokens turns thinking on with
    /// the model's `max_thinking_budget` (`api/chat/template.rs`).
    pub think_start_token_id: Option<u32>,
    /// 2026-09-26: `--tool-max-tokens`: the `max_tokens` cap for a request with
    /// tools active.
    pub tool_max_tokens: usize,
    /// 2026-09-26: MODEL.toml sampling presets, per request category.
    pub sampling_presets: metrale_kernels::SamplingPresets,
    /// 2026-09-26: The tool-call opener's token id (`<tool_call>`, or
    /// `<minimax:tool_call>` for `minimax_xml`) when it encodes to one token.
    /// Biased by repeat count when tools are active (`api/chat/sampling_setup.rs`).
    pub tool_call_start_token_id: Option<u32>,
    /// 2026-09-26: `--auto-compact`. Any value above 0 turns compaction on; the
    /// trigger is fixed at 70% of `max_seq_len` (`api/chat/template.rs`).
    pub auto_compact_threshold: Option<f32>,
    /// 2026-09-26: `--request-timeout` in seconds; 0 means no deadline.
    pub request_timeout: u32,
    /// 2026-09-26: Set to 0 at load (`serve_load.rs`) and read nowhere.
    pub effective_context: usize,
    /// 2026-09-26: MODEL.toml `[behavior]`, embedded at build time, with
    /// `--max-thinking-budget`, `--tool-grammar` and the `preserve_thinking`
    /// server kwarg applied over it (`serve_load.rs`).
    pub behavior: metrale_kernels::ModelBehavior,
    /// 2026-09-26: `--disable-thinking`: `resolve_thinking` turns thinking off
    /// whatever the request or MODEL.toml asks (`api/chat/thinking.rs`).
    pub disable_thinking: bool,
    /// 2026-09-26: The thinking directive from `--default-chat-template-kwargs`,
    /// used when the request gives no explicit one (`api/chat/prepare.rs`).
    pub default_thinking: crate::ir::ThinkingDirective,
    /// 2026-09-26: `reasoning_effort` from `--default-chat-template-kwargs`,
    /// used when the request gives none and thinking is on
    /// (`api/chat/prepare.rs`). `None` leaves the renderer's `"medium"`
    /// (`tokenizer/chat_render.rs`).
    pub default_reasoning_effort: Option<crate::ir::ReasoningEffort>,
    /// 2026-09-26: In-memory store behind Responses API `previous_response_id`
    /// and Chat Completions `store: true`.
    pub response_store: Arc<response_store::ResponseStore>,
    /// 2026-09-26: Per-client rate limiter; off when `METRALE_RATE_LIMIT_RPM`
    /// and `METRALE_RATE_LIMIT_TPM` are both 0 or unset.
    pub rate_limiter: Arc<rate_limiter::RateLimiter>,
    pub conversation_store: Arc<conversation_store::ConversationStore>,
    /// 2026-09-26: `--dump` JSONL writer; `None` when `--dump` is absent or its
    /// target could not be opened.
    pub dump_writer: Option<request_dumper::DumpHandle>,
    /// 2026-09-26: `--lora-stageable`: adapters a weight peer can promote into
    /// a cache slot, by name.
    pub lora_stageable:
        std::collections::HashMap<String, crate::main_modules::promotion::StageableAdapter>,
    /// 2026-09-26: `METRALE_LORA_PEER`, the weight peer a `--lora-stageable`
    /// promote reads from; without it those names are not promoted.
    pub lora_peer_addr: Option<String>,
    /// 2026-09-26: Single-flight coordinator for promotions. `Some` only with a
    /// resident pool and a stageable source: `--lora-stageable` with a peer,
    /// or `--lora-stageable-disk`.
    pub promotion: Option<Arc<crate::main_modules::promotion::PromotionManager>>,
    /// 2026-09-26: Promoted name → cache slot. A successful promote inserts its
    /// name and drops the evicted one, so later requests for it skip the
    /// promote.
    pub promoted_slots: Arc<std::sync::RwLock<std::collections::HashMap<String, i32>>>,
    /// 2026-09-26: `--lora-stageable-disk`: name → (adapter dir, PEFT config).
    /// The config is parsed and rank-checked at load; the disk swap reads it
    /// again at promote time.
    pub lora_disk_stageable:
        std::collections::HashMap<String, (std::path::PathBuf, metrale_config::PeftAdapterConfig)>,
}

use crate::main_modules::promotion::PromoteReject;

/// 2026-09-26: Turn a timeout budget in seconds into an absolute deadline; 0
/// or less means no deadline (`None`).
pub fn deadline_from(secs: f32, now: std::time::Instant) -> Option<std::time::Instant> {
    if secs > 0.0 {
        Some(now + std::time::Duration::from_secs_f32(secs))
    } else {
        None
    }
}

impl AppState {
    /// 2026-09-26: The absolute deadline for one request: `override_secs` (the
    /// request's `timeout`) when given, else `--request-timeout`.
    pub fn request_deadline(&self, override_secs: Option<f32>) -> Option<std::time::Instant> {
        deadline_from(
            override_secs.unwrap_or(self.request_timeout as f32),
            std::time::Instant::now(),
        )
    }

    /// 2026-09-26: For a name `resolve_adapter_slot` did not find, promote a
    /// stageable adapter into a cache slot.
    ///
    /// - `Ok(None)`: not stageable, or promotion is not armed for it.
    /// - `Ok(Some(slot))`: promoted earlier, or promoted now.
    /// - `Err`: the promote failed (`lora_control` answers 503 for
    ///   `PoolFull`, 502 for `Peer`).
    ///
    /// Concurrent calls for one name share one promote
    /// (`PromotionManager::coalesce`), whose lock is not held across the
    /// scheduler round-trip.
    pub async fn ensure_adapter_hot_opt(&self, name: &str) -> Result<Option<i32>, PromoteReject> {
        // 2026-09-26: Not gated on `lora_peer_addr`: a disk-stageable name needs
        // no peer, and a peer-stageable one without a peer returns `Ok(None)`.
        let (Some(promotion), Some(tx)) = (&self.promotion, &self.rotation_tx) else {
            return Ok(None);
        };
        enum Backing {
            Peer(crate::main_modules::promotion::StageableAdapter, String),
            Disk(std::path::PathBuf),
        }
        let backing = if let Some(st) = self.lora_stageable.get(name).cloned() {
            match &self.lora_peer_addr {
                Some(addr) => Backing::Peer(st, addr.clone()),
                None => return Ok(None),
            }
        } else if let Some((dir, _peft)) = self.lora_disk_stageable.get(name) {
            Backing::Disk(dir.clone())
        } else {
            return Ok(None);
        };

        if let Some(&slot) = self
            .promoted_slots
            .read()
            .expect("promoted_slots poisoned")
            .get(name)
        {
            return Ok(Some(slot));
        }

        let promoted_slots = Arc::clone(&self.promoted_slots);
        let tx = tx.clone();
        let name_owned = name.to_string();

        let slot = promotion
            .coalesce(name, move || async move {
                // 2026-09-26: A leader for this name that finished after the
                // check above has already inserted it.
                if let Some(&slot) = promoted_slots
                    .read()
                    .expect("promoted_slots poisoned")
                    .get(&name_owned)
                {
                    return Ok(slot);
                }
                let (slot, evicted) = match backing {
                    Backing::Peer(st, addr) => {
                        Self::dispatch_promote(&tx, &addr, &name_owned, &st).await?
                    }
                    Backing::Disk(dir) => {
                        Self::dispatch_promote_disk(&tx, &name_owned, &dir).await?
                    }
                };
                let mut ov = promoted_slots.write().expect("promoted_slots poisoned");
                if let Some(ev) = evicted {
                    ov.remove(&ev);
                }
                ov.insert(name_owned.clone(), slot);
                Ok(slot)
            })
            .await?;
        Ok(Some(slot))
    }

    /// 2026-09-26: Whether `name` is a peer- or disk-stageable adapter. Request
    /// routing and `GET /v1/models/{id}` accept such a name, which is not in
    /// `adapter_names`.
    pub fn is_stageable_name(&self, name: &str) -> bool {
        self.lora_stageable.contains_key(name) || self.lora_disk_stageable.contains_key(name)
    }

    /// 2026-09-26: Send one `Promote` to the scheduler and wait up to 30 s for
    /// its ack; a timeout is reported as `PoolFull`.
    async fn dispatch_promote(
        tx: &mpsc::Sender<crate::scheduler::LoraRotation>,
        peer_addr: &str,
        name: &str,
        stageable: &crate::main_modules::promotion::StageableAdapter,
    ) -> Result<(i32, Option<String>), PromoteReject> {
        use crate::scheduler::{LoraAck, LoraCommand};

        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let cmd = LoraCommand::Promote {
            peer_addr: peer_addr.to_string(),
            adapter_id: stageable.peer_stage_id.clone(),
            name: name.to_string(),
            peft: stageable.peft.clone(),
        };
        if tx.send((cmd, ack_tx)).await.is_err() {
            return Err(PromoteReject::Peer(
                "scheduler promote channel closed".to_string(),
            ));
        }
        let acked = tokio::time::timeout(std::time::Duration::from_secs(30), ack_rx).await;
        match acked {
            Err(_timeout) => Err(PromoteReject::PoolFull(
                "promotion timed out waiting for scheduler quiescence; retry".to_string(),
            )),
            Ok(Err(_recv)) => Err(PromoteReject::Peer(
                "scheduler dropped the promote ack (shutting down?)".to_string(),
            )),
            Ok(Ok(Err(reason))) => {
                if reason.contains("POOL_FULL") || reason.contains("ref_count>0") {
                    Err(PromoteReject::PoolFull(reason))
                } else {
                    Err(PromoteReject::Peer(reason))
                }
            }
            Ok(Ok(Ok(LoraAck::Promoted { slot, evicted }))) => Ok((slot as i32, evicted)),
            Ok(Ok(Ok(LoraAck::Done))) => Err(PromoteReject::Peer(
                "scheduler returned a non-promote ack for a promote".to_string(),
            )),
        }
    }

    /// 2026-09-26: [`Self::dispatch_promote`] for a disk-stageable adapter: sends
    /// `PromoteDisk`, with the same 30 s bound and error mapping. The model reads
    /// the directory's `adapter_config.json` itself, so no PEFT config is sent.
    async fn dispatch_promote_disk(
        tx: &mpsc::Sender<crate::scheduler::LoraRotation>,
        name: &str,
        dir: &std::path::Path,
    ) -> Result<(i32, Option<String>), PromoteReject> {
        use crate::scheduler::{LoraAck, LoraCommand};

        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let cmd = LoraCommand::PromoteDisk {
            name: name.to_string(),
            dir: dir.to_path_buf(),
        };
        if tx.send((cmd, ack_tx)).await.is_err() {
            return Err(PromoteReject::Peer(
                "scheduler promote channel closed".to_string(),
            ));
        }
        let acked = tokio::time::timeout(std::time::Duration::from_secs(30), ack_rx).await;
        match acked {
            Err(_timeout) => Err(PromoteReject::PoolFull(
                "disk promotion timed out waiting for scheduler quiescence; retry".to_string(),
            )),
            Ok(Err(_recv)) => Err(PromoteReject::Peer(
                "scheduler dropped the promote ack (shutting down?)".to_string(),
            )),
            Ok(Ok(Err(reason))) => {
                if reason.contains("POOL_FULL") || reason.contains("ref_count>0") {
                    Err(PromoteReject::PoolFull(reason))
                } else {
                    Err(PromoteReject::Peer(reason))
                }
            }
            Ok(Ok(Ok(LoraAck::Promoted { slot, evicted }))) => Ok((slot as i32, evicted)),
            Ok(Ok(Ok(LoraAck::Done))) => Err(PromoteReject::Peer(
                "scheduler returned a non-promote ack for a disk promote".to_string(),
            )),
        }
    }
}

pub type ModelBehavior = metrale_kernels::ModelBehavior;

#[cfg(test)]
mod tests {
    use super::{deadline_from, resolve_adapter_slot};

    #[test]
    fn zero_timeout_disables_the_deadline() {
        let now = std::time::Instant::now();
        assert!(deadline_from(0.0, now).is_none());
        assert!(deadline_from(-1.0, now).is_none());
    }

    #[test]
    fn positive_timeout_yields_that_budget() {
        let now = std::time::Instant::now();
        let d = deadline_from(5.0, now).expect("5s budget is a deadline");
        assert_eq!(d.saturating_duration_since(now).as_millis(), 5000);
    }

    #[test]
    fn adapter_slot_resolution_rules() {
        let names = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        assert_eq!(resolve_adapter_slot(&names, None), Some(-1));
        assert_eq!(resolve_adapter_slot(&names, Some("alpha")), Some(0));
        assert_eq!(resolve_adapter_slot(&names, Some("beta")), Some(1));
        assert_eq!(resolve_adapter_slot(&names, Some("gamma")), Some(2));
        assert_eq!(resolve_adapter_slot(&names, Some("delta")), None);
        assert_eq!(resolve_adapter_slot(&[], Some("alpha")), None);
        assert_eq!(resolve_adapter_slot(&[], None), Some(-1));
    }
}
