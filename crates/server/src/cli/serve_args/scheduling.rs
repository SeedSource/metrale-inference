// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The middle `met serve` flags, from `--disable-thinking` to
//! `--high-speed-swap-cache-blocks-per-seq`: generation, speculative decoding,
//! batching, scheduling, parallelism and swap. `ServeArgs` flattens this struct
//! last, and it flattens `ServeServiceArgs` last.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date. clap lists the flags in declaration order, so a flattened
//! struct's fields sit where its `#[command(flatten)]` field is: last.

use clap::Args;

// 2026-09-26: `ServeArgs` reaches these fields through `Deref`.
#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeSchedulingArgs {
    /// Global kill-switch for chain-of-thought / reasoning output.
    /// When set, the server forces thinking off regardless of what the
    /// client requests (reasoning_effort, thinking.budget_tokens, etc.)
    /// or what MODEL.toml declares as the default. Precedence (highest
    /// wins): this flag → request body → MODEL.toml `[behavior]`.thinking_default.
    ///
    /// Harry Potter alias: `--stupify` (stuns the model's inner monologue).
    #[arg(long, visible_alias = "stupify", default_value_t = false)]
    pub disable_thinking: bool,

    /// Override MODEL.toml's `[behavior].max_thinking_budget` (tokens).
    /// Sets the per-request thinking budget, and anchors the client
    /// `reasoning_effort` ladder (minimal/low/medium/high/xhigh = 1/4x, 1/2x, 1x,
    /// 2x, 4x of this value, unless MODEL.toml sets `effort_capped_at_ceiling`).
    /// An explicit client token budget (`thinking.budget_tokens`,
    /// `thinking_token_budget`) still wins; the budget is capped at 90% of
    /// max_tokens unless MODEL.toml sets `cap_thinking_at_max_tokens = false`.
    /// For a model with `effort_prompt_only` (GLM-5.3, no budget by default),
    /// giving this flag brings the budget ladder back.
    /// Refused together with `--disable-thinking`.
    #[arg(long)]
    pub max_thinking_budget: Option<u32>,

    /// 2026-10-14: Reply limit (tokens) of a request that sends no
    /// `max_tokens` / `max_completion_tokens` / `max_output_tokens`. Admission
    /// reserves prompt + min(this, max_seq_len) for such a request.
    #[arg(long, default_value_t = 4096)]
    pub default_max_tokens: usize,

    /// Override MODEL.toml's `[behavior].max_inter_tool_prose` (tokens):
    /// the cap on free-prose tokens between successive tool calls on a
    /// tool request. Past it the scheduler rolls back to the last boundary and
    /// re-steers, or ends the response when it cannot roll back. 0 disables the
    /// guard.
    /// Precedence (highest wins): this flag → METRALE_MAX_INTER_TOOL_PROSE
    /// → MODEL.toml → built-in default (3072).
    #[arg(long)]
    pub max_inter_tool_prose: Option<u32>,

    /// Content-loop watchdog: `auto` (default), `on` or `off`. `on`/`off`
    /// override MODEL.toml's `[behavior].enable_loop_watchdog` either way;
    /// `auto` leaves it to METRALE_CONTENT_LOOP_WATCHDOG, then MODEL.toml
    /// (which arms it for some models and not others, hence an enum rather
    /// than a presence flag). The watchdog ends (or rolls back) a response
    /// whose tail is a short-period token repeat; its built-in threshold (3
    /// end-anchored repeats of a period-2..64 pattern) can also match
    /// legitimately repetitive output. Toggle it while serving from the
    /// dashboard with `/watchdog on|off`.
    #[arg(long, default_value = "auto", value_name = "auto|on|off")]
    pub content_loop_watchdog: String,

    /// Override the content-loop watchdog's repeat threshold (end-anchored
    /// consecutive repeats that constitute a loop; built-in default 3).
    /// A per-request `repetition_detection` object still outranks this.
    /// Precedence: this flag → METRALE_CONTENT_LOOP_MIN_REPEATS → built-in
    /// default.
    #[arg(long)]
    pub content_loop_min_repeats: Option<u32>,

    /// XGrammar structural-tag enforcement on tool requests that do not require
    /// a tool call: `auto` (default: MODEL.toml's
    /// `[behavior].disable_tool_grammar` decides), `on` or `off`. `off` skips the
    /// grammar, and tool calls are parsed from the output afterwards;
    /// `tool_choice="required"`, a named tool and the `minimax_xml` parser keep
    /// the grammar. An enum because MODEL.toml turns the grammar off for some
    /// models, so the command line must be able to pin it either way.
    #[arg(long, default_value = "auto", value_name = "auto|on|off")]
    pub tool_grammar: String,

    /// Default chat template kwargs applied when the client sends no
    /// thinking parameters (no `reasoning.effort`, `chat_template_kwargs`,
    /// or `enable_thinking` in the request body). A JSON object with
    /// optional keys: `enable_thinking` (bool), `thinking_budget` (u32),
    /// `reasoning_effort` (none|minimal|low|medium|high|xhigh|max — the
    /// served default effort tier when clients are silent; unset = the
    /// neutral "medium"), `preserve_thinking` (bool). Invalid JSON, unknown
    /// keys or an unknown `reasoning_effort` value abort startup.
    ///
    /// Precedence (highest wins): request body → this flag → MODEL.toml.
    /// Example: `--default-chat-template-kwargs '{"reasoning_effort":"xhigh"}'`
    #[arg(long, value_name = "JSON")]
    pub default_chat_template_kwargs: Option<String>,

    /// Ignore the `jinja-templates/` override directory and render every
    /// model with its own chat template (from the checkpoint; ChatML when it
    /// has none). Default off: an override file's presence is the opt-in
    /// signal for its model type (see `jinja-templates/README.md`).
    #[arg(long, default_value_t = false)]
    pub disable_template_overrides: bool,

    /// Enable MTP speculative decoding. Unless `--mtp-gate force` is given, the
    /// MTP gate (`metrale_speculative::mtp_gate`) measures delivered throughput
    /// with and without MTP while serving and switches to plain decode while MTP
    /// is the slower of the two.
    #[arg(long, default_value_t = false)]
    pub speculative: bool,

    /// Enable self-speculative decoding: draft by skipping the SSM layers (no MTP weights
    /// needed), then verify with the full model.
    #[arg(long, default_value_t = false)]
    pub self_speculative: bool,

    /// Enable N-gram speculative decoding: a CPU-side pattern-matching proposer.
    /// No extra weights needed.
    #[arg(long, default_value_t = false)]
    pub ngram_speculative: bool,

    /// Enable DFlash block-diffusion speculative decoding (arXiv 2602.06036).
    /// Pairs the target with a small drafter checkpoint (e.g.
    /// `z-lab/Qwen3.6-35B-A3B-DFlash`) that drafts a block of γ tokens per step
    /// from captured target hidden states. Mutually exclusive with
    /// `--speculative`.
    #[arg(long, default_value_t = false, conflicts_with = "speculative")]
    pub dflash: bool,

    /// HuggingFace id (or local path) of the DFlash drafter checkpoint.
    /// When `--dflash` is set without `--draft-model`, the value falls
    /// through from the target's MODEL.toml `[dflash].draft_model` field.
    #[arg(long)]
    pub draft_model: Option<String>,

    /// DFlash block size γ (draft tokens per step).
    ///
    /// Unset (the default): derived from the drafter's trained block size
    /// (`dflash_config.block_size`, else its top-level `block_size`) as block
    /// + 2, capped at 15 (`default_dflash_gamma`). An explicit value overrides
    /// that, for ablation.
    #[arg(long)]
    pub dflash_gamma: Option<usize>,

    /// DFlash drafter sliding-window size, in tokens. Set to 0 to disable the
    /// window (full-prefix attention).
    #[arg(long, default_value_t = 4096)]
    pub dflash_window_size: usize,

    /// Number of draft tokens per speculative step (1=K=2, 2=K=3, 3=K=4 verify).
    /// Precedence (highest wins): this flag → MODEL.toml
    /// `[behavior].default_num_drafts` → 1 (`DEFAULT_NUM_DRAFTS`). An
    /// explicitly passed value always wins, including `--num-drafts 1` on a
    /// model whose MODEL.toml defaults higher. Ignored under `--dflash`, which
    /// drafts γ - 1.
    #[arg(long)]
    pub num_drafts: Option<usize>,

    /// Maximum concurrent sequences batched into one GPU decode step.
    #[arg(long, default_value_t = 8)]
    pub max_batch_size: usize,

    /// MTP head weight precision: bf16 (default), fp8 or nvfp4.
    #[arg(long, default_value = "bf16")]
    pub mtp_quantization: String,

    /// MTP draft vocabulary size: the draft head's LM-head GEMV covers only the
    /// first N token IDs. Set to 0 to use the full vocabulary.
    #[arg(long, default_value_t = 100000)]
    pub mtp_vocab: u32,

    /// Enable prefix caching via radix tree (RadixAttention).
    /// Caches KV blocks for recurring prompt prefixes. Turned off, with a
    /// warning, on a model whose per-sequence state is not all in the KV cache.
    /// SSM state snapshots for cache hits: `--ssm-cache-slots`.
    #[arg(long, default_value_t = false)]
    pub enable_prefix_caching: bool,

    /// Measure this server as a known-answer test: no state produced while
    /// serving one request may reach another.
    ///
    /// One name for the whole regime, expanded in code by `cli::hermetic`, which
    /// closes the prefix cache and the MTP gate's probes; SSM snapshot restores
    /// are limited to the request's own session. It is the name that lands in a
    /// gate record's `serve_overrides`.
    ///
    /// It overrides rather than merges: `--hermetic` beside
    /// `--enable-prefix-caching` or `--mtp-gate auto` is a contradiction, and
    /// `validate_serve_args` refuses the pair.
    ///
    /// Not a production setting.
    #[arg(long, default_value_t = false)]
    pub hermetic: bool,

    /// Dump every /v1/chat/completions and /v1/messages (Anthropic)
    /// request, plus the corresponding response (non-streaming) or
    /// aggregated stream, as JSONL to a file. Intended for extracting the
    /// exact system prompt and tool schema a client (opencode, Claude Code,
    /// etc.) is sending, and for replaying failure cases in fixtures.
    ///
    /// `--dump auto`: a file is created in the temp directory ($TMPDIR) and
    /// its path is logged at INFO on startup. `--dump PATH`: appends (never
    /// truncates) to that file (spell a file literally named `auto` as
    /// `./auto`). Each line is one JSON object:
    ///   `{ "ts": "<iso8601>", "endpoint": "...", "kind": "request"|"response",`
    ///     "seq": N, "body": { ... } }
    /// so entries can be grouped by `seq` to reconstruct pairs.
    #[arg(long, value_name = "PATH|auto")]
    pub dump: Option<String>,

    /// Scheduler policy: fifo (default) or slai (deadline-aware). SLAI skips new
    /// prefills while any active sequence has waited 80% of `--tbt-deadline-ms`
    /// since its last token, and prefills the front of the queue first, then the
    /// shortest prompts. This is the admission/ordering policy; the execution
    /// router is `--scheduler-config`.
    #[arg(long, default_value = "fifo")]
    pub scheduler: String,

    /// Scheduler execution router: sync (default) runs every device step
    /// to completion before the next host decision; async runs one plain
    /// decode step ahead of the host on a model with a device token feed
    /// (falls back to sync, with a warning, on a model without one).
    #[arg(long, default_value = "sync")]
    pub scheduler_config: String,

    /// Telemetry level: off (default) measures nothing and costs one branch
    /// per instrument; basic samples the GPU at 10 Hz and records scheduler,
    /// cache, speculation, request and J/token instruments; kernel adds
    /// CUDA-event GPU spans for every serve phase, and for every kernel on
    /// sampled steps. Exported on /metrics and streamed on /v1/events.
    #[arg(long, default_value = "off", value_name = "LEVEL")]
    pub telemetry: String,

    /// TBT deadline in milliseconds for the SLAI scheduling policy.
    /// Sequences approaching this deadline trigger decode-first priority.
    #[arg(long, default_value_t = 100)]
    pub tbt_deadline_ms: u64,

    /// Maximum tokens to prefill per chunk (chunked prefill). Long prompts are
    /// split into chunks of this size, interleaved with decode steps for active
    /// sequences.
    /// At 8192 (the default, or passed explicitly), a model with its own SSM
    /// prefill chunk size uses that size instead. 0 asks for no chunking: one
    /// chunk of --max-seq-len, unless the model has an SSM prefill chunk size.
    /// Chunks are capped below the CUDA grid limit (65535 tokens, block-aligned).
    #[arg(long, default_value_t = 8192)]
    pub max_prefill_tokens: usize,

    /// Minimum free GPU memory (in MB) to keep as a safety margin during
    /// model loading. If free memory drops below this threshold after any
    /// shard, loading is aborted.
    #[arg(long, default_value_t = 4096)]
    pub oom_guard_mb: usize,

    /// Global rank (0=head, 1=worker, …). Only used when --world-size > 1.
    #[arg(long, default_value_t = 0)]
    pub rank: usize,

    /// Total physical ranks across all parallelism dims. Set to 2 for two-node
    /// deployment. Must satisfy `world_size == tp_size × ep_size` (orthogonal
    /// mesh) or `world_size == tp_size == ep_size` (overlapping groups on the
    /// same physical ranks, used for 2-GPU TP+EP composition). Left at 1 beside a
    /// larger `--tp-size` or `--ep-size`, it is derived from them.
    #[arg(long, default_value_t = 1)]
    pub world_size: usize,

    /// Tensor-parallel dimension: weights are split across `tp_size` ranks.
    /// 1 = no TP.
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Expert-parallel dimension. Splits MoE expert weights across `ep_size`
    /// ranks. 1 = no EP.
    #[arg(long, default_value_t = 1)]
    pub ep_size: usize,

    /// NCCL bootstrap address (IP of rank 0 node).
    #[arg(long, default_value = "127.0.0.1")]
    pub master_addr: String,

    /// NCCL bootstrap port.
    #[arg(long, default_value_t = 29500)]
    pub master_port: u16,

    /// Tool call parser format. Enables OpenAI-compatible tool calling.
    /// Supported: "hermes", "qwen3_coder", "qwen3_xml", "gemma4", "mistral",
    /// "minimax_xml", "bare_json", "poolside_v1". See the `FromStr for
    /// ToolCallFormat` in tool_parser.rs.
    /// Unset: MODEL.toml `[behavior].tool_call_parser`, else the mapping for the
    /// model's `model_type` in `tool_defaults.toml`, else tool calling is off.
    #[arg(long, value_name = "FORMAT")]
    pub tool_call_parser: Option<String>,

    /// Maximum output tokens per tool-calling request: the client's max_tokens is
    /// capped at this value when tools are active. Must be high enough for Write
    /// tool calls with large file content.
    #[arg(long, default_value_t = 8192)]
    pub tool_max_tokens: usize,

    /// Number of SSM state snapshot slots for Marconi prefix caching.
    /// Each slot stores SSM h_state + conv_state for all SSM layers, so a
    /// prefix cache hit can restore SSM state as well as KV.
    /// 0 = disabled. The slots are not reserved while the prefix cache is
    /// inactive, unless METRALE_SSM_MARCONI_FULL is set. Intermediate checkpoints
    /// (--ssm-checkpoint-interval) use slots too.
    #[arg(long, default_value_t = 16)]
    pub ssm_cache_slots: usize,

    /// Save SSM state snapshots at regular block boundaries during prefill.
    /// When set to N > 0, a snapshot is saved at every chunked-prefill chunk
    /// boundary whose block index is a multiple of N. On future prefix cache
    /// hits, the deepest intermediate snapshot is restored, reducing SSM
    /// recomputation to the tokens between the checkpoint and the match point.
    /// Independent of this interval, a tail checkpoint is always saved at the
    /// prompt's last full-block boundary; warm multi-turn restores hit the tail
    /// checkpoint. Chunk size is never reduced to serve this interval.
    /// 0 = tail snapshots only. 256 = every 4096 tokens (block_size=16).
    #[arg(long, default_value_t = 256)]
    pub ssm_checkpoint_interval: usize,

    /// Minimum matched tokens before an SSM snapshot is restored rather than
    /// recomputed. 0 removes the floor; raise it very high to disable restore.
    ///
    /// Outside `--hermetic`, only tail snapshots are checked against the
    /// request's session, so a restore can use state another request computed.
    /// A known-answer gate can disable restore with this flag, and as a recipe
    /// key the setting is carried in its record.
    #[arg(long, default_value_t = metrale_model_layers::DEFAULT_MARCONI_MIN_TOKENS)]
    pub marconi_min_tokens: usize,

    /// Enable automatic context compaction for long conversations (off by
    /// default). Without it, a prompt that does not fit in --max-seq-len gets a
    /// 400 error (`Prompt too long`).
    ///
    /// With a value above 0, a chat request with more than 4 messages whose
    /// rendered prompt exceeds 70% of --max-seq-len has its older messages
    /// truncated (`api/compact.rs`) before tokenization. The value is required,
    /// but only whether it is above 0 is read: the 70% trigger is fixed.
    #[arg(long, value_name = "THRESHOLD")]
    pub auto_compact: Option<f32>,

    /// Default top-n-sigma for sampling, used when neither the request nor the
    /// model's sampling preset sets one: logits below mean - n·σ are masked before
    /// temperature is applied. 0.0 = disabled. Top-n-σ: arXiv:2411.07641.
    #[arg(long, default_value_t = 1.0)]
    pub default_top_n_sigma: f32,

    /// Default min-p for sampling, used when the request sets none and the model's
    /// sampling preset supplies none (a preset counts only for a model that applies
    /// it to the core samplers): keep tokens with prob >= min_p * max_prob.
    /// 0.0 = disabled.
    #[arg(long, default_value_t = 0.08)]
    pub default_min_p: f32,

    /// Swap space in GB for KV cache overflow to disk. When GPU blocks are
    /// exhausted, sequences are swapped to disk and resumed later.
    /// 0 = disabled. Swap files are stored in `$TMPDIR/metrale-swap-<pid>/`.
    /// Turned off, with a warning, on a model whose per-sequence state is not all
    /// in the KV cache.
    #[arg(long, default_value_t = 3)]
    pub swap_space_gb: usize,

    // 2026-09-26: `--high-speed-swap` has no help text. It streams KV blocks of a
    // sequence to and from disk (metrale-storage), while `--swap-space-gb` evicts and
    // restores whole sequences. `serve_phases::build_high_speed_swap_config` builds its
    // config and fills an unset `--high-speed-swap-dir`, `-gb` or `-resident-blocks`
    // with /var/tmp/metrale-hsw, 64 GiB or 8192 blocks.
    #[arg(long, default_value_t = false)]
    pub high_speed_swap: bool,

    /// Directory for the --high-speed-swap KV files. Unset: /var/tmp/metrale-hsw.
    #[arg(long)]
    pub high_speed_swap_dir: Option<std::path::PathBuf>,

    /// Total disk budget for --high-speed-swap, in GiB. Unset: 64.
    #[arg(long)]
    pub high_speed_swap_gb: Option<u64>,

    /// HBM scratch slot count (number of resident blocks). Unset: 8192.
    #[arg(long)]
    pub high_speed_swap_resident_blocks: Option<u32>,

    /// Predictor low-rank dimension, in 1..=128.
    #[arg(long, default_value_t = 32)]
    pub high_speed_swap_rank: u32,

    /// io_uring submission queue depth, in 1..=64.
    #[arg(long, default_value_t = 8)]
    pub high_speed_swap_qd: u32,

    /// Turn off the CUDA-graph capture and replay of the `--high-speed-swap`
    /// per-layer body, which is on whenever `--high-speed-swap` is.
    #[arg(long)]
    pub no_high_speed_swap_graph: bool,

    /// Per-sequence HBM cache cap for `--high-speed-swap`.
    /// When set together with --high-speed-swap, each sequence is limited
    /// to N HBM-resident KV blocks; older blocks are evicted to disk and
    /// streamed back on demand. The prefill chunk is capped at
    /// N × --block-size − --max-batch-size tokens. Set to
    /// max_seq_len/block_size to disable HBM-shrink (no eviction; useful
    /// for diff-against-no-swap correctness checks).
    #[arg(long, default_value_t = 64)]
    pub high_speed_swap_cache_blocks_per_seq: u32,

    // 2026-09-26: The last flags, in declaration order; `Deref`/`DerefMut` below reach
    // them as `args.<field>`.
    #[command(flatten)]
    pub(crate) service: super::service::ServeServiceArgs,
}

impl std::ops::Deref for ServeSchedulingArgs {
    type Target = super::service::ServeServiceArgs;

    fn deref(&self) -> &Self::Target {
        &self.service
    }
}

impl std::ops::DerefMut for ServeSchedulingArgs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.service
    }
}
