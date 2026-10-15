// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the OpenAI-compatible API.
//!
//! - `harness`: whole-stream driver for the content sanitizer.
//! - `sanitizer`: orphan tool-call fragments are dropped and prose survives.
//! - `envelope`: tags inside a declared envelope pass through.
//! - `error_frames`: a `/v1/completions` stream error reaches the client as JSON.
//! - `watchdog`: the repetition guard fires on loops and not on varied prose.
//! - `guard_replay`: replays saved streams (fixtures from `GUARD_REPLAY_DIR`) through the loop guards.
//! - `health_fault`: readiness reports a GPU fault ahead of a loaded model.
//! - `model_advertise`: `/v1/models` reports the ceiling the chat path enforces.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

mod envelope;
mod error_frames;
mod guard_replay;
mod harness;
mod health_fault;
mod model_advertise;
mod sanitizer;
mod watchdog;
