// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The chat IR shared by the API surfaces: OpenAI chat requests
//! (`openai/to_ir.rs`) and Anthropic Messages requests (`anthropic/to_ir.rs`) convert
//! into `ChatRequest`, and `build_msg_entries` reads `ir::Message`.
//!
//! Owner: server (chat IR).
//! Invariants: none beyond the types.

pub mod message;
pub mod request;
pub mod response;
pub mod stream;

pub use message::{ContentPart, ImageData, MediaKind, Message, Role, VideoSource};
pub use request::{
    ChatRequest, EffortLevel, ReasoningEffort, ResponseFormat, SamplingParams, ThinkingDirective,
    parse_wire_effort,
};
pub use response::{
    ChatResponse, Choice, ChoiceLogprobs, FINISH_REASON_REPETITION, FINISH_REASON_TIMEOUT,
    FinishReason, TokenLogprob, Usage,
};
pub use stream::{DeltaStream, StreamDelta};

#[cfg(test)]
mod tests;
