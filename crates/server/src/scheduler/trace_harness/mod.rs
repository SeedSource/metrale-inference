// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU trace harness for the scheduler loop: proves a re-layout keeps the model-call sequence and client-visible output byte-identical.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! A [`model::RecordingModel`] is a test-only `Model` that records its
//! calls (name and the arguments that shape scheduling) and answers from a
//! per-request token script. [`runner`] drives the real scheduler loop
//! (`crate::scheduler::run`) with it over the scenarios in [`scenarios`],
//! producing one text trace per scenario: the ordered model calls, then
//! the bytes each client received. [`tests`] compares each trace with its
//! golden file under `golden/`.
//!
//! The fake answers at once, and the scenarios avoid timing dependence:
//! deadlines have already passed when the scheduler checks them, and
//! cancel flags are flipped by the model at a scripted point.

mod decode_slice_tests;
mod model;
mod model_feed;
mod model_forward;
mod model_impl;
mod pipeline_tests;
mod runner;
mod scenarios;
mod scripted_tests;
mod telemetry_tests;
mod tests;
