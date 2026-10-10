// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: End a rank whose communicator went unhealthy.
//!
//! Owner: server startup (`met serve`).
//! In a serve an unhealthy `CommBackend` is terminal: `attempt_reconnect` needs every rank at
//! once and nothing calls it, and every later collective refuses (`ensure_healthy`). Before this,
//! the process stayed up with a dead communicator: on 2026-10-10 (ladg2m, MB=16) rank 1 left its
//! EP worker loop, rank 0 kept answering requests with errors, and n1's GPU spun for 35 minutes.
//! [`spawn`] starts one thread per multi-rank serve that polls `CommBackend::is_healthy` every
//! [`POLL`]; on the first `false` it logs one FATAL line and exits the process with
//! [`EXIT_CODE`], so the container stops and the harness sees a dead serve.
//! Invariants: [`watch`] calls `on_fatal` at most once, and only after `healthy` returned false.

use std::sync::Arc;
use std::time::Duration;

/// 2026-10-10: How often the watcher asks the communicator.
pub(crate) const POLL: Duration = Duration::from_secs(1);

/// 2026-10-10: Process exit code after a poisoned communicator (EX_TEMPFAIL).
pub(crate) const EXIT_CODE: i32 = 75;

/// 2026-10-10: Poll `healthy` every `poll` until it returns false, then call `on_fatal` with the
/// FATAL message and return. `on_fatal` exits the process in a serve; tests pass a closure.
pub(crate) fn watch(
    rank: usize,
    world: usize,
    poll: Duration,
    healthy: impl Fn() -> bool,
    on_fatal: impl FnOnce(String),
) {
    while healthy() {
        std::thread::sleep(poll);
    }
    on_fatal(format!(
        "FATAL: NCCL communicator unhealthy (rank {rank}/{world}): a collective failed or timed \
         out and the communicator is poisoned; exiting with code {EXIT_CODE} so every rank stops"
    ));
}

/// 2026-10-10: Start the watcher for a multi-rank serve (no thread for a single rank).
pub(crate) fn spawn(comm: &Option<Arc<dyn metrale_comm::CommBackend>>) {
    let Some(comm) = comm.clone() else { return };
    if comm.world_size() < 2 {
        return;
    }
    let (rank, world) = (comm.rank(), comm.world_size());
    let spawned = std::thread::Builder::new()
        .name("comm-watch".into())
        .spawn(move || {
            watch(
                rank,
                world,
                POLL,
                || comm.is_healthy(),
                |msg| {
                    tracing::error!("{msg}");
                    eprintln!("{msg}");
                    std::process::exit(EXIT_CODE);
                },
            )
        });
    if let Err(e) = spawned {
        tracing::warn!("comm watcher not started ({e}); a poisoned communicator will not exit");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// 2026-10-10: An injected poison (the flag `poison_on_error` sets) ends the watch with one
    /// FATAL message naming the rank; a healthy communicator never calls `on_fatal`.
    #[test]
    fn injected_poison_ends_the_watch_once_with_a_fatal_line() {
        let unhealthy = Arc::new(AtomicBool::new(false));
        let polls = Arc::new(AtomicUsize::new(0));
        let (u, p) = (unhealthy.clone(), polls.clone());
        let t = std::thread::spawn(move || {
            let mut got = Vec::new();
            watch(
                1,
                2,
                Duration::from_millis(1),
                || {
                    if p.fetch_add(1, Ordering::Relaxed) == 5 {
                        u.store(true, Ordering::Release);
                    }
                    !u.load(Ordering::Acquire)
                },
                |m| got.push(m),
            );
            got
        });
        let got = t.join().unwrap();
        assert_eq!(got.len(), 1);
        assert!(got[0].starts_with("FATAL: NCCL communicator unhealthy (rank 1/2)"));
        assert!(got[0].contains(&format!("code {EXIT_CODE}")));
        assert!(polls.load(Ordering::Relaxed) >= 6);
    }

    /// 2026-10-10: `serve_load` starts the watcher right after the communicator exists.
    #[test]
    fn serve_load_spawns_the_watcher_after_init() {
        let s = include_str!("../serve_load.rs");
        let init = s.find("serve_phases::init_nccl_comm(").expect("comm init");
        let watch = s
            .find("serve_phases::spawn_comm_watch(&comm);")
            .expect("watcher spawn");
        assert!(init < watch && watch - init < 400);
    }
}
