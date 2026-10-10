// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: On-disk snapshot of the grammar compiler's rule-level
//! token-mask cache, so a new process starts with the masks an earlier one
//! computed.
//!
//! The rule-level cache keys a mask by the structure of the rule's FSM, not
//! by the schema text, so a loaded snapshot also serves schemas this process
//! has never compiled. The file name carries the tokenizer fingerprint, and
//! `mask_snapshot::load_from_file` treats a file whose header names another
//! format version, tokenizer fingerprint or vocabulary size as a miss.
//!
//! `METRALE_GRAMMAR_CACHE=0` (or `false`, `off`, `no`) turns persistence
//! off. `METRALE_GRAMMAR_CACHE_DIR` sets the directory; otherwise it is
//! `.metrale-grammar-cache` under the model directory.
//!
//! Owner: server (grammar).
//! Invariants:
//! - The snapshot is written only on a `grammar-mask-save` thread spawned by
//!   the prewarm hook, never on the thread that runs the hook.
//! - At most one save per engine runs at a time (the shared `writing` flag).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use metrale_grammar::compiler::{RuleLevelCache, SnapshotIdentity, mask_snapshot};

use super::engine::GrammarEngine;

/// 2026-09-26: Snapshot directory under the model directory, used when
/// `METRALE_GRAMMAR_CACHE_DIR` is unset or blank.
const CACHE_DIR_NAME: &str = ".metrale-grammar-cache";

/// 2026-09-26: Most masks one save writes: `save_to_file` keeps the
/// most-recently-used end of the cache's LRU order.
const MAX_PERSISTED_MASKS: usize = 512;

/// 2026-09-26: Everything the background saver needs, without a
/// reference to the compiler.
pub(super) struct MaskSnapshot {
    path: PathBuf,
    identity: SnapshotIdentity,
    cache: RuleLevelCache,
    /// 2026-09-26: Masks written by the last successful save (the import
    /// count until then). The hook saves only when the cache holds more.
    saved_entries: Arc<AtomicUsize>,
    /// 2026-09-26: Set while a save runs; a hook call that finds it set
    /// skips its save.
    writing: Arc<AtomicBool>,
    /// 2026-10-10: A177: latched by the first FAILED save (e.g. read-only
    /// directory) so later prewarm hooks stop re-encoding ~10 MB per request.
    failed: Arc<AtomicBool>,
    /// 2026-10-10: Save attempts that reached `save_to_file` (test observable).
    attempts: Arc<AtomicUsize>,
}

/// 2026-09-26: Called by the prewarm with the number of masks it warmed,
/// once the masks are in the cache. A trait object, so the prewarm needs no
/// knowledge of persistence.
pub type PrewarmHook = Arc<dyn Fn(usize) + Send + Sync>;

/// 2026-09-26: `METRALE_GRAMMAR_CACHE` value `0`, `false`, `off` or `no`
/// (trimmed, any case) disables the on-disk mask cache. Anything else,
/// including unset, enables it.
pub(super) fn cache_enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

/// 2026-09-26: `masks-<fingerprint as 16 hex digits>.bin` in the override
/// directory when it is non-blank, else in [`CACHE_DIR_NAME`] under
/// `model_dir`. The override serves a read-only model directory, or several
/// model copies that share one tokenizer.
pub(super) fn snapshot_path(
    model_dir: &Path,
    dir_override: Option<&str>,
    fingerprint: u64,
) -> PathBuf {
    let dir = match dir_override.map(str::trim).filter(|s| !s.is_empty()) {
        Some(d) => PathBuf::from(d),
        None => model_dir.join(CACHE_DIR_NAME),
    };
    dir.join(format!("masks-{fingerprint:016x}.bin"))
}

impl GrammarEngine {
    /// 2026-09-25: Load the rule-level mask cache from an earlier process's
    /// snapshot and arm `Self::mask_snapshot_hook`. Serving calls it once per
    /// model load (`serve_load::load_model`: at startup and on a model swap),
    /// never from a request.
    ///
    /// Returns nothing. A missing, mismatched or corrupt file imports no
    /// masks, and an unreadable one is logged as a warning; in every such
    /// case the saver is still armed. Returns without arming it when
    /// `METRALE_GRAMMAR_CACHE` disables persistence or the compiler has no
    /// rule cache.
    pub fn attach_mask_cache(&mut self, model_dir: &Path) {
        if !cache_enabled_from(std::env::var("METRALE_GRAMMAR_CACHE").ok().as_deref()) {
            tracing::info!("Grammar: on-disk mask cache disabled (METRALE_GRAMMAR_CACHE)");
            return;
        }
        let Some(cache) = self.compiler.rule_cache_handle() else {
            return;
        };
        let identity = self.compiler.snapshot_identity();
        let path = snapshot_path(
            model_dir,
            std::env::var("METRALE_GRAMMAR_CACHE_DIR").ok().as_deref(),
            identity.tokenizer_fingerprint,
        );
        let imported = match self.compiler.load_mask_snapshot(&path) {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(
                    "Grammar: mask snapshot unreadable at {}: {e}",
                    path.display()
                );
                0
            }
        };
        if imported > 0 {
            tracing::info!(
                "Grammar: warmed {imported} token masks from {} — the first tool-call \
                 request skips the cold mask compile (#918)",
                path.display(),
            );
        } else {
            tracing::info!(
                "Grammar: no usable mask snapshot at {}; the first grammar of this \
                 process will compute and persist one (#918)",
                path.display(),
            );
        }
        self.snapshot = Some(MaskSnapshot {
            path,
            identity,
            cache,
            saved_entries: Arc::new(AtomicUsize::new(imported)),
            writing: Arc::new(AtomicBool::new(false)),
            failed: Arc::new(AtomicBool::new(false)),
            attempts: Arc::new(AtomicUsize::new(0)),
        });
    }

    /// 2026-10-10: Save attempts so far, and whether a failure is latched.
    #[cfg(test)]
    pub(crate) fn snapshot_save_state(&self) -> Option<(usize, bool)> {
        let s = self.snapshot.as_ref()?;
        Some((
            s.attempts.load(Ordering::Relaxed),
            s.failed.load(Ordering::Relaxed),
        ))
    }

    /// 2026-09-26: The hook handed to a request's prewarm. When the cache
    /// holds more masks than the last save wrote and no save is running, it
    /// spawns a `grammar-mask-save` thread that writes the snapshot, so the
    /// caller never waits on the file. `None` until
    /// [`Self::attach_mask_cache`] has armed a snapshot.
    pub(crate) fn mask_snapshot_hook(&self) -> Option<PrewarmHook> {
        let snap = self.snapshot.as_ref()?;
        let (path, identity) = (snap.path.clone(), snap.identity);
        let cache = snap.cache.clone();
        let saved = Arc::clone(&snap.saved_entries);
        let writing = Arc::clone(&snap.writing);
        let failed = Arc::clone(&snap.failed);
        let attempts = Arc::clone(&snap.attempts);
        Some(Arc::new(move |_warmed: usize| {
            if failed.load(Ordering::Relaxed) {
                return;
            }
            if cache.len() <= saved.load(Ordering::Relaxed) {
                return;
            }
            if writing.swap(true, Ordering::AcqRel) {
                return;
            }
            let (path, cache, saved) = (path.clone(), cache.clone(), Arc::clone(&saved));
            let done = Arc::clone(&writing);
            let (failed, attempts) = (Arc::clone(&failed), Arc::clone(&attempts));
            let spawned = std::thread::Builder::new()
                .name("grammar-mask-save".to_string())
                .spawn(move || {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    let written =
                        mask_snapshot::save_to_file(&cache, identity, &path, MAX_PERSISTED_MASKS);
                    match written {
                        Ok(n) => {
                            saved.store(n, Ordering::Relaxed);
                            tracing::debug!(
                                "Grammar: persisted {n} token masks to {}",
                                path.display()
                            );
                        }
                        // 2026-09-26: A failed save is only logged; the
                        // server keeps running without a new snapshot.
                        // 2026-10-10: A177: latch the failure and warn once
                        // per process; later hooks skip the re-encode.
                        Err(e) => {
                            if !failed.swap(true, Ordering::AcqRel) {
                                static WARNED: AtomicBool = AtomicBool::new(false);
                                if !WARNED.swap(true, Ordering::AcqRel) {
                                    tracing::warn!(
                                        "Grammar: could not persist masks to {}: {e}; \
                                         not retrying this process",
                                        path.display()
                                    );
                                }
                            }
                        }
                    }
                    done.store(false, Ordering::Release);
                });
            if let Err(e) = spawned {
                tracing::debug!("Grammar: mask-snapshot writer not spawned: {e}");
                writing.store(false, Ordering::Release);
            }
        }))
    }
}
