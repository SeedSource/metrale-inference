// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `impl LayerWriteOnAccept for Glm5NextLayer`: the GDN write-on-accept and carry hooks
//! keep their defaults (a GLM layer never stashes or carries), and a KDA layer answers
//! `kda_defer_commit_ready` from its KDA block (`METRALE_GLM_KDA_DEFER_COMMIT=1`,
//! glm5next_kda/snap_fuse/defer.rs). Moved out of `mod.rs`, where it was the empty impl.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A DSA layer, the MTP block and every layer with the lever off answer false.

use super::{Glm5NextLayer, Glm5NextMixer};

impl super::LayerWriteOnAccept for Glm5NextLayer {
    fn kda_defer_commit_ready(&self) -> bool {
        match &self.mixer {
            Glm5NextMixer::Kda { layer, .. } => layer.defer_commit_ready(),
            Glm5NextMixer::Dsa(_) => false,
        }
    }
}
