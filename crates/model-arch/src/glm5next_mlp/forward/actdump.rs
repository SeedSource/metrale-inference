// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `METRALE_GLM_MOE_ACTDUMP=<dir>`: a measurement tool, not a serving path. After a
//! routed-MoE layer runs a prefill window, it appends that window's BF16 input rows
//! (`[rows, hidden]`, the post-norm hidden state the router and the routed experts read) to
//! `<dir>/L{layer}_x.bin` and the selected ids (`[rows, top_k]` I32) to `<dir>/L{layer}_ids.bin`.
//! Offline tools weight expert-format quantization error by real activations (race-memory #79,
//! wfmt gate 0b).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Off unless the variable is set; when off, `maybe_dump` returns before touching the GPU.
//! - It synchronizes the stream, so a run with it set is never a timing run.
//! - Windows below `MIN_ROWS` rows are skipped, so decode steps and CUDA graph capture never
//!   reach the synchronize.
//! - `METRALE_GLM_MOE_ACTDUMP_LAYERS` (comma-separated layer indices; default all) picks the
//!   layers. Each layer stops after `METRALE_GLM_MOE_ACTDUMP_MAX_ROWS` rows (default 65536).

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::Glm5NextMlpConfig;

/// 2026-10-11: The narrowest window dumped. The car's decode steps carry at most
/// `max_batch * (drafts + 1)` rows (12 at MB4, 48 at MB16); prefill windows are 256-8192 rows.
const MIN_ROWS: usize = 64;

struct Cfg {
    dir: String,
    layers: Option<Vec<usize>>,
    max_rows: usize,
}

fn cfg() -> Option<&'static Cfg> {
    static C: OnceLock<Option<Cfg>> = OnceLock::new();
    C.get_or_init(|| {
        let dir = std::env::var("METRALE_GLM_MOE_ACTDUMP")
            .ok()
            .filter(|d| !d.is_empty())?;
        let layers = std::env::var("METRALE_GLM_MOE_ACTDUMP_LAYERS")
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect());
        let max_rows = std::env::var("METRALE_GLM_MOE_ACTDUMP_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(65536);
        tracing::warn!(
            "METRALE_GLM_MOE_ACTDUMP={dir}: ENGAGED - routed-MoE input rows + ids dumped per \
             prefill window (layers {layers:?}, <= {max_rows} rows each); synchronizes, not a timing run"
        );
        Some(Cfg {
            dir,
            layers,
            max_rows,
        })
    })
    .as_ref()
}

/// 2026-10-11: Rows already written per layer.
fn written() -> &'static Mutex<HashMap<usize, usize>> {
    static W: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    W.get_or_init(|| Mutex::new(HashMap::new()))
}

fn append(path: &str, bytes: &[u8]) -> Result<()> {
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(bytes))
        .with_context(|| format!("METRALE_GLM_MOE_ACTDUMP: writing {path}"))
}

/// 2026-10-11: Dumps `x` (`[rows, hidden]` BF16) and `ws.ids` for `layer` when the lever is set,
/// the layer is selected, `rows >= MIN_ROWS` and the layer is under its row budget. Call it after
/// `forward_moe_sliced` on the same stream, with the same `x` and `rows`.
#[allow(clippy::too_many_arguments)]
pub fn maybe_dump(
    gpu: &dyn GpuBackend,
    cfg_mlp: &Glm5NextMlpConfig,
    ws: &Glm5NextMlpWorkspace,
    layer: usize,
    x: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let Some(c) = cfg() else { return Ok(()) };
    if rows < MIN_ROWS || c.layers.as_ref().is_some_and(|l| !l.contains(&layer)) {
        return Ok(());
    }
    let mut w = written().lock().expect("actdump lock");
    let done = w.entry(layer).or_insert(0);
    let n = rows.min(c.max_rows.saturating_sub(*done));
    if n == 0 {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let mut xb = vec![0u8; n * cfg_mlp.hidden * 2];
    let mut ib = vec![0u8; n * cfg_mlp.top_k * 4];
    gpu.copy_d2h(x, &mut xb)?;
    gpu.copy_d2h(ws.ids, &mut ib)?;
    append(&format!("{}/L{layer}_x.bin", c.dir), &xb)?;
    append(&format!("{}/L{layer}_ids.bin", c.dir), &ib)?;
    *done += n;
    Ok(())
}
