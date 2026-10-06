// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Fixture of `dsa_pool_cache_parity_microtest`: one GLM-5.3 DSA indexer cache
//! kept twice with the same random rows. Side A is today's path (`METRALE_GLM_DSA_POOL_CACHE`
//! off): full-length `k`/`gate`/`valid` and a full `dsa_kpool_compress` into the select scratch
//! on every pass. Side B is a `Glm5NextDsaState` with the pool cache: an 8,448-row ring,
//! persistent pool arrays, `dsa_kpool_compress_incr` from the watermark, and its scratch planned
//! without pool regions. Every write and select mirrors a production caller, cited per
//! function, and every select is followed by a byte compare of the pools over
//! `[0, floor(S / 4))` and the selected token ids.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_dsa::select::{
    DsaSelectGeometry, DsaSelectInputs, DsaSelectLaunch, DsaSelectScratch, PoolRows,
    compress_pools_only, select_tokens,
};
use metrale_model_arch::glm5next_dsa::state::{Glm5NextDsaState, max_dsa_context};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

pub(crate) const H: usize = 32;
pub(crate) const D: usize = 128;
pub(crate) const KP: usize = 4;
const TOPK: usize = 2048;
/// 2026-10-06: `--max-seq-len` of the production 512K row.
pub(crate) const MAX_SEQ: usize = 540_672;
/// 2026-10-06: Query rows of the widest pass: a 256-row prefill sub-chunk (`wide.rs`).
pub(crate) const QMAX: usize = 256;
/// 2026-10-06: Rows of one verify step at K = 3.
pub(crate) const VROWS: usize = 3;

pub(crate) struct Lcg(pub(crate) u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn bf16_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                bf16::from_f32(lo + (hi - lo) * self.f())
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }
    fn f32_bytes(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (lo + (hi - lo) * self.f()).to_le_bytes())
            .collect()
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn i32s(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    v.into_iter()
        .flat_map(|x| (x as i32).to_le_bytes())
        .collect()
}

pub(crate) fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: H,
        index_head_dim: D,
        index_kpool: KP,
        index_topk: TOPK,
        always_select_tail: true,
        local_heads: 32,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: MAX_SEQ,
    }
}

/// 2026-10-06: Per verify-row device vectors of the graph step: staged row, position, `seq_len`,
/// query position and each side's geometry (A five slots, B six).
#[derive(Clone, Copy)]
struct GRow {
    stage_k: DevicePtr,
    stage_g: DevicePtr,
    pos: DevicePtr,
    sl: DevicePtr,
    q_pos: DevicePtr,
    geom_a: DevicePtr,
    geom_b: DevicePtr,
}

#[path = "pool_cache_parity_graph.rs"]
mod graph;

pub(crate) struct Pair<'g> {
    g: &'g dyn GpuBackend,
    pub(crate) s: u64,
    pub(crate) cfg: Glm5NextDsaConfig,
    k: Glm5NextDsaKernels,
    a_k: DevicePtr,
    a_g: DevicePtr,
    a_valid: DevicePtr,
    pub(crate) a_len: usize,
    pub(crate) b: Glm5NextDsaState,
    a_scr: DsaSelectScratch,
    b_scr: DsaSelectScratch,
    ape: DevicePtr,
    q: DevicePtr,
    w: DevicePtr,
    q_pos: DevicePtr,
    q_mask: DevicePtr,
    rows: [GRow; VROWS],
    graphs: Option<(GraphHandle, GraphHandle)>,
    rng: Lcg,
    /// 2026-10-06: Selects compared, and those that differed (with the first one's label).
    pub(crate) compared: usize,
    pub(crate) differed: usize,
    pub(crate) first_diff: Option<String>,
}

impl<'g> Pair<'g> {
    pub(crate) fn new(g: &'g dyn GpuBackend, k: Glm5NextDsaKernels, seed: u64) -> Result<Self> {
        let cfg = cfg();
        cfg.validate()?;
        let cap = max_dsa_context(&cfg);
        let mut rng = Lcg(seed);
        let plan = DsaSelectGeometry::plan(&cfg, cap, QMAX)?;
        let rows = std::array::from_fn(|_| GRow {
            stage_k: DevicePtr::NULL,
            stage_g: DevicePtr::NULL,
            pos: DevicePtr::NULL,
            sl: DevicePtr::NULL,
            q_pos: DevicePtr::NULL,
            geom_a: DevicePtr::NULL,
            geom_b: DevicePtr::NULL,
        });
        let mut p = Self {
            g,
            s: g.create_stream()?,
            cfg,
            k,
            a_k: g.alloc(cap * D * 2)?,
            a_g: g.alloc(cap * D * 2)?,
            a_valid: up(g, &vec![0u8; cap])?,
            a_len: 0,
            b: Glm5NextDsaState::alloc_pool_cache(g, &cfg, 0, None)?,
            a_scr: DsaSelectScratch::alloc_sized(
                g,
                DsaSelectScratch::plan_bytes_with(&cfg, &[plan], false),
            )?,
            b_scr: DsaSelectScratch::alloc_sized(
                g,
                DsaSelectScratch::plan_bytes_with(&cfg, &[plan], true),
            )?,
            ape: up(g, &rng.f32_bytes(KP * D, -0.5, 0.5))?,
            q: up(g, &rng.f32_bytes(QMAX * H * D, -1.0, 1.0))?,
            w: up(g, &rng.f32_bytes(QMAX * H, 0.0, 0.3))?,
            q_pos: g.alloc(QMAX * 4)?,
            q_mask: up(g, &[1u8; QMAX])?,
            rows,
            graphs: None,
            rng,
            compared: 0,
            differed: 0,
            first_diff: None,
        };
        for r in 0..VROWS {
            p.rows[r] = GRow {
                stage_k: g.alloc(D * 2)?,
                stage_g: g.alloc(D * 2)?,
                pos: g.alloc(4)?,
                sl: g.alloc(4)?,
                q_pos: g.alloc(4)?,
                geom_a: g.alloc(5 * 4)?,
                geom_b: g.alloc(6 * 4)?,
            };
        }
        Ok(p)
    }

    fn sync(&self) -> Result<()> {
        self.g.synchronize(self.s)
    }

    /// 2026-10-06: `n` random rows: `[n, D]` BF16 keys and gates.
    fn rows_random(&mut self, n: usize) -> (Vec<u8>, Vec<u8>) {
        (
            self.rng.bf16_bytes(n * D, -1.0, 1.0),
            self.rng.bf16_bytes(n * D, -2.0, 2.0),
        )
    }

    /// 2026-10-06: B's host-path write preamble (`indexer_forward`, `write_kv_rows`,
    /// `decode_k_wide`): `ensure_room(n)`, then `dsa_pk_len_clamp` when a rewind left the
    /// device watermark above the first written pool (`pool_clamp_before_write`).
    fn b_begin(&mut self, n: usize) -> Result<usize> {
        self.b.ensure_room(n)?;
        let pos0 = self.b.len();
        if let Some((p, pools)) = self.b.take_dev_clamp(pos0) {
            KernelLaunch::new(self.g, self.k.pk_len_clamp)
                .grid([1, 1, 1])
                .block([1, 1, 1])
                .arg_ptr(p)
                .arg_i32(pools as i32)
                .launch(self.s)?;
        }
        Ok(pos0)
    }

    /// 2026-10-06: Put rows `[pos0, pos0 + n)` on both sides (A at absolute rows, B at ring
    /// slots split at the wrap) and mark them valid; neither cursor moves.
    fn place(&mut self, pos0: usize, kb: &[u8], gb: &[u8]) -> Result<()> {
        let (g, n, row) = (self.g, kb.len() / (D * 2), D * 2);
        self.sync()?;
        g.copy_h2d(kb, self.a_k.offset(pos0 * row))?;
        g.copy_h2d(gb, self.a_g.offset(pos0 * row))?;
        g.copy_h2d(&vec![1u8; n], self.a_valid.offset(pos0))?;
        for (r0, m) in self.b.ring_runs(pos0, n) {
            let off = self.b.row_offset(pos0 + r0);
            g.copy_h2d(&kb[r0 * row..(r0 + m) * row], self.b.k_normed.offset(off))?;
            g.copy_h2d(&gb[r0 * row..(r0 + m) * row], self.b.gate.offset(off))?;
        }
        g.copy_h2d(&vec![1u8; n], self.b.valid.offset(pos0))
    }

    /// 2026-10-06: `n` rows written and published at once (decode / verify rows on the host
    /// path, one drafter tile).
    pub(crate) fn write(&mut self, n: usize) -> Result<()> {
        let pos0 = self.b_begin(n)?;
        let (kb, gb) = self.rows_random(n);
        self.place(pos0, &kb, &gb)?;
        self.b.advance(n)?;
        self.a_len += n;
        Ok(())
    }

    /// 2026-10-06: Select at the current length over the last `q_rows` positions, exact
    /// launch, on both sides (`select_row_at` / `select_rows_batched` / `decode_k_wide`), then
    /// compare.
    pub(crate) fn select(&mut self, q_rows: usize, label: &str) -> Result<bool> {
        let (g, cfg, s) = (self.g, self.cfg, self.a_len);
        self.sync()?;
        g.copy_h2d(&i32s(s - q_rows..s), self.q_pos)?;
        let base = DsaSelectInputs {
            k_normed: self.a_k,
            gate: self.a_g,
            valid: self.a_valid,
            ape: self.ape,
            q: self.q,
            weights: self.w,
            q_pos: self.q_pos,
            q_mask: self.q_mask,
            first_key: 0,
            geom_dev: DevicePtr::NULL,
            pool_cache: None,
        };
        let geom = DsaSelectGeometry::plan(&cfg, s, q_rows)?;
        let ex = DsaSelectLaunch::Exact;
        select_tokens(g, &self.k, &cfg, &geom, &base, &self.a_scr, ex, self.s)?;
        let b_in = DsaSelectInputs {
            k_normed: self.b.k_normed,
            gate: self.b.gate,
            valid: self.b.valid,
            pool_cache: self.b.pool_select_args()?,
            ..base
        };
        let geom = self.b.geometry(&cfg, q_rows)?;
        select_tokens(g, &self.k, &cfg, &geom, &b_in, &self.b_scr, ex, self.s)?;
        self.b.note_selected();
        self.compare(s, &[(0, q_rows)], label)
    }

    /// 2026-10-06: Byte-compare A's scratch pools with B's persistent pools over
    /// `[0, floor(s / 4))`, and the token rows `(first, count)` of both scratches. Tallies the
    /// result; returns whether all were equal.
    pub(crate) fn compare(
        &mut self,
        s: usize,
        rows: &[(usize, usize)],
        label: &str,
    ) -> Result<bool> {
        self.sync()?;
        let g = self.g;
        let down = |p: DevicePtr, n: usize| -> Result<Vec<u8>> {
            let mut v = vec![0u8; n];
            if n > 0 {
                g.copy_d2h(p, &mut v)?;
            }
            Ok(v)
        };
        let pools = s / KP;
        let pc = self.b.pool_cache().expect("pool cache");
        let [ak, ai, av] = self.a_scr.pool_regions();
        let mut eq = down(ak, pools * D * 4)? == down(pc.pk, pools * D * 4)?
            && down(ai, pools * KP * 4)? == down(pc.pidx, pools * KP * 4)?
            && down(av, pools)? == down(pc.pvalid, pools)?;
        let ow = self.cfg.out_width() * 4;
        for &(r0, n) in rows {
            let (ta, tb) = (self.a_scr.tokens(), self.b_scr.tokens());
            let a = down(ta.offset(r0 * ow), n * ow)?;
            eq &= a == down(tb.offset(r0 * ow), n * ow)?;
            // 2026-10-06: A selection of nothing would compare equal and prove nothing.
            if a.chunks_exact(4)
                .all(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]) < 0)
            {
                bail!("{label}: no token selected at S = {s}");
            }
        }
        self.compared += 1;
        if !eq {
            self.differed += 1;
            self.first_diff
                .get_or_insert_with(|| format!("{label} at S = {s}"));
        }
        Ok(eq)
    }

    /// 2026-10-06: One full-width prefill window (`decode_k_wide`): all `QMAX * subs` rows
    /// written before the first sub-chunk (`note_ring_write`), then per 256-row sub-chunk the
    /// cursor advances and a 256-row selection runs.
    pub(crate) fn wide(&mut self, subs: usize) -> Result<()> {
        let k = QMAX * subs;
        let pos0 = self.b_begin(k)?;
        self.b.note_ring_write(pos0 + k);
        let (kb, gb) = self.rows_random(k);
        self.place(pos0, &kb, &gb)?;
        for _ in 0..subs {
            self.b.advance(QMAX)?;
            self.a_len += QMAX;
            self.select(QMAX, "wide sub-chunk")?;
        }
        Ok(())
    }

    /// 2026-10-06: MTP drafter context rows to `target` (`write_kv_rows`): 256-row tiles with
    /// no selection, each followed by the compress-only pass (`pool_compress_written`).
    pub(crate) fn drafter_ctx(&mut self, target: usize) -> Result<()> {
        while self.a_len < target {
            self.write(QMAX.min(target - self.a_len))?;
            if self.b.pool_compress_due() {
                let pc = self.b.pool_select_args()?.expect("pool cache");
                let rows = PoolRows {
                    k_normed: self.b.k_normed,
                    gate: self.b.gate,
                    valid: self.b.valid,
                    ape: self.ape,
                    first_key: 0,
                };
                let (g, cfg, len) = (self.g, self.cfg, self.b.len());
                compress_pools_only(g, &self.k, &cfg, &rows, &pc, len, self.s)?;
                self.b.note_selected();
            }
        }
        Ok(())
    }

    /// 2026-10-06: A K = 3 verify step on the host path (`decode_k` without a graph): from
    /// `seq` (B rewound to it first, as `check_lockstep`), three rows each written and selected,
    /// leaving both at `seq + 3`.
    pub(crate) fn verify_exact(&mut self, seq: usize) -> Result<()> {
        if self.b.len() > seq {
            self.b.rewind_to(seq)?;
        }
        self.a_len = seq;
        for _ in 0..VROWS {
            self.write(1)?;
            self.select(1, "verify (exact)")?;
        }
        Ok(())
    }

    /// 2026-10-06: Aux blob v2 round trip: snapshot B, restore into a fresh pool-cache state,
    /// release the old one and go on with the new (its graph is recaptured on next use).
    pub(crate) fn aux_round_trip(&mut self) -> Result<usize> {
        self.sync()?;
        let blob = self.b.snapshot_blob(self.g, self.s)?;
        let mut fresh = Glm5NextDsaState::alloc_pool_cache(self.g, &self.cfg, 0, None)?;
        fresh.restore_blob(&blob, self.g, self.s)?;
        self.sync()?;
        let mut old = std::mem::replace(&mut self.b, fresh);
        old.free(self.g)?;
        if let Some((a, b)) = self.graphs.take() {
            self.g.destroy_graph(a)?;
            self.g.destroy_graph(b)?;
        }
        Ok(blob.len())
    }

    /// 2026-10-06: Control 1: fill up to a pool boundary, flip one bit of B's ring copy of the
    /// pool's last row, select. Returns whether the compare caught it (it must).
    pub(crate) fn control_flip(&mut self) -> Result<bool> {
        self.write(KP - self.a_len % KP)?;
        self.sync()?;
        let slot = self.b.row_offset(self.b.len() - 1);
        let mut two = [0u8; 2];
        self.g.copy_d2h(self.b.k_normed.offset(slot), &mut two)?;
        two[0] ^= 0x40;
        self.g.copy_h2d(&two, self.b.k_normed.offset(slot))?;
        let (n, d) = (self.compared, self.differed);
        let eq = self.select(1, "control: flipped ring bit")?;
        // 2026-10-06: The control's own compare is not a parity result.
        (self.compared, self.differed) = (n, d);
        Ok(!eq)
    }

    /// 2026-10-06: Control 2: a rewind deeper than the ring (off a pool boundary) is refused
    /// and moves nothing.
    pub(crate) fn control_deep_rewind(&mut self) -> bool {
        let len = self.b.len();
        let ring = self.b.pool_cache().map_or(1, |p| p.book.ring_rows());
        let mut n = len.saturating_sub(ring + 1);
        if n.is_multiple_of(KP) {
            n = n.saturating_sub(1);
        }
        self.b.rewind_to(n).is_err() && self.b.len() == len
    }

    pub(crate) fn free(mut self) -> Result<()> {
        let g = self.g;
        if let Some((a, b)) = self.graphs.take() {
            g.destroy_graph(a)?;
            g.destroy_graph(b)?;
        }
        self.b.free(g)?;
        self.a_scr.free(g)?;
        self.b_scr.free(g)?;
        for p in [
            self.a_k,
            self.a_g,
            self.a_valid,
            self.ape,
            self.q,
            self.w,
            self.q_pos,
        ] {
            g.free(p)?;
        }
        g.free(self.q_mask)?;
        for r in self.rows {
            for p in [
                r.stage_k, r.stage_g, r.pos, r.sl, r.q_pos, r.geom_a, r.geom_b,
            ] {
                g.free(p)?;
            }
        }
        Ok(())
    }
}
