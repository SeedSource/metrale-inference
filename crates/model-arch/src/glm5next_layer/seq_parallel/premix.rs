// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The FFN-to-attention mix handoff between consecutive layers of a
//! sequence-parallel staged prefill under `METRALE_GLM_MHC_POST_MIX` (`CrossLayerPremix`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: a ticket is taken at most once, and a front uses premixed rows only for a ticket
//! equal to its own (same layer, chunk start, chunk rows and hidden buffer).

use crate::glm5next_mhc::Glm5NextMhcSiteWeights;

/// 2026-10-06: One staged chunk's FFN-to-attention handoff under `METRALE_GLM_MHC_POST_MIX`:
/// the layer whose attention front may use mix rows written by the previous layer's FFN back,
/// and the chunk they belong to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PremixTicket {
    pub layer: usize,
    pub seq_len_start: usize,
    pub total: usize,
    pub hidden: u64,
    /// 2026-10-06: `METRALE_GLM_MHC_POST_MIX_FINISH`: the FFN back also ran the next layer's
    /// attention `hc_finish` (its `y`, `post` and `comb` rows are written, not only the mix).
    /// The issuer says what it did; `take_handoff` reports it back.
    pub finished: bool,
}

/// 2026-10-06: `METRALE_GLM_MHC_POST_MIX` across layers: the FFN back of a layer also writes
/// the next layer's attention mix (`next_attn`, that layer's own site weights and scratch) and
/// then issues a ticket for that layer and chunk; the next layer's attention front runs
/// `hc_finish` only when it takes a ticket equal to its own, and recomputes the mix otherwise.
/// Every text layer shares one `ticket` (the loader's post-pass); `next_attn` is `None` for
/// the last text layer and the MTP block.
#[derive(Debug, Clone, Default)]
pub struct CrossLayerPremix {
    pub next_attn: Option<Glm5NextMhcSiteWeights>,
    pub ticket: std::sync::Arc<std::sync::Mutex<Option<PremixTicket>>>,
}

impl CrossLayerPremix {
    /// 2026-10-06: Record that the next layer's mix rows of this chunk are written.
    pub fn issue(&self, t: PremixTicket) {
        if let Ok(mut g) = self.ticket.lock() {
            *g = Some(t);
        }
    }

    /// 2026-10-06: Clear the ticket and say whether it was `t`. A layer calls this once per
    /// chunk, so a ticket is used at most once.
    pub fn take_matches(&self, t: PremixTicket) -> bool {
        self.take_handoff(t) == Some(t.finished)
    }

    /// 2026-10-06: Clear the ticket; when it was `t` apart from `finished`, `Some` of the
    /// issuer's `finished` (whether the front may skip `hc_finish` too), else `None`.
    pub fn take_handoff(&self, t: PremixTicket) -> Option<bool> {
        let taken = self.ticket.lock().ok().and_then(|mut g| g.take())?;
        let same = PremixTicket {
            finished: t.finished,
            ..taken
        } == t;
        same.then_some(taken.finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(layer: usize) -> PremixTicket {
        PremixTicket {
            layer,
            seq_len_start: 8192,
            total: 8192,
            hidden: 0x1000,
            finished: false,
        }
    }

    #[test]
    fn a_ticket_matches_once_and_only_its_own_layer_and_chunk() {
        let p = CrossLayerPremix::default();
        assert!(!p.take_matches(t(1)), "no ticket issued");
        p.issue(t(2));
        assert!(!p.take_matches(t(1)), "another layer");
        assert!(!p.take_matches(t(2)), "the miss above consumed it");
        p.issue(t(2));
        assert!(p.take_matches(t(2)));
        assert!(!p.take_matches(t(2)), "taken once");
        p.issue(t(3));
        let other_chunk = PremixTicket { seq_len_start: 0, ..t(3) };
        assert!(!p.take_matches(other_chunk));
        let q = CrossLayerPremix {
            next_attn: None,
            ticket: p.ticket.clone(),
        };
        p.issue(t(4));
        assert!(q.take_matches(t(4)), "layers share one ticket");
    }

    /// 2026-10-06: `take_handoff` matches on the chunk identity and reports the issuer's
    /// `finished`, once.
    #[test]
    fn a_handoff_reports_whether_the_issuer_finished() {
        let p = CrossLayerPremix::default();
        assert_eq!(p.take_handoff(t(2)), None, "no ticket issued");
        p.issue(PremixTicket {
            finished: true,
            ..t(2)
        });
        assert_eq!(p.take_handoff(t(2)), Some(true));
        assert_eq!(p.take_handoff(t(2)), None, "taken once");
        p.issue(t(2));
        assert_eq!(p.take_handoff(t(2)), Some(false));
        p.issue(PremixTicket {
            finished: true,
            ..t(3)
        });
        assert_eq!(p.take_handoff(t(2)), None, "another layer");
        assert_eq!(p.take_handoff(t(3)), None, "the miss above consumed it");
    }
}
