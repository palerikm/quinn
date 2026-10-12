//! Adaptive reordering tolerance for loss detection
//!
//! RFC 9002 declares a packet lost once `packet_threshold` later packets have been acknowledged,
//! or once it is older than `time_threshold × rtt`. Both are static, so a path that reorders more
//! deeply than they allow has packets declared lost — and the congestion window cut — although
//! nothing was lost.
//!
//! A loss is proven spurious when the "lost" packet is acknowledged after all. That ACK tells us
//! how far the packet was displaced: how many later packets had been acknowledged when it was
//! declared lost, and how old it was when its ACK arrived. [`ReorderAdapter`] widens both
//! thresholds to cover what was observed. It narrows them again only once the path has gone
//! quiet: [`DECAY_AFTER_CONGESTION_EVENTS`] congestion events with no spurious loss proven for
//! the whole lost-packet horizon, after which the learned values are halved. Proofs arrive after
//! the congestion events they disprove, so counting events alone would discard what was learned
//! on exactly the paths that reorder most.
//!
//! Prior art: TCP RACK (RFC 8985 §6.2) widens its reordering window when a DSACK shows a
//! retransmission was spurious and resets it after 16 loss recoveries; Linux TCP raises its
//! duplicate-ACK threshold (`tp->reordering`) to the observed reordering degree, up to
//! `tcp_max_reordering` (300).

use std::cmp;

use crate::{Duration, Instant};

/// Upper bound on the adapted packet threshold, matching Linux's default `tcp_max_reordering`
pub(super) const MAX_PACKET_THRESHOLD: u64 = 300;

/// Congestion events without a spurious loss before the adaptation decays
///
/// RACK resets its window multiplier after 16 loss recoveries.
pub(super) const DECAY_AFTER_CONGESTION_EVENTS: u32 = 16;

/// A learned loss delay below this is dropped rather than halved further
const MIN_LEARNED_LOSS_DELAY: Duration = Duration::from_millis(1);

/// Per-path reordering tolerance learned from spurious losses
#[derive(Debug, Default, Clone)]
pub(super) struct ReorderAdapter {
    /// Packet threshold learned from spurious losses; 0 when nothing has been learned
    packet_threshold: u64,
    /// Time-threshold loss delay learned from spurious losses; zero when nothing has been learned
    loss_delay: Duration,
    /// Congestion events since the last spurious loss was proven, or since the last decay
    events_since_spurious: u32,
    /// When the last spurious loss was proven
    last_proof: Option<Instant>,
}

impl ReorderAdapter {
    /// The packet threshold to use, given the configured one
    pub(super) fn packet_threshold(&self, configured: u64) -> u64 {
        cmp::max(configured, self.packet_threshold)
    }

    /// The time-threshold loss delay to use, given the one computed from the RTT
    pub(super) fn loss_delay(&self, computed: Duration) -> Duration {
        cmp::max(computed, self.loss_delay)
    }

    /// A packet declared lost has been acknowledged after all, at `now`
    ///
    /// `displacement` is how many packet numbers above it had been acknowledged when it was
    /// declared lost; `age` is how long after it was sent its ACK arrived. The learned loss delay
    /// is capped at `max_loss_delay`, the horizon past which lost packets are forgotten.
    pub(super) fn on_spurious_loss(
        &mut self,
        now: Instant,
        displacement: u64,
        age: Duration,
        max_loss_delay: Duration,
    ) {
        self.events_since_spurious = 0;
        self.last_proof = Some(now);
        // Tolerate the observed displacement: the packet was lost at `displacement`, so the
        // threshold must exceed it. The eighth of margin covers the next reordering being a
        // little deeper, e.g. by a packet number skipped in between.
        let packets = displacement
            .saturating_add(displacement / 8)
            .saturating_add(1)
            .min(MAX_PACKET_THRESHOLD);
        self.packet_threshold = cmp::max(self.packet_threshold, packets);
        // Leave an eighth of margin, as RFC 9002's default time threshold does over the RTT.
        let delay = cmp::min(age + age / 8, max_loss_delay);
        self.loss_delay = cmp::max(self.loss_delay, delay);
    }

    /// A congestion event was raised at `now`
    ///
    /// `quiet` is the lost-packet horizon: a loss not proven spurious within it never will be.
    /// Once [`DECAY_AFTER_CONGESTION_EVENTS`] events pass AND no proof has arrived for `quiet`,
    /// the path is taken to reorder less than it did and the learned values are halved; a few
    /// such periods in a row bring the configured thresholds back. While proofs keep arriving,
    /// however late, nothing decays.
    pub(super) fn on_congestion_event(&mut self, now: Instant, quiet: Duration) {
        if self.packet_threshold == 0 && self.loss_delay.is_zero() {
            return;
        }
        self.events_since_spurious = self.events_since_spurious.saturating_add(1);
        let quiet_long_enough = self
            .last_proof
            .is_none_or(|t| now.saturating_duration_since(t) >= quiet);
        if self.events_since_spurious < DECAY_AFTER_CONGESTION_EVENTS || !quiet_long_enough {
            return;
        }
        self.events_since_spurious = 0;
        self.packet_threshold /= 2;
        self.loss_delay /= 2;
        if self.loss_delay < MIN_LEARNED_LOSS_DELAY {
            self.loss_delay = Duration::ZERO;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);
    /// The lost-packet horizon the tests decay against
    const QUIET: Duration = Duration::from_millis(10);

    #[test]
    fn unadapted_uses_the_configured_values() {
        let r = ReorderAdapter::default();
        assert_eq!(r.packet_threshold(3), 3);
        assert_eq!(r.loss_delay(MS), MS);
    }

    #[test]
    fn spurious_loss_widens_both_thresholds() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(Instant::now(), 40, 16 * MS, 100 * MS);
        assert_eq!(r.packet_threshold(3), 46);
        assert_eq!(r.loss_delay(MS), 18 * MS);
        // Never narrower than configured
        assert_eq!(r.packet_threshold(64), 64);
        assert_eq!(r.loss_delay(50 * MS), 50 * MS);
    }

    /// The next reordering may be a packet deeper, e.g. when the sender skips a packet number
    #[test]
    fn a_slightly_deeper_reordering_is_still_covered() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(Instant::now(), 9, 2 * MS, 100 * MS);
        assert!(r.packet_threshold(3) > 10);
    }

    #[test]
    fn adaptation_only_grows_until_it_decays() {
        let mut r = ReorderAdapter::default();
        let now = Instant::now();
        r.on_spurious_loss(now, 40, 16 * MS, 100 * MS);
        r.on_spurious_loss(now, 5, 2 * MS, 100 * MS);
        assert_eq!(r.packet_threshold(3), 46);
        assert_eq!(r.loss_delay(MS), 18 * MS);
    }

    #[test]
    fn caps_apply() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(Instant::now(), u64::MAX, 10 * MS, 4 * MS);
        assert_eq!(r.packet_threshold(3), MAX_PACKET_THRESHOLD);
        assert_eq!(r.loss_delay(MS), 4 * MS);
    }

    /// The flaw this replaces: under heavy reordering many congestion events land before the
    /// late ACKs that disprove them. Events alone, within the horizon of the last proof, must not
    /// discard what was learned.
    #[test]
    fn no_decay_while_proofs_are_recent() {
        let mut r = ReorderAdapter::default();
        let t0 = Instant::now();
        r.on_spurious_loss(t0, 40, 16 * MS, 100 * MS);
        for _ in 0..10 * DECAY_AFTER_CONGESTION_EVENTS {
            r.on_congestion_event(t0 + QUIET / 2, QUIET);
        }
        assert_eq!(r.packet_threshold(3), 46);
        assert_eq!(r.loss_delay(MS), 18 * MS);
    }

    #[test]
    fn halves_after_quiet_congestion_events() {
        let mut r = ReorderAdapter::default();
        let t0 = Instant::now();
        r.on_spurious_loss(t0, 40, 16 * MS, 100 * MS);
        let later = t0 + QUIET;
        for _ in 1..DECAY_AFTER_CONGESTION_EVENTS {
            r.on_congestion_event(later, QUIET);
        }
        assert_eq!(r.packet_threshold(3), 46, "decayed one event early");
        r.on_congestion_event(later, QUIET);
        assert_eq!(r.packet_threshold(3), 23, "halved");
        assert_eq!(r.loss_delay(MS), 9 * MS, "halved");
    }

    #[test]
    fn repeated_quiet_periods_return_to_configured() {
        let mut r = ReorderAdapter::default();
        let t0 = Instant::now();
        r.on_spurious_loss(t0, 40, 16 * MS, 100 * MS);
        let later = t0 + QUIET;
        for _ in 0..8 * DECAY_AFTER_CONGESTION_EVENTS {
            r.on_congestion_event(later, QUIET);
        }
        assert_eq!(r.packet_threshold(3), 3);
        assert_eq!(r.loss_delay(MS), MS);
    }

    #[test]
    fn a_new_spurious_loss_restarts_the_count() {
        let mut r = ReorderAdapter::default();
        let t0 = Instant::now();
        r.on_spurious_loss(t0, 40, 16 * MS, 100 * MS);
        let later = t0 + QUIET;
        for _ in 1..DECAY_AFTER_CONGESTION_EVENTS {
            r.on_congestion_event(later, QUIET);
        }
        r.on_spurious_loss(later, 10, 4 * MS, 100 * MS);
        r.on_congestion_event(later + QUIET, QUIET);
        assert_eq!(r.packet_threshold(3), 46);
    }
}
