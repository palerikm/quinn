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
//! thresholds to cover what was observed, and falls back to the configured values once
//! [`RESET_AFTER_CONGESTION_EVENTS`] congestion events pass with no new spurious loss.
//!
//! Prior art: TCP RACK (RFC 8985 §6.2) widens its reordering window when a DSACK shows a
//! retransmission was spurious and resets it after 16 loss recoveries; Linux TCP raises its
//! duplicate-ACK threshold (`tp->reordering`) to the observed reordering degree, up to
//! `tcp_max_reordering` (300).

use std::cmp;

use crate::Duration;

/// Upper bound on the adapted packet threshold, matching Linux's default `tcp_max_reordering`
pub(super) const MAX_PACKET_THRESHOLD: u64 = 300;

/// Congestion events without a new spurious loss before the adaptation is dropped
///
/// RACK resets its window multiplier after 16 loss recoveries.
pub(super) const RESET_AFTER_CONGESTION_EVENTS: u32 = 16;

/// Per-path reordering tolerance learned from spurious losses
#[derive(Debug, Default, Clone)]
pub(super) struct ReorderAdapter {
    /// Packet threshold learned from spurious losses; 0 when nothing has been learned
    packet_threshold: u64,
    /// Time-threshold loss delay learned from spurious losses; zero when nothing has been learned
    loss_delay: Duration,
    /// Congestion events since the last spurious loss was proven
    events_since_spurious: u32,
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

    /// A packet declared lost has been acknowledged after all
    ///
    /// `displacement` is how many packet numbers above it had been acknowledged when it was
    /// declared lost; `age` is how long after it was sent its ACK arrived. The learned loss delay
    /// is capped at `max_loss_delay`, the horizon past which lost packets are forgotten.
    pub(super) fn on_spurious_loss(
        &mut self,
        displacement: u64,
        age: Duration,
        max_loss_delay: Duration,
    ) {
        self.events_since_spurious = 0;
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

    /// A congestion event was raised
    ///
    /// Once enough pass without a new spurious loss, the path is assumed to have stopped
    /// reordering and the configured thresholds apply again.
    pub(super) fn on_congestion_event(&mut self) {
        if self.packet_threshold == 0 && self.loss_delay.is_zero() {
            return;
        }
        self.events_since_spurious += 1;
        if self.events_since_spurious >= RESET_AFTER_CONGESTION_EVENTS {
            *self = Self::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn unadapted_uses_the_configured_values() {
        let r = ReorderAdapter::default();
        assert_eq!(r.packet_threshold(3), 3);
        assert_eq!(r.loss_delay(MS), MS);
    }

    #[test]
    fn spurious_loss_widens_both_thresholds() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(40, 16 * MS, 100 * MS);
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
        r.on_spurious_loss(9, 2 * MS, 100 * MS);
        assert!(r.packet_threshold(3) > 10);
    }

    #[test]
    fn adaptation_only_grows_until_reset() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(40, 16 * MS, 100 * MS);
        r.on_spurious_loss(5, 2 * MS, 100 * MS);
        assert_eq!(r.packet_threshold(3), 46);
        assert_eq!(r.loss_delay(MS), 18 * MS);
    }

    #[test]
    fn caps_apply() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(u64::MAX, 10 * MS, 4 * MS);
        assert_eq!(r.packet_threshold(3), MAX_PACKET_THRESHOLD);
        assert_eq!(r.loss_delay(MS), 4 * MS);
    }

    #[test]
    fn resets_after_quiet_congestion_events() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(40, 16 * MS, 100 * MS);
        for _ in 1..RESET_AFTER_CONGESTION_EVENTS {
            r.on_congestion_event();
        }
        assert_eq!(r.packet_threshold(3), 46, "reset one event early");
        r.on_congestion_event();
        assert_eq!(r.packet_threshold(3), 3);
        assert_eq!(r.loss_delay(MS), MS);
    }

    #[test]
    fn a_new_spurious_loss_restarts_the_count() {
        let mut r = ReorderAdapter::default();
        r.on_spurious_loss(40, 16 * MS, 100 * MS);
        for _ in 1..RESET_AFTER_CONGESTION_EVENTS {
            r.on_congestion_event();
        }
        r.on_spurious_loss(10, 4 * MS, 100 * MS);
        r.on_congestion_event();
        assert_eq!(r.packet_threshold(3), 46);
    }
}
