//! What a player can count about one side from public events: which deck card
//! was played when, the cycle position that follows from it, and elixir.
//!
//! The simulator's own public play journal supplies the play history shown to
//! the policy. This tracker adds the two things that journal does not carry:
//! the card cycle and an elixir count.

use crownforge_core::{CardId, rules::ELIXIR_UNIT, rules::MatchRules};

/// Cards waiting between a play and their return to the hand.
pub const QUEUE_LEN: u32 = 4;

#[derive(Clone)]
pub struct Tracker {
    pub deck: [CardId; 8],
    /// Card level per deck slot.
    pub levels: [u8; 8],
    /// Value of `plays` when each deck card was last played; 0 means never.
    last_play: [u32; 8],
    pub plays: u32,
    /// Elixir counted from public information: regeneration minus observed
    /// card and ability spending.
    pub est_elixir: i32,
}

impl Tracker {
    pub fn new(deck: [CardId; 8], levels: [u8; 8], initial_elixir: i32) -> Self {
        Self {
            deck,
            levels,
            last_play: [0; 8],
            plays: 0,
            est_elixir: initial_elixir,
        }
    }

    pub fn reset(&mut self, deck: [CardId; 8], levels: [u8; 8], initial_elixir: i32) {
        *self = Self::new(deck, levels, initial_elixir);
    }

    /// Record an accepted play of the deck card `raw` (the card that left the
    /// hand, so Mirror counts as Mirror) costing `cost` elixir.
    pub fn record(&mut self, raw: CardId, cost: u8) {
        self.plays += 1;
        if let Some(index) = self.deck.iter().position(|&card| card == raw) {
            self.last_play[index] = self.plays;
        }
        self.spend(u32::from(cost));
    }

    /// Subtract a visible payment of whole elixir, such as an ability.
    pub fn spend(&mut self, elixir: u32) {
        self.est_elixir = (self.est_elixir - elixir as i32 * ELIXIR_UNIT).max(0);
    }

    /// Add a visible elixir gain, in elixir units, up to the cap.
    pub fn grant(&mut self, units: i32, cap: i32) {
        self.est_elixir = (self.est_elixir + units).min(cap);
    }

    /// Plays by this side since the card in deck slot `index` was last played;
    /// `None` if it has not been played. The cycle is a queue of
    /// [`QUEUE_LEN`]: a card played fewer than that many plays ago is still
    /// waiting, and every other card is in the hand or arriving.
    pub fn plays_since(&self, index: usize) -> Option<u32> {
        (self.last_play[index] != 0).then(|| self.plays - self.last_play[index])
    }

    pub fn slot_of(&self, card: CardId) -> Option<usize> {
        self.deck.iter().position(|&c| c == card)
    }

    /// Apply regeneration for `ticks` ticks starting at `start_tick`.
    pub fn regenerate(&mut self, timing: &MatchRules, start_tick: u32, ticks: u32) {
        for tick in start_tick..start_tick + ticks {
            let phase = if tick < timing.double_elixir_tick {
                0
            } else if tick < timing.triple_elixir_tick {
                1
            } else {
                2
            };
            self.est_elixir =
                (self.est_elixir + timing.elixir_per_tick[phase]).min(timing.max_elixir);
        }
    }
}
