//! Public play history for one side of a battle.
//!
//! Everything recorded here is visible to both players in a real match: which
//! card a side deployed, where, when and at what cost. The featurizer shows a
//! viewer its own full deck and only the opponent cards revealed so far.

use crownforge_core::{CardId, Point, rules::MatchRules};

pub const MAX_EVENTS: usize = 32;

#[derive(Clone, Copy)]
pub struct PlayEvent {
    /// The card as it resolved on the board (Mirror shows the copied card).
    pub card: CardId,
    pub level: u8,
    /// World coordinates of the deployment.
    pub position: Point,
    pub tick: u32,
}

#[derive(Clone)]
pub struct Tracker {
    pub deck: [CardId; 8],
    /// Newest last, at most `MAX_EVENTS`.
    pub events: Vec<PlayEvent>,
    /// Value of `plays` when each deck card was last played; 0 means never.
    pub last_play: [u32; 8],
    pub last_tick: [u32; 8],
    pub times: [u16; 8],
    /// Order in which deck cards were first revealed; `u8::MAX` means hidden.
    pub reveal_rank: [u8; 8],
    pub revealed: u8,
    pub plays: u32,
    /// Elixir counted from public information: regeneration minus observed spending.
    pub est_elixir: i32,
}

impl Tracker {
    pub fn new(deck: [CardId; 8], initial_elixir: i32) -> Self {
        Self {
            deck,
            events: Vec::with_capacity(MAX_EVENTS + 1),
            last_play: [0; 8],
            last_tick: [0; 8],
            times: [0; 8],
            reveal_rank: [u8::MAX; 8],
            revealed: 0,
            plays: 0,
            est_elixir: initial_elixir,
        }
    }

    pub fn reset(&mut self, deck: [CardId; 8], initial_elixir: i32) {
        self.deck = deck;
        self.events.clear();
        self.last_play = [0; 8];
        self.last_tick = [0; 8];
        self.times = [0; 8];
        self.reveal_rank = [u8::MAX; 8];
        self.revealed = 0;
        self.plays = 0;
        self.est_elixir = initial_elixir;
    }

    pub fn record(
        &mut self,
        raw: CardId,
        card: CardId,
        level: u8,
        position: Point,
        tick: u32,
        cost: u8,
    ) {
        self.plays += 1;
        if let Some(index) = self.deck.iter().position(|&c| c == raw) {
            self.last_play[index] = self.plays;
            self.last_tick[index] = tick;
            self.times[index] = self.times[index].saturating_add(1);
            if self.reveal_rank[index] == u8::MAX {
                self.reveal_rank[index] = self.revealed;
                self.revealed += 1;
            }
        }
        if self.events.len() == MAX_EVENTS {
            self.events.remove(0);
        }
        self.events.push(PlayEvent {
            card,
            level,
            position,
            tick,
        });
        self.est_elixir =
            (self.est_elixir - i32::from(cost) * crownforge_core::rules::ELIXIR_UNIT).max(0);
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
