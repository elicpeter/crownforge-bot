//! What one player can keep count of from its own observations, tick by tick:
//! its card cycle, the cards it has seen the opponent play, an estimate of the
//! opponent's elixir, how long each unit has been on the board and how fast it
//! moves.
//!
//! The server shows units and towers only. An opponent's card is therefore
//! inferred from the units that appear: a spell that leaves no unit behind
//! (Fireball, Zap, Poison, ...) is never seen, and the elixir estimate does not
//! know about it. The estimate is an upper bound, not a count.
//!
//! `update` must be called with every tick's view, in order. The training bridge
//! and the tournament client both do, so both see the same numbers.

use std::collections::{HashMap, VecDeque};

use crate::cards::{AbilityKind, CardType, Library};
use crate::view::{View, ACT_OK};

/// The action slot that fires the champion's ability.
pub const ABILITY_SLOT: i32 = 100;
/// Cards waiting between a play and their return to the hand.
pub const QUEUE_LEN: u32 = 4;
pub const HISTORY: usize = 32;
pub const MAX_ELIXIR: f64 = 10.0;
/// How far, in tiles, new units may appear from the unit that made them.
const SOURCE_REACH: f64 = 3.0;
const REGULATION_TICKS: i64 = 3600;
const DOUBLE_ELIXIR_AT: i64 = 2400;

/// Elixir gained on the tick that starts at `tick`.
pub fn elixir_rate(tick: i64) -> f64 {
    if tick >= REGULATION_TICKS {
        3.0 / 56.0
    } else if tick >= DOUBLE_ELIXIR_AT {
        1.0 / 28.0
    } else {
        1.0 / 56.0
    }
}

/// The card library with the lookups the tracker and the features need.
pub struct Meta {
    pub lib: Library,
    /// Units (not spells) that produce this unit while they stand: Witch, huts.
    spawn_src: Vec<Vec<usize>>,
    /// Units that leave this unit behind when they die.
    death_src: Vec<Vec<usize>>,
    /// Spells that put this unit on the board (Goblin Barrel, Graveyard).
    spell_src: Vec<Vec<usize>>,
    /// The card that deploys this unit next to its own (Goblin Gang's spear goblins).
    secondary_of: Vec<Option<usize>>,
    /// The card an evolved unit is the evolved form of.
    pub base_of: Vec<Option<usize>>,
    /// Elixir one body of this unit is worth, for trade accounting.
    pub body_value: Vec<f32>,
}

impl Meta {
    pub fn new(lib: Library) -> Meta {
        let n = lib.defs.len();
        let mut spawn_src = vec![Vec::new(); n];
        let mut death_src = vec![Vec::new(); n];
        let mut spell_src = vec![Vec::new(); n];
        let mut secondary_of = vec![None; n];
        let mut base_of = vec![None; n];
        for (id, def) in lib.defs.iter().enumerate() {
            if let Some(product) = def.spawn_id {
                if def.ctype == CardType::Spell {
                    spell_src[product].push(id);
                } else {
                    spawn_src[product].push(id);
                }
            }
            for product in [def.death_id, def.revive_as].into_iter().flatten() {
                death_src[product].push(id);
            }
            // A champion's summons arrive around the champion while it stands.
            if let Some(product) = def.ability_spawn_id {
                spawn_src[product].push(id);
            }
            if let Some(product) = def.deploy_spawn_id {
                secondary_of[product] = Some(id);
            }
            if let Some(evolved) = def.evolved_id {
                if def.evo_cycles > 0 {
                    base_of[evolved] = Some(id);
                }
            }
        }
        let mut body_value = vec![0.0; n];
        for (id, def) in lib.defs.iter().enumerate() {
            let card = if def.deployable { Some(id) } else { base_of[id] };
            let Some(card) = card else { continue };
            // Only bodies a card itself deploys, each at an equal share of its cost.
            // Bodies that destroy themselves by design are worth nothing.
            if def.ctype != CardType::Troop || def.suicide || def.lifetime_ticks > 0 {
                continue;
            }
            let bodies = (def.count.max(1) + lib.defs[card].deploy_spawn_count.max(0)) as f32;
            body_value[id] = lib.defs[card].elixir as f32 / bodies;
        }
        Meta { lib, spawn_src, death_src, spell_src, secondary_of, base_of, body_value }
    }

    pub fn has_ability(&self, def: usize) -> bool {
        let d = &self.lib.defs[def];
        d.is_champion && d.ability != AbilityKind::None && d.ability != AbilityKind::Revive
    }

}

#[derive(Clone, Copy)]
pub struct Play {
    pub mine: bool,
    pub card: usize,
    pub x: f32,
    pub y: f32,
    pub tick: i64,
}

#[derive(Clone, Copy)]
pub struct Reveal {
    pub card: usize,
    pub last_tick: i64,
    pub count: u32,
    /// Value of the opponent's play count when this card was last played.
    pub last_play: u32,
}

#[derive(Clone, Copy)]
pub struct Mem {
    pub first: i64,
    last: i64,
    def: usize,
    /// An opponent's unit, not a tower.
    enemy: bool,
    /// Where it was on the latest tick it was seen.
    pos: [f64; 2],
    snap: [f64; 2],
    /// Tiles per second over the last whole decision interval.
    pub velocity: [f32; 2],
}

/// A hand card as it would be played now: the unit or spell that comes out and
/// what it costs. Mirror stands for the last card played, one elixir dearer.
#[derive(Clone, Copy)]
pub struct Resolved {
    pub def: usize,
    pub cost: i32,
    pub playable: bool,
}

pub struct Tracker {
    pub deck: [usize; 8],
    /// Value of `plays` when each deck card was last played; 0 means never.
    last_play: [u32; 8],
    pub plays: u32,
    last_played: Option<usize>,
    /// Evolution charge by card, counted from own plays as the engine counts it.
    evo: Vec<i32>,
    pending: Option<(i32, f32, f32)>,
    prev_hand: [usize; 4],
    /// Own champion on the board and the tick its ability is ready again.
    champion: Option<(u32, i64)>,
    pub reveals: Vec<Reveal>,
    pub opp_plays: u32,
    pub opp_elixir: f64,
    spell_seen: Vec<i64>,
    pub history: VecDeque<Play>,
    pub mem: HashMap<u32, Mem>,
    gone: Vec<(usize, f64, f64)>,
    groups: Vec<(usize, u32, f64, f64)>,
    pub tick: i64,
    started: bool,
}

impl Tracker {
    pub fn new(meta: &Meta, deck: [usize; 8]) -> Tracker {
        let n = meta.lib.defs.len();
        Tracker {
            deck,
            last_play: [0; 8],
            plays: 0,
            last_played: None,
            evo: vec![0; n],
            pending: None,
            prev_hand: [0; 4],
            champion: None,
            reveals: Vec::with_capacity(8),
            opp_plays: 0,
            opp_elixir: 5.0,
            spell_seen: vec![i64::MIN / 2; n],
            history: VecDeque::with_capacity(HISTORY),
            mem: HashMap::new(),
            gone: Vec::new(),
            groups: Vec::new(),
            tick: 0,
            started: false,
        }
    }

    pub fn reset(&mut self, deck: [usize; 8]) {
        self.deck = deck;
        self.last_play = [0; 8];
        self.plays = 0;
        self.last_played = None;
        self.evo.fill(0);
        self.pending = None;
        self.prev_hand = [0; 4];
        self.champion = None;
        self.reveals.clear();
        self.opp_plays = 0;
        self.opp_elixir = 5.0;
        self.spell_seen.fill(i64::MIN / 2);
        self.history.clear();
        self.mem.clear();
        self.gone.clear();
        self.tick = 0;
        self.started = false;
    }

    /// Remember the move just sent: a hand slot (or `ABILITY_SLOT`) and the tile
    /// centre, in tiles. The next view says whether it was carried out.
    pub fn note_action(&mut self, slot: i32, x: f32, y: f32) {
        self.pending = Some((slot, x, y));
    }

    /// Plays by this side since deck slot `index` was last played.
    pub fn plays_since(&self, index: usize) -> Option<u32> {
        (self.last_play[index] != 0).then(|| self.plays - self.last_play[index])
    }

    fn evolved(&self, meta: &Meta, def: usize) -> usize {
        let d = &meta.lib.defs[def];
        match d.evolved_id {
            Some(evolved) if d.evo_cycles > 0 && self.evo[def] >= d.evo_cycles => evolved,
            _ => def,
        }
    }

    /// What playing hand card `card` would do right now.
    pub fn resolve(&self, meta: &Meta, card: usize) -> Resolved {
        let d = &meta.lib.defs[card];
        if d.is_mirror {
            return match self.last_played {
                Some(last) => Resolved {
                    def: self.evolved(meta, last),
                    cost: meta.lib.defs[last].elixir + 1,
                    playable: true,
                },
                None => Resolved { def: card, cost: d.elixir, playable: false },
            };
        }
        Resolved { def: self.evolved(meta, card), cost: d.elixir, playable: true }
    }

    /// Own champion with an ability: unit, whether it can fire now, and why not
    /// (0 ready, 1 still deploying, 2 cooling down, 3 too little elixir, 4 a
    /// dash with no enemy in reach).
    pub fn ability(&self, meta: &Meta, view: &View) -> Option<(usize, bool, u8)> {
        let (id, ready_at) = self.champion?;
        let ent = view.ents.iter().find(|e| e.mine && e.id == id)?;
        let d = &meta.lib.defs[ent.def];
        let state = if ent.deploying {
            1
        } else if view.tick < ready_at {
            2
        } else if view.elixir < d.ability_cost as f64 {
            3
        } else if d.ability == AbilityKind::Dash && {
            // The engine measures in fixed point: stay a hair inside its reach.
            let reach = d.ability_range.to_f64() - 1e-3;
            !view.ents.iter().any(|e| {
                !e.mine && !e.deploying && e.hp > 0 && (e.x - ent.x).hypot(e.y - ent.y) <= reach
            })
        } {
            4
        } else {
            0
        };
        Some((ent.def, state == 0, state))
    }

    pub fn update(&mut self, meta: &Meta, view: &View, decision_ticks: i64) {
        let lib = &meta.lib;
        // ---- what became of the move sent on the previous tick ----
        if let Some((slot, x, y)) = self.pending.take() {
            if view.last_action == ACT_OK && self.started {
                if slot == ABILITY_SLOT {
                    if let Some((id, _)) = self.champion {
                        if let Some(ent) = view.ents.iter().find(|e| e.mine && e.id == id) {
                            let cooldown = lib.defs[ent.def].ability_cooldown_ticks.max(1) as i64;
                            self.champion = Some((id, view.tick - 1 + cooldown));
                        }
                    }
                } else if (0..4).contains(&slot) {
                    let card = self.prev_hand[slot as usize];
                    let played = if lib.defs[card].is_mirror {
                        self.last_played.unwrap_or(card)
                    } else {
                        self.last_played = Some(card);
                        card
                    };
                    // The engine's evolution counter, kept in step.
                    let d = &lib.defs[played];
                    if d.evolved_id.is_some() && d.evo_cycles > 0 {
                        if self.evo[played] >= d.evo_cycles {
                            self.evo[played] = 0;
                        } else {
                            self.evo[played] += 1;
                        }
                    }
                    self.plays += 1;
                    if let Some(index) = self.deck.iter().position(|&c| c == card) {
                        self.last_play[index] = self.plays;
                    }
                    self.push_play(Play { mine: true, card, x, y, tick: view.tick - 1 });
                }
            }
        }

        // ---- units: first sight, velocity, and the opponent's new arrivals ----
        self.groups.clear();
        let grid = decision_ticks > 0 && view.tick % decision_ticks == 0;
        let seconds = decision_ticks as f64 / 20.0;
        let mut champion = None;
        for e in &view.ents {
            if e.mine && champion.is_none() && meta.has_ability(e.def) {
                champion = Some(e.id);
            }
            match self.mem.get_mut(&e.id) {
                Some(mem) => {
                    mem.last = view.tick;
                    mem.pos = [e.x, e.y];
                    if grid {
                        mem.velocity = [
                            ((e.x - mem.snap[0]) / seconds) as f32,
                            ((e.y - mem.snap[1]) / seconds) as f32,
                        ];
                        mem.snap = [e.x, e.y];
                    }
                }
                None => {
                    self.mem.insert(
                        e.id,
                        Mem {
                            first: view.tick,
                            last: view.tick,
                            def: e.def,
                            enemy: !e.mine && !e.tower,
                            pos: [e.x, e.y],
                            snap: [e.x, e.y],
                            velocity: [0.0; 2],
                        },
                    );
                    if !e.mine && !e.tower && self.started {
                        match self.groups.iter_mut().find(|g| g.0 == e.def) {
                            Some(group) => {
                                group.1 += 1;
                                group.2 += e.x;
                                group.3 += e.y;
                            }
                            None => self.groups.push((e.def, 1, e.x, e.y)),
                        }
                    }
                }
            }
        }
        let now = view.tick;
        // Opponent units that left the board this tick, and where they stood.
        self.gone.clear();
        for mem in self.mem.values() {
            if mem.last != now && mem.enemy {
                self.gone.push((mem.def, mem.pos[0], mem.pos[1]));
            }
        }
        self.mem.retain(|_, mem| mem.last == now);
        match (champion, self.champion) {
            (Some(id), Some((old, _))) if id == old => {}
            (Some(id), _) => self.champion = Some((id, 0)),
            (None, _) => self.champion = None,
        }

        // ---- which of the arrivals are cards the opponent played ----
        self.groups.sort_by_key(|g| g.0);
        for index in 0..self.groups.len() {
            let (def, count, sum_x, sum_y) = self.groups[index];
            let centre = [sum_x / count as f64, sum_y / count as f64];
            if let Some(card) = self.card_behind(meta, view, def, count, centre) {
                let x = centre[0] as f32;
                let y = centre[1] as f32;
                self.opp_plays += 1;
                let opp_plays = self.opp_plays;
                if let Some(index) = self.reveals.iter().position(|r| r.card == card) {
                    let seen = &mut self.reveals[index];
                    seen.last_tick = view.tick;
                    seen.count += 1;
                    seen.last_play = opp_plays;
                } else if self.reveals.len() < 8 {
                    self.reveals.push(Reveal { card, last_tick: view.tick, count: 1, last_play: opp_plays });
                }
                self.opp_elixir = (self.opp_elixir - lib.defs[card].elixir as f64).max(0.0);
                self.push_play(Play { mine: false, card, x, y, tick: view.tick - 1 });
            }
        }

        // ---- the opponent's elixir refills whether or not we saw it spent ----
        let from = if self.started { self.tick } else { view.tick };
        for tick in from..view.tick {
            self.opp_elixir = (self.opp_elixir + elixir_rate(tick)).min(MAX_ELIXIR);
        }
        self.tick = view.tick;
        self.prev_hand = view.hand;
        self.started = true;
    }

    fn push_play(&mut self, play: Play) {
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(play);
    }

    /// The card that put `count` new units of `def` on the opponent's side, or
    /// `None` if something already on the board made them.
    fn card_behind(&mut self, meta: &Meta, view: &View, def: usize, count: u32, centre: [f64; 2]) -> Option<usize> {
        let lib = &meta.lib;
        let tick = view.tick;
        // Something on the board explains the arrivals only if it is where they
        // arrived: a Tombstone in the other lane does not hide a Graveyard.
        let near = |x: f64, y: f64| (x - centre[0]).hypot(y - centre[1]) <= SOURCE_REACH;
        let makes = &meta.spawn_src[def];
        let leaves = &meta.death_src[def];
        let standing = view.ents.iter().any(|e| !e.mine && makes.contains(&e.def) && near(e.x, e.y))
            || self.gone.iter().any(|g| makes.contains(&g.0) && near(g.1, g.2));
        let died = self.gone.iter().any(|g| leaves.contains(&g.0) && near(g.1, g.2));
        if standing || died {
            return None;
        }
        let d = &lib.defs[def];
        if let Some(card) = meta.secondary_of[def] {
            // Arrived beside its card's own units: that group stands for the play.
            let primary = self.groups.iter().any(|g| g.0 == card || meta.base_of[g.0] == Some(card));
            if primary {
                return None;
            }
        }
        if d.deployable && d.ctype != CardType::Spell && count >= d.count.max(1) as u32 {
            return Some(def);
        }
        if let Some(base) = meta.base_of[def] {
            return Some(base);
        }
        for &spell in &meta.spell_src[def] {
            let quiet = lib.defs[spell].effect_ticks.max(0) as i64 + 40;
            if tick - self.spell_seen[spell] > quiet {
                self.spell_seen[spell] = tick;
                return Some(spell);
            }
            // More units from a spell already counted (Graveyard).
            return None;
        }
        if d.deployable && d.ctype != CardType::Spell && meta.secondary_of[def].is_none() {
            // Fewer bodies than the card deploys and nothing else explains them.
            return None;
        }
        meta.secondary_of[def]
    }
}
