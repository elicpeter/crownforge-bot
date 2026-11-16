//! Turns one player's view into fixed-width rows for the network, in the
//! planes and column order of the bot's other simulators (see the repository's
//! `src/features.rs`). Columns this simulator has nothing for stay zero:
//! shields, status flags, card levels and the whole hazard plane, since the
//! tournament server sends neither spell areas nor projectiles.
//!
//! Everything in `featurize` comes from the `View` and the `Tracker`, which hold
//! only what the server sends. `privileged` reads the engine and fills the
//! critic's training-only planes.

use std::cmp::Ordering;

use crate::cards::{CardDef, CardType, Library, SpellEffect, Targets};
use crate::engine::Engine;
use crate::track::{Meta, Tracker, MAX_ELIXIR, QUEUE_LEN};
use crate::view::View;

pub const SCHEMA_VERSION: u32 = 2;
/// Version of this simulator's planes. Checkpoints record it.
pub const BOT_SCHEMA: u32 = 100;
pub const GLOBAL_F: usize = 4;
pub const ENTITY_F: usize = 24;
pub const ENTITY_X_F: usize = 5;
pub const HAND_F: usize = 4;
pub const DECK_ROWS: usize = 8;
pub const DECK_F: usize = 2;
pub const DECK_X_F: usize = 4;
pub const PLAY_ROWS: usize = 32;
pub const PLAY_F: usize = 7;
pub const REVEAL_ROWS: usize = 8;
pub const REVEAL_F: usize = 3;
pub const REVEAL_X_F: usize = 2;
pub const HAZARD_ROWS: usize = 16;
pub const HAZARD_F: usize = 5;
pub const EXTRA_F: usize = 6;
pub const ABILITY_ROWS: usize = 2;
pub const ABILITY_F: usize = 9;
pub const PRIV_GLOBAL_F: usize = 3;
pub const CARD_F: usize = 32;
pub const UNIT_F: usize = 24;
const ENTITY_ROW: usize = ENTITY_F + ENTITY_X_F;

/// Vocabulary sizes, with headroom. Token 0 is "none"; a card or unit is its
/// library id plus one. The library lists cards and then towers by name.
pub const N_CARDS: usize = 160;
pub const N_UNITS: usize = 160;
pub const N_HAZARDS: usize = 32;

pub const WIDTH: usize = 18;
pub const HEIGHT: usize = 32;
pub const CELLS: usize = WIDTH * HEIGHT;
pub const PLAY_ACTION_COUNT: usize = 1 + 4 * CELLS;
pub const ABILITY_ACTION: usize = PLAY_ACTION_COUNT;
pub const SECONDARY_PLAY_START: usize = PLAY_ACTION_COUNT + 2;
pub const ACTION_COUNT: usize = SECONDARY_PLAY_START + 4 * CELLS;
pub const BASE_LEVEL: f32 = 11.0;
const TICKS_PER_SECOND: f32 = 20.0;
/// Rows of the board a troop may always be played on, and the last row of the
/// pocket a fallen princess tower opens in its lane.
const OWN_ROWS: usize = 15;
const POCKET_ROWS: usize = 21;
/// The river, and the columns of the left and right bridge across it.
const RIVER_ROWS: (usize, usize) = (15, 17);
const BRIDGE_COLUMNS: [(usize, usize); 2] = [(2, 5), (13, 16)];

pub fn token(def: usize) -> f32 {
    (def + 1) as f32
}

fn flag(value: bool) -> f32 {
    f32::from(u8::from(value))
}

fn log_scaled(value: i32, scale: f32) -> f32 {
    (value.max(0) as f32).ln_1p() / scale
}

fn write_stats(d: &CardDef, out: &mut [f32]) {
    out[0] = log_scaled(d.hp, 9.0);
    out[1] = log_scaled(d.shield, 9.0);
    out[2] = log_scaled(d.damage, 8.0);
    out[3] = d.range.to_f64() as f32 / 10.0;
    out[4] = d.speed.to_f64() as f32 * TICKS_PER_SECOND / 3.0;
    out[5] = d.hit_ticks as f32 / TICKS_PER_SECOND / 3.0;
    out[6] = flag(matches!(d.targets, Targets::Air | Targets::Both));
    out[7] = flag(matches!(d.targets, Targets::Ground | Targets::Both));
    out[8] = flag(d.flying);
    out[9] = flag(d.buildings_only);
    out[10] = d.splash.to_f64() as f32 / 3.0;
    out[11] = (d.mass.to_f64() as f32).ln_1p() / 3.0;
    out[12] = d.sight.to_f64() as f32 / 10.0;
    out[13] = d.radius.to_f64() as f32 / 2.0;
}

/// Static per-unit stats, indexed by unit token.
pub fn unit_table(lib: &Library) -> Vec<f32> {
    let mut table = vec![0.0; N_UNITS * UNIT_F];
    for (id, d) in lib.defs.iter().enumerate() {
        let row = &mut table[(id + 1) * UNIT_F..][..UNIT_F];
        write_stats(d, row);
        row[14] = flag(matches!(d.ctype, CardType::Building | CardType::Tower));
        row[15] = 1.0;
        row[16] = d.deploy_ticks.max(0) as f32 / TICKS_PER_SECOND / 3.0;
        row[17] = d.lifetime_ticks.max(0) as f32 / TICKS_PER_SECOND / 60.0;
        row[18] = flag(d.suicide);
        row[19] = flag(d.invisible);
        row[20] = flag(d.spawn_id.is_some());
        row[21] = flag(d.death_id.is_some() || d.death_damage > 0);
        row[22] = flag(d.jumps_river);
        row[23] = d.load_ticks.max(0) as f32 / TICKS_PER_SECOND / 2.0;
    }
    table
}

/// Static per-card features, indexed by card token. Evolved forms and summoned
/// units have a row too (a hand card can play as its evolved form); column 0
/// says whether the row is a deck card.
pub fn card_table(lib: &Library) -> Vec<f32> {
    let mut table = vec![0.0; N_CARDS * CARD_F];
    for (id, d) in lib.defs.iter().enumerate() {
        if d.ctype == CardType::Tower {
            continue;
        }
        let row = &mut table[(id + 1) * CARD_F..][..CARD_F];
        row[0] = flag(d.deployable);
        row[1] = d.elixir as f32 / 10.0;
        row[2] = flag(d.ctype == CardType::Troop);
        row[3] = flag(d.ctype == CardType::Building);
        row[4] = flag(d.ctype == CardType::Spell);
        row[5] = flag(d.is_mirror);
        row[6] = flag(d.is_champion);
        row[7] = (d.count.max(1) + d.deploy_spawn_count.max(0)) as f32 / 16.0;
        if d.ctype != CardType::Spell {
            write_stats(d, &mut row[8..22]);
        } else {
            row[22] = log_scaled(d.damage, 8.0);
            row[23] = d.crown_pct as f32 / 100.0;
            row[24] = d.spell_radius.to_f64() as f32 / 5.0;
            row[28] = d.effect_ticks.max(0) as f32 / TICKS_PER_SECOND / 10.0;
            row[29] = d.stun_ticks.max(0) as f32 / TICKS_PER_SECOND;
            row[30] = flag(d.effect != SpellEffect::None);
            row[31] = flag(d.spawn_id.is_some());
        }
        row[25] = flag(d.deploy_anywhere);
        row[26] = flag(d.evolved_id.is_some() && d.evo_cycles > 0);
        row[27] = flag(d.knockback.0 > 0);
    }
    table
}

/// Output rows of one player.
pub struct Rows {
    pub max_entities: usize,
    pub n_entities: usize,
    pub overflow: u32,
    pub n_plays: usize,
    pub n_reveals: usize,
    /// Whether the legal mask allows anything besides waiting.
    pub choice: bool,
    pub global: Vec<f32>,
    pub entities: Vec<f32>,
    pub entity_extras: Vec<f32>,
    pub abilities: Vec<f32>,
    pub hand: Vec<f32>,
    pub own_deck: Vec<f32>,
    pub deck_extras: Vec<f32>,
    pub plays: Vec<f32>,
    pub reveals: Vec<f32>,
    pub reveal_extras: Vec<f32>,
    pub extras: Vec<f32>,
    pub legal: Vec<u8>,
    pub priv_global: Vec<f32>,
    pub priv_hand: Vec<f32>,
    pub priv_deck: Vec<f32>,
    scratch: Vec<[f32; ENTITY_ROW]>,
}

impl Rows {
    pub fn new(max_entities: usize) -> Rows {
        Rows {
            max_entities,
            n_entities: 0,
            overflow: 0,
            n_plays: 0,
            n_reveals: 0,
            choice: false,
            global: vec![0.0; GLOBAL_F],
            entities: vec![0.0; max_entities * ENTITY_F],
            entity_extras: vec![0.0; max_entities * ENTITY_X_F],
            abilities: vec![0.0; ABILITY_ROWS * ABILITY_F],
            hand: vec![0.0; 4 * HAND_F],
            own_deck: vec![0.0; DECK_ROWS * DECK_F],
            deck_extras: vec![0.0; DECK_ROWS * DECK_X_F],
            plays: vec![0.0; PLAY_ROWS * PLAY_F],
            reveals: vec![0.0; REVEAL_ROWS * REVEAL_F],
            reveal_extras: vec![0.0; REVEAL_ROWS * REVEAL_X_F],
            extras: vec![0.0; EXTRA_F],
            legal: vec![0; ACTION_COUNT],
            priv_global: vec![0.0; PRIV_GLOBAL_F],
            priv_hand: vec![0.0; 4],
            priv_deck: vec![0.0; DECK_ROWS],
            scratch: Vec::with_capacity(max_entities),
        }
    }
}

fn compare_rows(left: &[f32; ENTITY_ROW], right: &[f32; ENTITY_ROW]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        let order = left.total_cmp(right);
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

/// Own then enemy tower health fractions: king, left, right for each side.
/// A fallen tower is gone from the view and reads zero.
pub fn tower_fractions(meta: &Meta, view: &View) -> [f32; 6] {
    let mut out = [0.0; 6];
    for e in view.ents.iter().filter(|e| e.tower) {
        let lane = if e.def == meta.lib.king_tower {
            0
        } else if e.x < WIDTH as f64 / 2.0 {
            1
        } else {
            2
        };
        let index = if e.mine { lane } else { 3 + lane };
        out[index] = if e.max_hp > 0 { (e.hp as f32 / e.max_hp as f32).clamp(0.0, 1.0) } else { 0.0 };
    }
    out
}

/// Whether the opponent's princess tower in each lane has fallen.
fn open_lanes(meta: &Meta, view: &View) -> [bool; 2] {
    let mut standing = [false; 2];
    for e in &view.ents {
        if !e.mine && e.tower && e.def == meta.lib.princess_tower && e.hp > 0 {
            standing[usize::from(e.x >= WIDTH as f64 / 2.0)] = true;
        }
    }
    [!standing[0], !standing[1]]
}

/// Whether the player can do anything but wait: a card it can afford, or a
/// champion's ability that is ready. A card always has a tile to go on.
pub fn has_choice(meta: &Meta, view: &View, tracker: &Tracker) -> bool {
    view.hand.iter().any(|&card| {
        let play = tracker.resolve(meta, card);
        play.playable && play.cost as f64 <= view.elixir
    }) || tracker.ability(meta, view).is_some_and(|ability| ability.1)
}

fn cycle_features(since: Option<u32>) -> (f32, bool) {
    match since {
        Some(since) => (since.min(8) as f32 / 8.0, since < QUEUE_LEN),
        None => (1.0, false),
    }
}

pub fn featurize(meta: &Meta, view: &View, tracker: &Tracker, rows: &mut Rows) {
    let lib = &meta.lib;
    let ability = tracker.ability(meta, view);

    // ---- global: elapsed seconds, elixir, ready abilities, history enabled ----
    rows.global[0] = (view.tick / 20) as f32;
    rows.global[1] = view.elixir as f32;
    rows.global[2] = flag(ability.is_some_and(|a| a.1));
    rows.global[3] = 1.0;

    // ---- entities, in a canonical order that hides creation order ----
    let cap = rows.max_entities;
    rows.scratch.clear();
    let mut overflow = 0;
    for e in &view.ents {
        if rows.scratch.len() == cap {
            overflow += 1;
            continue;
        }
        let d = &lib.defs[e.def];
        let kind = if e.tower {
            if e.def == lib.king_tower { 1.0 } else { 2.0 }
        } else if d.ctype == CardType::Building {
            4.0
        } else {
            3.0
        };
        let mem = tracker.mem.get(&e.id);
        let age = mem.map_or(0, |m| view.tick - m.first) as f32;
        let velocity = mem.map_or([0.0; 2], |m| m.velocity);
        let mut row = [0.0; ENTITY_ROW];
        row[0] = kind;
        row[1] = flag(e.mine);
        row[2] = token(e.def);
        row[3] = BASE_LEVEL;
        row[4] = (e.x / WIDTH as f64) as f32;
        row[5] = (e.y / HEIGHT as f64) as f32;
        row[6] = if e.max_hp > 0 { (e.hp as f32 / e.max_hp as f32).clamp(0.0, 1.0) } else { 0.0 };
        row[23] = flag(e.hp > 0);
        // Bot-owned columns: deploying, time to ready, lifetime left, velocity.
        row[24] = flag(e.deploying);
        if e.deploying {
            row[25] = ((d.deploy_ticks as f32 - age).max(0.0) / TICKS_PER_SECOND / 3.0).min(1.0);
        }
        if d.lifetime_ticks > 0 && !e.tower {
            row[26] = (1.0 - age / d.lifetime_ticks as f32).clamp(0.0, 1.0);
        }
        row[27] = (velocity[0] / 3.0).clamp(-1.0, 1.0);
        row[28] = (velocity[1] / 3.0).clamp(-1.0, 1.0);
        rows.scratch.push(row);
    }
    rows.scratch.sort_unstable_by(compare_rows);
    rows.entities.fill(0.0);
    rows.entity_extras.fill(0.0);
    for ((row, extra), values) in rows
        .entities
        .chunks_exact_mut(ENTITY_F)
        .zip(rows.entity_extras.chunks_exact_mut(ENTITY_X_F))
        .zip(&rows.scratch)
    {
        row.copy_from_slice(&values[..ENTITY_F]);
        extra.copy_from_slice(&values[ENTITY_F..]);
    }
    rows.n_entities = rows.scratch.len();
    rows.overflow = overflow;

    // ---- ability button: unit, usable, charges, then the button state ----
    rows.abilities.fill(0.0);
    let champion_card = tracker.deck.iter().copied().find(|&card| meta.has_ability(card));
    if let Some(card) = champion_card {
        let row = &mut rows.abilities[..ABILITY_F];
        row[0] = token(card);
        match ability {
            Some((def, usable, state)) => {
                row[0] = token(def);
                row[1] = flag(usable);
                row[[3, 4, 7, 8, 6][state as usize]] = 1.0;
            }
            None => row[5] = 1.0,
        }
    }

    // ---- hand: card, what it plays as, level, cost ----
    for slot in 0..4 {
        let play = tracker.resolve(meta, view.hand[slot]);
        let row = &mut rows.hand[slot * HAND_F..][..HAND_F];
        row[0] = token(view.hand[slot]);
        row[1] = if play.playable { token(play.def) } else { 0.0 };
        row[2] = BASE_LEVEL;
        row[3] = play.cost as f32;
    }

    // ---- own deck: card, level; then played, plays since, still queued, next ----
    for index in 0..DECK_ROWS {
        let card = tracker.deck[index];
        rows.own_deck[index * DECK_F] = token(card);
        rows.own_deck[index * DECK_F + 1] = BASE_LEVEL;
        let since = tracker.plays_since(index);
        let (age, queued) = cycle_features(since);
        let extra = &mut rows.deck_extras[index * DECK_X_F..][..DECK_X_F];
        extra[0] = flag(since.is_some());
        extra[1] = age;
        extra[2] = flag(queued);
        extra[3] = flag(card == view.next);
    }

    // ---- recent plays by both sides, oldest first ----
    rows.plays.fill(0.0);
    rows.n_plays = tracker.history.len();
    for (row, play) in rows.plays.chunks_exact_mut(PLAY_F).zip(&tracker.history) {
        row[0] = flag(play.mine);
        row[1] = token(play.card);
        row[2] = BASE_LEVEL;
        row[3] = 1.0;
        row[4] = play.x / WIDTH as f32;
        row[5] = play.y / HEIGHT as f32;
        row[6] = play.tick as f32;
    }

    // ---- opponent cards seen so far, in first-seen order ----
    rows.reveals.fill(0.0);
    rows.reveal_extras.fill(0.0);
    rows.n_reveals = tracker.reveals.len();
    for (index, seen) in tracker.reveals.iter().enumerate() {
        let row = &mut rows.reveals[index * REVEAL_F..][..REVEAL_F];
        row[0] = token(seen.card);
        row[1] = seen.last_tick as f32;
        row[2] = seen.count as f32;
        let (age, queued) = cycle_features(Some(tracker.opp_plays - seen.last_play));
        rows.reveal_extras[index * REVEAL_X_F] = age;
        // Not queued means the card is back in the opponent's hand or arriving.
        rows.reveal_extras[index * REVEAL_X_F + 1] = flag(!queued);
    }

    // ---- extras: estimated opponent elixir, time within the second, and how
    // far each side is into its first cycle ----
    rows.extras[0] = tracker.opp_elixir as f32;
    rows.extras[1] = flag(tracker.opp_elixir >= MAX_ELIXIR);
    rows.extras[2] = (view.tick % 20) as f32 / TICKS_PER_SECOND;
    rows.extras[3] = 0.0;
    rows.extras[4] = tracker.opp_plays.min(QUEUE_LEN) as f32 / QUEUE_LEN as f32;
    rows.extras[5] = tracker.plays.min(QUEUE_LEN) as f32 / QUEUE_LEN as f32;

    // ---- legal actions ----
    let legal = &mut rows.legal;
    legal.fill(0);
    legal[0] = 1;
    let open = open_lanes(meta, view);
    let mut choice = false;
    for slot in 0..4 {
        let play = tracker.resolve(meta, view.hand[slot]);
        if !play.playable || play.cost as f64 > view.elixir {
            continue;
        }
        choice = true;
        let cells = &mut legal[1 + slot * CELLS..][..CELLS];
        let d = &lib.defs[play.def];
        if d.ctype == CardType::Spell || d.deploy_anywhere {
            cells.fill(1);
            continue;
        }
        cells[..OWN_ROWS * WIDTH].fill(1);
        // The engine also takes the two river rows of an open pocket, but a
        // ground troop put in the water off the bridge never leaves it. Those
        // tiles are offered only to what flies, jumps or does not walk.
        let wades = d.flying || d.jumps_river || d.ctype == CardType::Building;
        for lane in 0..2 {
            if open[lane] {
                for y in OWN_ROWS..POCKET_ROWS {
                    let row = &mut cells[y * WIDTH + lane * (WIDTH / 2)..][..WIDTH / 2];
                    if y >= RIVER_ROWS.0 && y < RIVER_ROWS.1 && !wades {
                        let bridge = BRIDGE_COLUMNS[lane];
                        row[bridge.0 - lane * (WIDTH / 2)..bridge.1 - lane * (WIDTH / 2)].fill(1);
                    } else {
                        row.fill(1);
                    }
                }
            }
        }
    }
    if ability.is_some_and(|a| a.1) {
        legal[ABILITY_ACTION] = 1;
        choice = true;
    }
    rows.choice = choice;
}

/// The opponent's hidden state, for the training-only critic.
pub fn privileged(eng: &Engine, seat: usize, rows: &mut Rows) {
    let enemy = 1 - seat;
    rows.priv_global[0] = eng.elixir[enemy].to_f64() as f32;
    rows.priv_global[1] = 0.0;
    rows.priv_global[2] = 0.0;
    for (slot, &card) in eng.hand(enemy).iter().enumerate() {
        rows.priv_hand[slot] = token(card);
    }
    for (slot, &card) in eng.deck[enemy].iter().enumerate().take(DECK_ROWS) {
        rows.priv_deck[slot] = token(card);
    }
}
