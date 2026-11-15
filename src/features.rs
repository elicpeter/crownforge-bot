//! Turns one player's observation into fixed-width rows for the network.
//!
//! The policy tensors follow the simulator's own feature schema version 2
//! (`crownforge_train::encoding::policy` on crownforge main): the same planes,
//! column order, tokens and units. Once the bridge builds against a simulator
//! that ships that encoder, `featurize` can call it instead and the network is
//! unaffected. Until then the public play history comes from [`Tracker`].
//!
//! Three more planes are owned by the bot:
//! * `hazards` - spell areas, death bombs, containers and tunnel trails, which
//!   a player sees on the board but schema 2 does not carry;
//! * `extras` - the opponent's elixir counted from public plays;
//! * `priv_*` - the opponent's hidden hand, deck and elixir. Training-only
//!   inputs for the critic; they must never reach the policy.

use std::cmp::Ordering;

use crownforge_core::{
    Battle, CardId, EntityKind, Point, Ruleset, Team, UnitDefId,
    ability::AbilityButton,
    arena,
    rules::{CardKind, ELIXIR_UNIT, UnitDefKind, UnitStats},
};
use crownforge_train::PlayerObservation;

use crate::tracker::{PlayEvent, Tracker};

pub const SCHEMA_VERSION: u32 = 2;
pub const GLOBAL_F: usize = 4;
pub const ENTITY_F: usize = 24;
pub const HAND_F: usize = 4;
pub const DECK_ROWS: usize = 8;
pub const DECK_F: usize = 2;
pub const PLAY_ROWS: usize = 32;
pub const PLAY_F: usize = 7;
pub const REVEAL_ROWS: usize = 8;
pub const REVEAL_F: usize = 3;
pub const HAZARD_ROWS: usize = 16;
pub const HAZARD_F: usize = 5;
pub const EXTRA_F: usize = 2;
pub const PRIV_GLOBAL_F: usize = 3;
pub const CARD_F: usize = 28;
pub const UNIT_F: usize = 16;

/// Vocabulary sizes, with headroom so they stay stable as content is added.
/// Token 0 is always "none" or padding; a card or unit is `index() + 1`.
pub const N_CARDS: usize = 160;
pub const N_UNITS: usize = 160;
/// Hazard tokens: areas from 1, then bombs, containers and tunnel trails.
const HAZARD_BOMB: usize = 49;
const HAZARD_CONTAINER: usize = 57;
const HAZARD_TRAIL: usize = 58;
pub const N_HAZARDS: usize = 64;

pub fn card_token(card: CardId) -> f32 {
    (card.index() + 1) as f32
}

fn unit_token(unit: UnitDefId) -> f32 {
    let index = unit.index() + 1;
    if index < N_UNITS { index as f32 } else { 0.0 }
}

fn log_scaled(value: i32, scale: f32) -> f32 {
    (value.max(0) as f32).ln_1p() / scale
}

fn flag(value: bool) -> f32 {
    f32::from(u8::from(value))
}

fn position(point: Point, fraction: Point) -> [f32; 2] {
    [
        (point.x as f32 + fraction.x as f32 / 18.0) / arena::WIDTH as f32,
        (point.y as f32 + fraction.y as f32 / 18.0) / arena::HEIGHT as f32,
    ]
}

fn health_fraction(current: i32, maximum: i32) -> f32 {
    if maximum > 0 {
        (current as f32 / maximum as f32).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn write_stats(stats: &UnitStats, out: &mut [f32]) {
    out[0] = log_scaled(stats.hp, 9.0);
    out[1] = log_scaled(stats.shield_hp, 9.0);
    out[2] = log_scaled(stats.damage, 8.0);
    out[3] = stats.range as f32 / 10_000.0;
    out[4] = log_scaled(stats.speed, 8.0);
    out[5] = stats.attack_period_ms as f32 / 3_000.0;
    out[6] = flag(stats.attacks_air);
    out[7] = flag(stats.attacks_ground);
    out[8] = flag(stats.is_flying());
    out[9] = flag(stats.targets_buildings_only);
    out[10] = stats.attack_splash_radius as f32 / 3_000.0;
    out[11] = log_scaled(stats.mass, 5.0);
    out[12] = stats.sight as f32 / 10_000.0;
    out[13] = stats.radius as f32 / 2_000.0;
}

/// Static per-unit stats at the ruleset's base level, indexed by unit token.
pub fn unit_table(rules: &Ruleset) -> Vec<f32> {
    let mut table = vec![0.0; N_UNITS * UNIT_F];
    for unit in rules.units() {
        let token = unit_token(unit.id) as usize;
        if token == 0 {
            continue;
        }
        let row = &mut table[token * UNIT_F..][..UNIT_F];
        write_stats(&unit.stats, row);
        row[14] = flag(matches!(unit.kind, UnitDefKind::Building { .. }));
        row[15] = 1.0;
    }
    table
}

/// Static per-card features at the ruleset's base level, indexed by card token.
pub fn card_table(rules: &Ruleset) -> Vec<f32> {
    let mut table = vec![0.0; N_CARDS * CARD_F];
    for &card in &CardId::CURRENT_ALL {
        if !rules.supports_card(card) {
            continue;
        }
        let def = rules.card(card);
        let row = &mut table[(card.index() + 1) * CARD_F..][..CARD_F];
        row[0] = 1.0;
        row[1] = f32::from(def.cost) / 10.0;
        row[25] = flag(def.omit_from_starting_hand);
        let mut unit = None;
        match &def.kind {
            CardKind::Troop { members, .. } => {
                row[2] = 1.0;
                row[7] = members.len() as f32 / 16.0;
                unit = members.first().map(|member| member.unit);
            }
            CardKind::Building { unit: body, .. } => {
                row[3] = 1.0;
                unit = Some(*body);
            }
            CardKind::Spell(spell) => {
                row[4] = 1.0;
                row[22] = log_scaled(spell.damage, 8.0);
                row[23] = log_scaled(spell.tower_damage, 8.0);
                row[24] = spell.radius as f32 / 5_000.0;
            }
            CardKind::StrikingSpell(_)
            | CardKind::CentreStrikeSpell(_)
            | CardKind::RollingSpell(_)
            | CardKind::AreaSpell { .. } => row[4] = 1.0,
            CardKind::Mirror(_) => row[5] = 1.0,
            CardKind::Variant(_) => row[6] = 1.0,
        }
        if let Some(unit) = unit {
            write_stats(&rules.unit(unit).stats, &mut row[8..8 + 14]);
        }
    }
    table
}

/// Elixir one deployed body of a card is worth, for trade accounting. Bodies
/// that destroy themselves by design (spirits, wall breakers, timed troops)
/// are worth nothing, so losing them is not counted as the opponent's gain.
pub fn body_value(rules: &Ruleset, card: CardId, unit: Option<UnitDefId>) -> f32 {
    if !rules.supports_card(card) {
        return 0.0;
    }
    if let Some(unit) = unit {
        let def = rules.unit(unit);
        if def.stats.kamikaze || def.troop_lifetime_ms.is_some() {
            return 0.0;
        }
    }
    let def = rules.card(card);
    let bodies = match &def.kind {
        CardKind::Troop { members, .. } => members.len().max(1),
        _ => 1,
    };
    f32::from(def.cost) / bodies as f32
}

/// Output rows for the two players of one battle. Every buffer holds two rows.
pub struct Rows {
    pub max_entities: usize,
    pub n_entities: [usize; 2],
    pub overflow: [u32; 2],
    pub n_plays: [usize; 2],
    pub n_reveals: [usize; 2],
    pub n_hazards: [usize; 2],
    pub global: Vec<f32>,
    pub entities: Vec<f32>,
    pub hand: Vec<f32>,
    pub own_deck: Vec<f32>,
    pub plays: Vec<f32>,
    pub reveals: Vec<f32>,
    pub hazards: Vec<f32>,
    pub extras: Vec<f32>,
    pub legal: Vec<u8>,
    pub priv_global: Vec<f32>,
    pub priv_hand: Vec<f32>,
    pub priv_deck: Vec<f32>,
    scratch: Vec<[f32; ENTITY_F]>,
}

impl Rows {
    pub fn new(max_entities: usize) -> Self {
        Self {
            max_entities,
            n_entities: [0; 2],
            overflow: [0; 2],
            n_plays: [0; 2],
            n_reveals: [0; 2],
            n_hazards: [0; 2],
            global: vec![0.0; 2 * GLOBAL_F],
            entities: vec![0.0; 2 * max_entities * ENTITY_F],
            hand: vec![0.0; 2 * 4 * HAND_F],
            own_deck: vec![0.0; 2 * DECK_ROWS * DECK_F],
            plays: vec![0.0; 2 * PLAY_ROWS * PLAY_F],
            reveals: vec![0.0; 2 * REVEAL_ROWS * REVEAL_F],
            hazards: vec![0.0; 2 * HAZARD_ROWS * HAZARD_F],
            extras: vec![0.0; 2 * EXTRA_F],
            legal: vec![0; 2 * crownforge_train::grid::ACTION_COUNT],
            priv_global: vec![0.0; 2 * PRIV_GLOBAL_F],
            priv_hand: vec![0.0; 2 * 4],
            priv_deck: vec![0.0; 2 * DECK_ROWS],
            scratch: Vec::with_capacity(max_entities),
        }
    }
}

fn compare_rows(left: &[f32; ENTITY_F], right: &[f32; ENTITY_F]) -> Ordering {
    for (left, right) in left.iter().zip(right) {
        let order = left.total_cmp(right);
        if order != Ordering::Equal {
            return order;
        }
    }
    Ordering::Equal
}

/// Own then enemy tower health fractions: king, left, right for each side.
pub fn tower_fractions(observation: &PlayerObservation) -> [f32; 6] {
    let mut out = [0.0; 6];
    for entity in &observation.entities {
        let lane = match entity.kind {
            EntityKind::King => 0,
            EntityKind::Princess if entity.position.x < arena::WIDTH / 2 => 1,
            EntityKind::Princess => 2,
            _ => continue,
        };
        let index = if entity.friendly { lane } else { 3 + lane };
        out[index] = health_fraction(entity.hp, entity.max_hp);
    }
    out
}

pub fn featurize(
    rules: &Ruleset,
    battle: &Battle,
    observation: &PlayerObservation,
    trackers: &[Tracker; 2],
    rows: &mut Rows,
) {
    let team = observation.perspective;
    let side = team.index();
    let timing = rules.timing();
    let own = &trackers[side];
    let opponent = &trackers[1 - side];

    // ---- global: elapsed seconds, elixir, ready abilities, history enabled ----
    let global = &mut rows.global[side * GLOBAL_F..][..GLOBAL_F];
    global[0] = (observation.tick / (1000 / crownforge_core::TICK_MS)) as f32;
    global[1] = observation.elixir as f32 / ELIXIR_UNIT as f32;
    global[2] = observation
        .abilities
        .iter()
        .flatten()
        .filter(|ability| ability.button == AbilityButton::Ready)
        .count() as f32;
    global[3] = 1.0;

    // ---- entities, in a canonical order that hides creation order ----
    let cap = rows.max_entities;
    rows.scratch.clear();
    let mut overflow = 0;
    for entity in &observation.entities {
        if rows.scratch.len() == cap {
            overflow += 1;
            continue;
        }
        let [x, y] = position(entity.position, entity.position_fraction);
        rows.scratch.push([
            match entity.kind {
                EntityKind::King => 1.0,
                EntityKind::Princess => 2.0,
                EntityKind::Troop => 3.0,
                EntityKind::Building => 4.0,
            },
            flag(entity.friendly),
            entity.definition.map_or(0.0, unit_token),
            f32::from(entity.level.get()),
            x,
            y,
            health_fraction(entity.hp, entity.max_hp),
            health_fraction(entity.shield, entity.max_shield),
            entity
                .tower_choice
                .map_or(0.0, |choice| f32::from(choice as u8) + 1.0),
            entity.tower_ammo.map_or(0.0, |ammo| f32::from(ammo) + 1.0),
            flag(entity.cloned),
            flag(entity.raged),
            flag(entity.slowed),
            flag(entity.cursed),
            flag(entity.goblin_cursed),
            flag(entity.vined),
            flag(entity.charged),
            flag(entity.jumping),
            flag(entity.dashing),
            f32::from(entity.damage_stage),
            flag(entity.invisible),
            flag(entity.hidden),
            flag(entity.enchanted),
            flag(entity.hp > 0),
        ]);
    }
    rows.scratch.sort_unstable_by(compare_rows);
    let entities = &mut rows.entities[side * cap * ENTITY_F..][..cap * ENTITY_F];
    entities.fill(0.0);
    for (row, values) in entities.chunks_exact_mut(ENTITY_F).zip(&rows.scratch) {
        row.copy_from_slice(values);
    }
    rows.n_entities[side] = rows.scratch.len();
    rows.overflow[side] = overflow;

    // ---- hand: spent card, displayed card, level, cost ----
    let hand = &mut rows.hand[side * 4 * HAND_F..][..4 * HAND_F];
    hand.fill(0.0);
    for slot in 0..4 {
        let row = &mut hand[slot * HAND_F..][..HAND_F];
        row[0] = observation.hand[slot].map_or(0.0, card_token);
        if let Some(view) = observation.resolved_hand[slot] {
            row[1] = card_token(view.card);
            row[2] = f32::from(view.level.get());
            row[3] = f32::from(view.cost);
        }
    }

    // ---- own deck: card, level ----
    let level = f32::from(rules.level());
    let own_deck = &mut rows.own_deck[side * DECK_ROWS * DECK_F..][..DECK_ROWS * DECK_F];
    for (row, &card) in own_deck.chunks_exact_mut(DECK_F).zip(&own.deck) {
        row[0] = card_token(card);
        row[1] = level;
    }

    // ---- recent plays by both sides, oldest first ----
    let plays = &mut rows.plays[side * PLAY_ROWS * PLAY_F..][..PLAY_ROWS * PLAY_F];
    plays.fill(0.0);
    let mut merged: [Option<(bool, &PlayEvent)>; PLAY_ROWS] = [None; PLAY_ROWS];
    let mut count = 0;
    let mut own_events = own.events.iter().rev().peekable();
    let mut opponent_events = opponent.events.iter().rev().peekable();
    while count < PLAY_ROWS {
        let take_own = match (own_events.peek(), opponent_events.peek()) {
            (Some(a), Some(b)) => a.tick >= b.tick,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let event = if take_own {
            own_events.next()
        } else {
            opponent_events.next()
        }
        .expect("peeked event");
        merged[count] = Some((take_own, event));
        count += 1;
    }
    for (row, entry) in plays
        .chunks_exact_mut(PLAY_F)
        .zip(merged[..count].iter().rev())
    {
        let (friendly, event) = entry.expect("merged play");
        let [x, y] = position(team.orient(event.position), Point::default());
        row[0] = flag(friendly);
        row[1] = card_token(event.card);
        row[2] = f32::from(event.level);
        row[3] = 1.0;
        row[4] = x;
        row[5] = y;
        row[6] = event.tick as f32;
    }
    rows.n_plays[side] = count;

    // ---- opponent cards seen so far, in first-seen order ----
    let reveals = &mut rows.reveals[side * REVEAL_ROWS * REVEAL_F..][..REVEAL_ROWS * REVEAL_F];
    reveals.fill(0.0);
    for index in 0..DECK_ROWS {
        let rank = opponent.reveal_rank[index];
        if rank == u8::MAX {
            continue;
        }
        let row = &mut reveals[usize::from(rank) * REVEAL_F..][..REVEAL_F];
        row[0] = card_token(opponent.deck[index]);
        row[1] = opponent.last_tick[index] as f32;
        row[2] = f32::from(opponent.times[index]);
    }
    rows.n_reveals[side] = usize::from(opponent.revealed);

    // ---- hazards: token, friendly, x, y, radius ----
    let hazards = &mut rows.hazards[side * HAZARD_ROWS * HAZARD_F..][..HAZARD_ROWS * HAZARD_F];
    hazards.fill(0.0);
    let mut used = 0;
    let mut push = |token: usize, friendly: bool, point: [f32; 2], radius: f32| {
        if used < HAZARD_ROWS {
            hazards[used * HAZARD_F..][..HAZARD_F].copy_from_slice(&[
                token as f32,
                flag(friendly),
                point[0],
                point[1],
                radius,
            ]);
            used += 1;
        }
    };
    for area in &observation.areas {
        push(
            (area.definition.index() + 1).min(HAZARD_BOMB - 1),
            area.friendly,
            position(area.position, area.position_fraction),
            area.radius as f32 / arena::WIDTH as f32,
        );
    }
    for bomb in &observation.bombs {
        push(
            HAZARD_BOMB + bomb.definition.index(),
            bomb.friendly,
            position(bomb.position, bomb.position_fraction),
            0.0,
        );
    }
    for container in &observation.containers {
        push(
            HAZARD_CONTAINER,
            container.friendly,
            position(container.position, container.position_fraction),
            0.0,
        );
    }
    for trail in &observation.underground_trails {
        push(
            HAZARD_TRAIL,
            false,
            position(trail.position, Point::default()),
            0.0,
        );
    }
    rows.n_hazards[side] = used;

    // ---- extras: opponent elixir counted from public plays ----
    let extras = &mut rows.extras[side * EXTRA_F..][..EXTRA_F];
    extras[0] = opponent.est_elixir as f32 / ELIXIR_UNIT as f32;
    extras[1] = flag(opponent.est_elixir >= timing.max_elixir);

    // ---- legal actions ----
    let actions = crownforge_train::grid::ACTION_COUNT;
    let legal = &mut rows.legal[side * actions..][..actions];
    if observation.legal_actions.len() == actions {
        legal.copy_from_slice(&observation.legal_actions);
    } else {
        legal.fill(0);
        legal[0] = 1;
    }

    // ---- privileged: the opponent's hidden state ----
    let enemy: Team = team.opponent();
    let priv_global = &mut rows.priv_global[side * PRIV_GLOBAL_F..][..PRIV_GLOBAL_F];
    priv_global[0] = battle.elixir(enemy) as f32 / ELIXIR_UNIT as f32;
    priv_global[1] = battle.hand_refill_in(enemy) as f32;
    priv_global[2] = battle.hand_refill_in(team) as f32;
    let priv_hand = &mut rows.priv_hand[side * 4..][..4];
    for (slot, card) in battle.hand(enemy).iter().enumerate() {
        priv_hand[slot] = card.map_or(0.0, card_token);
    }
    let priv_deck = &mut rows.priv_deck[side * DECK_ROWS..][..DECK_ROWS];
    for (slot, &card) in opponent.deck.iter().enumerate() {
        priv_deck[slot] = card_token(card);
    }
}
