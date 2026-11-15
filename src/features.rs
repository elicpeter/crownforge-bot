//! Turns one player's observation into fixed-width rows for the network.
//!
//! The shared policy planes follow the simulator's feature schema version 2
//! (`crownforge_train::encoding::policy`): the same planes, column order,
//! tokens and units. Play history and revealed cards come from the simulator's
//! public play journal, which hides what an opponent's play does not show.
//!
//! The bot adds its own planes (`BOT_SCHEMA` versions them):
//! * `entity_extras` - deploy and stun timers and velocity per unit;
//! * `abilities` - state of the two ability buttons;
//! * `deck_extras`, `reveal_extras` - card cycle position counted from plays;
//! * `hazards` - spell areas, death bombs, containers and tunnel trails;
//! * `extras` - counted opponent elixir, sub-second time, own refill timer;
//! * `priv_*` - the opponent's hidden hand, deck and elixir. Training-only
//!   inputs for the critic; they must never reach the policy.

use std::{cmp::Ordering, collections::HashMap};

use crownforge_core::{
    Battle, CardId, EntityKind, Point, Ruleset, Team, UnitDefId,
    ability::AbilityButton,
    arena,
    rules::{CardKind, ELIXIR_UNIT, UnitDefKind, UnitStats},
};
use crownforge_train::PlayerObservation;

use crate::tracker::{QUEUE_LEN, Tracker};

pub const SCHEMA_VERSION: u32 = 2;
/// Version of the bot-owned planes below. Checkpoints record it.
pub const BOT_SCHEMA: u32 = 3;
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
pub const EXTRA_F: usize = 6;
pub const ENTITY_X_F: usize = 5;
pub const ABILITY_ROWS: usize = 2;
pub const ABILITY_F: usize = 9;
pub const DECK_X_F: usize = 3;
pub const REVEAL_X_F: usize = 2;
const ENTITY_ROW: usize = ENTITY_F + ENTITY_X_F;
const TICKS_PER_SECOND: f32 = 1000.0 / crownforge_core::TICK_MS as f32;
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

/// Elixir one body of a card is worth, for trade accounting. Only bodies the
/// card itself deploys count, each at an equal share of the cost: units spawned
/// later (Witch skeletons, Golemites, Graveyard) are worth nothing, or one card
/// would be credited many times over. Bodies that destroy themselves by design
/// (spirits, wall breakers, timed troops) are worth nothing either.
pub fn body_value(rules: &Ruleset, card: CardId, unit: Option<UnitDefId>) -> f32 {
    let Some(unit) = unit else {
        return 0.0;
    };
    if !rules.supports_card(card) {
        return 0.0;
    }
    let body = rules.unit(unit);
    if body.stats.kamikaze || body.troop_lifetime_ms.is_some() {
        return 0.0;
    }
    let def = rules.card(card);
    match &def.kind {
        CardKind::Troop { members, .. } if members.iter().any(|member| member.unit == unit) => {
            f32::from(def.cost) / members.len() as f32
        }
        _ => 0.0,
    }
}

/// Where each visible unit was at the previous observation, per viewer, keyed
/// by creation sequence. Used only to derive velocity; never exported.
#[derive(Default)]
pub struct Motion {
    last: HashMap<u64, [f32; 2]>,
    next: HashMap<u64, [f32; 2]>,
}

impl Motion {
    pub fn clear(&mut self) {
        self.last.clear();
    }
}

/// Output rows for the two players of one battle. Every buffer holds two rows.
pub struct Rows {
    pub max_entities: usize,
    pub n_entities: [usize; 2],
    pub overflow: [u32; 2],
    pub n_plays: [usize; 2],
    pub n_reveals: [usize; 2],
    pub n_hazards: [usize; 2],
    pub hazard_overflow: [u32; 2],
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
    pub hazards: Vec<f32>,
    pub extras: Vec<f32>,
    pub legal: Vec<u8>,
    pub priv_global: Vec<f32>,
    pub priv_hand: Vec<f32>,
    pub priv_deck: Vec<f32>,
    scratch: Vec<[f32; ENTITY_ROW]>,
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
            hazard_overflow: [0; 2],
            global: vec![0.0; 2 * GLOBAL_F],
            entities: vec![0.0; 2 * max_entities * ENTITY_F],
            entity_extras: vec![0.0; 2 * max_entities * ENTITY_X_F],
            abilities: vec![0.0; 2 * ABILITY_ROWS * ABILITY_F],
            hand: vec![0.0; 2 * 4 * HAND_F],
            own_deck: vec![0.0; 2 * DECK_ROWS * DECK_F],
            deck_extras: vec![0.0; 2 * DECK_ROWS * DECK_X_F],
            plays: vec![0.0; 2 * PLAY_ROWS * PLAY_F],
            reveals: vec![0.0; 2 * REVEAL_ROWS * REVEAL_F],
            reveal_extras: vec![0.0; 2 * REVEAL_ROWS * REVEAL_X_F],
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

/// Cycle features for a deck card last played `since` plays ago.
fn cycle_features(since: Option<u32>) -> (f32, bool) {
    match since {
        Some(since) => (since.min(8) as f32 / 8.0, since < QUEUE_LEN),
        None => (1.0, false),
    }
}

pub fn featurize(
    rules: &Ruleset,
    battle: &Battle,
    observation: &PlayerObservation,
    trackers: &[Tracker; 2],
    rows: &mut Rows,
    motion: &mut Motion,
    elapsed_ticks: u32,
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
    global[3] = flag(observation.public_history.is_some());

    // ---- entities, in a canonical order that hides creation order ----
    let cap = rows.max_entities;
    rows.scratch.clear();
    motion.next.clear();
    let seconds = elapsed_ticks as f32 / TICKS_PER_SECOND;
    let mut overflow = 0;
    for entity in &observation.entities {
        if rows.scratch.len() == cap {
            overflow += 1;
            continue;
        }
        let [x, y] = position(entity.position, entity.position_fraction);
        // Velocity in tiles per second, from where this viewer last saw the unit.
        let mut velocity = [0.0; 2];
        if seconds > 0.0
            && let Some(previous) = motion.last.get(&entity.sequence)
        {
            velocity = [
                ((x - previous[0]) * arena::WIDTH as f32 / seconds / 3.0).clamp(-1.0, 1.0),
                ((y - previous[1]) * arena::HEIGHT as f32 / seconds / 3.0).clamp(-1.0, 1.0),
            ];
        }
        motion.next.insert(entity.sequence, [x, y]);
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
            // Bot-owned columns from here.
            flag(entity.ready_in > 0),
            (entity.ready_in as f32 / TICKS_PER_SECOND / 3.0).min(1.0),
            (entity.stun_remaining as f32 / TICKS_PER_SECOND / 3.0).min(1.0),
            velocity[0],
            velocity[1],
        ]);
    }
    std::mem::swap(&mut motion.last, &mut motion.next);
    rows.scratch.sort_unstable_by(compare_rows);
    let entities = &mut rows.entities[side * cap * ENTITY_F..][..cap * ENTITY_F];
    let entity_extras = &mut rows.entity_extras[side * cap * ENTITY_X_F..][..cap * ENTITY_X_F];
    entities.fill(0.0);
    entity_extras.fill(0.0);
    for ((row, extra), values) in entities
        .chunks_exact_mut(ENTITY_F)
        .zip(entity_extras.chunks_exact_mut(ENTITY_X_F))
        .zip(&rows.scratch)
    {
        row.copy_from_slice(&values[..ENTITY_F]);
        extra.copy_from_slice(&values[ENTITY_F..]);
    }
    rows.n_entities[side] = rows.scratch.len();
    rows.overflow[side] = overflow;

    // ---- ability buttons: unit, usable, charges, then the button state ----
    let abilities =
        &mut rows.abilities[side * ABILITY_ROWS * ABILITY_F..][..ABILITY_ROWS * ABILITY_F];
    abilities.fill(0.0);
    for (row, ability) in abilities
        .chunks_exact_mut(ABILITY_F)
        .zip(&observation.abilities)
    {
        let Some(ability) = ability else {
            continue;
        };
        row[0] = unit_token(ability.definition);
        row[1] = flag(ability.usable);
        row[2] = ability.charges.clamp(0, 3) as f32 / 3.0;
        let state = match ability.button {
            AbilityButton::Ready => 3,
            AbilityButton::Deploying => 4,
            AbilityButton::Spent | AbilityButton::Absent | AbilityButton::Unknown => 5,
            AbilityButton::Pending | AbilityButton::Casting => 6,
            AbilityButton::Cooldown => 7,
            AbilityButton::Elixir => 8,
        };
        row[state] = 1.0;
    }

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

    // ---- own deck: card, level; then played, plays since, still queued ----
    let own_deck = &mut rows.own_deck[side * DECK_ROWS * DECK_F..][..DECK_ROWS * DECK_F];
    let deck_extras = &mut rows.deck_extras[side * DECK_ROWS * DECK_X_F..][..DECK_ROWS * DECK_X_F];
    for (index, (row, extra)) in own_deck
        .chunks_exact_mut(DECK_F)
        .zip(deck_extras.chunks_exact_mut(DECK_X_F))
        .enumerate()
    {
        row[0] = card_token(own.deck[index]);
        row[1] = f32::from(own.levels[index]);
        let since = own.plays_since(index);
        let (age, queued) = cycle_features(since);
        extra[0] = flag(since.is_some());
        extra[1] = age;
        extra[2] = flag(queued);
    }

    // ---- recent plays by both sides, oldest first, as the journal shows them ----
    let plays = &mut rows.plays[side * PLAY_ROWS * PLAY_F..][..PLAY_ROWS * PLAY_F];
    plays.fill(0.0);
    let reveals = &mut rows.reveals[side * REVEAL_ROWS * REVEAL_F..][..REVEAL_ROWS * REVEAL_F];
    let reveal_extras =
        &mut rows.reveal_extras[side * REVEAL_ROWS * REVEAL_X_F..][..REVEAL_ROWS * REVEAL_X_F];
    reveals.fill(0.0);
    reveal_extras.fill(0.0);
    rows.n_plays[side] = 0;
    rows.n_reveals[side] = 0;
    if let Some(history) = &observation.public_history {
        for (row, play) in plays.chunks_exact_mut(PLAY_F).zip(history.recent_plays()) {
            row[0] = flag(play.player == team);
            row[1] = card_token(play.card);
            row[2] = play.level.map_or(0.0, f32::from);
            if let Some(point) = play.position {
                let [x, y] = position(point, Point::default());
                row[3] = 1.0;
                row[4] = x;
                row[5] = y;
            }
            row[6] = play.tick as f32;
            rows.n_plays[side] += 1;
        }
        // ---- opponent cards seen so far, in first-seen order ----
        for ((row, extra), seen) in reveals
            .chunks_exact_mut(REVEAL_F)
            .zip(reveal_extras.chunks_exact_mut(REVEAL_X_F))
            .zip(history.revealed_opponent_deck())
        {
            row[0] = card_token(seen.card);
            row[1] = seen.last_seen_tick as f32;
            row[2] = seen.play_count as f32;
            let since = opponent
                .slot_of(seen.card)
                .and_then(|slot| opponent.plays_since(slot));
            let (age, queued) = cycle_features(since);
            extra[0] = age;
            // Not queued means the card is back in the opponent's hand or arriving.
            extra[1] = flag(since.is_some() && !queued);
            rows.n_reveals[side] += 1;
        }
    }

    // ---- hazards: token, friendly, x, y, radius ----
    let hazards = &mut rows.hazards[side * HAZARD_ROWS * HAZARD_F..][..HAZARD_ROWS * HAZARD_F];
    hazards.fill(0.0);
    let mut used = 0;
    let mut dropped = 0;
    let mut push = |token: usize, friendly: bool, point: [f32; 2], radius: f32| {
        if used == HAZARD_ROWS {
            dropped += 1;
        } else {
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
    rows.hazard_overflow[side] = dropped;

    // ---- extras: counted opponent elixir, time within the second, own hand
    // refill, and how far each side is into its first cycle ----
    let extras = &mut rows.extras[side * EXTRA_F..][..EXTRA_F];
    extras[0] = opponent.est_elixir as f32 / ELIXIR_UNIT as f32;
    extras[1] = flag(opponent.est_elixir >= timing.max_elixir);
    extras[2] = (observation.tick as f32 % TICKS_PER_SECOND) / TICKS_PER_SECOND;
    extras[3] = (observation.hand_refill_in as f32 / TICKS_PER_SECOND / 3.0).min(1.0);
    extras[4] = opponent.plays.min(QUEUE_LEN) as f32 / QUEUE_LEN as f32;
    extras[5] = own.plays.min(QUEUE_LEN) as f32 / QUEUE_LEN as f32;

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
