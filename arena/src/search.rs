//! Look-ahead from what one player sees.
//!
//! The server never shows the engine's state, only the view. `rebuild` makes an
//! engine that agrees with a view and the tracker's counts, the looking player
//! in seat 0, and `rollout` plays one candidate move in it and lets the board
//! run on for a while with nobody playing a card. The observation rows at the
//! end are what the policy's value head is asked about.
//!
//! What a rebuilt engine does not know, because the view does not say:
//! projectiles in the air, spells still falling, lingering spell areas, stuns
//! and slows, attack timers and targets, shields that have taken damage, the
//! opponent's hand. Units start a rollout with their
//! attack ready and no target, as if just arrived. Every candidate is rolled
//! out under the same omissions, so the comparison between candidates is fairer
//! than any one rollout is accurate.

use crate::engine::{Action, Engine, Outcome};
use crate::entity::Entity;
use crate::feat::{self, Rows};
use crate::fixed::{Fx, V2};
use crate::track::{Meta, Tracker, ABILITY_SLOT};
use crate::view::{self, View};

/// Units that were on the board when the rollout began keep their ids plus
/// this, so the engine's own counter (which restarts after the towers) cannot
/// hand a new unit the id of an old one.
const KEPT: u32 = 1 << 20;
/// Units born during the rollout show up under ids the tracker has never seen.
const BORN: u32 = 1 << 21;

/// What became of one candidate.
#[derive(Clone, Copy, Default)]
pub struct Leaf {
    /// The engine carried the move out (always true for waiting).
    pub ok: bool,
    /// The battle ended inside the rollout.
    pub done: bool,
    /// +1 won, -1 lost, 0 otherwise.
    pub result: f32,
}

fn unit(meta: &Meta, e: &view::Ent, tracker: &Tracker, tick: i64) -> Entity {
    let d = &meta.lib.defs[e.def];
    let age = tracker.mem.get(&e.id).map_or(0, |mem| tick - mem.first).max(0) as i32;
    let timer = |period: i32| period - age.rem_euclid(period.max(1));
    Entity {
        id: e.id + KEPT,
        owner: u8::from(!e.mine),
        def: e.def,
        pos: V2::new(Fx::from_f64(e.x), Fx::from_f64(e.y)),
        half: V2::default(),
        hp: e.hp,
        max_hp: e.max_hp,
        target: None,
        cooldown: 0,
        deploy: if e.deploying { (d.deploy_ticks - age).max(1) } else { 0 },
        lifetime: if d.lifetime_ticks > 0 { (d.lifetime_ticks - age).max(1) } else { d.lifetime_ticks },
        stun: 0,
        slow: 0,
        rage: 0,
        // The view shows health without the shield: assume it stands until the body is hurt.
        shield_hp: if e.hp >= e.max_hp { d.shield } else { 0 },
        reveal: 0,
        spawn_timer: if d.spawn_id.is_some() && d.spawn_interval_ticks > 0 {
            timer(d.spawn_interval_ticks)
        } else if d.elixir_per_pulse > 0 && d.elixir_pulse_ticks > 0 {
            timer(d.elixir_pulse_ticks)
        } else {
            -1
        },
        reflect_cd: 0,
        ability_cd: if e.mine { tracker.ability_wait(e.id, tick) } else { 0 },
        cloak: 0,
        deflect: 0,
        curse: 0,
        curse_spawn: -1,
        charge_dist: Fx(0),
        dash: 0,
        dash_cd: 0,
        ramp_timer: 0,
        ramp_target: 0,
        charge_shot: 0,
        is_tower: false,
        is_king: false,
        activated: true,
        engaged: false,
        revived: false,
        hatch: false,
        dead: false,
    }
}

/// Make `eng` agree with `view`, the viewer in seat 0. Returns the hand cards
/// that went in as another card, `(what the engine holds, the card it is)`.
pub fn rebuild(meta: &Meta, view: &View, tracker: &Tracker, eng: &mut Engine) -> Vec<(usize, usize)> {
    let lib = &meta.lib;
    // The viewer's cards in the order they will come: hand, next, the rest.
    // The engine keeps its evolution counters and the card Mirror copies to
    // itself, so a card whose next play is its evolved form goes in as that
    // form, and Mirror as the card it would copy (`rollout` charges the extra
    // elixir).
    let mut swapped = Vec::new();
    let mut own: Vec<usize> = view
        .hand
        .iter()
        .map(|&card| {
            let play = tracker.resolve(meta, card);
            if play.playable && play.def != card {
                swapped.push((play.def, card));
                play.def
            } else {
                card
            }
        })
        .collect();
    own.push(view.next);
    own.extend(tracker.rest_of_cycle(view));
    // The opponent plays nothing in a rollout; its deck only has to be a deck.
    let mut other: Vec<usize> = tracker.reveals.iter().map(|reveal| reveal.card).collect();
    other.truncate(8);
    for &card in tracker.deck.iter().chain(lib.deployable.iter()) {
        if other.len() == 8 {
            break;
        }
        if !other.contains(&card) && !lib.defs[card].is_champion {
            other.push(card);
        }
    }
    eng.reset(view.tick as u64, Some([tracker.deck.to_vec(), other.clone()]));
    eng.tick = view.tick;
    eng.elixir = [Fx::from_f64(view.elixir), Fx::from_f64(tracker.opp_elixir)];
    if own.len() == 8 {
        eng.queue[0] = own;
    }
    eng.queue[1] = other;

    // Towers: the reset put all six up, ids 1-3 the viewer's and 4-6 the other's.
    let mut down = [0i32; 2];
    let mut king_hurt = [false; 2];
    for tower in eng.entities.iter_mut() {
        match view.ents.iter().find(|e| e.tower && e.id == tower.id) {
            Some(seen) => {
                tower.hp = seen.hp;
                tower.max_hp = seen.max_hp;
                king_hurt[tower.owner as usize] |= tower.is_king && seen.hp < seen.max_hp;
            }
            None => {
                tower.dead = true;
                down[tower.owner as usize] += if tower.is_king { 3 } else { 1 };
            }
        }
    }
    eng.entities.retain(|tower| !tower.dead);
    for tower in eng.entities.iter_mut() {
        let owner = tower.owner as usize;
        if tower.is_king && (down[owner] > 0 || king_hurt[owner]) {
            tower.activated = true;
        }
    }
    eng.crowns = [down[1].min(3), down[0].min(3)];
    for e in view.ents.iter().filter(|e| !e.tower) {
        eng.entities.push(unit(meta, e, tracker, view.tick));
    }
    swapped
}

/// The view of seat 0 of a rebuilt engine, under the ids and card names the tracker knows.
fn look(eng: &Engine, swapped: &[(usize, usize)], out: &mut View) {
    view::from_engine(eng, 0, out);
    for card in out.hand.iter_mut() {
        if let Some(&(_, real)) = swapped.iter().find(|entry| entry.0 == *card) {
            *card = real;
        }
    }
    for e in out.ents.iter_mut().filter(|e| !e.tower) {
        e.id = if e.id >= KEPT { e.id - KEPT } else { e.id + BORN };
    }
    out.sort();
}

/// Play `action` (slot and tile centre in the viewer's frame; `None` waits) from
/// the position `view` shows and run `horizon` ticks on. `rows` receives the
/// observation at the end.
#[allow(clippy::too_many_arguments)]
pub fn rollout(
    meta: &Meta,
    view: &View,
    tracker: &Tracker,
    eng: &mut Engine,
    scratch: &mut View,
    action: Option<(i32, f32, f32)>,
    horizon: u32,
    decision_ticks: u32,
    rows: &mut Rows,
) -> Leaf {
    let swapped = rebuild(meta, view, tracker, eng);
    let mut tracker = tracker.clone();
    let mut first = Action::noop();
    if let Some((slot, x, y)) = action {
        first = Action { slot, x: Fx::from_f64(f64::from(x)), y: Fx::from_f64(f64::from(y)) };
        tracker.note_action(slot, x, y);
        // Mirror costs one more than the card that stands in for it.
        if (0..4).contains(&slot) && meta.lib.defs[view.hand[slot as usize]].is_mirror {
            eng.elixir[0] = eng.elixir[0] - Fx::from_int(1);
        }
    }
    let mut leaf = Leaf { ok: action.is_none(), ..Leaf::default() };
    for step in 0..horizon.max(1) {
        let acts = if step == 0 { [first, Action::noop()] } else { [Action::noop(); 2] };
        let outcome = eng.step(acts);
        look(eng, &swapped, scratch);
        if step == 0 && action.is_some() {
            leaf.ok = scratch.last_action == view::ACT_OK;
            if !leaf.ok {
                break;
            }
        }
        tracker.update(meta, scratch, i64::from(decision_ticks));
        if outcome != Outcome::Running {
            leaf.done = true;
            leaf.result = match outcome {
                Outcome::Win0 => 1.0,
                Outcome::Win1 => -1.0,
                _ => 0.0,
            };
            break;
        }
    }
    feat::featurize(meta, scratch, &tracker, rows);
    leaf
}

/// A flat action index as a rollout's move. `None` if it is no action at all.
pub fn candidate(index: i64) -> Option<Option<(i32, f32, f32)>> {
    if index == 0 {
        return Some(None);
    }
    let index = usize::try_from(index).ok()?;
    if index == feat::ABILITY_ACTION {
        return Some(Some((ABILITY_SLOT, 0.0, 0.0)));
    }
    if !(1..feat::PLAY_ACTION_COUNT).contains(&index) {
        return None;
    }
    let slot = (index - 1) / feat::CELLS;
    let cell = (index - 1) % feat::CELLS;
    Some(Some((slot as i32, (cell % feat::WIDTH) as f32 + 0.5, (cell / feat::WIDTH) as f32 + 0.5)))
}
