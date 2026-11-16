//! The cr-rl-arena simulator behind cfbot's batch-environment interface.
//!
//! The engine is compiled from the source files of a cr-rl-arena checkout (the
//! `#[path]` modules below), unchanged, so a battle here is the battle the
//! tournament server plays at that commit.
//!
//! * [`VecEnv`]: a batch of battles for training. Rows are `env * 2 + seat`.
//!   Both players see themselves at the bottom, as on the server. Finished
//!   battles reset inside [`VecEnv::step`]; the returned observation then
//!   belongs to the new episode while `reward`, `done` and `outcome` describe
//!   the old one.
//! * [`Featurizer`]: the same observation rows from the server's `obs`
//!   messages, for the tournament client.
//!
//! Both feed the same `View` to the same `Tracker` and `featurize`.

#[allow(dead_code, unused, clippy::all)]
#[path = "../cr-rl-arena/gym/src/arena.rs"]
mod arena;
#[allow(dead_code, unused, clippy::all)]
#[path = "../cr-rl-arena/gym/src/cards.rs"]
mod cards;
#[allow(dead_code, unused, clippy::all)]
#[path = "../cr-rl-arena/gym/src/engine.rs"]
mod engine;
#[allow(dead_code, unused, clippy::all)]
#[path = "../cr-rl-arena/gym/src/entity.rs"]
mod entity;
#[allow(dead_code, unused, clippy::all)]
#[path = "../cr-rl-arena/gym/src/fixed.rs"]
mod fixed;
#[allow(dead_code, unused, clippy::all)]
#[path = "../cr-rl-arena/gym/src/rng.rs"]
mod rng;

mod decks;
mod feat;
mod track;
mod view;

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use rayon::prelude::*;

use cards::Library;
use decks::{mix64, Deck, DeckSampler, Rng};
use engine::{Action, Engine, Outcome};
use feat::Rows;
use fixed::{Fx, ONE};
use track::{elixir_rate, Meta, Tracker, ABILITY_SLOT, MAX_ELIXIR};
use view::View;

struct Context {
    meta: Arc<Meta>,
    sampler: DeckSampler,
    decision_ticks: u32,
    /// Extra decisions a step may fast-forward while neither player can do
    /// anything but wait.
    max_skip: u32,
    master_seed: u64,
}

struct QueuedMatch {
    decks: [Deck; 2],
    seed: Option<u64>,
}

/// A flat action index as the engine's action for `seat`, and what the tracker
/// should remember of it: slot and tile centre in the seat's own frame.
fn decode(index: i64, seat: usize) -> Option<(Action, (i32, f32, f32))> {
    let index = usize::try_from(index).ok()?;
    if index == feat::ABILITY_ACTION {
        return Some((Action { slot: ABILITY_SLOT, x: Fx(0), y: Fx(0) }, (ABILITY_SLOT, 0.0, 0.0)));
    }
    if !(1..feat::PLAY_ACTION_COUNT).contains(&index) {
        return None;
    }
    let slot = (index - 1) / feat::CELLS;
    let cell = (index - 1) % feat::CELLS;
    let (cx, cy) = ((cell % feat::WIDTH) as i64, (cell / feat::WIDTH) as i64);
    let x = Fx::from_raw((2 * cx + 1) * ONE / 2);
    let own_y = Fx::from_raw((2 * cy + 1) * ONE / 2);
    // The top seat's tile is mirrored onto the board, as the server does.
    let y = if seat == 0 { own_y } else { arena::mirror_y(own_y) };
    Some((Action { slot: slot as i32, x, y }, (slot as i32, cx as f32 + 0.5, cy as f32 + 0.5)))
}

struct Slot {
    index: usize,
    episode: u64,
    eng: Engine,
    decks: [Deck; 2],
    views: [View; 2],
    trackers: [Tracker; 2],
    rows: [Rows; 2],
    queue: VecDeque<QueuedMatch>,
    queued: bool,
    /// The seat holding a focus deck in the running episode, or -1.
    focus_seat: i8,
    decisions: u32,
    /// Troop health after the previous decision, by entity id: remaining
    /// health, elixir value per health point, owner.
    health: HashMap<u32, (f32, f32, usize)>,
    health_next: HashMap<u32, (f32, f32, usize)>,
    // Outputs of the latest step, describing the episode that was stepped.
    reward: [f32; 2],
    trade: [f32; 2],
    leak: [f32; 2],
    ticks: u32,
    done: bool,
    truncated: bool,
    outcome: i8,
    verdict: [i8; 2],
    final_towers: [[f32; 6]; 2],
    episode_decisions: u32,
    episode_ticks: u32,
    finished_decks: [Deck; 2],
    finished_queued: bool,
    crashes: Vec<[Deck; 2]>,
}

impl Slot {
    fn new(context: &Context, index: usize, max_entities: usize) -> Slot {
        let fallback = context.sampler.fallback;
        let meta = &context.meta;
        let mut slot = Slot {
            index,
            episode: 0,
            eng: Engine::new(meta.lib.clone()),
            decks: [fallback; 2],
            views: [View::default(), View::default()],
            trackers: [Tracker::new(meta, fallback), Tracker::new(meta, fallback)],
            rows: [Rows::new(max_entities), Rows::new(max_entities)],
            queue: VecDeque::new(),
            queued: false,
            focus_seat: -1,
            decisions: 0,
            health: HashMap::new(),
            health_next: HashMap::new(),
            reward: [0.0; 2],
            trade: [0.0; 2],
            leak: [0.0; 2],
            ticks: 0,
            done: false,
            truncated: false,
            outcome: -1,
            verdict: [0; 2],
            final_towers: [[0.0; 6]; 2],
            episode_decisions: 0,
            episode_ticks: 0,
            finished_decks: [fallback; 2],
            finished_queued: false,
            crashes: Vec::new(),
        };
        slot.start_episode(context);
        slot.featurize(context);
        slot
    }

    /// Begin this slot's next episode: a queued match if one is waiting,
    /// otherwise freshly sampled decks.
    fn start_episode(&mut self, context: &Context) {
        let mut seed = mix64(
            context.master_seed
                ^ mix64((self.index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15))
                ^ mix64(self.episode.wrapping_add(0xd1b5_4a32_d192_ed03)),
        );
        self.episode += 1;
        self.queued = false;
        let mut rng = Rng(seed);
        self.focus_seat = -1;
        self.decks = match self.queue.pop_front() {
            Some(request) => {
                seed = request.seed.unwrap_or(seed);
                self.queued = true;
                request.decks
            }
            None => {
                let (decks, focus) = context.sampler.sample(&mut rng);
                self.focus_seat = focus.map_or(-1, |seat| seat as i8);
                decks
            }
        };
        self.eng.reset(seed, Some([self.decks[0].to_vec(), self.decks[1].to_vec()]));
        self.decisions = 0;
        self.health.clear();
        for seat in 0..2 {
            self.trackers[seat].reset(self.decks[seat]);
        }
        self.observe(context);
    }

    /// Refresh both views from the engine and let the trackers see them.
    fn observe(&mut self, context: &Context) {
        for seat in 0..2 {
            view::from_engine(&self.eng, seat, &mut self.views[seat]);
            self.trackers[seat].update(&context.meta, &self.views[seat], i64::from(context.decision_ticks));
        }
    }

    fn featurize(&mut self, context: &Context) {
        for seat in 0..2 {
            feat::featurize(&context.meta, &self.views[seat], &self.trackers[seat], &mut self.rows[seat]);
            feat::privileged(&self.eng, seat, &mut self.rows[seat]);
        }
    }

    fn has_choice(&self, context: &Context) -> bool {
        (0..2).any(|seat| feat::has_choice(&context.meta, &self.views[seat], &self.trackers[seat]))
    }

    /// Credit each side with the elixir value of health its opponent's troops lost.
    fn account_damage(&mut self, meta: &Meta) {
        self.health_next.clear();
        for e in &self.eng.entities {
            let value = meta.body_value[e.def];
            let full = (e.max_hp.max(0) + meta.lib.defs[e.def].shield.max(0)) as f32;
            if e.is_tower || value <= 0.0 || full <= 0.0 {
                continue;
            }
            let remaining = (e.hp.max(0) + e.shield_hp.max(0)) as f32;
            self.health_next.insert(e.id, (remaining, value / full, e.owner as usize));
        }
        for (id, &(before, rate, owner)) in &self.health {
            let after = self.health_next.get(id).map_or(0.0, |entry| entry.0);
            self.trade[1 - owner] += (before - after).max(0.0) * rate;
        }
        std::mem::swap(&mut self.health, &mut self.health_next);
    }

    /// Advance one decision interval: the actions go in on its first tick.
    /// Returns what the engine said of each action.
    fn advance(&mut self, context: &Context, actions: [Action; 2]) -> [u8; 2] {
        let mut verdicts = [view::ACT_NONE; 2];
        for step in 0..context.decision_ticks {
            let rate = elixir_rate(self.eng.tick) as f32;
            let acts = if step == 0 { actions } else { [Action::noop(); 2] };
            let outcome = self.eng.step(acts);
            self.ticks += 1;
            for seat in 0..2 {
                if self.eng.elixir[seat].to_f64() >= MAX_ELIXIR {
                    self.leak[seat] += rate;
                }
            }
            self.observe(context);
            if step == 0 {
                verdicts = [self.views[0].last_action, self.views[1].last_action];
            }
            if outcome != Outcome::Running {
                break;
            }
        }
        self.account_damage(&context.meta);
        match self.eng.outcome {
            Outcome::Running => {}
            Outcome::Win0 => {
                self.reward = [1.0, -1.0];
                self.outcome = 0;
            }
            Outcome::Win1 => {
                self.reward = [-1.0, 1.0];
                self.outcome = 1;
            }
            Outcome::Draw => self.outcome = 2,
        }
        self.done = self.eng.outcome != Outcome::Running;
        verdicts
    }

    /// Step, containing engine panics to this battle. A crashed episode ends as
    /// a truncated draw (outcome 3) and the battle is rebuilt from scratch.
    fn step(&mut self, context: &Context, indices: [i64; 2]) {
        let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.step_inner(context, indices);
        }));
        if stepped.is_ok() {
            return;
        }
        self.finished_decks = self.decks;
        self.finished_queued = self.queued;
        if self.crashes.len() < 64 {
            self.crashes.push(self.decks);
        }
        for _ in 0..8 {
            let rebuilt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.eng = Engine::new(context.meta.lib.clone());
                self.start_episode(context);
                self.featurize(context);
            }));
            if rebuilt.is_ok() {
                break;
            }
        }
        self.reward = [0.0; 2];
        self.trade = [0.0; 2];
        self.leak = [0.0; 2];
        self.ticks = context.decision_ticks;
        self.done = true;
        self.truncated = true;
        self.outcome = 3;
        self.verdict = [0; 2];
        self.final_towers = [[0.0; 6]; 2];
        self.episode_decisions = 0;
        self.episode_ticks = 0;
    }

    fn step_inner(&mut self, context: &Context, indices: [i64; 2]) {
        let mut actions = [Action::noop(); 2];
        let mut invalid = [false; 2];
        for seat in 0..2 {
            if indices[seat] == 0 {
                continue;
            }
            match decode(indices[seat], seat) {
                Some((action, (slot, x, y))) => {
                    actions[seat] = action;
                    self.trackers[seat].note_action(slot, x, y);
                }
                None => invalid[seat] = true,
            }
        }
        self.reward = [0.0; 2];
        self.trade = [0.0; 2];
        self.leak = [0.0; 2];
        self.ticks = 0;
        self.outcome = -1;
        self.truncated = false;
        let verdicts = self.advance(context, actions);
        self.decisions += 1;
        for seat in 0..2 {
            self.verdict[seat] = match verdicts[seat] {
                _ if invalid[seat] => 3,
                view::ACT_NONE => 0,
                view::ACT_OK => 1,
                _ => 2,
            };
        }
        // Nothing to decide while both players can only wait: keep the clock running.
        let mut skipped = 0;
        while !self.done && skipped < context.max_skip && !self.has_choice(context) {
            self.advance(context, [Action::noop(); 2]);
            skipped += 1;
        }
        for seat in 0..2 {
            self.final_towers[seat] = feat::tower_fractions(&context.meta, &self.views[seat]);
        }
        self.episode_decisions = 0;
        self.episode_ticks = 0;
        if self.done {
            self.episode_decisions = self.decisions;
            self.episode_ticks = self.eng.tick as u32;
            self.finished_decks = self.decks;
            self.finished_queued = self.queued;
            self.start_episode(context);
        }
        self.featurize(context);
    }
}

fn deck_tokens(deck: &Deck) -> impl Iterator<Item = i16> + '_ {
    deck.iter().map(|&card| (card + 1) as i16)
}

fn deck_from_tokens(sampler: &DeckSampler, tokens: &[usize]) -> PyResult<Deck> {
    if tokens.contains(&0) {
        return Err(PyValueError::new_err("card token 0 is not a card"));
    }
    let ids: Vec<usize> = tokens.iter().map(|&token| token - 1).collect();
    sampler.check(&ids).map_err(PyValueError::new_err)
}

fn pool_from_tokens(sampler: &DeckSampler, decks: &[Vec<usize>]) -> PyResult<Vec<Deck>> {
    decks.iter().map(|deck| deck_from_tokens(sampler, deck)).collect()
}

/// The built-in deck of the engine: its first eight non-spell cards.
fn default_deck(lib: &Library) -> Deck {
    let deck = Engine::new(lib.clone()).default_deck();
    let mut out = [0; 8];
    out.copy_from_slice(&deck[..8]);
    out
}

/// Static description of every array `VecEnv` and `Featurizer` return.
fn write_spec<'py>(
    py: Python<'py>,
    meta: &Meta,
    decision_ticks: u32,
    max_entities: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use feat::*;
    let lib = &meta.lib;
    let d = PyDict::new(py);
    d.set_item("simulator", "arena")?;
    d.set_item("schema_version", SCHEMA_VERSION)?;
    d.set_item("bot_schema", BOT_SCHEMA)?;
    d.set_item("num_actions", ACTION_COUNT)?;
    d.set_item("grid_width", WIDTH)?;
    d.set_item("grid_height", HEIGHT)?;
    d.set_item("play_action_count", PLAY_ACTION_COUNT)?;
    d.set_item("secondary_play_start", SECONDARY_PLAY_START)?;
    d.set_item("max_entities", max_entities)?;
    d.set_item("global_f", GLOBAL_F)?;
    d.set_item("entity_f", ENTITY_F)?;
    d.set_item("entity_x_f", ENTITY_X_F)?;
    d.set_item("hand_f", HAND_F)?;
    d.set_item("deck_f", DECK_F)?;
    d.set_item("deck_x_f", DECK_X_F)?;
    d.set_item("play_rows", PLAY_ROWS)?;
    d.set_item("play_f", PLAY_F)?;
    d.set_item("reveal_rows", REVEAL_ROWS)?;
    d.set_item("reveal_f", REVEAL_F)?;
    d.set_item("reveal_x_f", REVEAL_X_F)?;
    d.set_item("hazard_rows", HAZARD_ROWS)?;
    d.set_item("hazard_f", HAZARD_F)?;
    d.set_item("extra_f", EXTRA_F)?;
    d.set_item("ability_f", ABILITY_F)?;
    d.set_item("priv_global_f", PRIV_GLOBAL_F)?;
    d.set_item("card_f", CARD_F)?;
    d.set_item("unit_f", UNIT_F)?;
    d.set_item("n_cards", N_CARDS)?;
    d.set_item("n_units", N_UNITS)?;
    d.set_item("n_hazards", N_HAZARDS)?;
    d.set_item("decision_ticks", decision_ticks)?;
    d.set_item("regular_ticks", arena::REGULATION_TICKS)?;
    d.set_item("overtime_ticks", arena::OVERTIME_TICKS)?;
    d.set_item("double_elixir_tick", arena::DOUBLE_ELIXIR_AT)?;
    d.set_item("triple_elixir_tick", arena::REGULATION_TICKS)?;
    d.set_item("max_elixir", arena::MAX_ELIXIR)?;
    d.set_item("base_level", BASE_LEVEL as u32)?;
    d.set_item("tick_ms", 1000 / arena::TICKS_PER_SECOND)?;
    d.set_item("rules_id", "cr-rl-arena-level11")?;
    // A digest of the card data, so a checkpoint can tell it was trained on other cards.
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for byte in cards::DEFAULT_CARDS_JSON.bytes() {
        digest = (digest ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    d.set_item("rules_digest", format!("{digest:016x}"))?;
    let mut names = vec![String::new(); N_CARDS];
    let mut supported = vec![false; N_CARDS];
    for (id, def) in lib.defs.iter().enumerate() {
        names[id + 1] = def.name.clone();
    }
    for &id in &lib.deployable {
        supported[id + 1] = true;
    }
    d.set_item("card_names", names.clone())?;
    d.set_item("card_supported", supported)?;
    d.set_item("unit_names", names)?;
    d.set_item("default_deck", default_deck(lib).iter().map(|&c| lib.defs[c].name.clone()).collect::<Vec<_>>())?;
    d.set_item("card_table", card_table(lib).into_pyarray(py).reshape([N_CARDS, CARD_F])?)?;
    d.set_item("unit_table", unit_table(lib).into_pyarray(py).reshape([N_UNITS, UNIT_F])?)?;
    Ok(d)
}

/// The policy's planes for a list of row sets, as numpy arrays in `d`.
fn write_rows<'py>(py: Python<'py>, d: &Bound<'py, PyDict>, all: &[&Rows], private: bool) -> PyResult<()> {
    use feat::*;
    let rows = all.len();
    let cap = all[0].max_entities;
    // Entity rows are trimmed to the longest row in this batch, rounded up to a
    // multiple of 16 so the network sees only a few distinct shapes.
    let longest = all.iter().map(|r| r.n_entities).max().unwrap_or(0).max(1);
    let width = longest.div_ceil(16).saturating_mul(16).min(cap);
    let mut entities = Vec::with_capacity(rows * width * ENTITY_F);
    let mut entity_extras = Vec::with_capacity(rows * width * ENTITY_X_F);
    let mut entity_mask = Vec::with_capacity(rows * width);
    let mut overflow = Vec::with_capacity(rows);
    let mut entity_count = Vec::with_capacity(rows);
    let mut choice = Vec::with_capacity(rows);
    for r in all {
        entity_count.push(r.n_entities.min(width) as i64);
        choice.push(r.choice);
        entities.extend_from_slice(&r.entities[..width * ENTITY_F]);
        entity_extras.extend_from_slice(&r.entity_extras[..width * ENTITY_X_F]);
        entity_mask.extend((0..width).map(|index| u8::from(index < r.n_entities)));
        overflow.push(r.overflow);
    }
    d.set_item("entities", entities.into_pyarray(py).reshape([rows, width, ENTITY_F])?)?;
    d.set_item("entity_extras", entity_extras.into_pyarray(py).reshape([rows, width, ENTITY_X_F])?)?;
    d.set_item("entity_mask", entity_mask.into_pyarray(py).reshape([rows, width])?)?;
    d.set_item("entity_overflow", overflow.into_pyarray(py))?;
    d.set_item("hazard_overflow", vec![0_u32; rows].into_pyarray(py))?;
    d.set_item("entity_count", entity_count.into_pyarray(py))?;
    d.set_item("choice", choice.into_pyarray(py))?;

    macro_rules! gather {
        ($name:literal, $field:ident, $shape:expr) => {{
            let mut values = Vec::with_capacity(rows * all[0].$field.len());
            for r in all {
                values.extend_from_slice(&r.$field);
            }
            d.set_item($name, values.into_pyarray(py).reshape($shape)?)?;
        }};
    }
    macro_rules! mask {
        ($name:literal, $field:ident, $width:expr) => {{
            let mut values = Vec::with_capacity(rows * $width);
            for r in all {
                values.extend((0..$width).map(|index| u8::from(index < r.$field)));
            }
            d.set_item($name, values.into_pyarray(py).reshape([rows, $width])?)?;
        }};
    }
    gather!("global", global, [rows, GLOBAL_F]);
    gather!("hand", hand, [rows, 4, HAND_F]);
    gather!("abilities", abilities, [rows, ABILITY_ROWS, ABILITY_F]);
    gather!("own_deck", own_deck, [rows, DECK_ROWS, DECK_F]);
    gather!("deck_extras", deck_extras, [rows, DECK_ROWS, DECK_X_F]);
    gather!("recent_plays", plays, [rows, PLAY_ROWS, PLAY_F]);
    mask!("play_mask", n_plays, PLAY_ROWS);
    gather!("revealed_deck", reveals, [rows, REVEAL_ROWS, REVEAL_F]);
    gather!("reveal_extras", reveal_extras, [rows, REVEAL_ROWS, REVEAL_X_F]);
    mask!("reveal_mask", n_reveals, REVEAL_ROWS);
    // The server sends no spell areas or projectiles: the hazard plane is empty.
    d.set_item(
        "hazards",
        vec![0.0_f32; rows * HAZARD_ROWS * HAZARD_F].into_pyarray(py).reshape([rows, HAZARD_ROWS, HAZARD_F])?,
    )?;
    d.set_item("hazard_mask", vec![0_u8; rows * HAZARD_ROWS].into_pyarray(py).reshape([rows, HAZARD_ROWS])?)?;
    gather!("extras", extras, [rows, EXTRA_F]);
    gather!("legal", legal, [rows, ACTION_COUNT]);
    if private {
        gather!("priv_global", priv_global, [rows, PRIV_GLOBAL_F]);
        gather!("priv_hand", priv_hand, [rows, 4]);
        gather!("priv_deck", priv_deck, [rows, DECK_ROWS]);
    }
    Ok(())
}

#[pyclass]
struct VecEnv {
    context: Context,
    pool: rayon::ThreadPool,
    slots: Vec<Slot>,
    max_entities: usize,
}

#[pymethods]
impl VecEnv {
    /// Card ids are card tokens: the library id plus one, as listed by `spec()`.
    #[new]
    #[pyo3(signature = (num_envs, workers=0, decision_ticks=6, seed=0, max_entities=128,
                        deck_pool=None, pool_prob=1.0, mirror_decks=false, exclude_cards=None,
                        max_skip=8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        workers: usize,
        decision_ticks: u32,
        seed: u64,
        max_entities: usize,
        deck_pool: Option<Vec<Vec<usize>>>,
        pool_prob: f64,
        mirror_decks: bool,
        exclude_cards: Option<Vec<usize>>,
        max_skip: u32,
    ) -> PyResult<Self> {
        if num_envs == 0 || decision_ticks == 0 || max_entities < 16 {
            return Err(PyValueError::new_err(
                "num_envs and decision_ticks must be positive and max_entities at least 16",
            ));
        }
        let lib = Library::default_lib();
        if lib.defs.len() + 1 > feat::N_CARDS {
            return Err(PyValueError::new_err("the card library has outgrown the vocabulary"));
        }
        let excluded = exclude_cards.unwrap_or_default();
        let allowed: Vec<usize> =
            lib.deployable.iter().copied().filter(|&card| !excluded.contains(&(card + 1))).collect();
        let mut sampler = DeckSampler::new(&lib, &allowed, mirror_decks, default_deck(&lib));
        let pool_decks = pool_from_tokens(&sampler, &deck_pool.unwrap_or_default())?;
        if pool_decks.is_empty() && allowed.len() < 8 {
            return Err(PyValueError::new_err("fewer than eight cards remain after exclusions"));
        }
        sampler.set_pool(pool_decks, &[]);
        sampler.pool_prob = pool_prob;
        let workers = if workers == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            workers
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("cfbot-arena-{index}"))
            .build()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let context = Context {
            meta: Arc::new(Meta::new(lib)),
            sampler,
            decision_ticks,
            max_skip,
            master_seed: seed,
        };
        let slots = (0..num_envs).map(|index| Slot::new(&context, index, max_entities)).collect();
        Ok(Self { context, pool, slots, max_entities })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.slots.len()
    }

    /// Deck sources for new episodes: the chance of a pool deck, the chance of
    /// a pool deck with one to three cards replaced, and a random deck otherwise.
    #[pyo3(signature = (pool_prob, mutant_prob=0.0))]
    fn set_mix(&mut self, pool_prob: f64, mutant_prob: f64) {
        self.context.sampler.pool_prob = pool_prob;
        self.context.sampler.mutant_prob = mutant_prob;
    }

    /// Replace the deck pool. `weights` are relative; omitted means uniform.
    #[pyo3(signature = (decks, weights=None))]
    fn set_pool(&mut self, decks: Vec<Vec<usize>>, weights: Option<Vec<f64>>) -> PyResult<()> {
        let pool = pool_from_tokens(&self.context.sampler, &decks)?;
        let weights = weights.unwrap_or_default();
        if !weights.is_empty() && weights.len() != pool.len() {
            return Err(PyValueError::new_err("one weight per deck"));
        }
        self.context.sampler.set_pool(pool, &weights);
        Ok(())
    }

    /// Decks one seat (chosen at random) of a new battle plays with chance
    /// `prob`, whatever the sampler gives the other seat. Every observation
    /// carries `focus_seat`: per battle, the seat holding a focus deck in the
    /// running episode, or -1. A trainer that controls one seat takes that one.
    fn set_focus(&mut self, decks: Vec<Vec<usize>>, prob: f64) -> PyResult<()> {
        let focus = pool_from_tokens(&self.context.sampler, &decks)?;
        self.context.sampler.set_focus(focus, prob);
        Ok(())
    }

    /// Queue an exact match as battle `env`'s next episode. Queued matches run
    /// in order before sampled ones and are flagged in the step output.
    #[pyo3(signature = (env, blue, red, seed=None))]
    fn queue_match(&mut self, env: usize, blue: Vec<usize>, red: Vec<usize>, seed: Option<u64>) -> PyResult<()> {
        let decks = [
            deck_from_tokens(&self.context.sampler, &blue)?,
            deck_from_tokens(&self.context.sampler, &red)?,
        ];
        let slot = self.slots.get_mut(env).ok_or_else(|| PyValueError::new_err("no such battle"))?;
        slot.queue.push_back(QueuedMatch { decks, seed });
        Ok(())
    }

    /// `count` random decks under the sampler's playability rules, as card tokens.
    fn random_decks(&self, count: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut rng = Rng(mix64(seed ^ 0x5eed_dec5));
        (0..count)
            .map(|_| self.context.sampler.random(&mut rng).iter().map(|card| card + 1).collect())
            .collect()
    }

    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = write_spec(py, &self.context.meta, self.context.decision_ticks, self.max_entities)?;
        d.set_item("pool_decks", self.context.sampler.pool_len())?;
        d.set_item("random_deck_cards", self.context.sampler.card_count())?;
        Ok(d)
    }

    /// Restart every battle. A new seed also restarts each battle's episode stream.
    #[pyo3(signature = (seed=None))]
    fn reset<'py>(&mut self, py: Python<'py>, seed: Option<u64>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(seed) = seed {
            self.context.master_seed = seed;
        }
        let context = &self.context;
        let restart = seed.is_some();
        let slots = &mut self.slots;
        py.detach(|| {
            self.pool.install(|| {
                slots.par_iter_mut().for_each(|slot| {
                    if restart {
                        slot.episode = 0;
                    }
                    slot.start_episode(context);
                    slot.featurize(context);
                });
            });
        });
        let d = PyDict::new(py);
        self.write_observation(py, &d)?;
        Ok(d)
    }

    /// Advance every battle by one decision. `actions` holds one flat action
    /// index per row. Index 0 waits; an undecodable index waits and is reported
    /// as verdict 3. `ticks` is the game time each battle actually advanced.
    fn step<'py>(&mut self, py: Python<'py>, actions: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyDict>> {
        let actions = actions.as_slice()?.to_vec();
        if actions.len() != self.slots.len() * 2 {
            return Err(PyValueError::new_err(format!(
                "expected {} actions, received {}",
                self.slots.len() * 2,
                actions.len()
            )));
        }
        let context = &self.context;
        let slots = &mut self.slots;
        py.detach(|| {
            self.pool.install(|| {
                slots
                    .par_iter_mut()
                    .zip(actions.par_chunks_exact(2))
                    .for_each(|(slot, pair)| slot.step(context, [pair[0], pair[1]]));
            });
        });

        let d = PyDict::new(py);
        self.write_observation(py, &d)?;
        let envs = self.slots.len();
        let rows = envs * 2;
        let mut reward = Vec::with_capacity(rows);
        let mut trade = Vec::with_capacity(rows);
        let mut leak = Vec::with_capacity(rows);
        let mut done = Vec::with_capacity(rows);
        let mut truncated = Vec::with_capacity(rows);
        let mut verdict = Vec::with_capacity(rows);
        let mut ticks = Vec::with_capacity(rows);
        let mut towers = Vec::with_capacity(rows * 6);
        let mut outcome = Vec::with_capacity(envs);
        let mut decisions = Vec::with_capacity(envs);
        let mut episode_ticks = Vec::with_capacity(envs);
        let mut decks = Vec::with_capacity(envs * 16);
        let mut queued = Vec::with_capacity(envs);
        for slot in &self.slots {
            decks.extend(slot.finished_decks.iter().flat_map(deck_tokens));
            queued.push(slot.finished_queued);
            reward.extend_from_slice(&slot.reward);
            trade.extend_from_slice(&slot.trade);
            leak.extend_from_slice(&slot.leak);
            done.extend_from_slice(&[slot.done; 2]);
            truncated.extend_from_slice(&[slot.truncated; 2]);
            verdict.extend_from_slice(&slot.verdict);
            ticks.extend_from_slice(&[slot.ticks as f32; 2]);
            towers.extend_from_slice(&slot.final_towers[0]);
            towers.extend_from_slice(&slot.final_towers[1]);
            outcome.push(slot.outcome);
            decisions.push(slot.episode_decisions);
            episode_ticks.push(slot.episode_ticks);
        }
        d.set_item("reward", reward.into_pyarray(py))?;
        d.set_item("trade", trade.into_pyarray(py))?;
        d.set_item("leak", leak.into_pyarray(py))?;
        d.set_item("done", done.into_pyarray(py))?;
        d.set_item("truncated", truncated.into_pyarray(py))?;
        d.set_item("verdict", verdict.into_pyarray(py))?;
        d.set_item("ticks", ticks.into_pyarray(py))?;
        d.set_item("final_towers", towers.into_pyarray(py).reshape([rows, 6])?)?;
        d.set_item("outcome", outcome.into_pyarray(py))?;
        d.set_item("episode_decisions", decisions.into_pyarray(py))?;
        d.set_item("episode_ticks", episode_ticks.into_pyarray(py))?;
        d.set_item("decks", decks.into_pyarray(py).reshape([envs, 2, 8])?)?;
        d.set_item("queued", queued.into_pyarray(py))?;
        Ok(d)
    }

    /// The sampler never draws a deck the engine refuses.
    fn deck_retries(&self) -> u64 {
        0
    }

    /// Deck pairs (card names) of episodes that ended in an engine panic.
    fn crashes(&self) -> Vec<(Vec<String>, Vec<String>)> {
        let lib = &self.context.meta.lib;
        let names = |deck: &Deck| deck.iter().map(|&card| lib.defs[card].name.clone()).collect::<Vec<_>>();
        self.slots
            .iter()
            .flat_map(|slot| slot.crashes.iter())
            .map(|decks| (names(&decks[0]), names(&decks[1])))
            .collect()
    }

    /// Battle `env` as the server would describe it to `seat`: tick, elixir,
    /// hand, next card and the entity list. For tests of the tournament client.
    fn debug_view<'py>(&self, py: Python<'py>, env: usize, seat: usize) -> PyResult<Bound<'py, PyDict>> {
        let slot = self.slots.get(env).ok_or_else(|| PyValueError::new_err("no such battle"))?;
        let lib = &self.context.meta.lib;
        let view = &slot.views[seat.min(1)];
        let d = PyDict::new(py);
        d.set_item("tick", view.tick)?;
        d.set_item("elixir", view.elixir)?;
        d.set_item("hand", view.hand.iter().map(|&c| lib.defs[c].name.clone()).collect::<Vec<_>>())?;
        d.set_item("next_card", lib.defs[view.next].name.clone())?;
        let ents: Vec<_> = view
            .ents
            .iter()
            .map(|e| (u8::from(!e.mine), lib.defs[e.def].name.clone(), e.x, e.y, e.hp, e.max_hp, e.id))
            .collect();
        d.set_item("entities", ents)?;
        Ok(d)
    }
}

impl VecEnv {
    fn write_observation<'py>(&self, py: Python<'py>, d: &Bound<'py, PyDict>) -> PyResult<()> {
        let all: Vec<&Rows> = self.slots.iter().flat_map(|slot| slot.rows.iter()).collect();
        let focus: Vec<i8> = self.slots.iter().map(|slot| slot.focus_seat).collect();
        d.set_item("focus_seat", focus.into_pyarray(py))?;
        write_rows(py, d, &all, true)
    }
}

/// One player's observation rows from the tournament server's `obs` messages.
///
///     f = Featurizer(deck)
///     f.update(obs)              # every tick, in order
///     rows = f.observe()         # when the policy is to decide
///     action = f.act(flat)       # the reply's action; None for wait
#[pyclass]
struct Featurizer {
    meta: Arc<Meta>,
    tracker: Tracker,
    view: View,
    rows: Rows,
    decision_ticks: u32,
    max_entities: usize,
}

impl Featurizer {
    fn deck(meta: &Meta, names: &[String]) -> PyResult<Deck> {
        let lib = &meta.lib;
        let ids = names
            .iter()
            .map(|name| lib.id_of(name).ok_or_else(|| PyValueError::new_err(format!("unknown card '{name}'"))))
            .collect::<PyResult<Vec<_>>>()?;
        let allowed: Vec<usize> = lib.deployable.clone();
        DeckSampler::new(lib, &allowed, false, default_deck(lib)).check(&ids).map_err(PyValueError::new_err)
    }

    fn card(&self, name: &str) -> PyResult<usize> {
        self.meta.lib.id_of(name).ok_or_else(|| PyValueError::new_err(format!("unknown card '{name}'")))
    }
}

#[pymethods]
impl Featurizer {
    #[new]
    #[pyo3(signature = (deck, decision_ticks=6, max_entities=128))]
    fn new(deck: Vec<String>, decision_ticks: u32, max_entities: usize) -> PyResult<Self> {
        let meta = Arc::new(Meta::new(Library::default_lib()));
        let deck = Self::deck(&meta, &deck)?;
        Ok(Self {
            tracker: Tracker::new(&meta, deck),
            meta,
            view: View::default(),
            rows: Rows::new(max_entities),
            decision_ticks,
            max_entities,
        })
    }

    /// Start a new match with `deck`.
    fn reset(&mut self, deck: Vec<String>) -> PyResult<()> {
        let deck = Self::deck(&self.meta, &deck)?;
        self.tracker.reset(deck);
        Ok(())
    }

    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        write_spec(py, &self.meta, self.decision_ticks, self.max_entities)
    }

    /// Take in one `obs` message. Call it for every tick, in order.
    fn update(&mut self, obs: &Bound<'_, PyDict>) -> PyResult<()> {
        let item = |key: &str| {
            obs.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("the observation has no '{key}'")))
        };
        self.view.tick = item("tick")?.extract()?;
        self.view.elixir = item("elixir")?.extract()?;
        let hand: Vec<String> = item("hand")?.extract()?;
        if hand.len() != 4 {
            return Err(PyValueError::new_err("the hand is four cards"));
        }
        for (slot, name) in hand.iter().enumerate() {
            self.view.hand[slot] = self.card(name)?;
        }
        self.view.next = self.card(&item("next_card")?.extract::<String>()?)?;
        self.view.last_action = match item("last_action")?.extract::<String>()?.as_str() {
            "none" => view::ACT_NONE,
            "ok" => view::ACT_OK,
            _ => view::ACT_IGNORED,
        };
        self.view.ents.clear();
        // Each entity: [owner, card, x, y, hp, max_hp, flying, is_tower, deploying, id, half_w, half_h]
        for entity in item("entities")?.try_iter()? {
            let entity = entity?;
            let name: String = entity.get_item(1)?.extract()?;
            self.view.ents.push(view::Ent {
                id: entity.get_item(9)?.extract()?,
                mine: entity.get_item(0)?.extract::<u8>()? == 0,
                def: self.card(&name)?,
                x: entity.get_item(2)?.extract()?,
                y: entity.get_item(3)?.extract()?,
                hp: entity.get_item(4)?.extract()?,
                max_hp: entity.get_item(5)?.extract()?,
                deploying: entity.get_item(8)?.extract()?,
                tower: entity.get_item(7)?.extract()?,
            });
        }
        self.view.sort();
        self.tracker.update(&self.meta, &self.view, i64::from(self.decision_ticks));
        Ok(())
    }

    /// The policy's planes for the latest observation, one row each.
    fn observe<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        feat::featurize(&self.meta, &self.view, &self.tracker, &mut self.rows);
        let d = PyDict::new(py);
        write_rows(py, &d, &[&self.rows], false)?;
        Ok(d)
    }

    /// Whether anything but waiting is possible on the latest observation.
    fn has_choice(&self) -> bool {
        feat::has_choice(&self.meta, &self.view, &self.tracker)
    }

    /// The server action for flat action `index`, `[slot, x, y]`, or `None` to
    /// wait. Call it once for the action that is sent.
    fn act(&mut self, index: i64) -> Option<(i32, f64, f64)> {
        if index == 0 {
            return None;
        }
        let (_, (slot, x, y)) = decode(index, 0)?;
        self.tracker.note_action(slot, x, y);
        Some((slot, f64::from(x), f64::from(y)))
    }

    /// Card names by token, as in `spec()["card_names"]`.
    fn card_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut names = vec![String::new()];
        names.extend(self.meta.lib.defs.iter().map(|def| def.name.clone()));
        PyList::new(py, names)
    }
}

#[pymodule]
fn cfbot_arena(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<VecEnv>()?;
    module.add_class::<Featurizer>()?;
    module.add("ABILITY_SLOT", ABILITY_SLOT)?;
    module.add("DEFAULT_CARDS_JSON", cards::DEFAULT_CARDS_JSON)?;
    Ok(())
}
