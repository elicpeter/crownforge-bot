//! Python bindings: a batch of Crownforge battles that returns network-ready
//! numpy arrays for both players of every battle.
//!
//! Rows are ordered `env * 2 + side` with Blue first. Finished battles reset
//! automatically inside [`VecEnv::step`]; the returned observation then belongs
//! to the new episode while `reward`, `done` and `outcome` describe the old one.

mod decks;
mod features;
mod tracker;

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use crownforge_core::{
    Action, ActionVerdict, Battle, CardId, EntityKind, MatchConfig, Outcome, Ruleset, Team,
    UnitDefId, rules::ELIXIR_UNIT,
};
use crownforge_train::{PlayerObservation, grid};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rayon::prelude::*;

use decks::{DeckSampler, Rng, mix64};
use features::{Motion, Rows};
use tracker::Tracker;

struct Context {
    rules: Arc<Ruleset>,
    sampler: DeckSampler,
    decision_ticks: u32,
    /// Extra decisions a step may fast-forward while neither player can do
    /// anything but wait.
    max_skip: u32,
    time_limit_ticks: Option<u32>,
    master_seed: u64,
    /// Inclusive range of per-card level offsets from the ruleset's level,
    /// drawn independently for every card of both decks.
    level_jitter: Option<(i32, i32)>,
}

/// A payment decoded before the step: the deck card that leaves the hand (or
/// `None` for an ability) and its elixir cost.
type Payment = (Option<CardId>, u32);

/// A match requested from Python for this slot's next episode.
struct QueuedMatch {
    decks: [[CardId; 8]; 2],
    seed: Option<u64>,
}

struct Slot {
    index: usize,
    episode: u64,
    battle: Battle,
    observations: [PlayerObservation; 2],
    trackers: [Tracker; 2],
    motion: [Motion; 2],
    queue: VecDeque<QueuedMatch>,
    /// Whether the running episode came from the queue.
    queued: bool,
    rows: Rows,
    decisions: u32,
    /// Troop health seen after the previous tick batch, keyed by creation
    /// sequence: remaining health, elixir value per health point, owner.
    health: HashMap<u64, (f32, f32, usize)>,
    health_next: HashMap<u64, (f32, f32, usize)>,
    /// Living units that pay elixir to the other side when they die (Elixir
    /// Golem and its parts), keyed by creation sequence: amount, owner.
    bounties: HashMap<u64, (i32, usize)>,
    bounties_next: HashMap<u64, (i32, usize)>,
    // Outputs of the latest step, describing the episode that was stepped.
    reward: [f32; 2],
    /// Elixir value of damage each side dealt to enemy troops this step.
    trade: [f32; 2],
    /// Elixir each side wasted at the cap this step.
    leak: [f32; 2],
    ticks: u32,
    done: bool,
    truncated: bool,
    outcome: i8,
    verdict: [i8; 2],
    final_towers: [[f32; 6]; 2],
    episode_decisions: u32,
    episode_ticks: u32,
    /// Decks of the episode that just finished, and whether it was queued.
    finished_decks: [[CardId; 8]; 2],
    finished_queued: bool,
    deck_retries: u32,
    crashes: Vec<[[CardId; 8]; 2]>,
}

fn has_choice(observation: &PlayerObservation) -> bool {
    observation
        .legal_actions
        .iter()
        .skip(1)
        .any(|&legal| legal != 0)
}

impl Slot {
    fn new(context: &Context, index: usize, max_entities: usize) -> Result<Self, String> {
        let battle = Battle::new(Arc::clone(&context.rules), MatchConfig::default())
            .map_err(|error| error.to_string())?;
        let observations = PlayerObservation::pair(&battle, true);
        let initial = context.rules.timing().initial_elixir;
        let mut slot = Self {
            index,
            episode: 0,
            battle,
            observations,
            trackers: [
                Tracker::new(CardId::DEFAULT_DECK, [context.rules.level(); 8], initial),
                Tracker::new(CardId::DEFAULT_DECK, [context.rules.level(); 8], initial),
            ],
            motion: [Motion::default(), Motion::default()],
            queue: VecDeque::new(),
            queued: false,
            rows: Rows::new(max_entities),
            decisions: 0,
            health: HashMap::new(),
            health_next: HashMap::new(),
            bounties: HashMap::new(),
            bounties_next: HashMap::new(),
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
            finished_decks: [CardId::DEFAULT_DECK; 2],
            finished_queued: false,
            deck_retries: 0,
            crashes: Vec::new(),
        };
        slot.start_episode(context);
        slot.featurize(context, 0);
        Ok(slot)
    }

    /// Begin this slot's next episode: a queued match if one is waiting,
    /// otherwise freshly sampled decks.
    fn start_episode(&mut self, context: &Context) {
        let seed = mix64(
            context.master_seed
                ^ mix64((self.index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15))
                ^ mix64(self.episode.wrapping_add(0xd1b5_4a32_d192_ed03)),
        );
        self.episode += 1;
        let base = context.rules.level();
        let mut decks = [CardId::DEFAULT_DECK; 2];
        let mut levels = [[base; 8]; 2];
        let mut started = false;
        self.queued = false;
        if let Some(request) = self.queue.pop_front() {
            let seed = request.seed.unwrap_or(seed);
            let mut rng = Rng(seed);
            let config = Self::config(seed, &mut rng, request.decks, None);
            if self.battle.reset(config).is_ok() {
                decks = request.decks;
                started = true;
                self.queued = true;
            } else {
                self.deck_retries += 1;
            }
        }
        let mut rng = Rng(seed);
        // Some card combinations are refused by the simulator; draw again.
        for _ in 0..32 {
            if started {
                break;
            }
            decks = context.sampler.sample(&mut rng);
            let card_levels = context.level_jitter.map(|(low, high)| {
                let mut drawn = [[base; 8]; 2];
                for level in drawn.iter_mut().flatten() {
                    let offset = low + rng.below((high - low + 1) as usize) as i32;
                    *level = (i32::from(base) + offset).clamp(1, 17) as u8;
                }
                drawn
            });
            let config = Self::config(seed, &mut rng, decks, card_levels);
            if self.battle.reset(config).is_ok() {
                levels = card_levels.unwrap_or(levels);
                started = true;
            } else {
                self.deck_retries += 1;
            }
        }
        if !started {
            decks = [CardId::DEFAULT_DECK; 2];
            levels = [[base; 8]; 2];
            self.battle
                .reset(MatchConfig {
                    seed,
                    track_public_plays: true,
                    ..MatchConfig::default()
                })
                .expect("default match configuration is always accepted");
        }
        let initial = context.rules.timing().initial_elixir;
        self.trackers[0].reset(decks[0], levels[0], initial);
        self.trackers[1].reset(decks[1], levels[1], initial);
        self.motion[0].clear();
        self.motion[1].clear();
        self.decisions = 0;
        self.health.clear();
        self.bounties.clear();
        PlayerObservation::refresh_pair(&self.battle, &mut self.observations, true);
    }

    fn config(
        seed: u64,
        rng: &mut Rng,
        decks: [[CardId; 8]; 2],
        card_levels: Option<[[u8; 8]; 2]>,
    ) -> MatchConfig {
        MatchConfig {
            seed,
            battle_random_state: ((rng.next() >> 33) as i32).max(1),
            decks,
            shuffle: true,
            native_deck_seeds: Some([
                ((rng.next() >> 33) as i32).max(1),
                ((rng.next() >> 33) as i32).max(1),
            ]),
            card_levels,
            track_public_plays: true,
            ..MatchConfig::default()
        }
    }

    /// Write both players' rows. `elapsed` is the game time since the rows
    /// were last written, for velocities; zero at the start of an episode.
    fn featurize(&mut self, context: &Context, elapsed: u32) {
        for (observation, motion) in self.observations.iter().zip(&mut self.motion) {
            features::featurize(
                &context.rules,
                &self.battle,
                observation,
                &self.trackers,
                &mut self.rows,
                motion,
                elapsed,
            );
        }
    }

    /// Credit each side with the elixir value of health its opponent's troops lost.
    fn account_damage(&mut self, rules: &Ruleset) {
        self.health_next.clear();
        self.bounties_next.clear();
        for entity in self.battle.entities() {
            if let Some(amount) = entity
                .definition
                .and_then(|unit| rules.unit(unit).death_reward)
            {
                self.bounties_next
                    .insert(entity.sequence, (amount, entity.team.index()));
            }
            let (EntityKind::Troop, Some(card)) = (entity.kind, entity.card) else {
                continue;
            };
            let full = (entity.max_hp.max(0) + entity.max_shield.max(0)) as f32;
            if full <= 0.0 {
                continue;
            }
            let value = features::body_value(rules, card, entity.definition);
            let remaining = (entity.hp.max(0) + entity.shield.max(0)) as f32;
            self.health_next.insert(
                entity.sequence,
                (remaining, value / full, entity.team.index()),
            );
        }
        for (sequence, &(before, rate, owner)) in &self.health {
            let after = self.health_next.get(sequence).map_or(0.0, |entry| entry.0);
            self.trade[1 - owner] += (before - after).max(0.0) * rate;
        }
        std::mem::swap(&mut self.health, &mut self.health_next);
        // A death that pays elixir is visible to both players; count it.
        let cap = rules.timing().max_elixir;
        for (sequence, &(amount, owner)) in &self.bounties {
            if !self.bounties_next.contains_key(sequence) {
                self.trackers[1 - owner].grant(amount, cap);
            }
        }
        std::mem::swap(&mut self.bounties, &mut self.bounties_next);
    }

    /// Advance the battle once and update everything derived from it.
    fn advance(
        &mut self,
        context: &Context,
        actions: [Action; 2],
        payments: [Option<Payment>; 2],
    ) -> [ActionVerdict; 2] {
        let timing = context.rules.timing();
        let start_tick = self.battle.tick();
        let ticks = context
            .time_limit_ticks
            .map_or(context.decision_ticks, |limit| {
                context.decision_ticks.min(limit.saturating_sub(start_tick))
            });
        let result = self.battle.step(actions, ticks);
        self.ticks += result.actual_ticks;
        let phase = if start_tick < timing.double_elixir_tick {
            0
        } else if start_tick < timing.triple_elixir_tick {
            1
        } else {
            2
        };
        for team in Team::ALL {
            let side = team.index();
            if result.verdicts[side] == ActionVerdict::Accepted
                && let Some((card, cost)) = payments[side]
            {
                match card {
                    Some(card) => self.trackers[side].record(card, cost as u8),
                    None => self.trackers[side].spend(cost),
                }
            }
            self.trackers[side].regenerate(timing, start_tick, result.actual_ticks);
            if self.battle.elixir(team) >= timing.max_elixir {
                self.leak[side] += (timing.elixir_per_tick[phase] as f32 / ELIXIR_UNIT as f32)
                    * result.actual_ticks as f32;
            }
        }
        self.account_damage(&context.rules);

        let terminated = result.outcome.is_some();
        self.truncated = !terminated
            && context
                .time_limit_ticks
                .is_some_and(|limit| self.battle.tick() >= limit);
        self.done = terminated || self.truncated;
        match result.outcome {
            Some(Outcome::Win(winner)) => {
                self.reward[winner.index()] = 1.0;
                self.reward[winner.opponent().index()] = -1.0;
                self.outcome = winner.index() as i8;
            }
            Some(Outcome::Draw) => self.outcome = 2,
            None => {}
        }
        PlayerObservation::refresh_pair(&self.battle, &mut self.observations, true);
        result.verdicts
    }

    /// Step, containing simulator panics to this battle. A crashed episode ends
    /// as a truncated draw (outcome 3) and the battle is rebuilt from scratch.
    fn step(&mut self, context: &Context, indices: [i64; 2]) {
        let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.step_inner(context, indices);
        }));
        if stepped.is_ok() {
            return;
        }
        self.finished_decks = [self.trackers[0].deck, self.trackers[1].deck];
        self.finished_queued = self.queued;
        if self.crashes.len() < 64 {
            self.crashes.push(self.finished_decks);
        }
        for _ in 0..8 {
            let rebuilt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.battle = Battle::new(Arc::clone(&context.rules), MatchConfig::default())
                    .expect("default match configuration is always accepted");
                self.observations = PlayerObservation::pair(&self.battle, true);
                self.start_episode(context);
                self.featurize(context, 0);
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
        let mut actions = [Action::Wait; 2];
        let mut payments: [Option<Payment>; 2] = [None; 2];
        let mut invalid = [false; 2];
        for side in 0..2 {
            let observation = &self.observations[side];
            let index = indices[side];
            if index == 0 {
                continue;
            }
            let Some(action) = usize::try_from(index)
                .ok()
                .and_then(|index| grid::decode_observed(index, observation).ok())
            else {
                invalid[side] = true;
                continue;
            };
            actions[side] = action;
            let (slot, view) = match action {
                Action::Play { slot, .. } => {
                    (slot, observation.resolved_hand.get(usize::from(slot)))
                }
                Action::PlayOption { slot, .. } => {
                    (slot, observation.secondary_hand.get(usize::from(slot)))
                }
                Action::Ability { target } => {
                    // An ability costs elixir too, and the opponent sees it cast.
                    let cost = observation
                        .abilities
                        .iter()
                        .flatten()
                        .find(|ability| ability.target == Some(target))
                        .and_then(|ability| context.rules.unit(ability.definition).ability)
                        .map_or(0, |ability| ability.cost);
                    payments[side] = Some((None, cost));
                    continue;
                }
                Action::Wait => continue,
            };
            let raw = observation.hand.get(usize::from(slot)).copied().flatten();
            if let (Some(raw), Some(Some(view))) = (raw, view) {
                payments[side] = Some((Some(raw), u32::from(view.cost)));
            }
        }

        self.reward = [0.0; 2];
        self.trade = [0.0; 2];
        self.leak = [0.0; 2];
        self.ticks = 0;
        self.outcome = -1;
        let verdicts = self.advance(context, actions, payments);
        self.decisions += 1;
        for side in 0..2 {
            self.verdict[side] = match verdicts[side] {
                _ if invalid[side] => 3,
                ActionVerdict::Waited => 0,
                ActionVerdict::Accepted => 1,
                ActionVerdict::Rejected(_) => 2,
            };
        }
        // Nothing to decide while both players can only wait: keep the clock running.
        let mut skipped = 0;
        while !self.done && skipped < context.max_skip && !self.observations.iter().any(has_choice)
        {
            self.advance(context, [Action::Wait; 2], [None; 2]);
            skipped += 1;
        }

        for side in 0..2 {
            self.final_towers[side] = features::tower_fractions(&self.observations[side]);
        }
        self.episode_decisions = 0;
        self.episode_ticks = 0;
        let mut elapsed = self.ticks;
        if self.done {
            self.episode_decisions = self.decisions;
            self.episode_ticks = self.battle.tick();
            self.finished_decks = [self.trackers[0].deck, self.trackers[1].deck];
            self.finished_queued = self.queued;
            self.start_episode(context);
            elapsed = 0;
        }
        self.featurize(context, elapsed);
    }
}

#[pyclass]
struct VecEnv {
    context: Context,
    pool: rayon::ThreadPool,
    slots: Vec<Slot>,
    max_entities: usize,
}

fn card_from_vocab(rules: &Ruleset, id: usize) -> PyResult<CardId> {
    CardId::CURRENT_ALL
        .iter()
        .copied()
        .find(|card| card.index() + 1 == id && rules.supports_card(*card))
        .ok_or_else(|| {
            PyValueError::new_err(format!("card id {id} is not supported by the ruleset"))
        })
}

impl VecEnv {
    /// Turn card tokens into a deck the simulator can run.
    fn deck_from_vocab(&self, deck: &[usize]) -> PyResult<[CardId; 8]> {
        deck_from_vocab(&self.context.rules, &self.context.sampler, deck)
    }
}

fn deck_from_vocab(
    rules: &Ruleset,
    sampler: &DeckSampler,
    deck: &[usize],
) -> PyResult<[CardId; 8]> {
    if deck.len() != 8 {
        return Err(PyValueError::new_err("each deck needs exactly eight cards"));
    }
    let mut cards = [CardId::Knight; 8];
    for (slot, &id) in cards.iter_mut().zip(deck) {
        *slot = card_from_vocab(rules, id)?;
    }
    for (index, card) in cards.iter().enumerate() {
        if cards[..index].contains(card) {
            return Err(PyValueError::new_err(format!(
                "deck lists {} twice",
                card.name()
            )));
        }
    }
    if sampler.abilities(&cards) > 2 {
        return Err(PyValueError::new_err(
            "a deck may hold at most two ability units (the simulator's ability slots)",
        ));
    }
    Ok(cards)
}

/// A pool deck: a runnable deck that uses no excluded card.
fn pool_deck_from_vocab(
    rules: &Ruleset,
    sampler: &DeckSampler,
    deck: &[usize],
) -> PyResult<[CardId; 8]> {
    let cards = deck_from_vocab(rules, sampler, deck)?;
    match cards.iter().find(|&&card| !sampler.allows(card)) {
        Some(card) => Err(PyValueError::new_err(format!(
            "pool deck uses excluded card {}",
            card.name()
        ))),
        None => Ok(cards),
    }
}

#[pymethods]
impl VecEnv {
    /// Card ids are card tokens: `CardId::index() + 1`, as listed by `spec()`.
    ///
    /// `rules` is `None` for the bundled level-11 rules, `"complete"` for the
    /// bundled bank with every card level, or a path to a rules document.
    /// `level_jitter` draws a level offset per card from an inclusive range and
    /// needs rules that hold those levels.
    #[new]
    #[pyo3(signature = (num_envs, workers=0, decision_ticks=6, seed=0, max_entities=128,
                        deck_pool=None, pool_prob=1.0, mirror_decks=false, exclude_cards=None,
                        time_limit_ticks=None, max_skip=8, rules=None, level_jitter=None))]
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
        time_limit_ticks: Option<u32>,
        max_skip: u32,
        rules: Option<String>,
        level_jitter: Option<(i32, i32)>,
    ) -> PyResult<Self> {
        if num_envs == 0 || decision_ticks == 0 || max_entities < 16 {
            return Err(PyValueError::new_err(
                "num_envs and decision_ticks must be positive and max_entities at least 16",
            ));
        }
        let rules = match rules.as_deref() {
            None | Some("level11") => Ruleset::level11(),
            Some("complete") => Ruleset::complete_level11(),
            Some(path) => {
                let bytes = std::fs::read(path)
                    .map_err(|error| PyValueError::new_err(format!("{path}: {error}")))?;
                Ruleset::from_json(&bytes)
                    .map_err(|error| PyValueError::new_err(format!("{path}: {error:?}")))?
            }
        };
        if level_jitter.is_some_and(|(low, high)| low > high) {
            return Err(PyValueError::new_err("level_jitter needs low <= high"));
        }
        let excluded = exclude_cards.unwrap_or_default();
        let allowed: Vec<CardId> = CardId::ALL
            .iter()
            .copied()
            .filter(|card| rules.supports_card(*card) && !excluded.contains(&(card.index() + 1)))
            .collect();
        let mut sampler = DeckSampler::new(&rules, &allowed, mirror_decks);
        let mut pool_decks = Vec::new();
        for deck in deck_pool.unwrap_or_default() {
            pool_decks.push(pool_deck_from_vocab(&rules, &sampler, &deck)?);
        }
        if pool_decks.is_empty() && allowed.len() < 8 {
            return Err(PyValueError::new_err(
                "fewer than eight cards remain after exclusions",
            ));
        }
        let workers = if workers == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            workers
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("cfbot-env-{index}"))
            .build()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        sampler.set_pool(pool_decks, &[]);
        sampler.pool_prob = pool_prob;
        let context = Context {
            sampler,
            rules,
            decision_ticks,
            max_skip,
            time_limit_ticks,
            master_seed: seed,
            level_jitter,
        };
        let slots = (0..num_envs)
            .map(|index| Slot::new(&context, index, max_entities))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            context,
            pool,
            slots,
            max_entities,
        })
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
        let pool = decks
            .iter()
            .map(|deck| pool_deck_from_vocab(&self.context.rules, &self.context.sampler, deck))
            .collect::<PyResult<Vec<_>>>()?;
        let weights = weights.unwrap_or_default();
        if !weights.is_empty() && weights.len() != pool.len() {
            return Err(PyValueError::new_err("one weight per deck"));
        }
        self.context.sampler.set_pool(pool, &weights);
        Ok(())
    }

    /// Queue an exact match as battle `env`'s next episode. Queued matches run
    /// in order before sampled ones and are flagged in the step output, so an
    /// evaluator can play a chosen deck pair once and count only that game.
    #[pyo3(signature = (env, blue, red, seed=None))]
    fn queue_match(
        &mut self,
        env: usize,
        blue: Vec<usize>,
        red: Vec<usize>,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let decks = [self.deck_from_vocab(&blue)?, self.deck_from_vocab(&red)?];
        let slot = self
            .slots
            .get_mut(env)
            .ok_or_else(|| PyValueError::new_err("no such battle"))?;
        slot.queue.push_back(QueuedMatch { decks, seed });
        Ok(())
    }

    /// `count` random decks under the sampler's playability rules, as card tokens.
    fn random_decks(&self, count: usize, seed: u64) -> Vec<Vec<usize>> {
        let mut rng = Rng(mix64(seed ^ 0x5eed_dec5));
        (0..count)
            .map(|_| {
                self.context
                    .sampler
                    .random(&mut rng)
                    .iter()
                    .map(|card| card.index() + 1)
                    .collect()
            })
            .collect()
    }

    /// Static description of every array this environment returns.
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use features::*;
        let rules = &self.context.rules;
        let timing = rules.timing();
        let d = PyDict::new(py);
        d.set_item("schema_version", SCHEMA_VERSION)?;
        d.set_item("bot_schema", BOT_SCHEMA)?;
        d.set_item("entity_x_f", ENTITY_X_F)?;
        d.set_item("ability_f", ABILITY_F)?;
        d.set_item("deck_x_f", DECK_X_F)?;
        d.set_item("reveal_x_f", REVEAL_X_F)?;
        let digest: String = rules
            .digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        d.set_item("rules_digest", digest)?;
        d.set_item("pool_decks", self.context.sampler.pool_len())?;
        d.set_item("num_actions", grid::ACTION_COUNT)?;
        d.set_item("grid_width", grid::WIDTH)?;
        d.set_item("grid_height", grid::HEIGHT)?;
        d.set_item("play_action_count", grid::PLAY_ACTION_COUNT)?;
        d.set_item("secondary_play_start", grid::SECONDARY_PLAY_START)?;
        d.set_item("max_entities", self.max_entities)?;
        d.set_item("global_f", GLOBAL_F)?;
        d.set_item("entity_f", ENTITY_F)?;
        d.set_item("hand_f", HAND_F)?;
        d.set_item("deck_f", DECK_F)?;
        d.set_item("play_rows", PLAY_ROWS)?;
        d.set_item("play_f", PLAY_F)?;
        d.set_item("reveal_rows", REVEAL_ROWS)?;
        d.set_item("reveal_f", REVEAL_F)?;
        d.set_item("hazard_rows", HAZARD_ROWS)?;
        d.set_item("hazard_f", HAZARD_F)?;
        d.set_item("extra_f", EXTRA_F)?;
        d.set_item("priv_global_f", PRIV_GLOBAL_F)?;
        d.set_item("card_f", CARD_F)?;
        d.set_item("unit_f", UNIT_F)?;
        d.set_item("n_cards", N_CARDS)?;
        d.set_item("n_units", N_UNITS)?;
        d.set_item("n_hazards", N_HAZARDS)?;
        d.set_item("decision_ticks", self.context.decision_ticks)?;
        d.set_item("regular_ticks", timing.regular_ticks)?;
        d.set_item("overtime_ticks", timing.overtime_ticks)?;
        d.set_item("double_elixir_tick", timing.double_elixir_tick)?;
        d.set_item("triple_elixir_tick", timing.triple_elixir_tick)?;
        d.set_item("max_elixir", timing.max_elixir / ELIXIR_UNIT)?;
        d.set_item("base_level", rules.level())?;
        d.set_item("tick_ms", crownforge_core::TICK_MS)?;
        d.set_item("rules_id", rules.id())?;
        d.set_item("random_deck_cards", self.context.sampler.card_count())?;
        let mut card_names = vec![String::new(); N_CARDS];
        let mut supported = vec![false; N_CARDS];
        for card in CardId::CURRENT_ALL {
            card_names[card.index() + 1] = card.name().to_owned();
            supported[card.index() + 1] = rules.supports_card(card);
        }
        d.set_item("card_names", card_names)?;
        d.set_item("card_supported", supported)?;
        let mut unit_names = vec![String::new(); N_UNITS];
        for unit in UnitDefId::ALL {
            if unit.index() + 1 < N_UNITS {
                unit_names[unit.index() + 1] = unit.name().to_owned();
            }
        }
        d.set_item("unit_names", unit_names)?;
        d.set_item(
            "card_table",
            card_table(rules)
                .into_pyarray(py)
                .reshape([N_CARDS, CARD_F])?,
        )?;
        d.set_item(
            "unit_table",
            unit_table(rules)
                .into_pyarray(py)
                .reshape([N_UNITS, UNIT_F])?,
        )?;
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
                    slot.featurize(context, 0);
                });
            });
        });
        let d = PyDict::new(py);
        self.write_observation(py, &d)?;
        Ok(d)
    }

    /// Advance every battle by one decision. `actions` holds one grid action
    /// index per row. Index 0 waits; an undecodable index waits and is reported
    /// as verdict 3. `ticks` is the game time each battle actually advanced.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
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
        let rows = self.slots.len() * 2;
        let mut reward = Vec::with_capacity(rows);
        let mut trade = Vec::with_capacity(rows);
        let mut leak = Vec::with_capacity(rows);
        let mut done = Vec::with_capacity(rows);
        let mut truncated = Vec::with_capacity(rows);
        let mut verdict = Vec::with_capacity(rows);
        let mut ticks = Vec::with_capacity(rows);
        let mut towers = Vec::with_capacity(rows * 6);
        let mut outcome = Vec::with_capacity(self.slots.len());
        let mut decisions = Vec::with_capacity(self.slots.len());
        let mut episode_ticks = Vec::with_capacity(self.slots.len());
        let mut decks = Vec::with_capacity(self.slots.len() * 16);
        let mut queued = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            decks.extend(
                slot.finished_decks
                    .iter()
                    .flatten()
                    .map(|card| (card.index() + 1) as i16),
            );
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
        // Decks of the episode that finished in this step (where `outcome` is
        // set), and whether that episode came from the match queue.
        d.set_item(
            "decks",
            decks.into_pyarray(py).reshape([self.slots.len(), 2, 8])?,
        )?;
        d.set_item("queued", queued.into_pyarray(py))?;
        Ok(d)
    }

    /// Total deck draws the simulator refused since construction.
    fn deck_retries(&self) -> u64 {
        self.slots
            .iter()
            .map(|slot| u64::from(slot.deck_retries))
            .sum()
    }

    /// Deck pairs (Blue, Red card names) of episodes that ended in a simulator panic.
    fn crashes(&self) -> Vec<(Vec<&'static str>, Vec<&'static str>)> {
        let names = |deck: &[CardId; 8]| deck.iter().map(|card| card.name()).collect::<Vec<_>>();
        self.slots
            .iter()
            .flat_map(|slot| slot.crashes.iter())
            .map(|decks| (names(&decks[0]), names(&decks[1])))
            .collect()
    }
}

impl VecEnv {
    fn write_observation<'py>(&self, py: Python<'py>, d: &Bound<'py, PyDict>) -> PyResult<()> {
        use features::*;
        let rows = self.slots.len() * 2;
        let cap = self.max_entities;
        // Entity rows are trimmed to the longest row in this batch, rounded up
        // to a multiple of 16 so the network sees only a few distinct shapes
        // (GPU backends compile kernels per shape).
        let longest = self
            .slots
            .iter()
            .flat_map(|slot| slot.rows.n_entities)
            .max()
            .unwrap_or(0)
            .max(1);
        let width = longest.div_ceil(16).saturating_mul(16).min(cap);

        let mut entities = Vec::with_capacity(rows * width * ENTITY_F);
        let mut entity_extras = Vec::with_capacity(rows * width * ENTITY_X_F);
        let mut entity_mask = Vec::with_capacity(rows * width);
        let mut overflow = Vec::with_capacity(rows);
        let mut hazard_overflow = Vec::with_capacity(rows);
        // Two summaries of the row the trainer would otherwise recompute from
        // the planes every step: how many entity rows are valid, and whether
        // the exported legal mask allows anything besides waiting.
        let mut entity_count = Vec::with_capacity(rows);
        let mut choice = Vec::with_capacity(rows);
        for slot in &self.slots {
            for side in 0..2 {
                entity_count.push(slot.rows.n_entities[side].min(width) as i64);
                choice.push(slot.rows.choice[side]);
                entities.extend_from_slice(
                    &slot.rows.entities[side * cap * ENTITY_F..][..width * ENTITY_F],
                );
                entity_extras.extend_from_slice(
                    &slot.rows.entity_extras[side * cap * ENTITY_X_F..][..width * ENTITY_X_F],
                );
                let count = slot.rows.n_entities[side];
                entity_mask.extend((0..width).map(|index| u8::from(index < count)));
                overflow.push(slot.rows.overflow[side]);
                hazard_overflow.push(slot.rows.hazard_overflow[side]);
            }
        }
        d.set_item(
            "entities",
            entities.into_pyarray(py).reshape([rows, width, ENTITY_F])?,
        )?;
        d.set_item(
            "entity_extras",
            entity_extras
                .into_pyarray(py)
                .reshape([rows, width, ENTITY_X_F])?,
        )?;
        d.set_item("hazard_overflow", hazard_overflow.into_pyarray(py))?;
        d.set_item(
            "entity_mask",
            entity_mask.into_pyarray(py).reshape([rows, width])?,
        )?;
        d.set_item("entity_overflow", overflow.into_pyarray(py))?;
        d.set_item("entity_count", entity_count.into_pyarray(py))?;
        d.set_item("choice", choice.into_pyarray(py))?;

        macro_rules! gather {
            ($name:literal, $field:ident, $shape:expr) => {{
                let per_slot = self.slots[0].rows.$field.len();
                let mut values = Vec::with_capacity(self.slots.len() * per_slot);
                for slot in &self.slots {
                    values.extend_from_slice(&slot.rows.$field);
                }
                d.set_item($name, values.into_pyarray(py).reshape($shape)?)?;
            }};
        }
        macro_rules! mask {
            ($name:literal, $field:ident, $width:expr) => {{
                let mut values = Vec::with_capacity(rows * $width);
                for slot in &self.slots {
                    for count in slot.rows.$field {
                        values.extend((0..$width).map(|index| u8::from(index < count)));
                    }
                }
                d.set_item($name, values.into_pyarray(py).reshape([rows, $width])?)?;
            }};
        }
        gather!("global", global, [rows, GLOBAL_F]);
        gather!("hand", hand, [rows, 4, HAND_F]);
        gather!("abilities", abilities, [rows, ABILITY_ROWS, ABILITY_F]);
        gather!("own_deck", own_deck, [rows, DECK_ROWS, DECK_F]);
        gather!("deck_extras", deck_extras, [rows, DECK_ROWS, DECK_X_F]);
        gather!(
            "reveal_extras",
            reveal_extras,
            [rows, REVEAL_ROWS, REVEAL_X_F]
        );
        gather!("recent_plays", plays, [rows, PLAY_ROWS, PLAY_F]);
        mask!("play_mask", n_plays, PLAY_ROWS);
        gather!("revealed_deck", reveals, [rows, REVEAL_ROWS, REVEAL_F]);
        mask!("reveal_mask", n_reveals, REVEAL_ROWS);
        gather!("hazards", hazards, [rows, HAZARD_ROWS, HAZARD_F]);
        mask!("hazard_mask", n_hazards, HAZARD_ROWS);
        gather!("extras", extras, [rows, EXTRA_F]);
        gather!("legal", legal, [rows, grid::ACTION_COUNT]);
        gather!("priv_global", priv_global, [rows, PRIV_GLOBAL_F]);
        gather!("priv_hand", priv_hand, [rows, 4]);
        gather!("priv_deck", priv_deck, [rows, DECK_ROWS]);
        Ok(())
    }
}

#[pymodule]
fn _env(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<VecEnv>()?;
    module.add("BLUE", Team::Blue.index())?;
    module.add("RED", Team::Red.index())?;
    Ok(())
}
