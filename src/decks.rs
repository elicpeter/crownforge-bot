//! Deck sampling for training episodes.

use crownforge_core::{CardId, Ruleset, rules::CardKind};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Troop,
    Building,
    Spell,
    Other,
}

pub fn classify(kind: &CardKind) -> Class {
    match kind {
        CardKind::Troop { .. } => Class::Troop,
        CardKind::Building { .. } => Class::Building,
        CardKind::Spell(_)
        | CardKind::StrikingSpell(_)
        | CardKind::CentreStrikeSpell(_)
        | CardKind::RollingSpell(_)
        | CardKind::AreaSpell { .. } => Class::Spell,
        CardKind::Variant(_) | CardKind::Mirror(_) => Class::Other,
    }
}

/// SplitMix64. Wrapping arithmetic is intentional.
#[derive(Clone, Copy)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

pub fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Cost, class and ability-unit count of one card.
type Info = (u8, Class, u8);

#[derive(Clone)]
pub struct DeckSampler {
    /// Cards random decks and mutations may draw from.
    cards: Vec<CardId>,
    /// Card facts by `CardId::index()`, for every supported card.
    info: Vec<Option<Info>>,
    pool: Vec<[CardId; 8]>,
    /// Cumulative sampling weights of the pool decks.
    cumulative: Vec<f64>,
    /// Chance that a deck is a pool deck.
    pub pool_prob: f64,
    /// Chance that a deck is a pool deck with one to three cards replaced.
    pub mutant_prob: f64,
    pub mirror: bool,
}

impl DeckSampler {
    pub fn new(rules: &Ruleset, allowed: &[CardId], mirror: bool) -> Self {
        let mut info = vec![None; CardId::CURRENT_ALL.len()];
        for card in CardId::CURRENT_ALL {
            if rules.supports_card(card) {
                let def = rules.card(card);
                info[card.index()] =
                    Some((def.cost, classify(&def.kind), ability_units(rules, card)));
            }
        }
        Self {
            cards: allowed.to_vec(),
            info,
            pool: Vec::new(),
            cumulative: Vec::new(),
            pool_prob: 1.0,
            mutant_prob: 0.0,
            mirror,
        }
    }

    pub fn card_count(&self) -> usize {
        self.cards.len()
    }

    pub fn allows(&self, card: CardId) -> bool {
        self.cards.contains(&card)
    }

    pub fn pool_len(&self) -> usize {
        self.pool.len()
    }

    /// Replace the pool. `weights` are relative sampling weights, uniform when empty.
    pub fn set_pool(&mut self, pool: Vec<[CardId; 8]>, weights: &[f64]) {
        let mut total = 0.0;
        self.cumulative = (0..pool.len())
            .map(|index| {
                total += weights.get(index).copied().unwrap_or(1.0).max(0.0);
                total
            })
            .collect();
        self.pool = pool;
    }

    /// Ability units in a deck. The simulator has two ability slots per side.
    pub fn abilities(&self, deck: &[CardId; 8]) -> u32 {
        deck.iter()
            .filter_map(|card| self.info[card.index()])
            .map(|info| u32::from(info.2))
            .sum()
    }

    /// Keep decks playable: some spells, mostly troops, sane average cost and
    /// one champion, as in the game.
    fn playable(&self, deck: &[CardId; 8]) -> bool {
        let facts = || deck.iter().filter_map(|card| self.info[card.index()]);
        let count = |class: Class| facts().filter(|info| info.1 == class).count();
        let cost: u32 = facts().map(|info| u32::from(info.0)).sum();
        self.abilities(deck) <= 1
            && (1..=3).contains(&count(Class::Spell))
            && count(Class::Building) <= 2
            && count(Class::Troop) >= 4
            && (21..=37).contains(&cost)
    }

    pub fn sample(&self, rng: &mut Rng) -> [[CardId; 8]; 2] {
        let first = self.sample_one(rng);
        let second = if self.mirror {
            first
        } else {
            self.sample_one(rng)
        };
        [first, second]
    }

    fn pool_deck(&self, rng: &mut Rng) -> [CardId; 8] {
        let total = self.cumulative.last().copied().unwrap_or(0.0);
        if total <= 0.0 {
            return self.pool[rng.below(self.pool.len())];
        }
        let target = rng.unit() * total;
        let index = self.cumulative.partition_point(|&sum| sum <= target);
        self.pool[index.min(self.pool.len() - 1)]
    }

    /// A pool deck with k cards replaced, P(k) = 2^-k for k up to 3.
    fn mutant(&self, rng: &mut Rng) -> [CardId; 8] {
        let base = self.pool_deck(rng);
        if self.cards.is_empty() {
            return base;
        }
        let draw = rng.unit();
        let swaps = if draw < 0.5 {
            1
        } else if draw < 0.75 {
            2
        } else {
            3
        };
        for _ in 0..32 {
            let mut deck = base;
            for _ in 0..swaps {
                let card = self.cards[rng.below(self.cards.len())];
                if !deck.contains(&card) {
                    deck[rng.below(8)] = card;
                }
            }
            if deck != base && self.playable(&deck) {
                return deck;
            }
        }
        base
    }

    /// A random deck that passes the playability rules.
    pub fn random(&self, rng: &mut Rng) -> [CardId; 8] {
        let mut order: Vec<usize> = (0..self.cards.len()).collect();
        let mut deck = [CardId::Knight; 8];
        for _ in 0..64 {
            // Partial Fisher-Yates: the first eight entries are a uniform sample.
            for i in 0..8 {
                let j = i + rng.below(order.len() - i);
                order.swap(i, j);
            }
            for (slot, &index) in deck.iter_mut().zip(&order[..8]) {
                *slot = self.cards[index];
            }
            if self.playable(&deck) {
                return deck;
            }
        }
        // No playable draw: fall back to a deck the simulator always accepts.
        CardId::DEFAULT_DECK
    }

    fn sample_one(&self, rng: &mut Rng) -> [CardId; 8] {
        let draw = rng.unit();
        let has_pool = !self.pool.is_empty();
        if has_pool && (self.cards.len() < 8 || draw < self.pool_prob) {
            return self.pool_deck(rng);
        }
        if has_pool && draw < self.pool_prob + self.mutant_prob {
            return self.mutant(rng);
        }
        self.random(rng)
    }
}

/// Units with an ability button that one card deploys.
pub fn ability_units(rules: &Ruleset, card: CardId) -> u8 {
    match &rules.card(card).kind {
        CardKind::Troop { members, .. } => members
            .iter()
            .filter(|member| rules.unit(member.unit).ability.is_some())
            .count() as u8,
        _ => 0,
    }
}
