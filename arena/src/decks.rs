//! Deck sampling for training episodes: a pool deck, a pool deck with one to
//! three cards replaced, or a random playable deck. The same rules as the
//! bot's sampler for its other simulators.

use crate::cards::{CardType, Library};

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

pub type Deck = [usize; 8];

#[derive(Clone)]
pub struct DeckSampler {
    /// Cards random decks and mutations may draw from.
    cards: Vec<usize>,
    /// Cost, type and whether a champion, by library id.
    info: Vec<(i32, CardType, bool)>,
    deployable: Vec<bool>,
    pool: Vec<Deck>,
    cumulative: Vec<f64>,
    /// Decks one side of a battle is drawn from with `focus_prob`, whatever the
    /// other side plays: the decks being prepared for a tournament.
    focus: Vec<Deck>,
    pub focus_prob: f64,
    pub pool_prob: f64,
    pub mutant_prob: f64,
    pub mirror: bool,
    pub fallback: Deck,
}

impl DeckSampler {
    pub fn new(lib: &Library, allowed: &[usize], mirror: bool, fallback: Deck) -> Self {
        let mut deployable = vec![false; lib.defs.len()];
        for &id in &lib.deployable {
            deployable[id] = true;
        }
        Self {
            cards: allowed.to_vec(),
            info: lib.defs.iter().map(|d| (d.elixir, d.ctype, d.is_champion)).collect(),
            deployable,
            pool: Vec::new(),
            cumulative: Vec::new(),
            focus: Vec::new(),
            focus_prob: 0.0,
            pool_prob: 1.0,
            mutant_prob: 0.0,
            mirror,
            fallback,
        }
    }

    pub fn card_count(&self) -> usize {
        self.cards.len()
    }

    pub fn pool_len(&self) -> usize {
        self.pool.len()
    }

    /// The simulator's own deck rules: eight different deck cards, one champion at most.
    pub fn check(&self, deck: &[usize]) -> Result<Deck, String> {
        if deck.len() != 8 {
            return Err("each deck needs exactly eight cards".into());
        }
        let mut cards = [0; 8];
        for (index, &card) in deck.iter().enumerate() {
            if card >= self.deployable.len() || !self.deployable[card] {
                return Err(format!("card id {card} is not a deck card"));
            }
            if deck[..index].contains(&card) {
                return Err(format!("deck lists card id {card} twice"));
            }
            cards[index] = card;
        }
        if cards.iter().filter(|&&card| self.info[card].2).count() > 1 {
            return Err("a deck may hold at most one champion".into());
        }
        Ok(cards)
    }

    pub fn set_pool(&mut self, pool: Vec<Deck>, weights: &[f64]) {
        let mut total = 0.0;
        self.cumulative = (0..pool.len())
            .map(|index| {
                total += weights.get(index).copied().unwrap_or(1.0).max(0.0);
                total
            })
            .collect();
        self.pool = pool;
    }

    pub fn set_focus(&mut self, focus: Vec<Deck>, prob: f64) {
        self.focus = focus;
        self.focus_prob = prob;
    }

    /// Keep decks playable: some spells, mostly troops, a sane total cost.
    fn playable(&self, deck: &Deck) -> bool {
        let count = |kind: CardType| deck.iter().filter(|&&card| self.info[card].1 == kind).count();
        let champions = deck.iter().filter(|&&card| self.info[card].2).count();
        let cost: i32 = deck.iter().map(|&card| self.info[card].0).sum();
        champions <= 1
            && (1..=3).contains(&count(CardType::Spell))
            && count(CardType::Building) <= 2
            && count(CardType::Troop) >= 4
            && (21..=37).contains(&cost)
    }

    pub fn sample(&self, rng: &mut Rng) -> [Deck; 2] {
        let first = self.sample_one(rng);
        let mut decks = [first, if self.mirror { first } else { self.sample_one(rng) }];
        if !self.focus.is_empty() && rng.unit() < self.focus_prob {
            let seat = rng.below(2);
            decks[seat] = self.focus[rng.below(self.focus.len())];
            if self.mirror {
                decks[1 - seat] = decks[seat];
            }
        }
        decks
    }

    fn pool_deck(&self, rng: &mut Rng) -> Deck {
        let total = self.cumulative.last().copied().unwrap_or(0.0);
        if total <= 0.0 {
            return self.pool[rng.below(self.pool.len())];
        }
        let target = rng.unit() * total;
        let index = self.cumulative.partition_point(|&sum| sum <= target);
        self.pool[index.min(self.pool.len() - 1)]
    }

    /// A pool deck with k cards replaced, P(k) = 2^-k for k up to 3.
    fn mutant(&self, rng: &mut Rng) -> Deck {
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
    pub fn random(&self, rng: &mut Rng) -> Deck {
        if self.cards.len() < 8 {
            return self.fallback;
        }
        let mut order: Vec<usize> = (0..self.cards.len()).collect();
        let mut deck = [0; 8];
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
        self.fallback
    }

    fn sample_one(&self, rng: &mut Rng) -> Deck {
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
